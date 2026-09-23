//! The multi-token-prediction head a Qwen3.5 GGUF carries in `blk.{N}.nextn.*`.
//!
//! The same head as the safetensors `mtp.*`, under llama.cpp's names: `eh_proj` is the `[2h, h]`
//! concat projection, `enorm` and `hnorm` normalize the token embedding and the target hidden
//! state, `shared_head_norm` is the final norm, and `blk.{N}`'s own attention and MLP are the
//! decoder block between them. `N` is the transformer depth, since the converter counts the head
//! inside `block_count` and the model excludes it.
//!
//! The head is one [`MtpStep`]; [`Qwen3_5MtpProposer`] chains it into a draft.

use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_quant::QuantMethod;

use crate::attention::AttentionMask;
use crate::gguf::Content;
use crate::kv_cache::KvCache;
use crate::layers::QRmsNorm;
use crate::layers::Qwen3VLRotaryEmbedding;
use crate::models::quantized_qwen3_5_moe::{
    gguf_qmm, DecoderLayer, DenseMlp, GatedFullAttention, LayerImpl, ModelWeights, MoeOrMlp,
    PropsGGUF,
};
use crate::models::qwen3_5_mtp::{
    chain_cache, default_n_predict, MtpOut, MtpStep, Qwen3_5MtpProposer,
};
use crate::speculative::{MtpConfig, SelfSpeculative, SpeculativeProposer};

/// The tensor that says a GGUF carries a head, relative to its block prefix.
pub const NEXTN_EH_PROJ: &str = "nextn.eh_proj.weight";

pub struct Qwen35GgufMtpHead {
    enorm: QRmsNorm,
    hnorm: QRmsNorm,
    eh_proj: Arc<dyn QuantMethod>,
    layer: DecoderLayer,
    norm: QRmsNorm,
    cache: KvCache,
    device: Device,
    dtype: DType,
}

impl Qwen35GgufMtpHead {
    /// Load the head trailing the transformer blocks `props` describes.
    pub fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        props: &PropsGGUF,
        rotary: Arc<Qwen3VLRotaryEmbedding>,
        device: &Device,
        dtype: DType,
        max_draft: usize,
    ) -> Result<Self> {
        if props.nextn_predict_layers == 0 {
            hanzo_ml::bail!(
                "`--mtp-model` requested but this GGUF carries no built-in MTP head (nextn_predict_layers = 0)."
            );
        }
        if props.nextn_predict_layers != 1 {
            hanzo_ml::bail!(
                "Qwen3.5 MTP is one decoder block; this GGUF declares {}",
                props.nextn_predict_layers
            );
        }
        // The head trails the transformer, which already excludes it from `block_count`.
        let prefix = format!("blk.{}", props.block_count);
        let eps = props.rms_norm_eps;

        // The block keeps its own KV, so it takes no paged slot: `None` here is what routes its
        // attention through `cache` instead of the target's paged cache.
        let layer = DecoderLayer {
            layer_impl: LayerImpl::FullAttention(GatedFullAttention::load(
                ct, &prefix, props, rotary, None, device, dtype,
            )?),
            input_layernorm: QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                eps,
            )?,
            post_attention_layernorm: QRmsNorm::new(
                ct.tensor(&format!("{prefix}.post_attention_norm.weight"), device)?,
                eps,
            )?,
            mlp: MoeOrMlp::Mlp(DenseMlp::load(ct, &prefix, device)?),
        };

        Ok(Self {
            enorm: QRmsNorm::new(
                ct.tensor(&format!("{prefix}.nextn.enorm.weight"), device)?,
                eps,
            )?,
            hnorm: QRmsNorm::new(
                ct.tensor(&format!("{prefix}.nextn.hnorm.weight"), device)?,
                eps,
            )?,
            eh_proj: gguf_qmm(ct.tensor(&format!("{prefix}.{NEXTN_EH_PROJ}"), device)?)?,
            layer,
            norm: QRmsNorm::new(
                ct.tensor(&format!("{prefix}.nextn.shared_head_norm.weight"), device)?,
                eps,
            )?,
            cache: chain_cache(max_draft),
            device: device.clone(),
            dtype,
        })
    }

    fn forward(
        &mut self,
        input_embeds: &Tensor,
        target_hidden: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        // The two norms carry the dtype of their own GGUF weights; the projection reads one.
        let embeds = self.enorm.forward(input_embeds)?.to_dtype(self.dtype)?;
        let hidden = self.hnorm.forward(target_hidden)?.to_dtype(self.dtype)?;
        let xs = self
            .eh_proj
            .forward(&Tensor::cat(&[embeds, hidden], D::Minus1)?.contiguous()?)?;
        let cos_sin = self.layer.rotary_cos_sin(positions, xs.dtype())?;
        // Every cached row precedes this query, so the step needs no mask.
        let xs =
            self.layer
                .forward_attention(&xs, &AttentionMask::None, &cos_sin, &mut self.cache)?;
        self.norm.forward(&xs)
    }
}

impl MtpStep for Qwen35GgufMtpHead {
    fn step(
        &mut self,
        input_embeds: &Tensor,
        target_hidden: &Tensor,
        positions: &Tensor,
    ) -> Result<MtpOut> {
        self.forward(input_embeds, target_hidden, positions)
            .map(MtpOut::same)
    }

    fn reset(&mut self) {
        self.cache.reset();
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// A Qwen3.5 GGUF is self-speculative: the MTP head is in the file the model was read from, and
/// shares its token embeddings and output head. This impl is the one place that knowledge lives.
impl SelfSpeculative for ModelWeights {
    fn attach_mtp(&self, cfg: &MtpConfig) -> Result<Box<dyn SpeculativeProposer + Send + Sync>> {
        let path = cfg.resolve_path()?;
        let mut readers = [std::fs::File::open(&path).map_err(|e| {
            hanzo_ml::Error::Msg(format!("failed to open MTP GGUF {}: {e}", path.display()))
        })?];
        let mut readers_ref: Vec<&mut std::fs::File> = readers.iter_mut().collect();
        let mut ct = Content::from_readers(&mut readers_ref)?;
        let n_predict = cfg
            .n_predict
            .unwrap_or(default_n_predict(self.props().embedding_length))
            .max(1);
        let head = Qwen35GgufMtpHead::load(
            &mut ct,
            self.props(),
            self.rotary(),
            &self.device,
            self.compute_dtype(),
            n_predict,
        )?;
        self.set_store_spec(true);
        Ok(Box::new(Qwen3_5MtpProposer::new(
            Box::new(head),
            self.shared_heads(),
            n_predict,
            self.mtp_anchors(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen3_5_mtp::fixtures::{positions, shared_heads, synthetic};
    use crate::models::qwen3_5_mtp::AnchorPositions;
    use crate::speculative::{SpeculativeKvCache, SpeculativeProposeBatchCtx};
    use hanzo_ml::quantized::{gguf_file, GgmlDType, QTensor};
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::sync::Mutex;

    const HIDDEN: usize = 12;
    const HEAD_DIM: usize = 16;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const FFN: usize = 20;
    const VOCAB: usize = 32;
    const DEPTH: usize = 4;

    fn props() -> PropsGGUF {
        PropsGGUF {
            head_count: HEADS,
            head_count_kv: KV_HEADS,
            block_count: DEPTH,
            embedding_length: HIDDEN,
            rms_norm_eps: 1e-6,
            max_seq_len: 64,
            rope_freq_base: 10_000.0,
            head_dim: HEAD_DIM,
            rot_dim: 8,
            mrope_section: vec![2, 1, 1],
            full_attention_interval: 4,
            conv_kernel: 4,
            head_k_dim: 4,
            head_v_dim: 4,
            num_k_heads: 1,
            num_v_heads: 2,
            num_experts: None,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            is_moe: false,
            nextn_predict_layers: 1,
        }
    }

    /// The fifteen tensors a Qwen3.5 GGUF carries at `blk.{DEPTH}`, at tiny sizes and unquantized.
    /// GGUF dims are the transpose of the safetensors ones.
    fn write_head(path: &std::path::Path, device: &Device) -> Result<()> {
        let prefix = format!("blk.{DEPTH}");
        let shapes: &[(&str, usize, usize)] = &[
            ("nextn.eh_proj.weight", HIDDEN, 2 * HIDDEN),
            ("nextn.enorm.weight", HIDDEN, 1),
            ("nextn.hnorm.weight", HIDDEN, 1),
            ("nextn.shared_head_norm.weight", HIDDEN, 1),
            ("attn_norm.weight", HIDDEN, 1),
            ("post_attention_norm.weight", HIDDEN, 1),
            ("attn_q.weight", 2 * HEADS * HEAD_DIM, HIDDEN),
            ("attn_k.weight", KV_HEADS * HEAD_DIM, HIDDEN),
            ("attn_v.weight", KV_HEADS * HEAD_DIM, HIDDEN),
            ("attn_output.weight", HIDDEN, HEADS * HEAD_DIM),
            ("attn_q_norm.weight", HEAD_DIM, 1),
            ("attn_k_norm.weight", HEAD_DIM, 1),
            ("ffn_gate.weight", FFN, HIDDEN),
            ("ffn_up.weight", FFN, HIDDEN),
            ("ffn_down.weight", HIDDEN, FFN),
        ];
        let mut owned = Vec::with_capacity(shapes.len());
        for (seed, (name, rows, cols)) in shapes.iter().enumerate() {
            let t = synthetic(*rows, *cols, seed as u32 + 1, device)?;
            let t = if *cols == 1 { t.flatten_all()? } else { t };
            owned.push((
                format!("{prefix}.{name}"),
                QTensor::quantize(&t, GgmlDType::F32)?,
            ));
        }
        let tensors: Vec<(&str, &QTensor)> = owned
            .iter()
            .map(|(name, t)| (name.as_str(), t))
            .collect::<Vec<_>>();
        let arch = gguf_file::Value::String("qwen35".to_string());
        let mut file = std::fs::File::create(path).map_err(hanzo_ml::Error::msg)?;
        gguf_file::write(&mut file, &[("general.architecture", &arch)], &tensors)?;
        Ok(())
    }

    fn load_head(dir: &std::path::Path, n_predict: usize) -> Result<Qwen35GgufMtpHead> {
        let device = Device::Cpu;
        let path = dir.join("head.gguf");
        write_head(&path, &device)?;
        let mut files = [std::fs::File::open(&path).map_err(hanzo_ml::Error::msg)?];
        let mut readers: Vec<&mut std::fs::File> = files.iter_mut().collect();
        let mut ct = Content::from_readers(&mut readers)?;
        let props = props();
        let rotary = Arc::new(Qwen3VLRotaryEmbedding::new(
            props.rope_freq_base,
            props.rot_dim,
            &device,
            props.mrope_section.clone(),
        )?);
        Qwen35GgufMtpHead::load(&mut ct, &props, rotary, &device, DType::F32, n_predict)
    }

    /// The head maps one row per sequence to one hidden state per sequence, and a chained step
    /// over its own KV keeps that shape.
    #[test]
    fn gguf_mtp_head_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let mut head = load_head(dir.path(), 2)?;

        let batch = 2;
        let embeds = synthetic(batch, HIDDEN, 31, &device)?.reshape((batch, 1, HIDDEN))?;
        let hidden = synthetic(batch, HIDDEN, 32, &device)?.reshape((batch, 1, HIDDEN))?;
        let first = head.step(
            &embeds,
            &hidden,
            &positions(&[[7, 7, 7], [9, 9, 9]], &device)?,
        )?;
        assert_eq!(first.head.dims(), &[batch, 1, HIDDEN]);
        assert!(
            first.head.abs()?.sum_all()?.to_scalar::<f32>()? > 0.0,
            "the head returned an all-zero hidden state"
        );
        // Qwen3.5's head carries the state it hands `lm_head`.
        assert_eq!(
            (&first.carry - &first.head)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?,
            0.0
        );

        let second = head.step(
            &embeds,
            &first.carry,
            &positions(&[[8, 8, 8], [10, 10, 10]], &device)?,
        )?;
        assert_eq!(second.head.dims(), &[batch, 1, HIDDEN]);

        // A reset chain starts over: the same inputs at the same positions give the first step back.
        head.reset();
        let again = head.step(
            &embeds,
            &hidden,
            &positions(&[[7, 7, 7], [9, 9, 9]], &device)?,
        )?;
        let drift = (&again.head - &first.head)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(drift < 1e-5, "reset left {drift} of the old chain behind");
        Ok(())
    }

    /// The GGUF head drafts through the same proposer the safetensors head does.
    #[test]
    fn gguf_mtp_proposal_round_trip() -> Result<()> {
        const N_PREDICT: usize = 3;
        let device = Device::Cpu;
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let anchors: AnchorPositions = Arc::new(Mutex::new(None));
        let mut proposer = Qwen3_5MtpProposer::new(
            Box::new(load_head(dir.path(), N_PREDICT)?),
            shared_heads(VOCAB, HIDDEN, &device)?,
            N_PREDICT,
            anchors.clone(),
        );
        assert_eq!(proposer.proposal_len(), N_PREDICT);

        let batch = 2;
        let hidden = synthetic(batch, HIDDEN, 41, &device)?.reshape((batch, 1, HIDDEN))?;
        let sampled = [5u32, 11];
        let base_lens = [8usize, 12];
        let seq_ids = [0usize, 1];
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0)));
        let propose = |proposer: &mut Qwen3_5MtpProposer, hidden: Option<Tensor>| {
            proposer.propose(
                SpeculativeProposeBatchCtx {
                    sampled_tokens: &sampled,
                    sampled_tokens_emitted: true,
                    seq_ids: &seq_ids,
                    base_lens: &base_lens,
                    sequences: &[],
                    cache: SpeculativeKvCache::Normal,
                    target_hiddens: hidden,
                    target_hidden_layers: None,
                    rng: rng.clone(),
                },
                None,
            )
        };

        *anchors.lock().expect("anchors") = Some(vec![[7, 7, 7], [11, 11, 11]]);
        let batch_out = propose(&mut proposer, Some(hidden.clone()))?;
        assert_eq!(batch_out.proposals.len(), batch);
        for proposal in &batch_out.proposals {
            assert_eq!(proposal.tokens.len(), N_PREDICT);
            assert!(proposal.tokens.iter().all(|t| (*t as usize) < VOCAB));
            let logits = proposal.logits.as_ref().expect("per-draft logits");
            assert_eq!(logits.dims(), &[1, N_PREDICT, VOCAB]);
        }

        // Each round consumes its anchor positions, so a stale draft is impossible.
        assert!(anchors.lock().expect("anchors").is_none());
        assert!(propose(&mut proposer, Some(hidden)).is_err());
        Ok(())
    }
}
