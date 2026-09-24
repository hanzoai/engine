//! The multi-token-prediction head Qwen3.5 / Qwen3.8 checkpoints carry in `mtp.*`.
//!
//! One full-attention decoder block with the main stack's geometry, fed
//! `fc(concat(pre_fc_norm_embedding(embed(token)), pre_fc_norm_hidden(hidden)))` where `hidden` is
//! the target's final-norm hidden state at that position. It shares the target's token embeddings
//! and `lm_head`.
//!
//! The head is one [`MtpStep`]; [`Qwen3_5MtpProposer`] chains it into a draft.

use std::{fs, path::PathBuf, sync::Arc};

use hanzo_ml::{DType, Device, Module, Result, Tensor, D};
use hanzo_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use crate::{
    attention::AttentionMask,
    device_map::DeviceMapper,
    kv_cache::KvCache,
    layers::{GemmaRmsNorm, Qwen3VLRotaryEmbedding},
    models::qwen3_5_mtp::{chain_cache, default_n_predict, MtpOut, MtpStep, Qwen3_5MtpProposer},
    pipeline::text_models_inputs_processor::FlashParams,
    speculative::{MtpConfig, SelfSpeculative, SpeculativeProposer},
    utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor},
};

use super::{config::TextConfig, text::DecoderLayer, Qwen3_5Model};

/// The tensor that says a checkpoint carries a head.
pub const MTP_FC_WEIGHT: &str = "mtp.fc.weight";

pub struct Qwen3_5MtpHead {
    pre_fc_norm_embedding: GemmaRmsNorm,
    pre_fc_norm_hidden: GemmaRmsNorm,
    fc: Arc<dyn QuantMethod>,
    layer: DecoderLayer,
    norm: GemmaRmsNorm,
    cache: KvCache,
    device: Device,
    dtype: DType,
}

impl Qwen3_5MtpHead {
    /// `vb` is the checkpoint root. The head lives on the non-mapped device beside the final norm
    /// and `lm_head` it feeds, and holds KV for `max_draft` chained rows.
    pub fn load(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        mapper: &dyn DeviceMapper,
        device: &Device,
        max_draft: usize,
    ) -> Result<Self> {
        if !vb.contains_tensor(MTP_FC_WEIGHT) {
            hanzo_ml::bail!(
                "`--mtp` requested but this checkpoint carries no built-in MTP head (`{MTP_FC_WEIGHT}`)."
            );
        }
        if cfg.mtp_num_hidden_layers != 1 {
            hanzo_ml::bail!(
                "Qwen3.5 MTP is one decoder layer; this config declares {}",
                cfg.mtp_num_hidden_layers
            );
        }
        if cfg.mtp_use_dedicated_embeddings {
            hanzo_ml::bail!("Qwen3.5 MTP with dedicated embeddings is not supported");
        }
        // The checkpoint leaves `mtp.*` unquantized however the main stack is quantized, so the
        // head loads plain at the checkpoint dtype.
        let mut cfg = cfg.clone();
        cfg.quantization_config = None;

        let vb = mapper.set_nm_device(vb.pp("mtp"), false);
        // The head sits beside the main stack and shards the way it does.
        let comm = mapper.get_comm_for(0)?;
        let rotary_emb = Arc::new(Qwen3VLRotaryEmbedding::new(
            cfg.rope_theta() as f32,
            cfg.rot_dim(),
            device,
            cfg.mrope_section().to_vec(),
        )?);
        let layer = DecoderLayer::load_full_attention(
            vb.pp("layers").pp(0),
            vb.pp("layers").pp(0),
            &cfg,
            rotary_emb,
            None,
            &comm,
        )?;

        Ok(Self {
            pre_fc_norm_embedding: GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_fc_norm_embedding"),
            )?,
            pre_fc_norm_hidden: GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_fc_norm_hidden"),
            )?,
            fc: ReplicatedLayer::new(
                2 * cfg.hidden_size,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                vb.pp("fc"),
            )?,
            layer,
            norm: GemmaRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
            cache: chain_cache(max_draft),
            device: device.clone(),
            dtype: vb.dtype(),
        })
    }

    fn forward(
        &mut self,
        input_embeds: &Tensor,
        target_hidden: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let embeds = self.pre_fc_norm_embedding.forward(input_embeds)?;
        let hidden = self.pre_fc_norm_hidden.forward(target_hidden)?;
        let xs = self
            .fc
            .forward(&Tensor::cat(&[embeds, hidden], D::Minus1)?.contiguous()?)?;
        let cos_sin = self
            .layer
            .rotary_emb()
            .ok_or_else(|| hanzo_ml::Error::msg("Qwen3.5 MTP layer is not full attention"))?
            .compute_cos_sin(positions, xs.dtype())?;
        // Every cached row precedes this query, so the step needs no mask.
        let xs = self.layer.forward_attention(
            &xs,
            &AttentionMask::None,
            &cos_sin,
            &mut self.cache,
            None,
            &FlashParams::empty(false),
        )?;
        self.norm.forward(&xs)
    }
}

impl MtpStep for Qwen3_5MtpHead {
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

/// Qwen3.5 is self-speculative: the MTP head ships inside the checkpoint `cfg` names, sharing the
/// target's token embeddings and output head. This impl is the one place that knowledge lives.
impl SelfSpeculative for Qwen3_5Model {
    fn attach_mtp(&self, cfg: &MtpConfig) -> Result<Box<dyn SpeculativeProposer + Send + Sync>> {
        let path = cfg.resolve_path()?;
        let mut weights = fs::read_dir(&path)
            .map_err(|e| {
                hanzo_ml::Error::Msg(format!(
                    "failed to list MTP checkpoint {}: {e}",
                    path.display()
                ))
            })?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<PathBuf>>();
        weights.sort();
        if weights.is_empty() {
            hanzo_ml::bail!(
                "MTP checkpoint {} has no safetensors weights",
                path.display()
            );
        }

        let vb = from_mmaped_safetensors(
            weights,
            Vec::new(),
            Some(self.text.dtype),
            &self.text.device,
            Vec::new(),
            true,
            None,
            |name: String| name.starts_with("mtp."),
            Arc::new(|_| DeviceForLoadTensor::Base),
        )?;
        let n_predict = cfg
            .n_predict
            .unwrap_or(default_n_predict(self.text_config.hidden_size))
            .max(1);
        let head = Qwen3_5MtpHead::load(
            vb,
            &self.text_config,
            self.text.mapper(),
            &self.text.device,
            n_predict,
        )?;
        // The head reads the final-norm hidden state of every forward from here on.
        self.text.set_store_spec(true);
        Ok(Box::new(Qwen3_5MtpProposer::new(
            Box::new(head),
            self.text.shared_heads(),
            n_predict,
            self.mtp_anchors.clone(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_map::DummyDeviceMapper;
    use crate::models::qwen3_5_mtp::fixtures::{positions, shared_heads, synthetic};
    use crate::models::qwen3_5_mtp::AnchorPositions;
    use crate::sequence::test_sequence;
    use crate::speculative::{SpeculativeKvCache, SpeculativeProposeBatchCtx};
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::collections::HashMap;
    use std::sync::Mutex;

    const HIDDEN: usize = 12;
    const HEAD_DIM: usize = 16;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const FFN: usize = 20;
    const VOCAB: usize = 32;

    fn tiny_config() -> TextConfig {
        serde_json::from_value(serde_json::json!({
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "intermediate_size": FFN,
            "num_hidden_layers": 4,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "hidden_act": "silu",
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "mrope_section": [2, 1, 1],
                "partial_rotary_factor": 0.5
            },
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "mtp_num_hidden_layers": 1
        }))
        .expect("tiny Qwen3.5 text config")
    }

    /// The fifteen `mtp.*` tensors a Qwen3.5 checkpoint carries, at tiny sizes.
    fn write_head(dir: &std::path::Path, device: &Device) -> Result<PathBuf> {
        let mut st: HashMap<String, Tensor> = HashMap::new();
        let mut put = |name: &str, rows: usize, cols: usize, seed: u32| -> Result<()> {
            st.insert(name.to_string(), synthetic(rows, cols, seed, device)?);
            Ok(())
        };
        put("mtp.fc.weight", HIDDEN, 2 * HIDDEN, 1)?;
        put("mtp.pre_fc_norm_embedding.weight", HIDDEN, 1, 2)?;
        put("mtp.pre_fc_norm_hidden.weight", HIDDEN, 1, 3)?;
        put("mtp.norm.weight", HIDDEN, 1, 4)?;
        put("mtp.layers.0.input_layernorm.weight", HIDDEN, 1, 5)?;
        put("mtp.layers.0.post_attention_layernorm.weight", HIDDEN, 1, 6)?;
        put(
            "mtp.layers.0.self_attn.q_proj.weight",
            2 * HEADS * HEAD_DIM,
            HIDDEN,
            7,
        )?;
        put(
            "mtp.layers.0.self_attn.k_proj.weight",
            KV_HEADS * HEAD_DIM,
            HIDDEN,
            8,
        )?;
        put(
            "mtp.layers.0.self_attn.v_proj.weight",
            KV_HEADS * HEAD_DIM,
            HIDDEN,
            9,
        )?;
        put(
            "mtp.layers.0.self_attn.o_proj.weight",
            HIDDEN,
            HEADS * HEAD_DIM,
            10,
        )?;
        put("mtp.layers.0.self_attn.q_norm.weight", HEAD_DIM, 1, 11)?;
        put("mtp.layers.0.self_attn.k_norm.weight", HEAD_DIM, 1, 12)?;
        put("mtp.layers.0.mlp.gate_proj.weight", FFN, HIDDEN, 13)?;
        put("mtp.layers.0.mlp.up_proj.weight", FFN, HIDDEN, 14)?;
        put("mtp.layers.0.mlp.down_proj.weight", HIDDEN, FFN, 15)?;
        // The norms are one-dimensional in a checkpoint.
        for name in [
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.norm.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
        ] {
            let flat = st[name].flatten_all()?;
            st.insert(name.to_string(), flat);
        }
        let path = dir.join("model.safetensors");
        hanzo_ml::safetensors::save(&st, &path)?;
        Ok(path)
    }

    fn load_head(dir: &std::path::Path, n_predict: usize) -> Result<Qwen3_5MtpHead> {
        let device = Device::Cpu;
        let path = write_head(dir, &device)?;
        let vb = from_mmaped_safetensors(
            vec![path],
            Vec::new(),
            Some(DType::F32),
            &device,
            Vec::new(),
            true,
            None,
            |name: String| name.starts_with("mtp."),
            Arc::new(|_| DeviceForLoadTensor::Base),
        )?;
        let mapper = DummyDeviceMapper {
            nm_device: device.clone(),
        };
        Qwen3_5MtpHead::load(vb, &tiny_config(), &mapper, &device, n_predict)
    }

    /// The head maps one row per sequence to one hidden state per sequence, and a chained step
    /// over its own KV keeps that shape.
    #[test]
    fn mtp_head_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let mut head = load_head(dir.path(), 2)?;

        let batch = 2;
        let embeds = synthetic(batch, HIDDEN, 31, &device)?.reshape((batch, 1, HIDDEN))?;
        let hidden = synthetic(batch, HIDDEN, 32, &device)?.reshape((batch, 1, HIDDEN))?;
        let first = head.forward(
            &embeds,
            &hidden,
            &positions(&[[7, 7, 7], [9, 9, 9]], &device)?,
        )?;
        assert_eq!(first.dims(), &[batch, 1, HIDDEN]);
        assert!(
            first.abs()?.sum_all()?.to_scalar::<f32>()? > 0.0,
            "the head returned an all-zero hidden state"
        );

        let second = head.forward(
            &embeds,
            &first,
            &positions(&[[8, 8, 8], [10, 10, 10]], &device)?,
        )?;
        assert_eq!(second.dims(), &[batch, 1, HIDDEN]);

        // A reset chain starts over: the same inputs at the same positions give the first step back.
        MtpStep::reset(&mut head);
        let again = head.forward(
            &embeds,
            &hidden,
            &positions(&[[7, 7, 7], [9, 9, 9]], &device)?,
        )?;
        let drift = (&again - &first)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(drift < 1e-5, "reset left {drift} of the old chain behind");
        Ok(())
    }

    /// One proposal round through the proposer trait: every sequence gets `n_predict` drafts and
    /// the per-draft logit rows the verifier indexes.
    #[test]
    fn mtp_proposal_round_trip() -> Result<()> {
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
        let (first, second) = (test_sequence(vec![1], None), test_sequence(vec![2], None));
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
                    sequences: &[&first, &second],
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
