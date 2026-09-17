//! The multi-token-prediction head Qwen3.5 / Qwen3.8 checkpoints carry in `mtp.*`.
//!
//! One full-attention decoder block with the main stack's geometry, fed
//! `fc(concat(pre_fc_norm_embedding(embed(token)), pre_fc_norm_hidden(hidden)))` where `hidden` is
//! the target's final-norm hidden state at that position. It shares the target's token embeddings
//! and `lm_head`, so a draft token comes out on the verifier's own scale.
//!
//! A row fed token `t + 1` over the hidden at `t` predicts token `t + 2`. Chaining that — the
//! head's own hidden state and its argmax at the next position — drafts `n_predict` tokens per
//! target step. The chain attends over its own KV, which resets each round: the target hidden
//! already carries the context, and the target verify decides every emitted token, so the KV
//! only reaches draft acceptance, never correctness.

use std::{
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use hanzo_ml::{DType, Device, Module, Result, Tensor, D};
use hanzo_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use crate::{
    attention::AttentionMask,
    device_map::DeviceMapper,
    kv_cache::KvCache,
    layers::{GemmaRmsNorm, Qwen3VLRotaryEmbedding},
    pipeline::text_models_inputs_processor::FlashParams,
    speculative::{
        MtpConfig, SelfSpeculative, SpeculativeProposal, SpeculativeProposalBatch,
        SpeculativeProposeBatchCtx, SpeculativeProposer, SpeculativeSharedHeads,
        TargetTokenEmbedder,
    },
    utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor},
};

use super::{config::TextConfig, text::DecoderLayer, Qwen3_5Model};

/// The tensor that says a checkpoint carries a head.
pub const MTP_FC_WEIGHT: &str = "mtp.fc.weight";

/// Drafts per target step. A wider model amortizes the verify forward over more drafts, so it
/// drafts deeper; past that, acceptance falls faster than the saved forwards pay for.
const DEFAULT_N_PREDICT: usize = 2;
const DEEP_N_PREDICT: usize = 3;
const DEEP_HIDDEN_SIZE: usize = 4096;

/// The three MRoPE planes of one position.
type Mrope = [u32; 3];

/// The MRoPE positions of the target rows the next draft starts from, one per sequence. The
/// model fills it when it selects those rows; the proposer takes it when it drafts.
pub(super) type AnchorPositions = Arc<Mutex<Option<Vec<Mrope>>>>;

/// The head's KV, holding one row per chained draft and nothing else. It is the head's own, so
/// the head takes no slot in the target's paged cache and the target's cache stays sized by the
/// model's own layers; a paged head would instead take the slot after them, and this is the only
/// place that decides.
fn chain_cache(max_draft: usize) -> KvCache {
    KvCache::new_normal(2, max_draft, max_draft)
}

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

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Drop the chain's KV, so the next chain attends only to itself.
    pub fn reset(&mut self) {
        self.cache.reset();
    }

    /// One drafter step over `[batch, 1, hidden]` inputs at `[3, batch, 1]` MRoPE positions.
    /// Returns the normed hidden state, which is both the `lm_head` input and the next step's.
    pub fn forward(
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

/// Drives the head as the target's own speculative draft.
pub struct Qwen3_5MtpProposer {
    head: Qwen3_5MtpHead,
    heads: SpeculativeSharedHeads,
    n_predict: usize,
    anchors: AnchorPositions,
}

impl Qwen3_5MtpProposer {
    pub(super) fn new(
        head: Qwen3_5MtpHead,
        heads: SpeculativeSharedHeads,
        n_predict: usize,
        anchors: AnchorPositions,
    ) -> Self {
        Self {
            head,
            heads,
            n_predict,
            anchors,
        }
    }
}

impl SpeculativeProposer for Qwen3_5MtpProposer {
    fn proposal_len(&self) -> usize {
        self.n_predict
    }

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        _target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch> {
        let batch = ctx.sampled_tokens.len();
        if batch == 0 {
            return Ok(SpeculativeProposalBatch::new(Vec::new()));
        }
        let hidden = ctx.target_hiddens.ok_or_else(|| {
            hanzo_ml::Error::msg("Qwen3.5 MTP needs the target hidden state to draft")
        })?;
        let mut mrope = self
            .anchors
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
            .ok_or_else(|| {
                hanzo_ml::Error::msg("Qwen3.5 MTP needs the target row positions to draft")
            })?;
        if mrope.len() != batch || hidden.dim(0)? != batch {
            hanzo_ml::bail!(
                "Qwen3.5 MTP batch mismatch: {batch} anchors sampled, {} positions, {} hidden rows",
                mrope.len(),
                hidden.dim(0)?
            );
        }

        let device = self.head.device().clone();
        let mut hidden = match hidden.dims() {
            [_, _, _] => hidden.to_device(&device)?.to_dtype(self.head.dtype())?,
            [_, _] => hidden
                .unsqueeze(1)?
                .to_device(&device)?
                .to_dtype(self.head.dtype())?,
            other => hanzo_ml::bail!("Qwen3.5 MTP target hidden has shape {other:?}"),
        };
        let mut tokens = ctx.sampled_tokens.to_vec();
        self.head.reset();

        let mut drafts: Vec<Vec<u32>> = Vec::with_capacity(self.n_predict);
        let mut logits: Vec<Tensor> = Vec::with_capacity(self.n_predict);
        for step in 0..self.n_predict {
            let ids = Tensor::from_vec(tokens.clone(), (batch, 1), &device)?;
            let embeds = (self.heads.embed)(&ids)?.to_dtype(self.head.dtype())?;
            let mut planes = Vec::with_capacity(3 * batch);
            for plane in 0..3 {
                planes.extend(mrope.iter().map(|pos| pos[plane]));
            }
            let positions = Tensor::from_vec(planes, (3, batch, 1), &device)?;
            let normed = self.head.forward(&embeds, &hidden, &positions)?;
            let step_logits = (self.heads.lm_head)(&normed)?;
            let step_drafts: Vec<u32> = step_logits
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .flatten_all()?
                .to_vec1()?;
            if step_drafts.len() != batch {
                hanzo_ml::bail!(
                    "Qwen3.5 MTP drafted {} tokens for {batch} sequences",
                    step_drafts.len()
                );
            }
            logits.push(step_logits);
            if step + 1 < self.n_predict {
                // The draft becomes the next step's input, one position on, off the head's own
                // hidden state.
                tokens.clone_from(&step_drafts);
                hidden = normed;
                for pos in mrope.iter_mut() {
                    for plane in pos.iter_mut() {
                        *plane += 1;
                    }
                }
            }
            drafts.push(step_drafts);
        }

        // Per sequence: its drafted tokens and their logits as `[1, n_predict, vocab]`, the rows
        // the verifier indexes by draft position.
        let mut proposals = Vec::with_capacity(batch);
        for row in 0..batch {
            let row_logits = logits
                .iter()
                .map(|step| step.narrow(0, row, 1))
                .collect::<Result<Vec<_>>>()?;
            proposals.push(SpeculativeProposal::with_logits(
                drafts.iter().map(|step| step[row]).collect(),
                Tensor::cat(&row_logits, 1)?,
            ));
        }
        Ok(SpeculativeProposalBatch::new(proposals))
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
            .unwrap_or(if self.text_config.hidden_size >= DEEP_HIDDEN_SIZE {
                DEEP_N_PREDICT
            } else {
                DEFAULT_N_PREDICT
            })
            .max(1);
        let head = Qwen3_5MtpHead::load(
            vb,
            &self.text_config,
            self.text.mapper(),
            &self.text.device,
            n_predict,
        )?;
        Ok(Box::new(Qwen3_5MtpProposer::new(
            head,
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
    use crate::speculative::SpeculativeKvCache;
    use rand::SeedableRng;
    use rand_isaac::Isaac64Rng;
    use std::collections::HashMap;

    const HIDDEN: usize = 12;
    const HEAD_DIM: usize = 16;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const FFN: usize = 20;
    const VOCAB: usize = 32;

    /// Small, distinct, reproducible weights: enough spread that a zero output means a broken
    /// forward rather than a symmetric one.
    fn synthetic(rows: usize, cols: usize, seed: u32, device: &Device) -> Result<Tensor> {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
                ((x >> 8) % 1000) as f32 / 4000.0 - 0.125
            })
            .collect();
        Tensor::from_vec(data, (rows, cols), device)
    }

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

    /// Token embeddings and an output head, as the target lends them.
    fn shared_heads(device: &Device) -> Result<SpeculativeSharedHeads> {
        let embed = synthetic(VOCAB, HIDDEN, 21, device)?;
        let out = synthetic(VOCAB, HIDDEN, 22, device)?;
        Ok(SpeculativeSharedHeads {
            embed: Arc::new(move |ids: &Tensor| {
                let (batch, seq) = ids.dims2()?;
                embed
                    .index_select(&ids.flatten_all()?, 0)?
                    .reshape((batch, seq, HIDDEN))
            }),
            lm_head: Arc::new(move |hidden: &Tensor| hidden.broadcast_matmul(&out.t()?)),
        })
    }

    fn positions(anchors: &[Mrope], device: &Device) -> Result<Tensor> {
        let mut planes = Vec::with_capacity(3 * anchors.len());
        for plane in 0..3 {
            planes.extend(anchors.iter().map(|pos| pos[plane]));
        }
        Tensor::from_vec(planes, (3, anchors.len(), 1), device)
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
        head.reset();
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
            load_head(dir.path(), N_PREDICT)?,
            shared_heads(&device)?,
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
