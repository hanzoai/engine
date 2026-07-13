#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GLM-5 (`GlmMoeDsaForCausalLM`, `model_type = "glm_moe_dsa"`).
//!
//! Structurally this is DeepSeek-V3.2: MLA (q-LoRA + kv-LoRA compression) + a
//! fine-grained sigmoid `noaux_tc` MoE + the shared DSA lightning indexer
//! ([`super::dsa`]). Two things are GLM-specific:
//!
//! 1. Interleaved RoPE on the rope slice (`rope_interleave = true`). The pair
//!    `(x_{2i}, x_{2i+1})` is rotated by frequency `i` for both the main
//!    attention and the indexer. This is the same rotation the engine's
//!    [`DeepSeekV2RotaryEmbedding`] applies (its `is_gpt_neox = false` path), so
//!    the QK dot is identical to HF's `apply_rotary_pos_emb_interleave` and the
//!    same rotary primitive is reused unchanged.
//!
//! 2. IndexShare: `indexer_types[layer]` is `"full"` or `"shared"`. A `"full"`
//!    layer runs its own indexer and computes a fresh top-k selection; a
//!    `"shared"` layer carries no indexer weights and reuses the most recent
//!    full layer's selection. The selection is a [`DsaSelection`] value threaded
//!    explicitly through the decoder loop, never hidden state.
//!
//! As with [`super::deepseek3`], the DSA selection is applied on the eager
//! (`--no-paged-attn`) cold-cache prefill path where the `[Lq, Lk]` selection
//! aligns with the causal mask; warm-cache / paged / MLA-decode fall back to the
//! dense path (byte-identical), pending the FP8 lightning-indexer kernel.

use std::{collections::HashMap, sync::Arc};

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::Deserialize;

use super::dsa::{DsaConfig, DsaIndexer, DsaSelection};
use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        embedding, Activation, CausalMasker, DeepSeekV2RopeConfig, DeepSeekV2RotaryEmbedding, Mlp,
        RmsNorm, Sdpa,
    },
    layers_masker::CausalMaskConfig,
    mla::{
        mla_cache_forward, mla_decode_forward, should_use_mla_cache, should_use_mla_decode,
        MlaWeights,
    },
    moe::{MoEExperts, MoEExpertsConfig},
    ops::{SplitOp, TopKLastDimOp},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalLoadingMetadata,
        NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

/// GLM-5's indexer key norm is a real `LayerNorm` with a hardcoded `eps = 1e-6`
/// (HF `nn.LayerNorm(index_head_dim, eps=1e-6)`), independent of `rms_norm_eps`
/// (which is `1e-5` for GLM-5.2). Passed to [`DsaIndexer::load`] as the norm eps.
const INDEXER_KNORM_EPS: f64 = 1e-6;

/// The MLA q/kv compression norms (`q_a_layernorm`, `kv_a_layernorm`) are built
/// as bare `GlmMoeDsaRMSNorm(dim)` in HF, i.e. with the RMSNorm default `eps`,
/// NOT the model-wide `rms_norm_eps`. Only the block/final norms take
/// `rms_norm_eps`.
const MLA_LORA_NORM_EPS: f64 = 1e-6;

/// noaux_tc group mask: masked-out experts must be UNSELECTABLE by the top-k, matching HF
/// `masked_fill(~score_mask, -inf)`. A 0/1 multiply is wrong when an in-group score is negative
/// (a zeroed masked expert would then outrank it). Finite so `mask * bias` can never NaN.
const MASKED_EXPERT_BIAS: f64 = 1e30;

serde_default_fn!(f64, routed_scaling_factor, 1.0);
serde_default_fn!(usize, moe_layer_freq, 1);
serde_default_fn!(usize, first_k_dense_replace, 0);
serde_default_fn!(usize, n_group, 1);
serde_default_fn!(usize, topk_group, 1);
serde_default_fn!(bool, norm_topk_prob, true);
serde_default_fn!(usize, index_topk_freq, 1);
serde_default_fn!(usize, index_skip_topk_offset, 2);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
serde_default_fn!(bool, tie_word_embeddings, false);

#[derive(Deserialize, Clone, Debug)]
pub struct RopeParameters {
    pub(crate) rope_theta: f32,
    #[serde(default)]
    pub(crate) rope_type: String,
}

/// Per-layer indexer mode. `Full` runs the lightning indexer and computes a
/// fresh top-k; `Shared` reuses the previous `Full` layer's selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IndexerType {
    Full,
    Shared,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Glm5MoeConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) n_shared_experts: usize,
    pub(crate) n_routed_experts: usize,
    #[serde(default = "routed_scaling_factor")]
    pub(crate) routed_scaling_factor: f64,
    pub(crate) num_experts_per_tok: usize,
    #[serde(default = "moe_layer_freq")]
    pub(crate) moe_layer_freq: usize,
    #[serde(default = "first_k_dense_replace")]
    pub(crate) first_k_dense_replace: usize,
    #[serde(default = "n_group")]
    pub(crate) n_group: usize,
    #[serde(default = "topk_group")]
    pub(crate) topk_group: usize,
    #[serde(default = "norm_topk_prob")]
    pub(crate) norm_topk_prob: bool,
    #[serde(default = "hidden_act")]
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) rope_parameters: RopeParameters,
    #[serde(default)]
    pub(crate) attention_bias: bool,
    pub(crate) q_lora_rank: usize,
    pub(crate) qk_rope_head_dim: usize,
    pub(crate) kv_lora_rank: usize,
    pub(crate) v_head_dim: usize,
    pub(crate) qk_nope_head_dim: usize,
    #[serde(alias = "quantization")]
    pub(crate) quantization_config: Option<QuantizedConfig>,
    pub(crate) index_n_heads: usize,
    pub(crate) index_head_dim: usize,
    pub(crate) index_topk: usize,
    #[serde(default = "index_topk_freq")]
    pub(crate) index_topk_freq: usize,
    #[serde(default = "index_skip_topk_offset")]
    pub(crate) index_skip_topk_offset: usize,
    #[serde(default)]
    pub(crate) index_topk_pattern: Option<String>,
    #[serde(default)]
    pub(crate) indexer_types: Option<Vec<String>>,
    #[serde(default)]
    pub(crate) mlp_layer_types: Option<Vec<String>>,
}

impl Glm5MoeConfig {
    /// `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` (256 for GLM-5.2).
    pub(crate) fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    fn softmax_scale(&self) -> f32 {
        1.0 / (self.q_head_dim() as f32).sqrt()
    }

    /// The DSA lightning-indexer config, or `None` when DSA is off (`DSA=0`) or
    /// the checkpoint's indexer dims are degenerate. Routed through
    /// [`DsaConfig::new`] so GLM-5 shares the one `has_dsa` gate + env policy with
    /// every other arch.
    pub(crate) fn dsa(&self) -> Option<DsaConfig> {
        DsaConfig::new(
            self.index_n_heads,
            self.index_head_dim,
            self.index_topk,
            self.qk_rope_head_dim,
        )
    }

    /// Layer uses the MoE block (vs a dense MLP). Explicit `mlp_layer_types`
    /// wins; otherwise the leading `first_k_dense_replace` layers are dense.
    pub(crate) fn is_moe_layer(&self, layer_idx: usize) -> bool {
        match &self.mlp_layer_types {
            Some(types) => types.get(layer_idx).map(String::as_str) == Some("sparse"),
            None => {
                layer_idx >= self.first_k_dense_replace
                    && layer_idx.is_multiple_of(self.moe_layer_freq)
            }
        }
    }

    /// Per-layer indexer schedule. Explicit `indexer_types` wins; otherwise it is
    /// derived from `index_topk_pattern` or the `index_topk_freq` /
    /// `index_skip_topk_offset` cadence, matching HF `__post_init__`.
    pub(crate) fn indexer_schedule(&self) -> Vec<IndexerType> {
        if let Some(types) = &self.indexer_types {
            return types
                .iter()
                .map(|t| {
                    if t == "shared" {
                        IndexerType::Shared
                    } else {
                        IndexerType::Full
                    }
                })
                .collect();
        }
        if let Some(pattern) = &self.index_topk_pattern {
            return pattern
                .chars()
                .map(|c| {
                    if c == 'S' {
                        IndexerType::Shared
                    } else {
                        IndexerType::Full
                    }
                })
                .collect();
        }
        let freq = self.index_topk_freq.max(1);
        let offset = self.index_skip_topk_offset;
        (0..self.num_hidden_layers)
            .map(|i| {
                if (i + 1).saturating_sub(offset) % freq == 0 {
                    IndexerType::Full
                } else {
                    IndexerType::Shared
                }
            })
            .collect()
    }
}

/// GLM-5 always compresses the query through a LoRA (`q_lora_rank` is a required
/// field): `q_b_proj(q_a_layernorm(q_a_proj(x)))`.
struct QProj {
    a: Arc<dyn QuantMethod>,
    norm: RmsNorm,
    b: Arc<dyn QuantMethod>,
}

impl QProj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.b.forward(&self.norm.forward(&self.a.forward(xs)?)?)
    }

    /// The latent the indexer projects its query from: the q-LoRA-normalised
    /// `q_a_layernorm(q_a_proj(x))`.
    fn indexer_q_src(&self, xs: &Tensor) -> Result<Tensor> {
        self.norm.forward(&self.a.forward(xs)?)
    }
}

struct Attention {
    q: QProj,
    kv_a_proj_with_mqa: Arc<dyn QuantMethod>,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
    cfg: Glm5MoeConfig,
    q_head_dim: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    num_attention_heads: usize,
    mla_weights: MlaWeights,
    /// Present only on `Full` indexer layers; `None` on `Shared` layers, which
    /// reuse the threaded-in selection.
    indexer: Option<DsaIndexer>,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &Glm5MoeConfig,
        indexer_type: IndexerType,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<hanzo_quant::Comm>,
    ) -> Result<Self> {
        let q_head_dim = cfg.q_head_dim();
        let a = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.q_lora_rank,
            &cfg.quantization_config,
            cfg.attention_bias,
            mapper.set_device(layer_idx, vb.pp("q_a_proj"), loading_isq),
        )?;
        let norm = RmsNorm::new(
            cfg.q_lora_rank,
            MLA_LORA_NORM_EPS,
            mapper.set_device(layer_idx, vb.pp("q_a_layernorm"), false),
        )?;
        let b = ColumnParallelLayer::new(
            cfg.q_lora_rank,
            cfg.num_attention_heads * q_head_dim,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_b_proj"), loading_isq),
        )?;
        let q = QProj { a, norm, b };

        let kv_a_proj_with_mqa = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            mapper.set_device(layer_idx, vb.pp("kv_a_proj_with_mqa"), loading_isq),
        )?;
        let kv_a_layernorm = RmsNorm::new(
            cfg.kv_lora_rank,
            MLA_LORA_NORM_EPS,
            mapper.set_device(layer_idx, vb.pp("kv_a_layernorm"), false),
        )?;
        let kv_b_proj = ColumnParallelLayer::new(
            cfg.kv_lora_rank,
            cfg.num_attention_heads * (q_head_dim - cfg.qk_rope_head_dim + cfg.v_head_dim),
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("kv_b_proj"), loading_isq),
        )?;

        let o_proj = RowParallelLayer::new(
            cfg.num_attention_heads * cfg.v_head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;

        let mla_weights = MlaWeights::new(
            paged_attn.is_some(),
            mapper.device_for(layer_idx, loading_isq),
        );

        let indexer = match (indexer_type, cfg.dsa()) {
            (IndexerType::Full, Some(dsa_cfg)) => DsaIndexer::load(
                dsa_cfg,
                cfg.q_lora_rank,
                cfg.hidden_size,
                INDEXER_KNORM_EPS,
                &cfg.quantization_config,
                mapper.set_device(layer_idx, vb.pp("indexer"), loading_isq),
            )?,
            (IndexerType::Full, None) | (IndexerType::Shared, _) => None,
        };

        Ok(Self {
            q,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            cfg: cfg.clone(),
            q_head_dim,
            paged_attn,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            sdpa_params: SdpaParams {
                n_kv_groups: 1,
                softcap: None,
                softmax_scale: cfg.softmax_scale(),
                sliding_window: None,
                sinks: None,
            },
            mla_weights,
            indexer,
        })
    }

    /// Returns `(attn_out, selection)`. `selection` is `Some` on the eager
    /// cold-prefill DSA path (the value the next `Shared` layer reuses) and
    /// `None` on the dense fallback paths.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        prev_selection: Option<&DsaSelection>,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<(Tensor, Option<DsaSelection>)> {
        let (bs, seq_len, _) = xs.dims3()?;

        let mut q = self.q.forward(xs)?;
        q = q
            .reshape((bs, seq_len, self.num_attention_heads, self.q_head_dim))?
            .transpose(1, 2)?;
        let q_split = q.split(
            &[self.cfg.qk_nope_head_dim, self.cfg.qk_rope_head_dim],
            D::Minus1,
        )?;
        let q_nope = q_split[0].clone();
        let mut q_pe = q_split[1].clone();

        let mut compressed_kv = self.kv_a_proj_with_mqa.forward(xs)?;
        let ckv_split = compressed_kv.split(
            &[self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim],
            D::Minus1,
        )?;
        compressed_kv = ckv_split[0].clone();
        let mut k_pe = ckv_split[1].clone();
        k_pe = k_pe
            .reshape((bs, seq_len, 1, self.cfg.qk_rope_head_dim))?
            .transpose(1, 2)?;

        let ckv = self.kv_a_layernorm.forward(&compressed_kv)?;

        let rope_positions = ctx
            .rope_positions(q_pe.device())?
            .ok_or_else(|| hanzo_ml::Error::msg("missing RoPE positions"))?;
        (q_pe, k_pe) = self
            .rotary_emb
            .forward_positions(&q_pe, &k_pe, rope_positions)?;
        let metadata = ctx.paged_layer(layer_idx);

        let use_mla_decode = should_use_mla_decode(
            attention_mask,
            seq_len,
            self.paged_attn.is_some(),
            q_nope.device(),
            &metadata,
        );

        let mut selection_out = None;
        let mut attn_out = if use_mla_decode {
            mla_decode_forward(
                &q_nope,
                &q_pe,
                &ckv,
                &k_pe,
                &metadata,
                &self.mla_weights,
                self.kv_b_proj.as_ref(),
                &self.sdpa_params,
                self.num_attention_heads,
                self.cfg.kv_lora_rank,
                self.cfg.qk_rope_head_dim,
                self.cfg.qk_nope_head_dim,
                self.cfg.v_head_dim,
                bs,
                seq_len,
            )?
        } else {
            let mut kv = self.kv_b_proj.forward(&ckv)?;
            kv = kv
                .reshape((
                    bs,
                    seq_len,
                    self.num_attention_heads,
                    self.cfg.qk_nope_head_dim + self.cfg.v_head_dim,
                ))?
                .transpose(1, 2)?;

            let kv_split =
                kv.split(&[self.cfg.qk_nope_head_dim, self.cfg.v_head_dim], D::Minus1)?;
            let k_nope = kv_split[0].clone();
            let mut v = kv_split[1].clone();

            let q = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?.contiguous()?;
            let mut k = Tensor::cat(
                &[&k_nope, &k_pe.repeat((1, self.num_attention_heads, 1, 1))?],
                D::Minus1,
            )?
            .contiguous()?;

            let use_mla_cache = should_use_mla_cache(self.paged_attn.is_some(), q.device());

            if use_mla_cache {
                mla_cache_forward(
                    &q,
                    &k,
                    &v,
                    &ckv,
                    &k_pe,
                    attention_mask,
                    ctx.seqlen_offsets(),
                    &metadata,
                    ctx.flash_params(),
                    self.kv_b_proj.as_ref(),
                    &self.sdpa_params,
                    self.num_attention_heads,
                    self.cfg.kv_lora_rank,
                    self.cfg.qk_rope_head_dim,
                    self.cfg.qk_nope_head_dim,
                    self.cfg.v_head_dim,
                    bs,
                    seq_len,
                )?
            } else {
                match &self.paged_attn {
                    Some(paged_attn) => match metadata {
                        Some(((key_cache, value_cache), input_metadata)) => {
                            let v = v
                                .pad_with_zeros(
                                    D::Minus1,
                                    0,
                                    self.q_head_dim - self.cfg.v_head_dim,
                                )?
                                .contiguous()?;
                            paged_attn
                                .forward(
                                    &q,
                                    &k,
                                    &v,
                                    attention_mask,
                                    Some(key_cache),
                                    Some(value_cache),
                                    input_metadata,
                                    &self.sdpa_params,
                                    Some(ctx.flash_params()),
                                )?
                                .narrow(D::Minus1, 0, self.cfg.v_head_dim)?
                        }
                        None => {
                            let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                            assert!(attention_mask.is_custom());
                            let v = v
                                .pad_with_zeros(
                                    D::Minus1,
                                    0,
                                    self.q_head_dim - self.cfg.v_head_dim,
                                )?
                                .contiguous()?;
                            paged_attn
                                .forward(
                                    &q,
                                    &k,
                                    &v,
                                    attention_mask,
                                    None,
                                    None,
                                    &input_metadata,
                                    &self.sdpa_params,
                                    Some(ctx.flash_params()),
                                )?
                                .narrow(D::Minus1, 0, self.cfg.v_head_dim)?
                        }
                    },
                    None => {
                        (k, v) = kv_cache.append(&k, &v)?;

                        // DSA (eager cold-cache prefill only): fold the indexer's
                        // selected-key `-inf` bias into the causal mask. A `Full`
                        // layer computes a fresh selection; a `Shared` layer reuses
                        // `prev_selection`. Guarded to `Lk == seq_len` (cold cache)
                        // so the `[B, seq_len, seq_len]` selection aligns with the
                        // `[seq_len, Lk]` causal mask; warm-cache prefill and decode
                        // stay dense until the FP8 lightning-indexer kernel lands.
                        let dsa_mask = match attention_mask.as_option_tensor() {
                            Some(base) if base.dim(D::Minus1)? == seq_len => {
                                let selection = match self.indexer.as_ref() {
                                    Some(indexer) => {
                                        let positions = ctx
                                            .rope_positions(xs.device())?
                                            .ok_or_else(|| {
                                                hanzo_ml::Error::msg(
                                                    "missing RoPE positions for DSA",
                                                )
                                            })?
                                            .clone();
                                        let q_src = self.q.indexer_q_src(xs)?;
                                        Some(indexer.forward(
                                            &q_src,
                                            xs,
                                            Some(&self.rotary_emb),
                                            Some(&positions),
                                            true,
                                        )?)
                                    }
                                    None => prev_selection.cloned(),
                                };
                                match selection {
                                    Some(sel) => {
                                        let mask =
                                            AttentionMask::Custom(sel.combine_with_mask(base)?);
                                        selection_out = Some(sel);
                                        Some(mask)
                                    }
                                    None => None,
                                }
                            }
                            _ => None,
                        };

                        Sdpa.run_attention(
                            &q,
                            &k,
                            &v,
                            dsa_mask.as_ref().unwrap_or(attention_mask),
                            Some(ctx.flash_params()),
                            &self.sdpa_params,
                        )?
                    }
                }
            }
        };

        attn_out = if attention_mask.is_custom() {
            attn_out.transpose(1, 2)?.reshape((bs, seq_len, ()))?
        } else {
            attn_out.reshape((bs, seq_len, ()))?
        };

        Ok((self.o_proj.forward(&attn_out)?, selection_out))
    }
}

struct MoeGate {
    weight: Tensor,
    e_score_correction_bias: Tensor,
    top_k: usize,
    n_routed_experts: usize,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
}

impl MoeGate {
    fn new(cfg: &Glm5MoeConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let weight = vb.get((cfg.n_routed_experts, cfg.hidden_size), "weight")?;
        let e_score_correction_bias = vb.get_with_hints_dtype(
            cfg.n_routed_experts,
            "e_score_correction_bias",
            Default::default(),
            DType::F32,
        )?;
        Ok(Self {
            weight,
            e_score_correction_bias,
            top_k: cfg.num_experts_per_tok,
            n_routed_experts: cfg.n_routed_experts,
            n_group: cfg.n_group,
            topk_group: cfg.topk_group,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
        })
    }

    /// `noaux_tc` sigmoid routing: `(topk_idx, topk_weight)`.
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (bs, seq_len, h) = xs.dims3()?;
        let xs = xs.reshape(((), h))?;
        let logits = xs
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.weight.t()?.to_dtype(DType::F32)?)?;
        let scores = hanzo_nn::ops::sigmoid(&logits)?;

        let n = bs * seq_len;
        let scores_for_choice =
            scores.broadcast_add(&self.e_score_correction_bias.unsqueeze(0)?)?;
        let group_scores = scores_for_choice
            .reshape((n, self.n_group, ()))?
            .topk(2)?
            .values
            .sum(D::Minus1)?;
        let group_idx = group_scores.topk(self.topk_group)?.indices;
        let mut group_mask = group_scores.zeros_like()?;
        group_mask = group_mask.scatter_add(
            &group_idx,
            &group_idx.ones_like()?.to_dtype(group_mask.dtype())?,
            1,
        )?;
        let score_mask = group_mask
            .unsqueeze(D::Minus1)?
            .expand((n, self.n_group, self.n_routed_experts / self.n_group))?
            .reshape((n, ()))?;
        let masked_bias = score_mask.affine(MASKED_EXPERT_BIAS, -MASKED_EXPERT_BIAS)?;
        let tmp_scores = scores_for_choice.broadcast_add(&masked_bias)?;
        let topk_idx = tmp_scores.topk(self.top_k)?.indices;
        let mut topk_weight = scores.gather(&topk_idx, 1)?;

        if self.norm_topk_prob {
            let denominator = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weight = topk_weight.broadcast_div(&denominator)?;
        }
        topk_weight = (topk_weight * self.routed_scaling_factor)?;

        Ok((topk_idx, topk_weight))
    }
}

struct Moe {
    experts: MoEExperts,
    shared_experts: Mlp,
    gate: MoeGate,
}

impl Moe {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &Glm5MoeConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<hanzo_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or(real_device);

        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.n_routed_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
        };
        let experts = MoEExperts::new(
            &moe_cfg,
            mapper.set_device(layer_idx, vb.clone(), loading_isq),
            layer_device,
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;

        let shared_experts = Mlp::new(
            mapper.set_device(layer_idx, vb.pp("shared_experts"), loading_isq),
            cfg.hidden_size,
            cfg.moe_intermediate_size * cfg.n_shared_experts,
            &cfg.quantization_config,
            cfg.hidden_act,
            comm,
        )?;
        let gate = MoeGate::new(cfg, mapper.set_device(layer_idx, vb.pp("gate"), false))?;
        Ok(Self {
            experts,
            shared_experts,
            gate,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let identity = xs.clone();
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let (topk_idx, topk_weight) = self.gate.forward(xs)?;
        let mut y = self.experts.forward(xs, topk_weight, &topk_idx)?;
        y = y.reshape((b_size, seq_len, hidden_dim))?;
        y = (y + self.shared_experts.forward(&identity)?)?;
        Ok(y)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = self.experts.get_isq_layers();
        layers.push(&mut self.shared_experts.gate);
        layers.push(&mut self.shared_experts.up);
        layers.push(&mut self.shared_experts.down);
        layers
    }
}

enum MoeOrMlp {
    Moe(Box<Moe>),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn: Attention,
    moe_or_mlp: MoeOrMlp,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &Glm5MoeConfig,
        indexer_type: IndexerType,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<hanzo_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        let attn = Attention::new(
            rotary_emb,
            cfg,
            indexer_type,
            vb.pp("self_attn"),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            comm,
        )?;
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        let moe_or_mlp = if cfg.is_moe_layer(layer_idx) {
            MoeOrMlp::Moe(Box::new(Moe::new(
                cfg,
                vb.pp("mlp"),
                mapper,
                layer_idx,
                loading_isq,
                comm,
                real_device,
            )?))
        } else {
            MoeOrMlp::Mlp(Mlp::new(
                mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                comm,
            )?)
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attn,
            moe_or_mlp,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        prev_selection: Option<&DsaSelection>,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<(Tensor, Option<DsaSelection>)> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let (xs, selection) = self.attn.forward(
            &xs,
            attention_mask,
            prev_selection,
            kv_cache,
            ctx,
            layer_idx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .moe_or_mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        Ok(((residual + xs)?, selection))
    }
}

pub struct Glm5Moe {
    lm_head: Arc<dyn QuantMethod>,
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    cache: EitherCache,
    device: Device,
    max_seq_len: usize,
    cfg: ModelConfigMetadata,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl Glm5Moe {
    pub fn new(
        cfg: &Glm5MoeConfig,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let mapper = normal_loading_metadata.mapper;

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(hanzo_nn::Linear::new(
                mapper.cast_nm_device(
                    embed_tokens.embeddings(),
                    normal_loading_metadata.loading_isq,
                )?,
                None,
            ))?
        };
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let rope_type = cfg.rope_parameters.rope_type.as_str();
        if !matches!(rope_type, "default" | "") {
            hanzo_ml::bail!("GLM-5 rope_type `{rope_type}` unsupported (only `default` is wired)");
        }
        let mut ropes = HashMap::new();
        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: None,
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_parameters.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(DeepSeekV2RotaryEmbedding::new(
                    &rope_cfg,
                    vb.dtype(),
                    device,
                )?),
            );
        }

        let indexer_schedule = cfg.indexer_schedule();
        let vb_l = vb_m.pp("layers");
        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(
                    PagedAttention::new(cfg.v_head_dim, device, None)
                        .expect("Failed to create PagedAttention"),
                ),
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                indexer_schedule[layer_idx],
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                &comm,
                normal_loading_metadata.real_device.clone(),
            )
        })?;

        Ok(Self {
            lm_head,
            embed_tokens,
            norm,
            layers,
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            device: normal_loading_metadata.real_device.clone(),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: (cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: None,
                k_head_dim: cfg.q_head_dim(),
                v_head_dim: if matches!(
                    attention_mechanism,
                    AttentionImplementation::PagedAttention
                ) {
                    cfg.q_head_dim()
                } else {
                    cfg.v_head_dim
                },
                #[cfg(all(feature = "cuda", target_family = "unix"))]
                kv_cache_layout: if matches!(
                    attention_mechanism,
                    AttentionImplementation::PagedAttention
                ) {
                    crate::paged_attention::KvCacheLayout::Mla {
                        kv_lora_rank: cfg.kv_lora_rank,
                        kpe_head_dim: cfg.qk_rope_head_dim,
                    }
                } else {
                    crate::paged_attention::KvCacheLayout::Standard
                },
                #[cfg(not(all(feature = "cuda", target_family = "unix")))]
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let cache = &mut self.cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig::default(),
        )?;
        let attention_mask = if ctx.is_first_prompt_chunk() {
            attention_mask
        } else {
            AttentionMask::None
        };
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        let mut selection: Option<DsaSelection> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let (xs_next, sel) = layer.forward(
                &xs,
                &attention_mask.get(xs.device()),
                selection.as_ref(),
                &mut cache[i],
                ctx,
                i,
            )?;
            xs = xs_next;
            selection = sel;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        self.lm_head.forward(&xs)
    }
}

impl IsqModel for Glm5Moe {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.attn.q.a, Some(i)));
            tensors.push((&mut layer.attn.q.b, Some(i)));
            tensors.push((&mut layer.attn.kv_a_proj_with_mqa, Some(i)));
            tensors.push((&mut layer.attn.kv_b_proj, Some(i)));
            tensors.push((&mut layer.attn.o_proj, Some(i)));
            match &mut layer.moe_or_mlp {
                MoeOrMlp::Mlp(mlp) => {
                    tensors.push((&mut mlp.gate, Some(i)));
                    tensors.push((&mut mlp.up, Some(i)));
                    tensors.push((&mut mlp.down, Some(i)));
                }
                MoeOrMlp::Moe(moe) => {
                    for layer in moe.get_isq_layers() {
                        tensors.push((layer, Some(i)));
                    }
                }
            }
        }
        (tensors, &*self.mapper)
    }

    fn get_layers_moe_experts_only(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match &mut layer.moe_or_mlp {
                MoeOrMlp::Mlp(mlp) => {
                    tensors.push((&mut mlp.gate, Some(i)));
                    tensors.push((&mut mlp.up, Some(i)));
                    tensors.push((&mut mlp.down, Some(i)));
                }
                MoeOrMlp::Moe(moe) => {
                    for layer in moe.get_isq_layers() {
                        tensors.push((layer, Some(i)));
                    }
                }
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            uvb_l
                .pp("self_attn")
                .pp("kv_a_layernorm")
                .add(&layer.attn.kv_a_layernorm);

            if let MoeOrMlp::Moe(moe) = &layer.moe_or_mlp {
                uvb_l
                    .pp("mlp")
                    .pp("gate")
                    .add_tensor("weight", moe.gate.weight.clone());
                uvb_l.pp("mlp").pp("gate").add_tensor(
                    "e_score_correction_bias",
                    moe.gate.e_score_correction_bias.clone(),
                );
            }

            uvb_l
                .pp("self_attn")
                .pp("q_a_layernorm")
                .add(&layer.attn.q.norm);
        }

        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Glm5Moe {}

impl NormalModel for Glm5Moe {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for Glm5Moe {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::AttentionImplementation;
    use crate::pipeline::{ModelForwardContext, NormalLoaderType};
    use crate::DeviceMapSetting;
    use hanzo_ml::Shape;
    use hanzo_nn::var_builder::SimpleBackend;
    use hanzo_quant::ShardedSafeTensors;
    use indicatif::MultiProgress;

    /// Deterministic weight generator shared bit-for-bit with the Python
    /// reference (`sum-of-name-bytes` salt; norm weights centered at 1.0, the
    /// router correction bias widened so top-k expert order is unambiguous).
    fn formula(name: &str, numel: usize) -> Vec<f32> {
        let salt: u64 = name.bytes().map(u64::from).sum();
        let (base, amp): (f64, f64) = if name.ends_with("norm.weight") {
            (1.0, 0.1)
        } else if name.ends_with("e_score_correction_bias") {
            (0.0, 1.0)
        } else {
            (0.0, 0.05)
        };
        (0..numel)
            .map(|i| (base + ((i as f64) * 0.09 + (salt as f64) * 0.017).sin() * amp) as f32)
            .collect()
    }

    /// A `SimpleBackend` that materialises each requested tensor from `formula`,
    /// so the engine pulls exactly the checkpoint keys it needs and gets weights
    /// byte-identical to the transformers reference model. The CPU experts backend
    /// consumes the STACKED `gate_up_proj`/`down_proj` layout, so those are built
    /// here from the per-expert `formula` values (the same values the reference's
    /// fused `nn.Parameter`s were built from), transposed into the engine's
    /// `[E, in, out]` on-disk convention.
    struct FormulaBackend;

    fn expert_weight(name: &str, out: usize, inp: usize, dev: &Device) -> Result<Tensor> {
        Tensor::from_vec(formula(name, out * inp), (out, inp), dev)
    }

    impl SimpleBackend for FormulaBackend {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _h: hanzo_nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            if let Some(prefix) = name.strip_suffix("mlp.experts.gate_up_proj") {
                let (e, hidden, two_inter) = (s.dims()[0], s.dims()[1], s.dims()[2]);
                let inter = two_inter / 2;
                let experts: Vec<Tensor> = (0..e)
                    .map(|i| {
                        let g = expert_weight(
                            &format!("{prefix}mlp.experts.{i}.gate_proj.weight"),
                            inter,
                            hidden,
                            dev,
                        )?
                        .t()?;
                        let u = expert_weight(
                            &format!("{prefix}mlp.experts.{i}.up_proj.weight"),
                            inter,
                            hidden,
                            dev,
                        )?
                        .t()?;
                        Tensor::cat(&[&g, &u], 1)?.contiguous()
                    })
                    .collect::<Result<_>>()?;
                return Tensor::stack(&experts, 0)?.to_dtype(dtype);
            }
            if let Some(prefix) = name.strip_suffix("mlp.experts.down_proj") {
                let (e, inter, hidden) = (s.dims()[0], s.dims()[1], s.dims()[2]);
                let experts: Vec<Tensor> = (0..e)
                    .map(|i| {
                        expert_weight(
                            &format!("{prefix}mlp.experts.{i}.down_proj.weight"),
                            hidden,
                            inter,
                            dev,
                        )?
                        .t()?
                        .contiguous()
                    })
                    .collect::<Result<_>>()?;
                return Tensor::stack(&experts, 0)?.to_dtype(dtype);
            }
            Tensor::from_vec(formula(name, s.elem_count()), s, dev)?.to_dtype(dtype)
        }
        fn get_unchecked(&self, name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("FormulaBackend requires a shape; use `get` (asked for `{name}`)")
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    fn loading_metadata(device: &Device, num_layers: usize) -> Result<NormalLoadingMetadata> {
        let mapper = DeviceMapSetting::dummy().into_mapper(
            num_layers,
            device,
            None,
            std::slice::from_ref(device),
        )?;
        Ok(NormalLoadingMetadata {
            mapper,
            loading_isq: false,
            real_device: device.clone(),
            multi_progress: Arc::new(MultiProgress::new()),
            matformer_slicing_config: None,
        })
    }

    /// The real `zai-org/GLM-5.2` `config.json` (arrays inlined), verbatim.
    const REAL_CONFIG: &str = r#"{"architectures":["GlmMoeDsaForCausalLM"],"attention_bias":false,"attention_dropout":0.0,"dtype":"bfloat16","eos_token_id":[154820,154827,154829],"ep_size":1,"first_k_dense_replace":3,"head_dim":192,"hidden_act":"silu","hidden_size":6144,"index_head_dim":128,"index_n_heads":32,"index_share_for_mtp_iteration":true,"index_skip_topk_offset":3,"index_topk":2048,"index_topk_freq":4,"index_topk_pattern":null,"indexer_rope_interleave":true,"indexer_types":["full","full","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared","full","shared","shared","shared"],"initializer_range":0.02,"intermediate_size":12288,"kv_lora_rank":512,"max_position_embeddings":1048576,"mlp_layer_types":["dense","dense","dense","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse","sparse"],"model_type":"glm_moe_dsa","moe_intermediate_size":2048,"moe_layer_freq":1,"moe_router_dtype":"float32","n_group":1,"n_routed_experts":256,"n_shared_experts":1,"norm_topk_prob":true,"num_attention_heads":64,"num_experts_per_tok":8,"num_hidden_layers":78,"num_key_value_heads":64,"num_nextn_predict_layers":1,"pad_token_id":154820,"pretraining_tp":1,"q_lora_rank":2048,"qk_head_dim":256,"qk_nope_head_dim":192,"qk_rope_head_dim":64,"rms_norm_eps":1e-05,"rope_interleave":true,"rope_parameters":{"rope_theta":8000000,"rope_type":"default"},"routed_scaling_factor":2.5,"scoring_func":"sigmoid","tie_word_embeddings":false,"topk_group":1,"topk_method":"noaux_tc","transformers_version":"5.12.0","use_cache":true,"v_head_dim":256,"vocab_size":154880}"#;

    /// Tiny GLM-5-shaped config (MLA + DSA + IndexShare group of 2 + dense/sparse
    /// MLP mix) that runs on CPU in milliseconds; structurally identical to the
    /// real model, dims scaled down. `index_topk (4) < seq_len (6)` so the DSA
    /// cut actually masks keys, and `indexer_types` exercises full+shared reuse.
    const TINY_CONFIG: &str = r#"{
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 20, "moe_intermediate_size": 16,
        "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2,
        "n_shared_experts": 1, "n_routed_experts": 8, "num_experts_per_tok": 2,
        "routed_scaling_factor": 2.5, "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
        "first_k_dense_replace": 1, "q_lora_rank": 24, "kv_lora_rank": 16,
        "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "v_head_dim": 12,
        "index_n_heads": 3, "index_head_dim": 8, "index_topk": 4,
        "max_position_embeddings": 64, "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "hidden_act": "silu", "attention_bias": false, "tie_word_embeddings": false,
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
        "indexer_types": ["full", "shared", "full", "shared"]
    }"#;

    #[test]
    fn glm5_config_parses() {
        let cfg: Glm5MoeConfig = serde_json::from_str(REAL_CONFIG).unwrap();
        assert_eq!(cfg.hidden_size, 6144);
        assert_eq!(cfg.num_hidden_layers, 78);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.q_lora_rank, 2048);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.qk_nope_head_dim, 192);
        assert_eq!(cfg.qk_rope_head_dim, 64);
        assert_eq!(cfg.v_head_dim, 256);
        assert_eq!(cfg.q_head_dim(), 256);
        assert_eq!(cfg.n_routed_experts, 256);
        assert_eq!(cfg.n_shared_experts, 1);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.moe_intermediate_size, 2048);
        assert_eq!(cfg.first_k_dense_replace, 3);
        assert_eq!(cfg.index_n_heads, 32);
        assert_eq!(cfg.index_head_dim, 128);
        assert_eq!(cfg.index_topk, 2048);
        assert_eq!(cfg.rope_parameters.rope_theta, 8_000_000.0);
        assert_eq!(cfg.rope_parameters.rope_type, "default");

        let dsa = cfg.dsa().expect("GLM-5 config must enable DSA");
        assert_eq!(dsa.index_n_heads(), 32);
        assert_eq!(dsa.index_head_dim(), 128);
        assert_eq!(dsa.index_topk(), 2048);
        assert_eq!(dsa.rope_dim(), 64);

        let sched = cfg.indexer_schedule();
        assert_eq!(sched.len(), 78);
        assert_eq!(sched[0], IndexerType::Full);
        assert_eq!(sched[2], IndexerType::Full);
        assert_eq!(sched[3], IndexerType::Shared);
        assert_eq!(sched[6], IndexerType::Full);
        assert_eq!(sched[10], IndexerType::Full);

        assert!(!cfg.is_moe_layer(0));
        assert!(!cfg.is_moe_layer(2));
        assert!(cfg.is_moe_layer(3));
        assert!(cfg.is_moe_layer(77));
    }

    /// Gate regression: GLM-5's `dsa()` now routes through [`DsaConfig::new`], so a
    /// degenerate indexer dim falls back to dense instead of building a mis-shaped
    /// indexer (the struct-literal bypass this path used to have is gone).
    #[test]
    fn glm5_dsa_gate_rejects_degenerate_dims() {
        let mut cfg: Glm5MoeConfig = serde_json::from_str(REAL_CONFIG).unwrap();
        assert!(cfg.dsa().is_some(), "healthy GLM-5 config enables DSA");
        cfg.index_head_dim = 257; // > 256 fails `has_dsa`
        assert!(
            cfg.dsa().is_none(),
            "degenerate indexer head_dim must fall back to dense"
        );
    }

    /// The `index_topk_freq` / `index_skip_topk_offset` derivation reproduces the
    /// explicit 78-entry `indexer_types` array (HF `__post_init__` parity).
    #[test]
    fn indexer_schedule_derivation_matches_explicit() {
        let explicit: Glm5MoeConfig = serde_json::from_str(REAL_CONFIG).unwrap();
        let mut derived = explicit.clone();
        derived.indexer_types = None;
        assert_eq!(derived.indexer_schedule(), explicit.indexer_schedule());
    }

    /// The `index_topk_pattern` string branch ('S' -> Shared, else Full) plus the precedence
    /// contract: explicit `indexer_types` overrides `index_topk_pattern` overrides the cadence.
    #[test]
    fn indexer_schedule_from_topk_pattern_and_precedence() {
        let mut cfg: Glm5MoeConfig = serde_json::from_str(REAL_CONFIG).unwrap();
        cfg.indexer_types = None;
        cfg.index_topk_pattern = Some("FSSSFF".to_string());
        assert_eq!(
            cfg.indexer_schedule(),
            vec![
                IndexerType::Full,
                IndexerType::Shared,
                IndexerType::Shared,
                IndexerType::Shared,
                IndexerType::Full,
                IndexerType::Full,
            ]
        );
        cfg.indexer_types = Some(vec!["shared".to_string(), "full".to_string()]);
        assert_eq!(
            cfg.indexer_schedule(),
            vec![IndexerType::Shared, IndexerType::Full]
        );
    }

    /// noaux_tc group masking must be `masked_fill(-inf)` (HF), not a 0/1 multiply. With n_group>1
    /// and a correction bias that drives in-group scores negative, a zeroed masked expert would
    /// outrank a legitimately-routed negative-score expert. Two groups {0,1},{2,3}; group 1 wins
    /// the group top-k but its scores are negative -> a multiply-mask picks masked expert 0, the
    /// correct `-inf` semantics pick in-group expert 2.
    #[test]
    fn moe_gate_group_mask_matches_masked_fill_neg_inf() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::zeros((4, 1), DType::F32, &device)?;
        let bias = Tensor::from_vec(vec![-1.0f32, -1.0, -0.55, -0.6], 4, &device)?;
        let gate = MoeGate {
            weight,
            e_score_correction_bias: bias,
            top_k: 1,
            n_routed_experts: 4,
            n_group: 2,
            topk_group: 1,
            norm_topk_prob: false,
            routed_scaling_factor: 1.0,
        };
        let xs = Tensor::from_vec(vec![1.0f32], (1, 1, 1), &device)?;
        let (topk_idx, _) = gate.forward(&xs)?;
        assert_eq!(
            topk_idx.to_vec2::<u32>()?[0][0],
            2,
            "masked group experts must be unselectable: expected in-group expert 2, a 0/1 multiply would pick masked expert 0"
        );
        Ok(())
    }

    #[test]
    fn glm5_registry_dispatch() {
        assert_eq!(
            NormalLoaderType::from_causal_lm_name("GlmMoeDsaForCausalLM").unwrap(),
            NormalLoaderType::Glm5Moe
        );
        assert_eq!(
            "glm5moe".parse::<NormalLoaderType>().unwrap(),
            NormalLoaderType::Glm5Moe
        );
        assert_eq!(NormalLoaderType::Glm5Moe.to_string(), "glm5moe");
    }

    /// IndexShare wiring: `Full` layers own an indexer, `Shared` layers do not.
    #[test]
    fn indexshare_wiring_from_schedule() -> Result<()> {
        let device = Device::Cpu;
        let cfg: Glm5MoeConfig = serde_json::from_str(TINY_CONFIG).unwrap();
        let metadata = loading_metadata(&device, cfg.num_hidden_layers)?;
        let vb = ShardedSafeTensors::wrap(Box::new(FormulaBackend), DType::F32, device.clone());
        let model = Glm5Moe::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)?;
        let has_indexer: Vec<bool> = model
            .layers
            .iter()
            .map(|l| l.attn.indexer.is_some())
            .collect();
        assert_eq!(has_indexer, vec![true, false, true, false]);
        Ok(())
    }

    /// Gold gate: full in-process forward matches transformers `GlmMoeDsa` logits
    /// within 1e-4 (fp32 CPU). Exercises interleaved RoPE, MLA, the DSA indexer +
    /// top-k cut, IndexShare reuse across the shared layers, `noaux_tc` sigmoid
    /// routing, per-expert SwiGLU + shared expert, and the lm_head end to end.
    #[test]
    fn glm5_tiny_forward_matches_transformers() -> Result<()> {
        // transformers reference last-token logits (full-precision dump; the
        // extra digits round to the nearest representable f32).
        #[allow(clippy::excessive_precision)]
        const REF_LAST: [f32; 64] = [
            0.050513905,
            -0.0058533661,
            -0.039205465,
            0.081596658,
            -0.1184359,
            0.14721662,
            -0.16598049,
            0.17345087,
            -0.16911942,
            0.15328081,
            -0.12701274,
            0.092102617,
            -0.050925657,
            0.0062836595,
            0.038785879,
            -0.081216365,
            0.1181208,
            -0.14698815,
            0.16585419,
            -0.17343527,
            0.16921559,
            -0.15348227,
            0.12730578,
            -0.092467248,
            0.051337093,
            -0.0067139082,
            -0.038366083,
            0.080835633,
            -0.117805,
            0.14675871,
            -0.16572684,
            0.17341863,
            -0.16931078,
            0.15368283,
            -0.12759803,
            0.092831314,
            -0.051748246,
            0.0071441233,
            0.03794609,
            -0.080454387,
            0.11748845,
            -0.14652845,
            0.16559845,
            -0.17340091,
            0.16940489,
            -0.1538824,
            0.12788951,
            -0.093194827,
            0.052159052,
            -0.0075743459,
            -0.037525836,
            0.080072626,
            -0.11717117,
            0.14629725,
            -0.16546905,
            0.17338212,
            -0.16949803,
            0.15408103,
            -0.12818018,
            0.09355776,
            -0.052569576,
            0.0080044419,
            0.037105322,
            -0.079690382,
        ];
        const REF_SUM: f32 = -0.577956;

        let device = Device::Cpu;
        let cfg: Glm5MoeConfig = serde_json::from_str(TINY_CONFIG).unwrap();
        let metadata = loading_metadata(&device, cfg.num_hidden_layers)?;
        let vb = ShardedSafeTensors::wrap(Box::new(FormulaBackend), DType::F32, device.clone());
        let model = Glm5Moe::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)?;

        let seq_len = 6;
        let input_ids = Tensor::from_vec(vec![3u32, 1, 4, 1, 5, 9], (1, seq_len), &device)?;
        // One starting offset per batch element (fresh prompt at position 0); RoPE
        // derives per-token positions as `offset + seq_idx`.
        let seqlen_offsets: Vec<usize> = vec![0];
        let context_lens = vec![(0usize, seq_len)];
        let position_ids: Vec<usize> = (0..seq_len).collect();
        let flash_params = FlashParams::empty(true);
        let mut ctx = ModelForwardContext::new(
            &seqlen_offsets,
            &context_lens,
            &position_ids,
            None,
            &flash_params,
        );

        let logits = model.forward(&input_ids, &mut ctx)?;
        assert_eq!(logits.dims3()?, (1, seq_len, 64));

        let all = logits.to_vec3::<f32>()?;
        let last = &all[0][seq_len - 1];
        let max_diff = last
            .iter()
            .zip(REF_LAST.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "last-token logits diverge from transformers: max_diff={max_diff}"
        );

        let sum: f32 = all[0].iter().flatten().sum();
        assert!(
            (sum - REF_SUM).abs() < 1e-3,
            "full-logits checksum diverges: {sum} vs {REF_SUM}"
        );
        Ok(())
    }
}
