#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! MiniMax-M2 — a sparse-MoE causal LM (sigmoid-routed, no shared experts).
//!
//! Architecture (verified against the HF `modeling_minimax_m2.py` reference):
//! - GQA attention with a **per-layer** RMSNorm on the *flattened* q/k projections
//!   (`q_norm` over `num_attention_heads * head_dim`, `k_norm` over
//!   `num_key_value_heads * head_dim`) applied *before* the head reshape — this is the
//!   `qk_norm_type = "per_layer"` variant, distinct from qwen3's per-head qk-norm.
//! - **Partial** RoPE: only the first `rotary_dim` (< `head_dim`) channels are rotated, the
//!   rest pass through (`PhiRotaryEmbedding`, NeoX/`rotate_half` convention).
//! - Every layer is MoE: a sigmoid router with an additive `e_score_correction_bias` used for
//!   selection only (the returned weights are the bias-free sigmoid scores, renormalized over the
//!   top-k). No shared experts (`shared_intermediate_size == 0`).
//!
//! Attention math is the qwen3_moe template (paged/naive SDPA via `ctx`); the router is the
//! deepseek3 sigmoid + correction-bias path expressed through the shared `crate::ops::moe_router_topk`
//! (no node groups, so the group masking is dropped). Experts are loaded per-expert in the Mixtral
//! `w1/w2/w3` weight convention (the layout MiniMax-M2 ships on disk) and dispatched with the
//! standard masked gather/scatter loop, so the real checkpoint keys resolve directly.

use crate::layers_masker::CausalMaskConfig;
use hanzo_ml::{DType, Device, Module, Result, Tensor};
use hanzo_nn::Linear;
use hanzo_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        self, embedding, Activation, CausalMasker, PhiRopeConfig, PhiRotaryEmbedding, RmsNorm, Sdpa,
    },
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalLoadingMetadata,
        NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, default_false, false);
serde_default_fn!(String, default_scoring_func, String::from("sigmoid"));

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MiniMaxM2Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    /// Per-expert SwiGLU intermediate size (the "moe_intermediate" size; `1536` for M2).
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    /// Number of routed experts per layer (`num_local_experts == 256` for M2).
    #[serde(alias = "num_experts")]
    pub(crate) num_local_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) head_dim: Option<usize>,
    /// RoPE rotary width; `< head_dim` means partial rotary (`64` of `128` for M2).
    pub(crate) rotary_dim: Option<usize>,
    pub(crate) sliding_window: Option<usize>,
    #[serde(default = "default_false")]
    pub(crate) use_qk_norm: bool,
    #[serde(default = "default_false")]
    pub(crate) use_routing_bias: bool,
    #[serde(default = "default_scoring_func")]
    pub(crate) scoring_func: String,
    #[serde(default = "default_false")]
    pub(crate) tie_word_embeddings: bool,
    #[serde(alias = "quantization")]
    pub(crate) quantization_config: Option<QuantizedConfig>,
}

impl MiniMaxM2Config {
    pub(crate) fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Rotary width — defaults to the full head dim (no partial rotary) when unspecified.
    pub(crate) fn rotary_dim(&self) -> usize {
        self.rotary_dim.unwrap_or_else(|| self.head_dim())
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    // Per-layer (full-projection) qk-norm, applied to the flat q/k before the head reshape.
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<PhiRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<PhiRotaryEmbedding>,
        cfg: &MiniMaxM2Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<hanzo_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = hanzo_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            head_dim,
            comm,
        )?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;
        let (q_norm, k_norm) = if cfg.use_qk_norm {
            // Per-layer qk-norm: one RMSNorm over the entire flattened q/k projection.
            let q_norm = RmsNorm::new(
                num_heads * head_dim,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("q_norm"), false),
            )?;
            let k_norm = RmsNorm::new(
                num_kv_heads * head_dim,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("k_norm"), false),
            )?;
            (Some(q_norm), Some(k_norm))
        } else {
            (None, None)
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: hanzo_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                )?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let (mut q, mut k, v) =
            crate::ops::qkv_projections(xs, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        // Per-layer qk-norm over the full flat projection (before the head reshape).
        if let Some(q_norm) = &self.q_norm {
            q = q_norm.forward(&q)?;
        }
        if let Some(k_norm) = &self.k_norm {
            k = k_norm.forward(&k)?;
        }

        let (mut q, mut k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        let position_ids = ctx.position_ids_vec();
        let rope_positions = ctx
            .rope_positions(q.device())?
            .ok_or_else(|| hanzo_ml::Error::msg("missing RoPE positions"))?;
        (q, k) = self
            .rotary_emb
            .forward_positions(&q, &k, rope_positions, &position_ids)?;
        let metadata = ctx.paged_layer(layer_idx);

        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k.contiguous()?,
                    &v.contiguous()?,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(ctx.flash_params()),
                )?,
                None => {
                    // No metadata: most likely an imatrix pass, so we don't populate the cache.
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(!matches!(attention_mask, AttentionMask::None));
                    paged_attn.forward(
                        &q,
                        &k.contiguous()?,
                        &v.contiguous()?,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(ctx.flash_params()),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(ctx.flash_params()),
                    &self.sdpa_params,
                )?
            }
        };

        attn_output = if !matches!(attention_mask, AttentionMask::None) {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let res = self.o_proj.forward(&attn_output)?;
        Ok(res)
    }
}

/// One routed expert: a SwiGLU MLP in the Mixtral weight convention (`w1` = gate, `w3` = up,
/// `w2` = down) — exactly the layout MiniMax-M2 ships on disk under
/// `block_sparse_moe.experts.{i}.{w1,w2,w3}`. (The shared `crate::moe::MoEExperts` machinery only
/// reads the `gate_proj/up_proj/down_proj` convention, so per-expert loading here is what lets the
/// real checkpoint keys resolve; a fused 256-expert path would require teaching MoEExperts the
/// `w1/w2/w3` names.)
#[derive(Clone)]
struct ExpertMlp {
    w1: Arc<dyn QuantMethod>,
    w2: Arc<dyn QuantMethod>,
    w3: Arc<dyn QuantMethod>,
    act_fn: Activation,
}

impl ExpertMlp {
    fn new(
        cfg: &MiniMaxM2Config,
        vb: ShardedVarBuilder,
        comm: &Arc<hanzo_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let w1 = ColumnParallelLayer::new(
            hidden_sz,
            intermediate_sz,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("w1"),
        )?;
        let w2 = RowParallelLayer::new(
            intermediate_sz,
            hidden_sz,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("w2"),
        )?;
        let w3 = ColumnParallelLayer::new(
            hidden_sz,
            intermediate_sz,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("w3"),
        )?;
        Ok(Self {
            w1,
            w2,
            w3,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1_out = self.w1.forward(xs)?;
        let w3_out = self.w3.forward(xs)?;
        let activated = crate::ops::mul_and_act(&w1_out, &w3_out, self.act_fn)?;
        self.w2.forward(&activated)
    }
}

/// Sparse MoE block: sigmoid router + correction-bias top-k selection over per-expert SwiGLU MLPs.
struct MoeBlock {
    gate: Linear,
    e_score_correction_bias: Option<Tensor>,
    experts: Vec<ExpertMlp>,
    num_experts_per_tok: usize,
}

impl MoeBlock {
    fn new(
        cfg: &MiniMaxM2Config,
        vb: ShardedVarBuilder,
        comm: &Arc<hanzo_quant::Comm>,
    ) -> Result<Self> {
        // Router stays full precision (MiniMax-M2 lists `gate`/`e_score_correction_bias` in
        // `modules_to_not_convert`), so it is a plain unquantized linear.
        let gate = layers::linear_no_bias(cfg.hidden_size, cfg.num_local_experts, vb.pp("gate"))?;

        // Additive selection bias (`e_score_correction_bias`), a sibling of `gate`/`experts`.
        let e_score_correction_bias = if cfg.use_routing_bias {
            Some(vb.get_with_hints_dtype(
                cfg.num_local_experts,
                "e_score_correction_bias",
                Default::default(),
                DType::F32,
            )?)
        } else {
            None
        };

        let experts_vb = vb.pp("experts");
        let mut experts = Vec::with_capacity(cfg.num_local_experts);
        for idx in 0..cfg.num_local_experts {
            experts.push(ExpertMlp::new(cfg, experts_vb.pp(idx), comm)?);
        }

        Ok(Self {
            gate,
            e_score_correction_bias,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;

        let router_logits = self.gate.forward(&xs)?;
        // sigmoid(scores); add the correction bias for SELECTION only; the returned weights are the
        // bias-free sigmoid scores at the selected experts, renormalized over the top-k.
        let topk = crate::ops::moe_router_topk(
            &router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.num_experts_per_tok,
                score_function: crate::ops::MoeRouterScoreFunction::Sigmoid,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: true,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            self.e_score_correction_bias.as_ref(),
            None,
        )?;

        // Masked per-expert dispatch (Mixtral convention): gather each expert's routed tokens, run
        // its SwiGLU, scale by the routing weight, and scatter-add back.
        let selected_experts = topk.indices.to_vec2::<u32>()?;
        let routing_weights = topk.values.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        let mut top_x = vec![Vec::new(); self.experts.len()];
        let mut selected_rws = vec![Vec::new(); self.experts.len()];
        for (row_idx, (experts, weights)) in selected_experts
            .iter()
            .zip(routing_weights.iter())
            .enumerate()
        {
            for (&expert_idx, &routing_weight) in experts.iter().zip(weights.iter()) {
                top_x[expert_idx as usize].push(row_idx as u32);
                selected_rws[expert_idx as usize].push(routing_weight);
            }
        }

        let mut ys = xs.zeros_like()?;
        for (expert_idx, expert_layer) in self.experts.iter().enumerate() {
            let top_x = &top_x[expert_idx];
            if top_x.is_empty() {
                continue;
            }
            let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
            let selected_rws =
                Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?.reshape(((), 1))?;
            let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
            let current_hidden_states = expert_layer.forward(&current_state)?;
            let current_hidden_states = current_hidden_states.broadcast_mul(&selected_rws)?;
            ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
        }

        ys.reshape((b_size, seq_len, hidden_dim))
    }

    fn gate(&self) -> &Linear {
        &self.gate
    }
}

struct DecoderLayer {
    self_attn: Attention,
    block_sparse_moe: MoeBlock,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<PhiRotaryEmbedding>,
        cfg: &MiniMaxM2Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<hanzo_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            comm,
        )?;

        let block_sparse_moe = MoeBlock::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("block_sparse_moe"), loading_isq),
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
        Ok(Self {
            self_attn,
            block_sparse_moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, kv_cache, ctx, layer_idx)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .block_sparse_moe
            .forward(&xs.apply(&self.post_attention_layernorm)?)?
            .to_dtype(residual.dtype())?;
        residual + xs
    }
}

pub struct Model {
    embed_tokens: hanzo_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &MiniMaxM2Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rope_cfg = PhiRopeConfig {
                rope_scaling: None,
                max_position_embeddings: cfg.max_position_embeddings,
                original_max_position_embeddings: cfg.max_position_embeddings,
                rope_theta: cfg.rope_theta,
                head_dim: cfg.head_dim(),
                partial_rotary_factor: Some(cfg.rotary_dim() as f64 / cfg.head_dim() as f64),
            };
            ropes.insert(
                device.location(),
                Arc::new(PhiRotaryEmbedding::new(vb_m.dtype(), rope_cfg, device)?),
            );
        }

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
                    PagedAttention::new(cfg.head_dim(), device, None)
                        .expect("PagedAttention creation failed"),
                ),
            };
            let comm = mapper
                .get_comm_for(layer_idx)
                .expect("Failed to get comm for layer");
            DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                &comm,
            )
        })?;
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
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
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: normal_loading_metadata.real_device.clone(),
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let cache = &mut self.cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig::default(),
        )?;
        // PagedAttention prompt chunking
        let attention_mask = if ctx.is_first_prompt_chunk() {
            attention_mask
        } else {
            AttentionMask::None
        };
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(&xs, &attention_mask.get(xs.device()), &mut cache[i], ctx, i)?;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        self.lm_head.forward(&xs)
    }
}

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.q_proj, Some(i)));
            tensors.push((&mut layer.self_attn.k_proj, Some(i)));
            tensors.push((&mut layer.self_attn.v_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            for expert in &mut layer.block_sparse_moe.experts {
                tensors.push((&mut expert.w1, Some(i)));
                tensors.push((&mut expert.w2, Some(i)));
                tensors.push((&mut expert.w3, Some(i)));
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
            if let Some(q_norm) = &layer.self_attn.q_norm {
                uvb_l.pp("self_attn").pp("q_norm").add(q_norm);
            }
            if let Some(k_norm) = &layer.self_attn.k_norm {
                uvb_l.pp("self_attn").pp("k_norm").add(k_norm);
            }
            let uvb_moe = uvb_l.pp("block_sparse_moe");
            uvb_moe.pp("gate").add(layer.block_sparse_moe.gate());
            if let Some(bias) = &layer.block_sparse_moe.e_score_correction_bias {
                uvb_moe.add_tensor("e_score_correction_bias", bias.clone());
            }
        }

        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl NormalModel for Model {
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
    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn supports_cuda_decode_graphs(&self) -> bool {
        true
    }
}

impl AnyMoeBaseModelMixin for Model {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::AttentionImplementation;
    use crate::pipeline::ModelForwardContext;
    use crate::DeviceMapSetting;
    use hanzo_ml::{Device, Shape};
    use hanzo_quant::ShardedSafeTensors;
    use indicatif::MultiProgress;

    /// A `SimpleBackend` that returns a zero tensor of the requested shape for any name — lets us
    /// build the full model graph without the (~460 GB) real weights and assert it composes.
    struct ZerosBackend;

    impl hanzo_nn::var_builder::SimpleBackend for ZerosBackend {
        fn get(
            &self,
            s: Shape,
            _name: &str,
            _h: hanzo_nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            Tensor::zeros(s, dtype, dev)
        }
        fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("ZerosBackend requires a shape; use `get`")
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    fn zeros_vb(device: &Device) -> ShardedVarBuilder {
        ShardedSafeTensors::wrap(Box::new(ZerosBackend), DType::F32, device.clone())
    }

    /// Like `ZerosBackend`, but records every requested tensor path so a test can assert the loader
    /// asks for exactly the REAL MiniMax-M2 checkpoint keys (`block_sparse_moe.experts.{i}.w1/w2/w3`)
    /// and never the shared-`MoEExperts` `gate_proj/up_proj/down_proj` names.
    struct RecordingBackend {
        seen: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    }

    impl hanzo_nn::var_builder::SimpleBackend for RecordingBackend {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _h: hanzo_nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            self.seen.lock().unwrap().insert(name.to_string());
            Tensor::zeros(s, dtype, dev)
        }
        fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("RecordingBackend requires a shape; use `get`")
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    fn loading_metadata(device: &Device, num_layers: usize) -> Result<NormalLoadingMetadata> {
        let mapper =
            DeviceMapSetting::dummy().into_mapper(num_layers, device, None, &[device.clone()])?;
        Ok(NormalLoadingMetadata {
            mapper,
            loading_isq: false,
            real_device: device.clone(),
            multi_progress: Arc::new(MultiProgress::new()),
            matformer_slicing_config: None,
        })
    }

    // Byte-for-byte copy of the real MiniMax-M2 `config.json` arch block (the `attn_type_list`
    // padding is elided — every entry is `1`/full-attention, which the loader ignores). Embedded so
    // the test is hermetic instead of reading an absolute checkout path.
    const REAL_CONFIG: &str = r#"{
        "architectures": ["MiniMaxM2ForCausalLM"],
        "attention_dropout": 0.0,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 3072,
        "intermediate_size": 1536,
        "max_position_embeddings": 196608,
        "mlp_intermediate_size": 8192,
        "model_type": "minimax_m2",
        "num_attention_heads": 48,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 62,
        "num_key_value_heads": 8,
        "num_local_experts": 256,
        "qk_norm_type": "per_layer",
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "float8_e4m3fn",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["gate", "e_score_correction_bias", "lm_head"]
        },
        "rms_norm_eps": 1e-06,
        "rope_theta": 5000000,
        "rotary_dim": 64,
        "scoring_func": "sigmoid",
        "shared_intermediate_size": 0,
        "sliding_window": null,
        "tie_word_embeddings": false,
        "use_qk_norm": true,
        "use_routing_bias": true,
        "vocab_size": 200064
    }"#;

    // A tiny config exercising the same wiring (partial rotary + per-layer qk-norm + sigmoid-bias
    // MoE + per-expert SwiGLU + lm_head) at a size that runs instantly on CPU.
    const TOY_CONFIG: &str = r#"{
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "hidden_act": "silu",
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000,
        "rotary_dim": 8,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "use_qk_norm": true,
        "use_routing_bias": true,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": false
    }"#;

    #[test]
    fn minimax_m2_config_parses() {
        let cfg: MiniMaxM2Config = serde_json::from_str(REAL_CONFIG).unwrap();
        assert_eq!(cfg.num_hidden_layers, 62);
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_attention_heads, 48);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.rotary_dim(), 64);
        assert!(cfg.rotary_dim() < cfg.head_dim(), "partial rotary expected");
        assert_eq!(cfg.num_local_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.intermediate_size, 1536); // per-expert SwiGLU size
        assert!(cfg.use_qk_norm);
        assert!(cfg.use_routing_bias);
        assert_eq!(cfg.scoring_func, "sigmoid");
        assert!(!cfg.tie_word_embeddings);
        assert!(cfg.quantization_config.is_some()); // fp8 block parses
    }

    #[test]
    fn minimax_m2_toy_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let cfg: MiniMaxM2Config = serde_json::from_str(TOY_CONFIG).unwrap();
        let metadata = loading_metadata(&device, cfg.num_hidden_layers)?;
        let model = Model::new(
            &cfg,
            zeros_vb(&device),
            true,
            metadata,
            AttentionImplementation::Eager,
        )?;

        let seq_len = 5;
        let input_ids = Tensor::from_vec(vec![0u32, 1, 2, 3, 4], (1, seq_len), &device)?;
        // One starting offset per batch element (batch of 1, fresh prompt at position 0); RoPE
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
        assert_eq!(logits.dims3()?, (1, seq_len, 128));
        Ok(())
    }

    #[test]
    fn minimax_m2_loads_real_keys() -> Result<()> {
        let device = Device::Cpu;
        let cfg: MiniMaxM2Config = serde_json::from_str(TOY_CONFIG).unwrap();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        let vb = ShardedSafeTensors::wrap(
            Box::new(RecordingBackend { seen: seen.clone() }),
            DType::F32,
            device.clone(),
        );
        let metadata = loading_metadata(&device, cfg.num_hidden_layers)?;
        // Building the model issues a `get` for every weight — backed by the real-key names.
        let _ = Model::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)?;

        let seen = seen.lock().unwrap();
        for i in 0..cfg.num_hidden_layers {
            for j in 0..cfg.num_local_experts {
                for w in ["w1", "w2", "w3"] {
                    let key = format!("model.layers.{i}.block_sparse_moe.experts.{j}.{w}.weight");
                    assert!(
                        seen.contains(&key),
                        "loader never requested expert key `{key}`"
                    );
                }
            }
            assert!(seen.contains(&format!("model.layers.{i}.block_sparse_moe.gate.weight")));
            assert!(seen.contains(&format!(
                "model.layers.{i}.block_sparse_moe.e_score_correction_bias"
            )));
            assert!(seen.contains(&format!("model.layers.{i}.self_attn.q_norm.weight")));
            assert!(seen.contains(&format!("model.layers.{i}.self_attn.k_norm.weight")));
        }
        // The shared-`MoEExperts` naming must NOT appear — those keys would error on the real
        // checkpoint, which ships the Mixtral `w1/w2/w3` convention.
        assert!(
            !seen.iter().any(|k| k.contains("gate_proj")
                || k.contains("up_proj")
                || k.contains("down_proj")),
            "found a gate_proj/up_proj/down_proj key; real MiniMax-M2 uses w1/w2/w3"
        );
        Ok(())
    }

    #[test]
    fn minimax_m2_registry_dispatch() {
        use crate::pipeline::NormalLoaderType;
        assert_eq!(
            NormalLoaderType::from_causal_lm_name("MiniMaxM2ForCausalLM").unwrap(),
            NormalLoaderType::MiniMaxM2
        );
        assert_eq!(
            "minimax_m2".parse::<NormalLoaderType>().unwrap(),
            NormalLoaderType::MiniMaxM2
        );
    }
}
