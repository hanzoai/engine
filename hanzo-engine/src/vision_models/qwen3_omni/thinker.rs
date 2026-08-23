#![allow(dead_code)]

//! Qwen3-Omni-MoE **Thinker** text decoder (`thinker.model.*` + `thinker.lm_head`).
//!
//! Architecturally this is the Qwen3-MoE text stack (see [`crate::models::qwen3_moe`]) and shares
//! the exact attention used by the sibling [`super::talker`]: GQA with per-head q/k RMSNorm and 1D
//! RoPE evaluated through [`naive_sdpa`]. It differs from the talker MoE block in one way: there is
//! **no shared expert** (`shared_expert_intermediate_size == 0`), so each sparse layer is just a
//! router `gate` + per-expert SwiGLU experts, with the same softmax/top-k routing as
//! `qwen3_moe::MoeMlp`.
//!
//! For a text-only input the interleaved 3D mRoPE collapses to standard 1D RoPE — every position
//! axis (temporal/height/width) carries the same sequential position, so a plain
//! [`RotaryEmbedding`] is numerically exact.
//!
//! [`OmniThinkerText::forward`] returns the final `lm_head` logits **and** the full residual-stream
//! history (`hidden_states[0]` = token embeddings, `hidden_states[i]` = the stream after `i` decoder
//! layers) so the talker bridge can read layer `accept_hidden_layer` — matching HF
//! `output_hidden_states` indexing.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{Embedding, Linear, Module};
use hanzo_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};

use crate::{
    attention::{naive_sdpa, AttentionMask, SdpaParams},
    layers::{self, repeat_kv, Qwen3VLRotaryEmbedding, RmsNorm, RotaryEmbedding},
    ops::{moe_router_topk, MoeRouterScoreFunction, MoeRouterSelectedWeight, MoeRouterTopKConfig},
    paged_attention::{AttentionImplementation, PagedAttention},
    pipeline::{KvCache, ModelForwardContext},
    utils::unvarbuilder::UnVarBuilder,
};

use super::config::OmniTextConfig;

/// Qwen3 attention: GQA + per-head q/k RMSNorm + 1D RoPE via `naive_sdpa`. Identical to the talker
/// backbone attention; carries `self_attn.{q,k,v,o}_proj` (no bias) and `self_attn.{q,k}_norm`.
///
/// The four projections are [`QuantMethod`] layers so the Thinker is in-situ quantizable (ISQ) and
/// can load a pre-quantized (FP8/GPTQ) checkpoint; with no quantization they are `UnquantLinear`,
/// numerically identical to the plain `Linear` they replaced. Single-process [`ReplicatedLayer`]
/// (never sharded) keeps the weights complete for the cacheless reference math.
struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: Arc<RotaryEmbedding>,
    /// Interleaved 3D mRoPE for the multimodal serving path (vision). For text/audio the 3D
    /// positions collapse to 1D, so `rope` (above) is used; only image/video need this.
    mrope: Arc<Qwen3VLRotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Paged-attention kernel for the serving cache path; `None` under [`AttentionImplementation::Eager`].
    /// The validated cacheless forwards ([`Self::forward`] / [`Self::forward_mrope`]) never use it —
    /// only [`Self::attend_cached`] does, and only when `ctx` carries per-layer paged metadata. Built
    /// per [`AttentionImplementation`], mirroring [`crate::models::qwen3_moe`].
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vb: ShardedVarBuilder,
        rope: Arc<RotaryEmbedding>,
        mrope: Arc<Qwen3VLRotaryEmbedding>,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        qcfg: &Option<QuantizedConfig>,
        attention_mechanism: AttentionImplementation,
        device: &Device,
    ) -> Result<Self> {
        let q_proj = ReplicatedLayer::new(
            hidden_size,
            num_heads * head_dim,
            qcfg,
            false,
            vb.pp("q_proj"),
        )?;
        let k_proj = ReplicatedLayer::new(
            hidden_size,
            num_kv_heads * head_dim,
            qcfg,
            false,
            vb.pp("k_proj"),
        )?;
        let v_proj = ReplicatedLayer::new(
            hidden_size,
            num_kv_heads * head_dim,
            qcfg,
            false,
            vb.pp("v_proj"),
        )?;
        let o_proj = ReplicatedLayer::new(
            num_heads * head_dim,
            hidden_size,
            qcfg,
            false,
            vb.pp("o_proj"),
        )?;
        let q_norm = RmsNorm::new(head_dim, rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, rms_norm_eps, vb.pp("k_norm"))?;
        let paged_attn = match attention_mechanism {
            AttentionImplementation::Eager => None,
            AttentionImplementation::PagedAttention => {
                Some(PagedAttention::new(head_dim, device, None)?)
            }
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
            mrope,
            num_heads,
            num_kv_heads,
            head_dim,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
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
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _d) = xs.dims3()?;
        let mut q = self
            .q_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?;
        let mut k = self
            .k_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm over head_dim (Qwen3 QK-norm), applied before RoPE.
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;

        q = q.transpose(1, 2)?.contiguous()?;
        k = k.transpose(1, 2)?.contiguous()?;

        let (q, k) = self.rope.forward(&q, &k, seqlen_offsets)?;

        let k = repeat_kv(k, self.sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v, self.sdpa_params.n_kv_groups)?;

        let attn = naive_sdpa(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            mask,
            &self.sdpa_params,
        )?;
        let attn = attn.transpose(1, 2)?.reshape((b, t, ()))?;
        self.o_proj.forward(&attn)
    }

    /// Cache-aware attention for serving. Identical math to [`Self::forward`] — the same q/k/v
    /// projections, Qwen3 QK-norm, 1D RoPE, and [`naive_sdpa`] — but the freshly-RoPE'd K/V are
    /// appended into the engine [`KvCache`] and attention runs over the full cached sequence, so
    /// decode reuses past K/V instead of recomputing. `mask` is the additive mask for the current
    /// query rows (`None` for single-token decode, where the query attends to every cached key).
    #[allow(clippy::too_many_arguments)]
    fn forward_cached(
        &self,
        xs: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
        ctx: &ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b, t, _d) = xs.dims3()?;
        let mut q = self
            .q_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?;
        let mut k = self
            .k_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm over head_dim (Qwen3 QK-norm), applied before RoPE.
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;

        q = q.transpose(1, 2)?.contiguous()?;
        k = k.transpose(1, 2)?.contiguous()?;

        let (q, k) = self.rope.forward(&q, &k, seqlen_offsets)?;

        let attn = self.attend_cached(&q, &k, &v, mask, kv_cache, ctx, layer_idx)?;
        self.o_proj.forward(&attn)
    }

    /// Cache-aware attention using **interleaved 3D mRoPE** (the multimodal serving path). Identical
    /// to [`Self::forward_cached`] except the 1D `rope` is replaced by the precomputed `(cos, sin)`
    /// from [`Qwen3VLRotaryEmbedding`]; Qwen3 QK-norm + mRoPE are applied together by the validated
    /// `forward_qk_norm` (the exact rotary path `qwen3_vl`/`qwen3_vl_moe` use). For text/audio inputs
    /// these positions equal the 1D positions, so the two paths agree numerically.
    #[allow(clippy::too_many_arguments)]
    fn forward_cached_mrope(
        &self,
        xs: &Tensor,
        cos_sin: &(Tensor, Tensor),
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
        ctx: &ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b, t, _d) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Fused Qwen3 QK-norm + interleaved mRoPE on [b, heads, t, head_dim].
        let (q, k) = self.mrope.forward_qk_norm(
            cos_sin,
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
        )?;

        let attn = self.attend_cached(&q, &k, &v, mask, kv_cache, ctx, layer_idx)?;
        self.o_proj.forward(&attn)
    }

    /// Shared cache-aware attention core for the serving forwards ([`Self::forward_cached`] and
    /// [`Self::forward_cached_mrope`]). `q` is post-RoPE/mRoPE `[b, heads, t, head_dim]`; `k`/`v` are
    /// `[b, kv_heads, t, head_dim]`.
    ///
    /// When the model was built with paged attention **and** `ctx` carries this layer's paged
    /// metadata, attention runs through the paged kernel (continuous batching / KV paging). Otherwise
    /// it falls back to the validated engine [`KvCache`] append + [`naive_sdpa`] path — byte-identical
    /// to the pre-paged serving forward, which is why the serving tests (no paged metadata) keep their
    /// numerics. Returns the attention output reshaped to `[b, t, heads*head_dim]` (pre-`o_proj`).
    /// Mirrors [`crate::models::qwen3_moe`]'s attention branch.
    #[allow(clippy::too_many_arguments)]
    fn attend_cached(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
        ctx: &ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b, _heads, t, _hd) = q.dims4()?;
        match (&self.paged_attn, ctx.paged_layer(layer_idx)) {
            (Some(paged_attn), Some(((key_cache, value_cache), input_metadata))) => {
                // Serving against a real paged KV cache. `Custom` carries the prefill mask; decode
                // passes `None` (the single new token attends every cached key).
                let attention_mask = match mask {
                    Some(m) => AttentionMask::Custom(m.clone()),
                    None => AttentionMask::None,
                };
                let attn = paged_attn.forward(
                    &q.contiguous()?,
                    &k.contiguous()?,
                    &v.contiguous()?,
                    &attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(ctx.flash_params()),
                )?;
                // Prefill (Custom) returns `[b, heads, t, hd]`; decode (None) returns
                // `[b*t, heads, hd]`. Match `qwen3_moe`'s reshape exactly.
                if matches!(attention_mask, AttentionMask::None) {
                    attn.reshape((b, t, ()))
                } else {
                    attn.transpose(1, 2)?.reshape((b, t, ()))
                }
            }
            // Eager build, or no paged metadata: the validated cache + `naive_sdpa` serving path.
            // Append the RoPE'd K/V into the running cache, then attend over the full sequence
            // `[b, kv_heads, past + t, head_dim]`.
            _ => {
                let (k, v) = kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;
                let k = repeat_kv(k, self.sdpa_params.n_kv_groups)?;
                let v = repeat_kv(v, self.sdpa_params.n_kv_groups)?;
                let attn = naive_sdpa(
                    &q.contiguous()?,
                    &k.contiguous()?,
                    &v.contiguous()?,
                    mask,
                    &self.sdpa_params,
                )?;
                attn.transpose(1, 2)?.reshape((b, t, ()))
            }
        }
    }

    /// Cacheless attention using **interleaved 3D mRoPE** — the model-level reference for vision.
    /// Identical to [`Self::forward`] (full attention, no cache append) except the 1D RoPE is replaced
    /// by the precomputed `(cos, sin)` and Qwen3 QK-norm + mRoPE are fused by `forward_qk_norm`, the
    /// exact rotary path [`Self::forward_cached_mrope`] uses. For text/audio (equal-axis positions)
    /// this reduces to the 1D path.
    fn forward_mrope(
        &self,
        xs: &Tensor,
        cos_sin: &(Tensor, Tensor),
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _d) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.mrope.forward_qk_norm(
            cos_sin,
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
        )?;

        let k = repeat_kv(k, self.sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v, self.sdpa_params.n_kv_groups)?;

        let attn = naive_sdpa(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            mask,
            &self.sdpa_params,
        )?;
        let attn = attn.transpose(1, 2)?.reshape((b, t, ()))?;
        self.o_proj.forward(&attn)
    }
}

/// Dense SwiGLU MLP: `down(silu(gate(x)) * up(x))`. Used both as a single MoE expert and for any
/// `mlp_only_layers` (the Qwen3-Omni thinker has none, but the architecture allows them). No bias.
///
/// The three projections are [`QuantMethod`] layers (ISQ / pre-quantized capable; `UnquantLinear`
/// and bit-identical to a plain `Linear` when unquantized). The inline MoE gather/scatter in
/// [`MoeMlp::forward`] runs each engaged expert through [`Self::forward`], so per-expert
/// quantization composes with that path with no kernel changes — `QuantMethod::forward` casts the
/// activations to the weight's quantized type and back.
struct SwiGluMlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
}

impl SwiGluMlp {
    fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        qcfg: &Option<QuantizedConfig>,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: ReplicatedLayer::new(
                hidden_size,
                intermediate_size,
                qcfg,
                false,
                vb.pp("gate_proj"),
            )?,
            up_proj: ReplicatedLayer::new(
                hidden_size,
                intermediate_size,
                qcfg,
                false,
                vb.pp("up_proj"),
            )?,
            down_proj: ReplicatedLayer::new(
                intermediate_size,
                hidden_size,
                qcfg,
                false,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Qwen3-MoE block, **no shared expert**: a router `gate` + per-expert SwiGLU MLPs.
///
/// Routing mirrors `qwen3_moe::MoeMlp` exactly — softmax over all experts, top-`k` selection, and
/// (when `norm_topk_prob`) renormalization — via the shared [`moe_router_topk`]. The expert compute
/// is the same gather → expert → weighted scatter-add that `MoEExperts`' loop (`Slow`) backend runs
/// internally; it is inlined here as plain `index_select` / `index_add` so it is device-agnostic.
/// (The `MoEExperts` wrapper cannot serve this checkpoint on CPU: its `Slow` loader requires the
/// combined `gate_up_proj` layout while this checkpoint is per-expert, and its `Fast` gather kernel
/// rejects the 4-D down-projection input outside CUDA.)
struct MoeMlp {
    gate: Linear,
    experts: Vec<SwiGluMlp>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl MoeMlp {
    fn new(cfg: &OmniTextConfig, vb: ShardedVarBuilder) -> Result<Self> {
        // The router `gate` stays a full-precision `Linear`: it is tiny (`hidden × num_experts`) and
        // routing is precision-sensitive, so it is deliberately excluded from quantization (matches
        // the canonical `qwen3_moe`, where the gate lives in `residual_tensors`).
        let gate = layers::linear_no_bias(cfg.hidden_size, cfg.num_experts, vb.pp("gate"))?;
        let vb_e = vb.pp("experts");
        let experts = (0..cfg.num_experts)
            .map(|i| {
                SwiGluMlp::new(
                    vb_e.pp(i),
                    cfg.hidden_size,
                    cfg.moe_intermediate_size,
                    &cfg.quantization_config,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, hidden_dim) = xs.dims3()?;
        let device = xs.device();
        let xs_flat = xs.reshape(((), hidden_dim))?;

        let router_logits = self.gate.forward(&xs_flat)?;
        let topk = moe_router_topk(
            &router_logits,
            MoeRouterTopKConfig {
                top_k: self.num_experts_per_tok,
                score_function: MoeRouterScoreFunction::Softmax,
                selected_weight: MoeRouterSelectedWeight::Score,
                renormalize: self.norm_topk_prob,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )?;

        // Bucket the (token, weight) assignments by expert.
        let weights = topk.values.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let ids = topk.indices.to_vec2::<u32>()?;
        let n_experts = self.experts.len();
        let mut rows = vec![Vec::<u32>::new(); n_experts];
        let mut wts = vec![Vec::<f32>::new(); n_experts];
        for (row, (rw, ix)) in weights.iter().zip(ids.iter()).enumerate() {
            for (&w, &e) in rw.iter().zip(ix.iter()) {
                rows[e as usize].push(row as u32);
                wts[e as usize].push(w);
            }
        }

        // Run each engaged expert over its routed tokens and weighted scatter-add into the output.
        let mut ys = xs_flat.zeros_like()?;
        for (e, expert) in self.experts.iter().enumerate() {
            if rows[e].is_empty() {
                continue;
            }
            let idx = Tensor::new(rows[e].as_slice(), device)?;
            let sel = xs_flat.index_select(&idx, 0)?;
            let out = expert.forward(&sel)?;
            let w = Tensor::new(wts[e].as_slice(), device)?
                .reshape(((), 1))?
                .to_dtype(xs.dtype())?;
            ys = ys.index_add(&idx, &out.broadcast_mul(&w)?, 0)?;
        }

        ys.reshape((b, t, hidden_dim))
    }
}

/// Per-layer MLP: sparse MoE (every Qwen3-Omni thinker layer) or dense SwiGLU (`mlp_only_layers`).
enum Mlp {
    Moe(MoeMlp),
    Dense(SwiGluMlp),
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Moe(m) => m.forward(xs),
            Self::Dense(m) => m.forward(xs),
        }
    }
}

/// One Qwen3 decoder layer: pre-norm attention + pre-norm MLP, both residual.
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vb: ShardedVarBuilder,
        rope: Arc<RotaryEmbedding>,
        mrope: Arc<Qwen3VLRotaryEmbedding>,
        mlp: Mlp,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        qcfg: &Option<QuantizedConfig>,
        attention_mechanism: AttentionImplementation,
        device: &Device,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            vb.pp("self_attn"),
            rope,
            mrope,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps,
            qcfg,
            attention_mechanism,
            device,
        )?;
        let input_layernorm = RmsNorm::new(hidden_size, rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            RmsNorm::new(hidden_size, rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, seqlen_offsets, mask)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }

    /// Cache-aware decoder layer for serving: pre-norm cached attention + pre-norm MLP, both
    /// residual. Mirrors [`Self::forward`] but threads the layer's [`KvCache`].
    #[allow(clippy::too_many_arguments)]
    fn forward_cached(
        &self,
        xs: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
        ctx: &ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs =
            self.self_attn
                .forward_cached(&xs, seqlen_offsets, mask, kv_cache, ctx, layer_idx)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }

    /// Cache-aware decoder layer for the multimodal serving path: pre-norm mRoPE attention + pre-norm
    /// MLP, both residual. Mirrors [`Self::forward_cached`] but applies interleaved 3D mRoPE.
    #[allow(clippy::too_many_arguments)]
    fn forward_cached_mrope(
        &self,
        xs: &Tensor,
        cos_sin: &(Tensor, Tensor),
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
        ctx: &ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_cached_mrope(&xs, cos_sin, mask, kv_cache, ctx, layer_idx)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }

    /// Cacheless decoder layer with interleaved 3D mRoPE — the model-level reference for vision.
    /// Mirrors [`Self::forward`] but applies mRoPE instead of 1D RoPE (no cache).
    fn forward_mrope(
        &self,
        xs: &Tensor,
        cos_sin: &(Tensor, Tensor),
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_mrope(&xs, cos_sin, mask)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }
}

/// The Qwen3-Omni Thinker text decoder. Owns `embed_tokens`, the decoder stack, the final `norm`,
/// and the untied `lm_head`.
pub struct OmniThinkerText {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    /// Interleaved 3D mRoPE shared by every layer for the multimodal serving path.
    mrope: Arc<Qwen3VLRotaryEmbedding>,
}

impl OmniThinkerText {
    /// `vb` is the `thinker.*` namespace: the backbone lives under `thinker.model.*` and the head at
    /// `thinker.lm_head` (a sibling of `thinker.model`), so embeddings/layers/norm load from
    /// `vb.pp("model")` and the head from `vb.pp("lm_head")`.
    pub fn new(
        cfg: &OmniTextConfig,
        vb: ShardedVarBuilder,
        device: &Device,
        comm: &Arc<hanzo_quant::Comm>,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        // `comm` is part of the loader contract (tensor-parallel construction) but unused here:
        // experts are dense per-rank, single-process.
        let _ = comm;
        let head_dim = cfg.head_dim;
        // Text-only mRoPE == 1D RoPE; `is_gpt_neox = true` selects the rotate-half (NeoX) layout
        // that HF Qwen3 uses.
        let rope = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            head_dim,
            cfg.max_position_embeddings,
            device,
            true,
            vb.dtype(),
        )?);
        // Interleaved 3D mRoPE (mrope_section [24,20,20]) for multimodal serving. Default to the
        // text-collapse partition [hd/2,0,0] when the checkpoint omits rope_scaling so this never
        // panics; vision serving requires the real section, which the published config provides.
        let mrope_section = {
            let s = cfg.mrope_section();
            if s.iter().sum::<usize>() == head_dim / 2 {
                s
            } else {
                vec![head_dim / 2, 0, 0]
            }
        };
        let mrope = Arc::new(Qwen3VLRotaryEmbedding::new(
            cfg.rope_theta as f32,
            head_dim,
            device,
            mrope_section,
        )?);

        let vb_model = vb.pp("model");
        let embed_tokens = layers::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_model.pp("embed_tokens"),
            &None,
        )?;

        // In-checkpoint quantization (FP8/GPTQ) for the quantizable linears; `None` for a full-
        // precision checkpoint (runtime ISQ then applies on top). Shared by every layer + lm_head.
        let qcfg = &cfg.quantization_config;

        let vb_l = vb_model.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_vb = vb_l.pp(i);
            let mlp = if cfg.is_moe_layer(i) {
                Mlp::Moe(MoeMlp::new(cfg, layer_vb.pp("mlp"))?)
            } else {
                Mlp::Dense(SwiGluMlp::new(
                    layer_vb.pp("mlp"),
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    qcfg,
                )?)
            };
            layers.push(DecoderLayer::new(
                layer_vb,
                rope.clone(),
                mrope.clone(),
                mlp,
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                head_dim,
                cfg.rms_norm_eps,
                qcfg,
                attention_mechanism,
                device,
            )?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;
        // `lm_head` is a quantizable linear (untied; `thinker.lm_head.weight`). The exposed ISQ /
        // pre-quant path covers it like the canonical text models.
        let lm_head = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.vocab_size,
            qcfg,
            false,
            vb.pp("lm_head"),
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            mrope,
        })
    }

    /// Embed `input_ids` (`[batch, seq]` token ids) into the residual stream `[batch, seq, hidden]`
    /// (this is `hidden_states[0]`). Exposed so multimodal fusion can replace placeholder-token rows
    /// with encoder outputs before the decoder runs (see [`super::modality::fuse_modalities`]).
    pub fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(ids)
    }

    /// Run the decoder over pre-computed input embeddings `[batch, seq, hidden]`.
    ///
    /// Returns `(logits, hidden_states)` where `logits` is `[batch, seq, vocab]` and
    /// `hidden_states` has length `num_hidden_layers + 1`: `hidden_states[0]` is `inputs_embeds` and
    /// `hidden_states[i]` is the residual stream after the first `i` decoder layers (HF
    /// `output_hidden_states` indexing). `mask` is the additive causal mask broadcastable to
    /// `[batch, heads, seq, seq]`; `seqlen_offsets` carries the RoPE position offset per batch row.
    pub fn forward_embeds(
        &self,
        inputs_embeds: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut xs = inputs_embeds.clone();

        let mut hidden_states = Vec::with_capacity(self.layers.len() + 1);
        hidden_states.push(xs.clone());

        for layer in &self.layers {
            xs = layer.forward(&xs, seqlen_offsets, mask)?;
            hidden_states.push(xs.clone());
        }

        let hidden = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&hidden)?;
        Ok((logits, hidden_states))
    }

    /// Cacheless decoder over `inputs_embeds` `[batch, seq, hidden]` using **interleaved 3D mRoPE**
    /// positions `position_ids` `[3, batch, seq]` (temporal/height/width) — the model-level reference
    /// for vision. Mirrors [`Self::forward_embeds`] exactly (same per-layer attention/MoE, same
    /// `hidden_states` indexing) but applies mRoPE rather than 1D RoPE. For text/audio the three axes
    /// carry equal positions, so this collapses to [`Self::forward_embeds`] numerically.
    pub fn forward_embeds_mrope(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let cos_sin = self
            .mrope
            .compute_cos_sin(position_ids, inputs_embeds.dtype())?;
        let mut xs = inputs_embeds.clone();

        let mut hidden_states = Vec::with_capacity(self.layers.len() + 1);
        hidden_states.push(xs.clone());

        for layer in &self.layers {
            xs = layer.forward_mrope(&xs, &cos_sin, mask)?;
            hidden_states.push(xs.clone());
        }

        let hidden = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&hidden)?;
        Ok((logits, hidden_states))
    }

    /// Run the decoder over `input_ids` (`[batch, seq]` token ids): [`Self::embed_tokens`] followed
    /// by [`Self::forward_embeds`]. See [`Self::forward_embeds`] for the return contract.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let inputs_embeds = self.embed_tokens(input_ids)?;
        self.forward_embeds(&inputs_embeds, seqlen_offsets, mask)
    }

    /// Cache-aware serving forward over pre-computed `inputs_embeds` `[batch, seq, hidden]`: runs the
    /// decoder with the engine KV cache and returns the text `logits` for the rows selected by `ctx`
    /// (last token on decode). Mirrors the validated [`Self::forward_embeds`] math (the same
    /// per-layer attention/MoE) but threads one [`KvCache`] per decoder layer so decode reuses past
    /// K/V instead of recomputing — use [`Self::forward_embeds`] for the cacheless validation path.
    /// `mask` is `None` for single-token decode; `cache.len()` must equal the decoder depth.
    pub fn forward_cached(
        &self,
        inputs_embeds: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
        cache: &mut [KvCache],
        ctx: &ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut xs = inputs_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_cached(&xs, seqlen_offsets, mask, &mut cache[i], ctx, i)?;
        }
        let xs = self.norm.forward(&xs)?;
        // Select the rows we actually need logits for (decode = last token) before the lm_head.
        let xs = ctx.logits(&xs)?;
        self.lm_head.forward(&xs)
    }

    /// Cache-aware serving forward using **interleaved 3D mRoPE** — the multimodal (vision) path.
    /// Identical to [`Self::forward_cached`] except positions come from `position_ids` `[3, batch,
    /// seq]` (temporal/height/width) instead of the 1D `seqlen_offsets`. `(cos, sin)` are computed
    /// once and shared across layers. For text/audio inputs `position_ids` carry the same value on
    /// all three axes, so this reduces to the 1D path; image/video inputs genuinely use 3D.
    pub fn forward_cached_mrope(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut [KvCache],
        ctx: &ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let cos_sin = self
            .mrope
            .compute_cos_sin(position_ids, inputs_embeds.dtype())?;
        let mut xs = inputs_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_cached_mrope(&xs, &cos_sin, mask, &mut cache[i], ctx, i)?;
        }
        let xs = self.norm.forward(&xs)?;
        let xs = ctx.logits(&xs)?;
        self.lm_head.forward(&xs)
    }

    /// Every in-situ-quantizable linear in the Thinker, paired with its layer index (`None` for the
    /// non-layer `lm_head`), in a stable order. Drives [`IsqModel::get_layers`] for the whole Omni
    /// model: the loader quantizes exactly these (attention q/k/v/o, each MoE expert's gate/up/down,
    /// any dense-layer gate/up/down, and the head). The router `gate`, q/k norms, layernorms, final
    /// norm and embeddings are intentionally absent — they stay full precision (see
    /// [`Self::residual_tensors`]). The order mirrors the loader's ISQ regexes so an imatrix pairs up.
    pub fn get_isq_layers(&mut self) -> Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)> {
        let mut layers: Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)> = Vec::new();
        layers.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layers.push((&mut layer.self_attn.q_proj, Some(i)));
            layers.push((&mut layer.self_attn.k_proj, Some(i)));
            layers.push((&mut layer.self_attn.v_proj, Some(i)));
            layers.push((&mut layer.self_attn.o_proj, Some(i)));
            match &mut layer.mlp {
                Mlp::Moe(moe) => {
                    for expert in moe.experts.iter_mut() {
                        layers.push((&mut expert.gate_proj, Some(i)));
                        layers.push((&mut expert.up_proj, Some(i)));
                        layers.push((&mut expert.down_proj, Some(i)));
                    }
                }
                Mlp::Dense(mlp) => {
                    layers.push((&mut mlp.gate_proj, Some(i)));
                    layers.push((&mut mlp.up_proj, Some(i)));
                    layers.push((&mut mlp.down_proj, Some(i)));
                }
            }
        }
        layers
    }

    /// The full-precision Thinker tensors that ISQ never quantizes — embeddings, the final norm,
    /// every layernorm + q/k norm, and each MoE router `gate`. Keys are relative to the `thinker`
    /// namespace (e.g. `model.norm.weight`); the model prefixes `thinker.` (see
    /// [`super::Qwen3OmniModel`]'s `residual_tensors`). Counterpart to [`Self::get_isq_layers`] for
    /// UQFF serialization of the Thinker.
    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);
        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(i);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            uvb_l
                .pp("self_attn")
                .pp("q_norm")
                .add(&layer.self_attn.q_norm);
            uvb_l
                .pp("self_attn")
                .pp("k_norm")
                .add(&layer.self_attn.k_norm);
            if let Mlp::Moe(moe) = &layer.mlp {
                uvb_l.pp("mlp").pp("gate").add(&moe.gate);
            }
        }
        uvb.to_safetensors()
    }
}

#[cfg(test)]
mod thinker_tests {
    use super::*;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use hanzo_ml::{DType, Device, IndexOp, Tensor};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use super::super::config::Qwen3OmniConfig;

    fn read_f32_le(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb = 0f64;
        for (x, y) in a.iter().zip(b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    /// Loads the REAL zen-omni-30b thinker weights and asserts the greedy next token for the fixed
    /// prompt is 9707, plus cosine > 0.99 against the captured HF logits fixture. Env-gated on the
    /// weights dir so CI without the checkpoint skips cleanly.
    #[test]
    fn omni_thinker_greedy_matches_reference() {
        let dir = std::env::var("ZEN_OMNI_DIR")
            .unwrap_or_else(|_| "/home/z/work/zen/hf/zen-omni-30b-instruct".to_string());
        let dirp = PathBuf::from(&dir);
        let index = dirp.join("model.safetensors.index.json");
        if !index.is_file() {
            eprintln!("zen-omni weights absent ({index:?}); skipping thinker validation");
            return;
        }

        // CPU matmul (hanzo-ml) rejects BF16 and an f32 thinker (~122GB) will not fit in RAM, so the
        // CPU path uses F16; with the `cuda` feature we run BF16 on-device (matches the reference).
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F16
        };
        eprintln!("[thinker] device={device:?} dtype={dtype:?}");

        let cfg: Qwen3OmniConfig =
            serde_json::from_str(&std::fs::read_to_string(dirp.join("config.json")).unwrap())
                .unwrap();
        let tc = &cfg.thinker_config.text_config;

        // Collect every shard referenced by the index.
        let index_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index).unwrap()).unwrap();
        let mut shard_set = std::collections::BTreeSet::new();
        for v in index_json["weight_map"].as_object().unwrap().values() {
            shard_set.insert(v.as_str().unwrap().to_string());
        }
        let paths: Vec<PathBuf> = shard_set.iter().map(|s| dirp.join(s)).collect();
        eprintln!("[thinker] loading {} shards from {dir}", paths.len());

        // Single-process Comm (rank 0 / world 1) — no tensor parallelism.
        let comm = Arc::new(
            hanzo_quant::Comm::from_device(hanzo_quant::Id::new(), &device, 0, 1).unwrap(),
        );

        // Only materialize the thinker tensors.
        let vb = from_mmaped_safetensors(
            paths,
            Vec::new(),
            Some(dtype),
            &device,
            vec![None],
            true,
            None,
            |name: String| name.starts_with("thinker."),
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap();

        // Cacheless reference path under test never touches paged attention; build it `Eager`.
        let model = OmniThinkerText::new(
            tc,
            vb.pp("thinker"),
            &device,
            &comm,
            AttentionImplementation::Eager,
        )
        .unwrap();

        let ids: Vec<u32> = vec![
            151644, 872, 198, 9707, 11, 1879, 0, 151645, 198, 151644, 77091, 198,
        ];
        let t = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, t), &device).unwrap();

        // Additive causal mask [1, 1, t, t]: 0 on/under the diagonal, -inf above.
        let mut maskv = vec![0f32; t * t];
        for i in 0..t {
            for (j, m) in maskv[i * t..(i + 1) * t].iter_mut().enumerate() {
                if j > i {
                    *m = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(maskv, (1, 1, t, t), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let (logits, hidden_states) = model.forward(&input_ids, &[0], Some(&mask)).unwrap();
        eprintln!(
            "[thinker] logits {:?}, hidden_states len {}",
            logits.dims(),
            hidden_states.len()
        );

        let last = logits
            .i((0, t - 1))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got_argmax = argmax(&last);
        eprintln!("[thinker] greedy next-token argmax = {got_argmax} (expected 9707)");

        // Logits fixture: 152064 f32 LE for the last position.
        let logit_ref = "/home/z/work/zen/hf/omni_fixtures/thinker_logits.f32";
        let mut logit_cos = f32::NAN;
        if Path::new(logit_ref).is_file() {
            let refv = read_f32_le(logit_ref);
            assert_eq!(
                refv.len(),
                last.len(),
                "logits len {} vs ref {}",
                last.len(),
                refv.len()
            );
            logit_cos = cosine(&last, &refv);
            eprintln!(
                "[thinker] logits cosine = {logit_cos:.6}, ref argmax = {}",
                argmax(&refv)
            );
        } else {
            eprintln!("[thinker] logits fixture absent ({logit_ref})");
        }

        // Hidden-state fixture: layer `accept_hidden_layer` (= 24) residual stream, [t, hidden] f32.
        let hidden_ref = "/home/z/work/zen/hf/omni_fixtures/thinker_hidden.f32";
        if Path::new(hidden_ref).is_file() {
            let refv = read_f32_le(hidden_ref);
            for idx in [23usize, 24, 25] {
                if idx < hidden_states.len() {
                    let got: Vec<f32> = hidden_states[idx]
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    if got.len() == refv.len() {
                        eprintln!(
                            "[thinker] hidden_states[{idx}] cosine vs L24 fixture = {:.6}",
                            cosine(&got, &refv)
                        );
                    }
                }
            }
        }

        // Hard requirements.
        assert_eq!(got_argmax, 9707, "thinker greedy next-token != 9707");
        if logit_cos.is_finite() {
            assert!(logit_cos > 0.99, "logits cosine {logit_cos} <= 0.99");
        }
    }

    use super::super::config::OmniTextConfig;
    use hanzo_ml::Shape;

    /// Toy Thinker text config sized so every quantizable in-dim (hidden = 256, moe_intermediate =
    /// 256) is a multiple of the K-quant super-block (256) — lets ISQ Q*K apply to every selected
    /// linear. 2 layers × 4 experts mirrors the real 48 × 128 topology at micro scale.
    fn toy_text_config() -> OmniTextConfig {
        OmniTextConfig {
            vocab_size: 512,
            hidden_size: 256,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 64,
            hidden_act: crate::layers::Activation::Silu,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            moe_intermediate_size: 256,
            shared_expert_intermediate_size: 0,
            num_experts: 4,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            mlp_only_layers: Vec::new(),
            decoder_sparse_step: 1,
            use_qk_norm: true,
            tie_word_embeddings: false,
            rope_scaling: None,
            quantization_config: None,
        }
    }

    /// A `SimpleBackend` returning a small deterministic pattern tensor for any requested name/shape,
    /// so the Thinker graph (and ISQ) can be built with no checkpoint on disk. The fill mirrors the
    /// other Omni tests' `(i % 17) * 0.01 - 0.08`: small, finite, index-varying so MoE routing and
    /// argmax are non-degenerate.
    struct PatternBackend;
    impl hanzo_nn::var_builder::SimpleBackend for PatternBackend {
        fn get(
            &self,
            s: Shape,
            _name: &str,
            _h: hanzo_nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            let n = s.elem_count();
            let data: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.01 - 0.08).collect();
            Tensor::from_vec(data, s, dev)?.to_dtype(dtype)
        }
        fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("PatternBackend requires a shape; use `get`")
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    /// One no-checkpoint ISQ round-trip for a single quant type: build the toy Thinker, assert
    /// `get_isq_layers` exposes exactly the quantizable linears, in-situ quantize each, and forward.
    /// Returns `(selected_layer_count, greedy_token)`. Shared by the per-type structural test.
    fn isq_roundtrip(ty: hanzo_quant::IsqType) -> (usize, usize) {
        let device = Device::Cpu;
        let cfg = toy_text_config();
        let comm = Arc::new(
            hanzo_quant::Comm::from_device(hanzo_quant::Id::new(), &device, 0, 1).unwrap(),
        );
        let vb = hanzo_quant::ShardedSafeTensors::wrap(
            Box::new(PatternBackend),
            DType::F32,
            device.clone(),
        );

        let mut model = OmniThinkerText::new(
            &cfg,
            vb.pp("thinker"),
            &device,
            &comm,
            AttentionImplementation::Eager,
        )
        .unwrap();

        // SELECTION: lm_head + per layer (q,k,v,o = 4) + (num_experts × gate/up/down = 12), exactly.
        let expected = 1 + cfg.num_hidden_layers * (4 + cfg.num_experts * 3);
        {
            let layers = model.get_isq_layers();
            assert_eq!(layers.len(), expected, "ISQ layer count != expected");
            assert!(
                layers.iter().all(|(l, _)| l.name() == "unquant-linear"),
                "selected linears must start unquantized"
            );
        }

        // APPLY: synchronous in-situ quant (no thread pool / pending layers). Every GGML Q/K quant
        // produces a `gguf` layer; `n_quantized` must count all selected linears.
        let n_quantized = std::sync::atomic::AtomicUsize::new(0);
        let guard = hanzo_quant::QuantizeOntoGuard::new();
        {
            let layers = model.get_isq_layers();
            for (layer, _idx) in layers {
                let quant = layer
                    .clone()
                    .apply_isq(Some(ty), device.clone(), &n_quantized, None, guard.clone())
                    .unwrap();
                assert_eq!(
                    quant.name(),
                    "gguf",
                    "{ty:?} layer did not become a gguf layer"
                );
                *layer = quant;
            }
        }
        assert_eq!(
            n_quantized.load(std::sync::atomic::Ordering::Relaxed),
            expected,
            "n_quantized != selected layer count"
        );

        // FORWARD: quantized decoder → finite logits of the right shape.
        let ids: Vec<u32> = vec![1, 5, 9, 2, 7, 3];
        let t = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, t), &device).unwrap();
        let mut maskv = vec![0f32; t * t];
        for i in 0..t {
            for (j, m) in maskv[i * t..(i + 1) * t].iter_mut().enumerate() {
                if j > i {
                    *m = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(maskv, (1, 1, t, t), &device).unwrap();

        let (logits, hidden_states) = model.forward(&input_ids, &[0], Some(&mask)).unwrap();
        assert_eq!(logits.dims(), [1, t, cfg.vocab_size], "logits shape");
        assert_eq!(hidden_states.len(), cfg.num_hidden_layers + 1);
        let last = logits
            .i((0, t - 1))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            last.iter().all(|x| x.is_finite()),
            "{ty:?} logits must be finite"
        );
        let am = argmax(&last);
        assert!(am < cfg.vocab_size, "argmax {am} out of vocab range");
        (expected, am)
    }

    /// STRUCTURAL ISQ proof (no checkpoint) across the headline ISQ types (Q4K / Q2K / Q8_0). For each
    /// quant it guards the three things Omni ISQ depends on: SELECTION (`get_isq_layers` exposes exactly
    /// the quantizable linears — `lm_head` + per-layer attention q/k/v/o + every expert's gate/up/down),
    /// APPLY (each becomes a `gguf` layer; `n_quantized` counts all), and FORWARD (the quantized decoder
    /// still produces finite logits of the right shape). The exact SELECTION count also pins the loader
    /// regexes in sync (see `qwen3_omni_isq_regexes_scope_to_thinker_text`).
    #[test]
    fn omni_thinker_isq_selection_and_forward() {
        for ty in [
            hanzo_quant::IsqType::Q4K,
            hanzo_quant::IsqType::Q2K,
            hanzo_quant::IsqType::Q8_0,
        ] {
            let (n, greedy) = isq_roundtrip(ty);
            eprintln!("[isq-struct] {ty:?}: quantized {n} linears; finite logits; greedy={greedy}");
        }
    }

    /// REAL-weights ISQ validation (env-gated on `ZEN_OMNI_DIR`, like the other Omni tests; skips
    /// cleanly when the checkpoint is absent). Loads only the Thinker text decoder (`thinker.model.*`
    /// and `thinker.lm_head`), quantizes it to Q4K through the production *immediate*-ISQ path (the exact
    /// mechanism `hanzo run --isq Q4K` uses), and asserts a forward on the fixed prompt yields finite
    /// logits with a greedy token in range. Quant drift is expected, so 9707 is logged, not required.
    ///
    /// HEAVY (~30B params): this materializes the text decoder. Run with `earlyoom` stopped. The
    /// predicates below mirror `Qwen3OmniLoader::isq_layer_regexes` (kept in sync by the structural
    /// test's exact-count selection check).
    #[test]
    fn omni_thinker_isq_real_q4k_forward() {
        // Explicit opt-in: this materializes ~30B params, so it must never fire on a bare `cargo test`
        // (which could OOM a co-resident service). Run with `ZEN_OMNI_ISQ_REAL=1` and `earlyoom`
        // stopped; `ISQ_SINGLETHREAD=1` bounds the immediate-ISQ transient to one linear.
        if std::env::var("ZEN_OMNI_ISQ_REAL").is_err() {
            eprintln!("[isq-real] set ZEN_OMNI_ISQ_REAL=1 to run the real-weights Q4K validation; skipping");
            return;
        }
        let dir = std::env::var("ZEN_OMNI_DIR")
            .unwrap_or_else(|_| "/home/z/work/zen/hf/zen-omni-30b-instruct".to_string());
        let dirp = PathBuf::from(&dir);
        let index = dirp.join("model.safetensors.index.json");
        if !index.is_file() {
            eprintln!("[isq-real] zen-omni weights absent ({index:?}); skipping");
            return;
        }

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F16
        };
        eprintln!("[isq-real] device={device:?} dtype={dtype:?}");

        let cfg: Qwen3OmniConfig =
            serde_json::from_str(&std::fs::read_to_string(dirp.join("config.json")).unwrap())
                .unwrap();
        let tc = &cfg.thinker_config.text_config;

        let index_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index).unwrap()).unwrap();
        let mut shard_set = std::collections::BTreeSet::new();
        for v in index_json["weight_map"].as_object().unwrap().values() {
            shard_set.insert(v.as_str().unwrap().to_string());
        }
        let paths: Vec<PathBuf> = shard_set.iter().map(|s| dirp.join(s)).collect();

        let comm = Arc::new(
            hanzo_quant::Comm::from_device(hanzo_quant::Id::new(), &device, 0, 1).unwrap(),
        );

        // Production immediate-ISQ predicates, Thinker-scoped (mirror the loader). The router `gate`
        // and the audio/vision towers are deliberately excluded.
        let predicates: Vec<regex::Regex> = [
            r"thinker\.lm_head\.(weight|bias)$",
            r"thinker\.model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            r"thinker\.model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            r"thinker\.model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            r"thinker\.model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            r"thinker\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$",
            r"thinker\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$",
            r"thinker\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$",
        ]
        .iter()
        .map(|p| regex::Regex::new(p).unwrap())
        .collect();
        hanzo_quant::set_immediate_isq(Some(hanzo_quant::IsqType::Q4K), predicates);

        // Only the text decoder is needed for the Thinker forward; skip the towers/talker/code2wav to
        // keep the load light. The immediate-ISQ path quantizes the matched linears as they load.
        let vb = from_mmaped_safetensors(
            paths,
            Vec::new(),
            Some(dtype),
            &device,
            vec![None],
            true,
            None,
            |name: String| {
                name.starts_with("thinker.model.") || name.starts_with("thinker.lm_head")
            },
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap();

        let model = OmniThinkerText::new(
            tc,
            vb.pp("thinker"),
            &device,
            &comm,
            AttentionImplementation::Eager,
        )
        .unwrap();
        hanzo_quant::clear_immediate_isq();

        let ids: Vec<u32> = vec![
            151644, 872, 198, 9707, 11, 1879, 0, 151645, 198, 151644, 77091, 198,
        ];
        let t = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, t), &device).unwrap();
        let mut maskv = vec![0f32; t * t];
        for i in 0..t {
            for (j, m) in maskv[i * t..(i + 1) * t].iter_mut().enumerate() {
                if j > i {
                    *m = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(maskv, (1, 1, t, t), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let (logits, _hidden) = model.forward(&input_ids, &[0], Some(&mask)).unwrap();
        let last = logits
            .i((0, t - 1))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let am = argmax(&last);
        eprintln!(
            "[isq-real] Q4K greedy next-token = {am} (f16 reference = 9707); logits {:?}",
            logits.dims()
        );
        if let Ok(refv) =
            std::fs::read("/home/z/work/zen/hf/omni_fixtures/thinker_logits.f32").map(|b| {
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect::<Vec<f32>>()
            })
        {
            if refv.len() == last.len() {
                eprintln!(
                    "[isq-real] Q4K-vs-f16 logits cosine = {:.4}",
                    cosine(&last, &refv)
                );
            }
        }

        assert!(
            last.iter().all(|x| x.is_finite()),
            "Q4K logits must be finite"
        );
        assert!(am < tc.vocab_size, "argmax {am} out of vocab range");
    }
}
