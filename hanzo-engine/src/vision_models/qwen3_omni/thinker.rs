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

use std::sync::Arc;

use hanzo_ml::{Device, DType, Result, Tensor};
use hanzo_nn::{Embedding, Linear, Module};
use hanzo_quant::ShardedVarBuilder;

use crate::{
    attention::{naive_sdpa, SdpaParams},
    layers::{self, repeat_kv, Qwen3VLRotaryEmbedding, RmsNorm, RotaryEmbedding},
    ops::{moe_router_topk, MoeRouterScoreFunction, MoeRouterSelectedWeight, MoeRouterTopKConfig},
    pipeline::{KvCache, ModelForwardContext},
};

use super::config::OmniTextConfig;

/// Qwen3 attention: GQA + per-head q/k RMSNorm + 1D RoPE via `naive_sdpa`. Identical to the talker
/// backbone attention; carries `self_attn.{q,k,v,o}_proj` (no bias) and `self_attn.{q,k}_norm`.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: Arc<RotaryEmbedding>,
    /// Interleaved 3D mRoPE for the multimodal serving path (vision). For text/audio the 3D
    /// positions collapse to 1D, so `rope` (above) is used; only image/video need this.
    mrope: Arc<Qwen3VLRotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
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
    ) -> Result<Self> {
        let q_proj = layers::linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = layers::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = layers::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = layers::linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, rms_norm_eps, vb.pp("k_norm"))?;
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
    fn forward_cached(
        &self,
        xs: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
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

        // Append the RoPE'd K/V into the running cache, then attend over the full sequence
        // `[b, kv_heads, past + t, head_dim]`.
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
        let attn = attn.transpose(1, 2)?.reshape((b, t, ()))?;
        self.o_proj.forward(&attn)
    }

    /// Cache-aware attention using **interleaved 3D mRoPE** (the multimodal serving path). Identical
    /// to [`Self::forward_cached`] except the 1D `rope` is replaced by the precomputed `(cos, sin)`
    /// from [`Qwen3VLRotaryEmbedding`]; Qwen3 QK-norm + mRoPE are applied together by the validated
    /// `forward_qk_norm` (the exact rotary path `qwen3_vl`/`qwen3_vl_moe` use). For text/audio inputs
    /// these positions equal the 1D positions, so the two paths agree numerically.
    fn forward_cached_mrope(
        &self,
        xs: &Tensor,
        cos_sin: &(Tensor, Tensor),
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
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
        let attn = attn.transpose(1, 2)?.reshape((b, t, ()))?;
        self.o_proj.forward(&attn)
    }
}

/// Dense SwiGLU MLP: `down(silu(gate(x)) * up(x))`. Used only for any `mlp_only_layers` (the
/// Qwen3-Omni thinker has none, but the architecture allows them). No bias.
struct SwiGluMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGluMlp {
    fn new(vb: ShardedVarBuilder, hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        Ok(Self {
            gate_proj: layers::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: layers::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: layers::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
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
        let gate = layers::linear_no_bias(cfg.hidden_size, cfg.num_experts, vb.pp("gate"))?;
        let vb_e = vb.pp("experts");
        let experts = (0..cfg.num_experts)
            .map(|i| SwiGluMlp::new(vb_e.pp(i), cfg.hidden_size, cfg.moe_intermediate_size))
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
    fn forward_cached(
        &self,
        xs: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_cached(&xs, seqlen_offsets, mask, kv_cache)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }

    /// Cache-aware decoder layer for the multimodal serving path: pre-norm mRoPE attention + pre-norm
    /// MLP, both residual. Mirrors [`Self::forward_cached`] but applies interleaved 3D mRoPE.
    fn forward_cached_mrope(
        &self,
        xs: &Tensor,
        cos_sin: &(Tensor, Tensor),
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_cached_mrope(&xs, cos_sin, mask, kv_cache)?;
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
    lm_head: Linear,
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
        let embed_tokens =
            layers::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"), &None)?;

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
            )?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;
        let lm_head = layers::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;

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
            xs = layer.forward_cached(&xs, seqlen_offsets, mask, &mut cache[i])?;
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
        let cos_sin = self.mrope.compute_cos_sin(position_ids, inputs_embeds.dtype())?;
        let mut xs = inputs_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_cached_mrope(&xs, &cos_sin, mask, &mut cache[i])?;
        }
        let xs = self.norm.forward(&xs)?;
        let xs = ctx.logits(&xs)?;
        self.lm_head.forward(&xs)
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
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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

        let model = OmniThinkerText::new(tc, vb.pp("thinker"), &device, &comm).unwrap();

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
            assert_eq!(refv.len(), last.len(), "logits len {} vs ref {}", last.len(), refv.len());
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
}
