#![allow(dead_code)]

//! Qwen3-Omni-MoE **Talker** (`talker.*`): the autoregressive speech decoder that renders the
//! Thinker's hidden-state stream into neural-codec codes.
//!
//! Structurally this is the Qwen3-TTS talker (see `speech_models::qwen3_tts::model`) with one
//! difference: every decoder layer's MLP is a **Qwen3-MoE block with a shared expert** instead of a
//! dense SwiGLU. The MoE block routes through [`MoEExperts`] and adds a sigmoid-gated shared expert
//! exactly as HF Qwen3-MoE does:
//!
//! ```text
//! out = moe_experts(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)
//! ```
//!
//! The [`OmniCodePredictor`] (the MTP head, `talker.code_predictor.*`) is the *dense* Qwen3 stack —
//! identical to the qwen3_tts code predictor — that predicts codebooks 1..num_code_groups from the
//! group-0 hidden state.
//!
//! As in the speech path, the interleaved 3D mRoPE collapses to a plain 1D [`RotaryEmbedding`] here,
//! so a standard rotary embedding is exact.

use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{Embedding, Linear, Module};
use hanzo_quant::ShardedVarBuilder;

use crate::{
    attention::{naive_sdpa, SdpaParams},
    layers::{self, repeat_kv, RmsNorm, RotaryEmbedding},
    ops::{moe_router_topk, MoeRouterScoreFunction, MoeRouterSelectedWeight, MoeRouterTopKConfig},
};

use super::config::{CodePredictorConfig, TalkerConfig, TalkerTextConfig};

/// Qwen3 attention (GQA + per-head q/k RMSNorm + 1D RoPE via `naive_sdpa`). Shared by the talker
/// backbone and the code predictor; both carry `self_attn.{q,k,v,o}_proj` (no bias) and
/// `self_attn.{q,k}_norm`.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: Arc<RotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        vb: ShardedVarBuilder,
        rope: Arc<RotaryEmbedding>,
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
}

/// Dense SwiGLU MLP: `down(silu(gate(x)) * up(x))`. Used for the shared expert and for every code
/// predictor layer. No bias (matches the checkpoint).
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

/// Qwen3-MoE block with a shared expert: a router `gate` + per-expert SwiGLU experts + a
/// sigmoid-gated dense shared expert. Mirrors HF Qwen3-MoE semantics.
///
/// The expert compute is the inlined gather → expert → weighted scatter-add loop (identical to
/// `super::thinker::MoeMlp`), not the [`crate::moe::MoEExperts`] wrapper: that wrapper's loader
/// requires the combined `gate_up_proj` layout while this checkpoint is per-expert
/// (`experts.{i}.{gate,up,down}_proj`), and its fast gather kernel is CUDA-only. The inline path is
/// device-agnostic and exact.
struct MoeMlp {
    gate: Linear,
    experts: Vec<SwiGluMlp>,
    shared_expert: SwiGluMlp,
    shared_expert_gate: Linear,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl MoeMlp {
    fn new(
        cfg: &TalkerTextConfig,
        vb: ShardedVarBuilder,
        device: &Device,
        comm: &Arc<hanzo_quant::Comm>,
    ) -> Result<Self> {
        // Single-process, dense per-rank experts: tensor-parallel plumbing is unused here.
        let _ = (device, comm);
        let gate = layers::linear_no_bias(cfg.hidden_size, cfg.num_experts, vb.pp("gate"))?;
        let vb_e = vb.pp("experts");
        let experts = (0..cfg.num_experts)
            .map(|i| SwiGluMlp::new(vb_e.pp(i), cfg.hidden_size, cfg.moe_intermediate_size))
            .collect::<Result<Vec<_>>>()?;
        let shared_expert = SwiGluMlp::new(
            vb.pp("shared_expert"),
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
        )?;
        let shared_expert_gate =
            layers::linear_no_bias(cfg.hidden_size, 1, vb.pp("shared_expert_gate"))?;

        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
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

        // Bucket the (token, weight) assignments by expert, then run each engaged expert over its
        // routed tokens and weighted scatter-add into the output (identical to the thinker path).
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
        let routed = ys.reshape((b, t, hidden_dim))?;

        // shared_expert_gate: Linear(hidden -> 1); sigmoid-gate the dense shared expert output.
        let shared = self.shared_expert.forward(xs)?;
        let gate = hanzo_nn::ops::sigmoid(&self.shared_expert_gate.forward(xs)?)?;
        let shared = shared.broadcast_mul(&gate)?;

        routed + shared
    }
}

/// Per-layer MLP: MoE (talker backbone) or dense SwiGLU (code predictor).
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
}

/// `Qwen3OmniMoeTalkerResizeMLP`: `fc2(silu(fc1(x)))`, both linears with bias. Resizes the Thinker
/// hidden width (thinker_hidden_size) and Thinker text-token embeddings into the talker hidden width.
struct ResizeMlp {
    fc1: Linear,
    fc2: Linear,
}

impl ResizeMlp {
    fn new(
        vb: ShardedVarBuilder,
        input_size: usize,
        intermediate_size: usize,
        output_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            fc1: layers::linear(input_size, intermediate_size, vb.pp("linear_fc1"))?,
            fc2: layers::linear(intermediate_size, output_size, vb.pp("linear_fc2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(xs)?.silu()?)
    }
}

/// The autoregressive Qwen3-Omni talker backbone. Consumes pre-summed input embeddings (projected
/// Thinker hidden + projected Thinker text embeddings + codec embeddings) and emits hidden states +
/// codebook-0 logits per frame.
pub struct OmniTalker {
    codec_embed: Embedding,
    hidden_projection: ResizeMlp,
    text_projection: ResizeMlp,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    codec_head: Linear,
}

impl OmniTalker {
    /// `vb` is the `talker.*` namespace: the backbone lives under `talker.model.*`; the head and
    /// bridges (`codec_head`, `hidden_projection`, `text_projection`) live directly under `talker.*`.
    pub fn new(
        cfg: &TalkerConfig,
        vb: ShardedVarBuilder,
        device: &Device,
        comm: &Arc<hanzo_quant::Comm>,
    ) -> Result<Self> {
        let tc = &cfg.text_config;
        let head_dim = tc.head_dim;
        let rope = Arc::new(RotaryEmbedding::new(
            tc.rope_theta as f32,
            head_dim,
            tc.max_position_embeddings.max(32768),
            device,
            true,
            vb.dtype(),
        )?);

        let vb_model = vb.pp("model");
        let codec_embed = layers::embedding(
            tc.vocab_size,
            tc.hidden_size,
            vb_model.pp("codec_embedding"),
            &None,
        )?;

        let vb_l = vb_model.pp("layers");
        let mut layers = Vec::with_capacity(tc.num_hidden_layers);
        for i in 0..tc.num_hidden_layers {
            let layer_vb = vb_l.pp(i);
            let mlp = Mlp::Moe(MoeMlp::new(tc, layer_vb.pp("mlp"), device, comm)?);
            layers.push(DecoderLayer::new(
                layer_vb,
                rope.clone(),
                mlp,
                tc.hidden_size,
                tc.num_attention_heads,
                tc.num_key_value_heads,
                head_dim,
                tc.rms_norm_eps,
            )?);
        }

        let norm = RmsNorm::new(tc.hidden_size, tc.rms_norm_eps, vb_model.pp("norm"))?;
        let codec_head =
            layers::linear_no_bias(tc.hidden_size, tc.vocab_size, vb.pp("codec_head"))?;

        let hidden_projection = ResizeMlp::new(
            vb.pp("hidden_projection"),
            cfg.thinker_hidden_size,
            cfg.thinker_hidden_size,
            tc.hidden_size,
        )?;
        let text_projection = ResizeMlp::new(
            vb.pp("text_projection"),
            cfg.thinker_hidden_size,
            cfg.thinker_hidden_size,
            tc.hidden_size,
        )?;

        Ok(Self {
            codec_embed,
            hidden_projection,
            text_projection,
            layers,
            norm,
            codec_head,
        })
    }

    /// Codec-token ids -> talker hidden space.
    pub fn embed_codec(&self, ids: &Tensor) -> Result<Tensor> {
        self.codec_embed.forward(ids)
    }

    /// Project a Thinker hidden state (thinker_hidden_size) into the talker hidden space.
    pub fn project_thinker_hidden(&self, h: &Tensor) -> Result<Tensor> {
        self.hidden_projection.forward(h)
    }

    /// Project Thinker text-token embeddings (thinker_hidden_size) into the talker hidden space.
    pub fn project_text(&self, h: &Tensor) -> Result<Tensor> {
        self.text_projection.forward(h)
    }

    /// Run the talker decoder over pre-summed `inputs_embeds`; returns `(hidden, codec0_logits)`.
    ///
    /// `hidden` is the **post-`norm`** final hidden state (`last_hidden_state`). This is exactly HF's
    /// `outputs.hidden_states[-1]`: the `@capture_outputs` recorder runs with
    /// `tie_last_hidden_states=True`, which overwrites the last captured (pre-norm) entry with
    /// `last_hidden_state`. The talker generation loop feeds this post-norm hidden to the code
    /// predictor as `past_hidden`, and `codec0_logits = codec_head(hidden)` come from the same tensor.
    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let mut xs = inputs_embeds.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, seqlen_offsets, mask)?;
        }
        let hidden = self.norm.forward(&xs)?;
        let logits = self.codec_head.forward(&hidden)?;
        Ok((hidden, logits))
    }
}

/// Multi-token-prediction sub-talker (`talker.code_predictor.*`). Given the talker hidden state for
/// the current frame (group 0) plus the embedding of the just-predicted code, autoregressively
/// predicts codebooks 1..num_code_groups as a single attention sequence. Per-group input embeddings
/// (`model.codec_embedding.{g}`) and per-group output heads (`lm_head.{g}`); a dense Qwen3 stack.
pub struct OmniCodePredictor {
    codec_embed: Vec<Embedding>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    heads: Vec<Linear>,
}

impl OmniCodePredictor {
    pub fn new(cfg: &CodePredictorConfig, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let rope = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            head_dim,
            cfg.max_position_embeddings.max(8192),
            device,
            true,
            vb.dtype(),
        )?);

        // Groups 1..num_code_groups (codebook 0 comes from the talker head): one embedding/head each.
        let n_groups = cfg.num_code_groups.saturating_sub(1);

        let vb_model = vb.pp("model");
        let vb_ce = vb_model.pp("codec_embedding");
        let codec_embed = (0..n_groups)
            .map(|i| layers::embedding(cfg.vocab_size, cfg.hidden_size, vb_ce.pp(i), &None))
            .collect::<Result<Vec<_>>>()?;

        let vb_l = vb_model.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_vb = vb_l.pp(i);
            let mlp = Mlp::Dense(SwiGluMlp::new(
                layer_vb.pp("mlp"),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?);
            layers.push(DecoderLayer::new(
                layer_vb,
                rope.clone(),
                mlp,
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                head_dim,
                cfg.rms_norm_eps,
            )?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;

        let vb_h = vb.pp("lm_head");
        let heads = (0..n_groups)
            .map(|i| layers::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_h.pp(i)))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            codec_embed,
            layers,
            norm,
            heads,
        })
    }

    /// Embedding of group-`group` code (group 1..num_code_groups, embedding index = group-1).
    pub fn embed_group(&self, group: usize, ids: &Tensor) -> Result<Tensor> {
        self.codec_embed[group - 1].forward(ids)
    }

    /// Run the sub-talker over `inputs_embeds` and return logits for group `group` from its last
    /// position (head index = group-1).
    pub fn step(
        &self,
        inputs_embeds: &Tensor,
        mask: Option<&Tensor>,
        group: usize,
    ) -> Result<Tensor> {
        let mut xs = inputs_embeds.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, &[0], mask)?;
        }
        let hidden = self.norm.forward(&xs)?;
        self.heads[group - 1].forward(&hidden)
    }
}
