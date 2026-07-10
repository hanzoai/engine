#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! DeepSeek-V4 (`DeepseekV4ForCausalLM`, `model_type = "deepseek_v4"`, MIT).
//!
//! V4 keeps DeepSeek-V3's Multi-head Latent Attention (MLA) + fine-grained MoE
//! backbone but adds four mechanisms absent from V3, each ported here:
//!
//! 1. **Hyper-Connections** ([`HyperConnections`]) — the plain `x = x + f(x)`
//!    residual is replaced by an `n_hc`-stream (default 4) carrier. At the input
//!    every stream is the token embedding; each sublayer *reduces* the streams to
//!    one vector via learned, RMS-norm-gated **pre** weights, runs, then mixes
//!    its output back through learned **post** gates and a **Sinkhorn-normalized**
//!    `n_hc × n_hc` combine matrix (20 iterations). At the output the streams are
//!    reduced once more. Pure matmul/softmax — no new kernel.
//! 2. **DSA lightning indexer** — reuses [`super::dsa::DsaIndexer`] for top-`k`
//!    (512 Flash / 1024 Pro) key selection over the compressed rows.
//! 3. **Compressed KV cache** — per-layer `compress_ratios` pick the attention
//!    mode: `0` = dense full attention, `4` = DSA-indexed, `≥sliding_window`
//!    (128) = sliding-window. Realised here as the eager `NormalCache` append +
//!    additive-mask route (the cheap, correct path); the streaming softmax-pool
//!    that shrinks the cache itself is the memory-optimisation follow-on.
//! 4. **`sqrtsoftplus` MoE gate** + **o-LoRA** output projection + per-head
//!    **attention sinks** + **sliding window** + a **single 512-dim absorbed MLA
//!    latent** (`head_dim = 512`, partial RoPE on the leading `qk_rope_head_dim`
//!    = 64).
//!
//! # Numeric parity
//!
//! This is the *structural* port: it compiles, assembles, and the novel
//! mechanisms (Hyper-Connections, the `sqrtsoftplus` gate) are unit-tested
//! against hand-computed values. Bit-exact parity additionally needs the
//! official `deepseek-ai/DeepSeek-V4-{Flash,Pro}` safetensors **and** the
//! FP8/FP4 QAT dequant path — a very-large follow-on noted at each site.

use std::{collections::HashMap, sync::Arc};

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::Deserialize;

use super::dsa::{DsaConfig, DsaIndexer};
use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        embedding, Activation, CausalMasker, DeepSeekV2RopeConfig, DeepSeekV2RopeScaling,
        DeepSeekV2RotaryEmbedding, Mlp, RmsNorm, Sdpa,
    },
    layers_masker::CausalMaskConfig,
    moe::{MoEExperts, MoEExpertsConfig},
    ops::TopKLastDimOp,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::FlashParams, EitherCache, IsqModel, KvCache,
        ModelForwardContext, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(f64, routed_scaling_factor, 1.5);
serde_default_fn!(bool, norm_topk_prob, true);
serde_default_fn!(bool, tie_word_embeddings, false);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
serde_default_fn!(usize, hc_mult, 4);
serde_default_fn!(usize, hc_sinkhorn_iters, 20);
serde_default_fn!(f64, hc_eps, 1e-6);
serde_default_fn!(usize, o_groups, 8);
serde_default_fn!(usize, num_hash_layers, 3);
serde_default_fn!(f32, compress_rope_theta, 160000.0);

fn default_compress_ratios() -> Vec<u32> {
    Vec::new()
}

/// `architectures[0] == "DeepseekV4ForCausalLM"`.
///
/// Parsed verbatim from `deepseek-ai/DeepSeek-V4-Flash/config.json`. Extra HF
/// bookkeeping keys are ignored (this struct is not `deny_unknown_fields`).
#[derive(Deserialize, Clone, Debug)]
#[allow(dead_code)] // config mirrors config.json verbatim; not every key drives the forward
pub struct DeepSeekV4Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    /// Absorbed MLA latent / per-head attention dim (`head_dim`, 512). Also the
    /// dot-product width — `softmax_scale = 1/sqrt(head_dim)`.
    pub(crate) head_dim: usize,
    pub(crate) qk_rope_head_dim: usize,
    pub(crate) q_lora_rank: usize,
    /// Output o-LoRA rank (`o_lora_rank`, 1024).
    pub(crate) o_lora_rank: usize,
    #[serde(default = "o_groups")]
    pub(crate) o_groups: usize,
    pub(crate) num_attention_heads: usize,
    #[serde(default)]
    pub(crate) num_key_value_heads: Option<usize>,
    pub(crate) num_hidden_layers: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) n_routed_experts: usize,
    pub(crate) n_shared_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    #[serde(default = "num_hash_layers")]
    pub(crate) num_hash_layers: usize,
    #[serde(default = "routed_scaling_factor")]
    pub(crate) routed_scaling_factor: f64,
    #[serde(default = "norm_topk_prob")]
    pub(crate) norm_topk_prob: bool,
    /// Router scoring; V4 ships `"sqrtsoftplus"`. Kept as a string so future
    /// scoring variants parse without a code change (validated at gate build).
    #[serde(default)]
    pub(crate) scoring_func: Option<String>,
    #[serde(default)]
    pub(crate) topk_method: Option<String>,
    #[serde(default = "hidden_act")]
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) rope_theta: f32,
    pub(crate) rope_scaling: Option<DeepSeekV2RopeScaling>,
    #[serde(default)]
    pub(crate) attention_bias: bool,
    pub(crate) sliding_window: usize,
    #[serde(default)]
    pub(crate) swiglu_limit: Option<f64>,
    #[serde(alias = "quantization")]
    pub(crate) quantization_config: Option<QuantizedConfig>,
    // --- Hyper-Connections ---
    #[serde(default = "hc_mult")]
    pub(crate) hc_mult: usize,
    #[serde(default = "hc_sinkhorn_iters")]
    pub(crate) hc_sinkhorn_iters: usize,
    #[serde(default = "hc_eps")]
    pub(crate) hc_eps: f64,
    // --- DSA indexer ---
    pub(crate) index_n_heads: usize,
    pub(crate) index_head_dim: usize,
    pub(crate) index_topk: usize,
    // --- Compressed KV ---
    #[serde(default = "default_compress_ratios")]
    pub(crate) compress_ratios: Vec<u32>,
    #[serde(default = "compress_rope_theta")]
    pub(crate) compress_rope_theta: f32,
}

/// Per-layer attention mode, chosen by the `compress_ratios` schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttnMode {
    /// `ratio == 0`: dense full causal attention (e.g. first 2 + last layer).
    Full,
    /// `ratio == 4`: DSA top-`index_topk` indexed attention.
    Indexed,
    /// `ratio >= sliding_window` (128): sliding-window attention.
    Sliding,
}

impl DeepSeekV4Config {
    /// The DSA lightning-indexer config (always present in V4).
    pub(crate) fn dsa(&self) -> DsaConfig {
        DsaConfig {
            index_n_heads: self.index_n_heads,
            index_head_dim: self.index_head_dim,
            index_topk: self.index_topk,
            // The indexer reuses MLA's RoPE: rotate the leading qk_rope_head_dim.
            rope_dim: self.qk_rope_head_dim,
        }
    }

    /// MLA absorbed dot-product / value width.
    #[allow(dead_code)]
    pub(crate) fn q_head_dim(&self) -> usize {
        self.head_dim
    }

    /// Attention softmax scale — `1/sqrt(head_dim)`, matching the `ds4.c`
    /// `kq_scale`. (YaRN magnitude-scaling of the *score* is a numeric-parity
    /// follow-on; the rotary embedding still applies YaRN to the positions.)
    fn softmax_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Attention mode for layer `idx` from its compression ratio. Missing ratios
    /// (shorter `compress_ratios` than `num_hidden_layers`) default to `Full`.
    pub(crate) fn layer_mode(&self, idx: usize) -> AttnMode {
        match self.compress_ratios.get(idx).copied().unwrap_or(0) {
            0 => AttnMode::Full,
            r if (r as usize) >= self.sliding_window => AttnMode::Sliding,
            _ => AttnMode::Indexed,
        }
    }
}

// =============================================================================
// Hyper-Connections
// =============================================================================

/// Hyper-Connections residual carrier (`hc_mult` streams, Sinkhorn-mixed).
///
/// Wraps a single projection `fn: [n_hc·n_embd] -> [mix_dim]` plus the learned
/// per-region `scale` (pre/post/comb) and `base` (bias). `mix_dim` is
/// `2·n_hc + n_hc²` for a sublayer block (pre + post gates + combine matrix) or
/// `n_hc` for the output reduction (pre gates only).
///
/// The `n_hc × n_hc` combine matrix is the only non-elementwise piece: it is a
/// per-token Sinkhorn normalization (`softmax` over `src`, then alternating
/// column/row normalization for `iters` rounds).
pub(crate) struct HyperConnections {
    proj: Arc<dyn QuantMethod>,
    /// `[mix_dim]` — region scale broadcast (`pre`×n_hc, `post`×n_hc, `comb`×n_hc²).
    scale_vec: Tensor,
    /// `[mix_dim]` — additive bias.
    base: Tensor,
    #[allow(dead_code)]
    n_hc: usize,
    iters: usize,
    eps: f64,
    /// `true` for the output reduction (pre-gates only, `mix_dim == n_hc`).
    reduce_only: bool,
}

impl HyperConnections {
    /// Expand a plain embedding `[B, L, E]` into `n_hc` identical streams
    /// `[B, L, n_hc, E]` (every stream starts as the token vector).
    pub(crate) fn expand(x: &Tensor, n_hc: usize) -> Result<Tensor> {
        let (b, l, e) = x.dims3()?;
        x.unsqueeze(2)?.broadcast_as((b, l, n_hc, e))?.contiguous()
    }

    /// Load a sublayer (`reduce_only = false`) or output (`reduce_only = true`)
    /// Hyper-Connection from `fn.weight`, `scale`, `base` under `vb`.
    pub(crate) fn load(
        vb: ShardedVarBuilder,
        n_embd: usize,
        n_hc: usize,
        iters: usize,
        eps: f64,
        reduce_only: bool,
        quant: &Option<QuantizedConfig>,
    ) -> Result<Self> {
        let mix_dim = if reduce_only {
            n_hc
        } else {
            2 * n_hc + n_hc * n_hc
        };
        let proj = hanzo_quant::linear_no_bias(n_hc * n_embd, mix_dim, quant, vb.pp("fn"))?;

        let n_scale = if reduce_only { 1 } else { 3 };
        let scale = vb
            .get(n_scale, "scale")?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let base = vb.get(mix_dim, "base")?;

        // Broadcast the (pre, post, comb) scales over their regions once.
        let mut sv = Vec::with_capacity(mix_dim);
        sv.extend(std::iter::repeat_n(scale[0], n_hc)); // pre
        if !reduce_only {
            sv.extend(std::iter::repeat_n(scale[1], n_hc)); // post
            sv.extend(std::iter::repeat_n(scale[2], n_hc * n_hc)); // comb
        }
        let scale_vec = Tensor::from_vec(sv, mix_dim, base.device())?;

        Ok(Self {
            proj,
            scale_vec,
            base,
            n_hc,
            iters,
            eps,
            reduce_only,
        })
    }

    /// Build from already-loaded parts (the GGUF path: `proj` is a `GgufMatMul`,
    /// `scale`/`base` come from F32 GGUF tensors). `scale` is the raw `[n_scale]`
    /// per-region vector (1 for reduce-only, else 3 = pre/post/comb) — broadcast
    /// here exactly like [`Self::load`].
    pub(crate) fn from_parts(
        proj: Arc<dyn QuantMethod>,
        scale: Vec<f32>,
        base: Tensor,
        n_hc: usize,
        iters: usize,
        eps: f64,
        reduce_only: bool,
    ) -> Result<Self> {
        let mix_dim = if reduce_only {
            n_hc
        } else {
            2 * n_hc + n_hc * n_hc
        };
        let mut sv = Vec::with_capacity(mix_dim);
        sv.extend(std::iter::repeat_n(scale[0], n_hc));
        if !reduce_only {
            sv.extend(std::iter::repeat_n(scale[1], n_hc));
            sv.extend(std::iter::repeat_n(scale[2], n_hc * n_hc));
        }
        let scale_vec = Tensor::from_vec(sv, mix_dim, base.device())?;
        Ok(Self {
            proj,
            scale_vec,
            base,
            n_hc,
            iters,
            eps,
            reduce_only,
        })
    }

    /// RMS-norm (no learned weight) of the flattened streams: `[B, L, n_hc·E]`.
    fn rms_flat(&self, hc: &Tensor) -> Result<Tensor> {
        let (b, l, nh, e) = hc.dims4()?;
        let flat = hc.reshape((b, l, nh * e))?.to_dtype(DType::F32)?;
        let var = flat.sqr()?.mean_keepdim(D::Minus1)?;
        flat.broadcast_div(&(var + self.eps)?.sqrt()?)?
            .to_dtype(hc.dtype())
    }

    /// `z = fn(rms(flat)) · scale + base` — the raw, scaled, biased projection.
    fn projected(&self, hc: &Tensor) -> Result<Tensor> {
        let flat = self.rms_flat(hc)?;
        let raw = self.proj.forward(&flat)?;
        raw.broadcast_mul(&self.scale_vec.to_dtype(raw.dtype())?)?
            .broadcast_add(&self.base.to_dtype(raw.dtype())?)
    }

    /// Reduce streams to one vector via pre-gates: `Σ_h hc[h] · pre[h]`.
    fn reduce(&self, hc: &Tensor, pre: &Tensor) -> Result<Tensor> {
        hc.broadcast_mul(&pre.unsqueeze(D::Minus1)?)?.sum(2)
    }

    /// Sublayer **pre** step: reduce `[B,L,n_hc,E]` to the sublayer input
    /// `[B,L,E]`, returning the `post` gates `[B,L,n_hc]` and the Sinkhorn
    /// combine matrix `[B,L,n_hc,n_hc]` (`[…, dst, src]`) for the post step.
    pub(crate) fn pre(&self, hc: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        debug_assert!(!self.reduce_only);
        let (b, l, nh, _e) = hc.dims4()?;
        let z = self.projected(hc)?;

        let pre = (hanzo_nn::ops::sigmoid(&z.narrow(D::Minus1, 0, nh)?)? + self.eps)?;
        let reduced = self.reduce(hc, &pre)?;

        let post = (hanzo_nn::ops::sigmoid(&z.narrow(D::Minus1, nh, nh)?)? * 2.0)?;

        let comb_logits = z
            .narrow(D::Minus1, 2 * nh, nh * nh)?
            .reshape((b, l, nh, nh))?;
        let comb = self.sinkhorn(&comb_logits)?;

        Ok((reduced, post, comb))
    }

    /// Output reduction: `[B,L,n_hc,E] -> [B,L,E]` via pre-gates only.
    pub(crate) fn reduce_output(&self, hc: &Tensor) -> Result<Tensor> {
        debug_assert!(self.reduce_only);
        let pre = (hanzo_nn::ops::sigmoid(&self.projected(hc)?)? + self.eps)?;
        self.reduce(hc, &pre)
    }

    /// Sublayer **post** step: inject the sublayer output back and mix the prior
    /// streams through the combine matrix.
    ///
    /// `new_hc[dst] = post[dst] · block_out + Σ_src comb[dst+src·n_hc] · hc[src]`.
    /// With `comb` stored `[…, dst, src]`, the `comb[dst+src·n_hc]` index is the
    /// transpose, so `C = combᵀ` and `mixed = C @ hc`.
    pub(crate) fn post(
        &self,
        hc: &Tensor,
        block_out: &Tensor,
        post: &Tensor,
        comb: &Tensor,
    ) -> Result<Tensor> {
        let c = comb.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let mixed = c.matmul(&hc.contiguous()?)?;
        let inject = post
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&block_out.unsqueeze(2)?)?;
        mixed + inject
    }

    /// Per-token Sinkhorn over `[…, dst, src]`: softmax(src) + eps, then column
    /// (over `dst`) / row (over `src`) alternation, matching `ds4.c`
    /// `hc_split_sinkhorn_one`.
    fn sinkhorn(&self, logits: &Tensor) -> Result<Tensor> {
        // CUDA: one fused f32 kernel instead of ~120 tiny bf16 tensor-op launches
        // (the mHC swarm was ~20% of decode GPU time — see hanzo_ml Tensor::hc_sinkhorn).
        // Off-device keeps the tensor-op alternation (softmax + row/col norm).
        if logits.device().is_cuda() {
            return logits.hc_sinkhorn(self.iters, self.eps);
        }
        let mut c = (hanzo_nn::ops::softmax_last_dim(logits)? + self.eps)?;
        c = self.col_norm(&c)?;
        for _ in 1..self.iters {
            c = self.row_norm(&c)?;
            c = self.col_norm(&c)?;
        }
        Ok(c)
    }

    /// Normalize each row (sum over `src`, the last axis).
    fn row_norm(&self, c: &Tensor) -> Result<Tensor> {
        let s = (c.sum_keepdim(D::Minus1)? + self.eps)?;
        c.broadcast_div(&s)
    }

    /// Normalize each column (sum over `dst`, axis -2).
    fn col_norm(&self, c: &Tensor) -> Result<Tensor> {
        let s = (c.sum_keepdim(D::Minus2)? + self.eps)?;
        c.broadcast_div(&s)
    }
}

// =============================================================================
// Attention — absorbed MLA + o-LoRA + sinks + sliding-window + DSA
// =============================================================================

struct V4Attention {
    q_a: Arc<dyn QuantMethod>,
    q_a_norm: RmsNorm,
    q_b: Arc<dyn QuantMethod>,
    kv_a: Arc<dyn QuantMethod>,
    kv_a_norm: RmsNorm,
    /// o-LoRA: `heads -> o_lora_rank -> hidden` (mirrors `QProj::Lora`). The
    /// `o_groups` block-diagonal grouping (`out_low_dim = o_groups·o_lora_rank`)
    /// is the numeric-faithful follow-on; the rank-`o_lora_rank` form here is
    /// structurally a low-rank output projection.
    o_a: Arc<dyn QuantMethod>,
    o_b: Arc<dyn QuantMethod>,
    rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
    indexer: Option<DsaIndexer>,
    sdpa_params: SdpaParams,
    mode: AttnMode,
    num_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rms_eps: f64,
}

/// Unweighted per-head RMS norm over the last (`head_dim`) axis of `[B,nh,L,hd]`
/// — ds4.c `head_rms_norm_inplace` (ds4.c:4516), applied to the query after
/// `q_b` and **before** RoPE (ds4.c:6756). No learned gain (γ=1).
pub(crate) fn per_head_rms_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let scale = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.recip()?.sqrt()?;
    x.broadcast_mul(&scale)
}

impl V4Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV4Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<hanzo_quant::Comm>,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;

        let q_a = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.q_lora_rank,
            &cfg.quantization_config,
            cfg.attention_bias,
            mapper.set_device(layer_idx, vb.pp("q_a_proj"), loading_isq),
        )?;
        let q_a_norm = RmsNorm::new(
            cfg.q_lora_rank,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("q_a_layernorm"), false),
        )?;
        let q_b = ColumnParallelLayer::new(
            cfg.q_lora_rank,
            num_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_b_proj"), loading_isq),
        )?;

        // Single shared MLA latent (num_key_value_heads = 1): both K and V.
        let kv_a = ReplicatedLayer::new(
            cfg.hidden_size,
            head_dim,
            &cfg.quantization_config,
            cfg.attention_bias,
            mapper.set_device(layer_idx, vb.pp("kv_a_proj_with_mqa"), loading_isq),
        )?;
        let kv_a_norm = RmsNorm::new(
            head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("kv_a_layernorm"), false),
        )?;

        let o_a = ReplicatedLayer::new(
            num_heads * head_dim,
            cfg.o_lora_rank,
            &cfg.quantization_config,
            false,
            mapper.set_device(layer_idx, vb.pp("o_a_proj"), loading_isq),
        )?;
        let o_b = RowParallelLayer::new(
            cfg.o_lora_rank,
            cfg.hidden_size,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_b_proj"), loading_isq),
        )?;

        // Attention sinks: `[num_heads]`, absent => standard softmax.
        let sink_vb = mapper.set_device(layer_idx, vb.pp("attn_sinks"), false);
        let sinks = if sink_vb.contains_tensor("weight") {
            Some(sink_vb.get(num_heads, "weight")?)
        } else {
            None
        };

        let mode = cfg.layer_mode(layer_idx);
        let sliding_window = match mode {
            AttnMode::Sliding => Some(cfg.sliding_window),
            _ => None,
        };

        // DSA indexer: query from the q-LoRA latent (q_lora_rank), keys from the
        // normed hidden (hidden_size). Returns None (dense fallback) if the
        // indexer tensors are absent.
        let indexer = DsaIndexer::load(
            cfg.dsa(),
            cfg.q_lora_rank,
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &cfg.quantization_config,
            mapper.set_device(layer_idx, vb.pp("indexer"), loading_isq),
        )?;

        Ok(Self {
            q_a,
            q_a_norm,
            q_b,
            kv_a,
            kv_a_norm,
            o_a,
            o_b,
            rotary_emb,
            indexer,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / comm.world_size().max(1),
                softcap: None,
                softmax_scale: cfg.softmax_scale(),
                sliding_window,
                sinks,
            },
            mode,
            num_heads: num_heads / comm.world_size().max(1),
            head_dim,
            rope_dim: cfg.qk_rope_head_dim,
            rms_eps: cfg.rms_norm_eps,
        })
    }

    /// Apply partial RoPE to the **trailing** `rope_dim` of the query `[B,nh,L,hd]`
    /// and the single-head latent `[B,1,L,hd]`; the leading `n_nope` dims pass
    /// through. This matches ds4.c `rope_tail_ext_inplace` (ds4.c:6856), which
    /// rotates `x + n_nope` where `n_nope = head_dim - n_rot` — the positional
    /// part is the **last** `qk_rope_head_dim` dims (the `[q_nope first, q_pe last]`
    /// split shared by the GGUF and HF DeepSeek conventions), NOT the leading dims.
    fn rope_partial(
        &self,
        q: &Tensor,
        kv: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let hd = self.head_dim;
        if self.rope_dim >= hd {
            return self.rotary_emb.forward_positions(
                &q.contiguous()?,
                &kv.contiguous()?,
                positions,
            );
        }
        let n_nope = hd - self.rope_dim;
        let q_pass = q.narrow(D::Minus1, 0, n_nope)?.contiguous()?;
        let q_rot = q.narrow(D::Minus1, n_nope, self.rope_dim)?.contiguous()?;
        let k_pass = kv.narrow(D::Minus1, 0, n_nope)?.contiguous()?;
        let k_rot = kv.narrow(D::Minus1, n_nope, self.rope_dim)?.contiguous()?;
        let (q_rot, k_rot) = self
            .rotary_emb
            .forward_positions(&q_rot, &k_rot, positions)?;
        let q = Tensor::cat(&[&q_pass, &q_rot], D::Minus1)?.contiguous()?;
        let kv = Tensor::cat(&[&k_pass, &k_rot], D::Minus1)?.contiguous()?;
        Ok((q, kv))
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;

        // Query: hidden -> q-LoRA(a, norm, b) -> [B, nh, L, head_dim], then the
        // unweighted per-head RMS norm (ds4.c:6756) before RoPE.
        let q_lat = self.q_a_norm.forward(&self.q_a.forward(xs)?)?;
        let mut q = self
            .q_b
            .forward(&q_lat)?
            .reshape((bs, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        q = per_head_rms_norm(&q, self.rms_eps)?;

        // Single 512-dim latent (K == V), one shared KV head.
        let mut kv = self
            .kv_a_norm
            .forward(&self.kv_a.forward(xs)?)?
            .reshape((bs, seq_len, 1, self.head_dim))?
            .transpose(1, 2)?;

        let positions = ctx
            .rope_positions(q.device())?
            .ok_or_else(|| hanzo_ml::Error::msg("missing RoPE positions"))?
            .clone();
        (q, kv) = self.rope_partial(&q, &kv, &positions)?;

        let (k, v) = kv_cache.append(&kv, &kv)?;

        // DSA selection on the eager cold-cache prefill path (Indexed layers).
        let dsa_mask = if self.mode == AttnMode::Indexed {
            match (self.indexer.as_ref(), attention_mask.as_option_tensor()) {
                (Some(indexer), Some(base)) if base.dim(D::Minus1)? == seq_len => {
                    let selection = indexer.forward(
                        &q_lat,
                        xs,
                        Some(&self.rotary_emb),
                        Some(&positions),
                        true,
                    )?;
                    Some(AttentionMask::Custom(selection.combine_with_mask(base)?))
                }
                _ => None,
            }
        } else {
            None
        };

        let attn = Sdpa.run_attention(
            &q,
            &k,
            &v,
            dsa_mask.as_ref().unwrap_or(attention_mask),
            Some(ctx.flash_params()),
            &self.sdpa_params,
        )?;

        let attn = attn.transpose(1, 2)?.reshape((bs, seq_len, ()))?;
        self.o_b.forward(&self.o_a.forward(&attn)?)
    }
}

// =============================================================================
// MoE — sqrtsoftplus NoAuxTc gate (flat, no groups)
// =============================================================================

/// `softplus(x) = log(1 + e^x)` (the `granite.rs` formula).
pub(crate) fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
}

/// DeepSeek-V4 router gate: `probs = sqrt(softplus(logits))`, biased top-`k`
/// selection (NoAuxTc, **no** group masking), unbiased normalized weights ×
/// `routed_scaling_factor`.
struct V4MoeGate {
    weight: Tensor,
    e_score_correction_bias: Tensor,
    top_k: usize,
    routed_scaling_factor: f64,
    norm_topk: bool,
}

impl V4MoeGate {
    fn new(cfg: &DeepSeekV4Config, vb: ShardedVarBuilder) -> Result<Self> {
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
            routed_scaling_factor: cfg.routed_scaling_factor,
            norm_topk: cfg.norm_topk_prob,
        })
    }

    /// Returns `(topk_idx [n, top_k] u32, topk_weight [n, top_k] f32)`.
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_bs, _seq, h) = xs.dims3()?;
        let xs = xs.reshape(((), h))?;
        let logits = xs
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.weight.t()?.to_dtype(DType::F32)?)?;

        // sqrt(softplus(.)) scoring.
        let probs = softplus(&logits)?.sqrt()?;

        // Biased top-k selection; weights from the *unbiased* probs.
        let selection = probs.broadcast_add(&self.e_score_correction_bias.unsqueeze(0)?)?;
        let topk_idx = selection.topk(self.top_k)?.indices;
        let mut weights = probs.gather(&topk_idx, 1)?;

        if self.norm_topk {
            let denom = (weights.sum_keepdim(D::Minus1)? + 1e-20)?;
            weights = weights.broadcast_div(&denom)?;
        }
        weights = (weights * self.routed_scaling_factor)?;
        Ok((topk_idx, weights))
    }
}

struct V4Moe {
    experts: MoEExperts,
    shared_experts: Option<Mlp>,
    gate: V4MoeGate,
}

impl V4Moe {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &DeepSeekV4Config,
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

        let shared_experts = if cfg.n_shared_experts > 0 {
            Some(Mlp::new(
                mapper.set_device(layer_idx, vb.pp("shared_experts"), loading_isq),
                cfg.hidden_size,
                cfg.moe_intermediate_size * cfg.n_shared_experts,
                &cfg.quantization_config,
                cfg.hidden_act,
                comm,
            )?)
        } else {
            None
        };

        let gate = V4MoeGate::new(cfg, mapper.set_device(layer_idx, vb.pp("gate"), false))?;
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

        if let Some(ref shared_experts) = self.shared_experts {
            y = (y + shared_experts.forward(&identity)?)?;
        }
        Ok(y)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = self.experts.get_isq_layers();
        if let Some(ref mut shared) = self.shared_experts {
            layers.push(&mut shared.gate);
            layers.push(&mut shared.up);
            layers.push(&mut shared.down);
        }
        layers
    }
}

enum MoeOrMlp {
    Moe(Box<V4Moe>),
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

// =============================================================================
// Decoder layer — HC carrier wrapping attention + MoE/MLP sublayers
// =============================================================================

struct DecoderLayer {
    hc_attn: HyperConnections,
    attn_norm: RmsNorm,
    attn: V4Attention,
    hc_ffn: HyperConnections,
    ffn_norm: RmsNorm,
    moe_or_mlp: MoeOrMlp,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV4Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<hanzo_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        let attn = V4Attention::new(
            rotary_emb,
            cfg,
            vb.pp("self_attn"),
            mapper,
            layer_idx,
            loading_isq,
            comm,
        )?;
        let attn_norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let ffn_norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;

        let hc_attn = HyperConnections::load(
            mapper.set_device(layer_idx, vb.pp("hc_attn"), false),
            cfg.hidden_size,
            cfg.hc_mult,
            cfg.hc_sinkhorn_iters,
            cfg.hc_eps,
            false,
            &cfg.quantization_config,
        )?;
        let hc_ffn = HyperConnections::load(
            mapper.set_device(layer_idx, vb.pp("hc_ffn"), false),
            cfg.hidden_size,
            cfg.hc_mult,
            cfg.hc_sinkhorn_iters,
            cfg.hc_eps,
            false,
            &cfg.quantization_config,
        )?;

        // Hash-routed dense-ish early layers use a plain MLP; the rest are MoE.
        let moe_or_mlp = if layer_idx >= cfg.num_hash_layers {
            MoeOrMlp::Moe(Box::new(V4Moe::new(
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
                cfg.moe_intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                comm,
            )?)
        };

        Ok(Self {
            hc_attn,
            attn_norm,
            attn,
            hc_ffn,
            ffn_norm,
            moe_or_mlp,
        })
    }

    /// One layer over the HC carrier `[B, L, n_hc, E]`.
    fn forward(
        &self,
        hc: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        // Attention sublayer.
        let (reduced, post, comb) = self.hc_attn.pre(hc)?;
        let normed = self.attn_norm.forward(&reduced)?;
        let attn_out = self.attn.forward(&normed, attention_mask, kv_cache, ctx)?;
        let hc = self.hc_attn.post(hc, &attn_out, &post, &comb)?;

        // FFN / MoE sublayer.
        let (reduced, post, comb) = self.hc_ffn.pre(&hc)?;
        let normed = self.ffn_norm.forward(&reduced)?;
        let ffn_out = self.moe_or_mlp.forward(&normed)?;
        self.hc_ffn.post(&hc, &ffn_out, &post, &comb)
    }
}

// =============================================================================
// Model
// =============================================================================

pub struct DeepSeekV4 {
    lm_head: Arc<dyn QuantMethod>,
    embed_tokens: Embedding,
    output_hc: HyperConnections,
    norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    n_hc: usize,
    cache: EitherCache,
    device: Device,
    max_seq_len: usize,
    cfg: ModelConfigMetadata,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl DeepSeekV4 {
    pub fn new(
        cfg: &DeepSeekV4Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
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
        let output_hc = HyperConnections::load(
            mapper.set_nm_device(vb_m.pp("output_hc"), false),
            cfg.hidden_size,
            cfg.hc_mult,
            cfg.hc_sinkhorn_iters,
            cfg.hc_eps,
            true,
            &cfg.quantization_config,
        )?;

        // Per-layer RoPE schedule (ds4.c:6892-6904): **Full** layers (compress
        // ratio 0) use `rope_theta` (10000) with NO YaRN; **compressed** layers
        // (Indexed/Sliding) use `compress_rope_theta` (160000) WITH YaRN. Build
        // both rope variants per device and select by `layer_mode`.
        let rope_cfg_full = DeepSeekV2RopeConfig {
            rope_scaling: None,
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        let rope_cfg_comp = DeepSeekV2RopeConfig {
            rope_scaling: cfg.rope_scaling.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.compress_rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        let mut ropes_full = HashMap::new();
        let mut ropes_comp = HashMap::new();
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes_full.entry(device.location()).or_insert_with(|| {
                Arc::new(
                    DeepSeekV2RotaryEmbedding::new(&rope_cfg_full, vb.dtype(), device)
                        .expect("full RoPE"),
                )
            });
            ropes_comp.entry(device.location()).or_insert_with(|| {
                Arc::new(
                    DeepSeekV2RotaryEmbedding::new(&rope_cfg_comp, vb.dtype(), device)
                        .expect("compressed RoPE"),
                )
            });
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
            // Full layers -> θ=10000 (no YaRN); compressed layers -> θ=160000 (YaRN).
            let rope_map = if cfg.layer_mode(layer_idx) == AttnMode::Full {
                &ropes_full
            } else {
                &ropes_comp
            };
            let rotary_emb = rope_map
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let comm = mapper.get_comm_for(layer_idx)?;
            DecoderLayer::new(
                rotary_emb,
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                &comm,
                normal_loading_metadata.real_device.clone(),
            )
        })?;

        Ok(Self {
            lm_head,
            embed_tokens,
            output_hc,
            norm,
            layers,
            n_hc: cfg.hc_mult,
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
                num_kv_heads: 1,
                num_attn_heads: (cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: Some(cfg.sliding_window),
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        let cache = &mut self.cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig::default(),
        )?;
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;

        // Hyper-Connections carrier: [B, L, n_hc, E].
        let mut hc = HyperConnections::expand(&xs, self.n_hc)?;
        for (i, layer) in self.layers.iter().enumerate() {
            hc = self.mapper.map(hc, i)?;
            hc = layer.forward(&hc, &attention_mask.get(hc.device()), &mut cache[i], ctx)?;
        }

        let xs = self.output_hc.reduce_output(&hc)?.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        self.lm_head.forward(&xs)
    }
}

impl IsqModel for DeepSeekV4 {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors: Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)> = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.attn.q_a, Some(i)));
            tensors.push((&mut layer.attn.q_b, Some(i)));
            tensors.push((&mut layer.attn.kv_a, Some(i)));
            tensors.push((&mut layer.attn.o_a, Some(i)));
            tensors.push((&mut layer.attn.o_b, Some(i)));
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
        let mut tensors: Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)> = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if let MoeOrMlp::Moe(moe) = &mut layer.moe_or_mlp {
                for layer in moe.get_isq_layers() {
                    tensors.push((layer, Some(i)));
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
            uvb_l.pp("input_layernorm").add(&layer.attn_norm);
            uvb_l.pp("post_attention_layernorm").add(&layer.ffn_norm);
            uvb_l
                .pp("self_attn")
                .pp("q_a_layernorm")
                .add(&layer.attn.q_a_norm);
            uvb_l
                .pp("self_attn")
                .pp("kv_a_layernorm")
                .add(&layer.attn.kv_a_norm);
        }
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for DeepSeekV4 {}

impl NormalModel for DeepSeekV4 {
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

impl AnyMoeBaseModelMixin for DeepSeekV4 {}

/// Test-only helpers to read 3-D/4-D tensors as nested `Vec`s.
#[cfg(test)]
trait NestedVecExt {
    fn to_vec_dim3(&self) -> Vec<Vec<Vec<f32>>>;
    fn to_vec_dim4(&self) -> Vec<Vec<Vec<Vec<f32>>>>;
}

#[cfg(test)]
impl NestedVecExt for Tensor {
    fn to_vec_dim3(&self) -> Vec<Vec<Vec<f32>>> {
        self.to_dtype(DType::F32).unwrap().to_vec3::<f32>().unwrap()
    }
    fn to_vec_dim4(&self) -> Vec<Vec<Vec<Vec<f32>>>> {
        let (a, b, c, d) = self.dims4().unwrap();
        let flat = self
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut out = Vec::with_capacity(a);
        let mut it = flat.into_iter();
        for _ in 0..a {
            let mut bb = Vec::with_capacity(b);
            for _ in 0..b {
                let mut cc = Vec::with_capacity(c);
                for _ in 0..c {
                    let mut dd = Vec::with_capacity(d);
                    for _ in 0..d {
                        dd.push(it.next().unwrap());
                    }
                    cc.push(dd);
                }
                bb.push(cc);
            }
            out.push(bb);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_quant::ShardedSafeTensors;

    fn vb_from(tensors: HashMap<String, Tensor>) -> ShardedVarBuilder {
        ShardedSafeTensors::wrap(Box::new(tensors), DType::F32, Device::Cpu)
    }

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    // ---- Config ----

    /// The real `deepseek-ai/DeepSeek-V4-Flash/config.json`, verbatim.
    const DSV4_FLASH_CONFIG: &str = r#"{
  "architectures": ["DeepseekV4ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 1,
  "expert_dtype": "fp4",
  "hc_eps": 1e-06,
  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "head_dim": 512,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 512,
  "initializer_range": 0.02,
  "max_position_embeddings": 1048576,
  "model_type": "deepseek_v4",
  "moe_intermediate_size": 2048,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 64,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 43,
  "num_hash_layers": 3,
  "num_key_value_heads": 1,
  "num_nextn_predict_layers": 1,
  "o_groups": 8,
  "o_lora_rank": 1024,
  "q_lora_rank": 1024,
  "qk_rope_head_dim": 64,
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128]
  },
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 1.5,
  "scoring_func": "sqrtsoftplus",
  "sliding_window": 128,
  "swiglu_limit": 10.0,
  "tie_word_embeddings": false,
  "topk_method": "noaux_tc",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.57.1",
  "use_cache": true,
  "vocab_size": 129280,
  "compress_rope_theta": 160000,
  "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]
}"#;

    #[test]
    fn deepseek_v4_config_parses() {
        let cfg: DeepSeekV4Config = serde_json::from_str(DSV4_FLASH_CONFIG)
            .expect("DeepSeek-V4-Flash config must deserialize");

        // Absorbed MLA dims.
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.head_dim, 512);
        assert_eq!(cfg.qk_rope_head_dim, 64);
        assert_eq!(cfg.q_lora_rank, 1024);
        assert_eq!(cfg.o_lora_rank, 1024);
        assert_eq!(cfg.o_groups, 8);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, Some(1));
        assert_eq!(cfg.num_hidden_layers, 43);
        assert_eq!(cfg.vocab_size, 129280);
        assert_eq!(cfg.q_head_dim(), 512);

        // MoE (sqrtsoftplus NoAuxTc).
        assert_eq!(cfg.n_routed_experts, 256);
        assert_eq!(cfg.n_shared_experts, 1);
        assert_eq!(cfg.num_experts_per_tok, 6);
        assert_eq!(cfg.moe_intermediate_size, 2048);
        assert_eq!(cfg.num_hash_layers, 3);
        assert_eq!(cfg.routed_scaling_factor, 1.5);
        assert!(cfg.norm_topk_prob);
        assert_eq!(cfg.scoring_func.as_deref(), Some("sqrtsoftplus"));
        assert_eq!(cfg.topk_method.as_deref(), Some("noaux_tc"));

        // Hyper-Connections.
        assert_eq!(cfg.hc_mult, 4);
        assert_eq!(cfg.hc_sinkhorn_iters, 20);
        assert_eq!(cfg.hc_eps, 1e-6);

        // DSA indexer + RoPE/quant survive deserialization.
        assert_eq!(cfg.index_n_heads, 64);
        assert_eq!(cfg.index_head_dim, 128);
        assert_eq!(cfg.index_topk, 512);
        let dsa = cfg.dsa();
        assert_eq!(dsa.index_topk, 512);
        assert_eq!(dsa.rope_dim, 64);
        assert!(cfg.rope_scaling.is_some(), "YaRN (no mscale) must parse");
        assert!(cfg.quantization_config.is_some());

        // Sliding window + compressed-KV schedule.
        assert_eq!(cfg.sliding_window, 128);
        assert_eq!(cfg.compress_ratios.len(), 44);
        assert_eq!(cfg.layer_mode(0), AttnMode::Full); // ratio 0
        assert_eq!(cfg.layer_mode(1), AttnMode::Full); // ratio 0
        assert_eq!(cfg.layer_mode(2), AttnMode::Indexed); // ratio 4
        assert_eq!(cfg.layer_mode(3), AttnMode::Sliding); // ratio 128
        assert_eq!(cfg.layer_mode(43), AttnMode::Full); // ratio 0 (MTP tail)
    }

    // ---- Hyper-Connections ----

    /// Build a toy HC whose `fn` projection is all-zeros and `base` is all-zeros,
    /// so the gates collapse to their bias-free values:
    ///   pre  = sigmoid(0) + eps,  post = 2·sigmoid(0) = 1,  comb = uniform.
    fn toy_hc(
        n_hc: usize,
        n_embd: usize,
        iters: usize,
        eps: f64,
        reduce_only: bool,
    ) -> HyperConnections {
        let mix_dim = if reduce_only {
            n_hc
        } else {
            2 * n_hc + n_hc * n_hc
        };
        let n_scale = if reduce_only { 1 } else { 3 };
        let mut tensors = HashMap::new();
        tensors.insert(
            "fn.weight".to_string(),
            t(
                vec![0.0; mix_dim * n_hc * n_embd],
                &[mix_dim, n_hc * n_embd],
            ),
        );
        tensors.insert("scale".to_string(), t(vec![1.0; n_scale], &[n_scale]));
        tensors.insert("base".to_string(), t(vec![0.0; mix_dim], &[mix_dim]));
        HyperConnections::load(
            vb_from(tensors),
            n_embd,
            n_hc,
            iters,
            eps,
            reduce_only,
            &None,
        )
        .unwrap()
    }

    #[test]
    fn hc_expand_shapes() {
        let x = t(vec![1., 2., 3., 4., 5., 6.], &[1, 3, 2]); // [B=1, L=3, E=2]
        let hc = HyperConnections::expand(&x, 4).unwrap();
        assert_eq!(hc.dims(), &[1, 3, 4, 2]);
        // Every stream equals the token vector.
        let v = hc.to_vec_dim4();
        #[allow(clippy::needless_range_loop)]
        for s in 0..4 {
            assert_eq!(v[0][0][s], vec![1.0, 2.0]);
            assert_eq!(v[0][2][s], vec![5.0, 6.0]);
        }
    }

    /// With zero `fn`/`base` and `eps = 0`, `n_hc = 2`: pre-gate = 0.5, so the
    /// reduction `Σ_h 0.5·x = n_hc·0.5·x = x`. Exactly invertible.
    #[test]
    fn hc_pre_reduce_is_identity_when_uniform() {
        let hc_mod = toy_hc(2, 2, 1, 0.0, false);
        let x = t(vec![2.0, -3.0, 1.0, 4.0], &[1, 2, 2]); // [B=1, L=2, E=2]
        let hc = HyperConnections::expand(&x, 2).unwrap();
        let (reduced, post, comb) = hc_mod.pre(&hc).unwrap();
        assert_eq!(reduced.dims(), &[1, 2, 2]);
        assert_eq!(post.dims(), &[1, 2, 2]);
        assert_eq!(comb.dims(), &[1, 2, 2, 2]);

        let r = reduced.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in r.iter().zip(&xv) {
            assert!(
                (a - b).abs() < 1e-5,
                "reduce(expand(x)) must equal x: {a} vs {b}"
            );
        }
        // post = 2·sigmoid(0) = 1.0 everywhere.
        for p in post.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!((p - 1.0).abs() < 1e-5);
        }
    }

    /// post step with uniform combine + post=1: each new stream = block_out + x.
    #[test]
    fn hc_post_uniform_mix() {
        let hc_mod = toy_hc(2, 2, 1, 0.0, false);
        let x = t(vec![2.0, -3.0, 1.0, 4.0], &[1, 2, 2]);
        let hc = HyperConnections::expand(&x, 2).unwrap();
        let (_reduced, post, comb) = hc_mod.pre(&hc).unwrap();
        let block_out = t(vec![10.0, 20.0, 30.0, 40.0], &[1, 2, 2]);
        let new_hc = hc_mod.post(&hc, &block_out, &post, &comb).unwrap();
        assert_eq!(new_hc.dims(), &[1, 2, 2, 2]);
        // Expected: every stream = x + block_out.
        let v = new_hc.to_vec_dim4();
        let expect = [[12.0, 17.0], [31.0, 44.0]]; // x+block per (L, E)
        for li in 0..2 {
            #[allow(clippy::needless_range_loop)]
            for s in 0..2 {
                for e in 0..2 {
                    assert!(
                        (v[0][li][s][e] - expect[li][e]).abs() < 1e-4,
                        "post[{li}][{s}][{e}] = {} want {}",
                        v[0][li][s][e],
                        expect[li][e]
                    );
                }
            }
        }
    }

    /// Sinkhorn output columns (sum over `dst`) are exactly 1 (col-norm is last),
    /// the doubly-stochastic invariant the post-mix relies on.
    #[test]
    fn hc_sinkhorn_columns_sum_to_one() {
        let n_hc = 4;
        // Random-ish non-zero projection so the combine matrix is non-uniform.
        let n_embd = 3;
        let mix_dim = 2 * n_hc + n_hc * n_hc;
        let mut tensors = HashMap::new();
        let w: Vec<f32> = (0..mix_dim * n_hc * n_embd)
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();
        tensors.insert("fn.weight".to_string(), t(w, &[mix_dim, n_hc * n_embd]));
        tensors.insert("scale".to_string(), t(vec![1.0, 1.0, 1.0], &[3]));
        let base: Vec<f32> = (0..mix_dim).map(|i| ((i as f32) * 0.27).cos()).collect();
        tensors.insert("base".to_string(), t(base, &[mix_dim]));
        let hc_mod =
            HyperConnections::load(vb_from(tensors), n_embd, n_hc, 20, 1e-6, false, &None).unwrap();

        let x: Vec<f32> = (0..2 * n_embd).map(|i| (i as f32) - 3.0).collect();
        let x = t(x, &[1, 2, n_embd]);
        let hc = HyperConnections::expand(&x, n_hc).unwrap();
        let (_r, _p, comb) = hc_mod.pre(&hc).unwrap();

        // Sum over dst (axis 2) -> should be ~1 for every (token, src).
        let col_sums = comb.sum(2).unwrap().to_vec_dim3();
        for tok in &col_sums[0] {
            for s in tok {
                assert!((s - 1.0).abs() < 1e-4, "column must sum to 1, got {s}");
            }
        }
    }

    #[test]
    fn hc_output_reduce_shapes() {
        let hc_mod = toy_hc(2, 2, 1, 0.0, true);
        let x = t(vec![2.0, -3.0, 1.0, 4.0], &[1, 2, 2]);
        let hc = HyperConnections::expand(&x, 2).unwrap();
        let out = hc_mod.reduce_output(&hc).unwrap();
        assert_eq!(out.dims(), &[1, 2, 2]);
        // Uniform pre=0.5 (eps 0) -> reduce(expand(x)) == x again.
        let r = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in r.iter().zip(&xv) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // ---- sqrtsoftplus MoE gate ----

    #[test]
    fn v4_gate_sqrtsoftplus_selection() {
        // hidden=2, n_routed=4, top_k=2.
        let mut tensors = HashMap::new();
        // weight rows (expert,hidden): e0=[2,0] e1=[0,0] e2=[1,0] e3=[-1,0].
        tensors.insert(
            "weight".to_string(),
            t(vec![2., 0., 0., 0., 1., 0., -1., 0.], &[4, 2]),
        );
        // bias boosts expert 1 (low prob) into the selection.
        tensors.insert(
            "e_score_correction_bias".to_string(),
            t(vec![0., 5., 0., 0.], &[4]),
        );

        let mut cfg: DeepSeekV4Config = serde_json::from_str(DSV4_FLASH_CONFIG).unwrap();
        cfg.hidden_size = 2;
        cfg.n_routed_experts = 4;
        cfg.num_experts_per_tok = 2;
        cfg.routed_scaling_factor = 1.5;
        cfg.norm_topk_prob = true;
        let gate = V4MoeGate::new(&cfg, vb_from(tensors)).unwrap();

        // x = [1, 0] -> logits = [2, 0, 1, -1].
        let x = t(vec![1.0, 0.0], &[1, 1, 2]);
        let (idx, w) = gate.forward(&x).unwrap();
        assert_eq!(idx.dims(), &[1, 2]);

        // Selection = sqrt(softplus(logit)) + bias; top-2 = {e1 (bias), e0}.
        let idx = idx.to_vec2::<u32>().unwrap();
        assert_eq!(idx[0], vec![1, 0]);

        // Reference: unbiased sqrt(softplus) weights at {e1, e0}, normalized ×1.5.
        let sp = |z: f32| (1.0 + z.exp()).ln().sqrt();
        let (p1, p0) = (sp(0.0), sp(2.0));
        let sum = p1 + p0;
        let want = [p1 / sum * 1.5, p0 / sum * 1.5];
        let w = w.to_vec2::<f32>().unwrap();
        for (a, b) in w[0].iter().zip(&want) {
            assert!((a - b).abs() < 1e-4, "gate weight {a} want {b}");
        }
    }

    /// M1 — unweighted per-head RMS norm (ds4.c:4516 `head_rms_norm_inplace`):
    /// each head's `head_dim` vector is scaled to unit RMS, independently, with
    /// no learned gain. Verifies the post-q_b/pre-RoPE norm we apply per head.
    #[test]
    fn per_head_rms_norm_unit_rms_per_head() {
        // [B=1, nh=2, L=1, hd=4]; head0=[3,0,0,0] (RMS=1.5), head1=[1,1,1,1] (RMS=1).
        let x = t(vec![3., 0., 0., 0., 1., 1., 1., 1.], &[1, 2, 1, 4]);
        let y = per_head_rms_norm(&x, 0.0).unwrap().to_vec_dim4();
        // head0 -> [2,0,0,0] (3/1.5); head1 -> [1,1,1,1] (already unit RMS).
        let h0 = &y[0][0][0];
        let h1 = &y[0][1][0];
        assert!(
            (h0[0] - 2.0).abs() < 1e-4 && h0[1].abs() < 1e-4,
            "head0 {h0:?}"
        );
        for v in h1 {
            assert!((v - 1.0).abs() < 1e-4, "head1 {h1:?}");
        }
        // Each head now has unit RMS over head_dim.
        for h in [h0, h1] {
            let rms = (h.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
            assert!((rms - 1.0).abs() < 1e-4, "unit RMS, got {rms}");
        }
    }
}
