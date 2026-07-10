#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF DeepSeek-V4 (Flash/Pro) quantized model — `general.architecture = "deepseek4"`.
//!
//! Loads antirez's ds4 GGUF (`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-…`) and runs the
//! brand-new V4 architecture: **Hyper-Connections** (4-stream Sinkhorn residual),
//! **absorbed-latent MLA** (single 512-dim K==V latent, per-head q-RMS, trailing
//! partial-RoPE + inverse-RoPE on the output), attention **sinks**, **grouped
//! o-LoRA** (8 groups), **sqrtsoftplus** MoE gate (+ hash-routed early layers via
//! `ffn_gate_tid2eid`), shared expert, and per-layer SwiGLU clamp.
//!
//! The pure math (HyperConnections, `per_head_rms_norm`, `softplus`) is reused from
//! the safetensors [`super::deepseek4`] — one implementation, two loaders.
//!
//! SCOPE (v1): correct forward for `seqlen <= window_size` (128), where the
//! compressed-KV compressor + DSA indexer DON'T engage (verified vs the official
//! `inference/model.py`) — so those tensors are intentionally not loaded yet.
//! Longer-context (compressor/indexer) + the IQ2_XXS GPU mmvq kernel + MTP
//! speculative head are follow-ons.

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{
    CausalMaskConfig, CausalMasker, DeepSeekV2RopeConfig, DeepSeekV2RopeScaling,
    DeepSeekV2RotaryEmbedding, RmsNorm, ScaledRopeType, Sdpa,
};
use crate::layers_masker::PastKvLenCache;
use crate::ops::{TopKLastDimOp, TopKOutput};
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use hanzo_ml::quantized::QMatMul;
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use super::deepseek4::{per_head_rms_norm, softplus, HyperConnections};

const DEFAULT_MAX_SEQ_LEN: u32 = 1_048_576;

// ============================ metadata ============================

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct PropsGGUF {
    pub block_count: usize,
    pub embedding_length: usize,
    pub head_count: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    // MLA / attention.
    pub head_dim: usize,       // 512 (key_length == value_length)
    pub rope_head_dim: usize,  // 64
    pub q_lora_rank: usize,    // 1024
    pub o_lora_rank: usize,    // 1024
    pub o_groups: usize,       // 8
    pub sliding_window: usize, // 128
    // RoPE.
    pub rope_theta: f32,          // 10000 (Full layers)
    pub compress_rope_theta: f32, // 160000 (compressed layers)
    pub rope_scaling_factor: f32, // 16
    pub rope_orig_ctx: usize,     // 65536
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub compress_ratios: Vec<u32>, // per-layer schedule (0 = Full)
    // MoE.
    pub n_routed_experts: usize,    // 256
    pub num_experts_per_tok: usize, // 6
    pub n_shared_experts: usize,    // 1
    pub expert_ff_len: usize,       // 2048
    pub expert_weights_scale: f64,  // 1.5 (route_scale)
    pub norm_topk_prob: bool,
    pub hash_layer_count: usize, // 3
    pub swiglu_clamp: Vec<f32>,  // per-layer SwiGLU clamp (0 = none)
    // Hyper-Connections.
    pub hc_count: usize,          // 4
    pub hc_sinkhorn_iters: usize, // 20
    pub hc_eps: f64,              // 0
}

fn verify_arch(meta: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>) -> Result<()> {
    use crate::utils::gguf_metadata::TryValueInto;
    let arch: String = meta.get("general.architecture").cloned().try_value_into()?;
    if arch != "deepseek4" {
        hanzo_ml::bail!("Expected `deepseek4` architecture, got `{arch}`.");
    }
    Ok(())
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.has_required_keys(&[
            "block_count",
            "embedding_length",
            "attention.head_count",
            "attention.key_length",
            "rope.dimension_count",
            "expert_count",
            "expert_used_count",
        ])?;
        Ok(Self {
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: c.get_value::<u32>("embedding_length")? as usize,
            head_count: c.get_value::<u32>("attention.head_count")? as usize,
            rms_norm_eps: c
                .get_value::<f32>("attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-6),
            max_seq_len: c
                .get_value::<u64>("context_length")
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            head_dim: c.get_value::<u32>("attention.key_length")? as usize,
            rope_head_dim: c.get_value::<u32>("rope.dimension_count")? as usize,
            q_lora_rank: c.get_value::<u32>("attention.q_lora_rank")? as usize,
            o_lora_rank: c.get_value::<u32>("attention.output_lora_rank")? as usize,
            o_groups: c
                .get_value::<u32>("attention.output_group_count")
                .unwrap_or(1) as usize,
            sliding_window: c
                .get_value::<u32>("attention.sliding_window")
                .unwrap_or(128) as usize,
            rope_theta: c.get_value::<f32>("rope.freq_base").unwrap_or(10_000.0),
            compress_rope_theta: c
                .get_value::<f32>("attention.compress_rope_freq_base")
                .unwrap_or(160_000.0),
            rope_scaling_factor: c.get_value::<f32>("rope.scaling.factor").unwrap_or(1.0),
            rope_orig_ctx: c
                .get_value::<u32>("rope.scaling.original_context_length")
                .unwrap_or(65536) as usize,
            yarn_beta_fast: c
                .get_value::<f32>("rope.scaling.yarn_beta_fast")
                .unwrap_or(32.0),
            yarn_beta_slow: c
                .get_value::<f32>("rope.scaling.yarn_beta_slow")
                .unwrap_or(1.0),
            // The GGUF writes this as an I32 array; `Value::to_u32` refuses I32, so a
            // plain Vec<u32> read errors — and a silent `unwrap_or_default()` here made
            // EVERY layer Full-mode (wrong rope on 41/43 layers, compressor never
            // built, no sliding window). Accept any GGUF integer width; an absent key
            // (all-Full model) is the only case that yields an empty schedule.
            compress_ratios: c
                .get_value::<Vec<u32>>("attention.compress_ratios")
                .or_else(|_| {
                    c.get_value::<Vec<i32>>("attention.compress_ratios")
                        .map(|v| v.into_iter().map(|x| x.max(0) as u32).collect())
                })
                .or_else(|_| {
                    c.get_value::<Vec<i64>>("attention.compress_ratios")
                        .map(|v| v.into_iter().map(|x| x.max(0) as u32).collect())
                })
                .unwrap_or_default(),
            n_routed_experts: c.get_value::<u32>("expert_count")? as usize,
            num_experts_per_tok: c.get_value::<u32>("expert_used_count")? as usize,
            n_shared_experts: c.get_value::<u32>("expert_shared_count").unwrap_or(1) as usize,
            expert_ff_len: c.get_value::<u32>("expert_feed_forward_length")? as usize,
            expert_weights_scale: c.get_value::<f32>("expert_weights_scale").unwrap_or(1.0) as f64,
            norm_topk_prob: c.get_value::<bool>("expert_weights_norm").unwrap_or(true),
            hash_layer_count: c.get_value::<u32>("hash_layer_count").unwrap_or(0) as usize,
            swiglu_clamp: c
                .get_value::<Vec<f32>>("swiglu_clamp_exp")
                .unwrap_or_default(),
            hc_count: c.get_value::<u32>("hyper_connection.count").unwrap_or(1) as usize,
            hc_sinkhorn_iters: c
                .get_value::<u32>("hyper_connection.sinkhorn_iterations")
                .unwrap_or(20) as usize,
            hc_eps: c
                .get_value::<f32>("hyper_connection.epsilon")
                .unwrap_or(0.0) as f64,
        })
    }
}

// ============================ helpers ============================

pub(crate) fn gguf_linear(q: hanzo_ml::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(q),
        b: None,
    })?))
}

/// Dequantize a GGUF tensor to a plain F32 tensor (norms, sinks, bias, gate_inp,
/// hc base/scale, and the grouped-o `wo_a`).
pub(crate) fn deq(
    ct: &mut Content<'_, impl std::io::Seek + std::io::Read>,
    name: &str,
    dev: &Device,
) -> Result<Tensor> {
    ct.tensor(name, dev)?.dequantize(dev)?.to_dtype(DType::F32)
}

/// Per-layer attention mode from the `compress_ratios` schedule.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Full,
    Sliding,
    Indexed,
}

fn layer_mode(ratios: &[u32], idx: usize, window: usize) -> Mode {
    match ratios.get(idx).copied().unwrap_or(0) {
        0 => Mode::Full,
        r if (r as usize) >= window => Mode::Sliding,
        _ => Mode::Indexed,
    }
}

// ============================ MoE ============================

/// Routed experts (stacked GGUF banks) + 1 shared expert. Custom V4 routing:
/// `sqrt(softplus(logits))`, bias-shifted top-k selection (or `tid2eid` hash on
/// early layers), unbiased gathered weights renormalized × `route_scale`.
pub(crate) struct V4Moe {
    pub(crate) gate_inp: Tensor, // [hidden, n_experts] F32 (router weight, applied as x·W)
    pub(crate) bias: Option<Tensor>, // [n_experts] F32 (exp_probs_b) — MoE layers
    pub(crate) tid2eid: Option<Tensor>, // [used, vocab] I32 — hash layers
    pub(crate) gate_experts: QMatMul,
    pub(crate) up_experts: QMatMul,
    pub(crate) down_experts: QMatMul,
    pub(crate) shared_gate: Arc<dyn QuantMethod>,
    pub(crate) shared_up: Arc<dyn QuantMethod>,
    pub(crate) shared_down: Arc<dyn QuantMethod>,
    pub(crate) topk: usize,
    pub(crate) route_scale: f64,
    pub(crate) norm_topk: bool,
    pub(crate) swiglu_clamp: f32,
}

impl V4Moe {
    /// Routing weights+indices. `input_ids` (the token ids for this position) drive
    /// hash routing; `xs` is `[tokens, hidden]`.
    fn route(&self, xs: &Tensor, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        // scores = sqrt(softplus(xs @ gate_inp)).
        let logits = xs.to_dtype(DType::F32)?.matmul(&self.gate_inp)?; // [t, n_experts]
        let scores = softplus(&logits)?.sqrt()?;
        let (indices, weights) = match (&self.tid2eid, &self.bias) {
            (Some(tid2eid), _) => {
                // Hash routing: each token id maps to a fixed set of `used` experts.
                // GGUF stores the table file-shape [used=6, vocab]; the reader reverses
                // dims, so in candle it's [vocab, used]. Select the token's row (dim 0)
                // -> [t, used] directly (no transpose).
                let ids = tid2eid
                    .index_select(&input_ids.to_dtype(DType::U32)?, 0)? // [t, used]
                    .contiguous()?
                    .to_dtype(DType::U32)?;
                let w = scores.gather(&ids, 1)?;
                (ids, w)
            }
            (None, bias) => {
                let sel = match bias {
                    Some(b) => scores.broadcast_add(b)?,
                    None => scores.clone(),
                };
                let TopKOutput { values: _, indices } = sel.topk(self.topk)?;
                let indices = indices.contiguous()?;
                let w = scores.gather(&indices, 1)?; // UNBIASED scores at selected
                (indices, w)
            }
        };
        // Renormalize (sqrtsoftplus is not a simplex) and scale.
        let weights = weights.broadcast_div(&weights.sum_keepdim(D::Minus1)?)?;
        let weights = (weights * self.route_scale)?;
        Ok((indices.to_dtype(DType::U32)?, weights))
    }

    fn shared(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.shared_gate.forward(xs)?;
        let up = self.shared_up.forward(xs)?;
        let act = self.swiglu(&gate, &up)?;
        self.shared_down.forward(&act)
    }

    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let (gate, up) = if self.swiglu_clamp > 0.0 {
            let c = self.swiglu_clamp as f64;
            (gate.clamp(f64::NEG_INFINITY, c)?, up.clamp(-c, c)?)
        } else {
            (gate.clone(), up.clone())
        };
        crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)
    }

    pub(crate) fn forward(&self, xs: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        let (b, s, h) = xs.dims3()?;
        let orig_dtype = xs.dtype();
        // MoE compute runs in F32 (the quantized expert kernels + gate matmul expect
        // it, and routing is precision-sensitive); cast the carrier in here and back
        // to the model dtype at the end — same precision-op discipline as attention.
        let xs2 = xs.to_dtype(DType::F32)?.reshape((b * s, h))?;
        let ids = input_ids.reshape((b * s,))?;

        let (indices, weights) = self.route(&xs2, &ids)?;

        // Fused routed experts: gate/up → swiglu → down → weighted combine.
        let xs3 = xs2.reshape((b * s, 1, h))?;
        let (gate, up) =
            hanzo_ml::quantized::moe_gate_up(&xs3, &indices, &self.gate_experts, &self.up_experts)?;
        let act = self.swiglu(&gate, &up)?;
        let routed = self.down_experts.indexed_moe_forward(&act, &indices)?;
        let routed = hanzo_ml::quantized::moe_combine(&routed, &weights)?; // [b*s, h]

        let shared = self.shared(&xs2)?;
        (routed + shared)?.reshape((b, s, h))?.to_dtype(orig_dtype)
    }
}

// ============================ attention ============================

pub(crate) struct V4Attn {
    pub(crate) q_a: Arc<dyn QuantMethod>,
    pub(crate) q_a_norm: RmsNorm,
    pub(crate) q_b: Arc<dyn QuantMethod>,
    pub(crate) kv: Arc<dyn QuantMethod>,
    pub(crate) kv_norm: RmsNorm,
    pub(crate) wo_a_t: Tensor, // [groups, group_in, rank] — pre-transposed (matmul-ready) grouped-o weight
    pub(crate) wo_b: Arc<dyn QuantMethod>,
    pub(crate) rotary: Arc<DeepSeekV2RotaryEmbedding>,
    pub(crate) sdpa: SdpaParams,
    pub(crate) n_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) rope_dim: usize,
    pub(crate) o_groups: usize,
    pub(crate) o_lora_rank: usize,
    pub(crate) rms_eps: f64,
    pub(crate) attn_dtype: DType, // model/cache/mask dtype (BF16 on CUDA); q/k/v cast to it for SDPA + cache
    /// Present on compressed (Indexed/Sliding) layers: builds the compressed-KV rows the
    /// layer attends to alongside the raw window. `compress_ratio` is the window stride.
    pub(crate) compressor: Option<Compressor>,
    pub(crate) compress_ratio: usize,
}

impl V4Attn {
    /// Dense sliding-window + sinks attention over the raw KV (no compressed rows).
    /// For q_len 1..=8 (single-token decode AND MTP/EAGLE verify widths) take the
    /// fused F32 flash-decode kernel directly: the model KNOWS this path is
    /// plain-causal + sinks + per-row window (exactly the kernel's semantics),
    /// sidestepping the generic mask-variant ambiguity at the sinks backend.
    /// ds4-parity numerics AND kills the eager 3-pass on the latency-critical paths.
    /// `window_override`: `None` = the layer's own sliding window; `Some(w)` forces
    /// `w` (`Some(0)` = NO clamp — used for pre-sliced combined `[window ++ comp]`
    /// buffers where every row is already visible by construction).
    fn dense_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &AttentionMask,
        b: usize,
        window_override: Option<usize>,
    ) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        {
            let s_q = q.dim(2)?;
            if q.device().is_cuda() && b == 1 && (1..=8).contains(&s_q) {
                if let Some(sinks) = &self.sdpa.sinks {
                    let f32d = DType::F32;
                    let qk = q.squeeze(0)?.to_dtype(f32d)?.contiguous()?;
                    let kk = k.squeeze(0)?.to_dtype(f32d)?.contiguous()?;
                    let vv = v.squeeze(0)?.to_dtype(f32d)?.contiguous()?;
                    let sk = sinks.to_dtype(f32d)?.contiguous()?;
                    let kvl = kk.dim(1)?;
                    let win = match window_override {
                        Some(w) => w, // 0 = kernel no-clamp semantics
                        None => self.sdpa.sliding_window.unwrap_or(kvl),
                    };
                    let out = hanzo_ml::fattn_decode::fattn_decode_f32_hd512(
                        &qk,
                        &kk,
                        &vv,
                        Some(&sk),
                        win,
                        self.sdpa.softmax_scale,
                    )?;
                    return out.unsqueeze(0)?.to_dtype(q.dtype());
                }
            }
        }
        #[cfg(not(all(feature = "cuda", target_family = "unix")))]
        let _ = (b, window_override);
        Sdpa.run_attention(q, k, v, mask, None, &self.sdpa)
    }

    /// Trailing partial-RoPE on `[B, nh, L, hd]` (and the single-head latent),
    /// `inverse = true` un-rotates (the absorbed output path). ds4 rotates the
    /// last `rope_dim` dims; the leading `head_dim - rope_dim` pass through.
    fn rope(&self, x: &Tensor, positions: &Tensor, inverse: bool) -> Result<Tensor> {
        let hd = self.head_dim;
        let n_nope = hd - self.rope_dim;
        let orig = x.dtype();
        let pass = x.narrow(D::Minus1, 0, n_nope)?.contiguous()?;
        // RoPE is precision-sensitive and its cos/sin caches are F32: upcast the
        // rotated slice to F32 for the kernel, then return the caller's dtype
        // (rope is dtype-transparent — the precision op handles its own dtype).
        let rot = x
            .narrow(D::Minus1, n_nope, self.rope_dim)?
            .contiguous()?
            .to_dtype(DType::F32)?;
        let rot_in = rot;
        let rot = if inverse {
            self.rotary.forward_inverse_positions(&rot_in, positions)?
        } else {
            // forward_positions takes (q, k); reuse with a dummy second arg.
            let (rot, _) = self.rotary.forward_positions(&rot_in, &rot_in, positions)?;
            rot
        };
        // Diagnostic (env V4_TRACE_ROPE=1): dump the rope application — positions,
        // table fingerprint, in/out values at the LAST token — a handful of calls only.
        static TRACE_ROPE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static ROPE_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if *TRACE_ROPE.get_or_init(|| std::env::var("V4_TRACE_ROPE").is_ok_and(|v| v == "1"))
            && rot_in.dim(2).unwrap_or(1) > 1
        {
            let n = ROPE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 9 {
                let pos_v: Vec<u32> = positions
                    .to_dtype(DType::U32)
                    .and_then(|p| p.flatten_all()?.to_vec1())
                    .unwrap_or_default();
                let s = rot_in.dim(2).unwrap_or(1);
                let fp = |t: &Tensor| -> Vec<f32> {
                    t.narrow(2, s - 1, 1)
                        .and_then(|x| {
                            x.narrow(1, 0, 1)?
                                .flatten_all()?
                                .to_dtype(DType::F32)?
                                .to_vec1()
                        })
                        .map(|v: Vec<f32>| v.into_iter().take(4).collect())
                        .unwrap_or_default()
                };
                eprintln!(
                    "ROPETRACE call={n} inverse={inverse} s={s} positions={pos_v:?} in={:?} out={:?}",
                    fp(&rot_in),
                    fp(&rot)
                );
            }
        }
        let rot = rot.to_dtype(orig)?;
        Tensor::cat(&[&pass, &rot], D::Minus1)?.contiguous()
    }

    pub(crate) fn forward(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        positions: &Tensor,
        kv_cache: &mut KvCache,
        comp_state: Option<&mut CompressorState>,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        // Working dtype for attention + KV cache (the model dtype; BF16 on CUDA) —
        // the cache buffer + causal mask are allocated at this dtype, NOT the (F32)
        // carrier dtype. The QMatMul projections + RMS-norm + RoPE run in F32 for
        // precision; q/kv are cast to `attn_dtype` before SDPA + cache append so all
        // of q/k/v, the mask, and the cache buffer agree.
        let wdt = self.attn_dtype;

        // q: wq_a → q_norm → wq_b → [b, nh, s, hd] → per-head RMS → trailing rope.
        let q = self
            .q_b
            .forward(&self.q_a_norm.forward(&self.q_a.forward(x)?)?)?;
        let mut q = q
            .reshape((b, s, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        q = per_head_rms_norm(&q, self.rms_eps)?;
        q = self.rope(&q, positions, false)?.to_dtype(wdt)?;

        // kv latent: wkv → kv_norm → [b, 1, s, hd] → trailing rope (K == V).
        let kv = self
            .kv_norm
            .forward(&self.kv.forward(x)?)?
            .reshape((b, s, 1, self.head_dim))?
            .transpose(1, 2)?;
        let kv = self.rope(&kv, positions, false)?;
        // QAT: FP8-round the non-rope KV dims (per-64 block) to match the model's
        // training-time graph (model.py:506 `act_quant(kv[..., :-rd], 64, …)`). The
        // rope dims stay full-precision. Skipping this leaves the KV slightly off the
        // quantized graph the weights were trained against — a small per-step logit
        // drift that accumulates over decode into incoherence. Done in F32 (the QAT
        // round-trip dtype), then back to the working dtype.
        let kv = hanzo_ml::quantized::dsv4_qat::fp8_kv_quantize(
            &kv.to_dtype(DType::F32)?,
            self.rope_dim,
        )?
        .to_dtype(wdt)?;

        let (k, v) = kv_cache.append(&kv, &kv)?;

        // Compressed layers also attend to the compressor's compressed-KV rows (the
        // bulk of long-context info). Build them from the accumulated layer-input
        // history and concat to the raw window; a combined mask keeps both causal.
        // (seqlen<window: the raw window is fully causal-visible, the indexer selects
        // all rows since n_comp < top_k=512 — so dense over [raw ++ all-compressed] is
        // exact. Sliding window for >window context + the indexer top-k are follow-ons.)
        // The compressor runs only during PREFILL (s > 1). At single-token decode it is
        // (a) redundant for context ≤ window (the raw window already covers everything —
        // byte-identical output, verified) and (b) shape-variable + host-syncing, which
        // would break CUDA-graph capture and impose an O(n²) per-step rebuild. Skipping
        // it at decode makes the decode forward shape-stable + capture-eligible (the big
        // perf win) and keeps the O(n) decode. (>window long-context recall via the
        // compressed pool is the eager-path / indexer follow-on.)
        // Diagnostic kill-switch (env V4_DISABLE_COMPRESSOR=1): drop the compressed-KV
        // rows from prefill attention entirely — at ctx <= window the raw rows cover
        // everything, so this isolates whether the compressor path shifts the logits
        // vs the ds4 reference. Not for production (>window ctx needs the rows).
        static DISABLE_COMPRESSOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let disable_compressor = *DISABLE_COMPRESSOR
            .get_or_init(|| std::env::var("V4_DISABLE_COMPRESSOR").is_ok_and(|v| v == "1"));
        let attn = match (&self.compressor, comp_state) {
            (Some(comp), Some(state)) if s > 1 && is_prefill && !disable_compressor => {
                state.append(x)?;
                let total = state.len;
                let xh = state.slice(0, total)?;
                let hist_pos = Tensor::arange(0u32, total as u32, xh.device())?;
                match comp.compress(&xh, &hist_pos)? {
                    Some(kvc) => {
                        let nc = kvc.dim(1)?;
                        // Seed the decode-side row cache: emission continues from here.
                        state.seed_rows(Some(&kvc), nc);
                        let kvc = kvc
                            .reshape((b, nc, 1, self.head_dim))?
                            .transpose(1, 2)?
                            .to_dtype(wdt)?; // [b,1,nc,hd]
                        let kf = Tensor::cat(&[&k, &kvc], 2)?;
                        let vf = Tensor::cat(&[&v, &kvc], 2)?;
                        let lk = k.dim(2)?;
                        // `positions` is the per-sequence start offset; the queries are
                        // the `s` tokens at base..base+s (prefill) or the single token at
                        // base (decode). Build their absolute positions for the mask.
                        let base = positions
                            .to_dtype(DType::U32)?
                            .to_vec1::<u32>()?
                            .first()
                            .copied()
                            .unwrap_or(0);
                        let qpos: Vec<u32> = (0..s).map(|j| base + j as u32).collect();
                        let cmask = combined_compressed_mask(
                            &qpos,
                            0,
                            lk,
                            nc,
                            self.compress_ratio,
                            self.sdpa.sliding_window,
                            k.device(),
                            wdt,
                        )?;
                        Sdpa.run_attention(
                            &q,
                            &kf,
                            &vf,
                            &AttentionMask::Custom(cmask),
                            None,
                            &self.sdpa,
                        )?
                    }
                    None => Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa)?,
                }
            }
            // Compressed layers at DECODE/VERIFY beyond the raw window: the model's
            // long-context memory. Heal state from the absolute position (speculative
            // rollbacks), append this step's inputs, emit due compressed rows, and
            // attend [raw ++ compressed] with the per-row window/causal mask. Below
            // the window the compressed rows are provably redundant (raw covers all)
            // and the fused-kernel arm below handles it.
            (Some(comp), Some(state)) if !is_prefill && !disable_compressor => {
                let base = positions
                    .to_dtype(DType::U32)?
                    .to_vec1::<u32>()?
                    .first()
                    .copied()
                    .unwrap_or(0) as usize;
                state.sync_to(base, comp.ratio)?;
                state.append(x)?;
                let confirmed = base + s;
                let kv_len = k.dim(2)?;
                let window = self.sdpa.sliding_window.unwrap_or(usize::MAX);
                if kv_len <= window {
                    // Raw window still covers everything — fused fast path semantics.
                    self.dense_attn(&q, &k, &v, mask, b, None)?
                } else {
                    let due = confirmed / comp.ratio;
                    state.emit_due(comp, due, k.device())?;
                    let nc = state.emitted;
                    // Raw slice covering every query's window: abs [lo_min, base+s).
                    let lo_min = (base + 1).saturating_sub(window);
                    let raw_start = lo_min.min(kv_len - 1);
                    let raw_n = kv_len - raw_start;
                    let kw = k.narrow(2, raw_start, raw_n)?;
                    let vw = v.narrow(2, raw_start, raw_n)?;
                    let (kf, vf) = match &state.comp_rows {
                        Some(rows) if nc > 0 => {
                            let kvc = rows
                                .narrow(1, 0, nc)?
                                .reshape((b, nc, 1, self.head_dim))?
                                .transpose(1, 2)?
                                .to_dtype(wdt)?;
                            (Tensor::cat(&[&kw, &kvc], 2)?, Tensor::cat(&[&vw, &kvc], 2)?)
                        }
                        _ => (kw.clone(), vw.clone()),
                    };
                    if s == 1 {
                        // Single-token decode: after pre-slicing the raw window and
                        // emitting exactly the due rows, EVERY combined column is
                        // visible (raw slice == the query's window; comp rows 0..due
                        // are all fully past) — the mask is provably all-zeros. Run
                        // the fused kernel over the combined KV with the window
                        // clamp disabled (kv_len), keeping deep-context decode on
                        // the fast path instead of the eager 3-pass.
                        self.dense_attn(&q, &kf, &vf, &AttentionMask::None, b, Some(0))?
                    } else {
                        let qpos: Vec<u32> = (0..s).map(|j| (base + j) as u32).collect();
                        let cmask = combined_compressed_mask(
                            &qpos,
                            raw_start,
                            raw_n,
                            nc,
                            self.compress_ratio,
                            Some(window),
                            k.device(),
                            wdt,
                        )?;
                        Sdpa.run_attention(
                            &q,
                            &kf,
                            &vf,
                            &AttentionMask::Custom(cmask),
                            None,
                            &self.sdpa,
                        )?
                    }
                }
            }
            // Full layers (no compressor) + compressed layers at decode/verify —
            // dense sliding-window + sinks attention. For q_len 1..=8 (single-token
            // decode AND MTP/EAGLE verify widths) take the fused F32 flash-decode
            // kernel directly: the model KNOWS this arm is plain-causal + sinks +
            // per-row window (exactly the kernel's semantics), sidestepping the
            // generic mask-variant ambiguity at the sinks backend. ds4-parity
            // numerics AND kills the eager 3-pass on the latency-critical paths.
            _ => self.dense_attn(&q, &k, &v, mask, b, None)?,
        };

        // Inverse-RoPE on the output (absorbed: the V latent carried RoPE).
        let attn = self.rope(&attn, positions, true)?;

        // Grouped o-LoRA: o[b,s,groups,group_in] einsum wo_a[groups,rank,group_in]
        // -> [b,s,groups,rank]; then wo_b(flatten).
        let group_in = self.n_head * self.head_dim / self.o_groups;
        let o = attn
            .transpose(1, 2)? // [b, s, nh, hd]
            .reshape((b, s, self.o_groups, group_in))?
            .to_dtype(self.wo_a_t.dtype())?;
        // batched per-group matmul: [groups, b*s, group_in] @ [groups, group_in, rank].
        let o = o
            .transpose(1, 2)? // [b, groups, s, group_in]
            .transpose(0, 1)? // [groups, b, s, group_in]
            .reshape((self.o_groups, b * s, group_in))?
            .contiguous()?;
        let o = o.matmul(&self.wo_a_t)?; // [groups, b*s, rank] — wo_a_t precomputed at load
        let o = o
            .reshape((self.o_groups, b, s, self.o_lora_rank))?
            .transpose(0, 1)? // [b, groups, s, rank]
            .transpose(1, 2)? // [b, s, groups, rank]
            .reshape((b, s, self.o_groups * self.o_lora_rank))?
            .contiguous()?;
        self.wo_b.forward(&o.to_dtype(x.dtype())?)
    }
}

// ============================ compressor (1M-ctx KV compression) ============================

/// Streaming softmax-pool of the KV latent into compressed rows (model.py `Compressor`,
/// lines 316-377). Project KV + gate, window into `ratio` groups, softmax-pool over the
/// window (+ absolute-position `ape`), RMS-norm, trailing partial-RoPE. ratio-4 layers
/// use `overlap_transform` (coff==2) so each row pools across the window boundary;
/// ratio-128 layers don't (coff==1). `coff*head_dim` is read from the `ape` width, so
/// one struct serves both. Compressed rows are what Indexed layers attend to alongside
/// the raw sliding window — the bulk of long-context information.
pub(crate) struct Compressor {
    wkv: Tensor,   // [coff*head_dim, dim] F32
    wgate: Tensor, // [coff*head_dim, dim] F32
    norm: RmsNorm,
    ape: Tensor, // [ratio, coff*head_dim] F32
    rotary: Arc<DeepSeekV2RotaryEmbedding>,
    ratio: usize,
    head_dim: usize,
    rope_dim: usize,
    overlap: bool, // ratio == 4
}

impl Compressor {
    /// Compress `x [b,s,dim]` -> compressed KV `[b, n_comp, head_dim]`, trailing-RoPE'd.
    /// `n_comp = s / ratio` (drops the partial trailing window). Correct-by-reference to
    /// model.py:316-377 for the full-sequence (start_pos==0) path — which is what we call
    /// each step over the accumulated history (correctness-first; incremental state is a
    /// perf follow-on).
    fn compress(&self, x: &Tensor, positions: &Tensor) -> Result<Option<Tensor>> {
        let (b, s, _) = x.dims3()?;
        let cutoff = (s / self.ratio) * self.ratio;
        if cutoff == 0 {
            return Ok(None);
        }
        let n = cutoff / self.ratio;
        let coff_hd = self.ape.dim(D::Minus1)?;
        let xf = x.to_dtype(DType::F32)?;
        let kv = xf.broadcast_matmul(&self.wkv.t()?)?;
        let score = xf.broadcast_matmul(&self.wgate.t()?)?;
        let kv = kv
            .narrow(1, 0, cutoff)?
            .reshape((b, n, self.ratio, coff_hd))?;
        let score = score
            .narrow(1, 0, cutoff)?
            .reshape((b, n, self.ratio, coff_hd))?
            .broadcast_add(&self.ape.reshape((1, 1, self.ratio, coff_hd))?)?;
        let (kv, score) = if self.overlap {
            (
                overlap_transform(&kv, 0.0)?,
                overlap_transform(&score, f64::NEG_INFINITY)?,
            )
        } else {
            (kv, score)
        };
        let weights = hanzo_nn::ops::softmax_last_dim(&score.transpose(2, 3)?.contiguous()?)?
            .transpose(2, 3)?
            .contiguous()?;
        let pooled = kv.broadcast_mul(&weights)?.sum(2)?; // [b, n, head_dim]
        let pooled = self.norm.forward(&pooled.to_dtype(x.dtype())?)?;
        let hd = self.head_dim;
        let pooled = pooled.reshape((b, n, 1, hd))?.transpose(1, 2)?;
        let comp_pos = compressed_positions(n, self.ratio, positions)?;
        let pass = pooled
            .narrow(D::Minus1, 0, hd - self.rope_dim)?
            .contiguous()?;
        let rot = pooled
            .narrow(D::Minus1, hd - self.rope_dim, self.rope_dim)?
            .contiguous()?
            .to_dtype(DType::F32)?;
        let (rot, _) = self.rotary.forward_positions(&rot, &rot, &comp_pos)?;
        let rot = rot.to_dtype(pass.dtype())?;
        let pooled = Tensor::cat(&[&pass, &rot], D::Minus1)?
            .transpose(1, 2)?
            .reshape((b, n, hd))?;
        Ok(Some(pooled))
    }
}

/// model.py:307-314 `overlap_transform`: `[b,n,ratio,2d] -> [b,n,2*ratio,d]`, upper rows
/// = this window's tail, lower rows = the previous window's head (row 0 = `fill`).
fn overlap_transform(t: &Tensor, fill: f64) -> Result<Tensor> {
    let (b, n, ratio, two_d) = t.dims4()?;
    let d = two_d / 2;
    let dev = t.device();
    let second = t.narrow(D::Minus1, d, d)?;
    let first = t.narrow(D::Minus1, 0, d)?;
    let prev_first = if n > 1 {
        let shifted = first.narrow(1, 0, n - 1)?;
        let pad = Tensor::full(fill, (b, 1, ratio, d), dev)?.to_dtype(t.dtype())?;
        Tensor::cat(&[&pad, &shifted], 1)?
    } else {
        Tensor::full(fill, (b, 1, ratio, d), dev)?.to_dtype(t.dtype())?
    };
    Tensor::cat(&[&prev_first, &second], 2)
}

/// Compressed-row positions: row j covers tokens [j*ratio, (j+1)*ratio), rope'd at the
/// window-start stride (model.py `freqs_cis[:cutoff:ratio]`).
fn compressed_positions(n: usize, ratio: usize, positions: &Tensor) -> Result<Tensor> {
    let base = positions.to_dtype(DType::U32)?.to_vec1::<u32>()?;
    let start = base.first().copied().unwrap_or(0);
    let v: Vec<u32> = (0..n).map(|j| start + (j * ratio) as u32).collect();
    Tensor::from_vec(v, n, positions.device())
}

/// Per-layer compressor decode state: the layer-input history (chunked, `[b, s_i, dim]`
/// each) plus the incrementally-emitted compressed-KV rows. The history is chunked so
/// per-step appends are O(1) (no full-history copy); row emission cats only the small
/// slice it pools. State is SELF-HEALING under speculative rollback: every forward
/// calls [`Self::sync_to`] with the incoming absolute position, truncating any
/// history/rows contributed by rejected draft tokens (a row is kept only when its
/// whole `ratio` window is confirmed).
#[derive(Default)]
pub(crate) struct CompressorState {
    chunks: Vec<Tensor>, // each [b, s_i, dim]; sum(s_i) == len
    len: usize,
    comp_rows: Option<Tensor>, // [b, n_emitted, head_dim] — trailing-RoPE'd rows
    emitted: usize,
}

impl CompressorState {
    fn append(&mut self, x: &Tensor) -> Result<()> {
        self.len += x.dim(1)?;
        self.chunks.push(x.clone());
        Ok(())
    }

    fn reset(&mut self) {
        self.chunks.clear();
        self.len = 0;
        self.comp_rows = None;
        self.emitted = 0;
    }

    /// Truncate history + emitted rows to `base` confirmed tokens (rollback healing).
    fn sync_to(&mut self, base: usize, ratio: usize) -> Result<()> {
        while self.len > base {
            let last = self.chunks.pop().expect("len>0 implies chunks");
            let s = last.dim(1)?;
            if self.len - s >= base {
                self.len -= s;
            } else {
                let keep = base - (self.len - s);
                self.chunks.push(last.narrow(1, 0, keep)?);
                self.len = base;
            }
        }
        let max_rows = if ratio == 0 { 0 } else { base / ratio };
        if self.emitted > max_rows {
            self.comp_rows = match (&self.comp_rows, max_rows) {
                (_, 0) => None,
                (Some(r), n) => Some(r.narrow(1, 0, n)?),
                (None, _) => None,
            };
            self.emitted = max_rows;
        }
        Ok(())
    }

    /// Concatenate history rows `[start, end)` into one `[b, end-start, dim]` tensor.
    fn slice(&self, start: usize, end: usize) -> Result<Tensor> {
        let mut parts = Vec::new();
        let mut off = 0usize;
        for c in &self.chunks {
            let s = c.dim(1)?;
            let (c0, c1) = (off, off + s);
            off = c1;
            if c1 <= start || c0 >= end {
                continue;
            }
            let a = start.max(c0) - c0;
            let b = end.min(c1) - c0;
            parts.push(c.narrow(1, a, b - a)?);
        }
        if parts.len() == 1 {
            Ok(parts.pop().unwrap())
        } else {
            let refs: Vec<&Tensor> = parts.iter().collect();
            Tensor::cat(&refs, 1)
        }
    }

    /// Emit compressed rows up to `due` (== floor(confirmed_len / ratio)), pooling only
    /// the new rows' windows (+ the previous window for ratio-4 overlap semantics).
    fn emit_due(&mut self, comp: &Compressor, due: usize, device: &Device) -> Result<()> {
        while self.emitted < due {
            let j = self.emitted;
            let r = comp.ratio;
            let slice_start = j.saturating_sub(1) * r; // include window j-1 for overlap
            let slice_end = (j + 1) * r;
            let x = self.slice(slice_start, slice_end)?;
            let pos: Vec<u32> = (slice_start as u32..slice_end as u32).collect();
            let pos = Tensor::from_vec(pos, slice_end - slice_start, device)?;
            let rows = comp
                .compress(&x, &pos)?
                .ok_or_else(|| hanzo_ml::Error::Msg("compressor emitted no row".into()))?;
            let n = rows.dim(1)?;
            let newest = rows.narrow(1, n - 1, 1)?; // [b,1,hd]
            self.comp_rows = Some(match &self.comp_rows {
                None => newest,
                Some(rws) => Tensor::cat(&[rws, &newest], 1)?,
            });
            self.emitted += 1;
        }
        Ok(())
    }

    /// Seed the row cache from a full-prefill compress result.
    fn seed_rows(&mut self, rows: Option<&Tensor>, n: usize) {
        self.comp_rows = rows.cloned();
        self.emitted = n;
    }
}

/// Additive `[Lq, Lk + Nc]` mask for attention over `[raw_kv (Lk) ++ compressed (Nc)]`:
/// raw part is causal (query at abs-pos `p` sees raw `j <= p`); compressed part lets `p`
/// see row `jc` iff `jc < (p+1)/ratio` (the row's window is fully in the past). Matches
/// model.py `get_window_topk_idxs` (causal window) ++ `get_compress_topk_idxs`.
#[allow(clippy::too_many_arguments)]
fn combined_compressed_mask(
    q_positions: &[u32],
    raw_abs_start: usize,
    lk: usize,
    nc: usize,
    ratio: usize,
    window: Option<usize>,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let lq = q_positions.len();
    let neg = f32::NEG_INFINITY;
    let mut m = vec![0f32; lq * (lk + nc)];
    for (i, &p) in q_positions.iter().enumerate() {
        let p = p as usize;
        let row = i * (lk + nc);
        // Raw columns: absolute position `raw_abs_start + j`; visible iff within the
        // query's causal past AND its sliding window (model.py get_window_topk_idxs).
        let lo = window.map_or(0, |w| (p + 1).saturating_sub(w));
        for j in 0..lk {
            let abs = raw_abs_start + j;
            if abs > p || abs < lo {
                m[row + j] = neg;
            }
        }
        let visible = (p + 1) / ratio; // compressed rows 0..visible are in the past
        for jc in 0..nc {
            if jc >= visible {
                m[row + lk + jc] = neg;
            }
        }
    }
    Tensor::from_vec(m, (1, 1, lq, lk + nc), device)?.to_dtype(dtype)
}

// ============================ layer ============================

pub(crate) struct DecoderLayer {
    pub(crate) hc_attn: HyperConnections,
    pub(crate) attn_norm: RmsNorm,
    pub(crate) attn: V4Attn,
    pub(crate) hc_ffn: HyperConnections,
    pub(crate) ffn_norm: RmsNorm,
    pub(crate) moe: V4Moe,
}

impl DecoderLayer {
    /// Load one V4 decoder block (attn + MoE + per-sublayer Hyper-Connections) from
    /// `{p}.*` GGUF tensors. Shared by the main layer loop and the MTP head (which is
    /// itself one V4 block at `mtp.0`). `is_hash` selects hash-routed MoE (tid2eid) vs
    /// bias-routed; `compress_ratio == 0` (Full mode) means no compressor.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        p: &str,
        props: &PropsGGUF,
        dev: &Device,
        rotary: Arc<DeepSeekV2RotaryEmbedding>,
        compress_ratio: usize,
        group_in: usize,
        eps: f64,
        softmax_scale: f32,
        dtype: DType,
        is_hash: bool,
        swiglu_clamp: f32,
    ) -> Result<Self> {
        let is_full = compress_ratio == 0;
        // Attention grouped-o weight. Reshape to the ds4 row-major [groups, rank,
        // group_in] layout, then pre-transpose to the matmul-ready [groups, group_in,
        // rank] ONCE here — the forward previously re-transposed+copied this 32Mi bf16
        // constant every token (~33% of decode GPU time).
        let wo_a_t = ct
            .tensor(&format!("{p}.attn_output_a.weight"), dev)?
            .dequantize(dev)?
            .to_dtype(dtype)?
            .reshape((props.o_groups, props.o_lora_rank, group_in))?
            .transpose(1, 2)?
            .contiguous()?;
        let compressor = if is_full {
            None
        } else {
            Some(Compressor {
                wkv: deq(ct, &format!("{p}.attn_compressor_kv.weight"), dev)?,
                wgate: deq(ct, &format!("{p}.attn_compressor_gate.weight"), dev)?,
                norm: rms_from(
                    deq(ct, &format!("{p}.attn_compressor_norm.weight"), dev)?,
                    eps,
                )?,
                ape: deq(ct, &format!("{p}.attn_compressor_ape.weight"), dev)?,
                rotary: rotary.clone(),
                ratio: compress_ratio,
                head_dim: props.head_dim,
                rope_dim: props.rope_head_dim,
                overlap: compress_ratio == 4,
            })
        };
        let attn = V4Attn {
            q_a: gguf_linear(ct.tensor(&format!("{p}.attn_q_a.weight"), dev)?)?,
            q_a_norm: rms_from(deq(ct, &format!("{p}.attn_q_a_norm.weight"), dev)?, eps)?,
            q_b: gguf_linear(ct.tensor(&format!("{p}.attn_q_b.weight"), dev)?)?,
            kv: gguf_linear(ct.tensor(&format!("{p}.attn_kv.weight"), dev)?)?,
            kv_norm: rms_from(deq(ct, &format!("{p}.attn_kv_a_norm.weight"), dev)?, eps)?,
            wo_a_t,
            wo_b: gguf_linear(ct.tensor(&format!("{p}.attn_output_b.weight"), dev)?)?,
            rotary,
            sdpa: SdpaParams {
                n_kv_groups: props.head_count,
                softcap: None,
                softmax_scale,
                sliding_window: if is_full {
                    None
                } else {
                    Some(props.sliding_window)
                },
                sinks: Some(deq(ct, &format!("{p}.attn_sinks.weight"), dev)?),
            },
            n_head: props.head_count,
            head_dim: props.head_dim,
            rope_dim: props.rope_head_dim,
            o_groups: props.o_groups,
            o_lora_rank: props.o_lora_rank,
            rms_eps: eps,
            attn_dtype: dtype,
            compressor,
            compress_ratio: compress_ratio.max(1),
        };
        let moe = V4Moe {
            gate_inp: deq(ct, &format!("{p}.ffn_gate_inp.weight"), dev)?
                .t()?
                .contiguous()?,
            bias: if is_hash {
                None
            } else {
                Some(deq(ct, &format!("{p}.exp_probs_b.bias"), dev)?)
            },
            tid2eid: if is_hash {
                Some(
                    ct.tensor(&format!("{p}.ffn_gate_tid2eid.weight"), dev)?
                        .dequantize(dev)?,
                )
            } else {
                None
            },
            gate_experts: QMatMul::QTensor(Arc::new(
                ct.tensor(&format!("{p}.ffn_gate_exps.weight"), dev)?,
            )),
            up_experts: QMatMul::QTensor(Arc::new(
                ct.tensor(&format!("{p}.ffn_up_exps.weight"), dev)?,
            )),
            down_experts: QMatMul::QTensor(Arc::new(
                ct.tensor(&format!("{p}.ffn_down_exps.weight"), dev)?,
            )),
            shared_gate: gguf_linear(ct.tensor(&format!("{p}.ffn_gate_shexp.weight"), dev)?)?,
            shared_up: gguf_linear(ct.tensor(&format!("{p}.ffn_up_shexp.weight"), dev)?)?,
            shared_down: gguf_linear(ct.tensor(&format!("{p}.ffn_down_shexp.weight"), dev)?)?,
            topk: props.num_experts_per_tok,
            route_scale: props.expert_weights_scale,
            norm_topk: props.norm_topk_prob,
            swiglu_clamp,
        };
        let hc_attn = HyperConnections::from_parts(
            gguf_linear(ct.tensor(&format!("{p}.hc_attn_fn.weight"), dev)?)?,
            deq(ct, &format!("{p}.hc_attn_scale.weight"), dev)?.to_vec1::<f32>()?,
            deq(ct, &format!("{p}.hc_attn_base.weight"), dev)?,
            props.hc_count,
            props.hc_sinkhorn_iters,
            props.hc_eps,
            false,
        )?;
        let hc_ffn = HyperConnections::from_parts(
            gguf_linear(ct.tensor(&format!("{p}.hc_ffn_fn.weight"), dev)?)?,
            deq(ct, &format!("{p}.hc_ffn_scale.weight"), dev)?.to_vec1::<f32>()?,
            deq(ct, &format!("{p}.hc_ffn_base.weight"), dev)?,
            props.hc_count,
            props.hc_sinkhorn_iters,
            props.hc_eps,
            false,
        )?;
        Ok(DecoderLayer {
            hc_attn,
            attn_norm: rms_from(deq(ct, &format!("{p}.attn_norm.weight"), dev)?, eps)?,
            attn,
            hc_ffn,
            ffn_norm: rms_from(deq(ct, &format!("{p}.ffn_norm.weight"), dev)?, eps)?,
            moe,
        })
    }

    /// One layer over the HC carrier `[b, s, n_hc, e]`.
    pub(crate) fn forward(
        &self,
        hc: &Tensor,
        input_ids: &Tensor,
        mask: &AttentionMask,
        positions: &Tensor,
        kv_cache: &mut KvCache,
        comp_state: Option<&mut CompressorState>,
        is_prefill: bool,
    ) -> Result<Tensor> {
        // Attention sublayer (HC pre → norm → attn → HC post).
        let (xin, post, comb) = self.hc_attn.pre(hc)?;
        let attn = self.attn.forward(
            &self.attn_norm.forward(&xin)?,
            mask,
            positions,
            kv_cache,
            comp_state,
            is_prefill,
        )?;
        let hc = self.hc_attn.post(hc, &attn, &post, &comb)?;

        // FFN sublayer.
        let (xin, post, comb) = self.hc_ffn.pre(&hc)?;
        let ffn = self.moe.forward(&self.ffn_norm.forward(&xin)?, input_ids)?;
        self.hc_ffn.post(&hc, &ffn, &post, &comb)
    }
}

// ============================ model ============================

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<DecoderLayer>,
    output_hc: HyperConnections,
    norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    n_hc: usize,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    /// Per-layer compressor decode state (history of layer inputs), behind a Mutex like
    /// the KV cache. Reset at the start of each sequence (prefill, start_offset 0).
    comp_state: std::sync::Mutex<Vec<CompressorState>>,
    /// MTP self-speculative decode: when `store_spec_hidden` is set, the forward stashes
    /// the post-norm, per-sampled-row hidden state here (before the output projection) so
    /// the MTP proposer can read it. Same seam as gemma4's `last_spec_hidden`.
    spec_hidden: std::sync::Mutex<Option<Tensor>>,
    store_spec_hidden: std::sync::atomic::AtomicBool,
    /// Base-model config, retained so the MTP head (a V4 block sharing all hyperparams)
    /// can be loaded against it, and so the speculative wiring can read n_hc/head_dim.
    base_props: PropsGGUF,
}

pub(crate) fn rms_from(t: Tensor, eps: f64) -> Result<RmsNorm> {
    RmsNorm::from_w(t, eps)
}

/// Log the RMS of an activation at TRACE level — the per-layer numeric probe used
/// to localize forward bugs (a healthy residual grows smoothly; a discontinuity or
/// NaN pinpoints the broken layer). `enabled!` gates the (expensive, GPU-syncing)
/// norm computation so this is a single atomic load when tracing is off — and with
/// `tracing/release_max_level_info` it compiles to nothing in prod. Surface it with
/// `RUST_LOG=hanzo_engine::models::quantized_deepseek4=trace`. No more add/remove.
#[inline]
fn trace_rms(t: &Tensor, what: std::fmt::Arguments) {
    if tracing::enabled!(tracing::Level::TRACE) {
        let rms = t
            .to_dtype(DType::F32)
            .and_then(|t| t.sqr())
            .and_then(|t| t.mean_all())
            .and_then(|t| t.to_scalar::<f32>())
            .map(|m| m.sqrt())
            .unwrap_or(f32::NAN);
        tracing::trace!(rms, "{what}");
    }
}

/// Layers whose (stream-summed) Hyper-Connection carrier is captured for the
/// DSpark draft-head training cache — even spacing over V4-Flash's 43 blocks,
/// mirroring DeepSpec's `[1, 9, 17, 25, 33]` for 36 layers. See `write_capture`
/// and `examples/v4_cache_dump`.
const V4_CAPTURE_LAYERS: [usize; 5] = [1, 11, 21, 31, 41];

/// FNV-1a 64-bit over the token ids — a process-stable content hash for the
/// capture sidecar (identifies a sample independent of SipHash's per-process seed).
fn fnv1a_ids(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &v in ids {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        _attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        verify_arch(ct.get_metadata())?;

        // DeepSeek-V4 is W8A8 QAT-trained: it needs the int8 dp4a matmul path (Q8
        // activation × Q8_0/Q2K weight, int32 accumulate) for numeric fidelity. The
        // default W8A16 path (dequant weight × full-precision activation) drifts from
        // the trained graph and degenerates long open-ended generation. Enable the
        // fast mmq/mmvq path for this model (no-op without the cuda feature).
        #[cfg(feature = "cuda")]
        hanzo_ml::quantized::cuda::set_fast_mmq(true);

        // ── V4 DTYPE POLICY (one compute dtype, threaded — not pinned) ─────────────
        // The model HONORS the caller's `dtype` (the pipeline auto-selects it, and
        // allocates the KV cache + causal mask at it). We thread that one value to
        // every consumer — KV cache, mask, each V4Attn (`attn_dtype`), the embeds
        // entry, wo_a — so the whole forward agrees with the pipeline-owned cache.
        // The cascade of dtype bugs came from *disagreement* (model F32 vs cache
        // BF16), NOT from any particular dtype: consistency is the fix, threading is
        // how. Quant weights dequantize to it at the QMatMul boundary; precision-
        // sensitive ops (RMS-norm/RoPE/softmax-sinks) upcast to F32 internally and
        // return it. For ds4 NUMERIC parity, the *benchmark* requests F32 at the
        // pipeline (so cache + model are both F32) — that's a caller choice, not a
        // model hardcode.
        let props = PropsGGUF::try_from(ContentMetadata {
            path_prefix: "deepseek4",
            metadata: ct.get_metadata(),
        })
        .or_else(|err| hanzo_ml::bail!("{err}"))?;

        let eps = props.rms_norm_eps as f64;
        let softmax_scale = 1.0 / (props.head_dim as f32).sqrt();

        let tok = ct.tensor("token_embd.weight", device)?.dequantize(device)?;
        let norm = rms_from(deq(&mut ct, "output_norm.weight", device)?, eps)?;
        let output = gguf_linear(ct.tensor("output.weight", device)?)?;

        // Output Hyper-Connection (reduce-only).
        let output_hc = HyperConnections::from_parts(
            gguf_linear(ct.tensor("output_hc_fn.weight", device)?)?,
            deq(&mut ct, "output_hc_scale.weight", device)?.to_vec1::<f32>()?,
            deq(&mut ct, "output_hc_base.weight", device)?,
            props.hc_count,
            props.hc_sinkhorn_iters,
            props.hc_eps,
            true,
        )?;

        // Per-layer RoPE: Full (θ=rope_theta, no YaRN) vs compressed (θ=compress, YaRN).
        let mk_rope =
            |theta: f32, yarn: bool, dev: &Device| -> Result<Arc<DeepSeekV2RotaryEmbedding>> {
                let cfg = DeepSeekV2RopeConfig {
                    rope_scaling: if yarn {
                        Some(DeepSeekV2RopeScaling::Yarn {
                            original_max_position_embeddings: props.rope_orig_ctx,
                            beta_fast: props.yarn_beta_fast,
                            beta_slow: props.yarn_beta_slow,
                            factor: props.rope_scaling_factor,
                            mscale: 1.0,
                            mscale_all_dim: 1.0,
                            scaling_type: ScaledRopeType::Yarn,
                        })
                    } else {
                        None
                    },
                    max_position_embeddings: props.max_seq_len,
                    rope_theta: theta,
                    qk_rope_head_dim: props.rope_head_dim,
                };
                Ok(Arc::new(DeepSeekV2RotaryEmbedding::new(
                    &cfg,
                    DType::F32,
                    dev,
                )?))
            };
        let mut ropes_full = HashMap::new();
        let mut ropes_comp = HashMap::new();
        for i in 0..props.block_count {
            let dev = mapper.device_for(i, false).unwrap_or(device);
            ropes_full
                .entry(dev.location())
                .or_insert(mk_rope(props.rope_theta, false, dev)?);
            ropes_comp.entry(dev.location()).or_insert(mk_rope(
                props.compress_rope_theta,
                true,
                dev,
            )?);
        }

        let group_in = props.head_count * props.head_dim / props.o_groups;

        let mut layers = Vec::with_capacity(props.block_count);
        for i in NiceProgressBar::<_, 'b'>(
            0..props.block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let p = format!("blk.{i}");
            let dev = mapper.device_for(i, false).unwrap_or(device);
            let mode = layer_mode(&props.compress_ratios, i, props.sliding_window);
            // Diagnostic (env V4_FORCE_FULL_ROPE=1): give EVERY layer the Full rope map
            // (θ=rope_theta, no YaRN). Numerically wrong on compressed layers by design —
            // used to discriminate whether the layer-2+ drift vs ds4 comes from the
            // compressed rope path (drift signature changes) or elsewhere (identical).
            static FORCE_FULL_ROPE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let force_full = *FORCE_FULL_ROPE
                .get_or_init(|| std::env::var("V4_FORCE_FULL_ROPE").is_ok_and(|v| v == "1"));
            let rotary = if mode == Mode::Full || force_full {
                ropes_full[&dev.location()].clone()
            } else {
                ropes_comp[&dev.location()].clone()
            };

            // One V4 decoder block via the shared loader (same code path as the MTP head).
            let ratio = props.compress_ratios.get(i).copied().unwrap_or(0) as usize;
            layers.push(DecoderLayer::load(
                &mut ct,
                &p,
                &props,
                dev,
                rotary,
                ratio,
                group_in,
                eps,
                softmax_scale,
                dtype,
                i < props.hash_layer_count,
                props.swiglu_clamp.get(i).copied().unwrap_or(0.0),
            )?);
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok, props.embedding_length),
            layers,
            output_hc,
            norm,
            output,
            n_hc: props.hc_count,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(props.block_count, props.max_seq_len)),
            max_seq_len: props.max_seq_len,
            mapper: Some(mapper),
            dtype,
            comp_state: std::sync::Mutex::new(
                (0..props.block_count)
                    .map(|_| CompressorState::default())
                    .collect(),
            ),
            spec_hidden: std::sync::Mutex::new(None),
            store_spec_hidden: std::sync::atomic::AtomicBool::new(false),
            base_props: props.clone(),
        })
    }
}

impl ModelWeights {
    /// Base-model token embeddings (shared with the MTP head).
    pub fn embeddings(&self) -> &Embedding {
        &self.tok_embeddings
    }

    /// Base-model output projection (shared with the MTP head).
    pub fn output_head(&self) -> Arc<dyn QuantMethod> {
        self.output.clone()
    }

    /// Base-model config, for loading the MTP head against it.
    pub fn base_props(&self) -> &PropsGGUF {
        &self.base_props
    }

    /// The model's compute dtype (the MTP head loads at the same dtype).
    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    /// Shared handle to the normal KV cache, for `NormalSpeculativeCacheAccess`.
    pub fn normal_cache_arc(
        &self,
    ) -> Option<std::sync::Arc<std::sync::Mutex<crate::pipeline::NormalCache>>> {
        self.cache.normal_arc()
    }

    /// Enable/disable stashing the pre-output hidden state for MTP speculative
    /// proposal. Clearing also drops any stashed tensor so a finished step can't leak.
    pub fn set_store_spec_hidden(&self, store: bool) {
        self.store_spec_hidden
            .store(store, std::sync::atomic::Ordering::Relaxed);
        if !store {
            if let Ok(mut h) = self.spec_hidden.lock() {
                *h = None;
            }
        }
    }

    /// The hidden state stashed by the most recent forward (post-norm, per sampled row,
    /// pre output projection), if capture was enabled. Cloned; the stash is retained.
    pub fn last_spec_hidden(&self) -> Option<Tensor> {
        self.spec_hidden.lock().ok().and_then(|h| h.clone())
    }

    /// Write one teacher-forced sample's hidden-state shards to `dir` (env
    /// `V4_CAPTURE_DIR`). ALL capture I/O lives here so the caller stays a thin
    /// driver. `carriers` are the per-`V4_CAPTURE_LAYERS` stream-summed carriers,
    /// each `[1, s, E]`; `last` is the post-final-norm hidden `[1, s, E]`. Emits
    /// `<dir>/sample_<n>.safetensors` with tensors `input_ids [s] u32`,
    /// `target_hidden_states [s, 5, E] bf16`, `target_last_hidden_states [s, E]
    /// bf16`, and appends one line to `<dir>/index.jsonl`:
    /// `{"idx","file","seq_len","ids_hash"}`. A shared `AtomicUsize` names the
    /// sample so the driver's Nth request maps to `sample_<N>`.
    fn write_capture(
        &self,
        dir: &str,
        input_ids: &Tensor,
        carriers: &[Tensor],
        last: &Tensor,
    ) -> Result<()> {
        use std::io::Write as _;
        let ids = input_ids.squeeze(0)?.to_dtype(DType::U32)?; // [s]
        let seq_len = ids.dim(0)?;
        // [s, 5, E] bf16: stack the stream-summed carriers along a new axis 1.
        let planes = carriers
            .iter()
            .map(|c| c.squeeze(0))
            .collect::<Result<Vec<_>>>()?;
        let hidden = Tensor::stack(&planes, 1)?.to_dtype(DType::BF16)?;
        let last = last.squeeze(0)?.to_dtype(DType::BF16)?; // [s, E]

        let mut shards = HashMap::new();
        shards.insert("input_ids".to_string(), ids.clone());
        shards.insert("target_hidden_states".to_string(), hidden);
        shards.insert("target_last_hidden_states".to_string(), last);

        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let idx = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let file = format!("sample_{idx}.safetensors");
        let dir_path = std::path::Path::new(dir);
        std::fs::create_dir_all(dir_path).map_err(hanzo_ml::Error::wrap)?;
        hanzo_ml::safetensors::save(&shards, dir_path.join(&file))?;

        let ids_hash = fnv1a_ids(&ids.to_vec1::<u32>()?);
        let mut index = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir_path.join("index.jsonl"))
            .map_err(hanzo_ml::Error::wrap)?;
        writeln!(
            index,
            "{{\"idx\":{idx},\"file\":\"{file}\",\"seq_len\":{seq_len},\"ids_hash\":\"{ids_hash:016x}\"}}"
        )
        .map_err(hanzo_ml::Error::wrap)?;
        Ok(())
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (_b, _s) = input_ids.dims2()?;
        // Enter the single compute dtype HERE (the one entry point for activations):
        // the embeddings carry the whole HC carrier downstream, so casting them to
        // `self.dtype` makes embeds → carrier → attn → cache → output all flow in the
        // one dtype. (Decomplected: dtype is applied at the boundary where data enters
        // the model, then never re-decided.)
        let embeds = self
            .tok_embeddings
            .forward(input_ids)?
            .to_dtype(self.dtype)?;
        let cache = &mut self.cache.normal().0;
        let mask = CausalMasker.make_causal_mask(
            input_ids,
            cache as &dyn PastKvLenCache,
            self.dtype,
            &CausalMaskConfig::gguf(),
        )?;
        let mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(mask)
        };
        let positions = Tensor::from_vec(
            start_offsets.iter().map(|&o| o as u32).collect::<Vec<_>>(),
            start_offsets.len(),
            &self.device,
        )?;

        // Compressor decode state: reset at the start of a fresh sequence (any seq
        // starting at offset 0 — a prefill), then accumulate per layer across steps.
        // `is_prefill` also gates the compressor itself: it runs on the prefill only,
        // NOT on a multi-token decode (a speculative verify forward has s>1 but is a
        // decode continuation — gating on `s>1` alone would wrongly re-run it).
        let is_prefill = start_offsets.iter().any(|&o| o == 0);
        let mut comp = self.comp_state.lock().unwrap();
        if is_prefill {
            for s in comp.iter_mut() {
                s.reset();
            }
        }

        // V4 training-cache capture (env V4_CAPTURE_DIR): teacher-forced per-position
        // hidden-state dump for the DSpark draft-head. Prefill-only and single-
        // sequence; a single OnceLock load when the var is unset → zero cost off-path.
        static CAPTURE_DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let capture_dir = CAPTURE_DIR
            .get_or_init(|| {
                std::env::var("V4_CAPTURE_DIR")
                    .ok()
                    .filter(|s| !s.is_empty())
            })
            .as_deref();
        // `> 1` skips the load-time single-token warmup prefill (the engine runs one
        // 1-token forward at offset 0 to prime caches; seqlen 1 has no teacher-forcing
        // signal). Real samples are many tokens.
        let capturing = capture_dir.is_some() && is_prefill && input_ids.dim(1)? > 1;
        if capturing && input_ids.dim(0)? != 1 {
            hanzo_ml::bail!(
                "V4_CAPTURE_DIR requires a single sequence (max_num_seqs=1); got batch {}",
                input_ids.dim(0)?
            );
        }

        // Per-layer numeric trace (env V4_TRACE_LAYERS=1, prefill only): last-position
        // carrier stream-sum L2 + stream-0 L2 + first-4 values per layer — the hanzo
        // half of the ds4 divergence bisection (pairs with ds4's DS4_TRACE_LAYERS).
        static TRACE_LAYERS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let trace_layers = *TRACE_LAYERS
            .get_or_init(|| std::env::var("V4_TRACE_LAYERS").is_ok_and(|v| v == "1"))
            && is_prefill;
        let layer_trace = |tag: &str, t: &Tensor| -> Result<()> {
            // t: [B, s, n_hc, E] carrier or [B, s, E] plain hidden — dump last position.
            let s_len = t.dim(1)?;
            let last = t.narrow(1, s_len - 1, 1)?.squeeze(1)?.squeeze(0)?;
            let (summed, s0norm) = if last.dims().len() == 2 {
                let s0 = last.narrow(0, 0, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
                let s0n = s0.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                (last.sum(0)?.to_dtype(DType::F32)?, s0n)
            } else {
                let x = last.to_dtype(DType::F32)?;
                let n = x.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                (x, n)
            };
            let norm = summed.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let v: Vec<f32> = summed.narrow(0, 0, 4)?.to_vec1()?;
            eprintln!(
                "HZTRACE layer={tag} norm={norm:.6} s0norm={s0norm:.6} h0={:.6} h1={:.6} h2={:.6} h3={:.6}",
                v[0], v[1], v[2], v[3]
            );
            Ok(())
        };
        if trace_layers {
            layer_trace("-1", &embeds)?;
        }

        // Expand to the HC carrier, thread through layers, reduce at output.
        // (`captured` stays empty — no allocation — unless V4_CAPTURE_DIR is set.)
        let mut captured: Vec<Tensor> = Vec::new();
        let mut hc = HyperConnections::expand(&embeds, self.n_hc)?;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                hc = mapper.map(hc, i)?;
            }
            hc = layer.forward(
                &hc,
                input_ids,
                &mask.get(hc.device()),
                &positions,
                &mut cache[i],
                Some(&mut comp[i]),
                is_prefill,
            )?;
            trace_rms(&hc, format_args!("v4 carrier after layer {i}"));
            if trace_layers {
                layer_trace(&i.to_string(), &hc)?;
            }
            // V4 training-cache: stash this layer's carrier summed across the HC
            // streams -> [1, s, E] (full sequence), at the sampled layers only.
            if capturing && V4_CAPTURE_LAYERS.contains(&i) {
                captured.push(hc.sum(2)?);
            }
        }
        drop(comp);
        let x = self.output_hc.reduce_output(&hc)?;
        trace_rms(&x, format_args!("v4 reduce_output"));
        let x = self.norm.forward(&x)?;
        if trace_layers {
            layer_trace("99", &x)?;
        }
        // V4 training-cache: `x` here is the post-final-norm FULL-sequence hidden
        // [1, s, E] — captured BEFORE extract_logits narrows it to sampled rows.
        // Both the per-layer carriers and this norm are now in hand, so emit the
        // sample's shards. `capturing` guarantees prefill-only + single-sequence.
        if capturing {
            self.write_capture(
                capture_dir.expect("capturing implies V4_CAPTURE_DIR set"),
                input_ids,
                &captured,
                &x,
            )?;
        }
        let x = extract_logits(&x, context_lens)?;
        // MTP: stash the per-sampled-row hidden (post-norm, pre-output) for the proposer.
        if self
            .store_spec_hidden
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            if let Ok(mut h) = self.spec_hidden.lock() {
                *h = Some(x.clone());
            }
        }
        self.output.forward(&x.contiguous()?)
    }
}
