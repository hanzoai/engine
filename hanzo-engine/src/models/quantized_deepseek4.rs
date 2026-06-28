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
pub(crate) struct PropsGGUF {
    pub block_count: usize,
    pub embedding_length: usize,
    pub head_count: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    // MLA / attention.
    pub head_dim: usize,        // 512 (key_length == value_length)
    pub rope_head_dim: usize,   // 64
    pub q_lora_rank: usize,     // 1024
    pub o_lora_rank: usize,     // 1024
    pub o_groups: usize,        // 8
    pub sliding_window: usize,  // 128
    // RoPE.
    pub rope_theta: f32,            // 10000 (Full layers)
    pub compress_rope_theta: f32,   // 160000 (compressed layers)
    pub rope_scaling_factor: f32,   // 16
    pub rope_orig_ctx: usize,       // 65536
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub compress_ratios: Vec<u32>,  // per-layer schedule (0 = Full)
    // MoE.
    pub n_routed_experts: usize,    // 256
    pub num_experts_per_tok: usize, // 6
    pub n_shared_experts: usize,    // 1
    pub expert_ff_len: usize,       // 2048
    pub expert_weights_scale: f64,  // 1.5 (route_scale)
    pub norm_topk_prob: bool,
    pub hash_layer_count: usize,    // 3
    pub swiglu_clamp: Vec<f32>,     // per-layer SwiGLU clamp (0 = none)
    // Hyper-Connections.
    pub hc_count: usize,            // 4
    pub hc_sinkhorn_iters: usize,   // 20
    pub hc_eps: f64,                // 0
}

fn verify_arch(meta: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>) -> Result<()> {
    use crate::utils::gguf_metadata::TryValueInto;
    let arch: String = meta
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
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
            sliding_window: c.get_value::<u32>("attention.sliding_window").unwrap_or(128) as usize,
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
            compress_ratios: c
                .get_value::<Vec<u32>>("attention.compress_ratios")
                .unwrap_or_default(),
            n_routed_experts: c.get_value::<u32>("expert_count")? as usize,
            num_experts_per_tok: c.get_value::<u32>("expert_used_count")? as usize,
            n_shared_experts: c.get_value::<u32>("expert_shared_count").unwrap_or(1) as usize,
            expert_ff_len: c.get_value::<u32>("expert_feed_forward_length")? as usize,
            expert_weights_scale: c
                .get_value::<f32>("expert_weights_scale")
                .unwrap_or(1.0) as f64,
            norm_topk_prob: c.get_value::<bool>("expert_weights_norm").unwrap_or(true),
            hash_layer_count: c.get_value::<u32>("hash_layer_count").unwrap_or(0) as usize,
            swiglu_clamp: c
                .get_value::<Vec<f32>>("swiglu_clamp_exp")
                .unwrap_or_default(),
            hc_count: c.get_value::<u32>("hyper_connection.count").unwrap_or(1) as usize,
            hc_sinkhorn_iters: c
                .get_value::<u32>("hyper_connection.sinkhorn_iterations")
                .unwrap_or(20) as usize,
            hc_eps: c.get_value::<f32>("hyper_connection.epsilon").unwrap_or(0.0) as f64,
        })
    }
}

// ============================ helpers ============================

fn gguf_linear(q: hanzo_ml::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(q),
        b: None,
    })?))
}

/// Dequantize a GGUF tensor to a plain F32 tensor (norms, sinks, bias, gate_inp,
/// hc base/scale, and the grouped-o `wo_a`).
fn deq(ct: &mut Content<'_, impl std::io::Seek + std::io::Read>, name: &str, dev: &Device) -> Result<Tensor> {
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
struct V4Moe {
    gate_inp: Tensor,          // [hidden, n_experts] F32 (router weight, applied as x·W)
    bias: Option<Tensor>,      // [n_experts] F32 (exp_probs_b) — MoE layers
    tid2eid: Option<Tensor>,   // [used, vocab] I32 — hash layers
    gate_experts: QMatMul,
    up_experts: QMatMul,
    down_experts: QMatMul,
    shared_gate: Arc<dyn QuantMethod>,
    shared_up: Arc<dyn QuantMethod>,
    shared_down: Arc<dyn QuantMethod>,
    topk: usize,
    route_scale: f64,
    norm_topk: bool,
    swiglu_clamp: f32,
}

impl V4Moe {
    /// Routing weights+indices. `input_ids` (the token ids for this position) drive
    /// hash routing; `xs` is `[tokens, hidden]`.
    fn route(&self, xs: &Tensor, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        // scores = sqrt(softplus(xs @ gate_inp)).
        let logits = xs
            .to_dtype(DType::F32)?
            .matmul(&self.gate_inp)?; // [t, n_experts]
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
            (
                gate.clamp(f64::NEG_INFINITY, c)?,
                up.clamp(-c, c)?,
            )
        } else {
            (gate.clone(), up.clone())
        };
        crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)
    }

    fn forward(&self, xs: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
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
        (routed + shared)?
            .reshape((b, s, h))?
            .to_dtype(orig_dtype)
    }
}

// ============================ attention ============================

struct V4Attn {
    q_a: Arc<dyn QuantMethod>,
    q_a_norm: RmsNorm,
    q_b: Arc<dyn QuantMethod>,
    kv: Arc<dyn QuantMethod>,
    kv_norm: RmsNorm,
    wo_a: Tensor, // [groups, o_lora_rank, group_in] F32 for the grouped einsum
    wo_b: Arc<dyn QuantMethod>,
    rotary: Arc<DeepSeekV2RotaryEmbedding>,
    sdpa: SdpaParams,
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    o_groups: usize,
    o_lora_rank: usize,
    rms_eps: f64,
    attn_dtype: DType, // model/cache/mask dtype (BF16 on CUDA); q/k/v cast to it for SDPA + cache
    /// Present on compressed (Indexed/Sliding) layers: builds the compressed-KV rows the
    /// layer attends to alongside the raw window. `compress_ratio` is the window stride.
    compressor: Option<Compressor>,
    compress_ratio: usize,
}

impl V4Attn {
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
        let rot = if inverse {
            self.rotary.forward_inverse_positions(&rot, positions)?
        } else {
            // forward_positions takes (q, k); reuse with a dummy second arg.
            let (rot, _) = self.rotary.forward_positions(&rot, &rot, positions)?;
            rot
        };
        let rot = rot.to_dtype(orig)?;
        Tensor::cat(&[&pass, &rot], D::Minus1)?.contiguous()
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        positions: &Tensor,
        kv_cache: &mut KvCache,
        comp_state: Option<&mut CompressorState>,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        // Working dtype for attention + KV cache (the model dtype; BF16 on CUDA) —
        // the cache buffer + causal mask are allocated at this dtype, NOT the (F32)
        // carrier dtype. The QMatMul projections + RMS-norm + RoPE run in F32 for
        // precision; q/kv are cast to `attn_dtype` before SDPA + cache append so all
        // of q/k/v, the mask, and the cache buffer agree.
        let wdt = self.attn_dtype;

        // q: wq_a → q_norm → wq_b → [b, nh, s, hd] → per-head RMS → trailing rope.
        let q = self.q_b.forward(&self.q_a_norm.forward(&self.q_a.forward(x)?)?)?;
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
        let kv = self.rope(&kv, positions, false)?.to_dtype(wdt)?;

        let (k, v) = kv_cache.append(&kv, &kv)?;

        // Compressed layers also attend to the compressor's compressed-KV rows (the
        // bulk of long-context info). Build them from the accumulated layer-input
        // history and concat to the raw window; a combined mask keeps both causal.
        // (seqlen<window: the raw window is fully causal-visible, the indexer selects
        // all rows since n_comp < top_k=512 — so dense over [raw ++ all-compressed] is
        // exact. Sliding window for >window context + the indexer top-k are follow-ons.)
        let attn = match (&self.compressor, comp_state) {
            (Some(comp), Some(state)) => {
                state.append(x)?;
                let xh = state.x_history.as_ref().unwrap();
                let total = xh.dim(1)?;
                let hist_pos = Tensor::arange(0u32, total as u32, xh.device())?;
                match comp.compress(xh, &hist_pos)? {
                    Some(kvc) => {
                        let nc = kvc.dim(1)?;
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
                            lk,
                            nc,
                            self.compress_ratio,
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
            // Full layers (no compressor) — dense sliding-window + sinks attention.
            _ => Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa)?,
        };

        // Inverse-RoPE on the output (absorbed: the V latent carried RoPE).
        let attn = self.rope(&attn, positions, true)?;

        // Grouped o-LoRA: o[b,s,groups,group_in] einsum wo_a[groups,rank,group_in]
        // -> [b,s,groups,rank]; then wo_b(flatten).
        let group_in = self.n_head * self.head_dim / self.o_groups;
        let o = attn
            .transpose(1, 2)? // [b, s, nh, hd]
            .reshape((b, s, self.o_groups, group_in))?
            .to_dtype(self.wo_a.dtype())?;
        // batched per-group matmul: [groups, b*s, group_in] @ [groups, group_in, rank].
        let o = o
            .transpose(1, 2)? // [b, groups, s, group_in]
            .transpose(0, 1)? // [groups, b, s, group_in]
            .reshape((self.o_groups, b * s, group_in))?
            .contiguous()?;
        let wo_a_t = self.wo_a.transpose(1, 2)?.contiguous()?; // [groups, group_in, rank]
        let o = o.matmul(&wo_a_t)?; // [groups, b*s, rank]
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
struct Compressor {
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
        let kv = kv.narrow(1, 0, cutoff)?.reshape((b, n, self.ratio, coff_hd))?;
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
        let pass = pooled.narrow(D::Minus1, 0, hd - self.rope_dim)?.contiguous()?;
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

/// Per-layer compressor decode state: the accumulated layer-input history `[b, seq, dim]`.
/// Reset at the start of each sequence; appended every forward. `compress()` is rebuilt
/// over the full history each step (correctness-first).
#[derive(Default)]
struct CompressorState {
    x_history: Option<Tensor>,
}

impl CompressorState {
    fn append(&mut self, x: &Tensor) -> Result<()> {
        self.x_history = Some(match &self.x_history {
            None => x.clone(),
            Some(h) => Tensor::cat(&[h, x], 1)?,
        });
        Ok(())
    }
}

/// Additive `[Lq, Lk + Nc]` mask for attention over `[raw_kv (Lk) ++ compressed (Nc)]`:
/// raw part is causal (query at abs-pos `p` sees raw `j <= p`); compressed part lets `p`
/// see row `jc` iff `jc < (p+1)/ratio` (the row's window is fully in the past). Matches
/// model.py `get_window_topk_idxs` (causal window) ++ `get_compress_topk_idxs`.
fn combined_compressed_mask(
    q_positions: &[u32],
    lk: usize,
    nc: usize,
    ratio: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let lq = q_positions.len();
    let neg = f32::NEG_INFINITY;
    let mut m = vec![0f32; lq * (lk + nc)];
    for (i, &p) in q_positions.iter().enumerate() {
        let p = p as usize;
        let row = i * (lk + nc);
        for j in 0..lk {
            if j > p {
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

struct DecoderLayer {
    hc_attn: HyperConnections,
    attn_norm: RmsNorm,
    attn: V4Attn,
    hc_ffn: HyperConnections,
    ffn_norm: RmsNorm,
    moe: V4Moe,
}

impl DecoderLayer {
    /// One layer over the HC carrier `[b, s, n_hc, e]`.
    fn forward(
        &self,
        hc: &Tensor,
        input_ids: &Tensor,
        mask: &AttentionMask,
        positions: &Tensor,
        kv_cache: &mut KvCache,
        comp_state: Option<&mut CompressorState>,
    ) -> Result<Tensor> {
        // Attention sublayer (HC pre → norm → attn → HC post).
        let (xin, post, comb) = self.hc_attn.pre(hc)?;
        let attn = self.attn.forward(
            &self.attn_norm.forward(&xin)?,
            mask,
            positions,
            kv_cache,
            comp_state,
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
}

fn rms_from(t: Tensor, eps: f64) -> Result<RmsNorm> {
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

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        _attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        verify_arch(ct.get_metadata())?;

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
        let mk_rope = |theta: f32, yarn: bool, dev: &Device| -> Result<Arc<DeepSeekV2RotaryEmbedding>> {
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
            Ok(Arc::new(DeepSeekV2RotaryEmbedding::new(&cfg, DType::F32, dev)?))
        };
        let mut ropes_full = HashMap::new();
        let mut ropes_comp = HashMap::new();
        for i in 0..props.block_count {
            let dev = mapper.device_for(i, false).unwrap_or(device);
            ropes_full
                .entry(dev.location())
                .or_insert(mk_rope(props.rope_theta, false, dev)?);
            ropes_comp
                .entry(dev.location())
                .or_insert(mk_rope(props.compress_rope_theta, true, dev)?);
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
            let rotary = if mode == Mode::Full {
                ropes_full[&dev.location()].clone()
            } else {
                ropes_comp[&dev.location()].clone()
            };

            // Attention. wo_a dequantized to the single compute `dtype` (the grouped-o
            // einsum operand) — uniform with the rest of the model; no special-case
            // BF16 that could disagree at a dtype boundary.
            // GGUF stores output_a as file dims [group_in, groups*rank]; the reader
            // reverses to candle [groups*rank, group_in]. ds4.c reads row gr=g*rank+r
            // as `out[gr] = Σ_d A[gr][d]·heads[g][d]`, so the rows split groups-major:
            // reshape DIRECTLY to [groups, rank, group_in] (row gr -> [g=gr/rank, r=gr%rank]).
            // (A prior reshape((group_in,groups,rank)).permute scrambled the buffer —
            // it reinterpreted contiguous memory wrong, corrupting the o-projection.)
            let wo_a = ct
                .tensor(&format!("{p}.attn_output_a.weight"), dev)?
                .dequantize(dev)?
                .to_dtype(dtype)?
                .reshape((props.o_groups, props.o_lora_rank, group_in))?
                .contiguous()?;
            // Compressor (compressed layers only): projects the layer input into
            // compressed KV rows the attention sees alongside the raw window. wkv/wgate
            // load as candle [coff*head_dim, dim], ape as [ratio, coff*head_dim] (coff
            // read from ape width: 2 for ratio-4 / overlap, 1 for ratio-128). Shares the
            // compressed RoPE (θ=160000 + YaRN).
            let ratio = props.compress_ratios.get(i).copied().unwrap_or(0) as usize;
            let compressor = if mode == Mode::Full {
                None
            } else {
                Some(Compressor {
                    wkv: deq(&mut ct, &format!("{p}.attn_compressor_kv.weight"), dev)?,
                    wgate: deq(&mut ct, &format!("{p}.attn_compressor_gate.weight"), dev)?,
                    norm: rms_from(
                        deq(&mut ct, &format!("{p}.attn_compressor_norm.weight"), dev)?,
                        eps,
                    )?,
                    ape: deq(&mut ct, &format!("{p}.attn_compressor_ape.weight"), dev)?,
                    rotary: rotary.clone(),
                    ratio,
                    head_dim: props.head_dim,
                    rope_dim: props.rope_head_dim,
                    overlap: ratio == 4,
                })
            };

            let attn = V4Attn {
                q_a: gguf_linear(ct.tensor(&format!("{p}.attn_q_a.weight"), dev)?)?,
                q_a_norm: rms_from(deq(&mut ct, &format!("{p}.attn_q_a_norm.weight"), dev)?, eps)?,
                q_b: gguf_linear(ct.tensor(&format!("{p}.attn_q_b.weight"), dev)?)?,
                kv: gguf_linear(ct.tensor(&format!("{p}.attn_kv.weight"), dev)?)?,
                kv_norm: rms_from(deq(&mut ct, &format!("{p}.attn_kv_a_norm.weight"), dev)?, eps)?,
                wo_a,
                wo_b: gguf_linear(ct.tensor(&format!("{p}.attn_output_b.weight"), dev)?)?,
                rotary,
                sdpa: SdpaParams {
                    n_kv_groups: props.head_count,
                    softcap: None,
                    softmax_scale,
                    sliding_window: if mode == Mode::Full { None } else { Some(props.sliding_window) },
                    sinks: Some(deq(&mut ct, &format!("{p}.attn_sinks.weight"), dev)?),
                },
                n_head: props.head_count,
                head_dim: props.head_dim,
                rope_dim: props.rope_head_dim,
                o_groups: props.o_groups,
                o_lora_rank: props.o_lora_rank,
                rms_eps: eps,
                attn_dtype: dtype,
                compressor,
                compress_ratio: ratio.max(1),
            };

            // MoE.
            let is_hash = i < props.hash_layer_count;
            let moe = V4Moe {
                gate_inp: deq(&mut ct, &format!("{p}.ffn_gate_inp.weight"), dev)?.t()?.contiguous()?,
                bias: if is_hash {
                    None
                } else {
                    Some(deq(&mut ct, &format!("{p}.exp_probs_b.bias"), dev)?)
                },
                tid2eid: if is_hash {
                    Some(ct.tensor(&format!("{p}.ffn_gate_tid2eid.weight"), dev)?.dequantize(dev)?)
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
                swiglu_clamp: props.swiglu_clamp.get(i).copied().unwrap_or(0.0),
            };

            // Hyper-Connections (per sublayer).
            let hc_attn = HyperConnections::from_parts(
                gguf_linear(ct.tensor(&format!("{p}.hc_attn_fn.weight"), dev)?)?,
                deq(&mut ct, &format!("{p}.hc_attn_scale.weight"), dev)?.to_vec1::<f32>()?,
                deq(&mut ct, &format!("{p}.hc_attn_base.weight"), dev)?,
                props.hc_count,
                props.hc_sinkhorn_iters,
                props.hc_eps,
                false,
            )?;
            let hc_ffn = HyperConnections::from_parts(
                gguf_linear(ct.tensor(&format!("{p}.hc_ffn_fn.weight"), dev)?)?,
                deq(&mut ct, &format!("{p}.hc_ffn_scale.weight"), dev)?.to_vec1::<f32>()?,
                deq(&mut ct, &format!("{p}.hc_ffn_base.weight"), dev)?,
                props.hc_count,
                props.hc_sinkhorn_iters,
                props.hc_eps,
                false,
            )?;

            layers.push(DecoderLayer {
                hc_attn,
                attn_norm: rms_from(deq(&mut ct, &format!("{p}.attn_norm.weight"), dev)?, eps)?,
                attn,
                hc_ffn,
                ffn_norm: rms_from(deq(&mut ct, &format!("{p}.ffn_norm.weight"), dev)?, eps)?,
                moe,
            });
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
                (0..props.block_count).map(|_| CompressorState::default()).collect(),
            ),
        })
    }
}

impl ModelWeights {
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
        let embeds = self.tok_embeddings.forward(input_ids)?.to_dtype(self.dtype)?;
        let cache = &mut self.cache.normal().0;
        let mask = CausalMasker.make_causal_mask(
            input_ids,
            cache as &dyn PastKvLenCache,
            self.dtype,
            &CausalMaskConfig::default(),
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
        let mut comp = self.comp_state.lock().unwrap();
        if start_offsets.iter().any(|&o| o == 0) {
            for s in comp.iter_mut() {
                s.x_history = None;
            }
        }

        // Expand to the HC carrier, thread through layers, reduce at output.
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
            )?;
            trace_rms(&hc, format_args!("v4 carrier after layer {i}"));
        }
        drop(comp);
        let x = self.output_hc.reduce_output(&hc)?;
        trace_rms(&x, format_args!("v4 reduce_output"));
        let x = self.norm.forward(&x)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.forward(&x.contiguous()?)
    }
}
