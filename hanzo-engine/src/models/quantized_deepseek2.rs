#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF DeepSeek-V2/V3 (MLA) quantized model.
//!
//! Mirrors the safetensors `deepseek2`/`deepseek3` attention (Multi-head Latent Attention) but loads
//! weights from a GGUF `deepseek2` checkpoint and runs them through the quantized `GgufMatMul` path.
//! The same arch covers V3/V4-class checkpoints (the V3 no-aux-loss gate is scaffolded behind the
//! `expert_weights_norm`/`expert_gating_func` metadata; see TODOs). Layout/tensor names follow
//! llama.cpp's `LLM_ARCH_DEEPSEEK2` convention.
//!
//! NOTE: MLA here uses the un-absorbed (materialized K/V) eager path that the safetensors model uses
//! on the no-paged branch; the CUDA paged "absorbed" MLA decode in `crate::mla` is NOT wired here yet
//! (it needs paged-attn MLA cache layout for GGUF). See the report.
//!
//! DSA: a `glm-dsa` GGUF carrying the lightning-indexer (`attn_indexer_*` tensors + `index_*`
//! metadata) drives GLM-5.2 top-`index_topk` sparse key selection on the eager cold-prefill path via
//! the shared [`super::dsa`] primitive — the same selection the safetensors [`super::glm5_moe`] uses.
//! `DSA=0` forces dense; `DSA_TOPK=N` overrides. `index_topk >= context` is byte-identical to dense.

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{
    CausalMaskConfig, CausalMasker, DeepSeekV2RopeConfig, DeepSeekV2RopeScaling,
    DeepSeekV2RotaryEmbedding, QRmsNorm, ScaledRopeType, Sdpa,
};
use crate::layers_masker::PastKvLenCache;
use crate::ops::SplitOp;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::pipeline_parallel::{
    pp_head_forward, use_pipeline_parallel, PipelineParallelModel, RingLayout,
};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, LayerNorm, Linear, Module};
use hanzo_quant::{QuantMethod, QuantMethodConfig, RingPipeline, UnquantLinear};

const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

// llama.cpp expert gating funcs (LLM_EXPERT_GATING_FUNC_TYPE). V2 = softmax, V3 = sigmoid.
const EXPERT_GATING_SOFTMAX: u32 = 1;
const EXPERT_GATING_SIGMOID: u32 = 2;

/// GLM-5's DSA indexer key-norm is a real `LayerNorm` with a hardcoded `eps = 1e-6`
/// (independent of the model's `rms_norm_eps`), matching [`super::glm5_moe`]'s
/// `INDEXER_KNORM_EPS` and the colibrì `glm.c` reference.
const INDEXER_KNORM_EPS: f64 = 1e-6;

use crate::models::dsa::{DsaConfig, DsaIndexer, DsaSelection};
use crate::models::gguf_moe::{build_moe_or_mlp, gguf_linear, MoeOrMlp, MoeParams};

/// `MLA_ABSORB=1` enables the device-agnostic absorbed-MLA decode.
///
/// When on, the KV cache holds only the compressed latent `[kv_lora + qk_rope]` per token (~576
/// floats for GLM-5.2) instead of the materialized per-head K/V (`n_head * q_head_dim` +
/// `n_head * v_head_dim`, ~20k floats), and the `kv_b` projection is *absorbed*: its K rows fold
/// into the query (`ql_nope = q_nope @ w_uk`) and its V rows fold out after attention
/// (`out = (att @ ckv) @ w_uv_t`). This is the DeepSeek/colibri weight-absorption trick.
///
/// It is algebraically identical to the materialized path (`ql_nope·ckv == q_nope·k_nope`,
/// `Σ_t a_t (w_uv·ckv_t) == Σ_t a_t v_t`), so decode is token-for-token equal in exact arithmetic.
/// It is opt-in (default off) until validated against a real GGUF because absorption reassociates
/// the score/context reductions: floating-point rounding differs, so a near-tie argmax could in
/// principle flip (the same shape-dependent-kernel caveat colibri documents for its MTP/CUDA tiers).
/// Read once, cached.
fn mla_absorb_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MLA_ABSORB")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            .unwrap_or(false)
    })
}

/// Extract the absorbed-MLA weight views from the (un-absorbed) `kv_b` projection.
///
/// `kv_b` maps the latent `ckv [kv_lora]` to the per-head `[k_nope | v]` block, i.e. its weight is
/// `[n_head * (qk_nope + v_head), kv_lora]`. Split per head into:
///   - `w_uk = [n_head, qk_nope, kv_lora]` — the K-nope rows. `ql_nope = q_nope @ w_uk` gives the
///     latent query with `ql_nope · ckv == q_nope · (w_uk^T · ckv) == q_nope · k_nope`.
///   - `w_uv_t = [n_head, kv_lora, v_head]` — the V rows, transposed. After latent attention,
///     `attn_latent @ w_uv_t == Σ_t a_t (w_uv · ckv_t) == Σ_t a_t v_t`.
///
/// Held dense in `dtype` so the decode matmuls stay single-dtype. Mirrors `crate::mla::MlaWeights`
/// (which is CUDA/FlashInfer-only) but is device-agnostic (pure tensor ops).
fn absorb_weights_from_kv_b(
    kv_b_proj: &dyn QuantMethod,
    device: &Device,
    n_head: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let mut w = kv_b_proj.dequantize_w()?;
    if !w.device().same_device(device) {
        w = w.to_device(device)?;
    }
    let (out_dim, in_dim) = w.dims2()?;
    let per_head = qk_nope_head_dim + v_head_dim;
    if in_dim != kv_lora_rank || out_dim != n_head * per_head {
        hanzo_ml::bail!(
            "absorbed MLA: kv_b_proj weight is [{out_dim}, {in_dim}], expected [{}, {kv_lora_rank}]",
            n_head * per_head
        );
    }
    let w = w.reshape((n_head, per_head, kv_lora_rank))?;
    let w_uk = w
        .narrow(D::Minus2, 0, qk_nope_head_dim)?
        .contiguous()?
        .to_dtype(dtype)?;
    let w_uv_t = w
        .narrow(D::Minus2, qk_nope_head_dim, v_head_dim)?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(dtype)?;
    Ok((w_uk, w_uv_t))
}

/// DSA sparse attention is on by default whenever a checkpoint ships the indexer
/// tensors; `DSA=0` forces the dense path (byte-identical). Mirrors colibrì.
fn dsa_enabled() -> bool {
    !matches!(std::env::var("DSA").ok().as_deref(), Some("0"))
}

/// `DSA_TOPK=N` overrides the checkpoint's `index_topk` (test / ablation knob,
/// colibrì-compatible). `DSA_TOPK >= context` reproduces dense selection exactly.
fn dsa_topk_override() -> Option<usize> {
    std::env::var("DSA_TOPK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
}

/// The checkpoint's DSA config, or `None` when DSA is off (`DSA=0`) or the GGUF
/// lacks the indexer metadata. Gated by [`DsaConfig::new`] on colibrì's `has_dsa`
/// predicate (`glm.c`: topk/heads > 0 and `0 < index_head_dim <= 256`). Derived purely
/// from `props`, so the single block loader ([`LayerWeights::load`]) and the model-level
/// summary log agree by construction.
fn dsa_config(props: &PropsGGUF) -> Option<DsaConfig> {
    if !dsa_enabled() {
        return None;
    }
    match (props.index_topk, props.index_n_heads, props.index_head_dim) {
        // Gate on the raw checkpoint dims (colibrì `has_dsa`); the `DSA_TOPK`
        // ablation knob then overrides the kept-key count on the valid config.
        (Some(topk), Some(nh), Some(hd)) => DsaConfig::new(nh, hd, topk, props.qk_rope_head_dim)
            .map(|c| DsaConfig {
                index_topk: dsa_topk_override().unwrap_or(topk),
                ..c
            }),
        _ => None,
    }
}

/// Per-sequence RoPE base offsets as the position tensor the DSA indexer's
/// partial RoPE consumes. Length `batch` -> the gather expands each offset to
/// `off + 0..seq_len`, i.e. the exact cos/sin rows the main MLA RoPE selects
/// from the same `start_offsets` (cold prefill: `off == 0`, positions `0..L`).
fn dsa_rope_positions(offsets: &[usize], device: &Device) -> Result<Tensor> {
    let positions = offsets.iter().map(|&o| o as u32).collect::<Vec<_>>();
    Tensor::from_vec(positions, offsets.len(), device)
}

/// Load a layer's DSA lightning-indexer from a `glm-dsa` GGUF, or `None` when the
/// layer ships no indexer tensors (an IndexShare "shared" layer, or a non-DSA
/// checkpoint) — that layer then reuses the previous full layer's selection.
///
/// Tensor names (llama.cpp `blk.N.` convention, mapped from the HF
/// `self_attn.indexer.{wq_b,wk,weights_proj,k_norm}` the colibrì `--indexer`
/// converter extracts as `out-idx-*`):
///   `attn_indexer_q`      = `wq_b`         (q-LoRA latent -> n_head·head_dim)
///   `attn_indexer_k`      = `wk`           (hidden -> head_dim, shared MQA head)
///   `attn_indexer_w`      = `weights_proj` (hidden -> n_head)
///   `attn_indexer_k_norm` = `k_norm`       (LayerNorm weight + bias)
fn load_layer_indexer<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    prefix: &str,
    cfg: DsaConfig,
    device: &Device,
    dtype: DType,
) -> Result<Option<DsaIndexer>> {
    let wq_name = format!("{prefix}.attn_indexer_q.weight");
    if !ct.has_tensor(&wq_name) {
        return Ok(None);
    }
    let wq = gguf_linear(ct.tensor(&wq_name, device)?)?;
    let wk = gguf_linear(ct.tensor(&format!("{prefix}.attn_indexer_k.weight"), device)?)?;
    let weights_proj =
        gguf_linear(ct.tensor(&format!("{prefix}.attn_indexer_w.weight"), device)?)?;

    let k_norm = if ct.has_tensor(&format!("{prefix}.attn_indexer_k_norm.weight")) {
        let w = ct
            .tensor(&format!("{prefix}.attn_indexer_k_norm.weight"), device)?
            .dequantize(device)?
            .to_dtype(dtype)?;
        let b = ct
            .tensor(&format!("{prefix}.attn_indexer_k_norm.bias"), device)?
            .dequantize(device)?
            .to_dtype(dtype)?;
        Some(LayerNorm::new(w, b, INDEXER_KNORM_EPS))
    } else {
        None
    };

    Ok(Some(DsaIndexer::from_parts(cfg, wq, wk, weights_proj, k_norm)))
}

pub(crate) struct LayerWeights {
    // MLA projections. q is either plain (q_proj) or low-rank (q_a/q_b with q_a_norm).
    q_a_proj: Option<Arc<dyn QuantMethod>>,
    q_a_norm: Option<QRmsNorm>,
    q_b_proj: Option<Arc<dyn QuantMethod>>,
    q_proj: Option<Arc<dyn QuantMethod>>,
    kv_a_proj_with_mqa: Arc<dyn QuantMethod>,
    kv_a_norm: QRmsNorm,
    kv_b_proj: Arc<dyn QuantMethod>,
    // Absorbed-MLA weight views, precomputed from `kv_b_proj` only when `MLA_ABSORB` is on. `Some`
    // switches `forward_attn` to the compressed-latent decode; `None` keeps the materialized path.
    w_uk: Option<Tensor>,
    w_uv_t: Option<Tensor>,
    o_proj: Arc<dyn QuantMethod>,
    attn_norm: QRmsNorm,
    ffn_norm: QRmsNorm,
    mlp: MoeOrMlp,
    rotary: Arc<DeepSeekV2RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    n_head: usize,
    q_head_dim: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    kv_lora_rank: usize,
    v_head_dim: usize,
    dtype: DType,
    /// DSA lightning indexer on IndexShare "full" layers; `None` on "shared"
    /// layers (reuse the threaded-in selection) and non-DSA checkpoints (dense).
    indexer: Option<DsaIndexer>,
}

impl LayerWeights {
    /// The q-LoRA-normalised latent the DSA indexer projects its query from:
    /// `q_a_norm(q_a_proj(x))`. Every DSA checkpoint uses low-rank Q, so an
    /// indexer-bearing layer always has these; guard for the plain-Q case.
    fn indexer_q_src(&self, x: &Tensor) -> Result<Tensor> {
        match (&self.q_a_proj, &self.q_a_norm) {
            (Some(a), Some(norm)) => norm.forward(&a.forward(x)?),
            _ => hanzo_ml::bail!("deepseek2 gguf: DSA indexer requires low-rank Q (q_a_proj)"),
        }
    }

    /// Load one deepseek2/glm-dsa decoder block (`blk.{layer_idx}`): MLA attention +
    /// dense-or-MoE feed-forward. The single loader used by BOTH the main model loop and
    /// the in-band `nextn` MTP head (`crate::models::deepseek2_mtp`) — the block layout is
    /// identical, so it lives in exactly one place. `paged_attn` is `None` for the head
    /// (the draft runs Eager over its own `KvCache`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        layer_idx: usize,
        props: &PropsGGUF,
        device: &Device,
        rotary: Arc<DeepSeekV2RotaryEmbedding>,
        softmax_scale: f32,
        q_head_dim: usize,
        paged_attn: Option<PagedAttention>,
        dtype: DType,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        // Q: low-rank (attn_q_a/attn_q_b) if q_lora_rank set, else plain attn_q.
        let (q_a_proj, q_a_norm, q_b_proj, q_proj) = if props.q_lora_rank.is_some() {
            (
                Some(gguf_linear(
                    ct.tensor(&format!("{prefix}.attn_q_a.weight"), device)?,
                )?),
                Some(QRmsNorm::new_dtype(
                    ct.tensor(&format!("{prefix}.attn_q_a_norm.weight"), device)?,
                    props.rms_norm_eps,
                    dtype,
                )?),
                Some(gguf_linear(
                    ct.tensor(&format!("{prefix}.attn_q_b.weight"), device)?,
                )?),
                None,
            )
        } else {
            (
                None,
                None,
                None,
                Some(gguf_linear(
                    ct.tensor(&format!("{prefix}.attn_q.weight"), device)?,
                )?),
            )
        };

        let kv_a_proj_with_mqa =
            gguf_linear(ct.tensor(&format!("{prefix}.attn_kv_a_mqa.weight"), device)?)?;
        let kv_a_norm = QRmsNorm::new_dtype(
            ct.tensor(&format!("{prefix}.attn_kv_a_norm.weight"), device)?,
            props.rms_norm_eps,
            dtype,
        )?;
        // kv_b: classic GGUFs ship the combined `attn_kv_b` (kv_lora -> [k_nope; v]). Newer
        // split-MLA GGUFs (GLM-4.7-Flash) ship `attn_k_b` (qk_nope -> kv_lora, absorbed
        // orientation) + `attn_v_b` (kv_lora -> v). Reconstruct the un-absorbed combined weight
        // `cat(k_b^T, v_b)` so the materialized-K/V forward below is identical for both layouts.
        let kv_b_proj = if ct.has_tensor(&format!("{prefix}.attn_kv_b.weight")) {
            gguf_linear(ct.tensor(&format!("{prefix}.attn_kv_b.weight"), device)?)?
        } else {
            let k_b = ct
                .tensor(&format!("{prefix}.attn_k_b.weight"), device)?
                .dequantize(device)?;
            let v_b = ct
                .tensor(&format!("{prefix}.attn_v_b.weight"), device)?
                .dequantize(device)?;
            let k_part = k_b.transpose(1, 2)?;
            let kv_b = Tensor::cat(&[&k_part, &v_b], 1)?
                .reshape((
                    props.head_count * (props.qk_nope_head_dim + props.v_head_dim),
                    props.kv_lora_rank,
                ))?
                // Single-dtype: hold the materialized (un-absorbed) kv_b weight in the
                // compute dtype so `UnquantLinear`'s cuBLASlt matmul sees a uniform (a, w)
                // dtype. Held in F32, the bf16 single-dtype `ckv` activation hit the F32
                // weight -> `as_cuda_slice::<bf16>` on F32 storage (expected BF16, got F32).
                .to_dtype(dtype)?
                .contiguous()?;
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(kv_b, None),
            ))?)
        };
        // Absorbed-MLA views (opt-in, MLA_ABSORB=1): fold kv_b's K rows into the query and V rows
        // out of the context so decode stores only the compressed latent. Built once from the
        // (materialized-orientation) kv_b so classic and split-MLA GGUFs are both covered.
        let (w_uk, w_uv_t) = if mla_absorb_enabled() {
            let (uk, uvt) = absorb_weights_from_kv_b(
                kv_b_proj.as_ref(),
                device,
                props.head_count,
                props.kv_lora_rank,
                props.qk_nope_head_dim,
                props.v_head_dim,
                dtype,
            )?;
            (Some(uk), Some(uvt))
        } else {
            (None, None)
        };
        // DSA lightning indexer: `Some` on a "full" layer that ships indexer tensors, `None` on a
        // "shared" layer (reuses the threaded selection), a non-DSA checkpoint, or the nextn MTP
        // block (no indexer tensors). Derived from `props` — same source as the summary log.
        let indexer = match dsa_config(props) {
            Some(cfg) => load_layer_indexer(ct, &prefix, cfg, device, dtype)?,
            None => None,
        };
        let o_proj = gguf_linear(ct.tensor(&format!("{prefix}.attn_output.weight"), device)?)?;

        let attn_norm = QRmsNorm::new_dtype(
            ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
            props.rms_norm_eps,
            dtype,
        )?;
        let ffn_norm = QRmsNorm::new_dtype(
            ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?,
            props.rms_norm_eps,
            dtype,
        )?;

        let mlp = build_moe_or_mlp(
            ct,
            layer_idx,
            device,
            &MoeParams {
                n_routed_experts: props.n_routed_experts,
                num_experts_per_tok: props.num_experts_per_tok,
                n_group: props.n_group,
                topk_group: props.topk_group,
                routed_scaling_factor: props.expert_weights_scale,
                norm_topk_prob: props.norm_topk_prob,
                sigmoid_scoring: props.sigmoid_scoring,
                n_shared_experts: props.n_shared_experts,
                leading_dense_block_count: props.leading_dense_block_count,
            },
        )?;

        Ok(LayerWeights {
            q_a_proj,
            q_a_norm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_norm,
            kv_b_proj,
            w_uk,
            w_uv_t,
            o_proj,
            attn_norm,
            ffn_norm,
            mlp,
            rotary,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: 1,
                softcap: None,
                softmax_scale,
                sliding_window: None,
                sinks: None,
            },
            n_head: props.head_count,
            q_head_dim,
            qk_nope_head_dim: props.qk_nope_head_dim,
            qk_rope_head_dim: props.qk_rope_head_dim,
            kv_lora_rank: props.kv_lora_rank,
            v_head_dim: props.v_head_dim,
            dtype,
            indexer,
        })
    }

    /// Returns `(attn_out, selection)`. `selection` is `Some` on the eager
    /// cold-prefill DSA path (the value the next "shared" layer reuses) and
    /// `None` on every dense fallback (paged, warm cache, decode, non-DSA).
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        prev_selection: Option<&DsaSelection>,
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<(Tensor, Option<DsaSelection>)> {
        // Absorbed-MLA decode (MLA_ABSORB=1): compressed-latent KV cache + kv_b folded into q/out.
        // Taken only on the non-paged cache and only when DSA is not in play for this layer (no
        // indexer, no threaded selection): the absorbed path does dense latent attention, so it must
        // not pre-empt DSA's sparse key selection. Algebraically identical to the materialized path.
        if metadata.is_none() && self.indexer.is_none() && prev_selection.is_none() {
            if let (Some(w_uk), Some(w_uv_t)) = (&self.w_uk, &self.w_uv_t) {
                let out = self.forward_attn_absorbed(x, mask, start_offsets, kv_cache, w_uk, w_uv_t)?;
                return Ok((out, None));
            }
        }

        let (bs, seq_len, _) = x.dims3()?;

        let q = match (&self.q_proj, &self.q_a_proj, &self.q_b_proj, &self.q_a_norm) {
            (Some(q_proj), _, _, _) => q_proj.forward(x)?,
            (None, Some(a), Some(b), Some(norm)) => b.forward(&norm.forward(&a.forward(x)?)?)?,
            _ => hanzo_ml::bail!("deepseek2 gguf: inconsistent q projection weights"),
        };
        let q = q
            .reshape((bs, seq_len, self.n_head, self.q_head_dim))?
            .transpose(1, 2)?;
        let q_split = q.split(&[self.qk_nope_head_dim, self.qk_rope_head_dim], D::Minus1)?;
        let q_nope = q_split[0].clone();
        let mut q_pe = q_split[1].clone();

        let compressed_kv = self.kv_a_proj_with_mqa.forward(x)?;
        let ckv_split =
            compressed_kv.split(&[self.kv_lora_rank, self.qk_rope_head_dim], D::Minus1)?;
        let compressed_kv = ckv_split[0].clone();
        let mut k_pe = ckv_split[1].clone();
        // Broadcast the shared (MQA) rope key to n_head BEFORE rope: the fused CUDA rope kernel
        // requires q and k to have equal head counts, and rope is head-independent so this is exact.
        k_pe = k_pe
            .reshape((bs, seq_len, 1, self.qk_rope_head_dim))?
            .transpose(1, 2)?
            .repeat((1, self.n_head, 1, 1))?;
        let ckv = self.kv_a_norm.forward(&compressed_kv)?;

        (q_pe, k_pe) = self.rotary.forward(&q_pe, &k_pe, start_offsets)?;

        let mut kv = self.kv_b_proj.forward(&ckv)?;
        kv = kv
            .reshape((
                bs,
                seq_len,
                self.n_head,
                self.qk_nope_head_dim + self.v_head_dim,
            ))?
            .transpose(1, 2)?;
        let kv_split = kv.split(&[self.qk_nope_head_dim, self.v_head_dim], D::Minus1)?;
        let k_nope = kv_split[0].clone();
        let v = kv_split[1].clone();

        // Rope may hand back f32 pe parts; the nope parts follow the single-dtype residual, so pin
        // both cat operands to the compute dtype (keeps the concat and the whole MLA single-dtype).
        let q = Tensor::cat(
            &[&q_nope.to_dtype(self.dtype)?, &q_pe.to_dtype(self.dtype)?],
            D::Minus1,
        )?
        .contiguous()?;
        let k = Tensor::cat(
            &[&k_nope.to_dtype(self.dtype)?, &k_pe.to_dtype(self.dtype)?],
            D::Minus1,
        )?
        .contiguous()?;

        let v = v.to_dtype(self.dtype)?;

        let mut selection_out = None;
        let y = match &self.paged_attn {
            Some(paged_attn) => {
                let ((key_cache, value_cache), input_metadata) = metadata.unwrap();
                // Pad V to q_head_dim so the paged cache K/V share head_dim, then slice back.
                let v = v
                    .pad_with_zeros(D::Minus1, 0, self.q_head_dim - self.v_head_dim)?
                    .contiguous()?;
                paged_attn
                    .forward(
                        &q,
                        &k,
                        &v,
                        mask,
                        Some(key_cache),
                        Some(value_cache),
                        input_metadata,
                        &self.sdpa_params,
                        None,
                    )?
                    .narrow(D::Minus1, 0, self.v_head_dim)?
            }
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                // DSA (eager cold-cache prefill only): fold the lightning-indexer's
                // selected-key `-inf` bias into the causal mask so MLA attends over
                // the top-`index_topk` keys instead of dense O(L^2). A "full" layer
                // (has its own indexer) computes a fresh selection; a "shared" layer
                // reuses `prev_selection` (GLM-5 IndexShare). Guarded to `Lk == seq_len`
                // (cold cache, mask already Custom) so the `[B, seq_len, seq_len]`
                // selection aligns with the causal mask and the `mask.is_custom()`
                // reshape branch below is unchanged; warm cache / decode stay dense.
                // `index_topk >= seq_len` -> all keys -> zero bias -> byte-identical
                // to the dense path.
                let dsa_mask = match mask.as_option_tensor() {
                    Some(base) if base.dim(D::Minus1)? == seq_len => {
                        let selection = match &self.indexer {
                            Some(indexer) => {
                                let q_src = self.indexer_q_src(x)?;
                                let positions = dsa_rope_positions(start_offsets, x.device())?;
                                Some(indexer.forward(
                                    &q_src,
                                    x,
                                    Some(&self.rotary),
                                    Some(&positions),
                                    true,
                                )?)
                            }
                            None => prev_selection.cloned(),
                        };
                        match selection {
                            Some(sel) => {
                                let mask = AttentionMask::Custom(sel.combine_with_mask(base)?);
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
                    dsa_mask.as_ref().unwrap_or(mask),
                    None,
                    &self.sdpa_params,
                )?
            }
        };

        // Reshape decision keys off the ORIGINAL mask (Custom => eager `[B,H,L,Dv]`
        // layout), never the DSA-wrapped one, so a substituted mask can't flip it.
        let y = if mask.is_custom() {
            y.transpose(1, 2)?.reshape((bs, seq_len, ()))?
        } else {
            y.reshape((bs, seq_len, ()))?
        };
        Ok((self.o_proj.forward(&y.to_dtype(x.dtype())?)?, selection_out))
    }

    /// Absorbed-MLA attention: standard attention in the compressed latent space.
    ///
    /// The KV cache stores only `[ckv | k_pe]` (one shared MQA head), and `kv_b` never materializes
    /// per-head K/V. Instead its K rows are folded into the query (`w_uk`) and its V rows out of the
    /// context (`w_uv_t`). Numerically mirrors `naive_sdpa` (scale, F32 softmax, additive mask) so
    /// the only difference from the materialized path is the absorption reassociation itself.
    ///
    /// Shapes (bs=B, heads=H, new tokens=S, cache length=T): `q_latent [B,H,S,kv_lora+qk_rope]`,
    /// `k_latent [B,1,T,kv_lora+qk_rope]`, `v_latent [B,1,T,kv_lora]`. K/V stay one head and
    /// broadcast across the H query heads via `broadcast_matmul` — no per-head expansion, so the
    /// MLA memory profile holds at compute time too.
    fn forward_attn_absorbed(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        w_uk: &Tensor,
        w_uv_t: &Tensor,
    ) -> Result<Tensor> {
        if matches!(mask, AttentionMask::CausalFlash) {
            // GGUF deepseek2/glm-dsa always emits Custom/None (CausalMaskConfig::gguf). A CausalFlash
            // mask carries no tensor, so silently dropping it would make prefill non-causal — bail
            // instead of corrupting attention.
            hanzo_ml::bail!("absorbed MLA: unexpected CausalFlash mask on the eager GGUF path");
        }
        let (bs, seq_len, _) = x.dims3()?;

        // Query: identical projection, split, and RoPE to the materialized path.
        let q = match (&self.q_proj, &self.q_a_proj, &self.q_b_proj, &self.q_a_norm) {
            (Some(q_proj), _, _, _) => q_proj.forward(x)?,
            (None, Some(a), Some(b), Some(norm)) => b.forward(&norm.forward(&a.forward(x)?)?)?,
            _ => hanzo_ml::bail!("deepseek2 gguf: inconsistent q projection weights"),
        };
        let q = q
            .reshape((bs, seq_len, self.n_head, self.q_head_dim))?
            .transpose(1, 2)?;
        let q_split = q.split(&[self.qk_nope_head_dim, self.qk_rope_head_dim], D::Minus1)?;
        let q_nope = q_split[0].clone();
        let mut q_pe = q_split[1].clone();

        // Compressed latent (normed) + shared RoPE key.
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x)?;
        let ckv_split =
            compressed_kv.split(&[self.kv_lora_rank, self.qk_rope_head_dim], D::Minus1)?;
        let ckv = self.kv_a_norm.forward(&ckv_split[0])?; // [B, S, kv_lora]
        // Broadcast the shared rope key to n_head for the head-count-symmetric rope kernel, then
        // keep a single head: RoPE is head-independent and the inputs were identical across heads.
        let mut k_pe = ckv_split[1]
            .reshape((bs, seq_len, 1, self.qk_rope_head_dim))?
            .transpose(1, 2)?
            .repeat((1, self.n_head, 1, 1))?;
        (q_pe, k_pe) = self.rotary.forward(&q_pe, &k_pe, start_offsets)?;
        let k_pe = k_pe.narrow(1, 0, 1)?; // [B, 1, S, qk_rope]

        // Absorb kv_b K-rows into the query: ql_nope = q_nope @ w_uk -> [B, H, S, kv_lora].
        let ql_nope = q_nope
            .to_dtype(self.dtype)?
            .broadcast_matmul(&w_uk.unsqueeze(0)?)?;
        let q_latent = Tensor::cat(&[&ql_nope, &q_pe.to_dtype(self.dtype)?], D::Minus1)?
            .contiguous()?; // [B, H, S, kv_lora + qk_rope]

        // Latent key/value for the new tokens (one shared head), appended to the compressed cache.
        let ckv4 = ckv
            .to_dtype(self.dtype)?
            .reshape((bs, seq_len, 1, self.kv_lora_rank))?
            .transpose(1, 2)?
            .contiguous()?; // [B, 1, S, kv_lora]  (this IS v_latent for the new tokens)
        let k_latent_new =
            Tensor::cat(&[&ckv4, &k_pe.to_dtype(self.dtype)?], D::Minus1)?.contiguous()?;
        let (k_latent, v_latent) = kv_cache.append(&k_latent_new, &ckv4)?; // full [0, T)

        // Scores in latent space; K/V (one head) broadcast across the H query heads.
        let scores = q_latent.broadcast_matmul(&k_latent.transpose(2, 3)?.contiguous()?)?;
        let mut att = (scores * f64::from(self.sdpa_params.softmax_scale))?; // [B, H, S, T]
        if let Some(m) = mask.as_option_tensor() {
            att = att.broadcast_add(&m.to_dtype(att.dtype())?)?;
        }
        // Softmax in F32 for precision (mirrors naive_sdpa: BF16/F16 exp() loses information).
        let att_dtype = att.dtype();
        if matches!(att_dtype, DType::BF16 | DType::F16) {
            att = att.to_dtype(DType::F32)?;
        }
        att = hanzo_nn::ops::softmax_last_dim(&att)?;
        if att.dtype() != att_dtype {
            att = att.to_dtype(att_dtype)?;
        }

        // Context latent, then fold kv_b V-rows back out: out = (att @ v_latent) @ w_uv_t.
        let attn_latent = att.broadcast_matmul(&v_latent)?; // [B, H, S, kv_lora]
        let out = attn_latent
            .broadcast_matmul(&w_uv_t.unsqueeze(0)?)? // [B, H, S, v_head]
            .transpose(1, 2)?
            .reshape((bs, seq_len, self.n_head * self.v_head_dim))?;
        self.o_proj.forward(&out.to_dtype(x.dtype())?)
    }

    pub(crate) fn forward_block(
        &self,
        x: Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        prev_selection: Option<&DsaSelection>,
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<(Tensor, Option<DsaSelection>)> {
        let residual = &x;
        let xn = self.attn_norm.forward(&x)?;
        let (attn, selection) =
            self.forward_attn(&xn, mask, start_offsets, prev_selection, kv_cache, metadata)?;
        let x = (attn + residual)?;
        let residual = &x;
        let xn = self.ffn_norm.forward(&x)?;
        let xn = self.mlp.forward(&xn)?;
        Ok(((xn + residual)?, selection))
    }
}

pub struct ModelWeights {
    tok_embeddings: Option<Embedding>,
    layers: Vec<LayerWeights>,
    norm: Option<QRmsNorm>,
    output: Option<Arc<dyn QuantMethod>>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    pp: Option<Arc<RingPipeline>>,
    /// Base-model config, retained so the in-band `nextn` MTP head (GLM-5.2) can load
    /// against it. See [`crate::models::deepseek2_mtp`].
    base_props: PropsGGUF,
    /// MTP self-speculative decode: when `store_spec_hidden` is set, [`Self::forward`]
    /// stashes the post-norm, per-sampled-row hidden (before the output projection) here
    /// so the MTP proposer can read it. Same seam as `quantized_deepseek4`.
    spec_hidden: std::sync::Mutex<Option<Tensor>>,
    store_spec_hidden: std::sync::atomic::AtomicBool,
}

// Some fields mirror the full deepseek2 GGUF metadata surface but aren't needed at load time (GGUF
// tensors carry their own shapes); kept for clarity and future paged-MLA wiring.
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub block_count: usize,
    /// Count of trailing in-band `nextn` (MTP) blocks, excluded from `block_count`.
    /// `> 0` for GLM-5.2 (`glm-dsa`); the self-speculative draft head lives at
    /// `blk.{block_count}` and is loaded on demand by `SelfSpeculative::attach_mtp`.
    pub nextn_predict_layers: usize,
    pub embedding_length: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    // MLA dims.
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    // MoE.
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub moe_intermediate_size: usize,
    pub dense_intermediate_size: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub leading_dense_block_count: usize,
    pub expert_weights_scale: f64,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub sigmoid_scoring: bool,
    // YaRN rope scaling (optional).
    pub rope_scaling_factor: Option<f32>,
    pub rope_yarn_orig_ctx: Option<usize>,
    // DSA lightning indexer (glm-dsa). All three present + a per-layer
    // `attn_indexer_*` tensor set activate sparse attention; absent -> dense.
    pub index_topk: Option<usize>,
    pub index_n_heads: Option<usize>,
    pub index_head_dim: Option<usize>,
}

// GLM-5.2 (`glm-dsa`) is deepseek2-class in GGUF terms: split-MLA + 256-expert MoE. When the GGUF
// carries the DSA lightning-indexer (`attn_indexer_*` tensors + `index_*` metadata), cold-prefill
// MLA runs over the top-`index_topk` keys the indexer selects (see `forward_attn`); for any prompt
// <= index_topk the selection returns all keys, i.e. dense MLA, byte-identical to the pre-DSA path.
// A GGUF without those tensors runs dense unconditionally, exactly as before.
fn verify_arch(
    metadata: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>,
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    if arch != "deepseek2" && arch != "glm-dsa" {
        hanzo_ml::bail!("Expected `deepseek2` or `glm-dsa` architecture, got `{arch}`.");
    }
    Ok(arch)
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let required = [
            "attention.head_count",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
            "attention.kv_lora_rank",
            "attention.key_length", // qk_nope_head_dim (llama.cpp sets key_length = nope dim)
            "rope.dimension_count", // qk_rope_head_dim
            "expert_count",
            "expert_used_count",
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;
        let n_routed_experts = c.get_value::<u32>("expert_count")? as usize;

        let sigmoid_scoring = match c.get_value::<u32>("expert_gating_func") {
            Ok(EXPERT_GATING_SIGMOID) => true,
            Ok(EXPERT_GATING_SOFTMAX) | Err(_) => false,
            Ok(other) => anyhow::bail!("unknown deepseek2 expert_gating_func {other}"),
        };

        // MLA head dims. Newer GGUFs (GLM-4.7-Flash, recent DeepSeek) carry the split-MLA layout:
        // `key_length` is the compressed MQA width (kv_lora + rope) and the true per-head dims live in
        // `key_length_mla` (= q_head_dim) / `value_length_mla`. Classic GGUFs set `key_length` = qk_nope.
        let qk_rope_head_dim = c.get_value::<u32>("rope.dimension_count")? as usize;
        let qk_nope_head_dim = match c.get_value::<u32>("attention.key_length_mla") {
            Ok(klm) => klm as usize - qk_rope_head_dim,
            Err(_) => c.get_value::<u32>("attention.key_length")? as usize,
        };
        let v_head_dim = c
            .get_value::<u32>("attention.value_length_mla")
            .or_else(|_| c.get_value::<u32>("attention.value_length"))
            .map(|x| x as usize)
            .unwrap_or(qk_nope_head_dim);

        // GLM-5.2 (`glm-dsa`) trails `nextn_predict_layers` MTP/nextn blocks after the main
        // blocks inside `block_count`. They are the self-speculative draft head, NOT part of
        // the main forward path — so the main model uses `block_count - nextn` layers, while the
        // head's tensors stay reachable at `blk.{block_count}` for `SelfSpeculative::attach_mtp`.
        let nextn_predict_layers = c
            .get_value::<u32>("nextn_predict_layers")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(0);

        Ok(Self {
            head_count,
            block_count: c.get_value::<u32>("block_count")? as usize - nextn_predict_layers,
            nextn_predict_layers,
            embedding_length: embed_len,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
            q_lora_rank: c
                .get_value::<u32>("attention.q_lora_rank")
                .ok()
                .map(|x| x as usize),
            kv_lora_rank: c.get_value::<u32>("attention.kv_lora_rank")? as usize,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            n_routed_experts,
            n_shared_experts: c
                .get_value::<u32>("expert_shared_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
            moe_intermediate_size: c.get_value::<u32>("expert_feed_forward_length")? as usize,
            dense_intermediate_size: c.get_value::<u32>("feed_forward_length")? as usize,
            num_experts_per_tok: c.get_value::<u32>("expert_used_count")? as usize,
            first_k_dense_replace: c
                .get_value::<u32>("leading_dense_block_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
            leading_dense_block_count: c
                .get_value::<u32>("leading_dense_block_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
            expert_weights_scale: c
                .get_value::<f32>("expert_weights_scale")
                .ok()
                .map(|x| x as f64)
                .unwrap_or(1.0),
            n_group: c
                .get_value::<u32>("expert_group_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(1),
            topk_group: c
                .get_value::<u32>("expert_group_used_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(1),
            norm_topk_prob: c
                .get_value::<bool>("expert_weights_norm")
                .ok()
                .unwrap_or(true),
            sigmoid_scoring,
            rope_scaling_factor: c.get_value::<f32>("rope.scaling.factor").ok(),
            rope_yarn_orig_ctx: c
                .get_value::<u32>("rope.scaling.original_context_length")
                .ok()
                .map(|x| x as usize),
            index_topk: c
                .get_value::<u32>("attention.index_top_k")
                .ok()
                .map(|x| x as usize),
            index_n_heads: c
                .get_value::<u32>("attention.index_head_count")
                .ok()
                .map(|x| x as usize),
            index_head_dim: c
                .get_value::<u32>("attention.index_head_dim")
                .ok()
                .map(|x| x as usize),
        })
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        let meta = ct.get_metadata();
        let arch = verify_arch(meta)?;
        let metadata = ContentMetadata {
            path_prefix: &arch,
            metadata: meta,
        };
        let props = PropsGGUF::try_from(metadata).or_else(|err| hanzo_ml::bail!("{err}"))?;

        let pp = if use_pipeline_parallel() {
            let config = hanzo_quant::RingConfig::load();
            Some(Arc::new(RingPipeline::from_config(&config)?))
        } else {
            None
        };
        let layout = match &pp {
            Some(_) => Some(RingLayout::new(props.block_count)?),
            None => None,
        };
        let local_range = layout
            .as_ref()
            .map_or(0..props.block_count, RingLayout::local);
        let is_head = layout.as_ref().is_none_or(RingLayout::is_head);

        let q_head_dim = props.qk_nope_head_dim + props.qk_rope_head_dim;
        // softmax_scale = 1/sqrt(q_head_dim), optionally YaRN-mscaled (see deepseek3.rs softmax_scale).
        let mut softmax_scale = 1.0 / (q_head_dim as f32).sqrt();
        if let (Some(factor), Some(_orig)) = (props.rope_scaling_factor, props.rope_yarn_orig_ctx) {
            // mscale_all_dim defaults; replicate DeepSeekV2RotaryEmbedding::yarn_get_mscale(factor, 1.0).
            let mscale = DeepSeekV2RotaryEmbedding::yarn_get_mscale(factor, 1.0);
            softmax_scale = softmax_scale * mscale * mscale;
        }

        let (tok_embeddings, norm, output) = if is_head {
            let qtok = ct.tensor("token_embd.weight", device)?;
            let tok = Embedding::new(qtok.dequantize(device)?, props.embedding_length);
            let norm = QRmsNorm::new_dtype(
                ct.tensor("output_norm.weight", device)?,
                props.rms_norm_eps,
                dtype,
            )?;
            let out = if ct.has_tensor("output.weight") {
                ct.tensor("output.weight", device)?
            } else {
                ct.tensor("token_embd.weight", device)?
            };
            (Some(tok), Some(norm), Some(gguf_linear(out)?))
        } else {
            (None, None, None)
        };

        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: props.rope_scaling_factor.and_then(|factor| {
                props
                    .rope_yarn_orig_ctx
                    .map(|orig| DeepSeekV2RopeScaling::Yarn {
                        original_max_position_embeddings: orig,
                        beta_fast: 32.0,
                        beta_slow: 1.0,
                        factor,
                        mscale: 1.0,
                        mscale_all_dim: 1.0,
                        scaling_type: ScaledRopeType::Yarn,
                    })
            }),
            max_position_embeddings: props.max_seq_len,
            rope_theta: props.rope_freq_base,
            qk_rope_head_dim: props.qk_rope_head_dim,
        };

        let mut ropes = HashMap::new();
        for layer_idx in local_range.clone() {
            let ldev = if pp.is_some() {
                device
            } else {
                mapper.device_for(layer_idx, false).unwrap_or(device)
            };
            if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(ldev.location()) {
                e.insert(Arc::new(DeepSeekV2RotaryEmbedding::new(
                    &rope_cfg,
                    DType::F32,
                    ldev,
                )?));
            }
        }

        // DSA lightning-indexer config (glm-dsa). Active only when the metadata
        // carries the indexer dims (and `DSA != 0`); the per-layer `attn_indexer_*`
        // tensors then decide "full" (own indexer) vs "shared" (reuse) by presence.
        // colibrì `has_dsa`: topk/heads > 0 and 0 < index_head_dim <= 256 (else dense).
        let dsa_cfg = dsa_config(&props);
        let mut dsa_full_layers = 0usize;

        // Absorbed-MLA decode is proven token-equal to the materialized path
        // (`absorbed_equals_materialized`), but the KV cache is still preallocated at the
        // materialized per-head shape while the absorbed path appends the compressed latent
        // `[kv_lora(+rope)]` — the absorb-aware `add_request` preallocation must land first.
        // Fail loud at load rather than crash mid-generation; the default path is unaffected.
        if mla_absorb_enabled() {
            hanzo_ml::bail!(
                "MLA_ABSORB=1 is not yet wired through the KV cache (add_request preallocates the \
                 materialized K/V shape, absorbed decode appends the compressed latent). The math \
                 is validated (absorbed_equals_materialized); the absorb-aware cache must land \
                 first. Unset MLA_ABSORB to run."
            );
        }

        let mut layers = Vec::with_capacity(local_range.end - local_range.start);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            local_range.clone(),
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let device = if pp.is_some() {
                device
            } else {
                mapper.device_for(layer_idx, false).unwrap_or(device)
            };
            let rotary = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();

            let paged_attn = match &attention_mechanism {
                _ if pp.is_some() => None,
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(q_head_dim, device, None)?)
                }
            };

            // ONE block loader for the whole deepseek2/glm-dsa family: MLA + MoE + (opt-in)
            // absorbed-MLA views + (checkpoint-driven) DSA indexer. The nextn MTP head loads its
            // block through the SAME `load` — decomplected, one and only one place.
            let layer = LayerWeights::load(
                &mut ct,
                layer_idx,
                &props,
                device,
                rotary,
                softmax_scale,
                q_head_dim,
                paged_attn,
                dtype,
            )?;
            if layer.indexer.is_some() {
                dsa_full_layers += 1;
            }
            layers.push(layer);
        }

        if let Some(cfg) = dsa_cfg {
            if dsa_full_layers > 0 {
                tracing::info!(
                    "DSA sparse attention active: top-{} key selection across {dsa_full_layers} indexer layer(s) (dense beyond cold prefill)",
                    cfg.index_topk,
                );
            } else {
                tracing::warn!(
                    "DSA indexer configured (top-{}) but no `attn_indexer_*` tensors found in the GGUF; running dense. Reconvert with the colibrì `--indexer` mode (out-idx-*).",
                    cfg.index_topk,
                );
            }
        }

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(
                local_range.end - local_range.start,
                props.max_seq_len,
            )),
            max_seq_len: props.max_seq_len,
            mapper: if pp.is_some() { None } else { Some(mapper) },
            dtype,
            pp,
            base_props: props,
            spec_hidden: std::sync::Mutex::new(None),
            store_spec_hidden: std::sync::atomic::AtomicBool::new(false),
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        if self.pp.is_some() {
            return pp_head_forward(self, x, start_offsets, context_lens);
        }
        // Single-dtype residual: cast the F32 embedding output to the compute dtype so the
        // norm/MLA-attention/MoE chain stays one dtype (drops F32<->half casts; the MoE expert path
        // restores the input dtype so no F32 round-trip is reintroduced).
        let mut layer_in = self
            .tok_embeddings
            .as_ref()
            .unwrap()
            .forward(x)?
            .to_dtype(self.dtype)?;
        let cache = &mut self.cache.normal().0;
        let mask = CausalMasker.make_causal_mask(
            x,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.dtype,
            &CausalMaskConfig::gguf(),
        )?;
        let mask = if metadata
            .as_ref()
            .map(|(_, meta)| meta.is_first_prompt_chunk)
            .unwrap_or(true)
        {
            mask
        } else {
            AttentionMask::None
        };
        let mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(mask)
        };
        let mut selection: Option<DsaSelection> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }
            let dmask = mask.get(layer_in.device());
            let (xs_next, sel) = layer.forward_block(
                layer_in,
                &dmask,
                start_offsets,
                selection.as_ref(),
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
            )?;
            layer_in = xs_next;
            selection = sel;
        }
        let x = self.norm.as_ref().unwrap().forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        // MTP: stash the per-sampled-row hidden (post-norm, pre-output) for the in-band
        // `nextn` proposer. A single atomic load when speculation is off (mirrors deepseek4).
        if self
            .store_spec_hidden
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            if let Ok(mut h) = self.spec_hidden.lock() {
                *h = Some(x.clone());
            }
        }
        self.output.as_ref().unwrap().forward(&x.contiguous()?)
    }

    fn run_local_layers(&self, h: &Tensor, offsets: &[usize]) -> Result<Tensor> {
        let ids2d = h.narrow(2, 0, 1)?.squeeze(2)?;
        let mask = CausalMasker.make_causal_mask(
            &ids2d,
            &offsets as &dyn PastKvLenCache,
            self.dtype,
            &CausalMaskConfig::default(),
        )?;
        let mask = DeviceMappedMask::from_single(mask);
        let cache = &mut self.cache.normal().0;
        if !self.pp.as_ref().unwrap().is_head()
            && cache
                .first()
                .is_some_and(|c| c.current_seq_len() != offsets[0])
        {
            for c in cache.iter_mut() {
                c.reset();
            }
        }
        let mut layer_in = h.clone();
        let mut selection: Option<DsaSelection> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let dmask = mask.get(layer_in.device());
            let (xs_next, sel) =
                layer.forward_block(layer_in, &dmask, offsets, selection.as_ref(), &mut cache[i], None)?;
            layer_in = xs_next;
            selection = sel;
        }
        Ok(layer_in)
    }
}

impl ModelWeights {
    /// Base-model config, for loading the in-band `nextn` MTP head against it.
    pub(crate) fn base_props(&self) -> &PropsGGUF {
        &self.base_props
    }

    /// The model's compute dtype (the MTP head loads at the same dtype).
    pub(crate) fn compute_dtype(&self) -> DType {
        self.dtype
    }

    /// Base-model token embeddings (shared with the MTP head). Errors on a non-head
    /// pipeline-parallel shard, where they do not live.
    pub(crate) fn embeddings(&self) -> Result<Embedding> {
        self.tok_embeddings
            .clone()
            .ok_or_else(|| hanzo_ml::Error::msg("deepseek2: token embeddings not on this shard"))
    }

    /// Base-model output projection (shared with the MTP head). Errors on a non-head shard.
    pub(crate) fn output_head(&self) -> Result<Arc<dyn QuantMethod>> {
        self.output
            .clone()
            .ok_or_else(|| hanzo_ml::Error::msg("deepseek2: output head not on this shard"))
    }

    /// Shared handle to the normal KV cache, for `NormalSpeculativeCacheAccess`. Consumed by
    /// the pipeline's non-paged speculative dispatch (`try_sample_speculative_causal_gen`),
    /// which the CTO wires for `Model::Deepseek2` alongside the existing `Deepseek4` arm.
    #[allow(dead_code)]
    pub fn normal_cache_arc(
        &self,
    ) -> Option<std::sync::Arc<std::sync::Mutex<crate::pipeline::NormalCache>>> {
        self.cache.normal_arc()
    }

    /// Enable/disable stashing the pre-output hidden for MTP speculative proposal.
    /// Clearing also drops any stashed tensor so a finished step can't leak.
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
    /// Read by the pipeline's `speculative_target_hiddens`, which the CTO wires for
    /// `Model::Deepseek2` alongside the existing `Deepseek4` arm.
    #[allow(dead_code)]
    pub fn last_spec_hidden(&self) -> Option<Tensor> {
        self.spec_hidden.lock().ok().and_then(|h| h.clone())
    }
}

impl PipelineParallelModel for ModelWeights {
    fn ring(&self) -> &Arc<RingPipeline> {
        self.pp.as_ref().expect("pipeline parallel not enabled")
    }
    fn pp_device(&self) -> &Device {
        &self.device
    }
    fn pp_dtype(&self) -> DType {
        self.dtype
    }
    fn pp_embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.as_ref().unwrap().forward(tokens)
    }
    fn pp_run_local(&self, h: &Tensor, offsets: &[usize]) -> Result<Tensor> {
        self.run_local_layers(h, offsets)
    }
    fn pp_norm_head(&self, h: &Tensor, context_lens: Vec<(usize, usize)>) -> Result<Tensor> {
        let x = self.norm.as_ref().unwrap().forward(h)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.as_ref().unwrap().forward(&x.contiguous()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proves the load-bearing identity behind absorbed MLA: attention computed in the compressed
    /// latent space (kv_b folded into q via `w_uk` and out via `w_uv_t`) equals attention computed
    /// on the materialized per-head K/V. Exercises the real `absorb_weights_from_kv_b` extraction
    /// against an `UnquantLinear` `kv_b`, then runs both attentions in plain f32 so the comparison
    /// depends only on the algebra, not on any device kernel. This is the exactness guarantee that
    /// justifies wiring absorbed decode without a real GGUF on hand.
    #[test]
    fn absorbed_equals_materialized() -> Result<()> {
        let dev = Device::Cpu;
        let (h, t, kvl, nope, rope, vh) = (3usize, 5usize, 8usize, 4usize, 2usize, 4usize);
        let per_head = nope + vh;
        let scale = 1.0f32 / ((nope + rope) as f32).sqrt();

        // kv_b: [h*(nope+vh), kvl], the un-absorbed latent -> per-head [k_nope | v] projection.
        let kv_b_w = Tensor::randn(0f32, 1f32, (h * per_head, kvl), &dev)?;
        let kv_b: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(kv_b_w, None)),
        )?);

        // Compressed cache: normed latent `ckv` [t, kvl] and shared rope key `k_pe` [t, rope].
        let ckv = Tensor::randn(0f32, 1f32, (t, kvl), &dev)?;
        let k_pe = Tensor::randn(0f32, 1f32, (t, rope), &dev)?;
        // Single new query token, per head: [h, nope] nope part and [h, rope] rope part.
        let q_nope = Tensor::randn(0f32, 1f32, (h, nope), &dev)?;
        let q_pe = Tensor::randn(0f32, 1f32, (h, rope), &dev)?;

        // --- Materialized reference: reconstruct per-head k_nope / v from kv_b, then attend. ---
        let kv = kv_b.forward(&ckv)?.reshape((t, h, per_head))?;
        let k_nope = kv.narrow(D::Minus1, 0, nope)?.to_vec3::<f32>()?; // [t][h][nope]
        let v = kv.narrow(D::Minus1, nope, vh)?.to_vec3::<f32>()?; // [t][h][vh]
        let ckv_v = ckv.to_vec2::<f32>()?; // [t][kvl]
        let kpe_v = k_pe.to_vec2::<f32>()?; // [t][rope]
        let qn = q_nope.to_vec2::<f32>()?; // [h][nope]
        let qp = q_pe.to_vec2::<f32>()?; // [h][rope]

        let attend = |score: &[f32]| -> Vec<f32> {
            let m = score.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = score.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            exps.iter().map(|e| e / z).collect()
        };

        let mut out_mat = vec![vec![0f32; vh]; h];
        for hi in 0..h {
            let score: Vec<f32> = (0..t)
                .map(|ti| {
                    let sk: f32 = (0..nope).map(|d| qn[hi][d] * k_nope[ti][hi][d]).sum();
                    let sr: f32 = (0..rope).map(|r| qp[hi][r] * kpe_v[ti][r]).sum();
                    (sk + sr) * scale
                })
                .collect();
            let a = attend(&score);
            for (ti, &at) in a.iter().enumerate() {
                for d in 0..vh {
                    out_mat[hi][d] += at * v[ti][hi][d];
                }
            }
        }

        // --- Absorbed: fold kv_b into q (w_uk) / out (w_uv_t) and attend in latent space. ---
        let (w_uk, w_uv_t) =
            absorb_weights_from_kv_b(kv_b.as_ref(), &dev, h, kvl, nope, vh, DType::F32)?;
        let w_uk = w_uk.to_vec3::<f32>()?; // [h][nope][kvl]
        let w_uv_t = w_uv_t.to_vec3::<f32>()?; // [h][kvl][vh]

        let mut max_diff = 0f32;
        for hi in 0..h {
            // ql_nope = q_nope @ w_uk  -> [kvl]
            let ql: Vec<f32> = (0..kvl)
                .map(|i| (0..nope).map(|d| qn[hi][d] * w_uk[hi][d][i]).sum())
                .collect();
            let score: Vec<f32> = (0..t)
                .map(|ti| {
                    let sk: f32 = (0..kvl).map(|i| ql[i] * ckv_v[ti][i]).sum();
                    let sr: f32 = (0..rope).map(|r| qp[hi][r] * kpe_v[ti][r]).sum();
                    (sk + sr) * scale
                })
                .collect();
            let a = attend(&score);
            // attn_latent = Σ a_t ckv_t -> [kvl]; out = attn_latent @ w_uv_t -> [vh]
            let mut latent = vec![0f32; kvl];
            for (ti, &at) in a.iter().enumerate() {
                for i in 0..kvl {
                    latent[i] += at * ckv_v[ti][i];
                }
            }
            for d in 0..vh {
                let o: f32 = (0..kvl).map(|i| latent[i] * w_uv_t[hi][i][d]).sum();
                max_diff = max_diff.max((o - out_mat[hi][d]).abs());
            }
        }

        assert!(
            max_diff < 1e-4,
            "absorbed vs materialized MLA diverged: max_diff={max_diff}"
        );
        Ok(())
    }
}
