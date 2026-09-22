#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::pipeline::KvCache;
use crate::{attention::backends::cpu, pipeline::text_models_inputs_processor::FlashParams};

use hanzo_ml::{DType, Device, Result, Tensor};

/// Decode command-graph attention context (shared scale + meta buffers). Vulkan-only; the uninhabited
/// stand-in keeps the model forward signatures backend-uniform off-Vulkan, where the graph path is
/// never taken. One canonical alias so every GGUF text model names the same type.
#[cfg(feature = "vulkan")]
pub(crate) use hanzo_ml::VkGraphAttn;
#[cfg(not(feature = "vulkan"))]
pub(crate) enum VkGraphAttn {}

/// Attention mask passed to [`Sdpa::run_attention`].
///
/// Encodes both the mask data and the *intent*, whether the attention layer
/// should use flash attention (causal handled by the kernel), eager attention
/// with an explicit mask tensor, or no masking at all.
#[derive(Clone, Debug)]
pub enum AttentionMask {
    /// No masking. Used for single-token decode or truly unmasked attention.
    None,
    /// Flash attention with `is_causal = true`. No mask tensor is needed;
    /// the flash kernel applies causal masking internally. Also signals
    /// "this is a prefill" to the paged attention layer.
    CausalFlash,
    /// An explicit mask tensor (causal, sliding window, bidirectional, etc).
    /// Dispatches to the eager (non-flash) attention path.
    Custom(Tensor),
}

impl AttentionMask {
    /// Extract the inner tensor as `Option<&Tensor>`.
    ///
    /// Returns `Some(&tensor)` for [`Custom`](Self::Custom), `None` otherwise.
    /// Useful for interfacing with paged-attention and MLA helpers that still
    /// accept `Option<&Tensor>`.
    pub fn as_option_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Custom(t) => Some(t),
            _ => None,
        }
    }

    /// Returns `true` when the mask carries an explicit tensor
    /// ([`Custom`](Self::Custom) variant), mirroring the old
    /// `Option<Tensor>::is_some()` semantics.
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(_))
    }
}

mod backends;

#[allow(unused)]
pub(crate) use backends::{flash_attn, maybe_synchronize, naive_sdpa, sinks_attn, tiled_sdpa};

/// Chunk size for attention computation to avoid OOM on long sequences
/// Whether prompt attention over a gathered prefix runs a fused varlen kernel, which takes causality
/// as a flag. Every other path is eager and is causal only through its mask.
pub(crate) fn fused_varlen(device: &Device, dtype: DType) -> bool {
    device.is_cpu() || (device.is_cuda() && crate::using_flash_attn() && dtype != DType::F32)
}

pub(crate) const ATTENTION_CHUNK_SIZE: usize = 1024;

/// Key chunk for [`tiled_sdpa`]. Query chunking alone leaves a block that is linear in the context,
/// so the key axis is chunked too once the whole block stops fitting.
pub(crate) const ATTENTION_KV_CHUNK_SIZE: usize = 4096;

/// Bytes of f32 scores one eager attention block may hold before it is computed in tiles instead.
/// Above this a long prefill cannot be served at all: a 4096-token chunk against a 124K prefix is 24
/// heads x 4096 x 124K x 4 B. Below it the single-block path is left exactly as it is, so the shapes
/// that work today keep their arithmetic.
pub(crate) const ATTENTION_SCORE_BLOCK_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// The same ceiling on ROCm, where 2 GiB is not a safe operating point but the edge of a cliff.
///
/// The eager path holds about 5.6 block-sized temporaries at once (scores, their f32 copy, the
/// masked sum, the softmax, its cast back), so a block just under the general ceiling is an ~11 GiB
/// transient. On a discrete card that is VRAM nobody else wanted. On an APU it is GTT, which is
/// SYSTEM memory charged to no cgroup: it comes out of the same pool as the host and everything the
/// node serves, and nothing but the OOM killer bounds it.
///
/// It also never came back. A chunked prefill asks for a block of `n_heads * 1024 * kv_len * 4`
/// bytes and `kv_len` grows every chunk, so each chunk's temporaries are a size no earlier chunk
/// used; a pool that reuses a freed buffer only on an exact byte match reuses none of them. Summed
/// over the chunks of one prompt that is quadratic in its length -- measured on a 27B with 24 heads
/// at 0.275 GB per (1K tokens)^2: 7 GB at 4K, 23 GB at 8K, ~82 GB at 16K, taking down a node that
/// also served production. 5.6 x 94 MiB per 1K of context, halved by the sum, is that coefficient.
///
/// Tiling fixes both at once, because a tile is the SAME size whatever the context: the transient
/// stops growing with `kv_len`, and every chunk and every layer asks for buffers the last one just
/// freed. 256 MiB puts the switch at ~2.7K tokens of context for that model, and a full tile under
/// it (see [`kv_tile_within`]), so the largest attention transient is a few hundred MiB at any
/// context length instead of ~11 GiB at 21K.
pub(crate) const ROCM_SCORE_BLOCK_BYTES: usize = 256 * 1024 * 1024;

/// The score-block ceiling for `device`. Every other device keeps the general one, so the shapes
/// that work there today keep their arithmetic.
fn score_block_budget(device: &Device) -> usize {
    if device.is_rocm() {
        ROCM_SCORE_BLOCK_BYTES
    } else {
        ATTENTION_SCORE_BLOCK_BYTES
    }
}

/// Whether one [q_len, kv_len] block of f32 scores fits `budget` bytes.
fn score_block_fits(
    b_sz: usize,
    n_heads: usize,
    q_len: usize,
    kv_len: usize,
    budget: usize,
) -> bool {
    b_sz.saturating_mul(n_heads)
        .saturating_mul(q_len)
        .saturating_mul(kv_len)
        .saturating_mul(std::mem::size_of::<f32>())
        <= budget
}

/// [`naive_sdpa`] with the query axis taken `rows` at a time. The result is the same for any `rows`;
/// what changes is the [rows, kv_len] block in flight, which is how a caller holds it under a ceiling
/// without giving up the fused softmax.
///
/// It IS `naive_sdpa`, called once per run: a run is never longer than [`ATTENTION_CHUNK_SIZE`], so
/// the chunking inside it passes each one straight through. There is no second copy of the
/// arithmetic to drift from the first.
fn naive_sdpa_rows(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
    rows: usize,
) -> Result<Tensor> {
    chunked_attention_rows(
        q,
        k,
        v,
        mask,
        rows.min(ATTENTION_CHUNK_SIZE),
        |q, k, v, mask| naive_sdpa(q, k, v, mask, sdpa_params),
    )
}

/// How one eager attention block is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Blocking {
    /// The whole [q_len, kv_len] block at once.
    Whole,
    /// The query axis in runs of this many rows, each an ordinary fused softmax over the full key
    /// row. Softmax rows are independent, so where the query axis is cut changes nothing about the
    /// result: this is the SAME arithmetic as [`Blocking::Whole`], not an approximation of it.
    Rows(usize),
    /// Both axes, recombined with an online softmax. What is left once a single query row no longer
    /// fits, and the only blocking whose transient does not grow with the context at all.
    Tiles { q: usize, kv: usize },
}

/// Fewest query rows worth a launch. Below this the run count is the cost -- 1024 queries in runs of
/// 8 is 128 launches per layer -- and cutting the key axis instead is the better trade.
const MIN_Q_ROWS: usize = 16;

/// The largest power-of-two run of query rows whose [rows, kv_len] block fits `budget`, or 0.
fn rows_within(b_sz: usize, n_heads: usize, kv_len: usize, budget: usize) -> usize {
    let row_bytes = b_sz
        .saturating_mul(n_heads)
        .saturating_mul(kv_len)
        .saturating_mul(std::mem::size_of::<f32>())
        .max(1);
    match budget / row_bytes {
        0 => 0,
        fits => 1usize << (usize::BITS - 1 - fits.leading_zeros()),
    }
}

/// Decides how a [q_len, kv_len] block is computed under `budget`. Pure, so the policy is testable
/// without the device it is for.
///
/// `by_rows` is whether the query axis may be cut finer than [`ATTENTION_CHUNK_SIZE`]. Where it may,
/// that is preferred to tiling both axes: it keeps the fused softmax and the exact arithmetic, and
/// its blocks are near-constant in size (`rows * kv_len` tracks the budget), where the online softmax
/// costs about a dozen launches and three extra f32 passes over the scores per tile. It is declined
/// once the runs get too short to be worth launching.
pub(crate) fn blocking(
    budget: usize,
    by_rows: bool,
    b_sz: usize,
    n_heads: usize,
    q_len: usize,
    kv_len: usize,
) -> Blocking {
    if score_block_fits(b_sz, n_heads, q_len, kv_len, budget) {
        return Blocking::Whole;
    }
    if by_rows {
        let rows = rows_within(b_sz, n_heads, kv_len, budget);
        if rows >= MIN_Q_ROWS {
            return Blocking::Rows(rows.min(ATTENTION_CHUNK_SIZE));
        }
    }
    Blocking::Tiles {
        q: ATTENTION_CHUNK_SIZE,
        kv: kv_tile_within(b_sz, n_heads, ATTENTION_CHUNK_SIZE, budget),
    }
}

/// Smallest key tile worth a kernel launch. Below this the tile count, not the block size, is the
/// cost, and a budget that small is a misconfiguration rather than something to honour exactly.
const MIN_KV_TILE: usize = 256;

/// The key tile for a block that did not fit `budget`: the largest power of two, up to
/// [`ATTENTION_KV_CHUNK_SIZE`], whose [q_tile, kv_tile] block does fit.
///
/// A block sent to the tiled path for exceeding a ceiling must come back in pieces that respect it,
/// or the ceiling bounds nothing: with 24 heads a [1024, 4096] tile is 393 MiB, over a 256 MiB
/// budget it was meant to honour. A power of two keeps every full tile the same size across chunks
/// and layers, which is what lets a freed tile be handed straight to the next one.
fn kv_tile_within(b_sz: usize, n_heads: usize, q_tile: usize, budget: usize) -> usize {
    let row_bytes = b_sz
        .saturating_mul(n_heads)
        .saturating_mul(q_tile)
        .saturating_mul(std::mem::size_of::<f32>())
        .max(1);
    let fits = (budget / row_bytes).max(1);
    // Largest power of two <= fits.
    let tile = 1usize << (usize::BITS - 1 - fits.leading_zeros());
    tile.clamp(MIN_KV_TILE, ATTENTION_KV_CHUNK_SIZE)
}

/// Generic chunked attention computation that can be used by different backends
pub(crate) fn chunked_attention<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    attention_fn: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>) -> Result<Tensor>,
{
    chunked_attention_rows(q, k, v, mask, ATTENTION_CHUNK_SIZE, attention_fn)
}

/// [`chunked_attention`] with the run length chosen by the caller. `rows` is how many query rows
/// one call to `attention_fn` takes. Each row's softmax is its own, so the result does not depend
/// on it; only the size of the block in flight does.
pub(crate) fn chunked_attention_rows<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    rows: usize,
    attention_fn: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>) -> Result<Tensor>,
{
    let seq_len = q.dim(2)?;
    let rows = rows.max(1);

    if seq_len <= rows {
        // For short sequences, use the regular path
        return attention_fn(q, k, v, mask);
    }

    // Chunk the query to avoid OOM on long sequences
    let num_chunks = seq_len.div_ceil(rows);
    let mut attn_chunks = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let offset = chunk_idx * rows;
        let chunk_len = rows.min(seq_len - offset);

        // Extract query chunk
        let q_chunk = q.narrow(2, offset, chunk_len)?;

        // Extract mask chunk if present
        // The query axis is second from last at every rank: (q, kv), (b, q, kv), (b, h, q, kv).
        let mask_chunk = mask
            .map(|m| {
                let q_dim = m.rank().saturating_sub(2);
                if m.dim(q_dim)? == 1 {
                    // Broadcast over the query axis: every run takes the one row there is.
                    Ok(m.clone())
                } else {
                    m.narrow(q_dim, offset, chunk_len)
                }
            })
            .transpose()?;

        // Compute attention for this chunk
        let att_chunk = attention_fn(&q_chunk, k, v, mask_chunk.as_ref())?;

        attn_chunks.push(att_chunk);
    }

    // Concatenate all chunks along the sequence dimension
    Tensor::cat(&attn_chunks, 2)
}

pub(crate) fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

// Vulkan single-query (decode) attention: one fused on-GPU kernel (online softmax, GQA-aware) in
// place of repeat_kv + QK^T bmm + softmax + *V bmm + the contiguous glue (~10 dispatches/layer -> 1).
// Returns None (caller falls back to eager) unless: on Vulkan, q_len==1, head_dim is a power of two
// <= 128, and there's no softcap/sliding-window (the kernel handles neither). q:[B,H,1,D] attends the
// full cache k/v:[B,Hkv,L,D]; a lone decode query sees only past keys so no mask is needed.
#[cfg(feature = "vulkan")]
fn vulkan_decode_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sdpa_params: &SdpaParams,
) -> Result<Option<Tensor>> {
    use hanzo_ml::{DType, Device};
    if !crate::perf_flags::vulkan_fused_attn_enabled() {
        return Ok(None);
    }
    let (_b, h, q_len, d) = q.dims4()?;
    // The committed `sdpa_blk` .spv is f32 and specialized for head_dim 128 (d and the block width are
    // comptime in the kernel), so gate to f32 + 128; other dtypes/head dims need their own kernel. No
    // softcap / sliding-window (the kernel does neither). A lone decode query (q_len==1) attends the
    // full cache, so no attention mask is needed.
    if q_len != 1
        || d != 128
        || q.dtype() != DType::F32
        || k.dtype() != DType::F32
        || v.dtype() != DType::F32
        || sdpa_params.softcap.is_some()
        || sdpa_params.sliding_window.is_some()
    {
        return Ok(None);
    }
    if !matches!(q.device(), Device::Vulkan(_)) {
        return Ok(None);
    }
    let hkv = k.dim(1)?;
    if hkv == 0 || h % hkv != 0 {
        return Ok(None);
    }
    // One fused `sdpa_blk` dispatch replaces repeat_kv(copy2d) + QKᵀ bmm + softmax + ·V bmm: the op's
    // vulkan_fwd (hanzo_nn::ops::Sdpa) runs the GQA-native online-softmax kernel. q is a single decode
    // token -- cheap to materialize contiguous. k/v are the KV cache: pass them STRIDED (read in place)
    // when their layout is kernel-friendly (innermost head_dim contiguous, offset 0, matching strides),
    // which skips the per-layer `.contiguous()` copy of the whole active cache -- the dominant decode
    // cost. Otherwise materialize contiguous so the stride path always sees a layout it can read.
    // Non-causal: a decode query sees only past keys, so the whole cache is unmasked.
    let q = q.contiguous()?;
    let kv_readable = |t: &Tensor| t.layout().start_offset() == 0 && t.stride().last() == Some(&1);
    let (k, v) = if kv_readable(k) && kv_readable(v) && k.stride() == v.stride() {
        (k.clone(), v.clone())
    } else {
        (k.contiguous()?, v.contiguous()?)
    };
    let out = hanzo_nn::ops::sdpa(&q, &k, &v, None, false, sdpa_params.softmax_scale, 1.0)?;
    Ok(Some(out))
}

// ROCm single-query (decode) attention: one fused GQA-aware online-softmax kernel in place of
// repeat_kv (cat -> copy2d) + QK^T bmm + softmax + *V bmm + the per-layer `.contiguous()` copy of the
// whole active KV cache (~10 dispatches + ~653 copyBufferRectAligned/token -> 1). Returns None (caller
// falls back to eager) unless: ROCm device, q_len==1, head_dim 128, f16/bf16, no softcap, no sliding
// window. q:[B,H,1,128] attends the full cache k/v:[B,Hkv,L,128]; k/v are read STRIDED in place when
// the head dim is contiguous (the win), else materialized contiguous. A lone decode query sees only
// past keys, so the whole cache is unmasked.
#[cfg(feature = "rocm")]
fn rocm_decode_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sdpa_params: &SdpaParams,
) -> Result<Option<Tensor>> {
    use hanzo_ml::DType;
    let (_b, h, q_len, d) = q.dims4()?;
    if q_len != 1
        || !(d == 128 || d == 256)
        || !matches!(q.dtype(), DType::F16 | DType::BF16)
        || q.dtype() != k.dtype()
        || q.dtype() != v.dtype()
        || sdpa_params.softcap.is_some_and(|x| x != 1.0)
        || sdpa_params.sliding_window.is_some()
    {
        return Ok(None);
    }
    if !q.device().is_rocm() {
        return Ok(None);
    }
    let hkv = k.dim(1)?;
    if hkv == 0 || h % hkv != 0 || k.dim(3)? != d || v.dim(3)? != d {
        return Ok(None);
    }
    // Read k/v in place when the head dim is contiguous (the kernel handles arbitrary batch/head/seq
    // strides + start offset); otherwise materialize contiguous so the kernel always sees stride 1.
    let dim_contig = |t: &Tensor| t.stride().last() == Some(&1);
    let (k, v) = if dim_contig(k) && dim_contig(v) {
        (k.clone(), v.clone())
    } else {
        (k.contiguous()?, v.contiguous()?)
    };
    let out = hanzo_nn::attention::rocm_flash_attn_decode(q, &k, &v, sdpa_params.softmax_scale)?;
    Ok(Some(out))
}

pub struct SdpaParams {
    pub n_kv_groups: usize,
    pub softcap: Option<f32>,
    pub softmax_scale: f32,
    pub sliding_window: Option<usize>,
    pub sinks: Option<Tensor>,
}

pub struct Sdpa;

impl Sdpa {
    /// Computes softmax(QK^T*sqrt(d_k))V
    ///
    /// Inputs:
    /// - q: (b_sz, n_attn_heads, q_len, head_dim)
    /// - k: (b_sz, n_kv_heads, q_len, head_dim)
    /// - v: (b_sz, n_kv_heads, q_len, head_dim)
    ///
    /// Dispatch attention based on the `AttentionMask` variant:
    ///
    /// - `AttentionMask::CausalFlash`: flash attention with `is_causal = true`
    /// - `AttentionMask::None`: flash if available (decode), else eager without mask
    /// - `AttentionMask::Custom`: eager attention with the explicit mask tensor
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &AttentionMask,
        flash_params: Option<&FlashParams>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        // If sinks are present, dispatch to the sinks backend
        if let Some(sinks) = &sdpa_params.sinks {
            let mask_tensor = match mask {
                AttentionMask::Custom(t) => Some(t),
                _ => None,
            };
            return sinks_attn(q, k, v, sinks, mask_tensor, flash_params, sdpa_params);
        }

        // Vulkan decode fast-path: one fused kernel replaces repeat_kv + QK^T bmm + softmax + *V bmm
        // + contiguous glue. Plain causal/no-mask single-query only; else falls through to eager.
        #[cfg(feature = "vulkan")]
        if matches!(mask, AttentionMask::None | AttentionMask::CausalFlash) {
            if let Some(out) = vulkan_decode_attn(q, k, v, sdpa_params)? {
                return Ok(out);
            }
        }

        // ROCm decode fast-path: same gating as Vulkan (single query, no explicit mask needed since a
        // lone decode query attends the full past cache). Fuses repeat_kv + QK^T + softmax + *V and
        // reads the KV cache strided in place, eliminating the copy2d glue that dominates ROCm decode.
        #[cfg(feature = "rocm")]
        if matches!(mask, AttentionMask::None | AttentionMask::CausalFlash) {
            if let Some(out) = rocm_decode_attn(q, k, v, sdpa_params)? {
                return Ok(out);
            }
        }

        // The mask carries causality already; the kernel-level do_causal
        // early-exit is safe to enable only when the request is known causal.
        let do_causal = flash_params.is_some_and(|p| p.causal);
        // A request is explicitly NON-causal only when flash_params is present and says so:
        // bidirectional vision/audio encoders pass `FlashParams::empty(false)`. A causal decoder
        // passes `None` (quantized text models) or `causal = true`. This lets us tell a causal
        // Custom mask from a bidirectional one without inspecting the tensor, so we never apply
        // causal masking to a bidirectional encoder (which would corrupt its output).
        let explicitly_noncausal = flash_params.is_some_and(|p| !p.causal);

        // ROCm WMMA flash-attention: causal prefill at long sequences wins 1.23-1.49x over
        // rocBLAS+softmax on gfx1151. A non-SWA causal mask (Custom from the causal masker) or
        // CausalFlash is full-causal, so the kernel applies causality internally and the explicit
        // mask is dropped. SWA, non-causal (None+!do_causal), short seqs, and head_dim != 128 fall
        // through to the eager path. The kernel does GQA, so it takes the un-expanded k/v.
        #[cfg(feature = "rocm")]
        if q.device().is_rocm() && !matches!(mask, AttentionMask::None if !do_causal) {
            const ROCM_FLASH_MIN_SEQ: usize = 512;
            let (_, _, seq_len, head_dim) = q.dims4()?;
            let is_full_causal = matches!(mask, AttentionMask::CausalFlash)
                || (mask.is_custom()
                    && sdpa_params.sliding_window.is_none()
                    && !explicitly_noncausal);
            if is_full_causal
                && seq_len >= ROCM_FLASH_MIN_SEQ
                && (head_dim == 128 || head_dim == 256)
                && k.dim(3)? == head_dim
                && v.dim(3)? == head_dim
                && matches!(q.dtype(), DType::F16 | DType::BF16)
                && sdpa_params.softcap.is_none_or(|x| x == 1.0)
            {
                return hanzo_nn::attention::rocm_flash_attn(
                    q,
                    k,
                    v,
                    sdpa_params.softmax_scale,
                    true,
                );
            }
        }

        // Custom mask, eager attention (flash can't use arbitrary mask tensors). Reached under
        // FLASH_PREFILL=0, for GGUF/quantized models (CausalMaskConfig::gguf: no FlashParams plumbing,
        // flash corrupts their prefill), and force_custom models (gpt-oss). Dense safetensors otherwise
        // emit CausalFlash.
        if let AttentionMask::Custom(mask_tensor) = mask {
            return self.run_attention_noflash(q, k, v, Some(mask_tensor), sdpa_params, do_causal);
        }

        // CausalFlash or None: try flash attention, fall back to eager. `using_flash_attn()` is default
        // ON (flash-attn >= 0.11.35 f16-routes bf16 P, curing the old Q4K collapse). FLASH_PREFILL=0
        // forces eager at every flash site. See using_flash_attn() in utils/mod.rs.
        let can_use_flash = q.device().is_cpu()
            || q.device().is_cuda() && crate::using_flash_attn() && q.dtype() != DType::F32;

        if can_use_flash {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;

            if q.device().is_cpu() {
                match q.dtype() {
                    DType::F32 => {
                        return cpu::run_flash_attn_cpu::<f32>(&q, &k, &v, None, sdpa_params);
                    }
                    DType::F16 => {
                        return cpu::run_flash_attn_cpu::<half::f16>(&q, &k, &v, None, sdpa_params)
                    }
                    DType::BF16 => {
                        return cpu::run_flash_attn_cpu::<half::bf16>(
                            &q,
                            &k,
                            &v,
                            None,
                            sdpa_params,
                        );
                    }
                    _ => {
                        return Err(hanzo_ml::Error::Msg("Unsupported data type".into()));
                    }
                }
            } else {
                // flash_attn_v2 hands q/k/v straight to the CUDA kernel with no internal contiguous,
                // and the (b,H,s,d)->(b,s,H,d) transpose above is NON-CONTIGUOUS -> for seq>1 (prefill)
                // the kernel reads wrong strides and returns GARBAGE (decode seq==1 is trivially fine,
                // which is why only prefill corrupted the KV cache and the whole generation garbled).
                let q = q.contiguous()?;
                let k = k.contiguous()?;
                let v = v.contiguous()?;
                return flash_attn(&q, &k, &v, flash_params, sdpa_params)?.transpose(1, 2);
            }
        }

        self.run_attention_noflash(q, k, v, None, sdpa_params, do_causal)
    }

    /// Same as `run_attention`, but skips the flash-attention dispatch.
    ///
    /// `causal` tells the Metal SDPA-full kernel to enable its upper-triangle skip (`do_causal=true`).
    /// Pass `true` only when the caller's mask is causal-or-stricter.
    /// Pass false` for bidirectional masks (e.g. vision attention).
    #[allow(unused_variables, clippy::too_many_arguments)]
    pub fn run_attention_noflash(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        sdpa_params: &SdpaParams,
        causal: bool,
    ) -> Result<Tensor> {
        let (b_sz, n_attn_heads, seq_len, head_dim) = q.dims4()?;
        let (_, _, _, k_head_dim) = k.dims4()?;
        let (_, _, _, v_head_dim) = v.dims4()?;

        // We can use Metal SDPA (vector/full) if the mask is the correct size and head dims match.
        // If the mask is provided, then softcapping isn't allowed - default back to naive SDPA
        // Softcapping is implemented for vector SDPA.
        let all_head_dims_match = head_dim == k_head_dim && k_head_dim == v_head_dim;
        let tgt_mask_shape = vec![b_sz, n_attn_heads, seq_len, k.dim(2)?];
        let can_use_mask = mask.is_none_or(|mask| {
            mask.layout().broadcast_as(tgt_mask_shape.clone()).is_ok()
                && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        });
        let valid_head_dims: &[usize] = &[32, 64, 72, 80, 96, 128, 256, 512];
        // The Metal steel_attention (full) kernel handles q_seq != kv_seq via its qL_off, so the
        // non-square masked case (speculative-decode verify: gamma+1 queries vs a longer cache)
        // can use the fast kernel as long as q_seq <= kv_seq. The single-query decode path
        // (seq_len==1) keeps using the vector kernel. q_seq > kv_seq has no valid qL_off and stays
        // on naive_sdpa.
        let metal_supports_mask = mask.is_none() || seq_len <= k.dim(2)?;

        // Metal FA path for DK=512 BF16 with a mask. Two specializations:
        // prefill (seq_len > 8) goes through the BlockMMA kernel; decode
        // (seq_len == 1) uses a vector FA kernel ported from llama.cpp.
        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && head_dim == 512
            && k_head_dim == 512
            && v_head_dim == 512
            && q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && seq_len == 1
            && mask.is_some()
            && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        {
            if let Some(out) =
                crate::attention::backends::metal_flash_attn::try_flash_attn_ext_vec_bf16_dk512(
                    q,
                    k,
                    v,
                    mask,
                    sdpa_params.softmax_scale,
                )?
            {
                return Ok(out);
            }
        }
        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && head_dim == 512
            && k_head_dim == 512
            && v_head_dim == 512
            && q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && seq_len > 8
            && sdpa_params.softcap.is_none_or(|x| x == 1.0)
        {
            if let Some(mask) = mask {
                if let Some(out) =
                    crate::attention::backends::metal_flash_attn::try_flash_attn_ext_bf16_dk512(
                        q,
                        k,
                        v,
                        mask,
                        sdpa_params.softmax_scale,
                    )?
                {
                    return Ok(out);
                }
            }
        }

        if [q, k, v].into_iter().all(|x| x.device().is_metal())
            && all_head_dims_match
            && valid_head_dims.contains(&head_dim)
            && can_use_mask
            && metal_supports_mask
            && !(head_dim == 512 && seq_len > 8)
        {
            let mask = match mask {
                Some(mask) => Some(mask.broadcast_as(tgt_mask_shape)?),
                None => None,
            };
            // do_causal lets the steel_attention kernel bound its kb-loop to
            // the per-query position, skipping the upper triangle of Q*K^T
            // entirely (roughly halves matmul cost for prefill).
            let do_causal = seq_len > 1 && causal;
            return hanzo_nn::ops::sdpa(
                q,
                k,
                v,
                mask.as_ref(),
                do_causal,
                sdpa_params.softmax_scale,
                sdpa_params.softcap.unwrap_or(1.0),
            );
        }

        // Every path below materializes the whole score block, which is what a long prefill cannot
        // pay for; past the budget the same attention is computed over tiles.
        // Only ROCm cuts the query axis finer: it is the one device whose ceiling is low enough to
        // need it, and leaving the rest alone keeps the arithmetic of every shape that works today.
        let rows = match blocking(
            score_block_budget(q.device()),
            q.device().is_rocm(),
            b_sz,
            n_attn_heads,
            seq_len,
            k.dim(2)?,
        ) {
            Blocking::Tiles { q: q_tile, kv } => {
                return tiled_sdpa(q, k, v, mask, sdpa_params, q_tile, kv);
            }
            Blocking::Rows(rows) => rows,
            Blocking::Whole => ATTENTION_CHUNK_SIZE,
        };

        let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;

        if mask.is_some_and(|x| x.rank() == 2) || hanzo_quant::distributed::use_nccl() {
            return naive_sdpa_rows(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                mask,
                sdpa_params,
                rows,
            );
        }

        // TODO: bench?
        #[allow(unused)]
        if let (Device::Cuda(_), Some(cublaslt)) = (
            q.device(),
            hanzo_quant::cublaslt::CUBLASLT_CONTROLLER.get_for_device(q.device()),
        ) {
            #[cfg(feature = "cuda")]
            {
                maybe_synchronize(q.device())?;

                // Use chunked attention for cuBLASLt path
                let k_flat = k.flatten(0, 1)?;
                let v_flat = v.flatten(0, 1)?;

                chunked_attention(q, &k, &v, mask, |q_chunk, _k, _v, mask_chunk| {
                    // cuBLASLt batch matmul implementation requires inputs to be dims3
                    let (chunk_b_sz, chunk_n_heads, chunk_seq_len, chunk_head_dim) =
                        q_chunk.dims4()?;
                    let q_flat = q_chunk.flatten(0, 1)?;

                    let attention_bias = match mask_chunk {
                        Some(mask) if mask.rank() == 3 && mask.dims()[0] == 1 => {
                            Some(mask.repeat((chunk_n_heads, 1, 1))?)
                        }
                        Some(mask) if mask.rank() == 3 => Some(mask.clone()),
                        Some(mask) if mask.rank() == 4 => {
                            let tgt_shape =
                                vec![chunk_b_sz, chunk_n_heads, chunk_seq_len, k.dim(2)?];
                            Some(mask.broadcast_as(tgt_shape)?.flatten(0, 1)?)
                        }
                        Some(mask) => {
                            hanzo_ml::bail!("cublaslt attn mask: rank must be 3 or 4")
                        }
                        None => None,
                    };

                    // If attention_bias is set, we fuse the add by giving it as the output matrix
                    // and setting beta to 1.0
                    let beta = match attention_bias.is_some() {
                        true => Some(1.0),
                        false => None,
                    };

                    // Batch matrix multiplication
                    // Fuse softmax scale and attention_bias add
                    let mut attention_scores = cublaslt.batch_matmul(
                        &k_flat,
                        &q_flat,
                        attention_bias.as_ref(),
                        Some(sdpa_params.softmax_scale / sdpa_params.softcap.unwrap_or(1.0)),
                        beta,
                        None,
                        None,
                    )?;
                    if let Some(softcap) = sdpa_params.softcap {
                        attention_scores = (attention_scores.tanh()? * softcap as f64)?;
                    }
                    // Compute softmax in F32 for precision. BF16's 7 mantissa
                    // bits cause exp() to lose information on long sequences.
                    // Flash attention already computes softmax in F32; this
                    // matches that behaviour for the eager path.
                    let scores_dtype = attention_scores.dtype();
                    if scores_dtype == DType::BF16 || scores_dtype == DType::F16 {
                        attention_scores = attention_scores.to_dtype(DType::F32)?;
                    }
                    attention_scores = hanzo_nn::ops::softmax_last_dim(&attention_scores)?;
                    if attention_scores.dtype() != scores_dtype {
                        attention_scores = attention_scores.to_dtype(scores_dtype)?;
                    }

                    let context_layer = cublaslt.batch_matmul(
                        &v_flat.t()?.contiguous()?,
                        &attention_scores,
                        // We save one allocation
                        Some(&q_flat),
                        None,
                        None,
                        None,
                        None,
                    )?;

                    // Reshape to dims4
                    context_layer.reshape((chunk_b_sz, chunk_n_heads, chunk_seq_len, v_head_dim))
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                hanzo_ml::bail!("`cuda` feature is not enabled")
            }
        } else {
            naive_sdpa_rows(q, &k, &v, mask, sdpa_params, rows)
        }
    }
}

/// SDPA over the naive (non-paged) contiguous KV cache -- the shared decode/prefill attention tail for
/// the GGUF dense and MoE text models. When a decode command-graph attention context is supplied
/// (`vk_graph`, a Vulkan capture in flight) the new K/V is appended at the DEVICE-offset slot read from
/// `positions` and the fused shared-meta SDPA attends the growing `[0, seq_k)` span, so a captured
/// forward replays with the advancing write slot and span; otherwise the eager host-offset append + SDPA
/// runs. One home so both the dense and MoE models dispatch naive-cache attention identically.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_naive_cache(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &AttentionMask,
    kv_cache: &mut KvCache,
    positions: &Tensor,
    sdpa_params: &SdpaParams,
    vk_graph: Option<&VkGraphAttn>,
    b: usize,
    n_head: usize,
    head_dim: usize,
) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    if let Some(g) = vk_graph {
        // Contiguous q/k/v so the device-offset append and the fused SDPA read packed rows.
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let (k_full, v_full) = kv_cache.append_graph(&k, &v, positions)?;
        let q = q.contiguous()?;
        return vk_sdpa_graph(&q, &k_full, &v_full, g, b, n_head, head_dim);
    }
    #[cfg(not(feature = "vulkan"))]
    let _ = (positions, vk_graph, b, n_head, head_dim);
    let (k, v) = kv_cache.append(k, v)?;
    Sdpa.run_attention(q, &k, &v, mask, None, sdpa_params)
}

/// Fused GQA flash SDPA over the FULL fixed-shape KV cache for the decode command-graph: binds the
/// caller-owned shared `scale`/`meta` (seq_k advances per replay) and a fresh per-layer `out`, so a
/// captured forward attends the growing span without re-recording. `q`/`k_full`/`v_full` are the decode
/// query and the whole cache; returns `out` [b, n_head, 1, head_dim] -- the shape the eager SDPA yields.
#[cfg(feature = "vulkan")]
fn vk_sdpa_graph(
    q: &Tensor,
    k_full: &Tensor,
    v_full: &Tensor,
    g: &VkGraphAttn,
    b: usize,
    n_head: usize,
    head_dim: usize,
) -> Result<Tensor> {
    use hanzo_ml::{backend::BackendStorage, Storage};
    let out = Tensor::zeros((b, n_head, 1, head_dim), DType::F32, q.device())?;
    {
        let (qs, _) = q.storage_and_layout();
        let (ks, _) = k_full.storage_and_layout();
        let (vs, _) = v_full.storage_and_layout();
        let (os, _) = out.storage_and_layout();
        let (
            Storage::Vulkan(q_vk),
            Storage::Vulkan(k_vk),
            Storage::Vulkan(v_vk),
            Storage::Vulkan(o_vk),
        ) = (&*qs, &*ks, &*vs, &*os)
        else {
            hanzo_ml::bail!("vk sdpa graph: expected vulkan storage for q/k/v/out");
        };
        // Flash-decoding (default): split the KV sequence across n_split workgroups per head so the lone
        // decode query fills the GPU (the one-workgroup-per-head sdpa_blk runs at 256 VGPR / min
        // occupancy). VK_SDPA_SPLIT_OFF reverts to sdpa_blk_vk_graph for A/B; n_split via VK_SDPA_NSPLIT.
        if vk_sdpa_split_graph() {
            q_vk.device().sdpa_decode_split_vk_graph(
                q_vk,
                k_vk,
                v_vk,
                o_vk,
                g.meta(),
                b,
                n_head,
                g.n_kv(),
                head_dim,
                g.softmax_scale(),
                vk_sdpa_nsplit(),
                g.kv_batch_stride(),
                g.kv_head_stride(),
                g.key_stride(),
            )?;
        } else {
            q_vk.device().sdpa_blk_vk_graph(
                q_vk,
                k_vk,
                v_vk,
                o_vk,
                g.scale(),
                g.meta(),
                b,
                n_head,
                1,
            )?;
        }
    }
    Ok(out)
}

// Flash-decoding graph toggle (cached): default ON, VK_SDPA_SPLIT_OFF=1 reverts to sdpa_blk for A/B.
#[cfg(feature = "vulkan")]
fn vk_sdpa_split_graph() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("VK_SDPA_SPLIT_OFF")
            .map(|v| v == "0")
            .unwrap_or(true)
    })
}
#[cfg(feature = "vulkan")]
fn vk_sdpa_nsplit() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VK_SDPA_NSPLIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(8)
    })
}

#[cfg(test)]
mod score_block_budget {
    use super::{
        blocking, kv_tile_within, score_block_budget, score_block_fits, Blocking,
        ATTENTION_CHUNK_SIZE, ATTENTION_KV_CHUNK_SIZE, ATTENTION_SCORE_BLOCK_BYTES as GENERAL,
        MIN_KV_TILE, MIN_Q_ROWS, ROCM_SCORE_BLOCK_BYTES as ROCM,
    };
    use hanzo_ml::Device;

    #[test]
    fn a_long_prefill_is_the_only_shape_that_tiles() {
        // 4096 new tokens against a 124K prefix, 24 heads: the shape that wires 48 GB of scores.
        assert!(!score_block_fits(1, 24, 4096, 124_094, GENERAL));
        // The same context at decode width, and a speculative verify, stay on the single block.
        assert!(score_block_fits(1, 24, 1, 124_094, GENERAL));
        assert!(score_block_fits(1, 24, 8, 124_094, GENERAL));
        // A square 4K prefill fits, so its arithmetic is untouched.
        assert!(score_block_fits(1, 24, 4096, 4096, GENERAL));
    }

    /// Every device but ROCm keeps the general ceiling, so nothing that works today changes path.
    #[test]
    fn only_rocm_gets_the_lower_ceiling() {
        assert_eq!(score_block_budget(&Device::Cpu), GENERAL);
        assert!(ROCM < GENERAL);
    }

    /// The chunks that took a node down, under the ceiling that would have stopped them.
    ///
    /// A 1024-query chunk against a growing context, 24 heads. Under the general ceiling every one
    /// of these is a single block -- a DIFFERENT size each chunk, which is what made the retained
    /// memory quadratic -- right up to 21K of context. Under the ROCm ceiling they tile from ~2.7K.
    #[test]
    fn a_chunked_prefill_tiles_long_before_it_can_cost_the_node() {
        for kv_len in [4096, 8192, 16_384] {
            assert!(
                score_block_fits(1, 24, ATTENTION_CHUNK_SIZE, kv_len, GENERAL),
                "{kv_len}: the general ceiling still materializes this whole"
            );
            assert!(
                !score_block_fits(1, 24, ATTENTION_CHUNK_SIZE, kv_len, ROCM),
                "{kv_len}: the ROCm ceiling must tile this"
            );
        }
        // Short contexts and every decode step stay on the single block: one query against 262K of
        // context is 25 MiB, and tiling that would only add launches.
        assert!(score_block_fits(1, 24, ATTENTION_CHUNK_SIZE, 2048, ROCM));
        assert!(score_block_fits(1, 24, 1, 262_144, ROCM));
    }

    /// A tile must honour the ceiling that sent the block to the tiled path, or the ceiling bounds
    /// nothing: [1024, 4096] at 24 heads is 393 MiB, over the 256 MiB it was tiled to respect.
    #[test]
    fn a_tile_fits_the_budget_that_demanded_it() {
        for (heads, budget) in [
            (24, ROCM),
            (8, ROCM),
            (64, ROCM),
            (24, GENERAL),
            (128, GENERAL),
        ] {
            let tile = kv_tile_within(1, heads, ATTENTION_CHUNK_SIZE, budget);
            assert!(
                tile.is_power_of_two(),
                "{heads} heads: {tile} is not a power of two"
            );
            assert!((MIN_KV_TILE..=ATTENTION_KV_CHUNK_SIZE).contains(&tile));
            assert!(
                score_block_fits(1, heads, ATTENTION_CHUNK_SIZE, tile, budget)
                    || tile == MIN_KV_TILE,
                "{heads} heads under {budget}: a {tile}-key tile does not fit"
            );
        }
        // The measured model: 24 heads under the ROCm ceiling is a 2048-key tile, 196 MiB.
        assert_eq!(kv_tile_within(1, 24, ATTENTION_CHUNK_SIZE, ROCM), 2048);
        // The general ceiling is generous enough that the tile is the one already in use, so no
        // other device's tiled arithmetic moves.
        assert_eq!(
            kv_tile_within(1, 24, ATTENTION_CHUNK_SIZE, GENERAL),
            ATTENTION_KV_CHUNK_SIZE
        );
    }

    /// The measured model on the measured device: 24 heads, 1024-query chunks, 256 MiB. Every
    /// context evo can serve is cut by ROWS -- the fused softmax, the exact arithmetic -- and the
    /// online softmax is left for contexts where a run would be too short to be worth launching.
    #[test]
    fn rocm_cuts_the_query_axis_before_it_tiles_both() {
        let by = |kv_len| blocking(ROCM, true, 1, 24, ATTENTION_CHUNK_SIZE, kv_len);
        // Under the ceiling nothing is cut at all.
        assert_eq!(by(2048), Blocking::Whole);
        // 256 MiB / (24 * kv_len * 4 B), rounded down to a power of two.
        assert_eq!(by(4096), Blocking::Rows(512));
        assert_eq!(by(8192), Blocking::Rows(256));
        assert_eq!(by(16_384), Blocking::Rows(128));
        assert_eq!(by(65_536), Blocking::Rows(32));
        assert_eq!(by(131_072), Blocking::Rows(16));
        // Past that a run is under MIN_Q_ROWS, so the key axis is cut instead.
        assert_eq!(
            by(262_144),
            Blocking::Tiles {
                q: ATTENTION_CHUNK_SIZE,
                kv: 2048
            }
        );
    }

    /// Whatever `Rows` picks must itself fit, or it moved the problem instead of bounding it.
    #[test]
    fn a_run_of_rows_fits_the_budget_that_demanded_it() {
        for heads in [8, 24, 64] {
            for kv_len in [4096, 8192, 16_384, 40_000, 65_536, 131_072] {
                if let Blocking::Rows(rows) =
                    blocking(ROCM, true, 1, heads, ATTENTION_CHUNK_SIZE, kv_len)
                {
                    assert!(
                        rows.is_power_of_two() && rows >= MIN_Q_ROWS,
                        "{heads}h {kv_len}: {rows}"
                    );
                    assert!(
                        score_block_fits(1, heads, rows, kv_len, ROCM),
                        "{heads} heads, {kv_len} keys: a {rows}-row run does not fit"
                    );
                }
            }
        }
    }

    /// No other device is cut by rows, so the arithmetic of every shape that works there today is
    /// exactly what it was: whole under the general ceiling, the same 4096-key tiles over it.
    #[test]
    fn only_rocm_is_cut_by_rows() {
        for kv_len in [4096, 16_384, 124_094, 262_144] {
            let got = blocking(GENERAL, false, 1, 24, 4096, kv_len);
            assert!(!matches!(got, Blocking::Rows(_)), "{kv_len}: {got:?}");
        }
        assert_eq!(
            blocking(GENERAL, false, 1, 24, 4096, 124_094),
            Blocking::Tiles {
                q: ATTENTION_CHUNK_SIZE,
                kv: ATTENTION_KV_CHUNK_SIZE
            }
        );
        assert_eq!(blocking(GENERAL, false, 1, 24, 4096, 4096), Blocking::Whole);
    }

    /// A budget too small to be meant still yields a usable tile rather than zero or a panic.
    #[test]
    fn a_degenerate_budget_still_yields_a_tile() {
        assert_eq!(kv_tile_within(1, 24, ATTENTION_CHUNK_SIZE, 0), MIN_KV_TILE);
        assert_eq!(
            kv_tile_within(usize::MAX, usize::MAX, usize::MAX, 1),
            MIN_KV_TILE
        );
    }
}

#[cfg(test)]
mod rows_are_the_same_arithmetic {
    use super::{naive_sdpa, naive_sdpa_rows, SdpaParams};
    use hanzo_ml::{DType, Device, Result, Tensor};

    /// Deterministic, well-spread values; the CPU device's seed is a no-op in this stack.
    fn filled(shape: (usize, usize, usize, usize), salt: f32) -> Result<Tensor> {
        let n = shape.0 * shape.1 * shape.2 * shape.3;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.618_034 + salt).sin() * 1.7) + ((i % 7) as f32 - 3.0) * 0.11)
            .collect();
        Tensor::from_vec(data, shape, &Device::Cpu)
    }

    fn worst(a: &Tensor, b: &Tensor) -> Result<f32> {
        a.sub(b)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()
    }

    fn params() -> SdpaParams {
        SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 0.125,
            sliding_window: None,
            sinks: None,
        }
    }

    /// A causal prefill chunk against a longer context, which is the shape that gets cut: `q_len`
    /// new queries whose row i may see keys 0..=prefix+i.
    fn causal(q_len: usize, kv_len: usize, rank: usize) -> Result<Tensor> {
        let prefix = kv_len - q_len;
        let data: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j <= prefix + i {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        let m = Tensor::from_vec(data, (q_len, kv_len), &Device::Cpu)?;
        match rank {
            2 => Ok(m),
            3 => m.unsqueeze(0),
            _ => m.unsqueeze(0)?.unsqueeze(0),
        }
    }

    /// Runs that do not divide the query length, down to one row at a time, against the whole block.
    /// Each row's softmax is its own, so this is equality, not closeness: the bound is what a GEMM
    /// may differ by when only its row count changes, not a tolerance on the method.
    #[test]
    fn any_run_length_gives_the_whole_block() -> Result<()> {
        let (heads, q_len, kv_len, d) = (3, 37, 101, 16);
        let q = filled((1, heads, q_len, d), 0.1)?;
        let k = filled((1, heads, kv_len, d), 1.3)?;
        let v = filled((1, heads, kv_len, d), 2.9)?;
        for rank in [2, 3, 4] {
            let mask = causal(q_len, kv_len, rank)?;
            let whole = naive_sdpa(&q, &k, &v, Some(&mask), &params())?;
            for rows in [1, 2, 7, 16, 36, 37, 64] {
                let cut = naive_sdpa_rows(&q, &k, &v, Some(&mask), &params(), rows)?;
                assert_eq!(cut.dims(), whole.dims());
                let diff = worst(&cut, &whole)?;
                assert!(
                    diff <= 1e-6,
                    "rank-{rank} mask, runs of {rows}: off by {diff}"
                );
            }
        }
        Ok(())
    }

    /// A mask that broadcasts over the query axis has one row for every run to take. Narrowing it
    /// past row 0 is an error, which the fixed 1024-row chunk never reached and a finer cut does.
    #[test]
    fn a_mask_broadcast_over_queries_survives_the_cut() -> Result<()> {
        let (heads, q_len, kv_len, d) = (2, 19, 23, 8);
        let q = filled((1, heads, q_len, d), 0.4)?;
        let k = filled((1, heads, kv_len, d), 1.1)?;
        let v = filled((1, heads, kv_len, d), 2.2)?;
        // Hide the last five keys from every query alike: shape (1, 1, 1, kv_len).
        let data: Vec<f32> = (0..kv_len)
            .map(|j| {
                if j + 5 < kv_len {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let mask = Tensor::from_vec(data, (1, 1, 1, kv_len), &Device::Cpu)?;
        let whole = naive_sdpa(&q, &k, &v, Some(&mask), &params())?;
        let cut = naive_sdpa_rows(&q, &k, &v, Some(&mask), &params(), 4)?;
        let diff = worst(&cut, &whole)?;
        assert!(diff <= 1e-6, "off by {diff}");
        Ok(())
    }

    /// No mask at all, and a run longer than the query: both are the untouched single call.
    #[test]
    fn unmasked_and_oversized_runs_are_unchanged() -> Result<()> {
        let q = filled((1, 2, 9, 8), 0.7)?;
        let k = filled((1, 2, 13, 8), 1.9)?;
        let v = filled((1, 2, 13, 8), 3.1)?;
        let whole = naive_sdpa(&q, &k, &v, None, &params())?;
        for rows in [0, 1, 3, 9, 4096] {
            let diff = worst(&naive_sdpa_rows(&q, &k, &v, None, &params(), rows)?, &whole)?;
            assert!(diff <= 1e-6, "runs of {rows}: off by {diff}");
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "flash-attn"))]
mod flash_precision_probe {
    use super::{naive_sdpa, repeat_kv, SdpaParams};
    use crate::attention::backends::flash_attn;
    use hanzo_ml::{DType, Device, Tensor};

    fn causal_mask(s: usize, dev: &Device) -> Tensor {
        let mut m = vec![0f32; s * s];
        for i in 0..s {
            for j in (i + 1)..s {
                m[i * s + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(m, (s, s), dev).unwrap()
    }

    // Flash vs eager (exact f32-softmax reference) across seq lengths. If max_rel is FLAT with seq -> a
    // SYSTEMATIC error (real bug). If it SCALES with seq -> reduction-reorder noise (model sensitivity).
    #[test]
    fn flash_vs_eager_seq_sweep() {
        let dev = Device::new_cuda(0).expect("cuda");
        let (hq, hkv, d) = (32usize, 8usize, 128usize);
        let scale = 1.0 / (d as f32).sqrt();
        let p = SdpaParams {
            n_kv_groups: hq / hkv,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
        };
        for &s in &[8usize, 32, 128, 512] {
            let q = Tensor::randn(0f32, 1., (1, hq, s, d), &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let k = Tensor::randn(0f32, 1., (1, hkv, s, d), &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let v = Tensor::randn(0f32, 1., (1, hkv, s, d), &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let kr = repeat_kv(k.clone(), p.n_kv_groups).unwrap();
            let vr = repeat_kv(v.clone(), p.n_kv_groups).unwrap();
            let eager = naive_sdpa(
                &q,
                &kr,
                &vr,
                Some(&causal_mask(s, &dev).to_dtype(DType::BF16).unwrap()),
                &p,
            )
            .unwrap();
            let qf = q.transpose(1, 2).unwrap().contiguous().unwrap();
            let kf = kr.transpose(1, 2).unwrap().contiguous().unwrap();
            let vf = vr.transpose(1, 2).unwrap().contiguous().unwrap();
            let flash = flash_attn(&qf, &kf, &vf, None, &p)
                .unwrap()
                .transpose(1, 2)
                .unwrap();
            let e: Vec<f32> = eager
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let f: Vec<f32> = flash
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let (mut mr, mut ma) = (0f32, 0f32);
            for (a, b) in e.iter().zip(f.iter()) {
                let dd = (a - b).abs();
                ma = ma.max(dd);
                mr = mr.max(dd / a.abs().max(1e-3));
            }
            eprintln!(
                "[flash-vs-eager] seq={:<4} max_abs={:.4} max_rel={:.4}",
                s, ma, mr
            );
        }
    }
}
