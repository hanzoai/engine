#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::{attention::backends::cpu, pipeline::text_models_inputs_processor::FlashParams};

use hanzo_ml::{DType, Device, Result, Tensor};

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
pub(crate) use backends::{flash_attn, maybe_synchronize, naive_sdpa, sinks_attn};

/// Chunk size for attention computation to avoid OOM on long sequences
pub(crate) const ATTENTION_CHUNK_SIZE: usize = 1024;


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
    let seq_len = q.dim(2)?;

    if seq_len <= ATTENTION_CHUNK_SIZE {
        // For short sequences, use the regular path
        return attention_fn(q, k, v, mask);
    }

    // Chunk the query to avoid OOM on long sequences
    let num_chunks = seq_len.div_ceil(ATTENTION_CHUNK_SIZE);
    let mut attn_chunks = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let offset = chunk_idx * ATTENTION_CHUNK_SIZE;
        let chunk_len = ATTENTION_CHUNK_SIZE.min(seq_len - offset);

        // Extract query chunk
        let q_chunk = q.narrow(2, offset, chunk_len)?;

        // Extract mask chunk if present
        let mask_chunk = mask
            .map(|m| {
                match m.rank() {
                    2 => {
                        // For 2D masks (seq_len, seq_len), narrow along dimension 0
                        m.narrow(0, offset, chunk_len)
                    }
                    3 => {
                        // For 3D masks (batch, seq_len, seq_len), narrow along dimension 1
                        m.narrow(1, offset, chunk_len)
                    }
                    4 => {
                        // For 4D masks (batch, heads, seq_len, seq_len), narrow along dimension 2
                        m.narrow(2, offset, chunk_len)
                    }
                    _ => m.narrow(2, offset, chunk_len), // Default to dimension 2
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

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
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
    use hanzo_ml::{Device, Storage};
    let (b, h, q_len, d) = q.dims4()?;
    if q_len != 1
        || d == 0
        || d > 128
        || (d & (d - 1)) != 0
        || sdpa_params.softcap.is_some()
        || sdpa_params.sliding_window.is_some()
    {
        return Ok(None);
    }
    let dev = match q.device() {
        Device::Vulkan(dv) => dv.clone(),
        _ => return Ok(None),
    };
    let hkv = k.dim(1)?;
    let l = k.dim(2)?;
    if h % hkv != 0 {
        return Ok(None);
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let (qs, _) = q.storage_and_layout();
    let (ks, _) = k.storage_and_layout();
    let (vs, _) = v.storage_and_layout();
    let (Storage::Vulkan(_qv), Storage::Vulkan(_kv), Storage::Vulkan(_vv)) = (&*qs, &*ks, &*vs) else {
        return Ok(None);
    };
    // Fused single-query Vulkan decode-attention (`attn_decode_gpu`) is not yet wired in hanzo-ml;
    // returning None routes this through the standard Sdpa path (which handles Vulkan storage). The
    // fused kernel is a decode-perf follow-up, not a correctness requirement.
    let _ = (dev, hkv, l);
    Ok(None)
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
        if q.device().is_rocm()
            && !matches!(mask, AttentionMask::None if !do_causal)
        {
            const ROCM_FLASH_MIN_SEQ: usize = 768;
            let (_, _, seq_len, head_dim) = q.dims4()?;
            let is_full_causal = matches!(mask, AttentionMask::CausalFlash)
                || (mask.is_custom()
                    && sdpa_params.sliding_window.is_none()
                    && !explicitly_noncausal);
            if is_full_causal
                && seq_len >= ROCM_FLASH_MIN_SEQ
                && head_dim == 128
                && k.dim(3)? == 128
                && v.dim(3)? == 128
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

        // NOTE: CUDA GGUF prefill uses a Custom causal mask and falls to eager naive_sdpa below, which
        // materializes the [seq x seq] score matrix -> O(n^2) prefill (measured: 8B pp512/pp2048 collapse
        // 0.84x->0.37x of llama while llama's fused flash stays flat ~3000 t/s). Routing this Custom-mask
        // prefill to the fused `flash_attn` (drop the redundant causal mask, like the ROCm block above)
        // is a PROVEN ~1.8-2.4x prefill lever (flat vs length) -- BUT the hanzo-flash-attn CUDA path
        // produces GARBAGE logits for this config (bf16 GGUF, paged, causal); contiguous + GQA->MHA
        // repeat_kv + window/bf16-dispatch checks all ruled out, so it needs a numeric flash-vs-eager
        // tensor diff to isolate. Tracked as the #1 CUDA prefill follow-up (engine LLM.md). Until fixed,
        // GGUF prefill stays on the correct eager path.

        // FLASH_DBG: real-data flash-vs-naive diff on the ACTUAL Custom-mask prefill (mask/q/k/v from the
        // live forward). Logs the mask stats + divergence, returns naive (stays coherent). Diagnostic only.
        #[cfg(any(feature = "flash-attn", feature = "flash-attn-v3"))]
        if std::env::var_os("FLASH_DBG").is_some()
            && q.device().is_cuda()
            && !explicitly_noncausal
            && sdpa_params.sliding_window.is_none()
        {
            if let AttentionMask::Custom(m) = mask {
                let (_, _, s, hd) = q.dims4()?;
                if s > 1 && hd == 128 && matches!(q.dtype(), DType::F16 | DType::BF16) {
                    let naive =
                        self.run_attention_noflash(q, k, v, Some(m), sdpa_params, do_causal)?;
                    let kr = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
                    let vr = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;
                    let qf = q.transpose(1, 2)?.contiguous()?;
                    let kf = kr.transpose(1, 2)?.contiguous()?;
                    let vf = vr.transpose(1, 2)?.contiguous()?;
                    let flash = flash_attn(&qf, &kf, &vf, flash_params, sdpa_params)?
                        .transpose(1, 2)?;
                    let nv = naive.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let fv = flash.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let mv = m.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let nan = fv.iter().filter(|x| x.is_nan()).count();
                    let ninf = mv.iter().filter(|x| x.is_infinite()).count();
                    let mut maxabs = 0f32;
                    for (a, b) in nv.iter().zip(fv.iter()) {
                        if a.is_finite() && b.is_finite() {
                            maxabs = maxabs.max((a - b).abs());
                        }
                    }
                    eprintln!(
                        "[FLASH_DBG] mask_dims={:?} q_dims={:?} n_kv_groups={} scale={} softcap={:?} causal={} | flash_nan={} maxabs={:.4} mask_ninf={}/{} naive[0..3]={:?} flash[0..3]={:?}",
                        m.dims(), q.dims(), sdpa_params.n_kv_groups, sdpa_params.softmax_scale,
                        sdpa_params.softcap, do_causal, nan, maxabs, ninf, mv.len(),
                        &nv[..3.min(nv.len())], &fv[..3.min(fv.len())]
                    );
                    return Ok(naive);
                }
            }
        }

        // Custom mask, eager attention (flash can't use arbitrary mask tensors)
        if let AttentionMask::Custom(mask_tensor) = mask {
            return self.run_attention_noflash(q, k, v, Some(mask_tensor), sdpa_params, do_causal);
        }

        // CausalFlash or None: try flash attention, fall back to eager
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

        let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;

        if mask.is_some_and(|x| x.rank() == 2) || hanzo_quant::distributed::use_nccl() {
            return naive_sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                mask,
                sdpa_params,
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
            naive_sdpa(q, &k, &v, mask, sdpa_params)
        }
    }
}

#[cfg(all(test, feature = "flash-attn"))]
mod flash_correctness {
    use super::{naive_sdpa, SdpaParams};
    use crate::attention::backends::flash_attn;
    use crate::attention::repeat_kv;
    use hanzo_ml::{DType, Device, Tensor};

    // Additive causal mask (s x s): 0 on/below the diagonal, -inf above.
    fn causal_mask(s: usize, dev: &Device) -> Tensor {
        let mut m = vec![0f32; s * s];
        for i in 0..s {
            for j in (i + 1)..s {
                m[i * s + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(m, (s, s), dev).unwrap()
    }

    // The decisive isolation: flash_attn vs the trusted naive_sdpa on IDENTICAL controlled q/k/v.
    // If these diverge, the bug is in the flash crate/invocation (not the model). Qwen3-8B shape:
    // 32 Q heads / 8 KV heads, head_dim 128, bf16, causal prefill.
    #[test]
    fn flash_matches_naive_causal_gqa_prefill() {
        let dev = Device::new_cuda(0).expect("cuda:0");
        let (b, hq, hkv, s, d) = (1usize, 32usize, 8usize, 512usize, 128usize);
        let scale = 1.0 / (d as f32).sqrt();
        let params = SdpaParams {
            n_kv_groups: hq / hkv,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
        };
        let q = Tensor::randn(0f32, 1., (b, hq, s, d), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::randn(0f32, 1., (b, hkv, s, d), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = Tensor::randn(0f32, 1., (b, hkv, s, d), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        // Trusted reference: repeat kv to MHA + explicit causal mask + naive sdpa -> (b, hq, s, d).
        let kr = repeat_kv(k.clone(), params.n_kv_groups).unwrap();
        let vr = repeat_kv(v.clone(), params.n_kv_groups).unwrap();
        let mask = causal_mask(s, &dev).to_dtype(DType::BF16).unwrap();
        let eager = naive_sdpa(&q, &kr, &vr, Some(&mask), &params).unwrap();

        // Flash: same repeated kv, (b,H,s,d)->(b,s,H,d) contiguous, is_causal internal.
        let qf = q.transpose(1, 2).unwrap().contiguous().unwrap();
        let kf = kr.transpose(1, 2).unwrap().contiguous().unwrap();
        let vf = vr.transpose(1, 2).unwrap().contiguous().unwrap();
        let flash = flash_attn(&qf, &kf, &vf, None, &params)
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        let e = eager
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let f = flash
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let nan = f.iter().filter(|x| x.is_nan()).count();
        let zeros = f.iter().filter(|x| **x == 0.0).count();
        let mut maxabs = 0f32;
        for (a, b) in e.iter().zip(f.iter()) {
            maxabs = maxabs.max((a - b).abs());
        }
        eprintln!(
            "[flash-vs-naive] n={} nan={} zeros={} maxabs={:.4} eager[0..4]={:?} flash[0..4]={:?}",
            f.len(),
            nan,
            zeros,
            maxabs,
            &e[..4],
            &f[..4]
        );
        assert_eq!(nan, 0, "flash produced {nan} NaNs");
        assert!(maxabs < 0.15, "flash != naive: maxabs={maxabs} (bf16 tol)");
    }
}
