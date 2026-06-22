#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::MemoryUsage;

use hanzo_ml::{Device, Result, Tensor};
use hanzo_quant::MatMul;

use crate::attention::{chunked_attention, SdpaParams};

/// Low-VRAM synchronize guard: a CUDA-specific OOM workaround. Off CUDA the `MemoryUsage::query`
/// it calls is expensive (a full system scan) and ran once per attention layer (~28x/token on a
/// small model) -- the dominant decode-time cost. The guard is meaningless off CUDA, so skip it.
/// On integrated/unified-memory CUDA GPUs (GB10) `MemoryUsage::query` also does a full `sysinfo`
/// scan (~110ms) AND there is no separate VRAM pool to exhaust, so skip the guard there too.
pub(crate) fn maybe_synchronize(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    if is_unified_memory_device(device) {
        return Ok(());
    }
    // If less that 4 GB available, synchronize
    #[cfg(target_pointer_width = "64")]
    const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
    #[cfg(not(target_pointer_width = "64"))]
    const FOUR_GIB: usize = usize::MAX;
    if MemoryUsage.query(device)?.available() < FOUR_GIB {
        device.synchronize()?;
    }
    Ok(())
}

/// Cached `is_integrated_gpu` check so the per-layer guard never re-runs the CUDA attribute query.
fn is_unified_memory_device(device: &Device) -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(u8::MAX);
    let cached = CACHED.load(Ordering::Relaxed);
    if cached != u8::MAX {
        return cached == 1;
    }
    let unified = crate::utils::normal::is_integrated_gpu(device);
    CACHED.store(u8::from(unified), Ordering::Relaxed);
    unified
}

/// Computes softmax(QK^T*sqrt(d_k))V
pub(crate) fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    maybe_synchronize(q.device())?;

    // Use chunked attention with a closure that captures the necessary parameters
    chunked_attention(q, k, v, mask, |q_chunk, k, v, mask_chunk| {
        let mut att =
            MatMul.matmul_affine_mul(q_chunk, &k.t()?, sdpa_params.softmax_scale.into())?;

        if let Some(softcap) = sdpa_params.softcap {
            att = (att / softcap as f64)?;
            att = att.tanh()?;
            att = (att * softcap as f64)?;
        }

        if let Some(mask) = mask_chunk {
            att = att.broadcast_add(mask)?;
        }

        // Compute softmax in F32 for precision (BF16 exp() loses information).
        let att_dtype = att.dtype();
        if att_dtype == hanzo_ml::DType::BF16 || att_dtype == hanzo_ml::DType::F16 {
            att = att.to_dtype(hanzo_ml::DType::F32)?;
        }
        att = hanzo_nn::ops::softmax_last_dim(&att)?;
        if att.dtype() != att_dtype {
            att = att.to_dtype(att_dtype)?;
        }
        MatMul.matmul(&att, v)
    })
}
