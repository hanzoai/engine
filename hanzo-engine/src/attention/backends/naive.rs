#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::MemoryUsage;

use hanzo_ml::{Device, Result, Tensor};
use hanzo_quant::MatMul;

use crate::attention::{chunked_attention, SdpaParams};
use std::sync::atomic::{AtomicU8, Ordering};

// 0 = unknown, 1 = unified, 2 = discrete; caches is_integrated_gpu so the per-layer guard stays cheap.
static UNIFIED_MEM: AtomicU8 = AtomicU8::new(0);

fn is_unified_memory_device(device: &Device) -> bool {
    match UNIFIED_MEM.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let unified = crate::utils::normal::is_integrated_gpu(device);
            UNIFIED_MEM.store(if unified { 1 } else { 2 }, Ordering::Relaxed);
            unified
        }
    }
}

/// Low-VRAM synchronize guard (CUDA OOM workaround). `MemoryUsage::query` runs a ~110ms sysinfo
/// scan; called per attention layer it left the GPU idle ~94% of prefill on a unified-memory GPU --
/// and the guard is meaningless there (no separate VRAM pool). Skip off-CUDA and on unified CUDA.
pub(crate) fn maybe_synchronize(device: &Device) -> Result<()> {
    if !device.is_cuda() || is_unified_memory_device(device) {
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
