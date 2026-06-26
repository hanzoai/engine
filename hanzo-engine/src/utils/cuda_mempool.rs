#[cfg(feature = "cuda")]
use hanzo_ml::{Device, Result};

#[cfg(feature = "cuda")]
const RELEASE_THRESHOLD_BYTES: u64 = u64::MAX;

/// Raise the default CUDA mempool release threshold so freed blocks stay resident instead of being
/// returned to the OS; on unified-memory parts (GB10) re-backing a large block costs hundreds of ms,
/// a fixed per-request floor since decode frees and reallocates scratch each step.
#[cfg(feature = "cuda")]
pub(crate) fn set_pool_retain_all(device: &Device) -> Result<()> {
    use hanzo_ml::cuda_backend::cudarc::driver::sys as cuda_sys;

    if std::env::var("HANZO_NO_MEMPOOL_FIX").is_ok() {
        return Ok(());
    }

    let Device::Cuda(dev) = device else {
        return Ok(());
    };

    let cu_device = dev.cuda_stream().context().cu_device();
    let mut threshold = RELEASE_THRESHOLD_BYTES;

    unsafe {
        let mut pool: cuda_sys::CUmemoryPool = std::ptr::null_mut();
        let res = cuda_sys::cuDeviceGetDefaultMemPool(&mut pool, cu_device);
        if res != cuda_sys::CUresult::CUDA_SUCCESS {
            tracing::warn!("cuDeviceGetDefaultMemPool failed ({res:?}); leaving pool at default");
            return Ok(());
        }
        let res = cuda_sys::cuMemPoolSetAttribute(
            pool,
            cuda_sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut threshold as *mut u64).cast(),
        );
        if res != cuda_sys::CUresult::CUDA_SUCCESS {
            tracing::warn!("cuMemPoolSetAttribute(RELEASE_THRESHOLD) failed ({res:?})");
        }
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn set_pool_retain_all(_device: &hanzo_ml::Device) -> hanzo_ml::Result<()> {
    Ok(())
}
