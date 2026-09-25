//! C ABI of `kernels/moe` and the device-pointer plumbing the launchers share.

use std::ffi::{c_char, c_int, c_void, CStr};

use float8::F8E4M3;
use half::bf16;
use hanzo_ml::cuda::cudarc::driver::{CudaSlice, DeviceRepr};
use hanzo_ml::{CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor};

use crate::utils::slice_ptr;

pub(crate) type Stream = *mut c_void;

pub(crate) const FINALIZE: c_int = 0;
pub(crate) const SUM: c_int = 1;

extern "C" {
    pub(crate) fn hanzo_moe_error_string(status: c_int) -> *const c_char;
    pub(crate) fn hanzo_moe_prepare(device: c_int) -> c_int;
    pub(crate) fn hanzo_moe_route_scratch(r: c_int, e: c_int) -> usize;
    pub(crate) fn hanzo_moe_route(
        ids: *const i32,
        m: c_int,
        k: c_int,
        e: c_int,
        offsets: *mut i32,
        src: *mut i32,
        dst: *mut i32,
        group: *mut i32,
        active: *mut i32,
        nactive: *mut i32,
        scratch: *mut c_void,
        launches: *mut c_int,
        stream: Stream,
    ) -> c_int;
    pub(crate) fn hanzo_moe_combine(
        y: *const c_void,
        dst: *const i32,
        weights: *const f32,
        addend: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        h: c_int,
        rule: c_int,
        stream: Stream,
    ) -> c_int;
}

/// Turns a kernel status into a `Result`, naming the stage.
pub(crate) fn check(status: c_int, what: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let msg = unsafe { CStr::from_ptr(hanzo_moe_error_string(status)) };
    hanzo_ml::bail!(
        "moe {what} failed ({status}): {}",
        msg.to_string_lossy()
    )
}

pub(crate) fn cuda(dev: &Device) -> Result<CudaDevice> {
    match dev {
        Device::Cuda(d) => Ok(d.clone()),
        _ => hanzo_ml::bail!("routed experts run on CUDA only"),
    }
}

pub(crate) fn stream(dev: &CudaDevice) -> Stream {
    dev.cuda_stream().cu_stream() as Stream
}

/// Sets every MoE kernel's shared-memory attribute on this device, once.
pub(crate) fn prepare(dev: &CudaDevice) -> Result<()> {
    use std::sync::{Mutex, OnceLock};
    static DONE: OnceLock<Mutex<Vec<usize>>> = OnceLock::new();
    let ordinal = dev.cuda_stream().context().ordinal();
    let mut done = DONE.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap();
    if !done.contains(&ordinal) {
        check(unsafe { hanzo_moe_prepare(ordinal as c_int) }, "prepare")?;
        done.push(ordinal);
    }
    Ok(())
}

/// Device address of a contiguous CUDA tensor's first element.
///
/// Every launch here is ordered on the device's one stream, so the address is used on the
/// stream that owns the allocation and no cross-stream event is needed.
pub(crate) fn ptr(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    if !layout.is_contiguous() {
        hanzo_ml::bail!("moe expects contiguous tensors, got {:?}", layout);
    }
    let Storage::Cuda(s) = &*storage else {
        hanzo_ml::bail!("moe expects CUDA tensors")
    };
    let off = layout.start_offset();
    let p = match t.dtype() {
        DType::U8 => slice_ptr(s.as_cuda_slice::<u8>()?, off).0,
        DType::U32 => slice_ptr(s.as_cuda_slice::<u32>()?, off).0,
        DType::I32 => slice_ptr(s.as_cuda_slice::<i32>()?, off).0,
        DType::F32 => slice_ptr(s.as_cuda_slice::<f32>()?, off).0,
        DType::BF16 => slice_ptr(s.as_cuda_slice::<bf16>()?, off).0,
        DType::F8E4M3 => slice_ptr(s.as_cuda_slice::<F8E4M3>()?, off).0,
        d => hanzo_ml::bail!("moe has no {d:?} operand"),
    };
    Ok(p)
}

/// Device address of a slice at an element offset.
pub(crate) fn sptr<T: DeviceRepr>(s: &CudaSlice<T>, off: usize) -> u64 {
    slice_ptr(s, off).0
}

/// A tensor over a freshly allocated, uninitialized device buffer.
pub(crate) fn empty<T: hanzo_ml::cuda::CudaDType + hanzo_ml::WithDType + DeviceRepr>(
    dev: &CudaDevice,
    shape: impl Into<Shape>,
) -> Result<Tensor> {
    let shape = shape.into();
    let slice = unsafe { dev.alloc::<T>(shape.elem_count()) }?;
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
        shape,
    )))
}

/// Streaming multiprocessors on this device (the persistent GEMMs launch one CTA each).
pub(crate) fn sm_count(dev: &CudaDevice) -> Result<i32> {
    use hanzo_ml::cuda::cudarc::driver::sys::CUdevice_attribute as Attr;
    dev.cuda_stream()
        .context()
        .attribute(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(hanzo_ml::Error::wrap)
}
