//! Launchers for `kernels/quantize`.

use float8::F8E4M3;
use half::{bf16, f16};
use hanzo_ml::{
    cuda::cudarc::driver::sys::CUstream, CudaStorage, DType, Device, Result, Shape, Storage,
    Tensor,
};

use super::{Fp8Mode, FP8_GROUP, NVFP4_BLOCK};
use crate::utils::slice_ptr;

extern "C" {
    fn hanzo_quantize_fp8_f32(
        x: *const f32,
        q: *mut u8,
        s: *mut f32,
        m: i32,
        k: i32,
        mode: i32,
        stream: CUstream,
    );
    fn hanzo_quantize_fp8_f16(
        x: *const f16,
        q: *mut u8,
        s: *mut f32,
        m: i32,
        k: i32,
        mode: i32,
        stream: CUstream,
    );
    fn hanzo_quantize_fp8_bf16(
        x: *const bf16,
        q: *mut u8,
        s: *mut f32,
        m: i32,
        k: i32,
        mode: i32,
        stream: CUstream,
    );
    fn hanzo_quantize_nvfp4_f16(
        x: *const f16,
        codes: *mut u8,
        scales: *mut u8,
        m: i32,
        k: i32,
        gs: f32,
        stream: CUstream,
    );
    fn hanzo_quantize_nvfp4_bf16(
        x: *const bf16,
        codes: *mut u8,
        scales: *mut u8,
        m: i32,
        k: i32,
        gs: f32,
        stream: CUstream,
    );
    fn hanzo_e2m1_encode(v: *const f32, codes: *mut u8, n: i32, stream: CUstream);
}

fn cuda_dev(x: &Tensor) -> Result<hanzo_ml::CudaDevice> {
    match x.device() {
        Device::Cuda(d) => Ok(d.clone()),
        _ => hanzo_ml::bail!("expected a CUDA tensor"),
    }
}

fn wrap<T: hanzo_ml::cuda::CudaDType + hanzo_ml::WithDType>(
    slice: hanzo_ml::cuda::cudarc::driver::CudaSlice<T>,
    dev: &hanzo_ml::CudaDevice,
    shape: impl Into<Shape>,
) -> Tensor {
    Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
        shape.into(),
    ))
}

pub(super) fn fp8(x: &Tensor, mode: Fp8Mode) -> Result<(Tensor, Tensor)> {
    let dev = cuda_dev(x)?;
    let (m, k) = x.dims2()?;
    let q = dev.alloc_zeros::<F8E4M3>(m * k)?;
    let s = dev.alloc_zeros::<f32>(m * (k / FP8_GROUP))?;
    {
        let (storage, layout) = x.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            hanzo_ml::bail!("expected CUDA storage")
        };
        let (q_ptr, _qg) = slice_ptr(&q, 0);
        let (s_ptr, _sg) = slice_ptr(&s, 0);
        let stream = dev.cuda_stream().cu_stream();
        macro_rules! go {
            ($t:ty, $f:ident) => {{
                let (x_ptr, _xg) = slice_ptr(storage.as_cuda_slice::<$t>()?, layout.start_offset());
                unsafe {
                    $f(
                        x_ptr as *const $t,
                        q_ptr as *mut u8,
                        s_ptr as *mut f32,
                        m as i32,
                        k as i32,
                        mode.code(),
                        stream,
                    )
                }
            }};
        }
        match x.dtype() {
            DType::F32 => go!(f32, hanzo_quantize_fp8_f32),
            DType::F16 => go!(f16, hanzo_quantize_fp8_f16),
            DType::BF16 => go!(bf16, hanzo_quantize_fp8_bf16),
            d => hanzo_ml::bail!("fp8 activation quantization of {d:?}"),
        }
    }
    Ok((wrap(q, &dev, (m, k)), wrap(s, &dev, (m, k / FP8_GROUP))))
}

pub(super) fn nvfp4(x: &Tensor, gs: f32) -> Result<(Tensor, Tensor)> {
    let dev = cuda_dev(x)?;
    let (m, k) = x.dims2()?;
    let codes = dev.alloc_zeros::<u8>(m * k / 2)?;
    let scales = dev.alloc_zeros::<F8E4M3>(m * (k / NVFP4_BLOCK))?;
    {
        let (storage, layout) = x.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            hanzo_ml::bail!("expected CUDA storage")
        };
        let (c_ptr, _cg) = slice_ptr(&codes, 0);
        let (s_ptr, _sg) = slice_ptr(&scales, 0);
        let stream = dev.cuda_stream().cu_stream();
        macro_rules! go {
            ($t:ty, $f:ident) => {{
                let (x_ptr, _xg) = slice_ptr(storage.as_cuda_slice::<$t>()?, layout.start_offset());
                unsafe {
                    $f(
                        x_ptr as *const $t,
                        c_ptr as *mut u8,
                        s_ptr as *mut u8,
                        m as i32,
                        k as i32,
                        gs,
                        stream,
                    )
                }
            }};
        }
        match x.dtype() {
            DType::F16 => go!(f16, hanzo_quantize_nvfp4_f16),
            DType::BF16 => go!(bf16, hanzo_quantize_nvfp4_bf16),
            d => hanzo_ml::bail!("nvfp4 activation quantization of {d:?}"),
        }
    }
    Ok((
        wrap(codes, &dev, (m, k / 2)),
        wrap(scales, &dev, (m, k / NVFP4_BLOCK)),
    ))
}

pub(super) fn e2m1(v: &Tensor) -> Result<Tensor> {
    let dev = cuda_dev(v)?;
    let n = v.elem_count();
    let codes = dev.alloc_zeros::<u8>(n)?;
    {
        let (storage, layout) = v.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            hanzo_ml::bail!("expected CUDA storage")
        };
        let (v_ptr, _vg) = slice_ptr(storage.as_cuda_slice::<f32>()?, layout.start_offset());
        let (c_ptr, _cg) = slice_ptr(&codes, 0);
        unsafe {
            hanzo_e2m1_encode(
                v_ptr as *const f32,
                c_ptr as *mut u8,
                n as i32,
                dev.cuda_stream().cu_stream(),
            )
        };
    }
    Ok(wrap(codes, &dev, n))
}
