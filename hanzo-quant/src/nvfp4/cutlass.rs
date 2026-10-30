//! NVFP4 GEMM on Blackwell block-scaled tensor cores.
//!
//! The MMA takes both operands in E2M1, so activations are quantized against
//! the checkpoint's calibrated `input_scale` and the packed weights are fed in
//! untouched. Weight-side setup (scale swizzle, per-column global scale) is
//! built once per layer; only the activation quantization is per call.

use std::ffi::{c_void, CStr};
use std::sync::{Mutex, OnceLock};

use float8::F8E4M3;
use half::{bf16, f16};
use hanzo_ml::cuda::cudarc::driver::CudaSlice;
use hanzo_ml::{CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor};

use crate::utils::slice_ptr;

/// The block-scaled collective reads K in groups this wide.
const K_GRANULE: usize = 64;
/// Column alignment the collective requires of N.
const N_GRANULE: usize = 32;
/// The Blackwell consumer part that carries the block-scaled MMA.
const SM_MAJOR: i32 = 12;
const SM_MINOR: i32 = 1;

const DTYPE_BF16: i32 = 0;
const DTYPE_F16: i32 = 1;
const KERNEL_PREFILL: i32 = 0;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct Context {
    device: i32,
    sm_count: i32,
    dtype: i32,
    kernel: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ShapeMnk {
    m: i32,
    n: i32,
    k: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Launch {
    shape: ShapeMnk,
    context: Context,
    a_packed: *const c_void,
    w_packed: *const c_void,
    a_scale_swizzled: *const c_void,
    w_scale_swizzled: *const c_void,
    weight_global: *const f32,
    activation_global: *const f32,
    output: *mut c_void,
    workspace: *mut c_void,
    workspace_bytes: usize,
    stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Resources {
    context: Context,
    major: i32,
    minor: i32,
    threads: i32,
    registers_per_thread: i32,
    shared_bytes: usize,
    local_bytes: usize,
}

extern "C" {
    fn hanzo_nvfp4_error_string(status: i32) -> *const std::ffi::c_char;
    fn hanzo_nvfp4_prepare(device: i32, dtype: i32, kernel: i32, resources: *mut Resources) -> i32;
    fn hanzo_nvfp4_workspace_size(
        context: *const Context,
        shape: *const ShapeMnk,
        bytes: *mut usize,
    ) -> i32;
    fn hanzo_nvfp4_gemm(launch: *const Launch) -> i32;
    fn hanzo_nvfp4_swizzle_cuda(
        source: *const c_void,
        dest: *mut c_void,
        rows: i32,
        k: i32,
        dest_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    fn nvfp4_quantize_scale_bytes(rows: i32, k: i32) -> usize;
    fn nvfp4_quantize_activations_f16(
        input: *const f16,
        packed: *mut u8,
        scale_swizzled: *mut u8,
        m: i32,
        k: i32,
        global: f32,
        stream: *mut c_void,
    );
    fn nvfp4_quantize_activations_bf16(
        input: *const bf16,
        packed: *mut u8,
        scale_swizzled: *mut u8,
        m: i32,
        k: i32,
        global: f32,
        stream: *mut c_void,
    );
}

fn check(status: i32, what: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(hanzo_nvfp4_error_string(status)) };
    hanzo_ml::bail!(
        "NVFP4 block-scaled {what} failed: {}",
        message.to_string_lossy()
    )
}

fn dtype_code(dtype: DType) -> Result<i32> {
    match dtype {
        DType::BF16 => Ok(DTYPE_BF16),
        DType::F16 => Ok(DTYPE_F16),
        other => hanzo_ml::bail!("NVFP4 block-scaled GEMM has no {other:?} path"),
    }
}

fn compute_cap(dev: &CudaDevice) -> Result<(i32, i32)> {
    use hanzo_ml::cuda::cudarc::driver::sys::CUdevice_attribute as Attr;
    let stream = dev.cuda_stream();
    let ctx = stream.context();
    let major = ctx
        .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(hanzo_ml::Error::wrap)?;
    let minor = ctx
        .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(hanzo_ml::Error::wrap)?;
    Ok((major, minor))
}

/// Whether this device carries the block-scaled MMA.
pub fn device_supported(dev: &CudaDevice) -> bool {
    matches!(compute_cap(dev), Ok((SM_MAJOR, SM_MINOR)))
}

/// Whether a GEMM of this shape meets the collective's alignment.
pub fn shape_supported(n: usize, k: usize) -> bool {
    n % N_GRANULE == 0 && k % K_GRANULE == 0
}

/// Prepared once per (device, dtype): sets the kernel's shared-memory attribute
/// and reports the SM count the scheduler needs.
fn context_for(dev: &CudaDevice, dtype: DType) -> Result<Context> {
    static PREPARED: OnceLock<Mutex<Vec<(i32, i32, Context)>>> = OnceLock::new();
    let cache = PREPARED.get_or_init(|| Mutex::new(Vec::new()));
    let device = dev.cuda_stream().context().ordinal() as i32;
    let code = dtype_code(dtype)?;

    let mut guard = cache.lock().expect("NVFP4 context cache poisoned");
    if let Some((_, _, context)) = guard.iter().find(|(d, t, _)| *d == device && *t == code) {
        return Ok(*context);
    }
    let mut resources = Resources::default();
    check(
        unsafe { hanzo_nvfp4_prepare(device, code, KERNEL_PREFILL, &mut resources) },
        "prepare",
    )?;
    guard.push((device, code, resources.context));
    Ok(resources.context)
}

/// Weight-side state the GEMM needs, built once when the layer loads.
#[derive(Debug)]
pub struct Weights {
    scale_swizzled: CudaSlice<u8>,
    /// One FP32 scale per output column, which is what the epilogue broadcasts.
    weight_global: Tensor,
    activation_global: Tensor,
    activation_scale: f32,
    n: usize,
    k: usize,
    dtype: DType,
}

impl Weights {
    /// `scale` is the checkpoint's E4M3 block scale [N, K/16], `weight_global`
    /// its `weight_scale_2`, `activation_scale` its calibrated `input_scale`.
    pub fn new(
        scale: &Tensor,
        weight_global: f32,
        activation_scale: f32,
        n: usize,
        k: usize,
        dtype: DType,
    ) -> Result<Self> {
        let Device::Cuda(dev) = scale.device().clone() else {
            hanzo_ml::bail!("NVFP4 block-scaled weights need a CUDA tensor");
        };
        let device = scale.device().clone();
        let stream = dev.cuda_stream();

        let bytes = unsafe { nvfp4_quantize_scale_bytes(n as i32, k as i32) };
        let swizzled = dev.alloc_zeros::<u8>(bytes)?;

        let scale = scale.contiguous()?;
        let scale_storage = scale.storage_and_layout().0;
        let scale_slice = match &*scale_storage {
            Storage::Cuda(s) => s.as_cuda_slice::<F8E4M3>()?,
            _ => hanzo_ml::bail!("Expected CUDA storage for NVFP4 block scales"),
        };
        {
            let (src, _src_guard) = slice_ptr(scale_slice, scale.layout().start_offset());
            let (dst, _dst_guard) = slice_ptr(&swizzled, 0);
            check(
                unsafe {
                    hanzo_nvfp4_swizzle_cuda(
                        src as *const c_void,
                        dst as *mut c_void,
                        n as i32,
                        k as i32,
                        bytes,
                        stream.cu_stream() as *mut c_void,
                    )
                },
                "weight scale swizzle",
            )?;
        }

        Ok(Self {
            scale_swizzled: swizzled,
            weight_global: Tensor::from_vec(vec![weight_global; n], n, &device)?,
            activation_global: Tensor::from_vec(vec![activation_scale], 1, &device)?,
            activation_scale,
            n,
            k,
            dtype,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// `output = quantize(input) @ weight.T`, both operands E2M1 on tensor cores.
pub fn matmul(input: &Tensor, weight: &Tensor, weights: &Weights) -> Result<Tensor> {
    let Device::Cuda(dev) = input.device().clone() else {
        hanzo_ml::bail!("NVFP4 block-scaled GEMM needs a CUDA tensor");
    };
    if input.dtype() != weights.dtype {
        hanzo_ml::bail!(
            "NVFP4 block-scaled GEMM got {:?} activations for {:?} weights",
            input.dtype(),
            weights.dtype
        );
    }
    let m = input.dim(0)?;
    let k = input.dim(1)?;
    if k != weights.k {
        hanzo_ml::bail!("NVFP4 block-scaled GEMM K mismatch: {k} vs {}", weights.k);
    }
    let n = weights.n;

    let context = context_for(&dev, weights.dtype)?;
    let cu_stream = dev.cuda_stream().cu_stream() as *mut c_void;

    let a_packed = dev.alloc_zeros::<u8>(m * k / 2)?;
    let a_scale_bytes = unsafe { nvfp4_quantize_scale_bytes(m as i32, k as i32) };
    let a_scale = dev.alloc_zeros::<u8>(a_scale_bytes)?;

    let input = input.contiguous()?;
    let input_storage = input.storage_and_layout().0;
    let (a_ptr, _a_guard) = slice_ptr(&a_packed, 0);
    let (as_ptr, _as_guard) = slice_ptr(&a_scale, 0);

    match weights.dtype {
        DType::BF16 => {
            let s = match &*input_storage {
                Storage::Cuda(s) => s.as_cuda_slice::<bf16>()?,
                _ => hanzo_ml::bail!("Expected CUDA storage for NVFP4 activations"),
            };
            let (x_ptr, _x_guard) = slice_ptr(s, input.layout().start_offset());
            unsafe {
                nvfp4_quantize_activations_bf16(
                    x_ptr as *const bf16,
                    a_ptr as *mut u8,
                    as_ptr as *mut u8,
                    m as i32,
                    k as i32,
                    weights.activation_scale,
                    cu_stream,
                )
            };
        }
        _ => {
            let s = match &*input_storage {
                Storage::Cuda(s) => s.as_cuda_slice::<f16>()?,
                _ => hanzo_ml::bail!("Expected CUDA storage for NVFP4 activations"),
            };
            let (x_ptr, _x_guard) = slice_ptr(s, input.layout().start_offset());
            unsafe {
                nvfp4_quantize_activations_f16(
                    x_ptr as *const f16,
                    a_ptr as *mut u8,
                    as_ptr as *mut u8,
                    m as i32,
                    k as i32,
                    weights.activation_scale,
                    cu_stream,
                )
            };
        }
    }

    let weight = weight.contiguous()?;
    let weight_storage = weight.storage_and_layout().0;
    let weight_slice = match &*weight_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<u8>()?,
        _ => hanzo_ml::bail!("Expected CUDA storage for NVFP4 weights"),
    };
    let wg_storage = weights.weight_global.storage_and_layout().0;
    let wg_slice = match &*wg_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<f32>()?,
        _ => hanzo_ml::bail!("Expected CUDA storage for NVFP4 global weight scale"),
    };
    let ag_storage = weights.activation_global.storage_and_layout().0;
    let ag_slice = match &*ag_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<f32>()?,
        _ => hanzo_ml::bail!("Expected CUDA storage for NVFP4 global activation scale"),
    };

    let (w_ptr, _w_guard) = slice_ptr(weight_slice, weight.layout().start_offset());
    let (ws_ptr, _ws_guard) = slice_ptr(&weights.scale_swizzled, 0);
    let (wg_ptr, _wg_guard) = slice_ptr(wg_slice, weights.weight_global.layout().start_offset());
    let (ag_ptr, _ag_guard) =
        slice_ptr(ag_slice, weights.activation_global.layout().start_offset());

    let shape = ShapeMnk {
        m: m as i32,
        n: n as i32,
        k: k as i32,
    };
    let mut workspace_bytes = 0usize;
    check(
        unsafe { hanzo_nvfp4_workspace_size(&context, &shape, &mut workspace_bytes) },
        "workspace size",
    )?;
    let workspace = dev.alloc_zeros::<u8>(workspace_bytes.max(1))?;
    let (work_ptr, _work_guard) = slice_ptr(&workspace, 0);

    macro_rules! run {
        ($ty:ty) => {{
            let output = dev.alloc_zeros::<$ty>(m * n)?;
            {
                let (o_ptr, _o_guard) = slice_ptr(&output, 0);
                let launch = Launch {
                    shape,
                    context,
                    a_packed: a_ptr as *const c_void,
                    w_packed: w_ptr as *const c_void,
                    a_scale_swizzled: as_ptr as *const c_void,
                    w_scale_swizzled: ws_ptr as *const c_void,
                    weight_global: wg_ptr as *const f32,
                    activation_global: ag_ptr as *const f32,
                    output: o_ptr as *mut c_void,
                    workspace: if workspace_bytes == 0 {
                        std::ptr::null_mut()
                    } else {
                        work_ptr as *mut c_void
                    },
                    workspace_bytes,
                    stream: cu_stream,
                };
                check(unsafe { hanzo_nvfp4_gemm(&launch) }, "gemm")?;
            }
            Ok(Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                Shape::from((m, n)),
            )))
        }};
    }

    match weights.dtype {
        DType::BF16 => run!(bf16),
        _ => run!(f16),
    }
}

/// The activation quantizer alone: codes `[M, K/2]` and the block scales read back out of the
/// swizzled layout into `[M, K/16]` order, for comparing against the linear-layout quantizer.
#[cfg(test)]
pub(crate) fn quantize_linear(input: &Tensor, input_scale: f32) -> Result<(Tensor, Vec<u8>)> {
    let Device::Cuda(dev) = input.device().clone() else {
        hanzo_ml::bail!("NVFP4 activation quantization needs a CUDA tensor");
    };
    let (m, k) = input.dims2()?;
    let packed = dev.alloc_zeros::<u8>(m * k / 2)?;
    let scale_bytes = unsafe { nvfp4_quantize_scale_bytes(m as i32, k as i32) };
    let scale = dev.alloc_zeros::<u8>(scale_bytes)?;
    {
        let input = input.contiguous()?;
        let storage = input.storage_and_layout().0;
        let (a_ptr, _a) = slice_ptr(&packed, 0);
        let (s_ptr, _s) = slice_ptr(&scale, 0);
        let stream = dev.cuda_stream().cu_stream() as *mut c_void;
        match input.dtype() {
            DType::BF16 => {
                let Storage::Cuda(s) = &*storage else {
                    hanzo_ml::bail!("expected CUDA storage")
                };
                let (x_ptr, _x) = slice_ptr(s.as_cuda_slice::<bf16>()?, input.layout().start_offset());
                unsafe {
                    nvfp4_quantize_activations_bf16(
                        x_ptr as *const bf16,
                        a_ptr as *mut u8,
                        s_ptr as *mut u8,
                        m as i32,
                        k as i32,
                        input_scale,
                        stream,
                    )
                };
            }
            d => hanzo_ml::bail!("quantize_linear takes bf16, got {d:?}"),
        }
    }
    let swizzled = Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(scale, dev.clone())),
        Shape::from(scale_bytes),
    ))
    .to_vec1::<u8>()?;
    let cols = k / 16;
    let padded_cols = cols.div_ceil(4) * 4;
    let mut linear = Vec::with_capacity(m * cols);
    for row in 0..m {
        for col in 0..cols {
            let off = (((row / 128) * (padded_cols / 4) + col / 4) * 32 + row % 32) * 16
                + (row % 128) / 32 * 4
                + col % 4;
            linear.push(swizzled[off]);
        }
    }
    let codes = Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(packed, dev.clone())),
        Shape::from((m, k / 2)),
    ));
    Ok((codes, linear))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvfp4::ops::{nvfp4_dequantize, FP4_E2M1_LUT};

    const GLOBAL: f32 = 0.75;
    /// The E2M1 codebook's largest magnitude; a block holding it pins the block scale.
    const E2M1_MAX: f32 = 6.0;

    fn packed_weights(n: usize, k: usize, dev: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        let packed: Vec<u8> = (0..n * k / 2)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect();
        let scales: Vec<f32> = (0..n * (k / 16))
            .map(|i| [0.5f32, 1.0, 2.0, 0.25][i % 4])
            .collect();
        let q_cpu = Tensor::from_vec(packed, (n, k / 2), &Device::Cpu)?;
        let s_cpu = Tensor::from_vec(scales, (n, k / 16), &Device::Cpu)?.to_dtype(DType::F8E4M3)?;
        let s2 = Tensor::new(GLOBAL, &Device::Cpu)?;
        let dense = nvfp4_dequantize(&q_cpu, &s_cpu, Some(&s2), DType::F32)?
            .t()?
            .contiguous()?;
        Ok((q_cpu.to_device(dev)?, s_cpu.to_device(dev)?, dense))
    }

    /// Activations drawn from the E2M1 codebook with a 6.0 leading each block:
    /// the block scale lands on 1.0 exactly, so quantization loses nothing and
    /// any disagreement is the kernel, the scale swizzle or the epilogue.
    fn lossless_activations(m: usize, k: usize) -> Vec<f32> {
        (0..m * k)
            .map(|i| {
                if i % 16 == 0 {
                    E2M1_MAX
                } else {
                    let code = (i * 7 + 3) % 16;
                    FP4_E2M1_LUT[code]
                }
            })
            .collect()
    }

    #[test]
    fn blockscaled_matches_dequantized_on_exact_activations() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        if !device_supported(cuda) {
            return Ok(());
        }
        for (n, k) in [(64usize, 128usize), (512, 5120), (256, 17408)] {
            let (q, s, dense) = packed_weights(n, k, &dev)?;
            let weights = Weights::new(&s, GLOBAL, 1.0, n, k, DType::BF16)?;
            for m in [17usize, 64, 512] {
                let x_cpu = Tensor::from_vec(lossless_activations(m, k), (m, k), &Device::Cpu)?;
                let want: Vec<f32> = x_cpu.matmul(&dense)?.flatten_all()?.to_vec1()?;
                let x = x_cpu.to_device(&dev)?.to_dtype(DType::BF16)?;
                let got: Vec<f32> = matmul(&x, &q, &weights)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1()?;
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= 0.01 * w.abs().max(1.0),
                        "n={n} k={k} m={m} i={i}: got {g}, want {w}"
                    );
                }
            }
        }
        Ok(())
    }

    /// With activations it cannot represent exactly, the path trades accuracy for
    /// rate. This pins how much: the checkpoint's own calibration is what makes
    /// the trade sound, and a regression here means the plumbing drifted.
    #[test]
    fn blockscaled_holds_direction_on_general_activations() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        if !device_supported(cuda) {
            return Ok(());
        }
        let (n, k, m) = (512usize, 5120usize, 128usize);
        let (q, s, dense) = packed_weights(n, k, &dev)?;
        let weights = Weights::new(&s, GLOBAL, 1.0, n, k, DType::BF16)?;

        let mut state = 12345u32;
        let x: Vec<f32> = (0..m * k)
            .map(|_| {
                let mut acc = 0.0f32;
                for _ in 0..4 {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    acc += (state >> 8) as f32 / 8388608.0 - 1.0;
                }
                acc * 0.5
            })
            .collect();
        let x_cpu = Tensor::from_vec(x, (m, k), &Device::Cpu)?;
        let want: Vec<f32> = x_cpu.matmul(&dense)?.flatten_all()?.to_vec1()?;
        let got: Vec<f32> = matmul(&x_cpu.to_device(&dev)?.to_dtype(DType::BF16)?, &q, &weights)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        let (mut dot, mut ng, mut nw) = (0.0f64, 0.0f64, 0.0f64);
        for (g, w) in got.iter().zip(&want) {
            dot += (*g as f64) * (*w as f64);
            ng += (*g as f64) * (*g as f64);
            nw += (*w as f64) * (*w as f64);
        }
        let cosine = dot / (ng.sqrt() * nw.sqrt());
        assert!(
            cosine > 0.99,
            "cosine against bf16 activations was {cosine}"
        );
        Ok(())
    }
}
