//! Block-FP8 GEMM on sm_12x tensor cores, bit-identical to vLLM's.
//!
//! The kernels are vLLM v0.29.0's three CUTLASS sm120 blockwise tiles
//! (`kernels/blockwise_fp8_cutlass`), and [`Tile::for_rows`] is vLLM's M
//! dispatch. Weights are e4m3 `[N, K]` with F32 `weight_scale_inv`
//! `[N/128, K/128]`; activations are e4m3 `[M, K]` with per-token-group F32
//! scales `[M, K/128]`, row-major. Output is bf16 `[M, N]`.

use std::ffi::{c_void, CStr};
use std::sync::{Mutex, OnceLock};

use float8::F8E4M3;
use half::bf16;
use hanzo_ml::cuda::cudarc::driver::DeviceRepr;
use hanzo_ml::{CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor};

use crate::utils::slice_ptr;

/// Scale block edge: weights carry one scale per 128x128, activations per 1x128.
pub const BLOCK: usize = 128;
const OPERAND_ALIGN: u64 = 16;
const SCALE_ALIGN: u64 = 4;
const SM_MAJOR: i32 = 12;
const SM_MINOR: i32 = 1;

/// vLLM's three sm_12x tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile {
    /// 128x32x128 cooperative with A and B swapped: weights fill the MMA's M.
    SwapAb,
    /// 64x128x128 pingpong.
    Pingpong,
    /// 128x128x128 cooperative.
    Cooperative,
}

impl Tile {
    pub const ALL: [Tile; 3] = [Tile::SwapAb, Tile::Pingpong, Tile::Cooperative];

    /// vLLM v0.29.0 `cutlass_gemm_blockwise_sm120_fp8_dispatch`: M alone
    /// decides, with no alignment clause and no split-K.
    pub fn for_rows(m: usize) -> Tile {
        if m <= 64 {
            Tile::SwapAb
        } else if m <= 256 {
            Tile::Pingpong
        } else {
            Tile::Cooperative
        }
    }

    fn code(self) -> i32 {
        match self {
            Tile::SwapAb => 0,
            Tile::Pingpong => 1,
            Tile::Cooperative => 2,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Context {
    device: i32,
    sm_count: i32,
    tile: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct ShapeMnk {
    m: i32,
    n: i32,
    k: i32,
}

#[repr(C)]
struct Launch {
    shape: ShapeMnk,
    context: Context,
    a: *const c_void,
    a_scale: *const f32,
    w: *const c_void,
    w_scale: *const f32,
    output: *mut c_void,
    workspace: *mut c_void,
    workspace_bytes: usize,
    stream: *mut c_void,
}

/// What a prepared tile costs on this device.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Resources {
    context: Context,
    pub major: i32,
    pub minor: i32,
    pub threads: i32,
    pub registers_per_thread: i32,
    pub shared_bytes: usize,
    pub local_bytes: usize,
}

extern "C" {
    fn hanzo_blockwise_fp8_error_string(status: i32) -> *const std::ffi::c_char;
    fn hanzo_blockwise_fp8_prepare(device: i32, tile: i32, resources: *mut Resources) -> i32;
    fn hanzo_blockwise_fp8_workspace_size(
        context: *const Context,
        shape: *const ShapeMnk,
        bytes: *mut usize,
    ) -> i32;
    fn hanzo_blockwise_fp8_gemm(launch: *const Launch) -> i32;
}

fn check(status: i32, what: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(hanzo_blockwise_fp8_error_string(status)) };
    hanzo_ml::bail!("block-FP8 CUTLASS {what} failed: {}", message.to_string_lossy())
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

/// Whether this device runs the lib: it carries sm_121a code only.
pub fn device_supported(dev: &CudaDevice) -> bool {
    matches!(compute_cap(dev), Ok((SM_MAJOR, SM_MINOR)))
}

/// Whether CUTLASS can run an `[N, K]` weight: full scale blocks in N and K.
pub fn shape_supported(n: usize, k: usize) -> bool {
    n > 0 && k > 0 && n % BLOCK == 0 && k % BLOCK == 0
}

/// Prepared once per (device, tile), outside any graph capture: sets the
/// kernel's shared-memory attribute and reads the SM count.
pub(crate) fn prepare(dev: &CudaDevice, tile: Tile) -> Result<Resources> {
    static PREPARED: OnceLock<Mutex<Vec<(i32, Tile, Resources)>>> = OnceLock::new();
    let cache = PREPARED.get_or_init(|| Mutex::new(Vec::new()));
    let device = dev.cuda_stream().context().ordinal() as i32;
    let mut guard = cache.lock().expect("block-FP8 CUTLASS cache poisoned");
    if let Some((_, _, r)) = guard.iter().find(|(d, t, _)| *d == device && *t == tile) {
        return Ok(*r);
    }
    dev.cuda_stream().context().bind_to_thread().map_err(hanzo_ml::Error::wrap)?;
    let mut resources = Resources::default();
    check(
        unsafe { hanzo_blockwise_fp8_prepare(device, tile.code(), &mut resources) },
        "prepare",
    )?;
    guard.push((device, tile, resources));
    Ok(resources)
}

/// Bytes of scratch the tile needs at this shape (0 for all three: no split-K).
pub(crate) fn workspace_size(dev: &CudaDevice, tile: Tile, m: usize, n: usize, k: usize) -> Result<usize> {
    let resources = prepare(dev, tile)?;
    let shape = shape_mnk(m, n, k)?;
    let mut bytes = 0usize;
    check(
        unsafe { hanzo_blockwise_fp8_workspace_size(&resources.context, &shape, &mut bytes) },
        "workspace size",
    )?;
    Ok(bytes)
}

fn shape_mnk(m: usize, n: usize, k: usize) -> Result<ShapeMnk> {
    let fits = |v: usize| i32::try_from(v).map_err(|_| hanzo_ml::Error::Msg(format!("block-FP8 GEMM dimension {v} exceeds i32")));
    Ok(ShapeMnk { m: fits(m)?, n: fits(n)?, k: fits(k)? })
}

/// Device pointer of a tensor's first element, after `contiguous()`: the
/// layout's start offset is part of the address, so row views stay correct.
struct Operand {
    tensor: Tensor,
}

impl Operand {
    fn new(t: &Tensor) -> Result<Self> {
        Ok(Self { tensor: t.contiguous()? })
    }

    fn with_ptr<T: DeviceRepr + hanzo_ml::cuda::CudaDType, R>(
        &self,
        what: &str,
        align: u64,
        f: impl FnOnce(u64) -> Result<R>,
    ) -> Result<R> {
        let (storage, layout) = self.tensor.storage_and_layout();
        let Storage::Cuda(s) = &*storage else {
            hanzo_ml::bail!("block-FP8 GEMM needs {what} on a CUDA device");
        };
        let slice = s.as_cuda_slice::<T>()?;
        let (ptr, _guard) = slice_ptr(slice, layout.start_offset());
        if ptr % align != 0 {
            hanzo_ml::bail!("block-FP8 GEMM needs {what} {align}-byte aligned, got address {ptr:#x}");
        }
        f(ptr)
    }
}

fn expect_dtype(t: &Tensor, dtype: DType, what: &str) -> Result<()> {
    if t.dtype() != dtype {
        hanzo_ml::bail!("block-FP8 GEMM needs {what} as {dtype:?}, got {:?}", t.dtype());
    }
    Ok(())
}

/// `out[M, N] = sum_kb (sa[:, kb] * sw[:, kb]) * (qa_kb @ qw_kb^T)` in bf16,
/// on the tile vLLM picks for this M.
///
/// `qa`: e4m3 `[M, K]`, `sa`: f32 `[M, K/128]`, `qw`: e4m3 `[N, K]`,
/// `sw`: f32 `[N/128, K/128]`, all on one CUDA device. Any layout is
/// accepted (non-contiguous views are copied); N and K must be multiples of 128.
pub fn matmul(qa: &Tensor, sa: &Tensor, qw: &Tensor, sw: &Tensor) -> Result<Tensor> {
    let m = qa.dims2()?.0;
    matmul_tile(qa, sa, qw, sw, Tile::for_rows(m))
}

/// [`matmul`] on a chosen tile. All three give the same bits.
pub(crate) fn matmul_tile(qa: &Tensor, sa: &Tensor, qw: &Tensor, sw: &Tensor, tile: Tile) -> Result<Tensor> {
    expect_dtype(qa, DType::F8E4M3, "activations")?;
    expect_dtype(qw, DType::F8E4M3, "weights")?;
    expect_dtype(sa, DType::F32, "activation scales")?;
    expect_dtype(sw, DType::F32, "weight scales")?;
    let (m, k) = qa.dims2()?;
    let (n, kw) = qw.dims2()?;
    if kw != k {
        hanzo_ml::bail!("block-FP8 GEMM K mismatch: activations {k}, weights {kw}");
    }
    if !shape_supported(n, k) {
        hanzo_ml::bail!("block-FP8 GEMM needs N and K multiples of {BLOCK}, got N={n} K={k}");
    }
    if sa.dims2()? != (m, k / BLOCK) {
        hanzo_ml::bail!("block-FP8 GEMM needs activation scales [{m}, {}], got {:?}", k / BLOCK, sa.dims());
    }
    if sw.dims2()? != (n / BLOCK, k / BLOCK) {
        hanzo_ml::bail!(
            "block-FP8 GEMM needs weight scales [{}, {}], got {:?}",
            n / BLOCK,
            k / BLOCK,
            sw.dims()
        );
    }
    let Device::Cuda(dev) = qa.device().clone() else {
        hanzo_ml::bail!("block-FP8 GEMM needs activations on a CUDA device");
    };
    for (t, what) in [(sa, "activation scales"), (qw, "weights"), (sw, "weight scales")] {
        if !t.device().same_device(qa.device()) {
            hanzo_ml::bail!("block-FP8 GEMM needs {what} on the activations' CUDA device");
        }
    }
    if m == 0 {
        return Tensor::zeros((0, n), DType::BF16, qa.device());
    }

    let resources = prepare(&dev, tile)?;
    let shape = shape_mnk(m, n, k)?;
    let mut workspace_bytes = 0usize;
    check(
        unsafe { hanzo_blockwise_fp8_workspace_size(&resources.context, &shape, &mut workspace_bytes) },
        "workspace size",
    )?;
    let workspace = if workspace_bytes == 0 {
        None
    } else {
        Some(unsafe { dev.alloc::<u8>(workspace_bytes)? })
    };
    // Every output element is written by the epilogue: no memset.
    let output = unsafe { dev.alloc::<bf16>(m * n)? };
    let stream = dev.cuda_stream().cu_stream() as *mut c_void;

    let (qa, sa, qw, sw) = (Operand::new(qa)?, Operand::new(sa)?, Operand::new(qw)?, Operand::new(sw)?);
    qa.with_ptr::<F8E4M3, _>("activations", OPERAND_ALIGN, |a| {
        sa.with_ptr::<f32, _>("activation scales", SCALE_ALIGN, |a_scale| {
            qw.with_ptr::<F8E4M3, _>("weights", OPERAND_ALIGN, |w| {
                sw.with_ptr::<f32, _>("weight scales", SCALE_ALIGN, |w_scale| {
                    let (out, _out_guard) = slice_ptr(&output, 0);
                    let work = workspace.as_ref().map(|ws| slice_ptr(ws, 0));
                    let launch = Launch {
                        shape,
                        context: resources.context,
                        a: a as *const c_void,
                        a_scale: a_scale as *const f32,
                        w: w as *const c_void,
                        w_scale: w_scale as *const f32,
                        output: out as *mut c_void,
                        workspace: work.as_ref().map_or(std::ptr::null_mut(), |(p, _)| *p as *mut c_void),
                        workspace_bytes,
                        stream,
                    };
                    check(unsafe { hanzo_blockwise_fp8_gemm(&launch) }, "gemm")
                })
            })
        })
    })?;
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
        Shape::from((m, n)),
    )))
}

#[cfg(test)]
mod tests;
