#![allow(clippy::cast_possible_truncation)]

use hanzo_ml::{Result, Tensor};

#[cfg(feature = "cuda")]
use hanzo_ml::DType;

pub const DFLASH2_MAX_TOPK: usize = 128;
pub const DFLASH2_MAX_WALK_RANK: usize = 4096;

pub struct DFlash2BlockConvParams<'a> {
    pub x: &'a Tensor,
    pub dynamic: &'a Tensor,
    pub base_kernel: &'a Tensor,
    pub residual: Option<&'a Tensor>,
    pub kernel_size: usize,
    pub group_size: usize,
    pub half_idx: usize,
}

pub struct DFlash2GreedyWalkParams<'a> {
    pub topk_values: &'a Tensor,
    pub topk_indices: &'a Tensor,
    pub gate_hidden: &'a Tensor,
    pub pred_codebook: &'a Tensor,
    pub succ_codebook: &'a Tensor,
    pub anchor_token: u32,
}

#[cfg(feature = "cuda")]
pub fn dflash2_block_conv_cuda(params: DFlash2BlockConvParams<'_>) -> Result<Tensor> {
    use core::ffi::c_void;
    use hanzo_ml::cuda_backend::cudarc::driver::DevicePtr;

    fn launch<
        T: hanzo_ml::cuda_backend::CudaDType + hanzo_ml::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        params: DFlash2BlockConvParams<'_>,
        caller: unsafe extern "C" fn(
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_void,
            *mut c_void,
            i32,
            i32,
            i32,
            i32,
            i32,
            i64,
        ),
    ) -> Result<Tensor> {
        let x = params.x.contiguous()?;
        let dynamic = params.dynamic.contiguous()?;
        let base_kernel = params.base_kernel.contiguous()?;
        let residual = match params.residual {
            Some(r) => Some(r.contiguous()?),
            None => None,
        };

        let (slots, hidden) = x.dims2()?;
        let dev = x.device().as_cuda_device()?;

        let (x_s, x_l) = x.storage_and_layout();
        let x_s = match &*x_s {
            hanzo_ml::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => hanzo_ml::bail!("dflash2_block_conv_cuda expects CUDA x"),
        };
        let x_off = x_l.start_offset();

        let (dyn_s, dyn_l) = dynamic.storage_and_layout();
        let dyn_s = match &*dyn_s {
            hanzo_ml::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => hanzo_ml::bail!("dflash2_block_conv_cuda expects CUDA dynamic"),
        };
        let dyn_off = dyn_l.start_offset();

        let (base_s, base_l) = base_kernel.storage_and_layout();
        let base_s = match &*base_s {
            hanzo_ml::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => hanzo_ml::bail!("dflash2_block_conv_cuda expects CUDA base_kernel"),
        };
        let base_off = base_l.start_offset();

        let out_buf = unsafe { dev.alloc::<T>(slots * hidden) }?;
        let stream = dev.cuda_stream().cu_stream() as i64;

        if let Some(ref res) = residual {
            let (res_s, res_l) = res.storage_and_layout();
            let res_s = match &*res_s {
                hanzo_ml::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => hanzo_ml::bail!("dflash2_block_conv_cuda expects CUDA residual"),
            };
            let res_off = res_l.start_offset();
            unsafe {
                caller(
                    x_s.slice(x_off..).device_ptr(x_s.stream()).0 as *const c_void,
                    dyn_s.slice(dyn_off..).device_ptr(dyn_s.stream()).0 as *const c_void,
                    base_s.slice(base_off..).device_ptr(base_s.stream()).0 as *const c_void,
                    res_s.slice(res_off..).device_ptr(res_s.stream()).0 as *const c_void,
                    out_buf.device_ptr(out_buf.stream()).0 as *mut c_void,
                    slots as i32,
                    hidden as i32,
                    params.kernel_size as i32,
                    params.group_size as i32,
                    params.half_idx as i32,
                    stream,
                );
            }
        } else {
            unsafe {
                caller(
                    x_s.slice(x_off..).device_ptr(x_s.stream()).0 as *const c_void,
                    dyn_s.slice(dyn_off..).device_ptr(dyn_s.stream()).0 as *const c_void,
                    base_s.slice(base_off..).device_ptr(base_s.stream()).0 as *const c_void,
                    core::ptr::null(),
                    out_buf.device_ptr(out_buf.stream()).0 as *mut c_void,
                    slots as i32,
                    hidden as i32,
                    params.kernel_size as i32,
                    params.group_size as i32,
                    params.half_idx as i32,
                    stream,
                );
            }
        }

        let out_storage = hanzo_ml::CudaStorage::wrap_cuda_slice(out_buf, dev.clone());
        Ok(Tensor::from((
            hanzo_ml::Storage::Cuda(out_storage),
            (slots, hidden),
        )))
    }

    match params.x.dtype() {
        DType::F16 => launch::<half::f16>(params, crate::cuda::ffi::dflash2_block_conv_f16),
        DType::BF16 => launch::<half::bf16>(params, crate::cuda::ffi::dflash2_block_conv_bf16),
        DType::F32 => launch::<f32>(params, crate::cuda::ffi::dflash2_block_conv_f32),
        other => hanzo_ml::bail!("dflash2_block_conv_cuda unsupported dtype {other:?}"),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn dflash2_block_conv_cuda(_params: DFlash2BlockConvParams<'_>) -> Result<Tensor> {
    hanzo_ml::bail!("dflash2_block_conv_cuda requires the cuda feature")
}

#[cfg(feature = "cuda")]
pub fn dflash2_topk_cuda(logits: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    use core::ffi::c_void;
    use hanzo_ml::cuda_backend::cudarc::driver::DevicePtr;

    let logits = logits.contiguous()?;
    let (nrows, ncols) = logits.dims2()?;
    let k = k.min(ncols);
    if k == 0 || k > DFLASH2_MAX_TOPK {
        hanzo_ml::bail!("dflash2_topk_cuda k={k} must be in 1..={DFLASH2_MAX_TOPK}");
    }
    let dev = logits.device().as_cuda_device()?;
    let vals_buf = unsafe { dev.alloc::<f32>(nrows * k) }?;
    let ids_buf = unsafe { dev.alloc::<u32>(nrows * k) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    macro_rules! run_topk {
        ($ty:ty, $func:ident) => {{
            let (l_s, l_l) = logits.storage_and_layout();
            let l_s = match &*l_s {
                hanzo_ml::Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?,
                _ => hanzo_ml::bail!("dflash2_topk_cuda expects CUDA logits"),
            };
            let l_off = l_l.start_offset();
            unsafe {
                crate::cuda::ffi::$func(
                    l_s.slice(l_off..).device_ptr(l_s.stream()).0 as *const c_void,
                    vals_buf.device_ptr(vals_buf.stream()).0 as *mut f32,
                    ids_buf.device_ptr(ids_buf.stream()).0 as *mut u32,
                    nrows as i32,
                    ncols as i32,
                    k as i32,
                    stream,
                );
            }
        }};
    }

    match logits.dtype() {
        DType::F16 => run_topk!(half::f16, dflash2_topk_f16),
        DType::BF16 => run_topk!(half::bf16, dflash2_topk_bf16),
        DType::F32 => run_topk!(f32, dflash2_topk_f32),
        other => hanzo_ml::bail!("dflash2_topk_cuda unsupported dtype {other:?}"),
    }

    let vals = Tensor::from((
        hanzo_ml::Storage::Cuda(hanzo_ml::CudaStorage::wrap_cuda_slice(
            vals_buf,
            dev.clone(),
        )),
        (nrows, k),
    ));
    let ids = Tensor::from((
        hanzo_ml::Storage::Cuda(hanzo_ml::CudaStorage::wrap_cuda_slice(ids_buf, dev.clone())),
        (nrows, k),
    ));
    Ok((vals, ids))
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn dflash2_topk_cuda(_logits: &Tensor, _k: usize) -> Result<(Tensor, Tensor)> {
    hanzo_ml::bail!("dflash2_topk_cuda requires the cuda feature")
}

#[cfg(feature = "cuda")]
pub fn dflash2_greedy_walk_cuda(params: DFlash2GreedyWalkParams<'_>) -> Result<Tensor> {
    use core::ffi::c_void;
    use hanzo_ml::cuda_backend::cudarc::driver::DevicePtr;

    let topk_values = params.topk_values.contiguous()?;
    let topk_indices = params.topk_indices.contiguous()?;
    let gate_hidden = params.gate_hidden.contiguous()?;
    let pred_codebook = params.pred_codebook.contiguous()?;
    let succ_codebook = params.succ_codebook.contiguous()?;

    let (slots, top_k) = topk_values.dims2()?;
    let (_, rank) = gate_hidden.dims2()?;
    if top_k == 0 || top_k > DFLASH2_MAX_TOPK {
        hanzo_ml::bail!("dflash2_greedy_walk_cuda top_k={top_k} must be in 1..={DFLASH2_MAX_TOPK}");
    }
    if rank == 0 || rank > DFLASH2_MAX_WALK_RANK {
        hanzo_ml::bail!(
            "dflash2_greedy_walk_cuda rank={rank} must be in 1..={DFLASH2_MAX_WALK_RANK}"
        );
    }

    let dev = gate_hidden.device().as_cuda_device()?;
    let out_buf = unsafe { dev.alloc::<u32>(slots) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    let (v_s, v_l) = topk_values.storage_and_layout();
    let v_s = match &*v_s {
        hanzo_ml::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => hanzo_ml::bail!("dflash2_greedy_walk_cuda expects CUDA topk_values"),
    };
    let v_off = v_l.start_offset();

    let (i_s, i_l) = topk_indices.storage_and_layout();
    let i_s = match &*i_s {
        hanzo_ml::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => hanzo_ml::bail!("dflash2_greedy_walk_cuda expects CUDA topk_indices"),
    };
    let i_off = i_l.start_offset();

    macro_rules! run_walk {
        ($ty:ty, $func:ident) => {{
            let (gh_s, gh_l) = gate_hidden.storage_and_layout();
            let gh_s = match &*gh_s {
                hanzo_ml::Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?,
                _ => hanzo_ml::bail!("dflash2_greedy_walk_cuda expects CUDA gate_hidden"),
            };
            let gh_off = gh_l.start_offset();

            let (pr_s, pr_l) = pred_codebook.storage_and_layout();
            let pr_s = match &*pr_s {
                hanzo_ml::Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?,
                _ => hanzo_ml::bail!("dflash2_greedy_walk_cuda expects CUDA pred_codebook"),
            };
            let pr_off = pr_l.start_offset();

            let (su_s, su_l) = succ_codebook.storage_and_layout();
            let su_s = match &*su_s {
                hanzo_ml::Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?,
                _ => hanzo_ml::bail!("dflash2_greedy_walk_cuda expects CUDA succ_codebook"),
            };
            let su_off = su_l.start_offset();

            unsafe {
                crate::cuda::ffi::$func(
                    v_s.slice(v_off..).device_ptr(v_s.stream()).0 as *const f32,
                    i_s.slice(i_off..).device_ptr(i_s.stream()).0 as *const u32,
                    gh_s.slice(gh_off..).device_ptr(gh_s.stream()).0 as *const c_void,
                    pr_s.slice(pr_off..).device_ptr(pr_s.stream()).0 as *const c_void,
                    su_s.slice(su_off..).device_ptr(su_s.stream()).0 as *const c_void,
                    out_buf.device_ptr(out_buf.stream()).0 as *mut u32,
                    params.anchor_token,
                    slots as i32,
                    top_k as i32,
                    rank as i32,
                    stream,
                );
            }
        }};
    }

    match gate_hidden.dtype() {
        DType::F16 => run_walk!(half::f16, dflash2_greedy_walk_f16),
        DType::BF16 => run_walk!(half::bf16, dflash2_greedy_walk_bf16),
        DType::F32 => run_walk!(f32, dflash2_greedy_walk_f32),
        other => hanzo_ml::bail!("dflash2_greedy_walk_cuda unsupported dtype {other:?}"),
    }

    Ok(Tensor::from((
        hanzo_ml::Storage::Cuda(hanzo_ml::CudaStorage::wrap_cuda_slice(out_buf, dev.clone())),
        (slots,),
    )))
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn dflash2_greedy_walk_cuda(_params: DFlash2GreedyWalkParams<'_>) -> Result<Tensor> {
    hanzo_ml::bail!("dflash2_greedy_walk_cuda requires the cuda feature")
}
