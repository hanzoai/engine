//! Deterministic routing: a stable counting sort of the routed assignments by expert.

use std::ffi::c_void;

use hanzo_ml::{DType, Result, Tensor};

use super::ffi::{self, check};

/// Where every routed row goes, all on the device.
///
/// Flat row `f = t*k + j` (token `t`, slot `j`) lands at sorted position `dst[f]`; rows of one
/// expert are contiguous (`offsets[e]..offsets[e+1]`) and keep flat order.
#[derive(Debug)]
pub struct Route {
    /// I32 `[E+1]`.
    pub offsets: Tensor,
    /// I32 `[R]`: sorted position to flat row.
    pub src: Tensor,
    /// I32 `[R]`: flat row to sorted position.
    pub dst: Tensor,
    /// I32 `[E]`: compact index of an active expert, or -1.
    pub group: Tensor,
    /// I32 `[min(E, R)]`: expert of each compact group, -1 past `nactive`.
    pub active: Tensor,
    /// I32 `[1]`.
    pub nactive: Tensor,
    /// Kernels the routing ran (1 up to 4096 rows, 3 above).
    pub launches: usize,
}

/// Device buffers route writes, as addresses (the forward's arena provides them).
pub(crate) struct Buffers {
    pub offsets: u64,
    pub src: u64,
    pub dst: u64,
    pub group: u64,
    pub active: u64,
    pub nactive: u64,
    pub scratch: u64,
}

/// Scratch bytes route needs beyond its outputs.
pub(crate) fn scratch_bytes(r: usize, e: usize) -> usize {
    unsafe { ffi::hanzo_moe_route_scratch(r as i32, e as i32) }
}

pub(crate) fn launch(
    ids: u64,
    m: usize,
    k: usize,
    e: usize,
    b: &Buffers,
    stream: ffi::Stream,
) -> Result<usize> {
    let mut launches = 0;
    check(
        unsafe {
            ffi::hanzo_moe_route(
                ids as *const i32,
                m as i32,
                k as i32,
                e as i32,
                b.offsets as *mut i32,
                b.src as *mut i32,
                b.dst as *mut i32,
                b.group as *mut i32,
                b.active as *mut i32,
                b.nactive as *mut i32,
                b.scratch as *mut c_void,
                &mut launches,
                stream,
            )
        },
        "route",
    )?;
    Ok(launches as usize)
}

/// Routes `ids` (`[M,k]`, U32 or I32, every value below `experts`).
pub fn route(ids: &Tensor, experts: usize) -> Result<Route> {
    let dev = ffi::cuda(ids.device())?;
    ffi::prepare(&dev)?;
    let (m, k) = ids.dims2()?;
    if !matches!(ids.dtype(), DType::U32 | DType::I32) {
        hanzo_ml::bail!("route ids must be U32 or I32, got {:?}", ids.dtype());
    }
    let ids = ids.contiguous()?;
    let r = m * k;
    let g = r.min(experts);
    let offsets = ffi::empty::<i32>(&dev, experts + 1)?;
    let src = ffi::empty::<i32>(&dev, r)?;
    let dst = ffi::empty::<i32>(&dev, r)?;
    let group = ffi::empty::<i32>(&dev, experts)?;
    let active = ffi::empty::<i32>(&dev, g)?;
    let nactive = ffi::empty::<i32>(&dev, 1)?;
    let scratch = ffi::empty::<u8>(&dev, scratch_bytes(r, experts).max(1))?;
    let b = Buffers {
        offsets: ffi::ptr(&offsets)?,
        src: ffi::ptr(&src)?,
        dst: ffi::ptr(&dst)?,
        group: ffi::ptr(&group)?,
        active: ffi::ptr(&active)?,
        nactive: ffi::ptr(&nactive)?,
        scratch: ffi::ptr(&scratch)?,
    };
    let launches = launch(ffi::ptr(&ids)?, m, k, experts, &b, ffi::stream(&dev))?;
    Ok(Route {
        offsets,
        src,
        dst,
        group,
        active,
        nactive,
        launches,
    })
}
