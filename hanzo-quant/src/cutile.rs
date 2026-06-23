//! cuTile CUDA tile-based MoE kernels.
//!
//! WIP STUB — the kernels are not yet implemented; every entry point is a
//! `todo!()`. Signatures are inferred from the call sites in `hanzo-engine`'s
//! MoE expert backends. This module is gated behind `cuda` + `cutile` (see
//! `lib.rs`), so default and CI builds skip it entirely.

use hanzo_ml::cuda::cudarc::driver::CudaSlice;
use hanzo_ml::{CudaDevice, Device, Result, Tensor};

/// Launch configuration for a grouped-GEMM tile kernel.
#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    /// Tile size along the token (M) dimension.
    pub bm: usize,
}

/// Whether the cuTile kernels support the given CUDA device.
pub fn device_supported(_dev: &CudaDevice) -> bool {
    todo!("cuTile device support probe")
}

/// Pre-compile / warm up the MoE kernels for the device.
pub fn warmup_moe_kernels(_dev: &Device) -> Result<()> {
    todo!("cuTile MoE kernel warmup")
}

/// Register an expert weight shape so kernels can be specialized ahead of time.
pub fn register_moe_shape(
    _gate_up: Tensor,
    _down: Tensor,
    _num_experts: usize,
    _num_experts_per_tok: usize,
) {
    todo!("cuTile MoE shape registration")
}

/// Default tile config for a given token count and expert count.
pub fn get_default_config(_num_tokens: usize, _num_experts: usize) -> Cfg {
    todo!("cuTile default config")
}

/// Grouped GEMM over the sorted, expert-aligned token layout produced by
/// `moe::cuda::moe_align`.
#[allow(clippy::too_many_arguments)]
pub fn cutile_grouped_gemm(
    _xs: &Tensor,
    _w: &Tensor,
    _sorted_ids: &CudaSlice<u32>,
    _expert_ids: &CudaSlice<u32>,
    _num_tokens_post_pad: &CudaSlice<u32>,
    _scale: Option<&CudaSlice<f32>>,
    _expert_meta: usize,
    _num_valid: usize,
    _topk: usize,
    _mul: bool,
    _cfg: Cfg,
    _dev: &CudaDevice,
) -> Result<Tensor> {
    todo!("cuTile grouped GEMM")
}
