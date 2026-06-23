//! CUDA MoE kernels backing the cuTile grouped-GEMM expert path.
//!
//! WIP STUB — the kernels are not yet implemented; every entry point is a
//! `todo!()`. Signatures are inferred from `hanzo-engine`'s MoE expert
//! backends. Gated behind `cuda` + `cutile`, so default and CI builds skip it.

use hanzo_ml::cuda::cudarc::driver::CudaSlice;
use hanzo_ml::{CudaDevice, Result, Tensor};

/// Sort tokens by expert and compute padded per-expert offsets.
///
/// Returns `(sorted_ids, expert_ids, num_tokens_post_pad, expert_meta)`,
/// consumed by `cutile::cutile_grouped_gemm`.
#[allow(clippy::type_complexity)]
pub fn moe_align(
    _topk_ids: &CudaSlice<u32>,
    _num_tokens: usize,
    _num_experts: usize,
    _topk: usize,
    _block_m: usize,
    _dev: &CudaDevice,
) -> Result<(CudaSlice<u32>, CudaSlice<u32>, CudaSlice<u32>, usize)> {
    todo!("MoE token alignment")
}

/// Fused gated-GELU (tanh approximation) followed by elementwise multiply.
pub fn gelu_tanh_and_mul(_xs: &Tensor, _inter: usize, _dev: &CudaDevice) -> Result<Tensor> {
    todo!("fused gelu-tanh and mul")
}

/// Weighted sum-reduction of per-expert outputs back to per-token, in bf16.
pub fn moe_sum_bf16(
    _xs: &Tensor,
    _num_tokens: usize,
    _topk: usize,
    _dev: &CudaDevice,
) -> Result<Tensor> {
    todo!("MoE bf16 sum reduction")
}
