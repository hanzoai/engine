pub(super) mod cpu;
mod flash;
pub mod lsa;
#[cfg(feature = "metal")]
pub(crate) mod metal_flash_attn;
mod naive;
mod sinks;
mod tiled;

pub(crate) use flash::flash_attn;
pub use lsa::{
    build_buckets, build_hierarchy, build_shapes, lsa_sdpa, pca_projections, project, regroup,
    select_keep_indices, shape_buckets, sparse_gather_sdpa, Buckets, EarlyBudget, Growth,
    Hierarchy, LearnedProjection, LsaConfig, LsaSessionCache, RotaryLayout, Shapes,
};
pub(crate) use naive::{maybe_synchronize, naive_sdpa};
pub(crate) use sinks::sinks_attn;
pub(crate) use tiled::tiled_sdpa;

#[cfg(not(feature = "metal"))]
pub(crate) mod metal_flash_attn {
    use hanzo_ml::{Result, Tensor};
    pub fn try_flash_attn_ext_bf16_dk512(
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _mask: &Tensor,
        _scale: f32,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
    pub fn try_flash_attn_ext_vec_bf16_dk512(
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _mask: Option<&Tensor>,
        _scale: f32,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
}
