//! MoE CUDA helper kernels (token alignment, fused activation, reduction).
//!
//! WIP STUB — gated behind `cuda` + `cutile`; see [`cuda`]. Backs the cuTile
//! grouped-GEMM expert path in `hanzo-engine`.

pub mod cuda;
