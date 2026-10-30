//! Routed experts on sm_121a tensor cores: route, expand, grouped GEMM, act, grouped GEMM,
//! combine, with no host synchronization, so a forward can be captured in a CUDA graph.
//!
//! [`nvfp4::Experts`] are Flash-Next's NVFP4 routed experts (FlashInfer `cutlass_fused_moe` at
//! vLLM's production tactic); [`fp8::Experts`] its MTP drafter's block-FP8 experts with BF16
//! scales (vLLM's Triton path).

mod combine;
mod ffi;
pub mod fp8;
pub mod nvfp4;
mod route;

pub use combine::{combine, Rule};
pub use route::{route, Route};

/// Grouped-GEMM tile. `P` is FlashInfer's production 128x128x128; `D32` and `D64` swap A and B
/// (weights as A) so a handful of rows per expert fills a 32- or 64-wide N tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    P = 0,
    D32 = 1,
    D64 = 2,
}

impl Tile {
    pub const ALL: [Tile; 3] = [Tile::P, Tile::D32, Tile::D64];

    /// The tile for `rows` routed rows over `experts`, by mean rows per expert.
    pub fn pick(rows: usize, experts: usize) -> Tile {
        let mean = rows.div_ceil(experts.max(1));
        if mean <= 16 {
            Tile::D32
        } else if mean <= 64 {
            Tile::D64
        } else {
            Tile::P
        }
    }
}

/// `hanzo_moe_shape`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Shape {
    pub m: i32,
    pub k: i32,
    pub e: i32,
    pub h: i32,
    pub i: i32,
    pub tile: i32,
    pub sm_count: i32,
}

/// Byte offsets of one forward's transient workspace (`hanzo_moe_layout`). Region S (`a1`,
/// `s1`, `y1`) is dead once act has read `y1`, so GEMM2 writes `y2` over it.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Layout {
    pub offsets: usize,
    pub src: usize,
    pub dst: usize,
    pub group: usize,
    pub active: usize,
    pub nactive: usize,
    pub route: usize,
    pub a1: usize,
    pub s1: usize,
    pub y1: usize,
    pub a2: usize,
    pub s2: usize,
    pub y2: usize,
    pub w: usize,
    pub xq: usize,
    pub xs: usize,
    pub args1: usize,
    pub args2: usize,
    pub gemm: usize,
    pub total: usize,
}

#[cfg(test)]
mod tests;
