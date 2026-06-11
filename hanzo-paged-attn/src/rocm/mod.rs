// ROCm/HIP backend for PagedAttention, mirroring src/cuda/mod.rs.
//
// Stage 1 scope: the CORE v1 decode path only (paged_attention + reshape_and_cache).
// v2 / MLA / flashinfer / quantization are intentionally not ported here.

pub const USE_FP8: bool = false;

mod backend;
mod ffi;

pub use backend::{gather_kv_cache, kv_scale_update, paged_attention, reshape_and_cache};
