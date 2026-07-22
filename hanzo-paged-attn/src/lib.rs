// Backend-agnostic symmetric int8 KV-cache quantization (CPU reference + shared packing
// convention). Always compiled; it is the oracle every backend's int8 kernel is validated
// against.
pub mod quant;

#[cfg(all(feature = "cuda", target_family = "unix"))]
mod cuda;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub use cuda::*;

#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "metal")]
pub use metal::*;

#[cfg(feature = "rocm")]
mod rocm;
#[cfg(feature = "rocm")]
pub use rocm::*;

#[cfg(feature = "vulkan")]
mod vulkan;
#[cfg(feature = "vulkan")]
pub use vulkan::*;
