/// This is the lower-level manager of the cache. It manages swapping and copying the blocks and
/// actually allocates the KV cache for the CPU and GPU. It is used by the LLMEngine to execute
/// operations issued by the scheduler.
mod attention_backend;
/// Content-addressable block hashing for prefix caching (vLLM v1 approach).
pub mod block_hash;
/// Flat block pool with LRU free list for KV cache block management (vLLM v1 approach).
pub mod block_pool;
mod cache_engine;
mod config;
/// Encoder output cache for multimodal models (vision/audio encoder outputs).
pub mod encoder_cache;
/// KV Cache Manager: high-level block allocation, prefix cache lookups, per-request tracking.
pub mod kv_cache_manager;
mod layers;
mod scheduler;
pub const _PAD_SLOT_ID: i64 = -1;

pub use attention_backend::AttentionBackendKind;
#[cfg(any(
    all(feature = "cuda", target_family = "unix"),
    feature = "metal",
    feature = "rocm",
    feature = "vulkan"
))]
pub use attention_backend::{
    FLASHINFER_DECODE_MAX_HEAD_SIZE, STANDARD_PAGED_ATTENTION_MAX_HEAD_SIZE,
};
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub use attention_backend::{
    FLASHINFER_PREFILL_MAX_HEAD_SIZE, FLASHINFER_TENSOR_CORE_DECODE_ENABLED,
    FLASHINFER_TENSOR_CORE_DECODE_MAX_HEAD_SIZE,
};
pub use cache_engine::{CacheConfig, CacheEngine, PagedCacheType};
pub use config::{KvCacheLayout, ModelConfigLike, ModelConfigMetadata};
use hanzo_ml::{DType, Device};
pub use kv_cache_manager::KVCacheManager;
pub use layers::PagedAttention;
pub use scheduler::{
    PagedAttentionScheduler, PagedAttentionSchedulerConfig, PagedAttentionSchedulerOutput,
};

use crate::MemoryUsage;
use tracing::info;

pub const DEFAULT_PAGED_ATTENTION_BLOCK_SIZE: usize = 32;

/// All memory counts in MB. Default for block size is 32.
#[derive(Clone, Copy)]
pub struct PagedAttentionConfig {
    pub(crate) block_size: Option<usize>,
    pub(crate) mem_gpu: MemoryGpuConfig,
    pub(crate) cache_type: PagedCacheType,
}

impl PagedAttentionConfig {
    pub fn new(
        block_size: Option<usize>,
        mem_gpu: MemoryGpuConfig,
        cache_type: PagedCacheType,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            block_size,
            mem_gpu,
            cache_type,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionImplementation {
    Eager,
    PagedAttention,
}

#[derive(Clone, Copy)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
pub enum MemoryGpuConfig {
    MbAmount(usize),
    Utilization(f32),
    ContextSize(usize),
}

// See `pagedattention.cu` CALL_V1_LAUNCHER_BLOCK_SIZE
const SUPPORTED_BLOCK_SIZE: &[usize] = &[8, 16, 32];

const SIZE_IN_MB: usize = 1024 * 1024;

macro_rules! mb_to_blocks {
    ($mb_size:expr, $dtype_size:expr, $block_size:expr, $config:expr) => {
        $mb_size
            / $dtype_size
            / $block_size
            / $config.num_layers()
            / $config.kv_cache_elements_per_token()
    };
}

macro_rules! ctxt_to_blocks {
    ($context_len:expr, $dtype_size:expr, $block_size:expr, $config:expr) => {
        $context_len * $dtype_size * $config.num_layers() * $config.kv_cache_elements_per_token()
    };
}

/// Memory values are in MBs or a percentage in [0,1]. Specify block size or the default is 32.
///
/// `model_weight_size_in_bytes`: total model weight footprint. When provided, the per-device
/// share (divided by number of devices for tensor parallelism) is subtracted from the KV cache
/// memory budget. Pass `Some(total_model_size_in_bytes)` when calling **before** model loading
/// (e.g. during device mapping) so the KV cache estimate reflects memory that will actually
/// remain after the weights are loaded. Post-loading callers should pass `None` since
/// `get_memory_available()` already reflects the loaded model.
///
/// `max_num_tokens`: on Metal (unified memory), caps the KV cache to this many tokens.
/// Unlike CUDA with dedicated VRAM where unused memory is wasted, Metal's wired buffers
/// compete with the OS and CPU for the same physical RAM. On CUDA this is ignored.
/// If `None` on Metal, falls back to `config.max_seq_len()`.
#[allow(clippy::too_many_arguments)]
pub fn calculate_cache_config(
    mem_gpu: MemoryGpuConfig,
    block_size: Option<usize>,
    dtype: DType,
    cache_type: PagedCacheType,
    config: &dyn ModelConfigLike,
    device: &Device,
    layer_devices: &[Option<Device>],
    silent: bool,
    model_weight_size_in_bytes: Option<usize>,
    max_num_tokens: Option<usize>,
) -> anyhow::Result<CacheConfig> {
    let block_size = block_size.unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE);
    if !SUPPORTED_BLOCK_SIZE.contains(&block_size) {
        anyhow::bail!("Block size must be in {SUPPORTED_BLOCK_SIZE:?}, got {block_size}");
    }
    let dtype = cache_type.to_dtype(dtype);
    let dtype_size = dtype.size_in_bytes();

    // For tensor parallelism, each device holds a fraction of the model weights. Approximate it like this.
    let num_devices = layer_devices.len().max(1);
    let model_weight_per_device_mb =
        model_weight_size_in_bytes.unwrap_or(0) / num_devices / SIZE_IN_MB;

    let mut min_mem_gpu = usize::MAX;
    for dev in layer_devices {
        let device = dev.as_ref().unwrap_or(device);

        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let mem_gpu = match mem_gpu {
            MemoryGpuConfig::MbAmount(v) => v,
            MemoryGpuConfig::Utilization(f) => {
                let mem = MemoryUsage.query(device)?;
                let total = mem.total() as f32 / SIZE_IN_MB as f32;
                if model_weight_size_in_bytes.is_some() {
                    // Pre-loading: compute budget from total memory and known model size.
                    (total * f - model_weight_per_device_mb as f32).max(0.0) as usize
                } else {
                    let used = (mem.total() - mem.available()) as f32 / SIZE_IN_MB as f32;
                    (total * f - used).max(0.0) as usize
                }
            }
            MemoryGpuConfig::ContextSize(toks) => {
                // ContextSize is demand-driven (bytes needed for N tokens), not a memory budget, so model weight does not apply here.
                ctxt_to_blocks!(toks, dtype_size, block_size, config) / SIZE_IN_MB
            }
        };
        min_mem_gpu = min_mem_gpu.min(mem_gpu);
    }

    // Unified-memory devices share physical RAM with the OS/CPU, so KV-cache wired buffers compete
    // for it: grabbing ~90% (the discrete-VRAM vLLM approach) hangs or thrashes the allocation
    // (ROCm/WSL: an 88 GB alloc never returns; the Grace-Blackwell GB10 over-commits its 128 GB
    // unified pool). Cap KV to the model context instead. This covers Metal, the ROCm/Vulkan APUs,
    // AND integrated/coherent CUDA -- `is_integrated_gpu` reads CU_DEVICE_ATTRIBUTE_INTEGRATED, which
    // is 1 on Jetson and Grace-class parts (GB10/GH200). Discrete-VRAM CUDA stays on the vLLM path
    // (unused VRAM is otherwise wasted, so max request concurrency is the right default there).
    #[allow(unused_mut, unused_variables)]
    let mut mem_gpu = min_mem_gpu;
    let unified_memory = crate::utils::normal::is_integrated_gpu(device);
    #[cfg(feature = "rocm")]
    let unified_memory = unified_memory || device.is_rocm();
    #[cfg(feature = "vulkan")]
    let unified_memory = unified_memory || device.is_vulkan();
    if unified_memory {
        let one_ctx_mb =
            ctxt_to_blocks!(config.max_seq_len(), dtype_size, block_size, config) / SIZE_IN_MB;
        // KV competes with the model and the OS/compute working set for the shared pool. Reserve a
        // fixed headroom (20% of unified RAM, min 16 GB) and let KV use the rest: total - model -
        // reserve. On a large unified box (GB10 128 GB) that is tens of GB -> many concurrent
        // sequences; on a tight APU the reserve keeps KV from starving the OS (the ROCm/WSL 88 GB
        // alloc that never returns). Floor at one full context so a lone request always loads, and
        // never exceed the post-model budget `mem_gpu` already computed. An explicit
        // --pa-context-len (ContextSize upstream) sizes KV exactly and bypasses this.
        let total_mb = MemoryUsage.query(device)?.total() / SIZE_IN_MB;
        let reserve_mb = (total_mb / 5).max(16 * 1024);
        let kv_ceiling = total_mb
            .saturating_sub(model_weight_per_device_mb)
            .saturating_sub(reserve_mb)
            .max(one_ctx_mb);
        let target = mem_gpu.min(kv_ceiling).max(one_ctx_mb);
        if target != mem_gpu {
            if !silent {
                info!(
                    "Unified memory: KV cache {} MB -> {} MB ({} max-context sequences; {} MB OS reserve).",
                    mem_gpu,
                    target,
                    (target / one_ctx_mb.max(1)).max(1),
                    reserve_mb,
                );
            }
            mem_gpu = target;
        }
    }

    let num_gpu_blocks = mb_to_blocks!(mem_gpu * SIZE_IN_MB, dtype_size, block_size, config);
    if num_gpu_blocks == 0 {
        anyhow::bail!("Num GPU blocks is 0. This means there is not enough memory. Either reduce the memory amount/utilization/context size or disable PagedAttention.");
    }

    if !silent {
        info!("Allocating {mem_gpu} MB for PagedAttention KV cache per GPU");
        info!("PagedAttention KV cache type is {dtype:?}");
        info!("Using PagedAttention with block size {block_size} and {num_gpu_blocks} GPU blocks: available context length is {} tokens", num_gpu_blocks*block_size);
    }
    Ok(CacheConfig {
        block_size,
        num_gpu_blocks,
        cache_type,
    })
}
