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
pub use config::{KvCacheLayout, KvLayers, ModelConfigLike, ModelConfigMetadata};
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

// A token costs every cache entry its own layer's size (`kv_cache_elements_per_token` sums them),
// so a hybrid whose side caches are narrower than its attention layers is not charged the widest.
macro_rules! mb_to_blocks {
    ($mb_size:expr, $dtype_size:expr, $block_size:expr, $config:expr) => {
        $mb_size / $dtype_size / $block_size / $config.kv_cache_elements_per_token()
    };
}

macro_rules! ctxt_to_blocks {
    ($context_len:expr, $dtype_size:expr, $block_size:expr, $config:expr) => {
        $context_len * $dtype_size * $config.kv_cache_elements_per_token()
    };
}

/// What a unified-memory plan is asked to size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvRequest {
    /// The automatic path: the working set (`demand` tokens), at least one context (`floor`
    /// tokens), within what fits.
    Auto { demand: usize, floor: usize },
    /// An explicit context of this many tokens: it fits or it is an error.
    Tokens(usize),
    /// An explicit KV budget in bytes: it fits or it is an error.
    Bytes(usize),
}

/// The KV cache bytes a unified-memory device gives the paged cache. The ceiling is the device's
/// budget (`IGPU_MEMORY_FRACTION` of physical RAM, already applied by [`MemoryUsage`]); charged
/// against it are the weights, the fixed non-KV memory (recurrent states, prefix snapshots,
/// activations, runtime) and the KV. There is no second reserve on top of the fraction. An
/// explicit request that does not fit is an error naming the shortfall, never a silent cap.
pub(crate) fn plan_unified(
    ceiling: usize,
    weights: usize,
    fixed: usize,
    request: KvRequest,
    per_token: usize,
) -> anyhow::Result<usize> {
    const GIB: f64 = (1u64 << 30) as f64;
    let room = ceiling as i128 - weights as i128 - fixed as i128;
    let short = |need: usize| {
        anyhow::anyhow!(
            "unified memory: {:.2} GiB of KV does not fit; the {:.2} GiB ceiling holds {:.2} GiB of weights and {:.2} GiB fixed, {:.2} GiB short (raise IGPU_MEMORY_FRACTION or ask for less)",
            need as f64 / GIB,
            ceiling as f64 / GIB,
            weights as f64 / GIB,
            fixed as f64 / GIB,
            (need as i128 - room) as f64 / GIB,
        )
    };
    let need = match request {
        KvRequest::Tokens(t) => t * per_token,
        KvRequest::Bytes(b) => b,
        KvRequest::Auto { demand, floor } => {
            let room_bytes = room.max(0) as usize;
            let want = (demand * per_token).min(room_bytes).max(floor * per_token);
            if (want as i128) > room {
                return Err(short(want));
            }
            return Ok(want);
        }
    };
    if (need as i128) > room {
        return Err(short(need));
    }
    Ok(need)
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
/// `max_num_tokens`: on UNIFIED-memory devices (Metal, ROCm/Vulkan APUs, integrated/coherent
/// CUDA), caps the KV cache to this many tokens = the actual concurrent working set
/// (`max_seq_len * max_batch_size` = context * concurrency). Unlike discrete VRAM where unused
/// memory is otherwise wasted, wired KV buffers here compete with the model, OS, and compute for
/// one shared physical pool, so KV is sized to DEMAND, not capacity -- more concurrent
/// sessions/agents raise `max_batch_size` and grow KV automatically, always bounded by the
/// post-model/OS-reserve ceiling. On discrete CUDA this is ignored. Falls back to one context.
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
    if config.kv_layers().is_empty() {
        anyhow::bail!("Model has no layer that holds a KV cache; disable PagedAttention.");
    }
    let dtype = cache_type.to_dtype(dtype);
    let dtype_size = dtype.size_in_bytes();

    // For tensor parallelism, each device holds a fraction of the model weights. Approximate it like this.
    let num_devices = layer_devices.len().max(1);
    let model_weight_per_device_mb =
        model_weight_size_in_bytes.unwrap_or(0) / num_devices / SIZE_IN_MB;

    // A budget the caller named is an instruction, not a hint: the demand floor below applies only
    // to the automatic path.
    let mem_gpu_request = mem_gpu;

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
        // One pool holds the weights, the fixed working set and the KV, so the KV is planned
        // against the device budget (the fraction), sized to demand on the automatic path.
        let per_token = dtype_size * config.kv_cache_elements_per_token();
        let mem = MemoryUsage.query(device)?;
        let (ceiling, weights) = match model_weight_size_in_bytes {
            Some(w) => (mem.total(), w / num_devices),
            // After loading, what is allocated already holds the weights.
            None => (mem.available(), 0),
        };
        let request = match mem_gpu_request {
            MemoryGpuConfig::ContextSize(t) => KvRequest::Tokens(t),
            MemoryGpuConfig::MbAmount(mb) => KvRequest::Bytes(mb * SIZE_IN_MB),
            MemoryGpuConfig::Utilization(_) => KvRequest::Auto {
                demand: max_num_tokens.unwrap_or(config.max_seq_len()).max(config.max_seq_len()),
                floor: config.max_seq_len(),
            },
        };
        let kv = plan_unified(ceiling, weights, 0, request, per_token)?;
        let target = kv / SIZE_IN_MB;
        if target != mem_gpu && !silent {
            info!(
                "Unified memory: KV cache {} MB -> {} MB ({} max-context sequences).",
                mem_gpu,
                target,
                (kv / (per_token * config.max_seq_len()).max(1)).max(1),
            );
        }
        mem_gpu = target;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// GB10: MemTotal 127,600,508 KiB. Serving Qwen3.8-Flash-Next at 1,048,576 bf16 tokens with 8
    /// sequences: 77,632,650,240 B of weights, 27,456 B of KV per token, and fixed memory of 8 x
    /// 2 x 116,379,648 B of recurrent state plus 1 GiB of prefix snapshots and 3.5 GiB runtime.
    fn gb10(fraction: f64) -> anyhow::Result<usize> {
        let mem_total = 127_600_508usize * 1024;
        let ceiling = (mem_total as f64 * fraction) as usize;
        let fixed = 8 * 2 * 116_379_648 + (1 << 30) + (7 << 29);
        plan_unified(ceiling, 77_632_650_240, fixed, KvRequest::Tokens(1_048_576), 27_456)
            .map(|kv| ceiling - 77_632_650_240 - fixed - kv)
    }

    #[test]
    fn unified_plan_fits_1m_at_090() {
        let left = gb10(0.90).expect("1M fits at 0.90");
        assert!(left >= 4 << 30, "only {:.2} GiB of ceiling left", left as f64 / (1u64 << 30) as f64);
    }

    #[test]
    fn unified_plan_refuses_1m_at_085() {
        let err = gb10(0.85).expect_err("1M must not fit at 0.85").to_string();
        assert!(err.contains("1.91 GiB short"), "{err}");
    }

    #[test]
    fn unified_plan_auto_is_demand_capped() {
        // demand below room: the demand; demand above room: the room; floor above room: error
        let r = plan_unified(100, 20, 10, KvRequest::Auto { demand: 5, floor: 2 }, 4).unwrap();
        assert_eq!(r, 20);
        let r = plan_unified(100, 20, 10, KvRequest::Auto { demand: 50, floor: 2 }, 4).unwrap();
        assert_eq!(r, 70);
        assert!(plan_unified(100, 20, 10, KvRequest::Auto { demand: 50, floor: 20 }, 4).is_err());
        assert!(plan_unified(100, 20, 10, KvRequest::Bytes(71), 4).is_err());
        assert_eq!(plan_unified(100, 20, 10, KvRequest::Bytes(70), 4).unwrap(), 70);
    }

    /// Qwen3.5-27B shape: 64 decoder layers, 4 KV heads of 128, of which 16 are full attention.
    fn dense() -> ModelConfigMetadata {
        ModelConfigMetadata {
            max_seq_len: 262144,
            num_layers: 64,
            hidden_size: 4096,
            num_kv_heads: 4,
            num_attn_heads: 32,
            sliding_window: None,
            k_head_dim: 128,
            v_head_dim: 128,
            kv_cache_layout: KvCacheLayout::StandardNoFlashInfer,
        }
    }

    fn hybrid() -> KvLayers<ModelConfigMetadata> {
        KvLayers::new(dense(), (0..64).filter(|i| (i + 1) % 4 == 0).collect())
    }

    fn blocks(config: &dyn ModelConfigLike, mem_gpu: MemoryGpuConfig) -> usize {
        calculate_cache_config(
            mem_gpu,
            Some(32),
            DType::BF16,
            PagedCacheType::Auto,
            config,
            &Device::Cpu,
            &[None],
            true,
            None,
            None,
        )
        .unwrap()
        .num_gpu_blocks
    }

    #[test]
    fn hybrid_caches_only_its_attention_layers() {
        assert_eq!(hybrid().kv_layers().len(), 16);
        assert_eq!(
            blocks(&hybrid(), MemoryGpuConfig::MbAmount(8192)),
            4 * blocks(&dense(), MemoryGpuConfig::MbAmount(8192))
        );
    }

    #[test]
    fn hybrid_context_costs_a_quarter_of_the_bytes() {
        let bytes =
            |config: &dyn ModelConfigLike| ctxt_to_blocks!(262144usize, 2usize, 32usize, config);
        // 16 layers x 2 (K,V) x 4 kv heads x 128 head dim x 2 bytes = 32 KB/token.
        assert_eq!(bytes(&hybrid()), 262144 * 32 * 1024);
        assert_eq!(bytes(&dense()), 4 * bytes(&hybrid()));
    }

    /// qwen4exp shape: 48 decoder layers with gated attention (2 KV heads of 256) at every
    /// fourth, and after them each attention layer's QSA index cache (1 head of 128) at
    /// `48 + layer`, read by that layer.
    pub(super) struct Indexed;

    impl Indexed {
        const DEPTH: usize = 48;

        fn attention() -> impl Iterator<Item = usize> {
            (0..Self::DEPTH).filter(|i| (i + 1) % 4 == 0)
        }
    }

    impl ModelConfigLike for Indexed {
        fn max_seq_len(&self) -> usize {
            262144
        }
        fn num_layers(&self) -> usize {
            Self::DEPTH
        }
        fn hidden_size(&self) -> usize {
            2560
        }
        fn num_kv_heads(&self) -> usize {
            2
        }
        fn num_attn_heads(&self) -> usize {
            24
        }
        fn k_head_dim(&self) -> usize {
            256
        }
        fn v_head_dim(&self) -> usize {
            256
        }
        fn num_kv_heads_for_layer(&self, layer_idx: usize) -> usize {
            if layer_idx < Self::DEPTH {
                2
            } else {
                1
            }
        }
        fn k_head_dim_for_layer(&self, layer_idx: usize) -> usize {
            if layer_idx < Self::DEPTH {
                256
            } else {
                128
            }
        }
        fn v_head_dim_for_layer(&self, layer_idx: usize) -> usize {
            self.k_head_dim_for_layer(layer_idx)
        }
        fn kv_layers(&self) -> Vec<usize> {
            Self::attention()
                .chain(Self::attention().map(|l| Self::DEPTH + l))
                .collect()
        }
        fn kv_reader(&self, layer_idx: usize) -> Option<usize> {
            Some(layer_idx % Self::DEPTH)
        }
    }

    /// The budget as it was before entries were sized per layer: every entry charged the model's
    /// one `2 * kv_heads * max(k, v)`, or its MLA latent row.
    fn uniform(config: &ModelConfigMetadata, mb: usize) -> usize {
        let row = match config.kv_cache_layout {
            KvCacheLayout::Mla {
                kv_lora_rank,
                kpe_head_dim,
            } => kv_lora_rank + kpe_head_dim,
            _ => 2 * config.num_kv_heads * config.k_head_dim.max(config.v_head_dim),
        };
        let entries = config.kv_layers().len();
        mb * SIZE_IN_MB / 2 / 32 / entries / row
    }

    #[test]
    fn uniform_models_keep_their_block_counts() {
        let mla = ModelConfigMetadata {
            num_layers: 61,
            kv_cache_layout: KvCacheLayout::Mla {
                kv_lora_rank: 512,
                kpe_head_dim: 64,
            },
            ..dense()
        };
        // Absorbed-MLA GGUF shape: one head, K wider than V.
        let absorbed = ModelConfigMetadata {
            num_kv_heads: 1,
            k_head_dim: 576,
            v_head_dim: 512,
            ..dense()
        };
        for config in [&dense(), &mla, &absorbed] {
            for mb in [7, 8192, 12345, 65536] {
                assert_eq!(
                    blocks(config, MemoryGpuConfig::MbAmount(mb)),
                    uniform(config, mb)
                );
            }
        }
        for mb in [7, 8192, 12345] {
            let hybrid = hybrid();
            let entries = hybrid.kv_layers().len();
            assert_eq!(
                blocks(&hybrid, MemoryGpuConfig::MbAmount(mb)),
                mb * SIZE_IN_MB / 2 / 32 / entries / (2 * 4 * 128)
            );
        }
        for toks in [4096, 100_003, 262144] {
            let mb = toks * 2 * 64 * (2 * 4 * 128) / SIZE_IN_MB;
            assert_eq!(
                blocks(&dense(), MemoryGpuConfig::ContextSize(toks)),
                uniform(&dense(), mb)
            );
        }
    }

    #[test]
    fn mixed_cache_charges_each_entry_its_own_size() {
        // 12 x 2 (K,V) x 2 heads x 256 + 12 x 2 x 1 head x 128 elements per token.
        let row = 12 * 1024 + 12 * 256;
        assert_eq!(Indexed.kv_layers().len(), 24);
        assert_eq!(Indexed.kv_cache_elements_per_token(), row);
        for mb in [7, 8192, 12345] {
            let charged = blocks(&Indexed, MemoryGpuConfig::MbAmount(mb));
            assert_eq!(charged, mb * SIZE_IN_MB / 2 / 32 / row);
            // Charging all 24 entries the attention layers' 1024 would have lost 3/8 of them.
            assert!(charged > mb * SIZE_IN_MB / 2 / 32 / 24 / 1024);
        }

        // What the engine allocates for those blocks is exactly what the budget charged.
        let cache = CacheConfig {
            block_size: 32,
            num_gpu_blocks: 3,
            cache_type: PagedCacheType::Auto,
        };
        let engine = CacheEngine::new(&Indexed, &cache, DType::BF16, &Device::Cpu, vec![]).unwrap();
        let held: usize = engine
            .get_kv_cache()
            .iter()
            .map(|(k, v)| k.elem_count() + v.elem_count())
            .sum();
        assert_eq!(held, 3 * 32 * row);
    }
}
