use hanzo_ml::Tensor;

use crate::pipeline::text_models_inputs_processor::FLASHINFER_DECODE_GROUP_SIZES;

#[cfg(any(
    all(feature = "cuda", target_family = "unix"),
    feature = "metal",
    feature = "rocm",
    feature = "vulkan"
))]
pub const STANDARD_PAGED_ATTENTION_MAX_HEAD_SIZE: usize = 256;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const FLASHINFER_PREFILL_MAX_HEAD_SIZE: usize = 256;
#[cfg(any(
    all(feature = "cuda", target_family = "unix"),
    feature = "metal",
    feature = "rocm",
    feature = "vulkan"
))]
pub const FLASHINFER_DECODE_MAX_HEAD_SIZE: usize = 512;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const FLASHINFER_TENSOR_CORE_DECODE_ENABLED: bool = false;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const FLASHINFER_TENSOR_CORE_DECODE_MAX_HEAD_SIZE: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionBackendKind {
    Standard,
    FlashInfer,
}

impl AttentionBackendKind {
    pub fn from_cache(key_cache: &Tensor, value_cache: &Tensor) -> Self {
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        {
            if hanzo_paged_attn::is_flashinfer_cache(key_cache, value_cache) {
                return Self::FlashInfer;
            }
        }
        #[cfg(not(all(feature = "cuda", target_family = "unix")))]
        let _ = (key_cache, value_cache);

        Self::Standard
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionLayerSpec {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
}

pub trait AttentionBackend {
    fn kind(&self) -> AttentionBackendKind;
    fn supports_layer(&self, spec: AttentionLayerSpec) -> bool;
}

pub struct FlashInferAttentionBackend;

impl AttentionBackend for FlashInferAttentionBackend {
    fn kind(&self) -> AttentionBackendKind {
        AttentionBackendKind::FlashInfer
    }

    fn supports_layer(&self, spec: AttentionLayerSpec) -> bool {
        if !cfg!(feature = "cuda") || !crate::perf_flags::flashinfer_decode_enabled() {
            return false;
        }
        spec.k_head_dim == spec.v_head_dim
            && matches!(spec.k_head_dim, 64 | 128 | 256 | 512)
            && decode_group_supported(spec.q_heads, spec.kv_heads)
    }
}

fn decode_group_supported(q_heads: usize, kv_heads: usize) -> bool {
    kv_heads != 0
        && q_heads.is_multiple_of(kv_heads)
        && FLASHINFER_DECODE_GROUP_SIZES.contains(&(q_heads / kv_heads))
}

#[cfg(test)]
mod tests {
    use super::decode_group_supported;

    #[test]
    fn decode_groups_match_the_kernel_instantiations() {
        for group in [1, 2, 3, 4, 6, 8] {
            assert!(decode_group_supported(group * 2, 2), "group {group}");
        }
        for group in [5, 7, 9, 16] {
            assert!(!decode_group_supported(group * 2, 2), "group {group}");
        }
        assert!(!decode_group_supported(24, 5));
        assert!(!decode_group_supported(24, 0));
    }
}
