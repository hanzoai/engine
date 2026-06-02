#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal", feature = "vulkan"))]
pub mod paged_attention;
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal", feature = "vulkan"))]
pub use paged_attention::PagedAttention;

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal", feature = "vulkan")))]
pub mod paged_attention {
    use hanzo_ml::{Device, Result, Tensor};

    use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
    use crate::{
        attention::{AttentionMask, SdpaParams},
        pipeline::text_models_inputs_processor::FlashParams,
    };

    pub struct PagedAttention;

    impl PagedAttention {
        pub fn new(
            _head_dim: usize,
            _device: &Device,
            _alibi_slopes: Option<Vec<f32>>,
        ) -> Result<Self> {
            hanzo_ml::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        #[allow(clippy::too_many_arguments)]
        #[allow(unused_variables)]
        pub fn forward(
            &self,
            _query: &Tensor,
            _key: &Tensor,
            _value: &Tensor,
            _attention_mask: &AttentionMask,
            _key_cache: Option<Tensor>,
            _value_cache: Option<Tensor>,
            _input_metadata: &PagedAttentionInputMetadata,
            _sdpa_params: &SdpaParams,
            _flash_params: Option<&FlashParams>,
        ) -> Result<Tensor> {
            hanzo_ml::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        #[allow(clippy::too_many_arguments)]
        #[allow(unused_variables)]
        pub fn forward_donor_cache(
            &self,
            _query: &Tensor,
            _key_cache: &Tensor,
            _value_cache: &Tensor,
            _attention_mask: &AttentionMask,
            _input_metadata: &PagedAttentionInputMetadata,
            _sdpa_params: &SdpaParams,
            _flash_params: Option<&FlashParams>,
        ) -> Result<Tensor> {
            hanzo_ml::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }
    }
}

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal", feature = "vulkan")))]
pub use paged_attention::PagedAttention;
