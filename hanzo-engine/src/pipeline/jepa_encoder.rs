#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
//! I-JEPA image-embedding pipeline.
//!
//! Takes a single image per request and returns the mean-pooled per-patch embedding of
//! the I-JEPA ViT encoder (1280-d for ViT-H/14). The encoder itself is reused verbatim
//! from [`hanzo_transformers::models::ijepa`] — this module is only the engine glue:
//! image preprocessing (matching the HF `ViTImageProcessor`: resize→rescale→normalize
//! 0.5/0.5), a stateless [`Pipeline`] that emits [`ForwardInputsResult::Embeddings`], and
//! a tokenizer-free [`Loader`].
//!
//! Selected via the embedding CLI with an explicit arch, e.g.
//! `hanzo serve embedding --model-id facebook/ijepa_vith14_1k --arch ijepa`.

use std::any::Any;
use std::sync::Arc;

use anyhow::{Context, Result};
use hanzo_ml::{DType, Device, Tensor};
use hanzo_nn::VarBuilder;
use hanzo_transformers::models::ijepa::{Config as IjepaConfig, IJepaModel};
use hanzo_vision::{ImageTransform, Normalize, ToTensor};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use image::{imageops::FilterType, DynamicImage};
use rand_isaac::Isaac64Rng;
use tokio::sync::Mutex;
use tracing::info;

use super::inputs_processor::text_models_inputs_processor::PagedAttentionMeta;
use super::inputs_processor::{InputProcessorOutput, InputsProcessor, InputsProcessorType};
use super::processing::{MessagesAction, Processor};
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, ChatTemplate, EitherCache, ForwardInputsResult,
    GeneralMetadata, IsqPipelineMixin, Loader, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, TokenSource,
};
use crate::device_map::{DeviceMapSetting, DeviceMapper};
use crate::pipeline::sampling::sample_and_add_toks;
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::tokens::get_token;
use crate::{IsqType, Modalities, PagedAttentionConfig, Pipeline, SupportedModality, TryIntoDType};

/// ImageNet-symmetric normalization used by I-JEPA's `ViTImageProcessor`
/// (`image_mean = image_std = 0.5` → pixels mapped to `[-1, 1]`).
const IMAGE_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
const IMAGE_STD: [f64; 3] = [0.5, 0.5, 0.5];

/// Preprocess one image to a `(3, image_size, image_size)` tensor, matching the reference
/// `ViTImageProcessor`: bilinear resize → rescale to `[0, 1]` → normalize by mean/std.
fn preprocess_image(image: &DynamicImage, image_size: usize, device: &Device) -> Result<Tensor> {
    let rgb = DynamicImage::ImageRgb8(image.to_rgb8());
    let resized = rgb.resize_exact(image_size as u32, image_size as u32, FilterType::Triangle);
    let pixels = ToTensor.map(&resized, device)?;
    Normalize {
        mean: IMAGE_MEAN.to_vec(),
        std: IMAGE_STD.to_vec(),
    }
    .map(&pixels, device)
    .map_err(Into::into)
}

/// Downcast target produced by [`JepaInputsProcessor`] and consumed by
/// [`JepaEncoderPipeline::forward_inputs`]: a `(batch, 3, H, W)` pixel tensor.
struct JepaModelInputs {
    pixel_values: Tensor,
}

struct JepaInputsProcessor {
    image_size: usize,
}

impl InputsProcessor for JepaInputsProcessor {
    #[allow(clippy::too_many_arguments)]
    fn process_inputs(
        &self,
        _tokenizer: Option<Arc<tokenizers::Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _is_prompt: bool,
        _is_xlora: bool,
        device: &Device,
        _no_kv_cache: bool,
        _last_n_context_len: Option<(usize, usize)>,
        _return_raw_logits: bool,
        _sliding_window: Option<usize>,
        _other_config: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<PagedAttentionMeta>,
        _mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        let mut pixel_values = Vec::with_capacity(input_seqs.len());
        for seq in input_seqs.iter_mut() {
            let image = seq
                .take_images()
                .and_then(|imgs| imgs.into_iter().next())
                .context("I-JEPA embedding request carried no image")?;
            pixel_values.push(preprocess_image(&image, self.image_size, device)?);
        }
        let pixel_values = Tensor::stack(&pixel_values, 0)?;
        Ok(InputProcessorOutput {
            inputs: Box::new(JepaModelInputs { pixel_values }),
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }

    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }
}

struct JepaProcessor {
    image_size: usize,
}

impl Processor for JepaProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(JepaInputsProcessor {
            image_size: self.image_size,
        })
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

/// A stateless I-JEPA image encoder as an embedding pipeline.
pub struct JepaEncoderPipeline {
    model: IJepaModel,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    processor: Arc<dyn Processor + Send + Sync>,
    device: Device,
    dtype: DType,
}

impl PreProcessingMixin for JepaEncoderPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        self.processor.clone()
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for JepaEncoderPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!("ISQ is not supported for the I-JEPA image-embedding pipeline.")
    }
}

impl CacheManagerMixin for JepaEncoderPipeline {
    fn clone_in_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn clone_out_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn set_none_cache(
        &self,
        _seqs: &mut [&mut Sequence],
        _reset_non_granular: bool,
        _modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
    }
    fn cache(&self) -> &EitherCache {
        unreachable!("I-JEPA embedding pipeline has no KV cache")
    }
}

impl MetadataMixin for JepaEncoderPipeline {
    fn device(&self) -> Device {
        self.device.clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {}
    fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
        None
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

#[async_trait::async_trait]
impl Pipeline for JepaEncoderPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        _return_raw_logits: bool,
    ) -> hanzo_ml::Result<ForwardInputsResult> {
        let JepaModelInputs { pixel_values } = *inputs
            .downcast::<JepaModelInputs>()
            .expect("Downcast failed.");
        let pixel_values = pixel_values.to_device(&self.device)?.to_dtype(self.dtype)?;
        // (batch, num_patches, hidden) → mean over patches → (batch, hidden).
        let embeddings = self.model.forward_pooled(&pixel_values)?;
        Ok(ForwardInputsResult::Embeddings { embeddings })
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), hanzo_ml::Error> {
        // Never reached: OneShot embedding sequences return via the embedding arm of
        // `Pipeline::step` before any sampling. Kept to satisfy the trait.
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Embedding
    }
}

impl AnyMoePipelineMixin for JepaEncoderPipeline {}

/// A tokenizer-free loader that builds a [`JepaEncoderPipeline`] from an I-JEPA
/// checkpoint (`config.json` + `model.safetensors`).
pub struct JepaEncoderLoader {
    model_id: String,
    hf_cache_path: Option<std::path::PathBuf>,
}

impl JepaEncoderLoader {
    pub fn new(model_id: String, hf_cache_path: Option<std::path::PathBuf>) -> Self {
        Self {
            model_id,
            hf_cache_path,
        }
    }

    fn build(
        &self,
        config_path: &std::path::Path,
        weights_path: &std::path::Path,
        dtype: &dyn TryIntoDType,
        device: &Device,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let config: IjepaConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)
            .context("failed to parse I-JEPA config.json")?;
        let dtype = dtype.try_into_dtype(&[device])?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.to_path_buf()], dtype, device)?
        };
        let model = IJepaModel::new(&config, vb)?;

        let grid = config.image_size / config.patch_size;
        let num_patches = grid * grid;
        let mapper = DeviceMapSetting::dummy().into_mapper(
            config.num_hidden_layers,
            device,
            None,
            std::slice::from_ref(device),
        )?;
        let metadata = Arc::new(GeneralMetadata {
            max_seq_len: num_patches,
            llg_factory: None,
            is_xlora: false,
            no_prefix_cache: true,
            num_hidden_layers: 1,
            eos_tok: vec![],
            kind: ModelKind::Normal,
            no_kv_cache: true,
            activation_dtype: dtype,
            sliding_window: None,
            cache_config: None,
            cache_engine: None,
            model_metadata: None,
            modalities: Modalities {
                input: vec![SupportedModality::Vision],
                output: vec![SupportedModality::Embedding],
            },
        });
        info!(
            "I-JEPA encoder loaded: {} layers, hidden {}, {num_patches} patches.",
            config.num_hidden_layers, config.hidden_size
        );
        Ok(Arc::new(Mutex::new(JepaEncoderPipeline {
            model,
            model_id: self.model_id.clone(),
            metadata,
            mapper,
            processor: Arc::new(JepaProcessor {
                image_size: config.image_size,
            }),
            device: device.clone(),
            dtype,
        })))
    }
}

impl Loader for JepaEncoderLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        _mapper: DeviceMapSetting,
        _in_situ_quant: Option<IsqType>,
        _paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let mut builder = ApiBuilder::new().with_progress(!silent);
        if let Some(cache) = &self.hf_cache_path {
            builder = builder.with_cache_dir(cache.clone());
        }
        let api = builder.with_token(get_token(&token_source)?).build()?;
        let repo = api.repo(Repo::with_revision(
            self.model_id.clone(),
            RepoType::Model,
            revision.unwrap_or_else(|| "main".to_string()),
        ));
        let config_path = repo.get("config.json")?;
        let weights_path = repo.get("model.safetensors")?;
        self.build(&config_path, &weights_path, dtype, device)
    }

    #[allow(
        clippy::type_complexity,
        clippy::too_many_arguments,
        clippy::borrowed_box
    )]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        _silent: bool,
        _mapper: DeviceMapSetting,
        _in_situ_quant: Option<IsqType>,
        _paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let weights = paths
            .get_weight_filenames()
            .first()
            .context("no safetensors weight file for the I-JEPA model")?;
        self.build(paths.get_config_filename(), weights, dtype, device)
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::Normal
    }
}
