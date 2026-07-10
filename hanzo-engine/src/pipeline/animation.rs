use super::text_models_inputs_processor::PagedAttentionMeta;
use super::{
    animation_loader, AdapterPaths, AnimationComponents, AnimationLoaderType, AnyMoePipelineMixin,
    Cache, CacheManagerMixin, ChatTemplate, EitherCache, EmbeddingModulePaths, ForwardInputsResult,
    GeneralMetadata, InputProcessorOutput, InputsProcessor, InputsProcessorType, IsqPipelineMixin,
    Loader, MessagesAction, MetadataMixin, Modalities, ModelCategory, ModelKind, ModelPaths,
    MuseTalkComponents, PreProcessingMixin, Processor, SupportedModality, TokenSource,
};
use crate::device_map::DeviceMapper;
use crate::diffusion_models::animation::{
    AnimationRequest, DrivingAudio, FacialAnimator, VisualKind, VisualSource, OMNI_SAMPLE_RATE,
};
use crate::diffusion_models::musetalk::{AnimatorOptions, MuseTalkConfig, UNetConfig, VaeConfig};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::speech_models::whisper::WhisperConfig;
use crate::utils::progress::ProgressScopeGuard;
use crate::utils::tokens::get_token;
use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
use crate::{
    api_get_file, DeviceMapSetting, MessageContent, PagedAttentionConfig, Pipeline, TryIntoDType,
};
use anyhow::Result;
use hanzo_ml::{DType, Device, Tensor};
use hanzo_quant::IsqType;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use image::DynamicImage;
use indexmap::IndexMap;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

/// One sequence's worth of animation input, gathered by `AnimationInputsProcessor`
/// from the sequence's image (frames), audio (PCM), and `animation_params` slots.
pub struct AnimationInput {
    pub frames: Vec<DynamicImage>,
    pub pcm: Arc<Vec<f32>>,
    pub sample_rate: usize,
    pub fps: f64,
    pub kind: VisualKind,
}

pub struct ModelInputs {
    pub requests: Vec<AnimationInput>,
}

pub struct AnimationProcessor;

impl Processor for AnimationProcessor {
    fn process(
        &self,
        _pipeline: &dyn Pipeline,
        _messages: Vec<IndexMap<String, MessageContent>>,
        _add_generation_prompt: bool,
        _add_special_tokens: bool,
        _enable_thinking: Option<bool>,
        _reasoning_effort: Option<crate::request::ReasoningEffort>,
        _tools: Vec<crate::Tool>,
    ) -> Result<(Vec<u32>, String)> {
        anyhow::bail!("AnimationProcessor::process does not expect chat messages.")
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(AnimationInputsProcessor)
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

pub struct AnimationInputsProcessor;

impl InputsProcessor for AnimationInputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Text
    }

    #[allow(clippy::too_many_arguments)]
    fn process_inputs(
        &self,
        _tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _is_prompt: bool,
        _is_xlora: bool,
        _device: &Device,
        _no_kv_cache: bool,
        _last_n_context_len: Option<(usize, usize)>,
        _return_raw_logits: bool,
        _sliding_window: Option<usize>,
        _other_config: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<PagedAttentionMeta>,
        _mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        let mut requests = Vec::with_capacity(input_seqs.len());
        for seq in input_seqs.iter_mut() {
            let params = seq
                .animation_params()
                .ok_or_else(|| anyhow::anyhow!("animation sequence missing animation_params"))?;
            let frames = seq.take_images().unwrap_or_default();
            let (pcm, sample_rate) = match seq.take_audios().and_then(|a| a.into_iter().next()) {
                Some(audio) => (Arc::new(audio.samples), audio.sample_rate as usize),
                None => (Arc::new(Vec::new()), OMNI_SAMPLE_RATE),
            };
            requests.push(AnimationInput {
                frames,
                pcm,
                sample_rate,
                fps: params.fps,
                kind: params.kind,
            });
        }
        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs { requests }),
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }
}

/// Composition pipeline over a `FacialAnimator`. The kind-gate is enforced here,
/// once, before dispatch -- never inside an animator impl. Muxing frames to a
/// container is a server-core concern, never this pipeline's.
pub struct AnimationPipeline {
    model_id: String,
    animator: Box<dyn FacialAnimator>,
    metadata: Arc<GeneralMetadata>,
    dummy_cache: EitherCache,
}

impl AnimationPipeline {
    pub fn new(animator: Box<dyn FacialAnimator>, model_id: String, dtype: DType) -> Self {
        Self {
            model_id,
            animator,
            metadata: Arc::new(GeneralMetadata {
                max_seq_len: 1,
                llg_factory: None,
                is_xlora: false,
                no_prefix_cache: false,
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
                    input: vec![SupportedModality::Vision, SupportedModality::Audio],
                    output: vec![SupportedModality::Vision],
                },
            }),
            dummy_cache: EitherCache::Full(Cache::new(0, false)),
        }
    }
}

impl PreProcessingMixin for AnimationPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(AnimationProcessor)
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for AnimationPipeline {
    fn re_isq_model(&mut self, _dtype: hanzo_quant::IsqType) -> anyhow::Result<()> {
        anyhow::bail!("Animation models do not support ISQ.")
    }
}

impl CacheManagerMixin for AnimationPipeline {
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
        &self.dummy_cache
    }
}

impl MetadataMixin for AnimationPipeline {
    fn device(&self) -> Device {
        self.animator.device().clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {}
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        None
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
}

#[async_trait::async_trait]
impl Pipeline for AnimationPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> hanzo_ml::Result<ForwardInputsResult> {
        assert!(!return_raw_logits);

        let ModelInputs { requests } = *inputs.downcast().expect("Downcast failed.");
        let mut frames = Vec::with_capacity(requests.len());
        let mut fps = Vec::with_capacity(requests.len());
        for input in requests {
            let visual = match input.kind {
                VisualKind::Portrait => {
                    let image = input.frames.into_iter().next().ok_or_else(|| {
                        hanzo_ml::Error::msg("portrait animation requires at least one frame")
                    })?;
                    VisualSource::Portrait { image }
                }
                VisualKind::Footage | VisualKind::Either => VisualSource::Footage {
                    frames: input.frames,
                },
            };
            // The one kind-gate enforcement point.
            if !self.animator.accepts().admits(&visual) {
                hanzo_ml::bail!(
                    "{:?} animator does not accept a {:?} visual source",
                    self.animator.accepts(),
                    visual.kind()
                );
            }
            let req = AnimationRequest {
                driving: DrivingAudio {
                    pcm: input.pcm,
                    sample_rate: input.sample_rate,
                },
                visual,
                fps: input.fps,
            };
            let out = self.animator.animate(&req)?;
            frames.push(Arc::new(out.frames));
            fps.push(out.fps);
        }
        Ok(ForwardInputsResult::Frames { frames, fps })
    }

    async fn sample_causal_gen(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: Vec<Tensor>,
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _srng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), hanzo_ml::Error> {
        hanzo_ml::bail!("`sample_causal_gen` is incompatible with `AnimationPipeline`");
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Animation
    }
}

impl AnyMoePipelineMixin for AnimationPipeline {}

const RESIZED_IMG: usize = 256;
// Self-contained MuseTalk bundle: all six sources resolve relative to one `model_id`
// (a local dir or HF repo). safetensors is the canonical weight format here.
const MUSETALK_UNET_CONFIG: &str = "musetalkV15/musetalk.json";
const MUSETALK_UNET_WEIGHTS: &str = "musetalkV15/unet.safetensors";
const VAE_CONFIG: &str = "sd-vae-ft-mse/config.json";
const VAE_WEIGHTS: &str = "sd-vae-ft-mse/diffusion_pytorch_model.safetensors";
const WHISPER_WEIGHTS: &str = "whisper/tiny.safetensors";
const S3FD_WEIGHTS: &str = "s3fd.safetensors";

// TAESD fast VAE path (madebyollin/taesd). Weights live next to the bundle under `taesd/` or at
// `DUB_TAESD_DIR`. Off by default until the LSE-C quality gate clears; then flip the default.
const TAESD_SUBDIR: &str = "taesd";
const TAESD_ENCODER_FILE: &str = "taesd_encoder.safetensors";
const TAESD_DECODER_FILE: &str = "taesd_decoder.safetensors";
const DUB_TAESD_DEFAULT: bool = false;

fn dub_taesd_enabled() -> bool {
    match std::env::var("DUB_TAESD") {
        Ok(v) => {
            let v = v.trim();
            !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
        }
        Err(_) => DUB_TAESD_DEFAULT,
    }
}

// MuseTalk core (VAE/UNet/TAESD) dtype. Default F32; DUB_TAESD realtime runs set f16. S3FD +
// whisper stay F32 (the animator casts their outputs to the core dtype), so this only governs conv.
fn dub_dtype() -> DType {
    match std::env::var("DUB_DTYPE").as_deref() {
        Ok("f16") | Ok("F16") | Ok("half") => DType::F16,
        Ok("bf16") | Ok("BF16") => DType::BF16,
        _ => DType::F32,
    }
}

#[derive(Clone, Debug)]
pub struct AnimationModelPaths {
    unet_config: PathBuf,
    unet_weights: PathBuf,
    vae_config: PathBuf,
    vae_weights: PathBuf,
    whisper_weights: PathBuf,
    s3fd_weights: PathBuf,
    all_weights: Vec<PathBuf>,
}

impl AnimationModelPaths {
    pub fn new(
        unet_config: PathBuf,
        unet_weights: PathBuf,
        vae_config: PathBuf,
        vae_weights: PathBuf,
        whisper_weights: PathBuf,
        s3fd_weights: PathBuf,
    ) -> Self {
        let all_weights = vec![
            unet_weights.clone(),
            vae_weights.clone(),
            whisper_weights.clone(),
            s3fd_weights.clone(),
        ];
        Self {
            unet_config,
            unet_weights,
            vae_config,
            vae_weights,
            whisper_weights,
            s3fd_weights,
            all_weights,
        }
    }
}

impl ModelPaths for AnimationModelPaths {
    fn get_config_filename(&self) -> &PathBuf {
        &self.unet_config
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        &self.all_weights
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_template_filename(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_chat_template_explicit(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_adapter_paths(&self) -> &AdapterPaths {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]> {
        unreachable!("Use `std::any::Any`.")
    }
}

/// Loads a `FacialAnimator`-backed `AnimationPipeline` from one MuseTalk bundle: the
/// four weight sources (UNet, VAE, whisper, S3FD) all resolve as fixed sub-paths of
/// `model_id`; the arch picks how they compose.
pub struct AnimationLoader {
    pub model_id: String,
    pub arch: AnimationLoaderType,
    pub options: AnimatorOptions,
}

impl AnimationLoader {
    fn load_vb(
        path: &PathBuf,
        dtype: DType,
        device: &Device,
        silent: bool,
    ) -> Result<hanzo_quant::ShardedVarBuilder> {
        // safetensors is canonical; a .pth/.pt pickle (upstream MuseTalk) still loads.
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
        {
            Ok(from_mmaped_safetensors(
                vec![path.clone()],
                Vec::new(),
                Some(dtype),
                device,
                vec![None],
                silent,
                None,
                |_| true,
                Arc::new(|_| DeviceForLoadTensor::Base),
            )?)
        } else {
            let pth = hanzo_ml::pickle::PthTensors::new(path, None)?;
            Ok(hanzo_quant::ShardedSafeTensors::wrap(
                Box::new(pth),
                dtype,
                device.clone(),
            ))
        }
    }
}

impl Loader for AnimationLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let revision = revision.unwrap_or_else(|| "main".to_string());
        let api = ApiBuilder::new()
            .with_progress(!silent)
            .with_token(get_token(&token_source)?)
            .build()?;
        let repo = api.repo(Repo::with_revision(
            self.model_id.clone(),
            RepoType::Model,
            revision.clone(),
        ));
        let id = std::path::Path::new(&self.model_id);

        let paths = AnimationModelPaths::new(
            api_get_file!(repo, MUSETALK_UNET_CONFIG, &id, &revision),
            api_get_file!(repo, MUSETALK_UNET_WEIGHTS, &id, &revision),
            api_get_file!(repo, VAE_CONFIG, &id, &revision),
            api_get_file!(repo, VAE_WEIGHTS, &id, &revision),
            api_get_file!(repo, WHISPER_WEIGHTS, &id, &revision),
            api_get_file!(repo, S3FD_WEIGHTS, &id, &revision),
        );

        self.load_model_from_path(
            &(Box::new(paths) as Box<dyn ModelPaths>),
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        _dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        // Reject only a real multi-device map; dummy/Auto are no-ops on a single-device animation model.
        if matches!(&mapper, DeviceMapSetting::Map(m) if m.device_layers().is_some_and(|l| !l.is_empty()))
        {
            anyhow::bail!("Device mapping is not supported for animation models.");
        }
        if in_situ_quant.is_some() {
            anyhow::bail!("ISQ is not supported for animation models.");
        }
        if paged_attn_config.is_some() {
            tracing::warn!("PagedAttention is not supported for animation models, disabling it.");
        }
        let paths = paths
            .as_ref()
            .as_any()
            .downcast_ref::<AnimationModelPaths>()
            .expect("Path downcast failed.");

        // MuseTalk core dtype (F32 default, f16 for realtime); S3FD + whisper stay F32 and the
        // animator bridges their outputs to the core dtype, so no mixed-dtype conv2d.
        let dtype = dub_dtype();
        const AUX_DTYPE: DType = DType::F32;

        let unet: UNetConfig = serde_json::from_str(&std::fs::read_to_string(&paths.unet_config)?)?;
        let vae: VaeConfig = serde_json::from_str(&std::fs::read_to_string(&paths.vae_config)?)?;
        let musetalk_config = MuseTalkConfig {
            unet,
            vae,
            resized_img: RESIZED_IMG,
        };

        let taesd_vb = if dub_taesd_enabled() {
            let dir = std::env::var("DUB_TAESD_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| std::path::Path::new(&self.model_id).join(TAESD_SUBDIR));
            let (enc, dec) = (dir.join(TAESD_ENCODER_FILE), dir.join(TAESD_DECODER_FILE));
            if !enc.exists() || !dec.exists() {
                anyhow::bail!(
                    "DUB_TAESD is set but TAESD weights are missing under {}. Place {} + {} there (source: madebyollin/taesd).",
                    dir.display(),
                    TAESD_ENCODER_FILE,
                    TAESD_DECODER_FILE
                );
            }
            Some((
                Self::load_vb(&enc, dtype, device, silent)?,
                Self::load_vb(&dec, dtype, device, silent)?,
            ))
        } else {
            None
        };

        let components = AnimationComponents::MuseTalk(MuseTalkComponents {
            musetalk_config,
            vae_vb: Self::load_vb(&paths.vae_weights, dtype, device, silent)?,
            unet_vb: Self::load_vb(&paths.unet_weights, dtype, device, silent)?,
            whisper_config: WhisperConfig::tiny(),
            whisper_vb: Self::load_vb(&paths.whisper_weights, AUX_DTYPE, device, silent)?,
            s3fd_vb: Self::load_vb(&paths.s3fd_weights, AUX_DTYPE, device, silent)?,
            taesd_vb,
            device: device.clone(),
            dtype,
            options: self.options,
        });

        let animator = animation_loader(&self.arch).load(components)?;
        Ok(Arc::new(Mutex::new(AnimationPipeline::new(
            animator,
            self.model_id.clone(),
            dtype,
        ))))
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::Normal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion_models::animation::AnimationOutput;

    // Trivial animator so the test exercises the pipeline's forward_inputs + kind-gate
    // (not MuseTalk numerics, which dub_e2e::animate_native covers). Emits ceil(secs*fps)
    // blank frames.
    struct MockAnimator {
        device: Device,
        accepts: VisualKind,
    }

    impl FacialAnimator for MockAnimator {
        fn animate(&mut self, req: &AnimationRequest) -> hanzo_ml::Result<AnimationOutput> {
            let secs = req.driving.pcm.len() as f64 / req.driving.sample_rate as f64;
            let n = ((secs * req.fps).ceil() as usize).max(1);
            Ok(AnimationOutput {
                frames: (0..n).map(|_| DynamicImage::new_rgb8(8, 8)).collect(),
                fps: req.fps,
            })
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn accepts(&self) -> VisualKind {
            self.accepts
        }
    }

    fn pipeline(accepts: VisualKind) -> AnimationPipeline {
        let animator = Box::new(MockAnimator {
            device: Device::Cpu,
            accepts,
        });
        AnimationPipeline::new(animator, "mock".to_string(), DType::F32)
    }

    fn footage_input(kind: VisualKind) -> ModelInputs {
        ModelInputs {
            requests: vec![AnimationInput {
                frames: vec![DynamicImage::new_rgb8(8, 8), DynamicImage::new_rgb8(8, 8)],
                pcm: Arc::new(vec![0f32; 24_000]), // 1 s @ 24 kHz
                sample_rate: 24_000,
                fps: 25.0,
                kind,
            }],
        }
    }

    #[test]
    fn forward_inputs_emits_frames() {
        let mut p = pipeline(VisualKind::Footage);
        let out = p
            .forward_inputs(Box::new(footage_input(VisualKind::Footage)), false)
            .unwrap();
        let ForwardInputsResult::Frames { frames, fps } = out else {
            panic!("expected Frames");
        };
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), 25); // ceil(1.0 * 25)
        assert_eq!(fps, vec![25.0]);
    }

    #[test]
    fn kind_gate_rejects_portrait_for_footage_animator() {
        let mut p = pipeline(VisualKind::Footage);
        assert!(p
            .forward_inputs(Box::new(footage_input(VisualKind::Portrait)), false)
            .is_err());
    }

    #[test]
    fn either_animator_accepts_both() {
        let mut p = pipeline(VisualKind::Either);
        assert!(p
            .forward_inputs(Box::new(footage_input(VisualKind::Footage)), false)
            .is_ok());
        let mut p = pipeline(VisualKind::Either);
        assert!(p
            .forward_inputs(Box::new(footage_input(VisualKind::Portrait)), false)
            .is_ok());
    }
}
