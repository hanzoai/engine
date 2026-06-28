use super::text_models_inputs_processor::PagedAttentionMeta;
use super::{
    AnyMoePipelineMixin, Cache, CacheManagerMixin, ChatTemplate, EitherCache, ForwardInputsResult,
    GeneralMetadata, InputProcessorOutput, InputsProcessor, InputsProcessorType, IsqPipelineMixin,
    MessagesAction, MetadataMixin, Modalities, ModelCategory, ModelKind, PreProcessingMixin,
    Processor, SupportedModality,
};
use crate::device_map::DeviceMapper;
use crate::diffusion_models::animation::{
    AnimationRequest, DrivingAudio, FacialAnimator, VisualKind, VisualSource, OMNI_SAMPLE_RATE,
};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::{MessageContent, Pipeline};
use anyhow::Result;
use hanzo_ml::{DType, Device, Tensor};
use image::DynamicImage;
use indexmap::IndexMap;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::sync::Arc;
use tokenizers::Tokenizer;

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
                VisualKind::Footage | VisualKind::Either => {
                    VisualSource::Footage {
                        frames: input.frames,
                    }
                }
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
