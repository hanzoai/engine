use super::text_models_inputs_processor::PagedAttentionMeta;
use super::{
    AdapterPaths, AnyMoePipelineMixin, Cache, CacheManagerMixin, EitherCache, ForwardInputsResult,
    GeneralMetadata, InputProcessorOutput, InputsProcessor, InputsProcessorType, IsqPipelineMixin,
    Loader, MessagesAction, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, Processor, TokenSource,
};
use crate::device_map::{self, DeviceMapper};
use crate::distributed::{use_ring, WorkerTransferData};
use crate::pipeline::{ChatTemplate, EmbeddingModulePaths, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::speech_models::{
    DiaConfig, DiaPipeline, Qwen3TtsCodecConfig, Qwen3TtsConfig, Qwen3TtsPipeline,
    SpeechGenerationOutput, SpeechLoaderType,
};
use crate::utils::progress::ProgressScopeGuard;
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::utils::{tokens::get_token, varbuilder_utils::from_mmaped_safetensors};
use crate::{
    api_get_file, distributed, DeviceMapSetting, MessageContent, PagedAttentionConfig, Pipeline,
    SpeechGenerationConfig, TryIntoDType,
};
use anyhow::Result;
use hanzo_ml::{Device, Tensor};
use hanzo_nn::VarBuilder;
use hanzo_quant::IsqType;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use indexmap::IndexMap;
use rand_isaac::Isaac64Rng;
use regex::Regex;
use std::any::Any;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

// The zen3-tts repos ship a byte-level BPE as vocab.json + merges.txt (Qwen tokenizer), with no
// tokenizer.json. Reconstruct an equivalent ByteLevel BPE tokenizer from those two files.
fn build_qwen_bpe_tokenizer(
    vocab: &std::path::Path,
    merges: &std::path::Path,
) -> anyhow::Result<Tokenizer> {
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    };
    use tokenizers::tokenizer::normalizer::SplitDelimiterBehavior;

    let bpe = BPE::from_file(
        vocab.to_str().expect("vocab path"),
        merges.to_str().expect("merges path"),
    )
    .build()
    .map_err(anyhow::Error::msg)?;
    let mut tok = Tokenizer::new(bpe);

    // Qwen byte-level BPE pre-tokenizer: GPT-style split regex then ByteLevel mapping.
    let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let split = Split::new(
        SplitPattern::Regex(pattern.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(anyhow::Error::msg)?;
    let pre = Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ]);
    tok.with_pre_tokenizer(Some(pre));
    tok.with_decoder(Some(ByteLevelDecoder::new(false, false, false)));
    Ok(tok)
}

#[derive(Clone, Debug)]
pub struct SpeechModelPaths {
    weights: Vec<PathBuf>,
    config: PathBuf,
    // Qwen3-TTS extras: codec (speech_tokenizer) config + weights, and the byte-level BPE tokenizer
    // (vocab.json + merges.txt; the repo ships no tokenizer.json).
    codec_config: Option<PathBuf>,
    codec_weights: Vec<PathBuf>,
    vocab: Option<PathBuf>,
    merges: Option<PathBuf>,
}

impl ModelPaths for SpeechModelPaths {
    fn get_config_filename(&self) -> &PathBuf {
        &self.config
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        &self.weights
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

pub struct SpeechProcessor;

impl Processor for SpeechProcessor {
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
        anyhow::bail!(
            "SpeechProcessor::process should not be used. It does not expect chat messages."
        )
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(SpeechInputsProcessor)
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        // Just a default
        MessagesAction::FlattenOnlyText
    }
}

pub struct SpeechInputsProcessor;

#[derive(Clone)]
pub struct ModelInputs {
    pub(crate) prompts: Vec<String>,
}

impl InputsProcessor for SpeechInputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Text
    }

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
        let inputs = ModelInputs {
            prompts: input_seqs
                .iter()
                .map(|seq| seq.get_initial_prompt().to_string())
                .collect(),
        };
        Ok(InputProcessorOutput {
            inputs: Box::new(inputs),
            seq_indices: (0..input_seqs.len()).collect::<Vec<_>>(),
        })
    }
}

enum SpeechModelInner {
    Dia(Box<DiaPipeline>),
    Qwen3Tts(Box<Qwen3TtsPipeline>),
}

impl SpeechModelInner {
    fn generate(
        &self,
        prompt: &str,
        cfg: &SpeechGenerationConfig,
    ) -> hanzo_ml::Result<SpeechGenerationOutput> {
        match self {
            SpeechModelInner::Dia(m) => m.generate(prompt, cfg),
            SpeechModelInner::Qwen3Tts(m) => {
                let max_frames = match cfg {
                    SpeechGenerationConfig::Qwen3Tts { max_frames } => *max_frames,
                    _ => None,
                };
                m.generate(prompt, max_frames)
            }
        }
    }

    fn device(&self) -> &Device {
        match self {
            SpeechModelInner::Dia(m) => m.device(),
            SpeechModelInner::Qwen3Tts(m) => m.device(),
        }
    }
}

pub struct SpeechPipeline {
    model_id: String,
    model: SpeechModelInner,
    metadata: Arc<GeneralMetadata>,
    dummy_cache: EitherCache,
    cfg: SpeechGenerationConfig,
}

pub struct SpeechLoader {
    pub model_id: String,
    pub dac_model_id: Option<String>,
    pub arch: SpeechLoaderType,
    pub cfg: Option<SpeechGenerationConfig>,
}

impl Loader for SpeechLoader {
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
        let paths: anyhow::Result<Box<dyn ModelPaths>> = match self.arch {
            SpeechLoaderType::Dia => {
                // Main weights first, DAC is the final one.
                let mut weights = Vec::new();

                let config = {
                    let api = ApiBuilder::new()
                        .with_progress(!silent)
                        .with_token(get_token(&token_source)?)
                        .build()?;
                    let revision = revision.clone().unwrap_or("main".to_string());
                    let api = api.repo(Repo::with_revision(
                        self.model_id.to_string(),
                        RepoType::Model,
                        revision.clone(),
                    ));
                    let model_id = std::path::Path::new(&self.model_id);

                    let weight = api_get_file!(api, "model.safetensors", &model_id, &revision);
                    let config = api_get_file!(api, "config.json", &model_id, &revision);
                    weights.push(weight);
                    config
                };

                {
                    let api = ApiBuilder::new()
                        .with_progress(!silent)
                        .with_token(get_token(&token_source)?)
                        .build()?;
                    let revision = revision.unwrap_or("main".to_string());

                    let dac_model = self
                        .dac_model_id
                        .clone()
                        .unwrap_or_else(|| "hanzoai/dac_44khz".to_string());

                    let api = api.repo(Repo::with_revision(
                        dac_model.clone(),
                        RepoType::Model,
                        revision.clone(),
                    ));
                    let model_id = std::path::Path::new(&dac_model);

                    let weight = api_get_file!(api, "model.safetensors", &model_id, &revision);
                    weights.push(weight);
                }

                Ok(Box::new(SpeechModelPaths {
                    weights,
                    config,
                    codec_config: None,
                    codec_weights: Vec::new(),
                    vocab: None,
                    merges: None,
                }))
            }
            SpeechLoaderType::Qwen3Tts => {
                let api = ApiBuilder::new()
                    .with_progress(!silent)
                    .with_token(get_token(&token_source)?)
                    .build()?;
                let revision = revision.unwrap_or("main".to_string());
                let api = api.repo(Repo::with_revision(
                    self.model_id.to_string(),
                    RepoType::Model,
                    revision.clone(),
                ));
                let model_id = std::path::Path::new(&self.model_id);

                let config = api_get_file!(api, "config.json", &model_id, &revision);
                let weight = api_get_file!(api, "model.safetensors", &model_id, &revision);
                let codec_config =
                    api_get_file!(api, "speech_tokenizer/config.json", &model_id, &revision);
                let codec_weight = api_get_file!(
                    api,
                    "speech_tokenizer/model.safetensors",
                    &model_id,
                    &revision
                );
                let vocab = api_get_file!(api, "vocab.json", &model_id, &revision);
                let merges = api_get_file!(api, "merges.txt", &model_id, &revision);

                Ok(Box::new(SpeechModelPaths {
                    weights: vec![weight],
                    config,
                    codec_config: Some(codec_config),
                    codec_weights: vec![codec_weight],
                    vocab: Some(vocab),
                    merges: Some(merges),
                }))
            }
        };
        self.load_model_from_path(
            &paths?,
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
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        _paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let paths = &paths
            .as_ref()
            .as_any()
            .downcast_ref::<SpeechModelPaths>()
            .expect("Path downcast failed.");

        if matches!(mapper, DeviceMapSetting::Map(_)) {
            anyhow::bail!("Device mapping is not supported for speech models.")
        }

        hanzo_quant::set_immediate_isq(in_situ_quant, vec![Regex::new(".*")?]);

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = &device {
            unsafe { dev.disable_event_tracking() };
        }
        let use_nccl = hanzo_quant::distributed::use_nccl();
        let available_devices = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
            let payload: WorkerTransferData = serde_json::from_str(&payload)?;
            let WorkerTransferData::Init { id: _, worker_rank } = payload;
            vec![hanzo_ml::Device::new_cuda(worker_rank + 1)?]
        } else if use_nccl || use_ring() {
            vec![hanzo_ml::Device::new_cuda(0)?]
        } else {
            device_map::get_all_similar_devices(device)?
        };

        let mapper =
            DeviceMapSetting::dummy().into_mapper(usize::MAX, device, None, &available_devices)?;
        let dtype = mapper.get_min_dtype(dtype)?;

        let model = match self.arch {
            SpeechLoaderType::Dia => {
                let cfg: DiaConfig =
                    serde_json::from_str(&std::fs::read_to_string(&paths.config)?)?;
                // Last weight is the dac.
                let model_weights = paths.weights[..paths.weights.len() - 1].to_vec();
                let vb = from_mmaped_safetensors(
                    model_weights,
                    Vec::new(),
                    Some(dtype),
                    device,
                    vec![None],
                    silent,
                    None,
                    |_| true,
                    Arc::new(|_| DeviceForLoadTensor::Base),
                )?;
                let dac_vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[paths.weights.last().unwrap()],
                        dtype,
                        device,
                    )?
                };
                SpeechModelInner::Dia(Box::new(DiaPipeline::new(&cfg, vb, dac_vb)?))
            }
            SpeechLoaderType::Qwen3Tts => {
                let cfg: Qwen3TtsConfig =
                    serde_json::from_str(&std::fs::read_to_string(&paths.config)?)?;
                let codec_cfg: Qwen3TtsCodecConfig = serde_json::from_str(
                    &std::fs::read_to_string(paths.codec_config.as_ref().expect("codec config"))?,
                )?;
                let vb = from_mmaped_safetensors(
                    paths.weights.clone(),
                    Vec::new(),
                    Some(dtype),
                    device,
                    vec![None],
                    silent,
                    None,
                    |_| true,
                    Arc::new(|_| DeviceForLoadTensor::Base),
                )?;
                // Codec stays in F32 (matches the checkpoint dtype and the conv-heavy vocoder).
                let codec_vb = from_mmaped_safetensors(
                    paths.codec_weights.clone(),
                    Vec::new(),
                    Some(hanzo_ml::DType::F32),
                    device,
                    vec![None],
                    silent,
                    None,
                    |_| true,
                    Arc::new(|_| DeviceForLoadTensor::Base),
                )?;
                let tokenizer = build_qwen_bpe_tokenizer(
                    paths.vocab.as_ref().expect("vocab.json"),
                    paths.merges.as_ref().expect("merges.txt"),
                )?;
                SpeechModelInner::Qwen3Tts(Box::new(Qwen3TtsPipeline::new(
                    &cfg, &codec_cfg, vb, codec_vb, tokenizer, dtype, device,
                )?))
            }
        };

        Ok(Arc::new(Mutex::new(SpeechPipeline {
            model_id: self.model_id.clone(),
            model,
            metadata: Arc::new(GeneralMetadata {
                max_seq_len: 1024,
                llg_factory: None,
                is_xlora: false,
                no_prefix_cache: false,
                num_hidden_layers: 1, // FIXME(hanzoai): we know this is only for caching, so its OK.
                eos_tok: vec![],
                kind: ModelKind::Normal,
                no_kv_cache: true, // NOTE(hanzoai): no cache for these.
                activation_dtype: dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Audio],
                },
            }),
            dummy_cache: EitherCache::Full(Cache::new(0, false)),
            cfg: self
                .cfg
                .unwrap_or_else(|| SpeechGenerationConfig::default(self.arch)),
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::Normal
    }
}

impl PreProcessingMixin for SpeechPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(SpeechProcessor)
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for SpeechPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!("Speech models do not support ISQ for now.")
    }
}

impl CacheManagerMixin for SpeechPipeline {
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

impl MetadataMixin for SpeechPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
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
impl Pipeline for SpeechPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> hanzo_ml::Result<ForwardInputsResult> {
        assert!(!return_raw_logits);

        let ModelInputs { prompts } = *inputs.downcast().expect("Downcast failed.");
        let mut pcms = Vec::new();
        let mut rates = Vec::new();
        let mut channels_all = Vec::new();
        for prompt in prompts {
            let SpeechGenerationOutput {
                pcm,
                rate,
                channels,
            } = self.model.generate(&prompt, &self.cfg)?;
            pcms.push(pcm);
            rates.push(rate);
            channels_all.push(channels);
        }

        Ok(ForwardInputsResult::Speech {
            pcms,
            rates,
            channels: channels_all,
        })
    }

    async fn sample_causal_gen(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: Vec<Tensor>,
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _srng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), hanzo_ml::Error> {
        hanzo_ml::bail!("`sample_causal_gen` is incompatible with `SpeechPipeline`");
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Speech
    }
}

impl AnyMoePipelineMixin for SpeechPipeline {}
