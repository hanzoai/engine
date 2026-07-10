//! ASR pipeline: audio -> text via [`Qwen3AsrPipeline`], served at
//! `/v1/audio/transcriptions`. Mirrors the TTS `SpeechLoader`/`SpeechPipeline`
//! loading shape and the `AnimationInputsProcessor` audio-gathering shape:
//! the driving audio rides on the sequence (`take_audios`), the optional
//! teacher-forced language rides on `asr_language`, and `forward_inputs` runs
//! greedy decode once per clip. Category is [`ModelCategory::Audio`].

use super::text_models_inputs_processor::PagedAttentionMeta;
use super::{
    AdapterPaths, AnyMoePipelineMixin, Cache, CacheManagerMixin, ChatTemplate, EitherCache,
    EmbeddingModulePaths, ForwardInputsResult, GeneralMetadata, InputProcessorOutput,
    InputsProcessor, InputsProcessorType, IsqPipelineMixin, Loader, MessagesAction, MetadataMixin,
    Modalities, ModelCategory, ModelKind, ModelPaths, PreProcessingMixin, Processor,
    SupportedModality, TokenSource,
};
use crate::device_map::{self, DeviceMapper};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::speech_models::{AsrLoaderType, Qwen3AsrConfig, Qwen3AsrPipeline};
use crate::utils::progress::ProgressScopeGuard;
use crate::utils::tokens::get_token;
use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
use crate::{
    api_dir_list, api_get_file, DeviceMapSetting, MessageContent, PagedAttentionConfig, Pipeline,
    TryIntoDType,
};
use anyhow::Result;
use hanzo_audio::AudioInput;
use hanzo_ml::{Device, Tensor};
use hanzo_quant::IsqType;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use indexmap::IndexMap;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

/// Build the Qwen3 ByteLevel-BPE tokenizer from a directory holding either a
/// prebuilt `tokenizer.json` or `vocab.json` + `merges.txt` + `tokenizer_config.json`.
/// Special tokens (`<|im_start|>`, `<|audio_pad|>`, `<asr_text>`, ...) are registered
/// from `added_tokens_decoder` so the prompt tokenizes to the same ids as transformers.
/// Params mirror the `qwen3_asr_e2e` HF cross-check.
fn load_asr_tokenizer(dir: &Path) -> Result<Tokenizer> {
    use tokenizers::{
        decoders::byte_level::ByteLevel as ByteLevelDec, models::bpe::BPE,
        pre_tokenizers::byte_level::ByteLevel as ByteLevelPre,
        processors::byte_level::ByteLevel as ByteLevelPost, AddedToken,
    };

    let tok_json = dir.join("tokenizer.json");
    if tok_json.exists() {
        return Tokenizer::from_file(&tok_json).map_err(anyhow::Error::msg);
    }

    let vocab = dir.join("vocab.json");
    let merges = dir.join("merges.txt");
    let bpe = BPE::from_file(vocab.to_str().unwrap(), merges.to_str().unwrap())
        .build()
        .map_err(anyhow::Error::msg)?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevelPre::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevelDec::new(false, false, false)));
    tokenizer.with_post_processor(Some(ByteLevelPost::new(false, false, false)));

    if let Ok(s) = std::fs::read_to_string(dir.join("tokenizer_config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(map) = v.get("added_tokens_decoder").and_then(|m| m.as_object()) {
                let added: Vec<AddedToken> = map
                    .values()
                    .filter_map(|x| x.get("content").and_then(|c| c.as_str()))
                    .map(|c| AddedToken::from(c.to_string(), true))
                    .collect();
                tokenizer.add_special_tokens(&added);
            }
        }
    }
    Ok(tokenizer)
}

#[derive(Clone, Debug)]
pub struct AsrModelPaths {
    weights: Vec<PathBuf>,
    config: PathBuf,
    /// Directory holding the tokenizer files (tokenizer.json or vocab.json+merges.txt).
    tokenizer_dir: PathBuf,
}

impl ModelPaths for AsrModelPaths {
    fn get_config_filename(&self) -> &PathBuf {
        &self.config
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        &self.weights
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

/// One sequence's worth of ASR input, gathered from the sequence's audio and
/// `asr_language` slots by [`AsrInputsProcessor`].
pub struct AsrInput {
    pub audio: AudioInput,
    pub language: Option<String>,
}

pub struct ModelInputs {
    pub requests: Vec<AsrInput>,
}

pub struct AsrProcessor;

impl Processor for AsrProcessor {
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
        anyhow::bail!("AsrProcessor::process does not expect chat messages.")
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(AsrInputsProcessor)
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

pub struct AsrInputsProcessor;

impl InputsProcessor for AsrInputsProcessor {
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
            let audio = seq
                .take_audios()
                .and_then(|a| a.into_iter().next())
                .ok_or_else(|| anyhow::anyhow!("transcription sequence missing audio"))?;
            let language = seq.asr_language().map(str::to_string);
            requests.push(AsrInput { audio, language });
        }
        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs { requests }),
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }
}

pub struct AsrPipeline {
    model_id: String,
    pipeline: Qwen3AsrPipeline,
    tokenizer: Arc<Tokenizer>,
    metadata: Arc<GeneralMetadata>,
    dummy_cache: EitherCache,
}

pub struct AsrLoader {
    pub model_id: String,
    pub arch: AsrLoaderType,
}

impl Loader for AsrLoader {
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
        let api = api.repo(Repo::with_revision(
            self.model_id.clone(),
            RepoType::Model,
            revision.clone(),
        ));
        let model_id = std::path::Path::new(&self.model_id);

        // One or more safetensors shards live at the repo root (no codec subdir).
        let files = api_dir_list!(api, model_id, false, &revision).collect::<Vec<_>>();
        let mut weights = Vec::new();
        for f in files.iter().filter(|f| f.ends_with(".safetensors")) {
            weights.push(api_get_file!(api, f, model_id, &revision));
        }
        if weights.is_empty() {
            anyhow::bail!("no `.safetensors` weights found for `{}`", self.model_id);
        }
        let config = api_get_file!(api, "config.json", model_id, &revision);
        // The tokenizer files (tokenizer.json or vocab.json+merges.txt) sit next to config.json.
        let tokenizer_dir = config.parent().unwrap().to_path_buf();

        let paths: Box<dyn ModelPaths> = Box::new(AsrModelPaths {
            weights,
            config,
            tokenizer_dir,
        });
        self.load_model_from_path(
            &paths,
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
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let paths = paths
            .as_ref()
            .as_any()
            .downcast_ref::<AsrModelPaths>()
            .expect("Path downcast failed.");

        if matches!(mapper, DeviceMapSetting::Map(_)) {
            anyhow::bail!("Device mapping is not supported for ASR models.")
        }
        if in_situ_quant.is_some() {
            anyhow::bail!("ISQ is not supported for ASR models.")
        }
        if paged_attn_config.is_some() {
            tracing::warn!("PagedAttention is not supported for ASR models, disabling it.");
        }

        let available_devices = device_map::get_all_similar_devices(device)?;
        let mapper =
            DeviceMapSetting::dummy().into_mapper(usize::MAX, device, None, &available_devices)?;
        let dtype = mapper.get_min_dtype(dtype)?;

        let cfg: Qwen3AsrConfig = serde_json::from_str(&std::fs::read_to_string(&paths.config)?)?;
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
        let pipeline = Qwen3AsrPipeline::new(&cfg, vb)?;
        let tokenizer = Arc::new(load_asr_tokenizer(&paths.tokenizer_dir)?);

        Ok(Arc::new(Mutex::new(AsrPipeline {
            model_id: self.model_id.clone(),
            pipeline,
            tokenizer,
            metadata: Arc::new(GeneralMetadata {
                max_seq_len: 1024,
                llg_factory: None,
                is_xlora: false,
                no_prefix_cache: false,
                num_hidden_layers: 1, // Only used for caching, which ASR does not do.
                eos_tok: vec![],
                kind: ModelKind::Normal,
                no_kv_cache: true,
                activation_dtype: dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![SupportedModality::Audio],
                    output: vec![SupportedModality::Text],
                },
            }),
            dummy_cache: EitherCache::Full(Cache::new(0, false)),
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::Normal
    }
}

impl PreProcessingMixin for AsrPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(AsrProcessor)
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for AsrPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!("ASR models do not support ISQ.")
    }
}

impl CacheManagerMixin for AsrPipeline {
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

impl MetadataMixin for AsrPipeline {
    fn device(&self) -> Device {
        self.pipeline.device().clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {}
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
}

#[async_trait::async_trait]
impl Pipeline for AsrPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> hanzo_ml::Result<ForwardInputsResult> {
        assert!(!return_raw_logits);

        let ModelInputs { requests } = *inputs.downcast().expect("Downcast failed.");
        let mut texts = Vec::with_capacity(requests.len());
        for input in requests {
            let text = self.pipeline.transcribe_with_language(
                &input.audio,
                &self.tokenizer,
                None,
                input.language.as_deref(),
                None,
            )?;
            texts.push(text);
        }
        Ok(ForwardInputsResult::Transcription { texts })
    }

    async fn sample_causal_gen(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: Vec<Tensor>,
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _srng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), hanzo_ml::Error> {
        hanzo_ml::bail!("`sample_causal_gen` is incompatible with `AsrPipeline`");
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Audio
    }
}

impl AnyMoePipelineMixin for AsrPipeline {}
