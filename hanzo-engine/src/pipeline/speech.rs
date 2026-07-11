use super::text_models_inputs_processor::PagedAttentionMeta;
use super::{
    AdapterPaths, AnyMoePipelineMixin, Cache, CacheManagerMixin, EitherCache, ForwardInputsResult,
    GeneralMetadata, InputProcessorOutput, InputsProcessor, InputsProcessorType, IsqPipelineMixin,
    Loader, MessagesAction, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, Processor, TokenSource,
};
use crate::device_map::{self, DeviceMapper};
use crate::diffusion_models::ace_step::pipeline::AceStepPipeline;
use crate::diffusion_models::ace_step::text_encoder::Umt5TextEncoder;
use crate::diffusion_models::ace_step::transformer::AceStepTransformer;
use crate::diffusion_models::ace_step::MusicDcae;
use crate::diffusion_models::t5::Config as T5Config;
use crate::distributed::{use_ring, WorkerTransferData};
use crate::pipeline::{ChatTemplate, EmbeddingModulePaths, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::speech_models::{
    DiaConfig, DiaPipeline, Qwen3TtsCodecConfig, Qwen3TtsConfig, Qwen3TtsPipeline,
    SpeechGenerationOutput, SpeechLoaderType, ACE_STEP_SAMPLE_RATE,
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

#[derive(Clone, Debug)]
pub struct SpeechModelPaths {
    weights: Vec<PathBuf>,
    config: PathBuf,
    /// For qwen3_tts: the speech_tokenizer (codec) config.json. None for Dia.
    codec_config: Option<PathBuf>,
    /// For qwen3_tts: the BPE tokenizer files (tokenizer.json or vocab.json+merges.txt dir).
    tokenizer_dir: Option<PathBuf>,
}

#[allow(clippy::large_enum_variant)]
enum SpeechModel {
    Dia(DiaPipeline),
    Qwen3Tts {
        pipeline: Qwen3TtsPipeline,
        tokenizer: Tokenizer,
    },
    AceStep {
        pipeline: AceStepPipeline,
        tokenizer: Tokenizer,
    },
}

/// Build the Qwen3 ByteLevel-BPE tokenizer from a directory containing `vocab.json` + `merges.txt`
/// + `tokenizer_config.json`. Special tokens (`<|im_start|>`, `<tts_pad>`, ...) are registered from
/// `added_tokens_decoder` so the assistant chat template tokenizes to the same ids as transformers.
#[allow(clippy::doc_lazy_continuation)]
fn load_qwen3_tts_tokenizer(dir: &std::path::Path) -> anyhow::Result<Tokenizer> {
    use tokenizers::{
        decoders::byte_level::ByteLevel as ByteLevelDecoder,
        models::bpe::BPE,
        pre_tokenizers::byte_level::ByteLevel,
        tokenizer::{AddedToken, Tokenizer},
    };

    // If a prebuilt tokenizer.json exists, prefer it.
    let tok_json = dir.join("tokenizer.json");
    if tok_json.exists() {
        return Tokenizer::from_file(&tok_json).map_err(anyhow::Error::msg);
    }

    let vocab = dir.join("vocab.json");
    let merges = dir.join("merges.txt");
    let bpe = BPE::from_file(vocab.to_str().unwrap(), merges.to_str().unwrap())
        .unk_token("<|endoftext|>".to_string())
        .build()
        .map_err(anyhow::Error::msg)?;

    let mut tokenizer = Tokenizer::new(bpe);
    // Qwen uses ByteLevel pre-tokenizer/decoder with add_prefix_space=false.
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::new(true, true, true)));

    // Register special tokens from tokenizer_config.json's added_tokens_decoder.
    let cfg_path = dir.join("tokenizer_config.json");
    if let Ok(s) = std::fs::read_to_string(&cfg_path) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(map) = v.get("added_tokens_decoder").and_then(|m| m.as_object()) {
                let mut added = Vec::new();
                for (_id, info) in map {
                    if let Some(content) = info.get("content").and_then(|c| c.as_str()) {
                        added.push(AddedToken::from(content.to_string(), true));
                    }
                }
                tokenizer.add_special_tokens(&added);
            }
        }
    }
    Ok(tokenizer)
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

pub struct SpeechPipeline {
    model_id: String,
    model: SpeechModel,
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
        let paths: anyhow::Result<Box<dyn ModelPaths>> = {
            // Main weights first, codec/DAC is the final one.
            let mut weights = Vec::new();
            let mut codec_config = None;
            let mut tokenizer_dir = None;

            // Dia/Qwen3Tts ship a single root `model.safetensors`; ACE-Step ships four
            // component checkpoints in sub-folders, so its files are fetched in the arm below.
            let config = if matches!(self.arch, SpeechLoaderType::AceStep) {
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

                // Order is load-critical: [umt5, dit, dcae, vocoder] (see load_model_from_path).
                let umt5 = api_get_file!(api, "umt5-base/model.safetensors", &model_id, &revision);
                let dit = api_get_file!(
                    api,
                    "ace_step_transformer/diffusion_pytorch_model.safetensors",
                    &model_id,
                    &revision
                );
                let dcae = api_get_file!(
                    api,
                    "music_dcae_f8c8/diffusion_pytorch_model.safetensors",
                    &model_id,
                    &revision
                );
                let vocoder = api_get_file!(
                    api,
                    "music_vocoder/diffusion_pytorch_model.safetensors",
                    &model_id,
                    &revision
                );
                let config = api_get_file!(api, "umt5-base/config.json", &model_id, &revision);
                let tok = api_get_file!(api, "umt5-base/tokenizer.json", &model_id, &revision);
                weights.push(umt5);
                weights.push(dit);
                weights.push(dcae);
                weights.push(vocoder);
                tokenizer_dir = Some(tok.parent().unwrap().to_path_buf());
                config
            } else {
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

            match self.arch {
                SpeechLoaderType::AceStep => {}
                SpeechLoaderType::Dia => {
                    // DAC model lives in a separate repo.
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
                SpeechLoaderType::Qwen3Tts => {
                    // The codec ("speech_tokenizer") + BPE tokenizer live in the SAME repo.
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
                    let codec_weight = api_get_file!(
                        api,
                        "speech_tokenizer/model.safetensors",
                        &model_id,
                        &revision
                    );
                    let codec_cfg =
                        api_get_file!(api, "speech_tokenizer/config.json", &model_id, &revision);
                    let vocab = api_get_file!(api, "vocab.json", &model_id, &revision);
                    let _merges = api_get_file!(api, "merges.txt", &model_id, &revision);
                    let _tok_cfg =
                        api_get_file!(api, "tokenizer_config.json", &model_id, &revision);
                    weights.push(codec_weight);
                    codec_config = Some(codec_cfg);
                    tokenizer_dir = Some(vocab.parent().unwrap().to_path_buf());
                }
            }

            Ok(Box::new(SpeechModelPaths {
                weights,
                config,
                codec_config,
                tokenizer_dir,
            }))
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
        crate::utils::cuda_mempool::set_pool_retain_all(device)?;
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

        let load_component = |files: Vec<PathBuf>, ld: hanzo_ml::DType, keep: fn(&str) -> bool| {
            from_mmaped_safetensors(
                files,
                Vec::new(),
                Some(ld),
                device,
                vec![None],
                silent,
                None,
                move |n: String| keep(&n),
                Arc::new(|_| DeviceForLoadTensor::Base),
            )
        };
        // Dia/Qwen3Tts: all but the last weight (last = DAC/speech_tokenizer codec).
        let main_weights = || paths.weights[..paths.weights.len() - 1].to_vec();

        let model = match self.arch {
            SpeechLoaderType::Dia => {
                let vb = load_component(main_weights(), dtype, |_| true)?;
                let cfg: DiaConfig =
                    serde_json::from_str(&std::fs::read_to_string(&paths.config)?)?;
                let dac_vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[paths.weights.last().unwrap()],
                        dtype,
                        device,
                    )?
                };
                let pipeline = DiaPipeline::new(&cfg, vb, dac_vb)?;
                SpeechModel::Dia(pipeline)
            }
            SpeechLoaderType::Qwen3Tts => {
                let vb = load_component(main_weights(), dtype, |_| true)?;
                let cfg: Qwen3TtsConfig =
                    serde_json::from_str(&std::fs::read_to_string(&paths.config)?)?;
                let codec_cfg_path = paths
                    .codec_config
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("qwen3_tts requires speech_tokenizer config"))?;
                let codec_cfg: Qwen3TtsCodecConfig =
                    serde_json::from_str(&std::fs::read_to_string(codec_cfg_path)?)?;
                // The codec decoder ships f32 weights; load it in f32 for numerical fidelity.
                let codec_vb = from_mmaped_safetensors(
                    vec![paths.weights.last().unwrap().clone()],
                    Vec::new(),
                    Some(hanzo_ml::DType::F32),
                    device,
                    vec![None],
                    silent,
                    None,
                    |_| true,
                    Arc::new(|_| DeviceForLoadTensor::Base),
                )?;
                let tokenizer = load_qwen3_tts_tokenizer(
                    paths
                        .tokenizer_dir
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("qwen3_tts requires a tokenizer"))?,
                )?;
                let pipeline = Qwen3TtsPipeline::new(&cfg, &codec_cfg, vb, codec_vb)?;
                SpeechModel::Qwen3Tts {
                    pipeline,
                    tokenizer,
                }
            }
            SpeechLoaderType::AceStep => {
                // umt5/DCAE/vocoder stay f32 for fidelity; weights = [umt5, dit, dcae, vocoder].
                let f32 = hanzo_ml::DType::F32;
                let umt5_vb = load_component(vec![paths.weights[0].clone()], f32, |_| true)?;
                // DiT runs bf16 for Blackwell tensor cores (2x/forward, spectrogram-identical audio);
                // f16 is unusable here, the 12800-wide GLU overflows its range to NaN. F32 islands for
                // the RMS variance, LiteLA normaliser and RoPE phase live in AceStepTransformer.
                let dit_vb = load_component(vec![paths.weights[1].clone()], hanzo_ml::DType::BF16, |n| {
                    !n.contains(".add_")
                        && !n.contains(".to_add_out")
                        && !n.starts_with("lyric")
                        && !n.starts_with("projectors")
                })?;
                let dcae_vb = load_component(vec![paths.weights[2].clone()], f32, |n| {
                    n.starts_with("decoder.")
                })?;
                let voc_vb = load_component(vec![paths.weights[3].clone()], f32, |_| true)?;

                let mut cfg: T5Config =
                    serde_json::from_str(&std::fs::read_to_string(&paths.config)?)?;
                cfg.umt5 = true;
                let text_encoder = Umt5TextEncoder::new(&cfg, umt5_vb, device)?;
                let transformer = AceStepTransformer::new(dit_vb)?;
                let dcae = MusicDcae::new(dcae_vb, voc_vb)?;
                let pipeline =
                    AceStepPipeline::new(text_encoder, transformer, dcae, device.clone());

                let tok_path = paths
                    .tokenizer_dir
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("ace_step requires a tokenizer"))?
                    .join("tokenizer.json");
                let tokenizer = Tokenizer::from_file(&tok_path).map_err(anyhow::Error::msg)?;
                SpeechModel::AceStep {
                    pipeline,
                    tokenizer,
                }
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
        match &self.model {
            SpeechModel::Dia(m) => m.device().clone(),
            SpeechModel::Qwen3Tts { pipeline, .. } => pipeline.device().clone(),
            SpeechModel::AceStep { pipeline, .. } => pipeline.device().clone(),
        }
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
        let cfg = self.cfg;
        let mut pcms = Vec::new();
        let mut rates = Vec::new();
        let mut channels_all = Vec::new();
        for prompt in prompts {
            let SpeechGenerationOutput {
                pcm,
                rate,
                channels,
            } = match &mut self.model {
                SpeechModel::Dia(m) => m.generate(&prompt, &cfg)?,
                SpeechModel::Qwen3Tts {
                    pipeline,
                    tokenizer,
                } => {
                    // Apply the assistant chat template, tokenize, and pass the id stream through.
                    let templated = format!(
                        "<|im_start|>assistant\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    );
                    let enc = tokenizer
                        .encode_fast(templated, false)
                        .map_err(hanzo_ml::Error::msg)?;
                    let id_stream = enc
                        .get_ids()
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(" ");
                    pipeline.generate(&id_stream, &cfg)?
                }
                SpeechModel::AceStep {
                    pipeline,
                    tokenizer,
                } => {
                    let SpeechGenerationConfig::AceStep {
                        frames,
                        steps,
                        guidance_scale,
                    } = cfg
                    else {
                        hanzo_ml::bail!("ace_step requires SpeechGenerationConfig::AceStep");
                    };
                    let enc = tokenizer
                        .encode_fast(prompt.as_str(), true)
                        .map_err(hanzo_ml::Error::msg)?;
                    let ids = enc.get_ids().to_vec();
                    let n = ids.len();
                    let dev = pipeline.device().clone();
                    let input_ids = Tensor::from_vec(ids, (1, n), &dev)?;
                    let wav = pipeline.generate(&input_ids, frames, steps, guidance_scale)?;
                    // (1, C, S) planar -> interleaved [L0, R0, L1, R1, ...] PCM.
                    let planar = wav.narrow(0, 0, 1)?.squeeze(0)?;
                    let (chans, samples) = planar.dims2()?;
                    let per_ch: Vec<Vec<f32>> = (0..chans)
                        .map(|c| planar.narrow(0, c, 1)?.squeeze(0)?.to_vec1::<f32>())
                        .collect::<hanzo_ml::Result<_>>()?;
                    let mut pcm = vec![0f32; chans * samples];
                    for (c, ch) in per_ch.iter().enumerate() {
                        for (i, &s) in ch.iter().enumerate() {
                            pcm[i * chans + c] = s;
                        }
                    }
                    SpeechGenerationOutput {
                        pcm: Arc::new(pcm),
                        rate: ACE_STEP_SAMPLE_RATE,
                        channels: chans,
                    }
                }
            };
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
