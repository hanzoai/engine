use super::llg::build_llg_factory;
use super::{
    get_model_paths, get_xlora_paths, text_models_inputs_processor::ModelInputs, AdapterKind,
    CacheManager, GeneralMetadata, Loader, ModelKind, ModelPaths, PrettyName, QuantizationKind,
    TokenSource,
};
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, IsqPipelineMixin,
    MetadataMixin, ModelCategory, PreProcessingMixin,
};
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::{self, DeviceMapper};
use crate::distributed::WorkerTransferData;
use crate::gguf::{
    get_gguf_chat_template, {convert_gguf_to_hf_tokenizer, GgufTokenizerConversion},
};
use crate::gguf::{Content, GGUFArchitecture};
use crate::kv_cache::{FullCacheManager, HybridCacheManager, NormalCacheManager};
use crate::lora::Ordering;
use crate::paged_attention::{
    calculate_cache_config, AttentionImplementation, CacheEngine, ModelConfigLike,
};
use crate::pipeline::chat_template::{calculate_eos_tokens, BeginEndUnkPadTok, GenerationConfig};
use crate::pipeline::loaders::DeviceMappedModelLoader;
#[cfg(feature = "rocm")]
use crate::pipeline::rocm_graph::{
    rocm_decode_graphs_enabled, RocmDecodeGraphKey, RocmDecodeGraphMetadataBuffers,
    RocmGraphHandle, ROCM_DECODE_GRAPH_CACHE_CAPACITY,
};
use crate::pipeline::sampling::sample_and_add_toks;
#[cfg(feature = "rocm")]
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::ChatTemplate;
use crate::pipeline::{get_chat_template, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::gguf_metadata::{ContentConfig, GgufDeviceMapLoaderInner};
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::ProgressScopeGuard;
use crate::utils::tokenizer::get_tokenizer;
use crate::xlora_models::NonGranularState;
use crate::{
    distributed, get_mut_arcmutex, get_paths_gguf, DeviceMapSetting, LocalModelPaths,
    PagedAttentionConfig, Pipeline, Topology, TryIntoDType,
};
use crate::{
    models::quantized_deepseek2::ModelWeights as QDeepSeek2,
    models::quantized_llama::ModelWeights as QLlama,
    models::quantized_phi2::ModelWeights as QPhi,
    models::quantized_phi3::ModelWeights as QPhi3,
    models::quantized_qwen::ModelWeights as QQwen,
    models::quantized_qwen3::ModelWeights as QQwen3,
    models::quantized_qwen3_5_moe::ModelWeights as QQwen35,
    models::quantized_qwen3_moe::ModelWeights as QQwen3MoE,
    models::quantized_starcoder2::ModelWeights as QStarcoder2,
    utils::tokens::get_token,
    xlora_models::{XLoraQLlama, XLoraQPhi3},
};
use anyhow::{bail, Result};
use either::Either;
#[cfg(feature = "rocm")]
use hanzo_ml::Var;
use hanzo_ml::{Device, Tensor};
use hanzo_quant::IsqType;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::{env, fs};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

enum Model {
    Llama(QLlama),
    Phi2(QPhi),
    XLoraLlama(XLoraQLlama),
    XLoraPhi3(XLoraQPhi3),
    Phi3(QPhi3),
    Starcoder2(QStarcoder2),
    Qwen(QQwen),
    Qwen3(QQwen3),
    Qwen3MoE(QQwen3MoE),
    Qwen35(QQwen35),
    Deepseek2(QDeepSeek2),
}

pub struct GGUFPipeline {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    non_granular_state: Option<NonGranularState>,
    metadata: Arc<GeneralMetadata>,
    generation_defaults: Option<crate::ModelGenerationDefaults>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    draft_proposer: Option<crate::speculative::DraftModelProposer>,
    /// Captured ROCm/HIP decode graphs, keyed by decode bucket. See
    /// [`crate::pipeline::rocm_graph`]. Mirrors `NormalPipeline::cuda_decode_graph`.
    #[cfg(feature = "rocm")]
    rocm_decode_graph: std::sync::Mutex<RocmDecodeGraphState>,
}

#[cfg(feature = "rocm")]
#[derive(Default)]
struct RocmDecodeGraphState {
    entries: Vec<RocmDecodeGraphEntry>,
    disabled: bool,
}

#[cfg(feature = "rocm")]
struct RocmDecodeGraphEntry {
    key: RocmDecodeGraphKey,
    graph: RocmGraphHandle,
    /// Stable input-ids buffer the captured `tok_embeddings` reads; refreshed in
    /// place each replay.
    input_ids: Var,
    metadata_buffers: RocmDecodeGraphMetadataBuffers,
    /// Owns the rebound metadata referencing the stable device buffers so the
    /// cached tensors outlive the entry (mirrors the CUDA `_metadata` field).
    _metadata: PagedAttentionInputMetadata,
    /// The warmup logits tensor. Because the graph writes its output into this
    /// tensor's (stable) storage every replay, cloning it after a launch yields
    /// the current token's logits.
    logits: Tensor,
}

/// Loader for a GGUF model.
pub struct GGUFLoader {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    kind: ModelKind,
    tgt_non_granular_index: Option<usize>,
    config: GGUFSpecificConfig,
    jinja_explicit: Option<String>,
    lora_adapter_ids: Option<Vec<String>>,
}

#[derive(Clone, Default)]
/// Config for a GGUF loader.
pub struct GGUFSpecificConfig {
    pub topology: Option<Topology>,
}

#[derive(Default)]
/// A builder for a GGUF loader.
pub struct GGUFLoaderBuilder {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tgt_non_granular_index: Option<usize>,
    config: GGUFSpecificConfig,
    jinja_explicit: Option<String>,
}

impl GGUFLoaderBuilder {
    /// Create a loader builder for a GGUF model. `tok_model_id` is the model ID where you can find a
    /// `tokenizer_config.json` file. If the `chat_template` is specified, then it will be treated as a
    /// path and used over remote files, removing all remote accesses.
    pub fn new(
        chat_template: Option<String>,
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        config: GGUFSpecificConfig,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        let kind = ModelKind::GgufQuantized {
            quant: QuantizationKind::Gguf,
        };

        Self {
            chat_template,
            model_id: tok_model_id,
            kind,
            quantized_filenames,
            quantized_model_id,
            config,
            jinja_explicit,
            no_kv_cache,
            ..Default::default()
        }
    }

    fn with_adapter(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.xlora_model_id = Some(xlora_model_id);
        self.xlora_order = Some(xlora_order);
        self.no_kv_cache = no_kv_cache;
        self.tgt_non_granular_index = tgt_non_granular_index;
        self.model_id = if let Some(id) = self.model_id {
            Some(id)
        } else {
            info!(
                "Using adapter base model ID: `{}`",
                self.xlora_order.as_ref().unwrap().base_model_id
            );
            Some(self.xlora_order.as_ref().unwrap().base_model_id.clone())
        };
        self
    }

    pub fn with_xlora(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.kind = (AdapterKind::XLora, QuantizationKind::Gguf).into();

        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(mut self, lora_model_id: String, lora_order: Ordering) -> Self {
        self.kind = (AdapterKind::Lora, QuantizationKind::Gguf).into();

        self.with_adapter(lora_model_id, lora_order, false, None)
    }

    pub fn build(self) -> Box<dyn Loader> {
        Box::new(GGUFLoader {
            model_id: self.model_id,
            xlora_model_id: self.xlora_model_id,
            kind: self.kind,
            xlora_order: self.xlora_order,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tgt_non_granular_index: self.tgt_non_granular_index,
            quantized_filenames: self.quantized_filenames,
            quantized_model_id: self.quantized_model_id,
            config: self.config,
            jinja_explicit: self.jinja_explicit,
            lora_adapter_ids: None,
        })
    }
}

impl GGUFLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        xlora_model_id: Option<String>,
        kind: ModelKind,
        xlora_order: Option<Ordering>,
        no_kv_cache: bool,
        chat_template: Option<String>,
        tgt_non_granular_index: Option<usize>,
        config: GGUFSpecificConfig,
        jinja_explicit: Option<String>,
    ) -> Self {
        let model_id = if let Some(id) = model_id {
            Some(id)
        } else if let Some(xlora_order) = xlora_order.clone() {
            info!(
                "Using adapter base model ID: `{}`",
                xlora_order.base_model_id
            );
            Some(xlora_order.base_model_id.clone())
        } else {
            None
        };
        Self {
            model_id,
            quantized_model_id,
            quantized_filenames,
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            chat_template,
            kind,
            tgt_non_granular_index,
            config,
            jinja_explicit,
            lora_adapter_ids: None,
        }
    }
}

impl Loader for GGUFLoader {
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
        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths_gguf!(
            LocalModelPaths,
            &token_source,
            revision,
            self,
            self.quantized_model_id.clone(),
            self.quantized_filenames.clone(),
            silent
        );

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
        mut mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        if in_situ_quant.is_some() {
            anyhow::bail!(
                "You are trying to in-situ quantize a GGUF model. This will not do anything."
            );
        }

        debug!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        let mut readers = Vec::new();
        for filename in paths.get_weight_filenames() {
            readers.push(std::fs::File::open(filename)?);
        }
        let mut readers = readers.iter_mut().collect::<Vec<_>>();
        let model = Content::from_readers(&mut readers)?;

        if !silent {
            model.print_metadata()?;
        }

        let arch = model.arch();

        // If auto, convert to Map
        let num_layers = model.get_metadata()[&format!("{arch}.block_count")].to_u32()? as usize;

        let mut max_kv_tokens: Option<usize> = None;

        if let DeviceMapSetting::Auto(params) = mapper.clone() {
            let devices = device_map::get_all_similar_devices(device)?;
            // Initial dtype
            let dtype = dtype.try_into_dtype(&devices.iter().collect::<Vec<_>>())?;

            let model = GgufDeviceMapLoaderInner {
                model: &model,
                arch,
            };

            let layer_sizes_in_bytes =
                model.layer_sizes_in_bytes("this is a dummy config!", dtype, 1, None)?;
            let non_mapped_size_in_bytes =
                model.non_mapped_size_in_bytes("this is a dummy config!", dtype, 1, None)?;
            let total_model_size_in_bytes =
                layer_sizes_in_bytes.iter().sum::<usize>() + non_mapped_size_in_bytes;

            let new = model.get_device_layers(
                "this is a dummy config!",
                num_layers,
                layer_sizes_in_bytes,
                non_mapped_size_in_bytes,
                total_model_size_in_bytes,
                &devices,
                dtype,
                &params,
                paged_attn_config.as_ref(),
            )?;
            max_kv_tokens = Some(params.max_seq_len() * params.max_batch_size());
            mapper = DeviceMapSetting::Map(new);
        }

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = &device {
            unsafe { dev.disable_event_tracking() };
        }
        crate::utils::cuda_mempool::set_pool_retain_all(&device)?;

        let use_nccl = hanzo_quant::distributed::use_nccl();
        let available_devices = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
            let payload: WorkerTransferData = serde_json::from_str(&payload)?;
            let WorkerTransferData::Init { id: _, worker_rank } = payload;
            vec![hanzo_ml::Device::new_cuda(worker_rank + 1)?]
        } else if use_nccl {
            vec![hanzo_ml::Device::new_cuda(0)?]
        } else {
            device_map::get_all_similar_devices(device)?
        };

        let pipeline_mapper = mapper.into_mapper(
            num_layers,
            device,
            self.config.topology.as_ref(),
            &available_devices,
        )?;
        let mapper = mapper.into_mapper(
            num_layers,
            device,
            self.config.topology.as_ref(),
            &available_devices,
        )?;
        let mut layer_devices = Vec::new();
        for layer in 0..num_layers {
            let device = mapper.device_for(layer, false).cloned();
            layer_devices.push(device);
        }

        // TODO: PagedAttention is not supported with CPU for now.
        // This check is not really necessary because `get_device_layers` should prevent it.
        let mapping_uses_cpu = mapper.get_unique_devices().iter().any(Device::is_cpu);
        if mapping_uses_cpu {
            warn!("Device mapping contains a mix of GPU and CPU. There is no CPU support for PagedAttention, disabling PagedAttention.");
            paged_attn_config = None;
        }

        let GgufTokenizerConversion {
            tokenizer,
            bos,
            eos,
            unk,
        } = if paths.get_tokenizer_filename().to_string_lossy().is_empty() {
            convert_gguf_to_hf_tokenizer(&model)?
        } else {
            GgufTokenizerConversion {
                tokenizer: get_tokenizer(paths.get_tokenizer_filename(), None)?,
                bos: None,
                eos: None,
                unk: None,
            }
        };

        // Only load gguf chat template if there is nothing else
        let gguf_chat_template =
            if paths.get_template_filename().is_none() && self.chat_template.is_none() {
                get_gguf_chat_template(&model)?
            } else {
                None
            };

        let has_adapter = self.kind.is_adapted();
        let is_xlora = self.kind.is_adapted_and(|a| a.is_x_lora());

        let paged_attn_config = if matches!(self.kind, ModelKind::GgufAdapter { .. }) {
            warn!("Adapter models do not currently support PagedAttention, running without");
            None
        } else {
            paged_attn_config
        };

        let model_config_metadata: ContentConfig = (&model).into();
        let internal_dtype = mapper.get_min_dtype(dtype)?;

        let model_config = {
            // Base config (quantization only):
            let quant = ModelConfig::ParamsGGUF(
                model,
                (device, mapper).into(),
                if paged_attn_config.is_some() {
                    AttentionImplementation::PagedAttention
                } else {
                    AttentionImplementation::Eager
                },
                internal_dtype,
            );

            // With optional adapter config:
            let mut adapter = None;
            if has_adapter {
                adapter.replace(ModelConfig::Adapter::try_new(
                    paths, device, silent, is_xlora,
                )?);
            }

            ModelConfig::ModelParams::new(quant, adapter)
        };

        // Config into model:
        let model = match self.kind {
            ModelKind::GgufQuantized { .. } => match arch {
                GGUFArchitecture::Llama | GGUFArchitecture::Mistral3 => {
                    Model::Llama(QLlama::try_from(model_config)?)
                }
                GGUFArchitecture::Phi2 => Model::Phi2(QPhi::try_from(model_config)?),
                GGUFArchitecture::Phi3 => Model::Phi3(QPhi3::try_from(model_config)?),
                GGUFArchitecture::Starcoder2 => {
                    Model::Starcoder2(QStarcoder2::try_from(model_config)?)
                }
                GGUFArchitecture::Qwen2 => Model::Qwen(QQwen::try_from(model_config)?),
                GGUFArchitecture::Qwen3 => Model::Qwen3(QQwen3::try_from(model_config)?),
                GGUFArchitecture::Qwen3MoE => Model::Qwen3MoE(QQwen3MoE::try_from(model_config)?),
                GGUFArchitecture::Qwen35 | GGUFArchitecture::Qwen35MoE => {
                    Model::Qwen35(QQwen35::try_from(model_config)?)
                }
                GGUFArchitecture::Deepseek2 => {
                    Model::Deepseek2(QDeepSeek2::try_from(model_config)?)
                }
                a => bail!("Unsupported architecture `{a:?}` for GGUF"),
            },
            ModelKind::GgufAdapter { adapter, .. } => match arch {
                GGUFArchitecture::Llama | GGUFArchitecture::Mistral3 => {
                    Model::XLoraLlama(XLoraQLlama::try_from(model_config)?)
                }
                GGUFArchitecture::Phi3 => Model::XLoraPhi3(XLoraQPhi3::try_from(model_config)?),
                a => bail!(
                    "Unsupported architecture `{a:?}` for GGUF {kind}",
                    kind = adapter.pretty_name()
                ),
            },
            _ => unreachable!(),
        };

        let (cache_config, cache_engine) = if let Some(paged_attn_config) = paged_attn_config {
            let model_config: &dyn ModelConfigLike = &model_config_metadata;
            let cache_config = calculate_cache_config(
                paged_attn_config.mem_gpu,
                paged_attn_config.block_size,
                internal_dtype,
                paged_attn_config.cache_type,
                model_config,
                device,
                &layer_devices,
                silent,
                None,
                max_kv_tokens,
            )?;
            let cache_engine = CacheEngine::new(
                model_config,
                &cache_config,
                internal_dtype,
                device,
                layer_devices,
            )?;
            (Some(cache_config), Some(cache_engine))
        } else {
            (None, None)
        };

        let gen_conf: Option<GenerationConfig> = paths
            .get_gen_conf_filename()
            .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
        let chat_template_explicit = paths
            .get_chat_template_explicit()
            .as_ref()
            .map(|x| x.to_string_lossy().to_string());
        let mut chat_template = get_chat_template(
            paths,
            self.jinja_explicit.as_ref(),
            chat_template_explicit.as_ref(),
            self.chat_template.as_ref(),
            gguf_chat_template,
        );

        let max_seq_len = match model {
            Model::Llama(ref l) => l.max_seq_len,
            Model::Phi2(ref p) => p.max_seq_len,
            Model::XLoraLlama(ref xl) => xl.max_seq_len,
            Model::Phi3(ref p) => p.max_seq_len,
            Model::XLoraPhi3(ref p) => p.max_seq_len,
            Model::Starcoder2(ref p) => p.max_seq_len,
            Model::Qwen(ref p) => p.max_seq_len,
            Model::Qwen3(ref p) => p.max_seq_len,
            Model::Qwen3MoE(ref p) => p.max_seq_len,
            Model::Qwen35(ref p) => p.max_seq_len,
            Model::Deepseek2(ref p) => p.max_seq_len,
        };
        let llg_factory = build_llg_factory(tokenizer.clone())?;
        let num_hidden_layers = match model {
            Model::Llama(ref model) => model.cache.normal().0.len(),
            Model::Phi2(ref model) => model.cache.normal().0.len(),
            Model::XLoraLlama(ref model) => model.cache.full().lock().len(),
            Model::Phi3(ref model) => model.cache.normal().0.len(),
            Model::XLoraPhi3(ref model) => model.cache.full().lock().len(),
            Model::Starcoder2(ref model) => model.cache.normal().0.len(),
            Model::Qwen(ref model) => model.cache.normal().0.len(),
            Model::Qwen3(ref model) => model.cache.normal().0.len(),
            Model::Qwen3MoE(ref model) => model.cache.normal().0.len(),
            Model::Qwen35(ref model) => model.cache.hybrid().num_layers(),
            Model::Deepseek2(ref model) => model.cache.normal().0.len(),
        };

        if chat_template.bos_token.is_none() {
            if let Some(v) = bos {
                chat_template.bos_token = Some(BeginEndUnkPadTok(Either::Left(v)));
            }
        }
        if chat_template.eos_token.is_none() {
            if let Some(v) = eos {
                chat_template.eos_token = Some(BeginEndUnkPadTok(Either::Left(v)));
            }
        }
        if chat_template.unk_token.is_none() {
            if let Some(v) = unk {
                chat_template.unk_token = Some(BeginEndUnkPadTok(Either::Left(v)));
            }
        }

        let generation_defaults = gen_conf
            .as_ref()
            .and_then(GenerationConfig::generation_defaults);
        let eos = calculate_eos_tokens(&chat_template, gen_conf.as_ref(), &tokenizer);
        Ok(Arc::new(Mutex::new(GGUFPipeline {
            model,
            tokenizer: tokenizer.into(),
            no_kv_cache: self.no_kv_cache,
            chat_template: Arc::new(chat_template),
            model_id: self
                .model_id
                .clone()
                .unwrap_or(self.quantized_model_id.clone()),
            non_granular_state: self.tgt_non_granular_index.map(|tgt_non_granular_index| {
                NonGranularState {
                    non_granular_index: Arc::new(Mutex::new(0)),
                    tgt_non_granular_index,
                }
            }),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: Some(llg_factory),
                no_kv_cache: self.no_kv_cache,
                no_prefix_cache: false,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                is_xlora,
                activation_dtype: internal_dtype,
                sliding_window: None,
                cache_config,
                cache_engine,
                model_metadata: Some(Arc::new(model_config_metadata)),
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Text],
                },
            }),
            generation_defaults,
            mapper: pipeline_mapper,
            draft_proposer: None,
            #[cfg(feature = "rocm")]
            rocm_decode_graph: std::sync::Mutex::new(RocmDecodeGraphState::default()),
        })))
    }

    fn get_id(&self) -> String {
        self.xlora_model_id
            .as_deref()
            .unwrap_or(self.model_id.as_ref().unwrap_or(&self.quantized_model_id))
            .to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for GGUFPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for GGUFPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!(
            "You are trying to in-situ requantize a GGML model. This will not do anything."
        )
    }
}

impl CacheManagerMixin for GGUFPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) {
        match self.cache() {
            EitherCache::Full(_) => FullCacheManager.clone_in_cache(self, seqs, false),
            EitherCache::Normal(_) => NormalCacheManager.clone_in_cache(self, seqs, false),
            EitherCache::Hybrid(_) => HybridCacheManager.clone_in_cache(self, seqs, false),
        }
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        match self.cache() {
            EitherCache::Full(_) => FullCacheManager.clone_out_cache(self, seqs, false),
            EitherCache::Normal(_) => NormalCacheManager.clone_out_cache(self, seqs, false),
            EitherCache::Hybrid(_) => HybridCacheManager.clone_out_cache(self, seqs, false),
        }
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        match self.cache() {
            EitherCache::Full(_) => {
                FullCacheManager.set_none_cache(self, seqs, modify_draft_cache, false)
            }
            EitherCache::Normal(_) => NormalCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            ),
            EitherCache::Hybrid(_) => HybridCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            ),
        }
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &EitherCache {
        match self.model {
            Model::Llama(ref model) => &model.cache,
            Model::Phi2(ref model) => &model.cache,
            Model::XLoraLlama(ref model) => &model.cache,
            Model::Phi3(ref model) => &model.cache,
            Model::XLoraPhi3(ref model) => &model.cache,
            Model::Starcoder2(ref model) => &model.cache,
            Model::Qwen(ref model) => &model.cache,
            Model::Qwen3(ref model) => &model.cache,
            Model::Qwen3MoE(ref model) => &model.cache,
            Model::Qwen35(ref model) => &model.cache,
            Model::Deepseek2(ref model) => &model.cache,
        }
    }
}

impl MetadataMixin for GGUFPipeline {
    fn device(&self) -> Device {
        match self.model {
            Model::Llama(ref model) => model.device.clone(),
            Model::Phi2(ref model) => model.device.clone(),
            Model::XLoraLlama(ref model) => model.device.clone(),
            Model::Phi3(ref model) => model.device.clone(),
            Model::XLoraPhi3(ref model) => model.device.clone(),
            Model::Starcoder2(ref model) => model.device.clone(),
            Model::Qwen(ref model) => model.device.clone(),
            Model::Qwen3(ref model) => model.device.clone(),
            Model::Qwen3MoE(ref model) => model.device.clone(),
            Model::Qwen35(ref model) => model.device.clone(),
            Model::Deepseek2(ref model) => model.device.clone(),
        }
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {
        if let Some(s) = self.non_granular_state.as_ref() {
            *self.cache().full().get_scalings_cache() = None;
            *get_mut_arcmutex!(s.non_granular_index) = 0;
        }
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn generation_defaults(&self) -> Option<crate::ModelGenerationDefaults> {
        self.generation_defaults.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

#[cfg(feature = "rocm")]
impl GGUFPipeline {
    /// Whether the active model variant runs the standard
    /// `(input_ids, seqlen_offsets, context_lens, paged_meta)` decode forward that
    /// the ROCm decode graph captures. XLora and the alternate-signature variants
    /// (Phi3/Starcoder2) are excluded from the graph fast path for v1.
    fn model_supports_rocm_decode_graph(&self) -> bool {
        // Graph-eligible variants run their decode RoPE off a stable device positions tensor
        // (`metadata.rope_positions`), so capture bakes a buffer the graph runner refreshes in place
        // every token instead of a frozen host offset. Qwen3MoE qualifies once its attention reads
        // those positions like the dense path; before that fix its RoPE froze at the warmup position
        // and every replayed token rotated wrong -> attention drifted into stale repetition.
        // Qwen35 (Qwen3-VL mRoPE + non-paged Sdpa decode) still derives its cos/sin from a fresh
        // per-forward positions tensor, which would freeze under capture, so it stays eager until its
        // mRoPE is threaded through `rope_positions` too. XLora and the alternate-signature variants
        // (Phi3/Starcoder2) are excluded from the graph fast path for v1.
        matches!(
            self.model,
            Model::Llama(_)
                | Model::Phi2(_)
                | Model::Qwen(_)
                | Model::Qwen3(_)
                | Model::Qwen3MoE(_)
        )
    }

    /// Runs the decode forward for the graph-eligible variants with the supplied
    /// (rebound) paged metadata. Mirrors the eager dispatch in `forward_inputs`
    /// exactly so the captured forward is identical to what the eager path runs.
    fn run_rocm_decode_forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        paged_attn_meta: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor, hanzo_ml::Error> {
        match self.model {
            Model::Llama(ref model) => {
                model.forward(input_ids, seqlen_offsets, context_lens, paged_attn_meta)
            }
            Model::Phi2(ref model) => {
                model.forward(input_ids, seqlen_offsets, context_lens, paged_attn_meta)
            }
            Model::Qwen(ref model) => {
                model.forward(input_ids, seqlen_offsets, context_lens, paged_attn_meta)
            }
            Model::Qwen3(ref model) => {
                model.forward(input_ids, seqlen_offsets, context_lens, paged_attn_meta)
            }
            Model::Qwen3MoE(ref model) => {
                model.forward(input_ids, seqlen_offsets, context_lens, paged_attn_meta)
            }
            Model::Qwen35(ref model) => {
                model.forward(input_ids, seqlen_offsets, context_lens, paged_attn_meta)
            }
            _ => hanzo_ml::bail!("ROCm decode graph: unsupported model variant"),
        }
    }

    /// Attempts to satisfy a single decode step via a captured HIP graph. Returns
    /// `Ok(None)` when the graph path does not apply (caller runs eager). Mirrors
    /// `NormalPipeline::try_cuda_decode_graph_forward`.
    fn try_rocm_decode_graph_forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
        kv_cache: &[(Tensor, Tensor)],
        metadata: &PagedAttentionInputMetadata,
    ) -> Result<Option<Tensor>, hanzo_ml::Error> {
        if !rocm_decode_graphs_enabled() || !self.model_supports_rocm_decode_graph() {
            return Ok(None);
        }
        // Mirror the CUDA gate: only steady-state single-token decode with paged
        // metadata present and no prefix-cache prefill in flight.
        if metadata.is_first_prompt_chunk
            || metadata.disable_cuda_graphs
            || metadata.num_cached_tokens.is_some()
        {
            return Ok(None);
        }
        let (batch, q_len) = input_ids.dims2()?;
        if q_len != 1
            || seqlen_offsets.len() != batch
            || context_lens.len() != batch
            || !input_ids.device().is_rocm()
        {
            return Ok(None);
        }
        let Some(cache_config) = self.metadata.cache_config.as_ref() else {
            return Ok(None);
        };
        let block_size = cache_config.block_size;
        let key = RocmDecodeGraphKey::new(input_ids, metadata, block_size)?;

        let mut state = self
            .rocm_decode_graph
            .lock()
            .expect("ROCm graph mutex poisoned");
        if state.disabled {
            return Ok(None);
        }

        // Cache hit: refresh the stable buffers in place and replay in one launch.
        if let Some(pos) = state.entries.iter().position(|entry| entry.key == key) {
            let mut entry = state.entries.remove(pos);
            entry.input_ids.set(input_ids)?;
            entry.metadata_buffers.copy_from(metadata, seqlen_offsets)?;
            entry.graph.launch()?;
            let logits = entry.logits.clone();
            // Env-gated replay diagnostic: confirm the replayed logits advance
            // (argmax != fixed 0) — the AutoFreeOnLaunch stale-buffer signature
            // is a constant argmax=0. Reads to host force a stream sync, so this
            // is debug-only and off the hot path.
            if std::env::var("ROCM_GRAPH_DEBUG").is_ok() {
                let flat = logits.flatten_all().ok();
                let amax = flat
                    .as_ref()
                    .and_then(|t| t.argmax(hanzo_ml::D::Minus1).ok())
                    .and_then(|t| t.to_scalar::<u32>().ok());
                tracing::info!(
                    "[graph-replay] in_tok={:?} argmax={:?}",
                    input_ids
                        .flatten_all()
                        .ok()
                        .and_then(|t| t.to_vec1::<u32>().ok()),
                    amax
                );
            }
            state.entries.push(entry);
            return Ok(Some(logits));
        }

        // Cache miss: run a real (eager) warmup forward first so the caller gets a
        // correct first token, then capture a graph for subsequent tokens.
        let warmup_logits = self.run_rocm_decode_forward(
            input_ids,
            seqlen_offsets,
            context_lens.to_vec(),
            Some((kv_cache.to_vec(), metadata)),
        )?;
        input_ids.device().synchronize()?;

        // The warmup logits are a fully valid result for this token. If capture
        // fails we must NOT lose them: disable the graph path (so later tokens run
        // eager) and return the warmup logits. `capture_rocm_decode_graph` always
        // drains the stream on failure, so the eager fallback continues coherently.
        match self.capture_rocm_decode_graph(
            key,
            input_ids,
            seqlen_offsets,
            context_lens,
            kv_cache,
            metadata,
            block_size,
        ) {
            Ok(entry) => {
                if state.entries.len() >= ROCM_DECODE_GRAPH_CACHE_CAPACITY {
                    state.entries.remove(0);
                }
                state.entries.push(entry);
            }
            Err(err) => {
                if !state.disabled {
                    warn!("ROCm decode graph capture failed; falling back to eager decode: {err}");
                }
                state.disabled = true;
                state.entries.clear();
            }
        }
        Ok(Some(warmup_logits))
    }

    /// Captures the decode forward into a HIP graph against stable metadata
    /// buffers. Mirrors `NormalPipeline::capture_cuda_decode_graph`.
    #[allow(clippy::too_many_arguments)]
    fn capture_rocm_decode_graph(
        &self,
        key: RocmDecodeGraphKey,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
        kv_cache: &[(Tensor, Tensor)],
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
    ) -> Result<RocmDecodeGraphEntry, hanzo_ml::Error> {
        use crate::pipeline::rocm_graph::{
            begin_rocm_capture, end_rocm_capture_discard, rocm_device,
        };

        let input_ids_var = Var::from_tensor(input_ids)?;
        let (metadata_buffers, rebound_metadata) =
            RocmDecodeGraphMetadataBuffers::new(metadata, seqlen_offsets, block_size)?;
        let graph_input_ids = input_ids_var.as_detached_tensor();
        let device = rocm_device(graph_input_ids.device())?;

        // Open the caching-pool reservation scope BEFORE capture begins and keep
        // it open until capture ends. Every output buffer the captured forward
        // allocates (including the final `logits` tensor) is then reserved out of
        // the pool for good, so no later eager allocation can reuse a device
        // pointer baked into the graph and corrupt a replay. Without this the
        // graph replays against recycled scratch and the logits never advance
        // (the fluent-but-stale single-token loop). See `wrappers::PoolInner`.
        device.begin_graph_capture_scope();

        begin_rocm_capture(&device)?;
        let logits = match self.run_rocm_decode_forward(
            &graph_input_ids,
            seqlen_offsets,
            context_lens.to_vec(),
            Some((kv_cache.to_vec(), &rebound_metadata)),
        ) {
            Ok(logits) => logits,
            Err(err) => {
                end_rocm_capture_discard(&device);
                device.end_graph_capture_scope();
                return Err(err);
            }
        };

        let graph = match RocmGraphHandle::end_capture(&device) {
            Ok(Some(graph)) => graph,
            Ok(None) => {
                // Capture began but produced no graph; drain so the stream is
                // usable for the eager fallback.
                let _ = device.synchronize();
                device.end_graph_capture_scope();
                return Err(hanzo_ml::Error::msg("ROCm graph capture returned no graph"));
            }
            Err(err) => {
                end_rocm_capture_discard(&device);
                device.end_graph_capture_scope();
                return Err(err);
            }
        };
        // Capture is finished; the graph now owns the reserved pointers. Close the
        // reservation scope so steady-state eager allocations (sampling, next
        // token prep) recycle from the pool as usual.
        device.end_graph_capture_scope();

        graph.upload()?;

        Ok(RocmDecodeGraphEntry {
            key,
            graph,
            input_ids: input_ids_var,
            metadata_buffers,
            _metadata: rebound_metadata,
            logits,
        })
    }

    /// Disables the ROCm decode graph fast path after a capture/replay failure and
    /// clears the cache so the pipeline falls back to eager. Mirrors
    /// `NormalPipeline::disable_cuda_decode_graph`.
    fn disable_rocm_decode_graph(&self, err: &hanzo_ml::Error) {
        let mut state = self
            .rocm_decode_graph
            .lock()
            .expect("ROCm graph mutex poisoned");
        if !state.disabled {
            warn!("ROCm decode graphs disabled after capture/replay error: {err}");
        }
        state.disabled = true;
        state.entries.clear();
    }
}

#[async_trait::async_trait]
impl Pipeline for GGUFPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, hanzo_ml::Error> {
        let ModelInputs {
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            context_lens,
            position_ids: _, // NOTE(hanzoai): ignore, it is for phi3
            paged_attn_meta,
            flash_meta,
            flash_meta_full,
        } = *inputs.downcast().expect("Downcast failed.");
        let metadata = self.get_metadata();
        let paged_attn_meta = match (&metadata.cache_engine, &paged_attn_meta) {
            (Some(engine), Some(meta)) => Some((engine.get_kv_cache().clone(), meta)),
            (Some(_), None) => {
                // This can happen if Rust-side user code is wrong
                hanzo_ml::bail!("Forward step expected a PagedAttention input metadata. This was not provided, please ensure that the scheduler config is correctly configured for PagedAttention.")
            }
            (None, Some(_)) => {
                // This should never happen but we handle it anyway
                hanzo_ml::bail!("Forward step got a PagedAttention input metadata but there is no cache engine. Please raise an issue.")
            }
            (None, None) => None,
        };
        // ROCm/HIP decode graph fast path. On a steady-state single-token decode
        // with paged metadata present, replay the captured graph (or capture it on
        // the first such step) instead of re-launching every kernel eagerly. On any
        // capture/replay error we disable the path and fall through to eager so a
        // stale-state replay can never silently corrupt tokens. Other devices and
        // non-decode steps skip straight to the eager dispatch below.
        #[cfg(feature = "rocm")]
        {
            if let Some((ref kv_cache, meta)) = paged_attn_meta {
                match self.try_rocm_decode_graph_forward(
                    &input_ids,
                    &seqlen_offsets,
                    &context_lens,
                    kv_cache,
                    meta,
                ) {
                    Ok(Some(logits)) => {
                        return if return_raw_logits {
                            Ok(ForwardInputsResult::RawLogits { logits })
                        } else {
                            Ok(ForwardInputsResult::CausalGeneration { logits })
                        };
                    }
                    Ok(None) => {}
                    Err(err) => self.disable_rocm_decode_graph(&err),
                }
            }
        }
        let logits = match self.model {
            Model::Llama(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Phi2(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::XLoraLlama(ref model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
                &flash_meta,
                flash_meta_full.as_ref().unwrap_or(&flash_meta),
            )?,
            Model::Phi3(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, paged_attn_meta)?
            }
            Model::XLoraPhi3(ref model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
                &flash_meta,
                flash_meta_full.as_ref().unwrap_or(&flash_meta),
            )?,
            Model::Starcoder2(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, paged_attn_meta)?
            }
            Model::Qwen(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Qwen3(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Qwen3MoE(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Qwen35(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Deepseek2(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
        };
        if return_raw_logits {
            Ok(ForwardInputsResult::RawLogits { logits })
        } else {
            Ok(ForwardInputsResult::CausalGeneration { logits })
        }
    }
    fn attach_speculative(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
    ) -> Result<(), hanzo_ml::Error> {
        match config {
            crate::speculative::SpeculativeConfig::Off => Ok(()),
            crate::speculative::SpeculativeConfig::DraftModel { draft, gamma } => {
                if self.metadata.cache_engine.is_none() {
                    hanzo_ml::bail!(
                        "draft-model speculative decoding currently requires PagedAttention for this pipeline."
                    );
                }
                {
                    let target_tok = self.tokenizer().ok_or_else(|| {
                        hanzo_ml::Error::msg("target pipeline has no tokenizer for speculative decoding")
                    })?;
                    let draft_guard = draft.try_lock().map_err(|_| {
                        hanzo_ml::Error::msg("draft pipeline is not exclusively owned")
                    })?;
                    let draft_tok = draft_guard.tokenizer().ok_or_else(|| {
                        hanzo_ml::Error::msg("draft pipeline has no tokenizer for speculative decoding")
                    })?;
                    if target_tok.get_vocab(true) != draft_tok.get_vocab(true) {
                        hanzo_ml::bail!(
                            "target and draft tokenizer vocabularies differ; classic speculative decoding requires identical tokenizers."
                        );
                    }
                }
                let proposer = crate::speculative::DraftModelProposer::new(draft, gamma)?;
                let info = crate::speculative::SpeculativeAttachInfo::draft_model(gamma);
                crate::speculative::logging::log_attach(&info);
                self.draft_proposer = Some(proposer);
                Ok(())
            }
            other => hanzo_ml::bail!(
                "GGUF pipeline supports only draft-model speculative decoding, got {other:?}"
            ),
        }
    }

    fn retain_speculative_seqs(&mut self, live: &[usize]) {
        use crate::speculative::SpeculativeProposer;
        if let Some(proposer) = self.draft_proposer.as_mut() {
            proposer.retain_seqs(live);
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_sample_speculative_causal_gen(
        &mut self,
        seqs: &mut [&mut Sequence],
        logits: &[Tensor],
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        metadata: Option<crate::pipeline::text_models_inputs_processor::PagedAttentionMeta>,
    ) -> Result<bool, hanzo_ml::Error> {
        if self.draft_proposer.is_none() {
            crate::speculative::driver::clear_staged_speculative_tokens(seqs);
            return Ok(false);
        }

        let general_metadata = self.get_metadata();
        if let Some(cache_engine) = general_metadata.cache_engine.as_ref() {
            let Some(metadata) = metadata else {
                crate::speculative::driver::clear_staged_speculative_tokens(seqs);
                return Ok(false);
            };
            let cache = crate::speculative::cache::PagedSpeculativeCacheAccess::new(
                &metadata,
                cache_engine,
            );
            return crate::speculative::driver::try_sample_speculative_causal_gen(
                self,
                seqs,
                logits,
                prefix_cacher,
                disable_eos_stop,
                rng,
                &cache,
            )
            .await;
        }

        crate::speculative::driver::clear_staged_speculative_tokens(seqs);
        Ok(false)
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), hanzo_ml::Error> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }
}

impl crate::speculative::driver::SpeculativePipelineExt for GGUFPipeline {
    fn has_speculative_proposer(&self) -> bool {
        self.draft_proposer.is_some()
    }

    fn speculative_proposal_len(&self) -> Option<usize> {
        use crate::speculative::SpeculativeProposer;
        self.draft_proposer.as_ref().map(|p| p.proposal_len())
    }

    fn speculative_target_hiddens(
        &self,
        _rows: &[(usize, usize)],
    ) -> hanzo_ml::Result<Option<Tensor>> {
        Ok(None)
    }

    fn speculative_propose(
        &mut self,
        ctx: crate::speculative::SpeculativeProposeBatchCtx<'_>,
    ) -> hanzo_ml::Result<Option<crate::speculative::SpeculativeProposalBatch>> {
        use crate::speculative::SpeculativeProposer;
        match self.draft_proposer.as_mut() {
            Some(proposer) => Ok(Some(proposer.propose(ctx, None)?)),
            None => Ok(None),
        }
    }

    fn build_speculative_verify_inputs(
        &self,
        input_meta: crate::pipeline::text_models_inputs_processor::InputMetadata,
    ) -> hanzo_ml::Result<Box<dyn Any>> {
        Ok(Box::new(ModelInputs {
            input_ids: input_meta.input,
            input_ids_full: None,
            seqlen_offsets: input_meta.positions,
            seqlen_offsets_full: None,
            context_lens: input_meta.context_lens,
            position_ids: input_meta.position_ids,
            paged_attn_meta: input_meta.paged_attn_meta,
            flash_meta: input_meta.flash_meta,
            flash_meta_full: None,
        }))
    }
}

// TODO
impl AnyMoePipelineMixin for GGUFPipeline {}
