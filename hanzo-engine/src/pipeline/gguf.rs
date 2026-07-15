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
#[cfg(feature = "cuda")]
use crate::pipeline::cuda_graph::{
    cuda_decode_graphs_enabled, cuda_prefill_graphs_enabled, disable_event_tracking_for_capture,
    end_cuda_capture_discard, restore_event_tracking_after_capture, CudaDecodeGraphKey,
    CudaDecodeGraphMetadataBuffers, CudaGraphHandle, CUDA_DECODE_GRAPH_CACHE_CAPACITY,
};
use crate::pipeline::loaders::DeviceMappedModelLoader;
#[cfg(feature = "rocm")]
use crate::pipeline::rocm_graph::{
    rocm_decode_graphs_enabled, RocmDecodeGraphKey, RocmDecodeGraphMetadataBuffers,
    RocmGraphHandle, ROCM_DECODE_GRAPH_CACHE_CAPACITY,
};
use crate::pipeline::sampling::sample_and_add_toks;
#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::ChatTemplate;
use crate::pipeline::{get_chat_template, Modalities, SupportedModality};
use crate::pipeline_parallel::{pp_worker_step, use_pipeline_parallel};
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
    models::quantized_deepseek4::ModelWeights as QDeepSeek4,
    models::quantized_gemma::ModelWeights as QGemma,
    models::quantized_glm4_moe::ModelWeights as QGlm4Moe,
    models::quantized_gptoss::ModelWeights as QGptOss,
    models::quantized_llama::ModelWeights as QLlama,
    models::quantized_minimax::ModelWeights as QMiniMax,
    models::quantized_phi2::ModelWeights as QPhi,
    models::quantized_phi3::ModelWeights as QPhi3,
    models::quantized_qwen::ModelWeights as QQwen,
    models::quantized_qwen3::ModelWeights as QQwen3,
    models::quantized_qwen3_5_moe::ModelWeights as QQwen35,
    models::quantized_qwen3_moe::ModelWeights as QQwen3MoE,
    models::quantized_qwen3_next::ModelWeights as QQwen3Next,
    models::quantized_starcoder2::ModelWeights as QStarcoder2,
    utils::tokens::get_token,
    xlora_models::{XLoraQLlama, XLoraQPhi3},
};
use anyhow::{bail, Result};
use either::Either;
#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
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
    Gemma(QGemma),
    MiniMax(QMiniMax),
    Starcoder2(QStarcoder2),
    Qwen(QQwen),
    Qwen3(QQwen3),
    Qwen3MoE(QQwen3MoE),
    Qwen3Next(QQwen3Next),
    Qwen35(QQwen35),
    Deepseek2(QDeepSeek2),
    Deepseek4(QDeepSeek4),
    GptOss(QGptOss),
    Glm4Moe(QGlm4Moe),
}

impl Model {
    /// The model's self-speculative capability, if it carries an MTP head. This is
    /// the ONE place the enum maps a variant to [`SelfSpeculative`]; the speculative
    /// pipeline asks for the trait object and never matches on the architecture.
    /// Adding a self-speculative model = one arm here + the trait impl by its
    /// weights (e.g. GLM-5.2's in-band `nextn` head).
    ///
    /// [`SelfSpeculative`]: crate::speculative::SelfSpeculative
    fn as_self_speculative(&self) -> Option<&dyn crate::speculative::SelfSpeculative> {
        match self {
            // GLM-5.2 (`glm-dsa`) loads as Deepseek2 and carries an in-band `nextn` MTP head.
            Model::Deepseek2(m) => Some(m),
            Model::Deepseek4(m) => Some(m),
            _ => None,
        }
    }
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
    draft_proposer: Option<Box<dyn crate::speculative::SpeculativeProposer + Send + Sync>>,
    /// Captured ROCm/HIP decode graphs, keyed by decode bucket. See
    /// [`crate::pipeline::rocm_graph`]. Mirrors `NormalPipeline::cuda_decode_graph`.
    #[cfg(feature = "rocm")]
    rocm_decode_graph: std::sync::Mutex<RocmDecodeGraphState>,
    /// Captured CUDA decode graphs, keyed by decode bucket. The GGUF analog of
    /// `NormalPipeline::cuda_decode_graph` (which only wires the safetensors path);
    /// reuses the same `cuda_graph` machinery for the quantized decode forward.
    #[cfg(feature = "cuda")]
    cuda_decode_graph: std::sync::Mutex<CudaDecodeGraphState>,
    #[cfg(feature = "cuda")]
    cuda_prefill_graph: std::sync::Mutex<CudaPrefillGraphState>,
    /// Captured Vulkan decode command-graph (single-token, naive KV cache). One graph at a time,
    /// invalidated on a KV-cache capacity grow. See [`crate::perf_flags::vulkan_graphs_enabled`].
    #[cfg(feature = "vulkan")]
    vulkan_decode_graph: std::sync::Mutex<VulkanDecodeGraphState>,
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

#[cfg(feature = "vulkan")]
#[derive(Default)]
struct VulkanDecodeGraphState {
    entry: Option<VulkanDecodeGraphEntry>,
    disabled: bool,
}

#[cfg(feature = "vulkan")]
struct VulkanDecodeGraphEntry {
    /// The captured decode forward; `replay()` re-submits it in one queue submit + fence wait,
    /// replacing the eager per-token re-record of ~1.7k dispatches.
    graph: hanzo_ml::VkGraph,
    /// Shared per-forward attention buffers (scale + meta). One serves every layer; `seq_k` (the
    /// attended span) advances in place per replay via [`hanzo_ml::VkGraphAttn::set_seq_k`].
    attn: hanzo_ml::VkGraphAttn,
    /// Stable input-token buffer the captured embedding gather reads; refreshed each replay.
    input_ids: Var,
    /// Stable u32 position buffer feeding BOTH the captured RoPE and the device-offset KV append
    /// (the advancing write slot). Refreshed each replay; the two uses share one value (the position).
    positions: Var,
    /// The pinned logits tensor the captured output head writes; cloned after each replay for sampling.
    logits: Tensor,
    /// KV cache capacity the graph was captured against. The graph is invalidated (recaptured) once
    /// the write slot reaches this, since a cache grow reallocates the buffer whose handle the graph
    /// baked into its recorded descriptors.
    capacity: usize,
    /// The one decode position this graph is valid to replay next. A captured graph replays only a
    /// STRICTLY sequential continuation of the sequence it captured (position advances by exactly one
    /// per token, on the same KV buffers); any other position means a different / reset sequence, so
    /// the graph is invalidated and recaptured. Guards against replaying against another sequence's
    /// state (the naive cache is reallocated per sequence, unlike a persistent paged pool).
    next_position: usize,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct CudaDecodeGraphState {
    entries: Vec<CudaDecodeGraphEntry>,
    disabled: bool,
}

#[cfg(feature = "cuda")]
struct CudaDecodeGraphEntry {
    key: CudaDecodeGraphKey,
    graph: CudaGraphHandle,
    input_ids: Var,
    metadata_buffers: CudaDecodeGraphMetadataBuffers,
    _metadata: PagedAttentionInputMetadata,
    logits: Tensor,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct CudaPrefillGraphState {
    entries: Vec<CudaPrefillGraphEntry>,
    disabled: bool,
    // Shapes whose captured graph did not replay bit-exactly against the eager warmup: those buckets
    // stay eager forever (fail-closed) so the default-on path only ever serves verified-exact graphs.
    denied: Vec<CudaDecodeGraphKey>,
}

// Bitwise eager-vs-replay comparison of the prefill logits, computed once per captured shape. A
// graph is only trusted (cached for replay) when `bit_exact`; otherwise the bucket is denied and
// falls back to the always-correct eager prefill.
#[cfg(feature = "cuda")]
struct PrefillGraphDivergence {
    bit_exact: bool,
    max_abs_diff: f32,
    mismatched: usize,
    total: usize,
    argmax_eager: u32,
    argmax_replay: u32,
}

#[cfg(feature = "cuda")]
fn prefill_graph_divergence(
    eager: &Tensor,
    replay: &Tensor,
) -> Result<PrefillGraphDivergence, hanzo_ml::Error> {
    let eager = eager
        .to_dtype(hanzo_ml::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let replay = replay
        .to_dtype(hanzo_ml::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if eager.len() != replay.len() {
        hanzo_ml::bail!(
            "prefill graph logits length mismatch: eager={} replay={}",
            eager.len(),
            replay.len()
        );
    }
    let mut max_abs_diff = 0f32;
    let mut mismatched = 0usize;
    let (mut argmax_eager, mut argmax_replay) = (0u32, 0u32);
    let (mut best_eager, mut best_replay) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, (&e, &r)) in eager.iter().zip(replay.iter()).enumerate() {
        if e.to_bits() != r.to_bits() {
            mismatched += 1;
            max_abs_diff = max_abs_diff.max((e - r).abs());
        }
        if e > best_eager {
            best_eager = e;
            argmax_eager = i as u32;
        }
        if r > best_replay {
            best_replay = r;
            argmax_replay = i as u32;
        }
    }
    Ok(PrefillGraphDivergence {
        bit_exact: mismatched == 0,
        max_abs_diff,
        mismatched,
        total: eager.len(),
        argmax_eager,
        argmax_replay,
    })
}

// A captured dense prefill: one cuGraphLaunch replaces the ~1.5k eager kernel launches whose
// per-op CPU dispatch/alloc cost otherwise starves the GPU (measured prefill GPU util 83% vs
// llama's 98%). Keyed on the fixed [1, seq_len] bucket. `flash_params` is held so the varlen
// cumulative-seqlens device tensors the captured flash kernels reference stay live.
#[cfg(feature = "cuda")]
struct CudaPrefillGraphEntry {
    key: CudaDecodeGraphKey,
    graph: CudaGraphHandle,
    input_ids: Var,
    metadata_buffers: CudaDecodeGraphMetadataBuffers,
    _metadata: PagedAttentionInputMetadata,
    _flash_params: FlashParams,
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

        let mut files = Vec::new();
        let weight_filenames: Vec<std::path::PathBuf> = paths.get_weight_filenames().to_vec();
        for filename in &weight_filenames {
            files.push(std::fs::File::open(filename)?);
        }
        let mmaps: Vec<std::sync::Arc<memmap2::Mmap>> = files
            .iter()
            .map(|f| unsafe { memmap2::Mmap::map(f) }.map(std::sync::Arc::new))
            .collect::<std::io::Result<_>>()?;
        let mut readers = files.iter_mut().collect::<Vec<_>>();
        let mut model = Content::from_readers(&mut readers)?;
        model.set_mmaps(mmaps);

        // Disk-streaming expert cache (low-memory mode): keep the dense weights resident but stream
        // stacked MoE expert banks from NVMe, so a model far larger than RAM runs on one box. Gated
        // by the STREAM_EXPERTS env knob (default off -> resident behaviour unchanged).
        let stream_experts = env::var("STREAM_EXPERTS")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        if stream_experts {
            hanzo_ml::quantized::expert_stream::set_enabled(true);
            model.set_stream_paths(weight_filenames.clone());
            if let Some(dir) = weight_filenames.first().and_then(|p| p.parent()) {
                hanzo_ml::quantized::expert_stream::set_usage_sidecar(
                    dir.join(".hanzo_experts_usage"),
                );
            }
        }

        if !silent {
            model.print_metadata()?;
        }

        let arch = model.arch();

        // If auto, convert to Map
        let num_layers = model.get_metadata()[&format!("{arch}.block_count")].to_u32()? as usize;

        let mut max_kv_tokens: Option<usize> = None;

        // Pipeline parallelism owns layer placement itself (each rank loads only its own range
        // onto its single local device) and runs eager attention, so bypass the auto device map
        // and paged-attention cache engine.
        if use_pipeline_parallel() {
            mapper = DeviceMapSetting::dummy();
            paged_attn_config = None;
        }

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
        crate::utils::cuda_mempool::set_pool_retain_all(device)?;

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
        } else if matches!(arch, GGUFArchitecture::Deepseek2 | GGUFArchitecture::GlmDsa) {
            // quantized deepseek2 (incl. GLM-4.7-Flash, GLM-5.2) defaults to un-absorbed MLA
            // (materialized per-head K/V) on the eager, non-paged cache. `MLA_ABSORB=1` switches the
            // decode to the compressed [kv_lora, 1] latent with kv_b folded into q/out (see
            // quantized_deepseek2.rs) — device-agnostic, but still on the eager (contiguous) cache.
            // Wiring PagedAttention needs the *paged* MLA cache + FlashInfer MLA kernels (CUDA-only,
            // crate::mla::mla_decode_forward), which this eager path does not provide, so paged stays
            // off here regardless of MLA_ABSORB.
            warn!("GGUF deepseek2 (MLA) runs eager attention; paged MLA (FlashInfer) not wired (MLA_ABSORB gives the compressed-latent eager decode)");
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
                GGUFArchitecture::Gemma | GGUFArchitecture::Gemma2 => {
                    Model::Gemma(QGemma::try_from(model_config)?)
                }
                GGUFArchitecture::MiniMaxM2 => Model::MiniMax(QMiniMax::try_from(model_config)?),
                GGUFArchitecture::Starcoder2 => {
                    Model::Starcoder2(QStarcoder2::try_from(model_config)?)
                }
                GGUFArchitecture::Qwen2 => Model::Qwen(QQwen::try_from(model_config)?),
                GGUFArchitecture::Qwen3 => Model::Qwen3(QQwen3::try_from(model_config)?),
                GGUFArchitecture::Qwen3MoE => Model::Qwen3MoE(QQwen3MoE::try_from(model_config)?),
                // Qwen3-VL text backbone reuses the dense/MoE Qwen3 quantized weights.
                // Image-blind: the LLM GGUF carries only the text tower (`blk.*`); the
                // vision tower ships as a separate mmproj GGUF (not yet wired). For
                // text-only input interleaved-MRoPE == standard RoPE, so the forward is
                // bit-identical to plain Qwen3/Qwen3MoE.
                GGUFArchitecture::Qwen3Vl => Model::Qwen3(QQwen3::try_from(model_config)?),
                GGUFArchitecture::Qwen3VlMoE => Model::Qwen3MoE(QQwen3MoE::try_from(model_config)?),
                GGUFArchitecture::Qwen3Next => {
                    Model::Qwen3Next(QQwen3Next::try_from(model_config)?)
                }
                GGUFArchitecture::Qwen35 | GGUFArchitecture::Qwen35MoE => {
                    Model::Qwen35(QQwen35::try_from(model_config)?)
                }
                GGUFArchitecture::Deepseek2 | GGUFArchitecture::GlmDsa => {
                    Model::Deepseek2(QDeepSeek2::try_from(model_config)?)
                }
                GGUFArchitecture::Deepseek4 => {
                    Model::Deepseek4(QDeepSeek4::try_from(model_config)?)
                }
                GGUFArchitecture::GptOss => Model::GptOss(QGptOss::try_from(model_config)?),
                GGUFArchitecture::Glm4Moe => Model::Glm4Moe(QGlm4Moe::try_from(model_config)?),
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

        // Banks are open: size their LRU caches from RAM and pin the learned-hot experts.
        if stream_experts {
            hanzo_ml::quantized::expert_stream::finalize();
        }

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
            Model::Gemma(ref p) => p.max_seq_len,
            Model::MiniMax(ref p) => p.max_seq_len,
            Model::Starcoder2(ref p) => p.max_seq_len,
            Model::Qwen(ref p) => p.max_seq_len,
            Model::Qwen3(ref p) => p.max_seq_len,
            Model::Qwen3MoE(ref p) => p.max_seq_len,
            Model::Qwen3Next(ref p) => p.max_seq_len,
            Model::Qwen35(ref p) => p.max_seq_len,
            Model::Deepseek2(ref p) => p.max_seq_len,
            Model::Deepseek4(ref p) => p.max_seq_len,
            Model::GptOss(ref p) => p.max_seq_len,
            Model::Glm4Moe(ref p) => p.max_seq_len,
        };
        let llg_factory = build_llg_factory(tokenizer.clone())?;
        let num_hidden_layers = match model {
            Model::Llama(ref model) => model.cache.normal().0.len(),
            Model::Phi2(ref model) => model.cache.normal().0.len(),
            Model::XLoraLlama(ref model) => model.cache.full().lock().len(),
            Model::Phi3(ref model) => model.cache.normal().0.len(),
            Model::XLoraPhi3(ref model) => model.cache.full().lock().len(),
            Model::Gemma(ref model) => model.cache.normal().0.len(),
            Model::MiniMax(ref model) => model.cache.normal().0.len(),
            Model::Starcoder2(ref model) => model.cache.normal().0.len(),
            Model::Qwen(ref model) => model.cache.normal().0.len(),
            Model::Qwen3(ref model) => model.cache.normal().0.len(),
            Model::Qwen3MoE(ref model) => model.cache.normal().0.len(),
            Model::Qwen35(ref model) => model.cache.hybrid().num_layers(),
            Model::Qwen3Next(ref model) => model.cache.hybrid().num_layers(),
            Model::Deepseek2(ref model) => model.cache.normal().0.len(),
            Model::Deepseek4(ref model) => model.cache.normal().0.len(),
            Model::GptOss(ref model) => model.cache.normal().0.len(),
            Model::Glm4Moe(ref model) => model.cache.normal().0.len(),
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
            #[cfg(feature = "cuda")]
            cuda_decode_graph: std::sync::Mutex::new(CudaDecodeGraphState::default()),
            #[cfg(feature = "cuda")]
            cuda_prefill_graph: std::sync::Mutex::new(CudaPrefillGraphState::default()),
            #[cfg(feature = "vulkan")]
            vulkan_decode_graph: std::sync::Mutex::new(VulkanDecodeGraphState::default()),
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
        // A cache reset (new / restarted sequence) reallocates the naive KV buffers, so any captured
        // Vulkan decode graph — which baked the old buffer handles into its recorded commands — is now
        // stale. Drop it; the first decode of the new sequence recaptures against the fresh cache.
        #[cfg(feature = "vulkan")]
        if let Ok(mut state) = self.vulkan_decode_graph.lock() {
            state.entry = None;
        }
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
            Model::Gemma(ref model) => &model.cache,
            Model::MiniMax(ref model) => &model.cache,
            Model::Starcoder2(ref model) => &model.cache,
            Model::Qwen(ref model) => &model.cache,
            Model::Qwen3(ref model) => &model.cache,
            Model::Qwen3MoE(ref model) => &model.cache,
            Model::Qwen35(ref model) => &model.cache,
            Model::Qwen3Next(ref model) => &model.cache,
            Model::Deepseek2(ref model) => &model.cache,
            Model::Deepseek4(ref model) => &model.cache,
            Model::GptOss(ref model) => &model.cache,
            Model::Glm4Moe(ref model) => &model.cache,
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
            Model::Gemma(ref model) => model.device.clone(),
            Model::MiniMax(ref model) => model.device.clone(),
            Model::Starcoder2(ref model) => model.device.clone(),
            Model::Qwen(ref model) => model.device.clone(),
            Model::Qwen3(ref model) => model.device.clone(),
            Model::Qwen3MoE(ref model) => model.device.clone(),
            Model::Qwen35(ref model) => model.device.clone(),
            Model::Qwen3Next(ref model) => model.device.clone(),
            Model::Deepseek2(ref model) => model.device.clone(),
            Model::Deepseek4(ref model) => model.device.clone(),
            Model::GptOss(ref model) => model.device.clone(),
            Model::Glm4Moe(ref model) => model.device.clone(),
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
    fn cleanup_cuda_graphs(&self) {
        #[cfg(feature = "cuda")]
        {
            self.cuda_decode_graph
                .lock()
                .expect("CUDA graph mutex poisoned")
                .entries
                .clear();
        }
        #[cfg(feature = "rocm")]
        {
            self.rocm_decode_graph
                .lock()
                .expect("ROCm graph mutex poisoned")
                .entries
                .clear();
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

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl GGUFPipeline {
    /// Whether the active model variant's decode forward is position-invariant under
    /// graph capture, i.e. it reads its RoPE rotation from the device `metadata.rope_positions`
    /// tensor (refreshed in place between replays) rather than baking a host `seqlen_offset`.
    ///
    /// Qwen3, Qwen3MoE and dense Llama (also Mistral, which shares the llama arch)
    /// qualify: their `forward` honors `metadata.rope_positions` for the seq_len==1
    /// decode step (`RotaryEmbedding::forward_positions` -> on-device cos/sin gather via
    /// `apply_rotary_positions_qk`), so the captured RoPE index advances with the
    /// in-place-refreshed buffer instead of freezing at the warmup position. The other
    /// archs (Phi2/Qwen(2)/Starcoder2/...) still take the host path
    /// `RotaryEmbedding::forward` -> `cos.narrow(0, seqlen_offset, 1)`, a view whose
    /// storage offset is FROZEN at capture and never advances on replay (the classic
    /// frozen-position bug -> garbage from ~token 2); they fall back to the
    /// always-correct eager path. MoE Llama (Mixtral-style) is excluded too — its expert
    /// routing host-syncs (`.to_vec2()`) cannot be captured (see
    /// `QLlama::supports_decode_graph`). XLora / Phi3 are excluded for the same
    /// position-invariance reason or signature mismatch. This is the single
    /// source of truth for decode-graph eligibility, shared by the CUDA and ROCm graph
    /// paths (both capture the identical device-generic model forward).
    fn model_supports_decode_graph(&self) -> bool {
        // Multi-GPU device mapping freezes RoPE under capture: layers on a non-primary device do
        // `positions.to_device()` -> a fresh per-forward tensor that `copy_rope_positions` never
        // refreshes. Restrict capture to single-device so that frozen class is structurally
        // impossible (multi-GPU decode falls back to the always-correct eager path).
        self.mapper.get_unique_devices().len() <= 1
            && match &self.model {
                Model::Qwen3(_) | Model::Qwen3MoE(_) => true,
                // Qwen3.5/3.6 hybrid (Gated-DeltaNet + MoE): capturable now that (a) the full-attn
                // layers run through PagedAttention (position-invariant KV), (b) mRoPE reads the
                // stable `rope_positions` buffer, and (c) the GDN recurrent/conv state is gathered
                // and scattered in place via a constant host slot (no `to_vec1` sync). The baked
                // slot is only valid for a single sequence, so `model_decode_graph_single_seq_only`
                // gates capture to batch == 1.
                Model::Qwen35(_) => true,
                // DeepSeek-V2 / GLM-5.2 (`deepseek2`/`glm-dsa` arch): capture-eligible on the
                // paged decode step. The materialized MLA path (`Some(paged_attn)` branch) is
                // shape-stable at seq_len==1 — projections/splits/cats and the V-pad are all
                // fixed — and its RoPE now reads the device `rope_positions` buffer
                // (`forward_positions`), so the captured rotation advances per replay instead of
                // freezing at the warmup position. The 256-expert MoE routes through the same
                // capture-clean `indexed_moe` matvec Qwen3MoE captures at batch==1 (no host
                // counting sort), and the DSA lightning-indexer is dormant on the paged path
                // (it only fires on the eager non-paged cold-prefill `None` branch). MoE ⇒
                // batch==1 only (`model_decode_graph_single_seq_only`).
                Model::Deepseek2(_) => true,
                // DeepSeek-V4 is capture-eligible (compressor prefill-only → shape-stable
                // decode), and `run_decode_forward` handles it — BUT the decode is currently
                // KERNEL-bound (Q2_K down experts hit the CPU-bound generic MoE fallback,
                // not a resident dp4a kernel), so capture adds warmup overhead with no
                // payoff (measured regression). Re-enable once the Q2_K resident MoE dp4a
                // kernel lands; THEN the graph pays off (→ ds4-class ~24 t/s).
                Model::Llama(model) => model.supports_decode_graph(),
                _ => false,
            }
    }

    /// Runs the decode forward for the graph-eligible variants with the supplied
    /// (rebound) paged metadata. Mirrors the eager dispatch in `forward_inputs`
    /// exactly so the captured forward is identical to what the eager path runs.
    /// Device-agnostic: the captured kernels are the same for CUDA and ROCm.
    fn run_decode_forward(
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
            Model::Qwen3(ref model) => model.forward(
                input_ids,
                seqlen_offsets,
                context_lens,
                &FlashParams::empty(true),
                paged_attn_meta,
            ),
            Model::Qwen3MoE(ref model) => {
                model.forward(input_ids, seqlen_offsets, context_lens, paged_attn_meta)
            }
            Model::Qwen35(ref model) => {
                model.forward(input_ids, seqlen_offsets, context_lens, paged_attn_meta)
            }
            Model::Deepseek2(ref model) => {
                model.forward(input_ids, seqlen_offsets, context_lens, paged_attn_meta)
            }
            Model::Deepseek4(ref model) => {
                model.forward(input_ids, seqlen_offsets, context_lens, paged_attn_meta)
            }
            _ => hanzo_ml::bail!("decode graph: unsupported model variant"),
        }
    }

    /// Variants whose captured decode graph is only replay-valid for batch == 1: Qwen35 bakes the
    /// recurrent-pool slot offset, and MoE variants at batch > 1 route through
    /// `indexed_moe_grouped`, whose host-side counting sort is structurally uncapturable (RED-1 --
    /// capture would bake garbage routing into every replay; ml now bails there, so gating here
    /// keeps batch > 1 MoE decode on the always-correct eager path instead of thrashing capture).
    fn model_decode_graph_single_seq_only(&self) -> bool {
        matches!(
            self.model,
            Model::Qwen35(_) | Model::Qwen3MoE(_) | Model::Deepseek2(_)
        )
    }
}

#[cfg(feature = "cuda")]
impl GGUFPipeline {
    /// Attempts to satisfy a single decode step via a captured CUDA graph. Returns
    /// `Ok(None)` when the graph path does not apply (caller runs eager). The GGUF
    /// analog of `NormalPipeline::try_cuda_decode_graph_forward`, structured like the
    /// ROCm path below: on a capture failure it disables the path and returns the
    /// already-correct warmup logits instead of propagating (no lost/recomputed token).
    fn try_cuda_decode_graph_forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
        kv_cache: &[(Tensor, Tensor)],
        metadata: &PagedAttentionInputMetadata,
    ) -> Result<Option<Tensor>, hanzo_ml::Error> {
        if !cuda_decode_graphs_enabled() || !self.model_supports_decode_graph() {
            return Ok(None);
        }
        if self.draft_proposer.is_some() {
            return Ok(None);
        }
        // Only steady-state single-token decode with paged metadata present and no
        // prefix-cache prefill in flight. Prefill stays eager: its causal mask and
        // data-dependent grouped-MoE routing are not graph-capturable, so prefill
        // parity is a kernel-fusion problem, not a graph-capture one.
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
            || !input_ids.device().is_cuda()
            || (self.model_decode_graph_single_seq_only() && batch != 1)
        {
            return Ok(None);
        }
        let Some(cache_config) = self.metadata.cache_config.as_ref() else {
            return Ok(None);
        };
        let block_size = cache_config.block_size;
        let key = CudaDecodeGraphKey::new(input_ids, metadata, block_size)?;

        let mut state = self
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
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
            state.entries.push(entry);
            return Ok(Some(logits));
        }

        // Cache miss: run a real (eager) warmup forward first so the caller gets a
        // correct first token, then capture a graph for subsequent tokens. Hold the
        // HtoD cache guard across both warmup and capture so any host->device copies
        // in the forward use stable staging the captured graph can replay against
        // (mirrors `NormalPipeline::try_cuda_decode_graph_forward`).
        let Device::Cuda(cuda_device) = input_ids.device() else {
            return Ok(None);
        };
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();

        let warmup_logits = self.run_decode_forward(
            input_ids,
            seqlen_offsets,
            context_lens.to_vec(),
            Some((kv_cache.to_vec(), metadata)),
        )?;
        input_ids.device().synchronize()?;

        match self.capture_cuda_decode_graph(
            key,
            input_ids,
            seqlen_offsets,
            context_lens,
            kv_cache,
            metadata,
            block_size,
        ) {
            Ok(entry) => {
                if state.entries.len() >= CUDA_DECODE_GRAPH_CACHE_CAPACITY {
                    state.entries.remove(0);
                }
                state.entries.push(entry);
            }
            Err(err) => {
                if !state.disabled {
                    warn!("CUDA decode graph capture failed; falling back to eager decode: {err}");
                }
                state.disabled = true;
                state.entries.clear();
            }
        }
        Ok(Some(warmup_logits))
    }

    /// Captures the decode forward into a CUDA graph against stable metadata buffers.
    /// Mirrors `NormalPipeline::capture_cuda_decode_graph` but dispatches the GGUF
    /// `run_decode_forward` (no position_ids/flash_meta) instead of the safetensors
    /// `ModelForwardContext`. The CUDA backend has a graph-ordered allocator, so unlike
    /// the ROCm path this needs no manual pool-reservation scope (the AUTO_FREE_ON_LAUNCH
    /// instantiate flag handles per-launch temporaries).
    #[allow(clippy::too_many_arguments)]
    fn capture_cuda_decode_graph(
        &self,
        key: CudaDecodeGraphKey,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
        kv_cache: &[(Tensor, Tensor)],
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
    ) -> Result<CudaDecodeGraphEntry, hanzo_ml::Error> {
        use hanzo_ml::cuda_backend::cudarc::driver::sys;

        let input_ids_var = Var::from_tensor(input_ids)?;
        let (metadata_buffers, rebound_metadata) =
            CudaDecodeGraphMetadataBuffers::new(metadata, seqlen_offsets, block_size)?;
        let graph_input_ids = input_ids_var.as_detached_tensor();
        let Device::Cuda(cuda_device) = graph_input_ids.device() else {
            hanzo_ml::bail!("CUDA decode graph expected a CUDA device");
        };
        let stream = cuda_device.cuda_stream();
        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();

        if let Err(err) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            restore_event_tracking_after_capture(&stream, restore_event_tracking);
            return Err(
                hanzo_ml::Error::msg(err.to_string()).context("CUDA graph begin capture failed")
            );
        }

        let logits = match self.run_decode_forward(
            &graph_input_ids,
            seqlen_offsets,
            context_lens.to_vec(),
            Some((kv_cache.to_vec(), &rebound_metadata)),
        ) {
            Ok(logits) => logits,
            Err(err) => {
                end_cuda_capture_discard(&stream);
                restore_event_tracking_after_capture(&stream, restore_event_tracking);
                return Err(err);
            }
        };

        let graph = match CudaGraphHandle::end_capture(&stream) {
            Ok(Some(graph)) => graph,
            Ok(None) => {
                restore_event_tracking_after_capture(&stream, restore_event_tracking);
                return Err(hanzo_ml::Error::msg("CUDA graph capture returned no graph"));
            }
            Err(err) => {
                restore_event_tracking_after_capture(&stream, restore_event_tracking);
                return Err(err);
            }
        };
        restore_event_tracking_after_capture(&stream, restore_event_tracking);

        graph.upload()?;

        Ok(CudaDecodeGraphEntry {
            key,
            graph,
            input_ids: input_ids_var,
            metadata_buffers,
            _metadata: rebound_metadata,
            logits,
        })
    }

    /// Disables the CUDA decode graph fast path after a capture/replay failure and
    /// clears the cache so the pipeline falls back to eager. Mirrors
    /// `NormalPipeline::disable_cuda_decode_graph`.
    fn disable_cuda_decode_graph(&self, err: &hanzo_ml::Error) {
        let mut state = self
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        if !state.disabled {
            warn!("CUDA decode graphs disabled after capture/replay error: {err}");
        }
        state.disabled = true;
        state.entries.clear();
    }

    /// Dense fixed-shape prefill: only the pre-norm dense-attention Qwen3 GGUF variant on a single
    /// device. MoE routing (Qwen3MoE/Qwen35) is a host-side counting sort that cannot be captured;
    /// multi-device freezes RoPE under capture. Both fall through to the always-correct eager path.
    fn model_supports_prefill_graph(&self) -> bool {
        self.mapper.get_unique_devices().len() <= 1 && matches!(self.model, Model::Qwen3(_))
    }

    /// Runs the dense prefill forward for graph-eligible variants with the supplied (rebound) paged
    /// metadata and the real varlen `flash_params`. Mirrors the eager Qwen3 dispatch in
    /// `forward_inputs` so the captured forward is identical to what the eager path runs.
    fn run_prefill_forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        flash_params: &FlashParams,
        paged_attn_meta: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor, hanzo_ml::Error> {
        match self.model {
            Model::Qwen3(ref model) => model.forward(
                input_ids,
                seqlen_offsets,
                context_lens,
                flash_params,
                paged_attn_meta,
            ),
            _ => hanzo_ml::bail!("prefill graph: unsupported model variant"),
        }
    }

    /// Attempts to satisfy a full dense prefill via a captured CUDA graph. Returns `Ok(None)` when
    /// the graph path does not apply (caller runs eager). One `cuGraphLaunch` replaces the ~1.5k
    /// eager launches whose per-op CPU dispatch/alloc gaps hold prefill GPU util at ~83% vs llama's
    /// ~98%. Gated to the offset-0 single-sequence first prompt chunk, where the causal mask, RoPE
    /// positions and varlen cumulative-seqlens are all fixed for the `[1, seq_len]` bucket, so only
    /// the token ids and the KV slot mapping need refreshing between replays. On any capture/replay
    /// error the path disables and returns the already-correct warmup logits (no lost token).
    #[allow(clippy::too_many_arguments)]
    fn try_cuda_prefill_graph_forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
        kv_cache: &[(Tensor, Tensor)],
        metadata: &PagedAttentionInputMetadata,
        flash_params: &FlashParams,
    ) -> Result<Option<Tensor>, hanzo_ml::Error> {
        if !cuda_prefill_graphs_enabled() || !self.model_supports_prefill_graph() {
            return Ok(None);
        }
        if self.draft_proposer.is_some() {
            return Ok(None);
        }
        // Offset-0 first prompt chunk only: the mask/positions/cumulative-seqlens are then fixed for
        // the bucket. Chunked prefill (offset > 0), prefix-cache reuse and graph-disabled requests
        // stay eager.
        if !metadata.is_first_prompt_chunk
            || metadata.disable_cuda_graphs
            || metadata.num_cached_tokens.is_some()
        {
            return Ok(None);
        }
        let (batch, q_len) = input_ids.dims2()?;
        if batch != 1
            || q_len <= 1
            || seqlen_offsets.len() != 1
            || seqlen_offsets[0] != 0
            || context_lens.len() != 1
            || !input_ids.device().is_cuda()
        {
            return Ok(None);
        }
        let Some(cache_config) = self.metadata.cache_config.as_ref() else {
            return Ok(None);
        };
        let block_size = cache_config.block_size;
        let key = CudaDecodeGraphKey::new(input_ids, metadata, block_size)?;

        let mut state = self
            .cuda_prefill_graph
            .lock()
            .expect("CUDA prefill graph mutex poisoned");
        if state.disabled || state.denied.iter().any(|denied| *denied == key) {
            return Ok(None);
        }

        // Cache hit: refresh the stable token-id and slot-mapping buffers in place, replay in one launch.
        if let Some(pos) = state.entries.iter().position(|entry| entry.key == key) {
            let mut entry = state.entries.remove(pos);
            entry.input_ids.set(input_ids)?;
            entry.metadata_buffers.copy_from(metadata, seqlen_offsets)?;
            entry.graph.launch()?;
            let logits = entry.logits.clone();
            state.entries.push(entry);
            return Ok(Some(logits));
        }

        // Cache miss: run a real (eager) warmup prefill so the caller gets correct logits, then
        // capture a graph for subsequent same-bucket prefills. Hold the HtoD cache guard across both
        // so host->device copies (positions, mask, cumulative-seqlens) use stable staging.
        let Device::Cuda(cuda_device) = input_ids.device() else {
            return Ok(None);
        };
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();

        let warmup_logits = self.run_prefill_forward(
            input_ids,
            seqlen_offsets,
            context_lens.to_vec(),
            flash_params,
            Some((kv_cache.to_vec(), metadata)),
        )?;
        input_ids.device().synchronize()?;

        match self.capture_cuda_prefill_graph(
            key.clone(),
            input_ids,
            seqlen_offsets,
            context_lens,
            kv_cache,
            metadata,
            flash_params,
            block_size,
        ) {
            Ok(entry) => {
                // Verify the captured graph reproduces the eager warmup logits bit-for-bit before
                // ever serving a replay: launch it once against the same (unchanged) input buffers
                // and compare. A shape that is not bit-exact is denied and stays eager, so default-on
                // only serves verified-exact graphs. This one extra launch amortizes over all replays.
                match self.verify_prefill_graph(&entry, &warmup_logits) {
                    Ok(div) if div.bit_exact => {
                        info!(
                            "CUDA prefill graph verified bit-exact for [1,{q_len}] ({} logits, argmax {})",
                            div.total, div.argmax_eager
                        );
                        if state.entries.len() >= CUDA_DECODE_GRAPH_CACHE_CAPACITY {
                            state.entries.remove(0);
                        }
                        state.entries.push(entry);
                    }
                    Ok(div) => {
                        warn!(
                            "CUDA prefill graph NOT bit-exact for [1,{q_len}]: {}/{} logits differ, max|Δ|={:.3e}, argmax eager={} replay={}; denying bucket (eager fallback)",
                            div.mismatched, div.total, div.max_abs_diff, div.argmax_eager, div.argmax_replay
                        );
                        state.denied.push(key);
                    }
                    Err(err) => {
                        warn!("CUDA prefill graph verification failed for [1,{q_len}]; denying bucket: {err}");
                        state.denied.push(key);
                    }
                }
            }
            Err(err) => {
                if !state.disabled {
                    warn!(
                        "CUDA prefill graph capture failed; falling back to eager prefill: {err}"
                    );
                }
                state.disabled = true;
                state.entries.clear();
            }
        }
        Ok(Some(warmup_logits))
    }

    /// Launches a freshly-captured prefill graph once (inputs unchanged from the warmup) and returns
    /// the bitwise eager-vs-replay logits comparison. A synchronize bounds the replay so the readback
    /// sees the completed graph.
    fn verify_prefill_graph(
        &self,
        entry: &CudaPrefillGraphEntry,
        warmup_logits: &Tensor,
    ) -> Result<PrefillGraphDivergence, hanzo_ml::Error> {
        entry.graph.launch()?;
        warmup_logits.device().synchronize()?;
        prefill_graph_divergence(warmup_logits, &entry.logits)
    }

    /// Captures the dense prefill forward into a CUDA graph against stable input-id and metadata
    /// buffers. Mirrors `capture_cuda_decode_graph` but dispatches `run_prefill_forward` with the
    /// varlen `flash_params`, whose cumulative-seqlens device tensors are held live in the entry.
    #[allow(clippy::too_many_arguments)]
    fn capture_cuda_prefill_graph(
        &self,
        key: CudaDecodeGraphKey,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
        kv_cache: &[(Tensor, Tensor)],
        metadata: &PagedAttentionInputMetadata,
        flash_params: &FlashParams,
        block_size: usize,
    ) -> Result<CudaPrefillGraphEntry, hanzo_ml::Error> {
        use hanzo_ml::cuda_backend::cudarc::driver::sys;

        let input_ids_var = Var::from_tensor(input_ids)?;
        let (metadata_buffers, rebound_metadata) =
            CudaDecodeGraphMetadataBuffers::new(metadata, seqlen_offsets, block_size)?;
        let graph_input_ids = input_ids_var.as_detached_tensor();
        let Device::Cuda(cuda_device) = graph_input_ids.device() else {
            hanzo_ml::bail!("CUDA prefill graph expected a CUDA device");
        };
        let stream = cuda_device.cuda_stream();
        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();

        if let Err(err) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            restore_event_tracking_after_capture(&stream, restore_event_tracking);
            return Err(hanzo_ml::Error::msg(err.to_string())
                .context("CUDA prefill graph begin capture failed"));
        }

        let logits = match self.run_prefill_forward(
            &graph_input_ids,
            seqlen_offsets,
            context_lens.to_vec(),
            flash_params,
            Some((kv_cache.to_vec(), &rebound_metadata)),
        ) {
            Ok(logits) => logits,
            Err(err) => {
                end_cuda_capture_discard(&stream);
                restore_event_tracking_after_capture(&stream, restore_event_tracking);
                return Err(err);
            }
        };

        let graph = match CudaGraphHandle::end_capture(&stream) {
            Ok(Some(graph)) => graph,
            Ok(None) => {
                restore_event_tracking_after_capture(&stream, restore_event_tracking);
                return Err(hanzo_ml::Error::msg(
                    "CUDA prefill graph capture returned no graph",
                ));
            }
            Err(err) => {
                restore_event_tracking_after_capture(&stream, restore_event_tracking);
                return Err(err);
            }
        };
        restore_event_tracking_after_capture(&stream, restore_event_tracking);

        graph.upload()?;

        Ok(CudaPrefillGraphEntry {
            key,
            graph,
            input_ids: input_ids_var,
            metadata_buffers,
            _metadata: rebound_metadata,
            _flash_params: flash_params.clone(),
            logits,
        })
    }

    /// Disables the CUDA prefill graph fast path after a capture/replay failure and clears the
    /// cache so the pipeline falls back to eager.
    fn disable_cuda_prefill_graph(&self, err: &hanzo_ml::Error) {
        let mut state = self
            .cuda_prefill_graph
            .lock()
            .expect("CUDA prefill graph mutex poisoned");
        if !state.disabled {
            warn!("CUDA prefill graphs disabled after capture/replay error: {err}");
        }
        state.disabled = true;
        state.entries.clear();
    }
}

#[cfg(feature = "rocm")]
impl GGUFPipeline {
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
        if !rocm_decode_graphs_enabled() || !self.model_supports_decode_graph() {
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
            || (self.model_decode_graph_single_seq_only() && batch != 1)
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
        let warmup_logits = self.run_decode_forward(
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
        let logits = match self.run_decode_forward(
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

#[cfg(feature = "vulkan")]
impl GGUFPipeline {
    /// Attempts to satisfy a single Vulkan decode step via the captured command-graph. Returns
    /// `Ok(None)` when the graph path does not apply (caller runs eager). Unlike the paged ROCm/CUDA
    /// analogs this drives the model's NAIVE KV cache directly (no paged metadata): each replay
    /// refreshes exactly four buffers in place — the input token, the RoPE/KV-slot position, and the
    /// attended span `seq_k` — and re-submits the recorded forward once. Fail-closed: on any
    /// capture/replay error the path disables and the always-correct eager decode continues.
    fn try_vulkan_decode_graph_forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
    ) -> Result<Option<Tensor>, hanzo_ml::Error> {
        if !crate::perf_flags::vulkan_graphs_enabled() || self.draft_proposer.is_some() {
            return Ok(None);
        }
        // Steady-state single-token decode on a Vulkan device only. Prefill (q_len > 1) stays eager.
        let (batch, q_len) = input_ids.dims2()?;
        if q_len != 1 || batch != 1 || seqlen_offsets.len() != 1 || context_lens.len() != 1 {
            return Ok(None);
        }
        if !matches!(input_ids.device(), Device::Vulkan(_)) {
            return Ok(None);
        }
        // The naive-KV-cache decode graph is wired for the Q4_K / E=128 Qwen3-MoE decode path.
        let Model::Qwen3MoE(ref model) = self.model else {
            return Ok(None);
        };

        let position = seqlen_offsets[0];
        let seq_k = position + 1;

        let mut state = self
            .vulkan_decode_graph
            .lock()
            .expect("Vulkan graph mutex poisoned");
        if state.disabled {
            return Ok(None);
        }

        // Cache hit: refresh the four stable buffers in place and replay in one submit.
        if let Some(mut entry) = state.entry.take() {
            // Replay only a STRICTLY sequential continuation of the captured sequence on the same KV
            // buffers: the next position must be exactly the one the graph is primed for, and still
            // within the captured capacity (a grow reallocates the cache buffer the graph baked in).
            // Any mismatch means a reset / different sequence — drop the graph (entry falls out of
            // scope here, NOT put back) and fall back to eager; the next cache-miss recaptures.
            if position == entry.next_position && position < entry.capacity {
                entry.input_ids.set(input_ids)?;
                let pos_t = Tensor::from_vec(vec![position as u32], (1,), input_ids.device())?;
                entry.positions.set(&pos_t)?;
                entry.attn.set_seq_k(seq_k)?;
                // The `set` refreshes record device copies into the eager batch (input token, position);
                // drain them (and any staged meta upload) so the replay — a SEPARATE queue submit —
                // reads the fresh values, not the stale capture-time buffers. Without this the graph
                // replays against last token's inputs: fluent but STALE output.
                input_ids.device().synchronize()?;
                entry.graph.replay()?;
                let logits = entry.logits.clone();
                // Advance the host KV length so the cache stays consistent for a later eager fallback
                // or capacity-boundary recapture (the graph did the on-device write to slot `position`).
                model.vk_advance_kv_len(seq_k)?;
                entry.next_position = position + 1;
                state.entry = Some(entry);
                return Ok(Some(logits));
            }
            return Ok(None);
        }

        // Cache miss: run an eager warmup forward first (correct first token + cache populate), then
        // capture a graph for the subsequent tokens. If capture fails we keep the warmup logits and
        // disable the path (fail-closed), so no token is lost or recomputed.
        let warmup_logits = model.forward(input_ids, seqlen_offsets, context_lens.to_vec(), None)?;
        input_ids.device().synchronize()?;

        match self.capture_vulkan_decode_graph(input_ids, seqlen_offsets, context_lens, position) {
            Ok(entry) => {
                tracing::info!(
                    "vulkan decode graph: captured {} dispatches at position {} (KV capacity {}); replaying subsequent tokens",
                    entry.graph.n_dispatch(),
                    position,
                    entry.capacity
                );
                state.entry = Some(entry);
            }
            Err(err) => {
                if !state.disabled {
                    warn!("Vulkan decode graph capture failed; falling back to eager decode: {err}");
                }
                state.disabled = true;
                state.entry = None;
            }
        }
        Ok(Some(warmup_logits))
    }

    /// Captures the naive-KV-cache decode forward into a replayable Vulkan command-graph against stable
    /// per-token buffers. Mirrors `capture_rocm_decode_graph` but binds this pipeline's own scale/meta
    /// attention buffers and a shared position buffer for the device-offset KV append. On any error the
    /// in-flight capture is aborted so the eager path can resume on a clean submitter.
    fn capture_vulkan_decode_graph(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
        position: usize,
    ) -> Result<VulkanDecodeGraphEntry, hanzo_ml::Error> {
        let Model::Qwen3MoE(ref model) = self.model else {
            hanzo_ml::bail!("vulkan decode graph: unsupported model variant");
        };
        let Device::Vulkan(vdev) = input_ids.device().clone() else {
            hanzo_ml::bail!("vulkan decode graph: input not on a vulkan device");
        };
        let capacity = model.vk_kv_capacity().ok_or_else(|| {
            hanzo_ml::Error::msg("vulkan decode graph: KV cache not allocated after warmup")
        })?;

        // Stable per-token inputs: the token id (embedding gather reads it) and the u32 position
        // (RoPE + device-offset KV append read it). Refreshed in place before each replay.
        let input_ids_var = Var::from_tensor(input_ids)?;
        let pos_t = Tensor::from_vec(vec![position as u32], (1,), input_ids.device())?;
        let positions_var = Var::from_tensor(&pos_t)?;
        // Shared attention buffers; seq_k is the initial attended span, advanced per replay.
        let attn = model.vk_build_graph_attn(position + 1)?;

        vdev.begin_graph_capture()?;
        let logits = match model.forward_vk_graph(
            &input_ids_var.as_detached_tensor(),
            seqlen_offsets,
            context_lens.to_vec(),
            positions_var.as_tensor(),
            &attn,
        ) {
            Ok(logits) => logits,
            Err(err) => {
                vdev.abort_graph_capture();
                return Err(err);
            }
        };
        let graph = vdev.end_graph_capture()?;

        Ok(VulkanDecodeGraphEntry {
            graph,
            attn,
            input_ids: input_ids_var,
            positions: positions_var,
            logits,
            capacity,
            // The graph is captured for `position`; the first replay is the next token in sequence.
            next_position: position + 1,
        })
    }

    /// Disables the Vulkan decode graph fast path after a capture/replay failure and drops the captured
    /// graph so the pipeline falls back to the always-correct eager decode.
    fn disable_vulkan_decode_graph(&self, err: &hanzo_ml::Error) {
        let mut state = self
            .vulkan_decode_graph
            .lock()
            .expect("Vulkan graph mutex poisoned");
        if !state.disabled {
            warn!("Vulkan decode graphs disabled after capture/replay error: {err}");
        }
        state.disabled = true;
        state.entry = None;
    }
}

#[async_trait::async_trait]
impl Pipeline for GGUFPipeline {
    fn pipeline_parallel_worker(&self) -> Result<(), hanzo_ml::Error> {
        loop {
            let cont = match self.model {
                Model::Llama(ref m) => pp_worker_step(m)?,
                Model::Qwen3(ref m) => pp_worker_step(m)?,
                Model::Qwen3MoE(ref m) => pp_worker_step(m)?,
                Model::Qwen3Next(ref m) => pp_worker_step(m)?,
                Model::Deepseek2(ref m) => pp_worker_step(m)?,
                _ => {
                    hanzo_ml::bail!("pipeline parallelism is not wired for this GGUF architecture")
                }
            };
            if !cont {
                break;
            }
        }
        Ok(())
    }

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
        // Vulkan decode command-graph fast path. The Vulkan naive-KV-cache decode has no paged
        // metadata (so the ROCm/CUDA blocks below never fire); this replays the captured single-token
        // forward (or captures it on the first decode step) instead of re-recording ~1.7k dispatches
        // each token. Gated to single-token decode internally; fail-closed to eager on any error.
        #[cfg(feature = "vulkan")]
        {
            match self.try_vulkan_decode_graph_forward(
                &input_ids,
                &seqlen_offsets,
                &context_lens,
            ) {
                Ok(Some(logits)) => {
                    return if return_raw_logits {
                        Ok(ForwardInputsResult::RawLogits { logits })
                    } else {
                        Ok(ForwardInputsResult::CausalGeneration { logits })
                    };
                }
                Ok(None) => {}
                Err(err) => self.disable_vulkan_decode_graph(&err),
            }
        }
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
        // CUDA decode graph fast path. Same contract as the ROCm block above: on a
        // steady-state single-token decode with paged metadata present, replay (or
        // capture) the graph instead of re-launching ~hundreds of kernels eagerly.
        // On any capture/replay error, disable the path and fall through to eager so
        // a stale-state replay can never silently corrupt tokens.
        #[cfg(feature = "cuda")]
        {
            if let Some((ref kv_cache, meta)) = paged_attn_meta {
                match self.try_cuda_decode_graph_forward(
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
                    Err(err) => self.disable_cuda_decode_graph(&err),
                }
            }
        }
        // CUDA dense-prefill graph fast path. On an offset-0 single-sequence first prompt chunk,
        // replay (or capture) the whole prefill as one graph launch instead of re-dispatching ~1.5k
        // eager kernels whose per-op CPU cost starves the GPU. Same fail-closed contract as decode:
        // on any capture/replay error, disable and fall through to eager.
        #[cfg(feature = "cuda")]
        {
            if let Some((ref kv_cache, meta)) = paged_attn_meta {
                match self.try_cuda_prefill_graph_forward(
                    &input_ids,
                    &seqlen_offsets,
                    &context_lens,
                    kv_cache,
                    meta,
                    &flash_meta,
                ) {
                    Ok(Some(logits)) => {
                        return if return_raw_logits {
                            Ok(ForwardInputsResult::RawLogits { logits })
                        } else {
                            Ok(ForwardInputsResult::CausalGeneration { logits })
                        };
                    }
                    Ok(None) => {}
                    Err(err) => self.disable_cuda_prefill_graph(&err),
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
            Model::Gemma(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::MiniMax(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Starcoder2(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, paged_attn_meta)?
            }
            Model::Qwen(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Qwen3(ref model) => model.forward(
                &input_ids,
                &seqlen_offsets,
                context_lens,
                &flash_meta,
                paged_attn_meta,
            )?,
            Model::Qwen3MoE(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Qwen35(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Qwen3Next(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Deepseek2(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Deepseek4(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::GptOss(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::Glm4Moe(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
        };
        if return_raw_logits {
            Ok(ForwardInputsResult::RawLogits { logits })
        } else {
            Ok(ForwardInputsResult::CausalGeneration { logits })
        }
    }
    fn has_active_speculative_proposer(&self) -> bool {
        self.draft_proposer.is_some()
    }

    fn attach_speculative(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
    ) -> Result<(), hanzo_ml::Error> {
        match config {
            crate::speculative::SpeculativeConfig::Off => Ok(()),
            crate::speculative::SpeculativeConfig::Dspark { .. } => {
                hanzo_ml::bail!(
                    "DSpark speculative decoding targets the safetensors (normal) Qwen3 pipeline, not GGUF."
                );
            }
            crate::speculative::SpeculativeConfig::DraftModel { draft, gamma } => {
                if self.metadata.cache_engine.is_none() {
                    hanzo_ml::bail!(
                        "draft-model speculative decoding currently requires PagedAttention for this pipeline."
                    );
                }
                {
                    let target_tok = self.tokenizer().ok_or_else(|| {
                        hanzo_ml::Error::msg(
                            "target pipeline has no tokenizer for speculative decoding",
                        )
                    })?;
                    let draft_guard = draft.try_lock().map_err(|_| {
                        hanzo_ml::Error::msg("draft pipeline is not exclusively owned")
                    })?;
                    let draft_tok = draft_guard.tokenizer().ok_or_else(|| {
                        hanzo_ml::Error::msg(
                            "draft pipeline has no tokenizer for speculative decoding",
                        )
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
                self.draft_proposer = Some(Box::new(proposer));
                Ok(())
            }
            crate::speculative::SpeculativeConfig::Mtp(mtp_config) => {
                // Self-speculative MTP: the model provides its OWN draft head. This is
                // the ONE seam — the pipeline asks the model for its SelfSpeculative
                // capability and never names an architecture. A model without an MTP
                // head is reported honestly instead of silently unsupported.
                let n_predict = mtp_config.n_predict.unwrap_or(1);
                let proposer = self
                    .model
                    .as_self_speculative()
                    .ok_or_else(|| {
                        hanzo_ml::Error::msg(
                            "this model has no MTP head for self-speculative decoding",
                        )
                    })?
                    .attach_mtp(&mtp_config)?;
                let info =
                    crate::speculative::SpeculativeAttachInfo::mtp("mtp".to_string(), n_predict);
                crate::speculative::logging::log_attach(&info);
                self.draft_proposer = Some(proposer);
                Ok(())
            }
        }
    }

    fn retain_speculative_seqs(&mut self, live: &[usize]) {
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

        // Non-paged path (DeepSeek-V4 MTP): drive the SAME generic loop over the normal
        // KV backend. Extract the shared cache handle first so the model borrow is
        // dropped before the &mut-self driver call.
        let normal_cache = match self.model {
            Model::Deepseek4(ref model) => {
                model.normal_cache_arc().map(|arc| (arc, model.max_seq_len))
            }
            // GLM-5.2 (`glm-dsa`) loads as Deepseek2 and self-drafts through its in-band
            // `nextn` MTP head over the SAME normal KV backend.
            Model::Deepseek2(ref model) => {
                model.normal_cache_arc().map(|arc| (arc, model.max_seq_len))
            }
            _ => None,
        };
        if let Some((cache_arc, max_seq_len)) = normal_cache {
            let cache = crate::speculative::cache::NormalSpeculativeCacheAccess::new(
                cache_arc,
                max_seq_len,
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
        self.draft_proposer.as_ref().map(|p| p.proposal_len())
    }

    fn speculative_target_hiddens(
        &self,
        rows: &[(usize, usize)],
    ) -> hanzo_ml::Result<Option<Tensor>> {
        if self.draft_proposer.is_none() || rows.is_empty() {
            return Ok(None);
        }
        let hidden = match self.model {
            Model::Deepseek4(ref model) => model.last_spec_hidden(),
            // GLM-5.2 (`glm-dsa`) loads as Deepseek2; same MTP spec-hidden stash.
            Model::Deepseek2(ref model) => model.last_spec_hidden(),
            _ => return Ok(None),
        };
        let Some(hidden) = hidden else {
            return Ok(None);
        };
        // The forward stashed the per-row post-norm hidden ([batch, rows, H], aligned
        // with the extract_logits rows). Gather the (seq, row) the driver asked for
        // (row = accepted_drafts, the position to draft from) → [n, 1, H].
        let mut gathered = Vec::with_capacity(rows.len());
        for &(b, r) in rows {
            let h = match hidden.dims().len() {
                3 => hidden.narrow(0, b, 1)?.narrow(1, r, 1)?,
                2 => hidden.narrow(0, r, 1)?.unsqueeze(0)?,
                _ => hanzo_ml::bail!("unexpected speculative hidden shape {:?}", hidden.dims()),
            };
            gathered.push(h);
        }
        Ok(Some(Tensor::cat(&gathered, 0)?))
    }

    fn speculative_propose(
        &mut self,
        ctx: crate::speculative::SpeculativeProposeBatchCtx<'_>,
    ) -> hanzo_ml::Result<Option<crate::speculative::SpeculativeProposalBatch>> {
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
