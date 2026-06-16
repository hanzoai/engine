//! Canonical inference + embedding facade consumed by the Hanzo node and the
//! EVM precompiles `0x0201` (AI inference) / `0x0202` (AI embedding).
//!
//! The node never touches the low-level pipeline/scheduler machinery. It loads
//! one [`MistralEngine`] from a model path or HF repo, registers it in the
//! process-global registry, and the precompiles fetch it back to run real
//! generation and embedding. The loader mirrors the CLI's single-model build
//! path (`HanzoForServerBuilder::build_single_model`) so Metal + GGUF/quant
//! work exactly as in `hanzo run`.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use either::Either;
use indexmap::IndexMap;
use std::sync::{OnceLock, RwLock};
use tokio::sync::mpsc::channel;

use candle_core::Device;

use crate::{
    get_auto_device_map_params, get_model_dtype, paged_attn_supported, AutoDeviceMapParams,
    Constraint, DefaultSchedulerMethod, DeviceMapSetting, Hanzo, HanzoBuilder, Loader,
    LoaderBuilder, MemoryGpuConfig, MessageContent, ModelSelected, NormalRequest,
    PagedAttentionConfig, PagedCacheType, Request, RequestMessage, ResponseOk, SamplingParams,
    SchedulerConfig, TokenSource,
};

const DEFAULT_MAX_SEQS: usize = 16;
const DEFAULT_PREFIX_CACHE_N: usize = 16;
const GGUF_EXT: &str = "gguf";
/// Hard cap on generated tokens. Bounds a precompile call so it can't run
/// unboundedly if the model never emits EOS.
const DEFAULT_MAX_GEN_TOKENS: usize = 1024;

fn deterministic_sampling() -> SamplingParams {
    SamplingParams {
        max_len: Some(DEFAULT_MAX_GEN_TOKENS),
        ..SamplingParams::deterministic()
    }
}

/// Trait implemented by the canonical engine for text generation. Queried by
/// the inference precompile `0x0201`.
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Generate a completion for `prompt`, returning the assistant text.
    async fn generate(&self, prompt: &str) -> Result<String>;
}

/// Trait implemented by the canonical engine for embeddings. Queried by the
/// embedding precompile `0x0202`.
#[async_trait]
pub trait EmbeddingEngine: Send + Sync {
    /// Embed `text`, returning the raw embedding vector.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Canonical engine facade wrapping a fully-loaded [`Hanzo`] instance.
///
/// One instance serves both inference and embedding surfaces; whether
/// embeddings are available depends on the loaded model supporting them.
pub struct MistralEngine {
    hanzo: Arc<Hanzo>,
    model_id: String,
}

impl MistralEngine {
    /// Load a model from a local path: either a directory containing
    /// safetensors + config (loaded via the auto loader) or a `.gguf` file.
    /// Uses the same code path as `hanzo run`, so Metal + quantization apply.
    pub async fn from_model_path(path: &str) -> Result<Self> {
        let model = model_selected_from_path(path);
        Self::from_model_selected(model, path.to_string()).await
    }

    /// Load a model from a Hugging Face repo id (e.g. `Qwen/Qwen3-0.6B`) via
    /// the auto loader. Weights are downloaded and cached by `hf-hub`.
    pub async fn from_hf_repo(repo: &str) -> Result<Self> {
        let model = run_model_selected(repo.to_string());
        Self::from_model_selected(model, repo.to_string()).await
    }

    /// Shared loader: reproduces `HanzoForServerBuilder::build_single_model`
    /// using only engine-internal symbols (no `hanzo-server-core` dep).
    async fn from_model_selected(model: ModelSelected, model_id: String) -> Result<Self> {
        let dtype = get_model_dtype(&model)?;
        let auto_device_map_params = get_auto_device_map_params(&model)?;

        let device = default_device()?;
        let mapper = DeviceMapSetting::Auto(auto_device_map_params);

        // PagedAttention defaults: off on Metal/CPU, on for CUDA. We never
        // force it on here, matching the CLI's default behavior.
        let paged_attn = !device.is_cpu() && device.is_cuda();
        let cache_config = init_cache_config(device.is_cuda(), !paged_attn)?;

        let loader: Box<dyn Loader> = LoaderBuilder::new(model).build()?;

        let pipeline = loader.load_model_from_hf(
            None,
            TokenSource::CacheToken,
            &dtype,
            &device,
            false,
            mapper,
            None,
            cache_config,
        )?;

        let scheduler_config = match &cache_config {
            Some(_) => match pipeline.lock().await.get_metadata().cache_config.clone() {
                Some(cache_config) => SchedulerConfig::PagedAttentionMeta {
                    max_num_seqs: DEFAULT_MAX_SEQS,
                    config: cache_config,
                },
                None => default_scheduler(),
            },
            None => default_scheduler(),
        };

        let hanzo = HanzoBuilder::new(pipeline, scheduler_config, false, None)
            .with_no_kv_cache(false)
            .with_prefix_cache_n(DEFAULT_PREFIX_CACHE_N)
            .build()
            .await;

        Ok(Self { hanzo, model_id })
    }

    /// The id (path or repo) this engine was loaded from.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Underlying [`Hanzo`] handle, for callers needing the full request API.
    pub fn hanzo(&self) -> &Arc<Hanzo> {
        &self.hanzo
    }
}

#[async_trait]
impl InferenceEngine for MistralEngine {
    async fn generate(&self, prompt: &str) -> Result<String> {
        let sender = self
            .hanzo
            .get_sender(None)
            .map_err(|e| anyhow::anyhow!("engine sender unavailable: {e}"))?;

        let mut user_message: IndexMap<String, MessageContent> = IndexMap::new();
        user_message.insert("role".to_string(), Either::Left("user".to_string()));
        user_message.insert("content".to_string(), Either::Left(prompt.to_string()));

        let (tx, mut rx) = channel(10_000);
        let req = Request::Normal(Box::new(NormalRequest {
            id: self.hanzo.next_request_id(),
            messages: RequestMessage::Chat {
                messages: vec![user_message],
                enable_thinking: Some(false),
                reasoning_effort: None,
            },
            sampling_params: deterministic_sampling(),
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            constraint: Constraint::None,
            suffix: None,
            tool_choice: None,
            tools: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
            enable_code_execution: false,
            code_execution_permission: None,
            code_execution_approval_notifier: None,
            agent_permission: None,
            agent_approval_handler: None,
            agent_approval_notifier: None,
            session_id: None,
            max_tool_rounds: None,
            tool_dispatch_url: None,
            model_id: None,
            truncate_sequence: false,
            files: None,
        }));

        sender
            .send(req)
            .await
            .map_err(|e| anyhow::anyhow!("engine request send failed: {e}"))?;

        let response = rx
            .recv()
            .await
            .context("engine closed the response channel without replying")?;

        match response.as_result().map_err(|e| anyhow::anyhow!("{e}"))? {
            ResponseOk::Done(resp) => {
                let choice = resp
                    .choices
                    .into_iter()
                    .next()
                    .context("engine returned no choices")?;
                // Reasoning models split chain-of-thought into `reasoning_content`
                // and the user-facing answer into `content`; fall back to reasoning
                // only if the final content is empty.
                let text = match choice.message.content {
                    Some(c) if !c.is_empty() => c,
                    _ => choice.message.reasoning_content.unwrap_or_default(),
                };
                Ok(text)
            }
            other => anyhow::bail!("unexpected response variant for chat: {other:?}"),
        }
    }
}

#[async_trait]
impl EmbeddingEngine for MistralEngine {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let sender = self
            .hanzo
            .get_sender(None)
            .map_err(|e| anyhow::anyhow!("engine sender unavailable: {e}"))?;

        let (tx, mut rx) = channel(10_000);
        let req = Request::Normal(Box::new(NormalRequest {
            id: self.hanzo.next_request_id(),
            messages: RequestMessage::Embedding {
                prompt: text.to_string(),
            },
            sampling_params: SamplingParams::deterministic(),
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            constraint: Constraint::None,
            suffix: None,
            tool_choice: None,
            tools: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
            enable_code_execution: false,
            code_execution_permission: None,
            code_execution_approval_notifier: None,
            agent_permission: None,
            agent_approval_handler: None,
            agent_approval_notifier: None,
            session_id: None,
            max_tool_rounds: None,
            tool_dispatch_url: None,
            model_id: None,
            truncate_sequence: false,
            files: None,
        }));

        sender
            .send(req)
            .await
            .map_err(|e| anyhow::anyhow!("engine request send failed: {e}"))?;

        let response = rx
            .recv()
            .await
            .context("engine closed the response channel without replying")?;

        match response.as_result().map_err(|e| anyhow::anyhow!("{e}"))? {
            ResponseOk::Embeddings { embeddings, .. } => Ok(embeddings),
            other => anyhow::bail!("unexpected response variant for embedding: {other:?}"),
        }
    }
}

fn default_scheduler() -> SchedulerConfig {
    SchedulerConfig::DefaultScheduler {
        method: DefaultSchedulerMethod::Fixed(DEFAULT_MAX_SEQS.try_into().unwrap()),
    }
}

fn default_device() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        Ok(Device::new_metal(0)?)
    }
    #[cfg(all(not(feature = "metal"), feature = "cuda"))]
    {
        Ok(Device::cuda_if_available(0)?)
    }
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    {
        Ok(Device::Cpu)
    }
}

fn init_cache_config(is_cuda: bool, no_paged_attn: bool) -> Result<Option<PagedAttentionConfig>> {
    if no_paged_attn || !paged_attn_supported() || !is_cuda {
        return Ok(None);
    }
    Ok(Some(PagedAttentionConfig::new(
        None,
        MemoryGpuConfig::Utilization(0.9),
        PagedCacheType::Auto,
    )?))
}

fn model_selected_from_path(path: &str) -> ModelSelected {
    let p = std::path::Path::new(path);

    // Direct `.gguf` file.
    if has_gguf_ext(p) {
        return gguf_selected(p);
    }

    // Directory: prefer a bundled GGUF (the proven Metal/quant path) over
    // safetensors, looking in the dir itself then a `gguf/` subdir.
    if p.is_dir() {
        if let Some(gguf) = find_gguf_in_dir(p).or_else(|| find_gguf_in_dir(&p.join("gguf"))) {
            return gguf_selected(&gguf);
        }
    }

    run_model_selected(path.to_string())
}

fn has_gguf_ext(p: &std::path::Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case(GGUF_EXT))
        .unwrap_or(false)
}

/// First `.gguf` file directly inside `dir`, chosen deterministically by name.
fn find_gguf_in_dir(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    if !dir.is_dir() {
        return None;
    }
    let mut candidates: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| has_gguf_ext(p))
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

fn gguf_selected(p: &std::path::Path) -> ModelSelected {
    let dir = p
        .parent()
        .map(|d| d.to_string_lossy().to_string())
        .unwrap_or_else(|| ".".to_string());
    let file = p
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();
    ModelSelected::GGUF {
        tok_model_id: None,
        quantized_model_id: dir,
        quantized_filename: file,
        dtype: Default::default(),
        topology: None,
        max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
        max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
    }
}

fn run_model_selected(model_id: String) -> ModelSelected {
    ModelSelected::Run {
        model_id,
        tokenizer_json: None,
        dtype: Default::default(),
        topology: None,
        organization: None,
        write_uqff: None,
        from_uqff: None,
        imatrix: None,
        calibration_file: None,
        max_edge: None,
        max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
        max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
        max_num_images: None,
        max_image_length: None,
        hf_cache_path: None,
        matformer_config_path: None,
        matformer_slice_name: None,
    }
}

// ---- process-global registry -------------------------------------------------

static INFERENCE_ENGINE: OnceLock<RwLock<Option<Arc<dyn InferenceEngine>>>> = OnceLock::new();
static EMBEDDING_ENGINE: OnceLock<RwLock<Option<Arc<dyn EmbeddingEngine>>>> = OnceLock::new();

fn inference_slot() -> &'static RwLock<Option<Arc<dyn InferenceEngine>>> {
    INFERENCE_ENGINE.get_or_init(|| RwLock::new(None))
}

fn embedding_slot() -> &'static RwLock<Option<Arc<dyn EmbeddingEngine>>> {
    EMBEDDING_ENGINE.get_or_init(|| RwLock::new(None))
}

/// Register `engine` as the process-wide inference engine for precompile
/// `0x0201`. Replaces any previously registered engine.
pub fn register_inference_engine(engine: Arc<dyn InferenceEngine>) -> Result<()> {
    let mut slot = inference_slot()
        .write()
        .map_err(|_| anyhow::anyhow!("inference engine registry lock poisoned"))?;
    *slot = Some(engine);
    Ok(())
}

/// Register `engine` as the process-wide embedding engine for precompile
/// `0x0202`. Replaces any previously registered engine.
pub fn register_embedding_engine(engine: Arc<dyn EmbeddingEngine>) -> Result<()> {
    let mut slot = embedding_slot()
        .write()
        .map_err(|_| anyhow::anyhow!("embedding engine registry lock poisoned"))?;
    *slot = Some(engine);
    Ok(())
}

/// Fetch the registered inference engine, if any.
pub fn inference_engine() -> Option<Arc<dyn InferenceEngine>> {
    inference_slot().read().ok().and_then(|g| g.clone())
}

/// Fetch the registered embedding engine, if any.
pub fn embedding_engine() -> Option<Arc<dyn EmbeddingEngine>> {
    embedding_slot().read().ok().and_then(|g| g.clone())
}
