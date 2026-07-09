//! CLI argument definitions for hanzo-cli
//!
//! This module provides cleanly organized argument structs using clap's derive macros.
//! Arguments are grouped logically to improve discoverability and reduce duplication.

mod model;
mod paged_attn;
mod quantize;
mod sandbox;
mod server;

pub use model::*;
pub use paged_attn::*;
pub use quantize::*;
pub use sandbox::*;
pub use server::*;

use clap::{Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use hanzo_engine::{AnimationLoaderType, TokenSource};
use serde::Deserialize;
use std::path::PathBuf;
use std::str::FromStr;

/// Fast LLM inference engine
#[derive(Parser)]
#[command(name = "hanzo")]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    #[command(flatten)]
    pub global: GlobalOptions,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start HTTP/MCP server and (optionally) the UI at /ui
    Serve {
        #[command(subcommand)]
        model_type: Option<ModelType>,

        /// Default model options (used when model type is not specified)
        #[command(flatten)]
        default_model: DefaultModelOptions,

        #[command(flatten)]
        server: ServerOptions,

        #[command(flatten)]
        runtime: RuntimeOptions,

        #[command(flatten)]
        agent_options: AgentCliOptions,

        #[command(flatten)]
        sandbox: SandboxOptions,
    },

    /// Run model in interactive mode, or one-shot mode with `-i`
    Run {
        #[command(subcommand)]
        model_type: Option<ModelType>,

        /// Default model options (used when model type is not specified)
        #[command(flatten)]
        default_model: DefaultModelOptions,

        #[command(flatten)]
        runtime: RuntimeOptions,

        #[command(flatten)]
        agent_options: AgentCliOptions,

        #[command(flatten)]
        sandbox: SandboxOptions,

        /// Control thinking mode for models that support it.
        /// Use --thinking or --thinking true to force on, --thinking false to force off.
        /// Omit to defer to the chat template default.
        #[arg(long, num_args = 0..=1, default_missing_value = "true", value_parser = clap::value_parser!(bool))]
        thinking: Option<bool>,

        /// One-shot text prompt. When provided, sends a single request and exits
        /// instead of entering interactive mode.
        /// Combine with --image, --video, or --audio for multimodal requests.
        #[arg(short = 'i', long)]
        input: Option<String>,

        /// Image URL(s) or file path(s) to include in the request (requires -i).
        /// Can be specified multiple times: --image img1.jpg --image img2.png
        #[arg(long, requires = "input")]
        image: Vec<String>,

        /// Video URL(s) or file path(s) to include in the request (requires -i).
        /// Can be specified multiple times: --video vid1.mp4 --video vid2.webm
        #[arg(long, requires = "input")]
        video: Vec<String>,

        /// Audio URL(s) or file path(s) to include in the request (requires -i).
        /// Can be specified multiple times: --audio audio1.wav --audio audio2.mp3
        #[arg(long, requires = "input")]
        audio: Vec<String>,
    },

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },

    /// Generate UQFF quantized model file
    Quantize {
        #[command(subcommand)]
        model_type: Option<QuantizeModelType>,

        /// Default quantize options (used when model type is not specified)
        #[command(flatten)]
        default_quantize: QuantizeDefaultOptions,
    },

    /// Run system diagnostics and environment checks
    Doctor {
        /// Output JSON instead of human-readable text
        #[arg(long)]
        json: bool,
    },

    /// Recommend quantization + device mapping for a model.
    /// Rejects `--quant auto`; pass `--quant <level>` or `--isq <level>` to bias
    /// the recommendation toward a specific quantization target.
    Tune {
        #[command(subcommand)]
        model_type: Option<ModelType>,

        /// Default model options (used when model type is not specified)
        #[command(flatten)]
        default_model: DefaultModelOptions,

        /// Tuning profile (quality, balanced, fast)
        #[arg(long, value_enum, default_value = "balanced")]
        profile: TuneProfileArg,

        /// Output JSON instead of human-readable text
        #[arg(long)]
        json: bool,

        /// Emit a TOML config file with the recommended settings
        #[arg(long)]
        emit_config: Option<PathBuf>,
    },

    /// Authenticate with HuggingFace Hub
    Login {
        /// Provide token directly (non-interactive)
        #[arg(long)]
        token: Option<String>,
    },

    /// Manage the HuggingFace model cache
    Cache {
        #[command(subcommand)]
        cmd: CacheCommand,
    },

    /// Run performance benchmarks for plain model generation.
    Bench {
        #[command(subcommand)]
        model_type: Option<ModelType>,

        /// Default model options (used when model type is not specified)
        #[command(flatten)]
        default_model: DefaultModelOptions,

        #[command(flatten)]
        runtime: BenchRuntimeOptions,

        /// Number of tokens in prompt. Accepts comma-separated values for sweeps.
        #[arg(long, value_delimiter = ',', default_value = "512")]
        prompt_len: Vec<usize>,

        /// Number of tokens to generate
        #[arg(long, default_value = "128")]
        gen_len: usize,

        /// Number of prompt tokens to prefill before measuring decode. Accepts comma-separated values for sweeps.
        #[arg(long, value_delimiter = ',', default_value = "4")]
        depth: Vec<usize>,

        /// Number of benchmark iterations
        #[arg(long, default_value = "3")]
        iterations: usize,

        /// Number of warmup runs (discarded)
        #[arg(long, default_value = "1")]
        warmup: usize,
    },

    /// Run from a full TOML configuration file
    #[command(name = "from-config")]
    FromConfig {
        /// Path to configuration file (.toml)
        #[arg(short, long)]
        file: PathBuf,
    },

    /// Link machines into a distributed inference ring (scan-to-join UX over the ring backend)
    Cluster {
        #[command(subcommand)]
        action: ClusterAction,
    },
}

/// `hanzo cluster` actions: `init` starts the head node, `join` adds a worker.
#[derive(Subcommand)]
pub enum ClusterAction {
    /// Head node (rank 0): generate a join token + QR, then launch the ring.
    Init {
        /// Model every node loads (the token pins it so all ranks stay in lockstep).
        #[arg(short = 'm', long)]
        model: String,

        /// Total nodes in the ring. Must be a power of 2 (2, 4, 8, ...), a ring-backend requirement.
        #[arg(short = 'w', long)]
        world_size: usize,

        /// This head's reachable IP (what workers dial). Auto-detected if omitted.
        #[arg(long)]
        bind: Option<String>,

        /// A worker's reachable IP, in rank order. Repeat once per worker to bake the full ring into
        /// the token so workers self-assign their rank. Omit to just print a token.
        #[arg(long)]
        node: Vec<String>,

        /// First port of the ring's contiguous port block.
        #[arg(long, default_value_t = crate::cluster::DEFAULT_BASE_PORT)]
        base_port: u16,

        /// Print the token + config and exit without launching the engine.
        #[arg(long)]
        print: bool,

        /// Extra args forwarded verbatim to `hanzo serve` (pass after `--`).
        #[arg(last = true)]
        engine_args: Vec<String>,
    },

    /// Worker node (rank 1..N-1): decode a token from `init` and launch as a ring daemon.
    Join {
        /// Join token printed by `hanzo cluster init`.
        token: String,

        /// This node's rank. Auto-picked from the token's peer list when omitted.
        #[arg(long)]
        rank: Option<usize>,

        /// Print the config and exit without launching the engine.
        #[arg(long)]
        print: bool,

        /// Extra args forwarded verbatim to `hanzo serve` (pass after `--`).
        #[arg(last = true)]
        engine_args: Vec<String>,
    },
}

/// Cache management subcommands
#[derive(Subcommand, Clone)]
pub enum CacheCommand {
    /// List all cached models
    List,

    /// Delete a specific model from cache
    Delete {
        /// Model ID (e.g., "Qwen/Qwen3-4B")
        #[arg(short = 'm', long)]
        model_id: String,
    },
}

/// Default model options used when no model type subcommand is specified.
/// These mirror the Auto variant's options and are used to construct ModelType::Auto.
#[derive(clap::Args, Clone, Default)]
pub struct DefaultModelOptions {
    /// HuggingFace model ID or local path to model directory
    #[arg(short = 'm', long)]
    pub model_id: Option<String>,

    /// Path to local tokenizer.json file
    #[arg(short = 't', long)]
    pub tokenizer: Option<PathBuf>,

    /// Model architecture (auto-detected if not specified). Text archs use the auto loader;
    /// `musetalk` selects the facial-animation loader.
    #[arg(short = 'a', long, value_parser = parse_default_arch)]
    pub arch: Option<DefaultArch>,

    /// Model data type
    #[arg(long, default_value = "auto", value_parser = parse_dtype)]
    pub dtype: hanzo_engine::ModelDType,

    #[command(flatten)]
    pub format: FormatOptions,

    #[command(flatten)]
    pub adapter: AdapterOptions,

    #[command(flatten)]
    pub quantization: QuantizationOptions,

    #[command(flatten)]
    pub device: DeviceOptions,

    #[command(flatten)]
    pub cache: CacheOptions,

    #[command(flatten)]
    pub multimodal: MultimodalOptions,
}

impl DefaultModelOptions {
    /// Convert default options into a ModelType::Auto variant.
    /// Returns an error if model_id is not provided.
    pub fn into_model_type(self) -> anyhow::Result<ModelType> {
        let model_id = self
            .model_id
            .ok_or_else(|| anyhow::anyhow!("--model-id (-m) is required"))?;
        match self.arch {
            Some(DefaultArch::Animation(arch)) => Ok(ModelType::Animation {
                model_id,
                arch,
                dtype: self.dtype,
                device: self.device,
            }),
            arch => Ok(ModelType::Auto {
                model: ModelSourceOptions {
                    model_id,
                    tokenizer: self.tokenizer,
                    arch: match arch {
                        Some(DefaultArch::Normal(n)) => Some(n),
                        _ => None,
                    },
                    dtype: self.dtype,
                },
                format: self.format,
                adapter: self.adapter,
                quantization: self.quantization,
                device: self.device,
                cache: self.cache,
                multimodal: self.multimodal,
            }),
        }
    }
}

/// Architecture selector for the default (no-subcommand) path: `--arch` names an
/// architecture from any category and the category is derived from the name. Text
/// archs map to the auto loader, `musetalk` to the animation loader.
#[derive(Clone)]
pub enum DefaultArch {
    Normal(hanzo_engine::NormalLoaderType),
    Animation(AnimationLoaderType),
}

impl FromStr for DefaultArch {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok(arch) = s.parse::<AnimationLoaderType>() {
            return Ok(Self::Animation(arch));
        }
        s.parse::<hanzo_engine::NormalLoaderType>()
            .map(Self::Normal)
            .map_err(|e| format!("{e} (or `musetalk` for facial animation)"))
    }
}

/// Get the effective ModelType, using default options if no subcommand was provided.
/// Returns an error if no subcommand is provided and model_id is missing.
pub fn resolve_model_type(
    model_type: Option<ModelType>,
    default_model: DefaultModelOptions,
) -> anyhow::Result<ModelType> {
    match model_type {
        Some(mt) => Ok(mt),
        None => default_model.into_model_type(),
    }
}

fn parse_default_arch(s: &str) -> Result<DefaultArch, String> {
    s.parse()
}

fn parse_animation_arch(s: &str) -> Result<AnimationLoaderType, String> {
    s.parse()
}

fn parse_dtype(s: &str) -> Result<hanzo_engine::ModelDType, String> {
    s.parse()
}

/// Model type selection
#[derive(Subcommand, Clone)]
pub enum ModelType {
    /// Auto-detect model type (recommended)
    Auto {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        format: FormatOptions,

        #[command(flatten)]
        adapter: AdapterOptions,

        #[command(flatten)]
        quantization: QuantizationOptions,

        #[command(flatten)]
        device: DeviceOptions,

        #[command(flatten)]
        cache: CacheOptions,

        #[command(flatten)]
        multimodal: MultimodalOptions,
    },

    /// Text generation model with explicit configuration
    Text {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        format: FormatOptions,

        #[command(flatten)]
        adapter: AdapterOptions,

        #[command(flatten)]
        quantization: QuantizationOptions,

        #[command(flatten)]
        device: DeviceOptions,

        #[command(flatten)]
        cache: CacheOptions,
    },

    /// Multimodal model
    Multimodal {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        format: FormatOptions,

        #[command(flatten)]
        adapter: AdapterOptions,

        #[command(flatten)]
        quantization: QuantizationOptions,

        #[command(flatten)]
        device: DeviceOptions,

        #[command(flatten)]
        cache: CacheOptions,

        #[command(flatten)]
        multimodal: MultimodalOptions,
    },

    /// Image generation model (diffusion)
    Diffusion {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        device: DeviceOptions,
    },

    /// Speech synthesis model
    Speech {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        device: DeviceOptions,
    },

    /// Facial-animation model (lip-sync / avatar). Drive it via the /v1/animate endpoint.
    Animation {
        /// MuseTalk bundle dir (musetalkV15/, sd-vae-ft-mse/, whisper/tiny.safetensors, s3fd.safetensors) or HF repo.
        #[arg(short = 'm', long)]
        model_id: String,

        /// Animation architecture.
        #[arg(short = 'a', long, default_value = "musetalk", value_parser = parse_animation_arch)]
        arch: AnimationLoaderType,

        /// Model data type.
        #[arg(long, default_value = "auto", value_parser = parse_dtype)]
        dtype: hanzo_engine::ModelDType,

        #[command(flatten)]
        device: DeviceOptions,
    },

    /// Embedding model
    Embedding {
        #[command(flatten)]
        model: ModelSourceOptions,

        #[command(flatten)]
        format: FormatOptions,

        #[command(flatten)]
        quantization: QuantizationOptions,

        #[command(flatten)]
        device: DeviceOptions,

        #[command(flatten)]
        cache: CacheOptions,
    },
}

/// Global options that apply to all commands
#[derive(clap::Args, Clone, Deserialize)]
pub struct GlobalOptions {
    /// Random seed for reproducibility
    #[arg(long, global = true)]
    #[serde(default)]
    pub seed: Option<u64>,

    /// Log all requests and responses to this file
    #[arg(long, short, global = true)]
    #[serde(default)]
    pub log: Option<PathBuf>,

    /// Token source for HuggingFace authentication.
    /// Formats: `literal:<token>`, `env:<var>`, `path:<file>`, `cache`, `none`
    #[arg(long, default_value = "cache", global = true, value_parser = parse_token_source)]
    #[serde(default = "default_token_source")]
    pub token_source: TokenSource,

    /// Increase logging verbosity. Use -v for debug and -vv for trace-level internals.
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count)]
    #[serde(default)]
    pub verbose: u8,
}

/// Runtime options for inference
#[derive(clap::Args, Clone, Deserialize)]
pub struct RuntimeOptions {
    /// Maximum concurrent sequences
    #[arg(long, default_value_t = 32)]
    #[serde(default = "default_max_seqs")]
    pub max_seqs: usize,

    /// Disable KV cache entirely
    #[arg(long)]
    #[serde(default)]
    pub no_kv_cache: bool,

    /// Number of prefix caches to hold (0 to disable)
    #[arg(long, default_value_t = 16)]
    #[serde(default = "default_prefix_cache_n")]
    pub prefix_cache_n: usize,

    /// Custom chat template file (.json or .jinja)
    #[arg(long, short)]
    #[serde(default)]
    pub chat_template: Option<PathBuf>,

    /// Explicit JINJA template override
    #[arg(long, short)]
    #[serde(default)]
    pub jinja_explicit: Option<PathBuf>,

    /// Path to a MatFormer config (CSV/JSON describing available slices). See model card.
    #[arg(long)]
    #[serde(default)]
    pub matformer_config_path: Option<PathBuf>,

    /// MatFormer slice to load (must match a slice name in the config file).
    #[arg(long, requires = "matformer_config_path")]
    #[serde(default)]
    pub matformer_slice_name: Option<String>,

    /// MTP assistant model id or path.
    #[arg(long)]
    #[serde(default)]
    pub mtp_model: Option<String>,

    /// Number of MTP draft tokens to propose per target step.
    #[arg(long)]
    #[serde(default)]
    pub mtp_n_predict: Option<usize>,

    /// Draft model id/path for classic draft+target speculative decoding (works with GGUF).
    #[arg(long)]
    #[serde(default)]
    pub draft_model: Option<String>,

    /// Draft GGUF filename (used with --draft-model when the draft is a GGUF model).
    #[arg(long)]
    #[serde(default)]
    pub draft_quantized_file: Option<String>,

    /// Speculative gamma: draft tokens proposed per target verification step (default 4).
    #[arg(long)]
    #[serde(default)]
    pub gamma: Option<usize>,

    /// Path to an MCP client configuration JSON. Also reads `MCP_CONFIG_PATH` if unset.
    #[arg(long)]
    #[serde(default)]
    pub mcp_config: Option<PathBuf>,

    #[arg(skip)]
    #[serde(default)]
    pub agent: bool,

    #[arg(skip)]
    #[serde(default)]
    pub enable_search: bool,

    #[arg(skip)]
    #[serde(default)]
    pub search_embedding_model: Option<SearchEmbeddingModelArg>,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub enable_code_execution: bool,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub code_exec_python: Option<PathBuf>,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub code_exec_timeout: Option<u64>,

    #[cfg(feature = "code-execution")]
    #[arg(skip)]
    #[serde(default)]
    pub code_exec_workdir: Option<PathBuf>,

    #[arg(skip)]
    #[serde(default, rename = "agent_permission", alias = "code_exec_permission")]
    pub code_exec_permission: CodeExecPermissionArg,
}

#[derive(clap::Args, Clone, Default)]
pub struct AgentCliOptions {
    /// Build a local agent: enables web search and Python code execution, runs the agentic
    /// tool loop with a per-session temp workdir. Equivalent to passing
    /// `--enable-search --enable-code-execution` together.
    #[arg(long, alias = "agentic")]
    pub agent: bool,

    /// Enable web search (requires embedding model)
    #[arg(long)]
    pub enable_search: bool,

    /// Search embedding model to use. Requires `--enable-search` or `--agent`.
    #[arg(long)]
    pub search_embedding_model: Option<SearchEmbeddingModelArg>,

    /// Enable Python code execution tool (WARNING: allows arbitrary code execution)
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub enable_code_execution: bool,

    /// Python interpreter path for code execution. Requires code execution to be on
    /// (via `--enable-code-execution` or `--agent`). Defaults to `python3`.
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub code_exec_python: Option<PathBuf>,

    /// Code execution timeout in seconds (default: 30). Requires code execution to be on.
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub code_exec_timeout: Option<u64>,

    /// Working directory for code execution. Defaults to a temp dir; use "." for cwd.
    /// Requires code execution to be on.
    #[cfg(feature = "code-execution")]
    #[arg(long)]
    pub code_exec_workdir: Option<PathBuf>,

    /// Agent action permission mode.
    #[arg(long = "agent-permission", alias = "code-exec-permission", value_enum, default_value_t = CodeExecPermissionArg::Auto)]
    pub code_exec_permission: CodeExecPermissionArg,
}

impl AgentCliOptions {
    pub fn apply_to(self, runtime: &mut RuntimeOptions) {
        runtime.agent = self.agent;
        runtime.enable_search = self.enable_search;
        runtime.search_embedding_model = self.search_embedding_model;
        #[cfg(feature = "code-execution")]
        {
            runtime.enable_code_execution = self.enable_code_execution;
            runtime.code_exec_python = self.code_exec_python;
            runtime.code_exec_timeout = self.code_exec_timeout;
            runtime.code_exec_workdir = self.code_exec_workdir;
        }
        runtime.code_exec_permission = self.code_exec_permission;
    }
}

#[derive(clap::Args, Clone, Default)]
pub struct BenchRuntimeOptions {
    /// Disable KV cache entirely
    #[arg(long)]
    pub no_kv_cache: bool,

    /// Path to a MatFormer config (CSV/JSON describing available slices). See model card.
    #[arg(long)]
    pub matformer_config_path: Option<PathBuf>,

    /// MatFormer slice to load (must match a slice name in the config file).
    #[arg(long, requires = "matformer_config_path")]
    pub matformer_slice_name: Option<String>,

    /// MTP assistant model id or path.
    #[arg(long)]
    pub mtp_model: Option<String>,

    /// Number of MTP draft tokens to propose per target step.
    #[arg(long)]
    pub mtp_n_predict: Option<usize>,

    /// Draft model id/path for classic draft+target speculative decoding (works with GGUF).
    #[arg(long)]
    pub draft_model: Option<String>,

    /// Draft GGUF filename (used with --draft-model when the draft is a GGUF model).
    #[arg(long)]
    pub draft_quantized_file: Option<String>,

    /// Speculative gamma: draft tokens proposed per target verification step (default 4).
    #[arg(long)]
    pub gamma: Option<usize>,
}

impl BenchRuntimeOptions {
    pub fn matformer_selection(&self) -> MatformerSelection {
        MatformerSelection {
            config_path: self.matformer_config_path.clone(),
            slice_name: self.matformer_slice_name.clone(),
        }
    }

    pub fn mtp_config(&self) -> Option<hanzo_engine::MtpConfig> {
        self.mtp_model.clone().map(|model| hanzo_engine::MtpConfig {
            model,
            n_predict: self.mtp_n_predict,
        })
    }

    pub fn draft_model_selected(
        &self,
        max_seq_len: usize,
        max_batch_size: usize,
    ) -> Option<hanzo_engine::ModelSelected> {
        build_draft_model_selected(
            self.draft_model.as_deref(),
            self.draft_quantized_file.as_deref(),
            max_seq_len,
            max_batch_size,
        )
    }

    pub fn gamma(&self) -> usize {
        self.gamma.unwrap_or(DEFAULT_GAMMA)
    }
}

/// Search embedding model options
#[derive(Clone, Copy, ValueEnum, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchEmbeddingModelArg {
    EmbeddingGemma,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CodeExecPermissionArg {
    #[default]
    Auto,
    Ask,
    Deny,
}

/// Tuning profile options
#[derive(Clone, Copy, ValueEnum)]
pub enum TuneProfileArg {
    Quality,
    Balanced,
    Fast,
}

/// Selection of a MatFormer slice (config file + named slice). Used by loaders for
/// models like Gemma 3n that support elastic sizing.
#[derive(Clone, Default)]
pub struct MatformerSelection {
    pub config_path: Option<PathBuf>,
    pub slice_name: Option<String>,
}

impl RuntimeOptions {
    pub fn matformer_selection(&self) -> MatformerSelection {
        MatformerSelection {
            config_path: self.matformer_config_path.clone(),
            slice_name: self.matformer_slice_name.clone(),
        }
    }

    pub fn mtp_config(&self) -> Option<hanzo_engine::MtpConfig> {
        self.mtp_model.clone().map(|model| hanzo_engine::MtpConfig {
            model,
            n_predict: self.mtp_n_predict,
        })
    }

    /// Build the draft model selector for classic draft+target speculative decoding, if a draft was
    /// requested. GGUF draft (the common case): `--draft-model <dir/id>` + `--draft-quantized-file`.
    /// `max_seq_len`/`max_batch_size` are inherited from the target so the draft matches its context.
    pub fn draft_model_selected(
        &self,
        max_seq_len: usize,
        max_batch_size: usize,
    ) -> Option<hanzo_engine::ModelSelected> {
        build_draft_model_selected(
            self.draft_model.as_deref(),
            self.draft_quantized_file.as_deref(),
            max_seq_len,
            max_batch_size,
        )
    }

    pub fn gamma(&self) -> usize {
        self.gamma.unwrap_or(DEFAULT_GAMMA)
    }
}

impl From<TuneProfileArg> for hanzo_engine::TuneProfile {
    fn from(value: TuneProfileArg) -> Self {
        match value {
            TuneProfileArg::Quality => hanzo_engine::TuneProfile::Quality,
            TuneProfileArg::Balanced => hanzo_engine::TuneProfile::Balanced,
            TuneProfileArg::Fast => hanzo_engine::TuneProfile::Fast,
        }
    }
}

impl From<SearchEmbeddingModelArg> for hanzo_engine::SearchEmbeddingModel {
    fn from(value: SearchEmbeddingModelArg) -> Self {
        match value {
            SearchEmbeddingModelArg::EmbeddingGemma => {
                hanzo_engine::SearchEmbeddingModel::EmbeddingGemma300M
            }
        }
    }
}

impl From<CodeExecPermissionArg> for hanzo_engine::CodeExecutionPermission {
    fn from(value: CodeExecPermissionArg) -> Self {
        match value {
            CodeExecPermissionArg::Auto => hanzo_engine::CodeExecutionPermission::Auto,
            CodeExecPermissionArg::Ask => hanzo_engine::CodeExecutionPermission::Ask,
            CodeExecPermissionArg::Deny => hanzo_engine::CodeExecutionPermission::Deny,
        }
    }
}

impl From<CodeExecPermissionArg> for hanzo_engine::AgentPermission {
    fn from(value: CodeExecPermissionArg) -> Self {
        match value {
            CodeExecPermissionArg::Auto => hanzo_engine::AgentPermission::Auto,
            CodeExecPermissionArg::Ask => hanzo_engine::AgentPermission::Ask,
            CodeExecPermissionArg::Deny => hanzo_engine::AgentPermission::Deny,
        }
    }
}

impl Default for GlobalOptions {
    fn default() -> Self {
        Self {
            seed: None,
            log: None,
            token_source: TokenSource::CacheToken,
            verbose: 0,
        }
    }
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            max_seqs: 32,
            no_kv_cache: false,
            prefix_cache_n: 16,
            chat_template: None,
            jinja_explicit: None,
            matformer_config_path: None,
            matformer_slice_name: None,
            mtp_model: None,
            mtp_n_predict: None,
            draft_model: None,
            draft_quantized_file: None,
            gamma: None,
            mcp_config: None,
            agent: false,
            enable_search: false,
            search_embedding_model: None,
            #[cfg(feature = "code-execution")]
            enable_code_execution: false,
            #[cfg(feature = "code-execution")]
            code_exec_python: None,
            #[cfg(feature = "code-execution")]
            code_exec_timeout: None,
            #[cfg(feature = "code-execution")]
            code_exec_workdir: None,
            code_exec_permission: CodeExecPermissionArg::Auto,
        }
    }
}

fn parse_token_source(s: &str) -> Result<TokenSource, String> {
    s.parse()
}

fn default_token_source() -> TokenSource {
    TokenSource::CacheToken
}

fn default_max_seqs() -> usize {
    32
}

fn default_prefix_cache_n() -> usize {
    16
}

/// Default speculative gamma: draft tokens proposed per target verification step.
pub const DEFAULT_GAMMA: usize = 4;

/// Build the GGUF draft-model selector for classic draft+target speculative decoding, sized to the
/// target's context window so the draft keeps proposing across the full sequence instead of silently
/// stopping past a smaller hardcoded cap. Shared by the run/serve and bench runtime option structs.
fn build_draft_model_selected(
    draft_model: Option<&str>,
    draft_quantized_file: Option<&str>,
    max_seq_len: usize,
    max_batch_size: usize,
) -> Option<hanzo_engine::ModelSelected> {
    Some(hanzo_engine::ModelSelected::GGUF {
        tok_model_id: None,
        quantized_model_id: draft_model?.to_string(),
        quantized_filename: draft_quantized_file.unwrap_or_default().to_string(),
        dtype: hanzo_engine::ModelDType::Auto,
        topology: None,
        max_seq_len,
        max_batch_size,
    })
}
