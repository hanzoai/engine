//! Synchronous engine bridge for EVM precompiles.
//!
//! EVM precompiles (`0x0201` AI inference, `0x0202` AI embedding) run
//! synchronously inside smart-contract execution. This module exposes a
//! process-wide `Arc<Hanzo>` plus blocking [`infer`] / [`embed`] entry points
//! over the engine's async request channel, so a precompile handler — or any
//! consensus-time caller — can run real native inference without touching the
//! engine internals.
//!
//! One registry, one way: the node installs the engine here once at startup via
//! [`register_engine`]; both the node and the VM precompiles read it back
//! through this module (no second global slot).

use std::sync::{Arc, OnceLock};

use either::Either;
use indexmap::IndexMap;

use crate::{
    AutoDeviceMapParams, Hanzo, MessageContent, ModelDType, ModelSelected, NormalRequest, Request,
    RequestMessage, Response, SamplingParams,
};
use std::path::Path;

/// Cap on generated tokens for a single precompile inference call. Bounds gas
/// and matches the deterministic, one-shot nature of a precompile.
const PRECOMPILE_MAX_TOKENS: usize = 512;

/// Process-wide engine handle, installed once at node startup. Read by the EVM
/// precompile handlers for `0x0201` / `0x0202`.
static ENGINE: OnceLock<Arc<Hanzo>> = OnceLock::new();

/// Errors surfaced to the EVM precompile layer. Variants mirror exactly what the
/// `hanzo-vm` precompile handlers match on.
#[derive(Debug, Clone)]
pub enum EngineError {
    /// No inference engine installed on this node.
    NoInferenceEngine,
    /// No embedding engine installed on this node.
    NoEmbeddingEngine,
    /// The requested model id is not hosted.
    ModelNotFound(String),
    /// Any other engine-side failure.
    Other(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoInferenceEngine => write!(f, "no inference engine registered"),
            Self::NoEmbeddingEngine => write!(f, "no embedding engine registered"),
            Self::ModelNotFound(id) => write!(f, "model not found: {id}"),
            Self::Other(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for EngineError {}

/// Install the process-wide engine. First writer wins; returns `true` if this
/// call set the engine. Called once by the node at startup.
pub fn register_engine(engine: Arc<Hanzo>) -> bool {
    ENGINE.set(engine).is_ok()
}

/// Borrow the installed engine, if any. `None` means "no model loaded" — the
/// precompile then reverts ("fail open").
pub fn engine_handle() -> Option<Arc<Hanzo>> {
    ENGINE.get().cloned()
}

/// Whether an inference-capable engine is installed.
pub fn inference_engine_registered() -> bool {
    ENGINE.get().is_some()
}

/// Whether an embedding-capable engine is installed. (Same handle; whether the
/// hosted model can embed is resolved at call time.)
pub fn embedding_engine_registered() -> bool {
    ENGINE.get().is_some()
}

/// Blocking text-generation pass for the EVM precompile at `0x0201`.
///
/// `model` names the hosted model to route to (any loaded zen model);
/// empty/`"default"` uses the engine default. Returns the completion as UTF-8.
pub fn infer(model: &str, prompt: &[u8]) -> Result<Vec<u8>, EngineError> {
    let hanzo = engine_handle().ok_or(EngineError::NoInferenceEngine)?;
    let prompt = std::str::from_utf8(prompt)
        .map_err(|e| EngineError::Other(format!("prompt is not valid UTF-8: {e}")))?
        .to_string();
    let model_id = resolve_model(model);

    run_blocking(move || {
        let mut msg: IndexMap<String, MessageContent> = IndexMap::new();
        msg.insert("role".to_string(), Either::Left("user".to_string()));
        msg.insert("content".to_string(), Either::Left(prompt));
        let messages = RequestMessage::Chat {
            messages: vec![msg],
            enable_thinking: Some(false),
            reasoning_effort: None,
        };
        let mut sampling = SamplingParams::deterministic();
        sampling.max_len = Some(PRECOMPILE_MAX_TOKENS);

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        // Route to the named model via the engine's native multi-model support
        // (`NormalRequest.model_id` -> `Hanzo::get_sender`); `None` uses the
        // default model.
        let mut req = NormalRequest::new_simple(messages, sampling, tx, 0, None, None);
        req.model_id = model_id;
        hanzo
            .send_request(Request::Normal(Box::new(req)))
            .map_err(|e| EngineError::Other(format!("engine send failed: {e}")))?;

        match rx.blocking_recv() {
            Some(Response::Done(resp)) => Ok(resp
                .choices
                .first()
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default()
                .into_bytes()),
            Some(Response::ModelError(msg, _)) => Err(EngineError::Other(msg)),
            Some(Response::ValidationError(e)) | Some(Response::InternalError(e)) => {
                Err(EngineError::Other(e.to_string()))
            }
            Some(_) => Err(EngineError::Other("unexpected engine response".to_string())),
            None => Err(EngineError::Other("engine channel closed".to_string())),
        }
    })
}

/// Blocking embedding pass for the EVM precompile at `0x0202`.
///
/// `model` names the hosted embedding model to route to (any loaded
/// zen-embedding model); empty/`"default"` uses the engine default. Returns the
/// model's native embedding vector; the caller encodes it as little-endian f32.
pub fn embed(model: &str, text: &[u8]) -> Result<Vec<f32>, EngineError> {
    let hanzo = engine_handle().ok_or(EngineError::NoEmbeddingEngine)?;
    let prompt = std::str::from_utf8(text)
        .map_err(|e| EngineError::Other(format!("text is not valid UTF-8: {e}")))?
        .to_string();
    let model_id = resolve_model(model);

    run_blocking(move || {
        let messages = RequestMessage::Embedding { prompt };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let mut req = NormalRequest::new_simple(
            messages,
            SamplingParams::deterministic(),
            tx,
            0,
            None,
            None,
        );
        req.model_id = model_id;
        hanzo
            .send_request(Request::Normal(Box::new(req)))
            .map_err(|e| EngineError::Other(format!("engine send failed: {e}")))?;
        match rx.blocking_recv() {
            Some(Response::Embeddings { embeddings, .. }) => Ok(embeddings),
            Some(Response::ModelError(msg, _)) => Err(EngineError::Other(msg)),
            Some(Response::ValidationError(e)) | Some(Response::InternalError(e)) => {
                Err(EngineError::Other(e.to_string()))
            }
            Some(_) => Err(EngineError::Other(
                "model does not support embeddings".to_string(),
            )),
            None => Err(EngineError::Other("engine channel closed".to_string())),
        }
    })
}

/// Parse a `name=kind:source;...` model spec into `(name, ModelSelected)`
/// configs for `HanzoForServerBuilder`. `kind` ∈ {`gguf`,`plain`,`embedding`}.
/// `tok_dir` overrides the GGUF tokenizer dir (else the .gguf's own dir). Shared
/// by the FFI loader and the VM examples so there is exactly one config format
/// for "load these zen / zen-embedding models". The first entry is the default.
pub fn parse_model_spec(
    spec: &str,
    tok_dir: Option<&str>,
) -> Result<Vec<(String, ModelSelected)>, String> {
    let mut out = Vec::new();
    for entry in spec.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, rest) = entry
            .split_once('=')
            .ok_or_else(|| format!("bad model entry (need name=kind:source): {entry}"))?;
        let (kind, source) = rest
            .split_once(':')
            .ok_or_else(|| format!("bad model entry (need kind:source): {entry}"))?;
        let name = name.trim().to_string();
        let source = source.trim().to_string();
        let model = match kind.trim() {
            "gguf" => {
                let p = Path::new(&source);
                let dir = p
                    .parent()
                    .map(|d| d.to_string_lossy().to_string())
                    .unwrap_or_default();
                let file = p
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .ok_or_else(|| format!("gguf source has no filename: {source}"))?;
                ModelSelected::GGUF {
                    tok_model_id: Some(tok_dir.map(str::to_string).unwrap_or(dir.clone())),
                    quantized_model_id: dir,
                    quantized_filename: file,
                    dtype: ModelDType::Auto,
                    topology: None,
                    max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
                    max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
                }
            }
            "plain" => ModelSelected::Plain {
                model_id: source,
                tokenizer_json: None,
                arch: None,
                dtype: ModelDType::Auto,
                topology: None,
                organization: None,
                write_uqff: None,
                from_uqff: None,
                imatrix: None,
                calibration_file: None,
                max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
                max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
                hf_cache_path: None,
                matformer_config_path: None,
                matformer_slice_name: None,
            },
            "embedding" | "embed" => ModelSelected::Embedding {
                model_id: source,
                tokenizer_json: None,
                arch: None, // auto-detect (Qwen3Embedding / EmbeddingGemma)
                dtype: ModelDType::Auto,
                topology: None,
                write_uqff: None,
                from_uqff: None,
                hf_cache_path: None,
            },
            other => return Err(format!("unknown model kind '{other}' in: {entry}")),
        };
        out.push((name, model));
    }
    if out.is_empty() {
        return Err("empty model spec".to_string());
    }
    Ok(out)
}

/// Map a caller-supplied model name to the engine's optional model id. Empty,
/// all-NUL, or `"default"` → `None` (use the engine default); otherwise the
/// trimmed name routes through `Hanzo::get_sender` / `resolve_alias_or_default`
/// to that loaded model. Lets callers address *any* loaded zen / zen-embedding
/// model by name.
fn resolve_model(model: &str) -> Option<String> {
    let m = model.trim().trim_matches('\0').trim();
    if m.is_empty() || m.eq_ignore_ascii_case("default") {
        None
    } else {
        Some(m.to_string())
    }
}

/// Run `f` on a fresh OS thread so the engine's blocking channel ops never
/// execute inside a Tokio runtime (where `blocking_send` / `blocking_recv`
/// panic). The precompile caller blocks on the join, which is the intended
/// synchronous-precompile semantics.
fn run_blocking<T, F>(f: F) -> Result<T, EngineError>
where
    F: FnOnce() -> Result<T, EngineError> + Send,
    T: Send,
{
    match std::thread::scope(|s| s.spawn(f).join()) {
        Ok(inner) => inner,
        Err(_) => Err(EngineError::Other("inference thread panicked".to_string())),
    }
}
