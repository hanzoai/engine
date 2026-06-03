//! Canonical inference + embedding API. The runtime registers a real engine
//! at process start; every consumer (precompiles, RPC handlers, etc.) calls
//! [`infer`] / [`embed`] which dispatch through the global registry.
//!
//! There is exactly one inference engine and one embedding engine per process.
//! Attempting to register a second engine of either kind returns
//! [`EngineError::Other`]. This is intentional: a node either has the model
//! loaded or it does not.

use std::sync::{Arc, OnceLock};
use thiserror::Error;

/// All errors returned by the [`InferenceEngine`] / [`EmbeddingEngine`] surface.
///
/// The variants are deliberately coarse. Inference engines wrap their internal
/// errors into [`EngineError::Other`] with a human-readable message.
#[derive(Debug, Error)]
pub enum EngineError {
    /// No engine of the given kind ("inference" / "embedding") has been
    /// registered. The dispatch path emits this when a consumer calls
    /// [`infer`] / [`embed`] before [`register_inference_engine`] /
    /// [`register_embedding_engine`] was called.
    #[error("no {0} engine registered on this node")]
    EngineNotRegistered(String),

    /// The engine accepts requests but does not have the requested model
    /// loaded. `model_id` is the 32-byte ID rendered as hex by the engine
    /// for diagnostics.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// Caller-provided input was malformed (bad prompt encoding, wrong
    /// dimensionality, etc.). Surfaced by hanzo-vm precompiles as a revert
    /// reason rather than an engine-side failure.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The engine accepted the request but failed to produce a result
    /// (OOM, kernel failure, backend crash). Distinct from [`Other`] so
    /// downstream code can distinguish runtime faults from miscellaneous
    /// errors.
    #[error("engine failed: {0}")]
    EngineFailed(String),

    /// Any other engine-level error. Wrap your internal error type's
    /// `Display` impl into this when implementing [`InferenceEngine`] or
    /// [`EmbeddingEngine`].
    #[error("engine: {0}")]
    Other(String),
}

/// A backend that produces a token stream / completion for a prompt against
/// a specific model.
///
/// `model_id` is an opaque 32-byte identifier. Engines are expected to map
/// it to whatever they use internally (a Hugging Face repo, a path, a
/// content hash, etc.). The engine returns the error
/// [`EngineError::ModelNotFound`] if the ID is unknown.
pub trait InferenceEngine: Send + Sync + 'static {
    /// Run inference against `model_id` with the given prompt bytes
    /// (UTF-8 text) and return the completion bytes (UTF-8 text).
    fn infer(&self, model_id: &[u8; 32], prompt: &[u8]) -> Result<Vec<u8>, EngineError>;
}

/// A backend that produces a dense vector embedding for a piece of text.
///
/// `dim` is the requested embedding dimensionality. Engines that only
/// support a fixed dimensionality should validate `dim` against their
/// model and return [`EngineError::Other`] on mismatch.
pub trait EmbeddingEngine: Send + Sync + 'static {
    /// Embed `text` (UTF-8) and return a vector of length `dim`.
    fn embed(&self, dim: usize, text: &[u8]) -> Result<Vec<f32>, EngineError>;
}

static INFER: OnceLock<Arc<dyn InferenceEngine>> = OnceLock::new();
static EMBED: OnceLock<Arc<dyn EmbeddingEngine>> = OnceLock::new();

/// Register the process-wide inference engine. Returns an error if one was
/// already registered. There is exactly one inference engine per process.
pub fn register_inference_engine(e: Arc<dyn InferenceEngine>) -> Result<(), EngineError> {
    INFER
        .set(e)
        .map_err(|_| EngineError::Other("inference engine already registered".into()))
}

/// Register the process-wide embedding engine. Returns an error if one was
/// already registered. There is exactly one embedding engine per process.
pub fn register_embedding_engine(e: Arc<dyn EmbeddingEngine>) -> Result<(), EngineError> {
    EMBED
        .set(e)
        .map_err(|_| EngineError::Other("embedding engine already registered".into()))
}

/// Dispatch an inference call to the registered engine.
///
/// Returns [`EngineError::EngineNotRegistered("inference".into())`] if no engine is registered.
pub fn infer(model_id: &[u8; 32], prompt: &[u8]) -> Result<Vec<u8>, EngineError> {
    INFER
        .get()
        .ok_or(EngineError::EngineNotRegistered("inference".into()))?
        .infer(model_id, prompt)
}

/// Dispatch an embedding call to the registered engine.
///
/// Returns [`EngineError::EngineNotRegistered("embedding".into())`] if no engine is registered.
pub fn embed(dim: usize, text: &[u8]) -> Result<Vec<f32>, EngineError> {
    EMBED
        .get()
        .ok_or(EngineError::EngineNotRegistered("embedding".into()))?
        .embed(dim, text)
}

/// Returns `true` if [`register_inference_engine`] has been called.
pub fn inference_engine_registered() -> bool {
    INFER.get().is_some()
}

/// Returns `true` if [`register_embedding_engine`] has been called.
pub fn embedding_engine_registered() -> bool {
    EMBED.get().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Calling [`infer`] before any engine is registered must return
    /// [`EngineError::EngineNotRegistered("inference".into())`] (not panic, not deadlock).
    #[test]
    fn infer_without_engine_returns_not_registered() {
        // Note: this is a unit test, so the OnceLock here is the same one
        // any other unit test in this module would touch. We deliberately
        // do NOT register anything here; that's what the integration test
        // `tests/api.rs` covers.
        let id = [0u8; 32];
        match infer(&id, b"hello") {
            Err(EngineError::EngineNotRegistered(kind)) if kind == "inference" => {}
            other => panic!("expected EngineNotRegistered(inference), got {other:?}"),
        }
    }

    /// Same as above for the embedding side.
    #[test]
    fn embed_without_engine_returns_not_registered() {
        match embed(8, b"hello") {
            Err(EngineError::EngineNotRegistered(kind)) if kind == "embedding" => {}
            other => panic!("expected EngineNotRegistered(embedding), got {other:?}"),
        }
    }

    /// `EngineError` round-trips through Display so logs are readable.
    #[test]
    fn engine_error_display_is_useful() {
        assert_eq!(
            EngineError::EngineNotRegistered("inference".into()).to_string(),
            "no inference engine registered on this node"
        );
        assert_eq!(
            EngineError::EngineNotRegistered("embedding".into()).to_string(),
            "no embedding engine registered on this node"
        );
        assert_eq!(
            EngineError::ModelNotFound("abc".into()).to_string(),
            "model not found: abc"
        );
        assert_eq!(
            EngineError::Other("boom".into()).to_string(),
            "engine: boom"
        );
    }
}
