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
    /// No inference engine has been registered. A consumer asked for
    /// `infer()` before [`register_inference_engine`] was called.
    #[error("no inference engine registered on this node")]
    NoInferenceEngine,

    /// No embedding engine has been registered. A consumer asked for
    /// `embed()` before [`register_embedding_engine`] was called.
    #[error("no embedding engine registered on this node")]
    NoEmbeddingEngine,

    /// The engine accepts requests but does not have the requested model
    /// loaded. `model_id` is the 32-byte ID rendered as hex by the engine
    /// for diagnostics.
    #[error("model not found: {0}")]
    ModelNotFound(String),

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
/// Returns [`EngineError::NoInferenceEngine`] if no engine is registered.
pub fn infer(model_id: &[u8; 32], prompt: &[u8]) -> Result<Vec<u8>, EngineError> {
    INFER
        .get()
        .ok_or(EngineError::NoInferenceEngine)?
        .infer(model_id, prompt)
}

/// Dispatch an embedding call to the registered engine.
///
/// Returns [`EngineError::NoEmbeddingEngine`] if no engine is registered.
pub fn embed(dim: usize, text: &[u8]) -> Result<Vec<f32>, EngineError> {
    EMBED
        .get()
        .ok_or(EngineError::NoEmbeddingEngine)?
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
    /// [`EngineError::NoInferenceEngine`] (not panic, not deadlock).
    #[test]
    fn infer_without_engine_returns_not_registered() {
        // Note: this is a unit test, so the OnceLock here is the same one
        // any other unit test in this module would touch. We deliberately
        // do NOT register anything here; that's what the integration test
        // `tests/api.rs` covers.
        let id = [0u8; 32];
        match infer(&id, b"hello") {
            Err(EngineError::NoInferenceEngine) => {}
            other => panic!("expected NoInferenceEngine, got {other:?}"),
        }
    }

    /// Same as above for the embedding side.
    #[test]
    fn embed_without_engine_returns_not_registered() {
        match embed(8, b"hello") {
            Err(EngineError::NoEmbeddingEngine) => {}
            other => panic!("expected NoEmbeddingEngine, got {other:?}"),
        }
    }

    /// `EngineError` round-trips through Display so logs are readable.
    #[test]
    fn engine_error_display_is_useful() {
        assert_eq!(
            EngineError::NoInferenceEngine.to_string(),
            "no inference engine registered on this node"
        );
        assert_eq!(
            EngineError::NoEmbeddingEngine.to_string(),
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
