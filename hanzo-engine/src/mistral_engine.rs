//! [`MistralEngine`] — stub backend for the hanzo-engine registry.
//!
//! The full implementation, backed by the `hanzo` SDK / `mistralrs-core`,
//! lived in this file before the upstream merge wiped the engine source
//! tree. While the engine is being restored, this stub keeps the public
//! API surface intact so consumers (hanzod runner, hanzo-vm precompiles)
//! compile and link unchanged.
//!
//! At runtime `MistralEngine::from_hf_repo` / `from_model_path` return
//! [`EngineError::Other("hanzo-engine built without backend")`]. The
//! registry, traits, and dispatch path are all real — only the model
//! loader is stubbed.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::api::{EmbeddingEngine, EngineError, InferenceEngine};

/// Stub inference + embedding engine.
///
/// Carries enough state (model_id, source) to satisfy the API contract.
/// Calls to [`InferenceEngine::infer`] / [`EmbeddingEngine::embed`]
/// return [`EngineError::Other`] until a real backend is wired in.
pub struct MistralEngine {
    model_id: [u8; 32],
    source: String,
}

impl MistralEngine {
    /// Load from a Hugging Face repository identifier (e.g. `"Qwen/Qwen3-4B"`).
    ///
    /// Currently a stub: returns `EngineError::Other` indicating the
    /// engine was built without a backend.
    pub async fn from_hf_repo(repo: &str) -> Result<Self, EngineError> {
        Err(EngineError::Other(format!(
            "hanzo-engine built without backend; cannot load HF repo `{repo}`. \
             Restore the canonical engine source tree and rebuild with \
             `--features real-engine`."
        )))
    }

    /// Load from a local model directory.
    ///
    /// Currently a stub: returns `EngineError::Other` indicating the
    /// engine was built without a backend.
    pub async fn from_model_path(path: &Path) -> Result<Self, EngineError> {
        Err(EngineError::Other(format!(
            "hanzo-engine built without backend; cannot load model at `{}`. \
             Restore the canonical engine source tree and rebuild with \
             `--features real-engine`.",
            path.display()
        )))
    }

    /// Construct a stub engine bound to an arbitrary source identifier.
    ///
    /// The `model_id` is the SHA-256 of `source`. Used internally and by
    /// tests that exercise the registry without loading a model.
    pub fn stub(source: impl Into<String>) -> Self {
        let source = source.into();
        let mut hasher = Sha256::new();
        hasher.update(source.as_bytes());
        let model_id = hasher.finalize().into();
        Self { model_id, source }
    }

    /// 32-byte content hash of the model source. Stable across loads of
    /// the same source string.
    pub fn model_id(&self) -> &[u8; 32] {
        &self.model_id
    }

    /// Source identifier the engine was loaded from (HF repo or local
    /// path, in canonical form).
    pub fn source(&self) -> &str {
        &self.source
    }
}

impl InferenceEngine for MistralEngine {
    fn infer(
        &self,
        _model_id: &[u8; 32],
        _prompt: &[u8],
    ) -> Result<Vec<u8>, EngineError> {
        Err(EngineError::Other(format!(
            "hanzo-engine: infer() unavailable — engine `{}` has no backend",
            self.source
        )))
    }
}

impl EmbeddingEngine for MistralEngine {
    fn embed(&self, _dim: usize, _text: &[u8]) -> Result<Vec<f32>, EngineError> {
        Err(EngineError::Other(format!(
            "hanzo-engine: embed() unavailable — engine `{}` has no backend",
            self.source
        )))
    }
}

#[allow(dead_code)]
fn _path_marker(_p: &PathBuf) {}
