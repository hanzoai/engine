//! [`MistralEngine`] — the real [`InferenceEngine`] / [`EmbeddingEngine`]
//! implementation backed by `mistralrs-core`.
//!
//! A `MistralEngine` owns:
//! * one loaded `mistralrs::Model` (text completion + optional embedding)
//! * the 32-byte content hash that consumers use as `model_id`
//! * a dedicated tokio runtime so synchronous EVM-style call sites can
//!   block on async inference without a `tokio::main` context
//!
//! ```no_run
//! use std::sync::Arc;
//! use hanzo_engine::{MistralEngine, infer, register_inference_engine};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let engine = MistralEngine::from_hf_repo("Qwen/Qwen3-4B").await?;
//! let id = *engine.model_id();
//! register_inference_engine(Arc::new(engine))?;
//!
//! let out = infer(&id, b"Hello").unwrap();
//! println!("{}", String::from_utf8_lossy(&out));
//! # Ok(())
//! # }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mistralrs::{
    EmbeddingModelBuilder, EmbeddingRequest, Model, ModelBuilder, TextMessageRole, TextMessages,
};
use sha2::{Digest, Sha256};
use tokio::runtime::{Handle, Runtime};

use crate::api::{EmbeddingEngine, EngineError, InferenceEngine};

/// Real inference + embedding engine backed by mistralrs-core.
///
/// One `MistralEngine` holds exactly one loaded model. To serve multiple
/// models, register a router engine that delegates to several `MistralEngine`
/// instances.
pub struct MistralEngine {
    model: Arc<Model>,
    model_id: [u8; 32],
    /// Identifier (path or HF repo) used for diagnostics + ID derivation.
    source: String,
    /// Dedicated runtime for blocking dispatch from sync contexts.
    rt: Runtime,
}

impl MistralEngine {
    /// Load a text-completion model from a Hugging Face repo
    /// (e.g. `"Qwen/Qwen3-4B"`).
    ///
    /// This is async because model download + load is async. The returned
    /// engine can then be registered (synchronously) into the global
    /// registry.
    pub async fn from_hf_repo(repo: impl Into<String>) -> anyhow::Result<Self> {
        let source = repo.into();
        let model = ModelBuilder::new(&source).build().await?;
        Self::wrap(Arc::new(model), source)
    }

    /// Load a text-completion model from a local path (GGUF or safetensors
    /// directory). Internally still goes through [`ModelBuilder`], which
    /// auto-detects the model layout.
    pub async fn from_model_path(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let source = path.to_string_lossy().into_owned();
        let model = ModelBuilder::new(&source).build().await?;
        Self::wrap(Arc::new(model), source)
    }

    /// Load an embedding-only model from a Hugging Face repo
    /// (e.g. `"sentence-transformers/all-MiniLM-L6-v2"`).
    pub async fn embedding_from_hf_repo(repo: impl Into<String>) -> anyhow::Result<Self> {
        let source = repo.into();
        let model = EmbeddingModelBuilder::new(&source).build().await?;
        Self::wrap(Arc::new(model), source)
    }

    /// Wrap a pre-built [`Model`]. The `source` string is hashed to derive
    /// the 32-byte `model_id`. Use this when you already have a `Model`
    /// (e.g. from a custom builder).
    pub fn wrap(model: Arc<Model>, source: impl Into<String>) -> anyhow::Result<Self> {
        let source = source.into();
        let model_id = Self::hash_source(&source);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("hanzo-engine-dispatch")
            .build()?;
        Ok(Self {
            model,
            model_id,
            source,
            rt,
        })
    }

    /// Derive a 32-byte ID from an arbitrary source string. Stable: the
    /// same source always produces the same ID.
    pub fn hash_source(source: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(source.as_bytes());
        hasher.finalize().into()
    }

    /// The 32-byte content hash identifying the loaded model.
    pub fn model_id(&self) -> &[u8; 32] {
        &self.model_id
    }

    /// The original source string (path / repo) used to load this model.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Underlying mistralrs [`Model`] handle. Use this if you need a
    /// feature the trait surface doesn't expose (streaming, structured
    /// output, etc.).
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
}

impl MistralEngine {
    /// Block on the engine's dedicated runtime. Works whether the caller is
    /// in a sync context (the typical EVM-precompile case) or already inside
    /// another tokio runtime (e.g. an integration test using `#[tokio::test]`).
    ///
    /// Caveat: if you call this from inside a tokio runtime, the calling
    /// thread blocks on a sync channel until the engine completes. On a
    /// multi-threaded runtime this is fine; on a single-threaded runtime it
    /// will stall sibling tasks. Async callers should usually wrap the call
    /// in `tokio::task::spawn_blocking` rather than touching the trait method
    /// directly.
    fn run<F, T>(&self, fut: F) -> T
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        match Handle::try_current() {
            Err(_) => self.rt.block_on(fut),
            Ok(_) => {
                let (tx, rx) = std::sync::mpsc::channel();
                self.rt.spawn(async move {
                    let _ = tx.send(fut.await);
                });
                rx.recv().expect("engine runtime task panicked")
            }
        }
    }
}

impl InferenceEngine for MistralEngine {
    fn infer(&self, model_id: &[u8; 32], prompt: &[u8]) -> Result<Vec<u8>, EngineError> {
        if model_id != &self.model_id {
            return Err(EngineError::ModelNotFound(hex_id(model_id)));
        }
        let prompt_str = std::str::from_utf8(prompt)
            .map_err(|e| EngineError::Other(format!("prompt is not UTF-8: {e}")))?
            .to_owned();
        let model = Arc::clone(&self.model);
        let result = self.run(async move {
            let messages = TextMessages::new().add_message(TextMessageRole::User, prompt_str);
            let response = model
                .send_chat_request(messages)
                .await
                .map_err(|e| EngineError::Other(format!("chat request failed: {e}")))?;
            response
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.message.content)
                .ok_or_else(|| {
                    EngineError::Other("model returned no completion content".into())
                })
        })?;
        Ok(result.into_bytes())
    }
}

impl EmbeddingEngine for MistralEngine {
    fn embed(&self, dim: usize, text: &[u8]) -> Result<Vec<f32>, EngineError> {
        let text_str = std::str::from_utf8(text)
            .map_err(|e| EngineError::Other(format!("text is not UTF-8: {e}")))?
            .to_owned();
        let model = Arc::clone(&self.model);
        let vec = self.run(async move {
            let request = EmbeddingRequest::builder().add_prompt(text_str);
            let vecs = model
                .generate_embeddings(request)
                .await
                .map_err(|e| EngineError::Other(format!("embedding request failed: {e}")))?;
            vecs.into_iter()
                .next()
                .ok_or_else(|| EngineError::Other("embedding response was empty".into()))
        })?;
        if vec.len() != dim {
            return Err(EngineError::Other(format!(
                "embedding dim mismatch: requested {dim}, model returned {}",
                vec.len()
            )));
        }
        Ok(vec)
    }
}

fn hex_id(id: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in id {
        use std::fmt::Write;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}
