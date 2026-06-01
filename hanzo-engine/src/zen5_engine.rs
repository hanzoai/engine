//! [`Zen5InferenceAdapter`] — bridge between the [`hanzo_zen5::Zen5Engine`]
//! trait and the canonical [`crate::InferenceEngine`] surface.
//!
//! Why a separate adapter? `hanzo-zen5` exposes a streaming, async API
//! (`complete -> TokenStream`) keyed by a single in-memory model. The
//! `hanzo-engine` registry is a synchronous, sync-friendly API keyed by a
//! 32-byte `model_id`. The adapter:
//!
//! 1. Hashes a stable identifier (`zen-5-<variant>:<weights_path>`) into a
//!    `model_id` so EVM precompiles can pin a specific Zen5 weights file.
//! 2. Owns a dedicated tokio runtime so synchronous callers (precompiles,
//!    `infer()`-from-non-async-context) can block on the underlying async
//!    stream without dragging tokio context across the FFI boundary.
//! 3. Collects the token stream into a single UTF-8 completion. Streaming
//!    aware callers should go directly to `Zen5Engine::complete()`.
//!
//! ## Multi-model registry
//!
//! [`Zen5Registry`] holds N adapters keyed by `model_id`. It implements
//! [`InferenceEngine`] so a single registration covers the whole Zen5
//! family. [`register_zen5_engines_at_startup`] is the boot helper that
//! reads a weights directory + variant list, opens each one, and installs
//! the registry into the process-wide engine slot.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use futures::StreamExt;
use sha2::{Digest, Sha256};
use tokio::runtime::{Handle, Runtime};

use crate::api::{EngineError, InferenceEngine};
use hanzo_zen5::engine::{GenOpts, Zen5Engine};

/// One loaded Zen5 weights file behind a stable [`model_id`](Self::model_id).
///
/// Use [`Zen5Registry`] when you want to serve more than one variant.
pub struct Zen5InferenceAdapter {
    inner: Arc<dyn Zen5Engine>,
    model_id: [u8; 32],
    label: String,
    rt: Runtime,
}

impl std::fmt::Debug for Zen5InferenceAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Zen5InferenceAdapter")
            .field("label", &self.label)
            .field("backend", &self.inner.backend())
            .field("model_id", &hex_id(&self.model_id))
            .finish()
    }
}

impl Zen5InferenceAdapter {
    /// Open a GGUF weights file via the default zen5 backend (`hanzo_zen5::engine::open`).
    /// `label` is hashed into [`Self::model_id`]; conventionally
    /// `"<variant>:<absolute path>"` so the same weights file at the same
    /// path always derives the same id.
    pub fn open(label: impl Into<String>, path: &Path) -> Result<Self, EngineError> {
        let label = label.into();
        let engine = hanzo_zen5::engine::open(path)
            .map_err(|e| EngineError::Other(format!("zen5 open {label}: {e}")))?;
        Self::wrap(label, Arc::from(engine))
    }

    /// Wrap a caller-built [`Zen5Engine`]. Useful for tests and the
    /// `native` candle backend which has its own constructors.
    pub fn wrap(label: impl Into<String>, inner: Arc<dyn Zen5Engine>) -> Result<Self, EngineError> {
        let label = label.into();
        let model_id = hash_label(&label);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(2)
            .thread_name("hanzo-zen5-dispatch")
            .build()
            .map_err(|e| EngineError::Other(format!("zen5 runtime: {e}")))?;
        Ok(Self { inner, model_id, label, rt })
    }

    /// The 32-byte content hash identifying this loaded model.
    pub fn model_id(&self) -> &[u8; 32] {
        &self.model_id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn backend(&self) -> &'static str {
        self.inner.backend()
    }

    /// Apply a federation training delta to this engine. See
    /// [`Zen5Engine::apply_delta`].
    pub fn apply_delta(&self, delta: &[u8]) -> Result<(), EngineError> {
        let inner = Arc::clone(&self.inner);
        let delta = delta.to_vec();
        self.run(async move {
            inner
                .apply_delta(&delta)
                .await
                .map_err(|e| EngineError::Other(format!("zen5 apply_delta: {e}")))
        })
    }

    /// Block on the engine's dedicated runtime. Same caveats as
    /// `MistralEngine::run`: safe from sync contexts; from another tokio
    /// runtime this trampolines via a one-shot channel.
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
                rx.recv().expect("zen5 runtime task panicked")
            }
        }
    }

    fn complete_to_bytes(&self, prompt: &str) -> Result<Vec<u8>, EngineError> {
        let inner = Arc::clone(&self.inner);
        let prompt = prompt.to_owned();
        self.run(async move {
            let mut stream = inner
                .complete(&prompt, GenOpts::default())
                .await
                .map_err(|e| EngineError::Other(format!("zen5 complete: {e}")))?;
            let mut out = String::new();
            while let Some(tok) = stream.next().await {
                match tok {
                    Ok(t) => out.push_str(&t.text),
                    Err(e) => return Err(EngineError::Other(format!("zen5 token: {e}"))),
                }
            }
            Ok(out.into_bytes())
        })
    }
}

impl InferenceEngine for Zen5InferenceAdapter {
    fn infer(&self, model_id: &[u8; 32], prompt: &[u8]) -> Result<Vec<u8>, EngineError> {
        if model_id != &self.model_id {
            return Err(EngineError::ModelNotFound(hex_id(model_id)));
        }
        let prompt_str = std::str::from_utf8(prompt)
            .map_err(|e| EngineError::Other(format!("zen5 prompt not UTF-8: {e}")))?;
        self.complete_to_bytes(prompt_str)
    }
}

/// Multi-variant registry. Implements [`InferenceEngine`] so it can be
/// installed once into the process-wide [`crate::register_inference_engine`]
/// slot and serve all four `zen-5-*` model ids.
pub struct Zen5Registry {
    by_id: HashMap<[u8; 32], Arc<Zen5InferenceAdapter>>,
}

impl std::fmt::Debug for Zen5Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let labels: Vec<&str> = self
            .by_id
            .values()
            .map(|a| a.label())
            .collect();
        f.debug_struct("Zen5Registry")
            .field("models", &labels)
            .finish()
    }
}

impl Zen5Registry {
    pub fn new() -> Self {
        Self { by_id: HashMap::new() }
    }

    /// Number of models in the registry.
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Add an adapter under its [`Zen5InferenceAdapter::model_id`].
    pub fn insert(&mut self, adapter: Arc<Zen5InferenceAdapter>) {
        self.by_id.insert(*adapter.model_id(), adapter);
    }

    /// Look up the adapter for a `(label) -> model_id` mapping. Useful for
    /// telling the user "your `zen-5-flash` model id is …".
    pub fn id_for(&self, label: &str) -> [u8; 32] {
        hash_label(label)
    }

    /// Get an adapter by id. Returns `None` if not loaded.
    pub fn get(&self, id: &[u8; 32]) -> Option<Arc<Zen5InferenceAdapter>> {
        self.by_id.get(id).cloned()
    }

    /// Apply a delta to whatever model `id` is loaded. No-op error if the
    /// model isn't registered.
    pub fn apply_delta(&self, id: &[u8; 32], delta: &[u8]) -> Result<(), EngineError> {
        let adapter = self
            .by_id
            .get(id)
            .ok_or_else(|| EngineError::ModelNotFound(hex_id(id)))?;
        adapter.apply_delta(delta)
    }

    pub fn labels(&self) -> Vec<String> {
        self.by_id.values().map(|a| a.label().to_string()).collect()
    }
}

impl Default for Zen5Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceEngine for Zen5Registry {
    fn infer(&self, model_id: &[u8; 32], prompt: &[u8]) -> Result<Vec<u8>, EngineError> {
        let adapter = self
            .by_id
            .get(model_id)
            .ok_or_else(|| EngineError::ModelNotFound(hex_id(model_id)))?;
        adapter.infer(model_id, prompt)
    }
}

/// Discover one weights file per variant in `weights_dir` and build a
/// [`Zen5Registry`]. The file naming convention is `<variant>.gguf`
/// (e.g. `zen-5-flash.gguf`). Missing files are skipped with a warning so
/// a partial install still boots.
pub fn build_registry(
    weights_dir: &Path,
    variants: &[&str],
) -> Result<Zen5Registry, EngineError> {
    let mut registry = Zen5Registry::new();
    for &v in variants {
        let candidate = weights_dir.join(format!("{v}.gguf"));
        if !candidate.exists() {
            tracing::warn!(
                target: "hanzo_engine::zen5",
                variant = v,
                path = %candidate.display(),
                "zen5 weights file missing; skipping",
            );
            continue;
        }
        let label = format!("{v}:{}", candidate.display());
        match Zen5InferenceAdapter::open(label.clone(), &candidate) {
            Ok(a) => {
                tracing::info!(
                    target: "hanzo_engine::zen5",
                    variant = v,
                    backend = a.backend(),
                    model_id = %hex_id(a.model_id()),
                    "registered zen5 variant",
                );
                registry.insert(Arc::new(a));
            }
            Err(e) => {
                tracing::error!(
                    target: "hanzo_engine::zen5",
                    variant = v,
                    error = %e,
                    "failed to open zen5 weights; continuing",
                );
            }
        }
    }
    Ok(registry)
}

/// Boot helper: build a [`Zen5Registry`] for the standard zen5 variants and
/// install it into the process-wide engine slot via
/// [`crate::register_inference_engine`].
///
/// If an engine is already registered (e.g. `MistralEngine` from
/// `runner::install_engine`), this is a no-op error you should log + ignore
/// — first-writer wins, intentionally. The Zen5 boot should happen before
/// the Mistral boot if you want zen5 to be primary.
pub fn register_zen5_engines_at_startup(
    weights_dir: &Path,
    variants: &[&str],
) -> Result<Arc<Zen5Registry>, EngineError> {
    let registry = build_registry(weights_dir, variants)?;
    let arc = Arc::new(registry);
    crate::register_inference_engine(Arc::clone(&arc) as Arc<dyn InferenceEngine>)
        .map_err(|e| {
            EngineError::Other(format!(
                "zen5 register failed (another inference engine already registered): {e}"
            ))
        })?;
    Ok(arc)
}

/// Default variant list used by [`register_zen5_engines_at_startup`] when
/// the caller doesn't override.
pub const DEFAULT_VARIANTS: &[&str] = &[
    "zen-5-flash",
    "zen-5-pro",
    "zen-5-mini",
    "zen-5-coder",
];

fn hash_label(label: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(label.as_bytes());
    h.finalize().into()
}

fn hex_id(id: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(64);
    for b in id {
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

/// Convenience helper: turn a (variant, weights_dir) pair into the
/// 32-byte model_id that would be assigned by [`register_zen5_engines_at_startup`].
/// Callers (e.g. CLI tools, JSON-RPC handlers) use this to fill in the
/// `model_id` argument for `infer`.
pub fn model_id_for(variant: &str, weights_dir: &Path) -> [u8; 32] {
    let label = format!("{variant}:{}", weights_dir.join(format!("{variant}.gguf")).display());
    hash_label(&label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn model_id_is_stable_for_same_label() {
        let dir = PathBuf::from("/var/lib/hanzo/zen5");
        let a = model_id_for("zen-5-flash", &dir);
        let b = model_id_for("zen-5-flash", &dir);
        assert_eq!(a, b);
    }

    #[test]
    fn different_variants_get_different_ids() {
        let dir = PathBuf::from("/var/lib/hanzo/zen5");
        let a = model_id_for("zen-5-flash", &dir);
        let b = model_id_for("zen-5-pro", &dir);
        assert_ne!(a, b);
    }

    #[test]
    fn registry_returns_model_not_found_for_unknown() {
        let r = Zen5Registry::new();
        let id = [9u8; 32];
        match r.infer(&id, b"hi") {
            Err(EngineError::ModelNotFound(s)) => assert_eq!(s.len(), 64),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn build_registry_skips_missing_files() {
        let dir = tempfile::tempdir().unwrap();
        let registry = build_registry(dir.path(), DEFAULT_VARIANTS).unwrap();
        assert!(registry.is_empty(), "no .gguf files present");
    }

    #[test]
    fn default_variants_covers_the_four_target_models() {
        // Lockstep with the task spec — flash, pro, mini, coder.
        assert_eq!(DEFAULT_VARIANTS.len(), 4);
        assert!(DEFAULT_VARIANTS.contains(&"zen-5-flash"));
        assert!(DEFAULT_VARIANTS.contains(&"zen-5-pro"));
        assert!(DEFAULT_VARIANTS.contains(&"zen-5-mini"));
        assert!(DEFAULT_VARIANTS.contains(&"zen-5-coder"));
    }
}
