//! Backend-agnostic trait shared by the FFI and native implementations.
//!
//! Designed so hanzod can hold a `Box<dyn Zen5Engine>` without caring whether
//! the underlying weights are running through the C runtime or the candle-rs
//! port. The trait is intentionally narrow: load, complete (stream tokens),
//! embed. Federation training deltas go through a separate path (see the
//! `hanzo-federation` crate); this trait exposes only inference + the hook
//! `apply_delta` so a coordinator can hot-swap a fresh LoRA / DeltaSoup
//! aggregate without restarting the engine.

use std::path::Path;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A single generated token. The `id` is the vocab index; `text` is the UTF-8
/// piece if the decoder produced one this step (some BPE pieces stay buffered
/// until the next merge resolves).
#[derive(Debug, Clone)]
pub struct Token {
    pub id: i32,
    pub text: String,
    pub logprob: f32,
}

/// Sampling and reasoning controls. Mirrors the knobs in
/// `ds4_session_sample` and `ds4_think_mode`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenOpts {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub think: ThinkMode,
    /// Optional RNG seed for reproducible sampling.
    pub seed: Option<u64>,
    /// Stop generation on these strings (decoded). Empty = only stop on EOS.
    pub stop: Vec<String>,
}

impl Default for GenOpts {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
            min_p: 0.05,
            think: ThinkMode::None,
            seed: None,
            stop: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ThinkMode {
    #[default]
    None,
    High,
    Max,
}

#[derive(Debug, Error)]
pub enum Zen5Error {
    #[error("model load failed: {0}")]
    Load(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("backend unavailable: {0}")]
    Backend(String),
    #[error("io error")]
    Io(#[from] std::io::Error),
}

pub type TokenStream = Pin<Box<dyn Stream<Item = Result<Token, Zen5Error>> + Send>>;

/// Backend-agnostic Zen5 inference engine. Both `ffi::Engine` and
/// `native::Engine` implement this trait.
#[async_trait]
pub trait Zen5Engine: Send + Sync + std::fmt::Debug {
    /// Backend name for diagnostics ("ffi/metal", "ffi/cuda", "ffi/cpu", "native").
    fn backend(&self) -> &'static str;

    /// Stream completions for `prompt`. The stream ends when EOS is sampled,
    /// `opts.max_tokens` is reached, or a `stop` string is matched.
    async fn complete(&self, prompt: &str, opts: GenOpts) -> Result<TokenStream, Zen5Error>;

    /// Compute a single embedding vector for `text`. Returns the final-layer
    /// mean-pooled hidden state. Vector length is model-dependent.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, Zen5Error>;

    /// Apply a federation training delta in place. The delta format is owned
    /// by `hanzo-federation::codec`; this method is a no-op for backends that
    /// don't support hot-swap (FFI today; native once LoRA is plumbed).
    async fn apply_delta(&self, _delta: &[u8]) -> Result<(), Zen5Error> {
        Err(Zen5Error::Backend("apply_delta not implemented for this backend".into()))
    }
}

/// Constructor sugar so callers don't have to name the backend type. Returns
/// the FFI backend when the `ffi` feature is on, otherwise the native one.
/// hanzod uses this when the config doesn't pin a backend explicitly.
pub fn open(path: &Path) -> Result<Box<dyn Zen5Engine>, Zen5Error> {
    #[cfg(feature = "ffi")]
    {
        let e = crate::ffi::Engine::load(path, Default::default())?;
        return Ok(Box::new(e));
    }
    #[cfg(all(not(feature = "ffi"), feature = "native"))]
    {
        let e = crate::native::Engine::load(path)?;
        return Ok(Box::new(e));
    }
    #[cfg(not(any(feature = "ffi", feature = "native")))]
    {
        let _ = path;
        Err(Zen5Error::Backend(
            "no backend enabled — build with --features=ffi or --features=native".into(),
        ))
    }
}
