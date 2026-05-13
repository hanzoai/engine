//! # Hanzo Engine
//!
//! Canonical inference + embedding engine for the Hanzo stack, built on
//! top of `mistralrs-core`.
//!
//! Consumers (hanzo-vm precompiles, RPC handlers, agents) call
//! [`infer`] / [`embed`] which dispatch through a process-wide registry.
//! At startup the runtime registers one [`InferenceEngine`] and one
//! [`EmbeddingEngine`] — typically a [`MistralEngine`] loaded from a
//! Hugging Face repo or a local path.
//!
//! ```no_run
//! use std::sync::Arc;
//! use hanzo_engine::{MistralEngine, register_inference_engine, infer};
//!
//! # async fn boot() -> anyhow::Result<()> {
//! // 1) Load a model
//! let engine = MistralEngine::from_hf_repo("Qwen/Qwen3-4B").await?;
//! let id = *engine.model_id();
//!
//! // 2) Register it globally
//! register_inference_engine(Arc::new(engine))?;
//!
//! // 3) Sync call sites (EVM precompiles, etc.) dispatch through `infer`
//! let bytes = infer(&id, b"What is Rust?").unwrap();
//! println!("{}", String::from_utf8_lossy(&bytes));
//! # Ok(())
//! # }
//! ```
//!
//! ## Why a registry?
//!
//! The hanzo-vm precompiles `0x0201` (AI inference) and `0x0202` (AI
//! embedding) run in a synchronous EVM context. They cannot pass a model
//! handle through the EVM stack. Instead they look up the registered
//! engine at call time. The 32-byte `model_id` argument lets the engine
//! validate that the requested model is the one it has loaded.

pub mod api;
pub mod mistral_engine;

pub use api::{
    embed, embedding_engine_registered, infer, inference_engine_registered,
    register_embedding_engine, register_inference_engine, EmbeddingEngine, EngineError,
    InferenceEngine,
};
pub use mistral_engine::MistralEngine;
