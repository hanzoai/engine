//! # Hanzo Engine
//!
//! Canonical inference + embedding engine for the Hanzo stack.
//!
//! Consumers (hanzo-vm precompiles, RPC handlers, agents) call
//! [`infer`] / [`embed`] which dispatch through a process-wide registry.
//! At startup the runtime registers one [`InferenceEngine`] and one
//! [`EmbeddingEngine`] — typically a [`MistralEngine`] loaded from a
//! Hugging Face repo or a local path.
//!
//! ## Build status
//!
//! This crate currently ships the registry + traits + stub backends.
//! The real model loader lived in `mistralrs-core` and the SDK facade
//! `hanzo`, both of which are mid-rename and don't compile. Until the
//! engine source tree is restored, `MistralEngine::from_hf_repo` and
//! `register_zen5_engines_at_startup` return / log "no backend" — the
//! HTTP API still boots, the precompiles still dispatch, they just
//! report a runtime error instead of producing tokens.
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
pub mod zen5_engine;

pub use api::{
    embed, embedding_engine_registered, infer, inference_engine_registered,
    register_embedding_engine, register_inference_engine, EmbeddingEngine, EngineError,
    InferenceEngine,
};
pub use mistral_engine::MistralEngine;
pub use zen5_engine::{
    build_registry as build_zen5_registry, model_id_for as zen5_model_id_for,
    register_zen5_engines_at_startup, DEFAULT_VARIANTS,
};
