//! ZAP internal RPC surface for the engine<>node link.
//!
//! ZAP (Zero-Copy App Proto, the `hanzo-zap` crate, lib `zap`) is the binary
//! Cap'n Proto wire for INTERNAL hanzo service comms. This module is the
//! engine-side surface: a server adapter that drives the real `Hanzo` engine
//! over its existing `Sender<Request>` channel, plus a client trait the node
//! (or any first-party caller) implements/consumes.
//!
//! Scope: this replaces the INTERNAL JSON the node POSTs to the engine's
//! OpenAI HTTP API for first-party traffic. The EXTERNAL OpenAI/Anthropic
//! HTTP+JSON API in `hanzo-server-core` is untouched; ZAP is a parallel
//! transport, gated behind the `zap` cargo feature.
//!
//! Wire schema: `hanzo-engine/schema/inference.capnp`. The message structs
//! here mirror that schema 1:1 and bridge to `hanzo_engine::{Request,Response}`.
//! The capnp/capnp-rpc transport (TCP/unix via `zap://`) is the drop-in step:
//! `transport` documents the wiring; it is intentionally not compiled here so
//! the engine gains no `capnp`/`capnpc` build dependency before that step.

mod message;
mod server;
mod client;
pub mod transport;

pub use client::ZapInference;
pub use message::{
    ChatChunk, ChatRequest, ChatResponse, Constraint, DetokenizeRequest, HealthStatus, InferError,
    InferErrorKind, Message, ModelInfo, ModelList, ReIsqRequest, Role, SamplingParams, Tool,
    ToolCall, ToolChoiceKind, TokenizeRequest, Usage,
};
pub use server::ZapInferenceServer;

/// Default TCP port for the ZAP inference endpoint (distinct from the OpenAI
/// HTTP server port). Matches `zap::DEFAULT_PORT`.
pub const ZAP_DEFAULT_PORT: u16 = 9999;
