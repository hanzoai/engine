//! ZAP inference client trait.
//!
//! The consumer side of the engine<>node link. The node (or any first-party
//! caller) drives the engine through this trait instead of POSTing OpenAI JSON
//! to `/ai/chat/completions`. A capnp-rpc-backed implementation lives behind
//! the transport (see `transport`); the in-process `ZapInferenceServer` also
//! satisfies the same shape for tests and single-process deployments.

use async_trait::async_trait;

use super::message::{
    ChatChunk, ChatRequest, ChatResponse, DetokenizeRequest, HealthStatus, InferError, ModelList,
    ReIsqRequest, TokenizeRequest,
};

/// Internal inference RPC, the binary counterpart to the engine's OpenAI HTTP
/// API for first-party callers. Maps 1:1 to the `Inference` interface in
/// `schema/inference.capnp`.
#[async_trait]
pub trait ZapInference: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, InferError>;

    /// Streaming chat. Each chunk is delivered to `on_chunk`; resolves with the
    /// final aggregate response. Counterpart to `chatStream(request, sink)`.
    async fn chat_stream(
        &self,
        request: ChatRequest,
        on_chunk: Box<dyn FnMut(ChatChunk) + Send>,
    ) -> Result<ChatResponse, InferError>;

    async fn tokenize(&self, request: TokenizeRequest) -> Result<Vec<u32>, InferError>;
    async fn detokenize(&self, request: DetokenizeRequest) -> Result<String, InferError>;

    async fn list_models(&self) -> Result<ModelList, InferError>;
    async fn health(&self) -> Result<HealthStatus, InferError>;
    /// serde_json bytes of `hanzo_engine::SystemInfo`.
    async fn system_info(&self) -> Result<Vec<u8>, InferError>;
    async fn re_isq(&self, request: ReIsqRequest) -> Result<(), InferError>;
}

/// The in-process server satisfies the client trait directly, so callers in the
/// same process can use ZAP types without a socket. The capnp transport adds
/// the out-of-process `zap://` path with the identical surface.
#[async_trait]
impl ZapInference for super::server::ZapInferenceServer {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, InferError> {
        ZapInferenceServer::chat(self, request).await
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
        mut on_chunk: Box<dyn FnMut(ChatChunk) + Send>,
    ) -> Result<ChatResponse, InferError> {
        ZapInferenceServer::chat_stream(self, request, |c| on_chunk(c)).await
    }

    async fn tokenize(&self, request: TokenizeRequest) -> Result<Vec<u32>, InferError> {
        ZapInferenceServer::tokenize(self, request).await
    }

    async fn detokenize(&self, request: DetokenizeRequest) -> Result<String, InferError> {
        ZapInferenceServer::detokenize(self, request).await
    }

    async fn list_models(&self) -> Result<ModelList, InferError> {
        Ok(ZapInferenceServer::list_models(self))
    }

    async fn health(&self) -> Result<HealthStatus, InferError> {
        Ok(ZapInferenceServer::health(self))
    }

    async fn system_info(&self) -> Result<Vec<u8>, InferError> {
        ZapInferenceServer::system_info_json(self)
    }

    async fn re_isq(&self, request: ReIsqRequest) -> Result<(), InferError> {
        ZapInferenceServer::re_isq(self, request).await
    }
}

use super::server::ZapInferenceServer;
