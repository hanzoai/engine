//! ZAP inference server: the engine-side adapter.
//!
//! Wraps `Arc<Hanzo>` and serves the ZAP `Inference` interface
//! (`schema/inference.capnp`) by translating wire messages into the engine's
//! existing `Request`/`Response` mpsc protocol via `Hanzo::get_sender`. This is
//! exactly what the OpenAI HTTP handlers in `hanzo-server-core` do, minus the
//! HTTP+JSON layer; ZAP is the binary alternative for first-party callers.
//!
//! The capnp-rpc glue (implementing the generated `inference::Server` trait and
//! serving it over a `zap://` transport) wraps this struct unchanged; see
//! `transport`. Methods here are transport-agnostic so they are unit-testable
//! without the capnp toolchain.

use std::sync::Arc;

use crate::response::Response;
use crate::Hanzo;

use super::message::{
    ChatChunk, ChatRequest, ChatResponse, DetokenizeRequest, HealthStatus, InferError,
    InferErrorKind, ModelInfo, ModelList, ReIsqRequest, TokenizeRequest,
};

/// mpsc buffer for the per-request response channel. Streaming pushes many
/// chunks; a small buffer bounds memory while keeping the engine unblocked.
const RESPONSE_CHANNEL_BUF: usize = 256;

/// Engine-side ZAP inference service.
#[derive(Clone)]
pub struct ZapInferenceServer {
    engine: Arc<Hanzo>,
}

fn infer_err(kind: InferErrorKind, e: impl std::fmt::Display) -> InferError {
    InferError {
        kind,
        message: e.to_string(),
    }
}

impl ZapInferenceServer {
    pub fn new(engine: Arc<Hanzo>) -> Self {
        Self { engine }
    }

    /// Non-streaming chat. Mirrors `POST /v1/chat/completions` (stream=false):
    /// submit `Request::Normal`, await the terminal `Response`.
    pub async fn chat(&self, mut request: ChatRequest) -> Result<ChatResponse, InferError> {
        request.stream = false;
        let model_id = request.model_id.clone();
        let (tx, mut rx) = tokio::sync::mpsc::channel(RESPONSE_CHANNEL_BUF);
        let sender = self
            .engine
            .get_sender(model_id.as_deref())
            .map_err(|e| infer_err(InferErrorKind::Validation, e))?;

        let req = request.into_request(tx);
        sender
            .send(req)
            .await
            .map_err(|e| infer_err(InferErrorKind::Internal, e))?;

        while let Some(resp) = rx.recv().await {
            match resp {
                Response::Done(r) => return Ok(ChatResponse::from_completion(r)),
                Response::ModelError(msg, _) => {
                    return Err(infer_err(InferErrorKind::Model, msg))
                }
                Response::ValidationError(e) => {
                    return Err(infer_err(InferErrorKind::Validation, e))
                }
                Response::InternalError(e) => {
                    return Err(infer_err(InferErrorKind::Internal, e))
                }
                // Ignore mid-stream/agentic progress events for the non-streaming path.
                _ => continue,
            }
        }
        Err(infer_err(
            InferErrorKind::Internal,
            "engine closed response channel without a terminal response",
        ))
    }

    /// Streaming chat. Mirrors stream=true (SSE today). Pushes each `ChatChunk`
    /// to `on_chunk`; resolves with the final aggregate `ChatResponse`.
    /// On the wire this is `chatStream(request, sink)`; `on_chunk` is the
    /// `ResponseStream` capability.
    pub async fn chat_stream<F>(
        &self,
        mut request: ChatRequest,
        mut on_chunk: F,
    ) -> Result<ChatResponse, InferError>
    where
        F: FnMut(ChatChunk),
    {
        request.stream = true;
        let model_id = request.model_id.clone();
        let (tx, mut rx) = tokio::sync::mpsc::channel(RESPONSE_CHANNEL_BUF);
        let sender = self
            .engine
            .get_sender(model_id.as_deref())
            .map_err(|e| infer_err(InferErrorKind::Validation, e))?;

        let req = request.into_request(tx);
        sender
            .send(req)
            .await
            .map_err(|e| infer_err(InferErrorKind::Internal, e))?;

        let mut last = ChatResponse {
            id: String::new(),
            model_id: model_id.unwrap_or_default(),
            content: String::new(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            finish_reason: String::new(),
            usage: Default::default(),
            session_id: None,
        };

        while let Some(resp) = rx.recv().await {
            match resp {
                Response::Chunk(chunk) => {
                    let choice = chunk.choices.into_iter().next();
                    let (delta, reasoning, finish, tool_calls) = match choice {
                        Some(c) => (
                            c.delta.content.unwrap_or_default(),
                            c.delta.reasoning_content,
                            c.finish_reason,
                            c.delta
                                .tool_calls
                                .unwrap_or_default()
                                .into_iter()
                                .map(|tc| super::message::ToolCall {
                                    id: tc.id,
                                    name: tc.function.name,
                                    arguments: tc.function.arguments,
                                })
                                .collect(),
                        ),
                        None => (String::new(), None, None, Vec::new()),
                    };
                    last.id = chunk.id.clone();
                    last.content.push_str(&delta);
                    let usage = chunk.usage.map(|u| super::message::Usage {
                        prompt_tokens: u.prompt_tokens as u32,
                        completion_tokens: u.completion_tokens as u32,
                        total_tokens: u.total_tokens as u32,
                        avg_tok_per_sec: u.avg_compl_tok_per_sec,
                        total_time_sec: u.total_time_sec,
                    });
                    if let Some(ref u) = usage {
                        last.usage = u.clone();
                    }
                    let finish_reason = finish.unwrap_or_default();
                    if !finish_reason.is_empty() {
                        last.finish_reason = finish_reason.clone();
                    }
                    on_chunk(ChatChunk {
                        id: chunk.id,
                        delta_content: delta,
                        delta_reasoning: reasoning,
                        tool_calls,
                        finish_reason,
                        usage,
                    });
                }
                Response::Done(r) => return Ok(ChatResponse::from_completion(r)),
                Response::ModelError(msg, _) => {
                    return Err(infer_err(InferErrorKind::Model, msg))
                }
                Response::ValidationError(e) => {
                    return Err(infer_err(InferErrorKind::Validation, e))
                }
                Response::InternalError(e) => {
                    return Err(infer_err(InferErrorKind::Internal, e))
                }
                _ => continue,
            }
        }
        Ok(last)
    }

    pub async fn tokenize(&self, request: TokenizeRequest) -> Result<Vec<u32>, InferError> {
        // Bridges to `Request::Tokenize`; wiring mirrors `chat` (build the
        // engine TokenizationRequest with a oneshot responder). Left as the
        // next adapter step; the wire surface is defined.
        let _ = request;
        Err(infer_err(
            InferErrorKind::Internal,
            "tokenize over ZAP not yet wired",
        ))
    }

    pub async fn detokenize(&self, request: DetokenizeRequest) -> Result<String, InferError> {
        let _ = request;
        Err(infer_err(
            InferErrorKind::Internal,
            "detokenize over ZAP not yet wired",
        ))
    }

    /// Mirrors `GET /v1/models`.
    pub fn list_models(&self) -> ModelList {
        let models = self
            .engine
            .list_models_with_status()
            .unwrap_or_default()
            .into_iter()
            .map(|(id, status)| ModelInfo {
                loaded: status == crate::ModelStatus::Loaded,
                kind: status.to_string(),
                id,
            })
            .collect();
        ModelList { models }
    }

    /// Mirrors `GET /health`.
    pub fn health(&self) -> HealthStatus {
        HealthStatus {
            ok: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Mirrors `GET /v1/system/info`. Returns serde_json bytes of `SystemInfo`
    /// (the existing serializer), so no diagnostics re-modeling is needed.
    pub fn system_info_json(&self) -> Result<Vec<u8>, InferError> {
        let info = crate::collect_system_info();
        serde_json::to_vec(&info).map_err(|e| infer_err(InferErrorKind::Internal, e))
    }

    /// Mirrors `POST /re_isq`.
    pub async fn re_isq(&self, request: ReIsqRequest) -> Result<(), InferError> {
        let _ = request;
        Err(infer_err(
            InferErrorKind::Internal,
            "re_isq over ZAP not yet wired",
        ))
    }
}
