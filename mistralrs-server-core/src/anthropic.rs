//! Anthropic-compatible `/v1/messages` endpoint.
//!
//! Translates Anthropic Messages API requests into mistral.rs internal requests
//! via the existing chat-completion pipeline, then emits responses in Anthropic
//! shape — both non-streaming JSON and streaming SSE (`message_start`,
//! `content_block_*`, `message_delta`, `message_stop`).
//!
//! Claude Code, the Anthropic SDK, and any tool-using client that speaks the
//! Anthropic Messages API can target hanzo-engine directly via this route.

use std::{
    collections::HashMap,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use axum::{
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, KeepAliveStream, Sse},
        IntoResponse,
    },
    Json,
};
use either::Either;
use indexmap::IndexMap;
use mistralrs_core::{
    ChatCompletionChunkResponse, ChatCompletionResponse, Constraint, DrySamplingParams, Function,
    MessageContent, MistralRs, NormalRequest, Request, RequestMessage, Response, SamplingParams,
    StopTokens, Tool, ToolChoice, ToolType,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::{Receiver, Sender};
use uuid::Uuid;

use crate::{
    handler_core::{create_response_channel, send_request_with_model},
    streaming::{get_keep_alive_interval, DoneState},
    types::{ExtractedMistralRsState, SharedMistralRsState},
    util::validate_model_name,
};

// =============================================================================
// Anthropic request types
// =============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<AnthropicSystem>,
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub thinking: Option<AnthropicThinking>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicMessageContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlockIn>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlockIn {
    Text {
        text: String,
    },
    Image {
        source: AnthropicImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<Value>,
        #[serde(default)]
        is_error: Option<bool>,
    },
    Thinking {
        thinking: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    Text(String),
    Blocks(Vec<AnthropicSystemBlock>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    Auto,
    Any,
    Tool { name: String },
    None,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicThinking {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(default)]
    pub budget_tokens: Option<u32>,
}

// =============================================================================
// Anthropic response types
// =============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlockOut>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlockOut {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    Thinking {
        thinking: String,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

// =============================================================================
// Translation: Anthropic request -> mistralrs internal Request
// =============================================================================

fn collect_system_text(system: &Option<AnthropicSystem>) -> Option<String> {
    match system {
        None => None,
        Some(AnthropicSystem::Text(s)) => Some(s.clone()),
        Some(AnthropicSystem::Blocks(blocks)) => {
            let parts: Vec<String> = blocks.iter().map(|b| b.text.clone()).collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n\n"))
            }
        }
    }
}

fn flatten_message_content(content: &AnthropicMessageContent) -> String {
    match content {
        AnthropicMessageContent::Text(s) => s.clone(),
        AnthropicMessageContent::Blocks(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                match block {
                    AnthropicContentBlockIn::Text { text } => parts.push(text.clone()),
                    AnthropicContentBlockIn::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        let body = content
                            .as_ref()
                            .map(|v| match v {
                                Value::String(s) => s.clone(),
                                other => other.to_string(),
                            })
                            .unwrap_or_default();
                        let prefix = if is_error.unwrap_or(false) {
                            "[tool_result error]"
                        } else {
                            "[tool_result]"
                        };
                        parts.push(format!("{prefix} id={tool_use_id}\n{body}"));
                    }
                    AnthropicContentBlockIn::Image { .. } => {
                        parts.push("[image elided in text translation]".to_string());
                    }
                    AnthropicContentBlockIn::ToolUse { id, name, input } => {
                        parts.push(format!(
                            "[tool_use id={id} name={name}]\n{}",
                            serde_json::to_string(input).unwrap_or_default()
                        ));
                    }
                    AnthropicContentBlockIn::Thinking { thinking } => {
                        parts.push(format!("[thinking]\n{thinking}"));
                    }
                }
            }
            parts.join("\n\n")
        }
    }
}

fn anthropic_messages_to_internal(
    req: &AnthropicMessagesRequest,
) -> Vec<IndexMap<String, MessageContent>> {
    let mut out: Vec<IndexMap<String, MessageContent>> = Vec::new();

    if let Some(system) = collect_system_text(&req.system) {
        let mut m: IndexMap<String, MessageContent> = IndexMap::new();
        m.insert("role".to_string(), Either::Left("system".to_string()));
        m.insert("content".to_string(), Either::Left(system));
        out.push(m);
    }

    for msg in &req.messages {
        let mut m: IndexMap<String, MessageContent> = IndexMap::new();
        let role = if msg.role == "assistant" {
            "assistant".to_string()
        } else {
            "user".to_string()
        };
        m.insert("role".to_string(), Either::Left(role));
        m.insert(
            "content".to_string(),
            Either::Left(flatten_message_content(&msg.content)),
        );
        out.push(m);
    }

    out
}

fn build_sampling_params(req: &AnthropicMessagesRequest) -> SamplingParams {
    let stop_toks = req
        .stop_sequences
        .as_ref()
        .filter(|v| !v.is_empty())
        .map(|v| StopTokens::Seqs(v.clone()));

    SamplingParams {
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        // ds4 commit 613e9b2: default min-p filtering at 0.05. Anthropic's
        // Messages API does not expose min_p; we always apply the default.
        min_p: Some(0.05),
        top_n_logprobs: 0,
        frequency_penalty: None,
        presence_penalty: None,
        repetition_penalty: None,
        max_len: Some(req.max_tokens as usize),
        stop_toks,
        logits_bias: None,
        n_choices: 1,
        dry_params: Some(DrySamplingParams::default()),
    }
}

fn anthropic_tools_to_internal(tools: &[AnthropicTool]) -> Vec<Tool> {
    tools
        .iter()
        .map(|t| {
            let parameters = match &t.input_schema {
                Value::Object(map) => {
                    let mut h: HashMap<String, Value> = HashMap::new();
                    for (k, v) in map {
                        h.insert(k.clone(), v.clone());
                    }
                    Some(h)
                }
                _ => None,
            };
            Tool {
                tp: ToolType::Function,
                function: Function {
                    description: t.description.clone(),
                    name: t.name.clone(),
                    parameters,
                },
            }
        })
        .collect()
}

fn anthropic_tool_choice_to_internal(
    choice: &AnthropicToolChoice,
    tools: &[Tool],
) -> ToolChoice {
    match choice {
        AnthropicToolChoice::Auto | AnthropicToolChoice::Any => ToolChoice::Auto,
        AnthropicToolChoice::None => ToolChoice::None,
        AnthropicToolChoice::Tool { name } => tools
            .iter()
            .find(|t| t.function.name == *name)
            .cloned()
            .map(ToolChoice::Tool)
            .unwrap_or(ToolChoice::Auto),
    }
}

fn build_internal_request(
    req: AnthropicMessagesRequest,
    tx: Sender<Response>,
) -> (Request, bool, String) {
    let id = format!("msg_{}", Uuid::new_v4().simple());
    let is_streaming = req.stream.unwrap_or(false);
    let enable_thinking = req
        .thinking
        .as_ref()
        .map(|t| t.thinking_type == "enabled");
    let messages = anthropic_messages_to_internal(&req);

    let tools: Option<Vec<Tool>> = req
        .tools
        .as_ref()
        .map(|t| anthropic_tools_to_internal(t))
        .filter(|v| !v.is_empty());

    let tool_choice = req
        .tool_choice
        .as_ref()
        .and_then(|c| {
            tools
                .as_ref()
                .map(|t| anthropic_tool_choice_to_internal(c, t))
        });

    let sampling_params = build_sampling_params(&req);
    let model_id = if req.model == "default" {
        None
    } else {
        Some(req.model.clone())
    };

    let normal = NormalRequest {
        id: 0,
        messages: RequestMessage::Chat {
            messages,
            enable_thinking,
            reasoning_effort: None,
        },
        sampling_params,
        response: tx,
        return_logprobs: false,
        is_streaming,
        suffix: None,
        constraint: Constraint::None,
        tool_choice,
        tools,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        model_id,
        truncate_sequence: false,
    };

    (Request::Normal(Box::new(normal)), is_streaming, id)
}

// =============================================================================
// Translation: ChatCompletionResponse -> Anthropic
// =============================================================================

fn map_finish_reason(finish: &str) -> String {
    match finish {
        "stop" => "end_turn".to_string(),
        "length" => "max_tokens".to_string(),
        "tool_calls" => "tool_use".to_string(),
        "content_filter" => "stop_sequence".to_string(),
        other => other.to_string(),
    }
}

fn chat_response_to_anthropic(
    id: String,
    model: String,
    resp: &ChatCompletionResponse,
) -> AnthropicMessagesResponse {
    let mut content_blocks: Vec<AnthropicContentBlockOut> = Vec::new();
    let mut stop_reason: Option<String> = None;

    if let Some(choice) = resp.choices.first() {
        if let Some(reasoning) = choice.message.reasoning_content.as_deref() {
            if !reasoning.is_empty() {
                content_blocks.push(AnthropicContentBlockOut::Thinking {
                    thinking: reasoning.to_string(),
                });
            }
        }
        if let Some(text) = &choice.message.content {
            if !text.is_empty() {
                content_blocks.push(AnthropicContentBlockOut::Text { text: text.clone() });
            }
        }
        if let Some(tool_calls) = &choice.message.tool_calls {
            for call in tool_calls {
                let input = serde_json::from_str::<Value>(&call.function.arguments)
                    .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));
                content_blocks.push(AnthropicContentBlockOut::ToolUse {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    input,
                });
            }
        }
        stop_reason = Some(map_finish_reason(&choice.finish_reason));
    }

    let usage = AnthropicUsage {
        input_tokens: resp.usage.prompt_tokens as u32,
        output_tokens: resp.usage.completion_tokens as u32,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: None,
    };

    AnthropicMessagesResponse {
        id,
        message_type: "message".to_string(),
        role: "assistant".to_string(),
        content: content_blocks,
        model,
        stop_reason,
        stop_sequence: None,
        usage,
    }
}

// =============================================================================
// Responder + handler
// =============================================================================

pub enum AnthropicResponder {
    Json(AnthropicMessagesResponse),
    Sse(Sse<KeepAliveStream<AnthropicMessagesStreamer>>),
    Error(StatusCode, String, &'static str),
}

impl IntoResponse for AnthropicResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            AnthropicResponder::Json(v) => Json(v).into_response(),
            AnthropicResponder::Sse(s) => s.into_response(),
            AnthropicResponder::Error(code, msg, kind) => {
                // ds4 commit be43477: emit Anthropic-shaped error with the
                // right error.type per Anthropic spec. Caller picks the
                // protocol kind; we re-classify obvious cases here for
                // request-level errors that the engine still labels generic.
                let final_kind = classify_error_kind(kind, &msg);
                let body = serde_json::json!({
                    "type": "error",
                    "error": { "type": final_kind, "message": msg }
                });
                (code, Json(body)).into_response()
            }
        }
    }
}

/// Promote generic `api_error` to a more specific Anthropic error type
/// when the message clearly identifies the cause. Matches the ds4 be43477
/// behavior of distinguishing context-length and similar request-shape
/// errors from real server faults.
///
/// Anthropic taxonomy reference:
/// - `invalid_request_error` — bad parameters, ctx exceeded, model unknown
/// - `authentication_error` — auth header issues
/// - `permission_error` — model/account permission
/// - `not_found_error` — model/resource missing
/// - `request_too_large` — payload size
/// - `rate_limit_error` — throttled
/// - `api_error` — generic server fault
/// - `overloaded_error` — temporarily overloaded
fn classify_error_kind(initial: &'static str, msg: &str) -> &'static str {
    // Explicit callers (invalid_request_error, etc.) take precedence.
    if initial != "api_error" {
        return initial;
    }
    let lc = msg.to_ascii_lowercase();
    if lc.contains("context")
        && (lc.contains("exceed") || lc.contains("too long") || lc.contains("max"))
    {
        return "invalid_request_error";
    }
    if lc.contains("max_tokens") || lc.contains("token limit") || lc.contains("seq_len") {
        return "invalid_request_error";
    }
    if lc.contains("model") && (lc.contains("not found") || lc.contains("unknown")) {
        return "not_found_error";
    }
    if lc.contains("overload") || lc.contains("busy") {
        return "overloaded_error";
    }
    initial
}

/// Anthropic-compatible messages endpoint handler.
///
/// Documented in `mistralrs-server-core/src/anthropic.rs`. Not exposed via the
/// `swagger-ui` feature because the Anthropic request types are not annotated
/// with `utoipa::ToSchema` — they are deserialized via serde untagged unions
/// (`AnthropicMessageContent`, `AnthropicSystem`) which do not have a single
/// OpenAPI schema representation.
pub async fn messages(
    axum::extract::State(state): ExtractedMistralRsState,
    Json(req): Json<AnthropicMessagesRequest>,
) -> AnthropicResponder {
    if let Err(e) = validate_model_name(&req.model, state.clone()) {
        return AnthropicResponder::Error(
            StatusCode::BAD_REQUEST,
            e.to_string(),
            "invalid_request_error",
        );
    }

    let (tx, rx) = create_response_channel(None);
    let model_name = req.model.clone();
    let model_id_route = if req.model == "default" {
        None
    } else {
        Some(req.model.clone())
    };

    let (request, is_streaming, message_id) = build_internal_request(req, tx);

    if let Err(e) = send_request_with_model(&state, request, model_id_route.as_deref()).await {
        return AnthropicResponder::Error(
            StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
            "api_error",
        );
    }

    if is_streaming {
        let streamer = AnthropicMessagesStreamer::new(rx, state, message_id, model_name);
        let keep_alive_interval = get_keep_alive_interval();
        AnthropicResponder::Sse(
            Sse::new(streamer)
                .keep_alive(KeepAlive::new().interval(Duration::from_millis(keep_alive_interval))),
        )
    } else {
        let mut rx = rx;
        match rx.recv().await {
            Some(Response::Done(chat_resp)) => {
                MistralRs::maybe_log_response(state, &chat_resp);
                let response = chat_response_to_anthropic(message_id, model_name, &chat_resp);
                AnthropicResponder::Json(response)
            }
            Some(Response::ModelError(msg, _partial)) => AnthropicResponder::Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                msg,
                "api_error",
            ),
            Some(Response::ValidationError(e)) => AnthropicResponder::Error(
                StatusCode::UNPROCESSABLE_ENTITY,
                e.to_string(),
                "invalid_request_error",
            ),
            Some(Response::InternalError(e)) => AnthropicResponder::Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string(),
                "api_error",
            ),
            _ => AnthropicResponder::Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Unexpected response variant".to_string(),
                "api_error",
            ),
        }
    }
}

// =============================================================================
// Streamer
// =============================================================================

struct ToolBlockState {
    /// SSE content-block index for this tool call.
    index: usize,
}

pub struct AnthropicMessagesStreamer {
    rx: Receiver<Response>,
    state: SharedMistralRsState,
    message_id: String,
    model: String,
    sent_message_start: bool,
    text_block_open: bool,
    text_block_index: usize,
    /// Next free SSE content-block index.
    next_index: usize,
    /// Tool block state keyed by OpenAI tool-call index.
    tool_blocks: HashMap<usize, ToolBlockState>,
    /// Pending events to emit before reading more from rx.
    pending: Vec<Event>,
    done_state: DoneState,
    final_stop_reason: Option<String>,
    final_input_tokens: u32,
    final_output_tokens: u32,
}

impl AnthropicMessagesStreamer {
    fn new(
        rx: Receiver<Response>,
        state: SharedMistralRsState,
        message_id: String,
        model: String,
    ) -> Self {
        Self {
            rx,
            state,
            message_id,
            model,
            sent_message_start: false,
            text_block_open: false,
            text_block_index: 0,
            next_index: 0,
            tool_blocks: HashMap::new(),
            pending: Vec::new(),
            done_state: DoneState::Running,
            final_stop_reason: None,
            final_input_tokens: 0,
            final_output_tokens: 0,
        }
    }

    fn emit(&mut self, name: &'static str, payload: Value) {
        if let Ok(ev) = Event::default().event(name).json_data(payload) {
            self.pending.push(ev);
        }
    }

    fn ensure_message_start(&mut self) {
        if self.sent_message_start {
            return;
        }
        self.sent_message_start = true;
        let initial = AnthropicMessagesResponse {
            id: self.message_id.clone(),
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![],
            model: self.model.clone(),
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        self.emit(
            "message_start",
            serde_json::json!({ "type": "message_start", "message": initial }),
        );
    }

    fn ensure_text_block(&mut self) -> usize {
        if !self.text_block_open {
            let idx = self.next_index;
            self.next_index += 1;
            self.text_block_index = idx;
            self.text_block_open = true;
            self.emit(
                "content_block_start",
                serde_json::json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": { "type": "text", "text": "" },
                }),
            );
            idx
        } else {
            self.text_block_index
        }
    }

    fn close_text_block(&mut self) {
        if self.text_block_open {
            let idx = self.text_block_index;
            self.emit(
                "content_block_stop",
                serde_json::json!({ "type": "content_block_stop", "index": idx }),
            );
            self.text_block_open = false;
        }
    }

    fn handle_chunk(&mut self, chunk: &ChatCompletionChunkResponse) {
        self.ensure_message_start();

        let Some(choice) = chunk.choices.first() else {
            return;
        };

        if let Some(text) = choice.delta.content.as_deref() {
            if !text.is_empty() {
                let idx = self.ensure_text_block();
                self.emit(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": { "type": "text_delta", "text": text },
                    }),
                );
            }
        }

        if let Some(reasoning) = choice.delta.reasoning_content.as_deref() {
            if !reasoning.is_empty() {
                let idx = self.ensure_text_block();
                self.emit(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": { "type": "thinking_delta", "thinking": reasoning },
                    }),
                );
            }
        }

        if let Some(tool_call_deltas) = choice.delta.tool_calls.as_ref() {
            for tc in tool_call_deltas {
                let oai_idx = tc.index;
                if !self.tool_blocks.contains_key(&oai_idx) {
                    self.close_text_block();
                    let id = if tc.id.is_empty() {
                        format!("toolu_{}", Uuid::new_v4().simple())
                    } else {
                        tc.id.clone()
                    };
                    let idx = self.next_index;
                    self.next_index += 1;
                    self.tool_blocks.insert(oai_idx, ToolBlockState { index: idx });
                    self.emit(
                        "content_block_start",
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": id,
                                "name": tc.function.name,
                                "input": {},
                            },
                        }),
                    );
                }

                let args = &tc.function.arguments;
                if !args.is_empty() {
                    if let Some(state) = self.tool_blocks.get(&oai_idx) {
                        let idx = state.index;
                        self.emit(
                            "content_block_delta",
                            serde_json::json!({
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": args,
                                },
                            }),
                        );
                    }
                }
            }
        }

        if let Some(finish) = choice.finish_reason.as_deref() {
            self.final_stop_reason = Some(map_finish_reason(finish));
        }
        if let Some(usage) = chunk.usage.as_ref() {
            self.final_input_tokens = usage.prompt_tokens as u32;
            self.final_output_tokens = usage.completion_tokens as u32;
        }
    }

    fn finalize(&mut self) {
        self.close_text_block();
        let tool_indices: Vec<usize> = self.tool_blocks.values().map(|s| s.index).collect();
        for idx in tool_indices {
            self.emit(
                "content_block_stop",
                serde_json::json!({ "type": "content_block_stop", "index": idx }),
            );
        }
        self.emit(
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": self.final_stop_reason.clone().unwrap_or_else(|| "end_turn".to_string()),
                    "stop_sequence": Value::Null,
                },
                "usage": {
                    "input_tokens": self.final_input_tokens,
                    "output_tokens": self.final_output_tokens,
                },
            }),
        );
        self.emit(
            "message_stop",
            serde_json::json!({ "type": "message_stop" }),
        );
    }
}

impl futures::Stream for AnthropicMessagesStreamer {
    type Item = std::result::Result<Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if !self.pending.is_empty() {
            let ev = self.pending.remove(0);
            return Poll::Ready(Some(Ok(ev)));
        }

        match self.done_state {
            DoneState::SendingDone => {
                self.done_state = DoneState::Done;
                return Poll::Ready(None);
            }
            DoneState::Done => return Poll::Ready(None),
            DoneState::Running => {}
        }

        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(resp)) => {
                match resp {
                    Response::Chunk(chunk) => {
                        let is_final = !chunk.choices.is_empty()
                            && chunk.choices.iter().all(|c| c.finish_reason.is_some());
                        self.handle_chunk(&chunk);
                        if is_final {
                            self.finalize();
                            self.done_state = DoneState::SendingDone;
                        }
                    }
                    Response::ModelError(msg, _) => {
                        self.ensure_message_start();
                        self.emit(
                            "error",
                            serde_json::json!({
                                "type": "error",
                                "error": { "type": "api_error", "message": msg },
                            }),
                        );
                        self.done_state = DoneState::SendingDone;
                    }
                    Response::ValidationError(e) => {
                        self.ensure_message_start();
                        self.emit(
                            "error",
                            serde_json::json!({
                                "type": "error",
                                "error": { "type": "invalid_request_error", "message": e.to_string() },
                            }),
                        );
                        self.done_state = DoneState::SendingDone;
                    }
                    Response::InternalError(e) => {
                        self.ensure_message_start();
                        self.emit(
                            "error",
                            serde_json::json!({
                                "type": "error",
                                "error": { "type": "api_error", "message": e.to_string() },
                            }),
                        );
                        self.done_state = DoneState::SendingDone;
                    }
                    _ => {}
                }

                if !self.pending.is_empty() {
                    let ev = self.pending.remove(0);
                    Poll::Ready(Some(Ok(ev)))
                } else {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
            Poll::Ready(None) => {
                if !matches!(self.done_state, DoneState::SendingDone | DoneState::Done) {
                    self.finalize();
                    self.done_state = DoneState::SendingDone;
                    if !self.pending.is_empty() {
                        let ev = self.pending.remove(0);
                        return Poll::Ready(Some(Ok(ev)));
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

// Keep `state` referenced even when not used in some build configurations.
#[allow(dead_code)]
fn _state_used(s: &SharedMistralRsState) {
    let _ = s;
}
