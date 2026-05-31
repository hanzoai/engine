//! Native Anthropic Messages API (`POST /v1/messages`) for Claude Code harness support.
//!
//! Translates an Anthropic Messages request into the internal chat pipeline
//! (the same path as `/v1/chat/completions`) and translates the result back to
//! the Anthropic response shape. Both non-streaming (single JSON body) and
//! streaming (SSE event sequence) responses are supported, and tool_use /
//! tool_result content blocks are translated bidirectionally for full agentic
//! Claude Code compatibility.

use std::{pin::Pin, task::Poll, time::Duration};

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, KeepAliveStream},
        IntoResponse, Sse,
    },
    Extension, Json,
};
use hanzo_engine::{ChatCompletionChunkResponse, Hanzo, Response, ToolCallResponse};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc::Receiver;
use uuid::Uuid;

use crate::{
    chat_completion::{parse_request, process_non_streaming_response, ChatCompletionResponder},
    handler_core::{
        create_response_channel, send_request_with_model, ErrorToResponse, JsonError,
        ModelErrorMessage,
    },
    hanzo_server_router_builder::AgenticDefaults,
    openai::ChatCompletionRequest,
    streaming::{get_keep_alive_interval, DoneState},
    types::{ExtractedHanzoState, SharedHanzoState},
    util::sanitize_error_message,
};

#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: usize,
    #[serde(default)]
    pub system: Option<Value>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    /// String or array of content blocks (text/image/tool_use/tool_result).
    pub content: Value,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: Value,
    },
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<AnthropicContentBlock>,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// (event_name, json_payload) intermediary so we can unit-test the SSE sequence without parsing
/// `axum::response::sse::Event` (which has no public accessors).
pub(crate) type NamedEvent = (String, Value);

/// Anthropic content can be a bare string or an array of typed blocks; collect text-only.
fn content_text_only(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Per-message expansion: an Anthropic message with mixed content blocks may produce
/// multiple OpenAI messages (assistant w/ tool_calls, separate `tool` role messages).
fn translate_message(role: &str, content: &Value, out: &mut Vec<Value>) {
    match content {
        Value::String(_) => {
            out.push(json!({"role": role, "content": content_text_only(content)}));
        }
        Value::Array(blocks) => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<Value> = Vec::new();
            let mut tool_results: Vec<(String, String)> = Vec::new();
            for b in blocks {
                let ty = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match ty {
                    "text" => {
                        if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                            text_parts.push(t.to_string());
                        }
                    }
                    "tool_use" => {
                        let id = b
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = b
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = b.get("input").cloned().unwrap_or(json!({}));
                        let arguments = if input.is_string() {
                            input.as_str().unwrap_or("").to_string()
                        } else {
                            serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string())
                        };
                        tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }));
                    }
                    "tool_result" => {
                        let id = b
                            .get("tool_use_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let c = b
                            .get("content")
                            .cloned()
                            .unwrap_or(Value::String(String::new()));
                        let body = match &c {
                            Value::String(s) => s.clone(),
                            Value::Array(_) => content_text_only(&c),
                            other => serde_json::to_string(other).unwrap_or_default(),
                        };
                        tool_results.push((id, body));
                    }
                    _ => {}
                }
            }
            let text = text_parts.join("\n");
            if role == "assistant" && !tool_calls.is_empty() {
                let mut msg = json!({"role": "assistant"});
                if !text.is_empty() {
                    msg["content"] = json!(text);
                } else {
                    msg["content"] = Value::Null;
                }
                msg["tool_calls"] = json!(tool_calls);
                out.push(msg);
            } else if !text.is_empty() || tool_results.is_empty() {
                out.push(json!({"role": role, "content": text}));
            }
            for (tool_use_id, body) in tool_results {
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": body,
                }));
            }
        }
        _ => {
            out.push(json!({"role": role, "content": ""}));
        }
    }
}

fn anthropic_tools_to_openai(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            let name = t.get("name").cloned().unwrap_or(json!(""));
            let description = t.get("description").cloned().unwrap_or(json!(""));
            let parameters = t
                .get("input_schema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"}));
            json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            })
        })
        .collect()
}

fn anthropic_to_openai(
    areq: &AnthropicMessagesRequest,
) -> Result<ChatCompletionRequest, serde_json::Error> {
    let mut messages: Vec<Value> = Vec::with_capacity(areq.messages.len() + 1);
    if let Some(sys) = &areq.system {
        let sys = content_text_only(sys);
        if !sys.is_empty() {
            messages.push(json!({"role": "system", "content": sys}));
        }
    }
    for m in &areq.messages {
        translate_message(&m.role, &m.content, &mut messages);
    }
    // Claude Code sends Anthropic model ids (claude-sonnet-4-5-*, ...); route them to the
    // single loaded model. The response still echoes the originally requested id.
    let model = if areq.model.to_ascii_lowercase().starts_with("claude") {
        "default"
    } else {
        areq.model.as_str()
    };
    let mut obj = json!({
        "model": model,
        "messages": messages,
        "max_tokens": areq.max_tokens,
        "stream": false,
    });
    if let Some(t) = areq.temperature {
        obj["temperature"] = json!(t);
    }
    if let Some(p) = areq.top_p {
        obj["top_p"] = json!(p);
    }
    if let Some(s) = &areq.stop_sequences {
        obj["stop"] = json!(s);
    }
    if let Some(t) = &areq.tools {
        obj["tools"] = json!(anthropic_tools_to_openai(t));
    }
    if let Some(tc) = &areq.tool_choice {
        obj["tool_choice"] = tc.clone();
    }
    obj["enable_thinking"] = json!(false);
    serde_json::from_value(obj)
}

fn map_stop_reason(finish: &str) -> String {
    match finish {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
    .to_string()
}

/// Build the response content blocks from an OpenAI message: if tool_calls exist, emit
/// a `tool_use` block per call (preceded by any text); otherwise a single text block.
fn build_content_blocks(
    text: &str,
    tool_calls: Option<&Vec<ToolCallResponse>>,
) -> Vec<AnthropicContentBlock> {
    let mut blocks: Vec<AnthropicContentBlock> = Vec::new();
    if !text.is_empty() {
        blocks.push(AnthropicContentBlock::Text {
            text: text.to_string(),
        });
    }
    if let Some(calls) = tool_calls {
        for call in calls {
            let input: Value = serde_json::from_str(&call.function.arguments)
                .unwrap_or_else(|_| json!(call.function.arguments));
            blocks.push(AnthropicContentBlock::ToolUse {
                id: call.id.clone(),
                name: call.function.name.clone(),
                input,
            });
        }
    }
    if blocks.is_empty() {
        blocks.push(AnthropicContentBlock::Text {
            text: String::new(),
        });
    }
    blocks
}

#[derive(Default)]
pub(crate) struct StreamBuilder {
    started: bool,
    finalized: bool,
    next_index: usize,
    text_block: Option<TextBlockState>,
    tool_blocks: Vec<ToolBlockState>,
}

#[derive(Default)]
struct TextBlockState {
    index: usize,
}

#[derive(Default)]
struct ToolBlockState {
    index: usize,
    id: String,
    args_sent: String,
}

impl StreamBuilder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn start(&mut self, model: String, id: String, input_tokens: u32) -> Vec<NamedEvent> {
        self.started = true;
        let msg = json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            }
        });
        vec![
            ("message_start".to_string(), msg),
            ("ping".to_string(), json!({"type": "ping"})),
        ]
    }

    fn open_text(&mut self) -> Option<NamedEvent> {
        if self.text_block.is_some() {
            return None;
        }
        let index = self.next_index;
        self.next_index += 1;
        self.text_block = Some(TextBlockState { index });
        Some((
            "content_block_start".to_string(),
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            }),
        ))
    }

    fn close_text(&mut self) -> Option<NamedEvent> {
        let block = self.text_block.take()?;
        Some((
            "content_block_stop".to_string(),
            json!({"type": "content_block_stop", "index": block.index}),
        ))
    }

    fn text_delta(&self, text: &str) -> Option<NamedEvent> {
        let block = self.text_block.as_ref()?;
        Some((
            "content_block_delta".to_string(),
            json!({
                "type": "content_block_delta",
                "index": block.index,
                "delta": {"type": "text_delta", "text": text},
            }),
        ))
    }

    fn handle_tool_calls(&mut self, calls: &[ToolCallResponse]) -> Vec<NamedEvent> {
        let mut out = Vec::new();
        if let Some(close) = self.close_text() {
            out.push(close);
        }
        for call in calls {
            let existing = self.tool_blocks.iter().position(|b| b.id == call.id);
            match existing {
                None => {
                    let index = self.next_index;
                    self.next_index += 1;
                    out.push((
                        "content_block_start".to_string(),
                        json!({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.function.name,
                                "input": {},
                            }
                        }),
                    ));
                    let args_sent = if call.function.arguments.is_empty() {
                        String::new()
                    } else {
                        out.push((
                            "content_block_delta".to_string(),
                            json!({
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {"type": "input_json_delta", "partial_json": call.function.arguments},
                            }),
                        ));
                        call.function.arguments.clone()
                    };
                    self.tool_blocks.push(ToolBlockState {
                        index,
                        id: call.id.clone(),
                        args_sent,
                    });
                }
                Some(pos) => {
                    let block = &mut self.tool_blocks[pos];
                    let partial = call
                        .function
                        .arguments
                        .strip_prefix(block.args_sent.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| call.function.arguments.clone());
                    if !partial.is_empty() {
                        out.push((
                            "content_block_delta".to_string(),
                            json!({
                                "type": "content_block_delta",
                                "index": block.index,
                                "delta": {"type": "input_json_delta", "partial_json": partial},
                            }),
                        ));
                    }
                    block.args_sent = call.function.arguments.clone();
                }
            }
        }
        out
    }

    pub(crate) fn finalize(
        &mut self,
        finish_reason: Option<&str>,
        output_tokens: u32,
    ) -> Vec<NamedEvent> {
        if self.finalized {
            return Vec::new();
        }
        self.finalized = true;
        let mut out = Vec::new();
        if let Some(close) = self.close_text() {
            out.push(close);
        }
        for block in &self.tool_blocks {
            out.push((
                "content_block_stop".to_string(),
                json!({"type": "content_block_stop", "index": block.index}),
            ));
        }
        let stop_reason = finish_reason
            .map(map_stop_reason)
            .unwrap_or_else(|| "end_turn".to_string());
        out.push((
            "message_delta".to_string(),
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": Value::Null},
                "usage": {"output_tokens": output_tokens},
            }),
        ));
        out.push((
            "message_stop".to_string(),
            json!({"type": "message_stop"}),
        ));
        out
    }

    pub(crate) fn ingest_chunk(&mut self, chunk: &ChatCompletionChunkResponse) -> Vec<NamedEvent> {
        let mut events = Vec::new();
        if !self.started {
            let id = format!("msg_{}", if chunk.id.is_empty() { Uuid::new_v4().to_string() } else { chunk.id.clone() });
            let input_tokens = chunk
                .usage
                .as_ref()
                .map(|u| u.prompt_tokens as u32)
                .unwrap_or(0);
            events.extend(self.start(chunk.model.clone(), id, input_tokens));
        }
        let Some(choice) = chunk.choices.first() else {
            return events;
        };
        let has_tool_calls = choice
            .delta
            .tool_calls
            .as_ref()
            .is_some_and(|v| !v.is_empty());
        if let Some(text) = &choice.delta.content {
            if !text.is_empty() && !has_tool_calls {
                if let Some(start) = self.open_text() {
                    events.push(start);
                }
                if let Some(delta) = self.text_delta(text) {
                    events.push(delta);
                }
            }
        }
        if let Some(calls) = &choice.delta.tool_calls {
            if !calls.is_empty() {
                events.extend(self.handle_tool_calls(calls));
            }
        }
        if choice.finish_reason.is_some() {
            let output = chunk
                .usage
                .as_ref()
                .map(|u| u.completion_tokens as u32)
                .unwrap_or(0);
            events.extend(self.finalize(choice.finish_reason.as_deref(), output));
        }
        events
    }

    pub(crate) fn ingest_done(
        &mut self,
        resp: &hanzo_engine::ChatCompletionResponse,
    ) -> Vec<NamedEvent> {
        let mut events = Vec::new();
        if !self.started {
            let id = format!("msg_{}", if resp.id.is_empty() { Uuid::new_v4().to_string() } else { resp.id.clone() });
            events.extend(self.start(
                resp.model.clone(),
                id,
                resp.usage.prompt_tokens as u32,
            ));
        }
        let finish = resp.choices.first().map(|c| c.finish_reason.as_str());
        if let Some(c) = resp.choices.first() {
            if let Some(text) = &c.message.content {
                if !text.is_empty() {
                    if let Some(start) = self.open_text() {
                        events.push(start);
                    }
                    if let Some(delta) = self.text_delta(text) {
                        events.push(delta);
                    }
                }
            }
            if let Some(calls) = &c.message.tool_calls {
                if !calls.is_empty() {
                    events.extend(self.handle_tool_calls(calls));
                }
            }
        }
        events.extend(self.finalize(finish, resp.usage.completion_tokens as u32));
        events
    }
}

fn to_event((name, payload): NamedEvent) -> Event {
    Event::default()
        .event(&name)
        .data(serde_json::to_string(&payload).unwrap_or_default())
}

pub struct MessagesStreamer {
    rx: Receiver<Response>,
    state: SharedHanzoState,
    builder: StreamBuilder,
    buffered: std::collections::VecDeque<NamedEvent>,
    done_state: DoneState,
}

impl MessagesStreamer {
    fn new(rx: Receiver<Response>, state: SharedHanzoState) -> Self {
        Self {
            rx,
            state,
            builder: StreamBuilder::new(),
            buffered: std::collections::VecDeque::new(),
            done_state: DoneState::Running,
        }
    }
}

impl futures::Stream for MessagesStreamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if let Some(ev) = self.buffered.pop_front() {
            return Poll::Ready(Some(Ok(to_event(ev))));
        }
        match self.done_state {
            DoneState::SendingDone | DoneState::Done => return Poll::Ready(None),
            DoneState::Running => (),
        }
        loop {
            match self.rx.poll_recv(cx) {
                Poll::Ready(Some(Response::Chunk(chunk))) => {
                    Hanzo::maybe_log_response(self.state.clone(), &chunk);
                    let events = self.builder.ingest_chunk(&chunk);
                    let all_finished = chunk.choices.iter().all(|c| c.finish_reason.is_some());
                    for ev in events {
                        self.buffered.push_back(ev);
                    }
                    if all_finished {
                        self.done_state = DoneState::SendingDone;
                    }
                    if let Some(ev) = self.buffered.pop_front() {
                        return Poll::Ready(Some(Ok(to_event(ev))));
                    }
                    if matches!(self.done_state, DoneState::SendingDone) {
                        return Poll::Ready(None);
                    }
                }
                Poll::Ready(Some(Response::Done(resp))) => {
                    Hanzo::maybe_log_response(self.state.clone(), &resp);
                    let events = self.builder.ingest_done(&resp);
                    for ev in events {
                        self.buffered.push_back(ev);
                    }
                    self.done_state = DoneState::SendingDone;
                    if let Some(ev) = self.buffered.pop_front() {
                        return Poll::Ready(Some(Ok(to_event(ev))));
                    }
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Response::ModelError(msg, _))) => {
                    Hanzo::maybe_log_error(
                        self.state.clone(),
                        &ModelErrorMessage(msg.clone()),
                    );
                    let err = json!({
                        "type": "error",
                        "error": {"type": "api_error", "message": msg},
                    });
                    self.done_state = DoneState::Done;
                    return Poll::Ready(Some(Ok(to_event(("error".to_string(), err)))));
                }
                Poll::Ready(Some(Response::ValidationError(e))) => {
                    let err = json!({
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": sanitize_error_message(e.as_ref()),
                        },
                    });
                    self.done_state = DoneState::Done;
                    return Poll::Ready(Some(Ok(to_event(("error".to_string(), err)))));
                }
                Poll::Ready(Some(Response::InternalError(e))) => {
                    Hanzo::maybe_log_error(self.state.clone(), &*e);
                    let err = json!({
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": sanitize_error_message(e.as_ref()),
                        },
                    });
                    self.done_state = DoneState::Done;
                    return Poll::Ready(Some(Ok(to_event(("error".to_string(), err)))));
                }
                Poll::Ready(Some(_)) => continue,
                Poll::Ready(None) => {
                    self.done_state = DoneState::Done;
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

fn create_messages_streamer(
    rx: Receiver<Response>,
    state: SharedHanzoState,
) -> Sse<KeepAliveStream<MessagesStreamer>> {
    let streamer = MessagesStreamer::new(rx, state);
    let keep_alive_interval = get_keep_alive_interval();
    Sse::new(streamer)
        .keep_alive(KeepAlive::new().interval(Duration::from_millis(keep_alive_interval)))
}

/// `POST /v1/messages` - Anthropic-compatible chat for Claude Code.
pub async fn messages(
    State(state): ExtractedHanzoState,
    Extension(agentic_defaults): Extension<AgenticDefaults>,
    Json(areq): Json<AnthropicMessagesRequest>,
) -> axum::response::Response {
    let stream = areq.stream.unwrap_or(false);

    let mut openai_req = match anthropic_to_openai(&areq) {
        Ok(r) => r,
        Err(e) => {
            return JsonError::new(format!("anthropic->openai translation failed: {e}"))
                .to_response(StatusCode::BAD_REQUEST)
        }
    };
    openai_req.stream = Some(stream);

    let model = areq.model.clone();
    let model_id = if openai_req.model == "default" {
        None
    } else {
        Some(openai_req.model.clone())
    };

    let (tx, mut rx) = create_response_channel(None);
    let (request, _is_streaming) = match parse_request(
        openai_req,
        state.clone(),
        tx,
        agentic_defaults.tool_dispatch_url,
        None,
        None,
    )
    .await
    {
        Ok(x) => x,
        Err(e) => {
            return JsonError::new(format!("request parse failed: {e}"))
                .to_response(StatusCode::BAD_REQUEST)
        }
    };

    if let Err(e) = send_request_with_model(&state, request, model_id.as_deref()).await {
        return JsonError::new(format!("send failed: {e}"))
            .to_response(StatusCode::INTERNAL_SERVER_ERROR);
    }

    if stream {
        return create_messages_streamer(rx, state).into_response();
    }

    match process_non_streaming_response(&mut rx, state).await {
        ChatCompletionResponder::Json(resp) => {
            let (text, finish, tool_calls) = resp
                .choices
                .first()
                .map(|c| {
                    (
                        c.message.content.clone().unwrap_or_default(),
                        c.finish_reason.clone(),
                        c.message.tool_calls.clone(),
                    )
                })
                .unwrap_or_default();
            let out = AnthropicMessagesResponse {
                id: format!("msg_{}", resp.id),
                kind: "message",
                role: "assistant",
                model,
                content: build_content_blocks(&text, tool_calls.as_ref()),
                stop_reason: map_stop_reason(&finish),
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens: resp.usage.prompt_tokens as u32,
                    output_tokens: resp.usage.completion_tokens as u32,
                },
            };
            Json(out).into_response()
        }
        other => other.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_engine::{
        CalledFunction, ChatCompletionChunkResponse, ChunkChoice, Delta, ResponseMessage,
        ToolCallType, Usage,
    };

    #[test]
    fn translate_tool_result_user_message_to_openai_tool_role() {
        let req = AnthropicMessagesRequest {
            model: "default".to_string(),
            messages: vec![
                AnthropicMessage {
                    role: "user".to_string(),
                    content: json!("hi"),
                },
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: json!([
                        {"type": "text", "text": "calling tool"},
                        {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"q": "rust"}}
                    ]),
                },
                AnthropicMessage {
                    role: "user".to_string(),
                    content: json!([
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "found rust"}
                    ]),
                },
            ],
            max_tokens: 64,
            system: Some(json!("be brief")),
            stream: Some(false),
            temperature: None,
            top_p: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
        };
        let mut out: Vec<Value> = Vec::new();
        if let Some(sys) = &req.system {
            let s = content_text_only(sys);
            out.push(json!({"role": "system", "content": s}));
        }
        for m in &req.messages {
            translate_message(&m.role, &m.content, &mut out);
        }
        let roles: Vec<&str> = out.iter().filter_map(|v| v["role"].as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "tool"]);
        let assistant = &out[2];
        let tc = &assistant["tool_calls"][0];
        assert_eq!(tc["id"], json!("call-1"));
        assert_eq!(tc["function"]["name"], json!("lookup"));
        assert_eq!(tc["function"]["arguments"], json!("{\"q\":\"rust\"}"));
        let tool_msg = &out[3];
        assert_eq!(tool_msg["role"], json!("tool"));
        assert_eq!(tool_msg["tool_call_id"], json!("call-1"));
        assert_eq!(tool_msg["content"], json!("found rust"));
    }

    #[test]
    fn translate_anthropic_tools_to_openai_function_schema() {
        let tools = vec![json!({
            "name": "get_weather",
            "description": "weather lookup",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        })];
        let openai = anthropic_tools_to_openai(&tools);
        assert_eq!(openai[0]["type"], json!("function"));
        assert_eq!(openai[0]["function"]["name"], json!("get_weather"));
        assert_eq!(
            openai[0]["function"]["parameters"],
            json!({"type": "object", "properties": {"city": {"type": "string"}}})
        );
    }

    #[test]
    fn build_content_blocks_emits_text_and_tool_use() {
        let calls = vec![ToolCallResponse {
            index: 0,
            id: "call-42".to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: "lookup".to_string(),
                arguments: "{\"q\":\"rust\"}".to_string(),
            },
        }];
        let blocks = build_content_blocks("preface", Some(&calls));
        assert_eq!(blocks.len(), 2);
        assert!(matches!(&blocks[0], AnthropicContentBlock::Text { text } if text == "preface"));
        assert!(
            matches!(&blocks[1], AnthropicContentBlock::ToolUse { id, name, input }
                if id == "call-42" && name == "lookup" && input == &json!({"q": "rust"}))
        );
    }

    fn fake_chunk(text: &str, finish: Option<&str>) -> ChatCompletionChunkResponse {
        ChatCompletionChunkResponse {
            id: "abc".to_string(),
            choices: vec![ChunkChoice {
                finish_reason: finish.map(str::to_string),
                index: 0,
                delta: Delta {
                    content: Some(text.to_string()),
                    role: "assistant".to_string(),
                    tool_calls: None,
                    reasoning_content: None,
                },
                logprobs: None,
            }],
            created: 0,
            model: "test-model".to_string(),
            system_fingerprint: "local".to_string(),
            object: "chat.completion.chunk".to_string(),
            usage: finish.is_some().then(|| Usage {
                completion_tokens: 5,
                prompt_tokens: 3,
                total_tokens: 8,
                avg_tok_per_sec: 0.0,
                avg_prompt_tok_per_sec: 0.0,
                avg_compl_tok_per_sec: 0.0,
                total_time_sec: 0.0,
                total_prompt_time_sec: 0.0,
                total_completion_time_sec: 0.0,
            }),
            session_id: None,
        }
    }

    #[test]
    fn streaming_emits_full_anthropic_event_sequence_for_text_only() {
        let mut b = StreamBuilder::new();
        let mut all: Vec<NamedEvent> = Vec::new();
        all.extend(b.ingest_chunk(&fake_chunk("Hel", None)));
        all.extend(b.ingest_chunk(&fake_chunk("lo", None)));
        all.extend(b.ingest_chunk(&fake_chunk("!", Some("stop"))));
        let names: Vec<&str> = all.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let start_payload = &all[0].1;
        assert_eq!(start_payload["type"], json!("message_start"));
        assert_eq!(start_payload["message"]["role"], json!("assistant"));
        assert_eq!(start_payload["message"]["model"], json!("test-model"));
        let first_delta = &all[3].1;
        assert_eq!(first_delta["delta"]["type"], json!("text_delta"));
        assert_eq!(first_delta["delta"]["text"], json!("Hel"));
        assert_eq!(first_delta["index"], json!(0));
        let msg_delta = &all[7].1;
        assert_eq!(msg_delta["delta"]["stop_reason"], json!("end_turn"));
        assert_eq!(msg_delta["usage"]["output_tokens"], json!(5));
    }

    #[test]
    fn streaming_emits_tool_use_block_when_choices_have_tool_calls() {
        let mut b = StreamBuilder::new();
        let chunk = ChatCompletionChunkResponse {
            id: "abc".to_string(),
            choices: vec![ChunkChoice {
                finish_reason: Some("tool_calls".to_string()),
                index: 0,
                delta: Delta {
                    content: None,
                    role: "assistant".to_string(),
                    tool_calls: Some(vec![ToolCallResponse {
                        index: 0,
                        id: "call-99".to_string(),
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: "lookup".to_string(),
                            arguments: "{\"q\":\"x\"}".to_string(),
                        },
                    }]),
                    reasoning_content: None,
                },
                logprobs: None,
            }],
            created: 0,
            model: "m".to_string(),
            system_fingerprint: "local".to_string(),
            object: "chat.completion.chunk".to_string(),
            usage: Some(Usage {
                completion_tokens: 4,
                prompt_tokens: 7,
                total_tokens: 11,
                avg_tok_per_sec: 0.0,
                avg_prompt_tok_per_sec: 0.0,
                avg_compl_tok_per_sec: 0.0,
                total_time_sec: 0.0,
                total_prompt_time_sec: 0.0,
                total_completion_time_sec: 0.0,
            }),
            session_id: None,
        };
        let events = b.ingest_chunk(&chunk);
        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let start = &events[2].1;
        assert_eq!(start["content_block"]["type"], json!("tool_use"));
        assert_eq!(start["content_block"]["id"], json!("call-99"));
        assert_eq!(start["content_block"]["name"], json!("lookup"));
        let delta = &events[3].1;
        assert_eq!(delta["delta"]["type"], json!("input_json_delta"));
        assert_eq!(delta["delta"]["partial_json"], json!("{\"q\":\"x\"}"));
        let msg_delta = &events[5].1;
        assert_eq!(msg_delta["delta"]["stop_reason"], json!("tool_use"));
    }

    #[test]
    fn non_streaming_response_emits_correct_blocks_for_tool_calls() {
        let calls = vec![ToolCallResponse {
            index: 0,
            id: "call-1".to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: "x".to_string(),
                arguments: "{\"a\":1}".to_string(),
            },
        }];
        let _msg = ResponseMessage {
            content: Some("text".to_string()),
            role: "assistant".to_string(),
            tool_calls: Some(calls.clone()),
            reasoning_content: None,
        };
        let blocks = build_content_blocks("", Some(&calls));
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], AnthropicContentBlock::ToolUse { name, .. } if name == "x"));
    }

    #[test]
    fn event_framing_serializes_event_name_and_json_data() {
        let pair: NamedEvent = ("message_start".to_string(), json!({"type": "message_start"}));
        let ev = to_event(pair.clone());
        let formatted = format!("{ev:?}");
        assert!(
            formatted.contains("message_start"),
            "expected event name in {formatted}"
        );
        let data_json = serde_json::to_string(&pair.1).unwrap();
        assert!(data_json.contains("\"type\":\"message_start\""));
    }
}
