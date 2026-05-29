//! Native Anthropic Messages API (`POST /v1/messages`) for Claude Code harness support.
//!
//! Translates an Anthropic Messages request into the internal chat pipeline
//! (the same path as `/v1/chat/completions`) and translates the result back to
//! the Anthropic response shape. Non-streaming is implemented here; streaming
//! (SSE event sequence) + tool_use mapping are the next increment for full
//! agentic Claude Code compatibility.

use axum::{extract::State, http::StatusCode, response::IntoResponse, Extension, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    chat_completion::{parse_request, process_non_streaming_response, ChatCompletionResponder},
    handler_core::{create_response_channel, send_request_with_model, ErrorToResponse, JsonError},
    mistralrs_server_router_builder::AgenticDefaults,
    openai::ChatCompletionRequest,
    types::ExtractedMistralRsState,
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
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    /// String or array of content blocks (text/image/tool_use/tool_result).
    pub content: Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
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

/// Anthropic content can be a bare string or an array of typed blocks; collect text.
fn content_to_text(content: &Value) -> String {
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

fn anthropic_to_openai(areq: &AnthropicMessagesRequest) -> Result<ChatCompletionRequest, serde_json::Error> {
    let mut messages: Vec<Value> = Vec::with_capacity(areq.messages.len() + 1);
    if let Some(sys) = &areq.system {
        let sys = content_to_text(sys);
        if !sys.is_empty() {
            messages.push(json!({"role": "system", "content": sys}));
        }
    }
    for m in &areq.messages {
        messages.push(json!({"role": m.role, "content": content_to_text(&m.content)}));
    }
    let mut obj = json!({
        "model": areq.model,
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
    // Anthropic defaults to no extended thinking unless explicitly requested.
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

/// `POST /v1/messages` - Anthropic-compatible chat for Claude Code.
pub async fn messages(
    State(state): ExtractedMistralRsState,
    Extension(agentic_defaults): Extension<AgenticDefaults>,
    Json(areq): Json<AnthropicMessagesRequest>,
) -> axum::response::Response {
    if areq.stream.unwrap_or(false) {
        return JsonError::new(
            "streaming /v1/messages is not yet implemented; set \"stream\": false".to_string(),
        )
        .to_response(StatusCode::BAD_REQUEST);
    }

    let openai_req = match anthropic_to_openai(&areq) {
        Ok(r) => r,
        Err(e) => {
            return JsonError::new(format!("anthropic->openai translation failed: {e}"))
                .to_response(StatusCode::BAD_REQUEST)
        }
    };

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

    match process_non_streaming_response(&mut rx, state).await {
        ChatCompletionResponder::Json(resp) => {
            let (text, finish) = resp
                .choices
                .first()
                .map(|c| {
                    (
                        c.message.content.clone().unwrap_or_default(),
                        c.finish_reason.clone(),
                    )
                })
                .unwrap_or_default();
            let out = AnthropicMessagesResponse {
                id: resp.id,
                kind: "message",
                role: "assistant",
                model,
                content: vec![AnthropicContentBlock::Text { text }],
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
