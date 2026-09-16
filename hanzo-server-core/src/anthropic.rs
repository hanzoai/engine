//! Native Anthropic Messages API (`POST /v1/messages`) for Claude Code harness support.
//!
//! Requests are translated into the internal chat pipeline (the same path as
//! `/v1/chat/completions`) and the result is translated back to the Anthropic
//! response shape. `/v1/messages/count_tokens` builds the SAME
//! [`ChatCompletionRequest`] and renders it through the SAME `parse_request`, so
//! the count it reports cannot drift from what generation actually sees.

use std::{
    collections::{HashMap, HashSet},
    pin::Pin,
    sync::{Mutex, OnceLock},
    task::Poll,
    time::Duration,
};

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, KeepAliveStream},
        IntoResponse, Sse,
    },
    Extension, Json,
};
use either::Either;
use hanzo_engine::{
    AgentPermission, ApproximateUserLocation, ChatCompletionChunkResponse, CodeExecutionPermission,
    Function, Hanzo, Request, RequestMessage, Response, TokenizationRequest, Tool,
    ToolCallResponse, ToolChoice, ToolType, Usage, WebSearchOptions, WebSearchUserLocation,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc::{channel, Receiver};
use uuid::Uuid;

use crate::{
    chat_completion::{parse_request, process_non_streaming_response, ChatCompletionResponder},
    handler_core::{create_response_channel, send_request_with_model, ModelErrorMessage},
    openai::{
        ChatCompletionRequest, FunctionCalled, Grammar, JsonSchemaResponseFormat, Message,
        MessageContent, MessageInnerContent, ResponseFormat, StopTokens, ToolCall,
    },
    router::AgenticDefaults,
    streaming::{get_keep_alive_interval, DoneState},
    types::{ExtractedState, SharedState},
    util::sanitize_error_message,
};

/// Anthropic server-side tool families. `web_search`/`code_execution` run here; the rest are
/// executed by the client and only need a schema the model can fill in.
const WEB_SEARCH_PREFIX: &str = "web_search_";
/// Anthropic's dynamic web search filters results with code, so it also needs the code tool.
const DYNAMIC_WEB_SEARCH_TYPE: &str = "web_search_20260209";
const CODE_EXECUTION_PREFIX: &str = "code_execution_";
const BASH_PREFIX: &str = "bash_";
const TEXT_EDITOR_PREFIX: &str = "text_editor_";
const COMPUTER_PREFIX: &str = "computer_";
const WEB_SEARCH_NAME: &str = "web_search";
const CODE_EXECUTION_NAME: &str = "code_execution";
const OUTPUT_SCHEMA_NAME: &str = "anthropic_output";

/// (event_name, json_payload) intermediary so we can unit-test the SSE sequence without parsing
/// `axum::response::sse::Event` (which has no public accessors).
pub(crate) type NamedEvent = (String, Value);

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessagesRequest {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<AnthropicContent>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(default)]
    pub thinking: Option<AnthropicThinking>,
    #[serde(default)]
    pub output_config: Option<AnthropicOutputConfig>,
    /// hanzo extension, mirrors the chat completions field of the same name.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// hanzo extension, mirrors the chat completions field of the same name.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub max_tool_rounds: Option<usize>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<u32, f32>>,
    #[serde(default)]
    pub logprobs: bool,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub grammar: Option<Grammar>,
    #[serde(default)]
    pub dry_multiplier: Option<f32>,
    #[serde(default)]
    pub dry_base: Option<f32>,
    #[serde(default)]
    pub dry_allowed_length: Option<usize>,
    #[serde(default)]
    pub dry_sequence_breakers: Option<Vec<String>>,
    #[serde(default)]
    pub web_search_options: Option<WebSearchOptions>,
    #[serde(default)]
    pub enable_code_execution: bool,
    #[serde(default)]
    pub agent_permission: Option<AgentPermission>,
    #[serde(default)]
    pub code_execution_permission: Option<CodeExecutionPermission>,
    #[serde(default)]
    pub files: Option<Vec<hanzo_engine::RequestedFile>>,
    #[serde(default)]
    pub truncate_sequence: Option<bool>,
}

fn default_model() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

/// A bare string, an array of blocks, or something we do not recognize yet. The last arm keeps a
/// client that ships a new content shape from failing the whole request.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
    Other(Value),
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextBlock {
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThinkingBlock {
    pub thinking: String,
    /// Claude Code round-trips this opaque token; we accept it and do not mint one.
    #[serde(default)]
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MediaSource {
    #[serde(rename = "type")]
    pub tp: String,
    #[serde(default)]
    pub media_type: Option<String>,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    /// A `document` whose source is itself a block list.
    #[serde(default)]
    pub content: Option<Value>,
}

/// MCP tool results carry the payload flat as `data` + `mimeType` rather than under `source`.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageBlock {
    #[serde(default)]
    pub source: Option<MediaSource>,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default, alias = "mimeType")]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub media_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DocumentBlock {
    #[serde(default)]
    pub source: Option<MediaSource>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default, alias = "mimeType")]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub input: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    #[serde(default)]
    pub content: Option<Value>,
}

/// A ToolSearch result. The first-party API expands it into the tool's definition; we can only
/// name the tool, which is what the router's rewrite does too.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolReferenceBlock {
    #[serde(default, alias = "name")]
    pub tool_name: Option<String>,
}

/// Every Anthropic content block we know how to read, plus one arm for what we do not. Matching
/// this exhaustively is what turns "did we handle image?" into a compile error.
#[derive(Debug, Clone, Deserialize)]
#[serde(from = "Value")]
pub enum AnthropicBlock {
    Text(TextBlock),
    Thinking(ThinkingBlock),
    RedactedThinking,
    Image(ImageBlock),
    Document(DocumentBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    ToolReference(ToolReferenceBlock),
    /// Server-side tool traffic we replay as history but never re-execute.
    ServerTool,
    Unknown(Value),
}

fn typed<T: DeserializeOwned>(value: &Value) -> Option<T> {
    T::deserialize(value).ok()
}

impl From<Value> for AnthropicBlock {
    fn from(value: Value) -> Self {
        let Some(tp) = value
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_string)
        else {
            return Self::Unknown(value);
        };
        // A known tag whose payload does not fit still degrades to Unknown: a malformed block
        // costs one turn of fidelity, never the conversation.
        let parsed = match tp.as_str() {
            "text" => typed(&value).map(Self::Text),
            "thinking" => typed(&value).map(Self::Thinking),
            "redacted_thinking" => Some(Self::RedactedThinking),
            "image" => typed(&value).map(Self::Image),
            "document" => typed(&value).map(Self::Document),
            "tool_use" => typed(&value).map(Self::ToolUse),
            "tool_result" => typed(&value).map(Self::ToolResult),
            "tool_reference" => typed(&value).map(Self::ToolReference),
            "server_tool_use" | "web_search_tool_result" | "code_execution_tool_result" => {
                Some(Self::ServerTool)
            }
            _ => None,
        };
        parsed.unwrap_or_else(|| {
            warn_unknown_block(&tp);
            Self::Unknown(value)
        })
    }
}

/// Once per process per block type: a new Claude Code release should not produce a log line per
/// block per turn.
fn warn_unknown_block(tp: &str) {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let Ok(mut seen) = seen.lock() else {
        return;
    };
    if seen.insert(tp.to_string()) {
        tracing::warn!("Anthropic content block `{tp}` is not modeled; degrading it to text");
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTool {
    #[serde(rename = "type", default)]
    pub tp: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<Value>,
    #[serde(default)]
    pub user_location: Option<AnthropicUserLocation>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicUserLocation {
    #[serde(default)]
    pub city: Option<String>,
    #[serde(default)]
    pub country: Option<String>,
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub timezone: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicToolChoice {
    #[serde(rename = "type")]
    pub tp: String,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicThinking {
    #[serde(rename = "type")]
    pub tp: String,
    #[serde(default)]
    pub budget_tokens: Option<usize>,
    #[serde(default)]
    pub display: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicOutputConfig {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub format: Option<AnthropicOutputFormat>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicOutputFormat {
    #[serde(rename = "type")]
    pub tp: String,
    pub schema: Value,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum AnthropicResponseBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

/// `cache_creation_input_tokens` stays zero: the engine reports prefix-cache reads but does not
/// distinguish the turn that first wrote the prefix.
#[derive(Debug, Default, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub cache_creation_input_tokens: u32,
    pub cache_read_input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<AnthropicResponseBlock>,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct CountTokensResponse {
    pub input_tokens: u32,
}

/// Anthropic-shaped error body: `{"type":"error","error":{"type":..,"message":..}}`.
/// This is what the Anthropic SDKs parse, so it is a drop-in error surface.
fn anthropic_error(
    status: StatusCode,
    err_type: &str,
    message: impl Into<String>,
) -> axum::response::Response {
    let body = json!({
        "type": "error",
        "error": {"type": err_type, "message": message.into()},
    });
    (status, Json(body)).into_response()
}

fn bad_request(message: impl Into<String>) -> axum::response::Response {
    anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

/// Claude Code sends Anthropic model ids (`claude-sonnet-4-5-*`, ...); route them to the
/// single loaded model (`None` = default). Non-claude ids route by name. The response
/// still echoes the originally requested id.
fn resolve_model(model: &str) -> &str {
    if model.to_ascii_lowercase().starts_with("claude") {
        "default"
    } else {
        model
    }
}

fn model_id(model: &str) -> Option<String> {
    (model != "default").then(|| model.to_string())
}

impl AnthropicTool {
    fn matches(&self, prefix: &str) -> bool {
        self.tp.as_deref().is_some_and(|tp| tp.starts_with(prefix))
    }

    fn tool_name(&self, fallback: &str) -> String {
        self.name.clone().unwrap_or_else(|| fallback.to_string())
    }

    fn web_search_options(&self) -> WebSearchOptions {
        WebSearchOptions {
            user_location: self.user_location.as_ref().map(|location| {
                WebSearchUserLocation::Approximate {
                    approximate: ApproximateUserLocation {
                        city: location.city.clone().unwrap_or_default(),
                        country: location.country.clone().unwrap_or_default(),
                        region: location.region.clone().unwrap_or_default(),
                        timezone: location.timezone.clone().unwrap_or_default(),
                    },
                }
            }),
            ..Default::default()
        }
    }
}

/// Anthropic ships its client-executed built-ins without a schema, because the first-party model
/// was trained on them. A local model has not been, so it gets the published schema instead.
fn builtin_schema(tool: &AnthropicTool) -> Option<Value> {
    if tool.matches(BASH_PREFIX) {
        return Some(json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."},
                "restart": {"type": "boolean", "description": "Restart the bash session."},
            },
        }));
    }
    if tool.matches(TEXT_EDITOR_PREFIX) {
        return Some(json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                },
                "path": {"type": "string", "description": "Absolute path to the file."},
                "file_text": {"type": "string"},
                "insert_line": {"type": "integer"},
                "new_str": {"type": "string"},
                "old_str": {"type": "string"},
                "view_range": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["command", "path"],
        }));
    }
    if tool.matches(COMPUTER_PREFIX) {
        return Some(json!({
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "coordinate": {"type": "array", "items": {"type": "integer"}},
                "start_coordinate": {"type": "array", "items": {"type": "integer"}},
                "text": {"type": "string"},
                "duration": {"type": "number"},
                "scroll_amount": {"type": "integer"},
                "scroll_direction": {"type": "string"},
            },
            "required": ["action"],
        }));
    }
    None
}

fn function_tool(name: String, description: Option<String>, schema: Value) -> Result<Tool> {
    let Value::Object(schema) = schema else {
        anyhow::bail!("Anthropic tool `{name}` input_schema must be a JSON object.");
    };
    Ok(Tool {
        tp: ToolType::Function,
        function: Function {
            description,
            name,
            parameters: Some(schema.into_iter().collect::<HashMap<_, _>>()),
            strict: None,
        },
    })
}

#[derive(Default)]
struct ConvertedTools {
    tools: Vec<Tool>,
    web_search_options: Option<WebSearchOptions>,
    enable_code_execution: bool,
    /// Tools the server runs itself, so `tool_choice` naming one is satisfied without a schema.
    server_tool_names: Vec<String>,
}

fn convert_tools(tools: Option<Vec<AnthropicTool>>) -> Result<ConvertedTools> {
    let mut out = ConvertedTools::default();
    for mut tool in tools.unwrap_or_default() {
        if tool.matches(WEB_SEARCH_PREFIX) {
            out.web_search_options
                .get_or_insert_with(|| tool.web_search_options());
            out.enable_code_execution |= tool.tp.as_deref() == Some(DYNAMIC_WEB_SEARCH_TYPE);
            out.server_tool_names.push(tool.tool_name(WEB_SEARCH_NAME));
            continue;
        }
        if tool.matches(CODE_EXECUTION_PREFIX) {
            out.enable_code_execution = true;
            out.server_tool_names
                .push(tool.tool_name(CODE_EXECUTION_NAME));
            continue;
        }
        let name = tool
            .name
            .take()
            .context("Anthropic tool requires a `name`.")?;
        let schema = match tool.input_schema.take() {
            Some(schema) => schema,
            None => builtin_schema(&tool)
                .with_context(|| format!("Anthropic tool `{name}` requires `input_schema`."))?,
        };
        out.tools
            .push(function_tool(name, tool.description, schema)?);
    }
    Ok(out)
}

fn convert_tool_choice(
    tool_choice: Option<AnthropicToolChoice>,
    tools: &[Tool],
    server_tool_names: &[String],
) -> Result<Option<ToolChoice>> {
    let Some(tool_choice) = tool_choice else {
        return Ok(None);
    };
    match tool_choice.tp.as_str() {
        "auto" => Ok(Some(ToolChoice::Auto)),
        "none" => Ok(Some(ToolChoice::None)),
        "any" if !tools.is_empty() => Ok(Some(ToolChoice::Required)),
        // Only server-side tools are on offer, so there is no function for the model to be forced into.
        "any" if !server_tool_names.is_empty() => Ok(Some(ToolChoice::Auto)),
        "any" => anyhow::bail!("Anthropic tool_choice type `any` requires at least one tool."),
        "tool" => {
            let name = tool_choice
                .name
                .context("Anthropic tool_choice type `tool` requires a `name`.")?;
            if server_tool_names.iter().any(|known| known == &name) {
                return Ok(Some(ToolChoice::Auto));
            }
            let tool = tools
                .iter()
                .find(|tool| tool.function.name == name)
                .cloned()
                .with_context(|| {
                    format!("Anthropic tool_choice references unknown tool `{name}`.")
                })?;
            Ok(Some(ToolChoice::Tool(tool)))
        }
        other => anyhow::bail!("Unsupported Anthropic tool_choice type `{other}`."),
    }
}

fn resolve_thinking(
    thinking: Option<&AnthropicThinking>,
    extension: Option<bool>,
) -> Result<Option<bool>> {
    let native = thinking
        .map(|thinking| match thinking.tp.as_str() {
            "enabled" | "adaptive" => Ok(true),
            "disabled" => Ok(false),
            other => anyhow::bail!("Unsupported Anthropic thinking type `{other}`."),
        })
        .transpose()?;
    if let (Some(native), Some(extension)) = (native, extension) {
        if native != extension {
            anyhow::bail!("Anthropic `thinking.type` conflicts with `enable_thinking`.");
        }
    }
    // Claude Code asks for thinking explicitly; a silent default-on would change every harness turn.
    Ok(Some(extension.or(native).unwrap_or(false)))
}

/// `thinking.display: omitted` means the client does not want the reasoning echoed back.
fn omit_thinking(thinking: Option<&AnthropicThinking>) -> Result<bool> {
    let Some(thinking) = thinking else {
        return Ok(false);
    };
    let Some(display) = thinking.display.as_deref() else {
        return Ok(false);
    };
    if thinking.tp == "disabled" {
        anyhow::bail!("Anthropic `thinking.display` cannot be used with `thinking.type=disabled`.");
    }
    match display {
        "summarized" => Ok(false),
        "omitted" => Ok(true),
        other => anyhow::bail!("Unsupported Anthropic thinking display `{other}`."),
    }
}

fn resolve_response_format(
    output_config: Option<&AnthropicOutputConfig>,
    extension: Option<ResponseFormat>,
) -> Result<Option<ResponseFormat>> {
    let Some(format) = output_config.and_then(|config| config.format.as_ref()) else {
        return Ok(extension);
    };
    if extension.is_some() {
        anyhow::bail!("Anthropic `output_config.format` conflicts with `response_format`.");
    }
    if format.tp != "json_schema" {
        anyhow::bail!("Unsupported Anthropic output format `{}`.", format.tp);
    }
    Ok(Some(ResponseFormat::JsonSchema {
        json_schema: JsonSchemaResponseFormat {
            name: OUTPUT_SCHEMA_NAME.to_string(),
            schema: format.schema.clone(),
        },
    }))
}

fn message_with_text(role: impl Into<String>, text: String) -> Message {
    Message {
        content: Some(MessageContent::from_text(text)),
        role: role.into(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

/// A `document` is inlined when it carries text, and named when it carries bytes. Dumping a
/// base64 PDF into the prompt is worse than saying a PDF was attached.
fn document_to_text(block: &DocumentBlock) -> String {
    let title = block.title.clone();
    let source = block.source.as_ref();
    let media_type = source
        .and_then(|source| source.media_type.clone())
        .or_else(|| block.mime_type.clone());
    if let Some(source) = source {
        match source.tp.as_str() {
            "text" => {
                if let Some(data) = &source.data {
                    return data.clone();
                }
            }
            "content" => {
                if let Some(content) = &source.content {
                    return content_value_to_text(Some(content.clone()));
                }
            }
            _ => {}
        }
    }
    let label = media_type.unwrap_or_else(|| "document".to_string());
    match title {
        Some(title) => format!("[document: {label}, {title}]"),
        None => format!("[document: {label}]"),
    }
}

fn tool_reference_to_text(block: &ToolReferenceBlock) -> String {
    let name = block.tool_name.clone().unwrap_or_default();
    format!("Tool loaded: {name}")
}

/// A block we could not model still contributes whatever text it carries rather than vanishing.
fn unknown_to_text(value: &Value) -> Option<String> {
    value
        .get("text")
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|text| !text.is_empty())
}

impl ImageBlock {
    fn to_url(&self) -> Option<String> {
        if let Some(source) = &self.source {
            match source.tp.as_str() {
                "base64" => {
                    let media_type = source.media_type.as_deref()?;
                    let data = source.data.as_deref()?;
                    return Some(format!("data:{media_type};base64,{data}"));
                }
                "url" => return source.url.clone(),
                _ => return None,
            }
        }
        // MCP shape: the payload sits on the block itself.
        let data = self.data.as_deref()?;
        let media_type = self
            .mime_type
            .as_deref()
            .or(self.media_type.as_deref())
            .unwrap_or("image/png");
        Some(format!("data:{media_type};base64,{data}"))
    }
}

enum UserPart {
    Text(String),
    Image(HashMap<String, MessageInnerContent>),
}

fn flush_user_parts(out: &mut Vec<Message>, role: &str, parts: &mut Vec<UserPart>) {
    if parts.is_empty() {
        return;
    }
    let has_image = parts.iter().any(|part| matches!(part, UserPart::Image(_)));
    let parts = std::mem::take(parts);
    let content = if has_image {
        MessageContent::from_parts(
            parts
                .into_iter()
                .map(|part| match part {
                    UserPart::Text(text) => MessageContent::text_part(text),
                    UserPart::Image(image) => image,
                })
                .collect(),
        )
    } else {
        MessageContent::from_text(
            parts
                .into_iter()
                .map(|part| match part {
                    UserPart::Text(text) => text,
                    UserPart::Image(_) => String::new(),
                })
                .collect::<Vec<_>>()
                .join("\n"),
        )
    };
    out.push(Message {
        content: Some(content),
        role: role.to_string(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

fn tool_call_from_block(block: &ToolUseBlock) -> ToolCall {
    let input = block.input.clone().unwrap_or_else(|| json!({}));
    // A stringified input is already serialized arguments; anything else is re-serialized as is.
    let arguments = match input {
        Value::String(arguments) => arguments,
        other => serde_json::to_string(&other).unwrap_or_else(|_| "{}".to_string()),
    };
    ToolCall {
        id: Some(block.id.clone()),
        tp: ToolType::Function,
        function: FunctionCalled {
            name: block.name.clone(),
            arguments,
        },
    }
}

fn content_value_to_text(content: Option<Value>) -> String {
    match content {
        Some(Value::String(text)) => text,
        Some(Value::Array(items)) => items
            .into_iter()
            .map(|item| content_item_to_text(&AnthropicBlock::from(item)))
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Tool results reach the model as a `tool` role, which is text only, so every block collapses
/// to text here. Binary payloads are named, never inlined.
fn content_item_to_text(block: &AnthropicBlock) -> String {
    match block {
        AnthropicBlock::Text(text) => text.text.clone(),
        AnthropicBlock::Thinking(thinking) => thinking.thinking.clone(),
        AnthropicBlock::Image(_) => "[image]".to_string(),
        AnthropicBlock::Document(document) => document_to_text(document),
        AnthropicBlock::ToolReference(reference) => tool_reference_to_text(reference),
        AnthropicBlock::ToolResult(result) => content_value_to_text(result.content.clone()),
        AnthropicBlock::ToolUse(tool_use) => {
            let input = tool_use.input.clone().unwrap_or_else(|| json!({}));
            format!("[tool_use: {} {}]", tool_use.name, input)
        }
        AnthropicBlock::RedactedThinking | AnthropicBlock::ServerTool => String::new(),
        AnthropicBlock::Unknown(value) => unknown_to_text(value).unwrap_or_default(),
    }
}

fn system_to_text(content: &AnthropicContent) -> String {
    match content {
        AnthropicContent::Text(text) => text.clone(),
        AnthropicContent::Blocks(blocks) => blocks
            .iter()
            .map(content_item_to_text)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        AnthropicContent::Other(value) => unknown_to_text(value).unwrap_or_default(),
    }
}

fn append_assistant_blocks(out: &mut Vec<Message>, role: String, blocks: &[AnthropicBlock]) {
    let mut text_parts = Vec::new();
    let mut thinking_parts = Vec::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block {
            AnthropicBlock::Text(text) => text_parts.push(text.text.clone()),
            AnthropicBlock::Thinking(thinking) => thinking_parts.push(thinking.thinking.clone()),
            AnthropicBlock::ToolUse(tool_use) => tool_calls.push(tool_call_from_block(tool_use)),
            AnthropicBlock::Document(document) => text_parts.push(document_to_text(document)),
            AnthropicBlock::ToolReference(reference) => {
                text_parts.push(tool_reference_to_text(reference))
            }
            AnthropicBlock::Unknown(value) => text_parts.extend(unknown_to_text(value)),
            AnthropicBlock::Image(_)
            | AnthropicBlock::ToolResult(_)
            | AnthropicBlock::RedactedThinking
            | AnthropicBlock::ServerTool => {}
        }
    }
    let content =
        (!text_parts.is_empty()).then(|| MessageContent::from_text(text_parts.join("\n")));
    if content.is_none() && tool_calls.is_empty() && thinking_parts.is_empty() {
        return;
    }
    out.push(Message {
        content,
        role,
        name: None,
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        tool_call_id: None,
        reasoning_content: (!thinking_parts.is_empty()).then(|| thinking_parts.join("\n")),
    });
}

fn append_user_blocks(out: &mut Vec<Message>, role: String, blocks: &[AnthropicBlock]) {
    let mut parts: Vec<UserPart> = Vec::new();
    for block in blocks {
        match block {
            AnthropicBlock::Text(text) => parts.push(UserPart::Text(text.text.clone())),
            AnthropicBlock::Image(image) => match image.to_url() {
                Some(url) => parts.push(UserPart::Image(MessageContent::image_url_part(url))),
                None => parts.push(UserPart::Text("[image]".to_string())),
            },
            AnthropicBlock::Document(document) => {
                parts.push(UserPart::Text(document_to_text(document)))
            }
            AnthropicBlock::ToolReference(reference) => {
                parts.push(UserPart::Text(tool_reference_to_text(reference)))
            }
            AnthropicBlock::ToolResult(result) => {
                flush_user_parts(out, &role, &mut parts);
                out.push(Message {
                    content: Some(MessageContent::from_text(content_value_to_text(
                        result.content.clone(),
                    ))),
                    role: "tool".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: Some(result.tool_use_id.clone()),
                    reasoning_content: None,
                });
            }
            AnthropicBlock::ToolUse(_) => parts.push(UserPart::Text(content_item_to_text(block))),
            AnthropicBlock::Unknown(value) => {
                parts.extend(unknown_to_text(value).map(UserPart::Text))
            }
            AnthropicBlock::Thinking(_)
            | AnthropicBlock::RedactedThinking
            | AnthropicBlock::ServerTool => {}
        }
    }
    flush_user_parts(out, &role, &mut parts);
}

fn append_message(out: &mut Vec<Message>, message: &AnthropicMessage) {
    match &message.content {
        AnthropicContent::Text(text) => {
            out.push(message_with_text(message.role.clone(), text.clone()))
        }
        AnthropicContent::Blocks(blocks) => {
            if message.role == "system" {
                let text = system_to_text(&message.content);
                if !text.is_empty() {
                    out.push(message_with_text(message.role.clone(), text));
                }
            } else if message.role == "assistant" {
                append_assistant_blocks(out, message.role.clone(), blocks);
            } else {
                append_user_blocks(out, message.role.clone(), blocks);
            }
        }
        AnthropicContent::Other(value) => {
            if let Some(text) = unknown_to_text(value) {
                out.push(message_with_text(message.role.clone(), text));
            }
        }
    }
}

impl AnthropicMessagesRequest {
    fn validate(&self, require_max_tokens: bool) -> Result<()> {
        if self.messages.is_empty() {
            anyhow::bail!("messages: at least one message is required");
        }
        for (index, message) in self.messages.iter().enumerate() {
            if !matches!(message.role.as_str(), "user" | "assistant" | "system") {
                anyhow::bail!(
                    "messages.{index}.role: input should be 'user', 'assistant' or 'system', got '{}'",
                    message.role
                );
            }
        }
        if require_max_tokens {
            match self.max_tokens {
                None => anyhow::bail!("max_tokens: field required"),
                Some(0) => anyhow::bail!("max_tokens: must be greater than or equal to 1"),
                Some(_) => {}
            }
        }
        Ok(())
    }

    fn build_messages(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.messages.len() + 1);
        if let Some(system) = &self.system {
            let system = system_to_text(system);
            if !system.is_empty() {
                out.push(message_with_text("system", system));
            }
        }
        for message in &self.messages {
            append_message(&mut out, message);
        }
        out
    }

    /// The single translation both `/v1/messages` and `/v1/messages/count_tokens` go through.
    fn into_chat_completion_request(self) -> Result<ChatCompletionRequest> {
        let messages = self.build_messages();
        let model = resolve_model(&self.model).to_string();
        let enable_thinking = resolve_thinking(self.thinking.as_ref(), self.enable_thinking)?;
        let response_format =
            resolve_response_format(self.output_config.as_ref(), self.response_format)?;
        let reasoning_effort = self
            .reasoning_effort
            .or_else(|| self.output_config.and_then(|config| config.effort));
        let mut converted = convert_tools(self.tools)?;
        if converted.web_search_options.is_none() {
            converted.web_search_options = self.web_search_options;
        }
        converted.enable_code_execution |= self.enable_code_execution;
        let tool_choice = convert_tool_choice(
            self.tool_choice,
            &converted.tools,
            &converted.server_tool_names,
        )?;
        if matches!(tool_choice, Some(ToolChoice::None)) {
            converted = ConvertedTools::default();
        }

        Ok(ChatCompletionRequest {
            messages: Either::Left(messages),
            model,
            logit_bias: self.logit_bias,
            logprobs: self.logprobs,
            top_logprobs: self.top_logprobs,
            max_tokens: self.max_tokens,
            n_choices: 1,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repetition_penalty: self.repetition_penalty,
            stop_seqs: self.stop_sequences.map(StopTokens::Multi),
            temperature: self.temperature,
            top_p: self.top_p,
            stream: self.stream,
            tools: (!converted.tools.is_empty()).then_some(converted.tools),
            tool_choice,
            response_format,
            web_search_options: converted.web_search_options,
            enable_code_execution: converted.enable_code_execution,
            agent_permission: self.agent_permission,
            code_execution_permission: self.code_execution_permission,
            session_id: self.session_id,
            files: self.files,
            top_k: self.top_k,
            grammar: self.grammar,
            min_p: self.min_p,
            dry_multiplier: self.dry_multiplier,
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_sequence_breakers: self.dry_sequence_breakers,
            enable_thinking,
            reasoning_effort,
            max_tool_rounds: self.max_tool_rounds,
            truncate_sequence: self.truncate_sequence,
        })
    }
}

/// Anthropic counts a cached prefix separately, so `input_tokens` has to exclude what
/// `cache_read_input_tokens` already reports or a cached turn double-counts its prompt.
fn anthropic_usage(usage: &Usage) -> AnthropicUsage {
    let cache_read = usage.cached_prompt_tokens as u32;
    AnthropicUsage {
        input_tokens: (usage.prompt_tokens as u32).saturating_sub(cache_read),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: cache_read,
        output_tokens: usage.completion_tokens as u32,
    }
}

fn map_stop_reason(finish: &str) -> String {
    match finish {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        "stop_sequence" => "stop_sequence",
        _ => "end_turn",
    }
    .to_string()
}

fn tool_input(arguments: &str) -> Value {
    match serde_json::from_str(arguments) {
        Ok(Value::Object(input)) => Value::Object(input),
        Ok(other) => json!({"arguments": other}),
        Err(_) => json!({"arguments": arguments}),
    }
}

fn build_content_blocks(
    text: &str,
    thinking: Option<&str>,
    tool_calls: Option<&Vec<ToolCallResponse>>,
) -> Vec<AnthropicResponseBlock> {
    let mut blocks: Vec<AnthropicResponseBlock> = Vec::new();
    if let Some(thinking) = thinking.filter(|thinking| !thinking.is_empty()) {
        blocks.push(AnthropicResponseBlock::Thinking {
            thinking: thinking.to_string(),
            signature: None,
        });
    }
    if !text.is_empty() {
        blocks.push(AnthropicResponseBlock::Text {
            text: text.to_string(),
        });
    }
    if let Some(calls) = tool_calls {
        for call in calls {
            blocks.push(AnthropicResponseBlock::ToolUse {
                id: call.id.clone(),
                name: call.function.name.clone(),
                input: tool_input(&call.function.arguments),
            });
        }
    }
    if blocks.is_empty() {
        blocks.push(AnthropicResponseBlock::Text {
            text: String::new(),
        });
    }
    blocks
}

/// Which stop sequence fired is not carried on the response, so it is recovered from the tail of
/// the text. Streaming emits the stop text before the sequence ends, so it is there; the
/// non-streaming path is trimmed at the match (`pipeline/sampling.rs`) and reports None.
fn matched_stop_sequence(text: &str, stop_sequences: Option<&[String]>) -> Option<String> {
    stop_sequences?
        .iter()
        .find(|stop| !stop.is_empty() && text.ends_with(stop.as_str()))
        .cloned()
}

#[derive(Default)]
pub(crate) struct StreamBuilder {
    started: bool,
    finalized: bool,
    next_index: usize,
    omit_thinking: bool,
    stop_sequences: Option<Vec<String>>,
    text: String,
    thinking_block: Option<BlockState>,
    text_block: Option<BlockState>,
    tool_blocks: Vec<ToolBlockState>,
}

#[derive(Default)]
struct BlockState {
    index: usize,
}

#[derive(Default)]
struct ToolBlockState {
    index: usize,
    id: String,
    args_sent: String,
}

impl StreamBuilder {
    pub(crate) fn new(omit_thinking: bool, stop_sequences: Option<Vec<String>>) -> Self {
        Self {
            omit_thinking,
            stop_sequences,
            ..Self::default()
        }
    }

    pub(crate) fn start(
        &mut self,
        model: String,
        id: String,
        usage: AnthropicUsage,
    ) -> Vec<NamedEvent> {
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
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                    "cache_read_input_tokens": usage.cache_read_input_tokens,
                    "output_tokens": 0,
                },
            }
        });
        vec![
            ("message_start".to_string(), msg),
            ("ping".to_string(), json!({"type": "ping"})),
        ]
    }

    fn open_block(&mut self, content_block: Value) -> (usize, NamedEvent) {
        let index = self.next_index;
        self.next_index += 1;
        (
            index,
            (
                "content_block_start".to_string(),
                json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": content_block,
                }),
            ),
        )
    }

    fn stop_block(index: usize) -> NamedEvent {
        (
            "content_block_stop".to_string(),
            json!({"type": "content_block_stop", "index": index}),
        )
    }

    fn delta(index: usize, delta: Value) -> NamedEvent {
        (
            "content_block_delta".to_string(),
            json!({"type": "content_block_delta", "index": index, "delta": delta}),
        )
    }

    fn thinking_delta(&mut self, thinking: &str) -> Vec<NamedEvent> {
        if self.omit_thinking {
            return Vec::new();
        }
        let mut out = Vec::new();
        let index = match &self.thinking_block {
            Some(block) => block.index,
            None => {
                let (index, start) =
                    self.open_block(json!({"type": "thinking", "thinking": "", "signature": ""}));
                self.thinking_block = Some(BlockState { index });
                out.push(start);
                index
            }
        };
        out.push(Self::delta(
            index,
            json!({"type": "thinking_delta", "thinking": thinking}),
        ));
        out
    }

    fn close_thinking(&mut self) -> Option<NamedEvent> {
        let block = self.thinking_block.take()?;
        Some(Self::stop_block(block.index))
    }

    fn text_delta(&mut self, text: &str) -> Vec<NamedEvent> {
        let mut out = Vec::new();
        if let Some(close) = self.close_thinking() {
            out.push(close);
        }
        let index = match &self.text_block {
            Some(block) => block.index,
            None => {
                let (index, start) = self.open_block(json!({"type": "text", "text": ""}));
                self.text_block = Some(BlockState { index });
                out.push(start);
                index
            }
        };
        self.text.push_str(text);
        out.push(Self::delta(
            index,
            json!({"type": "text_delta", "text": text}),
        ));
        out
    }

    fn close_text(&mut self) -> Option<NamedEvent> {
        let block = self.text_block.take()?;
        Some(Self::stop_block(block.index))
    }

    fn handle_tool_calls(&mut self, calls: &[ToolCallResponse]) -> Vec<NamedEvent> {
        let mut out = Vec::new();
        if let Some(close) = self.close_thinking() {
            out.push(close);
        }
        if let Some(close) = self.close_text() {
            out.push(close);
        }
        for call in calls {
            match self.tool_blocks.iter().position(|b| b.id == call.id) {
                None => {
                    let (index, start) = self.open_block(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.function.name,
                        "input": {},
                    }));
                    out.push(start);
                    let args_sent = if call.function.arguments.is_empty() {
                        String::new()
                    } else {
                        out.push(Self::delta(
                            index,
                            json!({
                                "type": "input_json_delta",
                                "partial_json": call.function.arguments,
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
                    let index = block.index;
                    block.args_sent = call.function.arguments.clone();
                    if !partial.is_empty() {
                        out.push(Self::delta(
                            index,
                            json!({"type": "input_json_delta", "partial_json": partial}),
                        ));
                    }
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
        if let Some(close) = self.close_thinking() {
            out.push(close);
        }
        if let Some(close) = self.close_text() {
            out.push(close);
        }
        for block in &self.tool_blocks {
            out.push(Self::stop_block(block.index));
        }
        let stop_sequence = matched_stop_sequence(&self.text, self.stop_sequences.as_deref());
        let stop_reason = match (&stop_sequence, finish_reason) {
            (Some(_), _) => "stop_sequence".to_string(),
            (None, Some(finish)) => map_stop_reason(finish),
            (None, None) => "end_turn".to_string(),
        };
        out.push((
            "message_delta".to_string(),
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                "usage": {"output_tokens": output_tokens},
            }),
        ));
        out.push(("message_stop".to_string(), json!({"type": "message_stop"})));
        out
    }

    pub(crate) fn ingest_chunk(&mut self, chunk: &ChatCompletionChunkResponse) -> Vec<NamedEvent> {
        let mut events = Vec::new();
        if !self.started {
            let id = format!(
                "msg_{}",
                if chunk.id.is_empty() {
                    Uuid::new_v4().to_string()
                } else {
                    chunk.id.clone()
                }
            );
            let usage = chunk
                .usage
                .as_ref()
                .map(anthropic_usage)
                .unwrap_or_default();
            events.extend(self.start(chunk.model.clone(), id, usage));
        }
        let Some(choice) = chunk.choices.first() else {
            return events;
        };
        let has_tool_calls = choice
            .delta
            .tool_calls
            .as_ref()
            .is_some_and(|v| !v.is_empty());
        if let Some(thinking) = &choice.delta.reasoning_content {
            if !thinking.is_empty() {
                events.extend(self.thinking_delta(thinking));
            }
        }
        if let Some(text) = &choice.delta.content {
            if !text.is_empty() && !has_tool_calls {
                events.extend(self.text_delta(text));
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
            let id = format!(
                "msg_{}",
                if resp.id.is_empty() {
                    Uuid::new_v4().to_string()
                } else {
                    resp.id.clone()
                }
            );
            events.extend(self.start(resp.model.clone(), id, anthropic_usage(&resp.usage)));
        }
        let finish = resp.choices.first().map(|c| c.finish_reason.as_str());
        if let Some(c) = resp.choices.first() {
            if let Some(thinking) = &c.message.reasoning_content {
                if !thinking.is_empty() {
                    events.extend(self.thinking_delta(thinking));
                }
            }
            if let Some(text) = &c.message.content {
                if !text.is_empty() {
                    events.extend(self.text_delta(text));
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
    state: SharedState,
    builder: StreamBuilder,
    buffered: std::collections::VecDeque<NamedEvent>,
    done_state: DoneState,
}

impl MessagesStreamer {
    fn new(rx: Receiver<Response>, state: SharedState, builder: StreamBuilder) -> Self {
        Self {
            rx,
            state,
            builder,
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
                    Hanzo::maybe_log_error(self.state.clone(), &ModelErrorMessage(msg.clone()));
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
    state: SharedState,
    builder: StreamBuilder,
) -> Sse<KeepAliveStream<MessagesStreamer>> {
    let streamer = MessagesStreamer::new(rx, state, builder);
    let keep_alive_interval = get_keep_alive_interval();
    Sse::new(streamer)
        .keep_alive(KeepAlive::new().interval(Duration::from_millis(keep_alive_interval)))
}

/// `POST /v1/messages` - Anthropic-compatible chat for Claude Code.
///
/// The body is parsed from a raw `Value` so a bad field returns an Anthropic-shaped 400 rather
/// than axum's default 422, which the Anthropic SDKs do not understand.
pub async fn messages(
    State(state): ExtractedState,
    Extension(agentic_defaults): Extension<AgenticDefaults>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let areq: AnthropicMessagesRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return bad_request(e.to_string()),
    };
    if let Err(e) = areq.validate(true) {
        return bad_request(e.to_string());
    }
    let omit_thinking = match omit_thinking(areq.thinking.as_ref()) {
        Ok(omit) => omit,
        Err(e) => return bad_request(e.to_string()),
    };

    let model = areq.model.clone();
    let model_id = model_id(resolve_model(&areq.model));
    let stream = areq.stream.unwrap_or(false);
    let stop_sequences = areq.stop_sequences.clone();

    let mut oairequest = match areq.into_chat_completion_request() {
        Ok(r) => r,
        Err(e) => return bad_request(e.to_string()),
    };
    oairequest.stream = Some(stream);
    if matches!(oairequest.agent_permission, Some(AgentPermission::Ask)) && !stream {
        return bad_request(
            "agent_permission \"ask\" requires stream=true over HTTP; approve or deny emitted requests with POST /v1/agent/approvals/{approval_id}.",
        );
    }
    oairequest.max_tool_rounds = oairequest
        .max_tool_rounds
        .or(agentic_defaults.max_tool_rounds);

    let (tx, mut rx) = create_response_channel(None);
    let (request, _is_streaming) = match parse_request(
        oairequest,
        state.clone(),
        tx,
        agentic_defaults.tool_dispatch_url,
        None,
        None,
    )
    .await
    {
        Ok(x) => x,
        Err(e) => return bad_request(format!("request parse failed: {e}")),
    };

    if let Err(e) = send_request_with_model(&state, request, model_id.as_deref()).await {
        return anthropic_error(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("no model available: {e}"),
        );
    }

    if stream {
        let builder = StreamBuilder::new(omit_thinking, stop_sequences);
        return create_messages_streamer(rx, state, builder).into_response();
    }

    match process_non_streaming_response(&mut rx, state).await {
        ChatCompletionResponder::Json(resp) => {
            let choice = resp.choices.first();
            let text = choice
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default();
            let thinking = choice
                .and_then(|c| c.message.reasoning_content.clone())
                .filter(|_| !omit_thinking);
            let tool_calls = choice.and_then(|c| c.message.tool_calls.clone());
            let finish = choice.map(|c| c.finish_reason.clone()).unwrap_or_default();
            let stop_sequence = matched_stop_sequence(&text, stop_sequences.as_deref());
            let stop_reason = match &stop_sequence {
                Some(_) => "stop_sequence".to_string(),
                None => map_stop_reason(&finish),
            };
            let out = AnthropicMessagesResponse {
                id: format!("msg_{}", resp.id),
                kind: "message",
                role: "assistant",
                model,
                content: build_content_blocks(&text, thinking.as_deref(), tool_calls.as_ref()),
                stop_reason,
                stop_sequence,
                usage: anthropic_usage(&resp.usage),
            };
            Json(out).into_response()
        }
        other => other.into_response(),
    }
}

/// `POST /v1/messages/count_tokens` - Anthropic-compatible input token counter.
///
/// Builds the same request `/v1/messages` builds and renders it through the same `parse_request`,
/// then tokenizes the resulting chat messages without scheduling generation. The router admits or
/// refuses on this number, so it has to be the number the model would really see.
pub async fn count_tokens(
    State(state): ExtractedState,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let creq: AnthropicMessagesRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return bad_request(e.to_string()),
    };
    if let Err(e) = creq.validate(false) {
        return bad_request(e.to_string());
    }

    let model_id = model_id(resolve_model(&creq.model));
    let mut oairequest = match creq.into_chat_completion_request() {
        Ok(r) => r,
        Err(e) => return bad_request(e.to_string()),
    };
    oairequest.stream = Some(false);

    let (tx, _rx) = create_response_channel(Some(1));
    let (request, _) = match parse_request(oairequest, state.clone(), tx, None, None, None).await {
        Ok(x) => x,
        Err(e) => return bad_request(format!("request parse failed: {e}")),
    };

    let Request::Normal(request) = request else {
        return anthropic_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "expected a chat request for token counting",
        );
    };
    let (messages, enable_thinking, reasoning_effort) = match request.messages {
        RequestMessage::Chat {
            messages,
            enable_thinking,
            reasoning_effort,
        }
        | RequestMessage::MultimodalChat {
            messages,
            enable_thinking,
            reasoning_effort,
            ..
        } => (messages, enable_thinking, reasoning_effort),
        _ => return bad_request("only chat messages can be counted"),
    };

    // Tokenize carries its own oneshot channel (Vec<u32>); no generation is scheduled.
    let (response, mut rx) = channel(1);
    let tokenize = Request::Tokenize(TokenizationRequest {
        text: Either::Left(messages),
        tools: request.tools,
        add_generation_prompt: true,
        add_special_tokens: true,
        enable_thinking,
        reasoning_effort,
        response,
    });

    if let Err(e) = send_request_with_model(&state, tokenize, model_id.as_deref()).await {
        return anthropic_error(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("no model available to tokenize: {e}"),
        );
    }

    match rx.recv().await {
        Some(Ok(toks)) => Json(CountTokensResponse {
            input_tokens: toks.len() as u32,
        })
        .into_response(),
        Some(Err(e)) => anthropic_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            format!("tokenization failed: {e}"),
        ),
        None => anthropic_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "tokenizer channel closed unexpectedly",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_engine::{
        CalledFunction, ChatCompletionChunkResponse, ChunkChoice, Delta, ToolCallType, Usage,
    };

    fn request(value: Value) -> AnthropicMessagesRequest {
        serde_json::from_value(value).expect("request must parse")
    }

    fn text_of(message: &Message) -> String {
        message
            .content
            .as_ref()
            .and_then(MessageContent::to_text)
            .unwrap_or_default()
    }

    fn parts_of(message: &Message) -> Vec<HashMap<String, MessageInnerContent>> {
        match message.content.as_deref() {
            Some(Either::Right(parts)) => parts.clone(),
            other => panic!("expected multimodal parts, got {other:?}"),
        }
    }

    fn part_kind(part: &HashMap<String, MessageInnerContent>) -> String {
        match part.get("type").map(|c| &**c) {
            Some(Either::Left(kind)) => kind.clone(),
            other => panic!("expected a part type, got {other:?}"),
        }
    }

    #[test]
    fn translate_tool_result_user_message_to_openai_tool_role() {
        let req = request(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "system": "be brief",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "calling tool"},
                    {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"q": "rust"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call-1", "content": "found rust"}
                ]},
            ],
        }));
        let messages = req.build_messages();
        let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "tool"]);
        let calls = messages[2].tool_calls.as_ref().expect("tool_calls");
        assert_eq!(calls[0].id.as_deref(), Some("call-1"));
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(calls[0].function.arguments, "{\"q\":\"rust\"}");
        assert_eq!(messages[3].tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(text_of(&messages[3]), "found rust");
    }

    /// Breakage 1: a ToolSearch `tool_reference` names its tool instead of being dropped,
    /// at the top level and nested inside a tool_result.
    #[test]
    fn tool_reference_blocks_become_text_naming_the_tool() {
        let req = request(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "tool_reference", "tool_name": "Monitor"},
                    {"type": "text", "text": "kept"}
                ]},
                {"type": "tool_reference", "tool_name": "WebFetch"}
            ]}],
        }));
        let messages = req.build_messages();
        let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["tool", "user"]);
        assert_eq!(text_of(&messages[0]), "Tool loaded: Monitor\nkept");
        assert_eq!(text_of(&messages[1]), "Tool loaded: WebFetch");
    }

    /// Breakage 2a: a `document` block with a text source is inlined, and a binary one is named
    /// rather than dumping its base64 into the prompt.
    #[test]
    fn document_blocks_inline_text_and_name_binary() {
        let req = request(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "document", "source": {
                    "type": "text", "media_type": "text/plain", "data": "the readme body"}},
                {"type": "document", "title": "spec.pdf", "source": {
                    "type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjcK"}},
            ]}],
        }));
        let messages = req.build_messages();
        let text = text_of(&messages[0]);
        assert_eq!(
            text,
            "the readme body\n[document: application/pdf, spec.pdf]"
        );
        assert!(
            !text.contains("JVBERi0xLjcK"),
            "base64 must not reach the prompt"
        );
    }

    /// Breakage 2b: an MCP image carries `data` + `mimeType` with no `source`.
    #[test]
    fn mcp_image_shape_without_source_becomes_an_image_part() {
        let req = request(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "data": "aW1n", "mimeType": "image/png"},
                {"type": "text", "text": "after"},
            ]}],
        }));
        let messages = req.build_messages();
        let parts = parts_of(&messages[0]);
        let kinds: Vec<String> = parts.iter().map(part_kind).collect();
        assert_eq!(kinds, vec!["text", "image_url", "text"]);
        match parts[1].get("image_url").map(|c| &**c) {
            Some(Either::Right(url)) => {
                assert_eq!(
                    url.get("url").map(String::as_str),
                    Some("data:image/png;base64,aW1n")
                )
            }
            other => panic!("expected an image_url object, got {other:?}"),
        }
    }

    /// Breakage 2c: a tool_result carrying a document does not spill base64 into the tool text.
    #[test]
    fn tool_result_document_is_named_not_inlined() {
        let req = request(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "document", "source": {
                        "type": "base64", "media_type": "application/pdf", "data": "JVBERi0x"}},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": "aW1n"}},
                ]}
            ]}],
        }));
        let messages = req.build_messages();
        assert_eq!(
            text_of(&messages[0]),
            "[document: application/pdf]\n[image]"
        );
    }

    /// Breakage 3: built-in tools carry no schema, so one is synthesized and `tool_choice`
    /// naming one resolves to a function the model can actually call.
    #[test]
    fn builtin_tools_get_a_schema_and_can_be_forced() {
        let req = request(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "run ls"}],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
                {"type": "computer_20250124", "name": "computer",
                 "display_width_px": 1024, "display_height_px": 768},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
            "tool_choice": {"type": "tool", "name": "bash"},
        }));
        let oai = req.into_chat_completion_request().expect("must translate");
        let tools = oai.tools.expect("client tools");
        let names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["bash", "str_replace_based_edit_tool", "computer"]
        );
        let bash = tools[0].function.parameters.as_ref().expect("bash schema");
        assert!(
            bash.contains_key("properties"),
            "bash needs a callable schema"
        );
        assert!(
            oai.web_search_options.is_some(),
            "web_search stays server-side"
        );
        match oai.tool_choice {
            Some(ToolChoice::Tool(tool)) => assert_eq!(tool.function.name, "bash"),
            other => panic!("expected a forced bash tool, got {other:?}"),
        }
    }

    /// A `tool_choice` naming a server-side tool falls back to auto instead of 400ing, and
    /// `any` with only server-side tools does the same.
    #[test]
    fn server_side_tool_choice_does_not_reject() {
        for choice in [
            json!({"type": "tool", "name": "web_search"}),
            json!({"type": "any"}),
        ] {
            let req = request(json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "search"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "tool_choice": choice,
            }));
            let oai = req.into_chat_completion_request().expect("must translate");
            assert!(matches!(oai.tool_choice, Some(ToolChoice::Auto)));
        }
    }

    #[test]
    fn any_tool_choice_with_client_tools_is_required() {
        let req = request(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"},
        }));
        let oai = req.into_chat_completion_request().expect("must translate");
        assert!(matches!(oai.tool_choice, Some(ToolChoice::Required)));
    }

    /// Breakage 4: count_tokens and generation share one translation, so the tokenized
    /// messages and the render flags cannot disagree.
    #[test]
    fn count_tokens_and_messages_build_the_same_request() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 128,
            "system": [{"type": "text", "text": "sys"}],
            "messages": [
                {"role": "user", "content": "start"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "consider", "signature": "sig"},
                    {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"q": "rust"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call-1", "content": "found"}
                ]},
            ],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        });
        let generate = request(body.clone())
            .into_chat_completion_request()
            .expect("messages");
        let mut count = request(body).into_chat_completion_request().expect("count");
        // count_tokens is the only difference: it never streams.
        count.stream = Some(false);

        assert_eq!(count.enable_thinking, Some(true));
        assert_eq!(generate.enable_thinking, count.enable_thinking);
        assert_eq!(
            serde_json::to_value(&generate.messages).unwrap(),
            serde_json::to_value(&count.messages).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&generate.tools).unwrap(),
            serde_json::to_value(&count.tools).unwrap()
        );

        let messages = request(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 128,
            "system": [{"type": "text", "text": "sys"}],
            "messages": [
                {"role": "user", "content": "start"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "consider", "signature": "sig"},
                    {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"q": "rust"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call-1", "content": "found"}
                ]},
            ],
        }))
        .build_messages();
        assert_eq!(messages[2].reasoning_content.as_deref(), Some("consider"));
    }

    /// count_tokens accepts a body without `max_tokens`; /v1/messages does not.
    #[test]
    fn max_tokens_required_only_for_messages() {
        let req = request(json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert!(req.validate(false).is_ok());
        assert!(req
            .validate(true)
            .unwrap_err()
            .to_string()
            .contains("max_tokens"));
    }

    #[test]
    fn validation_rejects_empty_messages_zero_max_tokens_and_bad_roles() {
        let empty = request(json!({"model": "m", "max_tokens": 8, "messages": []}));
        assert!(empty
            .validate(true)
            .unwrap_err()
            .to_string()
            .contains("at least one message"));

        let zero = request(json!({
            "model": "m", "max_tokens": 0,
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert!(zero
            .validate(true)
            .unwrap_err()
            .to_string()
            .contains("max_tokens"));

        let bad = request(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{"role": "tool", "content": "hi"}],
        }));
        assert!(bad.validate(true).unwrap_err().to_string().contains("role"));
    }

    /// `system` is a legal role inside the messages array, not only a top-level field.
    #[test]
    fn system_role_inside_messages_is_accepted() {
        let req = request(json!({
            "model": "m", "max_tokens": 8,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "be brief"}]},
                {"role": "user", "content": "hi"},
            ],
        }));
        req.validate(true).expect("system role is legal");
        let messages = req.build_messages();
        assert_eq!(messages[0].role, "system");
        assert_eq!(text_of(&messages[0]), "be brief");
    }

    /// A block type nobody has modeled yet degrades to its text rather than 400ing or vanishing.
    #[test]
    fn unknown_block_degrades_to_text() {
        let req = request(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": [
                {"type": "some_future_block", "text": "salvaged"},
                {"type": "another_future_block", "payload": 3},
                {"type": "text", "text": "kept"},
            ]}],
        }));
        let messages = req.build_messages();
        assert_eq!(text_of(&messages[0]), "salvaged\nkept");
    }

    /// `output_config` folds into the chat request's reasoning effort and response format.
    #[test]
    fn output_config_maps_to_effort_and_response_format() {
        let req = request(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {
                "effort": "high",
                "format": {"type": "json_schema", "schema": {"type": "object"}},
            },
        }));
        let oai = req.into_chat_completion_request().expect("must translate");
        assert_eq!(oai.reasoning_effort.as_deref(), Some("high"));
        match oai.response_format {
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                assert_eq!(json_schema.name, OUTPUT_SCHEMA_NAME);
                assert_eq!(json_schema.schema, json!({"type": "object"}));
            }
            other => panic!("expected a json_schema format, got {other:?}"),
        }
    }

    #[test]
    fn thinking_defaults_off_and_honours_the_client() {
        let off = request(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(
            off.into_chat_completion_request().unwrap().enable_thinking,
            Some(false)
        );

        let on = request(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }));
        assert_eq!(
            on.into_chat_completion_request().unwrap().enable_thinking,
            Some(true)
        );

        let omitted = request(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "display": "omitted"},
        }));
        assert!(omit_thinking(omitted.thinking.as_ref()).unwrap());
    }

    #[test]
    fn build_content_blocks_emits_thinking_text_and_tool_use() {
        let calls = vec![ToolCallResponse {
            index: 0,
            id: "call-42".to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: "lookup".to_string(),
                arguments: "{\"q\":\"rust\"}".to_string(),
            },
        }];
        let blocks = build_content_blocks("preface", Some("pondering"), Some(&calls));
        assert_eq!(
            blocks[0],
            AnthropicResponseBlock::Thinking {
                thinking: "pondering".to_string(),
                signature: None
            }
        );
        assert_eq!(
            blocks[1],
            AnthropicResponseBlock::Text {
                text: "preface".to_string()
            }
        );
        assert_eq!(
            blocks[2],
            AnthropicResponseBlock::ToolUse {
                id: "call-42".to_string(),
                name: "lookup".to_string(),
                input: json!({"q": "rust"}),
            }
        );
    }

    #[test]
    fn usage_carries_the_cache_fields_anthropic_sdks_require() {
        let body = serde_json::to_value(AnthropicUsage {
            input_tokens: 7,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            output_tokens: 3,
        })
        .unwrap();
        assert_eq!(
            body,
            json!({
                "input_tokens": 7,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 3,
            })
        );
    }

    #[test]
    fn count_tokens_response_serializes_input_tokens_only() {
        let body = serde_json::to_value(CountTokensResponse { input_tokens: 42 }).unwrap();
        assert_eq!(body, json!({"input_tokens": 42}));
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
            usage: finish.is_some().then_some(Usage {
                cached_prompt_tokens: 0,
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
        let mut b = StreamBuilder::new(false, None);
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
        let start = &all[0].1;
        assert_eq!(start["message"]["role"], json!("assistant"));
        assert_eq!(start["message"]["model"], json!("test-model"));
        assert_eq!(
            start["message"]["usage"]["cache_read_input_tokens"],
            json!(0)
        );
        assert_eq!(all[3].1["delta"]["text"], json!("Hel"));
        assert_eq!(all[3].1["index"], json!(0));
        assert_eq!(all[7].1["delta"]["stop_reason"], json!("end_turn"));
        assert_eq!(all[7].1["usage"]["output_tokens"], json!(5));
    }

    #[test]
    fn streaming_opens_a_thinking_block_before_text() {
        let mut b = StreamBuilder::new(false, None);
        let mut chunk = fake_chunk("answer", None);
        chunk.choices[0].delta.content = None;
        chunk.choices[0].delta.reasoning_content = Some("pondering".to_string());
        let mut all = b.ingest_chunk(&chunk);
        all.extend(b.ingest_chunk(&fake_chunk("answer", Some("stop"))));
        let names: Vec<&str> = all.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(all[2].1["content_block"]["type"], json!("thinking"));
        assert_eq!(all[3].1["delta"]["type"], json!("thinking_delta"));
        assert_eq!(all[5].1["content_block"]["type"], json!("text"));
        assert_eq!(all[5].1["index"], json!(1));
    }

    #[test]
    fn streaming_suppresses_thinking_when_display_is_omitted() {
        let mut b = StreamBuilder::new(true, None);
        let mut chunk = fake_chunk("", None);
        chunk.choices[0].delta.content = None;
        chunk.choices[0].delta.reasoning_content = Some("hidden".to_string());
        let events = b.ingest_chunk(&chunk);
        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["message_start", "ping"]);
    }

    #[test]
    fn streaming_reports_the_stop_sequence_that_fired() {
        let mut b = StreamBuilder::new(false, Some(vec!["END".to_string()]));
        let mut all = b.ingest_chunk(&fake_chunk("done ", None));
        all.extend(b.ingest_chunk(&fake_chunk("END", Some("stop"))));
        let delta = all
            .iter()
            .find(|(name, _)| name == "message_delta")
            .expect("message_delta");
        assert_eq!(delta.1["delta"]["stop_reason"], json!("stop_sequence"));
        assert_eq!(delta.1["delta"]["stop_sequence"], json!("END"));
    }

    #[test]
    fn streaming_emits_tool_use_block_when_choices_have_tool_calls() {
        let mut b = StreamBuilder::new(false, None);
        let mut chunk = fake_chunk("", Some("tool_calls"));
        chunk.choices[0].delta.content = None;
        chunk.choices[0].delta.tool_calls = Some(vec![ToolCallResponse {
            index: 0,
            id: "call-99".to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: "lookup".to_string(),
                arguments: "{\"q\":\"x\"}".to_string(),
            },
        }]);
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
        assert_eq!(events[2].1["content_block"]["type"], json!("tool_use"));
        assert_eq!(events[2].1["content_block"]["id"], json!("call-99"));
        assert_eq!(events[3].1["delta"]["type"], json!("input_json_delta"));
        assert_eq!(events[5].1["delta"]["stop_reason"], json!("tool_use"));
    }

    #[test]
    fn event_framing_serializes_event_name_and_json_data() {
        let pair: NamedEvent = (
            "message_start".to_string(),
            json!({"type": "message_start"}),
        );
        let ev = to_event(pair.clone());
        let formatted = format!("{ev:?}");
        assert!(formatted.contains("message_start"), "{formatted}");
        assert!(serde_json::to_string(&pair.1)
            .unwrap()
            .contains("\"type\":\"message_start\""));
    }

    #[test]
    fn model_resolution_routes_claude_ids_to_the_loaded_model() {
        assert_eq!(model_id(resolve_model("claude-sonnet-4-5-20250929")), None);
        assert_eq!(model_id(resolve_model("default")), None);
        assert_eq!(model_id(resolve_model("qwen3")), Some("qwen3".to_string()));
    }

    #[test]
    fn stop_reason_maps_to_anthropic_enum() {
        assert_eq!(map_stop_reason("length"), "max_tokens");
        assert_eq!(map_stop_reason("tool_calls"), "tool_use");
        assert_eq!(map_stop_reason("stop"), "end_turn");
    }

    /// The sampling and agentic extensions the guide documents reach the chat request.
    #[test]
    fn engine_extensions_pass_through() {
        let req = request(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "min_p": 0.05,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.2,
            "dry_multiplier": 0.8,
            "session_id": "s-1",
            "max_tool_rounds": 4,
            "truncate_sequence": true,
            "stop_sequences": ["END"],
            "metadata": {"user_id": "ignored"},
        }));
        let oai = req.into_chat_completion_request().expect("must translate");
        assert_eq!(oai.min_p, Some(0.05));
        assert_eq!(oai.top_k, Some(40));
        assert_eq!(oai.repetition_penalty, Some(1.1));
        assert_eq!(oai.presence_penalty, Some(0.2));
        assert_eq!(oai.dry_multiplier, Some(0.8));
        assert_eq!(oai.session_id.as_deref(), Some("s-1"));
        assert_eq!(oai.max_tool_rounds, Some(4));
        assert_eq!(oai.truncate_sequence, Some(true));
        assert!(matches!(oai.stop_seqs, Some(StopTokens::Multi(ref s)) if s == &["END"]));
    }

    /// The dynamic web search variant filters with code, so it turns code execution on.
    #[test]
    fn dynamic_web_search_enables_code_execution() {
        let req = request(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": DYNAMIC_WEB_SEARCH_TYPE, "name": "web_search"}],
        }));
        let oai = req.into_chat_completion_request().expect("must translate");
        assert!(oai.web_search_options.is_some());
        assert!(oai.enable_code_execution);
    }

    /// `tool_choice: none` clears the tools so nothing can be called.
    #[test]
    fn tool_choice_none_clears_every_tool() {
        let req = request(json!({
            "model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"name": "lookup", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
            "tool_choice": {"type": "none"},
        }));
        let oai = req.into_chat_completion_request().expect("must translate");
        assert!(oai.tools.is_none());
        assert!(oai.web_search_options.is_none());
        assert!(!oai.enable_code_execution);
    }

    fn usage_with_cache(prompt: usize, cached: usize) -> Usage {
        Usage {
            completion_tokens: 7,
            prompt_tokens: prompt,
            cached_prompt_tokens: cached,
            total_tokens: prompt + 7,
            avg_tok_per_sec: 0.0,
            avg_prompt_tok_per_sec: 0.0,
            avg_compl_tok_per_sec: 0.0,
            total_time_sec: 0.0,
            total_prompt_time_sec: 0.0,
            total_completion_time_sec: 0.0,
        }
    }

    /// The wire mapping only: a cached count reaches `cache_read_input_tokens` and leaves
    /// `input_tokens`. A live cache hit needs a warm prefix cache and a model.
    #[test]
    fn anthropic_usage_maps_cache_reads_out_of_input_tokens() {
        let cached = anthropic_usage(&usage_with_cache(1000, 900));
        assert_eq!(cached.cache_read_input_tokens, 900);
        assert_eq!(cached.input_tokens, 100);
        assert_eq!(cached.output_tokens, 7);

        let fresh = anthropic_usage(&usage_with_cache(1000, 0));
        assert_eq!(fresh.cache_read_input_tokens, 0);
        assert_eq!(fresh.input_tokens, 1000);
    }

    #[test]
    fn message_start_usage_carries_cache_read_tokens() {
        let mut b = StreamBuilder::new(false, None);
        let mut chunk = fake_chunk("hi", None);
        chunk.usage = Some(usage_with_cache(1000, 900));
        let events = b.ingest_chunk(&chunk);
        let usage = &events[0].1["message"]["usage"];
        assert_eq!(usage["cache_read_input_tokens"], json!(900));
        assert_eq!(usage["input_tokens"], json!(100));
    }
}
