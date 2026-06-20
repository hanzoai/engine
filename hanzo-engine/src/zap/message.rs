//! Wire message types for ZAP inference, mirroring `schema/inference.capnp`.
//!
//! These are plain Rust structs (not the generated capnp readers) so the
//! surface is usable and testable without the capnp toolchain. Each type maps
//! to the like-named capnp struct; `into_request` / `from_response` bridge to
//! the engine's real `hanzo_engine::{Request,Response}` so the server side is a
//! thin adapter over the existing `Sender<Request>`.

use crate::{
    request::{NormalRequest, RequestMessage},
    response::{ChatCompletionResponse, Response},
    sampler::SamplingParams as EngineSamplingParams,
    Constraint as EngineConstraint, MessageContent, ToolChoice as EngineToolChoice,
};
use either::Either;
use indexmap::IndexMap;
use serde_json::Value;
use tokio::sync::mpsc::Sender;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// One chat message. `content` is either plain text or a JSON array of OpenAI
/// content parts; mirrors the capnp `Message` union.
#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: MsgContent,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug)]
pub enum MsgContent {
    Text(String),
    /// JSON array of content parts (text/image_url/...).
    Parts(Value),
}

/// Mirrors capnp `SamplingParams`. `None`/0 means "unset, use model default".
#[derive(Clone, Debug, Default)]
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub min_p: Option<f64>,
    pub max_tokens: Option<usize>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub stop_seqs: Vec<String>,
}

impl SamplingParams {
    fn into_engine(self) -> EngineSamplingParams {
        let mut p = EngineSamplingParams::neutral();
        p.temperature = self.temperature;
        p.top_p = self.top_p;
        p.top_k = self.top_k;
        p.min_p = self.min_p;
        p.max_len = self.max_tokens;
        p.frequency_penalty = self.frequency_penalty;
        p.presence_penalty = self.presence_penalty;
        if !self.stop_seqs.is_empty() {
            p.stop_toks = Some(crate::sampler::StopTokens::Seqs(self.stop_seqs));
        }
        p
    }
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool parameters.
    pub parameters: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolChoiceKind {
    Auto,
    None,
    Required,
    Named,
}

#[derive(Clone, Debug)]
pub enum Constraint {
    None,
    Regex(String),
    JsonSchema(Value),
    Lark(String),
}

impl Constraint {
    fn into_engine(self) -> EngineConstraint {
        match self {
            Constraint::None => EngineConstraint::None,
            Constraint::Regex(r) => EngineConstraint::Regex(r),
            Constraint::JsonSchema(v) => EngineConstraint::JsonSchema(v),
            Constraint::Lark(l) => EngineConstraint::Lark(l),
        }
    }
}

/// Mirrors capnp `ChatRequest` == `hanzo_engine::NormalRequest` (the
/// `Request::Normal` arm). Replaces the JSON body the node POSTs to
/// `/ai/chat/completions`.
#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub model_id: Option<String>,
    pub messages: Vec<Message>,
    pub sampling: SamplingParams,
    pub stream: bool,
    pub tools: Vec<Tool>,
    pub tool_choice: ToolChoiceKind,
    pub tool_choice_name: Option<String>,
    pub constraint: Constraint,
    pub enable_thinking: Option<bool>,
    pub return_logprobs: bool,
    pub enable_code_execution: bool,
    pub max_tool_rounds: Option<usize>,
    pub session_id: Option<String>,
    pub request_id: usize,
}

impl ChatRequest {
    /// Build the engine's `Request::Normal` from this wire request, wiring the
    /// supplied mpsc `Sender<Response>` as the reply channel (the role the
    /// capnp `ResponseStream` / RPC return plays on the wire).
    pub fn into_request(self, response: Sender<Response>) -> crate::Request {
        let messages = RequestMessage::Chat {
            messages: self.messages.into_iter().map(Message::into_engine).collect(),
            enable_thinking: self.enable_thinking,
            reasoning_effort: None,
        };
        let tools = if self.tools.is_empty() {
            None
        } else {
            Some(self.tools.into_iter().map(Tool::into_engine).collect())
        };
        // `Required`/`Named` would need the concrete tool struct to force; the
        // wire kind alone can't carry it, so fall back to engine auto-select.
        let tool_choice = match self.tool_choice {
            ToolChoiceKind::None => Some(EngineToolChoice::None),
            ToolChoiceKind::Auto | ToolChoiceKind::Required | ToolChoiceKind::Named => {
                Some(EngineToolChoice::Auto)
            }
        };
        let mut req = NormalRequest::new_simple(
            messages,
            self.sampling.into_engine(),
            response,
            self.request_id,
            tools,
            tool_choice,
        );
        req.is_streaming = self.stream;
        req.constraint = self.constraint.into_engine();
        req.return_logprobs = self.return_logprobs;
        req.enable_code_execution = self.enable_code_execution;
        req.max_tool_rounds = self.max_tool_rounds;
        req.model_id = self.model_id;
        req.session_id = self.session_id;
        crate::Request::Normal(Box::new(req))
    }
}

impl Message {
    fn into_engine(self) -> IndexMap<String, MessageContent> {
        let mut m: IndexMap<String, MessageContent> = IndexMap::new();
        m.insert(
            "role".to_string(),
            Either::Left(self.role.as_str().to_string()),
        );
        match self.content {
            MsgContent::Text(t) => {
                m.insert("content".to_string(), Either::Left(t));
            }
            MsgContent::Parts(v) => {
                let parts: Vec<IndexMap<String, Value>> = match v {
                    Value::Array(arr) => arr
                        .into_iter()
                        .filter_map(|p| serde_json::from_value(p).ok())
                        .collect(),
                    _ => Vec::new(),
                };
                m.insert("content".to_string(), Either::Right(parts));
            }
        }
        if let Some(name) = self.name {
            m.insert("name".to_string(), Either::Left(name));
        }
        if let Some(id) = self.tool_call_id {
            m.insert("tool_call_id".to_string(), Either::Left(id));
        }
        m
    }
}

impl Tool {
    fn into_engine(self) -> crate::Tool {
        hanzo_llm_mcp::Tool {
            tp: hanzo_llm_mcp::ToolType::Function,
            function: hanzo_llm_mcp::Function {
                description: Some(self.description),
                name: self.name,
                parameters: serde_json::from_value(self.parameters).ok(),
                strict: None,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON string of the arguments.
    pub arguments: String,
}

#[derive(Clone, Debug, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub avg_tok_per_sec: f32,
    pub total_time_sec: f32,
}

/// Mirrors capnp `ChatResponse` == `Response::Done(ChatCompletionResponse)`.
#[derive(Clone, Debug)]
pub struct ChatResponse {
    pub id: String,
    pub model_id: String,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
    pub usage: Usage,
    pub session_id: Option<String>,
}

impl ChatResponse {
    pub fn from_completion(r: ChatCompletionResponse) -> Self {
        let choice = r.choices.into_iter().next();
        let (content, reasoning, finish, tool_calls) = match choice {
            Some(c) => (
                c.message.content.unwrap_or_default(),
                c.message.reasoning_content,
                c.finish_reason,
                c.message
                    .tool_calls
                    .unwrap_or_default()
                    .into_iter()
                    .map(|tc| ToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    })
                    .collect(),
            ),
            None => (String::new(), None, String::new(), Vec::new()),
        };
        ChatResponse {
            id: r.id,
            model_id: r.model,
            content,
            reasoning_content: reasoning,
            tool_calls,
            finish_reason: finish,
            usage: Usage {
                prompt_tokens: r.usage.prompt_tokens as u32,
                completion_tokens: r.usage.completion_tokens as u32,
                total_tokens: r.usage.total_tokens as u32,
                avg_tok_per_sec: r.usage.avg_compl_tok_per_sec,
                total_time_sec: r.usage.total_time_sec,
            },
            session_id: r.session_id,
        }
    }
}

/// Mirrors capnp `ChatChunk` == `Response::Chunk`.
#[derive(Clone, Debug, Default)]
pub struct ChatChunk {
    pub id: String,
    pub delta_content: String,
    pub delta_reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
    pub usage: Option<Usage>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferErrorKind {
    Internal,
    Validation,
    Model,
}

/// Mirrors capnp `InferError`.
#[derive(Clone, Debug)]
pub struct InferError {
    pub kind: InferErrorKind,
    pub message: String,
}

#[derive(Clone, Debug)]
pub struct TokenizeRequest {
    pub model_id: Option<String>,
    pub messages: Option<Vec<Message>>,
    pub text: Option<String>,
    pub tools: Vec<Tool>,
    pub add_generation_prompt: bool,
    pub add_special_tokens: bool,
    pub enable_thinking: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct DetokenizeRequest {
    pub model_id: Option<String>,
    pub tokens: Vec<u32>,
    pub skip_special_tokens: bool,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub kind: String,
    pub loaded: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ModelList {
    pub models: Vec<ModelInfo>,
}

#[derive(Clone, Debug)]
pub struct HealthStatus {
    pub ok: bool,
    pub version: String,
}

#[derive(Clone, Debug)]
pub struct ReIsqRequest {
    pub model_id: Option<String>,
    pub isq_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chat_request() -> ChatRequest {
        ChatRequest {
            model_id: Some("qwen3".to_string()),
            messages: vec![Message {
                role: Role::User,
                content: MsgContent::Text("hello".to_string()),
                name: None,
                tool_call_id: None,
            }],
            sampling: SamplingParams {
                temperature: Some(0.7),
                max_tokens: Some(128),
                stop_seqs: vec!["</s>".to_string()],
                ..Default::default()
            },
            stream: true,
            tools: vec![Tool {
                name: "search".to_string(),
                description: "web search".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            tool_choice: ToolChoiceKind::Auto,
            tool_choice_name: None,
            constraint: Constraint::Regex("[0-9]+".to_string()),
            enable_thinking: Some(true),
            return_logprobs: false,
            enable_code_execution: true,
            max_tool_rounds: Some(3),
            session_id: Some("sess-1".to_string()),
            request_id: 42,
        }
    }

    #[test]
    fn chat_request_maps_to_normal_request() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let req = sample_chat_request().into_request(tx);
        let crate::Request::Normal(n) = req else {
            panic!("expected Request::Normal");
        };
        assert_eq!(n.id, 42);
        assert_eq!(n.model_id.as_deref(), Some("qwen3"));
        assert!(n.is_streaming);
        assert!(n.enable_code_execution);
        assert_eq!(n.max_tool_rounds, Some(3));
        assert_eq!(n.session_id.as_deref(), Some("sess-1"));
        assert!(matches!(n.constraint, EngineConstraint::Regex(_)));
        assert_eq!(n.sampling_params.temperature, Some(0.7));
        assert_eq!(n.sampling_params.max_len, Some(128));
        assert!(matches!(
            n.sampling_params.stop_toks,
            Some(crate::sampler::StopTokens::Seqs(_))
        ));
        let tools = n.tools.expect("tools forwarded");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "search");
        let RequestMessage::Chat { messages, enable_thinking, .. } = n.messages else {
            panic!("expected chat messages");
        };
        assert_eq!(enable_thinking, Some(true));
        assert_eq!(messages.len(), 1);
        assert!(matches!(messages[0]["role"], Either::Left(ref r) if r == "user"));
    }

    #[test]
    fn chat_response_from_completion_extracts_first_choice() {
        let completion = ChatCompletionResponse {
            id: "cmpl-1".to_string(),
            choices: vec![crate::response::Choice {
                finish_reason: "stop".to_string(),
                index: 0,
                message: crate::response::ResponseMessage {
                    content: Some("hi there".to_string()),
                    role: "assistant".to_string(),
                    tool_calls: None,
                    reasoning_content: Some("thinking".to_string()),
                },
                logprobs: None,
            }],
            created: 0,
            model: "qwen3".to_string(),
            system_fingerprint: "local".to_string(),
            object: "chat.completion".to_string(),
            usage: crate::response::Usage {
                completion_tokens: 3,
                prompt_tokens: 5,
                total_tokens: 8,
                avg_tok_per_sec: 0.0,
                avg_prompt_tok_per_sec: 0.0,
                avg_compl_tok_per_sec: 30.0,
                total_time_sec: 0.1,
                total_prompt_time_sec: 0.0,
                total_completion_time_sec: 0.0,
            },
            agentic_tool_calls: None,
            files: None,
            session_id: Some("sess-1".to_string()),
        };
        let r = ChatResponse::from_completion(completion);
        assert_eq!(r.id, "cmpl-1");
        assert_eq!(r.content, "hi there");
        assert_eq!(r.reasoning_content.as_deref(), Some("thinking"));
        assert_eq!(r.finish_reason, "stop");
        assert_eq!(r.usage.total_tokens, 8);
        assert_eq!(r.session_id.as_deref(), Some("sess-1"));
    }
}
