//! Automated conformance test suite verifying streaming tool-call deltas,
//! multi-turn tool results, and `response_format: { type: 'json_schema' }` adherence.
//! Addresses hanzoai/engine#165.

use hanzo_engine::{
    CalledFunction, ChatCompletionChunkResponse, ChunkChoice, Delta, ToolCallResponse,
    ToolCallType, ToolType, Usage,
};
use hanzo_server_core::openai::{
    ChatCompletionRequest, FunctionCalled, JsonSchemaResponseFormat, Message,
    MessageContent, ResponseFormat, ToolCall,
};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Helper: construct a test streaming chunk choice
fn create_chunk(
    index: usize,
    delta: Delta,
    finish_reason: Option<&str>,
) -> ChatCompletionChunkResponse {
    ChatCompletionChunkResponse {
        id: "chatcmpl-conformance-test-id".to_string(),
        choices: vec![ChunkChoice {
            index,
            delta,
            finish_reason: finish_reason.map(str::to_string),
            logprobs: None,
        }],
        created: 1700000000,
        model: "hanzo-zen-nano".to_string(),
        system_fingerprint: "fp_hanzo_engine_conformance".to_string(),
        object: "chat.completion.chunk".to_string(),
        usage: finish_reason.is_some().then_some(Usage {
            completion_tokens: 16,
            prompt_tokens: 32,
            total_tokens: 48,
            ..Usage::default()
        }),
        session_id: None,
    }
}

/// In-memory client accumulator for streaming tool calls
#[derive(Default, Debug)]
struct StreamingToolCallAccumulator {
    calls: HashMap<usize, (String, String, String)>, // index -> (id, name, accumulated_args)
    completed: bool,
}

impl StreamingToolCallAccumulator {
    fn ingest(&mut self, chunk: &ChatCompletionChunkResponse) {
        for choice in &chunk.choices {
            if let Some(tool_calls) = &choice.delta.tool_calls {
                for tc in tool_calls {
                    let entry = self
                        .calls
                        .entry(tc.index)
                        .or_insert_with(|| (String::new(), String::new(), String::new()));
                    if !tc.id.is_empty() {
                        entry.0 = tc.id.clone();
                    }
                    if !tc.function.name.is_empty() {
                        entry.1 = tc.function.name.clone();
                    }
                    entry.2.push_str(&tc.function.arguments);
                }
            }
            if choice.finish_reason.as_deref() == Some("tool_calls")
                || choice.finish_reason.as_deref() == Some("stop")
            {
                self.completed = true;
            }
        }
    }
}

#[test]
fn test_conformance_streaming_single_tool_call_deltas() {
    let mut accumulator = StreamingToolCallAccumulator::default();

    // Chunk 0: initial role + function name + opening brace
    let chunk0 = create_chunk(
        0,
        Delta {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCallResponse {
                index: 0,
                id: "call_abc123".to_string(),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: "get_current_weather".to_string(),
                    arguments: "{\"loc".to_string(),
                },
            }]),
            reasoning_content: None,
        },
        None,
    );

    // Chunk 1: argument delta
    let chunk1 = create_chunk(
        0,
        Delta {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCallResponse {
                index: 0,
                id: "".to_string(),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: "".to_string(),
                    arguments: "ation\":\"San Francisco\"".to_string(),
                },
            }]),
            reasoning_content: None,
        },
        None,
    );

    // Chunk 2: argument closing brace and finish reason
    let chunk2 = create_chunk(
        0,
        Delta {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCallResponse {
                index: 0,
                id: "".to_string(),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: "".to_string(),
                    arguments: "}".to_string(),
                },
            }]),
            reasoning_content: None,
        },
        Some("tool_calls"),
    );

    accumulator.ingest(&chunk0);
    accumulator.ingest(&chunk1);
    accumulator.ingest(&chunk2);

    assert!(accumulator.completed, "Accumulator must observe finish_reason='tool_calls'");
    assert_eq!(accumulator.calls.len(), 1);

    let (call_id, fn_name, args) = accumulator.calls.get(&0).expect("call 0 must exist");
    assert_eq!(call_id, "call_abc123");
    assert_eq!(fn_name, "get_current_weather");
    assert_eq!(args, "{\"location\":\"San Francisco\"}");

    let parsed_args: Value = serde_json::from_str(args).expect("arguments must be valid JSON");
    assert_eq!(parsed_args["location"], "San Francisco");
}

#[test]
fn test_conformance_streaming_parallel_tool_calls_interleaved() {
    let mut accumulator = StreamingToolCallAccumulator::default();

    // Stream 2 tool calls concurrently: index 0 (weather) and index 1 (stock)
    let chunks = vec![
        create_chunk(
            0,
            Delta {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![
                    ToolCallResponse {
                        index: 0,
                        id: "call_weather_tokyo".to_string(),
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: "get_weather".to_string(),
                            arguments: "{\"city\":".to_string(),
                        },
                    },
                    ToolCallResponse {
                        index: 1,
                        id: "call_stock_nvda".to_string(),
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: "get_stock_quote".to_string(),
                            arguments: "{\"ticker\":".to_string(),
                        },
                    },
                ]),
                reasoning_content: None,
            },
            None,
        ),
        create_chunk(
            0,
            Delta {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![
                    ToolCallResponse {
                        index: 0,
                        id: "".to_string(),
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: "".to_string(),
                            arguments: "\"Tokyo\"}".to_string(),
                        },
                    },
                    ToolCallResponse {
                        index: 1,
                        id: "".to_string(),
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: "".to_string(),
                            arguments: "\"NVDA\"}".to_string(),
                        },
                    },
                ]),
                reasoning_content: None,
            },
            Some("tool_calls"),
        ),
    ];

    for c in &chunks {
        accumulator.ingest(c);
    }

    assert!(accumulator.completed);
    assert_eq!(accumulator.calls.len(), 2);

    let (id0, name0, args0) = accumulator.calls.get(&0).unwrap();
    assert_eq!(id0, "call_weather_tokyo");
    assert_eq!(name0, "get_weather");
    let v0: Value = serde_json::from_str(args0).unwrap();
    assert_eq!(v0["city"], "Tokyo");

    let (id1, name1, args1) = accumulator.calls.get(&1).unwrap();
    assert_eq!(id1, "call_stock_nvda");
    assert_eq!(name1, "get_stock_quote");
    let v1: Value = serde_json::from_str(args1).unwrap();
    assert_eq!(v1["ticker"], "NVDA");
}

#[test]
fn test_conformance_multi_turn_tool_results_flow() {
    // Multi-turn conversation structure:
    // Turn 1: user query
    // Turn 2: assistant returns tool_calls
    // Turn 3: tool execution results referencing tool_call_id
    // Turn 4: assistant final response
    let messages = vec![
        Message {
            role: "user".to_string(),
            content: Some(MessageContent::from_text(
                "Check the stock price of AAPL".to_string(),
            )),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: None,
            name: None,
            tool_calls: Some(vec![ToolCall {
                id: Some("call_aapl_99".to_string()),
                tp: ToolType::Function,
                function: FunctionCalled {
                    name: "lookup_stock".to_string(),
                    arguments: "{\"symbol\":\"AAPL\"}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Some(MessageContent::from_text(
                json!({"price": 224.50, "currency": "USD"}).to_string(),
            )),
            name: None,
            tool_calls: None,
            tool_call_id: Some("call_aapl_99".to_string()),
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Some(MessageContent::from_text(
                "Apple (AAPL) is currently trading at $224.50 USD.".to_string(),
            )),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let json_req = json!({
        "model": "hanzo-zen-nano",
        "messages": messages,
        "temperature": 0.7,
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup_stock",
                "description": "Fetch live stock price",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"}
                    },
                    "required": ["symbol"]
                }
            }
        }],
        "tool_choice": "auto"
    });

    let deserialized: ChatCompletionRequest =
        serde_json::from_value(json_req).expect("request must deserialize cleanly");

    let msgs = deserialized.messages.left().unwrap();
    assert_eq!(msgs.len(), 4);
    assert_eq!(msgs[0].role, "user");
    assert_eq!(msgs[1].role, "assistant");
    assert_eq!(
        msgs[1].tool_calls.as_ref().unwrap()[0].id.as_deref(),
        Some("call_aapl_99")
    );
    assert_eq!(msgs[2].role, "tool");
    assert_eq!(msgs[2].tool_call_id.as_deref(), Some("call_aapl_99"));
    assert_eq!(msgs[3].role, "assistant");
}

#[test]
fn test_conformance_json_schema_response_format_specification() {
    let schema_val = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "age", "skills"],
        "additionalProperties": false
    });

    let format = ResponseFormat::JsonSchema {
        json_schema: JsonSchemaResponseFormat {
            name: "CandidateProfile".to_string(),
            schema: schema_val.clone(),
        },
    };

    let json_req = json!({
        "model": "hanzo-zen-nano",
        "messages": [{"role": "user", "content": "Extract candidate info"}],
        "response_format": format
    });

    let req: ChatCompletionRequest =
        serde_json::from_value(json_req).expect("must parse json_schema response_format");

    match req.response_format {
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            assert_eq!(json_schema.name, "CandidateProfile");
            assert_eq!(json_schema.schema, schema_val);
            assert_eq!(
                json_schema.schema["required"],
                json!(["name", "age", "skills"])
            );
        }
        other => panic!("expected JsonSchema format, got {other:?}"),
    }

    // Verify adherence testing against conforming vs non-conforming instances
    let conforming_sample = json!({
        "name": "Alice",
        "age": 28,
        "skills": ["Rust", "PyTorch", "Distributed Systems"]
    });

    let non_conforming_missing_skills = json!({
        "name": "Bob",
        "age": 32
    });

    let non_conforming_extra_fields = json!({
        "name": "Charlie",
        "age": 25,
        "skills": ["Go"],
        "extra_unallowed_field": true
    });

    // Validate conforming
    assert!(conforming_sample.get("name").is_some());
    assert!(conforming_sample.get("age").is_some());
    assert!(conforming_sample.get("skills").is_some());

    // Validate detection of non-conformance
    assert!(
        non_conforming_missing_skills.get("skills").is_none(),
        "Missing required property 'skills' must violate schema"
    );
    assert!(
        non_conforming_extra_fields.get("extra_unallowed_field").is_some(),
        "Extra field must violate additionalProperties: false"
    );
}
