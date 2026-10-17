//! What the client reads back: usage details, the `timings` object, the served model id,
//! `stream_options.include_usage` framing, and one tool call when parallel calls are off.

use std::collections::HashSet;

use hanzo_engine::{ChatCompletionChunkResponse, ChatCompletionResponse, Usage};
use serde::Serialize;
use serde_json::{json, Value};

/// Per-request settings for how responses are written.
#[derive(Debug, Clone, Default)]
pub struct Framing {
    /// The model id every response carries (`--served-model-name`); `None` keeps the engine's.
    pub model: Option<String>,
    /// `stream_options.include_usage`.
    pub include_usage: bool,
    /// `parallel_tool_calls: false`: only the first tool call is kept.
    pub single_call: bool,
}

impl Framing {
    /// A finished chat response: the served model id, and one tool call when parallel calls
    /// are off.
    pub fn finish(&self, response: &mut ChatCompletionResponse) {
        FirstCall::new(self.single_call).response(response);
        if let Some(model) = &self.model {
            response.model = model.clone();
        }
    }
}

/// Chat and completions usage: the engine's counters, plus
/// `completion_tokens_details.reasoning_tokens` always and `prompt_tokens_details.cached_tokens`
/// when the prompt cache covered some of the prompt.
pub fn usage(u: &Usage) -> Value {
    let mut out = json!(u);
    if u.cached_prompt_tokens > 0 {
        out["prompt_tokens_details"] = json!({ "cached_tokens": u.cached_prompt_tokens });
    }
    out["completion_tokens_details"] = json!({ "reasoning_tokens": u.reasoning_tokens });
    out
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// The `timings` object. `prompt_n` counts the prompt tokens actually
/// processed (the cached ones are `cache_n`); the draft counts appear when a proposer drafted.
pub fn timings(u: &Usage) -> Value {
    let prompt_n = u.prompt_tokens.saturating_sub(u.cached_prompt_tokens);
    let prompt_ms = round3(f64::from(u.total_prompt_time_sec) * 1000.0);
    let predicted_ms = round3(f64::from(u.total_completion_time_sec) * 1000.0);
    let rate = |n: usize, ms: f64| {
        if ms > 0.0 && n > 0 {
            round3(n as f64 / (ms / 1000.0))
        } else {
            0.0
        }
    };
    let mut out = json!({
        "prompt_n": prompt_n,
        "predicted_n": u.completion_tokens,
        "prompt_ms": prompt_ms,
        "predicted_ms": predicted_ms,
        "prompt_per_second": rate(prompt_n, prompt_ms),
        "predicted_per_second": rate(u.completion_tokens, predicted_ms),
        "cache_n": u.cached_prompt_tokens,
    });
    if u.draft_tokens > 0 {
        out["draft_n"] = json!(u.draft_tokens);
        out["draft_n_accepted"] = json!(u.draft_accepted);
    }
    out
}

/// A finished chat or completions body: its `usage` rendered with details, and `timings`.
pub fn body<T: Serialize>(response: &T, u: &Usage) -> Value {
    let mut out = json!(response);
    out["usage"] = usage(u);
    out["timings"] = timings(u);
    out
}

/// Frames one streamed chunk. The chunk that carries the usage gets
/// `timings`. Without `include_usage` no chunk carries `usage`; with it every chunk carries
/// `"usage": null`, and the usage goes in a chunk of its own with no choices, returned second,
/// to be sent just before `[DONE]`.
pub fn chunk<T: Serialize>(
    chunk: &T,
    u: Option<&Usage>,
    framing: &Framing,
) -> (Value, Option<Value>) {
    let mut out = json!(chunk);
    if let Some(model) = &framing.model {
        out["model"] = json!(model);
    }
    if let Some(object) = out.as_object_mut() {
        object.remove("usage");
    }
    if let Some(u) = u {
        out["timings"] = timings(u);
    }
    if !framing.include_usage {
        return (out, None);
    }
    out["usage"] = Value::Null;
    let tail = u.map(|u| {
        json!({
            "id": out["id"],
            "object": out["object"],
            "created": out["created"],
            "model": out["model"],
            "system_fingerprint": out["system_fingerprint"],
            "choices": [],
            "usage": usage(u),
            "timings": timings(u),
        })
    });
    (out, tail)
}

/// Keeps only each choice's first tool call when parallel calls are off. A stream sends a call
/// whole in one delta, so every call after a choice's first is dropped.
#[derive(Debug, Default)]
pub struct FirstCall {
    on: bool,
    seen: HashSet<usize>,
}

impl FirstCall {
    pub fn new(on: bool) -> Self {
        Self {
            on,
            seen: HashSet::new(),
        }
    }

    /// A finished response: each choice keeps its first call.
    pub fn response(&self, response: &mut ChatCompletionResponse) {
        if !self.on {
            return;
        }
        for choice in &mut response.choices {
            if let Some(calls) = &mut choice.message.tool_calls {
                calls.truncate(1);
            }
        }
    }

    /// A streamed chunk: the first call a choice sends goes through, later ones are dropped.
    pub fn chunk(&mut self, chunk: &mut ChatCompletionChunkResponse) {
        if !self.on {
            return;
        }
        for choice in &mut chunk.choices {
            let Some(calls) = &mut choice.delta.tool_calls else {
                continue;
            };
            if self.seen.insert(choice.index) {
                calls.truncate(1);
            } else {
                calls.clear();
            }
            if calls.is_empty() {
                choice.delta.tool_calls = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use hanzo_engine::{CalledFunction, ChunkChoice, Delta, ToolCallResponse, ToolCallType};

    use super::*;

    fn usage_of(prompt: usize, cached: usize, completion: usize) -> Usage {
        Usage {
            prompt_tokens: prompt,
            cached_prompt_tokens: cached,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            total_prompt_time_sec: 0.5,
            total_completion_time_sec: 2.0,
            reasoning_tokens: 7,
            ..Usage::default()
        }
    }

    #[test]
    fn usage_details() {
        let u = usage(&usage_of(100, 0, 20));
        assert_eq!(u["completion_tokens_details"]["reasoning_tokens"], 7);
        assert!(u.get("prompt_tokens_details").is_none());
        assert_eq!(u["prompt_tokens"], 100);

        let u = usage(&usage_of(100, 64, 20));
        assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 64);
    }

    #[test]
    fn timings_in_llama_server_shape() {
        let t = timings(&usage_of(100, 60, 20));
        assert_eq!(
            t,
            json!({
                "prompt_n": 40,
                "predicted_n": 20,
                "prompt_ms": 500.0,
                "predicted_ms": 2000.0,
                "prompt_per_second": 80.0,
                "predicted_per_second": 10.0,
                "cache_n": 60,
            })
        );
        let drafted = Usage {
            draft_tokens: 30,
            draft_accepted: 21,
            ..usage_of(10, 0, 5)
        };
        let t = timings(&drafted);
        assert_eq!(t["draft_n"], 30);
        assert_eq!(t["draft_n_accepted"], 21);
        let idle = timings(&Usage::default());
        assert_eq!(idle["prompt_per_second"], 0.0);
        assert_eq!(idle["predicted_per_second"], 0.0);
    }

    fn call(index: usize, name: &str) -> ToolCallResponse {
        ToolCallResponse {
            index,
            id: format!("call-{name}"),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    fn chat_chunk(
        calls: Option<Vec<ToolCallResponse>>,
        finish: Option<&str>,
        u: Option<Usage>,
    ) -> ChatCompletionChunkResponse {
        ChatCompletionChunkResponse {
            id: "1".to_string(),
            choices: vec![ChunkChoice {
                finish_reason: finish.map(str::to_string),
                index: 0,
                delta: Delta {
                    content: None,
                    role: "assistant".to_string(),
                    tool_calls: calls,
                    reasoning_content: None,
                },
                logprobs: None,
            }],
            created: 1,
            model: "engine-id".to_string(),
            system_fingerprint: "local".to_string(),
            object: "chat.completion.chunk".to_string(),
            usage: u,
            session_id: None,
        }
    }

    #[test]
    fn stream_without_include_usage_carries_no_usage() {
        let framing = Framing::default();
        let plain = chat_chunk(None, None, None);
        let (frame, tail) = chunk(&plain, plain.usage.as_ref(), &framing);
        assert!(frame.get("usage").is_none());
        assert!(frame.get("timings").is_none());
        assert!(tail.is_none());

        let last = chat_chunk(None, Some("stop"), Some(usage_of(10, 0, 3)));
        let (frame, tail) = chunk(&last, last.usage.as_ref(), &framing);
        assert!(frame.get("usage").is_none());
        assert_eq!(frame["timings"]["predicted_n"], 3);
        assert!(tail.is_none());
    }

    #[test]
    fn include_usage_adds_a_usage_chunk() {
        let framing = Framing {
            model: Some("served".to_string()),
            include_usage: true,
            single_call: false,
        };
        let plain = chat_chunk(None, None, None);
        let (frame, tail) = chunk(&plain, None, &framing);
        assert!(frame["usage"].is_null());
        assert!(frame.as_object().unwrap().contains_key("usage"));
        assert_eq!(frame["model"], "served");
        assert!(tail.is_none());

        let last = chat_chunk(None, Some("stop"), Some(usage_of(10, 0, 3)));
        let (frame, tail) = chunk(&last, last.usage.as_ref(), &framing);
        assert!(frame["usage"].is_null());
        assert_eq!(frame["timings"]["predicted_n"], 3);
        let tail = tail.expect("a usage chunk");
        assert_eq!(tail["choices"], json!([]));
        assert_eq!(tail["usage"]["completion_tokens"], 3);
        assert_eq!(
            tail["usage"]["completion_tokens_details"]["reasoning_tokens"],
            7
        );
        assert_eq!(tail["timings"]["predicted_n"], 3);
        assert_eq!(tail["model"], "served");
        assert_eq!(tail["object"], "chat.completion.chunk");
        assert_eq!(tail["id"], "1");
    }

    #[test]
    fn first_call_only_when_parallel_calls_are_off() {
        let mut off = FirstCall::new(true);
        let mut both = chat_chunk(Some(vec![call(0, "a"), call(1, "b")]), None, None);
        off.chunk(&mut both);
        let calls = both.choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "a");
        let mut later = chat_chunk(Some(vec![call(0, "c")]), Some("tool_calls"), None);
        off.chunk(&mut later);
        assert!(later.choices[0].delta.tool_calls.is_none());

        let mut on = FirstCall::new(false);
        let mut both = chat_chunk(Some(vec![call(0, "a"), call(1, "b")]), None, None);
        on.chunk(&mut both);
        assert_eq!(both.choices[0].delta.tool_calls.as_ref().unwrap().len(), 2);
    }
}
