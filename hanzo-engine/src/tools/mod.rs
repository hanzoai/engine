pub(crate) mod grammar;
pub(crate) mod parsers;
mod request;
mod response;

use hanzo_ml::Result;
pub use request::*;
pub use response::*;
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use uuid::Uuid;

use hanzo_llm_mcp::CalledFunction;
pub use parsers::ToolCallFormat;

pub use hanzo_llm_mcp::{ToolCallback, ToolCallbackWithTool};

/// Collection of callbacks keyed by tool name.
pub type ToolCallbacks = HashMap<String, Arc<ToolCallback>>;

/// Collection of callbacks with their tool definitions keyed by tool name.
pub type ToolCallbacksWithTools = HashMap<String, ToolCallbackWithTool>;

fn contains_tool_call_prefix(prefix: &str) -> bool {
    parsers::contains_tool_call_prefix(prefix)
}

fn process_model_specific_message(
    message: &str,
    format: Option<ToolCallFormat>,
) -> Result<String> {
    parsers::process_model_specific_message_in(message, format)
}

pub struct ToolCallingMatcher {
    tool_choice: ToolChoice,
    format: Option<ToolCallFormat>,
    known_tool_names: Option<std::collections::HashSet<String>>,
    tools: Option<Arc<Vec<crate::Tool>>>,
}

// Same as CalledFunction, but has different cases for variations on the names
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionParameters {
    #[serde(alias = "function")]
    pub name: String,
    #[serde(alias = "arguments", deserialize_with = "flexible_args")]
    pub parameters: Value,
}

// Accept either `{...}` **or** a `"stringified { ... }"`
fn flexible_args<'de, D>(d: D) -> std::result::Result<Value, D::Error>
where
    D: Deserializer<'de>,
{
    struct ArgVisitor;

    impl<'de> Visitor<'de> for ArgVisitor {
        type Value = Value;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("an object or a JSON-encoded string containing an object")
        }

        // Case 1 – the good case: already a JSON object
        fn visit_map<M>(self, mut m: M) -> std::result::Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut map = Map::new();
            while let Some((k, v)) = m.next_entry()? {
                map.insert(k, v);
            }
            Ok(Value::Object(map))
        }

        // Case 2 – got a *string*; try parsing it as JSON
        fn visit_str<E>(self, s: &str) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            serde_json::from_str(s).map_err(|e| E::custom(format!("inner JSON error: {e}")))
        }
    }

    d.deserialize_any(ArgVisitor)
}

/// Fixup potentially broken JSON
/// 1) allow/handle arguments as maps in quotations
fn fix_broken_json(raw: &str) -> anyhow::Result<String> {
    // Only apply the fix if the first pattern matches - otherwise we might corrupt valid JSON
    // where arguments is a properly escaped string containing `}`
    if raw.contains(r#""arguments":"{"#) {
        // 1) Delete the opening quote that shouldn't be there
        let tmp = raw.replacen(r#""arguments":"{"#, r#""arguments":{"#, 1);
        // 2) Delete the closing quote that matches it
        let fixed = tmp.replacen(r#"}"}"#, r#"}}"#, 1);
        Ok(fixed)
    } else {
        Ok(raw.to_string())
    }
}

impl ToolCallingMatcher {
    /// `required` and a named tool both oblige the model to emit a call.
    fn must_call(&self) -> bool {
        matches!(self.tool_choice, ToolChoice::Tool(_) | ToolChoice::Required)
    }

    /// `format` is what the model's chat template says it emits ([`ToolCallFormat::of_template`]);
    /// `None` reads every format the parsers know.
    pub fn new(
        tool_choice: ToolChoice,
        tools: Option<&[crate::Tool]>,
        format: Option<ToolCallFormat>,
    ) -> anyhow::Result<Self> {
        let known_tool_names = tools.map(|t| {
            t.iter()
                .map(|tool| tool.function.name.clone())
                .collect::<std::collections::HashSet<_>>()
        });
        let tools_arc = tools.map(|t| Arc::new(t.to_vec()));
        Ok(Self {
            tool_choice,
            format,
            known_tool_names,
            tools: tools_arc,
        })
    }

    /// Build a tool call grammar if a known format prefix is detected in
    /// `text` and tools are available.  Returns `None` when tool choice is
    /// `None`, no format matches, or the format is not yet ready (e.g.
    /// DeepSeek before the JSON fence).
    pub fn build_tool_call_grammar(&self, text: &str) -> Option<llguidance::api::TopLevelGrammar> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return None;
        }
        let tools = self.tools.as_ref()?;
        parsers::build_tool_call_grammar_in(text, tools, self.format)
    }

    /// Build a pure JSON object grammar for Harmony tool call arguments.
    /// When `tool_name` identifies a tool with `strict: true`, its
    /// parameters schema is used for constrained decoding.
    /// Returns `None` when tool choice is `None` or no tools are defined.
    pub fn build_harmony_tool_grammar(
        &self,
        tool_name: Option<&str>,
    ) -> Option<llguidance::api::TopLevelGrammar> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return None;
        }
        let tools = self.tools.as_ref()?;
        Some(parsers::harmony::tool_call_grammar_for_tool(
            tool_name,
            Some(tools),
        ))
    }

    // Checks if the `message_prefix` could be a tool call. If false, either
    // [`ToolChoice::None`] was selected, or the prefix could not match.
    //
    // If the start of a message could be a tool call, then it looks like an incomplete JSON of a given structure, e.g. `{"name": "foo", "param`.
    //
    // Returns a tuple of `(could_be_tool, is_complete_tool)`.
    pub fn prefix_could_be_tool(&self, message_prefix: &str) -> Result<(bool, bool)> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok((false, false));
        }
        let message_prefix = process_model_specific_message(message_prefix, self.format)?;
        let message_prefix = fix_broken_json(&message_prefix).map_err(hanzo_ml::Error::msg)?;

        // Check if the prefix could be a JSON serialization of any of the following types.
        Ok([
            could_be_json::<CalledFunctionParameters>,
            could_be_json::<Vec<CalledFunctionParameters>>,
        ]
        .iter()
        .find_map(|check| {
            let (could_be_tool, is_complete_tool) = check(&message_prefix);
            if could_be_tool || is_complete_tool {
                Some((could_be_tool, is_complete_tool))
            } else {
                None
            }
        })
        .unwrap_or((contains_tool_call_prefix(&message_prefix), false)))
    }

    pub fn get_call(&self, message: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok(Vec::new());
        }
        let message = process_model_specific_message(message, self.format)?;
        let message = fix_broken_json(&message)?;

        let mut calls = if let Ok(deser) =
            serde_json::from_str::<CalledFunctionParameters>(&message)
        {
            let id = format!("call-{}", Uuid::new_v4());
            vec![ToolCallResponse {
                index: 0,
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: serde_json::to_string(&deser.parameters)?,
                },
            }]
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionParameters>>(&message) {
            deser
                .into_iter()
                .enumerate()
                .map(|(idx, deser)| {
                    let id = format!("call-{}", Uuid::new_v4());
                    Ok(ToolCallResponse {
                        index: idx,
                        id,
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: deser.name,
                            arguments: serde_json::to_string(&deser.parameters)?,
                        },
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            if self.must_call() {
                anyhow::bail!("Tool choice was required but no tools were called.")
            }
            return Ok(Vec::new());
        };

        if self.format == Some(ToolCallFormat::Qwen3Coder) {
            if let Some(tools) = &self.tools {
                for call in &mut calls {
                    call.function.arguments =
                        typed_arguments(tools, &call.function.name, &call.function.arguments);
                }
            }
        }

        // Filter out hallucinated tool names.
        if let Some(ref known) = self.known_tool_names {
            let before = calls.len();
            calls.retain(|tc| {
                let valid = known.contains(&tc.function.name);
                if !valid {
                    tracing::warn!(
                        "Dropping hallucinated tool call `{}` (not in defined tools: {:?})",
                        tc.function.name,
                        known
                    );
                }
                valid
            });
            if calls.is_empty() && before > 0 && self.must_call() {
                anyhow::bail!("Tool choice was required but model called unknown tools.");
            }
        }

        Ok(calls)
    }
}

/// The Qwen3-Coder format writes every argument as text. Each takes the type its tool's schema
/// declares, as vLLM's `qwen3_coder` parser gives it: `null` is null, a string stays one, integers and
/// numbers parse (a whole number is an integer), booleans are `true` or not, objects and arrays are
/// JSON, and a value that does not fit its type stays text.
fn typed_arguments(tools: &[crate::Tool], name: &str, arguments: &str) -> String {
    let Some(tool) = tools.iter().find(|t| t.function.name == name) else {
        return arguments.to_string();
    };
    let Ok(Value::Object(mut args)) = serde_json::from_str::<Value>(arguments) else {
        return arguments.to_string();
    };
    let properties = tool
        .function
        .parameters
        .as_ref()
        .and_then(|p| p.get("properties"))
        .and_then(Value::as_object);
    for (key, value) in args.iter_mut() {
        if let Value::String(text) = value {
            let kind = properties.and_then(|p| p.get(key)).and_then(|p| p.get("type"));
            if let Some(typed) = typed_value(text, kind) {
                *value = typed;
            }
        }
    }
    serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| arguments.to_string())
}

fn typed_value(text: &str, kind: Option<&Value>) -> Option<Value> {
    if text.eq_ignore_ascii_case("null") {
        return Some(Value::Null);
    }
    let kinds: Vec<String> = match kind {
        Some(Value::String(k)) => vec![k.to_lowercase()],
        Some(Value::Array(ks)) => ks
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_lowercase)
            .collect(),
        _ => Vec::new(),
    };
    let json = || serde_json::from_str::<Value>(text).ok();
    if kinds.is_empty() {
        return json().filter(|v| matches!(v, Value::Number(_) | Value::Array(_) | Value::Object(_)));
    }
    if kinds
        .iter()
        .any(|k| matches!(k.as_str(), "string" | "str" | "text" | "varchar" | "char" | "enum"))
    {
        return None;
    }
    for k in &kinds {
        let typed = match k.as_str() {
            k if k.starts_with("int") || k.starts_with("uint") || matches!(k, "long" | "short" | "unsigned") => {
                text.trim().parse::<i64>().ok().map(Value::from)
            }
            "number" | "float" | "double" => text.trim().parse::<f64>().ok().and_then(|f| {
                if f.is_finite() && f.fract() == 0.0 && f.abs() < 9e15 {
                    Some(Value::from(f as i64))
                } else {
                    serde_json::Number::from_f64(f).map(Value::Number)
                }
            }),
            "boolean" | "bool" => Some(Value::Bool(text.trim().eq_ignore_ascii_case("true"))),
            "object" | "array" | "dict" | "list" => json().filter(|v| v.is_object() || v.is_array()),
            _ => None,
        };
        if typed.is_some() {
            return typed;
        }
    }
    None
}

/// Checks if the given prefix could be the start of, or the entire JSON serialization of a given type, `T`.
///
/// Returns a tuple of `(could_be_tool, is_entire_tool)`.
fn could_be_json<T>(text_prefix: &str) -> (bool, bool)
where
    T: serde::de::DeserializeOwned,
{
    if text_prefix.trim().is_empty() {
        return (false, false);
    }
    match serde_json::from_str::<T>(text_prefix) {
        Ok(_) => (false, true),
        // EOF show that JSON parsing was successful up to the end of the entire string.
        Err(e) if e.is_eof() => (true, false),
        _ => (false, false),
    }
}

/// Takes raw UTf8 text and parses any possible tool calls from it.
pub fn parse_text_tools(
    raw_text: &str,
    matcher: Option<Arc<ToolCallingMatcher>>,
) -> anyhow::Result<(Option<&str>, Vec<ToolCallResponse>)> {
    let mut tool_calls = Vec::new();
    let mut text_new = Some(raw_text);

    if let Some(ref matcher) = matcher {
        let calls = matcher.get_call(raw_text).map_err(hanzo_ml::Error::msg)?;
        if !calls.is_empty() {
            text_new = None;
            tool_calls = calls;
        }
    };
    Ok((text_new, tool_calls))
}

#[cfg(test)]
mod qwen3_coder_tests {
    use super::*;
    use crate::{Function, Tool, ToolType};
    use serde_json::json;
    use std::collections::HashMap;

    fn tool() -> Tool {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "scale": {"type": "number"},
                "metric": {"type": "boolean"},
                "tags": {"type": "array"},
                "zip": {"type": "string"},
            }
        }))
        .unwrap();
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: None,
                name: "weather".to_string(),
                parameters: Some(parameters),
                strict: None,
            },
        }
    }

    const MESSAGE: &str = "<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=scale>\n2.0\n</parameter>\n<parameter=metric>\nTrue\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n<parameter=zip>\n75001\n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn arguments_take_the_type_of_their_schema() {
        let matcher =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool()]), Some(ToolCallFormat::Qwen3Coder))
                .unwrap();
        let calls = matcher.get_call(MESSAGE).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "weather");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args,
            json!({"city": "Paris", "days": 3, "scale": 2, "metric": true, "tags": ["a", "b"], "zip": "75001"})
        );
    }

    #[test]
    fn a_call_in_progress_is_a_possible_tool_and_a_finished_one_is_not_json() {
        let matcher =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool()]), Some(ToolCallFormat::Qwen3Coder))
                .unwrap();
        let (possible, _) = matcher
            .prefix_could_be_tool("<tool_call>\n<function=weather>\n<parameter=ci")
            .unwrap();
        assert!(possible);
        assert!(matcher
            .build_tool_call_grammar("<tool_call>\n")
            .is_none());
    }

    #[test]
    fn a_json_call_still_parses_under_the_hint() {
        let matcher =
            ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool()]), Some(ToolCallFormat::Qwen3Coder))
                .unwrap();
        let calls = matcher
            .get_call("<tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Rome\"}}</tool_call>")
            .unwrap();
        assert_eq!(calls.len(), 1);
    }
}
