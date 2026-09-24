//! Qwen tool call parser.
//!
//! Qwen emits each call in its own `<tool_call>` block, in one of two bodies:
//!
//! - JSON (Qwen2.5 and Qwen3): `<tool_call>{"name":"...", "arguments":{...}}</tool_call>`
//! - XML (Qwen3.5 and later, whose template says `<function=`):
//!
//!   ```text
//!   <tool_call>
//!   <function=NAME>
//!   <parameter=KEY>
//!   VALUE
//!   </parameter>
//!   </function>
//!   </tool_call>
//!   ```
//!
//! An XML value is untyped text. The template renders a string raw and everything else through
//! `tojson`, so `3` is the same bytes whether the parameter is an integer or a string, and only
//! the tool's declared schema tells them apart: a `string` is kept verbatim, anything else is
//! parsed, and a value that does not parse as its declared type is kept as text rather than
//! dropped.

use llguidance::api::TopLevelGrammar;
use regex::Regex;
use serde_json::{Map, Value};
use std::sync::OnceLock;

use super::ToolFormatParser;
use crate::Tool;

static BLOCK: OnceLock<Regex> = OnceLock::new();
static FUNCTION: OnceLock<Regex> = OnceLock::new();
static PARAMETER: OnceLock<Regex> = OnceLock::new();
static AFTER_VALUE: OnceLock<Regex> = OnceLock::new();

const PARAM_END: &str = "</parameter>";

pub struct QwenParser;

impl ToolFormatParser for QwenParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Qwen
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        // `</tool_call>` is matched as a special token via bare
        // angle-bracket syntax (not a string literal).
        crate::tools::grammar::build_json_format_grammar(
            r#"start: @json_body </tool_call>"#.to_string(),
            tools,
            "arguments",
            false,
        )
    }

    /// Halogen spec §9.3: the text between and around the closed blocks. An unclosed trailing
    /// `<tool_call>` is not a call, so it stays.
    fn outside(&self, message: &str) -> String {
        block().replace_all(message, "").into_owned()
    }

    fn parse(&self, message: &str, tools: &[Tool]) -> hanzo_ml::Result<Option<String>> {
        let mut calls = Vec::new();
        for caps in block().captures_iter(message) {
            let inner = caps.name("inner").unwrap().as_str();
            let call = if inner.starts_with("<function=") {
                xml_call(inner, tools)
            } else {
                serde_json::from_str::<Value>(inner).ok()
            };
            // A block that is neither is left for the caller's JSON repair to see.
            match call {
                Some(call) => calls.push(call),
                None => return Ok(Some(inner.to_string())),
            }
        }
        Ok(match calls.len() {
            0 => None,
            1 => Some(calls.remove(0).to_string()),
            _ => Some(Value::Array(calls).to_string()),
        })
    }
}

/// One closed `<tool_call>` block, its body captured as `inner`.
fn block() -> &'static Regex {
    BLOCK.get_or_init(|| Regex::new(r"(?s)<tool_call>\s*(?P<inner>.*?)\s*</tool_call>").unwrap())
}

/// One XML call as `{"name": ..., "arguments": {...}}`.
fn xml_call(body: &str, tools: &[Tool]) -> Option<Value> {
    let function = FUNCTION.get_or_init(|| Regex::new(r"<function=([^>\n]*)>[ \t]*\n?").unwrap());
    let caps = function.captures(body)?;
    let name = caps.get(1)?.as_str().trim().to_string();
    let params = body[caps.get(0)?.end()..].to_string();
    let mut arguments = Map::new();
    for (key, raw) in parameters(&params) {
        let value = coerce(&raw, declared_type(tools, &name, &key).as_deref());
        arguments.insert(key, value);
    }
    let mut call = Map::new();
    call.insert("name".into(), Value::String(name));
    call.insert("arguments".into(), Value::Object(arguments));
    Some(Value::Object(call))
}

/// Every `(key, raw value)` in a function body, in order.
fn parameters(body: &str) -> Vec<(String, String)> {
    let parameter =
        PARAMETER.get_or_init(|| Regex::new(r"<parameter=([^>\n]*)>[ \t]*\n?").unwrap());
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(caps) = parameter.captures_at(body, at) {
        let whole = caps.get(0).unwrap();
        let (end, resume) = value_end(body, whole.end());
        let mut raw = body[whole.end()..end].to_string();
        // The newline before the closing tag is the template's separator, not data.
        if raw.ends_with('\n') {
            raw.pop();
        }
        out.push((caps.get(1).unwrap().as_str().trim().to_string(), raw));
        if resume <= whole.start() {
            break;
        }
        at = resume;
    }
    out
}

/// Where the value that starts at `start` ends, and where scanning resumes.
///
/// The next `<parameter=` bounds the search, so a value whose closing tag was omitted ends there
/// rather than swallowing the next parameter. Inside that window the closing tag is the
/// `</parameter>` followed by another parameter or the end of the function, because a value may
/// itself contain the text `</parameter>`.
fn value_end(body: &str, start: usize) -> (usize, usize) {
    let after = AFTER_VALUE
        .get_or_init(|| Regex::new(r"^\s*(<parameter=|</function>|</tool_call>|$)").unwrap());
    let next = body[start..].find("<parameter=").map(|i| start + i);
    let limit = next.unwrap_or(body.len());
    let mut last = None;
    let mut from = start;
    while let Some(i) = body[from..limit].find(PARAM_END).map(|i| from + i) {
        last = Some(i);
        if after.is_match(&body[i + PARAM_END.len()..]) {
            return (i, i + PARAM_END.len());
        }
        from = i + 1;
    }
    match (last, next) {
        (Some(i), _) => (i, i + PARAM_END.len()),
        (None, Some(n)) => (n, n),
        (None, None) => (body.len(), body.len()),
    }
}

/// The JSON Schema type a tool declares for one of its parameters.
fn declared_type(tools: &[Tool], function: &str, key: &str) -> Option<String> {
    let tool = tools.iter().find(|t| t.function.name == function)?;
    let props = tool.function.parameters.as_ref()?.get("properties")?;
    props.get(key)?.get("type")?.as_str().map(str::to_lowercase)
}

/// A raw value, typed by what its parameter declares.
fn coerce(raw: &str, declared: Option<&str>) -> Value {
    if declared == Some("string") {
        return Value::String(raw.to_string());
    }
    let v = raw.trim();
    let Some(declared) = declared else {
        // No schema: take a JSON literal only when it is one. A string the model quoted is not
        // what this template writes, so the raw text is kept then.
        return match serde_json::from_str::<Value>(v) {
            Ok(Value::String(_)) | Err(_) => Value::String(raw.to_string()),
            Ok(parsed) => parsed,
        };
    };
    if v.eq_ignore_ascii_case("null") {
        return Value::Null;
    }
    match declared {
        "boolean" if v.eq_ignore_ascii_case("true") => return Value::Bool(true),
        "boolean" if v.eq_ignore_ascii_case("false") => return Value::Bool(false),
        "integer" => {
            if let Ok(n) = v.trim_start_matches('+').parse::<i64>() {
                return Value::from(n);
            }
        }
        "number" => {
            if let Ok(n) = v.trim_start_matches('+').parse::<f64>() {
                if let Some(n) = serde_json::Number::from_f64(n) {
                    return Value::Number(n);
                }
            }
        }
        _ => {}
    }
    serde_json::from_str::<Value>(v).unwrap_or_else(|_| Value::String(raw.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_llm_mcp::{Function, ToolType};
    use serde_json::json;

    fn tool(name: &str, properties: Value) -> Tool {
        let schema = json!({"type": "object", "properties": properties});
        Tool {
            tp: ToolType::Function,
            function: Function {
                name: name.into(),
                description: None,
                parameters: serde_json::from_value(schema).ok(),
                strict: None,
            },
        }
    }

    fn parse(message: &str, tools: &[Tool]) -> Value {
        let out = QwenParser
            .parse(message, tools)
            .unwrap()
            .expect("a tool call");
        serde_json::from_str(&out).unwrap()
    }

    #[test]
    fn xml_values_take_their_declared_types() {
        let tools = [tool(
            "set",
            json!({
                "n": {"type": "integer"}, "x": {"type": "number"}, "on": {"type": "boolean"},
                "tags": {"type": "array"}, "s": {"type": "string"}, "none": {"type": "integer"}
            }),
        )];
        let msg = "<tool_call>\n<function=set>\n<parameter=n>\n+3\n</parameter>\n\
                   <parameter=x>\n2.5\n</parameter>\n<parameter=on>\nTrue\n</parameter>\n\
                   <parameter=tags>\n[\"a\", 1]\n</parameter>\n<parameter=s>\n007\n</parameter>\n\
                   <parameter=none>\nnull\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(
            parse(msg, &tools),
            json!({"name": "set", "arguments": {
                "n": 3, "x": 2.5, "on": true, "tags": ["a", 1], "s": "007", "none": null
            }})
        );
    }

    #[test]
    fn a_string_is_verbatim_even_when_it_contains_the_closing_tag() {
        let tools = [tool(
            "write",
            json!({"code": {"type": "string"}, "path": {"type": "string"}}),
        )];
        let code = "if s.endswith(\"</parameter>\"):\n    pass\n";
        let msg = format!(
            "<tool_call>\n<function=write>\n<parameter=code>\n{code}\n</parameter>\n\
             <parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>"
        );
        assert_eq!(
            parse(&msg, &tools),
            json!({"name": "write", "arguments": {"code": code, "path": "a.py"}})
        );
    }

    #[test]
    fn a_missing_closing_tag_ends_at_the_next_parameter() {
        let tools = [tool(
            "f",
            json!({"a": {"type": "integer"}, "b": {"type": "integer"}}),
        )];
        let msg = "<tool_call>\n<function=f>\n<parameter=a>\n1\n<parameter=b>\n2\n</parameter>\n\
                   </function>\n</tool_call>";
        assert_eq!(
            parse(msg, &tools),
            json!({"name": "f", "arguments": {"a": 1, "b": 2}})
        );
    }

    #[test]
    fn without_a_schema_only_a_literal_is_parsed() {
        let msg = "<tool_call>\n<function=g>\n<parameter=obj>\n{\"k\": 1}\n</parameter>\n\
                   <parameter=word>\nhello\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(
            parse(msg, &[]),
            json!({"name": "g", "arguments": {"obj": {"k": 1}, "word": "hello"}})
        );
    }

    #[test]
    fn every_block_is_a_call() {
        let msg = "<tool_call>\n<function=a>\n</function>\n</tool_call>\n\
                   <tool_call>{\"name\": \"b\", \"arguments\": {\"x\": 1}}</tool_call>";
        assert_eq!(
            parse(msg, &[]),
            json!([{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}])
        );
    }

    #[test]
    fn json_bodies_still_parse() {
        let msg = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>";
        assert_eq!(
            parse(msg, &[]),
            json!({"name": "get_weather", "arguments": {"city": "Oslo"}})
        );
    }

    #[test]
    fn text_without_a_block_is_not_a_call() {
        assert!(QwenParser.parse("no tools here", &[]).unwrap().is_none());
    }
}
