//! Qwen3-Coder tool call parser, the format of the Qwen3.5 / 3.8 chat templates (vLLM's `qwen3_coder`).
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! A message may hold several `<tool_call>` blocks. Every value is text; the matcher gives it the type
//! its tool's schema declares.

use llguidance::api::TopLevelGrammar;
use regex::Regex;
use serde_json::{json, Map, Value};
use std::sync::OnceLock;

use super::ToolFormatParser;
use crate::Tool;

static CALL: OnceLock<Regex> = OnceLock::new();

pub struct Qwen3CoderParser;

/// A value without the one newline that the template puts on each side of it.
fn value(raw: &str) -> &str {
    let raw = raw.strip_prefix('\n').unwrap_or(raw);
    raw.strip_suffix('\n').unwrap_or(raw)
}

impl ToolFormatParser for Qwen3CoderParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Qwen3Coder
    }

    /// The format is XML, not JSON: nothing constrains it, and the matcher never asks for a grammar
    /// when the template says the model writes it.
    fn tool_call_grammar(&self, _tools: &[Tool], _text: &str) -> TopLevelGrammar {
        unreachable!("the Qwen3-Coder format has no grammar")
    }

    /// `<function=…>` closed by `</function>`; a `</tool_call>` the model did not reach (it stopped on
    /// the end of its turn first) still closes the call. A parameter runs to its `</parameter>`, or to
    /// where the next one starts or the function ends.
    fn parse(&self, message: &str) -> hanzo_ml::Result<Option<String>> {
        let call = CALL.get_or_init(|| {
            Regex::new(r"(?s)<function=([^>\s]+)>(.*?)(?:</function>|</tool_call>|$)").unwrap()
        });
        let calls: Vec<Value> = message
            .split("<tool_call>")
            .skip(1)
            .filter_map(|block| call.captures(block))
            .map(|c| {
                let mut arguments = Map::new();
                for segment in c[2].split("<parameter=").skip(1) {
                    let Some((key, rest)) = segment.split_once('>') else {
                        continue;
                    };
                    let rest = rest.split("</parameter>").next().unwrap_or(rest);
                    arguments.insert(
                        key.trim().to_string(),
                        Value::String(value(rest).to_string()),
                    );
                }
                json!({"name": &c[1], "arguments": arguments})
            })
            .collect();
        if calls.is_empty() {
            Ok(None)
        } else {
            Ok(Some(
                serde_json::to_string(&calls).map_err(hanzo_ml::Error::wrap)?,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(message: &str) -> Option<Value> {
        Qwen3CoderParser
            .parse(message)
            .unwrap()
            .map(|s| serde_json::from_str(&s).unwrap())
    }

    #[test]
    fn one_call_with_two_parameters() {
        let got = parse(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
        )
        .unwrap();
        assert_eq!(
            got,
            json!([{"name": "get_weather", "arguments": {"city": "Paris", "days": "3"}}])
        );
    }

    #[test]
    fn several_calls_keep_their_order() {
        let got = parse(
            "I will look.\n<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
        )
        .unwrap();
        assert_eq!(
            got,
            json!([{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": "1"}}])
        );
    }

    #[test]
    fn a_multiline_value_keeps_its_inner_newlines_and_markup() {
        let got = parse(
            "<tool_call>\n<function=write>\n<parameter=text>\nline one\nline <b>two</b>\n</parameter>\n</function>\n</tool_call>",
        )
        .unwrap();
        assert_eq!(got[0]["arguments"]["text"], "line one\nline <b>two</b>");
    }

    #[test]
    fn a_turn_that_ends_before_the_closing_tags_still_parses() {
        let got = parse("<tool_call>\n<function=f>\n<parameter=k>\nv\n</parameter>\n</function>")
            .unwrap();
        assert_eq!(got, json!([{"name": "f", "arguments": {"k": "v"}}]));
    }

    #[test]
    fn text_without_a_call_is_not_one() {
        assert!(parse("no tools here").is_none());
        assert!(parse("<tool_call>{\"name\": \"f\"}</tool_call>").is_none());
    }
}
