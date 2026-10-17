//! Model-specific tool call format parsers.
//!
//! Each model family emits tool calls in a different format.  Parsers are
//! registered in [`PARSERS`] and tried in order — the first match wins.

mod deepseek;
mod gemma4;
mod gemma4_strict;
pub(crate) mod harmony;
mod llama;
mod mistral_nemo;
mod qwen;

use hanzo_ml::Result;
use llguidance::api::TopLevelGrammar;

use crate::Tool;

/// Identifies the detected tool call format so that the correct grammar
/// can be constructed for mid-stream constrained decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// `<tool_call>{"name":"...","arguments":{...}}</tool_call>`, or the same block holding
    /// `<function=NAME><parameter=KEY>VALUE</parameter></function>`
    Qwen,
    /// `<|python_tag|>{"name":"...","parameters":{...}}`
    Llama,
    /// `[TOOL_CALLS][{"name":"...","arguments":{...}}]`
    MistralNemo,
    /// Multi-line with ` ```json` fence and `<｜tool▁call▁end｜>` delimiter
    DeepSeek,
    /// `<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>` (non-JSON)
    Gemma4,
}

/// A parser that can detect and extract tool calls from model output.
pub trait ToolFormatParser: Send + Sync {
    /// Returns `true` if `text` contains a token sequence that could be (or
    /// is) a tool call in this format.  Used during streaming to decide
    /// whether to buffer output.
    fn could_be_tool_call(&self, text: &str) -> bool;

    /// Attempt to parse complete tool calls from `message`.
    ///
    /// Returns `Ok(Some(json))` if tool calls were found and parsed (the
    /// JSON is a serialised `Vec<CalledFunctionParameters>` or a single
    /// `CalledFunctionParameters`).
    ///
    /// Returns `Ok(None)` if this parser doesn't match the message format,
    /// OR if the tool call is incomplete (still being generated).
    ///
    /// The caller will fall through to the next parser on `Ok(None)`.
    ///
    /// `tools` are the request's tools, for formats whose arguments are untyped text that only the
    /// declared schema can type.
    fn parse(&self, message: &str, tools: &[Tool]) -> Result<Option<String>>;

    /// The tool call format this parser handles.
    fn format(&self) -> ToolCallFormat;

    /// The text of `message` outside the calls [`parse`](Self::parse) read from it. A format whose
    /// calls are the whole message has none.
    fn outside(&self, _message: &str) -> String {
        String::new()
    }

    /// Build an llguidance grammar that constrains model output to a valid
    /// tool call in this format.  The grammar covers the **post-prefix**
    /// content (the prefix itself is already generated when grammar
    /// activation occurs).
    ///
    /// `text` is the full generated text up to the point of grammar
    /// activation — parsers like DeepSeek can use it to extract the tool
    /// name from the prefix.
    fn tool_call_grammar(&self, tools: &[Tool], text: &str) -> TopLevelGrammar;
}

/// Static registry of all supported tool-call format parsers, tried in order.
///
/// **Order matters**: Gemma 4 must come before Qwen because `<|tool_call>`
/// contains `<tool_call>` as a substring.
static PARSERS: std::sync::LazyLock<Vec<Box<dyn ToolFormatParser>>> =
    std::sync::LazyLock::new(|| {
        vec![
            Box::new(gemma4::Gemma4Parser),
            Box::new(llama::LlamaParser),
            Box::new(qwen::QwenParser),
            Box::new(mistral_nemo::MistralNemoParser),
            Box::new(deepseek::DeepSeekParser),
        ]
    });

/// Check whether `text` could contain a tool call prefix from any supported
/// format.
pub fn contains_tool_call_prefix(text: &str) -> bool {
    PARSERS.iter().any(|p| p.could_be_tool_call(text))
}

/// Build a tool call grammar for the first matching format detected in
/// `text`.  Returns `None` if no format matches or if the format is not
/// yet ready for grammar activation (e.g. DeepSeek before the JSON fence).
pub fn build_tool_call_grammar(text: &str, tools: &[Tool]) -> Option<TopLevelGrammar> {
    for parser in PARSERS.iter() {
        if parser.could_be_tool_call(text) {
            // DeepSeek: wait until the JSON fence is present so we don't
            // activate grammar before the function name is complete.
            if parser.format() == ToolCallFormat::DeepSeek && !text.contains("```json\n") {
                return None;
            }
            return Some(parser.tool_call_grammar(tools, text));
        }
    }
    None
}

/// The text of `message` outside its tool calls, per the first parser that reads calls in it, as
/// [`process_model_specific_message`] picks it. Empty when no parser does.
pub fn outside_calls(message: &str, tools: &[Tool]) -> Result<String> {
    for parser in PARSERS.iter() {
        if parser.parse(message, tools)?.is_some() {
            return Ok(parser.outside(message));
        }
    }
    Ok(String::new())
}

/// Try each parser in order to extract tool calls from `message`.
/// Returns the original message unchanged if no parser matches.
pub fn process_model_specific_message(message: &str, tools: &[Tool]) -> Result<String> {
    for parser in PARSERS.iter() {
        if let Some(json) = parser.parse(message, tools)? {
            return Ok(json);
        }
    }
    Ok(message.to_string())
}
