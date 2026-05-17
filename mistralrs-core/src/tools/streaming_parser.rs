//! Incremental parser for tool-call argument deltas during streaming.
//!
//! The non-streaming tool parser (`mod.rs::ToolCallingMatcher`) only succeeds
//! once the entire tool-call JSON is buffered. That forces a streaming server
//! to withhold every byte until the tool block closes, then emit one giant
//! `tool_calls` chunk. OpenAI streaming clients (Codex CLI, opencode,
//! LiteLLM, the OpenAI SDK, Claude Code over the OAI bridge) instead expect:
//!
//! 1. A "header" chunk per tool index carrying `id` + `function.name` and an
//!    empty `function.arguments`.
//! 2. One or more "argument" chunks carrying just incremental fragments of
//!    `function.arguments` (concatenated client-side to form the final JSON).
//!
//! This module recognizes the common tool-call surface formats emitted by
//! mistral.rs's supported model families and produces incremental
//! `(header | argument | done)` events as new completion bytes arrive.
//!
//! Formats supported:
//! - Bare JSON object: `{"name": "f", "arguments": {...}}` /
//!   `{"name": "f", "parameters": {...}}`
//! - Bare JSON array of the same: `[{...}, {...}]`
//! - Qwen / Hermes style: `<tool_call>{json}</tool_call>` (one or more,
//!   possibly separated by whitespace)
//! - Llama 3.1+ Python tag: `<|python_tag|>{json}` (single object)
//! - Mistral Nemo: `[TOOL_CALLS][{json}, ...]`
//! - DeepSeek R1/V3:
//!   `<｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\n```json\n{json}\n```<｜tool▁call▁end｜>`
//!
//! The parser is byte-pull: caller feeds the full accumulated completion text
//! after each new token, and the parser keeps track of how much it has
//! already emitted.

use uuid::Uuid;

/// Marker tokens for the various tool-call formats.
const QWEN_OPEN: &str = "<tool_call>";
const QWEN_CLOSE: &str = "</tool_call>";
const LLAMA_PYTHON_TAG: &str = "<|python_tag|>";
const MISTRAL_TOOL_TAG: &str = "[TOOL_CALLS]";
const DEEPSEEK_BEGIN: &str = "<\u{ff5c}tool\u{2581}call\u{2581}begin\u{ff5c}>";
const DEEPSEEK_SEP: &str = "<\u{ff5c}tool\u{2581}sep\u{ff5c}>";
const DEEPSEEK_END: &str = "<\u{ff5c}tool\u{2581}call\u{2581}end\u{ff5c}>";

/// An event produced by the streaming tool parser.
///
/// Mirrors the OpenAI streaming-delta shape: `Header` is the first chunk for
/// a given `index` (carries `id` and `name`); `Args` is a continuation chunk
/// carrying additional bytes for the same `index`'s `function.arguments`
/// value; `Done` marks the end of a tool block (caller may close the block
/// or move on to the next).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolStreamEvent {
    Header {
        index: usize,
        id: String,
        name: String,
    },
    Args {
        index: usize,
        fragment: String,
    },
    Done {
        index: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    /// Have not yet committed to a tool format. Buffer everything; once we
    /// can prove this is a tool call (recognized prefix) we transition.
    Probing,
    /// Inside a tool block but still scanning for the `"name":"..."` value.
    SeekName { current_index: usize },
    /// Have emitted Header for the current tool; now streaming bytes of the
    /// arguments / parameters JSON value through to its matching close.
    StreamingArgs {
        current_index: usize,
        /// Brace/bracket depth of the args value; 0 when not yet inside (we
        /// emit the opening character of the value once and start at depth 1).
        depth: i32,
        /// Whether we are currently inside a JSON string literal in the args.
        in_string: bool,
        /// Whether the previous char was a backslash (for escape handling).
        escape: bool,
        /// Whether the args value is a primitive (string / number / bool /
        /// null). If true, terminates at the next `,` or container close at
        /// the parent level.
        primitive: bool,
        /// True if the args value is a JSON string (so we know `"` terminates,
        /// not just delimits an inner string).
        string_value: bool,
    },
    /// Scanning for the next tool call in this same response (multi-tool
    /// cases like a JSON array `[{...},{...}]` or repeated `<tool_call>`s).
    BetweenCalls { next_index: usize },
    /// No further parsing — either we proved this is not a tool call, or we
    /// finished the outermost group.
    Inert,
}

/// Detected tool-call surface format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    /// Bare JSON object or whitespace-separated JSON objects.
    BareJsonObject,
    /// Bare JSON array.
    BareJsonArray,
    /// `<tool_call>...</tool_call>` blocks.
    Qwen,
    /// `<|python_tag|>{json}` (single).
    LlamaPython,
    /// `[TOOL_CALLS][{json},...]` array.
    MistralNemo,
    /// DeepSeek tool-call begin/end markers.
    DeepSeek,
}

/// Incremental tool-call argument-delta parser.
///
/// Feed the *full accumulated completion text so far* via [`Self::feed`]
/// after each new token; drain emitted events via [`Self::take_events`].
pub struct StreamingToolCallParser {
    /// Full text observed so far (lives in caller's buffer; we only keep an
    /// offset into the slice supplied to `feed`).
    consumed: usize,
    /// Buffered tail (text between `consumed` and the end of the last feed)
    /// that the parser has not yet acted upon — typically because it might
    /// be the start of a marker spanning a token boundary.
    pending: String,
    state: State,
    format: Option<Format>,
    /// Pending events to drain.
    events: Vec<ToolStreamEvent>,
}

impl Default for StreamingToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingToolCallParser {
    pub fn new() -> Self {
        Self {
            consumed: 0,
            pending: String::new(),
            state: State::Probing,
            format: None,
            events: Vec::new(),
        }
    }

    /// Has this parser proven that the input *is* a tool call (i.e., emitted
    /// at least the format-recognition transition)?
    pub fn committed(&self) -> bool {
        self.format.is_some()
    }

    /// Drain emitted events.
    pub fn take_events(&mut self) -> Vec<ToolStreamEvent> {
        std::mem::take(&mut self.events)
    }

    /// Feed the *full* accumulated completion text. The parser tracks how
    /// much it has already consumed internally.
    pub fn feed(&mut self, full: &str) {
        if full.len() < self.consumed {
            // Caller reset; restart.
            self.consumed = 0;
            self.pending.clear();
            self.state = State::Probing;
            self.format = None;
            self.events.clear();
        }
        let new_bytes = &full[self.consumed..];
        self.consumed = full.len();
        self.pending.push_str(new_bytes);
        self.advance();
    }

    /// Mark end of input. Causes any open tool block to emit its `Done`.
    pub fn finalize(&mut self) {
        self.advance();
        match self.state {
            State::StreamingArgs { current_index, .. } | State::SeekName { current_index } => {
                self.events.push(ToolStreamEvent::Done {
                    index: current_index,
                });
            }
            _ => {}
        }
        self.state = State::Inert;
    }

    fn advance(&mut self) {
        loop {
            let before = (self.state.clone(), self.pending.len());
            self.step();
            let after = (self.state.clone(), self.pending.len());
            if before == after {
                break;
            }
        }
    }

    fn step(&mut self) {
        match self.state.clone() {
            State::Inert => {}
            State::Probing => self.probe(),
            State::SeekName { current_index } => self.seek_name(current_index),
            State::StreamingArgs {
                current_index,
                depth,
                in_string,
                escape,
                primitive,
                string_value,
            } => self.stream_args(
                current_index,
                depth,
                in_string,
                escape,
                primitive,
                string_value,
            ),
            State::BetweenCalls { next_index } => self.between(next_index),
        }
    }

    /// Decide which surface format we're looking at, or stay Probing if
    /// there isn't enough information yet.
    fn probe(&mut self) {
        let trimmed = self.pending.trim_start();
        let lead_trim = self.pending.len() - trimmed.len();

        // Recognize known wrapper prefixes. We require the *full* marker
        // before committing, so we don't false-trip on a partial token.
        if let Some(rest) = trimmed.strip_prefix(LLAMA_PYTHON_TAG) {
            self.format = Some(Format::LlamaPython);
            let advance = lead_trim + LLAMA_PYTHON_TAG.len();
            self.pending.drain(..advance);
            self.state = State::SeekName { current_index: 0 };
            return;
        }
        if let Some(rest) = trimmed.strip_prefix(MISTRAL_TOOL_TAG) {
            self.format = Some(Format::MistralNemo);
            let advance = lead_trim + MISTRAL_TOOL_TAG.len();
            self.pending.drain(..advance);
            self.state = State::BetweenCalls { next_index: 0 };
            return;
        }
        if let Some(rest) = trimmed.strip_prefix(QWEN_OPEN) {
            self.format = Some(Format::Qwen);
            let advance = lead_trim + QWEN_OPEN.len();
            self.pending.drain(..advance);
            self.state = State::SeekName { current_index: 0 };
            return;
        }
        if let Some(_rest) = trimmed.strip_prefix(DEEPSEEK_BEGIN) {
            self.format = Some(Format::DeepSeek);
            let advance = lead_trim + DEEPSEEK_BEGIN.len();
            self.pending.drain(..advance);
            self.state = State::SeekName { current_index: 0 };
            return;
        }
        // Bare JSON object or array? We need to peek at the first
        // non-whitespace char, but only commit once we have a recognizable
        // tool key (`"name"`) in the buffer to avoid false-tripping on
        // ordinary code-fenced JSON. We treat a bare `{` followed by
        // `"name"` early as the BareJsonObject format.
        if let Some(first) = trimmed.chars().next() {
            match first {
                '{' => {
                    // Wait until we see `"name"` somewhere in the object before
                    // committing — that keeps us out of false-tripping on
                    // ordinary JSON the model might emit unrelated to tools.
                    if trimmed.contains("\"name\"") || trimmed.contains("\"function\"") {
                        self.format = Some(Format::BareJsonObject);
                        self.pending.drain(..lead_trim + 1);
                        self.state = State::SeekName { current_index: 0 };
                    }
                    return;
                }
                '[' => {
                    if trimmed.contains("\"name\"") || trimmed.contains("\"function\"") {
                        self.format = Some(Format::BareJsonArray);
                        self.pending.drain(..lead_trim + 1);
                        self.state = State::BetweenCalls { next_index: 0 };
                    }
                    return;
                }
                _ => {
                    // Anything else: this is not a tool call we recognize.
                    // We can't conclude yet though — a marker might still
                    // arrive across a future token. Only give up if we see
                    // a substantial chunk of non-tool content.
                    if self.pending.len() > 4096 {
                        self.state = State::Inert;
                    }
                    return;
                }
            }
        }
    }

    fn between(&mut self, next_index: usize) {
        // Skip whitespace and structural separators (`,` between array
        // elements, `]` to end the outer array).
        let mut i = 0;
        let bytes = self.pending.as_bytes();
        while i < bytes.len() {
            let c = bytes[i] as char;
            if c.is_whitespace() || c == ',' {
                i += 1;
                continue;
            }
            break;
        }
        if i > 0 {
            self.pending.drain(..i);
        }
        if self.pending.is_empty() {
            return;
        }
        let first = self.pending.as_bytes()[0] as char;
        // Outer array close for BareJsonArray / MistralNemo
        if first == ']' {
            self.pending.drain(..1);
            self.state = State::Inert;
            return;
        }
        // Qwen: a fresh `<tool_call>` may appear (multi-tool stream).
        if matches!(self.format, Some(Format::Qwen))
            && self.pending.starts_with(QWEN_OPEN)
        {
            self.pending.drain(..QWEN_OPEN.len());
            self.state = State::SeekName {
                current_index: next_index,
            };
            return;
        }
        // DeepSeek: a fresh begin marker.
        if matches!(self.format, Some(Format::DeepSeek))
            && self.pending.starts_with(DEEPSEEK_BEGIN)
        {
            self.pending.drain(..DEEPSEEK_BEGIN.len());
            self.state = State::SeekName {
                current_index: next_index,
            };
            return;
        }
        // Otherwise an object: bare `{` opens the next entry.
        if first == '{' {
            self.pending.drain(..1);
            self.state = State::SeekName {
                current_index: next_index,
            };
        } else {
            // Unknown structure; stop trying.
            self.state = State::Inert;
        }
    }

    /// Within an object, look for `"name": "..."` so we can emit a Header,
    /// then locate the `"arguments":` or `"parameters":` value and enter
    /// StreamingArgs at the first character of the value.
    fn seek_name(&mut self, current_index: usize) {
        // DeepSeek has a different name-extraction path: after the begin
        // marker, the format is `function<|tool_sep|>NAME\n```json\n{args}\n```...`.
        if matches!(self.format, Some(Format::DeepSeek)) {
            if let Some(sep_pos) = self.pending.find(DEEPSEEK_SEP) {
                // Find the newline that terminates the NAME.
                let after_sep = sep_pos + DEEPSEEK_SEP.len();
                if let Some(nl) = self.pending[after_sep..].find('\n') {
                    let name = self.pending[after_sep..after_sep + nl].trim().to_string();
                    // Locate the ```json fence opening.
                    let after_name = after_sep + nl + 1;
                    if let Some(fence) = self.pending[after_name..].find("```json") {
                        let after_fence = after_name + fence + "```json".len();
                        // Skip an optional newline.
                        let after_fence = if self.pending.as_bytes().get(after_fence)
                            == Some(&b'\n')
                        {
                            after_fence + 1
                        } else {
                            after_fence
                        };
                        // Emit Header (synth id since DeepSeek format itself
                        // carries no per-call id).
                        let id = format!("call_{}", Uuid::new_v4().simple());
                        self.events.push(ToolStreamEvent::Header {
                            index: current_index,
                            id,
                            name,
                        });
                        // Drop the prefix; remainder is the args JSON object
                        // followed by ``` and the end marker.
                        self.pending.drain(..after_fence);
                        // First char of args (must be `{` for valid input).
                        // Stream the body up to the closing ``` (we'll then
                        // also have to consume the end marker in BetweenCalls).
                        // Emit the opening brace too as part of the args
                        // fragment so client receives valid JSON.
                        let Some((opening, primitive, string_value)) =
                            classify_value_start(&self.pending)
                        else {
                            return;
                        };
                        let _ = string_value;
                        // Emit the first char of the value.
                        let first_ch = self.pending.as_bytes()[0] as char;
                        self.events.push(ToolStreamEvent::Args {
                            index: current_index,
                            fragment: first_ch.to_string(),
                        });
                        self.pending.drain(..1);
                        let initial_depth = if primitive { 0 } else { 1 };
                        self.state = State::StreamingArgs {
                            current_index,
                            depth: initial_depth,
                            in_string: matches!(opening, '"'),
                            escape: false,
                            primitive,
                            string_value,
                        };
                        return;
                    }
                }
            }
            // Need more bytes.
            return;
        }

        // Generic JSON path: find `"name"` then `:` then the string value.
        let (name, args_value_start) = match locate_name_and_args(&self.pending) {
            Some(x) => x,
            None => return,
        };
        let id = format!("call_{}", Uuid::new_v4().simple());
        self.events.push(ToolStreamEvent::Header {
            index: current_index,
            id,
            name,
        });
        // Drop everything up to the first byte of the args value.
        self.pending.drain(..args_value_start);
        // Identify the value's first character to set up depth tracking.
        let (opening, primitive, string_value) = match classify_value_start(&self.pending) {
            Some((c, p, s)) => (Some(c), p, s),
            None => return, // wait for more bytes
        };
        let _ = opening;
        // Emit the first character as part of the args fragment.
        let first_ch = self.pending.as_bytes()[0] as char;
        self.events.push(ToolStreamEvent::Args {
            index: current_index,
            fragment: first_ch.to_string(),
        });
        self.pending.drain(..1);
        let initial_depth = if primitive { 0 } else { 1 };
        self.state = State::StreamingArgs {
            current_index,
            depth: initial_depth,
            in_string: string_value,
            escape: false,
            primitive,
            string_value,
        };
    }

    /// Stream argument bytes character-by-character (well, fragment-by-
    /// fragment from `pending`) until the args value closes.
    fn stream_args(
        &mut self,
        current_index: usize,
        mut depth: i32,
        mut in_string: bool,
        mut escape: bool,
        primitive: bool,
        string_value: bool,
    ) {
        if self.pending.is_empty() {
            return;
        }
        let bytes = self.pending.as_bytes().to_vec();
        let mut emit_up_to: Option<usize> = None; // exclusive — terminator NOT emitted
        let mut terminator_consumed: usize = 0; // bytes to drop after emit

        for (i, &b) in bytes.iter().enumerate() {
            let c = b as char;
            if escape {
                escape = false;
                continue;
            }
            if in_string {
                match c {
                    '\\' => {
                        escape = true;
                    }
                    '"' => {
                        in_string = false;
                        // For a string-typed args value at depth 0, the
                        // closing quote IS the terminator and we include it
                        // in the args (so client gets a valid JSON string).
                        if string_value && depth == 0 {
                            emit_up_to = Some(i + 1);
                            terminator_consumed = 0;
                            break;
                        }
                    }
                    _ => {}
                }
                continue;
            }
            match c {
                '"' => in_string = true,
                '{' | '[' => depth += 1,
                '}' | ']' => {
                    depth -= 1;
                    if depth == 0 && !primitive {
                        // Include the closing bracket; that's the end of args.
                        emit_up_to = Some(i + 1);
                        terminator_consumed = 0;
                        break;
                    }
                }
                ',' => {
                    if primitive && depth == 0 {
                        // primitive value terminated by comma (not emitted).
                        emit_up_to = Some(i);
                        terminator_consumed = 0;
                        break;
                    }
                }
                _ => {}
            }
        }
        match emit_up_to {
            Some(n) => {
                if n > 0 {
                    let frag = self.pending[..n].to_string();
                    self.events.push(ToolStreamEvent::Args {
                        index: current_index,
                        fragment: frag,
                    });
                }
                let drop = n + terminator_consumed;
                self.pending.drain(..drop);
                self.events.push(ToolStreamEvent::Done {
                    index: current_index,
                });
                // Move on to outer-format-aware "between" state.
                self.state = match self.format {
                    Some(Format::BareJsonArray) | Some(Format::MistralNemo) => {
                        State::BetweenCalls {
                            next_index: current_index + 1,
                        }
                    }
                    Some(Format::Qwen) => {
                        // Consume optional whitespace then `</tool_call>`.
                        // Look ahead in pending.
                        let after_ws = self
                            .pending
                            .trim_start_matches(char::is_whitespace);
                        let drop = self.pending.len() - after_ws.len();
                        self.pending.drain(..drop);
                        if self.pending.starts_with(QWEN_CLOSE) {
                            self.pending.drain(..QWEN_CLOSE.len());
                        }
                        State::BetweenCalls {
                            next_index: current_index + 1,
                        }
                    }
                    Some(Format::DeepSeek) => {
                        // Consume `\n```` then end marker.
                        let after = self.pending.trim_start_matches(|c: char| c.is_whitespace());
                        let drop = self.pending.len() - after.len();
                        self.pending.drain(..drop);
                        if self.pending.starts_with("```") {
                            self.pending.drain(.."```".len());
                        }
                        let after = self.pending.trim_start_matches(|c: char| c.is_whitespace());
                        let drop = self.pending.len() - after.len();
                        self.pending.drain(..drop);
                        if self.pending.starts_with(DEEPSEEK_END) {
                            self.pending.drain(..DEEPSEEK_END.len());
                        }
                        State::BetweenCalls {
                            next_index: current_index + 1,
                        }
                    }
                    _ => State::Inert,
                };
            }
            None => {
                // Emit everything seen so far as an args fragment, but for
                // safety stop just before any trailing partial escape (so
                // we don't split a JSON escape across two chunks in a way
                // that could confuse some clients). Simpler: emit all.
                let frag = self.pending.clone();
                if !frag.is_empty() {
                    self.events.push(ToolStreamEvent::Args {
                        index: current_index,
                        fragment: frag,
                    });
                }
                self.pending.clear();
                self.state = State::StreamingArgs {
                    current_index,
                    depth,
                    in_string,
                    escape,
                    primitive,
                    string_value,
                };
            }
        }
    }
}

/// Locate the `"name": "..."` pair and the start of the `"arguments":` or
/// `"parameters":` value in a JSON object body (we've already consumed the
/// opening `{`).
///
/// Returns `(name, args_value_start_byte)` if both are found; `None` if we
/// need more bytes. Returns `None` only if the input does not yet contain a
/// complete `name` field + an args/parameters key + `:`.
fn locate_name_and_args(s: &str) -> Option<(String, usize)> {
    // We scan tokens. The structure is JSON-object-body.
    // A robust approach: find `"name"` followed by `:` then a string value;
    // find `"arguments"` or `"parameters"` followed by `:` then take the
    // byte offset right after the colon (and any whitespace).
    let mut name: Option<String> = None;
    let mut args_start: Option<usize> = None;
    let mut i = 0;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        // Skip whitespace/commas/colons at top level (we re-handle below).
        let c = bytes[i] as char;
        if c.is_whitespace() || c == ',' {
            i += 1;
            continue;
        }
        if c == '"' {
            // Read a JSON string starting at i.
            let (key, key_end) = match read_json_string(s, i) {
                Some(x) => x,
                None => return None, // incomplete
            };
            // Skip whitespace and the colon.
            let mut j = key_end;
            while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                j += 1;
            }
            if j >= bytes.len() {
                return None;
            }
            if bytes[j] != b':' {
                // Malformed; stop scanning.
                return None;
            }
            j += 1;
            while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                j += 1;
            }
            if j >= bytes.len() {
                // Need more bytes (we have the key + colon but no value yet).
                return None;
            }
            match key.as_str() {
                "name" => {
                    // Value must be a string.
                    if bytes[j] != b'"' {
                        return None;
                    }
                    let (val, val_end) = match read_json_string(s, j) {
                        Some(x) => x,
                        None => return None,
                    };
                    name = Some(val);
                    i = val_end;
                }
                "arguments" | "parameters" => {
                    args_start = Some(j);
                    // Advance past the value entirely so we keep scanning for
                    // `name` if it wasn't seen yet.
                    let val_end = match skip_json_value(s, j) {
                        Some(end) => end,
                        None => {
                            // We have a value start but not a complete value;
                            // that's fine — args_start is what callers want.
                            // If name is also known, return.
                            if name.is_some() {
                                return name.map(|n| (n, args_start.unwrap()));
                            }
                            // Otherwise we can't know if `name` comes after.
                            return None;
                        }
                    };
                    i = val_end;
                }
                "function" => {
                    // Nested: `"function": {"name":..., "arguments":...}`
                    // Recurse into the nested object body.
                    if bytes[j] != b'{' {
                        return None;
                    }
                    let inner_start = j + 1;
                    // Find matching close to constrain recursion.
                    let inner_end = match find_matching_close(s, j) {
                        Some(end) => end,
                        None => {
                            // Try to recover by scanning what's available.
                            // Take the rest as the inner body.
                            return locate_name_and_args(&s[inner_start..])
                                .map(|(n, off)| (n, inner_start + off));
                        }
                    };
                    let inner_body = &s[inner_start..inner_end];
                    if let Some((n, off)) = locate_name_and_args(inner_body) {
                        return Some((n, inner_start + off));
                    }
                    i = inner_end + 1;
                }
                _ => {
                    // Unknown key — skip its value.
                    let val_end = match skip_json_value(s, j) {
                        Some(end) => end,
                        None => return None,
                    };
                    i = val_end;
                }
            }
            if let (Some(ref n), Some(off)) = (&name, args_start) {
                return Some((n.clone(), off));
            }
            continue;
        }
        // Anything else at this position is malformed for an object body
        // (e.g. a stray `}` would mean an empty object — no name).
        if c == '}' {
            return None;
        }
        i += 1;
    }
    None
}

/// Read a JSON string starting at byte offset `start` (which must point to
/// the opening quote). Returns `(decoded_value, end_offset_exclusive)`
/// where `end_offset_exclusive` is just after the closing quote.
fn read_json_string(s: &str, start: usize) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    debug_assert_eq!(bytes[start], b'"');
    let mut i = start + 1;
    let mut out = String::new();
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                if i + 1 >= bytes.len() {
                    return None;
                }
                let esc = bytes[i + 1];
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'u' => {
                        if i + 5 >= bytes.len() {
                            return None;
                        }
                        let hex = std::str::from_utf8(&bytes[i + 2..i + 6]).ok()?;
                        let code = u32::from_str_radix(hex, 16).ok()?;
                        if let Some(ch) = char::from_u32(code) {
                            out.push(ch);
                        }
                        i += 4;
                    }
                    _ => out.push(esc as char),
                }
                i += 2;
            }
            b'"' => return Some((out, i + 1)),
            other => {
                out.push(other as char);
                i += 1;
            }
        }
    }
    None
}

/// Skip a JSON value starting at byte offset `start`. Returns the byte
/// offset just past the value, or `None` if the value is incomplete.
fn skip_json_value(s: &str, start: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    if start >= bytes.len() {
        return None;
    }
    let c = bytes[start];
    match c {
        b'"' => read_json_string(s, start).map(|(_, end)| end),
        b'{' | b'[' => find_matching_close(s, start).map(|e| e + 1),
        b't' => {
            // true
            if start + 4 <= bytes.len() && &bytes[start..start + 4] == b"true" {
                Some(start + 4)
            } else {
                None
            }
        }
        b'f' => {
            if start + 5 <= bytes.len() && &bytes[start..start + 5] == b"false" {
                Some(start + 5)
            } else {
                None
            }
        }
        b'n' => {
            if start + 4 <= bytes.len() && &bytes[start..start + 4] == b"null" {
                Some(start + 4)
            } else {
                None
            }
        }
        b'-' | b'0'..=b'9' => {
            let mut j = start + 1;
            while j < bytes.len() {
                let cj = bytes[j];
                if cj.is_ascii_digit() || matches!(cj, b'.' | b'e' | b'E' | b'+' | b'-') {
                    j += 1;
                } else {
                    break;
                }
            }
            if j == bytes.len() {
                None // could still be growing
            } else {
                Some(j)
            }
        }
        _ => None,
    }
}

/// Given an opening `{` or `[` at byte offset `start`, find the matching
/// close. Returns its byte offset, or `None` if incomplete.
fn find_matching_close(s: &str, start: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    if start >= bytes.len() {
        return None;
    }
    let open = bytes[start];
    let close = match open {
        b'{' => b'}',
        b'[' => b']',
        _ => return None,
    };
    let mut depth: i32 = 1;
    let mut i = start + 1;
    let mut in_string = false;
    let mut escape = false;
    while i < bytes.len() {
        let b = bytes[i];
        if escape {
            escape = false;
            i += 1;
            continue;
        }
        if in_string {
            match b {
                b'\\' => escape = true,
                b'"' => in_string = false,
                _ => {}
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
        } else if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Classify the first character of a JSON value: returns `(opening_char,
/// is_primitive, is_string)`. Returns `None` if the buffer is empty.
fn classify_value_start(s: &str) -> Option<(char, bool, bool)> {
    let first = s.chars().next()?;
    let (primitive, string) = match first {
        '{' | '[' => (false, false),
        '"' => (true, true), // primitive in the sense of "depth 0 terminates on closing quote"
        _ => (true, false),  // number / true / false / null
    };
    // For strings we initialize depth=0 and in_string=true so that the
    // closing `"` is treated as the terminator; we represent that as
    // "primitive=true, string_value=true" in `StreamingArgs`.
    Some((first, primitive, string))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_events(p: &mut StreamingToolCallParser, full: &str) -> Vec<ToolStreamEvent> {
        p.feed(full);
        p.take_events()
    }

    fn run(full: &str) -> Vec<ToolStreamEvent> {
        let mut p = StreamingToolCallParser::new();
        // Feed in 4-char chunks to exercise incremental parsing.
        let mut acc = String::new();
        let mut out = Vec::new();
        let chars: Vec<char> = full.chars().collect();
        for chunk in chars.chunks(4) {
            for c in chunk {
                acc.push(*c);
            }
            out.extend(collect_events(&mut p, &acc));
        }
        p.finalize();
        out.extend(p.take_events());
        out
    }

    fn header_args_done(events: &[ToolStreamEvent]) -> (Option<String>, String, bool) {
        let mut name = None;
        let mut args = String::new();
        let mut saw_done = false;
        for e in events {
            match e {
                ToolStreamEvent::Header { name: n, .. } => name = Some(n.clone()),
                ToolStreamEvent::Args { fragment, .. } => args.push_str(fragment),
                ToolStreamEvent::Done { .. } => saw_done = true,
            }
        }
        (name, args, saw_done)
    }

    #[test]
    fn bare_json_object_object_args() {
        let evs = run(r#"{"name":"add","arguments":{"x":1,"y":2}}"#);
        let (name, args, done) = header_args_done(&evs);
        assert_eq!(name.as_deref(), Some("add"));
        assert_eq!(args, r#"{"x":1,"y":2}"#);
        assert!(done);
    }

    #[test]
    fn bare_json_object_parameters_alias() {
        let evs = run(r#"{"name":"f","parameters":{"a":"b"}}"#);
        let (name, args, done) = header_args_done(&evs);
        assert_eq!(name.as_deref(), Some("f"));
        assert_eq!(args, r#"{"a":"b"}"#);
        assert!(done);
    }

    #[test]
    fn qwen_format() {
        let evs = run(r#"<tool_call>{"name":"f","arguments":{"k":"v"}}</tool_call>"#);
        let (name, args, done) = header_args_done(&evs);
        assert_eq!(name.as_deref(), Some("f"));
        assert_eq!(args, r#"{"k":"v"}"#);
        assert!(done);
    }

    #[test]
    fn llama_python_tag() {
        let evs = run(r#"<|python_tag|>{"name":"search","arguments":{"q":"hi"}}"#);
        let (name, args, done) = header_args_done(&evs);
        assert_eq!(name.as_deref(), Some("search"));
        assert_eq!(args, r#"{"q":"hi"}"#);
        assert!(done);
    }

    #[test]
    fn bare_json_array() {
        let evs = run(r#"[{"name":"a","arguments":{"x":1}},{"name":"b","arguments":{"y":2}}]"#);
        // Expect two header/args/done sequences.
        let mut headers = 0;
        let mut dones = 0;
        let mut by_idx: std::collections::HashMap<usize, String> = Default::default();
        let mut names: Vec<String> = Vec::new();
        for e in evs {
            match e {
                ToolStreamEvent::Header { index, name, .. } => {
                    headers += 1;
                    assert_eq!(index, names.len());
                    names.push(name);
                }
                ToolStreamEvent::Args { index, fragment } => {
                    by_idx.entry(index).or_default().push_str(&fragment);
                }
                ToolStreamEvent::Done { .. } => dones += 1,
            }
        }
        assert_eq!(headers, 2);
        assert_eq!(dones, 2);
        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(by_idx.get(&0).unwrap(), r#"{"x":1}"#);
        assert_eq!(by_idx.get(&1).unwrap(), r#"{"y":2}"#);
    }

    #[test]
    fn mistral_nemo() {
        let evs =
            run(r#"[TOOL_CALLS][{"name":"sum","arguments":{"a":1,"b":2}}]"#);
        let (name, args, done) = header_args_done(&evs);
        assert_eq!(name.as_deref(), Some("sum"));
        assert_eq!(args, r#"{"a":1,"b":2}"#);
        assert!(done);
    }

    #[test]
    fn deepseek_format() {
        let input = "<\u{ff5c}tool\u{2581}call\u{2581}begin\u{ff5c}>function<\u{ff5c}tool\u{2581}sep\u{ff5c}>write_file\n```json\n{\"path\":\"a\"}\n```<\u{ff5c}tool\u{2581}call\u{2581}end\u{ff5c}>";
        let evs = run(input);
        let (name, args, done) = header_args_done(&evs);
        assert_eq!(name.as_deref(), Some("write_file"));
        assert_eq!(args, r#"{"path":"a"}"#);
        assert!(done);
    }

    #[test]
    fn streaming_emits_multiple_args_chunks() {
        // Feed in one-char chunks; verify args reassemble correctly.
        let mut p = StreamingToolCallParser::new();
        let full = r#"{"name":"echo","arguments":{"msg":"hello world"}}"#;
        let mut acc = String::new();
        let mut chunks = 0;
        let mut args = String::new();
        let mut got_header = false;
        for c in full.chars() {
            acc.push(c);
            p.feed(&acc);
            for ev in p.take_events() {
                match ev {
                    ToolStreamEvent::Header { name, .. } => {
                        assert_eq!(name, "echo");
                        got_header = true;
                    }
                    ToolStreamEvent::Args { fragment, .. } => {
                        chunks += 1;
                        args.push_str(&fragment);
                    }
                    ToolStreamEvent::Done { .. } => {}
                }
            }
        }
        p.finalize();
        for ev in p.take_events() {
            if let ToolStreamEvent::Args { fragment, .. } = ev {
                chunks += 1;
                args.push_str(&fragment);
            }
        }
        assert!(got_header);
        assert!(chunks > 1, "expected multiple arg chunks, got {}", chunks);
        assert_eq!(args, r#"{"msg":"hello world"}"#);
    }

    #[test]
    fn ignores_plain_text() {
        let mut p = StreamingToolCallParser::new();
        p.feed("hello there, this is not a tool call.");
        p.finalize();
        let evs = p.take_events();
        assert!(evs.is_empty());
        assert!(!p.committed());
    }

    #[test]
    fn ignores_unrelated_json() {
        // Plain JSON without `name`/`function` — we leave it alone.
        let mut p = StreamingToolCallParser::new();
        p.feed(r#"{"unrelated": true, "other": 1}"#);
        p.finalize();
        let evs = p.take_events();
        assert!(evs.is_empty());
    }
}
