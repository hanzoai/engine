//! Strict RFC 8259 JSON parser and AST (zero external dependencies).
//!
//! Used both to parse user-supplied JSON Schemas / HuggingFace `tokenizer.json`
//! vocabularies AND to independently validate generated outputs in benchmarks
//! (ensuring invalid-output rates are never self-graded by the PDA).

use std::fmt;

/// Maximum JSON nesting depth allowed by the parser to prevent stack exhaustion.
pub const MAX_JSON_PARSE_DEPTH: usize = 128;

/// Error returned when parsing malformed JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    pub offset: usize,
    pub message: String,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "JSON parse error at byte {}: {}",
            self.offset, self.message
        )
    }
}

impl std::error::Error for JsonError {}

/// Ordered JSON value AST (preserves object property insertion order).
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(entries) => Some(entries),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Int(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Int(v) => Some(*v as f64),
            JsonValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Canonical compact RFC 8259 JSON serialization.
    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    /// Pretty-printed RFC 8259 JSON serialization (2-space indentation).
    pub fn to_json_pretty(&self) -> String {
        let mut out = String::new();
        self.write_json_pretty(&mut out, 0);
        out
    }

    fn write_json(&self, out: &mut String) {
        match self {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(true) => out.push_str("true"),
            JsonValue::Bool(false) => out.push_str("false"),
            JsonValue::Int(n) => out.push_str(&n.to_string()),
            JsonValue::Float(v) => {
                if v.is_finite() {
                    let s = v.to_string();
                    out.push_str(&s);
                    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                        out.push_str(".0");
                    }
                } else {
                    out.push_str("null");
                }
            }
            JsonValue::String(s) => write_json_escaped_string(s, out),
            JsonValue::Array(items) => {
                out.push('[');
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    item.write_json(out);
                }
                out.push(']');
            }
            JsonValue::Object(entries) => {
                out.push('{');
                for (idx, (k, v)) in entries.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    write_json_escaped_string(k, out);
                    out.push(':');
                    v.write_json(out);
                }
                out.push('}');
            }
        }
    }

    fn write_json_pretty(&self, out: &mut String, indent: usize) {
        match self {
            JsonValue::Array(items) if !items.is_empty() => {
                out.push_str("[\n");
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(",\n");
                    }
                    push_indent(out, indent + 2);
                    item.write_json_pretty(out, indent + 2);
                }
                out.push('\n');
                push_indent(out, indent);
                out.push(']');
            }
            JsonValue::Object(entries) if !entries.is_empty() => {
                out.push_str("{\n");
                for (idx, (k, v)) in entries.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(",\n");
                    }
                    push_indent(out, indent + 2);
                    write_json_escaped_string(k, out);
                    out.push_str(": ");
                    v.write_json_pretty(out, indent + 2);
                }
                out.push('\n');
                push_indent(out, indent);
                out.push('}');
            }
            _ => self.write_json(out),
        }
    }
}

fn push_indent(out: &mut String, spaces: usize) {
    for _ in 0..spaces {
        out.push(' ');
    }
}

/// Format a Rust string slice as a double-quoted RFC 8259 JSON string literal.
pub fn write_json_escaped_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let code = c as u32;
                out.push_str(&format!("\\u{code:04x}"));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Escape a string as a quoted RFC 8259 JSON string literal.
pub fn json_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    write_json_escaped_string(s, &mut out);
    out
}

/// Parse an RFC 8259 JSON document strictly (rejects trailing tokens, trailing
/// commas, duplicate unescaped control chars, leading numeric zeros, etc.).
pub fn parse_json(input: &str) -> Result<JsonValue, JsonError> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
        depth: 0,
    };
    parser.skip_ws();
    let val = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(parser.err("trailing characters after JSON value"));
    }
    Ok(val)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: impl Into<String>) -> JsonError {
        JsonError {
            offset: self.pos,
            message: msg.into(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, JsonError> {
        let Some(b) = self.peek() else {
            return Err(self.err("unexpected end of JSON input"));
        };
        match b {
            b'n' => self.parse_literal(b"null", JsonValue::Null),
            b't' => self.parse_literal(b"true", JsonValue::Bool(true)),
            b'f' => self.parse_literal(b"false", JsonValue::Bool(false)),
            b'"' => self.parse_string().map(JsonValue::String),
            b'[' => self.parse_array(),
            b'{' => self.parse_object(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            _ => Err(self.err(format!("unexpected byte 0x{b:02x}"))),
        }
    }

    fn parse_literal(&mut self, expected: &[u8], value: JsonValue) -> Result<JsonValue, JsonError> {
        if self.bytes[self.pos..].starts_with(expected) {
            self.pos += expected.len();
            Ok(value)
        } else {
            Err(self.err("invalid JSON literal"))
        }
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        if self.peek() != Some(b'"') {
            return Err(self.err("expected '\"'"));
        }
        self.pos += 1;
        let mut out = String::new();
        while let Some(b) = self.peek() {
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let Some(esc) = self.peek() else {
                    return Err(self.err("unterminated escape sequence"));
                };
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{08}'),
                    b'f' => out.push('\u{0C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let code = self.parse_hex4()?;
                        if (0xD800..=0xDBFF).contains(&code) {
                            // High surrogate: must be followed by \uXXXX low surrogate.
                            if !self.bytes[self.pos..].starts_with(b"\\u") {
                                return Err(self.err("unpaired UTF-16 high surrogate"));
                            }
                            self.pos += 2;
                            let low = self.parse_hex4()?;
                            if !(0xDC00..=0xDFFF).contains(&low) {
                                return Err(self.err("invalid UTF-16 low surrogate"));
                            }
                            let combined =
                                0x10000 + (((code as u32 - 0xD800) << 10) | (low as u32 - 0xDC00));
                            let ch = char::from_u32(combined)
                                .ok_or_else(|| self.err("invalid Unicode scalar value"))?;
                            out.push(ch);
                        } else if (0xDC00..=0xDFFF).contains(&code) {
                            return Err(self.err("unexpected UTF-16 low surrogate"));
                        } else {
                            let ch = char::from_u32(code as u32)
                                .ok_or_else(|| self.err("invalid Unicode scalar value"))?;
                            out.push(ch);
                        }
                    }
                    _ => return Err(self.err(format!("invalid escape byte 0x{esc:02x}"))),
                }
            } else if b < 0x20 {
                return Err(self.err("unescaped control character in JSON string"));
            } else if b < 0x80 {
                out.push(b as char);
                self.pos += 1;
            } else {
                // Multi-byte UTF-8: input is &str so UTF-8 boundaries are valid.
                let rest = &self.bytes[self.pos..];
                let rest_str = std::str::from_utf8(rest)
                    .map_err(|_| self.err("invalid UTF-8 in JSON string"))?;
                let ch = rest_str
                    .chars()
                    .next()
                    .ok_or_else(|| self.err("unexpected end of string"))?;
                out.push(ch);
                self.pos += ch.len_utf8();
            }
        }
        Err(self.err("unterminated JSON string"))
    }

    fn parse_hex4(&mut self) -> Result<u16, JsonError> {
        if self.pos + 4 > self.bytes.len() {
            return Err(self.err("truncated \\uXXXX escape"));
        }
        let mut val: u16 = 0;
        for _ in 0..4 {
            let b = self.bytes[self.pos];
            self.pos += 1;
            let digit = match b {
                b'0'..=b'9' => (b - b'0') as u16,
                b'a'..=b'f' => (b - b'a' + 10) as u16,
                b'A'..=b'F' => (b - b'A' + 10) as u16,
                _ => return Err(self.err("non-hex character in \\uXXXX escape")),
            };
            val = (val << 4) | digit;
        }
        Ok(val)
    }

    fn parse_number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.err("leading zeros are not allowed in JSON numbers"));
                }
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(self.err("expected digit in JSON number")),
        }

        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err("expected digit after decimal point"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err("expected digit in exponent"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        let slice = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.err("invalid UTF-8 in number"))?;
        if !is_float {
            if let Ok(i) = slice.parse::<i64>() {
                return Ok(JsonValue::Int(i));
            }
        }
        let f = slice
            .parse::<f64>()
            .map_err(|_| self.err("invalid floating-point number"))?;
        if !f.is_finite() {
            return Err(self.err("non-finite JSON number"));
        }
        Ok(JsonValue::Float(f))
    }

    fn parse_array(&mut self) -> Result<JsonValue, JsonError> {
        self.depth += 1;
        if self.depth > MAX_JSON_PARSE_DEPTH {
            return Err(self.err("JSON nesting depth exceeds maximum limit"));
        }
        self.pos += 1; // '['
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, JsonError> {
        self.depth += 1;
        if self.depth > MAX_JSON_PARSE_DEPTH {
            return Err(self.err("JSON nesting depth exceeds maximum limit"));
        }
        self.pos += 1; // '{'
        self.skip_ws();
        let mut entries = Vec::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(self.err("expected ':' after object key"));
            }
            self.pos += 1;
            self.skip_ws();
            let val = self.parse_value()?;
            entries.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(JsonValue::Object(entries));
                }
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rfc8259_values_and_escapes() {
        let doc = r#"{"a": 1, "b": -2.5e2, "c": [true, false, null], "s": "hi\n\"\\\u0041\uD83D\uDE00日本語"}"#;
        let val = parse_json(doc).expect("valid JSON");
        assert_eq!(val.get("a").and_then(JsonValue::as_i64), Some(1));
        assert_eq!(val.get("b").and_then(JsonValue::as_f64), Some(-250.0));
        assert_eq!(
            val.get("s").and_then(JsonValue::as_str),
            Some("hi\n\"\\A😀日本語")
        );
    }

    #[test]
    fn rejects_invalid_json() {
        assert!(parse_json(r#"{"a": 01}"#).is_err());
        assert!(parse_json(r#"{"a": 1,}"#).is_err());
        assert!(parse_json("{\"a\": \"line\nbreak\"}").is_err());
        assert!(parse_json(r#"{"a": "\uD800"}"#).is_err());
        assert!(parse_json(r#"{} trailing"#).is_err());
    }

    #[test]
    fn enforces_max_parse_depth() {
        let deep = "[".repeat(MAX_JSON_PARSE_DEPTH + 4) + &"]".repeat(MAX_JSON_PARSE_DEPTH + 4);
        assert!(parse_json(&deep).is_err());
    }
}
