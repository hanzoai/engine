//! Minimal, zero-dependency, DoS-bounded JSON parser & canonical encoder.
//!
//! Enforces strict nesting depth (`MAX_JSON_DEPTH`) and payload byte caps
//! (`MAX_JSON_BYTES`) so malformed or malicious network payloads cannot cause
//! stack exhaustion or unbounded heap allocation on a provider host.

use std::collections::BTreeMap;
use std::fmt;

/// Maximum permitted JSON nesting depth (defends against stack-overflow payloads).
pub const MAX_JSON_DEPTH: usize = 32;
/// Maximum permitted JSON input size (1 MiB) for control/inference payloads.
pub const MAX_JSON_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonError(pub String);

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "json error: {}", self.0)
    }
}

impl std::error::Error for JsonError {}

impl JsonValue {
    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            JsonValue::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Uint(v) => Some(*v),
            JsonValue::Int(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Int(v) => Some(*v),
            JsonValue::Uint(v) => i64::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?.get(key)
    }

    pub fn get_str(&self, key: &str) -> Result<&str, JsonError> {
        self.get(key)
            .and_then(JsonValue::as_str)
            .ok_or_else(|| JsonError(format!("missing or invalid string field `{key}`")))
    }

    pub fn get_u64(&self, key: &str) -> Result<u64, JsonError> {
        self.get(key)
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| JsonError(format!("missing or invalid u64 field `{key}`")))
    }

    pub fn get_bool(&self, key: &str) -> Result<bool, JsonError> {
        self.get(key)
            .and_then(JsonValue::as_bool)
            .ok_or_else(|| JsonError(format!("missing or invalid bool field `{key}`")))
    }

    /// Encodes deterministically (BTreeMap keys sorted lexicographically, compact).
    pub fn to_compact_string(&self) -> String {
        let mut out = String::new();
        self.write_compact(&mut out);
        out
    }

    /// Encodes with 2-space pretty indentation (deterministic key order).
    pub fn to_pretty_string(&self) -> String {
        let mut out = String::new();
        self.write_pretty(&mut out, 0);
        out.push('\n');
        out
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            JsonValue::Int(v) => out.push_str(&v.to_string()),
            JsonValue::Uint(v) => out.push_str(&v.to_string()),
            JsonValue::Float(v) => {
                if v.is_finite() {
                    out.push_str(&format!("{v}"));
                } else {
                    out.push_str("null");
                }
            }
            JsonValue::Str(s) => write_escaped_str(out, s),
            JsonValue::Array(arr) => {
                out.push('[');
                for (i, item) in arr.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_compact(out);
                }
                out.push(']');
            }
            JsonValue::Object(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_escaped_str(out, k);
                    out.push(':');
                    v.write_compact(out);
                }
                out.push('}');
            }
        }
    }

    fn write_pretty(&self, out: &mut String, indent: usize) {
        match self {
            JsonValue::Array(arr) if !arr.is_empty() => {
                out.push_str("[\n");
                for (i, item) in arr.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    push_indent(out, indent + 2);
                    item.write_pretty(out, indent + 2);
                }
                out.push('\n');
                push_indent(out, indent);
                out.push(']');
            }
            JsonValue::Object(map) if !map.is_empty() => {
                out.push_str("{\n");
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    push_indent(out, indent + 2);
                    write_escaped_str(out, k);
                    out.push_str(": ");
                    v.write_pretty(out, indent + 2);
                }
                out.push('\n');
                push_indent(out, indent);
                out.push('}');
            }
            _ => self.write_compact(out),
        }
    }
}

fn push_indent(out: &mut String, n: usize) {
    for _ in 0..n {
        out.push(' ');
    }
}

pub fn write_escaped_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parses JSON with strict byte-size and recursion-depth caps.
pub fn parse_json(input: &str) -> Result<JsonValue, JsonError> {
    if input.len() > MAX_JSON_BYTES {
        return Err(JsonError(format!(
            "json payload exceeds max size ({} > {})",
            input.len(),
            MAX_JSON_BYTES
        )));
    }
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    let val = parser.parse_value(0)?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(JsonError(format!(
            "trailing bytes at offset {}",
            parser.pos
        )));
    }
    Ok(val)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        if depth > MAX_JSON_DEPTH {
            return Err(JsonError("json nesting depth exceeded".into()));
        }
        self.skip_ws();
        let Some(b) = self.peek() else {
            return Err(JsonError("unexpected end of json".into()));
        };
        match b {
            b'n' => self.parse_literal(b"null", JsonValue::Null),
            b't' => self.parse_literal(b"true", JsonValue::Bool(true)),
            b'f' => self.parse_literal(b"false", JsonValue::Bool(false)),
            b'"' => self.parse_string().map(JsonValue::Str),
            b'[' => self.parse_array(depth + 1),
            b'{' => self.parse_object(depth + 1),
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => Err(JsonError(format!(
                "unexpected byte 0x{other:02x} at offset {}",
                self.pos
            ))),
        }
    }

    fn parse_literal(&mut self, lit: &[u8], val: JsonValue) -> Result<JsonValue, JsonError> {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            Err(JsonError(format!("invalid literal at offset {}", self.pos)))
        }
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        if self.peek() != Some(b'"') {
            return Err(JsonError("expected '\"'".into()));
        }
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let Some(esc) = self.peek() else {
                    return Err(JsonError("unterminated escape".into()));
                };
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\x08'),
                    b'f' => out.push('\x0c'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if self.pos + 4 > self.bytes.len() {
                            return Err(JsonError("truncated \\u escape".into()));
                        }
                        let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                            .map_err(|_| JsonError("invalid utf8 in \\u escape".into()))?;
                        let code = u32::from_str_radix(hex, 16)
                            .map_err(|_| JsonError("invalid hex in \\u escape".into()))?;
                        self.pos += 4;
                        let ch = char::from_u32(code)
                            .ok_or_else(|| JsonError("invalid unicode scalar in \\u".into()))?;
                        out.push(ch);
                    }
                    _ => return Err(JsonError(format!("invalid escape \\{}", esc as char))),
                }
            } else if b < 0x20 {
                return Err(JsonError("unescaped control character in string".into()));
            } else {
                // Advance full UTF-8 character safely.
                let rest = std::str::from_utf8(&self.bytes[self.pos..])
                    .map_err(|_| JsonError("invalid utf8 in string".into()))?;
                let ch = rest
                    .chars()
                    .next()
                    .ok_or_else(|| JsonError("unexpected eof".into()))?;
                out.push(ch);
                self.pos += ch.len_utf8();
            }
        }
        Err(JsonError("unterminated string".into()))
    }

    fn parse_number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.pos;
        let is_neg = self.peek() == Some(b'-');
        if is_neg {
            self.pos += 1;
        }
        let mut is_float = false;
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'9' => self.pos += 1,
                b'.' | b'e' | b'E' | b'+' | b'-' if self.pos > start => {
                    is_float = true;
                    self.pos += 1;
                }
                _ => break,
            }
        }
        let slice = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| JsonError("invalid utf8 in number".into()))?;
        if is_float {
            let f = slice
                .parse::<f64>()
                .map_err(|_| JsonError(format!("invalid float `{slice}`")))?;
            if !f.is_finite() {
                return Err(JsonError("non-finite float".into()));
            }
            Ok(JsonValue::Float(f))
        } else if is_neg {
            let v = slice
                .parse::<i64>()
                .map_err(|_| JsonError(format!("invalid i64 `{slice}`")))?;
            Ok(JsonValue::Int(v))
        } else {
            let v = slice
                .parse::<u64>()
                .map_err(|_| JsonError(format!("invalid u64 `{slice}`")))?;
            Ok(JsonValue::Uint(v))
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value(depth)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => return Err(JsonError(format!("expected ',' or ']' at {}", self.pos))),
            }
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        self.pos += 1; // '{'
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(JsonError(format!("expected ':' at {}", self.pos)));
            }
            self.pos += 1;
            let val = self.parse_value(depth)?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(map));
                }
                _ => return Err(JsonError(format!("expected ',' or '}}' at {}", self.pos))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_canonical_order() {
        let parsed = parse_json(r#"{"z": 1, "a": [true, null, "hi\n\"there\""]}"#).unwrap();
        assert_eq!(
            parsed.to_compact_string(),
            r#"{"a":[true,null,"hi\n\"there\""],"z":1}"#
        );
    }

    #[test]
    fn rejects_excessive_nesting_depth() {
        let deep = "[".repeat(MAX_JSON_DEPTH + 4) + &"]".repeat(MAX_JSON_DEPTH + 4);
        assert!(parse_json(&deep).is_err());
    }
}
