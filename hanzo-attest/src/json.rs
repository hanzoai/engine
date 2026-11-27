//! Minimal pure-std RFC 8259 JSON parser and deterministic serializer.
//!
//! Enables zero-dependency canonical JSON interchange between the Rust
//! `hanzo-attest` crate and the independent Python 3.12 stdlib client.

use std::collections::BTreeMap;

use crate::error::{AttestError, ErrorCode, Result};

const MAX_DEPTH: usize = 64;

/// A JSON value represented with deterministic key ordering (`BTreeMap`).
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// JSON `null`.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// Non-negative or signed integer stored as `i64` (used for timestamps/epochs/TCB).
    Int(i64),
    /// Unsigned 64-bit integer when value exceeds `i64::MAX`.
    Uint(u64),
    /// Floating-point number (used for benchmark metrics).
    Float(f64),
    /// UTF-8 JSON string.
    Str(String),
    /// Ordered JSON array.
    Array(Vec<JsonValue>),
    /// Lexicographically sorted JSON object.
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    /// Returns this value as an object map or returns [`ErrorCode::InvalidJson`].
    pub fn as_object(&self) -> Result<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(map) => Ok(map),
            _ => Err(AttestError::new(
                ErrorCode::InvalidJson,
                "expected JSON object",
            )),
        }
    }

    /// Returns this value as a string slice.
    pub fn as_str(&self) -> Result<&str> {
        match self {
            Self::Str(s) => Ok(s.as_str()),
            _ => Err(AttestError::new(
                ErrorCode::InvalidJson,
                "expected JSON string",
            )),
        }
    }

    /// Returns this value as a boolean.
    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(b) => Ok(*b),
            _ => Err(AttestError::new(
                ErrorCode::InvalidJson,
                "expected JSON boolean",
            )),
        }
    }

    /// Returns this value as an unsigned 64-bit integer.
    pub fn as_u64(&self) -> Result<u64> {
        match self {
            Self::Uint(u) => Ok(*u),
            Self::Int(i) if *i >= 0 => Ok(*i as u64),
            _ => Err(AttestError::new(
                ErrorCode::InvalidJson,
                "expected non-negative JSON integer",
            )),
        }
    }

    /// Returns this value as an array slice.
    pub fn as_array(&self) -> Result<&[JsonValue]> {
        match self {
            Self::Array(arr) => Ok(arr.as_slice()),
            _ => Err(AttestError::new(
                ErrorCode::InvalidJson,
                "expected JSON array",
            )),
        }
    }

    /// Looks up a required field in a JSON object.
    pub fn req<'a>(obj: &'a BTreeMap<String, JsonValue>, key: &str) -> Result<&'a JsonValue> {
        obj.get(key).ok_or_else(|| {
            AttestError::new(
                ErrorCode::InvalidJson,
                format!("missing required JSON field '{key}'"),
            )
        })
    }

    /// Serializes this value into compact canonical JSON (sorted keys, no extra whitespace).
    #[must_use]
    pub fn to_canonical_string(&self) -> String {
        let mut out = String::new();
        self.write_compact(&mut out);
        out
    }

    /// Serializes this value into pretty-printed JSON with 2-space indentation.
    #[must_use]
    pub fn to_pretty_string(&self) -> String {
        let mut out = String::new();
        self.write_pretty(&mut out, 0);
        out.push('\n');
        out
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Int(i) => out.push_str(&i.to_string()),
            Self::Uint(u) => out.push_str(&u.to_string()),
            Self::Float(f) => {
                if f.is_finite() {
                    out.push_str(&format!("{f}"));
                } else {
                    out.push_str("null");
                }
            }
            Self::Str(s) => write_escaped_str(s, out),
            Self::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_compact(out);
                }
                out.push(']');
            }
            Self::Object(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_escaped_str(k, out);
                    out.push(':');
                    v.write_compact(out);
                }
                out.push('}');
            }
        }
    }

    fn write_pretty(&self, out: &mut String, indent: usize) {
        match self {
            Self::Array(items) if !items.is_empty() => {
                out.push_str("[\n");
                for (i, item) in items.iter().enumerate() {
                    push_indent(out, indent + 2);
                    item.write_pretty(out, indent + 2);
                    if i + 1 < items.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                push_indent(out, indent);
                out.push(']');
            }
            Self::Object(map) if !map.is_empty() => {
                out.push_str("{\n");
                for (i, (k, v)) in map.iter().enumerate() {
                    push_indent(out, indent + 2);
                    write_escaped_str(k, out);
                    out.push_str(": ");
                    v.write_pretty(out, indent + 2);
                    if i + 1 < map.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                push_indent(out, indent);
                out.push('}');
            }
            _ => self.write_compact(out),
        }
    }
}

fn push_indent(out: &mut String, spaces: usize) {
    for _ in 0..spaces {
        out.push(' ');
    }
}

fn write_escaped_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parses an RFC 8259 JSON document into a [`JsonValue`].
pub fn parse_json(input: &str) -> Result<JsonValue> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    let val = parser.parse_value(0)?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(AttestError::new(
            ErrorCode::InvalidJson,
            "trailing characters after JSON value",
        ));
    }
    Ok(val)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len()
            && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\r' | b'\n')
        {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue> {
        if depth > MAX_DEPTH {
            return Err(AttestError::new(
                ErrorCode::InvalidJson,
                "JSON nesting depth limit exceeded",
            ));
        }
        self.skip_ws();
        let Some(b) = self.peek() else {
            return Err(AttestError::new(
                ErrorCode::InvalidJson,
                "unexpected end of JSON input",
            ));
        };
        match b {
            b'{' => self.parse_object(depth + 1),
            b'[' => self.parse_array(depth + 1),
            b'"' => self.parse_string().map(JsonValue::Str),
            b't' => self.parse_literal(b"true", JsonValue::Bool(true)),
            b'f' => self.parse_literal(b"false", JsonValue::Bool(false)),
            b'n' => self.parse_literal(b"null", JsonValue::Null),
            b'-' | b'0'..=b'9' => self.parse_number(),
            _ => Err(AttestError::new(
                ErrorCode::InvalidJson,
                format!("unexpected JSON token byte 0x{b:02x}"),
            )),
        }
    }

    fn parse_literal(&mut self, lit: &[u8], val: JsonValue) -> Result<JsonValue> {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            Err(AttestError::new(
                ErrorCode::InvalidJson,
                "invalid JSON literal",
            ))
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue> {
        self.pos += 1; // '{'
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(AttestError::new(
                    ErrorCode::InvalidJson,
                    "expected string key in JSON object",
                ));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(AttestError::new(
                    ErrorCode::InvalidJson,
                    "expected ':' after JSON object key",
                ));
            }
            self.pos += 1;
            let val = self.parse_value(depth)?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(map));
                }
                _ => {
                    return Err(AttestError::new(
                        ErrorCode::InvalidJson,
                        "expected ',' or '}' in JSON object",
                    ));
                }
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            let val = self.parse_value(depth)?;
            items.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => {
                    return Err(AttestError::new(
                        ErrorCode::InvalidJson,
                        "expected ',' or ']' in JSON array",
                    ));
                }
            }
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.pos += 1; // '"'
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    if self.pos >= self.bytes.len() {
                        return Err(AttestError::new(
                            ErrorCode::InvalidJson,
                            "unterminated JSON string escape",
                        ));
                    }
                    let esc = self.bytes[self.pos];
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let code = self.parse_hex4()?;
                            let ch = char::from_u32(code).ok_or_else(|| {
                                AttestError::new(
                                    ErrorCode::InvalidJson,
                                    "invalid unicode escape in JSON string",
                                )
                            })?;
                            out.push(ch);
                        }
                        _ => {
                            return Err(AttestError::new(
                                ErrorCode::InvalidJson,
                                "unsupported JSON escape sequence",
                            ));
                        }
                    }
                }
                c if c < 0x20 => {
                    return Err(AttestError::new(
                        ErrorCode::InvalidJson,
                        "unescaped control character in JSON string",
                    ));
                }
                _ => {
                    // Decode UTF-8 sequence starting at self.pos - 1.
                    let start = self.pos - 1;
                    let rest = std::str::from_utf8(&self.bytes[start..]).map_err(|_| {
                        AttestError::new(ErrorCode::InvalidJson, "invalid UTF-8 in JSON string")
                    })?;
                    let ch = rest.chars().next().ok_or_else(|| {
                        AttestError::new(ErrorCode::InvalidJson, "empty UTF-8 char")
                    })?;
                    self.pos = start + ch.len_utf8();
                    out.push(ch);
                }
            }
        }
        Err(AttestError::new(
            ErrorCode::InvalidJson,
            "unterminated JSON string",
        ))
    }

    fn parse_hex4(&mut self) -> Result<u32> {
        if self.pos + 4 > self.bytes.len() {
            return Err(AttestError::new(
                ErrorCode::InvalidJson,
                "truncated \\uXXXX escape",
            ));
        }
        let s = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
            .map_err(|_| AttestError::new(ErrorCode::InvalidJson, "invalid \\uXXXX escape"))?;
        self.pos += 4;
        u32::from_str_radix(s, 16)
            .map_err(|_| AttestError::new(ErrorCode::InvalidJson, "invalid hex in \\uXXXX escape"))
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.pos;
        let mut is_float = false;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'9' => self.pos += 1,
                b'.' | b'e' | b'E' | b'+' | b'-' => {
                    is_float = true;
                    self.pos += 1;
                }
                _ => break,
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| AttestError::new(ErrorCode::InvalidJson, "invalid number encoding"))?;
        if is_float {
            let f: f64 = s
                .parse()
                .map_err(|_| AttestError::new(ErrorCode::InvalidJson, "invalid JSON float"))?;
            return Ok(JsonValue::Float(f));
        }
        if let Ok(u) = s.parse::<u64>() {
            Ok(JsonValue::Uint(u))
        } else if let Ok(i) = s.parse::<i64>() {
            Ok(JsonValue::Int(i))
        } else {
            Err(AttestError::new(
                ErrorCode::InvalidJson,
                "invalid JSON integer",
            ))
        }
    }
}
