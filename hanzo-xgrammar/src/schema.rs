//! JSON Schema & OpenAI/MCP Tool-Call Compiler -> Byte-Level PDA [`Grammar`].
//!
//! Supported JSON Schema features:
//! - Primitive types: `object`, `array`, `string`, `integer`, `number`, `boolean`, `null`
//! - `properties`, `required` (all-required or mixed required/optional in O(n) rules),
//!   `additionalProperties: false | true`
//! - `array` `items`, `minItems`, `maxItems`
//! - `enum`, `const`, `anyOf` / `oneOf`
//! - Recursive `$defs` / `definitions` + `$ref` (e.g., recursive AST/tree schemas)
//! - OpenAI / MCP tool-call envelopes (`{"tool": "...", "arguments": {...}}` and
//!   `{"name": "...", "arguments": {...}}`)
//!
//! Compiler optimizations & hardening:
//! - **Fused-whitespace (`Ws`-inlined) primitive rules**: whitespace `[ \t\r\n]*` is
//!   fused directly into leading delimiters and value starters so space-prefixed BPE
//!   tokens (`" the"`, `" function"`) are rejected context-independently rather than
//!   leaking into runtime uncertain-token sets.
//! - **Full RFC 3629 UTF-8 + RFC 8259 JSON escape support** (`\"`, `\\`, `\/`, `\b`,
//!   `\f`, `\n`, `\r`, `\t`, `\uXXXX`) inside `StringBody`, with 100% tail-call
//!   optimization (O(1) stack depth for arbitrary-length strings).
//! - **Explicit DoS & malformed-schema rejection**: enforces `MAX_SCHEMA_DEPTH`,
//!   `MAX_OBJECT_PROPERTIES`, `MAX_ENUM_VARIANTS`, `MAX_ARRAY_UNROLL`, rejects
//!   unresolved/left-recursive `$ref`, invalid `required` keys, empty `enum`/`anyOf`,
//!   and `minItems > maxItems`.
//! - **Independent RFC 8259 + Schema Validator** (`SchemaDocument::validate_json_str`)
//!   for ground-truth invalid-output-rate measurement.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::grammar::{ByteSet, Element, Grammar, GrammarBuilder, GrammarError, RuleId};
use crate::json::{json_escape_string, parse_json, JsonValue};

/// Maximum JSON Schema AST nesting depth.
pub const MAX_SCHEMA_DEPTH: usize = 32;
/// Maximum properties per object schema.
pub const MAX_OBJECT_PROPERTIES: usize = 128;
/// Maximum variants in an `enum` or `anyOf`.
pub const MAX_ENUM_VARIANTS: usize = 256;
/// Maximum `minItems` / `maxItems` unroll count in array schemas.
pub const MAX_ARRAY_UNROLL: usize = 64;

/// Error returned when parsing or compiling a malformed or DoS-unsafe JSON Schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    MalformedJson(String),
    UnsupportedSchema(String),
    DepthLimitExceeded { depth: usize, max: usize },
    PropertyLimitExceeded { count: usize, max: usize },
    VariantLimitExceeded { count: usize, max: usize },
    ArrayBoundLimitExceeded { bound: usize, max: usize },
    InvalidArrayBounds { min: usize, max: usize },
    MissingRequiredProperty { property: String },
    UnresolvedRef { reference: String },
    Grammar(GrammarError),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaError::MalformedJson(m) => write!(f, "malformed JSON schema: {m}"),
            SchemaError::UnsupportedSchema(m) => write!(f, "invalid/unsupported schema: {m}"),
            SchemaError::DepthLimitExceeded { depth, max } => {
                write!(f, "schema nesting depth {depth} exceeds max {max}")
            }
            SchemaError::PropertyLimitExceeded { count, max } => {
                write!(f, "object property count {count} exceeds max {max}")
            }
            SchemaError::VariantLimitExceeded { count, max } => {
                write!(f, "enum/anyOf variant count {count} exceeds max {max}")
            }
            SchemaError::ArrayBoundLimitExceeded { bound, max } => {
                write!(f, "array item bound {bound} exceeds max {max}")
            }
            SchemaError::InvalidArrayBounds { min, max } => {
                write!(f, "invalid array bounds: minItems ({min}) > maxItems ({max})")
            }
            SchemaError::MissingRequiredProperty { property } => write!(
                f,
                "required property '{property}' not declared in properties while additionalProperties is false"
            ),
            SchemaError::UnresolvedRef { reference } => {
                write!(f, "unresolved $ref '{reference}'")
            }
            SchemaError::Grammar(g) => write!(f, "grammar compilation error: {g}"),
        }
    }
}

impl std::error::Error for SchemaError {}

impl From<GrammarError> for SchemaError {
    fn from(e: GrammarError) -> Self {
        SchemaError::Grammar(e)
    }
}

/// Typed JSON Schema AST node.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonSchema {
    Null,
    Boolean,
    Integer,
    Number,
    String,
    Const(JsonValue),
    Enum(Vec<JsonValue>),
    Array {
        items: Box<JsonSchema>,
        min_items: usize,
        max_items: Option<usize>,
    },
    Object {
        properties: Vec<(String, JsonSchema)>,
        required: Vec<String>,
        additional_properties: bool,
    },
    AnyOf(Vec<JsonSchema>),
    Ref(String),
    Any,
}

/// Complete JSON Schema document including optional `$defs` / `definitions` for recursion.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaDocument {
    pub root: JsonSchema,
    pub defs: Vec<(String, JsonSchema)>,
}

/// Envelope key convention for OpenAI / MCP tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolEnvelopeFormat {
    /// `{"tool": "<name>", "arguments": {...}}`
    ToolAndArguments,
    /// `{"name": "<name>", "arguments": {...}}` (OpenAI / Qwen / Hanzo Zen `tools/grammar.rs`)
    NameAndArguments,
}

/// Specification of a single callable tool with a strict argument JSON Schema.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub arguments_schema: JsonSchema,
}

impl ToolSpec {
    pub fn new(name: impl Into<String>, arguments_schema: JsonSchema) -> Self {
        Self {
            name: name.into(),
            arguments_schema,
        }
    }
}

/// Build a strict multi-tool OpenAI/MCP tool-call envelope [`SchemaDocument`].
pub fn build_tool_call_schema(
    tools: &[ToolSpec],
    format: ToolEnvelopeFormat,
) -> Result<SchemaDocument, SchemaError> {
    if tools.is_empty() {
        return Err(SchemaError::UnsupportedSchema(
            "tool list must not be empty".into(),
        ));
    }
    if tools.len() > MAX_ENUM_VARIANTS {
        return Err(SchemaError::VariantLimitExceeded {
            count: tools.len(),
            max: MAX_ENUM_VARIANTS,
        });
    }
    let name_key = match format {
        ToolEnvelopeFormat::ToolAndArguments => "tool",
        ToolEnvelopeFormat::NameAndArguments => "name",
    };

    let mut variants = Vec::with_capacity(tools.len());
    for tool in tools {
        variants.push(JsonSchema::Object {
            properties: vec![
                (
                    name_key.to_string(),
                    JsonSchema::Const(JsonValue::String(tool.name.clone())),
                ),
                ("arguments".to_string(), tool.arguments_schema.clone()),
            ],
            required: vec![name_key.to_string(), "arguments".to_string()],
            additional_properties: false,
        });
    }

    let root = if variants.len() == 1 {
        variants.pop().expect("len == 1")
    } else {
        JsonSchema::AnyOf(variants)
    };
    Ok(SchemaDocument {
        root,
        defs: Vec::new(),
    })
}

impl SchemaDocument {
    /// Parse a JSON Schema document string (`Draft 7` / `2020-12` subset).
    pub fn from_json_str(input: &str) -> Result<Self, SchemaError> {
        let val = parse_json(input).map_err(|e| SchemaError::MalformedJson(e.to_string()))?;
        Self::from_json_value(&val)
    }

    /// Parse a JSON Schema from a parsed [`JsonValue`].
    pub fn from_json_value(val: &JsonValue) -> Result<Self, SchemaError> {
        let obj = val.as_object().ok_or_else(|| {
            SchemaError::UnsupportedSchema("root schema must be a JSON object".into())
        })?;

        let mut defs = Vec::new();
        for (k, v) in obj {
            if k == "$defs" || k == "definitions" {
                let def_entries = v.as_object().ok_or_else(|| {
                    SchemaError::UnsupportedSchema(format!("'{k}' must be a JSON object"))
                })?;
                if def_entries.len() > MAX_OBJECT_PROPERTIES {
                    return Err(SchemaError::PropertyLimitExceeded {
                        count: def_entries.len(),
                        max: MAX_OBJECT_PROPERTIES,
                    });
                }
                for (def_name, def_val) in def_entries {
                    let parsed = parse_schema_node(def_val, 1)?;
                    defs.push((def_name.clone(), parsed));
                }
            }
        }

        let root = parse_schema_node(val, 0)?;
        Ok(Self { root, defs })
    }

    /// Compile this JSON Schema document into a finalized byte-level PDA [`Grammar`].
    pub fn compile_grammar(&self) -> Result<Grammar, SchemaError> {
        SchemaCompiler::compile(self)
    }

    /// Compute a stable canonical hash of this schema document (for LRU cache keying).
    pub fn canonical_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        hash_schema_doc(self, &mut h);
        h
    }

    /// Independently parse `output` with our strict RFC 8259 JSON parser and validate
    /// the resulting AST against this [`SchemaDocument`].
    pub fn validate_json_str(&self, output: &str) -> Result<(), String> {
        let val = parse_json(output).map_err(|e| format!("RFC 8259 JSON syntax error: {e}"))?;
        let def_map: HashMap<&str, &JsonSchema> =
            self.defs.iter().map(|(k, v)| (k.as_str(), v)).collect();
        validate_node(&self.root, &val, &def_map, 0)
    }
}

fn parse_schema_node(val: &JsonValue, depth: usize) -> Result<JsonSchema, SchemaError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(SchemaError::DepthLimitExceeded {
            depth,
            max: MAX_SCHEMA_DEPTH,
        });
    }
    let obj = val.as_object().ok_or_else(|| {
        SchemaError::UnsupportedSchema("schema node must be a JSON object".into())
    })?;
    if obj.is_empty() {
        return Ok(JsonSchema::Any);
    }

    if let Some(ref_val) = val.get("$ref") {
        let ref_str = ref_val
            .as_str()
            .ok_or_else(|| SchemaError::UnsupportedSchema("$ref must be a string".into()))?;
        let name = ref_str
            .strip_prefix("#/$defs/")
            .or_else(|| ref_str.strip_prefix("#/definitions/"))
            .ok_or_else(|| SchemaError::UnresolvedRef {
                reference: ref_str.to_string(),
            })?;
        return Ok(JsonSchema::Ref(name.to_string()));
    }

    if let Some(const_val) = val.get("const") {
        return Ok(JsonSchema::Const(const_val.clone()));
    }

    if let Some(enum_val) = val.get("enum") {
        let items = enum_val
            .as_array()
            .ok_or_else(|| SchemaError::UnsupportedSchema("'enum' must be an array".into()))?;
        if items.is_empty() {
            return Err(SchemaError::UnsupportedSchema(
                "'enum' must have at least one value".into(),
            ));
        }
        if items.len() > MAX_ENUM_VARIANTS {
            return Err(SchemaError::VariantLimitExceeded {
                count: items.len(),
                max: MAX_ENUM_VARIANTS,
            });
        }
        return Ok(JsonSchema::Enum(items.to_vec()));
    }

    if let Some(any_of_val) = val.get("anyOf").or_else(|| val.get("oneOf")) {
        let items = any_of_val.as_array().ok_or_else(|| {
            SchemaError::UnsupportedSchema("'anyOf'/'oneOf' must be an array".into())
        })?;
        if items.is_empty() {
            return Err(SchemaError::UnsupportedSchema(
                "'anyOf'/'oneOf' must have at least one branch".into(),
            ));
        }
        if items.len() > MAX_ENUM_VARIANTS {
            return Err(SchemaError::VariantLimitExceeded {
                count: items.len(),
                max: MAX_ENUM_VARIANTS,
            });
        }
        let mut branches = Vec::with_capacity(items.len());
        for item in items {
            branches.push(parse_schema_node(item, depth + 1)?);
        }
        return Ok(JsonSchema::AnyOf(branches));
    }

    let Some(type_val) = val.get("type") else {
        // Object without explicit "type" but with "properties" defaults to object.
        if val.get("properties").is_some() {
            return parse_object_schema(val, depth);
        }
        return Ok(JsonSchema::Any);
    };

    let type_str = type_val
        .as_str()
        .ok_or_else(|| SchemaError::UnsupportedSchema("'type' must be a string".into()))?;
    match type_str {
        "null" => Ok(JsonSchema::Null),
        "boolean" => Ok(JsonSchema::Boolean),
        "integer" => Ok(JsonSchema::Integer),
        "number" => Ok(JsonSchema::Number),
        "string" => Ok(JsonSchema::String),
        "array" => {
            let items = match val.get("items") {
                Some(it) => Box::new(parse_schema_node(it, depth + 1)?),
                None => Box::new(JsonSchema::Any),
            };
            let min_items = match val.get("minItems") {
                Some(v) => v.as_u64().ok_or_else(|| {
                    SchemaError::UnsupportedSchema(
                        "'minItems' must be a non-negative integer".into(),
                    )
                })? as usize,
                None => 0,
            };
            let max_items = match val.get("maxItems") {
                Some(v) => Some(v.as_u64().ok_or_else(|| {
                    SchemaError::UnsupportedSchema(
                        "'maxItems' must be a non-negative integer".into(),
                    )
                })? as usize),
                None => None,
            };
            if min_items > MAX_ARRAY_UNROLL {
                return Err(SchemaError::ArrayBoundLimitExceeded {
                    bound: min_items,
                    max: MAX_ARRAY_UNROLL,
                });
            }
            if let Some(mx) = max_items {
                if mx > MAX_ARRAY_UNROLL {
                    return Err(SchemaError::ArrayBoundLimitExceeded {
                        bound: mx,
                        max: MAX_ARRAY_UNROLL,
                    });
                }
                if min_items > mx {
                    return Err(SchemaError::InvalidArrayBounds {
                        min: min_items,
                        max: mx,
                    });
                }
            }
            Ok(JsonSchema::Array {
                items,
                min_items,
                max_items,
            })
        }
        "object" => parse_object_schema(val, depth),
        other => Err(SchemaError::UnsupportedSchema(format!(
            "unknown JSON Schema type '{other}'"
        ))),
    }
}

fn parse_object_schema(val: &JsonValue, depth: usize) -> Result<JsonSchema, SchemaError> {
    let mut properties = Vec::new();
    let mut prop_names = HashSet::new();
    if let Some(props_val) = val.get("properties") {
        let entries = props_val.as_object().ok_or_else(|| {
            SchemaError::UnsupportedSchema("'properties' must be an object".into())
        })?;
        if entries.len() > MAX_OBJECT_PROPERTIES {
            return Err(SchemaError::PropertyLimitExceeded {
                count: entries.len(),
                max: MAX_OBJECT_PROPERTIES,
            });
        }
        for (k, v) in entries {
            prop_names.insert(k.clone());
            properties.push((k.clone(), parse_schema_node(v, depth + 1)?));
        }
    }

    let additional_properties = match val.get("additionalProperties") {
        Some(v) => v.as_bool().ok_or_else(|| {
            SchemaError::UnsupportedSchema("'additionalProperties' must be a boolean".into())
        })?,
        None => true,
    };

    let mut required = Vec::new();
    if let Some(req_val) = val.get("required") {
        let items = req_val
            .as_array()
            .ok_or_else(|| SchemaError::UnsupportedSchema("'required' must be an array".into()))?;
        for item in items {
            let key = item.as_str().ok_or_else(|| {
                SchemaError::UnsupportedSchema("'required' entries must be strings".into())
            })?;
            if !additional_properties && !prop_names.contains(key) {
                return Err(SchemaError::MissingRequiredProperty {
                    property: key.to_string(),
                });
            }
            required.push(key.to_string());
        }
    }

    Ok(JsonSchema::Object {
        properties,
        required,
        additional_properties,
    })
}

/// Independent JSON Schema value validator (never relies on the PDA!).
fn validate_node(
    schema: &JsonSchema,
    value: &JsonValue,
    defs: &HashMap<&str, &JsonSchema>,
    depth: usize,
) -> Result<(), String> {
    if depth > 128 {
        return Err("validation recursion depth exceeded".into());
    }
    match schema {
        JsonSchema::Any => Ok(()),
        JsonSchema::Null => match value {
            JsonValue::Null => Ok(()),
            _ => Err("expected null".into()),
        },
        JsonSchema::Boolean => match value {
            JsonValue::Bool(_) => Ok(()),
            _ => Err("expected boolean".into()),
        },
        JsonSchema::Integer => match value {
            JsonValue::Int(_) => Ok(()),
            _ => Err("expected integer".into()),
        },
        JsonSchema::Number => match value {
            JsonValue::Int(_) | JsonValue::Float(_) => Ok(()),
            _ => Err("expected number".into()),
        },
        JsonSchema::String => match value {
            JsonValue::String(_) => Ok(()),
            _ => Err("expected string".into()),
        },
        JsonSchema::Const(expected) => {
            if value == expected {
                Ok(())
            } else {
                Err(format!(
                    "expected const {}, got {}",
                    expected.to_json_string(),
                    value.to_json_string()
                ))
            }
        }
        JsonSchema::Enum(variants) => {
            if variants.iter().any(|v| v == value) {
                Ok(())
            } else {
                Err(format!("value {} not in enum", value.to_json_string()))
            }
        }
        JsonSchema::Array {
            items,
            min_items,
            max_items,
        } => {
            let Some(arr) = value.as_array() else {
                return Err("expected array".into());
            };
            if arr.len() < *min_items {
                return Err(format!(
                    "array length {} < minItems {}",
                    arr.len(),
                    min_items
                ));
            }
            if let Some(mx) = max_items {
                if arr.len() > *mx {
                    return Err(format!("array length {} > maxItems {}", arr.len(), mx));
                }
            }
            for (idx, item) in arr.iter().enumerate() {
                validate_node(items, item, defs, depth + 1)
                    .map_err(|e| format!("array[{idx}]: {e}"))?;
            }
            Ok(())
        }
        JsonSchema::Object {
            properties,
            required,
            additional_properties,
        } => {
            let Some(entries) = value.as_object() else {
                return Err("expected object".into());
            };
            let mut seen_keys = HashSet::new();
            for (k, _) in entries {
                if !seen_keys.insert(k.as_str()) {
                    return Err(format!("duplicate object key '{k}'"));
                }
            }
            for req in required {
                if !seen_keys.contains(req.as_str()) {
                    return Err(format!("missing required property '{req}'"));
                }
            }
            let prop_map: HashMap<&str, &JsonSchema> =
                properties.iter().map(|(k, v)| (k.as_str(), v)).collect();
            for (k, v) in entries {
                if let Some(sub_schema) = prop_map.get(k.as_str()) {
                    validate_node(sub_schema, v, defs, depth + 1)
                        .map_err(|e| format!("property '{k}': {e}"))?;
                } else if !additional_properties {
                    return Err(format!(
                        "unexpected additional property '{k}' (additionalProperties: false)"
                    ));
                }
            }
            Ok(())
        }
        JsonSchema::AnyOf(branches) => {
            for branch in branches {
                if validate_node(branch, value, defs, depth + 1).is_ok() {
                    return Ok(());
                }
            }
            Err("value matched none of anyOf branches".into())
        }
        JsonSchema::Ref(name) => {
            let target = defs
                .get(name.as_str())
                .ok_or_else(|| format!("unresolved $ref '{name}'"))?;
            validate_node(target, value, defs, depth + 1)
        }
    }
}

// ---------------------------------------------------------------------------
// Schema -> Byte-Level PDA Compiler (with Fused-Whitespace Optimization)
// ---------------------------------------------------------------------------

struct SchemaCompiler {
    b: GrammarBuilder,
    ws_bytes: ByteSet,
    // Deduplicated shared primitive rules
    ws_string: RuleId,
    ws_integer: RuleId,
    ws_number: RuleId,
    ws_boolean: RuleId,
    ws_null: RuleId,
    ws_colon: RuleId,
    ws_comma: RuleId,
    ws_close_brace: RuleId,
    ws_close_bracket: RuleId,
    ws_any: Option<RuleId>,
    def_rules: HashMap<String, RuleId>,
    counter: usize,
}

impl SchemaCompiler {
    fn compile(doc: &SchemaDocument) -> Result<Grammar, SchemaError> {
        let mut ws_bytes = ByteSet::empty();
        ws_bytes.insert(b' ');
        ws_bytes.insert(b'\t');
        ws_bytes.insert(b'\n');
        ws_bytes.insert(b'\r');

        let mut b = GrammarBuilder::new();

        // 1. StringBody: full RFC 8259 JSON escapes + full RFC 3629 UTF-8 with 100% TCO!
        let string_body = build_string_body_rule(&mut b)?;
        let ws_string = build_ws_prefixed_rule(
            &mut b,
            "ws_string",
            ws_bytes,
            vec![vec![
                Element::Byte(ByteSet::from_byte(b'"')),
                Element::RuleRef(string_body),
            ]],
        )?;

        // 2. Integer & Number (RFC 8259 compliant, no leading zeros, TCO digits)
        let ws_integer = build_ws_integer_rule(&mut b, ws_bytes)?;
        let ws_number = build_ws_number_rule(&mut b, ws_bytes)?;

        // 3. Boolean & Null
        let ws_boolean = build_ws_prefixed_rule(
            &mut b,
            "ws_boolean",
            ws_bytes,
            vec![
                GrammarBuilder::literal_elements(b"true"),
                GrammarBuilder::literal_elements(b"false"),
            ],
        )?;
        let ws_null = build_ws_prefixed_rule(
            &mut b,
            "ws_null",
            ws_bytes,
            vec![GrammarBuilder::literal_elements(b"null")],
        )?;

        // 4. Fused-whitespace structural delimiters
        let ws_colon = build_ws_byte_rule(&mut b, "ws_colon", ws_bytes, b':')?;
        let ws_comma = build_ws_byte_rule(&mut b, "ws_comma", ws_bytes, b',')?;
        let ws_close_brace = build_ws_byte_rule(&mut b, "ws_close_brace", ws_bytes, b'}')?;
        let ws_close_bracket = build_ws_byte_rule(&mut b, "ws_close_bracket", ws_bytes, b']')?;

        let mut compiler = Self {
            b,
            ws_bytes,
            ws_string,
            ws_integer,
            ws_number,
            ws_boolean,
            ws_null,
            ws_colon,
            ws_comma,
            ws_close_brace,
            ws_close_bracket,
            ws_any: None,
            def_rules: HashMap::new(),
            counter: 0,
        };

        // Pre-declare all $defs so recursive $ref can bind immediately.
        for (name, _) in &doc.defs {
            let rid = compiler.b.declare_rule(format!("def_{name}"))?;
            compiler.def_rules.insert(name.clone(), rid);
        }

        // Compile each $def body into its pre-declared RuleId.
        for (name, schema) in &doc.defs {
            let target_rid = compiler.def_rules[name];
            let body_rid = compiler.compile_node(schema, 0)?;
            compiler
                .b
                .add_alt(target_rid, vec![Element::RuleRef(body_rid)])?;
        }

        let root_rule = compiler.compile_node(&doc.root, 0)?;
        let grammar = compiler.b.finalize(root_rule)?;
        Ok(grammar)
    }

    fn fresh_name(&mut self, prefix: &str) -> String {
        let id = self.counter;
        self.counter += 1;
        format!("{prefix}_{id}")
    }

    fn get_or_build_ws_any(&mut self) -> Result<RuleId, SchemaError> {
        if let Some(rid) = self.ws_any {
            return Ok(rid);
        }
        let any_val = self.b.declare_rule("ws_any_value")?;
        self.ws_any = Some(any_val);

        // Generic JSON array of ws_any_value
        let any_arr_tail = self.b.declare_rule("ws_any_arr_tail")?;
        self.b
            .add_alt(any_arr_tail, vec![Element::RuleRef(self.ws_close_bracket)])?;
        self.b.add_alt(
            any_arr_tail,
            vec![
                Element::RuleRef(self.ws_comma),
                Element::RuleRef(any_val),
                Element::RuleRef(any_arr_tail),
            ],
        )?;
        let any_arr = build_ws_prefixed_rule(
            &mut self.b,
            "ws_any_arr",
            self.ws_bytes,
            vec![
                vec![
                    Element::Byte(ByteSet::from_byte(b'[')),
                    Element::RuleRef(self.ws_close_bracket),
                ],
                vec![
                    Element::Byte(ByteSet::from_byte(b'[')),
                    Element::RuleRef(any_val),
                    Element::RuleRef(any_arr_tail),
                ],
            ],
        )?;

        // Generic JSON object of "string": ws_any_value
        let any_obj_tail = self.b.declare_rule("ws_any_obj_tail")?;
        self.b
            .add_alt(any_obj_tail, vec![Element::RuleRef(self.ws_close_brace)])?;
        self.b.add_alt(
            any_obj_tail,
            vec![
                Element::RuleRef(self.ws_comma),
                Element::RuleRef(self.ws_string),
                Element::RuleRef(self.ws_colon),
                Element::RuleRef(any_val),
                Element::RuleRef(any_obj_tail),
            ],
        )?;
        let any_obj = build_ws_prefixed_rule(
            &mut self.b,
            "ws_any_obj",
            self.ws_bytes,
            vec![
                vec![
                    Element::Byte(ByteSet::from_byte(b'{')),
                    Element::RuleRef(self.ws_close_brace),
                ],
                vec![
                    Element::Byte(ByteSet::from_byte(b'{')),
                    Element::RuleRef(self.ws_string),
                    Element::RuleRef(self.ws_colon),
                    Element::RuleRef(any_val),
                    Element::RuleRef(any_obj_tail),
                ],
            ],
        )?;

        self.b
            .add_alt(any_val, vec![Element::RuleRef(self.ws_string)])?;
        self.b
            .add_alt(any_val, vec![Element::RuleRef(self.ws_number)])?;
        self.b
            .add_alt(any_val, vec![Element::RuleRef(self.ws_boolean)])?;
        self.b
            .add_alt(any_val, vec![Element::RuleRef(self.ws_null)])?;
        self.b.add_alt(any_val, vec![Element::RuleRef(any_arr)])?;
        self.b.add_alt(any_val, vec![Element::RuleRef(any_obj)])?;
        Ok(any_val)
    }

    fn compile_node(&mut self, node: &JsonSchema, depth: usize) -> Result<RuleId, SchemaError> {
        if depth > MAX_SCHEMA_DEPTH {
            return Err(SchemaError::DepthLimitExceeded {
                depth,
                max: MAX_SCHEMA_DEPTH,
            });
        }
        match node {
            JsonSchema::Null => Ok(self.ws_null),
            JsonSchema::Boolean => Ok(self.ws_boolean),
            JsonSchema::Integer => Ok(self.ws_integer),
            JsonSchema::Number => Ok(self.ws_number),
            JsonSchema::String => Ok(self.ws_string),
            JsonSchema::Any => self.get_or_build_ws_any(),
            JsonSchema::Ref(name) => {
                self.def_rules
                    .get(name)
                    .copied()
                    .ok_or_else(|| SchemaError::UnresolvedRef {
                        reference: name.clone(),
                    })
            }
            JsonSchema::Const(val) => {
                let lit = val.to_json_string();
                let name = self.fresh_name("ws_const");
                let rid = build_ws_prefixed_rule(
                    &mut self.b,
                    &name,
                    self.ws_bytes,
                    vec![GrammarBuilder::literal_elements(lit.as_bytes())],
                )?;
                Ok(rid)
            }
            JsonSchema::Enum(vals) => {
                if vals.is_empty() {
                    return Err(SchemaError::UnsupportedSchema("empty enum".into()));
                }
                if vals.len() > MAX_ENUM_VARIANTS {
                    return Err(SchemaError::VariantLimitExceeded {
                        count: vals.len(),
                        max: MAX_ENUM_VARIANTS,
                    });
                }
                let alts: Vec<Vec<Element>> = vals
                    .iter()
                    .map(|v| GrammarBuilder::literal_elements(v.to_json_string().as_bytes()))
                    .collect();
                let name = self.fresh_name("ws_enum");
                let rid = build_ws_prefixed_rule(&mut self.b, &name, self.ws_bytes, alts)?;
                Ok(rid)
            }
            JsonSchema::AnyOf(branches) => {
                if branches.is_empty() {
                    return Err(SchemaError::UnsupportedSchema("empty anyOf".into()));
                }
                if branches.len() > MAX_ENUM_VARIANTS {
                    return Err(SchemaError::VariantLimitExceeded {
                        count: branches.len(),
                        max: MAX_ENUM_VARIANTS,
                    });
                }
                let name = self.fresh_name("any_of");
                let rid = self.b.declare_rule(name)?;
                for branch in branches {
                    let sub = self.compile_node(branch, depth + 1)?;
                    self.b.add_alt(rid, vec![Element::RuleRef(sub)])?;
                }
                Ok(rid)
            }
            JsonSchema::Array {
                items,
                min_items,
                max_items,
            } => self.compile_array(items, *min_items, *max_items, depth),
            JsonSchema::Object {
                properties,
                required,
                additional_properties,
            } => self.compile_object(properties, required, *additional_properties, depth),
        }
    }

    fn compile_array(
        &mut self,
        items: &JsonSchema,
        min_items: usize,
        max_items: Option<usize>,
        depth: usize,
    ) -> Result<RuleId, SchemaError> {
        if min_items > MAX_ARRAY_UNROLL {
            return Err(SchemaError::ArrayBoundLimitExceeded {
                bound: min_items,
                max: MAX_ARRAY_UNROLL,
            });
        }
        if let Some(mx) = max_items {
            if mx > MAX_ARRAY_UNROLL {
                return Err(SchemaError::ArrayBoundLimitExceeded {
                    bound: mx,
                    max: MAX_ARRAY_UNROLL,
                });
            }
            if min_items > mx {
                return Err(SchemaError::InvalidArrayBounds {
                    min: min_items,
                    max: mx,
                });
            }
        }

        let item_rule = self.compile_node(items, depth + 1)?;
        let tail_rule = match max_items {
            None => {
                // Unbounded tail with TCO: `]` | `,` <item> <tail>
                let tname = self.fresh_name("arr_tail");
                let tail = self.b.declare_rule(tname)?;
                self.b
                    .add_alt(tail, vec![Element::RuleRef(self.ws_close_bracket)])?;
                self.b.add_alt(
                    tail,
                    vec![
                        Element::RuleRef(self.ws_comma),
                        Element::RuleRef(item_rule),
                        Element::RuleRef(tail),
                    ],
                )?;
                tail
            }
            Some(mx) => {
                // Bounded remaining items after the mandatory prefix (at least 1 if min==0).
                let first_count = min_items.max(1);
                let rem = mx.saturating_sub(first_count);
                let mut cur_tail = self.ws_close_bracket;
                for _ in 0..rem {
                    let tname = self.fresh_name("arr_btail");
                    let next_tail = self.b.declare_rule(tname)?;
                    self.b
                        .add_alt(next_tail, vec![Element::RuleRef(self.ws_close_bracket)])?;
                    self.b.add_alt(
                        next_tail,
                        vec![
                            Element::RuleRef(self.ws_comma),
                            Element::RuleRef(item_rule),
                            Element::RuleRef(cur_tail),
                        ],
                    )?;
                    cur_tail = next_tail;
                }
                cur_tail
            }
        };

        let mut alts = Vec::new();
        if min_items == 0 {
            // Empty array `[]`
            alts.push(vec![
                Element::Byte(ByteSet::from_byte(b'[')),
                Element::RuleRef(self.ws_close_bracket),
            ]);
            if max_items != Some(0) {
                alts.push(vec![
                    Element::Byte(ByteSet::from_byte(b'[')),
                    Element::RuleRef(item_rule),
                    Element::RuleRef(tail_rule),
                ]);
            }
        } else {
            let mut seq = Vec::new();
            seq.push(Element::Byte(ByteSet::from_byte(b'[')));
            seq.push(Element::RuleRef(item_rule));
            for _ in 1..min_items {
                seq.push(Element::RuleRef(self.ws_comma));
                seq.push(Element::RuleRef(item_rule));
            }
            seq.push(Element::RuleRef(tail_rule));
            alts.push(seq);
        }

        let name = self.fresh_name("ws_array");
        let rid = build_ws_prefixed_rule(&mut self.b, &name, self.ws_bytes, alts)?;
        Ok(rid)
    }

    fn compile_object(
        &mut self,
        properties: &[(String, JsonSchema)],
        required: &[String],
        additional_properties: bool,
        depth: usize,
    ) -> Result<RuleId, SchemaError> {
        if properties.len() > MAX_OBJECT_PROPERTIES {
            return Err(SchemaError::PropertyLimitExceeded {
                count: properties.len(),
                max: MAX_OBJECT_PROPERTIES,
            });
        }
        let req_set: HashSet<&str> = required.iter().map(String::as_str).collect();
        let prop_set: HashSet<&str> = properties.iter().map(|(k, _)| k.as_str()).collect();
        if !additional_properties {
            for &r in &req_set {
                if !prop_set.contains(r) {
                    return Err(SchemaError::MissingRequiredProperty {
                        property: r.to_string(),
                    });
                }
            }
        }

        // Compile each property value and its ws-prefixed quoted key rule.
        let mut compiled_props: Vec<(RuleId, RuleId, bool)> = Vec::with_capacity(properties.len());
        for (k, sub_schema) in properties {
            let quoted = json_escape_string(k);
            let kname = self.fresh_name("ws_key");
            let key_rule = build_ws_prefixed_rule(
                &mut self.b,
                &kname,
                self.ws_bytes,
                vec![GrammarBuilder::literal_elements(quoted.as_bytes())],
            )?;
            let val_rule = self.compile_node(sub_schema, depth + 1)?;
            let is_req = req_set.contains(k.as_str());
            compiled_props.push((key_rule, val_rule, is_req));
        }

        // Terminal rule after all declared properties (either `}` or optional extra props + `}`).
        let (end_no_comma, end_need_comma) = if additional_properties {
            let any_val = self.get_or_build_ws_any()?;
            let extra_tail_name = self.fresh_name("obj_extra_tail");
            let extra_tail = self.b.declare_rule(extra_tail_name)?;
            self.b
                .add_alt(extra_tail, vec![Element::RuleRef(self.ws_close_brace)])?;
            self.b.add_alt(
                extra_tail,
                vec![
                    Element::RuleRef(self.ws_comma),
                    Element::RuleRef(self.ws_string),
                    Element::RuleRef(self.ws_colon),
                    Element::RuleRef(any_val),
                    Element::RuleRef(extra_tail),
                ],
            )?;

            let extra_first_name = self.fresh_name("obj_extra_first");
            let extra_first = self.b.declare_rule(extra_first_name)?;
            self.b
                .add_alt(extra_first, vec![Element::RuleRef(self.ws_close_brace)])?;
            self.b.add_alt(
                extra_first,
                vec![
                    Element::RuleRef(self.ws_string),
                    Element::RuleRef(self.ws_colon),
                    Element::RuleRef(any_val),
                    Element::RuleRef(extra_tail),
                ],
            )?;
            (extra_first, extra_tail)
        } else {
            (self.ws_close_brace, self.ws_close_brace)
        };

        // Fast path: all properties required and no additionalProperties -> single flat sequence!
        let all_required = !compiled_props.is_empty() && compiled_props.iter().all(|(_, _, r)| *r);
        if all_required && !additional_properties {
            let mut seq = Vec::new();
            seq.push(Element::Byte(ByteSet::from_byte(b'{')));
            for (idx, &(key_rule, val_rule, _)) in compiled_props.iter().enumerate() {
                if idx > 0 {
                    seq.push(Element::RuleRef(self.ws_comma));
                }
                seq.push(Element::RuleRef(key_rule));
                seq.push(Element::RuleRef(self.ws_colon));
                seq.push(Element::RuleRef(val_rule));
            }
            seq.push(Element::RuleRef(self.ws_close_brace));
            let name = self.fresh_name("ws_obj_strict");
            let rid = build_ws_prefixed_rule(&mut self.b, &name, self.ws_bytes, vec![seq])?;
            return Ok(rid);
        }

        // General O(n) canonical-order optional-property state machine:
        // Build backwards from i = n-1 down to 0 tracking `(next_no_comma, next_need_comma)`.
        let mut next_no_comma = end_no_comma;
        let mut next_need_comma = end_need_comma;

        for &(key_rule, val_rule, is_req) in compiled_props.iter().rev() {
            let s0_name = self.fresh_name("obj_s0");
            let s0 = self.b.declare_rule(s0_name)?;
            let s1_name = self.fresh_name("obj_s1");
            let s1 = self.b.declare_rule(s1_name)?;

            // Include this property:
            self.b.add_alt(
                s0,
                vec![
                    Element::RuleRef(key_rule),
                    Element::RuleRef(self.ws_colon),
                    Element::RuleRef(val_rule),
                    Element::RuleRef(next_need_comma),
                ],
            )?;
            self.b.add_alt(
                s1,
                vec![
                    Element::RuleRef(self.ws_comma),
                    Element::RuleRef(key_rule),
                    Element::RuleRef(self.ws_colon),
                    Element::RuleRef(val_rule),
                    Element::RuleRef(next_need_comma),
                ],
            )?;

            // Skip this property if optional:
            if !is_req {
                self.b.add_alt(s0, vec![Element::RuleRef(next_no_comma)])?;
                self.b
                    .add_alt(s1, vec![Element::RuleRef(next_need_comma)])?;
            }

            next_no_comma = s0;
            next_need_comma = s1;
        }

        let name = self.fresh_name("ws_obj");
        let rid = build_ws_prefixed_rule(
            &mut self.b,
            &name,
            self.ws_bytes,
            vec![vec![
                Element::Byte(ByteSet::from_byte(b'{')),
                Element::RuleRef(next_no_comma),
            ]],
        )?;
        Ok(rid)
    }
}

/// Build a rule `[ \t\r\n]* <alts>` where whitespace is fused directly via TCO.
fn build_ws_prefixed_rule(
    b: &mut GrammarBuilder,
    name: &str,
    ws_bytes: ByteSet,
    alts: Vec<Vec<Element>>,
) -> Result<RuleId, GrammarError> {
    let rid = b.declare_rule(name)?;
    for alt in alts {
        b.add_alt(rid, alt)?;
    }
    b.add_alt(rid, vec![Element::Byte(ws_bytes), Element::RuleRef(rid)])?;
    Ok(rid)
}

fn build_ws_byte_rule(
    b: &mut GrammarBuilder,
    name: &str,
    ws_bytes: ByteSet,
    byte: u8,
) -> Result<RuleId, GrammarError> {
    build_ws_prefixed_rule(
        b,
        name,
        ws_bytes,
        vec![vec![Element::Byte(ByteSet::from_byte(byte))]],
    )
}

/// Build `StringBody` matching the remainder of a JSON string up to and including `"`,
/// supporting RFC 8259 escapes (`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, `\uXXXX`)
/// and exact RFC 3629 UTF-8 byte sequences (rejecting overlongs, surrogates, and > U+10FFFF).
///
/// Every recursive alternative ends in `Element::RuleRef(string_body)` (tail call!), so
/// strings of arbitrary length execute in O(1) PDA stack depth and non-quote tokens remain
/// 100% context-independent.
fn build_string_body_rule(b: &mut GrammarBuilder) -> Result<RuleId, GrammarError> {
    let rid = b.declare_rule("json_string_body")?;

    // Unescaped printable ASCII: 0x20..=0x7F excluding '"' (0x22) and '\\' (0x5C).
    let mut ascii_unescaped = ByteSet::from_range(0x20, 0x7F);
    ascii_unescaped.remove(b'"');
    ascii_unescaped.remove(b'\\');

    let mut simple_esc = ByteSet::empty();
    for &c in b"\"\\/bfnrt" {
        simple_esc.insert(c);
    }
    let hex_digit = ByteSet::from_range(b'0', b'9')
        .union(ByteSet::from_range(b'a', b'f'))
        .union(ByteSet::from_range(b'A', b'F'));

    let cont = ByteSet::from_range(0x80, 0xBF);
    let utf8_3_mid = ByteSet::from_range(0xE1, 0xEC).union(ByteSet::from_range(0xEE, 0xEF));

    // Alt 0: closing '"' terminates the string.
    b.add_alt(rid, vec![Element::Byte(ByteSet::from_byte(b'"'))])?;
    // Alt 1: single unescaped ASCII byte + TCO loop.
    b.add_alt(
        rid,
        vec![Element::Byte(ascii_unescaped), Element::RuleRef(rid)],
    )?;
    // Alt 2: simple JSON escape `\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`.
    b.add_alt(
        rid,
        vec![
            Element::Byte(ByteSet::from_byte(b'\\')),
            Element::Byte(simple_esc),
            Element::RuleRef(rid),
        ],
    )?;
    // Alt 3: Unicode escape `\uXXXX`.
    b.add_alt(
        rid,
        vec![
            Element::Byte(ByteSet::from_byte(b'\\')),
            Element::Byte(ByteSet::from_byte(b'u')),
            Element::Byte(hex_digit),
            Element::Byte(hex_digit),
            Element::Byte(hex_digit),
            Element::Byte(hex_digit),
            Element::RuleRef(rid),
        ],
    )?;
    // Alt 4: 2-byte UTF-8 [C2..DF][80..BF].
    b.add_alt(
        rid,
        vec![
            Element::Byte(ByteSet::from_range(0xC2, 0xDF)),
            Element::Byte(cont),
            Element::RuleRef(rid),
        ],
    )?;
    // Alt 5: 3-byte UTF-8 E0 [A0..BF][80..BF].
    b.add_alt(
        rid,
        vec![
            Element::Byte(ByteSet::from_byte(0xE0)),
            Element::Byte(ByteSet::from_range(0xA0, 0xBF)),
            Element::Byte(cont),
            Element::RuleRef(rid),
        ],
    )?;
    // Alt 6: 3-byte UTF-8 [E1..EC, EE..EF][80..BF][80..BF].
    b.add_alt(
        rid,
        vec![
            Element::Byte(utf8_3_mid),
            Element::Byte(cont),
            Element::Byte(cont),
            Element::RuleRef(rid),
        ],
    )?;
    // Alt 7: 3-byte UTF-8 ED [80..9F][80..BF] (excludes UTF-16 surrogates U+D800..U+DFFF).
    b.add_alt(
        rid,
        vec![
            Element::Byte(ByteSet::from_byte(0xED)),
            Element::Byte(ByteSet::from_range(0x80, 0x9F)),
            Element::Byte(cont),
            Element::RuleRef(rid),
        ],
    )?;
    // Alt 8: 4-byte UTF-8 F0 [90..BF][80..BF][80..BF].
    b.add_alt(
        rid,
        vec![
            Element::Byte(ByteSet::from_byte(0xF0)),
            Element::Byte(ByteSet::from_range(0x90, 0xBF)),
            Element::Byte(cont),
            Element::Byte(cont),
            Element::RuleRef(rid),
        ],
    )?;
    // Alt 9: 4-byte UTF-8 [F1..F3][80..BF]^3.
    b.add_alt(
        rid,
        vec![
            Element::Byte(ByteSet::from_range(0xF1, 0xF3)),
            Element::Byte(cont),
            Element::Byte(cont),
            Element::Byte(cont),
            Element::RuleRef(rid),
        ],
    )?;
    // Alt 10: 4-byte UTF-8 F4 [80..8F][80..BF]^2.
    b.add_alt(
        rid,
        vec![
            Element::Byte(ByteSet::from_byte(0xF4)),
            Element::Byte(ByteSet::from_range(0x80, 0x8F)),
            Element::Byte(cont),
            Element::Byte(cont),
            Element::RuleRef(rid),
        ],
    )?;

    Ok(rid)
}

fn build_ws_integer_rule(
    b: &mut GrammarBuilder,
    ws_bytes: ByteSet,
) -> Result<RuleId, GrammarError> {
    let digit = ByteSet::from_range(b'0', b'9');
    let nonzero = ByteSet::from_range(b'1', b'9');

    let digits_tail = b.declare_rule("int_digits_tail")?;
    b.add_alt(digits_tail, vec![])?;
    b.add_alt(
        digits_tail,
        vec![Element::Byte(digit), Element::RuleRef(digits_tail)],
    )?;

    let uint_body = b.declare_rule("uint_body")?;
    b.add_alt(uint_body, vec![Element::Byte(ByteSet::from_byte(b'0'))])?;
    b.add_alt(
        uint_body,
        vec![Element::Byte(nonzero), Element::RuleRef(digits_tail)],
    )?;

    build_ws_prefixed_rule(
        b,
        "ws_integer",
        ws_bytes,
        vec![
            vec![Element::Byte(ByteSet::from_byte(b'0'))],
            vec![Element::Byte(nonzero), Element::RuleRef(digits_tail)],
            vec![
                Element::Byte(ByteSet::from_byte(b'-')),
                Element::RuleRef(uint_body),
            ],
        ],
    )
}

fn build_ws_number_rule(b: &mut GrammarBuilder, ws_bytes: ByteSet) -> Result<RuleId, GrammarError> {
    let digit = ByteSet::from_range(b'0', b'9');
    let nonzero = ByteSet::from_range(b'1', b'9');
    let mut exp_char = ByteSet::empty();
    exp_char.insert(b'e');
    exp_char.insert(b'E');
    let mut sign_char = ByteSet::empty();
    sign_char.insert(b'+');
    sign_char.insert(b'-');

    let digits_tail = b.declare_rule("num_digits_tail")?;
    b.add_alt(digits_tail, vec![])?;
    b.add_alt(
        digits_tail,
        vec![Element::Byte(digit), Element::RuleRef(digits_tail)],
    )?;

    let exp_part = b.declare_rule("num_exp_part")?;
    b.add_alt(exp_part, vec![])?;
    b.add_alt(
        exp_part,
        vec![
            Element::Byte(exp_char),
            Element::Byte(digit),
            Element::RuleRef(digits_tail),
        ],
    )?;
    b.add_alt(
        exp_part,
        vec![
            Element::Byte(exp_char),
            Element::Byte(sign_char),
            Element::Byte(digit),
            Element::RuleRef(digits_tail),
        ],
    )?;

    let frac_or_exp = b.declare_rule("num_frac_or_exp")?;
    b.add_alt(frac_or_exp, vec![Element::RuleRef(exp_part)])?;
    b.add_alt(
        frac_or_exp,
        vec![
            Element::Byte(ByteSet::from_byte(b'.')),
            Element::Byte(digit),
            Element::RuleRef(digits_tail),
            Element::RuleRef(exp_part),
        ],
    )?;

    let int_part_tail = b.declare_rule("num_int_tail")?;
    b.add_alt(int_part_tail, vec![Element::RuleRef(frac_or_exp)])?;
    b.add_alt(
        int_part_tail,
        vec![Element::Byte(digit), Element::RuleRef(int_part_tail)],
    )?;

    let unsigned_num = b.declare_rule("num_unsigned")?;
    b.add_alt(
        unsigned_num,
        vec![
            Element::Byte(ByteSet::from_byte(b'0')),
            Element::RuleRef(frac_or_exp),
        ],
    )?;
    b.add_alt(
        unsigned_num,
        vec![Element::Byte(nonzero), Element::RuleRef(int_part_tail)],
    )?;

    build_ws_prefixed_rule(
        b,
        "ws_number",
        ws_bytes,
        vec![
            vec![
                Element::Byte(ByteSet::from_byte(b'0')),
                Element::RuleRef(frac_or_exp),
            ],
            vec![Element::Byte(nonzero), Element::RuleRef(int_part_tail)],
            vec![
                Element::Byte(ByteSet::from_byte(b'-')),
                Element::RuleRef(unsigned_num),
            ],
        ],
    )
}

fn mix_u64(h: &mut u64, v: u64) {
    *h ^= v;
    *h = h.wrapping_mul(0x100000001b3);
}

fn mix_str(h: &mut u64, s: &str) {
    mix_u64(h, s.len() as u64);
    for &b in s.as_bytes() {
        mix_u64(h, b as u64);
    }
}

fn hash_schema_doc(doc: &SchemaDocument, h: &mut u64) {
    mix_u64(h, doc.defs.len() as u64);
    for (k, v) in &doc.defs {
        mix_str(h, k);
        hash_schema_node(v, h);
    }
    hash_schema_node(&doc.root, h);
}

fn hash_schema_node(node: &JsonSchema, h: &mut u64) {
    match node {
        JsonSchema::Null => mix_u64(h, 1),
        JsonSchema::Boolean => mix_u64(h, 2),
        JsonSchema::Integer => mix_u64(h, 3),
        JsonSchema::Number => mix_u64(h, 4),
        JsonSchema::String => mix_u64(h, 5),
        JsonSchema::Any => mix_u64(h, 6),
        JsonSchema::Const(v) => {
            mix_u64(h, 7);
            mix_str(h, &v.to_json_string());
        }
        JsonSchema::Enum(vs) => {
            mix_u64(h, 8);
            mix_u64(h, vs.len() as u64);
            for v in vs {
                mix_str(h, &v.to_json_string());
            }
        }
        JsonSchema::Array {
            items,
            min_items,
            max_items,
        } => {
            mix_u64(h, 9);
            mix_u64(h, *min_items as u64);
            mix_u64(h, max_items.unwrap_or(usize::MAX) as u64);
            hash_schema_node(items, h);
        }
        JsonSchema::Object {
            properties,
            required,
            additional_properties,
        } => {
            mix_u64(h, 10);
            mix_u64(h, *additional_properties as u64);
            mix_u64(h, properties.len() as u64);
            for (k, v) in properties {
                mix_str(h, k);
                hash_schema_node(v, h);
            }
            mix_u64(h, required.len() as u64);
            for r in required {
                mix_str(h, r);
            }
        }
        JsonSchema::AnyOf(bs) => {
            mix_u64(h, 11);
            mix_u64(h, bs.len() as u64);
            for b in bs {
                hash_schema_node(b, h);
            }
        }
        JsonSchema::Ref(r) => {
            mix_u64(h, 12);
            mix_str(h, r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{PdaConfig, PdaState};

    fn accepts(grammar: &Grammar, input: &str) -> bool {
        let mut st = PdaState::new(grammar, PdaConfig::default());
        st.advance_bytes(grammar, input.as_bytes()) && st.can_accept()
    }

    #[test]
    fn compiles_and_enforces_strict_object_schema() {
        let schema_str = r#"{
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ok", "error"]},
                "count": {"type": "integer"},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
                "note": {"type": "string"}
            },
            "required": ["status", "count", "score", "active"],
            "additionalProperties": false
        }"#;
        let doc = SchemaDocument::from_json_str(schema_str).unwrap();
        let g = doc.compile_grammar().unwrap();

        let valid1 = r#"{"status": "ok", "count": 42, "score": -3.14e2, "active": true}"#;
        let valid2 = r#"{"status":"error","count":0,"score":1.0,"active":false,"note":"hi\n\"\\日本語🚀\u0041"}"#;
        assert!(accepts(&g, valid1));
        assert!(doc.validate_json_str(valid1).is_ok());
        assert!(accepts(&g, valid2));
        assert!(doc.validate_json_str(valid2).is_ok());

        // Rejects wrong enum, extra property, missing required, leading zero, raw newline, bad UTF-8
        assert!(!accepts(
            &g,
            r#"{"status":"unknown","count":1,"score":1.0,"active":true}"#
        ));
        assert!(!accepts(
            &g,
            r#"{"status":"ok","count":1,"score":1.0,"active":true,"extra":1}"#
        ));
        assert!(!accepts(
            &g,
            r#"{"status":"ok","count":01,"score":1.0,"active":true}"#
        ));
        assert!(!accepts(
            &g,
            "{\"status\":\"ok\",\"count\":1,\"score\":1.0,\"active\":true,\"note\":\"bad\nline\"}"
        ));
    }

    #[test]
    fn rejects_invalid_utf8_and_surrogates_in_json_strings() {
        let doc = SchemaDocument::from_json_str(r#"{"type":"string"}"#).unwrap();
        let g = doc.compile_grammar().unwrap();

        assert!(accepts(&g, "\"hello 日本語 🚀 café \\\" \\n \\u00e9\""));
        // Invalid raw UTF-8 bytes inside quotes: overlong 0xC0 0x80, stray 0x80, UTF-16 surrogate 0xED 0xA0 0x80
        let mut st = PdaState::new(&g, PdaConfig::default());
        assert!(!st.advance_bytes(&g, b"\"\xC0\x80\""));
        let mut st2 = PdaState::new(&g, PdaConfig::default());
        assert!(!st2.advance_bytes(&g, b"\"\xED\xA0\x80\""));
        let mut st3 = PdaState::new(&g, PdaConfig::default());
        assert!(!st3.advance_bytes(&g, b"\"\\q\""));
    }

    #[test]
    fn compiles_recursive_ast_schema_and_mcp_tool_calls() {
        let ast_schema = r##"{
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["lit", "add"]},
                        "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}
                    },
                    "required": ["kind", "children"],
                    "additionalProperties": false
                }
            },
            "$ref": "#/$defs/Node"
        }"##;
        let doc = SchemaDocument::from_json_str(ast_schema).unwrap();
        let g = doc.compile_grammar().unwrap();
        let tree = r#"{"kind":"add","children":[{"kind":"lit","children":[]},{"kind":"lit","children":[]}]}"#;
        assert!(accepts(&g, tree));
        assert!(doc.validate_json_str(tree).is_ok());

        let tool_doc = build_tool_call_schema(
            &[
                ToolSpec::new(
                    "get_weather",
                    JsonSchema::Object {
                        properties: vec![("city".into(), JsonSchema::String)],
                        required: vec!["city".into()],
                        additional_properties: false,
                    },
                ),
                ToolSpec::new(
                    "add",
                    JsonSchema::Object {
                        properties: vec![
                            ("a".into(), JsonSchema::Integer),
                            ("b".into(), JsonSchema::Integer),
                        ],
                        required: vec!["a".into(), "b".into()],
                        additional_properties: false,
                    },
                ),
            ],
            ToolEnvelopeFormat::ToolAndArguments,
        )
        .unwrap();
        let tg = tool_doc.compile_grammar().unwrap();
        let call = r#"{"tool":"get_weather","arguments":{"city":"San Francisco"}}"#;
        assert!(accepts(&tg, call));
        assert!(tool_doc.validate_json_str(call).is_ok());
        // Cross-tool argument mismatch must be rejected!
        assert!(!accepts(
            &tg,
            r#"{"tool":"get_weather","arguments":{"a":1,"b":2}}"#
        ));
    }

    #[test]
    fn rejects_dos_and_malformed_schemas() {
        // Missing required property when additionalProperties is false
        let bad_req = r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["b"],"additionalProperties":false}"#;
        assert!(matches!(
            SchemaDocument::from_json_str(bad_req),
            Err(SchemaError::MissingRequiredProperty { .. })
        ));

        // Direct cyclic $ref (unguarded left recursion)
        let bad_cycle = r##"{"$defs":{"A":{"$ref":"#/$defs/A"}},"$ref":"#/$defs/A"}"##;
        let cycle_doc = SchemaDocument::from_json_str(bad_cycle).unwrap();
        assert!(matches!(
            cycle_doc.compile_grammar(),
            Err(SchemaError::Grammar(
                GrammarError::LeftRecursionDetected { .. }
            ))
        ));

        // Invalid array bounds
        let bad_arr = r#"{"type":"array","items":{"type":"integer"},"minItems":5,"maxItems":2}"#;
        assert!(matches!(
            SchemaDocument::from_json_str(bad_arr),
            Err(SchemaError::InvalidArrayBounds { min: 5, max: 2 })
        ));

        // Unresolved ref
        let bad_ref = r##"{"$ref":"#/$defs/DoesNotExist"}"##;
        let ref_doc = SchemaDocument::from_json_str(bad_ref).unwrap();
        assert!(matches!(
            ref_doc.compile_grammar(),
            Err(SchemaError::UnresolvedRef { .. })
        ));

        // Excess nesting depth
        let mut deep = r#"{"type":"string"}"#.to_string();
        for _ in 0..(MAX_SCHEMA_DEPTH + 4) {
            deep = format!(r#"{{"type":"array","items":{deep}}}"#);
        }
        assert!(matches!(
            SchemaDocument::from_json_str(&deep),
            Err(SchemaError::DepthLimitExceeded { .. })
        ));
    }
}
