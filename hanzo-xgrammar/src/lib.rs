//! # `hanzo-xgrammar` — Compiled Grammar Masks for Constrained Generation (HIP-1346)
//!
//! Clean-room, zero-dependency (`std`-only, `#![forbid(unsafe_code)]`, `Apache-2.0`)
//! implementation of XGrammar / XGrammar-2 compiled pushdown-automaton (PDA) grammar
//! masks for structured text generation in Hanzo **Zen** (`hanzo-engine`).
//!
//! > **Architectural boundary**: This crate accelerates structured text generation
//! > (JSON Schema outputs and OpenAI/MCP tool-call envelopes) inside **Zen**.
//! > **Kai** already produces structured decisions without text generation and is
//! > neither modified nor replaced.
//!
//! ## Modules
//! - [`json`]: Strict RFC 8259 JSON parser, ordered AST, and serializer (used both
//!   for schema ingestion and independent output validation).
//! - [`grammar`]: Byte-level Pushdown Automaton (PDA) IR with tail-call optimization
//!   (TCO), compile-time left-recursion rejection, rollback checkpoints, and bounded
//!   stack/branch limits.
//! - [`schema`]: JSON Schema (`object`, `properties`, `required`,
//!   `additionalProperties`, `array`, `string`, `enum`, `const`, `integer`, `number`,
//!   `boolean`, `null`, `anyOf`, `$defs`/`$ref`) and OpenAI/MCP tool-call compiler
//!   with fused-whitespace optimization and full RFC 3629 UTF-8 + JSON escape support.
//! - [`vocab`]: Vocabulary management, HuggingFace `tokenizer.json` loader, real-corpus
//!   BPE vocabulary constructor, and compact [`vocab::VocabTrie`].
//! - [`mask`]: XGrammar 4-way context-independent / context-dependent token mask
//!   compiler, sparse-vs-dense [`mask::CompactTokenSet`], [`mask::XGrammarMatcher`],
//!   in-place `-inf` logits masking, and the matched [`mask::NaivePerStepConstrainedDecoder`]
//!   baseline.
//! - [`cache`]: Thread-safe adaptive LRU [`cache::GrammarCache`] bounded by entries
//!   and memory budget.

#![forbid(unsafe_code)]

pub mod cache;
pub mod grammar;
pub mod json;
pub mod mask;
pub mod schema;
pub mod vocab;

pub use cache::{CacheStats, GrammarCache, GrammarCacheKey};
pub use grammar::{
    ByteSet, Element, Grammar, GrammarBuilder, GrammarError, LeafId, PdaCheckpoint, PdaConfig,
    PdaState, RuleId, RulePos, DEFAULT_MAX_ACTIVE_STACKS, DEFAULT_MAX_RECURSION_DEPTH,
};
pub use json::{parse_json, JsonError, JsonValue};
pub use mask::{
    BitMask, CompactTokenSet, CompileStats, CompiledGrammar, NaivePerStepConstrainedDecoder,
    PositionMask, StepMaskStats, XGrammarMatcher,
};
pub use schema::{
    build_tool_call_schema, JsonSchema, SchemaDocument, SchemaError, ToolEnvelopeFormat, ToolSpec,
};
pub use vocab::{VocabStats, VocabTrie, Vocabulary};
