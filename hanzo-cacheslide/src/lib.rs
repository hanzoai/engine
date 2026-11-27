// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Clean-room **CacheSlide** position-shifted KV segment reuse engine (`HIP-1346` / `hanzoai/engine#85`).
//!
//! # Overview
//! Ordinary prefix caching (`hanzo-engine/src/prefix_cacher.rs`, `hanzo-engine/src/paged_attention/block_hash.rs`)
//! hashes KV blocks in a prefix chain from token 0. Whenever an earlier prompt segment changes
//! (dynamic system prompt metadata, inserted user profile, reordered MCP tool definitions, or
//! reranked RAG chunks), every downstream KV block misses 100% even if its token bytes are unchanged.
//!
//! `hanzo-cacheslide` eliminates this brittle prefix dependency using four zero-dependency
//! pure-Rust components:
//! - [`rope`]: Exact RoPE relative shift $R(\Delta p) = R(p_{\text{new}} - p_{\text{orig}})$
//!   applied directly to cached K blocks (`GptNeoX` and `Interleaved` layouts, partial RoPE,
//!   linear/YaRN frequency scaling) with exact group composition $R(a)R(b) = R(a+b)$.
//! - [`segment`]: Content-addressed segment keys scoped strictly by
//!   `(tenant_id, acl_group, model_id, kv_dtype, segment_sha256)` with FIPS 180-4 SHA-256
//!   integrity digests (`0.0%` cross-tenant KV reuse).
//! - [`cache`]: Thread-safe bounded LRU KV pools ([`cache::SegmentKvCache`] and matched
//!   [`cache::PrefixKvCache`]) with exact byte accounting (`current_bytes <= max_bytes`),
//!   minimum segment admission length (`min_segment_tokens`), and instant `evict_tenant`.
//! - [`attention`]: Real multi-layer causal GQA + RoPE + SwiGLU Transformer engine comparing
//!   exact `FullPrefill`, `PrefixCacheOnly`, `CacheSlidePureShift`, and
//!   `CacheSlideSelectiveRecompute` (boundary sink + Layer-0 cross-segment affinity selection +
//!   automatic Layer-1 K-drift escalation gate).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod attention;
pub mod cache;
pub mod rope;
pub mod segment;
pub mod sha256;

pub use attention::{
    evaluate_quality, ExecutionMode, PrefillOutput, QualityMetrics, SelectiveRecomputeConfig,
    TransformerConfig, TransformerModel,
};
pub use cache::{CacheStats, InsertOutcome, PrefixKvCache, SegmentKvCache};
pub use rope::{
    apply_inverse_rope_in_place, apply_rope_in_place, checked_position_shift, RopeCache,
    RopeConfig, RopeLayout, RopeShiftTable,
};
pub use segment::{
    chain_prefix_hash, hash_tokens, CorpusTokenizer, IsolationScope, KvDtype, PrefixChainKey,
    PromptSegment, SegmentKey, SegmentKind, SegmentKvPayload,
};
pub use sha256::{hex_encode, Digest, Sha256};
