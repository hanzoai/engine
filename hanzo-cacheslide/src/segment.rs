// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Prompt segmentation, content-addressed KV keys, tenant/ACL isolation, and KV payloads.
//!
//! Every cached KV segment is strictly scoped to an [`IsolationScope`]:
//! `(tenant_id, acl_group, model_id, kv_dtype)`.
//!
//! Keys use length-prefixed, domain-separated SHA-256 digests combined with full
//! field equality on `(tenant_id, acl_group, model_id, kv_dtype, segment_sha256)`,
//! preventing both cross-tenant KV leakage and hash-collision aliasing.

use crate::sha256::{Digest, Sha256};

const SEGMENT_TOKEN_DOMAIN: &[u8] = b"hanzo-cacheslide-v1:segment-tokens";
const SEGMENT_KEY_DOMAIN: &[u8] = b"hanzo-cacheslide-v1:segment-key";
const PREFIX_CHAIN_DOMAIN: &[u8] = b"hanzo-cacheslide-v1:prefix-chain";
const PAYLOAD_DIGEST_DOMAIN: &[u8] = b"hanzo-cacheslide-v1:kv-payload";

/// KV cache storage precision used for GPU memory footprint accounting and dtype isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvDtype {
    /// 32-bit IEEE 754 float (4 bytes / element).
    F32,
    /// 16-bit IEEE 754 float (2 bytes / element).
    F16,
    /// 16-bit Brain float (2 bytes / element).
    BF16,
    /// 8-bit E4M3 float (1 byte / element).
    Fp8E4M3,
    /// 4-bit asymmetric quantized KV (KIVI-4bit, 0.5 bytes / element + group scale overhead).
    Int4Asym,
}

impl KvDtype {
    /// Stable canonical string tag included in key hashing and JSON receipts.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::Fp8E4M3 => "fp8_e4m3",
            Self::Int4Asym => "int4_asym",
        }
    }

    /// Compute GPU KV cache storage bytes for `num_tokens` across `num_layers` and `num_kv_heads`
    /// (covering both K and V tensors).
    #[must_use]
    pub const fn kv_bytes(
        &self,
        num_layers: usize,
        num_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> usize {
        // Total scalar elements across K and V (factor of 2).
        let elems = 2 * num_layers * num_tokens * num_kv_heads * head_dim;
        match self {
            Self::F32 => elems * 4,
            Self::F16 | Self::BF16 => elems * 2,
            Self::Fp8E4M3 => elems,
            // 4 bits/elem + ~6.25% group-32 FP16 scale/zero overhead.
            Self::Int4Asym => elems.div_ceil(2) + (elems / 32) * 4,
        }
    }
}

/// Hard multi-tenant, ACL, model, and precision isolation boundary.
///
/// Two requests can reuse cached KV segments if and only if their [`IsolationScope`]
/// values are completely identical.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IsolationScope {
    /// Tenant / organization identifier (mirrors `ExtraHashKey::CacheSalt`).
    pub tenant_id: String,
    /// Role or ACL group within the tenant (e.g., `"acl_eng"`, `"acl_finance"`).
    pub acl_group: String,
    /// Serving model identifier / weight revision (e.g., `"zen-7b-v1"`).
    pub model_id: String,
    /// KV tensor dtype so quantized and full-precision KV entries never alias.
    pub kv_dtype: KvDtype,
}

impl IsolationScope {
    /// Construct a new isolation scope.
    #[must_use]
    pub fn new(
        tenant_id: impl Into<String>,
        acl_group: impl Into<String>,
        model_id: impl Into<String>,
        kv_dtype: KvDtype,
    ) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            acl_group: acl_group.into(),
            model_id: model_id.into(),
            kv_dtype,
        }
    }

    fn feed_into(&self, hasher: &mut Sha256) {
        hasher.update_len_prefixed(self.tenant_id.as_bytes());
        hasher.update_len_prefixed(self.acl_group.as_bytes());
        hasher.update_len_prefixed(self.model_id.as_bytes());
        hasher.update_len_prefixed(self.kv_dtype.as_str().as_bytes());
    }
}

/// Semantic role of a prompt segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentKind {
    /// System prompt or policy header.
    SystemPrompt,
    /// Tool / function / MCP schema definition block.
    ToolDefinition,
    /// Retrieved RAG document chunk or repository context block.
    RagChunk,
    /// Prior conversation turn.
    ConversationTurn,
    /// Ephemeral user query / suffix.
    UserQuery,
}

impl SegmentKind {
    /// Canonical lowercase label.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SystemPrompt => "system_prompt",
            Self::ToolDefinition => "tool_definition",
            Self::RagChunk => "rag_chunk",
            Self::ConversationTurn => "conversation_turn",
            Self::UserQuery => "user_query",
        }
    }
}

/// Compute the canonical SHA-256 content hash of a token slice.
#[must_use]
pub fn hash_tokens(tokens: &[u32]) -> Digest {
    let mut h = Sha256::new();
    h.update_len_prefixed(SEGMENT_TOKEN_DOMAIN);
    h.update(&(tokens.len() as u64).to_le_bytes());
    for &tok in tokens {
        h.update(&tok.to_le_bytes());
    }
    h.finalize()
}

/// Compute the cumulative prefix-chain hash of all segments preceding a given point.
#[must_use]
pub fn chain_prefix_hash(parent: &Digest, segment_sha256: &Digest) -> Digest {
    let mut h = Sha256::new();
    h.update_len_prefixed(PREFIX_CHAIN_DOMAIN);
    h.update(parent);
    h.update(segment_sha256);
    h.finalize()
}

/// A discrete, cacheable (or ephemeral) segment within a structured LLM prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSegment {
    /// Semantic segment role.
    pub kind: SegmentKind,
    /// Human-readable label for telemetry and diagnostics.
    pub label: String,
    /// Token IDs comprising this segment.
    pub tokens: Vec<u32>,
    /// Precomputed SHA-256 content digest of `tokens`.
    pub segment_sha256: Digest,
}

impl PromptSegment {
    /// Create a new prompt segment and compute its content SHA-256.
    #[must_use]
    pub fn new(kind: SegmentKind, label: impl Into<String>, tokens: Vec<u32>) -> Self {
        let segment_sha256 = hash_tokens(&tokens);
        Self {
            kind,
            label: label.into(),
            tokens,
            segment_sha256,
        }
    }

    /// Number of tokens in this segment.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the segment has zero tokens.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Content-addressed key for CacheSlide position-independent segment reuse.
///
/// Strict equality checks every isolation field in addition to the 256-bit digest,
/// guaranteeing zero cross-tenant or cross-ACL aliasing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentKey {
    /// Tenant/ACL/model/dtype isolation boundary.
    pub scope: IsolationScope,
    /// SHA-256 of the segment's exact token IDs.
    pub segment_sha256: Digest,
    /// Domain-separated combined SHA-256 digest (for external wire indexing / logging).
    pub key_digest: Digest,
}

impl SegmentKey {
    /// Construct a tenant-isolated segment key.
    #[must_use]
    pub fn new(scope: IsolationScope, segment_sha256: Digest) -> Self {
        let mut h = Sha256::new();
        h.update_len_prefixed(SEGMENT_KEY_DOMAIN);
        scope.feed_into(&mut h);
        h.update(&segment_sha256);
        let key_digest = h.finalize();
        Self {
            scope,
            segment_sha256,
            key_digest,
        }
    }

    /// Construct a key directly from a [`PromptSegment`].
    #[must_use]
    pub fn from_segment(scope: IsolationScope, segment: &PromptSegment) -> Self {
        Self::new(scope, segment.segment_sha256)
    }
}

/// Chain-hashed key for matched standard prefix caching (`PrefixKvCache`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixChainKey {
    /// Tenant/ACL/model/dtype isolation boundary.
    pub scope: IsolationScope,
    /// Cumulative SHA-256 chain hash from segment 0 through the current segment.
    pub chain_sha256: Digest,
}

impl PrefixChainKey {
    /// Construct a prefix-chain key.
    #[must_use]
    pub fn new(scope: IsolationScope, chain_sha256: Digest) -> Self {
        Self {
            scope,
            chain_sha256,
        }
    }
}

/// Cached multi-layer KV block for a single prompt segment.
///
/// Keys are stored with RoPE already applied at `orig_pos..orig_pos + num_tokens`
/// (identical to standard KV cache storage in `hanzo-engine`). On position-shifted reuse,
/// `RopeShiftTable::new(cfg, new_pos - orig_pos)` transforms `keys` in place without
/// recomputing Q/K/V/FFN projections.
#[derive(Debug, Clone)]
pub struct SegmentKvPayload {
    /// Absolute token offset where this segment was originally cached.
    pub orig_pos: usize,
    /// Prefix chain digest of all tokens preceding `orig_pos` when cached.
    /// Used to recognize exact prefix matches (`cur_pos == orig_pos && cur_prefix == orig_prefix`).
    pub orig_prefix_hash: Digest,
    /// `true` iff every token preceding `orig_pos` was exact (no pure-shift or partial-recompute
    /// approximation upstream) when this payload was computed. Prevents approximate KV from
    /// silently masquerading as an exact prefix block on subsequent turns.
    pub exact_prefix_valid: bool,
    /// Number of tokens covered by this KV block.
    pub num_tokens: usize,
    /// Number of Transformer layers.
    pub num_layers: usize,
    /// Number of KV heads per layer.
    pub num_kv_heads: usize,
    /// Dimension of each head.
    pub head_dim: usize,
    /// Logical KV storage precision (for GPU memory accounting).
    pub kv_dtype: KvDtype,
    /// Per-layer RoPE-encoded K tensors, each of length `num_tokens * num_kv_heads * head_dim`.
    pub keys: Vec<Vec<f32>>,
    /// Per-layer V tensors, each of length `num_tokens * num_kv_heads * head_dim`.
    pub values: Vec<Vec<f32>>,
    /// SHA-256 integrity digest over metadata and raw K/V bit patterns.
    pub payload_sha256: Digest,
}

impl SegmentKvPayload {
    /// Construct a validated KV payload and compute its integrity digest.
    ///
    /// # Panics
    /// Panics if `keys` or `values` dimensions do not match `(num_layers, num_tokens, num_kv_heads, head_dim)`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        orig_pos: usize,
        orig_prefix_hash: Digest,
        exact_prefix_valid: bool,
        num_tokens: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dtype: KvDtype,
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
    ) -> Self {
        assert_eq!(keys.len(), num_layers, "keys layer count mismatch");
        assert_eq!(values.len(), num_layers, "values layer count mismatch");
        let per_layer_elems = num_tokens * num_kv_heads * head_dim;
        for l in 0..num_layers {
            assert_eq!(keys[l].len(), per_layer_elems, "keys[{l}] length mismatch");
            assert_eq!(
                values[l].len(),
                per_layer_elems,
                "values[{l}] length mismatch"
            );
        }
        let payload_sha256 = Self::compute_digest(
            orig_pos,
            &orig_prefix_hash,
            exact_prefix_valid,
            num_tokens,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dtype,
            &keys,
            &values,
        );
        Self {
            orig_pos,
            orig_prefix_hash,
            exact_prefix_valid,
            num_tokens,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dtype,
            keys,
            values,
            payload_sha256,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_digest(
        orig_pos: usize,
        orig_prefix_hash: &Digest,
        exact_prefix_valid: bool,
        num_tokens: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dtype: KvDtype,
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
    ) -> Digest {
        let mut h = Sha256::new();
        h.update_len_prefixed(PAYLOAD_DIGEST_DOMAIN);
        h.update(&(orig_pos as u64).to_le_bytes());
        h.update(orig_prefix_hash);
        h.update(&[u8::from(exact_prefix_valid)]);
        h.update(&(num_tokens as u64).to_le_bytes());
        h.update(&(num_layers as u64).to_le_bytes());
        h.update(&(num_kv_heads as u64).to_le_bytes());
        h.update(&(head_dim as u64).to_le_bytes());
        h.update_len_prefixed(kv_dtype.as_str().as_bytes());
        for layer in keys {
            for &v in layer {
                h.update(&v.to_bits().to_le_bytes());
            }
        }
        for layer in values {
            for &v in layer {
                h.update(&v.to_bits().to_le_bytes());
            }
        }
        h.finalize()
    }

    /// Recompute and verify the payload's SHA-256 integrity digest.
    #[must_use]
    pub fn verify_integrity(&self) -> bool {
        let expected = Self::compute_digest(
            self.orig_pos,
            &self.orig_prefix_hash,
            self.exact_prefix_valid,
            self.num_tokens,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.kv_dtype,
            &self.keys,
            &self.values,
        );
        expected == self.payload_sha256
    }

    /// Exact in-memory byte footprint of the stored `f32` K/V buffers plus fixed metadata.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        let elems = 2 * self.num_layers * self.num_tokens * self.num_kv_heads * self.head_dim;
        elems * std::mem::size_of::<f32>() + std::mem::size_of::<Self>()
    }

    /// Projected GPU KV cache bytes at `self.kv_dtype`.
    #[must_use]
    pub const fn gpu_kv_bytes(&self) -> usize {
        self.kv_dtype.kv_bytes(
            self.num_layers,
            self.num_tokens,
            self.num_kv_heads,
            self.head_dim,
        )
    }
}

/// Deterministic zero-dependency word/punctuation/byte tokenizer for converting real
/// repository files (Jinja tool templates, Markdown docs, Rust sources) into token IDs
/// in `[0..vocab_size)`.
#[derive(Debug, Clone, Copy)]
pub struct CorpusTokenizer {
    vocab_size: usize,
}

impl CorpusTokenizer {
    /// Construct a tokenizer targeting `vocab_size` (must be >= 64).
    ///
    /// # Panics
    /// Panics if `vocab_size < 64`.
    #[must_use]
    pub fn new(vocab_size: usize) -> Self {
        assert!(vocab_size >= 64, "vocab_size must be at least 64");
        Self { vocab_size }
    }

    #[inline]
    fn hash_piece(&self, piece: &[u8]) -> u32 {
        let digest = Sha256::digest(piece);
        let h = u64::from_le_bytes([
            digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
        ]) as usize;
        if self.vocab_size > 256 {
            (256 + (h % (self.vocab_size - 256))) as u32
        } else {
            (h % self.vocab_size) as u32
        }
    }

    /// Encode UTF-8 text into deterministic subword/word token IDs.
    ///
    /// Splits on whitespace and punctuation boundaries (mirroring BPE pre-tokenization),
    /// mapping single-byte pieces to `0..256.min(vocab_size)` and multi-byte pieces via
    /// SHA-256 into `256..vocab_size` (or `0..vocab_size` when `vocab_size <= 256`).
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            let b = bytes[i];
            if b.is_ascii_alphanumeric() || b == b'_' {
                let start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                    && (i - start) < 12
                {
                    i += 1;
                }
                let piece = &bytes[start..i];
                if piece.len() == 1 {
                    tokens.push((usize::from(piece[0]) % self.vocab_size) as u32);
                } else {
                    tokens.push(self.hash_piece(piece));
                }
            } else if b.is_ascii_whitespace() {
                // Coalesce consecutive whitespace (up to 4 bytes) into one token.
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_whitespace() && (i - start) < 4 {
                    i += 1;
                }
                let piece = &bytes[start..i];
                if piece.len() == 1 {
                    tokens.push((usize::from(piece[0]) % self.vocab_size) as u32);
                } else {
                    tokens.push(self.hash_piece(piece));
                }
            } else {
                tokens.push((usize::from(b) % self.vocab_size) as u32);
                i += 1;
            }
        }
        tokens
    }

    /// Encode text and slice/pad deterministically to an exact target token count.
    #[must_use]
    pub fn encode_exact(&self, text: &str, target_len: usize) -> Vec<u32> {
        let mut toks = self.encode(text);
        if toks.is_empty() {
            toks.push(b' ' as u32);
        }
        let base_len = toks.len();
        while toks.len() < target_len {
            let idx = toks.len() % base_len;
            toks.push(toks[idx]);
        }
        toks.truncate(target_len);
        toks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolation_scope_separates_keys_across_all_dimensions() {
        let seg = PromptSegment::new(
            SegmentKind::ToolDefinition,
            "search_tool",
            vec![10, 20, 30, 40],
        );

        let base = SegmentKey::from_segment(
            IsolationScope::new("tenant_a", "acl_eng", "zen-7b", KvDtype::F16),
            &seg,
        );
        let diff_tenant = SegmentKey::from_segment(
            IsolationScope::new("tenant_b", "acl_eng", "zen-7b", KvDtype::F16),
            &seg,
        );
        let diff_acl = SegmentKey::from_segment(
            IsolationScope::new("tenant_a", "acl_fin", "zen-7b", KvDtype::F16),
            &seg,
        );
        let diff_model = SegmentKey::from_segment(
            IsolationScope::new("tenant_a", "acl_eng", "zen-32b", KvDtype::F16),
            &seg,
        );
        let diff_dtype = SegmentKey::from_segment(
            IsolationScope::new("tenant_a", "acl_eng", "zen-7b", KvDtype::Int4Asym),
            &seg,
        );

        for other in [&diff_tenant, &diff_acl, &diff_model, &diff_dtype] {
            assert_ne!(&base, other);
            assert_ne!(base.key_digest, other.key_digest);
        }
    }

    #[test]
    fn payload_integrity_detects_tampering() {
        let mut payload = SegmentKvPayload::new(
            64,
            [0u8; 32],
            true,
            4,
            2,
            2,
            8,
            KvDtype::F16,
            vec![vec![0.5; 4 * 2 * 8]; 2],
            vec![vec![-0.25; 4 * 2 * 8]; 2],
        );
        assert!(payload.verify_integrity());
        payload.keys[1][3] += 1e-4;
        assert!(!payload.verify_integrity());
    }

    #[test]
    fn tokenizer_is_deterministic_and_bounded() {
        let tok = CorpusTokenizer::new(1024);
        let a = tok.encode("{\"tool\": \"run_sql\", \"limit\": 50}");
        let b = tok.encode("{\"tool\": \"run_sql\", \"limit\": 50}");
        assert_eq!(a, b);
        assert!(a.iter().all(|&t| (t as usize) < 1024));
    }
}
