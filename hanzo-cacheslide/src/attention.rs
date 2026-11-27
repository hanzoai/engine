// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Multi-layer causal Transformer prefill engine with RoPE, GQA, SwiGLU, and CacheSlide
//! position-shifted KV reuse + selective boundary/drift recomputation.
//!
//! Executes four matched modes on identical model weights and prompt segments:
//! 1. [`ExecutionMode::FullPrefill`]: exact ground-truth causal forward pass from scratch.
//! 2. [`ExecutionMode::PrefixCacheOnly`]: standard chain-hashed prefix cache baseline.
//! 3. [`ExecutionMode::CacheSlidePureShift`]: non-prefix segment KV reuse with exact
//!    $R(\Delta p)$ RoPE key shifting and zero boundary recomputation.
//! 4. [`ExecutionMode::CacheSlideSelectiveRecompute`]: $R(\Delta p)$ RoPE key shifting
//!    plus selective recomputation of segment boundary/sink and high cross-segment
//!    sensitivity tokens, with an optional Layer-1 KV-drift automatic escalation gate.

use std::sync::Arc;

use crate::cache::{PrefixKvCache, SegmentKvCache};
use crate::rope::{checked_position_shift, RopeCache, RopeConfig, RopeLayout, RopeShiftTable};
use crate::segment::{
    chain_prefix_hash, IsolationScope, KvDtype, PrefixChainKey, PromptSegment, SegmentKey,
    SegmentKvPayload,
};
use crate::sha256::Digest;

/// Configuration for the pure-Rust multi-layer causal Transformer.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Number of Transformer decoder layers.
    pub num_layers: usize,
    /// Hidden dimension (`num_heads * head_dim`).
    pub d_model: usize,
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of KV attention heads (Grouped-Query Attention when `num_kv_heads < num_heads`).
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// SwiGLU feed-forward intermediate dimension.
    pub d_ff: usize,
    /// Maximum precomputed RoPE sequence length.
    pub max_seq_len: usize,
    /// RoPE base theta (e.g. `10_000.0` or `500_000.0`).
    pub rope_theta: f64,
    /// RoPE plane layout (`GptNeoX` or `Interleaved`).
    pub rope_layout: RopeLayout,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Output logit scale calibrating next-token entropy to realistic LLM sharpness.
    pub logit_scale: f32,
    /// Logical KV cache dtype used for GPU memory accounting and scope isolation.
    pub kv_dtype: KvDtype,
}

impl TransformerConfig {
    /// Default benchmark configuration: 6-layer GQA Transformer (`d_model=64`, 4 Q heads,
    /// 2 KV heads, `head_dim=16`, `d_ff=128`, `V=1024`, NeoX RoPE, FP16 KV accounting).
    #[must_use]
    pub fn benchmark_default() -> Self {
        Self {
            vocab_size: 1024,
            num_layers: 6,
            d_model: 64,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            d_ff: 128,
            max_seq_len: 2048,
            rope_theta: 10_000.0,
            rope_layout: RopeLayout::GptNeoX,
            rms_norm_eps: 1e-5,
            logit_scale: 2.2,
            kv_dtype: KvDtype::F16,
        }
    }

    /// Compact unit-test configuration for fast deterministic assertions.
    #[must_use]
    pub fn tiny_test() -> Self {
        Self {
            vocab_size: 256,
            num_layers: 3,
            d_model: 32,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            d_ff: 64,
            max_seq_len: 512,
            rope_theta: 10_000.0,
            rope_layout: RopeLayout::GptNeoX,
            rms_norm_eps: 1e-5,
            logit_scale: 3.0,
            kv_dtype: KvDtype::F16,
        }
    }

    /// Derived [`RopeConfig`].
    #[must_use]
    pub fn rope_config(&self) -> RopeConfig {
        RopeConfig::standard(self.head_dim, self.rope_theta, self.rope_layout)
    }

    /// KV projection dimension (`num_kv_heads * head_dim`).
    #[inline]
    #[must_use]
    pub const fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

/// Deterministic `SplitMix64` PRNG for reproducible weight initialization.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_f64_open01(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) / ((1u64 << 53) as f64)
    }

    /// Approximate standard normal draw via Box-Muller transform.
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f64_open01();
        let u2 = self.next_f64_open01();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        (r * theta.cos()) as f32
    }

    fn matrix(&mut self, rows: usize, cols: usize, std_dev: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            out.push(self.next_normal() * std_dev);
        }
        out
    }
}

/// Single causal Transformer decoder layer with pre-norm RMSNorm, GQA, RoPE, and SwiGLU FFN.
#[derive(Debug, Clone)]
struct TransformerLayer {
    attn_norm: Vec<f32>,
    /// `[d_model, d_model]`
    wq: Vec<f32>,
    /// `[kv_dim, d_model]`
    wk: Vec<f32>,
    /// `[kv_dim, d_model]`
    wv: Vec<f32>,
    /// `[d_model, d_model]`
    wo: Vec<f32>,
    ffn_norm: Vec<f32>,
    /// `[d_ff, d_model]`
    w_gate: Vec<f32>,
    /// `[d_ff, d_model]`
    w_up: Vec<f32>,
    /// `[d_model, d_ff]`
    w_down: Vec<f32>,
}

/// Pure-Rust multi-layer causal Transformer model.
#[derive(Debug, Clone)]
pub struct TransformerModel {
    config: TransformerConfig,
    /// `[vocab_size, d_model]`
    tok_embed: Vec<f32>,
    layers: Vec<TransformerLayer>,
    final_norm: Vec<f32>,
    /// `[vocab_size, d_model]`
    lm_head: Vec<f32>,
    rope: RopeCache,
}

/// Configuration for CacheSlide selective boundary & cross-segment recomputation.
#[derive(Debug, Clone, Copy)]
pub struct SelectiveRecomputeConfig {
    /// Target fraction of tokens in a shifted segment to recompute (`0.0..=1.0`).
    pub recompute_ratio: f32,
    /// Minimum leading boundary/sink tokens to recompute at the start of a shifted segment.
    pub lead_boundary_tokens: usize,
    /// Minimum trailing boundary tokens to recompute at the end of a shifted segment.
    pub trail_boundary_tokens: usize,
    /// Optional Layer-1 relative K-drift threshold (`||K_new - K_shifted|| / ||K_new||`).
    /// When exceeded on a shifted segment's boundary probe, recomputation automatically
    /// escalates to `escalated_recompute_ratio`.
    pub auto_escalate_drift_threshold: Option<f32>,
    /// Recomputation ratio applied when the Layer-1 drift gate triggers (`0.0..=1.0`).
    pub escalated_recompute_ratio: f32,
}

impl Default for SelectiveRecomputeConfig {
    fn default() -> Self {
        Self {
            recompute_ratio: 0.20,
            lead_boundary_tokens: 4,
            trail_boundary_tokens: 2,
            auto_escalate_drift_threshold: None,
            escalated_recompute_ratio: 1.0,
        }
    }
}

/// Execution strategy for evaluating a segmented prompt.
#[derive(Debug, Clone, Copy)]
pub enum ExecutionMode {
    /// Ground-truth exact prefill over every token from scratch.
    FullPrefill,
    /// Standard contiguous prefix caching (chain-hashed, zero position shifting).
    PrefixCacheOnly,
    /// CacheSlide pure RoPE position shift (`R(Δp) · K_orig`) with zero boundary recomputation.
    CacheSlidePureShift,
    /// CacheSlide RoPE position shift + selective boundary/drift token recomputation.
    CacheSlideSelectiveRecompute(SelectiveRecomputeConfig),
}

/// Detailed execution output and accounting metrics for one prompt prefill.
#[derive(Debug, Clone)]
pub struct PrefillOutput {
    /// Final-token next-token logits of length `vocab_size`.
    pub logits: Vec<f32>,
    /// Total tokens in the prompt.
    pub total_tokens: usize,
    /// Tokens whose KV was served directly from cache (exact or RoPE-shifted, not recomputed).
    pub tokens_reused: usize,
    /// Tokens fully computed (cache misses + selectively recomputed boundary/drift tokens).
    pub tokens_computed: usize,
    /// Subset of `tokens_computed` spent on selective recomputation inside shifted cached segments.
    pub tokens_selectively_recomputed: usize,
    /// Number of segments served with a non-zero RoPE position shift `Δp != 0`.
    pub segments_shifted: usize,
    /// Number of segments served as exact zero-shift prefix matches.
    pub segments_exact_hit: usize,
    /// Number of segments that missed cache and were fully computed.
    pub segments_missed: usize,
    /// Maximum Layer-1 relative K-drift measured across shifted segments (if probed).
    pub max_layer1_k_drift: f32,
    /// Number of shifted segments where the automatic drift gate escalated recomputation.
    pub drift_gate_escalations: usize,
    /// Exact floating-point operations executed (projections + attention + SwiGLU + RoPE shift + LM head).
    pub prefill_flops: u64,
}

/// Output quality metrics comparing a candidate prefill against exact `FullPrefill`.
#[derive(Debug, Clone, Copy, Default)]
pub struct QualityMetrics {
    /// Forward KL divergence $D_{\text{KL}}(P_{\text{ref}} \| P_{\text{cand}})$ in nats ($\ge 0$).
    pub kl_divergence: f64,
    /// Symmetric Jensen-Shannon divergence in nats.
    pub js_divergence: f64,
    /// Cosine similarity between raw final-token logit vectors.
    pub logit_cosine_similarity: f64,
    /// Maximum absolute difference across final-token logits.
    pub max_abs_logit_diff: f32,
    /// Whether `argmax(cand_logits) == argmax(ref_logits)`.
    pub top1_agree: bool,
    /// Fraction of reference top-5 tokens appearing in candidate top-5 (`0.0..=1.0`).
    pub top5_overlap: f64,
    /// Shannon entropy of the exact reference next-token distribution (nats).
    pub ref_entropy_nats: f64,
}

/// Compare candidate final-token logits against exact reference logits.
///
/// # Panics
/// Panics if `ref_logits.len() != cand_logits.len()` or slices are empty.
#[must_use]
pub fn evaluate_quality(ref_logits: &[f32], cand_logits: &[f32]) -> QualityMetrics {
    assert_eq!(ref_logits.len(), cand_logits.len());
    assert!(!ref_logits.is_empty());

    let p_ref = softmax_f64(ref_logits);
    let p_cand = softmax_f64(cand_logits);

    let mut kl = 0.0f64;
    let mut js_left = 0.0f64;
    let mut js_right = 0.0f64;
    let mut entropy = 0.0f64;
    let mut dot = 0.0f64;
    let mut norm_r = 0.0f64;
    let mut norm_c = 0.0f64;
    let mut max_diff = 0.0f32;

    for i in 0..ref_logits.len() {
        let pr = p_ref[i];
        let pc = p_cand[i].max(1e-30);
        if pr > 1e-30 {
            kl += pr * (pr / pc).ln();
            entropy -= pr * pr.ln();
        }
        let m = 0.5 * (pr + p_cand[i]);
        if pr > 1e-30 && m > 1e-30 {
            js_left += pr * (pr / m).ln();
        }
        if p_cand[i] > 1e-30 && m > 1e-30 {
            js_right += p_cand[i] * (p_cand[i] / m).ln();
        }

        let lr = f64::from(ref_logits[i]);
        let lc = f64::from(cand_logits[i]);
        dot += lr * lc;
        norm_r += lr * lr;
        norm_c += lc * lc;
        max_diff = max_diff.max((ref_logits[i] - cand_logits[i]).abs());
    }

    let cosine = if norm_r > 0.0 && norm_c > 0.0 {
        (dot / (norm_r.sqrt() * norm_c.sqrt())).clamp(-1.0, 1.0)
    } else {
        1.0
    };

    let top5_ref = top_k_indices(ref_logits, 5);
    let top5_cand = top_k_indices(cand_logits, 5);
    let top1_agree = top5_ref[0] == top5_cand[0];
    let overlap_count = top5_ref.iter().filter(|id| top5_cand.contains(id)).count();
    let top5_overlap = overlap_count as f64 / top5_ref.len() as f64;

    QualityMetrics {
        kl_divergence: kl.max(0.0),
        js_divergence: (0.5 * (js_left + js_right)).max(0.0),
        logit_cosine_similarity: cosine,
        max_abs_logit_diff: max_diff,
        top1_agree,
        top5_overlap,
        ref_entropy_nats: entropy.max(0.0),
    }
}

fn softmax_f64(logits: &[f32]) -> Vec<f64> {
    let max_v = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for &v in logits {
        let e = f64::from(v - max_v).exp();
        sum += e;
        exps.push(e);
    }
    let inv = 1.0 / sum.max(1e-30);
    for e in &mut exps {
        *e *= inv;
    }
    exps
}

fn top_k_indices(vals: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(vals.len());
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| {
        vals[b]
            .partial_cmp(&vals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

#[inline]
fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let inv_rms = 1.0 / (ss / (x.len() as f32) + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

#[inline]
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for (r, slot) in out.iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        *slot = acc;
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl TransformerModel {
    /// Initialize a deterministic multi-layer Transformer from `config` and `seed`.
    ///
    /// # Panics
    /// Panics if `config.d_model != config.num_heads * config.head_dim` or
    /// `config.num_heads % config.num_kv_heads != 0`.
    #[must_use]
    pub fn new_deterministic(config: TransformerConfig, seed: u64) -> Self {
        assert_eq!(config.d_model, config.num_heads * config.head_dim);
        assert!(config.num_kv_heads > 0 && config.num_heads.is_multiple_of(config.num_kv_heads));

        let mut rng = SplitMix64::new(seed);
        let d = config.d_model;
        let kv_dim = config.kv_dim();
        let d_ff = config.d_ff;

        let embed_std = 1.0 / (d as f32).sqrt();
        let proj_std = 1.0 / (d as f32).sqrt();
        let res_std = 1.0 / ((2 * config.num_layers * d) as f32).sqrt();
        let ff_down_std = 1.0 / ((2 * config.num_layers * d_ff) as f32).sqrt();

        let tok_embed = rng.matrix(config.vocab_size, d, embed_std * 1.4);
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(TransformerLayer {
                attn_norm: vec![1.0; d],
                wq: rng.matrix(d, d, proj_std * 1.25),
                wk: rng.matrix(kv_dim, d, proj_std * 1.25),
                wv: rng.matrix(kv_dim, d, proj_std),
                wo: rng.matrix(d, d, res_std),
                ffn_norm: vec![1.0; d],
                w_gate: rng.matrix(d_ff, d, proj_std),
                w_up: rng.matrix(d_ff, d, proj_std),
                w_down: rng.matrix(d, d_ff, ff_down_std),
            });
        }
        let final_norm = vec![1.0; d];
        let lm_head = rng.matrix(config.vocab_size, d, proj_std);
        let rope = RopeCache::new(config.rope_config(), config.max_seq_len);

        Self {
            config,
            tok_embed,
            layers,
            final_norm,
            lm_head,
            rope,
        }
    }

    /// Model configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Run a segmented prompt under the requested [`ExecutionMode`].
    ///
    /// - `prefix_cache` is used and populated when `mode == ExecutionMode::PrefixCacheOnly`.
    /// - `segment_cache` is used and populated when `mode` is `CacheSlidePureShift` or
    ///   `CacheSlideSelectiveRecompute`.
    ///
    /// # Panics
    /// Panics if `segments` produces zero total tokens.
    pub fn prefill_segments(
        &self,
        scope: &IsolationScope,
        segments: &[PromptSegment],
        mode: ExecutionMode,
        prefix_cache: Option<&PrefixKvCache>,
        segment_cache: Option<&SegmentKvCache>,
    ) -> PrefillOutput {
        let total_tokens: usize = segments.iter().map(PromptSegment::len).sum();
        assert!(total_tokens > 0, "prompt must contain at least one token");

        let mut all_tokens = Vec::with_capacity(total_tokens);
        let mut seg_ranges = Vec::with_capacity(segments.len());
        let mut prefix_hashes = Vec::with_capacity(segments.len());
        let mut running_prefix: Digest = [0u8; 32];

        for seg in segments {
            let start = all_tokens.len();
            all_tokens.extend_from_slice(&seg.tokens);
            let end = all_tokens.len();
            seg_ranges.push((start, end));
            prefix_hashes.push(running_prefix);
            running_prefix = chain_prefix_hash(&running_prefix, &seg.segment_sha256);
        }

        let num_layers = self.config.num_layers;
        let kv_dim = self.config.kv_dim();
        let mut kv_k: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; total_tokens * kv_dim])
            .collect();
        let mut kv_v: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; total_tokens * kv_dim])
            .collect();

        let mut active_flags = vec![false; total_tokens];
        let mut tokens_reused = 0usize;
        let mut tokens_selectively_recomputed = 0usize;
        let mut segments_shifted = 0usize;
        let mut segments_exact_hit = 0usize;
        let mut segments_missed = 0usize;
        let mut max_layer1_k_drift = 0.0f32;
        let mut drift_gate_escalations = 0usize;
        let mut extra_flops = 0u64;

        // Track missed segments so we store their newly computed KV after the forward pass.
        let mut missed_seg_indices: Vec<usize> = Vec::new();
        // Track shifted segments that need selective recomputation once Layer 0 KV is ready.
        struct ShiftedSegWork {
            seg_idx: usize,
            start: usize,
            end: usize,
        }
        let mut shifted_for_selective: Vec<ShiftedSegWork> = Vec::new();

        // Track whether each segment's KV is exact (zero upstream pure/partial-shift approximation).
        let mut seg_exact = vec![true; segments.len()];

        match mode {
            ExecutionMode::FullPrefill => {
                active_flags.fill(true);
                segments_missed = segments.len();
            }
            ExecutionMode::PrefixCacheOnly => {
                let mut prefix_still_matching = true;
                let mut chain_hash: Digest = [0u8; 32];
                for (i, seg) in segments.iter().enumerate() {
                    let (start, end) = seg_ranges[i];
                    chain_hash = chain_prefix_hash(&chain_hash, &seg.segment_sha256);
                    let hit = if prefix_still_matching {
                        prefix_cache
                            .and_then(|c| c.get(&PrefixChainKey::new(scope.clone(), chain_hash)))
                    } else {
                        None
                    };
                    if let Some(payload) = hit {
                        if payload.num_tokens == seg.len() && payload.orig_pos == start {
                            self.copy_payload_into_kv(&payload, start, 0, &mut kv_k, &mut kv_v);
                            tokens_reused += seg.len();
                            segments_exact_hit += 1;
                            continue;
                        }
                    }
                    prefix_still_matching = false;
                    for slot in &mut active_flags[start..end] {
                        *slot = true;
                    }
                    segments_missed += 1;
                    missed_seg_indices.push(i);
                }
            }
            ExecutionMode::CacheSlidePureShift | ExecutionMode::CacheSlideSelectiveRecompute(_) => {
                let min_toks = segment_cache.map_or(0, SegmentKvCache::min_segment_tokens);
                let mut upstream_exact = true;
                for (i, seg) in segments.iter().enumerate() {
                    let (start, end) = seg_ranges[i];
                    let seg_len = end - start;
                    if seg_len < min_toks {
                        for slot in &mut active_flags[start..end] {
                            *slot = true;
                        }
                        seg_exact[i] = upstream_exact;
                        segments_missed += 1;
                        missed_seg_indices.push(i);
                        continue;
                    }

                    let key = SegmentKey::from_segment(scope.clone(), seg);
                    if let Some(payload) = segment_cache.and_then(|c| c.get(&key)) {
                        if payload.num_tokens == seg_len {
                            if let Some(delta_pos) = checked_position_shift(payload.orig_pos, start)
                            {
                                let exact_prefix = upstream_exact
                                    && delta_pos == 0
                                    && payload.orig_prefix_hash == prefix_hashes[i]
                                    && payload.exact_prefix_valid;
                                let shift_flops = self.copy_payload_into_kv(
                                    &payload, start, delta_pos, &mut kv_k, &mut kv_v,
                                );
                                extra_flops += shift_flops;

                                if exact_prefix {
                                    tokens_reused += seg_len;
                                    segments_exact_hit += 1;
                                    seg_exact[i] = true;
                                } else {
                                    segments_shifted += 1;
                                    seg_exact[i] = false;
                                    upstream_exact = false;
                                    match mode {
                                        ExecutionMode::CacheSlidePureShift => {
                                            tokens_reused += seg_len;
                                        }
                                        ExecutionMode::CacheSlideSelectiveRecompute(_) => {
                                            shifted_for_selective.push(ShiftedSegWork {
                                                seg_idx: i,
                                                start,
                                                end,
                                            });
                                        }
                                        _ => unreachable!(),
                                    }
                                }
                                continue;
                            }
                        }
                    }

                    // Cache miss: compute full segment.
                    for slot in &mut active_flags[start..end] {
                        *slot = true;
                    }
                    seg_exact[i] = upstream_exact;
                    segments_missed += 1;
                    missed_seg_indices.push(i);
                }
            }
        }

        // If selective recomputation is active and any shifted segments exist,
        // first ensure Layer 0 K/V for any missed segments preceding them is populated,
        // then run the fast Layer-0 cross-segment sensitivity + Layer-1 boundary K-drift probe!
        if let ExecutionMode::CacheSlideSelectiveRecompute(sel_cfg) = mode {
            if !shifted_for_selective.is_empty() {
                // Populate Layer-0 K/V for all currently-active (missed) positions so Layer-0 KV
                // is 100% exact across the entire prompt [0..total_tokens).
                // Note: Layer-0 K/V depends only on token embeddings + RoPE, so computing them
                // here is exact and avoids redundant work!
                let mut l0_norm = vec![0.0f32; self.config.d_model];
                let layer0 = &self.layers[0];
                for pos in 0..total_tokens {
                    if active_flags[pos] {
                        let emb = self.embed_slice(all_tokens[pos]);
                        rms_norm(
                            emb,
                            &layer0.attn_norm,
                            self.config.rms_norm_eps,
                            &mut l0_norm,
                        );
                        let k_slot = &mut kv_k[0][pos * kv_dim..(pos + 1) * kv_dim];
                        let v_slot = &mut kv_v[0][pos * kv_dim..(pos + 1) * kv_dim];
                        matvec(&layer0.wk, kv_dim, self.config.d_model, &l0_norm, k_slot);
                        matvec(&layer0.wv, kv_dim, self.config.d_model, &l0_norm, v_slot);
                        for h in 0..self.config.num_kv_heads {
                            let hs = h * self.config.head_dim;
                            self.rope
                                .apply_at_pos(&mut k_slot[hs..hs + self.config.head_dim], pos);
                        }
                    }
                }

                for work in &shifted_for_selective {
                    let seg_len = work.end - work.start;
                    let mut effective_ratio = sel_cfg.recompute_ratio.clamp(0.0, 1.0);

                    // Optional automatic Layer-1 K-drift quality gate on boundary probe tokens.
                    if let Some(threshold) = sel_cfg.auto_escalate_drift_threshold {
                        if num_layers >= 2 && work.start > 0 {
                            let (drift, probe_flops) = self.measure_boundary_layer1_k_drift(
                                &all_tokens,
                                work.start,
                                work.end,
                                &kv_k[0],
                                &kv_v[0],
                                &kv_k[1],
                            );
                            extra_flops += probe_flops;
                            max_layer1_k_drift = max_layer1_k_drift.max(drift);
                            if drift > threshold {
                                effective_ratio = effective_ratio
                                    .max(sel_cfg.escalated_recompute_ratio.clamp(0.0, 1.0));
                                drift_gate_escalations += 1;
                            }
                        }
                    }

                    let (selected, sel_flops) = self.select_tokens_for_recompute(
                        &all_tokens,
                        work.start,
                        work.end,
                        effective_ratio,
                        sel_cfg.lead_boundary_tokens,
                        sel_cfg.trail_boundary_tokens,
                        &kv_k[0],
                    );
                    extra_flops += sel_flops;

                    let sel_count = selected.len();
                    for pos in selected {
                        active_flags[pos] = true;
                    }
                    tokens_selectively_recomputed += sel_count;
                    tokens_reused += seg_len - sel_count;
                    if sel_count == seg_len {
                        // Fully recomputed segment restores exactness iff all earlier segments are exact.
                        seg_exact[work.seg_idx] = seg_exact[..work.seg_idx].iter().all(|&e| e);
                    }
                }
            }
        }

        // Always ensure the final prompt token is active so we can compute exact next-token logits.
        let last_pos = total_tokens - 1;
        if !active_flags[last_pos] {
            active_flags[last_pos] = true;
            tokens_reused = tokens_reused.saturating_sub(1);
        }

        let active_positions: Vec<usize> = active_flags
            .iter()
            .enumerate()
            .filter_map(|(pos, &is_active)| is_active.then_some(pos))
            .collect();

        let (logits, forward_flops) =
            self.run_active_layers(&all_tokens, &active_positions, &mut kv_k, &mut kv_v);

        // Populate caches for missed segments.
        match mode {
            ExecutionMode::PrefixCacheOnly => {
                if let Some(cache) = prefix_cache {
                    let mut chain_hash: Digest = [0u8; 32];
                    for (i, seg) in segments.iter().enumerate() {
                        chain_hash = chain_prefix_hash(&chain_hash, &seg.segment_sha256);
                        if missed_seg_indices.contains(&i) {
                            let (start, end) = seg_ranges[i];
                            let payload = self.extract_segment_payload(
                                start,
                                end,
                                prefix_hashes[i],
                                true,
                                &kv_k,
                                &kv_v,
                            );
                            let _ = cache.insert(
                                PrefixChainKey::new(scope.clone(), chain_hash),
                                Arc::new(payload),
                            );
                        }
                    }
                }
            }
            ExecutionMode::CacheSlidePureShift | ExecutionMode::CacheSlideSelectiveRecompute(_) => {
                if let Some(cache) = segment_cache {
                    for &i in &missed_seg_indices {
                        let (start, end) = seg_ranges[i];
                        let payload = self.extract_segment_payload(
                            start,
                            end,
                            prefix_hashes[i],
                            seg_exact[i],
                            &kv_k,
                            &kv_v,
                        );
                        let key = SegmentKey::from_segment(scope.clone(), &segments[i]);
                        let _ = cache.insert(key, Arc::new(payload));
                    }
                }
            }
            ExecutionMode::FullPrefill => {}
        }

        PrefillOutput {
            logits,
            total_tokens,
            tokens_reused,
            tokens_computed: active_positions.len(),
            tokens_selectively_recomputed,
            segments_shifted,
            segments_exact_hit,
            segments_missed,
            max_layer1_k_drift,
            drift_gate_escalations,
            prefill_flops: forward_flops + extra_flops,
        }
    }

    #[inline]
    fn embed_slice(&self, token_id: u32) -> &[f32] {
        let idx = (token_id as usize) % self.config.vocab_size;
        let d = self.config.d_model;
        &self.tok_embed[idx * d..(idx + 1) * d]
    }

    /// Copy a cached segment payload into `kv_k`/`kv_v` at `[start..start+num_tokens)`,
    /// applying exact RoPE shift `R(delta_pos)` to keys when `delta_pos != 0`.
    /// Returns the exact RoPE shift FLOP count.
    fn copy_payload_into_kv(
        &self,
        payload: &SegmentKvPayload,
        start: usize,
        delta_pos: i64,
        kv_k: &mut [Vec<f32>],
        kv_v: &mut [Vec<f32>],
    ) -> u64 {
        let kv_dim = self.config.kv_dim();
        let byte_start = start * kv_dim;
        let byte_end = (start + payload.num_tokens) * kv_dim;

        let shift_table = if delta_pos != 0 {
            Some(RopeShiftTable::new(&self.config.rope_config(), delta_pos))
        } else {
            None
        };

        for l in 0..self.config.num_layers {
            kv_k[l][byte_start..byte_end].copy_from_slice(&payload.keys[l]);
            kv_v[l][byte_start..byte_end].copy_from_slice(&payload.values[l]);
            if let Some(ref table) = shift_table {
                table.apply_to_k_block(
                    &mut kv_k[l][byte_start..byte_end],
                    payload.num_tokens,
                    self.config.num_kv_heads,
                );
            }
        }

        if delta_pos == 0 {
            0
        } else {
            // 4 muls + 2 adds = 6 FLOPs per 2D plane per head per token per layer.
            6u64 * (self.config.num_layers as u64)
                * (payload.num_tokens as u64)
                * (self.config.num_kv_heads as u64)
                * ((self.config.head_dim / 2) as u64)
        }
    }

    fn extract_segment_payload(
        &self,
        start: usize,
        end: usize,
        orig_prefix_hash: Digest,
        exact_prefix_valid: bool,
        kv_k: &[Vec<f32>],
        kv_v: &[Vec<f32>],
    ) -> SegmentKvPayload {
        let kv_dim = self.config.kv_dim();
        let s = start * kv_dim;
        let e = end * kv_dim;
        let mut keys = Vec::with_capacity(self.config.num_layers);
        let mut values = Vec::with_capacity(self.config.num_layers);
        for l in 0..self.config.num_layers {
            keys.push(kv_k[l][s..e].to_vec());
            values.push(kv_v[l][s..e].to_vec());
        }
        SegmentKvPayload::new(
            start,
            orig_prefix_hash,
            exact_prefix_valid,
            end - start,
            self.config.num_layers,
            self.config.num_kv_heads,
            self.config.head_dim,
            self.config.kv_dtype,
            keys,
            values,
        )
    }

    /// Measure exact relative Layer-1 K-vector drift on up to 4 boundary probe tokens
    /// of a shifted segment `[start..end)`.
    ///
    /// Because Layer-0 K/V is exact across `[0..end)`, running Layer 0 on 4 probe tokens
    /// and projecting their Layer-1 K gives an exact, cheap measurement of cross-segment
    /// contextualization drift: `||K_new^(1) - K_shifted^(1)||_2 / (||K_new^(1)||_2 + eps)`.
    fn measure_boundary_layer1_k_drift(
        &self,
        all_tokens: &[u32],
        start: usize,
        end: usize,
        kv_k0: &[f32],
        kv_v0: &[f32],
        kv_k1_shifted: &[f32],
    ) -> (f32, u64) {
        let seg_len = end - start;
        let mut probes = Vec::with_capacity(4);
        probes.push(start);
        if seg_len >= 2 {
            probes.push(start + 1);
        }
        if seg_len >= 4 {
            probes.push(end - 2);
        }
        if seg_len >= 3 {
            probes.push(end - 1);
        }

        let d = self.config.d_model;
        let kv_dim = self.config.kv_dim();
        let d_ff = self.config.d_ff;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let heads_per_kv = num_heads / self.config.num_kv_heads;
        let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();

        let layer0 = &self.layers[0];
        let layer1 = &self.layers[1];

        let mut norm = vec![0.0f32; d];
        let mut q = vec![0.0f32; d];
        let mut attn_out = vec![0.0f32; d];
        let mut proj = vec![0.0f32; d];
        let mut gate = vec![0.0f32; d_ff];
        let mut up = vec![0.0f32; d_ff];
        let mut ffn_out = vec![0.0f32; d];
        let mut k1_new = vec![0.0f32; kv_dim];
        let mut scores = Vec::new();

        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        let mut flops = 0u64;

        for &pos in &probes {
            let mut h = self.embed_slice(all_tokens[pos]).to_vec();
            rms_norm(&h, &layer0.attn_norm, self.config.rms_norm_eps, &mut norm);
            matvec(&layer0.wq, d, d, &norm, &mut q);
            for head in 0..num_heads {
                let hs = head * head_dim;
                self.rope.apply_at_pos(&mut q[hs..hs + head_dim], pos);
            }

            attn_out.fill(0.0);
            scores.resize(pos + 1, 0.0);
            for head in 0..num_heads {
                let kvh = head / heads_per_kv;
                let qh = &q[head * head_dim..(head + 1) * head_dim];
                let mut max_s = f32::NEG_INFINITY;
                for (j, slot) in scores.iter_mut().enumerate() {
                    let kb = j * kv_dim + kvh * head_dim;
                    let kh = &kv_k0[kb..kb + head_dim];
                    let mut dot = 0.0f32;
                    for c in 0..head_dim {
                        dot += qh[c] * kh[c];
                    }
                    let s = dot * inv_sqrt_hd;
                    *slot = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut sum_e = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum_e += *s;
                }
                let inv_e = 1.0 / sum_e.max(1e-30);
                let oh = &mut attn_out[head * head_dim..(head + 1) * head_dim];
                for (j, &w) in scores.iter().enumerate() {
                    let p = w * inv_e;
                    let vb = j * kv_dim + kvh * head_dim;
                    let vh = &kv_v0[vb..vb + head_dim];
                    for c in 0..head_dim {
                        oh[c] += p * vh[c];
                    }
                }
            }

            matvec(&layer0.wo, d, d, &attn_out, &mut proj);
            for i in 0..d {
                h[i] += proj[i];
            }
            rms_norm(&h, &layer0.ffn_norm, self.config.rms_norm_eps, &mut norm);
            matvec(&layer0.w_gate, d_ff, d, &norm, &mut gate);
            matvec(&layer0.w_up, d_ff, d, &norm, &mut up);
            for i in 0..d_ff {
                gate[i] = silu(gate[i]) * up[i];
            }
            matvec(&layer0.w_down, d, d_ff, &gate, &mut ffn_out);
            for i in 0..d {
                h[i] += ffn_out[i];
            }

            // Project fresh Layer-1 key and compare against shifted cached Layer-1 key.
            rms_norm(&h, &layer1.attn_norm, self.config.rms_norm_eps, &mut norm);
            matvec(&layer1.wk, kv_dim, d, &norm, &mut k1_new);
            for kvh in 0..self.config.num_kv_heads {
                let hs = kvh * head_dim;
                self.rope.apply_at_pos(&mut k1_new[hs..hs + head_dim], pos);
            }

            let cached = &kv_k1_shifted[pos * kv_dim..(pos + 1) * kv_dim];
            for i in 0..kv_dim {
                let diff = f64::from(k1_new[i] - cached[i]);
                let r = f64::from(k1_new[i]);
                diff_sq += diff * diff;
                ref_sq += r * r;
            }

            flops +=
                2 * (d * d + d * d + 3 * d_ff * d + kv_dim * d) as u64 + 4 * ((pos + 1) * d) as u64;
        }

        let drift = (diff_sq.sqrt() / (ref_sq.sqrt() + 1e-8)) as f32;
        (drift, flops)
    }

    /// Select the top `k_recomp` token positions inside shifted segment `[start..end)`
    /// combining leading/trailing boundary sink tokens with the highest Layer-0
    /// cross-segment query-key affinity tokens.
    #[allow(clippy::too_many_arguments)]
    fn select_tokens_for_recompute(
        &self,
        all_tokens: &[u32],
        start: usize,
        end: usize,
        recompute_ratio: f32,
        lead_boundary: usize,
        trail_boundary: usize,
        kv_k0: &[f32],
    ) -> (Vec<usize>, u64) {
        let seg_len = end - start;
        if seg_len == 0 || (recompute_ratio <= 0.0 && lead_boundary == 0 && trail_boundary == 0) {
            return (Vec::new(), 0);
        }

        let ratio_target = ((seg_len as f32) * recompute_ratio).ceil() as usize;
        let min_b = lead_boundary.saturating_add(trail_boundary);
        let k_target = ratio_target
            .max(if recompute_ratio > 0.0 { min_b } else { 0 })
            .min(seg_len);

        if k_target == 0 {
            return (Vec::new(), 0);
        }
        if k_target >= seg_len {
            return ((start..end).collect(), 0);
        }

        let mut selected = Vec::with_capacity(k_target);
        let lead = lead_boundary.min(k_target);
        for p in start..(start + lead) {
            selected.push(p);
        }
        let trail = trail_boundary.min(k_target - selected.len());
        let trail_start = (end - trail).max(start + lead);
        for p in trail_start..end {
            selected.push(p);
        }

        let remaining = k_target.saturating_sub(selected.len());
        if remaining == 0 || start + lead >= trail_start {
            selected.sort_unstable();
            selected.dedup();
            return (selected, 0);
        }

        // Score interior positions by their exact Layer-0 cross-segment query affinity:
        // how strongly q_0(pos) aligns with the mean key of preceding tokens [0..start)
        // relative to the mean intra-segment key [start..end).
        let kv_dim = self.config.kv_dim();
        let d = self.config.d_model;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let heads_per_kv = num_heads / self.config.num_kv_heads;
        let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();

        let mut k_prefix_mean = vec![0.0f32; kv_dim];
        if start > 0 {
            for j in 0..start {
                let row = &kv_k0[j * kv_dim..(j + 1) * kv_dim];
                for c in 0..kv_dim {
                    k_prefix_mean[c] += row[c];
                }
            }
            let inv = 1.0 / (start as f32);
            for v in &mut k_prefix_mean {
                *v *= inv;
            }
        }
        let mut k_intra_mean = vec![0.0f32; kv_dim];
        for j in start..end {
            let row = &kv_k0[j * kv_dim..(j + 1) * kv_dim];
            for c in 0..kv_dim {
                k_intra_mean[c] += row[c];
            }
        }
        let inv_seg = 1.0 / (seg_len as f32);
        for v in &mut k_intra_mean {
            *v *= inv_seg;
        }

        let layer0 = &self.layers[0];
        let mut norm = vec![0.0f32; d];
        let mut q = vec![0.0f32; d];
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(trail_start - (start + lead));

        for (pos, &tok_id) in all_tokens
            .iter()
            .enumerate()
            .take(trail_start)
            .skip(start + lead)
        {
            let emb = self.embed_slice(tok_id);
            rms_norm(emb, &layer0.attn_norm, self.config.rms_norm_eps, &mut norm);
            matvec(&layer0.wq, d, d, &norm, &mut q);
            for head in 0..num_heads {
                let hs = head * head_dim;
                self.rope.apply_at_pos(&mut q[hs..hs + head_dim], pos);
            }
            let mut cross_score = 0.0f32;
            for head in 0..num_heads {
                let kvh = head / heads_per_kv;
                let qh = &q[head * head_dim..(head + 1) * head_dim];
                let kp = &k_prefix_mean[kvh * head_dim..(kvh + 1) * head_dim];
                let ki = &k_intra_mean[kvh * head_dim..(kvh + 1) * head_dim];
                let mut dot_p = 0.0f32;
                let mut dot_i = 0.0f32;
                for c in 0..head_dim {
                    dot_p += qh[c] * kp[c];
                    dot_i += qh[c] * ki[c];
                }
                cross_score += (dot_p - dot_i) * inv_sqrt_hd;
            }
            // Add slight recency/boundary prior so ties favor earlier positions where
            // prefix share of the causal window is highest.
            let pos_prior = 1.0 / ((pos - start + 1) as f32);
            scored.push((pos, cross_score / (num_heads as f32) + 0.15 * pos_prior));
        }

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        for &(pos, _) in scored.iter().take(remaining) {
            selected.push(pos);
        }
        selected.sort_unstable();
        selected.dedup();

        let interior_count = (trail_start - (start + lead)) as u64;
        let sel_flops = interior_count * (2 * (d * d) as u64 + 4 * d as u64);
        (selected, sel_flops)
    }

    /// Execute all Transformer layers for `active_positions` against `kv_k` / `kv_v`,
    /// writing freshly computed K/V for active positions and returning exact final-token
    /// logits + total FLOPs.
    fn run_active_layers(
        &self,
        all_tokens: &[u32],
        active_positions: &[usize],
        kv_k: &mut [Vec<f32>],
        kv_v: &mut [Vec<f32>],
    ) -> (Vec<f32>, u64) {
        let d = self.config.d_model;
        let kv_dim = self.config.kv_dim();
        let d_ff = self.config.d_ff;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let heads_per_kv = num_heads / self.config.num_kv_heads;
        let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();
        let n_active = active_positions.len();

        let mut hidden: Vec<Vec<f32>> = active_positions
            .iter()
            .map(|&pos| self.embed_slice(all_tokens[pos]).to_vec())
            .collect();

        let mut q_buf = vec![0.0f32; n_active * d];
        let mut norm = vec![0.0f32; d];
        let mut attn_out = vec![0.0f32; d];
        let mut proj = vec![0.0f32; d];
        let mut gate = vec![0.0f32; d_ff];
        let mut up = vec![0.0f32; d_ff];
        let mut ffn_out = vec![0.0f32; d];
        let mut scores = Vec::with_capacity(all_tokens.len());

        let mut flops = 0u64;
        let proj_flops_per_tok = 2u64 * ((d * d + 2 * kv_dim * d + d * d + 3 * d_ff * d) as u64);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Pass 1: project Q, K, V for all active positions and write K, V into layer KV cache.
            for (idx, &pos) in active_positions.iter().enumerate() {
                rms_norm(
                    &hidden[idx],
                    &layer.attn_norm,
                    self.config.rms_norm_eps,
                    &mut norm,
                );

                let q_slot = &mut q_buf[idx * d..(idx + 1) * d];
                matvec(&layer.wq, d, d, &norm, q_slot);
                for head in 0..num_heads {
                    let hs = head * head_dim;
                    self.rope.apply_at_pos(&mut q_slot[hs..hs + head_dim], pos);
                }

                let k_slot = &mut kv_k[layer_idx][pos * kv_dim..(pos + 1) * kv_dim];
                let v_slot = &mut kv_v[layer_idx][pos * kv_dim..(pos + 1) * kv_dim];
                matvec(&layer.wk, kv_dim, d, &norm, k_slot);
                matvec(&layer.wv, kv_dim, d, &norm, v_slot);
                for kvh in 0..self.config.num_kv_heads {
                    let hs = kvh * head_dim;
                    self.rope.apply_at_pos(&mut k_slot[hs..hs + head_dim], pos);
                }
            }

            // Pass 2: causal GQA attention + O projection + SwiGLU FFN for each active position.
            let layer_k = &kv_k[layer_idx];
            let layer_v = &kv_v[layer_idx];
            for (idx, &pos) in active_positions.iter().enumerate() {
                let q_pos = &q_buf[idx * d..(idx + 1) * d];
                attn_out.fill(0.0);
                scores.resize(pos + 1, 0.0);

                for head in 0..num_heads {
                    let kvh = head / heads_per_kv;
                    let qh = &q_pos[head * head_dim..(head + 1) * head_dim];
                    let mut max_s = f32::NEG_INFINITY;
                    for (j, slot) in scores.iter_mut().enumerate() {
                        let kb = j * kv_dim + kvh * head_dim;
                        let kh = &layer_k[kb..kb + head_dim];
                        let mut dot = 0.0f32;
                        for c in 0..head_dim {
                            dot += qh[c] * kh[c];
                        }
                        let s = dot * inv_sqrt_hd;
                        *slot = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }

                    let mut sum_e = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_s).exp();
                        sum_e += *s;
                    }
                    let inv_e = 1.0 / sum_e.max(1e-30);
                    let oh = &mut attn_out[head * head_dim..(head + 1) * head_dim];
                    for (j, &w) in scores.iter().enumerate() {
                        let p = w * inv_e;
                        let vb = j * kv_dim + kvh * head_dim;
                        let vh = &layer_v[vb..vb + head_dim];
                        for c in 0..head_dim {
                            oh[c] += p * vh[c];
                        }
                    }
                }

                matvec(&layer.wo, d, d, &attn_out, &mut proj);
                let h = &mut hidden[idx];
                for i in 0..d {
                    h[i] += proj[i];
                }

                rms_norm(h, &layer.ffn_norm, self.config.rms_norm_eps, &mut norm);
                matvec(&layer.w_gate, d_ff, d, &norm, &mut gate);
                matvec(&layer.w_up, d_ff, d, &norm, &mut up);
                for i in 0..d_ff {
                    gate[i] = silu(gate[i]) * up[i];
                }
                matvec(&layer.w_down, d, d_ff, &gate, &mut ffn_out);
                for i in 0..d {
                    h[i] += ffn_out[i];
                }

                flops += proj_flops_per_tok + 4 * ((pos + 1) * d) as u64;
            }
        }

        // Final RMSNorm + LM head on the last prompt token (always at active_positions.last()).
        let last_idx = active_positions.len() - 1;
        debug_assert_eq!(active_positions[last_idx], all_tokens.len() - 1);
        rms_norm(
            &hidden[last_idx],
            &self.final_norm,
            self.config.rms_norm_eps,
            &mut norm,
        );
        let mut logits = vec![0.0f32; self.config.vocab_size];
        matvec(&self.lm_head, self.config.vocab_size, d, &norm, &mut logits);
        let scale = self.config.logit_scale;
        for v in &mut logits {
            *v *= scale;
        }
        flops += 2 * (self.config.vocab_size * d) as u64;

        (logits, flops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::{CorpusTokenizer, SegmentKind};

    fn test_scope() -> IsolationScope {
        IsolationScope::new("tenant_test", "acl_dev", "zen-tiny", KvDtype::F16)
    }

    #[test]
    fn prefix_cache_hit_is_numerically_identical_to_full_prefill() {
        let model = TransformerModel::new_deterministic(TransformerConfig::tiny_test(), 1346);
        let tok = CorpusTokenizer::new(256);
        let scope = test_scope();
        let pcache = PrefixKvCache::new(16 * 1024 * 1024);

        let sys = PromptSegment::new(
            SegmentKind::SystemPrompt,
            "sys",
            tok.encode_exact("You are Hanzo Zen assistant.", 32),
        );
        let tool = PromptSegment::new(
            SegmentKind::ToolDefinition,
            "tool",
            tok.encode_exact("{\"name\": \"search\", \"args\": {\"q\": \"string\"}}", 48),
        );
        let q1 = PromptSegment::new(
            SegmentKind::UserQuery,
            "q1",
            tok.encode_exact("Find Rust crates", 16),
        );
        let q2 = PromptSegment::new(
            SegmentKind::UserQuery,
            "q2",
            tok.encode_exact("List KV tests", 16),
        );

        // Warm prefix cache on turn 1.
        let _ = model.prefill_segments(
            &scope,
            &[sys.clone(), tool.clone(), q1],
            ExecutionMode::PrefixCacheOnly,
            Some(&pcache),
            None,
        );

        // Turn 2 shares [sys, tool] prefix exactly.
        let prompt2 = [sys, tool, q2];
        let exact =
            model.prefill_segments(&scope, &prompt2, ExecutionMode::FullPrefill, None, None);
        let cached = model.prefill_segments(
            &scope,
            &prompt2,
            ExecutionMode::PrefixCacheOnly,
            Some(&pcache),
            None,
        );

        assert_eq!(cached.tokens_reused, 80);
        assert_eq!(cached.tokens_computed, 16);
        let q = evaluate_quality(&exact.logits, &cached.logits);
        assert!(q.kl_divergence < 1e-6, "KL={}", q.kl_divergence);
        assert!(q.max_abs_logit_diff < 1e-4, "diff={}", q.max_abs_logit_diff);
        assert!(q.top1_agree);
    }

    #[test]
    fn cacheslide_reuses_shifted_tool_segment_where_prefix_cache_misses() {
        let model = TransformerModel::new_deterministic(TransformerConfig::tiny_test(), 1346);
        let tok = CorpusTokenizer::new(256);
        let scope = test_scope();
        let pcache = PrefixKvCache::new(16 * 1024 * 1024);
        let scache = SegmentKvCache::new(16 * 1024 * 1024, 16);

        let sys_a = PromptSegment::new(
            SegmentKind::SystemPrompt,
            "sys_a",
            tok.encode_exact("System prompt alpha session one", 32),
        );
        let sys_b = PromptSegment::new(
            SegmentKind::SystemPrompt,
            "sys_b",
            tok.encode_exact("System prompt beta updated profile context session two", 48),
        );
        let tool = PromptSegment::new(
            SegmentKind::ToolDefinition,
            "tool_schema",
            tok.encode_exact(
                "{\"tool\": \"execute_sql\", \"params\": {\"query\": \"string\", \"limit\": \"int\"}}",
                64,
            ),
        );
        let q1 = PromptSegment::new(
            SegmentKind::UserQuery,
            "q1",
            tok.encode_exact("query one", 16),
        );
        let q2 = PromptSegment::new(
            SegmentKind::UserQuery,
            "q2",
            tok.encode_exact("query two", 16),
        );

        // Warm both caches on [sys_a, tool, q1].
        let _ = model.prefill_segments(
            &scope,
            &[sys_a.clone(), tool.clone(), q1.clone()],
            ExecutionMode::PrefixCacheOnly,
            Some(&pcache),
            None,
        );
        let _ = model.prefill_segments(
            &scope,
            &[sys_a, tool.clone(), q1],
            ExecutionMode::CacheSlidePureShift,
            None,
            Some(&scache),
        );

        // Turn 2: sys_b changes prefix length from 32 -> 48, shifting tool by Δp = +16.
        let turn2 = [sys_b, tool, q2];
        let full = model.prefill_segments(&scope, &turn2, ExecutionMode::FullPrefill, None, None);
        let pref = model.prefill_segments(
            &scope,
            &turn2,
            ExecutionMode::PrefixCacheOnly,
            Some(&pcache),
            None,
        );
        let slide_pure = model.prefill_segments(
            &scope,
            &turn2,
            ExecutionMode::CacheSlidePureShift,
            None,
            Some(&scache),
        );
        let slide_full_recomp = model.prefill_segments(
            &scope,
            &turn2,
            ExecutionMode::CacheSlideSelectiveRecompute(SelectiveRecomputeConfig {
                recompute_ratio: 1.0,
                lead_boundary_tokens: 0,
                trail_boundary_tokens: 0,
                auto_escalate_drift_threshold: None,
                escalated_recompute_ratio: 1.0,
            }),
            None,
            Some(&scache),
        );

        // Prefix cache misses 100% of shifted tool segment.
        assert_eq!(pref.tokens_reused, 0);
        // CacheSlide pure shift reuses all 64 tool tokens with Δp = +16.
        assert_eq!(slide_pure.tokens_reused, 64);
        assert_eq!(slide_pure.segments_shifted, 1);
        assert!(slide_pure.prefill_flops < full.prefill_flops);

        // At 100% selective recomputation, CacheSlide matches FullPrefill exactly!
        let q_100 = evaluate_quality(&full.logits, &slide_full_recomp.logits);
        assert!(
            q_100.kl_divergence < 1e-6,
            "100% recomp KL={}",
            q_100.kl_divergence
        );
        assert!(q_100.top1_agree);
    }

    #[test]
    fn cross_tenant_request_gets_zero_kv_reuse() {
        let model = TransformerModel::new_deterministic(TransformerConfig::tiny_test(), 77);
        let tok = CorpusTokenizer::new(256);
        let scache = SegmentKvCache::new(16 * 1024 * 1024, 16);

        let scope_a = IsolationScope::new("tenant_a", "acl_eng", "zen-tiny", KvDtype::F16);
        let scope_b = IsolationScope::new("tenant_b", "acl_eng", "zen-tiny", KvDtype::F16);

        let prompt = [
            PromptSegment::new(SegmentKind::SystemPrompt, "s", tok.encode_exact("sys", 32)),
            PromptSegment::new(
                SegmentKind::ToolDefinition,
                "t",
                tok.encode_exact("shared secret tool schema", 64),
            ),
            PromptSegment::new(SegmentKind::UserQuery, "q", tok.encode_exact("hello", 16)),
        ];

        let _ = model.prefill_segments(
            &scope_a,
            &prompt,
            ExecutionMode::CacheSlidePureShift,
            None,
            Some(&scache),
        );

        let out_b = model.prefill_segments(
            &scope_b,
            &prompt,
            ExecutionMode::CacheSlidePureShift,
            None,
            Some(&scache),
        );
        assert_eq!(out_b.tokens_reused, 0);
        assert_eq!(out_b.segments_exact_hit, 0);
        assert_eq!(out_b.segments_shifted, 0);
    }

    #[test]
    fn layer1_drift_gate_triggers_escalation_on_divergent_prefix() {
        let model = TransformerModel::new_deterministic(TransformerConfig::tiny_test(), 1346);
        let tok = CorpusTokenizer::new(256);
        let scache = SegmentKvCache::new(16 * 1024 * 1024, 16);
        let scope = test_scope();

        let p1 = PromptSegment::new(
            SegmentKind::SystemPrompt,
            "p1",
            tok.encode_exact("alpha alpha alpha 11111", 32),
        );
        let p2 = PromptSegment::new(
            SegmentKind::SystemPrompt,
            "p2",
            tok.encode_exact("ZZZZZ ##### !!!!! 99999 override context drastically", 64),
        );
        let seg = PromptSegment::new(
            SegmentKind::RagChunk,
            "doc",
            tok.encode_exact("Target document segment body for cross context test", 48),
        );
        let q1 = PromptSegment::new(
            SegmentKind::UserQuery,
            "q1",
            tok.encode_exact("initial question", 16),
        );
        let q2 = PromptSegment::new(
            SegmentKind::UserQuery,
            "q2",
            tok.encode_exact("second distinct query", 16),
        );

        let _ = model.prefill_segments(
            &scope,
            &[p1, seg.clone(), q1],
            ExecutionMode::CacheSlidePureShift,
            None,
            Some(&scache),
        );

        let gated = model.prefill_segments(
            &scope,
            &[p2, seg, q2],
            ExecutionMode::CacheSlideSelectiveRecompute(SelectiveRecomputeConfig {
                recompute_ratio: 0.15,
                lead_boundary_tokens: 2,
                trail_boundary_tokens: 2,
                auto_escalate_drift_threshold: Some(0.01),
                escalated_recompute_ratio: 1.0,
            }),
            None,
            Some(&scache),
        );
        assert!(gated.max_layer1_k_drift > 0.01);
        assert_eq!(gated.drift_gate_escalations, 1);
    }

    #[test]
    fn exact_prefix_taint_prevents_stale_approximation_compounding() {
        let model = TransformerModel::new_deterministic(TransformerConfig::tiny_test(), 99);
        let tok = CorpusTokenizer::new(256);
        let scache = SegmentKvCache::new(16 * 1024 * 1024, 16);
        let scope = test_scope();

        let p1 = PromptSegment::new(
            SegmentKind::SystemPrompt,
            "p1",
            tok.encode_exact("prefix one", 32),
        );
        let p2 = PromptSegment::new(
            SegmentKind::SystemPrompt,
            "p2",
            tok.encode_exact("different longer prefix two", 48),
        );
        let tool = PromptSegment::new(
            SegmentKind::ToolDefinition,
            "tool",
            tok.encode_exact("reusable tool definition body", 48),
        );
        let suffix = PromptSegment::new(
            SegmentKind::UserQuery,
            "suffix",
            tok.encode_exact("downstream suffix segment", 24),
        );

        // Turn 1: cache exact KV for [p1, tool, suffix_other].
        let _ = model.prefill_segments(
            &scope,
            &[
                p1,
                tool.clone(),
                PromptSegment::new(SegmentKind::UserQuery, "q0", tok.encode_exact("warmup", 24)),
            ],
            ExecutionMode::CacheSlidePureShift,
            None,
            Some(&scache),
        );

        // Turn 2 (pure shift): tool is shifted (approximate), so newly cached `suffix` MUST
        // carry exact_prefix_valid == false.
        let prompt2 = [p2, tool, suffix.clone()];
        let _ = model.prefill_segments(
            &scope,
            &prompt2,
            ExecutionMode::CacheSlidePureShift,
            None,
            Some(&scache),
        );
        let cached_suffix = scache
            .get(&SegmentKey::from_segment(scope.clone(), &suffix))
            .expect("suffix should be cached");
        assert!(
            !cached_suffix.exact_prefix_valid,
            "segment computed after a pure-shifted upstream block must not be marked exact"
        );

        // Turn 3 (100% selective recompute on identical prompt2): must NOT treat tainted `suffix`
        // as an exact prefix hit; must match FullPrefill to < 1e-6 KL!
        let exact =
            model.prefill_segments(&scope, &prompt2, ExecutionMode::FullPrefill, None, None);
        let full_recomp = model.prefill_segments(
            &scope,
            &prompt2,
            ExecutionMode::CacheSlideSelectiveRecompute(SelectiveRecomputeConfig {
                recompute_ratio: 1.0,
                lead_boundary_tokens: 0,
                trail_boundary_tokens: 0,
                auto_escalate_drift_threshold: None,
                escalated_recompute_ratio: 1.0,
            }),
            None,
            Some(&scache),
        );
        let q = evaluate_quality(&exact.logits, &full_recomp.logits);
        assert!(
            q.kl_divergence < 1e-6,
            "tainted suffix leaked KL={}",
            q.kl_divergence
        );
    }
}
