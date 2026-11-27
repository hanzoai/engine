// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust fused ragged prefill + decode paged attention over unified CSR-like
//! jagged batch descriptors (`qo_indptr`, `kv_indptr`, `kv_page_indices`, `kv_last_page_len`).
//!
//! ## Overview
//! Continuous-batching servers routinely mix chunked prefill requests (`q_len > 1`)
//! and single-token decode requests (`q_len == 1`) within the same forward step.
//! Executing prefill and decode in separate kernel stages introduces a serialization
//! barrier and doubles kernel-launch metadata overhead.
//!
//! A unified ragged batch descriptor packs all active query tokens contiguously with
//! `qo_indptr[i]..qo_indptr[i+1]` indexing request `i`'s query rows, while
//! `kv_indptr[i]..kv_indptr[i+1]` indexes request `i`'s physical KV pages in
//! `kv_page_indices`. Exact causal masking restricts query token `q_local` of request `i`
//! to KV positions `0 ..= (kv_seq_len - q_len) + q_local`.

use crate::cascade::{greedy_argmax_tokens, Lcg64, PagedKvPool};

/// Errors produced when validating a [`RaggedBatchDescriptor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaggedDescriptorError {
    /// `qo_indptr` or `kv_indptr` is empty or has mismatched batch length.
    InvalidIndptrLength,
    /// An indptr array does not start at `0` or is not monotonically non-decreasing.
    NonMonotonicIndptr,
    /// `kv_page_indices.len()` does not match the terminal `kv_indptr` offset.
    PageIndicesLengthMismatch,
    /// A request has `kv_seq_len < q_len` (violates causal self-attention invariant).
    CausalLengthViolation {
        /// Request index in the ragged batch.
        req_idx: usize,
        /// Query length for the request.
        q_len: usize,
        /// Total KV length for the request.
        kv_len: usize,
    },
    /// A physical page index is out of bounds for the supplied [`PagedKvPool`].
    PageOutOfBounds {
        /// Out-of-bounds physical page ID.
        page_id: usize,
        /// Number of physical pages present in the pool.
        num_pages: usize,
    },
    /// `kv_last_page_len` does not match the page count and `kv_seq_len`.
    LastPageLenMismatch {
        /// Request index in the ragged batch.
        req_idx: usize,
    },
}

/// Unified CSR-like ragged batch descriptor covering mixed prefill chunks and decode tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaggedBatchDescriptor {
    /// Query/output token offsets (`len = batch_size + 1`, `qo_indptr[0] = 0`).
    pub qo_indptr: Vec<usize>,
    /// KV page offsets (`len = batch_size + 1`, `kv_indptr[0] = 0`).
    pub kv_indptr: Vec<usize>,
    /// Flat physical page table indexed by `kv_indptr[i]..kv_indptr[i+1]`.
    pub kv_page_indices: Vec<usize>,
    /// Number of valid tokens in each request's last physical KV page (`1..=page_size`).
    pub kv_last_page_len: Vec<usize>,
    /// Total logical KV sequence length for each request (`past_len + q_len`).
    pub kv_seq_lens: Vec<usize>,
}

impl RaggedBatchDescriptor {
    /// Number of logical requests in the ragged batch.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.qo_indptr.len().saturating_sub(1)
    }

    /// Total packed query/output tokens across all prefill chunks and decode requests.
    #[must_use]
    pub fn total_q_tokens(&self) -> usize {
        self.qo_indptr.last().copied().unwrap_or(0)
    }

    /// Number of requests in the batch that are prefill chunks (`q_len > 1`).
    #[must_use]
    pub fn prefill_request_count(&self) -> usize {
        (0..self.batch_size())
            .filter(|&i| self.qo_indptr[i + 1] - self.qo_indptr[i] > 1)
            .count()
    }

    /// Number of requests in the batch that are single-token decode steps (`q_len == 1`).
    #[must_use]
    pub fn decode_request_count(&self) -> usize {
        (0..self.batch_size())
            .filter(|&i| self.qo_indptr[i + 1] - self.qo_indptr[i] == 1)
            .count()
    }

    /// Validates all structural, causal, and physical page-table invariants against `pool`.
    pub fn validate(&self, pool: &PagedKvPool) -> Result<(), RaggedDescriptorError> {
        if self.qo_indptr.len() < 2
            || self.kv_indptr.len() != self.qo_indptr.len()
            || self.kv_last_page_len.len() != self.batch_size()
            || self.kv_seq_lens.len() != self.batch_size()
        {
            return Err(RaggedDescriptorError::InvalidIndptrLength);
        }
        if self.qo_indptr[0] != 0 || self.kv_indptr[0] != 0 {
            return Err(RaggedDescriptorError::NonMonotonicIndptr);
        }
        for i in 0..self.batch_size() {
            if self.qo_indptr[i + 1] < self.qo_indptr[i]
                || self.kv_indptr[i + 1] < self.kv_indptr[i]
            {
                return Err(RaggedDescriptorError::NonMonotonicIndptr);
            }
        }
        if self.kv_indptr.last().copied().unwrap_or(0) != self.kv_page_indices.len() {
            return Err(RaggedDescriptorError::PageIndicesLengthMismatch);
        }
        let num_pages = pool.k_pages.len();
        for &pid in &self.kv_page_indices {
            if pid >= num_pages {
                return Err(RaggedDescriptorError::PageOutOfBounds {
                    page_id: pid,
                    num_pages,
                });
            }
        }
        for i in 0..self.batch_size() {
            let q_len = self.qo_indptr[i + 1] - self.qo_indptr[i];
            let kv_len = self.kv_seq_lens[i];
            if kv_len < q_len {
                return Err(RaggedDescriptorError::CausalLengthViolation {
                    req_idx: i,
                    q_len,
                    kv_len,
                });
            }
            let req_pages = self.kv_indptr[i + 1] - self.kv_indptr[i];
            let expected_pages = kv_len.div_ceil(pool.page_size);
            if req_pages != expected_pages {
                return Err(RaggedDescriptorError::LastPageLenMismatch { req_idx: i });
            }
            let expected_last = if kv_len == 0 {
                0
            } else {
                let rem = kv_len % pool.page_size;
                if rem == 0 {
                    pool.page_size
                } else {
                    rem
                }
            };
            if self.kv_last_page_len[i] != expected_last {
                return Err(RaggedDescriptorError::LastPageLenMismatch { req_idx: i });
            }
        }
        Ok(())
    }
}

/// Output and dispatch statistics from executing ragged prefill/decode attention.
#[derive(Debug, Clone, PartialEq)]
pub struct RaggedAttentionResult {
    /// Packed output buffer `[total_q_tokens, num_q_heads, head_dim]`.
    pub output: Vec<f32>,
    /// Per-`(q_token, q_head)` exact log-sum-exp normalizers.
    pub lse: Vec<f32>,
    /// Number of distinct attention kernel dispatch stages executed (`2` separate vs `1` fused).
    pub dispatch_stages: usize,
    /// Total FP32 KV bytes read across all queries in the batch.
    pub total_kv_bytes_fp32: u64,
}

/// Evaluates causal paged GQA for one request slice `req_idx` into `output`/`lse`.
#[allow(clippy::too_many_arguments)]
fn eval_single_ragged_request(
    pool: &PagedKvPool,
    queries: &[f32],
    desc: &RaggedBatchDescriptor,
    req_idx: usize,
    num_q_heads: usize,
    scale: f32,
    output: &mut [f32],
    lse: &mut [f32],
) -> u64 {
    let d = pool.head_dim;
    let gqa_group = num_q_heads / pool.num_kv_heads;
    let q_start = desc.qo_indptr[req_idx];
    let q_end = desc.qo_indptr[req_idx + 1];
    let q_len = q_end - q_start;
    let kv_len = desc.kv_seq_lens[req_idx];
    let past_len = kv_len - q_len;
    let page_slice = &desc.kv_page_indices[desc.kv_indptr[req_idx]..desc.kv_indptr[req_idx + 1]];

    let mut kv_bytes: u64 = 0;

    for q_local in 0..q_len {
        let q_global = q_start + q_local;
        let attended_kv_len = past_len + q_local + 1;

        for qh in 0..num_q_heads {
            let kv_h = qh / gqa_group;
            let head_idx = q_global * num_q_heads + qh;
            let q_off = head_idx * d;
            let q_vec = &queries[q_off..q_off + d];

            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            let mut acc = vec![0.0f32; d];
            let mut remaining = attended_kv_len;
            let head_offset = kv_h * pool.page_size * d;

            for &pid in page_slice {
                if remaining == 0 {
                    break;
                }
                let valid = remaining.min(pool.page_size);
                kv_bytes += (2 * valid * d * 4) as u64;
                let k_page = &pool.k_pages[pid];
                let v_page = &pool.v_pages[pid];

                let mut tile_scores = [0.0f32; 64];
                debug_assert!(valid <= tile_scores.len());
                let mut tile_max = f32::NEG_INFINITY;
                for (t, score_slot) in tile_scores.iter_mut().enumerate().take(valid) {
                    let base = head_offset + t * d;
                    let k_vec = &k_page[base..base + d];
                    let mut dot = 0.0f32;
                    for i in 0..d {
                        dot += q_vec[i] * k_vec[i];
                    }
                    let s = dot * scale;
                    *score_slot = s;
                    if s > tile_max {
                        tile_max = s;
                    }
                }

                let new_m = m.max(tile_max);
                let alpha = if m.is_finite() {
                    (m - new_m).exp()
                } else {
                    0.0
                };
                l *= alpha;
                for val in &mut acc {
                    *val *= alpha;
                }
                for (t, &score) in tile_scores.iter().enumerate().take(valid) {
                    let p = (score - new_m).exp();
                    l += p;
                    let base = head_offset + t * d;
                    let v_vec = &v_page[base..base + d];
                    for i in 0..d {
                        acc[i] += p * v_vec[i];
                    }
                }
                m = new_m;
                remaining -= valid;
            }

            let inv_l = 1.0 / l;
            for i in 0..d {
                output[q_off + i] = acc[i] * inv_l;
            }
            lse[head_idx] = m + l.ln();
        }
    }

    kv_bytes
}

/// Baseline separate prefill + decode execution: splits the batch into a prefill stage
/// (`q_len > 1`) followed by a decode stage (`q_len == 1`), incurring `2` dispatch stages
/// when both phases are present.
pub fn separate_prefill_and_decode_paged_gqa(
    pool: &PagedKvPool,
    queries: &[f32],
    desc: &RaggedBatchDescriptor,
    num_q_heads: usize,
) -> Result<RaggedAttentionResult, RaggedDescriptorError> {
    desc.validate(pool)?;
    let d = pool.head_dim;
    let total_q = desc.total_q_tokens();
    assert_eq!(queries.len(), total_q * num_q_heads * d);
    assert_eq!(num_q_heads % pool.num_kv_heads, 0);

    let scale = 1.0 / (d as f32).sqrt();
    let mut output = vec![0.0f32; total_q * num_q_heads * d];
    let mut lse = vec![f32::NEG_INFINITY; total_q * num_q_heads];
    let mut total_kv_bytes: u64 = 0;
    let mut stages = 0usize;

    // Stage 1: prefill chunks (`q_len > 1`).
    let mut ran_prefill = false;
    for req_idx in 0..desc.batch_size() {
        let q_len = desc.qo_indptr[req_idx + 1] - desc.qo_indptr[req_idx];
        if q_len > 1 {
            ran_prefill = true;
            total_kv_bytes += eval_single_ragged_request(
                pool,
                queries,
                desc,
                req_idx,
                num_q_heads,
                scale,
                &mut output,
                &mut lse,
            );
        }
    }
    if ran_prefill {
        stages += 1;
    }

    // Stage 2: decode tokens (`q_len == 1`).
    let mut ran_decode = false;
    for req_idx in 0..desc.batch_size() {
        let q_len = desc.qo_indptr[req_idx + 1] - desc.qo_indptr[req_idx];
        if q_len == 1 {
            ran_decode = true;
            total_kv_bytes += eval_single_ragged_request(
                pool,
                queries,
                desc,
                req_idx,
                num_q_heads,
                scale,
                &mut output,
                &mut lse,
            );
        }
    }
    if ran_decode {
        stages += 1;
    }

    Ok(RaggedAttentionResult {
        output,
        lse,
        dispatch_stages: stages,
        total_kv_bytes_fp32: total_kv_bytes,
    })
}

/// Fused ragged prefill + decode paged attention executing all mixed prefill chunks and
/// decode requests in a single unified pass over `qo_indptr` / `kv_indptr` / `kv_page_indices`.
pub fn fused_ragged_prefill_decode_gqa(
    pool: &PagedKvPool,
    queries: &[f32],
    desc: &RaggedBatchDescriptor,
    num_q_heads: usize,
) -> Result<RaggedAttentionResult, RaggedDescriptorError> {
    desc.validate(pool)?;
    let d = pool.head_dim;
    let total_q = desc.total_q_tokens();
    assert_eq!(queries.len(), total_q * num_q_heads * d);
    assert_eq!(num_q_heads % pool.num_kv_heads, 0);

    let scale = 1.0 / (d as f32).sqrt();
    let mut output = vec![0.0f32; total_q * num_q_heads * d];
    let mut lse = vec![f32::NEG_INFINITY; total_q * num_q_heads];
    let mut total_kv_bytes: u64 = 0;

    for req_idx in 0..desc.batch_size() {
        total_kv_bytes += eval_single_ragged_request(
            pool,
            queries,
            desc,
            req_idx,
            num_q_heads,
            scale,
            &mut output,
            &mut lse,
        );
    }

    Ok(RaggedAttentionResult {
        output,
        lse,
        dispatch_stages: 1,
        total_kv_bytes_fp32: total_kv_bytes,
    })
}

/// Summary of the directly measured CPU ragged prefill+decode equivalence verification.
#[derive(Debug, Clone)]
pub struct CpuRaggedVerificationReport {
    /// Number of prefill chunk requests verified.
    pub prefill_requests: usize,
    /// Number of decode requests verified.
    pub decode_requests: usize,
    /// Total packed query tokens verified across mixed prefill and decode.
    pub total_q_tokens: usize,
    /// Dispatch stages for separate prefill/decode (`2`).
    pub separate_dispatch_stages: usize,
    /// Dispatch stages for fused ragged prefill/decode (`1`).
    pub fused_dispatch_stages: usize,
    /// Maximum absolute FP32 output error between separate and fused ragged execution.
    pub max_abs_output_err: f32,
    /// Maximum absolute FP32 LSE error between separate and fused ragged execution.
    pub max_abs_lse_err: f32,
    /// Greedy argmax token match rate (`100.0` = exact).
    pub greedy_argmax_match_pct: f64,
}

/// Runs the directly measured CPU equivalence check comparing separate prefill+decode
/// against fused ragged prefill+decode on a mixed batch.
#[must_use]
pub fn run_cpu_ragged_verification(seed: u64) -> CpuRaggedVerificationReport {
    let mut rng = Lcg64::new(seed ^ 0x51a7_c0de);
    let num_q_heads = 8usize;
    let num_kv_heads = 2usize;
    let head_dim = 64usize;
    let page_size = 16usize;
    let hidden_dim = num_q_heads * head_dim;
    let vocab_size = 64usize;

    let mut pool = PagedKvPool::new(num_kv_heads, page_size, head_dim);
    // Mixed batch: 2 chunked prefills (q_len=32, past=64; q_len=48, past=32) +
    // 6 single-token decodes at varying KV lengths (65..193).
    let req_specs = [
        (32usize, 96usize), // prefill chunk 1: past=64, q=32
        (1usize, 73usize),  // decode 1
        (1usize, 129usize), // decode 2
        (48usize, 80usize), // prefill chunk 2: past=32, q=48
        (1usize, 160usize), // decode 3
        (1usize, 193usize), // decode 4
        (1usize, 65usize),  // decode 5
        (1usize, 112usize), // decode 6
    ];

    let mut qo_indptr = vec![0usize];
    let mut kv_indptr = vec![0usize];
    let mut kv_page_indices = Vec::new();
    let mut kv_last_page_len = Vec::new();
    let mut kv_seq_lens = Vec::new();

    for &(q_len, kv_len) in &req_specs {
        let pages = pool.allocate_random_sequence(kv_len, &mut rng);
        kv_page_indices.extend_from_slice(&pages);
        qo_indptr.push(qo_indptr.last().copied().unwrap_or(0) + q_len);
        kv_indptr.push(kv_page_indices.len());
        let rem = kv_len % page_size;
        kv_last_page_len.push(if rem == 0 { page_size } else { rem });
        kv_seq_lens.push(kv_len);
    }

    let desc = RaggedBatchDescriptor {
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        kv_last_page_len,
        kv_seq_lens,
    };

    let total_q = desc.total_q_tokens();
    let queries: Vec<f32> = (0..total_q * num_q_heads * head_dim)
        .map(|_| rng.next_f32_signed() * 0.4)
        .collect();
    let proj_weights: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|_| rng.next_f32_signed() * 0.25)
        .collect();

    let sep = separate_prefill_and_decode_paged_gqa(&pool, &queries, &desc, num_q_heads)
        .expect("valid descriptor must succeed on separate path");
    let fused = fused_ragged_prefill_decode_gqa(&pool, &queries, &desc, num_q_heads)
        .expect("valid descriptor must succeed on fused path");

    let mut max_out_err = 0.0f32;
    for (&a, &b) in sep.output.iter().zip(fused.output.iter()) {
        max_out_err = max_out_err.max((a - b).abs());
    }
    let mut max_lse_err = 0.0f32;
    for (&a, &b) in sep.lse.iter().zip(fused.lse.iter()) {
        max_lse_err = max_lse_err.max((a - b).abs());
    }

    let sep_toks =
        greedy_argmax_tokens(&sep.output, total_q, hidden_dim, vocab_size, &proj_weights);
    let fused_toks = greedy_argmax_tokens(
        &fused.output,
        total_q,
        hidden_dim,
        vocab_size,
        &proj_weights,
    );
    let matched = sep_toks
        .iter()
        .zip(fused_toks.iter())
        .filter(|(a, b)| a == b)
        .count();

    CpuRaggedVerificationReport {
        prefill_requests: desc.prefill_request_count(),
        decode_requests: desc.decode_request_count(),
        total_q_tokens: total_q,
        separate_dispatch_stages: sep.dispatch_stages,
        fused_dispatch_stages: fused.dispatch_stages,
        max_abs_output_err: max_out_err,
        max_abs_lse_err: max_lse_err,
        greedy_argmax_match_pct: (matched as f64) * 100.0 / (total_q as f64),
    }
}
