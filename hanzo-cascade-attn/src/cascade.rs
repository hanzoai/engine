// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust tiled online-softmax paged grouped-query attention (GQA) and
//! multi-level shared-prefix cascade attention with exact Log-Sum-Exp (LSE) state merge.
//!
//! ## Mathematical Formulation
//! Partitioning a request's key-value sequence $S = S_0 \cup S_1$ into a shared prefix
//! $S_0$ (reused across all $B$ requests in a batch) and a unique suffix $S_1$ yields
//! partial attention states $(O_0, \ell_0)$ and $(O_1, \ell_1)$, where
//! $\ell_k = \log \sum_{j \in S_k} \exp(q^\top k_j / \sqrt{d})$ is the exact log-sum-exp
//! normalizer. Merging the two partial states is mathematically exact:
//!
//! ```text
//! m      = max(lse_0, lse_1)
//! w_0    = exp(lse_0 - m)
//! w_1    = exp(lse_1 - m)
//! O      = (w_0 * O_0 + w_1 * O_1) / (w_0 + w_1)
//! lse    = m + ln(w_0 + w_1)
//! ```
//!
//! By induction, the same associative merge extends to arbitrary $L$-level prefix trees
//! (e.g., global system prompt -> repository sub-group prefix -> unique request suffix).
//! Loading each shared-prefix KV page once across all $B$ queries reduces shared-prefix
//! KV memory traffic by a factor of $B$, at the cost of a small LSE merge overhead that
//! only pays off when $B$ and the shared prefix length are sufficiently large.

use std::time::Instant;

/// Deterministic 64-bit linear congruential PRNG (zero external dependencies).
#[derive(Debug, Clone)]
pub struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    /// Creates a new deterministic PRNG initialized with `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    /// Advances the generator and returns the next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Returns a deterministic `f32` in `[-1.0, 1.0)`.
    pub fn next_f32_signed(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        let unit = (bits as f32) / ((1u32 << 24) as f32);
        unit * 2.0 - 1.0
    }
}

/// Exact byte-level KV memory traffic statistics captured during CPU attention execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvReadStats {
    /// Bytes read from shared-prefix KV pages (`K` and `V`, FP16-equivalent 2 bytes/elem and FP32 storage).
    pub shared_prefix_kv_bytes_fp32: u64,
    /// Bytes read from unique-suffix KV pages.
    pub suffix_kv_bytes_fp32: u64,
    /// Total KV bytes read across all levels.
    pub total_kv_bytes_fp32: u64,
    /// Additional bytes written and read for partial `(O, lse)` cascade state merge.
    pub lse_merge_bytes_rw: u64,
    /// Number of physical KV page loads issued.
    pub page_loads: u64,
}

impl KvReadStats {
    /// Total memory traffic including KV reads and LSE merge scratch traffic.
    #[must_use]
    pub const fn total_traffic_bytes(&self) -> u64 {
        self.total_kv_bytes_fp32 + self.lse_merge_bytes_rw
    }
}

/// Output tensor and per-`(batch, q_head)` log-sum-exp normalizers from an attention pass.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionOutput {
    /// Flattened `[batch_size, num_q_heads, head_dim]` output buffer.
    pub output: Vec<f32>,
    /// Flattened `[batch_size, num_q_heads]` natural-log sum-exp values.
    pub lse: Vec<f32>,
    /// Exact KV read and merge byte accounting recorded during execution.
    pub stats: KvReadStats,
}

/// Physical paged KV cache pool storing fixed-size KV pages for grouped-query attention.
///
/// Layout per physical page:
/// `k_pages[page_id]` has shape `[num_kv_heads, page_size, head_dim]`, and
/// `v_pages[page_id]` has shape `[num_kv_heads, page_size, head_dim]`.
#[derive(Debug, Clone)]
pub struct PagedKvPool {
    /// Number of KV heads per layer.
    pub num_kv_heads: usize,
    /// Number of tokens stored in each physical KV page.
    pub page_size: usize,
    /// Head dimension (`head_dim`).
    pub head_dim: usize,
    /// Physical key pages (`[num_pages][num_kv_heads * page_size * head_dim]`).
    pub k_pages: Vec<Vec<f32>>,
    /// Physical value pages (`[num_pages][num_kv_heads * page_size * head_dim]`).
    pub v_pages: Vec<Vec<f32>>,
}

impl PagedKvPool {
    /// Creates an empty paged KV pool with the specified head geometry and page size.
    #[must_use]
    pub fn new(num_kv_heads: usize, page_size: usize, head_dim: usize) -> Self {
        assert!(num_kv_heads > 0, "num_kv_heads must be positive");
        assert!(page_size > 0, "page_size must be positive");
        assert!(head_dim > 0, "head_dim must be positive");
        Self {
            num_kv_heads,
            page_size,
            head_dim,
            k_pages: Vec::new(),
            v_pages: Vec::new(),
        }
    }

    /// Number of `f32` elements stored in one key (or value) page across all KV heads.
    #[must_use]
    pub const fn page_elems(&self) -> usize {
        self.num_kv_heads * self.page_size * self.head_dim
    }

    /// Allocates and populates a sequence of `seq_len` random KV tokens into physical pages,
    /// returning the list of allocated physical page IDs.
    pub fn allocate_random_sequence(&mut self, seq_len: usize, rng: &mut Lcg64) -> Vec<usize> {
        let num_pages = seq_len.div_ceil(self.page_size);
        let mut page_ids = Vec::with_capacity(num_pages);
        for p in 0..num_pages {
            let valid_tokens = (seq_len - p * self.page_size).min(self.page_size);
            let mut k_page = vec![0.0f32; self.page_elems()];
            let mut v_page = vec![0.0f32; self.page_elems()];
            for kv_h in 0..self.num_kv_heads {
                for tok in 0..valid_tokens {
                    let base = (kv_h * self.page_size + tok) * self.head_dim;
                    for d in 0..self.head_dim {
                        k_page[base + d] = rng.next_f32_signed() * 0.35;
                        v_page[base + d] = rng.next_f32_signed() * 0.35;
                    }
                }
            }
            let pid = self.k_pages.len();
            self.k_pages.push(k_page);
            self.v_pages.push(v_page);
            page_ids.push(pid);
        }
        page_ids
    }

    /// Returns a slice of key elements for `(page_id, kv_head, tok_in_page)`.
    #[inline]
    #[must_use]
    pub fn key_slice(&self, page_id: usize, kv_head: usize, tok_in_page: usize) -> &[f32] {
        let base = (kv_head * self.page_size + tok_in_page) * self.head_dim;
        &self.k_pages[page_id][base..base + self.head_dim]
    }

    /// Returns a slice of value elements for `(page_id, kv_head, tok_in_page)`.
    #[inline]
    #[must_use]
    pub fn val_slice(&self, page_id: usize, kv_head: usize, tok_in_page: usize) -> &[f32] {
        let base = (kv_head * self.page_size + tok_in_page) * self.head_dim;
        &self.v_pages[page_id][base..base + self.head_dim]
    }
}

/// Exact numerically stable two-level Log-Sum-Exp (LSE) state merge for a single head vector.
///
/// Merges partial attention outputs `(o0, lse0)` and `(o1, lse1)` into `(merged_out, merged_lse)`.
/// Handles empty partitions (`lse == f32::NEG_INFINITY`) cleanly without `NaN`.
#[must_use]
pub fn merge_lse_states(o0: &[f32], lse0: f32, o1: &[f32], lse1: f32) -> (Vec<f32>, f32) {
    assert_eq!(o0.len(), o1.len(), "output head dimensions must match");
    let head_dim = o0.len();
    if !lse0.is_finite() && lse0.is_sign_negative() {
        return (o1.to_vec(), lse1);
    }
    if !lse1.is_finite() && lse1.is_sign_negative() {
        return (o0.to_vec(), lse0);
    }
    let m = lse0.max(lse1);
    let w0 = (lse0 - m).exp();
    let w1 = (lse1 - m).exp();
    let denom = w0 + w1;
    let inv_denom = 1.0 / denom;
    let mut out = vec![0.0f32; head_dim];
    for d in 0..head_dim {
        out[d] = (w0 * o0[d] + w1 * o1[d]) * inv_denom;
    }
    (out, m + denom.ln())
}

/// Exact numerically stable multi-level ($L$-level) Log-Sum-Exp state merge.
///
/// Merges an arbitrary slice of partial attention states `&[(&[f32], f32)]` in a single
/// normalized pass using global maximum shifting `m = max_k(lse_k)`.
#[must_use]
pub fn merge_lse_multi_level(levels: &[(&[f32], f32)]) -> (Vec<f32>, f32) {
    assert!(!levels.is_empty(), "must provide at least one level");
    let head_dim = levels[0].0.len();
    let mut m = f32::NEG_INFINITY;
    for &(out_k, lse_k) in levels {
        assert_eq!(out_k.len(), head_dim, "all levels must share head_dim");
        if lse_k > m {
            m = lse_k;
        }
    }
    if !m.is_finite() {
        return (vec![0.0f32; head_dim], f32::NEG_INFINITY);
    }
    let mut accum = vec![0.0f32; head_dim];
    let mut denom = 0.0f32;
    for &(out_k, lse_k) in levels {
        if !lse_k.is_finite() && lse_k.is_sign_negative() {
            continue;
        }
        let w = (lse_k - m).exp();
        denom += w;
        for d in 0..head_dim {
            accum[d] += w * out_k[d];
        }
    }
    let inv_denom = 1.0 / denom;
    for val in &mut accum {
        *val *= inv_denom;
    }
    (accum, m + denom.ln())
}

/// Computes a single query vector's tiled online-softmax paged attention over a slice of physical pages.
///
/// Returns `(output_vec, lse, kv_bytes_fp32_read, pages_loaded)`.
fn paged_single_query_online_softmax(
    pool: &PagedKvPool,
    q: &[f32],
    kv_head: usize,
    page_ids: &[usize],
    total_tokens: usize,
    scale: f32,
) -> (Vec<f32>, f32, u64, u64) {
    let d = pool.head_dim;
    if total_tokens == 0 || page_ids.is_empty() {
        return (vec![0.0f32; d], f32::NEG_INFINITY, 0, 0);
    }
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    let mut acc = vec![0.0f32; d];
    let mut remaining = total_tokens;
    let mut kv_bytes: u64 = 0;
    let mut pages_loaded: u64 = 0;

    for &pid in page_ids {
        if remaining == 0 {
            break;
        }
        let valid = remaining.min(pool.page_size);
        pages_loaded += 1;
        // 2 tensors (K + V) * valid tokens * head_dim * 4 bytes (FP32)
        kv_bytes += (2 * valid * d * 4) as u64;

        let k_page = &pool.k_pages[pid];
        let v_page = &pool.v_pages[pid];
        let head_offset = kv_head * pool.page_size * d;

        // First compute tile max for numerical stability (tiled online softmax).
        let mut tile_scores = [0.0f32; 64];
        debug_assert!(valid <= tile_scores.len());
        let mut tile_max = f32::NEG_INFINITY;
        for (t, score_slot) in tile_scores.iter_mut().enumerate().take(valid) {
            let base = head_offset + t * d;
            let k_vec = &k_page[base..base + d];
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += q[i] * k_vec[i];
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
    for val in &mut acc {
        *val *= inv_l;
    }
    let lse = m + l.ln();
    (acc, lse, kv_bytes, pages_loaded)
}

/// Standard unshared tiled online-softmax paged GQA decode across a batch of `B` requests.
///
/// Every request `b` independently reads both the shared-prefix pages and its own unique-suffix
/// pages, resulting in `B * prefix_len` KV reads over the shared prefix.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn unshared_paged_gqa_decode(
    pool: &PagedKvPool,
    queries: &[f32],
    batch_size: usize,
    num_q_heads: usize,
    shared_prefix_pages: &[usize],
    shared_prefix_len: usize,
    suffix_pages_per_req: &[Vec<usize>],
    suffix_lens_per_req: &[usize],
) -> AttentionOutput {
    let d = pool.head_dim;
    let gqa_group = num_q_heads / pool.num_kv_heads;
    assert_eq!(num_q_heads % pool.num_kv_heads, 0);
    assert_eq!(queries.len(), batch_size * num_q_heads * d);
    assert_eq!(suffix_pages_per_req.len(), batch_size);
    assert_eq!(suffix_lens_per_req.len(), batch_size);

    let scale = 1.0 / (d as f32).sqrt();
    let mut output = vec![0.0f32; batch_size * num_q_heads * d];
    let mut lse = vec![f32::NEG_INFINITY; batch_size * num_q_heads];
    let mut stats = KvReadStats::default();

    for b in 0..batch_size {
        let mut combined_pages =
            Vec::with_capacity(shared_prefix_pages.len() + suffix_pages_per_req[b].len());
        combined_pages.extend_from_slice(shared_prefix_pages);
        combined_pages.extend_from_slice(&suffix_pages_per_req[b]);
        let total_len = shared_prefix_len + suffix_lens_per_req[b];

        // When shared_prefix_len is not a multiple of page_size, evaluate prefix and suffix
        // page spans cleanly so logical token boundaries match exactly.
        let prefix_aligned = shared_prefix_len.is_multiple_of(pool.page_size);

        for qh in 0..num_q_heads {
            let kv_h = qh / gqa_group;
            let q_offset = (b * num_q_heads + qh) * d;
            let q_vec = &queries[q_offset..q_offset + d];

            let (out_vec, head_lse, pref_bytes, suff_bytes, pages) = if prefix_aligned {
                let (o, l_val, total_bytes, p_cnt) = paged_single_query_online_softmax(
                    pool,
                    q_vec,
                    kv_h,
                    &combined_pages,
                    total_len,
                    scale,
                );
                let pb = (2 * shared_prefix_len * d * 4) as u64;
                let sb = total_bytes.saturating_sub(pb);
                (o, l_val, pb, sb, p_cnt)
            } else {
                let (o0, l0, pb, p0) = paged_single_query_online_softmax(
                    pool,
                    q_vec,
                    kv_h,
                    shared_prefix_pages,
                    shared_prefix_len,
                    scale,
                );
                let (o1, l1, sb, p1) = paged_single_query_online_softmax(
                    pool,
                    q_vec,
                    kv_h,
                    &suffix_pages_per_req[b],
                    suffix_lens_per_req[b],
                    scale,
                );
                let (om, lm) = merge_lse_states(&o0, l0, &o1, l1);
                (om, lm, pb, sb, p0 + p1)
            };

            output[q_offset..q_offset + d].copy_from_slice(&out_vec);
            lse[b * num_q_heads + qh] = head_lse;
            stats.shared_prefix_kv_bytes_fp32 += pref_bytes;
            stats.suffix_kv_bytes_fp32 += suff_bytes;
            stats.total_kv_bytes_fp32 += pref_bytes + suff_bytes;
            stats.page_loads += pages;
        }
    }

    AttentionOutput { output, lse, stats }
}

/// Computes Level-0 shared-prefix attention by streaming each shared KV page **ONCE**
/// across all `B * gqa_group` queries that share each KV head.
///
/// Returns `(shared_output, shared_lse, kv_bytes_read, pages_loaded)`.
fn shared_prefix_batched_pass(
    pool: &PagedKvPool,
    queries: &[f32],
    batch_indices: &[usize],
    num_q_heads: usize,
    shared_pages: &[usize],
    shared_len: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>, u64, u64) {
    let d = pool.head_dim;
    let gqa_group = num_q_heads / pool.num_kv_heads;
    let group_batch = batch_indices.len();
    let total_rows = group_batch * num_q_heads;

    let mut out = vec![0.0f32; total_rows * d];
    let mut lse = vec![f32::NEG_INFINITY; total_rows];
    if shared_len == 0 || shared_pages.is_empty() || group_batch == 0 {
        return (out, lse, 0, 0);
    }

    let mut kv_bytes: u64 = 0;
    let mut pages_loaded: u64 = 0;

    // For each KV head, load every shared-prefix physical page ONCE and apply it to all
    // `group_batch * gqa_group` query vectors simultaneously (shared-KV tiling).
    let queries_per_kv = group_batch * gqa_group;
    let mut row_m = vec![f32::NEG_INFINITY; queries_per_kv];
    let mut row_l = vec![0.0f32; queries_per_kv];
    let mut row_acc = vec![0.0f32; queries_per_kv * d];
    let mut q_packed = vec![0.0f32; queries_per_kv * d];

    for kv_h in 0..pool.num_kv_heads {
        // Pack all queries belonging to this KV head contiguously for L1 cache locality.
        for (local_b, &orig_b) in batch_indices.iter().enumerate() {
            for g in 0..gqa_group {
                let qh = kv_h * gqa_group + g;
                let src_off = (orig_b * num_q_heads + qh) * d;
                let dst_row = local_b * gqa_group + g;
                q_packed[dst_row * d..(dst_row + 1) * d]
                    .copy_from_slice(&queries[src_off..src_off + d]);
            }
        }
        row_m.fill(f32::NEG_INFINITY);
        row_l.fill(0.0);
        row_acc.fill(0.0);

        let mut remaining = shared_len;
        let head_offset = kv_h * pool.page_size * d;
        let mut tile_scores = [0.0f32; 64];

        for &pid in shared_pages {
            if remaining == 0 {
                break;
            }
            let valid = remaining.min(pool.page_size);
            pages_loaded += 1;
            // Shared page is loaded ONCE for all `group_batch * gqa_group` queries!
            kv_bytes += (2 * valid * d * 4) as u64;

            let k_page = &pool.k_pages[pid];
            let v_page = &pool.v_pages[pid];

            // Genuine 4-way query register blocking: each shared K/V vector loaded from memory
            // is multiplied against 4 queries simultaneously (shared-KV GEMM tiling), turning
            // independent per-query KV streams into shared-KV compute tiles.
            let mut r = 0usize;
            while r + 4 <= queries_per_kv {
                let q0 = &q_packed[r * d..(r + 1) * d];
                let q1 = &q_packed[(r + 1) * d..(r + 2) * d];
                let q2 = &q_packed[(r + 2) * d..(r + 3) * d];
                let q3 = &q_packed[(r + 3) * d..(r + 4) * d];

                let mut s0 = [0.0f32; 64];
                let mut s1 = [0.0f32; 64];
                let mut s2 = [0.0f32; 64];
                let mut s3 = [0.0f32; 64];
                let mut tmax = [f32::NEG_INFINITY; 4];

                for t in 0..valid {
                    let base = head_offset + t * d;
                    let k_vec = &k_page[base..base + d];
                    let mut d0 = 0.0f32;
                    let mut d1 = 0.0f32;
                    let mut d2 = 0.0f32;
                    let mut d3 = 0.0f32;
                    for i in 0..d {
                        let kv = k_vec[i];
                        d0 += q0[i] * kv;
                        d1 += q1[i] * kv;
                        d2 += q2[i] * kv;
                        d3 += q3[i] * kv;
                    }
                    let v0 = d0 * scale;
                    let v1 = d1 * scale;
                    let v2 = d2 * scale;
                    let v3 = d3 * scale;
                    s0[t] = v0;
                    s1[t] = v1;
                    s2[t] = v2;
                    s3[t] = v3;
                    if v0 > tmax[0] {
                        tmax[0] = v0;
                    }
                    if v1 > tmax[1] {
                        tmax[1] = v1;
                    }
                    if v2 > tmax[2] {
                        tmax[2] = v2;
                    }
                    if v3 > tmax[3] {
                        tmax[3] = v3;
                    }
                }

                let mut new_m = [0.0f32; 4];
                for lane in 0..4 {
                    let old_m = row_m[r + lane];
                    let nm = old_m.max(tmax[lane]);
                    new_m[lane] = nm;
                    let alpha = if old_m.is_finite() {
                        (old_m - nm).exp()
                    } else {
                        0.0
                    };
                    row_l[r + lane] *= alpha;
                    let acc_lane = &mut row_acc[(r + lane) * d..(r + lane + 1) * d];
                    for val in acc_lane.iter_mut() {
                        *val *= alpha;
                    }
                    row_m[r + lane] = nm;
                }

                let (acc_01, acc_23) = row_acc[r * d..(r + 4) * d].split_at_mut(2 * d);
                let (acc0, acc1) = acc_01.split_at_mut(d);
                let (acc2, acc3) = acc_23.split_at_mut(d);
                let mut l_add = [0.0f32; 4];

                for t in 0..valid {
                    let p0 = (s0[t] - new_m[0]).exp();
                    let p1 = (s1[t] - new_m[1]).exp();
                    let p2 = (s2[t] - new_m[2]).exp();
                    let p3 = (s3[t] - new_m[3]).exp();
                    l_add[0] += p0;
                    l_add[1] += p1;
                    l_add[2] += p2;
                    l_add[3] += p3;
                    let base = head_offset + t * d;
                    let v_vec = &v_page[base..base + d];
                    for i in 0..d {
                        let vv = v_vec[i];
                        acc0[i] += p0 * vv;
                        acc1[i] += p1 * vv;
                        acc2[i] += p2 * vv;
                        acc3[i] += p3 * vv;
                    }
                }
                for lane in 0..4 {
                    row_l[r + lane] += l_add[lane];
                }
                r += 4;
            }

            while r < queries_per_kv {
                let q_vec = &q_packed[r * d..(r + 1) * d];
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

                let old_m = row_m[r];
                let new_m = old_m.max(tile_max);
                let alpha = if old_m.is_finite() {
                    (old_m - new_m).exp()
                } else {
                    0.0
                };
                row_l[r] *= alpha;
                let acc_slice = &mut row_acc[r * d..(r + 1) * d];
                for val in acc_slice.iter_mut() {
                    *val *= alpha;
                }

                let mut l_add = 0.0f32;
                for (t, &score) in tile_scores.iter().enumerate().take(valid) {
                    let p = (score - new_m).exp();
                    l_add += p;
                    let base = head_offset + t * d;
                    let v_vec = &v_page[base..base + d];
                    for i in 0..d {
                        acc_slice[i] += p * v_vec[i];
                    }
                }
                row_l[r] += l_add;
                row_m[r] = new_m;
                r += 1;
            }
            remaining -= valid;
        }

        // Normalize and scatter back into `[local_b, qh, d]`.
        for local_b in 0..group_batch {
            for g in 0..gqa_group {
                let r = local_b * gqa_group + g;
                let qh = kv_h * gqa_group + g;
                let dst_head_idx = local_b * num_q_heads + qh;
                let inv_l = 1.0 / row_l[r];
                let src_acc = &row_acc[r * d..(r + 1) * d];
                let dst_out = &mut out[dst_head_idx * d..(dst_head_idx + 1) * d];
                for i in 0..d {
                    dst_out[i] = src_acc[i] * inv_l;
                }
                lse[dst_head_idx] = row_m[r] + row_l[r].ln();
            }
        }
    }

    (out, lse, kv_bytes, pages_loaded)
}

/// Two-level shared-prefix cascade attention with exact LSE state merge.
///
/// 1. **Level 0 (Shared Prefix)**: Loads each shared-prefix KV page once across all `B` queries
///    and all `gqa_group` query heads sharing each KV head, producing `(O_shared, lse_shared)`.
/// 2. **Level 1 (Unique Suffix)**: Evaluates per-request paged attention over each request's
///    unique suffix pages, producing `(O_suffix, lse_suffix)`.
/// 3. **Exact LSE Merge**: Combines Level 0 and Level 1 via [`merge_lse_states`].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn two_level_cascade_gqa_decode(
    pool: &PagedKvPool,
    queries: &[f32],
    batch_size: usize,
    num_q_heads: usize,
    shared_prefix_pages: &[usize],
    shared_prefix_len: usize,
    suffix_pages_per_req: &[Vec<usize>],
    suffix_lens_per_req: &[usize],
) -> AttentionOutput {
    let d = pool.head_dim;
    let gqa_group = num_q_heads / pool.num_kv_heads;
    assert_eq!(num_q_heads % pool.num_kv_heads, 0);
    assert_eq!(queries.len(), batch_size * num_q_heads * d);
    assert_eq!(suffix_pages_per_req.len(), batch_size);
    assert_eq!(suffix_lens_per_req.len(), batch_size);

    let scale = 1.0 / (d as f32).sqrt();
    let batch_indices: Vec<usize> = (0..batch_size).collect();

    // Level 0: Shared prefix loaded ONCE across the batch.
    let (o_shared, lse_shared, shared_kv_bytes, shared_pages_loaded) = shared_prefix_batched_pass(
        pool,
        queries,
        &batch_indices,
        num_q_heads,
        shared_prefix_pages,
        shared_prefix_len,
        scale,
    );

    let mut output = vec![0.0f32; batch_size * num_q_heads * d];
    let mut lse = vec![f32::NEG_INFINITY; batch_size * num_q_heads];
    let mut suffix_kv_bytes: u64 = 0;
    let mut suffix_pages_loaded: u64 = 0;

    // Level 1 + LSE merge.
    for b in 0..batch_size {
        for qh in 0..num_q_heads {
            let kv_h = qh / gqa_group;
            let head_idx = b * num_q_heads + qh;
            let q_off = head_idx * d;
            let q_vec = &queries[q_off..q_off + d];

            let (o_suff, lse_suff, sb, sp) = paged_single_query_online_softmax(
                pool,
                q_vec,
                kv_h,
                &suffix_pages_per_req[b],
                suffix_lens_per_req[b],
                scale,
            );
            suffix_kv_bytes += sb;
            suffix_pages_loaded += sp;

            let (merged_o, merged_lse) = merge_lse_states(
                &o_shared[q_off..q_off + d],
                lse_shared[head_idx],
                &o_suff,
                lse_suff,
            );
            output[q_off..q_off + d].copy_from_slice(&merged_o);
            lse[head_idx] = merged_lse;
        }
    }

    // Account honestly for LSE merge write+read traffic: 2 partial states written & read
    // (`(O_0, lse_0)` and `(O_1, lse_1)`) + final `O` write per `(batch, q_head)`.
    let state_bytes_per_head = ((d + 1) * 4) as u64;
    let lse_merge_bytes_rw = (batch_size * num_q_heads) as u64 * state_bytes_per_head * 4;

    AttentionOutput {
        output,
        lse,
        stats: KvReadStats {
            shared_prefix_kv_bytes_fp32: shared_kv_bytes,
            suffix_kv_bytes_fp32: suffix_kv_bytes,
            total_kv_bytes_fp32: shared_kv_bytes + suffix_kv_bytes,
            lse_merge_bytes_rw,
            page_loads: shared_pages_loaded + suffix_pages_loaded,
        },
    }
}

/// Three-level prefix-tree cascade attention (Level 0 global system prefix, Level 1 sub-group
/// repository prefix, Level 2 unique per-request suffix) merged via [`merge_lse_multi_level`].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn three_level_cascade_gqa_decode(
    pool: &PagedKvPool,
    queries: &[f32],
    batch_size: usize,
    num_q_heads: usize,
    global_prefix_pages: &[usize],
    global_prefix_len: usize,
    subgroup_pages: &[Vec<usize>],
    subgroup_lens: &[usize],
    req_subgroup_id: &[usize],
    suffix_pages_per_req: &[Vec<usize>],
    suffix_lens_per_req: &[usize],
) -> AttentionOutput {
    let d = pool.head_dim;
    let gqa_group = num_q_heads / pool.num_kv_heads;
    assert_eq!(req_subgroup_id.len(), batch_size);
    assert_eq!(subgroup_pages.len(), subgroup_lens.len());

    let scale = 1.0 / (d as f32).sqrt();
    let all_indices: Vec<usize> = (0..batch_size).collect();

    // Level 0: global system prefix shared by all requests.
    let (o_l0, lse_l0, kv_l0, pages_l0) = shared_prefix_batched_pass(
        pool,
        queries,
        &all_indices,
        num_q_heads,
        global_prefix_pages,
        global_prefix_len,
        scale,
    );

    // Level 1: per-subgroup repository prefix shared by requests in each subgroup.
    let mut o_l1 = vec![0.0f32; batch_size * num_q_heads * d];
    let mut lse_l1 = vec![f32::NEG_INFINITY; batch_size * num_q_heads];
    let mut kv_l1: u64 = 0;
    let mut pages_l1: u64 = 0;

    for sg in 0..subgroup_pages.len() {
        let members: Vec<usize> = (0..batch_size)
            .filter(|&b| req_subgroup_id[b] == sg)
            .collect();
        if members.is_empty() {
            continue;
        }
        let (sg_o, sg_lse, sg_kv, sg_pages) = shared_prefix_batched_pass(
            pool,
            queries,
            &members,
            num_q_heads,
            &subgroup_pages[sg],
            subgroup_lens[sg],
            scale,
        );
        kv_l1 += sg_kv;
        pages_l1 += sg_pages;
        for (local_b, &orig_b) in members.iter().enumerate() {
            for qh in 0..num_q_heads {
                let src_idx = local_b * num_q_heads + qh;
                let dst_idx = orig_b * num_q_heads + qh;
                o_l1[dst_idx * d..(dst_idx + 1) * d]
                    .copy_from_slice(&sg_o[src_idx * d..(src_idx + 1) * d]);
                lse_l1[dst_idx] = sg_lse[src_idx];
            }
        }
    }

    // Level 2: unique suffix + 3-level LSE merge.
    let mut output = vec![0.0f32; batch_size * num_q_heads * d];
    let mut lse = vec![f32::NEG_INFINITY; batch_size * num_q_heads];
    let mut kv_l2: u64 = 0;
    let mut pages_l2: u64 = 0;

    for b in 0..batch_size {
        for qh in 0..num_q_heads {
            let kv_h = qh / gqa_group;
            let head_idx = b * num_q_heads + qh;
            let q_off = head_idx * d;
            let q_vec = &queries[q_off..q_off + d];

            let (o_l2, lse_l2, sb, sp) = paged_single_query_online_softmax(
                pool,
                q_vec,
                kv_h,
                &suffix_pages_per_req[b],
                suffix_lens_per_req[b],
                scale,
            );
            kv_l2 += sb;
            pages_l2 += sp;

            let levels = [
                (&o_l0[q_off..q_off + d], lse_l0[head_idx]),
                (&o_l1[q_off..q_off + d], lse_l1[head_idx]),
                (o_l2.as_slice(), lse_l2),
            ];
            let (merged_o, merged_lse) = merge_lse_multi_level(&levels);
            output[q_off..q_off + d].copy_from_slice(&merged_o);
            lse[head_idx] = merged_lse;
        }
    }

    let state_bytes_per_head = ((d + 1) * 4) as u64;
    let lse_merge_bytes_rw = (batch_size * num_q_heads) as u64 * state_bytes_per_head * 6;

    AttentionOutput {
        output,
        lse,
        stats: KvReadStats {
            shared_prefix_kv_bytes_fp32: kv_l0 + kv_l1,
            suffix_kv_bytes_fp32: kv_l2,
            total_kv_bytes_fp32: kv_l0 + kv_l1 + kv_l2,
            lse_merge_bytes_rw,
            page_loads: pages_l0 + pages_l1 + pages_l2,
        },
    }
}

/// Deterministic projection to miniature vocabulary logits for greedy argmax verification.
#[must_use]
pub fn greedy_argmax_tokens(
    attn_output: &[f32],
    batch_size: usize,
    hidden_dim: usize,
    vocab_size: usize,
    proj_weights: &[f32],
) -> Vec<usize> {
    assert_eq!(attn_output.len(), batch_size * hidden_dim);
    assert_eq!(proj_weights.len(), vocab_size * hidden_dim);
    let mut tokens = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let h = &attn_output[b * hidden_dim..(b + 1) * hidden_dim];
        let mut best_id = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for v in 0..vocab_size {
            let w = &proj_weights[v * hidden_dim..(v + 1) * hidden_dim];
            let mut dot = 0.0f32;
            for i in 0..hidden_dim {
                dot += h[i] * w[i];
            }
            if dot > best_val {
                best_val = dot;
                best_id = v;
            }
        }
        tokens.push(best_id);
    }
    tokens
}

/// Directly measured CPU equivalence & performance report for a single batch configuration.
#[derive(Debug, Clone)]
pub struct CpuCascadeBenchRow {
    /// Batch size `B`.
    pub batch_size: usize,
    /// Shared prefix token length.
    pub shared_prefix_len: usize,
    /// Per-request unique suffix token length.
    pub unique_suffix_len: usize,
    /// Maximum absolute FP32 output difference between unshared and two-level cascade.
    pub max_abs_output_err: f32,
    /// Maximum absolute FP32 LSE difference between unshared and two-level cascade.
    pub max_abs_lse_err: f32,
    /// Greedy argmax match rate in percent (`100.0` = lossless).
    pub greedy_argmax_match_pct: f64,
    /// Unshared total KV bytes read (FP32 storage).
    pub unshared_kv_bytes: u64,
    /// Cascade total KV bytes read (FP32 storage).
    pub cascade_kv_bytes: u64,
    /// Cascade LSE merge traffic bytes (write + read).
    pub cascade_lse_merge_bytes: u64,
    /// Exact KV-read reduction ratio (`unshared_kv_bytes / cascade_kv_bytes`).
    pub kv_read_reduction_ratio: f64,
    /// Directly measured unshared CPU wall time in microseconds (median over timed repeats).
    pub unshared_cpu_us: f64,
    /// Directly measured cascade CPU wall time in microseconds (median over timed repeats).
    pub cascade_cpu_us: f64,
    /// Directly measured CPU wall-clock speedup (`unshared_cpu_us / cascade_cpu_us`).
    pub cpu_wall_speedup: f64,
}

/// Directly measured summary across all CPU cascade verification cases (2-level + 3-level).
#[derive(Debug, Clone)]
pub struct CpuCascadeVerificationSuite {
    /// Per-configuration two-level cascade equivalence and CPU timing rows.
    pub rows: Vec<CpuCascadeBenchRow>,
    /// Maximum absolute FP32 error across all two-level rows.
    pub max_two_level_output_err: f32,
    /// Maximum absolute FP32 error on the three-level prefix-tree cascade check.
    pub three_level_max_output_err: f32,
    /// Three-level cascade KV read reduction ratio vs unshared baseline.
    pub three_level_kv_read_reduction: f64,
    /// Aggregate greedy argmax token match rate across all tested batches (`100.0` = exact).
    pub aggregate_argmax_match_pct: f64,
    /// Quality regression in percentage points (`0.0` = lossless).
    pub quality_regression_pp: f64,
}

/// Runs the complete directly measured CPU cascade equivalence & timing suite.
#[must_use]
pub fn run_cpu_cascade_verification(seed: u64) -> CpuCascadeVerificationSuite {
    let mut rng = Lcg64::new(seed);
    let num_q_heads = 8usize;
    let num_kv_heads = 2usize;
    let head_dim = 64usize;
    let page_size = 16usize;
    let hidden_dim = num_q_heads * head_dim;
    let vocab_size = 64usize;

    let proj_weights: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|_| rng.next_f32_signed() * 0.25)
        .collect();

    // Sweep B in {1, 4, 8, 16}, shared prefix 64..512, suffix 16..128.
    let configs = [
        (1usize, 64usize, 16usize, 8usize),
        (4usize, 128usize, 32usize, 8usize),
        (8usize, 256usize, 64usize, 8usize),
        (8usize, 512usize, 32usize, 10usize),
        (16usize, 512usize, 32usize, 10usize),
        (16usize, 256usize, 128usize, 8usize),
    ];

    let mut rows = Vec::with_capacity(configs.len());
    let mut max_two_level_err = 0.0f32;
    let mut total_tokens_checked = 0usize;
    let mut total_tokens_matched = 0usize;

    for (batch_size, prefix_len, suffix_len, repeats) in configs {
        let mut pool = PagedKvPool::new(num_kv_heads, page_size, head_dim);
        let shared_pages = pool.allocate_random_sequence(prefix_len, &mut rng);
        let mut suffix_pages = Vec::with_capacity(batch_size);
        let mut suffix_lens = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            suffix_pages.push(pool.allocate_random_sequence(suffix_len, &mut rng));
            suffix_lens.push(suffix_len);
        }
        let queries: Vec<f32> = (0..batch_size * num_q_heads * head_dim)
            .map(|_| rng.next_f32_signed() * 0.4)
            .collect();

        // Warmup & exact numerical comparison.
        let ref_out = unshared_paged_gqa_decode(
            &pool,
            &queries,
            batch_size,
            num_q_heads,
            &shared_pages,
            prefix_len,
            &suffix_pages,
            &suffix_lens,
        );
        let cas_out = two_level_cascade_gqa_decode(
            &pool,
            &queries,
            batch_size,
            num_q_heads,
            &shared_pages,
            prefix_len,
            &suffix_pages,
            &suffix_lens,
        );

        let mut max_out_err = 0.0f32;
        for (&a, &b) in ref_out.output.iter().zip(cas_out.output.iter()) {
            max_out_err = max_out_err.max((a - b).abs());
        }
        let mut max_lse_err = 0.0f32;
        for (&a, &b) in ref_out.lse.iter().zip(cas_out.lse.iter()) {
            max_lse_err = max_lse_err.max((a - b).abs());
        }
        max_two_level_err = max_two_level_err.max(max_out_err);

        let ref_toks = greedy_argmax_tokens(
            &ref_out.output,
            batch_size,
            hidden_dim,
            vocab_size,
            &proj_weights,
        );
        let cas_toks = greedy_argmax_tokens(
            &cas_out.output,
            batch_size,
            hidden_dim,
            vocab_size,
            &proj_weights,
        );
        let matched = ref_toks
            .iter()
            .zip(cas_toks.iter())
            .filter(|(a, b)| a == b)
            .count();
        total_tokens_checked += batch_size;
        total_tokens_matched += matched;
        let match_pct = (matched as f64) * 100.0 / (batch_size as f64);

        // Timed repeats (median) for honest directly measured CPU wall-clock timing.
        let mut unshared_samples = Vec::with_capacity(repeats);
        let mut cascade_samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let t0 = Instant::now();
            let o_u = unshared_paged_gqa_decode(
                &pool,
                &queries,
                batch_size,
                num_q_heads,
                &shared_pages,
                prefix_len,
                &suffix_pages,
                &suffix_lens,
            );
            let dt_u = t0.elapsed().as_secs_f64() * 1.0e6;
            std::hint::black_box(&o_u.output);
            unshared_samples.push(dt_u);

            let t1 = Instant::now();
            let o_c = two_level_cascade_gqa_decode(
                &pool,
                &queries,
                batch_size,
                num_q_heads,
                &shared_pages,
                prefix_len,
                &suffix_pages,
                &suffix_lens,
            );
            let dt_c = t1.elapsed().as_secs_f64() * 1.0e6;
            std::hint::black_box(&o_c.output);
            cascade_samples.push(dt_c);
        }
        unshared_samples.sort_by(|a, b| a.total_cmp(b));
        cascade_samples.sort_by(|a, b| a.total_cmp(b));
        let unshared_cpu_us = unshared_samples[repeats / 2];
        let cascade_cpu_us = cascade_samples[repeats / 2];

        let kv_ratio =
            (ref_out.stats.total_kv_bytes_fp32 as f64) / (cas_out.stats.total_kv_bytes_fp32 as f64);
        rows.push(CpuCascadeBenchRow {
            batch_size,
            shared_prefix_len: prefix_len,
            unique_suffix_len: suffix_len,
            max_abs_output_err: max_out_err,
            max_abs_lse_err: max_lse_err,
            greedy_argmax_match_pct: match_pct,
            unshared_kv_bytes: ref_out.stats.total_kv_bytes_fp32,
            cascade_kv_bytes: cas_out.stats.total_kv_bytes_fp32,
            cascade_lse_merge_bytes: cas_out.stats.lse_merge_bytes_rw,
            kv_read_reduction_ratio: kv_ratio,
            unshared_cpu_us,
            cascade_cpu_us,
            cpu_wall_speedup: unshared_cpu_us / cascade_cpu_us.max(1e-6),
        });
    }

    // Three-level prefix-tree cascade verification (B = 8: 2 subgroups of 4 requests).
    let b3 = 8usize;
    let l0_len = 128usize;
    let l1_len = 96usize;
    let l2_len = 32usize;
    let mut pool3 = PagedKvPool::new(num_kv_heads, page_size, head_dim);
    let global_pages = pool3.allocate_random_sequence(l0_len, &mut rng);
    let sg_pages = vec![
        pool3.allocate_random_sequence(l1_len, &mut rng),
        pool3.allocate_random_sequence(l1_len, &mut rng),
    ];
    let sg_lens = vec![l1_len, l1_len];
    let req_sg = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let mut suff3_pages = Vec::with_capacity(b3);
    let mut suff3_lens = Vec::with_capacity(b3);
    let mut unshared_combined_suffix_pages = Vec::with_capacity(b3);
    let mut unshared_combined_suffix_lens = Vec::with_capacity(b3);
    for &sg_id in &req_sg {
        let sp = pool3.allocate_random_sequence(l2_len, &mut rng);
        let mut comb = sg_pages[sg_id].clone();
        comb.extend_from_slice(&sp);
        suff3_pages.push(sp);
        suff3_lens.push(l2_len);
        unshared_combined_suffix_pages.push(comb);
        unshared_combined_suffix_lens.push(l1_len + l2_len);
    }
    let q3: Vec<f32> = (0..b3 * num_q_heads * head_dim)
        .map(|_| rng.next_f32_signed() * 0.4)
        .collect();

    let ref3 = unshared_paged_gqa_decode(
        &pool3,
        &q3,
        b3,
        num_q_heads,
        &global_pages,
        l0_len,
        &unshared_combined_suffix_pages,
        &unshared_combined_suffix_lens,
    );
    let cas3 = three_level_cascade_gqa_decode(
        &pool3,
        &q3,
        b3,
        num_q_heads,
        &global_pages,
        l0_len,
        &sg_pages,
        &sg_lens,
        &req_sg,
        &suff3_pages,
        &suff3_lens,
    );
    let mut three_level_max_err = 0.0f32;
    for (&a, &b) in ref3.output.iter().zip(cas3.output.iter()) {
        three_level_max_err = three_level_max_err.max((a - b).abs());
    }
    let three_level_kv_reduction =
        (ref3.stats.total_kv_bytes_fp32 as f64) / (cas3.stats.total_kv_bytes_fp32 as f64);

    let aggregate_argmax_match_pct =
        (total_tokens_matched as f64) * 100.0 / (total_tokens_checked.max(1) as f64);
    let quality_regression_pp = 100.0 - aggregate_argmax_match_pct;

    CpuCascadeVerificationSuite {
        rows,
        max_two_level_output_err: max_two_level_err,
        three_level_max_output_err: three_level_max_err,
        three_level_kv_read_reduction: three_level_kv_reduction,
        aggregate_argmax_match_pct,
        quality_regression_pp,
    }
}
