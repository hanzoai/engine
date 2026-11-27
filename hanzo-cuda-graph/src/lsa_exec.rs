// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust executable hybrid Linear-Attention (GDN/LSA) + Paged-Attention decode
//! and speculative verification engine.
//!
//! Provides a real, deterministic, zero-dependency forward engine preserving the 3:1
//! GDN:Attention hybrid layer ratio of the HIP-1346 `4B / 9B / 27B` matrix. Executes:
//! 1. Unbucketed eager decode (`step_eager`),
//! 2. Bucketed, masked, sanitizer-verified CUDA graph replay (`step_graph_captured`),
//! 3. Single-replay graph-compatible speculative verification with exact GDN state
//!    checkpoint/rollback on draft rejection (`verify_speculative_graph`), and
//! 4. A 6-case memory-lifetime & racecheck fault-injection suite (`run_sanitizer_fault_suite`).

use crate::bucket::{BucketPolicy, CudaDecodeGraphKey, ShapeBucket};
use crate::capture::{
    BreakableExecutionPlan, CaptureError, CapturedGraph, CapturedNode, DecodeGraphCache,
    GraphPieceKind,
};
use crate::lifetime::{
    BufferAccessKind, DeviceLocation, GraphMemoryPool, GraphSanitizer, PersistentGraphBuffers,
    SanitizerReport, SanitizerViolation,
};
use crate::model::QuantFormat;

/// Deterministic xorshift64* PRNG for weight initialization and reproducible workloads.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Creates a new deterministic PRNG from `seed` (zero maps to a non-zero constant).
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        let s = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self { state: s }
    }

    /// Generates the next `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Generates a uniform `f32` in `[-bound, bound]`.
    pub fn next_f32_range(&mut self, bound: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32);
        (u * 2.0 - 1.0) * bound
    }
}

/// Geometry configuration for the executable pure-Rust hybrid LSA/GDN engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridLsaConfig {
    /// Number of GDN/LSA recurrent layers (default `3`, matching 3:1 hybrid ratio).
    pub num_gdn_layers: usize,
    /// Number of paged full-attention layers (default `1`).
    pub num_attn_layers: usize,
    /// Hidden model dimension (`num_heads * head_dim`).
    pub d_model: usize,
    /// Number of heads for both GDN and attention layers.
    pub num_heads: usize,
    /// Per-head dimension (`d_model / num_heads`).
    pub head_dim: usize,
    /// Output vocabulary size.
    pub vocab_size: usize,
    /// Paged KV cache block size in tokens.
    pub block_size: usize,
}

impl Default for HybridLsaConfig {
    fn default() -> Self {
        Self {
            num_gdn_layers: 3,
            num_attn_layers: 1,
            d_model: 32,
            num_heads: 4,
            head_dim: 8,
            vocab_size: 64,
            block_size: 16,
        }
    }
}

/// Weights for a single Gated DeltaNet (GDN/LSA) recurrent layer.
#[derive(Debug, Clone)]
struct GdnLayerWeights {
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_alpha: Vec<f32>,
    w_beta: Vec<f32>,
    w_o: Vec<f32>,
    w_mlp_up: Vec<f32>,
    w_mlp_down: Vec<f32>,
}

/// Weights for a single Paged-Attention layer.
#[derive(Debug, Clone)]
struct AttnLayerWeights {
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    w_mlp_up: Vec<f32>,
    w_mlp_down: Vec<f32>,
}

/// Per-sequence mutable state: FP32 GDN recurrent matrices + paged KV blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceHybridState {
    /// GDN recurrent state matrices: `[num_gdn_layers][num_heads * head_dim * head_dim]`.
    pub gdn_states: Vec<Vec<f32>>,
    /// Paged KV cache keys per attention layer: `[num_attn_layers][block_idx][slot * d_model]`.
    pub kv_k_blocks: Vec<Vec<Vec<f32>>>,
    /// Paged KV cache values per attention layer: `[num_attn_layers][block_idx][slot * d_model]`.
    pub kv_v_blocks: Vec<Vec<Vec<f32>>>,
    /// Logical physical block table per sequence (`block_id` list).
    pub block_table: Vec<u32>,
    /// Number of committed tokens currently stored in this sequence's KV cache and GDN state.
    pub seq_len: usize,
}

impl SequenceHybridState {
    /// Allocates a clean zero-initialized sequence state for `cfg`.
    #[must_use]
    pub fn new(cfg: &HybridLsaConfig) -> Self {
        let gdn_elems = cfg.num_heads * cfg.head_dim * cfg.head_dim;
        Self {
            gdn_states: vec![vec![0.0f32; gdn_elems]; cfg.num_gdn_layers],
            kv_k_blocks: vec![Vec::new(); cfg.num_attn_layers],
            kv_v_blocks: vec![Vec::new(); cfg.num_attn_layers],
            block_table: Vec::new(),
            seq_len: 0,
        }
    }

    fn append_kv(&mut self, cfg: &HybridLsaConfig, layer_idx: usize, k: &[f32], v: &[f32]) {
        let block_idx = self.seq_len / cfg.block_size;
        let offset = self.seq_len % cfg.block_size;
        if block_idx >= self.kv_k_blocks[layer_idx].len() {
            self.kv_k_blocks[layer_idx].push(vec![0.0f32; cfg.block_size * cfg.d_model]);
            self.kv_v_blocks[layer_idx].push(vec![0.0f32; cfg.block_size * cfg.d_model]);
            if layer_idx == 0 {
                let next_blk_id = self.block_table.len() as u32;
                self.block_table.push(next_blk_id);
            }
        }
        let start = offset * cfg.d_model;
        self.kv_k_blocks[layer_idx][block_idx][start..start + cfg.d_model].copy_from_slice(k);
        self.kv_v_blocks[layer_idx][block_idx][start..start + cfg.d_model].copy_from_slice(v);
    }

    /// Computes a deterministic signature of the block table for copy-skip optimization.
    #[must_use]
    pub fn block_table_signature(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in &self.block_table {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h ^= self.block_table.len() as u64;
        h
    }
}

/// Multi-sequence batch state for [`HybridLsaEngine`].
#[derive(Debug, Clone, PartialEq)]
pub struct HybridLsaState {
    /// Per-sequence GDN + paged KV states (`len == max_slots`).
    pub sequences: Vec<SequenceHybridState>,
}

impl HybridLsaState {
    /// Creates a batch state with `num_sequences` independent sequence slots.
    #[must_use]
    pub fn new(cfg: &HybridLsaConfig, num_sequences: usize) -> Self {
        Self {
            sequences: (0..num_sequences)
                .map(|_| SequenceHybridState::new(cfg))
                .collect(),
        }
    }

    /// Maximum elementwise absolute difference across all GDN recurrent state matrices
    /// between `self` and `other`.
    #[must_use]
    pub fn max_abs_gdn_diff(&self, other: &Self) -> f32 {
        let mut max_err = 0.0f32;
        for (sa, sb) in self.sequences.iter().zip(other.sequences.iter()) {
            for (la, lb) in sa.gdn_states.iter().zip(sb.gdn_states.iter()) {
                for (&a, &b) in la.iter().zip(lb.iter()) {
                    max_err = max_err.max((a - b).abs());
                }
            }
        }
        max_err
    }

    /// Maximum elementwise absolute difference across all committed paged KV cache entries
    /// between `self` and `other`.
    #[must_use]
    pub fn max_abs_kv_diff(&self, other: &Self) -> f32 {
        let mut max_err = 0.0f32;
        for (sa, sb) in self.sequences.iter().zip(other.sequences.iter()) {
            if sa.seq_len != sb.seq_len {
                return f32::INFINITY;
            }
            for (la, lb) in sa.kv_k_blocks.iter().zip(sb.kv_k_blocks.iter()) {
                for (ba, bb) in la.iter().zip(lb.iter()) {
                    for (&a, &b) in ba.iter().zip(bb.iter()) {
                        max_err = max_err.max((a - b).abs());
                    }
                }
            }
            for (la, lb) in sa.kv_v_blocks.iter().zip(sb.kv_v_blocks.iter()) {
                for (ba, bb) in la.iter().zip(lb.iter()) {
                    for (&a, &b) in ba.iter().zip(bb.iter()) {
                        max_err = max_err.max((a - b).abs());
                    }
                }
            }
        }
        max_err
    }
}

/// Output of one graph-captured bucketed decode step.
#[derive(Debug, Clone)]
pub struct GraphDecodeOut {
    /// Active output logits sliced to `[active_batch][vocab_size]`.
    pub logits: Vec<Vec<f32>>,
    /// Execution plan used for this step.
    pub plan: BreakableExecutionPlan,
    /// Sanitizer verification report (present when graph replay ran).
    pub sanitizer_report: Option<SanitizerReport>,
    /// True if a cached graph was reused without re-capture.
    pub cache_hit: bool,
    /// True if `block_tables` in-place copy was skipped via matching signature.
    pub block_table_copy_skipped: bool,
}

/// Output of one graph-compatible speculative verification step.
#[derive(Debug, Clone)]
pub struct SpeculativeVerifyOut {
    /// Number of draft tokens accepted (`0..=draft_tokens.len()`).
    pub accepted_drafts: usize,
    /// All committed tokens produced in this step (`accepted_drafts` accepted drafts + 1 target token).
    pub committed_tokens: Vec<u32>,
    /// Target logits at the final committed position.
    pub final_logits: Vec<f32>,
    /// Shape bucket used for the single speculative graph replay.
    pub bucket: ShapeBucket,
    /// Sanitizer verification report from the replay.
    pub sanitizer_report: SanitizerReport,
    /// True if a cached speculative verification graph was hit.
    pub cache_hit: bool,
}

/// Deterministic pure-Rust hybrid GDN/LSA + Paged-Attention model engine.
#[derive(Debug, Clone)]
pub struct HybridLsaEngine {
    /// Engine geometry.
    pub cfg: HybridLsaConfig,
    embed: Vec<f32>,
    gdn_layers: Vec<GdnLayerWeights>,
    attn_layers: Vec<AttnLayerWeights>,
    lm_head: Vec<f32>,
}

impl HybridLsaEngine {
    /// Initializes deterministic weights from `seed`.
    #[must_use]
    pub fn new(cfg: HybridLsaConfig, seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed);
        let d = cfg.d_model;
        let scale = 1.0 / (d as f32).sqrt();

        let embed = (0..cfg.vocab_size * d)
            .map(|_| rng.next_f32_range(0.5))
            .collect();
        let lm_head = (0..cfg.vocab_size * d)
            .map(|_| rng.next_f32_range(scale))
            .collect();

        let gdn_layers = (0..cfg.num_gdn_layers)
            .map(|_| GdnLayerWeights {
                w_q: rand_mat(&mut rng, d * d, scale),
                w_k: rand_mat(&mut rng, d * d, scale),
                w_v: rand_mat(&mut rng, d * d, scale),
                w_alpha: rand_mat(&mut rng, cfg.num_heads * d, scale),
                w_beta: rand_mat(&mut rng, cfg.num_heads * d, scale),
                w_o: rand_mat(&mut rng, d * d, scale),
                w_mlp_up: rand_mat(&mut rng, d * d, scale),
                w_mlp_down: rand_mat(&mut rng, d * d, scale),
            })
            .collect();

        let attn_layers = (0..cfg.num_attn_layers)
            .map(|_| AttnLayerWeights {
                w_q: rand_mat(&mut rng, d * d, scale),
                w_k: rand_mat(&mut rng, d * d, scale),
                w_v: rand_mat(&mut rng, d * d, scale),
                w_o: rand_mat(&mut rng, d * d, scale),
                w_mlp_up: rand_mat(&mut rng, d * d, scale),
                w_mlp_down: rand_mat(&mut rng, d * d, scale),
            })
            .collect();

        Self {
            cfg,
            embed,
            gdn_layers,
            attn_layers,
            lm_head,
        }
    }

    /// Executes a single forward token on sequence `seq_state` in-place (mutating GDN state
    /// and paged KV cache) and returns output logits of length `vocab_size`.
    #[must_use]
    pub fn forward_single_token(
        &self,
        seq_state: &mut SequenceHybridState,
        token_id: u32,
    ) -> Vec<f32> {
        let d = self.cfg.d_model;
        let tok = (token_id as usize) % self.cfg.vocab_size;
        let mut h = self.embed[tok * d..(tok + 1) * d].to_vec();

        // 1. Linear-Attention (GDN/LSA) recurrent layers.
        for (layer_idx, layer) in self.gdn_layers.iter().enumerate() {
            let norm_h = rms_norm(&h);
            let q = matvec(&layer.w_q, &norm_h, d, d);
            let k = matvec(&layer.w_k, &norm_h, d, d);
            let v = matvec(&layer.w_v, &norm_h, d, d);
            let alpha_raw = matvec(&layer.w_alpha, &norm_h, self.cfg.num_heads, d);
            let beta_raw = matvec(&layer.w_beta, &norm_h, self.cfg.num_heads, d);

            let mut out = vec![0.0f32; d];
            let hd = self.cfg.head_dim;
            let state = &mut seq_state.gdn_states[layer_idx];

            for head in 0..self.cfg.num_heads {
                let h_off = head * hd;
                let s_off = head * hd * hd;

                // L2-normalize q and k per head for bounded recurrent dynamics.
                let q_h = l2_norm(&q[h_off..h_off + hd]);
                let k_h = l2_norm(&k[h_off..h_off + hd]);
                let v_h = &v[h_off..h_off + hd];

                // Data-dependent decay alpha in [0.85, 0.995] and update gate beta in [0.05, 0.45].
                let alpha = 0.85 + 0.145 * sigmoid(alpha_raw[head]);
                let beta = 0.05 + 0.40 * sigmoid(beta_raw[head]);

                // Prediction v_hat = S^T k_h
                let mut v_hat = vec![0.0f32; hd];
                for j in 0..hd {
                    let mut sum = 0.0f32;
                    for (i, &ki) in k_h.iter().enumerate() {
                        sum += state[s_off + i * hd + j] * ki;
                    }
                    v_hat[j] = sum;
                }

                // DeltaNet state update: S_t = alpha * S_{t-1} + beta * k_h outer (v_h - v_hat)
                for (i, &ki) in k_h.iter().enumerate() {
                    let row_off = s_off + i * hd;
                    for j in 0..hd {
                        let err = v_h[j] - v_hat[j];
                        state[row_off + j] = alpha * state[row_off + j] + beta * ki * err;
                    }
                }

                // Recurrent readout: o_h = S_t^T q_h
                for j in 0..hd {
                    let mut sum = 0.0f32;
                    for (i, &qi) in q_h.iter().enumerate() {
                        sum += state[s_off + i * hd + j] * qi;
                    }
                    out[h_off + j] = sum;
                }
            }

            let proj = matvec(&layer.w_o, &out, d, d);
            for i in 0..d {
                h[i] += proj[i];
            }

            // Gated MLP residual.
            let mlp_in = rms_norm(&h);
            let up = matvec(&layer.w_mlp_up, &mlp_in, d, d);
            let act: Vec<f32> = up.iter().map(|&x| x * sigmoid(x)).collect();
            let down = matvec(&layer.w_mlp_down, &act, d, d);
            for i in 0..d {
                h[i] += down[i];
            }
        }

        // 2. Paged full-attention layers.
        for (layer_idx, layer) in self.attn_layers.iter().enumerate() {
            let norm_h = rms_norm(&h);
            let q = matvec(&layer.w_q, &norm_h, d, d);
            let k = matvec(&layer.w_k, &norm_h, d, d);
            let v = matvec(&layer.w_v, &norm_h, d, d);

            seq_state.append_kv(&self.cfg, layer_idx, &k, &v);
            let ctx_len = seq_state.seq_len + 1;

            let hd = self.cfg.head_dim;
            let inv_sqrt_hd = 1.0 / (hd as f32).sqrt();
            let mut attn_out = vec![0.0f32; d];

            for head in 0..self.cfg.num_heads {
                let h_off = head * hd;
                let q_h = &q[h_off..h_off + hd];

                let mut scores = Vec::with_capacity(ctx_len);
                let mut max_s = f32::NEG_INFINITY;
                for pos in 0..ctx_len {
                    let blk = pos / self.cfg.block_size;
                    let slot = pos % self.cfg.block_size;
                    let k_base = slot * d + h_off;
                    let k_slice = &seq_state.kv_k_blocks[layer_idx][blk][k_base..k_base + hd];
                    let mut dot = 0.0f32;
                    for i in 0..hd {
                        dot += q_h[i] * k_slice[i];
                    }
                    let s = dot * inv_sqrt_hd;
                    if s > max_s {
                        max_s = s;
                    }
                    scores.push(s);
                }

                let mut exp_sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    exp_sum += *s;
                }
                let inv_sum = 1.0 / exp_sum.max(1e-12);

                for (pos, &w_raw) in scores.iter().enumerate() {
                    let w = w_raw * inv_sum;
                    let blk = pos / self.cfg.block_size;
                    let slot = pos % self.cfg.block_size;
                    let v_base = slot * d + h_off;
                    let v_slice = &seq_state.kv_v_blocks[layer_idx][blk][v_base..v_base + hd];
                    for i in 0..hd {
                        attn_out[h_off + i] += w * v_slice[i];
                    }
                }
            }

            let proj = matvec(&layer.w_o, &attn_out, d, d);
            for i in 0..d {
                h[i] += proj[i];
            }

            let mlp_in = rms_norm(&h);
            let up = matvec(&layer.w_mlp_up, &mlp_in, d, d);
            let act: Vec<f32> = up.iter().map(|&x| x * sigmoid(x)).collect();
            let down = matvec(&layer.w_mlp_down, &act, d, d);
            for i in 0..d {
                h[i] += down[i];
            }
        }

        seq_state.seq_len += 1;

        let final_h = rms_norm(&h);
        matvec(&self.lm_head, &final_h, self.cfg.vocab_size, d)
    }

    /// Unbucketed eager decode over `token_ids.len()` active sequences (`0..active_batch`).
    #[must_use]
    pub fn step_eager(&self, state: &mut HybridLsaState, token_ids: &[u32]) -> Vec<Vec<f32>> {
        token_ids
            .iter()
            .enumerate()
            .map(|(b, &tok)| self.forward_single_token(&mut state.sequences[b], tok))
            .collect()
    }

    /// Bucketed, masked, sanitizer-verified CUDA graph replay decode step.
    ///
    /// - Maps `(active_batch, K=1, max_context)` through `policy` into a [`ShapeBucket`].
    /// - On cache miss, allocates [`PersistentGraphBuffers`] and captures a [`CapturedGraph`]
    ///   (exercising capture-local scratch allocation and pre-`end_capture` free!).
    /// - Updates persistent buffers strictly in-place via `update_in_place` with
    ///   `block_table_signature` skip tracking.
    /// - Validates EVERY replay via [`GraphSanitizer`] before running the padded bucket kernel.
    /// - Masks padded lanes `active_batch..batch_bucket` so they NEVER mutate any sequence's
    ///   GDN state or KV cache, and slices back `[0..active_batch]` logits.
    pub fn step_graph_captured(
        &self,
        state: &mut HybridLsaState,
        pool: &mut GraphMemoryPool,
        cache: &mut DecodeGraphCache,
        policy: &BucketPolicy,
        token_ids: &[u32],
    ) -> Result<GraphDecodeOut, CaptureError> {
        let active_batch = token_ids.len();
        let active_context = state.sequences[..active_batch]
            .iter()
            .map(|s| s.seq_len + 1)
            .max()
            .unwrap_or(1);

        let plan = BreakableExecutionPlan::plan_step(
            policy,
            active_batch,
            1,
            active_context,
            false,
            false,
            false,
        );

        let Some(bucket) = plan.bucket else {
            // Breakable fallback to eager execution when batch/context exceeds max bucket.
            let logits = self.step_eager(state, token_ids);
            return Ok(GraphDecodeOut {
                logits,
                plan,
                sanitizer_report: None,
                cache_hit: false,
                block_table_copy_skipped: false,
            });
        };

        let key = CudaDecodeGraphKey::from_bucket(
            DeviceLocation::Cuda { gpu_id: 0 },
            &bucket,
            QuantFormat::Bf16,
            GraphPieceKind::LsaRecurrentAndMlp.tag(),
        );

        let mut cache_hit = true;
        if cache.get_mut(&key).is_none() {
            cache_hit = false;
            let buffers = PersistentGraphBuffers::allocate(
                pool,
                bucket.batch_bucket,
                bucket.verify_width_bucket,
                bucket.context_bucket,
                self.cfg.block_size,
                self.cfg.num_gdn_layers,
                self.cfg.vocab_size,
            )
            .map_err(|v| CaptureError::SanitizerFailed(vec![v]))?;

            let private_pool_bytes = 64 * 1024 + bucket.batch_bucket * self.cfg.d_model * 64;
            let graph = capture_hybrid_decode_graph(
                pool,
                key.clone(),
                &buffers,
                &self.cfg,
                private_pool_bytes,
                1.5,
            )?;
            cache.insert(pool, key.clone(), graph, buffers);
        }

        let entry = cache
            .get_mut(&key)
            .expect("entry guaranteed present after insert");
        // Note: adjust hit count so a fresh miss doesn't also register as a hit.
        // We keep exact semantics by reading `cache_hit` flag recorded above.
        let sig: Vec<u64> = state.sequences[..active_batch]
            .iter()
            .map(SequenceHybridState::block_table_signature)
            .collect();
        let max_blocks = bucket.context_bucket.div_ceil(self.cfg.block_size).max(1);
        let update = entry
            .buffers
            .update_in_place(
                pool,
                active_batch,
                1,
                active_batch * max_blocks,
                self.cfg.num_gdn_layers,
                Some(&sig),
            )
            .map_err(|v| CaptureError::SanitizerFailed(vec![v]))?;

        let sanitizer_report = entry.graph.validate_and_replay(pool)?;

        // Execute padded bucket `[0..batch_bucket]` with strict active-lane mask!
        let mask = bucket.lane_mask();
        let mut padded_logits = vec![vec![0.0f32; self.cfg.vocab_size]; bucket.batch_bucket];
        for lane in 0..bucket.batch_bucket {
            if mask[lane] {
                padded_logits[lane] =
                    self.forward_single_token(&mut state.sequences[lane], token_ids[lane]);
            } else {
                // Padded lane executes on an isolated scratch copy and NEVER touches state.sequences!
                let mut dummy = SequenceHybridState::new(&self.cfg);
                padded_logits[lane] = self.forward_single_token(&mut dummy, 0);
            }
        }

        padded_logits.truncate(active_batch);

        Ok(GraphDecodeOut {
            logits: padded_logits,
            plan,
            sanitizer_report: Some(sanitizer_report),
            cache_hit,
            block_table_copy_skipped: update.block_table_skipped,
        })
    }

    /// Executes graph-compatible speculative verification (`K = 1 + draft_tokens.len()`,
    /// padded to a bounded verify bucket `verify_width_bucket ∈ {1, 2, 4, 8}`) on sequence `seq_idx`.
    ///
    /// - Captures or reuses a verification graph keyed on `(B=1, K_bucket, ctx_bucket)`.
    /// - Validates all buffer lifetimes via [`GraphSanitizer`] before replay.
    /// - Evaluates all `K` positions in a single bucketed replay while saving lightweight per-step
    ///   GDN state checkpoints.
    /// - At the breakable graph boundary (`SpeculativeRejectionSampling`), accepts the longest
    ///   matching draft prefix `a <= draft_tokens.len()`, emits the target bonus/correction token
    ///   at position `a`, and rolls back GDN recurrent state and paged KV cache to the **exact**
    ///   post-step-`a` state (zero drift vs eager autoregressive decoding!).
    #[allow(clippy::too_many_arguments)]
    pub fn verify_speculative_graph(
        &self,
        state: &mut HybridLsaState,
        seq_idx: usize,
        pool: &mut GraphMemoryPool,
        cache: &mut DecodeGraphCache,
        policy: &BucketPolicy,
        anchor_token: u32,
        draft_tokens: &[u32],
    ) -> Result<SpeculativeVerifyOut, CaptureError> {
        let verify_width = 1 + draft_tokens.len();
        let active_context = state.sequences[seq_idx].seq_len + verify_width;
        let plan = BreakableExecutionPlan::plan_step(
            policy,
            1,
            verify_width,
            active_context,
            false,
            true,
            false,
        );
        let bucket = plan.bucket.ok_or_else(|| {
            CaptureError::SanitizerFailed(vec![SanitizerViolation::OutOfBounds {
                node_id: 0,
                buffer_id: crate::lifetime::BufferId(0),
                access_end: verify_width,
                capacity_bytes: 8,
                label: "spec_verify_width_exceeds_buckets".to_string(),
            }])
        })?;

        let key = CudaDecodeGraphKey::from_bucket(
            DeviceLocation::Cuda { gpu_id: 0 },
            &bucket,
            QuantFormat::Bf16,
            GraphPieceKind::PagedAttentionAndVerify.tag(),
        );

        let mut cache_hit = true;
        if cache.get_mut(&key).is_none() {
            cache_hit = false;
            let buffers = PersistentGraphBuffers::allocate(
                pool,
                bucket.batch_bucket,
                bucket.verify_width_bucket,
                bucket.context_bucket,
                self.cfg.block_size,
                self.cfg.num_gdn_layers,
                self.cfg.vocab_size,
            )
            .map_err(|v| CaptureError::SanitizerFailed(vec![v]))?;

            let private_pool_bytes = 64 * 1024
                + bucket.batch_bucket * bucket.verify_width_bucket * self.cfg.d_model * 64;
            let graph = capture_hybrid_decode_graph(
                pool,
                key.clone(),
                &buffers,
                &self.cfg,
                private_pool_bytes,
                1.8,
            )?;
            cache.insert(pool, key.clone(), graph, buffers);
        }

        let entry = cache
            .get_mut(&key)
            .expect("speculative graph entry guaranteed present");
        let sig = [state.sequences[seq_idx].block_table_signature()];
        let max_blocks = bucket.context_bucket.div_ceil(self.cfg.block_size).max(1);
        entry
            .buffers
            .update_in_place(
                pool,
                1,
                verify_width,
                max_blocks,
                self.cfg.num_gdn_layers,
                Some(&sig),
            )
            .map_err(|v| CaptureError::SanitizerFailed(vec![v]))?;

        let sanitizer_report = entry.graph.validate_and_replay(pool)?;

        // Snapshot state before speculative replay so rejected steps roll back bit-exactly.
        let mut working = state.sequences[seq_idx].clone();
        let mut step_logits = Vec::with_capacity(verify_width);
        let mut step_states = Vec::with_capacity(verify_width);

        let mut verify_inputs = Vec::with_capacity(verify_width);
        verify_inputs.push(anchor_token);
        verify_inputs.extend_from_slice(draft_tokens);

        // Execute all verify_width positions (plus masked padding slots up to verify_width_bucket).
        for step in 0..bucket.verify_width_bucket {
            if let Some(&tok) = verify_inputs.get(step) {
                let logits = self.forward_single_token(&mut working, tok);
                step_logits.push(logits);
                step_states.push(working.clone());
            } else {
                // Masked padding verification slot: executes on isolated scratch, zero state effect.
                let mut dummy = SequenceHybridState::new(&self.cfg);
                let _ = self.forward_single_token(&mut dummy, 0);
            }
        }

        // Breakable graph boundary: exact greedy verification + GDN/KV rollback.
        let mut accepted_drafts = 0usize;
        let mut committed_tokens = Vec::with_capacity(verify_width);
        for (idx, &draft_tok) in draft_tokens.iter().enumerate() {
            let target_tok = argmax(&step_logits[idx]);
            if target_tok == draft_tok {
                accepted_drafts += 1;
                committed_tokens.push(draft_tok);
            } else {
                // Rejection at `idx`: emit target correction token and roll back to step `idx`.
                committed_tokens.push(target_tok);
                state.sequences[seq_idx] = step_states[idx].clone();
                return Ok(SpeculativeVerifyOut {
                    accepted_drafts,
                    committed_tokens,
                    final_logits: step_logits[idx].clone(),
                    bucket,
                    sanitizer_report,
                    cache_hit,
                });
            }
        }

        // All `draft_tokens` were accepted! Emit the bonus target token from the final position.
        let bonus_tok = argmax(&step_logits[draft_tokens.len()]);
        committed_tokens.push(bonus_tok);
        state.sequences[seq_idx] = step_states[draft_tokens.len()].clone();

        Ok(SpeculativeVerifyOut {
            accepted_drafts,
            committed_tokens,
            final_logits: step_logits[draft_tokens.len()].clone(),
            bucket,
            sanitizer_report,
            cache_hit,
        })
    }
}

/// Captures the two piecewise nodes (`LsaRecurrentAndMlp` + `PagedAttentionAndVerify`) into
/// a [`CapturedGraph`], allocating and cleanly freeing a capture-local scratch buffer before
/// `end_capture` (matching Hanzo Engine's `capture_cuda_graph_with_output` invariant).
fn capture_hybrid_decode_graph(
    pool: &mut GraphMemoryPool,
    key: CudaDecodeGraphKey,
    buffers: &PersistentGraphBuffers,
    cfg: &HybridLsaConfig,
    private_pool_bytes: usize,
    capture_duration_ms: f64,
) -> Result<CapturedGraph, CaptureError> {
    CapturedGraph::capture(pool, key, private_pool_bytes, capture_duration_ms, |p| {
        // Allocate temporary capture-local scratch (like intermediate hidden states),
        // then explicitly copy outputs into persistent `output_logits` and free scratch
        // BEFORE `end_capture` so zero unfreed graph allocations remain!
        let scratch = p.allocate_capture_scratch("capture_hidden_scratch", cfg.d_model * 4)?;

        let r_input = p.make_ref(
            buffers.input_ids,
            0,
            p.get(buffers.input_ids).map_or(0, |b| b.byte_len),
            BufferAccessKind::Read,
        )?;
        let r_pos = p.make_ref(
            buffers.position_ids,
            0,
            p.get(buffers.position_ids).map_or(0, |b| b.byte_len),
            BufferAccessKind::Read,
        )?;
        let r_gdn = p.make_ref(
            buffers.gdn_recurrent_state_ptrs,
            0,
            p.get(buffers.gdn_recurrent_state_ptrs)
                .map_or(0, |b| b.byte_len),
            BufferAccessKind::ReadWrite,
        )?;
        let r_mask = p.make_ref(
            buffers.lane_active_mask,
            0,
            p.get(buffers.lane_active_mask).map_or(0, |b| b.byte_len),
            BufferAccessKind::Read,
        )?;
        let r_slot = p.make_ref(
            buffers.slot_mapping,
            0,
            p.get(buffers.slot_mapping).map_or(0, |b| b.byte_len),
            BufferAccessKind::Read,
        )?;
        let r_bt = p.make_ref(
            buffers.block_tables,
            0,
            p.get(buffers.block_tables).map_or(0, |b| b.byte_len),
            BufferAccessKind::Read,
        )?;
        let r_ctx = p.make_ref(
            buffers.context_lens,
            0,
            p.get(buffers.context_lens).map_or(0, |b| b.byte_len),
            BufferAccessKind::Read,
        )?;
        let r_draft = p.make_ref(
            buffers.spec_draft_tokens,
            0,
            p.get(buffers.spec_draft_tokens).map_or(0, |b| b.byte_len),
            BufferAccessKind::Read,
        )?;
        let r_out = p.make_ref(
            buffers.output_logits,
            0,
            p.get(buffers.output_logits).map_or(0, |b| b.byte_len),
            BufferAccessKind::Write,
        )?;

        // Free capture-local scratch BEFORE `end_capture` (no dangling scratch reference in nodes).
        p.free_buffer(scratch);

        Ok(vec![
            CapturedNode {
                node_id: 0,
                kernel_name: "lsa_gdn_recurrent_and_mlp".to_string(),
                piece_kind: GraphPieceKind::LsaRecurrentAndMlp,
                bindings: vec![r_input, r_pos, r_gdn, r_mask.clone()],
            },
            CapturedNode {
                node_id: 1,
                kernel_name: "paged_attn_and_spec_verify".to_string(),
                piece_kind: GraphPieceKind::PagedAttentionAndVerify,
                bindings: vec![r_slot, r_bt, r_ctx, r_draft, r_mask, r_out],
            },
        ])
    })
}

/// Summary of the 6-case intentional sanitizer fault-injection validation suite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SanitizerFaultSummary {
    /// Total intentional lifetime/corruption faults injected (`6`).
    pub total_faults_injected: usize,
    /// Number of injected faults caught fail-closed by [`GraphSanitizer`] (`6`).
    pub total_faults_caught: usize,
    /// Per-case `(name, caught, diagnostic)` details.
    pub cases: Vec<(String, bool, String)>,
}

/// Runs 6 deliberate CUDA graph memory-lifetime & racecheck fault injections, proving that
/// [`GraphSanitizer`] catches 100% (`6/6`) of them fail-closed:
/// 1. `use_after_free_output_logits` (`PoisonedBufferAccess` / `UseAfterFree`)
/// 2. `generation_mismatch_reallocated_metadata` (`GenerationMismatch`)
/// 3. `out_of_bounds_unbucketed_growth` (`OutOfBounds`)
/// 4. `dynamic_allocation_inside_capture` (`DynamicAllocationForbidden`)
/// 5. `host_sync_inside_capture` (`HostSyncForbidden`)
/// 6. `gdn_state_kv_alias_race` (`StateAliasRace`)
#[must_use]
pub fn run_sanitizer_fault_suite() -> SanitizerFaultSummary {
    let cfg = HybridLsaConfig::default();
    let mut cases = Vec::with_capacity(6);

    // Fault 1: Free persistent output_logits while live graph still references it -> PoisonedBufferAccess.
    {
        let mut pool = GraphMemoryPool::new();
        let buffers =
            PersistentGraphBuffers::allocate(&mut pool, 1, 1, 8_192, 16, cfg.num_gdn_layers, 64)
                .expect("alloc");
        let key = CudaDecodeGraphKey {
            device: DeviceLocation::Cuda { gpu_id: 0 },
            batch_bucket: 1,
            verify_width_bucket: 1,
            max_context_len: 8_192,
            quant: QuantFormat::Bf16,
            piece_tag: "fault_uaf",
        };
        let mut graph = capture_hybrid_decode_graph(&mut pool, key, &buffers, &cfg, 4096, 1.0)
            .expect("capture");
        pool.free_buffer(buffers.output_logits);
        let res = graph.validate_and_replay(&pool);
        let caught = matches!(
            &res,
            Err(CaptureError::SanitizerFailed(vs))
                if vs.iter().any(|v| matches!(v, SanitizerViolation::PoisonedBufferAccess { .. } | SanitizerViolation::UseAfterFree { .. }))
        );
        let diag = match res {
            Err(e) => e.to_string(),
            Ok(_) => "MISSED".to_string(),
        };
        graph.teardown(&mut pool);
        cases.push(("use_after_free_output_logits".to_string(), caught, diag));
    }

    // Fault 2: Reallocate `slot_mapping` (bumping generation) without re-capturing -> GenerationMismatch.
    {
        let mut pool = GraphMemoryPool::new();
        let buffers =
            PersistentGraphBuffers::allocate(&mut pool, 1, 1, 8_192, 16, cfg.num_gdn_layers, 64)
                .expect("alloc");
        let key = CudaDecodeGraphKey {
            device: DeviceLocation::Cuda { gpu_id: 0 },
            batch_bucket: 1,
            verify_width_bucket: 1,
            max_context_len: 8_192,
            quant: QuantFormat::Bf16,
            piece_tag: "fault_gen",
        };
        let mut graph = capture_hybrid_decode_graph(&mut pool, key, &buffers, &cfg, 4096, 1.0)
            .expect("capture");
        let _ = pool.reallocate_buffer(buffers.slot_mapping, 64);
        let res = graph.validate_and_replay(&pool);
        let caught = matches!(
            &res,
            Err(CaptureError::SanitizerFailed(vs))
                if vs.iter().any(|v| matches!(v, SanitizerViolation::GenerationMismatch { .. }))
        );
        let diag = match res {
            Err(e) => e.to_string(),
            Ok(_) => "MISSED".to_string(),
        };
        graph.teardown(&mut pool);
        cases.push((
            "generation_mismatch_reallocated_metadata".to_string(),
            caught,
            diag,
        ));
    }

    // Fault 3: Unbucketed sequence growth attempting to copy more bytes than persistent block_tables capacity -> OutOfBounds.
    {
        let mut pool = GraphMemoryPool::new();
        let mut buffers =
            PersistentGraphBuffers::allocate(&mut pool, 1, 1, 8_192, 16, cfg.num_gdn_layers, 64)
                .expect("alloc");
        // 8192/16 = 512 words capacity; try writing 1024 words without re-bucketing to 16K!
        let res = buffers.update_in_place(&mut pool, 1, 1, 1024, cfg.num_gdn_layers, None);
        let caught = matches!(&res, Err(SanitizerViolation::OutOfBounds { .. }));
        let diag = match res {
            Err(v) => v.to_string(),
            Ok(_) => "MISSED".to_string(),
        };
        cases.push(("out_of_bounds_unbucketed_growth".to_string(), caught, diag));
    }

    // Fault 4: Unpooled dynamic allocation attempted mid-capture -> DynamicAllocationForbidden.
    {
        let mut pool = GraphMemoryPool::new();
        let key = CudaDecodeGraphKey {
            device: DeviceLocation::Cuda { gpu_id: 0 },
            batch_bucket: 1,
            verify_width_bucket: 1,
            max_context_len: 8_192,
            quant: QuantFormat::Bf16,
            piece_tag: "fault_dyn_alloc",
        };
        let res = CapturedGraph::capture(&mut pool, key, 4096, 1.0, |p| {
            let _ = p.allocate_unpooled_dynamic("transient_kv_grow", 65_536)?;
            Ok(Vec::new())
        });
        let caught = matches!(&res, Err(CaptureError::DynamicAllocationForbidden(_)));
        let diag = match res {
            Err(e) => e.to_string(),
            Ok(_) => "MISSED".to_string(),
        };
        cases.push((
            "dynamic_allocation_inside_capture".to_string(),
            caught,
            diag,
        ));
    }

    // Fault 5: Host synchronization (`cudaStreamSynchronize`) attempted mid-capture -> HostSyncForbidden.
    {
        let mut pool = GraphMemoryPool::new();
        let key = CudaDecodeGraphKey {
            device: DeviceLocation::Cuda { gpu_id: 0 },
            batch_bucket: 1,
            verify_width_bucket: 1,
            max_context_len: 8_192,
            quant: QuantFormat::Bf16,
            piece_tag: "fault_host_sync",
        };
        let res = CapturedGraph::capture(&mut pool, key, 4096, 1.0, |p| {
            p.check_host_sync("cudaStreamSynchronize during speculative rejection check")?;
            Ok(Vec::new())
        });
        let caught = matches!(&res, Err(CaptureError::HostSyncForbidden(_)));
        let diag = match res {
            Err(e) => e.to_string(),
            Ok(_) => "MISSED".to_string(),
        };
        cases.push(("host_sync_inside_capture".to_string(), caught, diag));
    }

    // Fault 6: GDN recurrent state write-back aliases attention KV scratch virtual range -> StateAliasRace.
    {
        let mut pool = GraphMemoryPool::new();
        let gdn_buf = pool
            .allocate_persistent("gdn_state_buf", 4096, true)
            .expect("gdn alloc");
        let kv_buf = pool
            .allocate_persistent("attn_kv_scratch", 4096, true)
            .expect("kv alloc");
        let shared_vaddr = pool.get(gdn_buf).expect("gdn block").virtual_addr;

        let mut ref_gdn = pool
            .make_ref(gdn_buf, 0, 4096, BufferAccessKind::ReadWrite)
            .expect("ref gdn");
        let mut ref_kv = pool
            .make_ref(kv_buf, 0, 2048, BufferAccessKind::Write)
            .expect("ref kv");
        // Deliberately alias KV scratch onto the first 2048 bytes of GDN recurrent state!
        ref_gdn.virtual_addr_override = Some(shared_vaddr);
        ref_kv.virtual_addr_override = Some(shared_vaddr);

        let bindings = [ref_gdn, ref_kv];
        let res = GraphSanitizer::validate_replay(&pool, &[(0, &bindings)]);
        let caught = matches!(
            &res,
            Err(vs) if vs.iter().any(|v| matches!(v, SanitizerViolation::StateAliasRace { overlap_bytes: 2048, .. }))
        );
        let diag = match res {
            Err(vs) => vs
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; "),
            Ok(_) => "MISSED".to_string(),
        };
        cases.push(("gdn_state_kv_alias_race".to_string(), caught, diag));
    }

    let total_faults_caught = cases.iter().filter(|(_, c, _)| *c).count();
    SanitizerFaultSummary {
        total_faults_injected: cases.len(),
        total_faults_caught,
        cases,
    }
}

/// Returns the greedy argmax token index of `logits`.
#[must_use]
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx as u32
}

/// Computes maximum absolute elementwise difference between two logit batches.
#[must_use]
pub fn max_abs_logits_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    let mut max_err = 0.0f32;
    for (ra, rb) in a.iter().zip(b.iter()) {
        if ra.len() != rb.len() {
            return f32::INFINITY;
        }
        for (&va, &vb) in ra.iter().zip(rb.iter()) {
            max_err = max_err.max((va - vb).abs());
        }
    }
    max_err
}

fn rand_mat(rng: &mut DeterministicRng, len: usize, bound: f32) -> Vec<f32> {
    (0..len).map(|_| rng.next_f32_range(bound)).collect()
}

fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        out[r] = acc;
    }
    out
}

fn rms_norm(x: &[f32]) -> Vec<f32> {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len().max(1) as f32);
    let inv = 1.0 / (mean_sq + 1e-6).sqrt();
    x.iter().map(|&v| v * inv).collect()
}

fn l2_norm(x: &[f32]) -> Vec<f32> {
    let sq = x.iter().map(|&v| v * v).sum::<f32>();
    let inv = 1.0 / (sq + 1e-6).sqrt();
    x.iter().map(|&v| v * inv).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
