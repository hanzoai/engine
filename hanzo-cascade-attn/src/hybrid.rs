// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust hybrid Attention + Gated DeltaNet (GDN/LSA) + Dense SwiGLU MLP step verifier.
//!
//! ## Purpose (Amdahl's-Law Verification)
//! In the shared HIP-1346 `4B / 9B / 27B` hybrid models, only `25%` of decoder layers
//! use full paged attention (`9/36`, `10/40`, `16/64`), while `75%` use GDN recurrent
//! linear-attention layers (`S_t = decay * S_{t-1} + beta * k_t (x) (v_t - S_{t-1}^T k_t)`)
//! and `100%` of layers execute a dense SwiGLU MLP (`down(silu(gate(x)) * up(x))`).
//!
//! This module executes all three components directly on the CPU in the exact `1 : 3`
//! Attention : GDN layer ratio so we can:
//! 1. Directly measure the real CPU Amdahl fraction spent in Attention vs GDN/LSA vs Dense MLP.
//! 2. Verify end-to-end full-layer output equivalence (`max_abs_err <= 1e-5`, `100%` argmax match)
//!    when swapping unshared paged attention for two-level shared-prefix cascade attention.
//! 3. Contrast isolated attention speedup against actual full-step end-to-end speedup.

use std::time::Instant;

use crate::cascade::{
    greedy_argmax_tokens, two_level_cascade_gqa_decode, unshared_paged_gqa_decode, Lcg64,
    PagedKvPool,
};

/// Scaled hybrid decoder step geometry maintaining the exact `1 : 3` Attention : GDN layer ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridStepConfig {
    /// Batch size `B`.
    pub batch_size: usize,
    /// Number of full attention layers (`2` -> paired with `6` GDN layers = `1:3` ratio).
    pub num_attn_layers: usize,
    /// Number of GDN/LSA recurrent linear-attention layers (`6`).
    pub num_gdn_layers: usize,
    /// Hidden dimension (`num_q_heads * head_dim`).
    pub d_model: usize,
    /// Dense SwiGLU intermediate dimension.
    pub d_ff: usize,
    /// Number of query heads.
    pub num_q_heads: usize,
    /// Number of KV heads (GQA).
    pub num_kv_heads: usize,
    /// Attention head dimension.
    pub head_dim: usize,
    /// Number of GDN recurrent heads.
    pub gdn_heads: usize,
    /// GDN recurrent head dimension (`state = gdn_head_dim * gdn_head_dim` FP32 per head).
    pub gdn_head_dim: usize,
    /// Shared prefix length in attention KV cache.
    pub shared_prefix_len: usize,
    /// Per-request unique suffix length in attention KV cache.
    pub unique_suffix_len: usize,
    /// Physical KV page size.
    pub page_size: usize,
}

impl Default for HybridStepConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            num_attn_layers: 2,
            num_gdn_layers: 6,
            d_model: 128,
            d_ff: 320,
            num_q_heads: 4,
            num_kv_heads: 1,
            head_dim: 32,
            gdn_heads: 4,
            gdn_head_dim: 32,
            shared_prefix_len: 256,
            unique_suffix_len: 32,
            page_size: 16,
        }
    }
}

/// Deterministic layer weights for the pure-Rust hybrid step verifier.
#[derive(Debug, Clone)]
struct LayerMlpWeights {
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}

/// Deterministic GDN layer weights (`Q, K, V, O` projections + scalar decay/beta gates).
#[derive(Debug, Clone)]
struct GdnLayerWeights {
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    decay: f32,
    beta: f32,
}

/// Deterministic attention layer projections (`W_q`, `W_o`).
#[derive(Debug, Clone)]
struct AttnLayerWeights {
    w_q: Vec<f32>,
    w_o: Vec<f32>,
}

/// Directly measured CPU breakdown of a full hybrid decoder step (Attention + GDN/LSA + Dense MLP).
#[derive(Debug, Clone)]
pub struct CpuHybridStepReport {
    /// Batch size executed.
    pub batch_size: usize,
    /// Attention layers executed.
    pub num_attn_layers: usize,
    /// GDN/LSA layers executed.
    pub num_gdn_layers: usize,
    /// Directly measured unshared attention component time (us).
    pub unshared_attn_us: f64,
    /// Directly measured cascade attention component time (us).
    pub cascade_attn_us: f64,
    /// Directly measured GDN/LSA recurrent component time (us).
    pub gdn_time_us: f64,
    /// Directly measured dense SwiGLU MLP component time (us).
    pub mlp_time_us: f64,
    /// Total full-step time using unshared paged attention (us).
    pub unshared_total_step_us: f64,
    /// Total full-step time using two-level cascade attention (us).
    pub cascade_total_step_us: f64,
    /// Fraction of baseline full-step time spent in Attention (`Amdahl p`).
    pub attn_time_fraction: f64,
    /// Fraction of baseline full-step time spent in GDN/LSA recurrent layers.
    pub gdn_time_fraction: f64,
    /// Fraction of baseline full-step time spent in Dense SwiGLU MLP layers.
    pub mlp_time_fraction: f64,
    /// Isolated attention component speedup (`unshared_attn_us / cascade_attn_us`).
    pub isolated_attn_speedup: f64,
    /// End-to-end full hybrid step speedup (`unshared_total_step_us / cascade_total_step_us`).
    pub full_step_speedup: f64,
    /// Maximum absolute FP32 hidden-state error across the full 8-layer hybrid step.
    pub max_abs_hidden_err: f32,
    /// Maximum absolute FP32 GDN recurrent state error (must be `0.0` or `<= 1e-5`).
    pub max_abs_gdn_state_err: f32,
    /// Greedy argmax match rate on full hybrid step output (`100.0` = exact).
    pub greedy_argmax_match_pct: f64,
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        out[r] = acc;
    }
}

fn rms_norm_inplace(x: &mut [f32]) {
    let n = x.len() as f32;
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n + 1e-5).sqrt();
    for v in x.iter_mut() {
        *v *= inv_rms;
    }
}

/// Executes one GDN/LSA recurrent layer across a batch `B`, updating `gdn_state` in place:
/// `v_hat = S^T k`, `S_new = decay * S + beta * k (x) (v - v_hat)`, `y = S_new^T q`.
fn run_gdn_layer(
    hidden: &mut [f32],
    gdn_state: &mut [f32],
    weights: &GdnLayerWeights,
    cfg: &HybridStepConfig,
) {
    let d = cfg.d_model;
    let hd = cfg.gdn_head_dim;
    let state_per_head = hd * hd;
    let state_per_seq = cfg.gdn_heads * state_per_head;

    let mut q_buf = vec![0.0f32; d];
    let mut k_buf = vec![0.0f32; d];
    let mut v_buf = vec![0.0f32; d];
    let mut y_buf = vec![0.0f32; d];
    let mut o_buf = vec![0.0f32; d];
    let mut norm_x = vec![0.0f32; d];

    for b in 0..cfg.batch_size {
        let h_slice = &mut hidden[b * d..(b + 1) * d];
        norm_x.copy_from_slice(h_slice);
        rms_norm_inplace(&mut norm_x);

        matvec(&weights.w_q, &norm_x, d, d, &mut q_buf);
        matvec(&weights.w_k, &norm_x, d, d, &mut k_buf);
        matvec(&weights.w_v, &norm_x, d, d, &mut v_buf);

        let seq_state = &mut gdn_state[b * state_per_seq..(b + 1) * state_per_seq];
        for gh in 0..cfg.gdn_heads {
            let head_off = gh * hd;
            let q_h = &q_buf[head_off..head_off + hd];
            let k_h = &mut k_buf[head_off..head_off + hd];
            let v_h = &v_buf[head_off..head_off + hd];
            // L2-normalize key vector for bounded delta-rule recurrence stability.
            let k_norm = (k_h.iter().map(|&x| x * x).sum::<f32>() + 1e-6).sqrt();
            for x in k_h.iter_mut() {
                *x /= k_norm;
            }

            let s = &mut seq_state[gh * state_per_head..(gh + 1) * state_per_head];
            let y_h = &mut y_buf[head_off..head_off + hd];

            // v_hat[j] = sum_i S[i, j] * k[i]; delta[j] = beta * (v[j] - v_hat[j]).
            let mut delta = [0.0f32; 64];
            debug_assert!(hd <= delta.len());
            for j in 0..hd {
                let mut v_hat = 0.0f32;
                for (i, &ki) in k_h.iter().enumerate() {
                    v_hat += s[i * hd + j] * ki;
                }
                delta[j] = weights.beta * (v_h[j] - v_hat);
            }

            // State update: S[i, j] = decay * S[i, j] + k[i] * delta[j], and read y[j] = sum_i S[i, j] * q[i].
            y_h.fill(0.0);
            for i in 0..hd {
                let ki = k_h[i];
                let qi = q_h[i];
                let row = &mut s[i * hd..(i + 1) * hd];
                for j in 0..hd {
                    let updated = weights.decay * row[j] + ki * delta[j];
                    row[j] = updated;
                    y_h[j] += updated * qi;
                }
            }
        }

        matvec(&weights.w_o, &y_buf, d, d, &mut o_buf);
        for i in 0..d {
            h_slice[i] += o_buf[i] * 0.25;
        }
    }
}

/// Executes one dense SwiGLU MLP block (`down(silu(gate(norm(x))) * up(norm(x)))`) across batch `B`.
fn run_swiglu_mlp_layer(
    hidden: &mut [f32],
    weights: &LayerMlpWeights,
    batch_size: usize,
    d_model: usize,
    d_ff: usize,
) {
    let mut norm_x = vec![0.0f32; d_model];
    let mut gate = vec![0.0f32; d_ff];
    let mut up = vec![0.0f32; d_ff];
    let mut fused = vec![0.0f32; d_ff];
    let mut down = vec![0.0f32; d_model];

    for b in 0..batch_size {
        let h_slice = &mut hidden[b * d_model..(b + 1) * d_model];
        norm_x.copy_from_slice(h_slice);
        rms_norm_inplace(&mut norm_x);

        matvec(&weights.w_gate, &norm_x, d_ff, d_model, &mut gate);
        matvec(&weights.w_up, &norm_x, d_ff, d_model, &mut up);
        for i in 0..d_ff {
            fused[i] = silu(gate[i]) * up[i];
        }
        matvec(&weights.w_down, &fused, d_model, d_ff, &mut down);
        for i in 0..d_model {
            h_slice[i] += down[i] * 0.25;
        }
    }
}

/// Runs the directly measured CPU full hybrid decoder step verifier (Attention + GDN/LSA + Dense MLP).
#[must_use]
pub fn run_cpu_hybrid_step_verification(seed: u64, cfg: HybridStepConfig) -> CpuHybridStepReport {
    let mut rng = Lcg64::new(seed ^ 0x4b1d_a77e);
    let total_layers = cfg.num_attn_layers + cfg.num_gdn_layers;
    let d = cfg.d_model;
    let scale_w = 1.0 / (d as f32).sqrt();
    let scale_ff = 1.0 / (cfg.d_ff as f32).sqrt();

    // Build deterministic per-layer weights & paged KV pools.
    let mut attn_weights = Vec::with_capacity(cfg.num_attn_layers);
    let mut attn_pools = Vec::with_capacity(cfg.num_attn_layers);
    let mut attn_shared_pages = Vec::with_capacity(cfg.num_attn_layers);
    let mut attn_suffix_pages = Vec::with_capacity(cfg.num_attn_layers);
    let suffix_lens = vec![cfg.unique_suffix_len; cfg.batch_size];

    for _ in 0..cfg.num_attn_layers {
        attn_weights.push(AttnLayerWeights {
            w_q: (0..d * d)
                .map(|_| rng.next_f32_signed() * scale_w)
                .collect(),
            w_o: (0..d * d)
                .map(|_| rng.next_f32_signed() * scale_w)
                .collect(),
        });
        let mut pool = PagedKvPool::new(cfg.num_kv_heads, cfg.page_size, cfg.head_dim);
        let shared = pool.allocate_random_sequence(cfg.shared_prefix_len, &mut rng);
        let mut suff = Vec::with_capacity(cfg.batch_size);
        for _ in 0..cfg.batch_size {
            suff.push(pool.allocate_random_sequence(cfg.unique_suffix_len, &mut rng));
        }
        attn_pools.push(pool);
        attn_shared_pages.push(shared);
        attn_suffix_pages.push(suff);
    }

    let mut gdn_weights = Vec::with_capacity(cfg.num_gdn_layers);
    for _ in 0..cfg.num_gdn_layers {
        gdn_weights.push(GdnLayerWeights {
            w_q: (0..d * d)
                .map(|_| rng.next_f32_signed() * scale_w)
                .collect(),
            w_k: (0..d * d)
                .map(|_| rng.next_f32_signed() * scale_w)
                .collect(),
            w_v: (0..d * d)
                .map(|_| rng.next_f32_signed() * scale_w)
                .collect(),
            w_o: (0..d * d)
                .map(|_| rng.next_f32_signed() * scale_w)
                .collect(),
            decay: 0.92,
            beta: 0.15,
        });
    }

    let mut mlp_weights = Vec::with_capacity(total_layers);
    for _ in 0..total_layers {
        mlp_weights.push(LayerMlpWeights {
            w_gate: (0..cfg.d_ff * d)
                .map(|_| rng.next_f32_signed() * scale_w)
                .collect(),
            w_up: (0..cfg.d_ff * d)
                .map(|_| rng.next_f32_signed() * scale_w)
                .collect(),
            w_down: (0..d * cfg.d_ff)
                .map(|_| rng.next_f32_signed() * scale_ff)
                .collect(),
        });
    }

    let init_hidden: Vec<f32> = (0..cfg.batch_size * d)
        .map(|_| rng.next_f32_signed() * 0.5)
        .collect();
    let gdn_state_len =
        cfg.num_gdn_layers * cfg.batch_size * cfg.gdn_heads * cfg.gdn_head_dim * cfg.gdn_head_dim;
    let init_gdn_state: Vec<f32> = (0..gdn_state_len)
        .map(|_| rng.next_f32_signed() * 0.1)
        .collect();
    let vocab_size = 64usize;
    let lm_head: Vec<f32> = (0..vocab_size * d)
        .map(|_| rng.next_f32_signed() * scale_w)
        .collect();

    // Helper closure executing all 8 interleaved layers (3 GDN + 1 Attn repeated twice)
    // and timing Attention, GDN, and MLP separately.
    let run_full_step = |use_cascade: bool| -> (Vec<f32>, Vec<f32>, f64, f64, f64) {
        let mut hidden = init_hidden.clone();
        let mut gdn_state = init_gdn_state.clone();
        let gdn_layer_state_stride =
            cfg.batch_size * cfg.gdn_heads * cfg.gdn_head_dim * cfg.gdn_head_dim;

        let mut attn_us = 0.0f64;
        let mut gdn_us = 0.0f64;
        let mut mlp_us = 0.0f64;

        let mut attn_idx = 0usize;
        let mut gdn_idx = 0usize;
        let mut q_buf = vec![0.0f32; cfg.batch_size * d];
        let mut norm_x = vec![0.0f32; d];
        let mut o_proj = vec![0.0f32; d];

        for (layer_idx, layer_mlp) in mlp_weights.iter().enumerate().take(total_layers) {
            let is_attn = (layer_idx + 1) % 4 == 0 && attn_idx < cfg.num_attn_layers;
            if is_attn {
                let t0 = Instant::now();
                let aw = &attn_weights[attn_idx];
                for b in 0..cfg.batch_size {
                    norm_x.copy_from_slice(&hidden[b * d..(b + 1) * d]);
                    rms_norm_inplace(&mut norm_x);
                    matvec(&aw.w_q, &norm_x, d, d, &mut q_buf[b * d..(b + 1) * d]);
                }
                let attn_res = if use_cascade {
                    two_level_cascade_gqa_decode(
                        &attn_pools[attn_idx],
                        &q_buf,
                        cfg.batch_size,
                        cfg.num_q_heads,
                        &attn_shared_pages[attn_idx],
                        cfg.shared_prefix_len,
                        &attn_suffix_pages[attn_idx],
                        &suffix_lens,
                    )
                } else {
                    unshared_paged_gqa_decode(
                        &attn_pools[attn_idx],
                        &q_buf,
                        cfg.batch_size,
                        cfg.num_q_heads,
                        &attn_shared_pages[attn_idx],
                        cfg.shared_prefix_len,
                        &attn_suffix_pages[attn_idx],
                        &suffix_lens,
                    )
                };
                for b in 0..cfg.batch_size {
                    matvec(
                        &aw.w_o,
                        &attn_res.output[b * d..(b + 1) * d],
                        d,
                        d,
                        &mut o_proj,
                    );
                    let h_slice = &mut hidden[b * d..(b + 1) * d];
                    for i in 0..d {
                        h_slice[i] += o_proj[i] * 0.25;
                    }
                }
                attn_us += t0.elapsed().as_secs_f64() * 1.0e6;
                attn_idx += 1;
            } else {
                let t0 = Instant::now();
                let state_slice = &mut gdn_state
                    [gdn_idx * gdn_layer_state_stride..(gdn_idx + 1) * gdn_layer_state_stride];
                run_gdn_layer(&mut hidden, state_slice, &gdn_weights[gdn_idx], &cfg);
                gdn_us += t0.elapsed().as_secs_f64() * 1.0e6;
                gdn_idx += 1;
            }

            let t_mlp = Instant::now();
            run_swiglu_mlp_layer(&mut hidden, layer_mlp, cfg.batch_size, d, cfg.d_ff);
            mlp_us += t_mlp.elapsed().as_secs_f64() * 1.0e6;
        }

        (hidden, gdn_state, attn_us, gdn_us, mlp_us)
    };

    // Warmup + exact correctness verification.
    let (h_unshared, g_unshared, _, _, _) = run_full_step(false);
    let (h_cascade, g_cascade, _, _, _) = run_full_step(true);

    let mut max_hidden_err = 0.0f32;
    for (&a, &b) in h_unshared.iter().zip(h_cascade.iter()) {
        max_hidden_err = max_hidden_err.max((a - b).abs());
    }
    let mut max_gdn_err = 0.0f32;
    for (&a, &b) in g_unshared.iter().zip(g_cascade.iter()) {
        max_gdn_err = max_gdn_err.max((a - b).abs());
    }

    let tok_u = greedy_argmax_tokens(&h_unshared, cfg.batch_size, d, vocab_size, &lm_head);
    let tok_c = greedy_argmax_tokens(&h_cascade, cfg.batch_size, d, vocab_size, &lm_head);
    let matched = tok_u
        .iter()
        .zip(tok_c.iter())
        .filter(|(a, b)| a == b)
        .count();

    // Timed median over 7 runs.
    let repeats = 7usize;
    let mut u_attn_s = Vec::with_capacity(repeats);
    let mut c_attn_s = Vec::with_capacity(repeats);
    let mut gdn_s = Vec::with_capacity(repeats);
    let mut mlp_s = Vec::with_capacity(repeats);

    for _ in 0..repeats {
        let (hu, _, au, gu, mu) = run_full_step(false);
        std::hint::black_box(&hu);
        u_attn_s.push(au);
        gdn_s.push(gu);
        mlp_s.push(mu);

        let (hc, _, ac, _, _) = run_full_step(true);
        std::hint::black_box(&hc);
        c_attn_s.push(ac);
    }

    u_attn_s.sort_by(|a, b| a.total_cmp(b));
    c_attn_s.sort_by(|a, b| a.total_cmp(b));
    gdn_s.sort_by(|a, b| a.total_cmp(b));
    mlp_s.sort_by(|a, b| a.total_cmp(b));

    let unshared_attn_us = u_attn_s[repeats / 2];
    let cascade_attn_us = c_attn_s[repeats / 2];
    let gdn_time_us = gdn_s[repeats / 2];
    let mlp_time_us = mlp_s[repeats / 2];

    let unshared_total_step_us = unshared_attn_us + gdn_time_us + mlp_time_us;
    let cascade_total_step_us = cascade_attn_us + gdn_time_us + mlp_time_us;

    CpuHybridStepReport {
        batch_size: cfg.batch_size,
        num_attn_layers: cfg.num_attn_layers,
        num_gdn_layers: cfg.num_gdn_layers,
        unshared_attn_us,
        cascade_attn_us,
        gdn_time_us,
        mlp_time_us,
        unshared_total_step_us,
        cascade_total_step_us,
        attn_time_fraction: unshared_attn_us / unshared_total_step_us.max(1e-6),
        gdn_time_fraction: gdn_time_us / unshared_total_step_us.max(1e-6),
        mlp_time_fraction: mlp_time_us / unshared_total_step_us.max(1e-6),
        isolated_attn_speedup: unshared_attn_us / cascade_attn_us.max(1e-6),
        full_step_speedup: unshared_total_step_us / cascade_total_step_us.max(1e-6),
        max_abs_hidden_err: max_hidden_err,
        max_abs_gdn_state_err: max_gdn_err,
        greedy_argmax_match_pct: (matched as f64) * 100.0 / (cfg.batch_size as f64),
    }
}
