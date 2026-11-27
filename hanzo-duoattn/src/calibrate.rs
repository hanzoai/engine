// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Optimization-based offline per-model DuoAttention head calibration (`arXiv:2410.10819`).
//!
//! Learns continuous gating values `alpha_{l,h} in [0, 1]` for every `(attn_layer, kv_head)`
//! via projected gradient descent (Adam + L1 sparsity proximal step + box projection)
//! minimizing the teacher-forced L2 attention-output divergence between full attention and
//! blended `alpha * full_attn + (1 - alpha) * streaming_attn`, plus an L1 sparsity penalty
//! `lambda * sum_{l,h} |alpha_{l,h}|`, over synthetic passkey & distant-reference sequences.
//!
//! Calibration NEVER peeks at latent model labels — it discovers retrieval heads solely from
//! forward-pass attention discrepancies and produces a versioned, SHA-256 checksummed,
//! model-bound [`HeadCalibrationProfile`].

use std::time::Instant;

use crate::attention::DualAttentionModel;
use crate::error::DuoAttnError;
use crate::profile::{HeadCalibrationProfile, HeadRole};

/// One calibration example consisting of embedded tokens and evaluation query indices
/// where distant retrieval is required.
#[derive(Clone, Debug)]
pub struct CalibrationSequence {
    /// Embedded token sequence `[seq_len, d_model]`.
    pub embeddings: Vec<Vec<f32>>,
    /// Query positions (typically tail recall positions) included in the distillation loss.
    pub eval_positions: Vec<usize>,
}

/// Hyperparameters for offline DuoAttention head calibration.
#[derive(Clone, Debug)]
pub struct CalibrationConfig {
    /// Number of leading attention sink tokens retained by streaming heads.
    pub sink_tokens: usize,
    /// Number of recent sliding-window tokens retained by streaming heads.
    pub window_tokens: usize,
    /// Target fraction of KV heads assigned to [`HeadRole::Retrieval`].
    pub retrieval_ratio: f32,
    /// Number of projected Adam optimization steps.
    pub steps: usize,
    /// Learning rate for Adam updates on `alpha`.
    pub learning_rate: f64,
    /// L1 sparsity regularization coefficient `lambda`.
    pub l1_lambda: f32,
    /// Deterministic calibration seed recorded in the emitted profile.
    pub seed: u64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            sink_tokens: 4,
            window_tokens: 32,
            retrieval_ratio: 0.25,
            steps: 240,
            learning_rate: 0.05,
            l1_lambda: 0.035,
            seed: 1346,
        }
    }
}

/// Full outcome of offline per-model DuoAttention calibration.
#[derive(Clone, Debug)]
pub struct CalibrationReport {
    /// Checksummed, model-bound calibration profile ready for runtime verification.
    pub profile: HeadCalibrationProfile,
    /// Initial total objective (L2 + L1) at `alpha = 1.0`.
    pub initial_loss: f64,
    /// Final total objective (L2 + L1) after projected optimization.
    pub final_loss: f64,
    /// Pure L2 attention-output divergence term at convergence.
    pub final_l2_loss: f64,
    /// Mean learned `alpha` across heads classified as `Retrieval`.
    pub mean_retrieval_alpha: f64,
    /// Mean learned `alpha` across heads classified as `Streaming`.
    pub mean_streaming_alpha: f64,
    /// Wall-clock duration of offline calibration in milliseconds.
    pub calibration_wall_ms: f64,
    /// Number of calibration sequences processed.
    pub num_sequences: usize,
    /// Total evaluation query tokens aggregated into the Gram matrices.
    pub total_eval_tokens: usize,
}

impl CalibrationReport {
    /// Computes precision and recall of the calibrated retrieval head set against a model's
    /// latent diagnostic roles (used exclusively for offline validation & test assertions).
    #[must_use]
    pub fn diagnostic_precision_recall(&self, model: &DualAttentionModel) -> (f64, f64) {
        let truth = model.diagnostic_true_roles();
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_count = 0usize;
        for (head, &true_role) in self.profile.heads.iter().zip(truth.iter()) {
            match (head.role, true_role) {
                (HeadRole::Retrieval, HeadRole::Retrieval) => tp += 1,
                (HeadRole::Retrieval, HeadRole::Streaming) => fp += 1,
                (HeadRole::Streaming, HeadRole::Retrieval) => fn_count += 1,
                (HeadRole::Streaming, HeadRole::Streaming) => {}
            }
        }
        let precision = if tp + fp == 0 {
            1.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_count == 0 {
            1.0
        } else {
            tp as f64 / (tp + fn_count) as f64
        };
        (precision, recall)
    }
}

/// Executes offline optimization-based DuoAttention head calibration for `model`.
///
/// 1. Runs real teacher-forced full vs streaming multi-head attention over `sequences`
///    to collect `Delta_{l,h} = (O_full_{l,h} - O_stream_{l,h}) W_{O,h}` at each query token.
/// 2. Accumulates the exact per-layer Gram matrix `G_l[h1, h2] = mean <Delta_{l,h1}, Delta_{l,h2}>`.
/// 3. Minimizes `L(alpha) = sum_l (1 - alpha_l)^T G_l (1 - alpha_l) + lambda * sum_{l,h} alpha_{l,h}`
///    subject to `alpha_{l,h} in [0, 1]` via projected Adam starting from `alpha = 1.0`.
/// 4. Emits a cryptographically checksummed, model-bound [`HeadCalibrationProfile`].
pub fn calibrate_model(
    model: &DualAttentionModel,
    sequences: &[CalibrationSequence],
    config: &CalibrationConfig,
) -> Result<CalibrationReport, DuoAttnError> {
    if sequences.is_empty() {
        return Err(DuoAttnError::InvalidConfig(
            "at least one calibration sequence is required".to_string(),
        ));
    }
    if config.steps == 0 {
        return Err(DuoAttnError::InvalidConfig(
            "calibration steps must be >= 1".to_string(),
        ));
    }
    if config.sink_tokens == 0 || config.window_tokens == 0 {
        return Err(DuoAttnError::InvalidConfig(
            "sink_tokens and window_tokens must be >= 1".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&config.retrieval_ratio) || config.retrieval_ratio.is_nan() {
        return Err(DuoAttnError::InvalidConfig(format!(
            "retrieval_ratio must be in [0.0, 1.0], got {}",
            config.retrieval_ratio
        )));
    }
    if config.l1_lambda < 0.0 || !config.l1_lambda.is_finite() {
        return Err(DuoAttnError::InvalidConfig(format!(
            "l1_lambda must be finite and >= 0.0, got {}",
            config.l1_lambda
        )));
    }

    let start = Instant::now();
    let num_l = model.identity().num_attn_layers;
    let num_kv = model.identity().num_kv_heads;
    let total_kv = model.identity().total_kv_heads();

    // Exact per-layer symmetric PSD Gram matrix G_l[h1 * num_kv + h2].
    let mut gram = vec![vec![0.0_f64; num_kv * num_kv]; num_l];
    let mut total_eval_tokens = 0usize;

    for seq in sequences {
        let disc = model.collect_calibration_discrepancies(
            &seq.embeddings,
            &seq.eval_positions,
            config.sink_tokens,
            config.window_tokens,
        )?;
        let n_eval = disc[0][0].len();
        total_eval_tokens += n_eval;
        for l in 0..num_l {
            for h1 in 0..num_kv {
                for h2 in h1..num_kv {
                    let mut sum_dot = 0.0_f64;
                    for t in 0..n_eval {
                        let a = &disc[l][h1][t];
                        let b = &disc[l][h2][t];
                        let mut dot = 0.0_f64;
                        for i in 0..a.len() {
                            dot += f64::from(a[i]) * f64::from(b[i]);
                        }
                        sum_dot += dot;
                    }
                    gram[l][h1 * num_kv + h2] += sum_dot;
                    if h1 != h2 {
                        gram[l][h2 * num_kv + h1] += sum_dot;
                    }
                }
            }
        }
    }

    if total_eval_tokens == 0 {
        return Err(DuoAttnError::InvalidConfig(
            "calibration sequences produced zero valid eval positions".to_string(),
        ));
    }

    let inv_eval = 1.0 / (total_eval_tokens as f64);
    for g_l in &mut gram {
        for val in g_l.iter_mut() {
            *val *= inv_eval;
        }
    }

    // Initialize alpha_{l,h} = 1.0 (full attention), matching DuoAttention Section 3.1.
    let mut alphas = vec![1.0_f64; total_kv];
    let mut m = vec![0.0_f64; total_kv];
    let mut v = vec![0.0_f64; total_kv];
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let eps = 1e-8_f64;
    let lambda = f64::from(config.l1_lambda);

    let (initial_l2, initial_l1) = evaluate_objective(&gram, &alphas, num_l, num_kv, lambda);
    let initial_loss = initial_l2 + initial_l1;

    let mut grad = vec![0.0_f64; total_kv];
    for step in 1..=config.steps {
        // Compute exact analytical gradient: dL/d(alpha_l) = -2 * G_l * (1 - alpha_l) + lambda * 1
        for l in 0..num_l {
            let g_l = &gram[l];
            for h1 in 0..num_kv {
                let mut acc = 0.0_f64;
                for h2 in 0..num_kv {
                    let one_minus = 1.0 - alphas[l * num_kv + h2];
                    acc += g_l[h1 * num_kv + h2] * one_minus;
                }
                grad[l * num_kv + h1] = -2.0 * acc + lambda;
            }
        }

        let b1_corr = 1.0 - beta1.powi(step as i32);
        let b2_corr = 1.0 - beta2.powi(step as i32);
        for i in 0..total_kv {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            let m_hat = m[i] / b1_corr;
            let v_hat = v[i] / b2_corr;
            let updated = alphas[i] - config.learning_rate * m_hat / (v_hat.sqrt() + eps);
            // Box projection onto [0.0, 1.0].
            alphas[i] = updated.clamp(0.0, 1.0);
        }
    }

    let (final_l2, final_l1) = evaluate_objective(&gram, &alphas, num_l, num_kv, lambda);
    let final_loss = final_l2 + final_l1;

    let alphas_f32: Vec<f32> = alphas.iter().map(|&a| a as f32).collect();
    let profile = HeadCalibrationProfile::from_alphas(
        model.identity(),
        config.sink_tokens,
        config.window_tokens,
        config.retrieval_ratio,
        config.seed,
        config.steps,
        config.l1_lambda,
        initial_loss,
        final_loss,
        &alphas_f32,
    )?;

    let mut ret_sum = 0.0_f64;
    let mut ret_cnt = 0usize;
    let mut str_sum = 0.0_f64;
    let mut str_cnt = 0usize;
    for head in &profile.heads {
        match head.role {
            HeadRole::Retrieval => {
                ret_sum += f64::from(head.alpha);
                ret_cnt += 1;
            }
            HeadRole::Streaming => {
                str_sum += f64::from(head.alpha);
                str_cnt += 1;
            }
        }
    }

    let calibration_wall_ms = start.elapsed().as_secs_f64() * 1_000.0;
    Ok(CalibrationReport {
        profile,
        initial_loss,
        final_loss,
        final_l2_loss: final_l2,
        mean_retrieval_alpha: if ret_cnt > 0 {
            ret_sum / ret_cnt as f64
        } else {
            0.0
        },
        mean_streaming_alpha: if str_cnt > 0 {
            str_sum / str_cnt as f64
        } else {
            0.0
        },
        calibration_wall_ms,
        num_sequences: sequences.len(),
        total_eval_tokens,
    })
}

fn evaluate_objective(
    gram: &[Vec<f64>],
    alphas: &[f64],
    num_l: usize,
    num_kv: usize,
    lambda: f64,
) -> (f64, f64) {
    let mut l2 = 0.0_f64;
    let mut l1 = 0.0_f64;
    for l in 0..num_l {
        let g_l = &gram[l];
        for h1 in 0..num_kv {
            let a1 = alphas[l * num_kv + h1];
            l1 += lambda * a1.abs();
            let d1 = 1.0 - a1;
            for h2 in 0..num_kv {
                let d2 = 1.0 - alphas[l * num_kv + h2];
                l2 += d1 * g_l[h1 * num_kv + h2] * d2;
            }
        }
    }
    (l2, l1)
}
