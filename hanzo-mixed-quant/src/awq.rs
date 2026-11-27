// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Clean-room Activation-Aware Quantization (AWQ).
//!
//! Reference algorithm: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
//! Compression and Acceleration" (MLSys 2024). Independent clean-room Rust implementation.
//!
//! Collects real per-input-channel activation magnitudes $s_x[j] = \frac{1}{N}\sum_t |x_t[j]|$
//! on calibration batches, protects the top `1%` salient outlier channels, and grid-searches
//! $\alpha \in [0, 1]$ minimizing:
//!
//! $$\mathcal{L}(\alpha) = \| X W^T - (X \cdot \mathrm{diag}(s)^{-1}) \, Q(W \cdot \mathrm{diag}(s))^T \|_F^2$$

use crate::error::MixedQuantError;
use crate::formats::{
    f32_to_bf16_bits, matmul_batch_f32, PackedWeights, ProtectedChannels, QuantFormat,
    QuantizedLinear,
};
use crate::rotation::RotationTransform;

/// Per-input-channel activation statistics collected on calibration batches.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Input channel dimension.
    pub cols: usize,
    /// Number of calibration token vectors observed.
    pub num_tokens: usize,
    /// Mean absolute magnitude per input channel `s_x[j]`.
    pub mean_abs: Vec<f32>,
    /// Root-mean-square magnitude per input channel.
    pub rms: Vec<f32>,
    /// Maximum absolute magnitude observed per input channel.
    pub max_abs: Vec<f32>,
}

impl ActivationStats {
    /// Collects exact per-channel activation statistics from row-major `[num_tokens, cols]`.
    pub fn collect(
        activations: &[f32],
        num_tokens: usize,
        cols: usize,
    ) -> Result<Self, MixedQuantError> {
        if num_tokens == 0 || cols == 0 || activations.len() != num_tokens * cols {
            return Err(MixedQuantError::InvalidCalibrationData {
                reason: "activations must be non-empty and match num_tokens * cols",
            });
        }
        let mut sum_abs = vec![0.0f64; cols];
        let mut sum_sq = vec![0.0f64; cols];
        let mut max_abs = vec![0.0f32; cols];

        for t in 0..num_tokens {
            let row = &activations[t * cols..(t + 1) * cols];
            for (c, &raw) in row.iter().enumerate() {
                if !raw.is_finite() {
                    return Err(MixedQuantError::InvalidCalibrationData {
                        reason: "non-finite activation encountered during AWQ stats collection",
                    });
                }
                let a = raw.abs();
                sum_abs[c] += a as f64;
                sum_sq[c] += (raw as f64) * (raw as f64);
                if a > max_abs[c] {
                    max_abs[c] = a;
                }
            }
        }

        let inv_n = 1.0 / (num_tokens as f64);
        let mean_abs = sum_abs.into_iter().map(|s| (s * inv_n) as f32).collect();
        let rms = sum_sq
            .into_iter()
            .map(|s| (s * inv_n).sqrt() as f32)
            .collect();

        Ok(Self {
            cols,
            num_tokens,
            mean_abs,
            rms,
            max_abs,
        })
    }

    /// Returns the indices of the top `k` most salient channels sorted ascending by index.
    #[must_use]
    pub fn top_salient_channels(&self, ratio: f32) -> Vec<usize> {
        if ratio <= 0.0 || self.cols == 0 {
            return Vec::new();
        }
        let k = ((self.cols as f32 * ratio).ceil() as usize)
            .max(1)
            .min(self.cols);
        let mut order: Vec<usize> = (0..self.cols).collect();
        order.sort_by(|&a, &b| {
            self.mean_abs[b]
                .partial_cmp(&self.mean_abs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut selected = order[..k].to_vec();
        selected.sort_unstable();
        selected
    }

    /// Ratio between the max channel mean-abs and the median channel mean-abs (outlier sharpness).
    #[must_use]
    pub fn outlier_ratio(&self) -> f32 {
        if self.mean_abs.is_empty() {
            return 1.0;
        }
        let mut sorted = self.mean_abs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2].max(1e-6);
        let max_v = *sorted.last().unwrap_or(&median);
        max_v / median
    }
}

/// Configuration for AWQ calibration.
#[derive(Debug, Clone)]
pub struct AwqConfig {
    /// Target low-bit quantization format (`Int4Sym`, `Int4Asym`, or `Ternary`).
    pub format: QuantFormat,
    /// Per-group quantization block size.
    pub group_size: usize,
    /// Number of grid steps in `alpha in [0, 1]` (e.g. `20` => step `0.05`).
    pub grid_steps: usize,
    /// Fraction of top salient channels protected in BF16 (default `0.01` = 1%).
    pub salient_ratio: f32,
    /// Whether to keep top salient channels in BF16 alongside AWQ channel scaling.
    pub protect_salient_bf16: bool,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            format: QuantFormat::Int4Asym,
            group_size: 32,
            grid_steps: 20,
            salient_ratio: 0.01,
            protect_salient_bf16: true,
        }
    }
}

/// Result of calibrating a linear layer with AWQ.
#[derive(Debug, Clone)]
pub struct AwqCalibrationResult {
    /// Calibrated quantized linear layer with AWQ inverse scales & optional BF16 salient channels.
    pub linear: QuantizedLinear,
    /// Optimal salience exponent `alpha* in [0, 1]`.
    pub best_alpha: f32,
    /// Uncalibrated plain PTQ MSE (`alpha = 0`, no protection).
    pub baseline_ptq_mse: f64,
    /// AWQ calibrated MSE on the calibration batch.
    pub awq_mse: f64,
    /// Relative MSE reduction `(baseline - awq) / baseline`.
    pub relative_mse_reduction: f64,
    /// Salient channel indices protected in BF16 (if enabled).
    pub salient_channels: Vec<usize>,
    /// Channel salience outlier ratio `max(s_x) / median(s_x)`.
    pub outlier_ratio: f32,
}

/// Calibrates a linear projection with AWQ on real calibration activations.
pub fn calibrate_awq(
    weights: &[f32],
    rows: usize,
    cols: usize,
    activations: &[f32],
    num_tokens: usize,
    config: &AwqConfig,
) -> Result<AwqCalibrationResult, MixedQuantError> {
    calibrate_awq_with_rotation(weights, rows, cols, activations, num_tokens, config, None)
}

/// Calibrates a linear projection with optional SpinQuant orthogonal rotation followed by AWQ.
pub fn calibrate_awq_with_rotation(
    weights: &[f32],
    rows: usize,
    cols: usize,
    activations: &[f32],
    num_tokens: usize,
    config: &AwqConfig,
    rotation: Option<RotationTransform>,
) -> Result<AwqCalibrationResult, MixedQuantError> {
    if rows == 0 || cols == 0 || weights.len() != rows * cols {
        return Err(MixedQuantError::InvalidShape {
            context: "calibrate_awq weights",
            expected: rows.saturating_mul(cols),
            actual: weights.len(),
        });
    }
    if num_tokens == 0 || activations.len() != num_tokens * cols {
        return Err(MixedQuantError::InvalidCalibrationData {
            reason: "activations shape mismatch in calibrate_awq",
        });
    }

    // Exact FP32 reference output Y_ref = X * W^T
    let ref_out = matmul_batch_f32(weights, rows, cols, activations, num_tokens)?;

    // Plain unrotated, uncalibrated PTQ baseline MSE
    let baseline_lin =
        QuantizedLinear::quantize_plain(weights, rows, cols, config.format, config.group_size)?;
    let baseline_out = baseline_lin.forward_batch(activations, num_tokens)?;
    let baseline_ptq_mse = mse_f64(&ref_out, &baseline_out);

    // Apply optional SpinQuant rotation to both W and X before AWQ scaling
    let (w_work, x_work) = if let Some(ref rot) = rotation {
        (
            rot.rotate_matrix_rows(weights, rows, cols)?,
            rot.rotate_matrix_rows(activations, num_tokens, cols)?,
        )
    } else {
        (weights.to_vec(), activations.to_vec())
    };

    let stats = ActivationStats::collect(&x_work, num_tokens, cols)?;
    let outlier_ratio = stats.outlier_ratio();

    // Extract optional 1% BF16 protected channels (only for sub-8-bit formats)
    let (protected, salient_channels, w_base) = if config.protect_salient_bf16
        && matches!(
            config.format,
            QuantFormat::Ternary | QuantFormat::Int4Sym | QuantFormat::Int4Asym
        ) {
        let salient = stats.top_salient_channels(config.salient_ratio);
        let k = salient.len();
        let mut bf16_weights = vec![0u16; rows * k];
        let mut w_residual = w_work.clone();
        for r in 0..rows {
            for (idx, &c) in salient.iter().enumerate() {
                bf16_weights[r * k + idx] = f32_to_bf16_bits(w_residual[r * cols + c]);
                w_residual[r * cols + c] = 0.0;
            }
        }
        (
            Some(ProtectedChannels {
                channel_indices: salient.clone(),
                bf16_weights,
                rows,
            }),
            salient,
            w_residual,
        )
    } else {
        (None, Vec::new(), w_work)
    };

    // Precompute protected channel contribution on x_work once
    let mut prot_contrib = vec![0.0f32; num_tokens * rows];
    if let Some(ref p) = protected {
        for t in 0..num_tokens {
            p.accumulate_matvec(
                &x_work[t * cols..(t + 1) * cols],
                &mut prot_contrib[t * rows..(t + 1) * rows],
            );
        }
    }

    let steps = config.grid_steps.max(1);
    let mut best_alpha = 0.0f32;
    let mut best_mse = f64::INFINITY;
    let mut best_weights: Option<PackedWeights> = None;
    let mut best_inv_scales: Option<Vec<f32>> = None;

    let mut w_scaled = vec![0.0f32; rows * cols];
    let mut x_scaled = vec![0.0f32; cols];
    let mut out_buf = vec![0.0f32; rows];

    for step in 0..=steps {
        let alpha = (step as f32) / (steps as f32);
        let (scales, inv_scales) = compute_awq_scales(&stats.mean_abs, alpha);

        // Scale weight columns: W'[:, c] = W[:, c] * s[c]
        for r in 0..rows {
            let src_row = &w_base[r * cols..(r + 1) * cols];
            let dst_row = &mut w_scaled[r * cols..(r + 1) * cols];
            for c in 0..cols {
                dst_row[c] = src_row[c] * scales[c];
            }
        }

        let cand_packed =
            PackedWeights::quantize(&w_scaled, rows, cols, config.format, config.group_size)?;

        // Evaluate reconstruction error ||X W^T - (X diag(s)^-1) Q(W diag(s))^T||^2
        let mut sq_err = 0.0f64;
        for t in 0..num_tokens {
            let x_row = &x_work[t * cols..(t + 1) * cols];
            for c in 0..cols {
                x_scaled[c] = x_row[c] * inv_scales[c];
            }
            for &c in &salient_channels {
                x_scaled[c] = 0.0;
            }
            cand_packed.matvec(&x_scaled, &mut out_buf)?;
            let ref_row = &ref_out[t * rows..(t + 1) * rows];
            let prot_row = &prot_contrib[t * rows..(t + 1) * rows];
            for r in 0..rows {
                let diff = ((out_buf[r] + prot_row[r]) - ref_row[r]) as f64;
                sq_err += diff * diff;
            }
        }
        let mse = sq_err / (num_tokens * rows) as f64;
        if mse < best_mse {
            best_mse = mse;
            best_alpha = alpha;
            best_weights = Some(cand_packed);
            best_inv_scales = if alpha > 0.0 { Some(inv_scales) } else { None };
        }
    }

    let weights_packed = best_weights.ok_or(MixedQuantError::InvalidCalibrationData {
        reason: "AWQ grid search produced no candidate",
    })?;
    let linear = QuantizedLinear {
        weights: weights_packed,
        awq_inv_scales: best_inv_scales,
        protected_channels: protected,
        input_rotation: rotation,
        qad_row_scales: None,
    };

    let rel_reduction = if baseline_ptq_mse > 1e-18 {
        (baseline_ptq_mse - best_mse) / baseline_ptq_mse
    } else {
        0.0
    };

    Ok(AwqCalibrationResult {
        linear,
        best_alpha,
        baseline_ptq_mse,
        awq_mse: best_mse,
        relative_mse_reduction: rel_reduction,
        salient_channels,
        outlier_ratio,
    })
}

/// Computes normalized AWQ per-channel scaling factors `s` and their inverses `1 / s`.
///
/// Follows AWQ geometric-mean normalization `s_j = raw_j / sqrt(max(raw) * min(raw))`
/// to keep scaled weight dynamic ranges well-centered.
#[must_use]
pub fn compute_awq_scales(mean_abs: &[f32], alpha: f32) -> (Vec<f32>, Vec<f32>) {
    let cols = mean_abs.len();
    if alpha <= 0.0 || cols == 0 {
        return (vec![1.0; cols], vec![1.0; cols]);
    }
    let mut raw = Vec::with_capacity(cols);
    let mut min_s = f32::INFINITY;
    let mut max_s = 0.0f32;
    for &m in mean_abs {
        let s = m.max(1e-4).powf(alpha).clamp(1e-3, 1e3);
        if s < min_s {
            min_s = s;
        }
        if s > max_s {
            max_s = s;
        }
        raw.push(s);
    }
    let geom = (min_s * max_s).sqrt().max(1e-6);
    let mut scales = Vec::with_capacity(cols);
    let mut inv_scales = Vec::with_capacity(cols);
    for r in raw {
        let s = (r / geom).clamp(0.05, 20.0);
        scales.push(s);
        inv_scales.push(1.0 / s);
    }
    (scales, inv_scales)
}

/// Computes exact mean squared error in `f64`.
#[must_use]
pub fn mse_f64(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x - y) as f64;
            d * d
        })
        .sum();
    sum / (a.len() as f64)
}
