//! Two-level low-communication optimizer (inner AdamW + outer Nesterov
//! momentum), pseudo-gradient compression with error feedback, and
//! Byzantine-robust aggregation with declared trust-tier taxonomy.
//!
//! During each outer synchronization round `t`:
//! 1. Each worker independently executes `H` local inner AdamW steps on its
//!    local shard without cross-WAN traffic.
//! 2. Each worker computes its pseudo-gradient `delta_k = theta^(t) - theta_k^(t, H)`,
//!    optionally compresses it using block-symmetric INT8 quantization with
//!    error-feedback residual accumulation, and transmits the signed update.
//! 3. Surviving updates are filtered through cryptographic receipt checks,
//!    free-rider step checks, norm bounds, held-out validation-loss spot checks,
//!    and coordinate-wise trimmed-mean aggregation before applying an outer
//!    Nesterov momentum step.

use crate::model::SmallZenLm;

/// Explicit verification trust taxonomy so signed receipts + validation spot
/// checks are never conflated with cryptographic zk proofs or hardware TEEs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrustTier {
    /// Authenticated worker round receipts (HMAC-SHA256 + SHA-256 weight/delta
    /// commitments) combined with norm bounds, held-out validation-loss spot
    /// checking, and Byzantine-robust trimmed-mean aggregation.
    SignedDeltaAndLossSpotCheck,
    /// Hardware-rooted enclave measurement attestation (out of scope for CPU
    /// loopback tests; declared explicitly for taxonomy completeness).
    HardwareTeeAttestation,
    /// Succinct cryptographic execution proof over training transitions (out of
    /// scope here; declared explicitly for taxonomy completeness).
    CryptographicZkProof,
}

impl TrustTier {
    /// Canonical string identifier emitted in wire receipts and JSON dossiers.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SignedDeltaAndLossSpotCheck => "signed_delta_and_loss_spot_check",
            Self::HardwareTeeAttestation => "hardware_tee_attestation",
            Self::CryptographicZkProof => "cryptographic_zk_proof",
        }
    }
}

/// Wire compression mode for worker pseudo-gradients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressionMode {
    /// Exact little-endian IEEE-754 FP32 encoding (4 bytes/parameter).
    Fp32Exact,
    /// Per-block symmetric INT8 quantization (1 FP32 scale per block + 1 i8 per
    /// parameter) with worker-side error-feedback residual accumulation.
    Int8BlockSymmetric {
        /// Number of parameters per quantization scale block (e.g., 64).
        block_size: usize,
    },
}

impl CompressionMode {
    /// Wire tag byte used in binary TCP framing.
    #[must_use]
    pub fn wire_tag(&self) -> u8 {
        match self {
            Self::Fp32Exact => 0,
            Self::Int8BlockSymmetric { .. } => 1,
        }
    }

    /// Block size when INT8 compression is active, or 0 for FP32.
    #[must_use]
    pub fn block_size_u16(&self) -> u16 {
        match self {
            Self::Fp32Exact => 0,
            Self::Int8BlockSymmetric { block_size } => (*block_size).min(u16::MAX as usize) as u16,
        }
    }

    /// Reconstructs compression mode from wire fields.
    #[must_use]
    pub fn from_wire(tag: u8, block_size: u16) -> Option<Self> {
        match tag {
            0 => Some(Self::Fp32Exact),
            1 if block_size > 0 => Some(Self::Int8BlockSymmetric {
                block_size: block_size as usize,
            }),
            _ => None,
        }
    }
}

/// Encodes a pseudo-gradient vector into compact wire bytes and updates the
/// caller's error-feedback residual buffer when INT8 quantization is active.
#[must_use]
pub fn encode_pseudo_gradient(
    delta: &[f32],
    mode: CompressionMode,
    error_feedback: Option<&mut [f32]>,
) -> Vec<u8> {
    let mut compensated = delta.to_vec();
    if let Some(ref ef) = error_feedback {
        if ef.len() == compensated.len() {
            for (c, &e) in compensated.iter_mut().zip(ef.iter()) {
                *c += e;
            }
        }
    }

    match mode {
        CompressionMode::Fp32Exact => {
            if let Some(ef) = error_feedback {
                ef.fill(0.0);
            }
            let mut out = Vec::with_capacity(8 + compensated.len() * 4);
            out.push(0u8);
            out.extend_from_slice(&0u16.to_le_bytes());
            out.push(0u8);
            out.extend_from_slice(&(compensated.len() as u32).to_le_bytes());
            for &v in &compensated {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
        CompressionMode::Int8BlockSymmetric { block_size } => {
            let bs = block_size.max(1);
            let num_blocks = compensated.len().div_ceil(bs);
            let mut out = Vec::with_capacity(8 + num_blocks * 4 + compensated.len());
            out.push(1u8);
            out.extend_from_slice(&(bs as u16).to_le_bytes());
            out.push(0u8);
            out.extend_from_slice(&(compensated.len() as u32).to_le_bytes());

            let mut reconstructed = vec![0.0f32; compensated.len()];
            for (b_idx, chunk) in compensated.chunks(bs).enumerate() {
                let mut max_abs = 0.0f32;
                for &v in chunk {
                    let a = v.abs();
                    if a.is_finite() && a > max_abs {
                        max_abs = a;
                    }
                }
                let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
                out.extend_from_slice(&scale.to_le_bytes());
                let base = b_idx * bs;
                for (i, &v) in chunk.iter().enumerate() {
                    let q = if max_abs > 0.0 {
                        (v / scale).round().clamp(-127.0, 127.0) as i8
                    } else {
                        0i8
                    };
                    out.push(q as u8);
                    reconstructed[base + i] = (q as f32) * scale;
                }
            }

            if let Some(ef) = error_feedback {
                if ef.len() == compensated.len() {
                    for i in 0..compensated.len() {
                        ef[i] = compensated[i] - reconstructed[i];
                    }
                }
            }
            out
        }
    }
}

/// Decodes wire-encoded pseudo-gradient bytes back into FP32 values.
pub fn decode_pseudo_gradient(bytes: &[u8], expected_len: usize) -> Result<Vec<f32>, String> {
    if bytes.len() < 8 {
        return Err("encoded pseudo-gradient shorter than 8-byte header".to_string());
    }
    let tag = bytes[0];
    let block_size = u16::from_le_bytes([bytes[1], bytes[2]]) as usize;
    let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if count != expected_len {
        return Err(format!(
            "pseudo-gradient length mismatch: expected {expected_len}, got {count}"
        ));
    }

    match tag {
        0 => {
            let need = 8 + count * 4;
            if bytes.len() != need {
                return Err(format!(
                    "FP32 pseudo-gradient byte length mismatch: expected {need}, got {}",
                    bytes.len()
                ));
            }
            let mut out = Vec::with_capacity(count);
            for &chunk in bytes[8..].as_chunks::<4>().0 {
                out.push(f32::from_le_bytes(chunk));
            }
            Ok(out)
        }
        1 => {
            if block_size == 0 {
                return Err("INT8 pseudo-gradient has zero block_size".to_string());
            }
            let mut out = Vec::with_capacity(count);
            let mut cursor = 8usize;
            while out.len() < count {
                if cursor + 4 > bytes.len() {
                    return Err("truncated INT8 scale header".to_string());
                }
                let scale = f32::from_le_bytes([
                    bytes[cursor],
                    bytes[cursor + 1],
                    bytes[cursor + 2],
                    bytes[cursor + 3],
                ]);
                cursor += 4;
                let take = block_size.min(count - out.len());
                if cursor + take > bytes.len() {
                    return Err("truncated INT8 block payload".to_string());
                }
                for &q_u8 in &bytes[cursor..cursor + take] {
                    out.push((q_u8 as i8 as f32) * scale);
                }
                cursor += take;
            }
            if cursor != bytes.len() {
                return Err("trailing bytes in INT8 pseudo-gradient".to_string());
            }
            Ok(out)
        }
        other => Err(format!("unknown pseudo-gradient compression tag {other}")),
    }
}

/// Inner worker AdamW optimizer hyperparameters.
#[derive(Clone, Debug)]
pub struct InnerAdamWConfig {
    /// Inner step size.
    pub learning_rate: f32,
    /// First-moment decay `beta1`.
    pub beta1: f32,
    /// Second-moment decay `beta2`.
    pub beta2: f32,
    /// Numerical denominator epsilon.
    pub eps: f32,
    /// Decoupled weight decay coefficient.
    pub weight_decay: f32,
    /// Global gradient L2-norm clip ceiling.
    pub max_grad_norm: f32,
}

impl Default for InnerAdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 8e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-3,
            max_grad_norm: 2.0,
        }
    }
}

/// Stateful per-worker inner AdamW optimizer.
#[derive(Clone, Debug)]
pub struct InnerAdamW {
    /// Optimizer configuration.
    pub config: InnerAdamWConfig,
    /// First moment vector.
    m: Vec<f32>,
    /// Second moment vector.
    v: Vec<f32>,
    /// Step counter for bias correction.
    step: u64,
}

impl InnerAdamW {
    /// Creates zero-initialized inner AdamW moment buffers for `param_count`.
    #[must_use]
    pub fn new(param_count: usize, config: InnerAdamWConfig) -> Self {
        Self {
            config,
            m: vec![0.0f32; param_count],
            v: vec![0.0f32; param_count],
            step: 0,
        }
    }

    /// Resets moment state at the start of a new outer round or on recovery.
    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.step = 0;
    }

    /// Applies one decoupled-weight-decay AdamW update in place to `params`.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        let mut grad_sq_sum = 0.0f64;
        for &g in grads {
            grad_sq_sum += (g as f64) * (g as f64);
        }
        let grad_norm = grad_sq_sum.sqrt() as f32;
        let clip_scale = if grad_norm > self.config.max_grad_norm && grad_norm > 0.0 {
            self.config.max_grad_norm / grad_norm
        } else {
            1.0
        };

        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        let lr = self.config.learning_rate;
        let wd = self.config.weight_decay;
        let eps = self.config.eps;

        let bias_corr1 = 1.0 - b1.powi(self.step.min(i32::MAX as u64) as i32);
        let bias_corr2 = 1.0 - b2.powi(self.step.min(i32::MAX as u64) as i32);

        for i in 0..params.len() {
            let g = grads[i] * clip_scale;
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let m_hat = self.m[i] / bias_corr1;
            let v_hat = self.v[i] / bias_corr2;
            params[i] *= 1.0 - lr * wd;
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

/// Aggregation policy used when merging surviving worker pseudo-gradients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregationRule {
    /// Byzantine-robust coordinate-wise trimmed mean (dropping extreme tail
    /// values per coordinate when `n >= 3`) combined with norm and validation
    /// spot-check admission gates.
    ByzantineTrimmedMean {
        /// Number of extreme values trimmed from each tail when enough updates
        /// are present (`2 * trim_each_side < n`).
        trim_each_side: usize,
    },
    /// Unprotected arithmetic mean with validation/norm defenses disabled
    /// (used exclusively to demonstrate the negative adversarial regime).
    NaiveMeanUnprotected,
}

/// Outer Nesterov momentum and Byzantine defense configuration.
#[derive(Clone, Debug)]
pub struct OuterOptimizerConfig {
    /// Outer learning rate `eta_outer` (typically `0.7`).
    pub outer_lr: f32,
    /// Outer Nesterov momentum factor `mu` (typically `0.85`–`0.9`).
    pub nesterov_momentum: f32,
    /// Aggregation rule.
    pub aggregation_rule: AggregationRule,
    /// Minimum L2 norm required for a non-free-rider update.
    pub min_delta_norm: f32,
    /// Absolute ceiling on accepted pseudo-gradient L2 norm.
    pub max_delta_norm: f32,
    /// Maximum allowed ratio of a candidate update's norm to the round median.
    pub max_norm_median_ratio: f32,
    /// Maximum tolerated increase in spot-check validation loss when applying a
    /// candidate worker's pseudo-gradient individually.
    pub max_val_loss_regression: f64,
    /// Whether to run the held-out validation-loss spot-check gate.
    pub enable_loss_spot_check: bool,
}

impl Default for OuterOptimizerConfig {
    fn default() -> Self {
        Self {
            outer_lr: 0.7,
            nesterov_momentum: 0.7,
            aggregation_rule: AggregationRule::ByzantineTrimmedMean { trim_each_side: 1 },
            min_delta_norm: 1e-4,
            max_delta_norm: 25.0,
            max_norm_median_ratio: 3.5,
            max_val_loss_regression: 0.50,
            enable_loss_spot_check: true,
        }
    }
}

/// Structured reason for rejecting a worker's pseudo-gradient update.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RejectionReason {
    /// HMAC-SHA256 authentication failed on the worker receipt.
    InvalidHmac,
    /// Wire delta payload did not match `receipt.delta_sha256`.
    DeltaDigestMismatch,
    /// Receipt carried a stale or mismatched `round_id`.
    StaleRoundReplay,
    /// Receipt was computed against a diverged `base_checkpoint_sha256`.
    BaseCheckpointMismatch,
    /// Payload could not be decoded cleanly.
    MalformedPayload,
    /// Update contained NaN or Inf values.
    NonFiniteValues,
    /// Worker claimed inner steps with zero/negligible parameter change.
    FreeRiderZeroDelta,
    /// Update L2 norm exceeded absolute or median-relative outlier bounds.
    NormOutlier,
    /// Applying the candidate update regressed held-out spot-check loss beyond
    /// the declared tolerance (e.g., sign-flip or gradient-ascent poisoning).
    ValidationLossRegression,
}

impl RejectionReason {
    /// Canonical lowercase identifier for structured JSON telemetry.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InvalidHmac => "invalid_hmac",
            Self::DeltaDigestMismatch => "delta_digest_mismatch",
            Self::StaleRoundReplay => "stale_round_replay",
            Self::BaseCheckpointMismatch => "base_checkpoint_mismatch",
            Self::MalformedPayload => "malformed_payload",
            Self::NonFiniteValues => "non_finite_values",
            Self::FreeRiderZeroDelta => "free_rider_zero_delta",
            Self::NormOutlier => "norm_outlier",
            Self::ValidationLossRegression => "validation_loss_regression",
        }
    }
}

/// Candidate pseudo-gradient that already passed wire authentication and is
/// ready for statistical + validation spot-check screening.
#[derive(Clone, Debug)]
pub struct CandidateDelta {
    /// Submitting worker identifier.
    pub worker_id: String,
    /// Decoded FP32 pseudo-gradient `theta^(t) - theta_k^(t, H)`.
    pub delta: Vec<f32>,
    /// Claimed inner steps completed.
    pub inner_steps: u32,
}

/// Screening verdict for a single worker submission.
#[derive(Clone, Debug)]
pub struct CandidateVerdict {
    /// Worker identifier.
    pub worker_id: String,
    /// Whether the update was accepted into outer aggregation.
    pub accepted: bool,
    /// Rejection reason when `accepted == false`.
    pub rejection_reason: Option<RejectionReason>,
    /// Measured L2 norm of the decoded pseudo-gradient.
    pub l2_norm: f64,
    /// Spot-check validation loss delta (`cand_loss - base_loss`) when evaluated.
    pub spot_check_loss_delta: Option<f64>,
}

/// Stateful outer Nesterov momentum optimizer with Byzantine-robust screening.
#[derive(Clone, Debug)]
pub struct OuterNesterovAggregator {
    /// Outer optimization and Byzantine-filtering configuration.
    pub config: OuterOptimizerConfig,
    /// Outer Nesterov velocity buffer `v^(t)`.
    pub velocity: Vec<f32>,
}

impl OuterNesterovAggregator {
    /// Initializes zero velocity buffer for `param_count` parameters.
    #[must_use]
    pub fn new(param_count: usize, config: OuterOptimizerConfig) -> Self {
        Self {
            config,
            velocity: vec![0.0f32; param_count],
        }
    }

    /// Computes L2 norm of a vector in `f64` precision.
    #[must_use]
    pub fn l2_norm(v: &[f32]) -> f64 {
        let mut s = 0.0f64;
        for &x in v {
            s += (x as f64) * (x as f64);
        }
        s.sqrt()
    }

    /// Screens candidate worker deltas against non-finite values, free-rider
    /// zero deltas, norm outliers, and held-out validation-loss spot checks,
    /// then aggregates accepted deltas and returns `(aggregated_delta, verdicts)`.
    pub fn screen_and_aggregate(
        &self,
        base_model: &SmallZenLm,
        base_params: &[f32],
        spot_check_batch: &[&[u8]],
        candidates: &[CandidateDelta],
    ) -> (Option<Vec<f32>>, Vec<CandidateVerdict>) {
        if candidates.is_empty() {
            return (None, Vec::new());
        }

        let unprotected = matches!(
            self.config.aggregation_rule,
            AggregationRule::NaiveMeanUnprotected
        );

        let mut verdicts = Vec::with_capacity(candidates.len());
        let mut preliminary_indices = Vec::new();
        let mut prelim_norms = Vec::new();

        for (idx, cand) in candidates.iter().enumerate() {
            let finite = cand.delta.iter().all(|x| x.is_finite());
            let norm = if finite {
                Self::l2_norm(&cand.delta)
            } else {
                f64::NAN
            };

            if !finite {
                verdicts.push(CandidateVerdict {
                    worker_id: cand.worker_id.clone(),
                    accepted: false,
                    rejection_reason: Some(RejectionReason::NonFiniteValues),
                    l2_norm: norm,
                    spot_check_loss_delta: None,
                });
                continue;
            }

            if !unprotected && (norm < (self.config.min_delta_norm as f64) || cand.inner_steps == 0)
            {
                verdicts.push(CandidateVerdict {
                    worker_id: cand.worker_id.clone(),
                    accepted: false,
                    rejection_reason: Some(RejectionReason::FreeRiderZeroDelta),
                    l2_norm: norm,
                    spot_check_loss_delta: None,
                });
                continue;
            }

            if !unprotected && norm > (self.config.max_delta_norm as f64) {
                verdicts.push(CandidateVerdict {
                    worker_id: cand.worker_id.clone(),
                    accepted: false,
                    rejection_reason: Some(RejectionReason::NormOutlier),
                    l2_norm: norm,
                    spot_check_loss_delta: None,
                });
                continue;
            }

            preliminary_indices.push(idx);
            prelim_norms.push(norm);
            verdicts.push(CandidateVerdict {
                worker_id: cand.worker_id.clone(),
                accepted: true,
                rejection_reason: None,
                l2_norm: norm,
                spot_check_loss_delta: None,
            });
        }

        // Median-relative norm outlier gate when >= 3 updates survive initial checks.
        if !unprotected && prelim_norms.len() >= 3 {
            let mut sorted = prelim_norms.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let median = sorted[sorted.len() / 2].max(1e-6);
            let max_allowed = median * (self.config.max_norm_median_ratio as f64);
            for &idx in &preliminary_indices {
                if verdicts[idx].l2_norm > max_allowed {
                    verdicts[idx].accepted = false;
                    verdicts[idx].rejection_reason = Some(RejectionReason::NormOutlier);
                }
            }
        }

        // Held-out validation-loss spot-check gate catches sign-flip / gradient-ascent
        // poisoning even when norm matches honest updates!
        if !unprotected && self.config.enable_loss_spot_check && !spot_check_batch.is_empty() {
            let mut probe = base_model.clone();
            let _ = probe.load_lora_params(base_params);
            let base_loss = probe.eval_metrics(spot_check_batch).loss;

            let mut trial_params = vec![0.0f32; base_params.len()];
            for (idx, cand) in candidates.iter().enumerate() {
                if !verdicts[idx].accepted {
                    continue;
                }
                for i in 0..base_params.len() {
                    trial_params[i] = base_params[i] - cand.delta[i];
                }
                if probe.load_lora_params(&trial_params).is_ok() {
                    let cand_loss = probe.eval_metrics(spot_check_batch).loss;
                    let delta_loss = cand_loss - base_loss;
                    verdicts[idx].spot_check_loss_delta = Some(delta_loss);
                    if !cand_loss.is_finite() || delta_loss > self.config.max_val_loss_regression {
                        verdicts[idx].accepted = false;
                        verdicts[idx].rejection_reason =
                            Some(RejectionReason::ValidationLossRegression);
                    }
                }
            }
        }

        let accepted_deltas: Vec<&[f32]> = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                if verdicts[i].accepted {
                    Some(c.delta.as_slice())
                } else {
                    None
                }
            })
            .collect();

        if accepted_deltas.is_empty() {
            return (None, verdicts);
        }

        let dim = base_params.len();
        let mut agg = vec![0.0f32; dim];

        match self.config.aggregation_rule {
            AggregationRule::NaiveMeanUnprotected => {
                let n = accepted_deltas.len() as f32;
                for d_vec in &accepted_deltas {
                    for (a, &v) in agg.iter_mut().zip(d_vec.iter()) {
                        *a += v / n;
                    }
                }
            }
            AggregationRule::ByzantineTrimmedMean { trim_each_side } => {
                let n = accepted_deltas.len();
                let trim = if n >= 3 && 2 * trim_each_side < n {
                    trim_each_side
                } else {
                    0
                };
                let mut col = Vec::with_capacity(n);
                for coord in 0..dim {
                    col.clear();
                    for d_vec in &accepted_deltas {
                        col.push(d_vec[coord]);
                    }
                    if trim > 0 {
                        col.sort_by(|a, b| a.total_cmp(b));
                        let kept = &col[trim..n - trim];
                        let sum: f32 = kept.iter().copied().sum();
                        agg[coord] = sum / (kept.len() as f32);
                    } else {
                        let sum: f32 = col.iter().copied().sum();
                        agg[coord] = sum / (n as f32);
                    }
                }
            }
        }

        (Some(agg), verdicts)
    }

    /// Applies one outer Nesterov momentum step using the aggregated
    /// pseudo-gradient `agg_delta`:
    ///
    /// ```text
    /// v^(t) = mu * v^(t-1) + agg_delta
    /// theta^(t+1) = theta^(t) - outer_lr * (agg_delta + mu * v^(t))
    /// ```
    pub fn apply_outer_nesterov(&mut self, params: &mut [f32], agg_delta: &[f32]) {
        let mu = self.config.nesterov_momentum;
        let lr = self.config.outer_lr;
        for i in 0..params.len() {
            let v_new = mu * self.velocity[i] + agg_delta[i];
            self.velocity[i] = v_new;
            params[i] -= lr * (agg_delta[i] + mu * v_new);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::SplitMix64;

    #[test]
    fn int8_quantization_with_error_feedback_preserves_cumulative_sum() {
        let mut rng = SplitMix64::new(55);
        let dim = 256;
        let mut ef = vec![0.0f32; dim];
        let mut true_sum = vec![0.0f32; dim];
        let mut recon_sum = vec![0.0f32; dim];
        let mode = CompressionMode::Int8BlockSymmetric { block_size: 64 };

        for _ in 0..20 {
            let delta: Vec<f32> = (0..dim).map(|_| rng.next_gaussian_f32() * 0.01).collect();
            for i in 0..dim {
                true_sum[i] += delta[i];
            }
            let wire = encode_pseudo_gradient(&delta, mode, Some(&mut ef));
            // INT8 block-64 wire size must be >3.5x smaller than FP32.
            assert!(wire.len() * 3 < dim * 4);
            let dec = decode_pseudo_gradient(&wire, dim).expect("decode int8");
            for i in 0..dim {
                recon_sum[i] += dec[i];
            }
        }

        let err: f32 = true_sum
            .iter()
            .zip(recon_sum.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-3, "error-feedback drift too large: {err}");
    }
}
