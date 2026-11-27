// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Typed errors for `hanzo-mixed-quant`.

use std::fmt;

/// Error type returned by quantization, rotation, calibration, planning, and inference operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MixedQuantError {
    /// Matrix or vector shape mismatch or invalid zero dimension.
    InvalidShape {
        /// Context describing the failing operation.
        context: &'static str,
        /// Expected size or invariant.
        expected: usize,
        /// Observed size.
        actual: usize,
    },
    /// Invalid group size (must be > 0 and divide the inner dimension where required).
    InvalidGroupSize {
        /// Group size provided by caller.
        group_size: usize,
        /// Matrix inner dimension.
        cols: usize,
    },
    /// Dimension is not a power of two or not divisible by the Hadamard block size.
    InvalidRotationDim {
        /// Observed dimension.
        dim: usize,
        /// Explanation of constraint.
        reason: &'static str,
    },
    /// Singular or ill-conditioned linear system encountered during Cayley transform inversion.
    SingularMatrix {
        /// Pivot index where singularity was detected.
        pivot_index: usize,
    },
    /// Calibration batch is empty or contains non-finite activations.
    InvalidCalibrationData {
        /// Explanation of invalid calibration input.
        reason: &'static str,
    },
    /// Requested memory budget cannot satisfy mandatory precision floors.
    InfeasibleMemoryBudget {
        /// Requested target byte budget.
        budget_bytes: usize,
        /// Minimum feasible bytes under component floors.
        min_feasible_bytes: usize,
    },
    /// Underlying corpus or artifact I/O failure.
    IoError {
        /// Path or resource name.
        path: String,
        /// Human-readable error message.
        message: String,
    },
    /// Acceptance gate rejected a configuration due to quality or end-to-end latency regression.
    AcceptanceGateRejected {
        /// Human-readable reason for rejection.
        reason: String,
    },
}

impl fmt::Display for MixedQuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape {
                context,
                expected,
                actual,
            } => write!(
                f,
                "invalid shape in {context}: expected {expected}, got {actual}"
            ),
            Self::InvalidGroupSize { group_size, cols } => write!(
                f,
                "invalid group_size {group_size} for inner dimension {cols}"
            ),
            Self::InvalidRotationDim { dim, reason } => {
                write!(f, "invalid rotation dimension {dim}: {reason}")
            }
            Self::SingularMatrix { pivot_index } => {
                write!(f, "singular matrix at pivot index {pivot_index}")
            }
            Self::InvalidCalibrationData { reason } => {
                write!(f, "invalid calibration data: {reason}")
            }
            Self::InfeasibleMemoryBudget {
                budget_bytes,
                min_feasible_bytes,
            } => write!(
                f,
                "infeasible memory budget: target {budget_bytes} bytes < minimum feasible {min_feasible_bytes} bytes"
            ),
            Self::IoError { path, message } => write!(f, "I/O error for '{path}': {message}"),
            Self::AcceptanceGateRejected { reason } => {
                write!(f, "acceptance gate rejected: {reason}")
            }
        }
    }
}

impl std::error::Error for MixedQuantError {}
