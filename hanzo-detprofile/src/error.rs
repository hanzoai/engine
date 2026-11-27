// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Typed, non-panicking error definitions for `hanzo-detprofile`.

use std::fmt;

/// Result alias for deterministic execution profile operations.
pub type DetResult<T> = Result<T, DetError>;

/// Errors produced during profile validation, deterministic execution, wire
/// decoding, or receipt verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetError {
    /// An ineligible floating-point operator was submitted to a deterministic
    /// profile. Floating-point operators are NEVER silently treated as
    /// deterministic.
    UnsupportedFloatingPointOp {
        /// Canonical operator identifier.
        op_name: &'static str,
        /// Explicit mathematical/hardware non-determinism rationale.
        reason: &'static str,
    },
    /// Profile identifier or version is unsupported by this crate.
    UnknownProfileVersion {
        /// Supplied profile identifier.
        id: String,
        /// Supplied profile version number.
        version: u32,
    },
    /// Tensor or operator dimensions are invalid or incompatible.
    InvalidDimensions {
        /// Operator or validation site.
        context: &'static str,
        /// Expected constraint description.
        expected: String,
        /// Actual observed value.
        actual: String,
    },
    /// Dyadic scale parameters are out of range (`shift > 62` or zero when
    /// invalid).
    InvalidScale {
        /// Operator or validation site.
        context: &'static str,
        /// Explanation of the invalid scale parameter.
        detail: String,
    },
    /// Arithmetic overflow occurred under `OverflowMode::CheckedError`.
    ArithmeticOverflow {
        /// Operator where overflow occurred.
        op: &'static str,
        /// Contextual detail.
        detail: String,
    },
    /// Division by zero or invalid zero denominator.
    DivisionByZero {
        /// Operator where division by zero was detected.
        op: &'static str,
    },
    /// Packed low-bit weight tensor failed structural validation.
    InvalidPackedWeight {
        /// Validation site.
        context: &'static str,
        /// Failure detail.
        detail: String,
    },
    /// Execution graph is empty, malformed, or has mismatched stage shapes.
    InvalidGraph {
        /// Failure detail.
        detail: String,
    },
    /// Canonical binary wire format decoding error.
    WireDecodeError {
        /// Byte offset where decoding failed.
        offset: usize,
        /// Failure reason.
        detail: &'static str,
    },
    /// Receipt commitment or digest verification failed.
    CommitmentMismatch {
        /// Name of the mismatched digest field.
        field: &'static str,
    },
}

impl fmt::Display for DetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFloatingPointOp { op_name, reason } => write!(
                f,
                "unsupported floating-point operator `{op_name}` explicitly excluded from deterministic profile: {reason}"
            ),
            Self::UnknownProfileVersion { id, version } => {
                write!(f, "unknown deterministic profile `{id}` version {version}")
            }
            Self::InvalidDimensions {
                context,
                expected,
                actual,
            } => write!(
                f,
                "invalid dimensions in {context}: expected {expected}, got {actual}"
            ),
            Self::InvalidScale { context, detail } => {
                write!(f, "invalid dyadic scale in {context}: {detail}")
            }
            Self::ArithmeticOverflow { op, detail } => {
                write!(f, "arithmetic overflow in {op} (CheckedError mode): {detail}")
            }
            Self::DivisionByZero { op } => write!(f, "division by zero in {op}"),
            Self::InvalidPackedWeight { context, detail } => {
                write!(f, "invalid packed weight in {context}: {detail}")
            }
            Self::InvalidGraph { detail } => write!(f, "invalid operator graph: {detail}"),
            Self::WireDecodeError { offset, detail } => {
                write!(f, "canonical wire decode error at byte {offset}: {detail}")
            }
            Self::CommitmentMismatch { field } => {
                write!(f, "execution receipt commitment mismatch on `{field}`")
            }
        }
    }
}

impl std::error::Error for DetError {}
