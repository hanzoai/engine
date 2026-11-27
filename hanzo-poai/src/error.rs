// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Typed error definitions for `hanzo-poai`.

use std::fmt;

/// Error returned by profile validation, quantized weight unpacking, proving, or verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoaiError {
    /// Requested operator is explicitly outside the deterministic `poai.matmul.v1` profile.
    IneligibleOperator {
        operator: String,
        reason: &'static str,
    },
    /// Matrix dimensions are zero or exceed safe arithmetic limits.
    InvalidDimensions {
        m: usize,
        k: usize,
        n: usize,
        reason: &'static str,
    },
    /// Matrix buffer length or shape does not match the declared dimensions.
    DimensionMismatch {
        matrix: &'static str,
        expected: (usize, usize),
        actual: (usize, usize),
    },
    /// Buffer length does not match `rows * cols`.
    BufferLengthMismatch {
        matrix: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Reduction dimension `k` is not a valid multiple of the quantization block size.
    InvalidBlockLayout {
        format: &'static str,
        expected_multiple: usize,
        actual_k: usize,
    },
    /// Packed weight payload has wrong length or contains reserved/illegal bit codes.
    MalformedPackedWeights {
        format: &'static str,
        reason: &'static str,
    },
    /// Freivalds round count is outside the permitted `[min, max]` range.
    InvalidChallengeRounds {
        rounds: usize,
        min: usize,
        max: usize,
    },
    /// An intermediate or final integer accumulator exceeded the safe field bijection range `|x| <= (p - 1)/2`.
    AccumulatorOutOfFieldRange {
        row: usize,
        col: usize,
        value: i128,
        max_abs: i128,
    },
    /// An entry in the claimed output matrix `C` exceeded the safe signed range `|c| <= (p - 1)/2`.
    OutputOutOfFieldRange {
        row: usize,
        col: usize,
        value: i64,
        max_abs: i64,
    },
    /// Cryptographic SHA-256 commitment in the proof receipt does not match the supplied matrix.
    CommitmentMismatch { matrix: &'static str },
    /// Proof receipt declares a profile version or operator incompatible with the verifier's profile.
    ProfileMismatch { expected: String, actual: String },
    /// Proof receipt declares a finite-field modulus other than `p = 2^61 - 1`.
    FieldModulusMismatch { expected: u64, actual: u64 },
    /// Proof receipt challenge mode or round count does not match verifier policy.
    ChallengeModeMismatch { reason: &'static str },
    /// Canonical binary proof receipt could not be parsed.
    InvalidProofEncoding { reason: &'static str },
    /// Freivalds probabilistic check detected `A * (B * r) != C * r (mod p)`.
    FreivaldsCheckFailed {
        round: usize,
        first_mismatch_row: usize,
    },
    /// Round projection digest in the receipt does not match the verified `C * r` projection.
    RoundDigestMismatch { round: usize },
    /// OS entropy source could not supply an interactive verifier seed.
    EntropyUnavailable { reason: String },
}

impl fmt::Display for PoaiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IneligibleOperator { operator, reason } => {
                write!(f, "ineligible operator '{operator}': {reason}")
            }
            Self::InvalidDimensions { m, k, n, reason } => {
                write!(f, "invalid matrix dimensions ({m}x{k}x{n}): {reason}")
            }
            Self::DimensionMismatch {
                matrix,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch for matrix {matrix}: expected {}x{}, got {}x{}",
                expected.0, expected.1, actual.0, actual.1
            ),
            Self::BufferLengthMismatch {
                matrix,
                expected,
                actual,
            } => write!(
                f,
                "buffer length mismatch for matrix {matrix}: expected {expected} elements, got {actual}"
            ),
            Self::InvalidBlockLayout {
                format,
                expected_multiple,
                actual_k,
            } => write!(
                f,
                "invalid block layout for {format}: k={actual_k} must be a non-zero multiple of {expected_multiple}"
            ),
            Self::MalformedPackedWeights { format, reason } => {
                write!(f, "malformed packed weights for {format}: {reason}")
            }
            Self::InvalidChallengeRounds { rounds, min, max } => {
                write!(
                    f,
                    "invalid Freivalds round count {rounds}: must be within [{min}, {max}]"
                )
            }
            Self::AccumulatorOutOfFieldRange {
                row,
                col,
                value,
                max_abs,
            } => write!(
                f,
                "accumulator overflow at ({row}, {col}): |{value}| exceeds safe GF(p) bijection bound {max_abs}"
            ),
            Self::OutputOutOfFieldRange {
                row,
                col,
                value,
                max_abs,
            } => write!(
                f,
                "output entry at ({row}, {col}) value {value} exceeds safe GF(p) signed bound +/-{max_abs}"
            ),
            Self::CommitmentMismatch { matrix } => {
                write!(
                    f,
                    "SHA-256 commitment mismatch for matrix {matrix} against proof receipt"
                )
            }
            Self::ProfileMismatch { expected, actual } => {
                write!(
                    f,
                    "verification profile mismatch: expected '{expected}', receipt declared '{actual}'"
                )
            }
            Self::FieldModulusMismatch { expected, actual } => {
                write!(
                    f,
                    "finite field modulus mismatch: expected p={expected}, receipt declared p={actual}"
                )
            }
            Self::ChallengeModeMismatch { reason } => {
                write!(f, "challenge mode mismatch: {reason}")
            }
            Self::InvalidProofEncoding { reason } => {
                write!(f, "invalid binary proof receipt encoding: {reason}")
            }
            Self::FreivaldsCheckFailed {
                round,
                first_mismatch_row,
            } => write!(
                f,
                "Freivalds verification failed in round {round} at row {first_mismatch_row}: A*(B*r) != C*r (mod p)"
            ),
            Self::RoundDigestMismatch { round } => {
                write!(
                    f,
                    "round {round} projection digest in receipt does not match computed C*r"
                )
            }
            Self::EntropyUnavailable { reason } => {
                write!(f, "interactive verifier entropy unavailable: {reason}")
            }
        }
    }
}

impl std::error::Error for PoaiError {}
