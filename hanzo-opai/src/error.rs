// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the `hanzo-opai` optimistic AI execution and fraud-proof engine.

use std::fmt;

/// Unified error type for `hanzo-opai` execution, bisection games, wire codecs,
/// and sovereign transcript verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpAiError {
    /// A floating-point or non-deterministic execution configuration was requested.
    UnsupportedFloatingPointProfile {
        /// Human-readable description of the rejected floating-point feature.
        reason: String,
    },
    /// The execution profile parameters violate deterministic bounds.
    InvalidProfile(String),
    /// The model dimensions, scales, or weights are malformed.
    InvalidModel(String),
    /// The input commitment or token sequence is invalid.
    InvalidInput(String),
    /// A step index is outside the valid execution range `0..total_steps`.
    InvalidStepIndex {
        /// Requested step index.
        step: u64,
        /// Total steps in the execution schedule.
        total_steps: u64,
    },
    /// Fixed-point integer arithmetic overflow under `OverflowMode::CheckedError`
    /// or division by zero.
    ArithmeticOverflow(&'static str),
    /// A submitted claim violates structural or cryptographic invariants.
    InvalidClaim(String),
    /// A duplicate claim ID was submitted to the coordinator.
    DuplicateClaim,
    /// The claim was not found in the coordinator state.
    ClaimNotFound,
    /// The claim has already reached terminal settlement (finalized or slashed).
    ClaimAlreadySettled,
    /// The challenge window for the claim has already closed (or not yet elapsed for finalization).
    ChallengeWindowViolation(String),
    /// The posted bond is below the required minimum (including anti-griefing escalation).
    InsufficientBond {
        /// Minimum required bond in micro-units.
        required: u64,
        /// Actual bond supplied in micro-units.
        provided: u64,
    },
    /// The dispute ID was not found.
    DisputeNotFound,
    /// The dispute has already been resolved.
    DisputeAlreadyResolved,
    /// A party attempted a move out of turn or on an unexpected game phase.
    WrongTurn(String),
    /// A timestamp moved backwards or exceeded the active turn deadline.
    DeadlineViolation(String),
    /// A timeout forfeit was requested before the active turn deadline elapsed.
    TurnNotExpired {
        /// Deadline timestamp in seconds.
        deadline_secs: u64,
        /// Observed timestamp in seconds.
        observed_secs: u64,
    },
    /// The bisection midpoint did not match `lo + (hi - lo) / 2`.
    InvalidBisectionMidpoint {
        /// Expected canonical midpoint.
        expected: u64,
        /// Supplied midpoint.
        got: u64,
    },
    /// Single-step judging was invoked before the bisection interval converged to `hi == lo + 1`.
    NotReadyForSingleStep {
        /// Current lower bound.
        lo: u64,
        /// Current upper bound.
        hi: u64,
    },
    /// The maximum number of concurrent challenges against a single claim was reached.
    MaxConcurrentChallengesExceeded {
        /// Configured concurrency cap per claim.
        limit: usize,
    },
    /// The same challenger already has an active dispute against this claim.
    ChallengerAlreadyActive(String),
    /// Binary wire or JSON encoding/decoding failed.
    WireFormat(String),
    /// Transcript digest or structural integrity check failed.
    TranscriptIntegrity(String),
    /// Sovereign third-party replay disagreed with the recorded transcript outcome or settlement.
    ReplayMismatch(String),
}

impl fmt::Display for OpAiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFloatingPointProfile { reason } => {
                write!(f, "unsupported floating-point profile: {reason}")
            }
            Self::InvalidProfile(msg) => write!(f, "invalid execution profile: {msg}"),
            Self::InvalidModel(msg) => write!(f, "invalid quantized model: {msg}"),
            Self::InvalidInput(msg) => write!(f, "invalid input commitment: {msg}"),
            Self::InvalidStepIndex { step, total_steps } => {
                write!(
                    f,
                    "step index {step} out of bounds for total_steps={total_steps}"
                )
            }
            Self::ArithmeticOverflow(op) => write!(f, "deterministic arithmetic overflow in {op}"),
            Self::InvalidClaim(msg) => write!(f, "invalid claim: {msg}"),
            Self::DuplicateClaim => write!(f, "duplicate claim ID"),
            Self::ClaimNotFound => write!(f, "claim not found"),
            Self::ClaimAlreadySettled => write!(f, "claim already settled"),
            Self::ChallengeWindowViolation(msg) => write!(f, "challenge window violation: {msg}"),
            Self::InsufficientBond { required, provided } => {
                write!(
                    f,
                    "insufficient bond: required {required}, provided {provided}"
                )
            }
            Self::DisputeNotFound => write!(f, "dispute not found"),
            Self::DisputeAlreadyResolved => write!(f, "dispute already resolved"),
            Self::WrongTurn(msg) => write!(f, "wrong turn: {msg}"),
            Self::DeadlineViolation(msg) => write!(f, "deadline violation: {msg}"),
            Self::TurnNotExpired {
                deadline_secs,
                observed_secs,
            } => write!(
                f,
                "turn has not expired: deadline={deadline_secs}s, observed={observed_secs}s"
            ),
            Self::InvalidBisectionMidpoint { expected, got } => {
                write!(
                    f,
                    "invalid bisection midpoint: expected {expected}, got {got}"
                )
            }
            Self::NotReadyForSingleStep { lo, hi } => {
                write!(
                    f,
                    "bisection not converged for single-step judge: [{lo}, {hi}]"
                )
            }
            Self::MaxConcurrentChallengesExceeded { limit } => {
                write!(
                    f,
                    "max concurrent challenges per claim exceeded (limit={limit})"
                )
            }
            Self::ChallengerAlreadyActive(id) => {
                write!(
                    f,
                    "challenger '{id}' already has an active dispute on this claim"
                )
            }
            Self::WireFormat(msg) => write!(f, "wire format error: {msg}"),
            Self::TranscriptIntegrity(msg) => write!(f, "transcript integrity failure: {msg}"),
            Self::ReplayMismatch(msg) => write!(f, "sovereign replay mismatch: {msg}"),
        }
    }
}

impl std::error::Error for OpAiError {}
