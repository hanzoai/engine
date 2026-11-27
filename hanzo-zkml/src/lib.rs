// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! # `hanzo-zkml` — Clean-Room zkML Circuits and Deterministic Register zkVM (HIP-1350)
//!
//! Pure-Rust, zero-dependency, `#![forbid(unsafe_code)]`, Apache-2.0 reference
//! implementation for `hanzoai/engine#68` (*"zkML and zkVM proofs for small models
//! and decisions"*).
//!
//! ## Strict Separation of Verification Tiers (HIP-1350)
//!
//! This crate explicitly separates the three verification mechanisms defined by
//! the decentralized Hanzo program and never conflates them:
//!
//! 1. **Cryptographic proofs** ([`TrustTier::CryptographicProof`]) — produced and
//!    verified by this crate over Goldilocks `F_p` (`p = 2^64 - 2^32 + 1`) and
//!    `F_{p^2}` using Spartan multilinear Sumcheck + Ligero Reed-Solomon Merkle
//!    commitments across two architectures:
//!    - **Path A ([`circuit`])**: Model-specific R1CS arithmetic circuits (EZKL-style)
//!      for compact quantized classifiers, Kai policy decision gates, fixed
//!      preprocessing, and deterministic evaluation scorers.
//!    - **Path B ([`zkvm`])**: General deterministic register zkVM (RISC Zero-style)
//!      executing compiled `ZkVmProgram` bytecode with `ImageID` binding, Merkle
//!      execution traces, and exact algebraic transition constraints.
//! 2. **Hardware attestations** ([`TrustTier::HardwareAttestation`]) — **NOT**
//!    assumed or produced by this crate; any commodity CPU can verify every proof
//!    without TEEs or vendor keys.
//! 3. **Economic / optimistic challenges** ([`TrustTier::EconomicChallenge`]) —
//!    **NOT** implemented by this crate; [`envelope::HanzoProofEnvelope`] provides
//!    canonical receipts that can be embedded as validity or fraud proofs inside
//!    external settlement games.
//!
//! ## Scope Boundary
//!
//! Full cryptographic proving of every token of an interactive 27B autoregressive
//! LLM decode is **explicitly out of scope and rejected as infeasible on CPU**;
//! see [`bench_harness::evaluate_27b_negative_regime`] for measured extrapolations.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod bench_harness;
pub mod circuit;
pub mod envelope;
pub mod field;
pub mod merkle;
pub mod sha256;
pub mod zkvm;

use core::fmt;

/// Protocol schema version for `hanzo-zkml` envelopes and receipts.
pub const PROTOCOL_VERSION: &str = "hanzo.zkml.v1";

/// Explicit HIP-1350 verification trust tier.
///
/// Never conflate cryptographic proofs, hardware attestations, and economically
/// incentivized challenges.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TrustTier {
    /// Algebraic / cryptographic proof verifiable statelessly without trusting
    /// prover hardware or Hanzo-operated infrastructure.
    CryptographicProof,
    /// Hardware TEE / GPU vendor attestation quote (NOT produced by `hanzo-zkml`).
    HardwareAttestation,
    /// Economically bonded optimistic dispute / slashing game (NOT produced by `hanzo-zkml`).
    EconomicChallenge,
}

impl TrustTier {
    /// Canonical stable string identifier for serialization.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CryptographicProof => "cryptographic_proof",
            Self::HardwareAttestation => "hardware_attestation",
            Self::EconomicChallenge => "economic_challenge",
        }
    }

    /// Parses a canonical trust tier identifier.
    pub fn from_str_canonical(s: &str) -> Option<Self> {
        match s {
            "cryptographic_proof" => Some(Self::CryptographicProof),
            "hardware_attestation" => Some(Self::HardwareAttestation),
            "economic_challenge" => Some(Self::EconomicChallenge),
            _ => None,
        }
    }
}

/// Unified error type returned by `hanzo-zkml` provers, verifiers, and envelope parsers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZkmlError {
    /// Signed integer exceeded the injective field range `[-(p-1)/2, (p-1)/2]`.
    FieldRangeExceeded {
        /// Offending value.
        value: i64,
    },
    /// Proof domain separation tag did not match the verifier's expected domain.
    DomainSeparationMismatch {
        /// Expected domain tag.
        expected: String,
        /// Actual domain tag in proof.
        actual: String,
    },
    /// Circuit identifier or semantic version mismatch.
    CircuitVersionMismatch {
        /// Expected identifier/version.
        expected: String,
        /// Actual identifier/version.
        actual: String,
    },
    /// Circuit constraint-system digest mismatch.
    CircuitDigestMismatch,
    /// Committed weights digest or Merkle root mismatch.
    WeightsCommitmentMismatch,
    /// Public input count or binding mismatch.
    PublicInputMismatch {
        /// Detail message.
        detail: String,
    },
    /// Public output count or binding mismatch.
    PublicOutputMismatch {
        /// Detail message.
        detail: String,
    },
    /// Spartan Sumcheck polynomial identity verification failed.
    SumcheckFailed {
        /// Stage where sumcheck failed.
        stage: &'static str,
    },
    /// Ligero Reed-Solomon Merkle PCS opening failed (Merkle or RS proximity/eval check).
    PcsVerificationFailed {
        /// Which commitment failed (`"weights"` or `"witness"`).
        slice: &'static str,
    },
    /// Local witness failed to satisfy R1CS constraints during proving.
    UnsatisfiedConstraint {
        /// Index of first failing R1CS constraint.
        constraint_index: usize,
    },
    /// zkVM `ImageID` mismatch (program bytecode, weights, version, or cycle bound changed).
    ImageIdMismatch,
    /// zkVM ISA version mismatch.
    VmVersionMismatch {
        /// Expected VM version.
        expected: String,
        /// Actual VM version.
        actual: String,
    },
    /// zkVM execution exceeded or violated the declared `max_cycles` bound.
    CycleBoundExceeded {
        /// Executed or claimed cycles.
        cycles: usize,
        /// Declared maximum cycles.
        max_cycles: usize,
    },
    /// zkVM execution trace Merkle root or transition check failed.
    TraceVerificationFailed {
        /// Detail message.
        detail: String,
    },
    /// zkVM runtime program error (invalid register, index, or missing `HALT`).
    VmExecutionError {
        /// Detail message.
        detail: String,
    },
    /// Envelope serialization, parsing, or binding verification error.
    EnvelopeError {
        /// Detail message.
        detail: String,
    },
}

impl fmt::Display for ZkmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldRangeExceeded { value } => {
                write!(f, "signed value {value} exceeds injective Goldilocks range")
            }
            Self::DomainSeparationMismatch { expected, actual } => {
                write!(
                    f,
                    "domain separation mismatch: expected '{expected}', got '{actual}'"
                )
            }
            Self::CircuitVersionMismatch { expected, actual } => {
                write!(
                    f,
                    "circuit version mismatch: expected '{expected}', got '{actual}'"
                )
            }
            Self::CircuitDigestMismatch => write!(f, "circuit digest mismatch"),
            Self::WeightsCommitmentMismatch => write!(f, "weights commitment mismatch"),
            Self::PublicInputMismatch { detail } => {
                write!(f, "public input mismatch: {detail}")
            }
            Self::PublicOutputMismatch { detail } => {
                write!(f, "public output mismatch: {detail}")
            }
            Self::SumcheckFailed { stage } => {
                write!(f, "sumcheck verification failed at stage '{stage}'")
            }
            Self::PcsVerificationFailed { slice } => {
                write!(f, "Ligero RS-Merkle PCS verification failed for '{slice}'")
            }
            Self::UnsatisfiedConstraint { constraint_index } => {
                write!(f, "R1CS constraint {constraint_index} is unsatisfied")
            }
            Self::ImageIdMismatch => write!(f, "zkVM ImageID mismatch"),
            Self::VmVersionMismatch { expected, actual } => {
                write!(
                    f,
                    "zkVM version mismatch: expected '{expected}', got '{actual}'"
                )
            }
            Self::CycleBoundExceeded { cycles, max_cycles } => {
                write!(f, "zkVM cycle bound exceeded: {cycles} > {max_cycles}")
            }
            Self::TraceVerificationFailed { detail } => {
                write!(f, "zkVM trace verification failed: {detail}")
            }
            Self::VmExecutionError { detail } => {
                write!(f, "zkVM execution error: {detail}")
            }
            Self::EnvelopeError { detail } => {
                write!(f, "HanzoProofEnvelope error: {detail}")
            }
        }
    }
}

impl std::error::Error for ZkmlError {}
