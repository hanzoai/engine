// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `hanzo-opai` — Optimistic AI Execution & Interactive $O(\log_2 T)$ Bisection
//! Fraud-Proof Games (HIP-1350, `hanzoai/engine#67`).
//!
//! This crate is a clean-room, Apache-2.0, zero-external-dependency Rust
//! implementation providing:
//!
//! 1. **Deterministic Fixed-Point Step-Trace VM** ([`vm`], [`profile`]):
//!    Quantized transformer/MLP inference over Q12 activations and int8 weights
//!    with `i128` accumulators, exact bitwise `isqrt_u128` RMSNorm, compile-time
//!    257-knot Q15 SiLU LUT, explicit floating-point exclusion gate, canonical
//!    per-step SHA-256 `StateRoot` commitments (`0..=T`), and Merkle trace roots.
//! 2. **Interactive Binary-Search Bisection Dispute Game** ([`game`]):
//!    Bonded [`Claim`] submission, permissionless challenge opening within
//!    `challenge_window_secs`, $O(\log_2 T)$ interactive bisection rounds,
//!    $O(1)$ deterministic single-step judge ([`verify_single_step`]), per-turn
//!    timeout forfeit & slashing, anti-griefing bond escalation, and concurrent
//!    multi-challenger tournament accounting ([`OptimisticCoordinator`]).
//! 3. **Sovereign Transcript Export & Independent Replay** ([`transcript`]):
//!    Lossless canonical binary wire (`OPAI` v1) and JSON codecs plus
//!    [`verify_transcript_sovereign`] enabling any independent third party to
//!    replay disputes and verify bond settlement with **0 Hanzo control-plane calls**.
//! 4. **HIP-1350 5-Axis Telemetry** ([`telemetry`]):
//!    Structured Cost, Performance, Trust, Decentralization, and Sovereignty
//!    accounting paired with the reproducible `opai-bench` harness.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod crypto;
pub mod error;
pub mod game;
pub mod profile;
pub mod telemetry;
pub mod transcript;
pub mod vm;

pub use crypto::{
    ct_eq_digest, digest_to_hex, hex_to_digest, Digest, MerkleProof, Sha256, SplitMix64,
    TraceMerkleTree, ZERO_DIGEST,
};
pub use error::OpAiError;
pub use game::{
    max_bisection_rounds, run_dispute_to_completion, ActorStrategy, BisectionRange, BondSettlement,
    BranchChoice, Claim, ClaimRecord, ClaimStatus, DisputeGame, DisputeMove, DisputePhase,
    DisputeResolution, GameConfig, OptimisticCoordinator, Party, ResolutionReason,
};
pub use profile::{
    add_i32, fixed_point_silu_q12, isqrt_u128, narrow_i32, rescale_i128, round_div_i128,
    DyadicScale, ExecutionProfile, InputCommitment, NumericDomain, OverflowMode, RoundingMode,
    CANONICAL_PROFILE_ID, CANONICAL_PROFILE_VERSION, SIGMOID_LUT_Q15,
};
pub use telemetry::{
    ns_to_usd_per_1m_tokens, CostAxis, DecentralizationAxis, PerformanceAxis, RegimeTelemetry,
    SovereigntyAxis, TrustAxis, REFERENCE_VCPU_USD_PER_HOUR,
};
pub use transcript::{
    verify_transcript_sovereign, DisputeTranscript, SovereignReplayReport, WIRE_MAGIC, WIRE_VERSION,
};
pub use vm::{
    execute_single_step, execute_trace, execute_trace_with_fault, verify_single_step,
    verify_single_step_with_ctx, ExecutionTrace, FaultSpec, JudgeContext, QuantizedLayer,
    QuantizedModel, SingleStepVerdict, StepOp, VmState, BASE_STEPS_PER_TOKEN, STEPS_PER_LAYER,
};
