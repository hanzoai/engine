// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! # `hanzo-detprofile` — Deterministic AI Execution Profiles (HIP-1350, `det.zen-kai.v1`)
//!
//! Clean-room, zero-dependency Apache-2.0 Rust crate implementing a canonical,
//! versioned deterministic execution profile for eligible Zen & Kai operators:
//!
//! - **Exact integer & dyadic fixed-point arithmetic** (`profile`, `ops`):
//!   - `Int8Gemm`, `PackedInt4Gemm`, `Int32BiasAdd`, `FixedPointReLU`,
//!     `FixedPointSiLU_LUT`, `FixedPointRMSNorm_Int64`,
//!     `ArgMax_DeterministicTieBreak`, and `TopK_DeterministicOrder`.
//!   - Rational dyadic scaling (`multiplier * 2^-shift`, zero floating-point
//!     scaling), explicit `RoundingMode` (`RoundHalfToEven`,
//!     `RoundHalfAwayFromZero`), explicit `OverflowMode` (`Saturating`,
//!     `CheckedError`), and explicit `ReductionOrder` (`LeftToRight`,
//!     `FixedBinaryTree`).
//! - **Explicit Unsupported Floating-Point Exclusion Gate**:
//!   - `Fp16MatMul`, `Bf16MatMul`, `Fp32ParallelReduce`, `AtomicFloatAdd`,
//!     `NonDeterministicSoftmax`, `UncappedTopPSampling`,
//!     `HardwareFmaDependentOp`, and `Fp8DynamicScaleGemm` are unconditionally
//!     rejected with [`DetError::UnsupportedFloatingPointOp`] and NEVER silently
//!     treated as deterministic.
//! - **Canonical serialization & SHA-256 commitments** (`wire`, `commit`):
//!   - Little-endian length-prefixed binary encoding and clean-room FIPS 180-4
//!     SHA-256 [`ExecutionReceipt`] commitments.
//! - **Three independent execution engines + explicit FP32 negative control**
//!   (`backends`):
//!   - [`ScalarCpuReference`], [`TiledSimdLikeCpuBackend`], and
//!     [`SimGpuWarpBackend`] verify bit-identical outputs and commitments across
//!     distinct reduction/tiling topologies.
//!   - [`UnorderedFloatGpuSim`] vs [`ScalarFloatRef`] empirically demonstrates
//!     IEEE-754 floating-point divergence when parallel reduction order changes.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod backends;
pub mod commit;
pub mod error;
pub mod ops;
pub mod profile;
pub mod wire;

pub use backends::{
    compare_float_outputs, f32_ulp_distance, DeterministicBackend, ExecutionResult,
    FloatDivergenceReport, FloatNegativeControlOutput, FloatWarpSchedule, ScalarCpuReference,
    ScalarFloatRef, SimGpuWarpBackend, TiledSimdLikeCpuBackend, UnorderedFloatGpuSim,
};
pub use commit::{
    commit_inputs, commit_op_graph, commit_output, commit_weights, ct_eq_digest, digest_to_hex,
    Digest, ExecutionReceipt, Sha256,
};
pub use error::{DetError, DetResult};
pub use ops::{
    fixed_point_silu_elem, requantize_i32_to_i8, ExecutionOutput, GraphInput, Int32Tensor,
    Int8Tensor, OpGraph, OpNode, PackedInt4Block, PackedInt4Weight, RankedEntry, TopKOutput,
    QUANT_BLOCK_SIZE, SIGMOID_LUT_Q15,
};
pub use profile::{
    isqrt_u128, narrow_i128_to_i32, narrow_i128_to_i64, rescale_i128, round_div_i128,
    round_shift_right_i128, validate_operator, DyadicScale, EligibleOp, ExecutionProfile,
    IneligibleFloatingPointOp, OperatorId, OverflowMode, ReductionOrder, RoundingMode,
    DEFAULT_RMSNORM_EPS_Q24, FIXED_POINT_FRAC_BITS, MAX_DYADIC_SHIFT, PROFILE_ID_V1,
    PROFILE_SCOPE_DISCLAIMER, PROFILE_VERSION_V1, SILU_LUT_VERSION_V1,
};
pub use wire::{WireReader, WireWriter, MAX_WIRE_ELEMENTS, WIRE_MAGIC};
