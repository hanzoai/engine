// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Versioned Proof-of-AI execution profile (`poai.matmul.v1`) and explicit scope limits.
//!
//! Records exactly which packed low-bit integer matrix-product operators are eligible for
//! Freivalds verification over `GF(2^61 - 1)`, explicitly rejects non-deterministic floating-point
//! GEMMs and non-matmul LLM operations, and states the protocol boundary required by
//! `hanzoai/engine#66` and `zoo-apps/papers#13`.

use crate::error::PoaiError;
use crate::field::{MAX_SAFE_SIGNED, MODULUS_P};
use crate::quant::{PackedQuantFormat, QUANT_BLOCK_SIZE};

/// Canonical version identifier for the deterministic packed integer matrix-product profile.
pub const PROFILE_VERSION_V1: &str = "poai.matmul.v1";

/// Default number of independent Freivalds rounds (`soundness error < 1 / p^2 < 2^-121`).
pub const DEFAULT_FREIVALDS_ROUNDS: usize = 2;

/// Minimum allowed Freivalds rounds (`soundness error < 1 / p ~= 4.34e-19`).
pub const MIN_FREIVALDS_ROUNDS: usize = 1;

/// Maximum allowed Freivalds rounds (`soundness error < 1 / p^4 < 2^-243`).
pub const MAX_FREIVALDS_ROUNDS: usize = 4;

/// Mandatory scope boundary disclaimer (HIP-1350 / `hanzoai/engine#66` / `zoo-apps/papers#13`).
pub const PROOF_SCOPE_DISCLAIMER: &str =
    "A matrix-product proof verifies only the declared integer matrix product C = A * B \
     under profile poai.matmul.v1; attention, normalization (RMSNorm/LayerNorm), softmax, \
     sampling, KV-cache/recurrent state updates, and overall LLM response truthfulness or \
     scientific value are separate operations and are NOT covered by this proof.";

/// Operator classes considered by the execution profile gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OperatorKind {
    /// Deterministic `i8` activation x `Int8Q8_0` (`i16`-scaled 32-weight `i8`) GEMM.
    Int8Gemm = 1,
    /// Deterministic `i8` activation x `PackedInt4Q4_0` (`i16`-scaled 32-weight 4-bit nibble) GEMM.
    PackedInt4BlockGemm = 2,
    /// Deterministic `i8` activation x `PackedTernaryI2S` (custom `poai.matmul.v1` `i16`-scaled 32-weight 2-bit ternary; distinct from GGUF `TQ2_0`) GEMM.
    PackedTernaryI2SGemm = 3,
    /// Ineligible: IEEE-754 FP16 GEMM (non-associative across warp/tile reduction orders).
    Fp16Gemm = 100,
    /// Ineligible: BFloat16 GEMM (non-associative across hardware reduction orders).
    Bf16Gemm = 101,
    /// Ineligible: IEEE-754 FP32 GEMM (non-associative FMA / reduction scheduling).
    Fp32Gemm = 102,
    /// Ineligible: FP8 E4M3/E5M2 GEMM (non-deterministic float accumulation).
    Fp8Gemm = 103,
    /// Ineligible: Attention / FlashAttention (separate non-linear softmax + scaling operator).
    FlashAttention = 110,
    /// Ineligible: RMSNorm / LayerNorm (floating-point variance/rsqrt outside matmul boundary).
    Normalization = 111,
    /// Ineligible: Token sampling / temperature / top-p / top-k (stochastic outside matmul boundary).
    Sampling = 112,
    /// Ineligible: KV cache or recurrent Gated DeltaNet state updates.
    StateUpdate = 113,
}

impl OperatorKind {
    /// Stable wire tag used in transcripts and binary receipts.
    #[inline]
    pub const fn wire_tag(self) -> u8 {
        self as u8
    }

    /// Decode an eligible operator from its wire tag.
    pub fn from_eligible_wire_tag(tag: u8) -> Result<Self, PoaiError> {
        match tag {
            1 => Ok(Self::Int8Gemm),
            2 => Ok(Self::PackedInt4BlockGemm),
            3 => Ok(Self::PackedTernaryI2SGemm),
            _ => Err(PoaiError::InvalidProofEncoding {
                reason: "unknown or ineligible operator wire tag",
            }),
        }
    }

    /// Human-readable canonical operator name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Int8Gemm => "Int8Gemm",
            Self::PackedInt4BlockGemm => "PackedInt4BlockGemm",
            Self::PackedTernaryI2SGemm => "PackedTernaryI2SGemm",
            Self::Fp16Gemm => "Fp16Gemm",
            Self::Bf16Gemm => "Bf16Gemm",
            Self::Fp32Gemm => "Fp32Gemm",
            Self::Fp8Gemm => "Fp8Gemm",
            Self::FlashAttention => "FlashAttention",
            Self::Normalization => "Normalization",
            Self::Sampling => "Sampling",
            Self::StateUpdate => "StateUpdate",
        }
    }

    /// Corresponding `hanzo-quant` packed weight layout for eligible operators.
    pub const fn packed_format(self) -> Option<PackedQuantFormat> {
        match self {
            Self::Int8Gemm => Some(PackedQuantFormat::Int8Q8_0),
            Self::PackedInt4BlockGemm => Some(PackedQuantFormat::PackedInt4Q4_0),
            Self::PackedTernaryI2SGemm => Some(PackedQuantFormat::PackedTernaryI2S),
            _ => None,
        }
    }

    /// Map a packed quantization format to its eligible operator kind.
    pub const fn from_packed_format(format: PackedQuantFormat) -> Self {
        match format {
            PackedQuantFormat::Int8Q8_0 => Self::Int8Gemm,
            PackedQuantFormat::PackedInt4Q4_0 => Self::PackedInt4BlockGemm,
            PackedQuantFormat::PackedTernaryI2S => Self::PackedTernaryI2SGemm,
        }
    }
}

/// Versioned verification profile declaring accepted numerical semantics and operator eligibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationProfile {
    /// Versioned profile string (`poai.matmul.v1`).
    pub version: &'static str,
    /// Explicit prime field modulus (`p = 2^61 - 1`).
    pub field_modulus_p: u64,
    /// Maximum safe signed integer accumulator magnitude (`(p - 1)/2 = 2^60 - 1`).
    pub max_safe_abs_accumulator: i64,
    /// Quantization block size along reduction dimension `K` (`32` weights per `i16`-scaled block in `poai.matmul.v1`).
    pub block_size: usize,
    /// Default number of Freivalds rounds (`2` -> soundness error `< 1/p^2 < 2^-121`).
    pub default_rounds: usize,
    /// Explicit scope disclaimer required by `hanzoai/engine#66`.
    pub scope_disclaimer: &'static str,
}

impl VerificationProfile {
    /// Canonical `poai.matmul.v1` execution profile.
    pub const fn v1() -> Self {
        Self {
            version: PROFILE_VERSION_V1,
            field_modulus_p: MODULUS_P,
            max_safe_abs_accumulator: MAX_SAFE_SIGNED,
            block_size: QUANT_BLOCK_SIZE,
            default_rounds: DEFAULT_FREIVALDS_ROUNDS,
            scope_disclaimer: PROOF_SCOPE_DISCLAIMER,
        }
    }

    /// All operators eligible under this profile.
    pub const fn eligible_operators(&self) -> &'static [OperatorKind] {
        &[
            OperatorKind::Int8Gemm,
            OperatorKind::PackedInt4BlockGemm,
            OperatorKind::PackedTernaryI2SGemm,
        ]
    }

    /// Validate that an operator is explicitly eligible under `poai.matmul.v1`.
    ///
    /// Rejects all floating-point GEMMs and non-matmul LLM pipeline stages with a structured
    /// [`PoaiError::IneligibleOperator`].
    pub fn ensure_eligible(&self, op: OperatorKind) -> Result<(), PoaiError> {
        match op {
            OperatorKind::Int8Gemm
            | OperatorKind::PackedInt4BlockGemm
            | OperatorKind::PackedTernaryI2SGemm => Ok(()),
            OperatorKind::Fp16Gemm
            | OperatorKind::Bf16Gemm
            | OperatorKind::Fp32Gemm
            | OperatorKind::Fp8Gemm => Err(PoaiError::IneligibleOperator {
                operator: op.as_str().to_string(),
                reason: "floating-point matrix products are non-associative across GPU/CPU reduction schedules and are ineligible under poai.matmul.v1",
            }),
            OperatorKind::FlashAttention => Err(PoaiError::IneligibleOperator {
                operator: op.as_str().to_string(),
                reason: "attention is a separate non-linear operation not covered by a matrix-product proof",
            }),
            OperatorKind::Normalization => Err(PoaiError::IneligibleOperator {
                operator: op.as_str().to_string(),
                reason: "normalization (RMSNorm/LayerNorm) is outside the poai.matmul.v1 matrix-product boundary",
            }),
            OperatorKind::Sampling => Err(PoaiError::IneligibleOperator {
                operator: op.as_str().to_string(),
                reason: "token sampling is outside the poai.matmul.v1 matrix-product boundary",
            }),
            OperatorKind::StateUpdate => Err(PoaiError::IneligibleOperator {
                operator: op.as_str().to_string(),
                reason: "KV-cache and recurrent state updates are outside the poai.matmul.v1 matrix-product boundary",
            }),
        }
    }

    /// Validate Freivalds round count against global profile bounds `[MIN_FREIVALDS_ROUNDS, MAX_FREIVALDS_ROUNDS]`.
    pub fn validate_rounds(&self, rounds: usize) -> Result<(), PoaiError> {
        if !(MIN_FREIVALDS_ROUNDS..=MAX_FREIVALDS_ROUNDS).contains(&rounds) {
            return Err(PoaiError::InvalidChallengeRounds {
                rounds,
                min: MIN_FREIVALDS_ROUNDS,
                max: MAX_FREIVALDS_ROUNDS,
            });
        }
        Ok(())
    }

    /// Validate that a receipt's Freivalds round count satisfies a verifier-owned minimum round
    /// requirement `min_rounds` within `[MIN_FREIVALDS_ROUNDS, MAX_FREIVALDS_ROUNDS]`.
    pub fn validate_rounds_at_least(
        &self,
        rounds: usize,
        min_rounds: usize,
    ) -> Result<(), PoaiError> {
        self.validate_rounds(min_rounds)?;
        if !(min_rounds..=MAX_FREIVALDS_ROUNDS).contains(&rounds) {
            return Err(PoaiError::InvalidChallengeRounds {
                rounds,
                min: min_rounds,
                max: MAX_FREIVALDS_ROUNDS,
            });
        }
        Ok(())
    }

    /// Upper bound on cheating-prover acceptance probability `<= p^(-rounds)` for `rounds` independent
    /// uniform non-zero challenge vectors `r in GF(p)^n \ {0}`.
    ///
    /// Specifically, for any non-zero error matrix `D = A*B - C (mod p)` of width `n`,
    /// `Pr[D*r = 0 | r != 0] <= (p^(n-1) - 1) / (p^n - 1) < 1/p`.
    /// Across `k` independent rounds, the error bound is strictly `< p^(-k)`.
    pub fn soundness_error_upper_bound(&self, rounds: usize) -> f64 {
        (self.field_modulus_p as f64).powi(-(rounds as i32))
    }

    /// Soundness security bits `-log2(p^(-rounds)) = rounds * log2(2^61 - 1)`.
    pub fn soundness_bits(&self, rounds: usize) -> f64 {
        (rounds as f64) * (self.field_modulus_p as f64).log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_accepts_eligible_and_rejects_ineligible_operators() {
        let profile = VerificationProfile::v1();
        assert_eq!(profile.version, "poai.matmul.v1");
        assert_eq!(profile.field_modulus_p, MODULUS_P);
        assert!(profile.scope_disclaimer.contains("attention"));
        assert!(profile.scope_disclaimer.contains("normalization"));
        assert!(profile.scope_disclaimer.contains("sampling"));

        for &op in profile.eligible_operators() {
            assert!(profile.ensure_eligible(op).is_ok());
            assert!(op.packed_format().is_some());
        }

        let ineligible = [
            OperatorKind::Fp16Gemm,
            OperatorKind::Bf16Gemm,
            OperatorKind::Fp32Gemm,
            OperatorKind::Fp8Gemm,
            OperatorKind::FlashAttention,
            OperatorKind::Normalization,
            OperatorKind::Sampling,
            OperatorKind::StateUpdate,
        ];
        for op in ineligible {
            assert!(matches!(
                profile.ensure_eligible(op),
                Err(PoaiError::IneligibleOperator { .. })
            ));
            assert!(op.packed_format().is_none());
        }
    }

    #[test]
    fn soundness_bounds_match_spec() {
        let profile = VerificationProfile::v1();
        let eps1 = profile.soundness_error_upper_bound(1);
        let eps2 = profile.soundness_error_upper_bound(2);
        let eps4 = profile.soundness_error_upper_bound(4);
        assert!(eps1 < 4.34e-19 && eps1 > 4.33e-19);
        assert!(eps2 < 2.0f64.powi(-121));
        assert!(eps4 < 2.0f64.powi(-243));
        assert!(profile.soundness_bits(2) > 121.99);
        assert!(profile.validate_rounds(0).is_err());
        assert!(profile.validate_rounds(5).is_err());
        assert!(profile.validate_rounds(2).is_ok());
    }
}
