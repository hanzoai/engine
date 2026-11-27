// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Versioned canonical deterministic execution profile (`det.zen-kai.v1`),
//! exact dyadic fixed-point arithmetic, rounding/overflow policies, reduction
//! ordering, and explicit unsupported floating-point operator exclusion gate.

use crate::error::{DetError, DetResult};

/// Canonical profile identifier for HIP-1350 v1 deterministic Zen/Kai execution.
pub const PROFILE_ID_V1: &str = "det.zen-kai.v1";

/// Canonical profile major version number.
pub const PROFILE_VERSION_V1: u32 = 1;

/// Canonical fixed-point SiLU LUT revision bound into `det.zen-kai.v1`.
pub const SILU_LUT_VERSION_V1: u32 = 1;

/// Default integer RMSNorm epsilon in Q24 fixed-point units (`1 / 2^24`).
pub const DEFAULT_RMSNORM_EPS_Q24: u64 = 1;

/// Fixed-point fractional bits used by `FixedPointSiLU_LUT` and
/// `FixedPointRMSNorm_Int64` (`Q12`: `1.0 == 4096`).
pub const FIXED_POINT_FRAC_BITS: u8 = 12;

/// Maximum permitted dyadic right-shift (`2^-62`).
pub const MAX_DYADIC_SHIFT: u8 = 62;

/// Explicit scope disclaimer documenting what `det.zen-kai.v1` guarantees and
/// what remains outside its scope.
pub const PROFILE_SCOPE_DISCLAIMER: &str = "\
Deterministic execution profile det.zen-kai.v1 guarantees bit-identical outputs \
and SHA-256 commitments across independent CPU, SIMD-tiled, and GPU-warp \
implementations ONLY for the 8 declared integer/dyadic fixed-point operators. \
Floating-point kernels (FP16/BF16/FP32/FP8 matmuls, parallel float reductions, \
atomic float adds, non-deterministic softmax, uncapped top-p sampling, and \
hardware-FMA-dependent ops) are explicitly rejected and NEVER silently treated \
as deterministic. Deterministic execution is a reproducibility guarantee, not a \
hardware attestation, semantic truthfulness proof, or audited settlement contract.";

/// Explicit rounding policy applied to exact dyadic scaling and integer division.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RoundingMode {
    /// Banker's rounding: exact half-way ties round toward the nearest even integer.
    RoundHalfToEven = 1,
    /// Symmetric commercial rounding: exact half-way ties round away from zero
    /// (`+0.5 -> +1`, `-0.5 -> -1`).
    RoundHalfAwayFromZero = 2,
}

impl RoundingMode {
    /// Canonical byte tag for wire encoding and SHA-256 commitments.
    #[inline]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Decode a canonical byte tag.
    pub fn from_u8(v: u8, offset: usize) -> DetResult<Self> {
        match v {
            1 => Ok(Self::RoundHalfToEven),
            2 => Ok(Self::RoundHalfAwayFromZero),
            _ => Err(DetError::WireDecodeError {
                offset,
                detail: "invalid RoundingMode tag",
            }),
        }
    }

    /// Canonical human-readable name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RoundHalfToEven => "round_half_to_even",
            Self::RoundHalfAwayFromZero => "round_half_away_from_zero",
        }
    }
}

/// Explicit integer overflow policy enforced by all `det.zen-kai.v1` operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OverflowMode {
    /// Clamp out-of-range results to `[T::MIN, T::MAX]`.
    Saturating = 1,
    /// Abort execution immediately with `DetError::ArithmeticOverflow`.
    CheckedError = 2,
}

impl OverflowMode {
    /// Canonical byte tag for wire encoding and SHA-256 commitments.
    #[inline]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Decode a canonical byte tag.
    pub fn from_u8(v: u8, offset: usize) -> DetResult<Self> {
        match v {
            1 => Ok(Self::Saturating),
            2 => Ok(Self::CheckedError),
            _ => Err(DetError::WireDecodeError {
                offset,
                detail: "invalid OverflowMode tag",
            }),
        }
    }

    /// Canonical human-readable name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Saturating => "saturating",
            Self::CheckedError => "checked_error",
        }
    }
}

/// Canonical reduction topology declared in the execution profile.
///
/// For exact `i64`/`i128` integer accumulations without intermediate narrowing,
/// both `LeftToRight` and `FixedBinaryTree` are algebraically associative and
/// yield bit-identical sums; the profile binds the declared order into the
/// commitment so verifiers audit the exact protocol configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ReductionOrder {
    /// Strict sequential left-to-right index order `0, 1, ..., n - 1`.
    LeftToRight = 1,
    /// Fixed deterministic width-32 balanced binary reduction tree.
    FixedBinaryTree = 2,
}

impl ReductionOrder {
    /// Canonical byte tag for wire encoding and SHA-256 commitments.
    #[inline]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Decode a canonical byte tag.
    pub fn from_u8(v: u8, offset: usize) -> DetResult<Self> {
        match v {
            1 => Ok(Self::LeftToRight),
            2 => Ok(Self::FixedBinaryTree),
            _ => Err(DetError::WireDecodeError {
                offset,
                detail: "invalid ReductionOrder tag",
            }),
        }
    }

    /// Canonical human-readable name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LeftToRight => "left_to_right",
            Self::FixedBinaryTree => "fixed_binary_tree",
        }
    }
}

/// Exact rational dyadic quantization scale `multiplier * 2^(-shift)`.
///
/// Uses zero floating-point arithmetic. `shift` is bounded to `0..=62`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DyadicScale {
    /// Signed integer numerator multiplier.
    pub multiplier: i32,
    /// Right-shift exponent `shift` in `0..=62` representing division by `2^shift`.
    pub shift: u8,
}

impl DyadicScale {
    /// Construct and validate a dyadic scale `multiplier * 2^(-shift)`.
    pub fn new(multiplier: i32, shift: u8) -> DetResult<Self> {
        if shift > MAX_DYADIC_SHIFT {
            return Err(DetError::InvalidScale {
                context: "DyadicScale::new",
                detail: format!("shift {shift} exceeds MAX_DYADIC_SHIFT ({MAX_DYADIC_SHIFT})"),
            });
        }
        Ok(Self { multiplier, shift })
    }

    /// Identity dyadic scale `1 * 2^0 = 1`.
    pub const fn identity() -> Self {
        Self {
            multiplier: 1,
            shift: 0,
        }
    }

    /// Validate that this scale is well-formed.
    pub fn validate(&self) -> DetResult<()> {
        if self.shift > MAX_DYADIC_SHIFT {
            return Err(DetError::InvalidScale {
                context: "DyadicScale::validate",
                detail: format!(
                    "shift {} exceeds MAX_DYADIC_SHIFT ({MAX_DYADIC_SHIFT})",
                    self.shift
                ),
            });
        }
        Ok(())
    }
}

/// Canonical versioned execution profile (`det.zen-kai.v1`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExecutionProfile {
    /// Canonical profile identifier (must equal `PROFILE_ID_V1`).
    pub profile_id: String,
    /// Canonical profile version (must equal `PROFILE_VERSION_V1`).
    pub profile_version: u32,
    /// Rounding policy for dyadic scaling and fixed-point division.
    pub rounding_mode: RoundingMode,
    /// Overflow policy for integer narrowing and addition.
    pub overflow_mode: OverflowMode,
    /// Canonical reduction ordering policy.
    pub reduction_order: ReductionOrder,
    /// Fixed-point SiLU LUT revision identifier.
    pub silu_lut_version: u32,
    /// Integer RMSNorm epsilon in Q24 units (`>= 1`).
    pub rmsnorm_eps_q24: u64,
}

impl ExecutionProfile {
    /// Canonical `det.zen-kai.v1` profile with banker's rounding
    /// (`RoundHalfToEven`), saturating overflow (`Saturating`), and
    /// `FixedBinaryTree` reduction ordering.
    pub fn v1_default() -> Self {
        Self {
            profile_id: PROFILE_ID_V1.to_string(),
            profile_version: PROFILE_VERSION_V1,
            rounding_mode: RoundingMode::RoundHalfToEven,
            overflow_mode: OverflowMode::Saturating,
            reduction_order: ReductionOrder::FixedBinaryTree,
            silu_lut_version: SILU_LUT_VERSION_V1,
            rmsnorm_eps_q24: DEFAULT_RMSNORM_EPS_Q24,
        }
    }

    /// Construct a `det.zen-kai.v1` profile with explicit rounding, overflow,
    /// and reduction policies.
    pub fn v1(
        rounding_mode: RoundingMode,
        overflow_mode: OverflowMode,
        reduction_order: ReductionOrder,
    ) -> Self {
        Self {
            profile_id: PROFILE_ID_V1.to_string(),
            profile_version: PROFILE_VERSION_V1,
            rounding_mode,
            overflow_mode,
            reduction_order,
            silu_lut_version: SILU_LUT_VERSION_V1,
            rmsnorm_eps_q24: DEFAULT_RMSNORM_EPS_Q24,
        }
    }

    /// Validate profile identifier, version, LUT revision, and epsilon bounds.
    pub fn validate(&self) -> DetResult<()> {
        if self.profile_id != PROFILE_ID_V1 || self.profile_version != PROFILE_VERSION_V1 {
            return Err(DetError::UnknownProfileVersion {
                id: self.profile_id.clone(),
                version: self.profile_version,
            });
        }
        if self.silu_lut_version != SILU_LUT_VERSION_V1 {
            return Err(DetError::UnknownProfileVersion {
                id: format!("{}:silu_lut", self.profile_id),
                version: self.silu_lut_version,
            });
        }
        if self.rmsnorm_eps_q24 == 0 {
            return Err(DetError::DivisionByZero {
                op: "ExecutionProfile::validate(rmsnorm_eps_q24 == 0)",
            });
        }
        Ok(())
    }

    /// Enforce the explicit operator eligibility gate.
    ///
    /// Every ineligible floating-point operator is unconditionally rejected
    /// with `DetError::UnsupportedFloatingPointOp`.
    pub fn validate_operator(&self, op: OperatorId) -> DetResult<EligibleOp> {
        self.validate()?;
        validate_operator(op)
    }
}

/// The 8 eligible deterministic Zen & Kai operators under `det.zen-kai.v1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EligibleOp {
    /// Exact `i8 x i8 -> i64/i128` GEMM with dyadic fixed-point rescaling.
    Int8Gemm = 1,
    /// Exact `i8 x packed-i4 (32-element block, i16 scale)` GEMM with dyadic rescaling.
    PackedInt4Gemm = 2,
    /// Exact `i32` bias addition with declared `OverflowMode`.
    Int32BiasAdd = 3,
    /// Exact fixed-point ReLU (`max(0, x)` with optional upper clamp).
    FixedPointReLU = 4,
    /// Exact fixed-point SiLU using canonical versioned Q15 sigmoid LUT (zero float).
    FixedPointSiLULut = 5,
    /// Exact `i64`/`i128` RMSNorm with bitwise `isqrt_u128` and exact rounded division.
    FixedPointRmsNormInt64 = 6,
    /// Deterministic ArgMax with explicit lowest-index tie-breaking.
    ArgMaxDeterministicTieBreak = 7,
    /// Deterministic Top-K with explicit `(value desc, index asc)` total ordering.
    TopKDeterministicOrder = 8,
}

impl EligibleOp {
    /// All 8 eligible deterministic operators in canonical order.
    pub const ALL: [EligibleOp; 8] = [
        Self::Int8Gemm,
        Self::PackedInt4Gemm,
        Self::Int32BiasAdd,
        Self::FixedPointReLU,
        Self::FixedPointSiLULut,
        Self::FixedPointRmsNormInt64,
        Self::ArgMaxDeterministicTieBreak,
        Self::TopKDeterministicOrder,
    ];

    /// Canonical wire tag.
    #[inline]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Canonical profile specification name.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Int8Gemm => "Int8Gemm",
            Self::PackedInt4Gemm => "PackedInt4Gemm",
            Self::Int32BiasAdd => "Int32BiasAdd",
            Self::FixedPointReLU => "FixedPointReLU",
            Self::FixedPointSiLULut => "FixedPointSiLU_LUT",
            Self::FixedPointRmsNormInt64 => "FixedPointRMSNorm_Int64",
            Self::ArgMaxDeterministicTieBreak => "ArgMax_DeterministicTieBreak",
            Self::TopKDeterministicOrder => "TopK_DeterministicOrder",
        }
    }
}

/// Explicitly excluded floating-point operators that CANNOT be executed under
/// `det.zen-kai.v1` and MUST always fail with `DetError::UnsupportedFloatingPointOp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IneligibleFloatingPointOp {
    /// IEEE-754 binary16 GEMM (non-associative accumulation, vendor tensor-core rounding variance).
    Fp16MatMul = 101,
    /// BFloat16 GEMM (truncated mantissa, non-associative split-K / warp reductions).
    Bf16MatMul = 102,
    /// Parallel FP32 tree/warp reduction (order-dependent IEEE-754 rounding drift).
    Fp32ParallelReduce = 103,
    /// Concurrent `atomicAdd(float*, ...)` across GPU threadblocks (schedule-dependent race).
    AtomicFloatAdd = 104,
    /// Hardware-transcendental `expf`/parallel-sum softmax (vendor libm ULP & reduction drift).
    NonDeterministicSoftmax = 105,
    /// Floating-point cumulative probability nucleus sampling without integer canonicalization.
    UncappedTopPSampling = 106,
    /// Kernel relying on hardware fused multiply-add (`fmaf`) contraction variance across ISAs.
    HardwareFmaDependentOp = 107,
    /// Dynamic per-block FP8 (`E4M3`/`E5M2`) GEMM with floating-point scale re-accumulation.
    Fp8DynamicScaleGemm = 108,
}

impl IneligibleFloatingPointOp {
    /// All 8 explicitly excluded floating-point operators.
    pub const ALL: [IneligibleFloatingPointOp; 8] = [
        Self::Fp16MatMul,
        Self::Bf16MatMul,
        Self::Fp32ParallelReduce,
        Self::AtomicFloatAdd,
        Self::NonDeterministicSoftmax,
        Self::UncappedTopPSampling,
        Self::HardwareFmaDependentOp,
        Self::Fp8DynamicScaleGemm,
    ];

    /// Canonical wire tag.
    #[inline]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Canonical operator name reported in rejection errors.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Fp16MatMul => "Fp16MatMul",
            Self::Bf16MatMul => "Bf16MatMul",
            Self::Fp32ParallelReduce => "Fp32ParallelReduce",
            Self::AtomicFloatAdd => "AtomicFloatAdd",
            Self::NonDeterministicSoftmax => "NonDeterministicSoftmax",
            Self::UncappedTopPSampling => "UncappedTopPSampling",
            Self::HardwareFmaDependentOp => "HardwareFmaDependentOp",
            Self::Fp8DynamicScaleGemm => "Fp8DynamicScaleGemm",
        }
    }

    /// Explicit mathematical/hardware non-determinism rationale.
    pub const fn rejection_reason(self) -> &'static str {
        match self {
            Self::Fp16MatMul => {
                "FP16 accumulation is non-associative and varies across tensor-core/CPU rounding paths"
            }
            Self::Bf16MatMul => {
                "BF16 accumulation exhibits split-K / warp-order rounding divergence across architectures"
            }
            Self::Fp32ParallelReduce => {
                "IEEE-754 FP32 addition is non-associative; parallel reduction trees diverge bit-for-bit"
            }
            Self::AtomicFloatAdd => {
                "GPU atomic float accumulation order depends on non-deterministic hardware thread scheduling"
            }
            Self::NonDeterministicSoftmax => {
                "Floating-point expf and parallel normalization vary in ULP across libm/GPU implementations"
            }
            Self::UncappedTopPSampling => {
                "Floating-point cumulative sum thresholding flips token boundaries under 1-ULP drift"
            }
            Self::HardwareFmaDependentOp => {
                "Fused vs unfused multiply-add contraction differs across x86 AVX2, ARM NEON, CUDA, and Metal"
            }
            Self::Fp8DynamicScaleGemm => {
                "FP8 E4M3/E5M2 dynamic float scaling introduces vendor-specific saturation and rounding drift"
            }
        }
    }
}

/// Unified operator identifier spanning eligible deterministic operators and
/// explicitly excluded floating-point operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorId {
    /// Operator eligible under `det.zen-kai.v1`.
    Eligible(EligibleOp),
    /// Floating-point operator explicitly excluded from deterministic profiles.
    IneligibleFloatingPoint(IneligibleFloatingPointOp),
}

impl OperatorId {
    /// Canonical wire tag.
    #[inline]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Eligible(op) => op.to_u8(),
            Self::IneligibleFloatingPoint(op) => op.to_u8(),
        }
    }

    /// Canonical operator name.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Eligible(op) => op.canonical_name(),
            Self::IneligibleFloatingPoint(op) => op.canonical_name(),
        }
    }
}

/// Validate an operator against the `det.zen-kai.v1` eligibility gate.
///
/// Any floating-point operator returns `Err(DetError::UnsupportedFloatingPointOp)`
/// and is NEVER silently coerced or approximated.
pub fn validate_operator(op: OperatorId) -> DetResult<EligibleOp> {
    match op {
        OperatorId::Eligible(eligible) => Ok(eligible),
        OperatorId::IneligibleFloatingPoint(fp_op) => Err(DetError::UnsupportedFloatingPointOp {
            op_name: fp_op.canonical_name(),
            reason: fp_op.rejection_reason(),
        }),
    }
}

/// Exact dyadic scaling `round(value * scale.multiplier / 2^scale.shift)` in `i128`.
///
/// Uses unsigned magnitude decomposition (`unsigned_abs()`) so extreme negative
/// values (`i64::MIN`, large `i128` negatives) never suffer signed-negation
/// overflow or asymmetric shift truncation.
pub fn rescale_i128(
    value: i128,
    scale: DyadicScale,
    rounding: RoundingMode,
    op: &'static str,
) -> DetResult<i128> {
    scale.validate()?;
    let prod = value
        .checked_mul(i128::from(scale.multiplier))
        .ok_or_else(|| DetError::ArithmeticOverflow {
            op,
            detail: format!("i128 product overflow: {value} * {}", scale.multiplier),
        })?;
    round_shift_right_i128(prod, scale.shift, rounding)
}

/// Exact rounded division by `2^shift` (`shift <= 62`) over signed `i128`.
pub fn round_shift_right_i128(prod: i128, shift: u8, rounding: RoundingMode) -> DetResult<i128> {
    if shift > MAX_DYADIC_SHIFT {
        return Err(DetError::InvalidScale {
            context: "round_shift_right_i128",
            detail: format!("shift {shift} exceeds MAX_DYADIC_SHIFT ({MAX_DYADIC_SHIFT})"),
        });
    }
    if shift == 0 {
        return Ok(prod);
    }
    let neg = prod < 0;
    let mag: u128 = prod.unsigned_abs();
    let q: u128 = mag >> shift;
    let mask: u128 = (1u128 << shift) - 1;
    let rem: u128 = mag & mask;
    let half: u128 = 1u128 << (shift - 1);

    let round_up = match rem.cmp(&half) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => match rounding {
            RoundingMode::RoundHalfToEven => (q & 1) == 1,
            RoundingMode::RoundHalfAwayFromZero => true,
        },
    };
    let q_rounded = if round_up { q + 1 } else { q };
    signed_from_mag_u128(neg, q_rounded, "round_shift_right_i128")
}

/// Exact rounded integer division `round(num / den)` for signed `num: i128` and
/// positive `den: u128`, with zero intermediate overflow across the full `u128` range.
pub fn round_div_i128(
    num: i128,
    den: u128,
    rounding: RoundingMode,
    op: &'static str,
) -> DetResult<i128> {
    if den == 0 {
        return Err(DetError::DivisionByZero { op });
    }
    let neg = num < 0;
    let mag: u128 = num.unsigned_abs();
    let q: u128 = mag / den;
    let rem: u128 = mag % den;

    // Compare `2 * rem` against `den` without overflowing `u128`:
    // `2 * rem <=> den` is equivalent to `rem <=> (den - rem)` since `rem < den`.
    let round_up = match rem.cmp(&(den - rem)) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => match rounding {
            RoundingMode::RoundHalfToEven => (q & 1) == 1,
            RoundingMode::RoundHalfAwayFromZero => true,
        },
    };
    let q_rounded = if round_up {
        q.checked_add(1)
            .ok_or_else(|| DetError::ArithmeticOverflow {
                op,
                detail: "quotient increment overflow".to_string(),
            })?
    } else {
        q
    };
    signed_from_mag_u128(neg, q_rounded, op)
}

/// Narrow an `i128` value to `i32` under the profile's declared `OverflowMode`.
#[inline]
pub fn narrow_i128_to_i32(val: i128, overflow: OverflowMode, op: &'static str) -> DetResult<i32> {
    match overflow {
        OverflowMode::Saturating => {
            Ok(val.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32)
        }
        OverflowMode::CheckedError => {
            i32::try_from(val).map_err(|_| DetError::ArithmeticOverflow {
                op,
                detail: format!("value {val} out of i32 range"),
            })
        }
    }
}

/// Narrow an `i128` value to `i64` under the profile's declared `OverflowMode`.
#[inline]
pub fn narrow_i128_to_i64(val: i128, overflow: OverflowMode, op: &'static str) -> DetResult<i64> {
    match overflow {
        OverflowMode::Saturating => {
            Ok(val.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64)
        }
        OverflowMode::CheckedError => {
            i64::try_from(val).map_err(|_| DetError::ArithmeticOverflow {
                op,
                detail: format!("value {val} out of i64 range"),
            })
        }
    }
}

/// Exact `i32` addition under the declared `OverflowMode`.
#[inline]
pub fn add_i32(a: i32, b: i32, overflow: OverflowMode, op: &'static str) -> DetResult<i32> {
    match overflow {
        OverflowMode::Saturating => Ok(a.saturating_add(b)),
        OverflowMode::CheckedError => {
            a.checked_add(b)
                .ok_or_else(|| DetError::ArithmeticOverflow {
                    op,
                    detail: format!("i32 addition overflow: {a} + {b}"),
                })
        }
    }
}

/// Exact bitwise integer square root `floor(sqrt(n))` over full `u128`.
///
/// Uses zero floating-point operations and satisfies `r^2 <= n < (r + 1)^2` for
/// every `n in 0..=u128::MAX`.
pub fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let highest_even_bit = (127 - n.leading_zeros()) & !1;
    let mut bit: u128 = 1u128 << highest_even_bit;
    let mut rem: u128 = n;
    let mut res: u128 = 0;

    while bit != 0 {
        let trial = res + bit;
        if rem >= trial {
            rem -= trial;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    res
}

#[inline]
fn signed_from_mag_u128(neg: bool, mag: u128, op: &'static str) -> DetResult<i128> {
    const I128_MIN_MAG: u128 = 1u128 << 127;
    if !neg {
        i128::try_from(mag).map_err(|_| DetError::ArithmeticOverflow {
            op,
            detail: format!("positive magnitude {mag} exceeds i128::MAX"),
        })
    } else if mag == 0 {
        Ok(0)
    } else if mag == I128_MIN_MAG {
        Ok(i128::MIN)
    } else if mag < I128_MIN_MAG {
        Ok(-(mag as i128))
    } else {
        Err(DetError::ArithmeticOverflow {
            op,
            detail: format!("negative magnitude -{mag} below i128::MIN"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_midpoint_ties_positive_and_negative() {
        // shift = 1 => denominator = 2, exact ties at odd integers.
        let s = DyadicScale::new(1, 1).unwrap();
        // +0.5: even -> 0, away -> +1
        assert_eq!(
            rescale_i128(1, s, RoundingMode::RoundHalfToEven, "t").unwrap(),
            0
        );
        assert_eq!(
            rescale_i128(1, s, RoundingMode::RoundHalfAwayFromZero, "t").unwrap(),
            1
        );
        // -0.5: even -> 0, away -> -1
        assert_eq!(
            rescale_i128(-1, s, RoundingMode::RoundHalfToEven, "t").unwrap(),
            0
        );
        assert_eq!(
            rescale_i128(-1, s, RoundingMode::RoundHalfAwayFromZero, "t").unwrap(),
            -1
        );
        // +1.5: even -> +2, away -> +2
        assert_eq!(
            rescale_i128(3, s, RoundingMode::RoundHalfToEven, "t").unwrap(),
            2
        );
        assert_eq!(
            rescale_i128(3, s, RoundingMode::RoundHalfAwayFromZero, "t").unwrap(),
            2
        );
        // -1.5: even -> -2, away -> -2
        assert_eq!(
            rescale_i128(-3, s, RoundingMode::RoundHalfToEven, "t").unwrap(),
            -2
        );
        assert_eq!(
            rescale_i128(-3, s, RoundingMode::RoundHalfAwayFromZero, "t").unwrap(),
            -2
        );
        // +2.5: even -> +2, away -> +3
        assert_eq!(
            rescale_i128(5, s, RoundingMode::RoundHalfToEven, "t").unwrap(),
            2
        );
        assert_eq!(
            rescale_i128(5, s, RoundingMode::RoundHalfAwayFromZero, "t").unwrap(),
            3
        );
        // -2.5: even -> -2, away -> -3
        assert_eq!(
            rescale_i128(-5, s, RoundingMode::RoundHalfToEven, "t").unwrap(),
            -2
        );
        assert_eq!(
            rescale_i128(-5, s, RoundingMode::RoundHalfAwayFromZero, "t").unwrap(),
            -3
        );
    }

    #[test]
    fn extreme_i64_and_i128_min_no_panic_no_wrap() {
        let s = DyadicScale::new(1, 1).unwrap();
        assert_eq!(
            round_shift_right_i128(i128::MIN, 1, RoundingMode::RoundHalfToEven).unwrap(),
            i128::MIN / 2
        );
        assert_eq!(
            round_shift_right_i128(i128::MIN, 0, RoundingMode::RoundHalfToEven).unwrap(),
            i128::MIN
        );
        assert_eq!(
            rescale_i128(i128::from(i64::MIN), s, RoundingMode::RoundHalfToEven, "t").unwrap(),
            i128::from(i64::MIN) / 2
        );
        assert_eq!(
            round_div_i128(i128::MIN, 1, RoundingMode::RoundHalfToEven, "t").unwrap(),
            i128::MIN
        );
        assert_eq!(
            round_div_i128(i128::MIN, 2, RoundingMode::RoundHalfToEven, "t").unwrap(),
            i128::MIN / 2
        );
        // CheckedError rejects narrowing overflow, Saturating clamps cleanly.
        assert!(narrow_i128_to_i32(i128::from(i64::MIN), OverflowMode::CheckedError, "t").is_err());
        assert_eq!(
            narrow_i128_to_i32(i128::from(i64::MIN), OverflowMode::Saturating, "t").unwrap(),
            i32::MIN
        );
        assert_eq!(
            narrow_i128_to_i64(i128::MIN, OverflowMode::Saturating, "t").unwrap(),
            i64::MIN
        );
        assert!(narrow_i128_to_i64(i128::MIN, OverflowMode::CheckedError, "t").is_err());
    }

    #[test]
    fn isqrt_u128_exact_across_boundaries() {
        for n in [0u128, 1, 2, 3, 4, 8, 9, 15, 16, 17, 65535, 65536, 65537] {
            let r = isqrt_u128(n);
            assert!(r * r <= n);
            assert!((r + 1) * (r + 1) > n);
        }
        assert_eq!(isqrt_u128(u128::MAX), u128::from(u64::MAX));
        assert_eq!(isqrt_u128(1u128 << 126), 1u128 << 63);
        assert_eq!(isqrt_u128((1u128 << 126) - 1), (1u128 << 63) - 1);
    }

    #[test]
    fn all_eight_ineligible_float_ops_rejected() {
        let profile = ExecutionProfile::v1_default();
        for fp_op in IneligibleFloatingPointOp::ALL {
            let err = profile
                .validate_operator(OperatorId::IneligibleFloatingPoint(fp_op))
                .expect_err("floating-point op must be rejected");
            assert!(matches!(err, DetError::UnsupportedFloatingPointOp { .. }));
        }
        for ok_op in EligibleOp::ALL {
            assert_eq!(
                profile
                    .validate_operator(OperatorId::Eligible(ok_op))
                    .unwrap(),
                ok_op
            );
        }
    }
}
