// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic fixed-point execution profiles (`opai.det.zen-kai.v1`) and
//! cryptographic `InputCommitment` bindings aligned with HIP-1350.
//!
//! All arithmetic inside the optimistic execution VM is strictly integer /
//! dyadic fixed-point (`i128` accumulators, bitwise `isqrt_u128`, and a
//! canonical 257-knot Q15 sigmoid LUT). Any profile requesting floating-point
//! execution is explicitly rejected by [`ExecutionProfile::validate`].

use crate::crypto::{Digest, Sha256, DOMAIN_INPUT, DOMAIN_PROFILE};
use crate::error::OpAiError;

/// Canonical profile identifier aligned with `hanzo-detprofile` (`det.zen-kai.v1`).
pub const CANONICAL_PROFILE_ID: &str = "opai.det.zen-kai.v1";

/// Canonical profile schema version.
pub const CANONICAL_PROFILE_VERSION: u32 = 1;

/// Canonical Q15 sigmoid LUT version.
pub const CANONICAL_SILU_LUT_VERSION: u32 = 1;

/// Default RMSNorm epsilon in Q24 fixed-point (`~1e-5 * 2^24 = 168`).
pub const DEFAULT_RMSNORM_EPS_Q24: u64 = 168;

/// Rounding policy for exact dyadic rescaling and integer division.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RoundingMode {
    /// Banker's rounding (ties round to nearest even integer).
    RoundHalfToEven = 0,
    /// Symmetric half-away-from-zero rounding (`+0.5 -> +1`, `-0.5 -> -1`).
    RoundHalfAwayFromZero = 1,
}

impl RoundingMode {
    /// Decode a rounding mode from its canonical wire tag.
    pub fn from_tag(tag: u8) -> Result<Self, OpAiError> {
        match tag {
            0 => Ok(Self::RoundHalfToEven),
            1 => Ok(Self::RoundHalfAwayFromZero),
            other => Err(OpAiError::WireFormat(format!(
                "unknown RoundingMode tag: {other}"
            ))),
        }
    }

    /// Return the canonical wire tag.
    pub fn to_tag(self) -> u8 {
        self as u8
    }
}

/// Overflow policy for integer narrowing and residual/bias additions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OverflowMode {
    /// Clamp out-of-range values strictly to `[T::MIN, T::MAX]`.
    Saturating = 0,
    /// Deterministically abort with [`OpAiError::ArithmeticOverflow`].
    CheckedError = 1,
}

impl OverflowMode {
    /// Decode an overflow mode from its canonical wire tag.
    pub fn from_tag(tag: u8) -> Result<Self, OpAiError> {
        match tag {
            0 => Ok(Self::Saturating),
            1 => Ok(Self::CheckedError),
            other => Err(OpAiError::WireFormat(format!(
                "unknown OverflowMode tag: {other}"
            ))),
        }
    }

    /// Return the canonical wire tag.
    pub fn to_tag(self) -> u8 {
        self as u8
    }
}

/// Numeric domain declared by an execution profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NumericDomain {
    /// Pure integer and dyadic fixed-point arithmetic (cross-hardware bit-identical).
    FixedPointInt = 0,
    /// Floating-point (`FP16`/`BF16`/`FP32`/`FP8`) — explicitly rejected for fraud proofs.
    FloatingPointDisallowed = 1,
}

impl NumericDomain {
    /// Decode a numeric domain from its canonical wire tag.
    pub fn from_tag(tag: u8) -> Result<Self, OpAiError> {
        match tag {
            0 => Ok(Self::FixedPointInt),
            1 => Ok(Self::FloatingPointDisallowed),
            other => Err(OpAiError::WireFormat(format!(
                "unknown NumericDomain tag: {other}"
            ))),
        }
    }

    /// Return the canonical wire tag.
    pub fn to_tag(self) -> u8 {
        self as u8
    }
}

/// Exact rational dyadic scale $S = \text{multiplier} \cdot 2^{-\text{shift}}$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DyadicScale {
    /// Signed integer numerator.
    pub multiplier: i32,
    /// Right-shift exponent in `0..=62`.
    pub shift: u8,
}

impl DyadicScale {
    /// Construct and validate a dyadic scale.
    pub fn new(multiplier: i32, shift: u8) -> Result<Self, OpAiError> {
        if shift > 62 {
            return Err(OpAiError::InvalidProfile(format!(
                "dyadic scale shift {shift} exceeds maximum 62"
            )));
        }
        Ok(Self { multiplier, shift })
    }

    /// Absorb the dyadic scale into a SHA-256 hasher.
    pub fn hash_into(&self, h: &mut Sha256) {
        h.update_i32_le(self.multiplier);
        h.update_u8(self.shift);
    }
}

/// Complete deterministic execution profile cryptographically bound into every [`crate::game::Claim`].
///
/// Provider, challenger, and single-step judge must agree on `profile.digest()`
/// so every step `0..T` executes identical integer semantics across all hosts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionProfile {
    /// Human-readable canonical profile identifier (e.g. `opai.det.zen-kai.v1`).
    pub profile_id: String,
    /// Profile schema version (`1`).
    pub profile_version: u32,
    /// Numeric execution domain (must be [`NumericDomain::FixedPointInt`]).
    pub numeric_domain: NumericDomain,
    /// Rounding policy for dyadic scaling and integer division.
    pub rounding_mode: RoundingMode,
    /// Overflow policy for narrowing and bias/residual additions.
    pub overflow_mode: OverflowMode,
    /// Version of the compile-time 257-knot Q15 sigmoid LUT.
    pub silu_lut_version: u32,
    /// RMSNorm epsilon in Q24 fixed-point (must be `>= 1` to guarantee non-zero denominator).
    pub rmsnorm_eps_q24: u64,
}

impl ExecutionProfile {
    /// Return the canonical `opai.det.zen-kai.v1` deterministic execution profile.
    pub fn canonical() -> Self {
        Self {
            profile_id: CANONICAL_PROFILE_ID.to_string(),
            profile_version: CANONICAL_PROFILE_VERSION,
            numeric_domain: NumericDomain::FixedPointInt,
            rounding_mode: RoundingMode::RoundHalfToEven,
            overflow_mode: OverflowMode::Saturating,
            silu_lut_version: CANONICAL_SILU_LUT_VERSION,
            rmsnorm_eps_q24: DEFAULT_RMSNORM_EPS_Q24,
        }
    }

    /// Validate that this profile enforces bit-identical fixed-point execution.
    pub fn validate(&self) -> Result<(), OpAiError> {
        if self.numeric_domain != NumericDomain::FixedPointInt {
            return Err(OpAiError::UnsupportedFloatingPointProfile {
                reason: "floating-point execution is excluded from optimistic fraud-proof games; \
                         use NumericDomain::FixedPointInt"
                    .to_string(),
            });
        }
        if self.profile_id.is_empty() || self.profile_id.len() > 128 {
            return Err(OpAiError::InvalidProfile(
                "profile_id must be 1..=128 bytes".to_string(),
            ));
        }
        if self.profile_version == 0 {
            return Err(OpAiError::InvalidProfile(
                "profile_version must be >= 1".to_string(),
            ));
        }
        if self.silu_lut_version != CANONICAL_SILU_LUT_VERSION {
            return Err(OpAiError::InvalidProfile(format!(
                "unsupported silu_lut_version {}, expected {}",
                self.silu_lut_version, CANONICAL_SILU_LUT_VERSION
            )));
        }
        if self.rmsnorm_eps_q24 == 0 {
            return Err(OpAiError::InvalidProfile(
                "rmsnorm_eps_q24 must be >= 1 to guarantee a non-zero denominator".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute the canonical domain-separated SHA-256 digest of all profile parameters.
    pub fn digest(&self) -> Digest {
        let mut h = Sha256::with_domain(DOMAIN_PROFILE);
        h.update_str(&self.profile_id);
        h.update_u32_le(self.profile_version);
        h.update_u8(self.numeric_domain.to_tag());
        h.update_u8(self.rounding_mode.to_tag());
        h.update_u8(self.overflow_mode.to_tag());
        h.update_u32_le(self.silu_lut_version);
        h.update_u64_le(self.rmsnorm_eps_q24);
        h.finalize()
    }
}

/// Canonical commitment to a prompt / token input bound to a specific model & profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputCommitment {
    /// Deterministic prompt token IDs (at least 1 token).
    pub prompt_tokens: Vec<u32>,
    /// Number of autoregressive output tokens to generate (at least 1).
    pub max_new_tokens: u32,
    /// SHA-256 digest of the raw source prompt bytes (e.g., corpus slice).
    pub prompt_bytes_digest: Digest,
    /// Bound execution profile digest.
    pub profile_digest: Digest,
    /// Bound quantized model digest.
    pub model_digest: Digest,
}

impl InputCommitment {
    /// Construct and validate an `InputCommitment` from prompt bytes, tokens, and bindings.
    pub fn new(
        prompt_bytes: &[u8],
        prompt_tokens: Vec<u32>,
        max_new_tokens: u32,
        profile_digest: Digest,
        model_digest: Digest,
    ) -> Result<Self, OpAiError> {
        if prompt_tokens.is_empty() {
            return Err(OpAiError::InvalidInput(
                "prompt_tokens must be non-empty".to_string(),
            ));
        }
        if max_new_tokens == 0 {
            return Err(OpAiError::InvalidInput(
                "max_new_tokens must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            prompt_tokens,
            max_new_tokens,
            prompt_bytes_digest: Sha256::digest(prompt_bytes),
            profile_digest,
            model_digest,
        })
    }

    /// Compute the canonical domain-separated `input_hash` bound into every `Claim`.
    pub fn input_hash(&self) -> Digest {
        let mut h = Sha256::with_domain(DOMAIN_INPUT);
        h.update(&self.profile_digest);
        h.update(&self.model_digest);
        h.update(&self.prompt_bytes_digest);
        h.update_u32_le(self.max_new_tokens);
        h.update_u64_le(self.prompt_tokens.len() as u64);
        for &tok in &self.prompt_tokens {
            h.update_u32_le(tok);
        }
        h.finalize()
    }
}

/// Exact overflow-free dyadic rescaling of `val * scale.multiplier / 2^scale.shift`.
pub fn rescale_i128(
    val: i128,
    scale: DyadicScale,
    rounding: RoundingMode,
) -> Result<i128, OpAiError> {
    if scale.shift > 62 {
        return Err(OpAiError::InvalidProfile(format!(
            "invalid dyadic shift {}",
            scale.shift
        )));
    }
    let prod = val
        .checked_mul(i128::from(scale.multiplier))
        .ok_or(OpAiError::ArithmeticOverflow("rescale_i128::checked_mul"))?;
    if scale.shift == 0 {
        return Ok(prod);
    }
    let neg = prod < 0;
    let mag: u128 = prod.unsigned_abs();
    let shift = u32::from(scale.shift);
    let q: u128 = mag >> shift;
    let mask: u128 = (1u128 << shift) - 1;
    let rem: u128 = mag & mask;
    let half: u128 = 1u128 << (shift - 1);

    let round_up = match rounding {
        RoundingMode::RoundHalfToEven => rem > half || (rem == half && (q & 1) == 1),
        RoundingMode::RoundHalfAwayFromZero => rem >= half,
    };
    let rounded_mag = if round_up {
        q.checked_add(1)
            .ok_or(OpAiError::ArithmeticOverflow("rescale_i128::round_up"))?
    } else {
        q
    };
    apply_sign_u128(rounded_mag, neg)
}

/// Exact overflow-free integer division `num / den` under the declared [`RoundingMode`].
pub fn round_div_i128(num: i128, den: i128, rounding: RoundingMode) -> Result<i128, OpAiError> {
    if den == 0 {
        return Err(OpAiError::ArithmeticOverflow("round_div_i128::div_by_zero"));
    }
    let neg = (num < 0) ^ (den < 0);
    let n: u128 = num.unsigned_abs();
    let d: u128 = den.unsigned_abs();
    let q: u128 = n / d;
    let r: u128 = n % d;
    // Compare 2*r vs d without u128 overflow: 2*r > d <=> r > d - r.
    let cmp_half = r.cmp(&(d - r));
    let round_up = match rounding {
        RoundingMode::RoundHalfToEven => {
            cmp_half == std::cmp::Ordering::Greater
                || (cmp_half == std::cmp::Ordering::Equal && (q & 1) == 1)
        }
        RoundingMode::RoundHalfAwayFromZero => cmp_half != std::cmp::Ordering::Less,
    };
    let rounded_mag = if round_up {
        q.checked_add(1)
            .ok_or(OpAiError::ArithmeticOverflow("round_div_i128::round_up"))?
    } else {
        q
    };
    apply_sign_u128(rounded_mag, neg)
}

fn apply_sign_u128(mag: u128, neg: bool) -> Result<i128, OpAiError> {
    if !neg {
        i128::try_from(mag).map_err(|_| OpAiError::ArithmeticOverflow("apply_sign_u128::pos"))
    } else if mag == (i128::MAX as u128) + 1 {
        Ok(i128::MIN)
    } else {
        let pos = i128::try_from(mag)
            .map_err(|_| OpAiError::ArithmeticOverflow("apply_sign_u128::neg"))?;
        Ok(-pos)
    }
}

/// Exact bitwise integer square root $\lfloor \sqrt{n} \rfloor$ over `u128` (zero floats).
pub fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut rem = n;
    let mut root = 0u128;
    // Highest power of 4 <= n.
    let shift = (127 - n.leading_zeros()) & !1;
    let mut bit = 1u128 << shift;
    while bit != 0 {
        let trial = root + bit;
        if rem >= trial {
            rem -= trial;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

/// Narrow an `i128` accumulator to `i32` respecting [`OverflowMode`].
pub fn narrow_i32(val: i128, overflow: OverflowMode) -> Result<i32, OpAiError> {
    match overflow {
        OverflowMode::Saturating => {
            Ok(val.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32)
        }
        OverflowMode::CheckedError => {
            i32::try_from(val).map_err(|_| OpAiError::ArithmeticOverflow("narrow_i32"))
        }
    }
}

/// Add two `i32` values respecting [`OverflowMode`].
pub fn add_i32(a: i32, b: i32, overflow: OverflowMode) -> Result<i32, OpAiError> {
    match overflow {
        OverflowMode::Saturating => Ok(a.saturating_add(b)),
        OverflowMode::CheckedError => a
            .checked_add(b)
            .ok_or(OpAiError::ArithmeticOverflow("add_i32")),
    }
}

/// Canonical 257-knot Q15 sigmoid lookup table over $x \in [-8.0, +8.0]$ (`step = 1/16 = 256` in Q12).
///
/// Exact symmetry invariant: `SIGMOID_LUT_Q15[i] + SIGMOID_LUT_Q15[256 - i] == 32768`.
pub const SIGMOID_LUT_Q15: [i32; 257] = [
    11, 12, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 28, 30, 32, 34, 36, 38, 41, 43, 46,
    49, 52, 56, 59, 63, 67, 72, 76, 81, 86, 92, 98, 104, 111, 118, 125, 133, 142, 151, 161, 171,
    182, 194, 206, 219, 233, 248, 264, 281, 299, 318, 338, 360, 383, 407, 433, 461, 490, 521, 554,
    589, 627, 666, 708, 753, 800, 851, 904, 961, 1021, 1084, 1152, 1223, 1299, 1379, 1464, 1554,
    1649, 1750, 1856, 1969, 2088, 2213, 2346, 2486, 2633, 2789, 2952, 3124, 3306, 3496, 3696, 3906,
    4126, 4357, 4599, 4851, 5115, 5391, 5678, 5978, 6289, 6613, 6949, 7297, 7658, 8031, 8416, 8813,
    9221, 9641, 10072, 10513, 10964, 11424, 11894, 12371, 12856, 13348, 13845, 14347, 14852, 15361,
    15872, 16384, 16896, 17407, 17916, 18421, 18923, 19420, 19912, 20397, 20874, 21344, 21804,
    22255, 22696, 23127, 23547, 23955, 24352, 24737, 25110, 25471, 25819, 26155, 26479, 26790,
    27090, 27377, 27653, 27917, 28169, 28411, 28642, 28862, 29072, 29272, 29462, 29644, 29816,
    29979, 30135, 30282, 30422, 30555, 30680, 30799, 30912, 31018, 31119, 31214, 31304, 31389,
    31469, 31545, 31616, 31684, 31747, 31807, 31864, 31917, 31968, 32015, 32060, 32102, 32141,
    32179, 32214, 32247, 32278, 32307, 32335, 32361, 32385, 32408, 32430, 32450, 32469, 32487,
    32504, 32520, 32535, 32549, 32562, 32574, 32586, 32597, 32607, 32617, 32626, 32635, 32643,
    32650, 32657, 32664, 32670, 32676, 32682, 32687, 32692, 32696, 32701, 32705, 32709, 32712,
    32716, 32719, 32722, 32725, 32727, 32730, 32732, 32734, 32736, 32738, 32740, 32742, 32743,
    32745, 32746, 32747, 32749, 32750, 32751, 32752, 32753, 32754, 32755, 32756, 32756, 32757,
];

/// Exact fixed-point SiLU $x \cdot \sigma(x)$ for Q12 input (`1.0 = 4096`), zero floats.
pub fn fixed_point_silu_q12(
    x_q12: i32,
    rounding: RoundingMode,
    overflow: OverflowMode,
) -> Result<i32, OpAiError> {
    const MIN_X: i32 = -32768; // -8.0 in Q12
    const MAX_X: i32 = 32768; // +8.0 in Q12
    let sig_q15: i32 = if x_q12 <= MIN_X {
        0
    } else if x_q12 >= MAX_X {
        32768
    } else {
        let shifted = (i64::from(x_q12) - i64::from(MIN_X)) as u64;
        let idx = (shifted >> 8) as usize;
        let frac = (shifted & 0xff) as i64;
        let y0 = i64::from(SIGMOID_LUT_Q15[idx]);
        let y1 = i64::from(SIGMOID_LUT_Q15[idx + 1]);
        let interp = y0 + (((y1 - y0) * frac) >> 8);
        interp as i32
    };

    // Product x_q12 * sig_q15 has scale Q27; rescale by 2^-15 back to Q12.
    let scale = DyadicScale {
        multiplier: sig_q15,
        shift: 15,
    };
    let out = rescale_i128(i128::from(x_q12), scale, rounding)?;
    narrow_i32(out, overflow)
}
