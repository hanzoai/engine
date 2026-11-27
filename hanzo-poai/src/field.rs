// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Explicit prime finite field `GF(p)` for Mersenne prime `p = 2^61 - 1`.
//!
//! # Choice of Prime Modulus
//!
//! We use the 9th Mersenne prime:
//! `p = 2^61 - 1 = 2_305_843_009_213_693_951` (`0x1FFF_FFFF_FFFF_FFFF`).
//!
//! This modulus provides three structural properties required for CPU-verifiable
//! Proof of AI over packed low-bit quantized integer matrix products:
//!
//! 1. **Exact 128-bit multiply-reduce with zero division**: Because `2^61 == 1 (mod p)`,
//!    multiplying two 61-bit residues produces a 122-bit integer whose reduction modulo `p`
//!    requires only bit-shifts, bitmasks, and two conditional additions — zero `div`/`rem`
//!    instructions and zero overflow risk.
//! 2. **Wraparound-free signed integer bijection**: Every integer `x` satisfying
//!    `|x| <= (p - 1) / 2 = 2^60 - 1 = 1_152_921_504_606_846_975` maps injectively into
//!    `GF(p)`. For `i8` activations (`|a| <= 128`), `i8`/`i4`/`i2` weights (`|q| <= 128`),
//!    and `i16` block scales (`|s| <= 32767`), each product term satisfies
//!    `|s * a * q| < 2^29`, allowing reduction lengths up to `K < 2^30` (> 1 billion)
//!    with bit-for-bit agreement between exact `i64` arithmetic and `GF(p)`.
//! 3. **Cryptographic soundness in 2 rounds**: A single Freivalds round over uniform
//!    non-zero `r in GF(p)^n \ {0}` achieves soundness error strictly below `1/p ~= 4.34e-19`.
//!    Two independent rounds achieve error `< 1/p^2 ~= 1.88e-37 < 2^-121`.

use crate::error::PoaiError;

/// Mersenne prime modulus `p = 2^61 - 1 = 2_305_843_009_213_693_951`.
pub const MODULUS_P: u64 = (1u64 << 61) - 1;

/// Exact inclusive signed magnitude bound `(p - 1) / 2 = 2^60 - 1 = 1_152_921_504_606_846_975`.
///
/// Any signed integer `x` with `|x| <= MAX_SAFE_SIGNED` embeds injectively into `GF(p)`
/// without wraparound ambiguity.
pub const MAX_SAFE_SIGNED: i64 = ((MODULUS_P - 1) / 2) as i64;

/// Element of the prime field `GF(2^61 - 1)`.
///
/// Invariant: the inner `u64` is always canonical, i.e. `0 <= self.0 < MODULUS_P`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Fp(u64);

impl Fp {
    /// Additive identity `0`.
    pub const ZERO: Self = Self(0);

    /// Multiplicative identity `1`.
    pub const ONE: Self = Self(1);

    /// Construct a field element from a `u64` already known to satisfy `v < MODULUS_P`.
    #[inline]
    pub const fn from_canonical(v: u64) -> Option<Self> {
        if v < MODULUS_P {
            Some(Self(v))
        } else {
            None
        }
    }

    /// Return the canonical representative in `[0, p - 1]`.
    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Reduce an arbitrary `u64` modulo `p = 2^61 - 1` without division.
    #[inline]
    pub const fn from_u64_reduced(x: u64) -> Self {
        let mut r = (x & MODULUS_P) + (x >> 61);
        if r >= MODULUS_P {
            r -= MODULUS_P;
        }
        Self(r)
    }

    /// Reduce an arbitrary `u128` modulo `p = 2^61 - 1` without division or overflow.
    ///
    /// Proof of no overflow and exact canonical range:
    /// - `x = lo + mid * 2^61 + hi * 2^122` with `lo <= p`, `mid <= p`, `hi <= 63`.
    /// - `sum = lo + mid + hi <= 2p + 63 = 2^62 + 61 < 2^64` (cannot overflow `u64`).
    /// - `r = (sum & p) + (sum >> 61) <= p + 2 < 2p`.
    /// - One conditional subtraction `if r >= p { r -= p }` yields `0 <= r < p`.
    #[inline]
    pub const fn reduce_u128(x: u128) -> Self {
        let lo = (x & (MODULUS_P as u128)) as u64;
        let mid = ((x >> 61) & (MODULUS_P as u128)) as u64;
        let hi = (x >> 122) as u64;
        let sum = lo + mid + hi;
        let mut r = (sum & MODULUS_P) + (sum >> 61);
        if r >= MODULUS_P {
            r -= MODULUS_P;
        }
        Self(r)
    }

    /// Reduce an arbitrary signed `i128` modulo `p = 2^61 - 1`.
    #[inline]
    pub const fn reduce_i128(x: i128) -> Self {
        if x >= 0 {
            Self::reduce_u128(x as u128)
        } else {
            Self::reduce_u128(x.unsigned_abs()).neg()
        }
    }

    /// Embed a signed `i8` into `GF(p)`. Always inside the safe bijection range.
    #[inline]
    pub const fn from_i8(v: i8) -> Self {
        if v >= 0 {
            Self(v as u64)
        } else {
            Self(MODULUS_P - (v.unsigned_abs() as u64))
        }
    }

    /// Embed a signed `i16` into `GF(p)`. Always inside the safe bijection range.
    #[inline]
    pub const fn from_i16(v: i16) -> Self {
        if v >= 0 {
            Self(v as u64)
        } else {
            Self(MODULUS_P - (v.unsigned_abs() as u64))
        }
    }

    /// Embed a signed `i64` into `GF(p)`, rejecting any value with `|v| > (p - 1)/2`.
    ///
    /// Enforcing this strict range guarantees that no integer output wraparound can occur
    /// between `i64` and `GF(p)`.
    #[inline]
    pub fn from_i64_checked(v: i64, row: usize, col: usize) -> Result<Self, PoaiError> {
        let abs = v.unsigned_abs();
        if abs > MAX_SAFE_SIGNED as u64 {
            return Err(PoaiError::OutputOutOfFieldRange {
                row,
                col,
                value: v,
                max_abs: MAX_SAFE_SIGNED,
            });
        }
        if v >= 0 {
            Ok(Self(v as u64))
        } else {
            Ok(Self(MODULUS_P - abs))
        }
    }

    /// Map a canonical field element back to its unique signed representative in `[-(p-1)/2, +(p-1)/2]`.
    #[inline]
    pub const fn to_signed_i64(self) -> i64 {
        if self.0 <= MAX_SAFE_SIGNED as u64 {
            self.0 as i64
        } else {
            -((MODULUS_P - self.0) as i64)
        }
    }

    /// Additive inverse `-self (mod p)`.
    #[inline]
    pub const fn neg(self) -> Self {
        if self.0 == 0 {
            Self::ZERO
        } else {
            Self(MODULUS_P - self.0)
        }
    }

    /// Field addition `(self + rhs) mod p`.
    #[inline]
    pub const fn add(self, rhs: Self) -> Self {
        let sum = self.0 + rhs.0;
        if sum >= MODULUS_P {
            Self(sum - MODULUS_P)
        } else {
            Self(sum)
        }
    }

    /// Field subtraction `(self - rhs) mod p`.
    #[inline]
    pub const fn sub(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            Self(self.0 - rhs.0)
        } else {
            Self(MODULUS_P - (rhs.0 - self.0))
        }
    }

    /// Field multiplication `(self * rhs) mod p` via exact 128-bit multiply-reduce.
    #[inline]
    pub const fn mul(self, rhs: Self) -> Self {
        Self::reduce_u128((self.0 as u128) * (rhs.0 as u128))
    }

    /// Exponentiation `self^exp (mod p)` via square-and-multiply.
    #[inline]
    pub const fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut acc = Self::ONE;
        while exp > 0 {
            if (exp & 1) == 1 {
                acc = acc.mul(base);
            }
            base = base.mul(base);
            exp >>= 1;
        }
        acc
    }

    /// Multiplicative inverse via Fermat's Little Theorem `a^(p - 2) mod p`.
    /// Returns `None` for `0`.
    #[inline]
    pub const fn inv(self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            Some(self.pow(MODULUS_P - 2))
        }
    }

    /// Canonical 8-byte little-endian encoding.
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 8] {
        self.0.to_le_bytes()
    }
}

/// Constant-time equality for 32-byte SHA-256 digests.
#[inline]
pub fn ct_eq_digest(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulus_is_mersenne_m61() {
        assert_eq!(MODULUS_P, 2_305_843_009_213_693_951);
        assert_eq!(MAX_SAFE_SIGNED, 1_152_921_504_606_846_975);
        assert_eq!((MAX_SAFE_SIGNED as u64) * 2 + 1, MODULUS_P);
    }

    #[test]
    fn reduce_edge_cases_exact() {
        assert_eq!(Fp::from_u64_reduced(0), Fp::ZERO);
        assert_eq!(Fp::from_u64_reduced(MODULUS_P), Fp::ZERO);
        assert_eq!(Fp::from_u64_reduced(MODULUS_P - 1).raw(), MODULUS_P - 1);
        assert_eq!(Fp::from_u64_reduced(MODULUS_P + 1), Fp::ONE);
        assert_eq!(Fp::from_u64_reduced(u64::MAX).raw(), 7);

        assert_eq!(Fp::reduce_u128(0), Fp::ZERO);
        assert_eq!(Fp::reduce_u128(MODULUS_P as u128), Fp::ZERO);
        assert_eq!(Fp::reduce_u128(2 * (MODULUS_P as u128)), Fp::ZERO);
        assert_eq!(Fp::reduce_u128(2 * (MODULUS_P as u128) + 1), Fp::ONE);
        assert_eq!(Fp::reduce_u128(3 * (MODULUS_P as u128)), Fp::ZERO);
        // 2^128 - 1 = 64 * (2^61)^2 - 1 == 63 (mod 2^61 - 1)
        assert_eq!(Fp::reduce_u128(u128::MAX).raw(), 63);
        assert_eq!(
            Fp::reduce_u128(u128::MAX).raw(),
            (u128::MAX % (MODULUS_P as u128)) as u64
        );

        // Signed i128 edge cases including i128::MIN
        assert_eq!(Fp::reduce_i128(0), Fp::ZERO);
        assert_eq!(Fp::reduce_i128(-(MODULUS_P as i128)), Fp::ZERO);
        assert_eq!(Fp::reduce_i128(-1).raw(), MODULUS_P - 1);
        assert_eq!(
            Fp::reduce_i128(i128::MIN).add(Fp::reduce_u128(1u128 << 127)),
            Fp::ZERO
        );
    }

    #[test]
    fn signed_bijection_and_overflow_rejection() {
        for &v in &[
            0i64,
            1,
            -1,
            127,
            -128,
            i32::MAX as i64,
            i32::MIN as i64,
            MAX_SAFE_SIGNED,
            -MAX_SAFE_SIGNED,
        ] {
            let fp = Fp::from_i64_checked(v, 0, 0).expect("value within safe signed range");
            assert_eq!(fp.to_signed_i64(), v);
        }

        // Values strictly beyond +/- (p - 1) / 2 MUST be rejected to prevent wraparound ambiguity
        assert!(matches!(
            Fp::from_i64_checked(MAX_SAFE_SIGNED + 1, 2, 3),
            Err(PoaiError::OutputOutOfFieldRange { row: 2, col: 3, .. })
        ));
        assert!(matches!(
            Fp::from_i64_checked(-MAX_SAFE_SIGNED - 1, 4, 5),
            Err(PoaiError::OutputOutOfFieldRange { row: 4, col: 5, .. })
        ));
        assert!(Fp::from_i64_checked(i64::MAX, 0, 0).is_err());
        assert!(Fp::from_i64_checked(i64::MIN, 0, 0).is_err());
    }

    #[test]
    fn field_axioms_and_fermat_inverse() {
        let samples = [
            1u64,
            2,
            3,
            42,
            1_000_000_007,
            (1u64 << 60) - 1,
            MODULUS_P - 2,
            MODULUS_P - 1,
        ];
        assert_eq!(Fp::ZERO.inv(), None);
        for &a_raw in &samples {
            let a = Fp::from_canonical(a_raw).unwrap();
            let a_inv = a.inv().expect("non-zero element has inverse");
            assert_eq!(a.mul(a_inv), Fp::ONE);
            assert_eq!(a.add(a.neg()), Fp::ZERO);
            assert_eq!(a.sub(a), Fp::ZERO);
            for &b_raw in &samples {
                let b = Fp::from_canonical(b_raw).unwrap();
                let expected = (((a_raw as u128) * (b_raw as u128)) % (MODULUS_P as u128)) as u64;
                assert_eq!(a.mul(b).raw(), expected);
                assert_eq!(a.add(b), b.add(a));
                assert_eq!(a.mul(b), b.mul(a));
            }
        }
    }
}
