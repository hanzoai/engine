// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Goldilocks prime field (`p = 2^64 - 2^32 + 1`), quadratic extension `F_{p^2}`,
//! radix-2 NTT, and multilinear / univariate polynomial primitives.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Goldilocks prime modulus: `p = 2^64 - 2^32 + 1 = 18_446_744_069_414_584_321`.
pub const MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

/// `u128` representation of [`MODULUS`].
pub const MODULUS_U128: u128 = MODULUS as u128;

/// Maximum signed magnitude `|x|` admitted by the injective signed-to-field bijection:
/// `(p - 1) / 2 = 9_223_372_034_707_292_160`.
pub const MAX_SIGNED_MAGNITUDE: i64 = ((MODULUS - 1) / 2) as i64;

/// Quadratic non-residue `W = 7` defining `F_{p^2} = F_p[u] / (u^2 - 7)`.
/// Because `7` is a primitive generator of `F_p*`, `7^((p - 1) / 2) = -1 (mod p)`.
pub const EXTENSION_NON_RESIDUE: u64 = 7;

/// Two-adicity of `p - 1 = 2^32 * (2^32 - 1)`.
pub const TWO_ADICITY: u32 = 32;

/// Primitive `2^32`-th root of unity in Goldilocks `F_p`: `7^(2^32 - 1) mod p`.
pub const TWO_ADIC_GENERATOR: u64 = 1_753_635_133_440_165_772;

/// Element of the Goldilocks prime field `F_p` (`p = 2^64 - 2^32 + 1`).
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fp(u64);

impl Fp {
    /// Additive identity `0`.
    pub const ZERO: Self = Self(0);
    /// Multiplicative identity `1`.
    pub const ONE: Self = Self(1);
    /// Element `2`.
    pub const TWO: Self = Self(2);
    /// Element `3`.
    pub const THREE: Self = Self(3);
    /// Element `7` (primitive generator and extension non-residue).
    pub const SEVEN: Self = Self(7);

    /// Constructs a canonical field element from a `u64` by reducing modulo `p`.
    #[inline]
    pub const fn new(val: u64) -> Self {
        if val >= MODULUS {
            Self(val - MODULUS)
        } else {
            Self(val)
        }
    }

    /// Constructs a canonical field element from a `u128` by exact reduction modulo `p`.
    #[inline]
    pub const fn from_u128(val: u128) -> Self {
        Self((val % MODULUS_U128) as u64)
    }

    /// Maps a signed `i64` in `[-MAX_SIGNED_MAGNITUDE, MAX_SIGNED_MAGNITUDE]`
    /// injectively into `F_p`. Returns `None` if `|x| > MAX_SIGNED_MAGNITUDE`,
    /// closing modular wraparound attacks.
    #[inline]
    pub fn from_signed_checked(val: i64) -> Option<Self> {
        if val == i64::MIN {
            return None;
        }
        let mag = val.unsigned_abs();
        if mag > MAX_SIGNED_MAGNITUDE as u64 {
            return None;
        }
        if val >= 0 {
            Some(Self(mag))
        } else {
            Some(Self(MODULUS - mag))
        }
    }

    /// Converts a canonical field element back into its balanced signed integer
    /// representative in `[-(p - 1) / 2, (p - 1) / 2]`.
    #[inline]
    pub const fn to_signed(self) -> i64 {
        if self.0 <= MAX_SIGNED_MAGNITUDE as u64 {
            self.0 as i64
        } else {
            -((MODULUS - self.0) as i64)
        }
    }

    /// Returns the canonical `u64` representative in `[0, p - 1]`.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Serializes the canonical representative as 8 little-endian bytes.
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Deserializes a canonical field element from 8 little-endian bytes,
    /// rejecting non-canonical values `>= MODULUS`.
    #[inline]
    pub const fn from_le_bytes_canonical(bytes: [u8; 8]) -> Option<Self> {
        let v = u64::from_le_bytes(bytes);
        if v < MODULUS {
            Some(Self(v))
        } else {
            None
        }
    }

    /// Returns `true` if `self == 0`.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Computes `self^exp mod p` via square-and-multiply.
    #[inline]
    pub fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut acc = Self::ONE;
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= base;
            }
            base *= base;
            exp >>= 1;
        }
        acc
    }

    /// Computes the multiplicative inverse in `F_p`, or `None` if `self == 0`.
    #[inline]
    pub fn inv(self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            Some(self.pow(MODULUS - 2))
        }
    }

    /// Returns the primitive `2^log_n`-th root of unity in `F_p` for `log_n <= 32`.
    pub fn primitive_root_of_unity(log_n: u32) -> Option<Self> {
        if log_n > TWO_ADICITY {
            return None;
        }
        let mut root = Self(TWO_ADIC_GENERATOR);
        for _ in log_n..TWO_ADICITY {
            root *= root;
        }
        Some(root)
    }
}

impl fmt::Debug for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp({})", self.0)
    }
}

impl fmt::Display for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add for Fp {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let sum = self.0 as u128 + rhs.0 as u128;
        if sum >= MODULUS_U128 {
            Self((sum - MODULUS_U128) as u64)
        } else {
            Self(sum as u64)
        }
    }
}

impl AddAssign for Fp {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Fp {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            Self(self.0 - rhs.0)
        } else {
            Self(MODULUS - (rhs.0 - self.0))
        }
    }
}

impl SubAssign for Fp {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Fp {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        if self.0 == 0 {
            Self::ZERO
        } else {
            Self(MODULUS - self.0)
        }
    }
}

impl Mul for Fp {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_u128(self.0 as u128 * rhs.0 as u128)
    }
}

impl MulAssign for Fp {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Element of the quadratic extension field `F_{p^2} = F_p[u] / (u^2 - 7)`.
///
/// The field order is `p^2 ≈ 2^128`, providing ~128-bit Schwartz-Zippel and
/// Sumcheck soundness from a single challenge element.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct Fp2 {
    /// Constant coefficient `c0`.
    pub c0: Fp,
    /// Linear coefficient `c1` (coefficient of `u` where `u^2 = 7`).
    pub c1: Fp,
}

impl Fp2 {
    /// Additive identity `0 + 0u`.
    pub const ZERO: Self = Self {
        c0: Fp::ZERO,
        c1: Fp::ZERO,
    };
    /// Multiplicative identity `1 + 0u`.
    pub const ONE: Self = Self {
        c0: Fp::ONE,
        c1: Fp::ZERO,
    };

    /// Constructs `c0 + c1 * u`.
    #[inline]
    pub const fn new(c0: Fp, c1: Fp) -> Self {
        Self { c0, c1 }
    }

    /// Embeds a base field element `x` into `F_{p^2}` as `x + 0u`.
    #[inline]
    pub const fn from_base(c0: Fp) -> Self {
        Self { c0, c1: Fp::ZERO }
    }

    /// Embeds a `u64` into `F_{p^2}`.
    #[inline]
    pub const fn from_u64(v: u64) -> Self {
        Self::from_base(Fp::new(v))
    }

    /// Serializes as 16 little-endian bytes (`c0 || c1`).
    #[inline]
    pub fn to_le_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&self.c0.to_le_bytes());
        out[8..16].copy_from_slice(&self.c1.to_le_bytes());
        out
    }

    /// Deserializes a canonical `Fp2` from 16 little-endian bytes.
    #[inline]
    pub fn from_le_bytes_canonical(bytes: [u8; 16]) -> Option<Self> {
        let mut b0 = [0u8; 8];
        let mut b1 = [0u8; 8];
        b0.copy_from_slice(&bytes[0..8]);
        b1.copy_from_slice(&bytes[8..16]);
        Some(Self {
            c0: Fp::from_le_bytes_canonical(b0)?,
            c1: Fp::from_le_bytes_canonical(b1)?,
        })
    }

    /// Multiplies `self` by a base field scalar `s in F_p`.
    #[inline]
    pub fn mul_base(self, s: Fp) -> Self {
        Self {
            c0: self.c0 * s,
            c1: self.c1 * s,
        }
    }

    /// Computes the field norm `N(c0 + c1*u) = c0^2 - 7 * c1^2 in F_p`.
    #[inline]
    pub fn norm(self) -> Fp {
        self.c0 * self.c0 - Fp::SEVEN * (self.c1 * self.c1)
    }

    /// Computes the multiplicative inverse in `F_{p^2}`, or `None` if zero.
    #[inline]
    pub fn inv(self) -> Option<Self> {
        let n_inv = self.norm().inv()?;
        Some(Self {
            c0: self.c0 * n_inv,
            c1: -self.c1 * n_inv,
        })
    }

    /// Returns `true` if `self == 0`.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }
}

impl fmt::Debug for Fp2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp2({} + {}u)", self.c0.as_u64(), self.c1.as_u64())
    }
}

impl Add for Fp2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
        }
    }
}

impl AddAssign for Fp2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.c0 += rhs.c0;
        self.c1 += rhs.c1;
    }
}

impl Sub for Fp2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 - rhs.c0,
            c1: self.c1 - rhs.c1,
        }
    }
}

impl SubAssign for Fp2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.c0 -= rhs.c0;
        self.c1 -= rhs.c1;
    }
}

impl Neg for Fp2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            c0: -self.c0,
            c1: -self.c1,
        }
    }
}

impl Mul for Fp2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a0b0 = self.c0 * rhs.c0;
        let a1b1 = self.c1 * rhs.c1;
        let a0b1 = self.c0 * rhs.c1;
        let a1b0 = self.c1 * rhs.c0;
        Self {
            c0: a0b0 + Fp::SEVEN * a1b1,
            c1: a0b1 + a1b0,
        }
    }
}

impl MulAssign for Fp2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// In-place Cooley-Tukey radix-2 forward Number Theoretic Transform (NTT) over `F_p`.
///
/// Transforms polynomial coefficients `a[0..n]` into evaluations `P(omega^j)` for
/// `j = 0..n`, where `n` is a power of two (`n <= 2^32`).
pub fn ntt_forward_fp(values: &mut [Fp]) {
    let n = values.len();
    assert!(n.is_power_of_two(), "NTT length must be a power of two");
    if n <= 1 {
        return;
    }
    let log_n = n.trailing_zeros();

    // Bit-reversal permutation.
    for i in 0..n {
        let rev = i.reverse_bits() >> (usize::BITS - log_n);
        if i < rev {
            values.swap(i, rev);
        }
    }

    let mut len = 2usize;
    let mut stage = 1u32;
    while len <= n {
        let omega = Fp::primitive_root_of_unity(stage).expect("stage <= 32");
        let half = len / 2;
        for chunk in values.chunks_exact_mut(len) {
            let mut w = Fp::ONE;
            for j in 0..half {
                let u = chunk[j];
                let v = chunk[j + half] * w;
                chunk[j] = u + v;
                chunk[j + half] = u - v;
                w *= omega;
            }
        }
        len <<= 1;
        stage += 1;
    }
}

/// Encodes a degree-`< C` polynomial over `F_p` into a Reed-Solomon codeword of
/// length `L = C * blowup` via zero-padded forward NTT.
pub fn rs_encode_fp(coeffs: &[Fp], blowup: usize) -> Vec<Fp> {
    assert!(coeffs.len().is_power_of_two());
    assert!(blowup.is_power_of_two() && blowup >= 2);
    let codeword_len = coeffs.len() * blowup;
    let mut buf = vec![Fp::ZERO; codeword_len];
    buf[..coeffs.len()].copy_from_slice(coeffs);
    ntt_forward_fp(&mut buf);
    buf
}

/// Encodes a degree-`< C` polynomial over `F_{p^2}` into a Reed-Solomon codeword of
/// length `L = C * blowup` (exploiting `F_p`-linearity of NTT across `c0` and `c1`).
pub fn rs_encode_fp2(coeffs: &[Fp2], blowup: usize) -> Vec<Fp2> {
    assert!(coeffs.len().is_power_of_two());
    assert!(blowup.is_power_of_two() && blowup >= 2);
    let codeword_len = coeffs.len() * blowup;
    let mut c0 = vec![Fp::ZERO; codeword_len];
    let mut c1 = vec![Fp::ZERO; codeword_len];
    for (i, v) in coeffs.iter().enumerate() {
        c0[i] = v.c0;
        c1[i] = v.c1;
    }
    ntt_forward_fp(&mut c0);
    ntt_forward_fp(&mut c1);
    c0.into_iter()
        .zip(c1)
        .map(|(a, b)| Fp2::new(a, b))
        .collect()
}

/// Builds the full multilinear equality table `eq(r, x)` for all `x in {0,1}^k`
/// using LSB-first bit ordering (`bit_j = (x >> j) & 1`).
pub fn eq_eval_table(r: &[Fp2]) -> Vec<Fp2> {
    let mut table = vec![Fp2::ONE];
    for &ri in r {
        let one_minus = Fp2::ONE - ri;
        let mut next = Vec::with_capacity(table.len() * 2);
        for &val in &table {
            // Bit = 0 goes to index (2 * i) when built MSB-first, wait:
            // If we want bit j to be (x >> j) & 1:
            // At step j (from 0 to k-1), existing entries represent bits 0..j-1.
            // Placing bit j = 0 in the first half and bit j = 1 in the second half
            // means index = low_bits + (bit_j << j), which is EXACTLY LSB-first!
            next.push(val * one_minus);
        }
        for &val in &table {
            next.push(val * ri);
        }
        table = next;
    }
    table
}

/// Evaluates `eq(r, x)` at two points `r, x in F_{p^2}^k`.
pub fn eq_eval_point(r: &[Fp2], x: &[Fp2]) -> Fp2 {
    assert_eq!(r.len(), x.len());
    let mut acc = Fp2::ONE;
    for (&ri, &xi) in r.iter().zip(x.iter()) {
        let term = ri * xi + (Fp2::ONE - ri) * (Fp2::ONE - xi);
        acc *= term;
    }
    acc
}

/// Evaluates a degree-2 univariate polynomial given its values at `{0, 1, 2}` at point `r`.
pub fn eval_deg2_univariate(evals: [Fp2; 3], r: Fp2) -> Fp2 {
    let inv2 = Fp::TWO.inv().expect("2 is invertible");
    let one = Fp2::ONE;
    let two = Fp2::from_u64(2);

    let l0 = (r - one) * (r - two).mul_base(inv2);
    let l1 = -(r * (r - two));
    let l2 = r * (r - one).mul_base(inv2);

    evals[0] * l0 + evals[1] * l1 + evals[2] * l2
}

/// Evaluates a degree-3 univariate polynomial given its values at `{0, 1, 2, 3}` at point `r`.
pub fn eval_deg3_univariate(evals: [Fp2; 4], r: Fp2) -> Fp2 {
    let inv2 = Fp::TWO.inv().expect("2 is invertible");
    let inv6 = Fp::new(6).inv().expect("6 is invertible");
    let one = Fp2::ONE;
    let two = Fp2::from_u64(2);
    let three = Fp2::from_u64(3);

    let l0 = -((r - one) * (r - two) * (r - three)).mul_base(inv6);
    let l1 = (r * (r - two) * (r - three)).mul_base(inv2);
    let l2 = -(r * (r - one) * (r - three)).mul_base(inv2);
    let l3 = (r * (r - one) * (r - two)).mul_base(inv6);

    evals[0] * l0 + evals[1] * l1 + evals[2] * l2 + evals[3] * l3
}

/// Evaluates the multilinear extension of a base-field vector `v` (length `<= 2^r.len()`,
/// implicitly zero-padded) at point `r` (LSB-first bit order).
pub fn eval_mle_fp(values: &[Fp], r: &[Fp2]) -> Fp2 {
    let eq = eq_eval_table(r);
    let mut acc = Fp2::ZERO;
    for (val, &w) in values.iter().zip(eq.iter()) {
        acc += w.mul_base(*val);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goldilocks_generator_and_roots() {
        // Verify 7 is a quadratic non-residue mod p.
        let legendre = Fp::SEVEN.pow((MODULUS - 1) / 2);
        assert_eq!(legendre, -Fp::ONE);

        // Verify TWO_ADIC_GENERATOR has exact order 2^32.
        let g = Fp::primitive_root_of_unity(32).unwrap();
        assert_eq!(g.as_u64(), TWO_ADIC_GENERATOR);
        assert_eq!(g.pow(1u64 << 31), -Fp::ONE);
        assert_eq!(g.pow(1u64 << 32), Fp::ONE);

        // Verify order-8 root of unity.
        let w8 = Fp::primitive_root_of_unity(3).unwrap();
        assert_eq!(w8.pow(4), -Fp::ONE);
        assert_eq!(w8.pow(8), Fp::ONE);
    }

    #[test]
    fn fp2_arithmetic_and_inverse() {
        let a = Fp2::new(Fp::new(123456789), Fp::new(987654321));
        let b = Fp2::new(Fp::new(42), Fp::new(MODULUS - 17));
        let a_inv = a.inv().unwrap();
        assert_eq!(a * a_inv, Fp2::ONE);
        assert_eq!((a * b) * a_inv, b);
    }

    #[test]
    fn signed_bijection_and_range_check() {
        assert_eq!(
            Fp::from_signed_checked(MAX_SIGNED_MAGNITUDE)
                .unwrap()
                .to_signed(),
            MAX_SIGNED_MAGNITUDE
        );
        assert_eq!(
            Fp::from_signed_checked(-MAX_SIGNED_MAGNITUDE)
                .unwrap()
                .to_signed(),
            -MAX_SIGNED_MAGNITUDE
        );
        assert!(Fp::from_signed_checked(MAX_SIGNED_MAGNITUDE + 1).is_none());
        assert!(Fp::from_signed_checked(i64::MIN).is_none());
    }

    #[test]
    fn eq_table_matches_lsb_folding() {
        let r = [
            Fp2::new(Fp::new(11), Fp::new(3)),
            Fp2::new(Fp::new(19), Fp::new(5)),
            Fp2::new(Fp::new(29), Fp::new(7)),
        ];
        let vals: Vec<Fp> = (1..=8).map(Fp::new).collect();
        let mle = eval_mle_fp(&vals, &r);

        // Fold LSB-first manually.
        let mut table: Vec<Fp2> = vals.iter().map(|&x| Fp2::from_base(x)).collect();
        for &rk in &r {
            let mut next = Vec::with_capacity(table.len() / 2);
            for i in 0..table.len() / 2 {
                let u0 = table[2 * i];
                let u1 = table[2 * i + 1];
                next.push(u0 + rk * (u1 - u0));
            }
            table = next;
        }
        assert_eq!(table.len(), 1);
        assert_eq!(table[0], mle);
    }

    #[test]
    fn rs_encode_linearity_fp2() {
        let row0 = vec![Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4)];
        let row1 = vec![Fp::new(5), Fp::new(6), Fp::new(7), Fp::new(8)];
        let alpha = Fp2::new(Fp::new(13), Fp::new(29));
        let beta = Fp2::new(Fp::new(37), Fp::new(41));

        let c0 = rs_encode_fp(&row0, 8);
        let c1 = rs_encode_fp(&row1, 8);
        let comb_coeffs: Vec<Fp2> = row0
            .iter()
            .zip(row1.iter())
            .map(|(&a, &b)| alpha.mul_base(a) + beta.mul_base(b))
            .collect();
        let comb_codeword = rs_encode_fp2(&comb_coeffs, 8);

        for j in 0..comb_codeword.len() {
            let expected = alpha.mul_base(c0[j]) + beta.mul_base(c1[j]);
            assert_eq!(comb_codeword[j], expected);
        }
    }

    #[test]
    fn univariate_lagrange_interpolation() {
        let r = Fp2::new(Fp::new(101), Fp::new(202));
        // g2(x) = 3 + 5x + 7x^2
        let g2 = |x: Fp2| Fp2::from_u64(3) + Fp2::from_u64(5) * x + Fp2::from_u64(7) * x * x;
        let evals2 = [
            g2(Fp2::from_u64(0)),
            g2(Fp2::from_u64(1)),
            g2(Fp2::from_u64(2)),
        ];
        assert_eq!(eval_deg2_univariate(evals2, r), g2(r));

        // g3(x) = 2 + 4x + 6x^2 + 9x^3
        let g3 = |x: Fp2| {
            Fp2::from_u64(2)
                + Fp2::from_u64(4) * x
                + Fp2::from_u64(6) * x * x
                + Fp2::from_u64(9) * x * x * x
        };
        let evals3 = [
            g3(Fp2::from_u64(0)),
            g3(Fp2::from_u64(1)),
            g3(Fp2::from_u64(2)),
            g3(Fp2::from_u64(3)),
        ];
        assert_eq!(eval_deg3_univariate(evals3, r), g3(r));
    }
}
