// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Clean-room cryptographic primitives for `hanzo-fedverify` (zero external dependencies).
//!
//! Provides:
//! - FIPS 180-4 SHA-256 and SHA-512 with domain-separation helpers.
//! - RFC 8032 Ed25519 signing and public-key verification over Curve25519
//!   (twisted Edwards curve `-x^2 + y^2 = 1 + d*x^2*y^2` over `F_{2^255 - 19}`).
//! - Constant-time byte comparison, lowercase hex codecs, binary Merkle roots,
//!   and a deterministic `SplitMix64` PRNG for reproducible simulations.

use std::fmt;

/// 32-byte SHA-256 digest.
pub type Digest32 = [u8; 32];

/// 32-byte Ed25519 public key (compressed Edwards-y point).
pub type PublicKeyBytes = [u8; 32];

/// 64-byte RFC 8032 Ed25519 signature (`R || S`).
pub type SignatureBytes = [u8; 64];

// ============================================================================
// Constant-time comparison & hex codecs
// ============================================================================

/// Constant-time equality check over fixed-size or equal-length byte slices.
#[inline]
#[must_use]
pub fn ct_eq_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Encodes bytes as lowercase hexadecimal.
#[must_use]
pub fn hex_encode(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(LUT[(b >> 4) as usize] as char);
        out.push(LUT[(b & 0x0f) as usize] as char);
    }
    out
}

/// Decodes a hexadecimal string into raw bytes.
pub fn hex_decode(s: &str) -> Result<Vec<u8>, CryptoError> {
    let raw = s.as_bytes();
    if !raw.len().is_multiple_of(2) {
        return Err(CryptoError::InvalidHexLength(raw.len()));
    }
    let mut out = Vec::with_capacity(raw.len() / 2);
    for pair in raw.as_chunks::<2>().0 {
        let hi = hex_nibble(pair[0])?;
        let lo = hex_nibble(pair[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, CryptoError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        other => Err(CryptoError::InvalidHexChar(other)),
    }
}

/// Cryptographic error codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoError {
    /// Hex string has odd length.
    InvalidHexLength(usize),
    /// Hex string contains a non-hex byte.
    InvalidHexChar(u8),
    /// Public key is not a valid canonical Curve25519 Edwards point.
    InvalidPublicKey,
    /// Signature encoding or mathematical verification failed.
    InvalidSignature,
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHexLength(n) => write!(f, "invalid hex length: {n}"),
            Self::InvalidHexChar(c) => write!(f, "invalid hex byte: 0x{c:02x}"),
            Self::InvalidPublicKey => write!(f, "invalid Ed25519 public key"),
            Self::InvalidSignature => write!(f, "invalid Ed25519 signature"),
        }
    }
}

impl std::error::Error for CryptoError {}

// ============================================================================
// FIPS 180-4 SHA-256
// ============================================================================

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Streaming FIPS 180-4 SHA-256 hasher.
#[derive(Debug, Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_bits: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Creates a new SHA-256 hasher initialized to the FIPS 180-4 IV.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: SHA256_H0,
            buf: [0u8; 64],
            buf_len: 0,
            total_bits: 0,
        }
    }

    /// Feeds bytes into the hasher.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bits = self.total_bits.wrapping_add((data.len() as u64) << 3);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                Self::compress(&mut self.state, &block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            Self::compress(&mut self.state, &block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    /// Feeds a length-prefixed (`u64` LE) byte slice for unambiguous domain framing.
    pub fn update_len_prefixed(&mut self, bytes: &[u8]) {
        self.update(&(bytes.len() as u64).to_le_bytes());
        self.update(bytes);
    }

    /// Feeds a canonical `f64` value (normalizing `-0.0` to `+0.0`).
    pub fn update_f64(&mut self, v: f64) {
        let canonical = if v == 0.0 { 0.0f64 } else { v };
        self.update(&canonical.to_le_bytes());
    }

    /// Finalizes the hash and returns the 32-byte digest.
    #[must_use]
    pub fn finalize(mut self) -> Digest32 {
        let bit_len = self.total_bits;
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            for b in &mut self.buf[self.buf_len..64] {
                *b = 0;
            }
            let block = self.buf;
            Self::compress(&mut self.state, &block);
            self.buf = [0u8; 64];
            self.buf_len = 0;
        }
        for b in &mut self.buf[self.buf_len..56] {
            *b = 0;
        }
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        Self::compress(&mut self.state, &block);

        let mut out = [0u8; 32];
        for (i, &w) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }

    fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
}

/// Computes the raw SHA-256 digest of `data`.
#[must_use]
pub fn sha256(data: &[u8]) -> Digest32 {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

/// Computes a domain-separated SHA-256 digest.
#[must_use]
pub fn sha256_domain(domain: &[u8], payload: &[u8]) -> Digest32 {
    let mut h = Sha256::new();
    h.update_len_prefixed(domain);
    h.update_len_prefixed(payload);
    h.finalize()
}

/// Computes a canonical domain-separated binary Merkle root over leaf digests.
///
/// An empty leaf set returns `SHA-256("hanzo.fedverify.merkle.empty.v1")`.
#[must_use]
pub fn merkle_root(leaves: &[Digest32]) -> Digest32 {
    if leaves.is_empty() {
        return sha256(b"hanzo.fedverify.merkle.empty.v1");
    }
    let mut level: Vec<Digest32> = leaves
        .iter()
        .map(|leaf| {
            let mut h = Sha256::new();
            h.update(b"hanzo.fedverify.merkle.leaf.v1");
            h.update(leaf);
            h.finalize()
        })
        .collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let left = level[i];
            let right = if i + 1 < level.len() {
                level[i + 1]
            } else {
                level[i]
            };
            let mut h = Sha256::new();
            h.update(b"hanzo.fedverify.merkle.node.v1");
            h.update(&left);
            h.update(&right);
            next.push(h.finalize());
            i += 2;
        }
        level = next;
    }
    level[0]
}

// ============================================================================
// FIPS 180-4 SHA-512 (required by RFC 8032 Ed25519)
// ============================================================================

const SHA512_K: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

const SHA512_H0: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

/// Streaming FIPS 180-4 SHA-512 hasher.
#[derive(Debug, Clone)]
pub struct Sha512 {
    state: [u64; 8],
    buf: [u8; 128],
    buf_len: usize,
    total_bits: u128,
}

impl Default for Sha512 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha512 {
    /// Creates a new SHA-512 hasher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: SHA512_H0,
            buf: [0u8; 128],
            buf_len: 0,
            total_bits: 0,
        }
    }

    /// Feeds bytes into the hasher.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bits = self.total_bits.wrapping_add((data.len() as u128) << 3);
        if self.buf_len > 0 {
            let take = (128 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 128 {
                let block = self.buf;
                Self::compress(&mut self.state, &block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 128 {
            let mut block = [0u8; 128];
            block.copy_from_slice(&data[..128]);
            Self::compress(&mut self.state, &block);
            data = &data[128..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    /// Finalizes the hash and returns the 64-byte digest.
    #[must_use]
    pub fn finalize(mut self) -> [u8; 64] {
        let bit_len = self.total_bits;
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 112 {
            for b in &mut self.buf[self.buf_len..128] {
                *b = 0;
            }
            let block = self.buf;
            Self::compress(&mut self.state, &block);
            self.buf = [0u8; 128];
            self.buf_len = 0;
        }
        for b in &mut self.buf[self.buf_len..112] {
            *b = 0;
        }
        self.buf[112..128].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        Self::compress(&mut self.state, &block);

        let mut out = [0u8; 64];
        for (i, &w) in self.state.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&w.to_be_bytes());
        }
        out
    }

    fn compress(state: &mut [u64; 8], block: &[u8; 128]) {
        let mut w = [0u64; 80];
        for i in 0..16 {
            w[i] = u64::from_be_bytes([
                block[i * 8],
                block[i * 8 + 1],
                block[i * 8 + 2],
                block[i * 8 + 3],
                block[i * 8 + 4],
                block[i * 8 + 5],
                block[i * 8 + 6],
                block[i * 8 + 7],
            ]);
        }
        for i in 16..80 {
            let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
            let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for i in 0..80 {
            let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA512_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
}

/// Computes the raw SHA-512 digest of `data`.
#[must_use]
pub fn sha512(data: &[u8]) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(data);
    h.finalize()
}

// ============================================================================
// Pure-Rust RFC 8032 Ed25519 (F_{2^255 - 19}, radix 2^51)
// ============================================================================

const MASK51: u64 = (1u64 << 51) - 1;

/// Field element in `F_p` where `p = 2^255 - 19`, represented in 5 radix-`2^51` limbs.
#[derive(Clone, Copy, Debug)]
struct Fe([u64; 5]);

impl Fe {
    const ZERO: Self = Self([0, 0, 0, 0, 0]);
    const ONE: Self = Self([1, 0, 0, 0, 0]);

    /// Edwards curve constant `d = -121665 / 121666 mod p`.
    const D: Self = Self([
        0x00034dca135978a3,
        0x0001a8283b156ebd,
        0x0005e7a26001c029,
        0x000739c663a03cbb,
        0x00052036cee2b6ff,
    ]);

    /// `2 * d mod p`.
    const D2: Self = Self([
        0x00069b9426b2f159,
        0x00035050762add7a,
        0x0003cf44c0038052,
        0x0006738cc7407977,
        0x0002406d9dc56dff,
    ]);

    /// `sqrt(-1) = 2^((p-1)/4) mod p`.
    const SQRT_M1: Self = Self([
        0x00061b274a0ea0b0,
        0x0000d5a5fc8f189d,
        0x0007ef5e9cbd0c60,
        0x00078595a6804c9e,
        0x0002b8324804fc1d,
    ]);

    #[inline]
    fn from_bytes(bytes: &[u8; 32]) -> Self {
        let load64 = |offset: usize| -> u64 {
            u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ])
        };
        let l0 = load64(0) & MASK51;
        let l1 = (load64(6) >> 3) & MASK51;
        let l2 = (load64(12) >> 6) & MASK51;
        let l3 = (load64(19) >> 1) & MASK51;
        let l4 = (load64(24) >> 12) & MASK51;
        Self([l0, l1, l2, l3, l4])
    }

    #[inline]
    fn reduce(mut l: [u64; 5]) -> Self {
        let c0 = l[0] >> 51;
        l[0] &= MASK51;
        l[1] += c0;
        let c1 = l[1] >> 51;
        l[1] &= MASK51;
        l[2] += c1;
        let c2 = l[2] >> 51;
        l[2] &= MASK51;
        l[3] += c2;
        let c3 = l[3] >> 51;
        l[3] &= MASK51;
        l[4] += c3;
        let c4 = l[4] >> 51;
        l[4] &= MASK51;
        l[0] += c4 * 19;
        let c0b = l[0] >> 51;
        l[0] &= MASK51;
        l[1] += c0b;
        Self(l)
    }

    fn to_bytes(self) -> [u8; 32] {
        let mut h = Self::reduce(self.0).0;
        // Compute q = floor((h + 19) / 2^255) to conditionally subtract p = 2^255 - 19.
        let mut q = (h[0] + 19) >> 51;
        q = (h[1] + q) >> 51;
        q = (h[2] + q) >> 51;
        q = (h[3] + q) >> 51;
        q = (h[4] + q) >> 51;

        h[0] += 19 * q;
        let c0 = h[0] >> 51;
        h[0] &= MASK51;
        h[1] += c0;
        let c1 = h[1] >> 51;
        h[1] &= MASK51;
        h[2] += c1;
        let c2 = h[2] >> 51;
        h[2] &= MASK51;
        h[3] += c2;
        let c3 = h[3] >> 51;
        h[3] &= MASK51;
        h[4] += c3;
        h[4] &= MASK51;

        let mut out = [0u8; 32];
        let w0 = h[0] | (h[1] << 51);
        let w1 = (h[1] >> 13) | (h[2] << 38);
        let w2 = (h[2] >> 26) | (h[3] << 25);
        let w3 = (h[3] >> 39) | (h[4] << 12);
        out[0..8].copy_from_slice(&w0.to_le_bytes());
        out[8..16].copy_from_slice(&w1.to_le_bytes());
        out[16..24].copy_from_slice(&w2.to_le_bytes());
        out[24..32].copy_from_slice(&w3.to_le_bytes());
        out
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::reduce([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
        ])
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        // Add 2p so every limb remains strictly positive before reduction.
        const P0_2: u64 = 2 * (MASK51 - 18);
        const PI_2: u64 = 2 * MASK51;
        let a = Self::reduce(self.0).0;
        let b = Self::reduce(rhs.0).0;
        Self::reduce([
            (a[0] + P0_2) - b[0],
            (a[1] + PI_2) - b[1],
            (a[2] + PI_2) - b[2],
            (a[3] + PI_2) - b[3],
            (a[4] + PI_2) - b[4],
        ])
    }

    #[inline]
    fn neg(self) -> Self {
        Self::ZERO.sub(self)
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.0;
        let b = rhs.0;
        let a0 = u128::from(a[0]);
        let a1 = u128::from(a[1]);
        let a2 = u128::from(a[2]);
        let a3 = u128::from(a[3]);
        let a4 = u128::from(a[4]);
        let b0 = u128::from(b[0]);
        let b1 = u128::from(b[1]);
        let b2 = u128::from(b[2]);
        let b3 = u128::from(b[3]);
        let b4 = u128::from(b[4]);

        let b1_19 = b1 * 19;
        let b2_19 = b2 * 19;
        let b3_19 = b3 * 19;
        let b4_19 = b4 * 19;

        let mut c0 = a0 * b0 + a1 * b4_19 + a2 * b3_19 + a3 * b2_19 + a4 * b1_19;
        let mut c1 = a0 * b1 + a1 * b0 + a2 * b4_19 + a3 * b3_19 + a4 * b2_19;
        let mut c2 = a0 * b2 + a1 * b1 + a2 * b0 + a3 * b4_19 + a4 * b3_19;
        let mut c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + a4 * b4_19;
        let mut c4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;

        let mask = u128::from(MASK51);
        c1 += c0 >> 51;
        c0 &= mask;
        c2 += c1 >> 51;
        c1 &= mask;
        c3 += c2 >> 51;
        c2 &= mask;
        c4 += c3 >> 51;
        c3 &= mask;
        c0 += (c4 >> 51) * 19;
        c4 &= mask;
        c1 += c0 >> 51;
        c0 &= mask;

        Self([c0 as u64, c1 as u64, c2 as u64, c3 as u64, c4 as u64])
    }

    #[inline]
    fn square(self) -> Self {
        self.mul(self)
    }

    fn pow2k(mut self, k: usize) -> Self {
        for _ in 0..k {
            self = self.square();
        }
        self
    }

    /// Computes `(self^(2^252 - 3), self^(2^255 - 21))` via standard addition chain.
    fn pow22523_and_inv(self) -> (Self, Self) {
        let t0 = self.square();
        let t1 = t0.pow2k(2);
        let t1 = self.mul(t1);
        let t0 = t0.mul(t1);
        let t2 = t0.square();
        let t1 = t1.mul(t2);
        let t2 = t1.pow2k(5);
        let t1 = t2.mul(t1);
        let t2 = t1.pow2k(10);
        let t2 = t2.mul(t1);
        let t3 = t2.pow2k(20);
        let t2 = t3.mul(t2);
        let t2 = t2.pow2k(10);
        let t1 = t2.mul(t1);
        let t2 = t1.pow2k(50);
        let t2 = t2.mul(t1);
        let t3 = t2.pow2k(100);
        let t2 = t3.mul(t2);
        let t2 = t2.pow2k(50);
        let t1 = t2.mul(t1); // 2^250 - 1
        let pow22523 = t1.pow2k(2).mul(self); // 2^252 - 3
        let inv = t1.pow2k(5).mul(t0); // 2^255 - 32 + 11 = 2^255 - 21 = p - 2
        (pow22523, inv)
    }

    fn invert(self) -> Self {
        self.pow22523_and_inv().1
    }

    fn pow22523(self) -> Self {
        self.pow22523_and_inv().0
    }

    fn is_zero(self) -> bool {
        ct_eq_bytes(&self.to_bytes(), &[0u8; 32])
    }

    fn is_negative(self) -> bool {
        (self.to_bytes()[0] & 1) == 1
    }

    fn ct_eq(self, other: Self) -> bool {
        ct_eq_bytes(&self.to_bytes(), &other.to_bytes())
    }
}

/// Extended twisted Edwards point `(X : Y : Z : T)` where `x = X/Z`, `y = Y/Z`, `x*y = T/Z`.
#[derive(Clone, Copy, Debug)]
struct ExtPoint {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

impl ExtPoint {
    const IDENTITY: Self = Self {
        x: Fe::ZERO,
        y: Fe::ONE,
        z: Fe::ONE,
        t: Fe::ZERO,
    };

    /// RFC 8032 Ed25519 base point `B`.
    fn basepoint() -> Self {
        // y = 4/5 mod p, positive x (even LSB).
        let mut y_bytes = [0u8; 32];
        // 4/5 mod (2^255 - 19): 0x58666666...66666666 in little-endian
        y_bytes[0] = 0x58;
        for b in &mut y_bytes[1..32] {
            *b = 0x66;
        }
        Self::from_bytes(&y_bytes).expect("RFC 8032 basepoint must decompress")
    }

    fn add(self, other: Self) -> Self {
        let a = self.y.sub(self.x).mul(other.y.sub(other.x));
        let b = self.y.add(self.x).mul(other.y.add(other.x));
        let c = self.t.mul(Fe::D2).mul(other.t);
        let d = self.z.mul(other.z);
        let d = d.add(d);
        let e = b.sub(a);
        let f = d.sub(c);
        let g = d.add(c);
        let h = b.add(a);
        Self {
            x: e.mul(f),
            y: g.mul(h),
            z: f.mul(g),
            t: e.mul(h),
        }
    }

    fn double(self) -> Self {
        let a = self.x.square();
        let b = self.y.square();
        let c = self.z.square();
        let c = c.add(c);
        let h = a.add(b);
        let x_plus_y = self.x.add(self.y);
        let e = h.sub(x_plus_y.square());
        let g = a.sub(b);
        let f = c.add(g);
        Self {
            x: e.mul(f),
            y: g.mul(h),
            z: f.mul(g),
            t: e.mul(h),
        }
    }

    fn scalar_mul(self, scalar: &[u8; 32]) -> Self {
        let mut acc = Self::IDENTITY;
        let mut base = self;
        for &byte in scalar {
            for bit in 0..8 {
                if (byte >> bit) & 1 == 1 {
                    acc = acc.add(base);
                }
                base = base.double();
            }
        }
        acc
    }

    fn to_bytes(self) -> [u8; 32] {
        let z_inv = self.z.invert();
        let x = self.x.mul(z_inv);
        let y = self.y.mul(z_inv);
        let mut out = y.to_bytes();
        if x.is_negative() {
            out[31] |= 0x80;
        }
        out
    }

    fn from_bytes(bytes: &[u8; 32]) -> Result<Self, CryptoError> {
        let sign = (bytes[31] >> 7) & 1 == 1;
        let mut y_bytes = *bytes;
        y_bytes[31] &= 0x7f;

        // RFC 8032 strict canonical y check: reject y >= p = 2^255 - 19.
        let y = Fe::from_bytes(&y_bytes);
        if !ct_eq_bytes(&y.to_bytes(), &y_bytes) {
            return Err(CryptoError::InvalidPublicKey);
        }

        let y2 = y.square();
        let u = y2.sub(Fe::ONE);
        let v = Fe::D.mul(y2).add(Fe::ONE);
        let v3 = v.square().mul(v);
        let v7 = v3.square().mul(v);
        let mut x = u.mul(v3).mul(u.mul(v7).pow22523());

        let vx2 = v.mul(x.square());
        if !vx2.ct_eq(u) {
            if vx2.ct_eq(u.neg()) {
                x = x.mul(Fe::SQRT_M1);
            } else {
                return Err(CryptoError::InvalidPublicKey);
            }
        }
        if x.is_zero() && sign {
            return Err(CryptoError::InvalidPublicKey);
        }
        if x.is_negative() != sign {
            x = x.neg();
        }
        Ok(Self {
            x,
            y,
            z: Fe::ONE,
            t: x.mul(y),
        })
    }
}

/// Group order `L = 2^252 + 27742317777372353535851937790883648493` in little-endian `u64` limbs.
const ED25519_L: [u64; 4] = [
    0x5812631a5cf5d3ed,
    0x14def9dea2f79cd6,
    0x0000000000000000,
    0x1000000000000000,
];

fn u256_lt(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

/// Reduces a 512-bit little-endian integer modulo `L` using exact shift-and-subtract.
fn scalar_reduce_64(bytes: &[u8; 64]) -> [u8; 32] {
    // Load 512-bit value as little-endian u64 limbs.
    let mut num = [0u64; 8];
    for i in 0..8 {
        num[i] = u64::from_le_bytes([
            bytes[i * 8],
            bytes[i * 8 + 1],
            bytes[i * 8 + 2],
            bytes[i * 8 + 3],
            bytes[i * 8 + 4],
            bytes[i * 8 + 5],
            bytes[i * 8 + 6],
            bytes[i * 8 + 7],
        ]);
    }
    let mut rem = [0u64; 5];
    for bit_idx in (0..512).rev() {
        // rem <<= 1
        let mut carry = (num[bit_idx / 64] >> (bit_idx % 64)) & 1;
        for limb in &mut rem {
            let next_carry = *limb >> 63;
            *limb = (*limb << 1) | carry;
            carry = next_carry;
        }
        // If rem >= L, rem -= L
        let ge = rem[4] > 0 || !u256_lt(&[rem[0], rem[1], rem[2], rem[3]], &ED25519_L);
        if ge {
            let mut borrow = 0u64;
            for i in 0..4 {
                let (d1, b1) = rem[i].overflowing_sub(ED25519_L[i]);
                let (d2, b2) = d1.overflowing_sub(borrow);
                rem[i] = d2;
                borrow = u64::from(b1 || b2);
            }
            rem[4] = rem[4].wrapping_sub(borrow);
        }
    }
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..i * 8 + 8].copy_from_slice(&rem[i].to_le_bytes());
    }
    out
}

/// Computes `(r + k * a) mod L` for 256-bit little-endian scalars.
fn scalar_mul_add_mod_l(r: &[u8; 32], k: &[u8; 32], a: &[u8; 32]) -> [u8; 32] {
    let mut kl = [0u64; 4];
    let mut al = [0u64; 4];
    let mut rl = [0u64; 4];
    for i in 0..4 {
        kl[i] = u64::from_le_bytes(k[i * 8..i * 8 + 8].try_into().expect("8 bytes"));
        al[i] = u64::from_le_bytes(a[i * 8..i * 8 + 8].try_into().expect("8 bytes"));
        rl[i] = u64::from_le_bytes(r[i * 8..i * 8 + 8].try_into().expect("8 bytes"));
    }
    let mut acc = [0u128; 8];
    for i in 0..4 {
        acc[i] = u128::from(rl[i]);
    }
    for i in 0..4 {
        for j in 0..4 {
            let prod = u128::from(kl[i]) * u128::from(al[j]);
            let lo = prod & 0xffff_ffff_ffff_ffff;
            let hi = prod >> 64;
            acc[i + j] += lo;
            acc[i + j + 1] += hi;
        }
    }
    let mut prod_bytes = [0u8; 64];
    let mut carry = 0u128;
    for i in 0..8 {
        let v = acc[i] + carry;
        prod_bytes[i * 8..i * 8 + 8].copy_from_slice(&(v as u64).to_le_bytes());
        carry = v >> 64;
    }
    scalar_reduce_64(&prod_bytes)
}

fn clamp_scalar(mut s: [u8; 32]) -> [u8; 32] {
    s[0] &= 248;
    s[31] &= 127;
    s[31] |= 64;
    s
}

/// Checks whether a 32-byte little-endian scalar `s` is strictly less than `L`.
fn scalar_is_canonical(s: &[u8; 32]) -> bool {
    let mut limbs = [0u64; 4];
    for i in 0..4 {
        limbs[i] = u64::from_le_bytes(s[i * 8..i * 8 + 8].try_into().expect("8 bytes"));
    }
    u256_lt(&limbs, &ED25519_L)
}

/// RFC 8032 Ed25519 signing keypair derived from a 32-byte secret seed.
#[derive(Clone)]
pub struct Ed25519Keypair {
    seed: [u8; 32],
    scalar: [u8; 32],
    prefix: [u8; 32],
    public_key: PublicKeyBytes,
}

impl fmt::Debug for Ed25519Keypair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ed25519Keypair")
            .field("public_key", &hex_encode(&self.public_key))
            .finish_non_exhaustive()
    }
}

impl Ed25519Keypair {
    /// Derives an RFC 8032 Ed25519 keypair from a 32-byte secret seed.
    #[must_use]
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let h = sha512(&seed);
        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&h[0..32]);
        scalar = clamp_scalar(scalar);
        let mut prefix = [0u8; 32];
        prefix.copy_from_slice(&h[32..64]);
        let public_key = ExtPoint::basepoint().scalar_mul(&scalar).to_bytes();
        Self {
            seed,
            scalar,
            prefix,
            public_key,
        }
    }

    /// Deterministically derives a simulation keypair from a domain label and integer seed.
    #[must_use]
    pub fn from_label(label: &str, seed_u64: u64) -> Self {
        let mut h = Sha256::new();
        h.update(b"hanzo.fedverify.keygen.v1");
        h.update_len_prefixed(label.as_bytes());
        h.update(&seed_u64.to_le_bytes());
        Self::from_seed(h.finalize())
    }

    /// Returns the 32-byte secret seed.
    #[must_use]
    pub fn seed(&self) -> &[u8; 32] {
        &self.seed
    }

    /// Returns the 32-byte compressed Ed25519 public key.
    #[must_use]
    pub fn public_key(&self) -> PublicKeyBytes {
        self.public_key
    }

    /// Signs `message` deterministically according to RFC 8032 Section 5.1.6.
    #[must_use]
    pub fn sign(&self, message: &[u8]) -> SignatureBytes {
        let mut h_r = Sha512::new();
        h_r.update(&self.prefix);
        h_r.update(message);
        let r = scalar_reduce_64(&h_r.finalize());

        let r_point = ExtPoint::basepoint().scalar_mul(&r).to_bytes();

        let mut h_k = Sha512::new();
        h_k.update(&r_point);
        h_k.update(&self.public_key);
        h_k.update(message);
        let k = scalar_reduce_64(&h_k.finalize());

        let s = scalar_mul_add_mod_l(&r, &k, &self.scalar);

        let mut sig = [0u8; 64];
        sig[0..32].copy_from_slice(&r_point);
        sig[32..64].copy_from_slice(&s);
        sig
    }
}

/// Verifies an RFC 8032 Ed25519 signature over `message` under `public_key`.
///
/// Enforces canonical scalar range `S < L` and canonical curve point encoding
/// for both `public_key` and `R` to prevent signature malleability.
pub fn ed25519_verify(
    public_key: &PublicKeyBytes,
    message: &[u8],
    signature: &SignatureBytes,
) -> Result<(), CryptoError> {
    let a_point = ExtPoint::from_bytes(public_key)?;
    let mut r_bytes = [0u8; 32];
    r_bytes.copy_from_slice(&signature[0..32]);
    let r_point = ExtPoint::from_bytes(&r_bytes).map_err(|_| CryptoError::InvalidSignature)?;

    let mut s_bytes = [0u8; 32];
    s_bytes.copy_from_slice(&signature[32..64]);
    if !scalar_is_canonical(&s_bytes) {
        return Err(CryptoError::InvalidSignature);
    }

    let mut h_k = Sha512::new();
    h_k.update(&r_bytes);
    h_k.update(public_key);
    h_k.update(message);
    let k = scalar_reduce_64(&h_k.finalize());

    let lhs = ExtPoint::basepoint().scalar_mul(&s_bytes);
    let rhs = r_point.add(a_point.scalar_mul(&k));
    if ct_eq_bytes(&lhs.to_bytes(), &rhs.to_bytes()) {
        Ok(())
    } else {
        Err(CryptoError::InvalidSignature)
    }
}

// ============================================================================
// Deterministic SplitMix64 PRNG (for reproducible simulations only)
// ============================================================================

/// Deterministic SplitMix64 PRNG for reproducible federated experiments.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a new SplitMix64 instance from `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generates the next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generates a uniform `f64` in `[0.0, 1.0)`.
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Generates a deterministic index in `0..bound`.
    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }

    /// Generates a zero-mean unit-variance approximate Gaussian via Irwin-Hall (sum of 12 uniforms - 6).
    pub fn next_normal(&mut self) -> f64 {
        let mut sum = 0.0;
        for _ in 0..12 {
            sum += self.next_f64();
        }
        sum - 6.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_nist_vectors() {
        assert_eq!(
            hex_encode(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex_encode(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha512_nist_vector() {
        assert_eq!(
            hex_encode(&sha512(b"abc")),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );
    }

    #[test]
    fn ed25519_rfc8032_test_vector_1_empty_message() {
        let seed_vec =
            hex_decode("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
                .expect("valid hex");
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&seed_vec);

        let kp = Ed25519Keypair::from_seed(seed);
        assert_eq!(
            hex_encode(&kp.public_key()),
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
        );

        let sig = kp.sign(b"");
        assert_eq!(
            hex_encode(&sig),
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        );
        ed25519_verify(&kp.public_key(), b"", &sig).expect("RFC 8032 vector 1 verifies");

        let mut bad_sig = sig;
        bad_sig[0] ^= 0x01;
        assert!(ed25519_verify(&kp.public_key(), b"", &bad_sig).is_err());
    }

    #[test]
    fn ed25519_rfc8032_test_vector_2_one_byte() {
        let seed_vec =
            hex_decode("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb")
                .expect("valid hex");
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&seed_vec);

        let kp = Ed25519Keypair::from_seed(seed);
        assert_eq!(
            hex_encode(&kp.public_key()),
            "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"
        );

        let sig = kp.sign(&[0x72]);
        assert_eq!(
            hex_encode(&sig),
            "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
        );
        ed25519_verify(&kp.public_key(), &[0x72], &sig).expect("RFC 8032 vector 2 verifies");
        assert!(ed25519_verify(&kp.public_key(), &[0x73], &sig).is_err());
    }

    #[test]
    fn merkle_root_detects_single_leaf_mutation() {
        let l0 = sha256(b"leaf-0");
        let l1 = sha256(b"leaf-1");
        let l2 = sha256(b"leaf-2");
        let r_orig = merkle_root(&[l0, l1, l2]);
        let r_mut = merkle_root(&[l0, sha256(b"leaf-1-tampered"), l2]);
        assert_ne!(r_orig, r_mut);
    }
}
