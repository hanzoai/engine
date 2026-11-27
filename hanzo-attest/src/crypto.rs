//! Pure-Rust zero-dependency cryptographic primitives for `hanzo-attest`.
//!
//! Implements:
//! - FIPS 180-4 SHA-256 and SHA-512
//! - RFC 2104 HMAC-SHA256 and RFC 5869 HKDF-SHA256
//! - RFC 8032 Ed25519 signatures with strict canonical point and scalar (`S < L`) checks
//! - RFC 7748 X25519 Diffie-Hellman key agreement with all-zero low-order point rejection
//! - RFC 8439 ChaCha20-Poly1305 authenticated encryption with associated data (AEAD)
//! - Constant-time byte comparison and explicit secret zeroization helpers

use crate::error::{AttestError, ErrorCode, Result};

// ============================================================================
// Constant-time utilities, zeroization, and hex helpers
// ============================================================================

/// Constant-time comparison of two equal-length byte slices.
#[must_use]
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Overwrites a mutable byte slice with zeroes using a compiler fence to
/// discourage dead-store elimination without requiring `unsafe`.
pub fn zeroize_bytes(buf: &mut [u8]) {
    for b in buf.iter_mut() {
        *b = 0;
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// Encodes bytes as lowercase hexadecimal.
#[must_use]
pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Decodes a hexadecimal string into raw bytes.
pub fn hex_decode(s: &str) -> Result<Vec<u8>> {
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return Err(AttestError::new(
            ErrorCode::InvalidEncoding,
            "hex string length must be even",
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let (pairs, _) = bytes.as_chunks::<2>();
    for chunk in pairs {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

/// Decodes a hexadecimal string into a fixed-size byte array.
pub fn hex_decode_array<const N: usize>(s: &str) -> Result<[u8; N]> {
    let raw = hex_decode(s)?;
    if raw.len() != N {
        return Err(AttestError::new(
            ErrorCode::InvalidEncoding,
            format!(
                "expected {} bytes ({} hex chars), got {}",
                N,
                N * 2,
                raw.len()
            ),
        ));
    }
    let mut arr = [0u8; N];
    arr.copy_from_slice(&raw);
    Ok(arr)
}

fn hex_nibble(b: u8) -> Result<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(AttestError::new(
            ErrorCode::InvalidEncoding,
            format!("invalid hex character: 0x{:02x}", b),
        )),
    }
}

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

/// Incremental FIPS 180-4 SHA-256 hasher.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Creates a new SHA-256 hasher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    /// Absorbs additional input bytes.
    pub fn update(&mut self, data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        let mut offset = 0usize;

        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            offset += take;
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }

        while offset + 64 <= data.len() {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[offset..offset + 64]);
            self.compress(&block);
            offset += 64;
        }

        if offset < data.len() {
            let rem = data.len() - offset;
            self.buf[..rem].copy_from_slice(&data[offset..]);
            self.buf_len = rem;
        }
    }

    /// Finalizes the digest and returns the 32-byte SHA-256 output.
    #[must_use]
    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        if self.buf_len > 56 {
            for b in &mut self.buf[self.buf_len..64] {
                *b = 0;
            }
            let block = self.buf;
            self.compress(&block);
            self.buf_len = 0;
        }

        for b in &mut self.buf[self.buf_len..56] {
            *b = 0;
        }
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        self.compress(&block);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64]) {
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

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
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

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

/// Computes the FIPS 180-4 SHA-256 digest of `data`.
#[must_use]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

// ============================================================================
// FIPS 180-4 SHA-512
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

/// Incremental FIPS 180-4 SHA-512 hasher.
#[derive(Clone)]
pub struct Sha512 {
    state: [u64; 8],
    buf: [u8; 128],
    buf_len: usize,
    total_len: u128,
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
            state: [
                0x6a09e667f3bcc908,
                0xbb67ae8584caa73b,
                0x3c6ef372fe94f82b,
                0xa54ff53a5f1d36f1,
                0x510e527fade682d1,
                0x9b05688c2b3e6c1f,
                0x1f83d9abfb41bd6b,
                0x5be0cd19137e2179,
            ],
            buf: [0u8; 128],
            buf_len: 0,
            total_len: 0,
        }
    }

    /// Absorbs additional input bytes.
    pub fn update(&mut self, data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u128);
        let mut offset = 0usize;

        if self.buf_len > 0 {
            let take = (128 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            offset += take;
            if self.buf_len == 128 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }

        while offset + 128 <= data.len() {
            let mut block = [0u8; 128];
            block.copy_from_slice(&data[offset..offset + 128]);
            self.compress(&block);
            offset += 128;
        }

        if offset < data.len() {
            let rem = data.len() - offset;
            self.buf[..rem].copy_from_slice(&data[offset..]);
            self.buf_len = rem;
        }
    }

    /// Finalizes the digest and returns the 64-byte SHA-512 output.
    #[must_use]
    pub fn finalize(mut self) -> [u8; 64] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        if self.buf_len > 112 {
            for b in &mut self.buf[self.buf_len..128] {
                *b = 0;
            }
            let block = self.buf;
            self.compress(&block);
            self.buf_len = 0;
        }

        for b in &mut self.buf[self.buf_len..112] {
            *b = 0;
        }
        self.buf[112..128].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        self.compress(&block);

        let mut out = [0u8; 64];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; 128]) {
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

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
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

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

/// Computes the FIPS 180-4 SHA-512 digest of `data`.
#[must_use]
pub fn sha512(data: &[u8]) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(data);
    h.finalize()
}

// ============================================================================
// RFC 2104 HMAC-SHA256 & RFC 5869 HKDF-SHA256
// ============================================================================

/// Computes RFC 2104 HMAC-SHA256.
#[must_use]
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut k_block = [0u8; 64];
    if key.len() > 64 {
        let hashed = sha256(key);
        k_block[..32].copy_from_slice(&hashed);
    } else {
        k_block[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k_block[i];
        opad[i] ^= k_block[i];
    }

    let mut inner = Sha256::new();
    inner.update(&ipad);
    inner.update(data);
    let inner_digest = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(&inner_digest);
    let out = outer.finalize();

    zeroize_bytes(&mut k_block);
    zeroize_bytes(&mut ipad);
    zeroize_bytes(&mut opad);
    out
}

/// RFC 5869 HKDF-Extract using SHA-256.
#[must_use]
pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    if salt.is_empty() {
        hmac_sha256(&[0u8; 32], ikm)
    } else {
        hmac_sha256(salt, ikm)
    }
}

/// RFC 5869 HKDF-Expand using SHA-256 into `okm_len` bytes (max `255 * 32`).
pub fn hkdf_expand(prk: &[u8; 32], info: &[u8], okm_len: usize) -> Result<Vec<u8>> {
    if okm_len > 255 * 32 {
        return Err(AttestError::new(
            ErrorCode::InvalidEncoding,
            "HKDF-SHA256 output length exceeds 255 * 32 bytes",
        ));
    }
    let n = okm_len.div_ceil(32);
    let mut okm = Vec::with_capacity(n * 32);
    let mut prev: Vec<u8> = Vec::new();
    for counter in 1..=n {
        let mut msg = Vec::with_capacity(prev.len() + info.len() + 1);
        msg.extend_from_slice(&prev);
        msg.extend_from_slice(info);
        msg.push(counter as u8);
        let t = hmac_sha256(prk, &msg);
        okm.extend_from_slice(&t);
        prev.clear();
        prev.extend_from_slice(&t);
    }
    okm.truncate(okm_len);
    Ok(okm)
}

/// Convenience one-shot RFC 5869 HKDF-SHA256 producing a fixed 32-byte key.
#[must_use]
pub fn hkdf_sha256_32(salt: &[u8], ikm: &[u8], info: &[u8]) -> [u8; 32] {
    let mut prk = hkdf_extract(salt, ikm);
    let mut msg = Vec::with_capacity(info.len() + 1);
    msg.extend_from_slice(info);
    msg.push(1u8);
    let out = hmac_sha256(&prk, &msg);
    zeroize_bytes(&mut prk);
    out
}

// ============================================================================
// Field arithmetic over GF(2^255 - 19) using 5 x 51-bit limbs
// ============================================================================

const MASK51: u64 = (1u64 << 51) - 1;

/// Field element modulo `p = 2^255 - 19` stored in 5 radix-2^51 limbs.
#[derive(Clone, Copy, Debug)]
struct Fe([u64; 5]);

impl Fe {
    const ZERO: Self = Self([0, 0, 0, 0, 0]);
    const ONE: Self = Self([1, 0, 0, 0, 0]);

    #[inline]
    fn reduce(mut h: [u64; 5]) -> Self {
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
        let c4 = h[4] >> 51;
        h[4] &= MASK51;
        h[0] += c4 * 19;
        let c0b = h[0] >> 51;
        h[0] &= MASK51;
        h[1] += c0b;
        Self(h)
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
        const TWO_P0: u64 = 2 * ((1u64 << 51) - 19);
        const TWO_P: u64 = 2 * ((1u64 << 51) - 1);
        Self::reduce([
            (self.0[0] + TWO_P0) - rhs.0[0],
            (self.0[1] + TWO_P) - rhs.0[1],
            (self.0[2] + TWO_P) - rhs.0[2],
            (self.0[3] + TWO_P) - rhs.0[3],
            (self.0[4] + TWO_P) - rhs.0[4],
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
        let a0 = a[0] as u128;
        let a1 = a[1] as u128;
        let a2 = a[2] as u128;
        let a3 = a[3] as u128;
        let a4 = a[4] as u128;

        let b0 = b[0] as u128;
        let b1 = b[1] as u128;
        let b2 = b[2] as u128;
        let b3 = b[3] as u128;
        let b4 = b[4] as u128;

        let b1_19 = b1 * 19;
        let b2_19 = b2 * 19;
        let b3_19 = b3 * 19;
        let b4_19 = b4 * 19;

        let mut t0 = a0 * b0 + a1 * b4_19 + a2 * b3_19 + a3 * b2_19 + a4 * b1_19;
        let mut t1 = a0 * b1 + a1 * b0 + a2 * b4_19 + a3 * b3_19 + a4 * b2_19;
        let mut t2 = a0 * b2 + a1 * b1 + a2 * b0 + a3 * b4_19 + a4 * b3_19;
        let mut t3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + a4 * b4_19;
        let mut t4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;

        let mask = MASK51 as u128;
        t1 += t0 >> 51;
        t0 &= mask;
        t2 += t1 >> 51;
        t1 &= mask;
        t3 += t2 >> 51;
        t2 &= mask;
        t4 += t3 >> 51;
        t3 &= mask;
        t0 += (t4 >> 51) * 19;
        t4 &= mask;
        t1 += t0 >> 51;
        t0 &= mask;

        Self([t0 as u64, t1 as u64, t2 as u64, t3 as u64, t4 as u64])
    }

    #[inline]
    fn mul_small(self, rhs: u64) -> Self {
        let r = rhs as u128;
        let mut t0 = (self.0[0] as u128) * r;
        let mut t1 = (self.0[1] as u128) * r;
        let mut t2 = (self.0[2] as u128) * r;
        let mut t3 = (self.0[3] as u128) * r;
        let mut t4 = (self.0[4] as u128) * r;

        let mask = MASK51 as u128;
        t1 += t0 >> 51;
        t0 &= mask;
        t2 += t1 >> 51;
        t1 &= mask;
        t3 += t2 >> 51;
        t2 &= mask;
        t4 += t3 >> 51;
        t3 &= mask;
        t0 += (t4 >> 51) * 19;
        t4 &= mask;
        t1 += t0 >> 51;
        t0 &= mask;

        Self([t0 as u64, t1 as u64, t2 as u64, t3 as u64, t4 as u64])
    }

    #[inline]
    fn square(self) -> Self {
        self.mul(self)
    }

    fn pow_2k(mut self, k: usize) -> Self {
        for _ in 0..k {
            self = self.square();
        }
        self
    }

    /// Computes `self^(2^255 - 21) mod p` (multiplicative inverse for non-zero elements).
    fn invert(self) -> Self {
        let t0 = self.square();
        let t1 = t0.square().square();
        let t2 = self.mul(t1);
        let t3 = t0.mul(t2);
        let t4 = t3.square();
        let t5 = t2.mul(t4);
        let t6 = t5.pow_2k(5);
        let t7 = t6.mul(t5);
        let t8 = t7.pow_2k(10);
        let t9 = t8.mul(t7);
        let t10 = t9.pow_2k(20);
        let t11 = t10.mul(t9);
        let t12 = t11.pow_2k(10);
        let t13 = t12.mul(t7);
        let t14 = t13.pow_2k(50);
        let t15 = t14.mul(t13);
        let t16 = t15.pow_2k(100);
        let t17 = t16.mul(t15);
        let t18 = t17.pow_2k(50);
        let t19 = t18.mul(t13);
        let t20 = t19.pow_2k(5);
        t20.mul(t3)
    }

    /// Computes `self^((p - 5) / 8) = self^(2^252 - 3) mod p`.
    fn pow22523(self) -> Self {
        let t0 = self.square();
        let t1 = t0.square().square();
        let t2 = self.mul(t1);
        let t3 = t0.mul(t2);
        let t4 = t3.square();
        let t5 = t2.mul(t4);
        let t6 = t5.pow_2k(5);
        let t7 = t6.mul(t5);
        let t8 = t7.pow_2k(10);
        let t9 = t8.mul(t7);
        let t10 = t9.pow_2k(20);
        let t11 = t10.mul(t9);
        let t12 = t11.pow_2k(10);
        let t13 = t12.mul(t7);
        let t14 = t13.pow_2k(50);
        let t15 = t14.mul(t13);
        let t16 = t15.pow_2k(100);
        let t17 = t16.mul(t15);
        let t18 = t17.pow_2k(50);
        let t19 = t18.mul(t13);
        let t20 = t19.pow_2k(2);
        t20.mul(self)
    }

    /// Loads 32 little-endian bytes, masking off bit 255 as required by RFC 7748.
    fn from_bytes_masked(bytes: &[u8; 32]) -> Self {
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
        let w0 = load64(0);
        let w1 = load64(8);
        let w2 = load64(16);
        let w3 = load64(24);

        let h0 = w0 & MASK51;
        let h1 = ((w0 >> 51) | (w1 << 13)) & MASK51;
        let h2 = ((w1 >> 38) | (w2 << 26)) & MASK51;
        let h3 = ((w2 >> 25) | (w3 << 39)) & MASK51;
        let h4 = (w3 >> 12) & MASK51;
        Self::reduce([h0, h1, h2, h3, h4])
    }

    /// Canonical 32-byte little-endian encoding modulo `p = 2^255 - 19`.
    fn to_bytes(self) -> [u8; 32] {
        let mut h = Self::reduce(self.0).0;
        // Second carry pass ensures 0 <= h < 2*p and all limbs < 2^51.
        h = Self::reduce(h).0;

        // Compute q = 1 iff h >= p, else 0.
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

        let w0 = h[0] | (h[1] << 51);
        let w1 = (h[1] >> 13) | (h[2] << 38);
        let w2 = (h[2] >> 26) | (h[3] << 25);
        let w3 = (h[3] >> 39) | (h[4] << 12);

        let mut out = [0u8; 32];
        out[0..8].copy_from_slice(&w0.to_le_bytes());
        out[8..16].copy_from_slice(&w1.to_le_bytes());
        out[16..24].copy_from_slice(&w2.to_le_bytes());
        out[24..32].copy_from_slice(&w3.to_le_bytes());
        out
    }

    #[inline]
    fn cswap(swap: u64, a: &mut Self, b: &mut Self) {
        let mask = 0u64.wrapping_sub(swap & 1);
        for i in 0..5 {
            let t = mask & (a.0[i] ^ b.0[i]);
            a.0[i] ^= t;
            b.0[i] ^= t;
        }
    }

    fn is_zero(self) -> bool {
        ct_eq(&self.to_bytes(), &[0u8; 32])
    }

    fn eq_ct(self, other: Self) -> bool {
        ct_eq(&self.to_bytes(), &other.to_bytes())
    }
}

/// Checks strictly that a 255-bit little-endian integer (with bit 255 cleared) is `< 2^255 - 19`.
fn is_canonical_field_bytes(bytes: &[u8; 32]) -> bool {
    // p = 2^255 - 19:
    // bytes[0] = 0xed, bytes[1..31] = 0xff, bytes[31] = 0x7f
    if bytes[31] & 0x7f != 0x7f {
        return true;
    }
    for &b in &bytes[1..31] {
        if b != 0xff {
            return true;
        }
    }
    bytes[0] < 0xed
}

// ============================================================================
// RFC 7748 X25519
// ============================================================================

/// Computes the raw RFC 7748 X25519 scalar multiplication `X25519(k, u)`.
#[must_use]
pub fn x25519(scalar: &[u8; 32], u_coord: &[u8; 32]) -> [u8; 32] {
    let mut k = *scalar;
    k[0] &= 248;
    k[31] &= 127;
    k[31] |= 64;

    let x_1 = Fe::from_bytes_masked(u_coord);
    let mut x_2 = Fe::ONE;
    let mut z_2 = Fe::ZERO;
    let mut x_3 = x_1;
    let mut z_3 = Fe::ONE;
    let mut swap = 0u64;

    for t in (0..255).rev() {
        let k_t = ((k[t >> 3] >> (t & 7)) & 1) as u64;
        swap ^= k_t;
        Fe::cswap(swap, &mut x_2, &mut x_3);
        Fe::cswap(swap, &mut z_2, &mut z_3);
        swap = k_t;

        let a = x_2.add(z_2);
        let aa = a.square();
        let b = x_2.sub(z_2);
        let bb = b.square();
        let e = aa.sub(bb);
        let c = x_3.add(z_3);
        let d = x_3.sub(z_3);
        let da = d.mul(a);
        let cb = c.mul(b);
        x_3 = da.add(cb).square();
        z_3 = x_1.mul(da.sub(cb).square());
        x_2 = aa.mul(bb);
        z_2 = e.mul(aa.add(e.mul_small(121665)));
    }

    Fe::cswap(swap, &mut x_2, &mut x_3);
    Fe::cswap(swap, &mut z_2, &mut z_3);

    zeroize_bytes(&mut k);
    x_2.mul(z_2.invert()).to_bytes()
}

/// Derives the 32-byte X25519 public key from a 32-byte secret key.
#[must_use]
pub fn x25519_public_key(secret: &[u8; 32]) -> [u8; 32] {
    let mut basepoint = [0u8; 32];
    basepoint[0] = 9;
    x25519(secret, &basepoint)
}

/// Computes an X25519 Diffie-Hellman shared secret and rejects all-zero
/// outputs produced by low-order peer public keys (RFC 7748 §6.1).
pub fn x25519_diffie_hellman(secret: &[u8; 32], peer_public: &[u8; 32]) -> Result<[u8; 32]> {
    let shared = x25519(secret, peer_public);
    if ct_eq(&shared, &[0u8; 32]) {
        return Err(AttestError::new(
            ErrorCode::InvalidEphemeralKey,
            "X25519 shared secret is all-zero (low-order or invalid peer public key)",
        ));
    }
    Ok(shared)
}

// ============================================================================
// RFC 8032 Ed25519 (Edwards25519)
// ============================================================================

/// Edwards25519 curve constant `d = -121665 / 121666 mod p`.
fn ed_d() -> Fe {
    Fe::from_bytes_masked(&[
        0xa3, 0x78, 0x59, 0x13, 0xca, 0x4d, 0xeb, 0x75, 0xab, 0xd8, 0x41, 0x41, 0x4d, 0x0a, 0x70,
        0x00, 0x98, 0xe8, 0x79, 0x77, 0x79, 0x40, 0xc7, 0x8c, 0x73, 0xfe, 0x6f, 0x2b, 0xee, 0x6c,
        0x03, 0x52,
    ])
}

/// `2 * d mod p`.
fn ed_2d() -> Fe {
    let d = ed_d();
    d.add(d)
}

/// `sqrt(-1) = 2^((p - 1) / 4) mod p`.
fn ed_sqrt_m1() -> Fe {
    Fe::from_bytes_masked(&[
        0xb0, 0xa0, 0x0e, 0x4a, 0x27, 0x1b, 0xee, 0xc4, 0x78, 0xe4, 0x2f, 0xad, 0x06, 0x18, 0x43,
        0x2f, 0xa7, 0xd7, 0xfb, 0x3d, 0x99, 0x00, 0x4d, 0x2b, 0x0b, 0xdf, 0xc1, 0x4f, 0x80, 0x24,
        0x83, 0x2b,
    ])
}

/// Extended twisted Edwards point `(X, Y, Z, T)` with `x = X/Z, y = Y/Z, xy = T/Z`.
#[derive(Clone, Copy)]
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

    fn basepoint() -> Self {
        let bx = Fe::from_bytes_masked(&[
            0x1a, 0xd5, 0x25, 0x8f, 0x60, 0x2d, 0x56, 0xc9, 0xb2, 0xa7, 0x25, 0x95, 0x60, 0xc7,
            0x2c, 0x69, 0x5c, 0xdc, 0xd6, 0xfd, 0x31, 0xe2, 0xa4, 0xc0, 0xfe, 0x53, 0x6e, 0xcd,
            0xd3, 0x36, 0x69, 0x21,
        ]);
        let by = Fe::from_bytes_masked(&[
            0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
            0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
            0x66, 0x66, 0x66, 0x66,
        ]);
        Self {
            x: bx,
            y: by,
            z: Fe::ONE,
            t: bx.mul(by),
        }
    }

    /// Complete unified Edwards addition (valid for all points including doubling and identity).
    fn add(self, other: Self) -> Self {
        let a = self.y.sub(self.x).mul(other.y.sub(other.x));
        let b = self.y.add(self.x).mul(other.y.add(other.x));
        let c = self.t.mul(ed_2d()).mul(other.t);
        let d = self.z.add(self.z).mul(other.z);
        let e = b.sub(a);
        let f = d.sub(c);
        let g = d.add(c);
        let h = b.add(a);
        Self {
            x: e.mul(f),
            y: g.mul(h),
            t: e.mul(h),
            z: f.mul(g),
        }
    }

    fn double(self) -> Self {
        let a = self.x.square();
        let b = self.y.square();
        let c = self.z.square();
        let c = c.add(c);
        let h = a.add(b);
        let e = h.sub(self.x.add(self.y).square());
        let g = a.sub(b);
        let f = c.add(g);
        Self {
            x: e.mul(f),
            y: g.mul(h),
            t: e.mul(h),
            z: f.mul(g),
        }
    }

    /// Constant-time scalar multiplication `[scalar] * self` (256-bit little-endian scalar).
    fn scalar_mul(self, scalar: &[u8; 32]) -> Self {
        let mut r0 = Self::IDENTITY;
        let mut r1 = self;
        for byte_idx in (0..32).rev() {
            let byte = scalar[byte_idx];
            for bit_idx in (0..8).rev() {
                let bit = ((byte >> bit_idx) & 1) as u64;
                Self::cswap(bit, &mut r0, &mut r1);
                r1 = r0.add(r1);
                r0 = r0.double();
                Self::cswap(bit, &mut r0, &mut r1);
            }
        }
        r0
    }

    fn cswap(swap: u64, a: &mut Self, b: &mut Self) {
        Fe::cswap(swap, &mut a.x, &mut b.x);
        Fe::cswap(swap, &mut a.y, &mut b.y);
        Fe::cswap(swap, &mut a.z, &mut b.z);
        Fe::cswap(swap, &mut a.t, &mut b.t);
    }

    fn compress(self) -> [u8; 32] {
        let z_inv = self.z.invert();
        let x = self.x.mul(z_inv).to_bytes();
        let mut y = self.y.mul(z_inv).to_bytes();
        y[31] |= (x[0] & 1) << 7;
        y
    }

    /// Strict RFC 8032 canonical point decompression (rejects `y >= p` and non-canonical `x == 0` with sign 1).
    fn decompress(bytes: &[u8; 32]) -> Option<Self> {
        if !is_canonical_field_bytes(bytes) {
            return None;
        }
        let sign = (bytes[31] >> 7) & 1;
        let y = Fe::from_bytes_masked(bytes);
        let y2 = y.square();
        let u = y2.sub(Fe::ONE);
        let v = ed_d().mul(y2).add(Fe::ONE);

        // Compute x = u * v^3 * (u * v^7)^((p - 5) / 8)
        let v3 = v.square().mul(v);
        let v7 = v3.square().mul(v);
        let uv7 = u.mul(v7);
        let mut x = u.mul(v3).mul(uv7.pow22523());

        let vx2 = v.mul(x.square());
        if !vx2.eq_ct(u) {
            if vx2.eq_ct(u.neg()) {
                x = x.mul(ed_sqrt_m1());
            } else {
                return None;
            }
        }

        if x.is_zero() {
            if sign == 1 {
                return None;
            }
        } else {
            let x_bytes = x.to_bytes();
            if (x_bytes[0] & 1) != sign {
                x = x.neg();
            }
        }

        Some(Self {
            x,
            y,
            z: Fe::ONE,
            t: x.mul(y),
        })
    }

    fn eq_ct(self, other: Self) -> bool {
        // Check X1*Z2 == X2*Z1 and Y1*Z2 == Y2*Z1.
        let lhs_x = self.x.mul(other.z);
        let rhs_x = other.x.mul(self.z);
        let lhs_y = self.y.mul(other.z);
        let rhs_y = other.y.mul(self.z);
        lhs_x.eq_ct(rhs_x) && lhs_y.eq_ct(rhs_y)
    }
}

/// Group order `L = 2^252 + 27742317777372353535851937790883648493` in little-endian bytes.
const ED25519_L: [u8; 32] = [
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10,
];

/// Returns `true` iff the 256-bit little-endian integer `s` satisfies `s < L`.
fn scalar_is_canonical(s: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if s[i] < ED25519_L[i] {
            return true;
        }
        if s[i] > ED25519_L[i] {
            return false;
        }
    }
    false
}

/// Reduces a 512-bit little-endian unsigned integer modulo `L`.
///
/// Uses straightforward bit-by-bit Barrett/shift-subtract over 512 bits with
/// 256-bit remainder (`[u64; 5]` with top limb for overflow), keeping the
/// implementation compact, auditably exact, and zero-dependency.
fn scalar_reduce_512(input: &[u8; 64]) -> [u8; 32] {
    const L_U64: [u64; 4] = [
        0x5812_631a_5cf5_d3ed,
        0x14de_f9de_a2f7_9cd6,
        0x0000_0000_0000_0000,
        0x1000_0000_0000_0000,
    ];

    let mut rem = [0u64; 4];
    for byte_idx in (0..64).rev() {
        let byte = input[byte_idx];
        for bit_idx in (0..8).rev() {
            let bit = ((byte >> bit_idx) & 1) as u64;
            // rem = (rem << 1) | bit
            let mut carry = bit;
            for limb in &mut rem {
                let next_carry = *limb >> 63;
                *limb = (*limb << 1) | carry;
                carry = next_carry;
            }
            // If rem >= L (guaranteed rem < 2*L), subtract L.
            let ge = carry != 0 || ge_u256(&rem, &L_U64);
            if ge {
                sub_u256_in_place(&mut rem, &L_U64);
            }
        }
    }

    let mut out = [0u8; 32];
    for (i, &limb) in rem.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    out
}

fn ge_u256(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] > b[i] {
            return true;
        }
        if a[i] < b[i] {
            return false;
        }
    }
    true
}

fn sub_u256_in_place(a: &mut [u64; 4], b: &[u64; 4]) {
    let mut borrow = 0u128;
    for i in 0..4 {
        let lhs = a[i] as u128;
        let rhs = (b[i] as u128) + borrow;
        if lhs >= rhs {
            a[i] = (lhs - rhs) as u64;
            borrow = 0;
        } else {
            a[i] = ((1u128 << 64) + lhs - rhs) as u64;
            borrow = 1;
        }
    }
}

/// Computes `(a * b + c) mod L` for 256-bit little-endian scalars.
fn scalar_mul_add(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> [u8; 32] {
    let mut al = [0u64; 4];
    let mut bl = [0u64; 4];
    let mut cl = [0u64; 4];
    for i in 0..4 {
        al[i] = u64::from_le_bytes(a[i * 8..(i + 1) * 8].try_into().unwrap_or([0; 8]));
        bl[i] = u64::from_le_bytes(b[i * 8..(i + 1) * 8].try_into().unwrap_or([0; 8]));
        cl[i] = u64::from_le_bytes(c[i * 8..(i + 1) * 8].try_into().unwrap_or([0; 8]));
    }

    let mut prod = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let acc = (prod[i + j] as u128) + (al[i] as u128) * (bl[j] as u128) + carry;
            prod[i + j] = acc as u64;
            carry = acc >> 64;
        }
        prod[i + 4] = carry as u64;
    }

    let mut carry = 0u128;
    for i in 0..4 {
        let acc = (prod[i] as u128) + (cl[i] as u128) + carry;
        prod[i] = acc as u64;
        carry = acc >> 64;
    }
    for limb in &mut prod[4..8] {
        let acc = (*limb as u128) + carry;
        *limb = acc as u64;
        carry = acc >> 64;
    }

    let mut wide = [0u8; 64];
    for (i, &limb) in prod.iter().enumerate() {
        wide[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    scalar_reduce_512(&wide)
}

/// Derives the 32-byte RFC 8032 Ed25519 public key from a 32-byte private seed.
#[must_use]
pub fn ed25519_public_key(seed: &[u8; 32]) -> [u8; 32] {
    let mut h = sha512(seed);
    let mut s = [0u8; 32];
    s.copy_from_slice(&h[0..32]);
    s[0] &= 248;
    s[31] &= 63;
    s[31] |= 64;
    let pubkey = ExtPoint::basepoint().scalar_mul(&s).compress();
    zeroize_bytes(&mut h);
    zeroize_bytes(&mut s);
    pubkey
}

/// Produces an RFC 8032 deterministic Ed25519 signature over `message`.
#[must_use]
pub fn ed25519_sign(seed: &[u8; 32], message: &[u8]) -> [u8; 64] {
    let mut h = sha512(seed);
    let mut s = [0u8; 32];
    s.copy_from_slice(&h[0..32]);
    s[0] &= 248;
    s[31] &= 63;
    s[31] |= 64;

    let pubkey = ExtPoint::basepoint().scalar_mul(&s).compress();

    // r = SHA-512(prefix || message) mod L
    let mut hr = Sha512::new();
    hr.update(&h[32..64]);
    hr.update(message);
    let mut r_digest = hr.finalize();
    let mut r = scalar_reduce_512(&r_digest);

    let r_point = ExtPoint::basepoint().scalar_mul(&r).compress();

    // k = SHA-512(R || A || message) mod L
    let mut hk = Sha512::new();
    hk.update(&r_point);
    hk.update(&pubkey);
    hk.update(message);
    let k_digest = hk.finalize();
    let k = scalar_reduce_512(&k_digest);

    let sig_s = scalar_mul_add(&k, &s, &r);

    let mut sig = [0u8; 64];
    sig[0..32].copy_from_slice(&r_point);
    sig[32..64].copy_from_slice(&sig_s);

    zeroize_bytes(&mut h);
    zeroize_bytes(&mut s);
    zeroize_bytes(&mut r_digest);
    zeroize_bytes(&mut r);
    sig
}

/// Strictly verifies an RFC 8032 Ed25519 signature over `message`.
///
/// Rejects non-canonical `S >= L` (malleability) and non-canonical point
/// encodings (`y >= p` or `x == 0` with sign bit 1).
#[must_use]
pub fn ed25519_verify(public_key: &[u8; 32], message: &[u8], signature: &[u8; 64]) -> bool {
    let mut r_bytes = [0u8; 32];
    let mut s_bytes = [0u8; 32];
    r_bytes.copy_from_slice(&signature[0..32]);
    s_bytes.copy_from_slice(&signature[32..64]);

    if !scalar_is_canonical(&s_bytes) {
        return false;
    }
    let Some(a_point) = ExtPoint::decompress(public_key) else {
        return false;
    };
    let Some(r_point) = ExtPoint::decompress(&r_bytes) else {
        return false;
    };

    let mut hk = Sha512::new();
    hk.update(&r_bytes);
    hk.update(public_key);
    hk.update(message);
    let k_digest = hk.finalize();
    let k = scalar_reduce_512(&k_digest);

    let lhs = ExtPoint::basepoint().scalar_mul(&s_bytes);
    let rhs = r_point.add(a_point.scalar_mul(&k));
    lhs.eq_ct(rhs)
}

// ============================================================================
// RFC 8439 ChaCha20-Poly1305 AEAD
// ============================================================================

#[inline]
fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(16);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(12);

    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(8);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(7);
}

/// Computes a single 64-byte RFC 8439 ChaCha20 keystream block.
#[must_use]
pub fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    let mut state = [0u32; 16];
    state[0] = 0x6170_7865;
    state[1] = 0x3320_646e;
    state[2] = 0x7962_2d32;
    state[3] = 0x6b20_6574;

    for i in 0..8 {
        state[4 + i] =
            u32::from_le_bytes([key[i * 4], key[i * 4 + 1], key[i * 4 + 2], key[i * 4 + 3]]);
    }
    state[12] = counter;
    state[13] = u32::from_le_bytes([nonce[0], nonce[1], nonce[2], nonce[3]]);
    state[14] = u32::from_le_bytes([nonce[4], nonce[5], nonce[6], nonce[7]]);
    state[15] = u32::from_le_bytes([nonce[8], nonce[9], nonce[10], nonce[11]]);

    let mut working = state;
    for _ in 0..10 {
        quarter_round(&mut working, 0, 4, 8, 12);
        quarter_round(&mut working, 1, 5, 9, 13);
        quarter_round(&mut working, 2, 6, 10, 14);
        quarter_round(&mut working, 3, 7, 11, 15);
        quarter_round(&mut working, 0, 5, 10, 15);
        quarter_round(&mut working, 1, 6, 11, 12);
        quarter_round(&mut working, 2, 7, 8, 13);
        quarter_round(&mut working, 3, 4, 9, 14);
    }

    let mut out = [0u8; 64];
    for i in 0..16 {
        let word = working[i].wrapping_add(state[i]);
        out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

/// Applies the RFC 8439 ChaCha20 stream cipher (XORs `input` with keystream).
#[must_use]
pub fn chacha20_xor(
    key: &[u8; 32],
    initial_counter: u32,
    nonce: &[u8; 12],
    input: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut counter = initial_counter;
    for chunk in input.chunks(64) {
        let ks = chacha20_block(key, counter, nonce);
        for (&b, &k) in chunk.iter().zip(ks.iter()) {
            out.push(b ^ k);
        }
        counter = counter.wrapping_add(1);
    }
    out
}

/// Computes an RFC 8439 §2.5 Poly1305 16-byte MAC using exact 5×26-bit limb arithmetic.
#[must_use]
pub fn poly1305_mac(one_time_key: &[u8; 32], msg: &[u8]) -> [u8; 16] {
    // Clamp r (RFC 8439 §2.5)
    let mut r_bytes = [0u8; 16];
    r_bytes.copy_from_slice(&one_time_key[0..16]);
    r_bytes[3] &= 15;
    r_bytes[7] &= 15;
    r_bytes[11] &= 15;
    r_bytes[15] &= 15;
    r_bytes[4] &= 252;
    r_bytes[8] &= 252;
    r_bytes[12] &= 252;

    let t0 = u32::from_le_bytes([r_bytes[0], r_bytes[1], r_bytes[2], r_bytes[3]]) as u64;
    let t1 = u32::from_le_bytes([r_bytes[4], r_bytes[5], r_bytes[6], r_bytes[7]]) as u64;
    let t2 = u32::from_le_bytes([r_bytes[8], r_bytes[9], r_bytes[10], r_bytes[11]]) as u64;
    let t3 = u32::from_le_bytes([r_bytes[12], r_bytes[13], r_bytes[14], r_bytes[15]]) as u64;

    let r0 = t0 & 0x03ff_ffff;
    let r1 = ((t0 >> 26) | (t1 << 6)) & 0x03ff_ffff;
    let r2 = ((t1 >> 20) | (t2 << 12)) & 0x03ff_ffff;
    let r3 = ((t2 >> 14) | (t3 << 18)) & 0x03ff_ffff;
    let r4 = t3 >> 8;

    let s1 = r1 * 5;
    let s2 = r2 * 5;
    let s3 = r3 * 5;
    let s4 = r4 * 5;

    let mut h0 = 0u64;
    let mut h1 = 0u64;
    let mut h2 = 0u64;
    let mut h3 = 0u64;
    let mut h4 = 0u64;

    for chunk in msg.chunks(16) {
        let mut block = [0u8; 17];
        block[..chunk.len()].copy_from_slice(chunk);
        block[chunk.len()] = 1;

        let b0 = u32::from_le_bytes([block[0], block[1], block[2], block[3]]) as u64;
        let b1 = u32::from_le_bytes([block[4], block[5], block[6], block[7]]) as u64;
        let b2 = u32::from_le_bytes([block[8], block[9], block[10], block[11]]) as u64;
        let b3 = u32::from_le_bytes([block[12], block[13], block[14], block[15]]) as u64;
        let b4 = block[16] as u64;

        h0 += b0 & 0x03ff_ffff;
        h1 += ((b0 >> 26) | (b1 << 6)) & 0x03ff_ffff;
        h2 += ((b1 >> 20) | (b2 << 12)) & 0x03ff_ffff;
        h3 += ((b2 >> 14) | (b3 << 18)) & 0x03ff_ffff;
        h4 += (b3 >> 8) | (b4 << 24);

        let d0 = h0 * r0 + h1 * s4 + h2 * s3 + h3 * s2 + h4 * s1;
        let mut d1 = h0 * r1 + h1 * r0 + h2 * s4 + h3 * s3 + h4 * s2;
        let mut d2 = h0 * r2 + h1 * r1 + h2 * r0 + h3 * s4 + h4 * s3;
        let mut d3 = h0 * r3 + h1 * r2 + h2 * r1 + h3 * r0 + h4 * s4;
        let mut d4 = h0 * r4 + h1 * r3 + h2 * r2 + h3 * r1 + h4 * r0;

        let c = d0 >> 26;
        h0 = d0 & 0x03ff_ffff;
        d1 += c;
        let c = d1 >> 26;
        h1 = d1 & 0x03ff_ffff;
        d2 += c;
        let c = d2 >> 26;
        h2 = d2 & 0x03ff_ffff;
        d3 += c;
        let c = d3 >> 26;
        h3 = d3 & 0x03ff_ffff;
        d4 += c;
        let c = d4 >> 26;
        h4 = d4 & 0x03ff_ffff;
        h0 += c * 5;
        let c = h0 >> 26;
        h0 &= 0x03ff_ffff;
        h1 += c;
    }

    // Full carry propagation and conditional subtraction of 2^130 - 5.
    let c = h1 >> 26;
    h1 &= 0x03ff_ffff;
    h2 += c;
    let c = h2 >> 26;
    h2 &= 0x03ff_ffff;
    h3 += c;
    let c = h3 >> 26;
    h3 &= 0x03ff_ffff;
    h4 += c;
    let c = h4 >> 26;
    h4 &= 0x03ff_ffff;
    h0 += c * 5;
    let c = h0 >> 26;
    h0 &= 0x03ff_ffff;
    h1 += c;

    let mut g0 = h0 + 5;
    let c = g0 >> 26;
    g0 &= 0x03ff_ffff;
    let mut g1 = h1 + c;
    let c = g1 >> 26;
    g1 &= 0x03ff_ffff;
    let mut g2 = h2 + c;
    let c = g2 >> 26;
    g2 &= 0x03ff_ffff;
    let mut g3 = h3 + c;
    let c = g3 >> 26;
    g3 &= 0x03ff_ffff;
    let g4 = h4.wrapping_add(c).wrapping_sub(1 << 26);

    // Select g if h >= p (bit 63 of g4 == 0), else h.
    let mask = (g4 >> 63).wrapping_sub(1);
    h0 = (h0 & !mask) | (g0 & mask);
    h1 = (h1 & !mask) | (g1 & mask);
    h2 = (h2 & !mask) | (g2 & mask);
    h3 = (h3 & !mask) | (g3 & mask);
    h4 = (h4 & !mask) | (g4 & mask);

    // Pack into 128-bit integer and add pad s = one_time_key[16..32].
    let f0 = ((h0) | (h1 << 26)) & 0xffff_ffff;
    let f1 = ((h1 >> 6) | (h2 << 20)) & 0xffff_ffff;
    let f2 = ((h2 >> 12) | (h3 << 14)) & 0xffff_ffff;
    let f3 = ((h3 >> 18) | (h4 << 8)) & 0xffff_ffff;

    let p0 = u32::from_le_bytes([
        one_time_key[16],
        one_time_key[17],
        one_time_key[18],
        one_time_key[19],
    ]) as u64;
    let p1 = u32::from_le_bytes([
        one_time_key[20],
        one_time_key[21],
        one_time_key[22],
        one_time_key[23],
    ]) as u64;
    let p2 = u32::from_le_bytes([
        one_time_key[24],
        one_time_key[25],
        one_time_key[26],
        one_time_key[27],
    ]) as u64;
    let p3 = u32::from_le_bytes([
        one_time_key[28],
        one_time_key[29],
        one_time_key[30],
        one_time_key[31],
    ]) as u64;

    let sum0 = f0 + p0;
    let sum1 = f1 + p1 + (sum0 >> 32);
    let sum2 = f2 + p2 + (sum1 >> 32);
    let sum3 = f3 + p3 + (sum2 >> 32);

    let mut tag = [0u8; 16];
    tag[0..4].copy_from_slice(&(sum0 as u32).to_le_bytes());
    tag[4..8].copy_from_slice(&(sum1 as u32).to_le_bytes());
    tag[8..12].copy_from_slice(&(sum2 as u32).to_le_bytes());
    tag[12..16].copy_from_slice(&(sum3 as u32).to_le_bytes());
    tag
}

fn poly1305_aead_tag(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
    let mut block0 = chacha20_block(key, 0, nonce);
    let mut otk = [0u8; 32];
    otk.copy_from_slice(&block0[0..32]);
    zeroize_bytes(&mut block0);

    let aad_pad = (16 - (aad.len() % 16)) % 16;
    let ct_pad = (16 - (ciphertext.len() % 16)) % 16;
    let mut mac_data = Vec::with_capacity(aad.len() + aad_pad + ciphertext.len() + ct_pad + 16);
    mac_data.extend_from_slice(aad);
    mac_data.resize(mac_data.len() + aad_pad, 0);
    mac_data.extend_from_slice(ciphertext);
    mac_data.resize(mac_data.len() + ct_pad, 0);
    mac_data.extend_from_slice(&(aad.len() as u64).to_le_bytes());
    mac_data.extend_from_slice(&(ciphertext.len() as u64).to_le_bytes());

    let tag = poly1305_mac(&otk, &mac_data);
    zeroize_bytes(&mut otk);
    tag
}

/// Encrypts `plaintext` with RFC 8439 ChaCha20-Poly1305, returning `(ciphertext, tag)`.
#[must_use]
pub fn chacha20_poly1305_encrypt(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    plaintext: &[u8],
) -> (Vec<u8>, [u8; 16]) {
    let ciphertext = chacha20_xor(key, 1, nonce, plaintext);
    let tag = poly1305_aead_tag(key, nonce, aad, &ciphertext);
    (ciphertext, tag)
}

/// Verifies `tag` in constant time and decrypts `ciphertext` with RFC 8439 ChaCha20-Poly1305.
pub fn chacha20_poly1305_decrypt(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    ciphertext: &[u8],
    tag: &[u8; 16],
) -> Result<Vec<u8>> {
    let expected_tag = poly1305_aead_tag(key, nonce, aad, ciphertext);
    if !ct_eq(&expected_tag, tag) {
        return Err(AttestError::new(
            ErrorCode::AeadAuthFailed,
            "ChaCha20-Poly1305 authentication tag mismatch",
        ));
    }
    Ok(chacha20_xor(key, 1, nonce, ciphertext))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_and_sha512_vectors() {
        assert_eq!(
            hex_encode(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex_encode(&sha512(b"abc")),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );
    }

    #[test]
    fn test_rfc5869_hkdf_sha256_case1() {
        let ikm = [0x0bu8; 22];
        let salt = hex_decode("000102030405060708090a0b0c").unwrap();
        let info = hex_decode("f0f1f2f3f4f5f6f7f8f9").unwrap();
        let prk = hkdf_extract(&salt, &ikm);
        assert_eq!(
            hex_encode(&prk),
            "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
        );
        let okm = hkdf_expand(&prk, &info, 42).unwrap();
        assert_eq!(
            hex_encode(&okm),
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
        );
    }

    #[test]
    fn test_rfc8032_section_7_1_ed25519_vectors() {
        // TEST 1 (empty message)
        let sk1: [u8; 32] =
            hex_decode_array("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
                .unwrap();
        let pk1 = ed25519_public_key(&sk1);
        assert_eq!(
            hex_encode(&pk1),
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
        );
        let sig1 = ed25519_sign(&sk1, b"");
        assert_eq!(
            hex_encode(&sig1),
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        );
        assert!(ed25519_verify(&pk1, b"", &sig1));

        // TEST 2 (1-byte message 0x72)
        let sk2: [u8; 32] =
            hex_decode_array("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb")
                .unwrap();
        let pk2 = ed25519_public_key(&sk2);
        assert_eq!(
            hex_encode(&pk2),
            "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"
        );
        let sig2 = ed25519_sign(&sk2, &[0x72]);
        assert_eq!(
            hex_encode(&sig2),
            "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
        );
        assert!(ed25519_verify(&pk2, &[0x72], &sig2));

        // TEST 3 (2-byte message 0xaf82)
        let sk3: [u8; 32] =
            hex_decode_array("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7")
                .unwrap();
        let pk3 = ed25519_public_key(&sk3);
        assert_eq!(
            hex_encode(&pk3),
            "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025"
        );
        let sig3 = ed25519_sign(&sk3, &[0xaf, 0x82]);
        assert_eq!(
            hex_encode(&sig3),
            "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"
        );
        assert!(ed25519_verify(&pk3, &[0xaf, 0x82], &sig3));

        // Reject S + L malleability
        let mut malleable = sig1;
        let mut carry = 0u16;
        for i in 0..32 {
            let sum = (malleable[32 + i] as u16) + (ED25519_L[i] as u16) + carry;
            malleable[32 + i] = sum as u8;
            carry = sum >> 8;
        }
        assert!(!ed25519_verify(&pk1, b"", &malleable));
    }

    #[test]
    fn test_rfc7748_x25519_vectors() {
        // RFC 7748 §5.2 Vector 1
        let k1: [u8; 32] =
            hex_decode_array("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4")
                .unwrap();
        let u1: [u8; 32] =
            hex_decode_array("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c")
                .unwrap();
        assert_eq!(
            hex_encode(&x25519(&k1, &u1)),
            "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552"
        );

        // RFC 7748 §5.2 Vector 2
        let k2: [u8; 32] =
            hex_decode_array("4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d")
                .unwrap();
        let u2: [u8; 32] =
            hex_decode_array("e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493")
                .unwrap();
        assert_eq!(
            hex_encode(&x25519(&k2, &u2)),
            "95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957"
        );

        // RFC 7748 §6.1 Diffie-Hellman Alice/Bob
        let alice_sk: [u8; 32] =
            hex_decode_array("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
                .unwrap();
        let bob_sk: [u8; 32] =
            hex_decode_array("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb")
                .unwrap();
        let alice_pk = x25519_public_key(&alice_sk);
        let bob_pk = x25519_public_key(&bob_sk);
        assert_eq!(
            hex_encode(&alice_pk),
            "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a"
        );
        assert_eq!(
            hex_encode(&bob_pk),
            "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f"
        );
        let ss_a = x25519_diffie_hellman(&alice_sk, &bob_pk).unwrap();
        let ss_b = x25519_diffie_hellman(&bob_sk, &alice_pk).unwrap();
        assert_eq!(ss_a, ss_b);
        assert_eq!(
            hex_encode(&ss_a),
            "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742"
        );

        // Low-order all-zero public key must be rejected
        assert_eq!(
            x25519_diffie_hellman(&alice_sk, &[0u8; 32])
                .unwrap_err()
                .code,
            ErrorCode::InvalidEphemeralKey
        );
    }

    #[test]
    fn test_rfc8439_chacha20_poly1305_vector() {
        // RFC 8439 §2.8.2 AEAD test vector
        let key: [u8; 32] =
            hex_decode_array("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f")
                .unwrap();
        let nonce: [u8; 12] = hex_decode_array("070000004041424344454647").unwrap();
        let aad = hex_decode("50515253c0c1c2c3c4c5c6c7").unwrap();
        let plaintext = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";

        let (ct, tag) = chacha20_poly1305_encrypt(&key, &nonce, &aad, plaintext);
        assert_eq!(
            hex_encode(&ct),
            "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d63dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b3692ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc3ff4def08e4b7a9de576d26586cec64b6116"
        );
        assert_eq!(hex_encode(&tag), "1ae10b594f09e26a7e902ecbd0600691");

        let pt = chacha20_poly1305_decrypt(&key, &nonce, &aad, &ct, &tag).unwrap();
        assert_eq!(pt, plaintext);

        // Tampering ciphertext or AAD fails tag check
        let mut bad_ct = ct.clone();
        bad_ct[0] ^= 1;
        assert_eq!(
            chacha20_poly1305_decrypt(&key, &nonce, &aad, &bad_ct, &tag)
                .unwrap_err()
                .code,
            ErrorCode::AeadAuthFailed
        );
    }
}
