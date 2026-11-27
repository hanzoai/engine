//! Pure-Rust cryptographic primitives (`std`-only, `#![forbid(unsafe_code)]`):
//! - FIPS 180-4 SHA-256 and SHA-512
//! - Constant-time byte slice equality
//! - RFC 8032 Ed25519 digital signatures over Curve25519 (strict canonical decoding,
//!   rejecting non-canonical `y >= p`, non-canonical `S >= L`, and invalid zero-x sign bits)

use std::sync::OnceLock;

/// Constant-time equality over equal-length byte slices.
#[inline]
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

/// Lowercase hexadecimal encoder.
pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Strict hexadecimal decoder (accepts lowercase or uppercase ASCII hex).
pub fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err("hex string length must be even".into());
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex_nibble(bytes[i])?;
        let lo = hex_nibble(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

#[inline]
fn hex_nibble(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("invalid hex byte: 0x{b:02x}")),
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

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

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
        for (i, &word) in self.state.iter().enumerate() {
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

/// One-shot FIPS 180-4 SHA-256 digest.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

/// Lowercase hex-encoded SHA-256 digest.
pub fn sha256_hex(data: &[u8]) -> String {
    hex_encode(&sha256(data))
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

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u128);
        if self.buf_len > 0 {
            let take = (128 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 128 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 128 {
            let mut block = [0u8; 128];
            block.copy_from_slice(&data[..128]);
            self.compress(&block);
            data = &data[128..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

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
        for (i, &word) in self.state.iter().enumerate() {
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

/// One-shot FIPS 180-4 SHA-512 digest.
pub fn sha512(data: &[u8]) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(data);
    h.finalize()
}

// ============================================================================
// RFC 8032 Ed25519 over Curve25519 (pure Rust std, zero dependencies)
// ============================================================================

const MASK51: u64 = (1u64 << 51) - 1;

/// Field element in F_p where p = 2^255 - 19, stored in radix 2^51 (5 limbs).
#[derive(Clone, Copy, Debug)]
struct Fe([u64; 5]);

impl Fe {
    const ZERO: Fe = Fe([0, 0, 0, 0, 0]);
    const ONE: Fe = Fe([1, 0, 0, 0, 0]);

    #[inline]
    fn from_u64(v: u64) -> Self {
        Fe([v & MASK51, v >> 51, 0, 0, 0])
    }

    #[inline]
    fn carry_reduce(mut h: [u64; 5]) -> Self {
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
        Fe(h)
    }

    #[inline]
    fn add(self, rhs: Fe) -> Fe {
        Self::carry_reduce([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
        ])
    }

    #[inline]
    fn sub(self, rhs: Fe) -> Fe {
        // Add 2p in radix 2^51 so every limb stays strictly positive before carry_reduce.
        const TWO_P: [u64; 5] = [
            0x000f_ffff_ffff_ffda,
            0x000f_ffff_ffff_fffe,
            0x000f_ffff_ffff_fffe,
            0x000f_ffff_ffff_fffe,
            0x000f_ffff_ffff_fffe,
        ];
        Self::carry_reduce([
            self.0[0] + TWO_P[0] - rhs.0[0],
            self.0[1] + TWO_P[1] - rhs.0[1],
            self.0[2] + TWO_P[2] - rhs.0[2],
            self.0[3] + TWO_P[3] - rhs.0[3],
            self.0[4] + TWO_P[4] - rhs.0[4],
        ])
    }

    #[inline]
    fn neg(self) -> Fe {
        Fe::ZERO.sub(self)
    }

    #[inline]
    fn mul(self, rhs: Fe) -> Fe {
        let a = self.0;
        let b = rhs.0;

        let b1_19 = (b[1] as u128) * 19;
        let b2_19 = (b[2] as u128) * 19;
        let b3_19 = (b[3] as u128) * 19;
        let b4_19 = (b[4] as u128) * 19;

        let mut t0 = (a[0] as u128) * (b[0] as u128)
            + (a[1] as u128) * b4_19
            + (a[2] as u128) * b3_19
            + (a[3] as u128) * b2_19
            + (a[4] as u128) * b1_19;

        let mut t1 = (a[0] as u128) * (b[1] as u128)
            + (a[1] as u128) * (b[0] as u128)
            + (a[2] as u128) * b4_19
            + (a[3] as u128) * b3_19
            + (a[4] as u128) * b2_19;

        let mut t2 = (a[0] as u128) * (b[2] as u128)
            + (a[1] as u128) * (b[1] as u128)
            + (a[2] as u128) * (b[0] as u128)
            + (a[3] as u128) * b4_19
            + (a[4] as u128) * b3_19;

        let mut t3 = (a[0] as u128) * (b[3] as u128)
            + (a[1] as u128) * (b[2] as u128)
            + (a[2] as u128) * (b[1] as u128)
            + (a[3] as u128) * (b[0] as u128)
            + (a[4] as u128) * b4_19;

        let mut t4 = (a[0] as u128) * (b[4] as u128)
            + (a[1] as u128) * (b[3] as u128)
            + (a[2] as u128) * (b[2] as u128)
            + (a[3] as u128) * (b[1] as u128)
            + (a[4] as u128) * (b[0] as u128);

        let m = MASK51 as u128;
        t1 += t0 >> 51;
        t0 &= m;
        t2 += t1 >> 51;
        t1 &= m;
        t3 += t2 >> 51;
        t2 &= m;
        t4 += t3 >> 51;
        t3 &= m;
        t0 += (t4 >> 51) * 19;
        t4 &= m;
        t1 += t0 >> 51;
        t0 &= m;

        Fe([t0 as u64, t1 as u64, t2 as u64, t3 as u64, t4 as u64])
    }

    #[inline]
    fn square(self) -> Fe {
        self.mul(self)
    }

    fn pow_bytes_le(self, exp: &[u8; 32]) -> Fe {
        let mut res = Fe::ONE;
        let mut base = self;
        for &byte in exp {
            for bit in 0..8 {
                if ((byte >> bit) & 1) == 1 {
                    res = res.mul(base);
                }
                base = base.square();
            }
        }
        res
    }

    /// Multiplicative inverse `self^(p-2) mod p` where `p - 2 = 2^255 - 21`.
    fn invert(self) -> Fe {
        let mut exp = [0xffu8; 32];
        exp[0] = 0xeb; // 256 - 21 = 235 = 0xeb
        exp[31] = 0x7f;
        self.pow_bytes_le(&exp)
    }

    /// Raise to `(p - 5) / 8 = 2^252 - 3` (RFC 8032 §5.1.3 square-root helper).
    fn pow_p58(self) -> Fe {
        let mut exp = [0xffu8; 32];
        exp[0] = 0xfd; // 256 - 3 = 253 = 0xfd
        exp[31] = 0x0f;
        self.pow_bytes_le(&exp)
    }

    /// Canonical 32-byte little-endian representation in `[0, p - 1]`.
    fn to_bytes(self) -> [u8; 32] {
        let mut h = Self::carry_reduce(self.0).0;
        // One more carry propagation so every limb is < 2^51.
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
        let c0 = h[0] >> 51;
        h[0] &= MASK51;
        h[1] += c0;

        // Constant-time conditional subtraction of p: q = floor((h + 19) / 2^255).
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
        // Pack 5x51-bit limbs into 4x64-bit little-endian words.
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

    /// Strict decoding of a canonical 255-bit field element `< p` (top bit cleared by caller).
    fn from_canonical_bytes(bytes: &[u8; 32]) -> Option<Self> {
        let w0 = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        let w1 = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
        let w2 = u64::from_le_bytes(bytes[16..24].try_into().ok()?);
        let w3 = u64::from_le_bytes(bytes[24..32].try_into().ok()?);

        // p = 2^255 - 19 = [0xffff_ffff_ffff_ffed, 0xffff_ffff_ffff_ffff, 0xffff_ffff_ffff_ffff, 0x7fff_ffff_ffff_ffff]
        let is_lt_p = if w3 != 0x7fff_ffff_ffff_ffff {
            w3 < 0x7fff_ffff_ffff_ffff
        } else if w2 < u64::MAX || w1 < u64::MAX {
            true
        } else {
            w0 < 0xffff_ffff_ffff_ffed
        };
        if !is_lt_p {
            return None;
        }

        let h0 = w0 & MASK51;
        let h1 = ((w0 >> 51) | (w1 << 13)) & MASK51;
        let h2 = ((w1 >> 38) | (w2 << 26)) & MASK51;
        let h3 = ((w2 >> 25) | (w3 << 39)) & MASK51;
        let h4 = (w3 >> 12) & MASK51;
        Some(Fe([h0, h1, h2, h3, h4]))
    }

    fn eq_ct(self, other: Fe) -> bool {
        ct_eq(&self.to_bytes(), &other.to_bytes())
    }
}

struct CurveConstants {
    d: Fe,
    two_d: Fe,
    sqrt_m1: Fe,
    base: ExtendedPoint,
}

fn curve() -> &'static CurveConstants {
    static CONSTS: OnceLock<CurveConstants> = OnceLock::new();
    CONSTS.get_or_init(|| {
        // d = -121665 / 121666 mod p
        let d = Fe::ZERO
            .sub(Fe::from_u64(121665))
            .mul(Fe::from_u64(121666).invert());
        let two_d = d.add(d);
        // sqrt(-1) = 2^((p - 1) / 4) mod p, where (p - 1)/4 = 2^253 - 5
        let mut exp_i = [0xffu8; 32];
        exp_i[0] = 0xfb; // 256 - 5 = 251 = 0xfb
        exp_i[31] = 0x1f;
        let sqrt_m1 = Fe::from_u64(2).pow_bytes_le(&exp_i);

        // RFC 8032 base point B: y = 4/5 mod p, x > 0 (even)
        let by = Fe::from_u64(4).mul(Fe::from_u64(5).invert());
        let y2 = by.square();
        let u = y2.sub(Fe::ONE);
        let v = d.mul(y2).add(Fe::ONE);
        let v3 = v.square().mul(v);
        let v7 = v3.square().mul(v);
        let mut bx = u.mul(v3).mul(u.mul(v7).pow_p58());
        let check = v.mul(bx.square());
        if !check.eq_ct(u) {
            bx = bx.mul(sqrt_m1);
        }
        // Positive (even) x coordinate per RFC 8032 §5.1.
        if (bx.to_bytes()[0] & 1) == 1 {
            bx = bx.neg();
        }
        let base = ExtendedPoint {
            x: bx,
            y: by,
            z: Fe::ONE,
            t: bx.mul(by),
        };
        CurveConstants {
            d,
            two_d,
            sqrt_m1,
            base,
        }
    })
}

/// Extended twisted Edwards point `(X : Y : Z : T)` with `x = X/Z`, `y = Y/Z`, `xy = T/Z`.
#[derive(Clone, Copy, Debug)]
struct ExtendedPoint {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

impl ExtendedPoint {
    const IDENTITY: ExtendedPoint = ExtendedPoint {
        x: Fe::ZERO,
        y: Fe::ONE,
        z: Fe::ONE,
        t: Fe::ZERO,
    };

    /// Complete unified addition on Ed25519 (RFC 8032 §5.1.4).
    fn add(self, other: ExtendedPoint) -> ExtendedPoint {
        let two_d = curve().two_d;
        let a = self.y.sub(self.x).mul(other.y.sub(other.x));
        let b = self.y.add(self.x).mul(other.y.add(other.x));
        let c = self.t.mul(two_d).mul(other.t);
        let z2 = self.z.mul(other.z);
        let d = z2.add(z2);
        let e = b.sub(a);
        let f = d.sub(c);
        let g = d.add(c);
        let h = b.add(a);
        ExtendedPoint {
            x: e.mul(f),
            y: g.mul(h),
            z: f.mul(g),
            t: e.mul(h),
        }
    }

    fn double(self) -> ExtendedPoint {
        self.add(self)
    }

    fn scalar_mul(self, scalar: &[u8; 32]) -> ExtendedPoint {
        let mut acc = ExtendedPoint::IDENTITY;
        let mut addend = self;
        for &byte in scalar {
            for bit in 0..8 {
                if ((byte >> bit) & 1) == 1 {
                    acc = acc.add(addend);
                }
                addend = addend.double();
            }
        }
        acc
    }

    fn encode(self) -> [u8; 32] {
        let zinv = self.z.invert();
        let x = self.x.mul(zinv).to_bytes();
        let mut y = self.y.mul(zinv).to_bytes();
        y[31] |= (x[0] & 1) << 7;
        y
    }

    /// Strict RFC 8032 §5.1.3 point decompression.
    fn decode(bytes: &[u8; 32]) -> Option<ExtendedPoint> {
        let c = curve();
        let sign_bit = bytes[31] >> 7;
        let mut y_bytes = *bytes;
        y_bytes[31] &= 0x7f;
        let y = Fe::from_canonical_bytes(&y_bytes)?;

        let y2 = y.square();
        let u = y2.sub(Fe::ONE);
        let v = c.d.mul(y2).add(Fe::ONE);
        let v3 = v.square().mul(v);
        let v7 = v3.square().mul(v);
        let mut x = u.mul(v3).mul(u.mul(v7).pow_p58());

        let v_x2 = v.mul(x.square());
        if v_x2.eq_ct(u) {
            // valid root
        } else if v_x2.eq_ct(u.neg()) {
            x = x.mul(c.sqrt_m1);
        } else {
            return None;
        }

        let x_bytes = x.to_bytes();
        let x_is_zero = x_bytes.iter().all(|&b| b == 0);
        if x_is_zero && sign_bit == 1 {
            return None;
        }
        if (x_bytes[0] & 1) != sign_bit {
            x = x.neg();
        }

        Some(ExtendedPoint {
            x,
            y,
            z: Fe::ONE,
            t: x.mul(y),
        })
    }

    fn eq_ct(self, other: ExtendedPoint) -> bool {
        // X1*Z2 == X2*Z1 and Y1*Z2 == Y2*Z1
        let lhs_x = self.x.mul(other.z);
        let rhs_x = other.x.mul(self.z);
        let lhs_y = self.y.mul(other.z);
        let rhs_y = other.y.mul(self.z);
        lhs_x.eq_ct(rhs_x) && lhs_y.eq_ct(rhs_y)
    }
}

/// Group order L = 2^252 + 27742317777372353535851937790883648493 in little-endian u64 limbs.
const ORDER_L: [u64; 4] = [
    0x5812_631a_5cf5_d3ed,
    0x14de_f9de_a2f7_9cd6,
    0x0000_0000_0000_0000,
    0x1000_0000_0000_0000,
];

fn u256_lt(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

fn scalar_is_canonical(s: &[u8; 32]) -> bool {
    let limbs = [
        u64::from_le_bytes(s[0..8].try_into().unwrap()),
        u64::from_le_bytes(s[8..16].try_into().unwrap()),
        u64::from_le_bytes(s[16..24].try_into().unwrap()),
        u64::from_le_bytes(s[24..32].try_into().unwrap()),
    ];
    u256_lt(&limbs, &ORDER_L)
}

/// Exact unsigned 512-bit reduction modulo L using bit-by-bit shift-and-subtract.
fn reduce_512_mod_l(bytes: &[u8; 64]) -> [u8; 32] {
    // Process from bit 511 down to bit 0 maintaining 256-bit remainder r < L.
    let mut rem = [0u64; 5]; // 5th limb absorbs shift carry before comparing with L
    for bit_idx in (0..512).rev() {
        let byte = bytes[bit_idx / 8];
        let bit = ((byte >> (bit_idx % 8)) & 1) as u64;

        // rem = (rem << 1) | bit
        let mut carry = bit;
        for limb in &mut rem {
            let next_carry = *limb >> 63;
            *limb = (*limb << 1) | carry;
            carry = next_carry;
        }

        // If rem >= L, rem -= L.
        let ge = rem[4] > 0 || !u256_lt(&[rem[0], rem[1], rem[2], rem[3]], &ORDER_L);
        if ge {
            let mut borrow = 0u128;
            for i in 0..4 {
                let lhs = rem[i] as u128;
                let rhs = (ORDER_L[i] as u128) + borrow;
                if lhs >= rhs {
                    rem[i] = (lhs - rhs) as u64;
                    borrow = 0;
                } else {
                    rem[i] = ((1u128 << 64) + lhs - rhs) as u64;
                    borrow = 1;
                }
            }
            rem[4] = rem[4].wrapping_sub(borrow as u64);
        }
    }

    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..(i + 1) * 8].copy_from_slice(&rem[i].to_le_bytes());
    }
    out
}

/// Computes `(a * b + c) mod L` for 256-bit little-endian scalars.
fn mul_add_mod_l(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> [u8; 32] {
    let al = [
        u64::from_le_bytes(a[0..8].try_into().unwrap()),
        u64::from_le_bytes(a[8..16].try_into().unwrap()),
        u64::from_le_bytes(a[16..24].try_into().unwrap()),
        u64::from_le_bytes(a[24..32].try_into().unwrap()),
    ];
    let bl = [
        u64::from_le_bytes(b[0..8].try_into().unwrap()),
        u64::from_le_bytes(b[8..16].try_into().unwrap()),
        u64::from_le_bytes(b[16..24].try_into().unwrap()),
        u64::from_le_bytes(b[24..32].try_into().unwrap()),
    ];
    let cl = [
        u64::from_le_bytes(c[0..8].try_into().unwrap()),
        u64::from_le_bytes(c[8..16].try_into().unwrap()),
        u64::from_le_bytes(c[16..24].try_into().unwrap()),
        u64::from_le_bytes(c[24..32].try_into().unwrap()),
    ];

    let mut acc = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let prod = (al[i] as u128) * (bl[j] as u128) + (acc[i + j] as u128) + carry;
            acc[i + j] = prod as u64;
            carry = prod >> 64;
        }
        acc[i + 4] = carry as u64;
    }

    // Add c into acc
    let mut carry = 0u128;
    for i in 0..4 {
        let sum = (acc[i] as u128) + (cl[i] as u128) + carry;
        acc[i] = sum as u64;
        carry = sum >> 64;
    }
    let mut idx = 4;
    while carry > 0 && idx < 8 {
        let sum = (acc[idx] as u128) + carry;
        acc[idx] = sum as u64;
        carry = sum >> 64;
        idx += 1;
    }

    let mut wide = [0u8; 64];
    for i in 0..8 {
        wide[i * 8..(i + 1) * 8].copy_from_slice(&acc[i].to_le_bytes());
    }
    reduce_512_mod_l(&wide)
}

/// RFC 8032 Ed25519 signing keypair derived from a 32-byte seed.
#[derive(Clone)]
pub struct Ed25519Keypair {
    seed: [u8; 32],
    scalar: [u8; 32],
    prefix: [u8; 32],
    public_key: [u8; 32],
}

impl std::fmt::Debug for Ed25519Keypair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ed25519Keypair")
            .field("public_key", &hex_encode(&self.public_key))
            .finish_non_exhaustive()
    }
}

impl Ed25519Keypair {
    /// Derives an Ed25519 keypair from a 32-byte seed per RFC 8032 §5.1.5.
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let h = sha512(&seed);
        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&h[0..32]);
        scalar[0] &= 248;
        scalar[31] &= 127;
        scalar[31] |= 64;

        let mut prefix = [0u8; 32];
        prefix.copy_from_slice(&h[32..64]);

        let public_key = curve().base.scalar_mul(&scalar).encode();
        Self {
            seed,
            scalar,
            prefix,
            public_key,
        }
    }

    /// Returns the 32-byte seed (for sovereign operator key export/backup).
    pub fn seed(&self) -> [u8; 32] {
        self.seed
    }

    /// Returns the 32-byte compressed Ed25519 public key.
    pub fn public_key(&self) -> [u8; 32] {
        self.public_key
    }

    /// Signs `message` deterministically per RFC 8032 §5.1.6.
    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        let mut h_r = Sha512::new();
        h_r.update(&self.prefix);
        h_r.update(message);
        let r = reduce_512_mod_l(&h_r.finalize());

        let r_enc = curve().base.scalar_mul(&r).encode();

        let mut h_k = Sha512::new();
        h_k.update(&r_enc);
        h_k.update(&self.public_key);
        h_k.update(message);
        let k = reduce_512_mod_l(&h_k.finalize());

        let s = mul_add_mod_l(&k, &self.scalar, &r);

        let mut sig = [0u8; 64];
        sig[0..32].copy_from_slice(&r_enc);
        sig[32..64].copy_from_slice(&s);
        sig
    }
}

/// Strict RFC 8032 §5.1.7 Ed25519 signature verification.
///
/// Rejects non-canonical public keys, non-canonical `R`, malleable `S >= L`,
/// and invalid curve points.
pub fn ed25519_verify(public_key: &[u8; 32], message: &[u8], signature: &[u8; 64]) -> bool {
    let mut r_bytes = [0u8; 32];
    r_bytes.copy_from_slice(&signature[0..32]);
    let mut s_bytes = [0u8; 32];
    s_bytes.copy_from_slice(&signature[32..64]);

    if !scalar_is_canonical(&s_bytes) {
        return false;
    }

    let Some(a_point) = ExtendedPoint::decode(public_key) else {
        return false;
    };
    let Some(r_point) = ExtendedPoint::decode(&r_bytes) else {
        return false;
    };

    let mut h_k = Sha512::new();
    h_k.update(&r_bytes);
    h_k.update(public_key);
    h_k.update(message);
    let k = reduce_512_mod_l(&h_k.finalize());

    let lhs = curve().base.scalar_mul(&s_bytes);
    let rhs = r_point.add(a_point.scalar_mul(&k));
    lhs.eq_ct(rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_arr<const N: usize>(hex: &str) -> [u8; N] {
        let v = hex_decode(hex).expect("valid hex");
        assert_eq!(v.len(), N);
        let mut out = [0u8; N];
        out.copy_from_slice(&v);
        out
    }

    #[test]
    fn sha256_nist_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn sha512_nist_vectors() {
        assert_eq!(
            hex_encode(&sha512(b"abc")),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
             2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );
    }

    #[test]
    fn ed25519_rfc8032_test_vector_1_empty() {
        let seed: [u8; 32] =
            decode_arr("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
        let expected_pk: [u8; 32] =
            decode_arr("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
        let expected_sig: [u8; 64] = decode_arr(
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155\
             5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
        );

        let kp = Ed25519Keypair::from_seed(seed);
        assert_eq!(kp.public_key(), expected_pk);
        let sig = kp.sign(b"");
        assert_eq!(sig, expected_sig);
        assert!(ed25519_verify(&expected_pk, b"", &sig));
        assert!(!ed25519_verify(&expected_pk, b"x", &sig));
    }

    #[test]
    fn ed25519_rfc8032_test_vector_2_one_byte() {
        let seed: [u8; 32] =
            decode_arr("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb");
        let expected_pk: [u8; 32] =
            decode_arr("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c");
        let expected_sig: [u8; 64] = decode_arr(
            "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da\
             085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
        );

        let kp = Ed25519Keypair::from_seed(seed);
        assert_eq!(kp.public_key(), expected_pk);
        let sig = kp.sign(&[0x72]);
        assert_eq!(sig, expected_sig);
        assert!(ed25519_verify(&expected_pk, &[0x72], &sig));
    }

    #[test]
    fn ed25519_rfc8032_test_vector_3_two_bytes() {
        let seed: [u8; 32] =
            decode_arr("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7");
        let expected_pk: [u8; 32] =
            decode_arr("fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025");
        let expected_sig: [u8; 64] = decode_arr(
            "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac\
             18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
        );

        let kp = Ed25519Keypair::from_seed(seed);
        assert_eq!(kp.public_key(), expected_pk);
        let sig = kp.sign(&[0xaf, 0x82]);
        assert_eq!(sig, expected_sig);
        assert!(ed25519_verify(&expected_pk, &[0xaf, 0x82], &sig));
    }

    #[test]
    fn ed25519_rejects_malleability_and_bitflips() {
        let kp = Ed25519Keypair::from_seed([0x42; 32]);
        let msg = b"hanzo-provider-daemon-receipt-v1";
        let sig = kp.sign(msg);
        assert!(ed25519_verify(&kp.public_key(), msg, &sig));

        // Flip every byte of signature and verify rejection.
        for idx in [0usize, 15, 31, 32, 48, 63] {
            let mut bad = sig;
            bad[idx] ^= 0x01;
            assert!(!ed25519_verify(&kp.public_key(), msg, &bad));
        }

        // Non-canonical S >= L (set top bit of S).
        let mut malleable = sig;
        malleable[63] |= 0x10;
        assert!(!ed25519_verify(&kp.public_key(), msg, &malleable));

        // Non-canonical public key y >= p (all 0xff with sign bit cleared).
        let mut bad_pk = [0xffu8; 32];
        bad_pk[31] = 0x7f;
        assert!(!ed25519_verify(&bad_pk, msg, &sig));
    }
}
