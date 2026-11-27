// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust FIPS 180-4 SHA-256, RFC 2104 HMAC-SHA256, constant-time equality,
//! hex encoding, and deterministic SplitMix64 PRNG with zero external dependencies.

/// Length in bytes of a SHA-256 / HMAC-SHA256 digest.
pub const DIGEST_LEN: usize = 32;

const BLOCK_LEN: usize = 64;

const INITIAL_STATE: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Streaming FIPS 180-4 SHA-256 hasher.
#[derive(Clone, Debug)]
pub struct Sha256 {
    state: [u32; 8],
    buffer: [u8; BLOCK_LEN],
    buffer_len: usize,
    total_bits: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Creates a new SHA-256 hasher initialized with the FIPS 180-4 IV.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: INITIAL_STATE,
            buffer: [0u8; BLOCK_LEN],
            buffer_len: 0,
            total_bits: 0,
        }
    }

    /// Feeds arbitrary bytes into the hash state.
    pub fn update(&mut self, mut input: &[u8]) {
        self.total_bits = self.total_bits.wrapping_add((input.len() as u64) * 8);

        if self.buffer_len > 0 {
            let take = (BLOCK_LEN - self.buffer_len).min(input.len());
            self.buffer[self.buffer_len..self.buffer_len + take].copy_from_slice(&input[..take]);
            self.buffer_len += take;
            input = &input[take..];
            if self.buffer_len == BLOCK_LEN {
                let block = self.buffer;
                Self::compress(&mut self.state, &block);
                self.buffer_len = 0;
            }
        }

        while input.len() >= BLOCK_LEN {
            let mut block = [0u8; BLOCK_LEN];
            block.copy_from_slice(&input[..BLOCK_LEN]);
            Self::compress(&mut self.state, &block);
            input = &input[BLOCK_LEN..];
        }

        if !input.is_empty() {
            self.buffer[..input.len()].copy_from_slice(input);
            self.buffer_len = input.len();
        }
    }

    /// Finalizes the hash and returns the 32-byte SHA-256 digest.
    #[must_use]
    pub fn finalize(mut self) -> [u8; DIGEST_LEN] {
        let bit_len = self.total_bits;
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        if self.buffer_len > 56 {
            self.buffer[self.buffer_len..BLOCK_LEN].fill(0);
            let block = self.buffer;
            Self::compress(&mut self.state, &block);
            self.buffer_len = 0;
        }

        self.buffer[self.buffer_len..56].fill(0);
        self.buffer[56..BLOCK_LEN].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        Self::compress(&mut self.state, &block);

        let mut out = [0u8; DIGEST_LEN];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    /// Finalizes and returns the lowercase hexadecimal digest string.
    #[must_use]
    pub fn finalize_hex(self) -> String {
        to_hex(&self.finalize())
    }

    fn compress(state: &mut [u32; 8], block: &[u8; BLOCK_LEN]) {
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
                .wrapping_add(K[i])
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

/// Computes the 32-byte SHA-256 digest of a byte slice.
#[must_use]
pub fn sha256_bytes(data: &[u8]) -> [u8; DIGEST_LEN] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

/// Computes the lowercase hex SHA-256 digest of a byte slice.
#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    to_hex(&sha256_bytes(data))
}

/// Computes RFC 2104 HMAC-SHA256 over `msg` using `key`.
#[must_use]
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; DIGEST_LEN] {
    let mut key_block = [0u8; BLOCK_LEN];
    if key.len() > BLOCK_LEN {
        let hashed = sha256_bytes(key);
        key_block[..DIGEST_LEN].copy_from_slice(&hashed);
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; BLOCK_LEN];
    let mut opad = [0x5cu8; BLOCK_LEN];
    for i in 0..BLOCK_LEN {
        ipad[i] ^= key_block[i];
        opad[i] ^= key_block[i];
    }

    let mut inner = Sha256::new();
    inner.update(&ipad);
    inner.update(msg);
    let inner_digest = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(&inner_digest);
    outer.finalize()
}

/// Computes lowercase hex HMAC-SHA256 over `msg` using `key`.
#[must_use]
pub fn hmac_sha256_hex(key: &[u8], msg: &[u8]) -> String {
    to_hex(&hmac_sha256(key, msg))
}

/// Constant-time equality comparison for equal-length byte slices.
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

/// Verifies an expected lowercase hex HMAC-SHA256 tag in constant time.
#[must_use]
pub fn verify_hmac_sha256_hex(key: &[u8], msg: &[u8], expected_hex: &str) -> bool {
    let Some(expected) = from_hex(expected_hex) else {
        return false;
    };
    let actual = hmac_sha256(key, msg);
    ct_eq(&actual, &expected)
}

/// Encodes bytes into a lowercase hexadecimal string.
#[must_use]
pub fn to_hex(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(LUT[(b >> 4) as usize] as char);
        out.push(LUT[(b & 0x0f) as usize] as char);
    }
    out
}

/// Decodes a hexadecimal string into bytes, returning `None` on malformed input.
#[must_use]
pub fn from_hex(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    let bytes = hex.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let (pairs, _) = bytes.as_chunks::<2>();
    for pair in pairs {
        let hi = hex_nibble(pair[0])?;
        let lo = hex_nibble(pair[1])?;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

const fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Deterministic SplitMix64 pseudorandom number generator for reproducible experiments.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Constructs a new generator initialized with `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the state and returns the next 64-bit unsigned integer.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Returns a deterministic `f32` uniformly sampled in `[-1.0, 1.0)`.
    pub fn next_f32_signed(&mut self) -> f32 {
        let unit = ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32);
        unit * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_and_hmac_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // RFC 4231 Test Case 2: key = "Jefe", data = "what do ya want for nothing?"
        assert_eq!(
            hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?"),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert!(verify_hmac_sha256_hex(
            b"Jefe",
            b"what do ya want for nothing?",
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        ));
    }
}
