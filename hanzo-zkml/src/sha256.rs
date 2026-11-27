// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Clean-room NIST FIPS 180-4 SHA-256 and domain-separated Fiat-Shamir transcript.

use crate::field::{Fp, Fp2, MODULUS};

/// Initial SHA-256 hash values (first 32 bits of the fractional parts of the
/// square roots of the first 8 primes).
const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// SHA-256 round constants (first 32 bits of the fractional parts of the cube
/// roots of the first 64 primes).
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

/// Incremental SHA-256 hasher (NIST FIPS 180-4).
#[derive(Clone, Debug)]
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_bytes: u64,
}

impl Sha256 {
    /// Creates a new SHA-256 hasher.
    pub const fn new() -> Self {
        Self {
            state: H0,
            buf: [0u8; 64],
            buf_len: 0,
            total_bytes: 0,
        }
    }

    /// Feeds bytes into the hasher.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(data.len() as u64);

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

    /// Finalizes the hash and returns the 32-byte digest.
    pub fn finalize(mut self) -> [u8; 32] {
        let total_bits = self.total_bytes.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        if self.buf_len > 56 {
            for b in &mut self.buf[self.buf_len..64] {
                *b = 0;
            }
            let block = self.buf;
            Self::compress(&mut self.state, &block);
            self.buf_len = 0;
        }

        for b in &mut self.buf[self.buf_len..56] {
            *b = 0;
        }
        self.buf[56..64].copy_from_slice(&total_bits.to_be_bytes());
        let block = self.buf;
        Self::compress(&mut self.state, &block);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
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

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes SHA-256 of a single byte slice.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

/// Formats a byte slice as lowercase hexadecimal.
pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

/// Decodes a lowercase/uppercase 64-character hex string into a 32-byte digest.
pub fn hex_decode_32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let bytes = s.as_bytes();
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = hex_nibble(bytes[2 * i])?;
        let lo = hex_nibble(bytes[2 * i + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Domain-separated Fiat-Shamir transcript powered by SHA-256.
///
/// Every absorbed message is length-delimited and labeled to prevent canonicalization
/// ambiguity and cross-protocol replay attacks.
#[derive(Clone, Debug)]
pub struct Transcript {
    state: [u8; 32],
    squeeze_counter: u64,
}

impl Transcript {
    /// Initializes a new transcript bound to an explicit domain separation tag.
    pub fn new(domain_tag: &str) -> Self {
        let mut h = Sha256::new();
        h.update(b"hanzo.zkml.transcript.init.v1");
        h.update(&(domain_tag.len() as u64).to_le_bytes());
        h.update(domain_tag.as_bytes());
        Self {
            state: h.finalize(),
            squeeze_counter: 0,
        }
    }

    /// Absorbs a labeled byte payload into the transcript state.
    pub fn absorb_bytes(&mut self, label: &str, data: &[u8]) {
        let mut h = Sha256::new();
        h.update(&self.state);
        h.update(b"A");
        h.update(&(label.len() as u64).to_le_bytes());
        h.update(label.as_bytes());
        h.update(&(data.len() as u64).to_le_bytes());
        h.update(data);
        self.state = h.finalize();
        self.squeeze_counter = 0;
    }

    /// Absorbs a labeled `u64`.
    pub fn absorb_u64(&mut self, label: &str, val: u64) {
        self.absorb_bytes(label, &val.to_le_bytes());
    }

    /// Absorbs a labeled `i64` slice.
    pub fn absorb_i64_slice(&mut self, label: &str, vals: &[i64]) {
        let mut buf = Vec::with_capacity(8 + vals.len() * 8);
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for &v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        self.absorb_bytes(label, &buf);
    }

    /// Absorbs a labeled `Fp` element.
    pub fn absorb_fp(&mut self, label: &str, val: Fp) {
        self.absorb_bytes(label, &val.to_le_bytes());
    }

    /// Absorbs a labeled `Fp2` element.
    pub fn absorb_fp2(&mut self, label: &str, val: Fp2) {
        self.absorb_bytes(label, &val.to_le_bytes());
    }

    /// Absorbs a labeled slice of `Fp2` elements.
    pub fn absorb_fp2_slice(&mut self, label: &str, vals: &[Fp2]) {
        let mut buf = Vec::with_capacity(8 + vals.len() * 16);
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for &v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        self.absorb_bytes(label, &buf);
    }

    /// Squeezes a 32-byte pseudorandom block and ratchets the transcript state.
    fn squeeze_block(&mut self, label: &str) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(&self.state);
        h.update(b"C");
        h.update(&(label.len() as u64).to_le_bytes());
        h.update(label.as_bytes());
        h.update(&self.squeeze_counter.to_le_bytes());
        let block = h.finalize();
        self.squeeze_counter = self.squeeze_counter.wrapping_add(1);

        // Ratchet state so future absorbs depend on prior challenges.
        let mut ratchet = Sha256::new();
        ratchet.update(&self.state);
        ratchet.update(b"R");
        ratchet.update(&block);
        self.state = ratchet.finalize();
        block
    }

    /// Samples an exact uniformly distributed `Fp` challenge via rejection sampling.
    pub fn challenge_fp(&mut self, label: &str) -> Fp {
        loop {
            let blk = self.squeeze_block(label);
            for &arr in blk.as_chunks::<8>().0 {
                let v = u64::from_le_bytes(arr);
                if v < MODULUS {
                    return Fp::new(v);
                }
            }
        }
    }

    /// Samples a uniformly distributed `Fp2` challenge (`~128` bits of entropy).
    pub fn challenge_fp2(&mut self, label: &str) -> Fp2 {
        let c0 = self.challenge_fp(label);
        let c1 = self.challenge_fp(label);
        Fp2::new(c0, c1)
    }

    /// Samples `count` independent `Fp2` challenges.
    pub fn challenge_fp2_vec(&mut self, label: &str, count: usize) -> Vec<Fp2> {
        (0..count).map(|_| self.challenge_fp2(label)).collect()
    }

    /// Samples `count` distinct column query indices in `0..domain_size` (or all
    /// `domain_size` indices if `count >= domain_size`).
    pub fn challenge_distinct_indices(
        &mut self,
        label: &str,
        count: usize,
        domain_size: usize,
    ) -> Vec<usize> {
        assert!(domain_size > 0);
        let target = count.min(domain_size);
        let mut chosen = Vec::with_capacity(target);
        let mut seen = vec![false; domain_size];
        while chosen.len() < target {
            let blk = self.squeeze_block(label);
            for &arr in blk.as_chunks::<8>().0 {
                if chosen.len() == target {
                    break;
                }
                // domain_size is a small power of two so masking is exact uniform.
                let idx = if domain_size.is_power_of_two() {
                    (u64::from_le_bytes(arr) as usize) & (domain_size - 1)
                } else {
                    (u64::from_le_bytes(arr) as usize) % domain_size
                };
                if !seen[idx] {
                    seen[idx] = true;
                    chosen.push(idx);
                }
            }
        }
        chosen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nist_fips_180_4_vectors() {
        assert_eq!(
            hex_encode(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex_encode(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex_encode(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn transcript_domain_separation_prevents_replay() {
        let mut t1 = Transcript::new("hanzo.zkml.circuit.v1");
        let mut t2 = Transcript::new("hanzo.zkml.zkvm.v1");
        t1.absorb_u64("x", 42);
        t2.absorb_u64("x", 42);
        assert_ne!(t1.challenge_fp2("r"), t2.challenge_fp2("r"));
    }
}
