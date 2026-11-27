// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Clean-room FIPS 180-4 SHA-256 and domain-separated Fiat-Shamir / interactive
//! challenge transcript for `poai.matmul.v1`.
//!
//! Zero external dependencies: implements standard SHA-256 verified against NIST FIPS 180-4
//! test vectors, canonical matrix commitments, and unbiased uniform non-zero challenge
//! sampling in `GF(2^61 - 1)^n \ {0}`.

use std::fs::File;
use std::io::Read;

use crate::error::PoaiError;
use crate::field::{Fp, MODULUS_P};
use crate::profile::{OperatorKind, VerificationProfile};
use crate::quant::{ActivationMatrix, OutputMatrix, PackedWeightMatrix, QUANT_BLOCK_SIZE};

/// SHA-256 initial hash values (FIPS 180-4 section 5.3.3).
const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-256 round constants (FIPS 180-4 section 4.2.2).
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

/// Streaming clean-room FIPS 180-4 SHA-256 hasher.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    total_bits: u128,
}

impl Sha256 {
    /// Initialize a fresh SHA-256 state.
    pub const fn new() -> Self {
        Self {
            state: SHA256_IV,
            block: [0u8; 64],
            block_len: 0,
            total_bits: 0,
        }
    }

    /// One-shot SHA-256 helper.
    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
    }

    /// Absorb `data` into the SHA-256 state.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bits += (data.len() as u128) * 8;

        if self.block_len > 0 {
            let take = (64 - self.block_len).min(data.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&data[..take]);
            self.block_len += take;
            data = &data[take..];
            if self.block_len == 64 {
                let block = self.block;
                self.compress(&block);
                self.block_len = 0;
            }
        }

        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }

        if !data.is_empty() {
            self.block[..data.len()].copy_from_slice(data);
            self.block_len = data.len();
        }
    }

    /// Finalize and return the 32-byte digest.
    pub fn finalize(mut self) -> [u8; 32] {
        let total_bits = self.total_bits as u64;
        self.block[self.block_len] = 0x80;
        self.block_len += 1;

        if self.block_len > 56 {
            for b in &mut self.block[self.block_len..64] {
                *b = 0;
            }
            let block = self.block;
            self.compress(&block);
            self.block_len = 0;
        }

        for b in &mut self.block[self.block_len..56] {
            *b = 0;
        }
        self.block[56..64].copy_from_slice(&total_bits.to_be_bytes());
        let block = self.block;
        self.compress(&block);

        let mut out = [0u8; 32];
        for (i, &word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    #[inline]
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

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Challenge derivation mode for Freivalds verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChallengeMode {
    /// Non-interactive Fiat-Shamir heuristic bound to
    /// `(profile, op, p, m, k, n, rounds, commit_A, commit_B, commit_C)`.
    ///
    /// Note: With `rounds = 2` (`p^2 > 2^121`), grinding `C` to pass Fiat-Shamir requires `> 2^121`
    /// SHA-256 evaluations. For interactive or beacon-settled verification where zero grinding
    /// assumption is desired, prefer [`ChallengeMode::InteractiveSeed`].
    FiatShamir,
    /// Interactive / verifier-seeded mode combining the full commitment transcript with a fresh
    /// 256-bit verifier CSPRNG seed supplied *after* the prover commits to `commit_C`.
    InteractiveSeed([u8; 32]),
}

/// Read 32 cryptographically strong random bytes from `/dev/urandom` on Unix for interactive challenges.
pub fn os_random_seed() -> Result<[u8; 32], PoaiError> {
    let mut seed = [0u8; 32];
    let mut f = File::open("/dev/urandom").map_err(|e| PoaiError::EntropyUnavailable {
        reason: e.to_string(),
    })?;
    f.read_exact(&mut seed)
        .map_err(|e| PoaiError::EntropyUnavailable {
            reason: e.to_string(),
        })?;
    Ok(seed)
}

/// Domain-separated SHA-256 commitment to activation matrix `A` (`M x K`, `i8`).
pub fn commit_activation_matrix(a: &ActivationMatrix) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"hanzo.poai.commit.A.v1");
    h.update(&(a.rows() as u64).to_le_bytes());
    h.update(&(a.cols() as u64).to_le_bytes());
    // Cast &[i8] to &[u8] safely in fixed chunks without unsafe code.
    let mut buf = [0u8; 4096];
    for chunk in a.data().chunks(buf.len()) {
        for (dst, &src) in buf.iter_mut().zip(chunk.iter()) {
            *dst = src as u8;
        }
        h.update(&buf[..chunk.len()]);
    }
    h.finalize()
}

/// Domain-separated SHA-256 commitment to packed quantized weight matrix `B` (`K x N`).
///
/// Binds format tag, block size (`32`), dimensions `(K, N)`, all block scales, and all packed nibbles/bits.
/// Because LLM model weights are static across requests, this digest can be computed once at model
/// load time and reused across inferences.
pub fn commit_weight_matrix(b: &PackedWeightMatrix) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"hanzo.poai.commit.B.v1");
    h.update(&[b.format().wire_id()]);
    h.update(&(QUANT_BLOCK_SIZE as u32).to_le_bytes());
    h.update(&(b.rows() as u64).to_le_bytes());
    h.update(&(b.cols() as u64).to_le_bytes());
    h.update(b.raw_bytes());
    h.finalize()
}

/// Domain-separated SHA-256 commitment to exact integer output matrix `C` (`M x N`, `i64`).
pub fn commit_output_matrix(c: &OutputMatrix) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"hanzo.poai.commit.C.v1");
    h.update(&(c.rows() as u64).to_le_bytes());
    h.update(&(c.cols() as u64).to_le_bytes());
    let mut buf = [0u8; 4096];
    for chunk in c.data().chunks(buf.len() / 8) {
        for (idx, &val) in chunk.iter().enumerate() {
            buf[idx * 8..idx * 8 + 8].copy_from_slice(&val.to_le_bytes());
        }
        h.update(&buf[..chunk.len() * 8]);
    }
    h.finalize()
}

/// Compute the compact round digest committing to a projected vector `y = C * r in GF(p)^m`.
pub fn commit_round_projection(round_idx: u32, projection: &[Fp]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"hanzo.poai.round.proj.v1");
    h.update(&round_idx.to_le_bytes());
    h.update(&(projection.len() as u64).to_le_bytes());
    for &elem in projection {
        h.update(&elem.to_le_bytes());
    }
    h.finalize()
}

/// Derive the transcript root seed binding profile, operator, modulus, dimensions, round count,
/// commitments, and challenge mode.
#[allow(clippy::too_many_arguments)]
pub fn derive_transcript_seed(
    profile: &VerificationProfile,
    operator: OperatorKind,
    m: usize,
    k: usize,
    n: usize,
    rounds: usize,
    commit_a: &[u8; 32],
    commit_b: &[u8; 32],
    commit_c: &[u8; 32],
    mode: ChallengeMode,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"hanzo.poai.transcript.v1");
    h.update(&(profile.version.len() as u32).to_le_bytes());
    h.update(profile.version.as_bytes());
    h.update(&[operator.wire_tag()]);
    h.update(&profile.field_modulus_p.to_le_bytes());
    h.update(&(m as u64).to_le_bytes());
    h.update(&(k as u64).to_le_bytes());
    h.update(&(n as u64).to_le_bytes());
    h.update(&(rounds as u32).to_le_bytes());
    h.update(commit_a);
    h.update(commit_b);
    h.update(commit_c);
    match mode {
        ChallengeMode::FiatShamir => {
            h.update(&[0u8]);
        }
        ChallengeMode::InteractiveSeed(seed) => {
            h.update(&[1u8]);
            h.update(&seed);
        }
    }
    h.finalize()
}

/// Derive a uniform non-zero challenge vector `r in GF(p)^n \ {0}` for round `round_idx`.
///
/// Uses rejection sampling over 61-bit slices of SHA-256 counter blocks, rejecting only the single
/// out-of-range value `2^61 - 1` (probability `2^-61`) so every coordinate is strictly uniform on
/// `GF(p)` with zero modulo bias; if all `n` coordinates happen to be zero (probability `p^-n`),
/// increments the retry counter and resamples, yielding exact uniformity on `GF(p)^n \ {0}` and
/// guaranteeing per-round soundness error `<= (p^(n-1) - 1) / (p^n - 1) < 1/p`.
pub fn derive_challenge_vector(transcript_seed: &[u8; 32], round_idx: u32, n: usize) -> Vec<Fp> {
    let mut out = Vec::with_capacity(n);
    let mut retry: u32 = 0;

    loop {
        out.clear();
        let mut counter: u64 = 0;
        let mut any_nonzero = false;

        while out.len() < n {
            let mut h = Sha256::new();
            h.update(b"hanzo.poai.challenge.v1");
            h.update(transcript_seed);
            h.update(&round_idx.to_le_bytes());
            h.update(&retry.to_le_bytes());
            h.update(&counter.to_le_bytes());
            let block = h.finalize();
            counter += 1;

            for w_idx in 0..4 {
                if out.len() == n {
                    break;
                }
                let off = w_idx * 8;
                let word = u64::from_le_bytes([
                    block[off],
                    block[off + 1],
                    block[off + 2],
                    block[off + 3],
                    block[off + 4],
                    block[off + 5],
                    block[off + 6],
                    block[off + 7],
                ]);
                let candidate = word & MODULUS_P;
                // Reject `MODULUS_P` (2^61 - 1) for exact zero-bias uniformity on [0, p - 1].
                if candidate == MODULUS_P {
                    continue;
                }
                if candidate != 0 {
                    any_nonzero = true;
                }
                out.push(Fp::from_canonical(candidate).expect("candidate < MODULUS_P"));
            }
        }

        if any_nonzero {
            return out;
        }
        retry += 1;
    }
}

/// Format a 32-byte digest as lowercase hexadecimal.
pub fn digest_to_hex(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for &b in digest {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_nist_fips_180_4_vectors() {
        assert_eq!(
            digest_to_hex(&Sha256::digest(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest_to_hex(&Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            digest_to_hex(&Sha256::digest(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            digest_to_hex(&Sha256::digest(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            )),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );

        // Streaming chunked update must match one-shot exactly across block boundaries.
        let payload = vec![b'a'; 1_000_000];
        let mut h = Sha256::new();
        for chunk in payload.chunks(137) {
            h.update(chunk);
        }
        assert_eq!(
            digest_to_hex(&h.finalize()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn challenge_vectors_are_deterministic_nonzero_and_round_independent() {
        let seed = Sha256::digest(b"test-seed");
        let r0 = derive_challenge_vector(&seed, 0, 64);
        let r0_again = derive_challenge_vector(&seed, 0, 64);
        let r1 = derive_challenge_vector(&seed, 1, 64);
        assert_eq!(r0, r0_again);
        assert_ne!(r0, r1);
        assert!(r0.iter().any(|&x| x != Fp::ZERO));
        assert!(r1.iter().any(|&x| x != Fp::ZERO));
        for &x in r0.iter().chain(r1.iter()) {
            assert!(x.raw() < MODULUS_P);
        }
    }
}
