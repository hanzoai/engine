// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Clean-room FIPS 180-4 SHA-256, constant-time digest comparison, Merkle trace
//! commitments, hex codecs, and deterministic PRNG (zero external dependencies).

use crate::error::OpAiError;

/// Canonical 32-byte SHA-256 digest.
pub type Digest = [u8; 32];

/// Zero digest sentinel (used only for explicit null/absent fields where documented).
pub const ZERO_DIGEST: Digest = [0u8; 32];

/// Domain separator for execution profile commitments.
pub const DOMAIN_PROFILE: &[u8] = b"hanzo.opai.v1.profile";
/// Domain separator for quantized model weight commitments.
pub const DOMAIN_MODEL: &[u8] = b"hanzo.opai.v1.model";
/// Domain separator for prompt/input commitments.
pub const DOMAIN_INPUT: &[u8] = b"hanzo.opai.v1.input";
/// Domain separator for per-step VM state roots.
pub const DOMAIN_STATE: &[u8] = b"hanzo.opai.v1.state";
/// Domain separator for trace Merkle tree leaves.
pub const DOMAIN_MERKLE_LEAF: &[u8] = b"hanzo.opai.v1.merkle_leaf";
/// Domain separator for trace Merkle tree internal nodes.
pub const DOMAIN_MERKLE_NODE: &[u8] = b"hanzo.opai.v1.merkle_node";
/// Domain separator for provider claim identifiers.
pub const DOMAIN_CLAIM: &[u8] = b"hanzo.opai.v1.claim";
/// Domain separator for dispute identifiers.
pub const DOMAIN_DISPUTE: &[u8] = b"hanzo.opai.v1.dispute";
/// Domain separator for sovereign dispute transcripts.
pub const DOMAIN_TRANSCRIPT: &[u8] = b"hanzo.opai.v1.transcript";

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

/// Streaming FIPS 180-4 SHA-256 hasher (clean-room, zero external crates).
#[derive(Debug, Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    total_bits: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Create a fresh SHA-256 context initialized with standard FIPS 180-4 IV.
    pub fn new() -> Self {
        Self {
            state: SHA256_H0,
            buffer: [0u8; 64],
            buffer_len: 0,
            total_bits: 0,
        }
    }

    /// Create a domain-separated SHA-256 context by length-prefixing `domain`.
    pub fn with_domain(domain: &[u8]) -> Self {
        let mut h = Self::new();
        h.update_u64_le(domain.len() as u64);
        h.update(domain);
        h
    }

    /// Absorb an arbitrary byte slice into the hash state.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bits = self.total_bits.wrapping_add((data.len() as u64) << 3);

        if self.buffer_len > 0 {
            let take = (64 - self.buffer_len).min(data.len());
            self.buffer[self.buffer_len..self.buffer_len + take].copy_from_slice(&data[..take]);
            self.buffer_len += take;
            data = &data[take..];
            if self.buffer_len == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffer_len = 0;
            }
        }

        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }

        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffer_len = data.len();
        }
    }

    /// Absorb a `u8`.
    pub fn update_u8(&mut self, v: u8) {
        self.update(&[v]);
    }

    /// Absorb a little-endian `u32`.
    pub fn update_u32_le(&mut self, v: u32) {
        self.update(&v.to_le_bytes());
    }

    /// Absorb a little-endian `i32`.
    pub fn update_i32_le(&mut self, v: i32) {
        self.update(&v.to_le_bytes());
    }

    /// Absorb a little-endian `u64`.
    pub fn update_u64_le(&mut self, v: u64) {
        self.update(&v.to_le_bytes());
    }

    /// Absorb a length-prefixed byte slice (prevents canonical boundary ambiguity).
    pub fn update_len_prefixed(&mut self, bytes: &[u8]) {
        self.update_u64_le(bytes.len() as u64);
        self.update(bytes);
    }

    /// Absorb a length-prefixed UTF-8 string.
    pub fn update_str(&mut self, s: &str) {
        self.update_len_prefixed(s.as_bytes());
    }

    /// Finalize and return the 32-byte SHA-256 digest.
    pub fn finalize(mut self) -> Digest {
        let bit_len = self.total_bits;
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        if self.buffer_len > 56 {
            for b in &mut self.buffer[self.buffer_len..64] {
                *b = 0;
            }
            let block = self.buffer;
            self.compress(&block);
            self.buffer = [0u8; 64];
            self.buffer_len = 0;
        } else {
            for b in &mut self.buffer[self.buffer_len..56] {
                *b = 0;
            }
        }

        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    /// One-shot SHA-256 over a byte slice.
    pub fn digest(data: &[u8]) -> Digest {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
    }

    fn compress(&mut self, block: &[u8; 64]) {
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

/// Constant-time equality comparison for 32-byte digests.
pub fn ct_eq_digest(a: &Digest, b: &Digest) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Format a 32-byte digest as a 64-character lowercase hexadecimal string.
pub fn digest_to_hex(d: &Digest) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &byte in d {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Parse a 64-character hexadecimal string into a 32-byte digest.
pub fn hex_to_digest(s: &str) -> Result<Digest, OpAiError> {
    let bytes = s.as_bytes();
    if bytes.len() != 64 {
        return Err(OpAiError::WireFormat(format!(
            "expected 64 hex chars for digest, got {}",
            bytes.len()
        )));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = decode_nibble(bytes[2 * i])?;
        let lo = decode_nibble(bytes[2 * i + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn decode_nibble(b: u8) -> Result<u8, OpAiError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(OpAiError::WireFormat(format!(
            "invalid hex byte: 0x{b:02x}"
        ))),
    }
}

/// Domain-separated Merkle inclusion proof for a single step root in `0..=T`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleProof {
    /// Step index (`0..=T`) proved by this inclusion path.
    pub leaf_index: u64,
    /// Total number of leaves (`T + 1`) in the trace tree.
    pub leaf_count: u64,
    /// Sibling digests from bottom level up to the root.
    pub siblings: Vec<Digest>,
}

/// Domain-separated binary SHA-256 Merkle tree over step state roots `0..=T`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceMerkleTree {
    leaf_count: usize,
    levels: Vec<Vec<Digest>>,
}

impl TraceMerkleTree {
    /// Build a binary Merkle tree from non-empty step state roots `state_roots[0..=T]`.
    pub fn build(state_roots: &[Digest]) -> Result<Self, OpAiError> {
        if state_roots.is_empty() {
            return Err(OpAiError::InvalidClaim(
                "cannot build trace Merkle tree from empty state_roots".to_string(),
            ));
        }
        let mut current: Vec<Digest> = state_roots
            .iter()
            .enumerate()
            .map(|(idx, r)| hash_merkle_leaf(idx as u64, r))
            .collect();
        let mut levels = vec![current.clone()];
        while current.len() > 1 {
            let mut next = Vec::with_capacity(current.len().div_ceil(2));
            let mut i = 0;
            while i < current.len() {
                let left = current[i];
                let right = if i + 1 < current.len() {
                    current[i + 1]
                } else {
                    left
                };
                next.push(hash_merkle_node(&left, &right));
                i += 2;
            }
            levels.push(next.clone());
            current = next;
        }
        Ok(Self {
            leaf_count: state_roots.len(),
            levels,
        })
    }

    /// Return the canonical Merkle root digest.
    pub fn root(&self) -> Digest {
        self.levels
            .last()
            .and_then(|lvl| lvl.first())
            .copied()
            .unwrap_or(ZERO_DIGEST)
    }

    /// Generate an inclusion proof for `leaf_index`.
    pub fn prove(&self, leaf_index: usize) -> Result<MerkleProof, OpAiError> {
        if leaf_index >= self.leaf_count {
            return Err(OpAiError::InvalidStepIndex {
                step: leaf_index as u64,
                total_steps: self.leaf_count.saturating_sub(1) as u64,
            });
        }
        let mut siblings = Vec::with_capacity(self.levels.len().saturating_sub(1));
        let mut idx = leaf_index;
        for level in &self.levels[..self.levels.len() - 1] {
            let sibling_idx = if idx.is_multiple_of(2) {
                if idx + 1 < level.len() {
                    idx + 1
                } else {
                    idx
                }
            } else {
                idx - 1
            };
            siblings.push(level[sibling_idx]);
            idx /= 2;
        }
        Ok(MerkleProof {
            leaf_index: leaf_index as u64,
            leaf_count: self.leaf_count as u64,
            siblings,
        })
    }

    /// Verify an inclusion proof for `state_root` at `proof.leaf_index` against `expected_root`.
    pub fn verify_proof(expected_root: &Digest, state_root: &Digest, proof: &MerkleProof) -> bool {
        if proof.leaf_count == 0 || proof.leaf_index >= proof.leaf_count {
            return false;
        }
        let mut expected_depth = 0usize;
        let mut width = proof.leaf_count;
        while width > 1 {
            width = width.div_ceil(2);
            expected_depth += 1;
        }
        if proof.siblings.len() != expected_depth {
            return false;
        }
        let mut acc = hash_merkle_leaf(proof.leaf_index, state_root);
        let mut idx = proof.leaf_index;
        for sibling in &proof.siblings {
            acc = if idx.is_multiple_of(2) {
                hash_merkle_node(&acc, sibling)
            } else {
                hash_merkle_node(sibling, &acc)
            };
            idx /= 2;
        }
        ct_eq_digest(&acc, expected_root)
    }
}

fn hash_merkle_leaf(step_index: u64, state_root: &Digest) -> Digest {
    let mut h = Sha256::with_domain(DOMAIN_MERKLE_LEAF);
    h.update_u64_le(step_index);
    h.update(state_root);
    h.finalize()
}

fn hash_merkle_node(left: &Digest, right: &Digest) -> Digest {
    let mut h = Sha256::with_domain(DOMAIN_MERKLE_NODE);
    h.update(left);
    h.update(right);
    h.finalize()
}

/// Deterministic `SplitMix64` PRNG for reproducible model synthesis and benchmark sweeps.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a deterministic PRNG seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Draw the next pseudorandom `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Draw a uniform-ish value in `0..bound` (panics if `bound == 0`).
    pub fn next_bounded(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "SplitMix64::next_bounded requires bound > 0");
        self.next_u64() % bound
    }

    /// Draw a signed `i8` in `[-limit, limit]` (`limit <= 127`).
    pub fn next_i8_symmetric(&mut self, limit: i8) -> i8 {
        let span = (i16::from(limit) * 2 + 1) as u64;
        let v = (self.next_u64() % span) as i16 - i16::from(limit);
        v as i8
    }

    /// Draw a signed `i32` in `[-limit, limit]`.
    pub fn next_i32_symmetric(&mut self, limit: i32) -> i32 {
        let span = (i64::from(limit) * 2 + 1) as u64;
        let v = (self.next_u64() % span) as i64 - i64::from(limit);
        v as i32
    }
}
