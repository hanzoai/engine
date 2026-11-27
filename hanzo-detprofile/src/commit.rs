// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Clean-room FIPS 180-4 SHA-256 implementation and canonical
//! `ExecutionReceipt` output commitments.
//!
//! Zero external crates: SHA-256 is implemented directly from NIST FIPS 180-4
//! and validated against the official NIST test vectors in unit tests.

use crate::error::{DetError, DetResult};
use crate::profile::{ExecutionProfile, OverflowMode, ReductionOrder, RoundingMode};
use crate::wire::{WireReader, WireWriter, WIRE_MAGIC};

/// 32-byte SHA-256 digest.
pub type Digest = [u8; 32];

/// Canonical receipt wire format tag (`1` = `ExecutionReceipt` v1).
const RECEIPT_WIRE_TAG: u8 = 0x01;

/// Domain separation prefixes for canonical commitments.
const DOMAIN_OP_GRAPH: &[u8] = b"hanzo.detprofile.v1.op_graph\0";
const DOMAIN_INPUTS: &[u8] = b"hanzo.detprofile.v1.inputs\0";
const DOMAIN_WEIGHTS: &[u8] = b"hanzo.detprofile.v1.weights\0";
const DOMAIN_OUTPUT: &[u8] = b"hanzo.detprofile.v1.output\0";
const DOMAIN_RECEIPT: &[u8] = b"hanzo.detprofile.v1.receipt\0";

/// Clean-room NIST FIPS 180-4 SHA-256 streaming hasher.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    total_len: u64,
}

impl Sha256 {
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

    /// Create a fresh SHA-256 hasher.
    pub fn new() -> Self {
        Self {
            state: Self::H0,
            block: [0u8; 64],
            block_len: 0,
            total_len: 0,
        }
    }

    /// Absorb bytes into the SHA-256 state.
    pub fn update(&mut self, mut input: &[u8]) {
        self.total_len = self.total_len.wrapping_add(input.len() as u64);

        if self.block_len > 0 {
            let need = 64 - self.block_len;
            if input.len() < need {
                self.block[self.block_len..self.block_len + input.len()].copy_from_slice(input);
                self.block_len += input.len();
                return;
            }
            self.block[self.block_len..64].copy_from_slice(&input[..need]);
            let blk = self.block;
            Self::compress(&mut self.state, &blk);
            self.block_len = 0;
            input = &input[need..];
        }

        while input.len() >= 64 {
            let mut blk = [0u8; 64];
            blk.copy_from_slice(&input[..64]);
            Self::compress(&mut self.state, &blk);
            input = &input[64..];
        }

        if !input.is_empty() {
            self.block[..input.len()].copy_from_slice(input);
            self.block_len = input.len();
        }
    }

    /// Finalize and return the 32-byte digest.
    pub fn finalize(mut self) -> Digest {
        let bit_len = self.total_len.wrapping_mul(8);
        self.block[self.block_len] = 0x80;
        self.block_len += 1;

        if self.block_len > 56 {
            self.block[self.block_len..64].fill(0);
            let blk = self.block;
            Self::compress(&mut self.state, &blk);
            self.block_len = 0;
        }
        self.block[self.block_len..56].fill(0);
        self.block[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let blk = self.block;
        Self::compress(&mut self.state, &blk);

        let mut out = [0u8; 32];
        for (i, &word) in self.state.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    /// One-shot convenience SHA-256 digest.
    pub fn digest(data: &[u8]) -> Digest {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
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
        for (&wi, &ki) in w.iter().zip(Self::K.iter()) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(ki)
                .wrapping_add(wi);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
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

/// Constant-time equality comparison for 32-byte SHA-256 digests.
#[inline]
pub fn ct_eq_digest(a: &Digest, b: &Digest) -> bool {
    let mut diff: u8 = 0;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Format a 32-byte digest as lowercase hexadecimal.
pub fn digest_to_hex(d: &Digest) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in d {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Domain-separated SHA-256 digest of a canonical operator graph byte payload.
pub fn commit_op_graph(canonical_graph_bytes: &[u8]) -> Digest {
    let mut h = Sha256::new();
    h.update(DOMAIN_OP_GRAPH);
    h.update(&(canonical_graph_bytes.len() as u64).to_le_bytes());
    h.update(canonical_graph_bytes);
    h.finalize()
}

/// Domain-separated SHA-256 digest of canonical activation input bytes.
pub fn commit_inputs(canonical_input_bytes: &[u8]) -> Digest {
    let mut h = Sha256::new();
    h.update(DOMAIN_INPUTS);
    h.update(&(canonical_input_bytes.len() as u64).to_le_bytes());
    h.update(canonical_input_bytes);
    h.finalize()
}

/// Domain-separated SHA-256 digest of canonical quantized weight bytes.
pub fn commit_weights(canonical_weight_bytes: &[u8]) -> Digest {
    let mut h = Sha256::new();
    h.update(DOMAIN_WEIGHTS);
    h.update(&(canonical_weight_bytes.len() as u64).to_le_bytes());
    h.update(canonical_weight_bytes);
    h.finalize()
}

/// Domain-separated SHA-256 digest of canonical output tensor bytes.
pub fn commit_output(output_tensor_bytes: &[u8]) -> Digest {
    let mut h = Sha256::new();
    h.update(DOMAIN_OUTPUT);
    h.update(&(output_tensor_bytes.len() as u64).to_le_bytes());
    h.update(output_tensor_bytes);
    h.finalize()
}

/// Canonical bit-identical execution receipt binding complete profile metadata,
/// graph, inputs, weights, numerical policies, and output tensor bytes via SHA-256.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionReceipt {
    /// Profile identifier (`det.zen-kai.v1`).
    pub profile_id: String,
    /// Profile major version (`1`).
    pub profile_version: u32,
    /// Domain-separated SHA-256 digest of the canonical operator graph.
    pub op_graph_digest: Digest,
    /// Domain-separated SHA-256 digest of the canonical activation inputs.
    pub input_digest: Digest,
    /// Domain-separated SHA-256 digest of all quantized weights & biases.
    pub weights_digest: Digest,
    /// Explicit rounding policy bound into this execution.
    pub rounding_mode: RoundingMode,
    /// Explicit overflow policy bound into this execution.
    pub overflow_mode: OverflowMode,
    /// Explicit reduction order bound into this execution.
    pub reduction_order: ReductionOrder,
    /// Fixed-point SiLU LUT revision bound into this execution.
    pub silu_lut_version: u32,
    /// Integer RMSNorm epsilon in Q24 fixed-point units bound into this execution.
    pub rmsnorm_eps_q24: u64,
    /// Byte length of canonical `output_tensor_bytes`.
    pub output_byte_len: u64,
    /// Domain-separated SHA-256 digest of `output_tensor_bytes`.
    pub output_digest: Digest,
    /// Master SHA-256 commitment over all fields above AND raw `output_tensor_bytes`.
    pub commitment: Digest,
}

/// Complete context passed to `compute_receipt_commitment` so no profile field
/// can be omitted from the master SHA-256 receipt digest.
struct ReceiptCommitCtx<'a> {
    profile: &'a ExecutionProfile,
    op_graph_digest: &'a Digest,
    input_digest: &'a Digest,
    weights_digest: &'a Digest,
    output_tensor_bytes: &'a [u8],
}

impl ExecutionReceipt {
    /// Construct an `ExecutionReceipt` and compute its canonical SHA-256
    /// commitment over the complete `ExecutionProfile` (including
    /// `silu_lut_version` and `rmsnorm_eps_q24`), graph/input/weights digests,
    /// and raw `output_tensor_bytes`.
    pub fn build(
        profile: &ExecutionProfile,
        op_graph_digest: Digest,
        input_digest: Digest,
        weights_digest: Digest,
        output_tensor_bytes: &[u8],
    ) -> DetResult<Self> {
        profile.validate()?;
        let output_digest = commit_output(output_tensor_bytes);
        let ctx = ReceiptCommitCtx {
            profile,
            op_graph_digest: &op_graph_digest,
            input_digest: &input_digest,
            weights_digest: &weights_digest,
            output_tensor_bytes,
        };
        let commitment = compute_receipt_commitment(&ctx);
        Ok(Self {
            profile_id: profile.profile_id.clone(),
            profile_version: profile.profile_version,
            op_graph_digest,
            input_digest,
            weights_digest,
            rounding_mode: profile.rounding_mode,
            overflow_mode: profile.overflow_mode,
            reduction_order: profile.reduction_order,
            silu_lut_version: profile.silu_lut_version,
            rmsnorm_eps_q24: profile.rmsnorm_eps_q24,
            output_byte_len: output_tensor_bytes.len() as u64,
            output_digest,
            commitment,
        })
    }

    /// Reconstruct the complete [`ExecutionProfile`] bound into this receipt so
    /// an independent verifier can replay the exact numerical semantics.
    pub fn profile(&self) -> ExecutionProfile {
        ExecutionProfile {
            profile_id: self.profile_id.clone(),
            profile_version: self.profile_version,
            rounding_mode: self.rounding_mode,
            overflow_mode: self.overflow_mode,
            reduction_order: self.reduction_order,
            silu_lut_version: self.silu_lut_version,
            rmsnorm_eps_q24: self.rmsnorm_eps_q24,
        }
    }

    /// Verify this receipt against a candidate `output_tensor_bytes` payload
    /// using constant-time SHA-256 digest comparisons.
    pub fn verify_output(&self, output_tensor_bytes: &[u8]) -> DetResult<()> {
        if output_tensor_bytes.len() as u64 != self.output_byte_len {
            return Err(DetError::CommitmentMismatch {
                field: "output_byte_len",
            });
        }
        let expected_out_digest = commit_output(output_tensor_bytes);
        if !ct_eq_digest(&self.output_digest, &expected_out_digest) {
            return Err(DetError::CommitmentMismatch {
                field: "output_digest",
            });
        }
        let profile = self.profile();
        let ctx = ReceiptCommitCtx {
            profile: &profile,
            op_graph_digest: &self.op_graph_digest,
            input_digest: &self.input_digest,
            weights_digest: &self.weights_digest,
            output_tensor_bytes,
        };
        let expected_commit = compute_receipt_commitment(&ctx);
        if !ct_eq_digest(&self.commitment, &expected_commit) {
            return Err(DetError::CommitmentMismatch {
                field: "commitment",
            });
        }
        Ok(())
    }

    /// Encode this receipt into canonical little-endian binary wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = WireWriter::with_capacity(212);
        w.write_raw(&WIRE_MAGIC);
        w.write_u8(RECEIPT_WIRE_TAG);
        w.write_str(&self.profile_id);
        w.write_u32(self.profile_version);
        w.write_raw(&self.op_graph_digest);
        w.write_raw(&self.input_digest);
        w.write_raw(&self.weights_digest);
        w.write_u8(self.rounding_mode.to_u8());
        w.write_u8(self.overflow_mode.to_u8());
        w.write_u8(self.reduction_order.to_u8());
        w.write_u32(self.silu_lut_version);
        w.write_u64(self.rmsnorm_eps_q24);
        w.write_u64(self.output_byte_len);
        w.write_raw(&self.output_digest);
        w.write_raw(&self.commitment);
        w.into_bytes()
    }

    /// Decode an `ExecutionReceipt` from canonical little-endian binary bytes.
    pub fn from_bytes(bytes: &[u8]) -> DetResult<Self> {
        let mut r = WireReader::new(bytes);
        let magic = r.read_array::<4>()?;
        if magic != WIRE_MAGIC {
            return Err(DetError::WireDecodeError {
                offset: 0,
                detail: "invalid DZKP wire magic",
            });
        }
        let tag_off = r.offset();
        let tag = r.read_u8()?;
        if tag != RECEIPT_WIRE_TAG {
            return Err(DetError::WireDecodeError {
                offset: tag_off,
                detail: "unsupported wire envelope tag",
            });
        }
        let profile_id = r.read_str()?;
        let profile_version = r.read_u32()?;
        let op_graph_digest = r.read_array::<32>()?;
        let input_digest = r.read_array::<32>()?;
        let weights_digest = r.read_array::<32>()?;
        let rounding_off = r.offset();
        let rounding_mode = RoundingMode::from_u8(r.read_u8()?, rounding_off)?;
        let overflow_off = r.offset();
        let overflow_mode = OverflowMode::from_u8(r.read_u8()?, overflow_off)?;
        let reduction_off = r.offset();
        let reduction_order = ReductionOrder::from_u8(r.read_u8()?, reduction_off)?;
        let silu_lut_version = r.read_u32()?;
        let rmsnorm_eps_q24 = r.read_u64()?;
        let output_byte_len = r.read_u64()?;
        let output_digest = r.read_array::<32>()?;
        let commitment = r.read_array::<32>()?;
        r.ensure_eof()?;

        Ok(Self {
            profile_id,
            profile_version,
            op_graph_digest,
            input_digest,
            weights_digest,
            rounding_mode,
            overflow_mode,
            reduction_order,
            silu_lut_version,
            rmsnorm_eps_q24,
            output_byte_len,
            output_digest,
            commitment,
        })
    }
}

fn compute_receipt_commitment(ctx: &ReceiptCommitCtx<'_>) -> Digest {
    // Exhaustive destructure guarantees any new ExecutionProfile field is bound here.
    let ExecutionProfile {
        profile_id,
        profile_version,
        rounding_mode,
        overflow_mode,
        reduction_order,
        silu_lut_version,
        rmsnorm_eps_q24,
    } = ctx.profile;

    let mut h = Sha256::new();
    h.update(DOMAIN_RECEIPT);
    h.update(&(profile_id.len() as u32).to_le_bytes());
    h.update(profile_id.as_bytes());
    h.update(&profile_version.to_le_bytes());
    h.update(ctx.op_graph_digest);
    h.update(ctx.input_digest);
    h.update(ctx.weights_digest);
    h.update(&[
        rounding_mode.to_u8(),
        overflow_mode.to_u8(),
        reduction_order.to_u8(),
    ]);
    h.update(&silu_lut_version.to_le_bytes());
    h.update(&rmsnorm_eps_q24.to_le_bytes());
    h.update(&(ctx.output_tensor_bytes.len() as u64).to_le_bytes());
    h.update(ctx.output_tensor_bytes);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_nist_fips_180_4_vectors() {
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
        // 1,000,000 repetitions of 'a'
        let mut h = Sha256::new();
        let chunk = [b'a'; 1000];
        for _ in 0..1000 {
            h.update(&chunk);
        }
        assert_eq!(
            digest_to_hex(&h.finalize()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn receipt_wire_roundtrip_and_tamper_detection() {
        let profile = ExecutionProfile::v1_default();
        let g = commit_op_graph(b"graph");
        let i = commit_inputs(b"inputs");
        let w = commit_weights(b"weights");
        let out = b"canonical_output_bytes";

        let receipt = ExecutionReceipt::build(&profile, g, i, w, out).unwrap();
        receipt.verify_output(out).unwrap();

        let encoded = receipt.to_bytes();
        let decoded = ExecutionReceipt::from_bytes(&encoded).unwrap();
        assert_eq!(receipt, decoded);
        decoded.verify_output(out).unwrap();

        // Single-bit output tamper must fail verification.
        let mut tampered_out = out.to_vec();
        tampered_out[0] ^= 0x01;
        assert!(matches!(
            decoded.verify_output(&tampered_out),
            Err(DetError::CommitmentMismatch { .. })
        ));
    }

    #[test]
    fn receipt_binds_rmsnorm_eps_and_silu_lut_version() {
        let mut p1 = ExecutionProfile::v1_default();
        p1.rmsnorm_eps_q24 = 1;
        let mut p2 = ExecutionProfile::v1_default();
        p2.rmsnorm_eps_q24 = 1 << 24;

        let g = commit_op_graph(b"rmsnorm_graph");
        let i = commit_inputs(b"input_4096");
        let w = commit_weights(b"gamma_4096");
        let out = b"identical_output_on_4096";

        let r1 = ExecutionReceipt::build(&p1, g, i, w, out).unwrap();
        let r2 = ExecutionReceipt::build(&p2, g, i, w, out).unwrap();
        assert_ne!(r1.commitment, r2.commitment);
        assert_ne!(r1.to_bytes(), r2.to_bytes());
        assert_eq!(r1.profile(), p1);
        assert_eq!(r2.profile(), p2);
        assert_eq!(r1.to_bytes().len(), 210);

        let decoded = ExecutionReceipt::from_bytes(&r2.to_bytes()).unwrap();
        assert_eq!(decoded, r2);
        assert_eq!(decoded.profile(), p2);
        decoded.verify_output(out).unwrap();

        // Tampering with rmsnorm_eps_q24 or silu_lut_version must fail commitment verification.
        let mut eps_tampered = decoded.clone();
        eps_tampered.rmsnorm_eps_q24 = 1;
        assert!(matches!(
            eps_tampered.verify_output(out),
            Err(DetError::CommitmentMismatch {
                field: "commitment"
            })
        ));

        let mut lut_tampered = decoded;
        lut_tampered.silu_lut_version = 2;
        assert!(matches!(
            lut_tampered.verify_output(out),
            Err(DetError::CommitmentMismatch {
                field: "commitment"
            })
        ));
    }
}
