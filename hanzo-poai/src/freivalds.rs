// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Freivalds probabilistic verifier and canonical proof receipt over `GF(2^61 - 1)`.
//!
//! Verifies `A * (B * r) == C * r (mod p)` in `O(rounds * (m*k + k*n + m*n))` CPU work
//! versus `O(m*k*n)` prover matrix multiplication work, with per-round soundness error
//! strictly `< 1/p ~= 4.34e-19` (`< 1/p^2 < 2^-121` for 2 rounds).

use crate::error::PoaiError;
use crate::field::{ct_eq_digest, Fp};
use crate::profile::{
    OperatorKind, VerificationProfile, MAX_FREIVALDS_ROUNDS, MIN_FREIVALDS_ROUNDS,
};
use crate::quant::{ActivationMatrix, OutputMatrix, PackedWeightMatrix};
use crate::transcript::{
    commit_activation_matrix, commit_output_matrix, commit_round_projection, commit_weight_matrix,
    derive_challenge_vector, derive_transcript_seed, ChallengeMode,
};

const RECEIPT_MAGIC: [u8; 4] = *b"POAI";
const RECEIPT_WIRE_VERSION: u8 = 1;

/// Compact, canonical proof receipt binding a packed low-bit integer matrix product `C = A * B`.
///
/// Receipt size is `O(rounds)` (216 bytes in Fiat-Shamir mode and 248 bytes in Interactive mode
/// for `rounds = 2`), completely independent of matrix dimensions `(M, K, N)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixProofReceipt {
    pub profile_version: String,
    pub operator: OperatorKind,
    pub field_modulus_p: u64,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub rounds: usize,
    pub challenge_mode: ChallengeMode,
    pub commit_a: [u8; 32],
    pub commit_b: [u8; 32],
    pub commit_c: [u8; 32],
    pub round_digests: Vec<[u8; 32]>,
}

impl MatrixProofReceipt {
    /// Serialize the receipt to canonical little-endian wire bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let profile_bytes = self.profile_version.as_bytes();
        let mode_extra = match self.challenge_mode {
            ChallengeMode::FiatShamir => 0,
            ChallengeMode::InteractiveSeed(_) => 32,
        };
        let cap = 4
            + 1
            + 2
            + profile_bytes.len()
            + 1
            + 8
            + 24
            + 1
            + 1
            + mode_extra
            + 96
            + self.round_digests.len() * 32;
        let mut out = Vec::with_capacity(cap);
        out.extend_from_slice(&RECEIPT_MAGIC);
        out.push(RECEIPT_WIRE_VERSION);
        out.extend_from_slice(&(profile_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(profile_bytes);
        out.push(self.operator.wire_tag());
        out.extend_from_slice(&self.field_modulus_p.to_le_bytes());
        out.extend_from_slice(&(self.m as u64).to_le_bytes());
        out.extend_from_slice(&(self.k as u64).to_le_bytes());
        out.extend_from_slice(&(self.n as u64).to_le_bytes());
        out.push(self.rounds as u8);
        match self.challenge_mode {
            ChallengeMode::FiatShamir => out.push(0u8),
            ChallengeMode::InteractiveSeed(seed) => {
                out.push(1u8);
                out.extend_from_slice(&seed);
            }
        }
        out.extend_from_slice(&self.commit_a);
        out.extend_from_slice(&self.commit_b);
        out.extend_from_slice(&self.commit_c);
        for d in &self.round_digests {
            out.extend_from_slice(d);
        }
        out
    }

    /// Deserialize a canonical binary proof receipt, rejecting malformed or trailing bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PoaiError> {
        let mut cur = 0usize;
        let read_slice = |cur: &mut usize, len: usize| -> Result<&[u8], PoaiError> {
            let end = cur
                .checked_add(len)
                .ok_or(PoaiError::InvalidProofEncoding {
                    reason: "length overflow while parsing receipt",
                })?;
            if end > bytes.len() {
                return Err(PoaiError::InvalidProofEncoding {
                    reason: "unexpected EOF while parsing receipt",
                });
            }
            let s = &bytes[*cur..end];
            *cur = end;
            Ok(s)
        };

        let magic = read_slice(&mut cur, 4)?;
        if magic != RECEIPT_MAGIC {
            return Err(PoaiError::InvalidProofEncoding {
                reason: "invalid receipt magic header",
            });
        }
        let version = read_slice(&mut cur, 1)?[0];
        if version != RECEIPT_WIRE_VERSION {
            return Err(PoaiError::InvalidProofEncoding {
                reason: "unsupported receipt wire version",
            });
        }
        let plen_bytes = read_slice(&mut cur, 2)?;
        let plen = u16::from_le_bytes([plen_bytes[0], plen_bytes[1]]) as usize;
        if plen == 0 || plen > 128 {
            return Err(PoaiError::InvalidProofEncoding {
                reason: "invalid profile string length",
            });
        }
        let pbytes = read_slice(&mut cur, plen)?;
        let profile_version = std::str::from_utf8(pbytes)
            .map_err(|_| PoaiError::InvalidProofEncoding {
                reason: "profile version is not valid UTF-8",
            })?
            .to_string();

        let op_tag = read_slice(&mut cur, 1)?[0];
        let operator = OperatorKind::from_eligible_wire_tag(op_tag)?;

        let read_u64 = |cur: &mut usize| -> Result<u64, PoaiError> {
            let s = read_slice(cur, 8)?;
            Ok(u64::from_le_bytes([
                s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
            ]))
        };

        let field_modulus_p = read_u64(&mut cur)?;
        let m =
            usize::try_from(read_u64(&mut cur)?).map_err(|_| PoaiError::InvalidProofEncoding {
                reason: "dimension m overflows usize",
            })?;
        let k =
            usize::try_from(read_u64(&mut cur)?).map_err(|_| PoaiError::InvalidProofEncoding {
                reason: "dimension k overflows usize",
            })?;
        let n =
            usize::try_from(read_u64(&mut cur)?).map_err(|_| PoaiError::InvalidProofEncoding {
                reason: "dimension n overflows usize",
            })?;

        let rounds = read_slice(&mut cur, 1)?[0] as usize;
        if !(MIN_FREIVALDS_ROUNDS..=MAX_FREIVALDS_ROUNDS).contains(&rounds) {
            return Err(PoaiError::InvalidChallengeRounds {
                rounds,
                min: MIN_FREIVALDS_ROUNDS,
                max: MAX_FREIVALDS_ROUNDS,
            });
        }

        let mode_tag = read_slice(&mut cur, 1)?[0];
        let challenge_mode = match mode_tag {
            0 => ChallengeMode::FiatShamir,
            1 => {
                let s = read_slice(&mut cur, 32)?;
                let mut seed = [0u8; 32];
                seed.copy_from_slice(s);
                ChallengeMode::InteractiveSeed(seed)
            }
            _ => {
                return Err(PoaiError::InvalidProofEncoding {
                    reason: "invalid challenge mode tag",
                });
            }
        };

        let read_digest = |cur: &mut usize| -> Result<[u8; 32], PoaiError> {
            let s = read_slice(cur, 32)?;
            let mut d = [0u8; 32];
            d.copy_from_slice(s);
            Ok(d)
        };

        let commit_a = read_digest(&mut cur)?;
        let commit_b = read_digest(&mut cur)?;
        let commit_c = read_digest(&mut cur)?;

        let mut round_digests = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            round_digests.push(read_digest(&mut cur)?);
        }

        if cur != bytes.len() {
            return Err(PoaiError::InvalidProofEncoding {
                reason: "unexpected trailing bytes in receipt",
            });
        }

        Ok(Self {
            profile_version,
            operator,
            field_modulus_p,
            m,
            k,
            n,
            rounds,
            challenge_mode,
            commit_a,
            commit_b,
            commit_c,
            round_digests,
        })
    }
}

/// Verifier-owned security policy enforced during receipt verification.
///
/// Ensures the verifier—not the untrusted prover receipt—controls the minimum
/// number of independent Freivalds rounds and optional challenge mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationPolicy {
    /// Minimum number of independent Freivalds rounds required by the verifier.
    /// Defaults to `VerificationProfile::default_rounds` (`2`, > 121-bit soundness).
    pub min_rounds: usize,
    /// Optional verifier-chosen challenge mode (e.g. a fresh interactive CSPRNG seed).
    pub expected_mode: Option<ChallengeMode>,
}

impl VerificationPolicy {
    /// Default verifier policy for `profile`: enforces `profile.default_rounds` (`2` under
    /// `poai.matmul.v1`) and accepts either challenge mode.
    pub const fn default_for(profile: &VerificationProfile) -> Self {
        Self {
            min_rounds: profile.default_rounds,
            expected_mode: None,
        }
    }

    /// Construct a verifier policy with an explicit minimum round count (e.g. `1` when
    /// explicitly opting into 61-bit verification, or `4` for 244-bit security).
    pub const fn with_min_rounds(min_rounds: usize) -> Self {
        Self {
            min_rounds,
            expected_mode: None,
        }
    }

    /// Construct a verifier policy enforcing `profile.default_rounds` and a specific
    /// [`ChallengeMode`].
    pub const fn with_mode(profile: &VerificationProfile, expected_mode: ChallengeMode) -> Self {
        Self {
            min_rounds: profile.default_rounds,
            expected_mode: Some(expected_mode),
        }
    }

    /// Construct a fully explicit verifier policy.
    pub const fn new(min_rounds: usize, expected_mode: Option<ChallengeMode>) -> Self {
        Self {
            min_rounds,
            expected_mode,
        }
    }
}

/// Produce a canonical proof receipt for a claimed packed low-bit integer matrix product `C = A * B`.
pub fn prove_matrix_product(
    profile: &VerificationProfile,
    a: &ActivationMatrix,
    b: &PackedWeightMatrix,
    c: &OutputMatrix,
    rounds: usize,
    challenge_mode: ChallengeMode,
) -> Result<MatrixProofReceipt, PoaiError> {
    let commit_b = commit_weight_matrix(b);
    prove_matrix_product_with_cached_b_commit(profile, a, b, c, rounds, challenge_mode, commit_b)
}

/// Produce a proof receipt reusing a precomputed static weight commitment `commit_b`.
pub fn prove_matrix_product_with_cached_b_commit(
    profile: &VerificationProfile,
    a: &ActivationMatrix,
    b: &PackedWeightMatrix,
    c: &OutputMatrix,
    rounds: usize,
    challenge_mode: ChallengeMode,
    commit_b: [u8; 32],
) -> Result<MatrixProofReceipt, PoaiError> {
    let operator = b.format().operator_kind();
    profile.ensure_eligible(operator)?;
    profile.validate_rounds(rounds)?;
    validate_matrix_shapes(a, b, c)?;

    let m = a.rows();
    let k = a.cols();
    let n = b.cols();

    let commit_a = commit_activation_matrix(a);
    let commit_c = commit_output_matrix(c);

    let transcript_seed = derive_transcript_seed(
        profile,
        operator,
        m,
        k,
        n,
        rounds,
        &commit_a,
        &commit_b,
        &commit_c,
        challenge_mode,
    );

    let mut cr = vec![Fp::ZERO; m];
    let mut round_digests = Vec::with_capacity(rounds);
    for round_idx in 0..rounds {
        let r = derive_challenge_vector(&transcript_seed, round_idx as u32, n);
        c.mul_vec_fp(&r, &mut cr)?;
        round_digests.push(commit_round_projection(round_idx as u32, &cr));
    }

    Ok(MatrixProofReceipt {
        profile_version: profile.version.to_string(),
        operator,
        field_modulus_p: profile.field_modulus_p,
        m,
        k,
        n,
        rounds,
        challenge_mode,
        commit_a,
        commit_b,
        commit_c,
        round_digests,
    })
}

/// Verify a claimed matrix product `C = A * B` against its proof receipt (cold path: recomputes
/// all three commitments `commit_A`, `commit_B`, `commit_C`) while enforcing the profile's default
/// minimum round count (`profile.default_rounds`, i.e., `2` rounds under `poai.matmul.v1`).
pub fn verify_matrix_product(
    profile: &VerificationProfile,
    a: &ActivationMatrix,
    b: &PackedWeightMatrix,
    c: &OutputMatrix,
    receipt: &MatrixProofReceipt,
) -> Result<(), PoaiError> {
    verify_matrix_product_with_policy(
        profile,
        a,
        b,
        c,
        receipt,
        VerificationPolicy::default_for(profile),
    )
}

/// Verify a claimed matrix product while enforcing `profile.default_rounds` and an explicit
/// verifier-chosen [`ChallengeMode`] (e.g. requiring a fresh interactive CSPRNG seed so a
/// cheating prover cannot grind Fiat-Shamir).
pub fn verify_matrix_product_with_mode(
    profile: &VerificationProfile,
    a: &ActivationMatrix,
    b: &PackedWeightMatrix,
    c: &OutputMatrix,
    receipt: &MatrixProofReceipt,
    expected_mode: ChallengeMode,
) -> Result<(), PoaiError> {
    verify_matrix_product_with_policy(
        profile,
        a,
        b,
        c,
        receipt,
        VerificationPolicy::with_mode(profile, expected_mode),
    )
}

/// Verify a claimed matrix product (cold path) under an explicit verifier-owned
/// [`VerificationPolicy`] (controlling `min_rounds` and optional `expected_mode`).
pub fn verify_matrix_product_with_policy(
    profile: &VerificationProfile,
    a: &ActivationMatrix,
    b: &PackedWeightMatrix,
    c: &OutputMatrix,
    receipt: &MatrixProofReceipt,
    policy: VerificationPolicy,
) -> Result<(), PoaiError> {
    let commit_b = commit_weight_matrix(b);
    verify_matrix_product_inner(profile, a, b, c, receipt, commit_b, policy)
}

/// Verify a claimed matrix product using a pre-verified static model weight commitment
/// `trusted_commit_b` (hot inference serving path: hashes only dynamic `A` and `C`, still checks
/// `trusted_commit_b == receipt.commit_b`, enforces `profile.default_rounds` minimum rounds, and
/// runs full Freivalds `A * (B * r) == C * r (mod p)` against the resident packed weights `B`).
pub fn verify_matrix_product_with_cached_b_commit(
    profile: &VerificationProfile,
    a: &ActivationMatrix,
    b: &PackedWeightMatrix,
    c: &OutputMatrix,
    receipt: &MatrixProofReceipt,
    trusted_commit_b: [u8; 32],
    expected_mode: Option<ChallengeMode>,
) -> Result<(), PoaiError> {
    verify_matrix_product_inner(
        profile,
        a,
        b,
        c,
        receipt,
        trusted_commit_b,
        VerificationPolicy::new(profile.default_rounds, expected_mode),
    )
}

/// Verify a claimed matrix product using a pre-verified static model weight commitment
/// `trusted_commit_b` under an explicit verifier-owned [`VerificationPolicy`].
pub fn verify_matrix_product_with_cached_b_commit_and_policy(
    profile: &VerificationProfile,
    a: &ActivationMatrix,
    b: &PackedWeightMatrix,
    c: &OutputMatrix,
    receipt: &MatrixProofReceipt,
    trusted_commit_b: [u8; 32],
    policy: VerificationPolicy,
) -> Result<(), PoaiError> {
    verify_matrix_product_inner(profile, a, b, c, receipt, trusted_commit_b, policy)
}

fn verify_matrix_product_inner(
    profile: &VerificationProfile,
    a: &ActivationMatrix,
    b: &PackedWeightMatrix,
    c: &OutputMatrix,
    receipt: &MatrixProofReceipt,
    expected_commit_b: [u8; 32],
    policy: VerificationPolicy,
) -> Result<(), PoaiError> {
    if receipt.profile_version != profile.version {
        return Err(PoaiError::ProfileMismatch {
            expected: profile.version.to_string(),
            actual: receipt.profile_version.clone(),
        });
    }
    if receipt.field_modulus_p != profile.field_modulus_p {
        return Err(PoaiError::FieldModulusMismatch {
            expected: profile.field_modulus_p,
            actual: receipt.field_modulus_p,
        });
    }
    profile.ensure_eligible(receipt.operator)?;
    profile.validate_rounds_at_least(receipt.rounds, policy.min_rounds)?;

    let expected_op = b.format().operator_kind();
    if receipt.operator != expected_op {
        return Err(PoaiError::ProfileMismatch {
            expected: expected_op.as_str().to_string(),
            actual: receipt.operator.as_str().to_string(),
        });
    }
    if let Some(mode) = policy.expected_mode {
        if receipt.challenge_mode != mode {
            return Err(PoaiError::ChallengeModeMismatch {
                reason: "receipt challenge mode or interactive seed does not match verifier policy",
            });
        }
    }

    validate_matrix_shapes(a, b, c)?;
    let m = a.rows();
    let k = a.cols();
    let n = b.cols();
    if receipt.m != m || receipt.k != k || receipt.n != n {
        return Err(PoaiError::InvalidDimensions {
            m: receipt.m,
            k: receipt.k,
            n: receipt.n,
            reason: "receipt dimensions do not match supplied matrices",
        });
    }
    if receipt.round_digests.len() != receipt.rounds {
        return Err(PoaiError::InvalidProofEncoding {
            reason: "round_digests count does not match receipt.rounds",
        });
    }

    // Verify cryptographic commitments in constant time.
    let commit_a = commit_activation_matrix(a);
    if !ct_eq_digest(&commit_a, &receipt.commit_a) {
        return Err(PoaiError::CommitmentMismatch { matrix: "A" });
    }
    if !ct_eq_digest(&expected_commit_b, &receipt.commit_b) {
        return Err(PoaiError::CommitmentMismatch { matrix: "B" });
    }
    let commit_c = commit_output_matrix(c);
    if !ct_eq_digest(&commit_c, &receipt.commit_c) {
        return Err(PoaiError::CommitmentMismatch { matrix: "C" });
    }

    let transcript_seed = derive_transcript_seed(
        profile,
        receipt.operator,
        m,
        k,
        n,
        receipt.rounds,
        &receipt.commit_a,
        &receipt.commit_b,
        &receipt.commit_c,
        receipt.challenge_mode,
    );

    let mut br = vec![Fp::ZERO; k];
    let mut abr = vec![Fp::ZERO; m];
    let mut cr = vec![Fp::ZERO; m];

    for (round_idx, receipt_digest) in receipt.round_digests.iter().enumerate() {
        let r = derive_challenge_vector(&transcript_seed, round_idx as u32, n);

        // Compute C * r (mod p) while enforcing strict |C[i,j]| <= (p - 1)/2 range bound.
        c.mul_vec_fp(&r, &mut cr)?;

        // Confirm receipt round digest matches computed C * r projection.
        let expected_proj_digest = commit_round_projection(round_idx as u32, &cr);
        if !ct_eq_digest(&expected_proj_digest, receipt_digest) {
            return Err(PoaiError::RoundDigestMismatch { round: round_idx });
        }

        // Compute A * (B * r) (mod p) without materializing dense B.
        b.mul_vec_fp(&r, &mut br)?;
        a.mul_vec_fp(&br, &mut abr)?;

        for row in 0..m {
            if abr[row] != cr[row] {
                return Err(PoaiError::FreivaldsCheckFailed {
                    round: round_idx,
                    first_mismatch_row: row,
                });
            }
        }
    }

    Ok(())
}

/// Core algebraic Freivalds check step (`A * (B * r) == C * r (mod p)`) separated for
/// micro-benchmarking algebraic verifier cost independently from SHA-256 commitments
/// and challenge generation.
pub fn verify_freivalds_algebraic_only(
    a: &ActivationMatrix,
    b: &PackedWeightMatrix,
    c: &OutputMatrix,
    challenges: &[Vec<Fp>],
) -> Result<(), PoaiError> {
    validate_matrix_shapes(a, b, c)?;
    let m = a.rows();
    let k = a.cols();
    let mut br = vec![Fp::ZERO; k];
    let mut abr = vec![Fp::ZERO; m];
    let mut cr = vec![Fp::ZERO; m];

    for (round_idx, r) in challenges.iter().enumerate() {
        c.mul_vec_fp(r, &mut cr)?;
        b.mul_vec_fp(r, &mut br)?;
        a.mul_vec_fp(&br, &mut abr)?;
        for row in 0..m {
            if abr[row] != cr[row] {
                return Err(PoaiError::FreivaldsCheckFailed {
                    round: round_idx,
                    first_mismatch_row: row,
                });
            }
        }
    }
    Ok(())
}

fn validate_matrix_shapes(
    a: &ActivationMatrix,
    b: &PackedWeightMatrix,
    c: &OutputMatrix,
) -> Result<(), PoaiError> {
    if a.cols() != b.rows() {
        return Err(PoaiError::DimensionMismatch {
            matrix: "A/B",
            expected: (a.rows(), b.rows()),
            actual: (a.rows(), a.cols()),
        });
    }
    if c.rows() != a.rows() || c.cols() != b.cols() {
        return Err(PoaiError::DimensionMismatch {
            matrix: "C",
            expected: (a.rows(), b.cols()),
            actual: (c.rows(), c.cols()),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{MAX_SAFE_SIGNED, MODULUS_P};
    use crate::quant::{prover_matmul, PackedQuantFormat};
    use crate::transcript::os_random_seed;

    fn sample_instance(
        format: PackedQuantFormat,
        m: usize,
        k: usize,
        n: usize,
    ) -> (ActivationMatrix, PackedWeightMatrix, OutputMatrix) {
        let a_data: Vec<i8> = (0..m * k)
            .map(|i| (((i * 31 + 7) % 251) as i32 - 125) as i8)
            .collect();
        let a = ActivationMatrix::new(m, k, a_data).unwrap();
        let blocks = k / 32;
        let scales: Vec<i16> = (0..n * blocks)
            .map(|i| if i % 3 == 0 { -19 } else { (i % 29 + 1) as i16 })
            .collect();
        let q: Vec<i8> = match format {
            PackedQuantFormat::Int8Q8_0 => (0..k * n)
                .map(|i| (((i * 19 + 11) % 241) as i32 - 120) as i8)
                .collect(),
            PackedQuantFormat::PackedInt4Q4_0 => (0..k * n)
                .map(|i| (((i * 13 + 5) % 16) as i32 - 8) as i8)
                .collect(),
            PackedQuantFormat::PackedTernaryI2S => {
                (0..k * n).map(|i| ((i * 5 + 1) % 3) as i8 - 1).collect()
            }
        };
        let b = PackedWeightMatrix::pack_from_column_major(format, k, n, &q, &scales).unwrap();
        let c = prover_matmul(&a, &b).unwrap();
        (a, b, c)
    }

    #[test]
    fn end_to_end_fiat_shamir_and_interactive_roundtrip() {
        let profile = VerificationProfile::v1();
        for format in [
            PackedQuantFormat::Int8Q8_0,
            PackedQuantFormat::PackedInt4Q4_0,
            PackedQuantFormat::PackedTernaryI2S,
        ] {
            let (a, b, c) = sample_instance(format, 8, 64, 32);

            // Fiat-Shamir mode
            let receipt =
                prove_matrix_product(&profile, &a, &b, &c, 2, ChallengeMode::FiatShamir).unwrap();
            let wire = receipt.to_bytes();
            assert_eq!(wire.len(), 216);
            let decoded = MatrixProofReceipt::from_bytes(&wire).unwrap();
            assert_eq!(decoded, receipt);
            verify_matrix_product(&profile, &a, &b, &c, &decoded).unwrap();

            // Interactive CSPRNG seed mode
            let seed = os_random_seed().unwrap();
            let mode = ChallengeMode::InteractiveSeed(seed);
            let ireceipt = prove_matrix_product(&profile, &a, &b, &c, 2, mode).unwrap();
            let iwire = ireceipt.to_bytes();
            assert_eq!(iwire.len(), 248);
            let idecoded = MatrixProofReceipt::from_bytes(&iwire).unwrap();
            verify_matrix_product_with_mode(&profile, &a, &b, &c, &idecoded, mode).unwrap();

            // Downgrade from interactive to Fiat-Shamir must fail
            assert!(matches!(
                verify_matrix_product_with_mode(&profile, &a, &b, &c, &decoded, mode),
                Err(PoaiError::ChallengeModeMismatch { .. })
            ));
        }
    }

    #[test]
    fn detects_all_corruption_and_forgery_modes() {
        let profile = VerificationProfile::v1();
        let (a, b, c) = sample_instance(PackedQuantFormat::PackedInt4Q4_0, 6, 64, 32);
        let good_receipt =
            prove_matrix_product(&profile, &a, &b, &c, 2, ChallengeMode::FiatShamir).unwrap();

        // 1. +1 corruption with freshly re-proved receipt -> FreivaldsCheckFailed
        let mut c_plus1 = c.clone();
        c_plus1.data_mut()[17] += 1;
        let bad_receipt =
            prove_matrix_product(&profile, &a, &b, &c_plus1, 2, ChallengeMode::FiatShamir).unwrap();
        assert!(matches!(
            verify_matrix_product(&profile, &a, &b, &c_plus1, &bad_receipt),
            Err(PoaiError::FreivaldsCheckFailed { .. })
        ));

        // 2. +1 corruption against original receipt -> CommitmentMismatch
        assert!(matches!(
            verify_matrix_product(&profile, &a, &b, &c_plus1, &good_receipt),
            Err(PoaiError::CommitmentMismatch { matrix: "C" })
        ));

        // 3. Wraparound attack: adding +p to an element of C must be rejected by strict range bound
        let mut c_wrap = c.clone();
        c_wrap.data_mut()[0] = MAX_SAFE_SIGNED + 1;
        assert!(matches!(
            prove_matrix_product(&profile, &a, &b, &c_wrap, 2, ChallengeMode::FiatShamir),
            Err(PoaiError::OutputOutOfFieldRange { .. })
        ));
        // Even if a malicious receipt already existed with commit_c for c_wrap, verifier rejects it
        let mut forged = good_receipt.clone();
        forged.commit_c = commit_output_matrix(&c_wrap);
        assert!(matches!(
            verify_matrix_product(&profile, &a, &b, &c_wrap, &forged),
            Err(PoaiError::OutputOutOfFieldRange { .. })
        ));

        // 4. Quantized scale tamper: compute C with tampered scale, try to pass against real B
        let b_tampered = b.with_tampered_block_scale(3, 1, 999).unwrap();
        let c_from_tampered = prover_matmul(&a, &b_tampered).unwrap();
        let scale_receipt = prove_matrix_product(
            &profile,
            &a,
            &b,
            &c_from_tampered,
            2,
            ChallengeMode::FiatShamir,
        )
        .unwrap();
        assert!(matches!(
            verify_matrix_product(&profile, &a, &b, &c_from_tampered, &scale_receipt),
            Err(PoaiError::FreivaldsCheckFailed { .. })
        ));

        // 5. Tampered round digest in serialized receipt
        let mut bad_digest_receipt = good_receipt.clone();
        bad_digest_receipt.round_digests[0][0] ^= 0x01;
        assert!(matches!(
            verify_matrix_product(&profile, &a, &b, &c, &bad_digest_receipt),
            Err(PoaiError::RoundDigestMismatch { round: 0 })
        ));

        // 6. Wrong modulus in receipt
        let mut bad_mod = good_receipt.clone();
        bad_mod.field_modulus_p = MODULUS_P - 2;
        assert!(matches!(
            verify_matrix_product(&profile, &a, &b, &c, &bad_mod),
            Err(PoaiError::FieldModulusMismatch { .. })
        ));

        // 7. Trailing byte or truncated wire encoding rejected
        let mut wire = good_receipt.to_bytes();
        wire.push(0xFF);
        assert!(matches!(
            MatrixProofReceipt::from_bytes(&wire),
            Err(PoaiError::InvalidProofEncoding { .. })
        ));
        wire.truncate(10);
        assert!(matches!(
            MatrixProofReceipt::from_bytes(&wire),
            Err(PoaiError::InvalidProofEncoding { .. })
        ));
    }

    #[test]
    fn rejects_prover_round_downgrade_and_honors_verifier_owned_policy() {
        let profile = VerificationProfile::v1();
        let (a, b, c) = sample_instance(PackedQuantFormat::PackedInt4Q4_0, 8, 64, 32);
        let commit_b = commit_weight_matrix(&b);
        let interactive_mode = ChallengeMode::InteractiveSeed(os_random_seed().unwrap());

        for mode in [ChallengeMode::FiatShamir, interactive_mode] {
            // Reproduce the exact downgrade attack: take a valid 2-round receipt, mutate rounds
            // to 1, truncate round_digests, and round-trip through the wire codec.
            let two_round = prove_matrix_product(&profile, &a, &b, &c, 2, mode).unwrap();
            let mut downgraded = two_round.clone();
            downgraded.rounds = 1;
            downgraded.round_digests.truncate(1);
            let downgraded_wire = MatrixProofReceipt::from_bytes(&downgraded.to_bytes()).unwrap();

            // Cold, mode-enforced, and cached-B default verifiers MUST all reject with min=2.
            assert!(matches!(
                verify_matrix_product(&profile, &a, &b, &c, &downgraded_wire),
                Err(PoaiError::InvalidChallengeRounds {
                    rounds: 1,
                    min: 2,
                    max: 4
                })
            ));
            assert!(matches!(
                verify_matrix_product_with_mode(&profile, &a, &b, &c, &downgraded_wire, mode),
                Err(PoaiError::InvalidChallengeRounds {
                    rounds: 1,
                    min: 2,
                    max: 4
                })
            ));
            assert!(matches!(
                verify_matrix_product_with_cached_b_commit(
                    &profile,
                    &a,
                    &b,
                    &c,
                    &downgraded_wire,
                    commit_b,
                    Some(mode),
                ),
                Err(PoaiError::InvalidChallengeRounds {
                    rounds: 1,
                    min: 2,
                    max: 4
                })
            ));

            // Even if a verifier explicitly opts into min_rounds = 1, truncating a 2-round receipt
            // MUST fail because `rounds` is cryptographically bound into the transcript seed.
            let explicit_1 = VerificationPolicy::new(1, Some(mode));
            assert!(matches!(
                verify_matrix_product_with_policy(
                    &profile,
                    &a,
                    &b,
                    &c,
                    &downgraded_wire,
                    explicit_1
                ),
                Err(PoaiError::RoundDigestMismatch { round: 0 })
            ));
            assert!(matches!(
                verify_matrix_product_with_cached_b_commit_and_policy(
                    &profile,
                    &a,
                    &b,
                    &c,
                    &downgraded_wire,
                    commit_b,
                    explicit_1,
                ),
                Err(PoaiError::RoundDigestMismatch { round: 0 })
            ));

            // A freshly proved honest 1-round receipt is still rejected by default verifiers...
            let honest_one_round = prove_matrix_product(&profile, &a, &b, &c, 1, mode).unwrap();
            let honest_one_wire =
                MatrixProofReceipt::from_bytes(&honest_one_round.to_bytes()).unwrap();
            assert!(matches!(
                verify_matrix_product(&profile, &a, &b, &c, &honest_one_wire),
                Err(PoaiError::InvalidChallengeRounds {
                    rounds: 1,
                    min: 2,
                    max: 4
                })
            ));
            assert!(matches!(
                verify_matrix_product_with_mode(&profile, &a, &b, &c, &honest_one_wire, mode),
                Err(PoaiError::InvalidChallengeRounds {
                    rounds: 1,
                    min: 2,
                    max: 4
                })
            ));
            assert!(matches!(
                verify_matrix_product_with_cached_b_commit(
                    &profile,
                    &a,
                    &b,
                    &c,
                    &honest_one_wire,
                    commit_b,
                    Some(mode),
                ),
                Err(PoaiError::InvalidChallengeRounds {
                    rounds: 1,
                    min: 2,
                    max: 4
                })
            ));

            // ...and accepted ONLY when the verifier explicitly chooses min_rounds = 1.
            verify_matrix_product_with_policy(&profile, &a, &b, &c, &honest_one_wire, explicit_1)
                .unwrap();
            verify_matrix_product_with_cached_b_commit_and_policy(
                &profile,
                &a,
                &b,
                &c,
                &honest_one_wire,
                commit_b,
                explicit_1,
            )
            .unwrap();

            // Higher verifier-owned minimum (4 rounds) rejects 2-round receipts and accepts 4-round.
            let require_4 = VerificationPolicy::new(4, Some(mode));
            assert!(matches!(
                verify_matrix_product_with_policy(&profile, &a, &b, &c, &two_round, require_4),
                Err(PoaiError::InvalidChallengeRounds {
                    rounds: 2,
                    min: 4,
                    max: 4
                })
            ));
            let four_round = prove_matrix_product(&profile, &a, &b, &c, 4, mode).unwrap();
            verify_matrix_product_with_policy(&profile, &a, &b, &c, &four_round, require_4)
                .unwrap();

            // Out-of-range policy min_rounds (0 or 5) is itself rejected.
            assert!(matches!(
                verify_matrix_product_with_policy(
                    &profile,
                    &a,
                    &b,
                    &c,
                    &two_round,
                    VerificationPolicy::with_min_rounds(0),
                ),
                Err(PoaiError::InvalidChallengeRounds {
                    rounds: 0,
                    min: 1,
                    max: 4
                })
            ));
            assert!(matches!(
                verify_matrix_product_with_policy(
                    &profile,
                    &a,
                    &b,
                    &c,
                    &two_round,
                    VerificationPolicy::with_min_rounds(5),
                ),
                Err(PoaiError::InvalidChallengeRounds {
                    rounds: 5,
                    min: 1,
                    max: 4
                })
            ));
        }
    }
}
