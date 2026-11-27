// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Domain-separated SHA-256 Merkle commitment trees and Ligero-style Reed-Solomon
//! Merkle Polynomial Commitment Scheme (PCS) over Goldilocks `F_p` / `F_{p^2}`.

use crate::field::{eq_eval_table, rs_encode_fp, rs_encode_fp2, Fp, Fp2};
use crate::sha256::{Sha256, Transcript};

/// Domain separator prefix byte for real Merkle leaves.
const LEAF_PREFIX: u8 = 0x00;
/// Domain separator prefix byte for internal Merkle nodes.
const NODE_PREFIX: u8 = 0x01;
/// Domain separator prefix byte for padded empty leaves.
const EMPTY_LEAF_PREFIX: u8 = 0x02;

/// Hashes a Merkle leaf with explicit domain separation and leaf index binding.
pub fn hash_leaf(index: usize, payload: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(&[LEAF_PREFIX]);
    h.update(&(index as u64).to_le_bytes());
    h.update(&(payload.len() as u64).to_le_bytes());
    h.update(payload);
    h.finalize()
}

/// Hashes an empty padded leaf at `index`.
fn hash_empty_leaf(index: usize) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(&[EMPTY_LEAF_PREFIX]);
    h.update(&(index as u64).to_le_bytes());
    h.finalize()
}

/// Hashes two child digests into a parent digest.
pub fn hash_internal(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(&[NODE_PREFIX]);
    h.update(left);
    h.update(right);
    h.finalize()
}

/// Inclusion proof for a single leaf in a [`MerkleTree`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleProof {
    /// Zero-based index of the opened leaf.
    pub leaf_index: usize,
    /// Total number of real leaves committed in the tree.
    pub leaf_count: usize,
    /// Sibling digests ordered from bottom (leaf level) to top (below root).
    pub siblings: Vec<[u8; 32]>,
}

impl MerkleProof {
    /// Appends canonical wire encoding of this proof to `out`.
    pub fn write_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.leaf_index as u64).to_le_bytes());
        out.extend_from_slice(&(self.leaf_count as u64).to_le_bytes());
        out.extend_from_slice(&(self.siblings.len() as u32).to_le_bytes());
        for s in &self.siblings {
            out.extend_from_slice(s);
        }
    }

    /// Exact serialized byte length.
    pub fn byte_len(&self) -> usize {
        8 + 8 + 4 + self.siblings.len() * 32
    }
}

/// Complete binary SHA-256 Merkle commitment tree.
#[derive(Clone, Debug)]
pub struct MerkleTree {
    leaf_count: usize,
    padded_leaves: usize,
    levels: Vec<Vec<[u8; 32]>>,
}

impl MerkleTree {
    /// Builds a Merkle tree from a non-empty slice of leaf payloads.
    pub fn from_leaves(leaves: &[Vec<u8>]) -> Self {
        assert!(!leaves.is_empty(), "MerkleTree requires at least one leaf");
        let leaf_count = leaves.len();
        let padded_leaves = leaf_count.next_power_of_two();

        let mut base = Vec::with_capacity(padded_leaves);
        for (i, payload) in leaves.iter().enumerate() {
            base.push(hash_leaf(i, payload));
        }
        for i in leaf_count..padded_leaves {
            base.push(hash_empty_leaf(i));
        }

        let mut levels = Vec::new();
        levels.push(base);
        while levels.last().expect("non-empty").len() > 1 {
            let prev = levels.last().expect("non-empty");
            let mut next = Vec::with_capacity(prev.len() / 2);
            for pair in prev.as_chunks::<2>().0 {
                next.push(hash_internal(&pair[0], &pair[1]));
            }
            levels.push(next);
        }

        Self {
            leaf_count,
            padded_leaves,
            levels,
        }
    }

    /// Returns the 32-byte Merkle root digest.
    pub fn root(&self) -> [u8; 32] {
        self.levels.last().expect("non-empty")[0]
    }

    /// Returns the number of real leaves in the tree.
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Generates an inclusion proof for `leaf_index`.
    pub fn prove(&self, leaf_index: usize) -> MerkleProof {
        assert!(leaf_index < self.leaf_count, "leaf index out of range");
        let mut siblings = Vec::with_capacity(self.levels.len() - 1);
        let mut idx = leaf_index;
        for level in &self.levels[..self.levels.len() - 1] {
            let sibling_idx = idx ^ 1;
            siblings.push(level[sibling_idx]);
            idx >>= 1;
        }
        debug_assert_eq!(1usize << siblings.len(), self.padded_leaves);
        MerkleProof {
            leaf_index,
            leaf_count: self.leaf_count,
            siblings,
        }
    }
}

/// Statelessly verifies a [`MerkleProof`] against an expected 32-byte `root`.
pub fn verify_merkle_proof(root: &[u8; 32], payload: &[u8], proof: &MerkleProof) -> bool {
    if proof.leaf_count == 0 || proof.leaf_index >= proof.leaf_count {
        return false;
    }
    let padded = proof.leaf_count.next_power_of_two();
    let expected_depth = padded.trailing_zeros() as usize;
    if proof.siblings.len() != expected_depth {
        return false;
    }

    let mut hash = hash_leaf(proof.leaf_index, payload);
    let mut idx = proof.leaf_index;
    for sibling in &proof.siblings {
        hash = if idx & 1 == 0 {
            hash_internal(&hash, sibling)
        } else {
            hash_internal(sibling, &hash)
        };
        idx >>= 1;
    }
    &hash == root
}

/// Parameters for the Ligero-style Reed-Solomon Merkle Polynomial Commitment Scheme.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LigeroParams {
    /// Reed-Solomon blowup factor `B = 1 / rate` (power of two, `>= 4`).
    pub blowup: usize,
    /// Number of distinct codeword columns opened per proof.
    pub num_queries: usize,
}

impl Default for LigeroParams {
    fn default() -> Self {
        Self {
            blowup: 8,
            num_queries: 64,
        }
    }
}

impl LigeroParams {
    /// Returns the proven Ligero proximity soundness exponent (in bits) at proximity
    /// radius `delta = (1 - 1/B) / 2 - 1/16`, conservatively bounded by
    /// `-num_queries * log2(1 - delta)`.
    pub fn proximity_soundness_bits(&self) -> f64 {
        let rate = 1.0 / (self.blowup as f64);
        let rel_dist = 1.0 - rate;
        let unique_radius = rel_dist * 0.5;
        let delta = (unique_radius - 0.0625).max(0.125);
        -(self.num_queries as f64) * (1.0 - delta).log2()
    }
}

/// Public commitment to a multilinear vector over `F_p` using Ligero RS-Merkle.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LigeroCommitment {
    /// SHA-256 Merkle root over the `L = C * blowup` encoded codeword columns.
    pub root: [u8; 32],
    /// Number of rows `R` (power of two).
    pub num_rows: usize,
    /// Number of message columns `C` (power of two).
    pub num_cols: usize,
    /// RS blowup factor `B`.
    pub blowup: usize,
}

impl LigeroCommitment {
    /// Total vector length `N = R * C`.
    pub const fn vector_len(&self) -> usize {
        self.num_rows * self.num_cols
    }

    /// Number of Boolean variables `log2(N)`.
    pub const fn num_vars(&self) -> usize {
        self.vector_len().trailing_zeros() as usize
    }

    /// Encoded codeword length `L = C * blowup`.
    pub const fn codeword_len(&self) -> usize {
        self.num_cols * self.blowup
    }

    /// Appends canonical wire bytes to `out`.
    pub fn write_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.root);
        out.extend_from_slice(&(self.num_rows as u64).to_le_bytes());
        out.extend_from_slice(&(self.num_cols as u64).to_le_bytes());
        out.extend_from_slice(&(self.blowup as u64).to_le_bytes());
    }
}

/// Prover state retaining the reshaped message matrix, RS-encoded matrix, and Merkle tree.
#[derive(Clone, Debug)]
pub struct LigeroProverState {
    /// Public commitment header.
    pub commitment: LigeroCommitment,
    /// Original message rows (`R` rows of `C` elements).
    rows: Vec<Vec<Fp>>,
    /// Reed-Solomon encoded rows (`R` rows of `L = C * blowup` elements).
    encoded_rows: Vec<Vec<Fp>>,
    /// Merkle tree committing to the `L` encoded columns.
    tree: MerkleTree,
}

/// Serializes a single encoded column (`R` base-field elements) to canonical bytes.
fn encode_column_bytes(col: &[Fp]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(col.len() * 8);
    for &v in col {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Chooses near-square power-of-two matrix dimensions `(R, C)` for a vector of length `N = 2^k`.
pub fn matrix_dims_for_pow2(len: usize) -> (usize, usize) {
    assert!(len.is_power_of_two() && len >= 2);
    let k = len.trailing_zeros() as usize;
    let log_r = k / 2;
    let log_c = k - log_r;
    (1usize << log_r, 1usize << log_c)
}

/// Commits a base-field vector `values` (length padded to power of two `>= 2`) under Ligero RS-Merkle.
pub fn commit_vector_ligero(values: &[Fp], params: LigeroParams) -> LigeroProverState {
    assert!(params.blowup.is_power_of_two() && params.blowup >= 2);
    let padded_len = values.len().max(2).next_power_of_two();
    let (num_rows, num_cols) = matrix_dims_for_pow2(padded_len);

    let mut rows = Vec::with_capacity(num_rows);
    let mut encoded_rows = Vec::with_capacity(num_rows);
    for r in 0..num_rows {
        let mut row = vec![Fp::ZERO; num_cols];
        let start = r * num_cols;
        for (c, slot) in row.iter_mut().enumerate() {
            let idx = start + c;
            if idx < values.len() {
                *slot = values[idx];
            }
        }
        let encoded = rs_encode_fp(&row, params.blowup);
        rows.push(row);
        encoded_rows.push(encoded);
    }

    let codeword_len = num_cols * params.blowup;
    let mut column_leaves = Vec::with_capacity(codeword_len);
    for col_idx in 0..codeword_len {
        let mut col = Vec::with_capacity(num_rows);
        for row in &encoded_rows {
            col.push(row[col_idx]);
        }
        column_leaves.push(encode_column_bytes(&col));
    }

    let tree = MerkleTree::from_leaves(&column_leaves);
    let commitment = LigeroCommitment {
        root: tree.root(),
        num_rows,
        num_cols,
        blowup: params.blowup,
    };

    LigeroProverState {
        commitment,
        rows,
        encoded_rows,
        tree,
    }
}

/// Opened encoded column plus its Merkle authentication path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LigeroColumnOpening {
    /// Column index in `0..codeword_len`.
    pub col_index: usize,
    /// Column entries `U_hat[0..R, col_index]`.
    pub values: Vec<Fp>,
    /// Merkle authentication path against `LigeroCommitment::root`.
    pub merkle_proof: MerkleProof,
}

impl LigeroColumnOpening {
    /// Appends canonical wire bytes to `out`.
    pub fn write_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.col_index as u64).to_le_bytes());
        out.extend_from_slice(&(self.values.len() as u32).to_le_bytes());
        for &v in &self.values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        self.merkle_proof.write_bytes(out);
    }

    /// Exact serialized byte length.
    pub fn byte_len(&self) -> usize {
        8 + 4 + self.values.len() * 8 + self.merkle_proof.byte_len()
    }
}

/// Complete Ligero RS-Merkle multilinear evaluation + proximity opening proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LigeroOpeningProof {
    /// Proximity random-linear-combination message vector `w_prox in F_{p^2}^C`.
    pub w_prox: Vec<Fp2>,
    /// Evaluation linear-combination message vector `w_eval in F_{p^2}^C`.
    pub w_eval: Vec<Fp2>,
    /// Opened and Merkle-authenticated columns at Fiat-Shamir query indices.
    pub opened_columns: Vec<LigeroColumnOpening>,
}

impl LigeroOpeningProof {
    /// Appends canonical wire bytes to `out`.
    pub fn write_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.w_prox.len() as u32).to_le_bytes());
        for &v in &self.w_prox {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(self.w_eval.len() as u32).to_le_bytes());
        for &v in &self.w_eval {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(self.opened_columns.len() as u32).to_le_bytes());
        for col in &self.opened_columns {
            col.write_bytes(out);
        }
    }

    /// Exact serialized byte length.
    pub fn byte_len(&self) -> usize {
        4 + self.w_prox.len() * 16
            + 4
            + self.w_eval.len() * 16
            + 4
            + self
                .opened_columns
                .iter()
                .map(LigeroColumnOpening::byte_len)
                .sum::<usize>()
    }
}

/// Produces a Ligero opening proof for `V_tilde(point) = claimed_eval`, advancing `transcript`.
pub fn open_mle_ligero(
    state: &LigeroProverState,
    point: &[Fp2],
    params: LigeroParams,
    transcript: &mut Transcript,
) -> (Fp2, LigeroOpeningProof) {
    let num_rows = state.commitment.num_rows;
    let num_cols = state.commitment.num_cols;
    let log_c = num_cols.trailing_zeros() as usize;
    let log_r = num_rows.trailing_zeros() as usize;
    assert_eq!(point.len(), log_c + log_r);

    let q_col = eq_eval_table(&point[..log_c]);
    let q_row = eq_eval_table(&point[log_c..]);

    // 1. Sample random proximity combiner alpha in F_{p^2}^R.
    let alpha = transcript.challenge_fp2_vec("ligero_prox_alpha", num_rows);

    // 2. Compute w_prox = sum_r alpha[r] * rows[r] and w_eval = sum_r q_row[r] * rows[r].
    let mut w_prox = vec![Fp2::ZERO; num_cols];
    let mut w_eval = vec![Fp2::ZERO; num_cols];
    for r in 0..num_rows {
        let a_r = alpha[r];
        let q_r = q_row[r];
        for c in 0..num_cols {
            let v = state.rows[r][c];
            w_prox[c] += a_r.mul_base(v);
            w_eval[c] += q_r.mul_base(v);
        }
    }

    // 3. Compute exact MLE evaluation <q_col, w_eval>.
    let mut claimed_eval = Fp2::ZERO;
    for c in 0..num_cols {
        claimed_eval += q_col[c] * w_eval[c];
    }

    // 4. Absorb w_prox, w_eval, and claimed_eval BEFORE sampling query columns.
    transcript.absorb_fp2_slice("ligero_w_prox", &w_prox);
    transcript.absorb_fp2_slice("ligero_w_eval", &w_eval);
    transcript.absorb_fp2("ligero_claimed_eval", claimed_eval);

    let codeword_len = state.commitment.codeword_len();
    let indices = transcript.challenge_distinct_indices(
        "ligero_col_queries",
        params.num_queries,
        codeword_len,
    );

    let mut opened_columns = Vec::with_capacity(indices.len());
    for col_index in indices {
        let mut values = Vec::with_capacity(num_rows);
        for row in &state.encoded_rows {
            values.push(row[col_index]);
        }
        let merkle_proof = state.tree.prove(col_index);
        opened_columns.push(LigeroColumnOpening {
            col_index,
            values,
            merkle_proof,
        });
    }

    (
        claimed_eval,
        LigeroOpeningProof {
            w_prox,
            w_eval,
            opened_columns,
        },
    )
}

/// Statelessly verifies a [`LigeroOpeningProof`] against `commitment` at `point`.
pub fn verify_mle_ligero(
    commitment: &LigeroCommitment,
    point: &[Fp2],
    claimed_eval: Fp2,
    proof: &LigeroOpeningProof,
    params: LigeroParams,
    transcript: &mut Transcript,
) -> bool {
    if commitment.blowup != params.blowup
        || !commitment.num_rows.is_power_of_two()
        || !commitment.num_cols.is_power_of_two()
        || commitment.blowup < 2
    {
        return false;
    }

    let num_rows = commitment.num_rows;
    let num_cols = commitment.num_cols;
    let log_c = num_cols.trailing_zeros() as usize;
    let log_r = num_rows.trailing_zeros() as usize;
    if point.len() != log_c + log_r {
        return false;
    }
    if proof.w_prox.len() != num_cols || proof.w_eval.len() != num_cols {
        return false;
    }

    let q_col = eq_eval_table(&point[..log_c]);
    let q_row = eq_eval_table(&point[log_c..]);

    // 1. Re-derive proximity combiner alpha.
    let alpha = transcript.challenge_fp2_vec("ligero_prox_alpha", num_rows);

    // 2. Absorb prover's w_prox, w_eval, claimed_eval and re-derive column indices.
    transcript.absorb_fp2_slice("ligero_w_prox", &proof.w_prox);
    transcript.absorb_fp2_slice("ligero_w_eval", &proof.w_eval);
    transcript.absorb_fp2("ligero_claimed_eval", claimed_eval);

    let codeword_len = commitment.codeword_len();
    let expected_indices = transcript.challenge_distinct_indices(
        "ligero_col_queries",
        params.num_queries,
        codeword_len,
    );

    if proof.opened_columns.len() != expected_indices.len() {
        return false;
    }

    // 3. Check claimed_eval == <q_col, w_eval>.
    let mut computed_eval = Fp2::ZERO;
    for c in 0..num_cols {
        computed_eval += q_col[c] * proof.w_eval[c];
    }
    if computed_eval != claimed_eval {
        return false;
    }

    // 4. RS-encode w_prox and w_eval.
    let encoded_prox = rs_encode_fp2(&proof.w_prox, commitment.blowup);
    let encoded_eval = rs_encode_fp2(&proof.w_eval, commitment.blowup);

    // 5. Verify Merkle proofs and both linear-combination checks at every queried column.
    for (opening, &expected_col) in proof.opened_columns.iter().zip(expected_indices.iter()) {
        if opening.col_index != expected_col
            || opening.values.len() != num_rows
            || opening.merkle_proof.leaf_index != expected_col
            || opening.merkle_proof.leaf_count != codeword_len
        {
            return false;
        }
        let col_bytes = encode_column_bytes(&opening.values);
        if !verify_merkle_proof(&commitment.root, &col_bytes, &opening.merkle_proof) {
            return false;
        }

        let mut lhs_prox = Fp2::ZERO;
        let mut lhs_eval = Fp2::ZERO;
        for r in 0..num_rows {
            let v = opening.values[r];
            lhs_prox += alpha[r].mul_base(v);
            lhs_eval += q_row[r].mul_base(v);
        }
        if lhs_prox != encoded_prox[expected_col] || lhs_eval != encoded_eval[expected_col] {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::eval_mle_fp;

    #[test]
    fn merkle_tree_proves_and_rejects_tamper() {
        let leaves: Vec<Vec<u8>> = (0..13).map(|i| format!("leaf-{i}").into_bytes()).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        let root = tree.root();

        for (i, payload) in leaves.iter().enumerate() {
            let proof = tree.prove(i);
            assert!(verify_merkle_proof(&root, payload, &proof));
            let mut tampered = payload.clone();
            tampered[0] ^= 0x01;
            assert!(!verify_merkle_proof(&root, &tampered, &proof));
        }
    }

    #[test]
    fn ligero_pcs_roundtrip_and_tamper_rejection() {
        let values: Vec<Fp> = (0..64).map(|i| Fp::new(i * 17 + 3)).collect();
        let params = LigeroParams::default();
        let state = commit_vector_ligero(&values, params);

        let mut prover_t = Transcript::new("test.ligero.v1");
        let point = prover_t.challenge_fp2_vec("eval_point", 6);
        let (claimed, proof) = open_mle_ligero(&state, &point, params, &mut prover_t);
        assert_eq!(claimed, eval_mle_fp(&values, &point));

        // Valid verification succeeds.
        let mut verifier_t = Transcript::new("test.ligero.v1");
        let v_point = verifier_t.challenge_fp2_vec("eval_point", 6);
        assert!(verify_mle_ligero(
            &state.commitment,
            &v_point,
            claimed,
            &proof,
            params,
            &mut verifier_t
        ));

        // Wrong claimed evaluation rejected.
        let mut bad_t = Transcript::new("test.ligero.v1");
        let b_point = bad_t.challenge_fp2_vec("eval_point", 6);
        assert!(!verify_mle_ligero(
            &state.commitment,
            &b_point,
            claimed + Fp2::ONE,
            &proof,
            params,
            &mut bad_t
        ));

        // Tampered column value rejected.
        let mut bad_proof = proof.clone();
        bad_proof.opened_columns[0].values[0] += Fp::ONE;
        let mut bad_t2 = Transcript::new("test.ligero.v1");
        let b_point2 = bad_t2.challenge_fp2_vec("eval_point", 6);
        assert!(!verify_mle_ligero(
            &state.commitment,
            &b_point2,
            claimed,
            &bad_proof,
            params,
            &mut bad_t2
        ));
    }
}
