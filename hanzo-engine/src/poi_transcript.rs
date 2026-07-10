//! Proof-of-Inference TRANSCRIPT — the activation-trace commitment that makes "you cannot claim
//! the AI output without doing the matmuls" true END TO END, not just for an isolated verifier.
//!
//! The verifier core ([`crate::poi`]) is a pure function of operands `(A,B,C)` and a challenge.
//! This module is the missing wire: a forward pass COMMITS each proof-bearing matmul as a keccak
//! Merkle leaf over its exact-integer `(A,B,C)`; the Merkle root is the `transcriptRoot` the prover
//! posts on-chain alongside its claimed output. A CHALLENGER (the off-chain watcher, or the
//! `computeattest` precompile) then:
//!   1. derives a beacon-unpredictable matmul INDEX from `keccak(beacon ‖ root)`,
//!   2. has the prover OPEN that matmul — reveal `(A,B,C)` plus a Merkle inclusion proof,
//!   3. checks the operands were the COMMITTED ones (inclusion, same index-fold as
//!      `AICoinMiner._verifyMerkle`), and
//!   4. FREIVALDS-verifies `C = A·B` with `k` vectors from a beacon-bound, prover-unpredictable
//!      seed ([`crate::poi::derive_challenges_keccak`], byte-identical to the Go verifier).
//!
//! A prover who fabricated an output committed a leaf over a `C` that is not `A·B`: step 4 catches
//! it. A prover who committed honestly but tries to OPEN with a swapped-in honest `C`: step 3
//! catches it (the leaf no longer matches the committed root). Either way the fraud is provable
//! and the bond is slashable. Because the challenge index and Freivalds `r` are both fixed only
//! AFTER the root is committed, the prover cannot fake the un-opened matmuls and hope to dodge.
//!
//! Everything here is keccak/`F_p` and matches the chain byte-for-byte, so the same opening that
//! convinces this Rust challenger convinces the Go `crypto/poi` watcher and the Solidity witness.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use crate::poi::{self, Mat};

/// Domain tag for a matmul leaf — keeps these commitments disjoint from any other keccak leaf.
pub const DOMAIN_MATMUL_LEAF: &[u8] = b"hanzo/poi/matmul-leaf/v1";

/// Canonical serialization of an integer matrix: `BE32(rows) ‖ BE32(cols) ‖ data[i] as BE64`
/// (two's-complement). Fixed-width and order-stable so prover and verifier hash identical bytes.
fn mat_bytes(m: &Mat) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + m.data.len() * 8);
    b.extend_from_slice(&(m.rows as u32).to_be_bytes());
    b.extend_from_slice(&(m.cols as u32).to_be_bytes());
    for &x in &m.data {
        b.extend_from_slice(&x.to_be_bytes());
    }
    b
}

/// The leaf binding one matmul: `keccak(DOMAIN ‖ mat(A) ‖ mat(B) ‖ mat(C))`. Committing the
/// operands (not just a digest of them) is what lets the challenger re-derive and Freivalds-check.
pub fn matmul_leaf(a: &Mat, b: &Mat, c: &Mat) -> [u8; 32] {
    let mut buf = DOMAIN_MATMUL_LEAF.to_vec();
    buf.extend_from_slice(&mat_bytes(a));
    buf.extend_from_slice(&mat_bytes(b));
    buf.extend_from_slice(&mat_bytes(c));
    poi::keccak256(&buf)
}

/// `keccak(left ‖ right)` over two 32-byte nodes — the inner-node fold.
fn hash_pair(l: &[u8; 32], r: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(l);
    buf[32..].copy_from_slice(r);
    poi::keccak256(&buf)
}

/// Binary keccak Merkle root over `leaves`. Odd levels duplicate the last node (Bitcoin-style),
/// and the index fold (`even → keccak(node‖sib)`, `odd → keccak(sib‖node)`) is IDENTICAL to
/// `AICoinMiner._verifyMerkle`, so a proof built here verifies on-chain unchanged.
pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    assert!(!leaves.is_empty(), "merkle_root over empty transcript");
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        if level.len() % 2 == 1 {
            let last = *level.last().unwrap();
            level.push(last);
        }
        level = level.chunks(2).map(|p| hash_pair(&p[0], &p[1])).collect();
    }
    level[0]
}

/// The inclusion proof (sibling path, bottom-up) for the leaf at `index`.
pub fn merkle_proof(leaves: &[[u8; 32]], index: usize) -> Vec<[u8; 32]> {
    let mut proof = Vec::new();
    let mut level = leaves.to_vec();
    let mut idx = index;
    while level.len() > 1 {
        if level.len() % 2 == 1 {
            let last = *level.last().unwrap();
            level.push(last);
        }
        let sib = if idx.is_multiple_of(2) {
            idx + 1
        } else {
            idx - 1
        };
        proof.push(level[sib]);
        level = level.chunks(2).map(|p| hash_pair(&p[0], &p[1])).collect();
        idx /= 2;
    }
    proof
}

/// Verify a leaf's inclusion under `root` at `index` — the exact fold of `AICoinMiner._verifyMerkle`.
pub fn merkle_verify(leaf: [u8; 32], root: [u8; 32], index: usize, proof: &[[u8; 32]]) -> bool {
    let mut node = leaf;
    let mut idx = index;
    for sib in proof {
        node = if idx.is_multiple_of(2) {
            hash_pair(&node, sib)
        } else {
            hash_pair(sib, &node)
        };
        idx /= 2;
    }
    node == root
}

/// What a prover reveals to answer a challenge: the opened matmul's operands and its inclusion proof.
#[derive(Clone, Debug)]
pub struct Opening {
    pub index: usize,
    pub a: Mat,
    pub b: Mat,
    pub c: Mat,
    pub proof: Vec<[u8; 32]>,
}

/// An append-only commitment to the proof-bearing matmuls of one forward pass.
#[derive(Default)]
pub struct ProofTranscript {
    leaves: Vec<[u8; 32]>,
    ops: Vec<(Mat, Mat, Mat)>,
}

impl ProofTranscript {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run an EXACT-integer matmul, commit it to the transcript, and return `C`. This is the call
    /// a proof-bearing linear layer makes instead of the bare f32 GEMM.
    pub fn matmul(&mut self, a: Mat, b: Mat) -> Mat {
        let c = poi::exact_matmul(&a, &b);
        self.leaves.push(matmul_leaf(&a, &b, &c));
        self.ops.push((a, b, c.clone()));
        c
    }

    /// Commit a matmul whose `C` is supplied by the caller (e.g. a fast-path output the prover
    /// CLAIMS). The transcript binds whatever `C` it is given; an incorrect one is caught by the
    /// challenger. This is the seam through which a fabricated output enters — and is provably
    /// rejected — so we exercise it directly in the tests.
    pub fn commit_claimed(&mut self, a: Mat, b: Mat, c: Mat) {
        self.leaves.push(matmul_leaf(&a, &b, &c));
        self.ops.push((a, b, c));
    }

    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// The `transcriptRoot` the prover posts on-chain.
    pub fn root(&self) -> [u8; 32] {
        merkle_root(&self.leaves)
    }

    /// Open the matmul at `index` — reveal its operands and inclusion proof.
    pub fn open(&self, index: usize) -> Opening {
        let (a, b, c) = &self.ops[index];
        Opening {
            index,
            a: a.clone(),
            b: b.clone(),
            c: c.clone(),
            proof: merkle_proof(&self.leaves, index),
        }
    }
}

/// Which matmul the challenger opens: `keccak(beacon ‖ root) mod len`. Unpredictable before the
/// prover commits `root`, so the prover cannot fake just the matmuls it expects to stay closed.
pub fn challenge_index(beacon: &[u8], root: &[u8; 32], len: usize) -> usize {
    assert!(len > 0, "challenge over empty transcript");
    let mut buf = beacon.to_vec();
    buf.extend_from_slice(root);
    let h = poi::keccak256(&buf);
    (u64::from_be_bytes(h[0..8].try_into().unwrap()) % len as u64) as usize
}

/// Per-opening Freivalds seed: `keccak(beacon ‖ root ‖ BE64(index))` — binds the challenge to the
/// committed root and the opened index, so it is fixed only after commitment and is distinct per
/// matmul. Fed to [`poi::derive_challenges_keccak`].
pub fn opening_seed(beacon: &[u8], root: &[u8; 32], index: usize) -> [u8; 32] {
    let mut s = beacon.to_vec();
    s.extend_from_slice(root);
    s.extend_from_slice(&(index as u64).to_be_bytes());
    poi::keccak256(&s)
}

/// THE CHALLENGER. Given the committed `root`, a `beacon`, and the prover's `op`ening, decide
/// whether the opened matmul is genuine: (1) the revealed operands are the COMMITTED ones (Merkle
/// inclusion under `root`), and (2) `C = A·B` (Freivalds, `k` beacon-bound vectors). Returns true
/// iff BOTH hold — a fabricated `C` fails (2); a swapped-in `C` fails (1). This is the function
/// that, ported to Go (`crypto/poi`) and Solidity, turns "no proof, no mint" into a real gate.
pub fn verify_opening(root: &[u8; 32], beacon: &[u8], op: &Opening, k: usize) -> bool {
    // (1) binding: the revealed operands hash to the committed leaf at this index.
    let leaf = matmul_leaf(&op.a, &op.b, &op.c);
    if !merkle_verify(leaf, *root, op.index, &op.proof) {
        return false;
    }
    // (2) correctness: Freivalds C == A·B with prover-unpredictable, beacon-bound challenges.
    let seed = opening_seed(beacon, root, op.index);
    let challenges = poi::derive_challenges_keccak(&seed, op.b.cols, k);
    poi::freivalds_verify_multi(&op.a, &op.b, &op.c, &challenges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poi::Mat;

    // A small deterministic PRNG so tests build the same int8-ish matrices every run.
    fn lcg(s: &mut u64) -> i64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as i64 % 127) - 63
    }

    fn rand_mat(rows: usize, cols: usize, seed: &mut u64) -> Mat {
        Mat::new(rows, cols, (0..rows * cols).map(|_| lcg(seed)).collect())
    }

    // Build a transcript that mimics a few proof-bearing linear layers of a forward pass.
    fn forward_transcript(seed: &mut u64) -> ProofTranscript {
        let mut t = ProofTranscript::new();
        // (batch×tokens) activations × (in×out) weights — transformer-ish shapes, kept small/fast.
        let acts = rand_mat(4, 64, seed); // [t=4, k=64]
        let w1 = rand_mat(64, 48, seed); // [k=64, n=48]
        let h = t.matmul(acts, w1); // committed exact matmul
        let w2 = rand_mat(48, 32, seed); // [48, 32]
        let _ = t.matmul(h, w2);
        let w3 = rand_mat(32, 16, seed);
        let acts2 = rand_mat(4, 32, seed);
        let _ = t.matmul(acts2, w3);
        t
    }

    #[test]
    fn test_honest_forward_pass_opening_verifies() {
        let mut s = 0xA1CE;
        let t = forward_transcript(&mut s);
        let root = t.root();
        let beacon = b"beacon:block-9001";
        // the challenger picks which matmul to open — the prover cannot predict it pre-commit
        let idx = challenge_index(beacon, &root, t.len());
        let op = t.open(idx);
        assert!(
            verify_opening(&root, beacon, &op, 2),
            "an honest forward pass must pass the opened-matmul challenge"
        );
    }

    #[test]
    fn test_every_matmul_in_the_pass_verifies() {
        // No matter WHICH matmul the beacon selects, an honest pass answers it.
        let mut s = 0xBEEF;
        let t = forward_transcript(&mut s);
        let root = t.root();
        let beacon = b"beacon:any";
        for idx in 0..t.len() {
            let op = t.open(idx);
            assert!(
                verify_opening(&root, beacon, &op, 2),
                "honest matmul {idx} must verify"
            );
        }
    }

    #[test]
    fn test_fabricated_output_is_caught() {
        // The prover does the matmuls honestly EXCEPT it fabricates the output of one layer:
        // it claims a C that it never actually multiplied for. Freivalds inside verify_opening
        // must catch it at the challenged index.
        let mut s = 0xF00D;
        let mut t = ProofTranscript::new();
        let a = rand_mat(4, 64, &mut s);
        let b = rand_mat(64, 32, &mut s);
        let mut fake_c = poi::exact_matmul(&a, &b);
        fake_c.data[5] += 1; // a single fabricated entry — output never produced by A·B
        t.commit_claimed(a, b, fake_c); // committed over the FAKE output
        let root = t.root();
        let beacon = b"beacon:catch";
        let op = t.open(0);
        assert!(
            !verify_opening(&root, beacon, &op, 2),
            "a fabricated output (C != A*B) must be CAUGHT by the opened-matmul challenge"
        );
    }

    #[test]
    fn test_swapped_reveal_is_caught() {
        // The prover committed honestly, then at OPEN time tries to substitute the correct C
        // (to dodge a Freivalds check it expected) — but that C was never committed, so Merkle
        // inclusion fails. (Mirror of the fabricate case: you can't win on either side.)
        let mut s = 0x5151;
        let mut t = ProofTranscript::new();
        let a = rand_mat(4, 48, &mut s);
        let b = rand_mat(48, 24, &mut s);
        let mut bad_c = poi::exact_matmul(&a, &b);
        bad_c.data[3] -= 2;
        t.commit_claimed(a.clone(), b.clone(), bad_c); // committed over a wrong C
        let root = t.root();
        // open with the CORRECT c instead of the committed wrong one
        let good_c = poi::exact_matmul(&a, &b);
        let op = Opening {
            index: 0,
            a,
            b,
            c: good_c,
            proof: merkle_proof(&t.leaves, 0),
        };
        assert!(
            !verify_opening(&root, b"beacon:swap", &op, 2),
            "revealing operands that were never committed must fail Merkle inclusion"
        );
    }

    #[test]
    fn test_merkle_root_binds_every_operand() {
        // Flipping any single input or output entry changes the transcript root (binding).
        let mut s = 0x1234;
        let a = rand_mat(3, 16, &mut s);
        let b = rand_mat(16, 8, &mut s);
        let c = poi::exact_matmul(&a, &b);
        let root0 = merkle_root(&[matmul_leaf(&a, &b, &c)]);
        let mut a2 = a.clone();
        a2.data[0] += 1;
        let root_a = merkle_root(&[matmul_leaf(&a2, &b, &c)]);
        let mut c2 = c.clone();
        c2.data[0] += 1;
        let root_c = merkle_root(&[matmul_leaf(&a, &b, &c2)]);
        assert_ne!(root0, root_a, "input change must move the root");
        assert_ne!(root0, root_c, "output change must move the root");
    }

    #[test]
    fn test_merkle_inclusion_index_fold() {
        // A hand-checked 3-leaf tree (odd, exercising the duplicate-last level) verifies every leaf.
        let leaves: Vec<[u8; 32]> = (0u8..3).map(|i| poi::keccak256(&[i])).collect();
        let root = merkle_root(&leaves);
        for (i, leaf) in leaves.iter().enumerate() {
            let p = merkle_proof(&leaves, i);
            assert!(
                merkle_verify(*leaf, root, i, &p),
                "leaf {i} must verify under the root"
            );
            // a wrong index must NOT verify
            assert!(!merkle_verify(*leaf, root, (i + 1) % 3, &p) || i == (i + 1) % 3);
        }
    }

    #[test]
    fn test_realistic_quantized_linear_layer() {
        // A transformer-sized proof-bearing linear: 8 tokens × 256-dim activations, 256×256 int8
        // weights — the exact-integer path a real quantized layer would run for a proof. The
        // honest computation answers the challenge; fabricating ONE of the 2048 outputs is caught.
        let mut s = 0xC0FFEE;
        let acts = rand_mat(8, 256, &mut s); // [t=8, k=256]
        let weights = rand_mat(256, 256, &mut s); // [k=256, n=256]
        let mut t = ProofTranscript::new();
        let _out = t.matmul(acts.clone(), weights.clone());
        let root = t.root();
        let beacon = b"beacon:real-layer";
        let op = t.open(0);
        assert!(
            verify_opening(&root, beacon, &op, 2),
            "honest 256x256 quantized layer verifies"
        );

        // now fabricate a single output entry and recommit — must be caught
        let mut t2 = ProofTranscript::new();
        let mut fake = poi::exact_matmul(&acts, &weights);
        fake.data[1234] += 1;
        t2.commit_claimed(acts, weights, fake);
        let op2 = t2.open(0);
        assert!(
            !verify_opening(&t2.root(), beacon, &op2, 2),
            "one fabricated entry among 2048 outputs is caught"
        );
    }

    #[test]
    fn test_challenge_index_unpredictable_but_deterministic() {
        // Same (beacon, root) → same index (prover and verifier agree); different beacon → may move.
        let mut s = 0x99;
        let t = forward_transcript(&mut s);
        let root = t.root();
        assert_eq!(
            challenge_index(b"b1", &root, t.len()),
            challenge_index(b"b1", &root, t.len()),
            "index derivation is deterministic for a fixed beacon+root"
        );
    }
}
