//! Proof-of-Inference core: the Freivalds matmul verifier.
//!
//! This is the mathematical heart of "you cannot claim the AI work without doing the
//! matmuls." An LLM forward pass is ~95% matrix multiplications `C = A·B`. Recomputing `C`
//! to check it costs `O(t·k·n)`. Freivalds (1977) checks it for a random vector `r` via
//! `A·(B·r) == C·r` in `O(t·k + k·n + t·n)` — one order cheaper — and if `C ≠ A·B` then a
//! random `r` exposes it with probability `≥ 1 - 1/p` per vector. So a prover who fabricated
//! `C` (claimed an output without performing the multiply) is caught with overwhelming
//! probability, while an honest prover passes deterministically.
//!
//! We work over the prime field `F_p`, `p = 2^61 - 1` (a Mersenne prime), on the EXACT
//! INTEGER accumulator of the engine's int8 matmul path (`hanzo-quant` f8q8 i8·i8 → i32).
//! Integer arithmetic is bit-exact across CPU/GPU/backends, so the check has ZERO
//! false-reject — the determinism the rest of the Proof-of-AI scheme relies on. Soundness
//! error is `1/p ≈ 2^-61` per challenge vector; `k = 2` vectors give `≤ 2^-122`.
//!
//! This module is backend-agnostic and dependency-free (no engine forward pass needed) so
//! the soundness can be unit-tested in isolation: an honest `A·B` passes; a single tampered
//! entry of `C` is caught. The activation-trace Merkle commitment (keccak MMR) and the
//! per-layer commit hooks in the forward pass are separate, later increments that feed
//! `(A, B, C)` operands into this verifier under challenge.
//!
//! NOTE on the challenge: a verifier derives `r` from an on-chain beacon AFTER the prover
//! commits, so the prover cannot pre-pick `A, B, C` to satisfy a known `r`. The production
//! derivation is keccak over the beacon (to match the chain's wire spec); the deterministic
//! expansion here (splitmix64) has the identical soundness for the unit test, where what
//! matters is that `r` is fixed and reproducible by both sides.

/// The Mersenne prime field modulus `p = 2^61 - 1`. Bounds the int8 accumulator and the
/// Freivalds intermediates within 61 bits, so a single `u128` multiply never overflows.
pub const P: u64 = (1u64 << 61) - 1;

/// A dense row-major integer matrix (the int8 path's exact i32/i64 accumulators).
#[derive(Clone, Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<i64>, // row-major: element (i,j) = data[i*cols + j]
}

impl Mat {
    pub fn new(rows: usize, cols: usize, data: Vec<i64>) -> Self {
        assert_eq!(data.len(), rows * cols, "Mat data length must be rows*cols");
        Self { rows, cols, data }
    }
    #[inline]
    fn get(&self, i: usize, j: usize) -> i64 {
        self.data[i * self.cols + j]
    }
}

/// Reduce a signed integer into `[0, p)`. Handles negative int8 weights/activations.
#[inline]
fn to_field(x: i64) -> u64 {
    x.rem_euclid(P as i64) as u64
}

/// `(a + b) mod p` for `a, b < p` (sum `< 2p < 2^62`, no overflow).
#[inline]
fn fadd(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= P {
        s - P
    } else {
        s
    }
}

/// `(a * b) mod p` via a single 128-bit multiply (both `< p < 2^61`).
#[inline]
fn fmul(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % P as u128) as u64
}

/// `m · v` over `F_p`: `m` is an INTEGER matrix (reduced per entry), `v` a field vector of
/// length `m.cols`. Returns a field vector of length `m.rows`.
fn matvec(m: &Mat, v: &[u64]) -> Vec<u64> {
    assert_eq!(m.cols, v.len(), "matvec dim mismatch");
    let mut out = vec![0u64; m.rows];
    for i in 0..m.rows {
        let mut acc = 0u64;
        for j in 0..m.cols {
            acc = fadd(acc, fmul(to_field(m.get(i, j)), v[j]));
        }
        out[i] = acc;
    }
    out
}

/// Verify `C == A·B` probabilistically for ONE challenge vector `r` (length `n = A.cols-derived`).
/// Dims: `A` is `[t × k]`, `B` is `[k × n]`, `C` is `[t × n]`, `r` is length `n`.
/// Computes `A·(B·r)` and `C·r` over `F_p` and compares. `O(tk + kn + tn)`.
pub fn freivalds_verify(a: &Mat, b: &Mat, c: &Mat, r: &[u64]) -> bool {
    // shape contract
    if a.cols != b.rows || b.cols != c.cols || a.rows != c.rows || r.len() != b.cols {
        return false;
    }
    let br = matvec(b, r); // length k
    let abr = matvec(a, &br); // length t
    let cr = matvec(c, r); // length t
    abr == cr
}

/// Verify against `k` independent challenge vectors. Honest `C` passes ALL; a tampered `C`
/// fails at least one except with probability `≤ (1/p)^k`. Returns true iff every vector passes.
pub fn freivalds_verify_multi(a: &Mat, b: &Mat, c: &Mat, challenges: &[Vec<u64>]) -> bool {
    !challenges.is_empty() && challenges.iter().all(|r| freivalds_verify(a, b, c, r))
}

/// Deterministically derive `k` challenge vectors of length `n` over `F_p` from a `seed`.
/// Reproducible by prover and verifier; in production `seed` is keccak(beacon ‖ task ‖ root)
/// so it is unpredictable before commitment. Here a splitmix64 stream gives the same
/// reproducibility for tests (soundness is identical; only the unpredictability source differs).
pub fn derive_challenges(seed: u64, n: usize, k: usize) -> Vec<Vec<u64>> {
    let mut state = seed;
    let mut next = || -> u64 {
        // splitmix64
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (0..k)
        .map(|_| (0..n).map(|_| next() % P).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // A·B for the 2x2 fixture, computed exactly: A=[[1,2],[3,4]], B=[[5,6],[7,8]].
    fn fixture() -> (Mat, Mat, Mat) {
        let a = Mat::new(2, 2, vec![1, 2, 3, 4]);
        let b = Mat::new(2, 2, vec![5, 6, 7, 8]);
        // C = A·B = [[19,22],[43,50]]
        let c = Mat::new(2, 2, vec![19, 22, 43, 50]);
        (a, b, c)
    }

    #[test]
    fn test_honest_matmul_passes() {
        let (a, b, c) = fixture();
        let ch = derive_challenges(0xBE16A, 2, 4);
        assert!(freivalds_verify_multi(&a, &b, &c, &ch), "honest C = A·B must pass every challenge");
    }

    #[test]
    fn test_tampered_output_is_caught() {
        let (a, b, _) = fixture();
        // Flip ONE entry: 50 -> 51. A faker who fabricated this output never did the multiply.
        let c_bad = Mat::new(2, 2, vec![19, 22, 43, 51]);
        let ch = derive_challenges(0xBE16A, 2, 4);
        assert!(
            !freivalds_verify_multi(&a, &b, &c_bad, &ch),
            "a single tampered output entry must be CAUGHT (Freivalds soundness)"
        );
    }

    #[test]
    fn test_negative_entries_field_reduction() {
        // int8 activations/weights are signed; to_field must handle negatives exactly.
        let a = Mat::new(2, 2, vec![1, -2, 3, 4]);
        let b = Mat::new(2, 2, vec![5, 6, -7, 8]);
        // C = A·B = [[1*5+(-2)*(-7), 1*6+(-2)*8],[3*5+4*(-7), 3*6+4*8]] = [[19,-10],[-13,50]]
        let c = Mat::new(2, 2, vec![19, -10, -13, 50]);
        let ch = derive_challenges(7, 2, 4);
        assert!(freivalds_verify_multi(&a, &b, &c, &ch), "honest matmul with negatives passes");
        let c_bad = Mat::new(2, 2, vec![19, -10, -13, 49]); // flip -> caught
        assert!(!freivalds_verify_multi(&a, &b, &c_bad, &ch), "tampered negative-entry matmul caught");
    }

    #[test]
    fn test_challenge_derivation_deterministic() {
        // Prover and verifier must derive the IDENTICAL challenge from the same beacon seed.
        assert_eq!(derive_challenges(42, 8, 3), derive_challenges(42, 8, 3));
        assert_ne!(derive_challenges(42, 8, 3), derive_challenges(43, 8, 3));
    }

    #[test]
    fn test_wrong_dims_rejected() {
        let a = Mat::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let b = Mat::new(2, 2, vec![1, 2, 3, 4]); // b.rows(2) != a.cols(3)
        let c = Mat::new(2, 2, vec![1, 2, 3, 4]);
        assert!(!freivalds_verify(&a, &b, &c, &[1, 1]), "shape mismatch must fail closed");
    }

    #[test]
    fn test_larger_random_matmul_catches_sparse_cheat() {
        // 16x24 · 24x16: compute C exactly, then flip a single deep entry. Freivalds must catch
        // it — the sparse-cheat case that motivates challenging every layer.
        let (t, k, n) = (16usize, 24usize, 16usize);
        let mut sa = 0x1234_5678u64;
        let mut sb = 0x9abc_def0u64;
        let mut rnd = |s: &mut u64| -> i64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 33) as i64 % 127) - 63 // small signed, int8-ish
        };
        let a = Mat::new(t, k, (0..t * k).map(|_| rnd(&mut sa)).collect());
        let b = Mat::new(k, n, (0..k * n).map(|_| rnd(&mut sb)).collect());
        // exact integer product
        let mut cdata = vec![0i64; t * n];
        for i in 0..t {
            for j in 0..n {
                let mut acc = 0i64;
                for x in 0..k {
                    acc += a.data[i * k + x] * b.data[x * n + j];
                }
                cdata[i * n + j] = acc;
            }
        }
        let c = Mat::new(t, n, cdata.clone());
        let ch = derive_challenges(0xC0FFEE, n, 2);
        assert!(freivalds_verify_multi(&a, &b, &c, &ch), "honest large matmul passes");
        let mut bad = cdata.clone();
        bad[7 * n + 11] += 1; // one entry off by one
        let c_bad = Mat::new(t, n, bad);
        assert!(!freivalds_verify_multi(&a, &b, &c_bad, &ch), "off-by-one in one entry is caught");
    }
}
