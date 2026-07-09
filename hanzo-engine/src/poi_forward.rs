//! Proof-bearing forward pass — a real quantized-linear computation that EMITS a sound
//! activation-trace transcript.
//!
//! "Sound" is the whole point and the thing the audit's CC-01 said was missing: the value a layer
//! RETURNS must be the value it COMMITS. A [`ProvableLinear`] computes `y = dequant(x_i8 · W_i8)` on
//! the EXACT-INTEGER accumulator (`i8×i8 → i64`, whole-K, no cross-block `f32`), commits the exact
//! operands `(A=x_i8, B=W_i8, C=x_i8·W_i8)` to the active transcript, and returns `dequant(C)`. So a
//! challenger's Freivalds over the committed `(A,B,C)` re-checks the SAME computation that produced
//! the output — not an `f32` shadow that disagrees with it. Run a forward inside [`prove`] and you
//! get the output plus a [`ProofTranscript`] whose root the prover posts on-chain; a challenger
//! opens a beacon-selected matmul and the gate/precompile catch any fabrication.
//!
//! The quantization is deterministic (per-row symmetric int8), so two honest machines emit the
//! identical transcript — the reproducibility the proof needs. dequant is a public affine map of
//! the committed `C`, so committing `C` commits the output: a prover cannot claim an output its
//! committed matmuls did not produce.

use crate::poi::{exact_matmul, Mat};
use crate::poi_transcript::ProofTranscript;
use std::cell::RefCell;

thread_local! {
    /// The proof transcript that proof-bearing linears emit into, when one is installed.
    static PROOF_CTX: RefCell<Option<ProofTranscript>> = const { RefCell::new(None) };
}

/// Run `f` with proof emission ON. Every [`ProvableLinear::forward`] called inside commits its
/// exact-integer matmul; returns `f`'s result paired with the [`ProofTranscript`] (the prover keeps
/// it to answer challenges — `.root()` is posted on-chain, `.open(i)` answers a spot-check).
/// Re-entrant-safe: a previously-installed context is restored on exit.
pub fn prove<T>(f: impl FnOnce() -> T) -> (T, ProofTranscript) {
    let prev = PROOF_CTX.with(|c| c.borrow_mut().replace(ProofTranscript::new()));
    let out = f();
    let transcript = PROOF_CTX.with(|c| {
        let mut slot = c.borrow_mut();
        let t = slot
            .take()
            .expect("proof context present at end of prove()");
        *slot = prev;
        t
    });
    (out, transcript)
}

/// True while a proof is being emitted (a [`ProofTranscript`] is installed).
pub fn proving() -> bool {
    PROOF_CTX.with(|c| c.borrow().is_some())
}

/// Commit one matmul to the active transcript, if any. Called by proof-bearing layers.
fn commit(a: Mat, b: Mat, c: Mat) {
    PROOF_CTX.with(|cell| {
        if let Some(t) = cell.borrow_mut().as_mut() {
            t.commit_claimed(a, b, c);
        }
    });
}

/// Deterministic per-row symmetric int8 quantization of a row-major `[rows × cols]` f32 matrix.
/// Returns the int8 values (as i64, ready for [`exact_matmul`]) and the per-row dequant scale
/// `scale[r] = max|x[r]| / 127`. A zero row maps to scale 0 (all-zero int8) — exact.
pub fn quantize_rows_i8(x: &[f32], rows: usize, cols: usize) -> (Vec<i64>, Vec<f64>) {
    assert_eq!(x.len(), rows * cols, "quantize_rows_i8 shape");
    let mut q = vec![0i64; rows * cols];
    let mut scale = vec![0f64; rows];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs())) as f64;
        let s = amax / 127.0;
        scale[r] = s;
        if s > 0.0 {
            for c in 0..cols {
                // round-half-to-even-free: deterministic round-half-away, clamped to int8.
                let v = (row[c] as f64 / s).round().clamp(-127.0, 127.0);
                q[r * cols + c] = v as i64;
            }
        }
    }
    (q, scale)
}

/// A linear layer `y = x·Wᵀ` with int8-quantized weights, computed on the EXACT-INTEGER accumulator
/// and (when proving) emitting its matmul to the transcript. The standard quantized-linear: weights
/// are int8 with a per-output-row dequant scale; activations are quantized per-row at call time.
#[derive(Clone, Debug)]
pub struct ProvableLinear {
    in_features: usize,
    out_features: usize,
    /// `B` in `[in_features × out_features]` int8 (column = output unit), ready for exact_matmul.
    w_i8: Mat,
    /// Per-output-unit dequant scale (length `out_features`).
    w_scale: Vec<f64>,
}

impl ProvableLinear {
    /// Quantize an f32 weight given row-major as `[out_features × in_features]` (the PyTorch
    /// `nn.Linear` layout) into the proof-bearing representation. Per-output-row symmetric int8.
    pub fn from_f32(weight: &[f32], out_features: usize, in_features: usize) -> Self {
        assert_eq!(
            weight.len(),
            out_features * in_features,
            "ProvableLinear weight shape"
        );
        // quantize per output row, then store TRANSPOSED as [in × out] for x[t×in]·W[in×out].
        let (q_row, w_scale) = quantize_rows_i8(weight, out_features, in_features);
        let mut w_t = vec![0i64; in_features * out_features];
        for o in 0..out_features {
            for i in 0..in_features {
                w_t[i * out_features + o] = q_row[o * in_features + i];
            }
        }
        Self {
            in_features,
            out_features,
            w_i8: Mat::new(in_features, out_features, w_t),
            w_scale,
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// The per-output-unit dequant scales — PUBLIC model parameters (part of the model spec). The
    /// output is `dequant(C)` using these, so anyone can recompute the claimed output from the
    /// committed `C` and reject a prover that claims a different one.
    pub fn weight_scale(&self) -> &[f64] {
        &self.w_scale
    }

    /// `y = dequant(quant(x) · W_i8)`. `x` is row-major `[tokens × in_features]` f32; returns
    /// row-major `[tokens × out_features]` f32. Runs the EXACT matmul, commits `(A,B,C)` when
    /// proving, and dequantizes the committed `C` — so the returned value is exactly what was
    /// committed.
    pub fn forward(&self, x: &[f32], tokens: usize) -> Vec<f32> {
        assert_eq!(
            x.len(),
            tokens * self.in_features,
            "ProvableLinear forward shape"
        );
        let (x_q, x_scale) = quantize_rows_i8(x, tokens, self.in_features);
        let a = Mat::new(tokens, self.in_features, x_q);
        let c = exact_matmul(&a, &self.w_i8); // [tokens × out] EXACT i64
                                              // dequant: y[t][o] = C[t][o] * x_scale[t] * w_scale[o]
        let mut y = vec![0f32; tokens * self.out_features];
        for t in 0..tokens {
            for o in 0..self.out_features {
                y[t * self.out_features + o] =
                    (c.data[t * c.cols + o] as f64 * x_scale[t] * self.w_scale[o]) as f32;
            }
        }
        commit(a, self.w_i8.clone(), c);
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poi_transcript::{challenge_index, verify_opening};

    // a tiny deterministic f32 generator
    fn gen(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((*seed >> 33) as i64 % 200) - 100) as f32 / 50.0 // ~[-2, 2]
            })
            .collect()
    }

    // A real SwiGLU FFN block: gate/up project up, SiLU(gate)*up, down projects back.
    struct FFN {
        gate: ProvableLinear,
        up: ProvableLinear,
        down: ProvableLinear,
    }
    impl FFN {
        fn new(d: usize, hidden: usize, seed: &mut u64) -> Self {
            Self {
                gate: ProvableLinear::from_f32(&gen(hidden * d, seed), hidden, d),
                up: ProvableLinear::from_f32(&gen(hidden * d, seed), hidden, d),
                down: ProvableLinear::from_f32(&gen(d * hidden, seed), d, hidden),
            }
        }
        fn forward(&self, x: &[f32], tokens: usize, d: usize, hidden: usize) -> Vec<f32> {
            let g = self.gate.forward(x, tokens);
            let u = self.up.forward(x, tokens);
            // SiLU(g) * u
            let mut h = vec![0f32; tokens * hidden];
            for i in 0..h.len() {
                let s = g[i] / (1.0 + (-g[i]).exp());
                h[i] = s * u[i];
            }
            let _ = d;
            self.down.forward(&h, tokens)
        }
    }

    #[test]
    fn test_provable_ffn_emits_verifiable_transcript() {
        let (d, hidden, tokens) = (64usize, 128usize, 4usize);
        let mut seed = 0xF00D;
        let ffn = FFN::new(d, hidden, &mut seed);
        let x = gen(tokens * d, &mut seed);

        // run the REAL forward under proof emission
        let (y, transcript) = prove(|| ffn.forward(&x, tokens, d, hidden));
        assert_eq!(y.len(), tokens * d, "FFN returns [tokens × d]");
        assert_eq!(
            transcript.len(),
            3,
            "gate + up + down = 3 committed matmuls"
        );

        // the prover posts this root; a challenger opens a beacon-selected matmul
        let root = transcript.root();
        let beacon = b"beacon:ffn-block-1";
        let idx = challenge_index(beacon, &root, transcript.len());
        assert!(
            verify_opening(&root, beacon, &transcript.open(idx), 2),
            "an honest FFN forward must answer the opened-matmul challenge"
        );
        // every layer of the real forward verifies
        for i in 0..transcript.len() {
            assert!(
                verify_opening(&root, beacon, &transcript.open(i), 2),
                "honest matmul {i} verifies"
            );
        }
    }

    #[test]
    fn test_output_is_the_committed_value() {
        // Soundness link: the returned y equals dequant(committed C) — committing C commits y.
        let mut seed = 7;
        let lin = ProvableLinear::from_f32(&gen(8 * 16, &mut seed), 8, 16);
        let x = gen(2 * 16, &mut seed);
        let (y, t) = prove(|| lin.forward(&x, 2));
        let op = t.open(0);
        // recompute dequant(C) from the committed operands and compare to the returned output.
        let (_xq, x_scale) = quantize_rows_i8(&x, 2, 16);
        for tok in 0..2 {
            for o in 0..8 {
                let expect =
                    (op.c.data[tok * op.c.cols + o] as f64 * x_scale[tok] * lin.w_scale[o]) as f32;
                assert!(
                    (y[tok * 8 + o] - expect).abs() < 1e-6,
                    "returned output == dequant(committed C)"
                );
            }
        }
    }

    #[test]
    fn test_no_context_no_emission() {
        // Outside prove(), a ProvableLinear still computes correctly but emits nothing.
        let mut seed = 1;
        let lin = ProvableLinear::from_f32(&gen(4 * 4, &mut seed), 4, 4);
        let y = lin.forward(&gen(4, &mut seed), 1);
        assert_eq!(y.len(), 4);
        assert!(!proving(), "no context installed");
    }
}
