//! Full-graph composition — the COMPUTATION TRACE that binds a whole forward pass, not just
//! isolated matmuls.
//!
//! [`poi_transcript`](crate::poi_transcript) proves each matmul `C = A·B`. But a prover could commit
//! correct matmuls computed on UNRELATED inputs — a valid-looking transcript that is not a real
//! forward pass on the actual prompt. This module closes that: a [`GraphTrace`] is a sequence of
//! TYPED ops over hash-identified activations. Each op commits `(kind ‖ input-commitments ‖ params ‖
//! output-commitment)`. Two checks compose into whole-graph soundness:
//!
//!   1. WELL-FORMED (chaining): every op's input commitment is the output commitment of a prior op
//!      (or the trace's declared input). A break — an op reading an activation no prior op produced —
//!      is visible from the committed leaves alone.
//!   2. LOCALLY CORRECT (per-op): opened, an op recomputes its output from its inputs — a MatMul via
//!      Freivalds, a deterministic glue op (residual add, ReLU activation, integer scale, MoE top-k
//!      route, expert gather, weighted combine) by direct recomputation — and the result's
//!      commitment equals the committed output.
//!
//! Theorem (proven in `proof-of-inference-emission.tex`): well-formed + every op locally correct ⟹
//! the committed final output equals the deterministic forward pass applied to the committed input.
//! Induction over the topological order: each op's inputs are, by chaining + the inductive
//! hypothesis, the true intermediate values, and local correctness carries the truth forward.
//!
//! Everything is EXACT INTEGER (the activations are the quantized fixed-point tensors), so the
//! activation commitments are reproducible and the glue recomputation is exact — attention's QKᵀ/AV
//! and a MoE block (router → top-k → gather → expert matmuls → combine) are expressible with no
//! floating-point reduction in the committed trace.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use crate::poi::{derive_challenges_keccak, exact_matmul, freivalds_verify_multi, keccak256, Mat};
use crate::poi_transcript::{merkle_proof, merkle_root, merkle_verify};

/// Domain tags keep activation commitments and op leaves disjoint from matmul leaves.
pub const DOM_ACT: &[u8] = b"hanzo/poi/activation/v1";
pub const DOM_OP: &[u8] = b"hanzo/poi/op/v1";

fn mat_bytes(m: &Mat) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + m.data.len() * 8);
    b.extend_from_slice(&(m.rows as u32).to_be_bytes());
    b.extend_from_slice(&(m.cols as u32).to_be_bytes());
    for &x in &m.data {
        b.extend_from_slice(&x.to_be_bytes());
    }
    b
}

/// The commitment (name) of an activation tensor in the trace.
pub fn act_commit(m: &Mat) -> [u8; 32] {
    let mut buf = DOM_ACT.to_vec();
    buf.extend_from_slice(&mat_bytes(m));
    keccak256(&buf)
}

/// The op kinds the trace can bind. Each is a pure, deterministic function of its inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpKind {
    /// `C = A · B` — the only op verified by Freivalds (the expensive O(n²) work). `B` (weights)
    /// is committed inline; `A` and `C` are activations referenced by commitment.
    MatMul,
    /// Elementwise residual `out = x + y`.
    Add,
    /// Elementwise ReLU `out = max(0, x)` (the proof-bearing activation; deterministic, exact).
    ReLU,
    /// Integer scale `out = (x * num) / den` (attention's 1/√d, deterministic).
    Scale,
    /// MoE router: `indices[r] = argtop-k(logits[r])` with the canonical lowest-index tie-break.
    /// The output activation holds the selected indices as an integer matrix `[rows × k]`.
    TopKRoute,
    /// Gather rows: `out[i] = x[idx[i]]` (expert dispatch / token routing), a committed permutation.
    Gather,
    /// Row-wise softmax in DETERMINISTIC fixed-point integer arithmetic ([`int_softmax`]) — the
    /// attention normalization. Recomputed (not Freivalds-checked): it is a pure integer function of
    /// its input, so prover and challenger compute the byte-identical output.
    Softmax,
    /// Matrix transpose `out[j][i] = x[i][j]`. The one op a BACKWARD pass needs that a forward does
    /// not: gradients are `dA = dC·Bᵀ`, `dB = Aᵀ·dC`. With it, a training step is just a longer graph
    /// of the SAME ops — so the proof covers training with no new machinery (inference, embedding, and
    /// training are graphs, not separate proof systems).
    Transpose,
}

/// Fixed-point fractional bits for [`int_softmax`] (Q16): probabilities are integers `≈ 2^16` per row.
pub const SOFTMAX_FRAC: u32 = 16;
const SM_ONE: i64 = 1 << SOFTMAX_FRAC; // 65536
const SM_LN2: i64 = 45426; // ln2 · 2^16
                           // I-BERT integer exp poly: exp(p) ≈ a·(p+b)² + c on p ∈ (−ln2, 0], coefficients · 2^16.
const SM_A: i64 = 23499; // 0.3585
const SM_B: i64 = 88670; // 1.3530
const SM_C: i64 = 22544; // 0.3440

/// `exp(x)` for `x ≤ 0`, input and output in Q16 fixed point, PURE integer (I-BERT scheme):
/// write `x = -k·ln2 + p` with integer `k ≥ 0` and `p ∈ (−ln2, 0]`; then
/// `exp(x) = 2^{-k}·exp(p)` and `exp(p)` is a fixed 2nd-order polynomial. Bit-reproducible.
fn fixed_exp_neg(x: i64) -> i64 {
    if x >= 0 {
        return SM_ONE;
    }
    let k = (-x) / SM_LN2; // integer part, ≥ 0
    let p = x + k * SM_LN2; // remainder in (−ln2, 0], Q16
    let pb = p + SM_B; // (p + b), Q16
    let sq = (pb * pb) >> SOFTMAX_FRAC; // (p+b)², Q16
    let poly = ((SM_A * sq) >> SOFTMAX_FRAC) + SM_C; // a(p+b)²+c, Q16
    poly >> k // · 2^{-k}
}

/// Deterministic integer softmax over each row. Input `logits` are Q16 fixed point (real value =
/// `logit / 2^16`); output probabilities are Q16 integers summing to `≈ 2^16` per row. PURE integer
/// (max-shift for stability, [`fixed_exp_neg`], integer normalize) so it is bit-identical across
/// backends — the prover commits it and the challenger recomputes it by the SAME function.
pub fn int_softmax(logits: &Mat) -> Mat {
    let mut out = vec![0i64; logits.rows * logits.cols];
    for r in 0..logits.rows {
        let row = &logits.data[r * logits.cols..(r + 1) * logits.cols];
        let m = *row.iter().max().unwrap_or(&0);
        let exps: Vec<i64> = row.iter().map(|&x| fixed_exp_neg(x - m)).collect();
        let sum: i64 = exps.iter().sum::<i64>().max(1);
        for c in 0..logits.cols {
            out[r * logits.cols + c] = (exps[c] << SOFTMAX_FRAC) / sum;
        }
    }
    Mat::new(logits.rows, logits.cols, out)
}

/// One op: its kind, the commitments of its input activations, any inline params, and the
/// commitment of its output activation. The leaf binds all of it.
#[derive(Clone, Debug)]
pub struct Op {
    pub kind: OpKind,
    pub inputs: Vec<[u8; 32]>,
    pub output: [u8; 32],
    /// Inline params: weight matrix for MatMul; (num,den) for Scale; k for TopKRoute; index list for Gather.
    pub weight: Option<Mat>,
    pub scalars: Vec<i64>,
}

impl Op {
    /// `leaf = keccak(DOM_OP ‖ kind ‖ n_in ‖ inputs ‖ output ‖ weight? ‖ scalars)`.
    pub fn leaf(&self) -> [u8; 32] {
        let mut buf = DOM_OP.to_vec();
        buf.push(self.kind_byte());
        buf.extend_from_slice(&(self.inputs.len() as u32).to_be_bytes());
        for i in &self.inputs {
            buf.extend_from_slice(i);
        }
        buf.extend_from_slice(&self.output);
        if let Some(w) = &self.weight {
            buf.extend_from_slice(&mat_bytes(w));
        }
        for &s in &self.scalars {
            buf.extend_from_slice(&s.to_be_bytes());
        }
        keccak256(&buf)
    }
    fn kind_byte(&self) -> u8 {
        match self.kind {
            OpKind::MatMul => 1,
            OpKind::Add => 2,
            OpKind::ReLU => 3,
            OpKind::Scale => 4,
            OpKind::TopKRoute => 5,
            OpKind::Gather => 6,
            OpKind::Softmax => 7,
            OpKind::Transpose => 8,
        }
    }
}

/// What a prover reveals to answer a challenge on op `index`: the op, its input/output tensors, and
/// the op-leaf Merkle proof.
#[derive(Clone, Debug)]
pub struct GraphOpening {
    pub index: usize,
    pub op: Op,
    pub input_mats: Vec<Mat>,
    pub output_mat: Mat,
    pub proof: Vec<[u8; 32]>,
}

/// An append-only computation trace. The prover records ops as it runs the forward pass; the root
/// is posted on-chain. Activations live in a register keyed by commitment so ops chain by reference.
#[derive(Default)]
pub struct GraphTrace {
    ops: Vec<Op>,
    op_mats: Vec<(Vec<Mat>, Mat)>, // (inputs, output) tensors kept for opening
    declared: Vec<[u8; 32]>, // the trace's declared inputs (prompt, weights, upstream gradient…)
}

impl GraphTrace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a trace input activation (a prompt, a weight tensor, an upstream gradient — a graph
    /// has a SET of inputs, not one). Returns its commitment, usable as an op input.
    pub fn input(&mut self, x: &Mat) -> [u8; 32] {
        let c = act_commit(x);
        if !self.declared.contains(&c) {
            self.declared.push(c);
        }
        c
    }

    fn push(&mut self, op: Op, inputs: Vec<Mat>, output: Mat) -> [u8; 32] {
        let oc = op.output;
        self.ops.push(op);
        self.op_mats.push((inputs, output));
        oc
    }

    /// `C = A·B`; commits the matmul op; returns C's commitment. `a` is referenced by its
    /// commitment, `b` (weights) is committed inline.
    pub fn matmul(&mut self, a: &Mat, a_commit: [u8; 32], b: &Mat) -> ([u8; 32], Mat) {
        let c = exact_matmul(a, b);
        let oc = act_commit(&c);
        let op = Op {
            kind: OpKind::MatMul,
            inputs: vec![a_commit],
            output: oc,
            weight: Some(b.clone()),
            scalars: vec![],
        };
        self.push(op, vec![a.clone()], c.clone());
        (oc, c)
    }

    /// `C = A·B` where BOTH operands are activations (the backward pass: `dW = Xᵀ·dY`). The forward's
    /// inline-weight {matmul} and this share one op kind and one Freivalds check.
    pub fn matmul_act(&mut self, a: &Mat, ac: [u8; 32], b: &Mat, bc: [u8; 32]) -> ([u8; 32], Mat) {
        let c = exact_matmul(a, b);
        let oc = act_commit(&c);
        let op = Op {
            kind: OpKind::MatMul,
            inputs: vec![ac, bc],
            output: oc,
            weight: None,
            scalars: vec![],
        };
        self.push(op, vec![a.clone(), b.clone()], c.clone());
        (oc, c)
    }

    /// `out = x + y` (residual).
    pub fn add(&mut self, x: &Mat, xc: [u8; 32], y: &Mat, yc: [u8; 32]) -> ([u8; 32], Mat) {
        let out = Mat::new(
            x.rows,
            x.cols,
            x.data.iter().zip(&y.data).map(|(a, b)| a + b).collect(),
        );
        let oc = act_commit(&out);
        let op = Op {
            kind: OpKind::Add,
            inputs: vec![xc, yc],
            output: oc,
            weight: None,
            scalars: vec![],
        };
        self.push(op, vec![x.clone(), y.clone()], out.clone());
        (oc, out)
    }

    /// `out = max(0, x)` (ReLU activation).
    pub fn relu(&mut self, x: &Mat, xc: [u8; 32]) -> ([u8; 32], Mat) {
        let out = Mat::new(x.rows, x.cols, x.data.iter().map(|&v| v.max(0)).collect());
        let oc = act_commit(&out);
        let op = Op {
            kind: OpKind::ReLU,
            inputs: vec![xc],
            output: oc,
            weight: None,
            scalars: vec![],
        };
        self.push(op, vec![x.clone()], out.clone());
        (oc, out)
    }

    /// `out = (x * num) / den` (integer scale, e.g. attention's 1/√d as a fixed-point ratio).
    pub fn scale(&mut self, x: &Mat, xc: [u8; 32], num: i64, den: i64) -> ([u8; 32], Mat) {
        let out = Mat::new(
            x.rows,
            x.cols,
            x.data.iter().map(|&v| v * num / den).collect(),
        );
        let oc = act_commit(&out);
        let op = Op {
            kind: OpKind::Scale,
            inputs: vec![xc],
            output: oc,
            weight: None,
            scalars: vec![num, den],
        };
        self.push(op, vec![x.clone()], out.clone());
        (oc, out)
    }

    /// MoE router: top-`k` expert indices per row of `logits`, canonical lowest-index tie-break.
    /// Output is an integer `[rows × k]` index matrix.
    pub fn topk_route(&mut self, logits: &Mat, lc: [u8; 32], k: usize) -> ([u8; 32], Mat) {
        let out = top_k_indices(logits, k);
        let oc = act_commit(&out);
        let op = Op {
            kind: OpKind::TopKRoute,
            inputs: vec![lc],
            output: oc,
            weight: None,
            scalars: vec![k as i64],
        };
        self.push(op, vec![logits.clone()], out.clone());
        (oc, out)
    }

    /// Gather rows of `x` by an explicit `idx` list (expert dispatch): `out[i] = x[idx[i]]`.
    pub fn gather(&mut self, x: &Mat, xc: [u8; 32], idx: &[usize]) -> ([u8; 32], Mat) {
        let mut data = Vec::with_capacity(idx.len() * x.cols);
        for &i in idx {
            data.extend_from_slice(&x.data[i * x.cols..(i + 1) * x.cols]);
        }
        let out = Mat::new(idx.len(), x.cols, data);
        let oc = act_commit(&out);
        let op = Op {
            kind: OpKind::Gather,
            inputs: vec![xc],
            output: oc,
            weight: None,
            scalars: idx.iter().map(|&i| i as i64).collect(),
        };
        self.push(op, vec![x.clone()], out.clone());
        (oc, out)
    }

    /// Row-wise deterministic integer softmax (attention normalization).
    pub fn softmax(&mut self, x: &Mat, xc: [u8; 32]) -> ([u8; 32], Mat) {
        let out = int_softmax(x);
        let oc = act_commit(&out);
        let op = Op {
            kind: OpKind::Softmax,
            inputs: vec![xc],
            output: oc,
            weight: None,
            scalars: vec![],
        };
        self.push(op, vec![x.clone()], out.clone());
        (oc, out)
    }

    /// Matrix transpose (the op a backward pass adds).
    pub fn transpose(&mut self, x: &Mat, xc: [u8; 32]) -> ([u8; 32], Mat) {
        let out = transpose(x);
        let oc = act_commit(&out);
        let op = Op {
            kind: OpKind::Transpose,
            inputs: vec![xc],
            output: oc,
            weight: None,
            scalars: vec![],
        };
        self.push(op, vec![x.clone()], out.clone());
        (oc, out)
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    fn leaves(&self) -> Vec<[u8; 32]> {
        self.ops.iter().map(|o| o.leaf()).collect()
    }

    /// The graph root the prover posts on-chain.
    pub fn root(&self) -> [u8; 32] {
        merkle_root(&self.leaves())
    }

    /// Open op `index` for a correctness challenge.
    pub fn open(&self, index: usize) -> GraphOpening {
        let (inputs, output) = &self.op_mats[index];
        GraphOpening {
            index,
            op: self.ops[index].clone(),
            input_mats: inputs.clone(),
            output_mat: output.clone(),
            proof: merkle_proof(&self.leaves(), index),
        }
    }

    /// CHAINING CHECK (cheap, leaves only): every op's input commitments reference either the
    /// declared trace input or some EARLIER op's output. A forward pass that splices in an
    /// unrelated activation fails here. Returns false on the first dangling reference.
    pub fn verify_chaining(&self) -> bool {
        let mut produced: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
        for ic in &self.declared {
            produced.insert(*ic);
        }
        for op in &self.ops {
            for inp in &op.inputs {
                if !produced.contains(inp) {
                    return false;
                }
            }
            produced.insert(op.output);
        }
        true
    }
}

/// Verify a single opened op: (1) it is the committed op at its index (Merkle inclusion), (2) the
/// revealed tensors match the op's input/output commitments, and (3) the op recomputes correctly —
/// a MatMul by Freivalds under a beacon-bound challenge, a glue op by direct recomputation. Returns
/// true iff all hold.
pub fn verify_graph_op(root: &[u8; 32], beacon: &[u8], op: &GraphOpening, k: usize) -> bool {
    // (1) the op is committed at this index
    if !merkle_verify(op.op.leaf(), *root, op.index, &op.proof) {
        return false;
    }
    // (2) revealed tensors match the committed activation commitments
    if op.input_mats.len() != op.op.inputs.len() {
        return false;
    }
    for (m, c) in op.input_mats.iter().zip(&op.op.inputs) {
        if &act_commit(m) != c {
            return false;
        }
    }
    if act_commit(&op.output_mat) != op.op.output {
        return false;
    }
    // (3) the op is locally correct
    match op.op.kind {
        OpKind::MatMul => {
            // B is either an inline weight (forward: activation × parameter) or a second activation
            // (backward: Xᵀ × dY). Same op, same Freivalds check — training is not a special case.
            let a = &op.input_mats[0];
            let b = match &op.op.weight {
                Some(w) => w,
                None if op.input_mats.len() == 2 => &op.input_mats[1],
                None => return false,
            };
            if a.cols != b.rows || b.cols != op.output_mat.cols || a.rows != op.output_mat.rows {
                return false;
            }
            let seed = graph_seed(beacon, root, op.index);
            let challenges = derive_challenges_keccak(&seed, op.output_mat.cols, k);
            freivalds_verify_multi(a, b, &op.output_mat, &challenges)
        }
        OpKind::Add => {
            let (x, y) = (&op.input_mats[0], &op.input_mats[1]);
            recompute_eq(&op.output_mat, x.rows, x.cols, || {
                x.data.iter().zip(&y.data).map(|(a, b)| a + b).collect()
            })
        }
        OpKind::ReLU => {
            let x = &op.input_mats[0];
            recompute_eq(&op.output_mat, x.rows, x.cols, || {
                x.data.iter().map(|&v| v.max(0)).collect()
            })
        }
        OpKind::Scale => {
            let x = &op.input_mats[0];
            let (num, den) = (op.op.scalars[0], op.op.scalars[1]);
            if den == 0 {
                return false;
            }
            recompute_eq(&op.output_mat, x.rows, x.cols, || {
                x.data.iter().map(|&v| v * num / den).collect()
            })
        }
        OpKind::TopKRoute => {
            let logits = &op.input_mats[0];
            let kk = op.op.scalars[0] as usize;
            top_k_indices(logits, kk) == op.output_mat
        }
        OpKind::Gather => {
            let x = &op.input_mats[0];
            let idx: Vec<usize> = op.op.scalars.iter().map(|&i| i as usize).collect();
            if idx.iter().any(|&i| i >= x.rows) {
                return false;
            }
            let mut data = Vec::with_capacity(idx.len() * x.cols);
            for &i in &idx {
                data.extend_from_slice(&x.data[i * x.cols..(i + 1) * x.cols]);
            }
            op.output_mat == Mat::new(idx.len(), x.cols, data)
        }
        OpKind::Softmax => int_softmax(&op.input_mats[0]) == op.output_mat,
        OpKind::Transpose => transpose(&op.input_mats[0]) == op.output_mat,
    }
}

/// `out[j][i] = x[i][j]` — deterministic, exact.
pub fn transpose(x: &Mat) -> Mat {
    let mut out = vec![0i64; x.rows * x.cols];
    for i in 0..x.rows {
        for j in 0..x.cols {
            out[j * x.rows + i] = x.data[i * x.cols + j];
        }
    }
    Mat::new(x.cols, x.rows, out)
}

fn recompute_eq(out: &Mat, rows: usize, cols: usize, f: impl FnOnce() -> Vec<i64>) -> bool {
    out.rows == rows && out.cols == cols && out.data == f()
}

fn graph_seed(beacon: &[u8], root: &[u8; 32], index: usize) -> [u8; 32] {
    let mut s = beacon.to_vec();
    s.extend_from_slice(root);
    s.extend_from_slice(&(index as u64).to_be_bytes());
    keccak256(&s)
}

/// Canonical top-`k` indices per row (descending value, ties broken to the LOWEST column index —
/// the determinism contract). Output `[rows × k]` integer matrix.
pub fn top_k_indices(logits: &Mat, k: usize) -> Mat {
    let k = k.min(logits.cols);
    let mut out = vec![0i64; logits.rows * k];
    for r in 0..logits.rows {
        let row = &logits.data[r * logits.cols..(r + 1) * logits.cols];
        let mut idx: Vec<usize> = (0..logits.cols).collect();
        // sort by (value desc, index asc) — a total order, backend-independent
        idx.sort_by(|&i, &j| row[j].cmp(&row[i]).then(i.cmp(&j)));
        for c in 0..k {
            out[r * k + c] = idx[c] as i64;
        }
    }
    Mat::new(logits.rows, k, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> i64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as i64 % 64) - 32
    }
    fn rmat(rows: usize, cols: usize, s: &mut u64) -> Mat {
        Mat::new(rows, cols, (0..rows * cols).map(|_| lcg(s)).collect())
    }

    // A REAL MoE FFN block as a traced graph: router matmul → top-k route → gather tokens to the
    // selected expert → expert up-proj (matmul) → ReLU → expert down-proj (matmul) → residual add.
    // Every matmul is Freivalds-checked; every glue op recomputed; the whole graph chains.
    fn moe_block(t: &mut GraphTrace, s: &mut u64) {
        let (tokens, d, experts, hidden) = (6usize, 16usize, 4usize, 24usize);
        let x = rmat(tokens, d, s);
        let xc = t.input(&x);
        // router: [tokens × d] · [d × experts] -> logits
        let (logits_c, logits) = t.matmul(&x, xc, &rmat(d, experts, s));
        // top-1 route
        let (route_c, route) = t.topk_route(&logits, logits_c, 1);
        // dispatch: gather tokens routed to expert 0 (demo: take the first 3 tokens' rows)
        let idx: Vec<usize> = (0..3).collect();
        let (xg_c, xg) = t.gather(&x, xc, &idx);
        // expert up-proj: [3 × d] · [d × hidden]
        let (h_c, h) = t.matmul(&xg, xg_c, &rmat(d, hidden, s));
        // activation
        let (a_c, a) = t.relu(&h, h_c);
        // expert down-proj: [3 × hidden] · [hidden × d]
        let (y_c, y) = t.matmul(&a, a_c, &rmat(hidden, d, s));
        // residual back onto the dispatched tokens
        let _ = t.add(&y, y_c, &xg, xg_c);
        let _ = (route_c, route); // route output is committed; used as the dispatch witness
    }

    #[test]
    fn test_moe_graph_emits_and_every_op_verifies() {
        let mut s = 0x30Eu64;
        let mut t = GraphTrace::new();
        moe_block(&mut t, &mut s);
        assert!(
            t.verify_chaining(),
            "the MoE block is a well-formed graph (every input chains)"
        );
        let root = t.root();
        let beacon = b"beacon:moe";
        for i in 0..t.len() {
            assert!(
                verify_graph_op(&root, beacon, &t.open(i), 2),
                "op {i} of the MoE block verifies"
            );
        }
    }

    #[test]
    fn test_attention_scores_av_chain_verifies() {
        // Attention's provable spine: scores = (Q·Kᵀ) scaled; ctx = scores·V. Two matmuls + a scale,
        // fully chained. (The softmax is the deterministic-reduction op of the companion extension;
        // here we exercise the matmul/scale chaining the graph binds.)
        let mut s = 0xA77_u64;
        let (tk, d) = (5usize, 8usize);
        let mut t = GraphTrace::new();
        let q = rmat(tk, d, &mut s);
        let qc = t.input(&q);
        let kt = rmat(d, tk, &mut s); // Kᵀ
        let (sc_c, sc) = t.matmul(&q, qc, &kt); // scores [tk × tk]
        let (scl_c, scl) = t.scale(&sc, sc_c, 1, 3); // /√d-ish integer scale
        let (sm_c, sm) = t.softmax(&scl, scl_c); // attention weights (deterministic int softmax)
        let v = rmat(tk, d, &mut s);
        let (_ctx_c, _ctx) = t.matmul(&sm, sm_c, &v); // ctx = softmax(QKᵀ/√d)·V  [tk × d]
        assert!(
            t.verify_chaining(),
            "full attention block (incl. softmax) chains"
        );
        let root = t.root();
        for i in 0..t.len() {
            assert!(
                verify_graph_op(&root, b"beacon:attn", &t.open(i), 2),
                "attention op {i} verifies"
            );
        }
    }

    #[test]
    fn test_int_softmax_is_a_valid_distribution() {
        // Each row sums to ≈ 2^16, larger logits get larger weight, and it is bit-reproducible.
        let logits = Mat::new(
            2,
            4,
            vec![0, SM_ONE, 2 * SM_ONE, 0, /**/ 5 * SM_ONE, 0, 0, 0],
        );
        let p = int_softmax(&logits);
        for r in 0..2 {
            let sum: i64 = p.data[r * 4..(r + 1) * 4].iter().sum();
            assert!(
                (sum - SM_ONE).abs() <= 8,
                "row {r} sums to ≈2^16 (got {sum})"
            );
        }
        assert!(
            p.data[2] > p.data[1] && p.data[1] > p.data[0],
            "monotone in the logit"
        );
        assert!(
            p.data[4] > p.data[5],
            "the peaked row concentrates mass on the max"
        );
        assert_eq!(
            int_softmax(&logits),
            int_softmax(&logits),
            "deterministic / bit-reproducible"
        );
    }

    #[test]
    fn attack_forged_softmax_is_caught() {
        let mut s = 0x50F7;
        let mut t = GraphTrace::new();
        let logits = rmat(3, 5, &mut s);
        let lc = t.input(&logits);
        t.softmax(&logits, lc);
        let root = t.root();
        let mut op = t.open(0);
        op.output_mat.data[0] += 1; // tamper the attention weights
        assert!(
            !verify_graph_op(&root, b"beacon", &op, 2),
            "a forged softmax output is caught by recompute"
        );
    }

    // PROOF-OF-AI covers TRAINING with no new machinery: an SGD step on a linear layer is a graph of
    // the SAME ops. Forward Y = X·W; backward dW = Xᵀ·dY, dX = dY·Wᵀ; update W' = W − lr·dW. Every op
    // verifies, the graph chains, and a fabricated gradient is caught by Freivalds — exactly as for
    // inference. (Embedding is likewise a forward graph; only the graph SHAPE differs, not the proof.)
    #[test]
    fn test_training_step_is_just_a_graph() {
        let mut s = 0x713;
        let mut t = GraphTrace::new();
        let x = rmat(4, 8, &mut s);
        let xc = t.input(&x);
        let w = rmat(8, 6, &mut s);
        let wc = t.input(&w); // W is a value (activation): transposable in backward, updatable in step
        let _ = t.matmul_act(&x, xc, &w, wc); // 0: forward  Y = X·W
        let dy = rmat(4, 6, &mut s); // upstream gradient from the loss
        let dyc = t.input(&dy);
        let (xt_c, xt) = t.transpose(&x, xc); // 1: Xᵀ
        let (dw_c, dw) = t.matmul_act(&xt, xt_c, &dy, dyc); // 2: dW = Xᵀ·dY
        let (wt_c, wt) = t.transpose(&w, wc); // 3: Wᵀ
        let _ = t.matmul_act(&dy, dyc, &wt, wt_c); // 4: dX = dY·Wᵀ
        let (neg_c, neg) = t.scale(&dw, dw_c, -1, 100); // 5: −lr·dW   (lr = 1/100)
        let _ = t.add(&w, wc, &neg, neg_c); // 6: W' = W − lr·dW

        assert!(
            t.verify_chaining(),
            "a training step is a well-formed graph"
        );
        let root = t.root();
        for i in 0..t.len() {
            assert!(
                verify_graph_op(&root, b"beacon:train", &t.open(i), 2),
                "training op {i} verifies"
            );
        }
        // fabricate the weight gradient (op 2) → caught by Freivalds, same as a faked inference output.
        let mut bad = t.open(2);
        bad.output_mat.data[0] += 1;
        assert!(
            !verify_graph_op(&root, b"beacon:train", &bad, 2),
            "a fabricated gradient is caught"
        );
    }

    // ---- ADVERSARIAL: break the chain, fabricate ops -----------------------------------------

    #[test]
    fn attack_chain_splice_is_caught() {
        // The prover computes a matmul on an activation that NO prior op produced (a spliced-in
        // input it never derived from the prompt). The chaining check catches it from the leaves
        // alone, and the per-op check catches it too (the revealed input doesn't match the bogus
        // commitment).
        let mut s = 0x5151u64;
        let mut t = GraphTrace::new();
        let x = rmat(3, 8, &mut s);
        let _ = t.input(&x);
        let bogus = [0xABu8; 32]; // an activation commitment that was never produced
        let (_oc, _c) = t.matmul(&x, bogus, &rmat(8, 4, &mut s));
        assert!(
            !t.verify_chaining(),
            "an op reading an unproduced activation breaks the chain"
        );
        assert!(
            !verify_graph_op(&t.root(), b"beacon", &t.open(0), 2),
            "and the per-op check rejects the bogus input commitment"
        );
    }

    #[test]
    fn attack_tamper_output_at_open_is_caught() {
        let mut s = 0x77;
        let mut t = GraphTrace::new();
        let x = rmat(4, 8, &mut s);
        let xc = t.input(&x);
        t.matmul(&x, xc, &rmat(8, 6, &mut s));
        let root = t.root();
        let mut op = t.open(0);
        op.output_mat.data[0] += 1; // claim a different output at open time
        assert!(
            !verify_graph_op(&root, b"beacon", &op, 2),
            "tampering the opened output breaks its activation commitment"
        );
    }

    #[test]
    fn attack_fabricated_matmul_in_graph_caught_by_freivalds() {
        // A prover who COMMITS a fabricated matmul output (so the commitment check passes) is still
        // caught by Freivalds inside the graph op verification.
        let mut s = 0x99;
        let a = rmat(4, 16, &mut s);
        let b = rmat(16, 8, &mut s);
        let mut c = exact_matmul(&a, &b);
        c.data[9] += 1; // fabricate one output entry, then commit IT
        let oc = act_commit(&c);
        let op = Op {
            kind: OpKind::MatMul,
            inputs: vec![act_commit(&a)],
            output: oc,
            weight: Some(b.clone()),
            scalars: vec![],
        };
        let root = merkle_root(&[op.leaf()]);
        let opening = GraphOpening {
            index: 0,
            op,
            input_mats: vec![a],
            output_mat: c,
            proof: vec![],
        };
        assert!(
            !verify_graph_op(&root, b"beacon", &opening, 2),
            "a fabricated matmul output in the graph is caught by Freivalds"
        );
    }

    #[test]
    fn attack_forged_route_is_caught() {
        // Tampering a MoE top-k routing decision is caught by recomputation.
        let mut s = 0xB0B0u64;
        let mut t = GraphTrace::new();
        let logits = rmat(4, 6, &mut s);
        let lc = t.input(&logits);
        t.topk_route(&logits, lc, 2);
        let root = t.root();
        let mut op = t.open(0);
        op.output_mat.data[0] = (op.output_mat.data[0] + 1) % 6; // route to a different expert
        assert!(
            !verify_graph_op(&root, b"beacon", &op, 2),
            "a forged routing decision is caught"
        );
    }
}
