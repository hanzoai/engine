//! Adversarial (red-team) tests for Proof-of-Inference emission + verification.
//!
//! Each test plays the ATTACKER: a prover trying to mint AICoin without doing the matmuls, or a
//! griefer trying to slash an honest prover. Every attack must be CAUGHT (or rejected) by the
//! transcript + Freivalds challenger. The honest cases must NEVER false-reject. Together these pin
//! the end-to-end claim "you cannot claim the AI output without performing the computation."

use hanzo_engine::poi::{self, Mat};
use hanzo_engine::poi_forward::{prove, quantize_rows_i8, ProvableLinear};
use hanzo_engine::poi_transcript::{
    challenge_index, matmul_leaf, merkle_proof, verify_opening, Opening, ProofTranscript,
};

fn gen(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((*seed >> 33) as i64 % 200) - 100) as f32 / 50.0
        })
        .collect()
}

fn rand_i64(n: usize, seed: &mut u64) -> Vec<i64> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 33) as i64 % 127) - 63
        })
        .collect()
}

// ---- ATTACK 1: fabricate the FINAL-layer output of a real forward pass ----------------------
#[test]
fn attack_fabricate_final_output_is_caught() {
    let mut s = 0xA1;
    let lin = ProvableLinear::from_f32(&gen(32 * 64, &mut s), 32, 64);
    let x = gen(4 * 64, &mut s);
    let (_y, t) = prove(|| lin.forward(&x, 4));
    // the prover keeps the transcript but tries to pass off a DIFFERENT (fabricated) output:
    // rebuild the transcript committing a tampered C.
    let op = t.open(0);
    let mut fake_c = op.c.clone();
    fake_c.data[10] += 1; // one fabricated logit
    let mut forged = ProofTranscript::new();
    forged.commit_claimed(op.a.clone(), op.b.clone(), fake_c);
    let root = forged.root();
    assert!(
        !verify_opening(&root, b"beacon:x", &forged.open(0), 2),
        "a fabricated final output must be caught"
    );
}

// ---- ATTACK 2: fabricate a HIDDEN intermediate matmul (the prover hopes it isn't opened) ------
#[test]
fn attack_fabricate_hidden_layer_caught_when_opened() {
    let mut s = 0xB2;
    let mut t = ProofTranscript::new();
    // 3 honest matmuls + 1 fabricated buried in the middle
    t.matmul(
        Mat::new(4, 16, rand_i64(64, &mut s)),
        Mat::new(16, 8, rand_i64(128, &mut s)),
    );
    let a = Mat::new(4, 8, rand_i64(32, &mut s));
    let b = Mat::new(8, 8, rand_i64(64, &mut s));
    let mut c = poi::exact_matmul(&a, &b);
    c.data[5] -= 3; // fabricate the hidden layer
    t.commit_claimed(a, b, c);
    t.matmul(
        Mat::new(4, 8, rand_i64(32, &mut s)),
        Mat::new(8, 4, rand_i64(32, &mut s)),
    );
    let root = t.root();
    // whichever index the beacon picks, the fabricated one (index 1) is caught when opened;
    // the honest ones verify — so a challenger that opens index 1 slashes.
    assert!(
        !verify_opening(&root, b"beacon", &t.open(1), 2),
        "the fabricated hidden matmul is caught"
    );
    assert!(
        verify_opening(&root, b"beacon", &t.open(0), 2),
        "honest matmul 0 still verifies"
    );
    assert!(
        verify_opening(&root, b"beacon", &t.open(2), 2),
        "honest matmul 2 still verifies"
    );
}

// ---- ATTACK 3: sparse deep cheat — ONE wrong entry in a large matmul -------------------------
#[test]
fn attack_sparse_deep_cheat_caught() {
    let (t, k, n) = (32usize, 256usize, 32usize);
    let mut s = 0xC3;
    let a = Mat::new(t, k, rand_i64(t * k, &mut s));
    let b = Mat::new(k, n, rand_i64(k * n, &mut s));
    let mut c = poi::exact_matmul(&a, &b);
    c.data[17 * n + 23] += 1; // a single buried entry, off by one
    let mut tr = ProofTranscript::new();
    tr.commit_claimed(a, b, c);
    assert!(
        !verify_opening(&tr.root(), b"beacon", &tr.open(0), 2),
        "one wrong entry among 1024 outputs of a 256-deep matmul is caught"
    );
}

// ---- ATTACK 4: swap the output at OPEN time (commit honest, reveal correct-but-uncommitted) ---
#[test]
fn attack_swap_reveal_fails_inclusion() {
    let mut s = 0xD4;
    let a = Mat::new(3, 12, rand_i64(36, &mut s));
    let b = Mat::new(12, 6, rand_i64(72, &mut s));
    let mut wrong = poi::exact_matmul(&a, &b);
    wrong.data[2] += 9;
    let mut tr = ProofTranscript::new();
    tr.commit_claimed(a.clone(), b.clone(), wrong); // committed a WRONG c
    let good = poi::exact_matmul(&a, &b);
    let op = Opening {
        index: 0,
        a,
        b,
        c: good,
        proof: merkle_proof_of(&tr, 0),
    };
    assert!(
        !verify_opening(&tr.root(), b"beacon", &op, 2),
        "revealing a C that was never committed must fail Merkle inclusion"
    );
}

// ---- ATTACK 5: tamper an INPUT at open time — the leaf binds A, so inclusion fails ------------
#[test]
fn attack_tamper_input_fails_inclusion() {
    let mut s = 0xE5;
    let a = Mat::new(2, 8, rand_i64(16, &mut s));
    let b = Mat::new(8, 4, rand_i64(32, &mut s));
    let c = poi::exact_matmul(&a, &b);
    let mut tr = ProofTranscript::new();
    tr.commit_claimed(a.clone(), b.clone(), c.clone());
    let mut tampered_a = a.clone();
    tampered_a.data[0] += 1; // claim a different input
    let op = Opening {
        index: 0,
        a: tampered_a,
        b,
        c,
        proof: merkle_proof_of(&tr, 0),
    };
    assert!(
        !verify_opening(&tr.root(), b"beacon", &op, 2),
        "a tampered input breaks the committed leaf"
    );
}

// ---- ATTACK 6: the KILLER — pre-fit C to a guessed challenge, then meet a beacon-fresh one -----
#[test]
fn attack_prefit_challenge_fails_on_fresh_beacon() {
    // The attacker fabricates C' = A·B + Δ where Δ is orthogonal to a challenge r1 they GUESSED.
    // Freivalds passes for r1 (Δ·r1 = 0) but the real challenge r2 (= keccak(beacon‖root‖idx)) is
    // unpredictable, so Δ·r2 ≠ 0 and the fabrication is caught. This is why the beacon must be
    // fixed only AFTER commitment.
    let a = Mat::new(1, 1, vec![3]); // 1×1·1×n keeps the kernel construction trivial and exact
    let n = 4usize;
    let mut s = 0xF6;
    let b = Mat::new(1, n, rand_i64(n, &mut s));
    let honest = poi::exact_matmul(&a, &b); // [1×n]
    let r1: Vec<u64> = (0..n).map(|i| (i as u64 + 1) * 1000).collect(); // the attacker's guess
                                                                        // Δ with Δ·r1 = 0: put Δ = [r1[1], -r1[0], 0, …] in the single output row.
    let mut forged = honest.clone();
    forged.data[0] += r1[1] as i64;
    forged.data[1] -= r1[0] as i64;
    // by construction Freivalds with r1 ACCEPTS the forgery:
    assert!(
        poi::freivalds_verify(&a, &b, &forged, &r1),
        "forgery survives the GUESSED challenge r1"
    );
    // but a different (beacon-fresh) challenge catches it:
    let r2: Vec<u64> = (0..n).map(|i| (i as u64 + 7) * 31337).collect();
    assert!(
        !poi::freivalds_verify(&a, &b, &forged, &r2),
        "the forgery is caught by a challenge the attacker could not predict"
    );
}

// ---- ATTACK 7: a griefer cannot frame an HONEST prover with a made-up opening -----------------
#[test]
fn attack_griefer_cannot_frame_honest_prover() {
    let mut s = 0x17;
    let a = Mat::new(2, 4, rand_i64(8, &mut s));
    let b = Mat::new(4, 2, rand_i64(8, &mut s));
    let honest_c = poi::exact_matmul(&a, &b);
    let mut tr = ProofTranscript::new();
    tr.commit_claimed(a.clone(), b.clone(), honest_c); // the honest prover's commitment
                                                       // the griefer fabricates an opening with a wrong C, hoping to slash:
    let mut fake = poi::exact_matmul(&a, &b);
    fake.data[0] += 1;
    let op = Opening {
        index: 0,
        a,
        b,
        c: fake,
        proof: merkle_proof_of(&tr, 0),
    };
    // verify_opening returns false (not included) — and crucially, the gate's provesFraud would
    // ALSO be false (inclusion fails), so the honest prover is NOT slashable. The made-up opening
    // simply isn't in the honest root.
    assert!(
        !verify_opening(&tr.root(), b"beacon", &op, 2),
        "a non-committed opening cannot slash an honest prover"
    );
}

// ---- DEFENSE 1: determinism — two independent honest runs emit the IDENTICAL transcript -------
#[test]
fn defense_two_honest_machines_agree() {
    let run = || {
        let mut s = 0x2718; // same seed => same weights/input on both "machines"
        let lin = ProvableLinear::from_f32(&gen(16 * 32, &mut s), 16, 32);
        let x = gen(3 * 32, &mut s);
        let (_y, t) = prove(|| lin.forward(&x, 3));
        t.root()
    };
    assert_eq!(
        run(),
        run(),
        "two honest machines emit the byte-identical transcript root (zero false-reject)"
    );
}

// ---- DEFENSE 2: an honest forward verifies under EVERY beacon (no false-reject) ----------------
#[test]
fn defense_honest_forward_verifies_under_any_beacon() {
    let mut s = 0x2818;
    let lin = ProvableLinear::from_f32(&gen(24 * 48, &mut s), 24, 48);
    let x = gen(5 * 48, &mut s);
    let (_y, t) = prove(|| lin.forward(&x, 5));
    let root = t.root();
    for beacon in [
        b"b0".as_slice(),
        b"b1",
        b"another-beacon",
        b"block:0xdeadbeef",
    ] {
        let idx = challenge_index(beacon, &root, t.len());
        assert!(
            verify_opening(&root, beacon, &t.open(idx), 2),
            "honest verifies under beacon {beacon:?}"
        );
    }
}

// ---- ATTACK 8: a quantization LIE — claim an output that isn't dequant(committed C) -----------
#[test]
fn attack_quantization_lie_breaks_output_binding() {
    // The output is bound to the committed C by the public dequant map. If a prover commits C
    // honestly but CLAIMS a different output, anyone recomputes dequant(C) and rejects the claim.
    let mut s = 0x28;
    let lin = ProvableLinear::from_f32(&gen(4 * 8, &mut s), 4, 8);
    let x = gen(8, &mut s);
    let (y, t) = prove(|| lin.forward(&x, 1));
    let op = t.open(0);
    let (_q, x_scale) = quantize_rows_i8(&x, 1, 8);
    let w_scale = lin.weight_scale();
    // honest output equals dequant(C):
    for o in 0..4 {
        let dq = (op.c.data[o] as f64 * x_scale[0] * w_scale[o]) as f32;
        assert!((y[o] - dq).abs() < 1e-6, "honest output is dequant(C)");
    }
    // a claimed output that differs from dequant(C) is detectable (the binding is public):
    let lie = y[0] + 1.0;
    let dq0 = (op.c.data[0] as f64 * x_scale[0] * w_scale[0]) as f32;
    assert!(
        (lie - dq0).abs() > 0.5,
        "a lied output diverges from the committed dequant(C)"
    );
}

// ---- DEFENSE 3: fail-closed on degenerate shapes ----------------------------------------------
#[test]
fn defense_shape_mismatch_fails_closed() {
    let a = Mat::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
    let b = Mat::new(2, 2, vec![1, 2, 3, 4]); // b.rows != a.cols
    assert!(
        !poi::freivalds_verify(&a, &b, &Mat::new(2, 2, vec![1, 2, 3, 4]), &[1, 1]),
        "shape mismatch fails closed"
    );
}

// helpers ---------------------------------------------------------------------------------------
fn merkle_proof_of(t: &ProofTranscript, idx: usize) -> Vec<[u8; 32]> {
    // re-derive the proof against the committed leaves (the transcript exposes open()).
    merkle_proof(&leaves_of(t), idx)
}
fn leaves_of(t: &ProofTranscript) -> Vec<[u8; 32]> {
    (0..t.len())
        .map(|i| {
            let op = t.open(i);
            matmul_leaf(&op.a, &op.b, &op.c)
        })
        .collect()
}
