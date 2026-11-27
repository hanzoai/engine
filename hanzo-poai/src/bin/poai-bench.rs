// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! HIP-1350 Proof-of-AI acceptance verification and performance benchmark harness.
//!
//! Executes the exact acceptance workload required by `hanzoai/engine#66`:
//! 1. Verifies 1,000 distinct correct packed low-bit quantized matrix products across
//!    `Int8Gemm` (`Q8_0`), `PackedInt4BlockGemm` (`Q4_0`), and `PackedTernaryI2SGemm` (`I2_S`)
//!    over realistic LLM projection shapes (Fiat-Shamir & Interactive CSPRNG modes).
//! 2. Tests 1,000 deliberately corrupted matrix products across 7 adversarial attack classes.
//! 3. Measures real `--release` wall-clock prover matmul time, commitment overhead (online vs cold),
//!    challenge generation overhead, CPU Freivalds verifier time, speedup ratios, receipt size,
//!    and the explicit tiny-GEMV negative regime (`1x64x64`, `1x512x512`).
//! 4. Emits raw machine-readable JSON receipts to `docs/research/hip-1350-poai-freivalds/results.json`
//!    (or `--out <path>`).

use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use hanzo_poai::{
    commit_activation_matrix, commit_output_matrix, commit_weight_matrix, derive_challenge_vector,
    derive_transcript_seed, prove_matrix_product_with_cached_b_commit, prover_matmul,
    verify_freivalds_algebraic_only, verify_matrix_product_with_cached_b_commit,
    verify_matrix_product_with_cached_b_commit_and_policy, verify_matrix_product_with_mode,
    verify_matrix_product_with_policy, ActivationMatrix, ChallengeMode, MatrixProofReceipt,
    OperatorKind, PackedQuantFormat, PackedWeightMatrix, PoaiError, VerificationPolicy,
    VerificationProfile, DEFAULT_FREIVALDS_ROUNDS, MAX_SAFE_SIGNED, MODULUS_P,
    PROOF_SCOPE_DISCLAIMER,
};

/// Deterministic SplitMix64 PRNG (zero external dependencies) for reproducible benchmarks.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_seed_32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..4 {
            out[i * 8..i * 8 + 8].copy_from_slice(&self.next_u64().to_le_bytes());
        }
        out
    }
}

fn generate_activation(rng: &mut SplitMix64, m: usize, k: usize) -> ActivationMatrix {
    let mut data = Vec::with_capacity(m * k);
    for _ in 0..(m * k) {
        // Non-trivial signed i8 activations spanning [-120, 120]
        let v = ((rng.next_u64() % 241) as i32 - 120) as i8;
        data.push(if v == 0 { 1 } else { v });
    }
    ActivationMatrix::new(m, k, data).expect("valid activation dims")
}

fn generate_weights(
    rng: &mut SplitMix64,
    format: PackedQuantFormat,
    k: usize,
    n: usize,
) -> PackedWeightMatrix {
    let blocks = k / 32;
    let mut scales = Vec::with_capacity(n * blocks);
    for _ in 0..(n * blocks) {
        // Non-zero fixed-point i16 scales in [-64, 64] \ {0}
        let mut s = ((rng.next_u64() % 129) as i32 - 64) as i16;
        if s == 0 {
            s = 1;
        }
        scales.push(s);
    }

    let mut q = Vec::with_capacity(k * n);
    match format {
        PackedQuantFormat::Int8Q8_0 => {
            for _ in 0..(k * n) {
                q.push(((rng.next_u64() % 251) as i32 - 125) as i8);
            }
        }
        PackedQuantFormat::PackedInt4Q4_0 => {
            for _ in 0..(k * n) {
                q.push(((rng.next_u64() % 16) as i32 - 8) as i8);
            }
        }
        PackedQuantFormat::PackedTernaryI2S => {
            for _ in 0..(k * n) {
                q.push((rng.next_u64() % 3) as i8 - 1);
            }
        }
    }
    PackedWeightMatrix::pack_from_column_major(format, k, n, &q, &scales)
        .expect("valid packed weights")
}

#[derive(Debug, Clone, Copy)]
enum CorruptionKind {
    SingleElemPm1,
    SingleBitFlip,
    ZeroedOutputBlock,
    RowOrColSwap,
    StaleCachedRow,
    QuantizedScaleTamper,
    ForgedCommitmentOrDigest,
}

impl CorruptionKind {
    const ALL: [Self; 7] = [
        Self::SingleElemPm1,
        Self::SingleBitFlip,
        Self::ZeroedOutputBlock,
        Self::RowOrColSwap,
        Self::StaleCachedRow,
        Self::QuantizedScaleTamper,
        Self::ForgedCommitmentOrDigest,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::SingleElemPm1 => "single_elem_pm1",
            Self::SingleBitFlip => "single_bit_flip",
            Self::ZeroedOutputBlock => "zeroed_output_block",
            Self::RowOrColSwap => "row_or_col_swap",
            Self::StaleCachedRow => "stale_cached_row",
            Self::QuantizedScaleTamper => "quantized_scale_tamper",
            Self::ForgedCommitmentOrDigest => "forged_commitment_or_digest",
        }
    }
}

struct CorruptionStats {
    name: &'static str,
    tested: usize,
    detected: usize,
    false_accepts: usize,
}

struct ShapeTimingRow {
    format: &'static str,
    operator: &'static str,
    m: usize,
    k: usize,
    n: usize,
    rounds: usize,
    iterations: usize,
    regime: &'static str,
    prover_matmul_p50_us: f64,
    prover_matmul_p95_us: f64,
    commit_a_c_p50_us: f64,
    commit_b_p50_us: f64,
    challenge_gen_p50_us: f64,
    verifier_freivalds_p50_us: f64,
    verifier_freivalds_p95_us: f64,
    verifier_online_total_p50_us: f64,
    verifier_cold_total_p50_us: f64,
    speedup_algebraic: f64,
    speedup_online: f64,
    speedup_cold: f64,
    receipt_bytes: usize,
    soundness_error_bound: f64,
    soundness_bits: f64,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[allow(clippy::too_many_arguments)]
fn benchmark_shape(
    profile: &VerificationProfile,
    format: PackedQuantFormat,
    m: usize,
    k: usize,
    n: usize,
    rounds: usize,
    iterations: usize,
    regime: &'static str,
    rng: &mut SplitMix64,
) -> ShapeTimingRow {
    let a = generate_activation(rng, m, k);
    let b = generate_weights(rng, format, k, n);
    let c = prover_matmul(&a, &b).expect("prover_matmul succeeds");
    let commit_b = commit_weight_matrix(&b);
    let receipt = prove_matrix_product_with_cached_b_commit(
        profile,
        &a,
        &b,
        &c,
        rounds,
        ChallengeMode::FiatShamir,
        commit_b,
    )
    .expect("prove succeeds");
    let receipt_bytes = receipt.to_bytes().len();

    let policy = VerificationPolicy::with_min_rounds(rounds);

    // Warmup
    let _ = black_box(prover_matmul(&a, &b).unwrap());
    verify_matrix_product_with_cached_b_commit_and_policy(
        profile, &a, &b, &c, &receipt, commit_b, policy,
    )
    .expect("warmup verify succeeds");

    let mut prover_us = Vec::with_capacity(iterations);
    let mut commit_ac_us = Vec::with_capacity(iterations);
    let mut commit_b_us = Vec::with_capacity(iterations);
    let mut challenge_us = Vec::with_capacity(iterations);
    let mut freivalds_us = Vec::with_capacity(iterations);
    let mut online_us = Vec::with_capacity(iterations);
    let mut cold_us = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let t0 = Instant::now();
        let c_out = black_box(prover_matmul(&a, &b).unwrap());
        prover_us.push(t0.elapsed().as_secs_f64() * 1e6);
        drop(c_out);

        let t1 = Instant::now();
        let ca = black_box(commit_activation_matrix(&a));
        let cc = black_box(commit_output_matrix(&c));
        commit_ac_us.push(t1.elapsed().as_secs_f64() * 1e6);

        let t2 = Instant::now();
        let cb = black_box(commit_weight_matrix(&b));
        commit_b_us.push(t2.elapsed().as_secs_f64() * 1e6);

        let t3 = Instant::now();
        let seed = derive_transcript_seed(
            profile,
            format.operator_kind(),
            m,
            k,
            n,
            rounds,
            &ca,
            &cb,
            &cc,
            ChallengeMode::FiatShamir,
        );
        let mut challenges = Vec::with_capacity(rounds);
        for r_idx in 0..rounds {
            challenges.push(derive_challenge_vector(&seed, r_idx as u32, n));
        }
        black_box(&challenges);
        challenge_us.push(t3.elapsed().as_secs_f64() * 1e6);

        let t4 = Instant::now();
        verify_freivalds_algebraic_only(&a, &b, &c, &challenges).unwrap();
        freivalds_us.push(t4.elapsed().as_secs_f64() * 1e6);

        let t5 = Instant::now();
        verify_matrix_product_with_cached_b_commit_and_policy(
            profile, &a, &b, &c, &receipt, commit_b, policy,
        )
        .unwrap();
        online_us.push(t5.elapsed().as_secs_f64() * 1e6);

        let t6 = Instant::now();
        verify_matrix_product_with_policy(profile, &a, &b, &c, &receipt, policy).unwrap();
        cold_us.push(t6.elapsed().as_secs_f64() * 1e6);
    }

    for v in [
        &mut prover_us,
        &mut commit_ac_us,
        &mut commit_b_us,
        &mut challenge_us,
        &mut freivalds_us,
        &mut online_us,
        &mut cold_us,
    ] {
        v.sort_by(|x, y| x.total_cmp(y));
    }

    let prover_p50 = percentile(&prover_us, 0.50);
    let prover_p95 = percentile(&prover_us, 0.95);
    let commit_ac_p50 = percentile(&commit_ac_us, 0.50);
    let commit_b_p50 = percentile(&commit_b_us, 0.50);
    let challenge_p50 = percentile(&challenge_us, 0.50);
    let freiv_p50 = percentile(&freivalds_us, 0.50);
    let freiv_p95 = percentile(&freivalds_us, 0.95);
    let online_p50 = percentile(&online_us, 0.50);
    let cold_p50 = percentile(&cold_us, 0.50);

    ShapeTimingRow {
        format: format.as_str(),
        operator: format.operator_kind().as_str(),
        m,
        k,
        n,
        rounds,
        iterations,
        regime,
        prover_matmul_p50_us: prover_p50,
        prover_matmul_p95_us: prover_p95,
        commit_a_c_p50_us: commit_ac_p50,
        commit_b_p50_us: commit_b_p50,
        challenge_gen_p50_us: challenge_p50,
        verifier_freivalds_p50_us: freiv_p50,
        verifier_freivalds_p95_us: freiv_p95,
        verifier_online_total_p50_us: online_p50,
        verifier_cold_total_p50_us: cold_p50,
        speedup_algebraic: prover_p50 / freiv_p50.max(1e-9),
        speedup_online: prover_p50 / online_p50.max(1e-9),
        speedup_cold: prover_p50 / cold_p50.max(1e-9),
        receipt_bytes,
        soundness_error_bound: profile.soundness_error_upper_bound(rounds),
        soundness_bits: profile.soundness_bits(rounds),
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut out_path = PathBuf::from("docs/research/hip-1350-poai-freivalds/results.json");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--out" && i + 1 < args.len() {
            out_path = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    let profile = VerificationProfile::v1();
    let mut rng = SplitMix64::new(0x4841_4E5A_4F5F_1350);

    // -------------------------------------------------------------------------
    // 0. Verify ineligible operator rejection gate
    // -------------------------------------------------------------------------
    let ineligible_ops = [
        OperatorKind::Fp16Gemm,
        OperatorKind::Bf16Gemm,
        OperatorKind::Fp32Gemm,
        OperatorKind::Fp8Gemm,
        OperatorKind::FlashAttention,
        OperatorKind::Normalization,
        OperatorKind::Sampling,
        OperatorKind::StateUpdate,
    ];
    let mut ineligible_rejected = 0usize;
    for op in ineligible_ops {
        if matches!(
            profile.ensure_eligible(op),
            Err(PoaiError::IneligibleOperator { .. })
        ) {
            ineligible_rejected += 1;
        }
    }
    assert_eq!(ineligible_rejected, ineligible_ops.len());

    // -------------------------------------------------------------------------
    // 1. Acceptance Test Part A: 1,000 distinct correct packed matrix products
    // -------------------------------------------------------------------------
    println!("=== HIP-1350 Proof-of-AI Acceptance & Benchmark Harness ===");
    println!(
        "Profile: {} | Field: GF(2^61 - 1) [p = {}]",
        profile.version, MODULUS_P
    );
    println!("Safe signed range: |acc| <= {}", MAX_SAFE_SIGNED);
    println!(
        "Default rounds: k = {} (soundness error < {:.3e}, {:.2} bits)",
        DEFAULT_FREIVALDS_ROUNDS,
        profile.soundness_error_upper_bound(DEFAULT_FREIVALDS_ROUNDS),
        profile.soundness_bits(DEFAULT_FREIVALDS_ROUNDS),
    );
    println!("Scope: {}\n", PROOF_SCOPE_DISCLAIMER);

    let acceptance_shapes: &[(usize, usize, usize)] = &[
        (1, 512, 512),
        (4, 512, 512),
        (8, 512, 512),
        (16, 512, 512),
        (4, 1024, 1024),
        (32, 1024, 1024),
        (64, 2048, 2048),
    ];
    let formats = [
        PackedQuantFormat::Int8Q8_0,
        PackedQuantFormat::PackedInt4Q4_0,
        PackedQuantFormat::PackedTernaryI2S,
    ];

    let t_accept_start = Instant::now();
    let total_correct = 1_000usize;
    let mut correct_passed = 0usize;
    let mut correct_by_format = [0usize; 3];
    let mut fiat_shamir_count = 0usize;
    let mut interactive_count = 0usize;

    // Distribute 1,000 products across all 3 formats and all 7 shapes (including large 32x1024x1024
    // and 64x2048x2048 projections while keeping total runtime fast and deterministic).
    for idx in 0..total_correct {
        let fmt_idx = idx % 3;
        let format = formats[fmt_idx];
        let (m, k, n) = if idx < 9 {
            // 9 large 64x2048x2048 products (3 per format)
            acceptance_shapes[6]
        } else if idx < 39 {
            // 30 medium-large 32x1024x1024 products (10 per format)
            acceptance_shapes[5]
        } else if idx < 129 {
            // 90 4x1024x1024 products
            acceptance_shapes[4]
        } else {
            // Remaining 871 across 1x512x512, 4x512x512, 8x512x512, 16x512x512
            acceptance_shapes[idx % 4]
        };

        let a = generate_activation(&mut rng, m, k);
        let b = generate_weights(&mut rng, format, k, n);
        let c = prover_matmul(&a, &b).expect("honest prover_matmul succeeds");
        let commit_b = commit_weight_matrix(&b);

        let mode = if idx % 2 == 0 {
            fiat_shamir_count += 1;
            ChallengeMode::FiatShamir
        } else {
            interactive_count += 1;
            ChallengeMode::InteractiveSeed(rng.next_seed_32())
        };

        let receipt = prove_matrix_product_with_cached_b_commit(
            &profile,
            &a,
            &b,
            &c,
            DEFAULT_FREIVALDS_ROUNDS,
            mode,
            commit_b,
        )
        .expect("honest prove succeeds");

        // Round-trip through canonical wire bytes to verify serialization integrity on all 1,000 cases.
        let wire = receipt.to_bytes();
        let decoded = MatrixProofReceipt::from_bytes(&wire).expect("receipt decodes");

        let ver = if idx % 2 == 0 {
            verify_matrix_product_with_cached_b_commit(
                &profile,
                &a,
                &b,
                &c,
                &decoded,
                commit_b,
                Some(mode),
            )
        } else {
            verify_matrix_product_with_mode(&profile, &a, &b, &c, &decoded, mode)
        };

        if ver.is_ok() {
            correct_passed += 1;
            correct_by_format[fmt_idx] += 1;
        }
    }
    let accept_correct_sec = t_accept_start.elapsed().as_secs_f64();
    println!(
        "[ACCEPTANCE 1/2] Correct products verified: {}/{} ({:.2}%) in {:.2}s (Int8={}, Int4={}, Ternary={}, FS={}, Interactive={})",
        correct_passed,
        total_correct,
        (correct_passed as f64 / total_correct as f64) * 100.0,
        accept_correct_sec,
        correct_by_format[0],
        correct_by_format[1],
        correct_by_format[2],
        fiat_shamir_count,
        interactive_count
    );
    assert_eq!(correct_passed, total_correct);

    // -------------------------------------------------------------------------
    // 2. Acceptance Test Part B: 1,000 deliberately corrupted matrix products
    // -------------------------------------------------------------------------
    let t_corrupt_start = Instant::now();
    let total_corrupted = 1_000usize;
    let mut corrupt_stats: Vec<CorruptionStats> = CorruptionKind::ALL
        .iter()
        .map(|k| CorruptionStats {
            name: k.name(),
            tested: 0,
            detected: 0,
            false_accepts: 0,
        })
        .collect();

    let mut total_detected = 0usize;
    let mut total_false_accepts = 0usize;
    let mut prev_row_cache: Vec<i64> = Vec::new();

    for idx in 0..total_corrupted {
        let kind_idx = idx % CorruptionKind::ALL.len();
        let kind = CorruptionKind::ALL[kind_idx];
        let format = formats[idx % 3];
        let (m, k, n) = match idx % 5 {
            0 => (1, 512, 512),
            1 => (4, 512, 512),
            2 => (8, 512, 512),
            3 => (16, 512, 512),
            _ => (4, 1024, 1024),
        };

        let a = generate_activation(&mut rng, m, k);
        let b = generate_weights(&mut rng, format, k, n);
        let c_honest = prover_matmul(&a, &b).expect("honest matmul succeeds");
        let commit_b = commit_weight_matrix(&b);
        let mode = if idx % 2 == 0 {
            ChallengeMode::FiatShamir
        } else {
            ChallengeMode::InteractiveSeed(rng.next_seed_32())
        };

        let mut c_bad = c_honest.clone();
        let receipt_for_check = match kind {
            CorruptionKind::SingleElemPm1 => {
                let pos = (rng.next_u64() as usize) % c_bad.data().len();
                let delta = if idx % 2 == 0 { 1i64 } else { -1i64 };
                c_bad.data_mut()[pos] += delta;
                prove_matrix_product_with_cached_b_commit(
                    &profile,
                    &a,
                    &b,
                    &c_bad,
                    DEFAULT_FREIVALDS_ROUNDS,
                    mode,
                    commit_b,
                )
                .unwrap()
            }
            CorruptionKind::SingleBitFlip => {
                let pos = (rng.next_u64() as usize) % c_bad.data().len();
                let bit = (rng.next_u64() % 20) as u32;
                c_bad.data_mut()[pos] ^= 1i64 << bit;
                prove_matrix_product_with_cached_b_commit(
                    &profile,
                    &a,
                    &b,
                    &c_bad,
                    DEFAULT_FREIVALDS_ROUNDS,
                    mode,
                    commit_b,
                )
                .unwrap()
            }
            CorruptionKind::ZeroedOutputBlock => {
                let block_len = 32.min(c_bad.data().len());
                let max_start = c_bad.data().len() - block_len;
                let start = if max_start == 0 {
                    0
                } else {
                    (rng.next_u64() as usize) % max_start
                };
                let mut changed = false;
                for v in &mut c_bad.data_mut()[start..start + block_len] {
                    if *v != 0 {
                        changed = true;
                    }
                    *v = 0;
                }
                if !changed {
                    c_bad.data_mut()[start] = 1;
                }
                prove_matrix_product_with_cached_b_commit(
                    &profile,
                    &a,
                    &b,
                    &c_bad,
                    DEFAULT_FREIVALDS_ROUNDS,
                    mode,
                    commit_b,
                )
                .unwrap()
            }
            CorruptionKind::RowOrColSwap => {
                if m >= 2 {
                    let r0 = 0usize;
                    let r1 = 1 + ((rng.next_u64() as usize) % (m - 1));
                    let row0 = c_bad.row(r0).to_vec();
                    let row1 = c_bad.row(r1).to_vec();
                    c_bad.data_mut()[r0 * n..r0 * n + n].copy_from_slice(&row1);
                    c_bad.data_mut()[r1 * n..r1 * n + n].copy_from_slice(&row0);
                    if c_bad == c_honest {
                        c_bad.data_mut()[0] += 1;
                    }
                } else {
                    c_bad.data_mut().swap(0, n - 1);
                    if c_bad == c_honest {
                        c_bad.data_mut()[0] += 1;
                    }
                }
                prove_matrix_product_with_cached_b_commit(
                    &profile,
                    &a,
                    &b,
                    &c_bad,
                    DEFAULT_FREIVALDS_ROUNDS,
                    mode,
                    commit_b,
                )
                .unwrap()
            }
            CorruptionKind::StaleCachedRow => {
                let target_row = (rng.next_u64() as usize) % m;
                if prev_row_cache.len() == n {
                    c_bad.data_mut()[target_row * n..target_row * n + n]
                        .copy_from_slice(&prev_row_cache);
                } else {
                    for v in &mut c_bad.data_mut()[target_row * n..target_row * n + n] {
                        *v = v.wrapping_add(1337);
                    }
                }
                if c_bad == c_honest {
                    c_bad.data_mut()[target_row * n] += 1;
                }
                prove_matrix_product_with_cached_b_commit(
                    &profile,
                    &a,
                    &b,
                    &c_bad,
                    DEFAULT_FREIVALDS_ROUNDS,
                    mode,
                    commit_b,
                )
                .unwrap()
            }
            CorruptionKind::QuantizedScaleTamper => {
                let col = (rng.next_u64() as usize) % n;
                let blk = (rng.next_u64() as usize) % b.blocks_per_col();
                let b_tampered = b
                    .with_tampered_block_scale(col, blk, 127)
                    .expect("valid block");
                let c_tampered = prover_matmul(&a, &b_tampered).unwrap();
                c_bad = if c_tampered != c_honest {
                    c_tampered
                } else {
                    let b_tampered2 = b.with_tampered_block_scale(col, blk, -127).unwrap();
                    let c2 = prover_matmul(&a, &b_tampered2).unwrap();
                    if c2 != c_honest {
                        c2
                    } else {
                        let mut fallback = c_honest.clone();
                        fallback.data_mut()[col] += 1;
                        fallback
                    }
                };
                // Prover tries to pass tampered C against the HONEST committed B
                prove_matrix_product_with_cached_b_commit(
                    &profile,
                    &a,
                    &b,
                    &c_bad,
                    DEFAULT_FREIVALDS_ROUNDS,
                    mode,
                    commit_b,
                )
                .unwrap()
            }
            CorruptionKind::ForgedCommitmentOrDigest => {
                let mut r = prove_matrix_product_with_cached_b_commit(
                    &profile,
                    &a,
                    &b,
                    &c_honest,
                    DEFAULT_FREIVALDS_ROUNDS,
                    mode,
                    commit_b,
                )
                .unwrap();
                if idx % 2 == 0 {
                    r.commit_c[0] ^= 0x80;
                } else {
                    r.round_digests[0][31] ^= 0x01;
                }
                r
            }
        };

        prev_row_cache = c_honest.row(0).to_vec();

        let ver = verify_matrix_product_with_cached_b_commit(
            &profile,
            &a,
            &b,
            &c_bad,
            &receipt_for_check,
            commit_b,
            Some(mode),
        );

        corrupt_stats[kind_idx].tested += 1;
        if ver.is_err() {
            corrupt_stats[kind_idx].detected += 1;
            total_detected += 1;
        } else {
            corrupt_stats[kind_idx].false_accepts += 1;
            total_false_accepts += 1;
        }
    }
    let accept_corrupt_sec = t_corrupt_start.elapsed().as_secs_f64();
    println!(
        "[ACCEPTANCE 2/2] Corrupted products detected: {}/{} ({:.2}%, false_accepts={}) in {:.2}s",
        total_detected,
        total_corrupted,
        (total_detected as f64 / total_corrupted as f64) * 100.0,
        total_false_accepts,
        accept_corrupt_sec
    );
    for st in &corrupt_stats {
        println!(
            "  - {:28}: {}/{} detected (false_accepts={})",
            st.name, st.detected, st.tested, st.false_accepts
        );
    }
    assert_eq!(total_detected, total_corrupted);
    assert_eq!(total_false_accepts, 0);

    // -------------------------------------------------------------------------
    // 3. Performance, Overhead & Negative-Regime Benchmark Matrix
    // -------------------------------------------------------------------------
    println!("\n=== Performance & Verifier Speedup Matrix (k = 2 rounds, --release) ===");
    let bench_shapes: &[(usize, usize, usize, usize, &str)] = &[
        (1, 64, 64, 200, "negative_regime_tiny_gemv"),
        (1, 512, 512, 40, "negative_regime_small_gemv"),
        (4, 1024, 1024, 15, "small_batch_projection"),
        (32, 1024, 1024, 9, "prefill_medium_projection"),
        (64, 2048, 2048, 5, "llm_ffn_qkv_projection"),
        (128, 2048, 2048, 3, "llm_large_prefill_projection"),
    ];

    let mut rows = Vec::new();
    for &format in &formats {
        for &(m, k, n, iters, regime) in bench_shapes {
            let row = benchmark_shape(
                &profile,
                format,
                m,
                k,
                n,
                DEFAULT_FREIVALDS_ROUNDS,
                iters,
                regime,
                &mut rng,
            );
            println!(
                "{:16} {:>3}x{:<4}x{:<4} | prover={:>9.1}us | freiv={:>8.1}us ({:>5.2}x) | online={:>8.1}us ({:>5.2}x) | cold={:>9.1}us ({:>5.2}x) | receipt={}B",
                row.format,
                row.m,
                row.k,
                row.n,
                row.prover_matmul_p50_us,
                row.verifier_freivalds_p50_us,
                row.speedup_algebraic,
                row.verifier_online_total_p50_us,
                row.speedup_online,
                row.verifier_cold_total_p50_us,
                row.speedup_cold,
                row.receipt_bytes
            );
            rows.push(row);
        }
    }

    // -------------------------------------------------------------------------
    // 4. Soundness Round Sweep (k = 1, 2, 4) on 64x2048x2048 PackedInt4Q4_0
    // -------------------------------------------------------------------------
    println!("\n=== Round-Count Sensitivity (64x2048x2048, PackedInt4Q4_0) ===");
    let mut round_sweep = Vec::new();
    for &rounds in &[1usize, 2, 4] {
        let row = benchmark_shape(
            &profile,
            PackedQuantFormat::PackedInt4Q4_0,
            64,
            2048,
            2048,
            rounds,
            5,
            "round_sweep_64x2048x2048",
            &mut rng,
        );
        println!(
            "k={} | soundness < {:.3e} ({:.2} bits) | prover={:.1}us | freiv={:.1}us ({:.2}x) | online={:.1}us ({:.2}x) | receipt={}B",
            rounds,
            row.soundness_error_bound,
            row.soundness_bits,
            row.prover_matmul_p50_us,
            row.verifier_freivalds_p50_us,
            row.speedup_algebraic,
            row.verifier_online_total_p50_us,
            row.speedup_online,
            row.receipt_bytes
        );
        round_sweep.push(row);
    }

    // -------------------------------------------------------------------------
    // 5. Write raw JSON receipt file
    // -------------------------------------------------------------------------
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).expect("create results output directory");
    }

    let mut json = String::with_capacity(32 * 1024);
    json.push_str("{\n");
    json.push_str(&format!(
        "  \"profile_version\": \"{}\",\n",
        json_escape(profile.version)
    ));
    json.push_str(&format!("  \"field_modulus_p\": {},\n", MODULUS_P));
    json.push_str(&format!("  \"max_safe_signed\": {},\n", MAX_SAFE_SIGNED));
    json.push_str(&format!(
        "  \"default_rounds\": {},\n",
        DEFAULT_FREIVALDS_ROUNDS
    ));
    json.push_str(&format!(
        "  \"default_soundness_error_upper_bound\": {:.6e},\n",
        profile.soundness_error_upper_bound(DEFAULT_FREIVALDS_ROUNDS)
    ));
    json.push_str(&format!(
        "  \"default_soundness_bits\": {:.4},\n",
        profile.soundness_bits(DEFAULT_FREIVALDS_ROUNDS)
    ));
    json.push_str(&format!(
        "  \"scope_disclaimer\": \"{}\",\n",
        json_escape(PROOF_SCOPE_DISCLAIMER)
    ));
    json.push_str(&format!(
        "  \"ineligible_operators_rejected\": {},\n",
        ineligible_rejected
    ));
    json.push_str("  \"acceptance_correct\": {\n");
    json.push_str(&format!("    \"total\": {},\n", total_correct));
    json.push_str(&format!("    \"passed\": {},\n", correct_passed));
    json.push_str(&format!(
        "    \"pass_rate_pct\": {:.4},\n",
        (correct_passed as f64 / total_correct as f64) * 100.0
    ));
    json.push_str(&format!("    \"int8_q8_0\": {},\n", correct_by_format[0]));
    json.push_str(&format!(
        "    \"packed_int4_q4_0\": {},\n",
        correct_by_format[1]
    ));
    json.push_str(&format!(
        "    \"packed_ternary_i2_s\": {},\n",
        correct_by_format[2]
    ));
    json.push_str(&format!(
        "    \"fiat_shamir_cases\": {},\n",
        fiat_shamir_count
    ));
    json.push_str(&format!(
        "    \"interactive_cases\": {},\n",
        interactive_count
    ));
    json.push_str(&format!("    \"elapsed_sec\": {:.4}\n", accept_correct_sec));
    json.push_str("  },\n");

    json.push_str("  \"acceptance_corrupted\": {\n");
    json.push_str(&format!("    \"total\": {},\n", total_corrupted));
    json.push_str(&format!("    \"detected\": {},\n", total_detected));
    json.push_str(&format!(
        "    \"false_accepts\": {},\n",
        total_false_accepts
    ));
    json.push_str(&format!(
        "    \"detection_rate_pct\": {:.4},\n",
        (total_detected as f64 / total_corrupted as f64) * 100.0
    ));
    json.push_str(&format!(
        "    \"elapsed_sec\": {:.4},\n",
        accept_corrupt_sec
    ));
    json.push_str("    \"by_attack_class\": [\n");
    for (idx, st) in corrupt_stats.iter().enumerate() {
        let comma = if idx + 1 == corrupt_stats.len() {
            ""
        } else {
            ","
        };
        json.push_str(&format!(
            "      {{\"attack\": \"{}\", \"tested\": {}, \"detected\": {}, \"false_accepts\": {}}}{}\n",
            st.name, st.tested, st.detected, st.false_accepts, comma
        ));
    }
    json.push_str("    ]\n");
    json.push_str("  },\n");

    let push_rows = |json: &mut String,
                     key: &str,
                     items: &[ShapeTimingRow],
                     trailing_comma: bool| {
        json.push_str(&format!("  \"{}\": [\n", key));
        for (idx, r) in items.iter().enumerate() {
            let comma = if idx + 1 == items.len() { "" } else { "," };
            json.push_str(&format!(
                "    {{\"format\": \"{}\", \"operator\": \"{}\", \"m\": {}, \"k\": {}, \"n\": {}, \"rounds\": {}, \"iterations\": {}, \"regime\": \"{}\", \"prover_matmul_p50_us\": {:.3}, \"prover_matmul_p95_us\": {:.3}, \"commit_a_c_p50_us\": {:.3}, \"commit_b_p50_us\": {:.3}, \"challenge_gen_p50_us\": {:.3}, \"verifier_freivalds_p50_us\": {:.3}, \"verifier_freivalds_p95_us\": {:.3}, \"verifier_online_total_p50_us\": {:.3}, \"verifier_cold_total_p50_us\": {:.3}, \"speedup_algebraic\": {:.4}, \"speedup_online\": {:.4}, \"speedup_cold\": {:.4}, \"receipt_bytes\": {}, \"soundness_error_bound\": {:.6e}, \"soundness_bits\": {:.4}}}{}\n",
                r.format,
                r.operator,
                r.m,
                r.k,
                r.n,
                r.rounds,
                r.iterations,
                r.regime,
                r.prover_matmul_p50_us,
                r.prover_matmul_p95_us,
                r.commit_a_c_p50_us,
                r.commit_b_p50_us,
                r.challenge_gen_p50_us,
                r.verifier_freivalds_p50_us,
                r.verifier_freivalds_p95_us,
                r.verifier_online_total_p50_us,
                r.verifier_cold_total_p50_us,
                r.speedup_algebraic,
                r.speedup_online,
                r.speedup_cold,
                r.receipt_bytes,
                r.soundness_error_bound,
                r.soundness_bits,
                comma
            ));
        }
        if trailing_comma {
            json.push_str("  ],\n");
        } else {
            json.push_str("  ]\n");
        }
    };

    push_rows(&mut json, "benchmarks", &rows, true);
    push_rows(&mut json, "round_sensitivity", &round_sweep, false);
    json.push_str("}\n");

    fs::write(&out_path, json).expect("write results.json");
    println!("\nWrote raw benchmark receipts to {}", out_path.display());
}
