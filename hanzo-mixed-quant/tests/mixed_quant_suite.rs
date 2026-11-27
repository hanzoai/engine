// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive integration test suite for `hanzo-mixed-quant` (`hanzoai/engine#49`, HIP-1346).
//!
//! Covers:
//! 1. Bit-exact IEEE-754 Round-to-Nearest-Even (RNE) BF16 encoding/decoding (ties, signed zero, Inf, NaN, overflow).
//! 2. Bit-exact Ternary (1.58-bit packed 4 trits/byte) packing, absmean scale, and branchless matvec equivalence.
//! 3. Bit-exact INT4 symmetric & asymmetric nibble packing (2/byte) and hoisted-scale matvec equivalence.
//! 4. Bit-exact INT8 per-group/per-channel packing, exact error bounds, and matvec equivalence.
//! 5. Normalized Walsh-Hadamard transform orthogonality `||H^T H - I||_inf < 1e-5` and invariance.
//! 6. Cayley-parameterized orthogonal matrices `R = (I - A)(I + A)^-1` orthogonality `< 1e-5` and invariance.
//! 7. SpinQuant learned Cayley optimizer reducing crest factor & W4A4 quantization MSE over unrotated baseline.
//! 8. AWQ activation-aware calibration finding optimal `alpha*` and reducing MSE on heavy-tailed corpus activations.
//! 9. Grounded corpus SHA-256 verification across all 3 versioned repository corpus files.
//! 10. 2:4 semi-structured and 50% unstructured pruning sensitivity profiling.
//! 11. Equal-memory knapsack mixed-precision planner strictly respecting byte budget and critical floors.
//! 12. Quantization-Aware Distillation (QAD) measurably reducing MSE and KL vs PTQ alone with zero-byte scale folding.
//! 13. End-to-end equal-memory Pareto win: `MixedPrecisionAwqSpinQuant` <= `UniformInt4` bytes with higher 4-domain accuracy & much lower KL.
//! 14. Negative Regime 1 (Uniform Ternary collapse) & Negative Regime 2 (unmerged runtime rotation/dequant latency regression) rejected by acceptance gate.
//! 15. Negative Regime 3 (mismatched prose-only calibration) degrading AWQ quality vs matched multi-domain calibration.
//! 16. End-to-end determinism and FIPS 180-4 SHA-256 receipt reproducibility.

use hanzo_mixed_quant::awq::{calibrate_awq, mse_f64, ActivationStats, AwqConfig};
use hanzo_mixed_quant::bench::run_full_benchmark;
use hanzo_mixed_quant::compressor::{
    apply_2_4_semi_structured_pruning, apply_unstructured_magnitude_pruning, plan_mixed_precision,
    run_qad_linear_refinement,
};
use hanzo_mixed_quant::corpus::{EvalDomain, GroundedCorpus, SplitMix64};
use hanzo_mixed_quant::error::MixedQuantError;
use hanzo_mixed_quant::formats::{
    bf16_bits_to_f32, f32_to_bf16_bits, matvec_f32, PackedBf16Matrix, PackedInt4Matrix,
    PackedInt8Matrix, PackedTernaryMatrix, QuantFormat, QuantizedLinear,
};
use hanzo_mixed_quant::matrix::{AcceptanceDecision, HybridModelSpec};
use hanzo_mixed_quant::model::{Fp32HybridModel, ProxyModelConfig, QuantRegime};
use hanzo_mixed_quant::rotation::{
    fwht_normalized_in_place, optimize_spinquant_rotation, orthogonality_error_inf,
    OrthogonalMatrix, RotationTransform,
};
use hanzo_mixed_quant::sha256::Sha256;

#[test]
fn test_01_bf16_ieee754_rne_exactness() {
    // Signed zero preserved
    assert_eq!(f32_to_bf16_bits(0.0f32), 0x0000);
    assert_eq!(f32_to_bf16_bits(-0.0f32), 0x8000);
    assert_eq!(bf16_bits_to_f32(0x8000).to_bits(), 0x8000_0000);

    // Infinities preserved
    assert_eq!(f32_to_bf16_bits(f32::INFINITY), 0x7f80);
    assert_eq!(f32_to_bf16_bits(f32::NEG_INFINITY), 0xff80);
    assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::NAN)).is_nan());

    // Exact RNE tie-to-even test:
    // 1.0f32 = 0x3f80_0000 (upper 16 bits even: 0x3f80). Halfway to next BF16 is 0x3f80_8000 -> rounds to even 0x3f80
    let tie_even = f32::from_bits(0x3f80_8000);
    assert_eq!(f32_to_bf16_bits(tie_even), 0x3f80);

    // Upper 16 bits odd: 0x3f81. Halfway to next BF16 is 0x3f81_8000 -> rounds to even 0x3f82
    let tie_odd = f32::from_bits(0x3f81_8000);
    assert_eq!(f32_to_bf16_bits(tie_odd), 0x3f82);

    // Strictly above midpoint rounds up regardless of parity
    let above_mid = f32::from_bits(0x3f80_8001);
    assert_eq!(f32_to_bf16_bits(above_mid), 0x3f81);

    // Overflow of f32::MAX rounds to +Inf in BF16
    assert_eq!(f32_to_bf16_bits(f32::MAX), 0x7f80);

    // Matrix matvec matches dequantized reference exactly
    let w = vec![1.25f32, -0.75, 2.75, 0.0, -2.5, 0.125];
    let x = vec![0.5f32, -1.0, 2.0];
    let m = PackedBf16Matrix::quantize(&w, 2, 3).unwrap();
    let mut y_packed = vec![0.0f32; 2];
    let mut y_ref = vec![0.0f32; 2];
    m.matvec(&x, &mut y_packed).unwrap();
    matvec_f32(&m.dequantize(), 2, 3, &x, &mut y_ref).unwrap();
    for (&a, &b) in y_packed.iter().zip(y_ref.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
    assert_eq!(m.storage_bytes(), 12);
}

#[test]
fn test_02_ternary_packed_trits_and_branchless_matvec() {
    let mut rng = SplitMix64::new(134_602);
    let rows = 32;
    let cols = 64;
    let weights: Vec<f32> = (0..rows * cols).map(|_| rng.next_normal()).collect();
    let x: Vec<f32> = (0..cols).map(|_| rng.next_normal()).collect();

    let packed = PackedTernaryMatrix::quantize(&weights, rows, cols, 32).unwrap();
    assert_eq!(
        packed.storage_bytes(),
        QuantFormat::Ternary.exact_storage_bytes(rows, cols, 32)
    );
    // Every unpacked trit is strictly in {-1, 0, +1}
    for r in 0..rows {
        for c in 0..cols {
            let t = packed.get_trit(r, c);
            assert!(t == -1 || t == 0 || t == 1);
        }
    }

    let deq = packed.dequantize();
    let mut y_packed = vec![0.0f32; rows];
    let mut y_deq = vec![0.0f32; rows];
    packed.matvec(&x, &mut y_packed).unwrap();
    matvec_f32(&deq, rows, cols, &x, &mut y_deq).unwrap();
    for (&a, &b) in y_packed.iter().zip(y_deq.iter()) {
        assert!((a - b).abs() < 1e-4, "ternary matvec mismatch: {a} vs {b}");
    }
}

#[test]
fn test_03_int4_sym_and_asym_nibble_packing_and_matvec() {
    let mut rng = SplitMix64::new(134_603);
    let rows = 24;
    let cols = 64;
    // Skewed asymmetric weights to exercise both symmetric and asymmetric paths
    let weights: Vec<f32> = (0..rows * cols).map(|_| rng.next_normal() + 0.6).collect();
    let x: Vec<f32> = (0..cols).map(|_| rng.next_normal()).collect();

    for asym in [false, true] {
        let m = PackedInt4Matrix::quantize(&weights, rows, cols, 32, asym).unwrap();
        let expected_fmt = if asym {
            QuantFormat::Int4Asym
        } else {
            QuantFormat::Int4Sym
        };
        assert_eq!(
            m.storage_bytes(),
            expected_fmt.exact_storage_bytes(rows, cols, 32)
        );

        let deq = m.dequantize();
        let mut y_packed = vec![0.0f32; rows];
        let mut y_deq = vec![0.0f32; rows];
        m.matvec(&x, &mut y_packed).unwrap();
        matvec_f32(&deq, rows, cols, &x, &mut y_deq).unwrap();
        for (&a, &b) in y_packed.iter().zip(y_deq.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "int4 (asym={asym}) mismatch: {a} vs {b}"
            );
        }
    }

    // Asymmetric INT4 must have strictly lower reconstruction MSE than symmetric on biased weights
    let sym = PackedInt4Matrix::quantize(&weights, rows, cols, 32, false).unwrap();
    let asym = PackedInt4Matrix::quantize(&weights, rows, cols, 32, true).unwrap();
    assert!(mse_f64(&weights, &asym.dequantize()) < mse_f64(&weights, &sym.dequantize()));
}

#[test]
fn test_04_int8_symmetric_bounds_and_matvec() {
    let mut rng = SplitMix64::new(134_604);
    let rows = 16;
    let cols = 64;
    let weights: Vec<f32> = (0..rows * cols).map(|_| rng.next_normal() * 2.0).collect();
    let x: Vec<f32> = (0..cols).map(|_| rng.next_normal()).collect();

    let m = PackedInt8Matrix::quantize(&weights, rows, cols, 64).unwrap();
    assert_eq!(
        m.storage_bytes(),
        QuantFormat::Int8.exact_storage_bytes(rows, cols, 64)
    );
    let deq = m.dequantize();

    // Verify per-row quantization error bound <= max_abs / 254 + 1e-5
    for r in 0..rows {
        let row = &weights[r * cols..(r + 1) * cols];
        let max_abs = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let bound = max_abs / 254.0 + 1e-5;
        for c in 0..cols {
            let err = (row[c] - deq[r * cols + c]).abs();
            assert!(err <= bound, "INT8 error {err} exceeded bound {bound}");
        }
    }

    let mut y_packed = vec![0.0f32; rows];
    let mut y_deq = vec![0.0f32; rows];
    m.matvec(&x, &mut y_packed).unwrap();
    matvec_f32(&deq, rows, cols, &x, &mut y_deq).unwrap();
    for (&a, &b) in y_packed.iter().zip(y_deq.iter()) {
        assert!((a - b).abs() < 1e-4);
    }
}

#[test]
fn test_05_walsh_hadamard_exact_orthogonality_and_invariance() {
    let dim = 64;
    let rot = RotationTransform::randomized_hadamard(dim, dim, 1346).unwrap();
    let dense = rot.to_dense_matrix().unwrap();
    let orth_err = orthogonality_error_inf(&dense, dim);
    assert!(
        orth_err < 1e-5,
        "Walsh-Hadamard orthogonality error too high: {orth_err}"
    );

    // Verify pure FWHT is self-inverse (H^2 = I)
    let mut rng = SplitMix64::new(134_605);
    let orig: Vec<f32> = (0..dim).map(|_| rng.next_normal()).collect();
    let mut v = orig.clone();
    fwht_normalized_in_place(&mut v).unwrap();
    fwht_normalized_in_place(&mut v).unwrap();
    for (&a, &b) in orig.iter().zip(v.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_06_cayley_orthogonal_matrix_and_exact_invariance() {
    let dim = 32;
    let mut rng = SplitMix64::new(134_606);
    let num_params = dim * (dim - 1) / 2;
    let skew: Vec<f32> = (0..num_params)
        .map(|_| rng.next_f32_signed() * 0.8)
        .collect();
    let r = OrthogonalMatrix::from_cayley_skew(dim, &skew).unwrap();

    let orth_err = r.orthogonality_error_inf();
    assert!(
        orth_err < 1e-5,
        "Cayley orthogonality ||R^T R - I||_inf = {orth_err} >= 1e-5"
    );

    // Verify exact unquantized linear invariance: x W^T == (x R)(W R)^T
    let rows = 24;
    let w: Vec<f32> = (0..rows * dim).map(|_| rng.next_normal()).collect();
    let x: Vec<f32> = (0..dim).map(|_| rng.next_normal()).collect();
    let rot = RotationTransform::LearnedOrthogonal(r);
    let w_rot = rot.rotate_matrix_rows(&w, rows, dim).unwrap();
    let mut x_rot = x.clone();
    rot.apply_in_place(&mut x_rot).unwrap();

    let mut y_orig = vec![0.0f32; rows];
    let mut y_rot = vec![0.0f32; rows];
    matvec_f32(&w, rows, dim, &x, &mut y_orig).unwrap();
    matvec_f32(&w_rot, rows, dim, &x_rot, &mut y_rot).unwrap();
    for (&a, &b) in y_orig.iter().zip(y_rot.iter()) {
        assert!((a - b).abs() < 1e-4, "invariance violated: {a} vs {b}");
    }
}

#[test]
fn test_07_spinquant_cayley_reduces_crest_and_w4a4_mse() {
    let mut rng = SplitMix64::new(134_607);
    let rows = 32;
    let cols = 32;
    let batch = 48;
    let w: Vec<f32> = (0..rows * cols).map(|_| rng.next_normal() * 0.25).collect();
    // Construct heavy-tailed activations with 2 persistent 30x outlier channels
    let mut acts = vec![0.0f32; batch * cols];
    for t in 0..batch {
        for c in 0..cols {
            let mut v = rng.next_normal() * 0.6;
            if c == 3 || c == 19 {
                v += if (t + c) % 2 == 0 { 24.0 } else { -24.0 };
            }
            acts[t * cols + c] = v;
        }
    }

    let res = optimize_spinquant_rotation(
        &w,
        rows,
        cols,
        &acts,
        batch,
        QuantFormat::Int4Asym,
        32,
        4,
        8,
        1346,
    )
    .unwrap();

    assert!(res.orthogonality_err_inf < 1e-5);
    assert!(res.invariance_err_max < 1e-3);
    assert!(
        res.post_rotation_crest < res.pre_rotation_crest * 0.6,
        "expected crest reduction: {} -> {}",
        res.pre_rotation_crest,
        res.post_rotation_crest
    );
    assert!(
        res.learned_mse < res.baseline_mse,
        "expected SpinQuant W4A4 MSE ({}) < baseline ({})",
        res.learned_mse,
        res.baseline_mse
    );
    assert!(res.learned_mse <= res.hadamard_mse + 1e-9);
}

#[test]
fn test_08_awq_grid_search_reduces_mse_on_grounded_activations() {
    let corpus = GroundedCorpus::load(None).unwrap();
    let cfg = ProxyModelConfig::default();
    let teacher = Fp32HybridModel::new(cfg.clone(), 1346);
    let seqs = corpus.matched_calibration_sequences(8, 24, cfg.vocab_size, 1346);
    let trace = teacher.collect_calibration_trace(&seqs).unwrap();

    let stats =
        ActivationStats::collect(&trace.layer_in[0], trace.num_tokens, cfg.d_model).unwrap();
    assert!(stats.outlier_ratio() > 10.0);

    let res = calibrate_awq(
        &teacher.blocks[0].seq_in_proj,
        3 * cfg.d_model,
        cfg.d_model,
        &trace.layer_in[0],
        trace.num_tokens,
        &AwqConfig {
            format: QuantFormat::Int4Asym,
            group_size: cfg.group_size,
            grid_steps: 20,
            salient_ratio: 0.0625,
            protect_salient_bf16: true,
        },
    )
    .unwrap();

    assert!(res.best_alpha > 0.0 && res.best_alpha <= 1.0);
    assert!(
        res.awq_mse < res.baseline_ptq_mse * 0.5,
        "AWQ MSE ({}) should cut baseline PTQ MSE ({}) by >50%",
        res.awq_mse,
        res.baseline_ptq_mse
    );
    assert!(!res.salient_channels.is_empty());
}

#[test]
fn test_09_grounded_corpus_files_and_sha256_vector() {
    // FIPS 180-4 standard SHA-256 vector for "abc"
    assert_eq!(
        Sha256::digest_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );

    let corpus = GroundedCorpus::load(None).unwrap();
    assert_eq!(corpus.files.len(), 3);
    assert!(corpus.total_bytes > 100_000);
    assert_eq!(corpus.combined_sha256.len(), 64);

    // Reloading yields identical SHA-256 checksum
    let corpus2 = GroundedCorpus::load(None).unwrap();
    assert_eq!(corpus.combined_sha256, corpus2.combined_sha256);
}

#[test]
fn test_10_pruning_2_4_semi_structured_and_unstructured_50() {
    let mut rng = SplitMix64::new(134_610);
    let rows = 8;
    let cols = 32;
    let weights: Vec<f32> = (0..rows * cols).map(|_| rng.next_normal()).collect();

    let p24 = apply_2_4_semi_structured_pruning(&weights, rows, cols);
    for r in 0..rows {
        let (chunks, _) = p24[r * cols..(r + 1) * cols].as_chunks::<4>();
        for chunk in chunks {
            let zeros = chunk.iter().filter(|&&v| v == 0.0).count();
            assert_eq!(zeros, 2, "each 4-weight block must have exact 2:4 sparsity");
        }
    }

    let p50 = apply_unstructured_magnitude_pruning(&weights, 0.50);
    let zero_count = p50.iter().filter(|&&v| v == 0.0).count();
    assert!(zero_count >= (rows * cols) / 2 - 2);
}

#[test]
fn test_11_equal_memory_knapsack_planner_respects_budget_and_critical_floors() {
    let corpus = GroundedCorpus::load(None).unwrap();
    let cfg = ProxyModelConfig::default();
    let teacher = Fp32HybridModel::new(cfg.clone(), 1346);
    let seqs = corpus.matched_calibration_sequences(8, 24, cfg.vocab_size, 1346);
    let trace = teacher.collect_calibration_trace(&seqs).unwrap();

    let profiles = teacher.profile_components(&trace).unwrap();
    let uniform_int4_bytes: usize = profiles.iter().map(|p| p.bytes_at(1)).sum();

    let plan = plan_mixed_precision(&profiles, uniform_int4_bytes, true).unwrap();
    assert!(
        plan.total_bytes <= uniform_int4_bytes,
        "plan exceeded equal-memory budget: {} > {}",
        plan.total_bytes,
        uniform_int4_bytes
    );
    assert!(
        plan.weighted_rel_mse < plan.uniform_int4_weighted_rel_mse,
        "mixed-precision plan must reduce weighted MSE vs uniform INT4 at equal memory"
    );

    // Verify all critical components (Embedding, LmHead, RouterGate, GdnRecurrent) are >= Int8
    for (name, kind, fmt, _) in &plan.assignments {
        if kind.is_precision_critical() {
            assert!(
                *fmt >= QuantFormat::Int8,
                "critical component {name} ({kind:?}) dropped below INT8 to {fmt:?}"
            );
        }
    }

    // Verify infeasible tiny budget returns typed error
    let err = plan_mixed_precision(&profiles, 128, true).unwrap_err();
    assert!(matches!(
        err,
        MixedQuantError::InfeasibleMemoryBudget { .. }
    ));
}

#[test]
fn test_12_qad_reduces_mse_and_kl_with_zero_byte_scale_folding() {
    let corpus = GroundedCorpus::load(None).unwrap();
    let cfg = ProxyModelConfig::default();
    let teacher = Fp32HybridModel::new(cfg.clone(), 1346);
    let seqs = corpus.matched_calibration_sequences(8, 24, cfg.vocab_size, 1346);
    let trace = teacher.collect_calibration_trace(&seqs).unwrap();

    let mut qlin = QuantizedLinear::quantize_plain(
        &teacher.blocks[0].mlp_up,
        cfg.d_ff,
        cfg.d_model,
        QuantFormat::Ternary,
        cfg.group_size,
    )
    .unwrap();
    let bytes_before = qlin.storage_bytes();

    let report = run_qad_linear_refinement(
        &mut qlin,
        &teacher.blocks[0].mlp_up,
        &trace.mlp_in[0],
        trace.num_tokens,
        8,
        0.08,
    )
    .unwrap();

    assert!(
        report.post_qad_mse < report.pre_qad_mse,
        "QAD must strictly reduce MSE vs PTQ alone: {} -> {}",
        report.pre_qad_mse,
        report.post_qad_mse
    );
    assert!(
        report.post_qad_kl <= report.pre_qad_kl + 1e-6,
        "QAD must not worsen KL: {} -> {}",
        report.pre_qad_kl,
        report.post_qad_kl
    );
    // Zero-overhead scale folding preserves exact byte count
    assert_eq!(qlin.storage_bytes(), bytes_before);
    assert!(qlin.qad_row_scales.is_none());
}

#[test]
fn test_13_end_to_end_equal_memory_pareto_win_all_four_domains() {
    let bundle = run_full_benchmark(1346, None).unwrap();

    let u4 = bundle
        .regimes
        .iter()
        .find(|r| r.regime == QuantRegime::UniformInt4)
        .unwrap();
    let mixed = bundle
        .regimes
        .iter()
        .find(|r| r.regime == QuantRegime::MixedPrecisionAwqSpinQuant)
        .unwrap();

    // Strictly equal or smaller memory footprint than Uniform INT4
    assert!(
        mixed.model_bytes <= u4.model_bytes,
        "MixedPrecision ({}) must be <= UniformInt4 ({}) bytes",
        mixed.model_bytes,
        u4.model_bytes
    );

    // Massive quality and KL win at strictly equal/smaller memory
    assert!(
        mixed.mean_accuracy_pct > u4.mean_accuracy_pct + 10.0,
        "MixedPrecision ({:.2}%) should beat UniformInt4 ({:.2}%) by >10 pp",
        mixed.mean_accuracy_pct,
        u4.mean_accuracy_pct
    );
    assert!(
        mixed.mean_kl < u4.mean_kl * 0.1,
        "MixedPrecision KL ({:.5}) should be >10x lower than UniformInt4 KL ({:.5})",
        mixed.mean_kl,
        u4.mean_kl
    );
    assert_eq!(mixed.domains.len(), 4);
    for dom in EvalDomain::ALL {
        assert!(mixed.domains.iter().any(|d| d.domain == dom));
    }

    // Accepted across all 4B/9B/27B matrix entries
    for row in bundle
        .shared_matrix
        .iter()
        .filter(|r| r.regime == QuantRegime::MixedPrecisionAwqSpinQuant)
    {
        assert_eq!(row.acceptance, AcceptanceDecision::Accepted);
    }
}

#[test]
fn test_14_negative_regimes_1_and_2_rejected_by_acceptance_gate() {
    let bundle = run_full_benchmark(1346, None).unwrap();

    // Negative Regime 1: Uniform Ternary across critical layers causes catastrophic collapse
    let u_tern = bundle
        .regimes
        .iter()
        .find(|r| r.regime == QuantRegime::UniformTernary)
        .unwrap();
    assert!(
        u_tern.mean_accuracy_pct < 20.0,
        "Uniform Ternary should suffer catastrophic collapse, got {:.2}%",
        u_tern.mean_accuracy_pct
    );
    for row in bundle
        .shared_matrix
        .iter()
        .filter(|r| r.regime == QuantRegime::UniformTernary)
    {
        assert!(matches!(
            row.acceptance,
            AcceptanceDecision::Rejected { .. }
        ));
    }

    // Negative Regime 2: Unmerged dense runtime rotation + naive scalar dequantization
    // matches MixedPrecision bytes & accuracy (microbenchmark win) yet regresses wall time -> REJECTED!
    let mixed = bundle
        .regimes
        .iter()
        .find(|r| r.regime == QuantRegime::MixedPrecisionAwqSpinQuant)
        .unwrap();
    let neg2 = bundle
        .regimes
        .iter()
        .find(|r| r.regime == QuantRegime::UnmergedRotScalarDequant)
        .unwrap();
    assert_eq!(neg2.model_bytes, mixed.model_bytes);
    assert!((neg2.mean_accuracy_pct - mixed.mean_accuracy_pct).abs() < 0.01);
    assert!(
        neg2.cpu_wall_ms > mixed.cpu_wall_ms * 3.0,
        "Neg2 wall ({:.1} ms) must significantly regress vs fused MixedPrecision ({:.1} ms)",
        neg2.cpu_wall_ms,
        mixed.cpu_wall_ms
    );
    for row in bundle
        .shared_matrix
        .iter()
        .filter(|r| r.regime == QuantRegime::UnmergedRotScalarDequant)
    {
        match &row.acceptance {
            AcceptanceDecision::Rejected { reason } => {
                assert!(
                    reason.contains("end-to-end"),
                    "expected end-to-end latency rejection reason, got: {reason}"
                );
            }
            AcceptanceDecision::Accepted => panic!("Negative Regime 2 must be REJECTED"),
        }
    }
}

#[test]
fn test_15_negative_regime_3_mismatched_calibration_degrades_awq() {
    let bundle = run_full_benchmark(1346, None).unwrap();
    let matched_awq = bundle
        .regimes
        .iter()
        .find(|r| r.regime == QuantRegime::AwqInt4)
        .unwrap();
    let mismatched_awq = bundle
        .regimes
        .iter()
        .find(|r| r.regime == QuantRegime::MismatchedProseCalAwqInt4)
        .unwrap();

    assert!(
        mismatched_awq.mean_accuracy_pct < matched_awq.mean_accuracy_pct,
        "mismatched prose-only calibration ({:.2}%) should degrade vs matched AWQ ({:.2}%)",
        mismatched_awq.mean_accuracy_pct,
        matched_awq.mean_accuracy_pct
    );
    assert!(
        mismatched_awq.mean_kl > matched_awq.mean_kl,
        "mismatched prose-only KL ({:.5}) should exceed matched AWQ KL ({:.5})",
        mismatched_awq.mean_kl,
        matched_awq.mean_kl
    );
}

#[test]
fn test_16_shared_matrix_accounting_and_receipt_determinism() {
    // Verify shared 4B/9B/27B matrix KV-head alignment with hanzo-paged-attn (2, 4, 4)
    // and honest unclamped footprint accounting (no silent clamping of infeasible budgets).
    let expected_kv_heads = [2usize, 4, 4];
    for (spec, &want_kv) in HybridModelSpec::canonical_matrix()
        .iter()
        .zip(expected_kv_heads.iter())
    {
        assert_eq!(
            spec.num_kv_heads, want_kv,
            "{} num_kv_heads must match shared HIP-1346 matrix ({want_kv})",
            spec.name
        );
        let p = spec.component_params();
        let u4_gib = p.model_size_gib(QuantRegime::UniformInt4, 128);
        let mixed_gib = p.model_size_gib(QuantRegime::MixedPrecisionAwqSpinQuant, 128);
        if p.mixed_precision_fits_int4_budget(128) {
            assert!(
                mixed_gib <= u4_gib + 1e-9,
                "{}: feasible mixed ({mixed_gib:.3} GiB) exceeded uniform INT4 ({u4_gib:.3} GiB)",
                spec.name
            );
        } else {
            assert!(
                mixed_gib > u4_gib,
                "{}: infeasible critical-layer policy must report true unclamped size > uniform INT4 ({mixed_gib:.3} vs {u4_gib:.3})",
                spec.name
            );
        }
    }

    // Two independent runs with identical seed produce bit-identical deterministic receipt SHA-256
    let b1 = run_full_benchmark(1346, None).unwrap();
    let b2 = run_full_benchmark(1346, None).unwrap();
    assert_eq!(b1.receipt_sha256, b2.receipt_sha256);
    assert_eq!(b1.corpus_sha256, b2.corpus_sha256);
    assert_eq!(b1.shared_matrix.len(), 3 * 9);
}

#[test]
fn test_17_protected_salient_channels_excluded_from_lowbit_int4asym_residual() {
    // Exact Codex reproducer: weights [0, -1, 1], input [1000, 0, 0], channel 0 protected.
    // Int4Asym grid over [-1, 1] has non-zero zero-point residual at weight 0, so masking
    // protected coordinates out of the packed low-bit dot product is mandatory.
    let weights = vec![0.0_f32, -1.0, 1.0];
    let activations = vec![1000.0_f32, 0.0, 0.0, 950.0, 0.1, -0.1];
    let cfg = AwqConfig {
        format: QuantFormat::Int4Asym,
        group_size: 4,
        grid_steps: 4,
        salient_ratio: 0.10, // ceil(3 * 0.10) = 1 -> selects channel 0
        protect_salient_bf16: true,
    };
    let res = calibrate_awq(&weights, 1, 3, &activations, 2, &cfg).unwrap();
    assert_eq!(res.salient_channels, vec![0]);

    let mut out = vec![0.0_f32; 1];
    res.linear
        .forward_vec(&[1000.0, 0.0, 0.0], &mut out)
        .unwrap();
    assert!(
        out[0].abs() < 1e-6,
        "protected channel 0 with weight 0.0 must yield 0.0 (no Int4Asym zero-point leak), got {}",
        out[0]
    );
}

#[test]
fn test_18_rotation_payload_bytes_accounted_in_quantized_linear_storage_bytes() {
    let rows = 16usize;
    let cols = 16usize;
    let weights = vec![0.25_f32; rows * cols];

    let plain =
        QuantizedLinear::quantize_plain(&weights, rows, cols, QuantFormat::Int4Asym, 16).unwrap();
    let base_bytes = plain.storage_bytes();

    // Seeded randomized Hadamard accounts for its 16-byte descriptor (not 0 and not uncounted Vec<f32>).
    let mut with_seeded = plain.clone();
    with_seeded.input_rotation =
        Some(RotationTransform::randomized_hadamard(cols, cols, 1346).unwrap());
    assert_eq!(with_seeded.storage_bytes(), base_bytes + 16);

    // Explicit sign-vector Hadamard accounts for `dim * 4` bytes.
    let mut with_explicit = plain.clone();
    with_explicit.input_rotation = Some(RotationTransform::WalshHadamard {
        dim: cols,
        block_size: cols,
        signs: vec![1.0; cols],
    });
    assert_eq!(with_explicit.storage_bytes(), base_bytes + cols * 4);

    // Dense LearnedOrthogonal accounts for `dim * dim * 4` bytes.
    let mut with_learned = plain;
    let orth = OrthogonalMatrix::identity(cols);
    with_learned.input_rotation = Some(RotationTransform::LearnedOrthogonal(orth));
    assert_eq!(with_learned.storage_bytes(), base_bytes + cols * cols * 4);
}
