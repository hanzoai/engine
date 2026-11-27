// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Integration, adversarial, edge-case, negative-control, and concurrency
//! acceptance suite for `hanzo-detprofile` (`det.zen-kai.v1`, HIP-1350).

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::thread;

use hanzo_detprofile::{
    compare_float_outputs, f32_ulp_distance, DetError, DeterministicBackend, DyadicScale,
    EligibleOp, ExecutionOutput, ExecutionProfile, ExecutionReceipt, FloatWarpSchedule, GraphInput,
    IneligibleFloatingPointOp, Int32Tensor, Int8Tensor, OpGraph, OpNode, OperatorId, OverflowMode,
    PackedInt4Block, PackedInt4Weight, RankedEntry, ReductionOrder, RoundingMode,
    ScalarCpuReference, ScalarFloatRef, SimGpuWarpBackend, TiledSimdLikeCpuBackend,
    UnorderedFloatGpuSim, QUANT_BLOCK_SIZE,
};

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_i8(&mut self) -> i8 {
        (self.next_u64() & 0xFF) as i8
    }
    fn next_q4(&mut self) -> i8 {
        ((self.next_u64() % 16) as i8) - 8
    }
    fn next_i32_range(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (i64::from(hi) - i64::from(lo) + 1) as u64;
        (i64::from(lo) + (self.next_u64() % span) as i64) as i32
    }
}

fn make_packed_int4(rng: &mut SplitMix64, k: usize, n: usize) -> PackedInt4Weight {
    let bpc = k / QUANT_BLOCK_SIZE;
    let mut blocks = Vec::with_capacity(n * bpc);
    for _ in 0..(n * bpc) {
        let scale = rng.next_i32_range(-128, 127) as i16;
        let mut ws = [0i8; QUANT_BLOCK_SIZE];
        for w in &mut ws {
            *w = rng.next_q4();
        }
        blocks.push(PackedInt4Block::pack(scale, &ws).unwrap());
    }
    PackedInt4Weight::new(k, n, blocks).unwrap()
}

fn assert_all_three_backends_bit_identical(
    profile: &ExecutionProfile,
    graph: &OpGraph,
    input: &GraphInput,
) -> ExecutionReceipt {
    let r_scalar = ScalarCpuReference.execute(profile, graph, input).unwrap();
    let r_simd = TiledSimdLikeCpuBackend
        .execute(profile, graph, input)
        .unwrap();
    let r_gpu = SimGpuWarpBackend.execute(profile, graph, input).unwrap();

    assert_eq!(
        r_scalar.output, r_simd.output,
        "Scalar vs TiledSIMD output mismatch"
    );
    assert_eq!(
        r_scalar.output, r_gpu.output,
        "Scalar vs SimGpuWarp output mismatch"
    );
    assert_eq!(
        r_scalar.output_bytes, r_simd.output_bytes,
        "Scalar vs TiledSIMD canonical bytes mismatch"
    );
    assert_eq!(
        r_scalar.output_bytes, r_gpu.output_bytes,
        "Scalar vs SimGpuWarp canonical bytes mismatch"
    );
    assert_eq!(
        r_scalar.receipt, r_simd.receipt,
        "Scalar vs TiledSIMD receipt mismatch"
    );
    assert_eq!(
        r_scalar.receipt, r_gpu.receipt,
        "Scalar vs SimGpuWarp receipt mismatch"
    );
    assert_eq!(
        r_scalar.receipt.to_bytes(),
        r_gpu.receipt.to_bytes(),
        "Wire receipt bytes mismatch"
    );

    r_scalar
        .receipt
        .verify_output(&r_scalar.output_bytes)
        .unwrap();
    let decoded = ExecutionReceipt::from_bytes(&r_gpu.receipt.to_bytes()).unwrap();
    assert_eq!(decoded, r_scalar.receipt);
    decoded.verify_output(&r_simd.output_bytes).unwrap();
    r_scalar.receipt
}

#[test]
fn all_eight_eligible_ops_bit_identical_across_three_backends() {
    let mut rng = SplitMix64::new(0x4841_4E5A_4F5F_0065);
    let profiles = [
        ExecutionProfile::v1(
            RoundingMode::RoundHalfToEven,
            OverflowMode::Saturating,
            ReductionOrder::FixedBinaryTree,
        ),
        ExecutionProfile::v1(
            RoundingMode::RoundHalfAwayFromZero,
            OverflowMode::CheckedError,
            ReductionOrder::LeftToRight,
        ),
    ];

    for profile in &profiles {
        // 1. Int8Gemm (non-multiple of tile/warp sizes to stress tail loops)
        let (m, k, n) = (11, 71, 19);
        let a = Int8Tensor::new(m, k, (0..m * k).map(|_| rng.next_i8()).collect()).unwrap();
        let w = Int8Tensor::new(k, n, (0..k * n).map(|_| rng.next_i8()).collect()).unwrap();
        let g1 = OpGraph::new(
            "int8_gemm",
            vec![OpNode::Int8Gemm {
                weights: w,
                output_scale: DyadicScale::new(3, 5).unwrap(),
                input_requant: None,
            }],
        );
        assert_all_three_backends_bit_identical(profile, &g1, &GraphInput::Int8(a));

        // 2. PackedInt4Gemm
        let (m, k, n) = (13, 96, 21);
        let a4 = Int8Tensor::new(m, k, (0..m * k).map(|_| rng.next_i8()).collect()).unwrap();
        let w4 = make_packed_int4(&mut rng, k, n);
        let g2 = OpGraph::new(
            "packed_int4_gemm",
            vec![OpNode::PackedInt4Gemm {
                weights: w4,
                output_scale: DyadicScale::new(5, 8).unwrap(),
                input_requant: None,
            }],
        );
        assert_all_three_backends_bit_identical(profile, &g2, &GraphInput::Int8(a4));

        // 3. Int32BiasAdd
        let x3 = Int32Tensor::new(
            9,
            37,
            (0..9 * 37)
                .map(|_| rng.next_i32_range(-100_000, 100_000))
                .collect(),
        )
        .unwrap();
        let bias: Vec<i32> = (0..37)
            .map(|_| rng.next_i32_range(-50_000, 50_000))
            .collect();
        let g3 = OpGraph::new("bias_add", vec![OpNode::Int32BiasAdd { bias }]);
        assert_all_three_backends_bit_identical(profile, &g3, &GraphInput::Int32(x3.clone()));

        // 4. FixedPointReLU (unclamped and clamped)
        let g4 = OpGraph::new(
            "relu_clamped",
            vec![OpNode::FixedPointReLU {
                clamp_max: Some(24_576),
            }],
        );
        assert_all_three_backends_bit_identical(profile, &g4, &GraphInput::Int32(x3.clone()));

        // 5. FixedPointSiLU_LUT
        let x5 = Int32Tensor::new(
            7,
            45,
            (0..7 * 45)
                .map(|_| rng.next_i32_range(-40_000, 40_000))
                .collect(),
        )
        .unwrap();
        let g5 = OpGraph::new("silu_lut", vec![OpNode::FixedPointSiLULut]);
        assert_all_three_backends_bit_identical(profile, &g5, &GraphInput::Int32(x5));

        // 6. FixedPointRMSNorm_Int64
        let x6 = Int32Tensor::new(
            6,
            67,
            (0..6 * 67)
                .map(|_| rng.next_i32_range(-32_768, 32_768))
                .collect(),
        )
        .unwrap();
        let gamma: Vec<i32> = (0..67).map(|_| rng.next_i32_range(2048, 8192)).collect();
        let g6 = OpGraph::new(
            "rmsnorm_int64",
            vec![OpNode::FixedPointRmsNormInt64 { gamma }],
        );
        assert_all_three_backends_bit_identical(profile, &g6, &GraphInput::Int32(x6));

        // 7. ArgMax_DeterministicTieBreak
        let x7 = Int32Tensor::new(
            5,
            73,
            (0..5 * 73).map(|_| rng.next_i32_range(-5, 5)).collect(),
        )
        .unwrap();
        let g7 = OpGraph::new("argmax", vec![OpNode::ArgMaxDeterministicTieBreak]);
        assert_all_three_backends_bit_identical(profile, &g7, &GraphInput::Int32(x7.clone()));

        // 8. TopK_DeterministicOrder
        let g8 = OpGraph::new("topk", vec![OpNode::TopKDeterministicOrder { k: 7 }]);
        assert_all_three_backends_bit_identical(profile, &g8, &GraphInput::Int32(x7));
    }

    assert_eq!(EligibleOp::ALL.len(), 8);
}

#[test]
fn zen_kai_micro_transformer_block_end_to_end_bit_identical() {
    let mut rng = SplitMix64::new(0x2026_1350_0001);
    let profile = ExecutionProfile::v1_default();

    let (batch, d_model, d_ff, vocab) = (9, 64, 96, 67);
    let input = GraphInput::Int8(
        Int8Tensor::new(
            batch,
            d_model,
            (0..batch * d_model).map(|_| rng.next_i8()).collect(),
        )
        .unwrap(),
    );
    let w_up = Int8Tensor::new(
        d_model,
        d_ff,
        (0..d_model * d_ff).map(|_| rng.next_i8()).collect(),
    )
    .unwrap();
    let b_up: Vec<i32> = (0..d_ff).map(|_| rng.next_i32_range(-4096, 4096)).collect();
    let gamma: Vec<i32> = (0..d_ff).map(|_| rng.next_i32_range(3072, 5120)).collect();
    let w_down = make_packed_int4(&mut rng, d_ff, vocab);
    let b_down: Vec<i32> = (0..vocab)
        .map(|_| rng.next_i32_range(-2048, 2048))
        .collect();

    let graph = OpGraph::new(
        "zen_kai_micro_block_topk",
        vec![
            OpNode::Int8Gemm {
                weights: w_up,
                output_scale: DyadicScale::new(1, 2).unwrap(),
                input_requant: None,
            },
            OpNode::Int32BiasAdd { bias: b_up },
            OpNode::FixedPointRmsNormInt64 { gamma },
            OpNode::FixedPointSiLULut,
            OpNode::PackedInt4Gemm {
                weights: w_down,
                output_scale: DyadicScale::new(1, 6).unwrap(),
                input_requant: Some(DyadicScale::new(1, 7).unwrap()),
            },
            OpNode::Int32BiasAdd { bias: b_down },
            OpNode::TopKDeterministicOrder { k: 5 },
        ],
    );

    let receipt = assert_all_three_backends_bit_identical(&profile, &graph, &input);
    assert_eq!(receipt.profile_id, "det.zen-kai.v1");
    assert_eq!(receipt.profile_version, 1);
}

#[test]
fn deterministic_tie_breaking_across_warp_and_tile_boundaries() {
    let profile = ExecutionProfile::v1_default();

    // Ties spanning SIMD tile boundary (index 7 vs 8) and GPU warp boundary (31 vs 32)
    let mut row = vec![-100i32; 65];
    row[7] = 42;
    row[8] = 42;
    row[31] = 42;
    row[32] = 42;
    row[64] = 42;

    // Row 2: all identical negative extreme values (i32::MIN)
    let row2 = vec![i32::MIN; 65];
    let mut data = Vec::with_capacity(130);
    data.extend_from_slice(&row);
    data.extend_from_slice(&row2);

    let input = GraphInput::Int32(Int32Tensor::new(2, 65, data).unwrap());

    // ArgMax must always select lowest index (7 for row 0, 0 for row 1)
    let g_argmax = OpGraph::new("tie_argmax", vec![OpNode::ArgMaxDeterministicTieBreak]);
    assert_all_three_backends_bit_identical(&profile, &g_argmax, &input);
    let res = SimGpuWarpBackend
        .execute(&profile, &g_argmax, &input)
        .unwrap();
    match res.output {
        ExecutionOutput::ArgMax(entries) => {
            assert_eq!(
                entries[0],
                RankedEntry {
                    index: 7,
                    value: 42
                }
            );
            assert_eq!(
                entries[1],
                RankedEntry {
                    index: 0,
                    value: i32::MIN
                }
            );
        }
        other => panic!("expected ArgMax output, got {other:?}"),
    }

    // TopK(k=5) must return exact ascending indices [7, 8, 31, 32, 64] for row 0
    // and [0, 1, 2, 3, 4] for row 1 across all three backends
    let g_topk = OpGraph::new("tie_topk", vec![OpNode::TopKDeterministicOrder { k: 5 }]);
    assert_all_three_backends_bit_identical(&profile, &g_topk, &input);
    let res_tk = SimGpuWarpBackend
        .execute(&profile, &g_topk, &input)
        .unwrap();
    match res_tk.output {
        ExecutionOutput::TopK(tk) => {
            let idx_r0: Vec<u32> = tk.entries[0..5].iter().map(|e| e.index).collect();
            let idx_r1: Vec<u32> = tk.entries[5..10].iter().map(|e| e.index).collect();
            assert_eq!(idx_r0, vec![7, 8, 31, 32, 64]);
            assert_eq!(idx_r1, vec![0, 1, 2, 3, 4]);
        }
        other => panic!("expected TopK output, got {other:?}"),
    }
}

#[test]
fn overflow_mode_saturating_vs_checked_error_and_extreme_rmsnorm() {
    let sat = ExecutionProfile::v1(
        RoundingMode::RoundHalfToEven,
        OverflowMode::Saturating,
        ReductionOrder::FixedBinaryTree,
    );
    let chk = ExecutionProfile::v1(
        RoundingMode::RoundHalfToEven,
        OverflowMode::CheckedError,
        ReductionOrder::FixedBinaryTree,
    );

    // Int32BiasAdd overflow at i32::MAX + 1 and i32::MIN - 1
    let x = GraphInput::Int32(Int32Tensor::new(1, 2, vec![i32::MAX, i32::MIN]).unwrap());
    let g_bias = OpGraph::new(
        "overflow_bias",
        vec![OpNode::Int32BiasAdd {
            bias: vec![10, -10],
        }],
    );

    // Saturating clamps identically across all 3 backends
    assert_all_three_backends_bit_identical(&sat, &g_bias, &x);
    let res_sat = ScalarCpuReference.execute(&sat, &g_bias, &x).unwrap();
    assert_eq!(
        res_sat.output,
        ExecutionOutput::Int32(Int32Tensor::new(1, 2, vec![i32::MAX, i32::MIN]).unwrap())
    );

    // CheckedError fails with ArithmeticOverflow across all 3 backends
    for backend in [
        &ScalarCpuReference as &dyn DeterministicBackend,
        &TiledSimdLikeCpuBackend,
        &SimGpuWarpBackend,
    ] {
        let err = backend.execute(&chk, &g_bias, &x).unwrap_err();
        assert!(
            matches!(err, DetError::ArithmeticOverflow { .. }),
            "expected ArithmeticOverflow on {}, got {err:?}",
            backend.name()
        );
    }

    // Extreme RMSNorm with i32::MIN inputs must not overflow u128/i128
    let x_ext = GraphInput::Int32(Int32Tensor::new(2, 35, vec![i32::MIN; 70]).unwrap());
    let g_rms = OpGraph::new(
        "extreme_rmsnorm",
        vec![OpNode::FixedPointRmsNormInt64 {
            gamma: vec![4096; 35],
        }],
    );
    assert_all_three_backends_bit_identical(&chk, &g_rms, &x_ext);
}

#[test]
fn unsupported_floating_point_exclusion_gate_rejects_all_eight_ops() {
    let profile = ExecutionProfile::v1_default();
    let dummy_input = GraphInput::Int32(Int32Tensor::new(2, 4, vec![1; 8]).unwrap());
    let backends: [&dyn DeterministicBackend; 3] = [
        &ScalarCpuReference,
        &TiledSimdLikeCpuBackend,
        &SimGpuWarpBackend,
    ];

    assert_eq!(IneligibleFloatingPointOp::ALL.len(), 8);
    for fp_op in IneligibleFloatingPointOp::ALL {
        // Standalone check
        let direct = profile.validate_operator(OperatorId::IneligibleFloatingPoint(fp_op));
        assert!(matches!(
            direct,
            Err(DetError::UnsupportedFloatingPointOp { op_name, .. }) if op_name == fp_op.canonical_name()
        ));

        // Embedded inside an otherwise valid deterministic graph
        let poisoned_graph = OpGraph::new(
            format!("poisoned_{}", fp_op.canonical_name()),
            vec![
                OpNode::FixedPointReLU { clamp_max: None },
                OpNode::UnsupportedFloatingPoint(fp_op),
            ],
        );

        for backend in backends {
            let err = backend
                .execute(&profile, &poisoned_graph, &dummy_input)
                .expect_err("floating-point op must NEVER execute");
            assert!(
                matches!(
                    err,
                    DetError::UnsupportedFloatingPointOp { op_name, .. }
                        if op_name == fp_op.canonical_name()
                ),
                "backend {} returned wrong error for {}: {err:?}",
                backend.name(),
                fp_op.canonical_name()
            );
        }
    }
}

#[test]
fn negative_control_proves_unordered_fp32_gpu_diverges_bit_for_bit() {
    let mut rng = SplitMix64::new(0xBADF_00D0_1350);
    let (m, k, n) = (8, 128, 64);
    let a: Vec<f32> = (0..m * k)
        .map(|_| (rng.next_i32_range(-10_000, 10_000) as f32) * 1.13e-3)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|_| (rng.next_i32_range(-10_000, 10_000) as f32) * 0.97e-3)
        .collect();
    let gamma: Vec<f32> = (0..n)
        .map(|_| 0.75 + (rng.next_i32_range(0, 500) as f32) * 1e-3)
        .collect();

    let scalar_f32 = ScalarFloatRef::run_gemm_norm_silu(m, k, n, &a, &b, &gamma, 1e-6);

    for schedule in [
        FloatWarpSchedule::WarpStridedTree32,
        FloatWarpSchedule::SplitKAtomicReverse,
        FloatWarpSchedule::FmaWarpTree32,
    ] {
        let gpu_f32 =
            UnorderedFloatGpuSim::new(schedule).run_gemm_norm_silu(m, k, n, &a, &b, &gamma, 1e-6);
        let report = compare_float_outputs(&scalar_f32, &gpu_f32);
        assert!(
            report.digest_diverged,
            "schedule {} unexpectedly matched scalar f32 digest",
            schedule.as_str()
        );
        assert!(report.mismatched_elements > 0);
        assert!(report.max_ulp > 0);
    }

    assert_eq!(f32_ulp_distance(1.0f32, 1.0f32), 0);
    assert_eq!(
        f32_ulp_distance(1.0f32, f32::from_bits(1.0f32.to_bits() + 1)),
        1
    );
}

#[test]
fn concurrent_16_thread_execution_produces_identical_commitments() {
    let mut rng = SplitMix64::new(0xC0CC_0016_1350);
    let profile = Arc::new(ExecutionProfile::v1_default());
    let (m, k, n) = (8, 64, 48);
    let input = Arc::new(GraphInput::Int8(
        Int8Tensor::new(m, k, (0..m * k).map(|_| rng.next_i8()).collect()).unwrap(),
    ));
    let w = make_packed_int4(&mut rng, k, n);
    let gamma: Vec<i32> = (0..n).map(|_| 4096).collect();
    let graph = Arc::new(OpGraph::new(
        "concurrent_stress",
        vec![
            OpNode::PackedInt4Gemm {
                weights: w,
                output_scale: DyadicScale::new(1, 5).unwrap(),
                input_requant: None,
            },
            OpNode::FixedPointRmsNormInt64 { gamma },
            OpNode::FixedPointSiLULut,
            OpNode::ArgMaxDeterministicTieBreak,
        ],
    ));

    let expected = ScalarCpuReference
        .execute(&profile, &graph, &input)
        .unwrap()
        .receipt;

    let handles: Vec<_> = (0..16)
        .map(|tid| {
            let p = Arc::clone(&profile);
            let g = Arc::clone(&graph);
            let inp = Arc::clone(&input);
            thread::spawn(move || {
                let res = match tid % 3 {
                    0 => ScalarCpuReference.execute(&p, &g, &inp).unwrap(),
                    1 => TiledSimdLikeCpuBackend.execute(&p, &g, &inp).unwrap(),
                    _ => SimGpuWarpBackend.execute(&p, &g, &inp).unwrap(),
                };
                res.receipt
            })
        })
        .collect();

    for h in handles {
        let got = h.join().unwrap();
        assert_eq!(got, expected);
    }
}

#[test]
fn wire_security_profile_version_and_commitment_binding() {
    let profile_even = ExecutionProfile::v1(
        RoundingMode::RoundHalfToEven,
        OverflowMode::Saturating,
        ReductionOrder::FixedBinaryTree,
    );
    let profile_away = ExecutionProfile::v1(
        RoundingMode::RoundHalfAwayFromZero,
        OverflowMode::Saturating,
        ReductionOrder::FixedBinaryTree,
    );
    let profile_l2r = ExecutionProfile::v1(
        RoundingMode::RoundHalfToEven,
        OverflowMode::Saturating,
        ReductionOrder::LeftToRight,
    );

    // Build a GEMM that hits an exact half-way tie (+1 / 2 = +0.5)
    let a = GraphInput::Int8(Int8Tensor::new(1, 2, vec![1, 0]).unwrap());
    let w = Int8Tensor::new(2, 1, vec![1, 0]).unwrap();
    let g = OpGraph::new(
        "tie_gemm",
        vec![OpNode::Int8Gemm {
            weights: w,
            output_scale: DyadicScale::new(1, 1).unwrap(),
            input_requant: None,
        }],
    );

    let r_even = ScalarCpuReference.execute(&profile_even, &g, &a).unwrap();
    let r_away = ScalarCpuReference.execute(&profile_away, &g, &a).unwrap();
    let r_l2r = ScalarCpuReference.execute(&profile_l2r, &g, &a).unwrap();

    // Rounding mode changes both output (0 vs 1) and SHA-256 commitment
    assert_eq!(
        r_even.output,
        ExecutionOutput::Int32(Int32Tensor::new(1, 1, vec![0]).unwrap())
    );
    assert_eq!(
        r_away.output,
        ExecutionOutput::Int32(Int32Tensor::new(1, 1, vec![1]).unwrap())
    );
    assert_ne!(r_even.receipt.commitment, r_away.receipt.commitment);

    // ReductionOrder produces bit-identical output bytes, yet distinct receipt commitment
    assert_eq!(r_even.output_bytes, r_l2r.output_bytes);
    assert_ne!(r_even.receipt.commitment, r_l2r.receipt.commitment);

    // Unknown profile version must be rejected
    let mut bad_profile = profile_even.clone();
    bad_profile.profile_version = 99;
    assert!(matches!(
        ScalarCpuReference.execute(&bad_profile, &g, &a),
        Err(DetError::UnknownProfileVersion { version: 99, .. })
    ));

    // Invalid dyadic shift (> 62) must be rejected
    assert!(matches!(
        DyadicScale::new(1, 63),
        Err(DetError::InvalidScale { .. })
    ));

    // Truncated, trailing, and bad-magic wire payloads must fail decoding
    let valid_wire = r_even.receipt.to_bytes();
    assert!(matches!(
        ExecutionReceipt::from_bytes(&valid_wire[..valid_wire.len() - 1]),
        Err(DetError::WireDecodeError { .. })
    ));
    let mut trailing = valid_wire.clone();
    trailing.push(0x00);
    assert!(matches!(
        ExecutionReceipt::from_bytes(&trailing),
        Err(DetError::WireDecodeError { .. })
    ));
    let mut bad_magic = valid_wire;
    bad_magic[0] ^= 0xFF;
    assert!(matches!(
        ExecutionReceipt::from_bytes(&bad_magic),
        Err(DetError::WireDecodeError { offset: 0, .. })
    ));

    // Tampering with rounding_mode in receipt without recomputing commitment must fail verify_output
    let mut tampered = r_even.receipt.clone();
    tampered.rounding_mode = RoundingMode::RoundHalfAwayFromZero;
    assert!(matches!(
        tampered.verify_output(&r_even.output_bytes),
        Err(DetError::CommitmentMismatch {
            field: "commitment"
        })
    ));
}

#[test]
fn receipts_distinguish_non_default_rmsnorm_epsilon_and_enable_replay() {
    let mut profile_eps_1 = ExecutionProfile::v1_default();
    profile_eps_1.rmsnorm_eps_q24 = 1;
    let mut profile_eps_large = ExecutionProfile::v1_default();
    profile_eps_large.rmsnorm_eps_q24 = 1 << 24;

    let graph = OpGraph::new(
        "rmsnorm_eps_binding",
        vec![OpNode::FixedPointRmsNormInt64 { gamma: vec![4096] }],
    );

    // On [4096] with gamma [4096], both epsilons produce identical output bytes,
    // yet their receipts MUST differ so verifiers know the exact numerical profile.
    let input_4096 = GraphInput::Int32(Int32Tensor::new(1, 1, vec![4096]).unwrap());
    let r1 = assert_all_three_backends_bit_identical(&profile_eps_1, &graph, &input_4096);
    let r2 = assert_all_three_backends_bit_identical(&profile_eps_large, &graph, &input_4096);
    assert_eq!(r1.output_digest, r2.output_digest);
    assert_ne!(r1.commitment, r2.commitment);
    assert_ne!(r1.to_bytes(), r2.to_bytes());
    assert_eq!(r1.rmsnorm_eps_q24, 1);
    assert_eq!(r2.rmsnorm_eps_q24, 1 << 24);
    assert_eq!(r1.profile(), profile_eps_1);
    assert_eq!(r2.profile(), profile_eps_large);

    // Replaying with the decoded receipt profiles on input [1] reproduces distinct outputs
    // deterministically across all three backends.
    let decoded_1 = ExecutionReceipt::from_bytes(&r1.to_bytes()).unwrap();
    let decoded_2 = ExecutionReceipt::from_bytes(&r2.to_bytes()).unwrap();
    let input_1 = GraphInput::Int32(Int32Tensor::new(1, 1, vec![1]).unwrap());
    let out_1 = ScalarCpuReference
        .execute(&decoded_1.profile(), &graph, &input_1)
        .unwrap();
    let out_2 = ScalarCpuReference
        .execute(&decoded_2.profile(), &graph, &input_1)
        .unwrap();
    assert_ne!(out_1.output_bytes, out_2.output_bytes);
    let _ = assert_all_three_backends_bit_identical(&decoded_1.profile(), &graph, &input_1);
    let _ = assert_all_three_backends_bit_identical(&decoded_2.profile(), &graph, &input_1);
}

#[test]
fn backends_validate_directly_constructed_malformed_tensors_without_panicking() {
    let profile = ExecutionProfile::v1_default();
    let backends: [&dyn DeterministicBackend; 3] = [
        &ScalarCpuReference,
        &TiledSimdLikeCpuBackend,
        &SimGpuWarpBackend,
    ];

    // 1. Exact reproducer: empty data with non-zero Int32Tensor dims on ArgMax.
    let argmax_graph = OpGraph::new("argmax", vec![OpNode::ArgMaxDeterministicTieBreak]);
    let bad_i32_input = GraphInput::Int32(Int32Tensor {
        rows: 1,
        cols: 1,
        data: vec![],
    });
    for backend in backends {
        assert!(
            matches!(
                backend.execute(&profile, &argmax_graph, &bad_i32_input),
                Err(DetError::InvalidDimensions { .. })
            ),
            "{} failed to reject malformed Int32Tensor",
            backend.name()
        );
    }

    // 2. Zero-dimension and overflowing Int32Tensor inputs.
    let zero_i32_input = GraphInput::Int32(Int32Tensor {
        rows: 0,
        cols: 1,
        data: vec![],
    });
    let overflow_i32_input = GraphInput::Int32(Int32Tensor {
        rows: usize::MAX,
        cols: 2,
        data: vec![],
    });
    for backend in backends {
        assert!(matches!(
            backend.execute(&profile, &argmax_graph, &zero_i32_input),
            Err(DetError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            backend.execute(&profile, &argmax_graph, &overflow_i32_input),
            Err(DetError::InvalidDimensions { .. })
        ));
    }

    // 3. Malformed Int8Tensor input on Int8Gemm.
    let valid_w = Int8Tensor::new(2, 2, vec![1, 2, 3, 4]).unwrap();
    let gemm_graph = OpGraph::new(
        "int8_gemm",
        vec![OpNode::Int8Gemm {
            weights: valid_w,
            output_scale: DyadicScale::identity(),
            input_requant: None,
        }],
    );
    let bad_i8_input = GraphInput::Int8(Int8Tensor {
        rows: 1,
        cols: 2,
        data: vec![1],
    });
    for backend in backends {
        assert!(matches!(
            backend.execute(&profile, &gemm_graph, &bad_i8_input),
            Err(DetError::InvalidDimensions { .. })
        ));
    }

    // 4. Malformed Int8Gemm weight tensor constructed with public fields.
    let bad_weight_graph = OpGraph::new(
        "bad_int8_weight",
        vec![OpNode::Int8Gemm {
            weights: Int8Tensor {
                rows: 2,
                cols: 2,
                data: vec![],
            },
            output_scale: DyadicScale::identity(),
            input_requant: None,
        }],
    );
    let good_i8_input = GraphInput::Int8(Int8Tensor::new(1, 2, vec![1, 2]).unwrap());
    for backend in backends {
        assert!(matches!(
            backend.execute(&profile, &bad_weight_graph, &good_i8_input),
            Err(DetError::InvalidDimensions { .. })
        ));
    }

    // 5. Malformed PackedInt4Weight constructed with public fields.
    let bad_int4_graph = OpGraph::new(
        "bad_int4_weight",
        vec![OpNode::PackedInt4Gemm {
            weights: PackedInt4Weight {
                k: 32,
                n: 1,
                blocks: vec![],
            },
            output_scale: DyadicScale::identity(),
            input_requant: None,
        }],
    );
    let good_i8_32 = GraphInput::Int8(Int8Tensor::new(1, 32, vec![1; 32]).unwrap());
    for backend in backends {
        assert!(matches!(
            backend.execute(&profile, &bad_int4_graph, &good_i8_32),
            Err(DetError::InvalidPackedWeight { .. })
        ));
    }
}
