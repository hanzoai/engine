// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reproducible HIP-1350 acceptance and benchmark harness (`detprofile-bench`).
//!
//! Anchors a shared deterministic corpus of 512 Zen/Kai operator graphs and
//! quantized micro-transformer blocks to the repository's real public
//! `calibration_data/calibration_datav3.txt` corpus, verifies 100% bit-identical
//! outputs & SHA-256 commitments across 3 independent backends, verifies 8/8
//! floating-point exclusion rejections, runs the explicit FP32 negative control
//! (`UnorderedFloatGpuSim` vs `ScalarFloatRef`), measures throughput & latency
//! overhead, and writes raw JSON receipts to
//! `docs/research/hip-1350-deterministic-profiles/results.json`.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hanzo_detprofile::{
    compare_float_outputs, digest_to_hex, DetError, DeterministicBackend, DyadicScale, EligibleOp,
    ExecutionProfile, FloatWarpSchedule, GraphInput, IneligibleFloatingPointOp, Int32Tensor,
    Int8Tensor, OpGraph, OpNode, OverflowMode, PackedInt4Block, PackedInt4Weight, ReductionOrder,
    RoundingMode, ScalarCpuReference, ScalarFloatRef, Sha256, SimGpuWarpBackend,
    TiledSimdLikeCpuBackend, UnorderedFloatGpuSim, PROFILE_ID_V1, PROFILE_VERSION_V1,
    QUANT_BLOCK_SIZE,
};

const CORPUS_WORKLOADS: usize = 512;
const NEGATIVE_CONTROL_WORKLOADS: usize = 128;

/// Deterministic byte stream seeded directly by SHA-256 of real corpus slices
/// from `calibration_data/calibration_datav3.txt`.
struct CorpusPrng {
    state: u64,
}

impl CorpusPrng {
    fn from_corpus_window(corpus: &[u8], workload_idx: usize) -> Self {
        let stride = (corpus.len() / CORPUS_WORKLOADS.max(1)).max(64);
        let start = (workload_idx * stride) % corpus.len();
        let end = (start + 256).min(corpus.len());
        let mut h = Sha256::new();
        h.update(b"hanzo.detprofile.hip1350.corpus_window\0");
        h.update(&(workload_idx as u64).to_le_bytes());
        h.update(&corpus[start..end]);
        let digest = h.finalize();
        let seed = u64::from_le_bytes([
            digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
        ]);
        Self {
            state: seed ^ 0x4841_4E5A_4F5F_1350,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
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

fn find_repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(parent) = manifest_dir.parent() {
        if parent
            .join("calibration_data/calibration_datav3.txt")
            .exists()
        {
            return parent.to_path_buf();
        }
    }
    PathBuf::from(".")
}

fn make_packed_int4(rng: &mut CorpusPrng, k: usize, n: usize) -> PackedInt4Weight {
    let bpc = k / QUANT_BLOCK_SIZE;
    let mut blocks = Vec::with_capacity(n * bpc);
    for _ in 0..(n * bpc) {
        let scale = rng.next_i32_range(-64, 64) as i16;
        let mut ws = [0i8; QUANT_BLOCK_SIZE];
        for w in &mut ws {
            *w = rng.next_q4();
        }
        blocks.push(PackedInt4Block::pack(scale, &ws).expect("valid q4"));
    }
    PackedInt4Weight::new(k, n, blocks).expect("valid packed int4")
}

/// Build workload `idx in 0..512` covering all 8 eligible operators individually
/// AND full multi-layer quantized Zen/Kai micro-transformer blocks across both
/// rounding modes (`RoundHalfToEven`, `RoundHalfAwayFromZero`) and both overflow
/// modes (`Saturating`, `CheckedError`).
fn build_corpus_workload(
    corpus: &[u8],
    idx: usize,
) -> (&'static str, ExecutionProfile, OpGraph, GraphInput) {
    let mut rng = CorpusPrng::from_corpus_window(corpus, idx);
    let rounding = if idx.is_multiple_of(2) {
        RoundingMode::RoundHalfToEven
    } else {
        RoundingMode::RoundHalfAwayFromZero
    };
    let overflow = if (idx / 2).is_multiple_of(2) {
        OverflowMode::Saturating
    } else {
        OverflowMode::CheckedError
    };
    let reduction = if (idx / 4).is_multiple_of(2) {
        ReductionOrder::FixedBinaryTree
    } else {
        ReductionOrder::LeftToRight
    };
    let profile = ExecutionProfile::v1(rounding, overflow, reduction);

    match idx % 10 {
        0 => {
            let (m, k, n) = (1 + (idx % 12), 32 + (idx % 4) * 16 + 5, 17 + (idx % 19));
            let a = Int8Tensor::new(m, k, (0..m * k).map(|_| rng.next_i8()).collect()).unwrap();
            let w = Int8Tensor::new(k, n, (0..k * n).map(|_| rng.next_i8()).collect()).unwrap();
            let g = OpGraph::new(
                format!("corpus_{idx:03}_int8_gemm"),
                vec![OpNode::Int8Gemm {
                    weights: w,
                    output_scale: DyadicScale::new(3, 5).unwrap(),
                    input_requant: None,
                }],
            );
            ("Int8Gemm", profile, g, GraphInput::Int8(a))
        }
        1 => {
            let (m, k, n) = (1 + (idx % 10), 64 + ((idx % 3) * 32), 19 + (idx % 17));
            let a = Int8Tensor::new(m, k, (0..m * k).map(|_| rng.next_i8()).collect()).unwrap();
            let w = make_packed_int4(&mut rng, k, n);
            let g = OpGraph::new(
                format!("corpus_{idx:03}_packed_int4_gemm"),
                vec![OpNode::PackedInt4Gemm {
                    weights: w,
                    output_scale: DyadicScale::new(5, 7).unwrap(),
                    input_requant: None,
                }],
            );
            ("PackedInt4Gemm", profile, g, GraphInput::Int8(a))
        }
        2 => {
            let (m, n) = (4 + (idx % 9), 33 + (idx % 31));
            let x = Int32Tensor::new(
                m,
                n,
                (0..m * n)
                    .map(|_| rng.next_i32_range(-250_000, 250_000))
                    .collect(),
            )
            .unwrap();
            let bias: Vec<i32> = (0..n)
                .map(|_| rng.next_i32_range(-50_000, 50_000))
                .collect();
            let g = OpGraph::new(
                format!("corpus_{idx:03}_bias_add"),
                vec![OpNode::Int32BiasAdd { bias }],
            );
            ("Int32BiasAdd", profile, g, GraphInput::Int32(x))
        }
        3 => {
            let (m, n) = (4 + (idx % 9), 35 + (idx % 29));
            let x = Int32Tensor::new(
                m,
                n,
                (0..m * n)
                    .map(|_| rng.next_i32_range(-32_768, 32_768))
                    .collect(),
            )
            .unwrap();
            let clamp_max = if idx % 20 == 3 { Some(24_576) } else { None };
            let g = OpGraph::new(
                format!("corpus_{idx:03}_relu"),
                vec![OpNode::FixedPointReLU { clamp_max }],
            );
            ("FixedPointReLU", profile, g, GraphInput::Int32(x))
        }
        4 => {
            let (m, n) = (4 + (idx % 8), 41 + (idx % 27));
            let x = Int32Tensor::new(
                m,
                n,
                (0..m * n)
                    .map(|_| rng.next_i32_range(-40_000, 40_000))
                    .collect(),
            )
            .unwrap();
            let g = OpGraph::new(
                format!("corpus_{idx:03}_silu_lut"),
                vec![OpNode::FixedPointSiLULut],
            );
            ("FixedPointSiLU_LUT", profile, g, GraphInput::Int32(x))
        }
        5 => {
            let (m, d) = (3 + (idx % 8), 37 + (idx % 39));
            let x = Int32Tensor::new(
                m,
                d,
                (0..m * d)
                    .map(|_| rng.next_i32_range(-32_768, 32_768))
                    .collect(),
            )
            .unwrap();
            let gamma: Vec<i32> = (0..d).map(|_| rng.next_i32_range(2048, 6144)).collect();
            let g = OpGraph::new(
                format!("corpus_{idx:03}_rmsnorm"),
                vec![OpNode::FixedPointRmsNormInt64 { gamma }],
            );
            ("FixedPointRMSNorm_Int64", profile, g, GraphInput::Int32(x))
        }
        6 => {
            let (m, v) = (4 + (idx % 7), 48 + (idx % 49));
            // Dense small range intentionally exercises deterministic tie-breaking across warp lanes
            let x = Int32Tensor::new(
                m,
                v,
                (0..m * v).map(|_| rng.next_i32_range(-6, 6)).collect(),
            )
            .unwrap();
            let g = OpGraph::new(
                format!("corpus_{idx:03}_argmax"),
                vec![OpNode::ArgMaxDeterministicTieBreak],
            );
            (
                "ArgMax_DeterministicTieBreak",
                profile,
                g,
                GraphInput::Int32(x),
            )
        }
        7 => {
            let (m, v) = (4 + (idx % 7), 48 + (idx % 49));
            let k = 1 + (idx % 12);
            let x = Int32Tensor::new(
                m,
                v,
                (0..m * v).map(|_| rng.next_i32_range(-8, 8)).collect(),
            )
            .unwrap();
            let g = OpGraph::new(
                format!("corpus_{idx:03}_topk"),
                vec![OpNode::TopKDeterministicOrder { k }],
            );
            ("TopK_DeterministicOrder", profile, g, GraphInput::Int32(x))
        }
        8 => {
            // Full Zen/Kai Quantized Micro-Transformer Block (SiLU + ArgMax)
            let (m, d, ff, v) = (4 + (idx % 6), 64, 64, 49 + (idx % 23));
            let a = Int8Tensor::new(m, d, (0..m * d).map(|_| rng.next_i8()).collect()).unwrap();
            let w_up =
                Int8Tensor::new(d, ff, (0..d * ff).map(|_| rng.next_i8()).collect()).unwrap();
            let b_up: Vec<i32> = (0..ff).map(|_| rng.next_i32_range(-4096, 4096)).collect();
            let gamma: Vec<i32> = (0..ff).map(|_| rng.next_i32_range(3072, 5120)).collect();
            let w_down = make_packed_int4(&mut rng, ff, v);
            let b_down: Vec<i32> = (0..v).map(|_| rng.next_i32_range(-2048, 2048)).collect();
            let g = OpGraph::new(
                format!("corpus_{idx:03}_zen_micro_transformer_argmax"),
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
                    OpNode::ArgMaxDeterministicTieBreak,
                ],
            );
            (
                "ZenKaiMicroTransformer_ArgMax",
                profile,
                g,
                GraphInput::Int8(a),
            )
        }
        _ => {
            // Full Zen/Kai Quantized Micro-Transformer Block (ReLU + TopK)
            let (m, d, ff, v) = (4 + (idx % 6), 64, 64, 49 + (idx % 23));
            let a = Int8Tensor::new(m, d, (0..m * d).map(|_| rng.next_i8()).collect()).unwrap();
            let w_up = make_packed_int4(&mut rng, d, ff);
            let b_up: Vec<i32> = (0..ff).map(|_| rng.next_i32_range(-4096, 4096)).collect();
            let gamma: Vec<i32> = (0..ff).map(|_| rng.next_i32_range(3072, 5120)).collect();
            let w_down =
                Int8Tensor::new(ff, v, (0..ff * v).map(|_| rng.next_i8()).collect()).unwrap();
            let b_down: Vec<i32> = (0..v).map(|_| rng.next_i32_range(-2048, 2048)).collect();
            let g = OpGraph::new(
                format!("corpus_{idx:03}_zen_micro_transformer_topk"),
                vec![
                    OpNode::PackedInt4Gemm {
                        weights: w_up,
                        output_scale: DyadicScale::new(1, 5).unwrap(),
                        input_requant: None,
                    },
                    OpNode::Int32BiasAdd { bias: b_up },
                    OpNode::FixedPointRmsNormInt64 { gamma },
                    OpNode::FixedPointReLU {
                        clamp_max: Some(24_576),
                    },
                    OpNode::Int8Gemm {
                        weights: w_down,
                        output_scale: DyadicScale::new(1, 3).unwrap(),
                        input_requant: Some(DyadicScale::new(1, 8).unwrap()),
                    },
                    OpNode::Int32BiasAdd { bias: b_down },
                    OpNode::TopKDeterministicOrder { k: 5 },
                ],
            );
            (
                "ZenKaiMicroTransformer_TopK",
                profile,
                g,
                GraphInput::Int8(a),
            )
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn measure_us<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    (percentile(&samples, 0.50), percentile(&samples, 0.95))
}

fn read_cpu_model() -> String {
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                if let Some((_, val)) = rest.split_once(':') {
                    return val.trim().to_string();
                }
            }
        }
    }
    std::env::consts::ARCH.to_string()
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

struct NegControlSummary {
    comparison: String,
    workloads: usize,
    diverged_digests: usize,
    divergence_rate_pct: f64,
    total_elements: usize,
    mismatched_elements: usize,
    element_mismatch_rate_pct: f64,
    max_ulp: u32,
    mean_ulp: f64,
}

struct PerfRow {
    workload: String,
    shape: String,
    mac_ops: u64,
    scalar_det_p50_us: f64,
    scalar_det_p95_us: f64,
    tiled_simd_det_p50_us: f64,
    tiled_simd_det_p95_us: f64,
    sim_gpu_warp_det_p50_us: f64,
    sim_gpu_warp_det_p95_us: f64,
    unordered_f32_scalar_p50_us: f64,
    unordered_f32_gpu_p50_us: f64,
    tiled_det_vs_f32_ratio: f64,
    tiled_det_mops_per_sec: f64,
}

fn main() {
    let repo_root = find_repo_root();
    let corpus_path = repo_root.join("calibration_data/calibration_datav3.txt");
    let corpus_bytes = fs::read(&corpus_path).unwrap_or_else(|e| {
        panic!(
            "failed to read public calibration corpus {}: {e}",
            corpus_path.display()
        )
    });
    let corpus_sha256 = digest_to_hex(&Sha256::digest(&corpus_bytes));

    println!("=== hanzo-detprofile HIP-1350 Acceptance & Benchmark Harness ===");
    println!("Profile: {PROFILE_ID_V1} (v{PROFILE_VERSION_V1})");
    println!(
        "Corpus anchor: calibration_data/calibration_datav3.txt ({} bytes, sha256={corpus_sha256})",
        corpus_bytes.len()
    );

    // ------------------------------------------------------------------------
    // 1. Shared deterministic corpus acceptance (512 workloads x 3 backends)
    // ------------------------------------------------------------------------
    let t_corpus = Instant::now();
    let mut category_counts: Vec<(&'static str, usize, usize)> = Vec::new();
    let mut bit_identical_count = 0usize;
    let mut commitment_identical_count = 0usize;
    let mut receipt_wire_bytes_sample = 0usize;
    let mut sample_commitments: Vec<(usize, &'static str, String)> = Vec::new();

    for idx in 0..CORPUS_WORKLOADS {
        let (cat, profile, graph, input) = build_corpus_workload(&corpus_bytes, idx);
        let r_scalar = ScalarCpuReference
            .execute(&profile, &graph, &input)
            .expect("ScalarCpuReference failed");
        let r_simd = TiledSimdLikeCpuBackend
            .execute(&profile, &graph, &input)
            .expect("TiledSimdLikeCpuBackend failed");
        let r_gpu = SimGpuWarpBackend
            .execute(&profile, &graph, &input)
            .expect("SimGpuWarpBackend failed");

        let bytes_eq = r_scalar.output_bytes == r_simd.output_bytes
            && r_scalar.output_bytes == r_gpu.output_bytes;
        let commit_eq = r_scalar.receipt == r_simd.receipt && r_scalar.receipt == r_gpu.receipt;

        assert!(bytes_eq, "BIT DIVERGENCE on workload {idx} ({cat})!");
        assert!(
            commit_eq,
            "COMMITMENT DIVERGENCE on workload {idx} ({cat})!"
        );
        r_scalar
            .receipt
            .verify_output(&r_scalar.output_bytes)
            .expect("receipt self-verify failed");

        bit_identical_count += 1;
        commitment_identical_count += 1;
        if receipt_wire_bytes_sample == 0 {
            receipt_wire_bytes_sample = r_scalar.receipt.to_bytes().len();
        }
        if idx < 10 {
            sample_commitments.push((idx, cat, digest_to_hex(&r_scalar.receipt.commitment)));
        }

        if let Some(entry) = category_counts.iter_mut().find(|(c, _, _)| *c == cat) {
            entry.1 += 1;
            entry.2 += 1;
        } else {
            category_counts.push((cat, 1, 1));
        }
    }
    let corpus_wall_s = t_corpus.elapsed().as_secs_f64();
    println!(
        "[1/4] Corpus Determinism: {bit_identical_count}/{CORPUS_WORKLOADS} bit-identical bytes & SHA-256 commitments across 3 independent backends ({corpus_wall_s:.2} s)"
    );

    // ------------------------------------------------------------------------
    // 2. Unsupported floating-point operator exclusion gate (8 ops x 3 backends)
    // ------------------------------------------------------------------------
    let profile_v1 = ExecutionProfile::v1_default();
    let dummy_in = GraphInput::Int32(Int32Tensor::new(2, 8, vec![42; 16]).unwrap());
    let backends: [&dyn DeterministicBackend; 3] = [
        &ScalarCpuReference,
        &TiledSimdLikeCpuBackend,
        &SimGpuWarpBackend,
    ];
    let mut fp_gate_ops_rejected = 0usize;
    let mut fp_gate_invocations_rejected = 0usize;
    let mut fp_gate_records: Vec<(&'static str, &'static str, bool)> = Vec::new();

    for fp_op in IneligibleFloatingPointOp::ALL {
        let graph = OpGraph::new(
            format!("fp_reject_{}", fp_op.canonical_name()),
            vec![OpNode::UnsupportedFloatingPoint(fp_op)],
        );
        for backend in backends {
            match backend.execute(&profile_v1, &graph, &dummy_in) {
                Err(DetError::UnsupportedFloatingPointOp { op_name, .. })
                    if op_name == fp_op.canonical_name() =>
                {
                    fp_gate_invocations_rejected += 1;
                }
                other => {
                    panic!(
                        "FP exclusion gate FAILED for {} on {}: {other:?}",
                        fp_op.canonical_name(),
                        backend.name()
                    );
                }
            }
        }
        fp_gate_ops_rejected += 1;
        fp_gate_records.push((fp_op.canonical_name(), fp_op.rejection_reason(), true));
    }
    println!(
        "[2/4] Floating-Point Exclusion Gate: {fp_gate_ops_rejected}/{} ops rejected ({fp_gate_invocations_rejected}/24 backend invocations)",
        IneligibleFloatingPointOp::ALL.len()
    );

    // ------------------------------------------------------------------------
    // 3. Explicit Negative Control: UnorderedFloatGpuSim vs ScalarFloatRef
    // ------------------------------------------------------------------------
    let schedules = [
        FloatWarpSchedule::WarpStridedTree32,
        FloatWarpSchedule::SplitKAtomicReverse,
        FloatWarpSchedule::FmaWarpTree32,
    ];
    let mut neg_summaries: Vec<NegControlSummary> = Vec::new();

    for &sched in &schedules {
        let mut diverged = 0usize;
        let mut total_elems = 0usize;
        let mut mism_elems = 0usize;
        let mut max_ulp = 0u32;
        let mut sum_ulp = 0.0f64;

        for idx in 0..NEGATIVE_CONTROL_WORKLOADS {
            let mut rng = CorpusPrng::from_corpus_window(&corpus_bytes, idx);
            let (m, k, n) = (4, 128, 64);
            let a: Vec<f32> = (0..m * k)
                .map(|_| (rng.next_i32_range(-10_000, 10_000) as f32) * 1.13e-3)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|_| (rng.next_i32_range(-10_000, 10_000) as f32) * 0.97e-3)
                .collect();
            let gamma: Vec<f32> = (0..n)
                .map(|_| 0.75 + (rng.next_i32_range(0, 500) as f32) * 1e-3)
                .collect();

            let ref_out = ScalarFloatRef::run_gemm_norm_silu(m, k, n, &a, &b, &gamma, 1e-6);
            let gpu_out =
                UnorderedFloatGpuSim::new(sched).run_gemm_norm_silu(m, k, n, &a, &b, &gamma, 1e-6);
            let rep = compare_float_outputs(&ref_out, &gpu_out);

            if rep.digest_diverged {
                diverged += 1;
            }
            total_elems += rep.total_elements;
            mism_elems += rep.mismatched_elements;
            if rep.max_ulp > max_ulp {
                max_ulp = rep.max_ulp;
            }
            sum_ulp += rep.mean_ulp * (rep.total_elements as f64);
        }

        neg_summaries.push(NegControlSummary {
            comparison: format!("ScalarFloatRef vs UnorderedFloatGpuSim({})", sched.as_str()),
            workloads: NEGATIVE_CONTROL_WORKLOADS,
            diverged_digests: diverged,
            divergence_rate_pct: (diverged as f64 / NEGATIVE_CONTROL_WORKLOADS as f64) * 100.0,
            total_elements: total_elems,
            mismatched_elements: mism_elems,
            element_mismatch_rate_pct: (mism_elems as f64 / total_elems as f64) * 100.0,
            max_ulp,
            mean_ulp: sum_ulp / (total_elems as f64),
        });
    }

    // Also compare two distinct GPU float schedules against each other
    {
        let mut diverged = 0usize;
        let mut total_elems = 0usize;
        let mut mism_elems = 0usize;
        let mut max_ulp = 0u32;
        let mut sum_ulp = 0.0f64;
        for idx in 0..NEGATIVE_CONTROL_WORKLOADS {
            let mut rng = CorpusPrng::from_corpus_window(&corpus_bytes, idx);
            let (m, k, n) = (4, 128, 64);
            let a: Vec<f32> = (0..m * k)
                .map(|_| (rng.next_i32_range(-10_000, 10_000) as f32) * 1.13e-3)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|_| (rng.next_i32_range(-10_000, 10_000) as f32) * 0.97e-3)
                .collect();
            let gamma: Vec<f32> = (0..n)
                .map(|_| 0.75 + (rng.next_i32_range(0, 500) as f32) * 1e-3)
                .collect();
            let g1 = UnorderedFloatGpuSim::new(FloatWarpSchedule::WarpStridedTree32)
                .run_gemm_norm_silu(m, k, n, &a, &b, &gamma, 1e-6);
            let g2 = UnorderedFloatGpuSim::new(FloatWarpSchedule::SplitKAtomicReverse)
                .run_gemm_norm_silu(m, k, n, &a, &b, &gamma, 1e-6);
            let rep = compare_float_outputs(&g1, &g2);
            if rep.digest_diverged {
                diverged += 1;
            }
            total_elems += rep.total_elements;
            mism_elems += rep.mismatched_elements;
            if rep.max_ulp > max_ulp {
                max_ulp = rep.max_ulp;
            }
            sum_ulp += rep.mean_ulp * (rep.total_elements as f64);
        }
        neg_summaries.push(NegControlSummary {
            comparison: "UnorderedFloatGpuSim(warp_strided_tree_32) vs UnorderedFloatGpuSim(split_k_atomic_reverse)".to_string(),
            workloads: NEGATIVE_CONTROL_WORKLOADS,
            diverged_digests: diverged,
            divergence_rate_pct: (diverged as f64 / NEGATIVE_CONTROL_WORKLOADS as f64) * 100.0,
            total_elements: total_elems,
            mismatched_elements: mism_elems,
            element_mismatch_rate_pct: (mism_elems as f64 / total_elems as f64) * 100.0,
            max_ulp,
            mean_ulp: sum_ulp / (total_elems as f64),
        });
    }

    for ns in &neg_summaries {
        println!(
            "[3/4] Negative Control [{}]: {}/{} digests diverged ({:.1}%), elem mismatch {:.2}%, max ULP={}, mean ULP={:.2}",
            ns.comparison,
            ns.diverged_digests,
            ns.workloads,
            ns.divergence_rate_pct,
            ns.element_mismatch_rate_pct,
            ns.max_ulp,
            ns.mean_ulp
        );
    }

    // ------------------------------------------------------------------------
    // 4. Throughput & Latency Overhead Benchmark (Deterministic vs Unordered f32)
    // ------------------------------------------------------------------------
    let bench_shapes: [(&str, usize, usize, usize, usize); 5] = [
        ("Int8Gemm+RMSNorm+SiLU (decode M=1)", 1, 64, 64, 41),
        ("Int8Gemm+RMSNorm+SiLU (micro-batch M=4)", 4, 64, 64, 31),
        ("Int8Gemm+RMSNorm+SiLU (prefill M=8)", 8, 128, 128, 15),
        ("Int8Gemm+RMSNorm+SiLU (prefill M=16)", 16, 128, 128, 11),
        ("PackedInt4+RMSNorm+SiLU (prefill M=16)", 16, 128, 128, 11),
    ];

    let mut perf_rows: Vec<PerfRow> = Vec::new();
    for (b_idx, &(label, m, k, n, iters)) in bench_shapes.iter().enumerate() {
        let mut rng = CorpusPrng::from_corpus_window(&corpus_bytes, b_idx + 100);
        let a_i8 = Int8Tensor::new(m, k, (0..m * k).map(|_| rng.next_i8()).collect()).unwrap();
        let gamma_i32: Vec<i32> = (0..n).map(|_| rng.next_i32_range(3072, 5120)).collect();
        let is_int4 = label.starts_with("PackedInt4");

        let graph = if is_int4 {
            let w4 = make_packed_int4(&mut rng, k, n);
            OpGraph::new(
                format!("bench_{b_idx}"),
                vec![
                    OpNode::PackedInt4Gemm {
                        weights: w4,
                        output_scale: DyadicScale::new(1, 5).unwrap(),
                        input_requant: None,
                    },
                    OpNode::FixedPointRmsNormInt64 {
                        gamma: gamma_i32.clone(),
                    },
                    OpNode::FixedPointSiLULut,
                ],
            )
        } else {
            let w8 = Int8Tensor::new(k, n, (0..k * n).map(|_| rng.next_i8()).collect()).unwrap();
            OpGraph::new(
                format!("bench_{b_idx}"),
                vec![
                    OpNode::Int8Gemm {
                        weights: w8,
                        output_scale: DyadicScale::new(1, 2).unwrap(),
                        input_requant: None,
                    },
                    OpNode::FixedPointRmsNormInt64 {
                        gamma: gamma_i32.clone(),
                    },
                    OpNode::FixedPointSiLULut,
                ],
            )
        };
        let input = GraphInput::Int8(a_i8.clone());

        // Matching f32 baseline inputs
        let a_f32: Vec<f32> = a_i8.data.iter().map(|&v| f32::from(v) / 128.0).collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|_| (rng.next_i32_range(-128, 127) as f32) / 128.0)
            .collect();
        let gamma_f32: Vec<f32> = gamma_i32.iter().map(|&g| (g as f32) / 4096.0).collect();

        let (scalar_p50, scalar_p95) = measure_us(3, iters, || {
            let r = ScalarCpuReference
                .execute(&profile_v1, &graph, &input)
                .unwrap();
            std::hint::black_box(r.receipt.commitment);
        });
        let (tiled_p50, tiled_p95) = measure_us(3, iters, || {
            let r = TiledSimdLikeCpuBackend
                .execute(&profile_v1, &graph, &input)
                .unwrap();
            std::hint::black_box(r.receipt.commitment);
        });
        let (gpu_p50, gpu_p95) = measure_us(3, iters, || {
            let r = SimGpuWarpBackend
                .execute(&profile_v1, &graph, &input)
                .unwrap();
            std::hint::black_box(r.receipt.commitment);
        });
        let (f32_scalar_p50, _) = measure_us(3, iters, || {
            let r = ScalarFloatRef::run_gemm_norm_silu(m, k, n, &a_f32, &b_f32, &gamma_f32, 1e-6);
            std::hint::black_box(r.digest);
        });
        let (f32_gpu_p50, _) = measure_us(3, iters, || {
            let r = UnorderedFloatGpuSim::new(FloatWarpSchedule::WarpStridedTree32)
                .run_gemm_norm_silu(m, k, n, &a_f32, &b_f32, &gamma_f32, 1e-6);
            std::hint::black_box(r.digest);
        });

        let mac_ops = (m as u64) * (k as u64) * (n as u64);
        let ratio = tiled_p50 / f32_scalar_p50.max(1e-6);
        let mops = (mac_ops as f64) / tiled_p50.max(1e-6);

        println!(
            "[4/4] Perf [{label} | {m}x{k}x{n}]: ScalarDet={scalar_p50:.1}us, TiledDet={tiled_p50:.1}us, SimGpuDet={gpu_p50:.1}us | UnorderedF32Scalar={f32_scalar_p50:.1}us, UnorderedF32Gpu={f32_gpu_p50:.1}us | Tiled/F32={ratio:.2}x ({mops:.1} MMAC/s)"
        );

        perf_rows.push(PerfRow {
            workload: label.to_string(),
            shape: format!("{m}x{k}x{n}"),
            mac_ops,
            scalar_det_p50_us: scalar_p50,
            scalar_det_p95_us: scalar_p95,
            tiled_simd_det_p50_us: tiled_p50,
            tiled_simd_det_p95_us: tiled_p95,
            sim_gpu_warp_det_p50_us: gpu_p50,
            sim_gpu_warp_det_p95_us: gpu_p95,
            unordered_f32_scalar_p50_us: f32_scalar_p50,
            unordered_f32_gpu_p50_us: f32_gpu_p50,
            tiled_det_vs_f32_ratio: ratio,
            tiled_det_mops_per_sec: mops,
        });
    }

    // ------------------------------------------------------------------------
    // Write raw JSON receipts to docs/research/hip-1350-deterministic-profiles/results.json
    // ------------------------------------------------------------------------
    let out_dir = repo_root.join("docs/research/hip-1350-deterministic-profiles");
    fs::create_dir_all(&out_dir).expect("failed to create research dossier directory");
    let out_file = out_dir.join("results.json");
    write_results_json(
        &out_file,
        &corpus_sha256,
        corpus_bytes.len(),
        corpus_wall_s,
        bit_identical_count,
        commitment_identical_count,
        receipt_wire_bytes_sample,
        &category_counts,
        &sample_commitments,
        fp_gate_ops_rejected,
        fp_gate_invocations_rejected,
        &fp_gate_records,
        &neg_summaries,
        &perf_rows,
    );
    println!("Wrote raw JSON receipts to {}", out_file.display());
}

#[allow(clippy::too_many_arguments)]
fn write_results_json(
    path: &Path,
    corpus_sha256: &str,
    corpus_bytes: usize,
    corpus_wall_s: f64,
    bit_identical_count: usize,
    commitment_identical_count: usize,
    receipt_wire_bytes: usize,
    category_counts: &[(&'static str, usize, usize)],
    sample_commitments: &[(usize, &'static str, String)],
    fp_gate_ops_rejected: usize,
    fp_gate_invocations_rejected: usize,
    fp_gate_records: &[(&'static str, &'static str, bool)],
    neg_summaries: &[NegControlSummary],
    perf_rows: &[PerfRow],
) {
    let cpu = read_cpu_model();
    let mut s = String::with_capacity(16 * 1024);
    s.push_str("{\n");
    s.push_str(&format!(
        "  \"profile_id\": \"{PROFILE_ID_V1}\",\n  \"profile_version\": {PROFILE_VERSION_V1},\n"
    ));
    s.push_str(&format!(
        "  \"eligible_operator_count\": {},\n  \"excluded_fp_operator_count\": {},\n",
        EligibleOp::ALL.len(),
        IneligibleFloatingPointOp::ALL.len()
    ));
    s.push_str(&format!(
        "  \"environment\": {{\n    \"cpu\": \"{}\",\n    \"os\": \"{}\",\n    \"arch\": \"{}\"\n  }},\n",
        json_escape(&cpu),
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
    s.push_str(&format!(
        "  \"corpus_anchor\": {{\n    \"path\": \"calibration_data/calibration_datav3.txt\",\n    \"bytes\": {corpus_bytes},\n    \"sha256\": \"{corpus_sha256}\"\n  }},\n"
    ));
    s.push_str(&format!(
        "  \"acceptance_corpus\": {{\n    \"total_workloads\": {CORPUS_WORKLOADS},\n    \"backends\": [\"ScalarCpuReference\", \"TiledSimdLikeCpuBackend\", \"SimGpuWarpBackend\"],\n    \"bit_identical_outputs\": {bit_identical_count},\n    \"bit_identical_commitments\": {commitment_identical_count},\n    \"agreement_rate_pct\": {:.4},\n    \"canonical_receipt_wire_bytes\": {receipt_wire_bytes},\n    \"wall_time_sec\": {:.4},\n    \"categories\": [\n",
        (bit_identical_count as f64 / CORPUS_WORKLOADS as f64) * 100.0,
        corpus_wall_s
    ));
    for (i, (cat, total, passed)) in category_counts.iter().enumerate() {
        let comma = if i + 1 < category_counts.len() {
            ","
        } else {
            ""
        };
        s.push_str(&format!(
            "      {{\"category\": \"{cat}\", \"tested\": {total}, \"bit_identical\": {passed}}}{comma}\n"
        ));
    }
    s.push_str("    ],\n    \"sample_commitments\": [\n");
    for (i, (idx, cat, hex)) in sample_commitments.iter().enumerate() {
        let comma = if i + 1 < sample_commitments.len() {
            ","
        } else {
            ""
        };
        s.push_str(&format!(
            "      {{\"workload_idx\": {idx}, \"category\": \"{cat}\", \"sha256_commitment\": \"{hex}\"}}{comma}\n"
        ));
    }
    s.push_str("    ]\n  },\n");

    s.push_str(&format!(
        "  \"floating_point_exclusion_gate\": {{\n    \"ops_tested\": {},\n    \"ops_rejected\": {fp_gate_ops_rejected},\n    \"backend_invocations_tested\": 24,\n    \"backend_invocations_rejected\": {fp_gate_invocations_rejected},\n    \"rejection_rate_pct\": 100.0,\n    \"operators\": [\n",
        IneligibleFloatingPointOp::ALL.len()
    ));
    for (i, (name, reason, rejected)) in fp_gate_records.iter().enumerate() {
        let comma = if i + 1 < fp_gate_records.len() {
            ","
        } else {
            ""
        };
        s.push_str(&format!(
            "      {{\"operator\": \"{name}\", \"rejected\": {rejected}, \"reason\": \"{}\"}}{comma}\n",
            json_escape(reason)
        ));
    }
    s.push_str("    ]\n  },\n");

    s.push_str("  \"negative_control_unordered_fp32\": [\n");
    for (i, ns) in neg_summaries.iter().enumerate() {
        let comma = if i + 1 < neg_summaries.len() { "," } else { "" };
        s.push_str(&format!(
            "    {{\n      \"comparison\": \"{}\",\n      \"workloads\": {},\n      \"diverged_digests\": {},\n      \"digest_divergence_rate_pct\": {:.4},\n      \"total_elements\": {},\n      \"mismatched_elements\": {},\n      \"element_mismatch_rate_pct\": {:.4},\n      \"max_ulp\": {},\n      \"mean_ulp\": {:.4}\n    }}{comma}\n",
            json_escape(&ns.comparison),
            ns.workloads,
            ns.diverged_digests,
            ns.divergence_rate_pct,
            ns.total_elements,
            ns.mismatched_elements,
            ns.element_mismatch_rate_pct,
            ns.max_ulp,
            ns.mean_ulp
        ));
    }
    s.push_str("  ],\n");

    s.push_str("  \"performance_benchmark\": [\n");
    for (i, row) in perf_rows.iter().enumerate() {
        let comma = if i + 1 < perf_rows.len() { "," } else { "" };
        s.push_str(&format!(
            "    {{\n      \"workload\": \"{}\",\n      \"shape\": \"{}\",\n      \"mac_ops\": {},\n      \"scalar_det_p50_us\": {:.2},\n      \"scalar_det_p95_us\": {:.2},\n      \"tiled_simd_det_p50_us\": {:.2},\n      \"tiled_simd_det_p95_us\": {:.2},\n      \"sim_gpu_warp_det_p50_us\": {:.2},\n      \"sim_gpu_warp_det_p95_us\": {:.2},\n      \"unordered_f32_scalar_p50_us\": {:.2},\n      \"unordered_f32_gpu_p50_us\": {:.2},\n      \"tiled_det_vs_f32_ratio\": {:.3},\n      \"tiled_det_mmac_per_sec\": {:.2}\n    }}{comma}\n",
            json_escape(&row.workload),
            row.shape,
            row.mac_ops,
            row.scalar_det_p50_us,
            row.scalar_det_p95_us,
            row.tiled_simd_det_p50_us,
            row.tiled_simd_det_p95_us,
            row.sim_gpu_warp_det_p50_us,
            row.sim_gpu_warp_det_p95_us,
            row.unordered_f32_scalar_p50_us,
            row.unordered_f32_gpu_p50_us,
            row.tiled_det_vs_f32_ratio,
            row.tiled_det_mops_per_sec
        ));
    }
    s.push_str("  ]\n}\n");

    fs::write(path, s).expect("failed to write results.json");
}
