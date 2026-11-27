// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Independent deterministic reference backends (`ScalarCpuReference`,
//! `TiledSimdLikeCpuBackend`, `SimGpuWarpBackend`) plus the explicit floating-point
//! negative control (`ScalarFloatRef` vs `UnorderedFloatGpuSim`).
//!
//! Each deterministic backend implements its own reduction and tiling topology:
//! 1. `ScalarCpuReference` — strict sequential left-to-right scalar oracle.
//! 2. `TiledSimdLikeCpuBackend` — `8x8` cache-blocked micro-kernel tiling with
//!    4-way unrolled accumulator lanes and chunked vector reductions.
//! 3. `SimGpuWarpBackend` — `16x16` threadblock + `WARP_SIZE = 32` strided warp
//!    lanes with an explicit 5-stage butterfly binary tree reduction (`16->8->4->2->1`).
//!
//! Under `det.zen-kai.v1`, all three backends MUST produce 100% bit-identical
//! canonical output bytes AND identical SHA-256 `ExecutionReceipt` commitments.

use crate::commit::{Digest, ExecutionReceipt, Sha256};
use crate::error::{DetError, DetResult};
use crate::ops::{
    fixed_point_silu_elem, requantize_i32_to_i8, ExecutionOutput, GraphInput, Int32Tensor,
    Int8Tensor, OpGraph, OpNode, PackedInt4Weight, RankedEntry, TopKOutput, QUANT_BLOCK_SIZE,
};
use crate::profile::{
    add_i32, isqrt_u128, narrow_i128_to_i32, rescale_i128, round_div_i128, DyadicScale,
    ExecutionProfile, FIXED_POINT_FRAC_BITS,
};

const SIMD_TILE_M: usize = 8;
const SIMD_TILE_N: usize = 8;
const SIMD_TILE_K: usize = 32;

const GPU_BLOCK_M: usize = 16;
const GPU_BLOCK_N: usize = 16;
const GPU_WARP_SIZE: usize = 32;

/// Output bundle returned by every `DeterministicBackend::execute` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionResult {
    /// Backend identifier that produced this execution.
    pub backend_name: &'static str,
    /// Structured typed execution output.
    pub output: ExecutionOutput,
    /// Canonical little-endian byte encoding of `output`.
    pub output_bytes: Vec<u8>,
    /// Canonical SHA-256 receipt binding profile, graph, inputs, weights, and output bytes.
    pub receipt: ExecutionReceipt,
}

/// Trait implemented by independent execution engines conforming to `det.zen-kai.v1`.
pub trait DeterministicBackend {
    /// Canonical human-readable backend name.
    fn name(&self) -> &'static str;

    /// Validate profile + graph (including the unsupported FP exclusion gate),
    /// execute all nodes deterministically, and return canonical output + receipt.
    fn execute(
        &self,
        profile: &ExecutionProfile,
        graph: &OpGraph,
        input: &GraphInput,
    ) -> DetResult<ExecutionResult>;
}

/// Internal activation carrier between stages of an `OpGraph`.
enum StageTensor {
    Int8(Int8Tensor),
    Int32(Int32Tensor),
}

impl StageTensor {
    fn from_input(input: &GraphInput) -> DetResult<Self> {
        input.validate()?;
        Ok(match input {
            GraphInput::Int8(t) => Self::Int8(t.clone()),
            GraphInput::Int32(t) => Self::Int32(t.clone()),
        })
    }

    fn as_int32(&self, op: &'static str) -> DetResult<&Int32Tensor> {
        match self {
            Self::Int32(t) => Ok(t),
            Self::Int8(_) => Err(DetError::InvalidGraph {
                detail: format!("{op} requires Int32 activation input"),
            }),
        }
    }

    fn to_int8_for_gemm(
        &self,
        input_requant: Option<DyadicScale>,
        profile: &ExecutionProfile,
        op: &'static str,
    ) -> DetResult<Int8Tensor> {
        match self {
            Self::Int8(t) => Ok(t.clone()),
            Self::Int32(t) => {
                let scale = input_requant.ok_or_else(|| DetError::InvalidGraph {
                    detail: format!(
                        "{op} received Int32 activation without explicit dyadic input_requant scale"
                    ),
                })?;
                requantize_i32_to_i8(t, scale, profile.rounding_mode, profile.overflow_mode)
            }
        }
    }
}

/// Helper to compute exact `floor((sum_sq * 2^24) / d) + eps_q24` without `u128` overflow,
/// then exact integer square root `isqrt_u128`.
#[inline]
fn exact_rms_scaled_q24(sum_sq: u128, d: usize, eps_q24: u64) -> DetResult<u128> {
    let d_u128 = d as u128;
    if d_u128 == 0 {
        return Err(DetError::DivisionByZero {
            op: "FixedPointRMSNorm_Int64",
        });
    }
    let q = sum_sq / d_u128;
    let r = sum_sq % d_u128;
    let mean_sq_q24 = (q << 24)
        .checked_add((r << 24) / d_u128)
        .and_then(|v| v.checked_add(u128::from(eps_q24)))
        .ok_or_else(|| DetError::ArithmeticOverflow {
            op: "FixedPointRMSNorm_Int64",
            detail: "mean_sq_q24 u128 overflow".to_string(),
        })?;
    let rms = isqrt_u128(mean_sq_q24);
    if rms == 0 {
        return Err(DetError::DivisionByZero {
            op: "FixedPointRMSNorm_Int64",
        });
    }
    Ok(rms)
}

/// Merge two sorted top-`k` slices into a single sorted top-`k` vector under
/// canonical `(value desc, index asc)` total ordering.
fn merge_sorted_topk(a: &[RankedEntry], b: &[RankedEntry], k: usize) -> Vec<RankedEntry> {
    let mut out = Vec::with_capacity(k.min(a.len() + b.len()));
    let (mut i, mut j) = (0usize, 0usize);
    while out.len() < k && i < a.len() && j < b.len() {
        if a[i].cmp_deterministic(&b[j]) != std::cmp::Ordering::Greater {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    while out.len() < k && i < a.len() {
        out.push(a[i]);
        i += 1;
    }
    while out.len() < k && j < b.len() {
        out.push(b[j]);
        j += 1;
    }
    out
}

fn finalize_result(
    backend_name: &'static str,
    profile: &ExecutionProfile,
    graph: &OpGraph,
    input: &GraphInput,
    output: ExecutionOutput,
) -> DetResult<ExecutionResult> {
    let output_bytes = output.to_canonical_bytes();
    let receipt = ExecutionReceipt::build(
        profile,
        graph.structure_digest(),
        input.digest(),
        graph.weights_digest(),
        &output_bytes,
    )?;
    Ok(ExecutionResult {
        backend_name,
        output,
        output_bytes,
        receipt,
    })
}

// ============================================================================
// Backend 1: ScalarCpuReference (strict sequential scalar canonical oracle)
// ============================================================================

/// Backend 1: Strict sequential scalar CPU reference oracle.
///
/// Executes every reduction in strict left-to-right `0..n` index order over
/// exact `i64`/`i128` accumulators.
#[derive(Debug, Default, Clone, Copy)]
pub struct ScalarCpuReference;

impl ScalarCpuReference {
    fn int8_gemm(
        a: &Int8Tensor,
        w: &Int8Tensor,
        scale: DyadicScale,
        profile: &ExecutionProfile,
    ) -> DetResult<Int32Tensor> {
        if a.cols != w.rows {
            return Err(DetError::InvalidDimensions {
                context: "ScalarCpuReference::Int8Gemm",
                expected: format!("A.cols ({}) == W.rows", a.cols),
                actual: format!("W.rows={}", w.rows),
            });
        }
        let (m, k_dim, n) = (a.rows, a.cols, w.cols);
        let mut out = vec![0i32; m * n];
        for i in 0..m {
            let a_row = &a.data[i * k_dim..(i + 1) * k_dim];
            for j in 0..n {
                let mut acc: i128 = 0;
                for (k, &a_val) in a_row.iter().enumerate() {
                    let b_val = w.data[k * n + j];
                    acc += i128::from(a_val) * i128::from(b_val);
                }
                let scaled = rescale_i128(acc, scale, profile.rounding_mode, "Int8Gemm")?;
                out[i * n + j] = narrow_i128_to_i32(scaled, profile.overflow_mode, "Int8Gemm")?;
            }
        }
        Int32Tensor::new(m, n, out)
    }

    fn packed_int4_gemm(
        a: &Int8Tensor,
        w: &PackedInt4Weight,
        scale: DyadicScale,
        profile: &ExecutionProfile,
    ) -> DetResult<Int32Tensor> {
        if a.cols != w.k {
            return Err(DetError::InvalidDimensions {
                context: "ScalarCpuReference::PackedInt4Gemm",
                expected: format!("A.cols ({}) == W.k", a.cols),
                actual: format!("W.k={}", w.k),
            });
        }
        let (m, k_dim, n) = (a.rows, a.cols, w.n);
        let mut out = vec![0i32; m * n];
        for i in 0..m {
            let a_row = &a.data[i * k_dim..(i + 1) * k_dim];
            for j in 0..n {
                let mut acc: i128 = 0;
                for (blk_idx, blk) in w.col_blocks(j).iter().enumerate() {
                    let base = blk_idx * QUANT_BLOCK_SIZE;
                    let mut blk_acc: i64 = 0;
                    for t in 0..QUANT_BLOCK_SIZE {
                        blk_acc += i64::from(a_row[base + t]) * i64::from(blk.unpack(t));
                    }
                    acc += i128::from(blk.scale) * i128::from(blk_acc);
                }
                let scaled = rescale_i128(acc, scale, profile.rounding_mode, "PackedInt4Gemm")?;
                out[i * n + j] =
                    narrow_i128_to_i32(scaled, profile.overflow_mode, "PackedInt4Gemm")?;
            }
        }
        Int32Tensor::new(m, n, out)
    }

    fn bias_add(
        x: &Int32Tensor,
        bias: &[i32],
        profile: &ExecutionProfile,
    ) -> DetResult<Int32Tensor> {
        if x.cols != bias.len() {
            return Err(DetError::InvalidDimensions {
                context: "ScalarCpuReference::Int32BiasAdd",
                expected: format!("bias.len() == {}", x.cols),
                actual: format!("{}", bias.len()),
            });
        }
        let mut out = vec![0i32; x.data.len()];
        for i in 0..x.rows {
            let row_off = i * x.cols;
            for j in 0..x.cols {
                out[row_off + j] = add_i32(
                    x.data[row_off + j],
                    bias[j],
                    profile.overflow_mode,
                    "Int32BiasAdd",
                )?;
            }
        }
        Int32Tensor::new(x.rows, x.cols, out)
    }

    fn relu(x: &Int32Tensor, clamp_max: Option<i32>) -> DetResult<Int32Tensor> {
        let mut out = Vec::with_capacity(x.data.len());
        for &v in &x.data {
            let mut r = if v < 0 { 0 } else { v };
            if let Some(cap) = clamp_max {
                if r > cap {
                    r = cap;
                }
            }
            out.push(r);
        }
        Int32Tensor::new(x.rows, x.cols, out)
    }

    fn silu_lut(x: &Int32Tensor, profile: &ExecutionProfile) -> DetResult<Int32Tensor> {
        let mut out = Vec::with_capacity(x.data.len());
        for &v in &x.data {
            out.push(fixed_point_silu_elem(
                v,
                profile.rounding_mode,
                profile.overflow_mode,
            )?);
        }
        Int32Tensor::new(x.rows, x.cols, out)
    }

    fn rms_norm(
        x: &Int32Tensor,
        gamma: &[i32],
        profile: &ExecutionProfile,
    ) -> DetResult<Int32Tensor> {
        if x.cols != gamma.len() {
            return Err(DetError::InvalidDimensions {
                context: "ScalarCpuReference::FixedPointRMSNorm_Int64",
                expected: format!("gamma.len() == {}", x.cols),
                actual: format!("{}", gamma.len()),
            });
        }
        let mut out = vec![0i32; x.data.len()];
        for i in 0..x.rows {
            let row = &x.data[i * x.cols..(i + 1) * x.cols];
            let mut sum_sq: u128 = 0;
            for &v in row {
                let mag = i128::from(v).unsigned_abs();
                sum_sq += mag * mag;
            }
            let rms = exact_rms_scaled_q24(sum_sq, x.cols, profile.rmsnorm_eps_q24)?;
            let out_row = &mut out[i * x.cols..(i + 1) * x.cols];
            for (j, (&v, &g)) in row.iter().zip(gamma.iter()).enumerate() {
                let num = (i128::from(v) * i128::from(g)) << FIXED_POINT_FRAC_BITS;
                let norm =
                    round_div_i128(num, rms, profile.rounding_mode, "FixedPointRMSNorm_Int64")?;
                out_row[j] =
                    narrow_i128_to_i32(norm, profile.overflow_mode, "FixedPointRMSNorm_Int64")?;
            }
        }
        Int32Tensor::new(x.rows, x.cols, out)
    }

    fn argmax(x: &Int32Tensor) -> DetResult<Vec<RankedEntry>> {
        let mut winners = Vec::with_capacity(x.rows);
        for i in 0..x.rows {
            let row = &x.data[i * x.cols..(i + 1) * x.cols];
            let mut best = RankedEntry {
                index: 0,
                value: row[0],
            };
            for (j, &v) in row.iter().enumerate().skip(1) {
                if v > best.value {
                    best = RankedEntry {
                        index: j as u32,
                        value: v,
                    };
                }
            }
            winners.push(best);
        }
        Ok(winners)
    }

    fn topk(x: &Int32Tensor, k: usize) -> DetResult<TopKOutput> {
        if k == 0 || k > x.cols {
            return Err(DetError::InvalidDimensions {
                context: "ScalarCpuReference::TopK_DeterministicOrder",
                expected: format!("1 <= k <= {}", x.cols),
                actual: format!("k={k}"),
            });
        }
        let mut entries = Vec::with_capacity(x.rows * k);
        for i in 0..x.rows {
            let row = &x.data[i * x.cols..(i + 1) * x.cols];
            let mut ranked: Vec<RankedEntry> = row
                .iter()
                .enumerate()
                .map(|(j, &v)| RankedEntry {
                    index: j as u32,
                    value: v,
                })
                .collect();
            ranked.sort_by(RankedEntry::cmp_deterministic);
            entries.extend_from_slice(&ranked[..k]);
        }
        Ok(TopKOutput {
            rows: x.rows,
            k,
            entries,
        })
    }
}

impl DeterministicBackend for ScalarCpuReference {
    fn name(&self) -> &'static str {
        "ScalarCpuReference"
    }

    fn execute(
        &self,
        profile: &ExecutionProfile,
        graph: &OpGraph,
        input: &GraphInput,
    ) -> DetResult<ExecutionResult> {
        graph.validate(profile)?;
        let mut state = StageTensor::from_input(input)?;
        let mut terminal: Option<ExecutionOutput> = None;

        for node in &graph.nodes {
            match node {
                OpNode::Int8Gemm {
                    weights,
                    output_scale,
                    input_requant,
                } => {
                    let act = state.to_int8_for_gemm(*input_requant, profile, "Int8Gemm")?;
                    state =
                        StageTensor::Int32(Self::int8_gemm(&act, weights, *output_scale, profile)?);
                }
                OpNode::PackedInt4Gemm {
                    weights,
                    output_scale,
                    input_requant,
                } => {
                    let act = state.to_int8_for_gemm(*input_requant, profile, "PackedInt4Gemm")?;
                    state = StageTensor::Int32(Self::packed_int4_gemm(
                        &act,
                        weights,
                        *output_scale,
                        profile,
                    )?);
                }
                OpNode::Int32BiasAdd { bias } => {
                    let x = state.as_int32("Int32BiasAdd")?;
                    state = StageTensor::Int32(Self::bias_add(x, bias, profile)?);
                }
                OpNode::FixedPointReLU { clamp_max } => {
                    let x = state.as_int32("FixedPointReLU")?;
                    state = StageTensor::Int32(Self::relu(x, *clamp_max)?);
                }
                OpNode::FixedPointSiLULut => {
                    let x = state.as_int32("FixedPointSiLU_LUT")?;
                    state = StageTensor::Int32(Self::silu_lut(x, profile)?);
                }
                OpNode::FixedPointRmsNormInt64 { gamma } => {
                    let x = state.as_int32("FixedPointRMSNorm_Int64")?;
                    state = StageTensor::Int32(Self::rms_norm(x, gamma, profile)?);
                }
                OpNode::ArgMaxDeterministicTieBreak => {
                    let x = state.as_int32("ArgMax_DeterministicTieBreak")?;
                    terminal = Some(ExecutionOutput::ArgMax(Self::argmax(x)?));
                }
                OpNode::TopKDeterministicOrder { k } => {
                    let x = state.as_int32("TopK_DeterministicOrder")?;
                    terminal = Some(ExecutionOutput::TopK(Self::topk(x, *k)?));
                }
                OpNode::UnsupportedFloatingPoint(fp) => {
                    return Err(DetError::UnsupportedFloatingPointOp {
                        op_name: fp.canonical_name(),
                        reason: fp.rejection_reason(),
                    });
                }
            }
        }

        let output = match terminal {
            Some(out) => out,
            None => ExecutionOutput::Int32(state.as_int32("OpGraph::output")?.clone()),
        };
        finalize_result(self.name(), profile, graph, input, output)
    }
}

// ============================================================================
// Backend 2: TiledSimdLikeCpuBackend (8x8 micro-kernel + unrolled reduction)
// ============================================================================

/// Backend 2: Cache-blocked `8x8` micro-kernel tiled CPU backend with 4-way
/// unrolled lane accumulation simulating vectorized SIMD execution.
#[derive(Debug, Default, Clone, Copy)]
pub struct TiledSimdLikeCpuBackend;

impl TiledSimdLikeCpuBackend {
    fn int8_gemm(
        a: &Int8Tensor,
        w: &Int8Tensor,
        scale: DyadicScale,
        profile: &ExecutionProfile,
    ) -> DetResult<Int32Tensor> {
        if a.cols != w.rows {
            return Err(DetError::InvalidDimensions {
                context: "TiledSimdLikeCpuBackend::Int8Gemm",
                expected: format!("A.cols ({}) == W.rows", a.cols),
                actual: format!("W.rows={}", w.rows),
            });
        }
        let (m, k_dim, n) = (a.rows, a.cols, w.cols);
        let mut out = vec![0i32; m * n];

        let mut i0 = 0;
        while i0 < m {
            let imax = (i0 + SIMD_TILE_M).min(m);
            let mut j0 = 0;
            while j0 < n {
                let jmax = (j0 + SIMD_TILE_N).min(n);
                let mut tile_acc = [[0i128; SIMD_TILE_N]; SIMD_TILE_M];

                // Cache-blocked K tiles: pack 32x8 weight panel into L1 stack buffer
                // for contiguous 4-way unrolled dot products across the 8x8 register tile.
                let mut k0 = 0;
                while k0 < k_dim {
                    let kmax = (k0 + SIMD_TILE_K).min(k_dim);
                    let klen = kmax - k0;
                    let cols_in_tile = jmax - j0;
                    let mut w_panel = [[0i8; SIMD_TILE_K]; SIMD_TILE_N];
                    for (jj, col_buf) in w_panel.iter_mut().enumerate().take(cols_in_tile) {
                        let col = j0 + jj;
                        for (kk, slot) in col_buf.iter_mut().enumerate().take(klen) {
                            *slot = w.data[(k0 + kk) * n + col];
                        }
                    }

                    for (ii, acc_row) in tile_acc.iter_mut().enumerate().take(imax - i0) {
                        let a_slice = &a.data[(i0 + ii) * k_dim + k0..(i0 + ii) * k_dim + kmax];
                        for (jj, acc_slot) in acc_row.iter_mut().enumerate().take(cols_in_tile) {
                            let w_col = &w_panel[jj][..klen];
                            let mut lane0: i64 = 0;
                            let mut lane1: i64 = 0;
                            let mut lane2: i64 = 0;
                            let mut lane3: i64 = 0;
                            let mut kk = 0;
                            while kk + 4 <= klen {
                                lane0 += i64::from(a_slice[kk]) * i64::from(w_col[kk]);
                                lane1 += i64::from(a_slice[kk + 1]) * i64::from(w_col[kk + 1]);
                                lane2 += i64::from(a_slice[kk + 2]) * i64::from(w_col[kk + 2]);
                                lane3 += i64::from(a_slice[kk + 3]) * i64::from(w_col[kk + 3]);
                                kk += 4;
                            }
                            let mut tail: i64 = 0;
                            while kk < klen {
                                tail += i64::from(a_slice[kk]) * i64::from(w_col[kk]);
                                kk += 1;
                            }
                            let pair_a = i128::from(lane0) + i128::from(lane1);
                            let pair_b = i128::from(lane2) + i128::from(lane3);
                            *acc_slot += (pair_a + pair_b) + i128::from(tail);
                        }
                    }
                    k0 += SIMD_TILE_K;
                }

                // Tiled epilogue: dyadic scaling & overflow narrowing
                for (ii, acc_row) in tile_acc.iter().enumerate().take(imax - i0) {
                    let row_off = (i0 + ii) * n;
                    for (jj, &acc) in acc_row.iter().enumerate().take(jmax - j0) {
                        let scaled = rescale_i128(acc, scale, profile.rounding_mode, "Int8Gemm")?;
                        out[row_off + j0 + jj] =
                            narrow_i128_to_i32(scaled, profile.overflow_mode, "Int8Gemm")?;
                    }
                }
                j0 += SIMD_TILE_N;
            }
            i0 += SIMD_TILE_M;
        }
        Int32Tensor::new(m, n, out)
    }

    fn packed_int4_gemm(
        a: &Int8Tensor,
        w: &PackedInt4Weight,
        scale: DyadicScale,
        profile: &ExecutionProfile,
    ) -> DetResult<Int32Tensor> {
        if a.cols != w.k {
            return Err(DetError::InvalidDimensions {
                context: "TiledSimdLikeCpuBackend::PackedInt4Gemm",
                expected: format!("A.cols ({}) == W.k", a.cols),
                actual: format!("W.k={}", w.k),
            });
        }
        let (m, k_dim, n) = (a.rows, a.cols, w.n);
        let bpc = w.blocks_per_col();
        let mut out = vec![0i32; m * n];

        let mut i0 = 0;
        while i0 < m {
            let imax = (i0 + SIMD_TILE_M).min(m);
            let rows_in_tile = imax - i0;
            let mut j0 = 0;
            while j0 < n {
                let jmax = (j0 + SIMD_TILE_N).min(n);
                let mut tile_acc = [[0i128; SIMD_TILE_N]; SIMD_TILE_M];

                // Process column blocks across the 8x8 register tile, unpacking
                // low/high nibbles once per block and reusing across 8 rows.
                for jj in 0..(jmax - j0) {
                    let col_blocks = w.col_blocks(j0 + jj);
                    for (b, blk) in col_blocks.iter().enumerate().take(bpc) {
                        let base = b * QUANT_BLOCK_SIZE;
                        let mut lo_w = [0i8; 16];
                        let mut hi_w = [0i8; 16];
                        for t in 0..16 {
                            let packed = blk.qs[t];
                            lo_w[t] = ((packed & 0x0F) as i8) - 8;
                            hi_w[t] = ((packed >> 4) as i8) - 8;
                        }
                        let blk_scale = i128::from(blk.scale);
                        for (ii, acc_row) in tile_acc.iter_mut().enumerate().take(rows_in_tile) {
                            let a_lo =
                                &a.data[(i0 + ii) * k_dim + base..(i0 + ii) * k_dim + base + 16];
                            let a_hi = &a.data
                                [(i0 + ii) * k_dim + base + 16..(i0 + ii) * k_dim + base + 32];
                            let mut s0: i64 = 0;
                            let mut s1: i64 = 0;
                            for t in 0..16 {
                                s0 += i64::from(a_lo[t]) * i64::from(lo_w[t]);
                                s1 += i64::from(a_hi[t]) * i64::from(hi_w[t]);
                            }
                            acc_row[jj] += blk_scale * i128::from(s0 + s1);
                        }
                    }
                }

                for (ii, acc_row) in tile_acc.iter().enumerate().take(rows_in_tile) {
                    let row_off = (i0 + ii) * n;
                    for (jj, &acc) in acc_row.iter().enumerate().take(jmax - j0) {
                        let scaled =
                            rescale_i128(acc, scale, profile.rounding_mode, "PackedInt4Gemm")?;
                        out[row_off + j0 + jj] =
                            narrow_i128_to_i32(scaled, profile.overflow_mode, "PackedInt4Gemm")?;
                    }
                }
                j0 += SIMD_TILE_N;
            }
            i0 += SIMD_TILE_M;
        }
        Int32Tensor::new(m, n, out)
    }

    fn bias_add(
        x: &Int32Tensor,
        bias: &[i32],
        profile: &ExecutionProfile,
    ) -> DetResult<Int32Tensor> {
        if x.cols != bias.len() {
            return Err(DetError::InvalidDimensions {
                context: "TiledSimdLikeCpuBackend::Int32BiasAdd",
                expected: format!("bias.len() == {}", x.cols),
                actual: format!("{}", bias.len()),
            });
        }
        let mut out = vec![0i32; x.data.len()];
        for i in 0..x.rows {
            let row = &x.data[i * x.cols..(i + 1) * x.cols];
            let out_row = &mut out[i * x.cols..(i + 1) * x.cols];
            let mut j = 0;
            while j + 8 <= x.cols {
                for lane in 0..8 {
                    out_row[j + lane] = add_i32(
                        row[j + lane],
                        bias[j + lane],
                        profile.overflow_mode,
                        "Int32BiasAdd",
                    )?;
                }
                j += 8;
            }
            while j < x.cols {
                out_row[j] = add_i32(row[j], bias[j], profile.overflow_mode, "Int32BiasAdd")?;
                j += 1;
            }
        }
        Int32Tensor::new(x.rows, x.cols, out)
    }

    fn relu(x: &Int32Tensor, clamp_max: Option<i32>) -> DetResult<Int32Tensor> {
        let mut out = vec![0i32; x.data.len()];
        let chunks = x.data.len() / 8;
        for c in 0..chunks {
            let base = c * 8;
            for lane in 0..8 {
                let v = x.data[base + lane].max(0);
                out[base + lane] = match clamp_max {
                    Some(cap) => v.min(cap),
                    None => v,
                };
            }
        }
        for idx in (chunks * 8)..x.data.len() {
            let v = x.data[idx].max(0);
            out[idx] = match clamp_max {
                Some(cap) => v.min(cap),
                None => v,
            };
        }
        Int32Tensor::new(x.rows, x.cols, out)
    }

    fn silu_lut(x: &Int32Tensor, profile: &ExecutionProfile) -> DetResult<Int32Tensor> {
        let mut out = vec![0i32; x.data.len()];
        for (dst_chunk, src_chunk) in out.chunks_mut(8).zip(x.data.chunks(8)) {
            for (dst, &v) in dst_chunk.iter_mut().zip(src_chunk.iter()) {
                *dst = fixed_point_silu_elem(v, profile.rounding_mode, profile.overflow_mode)?;
            }
        }
        Int32Tensor::new(x.rows, x.cols, out)
    }

    fn rms_norm(
        x: &Int32Tensor,
        gamma: &[i32],
        profile: &ExecutionProfile,
    ) -> DetResult<Int32Tensor> {
        if x.cols != gamma.len() {
            return Err(DetError::InvalidDimensions {
                context: "TiledSimdLikeCpuBackend::FixedPointRMSNorm_Int64",
                expected: format!("gamma.len() == {}", x.cols),
                actual: format!("{}", gamma.len()),
            });
        }
        let mut out = vec![0i32; x.data.len()];
        for i in 0..x.rows {
            let row = &x.data[i * x.cols..(i + 1) * x.cols];
            // 4-lane SIMD-style unrolled squared-sum accumulation
            let mut lanes = [0u128; 4];
            let mut j = 0;
            while j + 4 <= x.cols {
                for l in 0..4 {
                    let mag = i128::from(row[j + l]).unsigned_abs();
                    lanes[l] += mag * mag;
                }
                j += 4;
            }
            let mut tail: u128 = 0;
            while j < x.cols {
                let mag = i128::from(row[j]).unsigned_abs();
                tail += mag * mag;
                j += 1;
            }
            let sum_sq = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) + tail;
            let rms = exact_rms_scaled_q24(sum_sq, x.cols, profile.rmsnorm_eps_q24)?;

            let out_row = &mut out[i * x.cols..(i + 1) * x.cols];
            for tile_start in (0..x.cols).step_by(8) {
                let tile_end = (tile_start + 8).min(x.cols);
                for idx in tile_start..tile_end {
                    let num =
                        (i128::from(row[idx]) * i128::from(gamma[idx])) << FIXED_POINT_FRAC_BITS;
                    let norm =
                        round_div_i128(num, rms, profile.rounding_mode, "FixedPointRMSNorm_Int64")?;
                    out_row[idx] =
                        narrow_i128_to_i32(norm, profile.overflow_mode, "FixedPointRMSNorm_Int64")?;
                }
            }
        }
        Int32Tensor::new(x.rows, x.cols, out)
    }

    fn argmax(x: &Int32Tensor) -> DetResult<Vec<RankedEntry>> {
        let mut winners = Vec::with_capacity(x.rows);
        for i in 0..x.rows {
            let row = &x.data[i * x.cols..(i + 1) * x.cols];
            let mut lanes: [Option<RankedEntry>; 8] = [None; 8];
            for (j, &v) in row.iter().enumerate() {
                let cand = RankedEntry {
                    index: j as u32,
                    value: v,
                };
                let slot = &mut lanes[j & 7];
                *slot = Some(match *slot {
                    Some(prev) => RankedEntry::better_of(prev, cand),
                    None => cand,
                });
            }
            // Pairwise lane fold: 8 -> 4 -> 2 -> 1
            let mut best: Option<RankedEntry> = None;
            for entry in lanes.into_iter().flatten() {
                best = Some(match best {
                    Some(prev) => RankedEntry::better_of(prev, entry),
                    None => entry,
                });
            }
            winners.push(best.expect("non-empty row guaranteed by Int32Tensor"));
        }
        Ok(winners)
    }

    fn topk(x: &Int32Tensor, k: usize) -> DetResult<TopKOutput> {
        if k == 0 || k > x.cols {
            return Err(DetError::InvalidDimensions {
                context: "TiledSimdLikeCpuBackend::TopK_DeterministicOrder",
                expected: format!("1 <= k <= {}", x.cols),
                actual: format!("k={k}"),
            });
        }
        let mut entries = Vec::with_capacity(x.rows * k);
        for i in 0..x.rows {
            let row = &x.data[i * x.cols..(i + 1) * x.cols];
            let mut running_topk: Vec<RankedEntry> = Vec::new();
            let mut tile_buf = Vec::with_capacity(8);
            for (tile_idx, chunk) in row.chunks(8).enumerate() {
                tile_buf.clear();
                let base = tile_idx * 8;
                for (offset, &v) in chunk.iter().enumerate() {
                    tile_buf.push(RankedEntry {
                        index: (base + offset) as u32,
                        value: v,
                    });
                }
                tile_buf.sort_by(RankedEntry::cmp_deterministic);
                running_topk = merge_sorted_topk(&running_topk, &tile_buf, k);
            }
            entries.extend_from_slice(&running_topk);
        }
        Ok(TopKOutput {
            rows: x.rows,
            k,
            entries,
        })
    }
}

impl DeterministicBackend for TiledSimdLikeCpuBackend {
    fn name(&self) -> &'static str {
        "TiledSimdLikeCpuBackend"
    }

    fn execute(
        &self,
        profile: &ExecutionProfile,
        graph: &OpGraph,
        input: &GraphInput,
    ) -> DetResult<ExecutionResult> {
        graph.validate(profile)?;
        let mut state = StageTensor::from_input(input)?;
        let mut terminal: Option<ExecutionOutput> = None;

        for node in &graph.nodes {
            match node {
                OpNode::Int8Gemm {
                    weights,
                    output_scale,
                    input_requant,
                } => {
                    let act = state.to_int8_for_gemm(*input_requant, profile, "Int8Gemm")?;
                    state =
                        StageTensor::Int32(Self::int8_gemm(&act, weights, *output_scale, profile)?);
                }
                OpNode::PackedInt4Gemm {
                    weights,
                    output_scale,
                    input_requant,
                } => {
                    let act = state.to_int8_for_gemm(*input_requant, profile, "PackedInt4Gemm")?;
                    state = StageTensor::Int32(Self::packed_int4_gemm(
                        &act,
                        weights,
                        *output_scale,
                        profile,
                    )?);
                }
                OpNode::Int32BiasAdd { bias } => {
                    let x = state.as_int32("Int32BiasAdd")?;
                    state = StageTensor::Int32(Self::bias_add(x, bias, profile)?);
                }
                OpNode::FixedPointReLU { clamp_max } => {
                    let x = state.as_int32("FixedPointReLU")?;
                    state = StageTensor::Int32(Self::relu(x, *clamp_max)?);
                }
                OpNode::FixedPointSiLULut => {
                    let x = state.as_int32("FixedPointSiLU_LUT")?;
                    state = StageTensor::Int32(Self::silu_lut(x, profile)?);
                }
                OpNode::FixedPointRmsNormInt64 { gamma } => {
                    let x = state.as_int32("FixedPointRMSNorm_Int64")?;
                    state = StageTensor::Int32(Self::rms_norm(x, gamma, profile)?);
                }
                OpNode::ArgMaxDeterministicTieBreak => {
                    let x = state.as_int32("ArgMax_DeterministicTieBreak")?;
                    terminal = Some(ExecutionOutput::ArgMax(Self::argmax(x)?));
                }
                OpNode::TopKDeterministicOrder { k } => {
                    let x = state.as_int32("TopK_DeterministicOrder")?;
                    terminal = Some(ExecutionOutput::TopK(Self::topk(x, *k)?));
                }
                OpNode::UnsupportedFloatingPoint(fp) => {
                    return Err(DetError::UnsupportedFloatingPointOp {
                        op_name: fp.canonical_name(),
                        reason: fp.rejection_reason(),
                    });
                }
            }
        }

        let output = match terminal {
            Some(out) => out,
            None => ExecutionOutput::Int32(state.as_int32("OpGraph::output")?.clone()),
        };
        finalize_result(self.name(), profile, graph, input, output)
    }
}

// ============================================================================
// Backend 3: SimGpuWarpBackend (16x16 threadblock + 32-lane warp tree reduce)
// ============================================================================

/// Backend 3: Simulated GPU threadblock (`16x16`) and warp (`32` lanes) backend
/// executing explicit 5-stage shuffle-down binary tree reductions (`16->8->4->2->1`)
/// over exact integer lane accumulators and deterministic fixed-point epilogues.
#[derive(Debug, Default, Clone, Copy)]
pub struct SimGpuWarpBackend;

impl SimGpuWarpBackend {
    /// 5-stage deterministic warp shuffle-down binary reduction tree for `i128`.
    #[inline]
    fn warp_tree_sum_i128(mut lanes: [i128; GPU_WARP_SIZE]) -> i128 {
        let mut stride = GPU_WARP_SIZE / 2;
        while stride > 0 {
            for i in 0..stride {
                lanes[i] += lanes[i + stride];
            }
            stride >>= 1;
        }
        lanes[0]
    }

    /// 5-stage deterministic warp shuffle-down binary reduction tree for `u128`.
    #[inline]
    fn warp_tree_sum_u128(mut lanes: [u128; GPU_WARP_SIZE]) -> u128 {
        let mut stride = GPU_WARP_SIZE / 2;
        while stride > 0 {
            for i in 0..stride {
                lanes[i] += lanes[i + stride];
            }
            stride >>= 1;
        }
        lanes[0]
    }

    fn int8_gemm(
        a: &Int8Tensor,
        w: &Int8Tensor,
        scale: DyadicScale,
        profile: &ExecutionProfile,
    ) -> DetResult<Int32Tensor> {
        if a.cols != w.rows {
            return Err(DetError::InvalidDimensions {
                context: "SimGpuWarpBackend::Int8Gemm",
                expected: format!("A.cols ({}) == W.rows", a.cols),
                actual: format!("W.rows={}", w.rows),
            });
        }
        let (m, k_dim, n) = (a.rows, a.cols, w.cols);
        let mut out = vec![0i32; m * n];
        let grid_y = m.div_ceil(GPU_BLOCK_M);
        let grid_x = n.div_ceil(GPU_BLOCK_N);

        for by in 0..grid_y {
            for bx in 0..grid_x {
                for ty in 0..GPU_BLOCK_M {
                    let row = by * GPU_BLOCK_M + ty;
                    if row >= m {
                        continue;
                    }
                    let a_row = &a.data[row * k_dim..(row + 1) * k_dim];
                    for tx in 0..GPU_BLOCK_N {
                        let col = bx * GPU_BLOCK_N + tx;
                        if col >= n {
                            continue;
                        }
                        // Each of the 32 warp lanes accumulates strided K indices.
                        let mut warp_lanes = [0i128; GPU_WARP_SIZE];
                        for (lane, lane_acc) in warp_lanes.iter_mut().enumerate() {
                            let mut k = lane;
                            let mut partial: i128 = 0;
                            while k < k_dim {
                                partial += i128::from(a_row[k]) * i128::from(w.data[k * n + col]);
                                k += GPU_WARP_SIZE;
                            }
                            *lane_acc = partial;
                        }
                        let acc = Self::warp_tree_sum_i128(warp_lanes);
                        let scaled = rescale_i128(acc, scale, profile.rounding_mode, "Int8Gemm")?;
                        out[row * n + col] =
                            narrow_i128_to_i32(scaled, profile.overflow_mode, "Int8Gemm")?;
                    }
                }
            }
        }
        Int32Tensor::new(m, n, out)
    }

    fn packed_int4_gemm(
        a: &Int8Tensor,
        w: &PackedInt4Weight,
        scale: DyadicScale,
        profile: &ExecutionProfile,
    ) -> DetResult<Int32Tensor> {
        if a.cols != w.k {
            return Err(DetError::InvalidDimensions {
                context: "SimGpuWarpBackend::PackedInt4Gemm",
                expected: format!("A.cols ({}) == W.k", a.cols),
                actual: format!("W.k={}", w.k),
            });
        }
        let (m, k_dim, n) = (a.rows, a.cols, w.n);
        let mut out = vec![0i32; m * n];
        let grid_y = m.div_ceil(GPU_BLOCK_M);
        let grid_x = n.div_ceil(GPU_BLOCK_N);

        for by in 0..grid_y {
            for bx in 0..grid_x {
                for ty in 0..GPU_BLOCK_M {
                    let row = by * GPU_BLOCK_M + ty;
                    if row >= m {
                        continue;
                    }
                    let a_row = &a.data[row * k_dim..(row + 1) * k_dim];
                    for tx in 0..GPU_BLOCK_N {
                        let col = bx * GPU_BLOCK_N + tx;
                        if col >= n {
                            continue;
                        }
                        // Each lane `l in 0..32` handles element `l` of every 32-weight block
                        // with pre-distributed block scale, then reduces via 5-stage warp tree.
                        let col_blocks = w.col_blocks(col);
                        let mut warp_lanes = [0i128; GPU_WARP_SIZE];
                        for (lane, lane_acc) in warp_lanes.iter_mut().enumerate() {
                            let mut partial: i128 = 0;
                            for (b, blk) in col_blocks.iter().enumerate() {
                                let k_idx = b * QUANT_BLOCK_SIZE + lane;
                                let prod = i64::from(a_row[k_idx]) * i64::from(blk.unpack(lane));
                                partial += i128::from(blk.scale) * i128::from(prod);
                            }
                            *lane_acc = partial;
                        }
                        let acc = Self::warp_tree_sum_i128(warp_lanes);
                        let scaled =
                            rescale_i128(acc, scale, profile.rounding_mode, "PackedInt4Gemm")?;
                        out[row * n + col] =
                            narrow_i128_to_i32(scaled, profile.overflow_mode, "PackedInt4Gemm")?;
                    }
                }
            }
        }
        Int32Tensor::new(m, n, out)
    }

    fn bias_add(
        x: &Int32Tensor,
        bias: &[i32],
        profile: &ExecutionProfile,
    ) -> DetResult<Int32Tensor> {
        if x.cols != bias.len() {
            return Err(DetError::InvalidDimensions {
                context: "SimGpuWarpBackend::Int32BiasAdd",
                expected: format!("bias.len() == {}", x.cols),
                actual: format!("{}", bias.len()),
            });
        }
        let mut out = vec![0i32; x.data.len()];
        let grid_y = x.rows.div_ceil(GPU_BLOCK_M);
        let grid_x = x.cols.div_ceil(GPU_BLOCK_N);
        for by in 0..grid_y {
            for bx in 0..grid_x {
                for ty in 0..GPU_BLOCK_M {
                    let row = by * GPU_BLOCK_M + ty;
                    if row >= x.rows {
                        continue;
                    }
                    for tx in 0..GPU_BLOCK_N {
                        let col = bx * GPU_BLOCK_N + tx;
                        if col >= x.cols {
                            continue;
                        }
                        let idx = row * x.cols + col;
                        out[idx] = add_i32(
                            x.data[idx],
                            bias[col],
                            profile.overflow_mode,
                            "Int32BiasAdd",
                        )?;
                    }
                }
            }
        }
        Int32Tensor::new(x.rows, x.cols, out)
    }

    fn relu(x: &Int32Tensor, clamp_max: Option<i32>) -> DetResult<Int32Tensor> {
        let mut out = vec![0i32; x.data.len()];
        let total = x.data.len();
        let num_warps = total.div_ceil(GPU_WARP_SIZE);
        for warp in 0..num_warps {
            for lane in 0..GPU_WARP_SIZE {
                let idx = warp * GPU_WARP_SIZE + lane;
                if idx >= total {
                    continue;
                }
                let mut v = x.data[idx].max(0);
                if let Some(cap) = clamp_max {
                    v = v.min(cap);
                }
                out[idx] = v;
            }
        }
        Int32Tensor::new(x.rows, x.cols, out)
    }

    fn silu_lut(x: &Int32Tensor, profile: &ExecutionProfile) -> DetResult<Int32Tensor> {
        let mut out = vec![0i32; x.data.len()];
        let total = x.data.len();
        let num_warps = total.div_ceil(GPU_WARP_SIZE);
        for warp in 0..num_warps {
            for lane in 0..GPU_WARP_SIZE {
                let idx = warp * GPU_WARP_SIZE + lane;
                if idx >= total {
                    continue;
                }
                out[idx] = fixed_point_silu_elem(
                    x.data[idx],
                    profile.rounding_mode,
                    profile.overflow_mode,
                )?;
            }
        }
        Int32Tensor::new(x.rows, x.cols, out)
    }

    fn rms_norm(
        x: &Int32Tensor,
        gamma: &[i32],
        profile: &ExecutionProfile,
    ) -> DetResult<Int32Tensor> {
        if x.cols != gamma.len() {
            return Err(DetError::InvalidDimensions {
                context: "SimGpuWarpBackend::FixedPointRMSNorm_Int64",
                expected: format!("gamma.len() == {}", x.cols),
                actual: format!("{}", gamma.len()),
            });
        }
        let mut out = vec![0i32; x.data.len()];
        // One warp per row: strided warp accumulation + 5-stage butterfly tree
        for i in 0..x.rows {
            let row = &x.data[i * x.cols..(i + 1) * x.cols];
            let mut lane_sq = [0u128; GPU_WARP_SIZE];
            for (lane, acc) in lane_sq.iter_mut().enumerate() {
                let mut j = lane;
                let mut partial = 0u128;
                while j < x.cols {
                    let mag = i128::from(row[j]).unsigned_abs();
                    partial += mag * mag;
                    j += GPU_WARP_SIZE;
                }
                *acc = partial;
            }
            let sum_sq = Self::warp_tree_sum_u128(lane_sq);
            let rms = exact_rms_scaled_q24(sum_sq, x.cols, profile.rmsnorm_eps_q24)?;

            let out_row = &mut out[i * x.cols..(i + 1) * x.cols];
            for lane in 0..GPU_WARP_SIZE {
                let mut j = lane;
                while j < x.cols {
                    let num = (i128::from(row[j]) * i128::from(gamma[j])) << FIXED_POINT_FRAC_BITS;
                    let norm =
                        round_div_i128(num, rms, profile.rounding_mode, "FixedPointRMSNorm_Int64")?;
                    out_row[j] =
                        narrow_i128_to_i32(norm, profile.overflow_mode, "FixedPointRMSNorm_Int64")?;
                    j += GPU_WARP_SIZE;
                }
            }
        }
        Int32Tensor::new(x.rows, x.cols, out)
    }

    fn argmax(x: &Int32Tensor) -> DetResult<Vec<RankedEntry>> {
        let mut winners = Vec::with_capacity(x.rows);
        for i in 0..x.rows {
            let row = &x.data[i * x.cols..(i + 1) * x.cols];
            let mut warp_best: [Option<RankedEntry>; GPU_WARP_SIZE] = [None; GPU_WARP_SIZE];
            for (lane, slot) in warp_best.iter_mut().enumerate() {
                let mut j = lane;
                let mut local: Option<RankedEntry> = None;
                while j < x.cols {
                    let cand = RankedEntry {
                        index: j as u32,
                        value: row[j],
                    };
                    local = Some(match local {
                        Some(prev) => RankedEntry::better_of(prev, cand),
                        None => cand,
                    });
                    j += GPU_WARP_SIZE;
                }
                *slot = local;
            }
            // 5-stage warp shuffle-down binary tree with associative tie-break
            let mut stride = GPU_WARP_SIZE / 2;
            while stride > 0 {
                for l in 0..stride {
                    warp_best[l] = match (warp_best[l], warp_best[l + stride]) {
                        (Some(a), Some(b)) => Some(RankedEntry::better_of(a, b)),
                        (Some(a), None) => Some(a),
                        (None, Some(b)) => Some(b),
                        (None, None) => None,
                    };
                }
                stride >>= 1;
            }
            winners.push(warp_best[0].expect("non-empty row guaranteed by Int32Tensor"));
        }
        Ok(winners)
    }

    fn topk(x: &Int32Tensor, k: usize) -> DetResult<TopKOutput> {
        if k == 0 || k > x.cols {
            return Err(DetError::InvalidDimensions {
                context: "SimGpuWarpBackend::TopK_DeterministicOrder",
                expected: format!("1 <= k <= {}", x.cols),
                actual: format!("k={k}"),
            });
        }
        let mut entries = Vec::with_capacity(x.rows * k);
        for i in 0..x.rows {
            let row = &x.data[i * x.cols..(i + 1) * x.cols];
            // Each warp lane builds its own locally sorted top-k list over strided indices.
            let mut warp_lists: Vec<Vec<RankedEntry>> = (0..GPU_WARP_SIZE)
                .map(|lane| {
                    let mut local = Vec::new();
                    let mut j = lane;
                    while j < x.cols {
                        local.push(RankedEntry {
                            index: j as u32,
                            value: row[j],
                        });
                        j += GPU_WARP_SIZE;
                    }
                    local.sort_by(RankedEntry::cmp_deterministic);
                    local.truncate(k);
                    local
                })
                .collect();

            // 5-stage warp binary tree merge
            let mut stride = GPU_WARP_SIZE / 2;
            while stride > 0 {
                for l in 0..stride {
                    let merged = merge_sorted_topk(&warp_lists[l], &warp_lists[l + stride], k);
                    warp_lists[l] = merged;
                }
                stride >>= 1;
            }
            entries.extend_from_slice(&warp_lists[0]);
        }
        Ok(TopKOutput {
            rows: x.rows,
            k,
            entries,
        })
    }
}

impl DeterministicBackend for SimGpuWarpBackend {
    fn name(&self) -> &'static str {
        "SimGpuWarpBackend"
    }

    fn execute(
        &self,
        profile: &ExecutionProfile,
        graph: &OpGraph,
        input: &GraphInput,
    ) -> DetResult<ExecutionResult> {
        graph.validate(profile)?;
        let mut state = StageTensor::from_input(input)?;
        let mut terminal: Option<ExecutionOutput> = None;

        for node in &graph.nodes {
            match node {
                OpNode::Int8Gemm {
                    weights,
                    output_scale,
                    input_requant,
                } => {
                    let act = state.to_int8_for_gemm(*input_requant, profile, "Int8Gemm")?;
                    state =
                        StageTensor::Int32(Self::int8_gemm(&act, weights, *output_scale, profile)?);
                }
                OpNode::PackedInt4Gemm {
                    weights,
                    output_scale,
                    input_requant,
                } => {
                    let act = state.to_int8_for_gemm(*input_requant, profile, "PackedInt4Gemm")?;
                    state = StageTensor::Int32(Self::packed_int4_gemm(
                        &act,
                        weights,
                        *output_scale,
                        profile,
                    )?);
                }
                OpNode::Int32BiasAdd { bias } => {
                    let x = state.as_int32("Int32BiasAdd")?;
                    state = StageTensor::Int32(Self::bias_add(x, bias, profile)?);
                }
                OpNode::FixedPointReLU { clamp_max } => {
                    let x = state.as_int32("FixedPointReLU")?;
                    state = StageTensor::Int32(Self::relu(x, *clamp_max)?);
                }
                OpNode::FixedPointSiLULut => {
                    let x = state.as_int32("FixedPointSiLU_LUT")?;
                    state = StageTensor::Int32(Self::silu_lut(x, profile)?);
                }
                OpNode::FixedPointRmsNormInt64 { gamma } => {
                    let x = state.as_int32("FixedPointRMSNorm_Int64")?;
                    state = StageTensor::Int32(Self::rms_norm(x, gamma, profile)?);
                }
                OpNode::ArgMaxDeterministicTieBreak => {
                    let x = state.as_int32("ArgMax_DeterministicTieBreak")?;
                    terminal = Some(ExecutionOutput::ArgMax(Self::argmax(x)?));
                }
                OpNode::TopKDeterministicOrder { k } => {
                    let x = state.as_int32("TopK_DeterministicOrder")?;
                    terminal = Some(ExecutionOutput::TopK(Self::topk(x, *k)?));
                }
                OpNode::UnsupportedFloatingPoint(fp) => {
                    return Err(DetError::UnsupportedFloatingPointOp {
                        op_name: fp.canonical_name(),
                        reason: fp.rejection_reason(),
                    });
                }
            }
        }

        let output = match terminal {
            Some(out) => out,
            None => ExecutionOutput::Int32(state.as_int32("OpGraph::output")?.clone()),
        };
        finalize_result(self.name(), profile, graph, input, output)
    }
}

// ============================================================================
// Explicit Negative Control: UnorderedFloatGpuSim vs ScalarFloatRef
// ============================================================================

/// Floating-point GPU warp/atomic reduction schedules used in the explicit
/// negative control to demonstrate real IEEE-754 non-associativity divergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatWarpSchedule {
    /// 32-lane strided warp accumulation + 5-stage binary reduction tree in `f32`.
    WarpStridedTree32,
    /// 4-way Split-K reverse-arrival order simulating `atomicAdd(float*)` races.
    SplitKAtomicReverse,
    /// 32-lane warp tree using hardware FMA (`f32::mul_add`) contraction.
    FmaWarpTree32,
}

impl FloatWarpSchedule {
    /// Human-readable schedule identifier.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WarpStridedTree32 => "warp_strided_tree_32",
            Self::SplitKAtomicReverse => "split_k_atomic_reverse",
            Self::FmaWarpTree32 => "fma_warp_tree_32",
        }
    }
}

/// Output of a floating-point negative-control execution (`f32` vector + SHA-256).
#[derive(Debug, Clone)]
pub struct FloatNegativeControlOutput {
    /// Raw `f32` output elements.
    pub values: Vec<f32>,
    /// SHA-256 over canonical little-endian IEEE-754 `f32` bytes.
    pub digest: Digest,
}

impl FloatNegativeControlOutput {
    fn from_values(values: Vec<f32>) -> Self {
        let mut h = Sha256::new();
        for &v in &values {
            h.update(&v.to_le_bytes());
        }
        let digest = h.finalize();
        Self { values, digest }
    }
}

/// Divergence metrics comparing two floating-point negative-control runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatDivergenceReport {
    /// True iff SHA-256 digests of raw `f32` output bytes differ.
    pub digest_diverged: bool,
    /// Number of elements whose IEEE-754 bit pattern differs.
    pub mismatched_elements: usize,
    /// Total elements compared.
    pub total_elements: usize,
    /// Maximum ULP distance observed across all elements.
    pub max_ulp: u32,
    /// Mean ULP distance across all elements.
    pub mean_ulp: f64,
}

/// Compute exact IEEE-754 ULP distance between two finite `f32` values.
pub fn f32_ulp_distance(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    let ua = f32_to_ordered_u32(a);
    let ub = f32_to_ordered_u32(b);
    ua.abs_diff(ub)
}

#[inline]
fn f32_to_ordered_u32(x: f32) -> u32 {
    let bits = x.to_bits();
    if (bits & 0x8000_0000) != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Compare two `FloatNegativeControlOutput` runs for bit-level and ULP divergence.
pub fn compare_float_outputs(
    reference: &FloatNegativeControlOutput,
    candidate: &FloatNegativeControlOutput,
) -> FloatDivergenceReport {
    assert_eq!(reference.values.len(), candidate.values.len());
    let total = reference.values.len();
    let mut mismatched = 0usize;
    let mut max_ulp = 0u32;
    let mut sum_ulp = 0u64;
    for (&a, &b) in reference.values.iter().zip(candidate.values.iter()) {
        if a.to_bits() != b.to_bits() {
            mismatched += 1;
            let ulp = f32_ulp_distance(a, b);
            if ulp > max_ulp {
                max_ulp = ulp;
            }
            sum_ulp = sum_ulp.saturating_add(u64::from(ulp));
        }
    }
    FloatDivergenceReport {
        digest_diverged: reference.digest != candidate.digest,
        mismatched_elements: mismatched,
        total_elements: total,
        max_ulp,
        mean_ulp: if total == 0 {
            0.0
        } else {
            sum_ulp as f64 / total as f64
        },
    }
}

/// Negative-control baseline: strict sequential left-to-right `f32` GEMM + RMSNorm + SiLU.
#[derive(Debug, Default, Clone, Copy)]
pub struct ScalarFloatRef;

impl ScalarFloatRef {
    /// Execute a sequential `f32` GEMM (`[M, K] * [K, N]`) + `f32` RMSNorm + `f32` SiLU.
    pub fn run_gemm_norm_silu(
        m: usize,
        k_dim: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        gamma: &[f32],
        eps: f32,
    ) -> FloatNegativeControlOutput {
        let mut hidden = vec![0.0f32; m * n];
        for i in 0..m {
            let a_row = &a[i * k_dim..(i + 1) * k_dim];
            for j in 0..n {
                let mut acc = 0.0f32;
                for (k, &av) in a_row.iter().enumerate() {
                    let prod = av * b[k * n + j];
                    acc += prod;
                }
                hidden[i * n + j] = acc;
            }
        }
        // Sequential f32 RMSNorm + SiLU
        for i in 0..m {
            let row = &mut hidden[i * n..(i + 1) * n];
            let mut sum_sq = 0.0f32;
            for &v in row.iter() {
                sum_sq += v * v;
            }
            let rms = (sum_sq / (n as f32) + eps).sqrt();
            for (j, slot) in row.iter_mut().enumerate() {
                let norm = (*slot / rms) * gamma[j];
                *slot = norm / (1.0f32 + (-norm).exp());
            }
        }
        FloatNegativeControlOutput::from_values(hidden)
    }
}

/// Explicit Negative Control: simulated GPU floating-point execution with
/// non-associative parallel warp/split-K/FMA reduction schedules.
///
/// This backend is NEVER eligible under `det.zen-kai.v1`; it exists solely to
/// prove empirically why unprofiled FP32 GPU reductions diverge bit-for-bit.
#[derive(Debug, Clone, Copy)]
pub struct UnorderedFloatGpuSim {
    /// Non-associative warp/split-K/FMA reduction schedule.
    pub schedule: FloatWarpSchedule,
}

impl UnorderedFloatGpuSim {
    /// Construct an unordered floating-point GPU simulator with the given schedule.
    pub const fn new(schedule: FloatWarpSchedule) -> Self {
        Self { schedule }
    }

    /// Execute the exact same `f32` GEMM + RMSNorm + SiLU workload under a
    /// non-associative parallel GPU reduction schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn run_gemm_norm_silu(
        &self,
        m: usize,
        k_dim: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        gamma: &[f32],
        eps: f32,
    ) -> FloatNegativeControlOutput {
        let mut hidden = vec![0.0f32; m * n];
        for i in 0..m {
            let a_row = &a[i * k_dim..(i + 1) * k_dim];
            for j in 0..n {
                hidden[i * n + j] = match self.schedule {
                    FloatWarpSchedule::WarpStridedTree32 => {
                        let mut lanes = [0.0f32; GPU_WARP_SIZE];
                        for (lane, slot) in lanes.iter_mut().enumerate() {
                            let mut k = lane;
                            let mut partial = 0.0f32;
                            while k < k_dim {
                                partial += a_row[k] * b[k * n + j];
                                k += GPU_WARP_SIZE;
                            }
                            *slot = partial;
                        }
                        Self::warp_tree_sum_f32(lanes)
                    }
                    FloatWarpSchedule::SplitKAtomicReverse => {
                        // 4 split-K CTA chunks arriving at atomicAdd(float*) in reverse order
                        let num_splits = 4usize;
                        let chunk = k_dim.div_ceil(num_splits);
                        let mut partials = [0.0f32; 4];
                        for (s, p) in partials.iter_mut().enumerate() {
                            let start = s * chunk;
                            let end = (start + chunk).min(k_dim);
                            let mut acc = 0.0f32;
                            for k in (start..end).rev() {
                                acc += a_row[k] * b[k * n + j];
                            }
                            *p = acc;
                        }
                        // Reverse-arrival atomic float accumulation
                        let mut total = 0.0f32;
                        for &p in partials.iter().rev() {
                            total += p;
                        }
                        total
                    }
                    FloatWarpSchedule::FmaWarpTree32 => {
                        let mut lanes = [0.0f32; GPU_WARP_SIZE];
                        for (lane, slot) in lanes.iter_mut().enumerate() {
                            let mut k = lane;
                            let mut partial = 0.0f32;
                            while k < k_dim {
                                partial = a_row[k].mul_add(b[k * n + j], partial);
                                k += GPU_WARP_SIZE;
                            }
                            *slot = partial;
                        }
                        Self::warp_tree_sum_f32(lanes)
                    }
                };
            }
        }

        // Parallel warp-tree f32 RMSNorm + SiLU
        for i in 0..m {
            let row = &mut hidden[i * n..(i + 1) * n];
            let mut sq_lanes = [0.0f32; GPU_WARP_SIZE];
            for (lane, slot) in sq_lanes.iter_mut().enumerate() {
                let mut j = lane;
                let mut partial = 0.0f32;
                while j < n {
                    partial = match self.schedule {
                        FloatWarpSchedule::FmaWarpTree32 => row[j].mul_add(row[j], partial),
                        _ => partial + row[j] * row[j],
                    };
                    j += GPU_WARP_SIZE;
                }
                *slot = partial;
            }
            let sum_sq = Self::warp_tree_sum_f32(sq_lanes);
            let inv_rms = 1.0f32 / (sum_sq / (n as f32) + eps).sqrt();
            for (j, slot) in row.iter_mut().enumerate() {
                let norm = (*slot * inv_rms) * gamma[j];
                *slot = norm / (1.0f32 + (-norm).exp());
            }
        }
        FloatNegativeControlOutput::from_values(hidden)
    }

    #[inline]
    fn warp_tree_sum_f32(mut lanes: [f32; GPU_WARP_SIZE]) -> f32 {
        let mut stride = GPU_WARP_SIZE / 2;
        while stride > 0 {
            for i in 0..stride {
                lanes[i] += lanes[i + stride];
            }
            stride >>= 1;
        }
        lanes[0]
    }
}
