// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Canonical Zen & Kai operator definitions, packed quantized weight formats,
//! canonical graph IR, and exact fixed-point SiLU LUT specification.

use crate::commit::{commit_inputs, commit_op_graph, commit_weights, Digest};
use crate::error::{DetError, DetResult};
use crate::profile::{
    narrow_i128_to_i32, rescale_i128, round_shift_right_i128, DyadicScale, EligibleOp,
    ExecutionProfile, IneligibleFloatingPointOp, OperatorId, OverflowMode, RoundingMode,
};
use crate::wire::{WireWriter, WIRE_MAGIC};

/// Block size for packed low-bit quantized weights (`32` weights/block,
/// aligned with `hanzo-quant` GGML `Q4_0` / `Q8_0` conventions).
pub const QUANT_BLOCK_SIZE: usize = 32;

/// Canonical versioned Q15 Sigmoid lookup table (`v1`) over `[-8.0, +8.0]` in
/// Q12 input space (`[-32768, +32768]`, step `256 = 1/16`, `257` knots).
///
/// Each entry equals `round(32768 / (1 + exp(-(-8.0 + i / 16.0))))`, verified
/// monotonically non-decreasing and antisymmetric (`LUT[i] + LUT[256 - i] == 32768`).
/// Runtime evaluation uses strictly integer arithmetic (zero floating-point ops).
pub const SIGMOID_LUT_Q15: [u16; 257] = [
    11, 12, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 28, 30, 32, 34, 36, 38, 41, 43, 46,
    49, 52, 56, 59, 63, 67, 72, 76, 81, 86, 92, 98, 104, 111, 118, 125, 133, 142, 151, 161, 171,
    182, 194, 206, 219, 233, 248, 264, 281, 299, 318, 338, 360, 383, 407, 433, 461, 490, 521, 554,
    589, 627, 666, 708, 753, 800, 851, 904, 961, 1021, 1084, 1152, 1223, 1299, 1379, 1464, 1554,
    1649, 1750, 1856, 1969, 2088, 2213, 2346, 2486, 2633, 2789, 2952, 3124, 3306, 3496, 3696, 3906,
    4126, 4357, 4599, 4851, 5115, 5391, 5678, 5978, 6289, 6613, 6949, 7297, 7658, 8031, 8416, 8813,
    9221, 9641, 10072, 10513, 10964, 11424, 11894, 12371, 12856, 13348, 13845, 14347, 14852, 15361,
    15872, 16384, 16896, 17407, 17916, 18421, 18923, 19420, 19912, 20397, 20874, 21344, 21804,
    22255, 22696, 23127, 23547, 23955, 24352, 24737, 25110, 25471, 25819, 26155, 26479, 26790,
    27090, 27377, 27653, 27917, 28169, 28411, 28642, 28862, 29072, 29272, 29462, 29644, 29816,
    29979, 30135, 30282, 30422, 30555, 30680, 30799, 30912, 31018, 31119, 31214, 31304, 31389,
    31469, 31545, 31616, 31684, 31747, 31807, 31864, 31917, 31968, 32015, 32060, 32102, 32141,
    32179, 32214, 32247, 32278, 32307, 32335, 32361, 32385, 32408, 32430, 32450, 32469, 32487,
    32504, 32520, 32535, 32549, 32562, 32574, 32586, 32597, 32607, 32617, 32626, 32635, 32643,
    32650, 32657, 32664, 32670, 32676, 32682, 32687, 32692, 32696, 32701, 32705, 32709, 32712,
    32716, 32719, 32722, 32725, 32727, 32730, 32732, 32734, 32736, 32738, 32740, 32742, 32743,
    32745, 32746, 32747, 32749, 32750, 32751, 32752, 32753, 32754, 32755, 32756, 32756, 32757,
];

/// Exact zero-float fixed-point SiLU (`x * sigmoid(x)`) for a single Q12 element.
#[inline]
pub fn fixed_point_silu_elem(
    x: i32,
    rounding: RoundingMode,
    overflow: OverflowMode,
) -> DetResult<i32> {
    if x <= -32768 {
        return Ok(0);
    }
    if x >= 32768 {
        return Ok(x);
    }
    let offset = (x + 32768) as u32;
    let idx = (offset >> 8) as usize;
    let frac = i128::from(offset & 0xFF);
    let y0 = i128::from(SIGMOID_LUT_Q15[idx]);
    let y1 = i128::from(SIGMOID_LUT_Q15[idx + 1]);
    let delta = round_shift_right_i128((y1 - y0) * frac, 8, rounding)?;
    let sig_q15 = y0 + delta;
    let prod = i128::from(x) * sig_q15;
    let out = round_shift_right_i128(prod, 15, rounding)?;
    narrow_i128_to_i32(out, overflow, "FixedPointSiLU_LUT")
}

/// Narrow an `i128` value to `i8` under the profile's declared `OverflowMode`.
#[inline]
pub fn narrow_i128_to_i8(val: i128, overflow: OverflowMode, op: &'static str) -> DetResult<i8> {
    match overflow {
        OverflowMode::Saturating => Ok(val.clamp(i128::from(i8::MIN), i128::from(i8::MAX)) as i8),
        OverflowMode::CheckedError => i8::try_from(val).map_err(|_| DetError::ArithmeticOverflow {
            op,
            detail: format!("value {val} out of i8 range"),
        }),
    }
}

/// Deterministically requantize an `i32` activation tensor into `i8` using an
/// explicit rational dyadic scale, rounding policy, and overflow policy.
pub fn requantize_i32_to_i8(
    src: &Int32Tensor,
    scale: DyadicScale,
    rounding: RoundingMode,
    overflow: OverflowMode,
) -> DetResult<Int8Tensor> {
    src.validate()?;
    scale.validate()?;
    let mut data = Vec::with_capacity(src.data.len());
    for &v in &src.data {
        let scaled = rescale_i128(i128::from(v), scale, rounding, "requantize_i32_to_i8")?;
        data.push(narrow_i128_to_i8(scaled, overflow, "requantize_i32_to_i8")?);
    }
    Int8Tensor::new(src.rows, src.cols, data)
}

/// Row-major 2D signed `i8` matrix `[rows, cols]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int8Tensor {
    /// Number of rows (`M`).
    pub rows: usize,
    /// Number of columns (`K` or `N`).
    pub cols: usize,
    /// Row-major signed 8-bit elements (`rows * cols`).
    pub data: Vec<i8>,
}

impl Int8Tensor {
    /// Construct and validate a row-major `Int8Tensor`.
    pub fn new(rows: usize, cols: usize, data: Vec<i8>) -> DetResult<Self> {
        let tensor = Self { rows, cols, data };
        tensor.validate()?;
        Ok(tensor)
    }

    /// Validate that shape dimensions are non-zero, do not overflow, and match `data.len()`.
    pub fn validate(&self) -> DetResult<()> {
        let expected = checked_elem_count(self.rows, self.cols, "Int8Tensor::validate")?;
        if self.data.len() != expected {
            return Err(DetError::InvalidDimensions {
                context: "Int8Tensor::validate",
                expected: format!("{}x{} = {expected} elements", self.rows, self.cols),
                actual: format!("{} elements", self.data.len()),
            });
        }
        Ok(())
    }

    /// Encode into canonical little-endian wire bytes.
    pub fn encode_into(&self, w: &mut WireWriter) {
        w.write_u8(1); // Int8 tag
        w.write_u32(self.rows as u32);
        w.write_u32(self.cols as u32);
        w.write_i8_slice(&self.data);
    }
}

/// Row-major 2D signed `i32` matrix `[rows, cols]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int32Tensor {
    /// Number of rows (`M`).
    pub rows: usize,
    /// Number of columns (`N`).
    pub cols: usize,
    /// Row-major signed 32-bit elements (`rows * cols`).
    pub data: Vec<i32>,
}

impl Int32Tensor {
    /// Construct and validate a row-major `Int32Tensor`.
    pub fn new(rows: usize, cols: usize, data: Vec<i32>) -> DetResult<Self> {
        let tensor = Self { rows, cols, data };
        tensor.validate()?;
        Ok(tensor)
    }

    /// Validate that shape dimensions are non-zero, do not overflow, and match `data.len()`.
    pub fn validate(&self) -> DetResult<()> {
        let expected = checked_elem_count(self.rows, self.cols, "Int32Tensor::validate")?;
        if self.data.len() != expected {
            return Err(DetError::InvalidDimensions {
                context: "Int32Tensor::validate",
                expected: format!("{}x{} = {expected} elements", self.rows, self.cols),
                actual: format!("{} elements", self.data.len()),
            });
        }
        Ok(())
    }

    /// Encode into canonical little-endian wire bytes.
    pub fn encode_into(&self, w: &mut WireWriter) {
        w.write_u8(2); // Int32 tag
        w.write_u32(self.rows as u32);
        w.write_u32(self.cols as u32);
        w.write_i32_slice(&self.data);
    }
}

/// Single 32-weight packed 4-bit block (`18` bytes: `i16` block scale + `16` bytes
/// split-half nibbles), matching `hanzo-quant` GGML `Q4_0` block geometry without
/// floating-point scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedInt4Block {
    /// Exact signed integer per-block scale factor.
    pub scale: i16,
    /// 16 packed bytes holding 32 signed 4-bit weights in `[-8, +7]`.
    /// Byte `t` (`0..16`) holds weight `t` in low nibble `(q + 8)` and weight
    /// `t + 16` in high nibble `(q + 8) << 4`.
    pub qs: [u8; QUANT_BLOCK_SIZE / 2],
}

impl PackedInt4Block {
    /// Pack 32 signed weights (each strictly in `-8..=7`) and an `i16` block scale.
    pub fn pack(scale: i16, weights: &[i8; QUANT_BLOCK_SIZE]) -> DetResult<Self> {
        let mut qs = [0u8; QUANT_BLOCK_SIZE / 2];
        for t in 0..(QUANT_BLOCK_SIZE / 2) {
            let lo = weights[t];
            let hi = weights[t + QUANT_BLOCK_SIZE / 2];
            if !(-8..=7).contains(&lo) || !(-8..=7).contains(&hi) {
                return Err(DetError::InvalidPackedWeight {
                    context: "PackedInt4Block::pack",
                    detail: format!("int4 weight out of [-8, 7]: lo={lo}, hi={hi}"),
                });
            }
            let lo_nib = (lo + 8) as u8;
            let hi_nib = (hi + 8) as u8;
            qs[t] = lo_nib | (hi_nib << 4);
        }
        Ok(Self { scale, qs })
    }

    /// Unpack weight at local block offset `idx in 0..32` into `[-8, +7]`.
    #[inline]
    pub const fn unpack(&self, idx: usize) -> i8 {
        if idx < QUANT_BLOCK_SIZE / 2 {
            ((self.qs[idx] & 0x0F) as i8) - 8
        } else {
            ((self.qs[idx - QUANT_BLOCK_SIZE / 2] >> 4) as i8) - 8
        }
    }
}

/// Column-blocked packed 4-bit weight matrix `[k, n]` (`k % 32 == 0`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedInt4Weight {
    /// Inner reduction dimension `K` (must be a positive multiple of 32).
    pub k: usize,
    /// Output channel count `N` (`> 0`).
    pub n: usize,
    /// Column-major blocks: column `j` occupies `blocks[j * num_blocks .. (j + 1) * num_blocks]`.
    pub blocks: Vec<PackedInt4Block>,
}

impl PackedInt4Weight {
    /// Construct and validate a column-blocked `PackedInt4Weight`.
    pub fn new(k: usize, n: usize, blocks: Vec<PackedInt4Block>) -> DetResult<Self> {
        let weight = Self { k, n, blocks };
        weight.validate()?;
        Ok(weight)
    }

    /// Validate that `k` is a positive multiple of `QUANT_BLOCK_SIZE`, `n > 0`,
    /// and `blocks.len() == n * (k / QUANT_BLOCK_SIZE)`.
    pub fn validate(&self) -> DetResult<()> {
        if self.k == 0 || self.n == 0 || !self.k.is_multiple_of(QUANT_BLOCK_SIZE) {
            return Err(DetError::InvalidDimensions {
                context: "PackedInt4Weight::validate",
                expected: format!("k > 0, k % {QUANT_BLOCK_SIZE} == 0, n > 0"),
                actual: format!("k={}, n={}", self.k, self.n),
            });
        }
        let blocks_per_col = self.k / QUANT_BLOCK_SIZE;
        let expected = checked_elem_count(self.n, blocks_per_col, "PackedInt4Weight::validate")?;
        if self.blocks.len() != expected {
            return Err(DetError::InvalidPackedWeight {
                context: "PackedInt4Weight::validate",
                detail: format!("expected {expected} blocks, got {}", self.blocks.len()),
            });
        }
        Ok(())
    }

    /// Number of 32-element blocks along `K`.
    #[inline]
    pub const fn blocks_per_col(&self) -> usize {
        self.k / QUANT_BLOCK_SIZE
    }

    /// Borrow all blocks for output column `col`.
    #[inline]
    pub fn col_blocks(&self, col: usize) -> &[PackedInt4Block] {
        let bpc = self.blocks_per_col();
        &self.blocks[col * bpc..(col + 1) * bpc]
    }

    /// Encode into canonical little-endian wire bytes.
    pub fn encode_into(&self, w: &mut WireWriter) {
        w.write_u8(3); // PackedInt4Weight tag
        w.write_u32(self.k as u32);
        w.write_u32(self.n as u32);
        w.write_u32(self.blocks.len() as u32);
        for blk in &self.blocks {
            w.write_i16(blk.scale);
            w.write_raw(&blk.qs);
        }
    }
}

/// Deterministic ArgMax / Top-K entry `(index, value)` with canonical total order:
/// higher `value` wins; ties break strictly toward smaller `index`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankedEntry {
    /// Zero-based column/token index.
    pub index: u32,
    /// Signed fixed-point logit value.
    pub value: i32,
}

impl RankedEntry {
    /// Canonical deterministic comparator: descending `value`, ascending `index` on ties.
    #[inline]
    pub fn cmp_deterministic(&self, other: &Self) -> std::cmp::Ordering {
        other
            .value
            .cmp(&self.value)
            .then_with(|| self.index.cmp(&other.index))
    }

    /// Associative binary reduction operator for `ArgMax_DeterministicTieBreak`.
    #[inline]
    pub fn better_of(a: Self, b: Self) -> Self {
        if a.value > b.value || (a.value == b.value && a.index <= b.index) {
            a
        } else {
            b
        }
    }
}

/// Deterministic Top-K output for a batch of `rows` logit vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopKOutput {
    /// Batch size (`rows`).
    pub rows: usize,
    /// Number of top entries per row (`k`).
    pub k: usize,
    /// Flat row-major entries (`rows * k`), each row strictly sorted by
    /// `(value desc, index asc)`.
    pub entries: Vec<RankedEntry>,
}

/// Canonical input activation tensor for an `OpGraph`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphInput {
    /// Signed 8-bit activation tensor `[M, K]`.
    Int8(Int8Tensor),
    /// Signed 32-bit fixed-point activation tensor `[M, D]`.
    Int32(Int32Tensor),
}

impl GraphInput {
    /// Validate that the underlying tensor satisfies shape and storage invariants.
    pub fn validate(&self) -> DetResult<()> {
        match self {
            Self::Int8(t) => t.validate(),
            Self::Int32(t) => t.validate(),
        }
    }

    /// Canonical little-endian byte representation for input commitments.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut w = WireWriter::new();
        w.write_raw(&WIRE_MAGIC);
        match self {
            Self::Int8(t) => t.encode_into(&mut w),
            Self::Int32(t) => t.encode_into(&mut w),
        }
        w.into_bytes()
    }

    /// Domain-separated SHA-256 commitment of this input tensor.
    pub fn digest(&self) -> Digest {
        commit_inputs(&self.to_canonical_bytes())
    }
}

/// Canonical output produced by executing an `OpGraph`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionOutput {
    /// Signed 32-bit tensor output `[rows, cols]`.
    Int32(Int32Tensor),
    /// Per-row deterministic ArgMax winners (`rows` entries).
    ArgMax(Vec<RankedEntry>),
    /// Per-row deterministic Top-K winners (`rows * k` entries).
    TopK(TopKOutput),
}

impl ExecutionOutput {
    /// Canonical little-endian byte encoding bound into `ExecutionReceipt`.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut w = WireWriter::new();
        w.write_raw(&WIRE_MAGIC);
        match self {
            Self::Int32(t) => {
                w.write_u8(0x10);
                t.encode_into(&mut w);
            }
            Self::ArgMax(entries) => {
                w.write_u8(0x11);
                w.write_u32(entries.len() as u32);
                for e in entries {
                    w.write_u32(e.index);
                    w.write_i32(e.value);
                }
            }
            Self::TopK(tk) => {
                w.write_u8(0x12);
                w.write_u32(tk.rows as u32);
                w.write_u32(tk.k as u32);
                w.write_u32(tk.entries.len() as u32);
                for e in &tk.entries {
                    w.write_u32(e.index);
                    w.write_i32(e.value);
                }
            }
        }
        w.into_bytes()
    }
}

/// Single node inside a deterministic Zen/Kai `OpGraph`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpNode {
    /// Exact `i8 x i8 -> i64/i128` GEMM with rational dyadic output scaling.
    /// If preceded by an `Int32` activation stage, `input_requant` specifies
    /// the deterministic dyadic scale converting `i32 -> i8`.
    Int8Gemm {
        /// Row-major `i8` weight matrix `[K, N]`.
        weights: Int8Tensor,
        /// Exact rational dyadic output scale.
        output_scale: DyadicScale,
        /// Optional dyadic requantization scale when consuming an `Int32` tensor.
        input_requant: Option<DyadicScale>,
    },
    /// Exact `i8 x packed-i4` block-quantized GEMM with rational dyadic scaling.
    PackedInt4Gemm {
        /// Column-blocked packed 4-bit weight matrix `[K, N]` (`K % 32 == 0`).
        weights: PackedInt4Weight,
        /// Exact rational dyadic output scale.
        output_scale: DyadicScale,
        /// Optional dyadic requantization scale when consuming an `Int32` tensor.
        input_requant: Option<DyadicScale>,
    },
    /// Exact `i32` bias addition broadcast over rows `[M, N] + [N]`.
    Int32BiasAdd {
        /// Per-column bias vector `[N]`.
        bias: Vec<i32>,
    },
    /// Exact fixed-point ReLU (`max(0, x)`, optionally capped at `clamp_max`).
    FixedPointReLU {
        /// Optional inclusive upper bound (e.g. Q12 ReLU6 = `24576`).
        clamp_max: Option<i32>,
    },
    /// Exact Q12 fixed-point SiLU via canonical `SIGMOID_LUT_Q15`.
    FixedPointSiLULut,
    /// Exact `i64`/`i128` RMSNorm over each row with Q12 `gamma` vector.
    FixedPointRmsNormInt64 {
        /// Per-feature Q12 scale vector `[D]` (`1.0 == 4096`).
        gamma: Vec<i32>,
    },
    /// Deterministic ArgMax over each row with lowest-index tie-breaking.
    ArgMaxDeterministicTieBreak,
    /// Deterministic Top-K over each row sorted by `(value desc, index asc)`.
    TopKDeterministicOrder {
        /// Number of top elements to retain (`1 <= k <= cols`).
        k: usize,
    },
    /// Explicitly ineligible floating-point operator sentinel.
    /// Any graph containing this node is ALWAYS rejected with
    /// `DetError::UnsupportedFloatingPointOp`.
    UnsupportedFloatingPoint(IneligibleFloatingPointOp),
}

impl OpNode {
    /// Return the unified `OperatorId` for eligibility gate checks.
    pub const fn operator_id(&self) -> OperatorId {
        match self {
            Self::Int8Gemm { .. } => OperatorId::Eligible(EligibleOp::Int8Gemm),
            Self::PackedInt4Gemm { .. } => OperatorId::Eligible(EligibleOp::PackedInt4Gemm),
            Self::Int32BiasAdd { .. } => OperatorId::Eligible(EligibleOp::Int32BiasAdd),
            Self::FixedPointReLU { .. } => OperatorId::Eligible(EligibleOp::FixedPointReLU),
            Self::FixedPointSiLULut => OperatorId::Eligible(EligibleOp::FixedPointSiLULut),
            Self::FixedPointRmsNormInt64 { .. } => {
                OperatorId::Eligible(EligibleOp::FixedPointRmsNormInt64)
            }
            Self::ArgMaxDeterministicTieBreak => {
                OperatorId::Eligible(EligibleOp::ArgMaxDeterministicTieBreak)
            }
            Self::TopKDeterministicOrder { .. } => {
                OperatorId::Eligible(EligibleOp::TopKDeterministicOrder)
            }
            Self::UnsupportedFloatingPoint(fp) => OperatorId::IneligibleFloatingPoint(*fp),
        }
    }

    /// Encode graph structural metadata (operator tags, shapes, scales — excluding bulk weights).
    fn encode_structure_into(&self, w: &mut WireWriter) {
        w.write_u8(self.operator_id().to_u8());
        match self {
            Self::Int8Gemm {
                weights,
                output_scale,
                input_requant,
            } => {
                w.write_u32(weights.rows as u32);
                w.write_u32(weights.cols as u32);
                encode_scale(w, *output_scale);
                encode_opt_scale(w, *input_requant);
            }
            Self::PackedInt4Gemm {
                weights,
                output_scale,
                input_requant,
            } => {
                w.write_u32(weights.k as u32);
                w.write_u32(weights.n as u32);
                encode_scale(w, *output_scale);
                encode_opt_scale(w, *input_requant);
            }
            Self::Int32BiasAdd { bias } => {
                w.write_u32(bias.len() as u32);
            }
            Self::FixedPointReLU { clamp_max } => match clamp_max {
                Some(cap) => {
                    w.write_u8(1);
                    w.write_i32(*cap);
                }
                None => w.write_u8(0),
            },
            Self::FixedPointSiLULut | Self::ArgMaxDeterministicTieBreak => {}
            Self::FixedPointRmsNormInt64 { gamma } => {
                w.write_u32(gamma.len() as u32);
            }
            Self::TopKDeterministicOrder { k } => {
                w.write_u32(*k as u32);
            }
            Self::UnsupportedFloatingPoint(fp) => {
                w.write_u8(fp.to_u8());
            }
        }
    }

    /// Encode node weight/parameter payload into canonical weight commitment stream.
    fn encode_weights_into(&self, w: &mut WireWriter) {
        w.write_u8(self.operator_id().to_u8());
        match self {
            Self::Int8Gemm { weights, .. } => weights.encode_into(w),
            Self::PackedInt4Gemm { weights, .. } => weights.encode_into(w),
            Self::Int32BiasAdd { bias } => w.write_i32_slice(bias),
            Self::FixedPointRmsNormInt64 { gamma } => w.write_i32_slice(gamma),
            Self::FixedPointReLU { .. }
            | Self::FixedPointSiLULut
            | Self::ArgMaxDeterministicTieBreak
            | Self::TopKDeterministicOrder { .. }
            | Self::UnsupportedFloatingPoint(_) => {}
        }
    }
}

/// Ordered sequence of deterministic Zen/Kai operators forming an executable graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpGraph {
    /// Human-readable graph label (bound into graph commitment).
    pub name: String,
    /// Ordered operator nodes executed sequentially.
    pub nodes: Vec<OpNode>,
}

impl OpGraph {
    /// Construct a new `OpGraph`.
    pub fn new(name: impl Into<String>, nodes: Vec<OpNode>) -> Self {
        Self {
            name: name.into(),
            nodes,
        }
    }

    /// Validate profile compatibility, explicit floating-point exclusion gate,
    /// dyadic scale bounds, and structural graph invariants.
    pub fn validate(&self, profile: &ExecutionProfile) -> DetResult<()> {
        profile.validate()?;
        if self.nodes.is_empty() {
            return Err(DetError::InvalidGraph {
                detail: "operator graph must contain at least one node".to_string(),
            });
        }
        for (idx, node) in self.nodes.iter().enumerate() {
            // Enforce explicit unsupported floating-point exclusion gate first.
            let _eligible = profile.validate_operator(node.operator_id())?;

            let is_last = idx + 1 == self.nodes.len();
            match node {
                OpNode::Int8Gemm {
                    weights,
                    output_scale,
                    input_requant,
                } => {
                    weights.validate()?;
                    output_scale.validate()?;
                    if let Some(req) = input_requant {
                        req.validate()?;
                    }
                }
                OpNode::PackedInt4Gemm {
                    weights,
                    output_scale,
                    input_requant,
                } => {
                    weights.validate()?;
                    output_scale.validate()?;
                    if let Some(req) = input_requant {
                        req.validate()?;
                    }
                }
                OpNode::Int32BiasAdd { bias } => {
                    if bias.is_empty() {
                        return Err(DetError::InvalidDimensions {
                            context: "Int32BiasAdd",
                            expected: "non-empty bias vector".to_string(),
                            actual: "0".to_string(),
                        });
                    }
                }
                OpNode::FixedPointReLU { clamp_max } => {
                    if let Some(cap) = clamp_max {
                        if *cap < 0 {
                            return Err(DetError::InvalidGraph {
                                detail: format!("FixedPointReLU clamp_max must be >= 0, got {cap}"),
                            });
                        }
                    }
                }
                OpNode::FixedPointRmsNormInt64 { gamma } => {
                    if gamma.is_empty() {
                        return Err(DetError::InvalidDimensions {
                            context: "FixedPointRMSNorm_Int64",
                            expected: "non-empty gamma vector".to_string(),
                            actual: "0".to_string(),
                        });
                    }
                }
                OpNode::ArgMaxDeterministicTieBreak => {
                    if !is_last {
                        return Err(DetError::InvalidGraph {
                            detail: "ArgMax_DeterministicTieBreak must be a terminal node"
                                .to_string(),
                        });
                    }
                }
                OpNode::TopKDeterministicOrder { k } => {
                    if *k == 0 {
                        return Err(DetError::InvalidDimensions {
                            context: "TopK_DeterministicOrder",
                            expected: "k >= 1".to_string(),
                            actual: "0".to_string(),
                        });
                    }
                    if !is_last {
                        return Err(DetError::InvalidGraph {
                            detail: "TopK_DeterministicOrder must be a terminal node".to_string(),
                        });
                    }
                }
                OpNode::FixedPointSiLULut => {}
                OpNode::UnsupportedFloatingPoint(_) => unreachable!("rejected above"),
            }
        }
        Ok(())
    }

    /// Canonical binary encoding of the graph structure (operator sequence,
    /// shapes, and dyadic scale parameters).
    pub fn canonical_structure_bytes(&self) -> Vec<u8> {
        let mut w = WireWriter::new();
        w.write_raw(&WIRE_MAGIC);
        w.write_u8(0x20);
        w.write_str(&self.name);
        w.write_u32(self.nodes.len() as u32);
        for node in &self.nodes {
            node.encode_structure_into(&mut w);
        }
        w.into_bytes()
    }

    /// Canonical binary encoding of all quantized weights, biases, and norm
    /// gammas contained in this graph.
    pub fn canonical_weights_bytes(&self) -> Vec<u8> {
        let mut w = WireWriter::new();
        w.write_raw(&WIRE_MAGIC);
        w.write_u8(0x21);
        w.write_u32(self.nodes.len() as u32);
        for node in &self.nodes {
            node.encode_weights_into(&mut w);
        }
        w.into_bytes()
    }

    /// Domain-separated SHA-256 commitment of the graph structure.
    pub fn structure_digest(&self) -> Digest {
        commit_op_graph(&self.canonical_structure_bytes())
    }

    /// Domain-separated SHA-256 commitment of all weights in the graph.
    pub fn weights_digest(&self) -> Digest {
        commit_weights(&self.canonical_weights_bytes())
    }
}

fn encode_scale(w: &mut WireWriter, scale: DyadicScale) {
    w.write_i32(scale.multiplier);
    w.write_u8(scale.shift);
}

fn encode_opt_scale(w: &mut WireWriter, scale: Option<DyadicScale>) {
    match scale {
        Some(s) => {
            w.write_u8(1);
            encode_scale(w, s);
        }
        None => w.write_u8(0),
    }
}

fn checked_elem_count(rows: usize, cols: usize, context: &'static str) -> DetResult<usize> {
    if rows == 0 || cols == 0 {
        return Err(DetError::InvalidDimensions {
            context,
            expected: "rows > 0 and cols > 0".to_string(),
            actual: format!("rows={rows}, cols={cols}"),
        });
    }
    rows.checked_mul(cols)
        .ok_or_else(|| DetError::InvalidDimensions {
            context,
            expected: "rows * cols <= usize::MAX".to_string(),
            actual: format!("{rows} * {cols} overflows"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_lut_is_monotone_and_symmetric() {
        for i in 0..256 {
            assert!(SIGMOID_LUT_Q15[i] <= SIGMOID_LUT_Q15[i + 1]);
        }
        for i in 0..257 {
            assert_eq!(
                u32::from(SIGMOID_LUT_Q15[i]) + u32::from(SIGMOID_LUT_Q15[256 - i]),
                32768
            );
        }
        assert_eq!(SIGMOID_LUT_Q15[128], 16384);
    }

    #[test]
    fn fixed_point_silu_handles_extremes_and_zero() {
        assert_eq!(
            fixed_point_silu_elem(0, RoundingMode::RoundHalfToEven, OverflowMode::CheckedError)
                .unwrap(),
            0
        );
        // x = +1.0 (4096): sigmoid(1.0) = 0.7310585786 -> 4096 * 0.7310586 ~= 2994.4 -> 2994
        let y1 = fixed_point_silu_elem(
            4096,
            RoundingMode::RoundHalfToEven,
            OverflowMode::CheckedError,
        )
        .unwrap();
        assert!((2993..=2996).contains(&y1), "unexpected SiLU(1.0)={y1}");
        // Extremes must not overflow or panic
        assert_eq!(
            fixed_point_silu_elem(
                i32::MIN,
                RoundingMode::RoundHalfToEven,
                OverflowMode::CheckedError
            )
            .unwrap(),
            0
        );
        assert_eq!(
            fixed_point_silu_elem(
                i32::MAX,
                RoundingMode::RoundHalfToEven,
                OverflowMode::CheckedError
            )
            .unwrap(),
            i32::MAX
        );
    }

    #[test]
    fn packed_int4_roundtrip_all_valid_values() {
        let mut weights = [0i8; QUANT_BLOCK_SIZE];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = ((i % 16) as i8) - 8; // spans -8..=7
        }
        let blk = PackedInt4Block::pack(-123, &weights).unwrap();
        for (i, &expected) in weights.iter().enumerate() {
            assert_eq!(blk.unpack(i), expected);
        }
        let mut bad = weights;
        bad[0] = 8;
        assert!(matches!(
            PackedInt4Block::pack(1, &bad),
            Err(DetError::InvalidPackedWeight { .. })
        ));
    }
}
