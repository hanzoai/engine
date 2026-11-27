// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Clean-room bit-exact quantization formats and packed storage kernels.
//!
//! Implements:
//! - [`QuantFormat::Ternary`]: BitNet-b1.58 style 1.58-bit trits `{-1, 0, +1}` packed 4 trits/byte
//!   with per-group absmean scales.
//! - [`QuantFormat::Int4Sym`]: Per-group symmetric 4-bit integers `[-7, +7]` packed 2 nibbles/byte.
//! - [`QuantFormat::Int4Asym`]: Per-group asymmetric affine 4-bit integers `[0, 15]` packed 2 nibbles/byte.
//! - [`QuantFormat::Int8`]: Per-channel / per-group symmetric 8-bit integers `[-127, +127]`.
//! - [`QuantFormat::Bf16`]: Exact IEEE-754 Round-to-Nearest-Even (RNE) BF16 (`u16`).

use crate::error::MixedQuantError;
use crate::rotation::RotationTransform;

/// Quantization format assignable per layer / projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QuantFormat {
    /// 1.58-bit ternary `{-1, 0, +1}` packed 4 trits per byte (2 bits/trit) with per-group absmean scale.
    Ternary,
    /// 4-bit symmetric `[-7, +7]` packed 2 nibbles per byte with per-group scale.
    Int4Sym,
    /// 4-bit asymmetric `[0, 15]` packed 2 nibbles per byte with per-group scale & zero-point.
    Int4Asym,
    /// 8-bit symmetric `[-127, +127]` with per-channel / per-group scale.
    Int8,
    /// Exact IEEE-754 BF16 (`u16`) with Round-to-Nearest-Even.
    Bf16,
}

impl QuantFormat {
    /// Canonical identifier string for serialization and reporting.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ternary => "ternary_1.58b",
            Self::Int4Sym => "int4_sym",
            Self::Int4Asym => "int4_asym",
            Self::Int8 => "int8",
            Self::Bf16 => "bf16",
        }
    }

    /// Nominal payload bits per weight (excluding metadata scales/zero-points).
    #[must_use]
    pub const fn nominal_bits(self) -> f64 {
        match self {
            Self::Ternary => 1.585,
            Self::Int4Sym | Self::Int4Asym => 4.0,
            Self::Int8 => 8.0,
            Self::Bf16 => 16.0,
        }
    }

    /// Exact packed storage bytes for a `[rows, cols]` weight matrix under this format and `group_size`.
    ///
    /// Includes packed weight payload plus per-group metadata (BF16/FP32 scales & zero-points).
    #[must_use]
    pub fn exact_storage_bytes(self, rows: usize, cols: usize, group_size: usize) -> usize {
        let g = group_size.max(1);
        let groups_per_row = cols.div_ceil(g);
        match self {
            Self::Ternary => {
                // 4 trits per byte (2 bits/trit packed) + 1 f32 absmean scale per group
                let packed_row = cols.div_ceil(4);
                rows * packed_row + rows * groups_per_row * 4
            }
            Self::Int4Sym => {
                // 2 nibbles per byte + 1 f32 scale per group
                let packed_row = cols.div_ceil(2);
                rows * packed_row + rows * groups_per_row * 4
            }
            Self::Int4Asym => {
                // 2 nibbles per byte + (1 f32 scale + 1 f32 zero_point) per group
                let packed_row = cols.div_ceil(2);
                rows * packed_row + rows * groups_per_row * 8
            }
            Self::Int8 => {
                // 1 byte per weight + 1 f32 scale per group (or per-channel when g >= cols)
                rows * cols + rows * groups_per_row * 4
            }
            Self::Bf16 => rows * cols * 2,
        }
    }

    /// Exact effective bits per weight on large matrices (`4B/9B/27B` scale), assuming FP16/BF16 group metadata
    /// (matching production GGUF/AWQ/BitNet block headers).
    #[must_use]
    pub fn effective_bits_production(self, group_size: usize) -> f64 {
        let g = group_size.max(1) as f64;
        match self {
            // Packed 2-bit trits (or 5-trit-per-byte base-3 = 1.60b) + 16-bit group scale
            Self::Ternary => 2.0 + 16.0 / g,
            // 4-bit nibbles + 16-bit group scale
            Self::Int4Sym => 4.0 + 16.0 / g,
            // 4-bit nibbles + 16-bit scale + 16-bit zero-point
            Self::Int4Asym => 4.0 + 32.0 / g,
            // 8-bit int + 16-bit per-group/channel scale
            Self::Int8 => 8.0 + 16.0 / g,
            Self::Bf16 => 16.0,
        }
    }
}

/// Converts `f32` to exact IEEE-754 BF16 (`u16`) using Round-to-Nearest-Even (RNE).
#[inline]
#[must_use]
pub fn f32_to_bf16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let abs_bits = bits & 0x7fff_ffff;
    // Preserve NaN as quiet NaN
    if abs_bits > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    // Preserve +/- Inf exactly
    if abs_bits == 0x7f80_0000 {
        return (bits >> 16) as u16;
    }
    // Round-to-nearest-even: bias = 0x7fff + LSB of upper 16 bits
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff_u32 + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Unpacks an IEEE-754 BF16 (`u16`) bit pattern back to `f32`.
#[inline]
#[must_use]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Sanitizes non-finite floats cleanly to `0.0` before quantization.
#[inline]
fn sanitize(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Packed BitNet-b1.58 ternary weight matrix (`{-1, 0, +1}`, 4 trits/byte + per-group absmean scale).
#[derive(Debug, Clone)]
pub struct PackedTernaryMatrix {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub groups_per_row: usize,
    pub packed_row_bytes: usize,
    /// Packed 2-bit trits: `00 => 0`, `01 => +1`, `11 => -1`.
    pub packed: Vec<u8>,
    /// Per-group absmean FP32 scales `[rows * groups_per_row]`.
    pub scales: Vec<f32>,
}

impl PackedTernaryMatrix {
    /// Quantizes row-major `[rows, cols]` weights into BitNet-b1.58 ternary packed representation.
    pub fn quantize(
        weights: &[f32],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Result<Self, MixedQuantError> {
        validate_matrix(weights, rows, cols, group_size)?;
        let groups_per_row = cols.div_ceil(group_size);
        let packed_row_bytes = cols.div_ceil(4);
        let mut packed = vec![0u8; rows * packed_row_bytes];
        let mut scales = vec![0.0f32; rows * groups_per_row];

        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            let p_row = &mut packed[r * packed_row_bytes..(r + 1) * packed_row_bytes];
            for g in 0..groups_per_row {
                let start = g * group_size;
                let end = (start + group_size).min(cols);
                let len = (end - start) as f32;
                let abs_sum: f32 = row[start..end].iter().map(|&v| sanitize(v).abs()).sum();
                let gamma = abs_sum / len;
                scales[r * groups_per_row + g] = gamma;

                if gamma > 1e-12 {
                    let inv = 1.0 / gamma;
                    for (idx, &raw) in row[start..end].iter().enumerate() {
                        let c = start + idx;
                        let q = (sanitize(raw) * inv).round().clamp(-1.0, 1.0) as i8;
                        let code: u8 = match q {
                            1 => 0b01,
                            -1 => 0b11,
                            _ => 0b00,
                        };
                        p_row[c >> 2] |= code << ((c & 3) * 2);
                    }
                }
            }
        }

        Ok(Self {
            rows,
            cols,
            group_size,
            groups_per_row,
            packed_row_bytes,
            packed,
            scales,
        })
    }

    /// Unpacks a single trit at `(row, col)` into `{-1, 0, +1}`.
    #[inline]
    #[must_use]
    pub fn get_trit(&self, row: usize, col: usize) -> i8 {
        let byte = self.packed[row * self.packed_row_bytes + (col >> 2)];
        let code = (byte >> ((col & 3) * 2)) & 0b11;
        match code {
            0b01 => 1,
            0b11 => -1,
            _ => 0,
        }
    }

    /// Dequantizes the entire matrix into row-major `f32`.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            let p_row = &self.packed[r * self.packed_row_bytes..(r + 1) * self.packed_row_bytes];
            for g in 0..self.groups_per_row {
                let scale = self.scales[r * self.groups_per_row + g];
                let start = g * self.group_size;
                let end = (start + self.group_size).min(self.cols);
                for c in start..end {
                    let code = (p_row[c >> 2] >> ((c & 3) * 2)) & 0b11;
                    out[r * self.cols + c] = match code {
                        0b01 => scale,
                        0b11 => -scale,
                        _ => 0.0,
                    };
                }
            }
        }
        out
    }

    /// Computes `y = W * x` directly over packed ternary trits without full dequantization
    /// using a branchless 4-entry trit LUT, multiplying once per group by `scale`.
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<(), MixedQuantError> {
        const TRIT_LUT: [f32; 4] = [0.0, 1.0, 0.0, -1.0];
        validate_matvec(self.rows, self.cols, x, out)?;
        for (r, out_val) in out.iter_mut().enumerate() {
            let p_row = &self.packed[r * self.packed_row_bytes..(r + 1) * self.packed_row_bytes];
            let mut row_acc = 0.0f32;
            for g in 0..self.groups_per_row {
                let scale = self.scales[r * self.groups_per_row + g];
                if scale == 0.0 {
                    continue;
                }
                let start = g * self.group_size;
                let end = (start + self.group_size).min(self.cols);
                let mut group_sum = 0.0f32;
                let mut c = start;
                while c + 4 <= end && (c & 3) == 0 {
                    let byte = p_row[c >> 2] as usize;
                    group_sum += TRIT_LUT[byte & 0b11] * x[c]
                        + TRIT_LUT[(byte >> 2) & 0b11] * x[c + 1]
                        + TRIT_LUT[(byte >> 4) & 0b11] * x[c + 2]
                        + TRIT_LUT[(byte >> 6) & 0b11] * x[c + 3];
                    c += 4;
                }
                while c < end {
                    let code = ((p_row[c >> 2] >> ((c & 3) * 2)) & 0b11) as usize;
                    group_sum += TRIT_LUT[code] * x[c];
                    c += 1;
                }
                row_acc += group_sum * scale;
            }
            *out_val = row_acc;
        }
        Ok(())
    }

    /// Exact resident storage bytes (`packed` + `scales`).
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        self.packed.len() + self.scales.len() * 4
    }
}

/// Packed 4-bit weight matrix supporting both Symmetric (`[-7, 7]`) and Asymmetric (`[0, 15]`) per-group modes.
#[derive(Debug, Clone)]
pub struct PackedInt4Matrix {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub groups_per_row: usize,
    pub packed_row_bytes: usize,
    pub asymmetric: bool,
    /// Packed 4-bit nibbles: low nibble = even col, high nibble = odd col.
    pub packed: Vec<u8>,
    /// Per-group scale `[rows * groups_per_row]`.
    pub scales: Vec<f32>,
    /// Per-group zero-points `[rows * groups_per_row]` (empty if symmetric).
    pub zero_points: Vec<f32>,
}

impl PackedInt4Matrix {
    /// Quantizes row-major `[rows, cols]` weights into symmetric or asymmetric 4-bit packed storage.
    pub fn quantize(
        weights: &[f32],
        rows: usize,
        cols: usize,
        group_size: usize,
        asymmetric: bool,
    ) -> Result<Self, MixedQuantError> {
        validate_matrix(weights, rows, cols, group_size)?;
        let groups_per_row = cols.div_ceil(group_size);
        let packed_row_bytes = cols.div_ceil(2);
        let mut packed = vec![0u8; rows * packed_row_bytes];
        let mut scales = vec![0.0f32; rows * groups_per_row];
        let mut zero_points = if asymmetric {
            vec![0.0f32; rows * groups_per_row]
        } else {
            Vec::new()
        };

        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            let p_row = &mut packed[r * packed_row_bytes..(r + 1) * packed_row_bytes];
            for g in 0..groups_per_row {
                let start = g * group_size;
                let end = (start + group_size).min(cols);
                let g_idx = r * groups_per_row + g;

                if asymmetric {
                    let mut min_v = f32::INFINITY;
                    let mut max_v = f32::NEG_INFINITY;
                    for &raw in &row[start..end] {
                        let v = sanitize(raw);
                        if v < min_v {
                            min_v = v;
                        }
                        if v > max_v {
                            max_v = v;
                        }
                    }
                    let span = max_v - min_v;
                    let scale = if span > 1e-12 { span / 15.0 } else { 0.0 };
                    scales[g_idx] = scale;
                    zero_points[g_idx] = min_v;
                    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    for (idx, &raw) in row[start..end].iter().enumerate() {
                        let c = start + idx;
                        let q = if scale > 0.0 {
                            ((sanitize(raw) - min_v) * inv).round().clamp(0.0, 15.0) as u8
                        } else {
                            0u8
                        };
                        if (c & 1) == 0 {
                            p_row[c >> 1] |= q & 0x0f;
                        } else {
                            p_row[c >> 1] |= (q & 0x0f) << 4;
                        }
                    }
                } else {
                    let mut max_abs = 0.0f32;
                    for &raw in &row[start..end] {
                        let a = sanitize(raw).abs();
                        if a > max_abs {
                            max_abs = a;
                        }
                    }
                    let scale = if max_abs > 1e-12 { max_abs / 7.0 } else { 0.0 };
                    scales[g_idx] = scale;
                    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    for (idx, &raw) in row[start..end].iter().enumerate() {
                        let c = start + idx;
                        let q_signed = if scale > 0.0 {
                            (sanitize(raw) * inv).round().clamp(-7.0, 7.0) as i8
                        } else {
                            0i8
                        };
                        let nibble = (q_signed + 8) as u8;
                        if (c & 1) == 0 {
                            p_row[c >> 1] |= nibble & 0x0f;
                        } else {
                            p_row[c >> 1] |= (nibble & 0x0f) << 4;
                        }
                    }
                }
            }
        }

        Ok(Self {
            rows,
            cols,
            group_size,
            groups_per_row,
            packed_row_bytes,
            asymmetric,
            packed,
            scales,
            zero_points,
        })
    }

    /// Reads the raw 4-bit nibble `[0..15]` at `(row, col)`.
    #[inline]
    #[must_use]
    pub fn get_nibble(&self, row: usize, col: usize) -> u8 {
        let byte = self.packed[row * self.packed_row_bytes + (col >> 1)];
        if (col & 1) == 0 {
            byte & 0x0f
        } else {
            (byte >> 4) & 0x0f
        }
    }

    /// Dequantizes the entire matrix into row-major `f32`.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            for g in 0..self.groups_per_row {
                let g_idx = r * self.groups_per_row + g;
                let scale = self.scales[g_idx];
                let start = g * self.group_size;
                let end = (start + self.group_size).min(self.cols);
                if self.asymmetric {
                    let zp = self.zero_points[g_idx];
                    for c in start..end {
                        let q = self.get_nibble(r, c) as f32;
                        out[r * self.cols + c] = zp + q * scale;
                    }
                } else {
                    for c in start..end {
                        let q = (self.get_nibble(r, c) as i32 - 8) as f32;
                        out[r * self.cols + c] = q * scale;
                    }
                }
            }
        }
        out
    }

    /// Computes `y = W * x` directly over packed 4-bit nibbles, hoisting group `scale` and `zero_point`.
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<(), MixedQuantError> {
        validate_matvec(self.rows, self.cols, x, out)?;
        for (r, out_val) in out.iter_mut().enumerate() {
            let p_row = &self.packed[r * self.packed_row_bytes..(r + 1) * self.packed_row_bytes];
            let mut row_acc = 0.0f32;
            for g in 0..self.groups_per_row {
                let g_idx = r * self.groups_per_row + g;
                let scale = self.scales[g_idx];
                let start = g * self.group_size;
                let end = (start + self.group_size).min(self.cols);
                let mut qx_acc = 0.0f32;

                if self.asymmetric {
                    let zp = self.zero_points[g_idx];
                    let mut x_sum = 0.0f32;
                    let mut c = start;
                    while c + 2 <= end && (c & 1) == 0 {
                        let b = p_row[c >> 1];
                        let q0 = (b & 0x0f) as f32;
                        let q1 = ((b >> 4) & 0x0f) as f32;
                        let x0 = x[c];
                        let x1 = x[c + 1];
                        qx_acc += q0 * x0 + q1 * x1;
                        x_sum += x0 + x1;
                        c += 2;
                    }
                    while c < end {
                        let q = self.get_nibble(r, c) as f32;
                        qx_acc += q * x[c];
                        x_sum += x[c];
                        c += 1;
                    }
                    row_acc += qx_acc * scale + x_sum * zp;
                } else {
                    if scale == 0.0 {
                        continue;
                    }
                    let mut c = start;
                    while c + 2 <= end && (c & 1) == 0 {
                        let b = p_row[c >> 1];
                        let q0 = ((b & 0x0f) as i32 - 8) as f32;
                        let q1 = (((b >> 4) & 0x0f) as i32 - 8) as f32;
                        qx_acc += q0 * x[c] + q1 * x[c + 1];
                        c += 2;
                    }
                    while c < end {
                        let q = (self.get_nibble(r, c) as i32 - 8) as f32;
                        qx_acc += q * x[c];
                        c += 1;
                    }
                    row_acc += qx_acc * scale;
                }
            }
            *out_val = row_acc;
        }
        Ok(())
    }

    /// Exact resident storage bytes (`packed` + `scales` + `zero_points`).
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        self.packed.len() + self.scales.len() * 4 + self.zero_points.len() * 4
    }
}

/// Packed symmetric 8-bit matrix (`[-127, +127]`, per-channel or per-group FP32 scales).
#[derive(Debug, Clone)]
pub struct PackedInt8Matrix {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub groups_per_row: usize,
    pub packed: Vec<i8>,
    pub scales: Vec<f32>,
}

impl PackedInt8Matrix {
    /// Quantizes row-major `[rows, cols]` weights into symmetric INT8 storage.
    pub fn quantize(
        weights: &[f32],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Result<Self, MixedQuantError> {
        validate_matrix(weights, rows, cols, group_size)?;
        let groups_per_row = cols.div_ceil(group_size);
        let mut packed = vec![0i8; rows * cols];
        let mut scales = vec![0.0f32; rows * groups_per_row];

        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            let p_row = &mut packed[r * cols..(r + 1) * cols];
            for g in 0..groups_per_row {
                let start = g * group_size;
                let end = (start + group_size).min(cols);
                let mut max_abs = 0.0f32;
                for &raw in &row[start..end] {
                    let a = sanitize(raw).abs();
                    if a > max_abs {
                        max_abs = a;
                    }
                }
                let scale = if max_abs > 1e-12 {
                    max_abs / 127.0
                } else {
                    0.0
                };
                scales[r * groups_per_row + g] = scale;
                if scale > 0.0 {
                    let inv = 1.0 / scale;
                    for c in start..end {
                        p_row[c] = (sanitize(row[c]) * inv).round().clamp(-127.0, 127.0) as i8;
                    }
                }
            }
        }

        Ok(Self {
            rows,
            cols,
            group_size,
            groups_per_row,
            packed,
            scales,
        })
    }

    /// Dequantizes into row-major `f32`.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            for g in 0..self.groups_per_row {
                let scale = self.scales[r * self.groups_per_row + g];
                let start = g * self.group_size;
                let end = (start + self.group_size).min(self.cols);
                for c in start..end {
                    out[r * self.cols + c] = self.packed[r * self.cols + c] as f32 * scale;
                }
            }
        }
        out
    }

    /// Computes `y = W * x` directly over packed INT8 bytes.
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<(), MixedQuantError> {
        validate_matvec(self.rows, self.cols, x, out)?;
        for (r, out_val) in out.iter_mut().enumerate() {
            let p_row = &self.packed[r * self.cols..(r + 1) * self.cols];
            let mut row_acc = 0.0f32;
            for g in 0..self.groups_per_row {
                let scale = self.scales[r * self.groups_per_row + g];
                if scale == 0.0 {
                    continue;
                }
                let start = g * self.group_size;
                let end = (start + self.group_size).min(self.cols);
                let mut g_acc = 0.0f32;
                for c in start..end {
                    g_acc += p_row[c] as f32 * x[c];
                }
                row_acc += g_acc * scale;
            }
            *out_val = row_acc;
        }
        Ok(())
    }

    /// Exact resident storage bytes.
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        self.packed.len() + self.scales.len() * 4
    }
}

/// Exact IEEE-754 RNE BF16 packed weight matrix (`Vec<u16>`).
#[derive(Debug, Clone)]
pub struct PackedBf16Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u16>,
}

impl PackedBf16Matrix {
    /// Rounds row-major `[rows, cols]` `f32` weights to exact IEEE-754 RNE BF16.
    pub fn quantize(weights: &[f32], rows: usize, cols: usize) -> Result<Self, MixedQuantError> {
        if rows == 0 || cols == 0 || weights.len() != rows * cols {
            return Err(MixedQuantError::InvalidShape {
                context: "PackedBf16Matrix::quantize",
                expected: rows.saturating_mul(cols),
                actual: weights.len(),
            });
        }
        let data = weights.iter().map(|&v| f32_to_bf16_bits(v)).collect();
        Ok(Self { rows, cols, data })
    }

    /// Unpacks all BF16 weights to `f32`.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        self.data.iter().map(|&b| bf16_bits_to_f32(b)).collect()
    }

    /// Computes `y = W * x` directly over BF16 words.
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<(), MixedQuantError> {
        validate_matvec(self.rows, self.cols, x, out)?;
        for (r, out_val) in out.iter_mut().enumerate() {
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            let mut acc = 0.0f32;
            for c in 0..self.cols {
                acc += bf16_bits_to_f32(row[c]) * x[c];
            }
            *out_val = acc;
        }
        Ok(())
    }

    /// Exact resident storage bytes (`2 * rows * cols`).
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        self.data.len() * 2
    }
}

/// Sparse high-precision (BF16) protection table for the top `~1%` salient outlier input channels (AWQ).
#[derive(Debug, Clone)]
pub struct ProtectedChannels {
    /// Sorted indices of protected input channels.
    pub channel_indices: Vec<usize>,
    /// Exact BF16 weights stored as `[rows, channel_indices.len()]`.
    pub bf16_weights: Vec<u16>,
    pub rows: usize,
}

impl ProtectedChannels {
    /// Exact storage bytes for the protected BF16 columns + index table.
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        self.bf16_weights.len() * 2 + self.channel_indices.len() * 4
    }

    /// Adds `W_protected * x_protected` directly into `out`.
    pub fn accumulate_matvec(&self, x: &[f32], out: &mut [f32]) {
        let k = self.channel_indices.len();
        if k == 0 {
            return;
        }
        for (r, out_val) in out.iter_mut().enumerate().take(self.rows) {
            let w_row = &self.bf16_weights[r * k..(r + 1) * k];
            let mut acc = 0.0f32;
            for (idx, &col) in self.channel_indices.iter().enumerate() {
                acc += bf16_bits_to_f32(w_row[idx]) * x[col];
            }
            *out_val += acc;
        }
    }
}

/// Core packed weight storage variants.
#[derive(Debug, Clone)]
pub enum PackedWeights {
    Ternary(PackedTernaryMatrix),
    Int4(PackedInt4Matrix),
    Int8(PackedInt8Matrix),
    Bf16(PackedBf16Matrix),
}

impl PackedWeights {
    /// Returns the quantization format.
    #[must_use]
    pub fn format(&self) -> QuantFormat {
        match self {
            Self::Ternary(_) => QuantFormat::Ternary,
            Self::Int4(m) => {
                if m.asymmetric {
                    QuantFormat::Int4Asym
                } else {
                    QuantFormat::Int4Sym
                }
            }
            Self::Int8(_) => QuantFormat::Int8,
            Self::Bf16(_) => QuantFormat::Bf16,
        }
    }

    /// Returns `(rows, cols)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::Ternary(m) => (m.rows, m.cols),
            Self::Int4(m) => (m.rows, m.cols),
            Self::Int8(m) => (m.rows, m.cols),
            Self::Bf16(m) => (m.rows, m.cols),
        }
    }

    /// Exact packed storage bytes.
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        match self {
            Self::Ternary(m) => m.storage_bytes(),
            Self::Int4(m) => m.storage_bytes(),
            Self::Int8(m) => m.storage_bytes(),
            Self::Bf16(m) => m.storage_bytes(),
        }
    }

    /// Dequantizes to dense `f32`.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        match self {
            Self::Ternary(m) => m.dequantize(),
            Self::Int4(m) => m.dequantize(),
            Self::Int8(m) => m.dequantize(),
            Self::Bf16(m) => m.dequantize(),
        }
    }

    /// Computes `y = W * x`.
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<(), MixedQuantError> {
        match self {
            Self::Ternary(m) => m.matvec(x, out),
            Self::Int4(m) => m.matvec(x, out),
            Self::Int8(m) => m.matvec(x, out),
            Self::Bf16(m) => m.matvec(x, out),
        }
    }

    /// Quantizes raw `[rows, cols]` weights into the requested format.
    pub fn quantize(
        weights: &[f32],
        rows: usize,
        cols: usize,
        format: QuantFormat,
        group_size: usize,
    ) -> Result<Self, MixedQuantError> {
        match format {
            QuantFormat::Ternary => Ok(Self::Ternary(PackedTernaryMatrix::quantize(
                weights, rows, cols, group_size,
            )?)),
            QuantFormat::Int4Sym => Ok(Self::Int4(PackedInt4Matrix::quantize(
                weights, rows, cols, group_size, false,
            )?)),
            QuantFormat::Int4Asym => Ok(Self::Int4(PackedInt4Matrix::quantize(
                weights, rows, cols, group_size, true,
            )?)),
            QuantFormat::Int8 => Ok(Self::Int8(PackedInt8Matrix::quantize(
                weights, rows, cols, group_size,
            )?)),
            QuantFormat::Bf16 => Ok(Self::Bf16(PackedBf16Matrix::quantize(weights, rows, cols)?)),
        }
    }

    /// Mutable access to per-group scales (used by Quantization-Aware Distillation).
    pub fn scales_mut(&mut self) -> Option<&mut [f32]> {
        match self {
            Self::Ternary(m) => Some(&mut m.scales),
            Self::Int4(m) => Some(&mut m.scales),
            Self::Int8(m) => Some(&mut m.scales),
            Self::Bf16(_) => None,
        }
    }

    /// Folds per-row output calibration scales directly into existing per-group scales
    /// (and zero-points for asymmetric INT4) with **zero extra storage or runtime cost**.
    ///
    /// Returns `true` if folding succeeded (all quantized formats except raw `Bf16`).
    pub fn fold_row_scales(&mut self, row_scales: &[f32]) -> bool {
        match self {
            Self::Ternary(m) => {
                if row_scales.len() != m.rows {
                    return false;
                }
                for (r, &alpha) in row_scales.iter().enumerate() {
                    for g in 0..m.groups_per_row {
                        m.scales[r * m.groups_per_row + g] *= alpha;
                    }
                }
                true
            }
            Self::Int4(m) => {
                if row_scales.len() != m.rows {
                    return false;
                }
                for (r, &alpha) in row_scales.iter().enumerate() {
                    for g in 0..m.groups_per_row {
                        let idx = r * m.groups_per_row + g;
                        m.scales[idx] *= alpha;
                        if m.asymmetric {
                            m.zero_points[idx] *= alpha;
                        }
                    }
                }
                true
            }
            Self::Int8(m) => {
                if row_scales.len() != m.rows {
                    return false;
                }
                for (r, &alpha) in row_scales.iter().enumerate() {
                    for g in 0..m.groups_per_row {
                        m.scales[r * m.groups_per_row + g] *= alpha;
                    }
                }
                true
            }
            Self::Bf16(_) => false,
        }
    }
}

/// Complete quantized linear projection combining packed weights, optional AWQ channel scales,
/// optional BF16 salient-channel protection, and optional SpinQuant orthogonal input rotation.
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    pub weights: PackedWeights,
    /// Optional AWQ inverse input scales `diag(s)^-1` applied to `x` before matvec.
    pub awq_inv_scales: Option<Vec<f32>>,
    /// Optional BF16 protection for the top ~1% salient input channels.
    pub protected_channels: Option<ProtectedChannels>,
    /// Optional SpinQuant orthogonal rotation applied online to input activations `x -> x R`.
    pub input_rotation: Option<RotationTransform>,
    /// Optional post-matvec scalar/per-row QAD output calibration scale.
    pub qad_row_scales: Option<Vec<f32>>,
}

impl QuantizedLinear {
    /// Creates a plain quantized linear layer without AWQ scaling or rotation.
    pub fn quantize_plain(
        weights: &[f32],
        rows: usize,
        cols: usize,
        format: QuantFormat,
        group_size: usize,
    ) -> Result<Self, MixedQuantError> {
        Ok(Self {
            weights: PackedWeights::quantize(weights, rows, cols, format, group_size)?,
            awq_inv_scales: None,
            protected_channels: None,
            input_rotation: None,
            qad_row_scales: None,
        })
    }

    /// Returns `(rows, cols)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.weights.shape()
    }

    /// Returns the primary quantization format.
    #[must_use]
    pub fn format(&self) -> QuantFormat {
        self.weights.format()
    }

    /// Exact total storage bytes including packed weights, AWQ scales, protected BF16 channels,
    /// QAD calibration scales, and owned input-rotation payloads.
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        let mut total = self.weights.storage_bytes();
        if let Some(ref s) = self.awq_inv_scales {
            total += s.len() * 4;
        }
        if let Some(ref p) = self.protected_channels {
            total += p.storage_bytes();
        }
        if let Some(ref rot) = self.input_rotation {
            total += rot.storage_bytes();
        }
        if let Some(ref q) = self.qad_row_scales {
            total += q.len() * 4;
        }
        total
    }

    /// Exact effective bits per weight parameter.
    #[must_use]
    pub fn effective_bits_per_weight(&self) -> f64 {
        let (rows, cols) = self.shape();
        (self.storage_bytes() as f64 * 8.0) / ((rows * cols) as f64)
    }

    /// Runs the full quantized forward matvec `y = W_hat * x` (with SpinQuant rotation, AWQ
    /// channel scaling, protected BF16 channel accumulation, and QAD row calibration).
    ///
    /// Protected salient coordinates are masked to zero before the packed low-bit matvec so
    /// asymmetric affine quantization grids (`Int4Asym`) cannot add zero-point residual error
    /// on top of the exact BF16 protected contribution.
    pub fn forward_vec(&self, x: &[f32], out: &mut [f32]) -> Result<(), MixedQuantError> {
        let (rows, cols) = self.shape();
        validate_matvec(rows, cols, x, out)?;

        let mut x_rot_buf: Option<Vec<f32>> = None;
        if let Some(ref rot) = self.input_rotation {
            let mut buf = x.to_vec();
            rot.apply_in_place(&mut buf)?;
            x_rot_buf = Some(buf);
        }
        let x_base: &[f32] = x_rot_buf.as_deref().unwrap_or(x);

        let mut scratch: Vec<f32>;
        let x_lowbit: &[f32] = if self.awq_inv_scales.is_some() || self.protected_channels.is_some()
        {
            scratch = x_base.to_vec();
            if let Some(ref inv_s) = self.awq_inv_scales {
                for (xi, &inv) in scratch.iter_mut().zip(inv_s.iter()) {
                    *xi *= inv;
                }
            }
            if let Some(ref prot) = self.protected_channels {
                for &c in &prot.channel_indices {
                    if c < scratch.len() {
                        scratch[c] = 0.0;
                    }
                }
            }
            &scratch
        } else {
            x_base
        };

        self.weights.matvec(x_lowbit, out)?;

        if let Some(ref prot) = self.protected_channels {
            prot.accumulate_matvec(x_base, out);
        }

        if let Some(ref row_s) = self.qad_row_scales {
            for (o, &s) in out.iter_mut().zip(row_s.iter()) {
                *o *= s;
            }
        }

        Ok(())
    }

    /// Runs batched forward pass on `batch_len` input vectors `[batch_len * cols] -> [batch_len * rows]`.
    pub fn forward_batch(
        &self,
        inputs: &[f32],
        batch_len: usize,
    ) -> Result<Vec<f32>, MixedQuantError> {
        let (rows, cols) = self.shape();
        if inputs.len() != batch_len * cols {
            return Err(MixedQuantError::InvalidShape {
                context: "QuantizedLinear::forward_batch",
                expected: batch_len * cols,
                actual: inputs.len(),
            });
        }
        let mut out = vec![0.0f32; batch_len * rows];
        for b in 0..batch_len {
            self.forward_vec(
                &inputs[b * cols..(b + 1) * cols],
                &mut out[b * rows..(b + 1) * rows],
            )?;
        }
        Ok(out)
    }
}

/// Exact FP32 reference matrix-vector multiplication `y = W * x`.
pub fn matvec_f32(
    weights: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), MixedQuantError> {
    if weights.len() != rows * cols {
        return Err(MixedQuantError::InvalidShape {
            context: "matvec_f32 weights",
            expected: rows * cols,
            actual: weights.len(),
        });
    }
    validate_matvec(rows, cols, x, out)?;
    for (r, out_val) in out.iter_mut().enumerate() {
        let row = &weights[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        *out_val = acc;
    }
    Ok(())
}

/// Exact FP32 reference batched matmul `Y = X * W^T`.
pub fn matmul_batch_f32(
    weights: &[f32],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    batch_len: usize,
) -> Result<Vec<f32>, MixedQuantError> {
    if inputs.len() != batch_len * cols {
        return Err(MixedQuantError::InvalidShape {
            context: "matmul_batch_f32 inputs",
            expected: batch_len * cols,
            actual: inputs.len(),
        });
    }
    let mut out = vec![0.0f32; batch_len * rows];
    for b in 0..batch_len {
        matvec_f32(
            weights,
            rows,
            cols,
            &inputs[b * cols..(b + 1) * cols],
            &mut out[b * rows..(b + 1) * rows],
        )?;
    }
    Ok(out)
}

fn validate_matrix(
    weights: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Result<(), MixedQuantError> {
    if rows == 0 || cols == 0 || weights.len() != rows * cols {
        return Err(MixedQuantError::InvalidShape {
            context: "quantize matrix",
            expected: rows.saturating_mul(cols),
            actual: weights.len(),
        });
    }
    if group_size == 0 {
        return Err(MixedQuantError::InvalidGroupSize { group_size, cols });
    }
    Ok(())
}

fn validate_matvec(
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &[f32],
) -> Result<(), MixedQuantError> {
    if x.len() != cols {
        return Err(MixedQuantError::InvalidShape {
            context: "matvec input",
            expected: cols,
            actual: x.len(),
        });
    }
    if out.len() != rows {
        return Err(MixedQuantError::InvalidShape {
            context: "matvec output",
            expected: rows,
            actual: out.len(),
        });
    }
    Ok(())
}
