// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Packed low-bit quantized integer matrix layouts for the `poai.matmul.v1` execution profile.
//!
//! # Accepted Numerical Semantics (`poai.matmul.v1`)
//!
//! Let `A` be an `M x K` row-major signed `i8` activation matrix (`A[i, k] in [-128, 127]`),
//! and let `B` be a `K x N` packed low-bit weight matrix stored in column-blocked layout
//! (`N` output channels, each partitioned into `K / 32` blocks of `QUANT_BLOCK_SIZE = 32`
//! weights along `K`).
//!
//! Each block `(j, b)` (`j in 0..N`, `b in 0..K/32`) stores:
//! - an exact signed `i16` fixed-point block scale `s_{j, b} in [-32768, 32767]`
//!   (little-endian; `s = 1` recovers raw unscaled integer GEMM — note that GGUF blocks in
//!   `hanzo-quant` use IEEE-754 `f16` scales, so GGUF weights require explicit scale conversion
//!   at load time), followed by
//! - 32 packed quantized integer weights `q_{j, 32b + t}`:
//!   - [`PackedQuantFormat::Int8Q8_0`]: 32 signed `i8` bytes (`34 bytes/block`, matching the
//!     32-weight payload geometry of GGML `block_q8_0` with an `i16` scale instead of `f16`).
//!   - [`PackedQuantFormat::PackedInt4Q4_0`]: 16 packed bytes (`18 bytes/block`, matching the
//!     32-weight split-half nibble payload geometry of GGML `block_q4_0`: low nibble = `q[t] + 8`
//!     for `t in 0..16`, high nibble = `q[t + 16] + 8`, with an `i16` scale instead of `f16`).
//!   - [`PackedQuantFormat::PackedTernaryI2S`]: 8 packed bytes (`10 bytes/block`, 4 weights/byte in
//!     2-bit little-endian fields: `0b00 -> -1`, `0b01 -> 0`, `0b10 -> +1`; `0b11` is rejected as
//!     malformed). **Distinct execution-profile format — NOT wire-compatible with GGUF `TQ2_0`**
//!     (`BlockTQ2_0` packs 256 weights into 66 bytes with `qs[64]` and a trailing `f16` scale,
//!     whereas `PackedTernaryI2S` requires 80 bytes per 256 weights across eight 10-byte blocks).
//!
//! The exact integer matrix product `C = A * B` (`M x N`, row-major `i64`) is defined by:
//!
//! ```text
//! C[i, j] = sum_{b=0}^{K/32 - 1} s_{j, b} * sum_{t=0}^{31} A[i, 32*b + t] * q_{j, 32*b + t}
//! ```
//!
//! Because addition and multiplication in `Z` are strictly associative and commutative, every
//! compliant GPU or CPU kernel computes bit-identical `C[i, j]`, and because `|C[i, j]| <= (p - 1)/2`
//! for all `K <= 2^30`, exact `i64` accumulation agrees 1-to-1 with `GF(2^61 - 1)`.
//! Any downstream floating-point dequantization scalar, bias addition, normalization, or
//! activation function occurs outside this verified boundary.

use crate::error::PoaiError;
use crate::field::{Fp, MAX_SAFE_SIGNED};
use crate::profile::OperatorKind;

/// Block size along reduction dimension `K` (`32` weights per `i16`-scaled block in `poai.matmul.v1`).
pub const QUANT_BLOCK_SIZE: usize = 32;

/// Maximum supported dimension along `K` or `N` to guarantee zero `i64`/`i128` overflow
/// in both prover accumulation and `GF(p)` Freivalds inner loops.
pub const MAX_SAFE_DIM: usize = 1 << 30;

/// Supported packed low-bit weight layouts under `poai.matmul.v1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PackedQuantFormat {
    /// 2-byte `i16` scale + 32 `i8` weights = 34 bytes per 32-weight block (`Q8_0` payload geometry, `i16` scale).
    Int8Q8_0 = 1,
    /// 2-byte `i16` scale + 16 split-half nibble bytes = 18 bytes per 32-weight block (`Q4_0` payload geometry, `i16` scale).
    PackedInt4Q4_0 = 2,
    /// Custom `poai.matmul.v1` ternary layout: 2-byte leading `i16` scale + 8 2-bit ternary bytes = 10 bytes per 32-weight block (distinct from GGUF `TQ2_0`'s 256-weight, 66-byte layout).
    PackedTernaryI2S = 3,
}

impl PackedQuantFormat {
    /// Stable wire identifier used in commitments and receipts.
    #[inline]
    pub const fn wire_id(self) -> u8 {
        self as u8
    }

    /// Human-readable format name.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Int8Q8_0 => "Int8Q8_0",
            Self::PackedInt4Q4_0 => "PackedInt4Q4_0",
            Self::PackedTernaryI2S => "PackedTernaryI2S",
        }
    }

    /// Number of packed weight payload bytes per 32-element block (excluding the 2-byte `i16` scale).
    #[inline]
    pub const fn payload_bytes_per_block(self) -> usize {
        match self {
            Self::Int8Q8_0 => 32,
            Self::PackedInt4Q4_0 => 16,
            Self::PackedTernaryI2S => 8,
        }
    }

    /// Total bytes per 32-element block (`2` bytes scale + payload bytes).
    #[inline]
    pub const fn block_bytes(self) -> usize {
        2 + self.payload_bytes_per_block()
    }

    /// Corresponding eligible [`OperatorKind`].
    #[inline]
    pub const fn operator_kind(self) -> OperatorKind {
        OperatorKind::from_packed_format(self)
    }
}

/// Row-major `M x K` signed `i8` activation matrix `A`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationMatrix {
    m: usize,
    k: usize,
    data: Vec<i8>,
}

impl ActivationMatrix {
    /// Construct an activation matrix after validating non-zero dimensions and exact buffer length.
    pub fn new(m: usize, k: usize, data: Vec<i8>) -> Result<Self, PoaiError> {
        if m == 0 || k == 0 || m > MAX_SAFE_DIM || k > MAX_SAFE_DIM {
            return Err(PoaiError::InvalidDimensions {
                m,
                k,
                n: 0,
                reason: "m and k must be in 1..=2^30",
            });
        }
        let expected = m.checked_mul(k).ok_or(PoaiError::InvalidDimensions {
            m,
            k,
            n: 0,
            reason: "m * k overflows usize",
        })?;
        if data.len() != expected {
            return Err(PoaiError::BufferLengthMismatch {
                matrix: "A",
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { m, k, data })
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.m
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[i8] {
        let start = i * self.k;
        &self.data[start..start + self.k]
    }

    /// Compute `out = A * v (mod p)` in `O(m * k)` CPU operations with a single modular reduction per row.
    pub fn mul_vec_fp(&self, v: &[Fp], out: &mut [Fp]) -> Result<(), PoaiError> {
        if v.len() != self.k || out.len() != self.m {
            return Err(PoaiError::DimensionMismatch {
                matrix: "A*v",
                expected: (self.m, self.k),
                actual: (out.len(), v.len()),
            });
        }
        for (i, dst) in out.iter_mut().enumerate() {
            let row = self.row(i);
            let mut acc: i128 = 0;
            for (&a_ik, &v_k) in row.iter().zip(v.iter()) {
                acc += (a_ik as i128) * (v_k.raw() as i128);
            }
            *dst = Fp::reduce_i128(acc);
        }
        Ok(())
    }
}

/// Column-blocked `K x N` packed low-bit quantized weight matrix `B`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedWeightMatrix {
    format: PackedQuantFormat,
    k: usize,
    n: usize,
    packed: Vec<u8>,
}

impl PackedWeightMatrix {
    /// Construct and validate a packed weight matrix from canonical raw block bytes.
    ///
    /// Rejects zero dimensions, `k % 32 != 0`, wrong byte lengths, and illegal `0b11` ternary codes.
    pub fn from_raw_bytes(
        format: PackedQuantFormat,
        k: usize,
        n: usize,
        packed: Vec<u8>,
    ) -> Result<Self, PoaiError> {
        if k == 0 || n == 0 || k > MAX_SAFE_DIM || n > MAX_SAFE_DIM {
            return Err(PoaiError::InvalidDimensions {
                m: 0,
                k,
                n,
                reason: "k and n must be in 1..=2^30",
            });
        }
        #[allow(clippy::manual_is_multiple_of)]
        if k % QUANT_BLOCK_SIZE != 0 {
            return Err(PoaiError::InvalidBlockLayout {
                format: format.as_str(),
                expected_multiple: QUANT_BLOCK_SIZE,
                actual_k: k,
            });
        }
        let blocks_per_col = k / QUANT_BLOCK_SIZE;
        let total_blocks = n
            .checked_mul(blocks_per_col)
            .ok_or(PoaiError::InvalidDimensions {
                m: 0,
                k,
                n,
                reason: "block count overflows usize",
            })?;
        let expected_bytes =
            total_blocks
                .checked_mul(format.block_bytes())
                .ok_or(PoaiError::InvalidDimensions {
                    m: 0,
                    k,
                    n,
                    reason: "packed byte size overflows usize",
                })?;
        if packed.len() != expected_bytes {
            return Err(PoaiError::MalformedPackedWeights {
                format: format.as_str(),
                reason: "packed byte buffer length does not match format * n * (k / 32)",
            });
        }

        // For I2_S packed ternary weights, explicitly forbid the reserved 2-bit code 0b11.
        if format == PackedQuantFormat::PackedTernaryI2S {
            let block_bytes = format.block_bytes();
            for blk_idx in 0..total_blocks {
                let payload_start = blk_idx * block_bytes + 2;
                let payload = &packed[payload_start..payload_start + 8];
                for &byte in payload {
                    if (byte & 0x03) == 0x03
                        || ((byte >> 2) & 0x03) == 0x03
                        || ((byte >> 4) & 0x03) == 0x03
                        || ((byte >> 6) & 0x03) == 0x03
                    {
                        return Err(PoaiError::MalformedPackedWeights {
                            format: format.as_str(),
                            reason:
                                "reserved 2-bit ternary code 0b11 encountered in packed payload",
                        });
                    }
                }
            }
        }

        Ok(Self {
            format,
            k,
            n,
            packed,
        })
    }

    /// Pack a column-major slice of quantized integer weights (`col_major_q[j * k + row]`) and
    /// per-block `i16` scales (`scales[j * (k / 32) + blk]`) into canonical `PackedWeightMatrix` bytes.
    pub fn pack_from_column_major(
        format: PackedQuantFormat,
        k: usize,
        n: usize,
        col_major_q: &[i8],
        scales: &[i16],
    ) -> Result<Self, PoaiError> {
        if k == 0 || n == 0 || k > MAX_SAFE_DIM || n > MAX_SAFE_DIM {
            return Err(PoaiError::InvalidDimensions {
                m: 0,
                k,
                n,
                reason: "k and n must be in 1..=2^30",
            });
        }
        #[allow(clippy::manual_is_multiple_of)]
        if k % QUANT_BLOCK_SIZE != 0 {
            return Err(PoaiError::InvalidBlockLayout {
                format: format.as_str(),
                expected_multiple: QUANT_BLOCK_SIZE,
                actual_k: k,
            });
        }
        let blocks_per_col = k / QUANT_BLOCK_SIZE;
        let expected_q = k.checked_mul(n).ok_or(PoaiError::InvalidDimensions {
            m: 0,
            k,
            n,
            reason: "k * n overflows usize",
        })?;
        let expected_scales =
            blocks_per_col
                .checked_mul(n)
                .ok_or(PoaiError::InvalidDimensions {
                    m: 0,
                    k,
                    n,
                    reason: "scale count overflows usize",
                })?;
        if col_major_q.len() != expected_q {
            return Err(PoaiError::BufferLengthMismatch {
                matrix: "B_weights",
                expected: expected_q,
                actual: col_major_q.len(),
            });
        }
        if scales.len() != expected_scales {
            return Err(PoaiError::BufferLengthMismatch {
                matrix: "B_scales",
                expected: expected_scales,
                actual: scales.len(),
            });
        }

        let mut packed = Vec::with_capacity(expected_scales * format.block_bytes());
        for j in 0..n {
            for blk in 0..blocks_per_col {
                let scale = scales[j * blocks_per_col + blk];
                packed.extend_from_slice(&scale.to_le_bytes());
                let base = j * k + blk * QUANT_BLOCK_SIZE;
                let slice = &col_major_q[base..base + QUANT_BLOCK_SIZE];
                match format {
                    PackedQuantFormat::Int8Q8_0 => {
                        for &w in slice {
                            packed.push(w as u8);
                        }
                    }
                    PackedQuantFormat::PackedInt4Q4_0 => {
                        // GGML Q4_0 split-half nibble packing:
                        // byte[t] low nibble = slice[t] + 8, high nibble = slice[t + 16] + 8.
                        for t in 0..16 {
                            let w_lo = slice[t];
                            let w_hi = slice[t + 16];
                            if !(-8..=7).contains(&w_lo) || !(-8..=7).contains(&w_hi) {
                                return Err(PoaiError::MalformedPackedWeights {
                                    format: format.as_str(),
                                    reason: "int4 weight outside [-8, 7]",
                                });
                            }
                            let lo = (w_lo + 8) as u8;
                            let hi = (w_hi + 8) as u8;
                            packed.push(lo | (hi << 4));
                        }
                    }
                    PackedQuantFormat::PackedTernaryI2S => {
                        // 4 ternary values per byte: -1 -> 0b00, 0 -> 0b01, +1 -> 0b10.
                        for byte_idx in 0..8 {
                            let mut byte = 0u8;
                            for sub in 0..4 {
                                let w = slice[byte_idx * 4 + sub];
                                let code: u8 = match w {
                                    -1 => 0b00,
                                    0 => 0b01,
                                    1 => 0b10,
                                    _ => {
                                        return Err(PoaiError::MalformedPackedWeights {
                                            format: format.as_str(),
                                            reason: "ternary weight outside {-1, 0, +1}",
                                        });
                                    }
                                };
                                byte |= code << (2 * sub);
                            }
                            packed.push(byte);
                        }
                    }
                }
            }
        }

        Ok(Self {
            format,
            k,
            n,
            packed,
        })
    }

    #[inline]
    pub fn format(&self) -> PackedQuantFormat {
        self.format
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn blocks_per_col(&self) -> usize {
        self.k / QUANT_BLOCK_SIZE
    }

    #[inline]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.packed
    }

    /// Return a cloned weight matrix with one block scale modified (used in adversarial tests/benchmarks).
    pub fn with_tampered_block_scale(
        &self,
        col: usize,
        block: usize,
        new_scale: i16,
    ) -> Result<Self, PoaiError> {
        let blocks_per_col = self.blocks_per_col();
        if col >= self.n || block >= blocks_per_col {
            return Err(PoaiError::DimensionMismatch {
                matrix: "B_block",
                expected: (self.n, blocks_per_col),
                actual: (col, block),
            });
        }
        let mut cloned = self.clone();
        let offset = (col * blocks_per_col + block) * self.format.block_bytes();
        let le = new_scale.to_le_bytes();
        cloned.packed[offset] = le[0];
        cloned.packed[offset + 1] = le[1];
        Ok(cloned)
    }

    /// Unpack column `j` into `col_buf` (length `k`) as exact scaled integer weights `s_{j, b} * q_{j, k}`.
    #[inline]
    pub fn unpack_column_scaled(&self, col: usize, col_buf: &mut [i32]) {
        debug_assert_eq!(col_buf.len(), self.k);
        let blocks_per_col = self.blocks_per_col();
        let block_bytes = self.format.block_bytes();
        let col_byte_base = col * blocks_per_col * block_bytes;

        match self.format {
            PackedQuantFormat::Int8Q8_0 => {
                for blk in 0..blocks_per_col {
                    let off = col_byte_base + blk * block_bytes;
                    let scale = i16::from_le_bytes([self.packed[off], self.packed[off + 1]]) as i32;
                    let payload = &self.packed[off + 2..off + 34];
                    let dst = &mut col_buf[blk * 32..blk * 32 + 32];
                    for t in 0..32 {
                        dst[t] = scale * (payload[t] as i8 as i32);
                    }
                }
            }
            PackedQuantFormat::PackedInt4Q4_0 => {
                for blk in 0..blocks_per_col {
                    let off = col_byte_base + blk * block_bytes;
                    let scale = i16::from_le_bytes([self.packed[off], self.packed[off + 1]]) as i32;
                    let payload = &self.packed[off + 2..off + 18];
                    let dst = &mut col_buf[blk * 32..blk * 32 + 32];
                    for t in 0..16 {
                        let byte = payload[t];
                        let q_lo = ((byte & 0x0F) as i32) - 8;
                        let q_hi = ((byte >> 4) as i32) - 8;
                        dst[t] = scale * q_lo;
                        dst[t + 16] = scale * q_hi;
                    }
                }
            }
            PackedQuantFormat::PackedTernaryI2S => {
                for blk in 0..blocks_per_col {
                    let off = col_byte_base + blk * block_bytes;
                    let scale = i16::from_le_bytes([self.packed[off], self.packed[off + 1]]) as i32;
                    let payload = &self.packed[off + 2..off + 10];
                    let dst = &mut col_buf[blk * 32..blk * 32 + 32];
                    for (byte_idx, &byte) in payload.iter().enumerate() {
                        let base = byte_idx * 4;
                        for sub in 0..4 {
                            let code = (byte >> (2 * sub)) & 0x03;
                            let q = (code as i32) - 1;
                            dst[base + sub] = scale * q;
                        }
                    }
                }
            }
        }
    }

    /// Compute `out = B * r (mod p)` directly from packed low-bit weights in streaming `O(K * N)`
    /// CPU time without materializing a dense `K x N` matrix.
    ///
    /// Uses one field multiply per 32-weight block (`r' = scale * r_j mod p`), splits the 61-bit
    /// residue `r' = r_lo + r_hi * 2^31` into native 64-bit accumulators (zero overflow for
    /// `n <= 2^24`), and performs only `K` modular reductions total.
    pub fn mul_vec_fp(&self, r: &[Fp], out: &mut [Fp]) -> Result<(), PoaiError> {
        if r.len() != self.n || out.len() != self.k {
            return Err(PoaiError::DimensionMismatch {
                matrix: "B*r",
                expected: (self.k, self.n),
                actual: (out.len(), r.len()),
            });
        }

        let blocks_per_col = self.blocks_per_col();
        let block_bytes = self.format.block_bytes();

        // Split 61-bit r_scaled = r_lo (31 bits) + r_hi (30 bits) * 2^31.
        // Per update: |q * r_lo| <= 128 * 2^31 = 2^38.
        // For n <= 2^24 (16,777,216 columns), |acc_lo| <= 2^62 < i64::MAX (zero overflow).
        if self.n <= (1 << 24) {
            let mut acc_lo = vec![0i64; self.k];
            let mut acc_hi = vec![0i64; self.k];

            match self.format {
                PackedQuantFormat::Int8Q8_0 => {
                    for (j, &r_j) in r.iter().enumerate() {
                        let col_byte_base = j * blocks_per_col * block_bytes;
                        for blk in 0..blocks_per_col {
                            let off = col_byte_base + blk * block_bytes;
                            let scale =
                                i16::from_le_bytes([self.packed[off], self.packed[off + 1]]);
                            if scale == 0 {
                                continue;
                            }
                            let r_scaled = Fp::from_i16(scale).mul(r_j).raw();
                            let r_lo = (r_scaled & 0x7FFF_FFFF) as i64;
                            let r_hi = (r_scaled >> 31) as i64;
                            let payload = &self.packed[off + 2..off + 34];
                            let dst_lo = &mut acc_lo[blk * 32..blk * 32 + 32];
                            let dst_hi = &mut acc_hi[blk * 32..blk * 32 + 32];
                            for t in 0..32 {
                                let q = payload[t] as i8 as i64;
                                dst_lo[t] += q * r_lo;
                                dst_hi[t] += q * r_hi;
                            }
                        }
                    }
                }
                PackedQuantFormat::PackedInt4Q4_0 => {
                    for (j, &r_j) in r.iter().enumerate() {
                        let col_byte_base = j * blocks_per_col * block_bytes;
                        for blk in 0..blocks_per_col {
                            let off = col_byte_base + blk * block_bytes;
                            let scale =
                                i16::from_le_bytes([self.packed[off], self.packed[off + 1]]);
                            if scale == 0 {
                                continue;
                            }
                            let r_scaled = Fp::from_i16(scale).mul(r_j).raw();
                            let r_lo = (r_scaled & 0x7FFF_FFFF) as i64;
                            let r_hi = (r_scaled >> 31) as i64;
                            let payload = &self.packed[off + 2..off + 18];
                            let dst_lo = &mut acc_lo[blk * 32..blk * 32 + 32];
                            let dst_hi = &mut acc_hi[blk * 32..blk * 32 + 32];
                            for t in 0..16 {
                                let byte = payload[t];
                                let q_lo = ((byte & 0x0F) as i64) - 8;
                                let q_hi = ((byte >> 4) as i64) - 8;
                                dst_lo[t] += q_lo * r_lo;
                                dst_hi[t] += q_lo * r_hi;
                                dst_lo[t + 16] += q_hi * r_lo;
                                dst_hi[t + 16] += q_hi * r_hi;
                            }
                        }
                    }
                }
                PackedQuantFormat::PackedTernaryI2S => {
                    for (j, &r_j) in r.iter().enumerate() {
                        let col_byte_base = j * blocks_per_col * block_bytes;
                        for blk in 0..blocks_per_col {
                            let off = col_byte_base + blk * block_bytes;
                            let scale =
                                i16::from_le_bytes([self.packed[off], self.packed[off + 1]]);
                            if scale == 0 {
                                continue;
                            }
                            let r_scaled = Fp::from_i16(scale).mul(r_j).raw();
                            let r_lo = (r_scaled & 0x7FFF_FFFF) as i64;
                            let r_hi = (r_scaled >> 31) as i64;
                            // Branchless 4-entry LUT: 0b00 -> -r, 0b01 -> 0, 0b10 -> +r.
                            // Reserved code 0b11 is rejected at construction time.
                            let lut_lo = [-r_lo, 0, r_lo, 0];
                            let lut_hi = [-r_hi, 0, r_hi, 0];
                            let payload = &self.packed[off + 2..off + 10];
                            let dst_lo = &mut acc_lo[blk * 32..blk * 32 + 32];
                            let dst_hi = &mut acc_hi[blk * 32..blk * 32 + 32];
                            for (byte_idx, &byte) in payload.iter().enumerate() {
                                let base = byte_idx * 4;
                                let c0 = (byte & 0x03) as usize;
                                let c1 = ((byte >> 2) & 0x03) as usize;
                                let c2 = ((byte >> 4) & 0x03) as usize;
                                let c3 = (byte >> 6) as usize;
                                dst_lo[base] += lut_lo[c0];
                                dst_hi[base] += lut_hi[c0];
                                dst_lo[base + 1] += lut_lo[c1];
                                dst_hi[base + 1] += lut_hi[c1];
                                dst_lo[base + 2] += lut_lo[c2];
                                dst_hi[base + 2] += lut_hi[c2];
                                dst_lo[base + 3] += lut_lo[c3];
                                dst_hi[base + 3] += lut_hi[c3];
                            }
                        }
                    }
                }
            }

            for ((dst, &lo), &hi) in out.iter_mut().zip(acc_lo.iter()).zip(acc_hi.iter()) {
                let full = (lo as i128) + ((hi as i128) << 31);
                *dst = Fp::reduce_i128(full);
            }
            return Ok(());
        }

        // Fallback wide i128 accumulator for matrices with n > 2^24 columns.
        let mut acc = vec![0i128; self.k];
        for (j, &r_j) in r.iter().enumerate() {
            let col_byte_base = j * blocks_per_col * block_bytes;
            for blk in 0..blocks_per_col {
                let off = col_byte_base + blk * block_bytes;
                let scale = i16::from_le_bytes([self.packed[off], self.packed[off + 1]]);
                if scale == 0 {
                    continue;
                }
                let r_scaled = Fp::from_i16(scale).mul(r_j).raw() as i128;
                let dst = &mut acc[blk * 32..blk * 32 + 32];
                match self.format {
                    PackedQuantFormat::Int8Q8_0 => {
                        let payload = &self.packed[off + 2..off + 34];
                        for t in 0..32 {
                            dst[t] += (payload[t] as i8 as i128) * r_scaled;
                        }
                    }
                    PackedQuantFormat::PackedInt4Q4_0 => {
                        let payload = &self.packed[off + 2..off + 18];
                        for t in 0..16 {
                            let byte = payload[t];
                            dst[t] += (((byte & 0x0F) as i128) - 8) * r_scaled;
                            dst[t + 16] += (((byte >> 4) as i128) - 8) * r_scaled;
                        }
                    }
                    PackedQuantFormat::PackedTernaryI2S => {
                        let lut = [-r_scaled, 0, r_scaled, 0];
                        let payload = &self.packed[off + 2..off + 10];
                        for (byte_idx, &byte) in payload.iter().enumerate() {
                            let base = byte_idx * 4;
                            dst[base] += lut[(byte & 0x03) as usize];
                            dst[base + 1] += lut[((byte >> 2) & 0x03) as usize];
                            dst[base + 2] += lut[((byte >> 4) & 0x03) as usize];
                            dst[base + 3] += lut[(byte >> 6) as usize];
                        }
                    }
                }
            }
        }
        for (dst, &sum) in out.iter_mut().zip(acc.iter()) {
            *dst = Fp::reduce_i128(sum);
        }
        Ok(())
    }
}

/// Row-major `M x N` exact signed `i64` output matrix `C`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputMatrix {
    m: usize,
    n: usize,
    data: Vec<i64>,
}

impl OutputMatrix {
    /// Construct an output matrix after validating non-zero dimensions and exact buffer length.
    pub fn new(m: usize, n: usize, data: Vec<i64>) -> Result<Self, PoaiError> {
        if m == 0 || n == 0 || m > MAX_SAFE_DIM || n > MAX_SAFE_DIM {
            return Err(PoaiError::InvalidDimensions {
                m,
                k: 0,
                n,
                reason: "m and n must be in 1..=2^30",
            });
        }
        let expected = m.checked_mul(n).ok_or(PoaiError::InvalidDimensions {
            m,
            k: 0,
            n,
            reason: "m * n overflows usize",
        })?;
        if data.len() != expected {
            return Err(PoaiError::BufferLengthMismatch {
                matrix: "C",
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { m, n, data })
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.m
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn data(&self) -> &[i64] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [i64] {
        &mut self.data
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[i64] {
        let start = i * self.n;
        &self.data[start..start + self.n]
    }

    /// Compute `out = C * r (mod p)` in `O(m * n)` while strictly validating that every entry
    /// `C[i, j]` satisfies `|C[i, j]| <= (p - 1)/2` (preventing any modular wraparound attack).
    pub fn mul_vec_fp(&self, r: &[Fp], out: &mut [Fp]) -> Result<(), PoaiError> {
        if r.len() != self.n || out.len() != self.m {
            return Err(PoaiError::DimensionMismatch {
                matrix: "C*r",
                expected: (self.m, self.n),
                actual: (out.len(), r.len()),
            });
        }

        // Accumulate 32 products per u128 reduction: 32 * (p - 1)^2 < 2^127 < u128::MAX.
        const CHUNK: usize = 32;
        for (i, dst) in out.iter_mut().enumerate() {
            let row = self.row(i);
            let mut acc: u128 = 0;
            let mut col = 0;
            while col < self.n {
                let end = (col + CHUNK).min(self.n);
                for j in col..end {
                    let c_fp = Fp::from_i64_checked(row[j], i, j)?;
                    acc += (c_fp.raw() as u128) * (r[j].raw() as u128);
                }
                acc = Fp::reduce_u128(acc).raw() as u128;
                col = end;
            }
            *dst = Fp::reduce_u128(acc);
        }
        Ok(())
    }
}

/// Deterministic prover execution of `C = A * B` for packed low-bit quantized weights.
///
/// Unpacks each column of `B` once into an L1-friendly scratch buffer and computes exact `i64`
/// inner products across all rows of `A`, rejecting any result whose magnitude exceeds the
/// safe `GF(2^61 - 1)` bijection bound `(p - 1)/2`.
pub fn prover_matmul(
    a: &ActivationMatrix,
    b: &PackedWeightMatrix,
) -> Result<OutputMatrix, PoaiError> {
    if a.cols() != b.rows() {
        return Err(PoaiError::DimensionMismatch {
            matrix: "A*B",
            expected: (a.rows(), b.rows()),
            actual: (a.rows(), a.cols()),
        });
    }

    let m = a.rows();
    let k = a.cols();
    let n = b.cols();
    let mut c_data = vec![0i64; m * n];
    let mut col_buf = vec![0i32; k];
    let max_abs = MAX_SAFE_SIGNED as u64;

    for j in 0..n {
        b.unpack_column_scaled(j, &mut col_buf);
        for i in 0..m {
            let a_row = a.row(i);
            // Each |a_ik * b_kj| <= 128 * (32768 * 128) = 2^29.
            // With k <= 2^30, |acc| <= 2^59 < 2^60 - 1 < i64::MAX (zero i64 overflow).
            let mut acc: i64 = 0;
            for idx in 0..k {
                acc += (a_row[idx] as i64) * (col_buf[idx] as i64);
            }
            if acc.unsigned_abs() > max_abs {
                return Err(PoaiError::AccumulatorOutOfFieldRange {
                    row: i,
                    col: j,
                    value: acc as i128,
                    max_abs: MAX_SAFE_SIGNED as i128,
                });
            }
            c_data[i * n + j] = acc;
        }
    }

    OutputMatrix::new(m, n, c_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_and_exact_matmul_all_formats() {
        let m = 3;
        let k = 64;
        let n = 4;
        let a_data: Vec<i8> = (0..m * k)
            .map(|i| ((i as i32 * 17 - 40) % 127) as i8)
            .collect();
        let a = ActivationMatrix::new(m, k, a_data).unwrap();
        let scales: Vec<i16> = (0..(n * (k / 32)))
            .map(|i| if i % 2 == 0 { 13 } else { -7 })
            .collect();

        for (format, range) in [
            (PackedQuantFormat::Int8Q8_0, (-120i8, 120i8)),
            (PackedQuantFormat::PackedInt4Q4_0, (-8i8, 7i8)),
            (PackedQuantFormat::PackedTernaryI2S, (-1i8, 1i8)),
        ] {
            let span = (range.1 as i32 - range.0 as i32 + 1) as usize;
            let q_vals: Vec<i8> = (0..k * n)
                .map(|i| (range.0 as i32 + ((i * 7 + 3) % span) as i32) as i8)
                .collect();
            let b =
                PackedWeightMatrix::pack_from_column_major(format, k, n, &q_vals, &scales).unwrap();
            let c = prover_matmul(&a, &b).unwrap();

            // Check against naive reference
            for i in 0..m {
                for j in 0..n {
                    let mut ref_sum: i128 = 0;
                    for idx in 0..k {
                        let blk = idx / 32;
                        let s = scales[j * (k / 32) + blk] as i128;
                        let q = q_vals[j * k + idx] as i128;
                        let av = a.row(i)[idx] as i128;
                        ref_sum += s * q * av;
                    }
                    assert_eq!(c.row(i)[j] as i128, ref_sum);
                }
            }

            // Check streaming B*r matches dense A*(B*r) == C*r
            let r: Vec<Fp> = (0..n)
                .map(|j| Fp::from_u64_reduced(1000 + j as u64))
                .collect();
            let mut br = vec![Fp::ZERO; k];
            let mut abr = vec![Fp::ZERO; m];
            let mut cr = vec![Fp::ZERO; m];
            b.mul_vec_fp(&r, &mut br).unwrap();
            a.mul_vec_fp(&br, &mut abr).unwrap();
            c.mul_vec_fp(&r, &mut cr).unwrap();
            assert_eq!(abr, cr);
        }
    }

    #[test]
    fn rejects_malformed_packed_weights_and_illegal_ternary_code() {
        // Non-multiple of 32
        assert!(matches!(
            PackedWeightMatrix::from_raw_bytes(PackedQuantFormat::Int8Q8_0, 30, 4, vec![0; 10]),
            Err(PoaiError::InvalidBlockLayout { .. })
        ));
        // Wrong byte length
        assert!(matches!(
            PackedWeightMatrix::from_raw_bytes(
                PackedQuantFormat::PackedInt4Q4_0,
                32,
                2,
                vec![0; 10]
            ),
            Err(PoaiError::MalformedPackedWeights { .. })
        ));
        // Reserved 0b11 ternary code
        let mut bad_ternary = vec![0u8; 10];
        bad_ternary[0] = 1; // scale = 1
        bad_ternary[2] = 0b00_01_10_11; // lowest 2 bits are 0b11
        assert!(matches!(
            PackedWeightMatrix::from_raw_bytes(
                PackedQuantFormat::PackedTernaryI2S,
                32,
                1,
                bad_ternary
            ),
            Err(PoaiError::MalformedPackedWeights { .. })
        ));
    }
}
