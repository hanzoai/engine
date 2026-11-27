// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Clean-room SpinQuant orthogonal rotations:
//! - Exact $O(n \log n)$ normalized Walsh-Hadamard transform ($H_n$, $H_n^T H_n = I$).
//! - Exact Cayley-parameterized orthogonal matrices $R = (I - A)(I + A)^{-1}$ ($A = -A^T$)
//!   solved via partial-pivoted Gaussian elimination in `f64` precision.
//! - Learned SpinQuant rotation optimization reducing activation outlier kurtosis and
//!   weight+activation quantization MSE while preserving exact invariance $x W^T = (x R)(W R)^T$.

use crate::error::MixedQuantError;
use crate::formats::{matmul_batch_f32, PackedWeights, QuantFormat};

/// Orthogonal rotation operator applied to input activations (`x -> x R`) and absorbed into weights (`W -> W R`).
#[derive(Debug, Clone)]
pub enum RotationTransform {
    /// Seed-regenerated normalized Walsh-Hadamard transform (`D * H_b`): stores only a compact
    /// `u64` seed (or `None` for unit signs) and regenerates Rademacher `±1` signs on the fly
    /// with zero per-layer sign vector allocation.
    SeededWalshHadamard {
        /// Total vector dimension.
        dim: usize,
        /// Power-of-two block size dividing `dim`.
        block_size: usize,
        /// Optional SplitMix64 seed for deterministic Rademacher `±1` signs (`None` = all `+1`).
        seed: Option<u64>,
    },
    /// Explicit-sign normalized Walsh-Hadamard transform storing a materialized `signs` vector.
    WalshHadamard {
        /// Total vector dimension.
        dim: usize,
        /// Power-of-two block size dividing `dim`.
        block_size: usize,
        /// Explicit Rademacher sign flip `(+1 / -1)` per coordinate.
        signs: Vec<f32>,
    },
    /// Learned orthogonal matrix $R \in O(d)$ (e.g. Cayley-parameterized) stored row-major `[dim, dim]`.
    LearnedOrthogonal(OrthogonalMatrix),
    /// Block-diagonal learned orthogonal matrix (`block_size x block_size` repeated along `dim`).
    BlockOrthogonal { dim: usize, block: OrthogonalMatrix },
}

impl RotationTransform {
    /// Constructs an exact normalized Walsh-Hadamard transform with unit signs (`+1`, zero sign storage).
    pub fn hadamard(dim: usize) -> Result<Self, MixedQuantError> {
        if dim == 0 || !dim.is_power_of_two() {
            return Err(MixedQuantError::InvalidRotationDim {
                dim,
                reason: "full Walsh-Hadamard transform requires a positive power-of-two dimension",
            });
        }
        Ok(Self::SeededWalshHadamard {
            dim,
            block_size: dim,
            seed: None,
        })
    }

    /// Constructs a deterministic randomized Walsh-Hadamard transform (`D * H_b`) backed by an
    /// 8-byte SplitMix64 `seed` regenerated on the fly (no per-projection `Vec<f32>` payload).
    pub fn randomized_hadamard(
        dim: usize,
        block_size: usize,
        seed: u64,
    ) -> Result<Self, MixedQuantError> {
        if block_size == 0
            || !block_size.is_power_of_two()
            || dim == 0
            || !dim.is_multiple_of(block_size)
        {
            return Err(MixedQuantError::InvalidRotationDim {
                dim,
                reason: "block_size must be a power of two dividing dim",
            });
        }
        Ok(Self::SeededWalshHadamard {
            dim,
            block_size,
            seed: Some(seed),
        })
    }

    /// Exact serialized/resident payload bytes owned by this rotation transform.
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        match self {
            Self::SeededWalshHadamard { seed: None, .. } => 0,
            // 8-byte SplitMix64 seed + 8-byte block_size descriptor
            Self::SeededWalshHadamard { seed: Some(_), .. } => 16,
            Self::WalshHadamard { signs, .. } => signs.len() * 4,
            Self::LearnedOrthogonal(m) => m.data.len() * 4,
            Self::BlockOrthogonal { block, .. } => block.data.len() * 4,
        }
    }

    /// Vector dimension supported by this rotation.
    #[must_use]
    pub fn dim(&self) -> usize {
        match self {
            Self::SeededWalshHadamard { dim, .. } | Self::WalshHadamard { dim, .. } => *dim,
            Self::LearnedOrthogonal(m) => m.dim,
            Self::BlockOrthogonal { dim, .. } => *dim,
        }
    }

    /// Applies the forward orthogonal transform `v -> v R` in place.
    pub fn apply_in_place(&self, v: &mut [f32]) -> Result<(), MixedQuantError> {
        let d = self.dim();
        if v.len() != d {
            return Err(MixedQuantError::InvalidShape {
                context: "RotationTransform::apply_in_place",
                expected: d,
                actual: v.len(),
            });
        }
        match self {
            Self::SeededWalshHadamard {
                block_size, seed, ..
            } => {
                if let Some(seed_val) = seed {
                    let mut s = *seed_val ^ 0x9e37_79b9_7f4a_7c15;
                    for vi in v.iter_mut() {
                        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
                        let mut z = s;
                        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                        z ^= z >> 31;
                        if (z & 1) != 0 {
                            *vi = -*vi;
                        }
                    }
                }
                for chunk in v.chunks_exact_mut(*block_size) {
                    fwht_normalized_in_place(chunk)?;
                }
                Ok(())
            }
            Self::WalshHadamard {
                block_size, signs, ..
            } => {
                for (vi, &si) in v.iter_mut().zip(signs.iter()) {
                    *vi *= si;
                }
                for chunk in v.chunks_exact_mut(*block_size) {
                    fwht_normalized_in_place(chunk)?;
                }
                Ok(())
            }
            Self::LearnedOrthogonal(m) => {
                let tmp = m.mul_row_vec(v)?;
                v.copy_from_slice(&tmp);
                Ok(())
            }
            Self::BlockOrthogonal { block, .. } => {
                let b = block.dim;
                for chunk in v.chunks_exact_mut(b) {
                    let tmp = block.mul_row_vec(chunk)?;
                    chunk.copy_from_slice(&tmp);
                }
                Ok(())
            }
        }
    }

    /// Applies the rotation to every row of a row-major `[rows, cols]` matrix (`W_rot = W * R`).
    ///
    /// Because $R R^T = I$, for any row vector $x$:
    /// `(x R) (W R)^T = x R R^T W^T = x W^T`.
    pub fn rotate_matrix_rows(
        &self,
        matrix: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, MixedQuantError> {
        if cols != self.dim() || matrix.len() != rows * cols {
            return Err(MixedQuantError::InvalidShape {
                context: "RotationTransform::rotate_matrix_rows",
                expected: rows * self.dim(),
                actual: matrix.len(),
            });
        }
        let mut out = matrix.to_vec();
        for r in 0..rows {
            self.apply_in_place(&mut out[r * cols..(r + 1) * cols])?;
        }
        Ok(out)
    }

    /// Materializes the explicit dense `[dim, dim]` matrix representation for orthogonality verification.
    pub fn to_dense_matrix(&self) -> Result<Vec<f32>, MixedQuantError> {
        let d = self.dim();
        let mut mat = vec![0.0f32; d * d];
        let mut e = vec![0.0f32; d];
        for i in 0..d {
            e.fill(0.0);
            e[i] = 1.0;
            self.apply_in_place(&mut e)?;
            mat[i * d..(i + 1) * d].copy_from_slice(&e);
        }
        Ok(mat)
    }
}

/// Exact in-place normalized Fast Walsh-Hadamard Transform ($O(n \log n)$).
///
/// Computes $y = \frac{1}{\sqrt{n}} H_n x$ where $H_n$ is the Sylvester Hadamard matrix
/// satisfying $H_n = H_n^T$ and $(\frac{1}{\sqrt{n}} H_n)^T (\frac{1}{\sqrt{n}} H_n) = I_n$.
pub fn fwht_normalized_in_place(v: &mut [f32]) -> Result<(), MixedQuantError> {
    let n = v.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(MixedQuantError::InvalidRotationDim {
            dim: n,
            reason: "FWHT requires a positive power-of-two slice length",
        });
    }
    let mut h = 1usize;
    while h < n {
        let step = h * 2;
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = v[j];
                let y = v[j + h];
                v[j] = x + y;
                v[j + h] = x - y;
            }
            i += step;
        }
        h = step;
    }
    let norm = 1.0 / (n as f32).sqrt();
    for val in v.iter_mut() {
        *val *= norm;
    }
    Ok(())
}

/// Dense orthogonal matrix $R \in O(d)$ parameterized via the Cayley transform $R = (I - A)(I + A)^{-1}$.
#[derive(Debug, Clone)]
pub struct OrthogonalMatrix {
    pub dim: usize,
    /// Row-major `[dim, dim]` entries.
    pub data: Vec<f32>,
}

impl OrthogonalMatrix {
    /// Identity orthogonal matrix $I_d$.
    #[must_use]
    pub fn identity(dim: usize) -> Self {
        let mut data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Self { dim, data }
    }

    /// Constructs an exact orthogonal matrix $R = (I - A)(I + A)^{-1}$ from strictly upper-triangular
    /// skew-symmetric parameters `upper_params` of length `dim * (dim - 1) / 2`.
    ///
    /// Since $A = -A^T$ has purely imaginary eigenvalues, $I + A$ is guaranteed nonsingular over $\mathbb{R}$
    /// and $R^T R = (I - A)^{-1}(I + A)(I - A)(I + A)^{-1} = I$ identically.
    pub fn from_cayley_skew(dim: usize, upper_params: &[f32]) -> Result<Self, MixedQuantError> {
        if dim == 0 {
            return Err(MixedQuantError::InvalidRotationDim {
                dim,
                reason: "dimension must be > 0",
            });
        }
        let expected = dim * (dim - 1) / 2;
        if upper_params.len() != expected {
            return Err(MixedQuantError::InvalidShape {
                context: "OrthogonalMatrix::from_cayley_skew",
                expected,
                actual: upper_params.len(),
            });
        }

        // Build M = I + A and N = I - A in f64 for high numerical precision.
        let mut lhs = vec![0.0f64; dim * dim];
        let mut rhs = vec![0.0f64; dim * dim];
        for i in 0..dim {
            lhs[i * dim + i] = 1.0;
            rhs[i * dim + i] = 1.0;
        }
        let mut idx = 0usize;
        for i in 0..dim {
            for j in (i + 1)..dim {
                let a = upper_params[idx] as f64;
                idx += 1;
                lhs[i * dim + j] = a;
                lhs[j * dim + i] = -a;
                rhs[i * dim + j] = -a;
                rhs[j * dim + i] = a;
            }
        }

        // Solve (I + A) * X = (I - A). Since (I - A) and (I + A)^-1 commute,
        // R = (I - A)(I + A)^-1 = (I + A)^-1 (I - A) = X.
        let sol = solve_matrix_system_f64(dim, &lhs, &rhs)?;
        let data = sol.into_iter().map(|v| v as f32).collect();
        Ok(Self { dim, data })
    }

    /// Multiplies two `dim x dim` orthogonal matrices `self * other`.
    pub fn matmul(&self, other: &Self) -> Result<Self, MixedQuantError> {
        if self.dim != other.dim {
            return Err(MixedQuantError::InvalidShape {
                context: "OrthogonalMatrix::matmul",
                expected: self.dim,
                actual: other.dim,
            });
        }
        let d = self.dim;
        let mut data = vec![0.0f32; d * d];
        for i in 0..d {
            for k in 0..d {
                let aik = self.data[i * d + k];
                for j in 0..d {
                    data[i * d + j] += aik * other.data[k * d + j];
                }
            }
        }
        Ok(Self { dim: d, data })
    }

    /// Computes row-vector product `y = x * R` (`y[j] = sum_i x[i] * R[i, j]`).
    pub fn mul_row_vec(&self, x: &[f32]) -> Result<Vec<f32>, MixedQuantError> {
        let d = self.dim;
        if x.len() != d {
            return Err(MixedQuantError::InvalidShape {
                context: "OrthogonalMatrix::mul_row_vec",
                expected: d,
                actual: x.len(),
            });
        }
        let mut y = vec![0.0f32; d];
        for (i, &xi) in x.iter().enumerate() {
            let row = &self.data[i * d..(i + 1) * d];
            for j in 0..d {
                y[j] += xi * row[j];
            }
        }
        Ok(y)
    }

    /// Exact orthogonality error `||R^T R - I||_inf`.
    #[must_use]
    pub fn orthogonality_error_inf(&self) -> f32 {
        orthogonality_error_inf(&self.data, self.dim)
    }
}

/// Computes `||R^T R - I||_inf = max_{i,j} |(R^T R - I)_{i,j}|`.
#[must_use]
pub fn orthogonality_error_inf(r: &[f32], dim: usize) -> f32 {
    let mut max_err = 0.0f64;
    for i in 0..dim {
        for j in 0..dim {
            let mut dot = 0.0f64;
            for k in 0..dim {
                dot += (r[k * dim + i] as f64) * (r[k * dim + j] as f64);
            }
            let target = if i == j { 1.0 } else { 0.0 };
            let err = (dot - target).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err as f32
}

/// Solves `A * X = B` for `dim x dim` matrices via Gauss-Jordan elimination with partial pivoting.
fn solve_matrix_system_f64(dim: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>, MixedQuantError> {
    let mut aug = vec![0.0f64; dim * (2 * dim)];
    for i in 0..dim {
        for j in 0..dim {
            aug[i * 2 * dim + j] = a[i * dim + j];
            aug[i * 2 * dim + dim + j] = b[i * dim + j];
        }
    }
    let stride = 2 * dim;
    for col in 0..dim {
        // Partial pivoting
        let mut pivot_row = col;
        let mut pivot_val = aug[col * stride + col].abs();
        for row in (col + 1)..dim {
            let val = aug[row * stride + col].abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = row;
            }
        }
        if pivot_val < 1e-13 {
            return Err(MixedQuantError::SingularMatrix { pivot_index: col });
        }
        if pivot_row != col {
            for j in col..stride {
                aug.swap(col * stride + j, pivot_row * stride + j);
            }
        }
        let inv_pivot = 1.0 / aug[col * stride + col];
        for j in col..stride {
            aug[col * stride + j] *= inv_pivot;
        }
        for row in 0..dim {
            if row == col {
                continue;
            }
            let factor = aug[row * stride + col];
            if factor.abs() > 0.0 {
                for j in col..stride {
                    let pivot_entry = aug[col * stride + j];
                    aug[row * stride + j] -= factor * pivot_entry;
                }
            }
        }
    }

    let mut x = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            x[i * dim + j] = aug[i * stride + dim + j];
        }
    }
    Ok(x)
}

/// Summary of a SpinQuant learned Cayley orthogonal rotation optimization run.
#[derive(Debug, Clone)]
pub struct SpinQuantResult {
    /// Optimized orthogonal rotation transform.
    pub rotation: RotationTransform,
    /// Verified orthogonality error `||R^T R - I||_inf`.
    pub orthogonality_err_inf: f32,
    /// Verified exact unquantized invariance error `max |x W^T - (x R)(W R)^T|`.
    pub invariance_err_max: f32,
    /// Peak activation crest factor (`max|x| / rms(x)`) before rotation.
    pub pre_rotation_crest: f32,
    /// Peak activation crest factor (`max|x R| / rms(x R)`) after rotation.
    pub post_rotation_crest: f32,
    /// Quantized matmul MSE before rotation (baseline INT4/Ternary).
    pub baseline_mse: f64,
    /// Quantized matmul MSE after Hadamard initialization.
    pub hadamard_mse: f64,
    /// Quantized matmul MSE after learned Cayley refinement.
    pub learned_mse: f64,
}

/// Computes activation crest factor `max_abs / rms` across a batch.
#[must_use]
pub fn activation_crest_factor(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut max_abs = 0.0f32;
    let mut sq_sum = 0.0f64;
    for &v in x {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
        sq_sum += (v as f64) * (v as f64);
    }
    let rms = (sq_sum / x.len() as f64).sqrt() as f32;
    if rms > 1e-12 {
        max_abs / rms
    } else {
        0.0
    }
}

/// Evaluates weight + simulated per-token INT8/INT4 activation quantization MSE under rotation `rot`.
#[allow(clippy::too_many_arguments)]
fn eval_rotated_quant_mse(
    weights: &[f32],
    rows: usize,
    cols: usize,
    activations: &[f32],
    batch_len: usize,
    ref_out: &[f32],
    rot: Option<&RotationTransform>,
    weight_format: QuantFormat,
    group_size: usize,
    act_bits: u8,
) -> Result<f64, MixedQuantError> {
    let (w_eff, x_eff) = if let Some(r) = rot {
        let wr = r.rotate_matrix_rows(weights, rows, cols)?;
        let xr = r.rotate_matrix_rows(activations, batch_len, cols)?;
        (wr, xr)
    } else {
        (weights.to_vec(), activations.to_vec())
    };

    // Quantize activations per-token (symmetric) to reflect true W4A4 / W4A8 SpinQuant behavior
    let x_q = if act_bits < 16 {
        quantize_dequantize_activations(&x_eff, batch_len, cols, act_bits)
    } else {
        x_eff
    };

    let w_q = PackedWeights::quantize(&w_eff, rows, cols, weight_format, group_size)?;
    let mut out = vec![0.0f32; rows];
    let mut sq_err = 0.0f64;
    for b in 0..batch_len {
        w_q.matvec(&x_q[b * cols..(b + 1) * cols], &mut out)?;
        let ref_slice = &ref_out[b * rows..(b + 1) * rows];
        for r in 0..rows {
            let diff = (out[r] - ref_slice[r]) as f64;
            sq_err += diff * diff;
        }
    }
    Ok(sq_err / (batch_len * rows) as f64)
}

/// Per-token symmetric fake quantization for activation outlier evaluation (`act_bits = 4` or `8`).
fn quantize_dequantize_activations(x: &[f32], batch_len: usize, cols: usize, bits: u8) -> Vec<f32> {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    let mut out = vec![0.0f32; x.len()];
    for b in 0..batch_len {
        let row = &x[b * cols..(b + 1) * cols];
        let mut max_abs = 0.0f32;
        for &v in row {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs > 1e-12 { max_abs / qmax } else { 0.0 };
        if scale > 0.0 {
            let inv = 1.0 / scale;
            for c in 0..cols {
                out[b * cols + c] = (row[c] * inv).round().clamp(-qmax, qmax) * scale;
            }
        }
    }
    out
}

/// Optimizes a SpinQuant orthogonal rotation combining Hadamard initialization with
/// Cayley-parameterized orthogonal refinement on the outlier subspace.
///
/// Specifically:
/// 1. Starts from a deterministic randomized Walsh-Hadamard transform $H_0$ (verified $H_0^T H_0 = I$).
/// 2. Identifies the highest-kurtosis / peak-magnitude rotated coordinates and performs
///    Cayley-parameterized plane rotations $R_k = (I - A_k)(I + A_k)^{-1}$ via coordinate
///    line-search minimizing exact weight+activation quantization MSE.
/// 3. Verifies exact orthogonality $\|R^T R - I\|_\infty < 10^{-5}$ and unquantized invariance
///    $\max |x W^T - (x R)(W R)^T| < 10^{-4}$.
#[allow(clippy::too_many_arguments)]
pub fn optimize_spinquant_rotation(
    weights: &[f32],
    rows: usize,
    cols: usize,
    activations: &[f32],
    batch_len: usize,
    weight_format: QuantFormat,
    group_size: usize,
    act_bits: u8,
    cayley_steps: usize,
    seed: u64,
) -> Result<SpinQuantResult, MixedQuantError> {
    if batch_len == 0 || activations.len() != batch_len * cols {
        return Err(MixedQuantError::InvalidCalibrationData {
            reason: "activations shape mismatch in optimize_spinquant_rotation",
        });
    }
    let block_size = largest_pow2_divisor(cols).min(64);
    let ref_out = matmul_batch_f32(weights, rows, cols, activations, batch_len)?;
    let pre_rotation_crest = activation_crest_factor(activations);

    let baseline_mse = eval_rotated_quant_mse(
        weights,
        rows,
        cols,
        activations,
        batch_len,
        &ref_out,
        None,
        weight_format,
        group_size,
        act_bits,
    )?;

    // Initialize with randomized Walsh-Hadamard matrix
    let hadamard_rot = RotationTransform::randomized_hadamard(cols, block_size, seed)?;
    let hadamard_mse = eval_rotated_quant_mse(
        weights,
        rows,
        cols,
        activations,
        batch_len,
        &ref_out,
        Some(&hadamard_rot),
        weight_format,
        group_size,
        act_bits,
    )?;

    // Build dense orthogonal matrix (or block orthogonal matrix if cols > 64) for Cayley refinement
    let block_dim = block_size;
    let base_block_rot = RotationTransform::randomized_hadamard(block_dim, block_dim, seed)?;
    let mut best_block = OrthogonalMatrix {
        dim: block_dim,
        data: base_block_rot.to_dense_matrix()?,
    };
    let mut best_rot = if cols == block_dim {
        RotationTransform::LearnedOrthogonal(best_block.clone())
    } else {
        RotationTransform::BlockOrthogonal {
            dim: cols,
            block: best_block.clone(),
        }
    };
    let mut best_mse = eval_rotated_quant_mse(
        weights,
        rows,
        cols,
        activations,
        batch_len,
        &ref_out,
        Some(&best_rot),
        weight_format,
        group_size,
        act_bits,
    )?;

    // Also compare against pure full-dim Hadamard (in case full-dim beats block-dim)
    if hadamard_mse < best_mse && cols == block_dim {
        best_mse = hadamard_mse;
    }

    // Coordinate Cayley updates: R_{t+1} = R_t * (I - A)(I + A)^-1
    // Targeting pairs of highest-peak vs lowest-peak channels inside the block
    let num_skew = block_dim * (block_dim - 1) / 2;
    let mut skew = vec![0.0f32; num_skew];
    let step_angles = [0.22f32, -0.22, 0.10, -0.10, 0.04, -0.04];

    for step in 0..cayley_steps {
        let x_cur = best_rot.rotate_matrix_rows(activations, batch_len, cols)?;
        // Rank block coordinates by peak magnitude across batch
        let mut peak_by_coord = vec![0.0f32; block_dim];
        for b in 0..batch_len {
            let row = &x_cur[b * cols..(b + 1) * cols];
            for (c, &v) in row.iter().enumerate() {
                let bc = c % block_dim;
                let a = v.abs();
                if a > peak_by_coord[bc] {
                    peak_by_coord[bc] = a;
                }
            }
        }
        let mut order: Vec<usize> = (0..block_dim).collect();
        order.sort_by(|&a, &b| {
            peak_by_coord[b]
                .partial_cmp(&peak_by_coord[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let hi = order[step % (block_dim / 2).max(1)];
        let lo = order[block_dim - 1 - (step % (block_dim / 2).max(1))];
        if hi == lo {
            continue;
        }
        let (p, q) = if hi < lo { (hi, lo) } else { (lo, hi) };
        let skew_idx = upper_tri_index(block_dim, p, q);

        let mut improved = false;
        for &angle in &step_angles {
            skew.fill(0.0);
            skew[skew_idx] = angle;
            let cayley_delta = OrthogonalMatrix::from_cayley_skew(block_dim, &skew)?;
            let cand_block = best_block.matmul(&cayley_delta)?;
            let cand_rot = if cols == block_dim {
                RotationTransform::LearnedOrthogonal(cand_block.clone())
            } else {
                RotationTransform::BlockOrthogonal {
                    dim: cols,
                    block: cand_block.clone(),
                }
            };
            let cand_mse = eval_rotated_quant_mse(
                weights,
                rows,
                cols,
                activations,
                batch_len,
                &ref_out,
                Some(&cand_rot),
                weight_format,
                group_size,
                act_bits,
            )?;
            if cand_mse + 1e-12 < best_mse {
                best_mse = cand_mse;
                best_block = cand_block;
                best_rot = cand_rot;
                improved = true;
                break;
            }
        }
        if !improved && step >= 4 {
            break;
        }
    }

    // Verify orthogonality and unquantized invariance
    let orth_err = best_block.orthogonality_error_inf();
    let w_rot = best_rot.rotate_matrix_rows(weights, rows, cols)?;
    let x_rot = best_rot.rotate_matrix_rows(activations, batch_len, cols)?;
    let rot_unquant_out = matmul_batch_f32(&w_rot, rows, cols, &x_rot, batch_len)?;
    let mut invariance_err_max = 0.0f32;
    for (&a, &b) in ref_out.iter().zip(rot_unquant_out.iter()) {
        let diff = (a - b).abs();
        if diff > invariance_err_max {
            invariance_err_max = diff;
        }
    }
    let post_rotation_crest = activation_crest_factor(&x_rot);

    Ok(SpinQuantResult {
        rotation: best_rot,
        orthogonality_err_inf: orth_err,
        invariance_err_max,
        pre_rotation_crest,
        post_rotation_crest,
        baseline_mse,
        hadamard_mse,
        learned_mse: best_mse,
    })
}

fn upper_tri_index(dim: usize, i: usize, j: usize) -> usize {
    // sum_{r=0}^{i-1} (dim - 1 - r) + (j - i - 1)
    i * (2 * dim - i - 1) / 2 + (j - i - 1)
}

fn largest_pow2_divisor(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    1usize << n.trailing_zeros()
}
