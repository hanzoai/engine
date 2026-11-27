//! Low-rank adaptation (LoRA) linear projection adapters with exact forward
//! evaluation and analytical backpropagation.
//!
//! For a frozen base linear map `W` of shape `[d_out, d_in]`, the adapted
//! projection adds a scaled rank-`r` update:
//!
//! ```text
//! y = W x + (alpha / r) * B (A x)
//! ```
//!
//! where `A` has shape `[r, d_in]` (He/Kaiming-scaled Gaussian initialization)
//! and `B` has shape `[d_out, r]` initialized to exact zeros so the adapter
//! preserves the base model output identically at step zero.

use crate::sha256::SplitMix64;

/// Low-rank adapter parameters `A: [rank, d_in]` and `B: [d_out, rank]`.
#[derive(Clone, Debug)]
pub struct LoraLinear {
    /// Input feature dimension.
    pub d_in: usize,
    /// Output feature dimension.
    pub d_out: usize,
    /// Adapter bottleneck rank `r`.
    pub rank: usize,
    /// Scaling factor `alpha / r`.
    pub scale: f32,
    /// Down-projection matrix `A` stored row-major as `[rank, d_in]`.
    pub a: Vec<f32>,
    /// Up-projection matrix `B` stored row-major as `[d_out, rank]`.
    pub b: Vec<f32>,
}

impl LoraLinear {
    /// Initializes a LoRA adapter with deterministic Gaussian `A` and exact-zero `B`.
    #[must_use]
    pub fn new(d_in: usize, d_out: usize, rank: usize, alpha: f32, rng: &mut SplitMix64) -> Self {
        let std = (1.0 / (d_in.max(1) as f32)).sqrt();
        let mut a = Vec::with_capacity(rank * d_in);
        for _ in 0..(rank * d_in) {
            a.push(rng.next_gaussian_f32() * std);
        }
        let b = vec![0.0f32; d_out * rank];
        let scale = alpha / (rank.max(1) as f32);
        Self {
            d_in,
            d_out,
            rank,
            scale,
            a,
            b,
        }
    }

    /// Total number of trainable scalar parameters (`A` + `B`).
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.a.len() + self.b.len()
    }

    /// Appends `A` followed by `B` into `out`.
    pub fn write_params(&self, out: &mut Vec<f32>) {
        out.extend_from_slice(&self.a);
        out.extend_from_slice(&self.b);
    }

    /// Loads `A` followed by `B` from `src` starting at `offset`, returning the
    /// next offset.
    pub fn read_params(&mut self, src: &[f32], offset: usize) -> Result<usize, String> {
        let need = self.param_count();
        if offset + need > src.len() {
            return Err(format!(
                "LoRA parameter slice too short: need {} from offset {}, got {}",
                need,
                offset,
                src.len()
            ));
        }
        let a_end = offset + self.a.len();
        let b_end = a_end + self.b.len();
        self.a.copy_from_slice(&src[offset..a_end]);
        self.b.copy_from_slice(&src[a_end..b_end]);
        Ok(b_end)
    }

    /// Adds `(alpha / r) * B (A x)` into `out` and returns the intermediate
    /// bottleneck activation `h = A x` of length `rank` for exact backprop.
    #[must_use]
    pub fn forward_add(&self, x: &[f32], out: &mut [f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.d_in);
        debug_assert_eq!(out.len(), self.d_out);

        let mut h = vec![0.0f32; self.rank];
        for (r_idx, h_val) in h.iter_mut().enumerate() {
            let a_row = &self.a[r_idx * self.d_in..(r_idx + 1) * self.d_in];
            let mut acc = 0.0f32;
            for (&ai, &xi) in a_row.iter().zip(x.iter()) {
                acc += ai * xi;
            }
            *h_val = acc;
        }

        for (o_idx, out_val) in out.iter_mut().enumerate() {
            let b_row = &self.b[o_idx * self.rank..(o_idx + 1) * self.rank];
            let mut acc = 0.0f32;
            for (&bi, &hi) in b_row.iter().zip(h.iter()) {
                acc += bi * hi;
            }
            *out_val += self.scale * acc;
        }

        h
    }

    /// Accumulates analytical gradients into `grad_a`, `grad_b`, and `grad_x`
    /// given input `x`, cached bottleneck `h = A x`, and upstream `grad_out`.
    pub fn backward_accumulate(
        &self,
        x: &[f32],
        h: &[f32],
        grad_out: &[f32],
        grad_a: &mut [f32],
        grad_b: &mut [f32],
        grad_x: &mut [f32],
    ) {
        debug_assert_eq!(x.len(), self.d_in);
        debug_assert_eq!(h.len(), self.rank);
        debug_assert_eq!(grad_out.len(), self.d_out);
        debug_assert_eq!(grad_a.len(), self.a.len());
        debug_assert_eq!(grad_b.len(), self.b.len());
        debug_assert_eq!(grad_x.len(), self.d_in);

        let mut grad_h = vec![0.0f32; self.rank];

        // dL/dB[o, r] += scale * grad_out[o] * h[r]
        // dL/dh[r]     = sum_o scale * grad_out[o] * B[o, r]
        for (o_idx, &go) in grad_out.iter().enumerate() {
            let sgo = self.scale * go;
            let b_row = &self.b[o_idx * self.rank..(o_idx + 1) * self.rank];
            let gb_row = &mut grad_b[o_idx * self.rank..(o_idx + 1) * self.rank];
            for r_idx in 0..self.rank {
                gb_row[r_idx] += sgo * h[r_idx];
                grad_h[r_idx] += sgo * b_row[r_idx];
            }
        }

        // dL/dA[r, i] += grad_h[r] * x[i]
        // dL/dx[i]    += sum_r grad_h[r] * A[r, i]
        for (r_idx, &gh) in grad_h.iter().enumerate() {
            let a_row = &self.a[r_idx * self.d_in..(r_idx + 1) * self.d_in];
            let ga_row = &mut grad_a[r_idx * self.d_in..(r_idx + 1) * self.d_in];
            for i in 0..self.d_in {
                ga_row[i] += gh * x[i];
                grad_x[i] += gh * a_row[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_starts_at_zero_delta() {
        let mut rng = SplitMix64::new(42);
        let lora = LoraLinear::new(16, 12, 4, 8.0, &mut rng);
        let x = vec![1.0f32; 16];
        let mut out = vec![0.5f32; 12];
        let _ = lora.forward_add(&x, &mut out);
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-7);
        }
    }

    #[test]
    fn lora_backward_matches_finite_differences() {
        let mut rng = SplitMix64::new(77);
        let mut lora = LoraLinear::new(6, 5, 3, 6.0, &mut rng);
        for v in &mut lora.b {
            *v = rng.next_gaussian_f32() * 0.2;
        }
        let x: Vec<f32> = (0..6).map(|_| rng.next_gaussian_f32()).collect();
        let target: Vec<f32> = (0..5).map(|_| rng.next_gaussian_f32()).collect();

        let eval_loss = |l: &LoraLinear, xv: &[f32]| -> f32 {
            let mut y = vec![0.0f32; 5];
            let _ = l.forward_add(xv, &mut y);
            y.iter()
                .zip(target.iter())
                .map(|(yi, ti)| 0.5 * (yi - ti) * (yi - ti))
                .sum()
        };

        let mut y = vec![0.0f32; 5];
        let h = lora.forward_add(&x, &mut y);
        let grad_out: Vec<f32> = y
            .iter()
            .zip(target.iter())
            .map(|(yi, ti)| yi - ti)
            .collect();
        let mut grad_a = vec![0.0f32; lora.a.len()];
        let mut grad_b = vec![0.0f32; lora.b.len()];
        let mut grad_x = vec![0.0f32; x.len()];
        lora.backward_accumulate(&x, &h, &grad_out, &mut grad_a, &mut grad_b, &mut grad_x);

        let eps = 1e-3f32;
        for i in 0..lora.a.len() {
            let orig = lora.a[i];
            lora.a[i] = orig + eps;
            let lp = eval_loss(&lora, &x);
            lora.a[i] = orig - eps;
            let lm = eval_loss(&lora, &x);
            lora.a[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (num - grad_a[i]).abs() < 1e-3,
                "grad_a[{i}] mismatch: num={num}, ana={}",
                grad_a[i]
            );
        }
        for i in 0..lora.b.len() {
            let orig = lora.b[i];
            lora.b[i] = orig + eps;
            let lp = eval_loss(&lora, &x);
            lora.b[i] = orig - eps;
            let lm = eval_loss(&lora, &x);
            lora.b[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (num - grad_b[i]).abs() < 1e-3,
                "grad_b[{i}] mismatch: num={num}, ana={}",
                grad_b[i]
            );
        }
        for i in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let num = (eval_loss(&lora, &xp) - eval_loss(&lora, &xm)) / (2.0 * eps);
            assert!(
                (num - grad_x[i]).abs() < 1e-3,
                "grad_x[{i}] mismatch: num={num}, ana={}",
                grad_x[i]
            );
        }
    }
}
