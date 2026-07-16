//! Supervised next-token loss with a per-position mask.

use hanzo_ml::{Result, Tensor, D};

/// Sum of mask-weighted next-token negative log-likelihood.
///
/// - `logits`: `(n, vocab)` raw logits
/// - `targets`: `(n,)` `u32` target token ids
/// - `weights`: `(n,)` loss mask (0 = ignore, 1 = train)
///
/// Returns the scalar `Σ_i weights[i] · -log p(targets[i])`. Divide by the total
/// weight to get the mean. Gradients flow to whatever produced `logits`.
pub fn masked_nll_sum(logits: &Tensor, targets: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let logp = hanzo_nn::ops::log_softmax(logits, D::Minus1)?;
    let picked = logp.gather(&targets.unsqueeze(1)?, 1)?.squeeze(1)?; // (n,)
    let nll = picked.neg()?;
    (nll * weights)?.sum_all()
}

#[cfg(test)]
mod tests {
    use hanzo_ml::{Device, Tensor};

    #[test]
    fn masked_nll_ignores_zero_weight_positions() {
        let dev = Device::Cpu;
        // 2 positions, vocab 3. Position 0 masked out (weight 0), position 1 trained.
        let logits = Tensor::from_vec(
            vec![0f32, 0f32, 0f32, /* pos1 */ 10f32, 0f32, 0f32],
            (2, 3),
            &dev,
        )
        .unwrap();
        let targets = Tensor::from_vec(vec![2u32, 0u32], (2,), &dev).unwrap();
        let weights = Tensor::from_vec(vec![0f32, 1f32], (2,), &dev).unwrap();
        let loss = super::masked_nll_sum(&logits, &targets, &weights)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // Only position 1 counts; it strongly predicts token 0 (its target) => small loss.
        assert!(loss < 0.01, "loss={loss}");
    }
}
