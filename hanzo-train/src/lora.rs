//! Trainable LoRA linear layer.
//!
//! Base weights are frozen plain [`Tensor`]s (no autograd tracking); the LoRA
//! `A` / `B` factors are tracked tensors sourced from a [`hanzo_nn::VarMap`], so
//! only they receive gradients. Shapes match the engine's inference-side merge
//! (`hanzo_quant::lora::merge_lora_weights`): `A` is `(rank, in)`, `B` is `(out, rank)`,
//! and the delta is `scale * (x·Aᵀ)·Bᵀ` with `scale = alpha / rank`.

use hanzo_ml::{Module, Result, Tensor};
use hanzo_nn::Linear;

/// The trainable low-rank factors attached to a projection.
#[derive(Clone)]
pub struct LoraDelta {
    /// Fully-qualified base name of the adapted module, e.g.
    /// `model.layers.0.self_attn.q_proj`. Used to name the saved adapter tensors.
    pub name: String,
    /// `A`: `(rank, in_dim)`, tracked.
    pub a: Tensor,
    /// `B`: `(out_dim, rank)`, tracked.
    pub b: Tensor,
    pub scale: f64,
}

impl LoraDelta {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Reuse Linear's batched matmul (handles 2D/3D/4D inputs) for both factors.
        let down = Linear::new(self.a.clone(), None).forward(x)?; // (.., rank)
        let up = Linear::new(self.b.clone(), None).forward(&down)?; // (.., out)
        up * self.scale
    }
}

/// A linear projection with a frozen base weight and an optional trainable LoRA delta.
#[derive(Clone)]
pub struct LoraLinear {
    base: Linear,
    delta: Option<LoraDelta>,
}

impl LoraLinear {
    pub fn frozen(base: Linear) -> Self {
        Self { base, delta: None }
    }

    pub fn with_delta(base: Linear, delta: LoraDelta) -> Self {
        Self {
            base,
            delta: Some(delta),
        }
    }

    /// The adapter factors, if this layer is trainable — used when saving.
    pub fn delta(&self) -> Option<&LoraDelta> {
        self.delta.as_ref()
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.base.forward(x)?;
        match &self.delta {
            None => Ok(out),
            Some(delta) => out + delta.forward(x)?,
        }
    }
}

/// LoRA factor shapes for a projection of `in_dim -> out_dim` at the given rank.
/// Returns `(A_shape, B_shape)` = `((rank, in_dim), (out_dim, rank))`.
pub fn lora_shapes(rank: usize, in_dim: usize, out_dim: usize) -> ((usize, usize), (usize, usize)) {
    ((rank, in_dim), (out_dim, rank))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_match_engine_merge_convention() {
        // in=8, out=4, rank=2  ->  A=(2,8), B=(4,2); B·A = (4,8) = weight delta.
        let (a, b) = lora_shapes(2, 8, 4);
        assert_eq!(a, (2, 8));
        assert_eq!(b, (4, 2));
        // B(out,rank) · A(rank,in) is conformable and yields (out,in).
        assert_eq!(b.1, a.0);
        assert_eq!((b.0, a.1), (4, 8));
    }

    #[test]
    fn zero_b_gives_zero_delta() {
        use hanzo_ml::{Device, Var};
        let dev = Device::Cpu;
        // A random, B = 0  =>  initial delta must be exactly zero (base preserved).
        let a = Var::randn(0f32, 1f32, (2, 8), &dev).unwrap().as_tensor().clone();
        let b = Var::zeros((4, 2), hanzo_ml::DType::F32, &dev)
            .unwrap()
            .as_tensor()
            .clone();
        let delta = LoraDelta {
            name: "t".into(),
            a,
            b,
            scale: 4.0,
        };
        let x = Tensor::randn(0f32, 1f32, (3, 8), &dev).unwrap();
        let out = delta.forward(&x).unwrap();
        let hi = out.max_all().unwrap().to_scalar::<f32>().unwrap();
        let lo = out.min_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!((hi, lo), (0.0, 0.0));
    }
}
