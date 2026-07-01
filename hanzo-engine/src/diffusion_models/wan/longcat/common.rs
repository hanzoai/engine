#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Shared DiT primitives for the LongCat avatar backbone: sinusoidal timestep embedding, 3D-RoPE
//! (rotate_half), adaLN modulation, and the block-causal self-attn mask. These are written as
//! standalone fns/structs so they can lift into a sibling `wan/dit_common.rs` once EchoMimic and
//! LongCat converge (short-term duplication, flagged for integration).

use crate::attention::AttentionMask;
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::LayerNorm;

pub(crate) const ROPE_THETA: f64 = 10000.0;
pub(crate) const LN_EPS: f64 = 1e-6;
pub(crate) const RMS_EPS: f64 = 1e-6;
const TIME_FACTOR: f64 = 1000.0;
const TIME_MAX_PERIOD: f64 = 10000.0;
// Additive mask sentinel; large-negative (not -inf) so bf16 softmax stays finite.
const MASK_NEG: f64 = -1e9;

/// Sinusoidal timestep features `[B, dim]`, computed in f32 then cast (diffusers `[cos, sin]`).
pub(crate) fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    if dim % 2 == 1 {
        hanzo_ml::bail!("timestep dim {dim} must be even");
    }
    let dev = t.device();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(DType::F32)?;
    let freqs = (arange * (-TIME_MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)
}

/// adaLN shift/scale/gate triple (flux `scale_shift` convention: `x * (1 + scale) + shift`).
pub(crate) struct ModulationOut {
    pub shift: Tensor,
    pub scale: Tensor,
    pub gate: Tensor,
}

impl ModulationOut {
    pub fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.0)?)?
            .broadcast_add(&self.shift)
    }

    pub fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

/// No-affine LayerNorm (weight 1, no bias); hanzo-nn upcasts bf16/f16 to f32 inside `forward`.
pub(crate) fn no_affine_layer_norm(dim: usize, dtype: DType, dev: &Device) -> Result<LayerNorm> {
    Ok(LayerNorm::new_no_bias(Tensor::ones(dim, dtype, dev)?, LN_EPS))
}

/// 3D-RoPE table builder: `head_dim` split across (t, h, w) axes (44/42/42 at head_dim 128).
#[derive(Debug, Clone)]
pub(crate) struct Rope3D {
    axes_dim: [usize; 3],
    theta: f64,
    device: Device,
}

impl Rope3D {
    pub fn new(axes_dim: [usize; 3], theta: f64, device: Device) -> Result<Self> {
        for d in axes_dim {
            if d % 2 != 0 {
                hanzo_ml::bail!("rope axis dim {d} must be even");
            }
        }
        Ok(Self {
            axes_dim,
            theta,
            device,
        })
    }

    // (n, axis_dim/2) frequencies for one axis at positions 0..n.
    fn axis_freqs(&self, axis: usize, n: usize) -> Result<Tensor> {
        let d = self.axes_dim[axis];
        let theta = self.theta;
        let inv: Vec<f32> = (0..d)
            .step_by(2)
            .map(|i| (1.0 / theta.powf(i as f64 / d as f64)) as f32)
            .collect();
        let half = inv.len();
        let inv = Tensor::from_vec(inv, (1, half), &self.device)?;
        let pos = Tensor::arange(0, n as u32, &self.device)?
            .to_dtype(DType::F32)?
            .reshape((n, 1))?;
        pos.broadcast_mul(&inv)
    }

    /// (cos, sin) each `[frames*h*w, head_dim]` in f32, for a flattened (t, h, w) token grid.
    pub fn table(&self, frames: usize, h: usize, w: usize) -> Result<(Tensor, Tensor)> {
        let ft = self.axis_freqs(0, frames)?;
        let fh = self.axis_freqs(1, h)?;
        let fw = self.axis_freqs(2, w)?;
        let (dt, dh, dw) = (ft.dim(1)?, fh.dim(1)?, fw.dim(1)?);
        let ft = ft.reshape((frames, 1, 1, dt))?.broadcast_as((frames, h, w, dt))?;
        let fh = fh.reshape((1, h, 1, dh))?.broadcast_as((frames, h, w, dh))?;
        let fw = fw.reshape((1, 1, w, dw))?.broadcast_as((frames, h, w, dw))?;
        let freqs =
            Tensor::cat(&[&ft, &fh, &fw], D::Minus1)?.reshape((frames * h * w, dt + dh + dw))?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        Ok((emb.cos()?, emb.sin()?))
    }
}

// rotate_half: split last dim in two, return cat(-x2, x1) (LLaMA/NeoX convention).
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// Apply RoPE to `x` `[B, H, seq, D]` with `cos`/`sin` `[seq, D]` (cast to x's dtype, broadcast).
pub(crate) fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let (seq, d) = cos.dims2()?;
    let cos = cos.to_dtype(dtype)?.reshape((1, 1, seq, d))?;
    let sin = sin.to_dtype(dtype)?.reshape((1, 1, seq, d))?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

/// Block-causal self-attn mask `[1, 1, N, N]`: condition queries (row < n_cond) cannot attend noisy
/// keys (col >= n_cond), so the condition stream is invariant across denoise steps (KV-cache valid).
/// Returns `None` when `n_cond == 0` (no condition tokens -> plain bidirectional flash path).
pub(crate) fn block_causal_mask(
    n_cond: usize,
    n_total: usize,
    dtype: DType,
    dev: &Device,
) -> Result<Option<AttentionMask>> {
    if n_cond == 0 {
        return Ok(None);
    }
    let row_is_cond: Vec<f32> = (0..n_total).map(|r| f32::from(r < n_cond)).collect();
    let col_is_noisy: Vec<f32> = (0..n_total).map(|c| f32::from(c >= n_cond)).collect();
    let row = Tensor::from_vec(row_is_cond, (n_total, 1), dev)?;
    let col = Tensor::from_vec(col_is_noisy, (1, n_total), dev)?;
    let blocked = row.broadcast_mul(&col)?;
    let mask = (blocked * MASK_NEG)?
        .reshape((1, 1, n_total, n_total))?
        .to_dtype(dtype)?;
    Ok(Some(AttentionMask::Custom(mask)))
}
