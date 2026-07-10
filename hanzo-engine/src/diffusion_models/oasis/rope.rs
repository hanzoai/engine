#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Axial rotary embeddings for Oasis, matching lucidrains `rotary-embedding-torch` exactly:
//! interleaved (adjacent-pair) `rotate_half`, freqs repeated `... n -> ... (n r)` with r=2. Spatial
//! and VAE axes use "pixel" freqs (`linspace(1, max_freq/2, n) * pi`, positions `linspace(-1,1,N)`);
//! the temporal axis uses "lang" freqs (`1/theta^(2i/dim)`, positions `0..T`). Rotation is applied to
//! q/k `[B, H, S, D]`; when the freq width is < D only the leading dims rotate (VAE case).

use std::f64::consts::PI;

use hanzo_ml::{Device, Result, Tensor, D};

const THETA: f64 = 10000.0;

fn linspace(start: f64, end: f64, steps: usize) -> Vec<f64> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f64;
    (0..steps).map(|i| start + i as f64 * step).collect()
}

fn pixel_base(dim: usize, max_freq: f64) -> Vec<f64> {
    linspace(1.0, max_freq / 2.0, dim / 2)
        .into_iter()
        .map(|f| f * PI)
        .collect()
}

fn lang_base(dim: usize) -> Vec<f64> {
    (0..dim)
        .step_by(2)
        .map(|i| 1.0 / THETA.powf(i as f64 / dim as f64))
        .collect()
}

/// Precomputed cos/sin tables `[S, rot_dim]` for one attention axis-group. `rot_dim` may be < head
/// dim (VAE), in which case only the leading `rot_dim` features rotate.
pub struct RotaryTable {
    cos: Tensor,
    sin: Tensor,
    rot_dim: usize,
}

impl RotaryTable {
    /// Axial table: each `(positions, base_freqs)` axis contributes `2*base.len()` interleaved
    /// features; axes are concatenated along the feature dim in order (row-major over sizes).
    fn axial(axes: &[(Vec<f64>, Vec<f64>)], device: &Device) -> Result<Self> {
        let sizes: Vec<usize> = axes.iter().map(|(p, _)| p.len()).collect();
        let total: usize = sizes.iter().product();
        let rot_dim: usize = axes.iter().map(|(_, b)| 2 * b.len()).sum();
        let mut cos = vec![0f32; total * rot_dim];
        let mut sin = vec![0f32; total * rot_dim];
        for flat in 0..total {
            let mut rem = flat;
            let mut coord = vec![0usize; axes.len()];
            for a in (0..axes.len()).rev() {
                coord[a] = rem % sizes[a];
                rem /= sizes[a];
            }
            let mut off = 0;
            for (a, (pos, base)) in axes.iter().enumerate() {
                let p = pos[coord[a]];
                for &b in base {
                    let v = (p * b) as f32;
                    let (c, s) = (v.cos(), v.sin());
                    cos[flat * rot_dim + off] = c;
                    cos[flat * rot_dim + off + 1] = c;
                    sin[flat * rot_dim + off] = s;
                    sin[flat * rot_dim + off + 1] = s;
                    off += 2;
                }
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (total, rot_dim), device)?,
            sin: Tensor::from_vec(sin, (total, rot_dim), device)?,
            rot_dim,
        })
    }

    /// Spatial pixel-axial table over an `h*w` grid (DiT `spatial_rotary_emb`, dim = head_dim/2).
    pub fn spatial(
        h: usize,
        w: usize,
        head_dim: usize,
        max_freq: f64,
        device: &Device,
    ) -> Result<Self> {
        let base = pixel_base(head_dim / 2, max_freq);
        let axes = vec![
            (linspace(-1.0, 1.0, h), base.clone()),
            (linspace(-1.0, 1.0, w), base),
        ];
        Self::axial(&axes, device)
    }

    /// VAE pixel-axial table over an `h*w` grid (dim = head_dim/4 -> rot covers head_dim/2).
    pub fn vae_spatial(h: usize, w: usize, head_dim: usize, device: &Device) -> Result<Self> {
        let base = pixel_base(head_dim / 4, (h * w) as f64);
        let axes = vec![
            (linspace(-1.0, 1.0, h), base.clone()),
            (linspace(-1.0, 1.0, w), base),
        ];
        Self::axial(&axes, device)
    }

    /// Temporal lang table for positions `0..max_t` (rot covers the full head dim).
    pub fn temporal(max_t: usize, head_dim: usize, device: &Device) -> Result<Self> {
        let base = lang_base(head_dim);
        let axes = vec![((0..max_t).map(|i| i as f64).collect(), base)];
        Self::axial(&axes, device)
    }

    /// Rotate `x` `[B, H, S, D]`. When the table covers `seq < S` (temporal sliding window) pass a
    /// pre-sliced view via [`Self::slice`]; here S must equal the table length.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, s, d) = x.dims4()?;
        let dt = x.dtype();
        let cos = self.cos.to_dtype(dt)?.reshape((1, 1, s, self.rot_dim))?;
        let sin = self.sin.to_dtype(dt)?.reshape((1, 1, s, self.rot_dim))?;
        if self.rot_dim == d {
            return x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?;
        }
        let mid = x.narrow(D::Minus1, 0, self.rot_dim)?;
        let right = x.narrow(D::Minus1, self.rot_dim, d - self.rot_dim)?;
        let rotated = (mid.broadcast_mul(&cos)? + rotate_half(&mid)?.broadcast_mul(&sin)?)?;
        Tensor::cat(&[&rotated, &right], D::Minus1)
    }

    /// Leading `t`-position slice of a temporal table (positions re-index from 0 each window).
    pub fn slice(&self, t: usize) -> Result<Self> {
        Ok(Self {
            cos: self.cos.narrow(0, 0, t)?,
            sin: self.sin.narrow(0, 0, t)?,
            rot_dim: self.rot_dim,
        })
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let d = dims[dims.len() - 1];
    let mut split = dims[..dims.len() - 1].to_vec();
    split.push(d / 2);
    split.push(2);
    let xr = x.reshape(split)?;
    let x1 = xr.narrow(D::Minus1, 0, 1)?;
    let x2 = xr.narrow(D::Minus1, 1, 1)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?.reshape(dims)
}
