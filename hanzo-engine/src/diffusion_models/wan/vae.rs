#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{Device, Result, Tensor, D};
use hanzo_nn::Conv2d;
use hanzo_quant::{Convolution, ShardedVarBuilder};

use crate::layers::{conv2d, MatMul};

use super::conv3d::{CausalConv3d, FeatCache, Slot};

// F.normalize floor; Wan's RMS_norm normalizes over the channel dim then scales by sqrt(dim).
const NORM_EPS: f64 = 1e-12;
// Per-channel latent stats from the shipped Wan2.1 VAE (replace a scalar scaling_factor).
const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];

#[derive(Debug, Clone)]
pub struct WanVaeConfig {
    pub base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub temperal_downsample: Vec<bool>,
}

impl Default for WanVaeConfig {
    fn default() -> Self {
        Self {
            base_dim: 96,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temperal_downsample: vec![false, true, true],
        }
    }
}

// Channel-wise RMS norm (F.normalize over dim 1, then * sqrt(dim) * gamma). gamma is stored as
// [dim,1,1,1] (images=false, 5D acts) or [dim,1,1] (images=true, per-frame 4D acts).
struct WanRmsNorm {
    gamma: Tensor,
    scale: f64,
}

impl WanRmsNorm {
    fn new(dim: usize, images: bool, vb: ShardedVarBuilder) -> Result<Self> {
        let shape: Vec<usize> = if images {
            vec![dim, 1, 1]
        } else {
            vec![dim, 1, 1, 1]
        };
        let gamma = vb.get(shape, "gamma")?.reshape(dim)?;
        Ok(Self {
            gamma,
            scale: (dim as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = x.dim(1)?;
        let denom = (x.sqr()?.sum_keepdim(1)? + NORM_EPS)?.sqrt()?;
        let g = match x.rank() {
            5 => self.gamma.reshape((1, c, 1, 1, 1))?,
            4 => self.gamma.reshape((1, c, 1, 1))?,
            r => hanzo_ml::bail!("WanRmsNorm: unsupported rank {r}"),
        };
        x.broadcast_div(&denom)?
            .affine(self.scale, 0.)?
            .broadcast_mul(&g)
    }
}

struct ResidualBlock {
    norm1: WanRmsNorm,
    conv1: CausalConv3d,
    norm2: WanRmsNorm,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    fn new(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let res = vb.pp("residual");
        let norm1 = WanRmsNorm::new(in_dim, false, res.pp(0))?;
        let conv1 = CausalConv3d::new(
            in_dim,
            out_dim,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
            res.pp(2),
        )?;
        let norm2 = WanRmsNorm::new(out_dim, false, res.pp(3))?;
        let conv2 = CausalConv3d::new(
            out_dim,
            out_dim,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
            res.pp(6),
        )?;
        let shortcut = if in_dim == out_dim {
            None
        } else {
            Some(CausalConv3d::new(
                in_dim,
                out_dim,
                (1, 1, 1),
                (1, 1, 1),
                (0, 0, 0),
                true,
                vb.pp("shortcut"),
            )?)
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let h = match &self.shortcut {
            Some(c) => c.forward(x, None)?,
            None => x.clone(),
        };
        let prev = cache.step(x)?;
        let mut y = self
            .conv1
            .forward(&self.norm1.forward(x)?.silu()?, prev.as_ref())?;
        let prev = cache.step(&y)?;
        y = self
            .conv2
            .forward(&self.norm2.forward(&y)?.silu()?, prev.as_ref())?;
        y + h
    }
}

struct AttentionBlock {
    norm: WanRmsNorm,
    to_qkv: Conv2d,
    proj: Conv2d,
    dim: usize,
}

impl AttentionBlock {
    fn new(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let norm = WanRmsNorm::new(dim, true, vb.pp("norm"))?;
        let to_qkv = conv2d(dim, dim * 3, 1, Default::default(), vb.pp("to_qkv"))?;
        let proj = conv2d(dim, dim, 1, Default::default(), vb.pp("proj"))?;
        Ok(Self {
            norm,
            to_qkv,
            proj,
            dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (bt, b, t) = merge_bt(x)?;
        let h = self.norm.forward(&bt)?;
        let qkv = Convolution.forward_2d(&self.to_qkv, &h)?;
        let (n, _, hh, ww) = qkv.dims4()?;
        let qkv = qkv.flatten_from(2)?.transpose(1, 2)?; // [n, hw, 3c]
        let q = qkv.narrow(D::Minus1, 0, self.dim)?.contiguous()?;
        let k = qkv.narrow(D::Minus1, self.dim, self.dim)?.contiguous()?;
        let v = qkv
            .narrow(D::Minus1, 2 * self.dim, self.dim)?
            .contiguous()?;
        let scale = 1.0 / (self.dim as f64).sqrt();
        let attn = (MatMul.matmul(&q, &k.t()?)? * scale)?;
        let out = MatMul.matmul(&hanzo_nn::ops::softmax_last_dim(&attn)?, &v)?; // [n, hw, c]
        let out = out.transpose(1, 2)?.reshape((n, self.dim, hh, ww))?;
        let out = Convolution.forward_2d(&self.proj, &out)?;
        split_bt(&out, b, t)? + x
    }
}

enum SampleMode {
    Down2d,
    Down3d,
    Up2d,
    Up3d,
}

struct WanResample {
    mode: SampleMode,
    conv: Conv2d, // spatial conv (resample.1)
    time_conv: Option<CausalConv3d>,
}

impl WanResample {
    fn new(dim: usize, mode: SampleMode, vb: ShardedVarBuilder) -> Result<Self> {
        let conv = match mode {
            SampleMode::Down2d | SampleMode::Down3d => {
                let cfg = hanzo_nn::Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                };
                conv2d(dim, dim, 3, cfg, vb.pp("resample").pp(1))?
            }
            SampleMode::Up2d | SampleMode::Up3d => {
                let cfg = hanzo_nn::Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                };
                conv2d(dim, dim / 2, 3, cfg, vb.pp("resample").pp(1))?
            }
        };
        let time_conv = match mode {
            SampleMode::Down3d => Some(CausalConv3d::new(
                dim,
                dim,
                (3, 1, 1),
                (2, 1, 1),
                (0, 0, 0),
                true,
                vb.pp("time_conv"),
            )?),
            SampleMode::Up3d => Some(CausalConv3d::new(
                dim,
                dim * 2,
                (3, 1, 1),
                (1, 1, 1),
                (1, 0, 0),
                true,
                vb.pp("time_conv"),
            )?),
            _ => None,
        };
        Ok(Self {
            mode,
            conv,
            time_conv,
        })
    }

    fn spatial(&self, x: &Tensor) -> Result<Tensor> {
        let (bt, b, t) = merge_bt(x)?;
        let y = match self.mode {
            SampleMode::Down2d | SampleMode::Down3d => {
                let p = bt
                    .pad_with_zeros(D::Minus1, 0, 1)?
                    .pad_with_zeros(D::Minus2, 0, 1)?;
                Convolution.forward_2d(&self.conv, &p)?
            }
            SampleMode::Up2d | SampleMode::Up3d => {
                let (_, _, h, w) = bt.dims4()?;
                let up = bt.upsample_nearest2d(h * 2, w * 2)?;
                Convolution.forward_2d(&self.conv, &up)?
            }
        };
        split_bt(&y, b, t)
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let x = match self.mode {
            SampleMode::Up3d => self.temporal_up(x, cache)?,
            _ => x.clone(),
        };
        let x = self.spatial(&x)?;
        match self.mode {
            SampleMode::Down3d => self.temporal_down(&x, cache),
            _ => Ok(x),
        }
    }

    // Temporal downsample (stride-2 causal conv) seeded by the previous window's last frame.
    fn temporal_down(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let tc = self.time_conv.as_ref().unwrap();
        let i = cache.advance();
        let t = x.dim(2)?;
        let last = x.narrow(2, t - 1, 1)?.contiguous()?;
        match cache.slot(i).clone() {
            Slot::Frames(prev) => {
                let pt = prev.dim(2)?;
                let seeded = Tensor::cat(&[&prev.narrow(2, pt - 1, 1)?, x], 2)?;
                cache.set(i, Slot::Frames(last));
                tc.forward(&seeded, None)
            }
            _ => {
                cache.set(i, Slot::Frames(last));
                Ok(x.clone())
            }
        }
    }

    // Temporal upsample: causal conv to 2*dim channels then interleave into 2x frames.
    fn temporal_up(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let tc = self.time_conv.as_ref().unwrap();
        let i = cache.advance();
        let slot = cache.slot(i).clone();
        if matches!(slot, Slot::Empty) {
            cache.set(i, Slot::Rep);
            return Ok(x.clone());
        }
        let t = x.dim(2)?;
        let take = super::conv3d::CACHE_T.min(t);
        let mut nc = x.narrow(2, t - take, take)?.contiguous()?;
        let y = if let Slot::Frames(prev) = &slot {
            if nc.dim(2)? < 2 {
                let pt = prev.dim(2)?;
                nc = Tensor::cat(&[&prev.narrow(2, pt - 1, 1)?, &nc], 2)?;
            }
            tc.forward(x, Some(prev))?
        } else {
            if nc.dim(2)? < 2 {
                nc = Tensor::cat(&[&nc.zeros_like()?, &nc], 2)?;
            }
            tc.forward(x, None)?
        };
        cache.set(i, Slot::Frames(nc));
        double_temporal(&y)
    }
}

enum Layer {
    Res(ResidualBlock),
    Attn(AttentionBlock),
    Resample(WanResample),
}

impl Layer {
    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        match self {
            Layer::Res(r) => r.forward(x, cache),
            Layer::Attn(a) => a.forward(x),
            Layer::Resample(s) => s.forward(x, cache),
        }
    }
}

struct Encoder3d {
    conv_in: CausalConv3d,
    downs: Vec<Layer>,
    middle: Vec<Layer>,
    head_norm: WanRmsNorm,
    head_conv: CausalConv3d,
}

impl Encoder3d {
    fn new(cfg: &WanVaeConfig, out_z: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let base = cfg.base_dim;
        let dims: Vec<usize> = std::iter::once(1)
            .chain(cfg.dim_mult.iter().copied())
            .map(|u| base * u)
            .collect();
        let conv_in = CausalConv3d::new(
            3,
            dims[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
            vb.pp("conv1"),
        )?;

        let vb_down = vb.pp("downsamples");
        let mut downs = Vec::new();
        let mut idx = 0usize;
        let n_stage = cfg.dim_mult.len();
        for i in 0..n_stage {
            let in_dim = dims[i];
            let out_dim = dims[i + 1];
            for j in 0..cfg.num_res_blocks {
                let ic = if j == 0 { in_dim } else { out_dim };
                downs.push(Layer::Res(ResidualBlock::new(
                    ic,
                    out_dim,
                    vb_down.pp(idx),
                )?));
                idx += 1;
            }
            if i != n_stage - 1 {
                let mode = if cfg.temperal_downsample[i] {
                    SampleMode::Down3d
                } else {
                    SampleMode::Down2d
                };
                downs.push(Layer::Resample(WanResample::new(
                    out_dim,
                    mode,
                    vb_down.pp(idx),
                )?));
                idx += 1;
            }
        }

        let mid_dim = *dims.last().unwrap();
        let vb_mid = vb.pp("middle");
        let middle = vec![
            Layer::Res(ResidualBlock::new(mid_dim, mid_dim, vb_mid.pp(0))?),
            Layer::Attn(AttentionBlock::new(mid_dim, vb_mid.pp(1))?),
            Layer::Res(ResidualBlock::new(mid_dim, mid_dim, vb_mid.pp(2))?),
        ];

        let head_norm = WanRmsNorm::new(mid_dim, false, vb.pp("head").pp(0))?;
        let head_conv = CausalConv3d::new(
            mid_dim,
            out_z,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
            vb.pp("head").pp(2),
        )?;
        Ok(Self {
            conv_in,
            downs,
            middle,
            head_norm,
            head_conv,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let prev = cache.step(x)?;
        let mut h = self.conv_in.forward(x, prev.as_ref())?;
        for l in &self.downs {
            h = l.forward(&h, cache)?;
        }
        for l in &self.middle {
            h = l.forward(&h, cache)?;
        }
        h = self.head_norm.forward(&h)?.silu()?;
        let prev = cache.step(&h)?;
        self.head_conv.forward(&h, prev.as_ref())
    }
}

struct Decoder3d {
    conv_in: CausalConv3d,
    middle: Vec<Layer>,
    ups: Vec<Layer>,
    head_norm: WanRmsNorm,
    head_conv: CausalConv3d,
}

impl Decoder3d {
    fn new(cfg: &WanVaeConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let base = cfg.base_dim;
        let last = *cfg.dim_mult.last().unwrap();
        let dims: Vec<usize> = std::iter::once(last)
            .chain(cfg.dim_mult.iter().rev().copied())
            .map(|u| base * u)
            .collect();
        let temperal_up: Vec<bool> = cfg.temperal_downsample.iter().rev().copied().collect();
        let n_stage = cfg.dim_mult.len();

        let conv_in = CausalConv3d::new(
            cfg.z_dim,
            dims[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
            vb.pp("conv1"),
        )?;
        let vb_mid = vb.pp("middle");
        let middle = vec![
            Layer::Res(ResidualBlock::new(dims[0], dims[0], vb_mid.pp(0))?),
            Layer::Attn(AttentionBlock::new(dims[0], vb_mid.pp(1))?),
            Layer::Res(ResidualBlock::new(dims[0], dims[0], vb_mid.pp(2))?),
        ];

        let vb_up = vb.pp("upsamples");
        let mut ups = Vec::new();
        let mut idx = 0usize;
        for i in 0..n_stage {
            let mut in_dim = dims[i];
            let out_dim = dims[i + 1];
            // upsample halves channels, so every stage after the first ingests in_dim/2.
            if i >= 1 {
                in_dim /= 2;
            }
            for j in 0..cfg.num_res_blocks + 1 {
                let ic = if j == 0 { in_dim } else { out_dim };
                ups.push(Layer::Res(ResidualBlock::new(ic, out_dim, vb_up.pp(idx))?));
                idx += 1;
            }
            if i != n_stage - 1 {
                let mode = if temperal_up[i] {
                    SampleMode::Up3d
                } else {
                    SampleMode::Up2d
                };
                ups.push(Layer::Resample(WanResample::new(
                    out_dim,
                    mode,
                    vb_up.pp(idx),
                )?));
                idx += 1;
            }
        }

        let base_out = *dims.last().unwrap();
        let head_norm = WanRmsNorm::new(base_out, false, vb.pp("head").pp(0))?;
        let head_conv = CausalConv3d::new(
            base_out,
            3,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
            vb.pp("head").pp(2),
        )?;
        Ok(Self {
            conv_in,
            middle,
            ups,
            head_norm,
            head_conv,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let prev = cache.step(x)?;
        let mut h = self.conv_in.forward(x, prev.as_ref())?;
        for l in &self.middle {
            h = l.forward(&h, cache)?;
        }
        for l in &self.ups {
            h = l.forward(&h, cache)?;
        }
        h = self.head_norm.forward(&h)?.silu()?;
        let prev = cache.step(&h)?;
        self.head_conv.forward(&h, prev.as_ref())
    }
}

// 3D causal VAE with 4x temporal / 8x spatial compression. Reused by EchoMimic, InfiniteTalk,
// LongCat. encode/decode walk the video in Wan's autoregressive windows (1,4,4,... frames) so
// the causal temporal convs see the right left-context; the streaming cache carries it across.
pub struct AutoencoderKLWan {
    encoder: Encoder3d,
    conv1: CausalConv3d, // post-encoder moments mix (z*2 -> z*2)
    conv2: CausalConv3d, // pre-decoder (z -> z)
    decoder: Decoder3d,
    z_dim: usize,
    mean: Tensor, // [1, z, 1, 1, 1]
    std: Tensor,  // [1, z, 1, 1, 1]
}

impl AutoencoderKLWan {
    pub fn new(cfg: &WanVaeConfig, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        let z = cfg.z_dim;
        let encoder = Encoder3d::new(cfg, z * 2, vb.pp("encoder"))?;
        let conv1 = CausalConv3d::new(
            z * 2,
            z * 2,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            true,
            vb.pp("conv1"),
        )?;
        let conv2 = CausalConv3d::new(z, z, (1, 1, 1), (1, 1, 1), (0, 0, 0), true, vb.pp("conv2"))?;
        let decoder = Decoder3d::new(cfg, vb.pp("decoder"))?;
        let mean = Tensor::from_slice(&LATENTS_MEAN[..z], (1, z, 1, 1, 1), device)?;
        let std = Tensor::from_slice(&LATENTS_STD[..z], (1, z, 1, 1, 1), device)?;
        Ok(Self {
            encoder,
            conv1,
            conv2,
            decoder,
            z_dim: z,
            mean,
            std,
        })
    }

    // RGB video [B,3,F,H,W] in [-1,1] -> normalized latent mode [B,z,(F-1)/4+1,H/8,W/8].
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(2)?;
        let iters = 1 + (t - 1) / 4;
        let mut cache = FeatCache::new();
        let mut out: Option<Tensor> = None;
        for i in 0..iters {
            cache.rewind();
            let chunk = if i == 0 {
                x.narrow(2, 0, 1)?
            } else {
                let start = 1 + 4 * (i - 1);
                x.narrow(2, start, (t - start).min(4))?
            };
            let o = self.encoder.forward(&chunk, &mut cache)?;
            out = Some(match out {
                None => o,
                Some(p) => Tensor::cat(&[&p, &o], 2)?,
            });
        }
        let moments = self.conv1.forward(&out.unwrap(), None)?;
        let mu = moments.narrow(1, 0, self.z_dim)?;
        let mean = self.mean.to_dtype(mu.dtype())?;
        let std = self.std.to_dtype(mu.dtype())?;
        mu.broadcast_sub(&mean)?.broadcast_div(&std)
    }

    // Normalized latent [B,z,T,H,W] -> RGB video [B,3,F,H,W] in [-1,1], F = 1+4*(T-1).
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let mean = self.mean.to_dtype(z.dtype())?;
        let std = self.std.to_dtype(z.dtype())?;
        let z = z.broadcast_mul(&std)?.broadcast_add(&mean)?;
        let x = self.conv2.forward(&z, None)?;
        let iters = x.dim(2)?;
        let mut cache = FeatCache::new();
        let mut out: Option<Tensor> = None;
        for i in 0..iters {
            cache.rewind();
            let chunk = x.narrow(2, i, 1)?;
            let o = self.decoder.forward(&chunk, &mut cache)?;
            out = Some(match out {
                None => o,
                Some(p) => Tensor::cat(&[&p, &o], 2)?,
            });
        }
        out.unwrap().clamp(-1f32, 1f32)
    }

    pub fn z_dim(&self) -> usize {
        self.z_dim
    }
}

fn merge_bt(x: &Tensor) -> Result<(Tensor, usize, usize)> {
    let (b, c, t, h, w) = x.dims5()?;
    let y = x
        .permute((0, 2, 1, 3, 4))?
        .contiguous()?
        .reshape((b * t, c, h, w))?;
    Ok((y, b, t))
}

fn split_bt(x: &Tensor, b: usize, t: usize) -> Result<Tensor> {
    let (_, c, h, w) = x.dims4()?;
    x.reshape((b, t, c, h, w))?
        .permute((0, 2, 1, 3, 4))?
        .contiguous()
}

// [B, 2C, T, H, W] -> [B, C, 2T, H, W] by interleaving the two channel halves across time.
fn double_temporal(y: &Tensor) -> Result<Tensor> {
    let (b, c2, t, h, w) = y.dims5()?;
    let c = c2 / 2;
    let y = y.reshape((b, 2, c, t, h, w))?;
    let y0 = y.narrow(1, 0, 1)?.squeeze(1)?;
    let y1 = y.narrow(1, 1, 1)?.squeeze(1)?;
    Tensor::stack(&[&y0, &y1], 3)?.reshape((b, c, t * 2, h, w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::{DType, Shape};
    use hanzo_nn::var_builder::SimpleBackend;
    use hanzo_nn::Init;
    use hanzo_quant::ShardedSafeTensors;

    struct RandnBackend;
    impl SimpleBackend for RandnBackend {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _h: Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            if name.ends_with("bias") {
                Tensor::zeros(s, dtype, dev)
            } else if name.ends_with("gamma") {
                Tensor::ones(s, dtype, dev)
            } else {
                Tensor::randn(0f64, 0.05, s, dev)?.to_dtype(dtype)
            }
        }
        fn get_unchecked(&self, _n: &str, _d: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("needs shape")
        }
        fn contains_tensor(&self, _n: &str) -> bool {
            true
        }
    }

    fn tiny_cfg() -> WanVaeConfig {
        WanVaeConfig {
            base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 1,
            temperal_downsample: vec![false, true, true],
        }
    }

    fn assert_finite(t: &Tensor) -> Result<()> {
        let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(v.iter().all(|x| x.is_finite()), "non-finite VAE output");
        let (mn, mx) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        assert!(mx != mn, "degenerate constant output");
        Ok(())
    }

    #[test]
    fn vae_round_trip_shapes_and_compression() -> Result<()> {
        let dev = Device::Cpu;
        let vb = ShardedSafeTensors::wrap(Box::new(RandnBackend), DType::F32, dev.clone());
        let cfg = tiny_cfg();
        let vae = AutoencoderKLWan::new(&cfg, vb, &dev)?;

        // 5 frames -> (5-1)/4+1 = 2 latent frames; 32px -> 32/8 = 4 latent spatial.
        let f = 5usize;
        let x = Tensor::randn(0f64, 1.0, (1, 3, f, 32, 32), &dev)?.to_dtype(DType::F32)?;
        let z = vae.encode(&x)?;
        assert_eq!(
            z.dims(),
            &[1, cfg.z_dim, 2, 4, 4],
            "latent shape / compression"
        );
        assert_finite(&z)?;

        let recon = vae.decode(&z)?;
        assert_eq!(
            recon.dims(),
            &[1, 3, f, 32, 32],
            "decode recovers frame count"
        );
        assert_finite(&recon)?;
        Ok(())
    }

    #[test]
    fn vae_round_trip_nine_frames() -> Result<()> {
        let dev = Device::Cpu;
        let vb = ShardedSafeTensors::wrap(Box::new(RandnBackend), DType::F32, dev.clone());
        let cfg = tiny_cfg();
        let vae = AutoencoderKLWan::new(&cfg, vb, &dev)?;
        let x = Tensor::randn(0f64, 1.0, (1, 3, 9, 16, 16), &dev)?.to_dtype(DType::F32)?;
        let z = vae.encode(&x)?;
        assert_eq!(z.dims(), &[1, cfg.z_dim, 3, 2, 2]);
        let recon = vae.decode(&z)?;
        assert_eq!(recon.dims(), &[1, 3, 9, 16, 16]);
        assert_finite(&recon)?;
        Ok(())
    }
}
