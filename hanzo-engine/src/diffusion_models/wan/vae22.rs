#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Wan2.2 high-compression 3D causal VAE (`AutoencoderKLWan`, `is_residual=True`), the TI2V-5B
//! backbone. 16x16x4 compression: a 2x spatial pixel-unshuffle patchify feeds a residual conv stack
//! (asymmetric base dims 160 enc / 256 dec, z_dim 48). Diffusers checkpoint naming
//! (`encoder.down_blocks.{i}.resnets.{j}`, `downsampler.{resample.1,time_conv}`, `mid_block`,
//! `quant_conv`, `post_quant_conv`) so it loads directly from `Wan2.2-TI2V-5B-Diffusers/vae`.
//!
//! The shared low-level pieces (`CausalConv3d`, `FeatCache`, `WanRmsNorm`, `AttentionBlock`, the
//! bt-merge helpers) are reused from the Wan2.1 VAE; only the 2.2-specific residual down/up blocks
//! with their param-free `AvgDown3D`/`DupUp3D` skip connections, explicit-out-dim resample, and
//! patchify live here.

use hanzo_ml::{Device, Result, Tensor, D};
use hanzo_quant::{Convolution, ShardedVarBuilder};

use crate::layers::conv2d;
use hanzo_nn::{Conv2d, Conv2dConfig};

use super::conv3d::{CausalConv3d, FeatCache, Slot, CACHE_T};
use super::vae::{double_temporal, merge_bt, split_bt, AttentionBlock, WanRmsNorm};

// Wan2.2-VAE per-channel latent stats (48 dims) from Wan2.2-TI2V-5B-Diffusers/vae/config.json.
const LATENTS_MEAN: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813,
    0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108,
    -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313,
    -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944,
    0.3573, -0.0548, -0.1681, -0.0667,
];
const LATENTS_STD: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

#[derive(Debug, Clone)]
pub struct Wan22VaeConfig {
    pub base_dim: usize,
    pub decoder_base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub temperal_downsample: Vec<bool>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl Wan22VaeConfig {
    pub fn ti2v_5b() -> Self {
        Self {
            base_dim: 160,
            decoder_base_dim: 256,
            z_dim: 48,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temperal_downsample: vec![false, true, true],
            in_channels: 12,
            out_channels: 12,
            patch_size: 2,
            latents_mean: LATENTS_MEAN.to_vec(),
            latents_std: LATENTS_STD.to_vec(),
        }
    }

    pub fn tiny() -> Self {
        Self {
            base_dim: 16,
            decoder_base_dim: 24,
            z_dim: 12,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 1,
            temperal_downsample: vec![false, true, true],
            in_channels: 12,
            out_channels: 12,
            patch_size: 2,
            latents_mean: vec![0.0; 12],
            latents_std: vec![1.0; 12],
        }
    }
}

// [B,C,F,H,W] -> [B, C*p*p, F, H/p, W/p] via 2x2 spatial pixel-unshuffle (channel order c,r,q).
fn patchify(x: &Tensor, p: usize) -> Result<Tensor> {
    if p == 1 {
        return Ok(x.clone());
    }
    let (b, c, f, h, w) = x.dims5()?;
    x.reshape(&[b, c, f, h / p, p, w / p, p])?
        .permute([0, 1, 6, 4, 2, 3, 5])?
        .contiguous()?
        .reshape((b, c * p * p, f, h / p, w / p))
}

// Inverse of `patchify`: [B, C*p*p, F, H, W] -> [B, C, F, H*p, W*p].
fn unpatchify(x: &Tensor, p: usize) -> Result<Tensor> {
    if p == 1 {
        return Ok(x.clone());
    }
    let (b, cpp, f, h, w) = x.dims5()?;
    let c = cpp / (p * p);
    x.reshape(&[b, c, p, p, f, h, w])?
        .permute([0, 1, 4, 5, 3, 6, 2])?
        .contiguous()?
        .reshape((b, c, f, h * p, w * p))
}

// Param-free residual downsample: group-average the input over the (factor_t, factor_s, factor_s)
// block so the skip matches the main path's downsampled shape.
struct AvgDown3D {
    factor_t: usize,
    factor_s: usize,
    out_channels: usize,
    group_size: usize,
}

impl AvgDown3D {
    fn new(in_channels: usize, out_channels: usize, factor_t: usize, factor_s: usize) -> Self {
        let factor = factor_t * factor_s * factor_s;
        Self {
            factor_t,
            factor_s,
            out_channels,
            group_size: in_channels * factor / out_channels,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t0 = x.dim(2)?;
        let pad_t = (self.factor_t - t0 % self.factor_t) % self.factor_t;
        let x = if pad_t > 0 {
            x.pad_with_zeros(2, pad_t, 0)?
        } else {
            x.clone()
        };
        let (b, c, t, h, w) = x.dims5()?;
        let (ft, fs) = (self.factor_t, self.factor_s);
        x.reshape(&[b, c, t / ft, ft, h / fs, fs, w / fs, fs])?
            .permute([0, 1, 3, 5, 7, 2, 4, 6])?
            .contiguous()?
            .reshape((
                b,
                self.out_channels,
                self.group_size,
                t / ft,
                h / fs,
                w / fs,
            ))?
            .mean_keepdim(2)?
            .squeeze(2)
    }
}

// Param-free residual upsample: channel-duplicate then scatter into the (factor_t, factor_s,
// factor_s) upsampled grid. `first_chunk` drops the leading (factor_t-1) synthetic frames.
struct DupUp3D {
    factor_t: usize,
    factor_s: usize,
    out_channels: usize,
    repeats: usize,
}

impl DupUp3D {
    fn new(in_channels: usize, out_channels: usize, factor_t: usize, factor_s: usize) -> Self {
        let factor = factor_t * factor_s * factor_s;
        Self {
            factor_t,
            factor_s,
            out_channels,
            repeats: out_channels * factor / in_channels,
        }
    }

    fn forward(&self, x: &Tensor, first_chunk: bool) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let x = x
            .unsqueeze(2)?
            .broadcast_as((b, c, self.repeats, t, h, w))?
            .reshape((b, c * self.repeats, t, h, w))?;
        let (ft, fs, oc) = (self.factor_t, self.factor_s, self.out_channels);
        let x = x
            .reshape(&[b, oc, ft, fs, fs, t, h, w])?
            .permute([0, 1, 5, 2, 6, 3, 7, 4])?
            .contiguous()?
            .reshape((b, oc, t * ft, h * fs, w * fs))?;
        if first_chunk && ft > 1 {
            let tt = t * ft;
            x.narrow(2, ft - 1, tt - (ft - 1))
        } else {
            Ok(x)
        }
    }
}

#[derive(Clone, Copy)]
enum SampleMode {
    Down2d,
    Down3d,
    Up2d,
    Up3d,
}

// Spatial conv (`resample.1`) plus an optional causal temporal conv (`time_conv`). Down: spatial
// then temporal; up: temporal then spatial. Streaming temporal state lives in `FeatCache`.
struct Resample {
    mode: SampleMode,
    conv: Conv2d,
    time_conv: Option<CausalConv3d>,
}

impl Resample {
    fn new(dim: usize, out_dim: usize, mode: SampleMode, vb: ShardedVarBuilder) -> Result<Self> {
        let conv = match mode {
            SampleMode::Down2d | SampleMode::Down3d => {
                let cfg = Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                };
                conv2d(dim, dim, 3, cfg, vb.pp("resample").pp(1))?
            }
            SampleMode::Up2d | SampleMode::Up3d => {
                let cfg = Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                };
                conv2d(dim, out_dim, 3, cfg, vb.pp("resample").pp(1))?
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

    fn temporal_up(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let tc = self.time_conv.as_ref().unwrap();
        let i = cache.advance();
        let slot = cache.slot(i).clone();
        if matches!(slot, Slot::Empty) {
            cache.set(i, Slot::Rep);
            return Ok(x.clone());
        }
        let t = x.dim(2)?;
        let take = CACHE_T.min(t);
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

// Diffusers `resnets.{j}`: norm1 -> silu -> conv1 -> norm2 -> silu -> conv2, + conv_shortcut skip.
// Cache is stashed from each conv's ACTUAL (post-norm/silu) input, matching the reference.
struct ResidualBlock {
    norm1: WanRmsNorm,
    conv1: CausalConv3d,
    norm2: WanRmsNorm,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    fn new(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let norm1 = WanRmsNorm::new(in_dim, false, vb.pp("norm1"))?;
        let conv1 = CausalConv3d::new(
            in_dim,
            out_dim,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
            vb.pp("conv1"),
        )?;
        let norm2 = WanRmsNorm::new(out_dim, false, vb.pp("norm2"))?;
        let conv2 = CausalConv3d::new(
            out_dim,
            out_dim,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
            vb.pp("conv2"),
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
                vb.pp("conv_shortcut"),
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
        let a = self.norm1.forward(x)?.silu()?;
        let prev = cache.step(&a)?;
        let a = self.conv1.forward(&a, prev.as_ref())?;
        let b = self.norm2.forward(&a)?.silu()?;
        let prev = cache.step(&b)?;
        let b = self.conv2.forward(&b, prev.as_ref())?;
        b + h
    }
}

struct ResidualDownBlock {
    avg: AvgDown3D,
    resnets: Vec<ResidualBlock>,
    downsampler: Option<Resample>,
}

impl ResidualDownBlock {
    fn new(
        in_dim: usize,
        out_dim: usize,
        num_res: usize,
        temporal_ds: bool,
        down_flag: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let avg = AvgDown3D::new(
            in_dim,
            out_dim,
            if temporal_ds { 2 } else { 1 },
            if down_flag { 2 } else { 1 },
        );
        let vb_r = vb.pp("resnets");
        let mut resnets = Vec::with_capacity(num_res);
        let mut d = in_dim;
        for j in 0..num_res {
            resnets.push(ResidualBlock::new(d, out_dim, vb_r.pp(j))?);
            d = out_dim;
        }
        let downsampler = if down_flag {
            let mode = if temporal_ds {
                SampleMode::Down3d
            } else {
                SampleMode::Down2d
            };
            Some(Resample::new(out_dim, out_dim, mode, vb.pp("downsampler"))?)
        } else {
            None
        };
        Ok(Self {
            avg,
            resnets,
            downsampler,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let x_copy = x.clone();
        let mut y = x.clone();
        for r in &self.resnets {
            y = r.forward(&y, cache)?;
        }
        if let Some(ds) = &self.downsampler {
            y = ds.forward(&y, cache)?;
        }
        y + self.avg.forward(&x_copy)?
    }
}

struct ResidualUpBlock {
    dup: Option<DupUp3D>,
    resnets: Vec<ResidualBlock>,
    upsampler: Option<Resample>,
}

impl ResidualUpBlock {
    fn new(
        in_dim: usize,
        out_dim: usize,
        num_res: usize,
        temporal_us: bool,
        up_flag: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let dup = if up_flag {
            Some(DupUp3D::new(
                in_dim,
                out_dim,
                if temporal_us { 2 } else { 1 },
                2,
            ))
        } else {
            None
        };
        let vb_r = vb.pp("resnets");
        let mut resnets = Vec::with_capacity(num_res + 1);
        let mut d = in_dim;
        for j in 0..num_res + 1 {
            resnets.push(ResidualBlock::new(d, out_dim, vb_r.pp(j))?);
            d = out_dim;
        }
        let upsampler = if up_flag {
            let mode = if temporal_us {
                SampleMode::Up3d
            } else {
                SampleMode::Up2d
            };
            Some(Resample::new(out_dim, out_dim, mode, vb.pp("upsampler"))?)
        } else {
            None
        };
        Ok(Self {
            dup,
            resnets,
            upsampler,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache, first_chunk: bool) -> Result<Tensor> {
        let x_copy = x.clone();
        let mut y = x.clone();
        for r in &self.resnets {
            y = r.forward(&y, cache)?;
        }
        if let Some(us) = &self.upsampler {
            y = us.forward(&y, cache)?;
        }
        match &self.dup {
            Some(dup) => y + dup.forward(&x_copy, first_chunk)?,
            None => Ok(y),
        }
    }
}

// resnet -> (attn -> resnet) x num_layers.
struct MidBlock {
    resnets: Vec<ResidualBlock>,
    attentions: Vec<AttentionBlock>,
}

impl MidBlock {
    fn new(dim: usize, num_layers: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let vb_r = vb.pp("resnets");
        let vb_a = vb.pp("attentions");
        let mut resnets = vec![ResidualBlock::new(dim, dim, vb_r.pp(0))?];
        let mut attentions = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            attentions.push(AttentionBlock::new(dim, vb_a.pp(i))?);
            resnets.push(ResidualBlock::new(dim, dim, vb_r.pp(i + 1))?);
        }
        Ok(Self {
            resnets,
            attentions,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let mut y = self.resnets[0].forward(x, cache)?;
        for (a, r) in self.attentions.iter().zip(self.resnets[1..].iter()) {
            y = a.forward(&y)?;
            y = r.forward(&y, cache)?;
        }
        Ok(y)
    }
}

struct Encoder {
    conv_in: CausalConv3d,
    down_blocks: Vec<ResidualDownBlock>,
    mid: MidBlock,
    norm_out: WanRmsNorm,
    conv_out: CausalConv3d,
}

impl Encoder {
    fn new(cfg: &Wan22VaeConfig, in_ch: usize, z2: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let base = cfg.base_dim;
        let dims: Vec<usize> = std::iter::once(1)
            .chain(cfg.dim_mult.iter().copied())
            .map(|u| base * u)
            .collect();
        let conv_in = CausalConv3d::new(
            in_ch,
            dims[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
            vb.pp("conv_in"),
        )?;
        let n = cfg.dim_mult.len();
        let vb_d = vb.pp("down_blocks");
        let mut down_blocks = Vec::with_capacity(n);
        for i in 0..n {
            let temporal_ds = if i != n - 1 {
                cfg.temperal_downsample[i]
            } else {
                false
            };
            down_blocks.push(ResidualDownBlock::new(
                dims[i],
                dims[i + 1],
                cfg.num_res_blocks,
                temporal_ds,
                i != n - 1,
                vb_d.pp(i),
            )?);
        }
        let mid_dim = *dims.last().unwrap();
        let mid = MidBlock::new(mid_dim, 1, vb.pp("mid_block"))?;
        let norm_out = WanRmsNorm::new(mid_dim, false, vb.pp("norm_out"))?;
        let conv_out = CausalConv3d::new(
            mid_dim,
            z2,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
            vb.pp("conv_out"),
        )?;
        Ok(Self {
            conv_in,
            down_blocks,
            mid,
            norm_out,
            conv_out,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let prev = cache.step(x)?;
        let mut h = self.conv_in.forward(x, prev.as_ref())?;
        for db in &self.down_blocks {
            h = db.forward(&h, cache)?;
        }
        h = self.mid.forward(&h, cache)?;
        h = self.norm_out.forward(&h)?.silu()?;
        let prev = cache.step(&h)?;
        self.conv_out.forward(&h, prev.as_ref())
    }
}

struct Decoder {
    conv_in: CausalConv3d,
    mid: MidBlock,
    up_blocks: Vec<ResidualUpBlock>,
    norm_out: WanRmsNorm,
    conv_out: CausalConv3d,
}

impl Decoder {
    fn new(cfg: &Wan22VaeConfig, z: usize, out_ch: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let base = cfg.decoder_base_dim;
        let mut mults = vec![*cfg.dim_mult.last().unwrap()];
        mults.extend(cfg.dim_mult.iter().rev().copied());
        let dims: Vec<usize> = mults.iter().map(|u| base * u).collect();
        let conv_in = CausalConv3d::new(
            z,
            dims[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
            vb.pp("conv_in"),
        )?;
        let mid = MidBlock::new(dims[0], 1, vb.pp("mid_block"))?;
        let temporal_up: Vec<bool> = cfg.temperal_downsample.iter().rev().copied().collect();
        let n = cfg.dim_mult.len();
        let vb_u = vb.pp("up_blocks");
        let mut up_blocks = Vec::with_capacity(n);
        for i in 0..n {
            let up_flag = i != n - 1;
            let temporal_us = if up_flag { temporal_up[i] } else { false };
            up_blocks.push(ResidualUpBlock::new(
                dims[i],
                dims[i + 1],
                cfg.num_res_blocks,
                temporal_us,
                up_flag,
                vb_u.pp(i),
            )?);
        }
        let base_out = *dims.last().unwrap();
        let norm_out = WanRmsNorm::new(base_out, false, vb.pp("norm_out"))?;
        let conv_out = CausalConv3d::new(
            base_out,
            out_ch,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
            vb.pp("conv_out"),
        )?;
        Ok(Self {
            conv_in,
            mid,
            up_blocks,
            norm_out,
            conv_out,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache, first_chunk: bool) -> Result<Tensor> {
        let prev = cache.step(x)?;
        let mut h = self.conv_in.forward(x, prev.as_ref())?;
        h = self.mid.forward(&h, cache)?;
        for ub in &self.up_blocks {
            h = ub.forward(&h, cache, first_chunk)?;
        }
        h = self.norm_out.forward(&h)?.silu()?;
        let prev = cache.step(&h)?;
        self.conv_out.forward(&h, prev.as_ref())
    }
}

/// Wan2.2 high-compression 3D causal VAE. `encode`/`decode` walk the video in Wan's autoregressive
/// frame windows (1,4,4,...) so the causal temporal convs see the right left-context.
pub struct Wan22Vae {
    encoder: Encoder,
    quant_conv: CausalConv3d,
    post_quant_conv: CausalConv3d,
    decoder: Decoder,
    z_dim: usize,
    patch_size: usize,
    mean: Tensor,
    std: Tensor,
}

impl Wan22Vae {
    pub fn new(cfg: &Wan22VaeConfig, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        let z = cfg.z_dim;
        let encoder = Encoder::new(cfg, cfg.in_channels, z * 2, vb.pp("encoder"))?;
        let quant_conv = CausalConv3d::new(
            z * 2,
            z * 2,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            true,
            vb.pp("quant_conv"),
        )?;
        let post_quant_conv = CausalConv3d::new(
            z,
            z,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            true,
            vb.pp("post_quant_conv"),
        )?;
        let decoder = Decoder::new(cfg, z, cfg.out_channels, vb.pp("decoder"))?;
        let mean = Tensor::from_slice(&cfg.latents_mean, (1, z, 1, 1, 1), device)?;
        let std = Tensor::from_slice(&cfg.latents_std, (1, z, 1, 1, 1), device)?;
        Ok(Self {
            encoder,
            quant_conv,
            post_quant_conv,
            decoder,
            z_dim: z,
            patch_size: cfg.patch_size,
            mean,
            std,
        })
    }

    pub fn z_dim(&self) -> usize {
        self.z_dim
    }

    /// RGB video `[B,3,F,H,W]` in `[-1,1]` -> normalized latent mode `[B,z,(F-1)/4+1,H/16,W/16]`.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let x = patchify(x, self.patch_size)?;
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
        let moments = self.quant_conv.forward(&out.unwrap(), None)?;
        let mu = moments.narrow(1, 0, self.z_dim)?;
        let mean = self.mean.to_dtype(mu.dtype())?;
        let std = self.std.to_dtype(mu.dtype())?;
        mu.broadcast_sub(&mean)?.broadcast_div(&std)
    }

    /// Normalized latent `[B,z,T,H,W]` -> RGB video `[B,3,F,H,W]` in `[-1,1]`, F = 1+4*(T-1).
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let mean = self.mean.to_dtype(z.dtype())?;
        let std = self.std.to_dtype(z.dtype())?;
        let z = z.broadcast_mul(&std)?.broadcast_add(&mean)?;
        let x = self.post_quant_conv.forward(&z, None)?;
        let iters = x.dim(2)?;
        let mut cache = FeatCache::new();
        let mut out: Option<Tensor> = None;
        for i in 0..iters {
            cache.rewind();
            let chunk = x.narrow(2, i, 1)?;
            let o = self.decoder.forward(&chunk, &mut cache, i == 0)?;
            out = Some(match out {
                None => o,
                Some(p) => Tensor::cat(&[&p, &o], 2)?,
            });
        }
        let out = unpatchify(&out.unwrap(), self.patch_size)?;
        out.clamp(-1f32, 1f32)
    }
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

    fn assert_finite(t: &Tensor) -> Result<()> {
        let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(v.iter().all(|x| x.is_finite()), "non-finite VAE output");
        let (mn, mx) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        assert!(mx != mn, "degenerate constant output");
        Ok(())
    }

    // Patchify/unpatchify must be exact inverses (spatial pixel-(un)shuffle round-trip).
    #[test]
    fn patchify_roundtrip() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f64, 1.0, (1, 3, 5, 8, 8), &dev)?.to_dtype(DType::F32)?;
        let p = patchify(&x, 2)?;
        assert_eq!(p.dims(), &[1, 12, 5, 4, 4]);
        let back = unpatchify(&p, 2)?;
        let diff = (&x - &back)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6, "patchify not invertible, max diff {diff}");
        Ok(())
    }

    // 16x16x4 compression: 9 frames -> (9-1)/4+1 = 3 latent frames; 64px -> 64/16 = 4 latent spatial.
    #[test]
    fn wan22_vae_shapes_and_compression() -> Result<()> {
        let dev = Device::Cpu;
        let vb = ShardedSafeTensors::wrap(Box::new(RandnBackend), DType::F32, dev.clone());
        let cfg = Wan22VaeConfig::tiny();
        let vae = Wan22Vae::new(&cfg, vb, &dev)?;
        let x = Tensor::randn(0f64, 1.0, (1, 3, 9, 64, 64), &dev)?.to_dtype(DType::F32)?;
        let z = vae.encode(&x)?;
        assert_eq!(
            z.dims(),
            &[1, cfg.z_dim, 3, 4, 4],
            "latent shape / compression"
        );
        assert_finite(&z)?;
        let recon = vae.decode(&z)?;
        assert_eq!(
            recon.dims(),
            &[1, 3, 9, 64, 64],
            "decode recovers frame count + resolution"
        );
        assert_finite(&recon)?;
        Ok(())
    }

    fn cosine(a: &Tensor, b: &Tensor) -> Result<f64> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += *x as f64 * *y as f64;
            na += (*x as f64).powi(2);
            nb += (*y as f64).powi(2);
        }
        Ok(dot / (na.sqrt() * nb.sqrt()))
    }

    fn psnr(a: &Tensor, b: &Tensor) -> Result<f64> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mse: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| ((*x - *y) as f64).powi(2))
            .sum::<f64>()
            / a.len() as f64;
        Ok(10.0 * (4.0 / mse).log10()) // peak-to-peak 2.0 for [-1,1]
    }

    // Real-weight parity vs the diffusers oracle. Set WAN_VAE_SAFETENSORS (the vae weights file) and
    // WAN_VAE_ORACLE (a safetensors of x/z/recon from wan_vae_oracle.py); skips when unset.
    #[test]
    fn vae_parity_vs_oracle() -> Result<()> {
        use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
        use std::path::PathBuf;
        use std::sync::Arc;

        let (Ok(weights), Ok(oracle_path)) = (
            std::env::var("WAN_VAE_SAFETENSORS"),
            std::env::var("WAN_VAE_ORACLE"),
        ) else {
            eprintln!("skip vae_parity_vs_oracle: set WAN_VAE_SAFETENSORS + WAN_VAE_ORACLE");
            return Ok(());
        };
        let dev = Device::Cpu;
        let vb = from_mmaped_safetensors(
            vec![PathBuf::from(weights)],
            Vec::new(),
            Some(DType::F32),
            &dev,
            vec![None],
            true,
            None,
            |_| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )?;
        let vae = Wan22Vae::new(&Wan22VaeConfig::ti2v_5b(), vb, &dev)?;
        let oracle = hanzo_ml::safetensors::load(&oracle_path, &dev)?;
        let x = oracle["x"].to_dtype(DType::F32)?;
        let z_oracle = oracle["z"].to_dtype(DType::F32)?;
        let recon_oracle = oracle["recon"].to_dtype(DType::F32)?;

        let z_rust = vae.encode(&x)?;
        let cos = cosine(&z_rust, &z_oracle)?;
        let psnr_decode = psnr(&vae.decode(&z_oracle)?, &recon_oracle)?;
        let psnr_roundtrip = psnr(&vae.decode(&z_rust)?, &x)?;
        eprintln!(
            "VAE parity: latent cosine={cos:.6}, decode PSNR={psnr_decode:.2}dB, roundtrip PSNR={psnr_roundtrip:.2}dB"
        );
        assert!(cos > 0.999, "latent cosine {cos} <= 0.999");
        assert!(psnr_decode > 35.0, "decode PSNR {psnr_decode} <= 35dB");
        Ok(())
    }
}
