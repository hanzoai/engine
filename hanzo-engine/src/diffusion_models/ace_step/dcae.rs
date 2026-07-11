#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
//! MusicDCAE decoder: 8-channel latent -> 2-channel (stereo) log-mel image.
//! Port of diffusers `AutoencoderDC` decoder (Sana DCAE) as configured by
//! ACE-Step `music_dcae_f8c8` (Apache-2.0). Decode path only.

use hanzo_ml::{DType, Result, Tensor};
use hanzo_nn::{Conv2d, Conv2dConfig, Linear, Module};
use hanzo_quant::ShardedVarBuilder;

use crate::layers::{conv2d, conv2d_no_bias, linear_no_bias};

const LATENT_CHANNELS: usize = 8;
const BLOCK_OUT: [usize; 4] = [128, 256, 512, 1024];
const LAYERS_PER_BLOCK: usize = 3;
const ATTENTION_HEAD_DIM: usize = 32;
const OUT_CHANNELS: usize = 2;
const RMS_EPS: f64 = 1e-5;
const ATTN_EPS: f64 = 1e-15;
const QKV_KERNEL: usize = 5;

fn pad1() -> Conv2dConfig {
    Conv2dConfig {
        padding: 1,
        ..Default::default()
    }
}

// RMSNorm over the channel dim of (B, C, H, W), affine weight + bias.
fn rms_norm_channels(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let c = x.dim(1)?;
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(1)?;
    let normed = xf
        .broadcast_div(&(var + RMS_EPS)?.sqrt()?)?
        .to_dtype(x.dtype())?;
    let w = w.reshape((1, c, 1, 1))?;
    let b = b.reshape((1, c, 1, 1))?;
    normed.broadcast_mul(&w)?.broadcast_add(&b)
}

// repeat_interleave along channel dim: (B,C,H,W) -> (B,C*n,H,W), each channel repeated n times.
fn repeat_interleave_ch(x: &Tensor, n: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, c, n, h, w))?
        .contiguous()?
        .reshape((b, c * n, h, w))
}

// PyTorch pixel_shuffle: (B, c*r*r, H, W) -> (B, c, H*r, W*r).
fn pixel_shuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, cin, h, w) = x.dims4()?;
    let c = cin / (r * r);
    x.reshape((b, c, r, r, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .contiguous()?
        .reshape((b, c, h * r, w * r))
}

#[derive(Debug, Clone)]
struct ResBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    norm_w: Tensor,
    norm_b: Tensor,
}

impl ResBlock {
    fn new(channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv1 = conv2d(channels, channels, 3, pad1(), vb.pp("conv1"))?;
        let conv2 = conv2d_no_bias(channels, channels, 3, pad1(), vb.pp("conv2"))?;
        let norm_w = vb.get(channels, "norm.weight")?;
        let norm_b = vb.get(channels, "norm.bias")?;
        Ok(Self {
            conv1,
            conv2,
            norm_w,
            norm_b,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.conv1.forward(x)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;
        let h = rms_norm_channels(&h, &self.norm_w, &self.norm_b)?;
        h + residual
    }
}

// Upsample (interpolate nearest x2) + channel-halving conv, with pixel-shuffle shortcut.
#[derive(Debug, Clone)]
struct DCUpBlock {
    conv: Conv2d,
    repeats: usize,
}

impl DCUpBlock {
    fn new(in_c: usize, out_c: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv = conv2d(in_c, out_c, 3, pad1(), vb.pp("conv"))?;
        let repeats = out_c * 4 / in_c;
        Ok(Self { conv, repeats })
    }

    fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let (_, _, ih, iw) = h.dims4()?;
        let x = h.upsample_nearest2d(ih * 2, iw * 2)?;
        let x = self.conv.forward(&x)?;
        let y = repeat_interleave_ch(h, self.repeats)?;
        let y = pixel_shuffle(&y, 2)?;
        x + y
    }
}

// Depthwise conv2d (groups == channels), square kernel k, stride 1, pad p, vectorised across channels.
// candle lowers a grouped conv to one conv per group (up to 8192 launches in stage3); this is the same
// cross-correlation expressed as k*k shifted channel-broadcast multiply-adds. x [B,C,H,W], w [1,C,k,k].
fn depthwise_conv2d(x: &Tensor, w: &Tensor, p: usize) -> Result<Tensor> {
    let (_, _, h, wd) = x.dims4()?;
    let k = w.dim(2)?;
    let xp = x.pad_with_zeros(2, p, p)?.pad_with_zeros(3, p, p)?;
    let mut acc: Option<Tensor> = None;
    for kh in 0..k {
        let row = xp.narrow(2, kh, h)?;
        let wr = w.narrow(2, kh, 1)?;
        for kw in 0..k {
            let term = row.narrow(3, kw, wd)?.broadcast_mul(&wr.narrow(3, kw, 1)?)?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
    }
    Ok(acc.expect("kernel is at least 1x1"))
}

#[derive(Debug, Clone)]
struct GlumbConv {
    conv_inverted: Conv2d,
    depth_w: Tensor,
    depth_b: Tensor,
    conv_point: Conv2d,
    norm_w: Tensor,
    norm_b: Tensor,
}

impl GlumbConv {
    fn new(channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden = channels * 4;
        let conv_inverted = conv2d(
            channels,
            hidden * 2,
            1,
            Conv2dConfig::default(),
            vb.pp("conv_inverted"),
        )?;
        let dvb = vb.pp("conv_depth");
        let depth_w = dvb
            .get((hidden * 2, 1, 3, 3), "weight")?
            .reshape((1, hidden * 2, 3, 3))?;
        let depth_b = dvb.get(hidden * 2, "bias")?.reshape((1, hidden * 2, 1, 1))?;
        let conv_point = conv2d_no_bias(
            hidden,
            channels,
            1,
            Conv2dConfig::default(),
            vb.pp("conv_point"),
        )?;
        let norm_w = vb.get(channels, "norm.weight")?;
        let norm_b = vb.get(channels, "norm.bias")?;
        Ok(Self {
            conv_inverted,
            depth_w,
            depth_b,
            conv_point,
            norm_w,
            norm_b,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.conv_inverted.forward(x)?;
        let h = h.silu()?;
        let h = depthwise_conv2d(&h, &self.depth_w, 1)?.broadcast_add(&self.depth_b)?;
        let parts = h.chunk(2, 1)?;
        let h = (&parts[0] * parts[1].silu()?)?;
        let h = self.conv_point.forward(&h)?;
        let h = rms_norm_channels(&h, &self.norm_w, &self.norm_b)?;
        h + residual
    }
}

#[derive(Debug, Clone)]
struct MultiscaleProj {
    proj_in_w: Tensor,
    proj_out: Conv2d,
}

impl MultiscaleProj {
    fn new(inner: usize, num_heads: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let ch = 3 * inner;
        let proj_in_w = vb
            .pp("proj_in")
            .get((ch, 1, QKV_KERNEL, QKV_KERNEL), "weight")?
            .reshape((1, ch, QKV_KERNEL, QKV_KERNEL))?;
        let out_cfg = Conv2dConfig {
            groups: 3 * num_heads,
            ..Default::default()
        };
        let proj_out = conv2d_no_bias(ch, ch, 1, out_cfg, vb.pp("proj_out"))?;
        Ok(Self { proj_in_w, proj_out })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = depthwise_conv2d(x, &self.proj_in_w, QKV_KERNEL / 2)?;
        self.proj_out.forward(&h)
    }
}

// Sana lightweight multi-scale linear attention (residual).
#[derive(Debug, Clone)]
struct LinearAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    multiscale: MultiscaleProj,
    to_out: Linear,
    norm_w: Tensor,
    norm_b: Tensor,
    head_dim: usize,
}

impl LinearAttention {
    fn new(channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let head_dim = ATTENTION_HEAD_DIM;
        let num_heads = channels / head_dim;
        let inner = channels;
        let to_q = linear_no_bias(channels, inner, vb.pp("to_q"))?;
        let to_k = linear_no_bias(channels, inner, vb.pp("to_k"))?;
        let to_v = linear_no_bias(channels, inner, vb.pp("to_v"))?;
        let multiscale = MultiscaleProj::new(inner, num_heads, vb.pp("to_qkv_multiscale").pp("0"))?;
        let to_out = linear_no_bias(inner * 2, channels, vb.pp("to_out"))?;
        let norm_w = vb.get(channels, "norm_out.weight")?;
        let norm_b = vb.get(channels, "norm_out.bias")?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            multiscale,
            to_out,
            norm_w,
            norm_b,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let (b, _, h, w) = x.dims4()?;
        let hw = h * w;
        let orig_dtype = x.dtype();

        // Project q,k,v channel-last, restack to (B, 3*inner, H, W).
        let xp = x.permute((0, 2, 3, 1))?.contiguous()?;
        let q = self.to_q.forward(&xp)?;
        let k = self.to_k.forward(&xp)?;
        let v = self.to_v.forward(&xp)?;
        let qkv = Tensor::cat(&[&q, &k, &v], 3)?
            .permute((0, 3, 1, 2))?
            .contiguous()?;

        let ms = self.multiscale.forward(&qkv)?;
        let hidden = Tensor::cat(&[&qkv, &ms], 1)?.to_dtype(DType::F32)?;

        // (B, 2*3*inner, H, W) -> (B, groups, 3*head_dim, HW) -> chunk into q,k,v.
        let groups = hidden.dim(1)? / (3 * self.head_dim);
        let hidden = hidden.reshape((b, groups, 3 * self.head_dim, hw))?;
        let parts = hidden.chunk(3, 2)?;
        let q = parts[0].relu()?;
        let k = parts[1].relu()?;
        let v = &parts[2];

        // Linear attention with the appended row of ones as normaliser.
        let (bg, ng, _, _) = q.dims4()?;
        let ones = Tensor::ones((bg, ng, 1, hw), DType::F32, x.device())?;
        let v_pad = Tensor::cat(&[v, &ones], 2)?.contiguous()?;
        let scores = v_pad.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let out = scores.matmul(&q.contiguous()?)?;
        let num = out.narrow(2, 0, self.head_dim)?;
        let den = out.narrow(2, self.head_dim, 1)?;
        let out = num.broadcast_div(&(den + ATTN_EPS)?)?;

        let flat = out.dim(1)? * out.dim(2)?;
        let out = out.reshape((b, flat, h, w))?.to_dtype(orig_dtype)?;
        let outp = out.permute((0, 2, 3, 1))?.contiguous()?;
        let outp = self.to_out.forward(&outp)?;
        let out = outp.permute((0, 3, 1, 2))?.contiguous()?;
        let out = rms_norm_channels(&out, &self.norm_w, &self.norm_b)?;
        out + residual
    }
}

#[derive(Debug, Clone)]
struct EfficientViTBlock {
    attn: LinearAttention,
    conv_out: GlumbConv,
}

impl EfficientViTBlock {
    fn new(channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            attn: LinearAttention::new(channels, vb.pp("attn"))?,
            conv_out: GlumbConv::new(channels, vb.pp("conv_out"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.attn.forward(x)?;
        self.conv_out.forward(&x)
    }
}

#[derive(Debug, Clone)]
pub struct DcaeDecoder {
    conv_in: Conv2d,
    stage3: Vec<EfficientViTBlock>,
    upsamples: Vec<(DCUpBlock, Vec<ResBlock>)>,
    norm_w: Tensor,
    norm_b: Tensor,
    conv_out: Conv2d,
}

impl DcaeDecoder {
    pub fn new(vb: ShardedVarBuilder) -> Result<Self> {
        let conv_in = conv2d(LATENT_CHANNELS, BLOCK_OUT[3], 3, pad1(), vb.pp("conv_in"))?;
        let ub = vb.pp("up_blocks");

        // up_blocks.3: deepest stage, EfficientViT blocks at 1024, no upsample.
        let s3 = ub.pp("3");
        let mut stage3 = Vec::new();
        for i in 0..LAYERS_PER_BLOCK {
            stage3.push(EfficientViTBlock::new(BLOCK_OUT[3], s3.pp(i.to_string()))?);
        }

        // up_blocks.2/1/0: DCUpBlock (halving) then 3 ResBlocks, processed 2 -> 0.
        let mut upsamples = Vec::new();
        for i in (0..3).rev() {
            let s = ub.pp(i.to_string());
            let up = DCUpBlock::new(BLOCK_OUT[i + 1], BLOCK_OUT[i], s.pp("0"))?;
            let mut res = Vec::new();
            for j in 0..LAYERS_PER_BLOCK {
                res.push(ResBlock::new(BLOCK_OUT[i], s.pp((j + 1).to_string()))?);
            }
            upsamples.push((up, res));
        }

        let norm_w = vb.get(BLOCK_OUT[0], "norm_out.weight")?;
        let norm_b = vb.get(BLOCK_OUT[0], "norm_out.bias")?;
        let conv_out = conv2d(BLOCK_OUT[0], OUT_CHANNELS, 3, pad1(), vb.pp("conv_out"))?;
        Ok(Self {
            conv_in,
            stage3,
            upsamples,
            norm_w,
            norm_b,
            conv_out,
        })
    }

    /// latent (B, 8, H, W) -> mel image (B, 2, H*8, W*8).
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let shortcut = repeat_interleave_ch(latent, BLOCK_OUT[3] / LATENT_CHANNELS)?;
        let mut h = (self.conv_in.forward(latent)? + shortcut)?;
        for blk in &self.stage3 {
            h = blk.forward(&h)?;
        }
        for (up, res) in &self.upsamples {
            h = up.forward(&h)?;
            for blk in res {
                h = blk.forward(&h)?;
            }
        }
        let h = rms_norm_channels(&h, &self.norm_w, &self.norm_b)?;
        let h = h.relu()?;
        self.conv_out.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;

    // The depthwise_conv2d reformulation must match candle's grouped Conv2d for the kernels stage3 uses:
    // GlumbConv conv_depth (k3/p1) and MultiscaleProj proj_in (k5/p2).
    #[test]
    fn depthwise_conv2d_matches_grouped_conv() {
        let dev = Device::Cpu;
        for (k, p) in [(3usize, 1usize), (5usize, 2usize)] {
            let (b, c, h, w) = (2usize, 24usize, 7usize, 9usize);
            let x = Tensor::randn(0f32, 1.0, (b, c, h, w), &dev).unwrap();
            let wt = Tensor::randn(0f32, 1.0, (c, 1, k, k), &dev).unwrap();
            let cfg = Conv2dConfig {
                padding: p,
                groups: c,
                ..Default::default()
            };
            let reference = Conv2d::new(wt.clone(), None, cfg).forward(&x).unwrap();
            let mine = depthwise_conv2d(&x, &wt.reshape((1, c, k, k)).unwrap(), p).unwrap();
            let diff = (reference - mine)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(diff < 1e-4, "k{k} p{p} max abs diff {diff}");
        }
    }
}
