#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
//! ADaMoSHiFiGANV1 music vocoder: mel -> waveform. ConvNeXt backbone + HiFi-GAN head.
//! Port of acestep/music_dcae/music_vocoder.py (Apache-2.0).

use hanzo_ml::{Result, Tensor, D};
use hanzo_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, Module};
use hanzo_quant::ShardedVarBuilder;

use crate::layers::{conv1d, linear};

const BACKBONE_DEPTHS: [usize; 4] = [3, 3, 9, 3];
const BACKBONE_DIMS: [usize; 4] = [128, 256, 384, 512];
const CONVNEXT_KERNEL: usize = 7;
const CONVNEXT_MLP_RATIO: usize = 4;
const LN_EPS: f64 = 1e-6;
const UPSAMPLE_RATES: [usize; 7] = [4, 4, 2, 2, 2, 2, 2];
const UPSAMPLE_KERNELS: [usize; 7] = [8, 8, 4, 4, 4, 4, 4];
const RESBLOCK_KERNELS: [usize; 4] = [3, 7, 11, 13];
const RESBLOCK_DILATIONS: [[usize; 3]; 4] = [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]];
const HEAD_IN_CHANNELS: usize = 512;
const UPSAMPLE_INITIAL: usize = 1024;
const PRE_POST_KERNEL: usize = 13;

fn get_padding(kernel: usize, dilation: usize) -> usize {
    (kernel * dilation - dilation) / 2
}

// PyTorch weight_norm: w = g * v / ||v|| where the L2 norm is over every dim except 0.
fn weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    let rank = v.dims().len();
    let mut norm = v.sqr()?;
    for d in (1..rank).rev() {
        norm = norm.sum_keepdim(d)?;
    }
    v.broadcast_mul(g)?.broadcast_div(&norm.sqrt()?)
}

fn wn_conv1d(
    in_c: usize,
    out_c: usize,
    kernel: usize,
    cfg: Conv1dConfig,
    vb: ShardedVarBuilder,
) -> Result<Conv1d> {
    let g = vb.get((out_c, 1, 1), "weight_g")?;
    let v = vb.get((out_c, in_c / cfg.groups, kernel), "weight_v")?;
    let b = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight_norm(&g, &v)?, Some(b), cfg))
}

fn wn_conv_transpose1d(
    in_c: usize,
    out_c: usize,
    kernel: usize,
    cfg: ConvTranspose1dConfig,
    vb: ShardedVarBuilder,
) -> Result<ConvTranspose1d> {
    let g = vb.get((in_c, 1, 1), "weight_g")?;
    let v = vb.get((in_c, out_c, kernel), "weight_v")?;
    let b = vb.get(out_c, "bias")?;
    Ok(ConvTranspose1d::new(weight_norm(&g, &v)?, Some(b), cfg))
}

// Channels-first LayerNorm over C for a (B, C, L) tensor.
#[derive(Debug, Clone)]
struct LayerNormCf {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNormCf {
    fn new(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = vb.get(dim, "bias")?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = x.dim(1)?;
        let u = x.mean_keepdim(1)?;
        let xc = x.broadcast_sub(&u)?;
        let s = xc.sqr()?.mean_keepdim(1)?;
        let xn = xc.broadcast_div(&(s + LN_EPS)?.sqrt()?)?;
        let w = self.weight.reshape((1, c, 1))?;
        let b = self.bias.reshape((1, c, 1))?;
        xn.broadcast_mul(&w)?.broadcast_add(&b)
    }
}

#[derive(Debug, Clone)]
struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: hanzo_nn::Linear,
    pwconv2: hanzo_nn::Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn new(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: (CONVNEXT_KERNEL - 1) / 2,
            groups: dim,
            ..Default::default()
        };
        let dwconv = conv1d(dim, dim, CONVNEXT_KERNEL, cfg, vb.pp("dwconv"))?;
        let nw = vb.get(dim, "norm.weight")?;
        let nb = vb.get(dim, "norm.bias")?;
        let norm = LayerNorm::new(nw, nb, LN_EPS);
        let pwconv1 = linear(dim, CONVNEXT_MLP_RATIO * dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(CONVNEXT_MLP_RATIO * dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let mut h = self.dwconv.forward(x)?;
        h = h.permute((0, 2, 1))?.contiguous()?;
        h = self.norm.forward(&h)?;
        h = self.pwconv1.forward(&h)?;
        h = h.gelu_erf()?;
        h = self.pwconv2.forward(&h)?;
        h = h.broadcast_mul(&self.gamma)?;
        h = h.permute((0, 2, 1))?.contiguous()?;
        h + residual
    }
}

#[derive(Debug, Clone)]
struct Backbone {
    stem_conv: Conv1d,
    stem_norm: LayerNormCf,
    mids: Vec<(LayerNormCf, Conv1d)>,
    stages: Vec<Vec<ConvNeXtBlock>>,
    norm: LayerNormCf,
}

impl Backbone {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        let cl = vb.pp("channel_layers");
        let stem_conv = conv1d(
            BACKBONE_DIMS[0],
            BACKBONE_DIMS[0],
            CONVNEXT_KERNEL,
            Conv1dConfig::default(),
            cl.pp("0").pp("0"),
        )?;
        let stem_norm = LayerNormCf::new(BACKBONE_DIMS[0], cl.pp("0").pp("1"))?;
        let mut mids = Vec::new();
        for i in 0..BACKBONE_DIMS.len() - 1 {
            let m = cl.pp((i + 1).to_string());
            let norm = LayerNormCf::new(BACKBONE_DIMS[i], m.pp("0"))?;
            let conv = conv1d(
                BACKBONE_DIMS[i],
                BACKBONE_DIMS[i + 1],
                1,
                Conv1dConfig::default(),
                m.pp("1"),
            )?;
            mids.push((norm, conv));
        }
        let mut stages = Vec::new();
        let sv = vb.pp("stages");
        for (i, &depth) in BACKBONE_DEPTHS.iter().enumerate() {
            let s = sv.pp(i.to_string());
            let mut blocks = Vec::new();
            for j in 0..depth {
                blocks.push(ConvNeXtBlock::new(BACKBONE_DIMS[i], s.pp(j.to_string()))?);
            }
            stages.push(blocks);
        }
        let norm = LayerNormCf::new(BACKBONE_DIMS[3], vb.pp("norm"))?;
        Ok(Self {
            stem_conv,
            stem_norm,
            mids,
            stages,
            norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // stem conv uses replicate padding
        let pad = (CONVNEXT_KERNEL - 1) / 2;
        let mut h = x.pad_with_same(D::Minus1, pad, pad)?;
        h = self.stem_conv.forward(&h)?;
        h = self.stem_norm.forward(&h)?;
        for blk in &self.stages[0] {
            h = blk.forward(&h)?;
        }
        for (i, (norm, conv)) in self.mids.iter().enumerate() {
            h = norm.forward(&h)?;
            h = conv.forward(&h)?;
            for blk in &self.stages[i + 1] {
                h = blk.forward(&h)?;
            }
        }
        self.norm.forward(&h)
    }
}

#[derive(Debug, Clone)]
struct ResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
}

impl ResBlock1 {
    fn new(
        channels: usize,
        kernel: usize,
        dilation: [usize; 3],
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let c1 = vb.pp("convs1");
        let c2 = vb.pp("convs2");
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        for (m, &d) in dilation.iter().enumerate() {
            convs1.push(wn_conv1d(
                channels,
                channels,
                kernel,
                Conv1dConfig {
                    padding: get_padding(kernel, d),
                    dilation: d,
                    ..Default::default()
                },
                c1.pp(m.to_string()),
            )?);
            convs2.push(wn_conv1d(
                channels,
                channels,
                kernel,
                Conv1dConfig {
                    padding: get_padding(kernel, 1),
                    ..Default::default()
                },
                c2.pp(m.to_string()),
            )?);
        }
        Ok(Self { convs1, convs2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (c1, c2) in self.convs1.iter().zip(self.convs2.iter()) {
            let xt = c1.forward(&x.silu()?)?;
            let xt = c2.forward(&xt.silu()?)?;
            x = (xt + x)?;
        }
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct Head {
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<ResBlock1>,
    conv_post: Conv1d,
}

impl Head {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        let conv_pre = wn_conv1d(
            HEAD_IN_CHANNELS,
            UPSAMPLE_INITIAL,
            PRE_POST_KERNEL,
            Conv1dConfig {
                padding: get_padding(PRE_POST_KERNEL, 1),
                ..Default::default()
            },
            vb.pp("conv_pre"),
        )?;
        let upv = vb.pp("ups");
        let rbv = vb.pp("resblocks");
        let mut ups = Vec::new();
        let mut resblocks = Vec::new();
        for i in 0..UPSAMPLE_RATES.len() {
            let in_c = UPSAMPLE_INITIAL >> i;
            let out_c = UPSAMPLE_INITIAL >> (i + 1);
            let (k, u) = (UPSAMPLE_KERNELS[i], UPSAMPLE_RATES[i]);
            ups.push(wn_conv_transpose1d(
                in_c,
                out_c,
                k,
                ConvTranspose1dConfig {
                    padding: (k - u) / 2,
                    stride: u,
                    ..Default::default()
                },
                upv.pp(i.to_string()),
            )?);
            for j in 0..RESBLOCK_KERNELS.len() {
                resblocks.push(ResBlock1::new(
                    out_c,
                    RESBLOCK_KERNELS[j],
                    RESBLOCK_DILATIONS[j],
                    rbv.pp((i * RESBLOCK_KERNELS.len() + j).to_string()),
                )?);
            }
        }
        let last_c = UPSAMPLE_INITIAL >> UPSAMPLE_RATES.len();
        let conv_post = wn_conv1d(
            last_c,
            1,
            PRE_POST_KERNEL,
            Conv1dConfig {
                padding: get_padding(PRE_POST_KERNEL, 1),
                ..Default::default()
            },
            vb.pp("conv_post"),
        )?;
        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            conv_post,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n_kernels = RESBLOCK_KERNELS.len();
        let mut x = self.conv_pre.forward(x)?;
        for (i, up) in self.ups.iter().enumerate() {
            x = up.forward(&x.silu()?)?;
            let mut xs = self.resblocks[i * n_kernels].forward(&x)?;
            for j in 1..n_kernels {
                xs = (xs + self.resblocks[i * n_kernels + j].forward(&x)?)?;
            }
            x = (xs / n_kernels as f64)?;
        }
        x = self.conv_post.forward(&x.silu()?)?;
        x.tanh()
    }
}

#[derive(Debug, Clone)]
pub struct Vocoder {
    backbone: Backbone,
    head: Head,
}

impl Vocoder {
    pub fn new(vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            backbone: Backbone::new(vb.pp("backbone"))?,
            head: Head::new(vb.pp("head"))?,
        })
    }

    /// mel (B, 128, T) -> waveform (B, 1, T*512).
    pub fn decode(&self, mel: &Tensor) -> Result<Tensor> {
        let y = self.backbone.forward(mel)?;
        self.head.forward(&y)
    }
}
