#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! TAESD (Tiny AutoEncoder for Stable Diffusion), madebyollin/taesd. A distilled 3x3-conv + ReLU
//! stack that stands in for the full SD-VAE: no GroupNorm, no attention, ~1000x fewer params. Both
//! halves are optional fast paths for MuseTalk. The encoder is the realtime lever: MuseTalk
//! re-encodes a masked + reference face every frame, and the full VAE encoder's high-res conv-GEMMs
//! are the profiled wall. TAESD shares the SD-VAE raw latent space, so the UNet is unchanged; the
//! only cost is the distillation's fidelity, gated by LSE-C. Quality parity needs the distilled
//! `taesd_encoder`/`taesd_decoder` weights trained against the same latent space.

use hanzo_ml::{DType, Result, Tensor};
use hanzo_nn::{Conv2d, Conv2dConfig, Module};
use hanzo_quant::{Convolution, ShardedVarBuilder};

use crate::layers::{conv2d, conv2d_no_bias};

use super::config::VaeConfig;

const HIDDEN: usize = 64;
const STAGES: usize = 3;
const BLOCKS_PER_STAGE: usize = 3;

fn conv3(in_c: usize, out_c: usize, vb: ShardedVarBuilder) -> Result<Conv2d> {
    conv2d(
        in_c,
        out_c,
        3,
        Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        vb,
    )
}

fn resample_conv(stride: usize, vb: ShardedVarBuilder) -> Result<Conv2d> {
    conv2d_no_bias(
        HIDDEN,
        HIDDEN,
        3,
        Conv2dConfig {
            padding: 1,
            stride,
            ..Default::default()
        },
        vb,
    )
}

/// TAESD residual block: conv-relu-conv-relu-conv fused with a 1x1 (or identity) skip, then relu.
#[derive(Debug, Clone)]
struct Block {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    skip: Option<Conv2d>,
}

impl Block {
    fn new(in_c: usize, out_c: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv = vb.pp("conv");
        let conv1 = conv3(in_c, out_c, conv.pp(0))?;
        let conv2 = conv3(out_c, out_c, conv.pp(2))?;
        let conv3 = conv3(out_c, out_c, conv.pp(4))?;
        let skip = if in_c == out_c {
            None
        } else {
            Some(conv2d_no_bias(
                in_c,
                out_c,
                1,
                Default::default(),
                vb.pp("skip"),
            )?)
        };
        Ok(Self {
            conv1,
            conv2,
            conv3,
            skip,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = Convolution.forward_2d(&self.conv1, xs)?.relu()?;
        let h = Convolution.forward_2d(&self.conv2, &h)?.relu()?;
        let h = Convolution.forward_2d(&self.conv3, &h)?;
        let skip = match self.skip.as_ref() {
            None => xs.clone(),
            Some(c) => Convolution.forward_2d(c, xs)?,
        };
        (h + skip)?.relu()
    }
}

/// TAESD encoder: conv(3->64) -> block -> 3 downsample stages (stride-2 conv + 3 blocks) ->
/// conv(64->latent). Mirrors the reference `Encoder` Sequential index-for-index.
#[derive(Debug, Clone)]
pub struct TaesdEncoder {
    conv_in: Conv2d,
    block0: Block,
    downsamplers: Vec<Conv2d>,
    blocks: Vec<Block>,
    conv_out: Conv2d,
    scaling_factor: f64,
}

impl TaesdEncoder {
    pub fn new(
        in_channels: usize,
        latent_channels: usize,
        scaling_factor: f64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let conv_in = conv3(in_channels, HIDDEN, vb.pp(0))?;
        let block0 = Block::new(HIDDEN, HIDDEN, vb.pp(1))?;
        let mut idx = 2usize;
        let mut downsamplers = Vec::with_capacity(STAGES);
        let mut blocks = Vec::with_capacity(STAGES * BLOCKS_PER_STAGE);
        for _ in 0..STAGES {
            downsamplers.push(resample_conv(2, vb.pp(idx))?);
            idx += 1;
            for _ in 0..BLOCKS_PER_STAGE {
                blocks.push(Block::new(HIDDEN, HIDDEN, vb.pp(idx))?);
                idx += 1;
            }
        }
        let conv_out = conv3(HIDDEN, latent_channels, vb.pp(idx))?;
        Ok(Self {
            conv_in,
            block0,
            downsamplers,
            blocks,
            conv_out,
            scaling_factor,
        })
    }

    /// Encode a `[0,1]` RGB face into UNet latents. TAESD emits RAW SD-VAE latents, so scale by
    /// `scaling_factor` to match the pipeline's scaled-latent convention (the full VAE's
    /// `encode_mode` multiplies its mean by the same factor).
    pub fn encode(&self, img: &Tensor) -> Result<Tensor> {
        let mut h = Convolution.forward_2d(&self.conv_in, img)?;
        h = self.block0.forward(&h)?;
        let mut bi = 0usize;
        for ds in self.downsamplers.iter() {
            h = Convolution.forward_2d(ds, &h)?;
            for _ in 0..BLOCKS_PER_STAGE {
                h = self.blocks[bi].forward(&h)?;
                bi += 1;
            }
        }
        Convolution.forward_2d(&self.conv_out, &h)? * self.scaling_factor
    }
}

/// TAESD decoder: clamp -> conv(latent->64) -> relu -> 3 upsample stages (3 blocks + nearest-2x +
/// conv) -> block -> conv(64->3). Mirrors the reference `Decoder` Sequential index-for-index.
#[derive(Debug, Clone)]
pub struct TaesdDecoder {
    conv_in: Conv2d,
    blocks: Vec<Block>,
    upsamplers: Vec<Conv2d>,
    conv_out: Conv2d,
    scaling_factor: f64,
}

impl TaesdDecoder {
    pub fn new(
        latent_channels: usize,
        out_channels: usize,
        scaling_factor: f64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let conv_in = conv3(latent_channels, HIDDEN, vb.pp(1))?;
        let mut idx = 3usize; // after Clamp(0), conv_in(1), ReLU(2)
        let mut blocks = Vec::new();
        let mut upsamplers = Vec::new();
        for _ in 0..STAGES {
            for _ in 0..BLOCKS_PER_STAGE {
                blocks.push(Block::new(HIDDEN, HIDDEN, vb.pp(idx))?);
                idx += 1;
            }
            idx += 1; // parameter-free nn.Upsample
            upsamplers.push(resample_conv(1, vb.pp(idx))?);
            idx += 1;
        }
        blocks.push(Block::new(HIDDEN, HIDDEN, vb.pp(idx))?);
        idx += 1;
        let conv_out = conv3(HIDDEN, out_channels, vb.pp(idx))?;
        Ok(Self {
            conv_in,
            blocks,
            upsamplers,
            conv_out,
            scaling_factor,
        })
    }

    /// Decode scaled UNet latents to a `[0,1]` RGB image. Undo `scaling_factor` to reach the raw
    /// latent space TAESD decodes; the repo's 0-1 magnitude/shift is for the uint8 preview only, so
    /// it is intentionally omitted. The final conv emits `[0,1]` directly (no tanh/denorm).
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let mut h = (latents / self.scaling_factor)?;
        h = Convolution.forward_2d(&self.conv_in, &h)?.relu()?;
        let mut bi = 0usize;
        for up in self.upsamplers.iter() {
            for _ in 0..BLOCKS_PER_STAGE {
                h = self.blocks[bi].forward(&h)?;
                bi += 1;
            }
            let (_, _, hh, ww) = h.dims4()?;
            h = h.upsample_nearest2d(hh * 2, ww * 2)?;
            h = Convolution.forward_2d(up, &h)?;
        }
        h = self.blocks[bi].forward(&h)?;
        Convolution
            .forward_2d(&self.conv_out, &h)?
            .to_dtype(DType::F32)?
            .clamp(0f32, 1f32)
    }
}

/// The matched TAESD encode+decode pair sharing the SD-VAE latent space. Attaching one to a
/// `MuseTalk` swaps both high-res VAE passes for the tiny distilled stacks.
#[derive(Debug, Clone)]
pub struct Taesd {
    pub encoder: TaesdEncoder,
    pub decoder: TaesdDecoder,
}

impl Taesd {
    pub fn new(
        cfg: &VaeConfig,
        encoder_vb: ShardedVarBuilder,
        decoder_vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let encoder = TaesdEncoder::new(
            cfg.in_channels,
            cfg.latent_channels,
            cfg.scaling_factor,
            encoder_vb,
        )?;
        let decoder = TaesdDecoder::new(
            cfg.latent_channels,
            cfg.out_channels,
            cfg.scaling_factor,
            decoder_vb,
        )?;
        Ok(Self { encoder, decoder })
    }
}
