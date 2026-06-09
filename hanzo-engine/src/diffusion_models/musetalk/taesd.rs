#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! TAESD (Tiny AutoEncoder for Stable Diffusion) decoder, madebyollin/taesd.
//! A distilled 3x3-conv + ReLU stack that replaces the full SD-VAE decoder: no GroupNorm, no
//! attention, ~1000x fewer params. Used as an optional fast decode path for MuseTalk; the encode
//! still uses the full VAE so the UNet latent space is unchanged. Quality parity requires the
//! distilled `taesd_decoder` weights trained against the same VAE latent space.

use hanzo_ml::{Result, Tensor};
use hanzo_nn::{Conv2d, Conv2dConfig, Module};
use hanzo_quant::{Convolution, ShardedVarBuilder};

use crate::layers::{conv2d, conv2d_no_bias};

const TAESD_LATENT_MAGNITUDE: f64 = 3.0;
const TAESD_LATENT_SHIFT: f64 = 0.5;
const HIDDEN: usize = 64;

fn conv3(in_c: usize, out_c: usize, vb: ShardedVarBuilder) -> Result<Conv2d> {
    let cfg = Conv2dConfig {
        padding: 1,
        ..Default::default()
    };
    conv2d(in_c, out_c, 3, cfg, vb)
}

/// TAESD residual block: conv-relu-conv-relu-conv with a 1x1 (or identity) skip, fused with relu.
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
            Some(conv2d_no_bias(in_c, out_c, 1, Default::default(), vb.pp("skip"))?)
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

/// TAESD decoder: clamp -> conv(4->64) -> relu -> 3 upsample stages (3 blocks each) -> conv(64->3).
#[derive(Debug, Clone)]
pub struct TaesdDecoder {
    conv_in: Conv2d,
    blocks: Vec<Block>,
    upsamplers: Vec<Conv2d>,
    conv_out: Conv2d,
    scaling_factor: f64,
}

impl TaesdDecoder {
    pub fn new(latent_channels: usize, out_channels: usize, scaling_factor: f64, vb: ShardedVarBuilder) -> Result<Self> {
        // Flat Sequential layout matching the reference: index 1 conv_in, then per stage 3 blocks +
        // upsample-conv, then a final block + conv_out. We index the Sequential explicitly.
        let conv_in = conv3(latent_channels, HIDDEN, vb.pp(1))?;
        let mut idx = 3usize; // after Clamp(0), conv_in(1), ReLU(2)
        let mut blocks = Vec::new();
        let mut upsamplers = Vec::new();
        for _stage in 0..3 {
            for _ in 0..3 {
                blocks.push(Block::new(HIDDEN, HIDDEN, vb.pp(idx))?);
                idx += 1;
            }
            idx += 1; // Upsample (parameter-free nn.Upsample)
            upsamplers.push(conv2d_no_bias(HIDDEN, HIDDEN, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp(idx))?);
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

    /// Map a full-VAE-scaled latent (scaled by `vae.scaling_factor`) into TAESD's expected range.
    /// TAESD trains on latents normalized to roughly [0,1] via `z/(2*mag)+shift`; the MuseTalk
    /// UNet emits latents in the full-VAE scaled space, so we undo that scale first.
    fn scale_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let raw = (latents / self.scaling_factor)?;
        ((raw / (2.0 * TAESD_LATENT_MAGNITUDE))? + TAESD_LATENT_SHIFT)?
            .clamp(0f32, 1f32)
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let mut h = self.scale_latents(latents)?.to_dtype(latents.dtype())?;
        h = Convolution.forward_2d(&self.conv_in, &h)?.relu()?;
        let mut bi = 0usize;
        for up in self.upsamplers.iter() {
            for _ in 0..3 {
                h = self.blocks[bi].forward(&h)?;
                bi += 1;
            }
            let (_, _, hh, ww) = h.dims4()?;
            h = h.upsample_nearest2d(hh * 2, ww * 2)?;
            h = Convolution.forward_2d(up, &h)?;
        }
        h = self.blocks[bi].forward(&h)?;
        // TAESD outputs in [0,1]; MuseTalk's denormalize expects a [-1,1]-ish image (img*0.5+0.5),
        // so map [0,1] -> [-1,1] to share the pipeline denormalize.
        let out = Convolution.forward_2d(&self.conv_out, &h)?;
        ((out * 2.0)? - 1.0)?.to_dtype(latents.dtype())
    }
}
