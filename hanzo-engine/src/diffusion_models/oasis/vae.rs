#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Oasis ViT-VAE (`vit-l-20-shallow-encoder`, `AutoencoderKL`). A pure vision transformer autoencoder:
//! 20x20 conv patchify of a 360x640 frame -> 6 encoder blocks -> `quant_conv` to a 16-dim diagonal
//! Gaussian (we take the mean) -> latent `[B, 576, 16]` reshaped to `[B, 16, 18, 32]`. Decode mirrors:
//! `post_quant_conv` -> 12 decoder blocks -> `predictor` -> unpatchify to RGB. Attention is 2D
//! axial-RoPE (pixel freqs, rot covers head_dim/2); MLP activation is exact GELU (timm `nn.GELU`).

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear};
use hanzo_quant::ShardedVarBuilder;

use crate::attention::AttentionMask;
use crate::diffusion_models::oasis::rope::RotaryTable;
use crate::diffusion_models::wan::longcat::blocks::sdpa;
use crate::layers;

const IN_H: usize = 360;
const IN_W: usize = 640;
const PATCH: usize = 20;
const DIM: usize = 1024;
const HEADS: usize = 16;
const HEAD_DIM: usize = DIM / HEADS;
const FFN: usize = DIM * 4;
const LATENT: usize = 16;
const PATCH_DIM: usize = 3 * PATCH * PATCH;
const ENC_DEPTH: usize = 6;
const DEC_DEPTH: usize = 12;
const EPS: f64 = 1e-6;

const SEQ_H: usize = IN_H / PATCH; // 18
const SEQ_W: usize = IN_W / PATCH; // 32

fn layer_norm(vb: ShardedVarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        vb.get(DIM, "weight")?,
        vb.get(DIM, "bias")?,
        EPS,
    ))
}

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: layers::linear(DIM, FFN, vb.pp("fc1"))?,
            fc2: layers::linear(FFN, DIM, vb.pp("fc2"))?,
        })
    }

    // exact GELU (timm nn.GELU default), unlike the DiT's tanh-approx blocks.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.fc1)?.gelu_erf()?.apply(&self.fc2)
    }
}

struct VaeAttention {
    qkv: Linear,
    proj: Linear,
}

impl VaeAttention {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: layers::linear(DIM, DIM * 3, vb.pp("qkv"))?,
            proj: layers::linear(DIM, DIM, vb.pp("proj"))?,
        })
    }

    // x [B, N, DIM]; rope covers the 18x32 grid (partial rot over head_dim/2).
    fn forward(&self, x: &Tensor, rope: &RotaryTable) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let qkv = x.apply(&self.qkv)?;
        let split = |i: usize| -> Result<Tensor> {
            qkv.narrow(D::Minus1, i * DIM, DIM)?
                .reshape((b, n, HEADS, HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = rope.apply(&split(0)?)?;
        let k = rope.apply(&split(1)?)?;
        let v = split(2)?;
        let out = sdpa(&q, &k, &v, &AttentionMask::None)?;
        out.transpose(1, 2)?.reshape((b, n, DIM))?.apply(&self.proj)
    }
}

struct Block {
    norm1: LayerNorm,
    attn: VaeAttention,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(vb.pp("norm1"))?,
            attn: VaeAttention::new(vb.pp("attn"))?,
            norm2: layer_norm(vb.pp("norm2"))?,
            mlp: Mlp::new(vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, rope: &RotaryTable) -> Result<Tensor> {
        let x = (x + self.attn.forward(&x.apply(&self.norm1)?, rope)?)?;
        &x + self.mlp.forward(&x.apply(&self.norm2)?)?
    }
}

/// The Oasis ViT-VAE. `encode` returns the posterior mean latent `[B, 16, 18, 32]`; `decode` maps a
/// latent back to an RGB frame `[B, 3, 360, 640]`. Callers apply the `scaling_factor` themselves.
pub struct VitVae {
    patch_proj: Conv2d,
    encoder: Vec<Block>,
    enc_norm: LayerNorm,
    quant_conv: Linear,
    post_quant_conv: Linear,
    decoder: Vec<Block>,
    dec_norm: LayerNorm,
    predictor: Linear,
    rope: RotaryTable,
    device: Device,
    dtype: DType,
}

impl VitVae {
    pub fn new(vb: ShardedVarBuilder, device: Device) -> Result<Self> {
        let dtype = vb.dtype();
        let pw = vb.get((DIM, 3, PATCH, PATCH), "patch_embed.proj.weight")?;
        let pb = vb.get(DIM, "patch_embed.proj.bias")?;
        let patch_proj = Conv2d::new(
            pw,
            Some(pb),
            Conv2dConfig {
                padding: 0,
                stride: PATCH,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );
        let enc_vb = vb.pp("encoder");
        let mut encoder = Vec::with_capacity(ENC_DEPTH);
        for i in 0..ENC_DEPTH {
            encoder.push(Block::new(enc_vb.pp(i))?);
        }
        let dec_vb = vb.pp("decoder");
        let mut decoder = Vec::with_capacity(DEC_DEPTH);
        for i in 0..DEC_DEPTH {
            decoder.push(Block::new(dec_vb.pp(i))?);
        }
        Ok(Self {
            patch_proj,
            encoder,
            enc_norm: layer_norm(vb.pp("enc_norm"))?,
            quant_conv: layers::linear(DIM, 2 * LATENT, vb.pp("quant_conv"))?,
            post_quant_conv: layers::linear(LATENT, DIM, vb.pp("post_quant_conv"))?,
            decoder,
            dec_norm: layer_norm(vb.pp("dec_norm"))?,
            predictor: layers::linear(DIM, PATCH_DIM, vb.pp("predictor"))?,
            rope: RotaryTable::vae_spatial(SEQ_H, SEQ_W, HEAD_DIM, &device)?,
            device,
            dtype,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Encode RGB frames `[B, 3, 360, 640]` in `[-1, 1]` to the posterior-mean latent `[B, 16, 18, 32]`.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(self.dtype)?.apply(&self.patch_proj)?; // [B, DIM, 18, 32]
        let b = x.dim(0)?;
        let mut x = x.flatten_from(2)?.transpose(1, 2)?.contiguous()?; // [B, 576, DIM]
        for blk in &self.encoder {
            x = blk.forward(&x, &self.rope)?;
        }
        let x = x.apply(&self.enc_norm)?.apply(&self.quant_conv)?; // [B, 576, 2*LATENT]
        let mean = x.narrow(D::Minus1, 0, LATENT)?; // posterior mode
                                                    // [B, 576, 16] -> [B, 16, 18, 32]
        mean.transpose(1, 2)?
            .reshape((b, LATENT, SEQ_H, SEQ_W))?
            .contiguous()
    }

    /// Decode a latent `[B, 16, 18, 32]` to an RGB frame `[B, 3, 360, 640]` (model range, pre-`(x+1)/2`).
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let b = z.dim(0)?;
        let z = z
            .to_dtype(self.dtype)?
            .reshape((b, LATENT, SEQ_H * SEQ_W))?
            .transpose(1, 2)?; // [B, 576, 16]
        let mut z = z.contiguous()?.apply(&self.post_quant_conv)?;
        for blk in &self.decoder {
            z = blk.forward(&z, &self.rope)?;
        }
        let z = z.apply(&self.dec_norm)?.apply(&self.predictor)?; // [B, 576, 1200]
        self.unpatchify(&z, b)
    }

    // [B, 576, 1200] -> [B, 3, 360, 640] (vae.py unpatchify).
    fn unpatchify(&self, x: &Tensor, b: usize) -> Result<Tensor> {
        x.reshape((b, SEQ_H, SEQ_W, PATCH_DIM))?
            .permute([0, 3, 1, 2])? // [b, 1200, 18, 32]
            .reshape((b, 3, PATCH, PATCH, SEQ_H, SEQ_W))?
            .permute([0, 1, 4, 2, 5, 3])? // [b, 3, 18, 20, 32, 20]
            .contiguous()?
            .reshape((b, 3, IN_H, IN_W))
    }
}
