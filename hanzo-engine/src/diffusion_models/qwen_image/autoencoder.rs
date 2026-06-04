#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{Device, Result, Tensor};
use hanzo_quant::ShardedVarBuilder;
use serde::Deserialize;

// AutoencoderKLQwenImage: a WAN-style 3D causal-conv VAE.
// The full encoder/decoder (CausalConv3d + frame caching) is not yet ported; this carries the
// config and the latent (de)normalization contract so the pipeline can wire and compile.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Config {
    pub base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub temperal_downsample: Vec<bool>,
    #[serde(default)]
    pub dropout: f64,
    #[serde(default)]
    pub attn_scales: Vec<f64>,
    pub latents_mean: Vec<f64>,
    pub latents_std: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct AutoEncoder {
    latents_mean: Tensor,
    latents_std: Tensor,
    z_dim: usize,
}

impl AutoEncoder {
    pub fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let dev = vb.device();
        let z = cfg.z_dim;
        let latents_mean = Tensor::from_vec(
            cfg.latents_mean.iter().map(|v| *v as f32).collect::<Vec<_>>(),
            (1, z, 1, 1, 1),
            dev,
        )?;
        let latents_std = Tensor::from_vec(
            cfg.latents_std.iter().map(|v| *v as f32).collect::<Vec<_>>(),
            (1, z, 1, 1, 1),
            dev,
        )?;
        let _ = vb;
        Ok(Self {
            latents_mean,
            latents_std,
            z_dim: z,
        })
    }

    pub fn z_dim(&self) -> usize {
        self.z_dim
    }

    // diffusers: z = z / latents_std + latents_mean before feeding the decoder.
    pub fn denormalize_latents(&self, z: &Tensor) -> Result<Tensor> {
        z.broadcast_mul(&self.latents_std)?
            .broadcast_add(&self.latents_mean)
    }

    // TODO(qwen-image): port QwenImageDecoder3d (CausalConv3d mid/up blocks + RMS norm) to map
    // (b, z_dim, f, h, w) latents to RGB frames. Until then decode is unimplemented.
    pub fn decode(&self, _z: &Tensor, _device: &Device) -> Result<Tensor> {
        hanzo_ml::bail!("QwenImage VAE decode is not yet implemented (3D causal-conv decoder pending)")
    }
}
