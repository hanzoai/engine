#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_quant::ShardedVarBuilder;

use super::config::MuseTalkConfig;
use super::unet::UNet2DConditionModel;
use super::vae::AutoencoderKl;

const NORM_MEAN: f64 = 0.5;
const NORM_STD: f64 = 0.5;

pub struct MuseTalk {
    vae: AutoencoderKl,
    unet: UNet2DConditionModel,
    cfg: MuseTalkConfig,
    device: Device,
    dtype: DType,
    mask: Tensor,
}

impl MuseTalk {
    pub fn new(
        cfg: MuseTalkConfig,
        vae_vb: ShardedVarBuilder,
        unet_vb: ShardedVarBuilder,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vae = AutoencoderKl::new(&cfg.vae, vae_vb)?;
        let unet = UNet2DConditionModel::new(&cfg.unet, unet_vb)?;
        let mask = Self::build_mask(cfg.resized_img, device, dtype)?;
        Ok(Self {
            vae,
            unet,
            cfg,
            device: device.clone(),
            dtype,
            mask,
        })
    }

    fn build_mask(size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let top = Tensor::ones((size / 2, size), dtype, device)?;
        let bottom = Tensor::zeros((size - size / 2, size), dtype, device)?;
        Tensor::cat(&[top, bottom], 0)
    }

    fn normalize(&self, img: &Tensor) -> Result<Tensor> {
        ((img - NORM_MEAN)? / NORM_STD)?.to_dtype(self.dtype)
    }

    fn denormalize(&self, img: &Tensor) -> Result<Tensor> {
        ((img.to_dtype(DType::F32)? * NORM_STD)? + NORM_MEAN)?.clamp(0f32, 1f32)
    }

    pub fn latents_for_unet(&self, face: &Tensor) -> Result<Tensor> {
        let face = self.normalize(face)?;
        let masked = face.broadcast_mul(&self.mask.unsqueeze(0)?.unsqueeze(0)?)?;
        let masked_latents = self.vae.encode_mode(&masked)?;
        let ref_latents = self.vae.encode_mode(&face)?;
        Tensor::cat(&[masked_latents, ref_latents], 1)
    }

    pub fn forward(&self, face: &Tensor, audio_feat: &Tensor) -> Result<Tensor> {
        let latent_input = self.latents_for_unet(face)?;
        let b = latent_input.dim(0)?;
        let timestep = Tensor::zeros(b, DType::F32, &self.device)?;
        let pred_latents = self.unet.forward(&latent_input, &timestep, audio_feat)?;
        let image = self.vae.decode(&pred_latents)?;
        self.denormalize(&image)
    }

    pub fn unet_forward(
        &self,
        latent_input: &Tensor,
        timestep: &Tensor,
        audio_feat: &Tensor,
    ) -> Result<Tensor> {
        self.unet.forward(latent_input, timestep, audio_feat)
    }

    pub fn blend(&self, original: &Tensor, generated: &Tensor) -> Result<Tensor> {
        let mask = self.mask.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::F32)?;
        let lower = (1f64 - &mask)?;
        let orig = original.to_dtype(DType::F32)?;
        let gen = generated.to_dtype(DType::F32)?;
        orig.broadcast_mul(&mask)? + gen.broadcast_mul(&lower)?
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn cross_attention_dim(&self) -> usize {
        self.cfg.unet.cross_attention_dim
    }

    pub fn latent_size(&self) -> usize {
        self.cfg.unet.sample_size
    }

    pub fn resized_img(&self) -> usize {
        self.cfg.resized_img
    }
}

pub fn audio_feature_seq_len() -> usize {
    50
}

pub fn reshape_whisper_chunk(chunk: &Tensor) -> Result<Tensor> {
    let dim = chunk.dim(D::Minus1)?;
    chunk.reshape(((), dim))
}
