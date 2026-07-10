#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_quant::ShardedVarBuilder;

use super::config::MuseTalkConfig;
use super::taesd::Taesd;
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
    taesd: Option<Taesd>,
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
            taesd: None,
        })
    }

    /// Attach the TAESD encode+decode pair as the fast VAE path. The tiny distilled encoder
    /// replaces the profiled realtime wall (the full VAE encoder's high-res conv-GEMMs); the full
    /// VAE stays loaded but unused. Both stacks share the SD-VAE latent space so the UNet is
    /// untouched. Enabled at load time behind `DUB_TAESD`.
    pub fn with_taesd(
        mut self,
        encoder_vb: ShardedVarBuilder,
        decoder_vb: ShardedVarBuilder,
    ) -> Result<Self> {
        self.taesd = Some(Taesd::new(&self.cfg.vae, encoder_vb, decoder_vb)?);
        Ok(self)
    }

    pub fn has_taesd(&self) -> bool {
        self.taesd.is_some()
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
        // Mask the lower half to black in [0,1] (MuseTalk's preprocess_img masks the raw image).
        // The full VAE wants [-1,1], so mask BEFORE normalize -> the mouth encodes as -1, not gray
        // 0 (which the VAE would inpaint as a flat patch). TAESD consumes [0,1] as-is.
        let masked = face.broadcast_mul(&self.mask.unsqueeze(0)?.unsqueeze(0)?)?;
        let (masked_latents, ref_latents) = match &self.taesd {
            Some(t) => (t.encoder.encode(&masked)?, t.encoder.encode(face)?),
            None => (
                self.vae.encode_mode(&self.normalize(&masked)?)?,
                self.vae.encode_mode(&self.normalize(face)?)?,
            ),
        };
        Tensor::cat(&[masked_latents, ref_latents], 1)
    }

    /// Scaled UNet latents -> `[0,1]` RGB face, via TAESD when attached (it emits [0,1] directly)
    /// or the full VAE (which emits [-1,1], so denormalize).
    pub fn decode_latents(&self, pred: &Tensor) -> Result<Tensor> {
        match &self.taesd {
            Some(t) => t.decoder.decode(pred),
            None => self.denormalize(&self.vae.decode(pred)?),
        }
    }

    pub fn forward(&self, face: &Tensor, audio_feat: &Tensor) -> Result<Tensor> {
        let latent_input = self.latents_for_unet(face)?;
        let b = latent_input.dim(0)?;
        let timestep = Tensor::zeros(b, DType::F32, &self.device)?;
        let pred_latents = self.unet.forward(&latent_input, &timestep, audio_feat)?;
        self.decode_latents(&pred_latents)
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

    pub fn dtype(&self) -> DType {
        self.dtype
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
