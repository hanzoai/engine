use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_quant::ShardedVarBuilder;

use crate::{
    diffusion_models::{qwen_image, DiffusionGenerationParams},
    pipeline::DiffusionModel,
};

use super::{autoencoder::AutoEncoder, model::QwenImageTransformer};

#[derive(Clone, Copy, Debug)]
pub struct QwenImageStepperConfig {
    pub num_steps: usize,
    pub base_shift: f64,
    pub max_shift: f64,
    pub guidance_scale: Option<f64>,
}

impl Default for QwenImageStepperConfig {
    fn default() -> Self {
        Self {
            num_steps: 50,
            base_shift: 0.5,
            max_shift: 1.15,
            guidance_scale: None,
        }
    }
}

pub struct QwenImageStepper {
    cfg: QwenImageStepperConfig,
    transformer: QwenImageTransformer,
    vae: AutoEncoder,
    device: Device,
    dtype: DType,
}

impl QwenImageStepper {
    pub fn new(
        cfg: QwenImageStepperConfig,
        (transformer_vb, transformer_cfg): (ShardedVarBuilder, &qwen_image::model::Config),
        (vae_vb, vae_cfg): (ShardedVarBuilder, &qwen_image::autoencoder::Config),
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let transformer =
            QwenImageTransformer::new(transformer_cfg, transformer_vb, device.clone())?;
        let vae = AutoEncoder::new(vae_cfg, vae_vb)?;
        Ok(Self {
            cfg,
            transformer,
            vae,
            device: device.clone(),
            dtype,
        })
    }

    // TODO(qwen-image): run the Qwen2.5-VL text encoder over the prompt and return its last
    // hidden states (b, txt_seq, joint_attention_dim). The transformer + scheduler below already
    // consume this tensor.
    fn encode_prompts(&self, _prompts: &[String]) -> Result<Tensor> {
        hanzo_ml::bail!(
            "QwenImage text encoder (Qwen2.5-VL) is not yet wired; transformer + scheduler are ready"
        )
    }
}

impl DiffusionModel for QwenImageStepper {
    fn forward(
        &mut self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
    ) -> Result<Tensor> {
        let txt = self.encode_prompts(&prompts)?.to_dtype(self.dtype)?;

        let z_dim = self.vae.z_dim();
        let img = qwen_image::sampling::get_noise(
            txt.dim(0)?,
            params.height,
            params.width,
            z_dim,
            &self.device,
        )?
        .to_dtype(self.dtype)?;

        let img_shape = qwen_image::sampling::packed_shape(params.height, params.width);
        let image_seq_len = img.dim(1)?;
        let timesteps = qwen_image::sampling::get_schedule(
            self.cfg.num_steps,
            image_seq_len,
            self.cfg.base_shift,
            self.cfg.max_shift,
        );

        let latents = qwen_image::sampling::denoise(
            &self.transformer,
            &img,
            &txt,
            img_shape,
            &timesteps,
            self.cfg.guidance_scale,
        )?;

        let latents = qwen_image::sampling::unpack(&latents, params.height, params.width, z_dim)?;
        let latents = self.vae.denormalize_latents(&latents)?;
        let img = self.vae.decode(&latents, &self.device)?;

        ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn max_seq_len(&self) -> usize {
        usize::MAX
    }
}
