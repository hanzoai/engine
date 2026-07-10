#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Oasis-500M interactive world model (Etched/Decart, MIT). An action-conditioned frame-autoregressive
//! latent-diffusion game engine: a ViT-VAE ([`vae`]) maps 360x640 Minecraft frames to `[16,18,32]`
//! latents; a spatiotemporal DiT ([`dit`]) denoises the next-frame latent conditioned on past frames
//! and the current keyboard/mouse [`ACTION_KEYS`] one-hot; diffusion-forcing rollout ([`sampling`])
//! generates frames autoregressively. The per-frame denoise step is fixed-shape — the regime CUDA
//! graphs pay off, mirroring decode.

pub mod dit;
pub mod rope;
pub mod sampling;
pub mod vae;

#[cfg(test)]
mod parity;

use std::path::PathBuf;
use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_quant::ShardedVarBuilder;
use image::{DynamicImage, RgbImage};

use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
use dit::Dit;
use sampling::SampleParams;
use vae::VitVae;

/// VPT-derived Minecraft action vocabulary (order is load-bearing: it indexes the 25-dim one-hot the
/// DiT `external_cond` consumes). `cameraX`/`cameraY` are continuous in `[-1, 1]`; the rest are 0/1.
pub const ACTION_KEYS: [&str; 25] = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
];

pub const FRAME_H: usize = 360;
pub const FRAME_W: usize = 640;
const SCALING_FACTOR: f64 = 0.078_431_372_55;

/// Oasis world model: the ViT-VAE and the DiT sharing a device/dtype.
pub struct WorldModel {
    vae: VitVae,
    dit: Dit,
    device: Device,
    dtype: DType,
}

impl WorldModel {
    /// Load the VAE (`vit-l-20.safetensors`) and DiT (`oasis500m.safetensors`) from disk at `dtype`.
    pub fn load(
        vae_path: PathBuf,
        dit_path: PathBuf,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let one = |p: PathBuf| -> Result<ShardedVarBuilder> {
            from_mmaped_safetensors(
                vec![p],
                vec![],
                Some(dtype),
                device,
                vec![None],
                false,
                None,
                |_| true,
                Arc::new(|_| DeviceForLoadTensor::Base),
            )
        };
        Self::new(one(vae_path)?, one(dit_path)?, device.clone())
    }

    pub fn new(
        vae_vb: ShardedVarBuilder,
        dit_vb: ShardedVarBuilder,
        device: Device,
    ) -> Result<Self> {
        let dtype = dit_vb.dtype();
        let vae = VitVae::new(vae_vb, device.clone())?;
        let dit = Dit::new(dit_vb, device.clone())?;
        Ok(Self {
            vae,
            dit,
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

    pub fn vae(&self) -> &VitVae {
        &self.vae
    }

    pub fn dit(&self) -> &Dit {
        &self.dit
    }

    /// Encode RGB frames `[T, 3, 360, 640]` in `[0, 1]` to per-frame scaled latents `[1, 16, 18, 32]`.
    pub fn encode_frames(&self, rgb: &Tensor) -> Result<Vec<Tensor>> {
        let t = rgb.dim(0)?;
        let mut out = Vec::with_capacity(t);
        for i in 0..t {
            let frame = rgb.narrow(0, i, 1)?; // [1,3,360,640]
            let x = ((frame * 2.0)? - 1.0)?;
            let latent = (self.vae.encode(&x)? * SCALING_FACTOR)?;
            out.push(latent);
        }
        Ok(out)
    }

    /// Decode per-frame scaled latents back to RGB `[T, 3, 360, 640]` in `[0, 1]`.
    pub fn decode_frames(&self, latents: &[Tensor]) -> Result<Tensor> {
        let mut frames = Vec::with_capacity(latents.len());
        for z in latents {
            let dec = self.vae.decode(&(z / SCALING_FACTOR)?)?;
            let rgb = (((dec + 1.0)? * 0.5)?).clamp(0f32, 1f32)?;
            frames.push(rgb); // [1,3,360,640]
        }
        Tensor::cat(&frames, 0)
    }

    /// Full generate: encode `prompt` `[n_prompt, 3, 360, 640]`, roll out under `actions`
    /// `[1, total_frames, 25]`, decode to RGB `[total_frames, 3, 360, 640]` in `[0, 1]`.
    pub fn generate(
        &self,
        prompt: &Tensor,
        actions: &Tensor,
        params: &SampleParams,
    ) -> Result<Tensor> {
        let latents = self.encode_frames(prompt)?;
        let out = sampling::rollout(&self.dit, &latents, actions, params)?;
        self.decode_frames(&out)
    }
}

/// A single zeroed action (NOOP): all keys released, camera centered.
pub fn noop_action(device: &Device) -> Result<Tensor> {
    Tensor::zeros((1, 1, ACTION_KEYS.len()), DType::F32, device)
}

/// Convert an RGB tensor `[T, 3, 360, 640]` in `[0, 1]` to per-frame images.
pub fn frames_to_images(rgb: &Tensor) -> Result<Vec<DynamicImage>> {
    let rgb = rgb.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let t = rgb.dim(0)?;
    let mut imgs = Vec::with_capacity(t);
    for f in 0..t {
        let frame: Vec<f32> = rgb.narrow(0, f, 1)?.flatten_all()?.to_vec1()?;
        let mut img = RgbImage::new(FRAME_W as u32, FRAME_H as u32);
        let plane = FRAME_H * FRAME_W;
        for y in 0..FRAME_H {
            for x in 0..FRAME_W {
                let idx = y * FRAME_W + x;
                let px = |c: usize| (frame[c * plane + idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                img.put_pixel(x as u32, y as u32, image::Rgb([px(0), px(1), px(2)]));
            }
        }
        imgs.push(DynamicImage::ImageRgb8(img));
    }
    Ok(imgs)
}

/// Load an RGB image, resize to 360x640 (Lanczos3), return a `[1, 3, 360, 640]` tensor in `[0, 1]`.
pub fn image_to_tensor(img: &DynamicImage, device: &Device) -> Result<Tensor> {
    let img = img
        .resize_exact(
            FRAME_W as u32,
            FRAME_H as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();
    let mut data = vec![0f32; 3 * FRAME_H * FRAME_W];
    let plane = FRAME_H * FRAME_W;
    for (x, y, px) in img.enumerate_pixels() {
        let idx = y as usize * FRAME_W + x as usize;
        for c in 0..3 {
            data[c * plane + idx] = px[c] as f32 / 255.0;
        }
    }
    Tensor::from_vec(data, (1, 3, FRAME_H, FRAME_W), device)
}
