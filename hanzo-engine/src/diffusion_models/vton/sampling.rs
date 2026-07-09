//! Rectified-flow (flow-matching) Euler sampler with classifier-free guidance, matching the
//! reference `TryOnPipeline._sample` and `get_rf_schedule`.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{Result, Tensor};

use super::model::TryOnModel;

pub const DEFAULT_TIMESTEPS: usize = 30;
pub const DEFAULT_TIME_SHIFT_MU: f64 = 1.5;
pub const DEFAULT_GUIDANCE_SCALE: f64 = 1.5;
pub const DEFAULT_SKIP_CFG_LAST_N: usize = 1;

fn time_shift(mu: f64, sigma: f64, t: f64) -> f64 {
    mu.exp() / (mu.exp() + (1.0 / t - 1.0).powf(sigma))
}

/// Timestep schedule from t=0 -> t=1 of length `num_steps + 1` (denoising direction).
pub fn get_rf_schedule(num_steps: usize, mu: f64) -> Vec<f64> {
    let mu = -mu;
    let n = num_steps + 1;
    let mut ts: Vec<f64> = (0..n)
        .map(|i| {
            let lin = 1.0 - i as f64 / (n as f64 - 1.0); // linspace(1, 0, n)
            time_shift(mu, 1.0, lin)
        })
        .collect();
    ts.reverse();
    ts
}

/// The conditioning inputs to a single denoising step. The unconditional (null) variant zeros every
/// image conditioning and uses the null category id 0.
pub struct Conditioning {
    pub ca_images: Tensor,
    pub garment_images: Tensor,
    pub person_poses: Tensor,
    pub garment_poses: Tensor,
    pub categories: Tensor,
}

impl Conditioning {
    fn null_like(&self) -> Result<Self> {
        Ok(Self {
            ca_images: self.ca_images.zeros_like()?,
            garment_images: self.garment_images.zeros_like()?,
            person_poses: self.person_poses.zeros_like()?,
            garment_poses: self.garment_poses.zeros_like()?,
            categories: self.categories.zeros_like()?,
        })
    }
}

fn predict(model: &TryOnModel, x: &Tensor, t: &Tensor, c: &Conditioning) -> Result<Tensor> {
    model.forward(
        x,
        t,
        &c.ca_images,
        &c.garment_images,
        &c.person_poses,
        &c.garment_poses,
        &c.categories,
    )
}

pub struct SampleParams {
    pub num_timesteps: usize,
    pub time_shift_mu: f64,
    pub guidance_scale: f64,
    pub skip_cfg_last_n_steps: usize,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            num_timesteps: DEFAULT_TIMESTEPS,
            time_shift_mu: DEFAULT_TIME_SHIFT_MU,
            guidance_scale: DEFAULT_GUIDANCE_SCALE,
            skip_cfg_last_n_steps: DEFAULT_SKIP_CFG_LAST_N,
        }
    }
}

/// Euler flow-matching integration from noise (`init`) to image, with CFG. Returns the raw
/// integrated tensor, shape (b, 3, H, W); clamping to [-1, 1] is a display concern done at image
/// conversion (`tensor_to_rgb`), matching the reference `tensor_to_pil` step.
pub fn denoise(
    model: &TryOnModel,
    cond: &Conditioning,
    init: &Tensor,
    p: &SampleParams,
) -> Result<Tensor> {
    let uncond = cond.null_like()?;
    let timesteps = get_rf_schedule(p.num_timesteps, p.time_shift_mu);
    let (bs, dtype, dev) = (init.dim(0)?, init.dtype(), init.device().clone());

    let mut images = init.clone();
    for step in 0..p.num_timesteps {
        let t_curr = timesteps[step];
        let dt = timesteps[step + 1] - t_curr;
        let t_vec = Tensor::full(t_curr as f32, bs, &dev)?.to_dtype(dtype)?;

        let v_c = predict(model, &images, &t_vec, cond)?;
        let v_guided = if step >= p.num_timesteps - p.skip_cfg_last_n_steps {
            v_c
        } else {
            let v_u = predict(model, &images, &t_vec, &uncond)?;
            (&v_u + ((&v_c - &v_u)? * p.guidance_scale)?)?
        };
        images = (images + (v_guided * dt)?)?;
    }
    Ok(images)
}
