#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{Device, Result, Tensor};

use super::model::QwenImageTransformer;

// VAE spatial compression for Qwen-Image (8x), then 2x patchify in the transformer.
pub const VAE_SCALE_FACTOR: usize = 8;
pub const PATCH_SIZE: usize = 2;

// Sample initial latent noise in packed form: (b, frame*ph*pw, z_dim * patch^2).
pub fn get_noise(
    num_samples: usize,
    height: usize,
    width: usize,
    z_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let (frame, ph, pw) = packed_shape(height, width);
    let c = z_dim * PATCH_SIZE * PATCH_SIZE;
    Tensor::randn(0f32, 1., (num_samples, frame * ph * pw, c), device)
}

// Packed latent grid (frame, packed_h, packed_w) for the given pixel dims.
pub fn packed_shape(height: usize, width: usize) -> (usize, usize, usize) {
    let lat_h = height / VAE_SCALE_FACTOR;
    let lat_w = width / VAE_SCALE_FACTOR;
    (1, lat_h / PATCH_SIZE, lat_w / PATCH_SIZE)
}

// Unpack (b, frame*ph*pw, z_dim*patch^2) back to a latent (b, z_dim, frame, h, w).
// Images use frame=1, so we fold frame into batch to stay within candle's 6-dim reshape limit.
pub fn unpack(xs: &Tensor, height: usize, width: usize, z_dim: usize) -> Result<Tensor> {
    let (b, _seq, _c) = xs.dims3()?;
    let (frame, ph, pw) = packed_shape(height, width);
    xs.reshape((b * frame, ph, pw, z_dim, PATCH_SIZE, PATCH_SIZE))? // (b*f, ph, pw, c, p, p)
        .permute((0, 3, 1, 4, 2, 5))? // (b*f, c, ph, p, pw, p)
        .reshape((b, frame, z_dim, ph * PATCH_SIZE, pw * PATCH_SIZE))?
        .permute((0, 2, 1, 3, 4))? // (b, c, f, h, w)
        .contiguous()
}

fn time_shift(mu: f64, sigma: f64, t: f64) -> f64 {
    let e = mu.exp();
    e / (e + (1. / t - 1.).powf(sigma))
}

// FlowMatchEulerDiscrete schedule with resolution-dependent shift, matching FLUX/Qwen-Image.
pub fn get_schedule(num_steps: usize, image_seq_len: usize, base_shift: f64, max_shift: f64) -> Vec<f64> {
    let timesteps: Vec<f64> = (0..=num_steps)
        .map(|v| v as f64 / num_steps as f64)
        .rev()
        .collect();
    let (x1, x2) = (256., 4096.);
    let m = (max_shift - base_shift) / (x2 - x1);
    let b = base_shift - m * x1;
    let mu = m * image_seq_len as f64 + b;
    timesteps.into_iter().map(|v| time_shift(mu, 1., v)).collect()
}

#[allow(clippy::too_many_arguments)]
pub fn denoise(
    model: &QwenImageTransformer,
    img: &Tensor,
    txt: &Tensor,
    img_shape: (usize, usize, usize),
    timesteps: &[f64],
    guidance: Option<f64>,
) -> Result<Tensor> {
    let b_sz = img.dim(0)?;
    let dev = img.device();
    let guidance = match guidance {
        Some(g) => Some(Tensor::full(g as f32, b_sz, dev)?),
        None => None,
    };
    let mut img = img.clone();
    for window in timesteps.windows(2) {
        let [t_curr, t_prev] = window else { continue };
        let t_vec = Tensor::full(*t_curr as f32, b_sz, dev)?;
        let pred = model.forward(&img, txt, &t_vec, img_shape, guidance.as_ref())?;
        img = (img + (pred * (t_prev - t_curr))?)?;
    }
    Ok(img)
}
