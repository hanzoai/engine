#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Diffusion-forcing autoregressive rollout (Oasis `generate.py`). Each new frame is appended as
//! clamped Gaussian noise and denoised over `ddim_steps` while every context frame is held at a fixed
//! low "stabilization" noise level; a per-frame v-prediction -> x0 -> DDIM step updates only the newest
//! frame. The context window slides at `MAX_FRAMES`. The noise schedule is the sigmoid-beta schedule.

use hanzo_ml::{DType, Device, IndexOp, Result, Tensor};

use crate::diffusion_models::oasis::dit::{Dit, MAX_FRAMES};

const MAX_NOISE_LEVEL: usize = 1000;
const NOISE_ABS_MAX: f64 = 20.0;
const STABILIZATION_LEVEL: usize = 15;

/// Rollout controls. `seed` makes the appended noise reproducible.
#[derive(Debug, Clone)]
pub struct SampleParams {
    pub total_frames: usize,
    pub ddim_steps: usize,
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            total_frames: 32,
            ddim_steps: 10,
            seed: 0,
        }
    }
}

// sigmoid_beta_schedule(1000) -> alphas_cumprod indexed by noise level (f64 to match torch float64).
fn alphas_cumprod() -> Vec<f64> {
    let (start, end) = (-3.0f64, 3.0f64);
    let n = MAX_NOISE_LEVEL;
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let v_start = sig(start);
    let v_end = sig(end);
    // schedule alphas_cumprod over n+1 grid points, normalized so [0] == 1.
    let sched: Vec<f64> = (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (-sig(t * (end - start) + start) + v_end) / (v_end - v_start)
        })
        .collect();
    let sched: Vec<f64> = sched.iter().map(|a| a / sched[0]).collect();
    let betas: Vec<f64> = (0..n)
        .map(|i| (1.0 - sched[i + 1] / sched[i]).clamp(0.0, 0.999))
        .collect();
    // generate.py: alphas_cumprod = cumprod(1 - betas), indexed by integer noise level.
    let mut acp = Vec::with_capacity(n);
    let mut prod = 1.0;
    for b in betas {
        prod *= 1.0 - b;
        acp.push(prod);
    }
    acp
}

fn noise_range(ddim_steps: usize) -> Vec<f64> {
    let n = ddim_steps;
    (0..=n)
        .map(|k| -1.0 + k as f64 * (MAX_NOISE_LEVEL as f64) / n as f64)
        .collect()
}

// per-frame scalar broadcast to [1, win, 1, 1, 1].
fn frame_scalars(vals: &[f64], dev: &Device, dtype: DType) -> Result<Tensor> {
    let win = vals.len();
    let v: Vec<f32> = vals.iter().map(|&x| x as f32).collect();
    Tensor::from_vec(v, (1, win, 1, 1, 1), dev)?.to_dtype(dtype)
}

/// Run the autoregressive rollout. `prompt` is the VAE-encoded, `scaling_factor`-scaled prompt
/// latents (each `[B, 16, 18, 32]`); `actions` is `[B, total_frames, 25]`. Returns `total_frames`
/// per-frame latents (still scaled) ready for VAE decode.
pub fn rollout(
    dit: &Dit,
    prompt: &[Tensor],
    actions: &Tensor,
    params: &SampleParams,
) -> Result<Vec<Tensor>> {
    let dev = dit.device().clone();
    let dtype = dit.dtype();
    if !matches!(dev, Device::Cpu) {
        dev.set_seed(params.seed)?; // candle CPU rng is not seedable; GPU is (stage-6 reproducibility)
    }
    let acp = alphas_cumprod();
    let nr = noise_range(params.ddim_steps);
    let n_prompt = prompt.len();
    let shape = prompt[0].dims().to_vec(); // [B, 16, 18, 32]

    let mut frames: Vec<Tensor> = prompt.to_vec();
    for i in n_prompt..params.total_frames {
        let chunk = Tensor::randn(0f32, 1f32, shape.clone(), &dev)?
            .clamp(-NOISE_ABS_MAX, NOISE_ABS_MAX)?
            .to_dtype(dtype)?;
        frames.push(chunk);
        let start = (i + 1).saturating_sub(MAX_FRAMES);
        let win = i + 1 - start;

        for noise_idx in (1..=params.ddim_steps).rev() {
            let last_level = nr[noise_idx].max(0.0) as usize;
            let next_raw = nr[noise_idx - 1];
            let next_level = if next_raw < 0.0 {
                last_level
            } else {
                next_raw as usize
            };

            // per-frame noise levels for the window (context pinned, newest = current step).
            let mut t_levels = vec![(STABILIZATION_LEVEL - 1) as f64; win];
            *t_levels.last_mut().unwrap() = last_level as f64;
            let acp_t: Vec<f64> = t_levels.iter().map(|&l| acp[l as usize]).collect();

            let x_curr = Tensor::stack(&frames[start..=i], 1)?; // [B, win, 16, 18, 32]
            let t = frame_scalars(&t_levels, &dev, DType::F32)?.reshape((1, win))?;
            let act = actions.i((.., start..=i, ..))?.contiguous()?;

            let v = dit.forward(&x_curr, &t, &act)?.to_dtype(DType::F32)?;
            let x_curr = x_curr.to_dtype(DType::F32)?;

            let acp_t = frame_scalars(&acp_t, &dev, DType::F32)?;
            let sqrt_acp = acp_t.sqrt()?;
            let sqrt_1m = (1.0 - &acp_t)?.sqrt()?;
            let x_start = (sqrt_acp.broadcast_mul(&x_curr)? - sqrt_1m.broadcast_mul(&v)?)?;
            let inv_sqrt = (1.0 / &acp_t)?.sqrt()?;
            let denom = ((1.0 / &acp_t)? - 1.0)?.sqrt()?;
            let x_noise = ((inv_sqrt.broadcast_mul(&x_curr)? - &x_start)?.broadcast_div(&denom))?;

            // alpha_next: context frames -> 1; newest -> acp[next_level] (or 1 on the final step).
            let mut an = vec![1.0f64; win];
            if noise_idx != 1 {
                *an.last_mut().unwrap() = acp[next_level];
            }
            let alpha_next = frame_scalars(&an, &dev, DType::F32)?;
            let x_pred = (alpha_next.sqrt()?.broadcast_mul(&x_start)?
                + (1.0 - &alpha_next)?.sqrt()?.broadcast_mul(&x_noise)?)?;

            let last = x_pred.i((.., win - 1, .., .., ..))?.to_dtype(dtype)?; // [B,16,18,32]
            frames[i] = last;
        }
    }
    Ok(frames)
}
