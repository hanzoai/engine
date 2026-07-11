#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
//! Flow-matching Euler scheduler + APG guidance for ACE-Step.
//! Port of schedulers/scheduling_flow_match_euler_discrete.py + apg_guidance.py (Apache-2.0).

use hanzo_ml::{Result, Tensor, D};

const NUM_TRAIN: f64 = 1000.0;
const SHIFT: f64 = 3.0;
const SIGMA_MIN_RAW: f64 = 1.0 / NUM_TRAIN;
const APG_MOMENTUM: f64 = -0.75;
const APG_NORM_THRESHOLD: f64 = 2.5;

fn logistic(x: f64, k: f64) -> f64 {
    let (l, u, x0) = (0.9, 1.1, 0.0);
    l + (u - l) / (1.0 + (-k * (x - x0)).exp())
}

#[derive(Debug, Clone)]
pub struct FlowMatchScheduler {
    pub sigmas: Vec<f64>,    // len num_steps + 1, descending to 0
    pub timesteps: Vec<f32>, // len num_steps, on the 0..1000 scale
    omega: f64,
}

impl FlowMatchScheduler {
    pub fn new(num_steps: usize, omega_scale: f64) -> Self {
        let smax = 1.0;
        let mut sigmas: Vec<f64> = (0..num_steps)
            .map(|i| {
                let f = i as f64 / (num_steps - 1) as f64;
                let s = smax + (SIGMA_MIN_RAW - smax) * f; // linspace(1, 1/1000)
                SHIFT * s / (1.0 + (SHIFT - 1.0) * s)
            })
            .collect();
        let timesteps: Vec<f32> = sigmas.iter().map(|s| (s * NUM_TRAIN) as f32).collect();
        sigmas.push(0.0);
        Self {
            sigmas,
            timesteps,
            omega: logistic(omega_scale, 0.1),
        }
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    // Euler with omega mean-preserving deviation rescale: x + ((s_next - s)*v - m)*omega + m.
    // Mean stays on device (no scalar readback) so the sampler loop never blocks on a host sync.
    pub fn step(&self, model_output: &Tensor, sample: &Tensor, i: usize) -> Result<Tensor> {
        let dt = self.sigmas[i + 1] - self.sigmas[i];
        let dx = (model_output * dt)?;
        let m = dx.mean_all()?.reshape((1, 1, 1, 1))?;
        let dx = ((dx.broadcast_sub(&m)? * self.omega)?.broadcast_add(&m))?;
        sample + dx
    }
}

// APG guidance (apg_forward, eta=0): momentum + norm-clip + orthogonal projection onto pred_cond.
pub struct MomentumBuffer {
    momentum: f64,
    running: Option<Tensor>,
}

impl MomentumBuffer {
    pub fn new() -> Self {
        Self {
            momentum: APG_MOMENTUM,
            running: None,
        }
    }

    fn update(&mut self, diff: &Tensor) -> Result<Tensor> {
        let avg = match &self.running {
            None => diff.clone(),
            Some(r) => (diff + (r * self.momentum)?)?,
        };
        self.running = Some(avg.clone());
        Ok(avg)
    }
}

impl Default for MomentumBuffer {
    fn default() -> Self {
        Self::new()
    }
}

fn frame_norm(x: &Tensor) -> Result<Tensor> {
    x.sqr()?
        .sum_keepdim(D::Minus1)?
        .sum_keepdim(D::Minus2)?
        .sqrt()
}

// Project v0 onto normalized v1 over the last two dims (per-(H,W) frame).
fn project(v0: &Tensor, v1: &Tensor) -> Result<Tensor> {
    let v1u = v1.broadcast_div(&frame_norm(v1)?)?;
    let coeff = (v0 * &v1u)?
        .sum_keepdim(D::Minus1)?
        .sum_keepdim(D::Minus2)?;
    let parallel = v1u.broadcast_mul(&coeff)?;
    v0 - parallel
}

pub fn apg_forward(
    pred_cond: &Tensor,
    pred_uncond: &Tensor,
    guidance_scale: f64,
    mb: &mut MomentumBuffer,
) -> Result<Tensor> {
    let diff = (pred_cond - pred_uncond)?;
    let diff = mb.update(&diff)?;
    let ratio = (frame_norm(&diff)?.recip()? * APG_NORM_THRESHOLD)?;
    let scale = ratio.minimum(&Tensor::ones_like(&ratio)?)?;
    let diff = diff.broadcast_mul(&scale)?;
    let orthogonal = project(&diff, pred_cond)?;
    pred_cond + (orthogonal * (guidance_scale - 1.0))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;

    #[test]
    fn scheduler_schedule_is_monotone() {
        let s = FlowMatchScheduler::new(60, 10.0);
        assert_eq!(s.sigmas.len(), 61);
        assert_eq!(s.timesteps.len(), 60);
        assert!((s.sigmas[0] - (3.0 / (1.0 + 2.0))).abs() < 1e-6); // sigma(1.0)=3/3=1.0
        assert!(*s.sigmas.last().unwrap() == 0.0);
        for i in 0..60 {
            assert!(s.sigmas[i] > s.sigmas[i + 1]); // strictly descending
        }
        // omega logistic(10, k=0.1) in (0.9, 1.1)
        assert!(s.omega > 1.0 && s.omega < 1.1);
    }

    #[test]
    fn euler_step_moves_toward_target() {
        let dev = Device::Cpu;
        let s = FlowMatchScheduler::new(4, 10.0);
        let v = Tensor::ones((1, 2, 2, 2), hanzo_ml::DType::F32, &dev).unwrap();
        let x = Tensor::zeros((1, 2, 2, 2), hanzo_ml::DType::F32, &dev).unwrap();
        let out = s.step(&v, &x, 0).unwrap();
        // dt<0, uniform v -> uniform update = dt (omega cancels on zero-deviation), so out ~= dt.
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let dt = (s.sigmas[1] - s.sigmas[0]) as f32;
        assert!((got - dt).abs() < 1e-5, "got {got} dt {dt}");
    }
}
