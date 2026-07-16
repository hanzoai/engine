//! Rectified-flow Euler sampler with classifier-free guidance interval (TRELLIS
//! `FlowEulerGuidanceIntervalSampler`), shared by both flow stages.
//!
//! The schedule is `linspace(1, 0, steps+1)` reparameterized by `rescale_t`, and each step is a plain
//! Euler update `x <- x - (t - t_prev) * v`. Inside `cfg_interval` the velocity is CFG-guided
//! `(1 + s) * v_cond - s * v_uncond`; outside it, cond-only. `v` is supplied as a closure so the
//! same sampler drives the sparse-structure and SLAT flows.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use hanzo_ml::{Result, Tensor};

#[derive(Debug, Clone)]
pub struct FlowEulerParams {
    pub steps: usize,
    pub cfg_strength: f64,
    pub cfg_interval: (f64, f64),
    pub rescale_t: f64,
}

impl Default for FlowEulerParams {
    /// TRELLIS-image-large pipeline.json defaults (both stages).
    fn default() -> Self {
        Self {
            steps: 25,
            cfg_strength: 5.0,
            cfg_interval: (0.5, 1.0),
            rescale_t: 3.0,
        }
    }
}

/// `linspace(1, 0, steps+1)` reparameterized by `rescale_t`. Endpoints stay 1 and 0.
pub fn t_schedule(steps: usize, rescale_t: f64) -> Vec<f64> {
    (0..=steps)
        .map(|i| {
            let t = 1.0 - i as f64 / steps as f64;
            rescale_t * t / (1.0 + (rescale_t - 1.0) * t)
        })
        .collect()
}

/// Sample from noise. `velocity(x_t, t, cond)` returns the model's predicted velocity; `cond` and
/// `neg_cond` are the positive/negative conditioning.
pub fn sample<F>(
    noise: &Tensor,
    cond: &Tensor,
    neg_cond: &Tensor,
    params: &FlowEulerParams,
    mut velocity: F,
) -> Result<Tensor>
where
    F: FnMut(&Tensor, f64, &Tensor) -> Result<Tensor>,
{
    let ts = t_schedule(params.steps, params.rescale_t);
    let (lo, hi) = params.cfg_interval;
    let mut x = noise.clone();
    for w in ts.windows(2) {
        let (t, t_prev) = (w[0], w[1]);
        let v = if lo <= t && t <= hi {
            let pos = velocity(&x, t, cond)?;
            let neg = velocity(&x, t, neg_cond)?;
            ((pos * (1.0 + params.cfg_strength))? - (neg * params.cfg_strength)?)?
        } else {
            velocity(&x, t, cond)?
        };
        x = (&x - (v * (t - t_prev))?)?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_endpoints_and_length() {
        let ts = t_schedule(25, 3.0);
        assert_eq!(ts.len(), 26);
        assert!((ts[0] - 1.0).abs() < 1e-12);
        assert!(ts[25].abs() < 1e-12);
        // strictly decreasing
        assert!(ts.windows(2).all(|w| w[0] > w[1]));
        // rescale_t=1 is the identity linspace
        let id = t_schedule(4, 1.0);
        for (i, v) in id.iter().enumerate() {
            assert!((v - (1.0 - i as f64 / 4.0)).abs() < 1e-12);
        }
    }
}
