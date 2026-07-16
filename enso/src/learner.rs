//! Piece 6/6: learning, in two halves on one seam.
//!
//! Offline: [`fit_base`] is a ridge regression of eval quality on the bilinear
//! feature `phi = vec(x p^T)`, solved in closed form -- the base `W`.
//!
//! Online: a per-user contextual bandit (LinUCB) keeps `A^-1`, `b`, and `theta`
//! and updates them from observed outcomes. Each user's `theta` starts at the
//! base `W` (a Gaussian prior centered there) and drifts toward that user's
//! taste; `theta - W = dW_u` is the per-user delta serving reads back. The
//! serving hot path is greedy on `theta` (cheap); the UCB exploration bonus is
//! computed only while learning, via [`Bandit::ucb`].
//!
//! Honest scope: this LinUCB core is solid and converges (the reward is linear
//! in `phi`, so it is realizable). The research-frontier variants -- a genuinely
//! low-rank `dW_u` LoRA over a neural encoder, self-adaptive expert vectors --
//! attach at exactly this seam (replace `theta`'s update / the `Policy` score)
//! and are deliberately not faked here.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::linalg::{self, bilinear_feature, dot, matvec, quad_form, sherman_morrison};
use crate::policy::{Policy, DK};

/// Closed-form ridge fit of the base policy. `samples` yields `(x, p, quality)`.
pub fn fit_base<'a>(
    samples: impl IntoIterator<Item = (&'a [f64], &'a [f64], f64)>,
    gamma: f64,
) -> Policy {
    let mut a = vec![0.0f64; DK * DK];
    for i in 0..DK {
        a[i * DK + i] = gamma;
    }
    let mut b = vec![0.0f64; DK];
    let mut phi = vec![0.0f64; DK];
    for (x, p, y) in samples {
        bilinear_feature(x, p, &mut phi);
        for i in 0..DK {
            let pi = phi[i];
            if pi == 0.0 {
                continue;
            }
            let row = i * DK;
            for j in 0..DK {
                a[row + j] += pi * phi[j];
            }
            b[i] += y * pi;
        }
    }
    linalg::cholesky(&mut a, DK);
    Policy::from_weights(linalg::cholesky_solve(&a, DK, &b))
}

/// Per-user LinUCB state over the bilinear feature.
#[derive(Clone)]
pub struct Bandit {
    ainv: Vec<f64>,
    b: Vec<f64>,
    theta: Vec<f64>,
    alpha: f64,
    phi: Vec<f64>,
    scratch: Vec<f64>,
    pub n: u32,
}

impl Bandit {
    /// Prior centered at `w_base`: `A = gamma*I`, `b = gamma*w_base`, so
    /// `theta = A^-1 b = w_base` before any observation.
    pub fn with_prior(w_base: &[f64], gamma: f64, alpha: f64) -> Self {
        let mut ainv = vec![0.0; DK * DK];
        for i in 0..DK {
            ainv[i * DK + i] = 1.0 / gamma;
        }
        Self {
            ainv,
            b: w_base.iter().map(|v| gamma * v).collect(),
            theta: w_base.to_vec(),
            alpha,
            phi: vec![0.0; DK],
            scratch: vec![0.0; DK],
            n: 0,
        }
    }

    pub fn theta(&self) -> &[f64] {
        &self.theta
    }

    /// UCB score for `(x, p)`: mean + alpha * sqrt(variance). Used to pick during
    /// online exploration, not on the serving hot path.
    pub fn ucb(&mut self, x: &[f64], p: &[f64]) -> f64 {
        bilinear_feature(x, p, &mut self.phi);
        let mean = dot(&self.phi, &self.theta);
        let var = quad_form(&self.ainv, DK, &self.phi, &mut self.scratch);
        mean + self.alpha * var.max(0.0).sqrt()
    }

    pub fn observe(&mut self, x: &[f64], p: &[f64], reward: f64) {
        bilinear_feature(x, p, &mut self.phi);
        sherman_morrison(&mut self.ainv, DK, &self.phi);
        for i in 0..DK {
            self.b[i] += reward * self.phi[i];
        }
        matvec(&self.ainv, DK, DK, &self.b, &mut self.theta);
        self.n += 1;
    }

    /// A serializable snapshot of the online state — everything needed to resume the
    /// LinUCB update after a restart (the scratch buffers are reconstructed on load).
    pub fn to_state(&self) -> BanditState {
        BanditState {
            ainv: self.ainv.clone(),
            b: self.b.clone(),
            theta: self.theta.clone(),
            n: self.n,
        }
    }

    /// Rebuild a bandit from a persisted snapshot at the learner's alpha. Rejects a
    /// snapshot whose matrices are the wrong shape (a stale/corrupt state file).
    pub fn from_state(state: &BanditState, alpha: f64) -> Option<Self> {
        if state.ainv.len() != DK * DK || state.b.len() != DK || state.theta.len() != DK {
            return None;
        }
        Some(Self {
            ainv: state.ainv.clone(),
            b: state.b.clone(),
            theta: state.theta.clone(),
            alpha,
            phi: vec![0.0; DK],
            scratch: vec![0.0; DK],
            n: state.n,
        })
    }
}

/// The persisted form of one user's LinUCB bandit — `A^-1`, `b`, `theta`, and the
/// observation count. Serialized into the state artifact's metadata so restarts do
/// not lose per-user adaptation. Same shape across every scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditState {
    pub ainv: Vec<f64>,
    pub b: Vec<f64>,
    pub theta: Vec<f64>,
    pub n: u32,
}

/// Owns the base policy and the per-user bandits.
pub struct Learner {
    pub base: Policy,
    users: HashMap<String, Bandit>,
    gamma: f64,
    alpha: f64,
}

impl Learner {
    pub fn new(base: Policy, gamma: f64, alpha: f64) -> Self {
        Self {
            base,
            users: HashMap::new(),
            gamma,
            alpha,
        }
    }

    /// Effective per-user weights: the user's `theta` if they have a bandit, else
    /// the base `W`.
    pub fn effective_w(&self, user_id: &str) -> &[f64] {
        self.users
            .get(user_id)
            .map(Bandit::theta)
            .unwrap_or(&self.base.w)
    }

    pub fn bandit_mut(&mut self, user_id: &str) -> &mut Bandit {
        let base = &self.base;
        let (gamma, alpha) = (self.gamma, self.alpha);
        self.users
            .entry(user_id.to_string())
            .or_insert_with(|| Bandit::with_prior(&base.w, gamma, alpha))
    }

    pub fn observe(&mut self, user_id: &str, x: &[f64], p: &[f64], reward: f64) {
        self.bandit_mut(user_id).observe(x, p, reward);
    }

    /// Export every user's bandit snapshot for persistence (empty when no user has
    /// been observed yet — a fresh base carries no online state).
    pub fn user_states(&self) -> Vec<(String, BanditState)> {
        self.users
            .iter()
            .map(|(id, b)| (id.clone(), b.to_state()))
            .collect()
    }

    /// Restore per-user bandits from persisted snapshots at this learner's alpha,
    /// skipping any wrong-shaped snapshot. The base `W` prior is unchanged — only the
    /// online deltas are layered back on.
    pub fn restore(&mut self, states: &[(String, BanditState)]) {
        for (id, st) in states {
            if let Some(b) = Bandit::from_state(st, self.alpha) {
                self.users.insert(id.clone(), b);
            }
        }
    }

    pub fn user_count(&self) -> usize {
        self.users.len()
    }

    /// `||dW_u|| = ||theta_u - W||`, the size of the per-user adaptation.
    pub fn delta_norm(&self, user_id: &str) -> f64 {
        self.users
            .get(user_id)
            .map(|b| {
                b.theta()
                    .iter()
                    .zip(&self.base.w)
                    .map(|(t, w)| (t - w) * (t - w))
                    .sum::<f64>()
                    .sqrt()
            })
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Policy, D, K};

    fn one_hot(i: usize, n: usize) -> Vec<f64> {
        let mut v = vec![0.0; n];
        v[i] = 1.0;
        v
    }

    #[test]
    fn fit_base_recovers_quality_reading() {
        // Reward = p's quality column for the request's task. Fit must learn it.
        let xs: Vec<Vec<f64>> = (0..4).map(|t| one_hot(t, D)).collect();
        let mut train = Vec::new();
        for (t, x) in xs.iter().enumerate() {
            for q in [0.2_f64, 0.5, 0.9] {
                let mut p = vec![0.0; K];
                p[t] = q;
                train.push((x.clone(), p, q));
            }
        }
        let base = fit_base(
            train
                .iter()
                .map(|(x, p, y)| (x.as_slice(), p.as_slice(), *y)),
            0.01,
        );
        let mut p = vec![0.0; K];
        p[2] = 0.7;
        let pred = base.utility(&one_hot(2, D), &p);
        assert!((pred - 0.7).abs() < 0.05, "predicted {pred}, want ~0.7");
    }

    #[test]
    fn bandit_prior_is_base_and_observation_shifts_it() {
        let base = Policy::from_weights(vec![0.0; D * K]);
        let mut learner = Learner::new(base, 1.0, 0.5);
        assert_eq!(learner.delta_norm("u"), 0.0);

        let mut x = vec![0.0; D];
        x[0] = 1.0;
        let mut p = vec![0.0; K];
        p[0] = 1.0;
        learner.observe("u", &x, &p, 1.0);
        assert!(
            learner.delta_norm("u") > 0.0,
            "an observation must move theta off the prior"
        );
    }
}
