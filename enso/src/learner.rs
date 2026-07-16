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
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Bandit {
    ainv: Vec<f64>,
    b: Vec<f64>,
    theta: Vec<f64>,
    alpha: f64,
    // `phi`/`scratch` are per-call working buffers, not learned state: they carry nothing between
    // observations, so they are neither persisted nor part of the numeric identity. On load they are
    // re-sized from the recovered `ainv`, so a reloaded bandit is byte-identical in every field that
    // `observe`/`ucb` read.
    #[serde(skip)]
    phi: Vec<f64>,
    #[serde(skip)]
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
}

impl Bandit {
    /// Re-size the working buffers after a `#[serde(skip)]` load. `ainv` is `DK*DK`, so `DK` is its
    /// integer square root; `phi`/`scratch` are `DK`. Deterministic from the recovered state.
    fn rehydrate(&mut self) {
        let dk = (self.ainv.len() as f64).sqrt() as usize;
        self.phi = vec![0.0; dk];
        self.scratch = vec![0.0; dk];
    }
}

/// A single, pure snapshot of what the learner has actually learned --- the answer to *"is enso
/// serving a learned policy, or silently falling back?"*, made observable rather than inferred.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LearnerStats {
    /// `true` once the base `W` is a real fit (any nonzero weight); `false` is the cold prior, i.e.
    /// serving is the rule-based fallback until data arrives.
    pub base_fit: bool,
    /// Users with an online bandit.
    pub users: usize,
    /// Total observations ingested across all users --- the size of the feedback the model has seen.
    pub total_observations: u64,
    /// Users whose bandit has drifted measurably from the base (`||dW_u|| > 1e-6`).
    pub adapted_users: usize,
    /// The largest per-user adaptation `max_u ||dW_u||` --- how far any user has personalized.
    pub max_delta_norm: f64,
}

/// Owns the base policy and the per-user bandits.
#[derive(serde::Serialize, serde::Deserialize)]
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

    /// What the learner has actually learned, as a pure value (see [`LearnerStats`]). This is the
    /// signal that keeps enso's fallback honest: an operator reads whether it is serving a fit policy
    /// and how much feedback it has absorbed, instead of assuming the loop is closed because the
    /// process is up. The two failure modes this exposes are "`base_fit == false`" (no eval data ever
    /// ingested, so serving is the rule-based fallback) and "`total_observations` flat across restarts"
    /// (outcomes not durably persisted, so learning resets every run).
    pub fn stats(&self) -> LearnerStats {
        let base_fit = self.base.w.iter().any(|&w| w != 0.0);
        let total_observations = self.users.values().map(|b| u64::from(b.n)).sum();
        let deltas: Vec<f64> = self.users.keys().map(|u| self.delta_norm(u)).collect();
        LearnerStats {
            base_fit,
            users: self.users.len(),
            total_observations,
            adapted_users: deltas.iter().filter(|&&d| d > 1e-6).count(),
            max_delta_norm: deltas.iter().copied().fold(0.0, f64::max),
        }
    }

    /// Persist the learner to `path` atomically (write a sibling temp file, then rename), so a crash
    /// or a signal during the write cannot leave a half-written snapshot. Durability is a property of
    /// this call, not of how the process ends: the owner checkpoints on a cadence (and after ingest),
    /// so a signal-terminated serving process loses at most the last interval --- unlike an `atexit`
    /// hook, which a signal never fires.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(self).map_err(std::io::Error::other)?)?;
        std::fs::rename(&tmp, path)
    }

    /// Load a learner previously written by [`Learner::save`], re-sizing each bandit's working buffers
    /// (which are not persisted). Returns `Ok(None)` when the file does not exist yet --- a first run
    /// starts from the given base rather than an error.
    pub fn load(path: &std::path::Path) -> std::io::Result<Option<Self>> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        let mut learner: Self = serde_json::from_slice(&bytes).map_err(std::io::Error::other)?;
        for bandit in learner.users.values_mut() {
            bandit.rehydrate();
        }
        Ok(Some(learner))
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

    #[test]
    fn save_load_roundtrip_preserves_learned_state() {
        // Build a learner with a fit base and a personalized user, persist it, reload it, and require
        // that the learned quantities survive exactly -- otherwise every restart silently resets the
        // model to its cold prior (the failure mode this durability exists to prevent).
        let base = Policy::from_weights((0..D * K).map(|i| (i as f64 % 7.0) - 3.0).collect());
        let mut learner = Learner::new(base, 1.0, 0.5);
        let mut x = vec![0.0; D];
        x[1] = 1.0;
        let mut p = vec![0.0; K];
        p[2] = 0.8;
        for _ in 0..5 {
            learner.observe("alice", &x, &p, 0.9);
        }
        let before = learner.stats();
        let w_before = learner.effective_w("alice").to_vec();
        assert!(before.base_fit && before.total_observations == 5 && before.adapted_users == 1);

        let dir = std::env::temp_dir().join(format!("enso-learner-{}", std::process::id()));
        let path = dir.join("state.json");
        learner.save(&path).unwrap();
        let reloaded = Learner::load(&path).unwrap().expect("file exists after save");

        // Observability survives...
        let after = reloaded.stats();
        assert_eq!(after.total_observations, before.total_observations);
        assert_eq!(after.users, before.users);
        assert_eq!(after.adapted_users, before.adapted_users);
        assert!((after.max_delta_norm - before.max_delta_norm).abs() < 1e-12);
        // ...and so does the numeric policy the reloaded learner will serve.
        let w_after = reloaded.effective_w("alice");
        assert_eq!(w_after.len(), w_before.len());
        assert!(
            w_before.iter().zip(w_after).all(|(a, b)| (a - b).abs() < 1e-12),
            "reloaded theta must match the persisted one bit-for-bit"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_missing_file_is_first_run_not_error() {
        let path = std::env::temp_dir().join("enso-nonexistent-xyz/none.json");
        assert!(Learner::load(&path).unwrap().is_none());
    }

    #[test]
    fn stats_distinguishes_fallback_from_learned() {
        // Cold prior (all-zero base, no users) is the fallback signal; a fit base flips base_fit.
        let cold = Learner::new(Policy::from_weights(vec![0.0; D * K]), 1.0, 0.5);
        assert!(!cold.stats().base_fit && cold.stats().total_observations == 0);
        let warm = Learner::new(Policy::from_weights(vec![0.1; D * K]), 1.0, 0.5);
        assert!(warm.stats().base_fit);
    }
}
