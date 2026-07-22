//! The learned routing head: the alternative scorer that [`Policy`] uses when a
//! fitted head is mounted. It carries the two learned objects --- the bilinear
//! weights `W` and one feature vector per arm (the arm's eval-measured profile) ---
//! and ranks arms by `utility(x, p) = x^T W p` minus the SLO's soft cost/latency
//! penalty, exactly as the offline selector does. It is plain data + arithmetic,
//! no eval or fit machinery, so it lives in the router next to [`Policy`] and the
//! learned path stays inside the one policy surface.
//!
//! `hanzo-router-retrain` fits `W` and the arm profiles and serializes them here;
//! this is the artifact the serving engine loads (`ROUTER_HEADS`).
//!
//! [`Policy`]: crate::policy::Policy

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::featurize::{FEAT_DIM, NUM_TASKS};
use crate::route::Slo;

/// Feature-vector index of the arm's normalized latency / cost. The profile layout
/// is [quality-by-task (8) | latency_norm | cost_norm | vram_norm | ctx_norm], so
/// these sit right after the task block.
const LAT_IDX: usize = NUM_TASKS;
const COST_IDX: usize = NUM_TASKS + 1;

const MARGIN_SCALE: f64 = 8.0;
const SOLE_CONFIDENCE: f32 = 0.85;

/// One candidate arm as the head sees it: its id and its profile feature vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Arm {
    pub model: String,
    pub feat: Vec<f64>,
}

/// The persisted learned head: bilinear weights + the arm profiles they rank.
/// `d`/`k` are the feature and profile dims the fit used, checked on load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heads {
    pub d: usize,
    pub k: usize,
    pub w: Vec<f64>,
    pub arms: Vec<Arm>,
}

/// `x^T W p` for a row-major `d x k` weight matrix.
fn bilinear(x: &[f64], w: &[f64], p: &[f64], d: usize, k: usize) -> f64 {
    let mut acc = 0.0;
    for (i, &xi) in x.iter().enumerate().take(d) {
        if xi == 0.0 {
            continue;
        }
        let row = i * k;
        let mut s = 0.0;
        for j in 0..k {
            s += w[row + j] * p[j];
        }
        acc += xi * s;
    }
    acc
}

impl Heads {
    pub fn new(w: Vec<f64>, arms: Vec<Arm>) -> Self {
        Self {
            d: FEAT_DIM,
            k: w.len() / FEAT_DIM.max(1),
            w,
            arms,
        }
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let heads: Heads = serde_json::from_slice(&bytes).map_err(std::io::Error::other)?;
        if heads.w.len() != heads.d * heads.k {
            return Err(std::io::Error::other("heads: w len != d*k"));
        }
        Ok(heads)
    }

    /// Persist atomically (temp sibling + rename) so a crash mid-write cannot leave
    /// a half-written bundle.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(
            &tmp,
            serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?,
        )?;
        std::fs::rename(&tmp, path)
    }

    /// The learned pick over the arms: argmax of `utility - lambda*cost - mu*latency`.
    /// Returns the arm id and a margin-squashed confidence, or `None` if empty.
    pub fn best(&self, x: &[f64], slo: &Slo) -> Option<(String, f32)> {
        let (mut best, mut best_obj, mut runner) = (None, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for a in &self.arms {
            let obj = bilinear(x, &self.w, &a.feat, self.d, self.k)
                - slo.lambda_cost as f64 * a.feat.get(COST_IDX).copied().unwrap_or(0.0)
                - slo.mu_latency as f64 * a.feat.get(LAT_IDX).copied().unwrap_or(0.0);
            if obj > best_obj {
                runner = best_obj;
                best_obj = obj;
                best = Some(a.model.clone());
            } else if obj > runner {
                runner = obj;
            }
        }
        best.map(|m| {
            let conf = if runner.is_finite() {
                (1.0 / (1.0 + (-(best_obj - runner) * MARGIN_SCALE).exp())) as f32
            } else {
                SOLE_CONFIDENCE
            };
            (m, conf)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Task;

    // Profile layout the head consumes: quality-by-task (8) + [lat, cost, vram, ctx].
    const K: usize = NUM_TASKS + 4;

    fn arm(model: &str, q_general: f64) -> Arm {
        let mut feat = vec![0.0; K];
        feat[Task::General.index()] = q_general;
        Arm {
            model: model.into(),
            feat,
        }
    }

    #[test]
    fn best_reads_the_general_quality_column() {
        // W that reads x[General] * p[General]: higher-quality arm wins.
        let g = Task::General.index();
        let k = K;
        let mut w = vec![0.0; FEAT_DIM * k];
        w[g * k + g] = 1.0;
        let heads = Heads::new(w, vec![arm("lo", 0.2), arm("hi", 0.9)]);
        let mut x = vec![0.0; FEAT_DIM];
        x[g] = 1.0;
        let (m, _c) = heads
            .best(
                &x,
                &Slo {
                    lambda_cost: 0.0,
                    mu_latency: 0.0,
                    ..Slo::default()
                },
            )
            .unwrap();
        assert_eq!(m, "hi");
    }
}
