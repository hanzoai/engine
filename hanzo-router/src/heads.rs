//! The learned routing head: the alternative scorer that [`Policy`] uses when a
//! fitted head is mounted. It carries the two learned objects --- the bilinear
//! weights `W` and one feature vector per arm (the arm's eval-measured profile) ---
//! and ranks arms by `utility(x, p) = x^T W p` minus the SLO's soft cost/latency
//! penalty, exactly as the offline selector does. It is plain data + arithmetic,
//! no eval or fit machinery, so it lives in the router next to [`Policy`] and the
//! learned path stays inside the one policy surface.
//!
//! A fit writes `W` and the arm profiles in this shape;
//! this is the artifact the serving engine loads (`ROUTER_HEADS`).
//!
//! [`Policy`]: crate::policy::Policy

use std::collections::BTreeMap;
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

/// Utilization ceiling the congestion penalty clamps to, so a fully-piled-on arm
/// yields a large-but-finite penalty (rho -> 1 would diverge).
pub const LOAD_CAP: f64 = 0.98;

/// The mean-field congestion penalty for a normalized arm load `load` (an arm's
/// utilization / demand share in `[0, 1)`): the M/M/1 sojourn-time factor
/// `rho / (1 - rho)`. Zero at zero load, strictly increasing, convex — an arm's
/// marginal delay blows up as it saturates. This is the field a load-aware pick
/// best-responds to: routing to the globally-best arm is only worth it until that
/// arm's congestion erodes the win (Wardrop routing — quality is population-dependent
/// once everyone chases the same arm).
pub fn congestion(load: f64) -> f64 {
    let rho = load.clamp(0.0, LOAD_CAP);
    rho / (1.0 - rho)
}

/// A per-arm load field: `model id -> normalized load` (utilization share). A missing
/// arm reads as unloaded (`0.0`). The offline harness feeds a simulated vector; the
/// serving path feeds the replica layer's live counters
/// ([`crate::replica::Balancer::load_field`]).
pub type LoadField = BTreeMap<String, f64>;

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

    /// The static per-arm objective: `x^T W p - lambda*cost - mu*latency`. The one
    /// place the head's utility is defined, so the static and load-aware picks score
    /// arms identically apart from the congestion term.
    fn base_obj(&self, x: &[f64], a: &Arm, slo: &Slo) -> f64 {
        bilinear(x, &self.w, &a.feat, self.d, self.k)
            - slo.lambda_cost as f64 * a.feat.get(COST_IDX).copied().unwrap_or(0.0)
            - slo.mu_latency as f64 * a.feat.get(LAT_IDX).copied().unwrap_or(0.0)
    }

    /// argmax of a per-arm objective, with the margin-squashed confidence both picks
    /// share. `None` when the arm set is empty.
    fn argmax_by(&self, obj: impl Fn(&Arm) -> f64) -> Option<(String, f32)> {
        let (mut best, mut best_obj, mut runner) = (None, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for a in &self.arms {
            let o = obj(a);
            if o > best_obj {
                runner = best_obj;
                best_obj = o;
                best = Some(a.model.clone());
            } else if o > runner {
                runner = o;
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

    /// The learned pick over the arms: argmax of `utility - lambda*cost - mu*latency`.
    /// Returns the arm id and a margin-squashed confidence, or `None` if empty.
    pub fn best(&self, x: &[f64], slo: &Slo) -> Option<(String, f32)> {
        self.argmax_by(|a| self.base_obj(x, a, slo))
    }

    /// The load-aware (mean-field) pick: argmax of `base_obj - gamma * congestion(load[arm])`.
    /// `gamma` is the field strength (`0.0` recovers [`Self::best`] exactly); `load`
    /// maps arm id -> normalized load (a missing arm is unloaded). Because the penalty
    /// is subtracted per arm, a *uniform* load field shifts every arm's objective by the
    /// same constant and the pick is identical to the static [`Self::best`] — congestion
    /// only bends routing once load is *unevenly* concentrated. Monotone: raising an
    /// arm's load never raises its objective, so it can only lose rank.
    pub fn best_field(
        &self,
        x: &[f64],
        slo: &Slo,
        load: &LoadField,
        gamma: f64,
    ) -> Option<(String, f32)> {
        self.argmax_by(|a| {
            self.base_obj(x, a, slo)
                - gamma * congestion(load.get(&a.model).copied().unwrap_or(0.0))
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

    /// A pure-quality SLO and an `x` that lights up the General column — the fixture
    /// the field tests score against.
    fn quality_slo() -> Slo {
        Slo {
            lambda_cost: 0.0,
            mu_latency: 0.0,
            ..Slo::default()
        }
    }

    fn general_heads() -> (Heads, Vec<f64>) {
        let g = Task::General.index();
        let mut w = vec![0.0; FEAT_DIM * K];
        w[g * K + g] = 1.0; // utility reads x[General] * p[General]
        let mut x = vec![0.0; FEAT_DIM];
        x[g] = 1.0;
        (Heads::new(w, vec![arm("lo", 0.2), arm("hi", 0.9)]), x)
    }

    #[test]
    fn best_reads_the_general_quality_column() {
        // W that reads x[General] * p[General]: higher-quality arm wins.
        let (heads, x) = general_heads();
        let (m, _c) = heads.best(&x, &quality_slo()).unwrap();
        assert_eq!(m, "hi");
    }

    #[test]
    fn congestion_is_zero_at_rest_and_strictly_increasing() {
        assert_eq!(congestion(0.0), 0.0);
        let (a, b, c) = (congestion(0.1), congestion(0.5), congestion(0.9));
        assert!(a < b && b < c, "congestion must be strictly increasing");
        // convex: the 0.5->0.9 jump dwarfs the 0.1->0.5 jump.
        assert!((c - b) > (b - a), "congestion must be convex");
        assert!(congestion(1.0).is_finite(), "clamped below rho=1 (finite)");
        assert_eq!(congestion(-1.0), 0.0, "negative load clamps to rest");
    }

    #[test]
    fn field_uniform_load_matches_static_exactly() {
        // A uniform load field is a constant offset across arms => identical pick AND
        // confidence to the static head (the "reduces to static at load=uniform" law).
        let (heads, x) = general_heads();
        let slo = quality_slo();
        let uniform: LoadField = [("lo".into(), 0.4), ("hi".into(), 0.4)].into();
        assert_eq!(
            heads.best_field(&x, &slo, &uniform, 2.0),
            heads.best(&x, &slo)
        );
        // An empty field (all arms unloaded) is uniform at zero — also static.
        assert_eq!(
            heads.best_field(&x, &slo, &LoadField::new(), 2.0),
            heads.best(&x, &slo)
        );
    }

    #[test]
    fn field_gamma_zero_is_static() {
        let (heads, x) = general_heads();
        let slo = quality_slo();
        let skewed: LoadField = [("lo".into(), 0.0), ("hi".into(), 0.95)].into();
        assert_eq!(
            heads.best_field(&x, &slo, &skewed, 0.0),
            heads.best(&x, &slo),
            "gamma=0 disables the field entirely"
        );
    }

    #[test]
    fn field_diverts_off_a_congested_best_arm() {
        // "hi" (q=0.9) wins statically, but pile load on it and a strong-enough field
        // best-responds to the low-loaded "lo" — the mean-field claim in miniature.
        let (heads, x) = general_heads();
        let slo = quality_slo();
        let hot: LoadField = [("lo".into(), 0.0), ("hi".into(), 0.9)].into();
        assert_eq!(heads.best_field(&x, &slo, &hot, 0.0).unwrap().0, "hi");
        assert_eq!(
            heads.best_field(&x, &slo, &hot, 1.0).unwrap().0,
            "lo",
            "a congested best arm yields to the idle runner-up under a strong field"
        );
    }

    #[test]
    fn field_is_monotone_in_load() {
        // Raising the winning arm's load only ever lowers its objective, so the pick
        // can flip away from it but never toward it.
        let (heads, x) = general_heads();
        let slo = quality_slo();
        let mut prev_hi_wins = true;
        for &l in &[0.0, 0.3, 0.6, 0.85, 0.95] {
            let field: LoadField = [("lo".into(), 0.0), ("hi".into(), l)].into();
            let hi_wins = heads.best_field(&x, &slo, &field, 1.0).unwrap().0 == "hi";
            assert!(
                prev_hi_wins || !hi_wins,
                "once diverted off the loaded arm it must not come back as load rises"
            );
            prev_hi_wins = hi_wins;
        }
        assert!(!prev_hi_wins, "at near-saturation the loaded arm must lose");
    }
}
