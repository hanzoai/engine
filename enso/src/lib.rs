//! # enso
//!
//! enso is the learned **policy** (the brain) for `hanzo-router` (the mechanism).
//! `hanzo-router` owns the registry, the SLO gate, placement and dispatch and
//! exposes one seam -- the [`RoutePolicy`] trait. enso implements it, turning a
//! request into a [`Route`] (which model, at what level, in what modality, with
//! what confidence), and falls back to the router's rule-based policy until it
//! has eval data.
//!
//! Six orthogonal pieces, one concern each:
//! 1. [`featurize`] -- request -> feature vector `x` (hashing, sub-microsecond).
//! 2. [`profile`] -- eval rows `p` per (model, level, modality); `eval_ingest`.
//! 3. [`policy`] -- the learnable bilinear utility `x^T W p`.
//! 4. [`guard`] -- a two-tier safety gate (hot path + escalation seam).
//! 5. [`selector`] -- safety-gated, SLO-feasible constrained argmax.
//! 6. [`learner`] -- offline ridge fit of `W` + online per-user LinUCB.
//!
//! The registry is cross-modal: LLMs, dub generators and image models are all
//! just rows -- adding a model is adding a row.
//!
//! Real eval data plugs in at [`profile::parse_jsonl`] / [`profile::ingest`]
//! (point them at a bench's JSONL); absent that, [`synth`] emits clearly-labeled
//! synthetic tuples so the mechanism is provable today.

pub mod featurize;
pub mod guard;
pub mod learner;
pub mod linalg;
pub mod policy;
pub mod profile;
pub mod selector;
pub mod synth;

use hanzo_router::classify::Request;
use hanzo_router::registry::{Level, Modality, Registry, Task};
use hanzo_router::{Route, RoutePolicy, Slo, User};

pub use featurize::{Featurizer, HashFeaturizer};
pub use guard::{
    DistilledTeacher, GuardTier, GuardVerdict, Safety, SafetyGuard, Teacher, TwoTierGuard,
};
pub use learner::{fit_base, Bandit, Learner, LearnerStats};
pub use policy::Policy;
pub use profile::{ingest, parse_jsonl, EvalSample, Profile, ProfileTable, PROFILE_DIM};
pub use selector::{Choice, SelectCtx, Selector};

/// The observability channel: everything the decision touched, for "show, don't
/// claim". Production reads [`Route`]; the proof reads this.
#[derive(Debug, Clone)]
pub struct Explain {
    pub guard: GuardVerdict,
    pub used_fallback: bool,
    pub task: Task,
    pub modality: Modality,
    pub scores: Vec<(String, Level, f64)>,
    pub margin: f64,
}

/// The composed learned policy. Generic over the featurizer and guard so either
/// piece swaps without touching the rest; the cold-start fallback is the
/// router's own rule-based [`hanzo_router::Policy`].
pub struct Enso<F: Featurizer = HashFeaturizer, G: SafetyGuard = TwoTierGuard> {
    feat: F,
    table: ProfileTable,
    learner: Learner,
    guard: G,
    selector: Selector,
    fallback: hanzo_router::Policy,
}

impl Enso<HashFeaturizer, TwoTierGuard> {
    pub fn new(table: ProfileTable, learner: Learner) -> Self {
        Self::with_pieces(
            HashFeaturizer::default(),
            table,
            learner,
            TwoTierGuard::default(),
        )
    }
}

impl<F: Featurizer, G: SafetyGuard> Enso<F, G> {
    /// Compose the pieces. Panics unless `feat.dim() == policy::D`: the bilinear
    /// utility fixes `W` at `D x K`, so [`crate::linalg::bilinear`] reads only
    /// `x[..D]` and would silently score a wider `x` on its truncation. `dim()`
    /// is the [`Featurizer`] contract and this is where it is enforced -- once,
    /// at wiring time, so the hot path stays a bare dot product. A trained
    /// encoder wider than `D` (zen-router's 256-wide `feature_head`) must
    /// project down to `D` before it can plug in.
    pub fn with_pieces(feat: F, table: ProfileTable, learner: Learner, guard: G) -> Self {
        assert_eq!(
            feat.dim(),
            policy::D,
            "featurizer dim {} != policy::D {}",
            feat.dim(),
            policy::D
        );
        Self {
            feat,
            table,
            learner,
            guard,
            selector: Selector,
            fallback: hanzo_router::Policy::default(),
        }
    }

    pub fn learner(&self) -> &Learner {
        &self.learner
    }

    pub fn learner_mut(&mut self) -> &mut Learner {
        &mut self.learner
    }

    pub fn table(&self) -> &ProfileTable {
        &self.table
    }

    pub fn guard(&self) -> &G {
        &self.guard
    }

    pub fn featurizer(&self) -> &F {
        &self.feat
    }

    /// Route and return the full [`Explain`] alongside.
    pub fn route_explained(
        &self,
        req: &Request,
        user: &User,
        slo: &Slo,
        reg: &Registry,
    ) -> (Route, Explain) {
        self.decide(req, user, slo, reg, true)
    }

    /// Record an outcome for `(user, request, chosen model/level/modality)`: the
    /// online feedback that drives per-user adaptation.
    pub fn observe(
        &mut self,
        user: &User,
        req: &Request,
        model: &str,
        level: Level,
        modality: Modality,
        reward: f64,
    ) {
        let x = self.feat.featurize(req);
        if let Some(p) = self.table.get(model, level, modality).map(|p| p.features()) {
            self.learner.observe(&user.id, &x, &p, reward);
        }
    }

    fn decide(
        &self,
        req: &Request,
        user: &User,
        slo: &Slo,
        reg: &Registry,
        collect: bool,
    ) -> (Route, Explain) {
        let task = self.feat.task_of(req);
        let modality = req.target_modality();
        let verdict = self.guard.check(req);
        if verdict.safety == Safety::Block {
            let route = Route::refused(verdict.confidence as f32);
            return (
                route,
                Explain {
                    guard: verdict,
                    used_fallback: false,
                    task,
                    modality,
                    scores: Vec::new(),
                    margin: 0.0,
                },
            );
        }
        if self.table.for_modality(modality).next().is_none() {
            let route = self.fallback.route(req, user, slo, reg);
            return (
                route,
                Explain {
                    guard: verdict,
                    used_fallback: true,
                    task,
                    modality,
                    scores: Vec::new(),
                    margin: 0.0,
                },
            );
        }
        let x = self.feat.featurize(req);
        let w = self.learner.effective_w(&user.id);
        let mut scores = collect.then(Vec::new);
        let ctx = SelectCtx {
            x: &x,
            w,
            table: &self.table,
            want: modality,
            task,
            slo,
        };
        match self.selector.select(&ctx, scores.as_mut()) {
            Some(choice) => {
                let route = Route {
                    model: choice.model,
                    level: choice.level,
                    modality,
                    confidence: choice.confidence as f32,
                };
                let margin = top_two_margin(scores.as_deref());
                (
                    route,
                    Explain {
                        guard: verdict,
                        used_fallback: false,
                        task,
                        modality,
                        scores: scores.unwrap_or_default(),
                        margin,
                    },
                )
            }
            None => {
                let route = self.fallback.route(req, user, slo, reg);
                (
                    route,
                    Explain {
                        guard: verdict,
                        used_fallback: true,
                        task,
                        modality,
                        scores: scores.unwrap_or_default(),
                        margin: 0.0,
                    },
                )
            }
        }
    }
}

impl<F: Featurizer, G: SafetyGuard> RoutePolicy for Enso<F, G> {
    fn route(&self, req: &Request, user: &User, slo: &Slo, reg: &Registry) -> Route {
        self.decide(req, user, slo, reg, false).0
    }
}

fn top_two_margin(scores: Option<&[(String, Level, f64)]>) -> f64 {
    let Some(scores) = scores else { return 0.0 };
    let mut best = f64::NEG_INFINITY;
    let mut second = f64::NEG_INFINITY;
    for &(_, _, s) in scores {
        if s > best {
            second = best;
            best = s;
        } else if s > second {
            second = s;
        }
    }
    if second.is_finite() {
        best - second
    } else {
        best.abs()
    }
}
