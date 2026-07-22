//! The trained-featurizer binding contract.
//!
//! [`Featurizer`] is the seam a trained encoder plugs into: it replaces the
//! hand-written [`HashFeaturizer`] without any other piece changing. The
//! contract is `dim()` -- the bilinear utility `x^T W p` fixes `W` at `D x K`
//! with `D = policy::D` a compile-time constant, so `x` is only meaningful at
//! exactly `D`.
//!
//! These tests pin both halves of that contract: an encoder AT `D` dispatches
//! through [`RoutePolicy`], and an encoder off `D` is refused where it is wired
//! rather than silently mis-scored where it serves.

use enso::policy::D;
use enso::{DistilledTeacher, Enso, Featurizer, Learner, Policy, ProfileTable, TwoTierGuard};
use hanzo_router::classify::Request;
use hanzo_router::registry::{Backend, ModelCard, Registry, Task};
use hanzo_router::{RoutePolicy, Slo, User};

/// The shape of the only trained featurizer that exists today: zen-router's
/// `feature_head`, a 1024 -> 256 linear map over the frozen backbone's pooled
/// last hidden state (`zenlm/zen-router`, `export/heads.safetensors`).
const ZEN_FEATURE_DIM: usize = 256;

/// A trained encoder stand-in, emitting `dim` constant features. The values do
/// not matter to the contract -- the dimension does.
struct Encoder {
    dim: usize,
    task: Task,
}

impl Featurizer for Encoder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn featurize(&self, _req: &Request) -> Vec<f64> {
        vec![0.5; self.dim]
    }

    fn task_of(&self, _req: &Request) -> Task {
        self.task
    }
}

fn learner() -> Learner {
    Learner::new(Policy::zeros(), 1.0, 0.5)
}

/// The default guard; named so the `Teacher` parameter is inferable at the
/// `with_pieces` call sites, where only the featurizer is under test.
fn guard() -> TwoTierGuard<DistilledTeacher> {
    TwoTierGuard::default()
}

/// A pool with one profile, so the learned path is live (an empty table for the
/// request's modality short-circuits to the rule-based fallback).
fn table() -> ProfileTable {
    enso::ingest(&enso::synth::gen_eval_samples(
        &enso::synth::lineup(),
        64,
        7,
    ))
}

fn registry() -> Registry {
    Registry::new(vec![ModelCard {
        id: "zen5-coder".into(),
        backend: Backend::Local { est_bytes: 1 },
        tasks: vec![Task::Code, Task::General],
        max_context: 0,
        vision: false,
        cost_per_1k: 0.0,
    }])
}

/// The seam is genuinely open AT `D`: a foreign featurizer that honors the
/// contract is accepted and its route is served through the trait.
#[test]
fn an_encoder_at_d_dispatches_through_the_seam() {
    let enso = Enso::with_pieces(
        Encoder {
            dim: D,
            task: Task::Code,
        },
        table(),
        learner(),
        guard(),
    );
    let route = enso.route(
        &Request {
            text: "hello".into(),
            ..Default::default()
        },
        &User::anonymous(),
        &Slo::default(),
        &registry(),
    );
    assert!(!route.is_refused(), "a D-dim encoder must serve a route");
}

/// The other half: an encoder off `D` is refused at wiring time. zen-router's
/// `feature_head` is 256; `D` is 16. Without this check `bilinear` reads only
/// `x[..D]` and scores a truncated request as if it were whole -- a wrong route
/// with no error anywhere.
#[test]
#[should_panic(expected = "featurizer dim 256 != policy::D 16")]
fn the_zen_router_feature_head_is_refused_at_wiring_time() {
    Enso::with_pieces(
        Encoder {
            dim: ZEN_FEATURE_DIM,
            task: Task::Code,
        },
        table(),
        learner(),
        guard(),
    );
}

/// Why the wiring check has to exist: the scoring primitive itself cannot
/// notice. `bilinear` walks `0..D`, so an over-long `x` loses `x[D..]` and still
/// returns a plausible number. This characterizes the primitive -- the guard
/// above is what keeps a mis-dimensioned encoder from ever reaching it.
#[test]
fn bilinear_cannot_see_the_truncation_itself() {
    let k = enso::policy::K;
    let w = vec![1.0; D * k];
    let p = vec![1.0; k];
    let short = vec![1.0; D];
    let long = {
        let mut v = vec![1.0; D];
        v.extend(std::iter::repeat_n(9999.0, ZEN_FEATURE_DIM - D));
        v
    };
    assert_eq!(
        enso::linalg::bilinear(&short, &w, &p, D, k),
        enso::linalg::bilinear(&long, &w, &p, D, k),
        "bilinear must be blind to x[D..] -- which is exactly why dim() is \
         enforced at construction, not here on the hot path"
    );
}
