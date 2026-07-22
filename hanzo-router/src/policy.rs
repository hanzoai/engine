//! The routing brain: given a classified task, the pool, what's already running,
//! and a memory snapshot, decide where to serve — preferring a model already
//! loaded (free), else a local model that *fits* available memory, else cloud.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::classify::{Classifier, Heuristic, Request};
use crate::featurize::{Featurizer, HashFeaturizer};
use crate::memory::{default_fraction, MemSnapshot};
use crate::registry::{Backend, Level, ModelCard, Registry, Task};
use crate::route::{Route, RoutePolicy, Slo, User, COLD_START_CONFIDENCE};

/// Per-task preferences + global knobs. Loadable from YAML (the same declarative
/// shape as the Python `router_policy.yaml`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Policy {
    /// Ordered model-id preference per task (first that's usable wins).
    #[serde(default)]
    pub prefer: std::collections::BTreeMap<String, Vec<String>>,
    /// Memory fraction override (else [`default_fraction`]).
    #[serde(default)]
    pub memory_fraction: Option<f64>,
    /// Optional cost ceiling (per-1k) for cloud selection.
    #[serde(default)]
    pub cost_ceiling: Option<f64>,
    /// A mounted learned head (bilinear `W` + arm profiles). When present,
    /// [`RoutePolicy::route`] scores arms with it instead of the rule-based walk,
    /// so the learned path lives inside this one policy surface. Not serialized:
    /// it is loaded from the `ROUTER_HEADS` bundle, not the YAML policy file.
    #[serde(skip)]
    pub learned: Option<std::sync::Arc<crate::heads::Heads>>,
}

/// Where the router decided to serve the request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "route", rename_all = "snake_case")]
pub enum Decision {
    /// Reuse an already-loaded local model (no load cost).
    Reuse { model: String },
    /// Load a local model that fits available memory, then serve.
    LoadLocal { model: String, est_bytes: u64 },
    /// Serve via a cloud provider.
    Cloud { provider: String, model: String },
    /// Nothing usable (no running model, nothing fits, no cloud) — caller errors.
    NoFit,
}

/// The decision context the caller assembles (all values — pure inputs).
pub struct Context<'a> {
    pub task: Task,
    pub registry: &'a Registry,
    pub mem: MemSnapshot,
    /// Model ids currently loaded/running in connectable engines.
    pub running: &'a BTreeSet<String>,
    pub vision_required: bool,
    pub min_context: usize,
}

impl Policy {
    fn fraction(&self, unified: bool) -> f64 {
        self.memory_fraction
            .unwrap_or_else(|| default_fraction(unified))
    }

    fn usable(&self, m: &ModelCard, ctx: &Context) -> bool {
        if ctx.vision_required && !m.vision {
            return false;
        }
        if m.max_context != 0 && m.max_context < ctx.min_context {
            return false;
        }
        if let (Some(ceiling), Backend::Cloud { .. }) = (self.cost_ceiling, &m.backend) {
            if m.cost_per_1k > ceiling {
                return false;
            }
        }
        true
    }

    /// Candidate model ids in preference order: (1) the policy's explicit
    /// `prefer[task]` list, (2) models that **explicitly** advertise the task,
    /// (3) `General` catch-all models last. So a task-specialist (even cloud)
    /// outranks a general-purpose model — task fit before convenience.
    fn candidates(&self, ctx: &Context) -> Vec<String> {
        let task_key = serde_json::to_value(ctx.task)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        let push = |id: &str, out: &mut Vec<String>, seen: &mut BTreeSet<String>| {
            if seen.insert(id.to_string()) {
                out.push(id.to_string());
            }
        };
        if let Some(pref) = self.prefer.get(&task_key) {
            for id in pref {
                push(id, &mut out, &mut seen);
            }
        }
        for m in &ctx.registry.models {
            if m.tasks.contains(&ctx.task) {
                push(&m.id, &mut out, &mut seen);
            }
        }
        for m in &ctx.registry.models {
            if m.tasks.contains(&Task::General) {
                push(&m.id, &mut out, &mut seen);
            }
        }
        out
    }

    /// The decision. (1) Reuse a running, usable candidate (zero load cost).
    /// Then a **single ordered walk** of candidates — the first that is either a
    /// local model that *fits* available memory (→ load it locally) or a usable
    /// cloud model (→ route to it). A higher-preference local model that doesn't
    /// fit is skipped, falling through to the next preference (which may be cloud)
    /// — "run it locally if the RAM is there, else the next-best wherever it is."
    pub fn select(&self, ctx: &Context) -> Decision {
        let cards: Vec<&ModelCard> = self
            .candidates(ctx)
            .iter()
            .filter_map(|id| ctx.registry.get(id))
            .filter(|m| self.usable(m, ctx))
            .collect();
        let frac = self.fraction(ctx.mem.unified);

        // (1) Reuse anything already loaded.
        for m in &cards {
            if m.backend.is_local() && ctx.running.contains(&m.id) {
                return Decision::Reuse {
                    model: m.id.clone(),
                };
            }
        }
        // (2) Ordered walk: first local-that-fits or cloud, by preference.
        for m in &cards {
            match &m.backend {
                Backend::Local { est_bytes } if ctx.mem.fits(*est_bytes, frac) => {
                    return Decision::LoadLocal {
                        model: m.id.clone(),
                        est_bytes: *est_bytes,
                    };
                }
                Backend::Cloud { provider } => {
                    return Decision::Cloud {
                        provider: provider.clone(),
                        model: m.id.clone(),
                    };
                }
                Backend::Local { .. } => {} // doesn't fit — try next preference
            }
        }
        Decision::NoFit
    }
}

/// The a-priori, quality-first per-task preference over the cloud-arm pool
/// {opus-4.8, gpt-5.5, deepseek-v4-pro, fable-5}, ported verbatim from the live
/// evaluation harness routing policy. Set from the arms' documented single-model
/// strengths, not fit to any eval. Keys are [`Task`] snake_case labels; the first
/// list entry present in the registry wins.
pub const PREFER: &[(&str, &[&str])] = &[
    (
        "code",
        &["gpt-5.5", "opus-4.8", "deepseek-v4-pro", "fable-5"],
    ),
    (
        "math",
        &["gpt-5.5", "opus-4.8", "deepseek-v4-pro", "fable-5"],
    ),
    (
        "reasoning",
        &["opus-4.8", "gpt-5.5", "deepseek-v4-pro", "fable-5"],
    ),
    (
        "creative",
        &["fable-5", "opus-4.8", "gpt-5.5", "deepseek-v4-pro"],
    ),
    (
        "vision",
        &["opus-4.8", "gpt-5.5", "fable-5", "deepseek-v4-pro"],
    ),
    (
        "long_context",
        &["opus-4.8", "gpt-5.5", "deepseek-v4-pro", "fable-5"],
    ),
    (
        "cheap_chat",
        &["deepseek-v4-pro", "fable-5", "gpt-5.5", "opus-4.8"],
    ),
    (
        "general",
        &["opus-4.8", "gpt-5.5", "deepseek-v4-pro", "fable-5"],
    ),
];

/// A [`Policy`] whose `prefer` table is [`PREFER`] --- the a-priori routing
/// heuristic as a first-class hanzo-router policy. The bare [`Policy::default`]
/// stays empty (registry-order routing); this is the preference-aware default the
/// serving path uses when no learned head is mounted.
pub fn prefer() -> Policy {
    Policy {
        prefer: PREFER
            .iter()
            .map(|(k, v)| (k.to_string(), v.iter().map(|s| s.to_string()).collect()))
            .collect(),
        memory_fraction: None,
        cost_ceiling: None,
        learned: None,
    }
}

impl Policy {
    /// Serve the learned head `heads`, falling back to the [`prefer`] table's
    /// rule-based walk only if the head has no arms. This keeps exactly one
    /// [`Policy`] type at the routing seam.
    pub fn with_heads(heads: crate::heads::Heads) -> Self {
        Self {
            learned: Some(std::sync::Arc::new(heads)),
            ..prefer()
        }
    }

    /// Load a `ROUTER_HEADS` serve bundle from `path` and mount it ([`with_heads`]).
    pub fn load_heads(path: &std::path::Path) -> std::io::Result<Self> {
        crate::heads::Heads::load(path).map(Self::with_heads)
    }
}

impl RoutePolicy for Policy {
    /// Cold-start routing: pick the first task-usable candidate in preference
    /// order, served at [`Level::Balanced`]. Placement-agnostic (memory and the
    /// running set are decided later by [`Policy::select`]); confidence is fixed
    /// low so a learned policy knows this is an un-personalized rule guess.
    fn route(&self, req: &Request, _user: &User, slo: &Slo, registry: &Registry) -> Route {
        // Learned head mounted: score arms by `x^T W p` minus the SLO penalty and
        // return its pick. Falls through to the rule-based walk only if the head is
        // empty, so a mounted head cannot silently degrade to registry order.
        if let Some(heads) = &self.learned {
            let x = HashFeaturizer::default().featurize(req);
            if let Some((model, confidence)) = heads.best(&x, slo) {
                return Route {
                    model,
                    level: Level::Balanced,
                    modality: req.target_modality(),
                    confidence,
                };
            }
        }
        let task = Heuristic.classify(req);
        let running = BTreeSet::new();
        let ctx = Context {
            task,
            registry,
            mem: MemSnapshot {
                available_bytes: u64::MAX,
                total_bytes: u64::MAX,
                unified: true,
            },
            running: &running,
            vision_required: req.has_media,
            min_context: req.approx_tokens,
        };
        let mut policy = self.clone();
        if slo.max_cost > 0.0 {
            policy.cost_ceiling = Some(slo.max_cost as f64);
        }
        let model = policy
            .candidates(&ctx)
            .into_iter()
            .find(|id| registry.get(id).is_some_and(|m| policy.usable(m, &ctx)));
        match model {
            Some(model) => Route {
                model,
                level: Level::Balanced,
                modality: req.target_modality(),
                confidence: COLD_START_CONFIDENCE,
            },
            None => Route::refused(0.0),
        }
    }
}

#[cfg(test)]
mod prefer_tests {
    use super::*;
    use crate::registry::{Backend, ModelCard, Task};
    use crate::route::{RoutePolicy, Slo, User};

    fn pool() -> Registry {
        Registry::new(
            ["opus-4.8", "gpt-5.5", "deepseek-v4-pro", "fable-5"]
                .iter()
                .map(|id| ModelCard {
                    id: (*id).into(),
                    backend: Backend::Cloud {
                        provider: "gateway".into(),
                    },
                    tasks: vec![Task::General],
                    max_context: 0,
                    vision: false,
                    cost_per_1k: 0.0,
                })
                .collect(),
        )
    }

    #[test]
    fn prefer_routes_per_bucket() {
        let p = prefer();
        let reg = pool();
        let route = |text: &str| {
            p.route(
                &Request {
                    text: text.into(),
                    approx_tokens: 200,
                    ..Default::default()
                },
                &User::anonymous(),
                &Slo::default(),
                &reg,
            )
            .model
        };
        // code bucket -> gpt-5.5 first; reasoning -> opus-4.8 first.
        assert_eq!(route("fix this ```rust``` bug"), "gpt-5.5");
        assert_eq!(route("analyze the trade-off step by step"), "opus-4.8");
        // default Policy has no prefer table -> falls to registry order (opus first).
        assert_eq!(
            Policy::default()
                .route(
                    &Request {
                        text: "fix this ```rust``` bug".into(),
                        approx_tokens: 200,
                        ..Default::default()
                    },
                    &User::anonymous(),
                    &Slo::default(),
                    &reg,
                )
                .model,
            "opus-4.8"
        );
    }

    #[test]
    fn learned_head_overrides_rule_based_route() {
        use crate::featurize::{FEAT_DIM, NUM_TASKS};
        use crate::heads::{Arm, Heads};
        let g = Task::General.index();
        let k = NUM_TASKS + 4;
        let mut w = vec![0.0; FEAT_DIM * k]; // utility reads x[General] * p[General]
        w[g * k + g] = 1.0;
        let mkarm = |m: &str, q: f64| {
            let mut f = vec![0.0; k];
            f[g] = q;
            Arm {
                model: m.into(),
                feat: f,
            }
        };
        // rule-based `prefer` would send General to opus-4.8; the head must win.
        let policy = Policy::with_heads(Heads::new(
            w,
            vec![mkarm("weak", 0.2), mkarm("strong", 0.9)],
        ));
        let reg = pool();
        let r = policy.route(
            &Request {
                text: "overview".into(),
                approx_tokens: 200,
                task_hint: Some(Task::General),
                ..Default::default()
            },
            &User::anonymous(),
            &Slo {
                lambda_cost: 0.0,
                mu_latency: 0.0,
                ..Slo::default()
            },
            &reg,
        );
        assert_eq!(
            r.model, "strong",
            "mounted head routes by learned quality, not the prefer table"
        );
    }
}
