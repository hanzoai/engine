//! The routing brain: given a classified task, the pool, what's already running,
//! and a memory snapshot, decide where to serve — preferring a model already
//! loaded (free), else a local model that *fits* available memory, else cloud.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::classify::{Classifier, Heuristic, Request};
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

impl RoutePolicy for Policy {
    /// Cold-start routing: pick the first task-usable candidate in preference
    /// order, served at [`Level::Balanced`]. Placement-agnostic (memory and the
    /// running set are decided later by [`Policy::select`]); confidence is fixed
    /// low so a learned policy knows this is an un-personalized rule guess.
    fn route(&self, req: &Request, _user: &User, slo: &Slo, registry: &Registry) -> Route {
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
