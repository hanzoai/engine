//! `/v1/route` — the model-routing endpoint.
//!
//! Exposes [`hanzo_router`] (the memory- and task-aware router that `hanzo-node` uses to pick a
//! model across local + cloud engines) as a first-class HTTP surface. A client sends the request
//! shape it's about to run — the prompt, its rough size, whether it carries media, and the set of
//! models currently available — and gets back a [`Decision`](hanzo_router::Decision): reuse a
//! loaded model, load a local one that fits memory, fall out to a cloud provider, or `no_fit`.
//!
//! The routing seam made callable: one POST, a pure function of its inputs, no side effects. The
//! engine stays orthogonal to the policy — the caller owns the model set and the memory snapshot;
//! the router owns the decision.

use axum::{
    extract::{Json, State},
    response::IntoResponse,
};
use enso::{Featurizer, HashFeaturizer};
use hanzo_router::{
    route, Backend, Decision, Heuristic, MemSnapshot, ModelCard, Policy, Registry,
    Request as RouteRequest, RoutePolicy, Slo, Task, User,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use utoipa::ToSchema;

use crate::types::{ExtractedState, SharedState};

/// A model the caller is willing to route to. Mirrors the fields
/// [`hanzo_router::ModelCard`] needs; deserialized straight into it.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct RouteModel {
    /// Canonical model id (e.g. `qwen3.6-35b-a3b`, `claude-sonnet-5`).
    pub id: String,
    /// Max context window (tokens); 0 = unknown/unbounded.
    #[serde(default)]
    pub max_context: usize,
    /// Whether the model accepts image/video input.
    #[serde(default)]
    pub vision: bool,
    /// Whether this model is a cloud provider (vs a local engine).
    #[serde(default)]
    pub cloud: bool,
    /// For cloud models, the provider id (e.g. `anthropic`).
    #[serde(default)]
    pub provider: Option<String>,
    /// For local models, the resident footprint in bytes (quantized weights + overhead) the
    /// router fits against available memory. 0 = unknown (treated as always-fits).
    #[serde(default)]
    pub est_bytes: u64,
    /// Cloud/serving cost per 1k tokens (0 = free/local).
    #[serde(default)]
    pub cost_per_1k: f64,
}

/// `POST /v1/route` body: the request to place + the models available for it.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct RouteBody {
    /// The prompt text (used for task classification: code / vision / long-context / chat).
    #[serde(default)]
    pub prompt: String,
    /// Rough token count of the full request; drives the long-context decision.
    #[serde(default)]
    pub approx_tokens: usize,
    /// Whether the request carries image/video input (forces a vision-capable model).
    #[serde(default)]
    pub has_media: bool,
    /// Models the caller can route to. Ids already loaded/running should set `running: true`.
    #[serde(default)]
    pub models: Vec<RouteModel>,
    /// Subset of `models[].id` that are already loaded (zero load cost — preferred).
    #[serde(default)]
    pub running: Vec<String>,
    /// Available device memory in bytes (0 = treat as unconstrained).
    #[serde(default)]
    pub available_bytes: u64,
    /// Total device memory in bytes.
    #[serde(default)]
    pub total_bytes: u64,
    /// Whether device memory is unified (APU/Metal) — affects the fit budget.
    #[serde(default)]
    pub unified: bool,
}

/// `POST /v1/route` response: the routing decision, tagged by `route`.
#[derive(Debug, Clone, Serialize)]
pub struct RouteResponse {
    /// The router's decision — `reuse` / `load_local` / `cloud` / `no_fit`.
    pub decision: Decision,
}

impl IntoResponse for RouteResponse {
    fn into_response(self) -> axum::response::Response {
        Json(self).into_response()
    }
}

/// Build a [`hanzo_router::ModelCard`] from the wire shape. Kept here (not `From`) so the wire type
/// stays a pure DTO and the router crate has no HTTP knowledge.
fn to_card(m: &RouteModel) -> hanzo_router::ModelCard {
    hanzo_router::ModelCard {
        id: m.id.clone(),
        backend: if m.cloud {
            hanzo_router::Backend::Cloud {
                provider: m.provider.clone().unwrap_or_else(|| "cloud".into()),
            }
        } else {
            hanzo_router::Backend::Local {
                est_bytes: if m.est_bytes == 0 { 1 } else { m.est_bytes },
            }
        },
        tasks: Vec::new(),
        max_context: m.max_context,
        vision: m.vision,
        cost_per_1k: m.cost_per_1k,
    }
}

/// The routing decision for a request. Pure over its inputs — no engine state is mutated.
#[cfg_attr(
    feature = "utoipa",
    utoipa::path(
        post,
        path = "/v1/route",
        request_body = RouteBody,
        responses((status = 200, description = "Routing decision", body = RouteResponse))
    )
)]
pub async fn route_handler(Json(body): Json<RouteBody>) -> impl IntoResponse {
    let registry = Registry::new(body.models.iter().map(to_card).collect());
    let running: BTreeSet<String> = body.running.into_iter().collect();
    let mem = MemSnapshot {
        available_bytes: if body.available_bytes == 0 { u64::MAX } else { body.available_bytes },
        total_bytes: if body.total_bytes == 0 { u64::MAX } else { body.total_bytes },
        unified: body.unified,
    };
    let req = RouteRequest {
        text: body.prompt,
        approx_tokens: body.approx_tokens,
        has_media: body.has_media,
        task_hint: None,
        modality_hint: None,
    };
    let decision = route(&req, &Heuristic, &Policy::default(), &registry, mem, &running);
    RouteResponse { decision }
}

// ---------------------------------------------------------------------------
// `/route` — the production routing contract consumed by hanzoai/ai cloud-api.
// ---------------------------------------------------------------------------
//
// Distinct from `/v1/route` above: that surface takes the caller's full model
// set + memory snapshot and returns a *placement* decision. This one is the
// learned-router-as-a-service that the cloud-api's `router.Client` calls — it
// owns the policy, classifies the prompt, picks from the models the engine
// actually has, and returns a feature vector for offline router training. It
// works on the pure-Rust path with no model loaded (returns task + features and
// the caller maps the task), and picks a concrete model when the engine has one.

/// `POST /route` request body — the wire contract in
/// `hanzoai/ai/router/client.go`. Only `prompt` is required.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct ClassifyBody {
    /// The prompt to route (the last user turn is enough for classification).
    #[serde(default)]
    pub prompt: String,
    /// Optional explicit task labels; the first recognized one (snake_case, e.g.
    /// `code`, `long_context`) overrides heuristic classification.
    #[serde(default)]
    pub tasks: Vec<String>,
    /// Optional per-request service-level objective (the operator's budget).
    #[serde(default)]
    pub slo: Option<ClassifySlo>,
}

/// The operator's per-request budget. `0`/absent disables a ceiling.
#[derive(Debug, Clone, Copy, Default, Deserialize, ToSchema)]
pub struct ClassifySlo {
    /// Max cloud cost per 1k tokens; filters models above it.
    #[serde(default)]
    pub max_cost: f64,
    /// Max acceptable latency (ms); carried through for the learned selector.
    #[serde(default)]
    pub max_latency_ms: f64,
}

/// `POST /route` response — exactly the four fields the Go client decodes.
/// `features` is the stable per-prompt vector (enso `HashFeaturizer`,
/// [`FEAT_DIM`](enso::featurize::FEAT_DIM) dims) collected for router training;
/// it is feature values only, never prompt text.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ClassifyResponse {
    /// Chosen model id, or empty when the engine has no servable model (the
    /// caller then maps `task` via its own policy table).
    pub model: String,
    /// The task the prompt was classified as (snake_case).
    pub task: String,
    /// Routing confidence in `[0, 1]`.
    pub confidence: f64,
    /// Stable per-prompt feature vector; omitted only if empty (it never is).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<f64>,
}

impl IntoResponse for ClassifyResponse {
    fn into_response(self) -> axum::response::Response {
        Json(self).into_response()
    }
}

/// A [`Task`]'s snake_case wire label (its serde name) — the single source of
/// the label strings shared with the Go client.
fn task_label(task: Task) -> String {
    serde_json::to_value(task)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

/// Parse a snake_case task label into a [`Task`] via the same serde mapping, so
/// there is no second label table to drift.
fn parse_task(label: &str) -> Option<Task> {
    serde_json::from_value(serde_json::Value::String(label.to_string())).ok()
}

/// Rough token estimate from raw prompt length (~4 chars/token). Drives the
/// long-context bucket; exactness doesn't matter to the decision.
fn approx_tokens(prompt: &str) -> usize {
    prompt.len() / 4
}

/// The `/route` decision as a pure function of its value inputs: prompt, an
/// optional caller task override, the SLO, and the pool of models the engine can
/// serve. Returns exactly the wire fields. Separated from the axum handler so it
/// is unit-testable with no engine instance, and allocation-light (one
/// featurize, one policy walk).
fn classify_route(
    prompt: &str,
    task_hint: Option<Task>,
    slo: Slo,
    registry: &Registry,
) -> (String, Task, f64, Vec<f64>) {
    let req = RouteRequest {
        text: prompt.to_string(),
        approx_tokens: approx_tokens(prompt),
        has_media: false,
        task_hint,
        modality_hint: None,
    };
    // Features + task from the enso Featurizer seam (single source of task truth).
    let featurizer = HashFeaturizer::default();
    let features = featurizer.featurize(&req);
    let task = featurizer.task_of(&req);
    // Model: the rule-based cold-start policy over the available pool, honoring
    // the SLO cost ceiling. Empty/refused -> no explicit model, caller maps task.
    let route = Policy::default().route(&req, &User::anonymous(), &slo, registry);
    let model = if route.is_refused() {
        String::new()
    } else {
        route.model
    };
    (model, task, route.confidence as f64, features)
}

/// Build the routing pool from the models the engine currently has. Loaded
/// engine models are general-purpose chat models with resident weights, so each
/// is a local, always-fits `General` card; the caller's policy owns any finer
/// task->model mapping. Empty when nothing is loaded (pure classification mode).
fn available_registry(state: &SharedState) -> Registry {
    let models = state
        .list_models_with_status()
        .unwrap_or_default()
        .into_iter()
        .filter(|(id, _)| id != "default")
        .map(|(id, _status)| ModelCard {
            id,
            backend: Backend::Local { est_bytes: 1 },
            tasks: vec![Task::General],
            max_context: 0,
            vision: false,
            cost_per_1k: 0.0,
        })
        .collect();
    Registry::new(models)
}

/// `POST /route` — classify a prompt, pick a servable model honoring the SLO, and
/// return the task + routing confidence + a stable feature vector for training.
#[cfg_attr(
    feature = "utoipa",
    utoipa::path(
        post,
        path = "/route",
        request_body = ClassifyBody,
        responses((status = 200, description = "Routing decision", body = ClassifyResponse))
    )
)]
pub async fn classify_handler(
    State(state): ExtractedState,
    Json(body): Json<ClassifyBody>,
) -> impl IntoResponse {
    let slo = body
        .slo
        .map(|s| Slo {
            max_cost: s.max_cost as f32,
            max_latency_ms: s.max_latency_ms as f32,
            ..Slo::default()
        })
        .unwrap_or_default();
    let task_hint = body.tasks.iter().find_map(|t| parse_task(t));
    let registry = available_registry(&state);
    let (model, task, confidence, features) =
        classify_route(&body.prompt, task_hint, slo, &registry);
    ClassifyResponse {
        model,
        task: task_label(task),
        confidence,
        features,
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use enso::featurize::FEAT_DIM;
    use std::time::Instant;

    fn cloud(id: &str, cost: f64) -> ModelCard {
        ModelCard {
            id: id.into(),
            backend: Backend::Cloud { provider: "test".into() },
            tasks: vec![Task::General],
            max_context: 0,
            vision: false,
            cost_per_1k: cost,
        }
    }

    #[test]
    fn body_deserializes_with_optional_fields_absent() {
        // Only `prompt` present — tasks + slo must default.
        let b: ClassifyBody = serde_json::from_str(r#"{"prompt":"hi"}"#).unwrap();
        assert_eq!(b.prompt, "hi");
        assert!(b.tasks.is_empty());
        assert!(b.slo.is_none());

        // Full shape round-trips.
        let full: ClassifyBody = serde_json::from_str(
            r#"{"prompt":"p","tasks":["code"],"slo":{"max_cost":2.0,"max_latency_ms":150}}"#,
        )
        .unwrap();
        assert_eq!(full.tasks, vec!["code".to_string()]);
        assert_eq!(full.slo.unwrap().max_cost, 2.0);
    }

    #[test]
    fn response_serializes_exact_contract_fields() {
        let resp = ClassifyResponse {
            model: "m".into(),
            task: "code".into(),
            confidence: 0.25,
            features: vec![0.0, 1.0, 2.0],
        };
        let v = serde_json::to_value(&resp).unwrap();
        let obj = v.as_object().unwrap();
        // Exactly the four contract keys, named exactly.
        assert_eq!(obj.len(), 4);
        assert!(obj.contains_key("model"));
        assert!(obj.contains_key("task"));
        assert!(obj.contains_key("confidence"));
        assert!(obj.contains_key("features"));
    }

    #[test]
    fn classifies_and_returns_all_fields() {
        let reg = Registry::new(vec![cloud("router-model", 0.5)]);
        let (model, task, confidence, features) =
            classify_route("fix this ```rust``` compile bug", None, Slo::default(), &reg);
        assert_eq!(task, Task::Code);
        assert_eq!(task_label(task), "code");
        assert_eq!(model, "router-model");
        assert_eq!(confidence, 0.25); // cold-start rule confidence
        assert_eq!(features.len(), FEAT_DIM);
    }

    #[test]
    fn task_hint_overrides_classification() {
        let reg = Registry::new(vec![cloud("m", 0.5)]);
        // Prompt looks like code, but caller asserts `math`.
        let (_m, task, _c, _f) =
            classify_route("```rust```", Some(Task::Math), Slo::default(), &reg);
        assert_eq!(task, Task::Math);
    }

    #[test]
    fn slo_max_cost_is_honored() {
        // Pricey model is first in preference order; the ceiling must skip it.
        let reg = Registry::new(vec![cloud("pricey", 5.0), cloud("cheap", 0.5)]);
        let prompt = "please recommend a nice restaurant for dinner tonight downtown near me";

        // No ceiling -> first candidate wins.
        let (m, _t, _c, _f) = classify_route(prompt, None, Slo::default(), &reg);
        assert_eq!(m, "pricey");

        // Ceiling below pricey -> pricey filtered, cheap wins.
        let slo = Slo { max_cost: 1.0, ..Slo::default() };
        let (m, _t, _c, _f) = classify_route(prompt, None, slo, &reg);
        assert_eq!(m, "cheap");
    }

    #[test]
    fn empty_pool_yields_task_only() {
        // No servable model -> empty model id (caller maps the task), but the
        // task + features are still produced (pure classification mode).
        let (model, task, _c, features) =
            classify_route("write a function", None, Slo::default(), &Registry::default());
        assert_eq!(model, "");
        assert_eq!(task, Task::Code);
        assert_eq!(features.len(), FEAT_DIM);
    }

    #[test]
    fn features_are_stable_and_correct_dim() {
        let reg = Registry::new(vec![cloud("m", 0.5)]);
        let (_m, _t, _c, a) = classify_route("prove sqrt(2) is irrational", None, Slo::default(), &reg);
        let (_m, _t, _c, b) = classify_route("prove sqrt(2) is irrational", None, Slo::default(), &reg);
        assert_eq!(a.len(), FEAT_DIM);
        assert_eq!(a, b); // deterministic per prompt
    }

    #[test]
    fn latency_within_budget() {
        let reg = Registry::new(vec![cloud("a", 3.0), cloud("b", 0.5)]);
        let prompt = "explain step by step why this stack trace points at a race condition";
        let iters = 20_000;
        // warm
        let _ = classify_route(prompt, None, Slo::default(), &reg);
        let t0 = Instant::now();
        for _ in 0..iters {
            let out = classify_route(prompt, None, Slo::default(), &reg);
            std::hint::black_box(&out);
        }
        let per = t0.elapsed() / iters;
        eprintln!("classify_route end-to-end: {per:?}/call over {iters} iters");
        // Whole client budget is 150ms; the pure decision must be a tiny fraction.
        assert!(per.as_micros() < 1_000, "decision too slow: {per:?}");
    }
}
