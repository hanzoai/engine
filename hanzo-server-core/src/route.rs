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

use axum::extract::Json;
use axum::response::IntoResponse;
use enso::Featurizer;
use hanzo_router::{
    route, Decision, Heuristic, MemSnapshot, Policy, Registry, Request as RouteRequest, User,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::Arc;
use utoipa::ToSchema;

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
    // ── cloud (enso) fields: sent by the ai gateway when it has no local replica ──
    /// Org-servable candidate model ids; enso picks among these. Empty with
    /// `models[]` non-empty selects the placement path.
    #[serde(default)]
    pub pool: Vec<String>,
    /// Verified principal "owner/name" — keys per-user adaptation + the ledger.
    #[serde(default)]
    pub user: String,
    /// Caller org id.
    #[serde(default)]
    pub org: String,
    /// Per-request SLO ceilings + the cost/latency trade the selector applies.
    /// Opaque to the schema (hanzo-router stays utoipa-free); the wire shape is
    /// the gateway's `Slo`.
    #[serde(default)]
    #[schema(value_type = serde_json::Value)]
    pub slo: hanzo_router::Slo,
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

/// `POST /v1/route` cloud response — the flat `{model, task, confidence, features}`
/// shape the ai gateway's router client expects (see ai/router/client.go
/// `engineResponse`). The cloud shape carries no models[] (the gateway sends the
/// servable pool as `pool`), so this branch serves `enso` rather than the
/// memory-placement policy. `features` is the request's hashing-trick feature
/// vector for the routing ledger — never prompt text.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct CloudRouteResponse {
    pub model: String,
    pub task: String,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<f64>,
}

impl IntoResponse for CloudRouteResponse {
    fn into_response(self) -> axum::response::Response {
        Json(self).into_response()
    }
}

/// The cold-start shared `enso` learned policy, injected as an axum Extension so
/// the cloud branch of `/v1/route` consults it. Empty ProfileTable + zeroed W +
/// default learner: enso delegates to the rule policy until eval data is ingested
/// (Phase 4d), so serving is correct day-1 with zero infrastructure. Phase 1
/// adds weight persistence (load/flush) on top.
pub type SharedEnso = Arc<enso::Enso>;

/// The cold-start `enso` weights. Empty `ProfileTable` (no eval data) + a zeroed
/// base `W` + the default LinUCB learner (gamma=1.0, alpha=0.5). With no profiles
/// for any modality `enso` delegates to the router's rule policy (see
/// `enso::Enso::decide`), so the cloud path is correct day-1 with no weights on
/// disk; eval data ingested in Phase 4 replaces this via `persist`.
pub fn cold_start_enso() -> SharedEnso {
    Arc::new(enso::Enso::new(
        Default::default(),
        enso::Learner::new(enso::Policy::zeros(), 1.0, 0.5),
    ))
}

/// Load the base `W` from `path` (a `heads-base.safetensors` written by the Phase
/// 4d retrain job) and wrap it in a serving `enso`. Any error (missing file,
/// corrupt, wrong shape) falls back to `cold_start_enso` rather than failing
/// startup: a stale or absent weight file degrades to the rule policy, never to
/// an outage. Empty ProfileTable is intentional — `enso` uses `W` as the prior
/// and the rule policy as the feasibility fallback.
pub fn enso_from_path(path: &std::path::Path) -> SharedEnso {
    match enso::persist::load_learner(path, 1.0, 0.5) {
        Ok(learner) => Arc::new(enso::Enso::new(Default::default(), learner)),
        Err(e) => {
            tracing::warn!("enso weights load failed at {}: {e}; serving cold start", path.display());
            cold_start_enso()
        }
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

/// The routing decision for a request. Pure over its inputs - no engine state is mutated.
///
/// Branches on body shape: `models[]` non-empty selects the memory-placement
/// path (hanzo-node, unchanged); a cloud body (no `models[]`, `pool` present)
/// selects the `enso` learned-policy path and returns the flat shape the ai
/// gateway expects. One registered route, two contracts, both pure.
pub async fn route_handler(
    axum::Extension(enso): axum::Extension<SharedEnso>,
    Json(body): Json<RouteBody>,
) -> axum::response::Response {
    if body.models.is_empty() {
        return cloud_route(&enso, body).into_response();
    }
    placement_route(body).into_response()
}

/// Memory-placement path for hanzo-node — byte-identical to the pre-branch
/// behavior, so the local-replica contract is preserved.
fn placement_route(body: RouteBody) -> RouteResponse {
    let registry = Registry::new(body.models.iter().map(to_card).collect());
    let running: BTreeSet<String> = body.running.into_iter().collect();
    let mem = MemSnapshot {
        available_bytes: if body.available_bytes == 0 {
            u64::MAX
        } else {
            body.available_bytes
        },
        total_bytes: if body.total_bytes == 0 {
            u64::MAX
        } else {
            body.total_bytes
        },
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

/// Cloud path: the ai gateway sends the org-servable `pool` (no local replica),
/// `enso` picks among them, and we return the flat `{model, task, confidence,
/// features}` the gateway's router client expects. enso falls back to the rule
/// policy until eval data is ingested, so this is correct at cold start.
fn cloud_route(enso: &SharedEnso, body: RouteBody) -> CloudRouteResponse {
    let cards = body.pool.iter().map(|id| pool_card(id)).collect();
    let registry = Registry::new(cards);
    let req = RouteRequest {
        text: body.prompt,
        approx_tokens: body.approx_tokens,
        has_media: body.has_media,
        task_hint: None,
        modality_hint: None,
    };
    let user = if body.user.is_empty() {
        User::anonymous()
    } else {
        User::new(body.user.clone())
    };
    let slo = body.slo;
    let (route, explain) = enso.route_explained(&req, &user, &slo, &registry);
    // The request feature vector drives the routing ledger's training join —
    // never prompt text. Same featurizer enso scored against.
    let features = enso.featurizer().featurize(&req);
    // The engine only asserts a model when it has real learned confidence in
    // one. At cold start (no profiles), on guard refusal, or when no feasible
    // arm survives the SLO gate, enso delegates to the rule policy and we emit
    // an empty model + the task label — the gateway then picks via its own
    // `ForTask`. Returning `refused` or a rule guess here would leak an internal
    // sentinel or assert a pick the engine isn't confident in.
    let model = if route.is_refused() || explain.used_fallback {
        String::new()
    } else {
        route.model
    };
    // The wire task label is the serde (snake_case) name, so it matches the
    // gateway's `TaskLongContext = "long_context"` / `TaskCheapChat = "cheap_chat"`
    // — not `Debug`'s `longcontext`/`cheapchat`. Single source of truth: the enum's
    // `#[serde(rename_all = "snake_case")]`.
    let task = serde_json::to_value(&explain.task)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default();
    CloudRouteResponse {
        model,
        task,
        confidence: route.confidence as f64,
        features,
    }
}

/// A cloud-pool model is always a cloud backend with unknown context/vision
/// (the gateway's `known` gate already filtered by servability + modality).
fn pool_card(id: &str) -> hanzo_router::ModelCard {
    hanzo_router::ModelCard {
        id: id.to_string(),
        backend: hanzo_router::Backend::Cloud {
            provider: "cloud".into(),
        },
        tasks: Vec::new(),
        max_context: 0,
        vision: false,
        cost_per_1k: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud_body(prompt: &str, tokens: usize, media: bool, pool: &[&str]) -> RouteBody {
        RouteBody {
            prompt: prompt.into(),
            approx_tokens: tokens,
            has_media: media,
            models: Vec::new(),
            running: Vec::new(),
            available_bytes: 0,
            total_bytes: 0,
            unified: false,
            pool: pool.iter().map(|s| (*s).into()).collect(),
            user: String::new(),
            org: String::new(),
            slo: hanzo_router::Slo::default(),
        }
    }

    // Cold start: no profiles, so enso delegates to the rule policy. The engine
    // must NOT assert a model here (no learned confidence) — it emits an empty
    // model + the task label, and the gateway picks via ForTask. Asserting
    // `refused` or a rule guess would leak an internal sentinel or a pick the
    // engine isn't confident in.
    #[test]
    fn cold_start_emits_empty_model_not_refused() {
        let enso = cold_start_enso();
        let res = cloud_route(&enso, cloud_body("fix this ```rust``` bug", 100, false, &["zen4", "zen5"]));
        assert!(res.model.is_empty(), "cold start must not assert a model, got {:?}", res.model);
        assert!(!res.task.is_empty(), "cold start still classifies the task");
        assert!(res.features.len() == enso::featurize::FEAT_DIM);
    }

    // The wire task label is the serde snake_case name, so it joins the gateway's
    // TaskLongContext = "long_context" / TaskCheapChat = "cheap_chat" — not
    // Debug's "longcontext"/"cheapchat". A mismatch silently drops those tasks to
    // the heuristic with a wrong label.
    #[test]
    fn task_label_is_snake_case_not_debug() {
        let enso = cold_start_enso();
        let long = cloud_route(&enso, cloud_body("summarize", 100_000, false, &["zen4"]));
        assert_eq!(res_task(&long), "long_context", "got {:?}", res_task(&long));

        let cheap = cloud_route(&enso, cloud_body("hi", 2, false, &["zen4-mini"]));
        assert_eq!(res_task(&cheap), "cheap_chat", "got {:?}", res_task(&cheap));

        let code = cloud_route(&enso, cloud_body("refactor this ```go``` function", 50, false, &["zen4-coder"]));
        assert_eq!(res_task(&code), "code");
    }

    fn res_task(r: &CloudRouteResponse) -> &str {
        &r.task
    }
}
