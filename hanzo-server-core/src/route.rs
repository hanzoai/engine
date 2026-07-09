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

use axum::{extract::Json, response::IntoResponse};
use hanzo_router::{
    route, Decision, Heuristic, MemSnapshot, Policy, Registry, Request as RouteRequest,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
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
    let decision = route(
        &req,
        &Heuristic,
        &Policy::default(),
        &registry,
        mem,
        &running,
    );
    RouteResponse { decision }
}
