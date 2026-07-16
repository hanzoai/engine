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
    /// Which policy produced the pick: "engine" when the learned selector asserted a
    /// concrete arm, "fallback" when enso delegated to the rule policy (cold start /
    /// no feasible arm). Lets a caller (and the deploy gate) confirm a non-heuristic,
    /// learned decision without inferring it from an empty model.
    pub source: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<f64>,
}

impl IntoResponse for CloudRouteResponse {
    fn into_response(self) -> axum::response::Response {
        Json(self).into_response()
    }
}

const ENSO_GAMMA: f64 = 1.0;
const ENSO_ALPHA: f64 = 0.5;
/// Bound on the number of per-org policies held live at once (LRU).
const SCOPE_CACHE_CAP: usize = 128;

/// A live per-scope policy behind interior mutability: `/v1/route` read-locks it to
/// decide; `/v1/route/observe` write-locks it to fold a reward into per-user LinUCB.
pub type SharedScope = Arc<std::sync::RwLock<enso::Enso>>;

/// The scope-aware serving registry — the axum Extension the route plane consults.
///
/// Weights are PER SCOPE: a global `heads-base.safetensors` plus optional per-org
/// `heads-<org>.safetensors` in the same artifact dir (identical format). A request
/// carries its org; the registry resolves that org's policy with fallback org → base
/// → the rule policy (enso's own cold-start fallback). Per-org policies are lazily
/// loaded on first use and held in a bounded LRU; the always-present base is the
/// fallback and needs no artifact.
///
/// Online learning is in-process: `observe` write-locks the resolved scope and drifts
/// per-user `theta`. The registry periodically flushes each live scope's full state
/// (W + arms + per-user bandits) to `state-<scope>.safetensors` in the writable state
/// dir, so a restart resumes learning. Offline `heads-<scope>` artifacts stay
/// read-only; a newer `heads` (a retrain) supersedes stale online state on reload.
pub struct EnsoRegistry {
    dir: std::path::PathBuf,       // read: heads-<scope>.safetensors
    state_dir: std::path::PathBuf, // write: state-<scope>.safetensors
    base: SharedScope,
    cache: std::sync::Mutex<ScopeCache>,
}

#[derive(Default)]
struct ScopeCache {
    map: std::collections::HashMap<String, ScopeEntry>,
    order: Vec<String>, // recency; last = most-recently-used
}

enum ScopeEntry {
    /// The org has no per-org artifact — it uses the base (cached so we don't re-stat).
    Base,
    /// The org's own loaded policy.
    Own(SharedScope),
}

fn cold_scope() -> SharedScope {
    Arc::new(std::sync::RwLock::new(enso::Enso::new(
        Default::default(),
        enso::Learner::new(enso::Policy::zeros(), ENSO_GAMMA, ENSO_ALPHA),
    )))
}

impl EnsoRegistry {
    fn artifact_path(&self, scope: &str) -> std::path::PathBuf {
        self.dir.join(format!("heads-{scope}.safetensors"))
    }
    fn state_path(&self, scope: &str) -> std::path::PathBuf {
        self.state_dir.join(format!("state-{scope}.safetensors"))
    }

    /// Load a scope's live policy, preferring whichever of `heads-<scope>` /
    /// `state-<scope>` is NEWER (a retrained `heads` supersedes stale online state;
    /// otherwise the flushed state resumes per-user bandits). `None` when neither
    /// exists.
    fn load_scope(&self, scope: &str) -> Option<enso::Enso> {
        let heads = self.artifact_path(scope);
        let state = self.state_path(scope);
        let pick = newer_of(&heads, &state)?;
        match enso::persist::load(&pick, ENSO_GAMMA, ENSO_ALPHA) {
            Ok((learner, table)) => Some(enso::Enso::new(table, learner)),
            Err(e) => {
                tracing::warn!("enso scope {scope} load from {} failed: {e}", pick.display());
                None
            }
        }
    }

    pub fn base(&self) -> SharedScope {
        self.base.clone()
    }

    /// Resolve the policy serving `org` (fallback org → base). Empty org → base.
    pub fn for_org(&self, org: &str) -> SharedScope {
        let org = org.trim();
        if org.is_empty() {
            return self.base();
        }
        {
            let mut c = self.cache.lock().unwrap();
            if let Some(entry) = c.map.get(org) {
                let scope = match entry {
                    ScopeEntry::Base => self.base(),
                    ScopeEntry::Own(a) => a.clone(),
                };
                c.touch(org);
                return scope;
            }
        }
        // Miss: load off-lock, then insert.
        let loaded = self.load_scope(org);
        let mut c = self.cache.lock().unwrap();
        // A racing loader may have inserted already.
        if let Some(entry) = c.map.get(org) {
            return match entry {
                ScopeEntry::Base => self.base(),
                ScopeEntry::Own(a) => a.clone(),
            };
        }
        let (entry, scope) = match loaded {
            Some(enso) => {
                let a: SharedScope = Arc::new(std::sync::RwLock::new(enso));
                (ScopeEntry::Own(a.clone()), a)
            }
            None => (ScopeEntry::Base, self.base()),
        };
        c.insert(org.to_string(), entry, SCOPE_CACHE_CAP, |k, e| {
            if let ScopeEntry::Own(a) = e {
                self.flush_scope(k, &a);
            }
        });
        scope
    }

    /// Flush one scope's full state (W + arms + per-user bandits) to `state-<scope>`.
    fn flush_scope(&self, scope: &str, s: &SharedScope) {
        let g = match s.read() {
            Ok(g) => g,
            Err(_) => return,
        };
        let path = self.state_path(scope);
        if let Err(e) = enso::persist::save(&path, g.learner(), g.table()) {
            tracing::warn!("enso flush scope {scope} to {} failed: {e}", path.display());
        }
    }

    /// Reload the base policy from disk and evict the per-org cache, so a freshly
    /// published `heads-<scope>.safetensors` (the retrain pipeline's artifact) takes
    /// effect WITHOUT a restart. Per-org scopes lazily reload on next use; their live
    /// online state is flushed first so nothing is lost. Returns whether the base was
    /// reloaded from a file. This is the real reload path the retrain script's
    /// DO_RELOAD hook calls after it publishes a new artifact to the mounted dir.
    pub fn reload(&self) -> bool {
        let base_heads = self.artifact_path("base");
        let base_state = self.state_path("base");
        let mut reloaded = false;
        if let Some(pick) = newer_of(&base_heads, &base_state) {
            match enso::persist::load(&pick, ENSO_GAMMA, ENSO_ALPHA) {
                Ok((learner, table)) => {
                    if let Ok(mut g) = self.base.write() {
                        *g = enso::Enso::new(table, learner);
                        reloaded = true;
                        tracing::info!("enso base reloaded from {}", pick.display());
                    }
                }
                Err(e) => tracing::warn!("enso reload from {} failed: {e}", pick.display()),
            }
        }
        // Evict per-org scopes (flushing their online state) so each reloads lazily.
        if let Ok(mut c) = self.cache.lock() {
            for (k, e) in c.map.drain() {
                if let ScopeEntry::Own(a) = e {
                    self.flush_scope(&k, &a);
                }
            }
            c.order.clear();
        }
        reloaded
    }

    /// Flush the base and every live per-org scope — the periodic online-state save.
    pub fn flush_all(&self) {
        self.flush_scope("base", &self.base);
        let entries: Vec<(String, SharedScope)> = {
            let c = self.cache.lock().unwrap();
            c.map
                .iter()
                .filter_map(|(k, e)| match e {
                    ScopeEntry::Own(a) => Some((k.clone(), a.clone())),
                    ScopeEntry::Base => None,
                })
                .collect()
        };
        for (scope, a) in entries {
            self.flush_scope(&scope, &a);
        }
    }

    /// Spawn the periodic online-state flush loop (every `secs`). A no-op when there is
    /// no tokio runtime (unit tests, non-async construction). Called once from
    /// `init_router`.
    pub fn spawn_flush(self: &Arc<Self>, secs: u64) {
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let this = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(secs.max(5)));
            tick.tick().await; // consume the immediate first tick
            loop {
                tick.tick().await;
                this.flush_all();
            }
        });
    }
}

impl ScopeCache {
    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos);
            self.order.push(k);
        }
    }
    fn insert(
        &mut self,
        key: String,
        entry: ScopeEntry,
        cap: usize,
        on_evict: impl Fn(&str, ScopeEntry),
    ) {
        while self.order.len() >= cap {
            let victim = self.order.remove(0);
            if let Some(e) = self.map.remove(&victim) {
                on_evict(&victim, e);
            }
        }
        self.map.insert(key.clone(), entry);
        self.order.push(key);
    }
}

/// The newer of two candidate files by mtime; `None` when neither exists.
fn newer_of(a: &std::path::Path, b: &std::path::Path) -> Option<std::path::PathBuf> {
    let ma = std::fs::metadata(a).ok().and_then(|m| m.modified().ok());
    let mb = std::fs::metadata(b).ok().and_then(|m| m.modified().ok());
    match (ma, mb) {
        (Some(ta), Some(tb)) => Some(if ta >= tb { a.into() } else { b.into() }),
        (Some(_), None) => Some(a.into()),
        (None, Some(_)) => Some(b.into()),
        (None, None) => None,
    }
}

/// The Extension the route plane holds.
pub type SharedEnso = Arc<EnsoRegistry>;

/// A cold-start registry: a zeroed base (rule-policy fallback) and no artifact dir.
/// Correct day-1 with zero infrastructure; `/v1/route` delegates to the rule policy
/// until a `heads-base.safetensors` is provided.
pub fn cold_start_enso() -> SharedEnso {
    Arc::new(EnsoRegistry {
        dir: std::path::PathBuf::from("."),
        state_dir: enso_state_dir(std::path::Path::new(".")),
        base: cold_scope(),
        cache: Default::default(),
    })
}

/// Build the scope registry rooted at `path`'s directory, loading the global base
/// from `path` (a `heads-base.safetensors`). Per-org `heads-<org>.safetensors` in the
/// same dir load lazily on first use. Any base-load error degrades to cold start
/// (rule policy) rather than failing startup — a stale/absent weight file is never an
/// outage. The writable online-state dir is `$ENSO_STATE_DIR` or the artifact dir.
pub fn enso_from_path(path: &std::path::Path) -> SharedEnso {
    let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| ".".into());
    let state_dir = enso_state_dir(&dir);
    // Prefer a flushed base state over the offline base artifact when it is newer.
    let base_state = state_dir.join("state-base.safetensors");
    let pick = newer_of(path, &base_state).unwrap_or_else(|| path.to_path_buf());
    let base = match enso::persist::load(&pick, ENSO_GAMMA, ENSO_ALPHA) {
        Ok((learner, table)) => Arc::new(std::sync::RwLock::new(enso::Enso::new(table, learner))),
        Err(e) => {
            tracing::warn!("enso base load failed at {}: {e}; serving cold start", pick.display());
            cold_scope()
        }
    };
    Arc::new(EnsoRegistry {
        dir,
        state_dir,
        base,
        cache: Default::default(),
    })
}

/// The writable dir for online-state flushes: `$ENSO_STATE_DIR` if set, else the
/// artifact dir (works when that mount is writable; a read-only artifact mount should
/// set `$ENSO_STATE_DIR` to a writable volume).
fn enso_state_dir(artifact_dir: &std::path::Path) -> std::path::PathBuf {
    std::env::var_os("ENSO_STATE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| artifact_dir.to_path_buf())
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
        // Cloud (enso) path: resolve this org's per-scope policy (org → base).
        let scope = enso.for_org(&body.org);
        return cloud_route(&scope, body).into_response();
    }
    placement_route(body).into_response()
}

/// `POST /v1/route/observe` body — the online reward callback the ai gateway posts
/// after it joins a per-request outcome. Carries the request's feature vector `x`
/// (the exact vector the engine returned at decision time), the arm that served, the
/// caller (keys the per-user LinUCB bandit), the reward in [0,1], and the caller org
/// (selects the scope to update). No prompt text.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct ObserveBody {
    #[serde(default)]
    pub user: String,
    #[serde(default)]
    pub org: String,
    #[serde(default)]
    pub features: Vec<f64>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub reward: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ObserveResponse {
    /// Whether the arm was known to the scope's pool (an unknown arm is a no-op).
    pub observed: bool,
}

impl IntoResponse for ObserveResponse {
    fn into_response(self) -> axum::response::Response {
        Json(self).into_response()
    }
}

/// `POST /v1/route/observe` — fold a joined reward into per-user LinUCB `theta` in the
/// caller's scope. Write-locks only that scope briefly; the update is the fast online
/// loop that drifts a user's policy toward their realized quality. Malformed/empty
/// inputs are a no-op (`observed=false`), never an error — a bad reward callback must
/// never disturb serving.
pub async fn route_observe_handler(
    axum::Extension(enso): axum::Extension<SharedEnso>,
    Json(body): Json<ObserveBody>,
) -> axum::response::Response {
    if body.features.is_empty() || body.model.is_empty() || body.user.is_empty() {
        return ObserveResponse { observed: false }.into_response();
    }
    let scope = enso.for_org(&body.org);
    let observed = match scope.write() {
        Ok(mut g) => g.observe_features(&body.user, &body.features, &body.model, body.reward),
        Err(_) => false,
    };
    ObserveResponse { observed }.into_response()
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
fn cloud_route(scope: &SharedScope, body: RouteBody) -> CloudRouteResponse {
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
    // Enso cloud routing maximizes measured QUALITY subject to the request's HARD
    // ceilings (max_latency_ms / max_cost / min_quality). The soft cost/latency trade
    // is zeroed: the gateway sends no soft weights, and folding perf into the utility
    // would demote the measured-best arm (e.g. a slower-but-more-accurate model) — the
    // opposite of warm-start intent. Cost/latency are opt-in hard ceilings.
    let slo = hanzo_router::Slo {
        lambda_cost: 0.0,
        mu_latency: 0.0,
        ..body.slo
    };
    let g = match scope.read() {
        Ok(g) => g,
        Err(_) => {
            return CloudRouteResponse {
                model: String::new(),
                task: String::new(),
                confidence: 0.0,
                source: "fallback".into(),
                features: Vec::new(),
            }
        }
    };
    let (route, explain) = g.route_explained(&req, &user, &slo, &registry);
    // The request feature vector drives the routing ledger's training join —
    // never prompt text. Same featurizer enso scored against.
    let features = g.featurizer().featurize(&req);
    // The engine only asserts a model when it has real learned confidence in
    // one. At cold start (no profiles), on guard refusal, or when no feasible
    // arm survives the SLO gate, enso delegates to the rule policy and we emit
    // an empty model + the task label — the gateway then picks via its own
    // `ForTask`. Returning `refused` or a rule guess here would leak an internal
    // sentinel or assert a pick the engine isn't confident in.
    let learned = !route.is_refused() && !explain.used_fallback;
    let model = if learned { route.model } else { String::new() };
    let source = if learned { "engine" } else { "fallback" };
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
        source: source.to_string(),
        features,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ReloadResponse {
    pub reloaded: bool,
}

impl IntoResponse for ReloadResponse {
    fn into_response(self) -> axum::response::Response {
        Json(self).into_response()
    }
}

/// `POST /v1/route/reload` — reload the base policy from the mounted artifact dir and
/// evict per-org scopes, so a newly published `heads-<scope>.safetensors` is served
/// without a restart. The retrain pipeline calls this after publishing.
pub async fn route_reload_handler(
    axum::Extension(enso): axum::Extension<SharedEnso>,
) -> axum::response::Response {
    ReloadResponse { reloaded: enso.reload() }.into_response()
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
        let res = cloud_route(&enso.for_org(""), cloud_body("fix this ```rust``` bug", 100, false, &["zen4", "zen5"]));
        assert!(res.model.is_empty(), "cold start must not assert a model, got {:?}", res.model);
        assert_eq!(res.source, "fallback", "cold start is a rule-policy fallback");
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
        let base = enso.for_org("");
        let long = cloud_route(&base, cloud_body("summarize", 100_000, false, &["zen4"]));
        assert_eq!(res_task(&long), "long_context", "got {:?}", res_task(&long));

        let cheap = cloud_route(&base, cloud_body("hi", 2, false, &["zen4-mini"]));
        assert_eq!(res_task(&cheap), "cheap_chat", "got {:?}", res_task(&cheap));

        let code = cloud_route(&base, cloud_body("refactor this ```go``` function", 50, false, &["zen4-coder"]));
        assert_eq!(res_task(&code), "code");
    }

    fn res_task(r: &CloudRouteResponse) -> &str {
        &r.task
    }
}
