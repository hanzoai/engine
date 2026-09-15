//! Replica load-balancing: N engine instances serving the *same* model. Composes
//! the pure [`Ring`] (prefix-affinity + spill order) with per-replica health and
//! in-flight load. HTTP-free and clock-free so it unit-tests without a network;
//! the effectful front (probe loop, request proxy) lives behind the `proxy`
//! feature in [`crate::proxy`] and drives this via [`ReplicaSet::set_health`] and
//! the RAII [`Lease`].
//!
//! Orthogonal to model *selection* ([`crate::policy`]): that picks *which model*
//! serves a request; this picks *which replica of a chosen model* does.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

#[path = "scheduler.rs"]
mod scheduler;
pub use scheduler::RoutingHints;
use scheduler::SchedulerState;

use serde::{Deserialize, Serialize};

use crate::ring::Ring;

/// Default per-replica in-flight ceiling before a request spills to the next ring
/// node. A soft hint, not a hard cap: a fully saturated set still serves (least
/// loaded) rather than dropping the request.
pub const DEFAULT_MAX_INFLIGHT: usize = 64;

/// One upstream engine instance serving a model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Replica {
    /// Stable ring-key member. Empty in config -> normalized to `url`.
    #[serde(default)]
    pub id: String,
    /// Base URL, e.g. `http://127.0.0.1:1234` (no trailing path).
    pub url: String,
    /// Concurrent request slots; zero inherits the pool's max_inflight.
    #[serde(default)]
    pub capacity: usize,
    /// Per-request prompt + generation token ceiling. Zero is unspecified.
    #[serde(default)]
    pub max_context: usize,
    /// Accepts image input; requests carrying images skip replicas without it.
    #[serde(default = "default_vision")]
    pub vision: bool,
    /// Relative measured throughput for allocating new sessions (100 = baseline).
    #[serde(default = "default_weight")]
    pub weight: u32,
    /// Preferred agent roles; preferences never override health or session pins.
    #[serde(default)]
    pub roles: Vec<String>,
    /// Optional backend model ID when replicas use different local aliases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_model: Option<String>,
}

fn default_weight() -> u32 {
    100
}

fn default_vision() -> bool {
    true
}

impl Replica {
    pub fn new(url: impl Into<String>) -> Self {
        let url = url.into();
        Self {
            id: url.clone(),
            url,
            capacity: 0,
            max_context: 0,
            vision: true,
            weight: default_weight(),
            roles: Vec::new(),
            upstream_model: None,
        }
    }

    fn normalized(mut self) -> Self {
        if self.id.is_empty() {
            self.id = self.url.clone();
        }
        self.weight = self.weight.max(1);
        self
    }
}

#[derive(Clone)]
struct WorkerSettings {
    capacity: usize,
    weight: u32,
    roles: Vec<String>,
}

impl From<&Replica> for WorkerSettings {
    fn from(r: &Replica) -> Self {
        Self {
            capacity: r.capacity,
            weight: r.weight,
            roles: r.roles.clone(),
        }
    }
}

struct Node {
    replica: Replica,
    settings: RwLock<WorkerSettings>,
    healthy: AtomicBool,
    inflight: AtomicUsize,
}

struct Inner {
    nodes: HashMap<String, Arc<Node>>,
    ring: Ring,
}

impl Inner {
    fn rebuild_ring(&mut self) {
        let healthy = self
            .nodes
            .values()
            .filter(|n| n.healthy.load(Ordering::Acquire))
            .map(|n| n.replica.id.clone());
        self.ring = Ring::new(healthy);
    }
}

/// The pool of replicas for one model. `pick` composes affinity, spill, and
/// health; the returned [`Lease`] holds the in-flight slot for the request's
/// (possibly streaming) lifetime and releases it on drop.
pub struct ReplicaSet {
    inner: RwLock<Inner>,
    max_inflight: usize,
    scheduler: Mutex<SchedulerState>,
}

impl ReplicaSet {
    pub fn new(replicas: impl IntoIterator<Item = Replica>, max_inflight: usize) -> Self {
        let nodes: HashMap<String, Arc<Node>> = replicas
            .into_iter()
            .map(|r| {
                let r = r.normalized();
                (
                    r.id.clone(),
                    Arc::new(Node {
                        settings: RwLock::new(WorkerSettings::from(&r)),
                        replica: r,
                        healthy: AtomicBool::new(true),
                        inflight: AtomicUsize::new(0),
                    }),
                )
            })
            .collect();
        let mut inner = Inner {
            nodes,
            ring: Ring::default(),
        };
        inner.rebuild_ring();
        Self {
            inner: RwLock::new(inner),
            max_inflight: max_inflight.max(1),
            scheduler: Mutex::new(SchedulerState::default()),
        }
    }

    /// Route `key` to a replica: the affinity primary first, spilling down the
    /// ring to the next replica under [`Self::max_inflight`], and if every healthy
    /// replica is saturated, the least-loaded one (never dropped). `None` only
    /// when no replica is healthy.
    pub fn pick(&self, key: &str) -> Option<Lease> {
        let inner = self.inner.read().unwrap();
        if inner.ring.is_empty() {
            return None;
        }
        let mut least: Option<&Arc<Node>> = None;
        for id in inner.ring.route(key) {
            let node = &inner.nodes[id];
            let load = node.inflight.load(Ordering::Acquire);
            if load < self.max_inflight {
                return Some(Lease::acquire(node.clone()));
            }
            least = match least {
                Some(l) if l.inflight.load(Ordering::Acquire) <= load => Some(l),
                _ => Some(node),
            };
        }
        least.map(|n| Lease::acquire(n.clone()))
    }

    /// Set a replica's health, rebuilding the ring only on a real transition.
    /// Absent id is a no-op.
    pub fn set_health(&self, id: &str, healthy: bool) {
        let mut inner = self.inner.write().unwrap();
        let changed = match inner.nodes.get(id) {
            Some(n) => n.healthy.swap(healthy, Ordering::AcqRel) != healthy,
            None => false,
        };
        if changed {
            self.scheduler.lock().unwrap().invalidate_cache(id);
            inner.rebuild_ring();
        }
    }

    pub fn mark_healthy(&self, id: &str) {
        self.set_health(id, true);
    }

    pub fn mark_unhealthy(&self, id: &str) {
        self.set_health(id, false);
    }

    /// Add a replica (healthy) or leave an existing one untouched.
    pub fn register(&self, replica: Replica) {
        let replica = replica.normalized();
        let mut inner = self.inner.write().unwrap();
        if inner.nodes.contains_key(&replica.id) {
            return;
        }
        inner.nodes.insert(
            replica.id.clone(),
            Arc::new(Node {
                settings: RwLock::new(WorkerSettings::from(&replica)),
                replica,
                healthy: AtomicBool::new(true),
                inflight: AtomicUsize::new(0),
            }),
        );
        inner.rebuild_ring();
    }

    /// Total configured replicas (healthy or not) — the proxy's retry budget.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn statuses(&self) -> Vec<ReplicaStatus> {
        let inner = self.inner.read().unwrap();
        let scheduler = self.scheduler.lock().unwrap();
        let mut out: Vec<ReplicaStatus> = inner
            .nodes
            .values()
            .map(|n| ReplicaStatus {
                id: n.replica.id.clone(),
                url: n.replica.url.clone(),
                healthy: n.healthy.load(Ordering::Acquire),
                inflight: n.inflight.load(Ordering::Acquire),
                capacity: self.slots(n),
                max_context: n.replica.max_context,
                vision: n.replica.vision,
                weight: n.settings.read().unwrap().weight,
                roles: n.settings.read().unwrap().roles.clone(),
                upstream_model: n.replica.upstream_model.clone(),
                ttft_ewma_ms: scheduler.ttft_ms(&n.replica.id),
            })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }
}

/// An in-flight slot on a replica. Increments on [`ReplicaSet::pick`], decrements
/// on drop — so a proxied stream releases its slot exactly when it completes or
/// the client aborts.
pub struct Lease {
    node: Arc<Node>,
}

impl Lease {
    fn acquire(node: Arc<Node>) -> Self {
        node.inflight.fetch_add(1, Ordering::AcqRel);
        Self { node }
    }

    pub fn id(&self) -> &str {
        &self.node.replica.id
    }

    pub fn url(&self) -> &str {
        &self.node.replica.url
    }

    pub fn upstream_model(&self) -> Option<&str> {
        self.node.replica.upstream_model.as_deref()
    }

    pub fn inflight(&self) -> usize {
        self.node.inflight.load(Ordering::Acquire)
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.node.inflight.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A replica's live state, for the `/v1/replicas` admin surface.
#[derive(Clone, Debug, Serialize)]
pub struct ReplicaStatus {
    pub id: String,
    pub url: String,
    pub healthy: bool,
    pub inflight: usize,
    pub capacity: usize,
    pub max_context: usize,
    pub vision: bool,
    pub weight: u32,
    pub roles: Vec<String>,
    pub upstream_model: Option<String>,
    /// Time to first body byte, observed by this router; not decode throughput.
    pub ttft_ewma_ms: Option<f64>,
}

/// Declarative pool: `model -> replicas`, plus the shared in-flight ceiling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BalancerConfig {
    #[serde(default = "default_max_inflight")]
    pub max_inflight: usize,
    #[serde(default)]
    pub models: BTreeMap<String, Vec<Replica>>,
}

impl Default for BalancerConfig {
    fn default() -> Self {
        Self {
            max_inflight: DEFAULT_MAX_INFLIGHT,
            models: BTreeMap::new(),
        }
    }
}

fn default_max_inflight() -> usize {
    DEFAULT_MAX_INFLIGHT
}

/// One [`ReplicaSet`] per model. The proxy front holds a `Balancer` and routes
/// each request to the set for its `model` field.
pub struct Balancer {
    sets: RwLock<HashMap<String, Arc<ReplicaSet>>>,
    max_inflight: usize,
}

impl Balancer {
    pub fn new(max_inflight: usize) -> Self {
        Self {
            sets: RwLock::new(HashMap::new()),
            max_inflight,
        }
    }

    pub fn from_config(cfg: BalancerConfig) -> Self {
        let max_inflight = cfg.max_inflight.max(1);
        let sets = cfg
            .models
            .into_iter()
            .map(|(model, replicas)| (model, Arc::new(ReplicaSet::new(replicas, max_inflight))))
            .collect();
        Self {
            sets: RwLock::new(sets),
            max_inflight,
        }
    }

    /// The set for `model`; falls back to the sole set when exactly one model is
    /// configured (so a client's model id need not match the config key in the
    /// common single-model deployment).
    pub fn set_for(&self, model: Option<&str>) -> Option<Arc<ReplicaSet>> {
        let sets = self.sets.read().unwrap();
        // Claude Code's explicit context suffix is a client capability hint,
        // not a different model or a reason to lose the requested pool.
        let model = model.map(|m| {
            m.strip_suffix("[1m]")
                .or_else(|| m.strip_suffix("[1M]"))
                .unwrap_or(m)
        });
        if let Some(set) = model.and_then(|m| sets.get(m)) {
            return Some(set.clone());
        }
        if sets.len() == 1 {
            return sets.values().next().cloned();
        }
        for fallback in &["zen5.8", "qwen3.8", "zen5.8-coder", "default"] {
            if let Some(set) = sets.get(*fallback) {
                return Some(set.clone());
            }
        }
        None
    }

    /// Register a replica for `model`, creating the set if new (dynamic
    /// announcement via `POST /v1/replicas`).
    pub fn register(&self, model: &str, replica: Replica) {
        let mut sets = self.sets.write().unwrap();
        sets.entry(model.to_string())
            .or_insert_with(|| Arc::new(ReplicaSet::new([], self.max_inflight)))
            .register(replica);
    }

    /// Retune an existing worker without replacing its leases, health, or pins.
    /// Exact model/worker lookup; never uses the inference alias fallback.
    pub fn update_worker(
        &self,
        model: &str,
        id: &str,
        capacity: usize,
        weight: u32,
        roles: Vec<String>,
    ) -> bool {
        if capacity == 0 || weight == 0 {
            return false;
        }
        let sets = self.sets.read().unwrap();
        let Some(set) = sets.get(model) else {
            return false;
        };
        let inner = set.inner.read().unwrap();
        let Some(node) = inner.nodes.get(id) else {
            return false;
        };
        *node.settings.write().unwrap() = WorkerSettings {
            capacity,
            weight,
            roles,
        };
        true
    }

    pub fn models(&self) -> Vec<String> {
        let mut m: Vec<String> = self.sets.read().unwrap().keys().cloned().collect();
        m.sort();
        m
    }

    pub fn statuses(&self) -> BTreeMap<String, Vec<ReplicaStatus>> {
        self.sets
            .read()
            .unwrap()
            .iter()
            .map(|(m, s)| (m.clone(), s.statuses()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set3() -> ReplicaSet {
        ReplicaSet::new(
            [
                Replica::new("http://a"),
                Replica::new("http://b"),
                Replica::new("http://c"),
            ],
            2,
        )
    }

    #[test]
    fn config_default_matches_yaml_default() {
        let yaml: BalancerConfig = serde_yaml::from_str("models: {}").unwrap();
        assert_eq!(BalancerConfig::default().max_inflight, yaml.max_inflight);
        assert_eq!(yaml.max_inflight, DEFAULT_MAX_INFLIGHT);
    }

    #[test]
    fn affinity_same_key_same_replica() {
        let set = set3();
        let first = set.pick("conv-9").unwrap().url().to_string();
        for _ in 0..50 {
            assert_eq!(set.pick("conv-9").unwrap().url(), first);
        }
    }

    #[test]
    fn least_loaded_spill_then_fallback_never_drops() {
        let set = ReplicaSet::new([Replica::new("http://a"), Replica::new("http://b")], 2);
        let key = "conv-spill";
        let primary = set.pick(key).unwrap();
        let p_url = primary.url().to_string();
        let l2 = set.pick(key).unwrap();
        assert_eq!(l2.url(), p_url, "under cap -> stays on primary");
        // primary at cap (2) -> spill to the successor ring node.
        let spill = set.pick(key).unwrap();
        assert_ne!(spill.url(), p_url, "at cap -> spill to successor");
        // saturate the successor too; a fully-loaded set still serves (least-loaded).
        let _s2 = set.pick(key).unwrap();
        let _over = set.pick(key).unwrap();
        // drain the primary -> affinity reclaims it.
        drop(primary);
        drop(l2);
        assert_eq!(
            set.pick(key).unwrap().url(),
            p_url,
            "drained primary reclaims affinity"
        );
    }

    #[test]
    fn lease_releases_inflight_on_drop() {
        let set = ReplicaSet::new([Replica::new("http://solo")], 8);
        {
            let l = set.pick("k").unwrap();
            assert_eq!(l.inflight(), 1);
            let l2 = set.pick("k").unwrap();
            assert_eq!(l2.inflight(), 2);
        }
        assert_eq!(set.pick("k").unwrap().inflight(), 1, "drops decremented");
    }

    #[test]
    fn health_eviction_and_restore() {
        let set = set3();
        let key = "conv-health";
        let primary_id = set.pick(key).unwrap().id().to_string();
        set.mark_unhealthy(&primary_id);
        for _ in 0..20 {
            assert_ne!(set.pick(key).unwrap().id(), primary_id, "evicted");
        }
        set.mark_healthy(&primary_id);
        assert_eq!(
            set.pick(key).unwrap().id(),
            primary_id,
            "restored + affinity"
        );
    }

    #[test]
    fn all_unhealthy_yields_none() {
        let set = set3();
        for id in ["http://a", "http://b", "http://c"] {
            set.mark_unhealthy(id);
        }
        assert!(set.pick("k").is_none());
    }

    #[test]
    fn dynamic_registration_grows_the_pool() {
        let bal = Balancer::new(4);
        bal.register("qwen3", Replica::new("http://127.0.0.1:1234"));
        bal.register("qwen3", Replica::new("http://127.0.0.1:1235"));
        bal.register("qwen3", Replica::new("http://127.0.0.1:1234")); // dup -> no-op
        let set = bal.set_for(Some("qwen3")).unwrap();
        assert_eq!(set.len(), 2);
        assert_eq!(bal.models(), vec!["qwen3"]);
    }

    #[test]
    fn single_set_fallback_ignores_model_mismatch() {
        let mut cfg = BalancerConfig::default();
        cfg.models
            .insert("configured".into(), vec![Replica::new("http://only")]);
        let bal = Balancer::from_config(cfg);
        assert!(bal.set_for(Some("some-other-id")).is_some());
        assert!(bal.set_for(None).is_some());
    }
}
