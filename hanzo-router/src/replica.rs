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
use std::sync::{Arc, RwLock};

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
}

impl Replica {
    pub fn new(url: impl Into<String>) -> Self {
        let url = url.into();
        Self {
            id: url.clone(),
            url,
        }
    }

    fn normalized(mut self) -> Self {
        if self.id.is_empty() {
            self.id = self.url.clone();
        }
        self
    }
}

struct Node {
    replica: Replica,
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
            max_inflight,
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
        let mut out: Vec<ReplicaStatus> = inner
            .nodes
            .values()
            .map(|n| ReplicaStatus {
                id: n.replica.id.clone(),
                url: n.replica.url.clone(),
                healthy: n.healthy.load(Ordering::Acquire),
                inflight: n.inflight.load(Ordering::Acquire),
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
}

/// Declarative pool: `model -> replicas`, plus the shared in-flight ceiling.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct BalancerConfig {
    #[serde(default = "default_max_inflight")]
    pub max_inflight: usize,
    #[serde(default)]
    pub models: BTreeMap<String, Vec<Replica>>,
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
        if let Some(set) = model.and_then(|m| sets.get(m)) {
            return Some(set.clone());
        }
        if sets.len() == 1 {
            return sets.values().next().cloned();
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
