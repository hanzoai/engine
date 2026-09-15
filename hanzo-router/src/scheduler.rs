//! Agent placement over the existing replica pool. Pins and locality hints are
//! process-local and bounded. The caller supplies monotonic time for testability.
use super::{Lease, Node, ReplicaSet};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SESSION_TTL: Duration = Duration::from_secs(1800);
const PREFIX_TTL: Duration = Duration::from_secs(300);
const MAX_ENTRIES: usize = 10_000;

/// Keys must be scoped to the authenticated principal and model by the caller.
/// A prefix is a locality hint, never proof that a worker still holds KV blocks.
#[derive(Clone, Debug, Default)]
pub struct RoutingHints {
    pub session: Option<String>,
    pub prefix: Option<String>,
    pub role: Option<String>,
    pub target: Option<String>,
    /// Prompt + output tokens estimated from the request bytes.
    pub approx_tokens: usize,
    /// Prompt + output tokens counted by each worker's own tokenizer.
    pub token_counts: HashMap<String, usize>,
}

impl RoutingHints {
    /// A worker's exact count when it answered, the byte estimate otherwise.
    pub fn required_tokens(&self, worker: &str) -> usize {
        self.token_counts
            .get(worker)
            .copied()
            .unwrap_or(self.approx_tokens)
    }
}

#[derive(Clone)]
struct Entry {
    worker: String,
    seen: Instant,
}

#[derive(Default)]
pub struct SchedulerState {
    sessions: HashMap<String, Entry>,
    prefixes: HashMap<String, Entry>,
    ttft: HashMap<String, f64>,
}

impl SchedulerState {
    pub(super) fn invalidate_cache(&mut self, worker: &str) {
        self.prefixes.retain(|_, entry| entry.worker != worker);
        self.ttft.remove(worker);
    }

    pub(super) fn ttft_ms(&self, worker: &str) -> Option<f64> {
        self.ttft.get(worker).copied()
    }

    fn prune(&mut self, now: Instant) {
        self.sessions
            .retain(|_, e| now.saturating_duration_since(e.seen) < SESSION_TTL);
        self.prefixes
            .retain(|_, e| now.saturating_duration_since(e.seen) < PREFIX_TTL);
    }
}

fn remember(map: &mut HashMap<String, Entry>, key: String, worker: &str, now: Instant) {
    if map.len() >= MAX_ENTRIES && !map.contains_key(&key) {
        if let Some(oldest) = map
            .iter()
            .min_by_key(|(_, e)| e.seen)
            .map(|(k, _)| k.clone())
        {
            map.remove(&oldest);
        }
    }
    map.insert(
        key,
        Entry {
            worker: worker.to_owned(),
            seen: now,
        },
    );
}

impl ReplicaSet {
    pub(super) fn slots(&self, node: &Node) -> usize {
        let capacity = node.settings.read().unwrap().capacity;
        if capacity == 0 {
            self.max_inflight
        } else {
            capacity
        }
    }

    /// Explicit target, healthy session pin, recent prefix, role, normalized
    /// load, weighted session allocation, then measured TTFT. Saturated pools
    /// refuse admission; existing pins tolerate up to twice their worker's slots
    /// to keep short tool gaps from repeatedly causing expensive re-prefills.
    /// Selection and lease acquisition are serialized, preventing oversubscription
    /// from simultaneous new sessions. Exclusions bound pre-connect failover.
    pub fn pick_agent(
        &self,
        hints: &RoutingHints,
        excluded: &HashSet<String>,
        now: Instant,
    ) -> Option<Lease> {
        let inner = self.inner.read().unwrap();
        let mut state = self.scheduler.lock().unwrap();
        state.prune(now);
        let healthy = |n: &&Arc<Node>| {
            n.healthy.load(Ordering::Acquire)
                && !excluded.contains(&n.replica.id)
                && (n.replica.max_context == 0
                    || hints.required_tokens(&n.replica.id) <= n.replica.max_context)
        };
        let available =
            |n: &&Arc<Node>| healthy(n) && n.inflight.load(Ordering::Acquire) < self.slots(n);
        let mut selected = None;
        if let Some(target) = &hints.target {
            // A target is a configured ID, never a caller-supplied URL. Fail closed.
            selected = inner.nodes.get(target).filter(available).cloned();
            selected.as_ref()?;
        } else if let Some(pin) = hints.session.as_ref().and_then(|s| state.sessions.get(s)) {
            selected = inner
                .nodes
                .get(&pin.worker)
                .filter(|n| {
                    healthy(n)
                        && n.inflight.load(Ordering::Acquire) < self.slots(n).saturating_mul(2)
                })
                .cloned();
        }
        if selected.is_none() {
            let cached = hints.prefix.as_ref().and_then(|p| state.prefixes.get(p));
            let mut sessions: HashMap<&str, usize> = HashMap::new();
            for e in state.sessions.values() {
                *sessions.entry(&e.worker).or_default() += 1;
            }
            // Stable ID breaks exact ties, independent of HashMap iteration order.
            let rank = |n: &Arc<Node>| {
                let settings = n.settings.read().unwrap().clone();
                let cache = cached.is_some_and(|e| e.worker == n.replica.id);
                let role = hints
                    .role
                    .as_ref()
                    .is_some_and(|r| settings.roles.contains(r));
                let load = n.inflight.load(Ordering::Acquire) as f64 / self.slots(n) as f64;
                let allocation = (sessions.get(n.replica.id.as_str()).copied().unwrap_or(0) + 1)
                    as f64
                    / settings.weight as f64;
                (
                    !cache,
                    !role,
                    load,
                    allocation,
                    state.ttft_ms(&n.replica.id).unwrap_or(0.0),
                )
            };
            selected = inner
                .nodes
                .values()
                .filter(available)
                .min_by(|a, b| {
                    rank(a)
                        .partial_cmp(&rank(b))
                        .unwrap()
                        .then_with(|| a.replica.id.cmp(&b.replica.id))
                })
                .cloned();
        }
        let node = selected?;
        if let Some(session) = &hints.session {
            remember(&mut state.sessions, session.clone(), &node.replica.id, now);
        }
        Some(Lease::acquire(node))
    }

    /// Only complete successful responses teach locality. Aborted/error streams
    /// do not claim reusable prefixes. Health transitions invalidate these hints.
    pub fn observe_completion(&self, worker: &str, hints: &RoutingHints, now: Instant) {
        let inner = self.inner.read().unwrap();
        if !inner
            .nodes
            .get(worker)
            .is_some_and(|n| n.healthy.load(Ordering::Acquire))
        {
            return;
        }
        let mut state = self.scheduler.lock().unwrap();
        state.prune(now);
        if let Some(prefix) = &hints.prefix {
            remember(&mut state.prefixes, prefix.clone(), worker, now);
        }
        if let Some(session) = &hints.session {
            if let Some(pin) = state.sessions.get_mut(session) {
                if pin.worker == worker {
                    pin.seen = now;
                }
            }
        }
    }

    pub fn observe_ttft(&self, worker: &str, elapsed: Duration) {
        let inner = self.inner.read().unwrap();
        if !inner
            .nodes
            .get(worker)
            .is_some_and(|n| n.healthy.load(Ordering::Acquire))
        {
            return;
        }
        let sample = elapsed.as_secs_f64() * 1000.0;
        self.scheduler
            .lock()
            .unwrap()
            .ttft
            .entry(worker.to_owned())
            .and_modify(|old| *old = 0.8 * *old + 0.2 * sample)
            .or_insert(sample);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replica::Replica;

    fn pool() -> ReplicaSet {
        let mut spark = Replica::new("http://spark");
        spark.capacity = 4;
        spark.weight = 300;
        spark.roles = vec!["main".into()];
        let mut evo = Replica::new("http://evo");
        evo.capacity = 2;
        evo.roles = vec!["reviewer".into()];
        ReplicaSet::new([spark, evo], 8)
    }
    fn hints(session: &str) -> RoutingHints {
        RoutingHints {
            session: Some(session.into()),
            ..Default::default()
        }
    }

    #[test]
    fn context_capacity_overrides_pins_cache_roles_and_targets() {
        let mut spark = Replica::new("spark");
        spark.max_context = 1_000_000;
        let mut evo = Replica::new("evo");
        evo.max_context = 65_536;
        evo.roles = vec!["reviewer".into()];
        let p = ReplicaSet::new([spark, evo], 8);
        let now = Instant::now();
        let mut h = hints("growing-session");
        h.role = Some("reviewer".into());
        h.prefix = Some("cached-prefix".into());
        h.approx_tokens = 65_536;
        assert_eq!(pick(&p, &h, now).id(), "evo");
        p.observe_completion("evo", &h, now);
        h.approx_tokens = 176_939;
        assert_eq!(pick(&p, &h, now).id(), "spark");
        h.target = Some("evo".into());
        assert!(p.pick_agent(&h, &HashSet::new(), now).is_none());
        h.target = None;
        h.approx_tokens = 1_000_000;
        assert_eq!(pick(&p, &h, now).id(), "spark");
        h.approx_tokens += 1;
        assert!(p.pick_agent(&h, &HashSet::new(), now).is_none());
        h.approx_tokens = 176_939;
        p.mark_unhealthy("spark");
        assert!(p.pick_agent(&h, &HashSet::new(), now).is_none());
    }
    fn pick(pool: &ReplicaSet, h: &RoutingHints, now: Instant) -> Lease {
        pool.pick_agent(h, &HashSet::new(), now).unwrap()
    }

    #[test]
    fn weighted_sessions_and_sticky_failover() {
        let p = pool();
        let now = Instant::now();
        let mut counts = HashMap::new();
        for i in 0..400 {
            let h = hints(&format!("s{i}"));
            let l = pick(&p, &h, now);
            *counts.entry(l.id().to_string()).or_insert(0) += 1;
            assert_eq!(pick(&p, &h, now).id(), l.id());
        }
        assert_eq!(counts["http://spark"], 300);
        assert_eq!(counts["http://evo"], 100);
        let h = hints("failover");
        let original = pick(&p, &h, now).id().to_string();
        p.mark_unhealthy(&original);
        let replacement = pick(&p, &h, now).id().to_string();
        assert_ne!(replacement, original);
        p.mark_healthy(&original);
        assert_eq!(pick(&p, &h, now).id(), replacement);
    }

    #[test]
    fn cache_role_target_and_expiry() {
        let p = pool();
        let now = Instant::now();
        let mut h = hints("review");
        h.role = Some("reviewer".into());
        h.prefix = Some("org/model/prefix".into());
        assert_eq!(pick(&p, &h, now).id(), "http://evo");
        p.observe_completion("http://evo", &h, now);
        let mut other = hints("main");
        other.prefix = h.prefix.clone();
        other.role = Some("main".into());
        assert_eq!(
            pick(&p, &other, now).id(),
            "http://evo",
            "cache before role"
        );
        other.target = Some("http://spark".into());
        assert_eq!(pick(&p, &other, now).id(), "http://spark");
        other.target = Some("http://unconfigured".into());
        assert!(p.pick_agent(&other, &HashSet::new(), now).is_none());
        h.session = Some("fresh".into());
        h.role = Some("main".into());
        assert_eq!(pick(&p, &h, now + SESSION_TTL).id(), "http://spark");
    }

    #[test]
    fn hard_admission_and_drop_release() {
        let p = Arc::new(pool());
        let now = Instant::now();
        let leases: Vec<_> = (0..6)
            .map(|i| pick(&p, &hints(&format!("s{i}")), now))
            .collect();
        assert!(p.pick_agent(&hints("new"), &HashSet::new(), now).is_none());
        drop(leases);
        assert!(p.pick_agent(&hints("new"), &HashSet::new(), now).is_some());
        let barrier = Arc::new(std::sync::Barrier::new(21));
        let handles: Vec<_> = (0..20)
            .map(|i| {
                let (p, barrier) = (p.clone(), barrier.clone());
                std::thread::spawn(move || {
                    let lease = p.pick_agent(&hints(&format!("parallel{i}")), &HashSet::new(), now);
                    barrier.wait();
                    lease
                })
            })
            .collect();
        barrier.wait();
        let leases: Vec<_> = handles
            .into_iter()
            .filter_map(|h| h.join().unwrap())
            .collect();
        assert_eq!(leases.len(), 6);
    }

    #[test]
    fn unhealthy_cache_is_invalidated_and_ttft_is_ewma() {
        let p = pool();
        let now = Instant::now();
        let h = RoutingHints {
            prefix: Some("p".into()),
            ..Default::default()
        };
        p.observe_completion("http://evo", &h, now);
        p.observe_ttft("http://evo", Duration::from_millis(100));
        p.observe_ttft("http://evo", Duration::from_millis(200));
        assert_eq!(p.statuses()[0].ttft_ewma_ms, Some(120.0));
        p.mark_unhealthy("http://evo");
        p.mark_healthy("http://evo");
        assert_eq!(pick(&p, &h, now).id(), "http://spark");
    }

    #[test]
    fn tables_are_bounded() {
        let mut m = HashMap::new();
        let now = Instant::now();
        for i in 0..MAX_ENTRIES + 1 {
            remember(
                &mut m,
                i.to_string(),
                "a",
                now + Duration::from_nanos(i as u64),
            );
        }
        assert_eq!(m.len(), MAX_ENTRIES);
        assert!(!m.contains_key("0"));
    }
}
