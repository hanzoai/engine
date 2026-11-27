// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Thread-safe bounded LRU KV-block pools for CacheSlide and matched prefix caching.
//!
//! Enforces three hard systems invariants:
//! 1. **Bounded memory**: `current_bytes <= max_bytes` at all times; oversized entries fail
//!    closed (`InsertOutcome::RejectedExceedsBudget`) rather than breaching the budget.
//! 2. **Strict tenant/ACL isolation**: keys include full [`crate::segment::IsolationScope`],
//!    and [`SegmentKvCache::evict_tenant`] purges all blocks belonging to a departing tenant
//!    without affecting any other tenant.
//! 3. **Short-segment admission gating**: segments shorter than `min_segment_tokens` (default
//!    `32`) can be bypassed so hashing/dispatch overhead never degrades tiny-segment latency.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use crate::segment::{PrefixChainKey, SegmentKey, SegmentKvPayload};

/// Outcome of attempting to insert a KV payload into a bounded cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertOutcome {
    /// New entry stored in cache.
    Inserted,
    /// Existing entry replaced in place.
    Updated,
    /// Segment length was below `min_segment_tokens`; bypassed cleanly.
    SkippedTooShort,
    /// Entry byte size exceeded the cache's total `max_bytes` budget; rejected fail-closed.
    RejectedExceedsBudget,
}

/// Snapshot of cache telemetry and byte accounting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Total successful cache lookups.
    pub hits: u64,
    /// Total failed cache lookups.
    pub misses: u64,
    /// Total LRU evictions performed to stay within `max_bytes`.
    pub evictions: u64,
    /// Entries rejected because `entry.byte_size() > max_bytes`.
    pub oversized_rejections: u64,
    /// Entries skipped because `entry.num_tokens < min_segment_tokens`.
    pub short_segment_skips: u64,
    /// Entries dropped on read due to SHA-256 payload integrity failure.
    pub integrity_failures: u64,
    /// Current in-memory byte usage (`<= max_bytes`).
    pub current_bytes: usize,
    /// High-water mark of in-memory byte usage (`<= max_bytes`).
    pub peak_bytes: usize,
    /// Current logical GPU KV bytes at each entry's `kv_dtype`.
    pub current_gpu_kv_bytes: usize,
    /// High-water mark of logical GPU KV bytes.
    pub peak_gpu_kv_bytes: usize,
    /// Number of resident entries.
    pub entry_count: usize,
    /// Configured maximum byte budget.
    pub max_bytes: usize,
}

impl CacheStats {
    /// Cache hit rate in `[0.0, 1.0]`.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[derive(Debug)]
struct LruEntry<V> {
    value: Arc<V>,
    bytes: usize,
    gpu_bytes: usize,
    generation: u64,
}

#[derive(Debug)]
struct BoundedLruInner<K> {
    max_bytes: usize,
    min_segment_tokens: usize,
    verify_integrity_on_get: bool,
    entries: HashMap<K, LruEntry<SegmentKvPayload>>,
    /// FIFO queue of `(key, generation)` tokens; stale generations are skipped lazily on eviction,
    /// giving amortized O(1) LRU touch and eviction with zero unsafe pointer lists.
    order: VecDeque<(K, u64)>,
    next_gen: u64,
    stats: CacheStats,
}

impl<K: Eq + Hash + Clone> BoundedLruInner<K> {
    fn new(max_bytes: usize, min_segment_tokens: usize, verify_integrity_on_get: bool) -> Self {
        Self {
            max_bytes,
            min_segment_tokens,
            verify_integrity_on_get,
            entries: HashMap::new(),
            order: VecDeque::new(),
            next_gen: 1,
            stats: CacheStats {
                max_bytes,
                ..CacheStats::default()
            },
        }
    }

    fn alloc_gen(&mut self) -> u64 {
        let g = self.next_gen;
        self.next_gen = self.next_gen.wrapping_add(1);
        g
    }

    fn compact_queue_if_needed(&mut self) {
        let threshold = (self.entries.len().saturating_mul(4)).max(64);
        if self.order.len() <= threshold {
            return;
        }
        let entries = &self.entries;
        self.order
            .retain(|(k, g)| entries.get(k).is_some_and(|e| e.generation == *g));
    }

    fn get(&mut self, key: &K) -> Option<Arc<SegmentKvPayload>> {
        let (val, valid) = match self.entries.get(key) {
            Some(entry) => {
                let ok = !self.verify_integrity_on_get || entry.value.verify_integrity();
                (Arc::clone(&entry.value), ok)
            }
            None => {
                self.stats.misses += 1;
                return None;
            }
        };

        if !valid {
            self.remove_key(key);
            self.stats.integrity_failures += 1;
            self.stats.misses += 1;
            return None;
        }

        let g = self.alloc_gen();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.generation = g;
        }
        self.order.push_back((key.clone(), g));
        self.compact_queue_if_needed();
        self.stats.hits += 1;
        Some(val)
    }

    fn remove_key(&mut self, key: &K) -> bool {
        if let Some(removed) = self.entries.remove(key) {
            self.stats.current_bytes -= removed.bytes;
            self.stats.current_gpu_kv_bytes -= removed.gpu_bytes;
            self.stats.entry_count = self.entries.len();
            true
        } else {
            false
        }
    }

    fn evict_one_lru(&mut self) -> bool {
        while let Some((cand_key, cand_gen)) = self.order.pop_front() {
            let is_live = self
                .entries
                .get(&cand_key)
                .is_some_and(|e| e.generation == cand_gen);
            if is_live {
                self.remove_key(&cand_key);
                self.stats.evictions += 1;
                return true;
            }
        }
        false
    }

    fn insert(&mut self, key: K, payload: Arc<SegmentKvPayload>) -> InsertOutcome {
        if payload.num_tokens < self.min_segment_tokens {
            self.stats.short_segment_skips += 1;
            return InsertOutcome::SkippedTooShort;
        }
        let bytes = payload.byte_size();
        if bytes > self.max_bytes {
            self.stats.oversized_rejections += 1;
            return InsertOutcome::RejectedExceedsBudget;
        }

        let gpu_bytes = payload.gpu_kv_bytes();
        let existed = self.remove_key(&key);

        while self.stats.current_bytes + bytes > self.max_bytes {
            if !self.evict_one_lru() {
                break;
            }
        }

        debug_assert!(self.stats.current_bytes + bytes <= self.max_bytes);
        let g = self.alloc_gen();
        self.entries.insert(
            key.clone(),
            LruEntry {
                value: payload,
                bytes,
                gpu_bytes,
                generation: g,
            },
        );
        self.order.push_back((key, g));
        self.stats.current_bytes += bytes;
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.current_bytes);
        self.stats.current_gpu_kv_bytes += gpu_bytes;
        self.stats.peak_gpu_kv_bytes = self
            .stats
            .peak_gpu_kv_bytes
            .max(self.stats.current_gpu_kv_bytes);
        self.stats.entry_count = self.entries.len();
        self.compact_queue_if_needed();

        if existed {
            InsertOutcome::Updated
        } else {
            InsertOutcome::Inserted
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.stats.current_bytes = 0;
        self.stats.current_gpu_kv_bytes = 0;
        self.stats.entry_count = 0;
    }
}

/// Thread-safe bounded LRU KV segment cache for CacheSlide.
#[derive(Debug)]
pub struct SegmentKvCache {
    inner: Mutex<BoundedLruInner<SegmentKey>>,
}

impl SegmentKvCache {
    /// Create a bounded cache with explicit byte budget and minimum segment admission length.
    ///
    /// Recommended production `min_segment_tokens` is `32` to avoid hash/dispatch overhead
    /// on tiny unique micro-segments (Negative Regime #2).
    #[must_use]
    pub fn new(max_bytes: usize, min_segment_tokens: usize) -> Self {
        Self {
            inner: Mutex::new(BoundedLruInner::new(max_bytes, min_segment_tokens, false)),
        }
    }

    /// Create a bounded cache that also verifies SHA-256 payload integrity on every `get`.
    #[must_use]
    pub fn with_integrity_checks(max_bytes: usize, min_segment_tokens: usize) -> Self {
        Self {
            inner: Mutex::new(BoundedLruInner::new(max_bytes, min_segment_tokens, true)),
        }
    }

    /// Configured minimum segment length admitted into the cache.
    #[must_use]
    pub fn min_segment_tokens(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .min_segment_tokens
    }

    /// Look up a cached KV segment by its tenant-isolated [`SegmentKey`].
    #[must_use]
    pub fn get(&self, key: &SegmentKey) -> Option<Arc<SegmentKvPayload>> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
    }

    /// Insert a KV segment into the bounded LRU pool.
    pub fn insert(&self, key: SegmentKey, payload: Arc<SegmentKvPayload>) -> InsertOutcome {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, payload)
    }

    /// Evict every cached segment belonging to `tenant_id` without disturbing other tenants.
    /// Returns the number of entries purged.
    pub fn evict_tenant(&self, tenant_id: &str) -> usize {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let to_remove: Vec<SegmentKey> = guard
            .entries
            .keys()
            .filter(|k| k.scope.tenant_id == tenant_id)
            .cloned()
            .collect();
        let count = to_remove.len();
        for k in &to_remove {
            guard.remove_key(k);
        }
        guard.compact_queue_if_needed();
        count
    }

    /// Snapshot current cache statistics and exact byte accounting.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).stats
    }

    /// Clear all resident entries (retaining cumulative counters).
    pub fn clear(&self) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

/// Matched thread-safe bounded LRU prefix cache baseline (`prefix_cache_only`).
///
/// Keyed by [`PrefixChainKey`] (cumulative chain hash of all preceding segments +
/// [`crate::segment::IsolationScope`]), mirroring `hanzo-engine/src/paged_attention/block_hash.rs`.
#[derive(Debug)]
pub struct PrefixKvCache {
    inner: Mutex<BoundedLruInner<PrefixChainKey>>,
}

impl PrefixKvCache {
    /// Create a bounded prefix cache with byte budget `max_bytes`.
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(BoundedLruInner::new(max_bytes, 1, false)),
        }
    }

    /// Look up a cached prefix block.
    #[must_use]
    pub fn get(&self, key: &PrefixChainKey) -> Option<Arc<SegmentKvPayload>> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
    }

    /// Insert a prefix block into the bounded LRU pool.
    pub fn insert(&self, key: PrefixChainKey, payload: Arc<SegmentKvPayload>) -> InsertOutcome {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, payload)
    }

    /// Evict all prefix blocks belonging to `tenant_id`.
    pub fn evict_tenant(&self, tenant_id: &str) -> usize {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let to_remove: Vec<PrefixChainKey> = guard
            .entries
            .keys()
            .filter(|k| k.scope.tenant_id == tenant_id)
            .cloned()
            .collect();
        let count = to_remove.len();
        for k in &to_remove {
            guard.remove_key(k);
        }
        guard.compact_queue_if_needed();
        count
    }

    /// Snapshot current cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).stats
    }

    /// Clear all resident entries.
    pub fn clear(&self) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::{IsolationScope, KvDtype, PromptSegment, SegmentKind};
    use std::thread;

    fn dummy_payload(orig_pos: usize, num_tokens: usize) -> Arc<SegmentKvPayload> {
        let num_layers = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let elems = num_tokens * num_kv_heads * head_dim;
        Arc::new(SegmentKvPayload::new(
            orig_pos,
            [0u8; 32],
            true,
            num_tokens,
            num_layers,
            num_kv_heads,
            head_dim,
            KvDtype::F16,
            vec![vec![0.1; elems]; num_layers],
            vec![vec![0.2; elems]; num_layers],
        ))
    }

    #[test]
    fn lru_enforces_exact_byte_budget_and_evicts_oldest() {
        let p = dummy_payload(0, 32);
        let one_size = p.byte_size();
        // Budget fits strictly 2 entries.
        let cache = SegmentKvCache::new(one_size * 2, 16);
        let scope = IsolationScope::new("t1", "acl_a", "zen-7b", KvDtype::F16);

        let k1 = SegmentKey::from_segment(
            scope.clone(),
            &PromptSegment::new(SegmentKind::ToolDefinition, "t1", vec![1; 32]),
        );
        let k2 = SegmentKey::from_segment(
            scope.clone(),
            &PromptSegment::new(SegmentKind::ToolDefinition, "t2", vec![2; 32]),
        );
        let k3 = SegmentKey::from_segment(
            scope,
            &PromptSegment::new(SegmentKind::ToolDefinition, "t3", vec![3; 32]),
        );

        assert_eq!(
            cache.insert(k1.clone(), Arc::clone(&p)),
            InsertOutcome::Inserted
        );
        assert_eq!(
            cache.insert(k2.clone(), Arc::clone(&p)),
            InsertOutcome::Inserted
        );
        // Touch k1 so k2 becomes least-recently-used.
        assert!(cache.get(&k1).is_some());
        assert_eq!(
            cache.insert(k3.clone(), Arc::clone(&p)),
            InsertOutcome::Inserted
        );

        let st = cache.stats();
        assert!(st.current_bytes <= st.max_bytes);
        assert!(st.peak_bytes <= st.max_bytes);
        assert_eq!(st.entry_count, 2);
        assert_eq!(st.evictions, 1);

        assert!(cache.get(&k1).is_some());
        assert!(cache.get(&k2).is_none());
        assert!(cache.get(&k3).is_some());

        // Oversized entry must be rejected fail-closed without evicting valid items.
        let huge = dummy_payload(0, 128);
        assert!(huge.byte_size() > one_size * 2);
        assert_eq!(cache.insert(k2, huge), InsertOutcome::RejectedExceedsBudget);
        assert_eq!(cache.stats().oversized_rejections, 1);
        assert_eq!(cache.stats().entry_count, 2);
    }

    #[test]
    fn tenant_isolation_and_selective_tenant_eviction() {
        let p = dummy_payload(64, 32);
        let cache = SegmentKvCache::new(p.byte_size() * 16, 16);
        let seg = PromptSegment::new(SegmentKind::ToolDefinition, "shared_schema", vec![7; 32]);

        let k_alpha = SegmentKey::from_segment(
            IsolationScope::new("tenant_alpha", "acl_eng", "zen-7b", KvDtype::F16),
            &seg,
        );
        let k_beta = SegmentKey::from_segment(
            IsolationScope::new("tenant_beta", "acl_eng", "zen-7b", KvDtype::F16),
            &seg,
        );
        let k_alpha_fin = SegmentKey::from_segment(
            IsolationScope::new("tenant_alpha", "acl_fin", "zen-7b", KvDtype::F16),
            &seg,
        );

        assert_eq!(
            cache.insert(k_alpha.clone(), Arc::clone(&p)),
            InsertOutcome::Inserted
        );
        // Zero cross-tenant or cross-ACL hits!
        assert!(cache.get(&k_beta).is_none());
        assert!(cache.get(&k_alpha_fin).is_none());

        assert_eq!(
            cache.insert(k_beta.clone(), Arc::clone(&p)),
            InsertOutcome::Inserted
        );
        assert_eq!(cache.evict_tenant("tenant_alpha"), 1);
        assert!(cache.get(&k_alpha).is_none());
        assert!(cache.get(&k_beta).is_some());
    }

    #[test]
    fn short_segment_admission_gate() {
        let short = dummy_payload(0, 12);
        let cache = SegmentKvCache::new(1024 * 1024, 32);
        let key = SegmentKey::from_segment(
            IsolationScope::new("t", "a", "m", KvDtype::F16),
            &PromptSegment::new(SegmentKind::UserQuery, "tiny", vec![1; 12]),
        );
        assert_eq!(cache.insert(key, short), InsertOutcome::SkippedTooShort);
        assert_eq!(cache.stats().short_segment_skips, 1);
        assert_eq!(cache.stats().entry_count, 0);
    }

    #[test]
    fn concurrent_multithreaded_cache_stress() {
        let p = dummy_payload(0, 32);
        let cache = Arc::new(SegmentKvCache::new(p.byte_size() * 8, 16));
        let mut handles = Vec::new();
        for tid in 0..4usize {
            let cache = Arc::clone(&cache);
            let p = Arc::clone(&p);
            handles.push(thread::spawn(move || {
                let scope = IsolationScope::new(
                    format!("tenant_{tid}"),
                    "acl_default",
                    "zen-7b",
                    KvDtype::F16,
                );
                for i in 0..40u32 {
                    let seg = PromptSegment::new(
                        SegmentKind::RagChunk,
                        format!("seg_{i}"),
                        vec![i + (tid as u32) * 100; 32],
                    );
                    let key = SegmentKey::from_segment(scope.clone(), &seg);
                    let _ = cache.insert(key.clone(), Arc::clone(&p));
                    let _ = cache.get(&key);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }
        let st = cache.stats();
        assert!(st.current_bytes <= st.max_bytes);
        assert!(st.peak_bytes <= st.max_bytes);
    }

    #[test]
    fn integrity_check_mode_rejects_corrupted_payload() {
        let mut p = (*dummy_payload(0, 32)).clone();
        // Corrupt a single key float after digest creation.
        p.keys[0][0] += 1.0;
        assert!(!p.verify_integrity());

        let cache = SegmentKvCache::with_integrity_checks(1024 * 1024, 16);
        let key = SegmentKey::from_segment(
            IsolationScope::new("t", "a", "m", KvDtype::F16),
            &PromptSegment::new(SegmentKind::ToolDefinition, "corrupt", vec![9; 32]),
        );
        assert_eq!(
            cache.insert(key.clone(), Arc::new(p)),
            InsertOutcome::Inserted
        );
        assert!(cache.get(&key).is_none());
        let st = cache.stats();
        assert_eq!(st.integrity_failures, 1);
        assert_eq!(st.entry_count, 0);
        assert_eq!(st.current_bytes, 0);
    }
}
