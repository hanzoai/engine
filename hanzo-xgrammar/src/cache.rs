//! Thread-safe Adaptive LRU Compiled-Grammar Cache.
//!
//! In multi-turn agent loops and production tool-calling APIs (OpenAI/MCP), the
//! same JSON Schema or toolset is reused across hundreds or thousands of Zen
//! inference requests. [`GrammarCache`] caches compiled [`CompiledGrammar`]
//! artifacts keyed by `(schema_hash, vocab_fingerprint, pda_config)` under both
//! an entry cap and a byte-budget cap, providing < 1 µs warm-cache hits and
//! bounded memory overhead.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::grammar::{Grammar, PdaConfig};
use crate::mask::CompiledGrammar;
use crate::schema::{SchemaDocument, SchemaError};
use crate::vocab::Vocabulary;

/// Composite cache key uniquely identifying a compiled grammar for a vocabulary & config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GrammarCacheKey {
    pub schema_hash: u64,
    pub vocab_fingerprint: u64,
    pub max_recursion_depth: usize,
    pub max_active_stacks: usize,
}

impl GrammarCacheKey {
    pub fn from_schema(doc: &SchemaDocument, vocab: &Vocabulary, config: PdaConfig) -> Self {
        Self {
            schema_hash: doc.canonical_hash(),
            vocab_fingerprint: vocab.fingerprint(),
            max_recursion_depth: config.max_recursion_depth,
            max_active_stacks: config.max_active_stacks,
        }
    }

    pub fn from_grammar(grammar: &Grammar, vocab: &Vocabulary, config: PdaConfig) -> Self {
        Self {
            schema_hash: grammar.fingerprint(),
            vocab_fingerprint: vocab.fingerprint(),
            max_recursion_depth: config.max_recursion_depth,
            max_active_stacks: config.max_active_stacks,
        }
    }
}

/// Runtime telemetry snapshot from [`GrammarCache`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    pub entries: usize,
    pub total_memory_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

struct CacheInner {
    max_entries: usize,
    max_bytes: usize,
    map: HashMap<GrammarCacheKey, Arc<CompiledGrammar>>,
    lru: VecDeque<GrammarCacheKey>,
    total_memory_bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

/// Thread-safe LRU cache of [`CompiledGrammar`] instances bounded by entry count and RAM bytes.
pub struct GrammarCache {
    inner: Mutex<CacheInner>,
}

impl GrammarCache {
    /// Create a new [`GrammarCache`] with an entry limit and a byte budget.
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                max_entries: max_entries.max(1),
                max_bytes: max_bytes.max(1024),
                map: HashMap::new(),
                lru: VecDeque::new(),
                total_memory_bytes: 0,
                hits: 0,
                misses: 0,
                evictions: 0,
            }),
        }
    }

    /// Look up a cached [`CompiledGrammar`] without compiling on miss.
    pub fn get(&self, key: &GrammarCacheKey) -> Option<Arc<CompiledGrammar>> {
        let mut inner = self.inner.lock().expect("GrammarCache mutex poisoned");
        if let Some(found) = inner.map.get(key).cloned() {
            inner.hits += 1;
            touch_lru(&mut inner.lru, *key);
            Some(found)
        } else {
            inner.misses += 1;
            None
        }
    }

    /// Insert a [`CompiledGrammar`] into the cache, evicting LRU entries as needed.
    pub fn insert(&self, key: GrammarCacheKey, compiled: Arc<CompiledGrammar>) {
        let mut inner = self.inner.lock().expect("GrammarCache mutex poisoned");
        let entry_bytes = compiled.memory_bytes();

        if let Some(prev) = inner.map.insert(key, compiled) {
            inner.total_memory_bytes = inner.total_memory_bytes.saturating_sub(prev.memory_bytes());
        }
        inner.total_memory_bytes = inner.total_memory_bytes.saturating_add(entry_bytes);
        touch_lru(&mut inner.lru, key);

        // Evict least-recently-used entries while exceeding capacity or byte budget
        // (always retaining at least the newly inserted entry if it exceeds max_bytes alone).
        while inner.map.len() > 1
            && (inner.map.len() > inner.max_entries || inner.total_memory_bytes > inner.max_bytes)
        {
            let Some(victim_key) = inner.lru.pop_front() else {
                break;
            };
            if victim_key == key {
                inner.lru.push_back(victim_key);
                break;
            }
            if let Some(victim) = inner.map.remove(&victim_key) {
                inner.total_memory_bytes = inner
                    .total_memory_bytes
                    .saturating_sub(victim.memory_bytes());
                inner.evictions += 1;
            }
        }
    }

    /// Get-or-compile a [`SchemaDocument`] against `vocab` and `config`.
    ///
    /// Note: compilation runs *outside* the cache mutex so concurrent threads
    /// hitting cached schemas are never blocked by a cold compile.
    pub fn get_or_compile_schema(
        &self,
        doc: &SchemaDocument,
        vocab: &Arc<Vocabulary>,
        config: PdaConfig,
    ) -> Result<(Arc<CompiledGrammar>, bool), SchemaError> {
        let key = GrammarCacheKey::from_schema(doc, vocab, config);
        if let Some(hit) = self.get(&key) {
            return Ok((hit, true));
        }
        let grammar = Arc::new(doc.compile_grammar()?);
        let compiled = Arc::new(CompiledGrammar::compile(grammar, Arc::clone(vocab), config));
        self.insert(key, Arc::clone(&compiled));
        Ok((compiled, false))
    }

    /// Get-or-compile a prebuilt [`Grammar`] against `vocab` and `config`.
    pub fn get_or_compile_grammar(
        &self,
        grammar: &Arc<Grammar>,
        vocab: &Arc<Vocabulary>,
        config: PdaConfig,
    ) -> (Arc<CompiledGrammar>, bool) {
        let key = GrammarCacheKey::from_grammar(grammar, vocab, config);
        if let Some(hit) = self.get(&key) {
            return (hit, true);
        }
        let compiled = Arc::new(CompiledGrammar::compile(
            Arc::clone(grammar),
            Arc::clone(vocab),
            config,
        ));
        self.insert(key, Arc::clone(&compiled));
        (compiled, false)
    }

    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock().expect("GrammarCache mutex poisoned");
        CacheStats {
            entries: inner.map.len(),
            total_memory_bytes: inner.total_memory_bytes,
            hits: inner.hits,
            misses: inner.misses,
            evictions: inner.evictions,
        }
    }

    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("GrammarCache mutex poisoned");
        inner.map.clear();
        inner.lru.clear();
        inner.total_memory_bytes = 0;
    }
}

fn touch_lru(lru: &mut VecDeque<GrammarCacheKey>, key: GrammarCacheKey) {
    if let Some(pos) = lru.iter().position(|k| k == &key) {
        lru.remove(pos);
    }
    lru.push_back(key);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn lru_cache_hits_evicts_and_is_thread_safe() {
        let vocab = Arc::new(Vocabulary::realistic_bpe(1024));
        let cache = Arc::new(GrammarCache::new(2, 16 * 1024 * 1024));
        let s1 = SchemaDocument::from_json_str(r#"{"type":"integer"}"#).unwrap();
        let s2 = SchemaDocument::from_json_str(r#"{"type":"boolean"}"#).unwrap();
        let s3 = SchemaDocument::from_json_str(r#"{"type":"string"}"#).unwrap();

        let (g1, hit1) = cache
            .get_or_compile_schema(&s1, &vocab, PdaConfig::default())
            .unwrap();
        assert!(!hit1);
        let (g1_again, hit2) = cache
            .get_or_compile_schema(&s1, &vocab, PdaConfig::default())
            .unwrap();
        assert!(hit2);
        assert!(Arc::ptr_eq(&g1, &g1_again));

        let (_, hit3) = cache
            .get_or_compile_schema(&s2, &vocab, PdaConfig::default())
            .unwrap();
        assert!(!hit3);
        // Touch s1 so s2 becomes LRU, then insert s3 -> s2 evicted.
        let (_, hit4) = cache
            .get_or_compile_schema(&s1, &vocab, PdaConfig::default())
            .unwrap();
        assert!(hit4);
        let (_, hit5) = cache
            .get_or_compile_schema(&s3, &vocab, PdaConfig::default())
            .unwrap();
        assert!(!hit5);

        let st = cache.stats();
        assert_eq!(st.entries, 2);
        assert_eq!(st.evictions, 1);
        assert_eq!(st.hits, 2);
        assert_eq!(st.misses, 3);

        // Concurrent read/compile stress test.
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let c = Arc::clone(&cache);
                let v = Arc::clone(&vocab);
                let doc = s1.clone();
                thread::spawn(move || {
                    for _ in 0..25 {
                        let (_, _) = c
                            .get_or_compile_schema(&doc, &v, PdaConfig::default())
                            .unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(cache.stats().hits >= 100);
    }
}
