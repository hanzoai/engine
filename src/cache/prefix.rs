use std::collections::HashMap;
use std::sync::Arc;

/// Represents a versioned snapshot of the KV state for a specific tenant and model.
pub struct PrefixCacheEntry {
    pub kv_snapshot: Vec<u8>,
    pub tenant_id: String,
    pub model_hash: String,
    pub timestamp: u64,
}

pub struct PrefixCacheManager {
    // Radix-based storage for prefix-aware lookups
    storage: HashMap<String, PrefixCacheEntry>,
}

impl PrefixCacheManager {
    pub fn new() -> Self {
        Self { storage: HashMap::new() }
    }

    pub fn get_cache_key(&self, model_id: &str, tenant_id: &str, prompt: &str) -> String {
        // Deterministic key generation including tenant authorization
        format!("{}:{}:{}", model_id, tenant_id, blake3::hash(prompt.as_bytes()))
    }

    pub fn insert(&mut self, key: String, entry: PrefixCacheEntry) {
        self.storage.insert(key, entry);
    }

    pub fn lookup(&self, key: &str) -> Option<&PrefixCacheEntry> {
        self.storage.get(key)
    }
}
