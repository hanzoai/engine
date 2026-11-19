use crate::cache::kv_manager::KVBlock;
use std::collections::HashMap;

/// CacheSlide implementation for position-shifted KV reuse.
/// HIP-1345 compliant: enforces per-tenant salt and bounds.
pub struct CacheSlideEngine {
    pub cache_salt: String,
    pub kv_computed_tokens: usize,
}

impl CacheSlideEngine {
    pub fn new(tenant_id: &str, max_tokens: usize) -> Self {
        Self {
            cache_salt: tenant_id.to_string(),
            kv_computed_tokens: max_tokens,
        }
    }

    /// Computes a position-invariant hash for a prompt segment.
    /// This allows matching KV blocks even if the absolute position shifts.
    pub fn compute_segment_hash(&self, tokens: &[u32]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut s = std::collections::hash_map::DefaultHasher::new();
        self.cache_salt.hash(&mut s);
        tokens.hash(&mut s);
        s.finish()
    }

    pub fn validate_reuse(&self, segment_len: usize) -> bool {
        segment_len <= self.kv_computed_tokens
    }
}
