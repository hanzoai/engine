// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! Position-shifted segment reuse for the KV fabric (HIP-1345 §2.2, Bounty #12).
//!
//! Hanzo's block-level prefix cache (`hanzo-engine/src/paged_attention/block_hash.rs`)
//! identifies blocks by a parent-chained hash. Chain hashing is *position
//! dependent*: when an unchanged system or tool segment shifts by even one
//! token, every chained block hash inside it changes and the cache misses.
//! Agent workloads repeat such segments at shifting absolute positions
//! constantly (multi-turn tool loops), which the CacheSlide paper (FAST '26,
//! <https://www.usenix.org/conference/fast26/presentation/liu-yang>) calls the
//! Relative-Position-Dependent Caching (RPDC) pattern.
//!
//! This module adds the fabric-level counterpart to block hashing:
//!
//! - [`salted_segment_hash`]: a *position independent*, salted content hash
//!   over a whole segment (a contiguous token range). No parent chaining, so
//!   the hash survives arbitrary absolute position shifts of the segment.
//! - [`SegmentDirectory`]: an epoch-fenced ownership directory for reusable
//!   segments, keyed by `(cache_salt, segment_hash)` per HIP-1345 §1.1, which
//!   records the publish-time absolute token offset so consumers can translate
//!   positions and detect drift.
//!
//! Invariants enforced here:
//!
//! - **Salted keys (HIP-1345 §1.1)**: lookups and publications MUST use the
//!   resolved tenant salt; two tenants with equal token content never share
//!   directory entries.
//! - **Computed bounds (HIP-1345 §1.2)**: a segment is publishable only when
//!   `num_computed_tokens >= num_tokens`, i.e. every K/V projection inside the
//!   segment was actually computed. Aborted, errored, or prefill-canceled
//!   sequences MUST NOT be published; admission fails closed.
//!
//! What this module does NOT do: it does not interpret attention semantics.
//! RoPE/window position correction for reused K/V happens in the engine
//! attention path; the directory brokers ownership, offsets, and fencing, and
//! documents the fail-closed contract consumers MUST follow when a published
//! offset does not match the consumer's current offset.

use std::net::SocketAddr;

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::KvFabricError;

/// Compute a salted, position-independent XXH3 content hash for a segment.
///
/// Unlike [`crate::kv_block_hash`], this hash is NOT chained to a parent and
/// therefore does not change when the segment shifts in absolute prompt
/// position. The tenant `cache_salt` is mixed into the buffer unconditionally
/// (HIP-1345 §1.1): identical token content under different tenants yields
/// different hashes, so cross-tenant reuse cannot occur by key collision.
pub fn salted_segment_hash(tokens: &[u32], layer_group: u16, cache_salt: u64) -> u64 {
    let mut buf = Vec::with_capacity(2 + 8 + std::mem::size_of_val(tokens));
    buf.extend_from_slice(&layer_group.to_le_bytes());
    buf.extend_from_slice(&cache_salt.to_le_bytes());
    for &t in tokens {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    xxhash_rust::xxh3::xxh3_64(&buf)
}

/// Metadata describing one publishable reusable segment.
///
/// `num_computed_tokens` is the fabric-side counterpart of the scheduler's
/// `kv_computed_tokens` count (HIP-1345 §1.2): it must reflect the number of
/// tokens whose K/V projections were actually computed, never the full
/// logical sequence length.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentMeta {
    /// Position-independent content hash over `(layer_group, cache_salt, tokens)`.
    pub segment_hash: u64,
    /// Resolved tenant salt mixed into [`salted_segment_hash`].
    pub cache_salt: u64,
    /// KV cache group (layer group) the segment's K/V belongs to.
    pub layer_group: u16,
    /// Total tokens the segment covers.
    pub num_tokens: u64,
    /// Tokens whose K/V projections were actually computed. Admission
    /// requires `num_computed_tokens >= num_tokens` (HIP-1345 §1.2).
    pub num_computed_tokens: u64,
    pub num_bytes: u64,
    pub sha256: [u8; 32],
    /// Absolute token offset of the segment start in the publishing sequence.
    pub token_offset: u64,
}

/// Fresh replica owner record stored in [`SegmentDirectory`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentLocation {
    pub worker_id: String,
    pub peer_addr: SocketAddr,
    pub epoch: u64,
    pub segment_hash: u64,
    pub cache_salt: u64,
    pub num_tokens: u64,
    pub num_computed_tokens: u64,
    /// Absolute token offset at which the segment was published.
    pub token_offset: u64,
    pub sha256: [u8; 32],
}

impl SegmentLocation {
    /// Signed position shift needed to reuse this segment at `current_offset`.
    ///
    /// A consumer that wants to reuse the K/V of this segment inside a
    /// sequence where the segment now starts at `current_offset` must apply
    /// this shift to the segment's position-dependent state. Returns `None`
    /// when either offset cannot be represented as `i64` (fail closed: treat
    /// as a miss and recompute rather than guessing).
    pub fn position_shift(&self, current_offset: u64) -> Option<i64> {
        let published = i64::try_from(self.token_offset).ok()?;
        let current = i64::try_from(current_offset).ok()?;
        current.checked_sub(published)
    }
}

/// Epoch-fenced directory mapping `(cache_salt, segment_hash)` to active
/// replica owners, with publish-time token offsets for position translation.
///
/// Fencing semantics mirror [`crate::EpochFencedKvDirectory`]: per-worker
/// epochs are monotonic, stale publishes are rejected, and fenced workers'
/// older entries are purged so preempted or restarted replicas never serve
/// stale segments.
#[derive(Debug, Default)]
pub struct SegmentDirectory {
    worker_epochs: RwLock<FxHashMap<String, u64>>,
    segments: RwLock<FxHashMap<(u64, u64), std::collections::HashMap<String, std::collections::BTreeMap<u64, SegmentLocation>>>>,
}

impl SegmentDirectory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or advance a worker's active epoch. Rejects regressive epochs.
    pub fn register_worker_epoch(&self, worker_id: &str, epoch: u64) -> Result<u64, KvFabricError> {
        let mut epochs = self.worker_epochs.write();
        let entry = epochs.entry(worker_id.to_string()).or_insert(epoch);
        if epoch < *entry {
            return Err(KvFabricError::StaleEpoch {
                worker_id: worker_id.to_string(),
                got: epoch,
                current: *entry,
            });
        }
        *entry = epoch;
        Ok(epoch)
    }

    /// Fence `worker_id` at least up to `min_new_epoch` and purge its older segments.
    pub fn fence_epoch(&self, worker_id: &str, min_new_epoch: u64) -> u64 {
        let next = {
            let mut epochs = self.worker_epochs.write();
            let entry = epochs.entry(worker_id.to_string()).or_insert(0);
            let next = (*entry + 1).max(min_new_epoch);
            *entry = next;
            next
        };
        self.purge_worker_segments_below_epoch(worker_id, next);
        next
    }

    /// Return the worker's current fenced epoch, if registered.
    pub fn current_epoch(&self, worker_id: &str) -> Option<u64> {
        self.worker_epochs.read().get(worker_id).copied()
    }

    /// Publish reusable-segment ownership for `worker_id` under `epoch`.
    ///
    /// Admission is fail closed (HIP-1345 §1.2):
    ///
    /// - empty segments (`num_tokens == 0`) are rejected as [`KvFabricError::EmptySegment`];
    /// - segments with `num_computed_tokens < num_tokens` are rejected as
    ///   [`KvFabricError::IncompleteSegment`] — nothing is published, so a
    ///   consumer can never observe a segment whose K/V is only partially
    ///   computed (the aborted-prefill / uncomputed-sampled-tail hazard).
    pub fn publish_segment(
        &self,
        worker_id: &str,
        epoch: u64,
        peer_addr: SocketAddr,
        meta: &SegmentMeta,
    ) -> Result<(), KvFabricError> {
        if meta.num_tokens == 0 {
            return Err(KvFabricError::EmptySegment {
                segment_hash: meta.segment_hash,
            });
        }
        if meta.num_computed_tokens < meta.num_tokens {
            return Err(KvFabricError::IncompleteSegment {
                segment_hash: meta.segment_hash,
                num_computed_tokens: meta.num_computed_tokens,
                num_tokens: meta.num_tokens,
            });
        }
        self.register_worker_epoch(worker_id, epoch)?;

        let loc = SegmentLocation {
            worker_id: worker_id.to_string(),
            peer_addr,
            epoch,
            segment_hash: meta.segment_hash,
            cache_salt: meta.cache_salt,
            num_tokens: meta.num_tokens,
            num_computed_tokens: meta.num_computed_tokens,
            token_offset: meta.token_offset,
            sha256: meta.sha256,
        };
        let mut segments = self.segments.write();
        segments
            .entry((meta.cache_salt, meta.segment_hash))
            .or_default()
            .entry(worker_id.to_string())
            .or_default()
            .insert(meta.token_offset, loc);
        Ok(())
    }

    /// Look up active replica owners for `(cache_salt, segment_hash)`,
    /// pruning stale-epoch entries.
    ///
    /// The returned entries carry the publish-time absolute offset; consumers
    /// MUST apply [`SegmentLocation::position_shift`] and validate relative
    /// ordering before reusing K/V, and MUST treat alignment failure as a
    /// miss (recompute) rather than forcing reuse.
    pub fn lookup_segments(&self, cache_salt: u64, segment_hash: u64) -> Vec<SegmentLocation> {
        let epochs = self.worker_epochs.read();
        let mut segments = self.segments.write();
        let Some(owners) = segments.get_mut(&(cache_salt, segment_hash)) else {
            return Vec::new();
        };
        owners.retain(|worker_id, offsets| {
            let keep = epochs.get(worker_id).is_some_and(|epoch| {
                offsets.retain(|_, loc| loc.epoch >= *epoch);
                !offsets.is_empty()
            });
            keep
        });
        let fresh: Vec<SegmentLocation> = owners.values().flat_map(|offsets| offsets.values().cloned()).collect();
        if owners.is_empty() {
            segments.remove(&(cache_salt, segment_hash));
        }
        fresh
    }

    /// Evict all segment ownership entries belonging to `worker_id`.
    pub fn evict_worker(&self, worker_id: &str) {
        self.purge_worker_segments_below_epoch(worker_id, u64::MAX);
    }

    fn purge_worker_segments_below_epoch(&self, worker_id: &str, min_epoch: u64) {
        let mut segments = self.segments.write();
        segments.retain(|_, owners| {
            if let Some(offsets) = owners.get_mut(worker_id) {
                offsets.retain(|_, loc| !(loc.worker_id == worker_id && loc.epoch < min_epoch));
            }
            !owners.is_empty()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYSTEM_PROMPT: [u32; 8] = [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008];

    /// Deterministic stand-in payload digest for tests: SHA-256 over the
    /// segment's token bytes, mirroring what a publisher would verify.
    fn fake_sha(tokens: &[u32], layer_group: u16) -> [u8; 32] {
        let mut buf = Vec::with_capacity(2 + std::mem::size_of_val(tokens));
        buf.extend_from_slice(&layer_group.to_le_bytes());
        for &t in tokens {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        crate::compute_block_digest(&buf)
    }

    fn meta_for(
        salt: u64,
        tokens: &[u32],
        offset: u64,
        computed: u64,
        layer_group: u16,
    ) -> SegmentMeta {
        let hash = salted_segment_hash(tokens, layer_group, salt);
        SegmentMeta {
            segment_hash: hash,
            cache_salt: salt,
            layer_group,
            num_tokens: tokens.len() as u64,
            num_computed_tokens: computed,
            num_bytes: (tokens.len() * 64) as u64,
            sha256: fake_sha(tokens, layer_group),
            token_offset: offset,
        }
    }

    #[test]
    fn salted_segment_hash_is_position_independent_and_salt_separated() {
        let tokens: Vec<u32> = SYSTEM_PROMPT.to_vec();
        let a = salted_segment_hash(&tokens, 0, 77);
        let b = salted_segment_hash(&tokens, 0, 77);
        assert_eq!(a, b, "same content under same salt must hash identically");

        let other_tenant = salted_segment_hash(&tokens, 0, 78);
        assert_ne!(a, other_tenant, "salt must separate tenants (HIP-1345 §1.1)");

        let other_group = salted_segment_hash(&tokens, 1, 77);
        assert_ne!(a, other_group, "layer group must separate K/V kinds");

        // Token order matters: swapping tokens changes the content hash.
        let swapped: Vec<u32> = tokens.iter().rev().copied().collect();
        let swapped_hash = salted_segment_hash(&swapped, 0, 77);
        assert_ne!(a, swapped_hash);
    }

    #[test]
    fn publish_rejects_empty_and_incomplete_segments() {
        let dir = SegmentDirectory::new();
        let addr: SocketAddr = "127.0.0.1:9101".parse().unwrap();
        let m = meta_for(1, &SYSTEM_PROMPT, 0, 4, 0); // 4 of 8 computed: aborted prefill
        let err = dir
            .publish_segment("worker-a", 1, addr, &m)
            .unwrap_err();
        assert!(matches!(err, KvFabricError::IncompleteSegment { .. }));
        assert!(
            dir.lookup_segments(1, m.segment_hash).is_empty(),
            "incomplete segment must not be published"
        );

        let empty = meta_for(1, &[], 0, 0, 0);
        let err = dir.publish_segment("worker-a", 1, addr, &empty).unwrap_err();
        assert!(matches!(err, KvFabricError::EmptySegment { .. }));
    }

    #[test]
    fn segments_reuse_across_shifted_positions() {
        let dir = SegmentDirectory::new();
        let addr: SocketAddr = "127.0.0.1:9102".parse().unwrap();

        // First turn: system segment published at offset 0.
        let m = meta_for(5, &SYSTEM_PROMPT, 0, SYSTEM_PROMPT.len() as u64, 0);
        dir.publish_segment("worker-a", 1, addr, &m).unwrap();
        let hits = dir.lookup_segments(5, m.segment_hash);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].token_offset, 0);
        assert_eq!(hits[0].position_shift(0), Some(0));

        // Third turn: same bytes, now shifted by 128 tokens (two prior turns inserted).
        let shifted_m = meta_for(5, &SYSTEM_PROMPT, 128, SYSTEM_PROMPT.len() as u64, 0);
        dir.publish_segment("worker-b", 1, addr, &shifted_m).unwrap();
        let hits = dir.lookup_segments(5, m.segment_hash);
        assert_eq!(hits.len(), 2, "shifted occurrence must hit the same content key");
        // The directory returns locations sorted by publish-time offset.
        // Find the shifted one (offset 128, published by worker-b).
        let shifted = hits
            .iter()
            .find(|loc| loc.worker_id == "worker-b")
            .expect("worker-b's shifted occurrence must be in the results");
        assert_eq!(shifted.token_offset, 128, "shifted occurrence must keep its publish offset");
        assert_eq!(shifted.position_shift(128), Some(0), "reusing at the shifted offset is a zero shift");
        assert_eq!(shifted.position_shift(129), Some(1), "one token past the shifted start is +1");
        assert_eq!(shifted.position_shift(0), Some(-128), "the original turn-0 position is -128 relative to the shift");
    }

    #[test]
    fn salt_isolates_tenants_in_the_directory() {
        let dir = SegmentDirectory::new();
        let addr: SocketAddr = "127.0.0.1:9103".parse().unwrap();

        let tenant_a = meta_for(11, &SYSTEM_PROMPT, 0, SYSTEM_PROMPT.len() as u64, 0);
        let tenant_b = meta_for(12, &SYSTEM_PROMPT, 0, SYSTEM_PROMPT.len() as u64, 0);
        assert_ne!(tenant_a.segment_hash, tenant_b.segment_hash);

        dir.publish_segment("worker-a", 1, addr, &tenant_a).unwrap();
        dir.publish_segment("worker-a", 1, addr, &tenant_b).unwrap();

        let hits_a = dir.lookup_segments(11, tenant_a.segment_hash);
        assert_eq!(hits_a.len(), 1);
        let hits_b = dir.lookup_segments(12, tenant_b.segment_hash);
        assert_eq!(hits_b.len(), 1);
        // No key collision across tenants despite identical token bytes.
        let hits_cross = dir.lookup_segments(12, tenant_a.segment_hash);
        assert!(hits_cross.is_empty());
    }

    #[test]
    fn epoch_fencing_prunes_stale_segment_owners() {
        let dir = SegmentDirectory::new();
        let addr_a: SocketAddr = "127.0.0.1:9104".parse().unwrap();
        let addr_b: SocketAddr = "127.0.0.1:9105".parse().unwrap();

        let m = meta_for(3, &SYSTEM_PROMPT, 0, SYSTEM_PROMPT.len() as u64, 0);
        dir.publish_segment("worker-a", 1, addr_a, &m).unwrap();
        dir.publish_segment("worker-b", 1, addr_b, &m).unwrap();
        assert_eq!(dir.lookup_segments(3, m.segment_hash).len(), 2);

        let fenced = dir.fence_epoch("worker-a", 2);
        assert_eq!(fenced, 2);
        let err = dir
            .publish_segment("worker-a", 1, addr_a, &m)
            .unwrap_err();
        assert!(matches!(err, KvFabricError::StaleEpoch { .. }));

        let owners = dir.lookup_segments(3, m.segment_hash);
        assert_eq!(owners.len(), 1);
        assert_eq!(owners[0].worker_id, "worker-b");

        dir.publish_segment("worker-a", 2, addr_a, &m).unwrap();
        assert_eq!(dir.lookup_segments(3, m.segment_hash).len(), 2);

        dir.evict_worker("worker-b");
        let owners = dir.lookup_segments(3, m.segment_hash);
        assert_eq!(owners.len(), 1);
        assert_eq!(owners[0].worker_id, "worker-a");
    }

    #[test]
    fn concurrent_publish_and_lookup_never_tear() {
        let dir = std::sync::Arc::new(SegmentDirectory::new());
        let addr: SocketAddr = "127.0.0.1:9106".parse().unwrap();
        let m = std::sync::Arc::new(meta_for(
            9,
            &SYSTEM_PROMPT,
            64,
            SYSTEM_PROMPT.len() as u64,
            0,
        ));

        std::thread::scope(|scope| {
            for w in 0..8 {
                let dir = std::sync::Arc::clone(&dir);
                let m = std::sync::Arc::clone(&m);
                let addr = addr;
                scope.spawn(move || {
                    for i in 0..200u64 {
                        let _ = dir.publish_segment(
                            &format!("worker-{w}"),
                            i + 1,
                            addr,
                            &SegmentMeta { token_offset: 64 + i, ..m.as_ref().clone() },
                        );
                        let _ = dir.lookup_segments(9, m.segment_hash);
                    }
                });
            }
        });

        let owners = dir.lookup_segments(9, m.segment_hash);
        assert_eq!(owners.len(), 8, "all workers fenced at their final epochs stay fresh");
        for owner in owners {
            assert_eq!(owner.num_computed_tokens, owner.num_tokens);
        }
    }
}
