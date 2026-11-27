// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Block-aligned RadixAttention prefix index (`arXiv:2312.07104`) with
//! reference-counted pinning, leaf-first LRU eviction, and strict tenant scope isolation.

use std::collections::{BTreeSet, HashMap};

use crate::crypto::Digest256;
use crate::error::PrefixCacheError;
use crate::key::{
    validate_scope_transition, CompatibilityProfile, PrefixCacheKey, ScopedTokenBlock,
    TenantAuthorization, TenantScope, TokenPrefixChain,
};

/// Storage residency tier of a prefix block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageTier {
    /// Tier 0: Hot HBM / device VRAM (fastest reuse, tightest capacity budget).
    Tier0Hbm = 0,
    /// Tier 1: Warm Host DRAM / local NVMe SSD directory (survives worker restart).
    Tier1WarmDisk = 1,
    /// Tier 2: Cold shared Object Store / S3-compatible backing directory (cross-worker).
    Tier2ColdObject = 2,
}

impl StorageTier {
    /// Human-readable tier label for scorecards.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Tier0Hbm => "tier0_hbm",
            Self::Tier1WarmDisk => "tier1_warm_disk",
            Self::Tier2ColdObject => "tier2_cold_object",
        }
    }
}

/// Metadata tracked for a single block-aligned node in [`RadixPrefixIndex`].
#[derive(Clone, Debug)]
pub struct RadixNode {
    /// Canonical 7-field cache key for this block.
    pub key: PrefixCacheKey,
    /// Parent block's 7-field key digest (`Digest256::ZERO` for root block 0).
    pub parent_key_digest: Digest256,
    /// Tenant authorization scope of this block.
    pub scope: TenantScope,
    /// Number of tokens in this block (`block_size`).
    pub token_len: u32,
    /// Starting token position of this block.
    pub token_start: u32,
    /// Accounted byte footprint of this block (used for tier budget enforcement).
    pub byte_size: u64,
    /// Hottest tier where this block is currently resident.
    pub residency: StorageTier,
    /// Whether this block is also persisted in Tier 1 (warm disk).
    pub in_tier1: bool,
    /// Whether this block is also persisted in Tier 2 (cold object store).
    pub in_tier2: bool,
    /// Active reference-count pins preventing eviction from Tier 0.
    pub pin_count: u32,
    /// Monotonic logical tick of the most recent hit or insertion.
    pub last_access_tick: u64,
    /// Cumulative number of cache hits served by this block.
    pub hit_count: u64,
    /// Child block key digests hanging off this node.
    pub children: BTreeSet<Digest256>,
}

/// A single matched block entry returned by [`RadixPrefixIndex::match_longest_prefix`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchedBlock {
    /// Zero-based block index along the matched prefix.
    pub block_index: u32,
    /// 7-field cache key of the matched block.
    pub key: PrefixCacheKey,
    /// Scope under which the block was authorized and matched.
    pub scope: TenantScope,
    /// Hottest tier where the block currently resides.
    pub tier: StorageTier,
    /// Starting token offset of this block.
    pub token_start: u32,
    /// Number of tokens covered by this block.
    pub token_len: u32,
}

/// Result of a longest-prefix lookup across an authorized block sequence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefixMatchResult {
    /// Contiguous sequence of matched blocks starting at block 0.
    pub matched_blocks: Vec<MatchedBlock>,
    /// Total number of tokens matched across all tiers.
    pub matched_tokens: usize,
    /// Number of matched blocks served from Tier 0 (HBM).
    pub tier0_blocks: usize,
    /// Number of matched blocks served from Tier 1 (Warm DRAM/SSD).
    pub tier1_blocks: usize,
    /// Number of matched blocks served from Tier 2 (Cold Object Store).
    pub tier2_blocks: usize,
}

/// Block-aligned RadixAttention prefix index supporting multi-tier residency,
/// reference-counted pinning, leaf-first LRU eviction, and strict tenant isolation.
#[derive(Clone, Debug, Default)]
pub struct RadixPrefixIndex {
    nodes: HashMap<Digest256, RadixNode>,
    tick: u64,
    tier0_bytes: u64,
    tier1_bytes: u64,
    tier2_bytes: u64,
}

impl RadixPrefixIndex {
    /// Create an empty radix prefix index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of indexed prefix nodes across all tiers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if no prefix blocks are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Current accounted byte footprint in Tier 0 (HBM).
    #[must_use]
    pub const fn tier0_bytes(&self) -> u64 {
        self.tier0_bytes
    }

    /// Current accounted byte footprint in Tier 1 (Warm DRAM/SSD).
    #[must_use]
    pub const fn tier1_bytes(&self) -> u64 {
        self.tier1_bytes
    }

    /// Current accounted byte footprint in Tier 2 (Cold Object Store).
    #[must_use]
    pub const fn tier2_bytes(&self) -> u64 {
        self.tier2_bytes
    }

    /// Borrow a node by its 7-field key digest.
    #[must_use]
    pub fn get(&self, key_digest: &Digest256) -> Option<&RadixNode> {
        self.nodes.get(key_digest)
    }

    /// Iterate over all indexed nodes.
    pub fn iter(&self) -> impl Iterator<Item = (&Digest256, &RadixNode)> {
        self.nodes.iter()
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.saturating_add(1);
        self.tick
    }

    /// Insert or refresh a single block node while enforcing write authorization and
    /// monotonic tenant scope boundaries against its parent node.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_node(
        &mut self,
        key: PrefixCacheKey,
        parent_key_digest: Digest256,
        scope: TenantScope,
        auth: &TenantAuthorization,
        token_start: u32,
        token_len: u32,
        byte_size: u64,
        tier: StorageTier,
    ) -> Result<(), PrefixCacheError> {
        auth.authorize_write(&scope)?;

        // Enforce monotonic scope transition if parent exists in the tree.
        if parent_key_digest != Digest256::ZERO {
            if let Some(parent) = self.nodes.get(&parent_key_digest) {
                validate_scope_transition(Some(&parent.scope), &scope)?;
            }
        }

        let now = self.next_tick();
        if let Some(existing) = self.nodes.get_mut(&key.key_digest) {
            if existing.scope != scope {
                return Err(PrefixCacheError::ScopeBoundaryViolation {
                    reason: "existing node scope mismatch on upsert".to_string(),
                });
            }
            existing.last_access_tick = now;
            if tier < existing.residency {
                // Promoted to a hotter tier.
                if tier == StorageTier::Tier0Hbm && existing.residency != StorageTier::Tier0Hbm {
                    self.tier0_bytes = self.tier0_bytes.saturating_add(existing.byte_size);
                }
                existing.residency = tier;
            }
            if tier == StorageTier::Tier1WarmDisk && !existing.in_tier1 {
                existing.in_tier1 = true;
                self.tier1_bytes = self.tier1_bytes.saturating_add(existing.byte_size);
            }
            if tier == StorageTier::Tier2ColdObject && !existing.in_tier2 {
                existing.in_tier2 = true;
                self.tier2_bytes = self.tier2_bytes.saturating_add(existing.byte_size);
            }
            return Ok(());
        }

        let in_tier1 = tier == StorageTier::Tier1WarmDisk;
        let in_tier2 = tier == StorageTier::Tier2ColdObject;
        if tier == StorageTier::Tier0Hbm {
            self.tier0_bytes = self.tier0_bytes.saturating_add(byte_size);
        } else if in_tier1 {
            self.tier1_bytes = self.tier1_bytes.saturating_add(byte_size);
        } else if in_tier2 {
            self.tier2_bytes = self.tier2_bytes.saturating_add(byte_size);
        }

        let digest = key.key_digest;
        self.nodes.insert(
            digest,
            RadixNode {
                key,
                parent_key_digest,
                scope,
                token_len,
                token_start,
                byte_size,
                residency: tier,
                in_tier1,
                in_tier2,
                pin_count: 0,
                last_access_tick: now,
                hit_count: 0,
                children: BTreeSet::new(),
            },
        );

        if parent_key_digest != Digest256::ZERO {
            if let Some(parent) = self.nodes.get_mut(&parent_key_digest) {
                parent.children.insert(digest);
            }
        }
        Ok(())
    }

    /// Record that an existing node is additionally persisted in `tier` (Tier 1 or Tier 2).
    pub fn mark_persisted(&mut self, key_digest: &Digest256, tier: StorageTier) {
        if let Some(node) = self.nodes.get_mut(key_digest) {
            match tier {
                StorageTier::Tier0Hbm => {
                    if node.residency != StorageTier::Tier0Hbm {
                        node.residency = StorageTier::Tier0Hbm;
                        self.tier0_bytes = self.tier0_bytes.saturating_add(node.byte_size);
                    }
                }
                StorageTier::Tier1WarmDisk => {
                    if !node.in_tier1 {
                        node.in_tier1 = true;
                        self.tier1_bytes = self.tier1_bytes.saturating_add(node.byte_size);
                    }
                    if node.residency == StorageTier::Tier2ColdObject {
                        node.residency = StorageTier::Tier1WarmDisk;
                    }
                }
                StorageTier::Tier2ColdObject => {
                    if !node.in_tier2 {
                        node.in_tier2 = true;
                        self.tier2_bytes = self.tier2_bytes.saturating_add(node.byte_size);
                    }
                }
            }
        }
    }

    /// Re-register a node recovered from persistent Tier 1 or Tier 2 storage during cold restart.
    ///
    /// Does not require an active user write session because the snapshot header already
    /// carries the verified 7-field key digest and compatibility digest; reads still enforce
    /// full caller [`TenantAuthorization`] on every lookup.
    #[allow(clippy::too_many_arguments)]
    pub fn register_recovered_node(
        &mut self,
        key: PrefixCacheKey,
        parent_key_digest: Digest256,
        scope: TenantScope,
        token_start: u32,
        token_len: u32,
        byte_size: u64,
        tier: StorageTier,
    ) {
        let now = self.next_tick();
        let digest = key.key_digest;
        if let Some(existing) = self.nodes.get_mut(&digest) {
            match tier {
                StorageTier::Tier1WarmDisk => {
                    if !existing.in_tier1 {
                        existing.in_tier1 = true;
                        self.tier1_bytes = self.tier1_bytes.saturating_add(existing.byte_size);
                    }
                    if existing.residency > StorageTier::Tier1WarmDisk {
                        existing.residency = StorageTier::Tier1WarmDisk;
                    }
                }
                StorageTier::Tier2ColdObject => {
                    if !existing.in_tier2 {
                        existing.in_tier2 = true;
                        self.tier2_bytes = self.tier2_bytes.saturating_add(existing.byte_size);
                    }
                }
                StorageTier::Tier0Hbm => {
                    if existing.residency != StorageTier::Tier0Hbm {
                        existing.residency = StorageTier::Tier0Hbm;
                        self.tier0_bytes = self.tier0_bytes.saturating_add(existing.byte_size);
                    }
                }
            }
            return;
        }

        let in_tier1 = tier == StorageTier::Tier1WarmDisk;
        let in_tier2 = tier == StorageTier::Tier2ColdObject;
        match tier {
            StorageTier::Tier0Hbm => self.tier0_bytes = self.tier0_bytes.saturating_add(byte_size),
            StorageTier::Tier1WarmDisk => {
                self.tier1_bytes = self.tier1_bytes.saturating_add(byte_size);
            }
            StorageTier::Tier2ColdObject => {
                self.tier2_bytes = self.tier2_bytes.saturating_add(byte_size);
            }
        }

        self.nodes.insert(
            digest,
            RadixNode {
                key,
                parent_key_digest,
                scope,
                token_len,
                token_start,
                byte_size,
                residency: tier,
                in_tier1,
                in_tier2,
                pin_count: 0,
                last_access_tick: now,
                hit_count: 0,
                children: BTreeSet::new(),
            },
        );
        if parent_key_digest != Digest256::ZERO {
            if let Some(parent) = self.nodes.get_mut(&parent_key_digest) {
                parent.children.insert(digest);
            }
        }
    }

    /// Rebuild missing parent->child edges after scanning recovered snapshots in arbitrary order.
    pub fn relink_children(&mut self) {
        let pairs: Vec<(Digest256, Digest256)> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.parent_key_digest != Digest256::ZERO)
            .map(|(child, n)| (n.parent_key_digest, *child))
            .collect();
        for (parent, child) in pairs {
            if let Some(p) = self.nodes.get_mut(&parent) {
                p.children.insert(child);
            }
        }
    }

    /// Perform longest-prefix matching across `blocks` for an authorized caller.
    ///
    /// Validates monotonic scope transitions, read ACLs on every block, and 7-field key
    /// identity. Updates LRU access ticks and hit counts on matched nodes when `record_hits` is `true`.
    pub fn match_longest_prefix(
        &mut self,
        compat: &CompatibilityProfile,
        auth: &TenantAuthorization,
        blocks: &[ScopedTokenBlock<'_>],
        record_hits: bool,
    ) -> Result<PrefixMatchResult, PrefixCacheError> {
        let mut chain = TokenPrefixChain::root();
        let mut prev_scope: Option<TenantScope> = None;
        let mut prev_digest = Digest256::ZERO;
        let mut result = PrefixMatchResult::default();
        let mut hit_digests = Vec::new();

        for (idx, blk) in blocks.iter().enumerate() {
            validate_scope_transition(prev_scope.as_ref(), &blk.scope)?;
            auth.authorize_read(&blk.scope)?;
            chain = chain.extend(blk.tokens);
            let key = PrefixCacheKey::derive(compat, &blk.scope, &chain);

            let Some(node) = self.nodes.get(&key.key_digest) else {
                break;
            };
            // Verify strict compatibility, scope, and parent-chain continuity.
            if !node.key.compat_digest.ct_eq(&key.compat_digest)
                || !node.key.scope_digest.ct_eq(&key.scope_digest)
                || node.scope != blk.scope
                || node.parent_key_digest != prev_digest
            {
                break;
            }

            let tier = node.residency;
            match tier {
                StorageTier::Tier0Hbm => result.tier0_blocks += 1,
                StorageTier::Tier1WarmDisk => result.tier1_blocks += 1,
                StorageTier::Tier2ColdObject => result.tier2_blocks += 1,
            }
            result.matched_tokens += node.token_len as usize;
            result.matched_blocks.push(MatchedBlock {
                block_index: idx as u32,
                key: key.clone(),
                scope: blk.scope.clone(),
                tier,
                token_start: node.token_start,
                token_len: node.token_len,
            });
            hit_digests.push(key.key_digest);

            prev_scope = Some(blk.scope.clone());
            prev_digest = key.key_digest;
        }

        if record_hits {
            for digest in hit_digests {
                let now = self.next_tick();
                if let Some(node) = self.nodes.get_mut(&digest) {
                    node.last_access_tick = now;
                    node.hit_count = node.hit_count.saturating_add(1);
                }
            }
        }

        Ok(result)
    }

    /// Increment reference count on a slice of active block digests so they cannot be evicted
    /// from Tier 0 HBM while an inference turn is executing.
    pub fn pin_blocks(&mut self, keys: &[Digest256]) -> Result<(), PrefixCacheError> {
        for k in keys {
            if !self.nodes.contains_key(k) {
                return Err(PrefixCacheError::InvalidPinOperation {
                    reason: format!("cannot pin missing block {}", k.to_hex()),
                });
            }
        }
        for k in keys {
            if let Some(node) = self.nodes.get_mut(k) {
                node.pin_count = node.pin_count.saturating_add(1);
            }
        }
        Ok(())
    }

    /// Decrement reference count on previously pinned blocks.
    pub fn unpin_blocks(&mut self, keys: &[Digest256]) -> Result<(), PrefixCacheError> {
        for k in keys {
            match self.nodes.get(k) {
                Some(n) if n.pin_count > 0 => {}
                Some(_) => {
                    return Err(PrefixCacheError::InvalidPinOperation {
                        reason: format!("block {} has pin_count == 0", k.to_hex()),
                    });
                }
                None => {
                    return Err(PrefixCacheError::InvalidPinOperation {
                        reason: format!("cannot unpin missing block {}", k.to_hex()),
                    });
                }
            }
        }
        for k in keys {
            if let Some(node) = self.nodes.get_mut(k) {
                node.pin_count -= 1;
            }
        }
        Ok(())
    }

    /// Select and demote unpinned Tier 0 leaf blocks in strict LRU order until
    /// `self.tier0_bytes <= budget_bytes` (or no unpinned leaves remain).
    ///
    /// When `demote_to_tier1` is `true`, unbacked Tier 0 victims remain in the index
    /// (transitioned to [`StorageTier::Tier1WarmDisk`]) so the storage layer can flush
    /// them to Tier 1 without losing radix metadata.
    ///
    /// Returns `(key_digest, scope, byte_size, already_backed)` for each victim.
    pub fn evict_tier0_to_budget(
        &mut self,
        budget_bytes: u64,
        demote_to_tier1: bool,
    ) -> Vec<(Digest256, TenantScope, u64, bool)> {
        let mut evicted = Vec::new();
        while self.tier0_bytes > budget_bytes {
            // Find the least-recently-used unpinned Tier 0 node that has NO Tier 0 children.
            let victim = self
                .nodes
                .iter()
                .filter(|(_, node)| {
                    node.residency == StorageTier::Tier0Hbm
                        && node.pin_count == 0
                        && !node.children.iter().any(|c| {
                            self.nodes
                                .get(c)
                                .is_some_and(|cn| cn.residency == StorageTier::Tier0Hbm)
                        })
                })
                .min_by_key(|(k, node)| (node.last_access_tick, *k))
                .map(|(k, _)| *k);

            let Some(victim_key) = victim else {
                break;
            };

            let (parent, scope, byte_size, already_backed, keep_in_index) = {
                let node = self.nodes.get_mut(&victim_key).expect("victim exists");
                self.tier0_bytes = self.tier0_bytes.saturating_sub(node.byte_size);
                let scope = node.scope.clone();
                let byte_size = node.byte_size;
                if node.in_tier1 {
                    node.residency = StorageTier::Tier1WarmDisk;
                    (node.parent_key_digest, scope, byte_size, true, true)
                } else if demote_to_tier1 {
                    node.residency = StorageTier::Tier1WarmDisk;
                    (node.parent_key_digest, scope, byte_size, false, true)
                } else if node.in_tier2 {
                    node.residency = StorageTier::Tier2ColdObject;
                    (node.parent_key_digest, scope, byte_size, true, true)
                } else {
                    (node.parent_key_digest, scope, byte_size, false, false)
                }
            };

            if !keep_in_index {
                let has_any_children = self
                    .nodes
                    .get(&victim_key)
                    .is_some_and(|n| !n.children.is_empty());
                if !has_any_children {
                    self.nodes.remove(&victim_key);
                    if parent != Digest256::ZERO {
                        if let Some(p) = self.nodes.get_mut(&parent) {
                            p.children.remove(&victim_key);
                        }
                    }
                }
            }
            evicted.push((victim_key, scope, byte_size, already_backed));
        }
        evicted
    }

    /// Select and evict unpinned Tier 1 warm-disk leaves in LRU order until
    /// `self.tier1_bytes <= budget_bytes`. Returns evicted key digests.
    pub fn evict_tier1_to_budget(&mut self, budget_bytes: u64) -> Vec<Digest256> {
        let mut evicted = Vec::new();
        while self.tier1_bytes > budget_bytes {
            let victim = self
                .nodes
                .iter()
                .filter(|(_, node)| {
                    node.in_tier1
                        && node.pin_count == 0
                        && node.residency != StorageTier::Tier0Hbm
                        && !node
                            .children
                            .iter()
                            .any(|c| self.nodes.get(c).is_some_and(|cn| cn.in_tier1))
                })
                .min_by_key(|(k, node)| (node.last_access_tick, *k))
                .map(|(k, _)| *k);

            let Some(victim_key) = victim else {
                break;
            };

            let (parent, drop_node) = {
                let node = self.nodes.get_mut(&victim_key).expect("victim exists");
                node.in_tier1 = false;
                self.tier1_bytes = self.tier1_bytes.saturating_sub(node.byte_size);
                if node.in_tier2 {
                    node.residency = StorageTier::Tier2ColdObject;
                    (node.parent_key_digest, false)
                } else {
                    (node.parent_key_digest, node.children.is_empty())
                }
            };

            if drop_node {
                self.nodes.remove(&victim_key);
                if parent != Digest256::ZERO {
                    if let Some(p) = self.nodes.get_mut(&parent) {
                        p.children.remove(&victim_key);
                    }
                }
            }
            evicted.push(victim_key);
        }
        evicted
    }

    /// Wipe all volatile Tier 0 HBM state (simulating a worker crash/restart), keeping
    /// only blocks backed by persistent Tier 1 or Tier 2 storage.
    pub fn simulate_volatile_hbm_reset(&mut self) -> usize {
        let mut dropped_hbm = 0usize;
        let keys: Vec<Digest256> = self.nodes.keys().copied().collect();
        let mut to_remove = Vec::new();

        for k in keys {
            if let Some(node) = self.nodes.get_mut(&k) {
                node.pin_count = 0;
                if node.residency == StorageTier::Tier0Hbm {
                    dropped_hbm += 1;
                    self.tier0_bytes = self.tier0_bytes.saturating_sub(node.byte_size);
                    if node.in_tier1 {
                        node.residency = StorageTier::Tier1WarmDisk;
                    } else if node.in_tier2 {
                        node.residency = StorageTier::Tier2ColdObject;
                    } else {
                        to_remove.push(k);
                    }
                }
            }
        }

        for k in to_remove {
            if let Some(node) = self.nodes.remove(&k) {
                if node.parent_key_digest != Digest256::ZERO {
                    if let Some(parent) = self.nodes.get_mut(&node.parent_key_digest) {
                        parent.children.remove(&k);
                    }
                }
            }
        }
        self.tier0_bytes = 0;
        dropped_hbm
    }
}
