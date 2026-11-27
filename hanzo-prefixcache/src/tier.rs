// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! 3-Tier persistent prefix storage hierarchy:
//! - **Tier 0 (`Hot HBM / VRAM`)**: In-memory budget-bounded snapshot pool with pin protection.
//! - **Tier 1 (`Warm Host DRAM / Local SSD`)**: Real filesystem-backed versioned `.hpcs` snapshots
//!   with atomic write-then-rename, CRC32 + SHA-256 verification, and cold-restart reload.
//! - **Tier 2 (`Cold Shared Object Store`)**: Cross-worker shareable backing directory with
//!   optional bandwidth/crossover admission gating to avoid TTFT regressions on slow links.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::crypto::Digest256;
use crate::error::PrefixCacheError;
use crate::key::{
    CompatibilityProfile, PrefixCacheKey, ScopedTokenBlock, TenantAuthorization, TenantScope,
};
use crate::radix::{PrefixMatchResult, RadixPrefixIndex, StorageTier};
use crate::snapshot::{PrefixBlockSnapshot, SnapshotHeader};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Persistence admission policy governing when blocks are flushed to Tier 1 and Tier 2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PersistencePolicy {
    /// Volatile HBM-only RadixAttention (no Tier 1 / Tier 2 writes; lost on restart).
    HbmOnly,
    /// Naive every-turn synchronous disk persistence without selectivity.
    NaivePersistEveryTurn,
    /// Selective multi-tier radix policy:
    /// - Persists shared scopes (`PublicSystem`, `OrgShared`) immediately to Tier 1 & Tier 2.
    /// - Persists private blocks to Tier 1 when demoted from Tier 0 or when hit count reaches
    ///   `private_persist_min_hits` (or within `private_eager_prefix_blocks`).
    /// - Promotes cross-worker reusable prefixes to Tier 2 when configured.
    SelectiveRadix {
        /// Persist private blocks eagerly up to this block depth (e.g. active coding context).
        private_eager_prefix_blocks: u32,
        /// Persist any block once its hit count reaches this threshold.
        private_persist_min_hits: u64,
        /// Demote evicted Tier 0 blocks into Tier 1 instead of dropping them.
        demote_on_tier0_evict: bool,
        /// Publish shared prefixes and demoted Tier 1 blocks to Tier 2 shared object store.
        enable_tier2_sharing: bool,
    },
}

/// Configuration for a [`TieredPrefixStore`].
#[derive(Clone, Debug)]
pub struct TieredStoreConfig {
    /// Accounted byte budget for Tier 0 (Hot HBM / VRAM).
    pub tier0_budget_bytes: u64,
    /// Accounted byte budget for Tier 1 (Warm Host DRAM / Local SSD).
    pub tier1_budget_bytes: u64,
    /// Optional local filesystem directory backing Tier 1 warm snapshots.
    pub tier1_dir: Option<PathBuf>,
    /// Optional shared filesystem directory backing Tier 2 cold object-store snapshots.
    pub tier2_dir: Option<PathBuf>,
    /// Persistence admission policy.
    pub policy: PersistencePolicy,
    /// Whether to enforce the Tier 2 bandwidth crossover admission guard during fetches.
    pub enable_tier2_bandwidth_guard: bool,
    /// Available Tier 2 cold-link bandwidth in bytes/sec (used by the crossover guard).
    pub tier2_link_bandwidth_bps: f64,
    /// Tier 2 round-trip latency in seconds (used by the crossover guard).
    pub tier2_rtt_sec: f64,
    /// Baseline local prefill time per token in seconds (used by the crossover guard).
    pub local_prefill_sec_per_token: f64,
}

impl TieredStoreConfig {
    /// Evaluate whether fetching a snapshot of `accounted_bytes` covering `token_len` tokens
    /// from Tier 2 is faster than locally recomputing prefill.
    #[must_use]
    pub fn tier2_fetch_beats_recompute(&self, accounted_bytes: u64, token_len: u32) -> bool {
        if !self.enable_tier2_bandwidth_guard {
            return true;
        }
        if self.tier2_link_bandwidth_bps <= 0.0 {
            return false;
        }
        let fetch_sec =
            self.tier2_rtt_sec + (accounted_bytes as f64 / self.tier2_link_bandwidth_bps);
        let recompute_sec = f64::from(token_len) * self.local_prefill_sec_per_token;
        fetch_sec < recompute_sec
    }
}

/// Operational telemetry counters collected by [`TieredPrefixStore`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TierStoreMetrics {
    /// Block hits satisfied directly from Tier 0 (HBM).
    pub tier0_hits: u64,
    /// Block hits satisfied from Tier 1 (Warm DRAM/SSD) with verified decode.
    pub tier1_hits: u64,
    /// Block hits satisfied from Tier 2 (Cold Object Store) with verified decode.
    pub tier2_hits: u64,
    /// Tier 2 candidates intentionally skipped because bandwidth guard preferred local recompute.
    pub tier2_guard_skips: u64,
    /// Total prefix block misses requiring fresh prefill.
    pub misses: u64,
    /// Promotions into Tier 0 HBM from Tier 1 or Tier 2.
    pub promotions_to_tier0: u64,
    /// Demotions from Tier 0 HBM into Tier 1.
    pub demotions_to_tier1: u64,
    /// Evictions from Tier 0 HBM (dropped or demoted).
    pub tier0_evictions: u64,
    /// Evictions from Tier 1 warm disk.
    pub tier1_evictions: u64,
    /// Snapshots rejected due to CRC32, SHA-256, version, or key/compat failure.
    pub corrupt_rejections: u64,
    /// Total real serialized bytes written to Tier 1 filesystem.
    pub tier1_bytes_written: u64,
    /// Total real serialized bytes read from Tier 1 filesystem.
    pub tier1_bytes_read: u64,
    /// Total real serialized bytes written to Tier 2 filesystem.
    pub tier2_bytes_written: u64,
    /// Total real serialized bytes read from Tier 2 filesystem.
    pub tier2_bytes_read: u64,
    /// Cumulative real host wall-clock nanoseconds spent on Tier 1/2 I/O + verification.
    pub io_and_verify_ns: u64,
}

/// Metadata sidecar entry written alongside persistent snapshots so cold restarts
/// can reconstruct scope-aware radix trees without exposing private payloads.
const SIDECAR_MAGIC: [u8; 4] = *b"HPSM";

fn encode_scope_sidecar(scope: &TenantScope, accounted_bytes: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(32);
    buf.extend_from_slice(&SIDECAR_MAGIC);
    buf.extend_from_slice(&accounted_bytes.to_le_bytes());
    match scope {
        TenantScope::PublicSystem => {
            buf.push(1);
            buf.extend_from_slice(&0u32.to_le_bytes());
        }
        TenantScope::OrgShared { org_id } => {
            buf.push(2);
            buf.extend_from_slice(&(org_id.len() as u32).to_le_bytes());
            buf.extend_from_slice(org_id.as_bytes());
        }
        TenantScope::TenantPrivate { tenant_id } => {
            buf.push(3);
            buf.extend_from_slice(&(tenant_id.len() as u32).to_le_bytes());
            buf.extend_from_slice(tenant_id.as_bytes());
        }
    }
    buf
}

fn decode_scope_sidecar(buf: &[u8]) -> Option<(TenantScope, u64)> {
    if buf.len() < 17 || buf[0..4] != SIDECAR_MAGIC {
        return None;
    }
    let accounted_bytes = u64::from_le_bytes([
        buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
    ]);
    let tag = buf[12];
    let len = u32::from_le_bytes([buf[13], buf[14], buf[15], buf[16]]) as usize;
    if buf.len() != 17 + len {
        return None;
    }
    let text = std::str::from_utf8(&buf[17..]).ok()?.to_string();
    let scope = match tag {
        1 => TenantScope::PublicSystem,
        2 => TenantScope::OrgShared { org_id: text },
        3 => TenantScope::TenantPrivate { tenant_id: text },
        _ => return None,
    };
    Some((scope, accounted_bytes))
}

/// 3-Tier persistent prefix cache engine combining [`RadixPrefixIndex`] with
/// Tier 0 in-memory HBM pool, Tier 1 local filesystem warm cache, and Tier 2 shared object store.
#[derive(Debug)]
pub struct TieredPrefixStore {
    config: TieredStoreConfig,
    index: RadixPrefixIndex,
    tier0_pool: HashMap<Digest256, PrefixBlockSnapshot>,
    metrics: TierStoreMetrics,
}

impl TieredPrefixStore {
    /// Create a new [`TieredPrefixStore`], creating Tier 1 / Tier 2 directories if configured,
    /// and automatically scanning any existing verified snapshots for cold-restart recovery.
    pub fn open(config: TieredStoreConfig) -> Result<Self, PrefixCacheError> {
        if let Some(dir) = &config.tier1_dir {
            fs::create_dir_all(dir)?;
        }
        if let Some(dir) = &config.tier2_dir {
            fs::create_dir_all(dir)?;
        }

        let mut store = Self {
            config,
            index: RadixPrefixIndex::new(),
            tier0_pool: HashMap::new(),
            metrics: TierStoreMetrics::default(),
        };
        store.recover_from_persistent_tiers()?;
        Ok(store)
    }

    /// Borrow the active configuration.
    #[must_use]
    pub const fn config(&self) -> &TieredStoreConfig {
        &self.config
    }

    /// Borrow the underlying [`RadixPrefixIndex`].
    #[must_use]
    pub const fn index(&self) -> &RadixPrefixIndex {
        &self.index
    }

    /// Borrow accumulated telemetry metrics.
    #[must_use]
    pub const fn metrics(&self) -> &TierStoreMetrics {
        &self.metrics
    }

    /// Compute the filesystem path for a snapshot in `base_dir`.
    #[must_use]
    pub fn snapshot_path(base_dir: &Path, key_digest: &Digest256) -> PathBuf {
        base_dir
            .join(key_digest.shard_prefix())
            .join(format!("{}.hpcs", key_digest.to_hex()))
    }

    fn sidecar_path(base_dir: &Path, key_digest: &Digest256) -> PathBuf {
        base_dir
            .join(key_digest.shard_prefix())
            .join(format!("{}.hpsm", key_digest.to_hex()))
    }

    /// Scan Tier 1 and Tier 2 directories on cold startup, verifying every snapshot header
    /// and CRC32/SHA-256 before restoring it into the radix index.
    pub fn recover_from_persistent_tiers(&mut self) -> Result<usize, PrefixCacheError> {
        let mut recovered = 0usize;
        if let Some(dir) = self.config.tier1_dir.clone() {
            recovered += self.scan_tier_dir(&dir, StorageTier::Tier1WarmDisk)?;
        }
        if let Some(dir) = self.config.tier2_dir.clone() {
            recovered += self.scan_tier_dir(&dir, StorageTier::Tier2ColdObject)?;
        }
        self.index.relink_children();
        Ok(recovered)
    }

    fn scan_tier_dir(&mut self, dir: &Path, tier: StorageTier) -> Result<usize, PrefixCacheError> {
        if !dir.exists() {
            return Ok(0);
        }
        let mut count = 0usize;
        for shard_entry in fs::read_dir(dir)? {
            let shard_entry = shard_entry?;
            let shard_path = shard_entry.path();
            if !shard_path.is_dir() {
                continue;
            }
            for file_entry in fs::read_dir(shard_path)? {
                let file_entry = file_entry?;
                let path = file_entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("hpcs") {
                    continue;
                }
                let start = Instant::now();
                let bytes = match fs::read(&path) {
                    Ok(b) => b,
                    Err(_) => {
                        self.metrics.corrupt_rejections += 1;
                        continue;
                    }
                };
                match PrefixBlockSnapshot::decode_self_consistent(&bytes) {
                    Ok(snap) => {
                        let meta_path = path.with_extension("hpsm");
                        let meta_bytes = fs::read(&meta_path).ok();
                        let Some((scope, accounted_bytes)) =
                            meta_bytes.as_deref().and_then(decode_scope_sidecar)
                        else {
                            self.metrics.corrupt_rejections += 1;
                            continue;
                        };
                        let hdr = &snap.header;
                        // Reconstruct PrefixCacheKey metadata from verified header + sidecar scope.
                        let key = PrefixCacheKey {
                            compat_digest: hdr.compat_digest,
                            scope_digest: scope.canonical_digest(),
                            token_chain_digest: Digest256::ZERO,
                            block_index: hdr.token_start / hdr.block_size.max(1),
                            key_digest: hdr.key_digest,
                        };
                        self.index.register_recovered_node(
                            key,
                            hdr.parent_block_digest,
                            scope,
                            hdr.token_start,
                            hdr.token_len,
                            accounted_bytes,
                            tier,
                        );
                        count += 1;
                    }
                    Err(_) => {
                        // Fail-closed: corrupt or version-mismatched snapshot is ignored.
                        self.metrics.corrupt_rejections += 1;
                    }
                }
                self.metrics.io_and_verify_ns = self
                    .metrics
                    .io_and_verify_ns
                    .saturating_add(start.elapsed().as_nanos() as u64);
            }
        }
        Ok(count)
    }

    /// Refresh cross-worker visibility by importing any newly published Tier 2 snapshots
    /// written by peer workers into the local radix index.
    pub fn sync_from_tier2(&mut self) -> Result<usize, PrefixCacheError> {
        let Some(dir) = self.config.tier2_dir.clone() else {
            return Ok(0);
        };
        let n = self.scan_tier_dir(&dir, StorageTier::Tier2ColdObject)?;
        self.index.relink_children();
        Ok(n)
    }

    /// Match the longest authorized prefix for `blocks` across all tiers.
    pub fn match_prefix(
        &mut self,
        compat: &CompatibilityProfile,
        auth: &TenantAuthorization,
        blocks: &[ScopedTokenBlock<'_>],
    ) -> Result<PrefixMatchResult, PrefixCacheError> {
        self.index.match_longest_prefix(compat, auth, blocks, false)
    }

    /// Load and fail-closed verify a matched block snapshot from Tier 0, Tier 1, or Tier 2.
    ///
    /// Evaluates the Tier 2 bandwidth admission guard using the single block's accounted size.
    /// For multi-block prefix fetches where one RTT and one GDN checkpoint are amortized over
    /// the entire matched prefix, prefer [`Self::load_verified_block_with_batch`].
    pub fn load_verified_block(
        &mut self,
        key: &PrefixCacheKey,
        compat: &CompatibilityProfile,
        scope: &TenantScope,
        auth: &TenantAuthorization,
        promote_to_tier0: bool,
    ) -> Result<Option<(PrefixBlockSnapshot, StorageTier)>, PrefixCacheError> {
        self.load_verified_block_with_batch(key, compat, scope, auth, promote_to_tier0, None)
    }

    /// Load and fail-closed verify a matched block snapshot with optional pipelined batch context
    /// `(total_transfer_bytes, total_tokens)` for the Tier 2 bandwidth crossover guard.
    pub fn load_verified_block_with_batch(
        &mut self,
        key: &PrefixCacheKey,
        compat: &CompatibilityProfile,
        scope: &TenantScope,
        auth: &TenantAuthorization,
        promote_to_tier0: bool,
        tier2_batch_context: Option<(u64, u32)>,
    ) -> Result<Option<(PrefixBlockSnapshot, StorageTier)>, PrefixCacheError> {
        key.verify_access(compat, scope, auth)?;

        // Tier 0: Hot HBM in-memory lookup.
        if let Some(snap) = self.tier0_pool.get(&key.key_digest).cloned() {
            self.metrics.tier0_hits += 1;
            if let Some(node) = self.index.get(&key.key_digest) {
                if !node.key.key_digest.ct_eq(&key.key_digest) {
                    return Err(PrefixCacheError::KeyMismatch);
                }
            }
            return Ok(Some((snap, StorageTier::Tier0Hbm)));
        }

        // Tier 1: Warm local filesystem lookup with full CRC32 + SHA-256 verification.
        if let Some(t1_dir) = self.config.tier1_dir.clone() {
            let path = Self::snapshot_path(&t1_dir, &key.key_digest);
            if path.exists() {
                let start = Instant::now();
                let bytes = fs::read(&path)?;
                self.metrics.tier1_bytes_read = self
                    .metrics
                    .tier1_bytes_read
                    .saturating_add(bytes.len() as u64);
                let decoded = PrefixBlockSnapshot::decode_verified(&bytes, key, compat);
                self.metrics.io_and_verify_ns = self
                    .metrics
                    .io_and_verify_ns
                    .saturating_add(start.elapsed().as_nanos() as u64);
                match decoded {
                    Ok(snap) => {
                        self.metrics.tier1_hits += 1;
                        if promote_to_tier0 {
                            self.promote_into_tier0(key, scope, auth, &snap)?;
                        }
                        return Ok(Some((snap, StorageTier::Tier1WarmDisk)));
                    }
                    Err(err) => {
                        self.metrics.corrupt_rejections += 1;
                        return Err(err);
                    }
                }
            }
        }

        // Tier 2: Cold shared object store lookup with optional bandwidth admission guard.
        if let Some(t2_dir) = self.config.tier2_dir.clone() {
            let path = Self::snapshot_path(&t2_dir, &key.key_digest);
            if path.exists() {
                let (check_bytes, check_tokens) = tier2_batch_context.unwrap_or_else(|| {
                    self.index
                        .get(&key.key_digest)
                        .map_or((0, compat.runtime_config.block_size), |n| {
                            (n.byte_size, n.token_len)
                        })
                });
                if check_bytes > 0
                    && !self
                        .config
                        .tier2_fetch_beats_recompute(check_bytes, check_tokens)
                {
                    self.metrics.tier2_guard_skips += 1;
                    self.metrics.misses += 1;
                    return Ok(None);
                }

                let start = Instant::now();
                let bytes = fs::read(&path)?;
                self.metrics.tier2_bytes_read = self
                    .metrics
                    .tier2_bytes_read
                    .saturating_add(bytes.len() as u64);
                let decoded = PrefixBlockSnapshot::decode_verified(&bytes, key, compat);
                self.metrics.io_and_verify_ns = self
                    .metrics
                    .io_and_verify_ns
                    .saturating_add(start.elapsed().as_nanos() as u64);
                match decoded {
                    Ok(snap) => {
                        self.metrics.tier2_hits += 1;
                        if promote_to_tier0 {
                            self.promote_into_tier0(key, scope, auth, &snap)?;
                        }
                        return Ok(Some((snap, StorageTier::Tier2ColdObject)));
                    }
                    Err(err) => {
                        self.metrics.corrupt_rejections += 1;
                        return Err(err);
                    }
                }
            }
        }

        self.metrics.misses += 1;
        Ok(None)
    }

    fn promote_into_tier0(
        &mut self,
        key: &PrefixCacheKey,
        scope: &TenantScope,
        auth: &TenantAuthorization,
        snap: &PrefixBlockSnapshot,
    ) -> Result<(), PrefixCacheError> {
        let accounted_bytes = self
            .index
            .get(&key.key_digest)
            .map_or(snap.serialized_len() as u64, |n| n.byte_size);
        // Only register node if missing; otherwise mark promoted in Tier 0.
        if self.index.get(&key.key_digest).is_none() {
            self.index.register_recovered_node(
                key.clone(),
                snap.header.parent_block_digest,
                scope.clone(),
                snap.header.token_start,
                snap.header.token_len,
                accounted_bytes,
                StorageTier::Tier0Hbm,
            );
        } else {
            self.index
                .mark_persisted(&key.key_digest, StorageTier::Tier0Hbm);
        }
        let _ = auth;
        self.tier0_pool.insert(key.key_digest, snap.clone());
        self.metrics.promotions_to_tier0 += 1;
        self.enforce_tier0_budget()?;
        Ok(())
    }

    /// Insert a newly computed block snapshot into Tier 0 and persist to Tier 1/Tier 2
    /// according to the active [`PersistencePolicy`].
    pub fn put_block(
        &mut self,
        key: PrefixCacheKey,
        scope: TenantScope,
        auth: &TenantAuthorization,
        snapshot: PrefixBlockSnapshot,
        accounted_bytes: u64,
    ) -> Result<(), PrefixCacheError> {
        auth.authorize_write(&scope)?;

        let parent = snapshot.header.parent_block_digest;
        let token_start = snapshot.header.token_start;
        let token_len = snapshot.header.token_len;
        let block_idx = key.block_index;
        let digest = key.key_digest;

        self.index.upsert_node(
            key,
            parent,
            scope.clone(),
            auth,
            token_start,
            token_len,
            accounted_bytes,
            StorageTier::Tier0Hbm,
        )?;
        self.tier0_pool.insert(digest, snapshot.clone());

        // Apply persistence policy for Tier 1 & Tier 2 writes.
        let (write_t1, write_t2) = match &self.config.policy {
            PersistencePolicy::HbmOnly => (false, false),
            PersistencePolicy::NaivePersistEveryTurn => (true, false),
            PersistencePolicy::SelectiveRadix {
                private_eager_prefix_blocks,
                private_persist_min_hits,
                enable_tier2_sharing,
                ..
            } => {
                if scope.is_shared() {
                    (true, *enable_tier2_sharing)
                } else {
                    let hits = self.index.get(&digest).map_or(0, |n| n.hit_count);
                    let eager = block_idx < *private_eager_prefix_blocks
                        || hits >= *private_persist_min_hits;
                    (eager, eager && *enable_tier2_sharing)
                }
            }
        };

        if write_t1 {
            self.write_to_tier1(&digest, &scope, &snapshot, accounted_bytes)?;
        }
        if write_t2 {
            self.write_to_tier2(&digest, &scope, &snapshot, accounted_bytes)?;
        }

        self.enforce_tier0_budget()?;
        self.enforce_tier1_budget()?;
        Ok(())
    }

    /// Pin active blocks in Tier 0 HBM while an inference turn is executing.
    pub fn pin_blocks(&mut self, keys: &[Digest256]) -> Result<(), PrefixCacheError> {
        self.index.pin_blocks(keys)
    }

    /// Release active block pins after an inference turn completes.
    pub fn unpin_blocks(&mut self, keys: &[Digest256]) -> Result<(), PrefixCacheError> {
        self.index.unpin_blocks(keys)
    }

    /// Simulate a worker restart: clears all volatile Tier 0 HBM snapshots while preserving
    /// Tier 1 warm SSD and Tier 2 shared object-store snapshots.
    pub fn simulate_worker_restart(&mut self) -> usize {
        self.tier0_pool.clear();
        self.index.simulate_volatile_hbm_reset()
    }

    fn enforce_tier0_budget(&mut self) -> Result<(), PrefixCacheError> {
        let demote = matches!(
            self.config.policy,
            PersistencePolicy::SelectiveRadix {
                demote_on_tier0_evict: true,
                ..
            }
        ) && self.config.tier1_dir.is_some();

        let victims = self
            .index
            .evict_tier0_to_budget(self.config.tier0_budget_bytes, demote);
        for (victim_key, scope, byte_size, already_backed) in victims {
            self.metrics.tier0_evictions += 1;
            if let Some(snap) = self.tier0_pool.remove(&victim_key) {
                if demote && !already_backed {
                    self.write_to_tier1(&victim_key, &scope, &snap, byte_size)?;
                    self.metrics.demotions_to_tier1 += 1;
                }
            }
        }
        Ok(())
    }

    fn enforce_tier1_budget(&mut self) -> Result<(), PrefixCacheError> {
        let Some(t1_dir) = self.config.tier1_dir.clone() else {
            return Ok(());
        };
        let victims = self
            .index
            .evict_tier1_to_budget(self.config.tier1_budget_bytes);
        for victim_key in victims {
            self.metrics.tier1_evictions += 1;
            let _ = fs::remove_file(Self::snapshot_path(&t1_dir, &victim_key));
            let _ = fs::remove_file(Self::sidecar_path(&t1_dir, &victim_key));
        }
        Ok(())
    }

    fn write_to_tier1(
        &mut self,
        key_digest: &Digest256,
        scope: &TenantScope,
        snapshot: &PrefixBlockSnapshot,
        accounted_bytes: u64,
    ) -> Result<(), PrefixCacheError> {
        let Some(dir) = self.config.tier1_dir.clone() else {
            return Ok(());
        };
        let written =
            self.atomic_write_snapshot(&dir, key_digest, scope, snapshot, accounted_bytes)?;
        self.metrics.tier1_bytes_written = self.metrics.tier1_bytes_written.saturating_add(written);
        self.index
            .mark_persisted(key_digest, StorageTier::Tier1WarmDisk);
        Ok(())
    }

    fn write_to_tier2(
        &mut self,
        key_digest: &Digest256,
        scope: &TenantScope,
        snapshot: &PrefixBlockSnapshot,
        accounted_bytes: u64,
    ) -> Result<(), PrefixCacheError> {
        let Some(dir) = self.config.tier2_dir.clone() else {
            return Ok(());
        };
        let written =
            self.atomic_write_snapshot(&dir, key_digest, scope, snapshot, accounted_bytes)?;
        self.metrics.tier2_bytes_written = self.metrics.tier2_bytes_written.saturating_add(written);
        self.index
            .mark_persisted(key_digest, StorageTier::Tier2ColdObject);
        Ok(())
    }

    fn atomic_write_snapshot(
        &mut self,
        base_dir: &Path,
        key_digest: &Digest256,
        scope: &TenantScope,
        snapshot: &PrefixBlockSnapshot,
        accounted_bytes: u64,
    ) -> Result<u64, PrefixCacheError> {
        let start = Instant::now();
        let snap_path = Self::snapshot_path(base_dir, key_digest);
        if let Some(parent) = snap_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let encoded = snapshot.encode();
        let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_snap = snap_path.with_extension(format!("hpcs.tmp.{seq}"));
        {
            let mut f = File::create(&tmp_snap)?;
            f.write_all(&encoded)?;
        }
        fs::rename(&tmp_snap, &snap_path)?;

        let meta_path = Self::sidecar_path(base_dir, key_digest);
        let meta_bytes = encode_scope_sidecar(scope, accounted_bytes);
        let tmp_meta = meta_path.with_extension(format!("hpsm.tmp.{seq}"));
        {
            let mut f = File::create(&tmp_meta)?;
            f.write_all(&meta_bytes)?;
        }
        fs::rename(&tmp_meta, &meta_path)?;

        self.metrics.io_and_verify_ns = self
            .metrics
            .io_and_verify_ns
            .saturating_add(start.elapsed().as_nanos() as u64);
        Ok((encoded.len() + meta_bytes.len()) as u64)
    }

    /// Inspect only the verified [`SnapshotHeader`] of a persisted Tier 1 or Tier 2 snapshot
    /// without loading payloads into Tier 0.
    pub fn inspect_persisted_header(
        base_dir: &Path,
        key_digest: &Digest256,
    ) -> Result<SnapshotHeader, PrefixCacheError> {
        let path = Self::snapshot_path(base_dir, key_digest);
        let bytes = fs::read(&path)?;
        SnapshotHeader::decode_verified(&bytes)
    }
}
