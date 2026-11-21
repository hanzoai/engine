// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sparse SSD KV offload planning for the KV fabric (HIP-1345 §2.1,
//! Bounty #11 / engine#92, SolidAttention
//! <https://www.usenix.org/conference/fast26/presentation/zheng>).
//!
//! Local Zen serving on memory-constrained PCs keeps the bulk of a long
//! context's KV cache on SSD. SolidAttention's co-design has three legs:
//! coarse-grained KV blocks (bandwidth), speculative prefetching of the
//! blocks temporal locality says come next (hiding SSD latency behind
//! decode compute), and fine-grained compute/I-O orchestration (overlap).
//! This module is the fabric-level *policy* layer for legs one and two
//! with a bounded overlap model:
//!
//! - [`BLOCK_BYTES`]: the coarse block size the pool manages.
//! - [`salted_block_key`]: tenant-salted block identity (HIP-1345 §1.1 —
//!   CacheSalt isolation MUST hold at the offload tier too).
//! - [`SsdOffloadPool`]: a staging-budgeted block pool — `prefetch` stages
//!   predicted blocks under an LRU budget, `fetch` serves from staging or
//!   pays the SSD read.
//! - [`overlap_stall_ns`]: the overlap model — I/O issued during a decode
//!   step hides under that step's compute; only the excess stalls.
//! - [`predict_next`]: the temporal-locality predictor (sequential
//!   successors + the most recent back-reference), deliberately simple.
//!
//! What is deliberately NOT here: sparse attention *algorithms* (which
//! blocks attention actually reads is the engine's job — the bench feeds
//! this module a deterministic access trace), attention accuracy effects,
//! and real device I/O. See the PR's HIP-1345 §2.1 update for the full
//! out-of-scope statement.

/// Coarse-grained offload block size: 64 MiB. At ~128 KiB of KV per token
/// (32 layers x 8 KV heads x 128 dim x K+V x fp16), one block is ~512
/// tokens; a 128k-token context is 256 blocks (16 GiB of KV).
pub const BLOCK_BYTES: u64 = 64 * 1024 * 1024;

/// SplitMix64 — stdlib-only, deterministic key mixing so that two tenants
/// with identical block indices get disjoint keys (HIP-1345 §1.1).
pub fn salted_block_key(tenant_salt: u64, block_index: u64) -> u64 {
    let mut z = tenant_salt ^ block_index.wrapping_mul(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Pool configuration: device and staging rates/budgets.
#[derive(Debug, Clone, Copy)]
pub struct OffloadConfig {
    /// SSD sequential read bandwidth, bytes per second.
    pub ssd_read_bps: u64,
    /// Per-block I/O setup cost (queue + command), nanoseconds.
    pub ssd_io_setup_ns: u64,
    /// Host staging budget for prefetched blocks, bytes.
    pub staging_bytes: u64,
    /// Attention compute per resident block per decode step, nanoseconds.
    pub decode_ns_per_block: u64,
}

impl Default for OffloadConfig {
    fn default() -> Self {
        // A consumer NVMe SSD (SolidAttention's AIPC target class):
        // ~2 GB/s read, ~50 µs setup per block; 512 MiB staging; AIPC-class
        // iGPU attention over one 64 MiB (~512-token) block ≈ 6 ms per
        // decode step — the regime where prefetch I/O can hide under
        // compute. Engine integration must replace these with measured
        // Hanzo-hardware rates.
        Self {
            ssd_read_bps: 2_000_000_000,
            ssd_io_setup_ns: 50_000,
            staging_bytes: 512 * 1024 * 1024,
            decode_ns_per_block: 6_000_000,
        }
    }
}

/// Temporal-locality predictor: from the most recent access window, predict
/// the next block indices the attention path is likely to touch. Bounded
/// and deterministic: the sequential successors of the current head plus
/// the last distinct back-reference (sliding-window/sink shape). Returns
/// block indices, deduplicated, most-likely first, at most `k` entries.
pub fn predict_next(recent: &[u64], k: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(k);
    let push = |v: u64, out: &mut Vec<u64>| {
        if !out.contains(&v) && out.len() < k {
            out.push(v);
        }
    };
    // Sequential successor of the newest access (decode advances forward).
    if let Some(&head) = recent.last() {
        push(head + 1, &mut out);
    }
    // The last distinct back-reference's successor (window slides).
    for &r in recent.iter().rev() {
        if out.contains(&(r + 1)) {
            continue;
        }
        // treat r as a window start: its successor block is next-likely
        push(r + 1, &mut out);
        if out.len() >= k {
            break;
        }
    }
    // Last resort: keep head itself (re-read within a step window).
    if let Some(&head) = recent.last() {
        push(head, &mut out);
    }
    out
}

/// One staged-block slot bookkeeping entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Slot {
    key: u64,
    bytes: u64,
    last_tick: u64,
    used_since_prefetch: bool,
}

fn block_io_ns(cfg: &OffloadConfig) -> u64 {
    cfg.ssd_io_setup_ns + BLOCK_BYTES.saturating_mul(1_000_000_000) / cfg.ssd_read_bps.max(1)
}

/// SSD-backed KV block pool with speculative prefetch under a staging
/// budget. Deterministic: no async, no threads — the overlap is modeled by
/// [`overlap_stall_ns`] at the bench level.
pub struct SsdOffloadPool {
    cfg: OffloadConfig,
    staged: Vec<Slot>,
    staged_bytes: u64,
    resident: Vec<u64>,
    pub stats: PoolStats,
}

    /// Counters for one simulated decode run.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PoolStats {
    /// Blocks served from staging (prefetch hit or re-read).
    pub staged_hits: u64,
    /// Blocks that required a synchronous SSD read (staging miss).
    pub ssd_misses: u64,
    /// Blocks speculatively staged ahead of use.
    pub prefetch_issued: u64,
    /// SSD bytes actually read (prefetch + demand).
    pub ssd_bytes_read: u64,
    /// Prefetched bytes that were never used before eviction (waste).
    pub wasted_prefetch_bytes: u64,
    /// Peak staging occupancy, bytes.
    pub peak_staging_bytes: u64,
}

impl SsdOffloadPool {
    pub fn new(cfg: OffloadConfig, resident_blocks: Vec<u64>) -> Self {
        Self {
            cfg,
            staged: Vec::new(),
            staged_bytes: 0,
            resident: resident_blocks,
            stats: PoolStats::default(),
        }
    }

    fn is_resident(&self, key: u64) -> bool {
        self.resident.contains(&key)
    }

    /// Stage one block (if not resident/already staged) under the LRU
    /// staging budget. Evicted-unprefetched blocks count as waste.
    fn stage(&mut self, key: u64, tick: u64) {
        if self.is_resident(key) || self.staged.iter().any(|s| s.key == key) {
            return;
        }
        while self.staged_bytes + BLOCK_BYTES > self.cfg.staging_bytes {
            // LRU evict
            let victim = self
                .staged
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.last_tick)
                .map(|(i, _)| i);
            let Some(i) = victim else { break };
            let slot = self.staged.remove(i);
            self.staged_bytes -= slot.bytes;
            if !slot.used_since_prefetch {
                self.stats.wasted_prefetch_bytes += slot.bytes;
            }
        }
        self.staged.push(Slot { key, bytes: BLOCK_BYTES, last_tick: tick, used_since_prefetch: false });
        self.staged_bytes += BLOCK_BYTES;
        self.stats.peak_staging_bytes = self.stats.peak_staging_bytes.max(self.staged_bytes);
    }

    /// Speculatively stage a block ahead of use. Returns the SSD I/O cost
    /// (nanoseconds) the prefetch itself incurs — 0 when the block is
    /// already resident or staged. The caller decides how that I/O overlaps
    /// with decode compute (see [`overlap_stall_ns`] and the bench's
    /// pending-I/O drain model).
    pub fn prefetch(&mut self, key: u64, tick: u64) -> u64 {
        if self.is_resident(key) || self.staged.iter().any(|s| s.key == key) {
            return 0;
        }
        self.stage(key, tick);
        self.stats.prefetch_issued += 1;
        self.stats.ssd_bytes_read += BLOCK_BYTES;
        block_io_ns(&self.cfg)
    }

    /// Fetch a block for the current decode step. Returns the SSD read
    /// latency (nanoseconds) the step must cover: 0 when resident or
    /// staged, else setup + transfer. Updates stats and touch times.
    pub fn fetch(&mut self, key: u64, tick: u64) -> u64 {
        if self.is_resident(key) {
            return 0;
        }
        if let Some(pos) = self.staged.iter().position(|s| s.key == key) {
            self.staged[pos].last_tick = tick;
            self.staged[pos].used_since_prefetch = true;
            self.stats.staged_hits += 1;
            return 0;
        }
        self.stats.ssd_misses += 1;
        self.stats.ssd_bytes_read += BLOCK_BYTES;
        let ns = block_io_ns(&self.cfg);
        // cache it post-read so repeat reads within the window hit staging
        self.stage(key, tick);
        if let Some(pos) = self.staged.iter().position(|s| s.key == key) {
            self.staged[pos].used_since_prefetch = true;
        }
        ns
    }

    /// Prefetched-but-never-used staging bytes at end of run (the waste
    /// that eviction-only accounting misses).
    pub fn unused_staged_bytes(&self) -> u64 {
        self.staged
            .iter()
            .filter(|s| !s.used_since_prefetch)
            .map(|s| s.bytes)
            .sum()
    }
}

/// Overlap model (SolidAttention leg three, bounded): I/O issued during a
/// decode step hides under that step's compute; only the excess is a
/// decode stall. `io_ns` is the block latency, `compute_ns` the step's
/// attention compute.
pub fn overlap_stall_ns(io_ns: u64, compute_ns: u64) -> u64 {
    io_ns.saturating_sub(compute_ns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn salted_keys_isolate_tenants() {
        // HIP-1345 §1.1: identical block indices under different tenant
        // salts must be disjoint keys.
        let a = salted_block_key(1000, 42);
        let b = salted_block_key(1001, 42);
        assert_ne!(a, b);
        assert_eq!(salted_block_key(1000, 42), a); // deterministic
        assert_ne!(salted_block_key(1000, 43), a);
    }

    #[test]
    fn predictor_prefers_sequential_successor() {
        let p = predict_next(&[10, 11, 12], 4);
        assert_eq!(p[0], 13); // decode advances forward
        assert!(p.contains(&12)); // re-read of head is in the tail
    }

    #[test]
    fn fetch_resident_is_free() {
        let cfg = OffloadConfig::default();
        let mut pool = SsdOffloadPool::new(cfg, vec![salted_block_key(7, 0)]);
        let ns = pool.fetch(salted_block_key(7, 0), 0);
        assert_eq!(ns, 0);
        assert_eq!(pool.stats.ssd_misses, 0);
    }

    #[test]
    fn fetch_miss_pays_ssd_then_stages() {
        let cfg = OffloadConfig::default();
        let key = salted_block_key(7, 3);
        let mut pool = SsdOffloadPool::new(cfg, vec![]);
        let ns1 = pool.fetch(key, 0);
        assert!(ns1 > 0, "cold fetch must pay SSD latency");
        let ns2 = pool.fetch(key, 1);
        assert_eq!(ns2, 0, "post-read staging must serve the re-read");
        assert_eq!(pool.stats.ssd_misses, 1);
        assert_eq!(pool.stats.staged_hits, 1);
    }

    #[test]
    fn staging_budget_is_enforced_lru() {
        let cfg = OffloadConfig { staging_bytes: 2 * BLOCK_BYTES, ..OffloadConfig::default() };
        let mut pool = SsdOffloadPool::new(cfg, vec![]);
        let k0 = salted_block_key(7, 0);
        let k1 = salted_block_key(7, 1);
        let k2 = salted_block_key(7, 2);
        pool.fetch(k0, 0);
        pool.fetch(k1, 1);
        assert_eq!(pool.stats.peak_staging_bytes, 2 * BLOCK_BYTES);
        pool.fetch(k2, 2); // forces LRU eviction of k0 (oldest tick)
        let staged_keys: Vec<u64> = pool.staged.iter().map(|s| s.key).collect();
        assert!(!staged_keys.contains(&k0), "oldest tick must be evicted");
        assert!(staged_keys.contains(&k1) && staged_keys.contains(&k2));
        // k0 was used before eviction? it was fetched (used=true), so no
        // waste; a prefetch-never-used block counts waste instead.
        assert_eq!(pool.stats.wasted_prefetch_bytes, 0);
    }

    #[test]
    fn wasted_prefetch_counts_unused_evictions() {
        let cfg = OffloadConfig { staging_bytes: BLOCK_BYTES, ..OffloadConfig::default() };
        let mut pool = SsdOffloadPool::new(cfg, vec![]);
        let never_used = salted_block_key(7, 99);
        let hot = salted_block_key(7, 100);
        pool.prefetch(never_used, 0);
        pool.prefetch(hot, 1); // evicts never_used (older tick), which was never used
        assert_eq!(pool.stats.wasted_prefetch_bytes, BLOCK_BYTES);
    }

    #[test]
    fn overlap_hides_io_under_compute() {
        assert_eq!(overlap_stall_ns(100_000, 200_000), 0); // fully hidden
        assert_eq!(overlap_stall_ns(400_000, 200_000), 200_000); // excess stalls
    }
}
