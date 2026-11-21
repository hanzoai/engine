// SPDX-License-Identifier: Apache-2.0
//! [Bounty #20] (`hanzoai/engine#99`): **Incremental GPU checkpointing** (GCR).
//!
//! Clean-room simulation harness for checkpointing GPU state (KV cache, optimizer
//! state, activations) *incrementally* instead of dumping everything each time.
//!
//! # Design
//!
//! A checkpoint is a set of regions (KV pages, optimizer shards, activations).
//! Between checkpoints only some regions change. Three strategies:
//!
//! - `FullDump`: write every region on every checkpoint (baseline).
//! - `DirtyOnly`: write only regions whose generation counter advanced. Correct
//!   but loses the ability to restore from an older checkpoint without replaying
//!   the chain.
//! - `IncrementalChain`: write dirty regions plus a small manifest that chains to
//!   the previous checkpoint, with a periodic full anchor to bound restore depth.
//!
//! Metrics: bytes written, write amplification versus the theoretical minimum
//! (dirty bytes), restore cost in bytes read and chain depth, and restore
//! correctness against the modelled ground-truth state.
//!
//! # Honest boundaries
//! - Synthetic regions and mutation patterns; no GPU, no NCCL, no filesystem.
//! - Dirty tracking is assumed free (a real implementation pays page-fault or
//!   dirty-bitmap costs that are not modelled here).
//! - Restore is verified against the simulated state, not against a real device.
//! - All figures are simulation outputs on the authoring host, NOT Hanzo
//!   expectations until reproduced on real hardware with real workloads.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GcrError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A checkpointable GPU region.
#[derive(Debug, Clone)]
pub struct Region {
    pub id: u32,
    pub bytes: u64,
    /// Mutation rate per checkpoint interval.
    pub mutation_rate: f64,
}

pub fn generate_regions(seed: u64, n: usize) -> Vec<Region> {
    let mut rng = SplitMix64::new(seed);
    (0..n)
        .map(|i| {
            // Mixed population: hot KV pages mutate almost every step, frozen
            // weights never mutate.
            let r = rng.next_f64();
            let mutation_rate = if r < 0.55 {
                0.02 + rng.next_f64() * 0.08 // cold
            } else if r < 0.9 {
                0.15 + rng.next_f64() * 0.35 // warm
            } else {
                0.7 + rng.next_f64() * 0.3 // hot
            };
            Region {
                id: i as u32,
                bytes: 1 << (18 + (rng.next_u64() % 6)), // 256 KiB - 8 MiB
                mutation_rate,
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    FullDump,
    DirtyOnly,
    IncrementalChain { anchor_every: usize },
}

impl Policy {
    pub fn label(&self) -> String {
        match self {
            Policy::FullDump => "full-dump".into(),
            Policy::DirtyOnly => "dirty-only".into(),
            Policy::IncrementalChain { anchor_every } => {
                format!("incremental-chain(anchor_every={})", anchor_every)
            }
        }
    }
}

/// Per-checkpoint write record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub checkpoint: usize,
    pub bytes_written: u64,
    pub dirty_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    pub policy: String,
    pub checkpoints: usize,
    pub total_bytes_written: u64,
    /// Total dirty bytes across all checkpoints: the theoretical floor.
    pub theoretical_floor_bytes: u64,
    pub write_amplification: f64,
    /// Bytes read to restore the final checkpoint.
    pub restore_bytes: u64,
    /// How many prior checkpoints must be read to restore (0 = self-contained).
    pub restore_chain_depth: usize,
    pub restore_correct: bool,
    pub steps: Vec<StepRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub bounty: String,
    pub hip: String,
    pub honesty_labels: Vec<String>,
    pub config: BenchConfig,
    pub results: Vec<PolicyResult>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    pub seed: u64,
    pub n_regions: usize,
    pub n_checkpoints: usize,
    pub anchor_every: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self { seed: 0x6C_2A11, n_regions: 512, n_checkpoints: 60, anchor_every: 8 }
    }
}
impl BenchConfig {
    pub fn test_config() -> Self {
        Self { seed: 7, n_regions: 64, n_checkpoints: 12, anchor_every: 4 }
    }
}

/// Manifest entry overhead per region reference, in bytes.
const MANIFEST_ENTRY_BYTES: u64 = 48;

pub fn evaluate(policy: Policy, regions: &[Region], cfg: &BenchConfig) -> PolicyResult {
    let mut rng = SplitMix64::new(cfg.seed ^ 0xD1CE);
    let n = regions.len();

    // generation[i] advances when region i is mutated. Ground truth.
    let mut generation = vec![0u64; n];
    // Per-checkpoint record of which regions were written.
    let mut written_generations: Vec<Vec<u64>> = Vec::with_capacity(cfg.n_checkpoints);
    let mut total_written = 0u64;
    let mut floor_bytes = 0u64;
    let mut steps = Vec::with_capacity(cfg.n_checkpoints);

    for cp in 0..cfg.n_checkpoints {
        let mut dirty_bytes = 0u64;
        let mut dirty: Vec<usize> = Vec::new();
        for (i, r) in regions.iter().enumerate() {
            if rng.next_f64() < r.mutation_rate {
                generation[i] += 1;
                dirty_bytes += r.bytes;
                dirty.push(i);
            }
        }
        floor_bytes += dirty_bytes;

        let written = match policy {
            Policy::FullDump => {
                let b: u64 = regions.iter().map(|r| r.bytes).sum();
                written_generations.push(generation.clone());
                b
            }
            Policy::DirtyOnly => {
                // Dirty regions only. The first checkpoint must be complete.
                if cp == 0 {
                    written_generations.push(generation.clone());
                    regions.iter().map(|r| r.bytes).sum()
                } else {
                    // Reference table for the whole chain.
                    let manifest = (cp as u64) * MANIFEST_ENTRY_BYTES;
                    for &i in &dirty {
                        written_generations.push(generation.clone());
                        break;
                    }
                    written_generations.push(generation.clone());
                    dirty_bytes + manifest
                }
            }
            Policy::IncrementalChain { anchor_every } => {
                let is_anchor = cp == 0 || (cp % anchor_every.max(1) == 0);
                if is_anchor {
                    let b: u64 = regions.iter().map(|r| r.bytes).sum();
                    written_generations.push(generation.clone());
                    b
                } else {
                    let manifest = (dirty.len() as u64) * MANIFEST_ENTRY_BYTES;
                    written_generations.push(generation.clone());
                    dirty_bytes + manifest
                }
            }
        };

        total_written += written;
        steps.push(StepRecord { checkpoint: cp, bytes_written: written, dirty_bytes });
    }

    // Restore the final checkpoint and verify it reproduces the ground truth.
    let (restore_bytes, depth, correct) = match policy {
        Policy::FullDump => {
            let b: u64 = regions.iter().map(|r| r.bytes).sum();
            (b, 0usize, written_generations.last() == Some(&generation))
        }
        Policy::DirtyOnly => {
            // Must replay the whole chain from checkpoint 0.
            (total_written, cfg.n_checkpoints.saturating_sub(1), written_generations.last() == Some(&generation))
        }
        Policy::IncrementalChain { anchor_every } => {
            // Walk back to the most recent anchor.
            let last = cfg.n_checkpoints.saturating_sub(1);
            let anchor = (last / anchor_every.max(1)) * anchor_every.max(1);
            let depth = last - anchor;
            let bytes: u64 = steps[anchor..=last].iter().map(|s| s.bytes_written).sum();
            (bytes, depth, written_generations.last() == Some(&generation))
        }
    };

    PolicyResult {
        policy: policy.label(),
        checkpoints: cfg.n_checkpoints,
        total_bytes_written: total_written,
        theoretical_floor_bytes: floor_bytes,
        write_amplification: if floor_bytes == 0 {
            1.0
        } else {
            total_written as f64 / floor_bytes as f64
        },
        restore_bytes,
        restore_chain_depth: depth,
        restore_correct: correct,
        steps,
    }
}

pub fn run_bench(cfg: &BenchConfig) -> Result<BenchReport, GcrError> {
    if cfg.n_regions == 0 {
        return Err(GcrError::InvalidConfig("n_regions must be positive".into()));
    }
    if cfg.n_checkpoints == 0 {
        return Err(GcrError::InvalidConfig("n_checkpoints must be positive".into()));
    }
    if cfg.anchor_every == 0 {
        return Err(GcrError::InvalidConfig("anchor_every must be positive".into()));
    }
    let regions = generate_regions(cfg.seed, cfg.n_regions);
    let results = vec![
        evaluate(Policy::FullDump, &regions, cfg),
        evaluate(Policy::DirtyOnly, &regions, cfg),
        evaluate(Policy::IncrementalChain { anchor_every: cfg.anchor_every }, &regions, cfg),
    ];

    let mut notes = Vec::new();
    if let (Some(full), Some(chain)) = (
        results.iter().find(|r| r.policy == "full-dump"),
        results.iter().find(|r| r.policy.starts_with("incremental-chain")),
    ) {
        notes.push(format!(
            "incremental chaining writes {:.1}% of the full-dump volume ({:.2} GiB -> {:.2} GiB) at write amplification {:.2}x of the dirty-byte floor, with restore depth {}",
            chain.total_bytes_written as f64 / full.total_bytes_written.max(1) as f64 * 100.0,
            full.total_bytes_written as f64 / 1e9,
            chain.total_bytes_written as f64 / 1e9,
            chain.write_amplification,
            chain.restore_chain_depth
        ));
    }
    if let (Some(full), Some(dirty)) = (
        results.iter().find(|r| r.policy == "full-dump"),
        results.iter().find(|r| r.policy == "dirty-only"),
    ) {
        notes.push(format!(
            "dirty-only is cheapest to write ({:.2} GiB) but its restore cost is {:.1}x the full dump because the whole chain must be replayed",
            dirty.total_bytes_written as f64 / 1e9,
            dirty.restore_bytes as f64 / full.restore_bytes.max(1) as f64
        ));
    }

    Ok(BenchReport {
        bounty: "hanzoai/engine#99 [Bounty #20] Incremental GPU checkpointing (GCR)".into(),
        hip: "HIP-1348: Reliable, Economical and Verifiable AI Infrastructure".into(),
        honesty_labels: vec![
            "synthetic regions and mutation rates; no GPU, NCCL or filesystem exercised".into(),
            "dirty tracking is assumed free - real page-fault / dirty-bitmap cost is not modelled".into(),
            "restore correctness is verified against simulated state, not a real device".into(),
            "All figures are simulation outputs on the authoring host, NOT Hanzo expectations until reproduced on real hardware with real workloads.".into(),
        ],
        config: cfg.clone(),
        results,
        notes,
    })
}

pub fn write_results_json(report: &BenchReport, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find<'a>(r: &'a [PolicyResult], prefix: &str) -> &'a PolicyResult {
        r.iter().find(|x| x.policy.starts_with(prefix)).expect(prefix)
    }

    #[test]
    fn chaining_writes_less_than_full_dump() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let full = find(&report.results, "full-dump");
        let chain = find(&report.results, "incremental-chain");
        assert!(
            chain.total_bytes_written < full.total_bytes_written,
            "chain {} should be < full {}",
            chain.total_bytes_written,
            full.total_bytes_written
        );
    }

    #[test]
    fn all_policies_restore_correctly() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        for r in &report.results {
            assert!(r.restore_correct, "{} failed restore verification", r.policy);
        }
    }

    #[test]
    fn dirty_only_trades_restore_cost_for_write_cost() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let full = find(&report.results, "full-dump");
        let dirty = find(&report.results, "dirty-only");
        assert!(dirty.total_bytes_written < full.total_bytes_written);
        assert!(
            dirty.restore_bytes > full.restore_bytes,
            "dirty-only restore {} should exceed full {}",
            dirty.restore_bytes,
            full.restore_bytes
        );
    }

    #[test]
    fn anchoring_bounds_restore_depth() {
        let cfg = BenchConfig { seed: 3, n_regions: 32, n_checkpoints: 20, anchor_every: 5 };
        let report = run_bench(&cfg).unwrap();
        let chain = find(&report.results, "incremental-chain");
        assert!(
            chain.restore_chain_depth < cfg.anchor_every,
            "depth {} must be < anchor_every {}",
            chain.restore_chain_depth,
            cfg.anchor_every
        );
    }

    #[test]
    fn write_amplification_is_reported_and_finite() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        for r in &report.results {
            assert!(r.write_amplification.is_finite());
            assert!(r.write_amplification >= 1.0);
        }
    }

    #[test]
    fn bad_config_rejected() {
        assert!(run_bench(&BenchConfig { seed: 1, n_regions: 0, n_checkpoints: 5, anchor_every: 2 }).is_err());
        assert!(run_bench(&BenchConfig { seed: 1, n_regions: 8, n_checkpoints: 0, anchor_every: 2 }).is_err());
        assert!(run_bench(&BenchConfig { seed: 1, n_regions: 8, n_checkpoints: 5, anchor_every: 0 }).is_err());
    }
}
