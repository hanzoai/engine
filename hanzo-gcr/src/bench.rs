// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reproducible corpus-grounded benchmark harness for HIP-1346 (`hanzoai/engine#91`).
//!
//! Grounds all VRAM weight, KV-cache, LoRA, and optimizer page payloads in real bytes loaded from
//! canonical `hanzoai/engine` repository files, evaluates 4 realistic GPU job pause/resume
//! workloads across 3 matched baselines, and runs an end-to-end security & corruption verification
//! suite.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use crate::checkpoint::{CheckpointConfig, CheckpointEngine, CheckpointMode};
use crate::crypto::{sha256, Digest, Sha256};
use crate::error::{GcrError, GcrResult};
use crate::store::{
    encode_chunk_frame, CheckpointStore, FsCheckpointStore, IoStats, MemoryCheckpointStore,
    StorageTierModel,
};
use crate::vram::{GpuMemorySpace, Namespace, PageRegion, PageSize, SharedWeightBase};

/// Canonical `hanzoai/engine` repository files used as real non-synthetic payload corpus bytes.
pub const DEFAULT_CORPUS_FILES: &[&str] = &[
    "sample_speech.wav",
    "Cargo.lock",
    "LLM.md",
    "hanzo-paged-attn/src/kivi.rs",
    "hanzo-kv-fabric/src/adapter_paging.rs",
    "hanzo-kv-fabric/src/lib.rs",
    "README.md",
    "LICENSE-APACHE",
];

/// Deterministic xorshift64* PRNG (zero external dependencies).
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Seeds the deterministic PRNG.
    pub const fn new(seed: u64) -> Self {
        let s = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self { state: s }
    }

    /// Generates the next pseudo-random `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform index in `0..bound`.
    #[inline]
    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }
}

/// Real repository file corpus used to materialize realistic GPU page bytes.
#[derive(Debug, Clone)]
pub struct CorpusBytes {
    /// Relative paths of loaded repository files.
    pub files: Vec<String>,
    /// Raw concatenated bytes from the repository files.
    pub bytes: Vec<u8>,
    /// SHA-256 digest of the corpus for exact reproducibility auditing.
    pub digest: Digest,
}

impl CorpusBytes {
    /// Loads real corpus bytes from `repo_root`, recording exact file list and SHA-256 digest.
    pub fn load_from_repo(repo_root: &Path) -> GcrResult<Self> {
        let mut files = Vec::new();
        let mut bytes = Vec::new();
        let mut hasher = Sha256::new();

        for &rel in DEFAULT_CORPUS_FILES {
            let full = repo_root.join(rel);
            if let Ok(data) = fs::read(&full) {
                if !data.is_empty() {
                    files.push(rel.to_string());
                    hasher.update(rel.as_bytes());
                    hasher.update(&(data.len() as u64).to_le_bytes());
                    hasher.update(&data);
                    bytes.extend_from_slice(&data);
                }
            }
        }

        if bytes.is_empty() {
            return Err(GcrError::Io(format!(
                "no repository corpus files found under {}",
                repo_root.display()
            )));
        }

        Ok(Self {
            files,
            bytes,
            digest: hasher.finalize(),
        })
    }

    /// Fills a page-sized buffer deterministically from real corpus bytes salted by `(page_id, step)`.
    pub fn materialize_page(
        &self,
        page_size: usize,
        page_id: usize,
        step: usize,
        seed: u64,
    ) -> Vec<u8> {
        let mut out = vec![0u8; page_size];
        let corpus_len = self.bytes.len();
        let salt_hash = sha256(
            &[
                seed.to_le_bytes().as_slice(),
                (page_id as u64).to_le_bytes().as_slice(),
                (step as u64).to_le_bytes().as_slice(),
            ]
            .concat(),
        );
        let salt = salt_hash.as_bytes();
        let start_offset = (u64::from_le_bytes([
            salt[0], salt[1], salt[2], salt[3], salt[4], salt[5], salt[6], salt[7],
        ]) as usize)
            % corpus_len;

        // Copy slices from real repo bytes and XOR-mix with the 32-byte per-(page, step) salt
        // so every page & update has realistic entropy and never collapses to identical copies.
        let mut pos = 0usize;
        let mut src = start_offset;
        while pos < page_size {
            let take = (page_size - pos).min(corpus_len - src);
            out[pos..pos + take].copy_from_slice(&self.bytes[src..src + take]);
            pos += take;
            src = 0;
        }
        for (i, b) in out.iter_mut().enumerate() {
            *b ^= salt[i & 31].wrapping_add((i >> 8) as u8);
        }
        out
    }
}

/// Summary distribution statistics (p50, p95, p99, mean, min, max).
#[derive(Debug, Clone, Copy, Default)]
pub struct LatencySummary {
    /// 50th percentile (median) in milliseconds.
    pub p50_ms: f64,
    /// 95th percentile in milliseconds.
    pub p95_ms: f64,
    /// 99th percentile in milliseconds.
    pub p99_ms: f64,
    /// Arithmetic mean in milliseconds.
    pub mean_ms: f64,
    /// Minimum observed latency in milliseconds.
    pub min_ms: f64,
    /// Maximum observed latency in milliseconds.
    pub max_ms: f64,
}

impl LatencySummary {
    /// Computes percentile statistics from raw millisecond samples.
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let sum: f64 = sorted.iter().sum();
        Self {
            p50_ms: percentile(&sorted, 0.50),
            p95_ms: percentile(&sorted, 0.95),
            p99_ms: percentile(&sorted, 0.99),
            mean_ms: sum / (sorted.len() as f64),
            min_ms: sorted[0],
            max_ms: *sorted.last().unwrap(),
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Aggregated metrics for a single strategy within a workload.
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Strategy name (e.g., `full_naive_checkpoint`, `full_mutable_only_checkpoint`, `gcr_incremental_delta`).
    pub strategy: String,
    /// Number of pauses executed per run.
    pub pauses: usize,
    /// Pause latency distribution on real OS file-backed store (ms).
    pub pause_latency: LatencySummary,
    /// Resume latency distribution on real OS file-backed store (ms).
    pub resume_latency: LatencySummary,
    /// Mean bytes written per pause (across steady-state incremental pauses `1..N`, plus all-pause mean).
    pub mean_bytes_written_per_pause: f64,
    /// Mean bytes written per steady-state delta pause (`pause >= 1`).
    pub steady_state_bytes_written_per_pause: f64,
    /// Mean bytes written per pause in MiB.
    pub mean_mb_written_per_pause: f64,
    /// Cumulative bytes written across all N pauses in MiB.
    pub cumulative_mb_written_all_pauses: f64,
    /// Steady-state bytes written per pause in MiB.
    pub steady_state_mb_written_per_pause: f64,
    /// Mean bytes read per restore in MiB.
    pub mean_mb_read_per_restore: f64,
    /// Total on-disk storage footprint in bytes across all N pauses (at end of run).
    pub storage_footprint_bytes: u64,
    /// Total on-disk storage footprint in MiB across all N pauses.
    pub storage_footprint_mb: f64,
    /// Mean manifests traversed per restore.
    pub mean_manifests_traversed: f64,
    /// Maximum manifest chain depth observed.
    pub max_chain_depth_observed: u32,
    /// Number of compactions triggered during a single run.
    pub compactions_triggered: u32,
    /// Number of chunks reclaimed by GC during a single run.
    pub gc_chunks_deleted: u64,
    /// Analytical NVMe Gen4 projected steady-state pause time (ms).
    pub nvme_projected_pause_ms: f64,
    /// Analytical NVMe Gen4 projected mean resume time (ms).
    pub nvme_projected_resume_ms: f64,
    /// Analytical Hanzo S3-VFS projected steady-state pause time (ms).
    pub s3_vfs_projected_pause_ms: f64,
    /// Analytical Hanzo S3-VFS projected mean resume time (ms).
    pub s3_vfs_projected_resume_ms: f64,
    /// Bit-exact SHA-256 post-resume verification pass rate (`100.0` = all verified).
    pub bit_exact_match_pct: f64,
    /// Reduction ratio of steady-state pause bytes written vs `full_naive_checkpoint`.
    pub reduction_vs_naive: f64,
    /// Reduction ratio of steady-state pause bytes written vs `full_mutable_only_checkpoint`.
    pub reduction_vs_mutable_only: f64,
    /// Overhead ratio of steady-state pause bytes written relative to `full_mutable_only_checkpoint`.
    pub overhead_vs_mutable_only: f64,
}

/// Results for one of the 4 benchmark workloads.
#[derive(Debug, Clone)]
pub struct WorkloadResult {
    /// Canonical workload identifier.
    pub workload: String,
    /// Concise workload description.
    pub description: String,
    /// Page size in bytes.
    pub page_size_bytes: usize,
    /// Immutable read-only weight pages.
    pub immutable_pages: usize,
    /// Mutable state pages.
    pub mutable_pages: usize,
    /// Total logical VRAM size in MiB.
    pub total_vram_mb: f64,
    /// Number of pause/resume cycles.
    pub pauses: usize,
    /// Mean dirty fraction of mutable pages per pause (`0.0..=1.0`).
    pub mean_mutable_dirty_fraction: f64,
    /// Whether this workload represents an explicit negative regime.
    pub is_negative_regime: bool,
    /// Per-strategy benchmark results.
    pub strategies: Vec<StrategyMetrics>,
}

/// Results of the live security, isolation, and corruption verification suite.
#[derive(Debug, Clone)]
pub struct SecurityVerificationReport {
    /// Percentage of post-resume bit-exact SHA-256 matches across all benchmark restores.
    pub bit_exact_post_resume_pct: f64,
    /// Total restores verified across all workloads.
    pub total_restores_verified: usize,
    /// Truncated chunk rejection rate (`100.0`).
    pub truncated_chunk_rejection_pct: f64,
    /// Bit-flipped chunk payload rejection rate (`100.0`).
    pub corrupted_chunk_rejection_pct: f64,
    /// Corrupted manifest rejection rate (`100.0`).
    pub corrupted_manifest_rejection_pct: f64,
    /// Missing parent manifest rejection rate (`100.0`).
    pub missing_parent_rejection_pct: f64,
    /// Cyclic parent chain rejection rate (`100.0`).
    pub cyclic_chain_rejection_pct: f64,
    /// Cross-tenant restore rejection rate (`100.0`).
    pub cross_tenant_rejection_pct: f64,
    /// Forged cross-tenant manifest injection rejection rate (`100.0`).
    pub forged_tenant_manifest_rejection_pct: f64,
    /// Cross-job (same tenant, different job) restore rejection rate (`100.0`).
    pub cross_job_rejection_pct: f64,
    /// Path-traversal namespace rejection rate (`100.0`).
    pub path_traversal_rejection_pct: f64,
    /// Cross-tenant page leak rate (`0.0`).
    pub cross_tenant_leak_pct: f64,
    /// Crash-safe fsync + atomic rename verification passed.
    pub crash_safe_fsync_atomic_rename_ok: bool,
}

/// Full benchmark suite report serialized to `docs/research/hip-1346-gcr/results.json`.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Deterministic PRNG seed (`1346`).
    pub seed: u64,
    /// Measured iterations per workload.
    pub iterations: usize,
    /// Whether OS `fsync` was enabled during timed runs.
    pub fsync_timed_runs: bool,
    /// Real repository corpus files loaded.
    pub corpus_files: Vec<String>,
    /// Total real repository corpus bytes.
    pub corpus_bytes: usize,
    /// SHA-256 digest of the corpus.
    pub corpus_sha256: String,
    /// Results across all 4 workloads.
    pub workloads: Vec<WorkloadResult>,
    /// Live security & integrity verification report.
    pub security: SecurityVerificationReport,
}

/// Configuration for running the benchmark suite.
#[derive(Debug, Clone)]
pub struct BenchOptions {
    /// Deterministic seed (default `1346`).
    pub seed: u64,
    /// Number of measured iterations per workload (default `3`).
    pub iterations: usize,
    /// Repository root containing real corpus files.
    pub repo_root: PathBuf,
    /// Whether to invoke `File::sync_all()` on every timed chunk write (default `false` for fast
    /// multi-iteration runs; crash-safe `fsync_enabled = true` is ALWAYS verified explicitly).
    pub fsync_timed_runs: bool,
    /// Number of parallel restore worker threads.
    pub restore_workers: usize,
    /// Optional page size override (used by compact fast unit/integration tests; `None` uses 64 KiB).
    pub page_size_override: Option<PageSize>,
}

impl BenchOptions {
    /// Creates default benchmark options rooted at `repo_root`.
    pub fn with_repo_root(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            seed: 1346,
            iterations: 3,
            repo_root: repo_root.into(),
            fsync_timed_runs: false,
            restore_workers: 4,
            page_size_override: None,
        }
    }
}

/// Executes all 4 workloads + security verification and returns the full [`BenchmarkReport`].
pub fn run_full_benchmark(opts: &BenchOptions) -> GcrResult<BenchmarkReport> {
    let corpus = CorpusBytes::load_from_repo(&opts.repo_root)?;
    let mut total_restores_verified = 0usize;
    let mut total_restores_matched = 0usize;

    // 1. paused_agent_inference_kv: 5 pauses, 64 KiB pages, 80% weights (256 pages = 16 MiB),
    //    20% KV cache (64 pages = 4 MiB), 5%-12% KV pages dirty per pause.
    let w1 = run_workload(
        &corpus,
        opts,
        WorkloadSpec {
            name: "paused_agent_inference_kv",
            description: "Long-context agent inference job paused 5x between tool calls (80% immutable weights, 5-12% KV pages dirty per pause).",
            page_size: PageSize::KiB64,
            immutable_pages: 256,
            mutable_regions: vec![PageRegion::KvCache; 64],
            pauses: 5,
            dirty_pattern: DirtyPattern::RangeFraction { min_pct: 5, max_pct: 12 },
            is_negative_regime: false,
            configs: vec![
                ("full_naive_checkpoint", CheckpointConfig {
                    mode: CheckpointMode::FullNaive,
                    restore_workers: opts.restore_workers,
                    ..CheckpointConfig::default()
                }),
                ("full_mutable_only_checkpoint", CheckpointConfig {
                    mode: CheckpointMode::FullMutableOnly,
                    restore_workers: opts.restore_workers,
                    ..CheckpointConfig::default()
                }),
                ("gcr_incremental_delta", CheckpointConfig {
                    mode: CheckpointMode::GcrIncremental,
                    max_chain_depth: 4,
                    max_chain_delta_ratio: 2.0,
                    gc_on_compaction: true,
                    restore_workers: opts.restore_workers,
                    ..CheckpointConfig::default()
                }),
            ],
        },
        &mut total_restores_verified,
        &mut total_restores_matched,
    )?;

    // 2. preemptible_lora_finetune: 8 preemptions, 64 KiB pages, 320 frozen base weight pages (20 MiB),
    //    64 mutable pages (16 LoRA + 32 optimizer + 16 activations = 4 MiB); LoRA + hot optimizer pages dirty (~18-28% of mutable).
    let mut lora_regions = Vec::with_capacity(64);
    lora_regions.extend(std::iter::repeat_n(PageRegion::LoRaAdapter, 16));
    lora_regions.extend(std::iter::repeat_n(PageRegion::OptimizerState, 32));
    lora_regions.extend(std::iter::repeat_n(PageRegion::Activations, 16));

    let w2 = run_workload(
        &corpus,
        opts,
        WorkloadSpec {
            name: "preemptible_lora_finetune",
            description: "Spot-GPU LoRA fine-tuning job preempted 8x (83.3% frozen base weights, active LoRA + hot optimizer buckets dirty ~18-28% of mutable state).",
            page_size: PageSize::KiB64,
            immutable_pages: 320,
            mutable_regions: lora_regions,
            pauses: 8,
            dirty_pattern: DirtyPattern::LoRaPreemption,
            is_negative_regime: false,
            configs: vec![
                ("full_naive_checkpoint", CheckpointConfig {
                    mode: CheckpointMode::FullNaive,
                    restore_workers: opts.restore_workers,
                    ..CheckpointConfig::default()
                }),
                ("full_mutable_only_checkpoint", CheckpointConfig {
                    mode: CheckpointMode::FullMutableOnly,
                    restore_workers: opts.restore_workers,
                    ..CheckpointConfig::default()
                }),
                ("gcr_incremental_delta", CheckpointConfig {
                    mode: CheckpointMode::GcrIncremental,
                    max_chain_depth: 4,
                    max_chain_delta_ratio: 2.0,
                    gc_on_compaction: true,
                    restore_workers: opts.restore_workers,
                    ..CheckpointConfig::default()
                }),
            ],
        },
        &mut total_restores_verified,
        &mut total_restores_matched,
    )?;

    // 3. chain_compaction_sweep: 20 consecutive pauses comparing uncompacted delta chain vs
    //    bounded max_chain_depth = 4 compaction + GC vs full_mutable_only.
    let w3 = run_workload(
        &corpus,
        opts,
        WorkloadSpec {
            name: "chain_compaction_sweep",
            description: "20 consecutive pauses comparing uncompacted delta chain (depth 19) vs bounded max_chain_depth=4 compaction+GC vs full_mutable_only.",
            page_size: PageSize::KiB64,
            immutable_pages: 128,
            mutable_regions: vec![PageRegion::KvCache; 64],
            pauses: 20,
            dirty_pattern: DirtyPattern::RangeFraction { min_pct: 10, max_pct: 16 },
            is_negative_regime: false,
            configs: vec![
                ("full_mutable_only_checkpoint", CheckpointConfig {
                    mode: CheckpointMode::FullMutableOnly,
                    restore_workers: opts.restore_workers,
                    ..CheckpointConfig::default()
                }),
                ("gcr_uncompacted", CheckpointConfig {
                    mode: CheckpointMode::GcrIncremental,
                    max_chain_depth: u32::MAX,
                    max_chain_delta_ratio: f64::INFINITY,
                    gc_on_compaction: false,
                    restore_workers: opts.restore_workers,
                    ..CheckpointConfig::default()
                }),
                ("gcr_compacted_d4", CheckpointConfig {
                    mode: CheckpointMode::GcrIncremental,
                    max_chain_depth: 4,
                    max_chain_delta_ratio: 2.0,
                    gc_on_compaction: true,
                    restore_workers: opts.restore_workers,
                    ..CheckpointConfig::default()
                }),
            ],
        },
        &mut total_restores_verified,
        &mut total_restores_matched,
    )?;

    // 4. Explicit Negative Regime: full_weight_and_state_churn_100pct_dirty
    //    Dense full-parameter training where 100% of mutable pages are dirtied on every pause.
    let mut dense_regions = Vec::with_capacity(96);
    dense_regions.extend(std::iter::repeat_n(PageRegion::OptimizerState, 64));
    dense_regions.extend(std::iter::repeat_n(PageRegion::Activations, 32));

    let w4 = run_workload(
        &corpus,
        opts,
        WorkloadSpec {
            name: "full_weight_and_state_churn_100pct_dirty",
            description: "EXPLICIT NEGATIVE REGIME: dense full-parameter training step dirtying 100% of mutable pages every pause; proves gcr_incremental_delta yields 0% dirty-page savings and incurs delta-manifest/storage overhead vs full_mutable_only_checkpoint.",
            page_size: PageSize::KiB64,
            immutable_pages: 32,
            mutable_regions: dense_regions,
            pauses: 6,
            dirty_pattern: DirtyPattern::AllMutableDirty,
            is_negative_regime: true,
            configs: vec![
                ("full_naive_checkpoint", CheckpointConfig {
                    mode: CheckpointMode::FullNaive,
                    restore_workers: opts.restore_workers,
                    ..CheckpointConfig::default()
                }),
                ("full_mutable_only_checkpoint", CheckpointConfig {
                    mode: CheckpointMode::FullMutableOnly,
                    restore_workers: opts.restore_workers,
                    ..CheckpointConfig::default()
                }),
                ("gcr_incremental_delta", CheckpointConfig {
                    mode: CheckpointMode::GcrIncremental,
                    max_chain_depth: 4,
                    max_chain_delta_ratio: 10.0,
                    gc_on_compaction: false,
                    restore_workers: opts.restore_workers,
                    ..CheckpointConfig::default()
                }),
            ],
        },
        &mut total_restores_verified,
        &mut total_restores_matched,
    )?;

    let security = run_security_verification(
        &corpus,
        opts.seed,
        total_restores_verified,
        total_restores_matched,
    )?;

    Ok(BenchmarkReport {
        seed: opts.seed,
        iterations: opts.iterations,
        fsync_timed_runs: opts.fsync_timed_runs,
        corpus_files: corpus.files,
        corpus_bytes: corpus.bytes.len(),
        corpus_sha256: corpus.digest.to_hex(),
        workloads: vec![w1, w2, w3, w4],
        security,
    })
}

enum DirtyPattern {
    RangeFraction { min_pct: usize, max_pct: usize },
    LoRaPreemption,
    AllMutableDirty,
}

struct WorkloadSpec<'a> {
    name: &'a str,
    description: &'a str,
    page_size: PageSize,
    immutable_pages: usize,
    mutable_regions: Vec<PageRegion>,
    pauses: usize,
    dirty_pattern: DirtyPattern,
    is_negative_regime: bool,
    configs: Vec<(&'a str, CheckpointConfig)>,
}

fn run_workload(
    corpus: &CorpusBytes,
    opts: &BenchOptions,
    spec: WorkloadSpec<'_>,
    total_restores_verified: &mut usize,
    total_restores_matched: &mut usize,
) -> GcrResult<WorkloadResult> {
    let page_size = opts.page_size_override.unwrap_or(spec.page_size);
    let ps = page_size.bytes()?;
    let mut weight_pages = Vec::with_capacity(spec.immutable_pages);
    for i in 0..spec.immutable_pages {
        weight_pages.push(corpus.materialize_page(ps, i, 0, opts.seed));
    }
    let weights = SharedWeightBase::new(page_size, weight_pages)?;

    // Pre-generate deterministic mutation schedule and pre-materialize payloads ONCE so all
    // matched strategies see identical byte sequences with zero setup duplication overhead.
    let mut rng = DeterministicRng::new(opts.seed ^ fnv1a(spec.name.as_bytes()));
    let mutable_count = spec.mutable_regions.len();
    let mut schedules: Vec<Vec<(usize, Vec<u8>)>> = Vec::with_capacity(spec.pauses);
    for pause_idx in 0..spec.pauses {
        let mut dirty_offsets = Vec::new();
        if pause_idx == 0 {
            dirty_offsets.extend(0..mutable_count);
        } else {
            match spec.dirty_pattern {
                DirtyPattern::RangeFraction { min_pct, max_pct } => {
                    let span = (max_pct - min_pct).max(1);
                    let pct = min_pct + rng.next_usize(span + 1);
                    let count = ((mutable_count * pct).div_ceil(100)).clamp(1, mutable_count);
                    let base = rng.next_usize(mutable_count);
                    for k in 0..count {
                        dirty_offsets.push((base + k) % mutable_count);
                    }
                }
                DirtyPattern::LoRaPreemption => {
                    let lora_dirty = 8 + rng.next_usize(5);
                    let opt_dirty = 4 + rng.next_usize(3);
                    for k in 0..lora_dirty.min(16) {
                        dirty_offsets.push((pause_idx * 3 + k) % 16);
                    }
                    for k in 0..opt_dirty.min(32) {
                        dirty_offsets.push(16 + ((pause_idx * 5 + k) % 32));
                    }
                }
                DirtyPattern::AllMutableDirty => {
                    dirty_offsets.extend(0..mutable_count);
                }
            }
        }
        let mut pause_writes = Vec::with_capacity(dirty_offsets.len());
        for m_off in dirty_offsets {
            let page_id = spec.immutable_pages + m_off;
            let page_data = corpus.materialize_page(ps, page_id, pause_idx + 1, opts.seed);
            pause_writes.push((page_id, page_data));
        }
        schedules.push(pause_writes);
    }

    let steady_dirty_fracs: Vec<f64> = schedules
        .iter()
        .skip(1)
        .map(|s| s.len() as f64 / mutable_count.max(1) as f64)
        .collect();
    let mean_mutable_dirty_fraction = if steady_dirty_fracs.is_empty() {
        1.0
    } else {
        steady_dirty_fracs.iter().sum::<f64>() / steady_dirty_fracs.len() as f64
    };

    let mut strategy_results = Vec::with_capacity(spec.configs.len());

    for (strat_name, config) in &spec.configs {
        let mut pause_samples = Vec::new();
        let mut resume_samples = Vec::new();
        let mut all_pause_bytes = Vec::new();
        let mut steady_pause_bytes = Vec::new();
        let mut restore_read_bytes = Vec::new();
        let mut manifests_traversed_samples = Vec::new();
        let mut nvme_pause_ms_samples = Vec::new();
        let mut nvme_resume_ms_samples = Vec::new();
        let mut s3_pause_ms_samples = Vec::new();
        let mut s3_resume_ms_samples = Vec::new();

        let mut last_storage_footprint = 0u64;
        let mut last_cumulative_written = 0u64;
        let mut max_chain_depth_observed = 0u32;
        let mut compactions_per_run = 0u32;
        let mut gc_chunks_per_run = 0u64;
        let mut strat_restores = 0usize;
        let mut strat_matches = 0usize;

        for iter in 0..opts.iterations.max(1) {
            let tmp_dir = std::env::temp_dir().join(format!(
                "hanzo_gcr_bench_{}_{}_{}_{}",
                spec.name,
                strat_name,
                std::process::id(),
                iter
            ));
            let _ = fs::remove_dir_all(&tmp_dir);
            let store = FsCheckpointStore::new(&tmp_dir, opts.fsync_timed_runs)?;
            let engine = CheckpointEngine::new(&store, config.clone());

            let ns = Namespace::new("hanzo-bench", format!("{}-{strat_name}-{iter}", spec.name))?;
            let mut vram =
                GpuMemorySpace::new(ns.clone(), weights.clone(), spec.mutable_regions.clone())?;
            let mut parent_id: Option<String> = None;

            let mut run_compactions = 0u32;
            let mut run_gc_chunks = 0u64;
            let mut run_bytes_written = 0u64;

            for (pause_idx, pause_writes) in schedules.iter().enumerate() {
                for (page_id, page_data) in pause_writes {
                    vram.overwrite_page(*page_id, page_data)?;
                }

                let cid = format!("pause-{pause_idx:03}");
                let ckpt = engine.checkpoint(&vram, &cid, parent_id.as_deref())?;
                max_chain_depth_observed = max_chain_depth_observed.max(ckpt.manifest.chain_depth);
                if ckpt.compacted {
                    run_compactions += 1;
                }
                run_gc_chunks += ckpt.gc_chunks_deleted;
                run_bytes_written += ckpt.io_delta.bytes_written;

                all_pause_bytes.push(ckpt.io_delta.bytes_written as f64);
                if pause_idx >= 1 {
                    pause_samples.push(ckpt.pause_latency_ms);
                    steady_pause_bytes.push(ckpt.io_delta.bytes_written as f64);
                    nvme_pause_ms_samples.push(
                        StorageTierModel::NVME_GEN4
                            .projected_write_ms(&ckpt.io_delta, opts.restore_workers),
                    );
                    s3_pause_ms_samples.push(
                        StorageTierModel::HANZO_S3_VFS
                            .projected_write_ms(&ckpt.io_delta, opts.restore_workers),
                    );
                }

                // Cold-boot restore & bit-exact SHA-256 verification.
                let mut cold = vram.cold_boot_clone_layout()?;
                let restored = engine.restore(&mut cold, &cid)?;
                let exact =
                    restored.bit_exact_verified && cold.state_digest() == vram.state_digest();
                strat_restores += 1;
                *total_restores_verified += 1;
                if exact {
                    strat_matches += 1;
                    *total_restores_matched += 1;
                }

                if pause_idx >= 1 {
                    resume_samples.push(restored.resume_latency_ms);
                    restore_read_bytes.push(restored.io_delta.bytes_read as f64);
                    manifests_traversed_samples.push(restored.manifests_traversed as f64);
                    nvme_resume_ms_samples.push(
                        StorageTierModel::NVME_GEN4
                            .projected_read_ms(&restored.io_delta, opts.restore_workers),
                    );
                    s3_resume_ms_samples.push(
                        StorageTierModel::HANZO_S3_VFS
                            .projected_read_ms(&restored.io_delta, opts.restore_workers),
                    );
                }

                parent_id = Some(cid);
            }

            last_storage_footprint = store.namespace_storage_bytes(&ns)?;
            last_cumulative_written = run_bytes_written;
            compactions_per_run = run_compactions;
            gc_chunks_per_run = run_gc_chunks;
            let _ = fs::remove_dir_all(&tmp_dir);
        }

        let mean_all_bytes = mean(&all_pause_bytes);
        let mean_steady_bytes = mean(&steady_pause_bytes);
        let mean_read_bytes = mean(&restore_read_bytes);

        strategy_results.push(StrategyMetrics {
            strategy: (*strat_name).to_string(),
            pauses: spec.pauses,
            pause_latency: LatencySummary::from_samples(&pause_samples),
            resume_latency: LatencySummary::from_samples(&resume_samples),
            mean_bytes_written_per_pause: mean_all_bytes,
            steady_state_bytes_written_per_pause: mean_steady_bytes,
            mean_mb_written_per_pause: mean_all_bytes / (1024.0 * 1024.0),
            cumulative_mb_written_all_pauses: last_cumulative_written as f64 / (1024.0 * 1024.0),
            steady_state_mb_written_per_pause: mean_steady_bytes / (1024.0 * 1024.0),
            mean_mb_read_per_restore: mean_read_bytes / (1024.0 * 1024.0),
            storage_footprint_bytes: last_storage_footprint,
            storage_footprint_mb: last_storage_footprint as f64 / (1024.0 * 1024.0),
            mean_manifests_traversed: mean(&manifests_traversed_samples),
            max_chain_depth_observed,
            compactions_triggered: compactions_per_run,
            gc_chunks_deleted: gc_chunks_per_run,
            nvme_projected_pause_ms: mean(&nvme_pause_ms_samples),
            nvme_projected_resume_ms: mean(&nvme_resume_ms_samples),
            s3_vfs_projected_pause_ms: mean(&s3_pause_ms_samples),
            s3_vfs_projected_resume_ms: mean(&s3_resume_ms_samples),
            bit_exact_match_pct: if strat_restores > 0 {
                (strat_matches as f64 / strat_restores as f64) * 100.0
            } else {
                0.0
            },
            reduction_vs_naive: 1.0,
            reduction_vs_mutable_only: 1.0,
            overhead_vs_mutable_only: 1.0,
        });
    }

    // Compute exact reduction / overhead ratios relative to naive and mutable-only baselines.
    let naive_steady = strategy_results
        .iter()
        .find(|s| s.strategy == "full_naive_checkpoint")
        .map(|s| s.steady_state_bytes_written_per_pause);
    let mutable_steady = strategy_results
        .iter()
        .find(|s| s.strategy == "full_mutable_only_checkpoint")
        .map(|s| s.steady_state_bytes_written_per_pause);

    for s in &mut strategy_results {
        if let Some(n) = naive_steady {
            if s.steady_state_bytes_written_per_pause > 0.0 {
                s.reduction_vs_naive = n / s.steady_state_bytes_written_per_pause;
            }
        }
        if let Some(m) = mutable_steady {
            if s.steady_state_bytes_written_per_pause > 0.0 && m > 0.0 {
                s.reduction_vs_mutable_only = m / s.steady_state_bytes_written_per_pause;
                s.overhead_vs_mutable_only = s.steady_state_bytes_written_per_pause / m;
            }
        }
    }

    let total_vram_mb = ((spec.immutable_pages + mutable_count) * ps) as f64 / (1024.0 * 1024.0);

    Ok(WorkloadResult {
        workload: spec.name.to_string(),
        description: spec.description.to_string(),
        page_size_bytes: ps,
        immutable_pages: spec.immutable_pages,
        mutable_pages: mutable_count,
        total_vram_mb,
        pauses: spec.pauses,
        mean_mutable_dirty_fraction,
        is_negative_regime: spec.is_negative_regime,
        strategies: strategy_results,
    })
}

fn run_security_verification(
    corpus: &CorpusBytes,
    seed: u64,
    total_restores_verified: usize,
    total_restores_matched: usize,
) -> GcrResult<SecurityVerificationReport> {
    let ps = 4096usize;
    let weights = SharedWeightBase::new(
        PageSize::KiB4,
        (0..4)
            .map(|i| corpus.materialize_page(ps, i, 0, seed))
            .collect(),
    )?;

    let store = MemoryCheckpointStore::new();
    let engine = CheckpointEngine::new(&store, CheckpointConfig::default());

    let ns_a = Namespace::new("tenant-alpha", "job-01")?;
    let mut vram_a =
        GpuMemorySpace::new(ns_a.clone(), weights.clone(), vec![PageRegion::KvCache; 8])?;
    vram_a.overwrite_page(4, &corpus.materialize_page(ps, 4, 1, seed))?;
    let r0 = engine.checkpoint(&vram_a, "s0", None)?;
    vram_a.overwrite_page(5, &corpus.materialize_page(ps, 5, 2, seed))?;
    let r1 = engine.checkpoint(&vram_a, "s1", Some("s0"))?;

    // 1. Cross-tenant restore rejection.
    let ns_b = Namespace::new("tenant-beta", "job-01")?;
    let mut vram_b =
        GpuMemorySpace::new(ns_b.clone(), weights.clone(), vec![PageRegion::KvCache; 8])?;
    let cross_tenant_rejected = engine.restore(&mut vram_b, "s1").is_err();

    // 2. Forged cross-tenant manifest injection rejection.
    let raw_s1 = store.get_manifest(&ns_a, "s1")?;
    store.inject_raw_manifest(&ns_b, "s1", raw_s1);
    let forged_tenant_rejected = matches!(
        engine.restore(&mut vram_b, "s1"),
        Err(GcrError::TenantMismatch { .. })
    );

    // 3. Cross-job (same tenant) restore rejection.
    let ns_a2 = Namespace::new("tenant-alpha", "job-02")?;
    let mut vram_a2 = GpuMemorySpace::new(ns_a2, weights.clone(), vec![PageRegion::KvCache; 8])?;
    let cross_job_rejected = engine.restore(&mut vram_a2, "s1").is_err();

    // 4. Path-traversal namespace injection rejection.
    let traversal_cases = [
        ("../etc", "job1"),
        ("tenant/a", "job1"),
        ("tenant\\a", "job1"),
        ("tenant-a", ".."),
        ("", "job1"),
    ];
    let traversal_rejected = traversal_cases
        .iter()
        .filter(|(t, j)| Namespace::new(*t, *j).is_err())
        .count();
    let path_traversal_rejection_pct =
        (traversal_rejected as f64 / traversal_cases.len() as f64) * 100.0;

    // 5. Truncated & bit-flipped chunk rejection.
    let target_digest = r1.manifest.entries[0].chunk_digest;
    let valid_frame = encode_chunk_frame(vram_a.read_page(5)?, false).0;
    let mut cold = vram_a.cold_boot_clone_layout()?;

    store.inject_raw_chunk(&ns_a, target_digest, valid_frame[..32].to_vec());
    let truncated_rejected = matches!(
        engine.restore(&mut cold, "s1"),
        Err(GcrError::CorruptedChunk { .. })
    );

    let mut flipped = valid_frame.clone();
    flipped[60] ^= 0x04;
    store.inject_raw_chunk(&ns_a, target_digest, flipped);
    let corrupted_chunk_rejected = matches!(
        engine.restore(&mut cold, "s1"),
        Err(GcrError::CorruptedChunk { .. })
    );
    store.inject_raw_chunk(&ns_a, target_digest, valid_frame);

    // 6. Corrupted manifest rejection.
    let (mut bad_env, _) = r1.manifest.to_envelope();
    bad_env[50] ^= 0xFF;
    store.inject_raw_manifest(&ns_a, "s1", bad_env);
    let corrupted_manifest_rejected = matches!(
        engine.restore(&mut cold, "s1"),
        Err(GcrError::CorruptedManifest { .. })
    );

    // 7. Cyclic parent chain rejection.
    let mut c0 = r0.manifest.clone();
    c0.parent_checkpoint_id = Some("s1".to_string());
    c0.chain_depth = 2;
    store.inject_raw_manifest(&ns_a, "s0", c0.to_envelope().0);
    let mut c1 = r1.manifest.clone();
    c1.parent_checkpoint_id = Some("s0".to_string());
    c1.chain_depth = 3;
    store.inject_raw_manifest(&ns_a, "s1", c1.to_envelope().0);
    let cyclic_rejected = matches!(
        engine.restore(&mut cold, "s1"),
        Err(GcrError::CyclicCheckpointChain { .. })
    );

    // 8. Missing parent rejection.
    store.inject_raw_manifest(&ns_a, "s1", r1.manifest.to_envelope().0);
    store.remove_manifest(&ns_a, "s0");
    let missing_parent_rejected = matches!(
        engine.restore(&mut cold, "s1"),
        Err(GcrError::MissingParentCheckpoint { .. })
    );

    // 9. Real OS crash-safe fsync + atomic rename verification.
    let fs_dir =
        std::env::temp_dir().join(format!("hanzo_gcr_fsync_verify_{}", std::process::id()));
    let _ = fs::remove_dir_all(&fs_dir);
    let fs_store = FsCheckpointStore::new(&fs_dir, true)?;
    let fs_engine = CheckpointEngine::new(&fs_store, CheckpointConfig::default());
    let fs_ckpt = fs_engine.checkpoint(&vram_a, "durability-0", None)?;
    let mut fs_cold = vram_a.cold_boot_clone_layout()?;
    let fs_restored = fs_engine.restore(&mut fs_cold, "durability-0")?;
    let crash_safe_ok = fs_ckpt.io_delta.fsync_calls >= 2
        && fs_restored.bit_exact_verified
        && fs_cold.state_digest() == vram_a.state_digest();
    let _ = fs::remove_dir_all(&fs_dir);

    let bit_exact_post_resume_pct = if total_restores_verified > 0 {
        (total_restores_matched as f64 / total_restores_verified as f64) * 100.0
    } else {
        0.0
    };

    Ok(SecurityVerificationReport {
        bit_exact_post_resume_pct,
        total_restores_verified,
        truncated_chunk_rejection_pct: if truncated_rejected { 100.0 } else { 0.0 },
        corrupted_chunk_rejection_pct: if corrupted_chunk_rejected { 100.0 } else { 0.0 },
        corrupted_manifest_rejection_pct: if corrupted_manifest_rejected {
            100.0
        } else {
            0.0
        },
        missing_parent_rejection_pct: if missing_parent_rejected { 100.0 } else { 0.0 },
        cyclic_chain_rejection_pct: if cyclic_rejected { 100.0 } else { 0.0 },
        cross_tenant_rejection_pct: if cross_tenant_rejected { 100.0 } else { 0.0 },
        forged_tenant_manifest_rejection_pct: if forged_tenant_rejected { 100.0 } else { 0.0 },
        cross_job_rejection_pct: if cross_job_rejected { 100.0 } else { 0.0 },
        path_traversal_rejection_pct,
        cross_tenant_leak_pct: if cross_tenant_rejected && forged_tenant_rejected {
            0.0
        } else {
            100.0
        },
        crash_safe_fsync_atomic_rename_ok: crash_safe_ok,
    })
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / (values.len() as f64)
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

impl BenchmarkReport {
    /// Serializes the benchmark report to clean, deterministic, pretty-printed JSON (zero external crates).
    pub fn to_json_pretty(&self) -> String {
        let mut s = String::with_capacity(16 * 1024);
        s.push_str("{\n");
        let _ = writeln!(s, "  \"hip\": \"HIP-1346\",");
        let _ = writeln!(s, "  \"issue\": \"hanzoai/engine#91\",");
        let _ = writeln!(s, "  \"crate\": \"hanzo-gcr\",");
        let _ = writeln!(s, "  \"seed\": {},", self.seed);
        let _ = writeln!(s, "  \"iterations\": {},", self.iterations);
        let _ = writeln!(s, "  \"fsync_timed_runs\": {},", self.fsync_timed_runs);
        let _ = writeln!(s, "  \"corpus\": {{");
        s.push_str("    \"files\": [");
        for (i, f) in self.corpus_files.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            let _ = write!(s, "\"{}\"", json_escape(f));
        }
        s.push_str("],\n");
        let _ = writeln!(s, "    \"total_bytes\": {},", self.corpus_bytes);
        let _ = writeln!(s, "    \"sha256\": \"{}\"", self.corpus_sha256);
        s.push_str("  },\n");

        s.push_str("  \"storage_tier_models\": {\n");
        let _ = writeln!(
            s,
            "    \"nvme_gen4\": {{ \"bandwidth_gb_s\": 3.5, \"per_op_latency_ms\": 0.08 }},"
        );
        let _ = writeln!(
            s,
            "    \"hanzo_s3_vfs\": {{ \"bandwidth_gb_s\": 0.8, \"per_op_latency_ms\": 4.0 }}"
        );
        s.push_str("  },\n");

        s.push_str("  \"workloads\": [\n");
        for (wi, w) in self.workloads.iter().enumerate() {
            s.push_str("    {\n");
            let _ = writeln!(s, "      \"workload\": \"{}\",", json_escape(&w.workload));
            let _ = writeln!(
                s,
                "      \"description\": \"{}\",",
                json_escape(&w.description)
            );
            let _ = writeln!(s, "      \"page_size_bytes\": {},", w.page_size_bytes);
            let _ = writeln!(s, "      \"immutable_pages\": {},", w.immutable_pages);
            let _ = writeln!(s, "      \"mutable_pages\": {},", w.mutable_pages);
            let _ = writeln!(s, "      \"total_vram_mb\": {:.4},", w.total_vram_mb);
            let _ = writeln!(s, "      \"pauses\": {},", w.pauses);
            let _ = writeln!(
                s,
                "      \"mean_mutable_dirty_fraction\": {:.4},",
                w.mean_mutable_dirty_fraction
            );
            let _ = writeln!(s, "      \"is_negative_regime\": {},", w.is_negative_regime);
            s.push_str("      \"strategies\": [\n");
            for (si, st) in w.strategies.iter().enumerate() {
                s.push_str("        {\n");
                let _ = writeln!(
                    s,
                    "          \"strategy\": \"{}\",",
                    json_escape(&st.strategy)
                );
                let _ = writeln!(s, "          \"pauses\": {},", st.pauses);
                let _ = writeln!(
                    s,
                    "          \"pause_latency_ms\": {{ \"p50\": {:.4}, \"p95\": {:.4}, \"p99\": {:.4}, \"mean\": {:.4}, \"min\": {:.4}, \"max\": {:.4} }},",
                    st.pause_latency.p50_ms,
                    st.pause_latency.p95_ms,
                    st.pause_latency.p99_ms,
                    st.pause_latency.mean_ms,
                    st.pause_latency.min_ms,
                    st.pause_latency.max_ms
                );
                let _ = writeln!(
                    s,
                    "          \"resume_latency_ms\": {{ \"p50\": {:.4}, \"p95\": {:.4}, \"p99\": {:.4}, \"mean\": {:.4}, \"min\": {:.4}, \"max\": {:.4} }},",
                    st.resume_latency.p50_ms,
                    st.resume_latency.p95_ms,
                    st.resume_latency.p99_ms,
                    st.resume_latency.mean_ms,
                    st.resume_latency.min_ms,
                    st.resume_latency.max_ms
                );
                let _ = writeln!(
                    s,
                    "          \"mean_bytes_written_per_pause\": {:.2},",
                    st.mean_bytes_written_per_pause
                );
                let _ = writeln!(
                    s,
                    "          \"steady_state_bytes_written_per_pause\": {:.2},",
                    st.steady_state_bytes_written_per_pause
                );
                let _ = writeln!(
                    s,
                    "          \"mean_mb_written_per_pause\": {:.4},",
                    st.mean_mb_written_per_pause
                );
                let _ = writeln!(
                    s,
                    "          \"cumulative_mb_written_all_pauses\": {:.4},",
                    st.cumulative_mb_written_all_pauses
                );
                let _ = writeln!(
                    s,
                    "          \"steady_state_mb_written_per_pause\": {:.4},",
                    st.steady_state_mb_written_per_pause
                );
                let _ = writeln!(
                    s,
                    "          \"mean_mb_read_per_restore\": {:.4},",
                    st.mean_mb_read_per_restore
                );
                let _ = writeln!(
                    s,
                    "          \"storage_footprint_bytes\": {},",
                    st.storage_footprint_bytes
                );
                let _ = writeln!(
                    s,
                    "          \"storage_footprint_mb\": {:.4},",
                    st.storage_footprint_mb
                );
                let _ = writeln!(
                    s,
                    "          \"mean_manifests_traversed\": {:.2},",
                    st.mean_manifests_traversed
                );
                let _ = writeln!(
                    s,
                    "          \"max_chain_depth_observed\": {},",
                    st.max_chain_depth_observed
                );
                let _ = writeln!(
                    s,
                    "          \"compactions_triggered\": {},",
                    st.compactions_triggered
                );
                let _ = writeln!(
                    s,
                    "          \"gc_chunks_deleted\": {},",
                    st.gc_chunks_deleted
                );
                let _ = writeln!(
                    s,
                    "          \"nvme_projected_pause_ms\": {:.4},",
                    st.nvme_projected_pause_ms
                );
                let _ = writeln!(
                    s,
                    "          \"nvme_projected_resume_ms\": {:.4},",
                    st.nvme_projected_resume_ms
                );
                let _ = writeln!(
                    s,
                    "          \"s3_vfs_projected_pause_ms\": {:.4},",
                    st.s3_vfs_projected_pause_ms
                );
                let _ = writeln!(
                    s,
                    "          \"s3_vfs_projected_resume_ms\": {:.4},",
                    st.s3_vfs_projected_resume_ms
                );
                let _ = writeln!(
                    s,
                    "          \"bit_exact_match_pct\": {:.2},",
                    st.bit_exact_match_pct
                );
                let _ = writeln!(
                    s,
                    "          \"reduction_vs_naive\": {:.4},",
                    st.reduction_vs_naive
                );
                let _ = writeln!(
                    s,
                    "          \"reduction_vs_mutable_only\": {:.4},",
                    st.reduction_vs_mutable_only
                );
                let _ = writeln!(
                    s,
                    "          \"overhead_vs_mutable_only\": {:.6}",
                    st.overhead_vs_mutable_only
                );
                if si + 1 < w.strategies.len() {
                    s.push_str("        },\n");
                } else {
                    s.push_str("        }\n");
                }
            }
            s.push_str("      ]\n");
            if wi + 1 < self.workloads.len() {
                s.push_str("    },\n");
            } else {
                s.push_str("    }\n");
            }
        }
        s.push_str("  ],\n");

        let sec = &self.security;
        s.push_str("  \"security_and_correctness\": {\n");
        let _ = writeln!(
            s,
            "    \"bit_exact_post_resume_pct\": {:.2},",
            sec.bit_exact_post_resume_pct
        );
        let _ = writeln!(
            s,
            "    \"total_restores_verified\": {},",
            sec.total_restores_verified
        );
        let _ = writeln!(
            s,
            "    \"truncated_chunk_rejection_pct\": {:.2},",
            sec.truncated_chunk_rejection_pct
        );
        let _ = writeln!(
            s,
            "    \"corrupted_chunk_rejection_pct\": {:.2},",
            sec.corrupted_chunk_rejection_pct
        );
        let _ = writeln!(
            s,
            "    \"corrupted_manifest_rejection_pct\": {:.2},",
            sec.corrupted_manifest_rejection_pct
        );
        let _ = writeln!(
            s,
            "    \"missing_parent_rejection_pct\": {:.2},",
            sec.missing_parent_rejection_pct
        );
        let _ = writeln!(
            s,
            "    \"cyclic_chain_rejection_pct\": {:.2},",
            sec.cyclic_chain_rejection_pct
        );
        let _ = writeln!(
            s,
            "    \"cross_tenant_rejection_pct\": {:.2},",
            sec.cross_tenant_rejection_pct
        );
        let _ = writeln!(
            s,
            "    \"forged_tenant_manifest_rejection_pct\": {:.2},",
            sec.forged_tenant_manifest_rejection_pct
        );
        let _ = writeln!(
            s,
            "    \"cross_job_rejection_pct\": {:.2},",
            sec.cross_job_rejection_pct
        );
        let _ = writeln!(
            s,
            "    \"path_traversal_rejection_pct\": {:.2},",
            sec.path_traversal_rejection_pct
        );
        let _ = writeln!(
            s,
            "    \"cross_tenant_leak_pct\": {:.2},",
            sec.cross_tenant_leak_pct
        );
        let _ = writeln!(
            s,
            "    \"crash_safe_fsync_atomic_rename_ok\": {}",
            sec.crash_safe_fsync_atomic_rename_ok
        );
        s.push_str("  }\n");
        s.push_str("}\n");
        s
    }
}

fn json_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Ensures `IoStats` re-export stays used in public benchmark API signatures.
pub fn summarize_io_delta(before: &IoStats, after: &IoStats) -> IoStats {
    after.delta_since(before)
}
