// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Operating profiles, 5 matched scheduling policies, SLO targets, and validated configuration.

use crate::kv_pool::SchedulerError;

/// Separate latency-vs-throughput operating profiles as required by HIP-1346.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerProfile {
    /// Protects single-user TTFT and ITL strictly with zero batch-formation wait and
    /// direct unchunked prefill fast path.
    SingleUserInteractive,
    /// Maximizes multi-request GPU throughput (`C = 4 / 16 / 64 / 256`) using high token
    /// budgets and chunked prefill + decode piggybacking.
    DatacenterThroughput,
    /// Protects both `p95 TTFT` and `p95 ITL` SLOs simultaneously via EDF/slack-aware decode
    /// prioritisation, roofline-bounded dynamic prefill chunk sizing, KV-pressure admission
    /// control, and automatic `C == 1` single-user bypass.
    LatencyAwareSlo,
}

impl SchedulerProfile {
    /// Stable lowercase identifier for receipts and tables.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SingleUserInteractive => "single_user_interactive",
            Self::DatacenterThroughput => "datacenter_throughput",
            Self::LatencyAwareSlo => "latency_aware_slo",
        }
    }
}

/// The 5 matched scheduling policies evaluated on identical workloads and KV pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// Policy 1: Classic static batching FCFS — forms a batch, runs all prefills, then decodes
    /// until the longest sequence completes before admitting any waiting request.
    StaticBatchingFcfs,
    /// Policy 2: Iteration-level continuous batching without chunked prefill, prioritizing
    /// waiting prefills (long prefills stall active decodes -> severe ITL spikes).
    ContinuousBatchingPrefillFirst,
    /// Policy 3: Iteration-level continuous batching without chunking, prioritizing active
    /// decodes (starves waiting prefills -> severe TTFT spikes, then still stalls ITL on run).
    ContinuousBatchingDecodePriority,
    /// Policy 4: Token-budget chunked prefill piggybacked onto active decode steps,
    /// accounting for both compute FLOPs and HBM bandwidth.
    ChunkedPrefillTokenBudget,
    /// Policy 5 (Ours): Token-budget chunked prefill + EDF/slack-aware decode prioritisation +
    /// KV-pressure admission control + automatic `C == 1` single-user fast-path bypass.
    LatencyAwareChunkedKvScheduler,
}

impl SchedulingPolicy {
    /// Returns all 5 matched policies in canonical comparison order.
    #[must_use]
    pub const fn all() -> [Self; 5] {
        [
            Self::StaticBatchingFcfs,
            Self::ContinuousBatchingPrefillFirst,
            Self::ContinuousBatchingDecodePriority,
            Self::ChunkedPrefillTokenBudget,
            Self::LatencyAwareChunkedKvScheduler,
        ]
    }

    /// Stable lowercase identifier for JSON receipts and markdown scorecards.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::StaticBatchingFcfs => "static_batching_fcfs",
            Self::ContinuousBatchingPrefillFirst => "continuous_prefill_first",
            Self::ContinuousBatchingDecodePriority => "continuous_decode_priority",
            Self::ChunkedPrefillTokenBudget => "chunked_prefill_token_budget",
            Self::LatencyAwareChunkedKvScheduler => "latency_aware_chunked_kv",
        }
    }
}

/// Mechanism used when KV block pressure forces eviction of an active sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionMode {
    /// Free KV blocks + GDN slot and recompute prompt/prefix tokens upon re-admission.
    Recompute,
    /// Copy KV blocks + FP32 GDN state over PCIe to host RAM and restore when GPU space frees.
    SwapToHost,
}

impl PreemptionMode {
    /// Stable lowercase identifier.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Recompute => "recompute",
            Self::SwapToHost => "swap_to_host",
        }
    }
}

/// Service-level objective (SLO) latency targets for interactive and mixed workloads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SloTargets {
    /// Target `p95` time-to-first-token (ms) for short interactive chats.
    pub short_ttft_p95_ms: f64,
    /// Target `p95` time-to-first-token (ms) for long repo prefills (32K tokens).
    pub long_ttft_p95_ms: f64,
    /// Target `p95` inter-token latency (ms) for active decode streams.
    pub itl_p95_ms: f64,
}

impl Default for SloTargets {
    fn default() -> Self {
        Self {
            short_ttft_p95_ms: 400.0,
            long_ttft_p95_ms: 6_000.0,
            itl_p95_ms: 50.0,
        }
    }
}

/// Complete configuration for [`crate::scheduler::SchedulerEngine`].
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerConfig {
    /// Active operating profile.
    pub profile: SchedulerProfile,
    /// Scheduling policy algorithm.
    pub policy: SchedulingPolicy,
    /// Maximum simultaneous resident sequences (capped by GDN slot capacity).
    pub max_num_seqs: usize,
    /// Paged attention KV block size in tokens (default `16`).
    pub block_size: usize,
    /// Maximum total tokens (prefill + decode) packed into one chunked iteration.
    pub token_budget: usize,
    /// Minimum dynamic prefill chunk size used by `LatencyAwareChunkedKvScheduler`
    /// to stay above the HBM roofline ridge point and avoid weight-reload thrashing.
    pub min_prefill_chunk: usize,
    /// Maximum dynamic prefill chunk size per iteration.
    pub max_prefill_chunk: usize,
    /// Enable automatic zero-overhead single-user bypass when `concurrency == 1`.
    pub enable_single_user_bypass: bool,
    /// Enable KV-pressure headroom admission control (reserves decode growth blocks).
    pub kv_admission_control: bool,
    /// Number of free KV blocks reserved per active decoding sequence under admission control.
    pub decode_headroom_blocks_per_seq: usize,
    /// High-water KV utilization fraction (`0.0..=1.0`) above which new long prefills wait.
    pub kv_high_watermark: f64,
    /// Preemption mechanism if an active decode still runs out of KV blocks.
    pub preemption_mode: PreemptionMode,
    /// Decode-priority starvation guard: max consecutive decode-only steps before admitting
    /// a waiting prefill under `ContinuousBatchingDecodePriority`.
    pub decode_priority_starvation_cap: usize,
    /// Artificial batch formation wait window (ms) for `StaticBatchingFcfs` (`0.0` for interactive).
    pub static_batch_wait_ms: f64,
    /// Latency SLO targets.
    pub slo: SloTargets,
    /// Optional label override for benchmark regime reporting (e.g. negative regimes).
    pub regime_label: Option<&'static str>,
}

impl SchedulerConfig {
    /// Validates configuration invariants.
    ///
    /// # Errors
    /// Returns [`SchedulerError::InvalidConfig`] on invalid parameter bounds.
    pub fn validate(&self) -> Result<(), SchedulerError> {
        if self.max_num_seqs == 0 {
            return Err(SchedulerError::InvalidConfig(
                "max_num_seqs must be > 0".to_string(),
            ));
        }
        if self.block_size == 0 {
            return Err(SchedulerError::InvalidConfig(
                "block_size must be > 0".to_string(),
            ));
        }
        if self.token_budget == 0 {
            return Err(SchedulerError::InvalidConfig(
                "token_budget must be > 0".to_string(),
            ));
        }
        if self.min_prefill_chunk == 0 || self.min_prefill_chunk > self.max_prefill_chunk {
            return Err(SchedulerError::InvalidConfig(
                "min_prefill_chunk must be in 1..=max_prefill_chunk".to_string(),
            ));
        }
        if !(0.1..=1.0).contains(&self.kv_high_watermark) {
            return Err(SchedulerError::InvalidConfig(
                "kv_high_watermark must be in 0.1..=1.0".to_string(),
            ));
        }
        Ok(())
    }

    /// Canonical [`SchedulerProfile::SingleUserInteractive`] configuration: zero batch wait,
    /// unchunked prefill fast path, strictly protecting `C = 1` TTFT and ITL.
    #[must_use]
    pub fn single_user_interactive() -> Self {
        Self {
            profile: SchedulerProfile::SingleUserInteractive,
            policy: SchedulingPolicy::LatencyAwareChunkedKvScheduler,
            max_num_seqs: 4,
            block_size: 16,
            token_budget: 4096,
            min_prefill_chunk: 512,
            max_prefill_chunk: 4096,
            enable_single_user_bypass: true,
            kv_admission_control: true,
            decode_headroom_blocks_per_seq: 4,
            kv_high_watermark: 0.90,
            preemption_mode: PreemptionMode::Recompute,
            decode_priority_starvation_cap: 64,
            static_batch_wait_ms: 0.0,
            slo: SloTargets::default(),
            regime_label: Some("profile_single_user_interactive"),
        }
    }

    /// Canonical [`SchedulerProfile::DatacenterThroughput`] configuration: high token budget
    /// (`2048`) with token-budget chunked prefill + decode piggybacking.
    #[must_use]
    pub fn datacenter_throughput() -> Self {
        Self {
            profile: SchedulerProfile::DatacenterThroughput,
            policy: SchedulingPolicy::ChunkedPrefillTokenBudget,
            max_num_seqs: 256,
            block_size: 16,
            token_budget: 2048,
            min_prefill_chunk: 512,
            max_prefill_chunk: 2048,
            enable_single_user_bypass: false,
            kv_admission_control: true,
            decode_headroom_blocks_per_seq: 2,
            kv_high_watermark: 0.92,
            preemption_mode: PreemptionMode::Recompute,
            decode_priority_starvation_cap: 64,
            static_batch_wait_ms: 0.0,
            slo: SloTargets::default(),
            regime_label: Some("profile_datacenter_throughput"),
        }
    }

    /// Canonical [`SchedulerProfile::LatencyAwareSlo`] configuration (our recommended policy):
    /// EDF/slack decode prioritisation + roofline-bounded dynamic chunk sizing + KV-pressure
    /// admission control + automatic `C == 1` single-user fast-path bypass.
    #[must_use]
    pub fn latency_aware_slo() -> Self {
        Self {
            profile: SchedulerProfile::LatencyAwareSlo,
            policy: SchedulingPolicy::LatencyAwareChunkedKvScheduler,
            max_num_seqs: 256,
            block_size: 16,
            token_budget: 1024,
            min_prefill_chunk: 256,
            max_prefill_chunk: 1024,
            enable_single_user_bypass: true,
            kv_admission_control: true,
            decode_headroom_blocks_per_seq: 4,
            kv_high_watermark: 0.85,
            preemption_mode: PreemptionMode::Recompute,
            decode_priority_starvation_cap: 64,
            static_batch_wait_ms: 0.0,
            slo: SloTargets::default(),
            regime_label: None,
        }
    }

    /// Builds a matched configuration for any of the 5 scheduling policies sharing identical
    /// `block_size = 16`, `max_num_seqs = 256`, and SLO targets.
    #[must_use]
    pub fn for_policy(policy: SchedulingPolicy) -> Self {
        match policy {
            SchedulingPolicy::StaticBatchingFcfs => Self {
                profile: SchedulerProfile::DatacenterThroughput,
                policy,
                max_num_seqs: 256,
                block_size: 16,
                token_budget: 4096,
                min_prefill_chunk: 512,
                max_prefill_chunk: 4096,
                enable_single_user_bypass: false,
                kv_admission_control: true,
                decode_headroom_blocks_per_seq: 2,
                kv_high_watermark: 0.92,
                preemption_mode: PreemptionMode::Recompute,
                decode_priority_starvation_cap: 64,
                static_batch_wait_ms: 8.0,
                slo: SloTargets::default(),
                regime_label: None,
            },
            SchedulingPolicy::ContinuousBatchingPrefillFirst => Self {
                profile: SchedulerProfile::DatacenterThroughput,
                policy,
                max_num_seqs: 256,
                block_size: 16,
                token_budget: 4096,
                min_prefill_chunk: 512,
                max_prefill_chunk: 4096,
                enable_single_user_bypass: false,
                kv_admission_control: true,
                decode_headroom_blocks_per_seq: 2,
                kv_high_watermark: 0.92,
                preemption_mode: PreemptionMode::Recompute,
                decode_priority_starvation_cap: 64,
                static_batch_wait_ms: 0.0,
                slo: SloTargets::default(),
                regime_label: None,
            },
            SchedulingPolicy::ContinuousBatchingDecodePriority => Self {
                profile: SchedulerProfile::DatacenterThroughput,
                policy,
                max_num_seqs: 256,
                block_size: 16,
                token_budget: 4096,
                min_prefill_chunk: 512,
                max_prefill_chunk: 4096,
                enable_single_user_bypass: false,
                kv_admission_control: true,
                decode_headroom_blocks_per_seq: 2,
                kv_high_watermark: 0.92,
                preemption_mode: PreemptionMode::Recompute,
                decode_priority_starvation_cap: 64,
                static_batch_wait_ms: 0.0,
                slo: SloTargets::default(),
                regime_label: None,
            },
            SchedulingPolicy::ChunkedPrefillTokenBudget => Self {
                profile: SchedulerProfile::DatacenterThroughput,
                policy,
                max_num_seqs: 256,
                block_size: 16,
                token_budget: 1024,
                min_prefill_chunk: 256,
                max_prefill_chunk: 1024,
                enable_single_user_bypass: false,
                kv_admission_control: true,
                decode_headroom_blocks_per_seq: 2,
                kv_high_watermark: 0.92,
                preemption_mode: PreemptionMode::Recompute,
                decode_priority_starvation_cap: 64,
                static_batch_wait_ms: 0.0,
                slo: SloTargets::default(),
                regime_label: None,
            },
            SchedulingPolicy::LatencyAwareChunkedKvScheduler => Self::latency_aware_slo(),
        }
    }

    /// Honest Negative Regime 1: Datacenter small-chunk (`256`) batching forced at `C = 1`
    /// without single-user bypass, exposing kernel-launch & weight-read tax on single-user latency.
    #[must_use]
    pub fn neg_single_user_small_chunk() -> Self {
        let mut cfg = Self::for_policy(SchedulingPolicy::ChunkedPrefillTokenBudget);
        cfg.token_budget = 256;
        cfg.min_prefill_chunk = 256;
        cfg.max_prefill_chunk = 256;
        cfg.enable_single_user_bypass = false;
        cfg.regime_label = Some("neg_single_user_concurrency_1");
        cfg
    }

    /// Honest Negative Regime 2: Unguarded KV admission (`kv_admission_control = false`) at
    /// `C = 256` with long 32K prompts exceeding GPU KV capacity -> preemption/recompute thrashing.
    #[must_use]
    pub fn neg_unguarded_kv_oversubscription() -> Self {
        let mut cfg = Self::for_policy(SchedulingPolicy::ChunkedPrefillTokenBudget);
        cfg.kv_admission_control = false;
        cfg.decode_headroom_blocks_per_seq = 0;
        cfg.kv_high_watermark = 1.0;
        cfg.preemption_mode = PreemptionMode::Recompute;
        cfg.regime_label = Some("neg_kv_memory_oversubscription_256");
        cfg
    }

    /// Honest Negative Regime 3: Ultra-small chunk (`chunk_size = 64`) that wins single-decode
    /// ITL microbenchmarks while triggering severe weight-reload prefill tax and regressing
    /// end-to-end 20-turn wall time (must be REJECTED by the end-to-end acceptance gate).
    #[must_use]
    pub fn neg_ultra_small_chunk_64() -> Self {
        let mut cfg = Self::for_policy(SchedulingPolicy::ChunkedPrefillTokenBudget);
        cfg.token_budget = 64;
        cfg.min_prefill_chunk = 64;
        cfg.max_prefill_chunk = 64;
        cfg.enable_single_user_bypass = false;
        cfg.regime_label = Some("neg_ultra_small_chunk_prefill_tax");
        cfg
    }

    /// Returns the reporting label for this configuration.
    #[must_use]
    pub fn label(&self) -> &'static str {
        self.regime_label.unwrap_or_else(|| self.policy.as_str())
    }
}
