// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! HIP-1350 5-axis telemetry structures for optimistic AI execution and
//! interactive bisection fraud-proof benchmarks.

/// Reference cloud vCPU-hour pricing (`\$0.05 / vCPU-hour`) used to convert
/// measured CPU execution seconds into transparent `$/1M tokens` cost telemetry.
pub const REFERENCE_VCPU_USD_PER_HOUR: f64 = 0.05;

/// Axis 1 (Cost) telemetry for a verification regime.
#[derive(Debug, Clone)]
pub struct CostAxis {
    /// Provider inference compute cost in USD per 1M generated tokens.
    pub provider_usd_per_1m_tokens: f64,
    /// Verification & amortized dispute compute cost in USD per 1M generated tokens.
    pub verification_usd_per_1m_tokens: f64,
    /// Total end-to-end cost (`provider + verification`) in USD per 1M generated tokens.
    pub total_usd_per_1m_tokens: f64,
    /// Ratio of verification compute time to provider compute time (`0.0` = unverified, `1.0` = full re-exec).
    pub verification_cost_ratio: f64,
}

/// Axis 2 (Performance) telemetry for a verification regime.
#[derive(Debug, Clone)]
pub struct PerformanceAxis {
    /// Mean time-to-first-token (ms) on the measured host.
    pub ttft_ms: f64,
    /// Effective end-to-end throughput (generated tokens / sec, including verification compute).
    pub throughput_tok_per_sec: f64,
    /// Mean settlement latency for unchallenged claims (seconds; equals `challenge_window_secs` in optimistic mode).
    pub optimistic_settlement_latency_secs: f64,
    /// Mean settlement latency for disputed claims (seconds; includes `ceil(log2 T)` interactive rounds).
    pub disputed_settlement_latency_secs: f64,
    /// Mean bisection rounds per disputed claim.
    pub mean_bisection_rounds: f64,
    /// Theoretical ceiling `ceil(log2 T)` for the workload.
    pub theoretical_max_rounds: u32,
    /// Mean single-step judge execution latency (microseconds).
    pub mean_single_step_judge_us: f64,
    /// Compute speedup factor of verification vs full re-execution (`full_reexec_ns / regime_verify_ns`).
    pub verification_speedup_vs_full: f64,
}

/// Axis 3 (Trust) telemetry for a verification regime.
#[derive(Debug, Clone)]
pub struct TrustAxis {
    /// Percentage of injected single-step corruptions detected and pinpointed to exact step index (`0.0..=100.0`).
    pub fraud_exact_pinpoint_pct: f64,
    /// Percentage of malicious/griefing challenges defeated & slashed (`0.0..=100.0`).
    pub malicious_challenges_defeated_pct: f64,
    /// Percentage of timeout forfeits enforced (`0.0..=100.0`).
    pub timeout_forfeits_enforced_pct: f64,
    /// Percentage of tampered transcripts rejected by sovereign replay (`0.0..=100.0`).
    pub transcript_tamper_rejection_pct: f64,
}

/// Axis 4 (Decentralization) telemetry for a verification regime.
#[derive(Debug, Clone)]
pub struct DecentralizationAxis {
    /// Whether any permissionless external party can challenge without allowlists.
    pub permissionless_challengers: bool,
    /// Number of Hanzo control-plane API calls required during execution, dispute, or replay (`0`).
    pub hanzo_control_plane_calls: u32,
    /// Number of independent challengers exercised in the sweep.
    pub independent_challengers_exercised: u32,
    /// Whether hardware TEEs or vendor attestation keys are assumed (`false`).
    pub tee_assumed: bool,
}

/// Axis 5 (Sovereignty) telemetry for a verification regime.
#[derive(Debug, Clone)]
pub struct SovereigntyAxis {
    /// Percentage of exported transcripts that round-trip losslessly across wire & JSON (`0.0..=100.0`).
    pub lossless_transcript_roundtrip_pct: f64,
    /// Percentage of transcripts independently replayed & verified with 0 control-plane calls (`0.0..=100.0`).
    pub third_party_replay_pass_pct: f64,
    /// Mean binary wire transcript size (bytes).
    pub mean_wire_transcript_bytes: f64,
    /// Mean canonical JSON transcript size (bytes).
    pub mean_json_transcript_bytes: f64,
}

/// Complete 5-axis HIP-1350 measurement bundle for one benchmark regime.
#[derive(Debug, Clone)]
pub struct RegimeTelemetry {
    /// Machine-readable regime name (`unverified_optimistic_control`, `full_reexecution_every_claim`, `hanzo_opai_bisection_game`).
    pub regime: String,
    /// Scenario label (e.g., `fraud_1pct_grief_0pct`).
    pub scenario: String,
    /// Number of claims evaluated.
    pub claims_evaluated: usize,
    /// Total atomic steps `T` per claim.
    pub steps_per_claim: u64,
    /// Injected fraud rate (`0.0..=1.0`).
    pub fraud_rate: f64,
    /// Malicious challenger griefing rate (`0.0..=1.0`).
    pub griefing_rate: f64,
    /// Axis 1: Cost.
    pub cost: CostAxis,
    /// Axis 2: Performance.
    pub performance: PerformanceAxis,
    /// Axis 3: Trust.
    pub trust: TrustAxis,
    /// Axis 4: Decentralization.
    pub decentralization: DecentralizationAxis,
    /// Axis 5: Sovereignty.
    pub sovereignty: SovereigntyAxis,
}

/// Convert total measured nanoseconds across `tokens` generated tokens into USD per 1M tokens
/// at [`REFERENCE_VCPU_USD_PER_HOUR`].
pub fn ns_to_usd_per_1m_tokens(total_ns: u128, tokens: u64) -> f64 {
    if tokens == 0 {
        return 0.0;
    }
    let total_secs = (total_ns as f64) / 1e9;
    let secs_per_1m_tokens = total_secs * (1_000_000.0 / (tokens as f64));
    (secs_per_1m_tokens / 3600.0) * REFERENCE_VCPU_USD_PER_HOUR
}
