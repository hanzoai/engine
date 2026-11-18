//! KV-residency-aware request cost model and tiered residency state (Bidaw, HIP-1346).
//!
//! Estimates expected Time-To-First-Token (`E[TTFT]`) per replica by combining:
//! 1. Resident KV prefix coverage across device HBM (`gpu_tokens`), host DRAM
//!    (`host_tokens`), and local NVMe (`nvme_tokens`),
//! 2. Bounded transfer-vs-recompute fallback (`min(load_ms, recompute_ms)`) so a
//!    slow storage tier never outranks direct prefill recomputation, and
//! 3. In-flight slot contention and EWMA queueing delay on each replica.

use serde::{Deserialize, Serialize};

/// Prefix token counts resident on each memory tier of a worker replica.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvTierResidency {
    /// Prefix tokens currently resident in GPU HBM or unified device memory.
    pub gpu_tokens: usize,
    /// Additional prefix tokens paged to host DRAM beyond `gpu_tokens`.
    pub host_tokens: usize,
    /// Additional prefix tokens paged to local NVMe beyond `gpu_tokens + host_tokens`.
    pub nvme_tokens: usize,
}

impl KvTierResidency {
    pub const fn gpu_only(gpu_tokens: usize) -> Self {
        Self {
            gpu_tokens,
            host_tokens: 0,
            nvme_tokens: 0,
        }
    }

    pub const fn total_tokens(&self) -> usize {
        self.gpu_tokens
            .saturating_add(self.host_tokens)
            .saturating_add(self.nvme_tokens)
    }
}

/// Decomposed TTFT estimate for placing one request on a candidate replica.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TieredEstimate {
    pub gpu_hit_tokens: usize,
    pub host_hit_tokens: usize,
    pub nvme_hit_tokens: usize,
    pub cold_recompute_tokens: usize,
    pub kv_load_ms: f64,
    pub prefill_compute_ms: f64,
    pub queue_wait_ms: f64,
    pub total_ttft_ms: f64,
}

/// Bandwidth and compute cost parameters for KV-residency-aware scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BidawCostModel {
    /// Baseline prefill compute rate (tokens/sec at replica `weight == 100`).
    pub prefill_tok_per_s: f64,
    /// Host DRAM -> GPU KV page-in rate (tokens/sec).
    pub host_kv_load_tok_per_s: f64,
    /// Local NVMe -> GPU KV page-in rate (tokens/sec).
    pub nvme_kv_load_tok_per_s: f64,
    /// Floor contention penalty (ms) at full replica utilization.
    pub slot_contention_ms: f64,
    /// Discretization quantum (ms) for expected-TTFT ranking buckets.
    pub tie_quantum_ms: f64,
}

impl Default for BidawCostModel {
    fn default() -> Self {
        Self {
            prefill_tok_per_s: 1_700.0,
            host_kv_load_tok_per_s: 12_000.0,
            nvme_kv_load_tok_per_s: 3_500.0,
            slot_contention_ms: 180.0,
            tie_quantum_ms: 5.0,
        }
    }
}

fn sanitize_rate(v: f64, fallback: f64) -> f64 {
    if v.is_finite() && v >= 1.0 {
        v
    } else {
        fallback
    }
}

fn sanitize_nonneg(v: f64, fallback: f64) -> f64 {
    if v.is_finite() && v >= 0.0 {
        v
    } else {
        fallback
    }
}

impl BidawCostModel {
    /// Estimate expected TTFT (ms) for `required_tokens` on a replica with `residency`.
    pub fn estimate(
        &self,
        required_tokens: usize,
        residency: KvTierResidency,
        inflight: usize,
        slots: usize,
        ewma_ttft_ms: Option<f64>,
        weight: u32,
    ) -> TieredEstimate {
        let scale = (weight.max(1) as f64) / 100.0;
        let prefill_rate = sanitize_rate(self.prefill_tok_per_s * scale, 1_700.0);
        let host_rate = sanitize_rate(self.host_kv_load_tok_per_s, 12_000.0);
        let nvme_rate = sanitize_rate(self.nvme_kv_load_tok_per_s, 3_500.0);
        let floor_wait = sanitize_nonneg(self.slot_contention_ms, 180.0);

        let gpu_hit = residency.gpu_tokens.min(required_tokens);
        let rem_after_gpu = required_tokens - gpu_hit;

        let host_hit = residency.host_tokens.min(rem_after_gpu);
        let rem_after_host = rem_after_gpu - host_hit;

        let nvme_hit = residency.nvme_tokens.min(rem_after_host);
        let cold_tokens = rem_after_host - nvme_hit;

        let host_load_ms = 1_000.0 * (host_hit as f64) / host_rate;
        let host_recompute_ms = 1_000.0 * (host_hit as f64) / prefill_rate;
        let (eff_host_load_ms, host_fallback_compute_ms) = if host_load_ms <= host_recompute_ms {
            (host_load_ms, 0.0)
        } else {
            (0.0, host_recompute_ms)
        };

        let nvme_load_ms = 1_000.0 * (nvme_hit as f64) / nvme_rate;
        let nvme_recompute_ms = 1_000.0 * (nvme_hit as f64) / prefill_rate;
        let (eff_nvme_load_ms, nvme_fallback_compute_ms) = if nvme_load_ms <= nvme_recompute_ms {
            (nvme_load_ms, 0.0)
        } else {
            (0.0, nvme_recompute_ms)
        };

        let cold_compute_ms = 1_000.0 * (cold_tokens as f64) / prefill_rate;
        let kv_load_ms = eff_host_load_ms + eff_nvme_load_ms;
        let prefill_compute_ms =
            cold_compute_ms + host_fallback_compute_ms + nvme_fallback_compute_ms;

        let cap = slots.max(1);
        let observed = ewma_ttft_ms
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(floor_wait)
            .max(floor_wait);
        let queue_wait_ms = if inflight < cap {
            (inflight as f64 / cap as f64) * observed
        } else {
            let overload = (inflight - cap + 1) as f64;
            observed * (1.0 + overload)
        };

        let total_ttft_ms = kv_load_ms + prefill_compute_ms + queue_wait_ms;
        TieredEstimate {
            gpu_hit_tokens: gpu_hit,
            host_hit_tokens: host_hit,
            nvme_hit_tokens: nvme_hit,
            cold_recompute_tokens: cold_tokens,
            kv_load_ms,
            prefill_compute_ms,
            queue_wait_ms,
            total_ttft_ms,
        }
    }

    /// Quantized bucket index for deterministic total ordering without NaN hazards.
    pub fn bucket(&self, total_ttft_ms: f64) -> u64 {
        let q = sanitize_rate(self.tie_quantum_ms, 5.0);
        if !total_ttft_ms.is_finite() || total_ttft_ms <= 0.0 {
            0
        } else {
            (total_ttft_ms / q).floor() as u64
        }
    }
}
