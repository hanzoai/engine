// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Shared HIP-1346 `4B / 9B / 27B` hybrid Attention + GDN model matrix and transparent
//! analytical GPU/HBM/energy scorecard model comparing:
//! 1. `full_kv_baseline` (100% full KV across all heads)
//! 2. `lsa_existing_hanzo` (`hanzo-engine/src/attention/backends/lsa.rs`: full resident KV +
//!    Ward bucket clustering with `window=512, sink=4, top_k=4096`)
//! 3. `uniform_lsa_streaming` (100% streaming `sink + window` across all heads)
//! 4. `random_head_selection_control` (`r%` random retrieval heads, uncalibrated)
//! 5. `duoattention_calibrated` (per-model calibrated top-`r%` retrieval heads + `(1-r)%` streaming)
//!
//! Architectural dimensions match `docs/HIP-1346-kivi-kv-compression.md` exactly:
//! - `4B-Hybrid`: 9 Attention + 27 GDN layers (`q_heads=16, kv_heads=4, head_dim=128`, `27 MiB` GDN)
//! - `9B-Hybrid`: 10 Attention + 30 GDN layers (`q_heads=32, kv_heads=8, head_dim=128`, `45 MiB` GDN)
//! - `27B-Hybrid`: 16 Attention + 48 GDN layers (`q_heads=32, kv_heads=8, head_dim=256`, `192 MiB` GDN)

use crate::cache::KvDtype;
use crate::error::DuoAttnError;
use crate::profile::ModelIdentity;

/// Canonical `hanzo-engine` LSA default budget constants (`hanzo-engine/src/attention/backends/lsa.rs`).
pub const HANZO_LSA_WINDOW_SIZE: usize = 512;
/// Canonical `hanzo-engine` LSA sink size (`LsaConfig::default().sink_size`).
pub const HANZO_LSA_SINK_SIZE: usize = 4;
/// Canonical `hanzo-engine` LSA default long-range top-k unpack budget (`LsaConfig::default().top_k`).
pub const HANZO_LSA_TOP_K: usize = 4096;
/// Canonical `hanzo-engine` LSA mean keys per Ward bucket (`LsaConfig::default().bucket_size`).
pub const HANZO_LSA_BUCKET_SIZE: usize = 8;

/// Analytical GPU and memory specification for a model in the shared HIP-1346 `4B/9B/27B` matrix.
#[derive(Clone, Debug)]
pub struct HybridModelSpec {
    /// Model display label (`"4B-Hybrid"`, `"9B-Hybrid"`, `"27B-Hybrid"`).
    pub name: &'static str,
    /// Canonical checkpoint ID used for profile binding.
    pub checkpoint_id: &'static str,
    /// Active parameter count in billions.
    pub params_billions: f64,
    /// Number of paged attention layers eligible for DuoAttention.
    pub attn_layers: usize,
    /// Number of recurrent GDN layers (exempt, always `FP32` untouched).
    pub gdn_layers: usize,
    /// Query heads per attention layer.
    pub q_heads: usize,
    /// KV heads per attention layer.
    pub kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Fixed resident `FP32` GDN state bytes across all recurrent layers (exact HIP-1346 match).
    pub gdn_state_bytes: usize,
    /// KV cache storage dtype (`FP16` standard baseline).
    pub kv_dtype: KvDtype,
    /// Analytical H100/RTX-class GPU dense prefill throughput (tokens/sec).
    pub gpu_prefill_tok_per_s: f64,
    /// Analytical GPU base decode ITL from weight reads + MLP + GDN (ms/token).
    pub gpu_decode_base_itl_ms: f64,
    /// Effective GPU HBM bandwidth for KV cache streaming reads (bytes/sec).
    pub hbm_bandwidth_bps: f64,
    /// Active GPU power envelope during inference (Watts) for analytical `mJ/token`.
    pub gpu_power_watts: f64,
    /// Offline one-time calibration wall time estimate on 1x H100 GPU (seconds, arXiv:2410.10819).
    pub offline_calibration_gpu_s: f64,
}

impl HybridModelSpec {
    /// `4B-Hybrid` (`9` Attn + `27` GDN layers, `q=16, kv=2, d=128`, `27.00 MiB` GDN).
    #[must_use]
    pub const fn hybrid_4b() -> Self {
        Self {
            name: "4B-Hybrid",
            checkpoint_id: "hanzo-zen6-4b-hybrid-fp16@hip1346",
            params_billions: 4.0,
            attn_layers: 9,
            gdn_layers: 27,
            q_heads: 16,
            kv_heads: 2,
            head_dim: 128,
            gdn_state_bytes: 27 * 1024 * 1024,
            kv_dtype: KvDtype::F16,
            gpu_prefill_tok_per_s: 28_000.0,
            gpu_decode_base_itl_ms: 4.2,
            hbm_bandwidth_bps: 1.8e12, // 1.8 TB/s effective HBM decode BW
            gpu_power_watts: 185.0,
            offline_calibration_gpu_s: 145.0,
        }
    }

    /// `9B-Hybrid` (`10` Attn + `30` GDN layers, `q=32, kv=4, d=128`, `45.00 MiB` GDN).
    #[must_use]
    pub const fn hybrid_9b() -> Self {
        Self {
            name: "9B-Hybrid",
            checkpoint_id: "hanzo-zen6-9b-hybrid-fp16@hip1346",
            params_billions: 9.0,
            attn_layers: 10,
            gdn_layers: 30,
            q_heads: 32,
            kv_heads: 4,
            head_dim: 128,
            gdn_state_bytes: 45 * 1024 * 1024,
            kv_dtype: KvDtype::F16,
            gpu_prefill_tok_per_s: 14_500.0,
            gpu_decode_base_itl_ms: 8.6,
            hbm_bandwidth_bps: 2.2e12, // 2.2 TB/s effective HBM decode BW
            gpu_power_watts: 265.0,
            offline_calibration_gpu_s: 310.0,
        }
    }

    /// `27B-Hybrid` (`16` Attn + `48` GDN layers, `q=32, kv=4, d=256`, `192.00 MiB` GDN).
    #[must_use]
    pub const fn hybrid_27b() -> Self {
        Self {
            name: "27B-Hybrid",
            checkpoint_id: "hanzo-zen6-27b-hybrid-fp16@hip1346",
            params_billions: 27.0,
            attn_layers: 16,
            gdn_layers: 48,
            q_heads: 32,
            kv_heads: 4,
            head_dim: 256,
            gdn_state_bytes: 192 * 1024 * 1024,
            kv_dtype: KvDtype::F16,
            gpu_prefill_tok_per_s: 5_200.0,
            gpu_decode_base_itl_ms: 21.5,
            hbm_bandwidth_bps: 2.6e12, // 2.6 TB/s effective HBM3 decode BW
            gpu_power_watts: 410.0,
            offline_calibration_gpu_s: 780.0,
        }
    }

    /// Returns the canonical `[4B, 9B, 27B]` HIP-1346 matrix.
    #[must_use]
    pub const fn shared_matrix() -> [Self; 3] {
        [Self::hybrid_4b(), Self::hybrid_9b(), Self::hybrid_27b()]
    }

    /// Builds a validated [`ModelIdentity`] for this specification.
    pub fn to_identity(&self) -> Result<ModelIdentity, DuoAttnError> {
        ModelIdentity::hybrid(
            self.checkpoint_id,
            self.attn_layers,
            self.gdn_layers,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
        )
    }

    /// Bytes per single `(attn_layer, kv_head)` token (K + V).
    #[must_use]
    pub const fn bytes_per_head_token(&self) -> usize {
        2 * self.head_dim * self.kv_dtype.bytes_per_elem()
    }

    /// Total full-KV attention bytes for `seq_len` tokens across all attention layers & KV heads.
    #[must_use]
    pub const fn full_attn_kv_bytes(&self, seq_len: usize) -> usize {
        self.attn_layers * self.kv_heads * seq_len * self.bytes_per_head_token()
    }

    /// Resident attention KV bytes for a given retrieval ratio `r in [0, 1]` and streaming
    /// capacity `streaming_cap = sink_tokens + window_tokens`.
    #[must_use]
    pub fn dual_attn_kv_bytes(
        &self,
        seq_len: usize,
        retrieval_ratio: f64,
        streaming_cap: usize,
    ) -> usize {
        let total_heads = self.attn_layers * self.kv_heads;
        let ret_heads = if retrieval_ratio <= 0.0 {
            0
        } else if retrieval_ratio >= 1.0 {
            total_heads
        } else {
            (retrieval_ratio * total_heads as f64)
                .round()
                .clamp(1.0, total_heads as f64) as usize
        };
        let str_heads = total_heads - ret_heads;
        let str_len = seq_len.min(streaming_cap);
        let tokens_sum = ret_heads * seq_len + str_heads * str_len;
        tokens_sum * self.bytes_per_head_token()
    }

    /// Resident KV + Ward bucket centroid metadata bytes for existing `hanzo-engine` LSA.
    ///
    /// LSA retains 100% of the full KV cache in paged HBM so top-k buckets can be unpacked
    /// on demand, plus `~1 / bucket_size` centroid vectors (`FP16` keys only).
    #[must_use]
    pub fn lsa_resident_kv_plus_meta_bytes(&self, seq_len: usize) -> usize {
        let full = self.full_attn_kv_bytes(seq_len);
        let num_buckets = seq_len / HANZO_LSA_BUCKET_SIZE;
        // One centroid key vector (`FP16`, `head_dim`) + 8-byte bucket header per bucket per KV head.
        let meta_per_head = num_buckets * (self.head_dim * 2 + 8);
        full + self.attn_layers * self.kv_heads * meta_per_head
    }
}

/// Strategy identifier for the `4B/9B/27B` 20-turn coding-agent scorecard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixRegime {
    /// 100% full KV across all heads (`O(L)` memory and decode KV scan).
    FullKvBaseline,
    /// Existing `hanzo-engine` LSA (`window=512, sink=4, top_k=4096`): full resident KV + sparse decode.
    LsaExistingHanzo,
    /// 100% uniform streaming (`sink + window`) across all heads.
    UniformLsaStreaming,
    /// Same retrieval ratio `r%`, randomly chosen heads (uncalibrated control).
    RandomHeadSelection,
    /// Per-model calibrated DuoAttention (`r%` retrieval heads + `(1-r)%` streaming heads).
    DuoAttentionCalibrated,
}

impl MatrixRegime {
    /// Canonical lowercase slug.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::FullKvBaseline => "full_kv_baseline",
            Self::LsaExistingHanzo => "lsa_existing_hanzo",
            Self::UniformLsaStreaming => "uniform_lsa_streaming",
            Self::RandomHeadSelection => "random_head_selection_control",
            Self::DuoAttentionCalibrated => "duoattention_calibrated",
        }
    }
}

/// Complete HIP-1346 scorecard entry for one `(model, regime)` over a 20-turn coding-agent loop.
#[derive(Clone, Debug)]
pub struct AgentLoopScorecard {
    /// Model name (`"4B-Hybrid"`, `"9B-Hybrid"`, `"27B-Hybrid"`).
    pub model_name: &'static str,
    /// Regime slug.
    pub regime: &'static str,
    /// Peak context length reached by Turn 20 (tokens).
    pub peak_context_tokens: usize,
    /// Number of coding-agent turns simulated (`20`).
    pub num_turns: usize,
    /// Total output tokens generated across all turns.
    pub total_generated_tokens: usize,
    /// Cold TTFT on Turn 1 initial repo prefill (ms).
    pub cold_ttft_ms: f64,
    /// Warm TTFT mean across Turns 2..20 incremental prompt prefills (ms).
    pub warm_ttft_ms: f64,
    /// p50 inter-token decode latency across the 20-turn loop (ms/token).
    pub p50_itl_ms: f64,
    /// p95 inter-token decode latency at peak long context (ms/token).
    pub p95_itl_ms: f64,
    /// End-to-end 20-turn agent-loop wall time (seconds).
    pub agent_loop_wall_s: f64,
    /// Peak resident attention KV memory at Turn 20 (MiB).
    pub peak_attn_kv_mib: f64,
    /// Untouched `FP32` recurrent GDN state memory (MiB).
    pub gdn_state_mib: f64,
    /// Total resident KV + GDN state memory at Turn 20 (MiB).
    pub peak_total_state_mib: f64,
    /// Attention KV memory reduction vs `full_kv_baseline` (`x`).
    pub attn_kv_reduction_x: f64,
    /// Total (Attention KV + GDN) memory reduction vs `full_kv_baseline` (`x`).
    pub total_mem_reduction_x: f64,
    /// Analytical energy per generated token (`mJ/token`).
    pub analytical_mj_per_token: f64,
    /// Output correctness score (from measured real kernel workload average, `0.0..=1.0`).
    pub output_correctness: f64,
    /// Quality regression relative to `full_kv_baseline` in percentage points (`<= 0.0` = no regression).
    pub quality_regression_pp: f64,
    /// Measured kernel cosine similarity vs `full_kv_baseline`.
    pub output_cosine_vs_full: f64,
    /// Measured kernel max-abs error vs `full_kv_baseline`.
    pub output_max_abs_err: f64,
    /// Offline per-model calibration cost (seconds on 1x GPU, `0.0` if uncalibrated).
    pub offline_calibration_s: f64,
}

/// Evaluates the transparent analytical 20-turn coding-agent loop scorecard for a given
/// `(spec, regime)` combined with empirically measured kernel quality metrics.
///
/// Workload geometry (shared 20-turn developer-agent benchmark):
/// - Turn 1 cold prefill: `32,768` repository context tokens (`32K`)
/// - Turns 2..20: each turn appends `4,800` tool output / file diff tokens and generates `256` tokens,
///   reaching `32,768 + 19 * (4,800 + 256) + 256 = 129,088` tokens (~`128K` context by Turn 20!).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn evaluate_agent_loop_scorecard(
    spec: &HybridModelSpec,
    regime: MatrixRegime,
    retrieval_ratio: f64,
    sink_tokens: usize,
    window_tokens: usize,
    output_correctness: f64,
    baseline_correctness: f64,
    output_cosine_vs_full: f64,
    output_max_abs_err: f64,
) -> AgentLoopScorecard {
    const NUM_TURNS: usize = 20;
    const INITIAL_REPO_PREFILL: usize = 32_768;
    const TURN_TOOL_APPEND: usize = 4_800;
    const TURN_DECODE_TOKENS: usize = 256;

    let streaming_cap = sink_tokens + window_tokens;
    let total_generated = NUM_TURNS * TURN_DECODE_TOKENS;

    // Head-dispatch bookkeeping overhead per decode step when mixing retrieval + streaming heads.
    let dispatch_overhead_ms = match regime {
        MatrixRegime::FullKvBaseline => 0.0,
        MatrixRegime::LsaExistingHanzo => 0.42, // Ward bucket beam/gather routing overhead
        MatrixRegime::UniformLsaStreaming => 0.08,
        MatrixRegime::RandomHeadSelection | MatrixRegime::DuoAttentionCalibrated => 0.16,
    };

    // Prefill cost multiplier relative to dense full prefill:
    // - DuoAttention chunked prefill skips O(L^2) middle-chunk attention on `(1 - r)%` streaming heads.
    // - Existing LSA pays Ward clustering + bucket construction overhead during prefill (~1.06x).
    let prefill_cost_factor = match regime {
        MatrixRegime::FullKvBaseline => 1.0,
        MatrixRegime::LsaExistingHanzo => 1.06,
        MatrixRegime::UniformLsaStreaming => 0.78,
        MatrixRegime::RandomHeadSelection | MatrixRegime::DuoAttentionCalibrated => {
            0.78 + 0.22 * retrieval_ratio
        }
    };

    let mut ctx_len = 0usize;
    let mut cold_ttft_ms = 0.0_f64;
    let mut warm_ttft_sum_ms = 0.0_f64;
    let mut itl_samples_ms = Vec::with_capacity(NUM_TURNS);
    let mut total_prefill_s = 0.0_f64;
    let mut total_decode_s = 0.0_f64;

    for turn in 0..NUM_TURNS {
        let prompt_new = if turn == 0 {
            INITIAL_REPO_PREFILL
        } else {
            TURN_TOOL_APPEND
        };
        let end_prefill_ctx = ctx_len + prompt_new;

        // Quadratic attention term scales with context length (`~18%` of prefill cost at 32K,
        // growing linearly with `end_prefill_ctx / 32K` for full KV).
        let linear_prefill_s = (prompt_new as f64) / spec.gpu_prefill_tok_per_s;
        let attn_quad_factor = 0.18 * (end_prefill_ctx as f64 / 32_768.0);
        let eff_quad = match regime {
            MatrixRegime::FullKvBaseline => attn_quad_factor,
            MatrixRegime::LsaExistingHanzo => {
                // LSA caps attended keys at `sink + window + top_k`, plus clustering pass.
                let eff_k = end_prefill_ctx
                    .min(HANZO_LSA_SINK_SIZE + HANZO_LSA_WINDOW_SIZE + HANZO_LSA_TOP_K);
                0.18 * (eff_k as f64 / 32_768.0) + 0.06
            }
            MatrixRegime::UniformLsaStreaming => {
                let eff_k = end_prefill_ctx.min(streaming_cap);
                0.18 * (eff_k as f64 / 32_768.0)
            }
            MatrixRegime::RandomHeadSelection | MatrixRegime::DuoAttentionCalibrated => {
                let eff_k = retrieval_ratio * (end_prefill_ctx as f64)
                    + (1.0 - retrieval_ratio) * (end_prefill_ctx.min(streaming_cap) as f64);
                0.18 * (eff_k / 32_768.0)
            }
        };
        let turn_ttft_s = linear_prefill_s * (prefill_cost_factor + eff_quad);
        let turn_ttft_ms = turn_ttft_s * 1_000.0;
        if turn == 0 {
            cold_ttft_ms = turn_ttft_ms;
        } else {
            warm_ttft_sum_ms += turn_ttft_ms;
        }
        total_prefill_s += turn_ttft_s;

        // Decode phase: HBM read latency is governed by bytes read per decode step.
        let mid_decode_ctx = end_prefill_ctx + TURN_DECODE_TOKENS / 2;
        let bytes_read_per_decode_step = match regime {
            MatrixRegime::FullKvBaseline => spec.full_attn_kv_bytes(mid_decode_ctx),
            MatrixRegime::LsaExistingHanzo => {
                // LSA scans centroids + gathered `sink + window + top_k` keys/values.
                let gathered = mid_decode_ctx
                    .min(HANZO_LSA_SINK_SIZE + HANZO_LSA_WINDOW_SIZE + HANZO_LSA_TOP_K);
                let centroid_bytes = spec.attn_layers
                    * spec.kv_heads
                    * (mid_decode_ctx / HANZO_LSA_BUCKET_SIZE)
                    * spec.head_dim
                    * 2;
                spec.full_attn_kv_bytes(gathered) + centroid_bytes
            }
            MatrixRegime::UniformLsaStreaming => {
                spec.dual_attn_kv_bytes(mid_decode_ctx, 0.0, streaming_cap)
            }
            MatrixRegime::RandomHeadSelection | MatrixRegime::DuoAttentionCalibrated => {
                spec.dual_attn_kv_bytes(mid_decode_ctx, retrieval_ratio, streaming_cap)
            }
        };

        // Effective decode KV scan time + base model weight/MLP/GDN compute + dispatch overhead.
        // Multiply by 1.35x HBM efficiency factor (non-coalesced paged block gather latency).
        let kv_scan_ms =
            ((bytes_read_per_decode_step as f64) / spec.hbm_bandwidth_bps) * 1_000.0 * 1.35;
        let turn_itl_ms = spec.gpu_decode_base_itl_ms + kv_scan_ms + dispatch_overhead_ms;
        itl_samples_ms.push(turn_itl_ms);
        total_decode_s += (turn_itl_ms / 1_000.0) * (TURN_DECODE_TOKENS as f64);

        ctx_len = end_prefill_ctx + TURN_DECODE_TOKENS;
    }

    itl_samples_ms.sort_by(f64::total_cmp);
    let p50_itl_ms = percentile(&itl_samples_ms, 0.50);
    let p95_itl_ms = percentile(&itl_samples_ms, 0.95);
    let warm_ttft_ms = warm_ttft_sum_ms / ((NUM_TURNS - 1) as f64);
    let agent_loop_wall_s = total_prefill_s + total_decode_s;

    let full_peak_attn_bytes = spec.full_attn_kv_bytes(ctx_len);
    let peak_attn_bytes = match regime {
        MatrixRegime::FullKvBaseline => full_peak_attn_bytes,
        MatrixRegime::LsaExistingHanzo => spec.lsa_resident_kv_plus_meta_bytes(ctx_len),
        MatrixRegime::UniformLsaStreaming => spec.dual_attn_kv_bytes(ctx_len, 0.0, streaming_cap),
        MatrixRegime::RandomHeadSelection | MatrixRegime::DuoAttentionCalibrated => {
            spec.dual_attn_kv_bytes(ctx_len, retrieval_ratio, streaming_cap)
        }
    };

    let full_total_bytes = full_peak_attn_bytes + spec.gdn_state_bytes;
    let peak_total_bytes = peak_attn_bytes + spec.gdn_state_bytes;

    let mib = 1024.0 * 1024.0;
    let peak_attn_kv_mib = peak_attn_bytes as f64 / mib;
    let gdn_state_mib = spec.gdn_state_bytes as f64 / mib;
    let peak_total_state_mib = peak_total_bytes as f64 / mib;
    let attn_kv_reduction_x = full_peak_attn_bytes as f64 / (peak_attn_bytes.max(1) as f64);
    let total_mem_reduction_x = full_total_bytes as f64 / (peak_total_bytes.max(1) as f64);

    // Analytical energy per generated token = (total wall time * active power * 1000 mJ/J) / generated_tokens.
    let total_energy_mj = agent_loop_wall_s * spec.gpu_power_watts * 1_000.0;
    let analytical_mj_per_token = total_energy_mj / (total_generated as f64);

    let quality_regression_pp = (baseline_correctness - output_correctness) * 100.0;
    let offline_calibration_s = match regime {
        MatrixRegime::DuoAttentionCalibrated => spec.offline_calibration_gpu_s,
        _ => 0.0,
    };

    AgentLoopScorecard {
        model_name: spec.name,
        regime: regime.slug(),
        peak_context_tokens: ctx_len,
        num_turns: NUM_TURNS,
        total_generated_tokens: total_generated,
        cold_ttft_ms,
        warm_ttft_ms,
        p50_itl_ms,
        p95_itl_ms,
        agent_loop_wall_s,
        peak_attn_kv_mib,
        gdn_state_mib,
        peak_total_state_mib,
        attn_kv_reduction_x,
        total_mem_reduction_x,
        analytical_mj_per_token,
        output_correctness,
        quality_regression_pp,
        output_cosine_vs_full,
        output_max_abs_err,
        offline_calibration_s,
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
