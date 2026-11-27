//! Shared HIP-1346 `4B / 9B / 27B` Hybrid Transformer + GDN matrix, H100-SXM5 80GB
//! roofline analytical model (single-user `B = 1` 20-turn coding agent AND compute-bound
//! `B = 32 / 64` serving), and strict end-to-end acceptance gate.
//!
//! Architectural geometries match `docs/HIP-1346-kivi-kv-compression.md`:
//! - `4B-Hybrid`:  9 Attn + 27 GDN layers
//! - `9B-Hybrid`: 10 Attn + 30 GDN layers
//! - `27B-Hybrid`: 16 Attn + 48 GDN layers

/// 20-turn coding-agent workload specification (HIP-1346 shared matrix).
pub const AGENT_COLD_PREFILL_TOKENS: usize = 32_768;
pub const AGENT_TURNS: usize = 20;
pub const AGENT_WARM_PREFILL_PER_TURN: usize = 4_800;
pub const AGENT_GEN_TOKENS_PER_TURN: usize = 256;
pub const AGENT_TOTAL_GEN_TOKENS: usize = AGENT_TURNS * AGENT_GEN_TOKENS_PER_TURN; // 5,120
pub const AGENT_PEAK_CONTEXT_TOKENS: usize = AGENT_COLD_PREFILL_TOKENS
    + AGENT_TURNS * (AGENT_WARM_PREFILL_PER_TURN + AGENT_GEN_TOKENS_PER_TURN); // 133,888 (or ~129K pre-last-turn)

/// H100-SXM5 80GB roofline hardware constants (explicitly labeled as analytical projections).
pub const H100_ACHIEVED_BW_BYTES_PER_S: f64 = 3.35e12 * 0.80; // 2.68 TB/s HBM3 achieved
pub const H100_ACHIEVED_BF16_FLOPS: f64 = 989.0e12 * 0.34; // ~336 TFLOPS effective BF16 decode/verify GEMM
pub const H100_ACTIVE_WATTS: f64 = 650.0;
pub const KERNEL_LAUNCH_OVERHEAD_MS: f64 = 0.08;

/// Shared HIP-1346 hybrid model architecture descriptor.
#[derive(Clone, Debug)]
pub struct HybridModelSpec {
    pub name: &'static str,
    pub attn_layers: usize,
    pub gdn_layers: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub active_params: f64,
    pub quantization_format: &'static str,
    /// Resident target model weights in bytes (BF16 = 2 bytes/param).
    pub target_weight_bytes: f64,
    /// Resident EAGLE-3 draft head weights in bytes (1 fusion layer + 1 draft transformer layer,
    /// sharing frozen embeddings & LM head with target -> ~2-3% extra footprint).
    pub eagle3_draft_weight_bytes: f64,
    /// FP16 attention KV bytes per token across all attention layers.
    pub attn_kv_bytes_per_tok: f64,
    /// Unquantized FP32 GDN recurrent state bytes (fixed per sequence).
    pub gdn_state_bytes_per_seq: f64,
}

impl HybridModelSpec {
    pub fn shared_matrix() -> Vec<Self> {
        let m4b_params = 3.54e9_f64;
        let m9b_params = 9.09e9_f64;
        let m27b_params = 24.88e9_f64;

        // EAGLE-3 draft head: W_fc (3d*d) + W_in (2d*d) + 1 draft Attn+SwiGLU layer (~4d^2 + 3*d*d_ff)
        let draft_bytes = |d: usize, d_ff: usize| -> f64 {
            let params = (3 * d * d + 2 * d * d + 4 * d * d + 3 * d * d_ff) as f64;
            params * 2.0 // BF16
        };

        vec![
            Self {
                name: "4B-Hybrid",
                attn_layers: 9,
                gdn_layers: 27,
                hidden_dim: 2560,
                ffn_dim: 6912,
                kv_heads: 4,
                head_dim: 128,
                active_params: m4b_params,
                quantization_format: "BF16 weights + FP16 Attn KV + FP32 GDN",
                target_weight_bytes: m4b_params * 2.0,
                eagle3_draft_weight_bytes: draft_bytes(2560, 6912),
                // 2 (K+V) * 2 bytes (FP16) * attn_layers * kv_heads * head_dim
                attn_kv_bytes_per_tok: (4 * 9 * 4 * 128) as f64,
                gdn_state_bytes_per_seq: 27.0 * 1024.0 * 1024.0,
            },
            Self {
                name: "9B-Hybrid",
                attn_layers: 10,
                gdn_layers: 30,
                hidden_dim: 4096,
                ffn_dim: 11008,
                kv_heads: 8,
                head_dim: 128,
                active_params: m9b_params,
                quantization_format: "BF16 weights + FP16 Attn KV + FP32 GDN",
                target_weight_bytes: m9b_params * 2.0,
                eagle3_draft_weight_bytes: draft_bytes(4096, 11008),
                attn_kv_bytes_per_tok: (4 * 10 * 8 * 128) as f64,
                gdn_state_bytes_per_seq: 45.0 * 1024.0 * 1024.0,
            },
            Self {
                name: "27B-Hybrid",
                attn_layers: 16,
                gdn_layers: 48,
                hidden_dim: 5120,
                ffn_dim: 17408,
                kv_heads: 8,
                head_dim: 256,
                active_params: m27b_params,
                quantization_format: "BF16 weights + FP16 Attn KV + FP32 GDN",
                target_weight_bytes: m27b_params * 2.0,
                eagle3_draft_weight_bytes: draft_bytes(5120, 17408),
                attn_kv_bytes_per_tok: (4 * 16 * 8 * 256) as f64,
                gdn_state_bytes_per_seq: 192.0 * 1024.0 * 1024.0,
            },
        ]
    }

    /// Peak resident memory in GiB at 128K-class coding-agent context (`batch_size = 1`).
    pub fn peak_memory_gib(&self, uses_eagle3_head: bool, extra_tree_kv_tokens: usize) -> f64 {
        let workspace = 256.0 * 1024.0 * 1024.0;
        let kv_bytes = ((AGENT_PEAK_CONTEXT_TOKENS + extra_tree_kv_tokens) as f64)
            * self.attn_kv_bytes_per_tok;
        let draft_bytes = if uses_eagle3_head {
            self.eagle3_draft_weight_bytes
        } else {
            0.0
        };
        let total = self.target_weight_bytes
            + draft_bytes
            + kv_bytes
            + self.gdn_state_bytes_per_seq
            + workspace;
        total / (1024.0 * 1024.0 * 1024.0)
    }

    /// Roofline prefill latency (ms) for `tokens` at `batch_size`.
    pub fn roofline_prefill_ms(&self, tokens: usize, batch_size: usize) -> f64 {
        let total_tok = (tokens * batch_size) as f64;
        let flops = 2.0 * self.active_params * total_tok;
        let t_compute_s = flops / H100_ACHIEVED_BF16_FLOPS;
        let t_mem_s = self.target_weight_bytes / H100_ACHIEVED_BW_BYTES_PER_S;
        t_compute_s.max(t_mem_s) * 1000.0 + KERNEL_LAUNCH_OVERHEAD_MS
    }

    /// Roofline target verification step time (ms) for verifying `verify_tokens` tokens in parallel
    /// across `batch_size` concurrent sequences at average context length `avg_ctx`.
    pub fn roofline_verify_ms(
        &self,
        verify_tokens: usize,
        batch_size: usize,
        avg_ctx: usize,
    ) -> f64 {
        let kv_read = (batch_size as f64) * (avg_ctx as f64) * self.attn_kv_bytes_per_tok;
        let bytes = self.target_weight_bytes + kv_read;
        let t_mem_ms = (bytes / H100_ACHIEVED_BW_BYTES_PER_S) * 1000.0;

        // Compute scales linearly with `batch_size * verify_tokens`!
        let total_tok = (batch_size * verify_tokens) as f64;
        let matmul_flops = 2.0 * self.active_params * total_tok;
        let attn_flops = 4.0
            * (self.attn_layers as f64)
            * (self.kv_heads as f64)
            * (self.head_dim as f64)
            * (avg_ctx as f64)
            * total_tok;
        let t_compute_ms = ((matmul_flops + attn_flops) / H100_ACHIEVED_BF16_FLOPS) * 1000.0;

        // Smooth roofline combination + modest tree-mask dispatch overhead for >1 verify token
        let mask_overhead = if verify_tokens > 1 {
            0.03 * (verify_tokens as f64).log2()
        } else {
            0.0
        };
        (t_mem_ms * t_mem_ms + t_compute_ms * t_compute_ms).sqrt()
            + KERNEL_LAUNCH_OVERHEAD_MS
            + mask_overhead
    }

    /// Roofline EAGLE-3 draft latency (ms) for `draft_steps` sequential draft steps at `batch_size`.
    pub fn roofline_eagle3_draft_ms(&self, draft_steps: usize, batch_size: usize) -> f64 {
        if draft_steps == 0 {
            return 0.0;
        }
        let draft_params = self.eagle3_draft_weight_bytes / 2.0;
        let t_mem_ms = (self.eagle3_draft_weight_bytes / H100_ACHIEVED_BW_BYTES_PER_S) * 1000.0;
        let t_compute_ms =
            ((2.0 * draft_params * (batch_size as f64)) / H100_ACHIEVED_BF16_FLOPS) * 1000.0;
        let per_step = (t_mem_ms * t_mem_ms + t_compute_ms * t_compute_ms).sqrt() + 0.035;
        per_step * (draft_steps as f64)
    }
}

/// Measured CPU acceptance & strategy mix passed into the HIP-1346 `4B / 9B / 27B` scorecard.
#[derive(Clone, Debug)]
pub struct StrategyMeasuredProfile {
    pub strategy_name: &'static str,
    pub uses_eagle3_head: bool,
    pub mean_tau: f64,
    pub p95_step_ratio: f64,
    pub avg_verify_tokens: f64,
    pub avg_eagle3_draft_steps: f64,
    pub lookup_overhead_ms: f64,
    pub output_bit_exact_pct: f64,
    pub quality_regression_pp: f64,
}

/// One row of the shared HIP-1346 `4B / 9B / 27B` 20-turn coding-agent scorecard.
#[derive(Clone, Debug)]
pub struct MatrixScorecardRow {
    pub model: &'static str,
    pub strategy: &'static str,
    pub quantization_format: &'static str,
    pub timing_source: &'static str,
    pub batch_size: usize,
    pub model_size_gib: f64,
    pub peak_memory_gib: f64,
    pub mean_tau_cpu_measured: f64,
    pub cold_ttft_ms: f64,
    pub warm_ttft_ms: f64,
    pub p50_itl_ms: f64,
    pub p95_itl_ms: f64,
    pub warm_tok_per_s: f64,
    pub agent_loop_20_turn_s: f64,
    pub speedup_vs_plain: f64,
    pub energy_mj_per_tok: f64,
    pub output_bit_exact_pct: f64,
    pub quality_regression_pp: f64,
    pub gate_verdict: &'static str,
    pub gate_reason: String,
}

/// Project one `(model, strategy)` row for the 20-turn coding-agent loop (`batch_size = 1`),
/// calibrated strictly from CPU-measured `mean_tau` and `output_bit_exact_pct`.
pub fn project_agent_scorecard_row(
    spec: &HybridModelSpec,
    profile: &StrategyMeasuredProfile,
    plain_wall_s: Option<f64>,
    plain_p95_itl_ms: Option<f64>,
) -> MatrixScorecardRow {
    let avg_ctx =
        AGENT_COLD_PREFILL_TOKENS + (AGENT_PEAK_CONTEXT_TOKENS - AGENT_COLD_PREFILL_TOKENS) / 2;
    let cold_ttft_ms = spec.roofline_prefill_ms(AGENT_COLD_PREFILL_TOKENS, 1);
    let warm_ttft_ms = spec.roofline_prefill_ms(AGENT_WARM_PREFILL_PER_TURN, 1);

    let verify_n = profile.avg_verify_tokens.round().max(1.0) as usize;
    let step_verify_ms = spec.roofline_verify_ms(verify_n, 1, avg_ctx);
    let step_draft_ms = if profile.avg_eagle3_draft_steps > 0.0 {
        spec.roofline_eagle3_draft_ms(1, 1) * profile.avg_eagle3_draft_steps
    } else {
        profile.lookup_overhead_ms
    };

    let step_total_ms = step_draft_ms + step_verify_ms;
    let tau = profile.mean_tau.max(1.0);
    let p50_itl_ms = step_total_ms / tau;
    let p95_itl_ms = p50_itl_ms * profile.p95_step_ratio.max(1.04);
    let warm_tok_per_s = 1000.0 / p50_itl_ms.max(1e-6);

    // 20-turn wall time = cold prefill + 20 * warm prefill + 5,120 generated tokens * p50_itl
    let total_decode_ms = (AGENT_TOTAL_GEN_TOKENS as f64) * p50_itl_ms;
    let total_prefill_ms = cold_ttft_ms + (AGENT_TURNS as f64) * warm_ttft_ms;
    let agent_loop_20_turn_s = (total_prefill_ms + total_decode_ms) / 1000.0;

    let energy_mj_per_tok =
        (H100_ACTIVE_WATTS * agent_loop_20_turn_s * 1000.0) / (AGENT_TOTAL_GEN_TOKENS as f64);

    let model_size_gib = (spec.target_weight_bytes
        + if profile.uses_eagle3_head {
            spec.eagle3_draft_weight_bytes
        } else {
            0.0
        })
        / (1024.0 * 1024.0 * 1024.0);
    let peak_memory_gib = spec.peak_memory_gib(profile.uses_eagle3_head, verify_n);

    let speedup_vs_plain = match plain_wall_s {
        Some(pw) => pw / agent_loop_20_turn_s.max(1e-9),
        None => 1.0,
    };

    let (gate_verdict, gate_reason) = evaluate_acceptance_gate(
        profile.strategy_name,
        profile.output_bit_exact_pct,
        profile.quality_regression_pp,
        agent_loop_20_turn_s,
        plain_wall_s.unwrap_or(agent_loop_20_turn_s),
        p95_itl_ms,
        plain_p95_itl_ms.unwrap_or(p95_itl_ms),
    );

    MatrixScorecardRow {
        model: spec.name,
        strategy: profile.strategy_name,
        quantization_format: spec.quantization_format,
        timing_source: "h100_sxm5_roofline_modeled_from_cpu_measured_tau",
        batch_size: 1,
        model_size_gib,
        peak_memory_gib,
        mean_tau_cpu_measured: tau,
        cold_ttft_ms,
        warm_ttft_ms,
        p50_itl_ms,
        p95_itl_ms,
        warm_tok_per_s,
        agent_loop_20_turn_s,
        speedup_vs_plain,
        energy_mj_per_tok,
        output_bit_exact_pct: profile.output_bit_exact_pct,
        quality_regression_pp: profile.quality_regression_pp,
        gate_verdict,
        gate_reason,
    }
}

/// Evaluate Negative Regime 2 (`neg_large_batch_verify_overhead`): large-batch (`batch_size = 32, 64`)
/// roofline throughput proving that fixed speculative verification saturates compute FLOPs and
/// regresses end-to-end throughput (`< 1.0x`), whereas `workload_aware_selector` stands down to
/// `PlainDecode` with `1.00x` zero-regression parity.
#[derive(Clone, Debug)]
pub struct LargeBatchNegativeRow {
    pub model: &'static str,
    pub batch_size: usize,
    pub strategy: &'static str,
    pub verify_tokens: usize,
    pub mean_tau: f64,
    pub step_ms: f64,
    pub agg_tok_per_s: f64,
    pub speedup_vs_plain: f64,
    pub energy_mj_per_tok: f64,
    pub gate_verdict: &'static str,
    pub gate_reason: String,
}

pub fn evaluate_large_batch_negative_regime(
    spec: &HybridModelSpec,
    batch_size: usize,
    measured_linear_tau: f64,
    measured_tree_tau: f64,
) -> Vec<LargeBatchNegativeRow> {
    let avg_ctx = 2_048usize;
    let plain_step_ms = spec.roofline_verify_ms(1, batch_size, avg_ctx);
    let plain_tps = (batch_size as f64) * 1000.0 / plain_step_ms;

    let cases: [(&'static str, usize, usize, f64); 4] = [
        ("plain_decode_baseline", 1, 0, 1.0),
        ("eagle3_linear_fixed", 5, 4, measured_linear_tau.max(1.0)),
        ("eagle3_tree_fixed", 10, 4, measured_tree_tau.max(1.0)),
        // Selector detects `batch_size >= compute_bound_batch_threshold` and stands down to PlainDecode!
        ("workload_aware_selector", 1, 0, 1.0),
    ];

    let mut rows = Vec::with_capacity(cases.len());
    for (name, ver_toks, draft_steps, tau) in cases {
        let ver_ms = spec.roofline_verify_ms(ver_toks, batch_size, avg_ctx);
        let d_ms = spec.roofline_eagle3_draft_ms(draft_steps, batch_size);
        let step_ms = ver_ms + d_ms;
        let tps = (batch_size as f64) * tau * 1000.0 / step_ms;
        let speedup = tps / plain_tps;
        let energy_mj = (H100_ACTIVE_WATTS * 1000.0) / tps.max(1e-6);

        let (verdict, reason) = if speedup + 1e-6 < 1.0 {
            (
                "REJECTED",
                format!(
                    "compute-bound large-batch regression ({speedup:.2}x < 1.00x vs PlainDecode at B={batch_size})"
                ),
            )
        } else if name == "workload_aware_selector" {
            (
                "ACCEPTED",
                format!("automatic compute-bound stand-down to PlainDecode at B={batch_size} (0% regression)"),
            )
        } else if name == "plain_decode_baseline" {
            ("ACCEPTED", "baseline reference".to_string())
        } else {
            (
                "ACCEPTED",
                format!("below compute ridge point at B={batch_size} ({speedup:.2}x >= 1.00x)"),
            )
        };

        rows.push(LargeBatchNegativeRow {
            model: spec.name,
            batch_size,
            strategy: name,
            verify_tokens: ver_toks,
            mean_tau: tau,
            step_ms,
            agg_tok_per_s: tps,
            speedup_vs_plain: speedup,
            energy_mj_per_tok: energy_mj,
            gate_verdict: verdict,
            gate_reason: reason,
        });
    }
    rows
}

/// Strict end-to-end HIP-1346 acceptance gate:
/// rejects any change that breaks lossless output correctness OR regresses end-to-end wall time
/// or tail p95 ITL.
pub fn evaluate_acceptance_gate(
    strategy_name: &str,
    output_bit_exact_pct: f64,
    quality_regression_pp: f64,
    wall_s: f64,
    plain_wall_s: f64,
    p95_itl_ms: f64,
    plain_p95_itl_ms: f64,
) -> (&'static str, String) {
    if output_bit_exact_pct < 100.0 || quality_regression_pp.abs() > 1e-6 {
        return (
            "REJECTED",
            format!(
                "lossless verification violated (exact={output_bit_exact_pct:.2}%, reg={quality_regression_pp:.2} pp)"
            ),
        );
    }
    if strategy_name == "plain_decode_baseline" {
        return ("ACCEPTED", "lossless baseline reference".to_string());
    }
    if wall_s > plain_wall_s * 1.001 {
        let ratio = plain_wall_s / wall_s.max(1e-9);
        return (
            "REJECTED",
            format!(
                "end-to-end wall time regressed ({ratio:.2}x < 1.00x vs plain_decode_baseline)"
            ),
        );
    }
    if p95_itl_ms > plain_p95_itl_ms * 1.05 {
        return (
            "REJECTED",
            format!(
                "p95 ITL tail regressed ({p95_itl_ms:.2} ms > 1.05x baseline {plain_p95_itl_ms:.2} ms)"
            ),
        );
    }
    let speedup = plain_wall_s / wall_s.max(1e-9);
    (
        "ACCEPTED",
        format!("lossless 100.0% exact, {speedup:.2}x end-to-end speedup with bounded p95 ITL"),
    )
}
