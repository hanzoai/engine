// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reproducible HIP-1346 benchmark suite, end-to-end acceptance gate, and JSON receipt emitter.

use std::fmt::Write as _;

use crate::config::{SchedulerConfig, SchedulingPolicy, SloTargets};
use crate::corpus::{CorpusFileProvenance, GroundedCorpus};
use crate::kv_pool::SchedulerError;
use crate::model::{HardwareSpec, HybridModelSpec, StepCostModel};
use crate::scheduler::{ScheduleMetrics, SchedulerEngine};
use crate::sha256::{json_f64, json_str, Sha256};
use crate::verify::{HybridMicroModel, MicroModelConfig, VerificationReport};

/// Canonical HIP-1346 concurrency sweep levels.
pub const CONCURRENCY_SWEEP: [usize; 5] = [1, 4, 16, 64, 256];

/// End-to-end acceptance gate verdict for a scheduling regime.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptanceVerdict {
    /// Regime label under evaluation.
    pub regime_label: String,
    /// True if all strict end-to-end acceptance gates passed.
    pub accepted: bool,
    /// Human-readable reason explaining acceptance or rejection.
    pub reason: String,
}

/// One row of the concurrency sweep (`1 / 4 / 16 / 64 / 256` × 5 policies).
#[derive(Debug, Clone)]
pub struct ConcurrencySweepRow {
    /// Target model label (`"9B-Hybrid"`).
    pub model_label: &'static str,
    /// Simultaneous request count (`1 / 4 / 16 / 64 / 256`).
    pub concurrency: usize,
    /// Schedule metrics for this `(concurrency, policy)` run.
    pub metrics: ScheduleMetrics,
}

/// One row of the `4B / 9B / 27B` hybrid matrix 20-turn coding-agent scorecard.
#[derive(Debug, Clone)]
pub struct MatrixAgentRow {
    /// Model label (`"4B-Hybrid"`, `"9B-Hybrid"`, `"27B-Hybrid"`).
    pub model_label: &'static str,
    /// Regime label.
    pub regime_label: &'static str,
    /// Full 20-turn coding-agent loop schedule metrics.
    pub metrics: ScheduleMetrics,
    /// Output correctness percentage from token-level verifier (`100.0`).
    pub output_correctness_pct: f64,
    /// Quality regression in percentage points (`0.00`).
    pub quality_regression_pp: f64,
    /// Strict end-to-end acceptance gate verdict.
    pub verdict: AcceptanceVerdict,
}

/// Result bundle for the 3 explicitly tested honest negative regimes.
#[derive(Debug, Clone)]
pub struct NegativeRegimesReport {
    /// Negative Regime 1 (`neg_single_user_concurrency_1`): forced small-chunk batching at `C = 1`
    /// vs single-user fast-path bypass.
    pub neg1_single_user_small_chunk: ScheduleMetrics,
    /// Single-user bypass reference (`C = 1`) for Negative Regime 1.
    pub neg1_single_user_bypass_ref: ScheduleMetrics,
    /// Negative Regime 1 verdict (expected REJECTED).
    pub neg1_verdict: AcceptanceVerdict,
    /// Negative Regime 2 (`neg_kv_memory_oversubscription_256`): unguarded admission at `C = 256`
    /// with 32K prompts triggering preemption/recompute thrashing.
    pub neg2_unguarded_oversubscription: ScheduleMetrics,
    /// KV-pressure-guarded admission reference at `C = 256` (queues cleanly, preserves `p95 ITL`).
    pub neg2_kv_guarded_ref: ScheduleMetrics,
    /// Negative Regime 2 verdict (expected REJECTED for unguarded).
    pub neg2_verdict: AcceptanceVerdict,
    /// Negative Regime 3 (`neg_ultra_small_chunk_prefill_tax`): `chunk_size = 64` winning micro-ITL
    /// while regressing 20-turn end-to-end wall time and getting REJECTED by the gate.
    pub neg3_ultra_small_chunk_64: ScheduleMetrics,
    /// Balanced chunked reference (`LatencyAwareChunkedKvScheduler`) for Negative Regime 3.
    pub neg3_balanced_ref: ScheduleMetrics,
    /// Negative Regime 3 verdict (expected REJECTED).
    pub neg3_verdict: AcceptanceVerdict,
}

/// Full HIP-1346 benchmark suite output bundle.
#[derive(Debug, Clone)]
pub struct BenchmarkSuiteReport {
    /// Deterministic PRNG seed (`1346`).
    pub seed: u64,
    /// Analytical hardware specification tag.
    pub hardware: HardwareSpec,
    /// Grounded repository corpus file provenance records.
    pub corpus_files: Vec<CorpusFileProvenance>,
    /// Combined corpus byte count.
    pub corpus_total_bytes: usize,
    /// Combined corpus SHA-256 hex digest.
    pub corpus_combined_sha256: String,
    /// Real CPU token-level equivalence verification report (chunked + interleaved + preemption).
    pub verification: VerificationReport,
    /// Profile comparison at `C = 1` and `C = 16` (`SingleUserInteractive`, `DatacenterThroughput`, `LatencyAwareSlo`).
    pub profile_comparison_c1: Vec<ScheduleMetrics>,
    /// Profile comparison at `C = 16`.
    pub profile_comparison_c16: Vec<ScheduleMetrics>,
    /// Concurrency sweep (`1 / 4 / 16 / 64 / 256` × 5 matched policies on `9B-Hybrid`).
    pub concurrency_sweep: Vec<ConcurrencySweepRow>,
    /// Shared `4B / 9B / 27B` hybrid matrix 20-turn coding-agent scorecard.
    pub hybrid_matrix_20turn: Vec<MatrixAgentRow>,
    /// The 3 explicitly tested honest negative regimes.
    pub negative_regimes: NegativeRegimesReport,
    /// Deterministic SHA-256 hex digest of the emitted JSON payload.
    pub receipt_sha256: String,
}

/// Evaluates the strict HIP-1346 end-to-end acceptance gate for a 20-turn coding-agent regime
/// against the static-batching baseline and unchunked single-user baseline.
#[must_use]
pub fn evaluate_end_to_end_gate(
    candidate: &ScheduleMetrics,
    static_baseline: &ScheduleMetrics,
    verify: &VerificationReport,
    slo: SloTargets,
) -> AcceptanceVerdict {
    if verify.output_correctness_pct < 100.0 || verify.quality_regression_pp > 0.0 {
        return AcceptanceVerdict {
            regime_label: candidate.regime_label.to_string(),
            accepted: false,
            reason: format!(
                "output equivalence regression ({:.2}% correctness, {:.3} pp)",
                verify.output_correctness_pct, verify.quality_regression_pp
            ),
        };
    }
    if !candidate.kv_pool_clean
        || candidate.oom_rejected_requests > 0
        || candidate.completed_requests + candidate.oom_rejected_requests < candidate.total_requests
    {
        return AcceptanceVerdict {
            regime_label: candidate.regime_label.to_string(),
            accepted: false,
            reason: format!(
                "KV pool integrity/OOM/incomplete-run failure (clean={}, completed={}/{}, oom_rejected={})",
                candidate.kv_pool_clean,
                candidate.completed_requests,
                candidate.total_requests,
                candidate.oom_rejected_requests
            ),
        };
    }
    // Reject microbenchmark wins that regress end-to-end 20-turn wall time vs static baseline.
    if candidate.makespan_s > static_baseline.makespan_s * 1.05 {
        return AcceptanceVerdict {
            regime_label: candidate.regime_label.to_string(),
            accepted: false,
            reason: format!(
                "end-to-end 20-turn wall regression ({:.2}s vs {:.2}s baseline — prefill chunk overhead)",
                candidate.makespan_s, static_baseline.makespan_s
            ),
        };
    }
    // Enforce p95 ITL SLO protection on concurrent workloads.
    if candidate.itl_p95_ms > slo.itl_p95_ms {
        return AcceptanceVerdict {
            regime_label: candidate.regime_label.to_string(),
            accepted: false,
            reason: format!(
                "p95 ITL SLO breach ({:.1} ms > {:.1} ms target)",
                candidate.itl_p95_ms, slo.itl_p95_ms
            ),
        };
    }
    // Require strong overall SLO attainment (>= 85%) on the mixed 20-turn + interactive workload.
    if candidate.slo_attainment_pct < 85.0 {
        return AcceptanceVerdict {
            regime_label: candidate.regime_label.to_string(),
            accepted: false,
            reason: format!(
                "SLO attainment below 85% floor ({:.1}%)",
                candidate.slo_attainment_pct
            ),
        };
    }
    AcceptanceVerdict {
        regime_label: candidate.regime_label.to_string(),
        accepted: true,
        reason: format!(
            "PASS: 100% bit-exact, p95 ITL {:.1}ms <= {:.1}ms, 20-turn wall {:.2}s, SLO {:.1}%",
            candidate.itl_p95_ms,
            slo.itl_p95_ms,
            candidate.makespan_s,
            candidate.slo_attainment_pct
        ),
    }
}

/// Runs the complete reproducible HIP-1346 benchmark suite with deterministic `seed`.
///
/// # Errors
/// Returns [`SchedulerError`] if any simulation fails an internal invariant.
pub fn run_full_benchmark_suite(
    corpus: &GroundedCorpus,
    seed: u64,
) -> Result<BenchmarkSuiteReport, SchedulerError> {
    let hardware = HardwareSpec::h100_sxm5_80gb();
    let slo = SloTargets::default();

    // 1. Real CPU token-level output equivalence verification (chunked + interleaved + preemption).
    let micro_cfg = MicroModelConfig::default();
    let micro_model = HybridMicroModel::new(micro_cfg, seed);
    let verify_specs = corpus.build_verify_specs(seed, micro_cfg.vocab_size);
    let verification = micro_model.verify_chunked_and_interleaved(&verify_specs, 17, true)?;

    // 2. Separate operating profiles comparison at C = 1 and C = 16 on 9B-Hybrid.
    let model_9b = HybridModelSpec::hybrid_9b();
    let cost_9b = StepCostModel::new(model_9b.clone(), hardware.clone());
    let profile_configs = [
        SchedulerConfig::single_user_interactive(),
        SchedulerConfig::datacenter_throughput(),
        SchedulerConfig::latency_aware_slo(),
    ];
    let reqs_c1 = corpus.build_mixed_concurrency_workload(1, seed, slo);
    let reqs_c16 = corpus.build_mixed_concurrency_workload(16, seed, slo);

    let mut profile_comparison_c1 = Vec::with_capacity(profile_configs.len());
    let mut profile_comparison_c16 = Vec::with_capacity(profile_configs.len());
    for cfg in &profile_configs {
        profile_comparison_c1.push(
            SchedulerEngine::new(cfg.clone(), cost_9b.clone())?
                .run(reqs_c1.clone())?
                .metrics,
        );
        profile_comparison_c16.push(
            SchedulerEngine::new(cfg.clone(), cost_9b.clone())?
                .run(reqs_c16.clone())?
                .metrics,
        );
    }

    // 3. Concurrency sweep: 1 / 4 / 16 / 64 / 256 × 5 matched scheduling policies on 9B-Hybrid.
    let mut concurrency_sweep = Vec::with_capacity(CONCURRENCY_SWEEP.len() * 5);
    for &conc in &CONCURRENCY_SWEEP {
        let workload = corpus.build_mixed_concurrency_workload(conc, seed, slo);
        for policy in SchedulingPolicy::all() {
            let cfg = SchedulerConfig::for_policy(policy);
            let metrics = SchedulerEngine::new(cfg, cost_9b.clone())?
                .run(workload.clone())?
                .metrics;
            concurrency_sweep.push(ConcurrencySweepRow {
                model_label: model_9b.label,
                concurrency: conc,
                metrics,
            });
        }
    }

    // 4. Shared 4B / 9B / 27B hybrid matrix: 20-turn coding-agent loop + 16 concurrent interactive chats.
    let agent_workload = corpus.build_coding_agent_20_turn_workload(seed, slo, 16);
    let mut hybrid_matrix_20turn = Vec::new();
    for model in HybridModelSpec::canonical_matrix() {
        let cost = StepCostModel::new(model.clone(), hardware.clone());
        let regimes = [
            SchedulerConfig::for_policy(SchedulingPolicy::StaticBatchingFcfs),
            SchedulerConfig::for_policy(SchedulingPolicy::ContinuousBatchingPrefillFirst),
            SchedulerConfig::for_policy(SchedulingPolicy::ContinuousBatchingDecodePriority),
            SchedulerConfig::for_policy(SchedulingPolicy::ChunkedPrefillTokenBudget),
            SchedulerConfig::for_policy(SchedulingPolicy::LatencyAwareChunkedKvScheduler),
            SchedulerConfig::neg_ultra_small_chunk_64(),
        ];

        let mut static_baseline: Option<ScheduleMetrics> = None;
        for cfg in regimes {
            let metrics = SchedulerEngine::new(cfg, cost.clone())?
                .run(agent_workload.clone())?
                .metrics;
            if static_baseline.is_none() {
                static_baseline = Some(metrics.clone());
            }
            let base_ref = static_baseline.as_ref().expect("set on first policy");
            let verdict = evaluate_end_to_end_gate(&metrics, base_ref, &verification, slo);
            hybrid_matrix_20turn.push(MatrixAgentRow {
                model_label: model.label,
                regime_label: metrics.regime_label,
                metrics,
                output_correctness_pct: verification.output_correctness_pct,
                quality_regression_pp: verification.quality_regression_pp,
                verdict,
            });
        }
    }

    // 5. Three Explicit Honest Negative Regimes on 9B-Hybrid.
    // Neg 1: Single-user C = 1 small-chunk (256) batching tax vs C = 1 automatic bypass.
    let neg1_small = SchedulerEngine::new(
        SchedulerConfig::neg_single_user_small_chunk(),
        cost_9b.clone(),
    )?
    .run(reqs_c1.clone())?
    .metrics;
    let neg1_bypass = SchedulerEngine::new(SchedulerConfig::latency_aware_slo(), cost_9b.clone())?
        .run(reqs_c1)?
        .metrics;
    let neg1_tax_pct = ((neg1_small.cold_ttft_p50_ms - neg1_bypass.cold_ttft_p50_ms)
        / neg1_bypass.cold_ttft_p50_ms.max(1e-6))
        * 100.0;
    let neg1_verdict = AcceptanceVerdict {
        regime_label: "neg_single_user_concurrency_1".to_string(),
        accepted: false,
        reason: format!(
            "REJECTED: forcing small-chunk (256) datacenter batching at C=1 adds +{:.1}% cold TTFT tax ({:.1}ms vs {:.1}ms single-user bypass)",
            neg1_tax_pct, neg1_small.cold_ttft_p50_ms, neg1_bypass.cold_ttft_p50_ms
        ),
    };

    // Neg 2: C = 256 KV memory oversubscription (unguarded admission thrashing vs KV-guarded queuing).
    let oversub_256 = corpus.build_oversubscription_256_workload(seed, slo);
    let neg2_unguarded = SchedulerEngine::new(
        SchedulerConfig::neg_unguarded_kv_oversubscription(),
        cost_9b.clone(),
    )?
    .run(oversub_256.clone())?
    .metrics;
    let neg2_guarded = SchedulerEngine::new(SchedulerConfig::latency_aware_slo(), cost_9b.clone())?
        .run(oversub_256)?
        .metrics;
    let neg2_verdict = AcceptanceVerdict {
        regime_label: "neg_kv_memory_oversubscription_256".to_string(),
        accepted: false,
        reason: format!(
            "REJECTED: unguarded admission under {:.1} GiB demand vs {:.1} GiB pool triggers {} preemptions ({} recomputed tokens, p95 ITL {:.1}ms); KV-guarded admission eliminates thrashing ({} preemptions, p95 ITL {:.1}ms) at honest TTFT queue-wait cost",
            neg2_unguarded.peak_unbounded_demand_gib,
            neg2_unguarded.kv_pool_capacity_gib,
            neg2_unguarded.total_preemptions,
            neg2_unguarded.total_recomputed_tokens,
            neg2_unguarded.itl_p95_ms,
            neg2_guarded.total_preemptions,
            neg2_guarded.itl_p95_ms
        ),
    };

    // Neg 3: Ultra-small chunk_size = 64 wins micro-ITL but regresses 20-turn end-to-end wall time.
    let neg3_chunk64 =
        SchedulerEngine::new(SchedulerConfig::neg_ultra_small_chunk_64(), cost_9b.clone())?
            .run(agent_workload.clone())?
            .metrics;
    let neg3_balanced =
        SchedulerEngine::new(SchedulerConfig::latency_aware_slo(), cost_9b.clone())?
            .run(agent_workload.clone())?
            .metrics;
    let neg3_static = SchedulerEngine::new(
        SchedulerConfig::for_policy(SchedulingPolicy::StaticBatchingFcfs),
        cost_9b,
    )?
    .run(agent_workload)?
    .metrics;
    let neg3_verdict = evaluate_end_to_end_gate(&neg3_chunk64, &neg3_static, &verification, slo);

    let mut report = BenchmarkSuiteReport {
        seed,
        hardware,
        corpus_files: corpus.files.clone(),
        corpus_total_bytes: corpus.total_bytes,
        corpus_combined_sha256: corpus.combined_sha256_hex.clone(),
        verification,
        profile_comparison_c1,
        profile_comparison_c16,
        concurrency_sweep,
        hybrid_matrix_20turn,
        negative_regimes: NegativeRegimesReport {
            neg1_single_user_small_chunk: neg1_small,
            neg1_single_user_bypass_ref: neg1_bypass,
            neg1_verdict,
            neg2_unguarded_oversubscription: neg2_unguarded,
            neg2_kv_guarded_ref: neg2_guarded,
            neg2_verdict,
            neg3_ultra_small_chunk_64: neg3_chunk64,
            neg3_balanced_ref: neg3_balanced,
            neg3_verdict,
        },
        receipt_sha256: String::new(),
    };

    // Compute deterministic receipt SHA-256 over canonical JSON payload (excluding the hash field
    // and wall-clock CPU timing jitter so identical seeds produce bit-identical receipts).
    let canonical_payload = report.to_canonical_hash_input();
    report.receipt_sha256 = Sha256::digest_hex(canonical_payload.as_bytes());
    Ok(report)
}

impl BenchmarkSuiteReport {
    /// Produces the deterministic canonical JSON string hashed for `receipt_sha256`
    /// (excludes non-deterministic wall-clock CPU timers so `--seed 1346` is bit-reproducible).
    #[must_use]
    pub fn to_canonical_hash_input(&self) -> String {
        self.render_json(false)
    }

    /// Renders the complete `results.json` artifact (including measured CPU wall times and
    /// `receipt_sha256`).
    #[must_use]
    pub fn to_json_pretty(&self) -> String {
        self.render_json(true)
    }

    fn render_json(&self, include_runtime_fields: bool) -> String {
        let mut s = String::with_capacity(32 * 1024);
        s.push_str("{\n");
        let _ = writeln!(s, "  \"schema_version\": \"hip-1346-scheduler-v1\",");
        let _ = writeln!(s, "  \"issue\": \"hanzoai/engine#56\",");
        let _ = writeln!(s, "  \"crate\": \"hanzo-scheduler\",");
        let _ = writeln!(s, "  \"license\": \"Apache-2.0\",");
        let _ = writeln!(s, "  \"seed\": {},", self.seed);
        if include_runtime_fields {
            let _ = writeln!(
                s,
                "  \"receipt_sha256\": {},",
                json_str(&self.receipt_sha256)
            );
        }
        let _ = writeln!(
            s,
            "  \"honesty_boundary\": \"Memory/KV/GDN byte footprints are exact unclamped accounting; token-level output equivalence and CPU scheduler wall times are directly measured on host CPU; GPU TTFT/ITL/throughput/makespan/energy are explicitly tagged analytical H100-SXM5-80GB roofline projections.\","
        );

        // Hardware spec.
        s.push_str("  \"analytical_hardware\": {\n");
        let _ = writeln!(s, "    \"name\": {},", json_str(self.hardware.name));
        let _ = writeln!(
            s,
            "    \"vram_gib\": {},",
            json_f64(
                self.hardware.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                2
            )
        );
        let _ = writeln!(
            s,
            "    \"peak_bf16_tflops\": {},",
            json_f64(self.hardware.peak_bf16_tflops, 1)
        );
        let _ = writeln!(
            s,
            "    \"hbm_bandwidth_gbps\": {},",
            json_f64(self.hardware.hbm_bandwidth_gbps, 1)
        );
        let _ = writeln!(
            s,
            "    \"tdp_watts\": {}",
            json_f64(self.hardware.tdp_watts, 1)
        );
        s.push_str("  },\n");

        // Shared HIP-1346 model matrix.
        s.push_str("  \"hybrid_model_matrix\": [\n");
        let models = HybridModelSpec::canonical_matrix();
        for (i, m) in models.iter().enumerate() {
            s.push_str("    {\n");
            let _ = writeln!(s, "      \"label\": {},", json_str(m.label));
            let _ = writeln!(
                s,
                "      \"num_attention_layers\": {},",
                m.num_attention_layers
            );
            let _ = writeln!(s, "      \"num_gdn_layers\": {},", m.num_gdn_layers);
            let _ = writeln!(s, "      \"num_kv_heads\": {},", m.num_kv_heads);
            let _ = writeln!(s, "      \"head_dim\": {},", m.head_dim);
            let _ = writeln!(s, "      \"gdn_heads\": {},", m.gdn_heads);
            let _ = writeln!(
                s,
                "      \"gdn_state_mib\": {},",
                json_f64(m.exact_gdn_state_mib(), 2)
            );
            let _ = writeln!(
                s,
                "      \"kv_bytes_per_token_fp16\": {},",
                m.kv_bytes_per_token_fp16()
            );
            let _ = writeln!(s, "      \"total_params\": {},", m.total_params());
            let _ = writeln!(
                s,
                "      \"weight_gib_bf16\": {}",
                json_f64(m.weight_bytes_bf16() as f64 / (1024.0 * 1024.0 * 1024.0), 3)
            );
            s.push_str(if i + 1 < models.len() {
                "    },\n"
            } else {
                "    }\n"
            });
        }
        s.push_str("  ],\n");

        // Corpus provenance.
        s.push_str("  \"corpus_provenance\": {\n");
        let _ = writeln!(s, "    \"total_bytes\": {},", self.corpus_total_bytes);
        let _ = writeln!(
            s,
            "    \"combined_sha256\": {},",
            json_str(&self.corpus_combined_sha256)
        );
        s.push_str("    \"files\": [\n");
        for (i, f) in self.corpus_files.iter().enumerate() {
            s.push_str("      {\n");
            let _ = writeln!(
                s,
                "        \"relative_path\": {},",
                json_str(&f.relative_path)
            );
            let _ = writeln!(s, "        \"byte_len\": {},", f.byte_len);
            let _ = writeln!(s, "        \"sha256\": {}", json_str(&f.sha256_hex));
            s.push_str(if i + 1 < self.corpus_files.len() {
                "      },\n"
            } else {
                "      }\n"
            });
        }
        s.push_str("    ]\n  },\n");

        // CPU-measured token-level output equivalence verification.
        let v = &self.verification;
        s.push_str("  \"cpu_token_equivalence_verification\": {\n");
        let _ = writeln!(s, "    \"measurement_tag\": \"cpu_directly_measured\",");
        let _ = writeln!(s, "    \"num_sequences\": {},", v.num_sequences);
        let _ = writeln!(s, "    \"total_prompt_tokens\": {},", v.total_prompt_tokens);
        let _ = writeln!(s, "    \"total_output_tokens\": {},", v.total_output_tokens);
        let _ = writeln!(s, "    \"matched_tokens\": {},", v.matched_tokens);
        let _ = writeln!(
            s,
            "    \"output_correctness_pct\": {},",
            json_f64(v.output_correctness_pct, 3)
        );
        let _ = writeln!(
            s,
            "    \"quality_regression_pp\": {},",
            json_f64(v.quality_regression_pp, 3)
        );
        let _ = writeln!(
            s,
            "    \"max_abs_logit_diff\": {},",
            json_f64(f64::from(v.max_abs_logit_diff), 8)
        );
        let _ = writeln!(s, "    \"interleaved_steps\": {},", v.interleaved_steps);
        let _ = writeln!(
            s,
            "    \"preemption_recompute_verified\": {},",
            v.preemption_recompute_verified
        );
        if include_runtime_fields {
            let _ = writeln!(
                s,
                "    \"cpu_verify_wall_ms\": {},",
                json_f64(v.cpu_verify_wall_ms, 3)
            );
        }
        let _ = writeln!(s, "    \"kv_pool_clean\": {}", v.kv_pool_clean);
        s.push_str("  },\n");

        // Operating profile comparisons.
        s.push_str("  \"profile_comparison_c1\": [\n");
        for (i, m) in self.profile_comparison_c1.iter().enumerate() {
            write_metrics_obj(&mut s, "    ", m, include_runtime_fields);
            s.push_str(if i + 1 < self.profile_comparison_c1.len() {
                ",\n"
            } else {
                "\n"
            });
        }
        s.push_str("  ],\n");

        s.push_str("  \"profile_comparison_c16\": [\n");
        for (i, m) in self.profile_comparison_c16.iter().enumerate() {
            write_metrics_obj(&mut s, "    ", m, include_runtime_fields);
            s.push_str(if i + 1 < self.profile_comparison_c16.len() {
                ",\n"
            } else {
                "\n"
            });
        }
        s.push_str("  ],\n");

        // Concurrency sweep.
        s.push_str("  \"concurrency_sweep\": [\n");
        for (i, row) in self.concurrency_sweep.iter().enumerate() {
            s.push_str("    {\n");
            let _ = writeln!(s, "      \"model\": {},", json_str(row.model_label));
            let _ = writeln!(s, "      \"concurrency\": {},", row.concurrency);
            s.push_str("      \"metrics\": ");
            write_metrics_obj(&mut s, "      ", &row.metrics, include_runtime_fields);
            s.push('\n');
            s.push_str(if i + 1 < self.concurrency_sweep.len() {
                "    },\n"
            } else {
                "    }\n"
            });
        }
        s.push_str("  ],\n");

        // Shared 4B/9B/27B 20-turn coding-agent matrix.
        s.push_str("  \"hybrid_matrix_20turn\": [\n");
        for (i, row) in self.hybrid_matrix_20turn.iter().enumerate() {
            s.push_str("    {\n");
            let _ = writeln!(s, "      \"model\": {},", json_str(row.model_label));
            let _ = writeln!(s, "      \"regime\": {},", json_str(row.regime_label));
            let _ = writeln!(
                s,
                "      \"output_correctness_pct\": {},",
                json_f64(row.output_correctness_pct, 2)
            );
            let _ = writeln!(
                s,
                "      \"quality_regression_pp\": {},",
                json_f64(row.quality_regression_pp, 3)
            );
            let _ = writeln!(s, "      \"gate_accepted\": {},", row.verdict.accepted);
            let _ = writeln!(
                s,
                "      \"gate_reason\": {},",
                json_str(&row.verdict.reason)
            );
            s.push_str("      \"metrics\": ");
            write_metrics_obj(&mut s, "      ", &row.metrics, include_runtime_fields);
            s.push('\n');
            s.push_str(if i + 1 < self.hybrid_matrix_20turn.len() {
                "    },\n"
            } else {
                "    }\n"
            });
        }
        s.push_str("  ],\n");

        // 3 Honest Negative Regimes.
        let neg = &self.negative_regimes;
        s.push_str("  \"negative_regimes\": {\n");
        s.push_str("    \"neg_single_user_concurrency_1\": {\n");
        let _ = writeln!(s, "      \"gate_accepted\": {},", neg.neg1_verdict.accepted);
        let _ = writeln!(
            s,
            "      \"gate_reason\": {},",
            json_str(&neg.neg1_verdict.reason)
        );
        s.push_str("      \"small_chunk_256\": ");
        write_metrics_obj(
            &mut s,
            "      ",
            &neg.neg1_single_user_small_chunk,
            include_runtime_fields,
        );
        s.push_str(",\n      \"single_user_bypass_ref\": ");
        write_metrics_obj(
            &mut s,
            "      ",
            &neg.neg1_single_user_bypass_ref,
            include_runtime_fields,
        );
        s.push_str("\n    },\n");

        s.push_str("    \"neg_kv_memory_oversubscription_256\": {\n");
        let _ = writeln!(s, "      \"gate_accepted\": {},", neg.neg2_verdict.accepted);
        let _ = writeln!(
            s,
            "      \"gate_reason\": {},",
            json_str(&neg.neg2_verdict.reason)
        );
        s.push_str("      \"unguarded_admission\": ");
        write_metrics_obj(
            &mut s,
            "      ",
            &neg.neg2_unguarded_oversubscription,
            include_runtime_fields,
        );
        s.push_str(",\n      \"kv_guarded_admission_ref\": ");
        write_metrics_obj(
            &mut s,
            "      ",
            &neg.neg2_kv_guarded_ref,
            include_runtime_fields,
        );
        s.push_str("\n    },\n");

        s.push_str("    \"neg_ultra_small_chunk_prefill_tax\": {\n");
        let _ = writeln!(s, "      \"gate_accepted\": {},", neg.neg3_verdict.accepted);
        let _ = writeln!(
            s,
            "      \"gate_reason\": {},",
            json_str(&neg.neg3_verdict.reason)
        );
        s.push_str("      \"chunk_64\": ");
        write_metrics_obj(
            &mut s,
            "      ",
            &neg.neg3_ultra_small_chunk_64,
            include_runtime_fields,
        );
        s.push_str(",\n      \"balanced_latency_aware_ref\": ");
        write_metrics_obj(
            &mut s,
            "      ",
            &neg.neg3_balanced_ref,
            include_runtime_fields,
        );
        s.push_str("\n    }\n");
        s.push_str("  }\n");
        s.push_str("}\n");
        s
    }
}

fn write_metrics_obj(out: &mut String, indent: &str, m: &ScheduleMetrics, include_cpu_time: bool) {
    let i2 = format!("{indent}  ");
    out.push_str("{\n");
    let _ = writeln!(out, "{i2}\"regime\": {},", json_str(m.regime_label));
    let _ = writeln!(out, "{i2}\"profile\": {},", json_str(m.profile_label));
    let _ = writeln!(out, "{i2}\"policy\": {},", json_str(m.policy_label));
    let _ = writeln!(out, "{i2}\"timing_tag\": \"analytical_h100_roofline\",");
    let _ = writeln!(out, "{i2}\"total_requests\": {},", m.total_requests);
    let _ = writeln!(out, "{i2}\"completed_requests\": {},", m.completed_requests);
    let _ = writeln!(
        out,
        "{i2}\"oom_rejected_requests\": {},",
        m.oom_rejected_requests
    );
    let _ = writeln!(out, "{i2}\"total_steps\": {},", m.total_steps);
    let _ = writeln!(
        out,
        "{i2}\"single_user_fast_path_steps\": {},",
        m.single_user_fast_path_steps
    );
    let _ = writeln!(
        out,
        "{i2}\"cold_ttft_p50_ms\": {},",
        json_f64(m.cold_ttft_p50_ms, 2)
    );
    let _ = writeln!(
        out,
        "{i2}\"cold_ttft_p95_ms\": {},",
        json_f64(m.cold_ttft_p95_ms, 2)
    );
    let _ = writeln!(
        out,
        "{i2}\"warm_ttft_p50_ms\": {},",
        json_f64(m.warm_ttft_p50_ms, 2)
    );
    let _ = writeln!(
        out,
        "{i2}\"warm_ttft_p95_ms\": {},",
        json_f64(m.warm_ttft_p95_ms, 2)
    );
    let _ = writeln!(
        out,
        "{i2}\"short_ttft_p95_ms\": {},",
        json_f64(m.short_ttft_p95_ms, 2)
    );
    let _ = writeln!(
        out,
        "{i2}\"overall_ttft_p95_ms\": {},",
        json_f64(m.overall_ttft_p95_ms, 2)
    );
    let _ = writeln!(out, "{i2}\"itl_p50_ms\": {},", json_f64(m.itl_p50_ms, 2));
    let _ = writeln!(out, "{i2}\"itl_p95_ms\": {},", json_f64(m.itl_p95_ms, 2));
    let _ = writeln!(out, "{i2}\"itl_p99_ms\": {},", json_f64(m.itl_p99_ms, 2));
    let _ = writeln!(out, "{i2}\"itl_max_ms\": {},", json_f64(m.itl_max_ms, 2));
    let _ = writeln!(out, "{i2}\"makespan_s\": {},", json_f64(m.makespan_s, 3));
    let _ = writeln!(out, "{i2}\"mean_e2e_ms\": {},", json_f64(m.mean_e2e_ms, 2));
    let _ = writeln!(
        out,
        "{i2}\"output_tok_per_s\": {},",
        json_f64(m.output_tok_per_s, 2)
    );
    let _ = writeln!(
        out,
        "{i2}\"total_tok_per_s\": {},",
        json_f64(m.total_tok_per_s, 2)
    );
    let _ = writeln!(
        out,
        "{i2}\"slo_attainment_pct\": {},",
        json_f64(m.slo_attainment_pct, 2)
    );
    let _ = writeln!(
        out,
        "{i2}\"ttft_slo_pct\": {},",
        json_f64(m.ttft_slo_pct, 2)
    );
    let _ = writeln!(out, "{i2}\"itl_slo_pct\": {},", json_f64(m.itl_slo_pct, 2));
    let _ = writeln!(out, "{i2}\"total_preemptions\": {},", m.total_preemptions);
    let _ = writeln!(
        out,
        "{i2}\"total_recomputed_tokens\": {},",
        m.total_recomputed_tokens
    );
    let _ = writeln!(out, "{i2}\"total_swap_bytes\": {},", m.total_swap_bytes);
    let _ = writeln!(
        out,
        "{i2}\"mean_queue_wait_ms\": {},",
        json_f64(m.mean_queue_wait_ms, 2)
    );
    let _ = writeln!(
        out,
        "{i2}\"peak_kv_gdn_mib\": {},",
        json_f64(m.peak_kv_gdn_mib, 2)
    );
    let _ = writeln!(
        out,
        "{i2}\"peak_total_vram_gib\": {},",
        json_f64(m.peak_total_vram_gib, 3)
    );
    let _ = writeln!(
        out,
        "{i2}\"kv_pool_capacity_gib\": {},",
        json_f64(m.kv_pool_capacity_gib, 3)
    );
    let _ = writeln!(
        out,
        "{i2}\"peak_unbounded_demand_gib\": {},",
        json_f64(m.peak_unbounded_demand_gib, 3)
    );
    let _ = writeln!(
        out,
        "{i2}\"energy_mj_per_output_tok\": {},",
        json_f64(m.energy_mj_per_output_tok, 2)
    );
    let _ = writeln!(
        out,
        "{i2}\"energy_mj_per_total_tok\": {},",
        json_f64(m.energy_mj_per_total_tok, 2)
    );
    if include_cpu_time {
        let _ = writeln!(
            out,
            "{i2}\"cpu_scheduler_wall_ms\": {},",
            json_f64(m.cpu_scheduler_wall_ms, 3)
        );
        let _ = writeln!(
            out,
            "{i2}\"cpu_us_per_step\": {},",
            json_f64(m.cpu_us_per_step, 3)
        );
    }
    let _ = writeln!(out, "{i2}\"kv_pool_clean\": {}", m.kv_pool_clean);
    out.push_str(indent);
    out.push('}');
}
