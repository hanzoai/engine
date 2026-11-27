// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end HIP-1346 benchmark orchestration and deterministic zero-dependency JSON serialization.

use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;

use crate::awq::{calibrate_awq, AwqConfig};
use crate::compressor::{ComponentSensitivityProfile, MixedPrecisionPlan, QadReport};
use crate::corpus::{EvalDomain, GroundedCorpus};
use crate::error::MixedQuantError;
use crate::formats::QuantFormat;
use crate::matrix::{build_shared_matrix_scorecard, MatrixScorecardRow};
use crate::model::{
    compute_bf16_reference_logits, evaluate_regime, Fp32HybridModel, ProxyModelConfig, QuantRegime,
    RegimeEvalReport,
};
use crate::rotation::{optimize_spinquant_rotation, SpinQuantResult};
use crate::sha256::Sha256;

/// Captured execution environment metadata (mandated by `hanzoai/engine#49` scorecard).
#[derive(Debug, Clone)]
pub struct HardwareEnvironment {
    pub cpu_arch: String,
    pub cpu_model: String,
    pub os_kernel: String,
    pub gpu_driver_note: String,
    pub git_commit: String,
    pub rustc_version: String,
}

impl HardwareEnvironment {
    /// Probes local CPU/OS/git/rustc metadata cleanly without failing if any probe is unavailable.
    #[must_use]
    pub fn probe() -> Self {
        let cpu_arch = std::env::consts::ARCH.to_string();
        let cpu_model = read_cpu_model().unwrap_or_else(|| format!("{cpu_arch} generic"));
        let os_kernel = run_stdout("uname", &["-srm"])
            .unwrap_or_else(|| format!("{} {}", std::env::consts::OS, std::env::consts::ARCH));
        let gpu_driver_note = run_stdout(
            "nvidia-smi",
            &["--query-gpu=name,driver_version", "--format=csv,noheader"],
        )
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            "CPU-only host (no local CUDA/ROCm/Metal GPU detected; 4B/9B/27B GPU timings are explicitly tagged analytical H100-SXM5 projections)".to_string()
        });
        let git_commit = run_stdout("git", &["rev-parse", "--short=12", "HEAD"])
            .unwrap_or_else(|| "unknown".to_string());
        let rustc_version =
            run_stdout("rustc", &["--version"]).unwrap_or_else(|| "rustc unknown".to_string());

        Self {
            cpu_arch,
            cpu_model,
            os_kernel,
            gpu_driver_note,
            git_commit,
            rustc_version,
        }
    }
}

/// Complete reproducible benchmark bundle for `hanzoai/engine#49`.
#[derive(Debug, Clone)]
pub struct BenchmarkBundle {
    pub seed: u64,
    pub env: HardwareEnvironment,
    pub corpus_sha256: String,
    pub corpus_total_bytes: usize,
    pub corpus_files: Vec<(String, usize, String)>,
    pub spinquant_kernel_summary: SpinQuantResult,
    pub awq_kernel_best_alpha: f32,
    pub awq_kernel_baseline_mse: f64,
    pub awq_kernel_awq_mse: f64,
    pub awq_kernel_rel_reduction_pct: f64,
    pub awq_kernel_outlier_ratio: f32,
    pub component_profiles: Vec<ComponentSensitivityProfile>,
    pub mixed_plan: MixedPrecisionPlan,
    pub qad_reports: Vec<(String, QadReport)>,
    pub regimes: Vec<RegimeEvalReport>,
    pub shared_matrix: Vec<MatrixScorecardRow>,
    pub receipt_sha256: String,
}

/// Executes the full reproducible HIP-1346 benchmark suite deterministically from `seed`.
pub fn run_full_benchmark(
    seed: u64,
    repo_root: Option<&Path>,
) -> Result<BenchmarkBundle, MixedQuantError> {
    let env = HardwareEnvironment::probe();
    let corpus = GroundedCorpus::load(repo_root)?;

    let cfg = ProxyModelConfig::default();
    let teacher = Fp32HybridModel::new(cfg.clone(), seed);

    // Matched multi-domain calibration trace (all 4 domains)
    let matched_seqs = corpus.matched_calibration_sequences(16, 24, cfg.vocab_size, seed);
    let matched_trace = teacher.collect_calibration_trace(&matched_seqs)?;

    // Mismatched prose-only calibration trace (Negative Regime 3)
    let prose_only_seqs = corpus.mismatched_prose_only_calibration(16, 24, cfg.vocab_size, seed);
    let prose_only_trace = teacher.collect_calibration_trace(&prose_only_seqs)?;

    // Standalone SpinQuant Cayley kernel verification on normal semantic weights under heavy-tailed layer_in[0] outliers
    let spin_tokens = matched_trace.num_tokens.min(96);
    let spinquant_kernel_summary = optimize_spinquant_rotation(
        &teacher.blocks[0].seq_out_proj,
        cfg.d_model,
        cfg.d_model,
        &matched_trace.layer_in[0][..spin_tokens * cfg.d_model],
        spin_tokens,
        QuantFormat::Int4Asym,
        cfg.group_size,
        4,
        8,
        seed,
    )?;

    // Standalone AWQ kernel verification on layer 0 seq_in
    let awq_res = calibrate_awq(
        &teacher.blocks[0].seq_in_proj,
        3 * cfg.d_model,
        cfg.d_model,
        &matched_trace.layer_in[0],
        matched_trace.num_tokens,
        &AwqConfig {
            format: QuantFormat::Int4Asym,
            group_size: cfg.group_size,
            grid_steps: 20,
            salient_ratio: 0.03,
            protect_salient_bf16: true,
        },
    )?;

    let component_profiles = teacher.profile_components(&matched_trace)?;

    // Build BF16 reference model and compute teacher reference logits
    let bf16_model = teacher.quantize(QuantRegime::Bf16, &matched_trace, seed)?;
    let bf16_bytes = bf16_model.storage_bytes();
    let bf16_refs = compute_bf16_reference_logits(&bf16_model, &corpus, seed)?;

    let ordered_regimes = [
        QuantRegime::Bf16,
        QuantRegime::UniformInt8,
        QuantRegime::UniformInt4,
        QuantRegime::UniformTernary,
        QuantRegime::AwqInt4,
        QuantRegime::SpinQuantInt4,
        QuantRegime::MixedPrecisionAwqSpinQuant,
        QuantRegime::UnmergedRotScalarDequant,
        QuantRegime::MismatchedProseCalAwqInt4,
    ];

    let mut regime_reports = Vec::with_capacity(ordered_regimes.len());
    let mut mixed_plan_opt: Option<MixedPrecisionPlan> = None;
    let mut qad_reports: Vec<(String, QadReport)> = Vec::new();

    for &regime in &ordered_regimes {
        let model = match regime {
            QuantRegime::Bf16 => bf16_model.clone(),
            QuantRegime::MismatchedProseCalAwqInt4 => {
                teacher.quantize(regime, &prose_only_trace, seed)?
            }
            _ => teacher.quantize(regime, &matched_trace, seed)?,
        };
        if regime == QuantRegime::MixedPrecisionAwqSpinQuant {
            mixed_plan_opt = model.plan.clone();
            qad_reports = model.qad_reports.clone();
        }
        let report = evaluate_regime(&model, &corpus, &bf16_refs, bf16_bytes, seed)?;
        regime_reports.push(report);
    }

    let mixed_plan = mixed_plan_opt.ok_or(MixedQuantError::InvalidCalibrationData {
        reason: "mixed precision plan was not generated",
    })?;

    let shared_matrix = build_shared_matrix_scorecard(&regime_reports);

    // Deterministic receipt hash over deterministic structural & quality outputs (excluding wall-clock jitter)
    let mut receipt = Sha256::new();
    receipt.update(&seed.to_le_bytes());
    receipt.update(corpus.combined_sha256.as_bytes());
    for r in &regime_reports {
        receipt.update(r.regime.as_str().as_bytes());
        receipt.update(&(r.model_bytes as u64).to_le_bytes());
        let acc_bps = (r.mean_accuracy_pct * 100.0).round() as i64;
        receipt.update(&acc_bps.to_le_bytes());
        for d in &r.domains {
            let d_bps = (d.accuracy_pct * 100.0).round() as i64;
            receipt.update(&d_bps.to_le_bytes());
        }
    }
    let receipt_sha256 = receipt.finalize_hex();

    let corpus_files = corpus
        .files
        .iter()
        .map(|f| (f.rel_path.clone(), f.bytes, f.sha256.clone()))
        .collect();

    Ok(BenchmarkBundle {
        seed,
        env,
        corpus_sha256: corpus.combined_sha256,
        corpus_total_bytes: corpus.total_bytes,
        corpus_files,
        spinquant_kernel_summary,
        awq_kernel_best_alpha: awq_res.best_alpha,
        awq_kernel_baseline_mse: awq_res.baseline_ptq_mse,
        awq_kernel_awq_mse: awq_res.awq_mse,
        awq_kernel_rel_reduction_pct: awq_res.relative_mse_reduction * 100.0,
        awq_kernel_outlier_ratio: awq_res.outlier_ratio,
        component_profiles,
        mixed_plan,
        qad_reports,
        regimes: regime_reports,
        shared_matrix,
        receipt_sha256,
    })
}

impl BenchmarkBundle {
    /// Serializes the benchmark bundle into clean, deterministic JSON with zero external dependencies.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(32 * 1024);
        s.push_str("{\n");
        let _ = writeln!(s, "  \"issue\": \"hanzoai/engine#49\",");
        let _ = writeln!(s, "  \"hip\": \"HIP-1346\",");
        let _ = writeln!(s, "  \"seed\": {},", self.seed);
        let _ = writeln!(s, "  \"receipt_sha256\": \"{}\",", self.receipt_sha256);

        // Environment
        s.push_str("  \"environment\": {\n");
        let _ = writeln!(s, "    \"cpu_arch\": \"{}\",", esc(&self.env.cpu_arch));
        let _ = writeln!(s, "    \"cpu_model\": \"{}\",", esc(&self.env.cpu_model));
        let _ = writeln!(s, "    \"os_kernel\": \"{}\",", esc(&self.env.os_kernel));
        let _ = writeln!(
            s,
            "    \"gpu_driver\": \"{}\",",
            esc(&self.env.gpu_driver_note)
        );
        let _ = writeln!(s, "    \"git_commit\": \"{}\",", esc(&self.env.git_commit));
        let _ = writeln!(
            s,
            "    \"rustc_version\": \"{}\"",
            esc(&self.env.rustc_version)
        );
        s.push_str("  },\n");

        // Corpus
        s.push_str("  \"corpus\": {\n");
        let _ = writeln!(s, "    \"combined_sha256\": \"{}\",", self.corpus_sha256);
        let _ = writeln!(s, "    \"total_bytes\": {},", self.corpus_total_bytes);
        s.push_str("    \"files\": [\n");
        for (i, (path, bytes, sha)) in self.corpus_files.iter().enumerate() {
            let comma = if i + 1 < self.corpus_files.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(
                s,
                "      {{\"path\": \"{}\", \"bytes\": {}, \"sha256\": \"{}\"}}{}",
                esc(path),
                bytes,
                sha,
                comma
            );
        }
        s.push_str("    ]\n  },\n");

        // Kernel verifications (AWQ + SpinQuant + QAD)
        s.push_str("  \"kernel_verifications\": {\n");
        s.push_str("    \"awq\": {\n");
        let _ = writeln!(
            s,
            "      \"best_alpha\": {:.4},",
            self.awq_kernel_best_alpha
        );
        let _ = writeln!(
            s,
            "      \"baseline_ptq_mse\": {:.8},",
            self.awq_kernel_baseline_mse
        );
        let _ = writeln!(s, "      \"awq_mse\": {:.8},", self.awq_kernel_awq_mse);
        let _ = writeln!(
            s,
            "      \"relative_mse_reduction_pct\": {:.2},",
            self.awq_kernel_rel_reduction_pct
        );
        let _ = writeln!(
            s,
            "      \"channel_outlier_ratio\": {:.3}",
            self.awq_kernel_outlier_ratio
        );
        s.push_str("    },\n");

        s.push_str("    \"spinquant_cayley\": {\n");
        let _ = writeln!(
            s,
            "      \"orthogonality_err_inf\": {:.8},",
            self.spinquant_kernel_summary.orthogonality_err_inf
        );
        let _ = writeln!(
            s,
            "      \"invariance_err_max\": {:.8},",
            self.spinquant_kernel_summary.invariance_err_max
        );
        let _ = writeln!(
            s,
            "      \"pre_rotation_crest\": {:.4},",
            self.spinquant_kernel_summary.pre_rotation_crest
        );
        let _ = writeln!(
            s,
            "      \"post_rotation_crest\": {:.4},",
            self.spinquant_kernel_summary.post_rotation_crest
        );
        let _ = writeln!(
            s,
            "      \"baseline_w4a4_mse\": {:.8},",
            self.spinquant_kernel_summary.baseline_mse
        );
        let _ = writeln!(
            s,
            "      \"hadamard_w4a4_mse\": {:.8},",
            self.spinquant_kernel_summary.hadamard_mse
        );
        let _ = writeln!(
            s,
            "      \"learned_cayley_w4a4_mse\": {:.8}",
            self.spinquant_kernel_summary.learned_mse
        );
        s.push_str("    },\n");

        // QAD aggregate summary
        let qad_n = self.qad_reports.len().max(1) as f64;
        let pre_mse: f64 = self
            .qad_reports
            .iter()
            .map(|(_, q)| q.pre_qad_mse)
            .sum::<f64>()
            / qad_n;
        let post_mse: f64 = self
            .qad_reports
            .iter()
            .map(|(_, q)| q.post_qad_mse)
            .sum::<f64>()
            / qad_n;
        let pre_kl: f64 = self
            .qad_reports
            .iter()
            .map(|(_, q)| q.pre_qad_kl)
            .sum::<f64>()
            / qad_n;
        let post_kl: f64 = self
            .qad_reports
            .iter()
            .map(|(_, q)| q.post_qad_kl)
            .sum::<f64>()
            / qad_n;
        s.push_str("    \"qad_refinement\": {\n");
        let _ = writeln!(s, "      \"refined_layers\": {},", self.qad_reports.len());
        let _ = writeln!(s, "      \"mean_pre_qad_mse\": {:.8},", pre_mse);
        let _ = writeln!(s, "      \"mean_post_qad_mse\": {:.8},", post_mse);
        let _ = writeln!(s, "      \"mean_pre_qad_kl\": {:.8},", pre_kl);
        let _ = writeln!(s, "      \"mean_post_qad_kl\": {:.8}", post_kl);
        s.push_str("    }\n  },\n");

        // Mixed-precision plan
        s.push_str("  \"mixed_precision_plan\": {\n");
        let _ = writeln!(s, "    \"budget_bytes\": {},", self.mixed_plan.budget_bytes);
        let _ = writeln!(s, "    \"planned_bytes\": {},", self.mixed_plan.total_bytes);
        let _ = writeln!(
            s,
            "    \"weighted_rel_mse\": {:.6},",
            self.mixed_plan.weighted_rel_mse
        );
        let _ = writeln!(
            s,
            "    \"uniform_int4_weighted_rel_mse\": {:.6},",
            self.mixed_plan.uniform_int4_weighted_rel_mse
        );
        s.push_str("    \"assignments\": [\n");
        for (i, (name, kind, fmt, bytes)) in self.mixed_plan.assignments.iter().enumerate() {
            let comma = if i + 1 < self.mixed_plan.assignments.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(
                s,
                "      {{\"name\": \"{}\", \"kind\": \"{}\", \"format\": \"{}\", \"bytes\": {}}}{}",
                esc(name),
                kind.as_str(),
                fmt.as_str(),
                bytes,
                comma
            );
        }
        s.push_str("    ]\n  },\n");

        // 2:4 pruning sensitivity summary
        s.push_str("  \"pruning_sensitivity\": [\n");
        for (i, p) in self.component_profiles.iter().enumerate() {
            let comma = if i + 1 < self.component_profiles.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(
                s,
                "    {{\"name\": \"{}\", \"kind\": \"{}\", \"rel_mse_2_4\": {:.6}, \"rel_mse_unstruct_50\": {:.6}, \"tolerates_2_4\": {}}}{}",
                esc(&p.name),
                p.kind.as_str(),
                p.pruning_2_4_rel_mse,
                p.pruning_unstructured_50_rel_mse,
                p.tolerates_2_4_pruning,
                comma
            );
        }
        s.push_str("  ],\n");

        // CPU measured end-to-end regimes across all 4 quality domains
        s.push_str("  \"cpu_measured_regimes\": [\n");
        for (i, r) in self.regimes.iter().enumerate() {
            let comma = if i + 1 < self.regimes.len() { "," } else { "" };
            s.push_str("    {\n");
            let _ = writeln!(s, "      \"regime\": \"{}\",", r.regime.as_str());
            let _ = writeln!(s, "      \"model_bytes\": {},", r.model_bytes);
            let _ = writeln!(
                s,
                "      \"effective_bits_per_weight\": {:.4},",
                r.effective_bits_per_weight
            );
            let _ = writeln!(
                s,
                "      \"compression_vs_bf16\": {:.4},",
                r.compression_vs_bf16
            );
            let _ = writeln!(
                s,
                "      \"mean_accuracy_pct\": {:.3},",
                r.mean_accuracy_pct
            );
            let _ = writeln!(
                s,
                "      \"quality_regression_pp\": {:.3},",
                r.quality_regression_pp
            );
            let _ = writeln!(
                s,
                "      \"worst_domain_regression_pp\": {:.3},",
                r.worst_domain_regression_pp
            );
            let _ = writeln!(s, "      \"mean_kl\": {:.6},", r.mean_kl);
            let _ = writeln!(s, "      \"cpu_cold_ttft_ms\": {:.3},", r.cpu_cold_ttft_ms);
            let _ = writeln!(s, "      \"cpu_warm_ttft_ms\": {:.3},", r.cpu_warm_ttft_ms);
            let _ = writeln!(s, "      \"cpu_p50_itl_ms\": {:.3},", r.cpu_p50_itl_ms);
            let _ = writeln!(s, "      \"cpu_p95_itl_ms\": {:.3},", r.cpu_p95_itl_ms);
            let _ = writeln!(
                s,
                "      \"cpu_warm_tok_per_s\": {:.2},",
                r.cpu_warm_tok_per_s
            );
            let _ = writeln!(s, "      \"cpu_wall_ms\": {:.3},", r.cpu_wall_ms);
            s.push_str("      \"domains\": {\n");
            for (di, d) in r.domains.iter().enumerate() {
                let d_comma = if di + 1 < r.domains.len() { "," } else { "" };
                let _ = writeln!(
                    s,
                    "        \"{}\": {{\"accuracy_pct\": {:.3}, \"mean_kl\": {:.6}, \"mean_cosine\": {:.6}, \"scored_positions\": {}}}{}",
                    d.domain.as_str(),
                    d.accuracy_pct,
                    d.mean_kl,
                    d.mean_cosine,
                    d.scored_positions,
                    d_comma
                );
            }
            let _ = writeln!(s, "      }}\n    }}{}", comma);
        }
        s.push_str("  ],\n");

        // Shared 4B / 9B / 27B 20-turn coding-agent scorecard
        s.push_str("  \"shared_matrix_scorecard\": [\n");
        for (i, row) in self.shared_matrix.iter().enumerate() {
            let comma = if i + 1 < self.shared_matrix.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(
                s,
                "    {{\"model\": \"{}\", \"params_b\": {:.2}, \"regime\": \"{}\", \"measurement_source\": \"{}\", \"model_size_gib\": {:.3}, \"peak_memory_gib\": {:.3}, \"avg_bits_per_weight\": {:.3}, \"cold_ttft_ms\": {:.2}, \"warm_ttft_ms\": {:.2}, \"p50_itl_ms\": {:.3}, \"p95_itl_ms\": {:.3}, \"warm_throughput_tok_s\": {:.2}, \"agent_loop_20turn_wall_s\": {:.2}, \"energy_mj_per_tok\": {:.2}, \"output_correctness_pct\": {:.2}, \"quality_regression_pp\": {:.2}, \"mean_kl\": {:.6}, \"acceptance\": \"{}\", \"acceptance_reason\": \"{}\"}}{}",
                esc(&row.model_name),
                row.total_params_b,
                row.regime.as_str(),
                esc(row.measurement_source),
                row.model_size_gib,
                row.peak_memory_gib,
                row.avg_bits_per_weight,
                row.cold_ttft_ms,
                row.warm_ttft_ms,
                row.p50_itl_ms,
                row.p95_itl_ms,
                row.warm_throughput_tok_s,
                row.agent_loop_20turn_wall_s,
                row.energy_mj_per_tok,
                row.output_correctness_pct,
                row.quality_regression_pp,
                row.mean_kl_divergence,
                row.acceptance.as_str(),
                esc(row.acceptance.reason()),
                comma
            );
        }
        s.push_str("  ]\n}\n");
        s
    }

    /// Formats a human-readable scorecard summary for terminal output.
    #[must_use]
    pub fn format_summary_table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "=== HIP-1346 Mixed-Precision Quantization Benchmark (hanzoai/engine#49, seed={}) ===",
            self.seed
        );
        let _ = writeln!(
            out,
            "Host: {} | {} | {} | commit={} | {}",
            self.env.cpu_arch,
            self.env.cpu_model,
            self.env.os_kernel,
            self.env.git_commit,
            self.env.rustc_version
        );
        let _ = writeln!(
            out,
            "Corpus SHA-256: {} ({} bytes across {} files)",
            self.corpus_sha256,
            self.corpus_total_bytes,
            self.corpus_files.len()
        );
        let _ = writeln!(
            out,
            "AWQ Kernel: alpha*={:.2}, outlier_ratio={:.2}x, MSE {:.6} -> {:.6} ({:.1}% reduction)",
            self.awq_kernel_best_alpha,
            self.awq_kernel_outlier_ratio,
            self.awq_kernel_baseline_mse,
            self.awq_kernel_awq_mse,
            self.awq_kernel_rel_reduction_pct
        );
        let _ = writeln!(
            out,
            "SpinQuant Cayley: ||R^T R - I||_inf={:.2e}, invariance={:.2e}, crest {:.2} -> {:.2}, W4A4 MSE {:.6} -> Hadamard {:.6} -> Cayley {:.6}",
            self.spinquant_kernel_summary.orthogonality_err_inf,
            self.spinquant_kernel_summary.invariance_err_max,
            self.spinquant_kernel_summary.pre_rotation_crest,
            self.spinquant_kernel_summary.post_rotation_crest,
            self.spinquant_kernel_summary.baseline_mse,
            self.spinquant_kernel_summary.hadamard_mse,
            self.spinquant_kernel_summary.learned_mse
        );
        out.push_str(
            "\n--- CPU-Measured End-to-End 4-Domain Quality & Equal-Memory Comparison ---\n",
        );
        let _ = writeln!(
            out,
            "{:<34} | {:>8} | {:>5} | {:>6} | {:>6} | {:>6} | {:>6} | {:>7} | {:>7} | {:>7}",
            "Regime",
            "Bytes",
            "BPW",
            "Instr%",
            "Code%",
            "Multi%",
            "Long%",
            "Mean%",
            "KL",
            "Wall_ms"
        );
        for r in &self.regimes {
            let acc = |dom: EvalDomain| {
                r.domains
                    .iter()
                    .find(|d| d.domain == dom)
                    .map(|d| d.accuracy_pct)
                    .unwrap_or(0.0)
            };
            let _ = writeln!(
                out,
                "{:<34} | {:>8} | {:>5.2} | {:>6.1} | {:>6.1} | {:>6.1} | {:>6.1} | {:>7.2} | {:>7.4} | {:>7.1}",
                r.regime.as_str(),
                r.model_bytes,
                r.effective_bits_per_weight,
                acc(EvalDomain::InstructionFollowing),
                acc(EvalDomain::CodeEditAccuracy),
                acc(EvalDomain::Multilingual),
                acc(EvalDomain::LongContextRetrieval),
                r.mean_accuracy_pct,
                r.mean_kl,
                r.cpu_wall_ms
            );
        }
        out.push_str("\n--- Shared 4B / 9B / 27B 20-Turn Coding-Agent Scorecard (Exact Bytes + CPU Quality + Modeled H100 GPU) ---\n");
        let _ = writeln!(
            out,
            "{:<10} | {:<34} | {:>6} | {:>6} | {:>7} | {:>7} | {:>8} | {:>6} | {:>7} | {:>8} | {:<8}",
            "Model",
            "Regime",
            "SizeG",
            "PeakG",
            "ColdTT",
            "WarmTT",
            "p50/p95",
            "Wall_s",
            "mJ/tok",
            "Corr%",
            "Gate"
        );
        for row in &self.shared_matrix {
            let _ = writeln!(
                out,
                "{:<10} | {:<34} | {:>6.2} | {:>6.2} | {:>7.1} | {:>7.1} | {:>3.1}/{:>4.1} | {:>6.2} | {:>7.1} | {:>7.1}% | {:<8}",
                row.model_name,
                row.regime.as_str(),
                row.model_size_gib,
                row.peak_memory_gib,
                row.cold_ttft_ms,
                row.warm_ttft_ms,
                row.p50_itl_ms,
                row.p95_itl_ms,
                row.agent_loop_20turn_wall_s,
                row.energy_mj_per_tok,
                row.output_correctness_pct,
                row.acceptance.as_str()
            );
        }
        let _ = writeln!(out, "\nReceipt SHA-256: {}", self.receipt_sha256);
        out
    }
}

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn run_stdout(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    Some(s.trim().to_string())
}

fn read_cpu_model() -> Option<String> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            if let Some((_, val)) = rest.split_once(':') {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
    }
    None
}
