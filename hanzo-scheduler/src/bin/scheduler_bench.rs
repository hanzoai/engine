// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reproducible HIP-1346 benchmark CLI (`scheduler-bench`) for `hanzo-scheduler`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use hanzo_scheduler::bench::run_full_benchmark_suite;
use hanzo_scheduler::corpus::GroundedCorpus;

fn parse_args() -> Result<(u64, PathBuf), String> {
    let mut seed: u64 = 1346;
    let mut output_dir = PathBuf::from("docs/research/hip-1346-continuous-batching-scheduler");
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => {
                let val = args
                    .next()
                    .ok_or_else(|| "--seed requires an integer value".to_string())?;
                seed = val
                    .parse::<u64>()
                    .map_err(|e| format!("invalid --seed '{val}': {e}"))?;
            }
            "--output-dir" => {
                let val = args
                    .next()
                    .ok_or_else(|| "--output-dir requires a directory path".to_string())?;
                output_dir = PathBuf::from(val);
            }
            "--help" | "-h" => {
                println!(
                    "Usage: scheduler-bench [--seed <u64>] [--output-dir <path>]\n\
                     Runs the HIP-1346 continuous batching & chunked prefill benchmark suite \
                     and writes results.json."
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok((seed, output_dir))
}

fn main() -> ExitCode {
    let (seed, output_dir) = match parse_args() {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("error: {msg}");
            return ExitCode::FAILURE;
        }
    };

    let repo_root = GroundedCorpus::find_repo_root();
    let corpus = match GroundedCorpus::load(&repo_root) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error loading grounded corpus from {:?}: {e}", repo_root);
            return ExitCode::FAILURE;
        }
    };

    let report = match run_full_benchmark_suite(&corpus, seed) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("benchmark error: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = fs::create_dir_all(&output_dir) {
        eprintln!("failed to create output directory {:?}: {e}", output_dir);
        return ExitCode::FAILURE;
    }

    let json_path = output_dir.join("results.json");
    let json = report.to_json_pretty();
    if let Err(e) = fs::write(&json_path, &json) {
        eprintln!("failed to write {:?}: {e}", json_path);
        return ExitCode::FAILURE;
    }

    println!(
        "=== HIP-1346 Continuous Batching & Chunked Prefill Benchmark (hanzoai/engine#56) ==="
    );
    println!("Seed: {}", report.seed);
    println!(
        "Corpus: {} files, {} bytes, combined SHA-256: {}",
        report.corpus_files.len(),
        report.corpus_total_bytes,
        report.corpus_combined_sha256
    );
    println!(
        "CPU Token-Level Equivalence Verifier: {}/{} tokens ({:.2}%, reg {:.3} pp, max_logit_diff={:.2e}, preempt_verified={}, clean={}, cpu={:.2}ms)",
        report.verification.matched_tokens,
        report.verification.total_output_tokens,
        report.verification.output_correctness_pct,
        report.verification.quality_regression_pp,
        report.verification.max_abs_logit_diff,
        report.verification.preemption_recompute_verified,
        report.verification.kv_pool_clean,
        report.verification.cpu_verify_wall_ms,
    );
    println!("\n--- Concurrency Sweep (9B-Hybrid, H100-SXM5 80GB Analytical) ---");
    for row in &report.concurrency_sweep {
        let m = &row.metrics;
        println!(
            "C={:>3} | {:<28} | TTFT_p95={:>8.1}ms | ITL_p50/p95={:>5.1}/{:>6.1}ms | out={:>7.1} tok/s | SLO={:>5.1}% | preempt={:>3} | peak={:>5.2}GiB",
            row.concurrency,
            m.regime_label,
            m.overall_ttft_p95_ms,
            m.itl_p50_ms,
            m.itl_p95_ms,
            m.output_tok_per_s,
            m.slo_attainment_pct,
            m.total_preemptions,
            m.peak_total_vram_gib,
        );
    }
    println!("\n--- 4B / 9B / 27B Hybrid Matrix (20-Turn Agent + 16 Interactive) ---");
    for row in &report.hybrid_matrix_20turn {
        let m = &row.metrics;
        println!(
            "{:<10} | {:<33} | cold/warm TTFT={:>6.1}/{:>5.1}ms | ITL p50/p95={:>4.1}/{:>6.1}ms | wall={:>6.2}s | tok/s={:>6.1} | mJ/tok={:>7.1} | SLO={:>5.1}% | {}",
            row.model_label,
            row.regime_label,
            m.cold_ttft_p50_ms,
            m.warm_ttft_p50_ms,
            m.itl_p50_ms,
            m.itl_p95_ms,
            m.makespan_s,
            m.output_tok_per_s,
            m.energy_mj_per_output_tok,
            m.slo_attainment_pct,
            if row.verdict.accepted { "ACCEPTED" } else { "REJECTED" }
        );
    }
    println!("\n--- Honest Negative Regimes ---");
    println!("Neg1: {}", report.negative_regimes.neg1_verdict.reason);
    println!("Neg2: {}", report.negative_regimes.neg2_verdict.reason);
    println!("Neg3: {}", report.negative_regimes.neg3_verdict.reason);
    println!(
        "\nWrote {:?} (Receipt SHA-256: {})",
        json_path, report.receipt_sha256
    );

    ExitCode::SUCCESS
}
