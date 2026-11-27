// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! CLI benchmark runner for `hanzo-zkml` (HIP-1350, `hanzoai/engine#68`).
//!
//! Usage:
//! ```bash
//! cargo run -p hanzo-zkml --release --offline --bin zkml-bench -- \
//!     --seed 1350 --trials 15 --output-dir docs/research/hip-1350-zkml-zkvm-proofs
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use hanzo_zkml::bench_harness::run_full_benchmark_suite;
use hanzo_zkml::merkle::LigeroParams;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let mut seed: u64 = 1350;
    let mut trials: usize = 15;
    let mut output_dir: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                if let Some(val) = args.get(i + 1).and_then(|s| s.parse::<u64>().ok()) {
                    seed = val;
                    i += 2;
                } else {
                    eprintln!("error: --seed expects a u64");
                    return ExitCode::FAILURE;
                }
            }
            "--trials" => {
                if let Some(val) = args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                    trials = val;
                    i += 2;
                } else {
                    eprintln!("error: --trials expects a usize");
                    return ExitCode::FAILURE;
                }
            }
            "--output-dir" => {
                if let Some(val) = args.get(i + 1) {
                    output_dir = Some(PathBuf::from(val));
                    i += 2;
                } else {
                    eprintln!("error: --output-dir expects a directory path");
                    return ExitCode::FAILURE;
                }
            }
            other => {
                eprintln!("error: unknown argument '{other}'");
                return ExitCode::FAILURE;
            }
        }
    }

    let params = LigeroParams::default();
    let report = match run_full_benchmark_suite(seed, trials, params) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("benchmark failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let (json, digest_hex) = report.to_canonical_json_with_sha256();

    println!("=== hanzo-zkml (HIP-1350 / hanzoai/engine#68) Benchmark Report ===");
    println!(
        "Seed: {} | Trials: {} | Ligero: blowup={}, queries={} ({:.2}-bit proximity) | Sumcheck: F_{{p^2}} (~128-bit)",
        report.seed,
        report.trials,
        report.ligero_blowup,
        report.ligero_queries,
        report.ligero_proximity_soundness_bits,
    );
    println!();
    println!(
        "{:<36} {:<22} {:>7} {:>6} {:>9} {:>9} {:>9} {:>10}",
        "Workload", "Backend", "Constr", "Cycles", "Prove p50", "Verify p50", "Proof B", "$/M tok"
    );
    println!("{}", "-".repeat(118));
    for w in &report.workloads {
        println!(
            "{:<36} {:<22} {:>7} {:>6} {:>7.3}ms {:>8.3}ms {:>9} {:>10.6}",
            w.name,
            w.backend,
            w.constraints_count,
            w.trace_cycles,
            w.prove_p50_ms,
            w.verify_p50_ms,
            w.proof_bytes,
            w.amortized_cost_usd_per_million_tokens,
        );
    }
    println!();
    let rejected = report
        .adversarial_checks
        .iter()
        .filter(|c| c.rejected)
        .count();
    println!(
        "Adversarial Rejection Matrix: {}/{} rejected ({:.1}%)",
        rejected,
        report.adversarial_checks.len(),
        report.adversarial_detection_rate * 100.0
    );
    println!(
        "27B Decode Negative Regime: {:.2} ns/constraint => {:.2} hrs/token (~${:.2}/M tok) [{}]",
        report.negative_regime_27b.measured_ns_per_constraint,
        report
            .negative_regime_27b
            .extrapolated_single_token_prove_hours,
        report
            .negative_regime_27b
            .extrapolated_cost_usd_per_million_tokens,
        report.negative_regime_27b.verdict
    );
    println!("Results SHA-256: {digest_hex}");

    if let Some(dir) = output_dir {
        if let Err(e) = fs::create_dir_all(&dir) {
            eprintln!("failed to create output directory: {e}");
            return ExitCode::FAILURE;
        }
        let json_path = dir.join("results.json");
        if let Err(e) = fs::write(&json_path, json.as_bytes()) {
            eprintln!("failed to write results.json: {e}");
            return ExitCode::FAILURE;
        }
        let sha_path = dir.join("results.json.sha256");
        if let Err(e) = fs::write(&sha_path, format!("{digest_hex}  results.json\n")) {
            eprintln!("failed to write results.json.sha256: {e}");
            return ExitCode::FAILURE;
        }
        println!("Wrote {} and {}", json_path.display(), sha_path.display());
    }

    ExitCode::SUCCESS
}
