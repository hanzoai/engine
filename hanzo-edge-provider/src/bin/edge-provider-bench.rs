// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reproducible HIP-1350 CLI benchmark (`edge-provider-bench`).
//!
//! Usage:
//! ```text
//! edge-provider-bench --seed 1350 --output-dir docs/research/hip-1350-edge-mobile-providers
//! ```

#![forbid(unsafe_code)]

use hanzo_edge_provider::bench::{
    detect_repo_root, render_report_json, run_full_hip1350_benchmark,
};
use hanzo_edge_provider::sha256::sha256_hex;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let mut seed: u64 = 1350;
    let mut output_dir: Option<PathBuf> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                if let Some(val) = args.get(i + 1) {
                    match val.parse::<u64>() {
                        Ok(s) => seed = s,
                        Err(e) => {
                            eprintln!("invalid --seed value '{val}': {e}");
                            return ExitCode::FAILURE;
                        }
                    }
                    i += 2;
                } else {
                    eprintln!("missing value after --seed");
                    return ExitCode::FAILURE;
                }
            }
            "--output-dir" => {
                if let Some(val) = args.get(i + 1) {
                    output_dir = Some(PathBuf::from(val));
                    i += 2;
                } else {
                    eprintln!("missing value after --output-dir");
                    return ExitCode::FAILURE;
                }
            }
            "--help" | "-h" => {
                println!(
                    "edge-provider-bench — Hanzo Engine HIP-1350 Decentralized Edge & Mobile Provider Benchmark\n\
                     Options:\n\
                       --seed <u64>          Deterministic seed (default: 1350)\n\
                       --output-dir <path>   Directory to write results.json and results.json.sha256"
                );
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::FAILURE;
            }
        }
    }

    let repo_root = detect_repo_root();
    let report = match run_full_hip1350_benchmark(&repo_root, seed) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("HIP-1350 edge-provider benchmark failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let json = render_report_json(&report);
    let digest_hex = sha256_hex(json.as_bytes());

    if let Some(dir) = output_dir {
        if let Err(e) = fs::create_dir_all(&dir) {
            eprintln!("failed to create output dir {}: {e}", dir.display());
            return ExitCode::FAILURE;
        }
        let json_path = dir.join("results.json");
        let sha_path = dir.join("results.json.sha256");
        if let Err(e) = fs::write(&json_path, &json) {
            eprintln!("failed to write {}: {e}", json_path.display());
            return ExitCode::FAILURE;
        }
        // For relative output directories (e.g. docs/research/hip-1350-edge-mobile-providers),
        // record the repository-relative path so `sha256sum -c <dir>/results.json.sha256`
        // verifies cleanly from the repository root without leaking absolute host paths.
        let sha_target = if dir.is_relative() {
            json_path.to_string_lossy().replace('\\', "/")
        } else {
            "results.json".to_string()
        };
        let sha_line = format!("{digest_hex}  {sha_target}\n");
        if let Err(e) = fs::write(&sha_path, sha_line) {
            eprintln!("failed to write {}: {e}", sha_path.display());
            return ExitCode::FAILURE;
        }
        println!("Wrote {} (sha256={digest_hex})", json_path.display());
    }

    let lt = &report.live_tcp_market;
    let fraud_ok = report.fraud_cases.iter().filter(|c| c.rejected).count();
    let neg_rejected = report
        .negative_regimes
        .iter()
        .filter(|n| n.verdict == "REJECTED")
        .count();

    println!("=== Hanzo Engine HIP-1350 Decentralized Edge & Mobile Providers Benchmark ===");
    println!(
        "Seed: {} | Corpus SHA-256: {}",
        report.seed, report.corpus.combined_sha256_hex
    );
    println!(
        "Live TCP Mac+AMD Market: completed={}/{} ({:.1}%) | mac_served={} amd_served={} handovers={}",
        lt.completed_requests,
        lt.total_requests,
        lt.completion_rate_pct,
        lt.mac_requests_served,
        lt.amd_requests_served,
        lt.handover_count
    );
    println!(
        "Live Thermal DVFS: mac_peak={:.2}C (max_throttle={}) amd_peak={:.2}C | reason={} | post_withdraw_rejected={}",
        lt.mac_peak_temp_c,
        lt.mac_max_throttle,
        lt.amd_peak_temp_c,
        lt.withdrawal_reason_code,
        lt.post_withdrawal_rejected
    );
    println!(
        "Live Wall ITL Progression: Full={:.3}ms -> Throttle70={:.3}ms -> Throttle45={:.3}ms | p95={:.3}ms | TTFT={:.3}ms",
        lt.measured_wall_full_itl_ms,
        lt.measured_wall_throttle70_itl_ms,
        lt.measured_wall_throttle45_itl_ms,
        lt.measured_wall_p95_itl_ms,
        lt.measured_wall_mean_ttft_ms
    );
    println!(
        "Receipts Verified: {} | Spot Checks: {} | Guardrails: {}/{} | Fraud Suite: {}/{} | Neg Regimes Rejected: {}/{}",
        lt.receipts_verified,
        lt.spot_checks_executed,
        lt.guardrails_verified,
        lt.guardrails_total,
        fraud_ok,
        report.fraud_cases.len(),
        neg_rejected,
        report.negative_regimes.len()
    );
    println!(
        "Coordinator Required: {} | Acceptance Gate: {}",
        report.sample_advertisement.hanzo_coordinator_required,
        if report.acceptance_gate_passed {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!(
        "Canonical SHA-256: {}",
        report.deterministic_canonical_sha256
    );
    println!("Receipt SHA-256:   {digest_hex}");

    if report.acceptance_gate_passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
