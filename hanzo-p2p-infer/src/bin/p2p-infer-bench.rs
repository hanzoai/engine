// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reproducible HIP-1350 benchmark CLI (`p2p-infer-bench`).
//!
//! Usage:
//! ```text
//! p2p-infer-bench --seed 1350 --output-dir docs/research/hip-1350-p2p-distributed-inference
//! ```

#![forbid(unsafe_code)]

use hanzo_p2p_infer::bench::{detect_repo_root, render_report_json, run_full_hip1350_benchmark};
use hanzo_p2p_infer::sha256::{hex_encode, sha256};
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
                    "p2p-infer-bench — Hanzo Engine HIP-1350 P2P Distributed Inference Benchmark\n\
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
            eprintln!("HIP-1350 benchmark failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let json = render_report_json(&report);
    let digest_hex = hex_encode(&sha256(json.as_bytes()));

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
        let sha_line = format!("{digest_hex}  results.json\n");
        if let Err(e) = fs::write(&sha_path, sha_line) {
            eprintln!("failed to write {}: {e}", sha_path.display());
            return ExitCode::FAILURE;
        }
        println!("Wrote {} (sha256={digest_hex})", json_path.display());
    }

    let lt = &report.live_tcp;
    let detected = report
        .trust_fault_cases
        .iter()
        .filter(|c| c.detected)
        .count();
    println!("=== Hanzo Engine HIP-1350 P2P Distributed Inference Benchmark ===");
    println!(
        "Seed: {} | Corpus SHA-256: {}",
        report.seed, report.corpus.combined_sha256_hex
    );
    println!(
        "Live 3-Node TCP BF16: wall={:.2}ms tok/s={:.1} exact={}/{} max_diff={:.3e} receipts={} spot_checks={}",
        lt.tcp_3node_bf16_wall_ms,
        lt.tcp_3node_bf16_tok_per_sec,
        lt.tcp_3node_bf16_exact_matches,
        lt.gen_tokens,
        lt.tcp_3node_bf16_max_abs_logit_diff,
        lt.tcp_3node_bf16_verified_receipts,
        lt.tcp_3node_bf16_spot_checks_passed
    );
    println!(
        "Live 3-Node TCP INT8: wall={:.2}ms tok/s={:.1} compression={:.2}x cos={:.6} matches={}/{}",
        lt.tcp_3node_int8_wall_ms,
        lt.tcp_3node_int8_tok_per_sec,
        lt.tcp_3node_int8_activation_compression_ratio,
        lt.tcp_3node_int8_mean_logit_cosine,
        lt.tcp_3node_int8_token_matches,
        lt.gen_tokens
    );
    println!(
        "Live 3-Node Node-B Hard-Kill Failover: recovery={:.2}ms replayed_toks={} replacement={} completion={:.1}% exact={}/{} max_diff={:.3e}",
        lt.failover_recovery_ms,
        lt.failover_replayed_tokens,
        lt.failover_replacement_peer,
        lt.failover_completion_rate_pct,
        lt.failover_exact_matches,
        lt.gen_tokens,
        lt.failover_max_abs_logit_diff
    );
    println!(
        "Live Trust/Tamper Suite: {}/{} detected (100.0%={}) | Coordinator Required: {}",
        detected,
        report.trust_fault_cases.len(),
        detected == report.trust_fault_cases.len(),
        report.manifest.hanzo_coordinator_required
    );
    println!(
        "Acceptance Gate: {}",
        if report.acceptance_gate_passed {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!("Receipt SHA-256: {digest_hex}");

    if report.acceptance_gate_passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
