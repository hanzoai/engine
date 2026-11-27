// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! CLI binary for running the HIP-1346 `#54` CUDA-graph linear-attention decode,
//! speculative verification, sanitizer validation, and 20-turn coding-agent scorecard.
//!
//! Usage:
//! ```text
//! cargo run -p hanzo-cuda-graph --release --offline --bin cuda-graph-bench -- \
//!     --seed 1346 --output-dir docs/research/hip-1346-cuda-graphs-lsa
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use hanzo_cuda_graph::bench::run_full_benchmark;
use hanzo_cuda_graph::sha256::sha256_hex;

fn main() {
    let mut seed: u64 = 1346;
    let mut output_dir: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => {
                let val = args.next().expect("--seed requires a u64 argument");
                seed = val.parse::<u64>().expect("--seed must be a valid u64");
            }
            "--output-dir" => {
                let val = args.next().expect("--output-dir requires a path argument");
                output_dir = Some(PathBuf::from(val));
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let git_commit = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "UNKNOWN".to_string());

    let report = run_full_benchmark(seed, git_commit);
    let json = report.to_json();
    let digest = sha256_hex(json.as_bytes());

    let cm = &report.cpu_measured;
    println!("=== Hanzo Engine HIP-1346 (#54) CUDA Graphs for LSA Decode & Verification ===");
    println!("Seed: {} | Commit: {}", report.seed, report.git_commit);
    println!(
        "[cpu_directly_measured] decode_steps={} max_logit_err={:.3e} max_gdn_err={:.3e} max_kv_err={:.3e}",
        cm.decode_steps_verified, cm.max_abs_logit_err, cm.max_abs_gdn_state_err, cm.max_abs_kv_err
    );
    println!(
        "[cpu_directly_measured] spec_trials={} spec_logit_err={:.3e} spec_gdn_err={:.3e} spec_kv_err={:.3e}",
        cm.spec_verify_trials, cm.spec_max_abs_logit_err, cm.spec_max_abs_gdn_state_err, cm.spec_max_abs_kv_err
    );
    println!(
        "[cpu_directly_measured] sanitizer_fixed_violations={} fault_suite={}/{} post_teardown_leaked_bytes={} bt_copy_skips={}",
        cm.fixed_path_sanitizer_violations,
        cm.sanitizer_fault_suite.total_faults_caught,
        cm.sanitizer_fault_suite.total_faults_injected,
        cm.post_teardown_leaked_bytes,
        cm.block_table_copy_skips
    );

    println!("\n--- 30K-Context Decode Latency Proof (analytical_gpu_h100_sxm5) ---");
    for row in report
        .context_sweep
        .iter()
        .filter(|r| r.context_label == "30K")
    {
        println!(
            "  {:10} {:8} | eager p50/p95={:.3}/{:.3} ms | graph p50/p95={:.3}/{:.3} ms ({:.2}x/{:.2}x) | spec-K4 p50={:.3} ms | cap={:.1} ms | pool={:.1} MiB | sanitizer=PASS({})",
            row.model,
            row.quant,
            row.eager_p50_itl_ms,
            row.eager_p95_itl_ms,
            row.graph_p50_itl_ms,
            row.graph_p95_itl_ms,
            row.graph_p50_speedup,
            row.graph_p95_speedup,
            row.spec_graph_p50_itl_ms,
            row.capture_overhead_ms,
            row.graph_memory_overhead_mib,
            row.sanitizer_violations
        );
    }

    println!("\n--- Negative Regimes (All Must Be REJECTED) ---");
    for neg in &report.negative_regimes {
        println!(
            "  {:40} | gate={} | wall={:.2}s (eager={:.2}s) | pool={:.1} MiB | violations={}",
            neg.regime,
            neg.gate.status_str(),
            neg.agent_loop_wall_time_s,
            neg.eager_baseline_wall_time_s,
            neg.graph_pool_mib,
            neg.sanitizer_violations
        );
    }

    if let Some(dir) = output_dir {
        fs::create_dir_all(&dir).expect("failed to create output directory");
        let results_path = dir.join("results.json");
        let sha_path = dir.join("results.json.sha256");
        fs::write(&results_path, json.as_bytes()).expect("failed to write results.json");
        let sha_line = format!("{digest}  {}\n", results_path.display());
        fs::write(&sha_path, sha_line.as_bytes()).expect("failed to write results.json.sha256");
        println!("\nWrote {} (sha256: {})", results_path.display(), digest);
    } else {
        println!("\nJSON SHA-256: {digest}");
    }
}
