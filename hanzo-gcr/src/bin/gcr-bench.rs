// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reproducible CLI benchmark runner for `hanzo-gcr` (HIP-1346 / `hanzoai/engine#91`).
//!
//! Usage:
//! ```bash
//! cargo run -p hanzo-gcr --release --offline --bin gcr-bench -- \
//!     --seed 1346 --iterations 3 --output docs/research/hip-1346-gcr/results.json
//! ```

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

use hanzo_gcr::bench::{run_full_benchmark, BenchOptions};

fn main() {
    if let Err(err) = run() {
        eprintln!("gcr-bench error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let default_repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    let mut seed: u64 = 1346;
    let mut iterations: usize = 3;
    let mut repo_root = default_repo_root.clone();
    let mut output_path = default_repo_root.join("docs/research/hip-1346-gcr/results.json");
    let mut fsync_timed_runs = false;
    let mut restore_workers: usize = 4;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => {
                let v = args.next().ok_or("--seed requires a u64 argument")?;
                seed = v.parse()?;
            }
            "--iterations" => {
                let v = args
                    .next()
                    .ok_or("--iterations requires a usize argument")?;
                iterations = v.parse()?;
            }
            "--repo-root" => {
                let v = args.next().ok_or("--repo-root requires a path argument")?;
                repo_root = PathBuf::from(v);
            }
            "--output" => {
                let v = args.next().ok_or("--output requires a path argument")?;
                output_path = PathBuf::from(v);
            }
            "--workers" => {
                let v = args.next().ok_or("--workers requires a usize argument")?;
                restore_workers = v.parse()?;
            }
            "--fsync" => {
                fsync_timed_runs = true;
            }
            "--help" | "-h" => {
                println!(
                    "gcr-bench (HIP-1346 / hanzoai/engine#91)\n\
                     Options:\n\
                       --seed <u64>          Deterministic seed (default: 1346)\n\
                       --iterations <usize>  Measured iterations per workload (default: 3)\n\
                       --repo-root <path>    Path to hanzoai/engine root for real corpus bytes\n\
                       --output <path>       JSON output path (default: docs/research/hip-1346-gcr/results.json)\n\
                       --workers <usize>     Parallel restore worker threads (default: 4)\n\
                       --fsync               Enable File::sync_all() on every timed chunk write"
                );
                return Ok(());
            }
            other => {
                return Err(format!("unknown argument: {other}").into());
            }
        }
    }

    let opts = BenchOptions {
        seed,
        iterations,
        repo_root,
        fsync_timed_runs,
        restore_workers,
        page_size_override: None,
    };

    let report = run_full_benchmark(&opts)?;

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = report.to_json_pretty();
    fs::write(&output_path, &json)?;

    println!("=== hanzo-gcr Benchmark Report (HIP-1346 / engine#91) ===");
    println!(
        "Seed: {} | Iterations: {} | Corpus: {} files ({} bytes, sha256={})",
        report.seed,
        report.iterations,
        report.corpus_files.len(),
        report.corpus_bytes,
        &report.corpus_sha256[..16]
    );
    println!();

    for w in &report.workloads {
        println!(
            "Workload: {} (page={}B, imm={}, mut={}, total={:.1} MiB, pauses={}, dirty={:.1}%{})",
            w.workload,
            w.page_size_bytes,
            w.immutable_pages,
            w.mutable_pages,
            w.total_vram_mb,
            w.pauses,
            w.mean_mutable_dirty_fraction * 100.0,
            if w.is_negative_regime {
                ", NEGATIVE REGIME"
            } else {
                ""
            }
        );
        println!(
            "  {:<30} {:>9} {:>9} {:>10} {:>10} {:>9} {:>8} {:>8}",
            "Strategy",
            "Pause p50",
            "Resum p50",
            "Write MB",
            "Store MB",
            "vsNaive",
            "vsMut",
            "Exact%"
        );
        for st in &w.strategies {
            println!(
                "  {:<30} {:>7.2}ms {:>7.2}ms {:>8.3}MB {:>8.3}MB {:>8.2}x {:>7.2}x {:>7.1}%",
                st.strategy,
                st.pause_latency.p50_ms,
                st.resume_latency.p50_ms,
                st.steady_state_mb_written_per_pause,
                st.storage_footprint_mb,
                st.reduction_vs_naive,
                st.reduction_vs_mutable_only,
                st.bit_exact_match_pct
            );
        }
        println!();
    }

    let sec = &report.security;
    println!(
        "Security & Correctness: bit_exact={:.1}% ({} restores) | corrupt_reject={:.1}% | trunc_reject={:.1}% | cycle_reject={:.1}% | missing_parent_reject={:.1}% | cross_tenant_reject={:.1}% | cross_tenant_leak={:.1}% | fsync_atomic_ok={}",
        sec.bit_exact_post_resume_pct,
        sec.total_restores_verified,
        sec.corrupted_chunk_rejection_pct,
        sec.truncated_chunk_rejection_pct,
        sec.cyclic_chain_rejection_pct,
        sec.missing_parent_rejection_pct,
        sec.cross_tenant_rejection_pct,
        sec.cross_tenant_leak_pct,
        sec.crash_safe_fsync_atomic_rename_ok
    );
    println!("Wrote JSON results to: {}", output_path.display());

    Ok(())
}
