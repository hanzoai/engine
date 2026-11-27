// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! CLI binary for running the HIP-1346 `#53` hardware-specialised attention, shared-prefix
//! cascade LSE merge, fused ragged prefill/decode, hybrid Amdahl verification, and
//! per-architecture (`Hopper H100`, `Blackwell B200`, `GB10`, `Consumer RTX 4090`)
//! 20-turn coding-agent serving scorecard.
//!
//! Usage:
//! ```text
//! cargo run -p hanzo-cascade-attn --release --offline --bin cascade-attn-bench -- \
//!     --seed 1346 --output-dir docs/research/hip-1346-hardware-specialised-attention
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use hanzo_cascade_attn::arch::HardwareArch;
use hanzo_cascade_attn::bench::{run_full_benchmark, AttentionBackend};
use hanzo_cascade_attn::sha256::sha256_hex;

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

    println!("=== Hanzo Engine HIP-1346 (#53) Hardware-Specialised Attention & Cascade LSE ===");
    println!("Seed: {} | Commit: {}", report.seed, report.git_commit);
    println!(
        "[cpu_directly_measured] 2-level max_err={:.3e} | 3-level max_err={:.3e} (kv_red={:.2}x) | argmax={:.2}% | reg={:.3}pp",
        report.cpu_cascade.max_two_level_output_err,
        report.cpu_cascade.three_level_max_output_err,
        report.cpu_cascade.three_level_kv_read_reduction,
        report.cpu_cascade.aggregate_argmax_match_pct,
        report.cpu_cascade.quality_regression_pp
    );
    for row in &report.cpu_cascade.rows {
        println!(
            "  B={:2} pref={:3} suff={:3} | err={:.2e} | kv_red={:.2}x | cpu={:.1}->{:.1}us ({:.2}x)",
            row.batch_size,
            row.shared_prefix_len,
            row.unique_suffix_len,
            row.max_abs_output_err,
            row.kv_read_reduction_ratio,
            row.unshared_cpu_us,
            row.cascade_cpu_us,
            row.cpu_wall_speedup
        );
    }
    println!(
        "[cpu_directly_measured] ragged: pref={} dec={} q_toks={} stages={}->{} err={:.2e} argmax={:.2}%",
        report.cpu_ragged.prefill_requests,
        report.cpu_ragged.decode_requests,
        report.cpu_ragged.total_q_tokens,
        report.cpu_ragged.separate_dispatch_stages,
        report.cpu_ragged.fused_dispatch_stages,
        report.cpu_ragged.max_abs_output_err,
        report.cpu_ragged.greedy_argmax_match_pct
    );
    println!(
        "[cpu_directly_measured] hybrid Amdahl ({}A+{}G): attn={:.1}% gdn={:.1}% mlp={:.1}% | iso_attn={:.2}x full_step={:.2}x | err={:.2e}",
        report.cpu_hybrid.num_attn_layers,
        report.cpu_hybrid.num_gdn_layers,
        report.cpu_hybrid.attn_time_fraction * 100.0,
        report.cpu_hybrid.gdn_time_fraction * 100.0,
        report.cpu_hybrid.mlp_time_fraction * 100.0,
        report.cpu_hybrid.isolated_attn_speedup,
        report.cpu_hybrid.full_step_speedup,
        report.cpu_hybrid.max_abs_hidden_err
    );

    println!("\n--- Per-Architecture Adaptive Hybrid Dispatch Highlights (9B-Hybrid INT4-AWQ, B=8, 20-turn) ---");
    for arch in HardwareArch::all() {
        if let Some(row) = report.scorecards.iter().find(|r| {
            r.hardware_arch == arch.id()
                && r.model == "9B-Hybrid"
                && r.quant_format == "INT4-AWQ"
                && r.batch_size == 8
                && r.backend == AttentionBackend::AdaptiveHybridDispatch.as_str()
        }) {
            println!(
                "  {:28} | wall={:6.2}s ({:.2}x) | cold/warm TTFT={:.1}/{:.1}ms | p50/p95={:.2}/{:.2}ms | VRAM={:.0}/{:.0}MiB | {:.1}mJ/tok | {}",
                row.hardware_display,
                row.agent_loop_wall_time_s,
                row.wall_speedup_vs_baseline,
                row.cold_ttft_ms,
                row.warm_ttft_ms,
                row.p50_itl_ms,
                row.p95_itl_ms,
                row.peak_vram_mib,
                row.vram_capacity_mib,
                row.energy_mj_per_token,
                row.gate.status_str()
            );
        }
    }

    println!("\n--- Explicit Honest Negative Regimes (All Must Be REJECTED) ---");
    for neg in &report.negative_regimes {
        println!(
            "  {:38} | gate={} | wall={:.2}s (base={:.2}s, {:.2}x) | iso_attn={:.2}x | VRAM={:.0}/{:.0}MiB",
            neg.regime,
            neg.scorecard.gate.status_str(),
            neg.scorecard.agent_loop_wall_time_s,
            neg.baseline_wall_time_s,
            neg.scorecard.wall_speedup_vs_baseline,
            neg.scorecard.isolated_attn_kernel_speedup,
            neg.scorecard.peak_vram_mib,
            neg.scorecard.vram_capacity_mib
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
