//! Reproducible HIP-1346 `#57` benchmark CLI (`hetero-bench`).
//!
//! Usage:
//! ```bash
//! cargo run -p hanzo-hetero-mem --release --offline --bin hetero-bench -- \
//!   --seed 1346 --output-dir docs/research/hip-1346-heterogeneous-memory-offload
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::env;
use std::path::PathBuf;
use std::process;

use hanzo_hetero_mem::bench::{run_full_benchmark, write_results_json};
use hanzo_hetero_mem::model::GIB;

fn main() {
    let mut seed: u64 = 1346;
    let mut output_dir = PathBuf::from("docs/research/hip-1346-heterogeneous-memory-offload");

    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --seed requires a u64 argument");
                    process::exit(2);
                }
                seed = args[i].parse().unwrap_or_else(|e| {
                    eprintln!("error: invalid --seed '{}': {e}", args[i]);
                    process::exit(2);
                });
            }
            "--output-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --output-dir requires a path argument");
                    process::exit(2);
                }
                output_dir = PathBuf::from(&args[i]);
            }
            "--help" | "-h" => {
                println!(
                    "hetero-bench [--seed <u64>] [--output-dir <path>]\n\
                     Default: --seed 1346 --output-dir docs/research/hip-1346-heterogeneous-memory-offload"
                );
                return;
            }
            other => {
                eprintln!("error: unknown argument '{other}'");
                process::exit(2);
            }
        }
        i += 1;
    }

    let bundle = run_full_benchmark(seed).unwrap_or_else(|e| {
        eprintln!("benchmark failed: {e}");
        process::exit(1);
    });

    write_results_json(&bundle, &output_dir).unwrap_or_else(|e| {
        eprintln!("failed to write results.json: {e}");
        process::exit(1);
    });

    println!("=== HIP-1346 #57 Heterogeneous Memory & Cold-Start Benchmark ===");
    println!("Seed: {}", bundle.seed);
    println!(
        "Corpus: {} bytes, SHA-256 {}",
        bundle.corpus.total_bytes, bundle.corpus.combined_sha256_hex
    );
    println!(
        "Grounded Skew: Gini={:.4}, top-20% coverage={:.2}%",
        bundle.corpus_skew.gini,
        bundle.corpus_skew.top_20_pct_coverage * 100.0
    );
    println!(
        "Deterministic Receipt SHA-256: {}",
        bundle.deterministic_receipt_sha256
    );
    println!(
        "\n--- Directly Measured Host On-Disk WeightFile I/O ({:.2} MiB, SHA-256 verified={}) ---",
        bundle.host_io.file_bytes as f64 / (1024.0 * 1024.0),
        bundle.host_io.sha256_verified
    );
    for m in &bundle.host_io.modes {
        println!(
            "  {:<26} cold={:>6.3}ms warm={:>6.3}ms stream={:>6.2}ms bw={:>7.1}MiB/s cold_maj={:<5} tot_maj={:<5} min={:<5} pf_hit={:>5.1}% copy={:>8}B",
            m.mode_label,
            m.cold_first_tile_ms,
            m.warm_first_tile_ms,
            m.full_stream_ms,
            m.sustained_decode_mibs,
            m.cold_start_major_faults,
            m.total_major_faults,
            m.total_minor_faults,
            m.prefetch_hit_rate * 100.0,
            m.copy_bytes
        );
    }

    println!(
        "\n--- Directly Measured CPU Execution Equivalence Verifier (all_passed={}, wall={:.2}ms) ---",
        bundle.execution_verify.all_passed(),
        bundle.execution_verify.cpu_wall_time_ms
    );
    for r in &bundle.execution_verify.regimes {
        println!(
            "  {:<28} tokens={}/{} max_abs_err={:.3e} reg={:.3}pp copy={}B zc_ptr={}",
            r.regime,
            r.matched_tokens,
            r.total_tokens,
            r.max_abs_err,
            r.quality_regression_pp,
            r.weight_copy_bytes,
            r.zero_copy_ptr_verified
        );
    }

    println!("\n--- Key DiscreteGpu24G & UnifiedMemory Highlight (Analytical + Exact Memory) ---");
    for sc in &bundle.matrix_scorecards {
        if (sc.hardware_id == "DiscreteGpu24G" || sc.hardware_id == "UnifiedMemory24G")
            && (sc.plan.regime_name == "eager_full_read_naive_offload"
                || sc.plan.regime_name == "layer_split_static_offload"
                || sc.plan.regime_name == "powerinfer_hot_cold_hybrid"
                || sc.plan.regime_name == "unified_zero_copy_view"
                || sc.plan.regime_name == "adaptive_hetero_planner")
        {
            println!(
                "  {:<10} {:<16} {:<30} cold={:>8.1}ms itl50={:>6.2}ms wall20={:>7.2}s vram={:>6.2}G host={:>6.2}G copy={:>5.2}G mJ={:>7.1} gate={}",
                sc.model_label,
                sc.hardware_id,
                sc.plan.regime_name,
                sc.cold_ttft_ms,
                sc.itl_p50_ms,
                sc.agent_20_turn_wall_s,
                sc.plan.peak_vram_bytes as f64 / GIB as f64,
                sc.plan.peak_host_ram_bytes as f64 / GIB as f64,
                sc.copy_bytes_gib,
                sc.energy_mj_per_tok,
                if sc.gate_accepted { "ACCEPTED" } else { "REJECTED" }
            );
        }
    }

    println!("\n--- Explicit Honest Negative Regimes ---");
    for neg in &bundle.negative_regimes {
        println!(
            "  {:<42} wall_delta={:+.1}% energy_delta={:+.1}% mem_delta={:+.1}% gate={} ({})",
            neg.kind.as_str(),
            neg.wall_time_delta_pct,
            neg.energy_delta_pct,
            neg.peak_mem_delta_pct,
            if neg.negative.gate_accepted {
                "ACCEPTED"
            } else {
                "REJECTED"
            },
            neg.negative.gate_reason
        );
    }
    println!("\nWrote {}", output_dir.join("results.json").display());
}
