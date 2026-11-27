//! Reproducible benchmark CLI (`diloco-bench`) for Hanzo Engine permissionless
//! internet-scale elastic low-communication LoRA training (HIP-1350).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::path::PathBuf;

use hanzo_diloco_train::bench::run_full_benchmark;
use hanzo_diloco_train::corpus::GroundedCorpus;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut seed: u64 = 1350;
    let mut output_dir: Option<PathBuf> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                if let Some(val) = args.get(i + 1) {
                    seed = val.parse::<u64>().unwrap_or_else(|_| {
                        eprintln!("invalid --seed value: {val}");
                        std::process::exit(2);
                    });
                    i += 2;
                } else {
                    eprintln!("--seed requires a u64 argument");
                    std::process::exit(2);
                }
            }
            "--output-dir" => {
                if let Some(val) = args.get(i + 1) {
                    output_dir = Some(PathBuf::from(val));
                    i += 2;
                } else {
                    eprintln!("--output-dir requires a path argument");
                    std::process::exit(2);
                }
            }
            "--help" | "-h" => {
                println!(
                    "Usage: diloco-bench [--seed <u64>] [--output-dir <path>]\n\
                     Runs the reproducible Hanzo Engine 3-node live TCP elastic LoRA training\n\
                     benchmark, churn/recovery test, 6/6 Byzantine defense suite, and negative regimes."
                );
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let repo_root = GroundedCorpus::discover_repo_root();
    match run_full_benchmark(seed, &repo_root, output_dir.as_deref()) {
        Ok(d) => {
            println!(
                "=== Hanzo Engine Elastic Low-Communication LoRA Training Benchmark (HIP-1350) ==="
            );
            println!(
                "Seed: {} | Grounded Corpus SHA-256: {} ({} bytes)",
                d.seed, d.corpus_sha256, d.corpus_raw_bytes
            );
            println!(
                "Centralized LAN (H=1, FP32):  loss {:.4} -> {:.4} (ppl {:.2} -> {:.2}) | wire {} B | eff_tok/s {:.1}",
                d.centralized_baseline.initial_eval.loss,
                d.centralized_baseline.final_eval.loss,
                d.centralized_baseline.initial_eval.perplexity,
                d.centralized_baseline.final_eval.perplexity,
                d.centralized_baseline.total_wire_bytes,
                d.centralized_baseline.effective_wan_tokens_per_sec,
            );
            println!(
                "3-Node WAN Steady (H=16, INT8): loss {:.4} -> {:.4} (ppl {:.2} -> {:.2}) | wire {} B ({:.1}x less) | delta upload {:.1}x less | eff_tok/s {:.1}",
                d.wan_3node_steady.initial_eval.loss,
                d.wan_3node_steady.final_eval.loss,
                d.wan_3node_steady.initial_eval.perplexity,
                d.wan_3node_steady.final_eval.perplexity,
                d.wan_3node_steady.total_wire_bytes,
                d.wire_bytes_reduction_factor,
                d.delta_upload_reduction_factor,
                d.wan_3node_steady.effective_wan_tokens_per_sec,
            );
            println!(
                "3-Node WAN Churn+Recovery:      loss {:.4} -> {:.4} (ppl {:.2} -> {:.2}) | completion {:.0}% | wire {} B",
                d.wan_3node_churn_recovery.initial_eval.loss,
                d.wan_3node_churn_recovery.final_eval.loss,
                d.wan_3node_churn_recovery.initial_eval.perplexity,
                d.wan_3node_churn_recovery.final_eval.perplexity,
                d.wan_3node_churn_recovery.completion_rate * 100.0,
                d.wan_3node_churn_recovery.total_wire_bytes,
            );
            println!(
                "Byzantine & Integrity Suite:    {}/{} attacks detected & rejected (tier: {})",
                d.adversarial_suite.detected_count,
                d.adversarial_suite.total_attacks,
                d.adversarial_suite.trust_tier.as_str(),
            );
            for n in &d.negative_regimes {
                println!(
                    "Negative Regime [{}]: {} -> {}",
                    n.verdict, n.regime_id, n.failure_summary
                );
            }
            if let Some(dir) = output_dir {
                println!(
                    "Wrote deterministic results.json + results.json.sha256 to {}",
                    dir.display()
                );
            }
        }
        Err(e) => {
            eprintln!("benchmark failed: {e}");
            std::process::exit(1);
        }
    }
}
