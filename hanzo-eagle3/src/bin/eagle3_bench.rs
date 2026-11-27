#![forbid(unsafe_code)]
//! CLI runner for the HIP-1346 `hanzoai/engine#55` EAGLE-3 & workload-aware draft selector benchmark.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use hanzo_eagle3::bench::{format_report_json, run_full_benchmark, BenchEnvironment};
use hanzo_eagle3::corpus::{find_repo_root, GroundedCorpus};

fn cmd_stdout(prog: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(prog).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn detect_cpu_model() -> String {
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                if let Some((_, val)) = rest.split_once(':') {
                    let trimmed = val.trim();
                    if !trimmed.is_empty() {
                        return trimmed.to_string();
                    }
                }
            }
        }
    }
    std::env::consts::ARCH.to_string()
}

fn detect_gpu_driver() -> String {
    if let Some(out) = cmd_stdout(
        "nvidia-smi",
        &["--query-gpu=name,driver_version", "--format=csv,noheader"],
    ) {
        return out;
    }
    "CPU-only host (no local CUDA/ROCm/Metal GPU detected; 4B/9B/27B GPU timings are explicitly tagged H100-SXM5 roofline projections calibrated from CPU-measured tau)".to_string()
}

fn probe_environment() -> BenchEnvironment {
    let cpu_arch = std::env::consts::ARCH.to_string();
    let cpu_model = detect_cpu_model();
    let os_kernel =
        cmd_stdout("uname", &["-srm"]).unwrap_or_else(|| std::env::consts::OS.to_string());
    let gpu_driver = detect_gpu_driver();
    let git_commit = cmd_stdout("git", &["rev-parse", "--short=12", "HEAD"])
        .unwrap_or_else(|| "unknown".to_string());
    let rustc_version =
        cmd_stdout("rustc", &["--version"]).unwrap_or_else(|| "rustc unknown".to_string());

    BenchEnvironment {
        cpu_arch,
        cpu_model,
        os_kernel,
        gpu_driver,
        git_commit,
        rustc_version,
    }
}

fn main() {
    let mut seed: u64 = 1346;
    let mut output_path: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => {
                if let Some(v) = args.next() {
                    seed = v.parse::<u64>().expect("--seed expects an integer u64");
                }
            }
            "--output" => {
                if let Some(v) = args.next() {
                    output_path = Some(PathBuf::from(v));
                }
            }
            other => {
                eprintln!("Unknown argument: {other}");
                eprintln!("Usage: eagle3-bench [--seed <u64>] [--output <path>]");
                std::process::exit(2);
            }
        }
    }

    let repo_root = find_repo_root().expect("failed to locate repository root with corpus files");
    let corpus =
        GroundedCorpus::load_from_repo(&repo_root).expect("failed loading grounded corpus");
    let env = probe_environment();

    let report = run_full_benchmark(corpus, env, seed);
    let json = format_report_json(&report);

    println!("=== HIP-1346 Benchmark Summary (hanzoai/engine#55, seed={seed}) ===");
    println!(
        "Corpus: {} bytes (SHA-256 {})",
        report.corpus.total_bytes, report.corpus.combined_sha256
    );
    println!(
        "EAGLE-3 TTT Training ({} epochs, {} unroll): CE {:.4} -> {:.4} | step1 top1 {:.1}% -> {:.1}% | linear tau {:.3} -> {:.3} | tree tau {:.3} -> {:.3}",
        report.training_lift.training.epochs,
        report.training_lift.training.unroll_steps,
        report.training_lift.training.untrained_loss,
        report.training_lift.training.trained_loss,
        report.training_lift.training.untrained_top1_acc * 100.0,
        report.training_lift.training.trained_top1_acc * 100.0,
        report.training_lift.untrained_linear_tau,
        report.training_lift.trained_linear_tau,
        report.training_lift.untrained_tree_tau,
        report.training_lift.trained_tree_tau,
    );
    println!(
        "Stochastic Rejection Verification ({} trials): linear TV={:.5} (chi2={:.2}) | tree TV={:.5} (chi2={:.2}) | passed={}",
        report.stochastic_verification.trials,
        report.stochastic_verification.linear_tv_distance,
        report.stochastic_verification.linear_chi_square,
        report.stochastic_verification.tree_tv_distance,
        report.stochastic_verification.tree_chi_square,
        report.stochastic_verification.passed,
    );
    println!();
    println!(
        "{:<28} {:<24} {:>6} {:>7} {:>8} {:>7} {:>7}",
        "Workload", "Strategy", "Tau", "Acc%", "Wall(ms)", "Spd", "Exact%"
    );
    for w in &report.cpu_workloads {
        println!(
            "{:<28} {:<24} {:>6.3} {:>6.1}% {:>8.2} {:>6.2}x {:>6.1}%",
            w.workload,
            w.strategy,
            w.mean_tau,
            w.acceptance_rate_pct,
            w.cpu_wall_ms,
            w.cpu_speedup_vs_plain,
            w.output_bit_exact_pct
        );
    }

    println!();
    println!("--- Shared 4B/9B/27B 20-Turn Agent Scorecard (H100-SXM5 Modeled from CPU-Measured Tau) ---");
    for r in &report.matrix_scorecard {
        println!(
            "{:<10} {:<24} tau={:.2} p50={:>5.2}ms p95={:>5.2}ms wall={:>6.2}s ({:.2}x) E={:>7.1}mJ/t mem={:>5.2}GiB [{}]",
            r.model,
            r.strategy,
            r.mean_tau_cpu_measured,
            r.p50_itl_ms,
            r.p95_itl_ms,
            r.agent_loop_20_turn_s,
            r.speedup_vs_plain,
            r.energy_mj_per_tok,
            r.peak_memory_gib,
            r.gate_verdict
        );
    }

    if let Some(path) = output_path {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).expect("failed creating output parent directory");
            }
        }
        fs::write(&path, &json).expect("failed writing benchmark JSON receipt");
        println!("\nWrote benchmark JSON receipt to {}", path.display());
    }
}
