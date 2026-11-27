//! CLI runner for the reproducible HIP-1350 permissionless GPU provider daemon benchmark.

#![forbid(unsafe_code)]

use std::env;
use std::path::PathBuf;
use std::process;

use hanzo_provider_daemon::bench::{run_full_benchmark, BenchConfig};

fn main() {
    let mut seed: u64 = 1350;
    let mut jobs_per_provider: usize = 20;
    let mut repo_root = PathBuf::from(".");
    let mut output_path = Some(PathBuf::from(
        "docs/research/hip-1350-provider-daemon/results.json",
    ));

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("invalid --seed value: {}", args[i + 1]);
                    process::exit(2);
                });
                i += 2;
            }
            "--jobs-per-provider" if i + 1 < args.len() => {
                jobs_per_provider = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("invalid --jobs-per-provider value: {}", args[i + 1]);
                    process::exit(2);
                });
                i += 2;
            }
            "--repo-root" if i + 1 < args.len() => {
                repo_root = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--output" if i + 1 < args.len() => {
                output_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--no-output" => {
                output_path = None;
                i += 1;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: provider-daemon-bench [--seed N] [--jobs-per-provider N] [--repo-root PATH] [--output PATH | --no-output]"
                );
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                process::exit(2);
            }
        }
    }

    let cfg = BenchConfig {
        repo_root,
        seed,
        jobs_per_provider,
        output_path: output_path.clone(),
    };

    match run_full_benchmark(&cfg) {
        Ok(report) => {
            println!("{}", report.to_pretty_string());
            if let Some(path) = output_path {
                eprintln!("Wrote HIP-1350 raw JSON receipt to {}", path.display());
            }
        }
        Err(err) => {
            eprintln!("provider-daemon-bench failed: {err}");
            process::exit(1);
        }
    }
}
