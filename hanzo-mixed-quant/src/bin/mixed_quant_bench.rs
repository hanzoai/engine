// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! CLI benchmark runner for `hanzo-mixed-quant` (`hanzoai/engine#49`, HIP-1346).
//!
//! Usage:
//! ```bash
//! cargo run -p hanzo-mixed-quant --release --offline --bin mixed-quant-bench -- --seed 1346
//! cargo run -p hanzo-mixed-quant --release --offline --bin mixed-quant-bench -- --seed 1346 --output docs/research/hip-1346-mixed-precision-quant/results.json
//! ```

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use hanzo_mixed_quant::bench::run_full_benchmark;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut seed: u64 = 1346;
    let mut output_path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --seed requires a u64 argument");
                    return ExitCode::FAILURE;
                }
                match args[i].parse::<u64>() {
                    Ok(v) => seed = v,
                    Err(e) => {
                        eprintln!("error: invalid --seed '{}': {e}", args[i]);
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --output requires a file path argument");
                    return ExitCode::FAILURE;
                }
                output_path = Some(PathBuf::from(&args[i]));
            }
            "--help" | "-h" => {
                println!(
                    "mixed-quant-bench [--seed <u64>] [--output <path/to/results.json>]\n\
                     Reproducible HIP-1346 mixed-precision quantization benchmark (hanzoai/engine#49)."
                );
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("error: unknown argument '{other}'");
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    match run_full_benchmark(seed, None) {
        Ok(bundle) => {
            println!("{}", bundle.format_summary_table());
            if let Some(path) = output_path {
                if let Some(parent) = path.parent() {
                    if !parent.as_os_str().is_empty() {
                        if let Err(e) = fs::create_dir_all(parent) {
                            eprintln!("error: failed to create parent directory: {e}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
                let json = bundle.to_json();
                if let Err(e) = fs::write(&path, json) {
                    eprintln!("error: failed to write '{}': {e}", path.display());
                    return ExitCode::FAILURE;
                }
                println!("Wrote benchmark receipt JSON to {}", path.display());
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("benchmark failed: {e}");
            ExitCode::FAILURE
        }
    }
}
