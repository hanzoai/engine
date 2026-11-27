// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reproducible HIP-1346 benchmark runner (`hanzoai/engine#51`) emitting
//! verified `results.json` and `README.md` scorecards.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use hanzo_prefixcache::bench::{
    render_readme_markdown, render_results_json, run_full_benchmark_suite, DEFAULT_BENCH_SEED,
};

fn parse_args() -> Result<(PathBuf, PathBuf, u64), String> {
    let mut repo_root = env::current_dir().map_err(|e| format!("failed to read cwd: {e}"))?;
    let mut out_dir = repo_root.join("docs/research/hip-1346-persistent-prefix-cache");
    let mut seed = DEFAULT_BENCH_SEED;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--repo-root" => {
                let val = args
                    .next()
                    .ok_or_else(|| "--repo-root requires a path".to_string())?;
                repo_root = PathBuf::from(val);
            }
            "--out-dir" => {
                let val = args
                    .next()
                    .ok_or_else(|| "--out-dir requires a path".to_string())?;
                out_dir = PathBuf::from(val);
            }
            "--seed" => {
                let val = args
                    .next()
                    .ok_or_else(|| "--seed requires an integer".to_string())?;
                seed = val
                    .parse::<u64>()
                    .map_err(|e| format!("invalid --seed '{val}': {e}"))?;
            }
            "--help" | "-h" => {
                return Err(
                    "Usage: prefixcache-bench [--repo-root PATH] [--out-dir PATH] [--seed N]"
                        .to_string(),
                );
            }
            other => {
                return Err(format!("unknown argument: {other}"));
            }
        }
    }
    Ok((repo_root, out_dir, seed))
}

fn main() -> ExitCode {
    let (repo_root, out_dir, seed) = match parse_args() {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };

    let report = match run_full_benchmark_suite(&repo_root, seed) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("benchmark failed: {err}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(err) = fs::create_dir_all(&out_dir) {
        eprintln!(
            "failed to create output directory {}: {err}",
            out_dir.display()
        );
        return ExitCode::FAILURE;
    }

    let json_text = render_results_json(&report);
    let md_text = render_readme_markdown(&report);

    let json_path = out_dir.join("results.json");
    let md_path = out_dir.join("README.md");

    if let Err(err) = fs::write(&json_path, &json_text) {
        eprintln!("failed to write {}: {err}", json_path.display());
        return ExitCode::FAILURE;
    }
    if let Err(err) = fs::write(&md_path, &md_text) {
        eprintln!("failed to write {}: {err}", md_path.display());
        return ExitCode::FAILURE;
    }

    println!(
        "Wrote verified HIP-1346 artifacts:\n  - {}\n  - {}",
        md_path.display(),
        json_path.display()
    );
    println!(
        "Tenant isolation audit: {}/{} rejected, {} leaks",
        report.isolation_audit.total_probes - report.isolation_audit.cross_tenant_leaks,
        report.isolation_audit.total_probes,
        report.isolation_audit.cross_tenant_leaks
    );
    ExitCode::SUCCESS
}
