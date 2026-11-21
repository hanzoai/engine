// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! Sparse SSD KV offload demo at the fabric level
//! (HIP-1345 §2.1, Bounty #11 / engine#92, SolidAttention).
//!
//! Workload: 128k-token local Zen context, ~32 KiB KV per token (AIPC-class
//! model: 16 layers x 4 KV heads x 128 dim x K+V x fp16) = 4 GiB of KV in
//! 64 blocks of 64 MiB; 6 blocks resident on device, the rest SSD-backed.
//! 96 decode steps of sparse attention: sliding window of 3 blocks (head-2
//! .. head), head advances every 8 steps, attention-sink block 0 re-read
//! every 16 steps. Three policies on the identical trace:
//!   A) demand fetch — no prefetch, synchronous SSD reads.
//!   B) speculative — temporal-locality predictor stages likely blocks
//!      ahead of use; prefetch I/O overlaps decode compute via a
//!      pending-I/O queue drained by each step's compute.
//!   C) next-step oracle — perfect next-step prediction, issued one step
//!      before use (depth 1; not a wall-clock upper bound — see below).
//! Per step: wall = compute + stall, where stall is the pending I/O
//! backlog exceeding the step's compute (per-block completion model).
//! Policy-layer only: the access trace is an input — which blocks sparse
//! attention reads is the engine's job. Honest note: the depth-1 oracle is
//! NOT a wall-clock upper bound in this regime; it is the no-lookahead
//! limit. Issued one step before use, a full SSD block read cannot hide
//! under one decode step, so it stalls — prefetch DEPTH dominates hiding,
//! which is why the recency predictor (implicit depth 8) beats it.
//!
//! Run: `cargo run --release --example solid_attention_bench`

use hanzo_kv_solidattn::{predict_next, salted_block_key, OffloadConfig, SsdOffloadPool};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const TOTAL_BLOCKS: u64 = 64; // 128k tokens * 32 KiB/token = 4 GiB
const RESIDENT_BLOCKS: u64 = 6;
const STEPS: usize = 96;
const HEAD_ADVANCE_EVERY: usize = 8; // head block advances every N steps
const SINK_EVERY: usize = 16; // attention-sink re-read cadence
const TENANT_SALT: u64 = 7;

/// Deterministic sparse-attention access trace: per step, the block indices
/// the attention path touches (head window + periodic sink).
fn access_trace() -> Vec<Vec<u64>> {
    let mut trace = Vec::with_capacity(STEPS);
    let mut head: u64 = 0;
    for i in 0..STEPS {
        if i > 0 && i % HEAD_ADVANCE_EVERY == 0 {
            head = (head + 1).min(TOTAL_BLOCKS - 1);
        }
        let mut blocks: Vec<u64> = vec![head];
        if head >= 1 {
            blocks.push(head - 1);
        }
        if head >= 2 {
            blocks.push(head - 2);
        }
        if i % SINK_EVERY == 0 && blocks[0] != 0 {
            blocks.push(0); // attention sink
        }
        blocks.sort_unstable();
        blocks.dedup();
        trace.push(blocks);
    }
    trace
}

fn resident_keys() -> Vec<u64> {
    (0..RESIDENT_BLOCKS).map(|b| salted_block_key(TENANT_SALT, b)).collect()
}

/// One simulated run of a policy over the trace. Returns (wall_ns,
/// per-step stalls, pool stats snapshot fields).
fn run_policy(
    cfg: &OffloadConfig,
    trace: &[Vec<u64>],
    policy: Policy,
) -> (u128, Vec<u128>, u64, u64, u64, u64) {
    let mut pool = SsdOffloadPool::new(*cfg, resident_keys());
    // Per-block completion model: a prefetch issued at step j for a block
    // used at step i hides its I/O under the compute of steps j..i-1;
    // residual = io_ns - slack only stalls at the use step. Demand fetches
    // are synchronous: their I/O stalls the use step in full.
    struct Inflight {
        key: u64,
        io_ns: u64,
        cum_at_issue: u128,
    }
    let mut inflight: Vec<Inflight> = Vec::new();
    let mut cum_compute: u128 = 0;
    let mut wall: u128 = 0;
    let mut stalls: Vec<u128> = Vec::with_capacity(trace.len());
    let recent_cap = 8;
    let mut recent: Vec<u64> = Vec::new();
    for (i, blocks) in trace.iter().enumerate() {
        let compute: u64 = blocks.len() as u64 * cfg.decode_ns_per_block;
        let mut stall: u128 = 0;
        match policy {
            Policy::Demand => {
                for &b in blocks {
                    stall += pool.fetch(salted_block_key(TENANT_SALT, b), i as u64) as u128;
                }
            }
            Policy::Speculative | Policy::Oracle => {
                let predicted: Vec<u64> = match policy {
                    Policy::Oracle => trace.get(i + 1).cloned().unwrap_or_default(),
                    _ => predict_next(&recent, 3),
                };
                for b in predicted {
                    let io = pool.prefetch(salted_block_key(TENANT_SALT, b), i as u64);
                    if io > 0 {
                        inflight.push(Inflight { key: salted_block_key(TENANT_SALT, b), io_ns: io, cum_at_issue: cum_compute });
                    }
                }
                for &b in blocks {
                    let key = salted_block_key(TENANT_SALT, b);
                    if let Some(pos) = inflight.iter().position(|f| f.key == key) {
                        // prefetched earlier: how much compute ran since?
                        let slack = cum_compute - inflight[pos].cum_at_issue;
                        stall += (inflight[pos].io_ns as u128).saturating_sub(slack);
                        inflight.swap_remove(pos);
                    } else {
                        // predictor miss: synchronous SSD read stalls in full
                        stall += pool.fetch(key, i as u64) as u128;
                    }
                }
            }
        }
        stalls.push(stall);
        wall += compute as u128 + stall;
        cum_compute += compute as u128;
        // maintain recency window for the predictor
        for &b in blocks {
            if !recent.contains(&b) {
                recent.push(b);
            }
        }
        while recent.len() > recent_cap {
            recent.remove(0);
        }
    }
    let _ = pool.unused_staged_bytes(); // end-of-run waste visible in stats below
    let s = &pool.stats;
    (
        wall,
        stalls,
        s.ssd_misses,
        s.prefetch_issued,
        s.wasted_prefetch_bytes + pool.unused_staged_bytes(),
        s.peak_staging_bytes,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    Demand,
    Speculative,
    Oracle,
}

fn report(name: &str, cfg: &OffloadConfig, trace: &[Vec<u64>], policy: Policy) -> u128 {
    let (wall, stalls, misses, prefetches, wasted, peak) = run_policy(cfg, trace, policy);
    let mut sorted = stalls.clone();
    sorted.sort_unstable();
    let p99 = if sorted.is_empty() {
        0
    } else {
        sorted[((99 * sorted.len()) / 100).min(sorted.len() - 1)]
    };
    let total_stall: u128 = stalls.iter().sum();
    println!("{name}:");
    println!(
        "  wall {: >10.1} ms   throughput {: >6.1} steps/s   p99 stall {: >8.1} ms   total stall {: >8.1} ms",
        wall as f64 / 1e6,
        trace.len() as f64 * 1e9 / wall.max(1) as f64,
        p99 as f64 / 1e6,
        total_stall as f64 / 1e6,
    );
    println!(
        "  ssd misses {}   prefetches {}   wasted prefetch {} MiB   peak staging {} MiB",
        misses,
        prefetches,
        wasted / MIB,
        peak / MIB,
    );
    wall
}

fn main() {
    let fast = OffloadConfig::default(); // 2 GB/s SSD
    let slow = OffloadConfig { ssd_read_bps: 500_000_000, ..OffloadConfig::default() }; // SATA-class SSD
    for (label, cfg) in [("scenario 1: 2 GB/s NVMe SSD", fast), ("scenario 2: 500 MB/s SATA SSD", slow)] {
        println!();
        println!("### {label}");
        let trace = access_trace();
        println!(
            "Sparse SSD KV offload demo: 128k-token context, {} blocks x 64 MiB (4 GiB KV), {} resident; {} decode steps, window 3, sink/{} steps",
            TOTAL_BLOCKS, RESIDENT_BLOCKS, STEPS, SINK_EVERY
        );
        println!("=====================================================================");

        let demand = report("A) demand fetch (no prefetch)", &cfg, &trace, Policy::Demand);
        let spec = report("B) speculative prefetch (predictor)", &cfg, &trace, Policy::Speculative);
        let oracle = report("C) next-step oracle (perfect prediction, depth 1)", &cfg, &trace, Policy::Oracle);

        println!();
        println!(
            "wall: demand {:.1} ms -> speculative {:.1} ms ({:.2}x) -> depth-1 oracle {:.1} ms ({:.2}x)",
            demand as f64 / 1e6,
            spec as f64 / 1e6,
            demand as f64 / spec.max(1) as f64,
            oracle as f64 / 1e6,
            demand as f64 / oracle.max(1) as f64,
        );
    }
    println!();
    println!("The predictor stages likely blocks several steps ahead of use,");
    println!("so SSD reads hide under attention compute instead of stalling it");
    println!("— SolidAttention legs 1+2 (coarse blocks + speculative prefetch)");
    println!("with a per-block completion model, at the fabric policy layer");
    println!("(HIP-1345 §2.1). Honest finding: the depth-1 oracle (perfect");
    println!("next-step prediction issued one step before use) still stalls — a");
    println!("134 ms SATA read cannot hide under one 18 ms step. PREFETCH DEPTH,");
    println!("not prediction accuracy, dominates hiding in this regime; the");
    println!("recency predictor wins by issuing at epoch start (implicit depth");
    println!("8). Sparse attention algorithms and accuracy effects are the");
    println!("engine's job — out of scope here.");
}
