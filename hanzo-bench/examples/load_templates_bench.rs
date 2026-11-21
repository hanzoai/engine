// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! Weight-loading IO templates demo (HIP-1345 §2.4, Bounty #14 /
//! engine#95, PPC/MAIO).
//!
//! Model: 8 GiB of weights in 512 pages of 16 MiB. Three checkpoint
//! layouts (contiguous / 4-shard strided / mmap-random) x two storage
//! scenarios (Gen4 NVMe 7 GB/s sequential, SATA 500 MB/s). Two policies
//! on the identical page set:
//!   A) demand load — synchronous serial NVMe->host->device, page by page.
//!   B) template pipeline — recorded access order replayed as a two-
//!      channel pipeline (NVMe producer stages into a bounded host RAM
//!      window while the PCIe consumer streams to device).
//! Plus a head pre-warm row: the first 128 pages of the learned order
//! staged during the idle window before serving begins, and the
//! break-even serving-session count for that window.
//!
//! Honest regime note: the pipeline only overlaps the two transfer
//! channels — when the storage tier dominates (SATA scenario), the win
//! collapses toward 1.0x. No figure here is a Hanzo expectation until
//! reproduced with measured device rates.
//!
//! Run: `cargo run --release --example load_templates_bench`

use hanzo_bench::load_templates::{
    break_even_turns, demand_load, head_prewarm, template_prewarm, LoadOutcome, LoadTemplate,
    LoaderConfig, WeightFormat, PAGE_BYTES,
};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const PAGES: u64 = 512; // 8 GiB / 16 MiB
const HEAD_PAGES: u64 = 128; // 2 GiB head pre-warm window

fn fmt_name(f: &WeightFormat) -> &'static str {
    match f {
        WeightFormat::Contiguous => "contiguous (safetensors-like)",
        WeightFormat::Sharded { shards: 4 } => "sharded x4 (strided)",
        WeightFormat::Sharded { .. } => "sharded (strided)",
        WeightFormat::MmapRandom => "mmap-order (random)",
    }
}

fn row(label: &str, o: &LoadOutcome) {
    println!(
        "  {label:<28} cold wall {: >9.2} s   prewarm window {: >7.2} s   window peak {: >5} MiB   occ peak {: >3} pages",
        o.cold_wall_ns as f64 / 1e9,
        o.prewarm_ns as f64 / 1e9,
        o.peak_staged_bytes / MIB,
        o.peak_lag_pages,
    );
}

fn main() {
    let gen4 = LoaderConfig::default(); // 7 GB/s seq NVMe, 12 GB/s PCIe, 2 GiB staging
    let sata = LoaderConfig {
        nvme_seq_bps: 500_000_000,
        device_pcie_bps: 8_000_000_000,
        ..LoaderConfig::default()
    };
    let link = LoaderConfig {
        nvme_seq_bps: 7_000_000_000,
        device_pcie_bps: 3_000_000_000, // PCIe3 x4-class link
        ..LoaderConfig::default()
    };
    for (label, cfg) in [
        ("scenario 1: Gen4 NVMe 7 GB/s, PCIe 12", gen4),
        ("scenario 2: SATA SSD 500 MB/s, PCIe 8", sata),
        ("scenario 3: NVMe 7 GB/s, PCIe 3 (link-bound)", link),
    ] {
        println!();
        println!("### {label}");
        println!(
            "model 8 GiB = {} pages x 16 MiB; head pre-warm = {} pages ({} GiB)",
            PAGES, HEAD_PAGES, HEAD_PAGES * PAGE_BYTES / GIB
        );
        println!("=====================================================================");
        for fmt in [
            WeightFormat::Contiguous,
            WeightFormat::Sharded { shards: 4 },
            WeightFormat::MmapRandom,
        ] {
            let t = LoadTemplate::learn(fmt, PAGES);
            println!("-- {}", fmt_name(&fmt));
            let d = demand_load(&cfg, &t);
            let tp = template_prewarm(&cfg, &t);
            let hp = head_prewarm(&cfg, &t, HEAD_PAGES);
            let be = break_even_turns(&d, &hp);
            row("A) demand (serial)", &d);
            row("B) template pipeline", &tp);
            row("C) head pre-warm (128p) + pipeline", &hp);
            println!(
                "     speedup B {:>5.2}x   C {:>5.2}x   head pre-warm break-even: {} serving sessions",
                d.cold_wall_ns as f64 / tp.cold_wall_ns.max(1) as f64,
                d.cold_wall_ns as f64 / hp.cold_wall_ns.max(1) as f64,
                be,
            );
        }
    }
    println!();
    println!("The template replays the recorded page order as a two-channel");
    println!("pipeline: NVMe producer stages into a bounded host RAM window");
    println!("while the PCIe consumer streams pages onto the device, so the");
    println!("slow tier's reads overlap the device link — PPC/MAIO's");
    println!("record-once/replay-pipelined mechanism at the template policy");
    println!("layer (HIP-1345 §2.4). Honest regime notes: (1) the pipeline");
    println!("buys OVERLAP, not bandwidth — when the storage tier dominates");
    println!("(SATA scenario) the win collapses toward 1.0x; (2) the staging");
    println!("window only binds when the device link is slower than the");
    println!("storage tier (scenario 3), where occupancy pins at the 2 GiB");
    println!("budget; (3) layout penalties (strided/random NVMe rates) dwarf");
    println!("the pipeline win for mmap-random formats. All rates are");
    println!("configured, not measured; engine integration must replace");
    println!("them with measured Hanzo-hardware rates.");
}
