// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Weight-loading IO templates for multi-tier model loads (HIP-1345 §2.4,
//! Bounty #14 / engine#95, PPC/MAIO
//! <https://www.usenix.org/conference/fast26/presentation/liu-yubo>).
//!
//! Cold-loading an 8 GiB model serially (NVMe -> host -> device, page by
//! page) pays the full sum of both transfers. PPC/MAIO's mechanism is to
//! record the page-level access sequence of a model load once, turn it
//! into a **template**, and replay it as a two-channel pipeline — an NVMe
//! producer staging pages into host RAM while a PCIe consumer streams them
//! onto the device — so the slow tier's reads overlap the device link.
//!
//! This module is the *template + policy* layer:
//!
//! - [`PAGE_BYTES`]: the weight-page granularity templates are recorded at.
//! - [`WeightFormat`]: the checkpoint layouts and their canonical page
//!   access orders (contiguous safetensors-like, sharded-strided, and a
//!   deterministic mmap-random permutation), plus the effective NVMe
//!   read bandwidth each layout incurs (sequential reads run at full
//!   device bandwidth; strided and random layouts pay seek/queueing
//!   penalties — the honest "cold-load latency across weight formats"
//!   axis).
//! - [`LoadTemplate`]: a recorded access order (what a recording run
//!   captures) + the derived page timeline.
//! - [`demand_load`] vs [`template_prewarm`]: the matched policies —
//!   synchronous serial load vs the two-channel pipelined replay with a
//!   bounded host staging window (producer may only run `staging_bytes`
//!   ahead of the consumer).
//! - [`head_prewarm`]: pre-warm the first K pages of the learned order
//!   during the idle window before serving begins; the rest streams.
//!
//! What is deliberately NOT here: the actual IO engine (io_uring / CUDA
//! host staging), tensor-level placement, quantization or mmap syscalls,
//! and any learned predictor beyond replaying a recorded order. Engine
//! integration must replace the configured bandwidths with measured
//! Hanzo-hardware rates; see the PR's HIP-1345 §2.4 update.

/// Weight-page granularity: 16 MiB pages (a middle ground between 4 KiB
/// page-cache pages and whole-tensor reads).
pub const PAGE_BYTES: u64 = 16 * 1024 * 1024;

/// SplitMix64 — stdlib-only deterministic mixing for the random layout's
/// permutation (no RNG dependency, reproducible across runs and hosts).
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_mul(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Checkpoint layout families and the NVMe read bandwidth each incurs.
///
/// The bandwidth axis is the honest "cold-load latency across weight
/// formats" effect: page-granular reads over a modern NVMe device run at
/// full sequential bandwidth only when the access order is sequential;
/// strided (sharded) and random (mmap-order) layouts pay seek/queueing
/// penalties. These are configured rates, not measured device numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// One file, tensors in file order — sequential page reads.
    Contiguous,
    /// N shard files read round-robin per layer — strided reads.
    Sharded { shards: u64 },
    /// mmap-order access (GGUF-style) — scattered reads, deterministic
    /// permutation in this model.
    MmapRandom,
}

impl WeightFormat {
    /// Canonical page access order for a model of `pages` pages. Each
    /// order is a permutation of `0..pages` (every page exactly once).
    pub fn access_order(&self, pages: u64) -> Vec<u64> {
        match self {
            WeightFormat::Contiguous => (0..pages).collect(),
            WeightFormat::Sharded { shards } => {
                let k = (*shards).max(1);
                let per = pages.div_ceil(k);
                let mut order = Vec::with_capacity(pages as usize);
                for i in 0..pages {
                    order.push((i % k) * per + i / k);
                }
                order
            }
            WeightFormat::MmapRandom => {
                let mut keyed: Vec<(u64, u64)> = (0..pages).map(|p| (splitmix64(p ^ 0xA5A5), p)).collect();
                keyed.sort_unstable();
                keyed.into_iter().map(|(_, p)| p).collect()
            }
        }
    }

    /// Effective NVMe read bandwidth for this layout (bytes/sec), given the
    /// device's sequential bandwidth. Sequential = full; sharded strided =
    /// 1/2; random = 1/4 (16 MiB reads are large, so the penalty is
    /// moderate — configured, not measured).
    pub fn effective_nvme_bps(&self, cfg: &LoaderConfig) -> u64 {
        match self {
            WeightFormat::Contiguous => cfg.nvme_seq_bps,
            WeightFormat::Sharded { .. } => cfg.nvme_seq_bps / 2,
            WeightFormat::MmapRandom => cfg.nvme_seq_bps / 4,
        }
    }
}

/// Loader configuration: tier bandwidths and the host staging window.
#[derive(Debug, Clone, Copy)]
pub struct LoaderConfig {
    /// NVMe sequential read bandwidth, bytes per second.
    pub nvme_seq_bps: u64,
    /// PCIe host->device copy bandwidth, bytes per second.
    pub device_pcie_bps: u64,
    /// Host RAM staging window the producer may fill ahead of the
    /// consumer, bytes.
    pub staging_bytes: u64,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        // A desktop-class serving host: Gen4 NVMe ~7 GB/s sequential,
        // PCIe4 x16 ~12 GB/s effective device link, 2 GiB staging window.
        Self {
            nvme_seq_bps: 7_000_000_000,
            device_pcie_bps: 12_000_000_000,
            staging_bytes: 2 * 1024 * 1024 * 1024,
        }
    }
}

/// A recorded weight-loading access template (what a recording run of one
/// cold load captures): the page order plus the page size it was
/// recorded at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadTemplate {
    /// Checkpoint layout the template was recorded for.
    pub format: WeightFormat,
    /// Page access order (a permutation of 0..page_count).
    pub order: Vec<u64>,
    /// Page size at recording time (bytes).
    pub page_bytes: u64,
}

impl LoadTemplate {
    /// Record a template from a format's canonical access order. In
    /// production this vector comes from a recorded cold-load run; here
    /// the layout's canonical order stands in for the recording.
    pub fn learn(format: WeightFormat, pages: u64) -> Self {
        Self { order: format.access_order(pages), format, page_bytes: PAGE_BYTES }
    }

    pub fn page_count(&self) -> u64 {
        self.order.len() as u64
    }
}

fn read_ns(page_bytes: u64, bps: u64) -> u64 {
    page_bytes.saturating_mul(1_000_000_000) / bps.max(1)
}

/// One simulated cold-load outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOutcome {
    /// Wall time from "serve requested" to the last page resident on
    /// device (nanoseconds).
    pub cold_wall_ns: u64,
    /// Duration of the pre-servicing pre-warm window (nanoseconds) —
    /// zero for pure demand.
    pub prewarm_ns: u64,
    /// Peak staging-window occupancy in bytes (pages staged NVMe->host
    /// but not yet copied to device).
    pub peak_staged_bytes: u64,
    /// Peak staging-window occupancy in pages (same quantity as
    /// [`LoadOutcome::peak_staged_bytes`]).
    pub peak_lag_pages: u64,
}

/// Baseline: synchronous serial load — for each page in access order,
/// read it from NVMe into host, then copy it to the device; the next page
/// starts only after the current one is resident. Wall = Σ(read + copy).
pub fn demand_load(cfg: &LoaderConfig, template: &LoadTemplate) -> LoadOutcome {
    let read = read_ns(template.page_bytes, template.format.effective_nvme_bps(cfg));
    let copy = read_ns(template.page_bytes, cfg.device_pcie_bps);
    let wall = (read + copy).saturating_mul(template.page_count());
    LoadOutcome {
        cold_wall_ns: wall,
        prewarm_ns: 0,
        peak_staged_bytes: template.page_bytes,
        peak_lag_pages: 1,
    }
}

/// Two-channel pipelined replay (PPC/MAIO mechanism): the producer reads
/// pages NVMe->host in the template order while the consumer copies
/// staged pages host->device in the same order, starting at serving
/// time t=0. The producer may run at most `staging_bytes / page_bytes`
/// pages ahead of the consumer (the staging window bound). Deterministic
/// recurrence over the page timeline.
pub fn template_prewarm(cfg: &LoaderConfig, template: &LoadTemplate) -> LoadOutcome {
    head_prewarm(cfg, template, 0)
}

/// Pre-warm the first `head` pages of the learned order during the idle
/// window before serving begins (they are staged at t=0 when serving
/// starts); the remaining pages stream through the two-channel pipeline.
/// The head is capped by the staging window: pre-warmed pages occupy
/// host RAM, so `head > staging_bytes / page_bytes` would overfill it.
/// `head == 0` degenerates to the streaming [`template_prewarm`].
pub fn head_prewarm(cfg: &LoaderConfig, template: &LoadTemplate, head: u64) -> LoadOutcome {
    let p = template.page_count();
    if p == 0 {
        return LoadOutcome { cold_wall_ns: 0, prewarm_ns: 0, peak_staged_bytes: 0, peak_lag_pages: 0 };
    }
    let read = read_ns(template.page_bytes, template.format.effective_nvme_bps(cfg));
    let copy = read_ns(template.page_bytes, cfg.device_pcie_bps);
    let lag_pages = (cfg.staging_bytes / template.page_bytes).max(1);
    // Pre-warmed pages occupy the staging window too — cap the head.
    let head = head.min(lag_pages).min(p);

    // Pre-warm window: stage the first `head` pages NVMe->host.
    let prewarm_ns = read.saturating_mul(head);

    // Timeline: consumer copies in access order; the producer streams the
    // tail pages in access order, gated so that at any instant the count
    // of staged-but-not-yet-copied pages stays within `lag_pages`
    // (pre-warmed head pages included, since they sit in the window until
    // copied).
    let mut copy_end: Vec<u64> = vec![0; p as usize];
    let mut copy_end_prev: u64 = 0;
    let mut producer_ns: u64 = 0;
    let mut cons_done: u64 = 0; // pages whose device copy has completed
    let mut peak_occ: u64 = 0;
    for i in 0..p {
        let staged_at = if i < head {
            0 // pre-warmed before serving
        } else {
            producer_ns += read;
            let gate = if i >= lag_pages { copy_end[(i - lag_pages) as usize] } else { 0 };
            producer_ns = producer_ns.max(gate);
            producer_ns
        };
        let start = copy_end_prev.max(staged_at);
        copy_end_prev = start + copy;
        copy_end[i as usize] = copy_end_prev;
        // Occupancy at this instant: pages staged (0..=i) minus pages
        // whose device copy has completed by `staged_at`.
        while cons_done < p && copy_end[cons_done as usize] <= staged_at {
            cons_done += 1;
        }
        let occ = i + 1 - cons_done;
        peak_occ = peak_occ.max(occ);
    }
    LoadOutcome {
        cold_wall_ns: copy_end_prev,
        prewarm_ns,
        peak_staged_bytes: peak_occ.max(1) * template.page_bytes,
        peak_lag_pages: peak_occ.max(1),
    }
}

/// Break-even serving-session count for a head pre-warm: the pre-warm
/// window cost amortizes after `ceil(prewarm_ns / savings_ns)` sessions
/// where the same template is replayed. Returns 0 when the pipelined
/// replay saves nothing (storage-bound regime).
pub fn break_even_turns(demand: &LoadOutcome, pipelined: &LoadOutcome) -> u64 {
    let savings = demand.cold_wall_ns.saturating_sub(pipelined.cold_wall_ns);
    if savings == 0 {
        return 0;
    }
    pipelined.prewarm_ns.div_ceil(savings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_orders_are_permutations() {
        for fmt in [
            WeightFormat::Contiguous,
            WeightFormat::Sharded { shards: 4 },
            WeightFormat::MmapRandom,
        ] {
            let order = fmt.access_order(64);
            let mut seen = order.clone();
            seen.sort_unstable();
            let expected: Vec<u64> = (0..64).collect();
            assert_eq!(seen, expected, "{fmt:?} order must be a permutation of 0..64");
        }
        // Deterministic across calls.
        assert_eq!(
            WeightFormat::MmapRandom.access_order(64),
            WeightFormat::MmapRandom.access_order(64)
        );
    }

    #[test]
    fn demand_is_exact_serial_sum() {
        let cfg = LoaderConfig::default();
        let t = LoadTemplate::learn(WeightFormat::Contiguous, 16);
        let o = demand_load(&cfg, &t);
        let read = read_ns(PAGE_BYTES, cfg.nvme_seq_bps);
        let copy = read_ns(PAGE_BYTES, cfg.device_pcie_bps);
        assert_eq!(o.cold_wall_ns, (read + copy) * 16);
        assert_eq!(o.prewarm_ns, 0);
    }

    #[test]
    fn pipelined_never_slower_than_demand() {
        let cfg = LoaderConfig::default();
        for fmt in [
            WeightFormat::Contiguous,
            WeightFormat::Sharded { shards: 4 },
            WeightFormat::MmapRandom,
        ] {
            let t = LoadTemplate::learn(fmt, 256);
            let d = demand_load(&cfg, &t);
            let tp = template_prewarm(&cfg, &t);
            assert!(
                tp.cold_wall_ns <= d.cold_wall_ns,
                "{fmt:?}: pipelined {} must be <= demand {}",
                tp.cold_wall_ns,
                d.cold_wall_ns
            );
        }
    }

    #[test]
    fn staging_window_bounds_producer_lead() {
        let cfg = LoaderConfig { staging_bytes: 32 * PAGE_BYTES, ..LoaderConfig::default() };
        let t = LoadTemplate::learn(WeightFormat::Contiguous, 512);
        let o = template_prewarm(&cfg, &t);
        assert!(
            o.peak_lag_pages <= 32,
            "producer lead {} pages must stay within the 32-page window",
            o.peak_lag_pages
        );
    }

    #[test]
    fn head_prewarm_monotonic_in_head() {
        let cfg = LoaderConfig::default();
        let t = LoadTemplate::learn(WeightFormat::Contiguous, 256);
        let mut prev = u64::MAX;
        for head in [0u64, 32, 64, 128, 256] {
            let o = head_prewarm(&cfg, &t, head);
            assert!(
                o.cold_wall_ns <= prev,
                "head {head}: wall {} must be <= previous {}",
                o.cold_wall_ns,
                prev
            );
            prev = o.cold_wall_ns;
        }
    }

    #[test]
    fn zero_head_equals_streaming_pipeline() {
        let cfg = LoaderConfig::default();
        let t = LoadTemplate::learn(WeightFormat::Contiguous, 64);
        // head=0 is exactly the streaming pipeline: no pre-warm phase.
        let h0 = head_prewarm(&cfg, &t, 0);
        let full = template_prewarm(&cfg, &t);
        assert_eq!(h0, full);
    }

    #[test]
    fn empty_template_is_zeroed() {
        let cfg = LoaderConfig::default();
        let t = LoadTemplate { format: WeightFormat::Contiguous, order: vec![], page_bytes: PAGE_BYTES };
        let o = demand_load(&cfg, &t);
        assert_eq!(o.cold_wall_ns, 0);
        let p = head_prewarm(&cfg, &t, 4);
        assert_eq!(p.cold_wall_ns, 0);
    }

    #[test]
    fn break_even_is_zero_when_no_savings() {
        let cfg = LoaderConfig::default();
        // An empty template saves nothing, so the pre-warm never
        // amortizes.
        let t = LoadTemplate { format: WeightFormat::Contiguous, order: vec![], page_bytes: PAGE_BYTES };
        let d = demand_load(&cfg, &t);
        let tp = head_prewarm(&cfg, &t, 0);
        assert_eq!(break_even_turns(&d, &tp), 0);
    }
}
