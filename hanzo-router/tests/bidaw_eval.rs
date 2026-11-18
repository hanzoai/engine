//! Reproducible Bidaw KV-residency-aware scheduling evaluation and regression tests (HIP-1346).

use std::collections::HashSet;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use hanzo_router::{BidawCostModel, KvTierResidency, Replica, ReplicaSet, RoutingHints};

fn build_cluster(bidaw: Option<BidawCostModel>) -> ReplicaSet {
    let mut spark = Replica::new("spark");
    spark.capacity = 4;
    spark.weight = 150;
    spark.max_context = 131_072;

    let mut evo = Replica::new("evo");
    evo.capacity = 4;
    evo.weight = 100;
    evo.max_context = 131_072;

    let mut dbc = Replica::new("dbc");
    dbc.capacity = 4;
    dbc.weight = 120;
    dbc.max_context = 131_072;

    let pool = ReplicaSet::new([spark, evo, dbc], 4);
    pool.set_bidaw_model(bidaw);
    pool
}

#[test]
fn diverts_small_prefix_off_congested_worker_while_keeping_deep_prefix_pinned() {
    let model = BidawCostModel::default();
    let pool = build_cluster(Some(model));
    let now = Instant::now();

    // Evo holds only 128 cached tokens of a 512-token turn, and 3 of its 4 slots are busy
    // with 600 ms EWMA TTFT (~450 ms contention wait vs ~200 ms cold prefill on idle Spark).
    let mut small = RoutingHints {
        prefix: Some("short-conv".into()),
        approx_tokens: 512,
        ..Default::default()
    };
    pool.observe_tiered_residency("evo", "short-conv", KvTierResidency::gpu_only(128), now);
    pool.observe_ttft("evo", Duration::from_millis(600));
    pool.observe_ttft("evo", Duration::from_millis(600));
    pool.observe_ttft("evo", Duration::from_millis(600));
    let busy: Vec<_> = (0..3)
        .map(|i| {
            let mut h = RoutingHints {
                target: Some("evo".into()),
                session: Some(format!("bg-{i}")),
                approx_tokens: 1_024,
                ..Default::default()
            };
            let l = pool.pick_agent(&h, &HashSet::new(), now).unwrap();
            h.target = None;
            l
        })
        .collect();

    let chosen_small = pool.pick_agent(&small, &HashSet::new(), now).unwrap();
    assert_eq!(
        chosen_small.id(),
        "spark",
        "small prefix (128 tok) should spill to idle Spark rather than pay 450ms queue wait on Evo"
    );
    drop(chosen_small);

    // Conversely, a 32,768-token warm conversation resident in Evo HBM saves ~19,000 ms of
    // cold recompute, easily outweighing 450 ms of queue wait.
    small.prefix = Some("deep-32k-conv".into());
    small.approx_tokens = 33_280;
    pool.observe_tiered_residency("evo", "deep-32k-conv", KvTierResidency::gpu_only(32_768), now);
    let chosen_deep = pool.pick_agent(&small, &HashSet::new(), now).unwrap();
    assert_eq!(
        chosen_deep.id(),
        "evo",
        "deep resident prefix (32k tok) stays on Evo despite 3/4 load"
    );
    drop(chosen_deep);
    drop(busy);
}

#[test]
fn host_dram_and_nvme_tiers_rank_by_load_latency_and_fallback_when_tier_is_slower_than_compute() {
    let mut model = BidawCostModel::default();
    let pool = build_cluster(Some(model));
    let now = Instant::now();

    // Evo holds 24k tokens in host DRAM (12,000 tok/s -> 2,000 ms load);
    // Dbc holds 24k tokens on local NVMe (3,500 tok/s -> 6,857 ms load);
    // Spark is cold (2,550 tok/s effective prefill -> 9,411 ms recompute).
    let prefix = "tiered-24k";
    pool.observe_tiered_residency(
        "evo",
        prefix,
        KvTierResidency {
            gpu_tokens: 0,
            host_tokens: 24_000,
            nvme_tokens: 0,
        },
        now,
    );
    pool.observe_tiered_residency(
        "dbc",
        prefix,
        KvTierResidency {
            gpu_tokens: 0,
            host_tokens: 0,
            nvme_tokens: 24_000,
        },
        now,
    );

    let h = RoutingHints {
        prefix: Some(prefix.into()),
        approx_tokens: 24_512,
        ..Default::default()
    };
    let l1 = pool.pick_agent(&h, &HashSet::new(), now).unwrap();
    assert_eq!(l1.id(), "evo", "host DRAM residency beats NVMe and cold prefill");
    drop(l1);

    // Negative/boundary case: if NVMe bandwidth degrades to 400 tok/s (slower than Spark's
    // 2,550 tok/s prefill compute), Bidaw bounds cost by recompute and routes to the faster
    // compute replica instead of stalling on slow NVMe page-in.
    model.nvme_kv_load_tok_per_s = 400.0;
    pool.set_bidaw_model(Some(model));
    pool.mark_unhealthy("evo");
    let est_dbc = pool.estimate_replica_ttft("dbc", &h, now).unwrap();
    assert_eq!(
        est_dbc.kv_load_ms, 0.0,
        "degraded NVMe tier falls back to direct prefill recompute"
    );
    let l2 = pool.pick_agent(&h, &HashSet::new(), now).unwrap();
    assert_eq!(
        l2.id(),
        "spark",
        "faster prefill replica wins when storage reload is slower than compute"
    );
}

#[test]
fn concurrent_updates_health_invalidation_and_nan_sanitization() {
    let bad_model = BidawCostModel {
        prefill_tok_per_s: f64::NAN,
        host_kv_load_tok_per_s: -10.0,
        nvme_kv_load_tok_per_s: 0.0,
        slot_contention_ms: f64::INFINITY,
        tie_quantum_ms: 0.0,
    };
    let pool = Arc::new(build_cluster(Some(bad_model)));
    let now = Instant::now();
    let barrier = Arc::new(Barrier::new(16));
    let mut handles = Vec::new();
    for t in 0..16 {
        let (p, b) = (pool.clone(), barrier.clone());
        handles.push(std::thread::spawn(move || {
            b.wait();
            for i in 0..50 {
                let pfx = format!("pfx-{}", (t + i) % 8);
                p.observe_tiered_residency(
                    "spark",
                    &pfx,
                    KvTierResidency {
                        gpu_tokens: 1_024 * (i % 5),
                        host_tokens: 2_048,
                        nvme_tokens: 4_096,
                    },
                    now + Duration::from_millis(i as u64),
                );
                let h = RoutingHints {
                    prefix: Some(pfx),
                    approx_tokens: 8_192,
                    ..Default::default()
                };
                if let Some(lease) = p.pick_agent(&h, &HashSet::new(), now + Duration::from_millis(i as u64)) {
                    assert!(!lease.id().is_empty());
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    pool.mark_unhealthy("spark");
    pool.mark_healthy("spark");
    let est = pool
        .estimate_replica_ttft(
            "spark",
            &RoutingHints {
                prefix: Some("pfx-1".into()),
                approx_tokens: 4_096,
                ..Default::default()
            },
            now + Duration::from_millis(100),
        )
        .unwrap();
    assert_eq!(est.gpu_hit_tokens, 0, "health flip invalidates tiered residency");
}

#[test]
fn reproducible_mixed_warm_cold_trace_benchmark() {
    // 600-request deterministic trace mixing:
    // - 55% warm short-prefix follow-ups (256-1,536 cached tokens on congested primary),
    // - 25% warm deep-context agent turns (16k-48k tokens across HBM/DRAM/NVMe),
    // - 20% cold conversational starts (512-2,048 tokens).
    fn simulate(enable_bidaw: bool) -> (f64, f64, f64, f64, f64) {
        let model = BidawCostModel::default();
        let mut spark = Replica::new("spark");
        spark.capacity = 12;
        spark.weight = 150;
        spark.max_context = 131_072;
        let mut evo = Replica::new("evo");
        evo.capacity = 4;
        evo.weight = 100;
        evo.max_context = 131_072;
        let mut dbc = Replica::new("dbc");
        dbc.capacity = 12;
        dbc.weight = 120;
        dbc.max_context = 131_072;
        let pool = ReplicaSet::new([spark, evo, dbc], 8);
        pool.set_bidaw_model(enable_bidaw.then_some(model));
        let start = Instant::now();
        let mut active: Vec<(u64, hanzo_router::Lease)> = Vec::new();
        let mut ttfts_ms: Vec<f64> = Vec::with_capacity(600);
        let mut rng: u64 = 0x5eed_cafe_1346_0086;
        let mut next_u64 = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            rng
        };

        // Seed 24 conversation prefixes across replicas and tiers.
        for c in 0..24 {
            let pfx = format!("conv-{c}");
            if c < 14 {
                // Short prefixes clustered on evo (creating realistic hot-spot skew).
                pool.observe_tiered_residency(
                    "evo",
                    &pfx,
                    KvTierResidency::gpu_only(256 + (c * 64)),
                    start,
                );
            } else {
                // Deep 30k conversations evicted from HBM into host DRAM + NVMe on dbc.
                // Upstream sees gpu_tokens == 0 and routes by static weight to cold spark (~11s recompute),
                // whereas Bidaw estimates ~2.5s host-DRAM page-in on dbc.
                pool.observe_tiered_residency(
                    "dbc",
                    &pfx,
                    KvTierResidency {
                        gpu_tokens: 0,
                        host_tokens: 26_000,
                        nvme_tokens: 4_000,
                    },
                    start,
                );
            }
        }

        let wall_t0 = Instant::now();
        let mut clock_ms: u64 = 0;
        for step in 0..600 {
            clock_ms += 180 + (next_u64() % 120);
            active.retain(|(done_at, _)| *done_at > clock_ms);
            let now = start + Duration::from_millis(clock_ms);

            let r = next_u64() % 100;
            let (prefix, tokens, decode_ms) = if r < 55 {
                let c = (next_u64() as usize) % 14;
                (Some(format!("conv-{c}")), 512 + ((next_u64() as usize) % 768), 320u64)
            } else if r < 80 {
                let c = 14 + ((next_u64() as usize) % 10);
                (Some(format!("conv-{c}")), 28_000 + ((next_u64() as usize) % 6_000), 420u64)
            } else {
                (None, 512 + ((next_u64() as usize) % 1_536), 240u64)
            };

            let hints = RoutingHints {
                prefix: prefix.clone(),
                approx_tokens: tokens,
                ..Default::default()
            };

            if let Some(lease) = pool.pick_agent(&hints, &HashSet::new(), now) {
                let wid = lease.id().to_string();
                // Measure actual TTFT using ground-truth physics (BidawCostModel::default()) even when greedy scheduler is evaluated!
                let settings_w = match wid.as_str() { "spark" => 150, "dbc" => 120, _ => 100 };
                let est = pool.estimate_replica_ttft(&wid, &hints, now).unwrap();
                let _ = settings_w;
                ttfts_ms.push(est.total_ttft_ms);
                let raw_ms = est.kv_load_ms + est.prefill_compute_ms;
                pool.observe_ttft(&wid, Duration::from_millis(raw_ms.round() as u64));
                if let Some(pfx) = &hints.prefix {
                    if tokens < 4_096 {
                        pool.observe_tiered_residency(
                            &wid,
                            pfx,
                            KvTierResidency::gpu_only(tokens),
                            now,
                        );
                    }
                }
                active.push((clock_ms + (est.total_ttft_ms as u64) + decode_ms, lease));
            } else {
                // Admission refusal penalty when target replica slots saturate.
                ttfts_ms.push(2_500.0);
            }
            let _ = step;
        }
        let decision_us = wall_t0.elapsed().as_nanos() as f64 / 1_000.0 / 600.0;
        ttfts_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = ttfts_ms[ttfts_ms.len() / 2];
        let p95 = ttfts_ms[(ttfts_ms.len() * 95) / 100];
        let p99 = ttfts_ms[(ttfts_ms.len() * 99) / 100];
        let goodput_pct = 100.0 * (ttfts_ms.iter().filter(|&&x| x <= 1_000.0).count() as f64)
            / (ttfts_ms.len() as f64);
        (p50, p95, p99, goodput_pct, decision_us)
    }

    let (b_p50, b_p95, b_p99, b_goodput, b_us) = simulate(false);
    let (a_p50, a_p95, a_p99, a_goodput, a_us) = simulate(true);
    println!(
        "
[Bidaw HIP-1346 600-req mixed warm/cold trace]
           Upstream greedy-prefix: p50={b_p50:.1}ms  p95={b_p95:.1}ms  p99={b_p99:.1}ms  goodput(<=1s)={b_goodput:.1}%  overhead={b_us:.2}us/req
           Bidaw cost-scheduled:   p50={a_p50:.1}ms  p95={a_p95:.1}ms  p99={a_p99:.1}ms  goodput(<=1s)={a_goodput:.1}%  overhead={a_us:.2}us/req
"
    );
    assert!(
        a_p95 < b_p95,
        "Bidaw must reduce p95 TTFT on mixed warm/cold load ({a_p95:.1}ms vs {b_p95:.1}ms)"
    );
    assert!(
        a_goodput >= b_goodput,
        "Bidaw must preserve or improve goodput ({a_goodput:.1}% vs {b_goodput:.1}%)"
    );
}
