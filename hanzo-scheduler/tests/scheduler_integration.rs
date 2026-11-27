// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive integration test suite for `hanzo-scheduler` (`hanzoai/engine#56`, HIP-1346).

use hanzo_scheduler::bench::{
    evaluate_end_to_end_gate, run_full_benchmark_suite, CONCURRENCY_SWEEP,
};
use hanzo_scheduler::config::{
    PreemptionMode, SchedulerConfig, SchedulerProfile, SchedulingPolicy, SloTargets,
};
use hanzo_scheduler::corpus::GroundedCorpus;
use hanzo_scheduler::kv_pool::{PagedKvPool, SchedulerError};
use hanzo_scheduler::model::{HardwareSpec, HybridModelSpec, PrefillChunkCostInput, StepCostModel};
use hanzo_scheduler::request::{InferenceRequest, RequestClass};
use hanzo_scheduler::scheduler::SchedulerEngine;
use hanzo_scheduler::verify::{HybridMicroModel, MicroModelConfig};

/// Test 01: Exact HIP-1346 `4B / 9B / 27B` hybrid matrix geometry alignment with
/// `hanzo-paged-attn/tests/kivi_kv_quant.rs:450-478`.
#[test]
fn test_01_hip1346_hybrid_matrix_exact_alignment() {
    let matrix = HybridModelSpec::canonical_matrix();
    assert_eq!(matrix.len(), 3);

    // 4B-Hybrid: 9 Attn + 27 GDN, num_kv_heads = 2, head_dim = 128, gdn_state_mib = 27.0
    assert_eq!(matrix[0].label, "4B-Hybrid");
    assert_eq!(matrix[0].num_attention_layers, 9);
    assert_eq!(matrix[0].num_gdn_layers, 27);
    assert_eq!(matrix[0].num_kv_heads, 2);
    assert_eq!(matrix[0].head_dim, 128);
    assert!((matrix[0].exact_gdn_state_mib() - 27.0).abs() < 1e-9);
    assert_eq!(matrix[0].kv_bytes_per_token_fp16(), 9 * 2 * 128 * 4);

    // 9B-Hybrid: 10 Attn + 30 GDN, num_kv_heads = 4, head_dim = 128, gdn_state_mib = 45.0
    assert_eq!(matrix[1].label, "9B-Hybrid");
    assert_eq!(matrix[1].num_attention_layers, 10);
    assert_eq!(matrix[1].num_gdn_layers, 30);
    assert_eq!(matrix[1].num_kv_heads, 4);
    assert_eq!(matrix[1].head_dim, 128);
    assert!((matrix[1].exact_gdn_state_mib() - 45.0).abs() < 1e-9);
    assert_eq!(matrix[1].kv_bytes_per_token_fp16(), 10 * 4 * 128 * 4);

    // 27B-Hybrid: 16 Attn + 48 GDN, num_kv_heads = 4, head_dim = 256, gdn_state_mib = 192.0
    assert_eq!(matrix[2].label, "27B-Hybrid");
    assert_eq!(matrix[2].num_attention_layers, 16);
    assert_eq!(matrix[2].num_gdn_layers, 48);
    assert_eq!(matrix[2].num_kv_heads, 4);
    assert_eq!(matrix[2].head_dim, 256);
    assert!((matrix[2].exact_gdn_state_mib() - 192.0).abs() < 1e-9);
    assert_eq!(matrix[2].kv_bytes_per_token_fp16(), 16 * 4 * 256 * 4);
}

/// Test 02: Paged KV block pool + fixed FP32 GDN slot allocator enforces strict VRAM OOM guards
/// and leaks zero blocks or slots on release.
#[test]
fn test_02_paged_kv_pool_oom_guard_and_zero_leak() {
    let mut pool = PagedKvPool::with_explicit_capacity(16, 327_680, 45 * 1024 * 1024, 10, 2)
        .expect("valid pool");
    assert!(pool.is_clean());

    // Admit seq 0 (needs 4 blocks for 64 tokens, 1 GDN slot).
    let t0 = pool.admit_sequence(0, 64).expect("admit seq 0");
    assert_eq!(t0.block_ids.len(), 4);
    assert_eq!(pool.free_kv_blocks(), 6);
    assert_eq!(pool.free_gdn_slots(), 1);

    // Duplicate sequence rejected.
    assert_eq!(
        pool.admit_sequence(0, 16),
        Err(SchedulerError::DuplicateSequence { seq_id: 0 })
    );

    // Admit seq 1 (needs 6 blocks for 96 tokens -> fills pool to 10/10 blocks, 2/2 GDN slots).
    pool.admit_sequence(1, 96).expect("admit seq 1");
    assert_eq!(pool.free_kv_blocks(), 0);
    assert_eq!(pool.free_gdn_slots(), 0);

    // Third sequence fails with InsufficientGdnSlots.
    assert_eq!(
        pool.admit_sequence(2, 16),
        Err(SchedulerError::InsufficientGdnSlots {
            seq_id: 2,
            available: 0
        })
    );

    // Growing seq 0 beyond 64 tokens fails with InsufficientKvBlocks without leaking.
    assert_eq!(
        pool.allocate_for_len(0, 65),
        Err(SchedulerError::InsufficientKvBlocks {
            seq_id: 0,
            required: 1,
            available: 0
        })
    );

    // Freeing both sequences restores 100% clean pool.
    pool.free_sequence(0);
    pool.free_sequence(1);
    assert!(pool.is_clean());
    assert_eq!(pool.peak_kv_blocks_used(), 10);
    assert_eq!(pool.peak_gdn_slots_used(), 2);
}

/// Test 03: Preemption via both Recompute and SwapToHost frees GPU KV+GDN capacity and restores cleanly.
#[test]
fn test_03_preemption_recompute_and_host_swap() {
    let mut pool = PagedKvPool::with_explicit_capacity(16, 327_680, 45 * 1024 * 1024, 8, 2)
        .expect("valid pool");

    pool.admit_sequence(10, 64).expect("admit 10"); // 4 blocks
    pool.admit_sequence(11, 64).expect("admit 11"); // 4 blocks
    assert_eq!(pool.free_kv_blocks(), 0);

    // Recompute preemption returns materialized tokens and frees blocks + GDN slot.
    let recomp_tokens = pool.preempt_recompute(11).expect("preempt 11");
    assert_eq!(recomp_tokens, 64);
    assert_eq!(pool.free_kv_blocks(), 4);
    assert_eq!(pool.free_gdn_slots(), 1);

    // Grow seq 10 into freed blocks, then swap out seq 10 to host RAM.
    pool.allocate_for_len(10, 128).expect("grow 10");
    assert_eq!(pool.free_kv_blocks(), 0);
    let swapped_bytes = pool.swap_out(10).expect("swap out 10");
    assert_eq!(swapped_bytes, 8 * 327_680 + 45 * 1024 * 1024);
    assert!(pool.is_swapped(10));
    assert_eq!(pool.free_kv_blocks(), 8);
    assert_eq!(pool.free_gdn_slots(), 2);

    // Swap back in and free.
    assert!(pool.can_swap_in(10, 0));
    let restored = pool.swap_in(10).expect("swap in 10");
    assert_eq!(restored, swapped_bytes);
    assert!(pool.is_resident(10));
    pool.free_sequence(10);
    assert!(pool.is_clean());
}

/// Test 04: Real CPU token-level output equivalence verifier proves 100% bit-identical outputs
/// (`0.000 pp` quality regression) across multiple arbitrary chunk sizes (`17`, `32`, `64`).
#[test]
fn test_04_token_level_output_equivalence_across_chunk_sizes() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    let cfg = MicroModelConfig::default();
    let model = HybridMicroModel::new(cfg, 1346);
    let specs = corpus.build_verify_specs(1346, cfg.vocab_size);

    for &chunk_budget in &[17usize, 32, 64] {
        let rep = model
            .verify_chunked_and_interleaved(&specs, chunk_budget, false)
            .expect("verify chunked");
        assert_eq!(rep.matched_tokens, rep.total_output_tokens);
        assert!((rep.output_correctness_pct - 100.0).abs() < 1e-9);
        assert!((rep.quality_regression_pp - 0.0).abs() < 1e-9);
        assert!(
            rep.max_abs_logit_diff <= 1e-6,
            "chunk_budget={chunk_budget} max_abs_logit_diff={}",
            rep.max_abs_logit_diff
        );
        assert!(rep.kv_pool_clean);
    }
}

/// Test 05: Token-level output equivalence holds bit-identically even across forced mid-prefill
/// preemption + recomputation.
#[test]
fn test_05_token_level_equivalence_with_preemption_recompute() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    let cfg = MicroModelConfig::default();
    let model = HybridMicroModel::new(cfg, 2026);
    let specs = corpus.build_verify_specs(2026, cfg.vocab_size);

    let rep = model
        .verify_chunked_and_interleaved(&specs, 17, true)
        .expect("verify with preemption");
    assert!(rep.preemption_recompute_verified);
    assert_eq!(rep.matched_tokens, rep.total_output_tokens);
    assert!((rep.output_correctness_pct - 100.0).abs() < 1e-9);
    assert!((rep.quality_regression_pp - 0.0).abs() < 1e-9);
    assert!(rep.max_abs_logit_diff <= 1e-6);
    assert!(rep.kv_pool_clean);
}

/// Test 06: Policy 1 (`StaticBatchingFcfs`) exhibits head-of-line blocking under mixed short/long requests.
#[test]
fn test_06_static_batching_exhibits_hol_blocking() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    let slo = SloTargets::default();
    let reqs = corpus.build_mixed_concurrency_workload(16, 1346, slo);
    let cost = StepCostModel::new(HybridModelSpec::hybrid_9b(), HardwareSpec::h100_sxm5_80gb());

    let static_m = SchedulerEngine::new(
        SchedulerConfig::for_policy(SchedulingPolicy::StaticBatchingFcfs),
        cost.clone(),
    )
    .unwrap()
    .run(reqs.clone())
    .unwrap()
    .metrics;
    let latency_m = SchedulerEngine::new(SchedulerConfig::latency_aware_slo(), cost)
        .unwrap()
        .run(reqs)
        .unwrap()
        .metrics;

    assert!(static_m.kv_pool_clean);
    assert!(
        static_m.short_ttft_p95_ms > latency_m.short_ttft_p95_ms * 10.0,
        "static short TTFT ({:.1}ms) must show strong HoL blocking vs latency-aware ({:.1}ms)",
        static_m.short_ttft_p95_ms,
        latency_m.short_ttft_p95_ms
    );
    assert!(static_m.slo_attainment_pct < latency_m.slo_attainment_pct);
}

/// Test 07: Policy 2 (`ContinuousBatchingPrefillFirst`) exhibits severe decode ITL stall spikes
/// when long 32K prefills interrupt active decodes.
#[test]
fn test_07_prefill_first_exhibits_decode_itl_stall_spikes() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    let slo = SloTargets::default();
    let reqs = corpus.build_mixed_concurrency_workload(16, 1346, slo);
    let cost = StepCostModel::new(HybridModelSpec::hybrid_9b(), HardwareSpec::h100_sxm5_80gb());

    let pf = SchedulerEngine::new(
        SchedulerConfig::for_policy(SchedulingPolicy::ContinuousBatchingPrefillFirst),
        cost.clone(),
    )
    .unwrap()
    .run(reqs.clone())
    .unwrap()
    .metrics;
    let ours = SchedulerEngine::new(SchedulerConfig::latency_aware_slo(), cost)
        .unwrap()
        .run(reqs)
        .unwrap()
        .metrics;

    assert!(
        pf.itl_max_ms > 1_000.0,
        "prefill-first must show >1s decode stall spike, got {:.1}ms",
        pf.itl_max_ms
    );
    assert!(
        ours.itl_max_ms < slo.itl_p95_ms,
        "latency-aware chunked scheduler must bound max ITL < {:.1}ms, got {:.1}ms",
        slo.itl_p95_ms,
        ours.itl_max_ms
    );
}

/// Test 08: Policy 3 (`ContinuousBatchingDecodePriority`) starves waiting prefills (high TTFT spikes).
#[test]
fn test_08_decode_priority_starves_waiting_prefills() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    let slo = SloTargets::default();
    let reqs = corpus.build_mixed_concurrency_workload(16, 1346, slo);
    let cost = StepCostModel::new(HybridModelSpec::hybrid_9b(), HardwareSpec::h100_sxm5_80gb());

    let dp = SchedulerEngine::new(
        SchedulerConfig::for_policy(SchedulingPolicy::ContinuousBatchingDecodePriority),
        cost.clone(),
    )
    .unwrap()
    .run(reqs.clone())
    .unwrap()
    .metrics;
    let ours = SchedulerEngine::new(SchedulerConfig::latency_aware_slo(), cost)
        .unwrap()
        .run(reqs)
        .unwrap()
        .metrics;

    assert!(
        dp.overall_ttft_p95_ms > ours.overall_ttft_p95_ms,
        "decode-priority TTFT ({:.1}ms) should exceed latency-aware ({:.1}ms)",
        dp.overall_ttft_p95_ms,
        ours.overall_ttft_p95_ms
    );
    assert!(dp.slo_attainment_pct < 50.0);
}

/// Test 09: Policy 4 (`ChunkedPrefillTokenBudget`) bounds max ITL spikes vs unchunked prefill-first
/// while piggybacking decodes onto prefill chunks.
#[test]
fn test_09_chunked_prefill_bounds_itl_vs_prefill_first() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    let slo = SloTargets::default();
    let reqs = corpus.build_mixed_concurrency_workload(16, 1346, slo);
    let cost = StepCostModel::new(HybridModelSpec::hybrid_9b(), HardwareSpec::h100_sxm5_80gb());

    let pf = SchedulerEngine::new(
        SchedulerConfig::for_policy(SchedulingPolicy::ContinuousBatchingPrefillFirst),
        cost.clone(),
    )
    .unwrap()
    .run(reqs.clone())
    .unwrap()
    .metrics;
    let chunked = SchedulerEngine::new(
        SchedulerConfig::for_policy(SchedulingPolicy::ChunkedPrefillTokenBudget),
        cost,
    )
    .unwrap()
    .run(reqs)
    .unwrap()
    .metrics;

    assert!(chunked.itl_max_ms < slo.itl_p95_ms);
    assert!(chunked.itl_max_ms * 20.0 < pf.itl_max_ms);
    assert!(chunked.kv_pool_clean);
}

/// Test 10: Policy 5 (`LatencyAwareChunkedKvScheduler`) protects BOTH short TTFT and p95 ITL SLOs
/// simultaneously on mixed workloads, achieving the highest SLO attainment of all 5 policies.
#[test]
fn test_10_latency_aware_scheduler_dominates_mixed_slo_attainment() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    let slo = SloTargets::default();
    let reqs = corpus.build_mixed_concurrency_workload(16, 1346, slo);
    let cost = StepCostModel::new(HybridModelSpec::hybrid_9b(), HardwareSpec::h100_sxm5_80gb());

    let ours = SchedulerEngine::new(SchedulerConfig::latency_aware_slo(), cost.clone())
        .unwrap()
        .run(reqs.clone())
        .unwrap()
        .metrics;
    assert_eq!(ours.slo_attainment_pct, 100.0);
    assert!(ours.short_ttft_p95_ms <= slo.short_ttft_p95_ms);
    assert!(ours.itl_p95_ms <= slo.itl_p95_ms);

    for policy in [
        SchedulingPolicy::StaticBatchingFcfs,
        SchedulingPolicy::ContinuousBatchingPrefillFirst,
        SchedulingPolicy::ContinuousBatchingDecodePriority,
        SchedulingPolicy::ChunkedPrefillTokenBudget,
    ] {
        let other = SchedulerEngine::new(SchedulerConfig::for_policy(policy), cost.clone())
            .unwrap()
            .run(reqs.clone())
            .unwrap()
            .metrics;
        assert!(
            ours.slo_attainment_pct > other.slo_attainment_pct,
            "ours ({:.1}%) must strictly beat {} ({:.1}%)",
            ours.slo_attainment_pct,
            policy.as_str(),
            other.slo_attainment_pct
        );
    }
}

/// Test 11: Operating profiles (`SingleUserInteractive`, `DatacenterThroughput`, `LatencyAwareSlo`)
/// have distinct configurations and protect their respective targets cleanly.
#[test]
fn test_11_separate_operating_profiles() {
    let su = SchedulerConfig::single_user_interactive();
    let dc = SchedulerConfig::datacenter_throughput();
    let la = SchedulerConfig::latency_aware_slo();

    assert_eq!(su.profile, SchedulerProfile::SingleUserInteractive);
    assert_eq!(dc.profile, SchedulerProfile::DatacenterThroughput);
    assert_eq!(la.profile, SchedulerProfile::LatencyAwareSlo);
    assert!(su.enable_single_user_bypass);
    assert!(!dc.enable_single_user_bypass);
    assert!(la.enable_single_user_bypass);
    assert!(dc.token_budget > la.token_budget);
    su.validate().unwrap();
    dc.validate().unwrap();
    la.validate().unwrap();
}

/// Test 12: Single-user bypass invariant — at `C = 1`, `LatencyAwareChunkedKvScheduler` takes
/// the zero-overhead single-user fast path and exactly matches `SingleUserInteractive`.
#[test]
fn test_12_single_user_bypass_zero_overhead_at_c1() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    let slo = SloTargets::default();
    let reqs = corpus.build_single_user_workload(slo);
    let cost = StepCostModel::new(HybridModelSpec::hybrid_9b(), HardwareSpec::h100_sxm5_80gb());

    let su = SchedulerEngine::new(SchedulerConfig::single_user_interactive(), cost.clone())
        .unwrap()
        .run(reqs.clone())
        .unwrap()
        .metrics;
    let la = SchedulerEngine::new(SchedulerConfig::latency_aware_slo(), cost)
        .unwrap()
        .run(reqs)
        .unwrap()
        .metrics;

    assert!(la.single_user_fast_path_steps > 0);
    assert_eq!(la.single_user_fast_path_steps, la.total_steps);
    assert!((la.cold_ttft_p50_ms - su.cold_ttft_p50_ms).abs() < 1e-6);
    assert!((la.itl_p95_ms - su.itl_p95_ms).abs() < 1e-6);
    assert!((la.makespan_s - su.makespan_s).abs() < 1e-6);
}

/// Test 13: Honest Negative Regime 1 (`neg_single_user_concurrency_1`) — forcing small-chunk
/// datacenter batching at `C = 1` measurably regresses TTFT and makespan vs single-user bypass.
#[test]
fn test_13_negative_regime_1_single_user_chunk_tax() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    let slo = SloTargets::default();
    let reqs = corpus.build_single_user_workload(slo);
    let cost = StepCostModel::new(HybridModelSpec::hybrid_9b(), HardwareSpec::h100_sxm5_80gb());

    let small_chunk =
        SchedulerEngine::new(SchedulerConfig::neg_single_user_small_chunk(), cost.clone())
            .unwrap()
            .run(reqs.clone())
            .unwrap()
            .metrics;
    let bypass = SchedulerEngine::new(SchedulerConfig::latency_aware_slo(), cost)
        .unwrap()
        .run(reqs)
        .unwrap()
        .metrics;

    assert_eq!(small_chunk.single_user_fast_path_steps, 0);
    assert!(
        small_chunk.cold_ttft_p50_ms > bypass.cold_ttft_p50_ms * 1.05,
        "small-chunk C=1 TTFT ({:.1}ms) must show >5% tax over bypass ({:.1}ms)",
        small_chunk.cold_ttft_p50_ms,
        bypass.cold_ttft_p50_ms
    );
    assert!(small_chunk.makespan_s > bypass.makespan_s);
}

/// Test 14: Honest Negative Regime 2 (`neg_kv_memory_oversubscription_256`) — at `C = 256` with
/// long 32K prompts exceeding GPU KV capacity, unguarded admission collapses into preemption
/// thrashing while KV-aware admission control queues cleanly with 0 preemptions and protected ITL.
#[test]
fn test_14_negative_regime_2_kv_oversubscription_256() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    let slo = SloTargets::default();
    let reqs = corpus.build_oversubscription_256_workload(1346, slo);
    let cost = StepCostModel::new(HybridModelSpec::hybrid_9b(), HardwareSpec::h100_sxm5_80gb());

    let unguarded = SchedulerEngine::new(
        SchedulerConfig::neg_unguarded_kv_oversubscription(),
        cost.clone(),
    )
    .unwrap()
    .run(reqs.clone())
    .unwrap()
    .metrics;
    let guarded = SchedulerEngine::new(SchedulerConfig::latency_aware_slo(), cost)
        .unwrap()
        .run(reqs)
        .unwrap()
        .metrics;

    // Exact unbounded demand must honestly exceed physical KV pool capacity (never clamped!).
    assert!(
        unguarded.peak_unbounded_demand_gib > unguarded.kv_pool_capacity_gib * 1.5,
        "demand ({:.1} GiB) must genuinely exceed pool ({:.1} GiB)",
        unguarded.peak_unbounded_demand_gib,
        unguarded.kv_pool_capacity_gib
    );
    assert!(
        unguarded.total_preemptions > 100,
        "unguarded oversubscription must exhibit preemption thrashing, got {}",
        unguarded.total_preemptions
    );
    assert!(unguarded.total_recomputed_tokens > 100_000);
    assert_eq!(guarded.total_preemptions, 0);
    assert_eq!(guarded.total_recomputed_tokens, 0);
    assert!(guarded.itl_p95_ms <= slo.itl_p95_ms);
    assert!(guarded.output_tok_per_s > unguarded.output_tok_per_s);
    assert!(guarded.mean_queue_wait_ms > 0.0);
    assert!(unguarded.kv_pool_clean && guarded.kv_pool_clean);
}

/// Test 15: Honest Negative Regime 3 (`neg_ultra_small_chunk_prefill_tax`) — `chunk_size = 64`
/// wins decode ITL microbenchmark but regresses 20-turn end-to-end wall time and is REJECTED.
#[test]
fn test_15_negative_regime_3_ultra_small_chunk_64_rejected() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    let slo = SloTargets::default();
    let workload = corpus.build_coding_agent_20_turn_workload(1346, slo, 16);
    let cost = StepCostModel::new(HybridModelSpec::hybrid_9b(), HardwareSpec::h100_sxm5_80gb());

    let chunk64 = SchedulerEngine::new(SchedulerConfig::neg_ultra_small_chunk_64(), cost.clone())
        .unwrap()
        .run(workload.clone())
        .unwrap()
        .metrics;
    let static_base = SchedulerEngine::new(
        SchedulerConfig::for_policy(SchedulingPolicy::StaticBatchingFcfs),
        cost.clone(),
    )
    .unwrap()
    .run(workload.clone())
    .unwrap()
    .metrics;
    let balanced = SchedulerEngine::new(SchedulerConfig::latency_aware_slo(), cost)
        .unwrap()
        .run(workload)
        .unwrap()
        .metrics;

    // Wins single-decode ITL max microbenchmark...
    assert!(chunk64.itl_max_ms < balanced.itl_max_ms);
    // ...yet severely regresses 20-turn end-to-end wall time and energy/token!
    assert!(chunk64.makespan_s > static_base.makespan_s * 1.15);
    assert!(chunk64.energy_mj_per_output_tok > balanced.energy_mj_per_output_tok * 1.15);

    let micro_cfg = MicroModelConfig::default();
    let verify = HybridMicroModel::new(micro_cfg, 1346)
        .verify_chunked_and_interleaved(
            &corpus.build_verify_specs(1346, micro_cfg.vocab_size),
            17,
            true,
        )
        .unwrap();
    let verdict_bad = evaluate_end_to_end_gate(&chunk64, &static_base, &verify, slo);
    let verdict_good = evaluate_end_to_end_gate(&balanced, &static_base, &verify, slo);
    assert!(!verdict_bad.accepted, "chunk_size=64 must be REJECTED");
    assert!(
        verdict_good.accepted,
        "latency_aware_chunked_kv must be ACCEPTED"
    );
}

/// Test 16: SwapToHost preemption mode works end-to-end inside SchedulerEngine under tight KV pool.
#[test]
fn test_16_swap_to_host_preemption_mode_end_to_end() {
    let model = HybridModelSpec::hybrid_9b();
    let hw = HardwareSpec::h100_sxm5_80gb();
    let cost = StepCostModel::new(model.clone(), hw);

    // Deliberately tight pool: only enough KV blocks for ~1.5 long sequences at a time.
    let pool = PagedKvPool::with_explicit_capacity(
        16,
        model.kv_bytes_per_block(16),
        model.gdn_state_bytes_per_seq(),
        250,
        4,
    )
    .unwrap();

    let mut cfg = SchedulerConfig::for_policy(SchedulingPolicy::ChunkedPrefillTokenBudget);
    cfg.kv_admission_control = false;
    cfg.preemption_mode = PreemptionMode::SwapToHost;
    cfg.token_budget = 256;

    let reqs: Vec<InferenceRequest> = (0..4)
        .map(|id| InferenceRequest {
            id,
            class: RequestClass::LongRepoPrefill,
            arrival_ms: id as f64,
            prompt_tokens: 1_536, // 96 blocks each -> 4 * 96 = 384 > 250 blocks!
            warm_prefix_tokens: 0,
            output_tokens: 16,
            slo_ttft_ms: 4_000.0,
            slo_itl_ms: 50.0,
            session_id: None,
            turn_index: 0,
            depends_on: None,
        })
        .collect();

    let out = SchedulerEngine::with_pool(cfg, cost, pool)
        .unwrap()
        .run(reqs)
        .unwrap();
    assert_eq!(out.metrics.completed_requests, 4);
    assert!(out.metrics.total_preemptions > 0);
    assert!(out.metrics.total_swap_bytes > 0);
    assert_eq!(out.metrics.total_recomputed_tokens, 0);
    assert!(out.metrics.kv_pool_clean);
}

/// Test 17: Grounded repository corpus SHA-256 provenance matches exact expected digests.
#[test]
fn test_17_grounded_corpus_sha256_provenance() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    assert_eq!(corpus.files.len(), 3);
    assert_eq!(corpus.total_bytes, 140_742);
    assert_eq!(
        corpus.files[0].sha256_hex,
        "aabd956dba34ff31f40e78b9de4e234fdd4865f90ba231e8e3deec8dfb22cbc0"
    );
    assert_eq!(
        corpus.files[1].sha256_hex,
        "67d1d402738727aa3059d28590fa3488c44740a358be4b5c2135c60f58ec9d7a"
    );
    assert_eq!(
        corpus.files[2].sha256_hex,
        "8d0650ea231eeade0b97e5c026411365ff9feed41b2b07ecf10853c2a399043a"
    );
    assert_eq!(
        corpus.combined_sha256_hex,
        "ebf55f7a1753ed837efcc218dd089c2f74559a7a6f5ff37d8db099a39c241a6d"
    );
}

/// Test 18: Full benchmark suite determinism + roofline compute/memory sanity + concurrency sweep.
#[test]
fn test_18_full_suite_determinism_and_roofline_physics() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");

    // Verify roofline physics: single decode is memory-bound, 4K prefill is compute-bound.
    let cost = StepCostModel::new(HybridModelSpec::hybrid_9b(), HardwareSpec::h100_sxm5_80gb());
    let dec = cost.estimate_step(&[], &[1024], true);
    let pre = cost.estimate_step(
        &[PrefillChunkCostInput {
            prefix_len: 0,
            chunk_len: 4096,
        }],
        &[],
        true,
    );
    assert!(
        !dec.compute_bound,
        "single-user decode must be memory-bound"
    );
    assert!(pre.compute_bound, "4K prefill must be compute-bound");

    // Run full suite twice with identical seed and verify bit-identical receipt SHA-256.
    let r1 = run_full_benchmark_suite(&corpus, 1346).expect("run 1");
    let r2 = run_full_benchmark_suite(&corpus, 1346).expect("run 2");
    assert_eq!(r1.receipt_sha256, r2.receipt_sha256);
    assert_eq!(r1.concurrency_sweep.len(), CONCURRENCY_SWEEP.len() * 5);
    assert_eq!(r1.hybrid_matrix_20turn.len(), 3 * 6);
    // All 3 models accept LatencyAwareChunkedKvScheduler and reject chunk_size=64.
    for model_label in ["4B-Hybrid", "9B-Hybrid", "27B-Hybrid"] {
        let ours = r1
            .hybrid_matrix_20turn
            .iter()
            .find(|r| r.model_label == model_label && r.regime_label == "latency_aware_chunked_kv")
            .unwrap();
        let bad64 = r1
            .hybrid_matrix_20turn
            .iter()
            .find(|r| {
                r.model_label == model_label
                    && r.regime_label == "neg_ultra_small_chunk_prefill_tax"
            })
            .unwrap();
        assert!(ours.verdict.accepted, "{model_label} ours must be ACCEPTED");
        assert!(
            !bad64.verdict.accepted,
            "{model_label} chunk64 must be REJECTED"
        );
    }
}

/// Test 19 (Codex P1 #1): No step may simultaneously preempt a sequence and schedule it in prefill/decode.
#[test]
fn test_19_codex_p1_no_step_preempts_and_schedules_same_seq() {
    let root = GroundedCorpus::find_repo_root();
    let corpus = GroundedCorpus::load(&root).expect("load corpus");
    let model = HybridModelSpec::hybrid_9b();
    let hw = HardwareSpec::h100_sxm5_80gb();
    let cost = StepCostModel::new(model, hw);
    let workload = corpus.build_oversubscription_256_workload(1346, SloTargets::default());

    let out = SchedulerEngine::new(SchedulerConfig::neg_unguarded_kv_oversubscription(), cost)
        .unwrap()
        .with_trace(true)
        .run(workload)
        .unwrap();

    assert!(
        out.metrics.total_preemptions > 0,
        "oversubscription workload must exercise preemption"
    );
    for plan in &out.trace {
        for &preempted_id in &plan.preempted_seqs {
            assert!(
                !plan.decode_seqs.contains(&preempted_id),
                "step {} both preempted and decoded seq {}",
                plan.step_index,
                preempted_id
            );
            assert!(
                !plan.prefill_chunks.iter().any(|c| c.seq_id == preempted_id),
                "step {} both preempted and prefilled seq {}",
                plan.step_index,
                preempted_id
            );
        }
    }
    assert!(out.metrics.kv_pool_clean);
}

/// Test 20 (Codex P1 #2): Recomputed decode replays prompt + consumed generated suffix from prefix_len=0
/// without double-counting already-emitted output tokens.
#[test]
fn test_20_codex_p1_recomputed_decode_replays_generated_history() {
    let model = HybridModelSpec::hybrid_9b();
    let hw = HardwareSpec::h100_sxm5_80gb();
    let cost = StepCostModel::new(model.clone(), hw);

    // 10 blocks * 16 tokens/block = 160 tokens total capacity; two 64-prompt + 32-output sequences
    // need 2 * 6 = 12 blocks -> forces mid-decode recompute preemption of seq 1.
    let pool = PagedKvPool::with_explicit_capacity(
        16,
        model.kv_bytes_per_block(16),
        model.gdn_state_bytes_per_seq(),
        10,
        2,
    )
    .unwrap();

    let mut cfg = SchedulerConfig::for_policy(SchedulingPolicy::ChunkedPrefillTokenBudget);
    cfg.kv_admission_control = false;
    cfg.preemption_mode = PreemptionMode::Recompute;
    cfg.token_budget = 64;
    cfg.max_num_seqs = 2;

    let reqs = vec![
        InferenceRequest {
            id: 0,
            class: RequestClass::ShortInteractive,
            arrival_ms: 0.0,
            prompt_tokens: 64,
            warm_prefix_tokens: 0,
            output_tokens: 32,
            slo_ttft_ms: 400.0,
            slo_itl_ms: 50.0,
            session_id: None,
            turn_index: 0,
            depends_on: None,
        },
        InferenceRequest {
            id: 1,
            class: RequestClass::ShortInteractive,
            arrival_ms: 0.0,
            prompt_tokens: 64,
            warm_prefix_tokens: 0,
            output_tokens: 32,
            slo_ttft_ms: 400.0,
            slo_itl_ms: 50.0,
            session_id: None,
            turn_index: 0,
            depends_on: None,
        },
    ];

    let out = SchedulerEngine::with_pool(cfg, cost, pool)
        .unwrap()
        .with_trace(true)
        .run(reqs)
        .unwrap();

    assert_eq!(out.metrics.completed_requests, 2);
    assert!(out.metrics.total_preemptions > 0);
    assert!(out.metrics.kv_pool_clean);
    for seq in &out.sequences {
        assert_eq!(seq.generated_tokens, 32);
        assert_eq!(seq.materialized_generated, 32);
    }

    // Verify that after preemption, the victim's first resumed replay chunk starts at prefix_len == 0
    // (never at prefix_len == generated_tokens) and replays more than the 64-token prompt.
    let mut saw_preempt_step: Option<usize> = None;
    let mut victim_id: Option<usize> = None;
    for plan in &out.trace {
        if saw_preempt_step.is_none() && !plan.preempted_seqs.is_empty() {
            saw_preempt_step = Some(plan.step_index);
            victim_id = Some(plan.preempted_seqs[0]);
            break;
        }
    }
    let p_step = saw_preempt_step.expect("must have at least one preemption");
    let v_id = victim_id.unwrap();
    let resumed_chunks: Vec<_> = out
        .trace
        .iter()
        .filter(|p| p.step_index > p_step)
        .flat_map(|p| p.prefill_chunks.iter())
        .filter(|c| c.seq_id == v_id)
        .collect();
    assert!(!resumed_chunks.is_empty(), "preempted victim must replay");
    assert_eq!(
        resumed_chunks[0].prefix_len, 0,
        "resumed replay after recompute must start at prefix_len=0"
    );
    let total_replayed: usize = resumed_chunks.iter().map(|c| c.chunk_len).sum();
    assert!(
        total_replayed > 64,
        "replay must cover prompt (64) + consumed generated suffix, got {total_replayed}"
    );
}

/// Test 21 (Codex P1 #3): Streamed step HBM bytes exclude the full input-embedding matrix
/// while keeping the full matrix in resident VRAM capacity accounting.
#[test]
fn test_21_codex_p1_streamed_weights_separate_embedding_table() {
    let model = HybridModelSpec::hybrid_9b();
    let hw = HardwareSpec::h100_sxm5_80gb();
    let cost = StepCostModel::new(model.clone(), hw.clone());

    // Full resident embedding table is ~1.245 GB and remains counted in VRAM capacity.
    let embed_table_bytes = model.embedding_table_bytes_bf16();
    assert_eq!(embed_table_bytes, 151_936 * 4096 * 2);
    assert_eq!(
        model.weight_bytes_bf16(),
        model.streamed_weight_bytes_bf16() + embed_table_bytes
    );
    assert_eq!(
        hw.available_kv_and_gdn_bytes(&model),
        hw.vram_bytes - (model.weight_bytes_bf16() + hw.workspace_bytes)
    );

    // Single-token decode reads streamed weights + 1 embed_tokens row (8 KiB), NOT the 1.24 GB table.
    let est = cost.estimate_step(&[], &[1], true);
    let expected_hbm = model.streamed_weight_bytes_bf16()
        + model.embedding_row_bytes_bf16()
        + model.kv_bytes_per_token_fp16() * 2
        + model.gdn_state_bytes_per_seq() * 2;
    assert_eq!(model.embedding_row_bytes_bf16(), 8_192);
    assert_eq!(est.hbm_bytes, expected_hbm);
    assert!(est.hbm_bytes < model.weight_bytes_bf16());
}

/// Test 22 (Codex P1 #4): Capped simulation with unfinished work returns Err instead of Ok(100% SLO).
#[test]
fn test_22_codex_p1_capped_simulation_rejects_unfinished_requests() {
    let model = HybridModelSpec::hybrid_9b();
    let hw = HardwareSpec::h100_sxm5_80gb();
    let cost = StepCostModel::new(model.clone(), hw);
    let pool = PagedKvPool::with_explicit_capacity(
        16,
        model.kv_bytes_per_block(16),
        model.gdn_state_bytes_per_seq(),
        20_000,
        4,
    )
    .unwrap();

    let reqs = vec![
        InferenceRequest {
            id: 0,
            class: RequestClass::ShortInteractive,
            arrival_ms: 0.0,
            prompt_tokens: 16,
            warm_prefix_tokens: 0,
            output_tokens: 1,
            slo_ttft_ms: 400.0,
            slo_itl_ms: 50.0,
            session_id: None,
            turn_index: 0,
            depends_on: None,
        },
        InferenceRequest {
            id: 1,
            class: RequestClass::ShortInteractive,
            arrival_ms: 0.0,
            prompt_tokens: 16,
            warm_prefix_tokens: 0,
            output_tokens: 200_005,
            slo_ttft_ms: 400.0,
            slo_itl_ms: 50.0,
            session_id: None,
            turn_index: 0,
            depends_on: None,
        },
    ];

    let err = SchedulerEngine::with_pool(SchedulerConfig::latency_aware_slo(), cost, pool)
        .unwrap()
        .run(reqs)
        .unwrap_err();
    match err {
        SchedulerError::SimulationStepLimitExceeded {
            completed_requests,
            unfinished_requests,
            ..
        } => {
            assert_eq!(completed_requests, 1);
            assert_eq!(unfinished_requests, 1);
        }
        other => panic!("expected SimulationStepLimitExceeded, got {other:?}"),
    }
}

/// Test 23 (Codex P2 #5): FCFS releases multi-arrival requests in chronological readiness order, not ID order.
#[test]
fn test_23_codex_p2_fcfs_releases_chronologically_by_ready_ms() {
    let model = HybridModelSpec::hybrid_9b();
    let hw = HardwareSpec::h100_sxm5_80gb();
    let cost = StepCostModel::new(model, hw);

    let mut cfg = SchedulerConfig::for_policy(SchedulingPolicy::ContinuousBatchingPrefillFirst);
    cfg.max_num_seqs = 1;

    let reqs = vec![
        InferenceRequest {
            id: 0,
            class: RequestClass::LongRepoPrefill,
            arrival_ms: 0.0,
            prompt_tokens: 4_096,
            warm_prefix_tokens: 0,
            output_tokens: 4,
            slo_ttft_ms: 6_000.0,
            slo_itl_ms: 50.0,
            session_id: None,
            turn_index: 0,
            depends_on: None,
        },
        InferenceRequest {
            id: 1,
            class: RequestClass::ShortInteractive,
            arrival_ms: 100.0,
            prompt_tokens: 64,
            warm_prefix_tokens: 0,
            output_tokens: 4,
            slo_ttft_ms: 400.0,
            slo_itl_ms: 50.0,
            session_id: None,
            turn_index: 0,
            depends_on: None,
        },
        InferenceRequest {
            id: 2,
            class: RequestClass::ShortInteractive,
            arrival_ms: 10.0,
            prompt_tokens: 64,
            warm_prefix_tokens: 0,
            output_tokens: 4,
            slo_ttft_ms: 400.0,
            slo_itl_ms: 50.0,
            session_id: None,
            turn_index: 0,
            depends_on: None,
        },
    ];

    let out = SchedulerEngine::new(cfg, cost)
        .unwrap()
        .with_trace(true)
        .run(reqs)
        .unwrap();
    let admission_order: Vec<usize> = out
        .trace
        .iter()
        .flat_map(|p| p.admitted_seqs.iter().copied())
        .collect();
    assert_eq!(admission_order, vec![0, 2, 1]);
}

/// Test 24 (Codex P2 #6): LatencyAwareChunkedKvScheduler strictly honors token_budget even when no decodes are active.
#[test]
fn test_24_codex_p2_latency_aware_honors_token_budget_when_no_decodes_active() {
    let model = HybridModelSpec::hybrid_9b();
    let hw = HardwareSpec::h100_sxm5_80gb();
    let cost = StepCostModel::new(model, hw);

    let mut cfg = SchedulerConfig::latency_aware_slo();
    cfg.enable_single_user_bypass = false;
    cfg.token_budget = 16;

    let reqs = vec![
        InferenceRequest {
            id: 0,
            class: RequestClass::ShortInteractive,
            arrival_ms: 0.0,
            prompt_tokens: 128,
            warm_prefix_tokens: 0,
            output_tokens: 4,
            slo_ttft_ms: 400.0,
            slo_itl_ms: 50.0,
            session_id: None,
            turn_index: 0,
            depends_on: None,
        },
        InferenceRequest {
            id: 1,
            class: RequestClass::ShortInteractive,
            arrival_ms: 0.0,
            prompt_tokens: 128,
            warm_prefix_tokens: 0,
            output_tokens: 4,
            slo_ttft_ms: 400.0,
            slo_itl_ms: 50.0,
            session_id: None,
            turn_index: 0,
            depends_on: None,
        },
    ];

    let out = SchedulerEngine::new(cfg, cost)
        .unwrap()
        .with_trace(true)
        .run(reqs)
        .unwrap();
    assert_eq!(out.metrics.completed_requests, 2);
    for plan in &out.trace {
        assert!(
            plan.total_step_tokens <= 16,
            "step {} emitted {} tokens, exceeding token_budget=16",
            plan.step_index,
            plan.total_step_tokens
        );
    }
}
