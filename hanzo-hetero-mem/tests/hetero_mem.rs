//! Comprehensive integration test suite for `hanzo-hetero-mem` (`hanzoai/engine#57`, HIP-1346).

use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

use hanzo_hetero_mem::bench::run_full_benchmark;
use hanzo_hetero_mem::corpus::{GroundedCorpus, SplitMix64};
use hanzo_hetero_mem::hardware::{HardwareProfile, MemoryArchitecture};
use hanzo_hetero_mem::mmap_store::{
    decode_tile, encode_tile, run_real_host_io_benchmark, LoadMode, PagedWeightStore, WeightFile,
    DEFAULT_GROUP_SIZE, PAGE_SIZE,
};
use hanzo_hetero_mem::model::{FitRegime, HybridModelSpec, WeightFormat, CONTEXT_128K, GIB};
use hanzo_hetero_mem::partition::{
    ActivationProfile, NegativeRegimeKind, PlacementPlanner, PlacementRegime,
};
use hanzo_hetero_mem::sha256::Sha256;
use hanzo_hetero_mem::verify::run_execution_equivalence_verifier;

#[test]
fn t01_exact_hip_1346_shared_model_matrix_alignment() {
    let m4 = HybridModelSpec::model_4b_hybrid();
    let m9 = HybridModelSpec::model_9b_hybrid();
    let m27 = HybridModelSpec::model_27b_hybrid();

    // Matches hanzo-paged-attn/tests/kivi_kv_quant.rs:450-478 byte-for-byte.
    assert_eq!(m4.num_attention_layers, 9);
    assert_eq!(m4.num_gdn_layers, 27);
    assert_eq!(m4.num_kv_heads, 2);
    assert_eq!(m4.head_dim, 128);
    assert!((m4.gdn_state_mib() - 27.0).abs() < 1e-9);
    assert_eq!(m4.kv_bytes_per_token(), 9_216);

    assert_eq!(m9.num_attention_layers, 10);
    assert_eq!(m9.num_gdn_layers, 30);
    assert_eq!(m9.num_kv_heads, 4);
    assert_eq!(m9.head_dim, 128);
    assert!((m9.gdn_state_mib() - 45.0).abs() < 1e-9);
    assert_eq!(m9.kv_bytes_per_token(), 20_480);

    assert_eq!(m27.num_attention_layers, 16);
    assert_eq!(m27.num_gdn_layers, 48);
    assert_eq!(m27.num_kv_heads, 4);
    assert_eq!(m27.head_dim, 256);
    assert!((m27.gdn_state_mib() - 192.0).abs() < 1e-9);
    assert_eq!(m27.kv_bytes_per_token(), 65_536);
}

#[test]
fn t02_fit_regime_classification_on_discrete_24g_and_unified() {
    let gpu24 = HardwareProfile::discrete_gpu_24g();
    let gpu80 = HardwareProfile::discrete_gpu_80g();
    let uni24 = HardwareProfile::unified_memory_24g();
    let uni64 = HardwareProfile::unified_memory_64g();

    let m4 = HybridModelSpec::model_4b_hybrid();
    let m9 = HybridModelSpec::model_9b_hybrid();
    let m27 = HybridModelSpec::model_27b_hybrid();

    // On DiscreteGpu24G at 128K BF16: 4B fully fits, 9B barely fits, 27B exceeds VRAM!
    assert_eq!(
        FitRegime::classify(
            m4.total_working_set_bytes(WeightFormat::Bf16, CONTEXT_128K, 1),
            gpu24.usable_vram_bytes
        ),
        FitRegime::FullyFits
    );
    assert_eq!(
        FitRegime::classify(
            m9.total_working_set_bytes(WeightFormat::Bf16, CONTEXT_128K, 1),
            gpu24.usable_vram_bytes
        ),
        FitRegime::BarelyFits
    );
    assert_eq!(
        FitRegime::classify(
            m27.total_working_set_bytes(WeightFormat::Bf16, CONTEXT_128K, 1),
            gpu24.usable_vram_bytes
        ),
        FitRegime::ExceedsVram
    );
    // Even INT4 27B at 128K still exceeds DiscreteGpu24G (proving need for PowerInfer offload).
    assert_eq!(
        FitRegime::classify(
            m27.total_working_set_bytes(WeightFormat::Int4Packed, CONTEXT_128K, 1),
            gpu24.usable_vram_bytes
        ),
        FitRegime::ExceedsVram
    );
    // On DiscreteGpu80G at C=1 128K, 27B BF16 barely fits / fits; at C=4 it exceeds HBM.
    assert_eq!(
        FitRegime::classify(
            m27.total_working_set_bytes(WeightFormat::Bf16, CONTEXT_128K, 4),
            gpu80.usable_vram_bytes
        ),
        FitRegime::ExceedsVram
    );

    // Unified memory never double-counts physical RAM.
    assert_eq!(uni24.architecture, MemoryArchitecture::UnifiedZeroCopy);
    assert_eq!(uni24.total_physical_memory_bytes(), 24 * GIB);
    assert_eq!(uni64.total_physical_memory_bytes(), 64 * GIB);
    assert_eq!(gpu24.total_physical_memory_bytes(), 88 * GIB);
}

#[test]
fn t03_sha256_matches_nist_standard_vectors() {
    assert_eq!(
        Sha256::hex_digest(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        Sha256::hex_digest(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        Sha256::hex_digest(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn t04_grounded_corpus_loads_and_matches_exact_sha256() {
    let corpus = GroundedCorpus::load_from_workspace().expect("load grounded corpus");
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
        "84b45bc2b79cb4612938566807c5eb8b478824036f382307c869cb112a8cd856"
    );
}

#[test]
fn t05_int4_int8_bf16_tile_roundtrip_accuracy() {
    let mut rng = SplitMix64::new(0x1346_2026);
    let vals: Vec<f32> = (0..1024).map(|_| rng.next_signed_f32() * 2.0).collect();

    for (fmt, max_tol) in [
        (WeightFormat::Bf16, 0.02f32),
        (WeightFormat::Int8Packed, 0.02f32),
        (WeightFormat::Int4Packed, 0.20f32),
    ] {
        let encoded = encode_tile(fmt, &vals, DEFAULT_GROUP_SIZE);
        let decoded = decode_tile(fmt, &encoded, vals.len(), DEFAULT_GROUP_SIZE).unwrap();
        assert_eq!(decoded.len(), vals.len());
        let max_err = vals
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err <= max_tol,
            "format {:?} max_err={max_err} > {max_tol}",
            fmt
        );
    }
}

#[test]
fn t06_weight_file_4k_alignment_sha256_and_tamper_detection() {
    let mut rng = SplitMix64::new(42);
    let tiles: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..4096).map(|_| rng.next_signed_f32()).collect())
        .collect();
    let path =
        std::env::temp_dir().join(format!("hanzo_hetero_mem_t06_{}.hzwt", std::process::id()));

    let wf = WeightFile::create(
        &path,
        WeightFormat::Int4Packed,
        DEFAULT_GROUP_SIZE,
        &tiles,
        true,
    )
    .unwrap();
    assert_eq!(wf.header().data_offset, PAGE_SIZE as u64);
    assert_eq!(wf.header().tile_stride_bytes % PAGE_SIZE, 0);
    assert!(wf.verify_sha256().unwrap());

    // Tamper with a single byte in tile 3 payload -> SHA-256 verification MUST fail!
    {
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let off = wf.header().tile_offset(3).unwrap() + 17;
        f.seek(SeekFrom::Start(off)).unwrap();
        f.write_all(&[0xFF]).unwrap();
        f.flush().unwrap();
    }
    assert!(!wf.verify_sha256().unwrap());
}

#[test]
fn t07_demand_paged_cold_start_slashes_initial_major_faults_vs_eager() {
    let mut rng = SplitMix64::new(77);
    let tiles: Vec<Vec<f32>> = (0..64)
        .map(|_| (0..16_384).map(|_| rng.next_signed_f32()).collect())
        .collect();
    let path =
        std::env::temp_dir().join(format!("hanzo_hetero_mem_t07_{}.hzwt", std::process::id()));
    let wf = WeightFile::create(
        &path,
        WeightFormat::Int4Packed,
        DEFAULT_GROUP_SIZE,
        &tiles,
        true,
    )
    .unwrap();

    let mut eager = PagedWeightStore::open(&wf, LoadMode::EagerFullRead).unwrap();
    let e0 = eager.fetch_tile_f32(0).unwrap();
    let eager_cold_faults = eager.stats().major_faults;

    let mut demand = PagedWeightStore::open(
        &wf,
        LoadMode::DemandPaged {
            max_resident_tiles: 16,
        },
    )
    .unwrap();
    let d0 = demand.fetch_tile_f32(0).unwrap();
    let demand_cold_faults = demand.stats().major_faults;

    assert_eq!(e0, d0);
    // Demand-paged faults only header + tile 0; eager faults header + all 64 tiles!
    assert!(
        demand_cold_faults * 10 < eager_cold_faults,
        "demand={demand_cold_faults} should be << eager={eager_cold_faults}"
    );
}

#[test]
fn t08_async_lookahead_prefetch_achieves_hits_and_bit_exact_tiles() {
    let report = run_real_host_io_benchmark(1346, 64, 32_768).unwrap();
    assert!(report.sha256_verified);
    assert_eq!(report.modes.len(), 5);

    let eager = report
        .modes
        .iter()
        .find(|m| m.mode_label == "eager_full_read")
        .unwrap();
    let async_pf = report
        .modes
        .iter()
        .find(|m| m.mode_label == "async_lookahead_prefetch")
        .unwrap();
    let unified_zc = report
        .modes
        .iter()
        .find(|m| m.mode_label == "unified_zero_copy_view")
        .unwrap();
    let unified_cb = report
        .modes
        .iter()
        .find(|m| m.mode_label == "unified_naive_copy_bounce")
        .unwrap();

    // All 5 real I/O modes must produce bit-identical decoded checksums!
    for m in &report.modes {
        assert!(
            (m.checksum_acc - eager.checksum_acc).abs() < 1e-9,
            "mode {} checksum mismatch: {} vs {}",
            m.mode_label,
            m.checksum_acc,
            eager.checksum_acc
        );
    }

    assert!(async_pf.cold_start_major_faults < eager.cold_start_major_faults);
    assert!(
        async_pf.prefetch_hit_rate > 0.50,
        "expected real background prefetch hits, got {:.3}",
        async_pf.prefetch_hit_rate
    );
    assert_eq!(unified_zc.copy_bytes, 0);
    assert!(unified_cb.copy_bytes >= report.file_bytes - PAGE_SIZE as u64);
    assert!(unified_cb.peak_resident_pages > unified_zc.peak_resident_pages);
}

#[test]
fn t09_zero_copy_unified_view_pointer_equality() {
    let mut rng = SplitMix64::new(99);
    let tiles: Vec<Vec<f32>> = (0..4)
        .map(|_| (0..4096).map(|_| rng.next_signed_f32()).collect())
        .collect();
    let path =
        std::env::temp_dir().join(format!("hanzo_hetero_mem_t09_{}.hzwt", std::process::id()));
    let wf = WeightFile::create(
        &path,
        WeightFormat::Int8Packed,
        DEFAULT_GROUP_SIZE,
        &tiles,
        true,
    )
    .unwrap();
    let mut store = PagedWeightStore::open(
        &wf,
        LoadMode::UnifiedZeroCopyView {
            max_resident_tiles: 4,
        },
    )
    .unwrap();

    let zc = store.fetch_unified_zero_copy_view(0).unwrap();
    assert!(zc.is_zero_copy());
    assert_eq!(store.stats().copy_bytes, 0);

    let cb = store.fetch_naive_copy_bounce_view(1).unwrap();
    assert!(!cb.is_zero_copy());
    assert!(cb.bytes_copied > 0);
    assert_eq!(store.stats().copy_bytes, cb.bytes_copied as u64);
}

#[test]
fn t10_grounded_corpus_activation_profiler_exhibits_power_law_skew() {
    let corpus = GroundedCorpus::load_from_workspace().unwrap();
    let skewed = ActivationProfile::from_grounded_corpus(&corpus, 224);
    let uniform = ActivationProfile::low_skew_uniform(224, 1346);

    assert!(
        skewed.is_skewed,
        "corpus must exhibit strong PowerInfer skew"
    );
    assert!(skewed.gini >= 0.55, "gini={}", skewed.gini);
    assert!(
        skewed.top_20_pct_coverage >= 0.65,
        "top20={}",
        skewed.top_20_pct_coverage
    );

    assert!(!uniform.is_skewed);
    assert!(uniform.gini < 0.10, "uniform gini={}", uniform.gini);
    assert!(uniform.top_20_pct_coverage < 0.25);
}

#[test]
fn t11_bit_exact_pure_rust_hybrid_execution_verifier() {
    let corpus = GroundedCorpus::load_from_workspace().unwrap();
    let report = run_execution_equivalence_verifier(&corpus, 1346, 48);

    assert!(report.all_passed());
    assert_eq!(report.regimes.len(), 4);
    for r in &report.regimes {
        assert_eq!(r.matched_tokens, 48);
        assert_eq!(r.quality_regression_pp, 0.0);
        assert!(
            r.max_abs_err <= 1e-5,
            "regime {} max_abs_err={:e}",
            r.regime,
            r.max_abs_err
        );
    }
    let unified = report
        .regimes
        .iter()
        .find(|r| r.regime == "unified_zero_copy_view")
        .unwrap();
    assert_eq!(unified.weight_copy_bytes, 0);
    assert!(unified.zero_copy_ptr_verified);
}

#[test]
fn t12_powerinfer_beats_static_split_and_naive_on_27b_discrete_24g() {
    let corpus = GroundedCorpus::load_from_workspace().unwrap();
    let model = HybridModelSpec::model_27b_hybrid();
    let hw = HardwareProfile::discrete_gpu_24g();
    let skew = ActivationProfile::from_grounded_corpus(&corpus, model.neuron_blocks_per_layer);

    let eager = PlacementPlanner::evaluate_regime(
        &model,
        &hw,
        PlacementRegime::EagerFullReadNaiveOffload,
        &skew,
        100.0,
        0.0,
    );
    let split = PlacementPlanner::evaluate_regime(
        &model,
        &hw,
        PlacementRegime::LayerSplitStaticOffload,
        &skew,
        100.0,
        0.0,
    );
    let powerinfer = PlacementPlanner::evaluate_regime(
        &model,
        &hw,
        PlacementRegime::PowerInferHotColdHybrid,
        &skew,
        100.0,
        0.0,
    );

    // PowerInfer keeps VRAM strictly within usable 22.5 GiB while beating both naive and static split.
    assert!(powerinfer.plan.peak_vram_bytes <= hw.usable_vram_bytes);
    assert!(powerinfer.plan.gpu_active_mac_coverage > 0.90);
    assert!(powerinfer.itl_p50_ms < split.itl_p50_ms);
    assert!(powerinfer.agent_20_turn_wall_s < split.agent_20_turn_wall_s);
    assert!(powerinfer.agent_20_turn_wall_s * 5.0 < eager.agent_20_turn_wall_s);
    assert!(powerinfer.gate_accepted);
    assert!(!eager.gate_accepted);
}

#[test]
fn t13_adaptive_planner_selects_topology_and_skew_appropriate_strategy() {
    let corpus = GroundedCorpus::load_from_workspace().unwrap();
    let m4 = HybridModelSpec::model_4b_hybrid();
    let m9 = HybridModelSpec::model_9b_hybrid();
    let m27 = HybridModelSpec::model_27b_hybrid();

    let skew4 = ActivationProfile::from_grounded_corpus(&corpus, m4.neuron_blocks_per_layer);
    let skew9 = ActivationProfile::from_grounded_corpus(&corpus, m9.neuron_blocks_per_layer);
    let skew27 = ActivationProfile::from_grounded_corpus(&corpus, m27.neuron_blocks_per_layer);
    let flat27 = ActivationProfile::low_skew_uniform(m27.neuron_blocks_per_layer, 1346);

    // 1. Unified memory -> unified_zero_copy_view with 0 copy bytes.
    let uni = PlacementPlanner::evaluate_regime(
        &m9,
        &HardwareProfile::unified_memory_24g(),
        PlacementRegime::AdaptiveHeteroPlanner,
        &skew9,
        100.0,
        0.0,
    );
    assert_eq!(uni.plan.resolved_strategy, "unified_zero_copy_view");
    assert!(uni.plan.zero_copy_active);
    assert_eq!(uni.copy_bytes_gib, 0.0);
    assert!(uni.gate_accepted);

    // 2. 4B on DiscreteGpu24G (FullyFits) -> full_gpu_demand_paged_bf16.
    let fit4 = PlacementPlanner::evaluate_regime(
        &m4,
        &HardwareProfile::discrete_gpu_24g(),
        PlacementRegime::AdaptiveHeteroPlanner,
        &skew4,
        100.0,
        0.0,
    );
    assert_eq!(fit4.plan.resolved_strategy, "full_gpu_demand_paged_bf16");
    assert!(fit4.gate_accepted);

    // 3. 27B on DiscreteGpu24G (ExceedsVram) with skewed activations -> powerinfer_hot_cold_hybrid.
    let ex27_skew = PlacementPlanner::evaluate_regime(
        &m27,
        &HardwareProfile::discrete_gpu_24g(),
        PlacementRegime::AdaptiveHeteroPlanner,
        &skew27,
        100.0,
        0.0,
    );
    assert_eq!(
        ex27_skew.plan.resolved_strategy,
        "powerinfer_hot_cold_hybrid"
    );
    assert!(ex27_skew.gate_accepted);

    // 4. 27B on DiscreteGpu24G with flat/uniform activations -> automatic safe fallback to static layer split!
    let ex27_flat = PlacementPlanner::evaluate_regime(
        &m27,
        &HardwareProfile::discrete_gpu_24g(),
        PlacementRegime::AdaptiveHeteroPlanner,
        &flat27,
        100.0,
        0.0,
    );
    assert_eq!(
        ex27_flat.plan.resolved_strategy,
        "layer_split_static_offload_low_skew_fallback"
    );
    assert!(ex27_flat.gate_accepted);
}

#[test]
fn t14_unified_zero_copy_rejects_false_claim_on_discrete_pcie() {
    let corpus = GroundedCorpus::load_from_workspace().unwrap();
    let m9 = HybridModelSpec::model_9b_hybrid();
    let skew = ActivationProfile::from_grounded_corpus(&corpus, m9.neuron_blocks_per_layer);
    let sc = PlacementPlanner::evaluate_regime(
        &m9,
        &HardwareProfile::discrete_gpu_24g(),
        PlacementRegime::UnifiedZeroCopyView,
        &skew,
        100.0,
        0.0,
    );
    assert!(!sc.plan.zero_copy_active);
    assert!(!sc.gate_accepted);
    assert_eq!(
        sc.gate_reason,
        "rejected_zero_copy_unsupported_on_discrete_pcie"
    );
}

#[test]
fn t15_negative_regime_1_unified_memory_naive_copy_bounce_rejected() {
    let corpus = GroundedCorpus::load_from_workspace().unwrap();
    let negs = PlacementPlanner::evaluate_negative_regimes(&corpus, 1346, 100.0, 0.0);
    let neg1 = negs
        .iter()
        .find(|n| n.kind == NegativeRegimeKind::UnifiedMemoryNaiveCopyBounce)
        .unwrap();

    assert!(!neg1.negative.gate_accepted);
    assert!(neg1.baseline.gate_accepted);
    assert_eq!(neg1.baseline.copy_bytes_gib, 0.0);
    assert!(neg1.negative.copy_bytes_gib > 50.0);
    assert!(
        neg1.peak_mem_delta_pct > 50.0,
        "must significantly increase unified residency"
    );
    assert!(
        neg1.wall_time_delta_pct > 15.0,
        "must regress 20-turn wall time"
    );
    assert!(neg1.energy_delta_pct > 15.0, "must regress energy/token");
}

#[test]
fn t16_negative_regime_2_low_skew_uniform_activation_offload_rejected() {
    let corpus = GroundedCorpus::load_from_workspace().unwrap();
    let negs = PlacementPlanner::evaluate_negative_regimes(&corpus, 1346, 100.0, 0.0);
    let neg2 = negs
        .iter()
        .find(|n| n.kind == NegativeRegimeKind::LowSkewUniformActivationOffload)
        .unwrap();

    assert!(!neg2.negative.gate_accepted);
    assert!(neg2.baseline.gate_accepted);
    assert!(
        neg2.wall_time_delta_pct > 10.0,
        "uniform activation offload must regress vs static layer split, got {:+.2}%",
        neg2.wall_time_delta_pct
    );
    assert!(neg2.negative.itl_p50_ms > neg2.baseline.itl_p50_ms);
}

#[test]
fn t17_negative_regime_3_aggressive_prefetch_thrash_tight_ram_rejected() {
    let corpus = GroundedCorpus::load_from_workspace().unwrap();
    let negs = PlacementPlanner::evaluate_negative_regimes(&corpus, 1346, 100.0, 0.0);
    let neg3 = negs
        .iter()
        .find(|n| n.kind == NegativeRegimeKind::AggressivePrefetchThrashTightRam)
        .unwrap();

    // Microbenchmark win: higher prefetch hit rate!
    assert!(neg3.negative.plan.prefetch_hit_rate > neg3.baseline.plan.prefetch_hit_rate);
    // End-to-end loss: overflows usable VRAM, evicts live KV, regresses 20-turn wall time -> REJECTED!
    assert!(neg3.negative.plan.vram_eviction_overflow_bytes > 0);
    assert_eq!(neg3.baseline.plan.vram_eviction_overflow_bytes, 0);
    assert!(neg3.wall_time_delta_pct > 15.0);
    assert!(!neg3.negative.gate_accepted);
    assert!(neg3.baseline.gate_accepted);
}

#[test]
fn t18_full_benchmark_determinism_and_unclamped_infeasible_reporting() {
    let b1 = run_full_benchmark(1346).unwrap();
    let b2 = run_full_benchmark(1346).unwrap();

    assert_eq!(b1.matrix_scorecards.len(), 60);
    assert_eq!(b1.negative_regimes.len(), 3);
    assert_eq!(
        b1.deterministic_receipt_sha256,
        b2.deterministic_receipt_sha256
    );

    // Unclamped check: 27B-Hybrid on UnifiedMemory24G must honestly report > 20 GiB exact demand.
    let u24_27b = b1
        .matrix_scorecards
        .iter()
        .find(|sc| {
            sc.model_label == "27B-Hybrid"
                && sc.hardware_id == "UnifiedMemory24G"
                && sc.plan.regime_name == "unified_zero_copy_view"
        })
        .unwrap();
    assert!(
        u24_27b.plan.peak_total_mem_bytes > HardwareProfile::unified_memory_24g().usable_vram_bytes
    );
    assert!(!u24_27b.gate_accepted);
}
