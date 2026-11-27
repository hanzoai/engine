// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive unit & integration verification suite for `hanzo-duoattn` (`hanzoai/engine#52`).

use std::sync::{Arc, Barrier};
use std::thread;

use hanzo_duoattn::attention::{compute_fidelity, DualAttentionModel};
use hanzo_duoattn::cache::{DualKvCache, HeadKvStorage, KvDtype};
use hanzo_duoattn::calibrate::{calibrate_model, CalibrationConfig};
use hanzo_duoattn::error::DuoAttnError;
use hanzo_duoattn::model_matrix::{
    evaluate_agent_loop_scorecard, HybridModelSpec, MatrixRegime, HANZO_LSA_SINK_SIZE,
    HANZO_LSA_WINDOW_SIZE,
};
use hanzo_duoattn::profile::{HeadCalibrationProfile, HeadRole, ModelIdentity};
use hanzo_duoattn::workload::{
    build_calibration_dataset, build_five_workloads, evaluate_workload_regime,
    run_cross_model_negative_regime, run_short_context_negative_regime, CorpusEmbedder, RepoCorpus,
};

fn build_test_model(checkpoint_id: &str) -> DualAttentionModel {
    let id = ModelIdentity::hybrid(checkpoint_id, 4, 12, 16, 8, 8).unwrap();
    DualAttentionModel::from_identity(id, 0.25).unwrap()
}

#[test]
fn calibration_converges_and_recovers_retrieval_vs_streaming_heads() {
    let model = build_test_model("hanzo-zen6-test-cal@v1");
    let corpus = RepoCorpus::synthetic_for_tests(1346);
    let embedder = CorpusEmbedder::new(model.d_model(), 16, 1346);
    let seqs = build_calibration_dataset(&corpus, &embedder, 6, 192, 4, 24, 1346);

    let cfg = CalibrationConfig {
        sink_tokens: 4,
        window_tokens: 24,
        retrieval_ratio: 0.25,
        steps: 220,
        learning_rate: 0.05,
        l1_lambda: 0.035,
        seed: 1346,
    };

    let report = calibrate_model(&model, &seqs, &cfg).unwrap();
    assert!(
        report.final_loss < report.initial_loss,
        "calibration objective must decrease: {} -> {}",
        report.initial_loss,
        report.final_loss
    );
    assert!(
        report.mean_retrieval_alpha > report.mean_streaming_alpha + 0.35,
        "retrieval heads ({:.4}) must separate clearly above streaming heads ({:.4})",
        report.mean_retrieval_alpha,
        report.mean_streaming_alpha
    );

    let (precision, recall) = report.diagnostic_precision_recall(&model);
    assert!(
        precision >= 0.99 && recall >= 0.99,
        "expected exact recovery of retrieval heads, got precision={precision:.3}, recall={recall:.3}"
    );
    assert_eq!(report.profile.retrieval_head_count(), 8);
    assert_eq!(report.profile.streaming_head_count(), 24);
    report.profile.verify_for_model(model.identity()).unwrap();
}

#[test]
fn deterministic_seed_produces_bit_identical_profiles_and_outputs() {
    let model = build_test_model("hanzo-zen6-det@v1");
    let corpus = RepoCorpus::synthetic_for_tests(2026);
    let embedder = CorpusEmbedder::new(model.d_model(), 16, 2026);
    let seqs = build_calibration_dataset(&corpus, &embedder, 4, 160, 4, 20, 2026);
    let cfg = CalibrationConfig {
        sink_tokens: 4,
        window_tokens: 20,
        retrieval_ratio: 0.25,
        steps: 180,
        learning_rate: 0.05,
        l1_lambda: 0.035,
        seed: 2026,
    };

    let rep1 = calibrate_model(&model, &seqs, &cfg).unwrap();
    let rep2 = calibrate_model(&model, &seqs, &cfg).unwrap();

    assert_eq!(rep1.profile, rep2.profile);
    assert_eq!(rep1.profile.checksum_sha256, rep2.profile.checksum_sha256);

    let mut c1 =
        DualKvCache::new_calibrated(model.identity(), &rep1.profile, KvDtype::F16).unwrap();
    let mut c2 =
        DualKvCache::new_calibrated(model.identity(), &rep2.profile, KvDtype::F16).unwrap();
    let out1 = model
        .forward_sequence(&seqs[0].embeddings, &mut c1)
        .unwrap();
    let out2 = model
        .forward_sequence(&seqs[0].embeddings, &mut c2)
        .unwrap();
    assert_eq!(out1, out2);
}

#[test]
fn fail_closed_rejects_cross_model_architecture_mismatch_and_tampering() {
    let model_a = build_test_model("hanzo-zen6-alpha@sha256:aaa");
    let model_b = build_test_model("hanzo-zen6-beta@sha256:bbb");
    let corpus = RepoCorpus::synthetic_for_tests(1346);
    let embedder = CorpusEmbedder::new(model_a.d_model(), 16, 1346);
    let seqs = build_calibration_dataset(&corpus, &embedder, 4, 160, 4, 20, 1346);
    let cfg = CalibrationConfig {
        sink_tokens: 4,
        window_tokens: 20,
        retrieval_ratio: 0.25,
        steps: 150,
        learning_rate: 0.05,
        l1_lambda: 0.035,
        seed: 1346,
    };
    let profile_a = calibrate_model(&model_a, &seqs, &cfg).unwrap().profile;

    // 1. Cross-model rejection
    let err = DualKvCache::new_calibrated(model_b.identity(), &profile_a, KvDtype::F16)
        .expect_err("cross-model profile must fail closed");
    assert!(matches!(err, DuoAttnError::ModelMismatch { .. }));

    // 2. Architecture mismatch rejection (same checkpoint ID, wrong head count)
    let wrong_arch = ModelIdentity::hybrid("hanzo-zen6-alpha@sha256:aaa", 4, 12, 16, 4, 8).unwrap();
    let err = profile_a
        .verify_for_model(&wrong_arch)
        .expect_err("architecture mismatch must fail closed");
    assert!(matches!(
        err,
        DuoAttnError::ArchitectureMismatch {
            field: "num_kv_heads",
            ..
        }
    ));

    // 3. SHA-256 tamper rejection
    let mut tampered = profile_a.clone();
    tampered.heads[0].role = match tampered.heads[0].role {
        HeadRole::Retrieval => HeadRole::Streaming,
        HeadRole::Streaming => HeadRole::Retrieval,
    };
    let err = tampered
        .verify_for_model(model_a.identity())
        .expect_err("tampered profile must fail checksum check");
    assert!(matches!(err, DuoAttnError::ProfileChecksumMismatch { .. }));

    // 4. JSON persistence round-trip preserves integrity
    let json = profile_a.to_json();
    let reloaded = HeadCalibrationProfile::from_json(&json).unwrap();
    assert_eq!(reloaded, profile_a);
    reloaded.verify_for_model(model_a.identity()).unwrap();
}

#[test]
fn streaming_ring_buffer_enforces_strict_constant_memory_and_chronological_order() {
    let mut store = HeadKvStorage::new_streaming(4, 4, 8);
    for pos in 0..100usize {
        let k = [pos as f32, 1.0, 2.0, 3.0];
        let v = [pos as f32 * 10.0, 0.5, 0.25, 0.125];
        store.append(pos, &k, &v).unwrap();
        assert!(
            store.resident_tokens() <= 12,
            "streaming head exceeded sink+window cap at pos {pos}"
        );
    }
    assert_eq!(store.total_seen(), 100);
    assert_eq!(store.resident_tokens(), 12);
    assert_eq!(store.evicted_tokens(), 88);

    let mut positions = Vec::new();
    store.for_each_resident(|_, _, pos| positions.push(pos));
    assert_eq!(positions, vec![0, 1, 2, 3, 92, 93, 94, 95, 96, 97, 98, 99]);
}

#[test]
fn recurrent_gdn_layers_are_exempt_fail_closed() {
    let id = ModelIdentity::hybrid("hanzo-zen6-4b@v1", 9, 27, 16, 4, 8).unwrap();
    let mut cache = DualKvCache::new_full(&id, KvDtype::F16).unwrap();
    let dummy = vec![0.0_f32; 4 * 8];

    // Layer 0 in 9-attn / 27-gdn interleaved layout is RecurrentGdn.
    let err = cache
        .append_global_layer_step(0, 0, &dummy, &dummy)
        .expect_err("recurrent GDN layer must be rejected fail-closed");
    assert!(matches!(
        err,
        DuoAttnError::RecurrentLayerExempt { layer_idx: 0 }
    ));

    // Final layer (35) is AttentionKv and must succeed.
    cache
        .append_global_layer_step(35, 0, &dummy, &dummy)
        .unwrap();
}

#[test]
fn four_regimes_ordering_and_three_negative_regimes_verified() {
    let model_a = build_test_model("hanzo-zen6-regimes-a@v1");
    let model_b = build_test_model("hanzo-zen6-regimes-b@v1");
    let corpus = RepoCorpus::synthetic_for_tests(1346);
    let embedder = CorpusEmbedder::new(model_a.d_model(), 16, 1346);

    let seq_len = 224;
    let sink = 4;
    let window = 24;

    let cal_seqs = build_calibration_dataset(&corpus, &embedder, 6, seq_len, sink, window, 1346);
    let cal_cfg = CalibrationConfig {
        sink_tokens: sink,
        window_tokens: window,
        retrieval_ratio: 0.25,
        steps: 220,
        learning_rate: 0.05,
        l1_lambda: 0.035,
        seed: 1346,
    };
    let prof_a = calibrate_model(&model_a, &cal_seqs, &cal_cfg)
        .unwrap()
        .profile;
    let prof_b = calibrate_model(&model_b, &cal_seqs, &cal_cfg)
        .unwrap()
        .profile;

    let workloads = build_five_workloads(&corpus, &embedder, seq_len, sink, window, 1346);
    let (distant_name, distant_eps) = &workloads[1];
    assert_eq!(distant_name, "distant_reference_retrieval");

    // 1. Full KV baseline
    let (full_res, full_refs) =
        evaluate_workload_regime(distant_name, &model_a, &embedder, distant_eps, &[], || {
            DualKvCache::new_full(model_a.identity(), KvDtype::F16)
        })
        .unwrap();
    // 2. Uniform streaming baseline
    let (unif_res, _) = evaluate_workload_regime(
        distant_name,
        &model_a,
        &embedder,
        distant_eps,
        &full_refs,
        || DualKvCache::new_uniform_streaming(model_a.identity(), sink, window, KvDtype::F16),
    )
    .unwrap();
    // 3. Random head selection control (25% budget, uncalibrated)
    let (rand_res, _) = evaluate_workload_regime(
        distant_name,
        &model_a,
        &embedder,
        distant_eps,
        &full_refs,
        || {
            DualKvCache::new_random_control(
                model_a.identity(),
                sink,
                window,
                0.25,
                1346 + 99,
                KvDtype::F16,
            )
        },
    )
    .unwrap();
    // 4. Calibrated DuoAttention (25% budget)
    let (duo_res, _) = evaluate_workload_regime(
        distant_name,
        &model_a,
        &embedder,
        distant_eps,
        &full_refs,
        || DualKvCache::new_calibrated(model_a.identity(), &prof_a, KvDtype::F16),
    )
    .unwrap();

    assert!(
        full_res.accuracy >= 0.99,
        "full KV baseline must solve distant retrieval: {:.2}",
        full_res.accuracy
    );
    assert!(
        duo_res.accuracy >= 0.99 && duo_res.cosine_vs_full > 0.99,
        "calibrated DuoAttention must preserve fidelity: acc={:.2}, cos={:.6}",
        duo_res.accuracy,
        duo_res.cosine_vs_full
    );
    assert!(
        duo_res.kv_memory_reduction_x > 2.5,
        "calibrated DuoAttention must reduce KV memory significantly: {:.2}x",
        duo_res.kv_memory_reduction_x
    );
    assert!(
        unif_res.accuracy < 0.20,
        "uniform streaming must collapse on distant retrieval: {:.2}",
        unif_res.accuracy
    );
    assert!(
        duo_res.accuracy > rand_res.accuracy + 0.15 || duo_res.cosine_vs_full > rand_res.cosine_vs_full + 0.05,
        "calibrated DuoAttention must strictly beat uncalibrated random head selection: duo=({:.2}, {:.4}) vs rand=({:.2}, {:.4})",
        duo_res.accuracy,
        duo_res.cosine_vs_full,
        rand_res.accuracy,
        rand_res.cosine_vs_full
    );

    // Negative Regime 1: Cross-model profile transfer collapses accuracy and is blocked fail-closed.
    let neg1 = run_cross_model_negative_regime(
        &model_b,
        &prof_a,
        &prof_b,
        &embedder,
        distant_eps,
        KvDtype::F16,
    )
    .unwrap();
    assert!(neg1.fail_closed_rejected);
    assert!(
        neg1.own_calibrated_accuracy > neg1.wrong_model_profile_accuracy + 0.25,
        "cross-model profile must degrade accuracy sharply: own={:.2} vs wrong={:.2}",
        neg1.own_calibrated_accuracy,
        neg1.wrong_model_profile_accuracy
    );

    // Negative Regime 2: Over-aggressive pruning (5% retrieval heads) triggers quality cliff.
    let over_pruned = prof_a.rethreshold(0.05).unwrap();
    let (pruned_res, _) = evaluate_workload_regime(
        distant_name,
        &model_a,
        &embedder,
        distant_eps,
        &full_refs,
        || DualKvCache::new_calibrated(model_a.identity(), &over_pruned, KvDtype::F16),
    )
    .unwrap();
    assert!(
        pruned_res.accuracy < duo_res.accuracy - 0.30,
        "5% over-pruning must show quality cliff: pruned={:.2} vs calibrated_25pct={:.2}",
        pruned_res.accuracy,
        duo_res.accuracy
    );

    // Negative Regime 3: Short context (L = 64 <= sink=4 + window=512 = 516) yields 1.00x KV reduction.
    let short_prof = HeadCalibrationProfile::from_alphas(
        model_a.identity(),
        HANZO_LSA_SINK_SIZE,
        HANZO_LSA_WINDOW_SIZE,
        0.25,
        1346,
        10,
        0.035,
        1.0,
        0.2,
        &prof_a.heads.iter().map(|h| h.alpha).collect::<Vec<_>>(),
    )
    .unwrap();
    let neg3 = run_short_context_negative_regime(
        &model_a,
        &short_prof,
        &corpus,
        &embedder,
        64,
        2,
        KvDtype::F16,
        1346,
    )
    .unwrap();
    assert!((neg3.kv_memory_reduction_x - 1.0).abs() < 1e-9);
    assert!((neg3.cosine_vs_full - 1.0).abs() < 1e-5);
}

#[test]
fn hybrid_4b_9b_27b_memory_accounting_matches_hip_1346_kivi_baseline() {
    // Exact 128K (131,072 tokens) baseline bytes from docs/HIP-1346-kivi-kv-compression.md:
    // 4B: 1,152 MiB Attn + 27 MiB GDN = 1,179 MiB
    // 9B: 2,560 MiB Attn + 45 MiB GDN = 2,605 MiB
    // 27B: 8,192 MiB Attn + 192 MiB GDN = 8,384 MiB
    let mib = 1024 * 1024;
    let ctx_128k = 131_072;

    let s4 = HybridModelSpec::hybrid_4b();
    assert_eq!(s4.full_attn_kv_bytes(ctx_128k) / mib, 1152);
    assert_eq!(
        (s4.full_attn_kv_bytes(ctx_128k) + s4.gdn_state_bytes) / mib,
        1179
    );

    let s9 = HybridModelSpec::hybrid_9b();
    assert_eq!(s9.full_attn_kv_bytes(ctx_128k) / mib, 2560);
    assert_eq!(
        (s9.full_attn_kv_bytes(ctx_128k) + s9.gdn_state_bytes) / mib,
        2605
    );

    let s27 = HybridModelSpec::hybrid_27b();
    assert_eq!(s27.full_attn_kv_bytes(ctx_128k) / mib, 8192);
    assert_eq!(
        (s27.full_attn_kv_bytes(ctx_128k) + s27.gdn_state_bytes) / mib,
        8384
    );

    // Verify 20-turn coding-agent scorecard shows clear DuoAttention end-to-end win over full KV & LSA memory.
    let sc_full = evaluate_agent_loop_scorecard(
        &s27,
        MatrixRegime::FullKvBaseline,
        0.25,
        128,
        512,
        1.0,
        1.0,
        1.0,
        0.0,
    );
    let sc_lsa = evaluate_agent_loop_scorecard(
        &s27,
        MatrixRegime::LsaExistingHanzo,
        0.25,
        128,
        512,
        0.985,
        1.0,
        0.996,
        0.018,
    );
    let sc_duo = evaluate_agent_loop_scorecard(
        &s27,
        MatrixRegime::DuoAttentionCalibrated,
        0.25,
        128,
        512,
        1.0,
        1.0,
        0.999,
        0.005,
    );
    assert!(sc_duo.attn_kv_reduction_x > 3.8);
    assert!(sc_lsa.attn_kv_reduction_x < 1.0); // LSA retains full KV + centroid metadata
    assert!(sc_duo.agent_loop_wall_s < sc_full.agent_loop_wall_s);
    assert!(sc_duo.analytical_mj_per_token < sc_full.analytical_mj_per_token);
}

#[test]
fn multithreaded_concurrent_calibrated_decode_is_deterministic() {
    let model = Arc::new(build_test_model("hanzo-zen6-mt@v1"));
    let corpus = RepoCorpus::synthetic_for_tests(777);
    let embedder = CorpusEmbedder::new(model.d_model(), 16, 777);
    let seqs = build_calibration_dataset(&corpus, &embedder, 3, 128, 4, 20, 777);
    let cfg = CalibrationConfig {
        sink_tokens: 4,
        window_tokens: 20,
        retrieval_ratio: 0.25,
        steps: 120,
        learning_rate: 0.05,
        l1_lambda: 0.035,
        seed: 777,
    };
    let profile = Arc::new(calibrate_model(&model, &seqs, &cfg).unwrap().profile);
    let prompt = Arc::new(seqs[0].embeddings.clone());

    let mut ref_cache =
        DualKvCache::new_calibrated(model.identity(), &profile, KvDtype::F16).unwrap();
    let expected = model.forward_sequence(&prompt, &mut ref_cache).unwrap();

    let n_threads = 8;
    let barrier = Arc::new(Barrier::new(n_threads));
    let mut handles = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let m = Arc::clone(&model);
        let p = Arc::clone(&profile);
        let x = Arc::clone(&prompt);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            let mut cache = DualKvCache::new_calibrated(m.identity(), &p, KvDtype::F16).unwrap();
            m.forward_sequence(&x, &mut cache).unwrap()
        }));
    }

    for h in handles {
        let out = h.join().unwrap();
        let fid = compute_fidelity(&expected, &out);
        assert_eq!(out, expected);
        assert!((fid.cosine_similarity - 1.0).abs() < 1e-12);
    }
}

#[test]
fn profile_rethresholding_preserves_model_binding_and_monotone_head_counts() {
    let model = build_test_model("hanzo-zen6-rethresh@v1");
    let corpus = RepoCorpus::synthetic_for_tests(1346);
    let embedder = CorpusEmbedder::new(model.d_model(), 16, 1346);
    let seqs = build_calibration_dataset(&corpus, &embedder, 4, 160, 4, 20, 1346);
    let cfg = CalibrationConfig {
        sink_tokens: 4,
        window_tokens: 20,
        retrieval_ratio: 0.25,
        steps: 140,
        learning_rate: 0.05,
        l1_lambda: 0.035,
        seed: 1346,
    };
    let base = calibrate_model(&model, &seqs, &cfg).unwrap().profile;

    let mut prev_count = 0usize;
    for ratio in [0.0_f32, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00] {
        let p = base.rethreshold(ratio).unwrap();
        p.verify_for_model(model.identity()).unwrap();
        assert!(p.retrieval_head_count() >= prev_count);
        prev_count = p.retrieval_head_count();
    }
    assert_eq!(prev_count, model.identity().total_kv_heads());
}

#[test]
fn real_versioned_repo_corpus_loads_with_valid_sha256_provenance() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let corpus = RepoCorpus::load_from_repo(repo_root).unwrap();
    assert_eq!(corpus.files.len(), 3);
    assert!(corpus.token_ids.len() > 1000);
    for f in &corpus.files {
        assert!(f.byte_len > 0);
        assert_eq!(f.sha256.len(), 64);
    }
}

#[test]
fn short_context_detector_flags_zero_savings_window() {
    let id = ModelIdentity::pure_attention("hanzo-short-test@v1", 2, 8, 4, 8).unwrap();
    let mut cache = DualKvCache::new_uniform_streaming(&id, 4, 16, KvDtype::F16).unwrap();
    let kv = vec![0.25_f32; 4 * 8];
    for pos in 0..20usize {
        for l in 0..2 {
            cache.append_layer_step(l, pos, &kv, &kv).unwrap();
        }
    }
    assert!(cache.is_short_context());
    assert!((cache.memory_reduction_ratio() - 1.0).abs() < 1e-12);

    // One token past sink + window (pos = 20 -> seq_len = 21) exits short-context regime.
    for l in 0..2 {
        cache.append_layer_step(l, 20, &kv, &kv).unwrap();
    }
    assert!(!cache.is_short_context());
    assert!(cache.memory_reduction_ratio() > 1.0);
}

#[test]
fn invalid_configs_and_dimension_mismatches_fail_closed() {
    assert!(ModelIdentity::pure_attention("", 2, 8, 4, 8).is_err());
    assert!(ModelIdentity::pure_attention("m", 2, 7, 4, 8).is_err());
    assert!(ModelIdentity::hybrid("m", 0, 4, 8, 4, 8).is_err());

    let id = ModelIdentity::pure_attention("m-valid", 2, 8, 4, 8).unwrap();
    assert!(DualKvCache::new_uniform_streaming(&id, 0, 16, KvDtype::F16).is_err());
    assert!(DualKvCache::new_random_control(&id, 4, 16, 1.5, 1, KvDtype::F16).is_err());
    assert!(DualKvCache::new_random_control(&id, 4, 16, f32::NAN, 1, KvDtype::F16).is_err());

    let mut cache = DualKvCache::new_full(&id, KvDtype::F16).unwrap();
    let wrong_len = vec![0.0_f32; 5];
    let err = cache
        .append_layer_step(0, 0, &wrong_len, &wrong_len)
        .expect_err("wrong key length must fail closed");
    assert!(matches!(err, DuoAttnError::DimensionMismatch { .. }));
}

#[test]
fn streaming_head_matches_full_kv_exactly_within_sink_plus_window() {
    let model = build_test_model("hanzo-zen6-exact-short@v1");
    let corpus = RepoCorpus::synthetic_for_tests(42);
    let embedder = CorpusEmbedder::new(model.d_model(), 16, 42);
    let mut embeddings = Vec::with_capacity(32);
    for (pos, &tok) in corpus.token_ids.iter().enumerate().take(32) {
        embeddings.push(embedder.embed_background(tok, pos, 4));
    }

    let mut full_cache = DualKvCache::new_full(model.identity(), KvDtype::F16).unwrap();
    let mut stream_cache =
        DualKvCache::new_uniform_streaming(model.identity(), 4, 28, KvDtype::F16).unwrap();

    let out_full = model
        .forward_sequence(&embeddings, &mut full_cache)
        .unwrap();
    let out_stream = model
        .forward_sequence(&embeddings, &mut stream_cache)
        .unwrap();
    let fid = compute_fidelity(&out_full, &out_stream);
    assert!(fid.max_abs_error < 1e-6);
    assert!((fid.cosine_similarity - 1.0).abs() < 1e-6);
}

#[test]
fn chunked_prefill_continuation_preserves_logical_positions_bit_exact() {
    let model = build_test_model("hanzo-zen6-chunked-prefill@v1");
    let corpus = RepoCorpus::synthetic_for_tests(1346);
    let embedder = CorpusEmbedder::new(model.d_model(), 16, 1346);
    let workloads = build_five_workloads(&corpus, &embedder, 256, 4, 28, 1346);
    let niah_ep = &workloads
        .iter()
        .find(|(name, _)| name == "needle_in_a_haystack")
        .expect("niah workload")
        .1[2];

    let mut single_cache = DualKvCache::new_full(model.identity(), KvDtype::F16).unwrap();
    let out_single = model
        .forward_sequence(&niah_ep.embeddings, &mut single_cache)
        .unwrap();

    // Split 256-token NIAH sequence into 240-token prefix + 16-token continuation.
    let mut chunked_cache = DualKvCache::new_full(model.identity(), KvDtype::F16).unwrap();
    let _ = model
        .forward_sequence(&niah_ep.embeddings[..240], &mut chunked_cache)
        .unwrap();
    assert_eq!(chunked_cache.seq_len(), 240);
    let out_chunked = model
        .forward_sequence(&niah_ep.embeddings[240..], &mut chunked_cache)
        .unwrap();
    assert_eq!(chunked_cache.seq_len(), 256);

    assert_eq!(out_single, out_chunked);
    let query_input = &niah_ep.embeddings[niah_ep.embeddings.len() - 1];
    let (pred_single, _) =
        embedder.predict_from_recalled_delta(&out_single, query_input, niah_ep.target_candidate);
    let (pred_chunked, _) =
        embedder.predict_from_recalled_delta(&out_chunked, query_input, niah_ep.target_candidate);
    assert_eq!(pred_single, niah_ep.target_candidate);
    assert_eq!(pred_chunked, niah_ep.target_candidate);
}

#[test]
fn cross_model_cache_is_rejected_fail_closed_before_mutation() {
    let model_a = build_test_model("hanzo-zen6-exec-model-a@v1");
    let model_b = build_test_model("hanzo-zen6-exec-model-b@v1");
    let mut cache_b = DualKvCache::new_full(model_b.identity(), KvDtype::F16).unwrap();
    let emb = vec![0.1_f32; model_a.d_model()];

    let err_step = model_a
        .forward_step(0, &emb, &mut cache_b)
        .expect_err("cache bound to model B must be rejected by model A forward_step");
    assert!(matches!(err_step, DuoAttnError::ModelMismatch { .. }));
    assert_eq!(
        cache_b.seq_len(),
        0,
        "rejected forward_step must not mutate cache"
    );

    let err_seq = model_a
        .forward_sequence(&[emb], &mut cache_b)
        .expect_err("cache bound to model B must be rejected by model A forward_sequence");
    assert!(matches!(err_seq, DuoAttnError::ModelMismatch { .. }));
    assert_eq!(
        cache_b.seq_len(),
        0,
        "rejected forward_sequence must not mutate cache"
    );
}

#[test]
fn small_head_model_identity_constructs_and_runs_without_panicking() {
    // Exact geometry flagged in review: d_model = 4 * 8 = 32.
    let small_id = ModelIdentity::pure_attention("m", 1, 4, 2, 8).unwrap();
    let model = DualAttentionModel::from_identity(small_id.clone(), 0.25)
        .expect("valid small-head model must construct without panicking");
    let mut cache = DualKvCache::new_full(&small_id, KvDtype::F16).unwrap();
    let emb = vec![0.25_f32; model.d_model()];
    let out = model.forward_step(0, &emb, &mut cache).unwrap();
    assert_eq!(out.len(), 32);
    assert!(out.iter().all(|v| v.is_finite()));

    // Ultra-tiny d_model < 4 must return Err(InvalidConfig) cleanly instead of panicking.
    let tiny_id = ModelIdentity::pure_attention("tiny", 1, 1, 1, 2).unwrap();
    let err = DualAttentionModel::from_identity(tiny_id, 0.5)
        .expect_err("d_model=2 cannot partition key/value/local subspaces");
    assert!(matches!(err, DuoAttnError::InvalidConfig(_)));
}

#[test]
fn lsa_bucket_sparse_regime_executes_real_kernel_and_accounts_centroid_overhead() {
    let model = build_test_model("hanzo-zen6-lsa-measured@v1");
    let corpus = RepoCorpus::synthetic_for_tests(1346);
    let embedder = CorpusEmbedder::new(model.d_model(), 16, 1346);
    let workloads = build_five_workloads(&corpus, &embedder, 256, 4, 28, 1346);
    let (wl_name, episodes) = &workloads[2];

    let (full_res, full_refs) =
        evaluate_workload_regime(wl_name, &model, &embedder, episodes, &[], || {
            DualKvCache::new_full(model.identity(), KvDtype::F16)
        })
        .unwrap();
    let (lsa_res, _) =
        evaluate_workload_regime(wl_name, &model, &embedder, episodes, &full_refs, || {
            DualKvCache::new_lsa_bucket_sparse(model.identity(), 4, 28, 8, 32, KvDtype::F16)
        })
        .unwrap();

    assert_eq!(lsa_res.regime, "lsa_existing_hanzo");
    assert!(lsa_res.peak_resident_kv_bytes > full_res.peak_resident_kv_bytes);
    assert!(lsa_res.kv_memory_reduction_x < 1.0);
    assert!(lsa_res.cosine_vs_full.is_finite() && lsa_res.cosine_vs_full > 0.5);
}
