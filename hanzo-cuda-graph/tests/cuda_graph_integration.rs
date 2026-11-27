// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive integration test suite for `hanzo-cuda-graph` (HIP-1346 / #54).

use hanzo_cuda_graph::bench::{evaluate_end_to_end_gate, run_full_benchmark, CONTEXT_SWEEP_LENS};
use hanzo_cuda_graph::bucket::{
    BucketPolicy, CudaDecodeGraphKey, DEFAULT_BATCH_BUCKETS, DEFAULT_CONTEXT_BUCKETS,
    DEFAULT_VERIFY_BUCKETS,
};
use hanzo_cuda_graph::capture::{
    BreakableExecutionPlan, CaptureError, CapturedGraph, CapturedNode, DecodeGraphCache,
    GraphBreakReason, GraphPieceKind, CUDA_DECODE_GRAPH_CACHE_CAPACITY,
};
use hanzo_cuda_graph::lifetime::{
    BufferAccessKind, DeviceLocation, GraphMemoryPool, GraphSanitizer, PersistentGraphBuffers,
    PoolBlockState, SanitizerViolation,
};
use hanzo_cuda_graph::lsa_exec::{
    argmax, max_abs_logits_diff, run_sanitizer_fault_suite, DeterministicRng, HybridLsaConfig,
    HybridLsaEngine, HybridLsaState,
};
use hanzo_cuda_graph::model::{project_decode_step, HardwareSpec, HybridModelSpec, QuantFormat};
use hanzo_cuda_graph::sha256::sha256_hex;

#[test]
fn sha256_matches_nist_test_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn hip_1346_shared_hybrid_matrix_exact_geometry() {
    let models = HybridModelSpec::canonical_matrix();
    let expected_kv_heads = [2usize, 4, 4];
    let expected_gdn_mib = [27.0f64, 45.0, 192.0];
    let expected_attn_layers = [9usize, 10, 16];
    let expected_gdn_layers = [27usize, 30, 48];

    for i in 0..3 {
        assert_eq!(models[i].num_kv_heads, expected_kv_heads[i]);
        assert_eq!(models[i].num_attention_layers, expected_attn_layers[i]);
        assert_eq!(models[i].num_gdn_layers, expected_gdn_layers[i]);
        assert!((models[i].exact_gdn_state_mib() - expected_gdn_mib[i]).abs() < 1e-9);
        assert!((models[i].gdn_state_mib - expected_gdn_mib[i]).abs() < 1e-9);
    }
}

#[test]
fn resident_vs_streamed_weights_separation() {
    for model in HybridModelSpec::canonical_matrix() {
        let embed_table_bytes = model.embedding_table_params() * 2;
        assert_eq!(
            embed_table_bytes,
            (model.vocab_size as u64) * (model.d_model as u64) * 2
        );
        assert_eq!(model.embedding_row_bytes_bf16(), (model.d_model as u64) * 2);

        for quant in [QuantFormat::Bf16, QuantFormat::Int4Awq] {
            let resident = model.resident_weight_bytes(quant);
            let streamed = model.streamed_weight_bytes(quant);
            // Resident VRAM includes the full input embedding table; streamed per-step weights omit it.
            assert_eq!(resident - streamed, embed_table_bytes);
        }

        // INT4-AWQ must be significantly smaller than BF16 in both resident and streamed bytes.
        assert!(
            model.streamed_weight_bytes(QuantFormat::Int4Awq)
                < model.streamed_weight_bytes(QuantFormat::Bf16) / 2
        );
    }
}

#[test]
fn persistent_buffers_preserve_addresses_and_skip_unchanged_block_tables() {
    let mut pool = GraphMemoryPool::new();
    let mut buffers = PersistentGraphBuffers::allocate(&mut pool, 2, 1, 8_192, 16, 27, 64)
        .expect("allocate persistent buffers");

    let addr_before: Vec<u64> = buffers
        .all_buffer_ids()
        .iter()
        .map(|id| pool.get(*id).expect("exists").virtual_addr)
        .collect();

    let sig = [0x1111_u64, 0x2222_u64];
    let first = buffers
        .update_in_place(&mut pool, 2, 1, 2 * 512, 27, Some(&sig))
        .expect("first update");
    assert!(!first.block_table_skipped);

    let second = buffers
        .update_in_place(&mut pool, 2, 1, 2 * 512, 27, Some(&sig))
        .expect("second update");
    assert!(second.block_table_skipped);
    assert!(second.bytes_copied < first.bytes_copied);
    assert_eq!(buffers.block_table_copy_skips, 1);

    let sig_changed = [0x1111_u64, 0x3333_u64];
    let third = buffers
        .update_in_place(&mut pool, 2, 1, 2 * 512, 27, Some(&sig_changed))
        .expect("third update");
    assert!(!third.block_table_skipped);

    let addr_after: Vec<u64> = buffers
        .all_buffer_ids()
        .iter()
        .map(|id| pool.get(*id).expect("exists").virtual_addr)
        .collect();
    assert_eq!(addr_before, addr_after);

    buffers.release(&mut pool);
    assert_eq!(pool.live_leaked_bytes(), 0);
}

#[test]
fn sanitizer_catches_use_after_free_and_poisoned_buffer() {
    let mut pool = GraphMemoryPool::new();
    let buf = pool
        .allocate_persistent("input_ids", 256, true)
        .expect("alloc");
    let key = CudaDecodeGraphKey {
        device: DeviceLocation::Cuda { gpu_id: 0 },
        batch_bucket: 1,
        verify_width_bucket: 1,
        max_context_len: 8_192,
        quant: QuantFormat::Bf16,
        piece_tag: "uaf_test",
    };

    let mut graph = CapturedGraph::capture(&mut pool, key, 4096, 1.0, |p| {
        let r = p.make_ref(buf, 0, 256, BufferAccessKind::Read)?;
        Ok(vec![CapturedNode {
            node_id: 0,
            kernel_name: "embed_lookup".to_string(),
            piece_kind: GraphPieceKind::LsaRecurrentAndMlp,
            bindings: vec![r],
        }])
    })
    .expect("capture");

    assert!(graph.validate_and_replay(&pool).is_ok());
    // Free while graph holds refcount -> transitions to Poisoned.
    assert!(pool.free_buffer(buf));
    assert_eq!(pool.get(buf).unwrap().state, PoolBlockState::Poisoned);

    let err = graph.validate_and_replay(&pool).unwrap_err();
    assert!(matches!(
        err,
        CaptureError::SanitizerFailed(ref vs)
            if vs.iter().any(|v| matches!(v, SanitizerViolation::PoisonedBufferAccess { .. }))
    ));

    graph.teardown(&mut pool);
}

#[test]
fn sanitizer_catches_generation_mismatch_on_reallocation() {
    let mut pool = GraphMemoryPool::new();
    let buf = pool
        .allocate_persistent("slot_mapping", 256, true)
        .expect("alloc");
    let key = CudaDecodeGraphKey {
        device: DeviceLocation::Cuda { gpu_id: 0 },
        batch_bucket: 1,
        verify_width_bucket: 1,
        max_context_len: 8_192,
        quant: QuantFormat::Bf16,
        piece_tag: "gen_test",
    };

    let mut graph = CapturedGraph::capture(&mut pool, key, 4096, 1.0, |p| {
        let r = p.make_ref(buf, 0, 256, BufferAccessKind::Read)?;
        Ok(vec![CapturedNode {
            node_id: 0,
            kernel_name: "kv_scatter".to_string(),
            piece_kind: GraphPieceKind::PagedAttentionAndVerify,
            bindings: vec![r],
        }])
    })
    .expect("capture");

    let new_gen = pool.reallocate_buffer(buf, 512).expect("reallocate");
    assert_eq!(new_gen.0, 2);

    let err = graph.validate_and_replay(&pool).unwrap_err();
    assert!(matches!(
        err,
        CaptureError::SanitizerFailed(ref vs)
            if vs.iter().any(|v| matches!(v, SanitizerViolation::GenerationMismatch { .. }))
    ));

    graph.teardown(&mut pool);
    pool.free_buffer(buf);
    assert_eq!(pool.live_leaked_bytes(), 0);
}

#[test]
fn sanitizer_catches_out_of_bounds_buffer_access() {
    let mut pool = GraphMemoryPool::new();
    let buf = pool
        .allocate_persistent("context_lens", 64, true)
        .expect("alloc");
    let bad_ref = pool
        .make_ref(buf, 32, 64, BufferAccessKind::Read)
        .expect("ref");
    let err = GraphSanitizer::validate_replay(&pool, &[(0, &[bad_ref])]).unwrap_err();
    assert!(err.iter().any(|v| matches!(
        v,
        SanitizerViolation::OutOfBounds {
            access_end: 96,
            capacity_bytes: 64,
            ..
        }
    )));
}

#[test]
fn sanitizer_catches_uninitialized_read() {
    let mut pool = GraphMemoryPool::new();
    let buf = pool
        .allocate_persistent("uninit_state", 128, false)
        .expect("alloc");
    let r = pool
        .make_ref(buf, 0, 128, BufferAccessKind::Read)
        .expect("ref");
    let err = GraphSanitizer::validate_replay(&pool, &[(0, &[r])]).unwrap_err();
    assert!(err
        .iter()
        .any(|v| matches!(v, SanitizerViolation::UninitializedRead { .. })));

    // Once initialized via copy_into_persistent, initcheck passes cleanly.
    pool.copy_into_persistent(buf, 128).expect("copy");
    let r2 = pool
        .make_ref(buf, 0, 128, BufferAccessKind::Read)
        .expect("ref2");
    assert!(GraphSanitizer::validate_replay(&pool, &[(0, &[r2])]).is_ok());
}

#[test]
fn sanitizer_catches_leaked_capture_scratch_past_end_capture() {
    let mut pool = GraphMemoryPool::new();
    let key = CudaDecodeGraphKey {
        device: DeviceLocation::Cuda { gpu_id: 0 },
        batch_bucket: 1,
        verify_width_bucket: 1,
        max_context_len: 8_192,
        quant: QuantFormat::Bf16,
        piece_tag: "leak_test",
    };

    let res = CapturedGraph::capture(&mut pool, key, 4096, 1.0, |p| {
        // Deliberately forget to free capture-local scratch before end_capture!
        let _leaked = p.allocate_capture_scratch("forgotten_tmp_tensor", 4096)?;
        Ok(Vec::new())
    });

    assert!(matches!(
        res,
        Err(CaptureError::SanitizerFailed(ref vs))
            if vs.iter().any(|v| matches!(v, SanitizerViolation::LeakedCaptureScratch { byte_len: 4096, .. }))
    ));
}

#[test]
fn sanitizer_catches_dynamic_allocation_during_capture() {
    let mut pool = GraphMemoryPool::new();
    let key = CudaDecodeGraphKey {
        device: DeviceLocation::Cuda { gpu_id: 0 },
        batch_bucket: 1,
        verify_width_bucket: 1,
        max_context_len: 8_192,
        quant: QuantFormat::Bf16,
        piece_tag: "dyn_alloc_test",
    };

    let res = CapturedGraph::capture(&mut pool, key, 4096, 1.0, |p| {
        let _ = p.allocate_unpooled_dynamic("cudaMalloc_transient", 8192)?;
        Ok(Vec::new())
    });

    assert!(matches!(
        res,
        Err(CaptureError::DynamicAllocationForbidden(_))
    ));
    assert!(!pool.is_capturing());
}

#[test]
fn sanitizer_catches_host_sync_during_capture() {
    let mut pool = GraphMemoryPool::new();
    let key = CudaDecodeGraphKey {
        device: DeviceLocation::Cuda { gpu_id: 0 },
        batch_bucket: 1,
        verify_width_bucket: 1,
        max_context_len: 8_192,
        quant: QuantFormat::Bf16,
        piece_tag: "host_sync_test",
    };

    let res = CapturedGraph::capture(&mut pool, key, 4096, 1.0, |p| {
        p.check_host_sync("cudaStreamSynchronize")?;
        Ok(Vec::new())
    });

    assert!(matches!(res, Err(CaptureError::HostSyncForbidden(_))));
    assert!(!pool.is_capturing());
}

#[test]
fn sanitizer_catches_gdn_state_kv_alias_race() {
    let mut pool = GraphMemoryPool::new();
    let gdn = pool
        .allocate_persistent("gdn_recurrent_state", 8192, true)
        .expect("alloc gdn");
    let kv = pool
        .allocate_persistent("attn_kv_scratch", 8192, true)
        .expect("alloc kv");
    let base = pool.get(gdn).unwrap().virtual_addr;

    let mut r_gdn = pool
        .make_ref(gdn, 0, 8192, BufferAccessKind::ReadWrite)
        .unwrap();
    let mut r_kv = pool.make_ref(kv, 0, 4096, BufferAccessKind::Write).unwrap();
    r_gdn.virtual_addr_override = Some(base);
    r_kv.virtual_addr_override = Some(base + 1024);

    let err = GraphSanitizer::validate_replay(&pool, &[(0, &[r_gdn, r_kv])]).unwrap_err();
    assert!(err.iter().any(|v| matches!(
        v,
        SanitizerViolation::StateAliasRace {
            overlap_bytes: 4096,
            ..
        }
    )));
}

#[test]
fn sanitizer_fault_suite_catches_all_six_injected_bugs() {
    let summary = run_sanitizer_fault_suite();
    assert_eq!(summary.total_faults_injected, 6);
    assert_eq!(summary.total_faults_caught, 6);
    for (name, caught, diag) in &summary.cases {
        assert!(*caught, "fault {name} was missed: {diag}");
    }
}

#[test]
fn bounded_shape_buckets_cover_variable_batches_verify_widths_and_contexts() {
    let policy = BucketPolicy::bounded_default();
    assert_eq!(DEFAULT_BATCH_BUCKETS, [1, 2, 4, 8, 16, 32]);
    assert_eq!(DEFAULT_VERIFY_BUCKETS, [1, 2, 4, 8]);
    assert_eq!(DEFAULT_CONTEXT_BUCKETS, [8_192, 16_384, 30_720, 65_536]);

    let b3 = policy.resolve(3, 1, 30_000).expect("bucket");
    assert_eq!(b3.batch_bucket, 4);
    assert_eq!(b3.verify_width_bucket, 1);
    assert_eq!(b3.context_bucket, 30_720);
    assert_eq!(b3.padding_lanes(), 1);
    assert_eq!(b3.lane_mask(), vec![true, true, true, false]);

    let spec_b5 = policy.resolve(5, 5, 12_000).expect("spec bucket");
    assert_eq!(spec_b5.batch_bucket, 8);
    assert_eq!(spec_b5.verify_width_bucket, 8);
    assert_eq!(spec_b5.context_bucket, 16_384);

    // Out-of-range batch or context falls back (None -> breakable eager boundary).
    assert!(policy.resolve(64, 1, 8_192).is_none());
    assert!(policy.resolve(1, 1, 131_072).is_none());
}

#[test]
fn breakable_execution_plan_handles_prefill_speculative_profiler_and_overflow() {
    let policy = BucketPolicy::bounded_default();

    let std_decode = BreakableExecutionPlan::plan_step(&policy, 1, 1, 30_000, false, false, false);
    assert!(std_decode.uses_graph_capture());
    assert_eq!(std_decode.capturable_piece_count(), 2);
    assert_eq!(std_decode.primary_break, GraphBreakReason::None);

    let spec_verify = BreakableExecutionPlan::plan_step(&policy, 1, 4, 30_000, false, true, false);
    assert!(spec_verify.uses_graph_capture());
    assert_eq!(spec_verify.capturable_piece_count(), 2);
    assert_eq!(
        spec_verify.primary_break,
        GraphBreakReason::SpeculativeRejectionSampling
    );

    let prefill = BreakableExecutionPlan::plan_step(&policy, 1, 1, 4_096, true, false, false);
    assert!(!prefill.uses_graph_capture());
    assert_eq!(prefill.primary_break, GraphBreakReason::PrefillDynamicShape);

    let prof = BreakableExecutionPlan::plan_step(&policy, 1, 1, 8_192, false, false, true);
    assert!(!prof.uses_graph_capture());
    assert_eq!(
        prof.primary_break,
        GraphBreakReason::CaptureDisabledByProfiler
    );

    let overflow_b = BreakableExecutionPlan::plan_step(&policy, 48, 1, 8_192, false, false, false);
    assert!(!overflow_b.uses_graph_capture());
    assert_eq!(
        overflow_b.primary_break,
        GraphBreakReason::BatchExceedsMaxBucket
    );
}

#[test]
fn lru_decode_graph_cache_evicts_cleanly_with_zero_leaked_bytes() {
    assert_eq!(CUDA_DECODE_GRAPH_CACHE_CAPACITY, 32);
    let mut pool = GraphMemoryPool::new();
    let mut cache = DecodeGraphCache::new(3);

    for i in 0..7usize {
        let key = CudaDecodeGraphKey {
            device: DeviceLocation::Cuda { gpu_id: 0 },
            batch_bucket: 1 << (i % 4),
            verify_width_bucket: 1,
            max_context_len: 8_192 * (1 + (i % 3)),
            quant: QuantFormat::Bf16,
            piece_tag: "lru_test",
        };
        let buffers =
            PersistentGraphBuffers::allocate(&mut pool, 1, 1, 8_192, 16, 3, 64).expect("alloc");
        let graph = CapturedGraph::capture(&mut pool, key.clone(), 16_384, 1.2, |p| {
            let r = p.make_ref(buffers.input_ids, 0, 4, BufferAccessKind::Read)?;
            Ok(vec![CapturedNode {
                node_id: 0,
                kernel_name: "k".to_string(),
                piece_kind: GraphPieceKind::LsaRecurrentAndMlp,
                bindings: vec![r],
            }])
        })
        .expect("capture");
        cache.insert(&mut pool, key, graph, buffers);
    }

    let st = cache.stats();
    assert_eq!(st.live_entries, 3);
    assert!(st.evictions >= 4);
    assert_eq!(st.current_graph_pool_bytes, 3 * 16_384);
    assert_eq!(pool.live_private_graph_pool_bytes(), 3 * 16_384);

    cache.clear(&mut pool);
    assert_eq!(cache.stats().live_entries, 0);
    assert_eq!(cache.stats().current_graph_pool_bytes, 0);
    assert_eq!(pool.live_leaked_bytes(), 0);
}

#[test]
fn eager_vs_graph_single_step_and_multi_step_bit_exact() {
    let cfg = HybridLsaConfig::default();
    let engine = HybridLsaEngine::new(cfg.clone(), 1346);
    let policy = BucketPolicy::bounded_default();
    let mut rng = DeterministicRng::new(777);

    let mut s_eager = HybridLsaState::new(&cfg, 4);
    let mut s_graph = HybridLsaState::new(&cfg, 4);
    let mut pool = GraphMemoryPool::new();
    let mut cache = DecodeGraphCache::default();

    for _ in 0..24 {
        let toks: Vec<u32> = (0..4)
            .map(|_| (rng.next_u64() as u32) % (cfg.vocab_size as u32))
            .collect();
        let l_e = engine.step_eager(&mut s_eager, &toks);
        let out_g = engine
            .step_graph_captured(&mut s_graph, &mut pool, &mut cache, &policy, &toks)
            .expect("graph step");
        assert_eq!(max_abs_logits_diff(&l_e, &out_g.logits), 0.0);
        assert_eq!(out_g.sanitizer_report.unwrap().violation_count, 0);
    }

    assert_eq!(s_eager.max_abs_gdn_diff(&s_graph), 0.0);
    assert_eq!(s_eager.max_abs_kv_diff(&s_graph), 0.0);

    cache.clear(&mut pool);
    assert_eq!(pool.live_leaked_bytes(), 0);
}

#[test]
fn variable_batch_padded_lanes_never_corrupt_gdn_or_kv_state() {
    let cfg = HybridLsaConfig::default();
    let engine = HybridLsaEngine::new(cfg.clone(), 2026);
    let policy = BucketPolicy::bounded_default();

    // Allocate 8 sequence slots; execute B=3 (padded to 4) and B=5 (padded to 8),
    // verifying that inactive slots [5..8] stay strictly zero/untouched AND active
    // slots match unbucketed eager execution bit-for-bit!
    let mut s_eager = HybridLsaState::new(&cfg, 8);
    let mut s_graph = HybridLsaState::new(&cfg, 8);
    let mut pool = GraphMemoryPool::new();
    let mut cache = DecodeGraphCache::default();

    let shapes = [3usize, 5, 1, 3, 2, 5, 3];
    for (step, &b) in shapes.iter().enumerate() {
        let toks: Vec<u32> = (0..b)
            .map(|i| ((step * 11 + i * 5) % cfg.vocab_size) as u32)
            .collect();
        let l_e = engine.step_eager(&mut s_eager, &toks);
        let out_g = engine
            .step_graph_captured(&mut s_graph, &mut pool, &mut cache, &policy, &toks)
            .expect("graph step");
        assert_eq!(out_g.logits.len(), b);
        assert_eq!(max_abs_logits_diff(&l_e, &out_g.logits), 0.0);
    }

    // Slots 5, 6, 7 were NEVER active (only padded lanes in B=5 -> bucket 8): must stay 100% untouched!
    for seq in &s_graph.sequences[5..8] {
        assert_eq!(seq.seq_len, 0);
        for layer in &seq.gdn_states {
            assert!(layer.iter().all(|&v| v == 0.0));
        }
    }

    assert_eq!(s_eager.max_abs_gdn_diff(&s_graph), 0.0);
    assert_eq!(s_eager.max_abs_kv_diff(&s_graph), 0.0);

    cache.clear(&mut pool);
    assert_eq!(pool.live_leaked_bytes(), 0);
}

#[test]
fn speculative_verify_k4_all_acceptance_counts_bit_exact_rollback() {
    let cfg = HybridLsaConfig::default();
    let engine = HybridLsaEngine::new(cfg.clone(), 9001);
    let policy = BucketPolicy::bounded_default();
    let mut pool = GraphMemoryPool::new();
    let mut cache = DecodeGraphCache::default();

    for expected_accept in 0..=3usize {
        let mut s_eager = HybridLsaState::new(&cfg, 1);
        let mut s_spec = HybridLsaState::new(&cfg, 1);

        let mut anchor = 1u32;
        for t in 0..8u32 {
            let tok = (t * 9 + expected_accept as u32) % (cfg.vocab_size as u32);
            let _ = engine.forward_single_token(&mut s_eager.sequences[0], tok);
            let _ = engine.forward_single_token(&mut s_spec.sequences[0], tok);
            anchor = tok;
        }

        // Ground-truth continuation of 3 draft tokens (total K = 4 tokens verified in 1 graph replay).
        let mut probe = s_eager.sequences[0].clone();
        let mut cur = anchor;
        let mut gt = Vec::new();
        for _ in 0..3 {
            let logits = engine.forward_single_token(&mut probe, cur);
            cur = argmax(&logits);
            gt.push(cur);
        }

        let drafts: Vec<u32> = gt
            .iter()
            .enumerate()
            .map(|(i, &tok)| {
                if i < expected_accept {
                    tok
                } else {
                    (tok + 3) % (cfg.vocab_size as u32)
                }
            })
            .collect();

        let out = engine
            .verify_speculative_graph(
                &mut s_spec,
                0,
                &mut pool,
                &mut cache,
                &policy,
                anchor,
                &drafts,
            )
            .expect("spec verify");
        assert_eq!(out.accepted_drafts, expected_accept);
        assert_eq!(out.bucket.verify_width_bucket, 4);
        assert_eq!(out.sanitizer_report.violation_count, 0);

        let mut last_eager = engine.forward_single_token(&mut s_eager.sequences[0], anchor);
        for &d in drafts.iter().take(expected_accept) {
            last_eager = engine.forward_single_token(&mut s_eager.sequences[0], d);
        }

        assert_eq!(max_abs_logits_diff(&[last_eager], &[out.final_logits]), 0.0);
        assert_eq!(s_eager.max_abs_gdn_diff(&s_spec), 0.0);
        assert_eq!(s_eager.max_abs_kv_diff(&s_spec), 0.0);
    }

    cache.clear(&mut pool);
    assert_eq!(pool.live_leaked_bytes(), 0);
}

#[test]
fn speculative_verify_padded_width_bucket_k5_to_k8_bit_exact() {
    let cfg = HybridLsaConfig::default();
    let engine = HybridLsaEngine::new(cfg.clone(), 4242);
    let policy = BucketPolicy::bounded_default();
    let mut pool = GraphMemoryPool::new();
    let mut cache = DecodeGraphCache::default();

    let mut s_eager = HybridLsaState::new(&cfg, 1);
    let mut s_spec = HybridLsaState::new(&cfg, 1);
    let anchor = 5u32;

    // 4 draft tokens -> K = 5 -> padded into verify_width_bucket = 8 (3 masked padding slots).
    let mut probe = s_eager.sequences[0].clone();
    let mut cur = anchor;
    let mut drafts = Vec::new();
    for step in 0..4 {
        let logits = engine.forward_single_token(&mut probe, cur);
        let next = argmax(&logits);
        if step < 2 {
            drafts.push(next);
            cur = next;
        } else {
            drafts.push((next + 5) % (cfg.vocab_size as u32));
            cur = drafts[step];
        }
    }

    let out = engine
        .verify_speculative_graph(
            &mut s_spec,
            0,
            &mut pool,
            &mut cache,
            &policy,
            anchor,
            &drafts,
        )
        .expect("padded K=5->8 spec verify");
    assert_eq!(out.bucket.verify_width_bucket, 8);
    assert_eq!(out.accepted_drafts, 2);

    let mut last_eager = engine.forward_single_token(&mut s_eager.sequences[0], anchor);
    for &d in drafts.iter().take(2) {
        last_eager = engine.forward_single_token(&mut s_eager.sequences[0], d);
    }
    assert_eq!(max_abs_logits_diff(&[last_eager], &[out.final_logits]), 0.0);
    assert_eq!(s_eager.max_abs_gdn_diff(&s_spec), 0.0);
    assert_eq!(s_eager.max_abs_kv_diff(&s_spec), 0.0);

    cache.clear(&mut pool);
    assert_eq!(pool.live_leaked_bytes(), 0);
}

#[test]
fn context_sweep_proves_lower_30k_decode_latency_across_all_models_and_quants() {
    let hw = HardwareSpec::h100_sxm5_80gb();
    for model in HybridModelSpec::canonical_matrix() {
        for quant in [QuantFormat::Bf16, QuantFormat::Int4Awq] {
            for &ctx in &CONTEXT_SWEEP_LENS {
                let eager = project_decode_step(&model, &hw, quant, 1, 1, ctx, false, 1, false);
                let graph = project_decode_step(&model, &hw, quant, 1, 1, ctx, true, 2, true);
                assert!(
                    graph.p50_ms < eager.p50_ms,
                    "{} {} ctx={ctx}: graph p50 ({}) must beat eager p50 ({})",
                    model.label,
                    quant.as_str(),
                    graph.p50_ms,
                    eager.p50_ms
                );
                assert!(
                    graph.p95_ms < eager.p95_ms,
                    "{} {} ctx={ctx}: graph p95 ({}) must beat eager p95 ({})",
                    model.label,
                    quant.as_str(),
                    graph.p95_ms,
                    eager.p95_ms
                );
            }
        }
    }
}

#[test]
fn scorecard_20_turn_accepts_breakable_cuda_graphs_and_spec_verify() {
    let report = run_full_benchmark(1346, "testcommit".to_string());
    assert_eq!(report.context_sweep.len(), 3 * 2 * 4);
    assert_eq!(report.scorecard.len(), 3 * 2 * 3);

    for row in &report.scorecard {
        assert!(
            row.gate.is_accepted(),
            "scorecard row {} {} {} unexpectedly rejected: {:?}",
            row.model,
            row.quant,
            row.regime,
            row.gate
        );
        assert_eq!(row.sanitizer_violations, 0);
        assert_eq!(row.output_correctness_pct, 100.0);
        assert_eq!(row.quality_regression_pp, 0.0);
        if row.regime != "eager_no_graphs" {
            assert!(
                row.wall_time_speedup_vs_eager > 1.0,
                "{} {} {} must beat eager 20-turn wall time",
                row.model,
                row.quant,
                row.regime
            );
        }
    }
}

#[test]
fn all_three_negative_regimes_are_explicitly_rejected_by_gate() {
    let report = run_full_benchmark(1346, "testcommit".to_string());
    assert_eq!(report.negative_regimes.len(), 3);
    let expected = [
        "neg_unbounded_shape_graph_explosion",
        "neg_graph_capture_without_lifetime_fix",
        "neg_oversized_padding_bucket_waste",
    ];
    for (neg, exp_name) in report.negative_regimes.iter().zip(expected.iter()) {
        assert_eq!(neg.regime, *exp_name);
        assert!(
            !neg.gate.is_accepted(),
            "negative regime {} MUST be rejected",
            neg.regime
        );
    }

    // Also directly verify gate rejection when a microbenchmark wins ITL but regresses wall time.
    let micro_only = evaluate_end_to_end_gate(
        25.0, // worse 20-turn wall time
        20.0, // eager baseline
        2.0,  // faster ITL microbenchmark
        3.0,  // eager ITL
        20_000.0, 81_920.0, 64.0, 100.0, 0.0, 0,
    );
    assert!(!micro_only.is_accepted());
}

#[test]
fn benchmark_report_json_structure_and_determinism() {
    let report = run_full_benchmark(1346, "abc123def456".to_string());
    assert_eq!(report.cpu_measured.max_abs_logit_err, 0.0);
    assert_eq!(report.cpu_measured.max_abs_gdn_state_err, 0.0);
    assert_eq!(report.cpu_measured.max_abs_kv_err, 0.0);
    assert_eq!(report.cpu_measured.spec_max_abs_logit_err, 0.0);
    assert_eq!(report.cpu_measured.spec_max_abs_gdn_state_err, 0.0);
    assert_eq!(report.cpu_measured.spec_max_abs_kv_err, 0.0);
    assert_eq!(report.cpu_measured.fixed_path_sanitizer_violations, 0);
    assert_eq!(report.cpu_measured.post_teardown_leaked_bytes, 0);
    assert_eq!(
        report
            .cpu_measured
            .sanitizer_fault_suite
            .total_faults_caught,
        6
    );

    let json = report.to_json();
    assert!(json.contains("\"issue\": \"hanzoai/engine#54\""));
    assert!(json.contains("\"cpu_directly_measured\""));
    assert!(json.contains("\"analytical_gpu_h100_sxm5\""));
    assert!(json.contains("\"neg_unbounded_shape_graph_explosion\""));
    assert!(json.contains("\"neg_graph_capture_without_lifetime_fix\""));
    assert!(json.contains("\"neg_oversized_padding_bucket_waste\""));
}
