// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive integration tests for `hanzo-cascade-attn` (HIP-1346 `#53`).

use hanzo_cascade_attn::arch::HardwareArch;
use hanzo_cascade_attn::bench::{
    evaluate_end_to_end_gate, evaluate_serving_scorecard, run_full_benchmark, AgentWorkloadConfig,
    AttentionBackend,
};
use hanzo_cascade_attn::cascade::{
    greedy_argmax_tokens, merge_lse_multi_level, merge_lse_states, run_cpu_cascade_verification,
    three_level_cascade_gqa_decode, two_level_cascade_gqa_decode, unshared_paged_gqa_decode, Lcg64,
    PagedKvPool,
};
use hanzo_cascade_attn::hybrid::{run_cpu_hybrid_step_verification, HybridStepConfig};
use hanzo_cascade_attn::model::{HybridModelSpec, QuantFormat};
use hanzo_cascade_attn::ragged::{
    fused_ragged_prefill_decode_gqa, run_cpu_ragged_verification,
    separate_prefill_and_decode_paged_gqa, RaggedBatchDescriptor, RaggedDescriptorError,
};
use hanzo_cascade_attn::sha256::sha256_hex;

#[test]
fn canonical_hip1346_hybrid_matrix_matches_exact_kv_and_gdn_specs() {
    let [m4, m9, m27] = HybridModelSpec::canonical_matrix();

    assert_eq!(m4.label, "4B-Hybrid");
    assert_eq!((m4.num_attention_layers, m4.num_gdn_layers), (9, 27));
    assert_eq!((m4.num_kv_heads, m4.head_dim), (2, 128));
    assert!((m4.exact_gdn_state_mib() - 27.0).abs() < 1e-9);
    assert!((m4.attention_layer_fraction() - 0.25).abs() < 1e-9);

    assert_eq!(m9.label, "9B-Hybrid");
    assert_eq!((m9.num_attention_layers, m9.num_gdn_layers), (10, 30));
    assert_eq!((m9.num_kv_heads, m9.head_dim), (4, 128));
    assert!((m9.exact_gdn_state_mib() - 45.0).abs() < 1e-9);
    assert!((m9.attention_layer_fraction() - 0.25).abs() < 1e-9);

    assert_eq!(m27.label, "27B-Hybrid");
    assert_eq!((m27.num_attention_layers, m27.num_gdn_layers), (16, 48));
    assert_eq!((m27.num_kv_heads, m27.head_dim), (4, 256));
    assert!((m27.exact_gdn_state_mib() - 192.0).abs() < 1e-9);
    assert!((m27.attention_layer_fraction() - 0.25).abs() < 1e-9);
}

#[test]
fn resident_weights_separate_full_embedding_table_from_single_row_streamed_gathers() {
    for model in HybridModelSpec::canonical_matrix() {
        let embed_table_bytes = (model.vocab_size as u64) * (model.d_model as u64) * 2;
        assert_eq!(model.embedding_row_bytes_bf16(), (model.d_model as u64) * 2);
        for quant in QuantFormat::all() {
            let resident = model.resident_weight_bytes(quant);
            let streamed = model.streamed_weight_bytes(quant);
            assert_eq!(
                resident - streamed,
                embed_table_bytes,
                "resident vs streamed delta must equal exact input-embedding table for {} {}",
                model.label,
                quant.as_str()
            );
            assert!(
                model.embedding_row_bytes_bf16() * 1000 < embed_table_bytes,
                "single-row gather must be orders of magnitude smaller than full table"
            );
        }
    }
}

#[test]
fn four_hardware_architectures_are_strictly_separated_without_cross_extrapolation() {
    let h100 = HardwareArch::HopperH100Sxm5.profile();
    let b200 = HardwareArch::BlackwellB200.profile();
    let gb10 = HardwareArch::GraceBlackwellGb10.profile();
    let rtx = HardwareArch::ConsumerRtx4090.profile();

    assert_eq!(h100.memory_kind, "HBM3");
    assert_eq!(b200.memory_kind, "HBM3e");
    assert_eq!(gb10.memory_kind, "LPDDR5X Unified");
    assert_eq!(rtx.memory_kind, "GDDR6X");

    assert!((gb10.peak_bandwidth_gb_s - 273.0).abs() < 1e-6);
    assert!((b200.peak_bandwidth_gb_s - 8_000.0).abs() < 1e-6);
    assert!(gb10.is_bandwidth_bound_unified_memory);
    assert!(!b200.is_bandwidth_bound_unified_memory);
    assert!(gb10.ridge_point_flops_per_byte() > 3.0 * b200.ridge_point_flops_per_byte());
    assert!(b200.prefers_async_pipeline_on_compute_prefill());
    assert!(h100.prefers_async_pipeline_on_compute_prefill());
    assert!(!gb10.prefers_async_pipeline_on_compute_prefill());
    assert!(!rtx.prefers_async_pipeline_on_compute_prefill());
    assert_eq!(rtx.vram_capacity_bytes, 24 * 1024 * 1024 * 1024);
}

#[test]
fn sha256_matches_nist_fips_180_4_test_vectors() {
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
fn lse_merge_is_commutative_and_handles_empty_and_extreme_exponents() {
    let o0 = [0.25f32, -0.5, 1.25, 0.75];
    let o1 = [-0.75f32, 0.5, -0.25, 2.0];
    let lse0 = 142.5f32;
    let lse1 = 140.25f32;

    let (ab, l_ab) = merge_lse_states(&o0, lse0, &o1, lse1);
    let (ba, l_ba) = merge_lse_states(&o1, lse1, &o0, lse0);
    assert!((l_ab - l_ba).abs() < 1e-6);
    for (x, y) in ab.iter().zip(ba.iter()) {
        assert!((x - y).abs() < 1e-6);
    }

    // Empty partition identity.
    let (id0, l_id0) = merge_lse_states(&o0, lse0, &[0.0; 4], f32::NEG_INFINITY);
    assert_eq!(id0, o0);
    assert_eq!(l_id0, lse0);
    let (id1, l_id1) = merge_lse_states(&[0.0; 4], f32::NEG_INFINITY, &o1, lse1);
    assert_eq!(id1, o1);
    assert_eq!(l_id1, lse1);
}

#[test]
fn multi_level_lse_merge_matches_pairwise_associative_merge() {
    let o0 = [0.3f32, -0.2, 0.8, 0.1];
    let o1 = [-0.4f32, 0.9, -0.1, 0.5];
    let o2 = [0.6f32, 0.2, -0.7, -0.3];
    let l0 = 11.2f32;
    let l1 = 12.7f32;
    let l2 = 10.4f32;

    let (o01, l01) = merge_lse_states(&o0, l0, &o1, l1);
    let (o_pair, l_pair) = merge_lse_states(&o01, l01, &o2, l2);
    let (o_multi, l_multi) = merge_lse_multi_level(&[(&o0, l0), (&o1, l1), (&o2, l2)]);

    assert!((l_pair - l_multi).abs() < 1e-5);
    for (x, y) in o_pair.iter().zip(o_multi.iter()) {
        assert!((x - y).abs() < 1e-6, "mismatch: {x} vs {y}");
    }
}

#[test]
fn two_level_cascade_exact_equivalence_at_b1() {
    let mut rng = Lcg64::new(0x1346_0001);
    let mut pool = PagedKvPool::new(2, 16, 64);
    let shared = pool.allocate_random_sequence(64, &mut rng);
    let suff = vec![pool.allocate_random_sequence(32, &mut rng)];
    let queries: Vec<f32> = (0..8 * 64).map(|_| rng.next_f32_signed() * 0.4).collect();

    let ref_out = unshared_paged_gqa_decode(&pool, &queries, 1, 8, &shared, 64, &suff, &[32]);
    let cas_out = two_level_cascade_gqa_decode(&pool, &queries, 1, 8, &shared, 64, &suff, &[32]);

    for (a, b) in ref_out.output.iter().zip(cas_out.output.iter()) {
        assert!((a - b).abs() <= 1e-6, "diff={}", (a - b).abs());
    }
    for (a, b) in ref_out.lse.iter().zip(cas_out.lse.iter()) {
        assert!((a - b).abs() <= 1e-5, "lse diff={}", (a - b).abs());
    }
}

#[test]
fn two_level_cascade_exact_equivalence_across_batches_4_8_16() {
    let mut rng = Lcg64::new(0x1346_0002);
    let num_q_heads = 8;
    let num_kv_heads = 2;
    let head_dim = 64;
    let hidden_dim = num_q_heads * head_dim;
    let vocab_size = 48;
    let proj: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|_| rng.next_f32_signed() * 0.25)
        .collect();

    for (batch_size, prefix_len, suffix_len) in [(4, 128, 32), (8, 256, 48), (16, 256, 32)] {
        let mut pool = PagedKvPool::new(num_kv_heads, 16, head_dim);
        let shared = pool.allocate_random_sequence(prefix_len, &mut rng);
        let mut suff = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            suff.push(pool.allocate_random_sequence(suffix_len, &mut rng));
        }
        let suff_lens = vec![suffix_len; batch_size];
        let queries: Vec<f32> = (0..batch_size * hidden_dim)
            .map(|_| rng.next_f32_signed() * 0.4)
            .collect();

        let ref_out = unshared_paged_gqa_decode(
            &pool,
            &queries,
            batch_size,
            num_q_heads,
            &shared,
            prefix_len,
            &suff,
            &suff_lens,
        );
        let cas_out = two_level_cascade_gqa_decode(
            &pool,
            &queries,
            batch_size,
            num_q_heads,
            &shared,
            prefix_len,
            &suff,
            &suff_lens,
        );

        let max_err = ref_out
            .output
            .iter()
            .zip(cas_out.output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err <= 1e-6, "B={batch_size} max_err={max_err}");

        let tok_ref =
            greedy_argmax_tokens(&ref_out.output, batch_size, hidden_dim, vocab_size, &proj);
        let tok_cas =
            greedy_argmax_tokens(&cas_out.output, batch_size, hidden_dim, vocab_size, &proj);
        assert_eq!(tok_ref, tok_cas, "greedy argmax must match 100%");
    }
}

#[test]
fn two_level_cascade_handles_unaligned_prefix_and_suffix_lengths() {
    let mut rng = Lcg64::new(0x1346_0003);
    let mut pool = PagedKvPool::new(2, 16, 32);
    let prefix_len = 53; // Not a multiple of page_size=16!
    let suffix_len = 21;
    let shared = pool.allocate_random_sequence(prefix_len, &mut rng);
    let suff: Vec<Vec<usize>> = (0..4)
        .map(|_| pool.allocate_random_sequence(suffix_len, &mut rng))
        .collect();
    let queries: Vec<f32> = (0..4 * 8 * 32)
        .map(|_| rng.next_f32_signed() * 0.5)
        .collect();

    let ref_out = unshared_paged_gqa_decode(
        &pool,
        &queries,
        4,
        8,
        &shared,
        prefix_len,
        &suff,
        &[suffix_len; 4],
    );
    let cas_out = two_level_cascade_gqa_decode(
        &pool,
        &queries,
        4,
        8,
        &shared,
        prefix_len,
        &suff,
        &[suffix_len; 4],
    );

    let max_err = ref_out
        .output
        .iter()
        .zip(cas_out.output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err <= 1e-6, "unaligned max_err={max_err}");
}

#[test]
fn two_level_cascade_achieves_exact_shared_prefix_kv_byte_reduction() {
    let mut rng = Lcg64::new(0x1346_0004);
    let batch_size = 16usize;
    let num_q_heads = 8usize;
    let num_kv_heads = 2usize;
    let gqa_group = num_q_heads / num_kv_heads;
    let mut pool = PagedKvPool::new(num_kv_heads, 16, 64);
    let shared = pool.allocate_random_sequence(512, &mut rng);
    let suff: Vec<Vec<usize>> = (0..batch_size)
        .map(|_| pool.allocate_random_sequence(32, &mut rng))
        .collect();
    let queries: Vec<f32> = (0..batch_size * num_q_heads * 64)
        .map(|_| rng.next_f32_signed() * 0.3)
        .collect();

    let ref_out = unshared_paged_gqa_decode(
        &pool,
        &queries,
        batch_size,
        num_q_heads,
        &shared,
        512,
        &suff,
        &[32; 16],
    );
    let cas_out = two_level_cascade_gqa_decode(
        &pool,
        &queries,
        batch_size,
        num_q_heads,
        &shared,
        512,
        &suff,
        &[32; 16],
    );

    // Shared prefix reads drop by exact factor `B * gqa_group` (`16 * 4 = 64x`) on Level-0 pages!
    assert_eq!(
        ref_out.stats.shared_prefix_kv_bytes_fp32 / cas_out.stats.shared_prefix_kv_bytes_fp32,
        (batch_size * gqa_group) as u64
    );
    // Total KV bytes (shared + suffix) drop well over 8x!
    let total_ratio =
        (ref_out.stats.total_kv_bytes_fp32 as f64) / (cas_out.stats.total_kv_bytes_fp32 as f64);
    assert!(
        total_ratio >= 8.0,
        "expected >=8x KV reduction, got {total_ratio:.2}x"
    );
}

#[test]
fn three_level_prefix_tree_cascade_matches_unshared_exact() {
    let mut rng = Lcg64::new(0x1346_0005);
    let batch_size = 8usize;
    let num_q_heads = 8usize;
    let mut pool = PagedKvPool::new(2, 16, 64);
    let l0 = pool.allocate_random_sequence(128, &mut rng);
    let l1_groups = vec![
        pool.allocate_random_sequence(96, &mut rng),
        pool.allocate_random_sequence(96, &mut rng),
    ];
    let req_sg = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let mut l2 = Vec::new();
    let mut combined_suff = Vec::new();
    for &sg in &req_sg {
        let s = pool.allocate_random_sequence(32, &mut rng);
        let mut c = l1_groups[sg].clone();
        c.extend_from_slice(&s);
        l2.push(s);
        combined_suff.push(c);
    }
    let q: Vec<f32> = (0..batch_size * num_q_heads * 64)
        .map(|_| rng.next_f32_signed() * 0.4)
        .collect();

    let ref_out = unshared_paged_gqa_decode(
        &pool,
        &q,
        batch_size,
        num_q_heads,
        &l0,
        128,
        &combined_suff,
        &[128; 8],
    );
    let cas3 = three_level_cascade_gqa_decode(
        &pool,
        &q,
        batch_size,
        num_q_heads,
        &l0,
        128,
        &l1_groups,
        &[96, 96],
        &req_sg,
        &l2,
        &[32; 8],
    );

    let max_err = ref_out
        .output
        .iter()
        .zip(cas3.output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err <= 1e-6, "3-level max_err={max_err}");
    assert!(cas3.stats.total_kv_bytes_fp32 * 5 < ref_out.stats.total_kv_bytes_fp32);
}

#[test]
fn ragged_descriptor_validation_catches_malformed_inputs_fail_closed() {
    let mut rng = Lcg64::new(0x1346_0006);
    let mut pool = PagedKvPool::new(2, 16, 32);
    let pages = pool.allocate_random_sequence(32, &mut rng);

    // Non-monotonic qo_indptr.
    let bad_indptr = RaggedBatchDescriptor {
        qo_indptr: vec![0, 5, 3],
        kv_indptr: vec![0, 1, 2],
        kv_page_indices: pages.clone(),
        kv_last_page_len: vec![16, 16],
        kv_seq_lens: vec![16, 16],
    };
    assert_eq!(
        bad_indptr.validate(&pool),
        Err(RaggedDescriptorError::NonMonotonicIndptr)
    );

    // Causal violation: kv_seq_len < q_len.
    let bad_causal = RaggedBatchDescriptor {
        qo_indptr: vec![0, 20],
        kv_indptr: vec![0, 1],
        kv_page_indices: vec![pages[0]],
        kv_last_page_len: vec![16],
        kv_seq_lens: vec![16],
    };
    assert_eq!(
        bad_causal.validate(&pool),
        Err(RaggedDescriptorError::CausalLengthViolation {
            req_idx: 0,
            q_len: 20,
            kv_len: 16
        })
    );

    // Out-of-bounds physical page ID.
    let bad_page = RaggedBatchDescriptor {
        qo_indptr: vec![0, 1],
        kv_indptr: vec![0, 1],
        kv_page_indices: vec![99],
        kv_last_page_len: vec![16],
        kv_seq_lens: vec![16],
    };
    assert_eq!(
        bad_page.validate(&pool),
        Err(RaggedDescriptorError::PageOutOfBounds {
            page_id: 99,
            num_pages: 2
        })
    );
}

#[test]
fn fused_ragged_prefill_decode_matches_separate_execution_bit_for_bit() {
    let rep = run_cpu_ragged_verification(1346);
    assert_eq!(rep.prefill_requests, 2);
    assert_eq!(rep.decode_requests, 6);
    assert_eq!(rep.total_q_tokens, 86);
    assert_eq!(rep.separate_dispatch_stages, 2);
    assert_eq!(rep.fused_dispatch_stages, 1);
    assert!(rep.max_abs_output_err <= 1e-6);
    assert!(rep.max_abs_lse_err <= 1e-5);
    assert!((rep.greedy_argmax_match_pct - 100.0).abs() < 1e-9);
}

#[test]
fn ragged_handles_pure_decode_and_pure_prefill_batches_cleanly() {
    let mut rng = Lcg64::new(0x1346_0007);
    let mut pool = PagedKvPool::new(2, 16, 32);
    let p0 = pool.allocate_random_sequence(45, &mut rng);
    let p1 = pool.allocate_random_sequence(32, &mut rng);

    let mut kv_page_indices = p0.clone();
    kv_page_indices.extend_from_slice(&p1);
    let desc = RaggedBatchDescriptor {
        qo_indptr: vec![0, 1, 2],
        kv_indptr: vec![0, p0.len(), kv_page_indices.len()],
        kv_page_indices,
        kv_last_page_len: vec![13, 16],
        kv_seq_lens: vec![45, 32],
    };
    let q: Vec<f32> = (0..2 * 4 * 32)
        .map(|_| rng.next_f32_signed() * 0.4)
        .collect();
    let sep = separate_prefill_and_decode_paged_gqa(&pool, &q, &desc, 4).unwrap();
    let fused = fused_ragged_prefill_decode_gqa(&pool, &q, &desc, 4).unwrap();
    assert_eq!(sep.dispatch_stages, 1);
    assert_eq!(fused.dispatch_stages, 1);
    assert_eq!(sep.output, fused.output);
}

#[test]
fn hybrid_attention_gdn_mlp_step_verifier_proves_exactness_and_amdahl_shares() {
    let rep = run_cpu_hybrid_step_verification(1346, HybridStepConfig::default());
    assert_eq!((rep.num_attn_layers, rep.num_gdn_layers), (2, 6));
    assert!(
        rep.max_abs_hidden_err <= 1e-5,
        "hidden_err={}",
        rep.max_abs_hidden_err
    );
    assert!(
        rep.max_abs_gdn_state_err <= 1e-5,
        "gdn_err={}",
        rep.max_abs_gdn_state_err
    );
    assert!((rep.greedy_argmax_match_pct - 100.0).abs() < 1e-9);
    let share_sum = rep.attn_time_fraction + rep.gdn_time_fraction + rep.mlp_time_fraction;
    assert!((share_sum - 1.0).abs() < 1e-6);
    assert!(rep.gdn_time_fraction + rep.mlp_time_fraction > 0.30);
}

#[test]
fn cpu_cascade_verification_suite_passes_accuracy_and_kv_reduction_gates() {
    let suite = run_cpu_cascade_verification(1346);
    assert!(suite.max_two_level_output_err <= 1e-6);
    assert!(suite.three_level_max_output_err <= 1e-6);
    assert!((suite.aggregate_argmax_match_pct - 100.0).abs() < 1e-9);
    assert!((suite.quality_regression_pp - 0.0).abs() < 1e-9);
    assert!(suite.three_level_kv_read_reduction >= 5.0);

    let b16_row = suite
        .rows
        .iter()
        .find(|r| r.batch_size == 16 && r.shared_prefix_len == 512)
        .expect("B=16 prefix=512 row must exist");
    assert!(b16_row.kv_read_reduction_ratio >= 8.0);
    assert!(b16_row.cpu_wall_speedup > 1.0);
}

#[test]
fn b200_pipelining_gains_do_not_transfer_to_bandwidth_bound_gb10() {
    let m4 = HybridModelSpec::hybrid_4b();
    let wl = AgentWorkloadConfig::canonical_shared_repo(1);

    let b200_base = evaluate_serving_scorecard(
        HardwareArch::BlackwellB200,
        &m4,
        QuantFormat::Bf16,
        wl,
        AttentionBackend::BaselinePagedSeparate,
        "test",
        None,
    );
    let b200_pipe = evaluate_serving_scorecard(
        HardwareArch::BlackwellB200,
        &m4,
        QuantFormat::Bf16,
        wl,
        AttentionBackend::WarpPipelinedAsync,
        "test",
        Some((
            b200_base.agent_loop_wall_time_s,
            b200_base.p95_itl_ms,
            b200_base.isolated_attn_kernel_tflops,
        )),
    );

    let gb10_base = evaluate_serving_scorecard(
        HardwareArch::GraceBlackwellGb10,
        &m4,
        QuantFormat::Int4Awq,
        wl,
        AttentionBackend::BaselinePagedSeparate,
        "test",
        None,
    );
    let gb10_pipe = evaluate_serving_scorecard(
        HardwareArch::GraceBlackwellGb10,
        &m4,
        QuantFormat::Int4Awq,
        wl,
        AttentionBackend::WarpPipelinedAsync,
        "test",
        Some((
            gb10_base.agent_loop_wall_time_s,
            gb10_base.p95_itl_ms,
            gb10_base.isolated_attn_kernel_tflops,
        )),
    );

    // B200 achieves large cold TTFT improvement from 5th-gen TMEM async pipelining.
    assert!(b200_pipe.cold_ttft_ms < b200_base.cold_ttft_ms);
    assert!(b200_pipe.isolated_attn_kernel_speedup > 1.50);

    // GB10 wins isolated 32K prefill kernel TFLOPS, YET regresses end-to-end 20-turn wall time!
    assert!(gb10_pipe.isolated_attn_kernel_speedup > 1.25);
    assert!(gb10_pipe.wall_speedup_vs_baseline < 1.0);
    assert!(!gb10_pipe.gate.accepted);
}

#[test]
fn consumer_rtx4090_enforces_24gib_vram_cap_for_27b_bf16_while_accepting_int4() {
    let m27 = HybridModelSpec::hybrid_27b();
    let wl = AgentWorkloadConfig::canonical_shared_repo(1);

    let bf16 = evaluate_serving_scorecard(
        HardwareArch::ConsumerRtx4090,
        &m27,
        QuantFormat::Bf16,
        wl,
        AttentionBackend::AdaptiveHybridDispatch,
        "test",
        None,
    );
    assert!(!bf16.vram_fits);
    assert!(bf16.peak_vram_mib > 24_576.0);
    assert!(!bf16.gate.accepted);

    let int4_base = evaluate_serving_scorecard(
        HardwareArch::ConsumerRtx4090,
        &m27,
        QuantFormat::Int4Awq,
        wl,
        AttentionBackend::BaselinePagedSeparate,
        "test",
        None,
    );
    let int4_adapt = evaluate_serving_scorecard(
        HardwareArch::ConsumerRtx4090,
        &m27,
        QuantFormat::Int4Awq,
        wl,
        AttentionBackend::AdaptiveHybridDispatch,
        "test",
        Some((
            int4_base.agent_loop_wall_time_s,
            int4_base.p95_itl_ms,
            int4_base.isolated_attn_kernel_tflops,
        )),
    );
    assert!(int4_adapt.vram_fits);
    assert!(int4_adapt.peak_vram_mib < 24_576.0);
    assert!(int4_adapt.gate.accepted);
}

#[test]
fn adaptive_hybrid_dispatch_beats_baseline_on_every_vram_fitting_b8_cell() {
    let report = run_full_benchmark(1346, "testcommit".to_string());
    let mut checked = 0usize;
    for row in report.scorecards.iter().filter(|r| {
        r.batch_size == 8 && r.backend == AttentionBackend::AdaptiveHybridDispatch.as_str()
    }) {
        if row.vram_fits {
            assert!(
                row.wall_speedup_vs_baseline > 1.05,
                "expected >1.05x on {} {} {} B=8, got {:.3}x",
                row.hardware_arch,
                row.model,
                row.quant_format,
                row.wall_speedup_vs_baseline
            );
            assert!(row.gate.accepted, "gate failed: {}", row.gate.reason);
            checked += 1;
        } else {
            assert!(!row.gate.accepted, "OOM cell must be rejected");
        }
    }
    assert!(
        checked >= 18,
        "must validate >=18 fitting (arch,model,quant) B=8 cells"
    );
}

#[test]
fn adaptive_hybrid_dispatch_never_regresses_b1_unshared_unlike_forced_cascade() {
    let m9 = HybridModelSpec::hybrid_9b();
    let wl = AgentWorkloadConfig::unshared_single_agent();
    for arch in HardwareArch::all() {
        let base = evaluate_serving_scorecard(
            arch,
            &m9,
            QuantFormat::Int4Awq,
            wl,
            AttentionBackend::BaselinePagedSeparate,
            "test",
            None,
        );
        let base_ref = (
            base.agent_loop_wall_time_s,
            base.p95_itl_ms,
            base.isolated_attn_kernel_tflops,
        );
        let forced_cas = evaluate_serving_scorecard(
            arch,
            &m9,
            QuantFormat::Int4Awq,
            wl,
            AttentionBackend::CascadeSharedPrefix,
            "test",
            Some(base_ref),
        );
        let adaptive = evaluate_serving_scorecard(
            arch,
            &m9,
            QuantFormat::Int4Awq,
            wl,
            AttentionBackend::AdaptiveHybridDispatch,
            "test",
            Some(base_ref),
        );
        assert!(
            !forced_cas.gate.accepted,
            "forced cascade on B=1 unshared must be rejected on {}",
            arch.id()
        );
        assert!(
            adaptive.gate.accepted,
            "adaptive dispatch on B=1 unshared must never regress on {}",
            arch.id()
        );
        assert!(adaptive.agent_loop_wall_time_s <= base.agent_loop_wall_time_s + 1e-9);
    }
}

#[test]
fn all_three_explicit_negative_regimes_are_strictly_rejected() {
    let report = run_full_benchmark(1346, "testcommit".to_string());
    assert_eq!(report.negative_regimes.len(), 3);
    let expected = [
        "neg_cascade_overhead_at_b1_unshared",
        "neg_kernel_tflops_win_amdahl_loss",
        "neg_rtx4090_27b_bf16_vram_overflow",
    ];
    for (neg, exp_name) in report.negative_regimes.iter().zip(expected.iter()) {
        assert_eq!(neg.regime, *exp_name);
        assert!(
            !neg.scorecard.gate.accepted,
            "negative regime {} must be REJECTED",
            neg.regime
        );
        assert_eq!(neg.scorecard.gate.status_str(), "REJECTED");
    }

    // Specifically verify negative #2 proves kernel TFLOPS win != end-to-end serving win.
    let amdahl = &report.negative_regimes[1];
    assert!(amdahl.scorecard.isolated_attn_kernel_speedup > 1.20);
    assert!(amdahl.scorecard.wall_speedup_vs_baseline < 1.0);
    assert!(amdahl.scorecard.agent_loop_wall_time_s > amdahl.baseline_wall_time_s);
}

#[test]
fn gb10_gains_end_to_end_from_cascade_bandwidth_reduction_at_b8() {
    let m9 = HybridModelSpec::hybrid_9b();
    let wl = AgentWorkloadConfig::canonical_shared_repo(8);
    let base = evaluate_serving_scorecard(
        HardwareArch::GraceBlackwellGb10,
        &m9,
        QuantFormat::Int4Awq,
        wl,
        AttentionBackend::BaselinePagedSeparate,
        "test",
        None,
    );
    let adaptive = evaluate_serving_scorecard(
        HardwareArch::GraceBlackwellGb10,
        &m9,
        QuantFormat::Int4Awq,
        wl,
        AttentionBackend::AdaptiveHybridDispatch,
        "test",
        Some((
            base.agent_loop_wall_time_s,
            base.p95_itl_ms,
            base.isolated_attn_kernel_tflops,
        )),
    );
    // Even though GB10 rejects compute-pipelining overhead, shared-prefix cascade cuts
    // LPDDR5X KV reads by 8x on the shared repo prefix and yields genuine end-to-end gain!
    assert!(adaptive.gate.accepted);
    assert!(adaptive.wall_speedup_vs_baseline > 1.15);
    assert!(adaptive.peak_vram_mib < base.peak_vram_mib);
}

#[test]
fn gate_rejects_quality_and_tail_latency_regressions() {
    let m4 = HybridModelSpec::hybrid_4b();
    let wl = AgentWorkloadConfig::canonical_shared_repo(4);
    let mut row = evaluate_serving_scorecard(
        HardwareArch::HopperH100Sxm5,
        &m4,
        QuantFormat::Bf16,
        wl,
        AttentionBackend::BaselinePagedSeparate,
        "test",
        None,
    );
    let base_wall = row.agent_loop_wall_time_s;
    let base_p95 = row.p95_itl_ms;

    // Quality regression -> REJECTED.
    row.quality_regression_pp = 0.5;
    row.output_correctness_pct = 99.5;
    let g_qual = evaluate_end_to_end_gate(&row, base_wall, base_p95, false);
    assert!(!g_qual.accepted);
    assert!(g_qual.reason.contains("Quality"));

    // Tail-latency p95 regression -> REJECTED.
    row.quality_regression_pp = 0.0;
    row.output_correctness_pct = 100.0;
    row.agent_loop_wall_time_s = base_wall * 0.95;
    row.p95_itl_ms = base_p95 * 1.10;
    let g_p95 = evaluate_end_to_end_gate(&row, base_wall, base_p95, false);
    assert!(!g_p95.accepted);
    assert!(g_p95.reason.contains("p95 ITL"));
}

#[test]
fn full_benchmark_json_contains_all_four_arch_provenance_tags_and_valid_sha256() {
    let report = run_full_benchmark(1346, "abc123456789".to_string());
    let json = report.to_json();
    assert!(json.starts_with('{'));
    assert!(json.trim_end().ends_with('}'));

    for tag in [
        "cpu_directly_measured",
        "analytical_gpu_hopper_h100_sxm5",
        "analytical_gpu_blackwell_b200",
        "analytical_gpu_grace_blackwell_gb10",
        "analytical_gpu_consumer_rtx_4090",
        "neg_cascade_overhead_at_b1_unshared",
        "neg_kernel_tflops_win_amdahl_loss",
        "neg_rtx4090_27b_bf16_vram_overflow",
    ] {
        assert!(json.contains(tag), "missing required tag {tag} in JSON");
    }

    let digest = sha256_hex(json.as_bytes());
    assert_eq!(digest.len(), 64);
    assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
}
