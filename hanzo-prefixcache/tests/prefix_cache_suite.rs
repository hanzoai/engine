// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive integration tests for `hanzo-prefixcache` (`hanzoai/engine#51`, HIP-1346).

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use hanzo_prefixcache::bench::{
    run_full_benchmark_suite, run_negative_regimes, run_tenant_isolation_audit, AgentCorpus,
    DEFAULT_BENCH_SEED,
};
use hanzo_prefixcache::crypto::{Digest256, Sha256};
use hanzo_prefixcache::error::PrefixCacheError;
use hanzo_prefixcache::key::{
    partition_scoped_blocks, validate_scope_transition, CompatibilityProfile, PrefixCacheKey,
    QuantFormat, RopePosEncoding, RopeScheme, TenantAuthorization, TenantScope, TokenPrefixChain,
};
use hanzo_prefixcache::model::ModelSpec;
use hanzo_prefixcache::radix::{RadixPrefixIndex, StorageTier};
use hanzo_prefixcache::scheduler::{PrefixAwareScheduler, SchedulerConfig, WorkerNode};
use hanzo_prefixcache::snapshot::{
    decode_gdn_f32_state, synthesize_block_kv_gdn_state, PrefixBlockSnapshot,
    SNAPSHOT_FORMAT_VERSION, SNAPSHOT_HEADER_BYTES,
};
use hanzo_prefixcache::tier::{PersistencePolicy, TieredPrefixStore, TieredStoreConfig};

static TEST_DIR_SEQ: AtomicU64 = AtomicU64::new(0);

fn fresh_test_dir(label: &str) -> PathBuf {
    let seq = TEST_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "hanzo_prefixcache_test_{}_{}_{}",
        std::process::id(),
        seq,
        label
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp test dir");
    dir
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn sample_compat() -> CompatibilityProfile {
    ModelSpec::hybrid_9b().compatibility_profile(b"llama3-test-tokenizer-v1")
}

#[test]
fn seven_field_prefix_cache_key_binds_all_dimensions() {
    let base_compat = sample_compat();
    let scope = TenantScope::TenantPrivate {
        tenant_id: "alice".to_string(),
    };
    let tokens: Vec<u32> = (1..=64).collect();
    let chain = TokenPrefixChain::root().extend(&tokens);
    let base_key = PrefixCacheKey::derive(&base_compat, &scope, &chain);

    // 1. Model checkpoint change
    let mut c1 = base_compat.clone();
    c1.model_checkpoint.checkpoint_digest = Sha256::digest(b"other-weights");
    assert_ne!(
        base_key.key_digest,
        PrefixCacheKey::derive(&c1, &scope, &chain).key_digest
    );

    // 2. Tokenizer hash change
    let mut c2 = base_compat.clone();
    c2.tokenizer.vocab_hash = Sha256::digest(b"other-tokenizer");
    assert_ne!(
        base_key.key_digest,
        PrefixCacheKey::derive(&c2, &scope, &chain).key_digest
    );

    // 3. Quantization format change
    let mut c3 = base_compat.clone();
    c3.quant_format = QuantFormat::KiviInt4 {
        group_size: 64,
        residual_len: 64,
    };
    assert_ne!(
        base_key.key_digest,
        PrefixCacheKey::derive(&c3, &scope, &chain).key_digest
    );

    // 4. Position encoding change
    let mut c4 = base_compat.clone();
    c4.rope_pos_encoding = RopePosEncoding::new(RopeScheme::Yarn, 1_000_000.0, 2.0, 131_072);
    assert_ne!(
        base_key.key_digest,
        PrefixCacheKey::derive(&c4, &scope, &chain).key_digest
    );

    // 5. Runtime config change
    let mut c5 = base_compat.clone();
    c5.runtime_config.runtime_flags_hash = Sha256::digest(b"different-flags");
    assert_ne!(
        base_key.key_digest,
        PrefixCacheKey::derive(&c5, &scope, &chain).key_digest
    );

    // 6. Tenant scope ACL change
    let other_scope = TenantScope::TenantPrivate {
        tenant_id: "bob".to_string(),
    };
    assert_ne!(
        base_key.key_digest,
        PrefixCacheKey::derive(&base_compat, &other_scope, &chain).key_digest
    );

    // 7. Token prefix chain change
    let mut mutated_tokens = tokens;
    mutated_tokens[17] ^= 1;
    let mutated_chain = TokenPrefixChain::root().extend(&mutated_tokens);
    assert_ne!(
        base_key.key_digest,
        PrefixCacheKey::derive(&base_compat, &scope, &mutated_chain).key_digest
    );
}

#[test]
fn snapshot_encode_decode_roundtrip_and_fail_closed_corruption_rejection() {
    let compat = sample_compat();
    let scope = TenantScope::PublicSystem;
    let tokens: Vec<u32> = (100..164).collect();
    let chain = TokenPrefixChain::root().extend(&tokens);
    let key = PrefixCacheKey::derive(&compat, &scope, &chain);

    let (kv_bytes, gdn_f32, gdn_bytes) = synthesize_block_kv_gdn_state(&[], &tokens, 0, 2048, 128);
    let snap =
        PrefixBlockSnapshot::build(&key, &compat, Digest256::ZERO, 0, 64, kv_bytes, gdn_bytes)
            .expect("build snapshot");

    let encoded = snap.encode();
    let decoded =
        PrefixBlockSnapshot::decode_verified(&encoded, &key, &compat).expect("verify snapshot");
    assert_eq!(snap, decoded);
    let restored_gdn = decode_gdn_f32_state(&decoded.gdn_payload).expect("decode gdn");
    assert_eq!(gdn_f32, restored_gdn);

    // Bad magic
    let mut bad_magic = encoded.clone();
    bad_magic[0] = b'X';
    assert!(matches!(
        PrefixBlockSnapshot::decode_verified(&bad_magic, &key, &compat),
        Err(PrefixCacheError::BadMagic(_))
    ));

    // Unsupported format version
    let mut bad_ver = encoded.clone();
    let next_ver = (SNAPSHOT_FORMAT_VERSION + 1).to_le_bytes();
    bad_ver[4] = next_ver[0];
    bad_ver[5] = next_ver[1];
    assert!(matches!(
        PrefixBlockSnapshot::decode_verified(&bad_ver, &key, &compat),
        Err(PrefixCacheError::UnsupportedVersion { .. })
    ));

    // Header CRC mismatch
    let mut bad_hdr = encoded.clone();
    bad_hdr[16] ^= 0x04;
    assert!(matches!(
        PrefixBlockSnapshot::decode_verified(&bad_hdr, &key, &compat),
        Err(PrefixCacheError::HeaderCrcMismatch { .. })
    ));

    // KV payload CRC mismatch
    let mut bad_kv = encoded.clone();
    bad_kv[SNAPSHOT_HEADER_BYTES + 10] ^= 0x10;
    assert!(matches!(
        PrefixBlockSnapshot::decode_verified(&bad_kv, &key, &compat),
        Err(PrefixCacheError::KvCrcMismatch { .. })
    ));

    // GDN payload CRC mismatch
    let mut bad_gdn = encoded.clone();
    let last = bad_gdn.len() - 4;
    bad_gdn[last] ^= 0x20;
    assert!(matches!(
        PrefixBlockSnapshot::decode_verified(&bad_gdn, &key, &compat),
        Err(PrefixCacheError::GdnCrcMismatch { .. })
    ));

    // Truncated buffer
    assert!(matches!(
        PrefixBlockSnapshot::decode_verified(&encoded[..encoded.len() - 5], &key, &compat),
        Err(PrefixCacheError::MalformedSnapshotLength { .. })
    ));

    // Key mismatch
    let other_key = PrefixCacheKey::derive(
        &compat,
        &TenantScope::TenantPrivate {
            tenant_id: "mallory".to_string(),
        },
        &chain,
    );
    assert!(matches!(
        PrefixBlockSnapshot::decode_verified(&encoded, &other_key, &compat),
        Err(PrefixCacheError::KeyMismatch)
    ));
}

#[test]
fn radix_pinning_leaf_first_lru_eviction_and_monotonic_scope_boundaries() {
    let compat = sample_compat();
    let pub_auth = TenantAuthorization::publisher("alice", vec!["org-alpha".to_string()], true);
    let mut index = RadixPrefixIndex::new();

    // 3-block sequence: PublicSystem -> OrgShared("org-alpha") -> TenantPrivate("alice")
    let tokens: Vec<u32> = (1..=192).collect();
    let blocks =
        partition_scoped_blocks(&tokens, 64, 1, 1, Some("org-alpha"), "alice").expect("partition");

    let mut chain = TokenPrefixChain::root();
    let mut prev = Digest256::ZERO;
    let mut keys = Vec::new();
    for (i, blk) in blocks.iter().enumerate() {
        chain = chain.extend(blk.tokens);
        let key = PrefixCacheKey::derive(&compat, &blk.scope, &chain);
        index
            .upsert_node(
                key.clone(),
                prev,
                blk.scope.clone(),
                &pub_auth,
                (i * 64) as u32,
                64,
                1000,
                StorageTier::Tier0Hbm,
            )
            .expect("upsert");
        keys.push(key.key_digest);
        prev = keys[i];
    }

    assert_eq!(index.tier0_bytes(), 3000);

    // Pin the leaf block -> cannot evict leaf or its ancestors from Tier 0!
    index.pin_blocks(&[keys[2]]).expect("pin leaf");
    let ev_pinned = index.evict_tier0_to_budget(1000, false);
    assert!(
        ev_pinned.is_empty(),
        "pinned leaf must protect entire active branch"
    );
    assert_eq!(index.tier0_bytes(), 3000);

    // Unpin leaf -> eviction must proceed strictly leaf-first (keys[2] first, then keys[1]).
    index.unpin_blocks(&[keys[2]]).expect("unpin leaf");
    let ev = index.evict_tier0_to_budget(1000, false);
    assert_eq!(ev.len(), 2);
    assert_eq!(ev[0].0, keys[2]);
    assert_eq!(ev[1].0, keys[1]);
    assert_eq!(index.tier0_bytes(), 1000);
    assert!(
        index.get(&keys[0]).is_some(),
        "shared root ancestor must survive"
    );

    // Monotonic scope boundary guard rejects blending private -> shared.
    assert!(matches!(
        validate_scope_transition(
            Some(&TenantScope::TenantPrivate {
                tenant_id: "alice".to_string()
            }),
            &TenantScope::PublicSystem
        ),
        Err(PrefixCacheError::ScopeBoundaryViolation { .. })
    ));
    assert!(matches!(
        validate_scope_transition(
            Some(&TenantScope::OrgShared {
                org_id: "org-a".to_string()
            }),
            &TenantScope::OrgShared {
                org_id: "org-b".to_string()
            }
        ),
        Err(PrefixCacheError::ScopeBoundaryViolation { .. })
    ));
}

#[test]
fn tier1_cold_restart_reload_and_tier2_cross_worker_sharing() {
    let dir = fresh_test_dir("tier1_tier2_restart");
    let t1_a = dir.join("worker_a_tier1");
    let t1_b = dir.join("worker_b_tier1");
    let t2_shared = dir.join("cluster_tier2");

    let model = ModelSpec::hybrid_4b();
    let compat = model.compatibility_profile(b"tok-v1");
    let auth_alice = TenantAuthorization::publisher("alice", vec!["hanzo-eng".to_string()], true);
    let auth_bob = TenantAuthorization::standard_tenant("bob", vec!["hanzo-eng".to_string()]);

    let tokens: Vec<u32> = (500..756).collect(); // 4 blocks: 1 Public, 1 OrgShared, 2 Private
    let blocks =
        partition_scoped_blocks(&tokens, 64, 1, 1, Some("hanzo-eng"), "alice").expect("partition");

    let cfg_a = TieredStoreConfig {
        tier0_budget_bytes: 2 * model.accounted_bytes_per_block(),
        tier1_budget_bytes: 32 * model.accounted_bytes_per_block(),
        tier1_dir: Some(t1_a.clone()),
        tier2_dir: Some(t2_shared.clone()),
        policy: PersistencePolicy::SelectiveRadix {
            private_eager_prefix_blocks: 8,
            private_persist_min_hits: 1,
            demote_on_tier0_evict: true,
            enable_tier2_sharing: true,
        },
        enable_tier2_bandwidth_guard: true,
        tier2_link_bandwidth_bps: model.tier2_nominal_bw_bps,
        tier2_rtt_sec: model.tier2_rtt_sec,
        local_prefill_sec_per_token: model.prefill_sec_per_token(),
    };

    let mut expected_keys = Vec::new();
    let mut expected_gdn = vec![0.0f32; 128];
    {
        let mut worker_a = TieredPrefixStore::open(cfg_a.clone()).expect("open worker A");
        let mut chain = TokenPrefixChain::root();
        let mut prev = Digest256::ZERO;
        for (i, blk) in blocks.iter().enumerate() {
            chain = chain.extend(blk.tokens);
            let key = PrefixCacheKey::derive(&compat, &blk.scope, &chain);
            let (kv, next_gdn, gdn_bytes) =
                synthesize_block_kv_gdn_state(&expected_gdn, blk.tokens, i as u32, 1024, 128);
            expected_gdn = next_gdn;
            let snap =
                PrefixBlockSnapshot::build(&key, &compat, prev, (i * 64) as u32, 64, kv, gdn_bytes)
                    .expect("build snap");
            worker_a
                .put_block(
                    key.clone(),
                    blk.scope.clone(),
                    &auth_alice,
                    snap,
                    model.accounted_bytes_per_block(),
                )
                .expect("put block");
            expected_keys.push((key.clone(), blk.scope.clone()));
            prev = key.key_digest;
        }
        // Drop worker_a completely to simulate a full process restart!
    }

    // Reopen worker A from disk (cold restart): Tier 0 is empty, Tier 1 recovers all 4 blocks!
    let mut worker_a_restarted = TieredPrefixStore::open(cfg_a).expect("cold restart worker A");
    assert_eq!(worker_a_restarted.index().tier0_bytes(), 0);
    let m_a = worker_a_restarted
        .match_prefix(&compat, &auth_alice, &blocks)
        .expect("match after cold restart");
    assert_eq!(m_a.matched_blocks.len(), 4);
    assert_eq!(m_a.tier1_blocks, 4);

    // Load last block from Tier 1 and verify bit-exact GDN state equality!
    let (last_snap, tier) = worker_a_restarted
        .load_verified_block(
            &expected_keys[3].0,
            &compat,
            &expected_keys[3].1,
            &auth_alice,
            true,
        )
        .expect("load block 3")
        .expect("present in tier 1");
    assert_eq!(tier, StorageTier::Tier1WarmDisk);
    assert_eq!(
        decode_gdn_f32_state(&last_snap.gdn_payload).expect("gdn"),
        expected_gdn
    );

    // Open fresh Worker B sharing Tier 2 object store: Bob (same org "hanzo-eng") can reuse
    // PublicSystem (block 0) + OrgShared("hanzo-eng") (block 1), but CANNOT read Alice's private blocks!
    let cfg_b = TieredStoreConfig {
        tier0_budget_bytes: 16 * model.accounted_bytes_per_block(),
        tier1_budget_bytes: 16 * model.accounted_bytes_per_block(),
        tier1_dir: Some(t1_b),
        tier2_dir: Some(t2_shared),
        policy: PersistencePolicy::HbmOnly,
        enable_tier2_bandwidth_guard: true,
        tier2_link_bandwidth_bps: model.tier2_nominal_bw_bps,
        tier2_rtt_sec: model.tier2_rtt_sec,
        local_prefill_sec_per_token: model.prefill_sec_per_token(),
    };
    let mut worker_b = TieredPrefixStore::open(cfg_b).expect("open worker B");
    let bob_blocks =
        partition_scoped_blocks(&tokens, 64, 1, 1, Some("hanzo-eng"), "bob").expect("bob blocks");
    let m_b = worker_b
        .match_prefix(&compat, &auth_bob, &bob_blocks)
        .expect("bob prefix match");
    assert_eq!(
        m_b.matched_blocks.len(),
        2,
        "Bob must reuse only Public + OrgShared blocks"
    );
    assert_eq!(m_b.tier2_blocks, 2);

    // Direct cross-tenant load attempt by Bob for Alice's private block 2 MUST fail closed!
    assert!(matches!(
        worker_b.load_verified_block(
            &expected_keys[2].0,
            &compat,
            &expected_keys[2].1,
            &auth_bob,
            false
        ),
        Err(PrefixCacheError::UnauthorizedTenantAccess { .. })
    ));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn prefix_aware_scheduler_routes_to_warm_worker_and_enforces_load_cap() {
    let dir = fresh_test_dir("scheduler");
    let model = ModelSpec::hybrid_9b();
    let compat = model.compatibility_profile(b"tok-v1");
    let mut wrong_compat = compat.clone();
    wrong_compat.quant_format = QuantFormat::Int8Symmetric;

    let make_store = |idx: usize| {
        TieredPrefixStore::open(TieredStoreConfig {
            tier0_budget_bytes: 32 * model.accounted_bytes_per_block(),
            tier1_budget_bytes: 0,
            tier1_dir: Some(dir.join(format!("w{idx}_t1"))),
            tier2_dir: None,
            policy: PersistencePolicy::HbmOnly,
            enable_tier2_bandwidth_guard: true,
            tier2_link_bandwidth_bps: model.tier2_nominal_bw_bps,
            tier2_rtt_sec: model.tier2_rtt_sec,
            local_prefill_sec_per_token: model.prefill_sec_per_token(),
        })
        .expect("open store")
    };

    let mut w0 = WorkerNode::new("worker-0", compat.clone(), 6, make_store(0));
    let w1 = WorkerNode::new("worker-1", compat.clone(), 6, make_store(1));
    let w2_incompat = WorkerNode::new("worker-2-int8", wrong_compat, 6, make_store(2));

    let auth = TenantAuthorization::publisher("alice", vec!["eng".to_string()], true);
    let tokens: Vec<u32> = (1..=128).collect();
    let blocks = partition_scoped_blocks(&tokens, 64, 2, 0, None, "alice").expect("blocks");

    // Warm Worker 0 with both blocks in Tier 0 HBM.
    let mut chain = TokenPrefixChain::root();
    let mut prev = Digest256::ZERO;
    for (i, blk) in blocks.iter().enumerate() {
        chain = chain.extend(blk.tokens);
        let key = PrefixCacheKey::derive(&compat, &blk.scope, &chain);
        let (kv, _, gdn) = synthesize_block_kv_gdn_state(&[], blk.tokens, i as u32, 512, 64);
        let snap =
            PrefixBlockSnapshot::build(&key, &compat, prev, (i * 64) as u32, 64, kv, gdn).unwrap();
        w0.store
            .put_block(key.clone(), blk.scope.clone(), &auth, snap, 1024)
            .unwrap();
        prev = key.key_digest;
    }

    let mut sched =
        PrefixAwareScheduler::new(SchedulerConfig::default(), vec![w0, w1, w2_incompat]);

    // Under balanced load, scheduler routes to Worker 0 (deepest hot prefix overlap).
    let dec1 = sched.route(&compat, &auth, &blocks).expect("route 1");
    assert_eq!(dec1.worker_id, "worker-0");
    assert_eq!(dec1.prefix_match.tier0_blocks, 2);
    assert!(!dec1.load_guard_rerouted);

    // Overload Worker 0 beyond max_load_imbalance -> scheduler MUST protect against hot-spot
    // and route to compatible Worker 1 (never to incompatible Worker 2!).
    sched.workers_mut()[0].active_requests = 4;
    sched.workers_mut()[1].active_requests = 0;
    let dec2 = sched.route(&compat, &auth, &blocks).expect("route 2");
    assert_eq!(dec2.worker_id, "worker-1");
    assert!(dec2.load_guard_rerouted);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn adversarial_200_probe_tenant_isolation_audit_has_zero_leaks() {
    let dir = fresh_test_dir("audit_200");
    let corpus = AgentCorpus::load_from_repo(&repo_root()).expect("load repo corpus");
    let audit = run_tenant_isolation_audit(&corpus, &dir).expect("run 200-probe audit");
    assert_eq!(audit.total_probes, 200);
    assert_eq!(audit.cross_tenant_leaks, 0);
    assert_eq!(audit.cross_tenant_private_rejected, 40);
    assert_eq!(audit.cross_org_rejected, 40);
    assert_eq!(audit.scope_blending_rejected, 40);
    assert_eq!(audit.compat_mismatch_rejected, 40);
    assert_eq!(audit.corrupt_or_version_rejected, 40);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn honest_negative_regimes_demonstrate_exact_overhead_and_crossover_boundaries() {
    let dir = fresh_test_dir("negative_regimes");
    let corpus = AgentCorpus::load_from_repo(&repo_root()).expect("load repo corpus");
    let neg = run_negative_regimes(&corpus, DEFAULT_BENCH_SEED, &dir).expect("run negatives");

    // Negative 1: 0% prefix overlap causes zero savings and positive wall-time overhead.
    assert_eq!(
        neg.unique_zero_reuse_tiered.recomputed_tokens_saved_pct,
        0.0
    );
    assert!(
        neg.unique_zero_reuse_wall_regression_pct > 0.0,
        "zero-reuse prompts must show honest control-plane overhead regression"
    );

    // Negative 2: Block-0 mutation invalidates entire downstream chain (0% warm savings).
    assert_eq!(neg.early_mutation_tiered.recomputed_tokens_saved_pct, 0.0);
    assert!(neg.early_mutation_wall_regression_pct > 0.0);

    // Negative 3: 27B over 100 MB/s slow link regresses TTFT severely without guard,
    // and bandwidth admission guard recovers to local recompute!
    assert!(neg.slow_tier2_27b_crossover_mbps > neg.slow_tier2_27b_link_mbps);
    assert!(
        neg.slow_tier2_27b_unguarded_fetch_ttft_ms
            > neg.slow_tier2_27b_local_recompute_ttft_ms * 5.0
    );
    assert!(neg.slow_tier2_27b_guarded_ttft_ms < neg.slow_tier2_27b_unguarded_fetch_ttft_ms * 0.25);
    assert_eq!(neg.slow_tier2_27b_guard_skips, 32);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn full_4b_9b_27b_matrix_suite_verifies_bit_exact_and_tiered_speedup() {
    let report =
        run_full_benchmark_suite(&repo_root(), DEFAULT_BENCH_SEED).expect("run full matrix suite");
    assert_eq!(report.matrix_scorecards.len(), 12);
    assert_eq!(report.isolation_audit.cross_tenant_leaks, 0);

    for sc in &report.matrix_scorecards {
        assert_eq!(
            sc.bit_exact_verified_turns, sc.total_turns,
            "all turns must be bit-exact for {} / {}",
            sc.model, sc.regime
        );
        assert_eq!(sc.quality_regression, 0.0);
    }

    // Verify Hanzo 3-tier cache beats all three baselines on recomputed token savings and p95 warm TTFT
    // across 4B, 9B, and 27B!
    for model_name in ["4B-Hybrid", "9B-Hybrid", "27B-Hybrid"] {
        let base = report
            .matrix_scorecards
            .iter()
            .find(|s| s.model == model_name && s.regime == "no_cache_baseline")
            .unwrap();
        let hbm = report
            .matrix_scorecards
            .iter()
            .find(|s| s.model == model_name && s.regime == "hbm_only_radix_cache")
            .unwrap();
        let naive = report
            .matrix_scorecards
            .iter()
            .find(|s| s.model == model_name && s.regime == "naive_disk_kv_every_turn")
            .unwrap();
        let tiered = report
            .matrix_scorecards
            .iter()
            .find(|s| s.model == model_name && s.regime == "hanzo_tiered_prefix_cache")
            .unwrap();

        assert!(
            tiered.recomputed_tokens_saved_pct > 75.0,
            "{model_name} tiered should save >75% recomputed tokens, got {:.2}%",
            tiered.recomputed_tokens_saved_pct
        );
        assert!(tiered.warm_ttft_p95_ms < base.warm_ttft_p95_ms);
        assert!(tiered.warm_ttft_p95_ms < hbm.warm_ttft_p95_ms);
        assert!(tiered.warm_ttft_p95_ms < naive.warm_ttft_p95_ms);
        assert!(tiered.agent_loop_wall_time_ms < base.agent_loop_wall_time_ms);
        assert!(
            tiered.tier1_hits > 0,
            "{model_name} must hit Tier 1 warm SSD"
        );
        assert!(
            tiered.tier2_hits > 0,
            "{model_name} must hit Tier 2 cold shared object store"
        );
    }
}

#[test]
fn cold_restart_rejects_corrupted_on_disk_snapshot_fail_closed() {
    let dir = fresh_test_dir("corrupt_restart");
    let t1_dir = dir.join("tier1");
    let model = ModelSpec::hybrid_4b();
    let compat = model.compatibility_profile(b"tok-v1");
    let auth = TenantAuthorization::publisher("alice", vec!["eng".to_string()], true);

    let cfg = TieredStoreConfig {
        tier0_budget_bytes: 8 * model.accounted_bytes_per_block(),
        tier1_budget_bytes: 32 * model.accounted_bytes_per_block(),
        tier1_dir: Some(t1_dir.clone()),
        tier2_dir: None,
        policy: PersistencePolicy::NaivePersistEveryTurn,
        enable_tier2_bandwidth_guard: true,
        tier2_link_bandwidth_bps: model.tier2_nominal_bw_bps,
        tier2_rtt_sec: model.tier2_rtt_sec,
        local_prefill_sec_per_token: model.prefill_sec_per_token(),
    };

    let tokens: Vec<u32> = (10..74).collect();
    let scope = TenantScope::PublicSystem;
    let chain = TokenPrefixChain::root().extend(&tokens);
    let key = PrefixCacheKey::derive(&compat, &scope, &chain);
    {
        let mut store = TieredPrefixStore::open(cfg.clone()).expect("open");
        let (kv, _, gdn) = synthesize_block_kv_gdn_state(&[], &tokens, 0, 512, 64);
        let snap = PrefixBlockSnapshot::build(&key, &compat, Digest256::ZERO, 0, 64, kv, gdn)
            .expect("build");
        store
            .put_block(
                key.clone(),
                scope.clone(),
                &auth,
                snap,
                model.accounted_bytes_per_block(),
            )
            .expect("put");
    }

    // Corrupt a byte inside the persisted KV payload on disk!
    let snap_path = TieredPrefixStore::snapshot_path(&t1_dir, &key.key_digest);
    let mut raw = fs::read(&snap_path).expect("read persisted snapshot");
    raw[SNAPSHOT_HEADER_BYTES + 12] ^= 0xff;
    fs::write(&snap_path, &raw).expect("write corrupted snapshot");

    // Cold restart MUST fail closed: 0 recovered nodes, 1 corrupt rejection, and direct load errors!
    let mut restarted = TieredPrefixStore::open(cfg).expect("open after corruption");
    assert_eq!(restarted.index().len(), 0);
    assert_eq!(restarted.metrics().corrupt_rejections, 1);
    assert!(matches!(
        restarted.load_verified_block(&key, &compat, &scope, &auth, false),
        Err(PrefixCacheError::KvCrcMismatch { .. })
    ));

    let _ = fs::remove_dir_all(&dir);
}
