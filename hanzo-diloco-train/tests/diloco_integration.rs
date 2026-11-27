//! Integration tests for `hanzo-diloco-train` covering grounded corpus
//! verification, exact LoRA training, live multi-socket TCP 3-node clusters,
//! mid-run worker `hard_kill()` churn + live TCP VFS checkpoint recovery,
//! all 6 adversarial/Byzantine attack rejections, and all 3 negative regimes.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use hanzo_diloco_train::bench::{
    compute_analytical_scale_projections, render_dossier_json, run_adversarial_defense_suite,
    run_full_benchmark,
};
use hanzo_diloco_train::checkpoint::{
    CheckpointError, TrainingCheckpoint, VfsCheckpointStore, HANZO_ELASTIC_TRAIN_PROTOCOL,
};
use hanzo_diloco_train::cluster::{
    ChurnSchedule, ClusterRunConfig, ElasticTrainingCluster, WorkerSlotSpec,
};
use hanzo_diloco_train::corpus::{GroundedCorpus, EXPECTED_CORPUS_SHA256};
use hanzo_diloco_train::model::{SmallZenConfig, SmallZenLm};
use hanzo_diloco_train::optimizer::{
    decode_pseudo_gradient, encode_pseudo_gradient, AggregationRule, CompressionMode, InnerAdamW,
    InnerAdamWConfig, OuterNesterovAggregator, OuterOptimizerConfig, TrustTier,
};
use hanzo_diloco_train::sha256::sha256_hex;
use hanzo_diloco_train::transport::{TrainingRoundReceipt, WanLinkProfile, RECEIPT_VERSION};
use hanzo_diloco_train::worker::WorkerBehavior;

static TEST_DIR_SEQ: AtomicU64 = AtomicU64::new(1);

fn unique_test_dir(tag: &str) -> PathBuf {
    let n = TEST_DIR_SEQ.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "hanzo-diloco-it-{}-{}-{}",
        tag,
        std::process::id(),
        n
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create test dir");
    dir
}

fn load_corpus() -> GroundedCorpus {
    let root = GroundedCorpus::discover_repo_root();
    GroundedCorpus::load_from_repo(&root, 16).expect("grounded corpus must load and verify")
}

fn fast_wan_workers(behavior_ap: WorkerBehavior) -> Vec<WorkerSlotSpec> {
    vec![
        WorkerSlotSpec {
            worker_id: "worker-us-east".to_string(),
            wan_profile: WanLinkProfile::us_east(0.05),
            behavior: WorkerBehavior::Honest,
        },
        WorkerSlotSpec {
            worker_id: "worker-eu-west".to_string(),
            wan_profile: WanLinkProfile::eu_west(0.05),
            behavior: WorkerBehavior::Honest,
        },
        WorkerSlotSpec {
            worker_id: "worker-ap-south".to_string(),
            wan_profile: WanLinkProfile::ap_south(0.05),
            behavior: behavior_ap,
        },
    ]
}

#[test]
fn t01_grounded_corpus_sha256_and_disjoint_shards() {
    let corpus = load_corpus();
    assert_eq!(corpus.corpus_sha256, EXPECTED_CORPUS_SHA256);
    assert_eq!(corpus.raw_bytes_len, 140_742);
    let s0 = corpus.shard_for_worker(0, 3);
    let s1 = corpus.shard_for_worker(1, 3);
    let s2 = corpus.shard_for_worker(2, 3);
    assert_eq!(s0.len() + s1.len() + s2.len(), corpus.train_sequences.len());
    assert!(corpus.val_sequences.len() >= 500);
    assert!(corpus.spot_check_sequences.len() >= 200);
}

#[test]
fn t02_lora_model_zero_delta_init_and_param_roundtrip() {
    let cfg = SmallZenConfig::default();
    let mut model = SmallZenLm::new(cfg);
    let params = model.export_lora_params();
    assert_eq!(params.len(), model.lora_param_count());
    assert!(params.len() > 4_000);
    let d1 = model.lora_params_sha256();
    model.load_lora_params(&params).expect("reload params");
    assert_eq!(d1, model.lora_params_sha256());
}

#[test]
fn t03_residency_accounting_separates_full_table_from_single_row() {
    let cfg = SmallZenConfig::default();
    let acc = cfg.residency_accounting_bf16();
    assert_eq!(
        acc.resident_input_embedding_bytes,
        (cfg.vocab_size * cfg.d_model * 2) as u64
    );
    assert_eq!(
        acc.single_row_embedding_lookup_bytes,
        (cfg.d_model * 2) as u64
    );

    let projections = compute_analytical_scale_projections();
    assert_eq!(projections.len(), 3);
    for p in &projections {
        assert_eq!(
            p.resident_input_embedding_bytes,
            p.vocab_size * p.d_model * 2
        );
        assert_eq!(p.single_row_embedding_lookup_bytes, p.d_model * 2);
        assert!(p.lora_int8_round_bytes * 20 < p.full_param_fp32_round_bytes);
        assert!(p.decentralized_usd_per_1m_tokens > 0.0);
    }
}

#[test]
fn t04_int8_error_feedback_compression_ratio_and_accuracy() {
    let dim = 5_776;
    let delta: Vec<f32> = (0..dim)
        .map(|i| ((i as f32) * 0.013).sin() * 0.05)
        .collect();
    let mut ef = vec![0.0f32; dim];
    let fp32 = encode_pseudo_gradient(&delta, CompressionMode::Fp32Exact, None);
    let int8 = encode_pseudo_gradient(
        &delta,
        CompressionMode::Int8BlockSymmetric { block_size: 64 },
        Some(&mut ef),
    );
    let ratio = (fp32.len() as f64) / (int8.len() as f64);
    assert!(
        ratio > 3.5,
        "expected >3.5x INT8 compression, got {ratio:.2}x"
    );
    let dec = decode_pseudo_gradient(&int8, dim).expect("decode int8");
    let max_err = delta
        .iter()
        .zip(dec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-3, "max quant error too large: {max_err}");
}

#[test]
fn t05_inner_adamw_reduces_real_corpus_loss() {
    let corpus = load_corpus();
    let mut model = SmallZenLm::new(SmallZenConfig::default());
    let batch: Vec<&[u8]> = corpus.train_sequences[..8]
        .iter()
        .map(Vec::as_slice)
        .collect();
    let initial = model.eval_metrics(&batch);
    let mut params = model.export_lora_params();
    let mut opt = InnerAdamW::new(params.len(), InnerAdamWConfig::default());

    for _ in 0..12 {
        let (_loss, grads, _toks) = model.forward_and_backward_batch(&batch);
        opt.step(&mut params, &grads);
        model.load_lora_params(&params).unwrap();
    }
    let after = model.eval_metrics(&batch);
    assert!(
        after.loss < initial.loss - 0.5,
        "expected strong real loss drop: before={:.4}, after={:.4}",
        initial.loss,
        after.loss
    );
}

#[test]
fn t06_vfs_checkpoint_store_roundtrip_and_uri() {
    let dir = unique_test_dir("vfs_roundtrip");
    let key = b"test-vfs-signing-key".to_vec();
    let mut store = VfsCheckpointStore::new(dir.clone(), key.clone()).unwrap();
    let ckpt = TrainingCheckpoint::build_signed(
        3,
        12_288,
        "genesis".to_string(),
        SmallZenConfig::default().config_sha256(),
        3.12,
        22.6,
        vec![0.1, -0.2, 0.3, 0.4],
        vec![0.01, 0.02, -0.01, 0.0],
        &key,
    );
    let (uri, written) = store.save_checkpoint(&ckpt).unwrap();
    assert!(uri.starts_with("vfs://checkpoints/"));
    assert!(written > 128);
    let loaded = store.load_verified(&ckpt.checkpoint_sha256).unwrap();
    assert_eq!(loaded.round_id, 3);
    assert_eq!(loaded.params, ckpt.params);
    assert_eq!(loaded.outer_velocity, ckpt.outer_velocity);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn t07_corrupted_checkpoint_rejected_and_falls_back_to_last_verified() {
    let dir = unique_test_dir("vfs_corrupt_fallback");
    let key = b"fallback-key".to_vec();
    let mut store = VfsCheckpointStore::new(dir.clone(), key.clone()).unwrap();

    let c1 = TrainingCheckpoint::build_signed(
        1,
        4096,
        "genesis".to_string(),
        "cfg".to_string(),
        3.8,
        44.7,
        vec![1.0, 2.0, 3.0],
        vec![0.0, 0.0, 0.0],
        &key,
    );
    let c2 = TrainingCheckpoint::build_signed(
        2,
        8192,
        c1.checkpoint_sha256.clone(),
        "cfg".to_string(),
        3.4,
        29.9,
        vec![1.5, 2.5, 3.5],
        vec![0.1, 0.1, 0.1],
        &key,
    );
    store.save_checkpoint(&c1).unwrap();
    store.save_checkpoint(&c2).unwrap();

    // Corrupt checkpoint 2 on disk.
    let p2 = store.path_for_digest(&c2.checkpoint_sha256);
    let mut raw = fs::read(&p2).unwrap();
    raw[20] ^= 0xFF;
    fs::write(&p2, &raw).unwrap();

    assert!(matches!(
        store.load_verified(&c2.checkpoint_sha256),
        Err(CheckpointError::DigestMismatch { .. })
    ));
    let (recovered, rejected) = store.load_latest_verified_with_fallback().unwrap();
    assert_eq!(recovered.round_id, 1);
    assert_eq!(rejected, vec![c2.checkpoint_sha256]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn t08_forged_checkpoint_hmac_rejected() {
    let dir = unique_test_dir("vfs_bad_hmac");
    let key = b"real-key".to_vec();
    let mut store = VfsCheckpointStore::new(dir.clone(), key).unwrap();
    let ckpt = TrainingCheckpoint::build_signed(
        1,
        512,
        "genesis".to_string(),
        "cfg".to_string(),
        4.0,
        54.6,
        vec![0.5; 4],
        vec![0.0; 4],
        b"wrong-key",
    );
    store.save_checkpoint(&ckpt).unwrap();
    assert_eq!(
        store.load_verified(&ckpt.checkpoint_sha256).unwrap_err(),
        CheckpointError::InvalidSignature
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn t09_training_round_receipt_hmac_detects_tampering() {
    let mut r = TrainingRoundReceipt {
        receipt_version: RECEIPT_VERSION.to_string(),
        round_id: 4,
        worker_id: "worker-us-east".to_string(),
        region: "us-east".to_string(),
        base_checkpoint_sha256: sha256_hex(b"base"),
        delta_sha256: sha256_hex(b"delta"),
        inner_steps_completed: 16,
        tokens_processed: 2048,
        local_loss_before: 4.1,
        local_loss_after: 3.2,
        delta_l2_norm: 0.85,
        wire_bytes: 6150,
        trust_tier: TrustTier::SignedDeltaAndLossSpotCheck,
        hmac_sha256: String::new(),
    };
    let key = b"worker-secret-key";
    r.sign(key);
    assert!(r.verify_hmac(key));
    assert!(!r.verify_hmac(b"wrong-key"));
    r.tokens_processed += 1;
    assert!(!r.verify_hmac(key));
}

#[test]
fn t10_live_3node_wan_cluster_converges_without_hanzo_coordinator() {
    let corpus = load_corpus();
    let dir = unique_test_dir("live_3node_steady");
    let cfg = ClusterRunConfig {
        scenario_name: "it_3node_wan_steady".to_string(),
        model_config: SmallZenConfig::default(),
        workers: fast_wan_workers(WorkerBehavior::Honest),
        outer_rounds: 5,
        inner_steps_per_round: 12,
        batch_size: 8,
        inner_lr: 9e-3,
        outer_config: OuterOptimizerConfig::default(),
        compression: CompressionMode::Int8BlockSymmetric { block_size: 64 },
        min_quorum: 2,
        eval_sequences: 64,
        spot_check_sequences: 16,
        seed: 1350,
        churn: None,
        checkpoint_dir: dir.clone(),
    };
    let report = ElasticTrainingCluster::run(&cfg, &corpus).expect("cluster run");
    assert_eq!(
        report.manifest.protocol_version,
        HANZO_ELASTIC_TRAIN_PROTOCOL
    );
    assert!(!report.manifest.hanzo_coordinator_required);
    assert_eq!(report.completed_rounds, 5);
    assert!((report.completion_rate - 1.0).abs() < 1e-9);
    assert!(
        report.final_eval.loss < report.initial_eval.loss - 0.6,
        "expected significant held-out loss drop: {:.4} -> {:.4}",
        report.initial_eval.loss,
        report.final_eval.loss
    );
    assert!(report.total_wire_bytes > 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn t11_low_comm_wan_achieves_major_network_reduction_vs_sync_h1() {
    let corpus = load_corpus();
    let dir_sync = unique_test_dir("comm_sync");
    let dir_low = unique_test_dir("comm_low");

    let sync_cfg = ClusterRunConfig {
        scenario_name: "sync_h1".to_string(),
        model_config: SmallZenConfig::default(),
        workers: fast_wan_workers(WorkerBehavior::Honest),
        outer_rounds: 32,
        inner_steps_per_round: 1,
        batch_size: 6,
        inner_lr: 8e-3,
        outer_config: OuterOptimizerConfig {
            outer_lr: 1.0,
            nesterov_momentum: 0.0,
            aggregation_rule: AggregationRule::ByzantineTrimmedMean { trim_each_side: 0 },
            enable_loss_spot_check: false,
            ..OuterOptimizerConfig::default()
        },
        compression: CompressionMode::Fp32Exact,
        min_quorum: 3,
        eval_sequences: 32,
        spot_check_sequences: 8,
        seed: 1350,
        churn: None,
        checkpoint_dir: dir_sync.clone(),
    };
    let low_cfg = ClusterRunConfig {
        scenario_name: "lowcomm_h16_int8".to_string(),
        outer_rounds: 2,
        inner_steps_per_round: 16,
        compression: CompressionMode::Int8BlockSymmetric { block_size: 64 },
        outer_config: OuterOptimizerConfig::default(),
        checkpoint_dir: dir_low.clone(),
        ..sync_cfg.clone()
    };

    let sync_rep = ElasticTrainingCluster::run(&sync_cfg, &corpus).unwrap();
    let low_rep = ElasticTrainingCluster::run(&low_cfg, &corpus).unwrap();

    assert_eq!(sync_rep.total_tokens_trained, low_rep.total_tokens_trained);
    let wire_ratio = (sync_rep.total_wire_bytes as f64) / (low_rep.total_wire_bytes as f64);
    let upload_ratio =
        (sync_rep.total_delta_upload_bytes as f64) / (low_rep.total_delta_upload_bytes as f64);
    assert!(
        wire_ratio >= 18.0,
        "expected >=18x total wire reduction, got {wire_ratio:.2}x"
    );
    assert!(
        upload_ratio >= 55.0,
        "expected >=55x delta upload reduction, got {upload_ratio:.2}x"
    );
    let _ = fs::remove_dir_all(&dir_sync);
    let _ = fs::remove_dir_all(&dir_low);
}

#[test]
fn t12_live_worker_hard_kill_churn_and_tcp_checkpoint_recovery() {
    let corpus = load_corpus();
    let dir = unique_test_dir("live_churn_recovery");
    let cfg = ClusterRunConfig {
        scenario_name: "it_live_churn".to_string(),
        model_config: SmallZenConfig::default(),
        workers: fast_wan_workers(WorkerBehavior::Honest),
        outer_rounds: 6,
        inner_steps_per_round: 12,
        batch_size: 8,
        inner_lr: 9e-3,
        outer_config: OuterOptimizerConfig::default(),
        compression: CompressionMode::Int8BlockSymmetric { block_size: 64 },
        min_quorum: 2,
        eval_sequences: 64,
        spot_check_sequences: 16,
        seed: 1350,
        churn: Some(ChurnSchedule {
            victim_worker_id: "worker-eu-west".to_string(),
            kill_before_round: 3,
            replacement_worker_id: "worker-eu-replacement".to_string(),
            rejoin_before_round: 5,
            replacement_wan_profile: WanLinkProfile::eu_west(0.05),
        }),
        checkpoint_dir: dir.clone(),
    };
    let report = ElasticTrainingCluster::run(&cfg, &corpus).expect("churn run succeeds");
    assert_eq!(report.completed_rounds, 6);
    assert!((report.completion_rate - 1.0).abs() < 1e-9);
    // Rounds 3 & 4 proceed with 2 surviving workers; rounds 5 & 6 return to 3 active workers.
    assert_eq!(report.rounds[2].accepted_workers, 2);
    assert_eq!(report.rounds[3].accepted_workers, 2);
    assert_eq!(report.rounds[4].accepted_workers, 3);
    assert_eq!(report.rounds[5].accepted_workers, 3);
    let rec = report
        .recovery_report
        .as_ref()
        .expect("checkpoint recovery report must exist");
    assert_eq!(rec.worker_id, "worker-eu-replacement");
    assert_eq!(rec.recovered_round_id, 4);
    assert!(rec.recovery_wire_bytes > 20_000);
    assert!(report.final_eval.loss < report.initial_eval.loss - 0.6);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn t13_attack_1_sign_flip_poison_rejected_and_honest_peers_converge() {
    let corpus = load_corpus();
    let dir = unique_test_dir("adv_sign_flip");
    let cfg = ClusterRunConfig {
        scenario_name: "adv_sign_flip".to_string(),
        model_config: SmallZenConfig::default(),
        workers: fast_wan_workers(WorkerBehavior::SignFlipPoison { scale: 6.0 }),
        outer_rounds: 3,
        inner_steps_per_round: 8,
        batch_size: 6,
        inner_lr: 9e-3,
        outer_config: OuterOptimizerConfig::default(),
        compression: CompressionMode::Int8BlockSymmetric { block_size: 64 },
        min_quorum: 2,
        eval_sequences: 32,
        spot_check_sequences: 16,
        seed: 1350,
        churn: None,
        checkpoint_dir: dir.clone(),
    };
    let report = ElasticTrainingCluster::run(&cfg, &corpus).unwrap();
    assert!(report.total_rejected_updates >= 3);
    assert!(report
        .all_rejection_reasons
        .iter()
        .any(|r| { r.contains("validation_loss_regression") || r.contains("norm_outlier") }));
    assert!(report.final_eval.loss < report.initial_eval.loss - 0.3);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn t14_attack_2_free_rider_zero_delta_rejected() {
    let corpus = load_corpus();
    let dir = unique_test_dir("adv_freerider");
    let cfg = ClusterRunConfig {
        scenario_name: "adv_freerider".to_string(),
        model_config: SmallZenConfig::default(),
        workers: fast_wan_workers(WorkerBehavior::FreeRiderZeroDelta),
        outer_rounds: 2,
        inner_steps_per_round: 6,
        batch_size: 6,
        inner_lr: 9e-3,
        outer_config: OuterOptimizerConfig::default(),
        compression: CompressionMode::Int8BlockSymmetric { block_size: 64 },
        min_quorum: 2,
        eval_sequences: 32,
        spot_check_sequences: 12,
        seed: 1350,
        churn: None,
        checkpoint_dir: dir.clone(),
    };
    let report = ElasticTrainingCluster::run(&cfg, &corpus).unwrap();
    assert_eq!(report.total_rejected_updates, 2);
    assert!(report
        .all_rejection_reasons
        .iter()
        .all(|r| r.contains("free_rider_zero_delta")));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn t15_attack_3_stale_round_replay_rejected() {
    let corpus = load_corpus();
    let dir = unique_test_dir("adv_stale_round");
    let cfg = ClusterRunConfig {
        scenario_name: "adv_stale_round".to_string(),
        model_config: SmallZenConfig::default(),
        workers: fast_wan_workers(WorkerBehavior::StaleRoundReplay { replay_round_id: 0 }),
        outer_rounds: 2,
        inner_steps_per_round: 6,
        batch_size: 6,
        inner_lr: 9e-3,
        outer_config: OuterOptimizerConfig::default(),
        compression: CompressionMode::Int8BlockSymmetric { block_size: 64 },
        min_quorum: 2,
        eval_sequences: 32,
        spot_check_sequences: 12,
        seed: 1350,
        churn: None,
        checkpoint_dir: dir.clone(),
    };
    let report = ElasticTrainingCluster::run(&cfg, &corpus).unwrap();
    assert_eq!(report.total_rejected_updates, 2);
    assert!(report
        .all_rejection_reasons
        .iter()
        .all(|r| r.contains("stale_round_replay")));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn t16_attack_4_wrong_base_checkpoint_rejected() {
    let corpus = load_corpus();
    let dir = unique_test_dir("adv_wrong_base");
    let cfg = ClusterRunConfig {
        scenario_name: "adv_wrong_base".to_string(),
        model_config: SmallZenConfig::default(),
        workers: fast_wan_workers(WorkerBehavior::WrongBaseCheckpoint),
        outer_rounds: 2,
        inner_steps_per_round: 6,
        batch_size: 6,
        inner_lr: 9e-3,
        outer_config: OuterOptimizerConfig::default(),
        compression: CompressionMode::Int8BlockSymmetric { block_size: 64 },
        min_quorum: 2,
        eval_sequences: 32,
        spot_check_sequences: 12,
        seed: 1350,
        churn: None,
        checkpoint_dir: dir.clone(),
    };
    let report = ElasticTrainingCluster::run(&cfg, &corpus).unwrap();
    assert_eq!(report.total_rejected_updates, 2);
    assert!(report
        .all_rejection_reasons
        .iter()
        .all(|r| r.contains("base_checkpoint_mismatch")));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn t17_attack_5_tampered_wire_bytes_and_forged_hmac_rejected() {
    let corpus = load_corpus();
    let dir_a = unique_test_dir("adv_tamper");
    let dir_b = unique_test_dir("adv_hmac");
    let base_cfg = |dir: PathBuf, b: WorkerBehavior| ClusterRunConfig {
        scenario_name: "adv_wire".to_string(),
        model_config: SmallZenConfig::default(),
        workers: fast_wan_workers(b),
        outer_rounds: 2,
        inner_steps_per_round: 4,
        batch_size: 4,
        inner_lr: 9e-3,
        outer_config: OuterOptimizerConfig::default(),
        compression: CompressionMode::Int8BlockSymmetric { block_size: 64 },
        min_quorum: 2,
        eval_sequences: 16,
        spot_check_sequences: 8,
        seed: 1350,
        churn: None,
        checkpoint_dir: dir,
    };
    let rep_a = ElasticTrainingCluster::run(
        &base_cfg(dir_a.clone(), WorkerBehavior::TamperedWireBytes),
        &corpus,
    )
    .unwrap();
    let rep_b = ElasticTrainingCluster::run(
        &base_cfg(dir_b.clone(), WorkerBehavior::ForgedReceiptHmac),
        &corpus,
    )
    .unwrap();
    assert!(rep_a
        .all_rejection_reasons
        .iter()
        .any(|r| r.contains("delta_digest_mismatch")));
    assert!(rep_b
        .all_rejection_reasons
        .iter()
        .any(|r| r.contains("invalid_hmac")));
    let _ = fs::remove_dir_all(&dir_a);
    let _ = fs::remove_dir_all(&dir_b);
}

#[test]
fn t18_full_6_of_6_adversarial_defense_suite_passes() {
    let corpus = load_corpus();
    let dir = unique_test_dir("adv_suite_6of6");
    let suite =
        run_adversarial_defense_suite(1350, &SmallZenConfig::default(), &corpus, &dir).unwrap();
    assert_eq!(suite.total_attacks, 6);
    assert_eq!(suite.detected_count, 6);
    assert_eq!(suite.trust_tier, TrustTier::SignedDeltaAndLossSpotCheck);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn t19_negative_unprotected_mean_diverges_under_byzantine_poison() {
    let corpus = load_corpus();
    let dir = unique_test_dir("neg_unprotected");
    let cfg = ClusterRunConfig {
        scenario_name: "neg_unprotected".to_string(),
        model_config: SmallZenConfig::default(),
        workers: fast_wan_workers(WorkerBehavior::SignFlipPoison { scale: 8.0 }),
        outer_rounds: 4,
        inner_steps_per_round: 10,
        batch_size: 6,
        inner_lr: 9e-3,
        outer_config: OuterOptimizerConfig {
            outer_lr: 0.75,
            nesterov_momentum: 0.75,
            aggregation_rule: AggregationRule::NaiveMeanUnprotected,
            enable_loss_spot_check: false,
            ..OuterOptimizerConfig::default()
        },
        compression: CompressionMode::Int8BlockSymmetric { block_size: 64 },
        min_quorum: 2,
        eval_sequences: 32,
        spot_check_sequences: 12,
        seed: 1350,
        churn: None,
        checkpoint_dir: dir.clone(),
    };
    let report = ElasticTrainingCluster::run(&cfg, &corpus).unwrap();
    assert!(
        report.final_eval.loss > report.initial_eval.loss,
        "unprotected mean should diverge under 1/3 -8x sign-flip: {:.4} -> {:.4}",
        report.initial_eval.loss,
        report.final_eval.loss
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn t20_outer_nesterov_velocity_accelerates_consistent_pseudo_gradient() {
    let mut agg = OuterNesterovAggregator::new(
        4,
        OuterOptimizerConfig {
            outer_lr: 0.5,
            nesterov_momentum: 0.8,
            ..OuterOptimizerConfig::default()
        },
    );
    let mut params = vec![1.0f32; 4];
    let delta = vec![0.1f32; 4];
    agg.apply_outer_nesterov(&mut params, &delta);
    let step1 = 1.0 - params[0];
    let prev = params[0];
    agg.apply_outer_nesterov(&mut params, &delta);
    let step2 = prev - params[0];
    assert!(
        step2 > step1,
        "Nesterov momentum should accelerate aligned step: step1={step1}, step2={step2}"
    );
}

#[test]
fn t21_end_to_end_benchmark_dossier_and_json_contract() {
    let root = GroundedCorpus::discover_repo_root();
    let out_dir = unique_test_dir("bench_e2e");
    let dossier = run_full_benchmark(1350, &root, Some(&out_dir)).expect("full benchmark passes");

    assert_eq!(dossier.corpus_sha256, EXPECTED_CORPUS_SHA256);
    assert_eq!(dossier.adversarial_suite.detected_count, 6);
    assert_eq!(dossier.adversarial_suite.total_attacks, 6);
    assert_eq!(dossier.negative_regimes.len(), 3);
    assert!(dossier
        .negative_regimes
        .iter()
        .all(|n| n.verdict == "REJECTED"));
    assert!(dossier.wire_bytes_reduction_factor >= 16.0);
    assert!((dossier.wan_3node_churn_recovery.completion_rate - 1.0).abs() < 1e-9);
    assert!(
        dossier.wan_3node_steady.final_eval.loss < dossier.wan_3node_steady.initial_eval.loss - 0.8
    );
    assert!(
        dossier.wan_3node_churn_recovery.final_eval.loss
            < dossier.wan_3node_churn_recovery.initial_eval.loss - 0.8
    );

    let json = render_dossier_json(&dossier);
    assert!(json.contains("\"hanzo_coordinator_required\": false"));
    assert!(json.contains("\"trust_tier\": \"signed_delta_and_loss_spot_check\""));
    assert!(json.contains("\"resident_input_embedding_bytes_bf16\":"));
    assert!(json.contains("\"single_row_embedding_lookup_bytes_bf16\":"));
    assert!(out_dir.join("results.json").is_file());
    assert!(out_dir.join("results.json.sha256").is_file());

    let _ = fs::remove_dir_all(&out_dir);
}
