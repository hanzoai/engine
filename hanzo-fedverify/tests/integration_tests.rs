// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `hanzo-fedverify` (`hanzoai/engine#73`).
//!
//! Verifies >= 24 simulated participants, all 5 adversarial attack classes,
//! collusion resistance, honest-update rejection bounds, cryptographic tamper/replay
//! rejection, LoRA adapter updates, both negative regimes, sovereign ledger replay,
//! and bit-for-bit determinism.

use hanzo_fedverify::aggregator::{AggregationRegime, DefenseConfig, FederatedAggregator};
use hanzo_fedverify::identity::{ParticipantSigner, RejectionReason, UpdateKind};
use hanzo_fedverify::ledger::{verify_ledger_sovereign, FederationAuditLedger};
use hanzo_fedverify::model::{
    CorpusDataset, FederatedModel, PartitionStrategy, FEATURE_DIM, MODEL_DIM, NUM_CLASSES,
};
use hanzo_fedverify::simulation::{run_simulation, ParticipantRole, SimulationConfig};

fn load_repo_dataset() -> CorpusDataset {
    let root = CorpusDataset::find_repo_root();
    CorpusDataset::load_from_repo_root(&root, 1350)
        .expect("pinned versioned repository corpus files must load")
}

fn default_sim_config(
    name: &str,
    roles: Vec<ParticipantRole>,
    partition: PartitionStrategy,
    lora_rank: Option<u32>,
    defense_config: DefenseConfig,
) -> SimulationConfig {
    SimulationConfig {
        scenario_name: name.to_string(),
        seed: 1350,
        num_participants: roles.len(),
        rounds: 10,
        local_lr: 0.22,
        l2_reg: 1e-4,
        local_epochs: 2,
        partition,
        lora_rank,
        roles,
        defense_config,
    }
}

#[test]
fn at_least_24_participants_mixed_byzantine_end_to_end() {
    let dataset = load_repo_dataset();
    let cfg = default_sim_config(
        "mixed_29pct_24nodes",
        SimulationConfig::mixed_roles(24, 7),
        PartitionStrategy::StratifiedIid,
        None,
        DefenseConfig::balanced_default(),
    );

    let naive = run_simulation(&dataset, &cfg, AggregationRegime::NaiveFedAvgBaseline);
    let krum = run_simulation(&dataset, &cfg, AggregationRegime::TrimmedMeanKrum);
    let dso = run_simulation(&dataset, &cfg, AggregationRegime::HanzoFedVerifyDso);
    let oracle = run_simulation(&dataset, &cfg, AggregationRegime::OracleHonestOnlyCeiling);

    assert!(dso.participants_total >= 24);
    assert_eq!(dso.hanzo_control_plane_calls, 0);
    assert!(dso.sovereign_replay_verified);

    // Naive FedAvg collapses catastrophically under 100x scaling + mixed attacks.
    assert!(naive.final_test_metrics.cross_entropy_loss > 20.0);
    assert_eq!(naive.attack_success_rate, 1.0);

    // TrimmedMeanKrum admits substantial collusion/fake-claim/tamper updates.
    assert!(krum.attack_success_rate > 0.25);

    // HanzoFedVerifyDso stops all attacks with 0% honest false rejections and matches Oracle quality.
    assert_eq!(dso.attack_success_rate, 0.0);
    assert_eq!(dso.honest_rejection_rate, 0.0);
    assert!(
        (dso.final_test_metrics.cross_entropy_loss - oracle.final_test_metrics.cross_entropy_loss)
            .abs()
            < 0.015,
        "dso loss {} should match oracle loss {}",
        dso.final_test_metrics.cross_entropy_loss,
        oracle.final_test_metrics.cross_entropy_loss
    );
    // Verify genuine learning happened on real versioned corpus text.
    assert!(
        dso.final_test_metrics.cross_entropy_loss + 0.20
            < dso.initial_test_metrics.cross_entropy_loss
    );
}

#[test]
fn all_five_attack_classes_detected_and_attributed_with_signed_receipts() {
    let dataset = load_repo_dataset();
    let cases = [
        (
            "sign_flip",
            ParticipantRole::SignFlip,
            RejectionReason::CosineDirectionAnomaly,
        ),
        (
            "scaling_100x",
            ParticipantRole::ScalingExplosion,
            RejectionReason::NormBoundViolation,
        ),
        (
            "collusion_cartel",
            ParticipantRole::CollusionCartel,
            RejectionReason::CollusionClusterDetected,
        ),
        (
            "fake_claim",
            ParticipantRole::FakeClaimedImprovement,
            RejectionReason::ValidationLossRegression,
        ),
        (
            "tamper_replay",
            ParticipantRole::TamperedOrReplay,
            RejectionReason::InvalidSignature,
        ),
    ];

    for (label, role, expected_primary_reason) in cases {
        let cfg = default_sim_config(
            label,
            SimulationConfig::pure_attack_roles(24, 7, role),
            PartitionStrategy::StratifiedIid,
            None,
            DefenseConfig::balanced_default(),
        );
        let rep = run_simulation(&dataset, &cfg, AggregationRegime::HanzoFedVerifyDso);
        assert_eq!(
            rep.attack_success_rate, 0.0,
            "attack {label} must have 0% admission rate under hanzo_fedverify_dso"
        );
        assert_eq!(
            rep.honest_rejection_rate, 0.0,
            "attack {label} must not falsely reject honest IID participants"
        );
        assert!(
            rep.rejection_breakdown
                .get(&expected_primary_reason)
                .copied()
                .unwrap_or(0)
                > 0,
            "expected primary reason {expected_primary_reason} for attack {label}, got {:?}",
            rep.rejection_breakdown
        );
        // Every single RejectionReceipt must verify offline with Ed25519.
        let total_receipts: usize = rep.ledger.rounds.iter().map(|r| r.rejections.len()).sum();
        assert_eq!(total_receipts, rep.malicious_submissions_total);
        for round in &rep.ledger.rounds {
            for receipt in &round.rejections {
                receipt
                    .verify()
                    .expect("every RejectionReceipt Ed25519 signature must verify");
            }
        }
    }
}

#[test]
fn collusion_cartel_hijacks_naive_krum_but_stopped_by_fedverify() {
    let dataset = load_repo_dataset();
    let cfg = default_sim_config(
        "collusion_cartel_29pct",
        SimulationConfig::pure_attack_roles(24, 7, ParticipantRole::CollusionCartel),
        PartitionStrategy::StratifiedIid,
        None,
        DefenseConfig::balanced_default(),
    );

    let krum = run_simulation(&dataset, &cfg, AggregationRegime::TrimmedMeanKrum);
    let dso = run_simulation(&dataset, &cfg, AggregationRegime::HanzoFedVerifyDso);

    // Pure Multi-Krum is hijacked by identical near-zero-distance clones.
    assert!(
        krum.attack_success_rate > 0.50,
        "expected pure Krum to be vulnerable to Sybil cartel, got {}",
        krum.attack_success_rate
    );
    // HanzoFedVerifyDso catches 100% via CollusionClusterDetected.
    assert_eq!(dso.attack_success_rate, 0.0);
    assert_eq!(
        dso.rejection_breakdown
            .get(&RejectionReason::CollusionClusterDetected)
            .copied()
            .unwrap_or(0),
        70
    );
}

#[test]
fn layer1_rejects_unknown_forged_tampered_stale_replay_duplicate_and_malformed() {
    let dataset = load_repo_dataset();
    let alice = ParticipantSigner::from_seed_label("alice", 101);
    let bob = ParticipantSigner::from_seed_label("bob", 102);
    let carol = ParticipantSigner::from_seed_label("carol", 103);
    let dave = ParticipantSigner::from_seed_label("dave", 104);
    let erin = ParticipantSigner::from_seed_label("erin", 105);
    let mallory_unregistered = ParticipantSigner::from_seed_label("mallory", 999);
    let auditor = ParticipantSigner::from_seed_label("auditor", 777);

    let mut agg = FederatedAggregator::new(
        AggregationRegime::HanzoFedVerifyDso,
        DefenseConfig::balanced_default(),
        auditor,
        &[
            alice.identity.clone(),
            bob.identity.clone(),
            carol.identity.clone(),
            dave.identity.clone(),
            erin.identity.clone(),
        ],
    );
    let mut model = FederatedModel::init(1350);
    let base_hash = model.model_hash();
    let (honest_delta, lb, la) = model.local_train_delta(&dataset.train_pool[..200], 0.2, 1e-4, 2);

    // Valid submission for round 0 (nonce 1)
    let valid_alice = alice.sign_update(
        0,
        base_hash,
        UpdateKind::GradientDelta,
        honest_delta.clone(),
        lb,
        la,
        200,
        1000,
        1,
    );
    // Duplicate submission for alice in the same round (nonce 2)
    let dup_alice = alice.sign_update(
        0,
        base_hash,
        UpdateKind::GradientDelta,
        honest_delta.clone(),
        lb,
        la,
        200,
        1000,
        2,
    );
    // Unregistered participant
    let unknown_env = mallory_unregistered.sign_update(
        0,
        base_hash,
        UpdateKind::GradientDelta,
        honest_delta.clone(),
        lb,
        la,
        200,
        1000,
        1,
    );
    // Corrupted Ed25519 signature
    let mut bad_sig = bob.sign_update(
        0,
        base_hash,
        UpdateKind::GradientDelta,
        honest_delta.clone(),
        lb,
        la,
        200,
        1000,
        1,
    );
    bad_sig.signature[10] ^= 0x04;
    // Post-signing payload tamper
    let mut tampered_payload = carol.sign_update(
        0,
        base_hash,
        UpdateKind::GradientDelta,
        honest_delta.clone(),
        lb,
        la,
        200,
        1000,
        1,
    );
    tampered_payload.payload[0] += 1.0;
    // Stale base_model_hash
    let mut stale_hash = base_hash;
    stale_hash[0] ^= 0xff;
    let stale_env = dave.sign_update(
        0,
        stale_hash,
        UpdateKind::GradientDelta,
        honest_delta.clone(),
        lb,
        la,
        200,
        1000,
        1,
    );
    // Malformed NaN payload
    let mut nan_delta = honest_delta.clone();
    nan_delta[3] = f64::NAN;
    let nan_env = erin.sign_update(
        0,
        base_hash,
        UpdateKind::GradientDelta,
        nan_delta,
        lb,
        la,
        200,
        1000,
        1,
    );

    let out0 = agg.process_round(
        0,
        &mut model,
        &[
            valid_alice,
            dup_alice,
            unknown_env,
            bad_sig,
            tampered_payload,
            stale_env,
            nan_env,
        ],
        &dataset.independent_val,
        None,
    );

    assert_eq!(out0.audit_record.accepted.len(), 1);
    assert_eq!(out0.audit_record.accepted[0].participant_id, "alice");
    assert_eq!(
        out0.rejection_breakdown.get(&RejectionReason::ReplayNonce),
        Some(&1)
    );
    assert_eq!(
        out0.rejection_breakdown
            .get(&RejectionReason::UnknownParticipant),
        Some(&1)
    );
    assert_eq!(
        out0.rejection_breakdown
            .get(&RejectionReason::InvalidSignature),
        Some(&2)
    );
    assert_eq!(
        out0.rejection_breakdown
            .get(&RejectionReason::StaleBaseModel),
        Some(&1)
    );
    assert_eq!(
        out0.rejection_breakdown
            .get(&RejectionReason::NonFiniteOrMalformed),
        Some(&1)
    );

    // Round 1: Alice attempts to replay nonce=1 (already consumed in round 0!)
    let next_hash = model.model_hash();
    let replay_alice = alice.sign_update(
        1,
        next_hash,
        UpdateKind::GradientDelta,
        honest_delta,
        lb,
        la,
        200,
        1000,
        1,
    );
    let out1 = agg.process_round(
        1,
        &mut model,
        &[replay_alice],
        &dataset.independent_val,
        None,
    );
    assert_eq!(out1.audit_record.accepted.len(), 0);
    assert_eq!(
        out1.rejection_breakdown.get(&RejectionReason::ReplayNonce),
        Some(&1)
    );
}

#[test]
fn lora_rank4_adapter_updates_verified_and_defend_against_mixed_attacks() {
    let dataset = load_repo_dataset();
    // Sanity-check LoRA factorization wire compression: 4*32 + 12*4 = 176 vs 384 dense.
    let dummy_dense = vec![0.05; MODEL_DIM];
    let lora_payload = FederatedModel::compress_to_lora_adapter(&dummy_dense, 4);
    assert_eq!(lora_payload.len(), 4 * FEATURE_DIM + NUM_CLASSES * 4);
    let kind = UpdateKind::LoraAdapter {
        rank: 4,
        rows: NUM_CLASSES as u32,
        cols: FEATURE_DIM as u32,
    };
    let dense_back = kind
        .materialize_dense(&lora_payload, MODEL_DIM)
        .expect("LoRA adapter materializes cleanly");
    assert_eq!(dense_back.len(), MODEL_DIM);

    let cfg = default_sim_config(
        "lora_r4_mixed",
        SimulationConfig::mixed_roles(24, 7),
        PartitionStrategy::StratifiedIid,
        Some(4),
        DefenseConfig::balanced_default(),
    );
    let dso = run_simulation(&dataset, &cfg, AggregationRegime::HanzoFedVerifyDso);
    let oracle = run_simulation(&dataset, &cfg, AggregationRegime::OracleHonestOnlyCeiling);

    assert_eq!(dso.attack_success_rate, 0.0);
    assert_eq!(dso.honest_rejection_rate, 0.0);
    assert!(
        (dso.final_test_metrics.cross_entropy_loss - oracle.final_test_metrics.cross_entropy_loss)
            .abs()
            < 0.015
    );
}

#[test]
fn negative_regime_supermajority_preserves_safety_and_surfaces_liveness_stall() {
    let dataset = load_repo_dataset();
    let cfg = default_sim_config(
        "supermajority_54pct",
        SimulationConfig::pure_attack_roles(24, 13, ParticipantRole::SuperMajorityDiversePoison),
        PartitionStrategy::StratifiedIid,
        None,
        DefenseConfig::balanced_default(),
    );

    let krum = run_simulation(&dataset, &cfg, AggregationRegime::TrimmedMeanKrum);
    let dso = run_simulation(&dataset, &cfg, AggregationRegime::HanzoFedVerifyDso);

    // Statistical Krum breaks down mathematically when f >= n/2 and worsens the model.
    assert!(krum.attack_success_rate > 0.50);
    assert!(
        krum.final_test_metrics.cross_entropy_loss > krum.initial_test_metrics.cross_entropy_loss
    );

    // HanzoFedVerifyDso preserves safety (0% malicious admission, no model corruption)
    // while transparently stalling liveness (high honest rejection at f >= n/2).
    assert_eq!(dso.attack_success_rate, 0.0);
    assert!(
        dso.final_test_metrics.cross_entropy_loss
            <= dso.initial_test_metrics.cross_entropy_loss + 1e-6
    );
    assert!(dso.honest_rejection_rate > 0.50);
}

#[test]
fn negative_regime_extreme_non_iid_shows_honest_rejection_tradeoff() {
    let dataset = load_repo_dataset();
    let roles = SimulationConfig::mixed_roles(24, 0);

    let strict = run_simulation(
        &dataset,
        &default_sim_config(
            "non_iid_strict",
            roles.clone(),
            PartitionStrategy::ExtremeNonIid,
            None,
            DefenseConfig::strict_defense(),
        ),
        AggregationRegime::HanzoFedVerifyDso,
    );
    let relaxed = run_simulation(
        &dataset,
        &default_sim_config(
            "non_iid_relaxed",
            roles,
            PartitionStrategy::ExtremeNonIid,
            None,
            DefenseConfig::non_iid_relaxed(),
        ),
        AggregationRegime::HanzoFedVerifyDso,
    );

    assert!(
        strict.honest_rejection_rate > relaxed.honest_rejection_rate + 0.50,
        "strict thresholds must reject significantly more Non-IID honest updates than relaxed"
    );
    assert!(
        relaxed.final_test_metrics.cross_entropy_loss
            < strict.final_test_metrics.cross_entropy_loss - 0.10
    );
}

#[test]
fn sovereign_ledger_roundtrip_and_six_tamper_attacks_rejected() {
    let dataset = load_repo_dataset();
    let cfg = default_sim_config(
        "sovereign_audit_29pct",
        SimulationConfig::mixed_roles(24, 7),
        PartitionStrategy::StratifiedIid,
        None,
        DefenseConfig::balanced_default(),
    );
    let rep = run_simulation(&dataset, &cfg, AggregationRegime::HanzoFedVerifyDso);
    let ledger = &rep.ledger;

    // 1. Lossless FVER v1 binary round-trip + independent offline verification.
    let wire = ledger.encode_wire();
    let decoded = FederationAuditLedger::decode_wire(&wire).expect("FVER wire decodes cleanly");
    assert_eq!(&decoded, ledger);
    let summary = verify_ledger_sovereign(&decoded).expect("sovereign replay verifies");
    assert_eq!(summary.hanzo_control_plane_calls, 0);
    assert_eq!(summary.verified_rounds, cfg.rounds);
    assert!(summary.verified_rejections >= 70);

    // 2. Clean-room JSON export contains key ledger commitments.
    let json = ledger.to_json_pretty();
    assert!(json.contains("\"protocol_id\": \"hanzo.fedverify.dso.v1\""));
    assert!(json.contains("\"rejections\": ["));

    // 3. Six distinct tamper attacks must all fail verification.
    // (a) Wire bit-flip
    let mut bad_wire = wire.clone();
    bad_wire[wire.len() / 2] ^= 0x01;
    assert!(FederationAuditLedger::decode_wire(&bad_wire)
        .and_then(|l| verify_ledger_sovereign(&l))
        .is_err());

    // (b) Trailing garbage bytes after EOF
    let mut trailing = wire.clone();
    trailing.push(0xaa);
    assert!(FederationAuditLedger::decode_wire(&trailing).is_err());

    // (c) Forged rejection reason code with recomputed top-level ledger_digest
    let mut bad_reason = ledger.clone();
    bad_reason.rounds[0].rejections[0].reason = RejectionReason::UnknownParticipant;
    bad_reason.ledger_digest = bad_reason.compute_digest();
    assert!(verify_ledger_sovereign(&bad_reason).is_err());

    // (d) Deleted rejection receipt
    let mut deleted = ledger.clone();
    deleted.rounds[0].rejections.pop();
    deleted.ledger_digest = deleted.compute_digest();
    assert!(verify_ledger_sovereign(&deleted).is_err());

    // (e) Altered final_model_hash
    let mut bad_final = ledger.clone();
    bad_final.final_model_hash[0] ^= 0xff;
    bad_final.ledger_digest = bad_final.compute_digest();
    assert!(verify_ledger_sovereign(&bad_final).is_err());

    // (f) Forged auditor signature
    let mut bad_sig = ledger.clone();
    bad_sig.rounds[0].auditor_signature[5] ^= 0x10;
    bad_sig.ledger_digest = bad_sig.compute_digest();
    assert!(verify_ledger_sovereign(&bad_sig).is_err());
}

#[test]
fn deterministic_reproducibility_bit_for_bit() {
    let dataset = load_repo_dataset();
    let cfg = default_sim_config(
        "determinism_check",
        SimulationConfig::mixed_roles(24, 7),
        PartitionStrategy::StratifiedIid,
        None,
        DefenseConfig::balanced_default(),
    );

    let r1 = run_simulation(&dataset, &cfg, AggregationRegime::HanzoFedVerifyDso);
    let r2 = run_simulation(&dataset, &cfg, AggregationRegime::HanzoFedVerifyDso);

    assert_eq!(r1.ledger.final_model_hash, r2.ledger.final_model_hash);
    assert_eq!(r1.ledger.ledger_digest, r2.ledger.ledger_digest);
    assert_eq!(r1.final_test_metrics, r2.final_test_metrics);
    assert_eq!(r1.ledger.encode_wire(), r2.ledger.encode_wire());
}
