// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reproducible HIP-1350 benchmark runner (`fedverify-bench`) for `hanzoai/engine#73`.
//!
//! Executes matched 4-regime federated training sweeps (`N = 24` participants, pinned to
//! versioned `hanzoai/engine` corpus files), per-attack-class isolation benchmarks,
//! LoRA adapter verification, honest negative regimes (super-majority `>= 50%` Byzantine,
//! extreme Non-IID honest heterogeneity tradeoff, 0% attack verification overhead tax),
//! and sovereign audit ledger tamper tests. Emits deterministic JSON results.

use hanzo_fedverify::aggregator::{AggregationRegime, DefenseConfig};
use hanzo_fedverify::crypto::hex_encode;
use hanzo_fedverify::identity::RejectionReason;
use hanzo_fedverify::ledger::{escape_json, verify_ledger_sovereign, FederationAuditLedger};
use hanzo_fedverify::model::{CorpusDataset, PartitionStrategy};
use hanzo_fedverify::simulation::{
    run_simulation, ParticipantRole, SimulationConfig, SimulationReport,
    REFERENCE_USD_PER_VCPU_HOUR,
};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let mut seed: u64 = 1350;
    let mut num_participants: usize = 24;
    let mut rounds: usize = 12;
    let mut out_json: Option<PathBuf> = None;

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args
                    .get(i)
                    .expect("missing --seed value")
                    .parse()
                    .expect("invalid --seed u64");
            }
            "--participants" => {
                i += 1;
                num_participants = args
                    .get(i)
                    .expect("missing --participants value")
                    .parse()
                    .expect("invalid --participants usize");
            }
            "--rounds" => {
                i += 1;
                rounds = args
                    .get(i)
                    .expect("missing --rounds value")
                    .parse()
                    .expect("invalid --rounds usize");
            }
            "--out-json" => {
                i += 1;
                out_json = Some(PathBuf::from(
                    args.get(i).expect("missing --out-json value"),
                ));
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    assert!(
        num_participants >= 20,
        "Acceptance criterion requires at least 20 simulated participants"
    );

    let repo_root = CorpusDataset::find_repo_root();
    let dataset = CorpusDataset::load_from_repo_root(&repo_root, seed)
        .expect("failed to load pinned versioned repository corpus files");

    let regimes = [
        AggregationRegime::NaiveFedAvgBaseline,
        AggregationRegime::TrimmedMeanKrum,
        AggregationRegime::HanzoFedVerifyDso,
        AggregationRegime::OracleHonestOnlyCeiling,
    ];

    // 1. Main Mixed Byzantine Ratio Sweep (0%, 16.7%, 29.2%, 41.7%, 54.2% super-majority)
    let mixed_specs: Vec<(&str, Vec<ParticipantRole>)> = vec![
        (
            "byz_0pct_clean",
            SimulationConfig::mixed_roles(num_participants, 0),
        ),
        (
            "byz_17pct_mixed",
            SimulationConfig::mixed_roles(num_participants, 4),
        ),
        (
            "byz_29pct_mixed",
            SimulationConfig::mixed_roles(num_participants, 7),
        ),
        (
            "byz_42pct_mixed",
            SimulationConfig::mixed_roles(num_participants, 10),
        ),
        (
            "byz_54pct_supermajority",
            SimulationConfig::pure_attack_roles(
                num_participants,
                (num_participants / 2) + 1,
                ParticipantRole::SuperMajorityDiversePoison,
            ),
        ),
    ];

    let mut main_sweep: Vec<SimulationReport> = Vec::new();
    for (name, roles) in &mixed_specs {
        let cfg = base_config(
            name,
            seed,
            num_participants,
            rounds,
            PartitionStrategy::StratifiedIid,
            None,
            roles.clone(),
            DefenseConfig::balanced_default(),
        );
        for &regime in &regimes {
            main_sweep.push(run_simulation(&dataset, &cfg, regime));
        }
    }

    // 2. Per-Attack-Class Isolation Matrix (7/24 = 29.2% pure single attack class)
    let isolation_specs: Vec<(&str, ParticipantRole)> = vec![
        ("pure_sign_flip_29pct", ParticipantRole::SignFlip),
        ("pure_scaling_100x_29pct", ParticipantRole::ScalingExplosion),
        (
            "pure_collusion_cartel_29pct",
            ParticipantRole::CollusionCartel,
        ),
        (
            "pure_fake_claim_29pct",
            ParticipantRole::FakeClaimedImprovement,
        ),
        (
            "pure_tamper_replay_29pct",
            ParticipantRole::TamperedOrReplay,
        ),
    ];
    let mut attack_isolation: Vec<SimulationReport> = Vec::new();
    for (name, attack_role) in &isolation_specs {
        let cfg = base_config(
            name,
            seed,
            num_participants,
            rounds,
            PartitionStrategy::StratifiedIid,
            None,
            SimulationConfig::pure_attack_roles(num_participants, 7, *attack_role),
            DefenseConfig::balanced_default(),
        );
        for &regime in &regimes {
            attack_isolation.push(run_simulation(&dataset, &cfg, regime));
        }
    }

    // 3. LoRA Adapter Update Mode (rank 4: 176 params vs 384 dense = 2.18x wire reduction)
    let lora_cfg = base_config(
        "lora_r4_byz_29pct_mixed",
        seed,
        num_participants,
        rounds,
        PartitionStrategy::StratifiedIid,
        Some(4),
        SimulationConfig::mixed_roles(num_participants, 7),
        DefenseConfig::balanced_default(),
    );
    let mut lora_reports: Vec<SimulationReport> = Vec::new();
    for &regime in &regimes {
        lora_reports.push(run_simulation(&dataset, &lora_cfg, regime));
    }

    // 4. Negative Regime #2: Extreme Non-IID Honest Heterogeneity Tradeoff
    let defense_profiles = [
        DefenseConfig::strict_defense(),
        DefenseConfig::balanced_default(),
        DefenseConfig::non_iid_relaxed(),
    ];
    let mut non_iid_tradeoff: Vec<SimulationReport> = Vec::new();
    for &dcfg in &defense_profiles {
        // 0% attack Extreme Non-IID (measures pure honest false-rejection rate)
        let cfg_clean = base_config(
            "extreme_non_iid_0pct_attack",
            seed,
            num_participants,
            rounds,
            PartitionStrategy::ExtremeNonIid,
            None,
            SimulationConfig::mixed_roles(num_participants, 0),
            dcfg,
        );
        non_iid_tradeoff.push(run_simulation(
            &dataset,
            &cfg_clean,
            AggregationRegime::HanzoFedVerifyDso,
        ));

        // 29.2% mixed attack Extreme Non-IID (measures attack resistance vs honest rejection Pareto)
        let cfg_attack = base_config(
            "extreme_non_iid_29pct_attack",
            seed,
            num_participants,
            rounds,
            PartitionStrategy::ExtremeNonIid,
            None,
            SimulationConfig::mixed_roles(num_participants, 7),
            dcfg,
        );
        non_iid_tradeoff.push(run_simulation(
            &dataset,
            &cfg_attack,
            AggregationRegime::HanzoFedVerifyDso,
        ));
    }
    // Also include Naive & Oracle ceilings on Extreme Non-IID clean for reference
    let non_iid_oracle_cfg = base_config(
        "extreme_non_iid_0pct_attack",
        seed,
        num_participants,
        rounds,
        PartitionStrategy::ExtremeNonIid,
        None,
        SimulationConfig::mixed_roles(num_participants, 0),
        DefenseConfig::balanced_default(),
    );
    non_iid_tradeoff.push(run_simulation(
        &dataset,
        &non_iid_oracle_cfg,
        AggregationRegime::OracleHonestOnlyCeiling,
    ));

    // 5. Sovereign Ledger & Tamper Detection Audit (using the 29% mixed HanzoFedVerifyDso ledger)
    let sample_report = main_sweep
        .iter()
        .find(|r| {
            r.scenario_name == "byz_29pct_mixed" && r.regime == AggregationRegime::HanzoFedVerifyDso
        })
        .expect("byz_29pct_mixed hanzo_fedverify_dso report present");
    let security_audit = run_security_audit(&sample_report.ledger);

    // Emit concise console summary
    println!("=== hanzo-fedverify HIP-1350 Benchmark (seed={seed}, N={num_participants}, rounds={rounds}) ===");
    println!(
        "Corpus: {} tokens ({} train, {} independent_val, {} held_out_test)",
        dataset.total_tokens,
        dataset.train_pool.len(),
        dataset.independent_val.len(),
        dataset.held_out_test.len()
    );
    println!("\n--- Main Mixed Byzantine Sweep ---");
    for r in &main_sweep {
        println!(
            "{:<25} {:<26} | atk_ok={:6.2}% | hon_rej={:6.2}% | test_loss={:6.4} (ppl={:7.2}, acc={:5.2}%) | verify={:6.3}ms/r | $/1M={:.6}",
            r.scenario_name,
            r.regime.as_str(),
            r.attack_success_rate * 100.0,
            r.honest_rejection_rate * 100.0,
            r.final_test_metrics.cross_entropy_loss,
            r.final_test_metrics.perplexity,
            r.final_test_metrics.top1_accuracy * 100.0,
            r.mean_verify_ms_per_round,
            r.cost_usd_per_1m_useful_tokens,
        );
    }
    println!("\n--- Per-Attack-Class Isolation (29.2% pure attack) ---");
    for r in &attack_isolation {
        println!(
            "{:<28} {:<26} | atk_ok={:6.2}% | hon_rej={:6.2}% | test_loss={:6.4} (acc={:5.2}%, p0={:5.2}%)",
            r.scenario_name,
            r.regime.as_str(),
            r.attack_success_rate * 100.0,
            r.honest_rejection_rate * 100.0,
            r.final_test_metrics.cross_entropy_loss,
            r.final_test_metrics.top1_accuracy * 100.0,
            r.final_test_metrics.poison_class_rate * 100.0,
        );
    }
    println!("\n--- LoRA Rank-4 Mode (29.2% mixed attack) ---");
    for r in &lora_reports {
        println!(
            "{:<28} {:<26} | atk_ok={:6.2}% | hon_rej={:6.2}% | test_loss={:6.4} (ppl={:7.2}, acc={:5.2}%)",
            r.scenario_name,
            r.regime.as_str(),
            r.attack_success_rate * 100.0,
            r.honest_rejection_rate * 100.0,
            r.final_test_metrics.cross_entropy_loss,
            r.final_test_metrics.perplexity,
            r.final_test_metrics.top1_accuracy * 100.0,
        );
    }
    println!("\n--- Extreme Non-IID Honest Heterogeneity Tradeoff ---");
    for r in &non_iid_tradeoff {
        println!(
            "{:<30} {:<18} {:<26} | atk_ok={:6.2}% | hon_rej={:6.2}% | test_loss={:6.4} (acc={:5.2}%)",
            r.scenario_name,
            r.defense_profile,
            r.regime.as_str(),
            r.attack_success_rate * 100.0,
            r.honest_rejection_rate * 100.0,
            r.final_test_metrics.cross_entropy_loss,
            r.final_test_metrics.top1_accuracy * 100.0,
        );
    }
    println!(
        "\nSovereign Audit: replay_ok={}, tamper_checks={}/{}, wire={}B, json={}B, control_plane_calls=0",
        security_audit.sovereign_wire_replay_ok,
        security_audit.tamper_tests_passed,
        security_audit.tamper_tests_total,
        sample_report.ledger_wire_bytes,
        sample_report.ledger_json_bytes,
    );

    if let Some(path) = out_json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create output parent directory");
        }
        let json = render_results_json(
            seed,
            num_participants,
            rounds,
            &dataset,
            &main_sweep,
            &attack_isolation,
            &lora_reports,
            &non_iid_tradeoff,
            &security_audit,
        );
        fs::write(&path, json).expect("write benchmark results.json");
        println!("Wrote JSON artifact to {}", path.display());
    }
}

#[allow(clippy::too_many_arguments)]
fn base_config(
    name: &str,
    seed: u64,
    num_participants: usize,
    rounds: usize,
    partition: PartitionStrategy,
    lora_rank: Option<u32>,
    roles: Vec<ParticipantRole>,
    defense_config: DefenseConfig,
) -> SimulationConfig {
    SimulationConfig {
        scenario_name: name.to_string(),
        seed,
        num_participants,
        rounds,
        local_lr: 0.22,
        l2_reg: 1e-4,
        local_epochs: 2,
        partition,
        lora_rank,
        roles,
        defense_config,
    }
}

struct SecurityAuditResult {
    sovereign_wire_replay_ok: bool,
    verified_rounds: usize,
    verified_rejections: usize,
    tamper_tests_passed: usize,
    tamper_tests_total: usize,
    tamper_cases: Vec<(&'static str, bool)>,
    hanzo_control_plane_calls: u64,
}

fn run_security_audit(ledger: &FederationAuditLedger) -> SecurityAuditResult {
    let wire = ledger.encode_wire();
    let decoded = FederationAuditLedger::decode_wire(&wire).expect("valid wire decode");
    let summary = verify_ledger_sovereign(&decoded).expect("valid sovereign replay");

    let mut cases: Vec<(&'static str, bool)> = Vec::new();

    // 1. Wire bit-flip inside ledger payload
    let mut bad_wire = wire.clone();
    let mid = bad_wire.len() / 2;
    bad_wire[mid] ^= 0x01;
    let c1 = FederationAuditLedger::decode_wire(&bad_wire)
        .and_then(|l| verify_ledger_sovereign(&l))
        .is_err();
    cases.push(("wire_bit_flip_rejected", c1));

    // 2. Forged rejection reason code (even if top-level ledger_digest is recomputed!)
    let mut bad_reason = ledger.clone();
    if let Some(r0) = bad_reason.rounds.first_mut() {
        if let Some(rej) = r0.rejections.first_mut() {
            rej.reason = if rej.reason == RejectionReason::InvalidSignature {
                RejectionReason::NormBoundViolation
            } else {
                RejectionReason::InvalidSignature
            };
        }
    }
    bad_reason.ledger_digest = bad_reason.compute_digest();
    cases.push((
        "forged_rejection_reason_rejected",
        verify_ledger_sovereign(&bad_reason).is_err(),
    ));

    // 3. Deleted rejection receipt (attempt to hide Byzantine attribution)
    let mut deleted_rej = ledger.clone();
    if let Some(r0) = deleted_rej.rounds.first_mut() {
        if !r0.rejections.is_empty() {
            r0.rejections.pop();
        }
    }
    deleted_rej.ledger_digest = deleted_rej.compute_digest();
    cases.push((
        "deleted_rejection_receipt_rejected",
        verify_ledger_sovereign(&deleted_rej).is_err(),
    ));

    // 4. Altered final_model_hash (attempt to substitute a poisoned checkpoint)
    let mut bad_hash = ledger.clone();
    bad_hash.final_model_hash[0] ^= 0xff;
    bad_hash.ledger_digest = bad_hash.compute_digest();
    cases.push((
        "altered_final_model_hash_rejected",
        verify_ledger_sovereign(&bad_hash).is_err(),
    ));

    // 5. Forged auditor Ed25519 signature on round 0
    let mut bad_sig = ledger.clone();
    if let Some(r0) = bad_sig.rounds.first_mut() {
        r0.auditor_signature[0] ^= 0x20;
    }
    bad_sig.ledger_digest = bad_sig.compute_digest();
    cases.push((
        "forged_auditor_signature_rejected",
        verify_ledger_sovereign(&bad_sig).is_err(),
    ));

    let passed = cases.iter().filter(|(_, ok)| *ok).count();
    SecurityAuditResult {
        sovereign_wire_replay_ok: true,
        verified_rounds: summary.verified_rounds,
        verified_rejections: summary.verified_rejections,
        tamper_tests_passed: passed,
        tamper_tests_total: cases.len(),
        tamper_cases: cases,
        hanzo_control_plane_calls: summary.hanzo_control_plane_calls,
    }
}

#[allow(clippy::too_many_arguments)]
fn render_results_json(
    seed: u64,
    num_participants: usize,
    rounds: usize,
    dataset: &CorpusDataset,
    main_sweep: &[SimulationReport],
    attack_isolation: &[SimulationReport],
    lora_reports: &[SimulationReport],
    non_iid_tradeoff: &[SimulationReport],
    audit: &SecurityAuditResult,
) -> String {
    let mut s = String::with_capacity(32_768);
    s.push_str("{\n");
    s.push_str("  \"schema_version\": \"hip-1350.fedverify.results.v1\",\n");
    s.push_str("  \"issue\": \"hanzoai/engine#73\",\n");
    s.push_str("  \"crate\": \"hanzo-fedverify\",\n");
    s.push_str(&format!("  \"seed\": {seed},\n"));
    s.push_str(&format!("  \"participants_total\": {num_participants},\n"));
    s.push_str(&format!("  \"rounds\": {rounds},\n"));
    s.push_str(&format!(
        "  \"reference_usd_per_vcpu_hour\": {:.4},\n",
        REFERENCE_USD_PER_VCPU_HOUR
    ));
    s.push_str(&format!(
        "  \"corpus_total_tokens\": {},\n",
        dataset.total_tokens
    ));
    s.push_str(&format!(
        "  \"train_samples\": {},\n",
        dataset.train_pool.len()
    ));
    s.push_str(&format!(
        "  \"independent_val_samples\": {},\n",
        dataset.independent_val.len()
    ));
    s.push_str(&format!(
        "  \"held_out_test_samples\": {},\n",
        dataset.held_out_test.len()
    ));

    s.push_str("  \"pinned_corpus_files\": [\n");
    for (idx, pin) in dataset.pins.iter().enumerate() {
        let comma = if idx + 1 < dataset.pins.len() {
            ","
        } else {
            ""
        };
        s.push_str(&format!(
            "    {{\"path\": \"{}\", \"bytes\": {}, \"sha256\": \"{}\"}}{}\n",
            escape_json(&pin.relative_path),
            pin.byte_len,
            hex_encode(&pin.sha256_digest),
            comma
        ));
    }
    s.push_str("  ],\n");

    s.push_str("  \"main_mixed_byzantine_sweep\": [\n");
    write_reports_array(&mut s, main_sweep);
    s.push_str("  ],\n");

    s.push_str("  \"attack_class_isolation_29pct\": [\n");
    write_reports_array(&mut s, attack_isolation);
    s.push_str("  ],\n");

    s.push_str("  \"lora_rank4_verification\": [\n");
    write_reports_array(&mut s, lora_reports);
    s.push_str("  ],\n");

    s.push_str("  \"negative_regime_extreme_non_iid_tradeoff\": [\n");
    write_reports_array(&mut s, non_iid_tradeoff);
    s.push_str("  ],\n");

    s.push_str("  \"security_and_sovereignty_audit\": {\n");
    s.push_str(&format!(
        "    \"sovereign_wire_replay_ok\": {},\n",
        audit.sovereign_wire_replay_ok
    ));
    s.push_str(&format!(
        "    \"verified_rounds\": {},\n",
        audit.verified_rounds
    ));
    s.push_str(&format!(
        "    \"verified_rejection_receipts\": {},\n",
        audit.verified_rejections
    ));
    s.push_str(&format!(
        "    \"tamper_tests_passed\": {},\n",
        audit.tamper_tests_passed
    ));
    s.push_str(&format!(
        "    \"tamper_tests_total\": {},\n",
        audit.tamper_tests_total
    ));
    s.push_str(&format!(
        "    \"hanzo_control_plane_calls\": {},\n",
        audit.hanzo_control_plane_calls
    ));
    s.push_str("    \"tamper_cases\": {\n");
    for (idx, (label, ok)) in audit.tamper_cases.iter().enumerate() {
        let comma = if idx + 1 < audit.tamper_cases.len() {
            ","
        } else {
            ""
        };
        s.push_str(&format!("      \"{label}\": {ok}{comma}\n"));
    }
    s.push_str("    }\n");
    s.push_str("  }\n");
    s.push_str("}\n");
    s
}

fn write_reports_array(s: &mut String, reports: &[SimulationReport]) {
    for (idx, r) in reports.iter().enumerate() {
        let comma = if idx + 1 < reports.len() { "," } else { "" };
        s.push_str("    {\n");
        s.push_str(&format!(
            "      \"scenario\": \"{}\",\n",
            escape_json(&r.scenario_name)
        ));
        s.push_str(&format!("      \"regime\": \"{}\",\n", r.regime.as_str()));
        s.push_str(&format!(
            "      \"defense_profile\": \"{}\",\n",
            escape_json(&r.defense_profile)
        ));
        s.push_str(&format!(
            "      \"participants_total\": {},\n",
            r.participants_total
        ));
        s.push_str(&format!(
            "      \"honest_participants\": {},\n",
            r.honest_participants
        ));
        s.push_str(&format!(
            "      \"byzantine_participants\": {},\n",
            r.byzantine_participants
        ));
        s.push_str(&format!("      \"rounds\": {},\n", r.rounds));
        s.push_str(&format!(
            "      \"rounds_committed\": {},\n",
            r.rounds_committed
        ));
        s.push_str(&format!(
            "      \"initial_test_loss\": {:.6},\n",
            r.initial_test_metrics.cross_entropy_loss
        ));
        s.push_str(&format!(
            "      \"final_test_loss\": {:.6},\n",
            r.final_test_metrics.cross_entropy_loss
        ));
        s.push_str(&format!(
            "      \"final_test_perplexity\": {:.4},\n",
            r.final_test_metrics.perplexity
        ));
        s.push_str(&format!(
            "      \"final_test_top1_accuracy\": {:.6},\n",
            r.final_test_metrics.top1_accuracy
        ));
        s.push_str(&format!(
            "      \"final_poison_class_rate\": {:.6},\n",
            r.final_test_metrics.poison_class_rate
        ));
        s.push_str(&format!(
            "      \"malicious_submissions_total\": {},\n",
            r.malicious_submissions_total
        ));
        s.push_str(&format!(
            "      \"malicious_accepted_total\": {},\n",
            r.malicious_accepted_total
        ));
        s.push_str(&format!(
            "      \"attack_success_rate\": {:.6},\n",
            r.attack_success_rate
        ));
        s.push_str(&format!(
            "      \"honest_submissions_total\": {},\n",
            r.honest_submissions_total
        ));
        s.push_str(&format!(
            "      \"honest_rejected_total\": {},\n",
            r.honest_rejected_total
        ));
        s.push_str(&format!(
            "      \"honest_rejection_rate\": {:.6},\n",
            r.honest_rejection_rate
        ));
        s.push_str(&format!(
            "      \"client_train_ns_total\": {},\n",
            r.client_train_ns_total
        ));
        s.push_str(&format!(
            "      \"verify_aggregate_ns_total\": {},\n",
            r.verify_aggregate_ns_total
        ));
        s.push_str(&format!(
            "      \"mean_verify_ms_per_round\": {:.4},\n",
            r.mean_verify_ms_per_round
        ));
        s.push_str(&format!(
            "      \"useful_honest_tokens\": {},\n",
            r.useful_honest_tokens
        ));
        s.push_str(&format!(
            "      \"total_accepted_tokens\": {},\n",
            r.total_accepted_tokens
        ));
        s.push_str(&format!(
            "      \"cost_usd_per_1m_useful_tokens\": {:.6},\n",
            r.cost_usd_per_1m_useful_tokens
        ));
        s.push_str(&format!(
            "      \"verification_compute_ratio\": {:.6},\n",
            r.verification_compute_ratio
        ));
        s.push_str(&format!(
            "      \"useful_tokens_per_sec\": {:.2},\n",
            r.useful_tokens_per_sec
        ));
        s.push_str(&format!(
            "      \"hanzo_control_plane_calls\": {},\n",
            r.hanzo_control_plane_calls
        ));
        s.push_str(&format!(
            "      \"ledger_wire_bytes\": {},\n",
            r.ledger_wire_bytes
        ));
        s.push_str(&format!(
            "      \"ledger_json_bytes\": {},\n",
            r.ledger_json_bytes
        ));
        s.push_str(&format!(
            "      \"sovereign_replay_verified\": {},\n",
            r.sovereign_replay_verified
        ));
        s.push_str(&format!(
            "      \"final_model_hash\": \"{}\",\n",
            hex_encode(&r.ledger.final_model_hash)
        ));
        s.push_str(&format!(
            "      \"ledger_digest\": \"{}\",\n",
            hex_encode(&r.ledger.ledger_digest)
        ));
        s.push_str("      \"rejection_breakdown\": {");
        for (bi, (reason, count)) in r.rejection_breakdown.iter().enumerate() {
            let bcomma = if bi + 1 < r.rejection_breakdown.len() {
                ", "
            } else {
                ""
            };
            s.push_str(&format!("\"{}\": {count}{bcomma}", reason.as_str()));
        }
        s.push_str("}\n");
        s.push_str(&format!("    }}{comma}\n"));
    }
}
