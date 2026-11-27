// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic multi-participant federated training simulation harness (`N >= 20`)
//! covering 4 matched aggregation regimes, 5 adversarial attack classes, and
//! HIP-1350 5-axis telemetry.

use crate::aggregator::{AggregationRegime, DefenseConfig, FederatedAggregator, RoundAuditRecord};
use crate::crypto::{Digest32, SplitMix64};
use crate::identity::{
    ParticipantIdentity, ParticipantSigner, RejectionReason, SignedUpdateEnvelope, UpdateKind,
};
use crate::ledger::{verify_ledger_sovereign, FederationAuditLedger};
use crate::model::{
    l2_norm, CorpusDataset, EvalMetrics, FederatedModel, PartitionStrategy, FEATURE_DIM, MODEL_DIM,
    NUM_CLASSES, TOKENS_PER_SAMPLE,
};
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

/// Reference compute price (`$0.05 / vCPU-hour`) aligned with HIP-1350 benchmarks.
pub const REFERENCE_USD_PER_VCPU_HOUR: f64 = 0.05;

/// Converts compute time in nanoseconds and useful tokens into `$/1M useful tokens`.
#[must_use]
pub fn ns_to_usd_per_1m_tokens(compute_ns: u64, useful_tokens: u64) -> f64 {
    if useful_tokens == 0 {
        return 0.0;
    }
    let hours = (compute_ns as f64) / 3.6e12;
    let usd = hours * REFERENCE_USD_PER_VCPU_HOUR;
    usd * (1_000_000.0 / (useful_tokens as f64))
}

/// Behavior profile assigned to each simulated federated participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParticipantRole {
    /// Honest node training with local SGD on its local shard and signing faithfully.
    Honest,
    /// Submits negated gradient (`-1.8×` honest delta) to drive gradient ascent.
    SignFlip,
    /// Submits `100×` norm-exploded update to destroy unclipped averages.
    ScalingExplosion,
    /// Coordinated Sybil cartel submitting nearly identical targeted backdoor deltas
    /// calibrated within honest norm bounds to defeat naive Multi-Krum.
    CollusionCartel,
    /// Submits a subtly misaligned poisoning delta within norm bounds while forging
    /// `claimed_loss_delta` to claim large improvement.
    FakeClaimedImprovement,
    /// Cycles across corrupted Ed25519 signature, stale `base_model_hash`, replayed `nonce`,
    /// and post-signing delta tampering.
    TamperedOrReplay,
    /// Diverse independent adversarial poisoning (used in `>= 50%` super-majority stress test
    /// where adversaries do not share identical clones, breaking `f < n/2` statistical medians).
    SuperMajorityDiversePoison,
}

impl ParticipantRole {
    /// Returns `true` iff the participant is honest.
    #[must_use]
    pub fn is_honest(self) -> bool {
        matches!(self, Self::Honest)
    }

    /// Stable snake_case label for telemetry.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Honest => "honest",
            Self::SignFlip => "sign_flip",
            Self::ScalingExplosion => "scaling_100x",
            Self::CollusionCartel => "collusion_cartel",
            Self::FakeClaimedImprovement => "fake_claimed_improvement",
            Self::TamperedOrReplay => "tampered_or_replay",
            Self::SuperMajorityDiversePoison => "supermajority_diverse_poison",
        }
    }
}

/// Configuration for a complete multi-round federated simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Scenario identifier.
    pub scenario_name: String,
    /// Deterministic master seed (default `1350`).
    pub seed: u64,
    /// Total simulated participants (acceptance requires `>= 20`; default `24`).
    pub num_participants: usize,
    /// Number of federated communication rounds.
    pub rounds: usize,
    /// Local SGD learning rate.
    pub local_lr: f64,
    /// L2 regularization weight.
    pub l2_reg: f64,
    /// Local epochs per round.
    pub local_epochs: usize,
    /// Corpus partitioning mode (IID vs Extreme Non-IID).
    pub partition: PartitionStrategy,
    /// If `Some(rank)`, honest updates are transmitted as factorized LoRA adapters of that rank.
    pub lora_rank: Option<u32>,
    /// Explicit per-participant behavior roles (length must equal `num_participants`).
    pub roles: Vec<ParticipantRole>,
    /// Defense configuration used when running [`AggregationRegime::HanzoFedVerifyDso`].
    pub defense_config: DefenseConfig,
}

impl SimulationConfig {
    /// Constructs a mixed-Byzantine role vector of length `num_participants` with `byzantine_count` adversaries.
    #[must_use]
    pub fn mixed_roles(num_participants: usize, byzantine_count: usize) -> Vec<ParticipantRole> {
        let mut roles = vec![ParticipantRole::Honest; num_participants];
        let b = byzantine_count.min(num_participants);
        if b == 0 {
            return roles;
        }
        // Assign attacks deterministically at the tail of the participant list so honest shards stay aligned.
        let start = num_participants - b;
        if b >= 7 {
            // Ensure a >= 3-node Sybil collusion cartel plus all other 4 attack classes.
            let cartel_len = if b >= 10 { 4 } else { 3 };
            let mut idx = start;
            for _ in 0..cartel_len {
                roles[idx] = ParticipantRole::CollusionCartel;
                idx += 1;
            }
            let cycle = [
                ParticipantRole::SignFlip,
                ParticipantRole::ScalingExplosion,
                ParticipantRole::FakeClaimedImprovement,
                ParticipantRole::TamperedOrReplay,
            ];
            let mut c = 0;
            while idx < num_participants {
                roles[idx] = cycle[c % cycle.len()];
                idx += 1;
                c += 1;
            }
        } else {
            let cycle = [
                ParticipantRole::SignFlip,
                ParticipantRole::ScalingExplosion,
                ParticipantRole::FakeClaimedImprovement,
                ParticipantRole::TamperedOrReplay,
                ParticipantRole::CollusionCartel,
            ];
            for (k, slot) in roles[start..].iter_mut().enumerate() {
                *slot = cycle[k % cycle.len()];
            }
        }
        roles
    }

    /// Constructs a role vector with `byzantine_count` participants all executing `attack`.
    #[must_use]
    pub fn pure_attack_roles(
        num_participants: usize,
        byzantine_count: usize,
        attack: ParticipantRole,
    ) -> Vec<ParticipantRole> {
        let mut roles = vec![ParticipantRole::Honest; num_participants];
        let b = byzantine_count.min(num_participants);
        let start = num_participants - b;
        for slot in &mut roles[start..] {
            *slot = attack;
        }
        roles
    }
}

/// Complete measured report from running one aggregation regime on a [`SimulationConfig`].
#[derive(Debug, Clone)]
pub struct SimulationReport {
    /// Scenario name.
    pub scenario_name: String,
    /// Regime executed.
    pub regime: AggregationRegime,
    /// Defense profile label.
    pub defense_profile: String,
    /// Total simulated participants.
    pub participants_total: usize,
    /// Number of honest participants.
    pub honest_participants: usize,
    /// Number of Byzantine participants.
    pub byzantine_participants: usize,
    /// Number of rounds executed.
    pub rounds: usize,
    /// Number of rounds where an update passed safety checks and committed.
    pub rounds_committed: usize,
    /// Initial held-out test metrics before round 0.
    pub initial_test_metrics: EvalMetrics,
    /// Final held-out test metrics after the last round.
    pub final_test_metrics: EvalMetrics,
    /// Total malicious update submissions across all rounds.
    pub malicious_submissions_total: usize,
    /// Malicious updates accepted into aggregation across all rounds.
    pub malicious_accepted_total: usize,
    /// Malicious admission rate (`malicious_accepted_total / malicious_submissions_total`, `0.0` if 0 adversaries).
    pub attack_success_rate: f64,
    /// Total honest update submissions across all rounds.
    pub honest_submissions_total: usize,
    /// Honest updates rejected across all rounds.
    pub honest_rejected_total: usize,
    /// Honest false-rejection rate (`honest_rejected_total / honest_submissions_total`).
    pub honest_rejection_rate: f64,
    /// Breakdown of rejections by [`RejectionReason`].
    pub rejection_breakdown: BTreeMap<RejectionReason, usize>,
    /// Total local training compute time across all participants & rounds (ns).
    pub client_train_ns_total: u64,
    /// Total aggregator verification & aggregation time across all rounds (ns).
    pub verify_aggregate_ns_total: u64,
    /// Mean aggregator verification & aggregation latency per round (ms).
    pub mean_verify_ms_per_round: f64,
    /// Useful tokens contributed by accepted honest updates.
    pub useful_honest_tokens: u64,
    /// Total tokens accepted (honest + any admitted malicious).
    pub total_accepted_tokens: u64,
    /// Axis 1: Total compute cost (`client + verifier`) in `$/1M useful tokens`.
    pub cost_usd_per_1m_useful_tokens: f64,
    /// Axis 1: Verification compute share (`verify_ns / (client_ns + verify_ns)`).
    pub verification_compute_ratio: f64,
    /// Axis 2: Effective useful throughput (`useful_honest_tokens / total_secs`).
    pub useful_tokens_per_sec: f64,
    /// Axis 4: Mandatory Hanzo control-plane calls (always `0`).
    pub hanzo_control_plane_calls: u64,
    /// Axis 5: Binary wire size (`FVER` v1) of the exported sovereign ledger in bytes.
    pub ledger_wire_bytes: usize,
    /// Axis 5: Pretty JSON size of the exported sovereign ledger in bytes.
    pub ledger_json_bytes: usize,
    /// Axis 5: Whether independent offline replay of the decoded wire ledger passed.
    pub sovereign_replay_verified: bool,
    /// Exported sovereign audit ledger.
    pub ledger: FederationAuditLedger,
}

/// Runs a complete deterministic federated training simulation for `regime` on `dataset`.
#[must_use]
pub fn run_simulation(
    dataset: &CorpusDataset,
    cfg: &SimulationConfig,
    regime: AggregationRegime,
) -> SimulationReport {
    assert_eq!(
        cfg.roles.len(),
        cfg.num_participants,
        "roles length must equal num_participants"
    );

    let shards = dataset.partition_for_participants(cfg.num_participants, cfg.partition);
    let mut signers: Vec<ParticipantSigner> = Vec::with_capacity(cfg.num_participants);
    let mut identities: Vec<ParticipantIdentity> = Vec::with_capacity(cfg.num_participants);
    let mut honest_ids: HashSet<String> = HashSet::new();

    for i in 0..cfg.num_participants {
        let id = format!("node-{i:02}");
        let signer = ParticipantSigner::from_seed_label(id.clone(), cfg.seed ^ (i as u64));
        if cfg.roles[i].is_honest() {
            honest_ids.insert(id);
        }
        identities.push(signer.identity.clone());
        signers.push(signer);
    }

    let auditor = ParticipantSigner::from_seed_label("auditor-local-00", cfg.seed ^ 0xA0D1_7000);
    let mut aggregator = FederatedAggregator::new(regime, cfg.defense_config, auditor, &identities);

    let mut model = FederatedModel::init(cfg.seed);
    let initial_model_hash = model.model_hash();
    let initial_test_metrics = model.evaluate(&dataset.held_out_test);

    let mut nonces = vec![0u64; cfg.num_participants];
    let mut audit_rounds: Vec<RoundAuditRecord> = Vec::with_capacity(cfg.rounds);
    let mut rejection_breakdown: BTreeMap<RejectionReason, usize> = BTreeMap::new();

    let mut client_train_ns_total = 0u64;
    let mut verify_aggregate_ns_total = 0u64;
    let mut malicious_submissions_total = 0usize;
    let mut malicious_accepted_total = 0usize;
    let mut honest_submissions_total = 0usize;
    let mut honest_rejected_total = 0usize;
    let mut useful_honest_tokens = 0u64;
    let mut total_accepted_tokens = 0u64;
    let mut rounds_committed = 0usize;

    let mut stale_hash_memory = initial_model_hash;
    let mut rng = SplitMix64::new(cfg.seed ^ 0x51A0_0073);

    for round in 0..cfg.rounds {
        let round_id = round as u64;
        let base_model_hash = model.model_hash();
        let mut envelopes: Vec<SignedUpdateEnvelope> = Vec::with_capacity(cfg.num_participants);

        // Compute one reference honest delta norm per round so collusion/fake-claim attacks
        // are realistically calibrated to slip past naive magnitude checks.
        let ref_honest_delta = if let Some(h_idx) = cfg.roles.iter().position(|r| r.is_honest()) {
            let (d, _, _) =
                model.local_train_delta(&shards[h_idx], cfg.local_lr, cfg.l2_reg, cfg.local_epochs);
            d
        } else {
            vec![0.01; MODEL_DIM]
        };
        let ref_norm = l2_norm(&ref_honest_delta).max(0.05);
        let cartel_vector = build_cartel_backdoor_vector(&ref_honest_delta, ref_norm);

        let t_clients = Instant::now();
        for p in 0..cfg.num_participants {
            let role = cfg.roles[p];
            let shard = &shards[p];
            let sample_count = shard.len() as u64;
            let tokens_processed = sample_count * TOKENS_PER_SAMPLE;
            nonces[p] += 1;

            let env = match role {
                ParticipantRole::Honest => {
                    honest_submissions_total += 1;
                    let (dense_delta, loss_before, loss_after) =
                        model.local_train_delta(shard, cfg.local_lr, cfg.l2_reg, cfg.local_epochs);
                    let (kind, payload) = if let Some(rank) = cfg.lora_rank {
                        (
                            UpdateKind::LoraAdapter {
                                rank,
                                rows: NUM_CLASSES as u32,
                                cols: FEATURE_DIM as u32,
                            },
                            FederatedModel::compress_to_lora_adapter(&dense_delta, rank as usize),
                        )
                    } else {
                        (UpdateKind::GradientDelta, dense_delta)
                    };
                    signers[p].sign_update(
                        round_id,
                        base_model_hash,
                        kind,
                        payload,
                        loss_before,
                        loss_after,
                        sample_count,
                        tokens_processed,
                        nonces[p],
                    )
                }
                ParticipantRole::SignFlip => {
                    malicious_submissions_total += 1;
                    let (dense_delta, loss_before, _) =
                        model.local_train_delta(shard, cfg.local_lr, cfg.l2_reg, cfg.local_epochs);
                    let flipped: Vec<f64> = dense_delta.iter().map(|&v| -1.8 * v).collect();
                    signers[p].sign_update(
                        round_id,
                        base_model_hash,
                        UpdateKind::GradientDelta,
                        flipped,
                        loss_before,
                        (loss_before - 0.25).max(0.01),
                        sample_count,
                        tokens_processed,
                        nonces[p],
                    )
                }
                ParticipantRole::ScalingExplosion => {
                    malicious_submissions_total += 1;
                    let (dense_delta, loss_before, _) =
                        model.local_train_delta(shard, cfg.local_lr, cfg.l2_reg, cfg.local_epochs);
                    let exploded: Vec<f64> = dense_delta.iter().map(|&v| -100.0 * v).collect();
                    signers[p].sign_update(
                        round_id,
                        base_model_hash,
                        UpdateKind::GradientDelta,
                        exploded,
                        loss_before,
                        (loss_before - 0.40).max(0.01),
                        sample_count,
                        tokens_processed,
                        nonces[p],
                    )
                }
                ParticipantRole::CollusionCartel => {
                    malicious_submissions_total += 1;
                    // All cartel members submit nearly identical targeted backdoor updates
                    // inside honest norm bounds (cos >= 0.999) to hijack naive Multi-Krum.
                    let mut poisoned = cartel_vector.clone();
                    for v in &mut poisoned {
                        *v += rng.next_normal() * 1e-4;
                    }
                    signers[p].sign_update(
                        round_id,
                        base_model_hash,
                        UpdateKind::GradientDelta,
                        poisoned,
                        2.10,
                        1.75,
                        sample_count,
                        tokens_processed,
                        nonces[p],
                    )
                }
                ParticipantRole::FakeClaimedImprovement => {
                    malicious_submissions_total += 1;
                    // Calibrated to stay inside norm bounds and maintain mild positive cosine
                    // (~0.25) with honest direction while degrading held-out validation loss
                    // and forging a massive claimed improvement (+0.48).
                    let poisoned =
                        build_fake_claim_poison_vector(&ref_honest_delta, ref_norm, &mut rng);
                    signers[p].sign_update(
                        round_id,
                        base_model_hash,
                        UpdateKind::GradientDelta,
                        poisoned,
                        2.30,
                        1.82,
                        sample_count,
                        tokens_processed,
                        nonces[p],
                    )
                }
                ParticipantRole::TamperedOrReplay => {
                    malicious_submissions_total += 1;
                    let (dense_delta, loss_before, _) =
                        model.local_train_delta(shard, cfg.local_lr, cfg.l2_reg, cfg.local_epochs);
                    // Build a statistically plausible update near honest norm so pure statistical
                    // aggregators (without Ed25519/nonce/hash verification) admit it.
                    let stealth_poison: Vec<f64> = dense_delta
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            let bias = if i < FEATURE_DIM { 0.06 } else { -0.01 };
                            0.45 * v + bias
                        })
                        .collect();
                    match round % 4 {
                        0 => {
                            // Corrupted Ed25519 signature bit-flip
                            let mut e = signers[p].sign_update(
                                round_id,
                                base_model_hash,
                                UpdateKind::GradientDelta,
                                stealth_poison,
                                loss_before,
                                (loss_before - 0.15).max(0.01),
                                sample_count,
                                tokens_processed,
                                nonces[p],
                            );
                            e.signature[0] ^= 0x40;
                            e
                        }
                        1 => {
                            // Stale base_model_hash from earlier round
                            let stale = if stale_hash_memory == base_model_hash {
                                let mut h = base_model_hash;
                                h[0] ^= 0xff;
                                h
                            } else {
                                stale_hash_memory
                            };
                            signers[p].sign_update(
                                round_id,
                                stale,
                                UpdateKind::GradientDelta,
                                stealth_poison,
                                loss_before,
                                (loss_before - 0.15).max(0.01),
                                sample_count,
                                tokens_processed,
                                nonces[p],
                            )
                        }
                        2 => {
                            // Replayed/invalid genesis nonce (nonce = 0)
                            signers[p].sign_update(
                                round_id,
                                base_model_hash,
                                UpdateKind::GradientDelta,
                                stealth_poison,
                                loss_before,
                                (loss_before - 0.15).max(0.01),
                                sample_count,
                                tokens_processed,
                                0,
                            )
                        }
                        _ => {
                            // Post-signing payload mutation (delta_hash mismatch)
                            let mut e = signers[p].sign_update(
                                round_id,
                                base_model_hash,
                                UpdateKind::GradientDelta,
                                stealth_poison,
                                loss_before,
                                (loss_before - 0.15).max(0.01),
                                sample_count,
                                tokens_processed,
                                nonces[p],
                            );
                            e.payload[0] += 0.15;
                            e
                        }
                    }
                }
                ParticipantRole::SuperMajorityDiversePoison => {
                    malicious_submissions_total += 1;
                    // Independent diverse gradient-ascent + random-orthogonal poisoning so the
                    // >= 50% adversaries are NOT clones (bypassing clone clustering) yet still
                    // corrupt the coordinate-wise median!
                    let poisoned =
                        build_diverse_supermajority_poison(&ref_honest_delta, ref_norm, &mut rng);
                    signers[p].sign_update(
                        round_id,
                        base_model_hash,
                        UpdateKind::GradientDelta,
                        poisoned,
                        2.15,
                        1.85,
                        sample_count,
                        tokens_processed,
                        nonces[p],
                    )
                }
            };
            envelopes.push(env);
        }
        client_train_ns_total =
            client_train_ns_total.saturating_add(t_clients.elapsed().as_nanos() as u64);

        let outcome = aggregator.process_round(
            round_id,
            &mut model,
            &envelopes,
            &dataset.independent_val,
            Some(&honest_ids),
        );
        verify_aggregate_ns_total =
            verify_aggregate_ns_total.saturating_add(outcome.verify_aggregate_ns);
        total_accepted_tokens = total_accepted_tokens.saturating_add(outcome.accepted_tokens);
        if outcome.audit_record.update_committed {
            rounds_committed += 1;
        }

        let accepted_ids: HashSet<&str> = outcome
            .audit_record
            .accepted
            .iter()
            .map(|a| a.participant_id.as_str())
            .collect();

        for (p, role) in cfg.roles.iter().enumerate() {
            let pid = &signers[p].identity.participant_id;
            let was_accepted = accepted_ids.contains(pid.as_str());
            if role.is_honest() {
                if was_accepted {
                    useful_honest_tokens = useful_honest_tokens
                        .saturating_add((shards[p].len() as u64) * TOKENS_PER_SAMPLE);
                } else {
                    honest_rejected_total += 1;
                }
            } else if was_accepted {
                malicious_accepted_total += 1;
            }
        }

        for (&reason, &count) in &outcome.rejection_breakdown {
            *rejection_breakdown.entry(reason).or_insert(0) += count;
        }

        stale_hash_memory = base_model_hash;
        audit_rounds.push(outcome.audit_record);
    }

    let final_model_hash = model.model_hash();
    let final_test_metrics = model.evaluate(&dataset.held_out_test);

    let ledger = FederationAuditLedger::build(
        regime.as_str(),
        dataset.pins.clone(),
        identities,
        initial_model_hash,
        final_model_hash,
        audit_rounds,
    );
    let wire = ledger.encode_wire();
    let json = ledger.to_json_pretty();
    let sovereign_replay_verified = FederationAuditLedger::decode_wire(&wire)
        .and_then(|decoded| verify_ledger_sovereign(&decoded))
        .is_ok();

    let honest_participants = cfg.roles.iter().filter(|r| r.is_honest()).count();
    let byzantine_participants = cfg.num_participants - honest_participants;

    let attack_success_rate = if malicious_submissions_total == 0 {
        0.0
    } else {
        (malicious_accepted_total as f64) / (malicious_submissions_total as f64)
    };
    let honest_rejection_rate = if honest_submissions_total == 0 {
        0.0
    } else {
        (honest_rejected_total as f64) / (honest_submissions_total as f64)
    };

    let total_ns = client_train_ns_total.saturating_add(verify_aggregate_ns_total);
    let cost_usd_per_1m_useful_tokens = ns_to_usd_per_1m_tokens(total_ns, useful_honest_tokens);
    let verification_compute_ratio = if total_ns == 0 {
        0.0
    } else {
        (verify_aggregate_ns_total as f64) / (total_ns as f64)
    };
    let useful_tokens_per_sec = if total_ns == 0 {
        0.0
    } else {
        (useful_honest_tokens as f64) / ((total_ns as f64) / 1e9)
    };
    let mean_verify_ms_per_round = if cfg.rounds == 0 {
        0.0
    } else {
        ((verify_aggregate_ns_total as f64) / 1e6) / (cfg.rounds as f64)
    };

    SimulationReport {
        scenario_name: cfg.scenario_name.clone(),
        regime,
        defense_profile: cfg.defense_config.profile_name.to_string(),
        participants_total: cfg.num_participants,
        honest_participants,
        byzantine_participants,
        rounds: cfg.rounds,
        rounds_committed,
        initial_test_metrics,
        final_test_metrics,
        malicious_submissions_total,
        malicious_accepted_total,
        attack_success_rate,
        honest_submissions_total,
        honest_rejected_total,
        honest_rejection_rate,
        rejection_breakdown,
        client_train_ns_total,
        verify_aggregate_ns_total,
        mean_verify_ms_per_round,
        useful_honest_tokens,
        total_accepted_tokens,
        cost_usd_per_1m_useful_tokens,
        verification_compute_ratio,
        useful_tokens_per_sec,
        hanzo_control_plane_calls: 0,
        ledger_wire_bytes: wire.len(),
        ledger_json_bytes: json.len(),
        sovereign_replay_verified,
        ledger,
    }
}

/// Builds a targeted backdoor/poison delta vector for a coordinated Sybil cartel:
/// pushes class-0 logits sharply upward and suppresses non-0 class rows while
/// staying strictly inside typical honest L2 norm bounds (`0.95 * ref_norm`).
fn build_cartel_backdoor_vector(ref_honest: &[f64], ref_norm: f64) -> Vec<f64> {
    let mut v = vec![0.0f64; MODEL_DIM];
    for c in 0..NUM_CLASSES {
        for f in 0..FEATURE_DIM {
            let idx = c * FEATURE_DIM + f;
            v[idx] = if c == 0 {
                // Strongly boost backdoor target class 0 across all active context features.
                2.2
            } else {
                // Reverse honest updates on all other classes to degrade generalization.
                -1.2 * ref_honest[idx] - 0.18
            };
        }
    }
    let n = l2_norm(&v).max(1e-12);
    let target = ref_norm * 0.95;
    for x in &mut v {
        *x *= target / n;
    }
    v
}

/// Builds a deceptive update for `FakeClaimedImprovement` that stays inside honest norm bounds
/// and retains mild positive cosine (~0.10) to evade naive direction filters, yet strictly
/// degrades held-out validation loss (caught by Layer 5 independent validation gate!).
fn build_fake_claim_poison_vector(
    ref_honest: &[f64],
    ref_norm: f64,
    rng: &mut SplitMix64,
) -> Vec<f64> {
    let mut adv = vec![0.0f64; MODEL_DIM];
    for c in 0..NUM_CLASSES {
        for f in 0..FEATURE_DIM {
            let idx = c * FEATURE_DIM + f;
            // Push wrong class logits upward with per-node noise so it is not a clone.
            let bias_poison = if c == 0 { 2.4 } else { -0.35 };
            adv[idx] = bias_poison + rng.next_normal() * 0.30;
        }
    }
    // Remove projection onto ref_honest, then blend 0.10 * ref_unit + 0.995 * orth_unit
    let mut dot = 0.0;
    let rn2 = (ref_norm * ref_norm).max(1e-12);
    for i in 0..MODEL_DIM {
        dot += adv[i] * ref_honest[i];
    }
    for i in 0..MODEL_DIM {
        adv[i] -= (dot / rn2) * ref_honest[i];
    }
    let an = l2_norm(&adv).max(1e-12);
    let mut out = vec![0.0f64; MODEL_DIM];
    for i in 0..MODEL_DIM {
        let h_unit = ref_honest[i] / ref_norm;
        let a_unit = adv[i] / an;
        out[i] = ref_norm * 1.25 * (0.10 * h_unit + 0.995 * a_unit);
    }
    out
}

/// Builds diverse non-clone adversarial updates for the `>= 50%` super-majority negative regime.
fn build_diverse_supermajority_poison(
    ref_honest: &[f64],
    ref_norm: f64,
    rng: &mut SplitMix64,
) -> Vec<f64> {
    let mut v = vec![0.0f64; MODEL_DIM];
    for i in 0..MODEL_DIM {
        v[i] =
            -1.2 * ref_honest[i] + rng.next_normal() * (ref_norm / (MODEL_DIM as f64).sqrt()) * 0.9;
    }
    let n = l2_norm(&v).max(1e-12);
    let target = ref_norm * (0.85 + 0.30 * rng.next_f64());
    for x in &mut v {
        *x *= target / n;
    }
    v
}

/// Helper to return a `Digest32` zero constant when testing stale hashes.
#[must_use]
pub const fn zero_digest() -> Digest32 {
    [0u8; 32]
}
