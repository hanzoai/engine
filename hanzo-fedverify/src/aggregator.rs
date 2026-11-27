// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Multi-layer Byzantine-robust federated aggregation engine and matched comparison regimes.
//!
//! Implements 4 matched aggregation regimes:
//! 1. [`AggregationRegime::NaiveFedAvgBaseline`] — standard unverified FedAvg.
//! 2. [`AggregationRegime::TrimmedMeanKrum`] — pure statistical Multi-Krum + Trimmed Mean.
//! 3. [`AggregationRegime::HanzoFedVerifyDso`] — full 5-layer defense (Ed25519 identity &
//!    anti-replay, norm bounds, Sybil collusion cluster detection, Multi-Krum/cosine filtering,
//!    independent held-out validation gating, and signed `RejectionReceipt` attribution).
//! 4. [`AggregationRegime::OracleHonestOnlyCeiling`] — ground-truth honest-only quality ceiling.

use crate::crypto::{ct_eq_bytes, merkle_root, Digest32, PublicKeyBytes, Sha256};
use crate::identity::{
    AcceptedRecord, ParticipantIdentity, ParticipantSigner, RejectionReason, RejectionReceipt,
    SignedUpdateEnvelope, PROTOCOL_VERSION,
};
use crate::model::{
    cosine_similarity, l2_distance_sq, l2_norm, FederatedModel, TokenSample, MODEL_DIM,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

/// Matched federated aggregation regimes evaluated in HIP-1350 benchmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregationRegime {
    /// Unverified sample-weighted FedAvg baseline (collapses under Byzantine updates).
    NaiveFedAvgBaseline,
    /// Statistical Multi-Krum + coordinate-wise Trimmed Mean (no identity/collusion/val gate).
    TrimmedMeanKrum,
    /// Full verifiable pipeline with Ed25519 identity, collusion filter, independent validation gate,
    /// and signed `RejectionReceipt` attribution.
    HanzoFedVerifyDso,
    /// Omniscient honest-only aggregation ceiling for quality comparison.
    OracleHonestOnlyCeiling,
}

impl AggregationRegime {
    /// Stable snake_case identifier for JSON and markdown tables.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NaiveFedAvgBaseline => "naive_fedavg_baseline",
            Self::TrimmedMeanKrum => "trimmed_mean_krum",
            Self::HanzoFedVerifyDso => "hanzo_fedverify_dso",
            Self::OracleHonestOnlyCeiling => "oracle_honest_only_ceiling",
        }
    }
}

/// Configurable thresholds for [`AggregationRegime::HanzoFedVerifyDso`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DefenseConfig {
    /// Profile label for reporting.
    pub profile_name: &'static str,
    /// Hard upper ceiling on materialized delta L2 norm.
    pub hard_max_l2_norm: f64,
    /// Adaptive norm ceiling relative to the round's median L2 norm.
    pub adaptive_norm_multiplier: f64,
    /// Pairwise cosine similarity threshold for flagging candidate Sybil clones.
    pub collusion_cosine_threshold: f64,
    /// Pairwise relative L2 gap `||u_i - u_j|| / max(||u_i||, ||u_j||)` threshold for Sybil clones.
    pub collusion_rel_l2_threshold: f64,
    /// Minimum connected component size required to classify a clone set as a collusion cluster.
    pub min_collusion_cluster_size: usize,
    /// If a clone cluster's centroid held-out validation gain is below this floor, reject the cluster.
    pub collusion_val_improvement_floor: f64,
    /// Minimum cosine similarity to the robust coordinate-wise median reference direction.
    pub min_cosine_similarity: f64,
    /// Maximum Multi-Krum distance score multiple above median + `krum_mad_threshold * MAD`.
    pub krum_mad_threshold: f64,
    /// Probe step multiplier when evaluating a single candidate update on the held-out validation split.
    pub val_probe_scale: f64,
    /// Maximum tolerated held-out validation loss increase for an individual candidate update.
    pub max_val_loss_regression: f64,
    /// Minimum claimed loss improvement that triggers fake-claim contradiction checking.
    pub fake_claim_min_claimed_gain: f64,
    /// If `claimed_loss_delta >= fake_claim_min_claimed_gain` and actual held-out gain is below this, reject.
    pub fake_claim_max_actual_gain: f64,
    /// Fraction trimmed from each tail in coordinate-wise trimmed aggregation (when `>= 4` accepted).
    pub coord_trim_fraction: f64,
    /// Maximum tolerated held-out validation loss increase for the final aggregated round update.
    pub post_agg_max_regression: f64,
    /// Number of held-out validation samples evaluated per candidate (caps verification compute cleanly).
    pub val_eval_batch_size: usize,
}

impl Default for DefenseConfig {
    fn default() -> Self {
        Self::balanced_default()
    }
}

impl DefenseConfig {
    /// Balanced production default for IID and moderately heterogeneous federations.
    #[must_use]
    pub const fn balanced_default() -> Self {
        Self {
            profile_name: "balanced_default",
            hard_max_l2_norm: 6.0,
            adaptive_norm_multiplier: 3.0,
            collusion_cosine_threshold: 0.992,
            collusion_rel_l2_threshold: 0.08,
            min_collusion_cluster_size: 3,
            collusion_val_improvement_floor: 0.002,
            min_cosine_similarity: 0.0,
            krum_mad_threshold: 6.5,
            val_probe_scale: 0.25,
            max_val_loss_regression: 0.008,
            fake_claim_min_claimed_gain: 0.20,
            fake_claim_max_actual_gain: 0.001,
            coord_trim_fraction: 0.10,
            post_agg_max_regression: 0.005,
            val_eval_batch_size: 512,
        }
    }

    /// Tighter statistical thresholds (maximizes early filtering; increases false rejections under Non-IID).
    #[must_use]
    pub const fn strict_defense() -> Self {
        Self {
            profile_name: "strict_defense",
            hard_max_l2_norm: 4.0,
            adaptive_norm_multiplier: 1.50,
            collusion_cosine_threshold: 0.985,
            collusion_rel_l2_threshold: 0.12,
            min_collusion_cluster_size: 3,
            collusion_val_improvement_floor: 0.005,
            min_cosine_similarity: 0.55,
            krum_mad_threshold: 2.5,
            val_probe_scale: 0.25,
            max_val_loss_regression: -0.001,
            fake_claim_min_claimed_gain: 0.15,
            fake_claim_max_actual_gain: 0.003,
            coord_trim_fraction: 0.15,
            post_agg_max_regression: 0.002,
            val_eval_batch_size: 512,
        }
    }

    /// Relaxed statistical thresholds designed for extreme Non-IID honest shards (relies more heavily on validation gate).
    #[must_use]
    pub const fn non_iid_relaxed() -> Self {
        Self {
            profile_name: "non_iid_relaxed",
            hard_max_l2_norm: 10.0,
            adaptive_norm_multiplier: 5.0,
            collusion_cosine_threshold: 0.996,
            collusion_rel_l2_threshold: 0.05,
            min_collusion_cluster_size: 3,
            collusion_val_improvement_floor: 0.0005,
            min_cosine_similarity: -0.30,
            krum_mad_threshold: 12.0,
            val_probe_scale: 0.15,
            max_val_loss_regression: 0.025,
            fake_claim_min_claimed_gain: 0.30,
            fake_claim_max_actual_gain: -0.015,
            coord_trim_fraction: 0.05,
            post_agg_max_regression: 0.010,
            val_eval_batch_size: 512,
        }
    }
}

/// Signed cryptographic record emitted for every federated round.
#[derive(Debug, Clone, PartialEq)]
pub struct RoundAuditRecord {
    /// Zero-based round index.
    pub round_id: u64,
    /// Global model hash before aggregation.
    pub base_model_hash: Digest32,
    /// Global model hash after aggregation.
    pub new_model_hash: Digest32,
    /// Independent held-out validation loss before aggregation.
    pub base_val_loss: f64,
    /// Independent held-out validation loss after aggregation.
    pub new_val_loss: f64,
    /// Whether the aggregated update passed the post-aggregation zero-regression check and was applied.
    pub update_committed: bool,
    /// Accepted participant records in canonical order.
    pub accepted: Vec<AcceptedRecord>,
    /// Signed rejection attribution receipts in canonical order.
    pub rejections: Vec<RejectionReceipt>,
    /// Binary Merkle root over `accepted` leaf digests.
    pub accepted_merkle_root: Digest32,
    /// Binary Merkle root over `rejections` `receipt_digest`s.
    pub rejection_merkle_root: Digest32,
    /// Canonical SHA-256 digest of the round audit record.
    pub round_digest: Digest32,
    /// Auditor Ed25519 public key that signed this round record.
    pub auditor_pubkey: PublicKeyBytes,
    /// Ed25519 signature over `round_digest`.
    pub auditor_signature: [u8; 64],
}

impl RoundAuditRecord {
    /// Computes the canonical digest of the round audit record.
    #[must_use]
    pub fn compute_digest(&self) -> Digest32 {
        let mut h = Sha256::new();
        h.update(b"hanzo.fedverify.round.v1");
        h.update(&PROTOCOL_VERSION.to_le_bytes());
        h.update(&self.round_id.to_le_bytes());
        h.update(&self.base_model_hash);
        h.update(&self.new_model_hash);
        h.update_f64(self.base_val_loss);
        h.update_f64(self.new_val_loss);
        h.update(&[u8::from(self.update_committed)]);
        h.update(&(self.accepted.len() as u64).to_le_bytes());
        h.update(&(self.rejections.len() as u64).to_le_bytes());
        h.update(&self.accepted_merkle_root);
        h.update(&self.rejection_merkle_root);
        h.update(&self.auditor_pubkey);
        h.finalize()
    }
}

/// Output of executing a single aggregation round.
#[derive(Debug, Clone)]
pub struct RoundOutcome {
    /// Aggregated delta applied to the global model (zeros if round rolled back / no survivors).
    pub applied_delta: Vec<f64>,
    /// Signed cryptographic audit record for this round.
    pub audit_record: RoundAuditRecord,
    /// Wall-clock verification and aggregation time in nanoseconds.
    pub verify_aggregate_ns: u64,
    /// Number of accepted tokens processed by admitted updates in this round.
    pub accepted_tokens: u64,
    /// Rejection counts broken down by [`RejectionReason`].
    pub rejection_breakdown: BTreeMap<RejectionReason, usize>,
}

/// Stateful decentralized verifier and aggregator node.
///
/// Requires zero external control-plane calls: holds only the registered participant
/// public keys, anti-replay nonce watermarks, and its own local Ed25519 auditor keypair.
pub struct FederatedAggregator {
    regime: AggregationRegime,
    config: DefenseConfig,
    auditor: ParticipantSigner,
    registered_keys: HashMap<String, PublicKeyBytes>,
    last_nonce: HashMap<String, u64>,
}

struct MaterializedCandidate<'a> {
    envelope: &'a SignedUpdateEnvelope,
    delta: Vec<f64>,
    l2_norm: f64,
    cosine: f64,
    val_loss_delta: f64,
    collusion_cluster_size: u32,
}

impl FederatedAggregator {
    /// Creates a new local aggregator/auditor for `regime`.
    #[must_use]
    pub fn new(
        regime: AggregationRegime,
        config: DefenseConfig,
        auditor: ParticipantSigner,
        participants: &[ParticipantIdentity],
    ) -> Self {
        let mut registered_keys = HashMap::with_capacity(participants.len());
        for p in participants {
            registered_keys.insert(p.participant_id.clone(), p.public_key);
        }
        Self {
            regime,
            config,
            auditor,
            registered_keys,
            last_nonce: HashMap::new(),
        }
    }

    /// Returns the auditor's Ed25519 public key.
    #[must_use]
    pub fn auditor_pubkey(&self) -> PublicKeyBytes {
        self.auditor.identity.public_key
    }

    /// Runs one complete verification and aggregation round and mutates `model` in place.
    ///
    /// `oracle_honest_ids` is used exclusively by [`AggregationRegime::OracleHonestOnlyCeiling`]
    /// and completely ignored by all realistic regimes.
    pub fn process_round(
        &mut self,
        round_id: u64,
        model: &mut FederatedModel,
        submissions: &[SignedUpdateEnvelope],
        independent_val: &[TokenSample],
        oracle_honest_ids: Option<&HashSet<String>>,
    ) -> RoundOutcome {
        let t0 = Instant::now();
        let base_model_hash = model.model_hash();
        let val_slice = if independent_val.len() > self.config.val_eval_batch_size {
            &independent_val[..self.config.val_eval_batch_size]
        } else {
            independent_val
        };

        let outcome = match self.regime {
            AggregationRegime::NaiveFedAvgBaseline => {
                self.run_naive_fedavg(round_id, model, base_model_hash, submissions, val_slice)
            }
            AggregationRegime::TrimmedMeanKrum => {
                self.run_trimmed_mean_krum(round_id, model, base_model_hash, submissions, val_slice)
            }
            AggregationRegime::HanzoFedVerifyDso => {
                self.run_hanzo_fedverify(round_id, model, base_model_hash, submissions, val_slice)
            }
            AggregationRegime::OracleHonestOnlyCeiling => self.run_oracle_honest_only(
                round_id,
                model,
                base_model_hash,
                submissions,
                val_slice,
                oracle_honest_ids,
            ),
        };

        let elapsed_ns = t0.elapsed().as_nanos() as u64;
        RoundOutcome {
            verify_aggregate_ns: elapsed_ns,
            ..outcome
        }
    }

    fn run_naive_fedavg(
        &self,
        round_id: u64,
        model: &mut FederatedModel,
        base_model_hash: Digest32,
        submissions: &[SignedUpdateEnvelope],
        val_slice: &[TokenSample],
    ) -> RoundOutcome {
        let mut weighted_sum = vec![0.0f64; MODEL_DIM];
        let mut total_weight = 0.0f64;
        let mut accepted = Vec::new();
        let mut accepted_tokens = 0u64;

        for env in submissions {
            if let Some(delta) = env
                .header
                .update_kind
                .materialize_dense(&env.payload, MODEL_DIM)
            {
                let w = (env.header.sample_count.max(1)) as f64;
                for (acc, &d) in weighted_sum.iter_mut().zip(delta.iter()) {
                    *acc += w * d;
                }
                total_weight += w;
                accepted_tokens = accepted_tokens.saturating_add(env.header.tokens_processed);
                accepted.push(AcceptedRecord {
                    participant_id: env.header.participant_id.clone(),
                    participant_pubkey: env.header.participant_pubkey,
                    envelope_digest: env.envelope_digest,
                    delta_hash: env.header.delta_hash,
                    sample_count: env.header.sample_count,
                    tokens_processed: env.header.tokens_processed,
                    observed_l2_norm: l2_norm(&delta),
                    observed_cosine: 0.0,
                    observed_val_loss_delta: 0.0,
                    aggregation_weight: w,
                });
            }
        }

        let applied_delta = if total_weight > 0.0 {
            let inv = 1.0 / total_weight;
            for rec in &mut accepted {
                rec.aggregation_weight *= inv;
            }
            for v in &mut weighted_sum {
                *v *= inv;
            }
            model.apply_delta(&weighted_sum);
            weighted_sum
        } else {
            vec![0.0; MODEL_DIM]
        };

        let _ = val_slice;
        let new_model_hash = model.model_hash();
        let audit_record = self.build_audit_record(
            round_id,
            base_model_hash,
            new_model_hash,
            0.0,
            0.0,
            total_weight > 0.0,
            accepted,
            Vec::new(),
        );
        RoundOutcome {
            applied_delta,
            audit_record,
            verify_aggregate_ns: 0,
            accepted_tokens,
            rejection_breakdown: BTreeMap::new(),
        }
    }

    fn run_trimmed_mean_krum(
        &self,
        round_id: u64,
        model: &mut FederatedModel,
        base_model_hash: Digest32,
        submissions: &[SignedUpdateEnvelope],
        _val_slice: &[TokenSample],
    ) -> RoundOutcome {
        let mut cands: Vec<MaterializedCandidate<'_>> = Vec::new();
        let mut rejections = Vec::new();
        let mut rejection_breakdown = BTreeMap::new();

        for env in submissions {
            match env
                .header
                .update_kind
                .materialize_dense(&env.payload, MODEL_DIM)
            {
                Some(delta) => {
                    let norm = l2_norm(&delta);
                    cands.push(MaterializedCandidate {
                        envelope: env,
                        delta,
                        l2_norm: norm,
                        cosine: 0.0,
                        val_loss_delta: 0.0,
                        collusion_cluster_size: 1,
                    });
                }
                None => {
                    *rejection_breakdown
                        .entry(RejectionReason::NonFiniteOrMalformed)
                        .or_insert(0) += 1;
                    rejections.push(RejectionReceipt::issue(
                        &self.auditor,
                        env,
                        round_id,
                        RejectionReason::NonFiniteOrMalformed,
                        0.0,
                        0.0,
                        0.0,
                        1,
                    ));
                }
            }
        }

        let n = cands.len();
        if n == 0 {
            let audit_record = self.build_audit_record(
                round_id,
                base_model_hash,
                base_model_hash,
                0.0,
                0.0,
                false,
                Vec::new(),
                rejections,
            );
            return RoundOutcome {
                applied_delta: vec![0.0; MODEL_DIM],
                audit_record,
                verify_aggregate_ns: 0,
                accepted_tokens: 0,
                rejection_breakdown,
            };
        }

        // Standard Multi-Krum assuming up to f = floor((n - 3) / 3) adversaries.
        let f_est = (n.saturating_sub(3)) / 3;
        let neighbors = n.saturating_sub(f_est + 2).max(1);
        let m_keep = n.saturating_sub(f_est).max(1);

        let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            let mut dists = Vec::with_capacity(n.saturating_sub(1));
            for j in 0..n {
                if i != j {
                    dists.push(l2_distance_sq(&cands[i].delta, &cands[j].delta));
                }
            }
            dists.sort_by(|a, b| a.total_cmp(b));
            let score: f64 = dists.iter().take(neighbors).sum();
            scores.push((i, score));
        }
        scores.sort_by(|a, b| a.1.total_cmp(&b.1));

        let kept_set: HashSet<usize> = scores.iter().take(m_keep).map(|&(idx, _)| idx).collect();
        let mut kept_deltas: Vec<&[f64]> = Vec::new();
        let mut accepted = Vec::new();
        let mut accepted_tokens = 0u64;

        for (idx, c) in cands.iter().enumerate() {
            if kept_set.contains(&idx) {
                kept_deltas.push(&c.delta);
                accepted_tokens =
                    accepted_tokens.saturating_add(c.envelope.header.tokens_processed);
                accepted.push(AcceptedRecord {
                    participant_id: c.envelope.header.participant_id.clone(),
                    participant_pubkey: c.envelope.header.participant_pubkey,
                    envelope_digest: c.envelope.envelope_digest,
                    delta_hash: c.envelope.header.delta_hash,
                    sample_count: c.envelope.header.sample_count,
                    tokens_processed: c.envelope.header.tokens_processed,
                    observed_l2_norm: c.l2_norm,
                    observed_cosine: 0.0,
                    observed_val_loss_delta: 0.0,
                    aggregation_weight: 1.0 / (m_keep as f64),
                });
            } else {
                *rejection_breakdown
                    .entry(RejectionReason::CoordinateOutlier)
                    .or_insert(0) += 1;
                rejections.push(RejectionReceipt::issue(
                    &self.auditor,
                    c.envelope,
                    round_id,
                    RejectionReason::CoordinateOutlier,
                    c.l2_norm,
                    0.0,
                    0.0,
                    1,
                ));
            }
        }

        let applied_delta = coordinate_trimmed_mean(&kept_deltas, 0.20);
        model.apply_delta(&applied_delta);
        let new_model_hash = model.model_hash();
        let audit_record = self.build_audit_record(
            round_id,
            base_model_hash,
            new_model_hash,
            0.0,
            0.0,
            true,
            accepted,
            rejections,
        );
        RoundOutcome {
            applied_delta,
            audit_record,
            verify_aggregate_ns: 0,
            accepted_tokens,
            rejection_breakdown,
        }
    }

    fn run_oracle_honest_only(
        &self,
        round_id: u64,
        model: &mut FederatedModel,
        base_model_hash: Digest32,
        submissions: &[SignedUpdateEnvelope],
        val_slice: &[TokenSample],
        oracle_honest_ids: Option<&HashSet<String>>,
    ) -> RoundOutcome {
        let base_val_loss = model.evaluate(val_slice).cross_entropy_loss;
        let mut sum = vec![0.0f64; MODEL_DIM];
        let mut total_w = 0.0f64;
        let mut accepted = Vec::new();
        let mut accepted_tokens = 0u64;

        for env in submissions {
            let is_honest =
                oracle_honest_ids.is_none_or(|set| set.contains(&env.header.participant_id));
            if !is_honest {
                continue;
            }
            if let Some(delta) = env
                .header
                .update_kind
                .materialize_dense(&env.payload, MODEL_DIM)
            {
                let w = env.header.sample_count.max(1) as f64;
                for (acc, &d) in sum.iter_mut().zip(delta.iter()) {
                    *acc += w * d;
                }
                total_w += w;
                accepted_tokens = accepted_tokens.saturating_add(env.header.tokens_processed);
                accepted.push(AcceptedRecord {
                    participant_id: env.header.participant_id.clone(),
                    participant_pubkey: env.header.participant_pubkey,
                    envelope_digest: env.envelope_digest,
                    delta_hash: env.header.delta_hash,
                    sample_count: env.header.sample_count,
                    tokens_processed: env.header.tokens_processed,
                    observed_l2_norm: l2_norm(&delta),
                    observed_cosine: 1.0,
                    observed_val_loss_delta: 0.0,
                    aggregation_weight: w,
                });
            }
        }

        let applied_delta = if total_w > 0.0 {
            let inv = 1.0 / total_w;
            for rec in &mut accepted {
                rec.aggregation_weight *= inv;
            }
            for v in &mut sum {
                *v *= inv;
            }
            model.apply_delta(&sum);
            sum
        } else {
            vec![0.0; MODEL_DIM]
        };

        let new_model_hash = model.model_hash();
        let new_val_loss = model.evaluate(val_slice).cross_entropy_loss;
        let audit_record = self.build_audit_record(
            round_id,
            base_model_hash,
            new_model_hash,
            base_val_loss,
            new_val_loss,
            total_w > 0.0,
            accepted,
            Vec::new(),
        );
        RoundOutcome {
            applied_delta,
            audit_record,
            verify_aggregate_ns: 0,
            accepted_tokens,
            rejection_breakdown: BTreeMap::new(),
        }
    }

    fn run_hanzo_fedverify(
        &mut self,
        round_id: u64,
        model: &mut FederatedModel,
        base_model_hash: Digest32,
        submissions: &[SignedUpdateEnvelope],
        val_slice: &[TokenSample],
    ) -> RoundOutcome {
        let base_val_loss = model.evaluate(val_slice).cross_entropy_loss;
        let mut rejections: Vec<RejectionReceipt> = Vec::new();
        let mut rejection_breakdown: BTreeMap<RejectionReason, usize> = BTreeMap::new();
        let mut seen_this_round: HashSet<String> = HashSet::with_capacity(submissions.len());

        let mut push_reject = |auditor: &ParticipantSigner,
                               env: &SignedUpdateEnvelope,
                               reason: RejectionReason,
                               norm: f64,
                               cos: f64,
                               val_delta: f64,
                               cluster_sz: u32| {
            *rejection_breakdown.entry(reason).or_insert(0) += 1;
            rejections.push(RejectionReceipt::issue(
                auditor, env, round_id, reason, norm, cos, val_delta, cluster_sz,
            ));
        };

        // --------------------------------------------------------------------
        // Layer 1: Cryptographic Identity, Signature, Nonce Replay & Freshness
        // --------------------------------------------------------------------
        let mut layer1: Vec<MaterializedCandidate<'_>> = Vec::with_capacity(submissions.len());
        for env in submissions {
            let Some(&expected_pk) = self.registered_keys.get(&env.header.participant_id) else {
                push_reject(
                    &self.auditor,
                    env,
                    RejectionReason::UnknownParticipant,
                    0.0,
                    0.0,
                    0.0,
                    1,
                );
                continue;
            };
            if !ct_eq_bytes(&expected_pk, &env.header.participant_pubkey) {
                push_reject(
                    &self.auditor,
                    env,
                    RejectionReason::UnknownParticipant,
                    0.0,
                    0.0,
                    0.0,
                    1,
                );
                continue;
            }
            if env.verify_cryptography().is_err() {
                push_reject(
                    &self.auditor,
                    env,
                    RejectionReason::InvalidSignature,
                    0.0,
                    0.0,
                    0.0,
                    1,
                );
                continue;
            }
            if !seen_this_round.insert(env.header.participant_id.clone()) {
                push_reject(
                    &self.auditor,
                    env,
                    RejectionReason::ReplayNonce,
                    0.0,
                    0.0,
                    0.0,
                    1,
                );
                continue;
            }
            if env.header.nonce == 0
                || self
                    .last_nonce
                    .get(&env.header.participant_id)
                    .is_some_and(|&prev_nonce| env.header.nonce <= prev_nonce)
            {
                push_reject(
                    &self.auditor,
                    env,
                    RejectionReason::ReplayNonce,
                    0.0,
                    0.0,
                    0.0,
                    1,
                );
                continue;
            }
            if env.header.round_id != round_id
                || !ct_eq_bytes(&env.header.base_model_hash, &base_model_hash)
            {
                push_reject(
                    &self.auditor,
                    env,
                    RejectionReason::StaleBaseModel,
                    0.0,
                    0.0,
                    0.0,
                    1,
                );
                continue;
            }
            if !env.header.claimed_loss_before.is_finite()
                || !env.header.claimed_loss_after.is_finite()
                || !env.header.claimed_loss_delta.is_finite()
            {
                push_reject(
                    &self.auditor,
                    env,
                    RejectionReason::NonFiniteOrMalformed,
                    0.0,
                    0.0,
                    0.0,
                    1,
                );
                continue;
            }
            let Some(delta) = env
                .header
                .update_kind
                .materialize_dense(&env.payload, MODEL_DIM)
            else {
                push_reject(
                    &self.auditor,
                    env,
                    RejectionReason::NonFiniteOrMalformed,
                    0.0,
                    0.0,
                    0.0,
                    1,
                );
                continue;
            };

            // Advance anti-replay watermark only for cryptographically authentic, fresh envelopes.
            self.last_nonce
                .insert(env.header.participant_id.clone(), env.header.nonce);

            let norm = l2_norm(&delta);
            layer1.push(MaterializedCandidate {
                envelope: env,
                delta,
                l2_norm: norm,
                cosine: 0.0,
                val_loss_delta: 0.0,
                collusion_cluster_size: 1,
            });
        }

        // --------------------------------------------------------------------
        // Layer 2: Norm Bound & Adaptive Magnitude Gate
        // --------------------------------------------------------------------
        let mut norms: Vec<f64> = layer1.iter().map(|c| c.l2_norm).collect();
        let median_norm = median_of(&mut norms);
        let adaptive_cap = if median_norm > 1e-6 {
            (median_norm * self.config.adaptive_norm_multiplier).min(self.config.hard_max_l2_norm)
        } else {
            self.config.hard_max_l2_norm
        };

        let mut layer2 = Vec::with_capacity(layer1.len());
        for c in layer1 {
            if !c.l2_norm.is_finite() || c.l2_norm > adaptive_cap {
                push_reject(
                    &self.auditor,
                    c.envelope,
                    RejectionReason::NormBoundViolation,
                    c.l2_norm,
                    0.0,
                    0.0,
                    1,
                );
            } else {
                layer2.push(c);
            }
        }

        // --------------------------------------------------------------------
        // Layer 3: Sybil / Collusion Cluster Detection + Validation Cross-Check
        // --------------------------------------------------------------------
        let n2 = layer2.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n2];
        for i in 0..n2 {
            for j in (i + 1)..n2 {
                let cos = cosine_similarity(&layer2[i].delta, &layer2[j].delta);
                let max_n = layer2[i].l2_norm.max(layer2[j].l2_norm).max(1e-12);
                let rel_gap = l2_distance_sq(&layer2[i].delta, &layer2[j].delta).sqrt() / max_n;
                if cos >= self.config.collusion_cosine_threshold
                    && rel_gap <= self.config.collusion_rel_l2_threshold
                {
                    adj[i].push(j);
                    adj[j].push(i);
                }
            }
        }

        let mut visited = vec![false; n2];
        let mut flagged_collusion = vec![false; n2];
        let mut cluster_sizes = vec![1u32; n2];

        for start in 0..n2 {
            if visited[start] {
                continue;
            }
            let mut stack = vec![start];
            let mut comp = Vec::new();
            visited[start] = true;
            while let Some(u) = stack.pop() {
                comp.push(u);
                for &v in &adj[u] {
                    if !visited[v] {
                        visited[v] = true;
                        stack.push(v);
                    }
                }
            }
            let sz = comp.len() as u32;
            for &idx in &comp {
                cluster_sizes[idx] = sz;
            }
            if comp.len() >= self.config.min_collusion_cluster_size {
                // Cross-verify the clone cluster centroid on the independent held-out validation
                // split against both an absolute improvement floor and the non-cloned peers'
                // median validation gain, preventing false positives on honest correlated nodes.
                let mut centroid = vec![0.0f64; MODEL_DIM];
                for &idx in &comp {
                    for (acc, &d) in centroid.iter_mut().zip(layer2[idx].delta.iter()) {
                        *acc += d;
                    }
                }
                let inv = 1.0 / (comp.len() as f64);
                for v in &mut centroid {
                    *v *= inv;
                }
                let centroid_val_loss = model
                    .evaluate_with_delta(&centroid, self.config.val_probe_scale, val_slice)
                    .cross_entropy_loss;
                let centroid_gain = base_val_loss - centroid_val_loss;

                let comp_set: HashSet<usize> = comp.iter().copied().collect();
                let non_cluster: Vec<&[f64]> = layer2
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| !comp_set.contains(idx))
                    .map(|(_, c)| c.delta.as_slice())
                    .collect();
                let peer_gain = if non_cluster.is_empty() {
                    0.0
                } else {
                    let peer_med = coordinate_median(&non_cluster);
                    let peer_loss = model
                        .evaluate_with_delta(&peer_med, self.config.val_probe_scale, val_slice)
                        .cross_entropy_loss;
                    base_val_loss - peer_loss
                };

                if centroid_gain < self.config.collusion_val_improvement_floor
                    || (peer_gain > 0.0 && centroid_gain < 0.5 * peer_gain)
                {
                    for &idx in &comp {
                        flagged_collusion[idx] = true;
                    }
                }
            }
        }

        let mut layer3 = Vec::with_capacity(n2);
        for (idx, mut c) in layer2.into_iter().enumerate() {
            c.collusion_cluster_size = cluster_sizes[idx];
            if flagged_collusion[idx] {
                let probe_loss = model
                    .evaluate_with_delta(&c.delta, self.config.val_probe_scale, val_slice)
                    .cross_entropy_loss;
                let val_gain = base_val_loss - probe_loss;
                push_reject(
                    &self.auditor,
                    c.envelope,
                    RejectionReason::CollusionClusterDetected,
                    c.l2_norm,
                    0.0,
                    val_gain,
                    c.collusion_cluster_size,
                );
            } else {
                layer3.push(c);
            }
        }

        // --------------------------------------------------------------------
        // Layer 4: Robust Direction (Cosine vs Coordinate Median) & Multi-Krum MAD
        // --------------------------------------------------------------------
        let mut layer4_dir = Vec::with_capacity(layer3.len());
        if !layer3.is_empty() {
            let ref_deltas: Vec<&[f64]> = layer3.iter().map(|c| c.delta.as_slice()).collect();
            let ref_median = coordinate_median(&ref_deltas);
            let ref_norm = l2_norm(&ref_median);

            for mut c in layer3 {
                c.cosine = if ref_norm > 1e-12 {
                    cosine_similarity(&c.delta, &ref_median)
                } else {
                    1.0
                };
                if c.cosine < self.config.min_cosine_similarity {
                    push_reject(
                        &self.auditor,
                        c.envelope,
                        RejectionReason::CosineDirectionAnomaly,
                        c.l2_norm,
                        c.cosine,
                        0.0,
                        c.collusion_cluster_size,
                    );
                } else {
                    layer4_dir.push(c);
                }
            }
        }

        // Multi-Krum MAD outlier filter on Layer-4 directional survivors
        let mut layer4 = Vec::with_capacity(layer4_dir.len());
        let n4 = layer4_dir.len();
        if n4 >= 5 {
            let k_neighbors = (n4 / 2).max(2);
            let mut krum_scores = vec![0.0f64; n4];
            for i in 0..n4 {
                let mut dists = Vec::with_capacity(n4 - 1);
                for j in 0..n4 {
                    if i != j {
                        dists.push(l2_distance_sq(&layer4_dir[i].delta, &layer4_dir[j].delta));
                    }
                }
                dists.sort_by(|a, b| a.total_cmp(b));
                krum_scores[i] = dists.iter().take(k_neighbors).sum();
            }
            let mut sorted_scores = krum_scores.clone();
            let med_score = median_of(&mut sorted_scores);
            let mut abs_devs: Vec<f64> =
                krum_scores.iter().map(|&s| (s - med_score).abs()).collect();
            let mad = median_of(&mut abs_devs).max(med_score * 0.15).max(1e-6);
            let max_score = med_score + self.config.krum_mad_threshold * mad;

            for (i, c) in layer4_dir.into_iter().enumerate() {
                if krum_scores[i] > max_score {
                    push_reject(
                        &self.auditor,
                        c.envelope,
                        RejectionReason::CoordinateOutlier,
                        c.l2_norm,
                        c.cosine,
                        0.0,
                        c.collusion_cluster_size,
                    );
                } else {
                    layer4.push(c);
                }
            }
        } else {
            layer4 = layer4_dir;
        }

        // --------------------------------------------------------------------
        // Layer 5: Independent Held-Out Validation Quality Gate & Claim Audit
        // --------------------------------------------------------------------
        let mut probed: Vec<(MaterializedCandidate<'_>, f64, bool)> =
            Vec::with_capacity(layer4.len());
        let mut finite_gains: Vec<f64> = Vec::with_capacity(layer4.len());
        for mut c in layer4 {
            let probe_loss = model
                .evaluate_with_delta(&c.delta, self.config.val_probe_scale, val_slice)
                .cross_entropy_loss;
            let is_finite = probe_loss.is_finite();
            let actual_gain = if is_finite {
                base_val_loss - probe_loss
            } else {
                f64::NEG_INFINITY
            };
            c.val_loss_delta = if is_finite { actual_gain } else { -50.0 };
            if is_finite {
                finite_gains.push(actual_gain);
            }
            probed.push((c, actual_gain, is_finite));
        }
        let median_actual_gain = median_of(&mut finite_gains);

        let mut survivors: Vec<MaterializedCandidate<'_>> = Vec::with_capacity(probed.len());
        for (c, actual_gain, is_finite) in probed {
            let claimed = c.envelope.header.claimed_loss_delta;
            let regresses = !is_finite || actual_gain < -self.config.max_val_loss_regression;
            let contradicts_claim = claimed >= self.config.fake_claim_min_claimed_gain
                && (actual_gain < self.config.fake_claim_max_actual_gain
                    || (self.config.fake_claim_max_actual_gain >= 0.0
                        && actual_gain < 0.05 * claimed));
            let _ = median_actual_gain;

            if regresses || contradicts_claim {
                push_reject(
                    &self.auditor,
                    c.envelope,
                    RejectionReason::ValidationLossRegression,
                    c.l2_norm,
                    c.cosine,
                    c.val_loss_delta,
                    c.collusion_cluster_size,
                );
            } else {
                survivors.push(c);
            }
        }

        // Aggregate surviving updates using coordinate-wise trimmed mean + post-aggregation safety gate.
        let mut accepted = Vec::with_capacity(survivors.len());
        let mut accepted_tokens = 0u64;
        let (applied_delta, update_committed, new_val_loss) = if survivors.is_empty() {
            (vec![0.0; MODEL_DIM], false, base_val_loss)
        } else {
            let cand_deltas: Vec<&[f64]> = survivors.iter().map(|c| c.delta.as_slice()).collect();
            let trim_frac = if survivors.len() >= 4 {
                self.config.coord_trim_fraction
            } else {
                0.0
            };
            let candidate_agg = coordinate_trimmed_mean(&cand_deltas, trim_frac);
            let post_loss = model
                .evaluate_with_delta(&candidate_agg, 1.0, val_slice)
                .cross_entropy_loss;

            if post_loss.is_finite()
                && post_loss <= base_val_loss + self.config.post_agg_max_regression
            {
                let weight = 1.0 / (survivors.len() as f64);
                for c in &survivors {
                    accepted_tokens =
                        accepted_tokens.saturating_add(c.envelope.header.tokens_processed);
                    accepted.push(AcceptedRecord {
                        participant_id: c.envelope.header.participant_id.clone(),
                        participant_pubkey: c.envelope.header.participant_pubkey,
                        envelope_digest: c.envelope.envelope_digest,
                        delta_hash: c.envelope.header.delta_hash,
                        sample_count: c.envelope.header.sample_count,
                        tokens_processed: c.envelope.header.tokens_processed,
                        observed_l2_norm: c.l2_norm,
                        observed_cosine: c.cosine,
                        observed_val_loss_delta: c.val_loss_delta,
                        aggregation_weight: weight,
                    });
                }
                model.apply_delta(&candidate_agg);
                (candidate_agg, true, post_loss)
            } else {
                // Zero-regression safety fallback: reject remaining candidates and preserve global model.
                for c in &survivors {
                    push_reject(
                        &self.auditor,
                        c.envelope,
                        RejectionReason::ValidationLossRegression,
                        c.l2_norm,
                        c.cosine,
                        c.val_loss_delta,
                        c.collusion_cluster_size,
                    );
                }
                (vec![0.0; MODEL_DIM], false, base_val_loss)
            }
        };

        let new_model_hash = model.model_hash();
        let audit_record = self.build_audit_record(
            round_id,
            base_model_hash,
            new_model_hash,
            base_val_loss,
            new_val_loss,
            update_committed,
            accepted,
            rejections,
        );

        RoundOutcome {
            applied_delta,
            audit_record,
            verify_aggregate_ns: 0,
            accepted_tokens,
            rejection_breakdown,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_audit_record(
        &self,
        round_id: u64,
        base_model_hash: Digest32,
        new_model_hash: Digest32,
        base_val_loss: f64,
        new_val_loss: f64,
        update_committed: bool,
        accepted: Vec<AcceptedRecord>,
        rejections: Vec<RejectionReceipt>,
    ) -> RoundAuditRecord {
        let accepted_leaves: Vec<Digest32> = accepted.iter().map(AcceptedRecord::digest).collect();
        let rejection_leaves: Vec<Digest32> = rejections.iter().map(|r| r.receipt_digest).collect();
        let accepted_merkle_root = merkle_root(&accepted_leaves);
        let rejection_merkle_root = merkle_root(&rejection_leaves);

        let mut rec = RoundAuditRecord {
            round_id,
            base_model_hash,
            new_model_hash,
            base_val_loss,
            new_val_loss,
            update_committed,
            accepted,
            rejections,
            accepted_merkle_root,
            rejection_merkle_root,
            round_digest: [0u8; 32],
            auditor_pubkey: self.auditor.identity.public_key,
            auditor_signature: [0u8; 64],
        };
        rec.round_digest = rec.compute_digest();
        rec.auditor_signature = self.auditor.sign_bytes(&rec.round_digest);
        rec
    }
}

/// Computes the exact median of a non-empty mutable slice (`0.0` if empty).
pub fn median_of(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

/// Computes the coordinate-wise median across `deltas`.
#[must_use]
pub fn coordinate_median(deltas: &[&[f64]]) -> Vec<f64> {
    if deltas.is_empty() {
        return vec![0.0; MODEL_DIM];
    }
    let dim = deltas[0].len();
    let mut out = vec![0.0f64; dim];
    let mut col = Vec::with_capacity(deltas.len());
    for d in 0..dim {
        col.clear();
        for v in deltas {
            col.push(v[d]);
        }
        out[d] = median_of(&mut col);
    }
    out
}

/// Computes the coordinate-wise trimmed mean across `deltas`.
#[must_use]
pub fn coordinate_trimmed_mean(deltas: &[&[f64]], trim_fraction: f64) -> Vec<f64> {
    if deltas.is_empty() {
        return vec![0.0; MODEL_DIM];
    }
    let n = deltas.len();
    let dim = deltas[0].len();
    let trim = ((n as f64) * trim_fraction.clamp(0.0, 0.45)).floor() as usize;
    let trim = trim.min((n.saturating_sub(1)) / 2);
    let mut out = vec![0.0f64; dim];
    let mut col = Vec::with_capacity(n);

    for d in 0..dim {
        col.clear();
        for v in deltas {
            col.push(v[d]);
        }
        col.sort_by(|a, b| a.total_cmp(b));
        let slice = &col[trim..(n - trim)];
        let sum: f64 = slice.iter().sum();
        out[d] = sum / (slice.len().max(1) as f64);
    }
    out
}
