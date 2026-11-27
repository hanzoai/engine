// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Interactive $O(\log_2 T)$ binary-search bisection fraud-proof dispute game,
//! bond slashing, per-turn timeout enforcement, anti-griefing escalation, and
//! permissionless multi-challenger tournament coordination.

use std::collections::{BTreeMap, BTreeSet};

use crate::crypto::{ct_eq_digest, Digest, Sha256, DOMAIN_CLAIM, DOMAIN_DISPUTE};
use crate::error::OpAiError;
use crate::profile::{ExecutionProfile, InputCommitment};
use crate::vm::{verify_single_step, ExecutionTrace, QuantizedModel, SingleStepVerdict, VmState};

/// Economic and liveness parameters governing optimistic settlement and dispute games.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameConfig {
    /// Minimum provider bond locked per submitted claim (micro-units).
    pub provider_bond: u64,
    /// Base counter-bond required for the first challenger on a claim (micro-units).
    pub challenger_base_bond: u64,
    /// Anti-griefing bond escalation multiplier per concurrent challenge slot in basis points
    /// (`10_000` = 1.0x flat, `15_000` = 1.5x per additional concurrent challenge).
    pub griefing_escalation_bps: u64,
    /// Maximum number of concurrent active challenges permitted against a single claim.
    pub max_concurrent_challenges_per_claim: usize,
    /// Challenge window duration after claim submission (seconds).
    pub challenge_window_secs: u64,
    /// Maximum response time allowed per bisection / single-step turn (seconds).
    pub turn_deadline_secs: u64,
    /// Share of a slashed provider bond awarded to the winning challenger (`1..=9_999` bps).
    /// Strictly `< 10_000` so the burned remainder makes self-challenging strictly negative-EV.
    pub challenger_reward_bps: u64,
    /// Share of a slashed malicious challenger bond paid to the honest defender (`1..=9_999` bps).
    pub provider_compensation_bps: u64,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            provider_bond: 1_000_000,
            challenger_base_bond: 500_000,
            griefing_escalation_bps: 15_000,
            max_concurrent_challenges_per_claim: 8,
            challenge_window_secs: 600,
            turn_deadline_secs: 60,
            challenger_reward_bps: 5_000,
            provider_compensation_bps: 5_000,
        }
    }
}

impl GameConfig {
    /// Validate economic and liveness invariants of the game configuration.
    pub fn validate(&self) -> Result<(), OpAiError> {
        if self.provider_bond == 0 {
            return Err(OpAiError::InvalidClaim(
                "provider_bond must be > 0".to_string(),
            ));
        }
        if self.challenger_base_bond == 0 {
            return Err(OpAiError::InvalidClaim(
                "challenger_base_bond must be > 0".to_string(),
            ));
        }
        if self.griefing_escalation_bps < 10_000 {
            return Err(OpAiError::InvalidClaim(
                "griefing_escalation_bps must be >= 10_000".to_string(),
            ));
        }
        if self.max_concurrent_challenges_per_claim == 0 {
            return Err(OpAiError::InvalidClaim(
                "max_concurrent_challenges_per_claim must be >= 1".to_string(),
            ));
        }
        if self.challenge_window_secs == 0 || self.turn_deadline_secs == 0 {
            return Err(OpAiError::InvalidClaim(
                "challenge_window_secs and turn_deadline_secs must be >= 1".to_string(),
            ));
        }
        if self.challenger_reward_bps == 0 || self.challenger_reward_bps >= 10_000 {
            return Err(OpAiError::InvalidClaim(
                "challenger_reward_bps must be in 1..=9_999 (burn > 0 required against self-challenge)"
                    .to_string(),
            ));
        }
        if self.provider_compensation_bps == 0 || self.provider_compensation_bps >= 10_000 {
            return Err(OpAiError::InvalidClaim(
                "provider_compensation_bps must be in 1..=9_999".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute the escalated counter-bond required for challenge index `slot_index` (`0`-indexed).
    pub fn required_challenger_bond(&self, slot_index: usize) -> u64 {
        let mut bond = u128::from(self.challenger_base_bond);
        let mult = u128::from(self.griefing_escalation_bps);
        for _ in 0..slot_index {
            // Ceiling division ensures strictly increasing bond when mult > 10_000.
            bond = bond.saturating_mul(mult).div_ceil(10_000);
            if bond >= u128::from(u64::MAX) {
                return u64::MAX;
            }
        }
        bond as u64
    }
}

/// A bonded optimistic AI execution claim posted by an inference provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    /// Canonical SHA-256 identifier binding all claim fields.
    pub claim_id: Digest,
    /// Provider identifier (address / key ID).
    pub provider_id: String,
    /// Monotonic or unique provider nonce distinguishing identical prompts.
    pub nonce: u64,
    /// SHA-256 digest of the bound [`ExecutionProfile`].
    pub profile_digest: Digest,
    /// SHA-256 digest of the bound [`QuantizedModel`].
    pub model_digest: Digest,
    /// SHA-256 digest of the bound [`InputCommitment`].
    pub input_hash: Digest,
    /// Total atomic VM steps `T` (`>= 1`).
    pub total_steps: u64,
    /// Committed `StateRoot` at step `0`.
    pub start_root: Digest,
    /// Committed `StateRoot` at step `T`.
    pub end_root: Digest,
    /// Binary SHA-256 Merkle root over `state_roots[0..=T]`.
    pub trace_merkle_root: Digest,
    /// Claimed autoregressive output token IDs.
    pub output_tokens: Vec<u32>,
    /// Provider bond locked for this claim (micro-units).
    pub bond: u64,
    /// Submission timestamp in seconds.
    pub submitted_at_secs: u64,
}

impl Claim {
    /// Build a canonical [`Claim`] from a completed [`ExecutionTrace`].
    pub fn from_trace(
        provider_id: impl Into<String>,
        nonce: u64,
        trace: &ExecutionTrace,
        bond: u64,
        submitted_at_secs: u64,
    ) -> Result<Self, OpAiError> {
        let provider_id = provider_id.into();
        if provider_id.is_empty() {
            return Err(OpAiError::InvalidClaim(
                "provider_id must be non-empty".to_string(),
            ));
        }
        if trace.total_steps == 0 {
            return Err(OpAiError::InvalidClaim(
                "total_steps must be >= 1".to_string(),
            ));
        }
        let mut claim = Self {
            claim_id: [0u8; 32],
            provider_id,
            nonce,
            profile_digest: trace.profile_digest,
            model_digest: trace.model_digest,
            input_hash: trace.input_hash,
            total_steps: trace.total_steps,
            start_root: trace.start_root(),
            end_root: trace.end_root(),
            trace_merkle_root: trace.trace_merkle_root,
            output_tokens: trace.output_tokens.clone(),
            bond,
            submitted_at_secs,
        };
        claim.claim_id = claim.compute_id();
        Ok(claim)
    }

    /// Recompute the canonical domain-separated `claim_id` from all bound fields.
    pub fn compute_id(&self) -> Digest {
        let mut h = Sha256::with_domain(DOMAIN_CLAIM);
        h.update_str(&self.provider_id);
        h.update_u64_le(self.nonce);
        h.update(&self.profile_digest);
        h.update(&self.model_digest);
        h.update(&self.input_hash);
        h.update_u64_le(self.total_steps);
        h.update(&self.start_root);
        h.update(&self.end_root);
        h.update(&self.trace_merkle_root);
        h.update_u64_le(self.output_tokens.len() as u64);
        for &tok in &self.output_tokens {
            h.update_u32_le(tok);
        }
        h.update_u64_le(self.bond);
        h.update_u64_le(self.submitted_at_secs);
        h.finalize()
    }

    /// Verify that `self.claim_id` matches the canonical recomputed digest in constant time.
    pub fn verify_id(&self) -> bool {
        ct_eq_digest(&self.claim_id, &self.compute_id())
    }
}

/// Party participating in a two-player interactive bisection dispute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Party {
    /// Original claim provider defending the committed `end_root`.
    Defender = 0,
    /// Permissionless verifier disputing the claim.
    Challenger = 1,
}

impl Party {
    /// Decode party from wire tag.
    pub fn from_tag(tag: u8) -> Result<Self, OpAiError> {
        match tag {
            0 => Ok(Self::Defender),
            1 => Ok(Self::Challenger),
            other => Err(OpAiError::WireFormat(format!("unknown Party tag: {other}"))),
        }
    }

    /// Return wire tag.
    pub fn to_tag(self) -> u8 {
        self as u8
    }
}

/// Binary-search branch chosen by the challenger after observing `mid_root`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BranchChoice {
    /// Challenger disagrees with defender's `mid_root` -> narrow to `[lo, mid]`.
    Left = 0,
    /// Challenger agrees with defender's `mid_root` -> narrow to `[mid, hi]`.
    Right = 1,
}

impl BranchChoice {
    /// Decode branch choice from wire tag.
    pub fn from_tag(tag: u8) -> Result<Self, OpAiError> {
        match tag {
            0 => Ok(Self::Left),
            1 => Ok(Self::Right),
            other => Err(OpAiError::WireFormat(format!(
                "unknown BranchChoice tag: {other}"
            ))),
        }
    }

    /// Return wire tag.
    pub fn to_tag(self) -> u8 {
        self as u8
    }
}

/// Active bisection interval `[lo, hi]` with agreed `lo_root` and disputed `hi_root`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BisectionRange {
    /// Lower step index where defender and challenger still agree on `lo_root`.
    pub lo: u64,
    /// Upper step index where defender's `hi_root` is disputed by challenger.
    pub hi: u64,
    /// Agreed `StateRoot` at step `lo`.
    pub lo_root: Digest,
    /// Defender's claimed (and disputed) `StateRoot` at step `hi`.
    pub hi_root: Digest,
}

impl BisectionRange {
    /// Canonical midpoint `lo + (hi - lo) / 2` for `hi > lo + 1`.
    pub fn midpoint(&self) -> u64 {
        self.lo + (self.hi - self.lo) / 2
    }

    /// True when the interval has converged to a single atomic transition `lo -> lo + 1`.
    pub fn is_single_step(&self) -> bool {
        self.hi == self.lo + 1
    }
}

/// Exact reason why a dispute terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ResolutionReason {
    /// Single-step judge proved step `lo -> lo + 1` does NOT produce `hi_root`.
    SingleStepFraudProven = 0,
    /// Initial `start_root` did not match deterministic `VmState::initial`.
    InvalidStartRootProven = 1,
    /// Defender supplied a `pre_state` witness whose digest did not match agreed `lo_root`.
    InvalidPreStateWitness = 2,
    /// Single-step judge verified step `lo -> lo + 1` honestly produces `hi_root`.
    SingleStepVerifiedHonest = 3,
    /// Challenger falsely accused a valid `start_root`.
    ValidStartRootVerified = 4,
    /// Defender failed to respond before `turn_deadline_secs`.
    DefenderTimeoutForfeit = 5,
    /// Challenger failed to respond before `turn_deadline_secs`.
    ChallengerTimeoutForfeit = 6,
    /// Concurrent challenge refunded because another challenger already slashed the claim.
    RefundedClaimAlreadySlashed = 7,
}

impl ResolutionReason {
    /// Decode reason from canonical wire tag.
    pub fn from_tag(tag: u8) -> Result<Self, OpAiError> {
        match tag {
            0 => Ok(Self::SingleStepFraudProven),
            1 => Ok(Self::InvalidStartRootProven),
            2 => Ok(Self::InvalidPreStateWitness),
            3 => Ok(Self::SingleStepVerifiedHonest),
            4 => Ok(Self::ValidStartRootVerified),
            5 => Ok(Self::DefenderTimeoutForfeit),
            6 => Ok(Self::ChallengerTimeoutForfeit),
            7 => Ok(Self::RefundedClaimAlreadySlashed),
            other => Err(OpAiError::WireFormat(format!(
                "unknown ResolutionReason tag: {other}"
            ))),
        }
    }

    /// Return canonical wire tag.
    pub fn to_tag(self) -> u8 {
        self as u8
    }
}

/// Economic bond settlement for a single resolved dispute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BondSettlement {
    /// Amount returned/paid to the provider (returned bond if single-dispute settlement,
    /// or compensation delta in multi-challenger coordinator).
    pub provider_payout: u64,
    /// Amount slashed from the provider's bond.
    pub provider_slashed: u64,
    /// Amount returned/paid to the challenger.
    pub challenger_payout: u64,
    /// Amount slashed from the challenger's bond.
    pub challenger_slashed: u64,
    /// Amount burned / sent to protocol burn sink (strictly `> 0` on every slash).
    pub burned: u64,
}

impl BondSettlement {
    /// Compute deterministic bond settlement when the defender is slashed.
    pub fn defender_slashed(provider_bond: u64, challenger_bond: u64, config: &GameConfig) -> Self {
        let reward = ((u128::from(provider_bond) * u128::from(config.challenger_reward_bps))
            / 10_000) as u64;
        let burned = provider_bond - reward;
        Self {
            provider_payout: 0,
            provider_slashed: provider_bond,
            challenger_payout: challenger_bond.saturating_add(reward),
            challenger_slashed: 0,
            burned,
        }
    }

    /// Compute deterministic bond settlement when a malicious/stalling challenger is slashed.
    pub fn challenger_slashed(
        provider_bond: u64,
        challenger_bond: u64,
        config: &GameConfig,
    ) -> Self {
        let comp = ((u128::from(challenger_bond) * u128::from(config.provider_compensation_bps))
            / 10_000) as u64;
        let burned = challenger_bond - comp;
        Self {
            provider_payout: provider_bond.saturating_add(comp),
            provider_slashed: 0,
            challenger_payout: 0,
            challenger_slashed: challenger_bond,
            burned,
        }
    }

    /// Compute clean refund settlement for concurrent watchtowers when another challenger
    /// already proved fraud and slashed the claim.
    pub fn refund_concurrent_challenger(challenger_bond: u64) -> Self {
        Self {
            provider_payout: 0,
            provider_slashed: 0,
            challenger_payout: challenger_bond,
            challenger_slashed: 0,
            burned: 0,
        }
    }

    /// Verify exact conservation of bonds:
    /// `provider_payout + challenger_payout + burned == provider_bond + challenger_bond`.
    pub fn verify_conservation(&self, provider_bond: u64, challenger_bond: u64) -> bool {
        let lhs = u128::from(self.provider_payout)
            + u128::from(self.challenger_payout)
            + u128::from(self.burned);
        let rhs = u128::from(provider_bond) + u128::from(challenger_bond);
        lhs == rhs
    }
}

/// Terminal outcome of an interactive bisection dispute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisputeResolution {
    /// Winning party (`Party::Challenger` if defender slashed, `Party::Defender` if challenger slashed).
    pub winner: Party,
    /// Exact pinpointed step index when resolved by single-step or start-root verification
    /// (`None` on timeout forfeit or concurrent refund).
    pub pinpointed_step: Option<u64>,
    /// Canonical resolution reason.
    pub reason: ResolutionReason,
    /// Timestamp at which the dispute finalized (seconds).
    pub resolved_at_secs: u64,
    /// Deterministic bond payout and slashing breakdown.
    pub settlement: BondSettlement,
}

/// Current state machine phase of a [`DisputeGame`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisputePhase {
    /// Waiting for the defender to publish `state_root(mid)`.
    AwaitDefenderMidpoint {
        /// Required midpoint step index.
        mid: u64,
    },
    /// Waiting for the challenger to pick [`BranchChoice::Left`] or [`BranchChoice::Right`].
    AwaitChallengerBranch {
        /// Midpoint step index committed by defender.
        mid: u64,
        /// Defender's claimed `StateRoot` at `mid`.
        mid_root: Digest,
    },
    /// Bisection converged to `hi == lo + 1`; waiting for defender to supply `pre_state` at `lo`.
    AwaitDefenderSingleStep,
    /// Dispute has reached terminal resolution.
    Resolved(DisputeResolution),
}

/// Immutable audit log entry recorded for every move in a [`DisputeGame`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisputeMove {
    /// Initial challenge opened against a claim.
    Opened {
        /// Challenger identifier.
        challenger_id: String,
        /// Counter-bond locked by the challenger.
        challenger_bond: u64,
        /// Opening timestamp in seconds.
        at_secs: u64,
    },
    /// Zero-round `start_root` fraud verification executed.
    StartRootChecked {
        /// Deterministically recomputed initial state root.
        expected_start_root: Digest,
        /// Claim's submitted `start_root`.
        claimed_start_root: Digest,
        /// Timestamp in seconds.
        at_secs: u64,
    },
    /// Defender submitted the state root at `mid = lo + (hi - lo) / 2`.
    DefenderMidpoint {
        /// 0-indexed bisection round number.
        round: u32,
        /// Midpoint step index.
        mid: u64,
        /// Defender's committed `StateRoot` at `mid`.
        mid_root: Digest,
        /// Timestamp in seconds.
        at_secs: u64,
    },
    /// Challenger narrowed the active interval to `[next_lo, next_hi]`.
    ChallengerBranch {
        /// 1-indexed completed bisection round count.
        round: u32,
        /// Left (`[lo, mid]`) or Right (`[mid, hi]`).
        choice: BranchChoice,
        /// Resulting lower bound.
        next_lo: u64,
        /// Resulting upper bound.
        next_hi: u64,
        /// Timestamp in seconds.
        at_secs: u64,
    },
    /// On-chain-equivalent single-step judge executed step `step_index -> step_index + 1`.
    SingleStepJudged {
        /// Judged step index `lo`.
        step_index: u64,
        /// Operation tag for step `lo`.
        op_tag: String,
        /// Whether `pre_state` hashed to agreed `lo_root`.
        pre_root_valid: bool,
        /// Recomputed post-state root after executing step `lo`.
        actual_post_root: Digest,
        /// Defender's claimed post-state root at `lo + 1`.
        claimed_post_root: Digest,
        /// True iff defender's single step was valid.
        defender_valid: bool,
        /// Timestamp in seconds.
        at_secs: u64,
    },
    /// Automatic default forfeit enforced after a party exceeded `turn_deadline_secs`.
    TimeoutForfeit {
        /// Party that failed to respond before the deadline.
        timed_out_party: Party,
        /// Active turn deadline in seconds.
        deadline_secs: u64,
        /// Observed timestamp strictly greater than `deadline_secs`.
        observed_at_secs: u64,
    },
    /// Concurrent dispute cleanly refunded because another challenger already slashed the claim.
    ConcurrentRefund {
        /// Timestamp in seconds.
        at_secs: u64,
    },
}

/// Interactive $O(\log_2 T)$ binary-search bisection fraud-proof game between
/// one claim provider (Defender) and one permissionless Challenger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisputeGame {
    /// Canonical SHA-256 dispute identifier.
    pub dispute_id: Digest,
    /// Disputed optimistic claim.
    pub claim: Claim,
    /// Permissionless challenger identifier.
    pub challenger_id: String,
    /// Counter-bond locked by the challenger.
    pub challenger_bond: u64,
    /// Zero-indexed challenge slot on this claim (determines escalated bond and dispute_id).
    pub challenge_slot: usize,
    /// Bound economic & liveness configuration.
    pub config: GameConfig,
    /// Opening timestamp (seconds).
    pub opened_at_secs: u64,
    /// Timestamp of the most recent valid move (enforces monotonicity & deadlines).
    pub last_action_at_secs: u64,
    /// Current bisection interval `[lo, hi]`.
    pub range: BisectionRange,
    /// Number of completed bisection rounds (`<= ceil(log2 T)`).
    pub rounds_completed: u32,
    /// Current turn / resolution phase.
    pub phase: DisputePhase,
    /// Chronological audit trail of every move in this game.
    pub moves: Vec<DisputeMove>,
    /// Pre-state witness supplied by the defender at `hi == lo + 1` (for sovereign replay).
    pub single_step_witness: Option<VmState>,
    /// Final single-step judge verdict if resolved via single-step execution.
    pub single_step_verdict: Option<SingleStepVerdict>,
}

impl DisputeGame {
    /// Open a new interactive bisection dispute against `claim` within its challenge window.
    pub fn open(
        claim: Claim,
        challenger_id: impl Into<String>,
        challenger_bond: u64,
        config: GameConfig,
        opened_at_secs: u64,
        challenge_slot: usize,
    ) -> Result<Self, OpAiError> {
        config.validate()?;
        if !claim.verify_id() {
            return Err(OpAiError::InvalidClaim(
                "claim_id does not match canonical claim fields".to_string(),
            ));
        }
        if claim.total_steps == 0 {
            return Err(OpAiError::InvalidClaim(
                "claim.total_steps must be >= 1".to_string(),
            ));
        }
        if claim.bond < config.provider_bond {
            return Err(OpAiError::InsufficientBond {
                required: config.provider_bond,
                provided: claim.bond,
            });
        }
        let challenger_id = challenger_id.into();
        if challenger_id.is_empty() {
            return Err(OpAiError::InvalidClaim(
                "challenger_id must be non-empty".to_string(),
            ));
        }
        if opened_at_secs < claim.submitted_at_secs {
            return Err(OpAiError::ChallengeWindowViolation(
                "challenge timestamp precedes claim submission".to_string(),
            ));
        }
        let window_end = claim
            .submitted_at_secs
            .saturating_add(config.challenge_window_secs);
        if opened_at_secs > window_end {
            return Err(OpAiError::ChallengeWindowViolation(format!(
                "challenge window closed at {window_end}s (got {opened_at_secs}s)"
            )));
        }
        let required_bond = config.required_challenger_bond(challenge_slot);
        if challenger_bond < required_bond {
            return Err(OpAiError::InsufficientBond {
                required: required_bond,
                provided: challenger_bond,
            });
        }

        let dispute_id = compute_dispute_id(
            &claim.claim_id,
            &challenger_id,
            opened_at_secs,
            challenge_slot,
        );
        let range = BisectionRange {
            lo: 0,
            hi: claim.total_steps,
            lo_root: claim.start_root,
            hi_root: claim.end_root,
        };
        let phase = if range.is_single_step() {
            DisputePhase::AwaitDefenderSingleStep
        } else {
            DisputePhase::AwaitDefenderMidpoint {
                mid: range.midpoint(),
            }
        };

        Ok(Self {
            dispute_id,
            claim,
            challenger_id: challenger_id.clone(),
            challenger_bond,
            challenge_slot,
            config,
            opened_at_secs,
            last_action_at_secs: opened_at_secs,
            range,
            rounds_completed: 0,
            phase,
            moves: vec![DisputeMove::Opened {
                challenger_id,
                challenger_bond,
                at_secs: opened_at_secs,
            }],
            single_step_witness: None,
            single_step_verdict: None,
        })
    }

    /// Return the active turn deadline timestamp (`last_action_at_secs + turn_deadline_secs`).
    pub fn turn_deadline_secs(&self) -> u64 {
        self.last_action_at_secs
            .saturating_add(self.config.turn_deadline_secs)
    }

    /// Return the terminal [`DisputeResolution`] if the game has resolved.
    pub fn resolution(&self) -> Option<&DisputeResolution> {
        match &self.phase {
            DisputePhase::Resolved(res) => Some(res),
            _ => None,
        }
    }

    /// Zero-round check verifying whether `claim.start_root` matches the canonical
    /// deterministic `VmState::initial(model, input)` root.
    ///
    /// - If `claim.start_root` is forged, defender is immediately slashed (`pinpointed_step = Some(0)`).
    /// - If `claim.start_root` is valid, the challenger's false start-root accusation is immediately slashed.
    pub fn assert_invalid_start_root(
        &mut self,
        profile: &ExecutionProfile,
        model: &QuantizedModel,
        input: &InputCommitment,
        at_secs: u64,
    ) -> Result<&DisputeResolution, OpAiError> {
        if matches!(self.phase, DisputePhase::Resolved(_)) {
            return Err(OpAiError::DisputeAlreadyResolved);
        }
        self.check_turn_timestamp(at_secs)?;
        profile.validate()?;
        model.validate()?;

        let init = VmState::initial(model, input)?;
        let expected_start_root =
            init.state_root(&profile.digest(), &model.digest(), &input.input_hash());
        self.moves.push(DisputeMove::StartRootChecked {
            expected_start_root,
            claimed_start_root: self.claim.start_root,
            at_secs,
        });
        self.last_action_at_secs = at_secs;

        let res = if !ct_eq_digest(&expected_start_root, &self.claim.start_root) {
            DisputeResolution {
                winner: Party::Challenger,
                pinpointed_step: Some(0),
                reason: ResolutionReason::InvalidStartRootProven,
                resolved_at_secs: at_secs,
                settlement: BondSettlement::defender_slashed(
                    self.claim.bond,
                    self.challenger_bond,
                    &self.config,
                ),
            }
        } else {
            DisputeResolution {
                winner: Party::Defender,
                pinpointed_step: Some(0),
                reason: ResolutionReason::ValidStartRootVerified,
                resolved_at_secs: at_secs,
                settlement: BondSettlement::challenger_slashed(
                    self.claim.bond,
                    self.challenger_bond,
                    &self.config,
                ),
            }
        };
        self.phase = DisputePhase::Resolved(res);
        Ok(self.resolution().expect("just resolved"))
    }

    /// Defender submits the `StateRoot` at the required midpoint `mid = lo + (hi - lo) / 2`.
    pub fn submit_defender_midpoint(
        &mut self,
        mid: u64,
        mid_root: Digest,
        at_secs: u64,
    ) -> Result<(), OpAiError> {
        let expected_mid = match self.phase {
            DisputePhase::AwaitDefenderMidpoint { mid: m } => m,
            DisputePhase::Resolved(_) => return Err(OpAiError::DisputeAlreadyResolved),
            _ => {
                return Err(OpAiError::WrongTurn(
                    "not expecting defender midpoint move".to_string(),
                ))
            }
        };
        if mid != expected_mid {
            return Err(OpAiError::InvalidBisectionMidpoint {
                expected: expected_mid,
                got: mid,
            });
        }
        self.check_turn_timestamp(at_secs)?;

        self.moves.push(DisputeMove::DefenderMidpoint {
            round: self.rounds_completed,
            mid,
            mid_root,
            at_secs,
        });
        self.last_action_at_secs = at_secs;
        self.phase = DisputePhase::AwaitChallengerBranch { mid, mid_root };
        Ok(())
    }

    /// Challenger chooses the half-interval containing the first disagreement.
    pub fn submit_challenger_branch(
        &mut self,
        choice: BranchChoice,
        at_secs: u64,
    ) -> Result<(), OpAiError> {
        let (mid, mid_root) = match self.phase {
            DisputePhase::AwaitChallengerBranch { mid, mid_root } => (mid, mid_root),
            DisputePhase::Resolved(_) => return Err(OpAiError::DisputeAlreadyResolved),
            _ => {
                return Err(OpAiError::WrongTurn(
                    "not expecting challenger branch choice".to_string(),
                ))
            }
        };
        self.check_turn_timestamp(at_secs)?;

        match choice {
            BranchChoice::Left => {
                self.range.hi = mid;
                self.range.hi_root = mid_root;
            }
            BranchChoice::Right => {
                self.range.lo = mid;
                self.range.lo_root = mid_root;
            }
        }
        self.rounds_completed += 1;
        self.last_action_at_secs = at_secs;
        self.moves.push(DisputeMove::ChallengerBranch {
            round: self.rounds_completed,
            choice,
            next_lo: self.range.lo,
            next_hi: self.range.hi,
            at_secs,
        });

        self.phase = if self.range.is_single_step() {
            DisputePhase::AwaitDefenderSingleStep
        } else {
            DisputePhase::AwaitDefenderMidpoint {
                mid: self.range.midpoint(),
            }
        };
        Ok(())
    }

    /// Once bisection converges to `hi == lo + 1`, defender submits `pre_state` at step `lo`
    /// and the deterministic single-step judge executes ONLY step `lo -> lo + 1`.
    pub fn submit_single_step_proof(
        &mut self,
        profile: &ExecutionProfile,
        model: &QuantizedModel,
        input: &InputCommitment,
        pre_state: VmState,
        at_secs: u64,
    ) -> Result<&DisputeResolution, OpAiError> {
        match self.phase {
            DisputePhase::AwaitDefenderSingleStep => {}
            DisputePhase::Resolved(_) => return Err(OpAiError::DisputeAlreadyResolved),
            _ => {
                return Err(OpAiError::NotReadyForSingleStep {
                    lo: self.range.lo,
                    hi: self.range.hi,
                })
            }
        }
        if !self.range.is_single_step() {
            return Err(OpAiError::NotReadyForSingleStep {
                lo: self.range.lo,
                hi: self.range.hi,
            });
        }
        self.check_turn_timestamp(at_secs)?;

        let verdict = verify_single_step(
            profile,
            model,
            input,
            self.claim.total_steps,
            self.range.lo,
            &pre_state,
            &self.range.lo_root,
            &self.range.hi_root,
        )?;

        self.single_step_witness = Some(pre_state);
        self.single_step_verdict = Some(verdict.clone());
        self.last_action_at_secs = at_secs;
        self.moves.push(DisputeMove::SingleStepJudged {
            step_index: verdict.step_index,
            op_tag: verdict.op_tag.clone(),
            pre_root_valid: verdict.pre_root_valid,
            actual_post_root: verdict.actual_post_root,
            claimed_post_root: verdict.claimed_post_root,
            defender_valid: verdict.defender_valid,
            at_secs,
        });

        let res = if !verdict.pre_root_valid {
            DisputeResolution {
                winner: Party::Challenger,
                pinpointed_step: Some(verdict.step_index),
                reason: ResolutionReason::InvalidPreStateWitness,
                resolved_at_secs: at_secs,
                settlement: BondSettlement::defender_slashed(
                    self.claim.bond,
                    self.challenger_bond,
                    &self.config,
                ),
            }
        } else if !verdict.defender_valid {
            DisputeResolution {
                winner: Party::Challenger,
                pinpointed_step: Some(verdict.step_index),
                reason: ResolutionReason::SingleStepFraudProven,
                resolved_at_secs: at_secs,
                settlement: BondSettlement::defender_slashed(
                    self.claim.bond,
                    self.challenger_bond,
                    &self.config,
                ),
            }
        } else {
            DisputeResolution {
                winner: Party::Defender,
                pinpointed_step: Some(verdict.step_index),
                reason: ResolutionReason::SingleStepVerifiedHonest,
                resolved_at_secs: at_secs,
                settlement: BondSettlement::challenger_slashed(
                    self.claim.bond,
                    self.challenger_bond,
                    &self.config,
                ),
            }
        };

        self.phase = DisputePhase::Resolved(res);
        Ok(self.resolution().expect("just resolved"))
    }

    /// Enforce automatic liveness forfeit & slashing when the active party fails to
    /// submit a move before `turn_deadline_secs` (`observed_at_secs > deadline`).
    pub fn claim_timeout_forfeit(
        &mut self,
        observed_at_secs: u64,
    ) -> Result<&DisputeResolution, OpAiError> {
        let timed_out_party = match self.phase {
            DisputePhase::AwaitDefenderMidpoint { .. } | DisputePhase::AwaitDefenderSingleStep => {
                Party::Defender
            }
            DisputePhase::AwaitChallengerBranch { .. } => Party::Challenger,
            DisputePhase::Resolved(_) => return Err(OpAiError::DisputeAlreadyResolved),
        };
        let deadline = self.turn_deadline_secs();
        if observed_at_secs <= deadline {
            return Err(OpAiError::TurnNotExpired {
                deadline_secs: deadline,
                observed_secs: observed_at_secs,
            });
        }

        self.last_action_at_secs = observed_at_secs;
        self.moves.push(DisputeMove::TimeoutForfeit {
            timed_out_party,
            deadline_secs: deadline,
            observed_at_secs,
        });

        let res = match timed_out_party {
            Party::Defender => DisputeResolution {
                winner: Party::Challenger,
                pinpointed_step: None,
                reason: ResolutionReason::DefenderTimeoutForfeit,
                resolved_at_secs: observed_at_secs,
                settlement: BondSettlement::defender_slashed(
                    self.claim.bond,
                    self.challenger_bond,
                    &self.config,
                ),
            },
            Party::Challenger => DisputeResolution {
                winner: Party::Defender,
                pinpointed_step: None,
                reason: ResolutionReason::ChallengerTimeoutForfeit,
                resolved_at_secs: observed_at_secs,
                settlement: BondSettlement::challenger_slashed(
                    self.claim.bond,
                    self.challenger_bond,
                    &self.config,
                ),
            },
        };
        self.phase = DisputePhase::Resolved(res);
        Ok(self.resolution().expect("just resolved"))
    }

    /// Mark an unresolved concurrent challenge as refunded because another challenger
    /// already slashed the underlying claim.
    pub(crate) fn mark_refunded_claim_already_slashed(&mut self, at_secs: u64) {
        if matches!(self.phase, DisputePhase::Resolved(_)) {
            return;
        }
        let ts = at_secs.max(self.last_action_at_secs);
        self.last_action_at_secs = ts;
        self.moves
            .push(DisputeMove::ConcurrentRefund { at_secs: ts });
        self.phase = DisputePhase::Resolved(DisputeResolution {
            winner: Party::Challenger,
            pinpointed_step: None,
            reason: ResolutionReason::RefundedClaimAlreadySlashed,
            resolved_at_secs: ts,
            settlement: BondSettlement::refund_concurrent_challenger(self.challenger_bond),
        });
    }

    fn check_turn_timestamp(&self, at_secs: u64) -> Result<(), OpAiError> {
        if at_secs < self.last_action_at_secs {
            return Err(OpAiError::DeadlineViolation(format!(
                "move timestamp {at_secs}s precedes last action {}s",
                self.last_action_at_secs
            )));
        }
        let deadline = self.turn_deadline_secs();
        if at_secs > deadline {
            return Err(OpAiError::DeadlineViolation(format!(
                "move timestamp {at_secs}s exceeded turn deadline {deadline}s"
            )));
        }
        Ok(())
    }
}

/// Theoretical upper bound on interactive bisection rounds $\lceil \log_2 T \rceil$.
pub fn max_bisection_rounds(total_steps: u64) -> u32 {
    if total_steps <= 1 {
        0
    } else {
        (total_steps - 1).ilog2() + 1
    }
}

fn compute_dispute_id(
    claim_id: &Digest,
    challenger_id: &str,
    opened_at_secs: u64,
    challenge_slot: usize,
) -> Digest {
    let mut h = Sha256::with_domain(DOMAIN_DISPUTE);
    h.update(claim_id);
    h.update_str(challenger_id);
    h.update_u64_le(opened_at_secs);
    h.update_u64_le(challenge_slot as u64);
    h.finalize()
}

/// Strategy governing defender or challenger moves in automated dispute execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActorStrategy {
    /// Follow the exact protocol using the actor's computed [`ExecutionTrace`].
    Honest,
    /// Stall / drop offline at bisection round `round` (triggers timeout forfeit).
    StallAtRound(u32),
    /// Stall / drop offline when single-step pre-state submission is required.
    StallAtSingleStep,
    /// Defender supplies a corrupted `pre_state` witness that does not hash to `lo_root`.
    SubmitCorruptPreState,
    /// Malicious griefing challenger challenging an honest claim by picking a fixed branch.
    MaliciousGrief {
        /// Whether to always pick `Left` (`true`) or `Right` (`false`).
        prefer_left: bool,
    },
}

/// Drive a [`DisputeGame`] to completion between a defender trace and a challenger trace
/// using explicit per-turn latency (`turn_step_secs`) and actor strategies.
#[allow(clippy::too_many_arguments)]
pub fn run_dispute_to_completion(
    game: &mut DisputeGame,
    profile: &ExecutionProfile,
    model: &QuantizedModel,
    input: &InputCommitment,
    defender_trace: &ExecutionTrace,
    challenger_trace: &ExecutionTrace,
    defender_strategy: ActorStrategy,
    challenger_strategy: ActorStrategy,
    turn_step_secs: u64,
) -> Result<DisputeResolution, OpAiError> {
    let step_dt = turn_step_secs.max(1);
    let mut now = game.last_action_at_secs;

    // First, an honest challenger checks start_root before wasting bisection rounds.
    if matches!(challenger_strategy, ActorStrategy::Honest)
        && !ct_eq_digest(&game.claim.start_root, &challenger_trace.start_root())
    {
        now = now.saturating_add(step_dt.min(game.config.turn_deadline_secs));
        let res = game.assert_invalid_start_root(profile, model, input, now)?;
        return Ok(res.clone());
    }

    loop {
        match game.phase.clone() {
            DisputePhase::Resolved(res) => return Ok(res),
            DisputePhase::AwaitDefenderMidpoint { mid } => {
                if matches!(defender_strategy, ActorStrategy::StallAtRound(r) if r == game.rounds_completed)
                {
                    let expired_at = game.turn_deadline_secs().saturating_add(1);
                    let res = game.claim_timeout_forfeit(expired_at)?;
                    return Ok(res.clone());
                }
                now = now.saturating_add(step_dt);
                let mid_root = defender_trace.root_at(mid)?;
                game.submit_defender_midpoint(mid, mid_root, now)?;
            }
            DisputePhase::AwaitChallengerBranch { mid, mid_root } => {
                if matches!(challenger_strategy, ActorStrategy::StallAtRound(r) if r == game.rounds_completed)
                {
                    let expired_at = game.turn_deadline_secs().saturating_add(1);
                    let res = game.claim_timeout_forfeit(expired_at)?;
                    return Ok(res.clone());
                }
                now = now.saturating_add(step_dt);
                let choice = match challenger_strategy {
                    ActorStrategy::MaliciousGrief { prefer_left } => {
                        if prefer_left {
                            BranchChoice::Left
                        } else {
                            BranchChoice::Right
                        }
                    }
                    _ => {
                        let expected_mid = challenger_trace.root_at(mid)?;
                        if ct_eq_digest(&mid_root, &expected_mid) {
                            BranchChoice::Right
                        } else {
                            BranchChoice::Left
                        }
                    }
                };
                game.submit_challenger_branch(choice, now)?;
            }
            DisputePhase::AwaitDefenderSingleStep => {
                if matches!(defender_strategy, ActorStrategy::StallAtSingleStep) {
                    let expired_at = game.turn_deadline_secs().saturating_add(1);
                    let res = game.claim_timeout_forfeit(expired_at)?;
                    return Ok(res.clone());
                }
                now = now.saturating_add(step_dt);
                let mut pre_state = defender_trace.state_at(game.range.lo)?.clone();
                if matches!(defender_strategy, ActorStrategy::SubmitCorruptPreState) {
                    pre_state.hidden[0] = pre_state.hidden[0].wrapping_add(1);
                }
                let res = game.submit_single_step_proof(profile, model, input, pre_state, now)?;
                return Ok(res.clone());
            }
        }
    }
}

/// Lifecycle status of a bonded claim inside [`OptimisticCoordinator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimStatus {
    /// Inside challenge window or awaiting resolution of active disputes.
    Pending,
    /// Challenge window elapsed with zero active disputes; provider bond released.
    Finalized,
    /// Fraud or defender timeout proven; provider bond slashed.
    Slashed,
}

/// Tracked claim entry inside [`OptimisticCoordinator`].
#[derive(Debug, Clone)]
pub struct ClaimRecord {
    /// Underlying bonded claim.
    pub claim: Claim,
    /// Current lifecycle status.
    pub status: ClaimStatus,
    /// Total challenges ever opened against this claim (drives anti-griefing escalation).
    pub total_challenges_opened: usize,
    /// IDs of currently unresolved concurrent disputes against this claim.
    pub active_disputes: BTreeSet<Digest>,
    /// IDs of all disputes (active + resolved) against this claim.
    pub all_disputes: Vec<Digest>,
    /// Active challenger IDs currently fighting this claim (prevents duplicate same-ID spam).
    pub active_challengers: BTreeSet<String>,
    /// Cumulative anti-griefing compensation already paid to the provider from slashed challengers.
    pub provider_compensation_earned: u64,
}

/// Permissionless, multi-tenant optimistic claim & concurrent tournament coordinator.
///
/// Tracks bonds across arbitrary providers and challengers with exact conservation
/// of funds, anti-griefing escalation, and clean concurrent watchtower refunds.
#[derive(Debug, Clone)]
pub struct OptimisticCoordinator {
    /// Protocol economic and liveness settings.
    pub config: GameConfig,
    claims: BTreeMap<Digest, ClaimRecord>,
    disputes: BTreeMap<Digest, DisputeGame>,
    /// Cumulative bond inflows locked into the coordinator.
    pub total_deposited: u128,
    /// Currently locked bonds across pending claims & active disputes.
    pub total_locked: u128,
    /// Cumulative payouts credited to participant balances.
    pub total_paid_out: u128,
    /// Cumulative slashed bonds sent to the burn sink.
    pub total_burned: u128,
    /// Per-participant realized payout ledger (returned bonds + rewards + compensations).
    pub participant_payouts: BTreeMap<String, u128>,
}

impl OptimisticCoordinator {
    /// Create a fresh coordinator with validated [`GameConfig`].
    pub fn new(config: GameConfig) -> Result<Self, OpAiError> {
        config.validate()?;
        Ok(Self {
            config,
            claims: BTreeMap::new(),
            disputes: BTreeMap::new(),
            total_deposited: 0,
            total_locked: 0,
            total_paid_out: 0,
            total_burned: 0,
            participant_payouts: BTreeMap::new(),
        })
    }

    /// Submit a bonded optimistic claim into the coordinator.
    pub fn submit_claim(&mut self, claim: Claim) -> Result<Digest, OpAiError> {
        if !claim.verify_id() {
            return Err(OpAiError::InvalidClaim(
                "claim_id mismatch on submit_claim".to_string(),
            ));
        }
        if claim.bond < self.config.provider_bond {
            return Err(OpAiError::InsufficientBond {
                required: self.config.provider_bond,
                provided: claim.bond,
            });
        }
        if self.claims.contains_key(&claim.claim_id) {
            return Err(OpAiError::DuplicateClaim);
        }
        let claim_id = claim.claim_id;
        let bond = u128::from(claim.bond);
        self.total_deposited += bond;
        self.total_locked += bond;
        self.claims.insert(
            claim_id,
            ClaimRecord {
                claim,
                status: ClaimStatus::Pending,
                total_challenges_opened: 0,
                active_disputes: BTreeSet::new(),
                all_disputes: Vec::new(),
                active_challengers: BTreeSet::new(),
                provider_compensation_earned: 0,
            },
        );
        Ok(claim_id)
    }

    /// Open a permissionless challenge against `claim_id` with anti-griefing bond escalation.
    pub fn open_challenge(
        &mut self,
        claim_id: &Digest,
        challenger_id: impl Into<String>,
        challenger_bond: u64,
        opened_at_secs: u64,
    ) -> Result<Digest, OpAiError> {
        let challenger_id = challenger_id.into();
        let rec = self.claims.get(claim_id).ok_or(OpAiError::ClaimNotFound)?;
        if rec.status != ClaimStatus::Pending {
            return Err(OpAiError::ClaimAlreadySettled);
        }
        if rec.active_disputes.len() >= self.config.max_concurrent_challenges_per_claim {
            return Err(OpAiError::MaxConcurrentChallengesExceeded {
                limit: self.config.max_concurrent_challenges_per_claim,
            });
        }
        if rec.active_challengers.contains(&challenger_id) {
            return Err(OpAiError::ChallengerAlreadyActive(challenger_id));
        }
        let slot = rec.total_challenges_opened;
        let game = DisputeGame::open(
            rec.claim.clone(),
            challenger_id.clone(),
            challenger_bond,
            self.config.clone(),
            opened_at_secs,
            slot,
        )?;
        let dispute_id = game.dispute_id;

        let rec_mut = self.claims.get_mut(claim_id).expect("claim checked above");
        rec_mut.total_challenges_opened += 1;
        rec_mut.active_disputes.insert(dispute_id);
        rec_mut.all_disputes.push(dispute_id);
        rec_mut.active_challengers.insert(challenger_id);

        let bond = u128::from(challenger_bond);
        self.total_deposited += bond;
        self.total_locked += bond;
        self.disputes.insert(dispute_id, game);
        Ok(dispute_id)
    }

    /// Borrow a claim record by `claim_id`.
    pub fn claim(&self, claim_id: &Digest) -> Option<&ClaimRecord> {
        self.claims.get(claim_id)
    }

    /// Borrow a dispute game by `dispute_id`.
    pub fn dispute(&self, dispute_id: &Digest) -> Option<&DisputeGame> {
        self.disputes.get(dispute_id)
    }

    /// Mutably borrow a dispute game to advance bisection moves, then call
    /// [`Self::settle_dispute`] once the game reaches [`DisputePhase::Resolved`].
    pub fn dispute_mut(&mut self, dispute_id: &Digest) -> Result<&mut DisputeGame, OpAiError> {
        self.disputes
            .get_mut(dispute_id)
            .ok_or(OpAiError::DisputeNotFound)
    }

    /// Apply bond accounting for a resolved dispute inside the coordinator,
    /// handling concurrent watchtower refunds if the defender is slashed.
    pub fn settle_dispute(&mut self, dispute_id: &Digest) -> Result<DisputeResolution, OpAiError> {
        let (claim_id, challenger_id, challenger_bond, resolution) = {
            let game = self
                .disputes
                .get(dispute_id)
                .ok_or(OpAiError::DisputeNotFound)?;
            let res = game.resolution().ok_or_else(|| {
                OpAiError::WrongTurn("cannot settle unresolved dispute".to_string())
            })?;
            (
                game.claim.claim_id,
                game.challenger_id.clone(),
                game.challenger_bond,
                res.clone(),
            )
        };

        let rec = self
            .claims
            .get_mut(&claim_id)
            .ok_or(OpAiError::ClaimNotFound)?;
        if !rec.active_disputes.remove(dispute_id) {
            // Already settled earlier (idempotent return).
            return Ok(resolution);
        }
        rec.active_challengers.remove(&challenger_id);

        let provider_id = rec.claim.provider_id.clone();
        let provider_bond = rec.claim.bond;

        match resolution.winner {
            Party::Challenger => {
                // Defender was slashed!
                if rec.status == ClaimStatus::Pending {
                    rec.status = ClaimStatus::Slashed;
                    let sibling_ids: Vec<Digest> = rec.active_disputes.iter().copied().collect();
                    rec.active_disputes.clear();
                    rec.active_challengers.clear();

                    let reward = resolution
                        .settlement
                        .challenger_payout
                        .saturating_sub(challenger_bond);
                    let burned = resolution.settlement.burned;

                    // Unlock provider_bond + winning challenger_bond.
                    self.total_locked -= u128::from(provider_bond) + u128::from(challenger_bond);
                    self.credit(
                        &challenger_id,
                        u128::from(challenger_bond) + u128::from(reward),
                    );
                    self.total_burned += u128::from(burned);

                    // Cleanly refund all other active concurrent watchtowers on this claim.
                    for sib_id in sibling_ids {
                        let refund_info = if let Some(sib) = self.disputes.get_mut(&sib_id) {
                            let sib_challenger = sib.challenger_id.clone();
                            let sib_bond = sib.challenger_bond;
                            sib.mark_refunded_claim_already_slashed(resolution.resolved_at_secs);
                            Some((sib_challenger, sib_bond))
                        } else {
                            None
                        };
                        if let Some((sib_challenger, sib_bond)) = refund_info {
                            self.total_locked -= u128::from(sib_bond);
                            self.credit(&sib_challenger, u128::from(sib_bond));
                        }
                    }
                } else {
                    // Claim was already slashed; refund challenger bond cleanly.
                    self.total_locked -= u128::from(challenger_bond);
                    self.credit(&challenger_id, u128::from(challenger_bond));
                }
            }
            Party::Defender => {
                // Malicious or stalling challenger slashed; provider stays active until window closes.
                let comp = resolution
                    .settlement
                    .provider_payout
                    .saturating_sub(provider_bond);
                let burned = resolution.settlement.burned;
                rec.provider_compensation_earned =
                    rec.provider_compensation_earned.saturating_add(comp);
                self.total_locked -= u128::from(challenger_bond);
                self.credit(&provider_id, u128::from(comp));
                self.total_burned += u128::from(burned);
            }
        }

        Ok(resolution)
    }

    /// Finalize an unslashed claim once its challenge window has elapsed AND all
    /// concurrent disputes against it have been resolved in the provider's favor.
    pub fn finalize_claim(&mut self, claim_id: &Digest, now_secs: u64) -> Result<u64, OpAiError> {
        let rec = self
            .claims
            .get_mut(claim_id)
            .ok_or(OpAiError::ClaimNotFound)?;
        if rec.status != ClaimStatus::Pending {
            return Err(OpAiError::ClaimAlreadySettled);
        }
        let window_end = rec
            .claim
            .submitted_at_secs
            .saturating_add(self.config.challenge_window_secs);
        if now_secs < window_end {
            return Err(OpAiError::ChallengeWindowViolation(format!(
                "challenge window open until {window_end}s (now={now_secs}s)"
            )));
        }
        if !rec.active_disputes.is_empty() {
            return Err(OpAiError::ChallengeWindowViolation(format!(
                "cannot finalize claim while {} active dispute(s) remain unresolved",
                rec.active_disputes.len()
            )));
        }
        rec.status = ClaimStatus::Finalized;
        let provider_id = rec.claim.provider_id.clone();
        let bond = rec.claim.bond;
        self.total_locked -= u128::from(bond);
        self.credit(&provider_id, u128::from(bond));
        Ok(bond)
    }

    /// Verify exact global bond conservation across all deposits, locks, payouts, and burns.
    pub fn verify_conservation(&self) -> bool {
        let sum_payouts: u128 = self.participant_payouts.values().copied().sum();
        sum_payouts == self.total_paid_out
            && self.total_deposited == self.total_locked + self.total_paid_out + self.total_burned
    }

    fn credit(&mut self, participant: &str, amount: u128) {
        self.total_paid_out += amount;
        *self
            .participant_payouts
            .entry(participant.to_string())
            .or_insert(0) += amount;
    }
}
