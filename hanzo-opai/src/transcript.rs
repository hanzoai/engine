// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Sovereign dispute transcript wire & JSON codecs and independent third-party
//! replay verifier (`verify_transcript_sovereign`, zero Hanzo control-plane calls).

use std::collections::BTreeMap;

use crate::crypto::{
    ct_eq_digest, digest_to_hex, hex_to_digest, Digest, Sha256, DOMAIN_TRANSCRIPT,
};
use crate::error::OpAiError;
use crate::game::{
    BondSettlement, BranchChoice, Claim, DisputeGame, DisputeMove, DisputeResolution, GameConfig,
    Party, ResolutionReason,
};
use crate::profile::{ExecutionProfile, InputCommitment};
use crate::vm::{QuantizedModel, VmState};

/// Canonical binary wire envelope magic header (`OPAI`).
pub const WIRE_MAGIC: [u8; 4] = *b"OPAI";

/// Canonical binary wire schema version.
pub const WIRE_VERSION: u16 = 1;

/// Anti-DoS upper bound on vector element counts during wire deserialization.
const MAX_WIRE_VEC_LEN: usize = 1 << 20;

/// Self-contained, sovereign dispute transcript exportable to canonical binary
/// wire bytes or JSON and independently replayable with zero Hanzo calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisputeTranscript {
    /// Canonical SHA-256 digest binding the entire transcript body.
    pub transcript_digest: Digest,
    /// Dispute identifier (`dispute_id`).
    pub dispute_id: Digest,
    /// Disputed provider claim.
    pub claim: Claim,
    /// Permissionless challenger identifier.
    pub challenger_id: String,
    /// Counter-bond locked by the challenger.
    pub challenger_bond: u64,
    /// Zero-indexed challenge slot on this claim.
    pub challenge_slot: usize,
    /// Bound economic & liveness configuration.
    pub config: GameConfig,
    /// Opening timestamp in seconds.
    pub opened_at_secs: u64,
    /// Completed bisection rounds.
    pub rounds_completed: u32,
    /// Complete chronological audit trail of moves.
    pub moves: Vec<DisputeMove>,
    /// Optional `VmState` pre-state witness supplied at `hi == lo + 1`.
    pub single_step_witness: Option<VmState>,
    /// Terminal dispute resolution & bond settlement.
    pub resolution: DisputeResolution,
}

impl DisputeTranscript {
    /// Export a self-contained [`DisputeTranscript`] from a resolved [`DisputeGame`].
    pub fn from_resolved_game(game: &DisputeGame) -> Result<Self, OpAiError> {
        let resolution = game
            .resolution()
            .ok_or_else(|| {
                OpAiError::TranscriptIntegrity(
                    "cannot export transcript from an unresolved DisputeGame".to_string(),
                )
            })?
            .clone();

        let mut transcript = Self {
            transcript_digest: [0u8; 32],
            dispute_id: game.dispute_id,
            claim: game.claim.clone(),
            challenger_id: game.challenger_id.clone(),
            challenger_bond: game.challenger_bond,
            challenge_slot: game.challenge_slot,
            config: game.config.clone(),
            opened_at_secs: game.opened_at_secs,
            rounds_completed: game.rounds_completed,
            moves: game.moves.clone(),
            single_step_witness: game.single_step_witness.clone(),
            resolution,
        };
        transcript.transcript_digest = transcript.compute_digest();
        Ok(transcript)
    }

    /// Compute the canonical domain-separated SHA-256 digest over all transcript fields.
    pub fn compute_digest(&self) -> Digest {
        let body = self.encode_body_bytes();
        let mut h = Sha256::with_domain(DOMAIN_TRANSCRIPT);
        h.update_len_prefixed(&body);
        h.finalize()
    }

    /// Verify `transcript_digest` against the canonical body digest in constant time.
    pub fn verify_digest(&self) -> bool {
        ct_eq_digest(&self.transcript_digest, &self.compute_digest())
    }

    /// Encode this transcript into canonical little-endian binary wire format (`OPAI` v1).
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        let mut w = WireWriter::new();
        w.write_bytes(&WIRE_MAGIC);
        w.write_u16(WIRE_VERSION);
        w.write_digest(&self.transcript_digest);
        let body = self.encode_body_bytes();
        w.write_bytes(&body);
        w.into_vec()
    }

    /// Decode a [`DisputeTranscript`] from canonical binary wire bytes with strict EOF checking.
    pub fn from_wire_bytes(bytes: &[u8]) -> Result<Self, OpAiError> {
        let mut r = WireReader::new(bytes);
        let magic = r.read_fixed::<4>()?;
        if magic != WIRE_MAGIC {
            return Err(OpAiError::WireFormat(format!(
                "invalid OPAI wire magic: {magic:?}"
            )));
        }
        let version = r.read_u16()?;
        if version != WIRE_VERSION {
            return Err(OpAiError::WireFormat(format!(
                "unsupported OPAI wire version: {version}"
            )));
        }
        let transcript_digest = r.read_digest()?;
        let body_start = r.pos;
        let mut t = Self::decode_body(&mut r)?;
        let body_end = r.pos;
        r.expect_eof()?;

        t.transcript_digest = transcript_digest;
        let expected = {
            let mut h = Sha256::with_domain(DOMAIN_TRANSCRIPT);
            h.update_len_prefixed(&bytes[body_start..body_end]);
            h.finalize()
        };
        if !ct_eq_digest(&transcript_digest, &expected) {
            return Err(OpAiError::TranscriptIntegrity(
                "wire transcript_digest mismatch".to_string(),
            ));
        }
        Ok(t)
    }

    /// Export this transcript as canonical, deterministic, self-contained JSON (zero external crates).
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push('{');
        push_kv_str(&mut out, "schema", "hanzo.opai.transcript.v1", false);
        push_kv_str(
            &mut out,
            "transcript_digest",
            &digest_to_hex(&self.transcript_digest),
            true,
        );
        push_kv_str(
            &mut out,
            "dispute_id",
            &digest_to_hex(&self.dispute_id),
            true,
        );

        // claim
        out.push_str(",\"claim\":{");
        push_kv_str(
            &mut out,
            "claim_id",
            &digest_to_hex(&self.claim.claim_id),
            false,
        );
        push_kv_str(&mut out, "provider_id", &self.claim.provider_id, true);
        push_kv_u64(&mut out, "nonce", self.claim.nonce, true);
        push_kv_str(
            &mut out,
            "profile_digest",
            &digest_to_hex(&self.claim.profile_digest),
            true,
        );
        push_kv_str(
            &mut out,
            "model_digest",
            &digest_to_hex(&self.claim.model_digest),
            true,
        );
        push_kv_str(
            &mut out,
            "input_hash",
            &digest_to_hex(&self.claim.input_hash),
            true,
        );
        push_kv_u64(&mut out, "total_steps", self.claim.total_steps, true);
        push_kv_str(
            &mut out,
            "start_root",
            &digest_to_hex(&self.claim.start_root),
            true,
        );
        push_kv_str(
            &mut out,
            "end_root",
            &digest_to_hex(&self.claim.end_root),
            true,
        );
        push_kv_str(
            &mut out,
            "trace_merkle_root",
            &digest_to_hex(&self.claim.trace_merkle_root),
            true,
        );
        out.push_str(",\"output_tokens\":");
        push_u32_array(&mut out, &self.claim.output_tokens);
        push_kv_u64(&mut out, "bond", self.claim.bond, true);
        push_kv_u64(
            &mut out,
            "submitted_at_secs",
            self.claim.submitted_at_secs,
            true,
        );
        out.push('}');

        push_kv_str(&mut out, "challenger_id", &self.challenger_id, true);
        push_kv_u64(&mut out, "challenger_bond", self.challenger_bond, true);
        push_kv_u64(&mut out, "challenge_slot", self.challenge_slot as u64, true);

        // config
        out.push_str(",\"config\":{");
        push_kv_u64(&mut out, "provider_bond", self.config.provider_bond, false);
        push_kv_u64(
            &mut out,
            "challenger_base_bond",
            self.config.challenger_base_bond,
            true,
        );
        push_kv_u64(
            &mut out,
            "griefing_escalation_bps",
            self.config.griefing_escalation_bps,
            true,
        );
        push_kv_u64(
            &mut out,
            "max_concurrent_challenges_per_claim",
            self.config.max_concurrent_challenges_per_claim as u64,
            true,
        );
        push_kv_u64(
            &mut out,
            "challenge_window_secs",
            self.config.challenge_window_secs,
            true,
        );
        push_kv_u64(
            &mut out,
            "turn_deadline_secs",
            self.config.turn_deadline_secs,
            true,
        );
        push_kv_u64(
            &mut out,
            "challenger_reward_bps",
            self.config.challenger_reward_bps,
            true,
        );
        push_kv_u64(
            &mut out,
            "provider_compensation_bps",
            self.config.provider_compensation_bps,
            true,
        );
        out.push('}');

        push_kv_u64(&mut out, "opened_at_secs", self.opened_at_secs, true);
        push_kv_u64(
            &mut out,
            "rounds_completed",
            u64::from(self.rounds_completed),
            true,
        );

        // moves
        out.push_str(",\"moves\":[");
        for (i, mv) in self.moves.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            encode_move_json(&mut out, mv);
        }
        out.push(']');

        // single_step_witness
        out.push_str(",\"single_step_witness\":");
        match &self.single_step_witness {
            None => out.push_str("null"),
            Some(st) => {
                out.push('{');
                push_kv_u64(&mut out, "step_index", st.step_index, false);
                push_kv_u64(&mut out, "active_token", u64::from(st.active_token), true);
                out.push_str(",\"hidden\":");
                push_i32_array(&mut out, &st.hidden);
                out.push_str(",\"residual\":");
                push_i32_array(&mut out, &st.residual);
                out.push_str(",\"scratch_d\":");
                push_i32_array(&mut out, &st.scratch_d);
                out.push_str(",\"scratch_f\":");
                push_i32_array(&mut out, &st.scratch_f);
                out.push_str(",\"logits\":");
                push_i32_array(&mut out, &st.logits);
                out.push_str(",\"output_tokens\":");
                push_u32_array(&mut out, &st.output_tokens);
                out.push('}');
            }
        }

        // resolution
        out.push_str(",\"resolution\":{");
        push_kv_u64(
            &mut out,
            "winner",
            u64::from(self.resolution.winner.to_tag()),
            false,
        );
        out.push_str(",\"pinpointed_step\":");
        match self.resolution.pinpointed_step {
            None => out.push_str("null"),
            Some(step) => out.push_str(&step.to_string()),
        }
        push_kv_u64(
            &mut out,
            "reason",
            u64::from(self.resolution.reason.to_tag()),
            true,
        );
        push_kv_u64(
            &mut out,
            "resolved_at_secs",
            self.resolution.resolved_at_secs,
            true,
        );
        out.push_str(",\"settlement\":{");
        push_kv_u64(
            &mut out,
            "provider_payout",
            self.resolution.settlement.provider_payout,
            false,
        );
        push_kv_u64(
            &mut out,
            "provider_slashed",
            self.resolution.settlement.provider_slashed,
            true,
        );
        push_kv_u64(
            &mut out,
            "challenger_payout",
            self.resolution.settlement.challenger_payout,
            true,
        );
        push_kv_u64(
            &mut out,
            "challenger_slashed",
            self.resolution.settlement.challenger_slashed,
            true,
        );
        push_kv_u64(&mut out, "burned", self.resolution.settlement.burned, true);
        out.push_str("}}");

        out.push('}');
        out
    }

    /// Parse a sovereign [`DisputeTranscript`] from JSON and verify its canonical digest.
    pub fn from_json_str(json: &str) -> Result<Self, OpAiError> {
        let root = parse_json(json)?;
        let obj = root.as_object()?;
        let schema = req_str(obj, "schema")?;
        if schema != "hanzo.opai.transcript.v1" {
            return Err(OpAiError::WireFormat(format!(
                "unsupported JSON transcript schema: {schema}"
            )));
        }
        let transcript_digest = hex_to_digest(req_str(obj, "transcript_digest")?)?;
        let dispute_id = hex_to_digest(req_str(obj, "dispute_id")?)?;

        let c_obj = req_obj(obj, "claim")?;
        let claim = Claim {
            claim_id: hex_to_digest(req_str(c_obj, "claim_id")?)?,
            provider_id: req_str(c_obj, "provider_id")?.to_string(),
            nonce: req_u64(c_obj, "nonce")?,
            profile_digest: hex_to_digest(req_str(c_obj, "profile_digest")?)?,
            model_digest: hex_to_digest(req_str(c_obj, "model_digest")?)?,
            input_hash: hex_to_digest(req_str(c_obj, "input_hash")?)?,
            total_steps: req_u64(c_obj, "total_steps")?,
            start_root: hex_to_digest(req_str(c_obj, "start_root")?)?,
            end_root: hex_to_digest(req_str(c_obj, "end_root")?)?,
            trace_merkle_root: hex_to_digest(req_str(c_obj, "trace_merkle_root")?)?,
            output_tokens: req_u32_vec(c_obj, "output_tokens")?,
            bond: req_u64(c_obj, "bond")?,
            submitted_at_secs: req_u64(c_obj, "submitted_at_secs")?,
        };

        let challenger_id = req_str(obj, "challenger_id")?.to_string();
        let challenger_bond = req_u64(obj, "challenger_bond")?;
        let challenge_slot = req_u64(obj, "challenge_slot")? as usize;

        let cfg_obj = req_obj(obj, "config")?;
        let config = GameConfig {
            provider_bond: req_u64(cfg_obj, "provider_bond")?,
            challenger_base_bond: req_u64(cfg_obj, "challenger_base_bond")?,
            griefing_escalation_bps: req_u64(cfg_obj, "griefing_escalation_bps")?,
            max_concurrent_challenges_per_claim: req_u64(
                cfg_obj,
                "max_concurrent_challenges_per_claim",
            )? as usize,
            challenge_window_secs: req_u64(cfg_obj, "challenge_window_secs")?,
            turn_deadline_secs: req_u64(cfg_obj, "turn_deadline_secs")?,
            challenger_reward_bps: req_u64(cfg_obj, "challenger_reward_bps")?,
            provider_compensation_bps: req_u64(cfg_obj, "provider_compensation_bps")?,
        };

        let opened_at_secs = req_u64(obj, "opened_at_secs")?;
        let rounds_completed = u32::try_from(req_u64(obj, "rounds_completed")?)
            .map_err(|_| OpAiError::WireFormat("rounds_completed exceeds u32".to_string()))?;

        let moves_arr = req_arr(obj, "moves")?;
        let mut moves = Vec::with_capacity(moves_arr.len());
        for item in moves_arr {
            moves.push(decode_move_json(item.as_object()?)?);
        }

        let single_step_witness = match obj.get("single_step_witness") {
            Some(JsonValue::Null) => None,
            Some(JsonValue::Object(st)) => Some(VmState {
                step_index: req_u64(st, "step_index")?,
                active_token: u32::try_from(req_u64(st, "active_token")?)
                    .map_err(|_| OpAiError::WireFormat("active_token exceeds u32".to_string()))?,
                hidden: req_i32_vec(st, "hidden")?,
                residual: req_i32_vec(st, "residual")?,
                scratch_d: req_i32_vec(st, "scratch_d")?,
                scratch_f: req_i32_vec(st, "scratch_f")?,
                logits: req_i32_vec(st, "logits")?,
                output_tokens: req_u32_vec(st, "output_tokens")?,
            }),
            _ => {
                return Err(OpAiError::WireFormat(
                    "missing or invalid single_step_witness".to_string(),
                ))
            }
        };

        let r_obj = req_obj(obj, "resolution")?;
        let winner = Party::from_tag(
            u8::try_from(req_u64(r_obj, "winner")?)
                .map_err(|_| OpAiError::WireFormat("winner tag exceeds u8".to_string()))?,
        )?;
        let pinpointed_step = match r_obj.get("pinpointed_step") {
            Some(JsonValue::Null) => None,
            Some(JsonValue::Int(v)) if *v >= 0 => Some(*v as u64),
            _ => return Err(OpAiError::WireFormat("invalid pinpointed_step".to_string())),
        };
        let reason = ResolutionReason::from_tag(
            u8::try_from(req_u64(r_obj, "reason")?)
                .map_err(|_| OpAiError::WireFormat("reason tag exceeds u8".to_string()))?,
        )?;
        let resolved_at_secs = req_u64(r_obj, "resolved_at_secs")?;
        let s_obj = req_obj(r_obj, "settlement")?;
        let settlement = BondSettlement {
            provider_payout: req_u64(s_obj, "provider_payout")?,
            provider_slashed: req_u64(s_obj, "provider_slashed")?,
            challenger_payout: req_u64(s_obj, "challenger_payout")?,
            challenger_slashed: req_u64(s_obj, "challenger_slashed")?,
            burned: req_u64(s_obj, "burned")?,
        };

        let transcript = Self {
            transcript_digest,
            dispute_id,
            claim,
            challenger_id,
            challenger_bond,
            challenge_slot,
            config,
            opened_at_secs,
            rounds_completed,
            moves,
            single_step_witness,
            resolution: DisputeResolution {
                winner,
                pinpointed_step,
                reason,
                resolved_at_secs,
                settlement,
            },
        };
        if !transcript.verify_digest() {
            return Err(OpAiError::TranscriptIntegrity(
                "JSON transcript_digest mismatch".to_string(),
            ));
        }
        Ok(transcript)
    }

    fn encode_body_bytes(&self) -> Vec<u8> {
        let mut w = WireWriter::new();
        w.write_digest(&self.dispute_id);

        // Claim
        w.write_digest(&self.claim.claim_id);
        w.write_str(&self.claim.provider_id);
        w.write_u64(self.claim.nonce);
        w.write_digest(&self.claim.profile_digest);
        w.write_digest(&self.claim.model_digest);
        w.write_digest(&self.claim.input_hash);
        w.write_u64(self.claim.total_steps);
        w.write_digest(&self.claim.start_root);
        w.write_digest(&self.claim.end_root);
        w.write_digest(&self.claim.trace_merkle_root);
        w.write_u32_slice(&self.claim.output_tokens);
        w.write_u64(self.claim.bond);
        w.write_u64(self.claim.submitted_at_secs);

        // Challenger & config
        w.write_str(&self.challenger_id);
        w.write_u64(self.challenger_bond);
        w.write_u64(self.challenge_slot as u64);
        w.write_u64(self.config.provider_bond);
        w.write_u64(self.config.challenger_base_bond);
        w.write_u64(self.config.griefing_escalation_bps);
        w.write_u64(self.config.max_concurrent_challenges_per_claim as u64);
        w.write_u64(self.config.challenge_window_secs);
        w.write_u64(self.config.turn_deadline_secs);
        w.write_u64(self.config.challenger_reward_bps);
        w.write_u64(self.config.provider_compensation_bps);

        w.write_u64(self.opened_at_secs);
        w.write_u32(self.rounds_completed);

        // Moves
        w.write_u32(self.moves.len() as u32);
        for mv in &self.moves {
            match mv {
                DisputeMove::Opened {
                    challenger_id,
                    challenger_bond,
                    at_secs,
                } => {
                    w.write_u8(0);
                    w.write_str(challenger_id);
                    w.write_u64(*challenger_bond);
                    w.write_u64(*at_secs);
                }
                DisputeMove::StartRootChecked {
                    expected_start_root,
                    claimed_start_root,
                    at_secs,
                } => {
                    w.write_u8(1);
                    w.write_digest(expected_start_root);
                    w.write_digest(claimed_start_root);
                    w.write_u64(*at_secs);
                }
                DisputeMove::DefenderMidpoint {
                    round,
                    mid,
                    mid_root,
                    at_secs,
                } => {
                    w.write_u8(2);
                    w.write_u32(*round);
                    w.write_u64(*mid);
                    w.write_digest(mid_root);
                    w.write_u64(*at_secs);
                }
                DisputeMove::ChallengerBranch {
                    round,
                    choice,
                    next_lo,
                    next_hi,
                    at_secs,
                } => {
                    w.write_u8(3);
                    w.write_u32(*round);
                    w.write_u8(choice.to_tag());
                    w.write_u64(*next_lo);
                    w.write_u64(*next_hi);
                    w.write_u64(*at_secs);
                }
                DisputeMove::SingleStepJudged {
                    step_index,
                    op_tag,
                    pre_root_valid,
                    actual_post_root,
                    claimed_post_root,
                    defender_valid,
                    at_secs,
                } => {
                    w.write_u8(4);
                    w.write_u64(*step_index);
                    w.write_str(op_tag);
                    w.write_u8(u8::from(*pre_root_valid));
                    w.write_digest(actual_post_root);
                    w.write_digest(claimed_post_root);
                    w.write_u8(u8::from(*defender_valid));
                    w.write_u64(*at_secs);
                }
                DisputeMove::TimeoutForfeit {
                    timed_out_party,
                    deadline_secs,
                    observed_at_secs,
                } => {
                    w.write_u8(5);
                    w.write_u8(timed_out_party.to_tag());
                    w.write_u64(*deadline_secs);
                    w.write_u64(*observed_at_secs);
                }
                DisputeMove::ConcurrentRefund { at_secs } => {
                    w.write_u8(6);
                    w.write_u64(*at_secs);
                }
            }
        }

        // Single-step witness
        match &self.single_step_witness {
            None => w.write_u8(0),
            Some(st) => {
                w.write_u8(1);
                w.write_u64(st.step_index);
                w.write_u32(st.active_token);
                w.write_i32_slice(&st.hidden);
                w.write_i32_slice(&st.residual);
                w.write_i32_slice(&st.scratch_d);
                w.write_i32_slice(&st.scratch_f);
                w.write_i32_slice(&st.logits);
                w.write_u32_slice(&st.output_tokens);
            }
        }

        // Resolution
        w.write_u8(self.resolution.winner.to_tag());
        match self.resolution.pinpointed_step {
            None => w.write_u8(0),
            Some(step) => {
                w.write_u8(1);
                w.write_u64(step);
            }
        }
        w.write_u8(self.resolution.reason.to_tag());
        w.write_u64(self.resolution.resolved_at_secs);
        w.write_u64(self.resolution.settlement.provider_payout);
        w.write_u64(self.resolution.settlement.provider_slashed);
        w.write_u64(self.resolution.settlement.challenger_payout);
        w.write_u64(self.resolution.settlement.challenger_slashed);
        w.write_u64(self.resolution.settlement.burned);

        w.into_vec()
    }

    fn decode_body(r: &mut WireReader<'_>) -> Result<Self, OpAiError> {
        let dispute_id = r.read_digest()?;
        let claim = Claim {
            claim_id: r.read_digest()?,
            provider_id: r.read_string()?,
            nonce: r.read_u64()?,
            profile_digest: r.read_digest()?,
            model_digest: r.read_digest()?,
            input_hash: r.read_digest()?,
            total_steps: r.read_u64()?,
            start_root: r.read_digest()?,
            end_root: r.read_digest()?,
            trace_merkle_root: r.read_digest()?,
            output_tokens: r.read_u32_vec()?,
            bond: r.read_u64()?,
            submitted_at_secs: r.read_u64()?,
        };
        let challenger_id = r.read_string()?;
        let challenger_bond = r.read_u64()?;
        let challenge_slot = r.read_u64()? as usize;
        let config = GameConfig {
            provider_bond: r.read_u64()?,
            challenger_base_bond: r.read_u64()?,
            griefing_escalation_bps: r.read_u64()?,
            max_concurrent_challenges_per_claim: r.read_u64()? as usize,
            challenge_window_secs: r.read_u64()?,
            turn_deadline_secs: r.read_u64()?,
            challenger_reward_bps: r.read_u64()?,
            provider_compensation_bps: r.read_u64()?,
        };
        let opened_at_secs = r.read_u64()?;
        let rounds_completed = r.read_u32()?;

        let move_count = r.read_u32()? as usize;
        if move_count > MAX_WIRE_VEC_LEN {
            return Err(OpAiError::WireFormat(
                "move_count exceeds wire safety limit".to_string(),
            ));
        }
        let mut moves = Vec::with_capacity(move_count);
        for _ in 0..move_count {
            let tag = r.read_u8()?;
            let mv = match tag {
                0 => DisputeMove::Opened {
                    challenger_id: r.read_string()?,
                    challenger_bond: r.read_u64()?,
                    at_secs: r.read_u64()?,
                },
                1 => DisputeMove::StartRootChecked {
                    expected_start_root: r.read_digest()?,
                    claimed_start_root: r.read_digest()?,
                    at_secs: r.read_u64()?,
                },
                2 => DisputeMove::DefenderMidpoint {
                    round: r.read_u32()?,
                    mid: r.read_u64()?,
                    mid_root: r.read_digest()?,
                    at_secs: r.read_u64()?,
                },
                3 => DisputeMove::ChallengerBranch {
                    round: r.read_u32()?,
                    choice: BranchChoice::from_tag(r.read_u8()?)?,
                    next_lo: r.read_u64()?,
                    next_hi: r.read_u64()?,
                    at_secs: r.read_u64()?,
                },
                4 => DisputeMove::SingleStepJudged {
                    step_index: r.read_u64()?,
                    op_tag: r.read_string()?,
                    pre_root_valid: r.read_bool()?,
                    actual_post_root: r.read_digest()?,
                    claimed_post_root: r.read_digest()?,
                    defender_valid: r.read_bool()?,
                    at_secs: r.read_u64()?,
                },
                5 => DisputeMove::TimeoutForfeit {
                    timed_out_party: Party::from_tag(r.read_u8()?)?,
                    deadline_secs: r.read_u64()?,
                    observed_at_secs: r.read_u64()?,
                },
                6 => DisputeMove::ConcurrentRefund {
                    at_secs: r.read_u64()?,
                },
                other => {
                    return Err(OpAiError::WireFormat(format!(
                        "unknown DisputeMove tag: {other}"
                    )))
                }
            };
            moves.push(mv);
        }

        let has_witness = r.read_bool()?;
        let single_step_witness = if has_witness {
            Some(VmState {
                step_index: r.read_u64()?,
                active_token: r.read_u32()?,
                hidden: r.read_i32_vec()?,
                residual: r.read_i32_vec()?,
                scratch_d: r.read_i32_vec()?,
                scratch_f: r.read_i32_vec()?,
                logits: r.read_i32_vec()?,
                output_tokens: r.read_u32_vec()?,
            })
        } else {
            None
        };

        let winner = Party::from_tag(r.read_u8()?)?;
        let has_step = r.read_bool()?;
        let pinpointed_step = if has_step { Some(r.read_u64()?) } else { None };
        let reason = ResolutionReason::from_tag(r.read_u8()?)?;
        let resolved_at_secs = r.read_u64()?;
        let settlement = BondSettlement {
            provider_payout: r.read_u64()?,
            provider_slashed: r.read_u64()?,
            challenger_payout: r.read_u64()?,
            challenger_slashed: r.read_u64()?,
            burned: r.read_u64()?,
        };

        Ok(Self {
            transcript_digest: [0u8; 32],
            dispute_id,
            claim,
            challenger_id,
            challenger_bond,
            challenge_slot,
            config,
            opened_at_secs,
            rounds_completed,
            moves,
            single_step_witness,
            resolution: DisputeResolution {
                winner,
                pinpointed_step,
                reason,
                resolved_at_secs,
                settlement,
            },
        })
    }
}

/// Summary returned by [`verify_transcript_sovereign`] upon successful independent replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SovereignReplayReport {
    /// Verified dispute identifier.
    pub dispute_id: Digest,
    /// Verified rounds replayed.
    pub rounds_replayed: u32,
    /// Whether the single-step deterministic ML judge was independently executed.
    pub single_step_judge_executed: bool,
    /// Verified winning party.
    pub winner: Party,
    /// Verified pinpointed step (if applicable).
    pub pinpointed_step: Option<u64>,
    /// Verified bond conservation check (`true`).
    pub bond_conservation_valid: bool,
    /// Always `0` — confirms zero Hanzo control-plane calls were required.
    pub control_plane_calls: u32,
}

/// Independently replay and cryptographically verify a [`DisputeTranscript`]
/// with **0 Hanzo control-plane calls**.
///
/// Reconstructs the entire bisection state machine from scratch, enforces all
/// deadline and interval invariants, re-runs [`crate::vm::verify_single_step`]
/// (or initial state verification) locally, and confirms bit-identical resolution
/// and bond conservation. Any tampering rejects deterministically.
pub fn verify_transcript_sovereign(
    transcript: &DisputeTranscript,
    profile: &ExecutionProfile,
    model: &QuantizedModel,
    input: &InputCommitment,
) -> Result<SovereignReplayReport, OpAiError> {
    if !transcript.verify_digest() {
        return Err(OpAiError::TranscriptIntegrity(
            "transcript_digest does not match body".to_string(),
        ));
    }
    profile.validate()?;
    model.validate()?;
    transcript.config.validate()?;

    if !transcript.claim.verify_id() {
        return Err(OpAiError::TranscriptIntegrity(
            "claim_id does not match claim fields".to_string(),
        ));
    }
    let profile_digest = profile.digest();
    let model_digest = model.digest();
    let input_hash = input.input_hash();
    if !ct_eq_digest(&transcript.claim.profile_digest, &profile_digest) {
        return Err(OpAiError::ReplayMismatch(
            "claim profile_digest does not match supplied ExecutionProfile".to_string(),
        ));
    }
    if !ct_eq_digest(&transcript.claim.model_digest, &model_digest) {
        return Err(OpAiError::ReplayMismatch(
            "claim model_digest does not match supplied QuantizedModel".to_string(),
        ));
    }
    if !ct_eq_digest(&transcript.claim.input_hash, &input_hash) {
        return Err(OpAiError::ReplayMismatch(
            "claim input_hash does not match supplied InputCommitment".to_string(),
        ));
    }

    if transcript.moves.is_empty() {
        return Err(OpAiError::TranscriptIntegrity(
            "transcript has zero moves".to_string(),
        ));
    }
    match &transcript.moves[0] {
        DisputeMove::Opened {
            challenger_id,
            challenger_bond,
            at_secs,
        } => {
            if challenger_id != &transcript.challenger_id
                || *challenger_bond != transcript.challenger_bond
                || *at_secs != transcript.opened_at_secs
            {
                return Err(OpAiError::ReplayMismatch(
                    "Opened move does not match transcript header".to_string(),
                ));
            }
        }
        _ => {
            return Err(OpAiError::TranscriptIntegrity(
                "first transcript move must be DisputeMove::Opened".to_string(),
            ))
        }
    }

    let mut game = DisputeGame::open(
        transcript.claim.clone(),
        transcript.challenger_id.clone(),
        transcript.challenger_bond,
        transcript.config.clone(),
        transcript.opened_at_secs,
        transcript.challenge_slot,
    )?;

    if !ct_eq_digest(&game.dispute_id, &transcript.dispute_id) {
        return Err(OpAiError::ReplayMismatch(
            "recomputed dispute_id does not match transcript".to_string(),
        ));
    }

    let mut single_step_executed = false;
    for mv in &transcript.moves[1..] {
        if game.resolution().is_some() {
            return Err(OpAiError::TranscriptIntegrity(
                "extra move present after dispute already resolved".to_string(),
            ));
        }
        match mv {
            DisputeMove::Opened { .. } => {
                return Err(OpAiError::TranscriptIntegrity(
                    "duplicate Opened move in transcript".to_string(),
                ));
            }
            DisputeMove::StartRootChecked {
                expected_start_root,
                claimed_start_root,
                at_secs,
            } => {
                game.assert_invalid_start_root(profile, model, input, *at_secs)?;
                match game.moves.last() {
                    Some(DisputeMove::StartRootChecked {
                        expected_start_root: exp,
                        claimed_start_root: clm,
                        ..
                    }) if ct_eq_digest(exp, expected_start_root)
                        && ct_eq_digest(clm, claimed_start_root) => {}
                    _ => {
                        return Err(OpAiError::ReplayMismatch(
                            "StartRootChecked move mismatch during replay".to_string(),
                        ))
                    }
                }
            }
            DisputeMove::DefenderMidpoint {
                round,
                mid,
                mid_root,
                at_secs,
            } => {
                if *round != game.rounds_completed {
                    return Err(OpAiError::ReplayMismatch(format!(
                        "DefenderMidpoint round mismatch: transcript={round}, expected={}",
                        game.rounds_completed
                    )));
                }
                game.submit_defender_midpoint(*mid, *mid_root, *at_secs)?;
            }
            DisputeMove::ChallengerBranch {
                round,
                choice,
                next_lo,
                next_hi,
                at_secs,
            } => {
                game.submit_challenger_branch(*choice, *at_secs)?;
                if *round != game.rounds_completed
                    || *next_lo != game.range.lo
                    || *next_hi != game.range.hi
                {
                    return Err(OpAiError::ReplayMismatch(
                        "ChallengerBranch resulting range or round mismatch".to_string(),
                    ));
                }
            }
            DisputeMove::SingleStepJudged {
                step_index,
                op_tag,
                pre_root_valid,
                actual_post_root,
                claimed_post_root,
                defender_valid,
                at_secs,
            } => {
                let witness = transcript.single_step_witness.clone().ok_or_else(|| {
                    OpAiError::TranscriptIntegrity(
                        "missing single_step_witness for SingleStepJudged move".to_string(),
                    )
                })?;
                game.submit_single_step_proof(profile, model, input, witness, *at_secs)?;
                single_step_executed = true;
                let verdict = game.single_step_verdict.as_ref().ok_or_else(|| {
                    OpAiError::ReplayMismatch("missing replayed single_step_verdict".to_string())
                })?;
                if verdict.step_index != *step_index
                    || verdict.op_tag != *op_tag
                    || verdict.pre_root_valid != *pre_root_valid
                    || !ct_eq_digest(&verdict.actual_post_root, actual_post_root)
                    || !ct_eq_digest(&verdict.claimed_post_root, claimed_post_root)
                    || verdict.defender_valid != *defender_valid
                {
                    return Err(OpAiError::ReplayMismatch(
                        "replayed SingleStepVerdict disagreed with transcript".to_string(),
                    ));
                }
            }
            DisputeMove::TimeoutForfeit {
                timed_out_party,
                deadline_secs,
                observed_at_secs,
            } => {
                let expected_deadline = game.turn_deadline_secs();
                if *deadline_secs != expected_deadline {
                    return Err(OpAiError::ReplayMismatch(format!(
                        "TimeoutForfeit deadline mismatch: transcript={deadline_secs}, expected={expected_deadline}"
                    )));
                }
                game.claim_timeout_forfeit(*observed_at_secs)?;
                match game.moves.last() {
                    Some(DisputeMove::TimeoutForfeit {
                        timed_out_party: p, ..
                    }) if p == timed_out_party => {}
                    _ => {
                        return Err(OpAiError::ReplayMismatch(
                            "TimeoutForfeit party mismatch during replay".to_string(),
                        ))
                    }
                }
            }
            DisputeMove::ConcurrentRefund { at_secs } => {
                if *at_secs < game.last_action_at_secs {
                    return Err(OpAiError::DeadlineViolation(
                        "ConcurrentRefund timestamp precedes last action".to_string(),
                    ));
                }
                game.mark_refunded_claim_already_slashed(*at_secs);
            }
        }
    }

    if game.rounds_completed != transcript.rounds_completed {
        return Err(OpAiError::ReplayMismatch(format!(
            "rounds_completed mismatch: transcript={}, replayed={}",
            transcript.rounds_completed, game.rounds_completed
        )));
    }
    let replayed_res = game.resolution().ok_or_else(|| {
        OpAiError::ReplayMismatch("transcript ended without resolving the dispute".to_string())
    })?;
    if replayed_res != &transcript.resolution {
        return Err(OpAiError::ReplayMismatch(
            "replayed DisputeResolution disagreed with transcript".to_string(),
        ));
    }

    let bond_conservation_valid = match replayed_res.reason {
        ResolutionReason::RefundedClaimAlreadySlashed => replayed_res
            .settlement
            .verify_conservation(0, transcript.challenger_bond),
        _ => replayed_res
            .settlement
            .verify_conservation(transcript.claim.bond, transcript.challenger_bond),
    };
    if !bond_conservation_valid {
        return Err(OpAiError::ReplayMismatch(
            "bond settlement violated conservation of funds".to_string(),
        ));
    }

    Ok(SovereignReplayReport {
        dispute_id: game.dispute_id,
        rounds_replayed: game.rounds_completed,
        single_step_judge_executed: single_step_executed,
        winner: replayed_res.winner,
        pinpointed_step: replayed_res.pinpointed_step,
        bond_conservation_valid,
        control_plane_calls: 0,
    })
}

// ---------- Canonical Binary Wire Writer / Reader ----------

struct WireWriter {
    buf: Vec<u8>,
}

impl WireWriter {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(512),
        }
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn write_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_digest(&mut self, d: &Digest) {
        self.buf.extend_from_slice(d);
    }

    fn write_str(&mut self, s: &str) {
        self.write_u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }

    fn write_u32_slice(&mut self, slice: &[u32]) {
        self.write_u32(slice.len() as u32);
        for &v in slice {
            self.write_u32(v);
        }
    }

    fn write_i32_slice(&mut self, slice: &[i32]) {
        self.write_u32(slice.len() as u32);
        for &v in slice {
            self.write_i32(v);
        }
    }

    fn into_vec(self) -> Vec<u8> {
        self.buf
    }
}

struct WireReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> WireReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read_fixed<const N: usize>(&mut self) -> Result<[u8; N], OpAiError> {
        let end = self
            .pos
            .checked_add(N)
            .ok_or_else(|| OpAiError::WireFormat("reader offset overflow".to_string()))?;
        if end > self.buf.len() {
            return Err(OpAiError::WireFormat(format!(
                "unexpected EOF: needed {N} bytes at offset {}, len={}",
                self.pos,
                self.buf.len()
            )));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.buf[self.pos..end]);
        self.pos = end;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8, OpAiError> {
        Ok(self.read_fixed::<1>()?[0])
    }

    fn read_bool(&mut self) -> Result<bool, OpAiError> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(OpAiError::WireFormat(format!("invalid bool byte: {other}"))),
        }
    }

    fn read_u16(&mut self) -> Result<u16, OpAiError> {
        Ok(u16::from_le_bytes(self.read_fixed::<2>()?))
    }

    fn read_u32(&mut self) -> Result<u32, OpAiError> {
        Ok(u32::from_le_bytes(self.read_fixed::<4>()?))
    }

    fn read_i32(&mut self) -> Result<i32, OpAiError> {
        Ok(i32::from_le_bytes(self.read_fixed::<4>()?))
    }

    fn read_u64(&mut self) -> Result<u64, OpAiError> {
        Ok(u64::from_le_bytes(self.read_fixed::<8>()?))
    }

    fn read_digest(&mut self) -> Result<Digest, OpAiError> {
        self.read_fixed::<32>()
    }

    fn read_string(&mut self) -> Result<String, OpAiError> {
        let len = self.read_u32()? as usize;
        if len > MAX_WIRE_VEC_LEN {
            return Err(OpAiError::WireFormat(
                "string length exceeds wire safety limit".to_string(),
            ));
        }
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| OpAiError::WireFormat("string offset overflow".to_string()))?;
        if end > self.buf.len() {
            return Err(OpAiError::WireFormat(
                "unexpected EOF in string".to_string(),
            ));
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        String::from_utf8(slice.to_vec())
            .map_err(|_| OpAiError::WireFormat("invalid UTF-8 string in wire".to_string()))
    }

    fn read_u32_vec(&mut self) -> Result<Vec<u32>, OpAiError> {
        let len = self.read_u32()? as usize;
        if len > MAX_WIRE_VEC_LEN {
            return Err(OpAiError::WireFormat(
                "u32 vec length exceeds wire safety limit".to_string(),
            ));
        }
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_u32()?);
        }
        Ok(out)
    }

    fn read_i32_vec(&mut self) -> Result<Vec<i32>, OpAiError> {
        let len = self.read_u32()? as usize;
        if len > MAX_WIRE_VEC_LEN {
            return Err(OpAiError::WireFormat(
                "i32 vec length exceeds wire safety limit".to_string(),
            ));
        }
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_i32()?);
        }
        Ok(out)
    }

    fn expect_eof(&self) -> Result<(), OpAiError> {
        if self.pos != self.buf.len() {
            return Err(OpAiError::WireFormat(format!(
                "trailing {} byte(s) after wire payload",
                self.buf.len() - self.pos
            )));
        }
        Ok(())
    }
}

// ---------- Clean-Room JSON Serializer & Parser (Zero External Crates) ----------

fn push_escaped_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn push_kv_str(out: &mut String, key: &str, val: &str, comma: bool) {
    if comma {
        out.push(',');
    }
    push_escaped_str(out, key);
    out.push(':');
    push_escaped_str(out, val);
}

fn push_kv_u64(out: &mut String, key: &str, val: u64, comma: bool) {
    if comma {
        out.push(',');
    }
    push_escaped_str(out, key);
    out.push(':');
    out.push_str(&val.to_string());
}

fn push_kv_bool(out: &mut String, key: &str, val: bool, comma: bool) {
    if comma {
        out.push(',');
    }
    push_escaped_str(out, key);
    out.push(':');
    out.push_str(if val { "true" } else { "false" });
}

fn push_u32_array(out: &mut String, slice: &[u32]) {
    out.push('[');
    for (i, &v) in slice.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&v.to_string());
    }
    out.push(']');
}

fn push_i32_array(out: &mut String, slice: &[i32]) {
    out.push('[');
    for (i, &v) in slice.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&v.to_string());
    }
    out.push(']');
}

fn encode_move_json(out: &mut String, mv: &DisputeMove) {
    out.push('{');
    match mv {
        DisputeMove::Opened {
            challenger_id,
            challenger_bond,
            at_secs,
        } => {
            push_kv_str(out, "kind", "opened", false);
            push_kv_str(out, "challenger_id", challenger_id, true);
            push_kv_u64(out, "challenger_bond", *challenger_bond, true);
            push_kv_u64(out, "at_secs", *at_secs, true);
        }
        DisputeMove::StartRootChecked {
            expected_start_root,
            claimed_start_root,
            at_secs,
        } => {
            push_kv_str(out, "kind", "start_root_checked", false);
            push_kv_str(
                out,
                "expected_start_root",
                &digest_to_hex(expected_start_root),
                true,
            );
            push_kv_str(
                out,
                "claimed_start_root",
                &digest_to_hex(claimed_start_root),
                true,
            );
            push_kv_u64(out, "at_secs", *at_secs, true);
        }
        DisputeMove::DefenderMidpoint {
            round,
            mid,
            mid_root,
            at_secs,
        } => {
            push_kv_str(out, "kind", "defender_midpoint", false);
            push_kv_u64(out, "round", u64::from(*round), true);
            push_kv_u64(out, "mid", *mid, true);
            push_kv_str(out, "mid_root", &digest_to_hex(mid_root), true);
            push_kv_u64(out, "at_secs", *at_secs, true);
        }
        DisputeMove::ChallengerBranch {
            round,
            choice,
            next_lo,
            next_hi,
            at_secs,
        } => {
            push_kv_str(out, "kind", "challenger_branch", false);
            push_kv_u64(out, "round", u64::from(*round), true);
            push_kv_u64(out, "choice", u64::from(choice.to_tag()), true);
            push_kv_u64(out, "next_lo", *next_lo, true);
            push_kv_u64(out, "next_hi", *next_hi, true);
            push_kv_u64(out, "at_secs", *at_secs, true);
        }
        DisputeMove::SingleStepJudged {
            step_index,
            op_tag,
            pre_root_valid,
            actual_post_root,
            claimed_post_root,
            defender_valid,
            at_secs,
        } => {
            push_kv_str(out, "kind", "single_step_judged", false);
            push_kv_u64(out, "step_index", *step_index, true);
            push_kv_str(out, "op_tag", op_tag, true);
            push_kv_bool(out, "pre_root_valid", *pre_root_valid, true);
            push_kv_str(
                out,
                "actual_post_root",
                &digest_to_hex(actual_post_root),
                true,
            );
            push_kv_str(
                out,
                "claimed_post_root",
                &digest_to_hex(claimed_post_root),
                true,
            );
            push_kv_bool(out, "defender_valid", *defender_valid, true);
            push_kv_u64(out, "at_secs", *at_secs, true);
        }
        DisputeMove::TimeoutForfeit {
            timed_out_party,
            deadline_secs,
            observed_at_secs,
        } => {
            push_kv_str(out, "kind", "timeout_forfeit", false);
            push_kv_u64(
                out,
                "timed_out_party",
                u64::from(timed_out_party.to_tag()),
                true,
            );
            push_kv_u64(out, "deadline_secs", *deadline_secs, true);
            push_kv_u64(out, "observed_at_secs", *observed_at_secs, true);
        }
        DisputeMove::ConcurrentRefund { at_secs } => {
            push_kv_str(out, "kind", "concurrent_refund", false);
            push_kv_u64(out, "at_secs", *at_secs, true);
        }
    }
    out.push('}');
}

fn decode_move_json(obj: &BTreeMap<String, JsonValue>) -> Result<DisputeMove, OpAiError> {
    match req_str(obj, "kind")? {
        "opened" => Ok(DisputeMove::Opened {
            challenger_id: req_str(obj, "challenger_id")?.to_string(),
            challenger_bond: req_u64(obj, "challenger_bond")?,
            at_secs: req_u64(obj, "at_secs")?,
        }),
        "start_root_checked" => Ok(DisputeMove::StartRootChecked {
            expected_start_root: hex_to_digest(req_str(obj, "expected_start_root")?)?,
            claimed_start_root: hex_to_digest(req_str(obj, "claimed_start_root")?)?,
            at_secs: req_u64(obj, "at_secs")?,
        }),
        "defender_midpoint" => Ok(DisputeMove::DefenderMidpoint {
            round: u32::try_from(req_u64(obj, "round")?)
                .map_err(|_| OpAiError::WireFormat("round exceeds u32".to_string()))?,
            mid: req_u64(obj, "mid")?,
            mid_root: hex_to_digest(req_str(obj, "mid_root")?)?,
            at_secs: req_u64(obj, "at_secs")?,
        }),
        "challenger_branch" => Ok(DisputeMove::ChallengerBranch {
            round: u32::try_from(req_u64(obj, "round")?)
                .map_err(|_| OpAiError::WireFormat("round exceeds u32".to_string()))?,
            choice: BranchChoice::from_tag(
                u8::try_from(req_u64(obj, "choice")?)
                    .map_err(|_| OpAiError::WireFormat("choice exceeds u8".to_string()))?,
            )?,
            next_lo: req_u64(obj, "next_lo")?,
            next_hi: req_u64(obj, "next_hi")?,
            at_secs: req_u64(obj, "at_secs")?,
        }),
        "single_step_judged" => Ok(DisputeMove::SingleStepJudged {
            step_index: req_u64(obj, "step_index")?,
            op_tag: req_str(obj, "op_tag")?.to_string(),
            pre_root_valid: req_bool(obj, "pre_root_valid")?,
            actual_post_root: hex_to_digest(req_str(obj, "actual_post_root")?)?,
            claimed_post_root: hex_to_digest(req_str(obj, "claimed_post_root")?)?,
            defender_valid: req_bool(obj, "defender_valid")?,
            at_secs: req_u64(obj, "at_secs")?,
        }),
        "timeout_forfeit" => Ok(DisputeMove::TimeoutForfeit {
            timed_out_party: Party::from_tag(
                u8::try_from(req_u64(obj, "timed_out_party")?)
                    .map_err(|_| OpAiError::WireFormat("timed_out_party exceeds u8".to_string()))?,
            )?,
            deadline_secs: req_u64(obj, "deadline_secs")?,
            observed_at_secs: req_u64(obj, "observed_at_secs")?,
        }),
        "concurrent_refund" => Ok(DisputeMove::ConcurrentRefund {
            at_secs: req_u64(obj, "at_secs")?,
        }),
        other => Err(OpAiError::WireFormat(format!(
            "unknown JSON move kind: {other}"
        ))),
    }
}

#[derive(Debug)]
enum JsonValue {
    Null,
    Bool(bool),
    Int(i128),
    Str(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    fn as_object(&self) -> Result<&BTreeMap<String, JsonValue>, OpAiError> {
        match self {
            Self::Object(m) => Ok(m),
            _ => Err(OpAiError::WireFormat("expected JSON object".to_string())),
        }
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

fn parse_json(input: &str) -> Result<JsonValue, OpAiError> {
    let mut p = JsonParser {
        bytes: input.as_bytes(),
        pos: 0,
        depth: 0,
    };
    let val = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(OpAiError::WireFormat(
            "trailing characters after JSON document".to_string(),
        ));
    }
    Ok(val)
}

impl<'a> JsonParser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len()
            && matches!(self.bytes[self.pos], b' ' | b'\n' | b'\r' | b'\t')
        {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<JsonValue, OpAiError> {
        if self.depth > 64 {
            return Err(OpAiError::WireFormat(
                "JSON nesting depth exceeded".to_string(),
            ));
        }
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(JsonValue::Str(self.parse_string()?)),
            Some(b't') => {
                self.expect_literal(b"true")?;
                Ok(JsonValue::Bool(true))
            }
            Some(b'f') => {
                self.expect_literal(b"false")?;
                Ok(JsonValue::Bool(false))
            }
            Some(b'n') => {
                self.expect_literal(b"null")?;
                Ok(JsonValue::Null)
            }
            Some(b'-' | b'0'..=b'9') => Ok(JsonValue::Int(self.parse_int()?)),
            _ => Err(OpAiError::WireFormat(format!(
                "unexpected JSON token at offset {}",
                self.pos
            ))),
        }
    }

    fn expect_literal(&mut self, lit: &[u8]) -> Result<(), OpAiError> {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(())
        } else {
            Err(OpAiError::WireFormat(format!(
                "invalid JSON literal at offset {}",
                self.pos
            )))
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, OpAiError> {
        self.depth += 1;
        self.pos += 1; // '{'
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(OpAiError::WireFormat(
                    "expected ':' in JSON object".to_string(),
                ));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(JsonValue::Object(map));
                }
                _ => {
                    return Err(OpAiError::WireFormat(
                        "expected ',' or '}' in JSON object".to_string(),
                    ))
                }
            }
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue, OpAiError> {
        self.depth += 1;
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => {
                    return Err(OpAiError::WireFormat(
                        "expected ',' or ']' in JSON array".to_string(),
                    ))
                }
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, OpAiError> {
        if self.peek() != Some(b'"') {
            return Err(OpAiError::WireFormat(
                "expected '\"' for JSON string".to_string(),
            ));
        }
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    if self.pos >= self.bytes.len() {
                        return Err(OpAiError::WireFormat(
                            "unterminated JSON escape".to_string(),
                        ));
                    }
                    let esc = self.bytes[self.pos];
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            if self.pos + 4 > self.bytes.len() {
                                return Err(OpAiError::WireFormat(
                                    "short \\uXXXX escape".to_string(),
                                ));
                            }
                            let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                                .map_err(|_| {
                                    OpAiError::WireFormat("invalid \\uXXXX".to_string())
                                })?;
                            let code = u32::from_str_radix(hex, 16).map_err(|_| {
                                OpAiError::WireFormat("invalid hex in \\uXXXX".to_string())
                            })?;
                            let ch = char::from_u32(code).ok_or_else(|| {
                                OpAiError::WireFormat("invalid unicode scalar".to_string())
                            })?;
                            out.push(ch);
                            self.pos += 4;
                        }
                        _ => {
                            return Err(OpAiError::WireFormat(format!(
                                "unsupported JSON escape \\{}",
                                esc as char
                            )))
                        }
                    }
                }
                _ => {
                    // Advance full UTF-8 character safely.
                    let start = self.pos - 1;
                    let rest = std::str::from_utf8(&self.bytes[start..])
                        .map_err(|_| OpAiError::WireFormat("invalid UTF-8 in JSON".to_string()))?;
                    let ch = rest
                        .chars()
                        .next()
                        .ok_or_else(|| OpAiError::WireFormat("EOF in JSON string".to_string()))?;
                    out.push(ch);
                    self.pos = start + ch.len_utf8();
                }
            }
        }
        Err(OpAiError::WireFormat(
            "unterminated JSON string".to_string(),
        ))
    }

    fn parse_int(&mut self) -> Result<i128, OpAiError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| OpAiError::WireFormat("invalid integer bytes".to_string()))?;
        s.parse::<i128>()
            .map_err(|_| OpAiError::WireFormat(format!("invalid integer '{s}'")))
    }
}

fn req_str<'a>(obj: &'a BTreeMap<String, JsonValue>, key: &str) -> Result<&'a str, OpAiError> {
    match obj.get(key) {
        Some(JsonValue::Str(s)) => Ok(s.as_str()),
        _ => Err(OpAiError::WireFormat(format!(
            "missing string field '{key}'"
        ))),
    }
}

fn req_u64(obj: &BTreeMap<String, JsonValue>, key: &str) -> Result<u64, OpAiError> {
    match obj.get(key) {
        Some(JsonValue::Int(v)) => u64::try_from(*v)
            .map_err(|_| OpAiError::WireFormat(format!("field '{key}' out of u64 range"))),
        _ => Err(OpAiError::WireFormat(format!(
            "missing integer field '{key}'"
        ))),
    }
}

fn req_bool(obj: &BTreeMap<String, JsonValue>, key: &str) -> Result<bool, OpAiError> {
    match obj.get(key) {
        Some(JsonValue::Bool(b)) => Ok(*b),
        _ => Err(OpAiError::WireFormat(format!("missing bool field '{key}'"))),
    }
}

fn req_obj<'a>(
    obj: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<&'a BTreeMap<String, JsonValue>, OpAiError> {
    match obj.get(key) {
        Some(JsonValue::Object(m)) => Ok(m),
        _ => Err(OpAiError::WireFormat(format!(
            "missing object field '{key}'"
        ))),
    }
}

fn req_arr<'a>(
    obj: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<&'a [JsonValue], OpAiError> {
    match obj.get(key) {
        Some(JsonValue::Array(a)) => Ok(a.as_slice()),
        _ => Err(OpAiError::WireFormat(format!(
            "missing array field '{key}'"
        ))),
    }
}

fn req_u32_vec(obj: &BTreeMap<String, JsonValue>, key: &str) -> Result<Vec<u32>, OpAiError> {
    let arr = req_arr(obj, key)?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        match item {
            JsonValue::Int(v) => out
                .push(u32::try_from(*v).map_err(|_| {
                    OpAiError::WireFormat(format!("element in '{key}' out of u32"))
                })?),
            _ => {
                return Err(OpAiError::WireFormat(format!(
                    "non-integer element in '{key}'"
                )))
            }
        }
    }
    Ok(out)
}

fn req_i32_vec(obj: &BTreeMap<String, JsonValue>, key: &str) -> Result<Vec<i32>, OpAiError> {
    let arr = req_arr(obj, key)?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        match item {
            JsonValue::Int(v) => out
                .push(i32::try_from(*v).map_err(|_| {
                    OpAiError::WireFormat(format!("element in '{key}' out of i32"))
                })?),
            _ => {
                return Err(OpAiError::WireFormat(format!(
                    "non-integer element in '{key}'"
                )))
            }
        }
    }
    Ok(out)
}
