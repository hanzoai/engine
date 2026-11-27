// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Lossless sovereign federation audit ledger (`FVER` v1 binary wire + JSON export)
//! and independent offline third-party verification (`0` Hanzo control-plane calls).

use crate::aggregator::RoundAuditRecord;
use crate::crypto::{ct_eq_bytes, ed25519_verify, hex_encode, merkle_root, Digest32, Sha256};
use crate::identity::{
    AcceptedRecord, ParticipantIdentity, RejectionReason, RejectionReceipt, PROTOCOL_VERSION,
};
use crate::model::CorpusPin;
use std::collections::BTreeMap;
use std::fmt;

/// Magic header for canonical `hanzo-fedverify` sovereign binary ledgers.
pub const WIRE_MAGIC: &[u8; 4] = b"FVER";

/// Protocol identifier embedded in every sovereign ledger.
pub const PROTOCOL_ID: &str = "hanzo.fedverify.dso.v1";

/// Anti-DoS bounds on binary wire decoding.
const MAX_WIRE_ITEMS: usize = 65_536;
const MAX_WIRE_STR_LEN: usize = 4_096;

/// Errors returned when decoding or independently verifying a sovereign audit ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    /// Wire magic or version mismatch.
    InvalidHeader,
    /// Unexpected end of wire input or trailing unconsumed bytes.
    InvalidLength(&'static str),
    /// Non-UTF-8 string or out-of-bounds enum tag in wire payload.
    MalformedEncoding(&'static str),
    /// Participant key fingerprint mismatch.
    InvalidParticipantFingerprint(String),
    /// Model hash chain broken at round `u64`.
    BrokenModelHashChain(u64),
    /// Round `accepted_merkle_root` or `rejection_merkle_root` mismatch.
    MerkleRootMismatch(u64),
    /// Round digest or auditor Ed25519 signature invalid.
    InvalidRoundSignature(u64),
    /// Individual `RejectionReceipt` digest or auditor Ed25519 signature invalid.
    InvalidRejectionReceipt {
        /// Round index where invalid receipt was found.
        round_id: u64,
        /// Blamed participant ID in the receipt.
        participant_id: String,
    },
    /// Top-level `ledger_digest` does not match recomputed digest.
    LedgerDigestMismatch,
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeader => write!(f, "invalid FVER wire header or version"),
            Self::InvalidLength(ctx) => write!(f, "invalid wire length: {ctx}"),
            Self::MalformedEncoding(ctx) => write!(f, "malformed wire encoding: {ctx}"),
            Self::InvalidParticipantFingerprint(id) => {
                write!(f, "invalid identity fingerprint for participant {id}")
            }
            Self::BrokenModelHashChain(r) => write!(f, "broken model hash chain at round {r}"),
            Self::MerkleRootMismatch(r) => write!(f, "merkle root mismatch at round {r}"),
            Self::InvalidRoundSignature(r) => {
                write!(f, "invalid round digest or auditor signature at round {r}")
            }
            Self::InvalidRejectionReceipt {
                round_id,
                participant_id,
            } => write!(
                f,
                "invalid rejection receipt signature at round {round_id} for {participant_id}"
            ),
            Self::LedgerDigestMismatch => write!(f, "top-level ledger digest mismatch"),
        }
    }
}

impl std::error::Error for LedgerError {}

/// Complete exportable audit ledger covering an entire federated training run.
#[derive(Debug, Clone, PartialEq)]
pub struct FederationAuditLedger {
    /// Protocol string (`PROTOCOL_ID`).
    pub protocol_id: String,
    /// Regime identifier (e.g., `"hanzo_fedverify_dso"`).
    pub regime_name: String,
    /// Cryptographic pins of all versioned repository corpus files used.
    pub corpus_pins: Vec<CorpusPin>,
    /// Registered participant public identities.
    pub participants: Vec<ParticipantIdentity>,
    /// Global model hash before round 0.
    pub initial_model_hash: Digest32,
    /// Global model hash after the final round.
    pub final_model_hash: Digest32,
    /// Ordered sequence of signed round audit records.
    pub rounds: Vec<RoundAuditRecord>,
    /// Canonical SHA-256 digest committing the entire ledger.
    pub ledger_digest: Digest32,
}

impl FederationAuditLedger {
    /// Constructs a finalized `FederationAuditLedger` and computes its canonical `ledger_digest`.
    #[must_use]
    pub fn build(
        regime_name: impl Into<String>,
        corpus_pins: Vec<CorpusPin>,
        participants: Vec<ParticipantIdentity>,
        initial_model_hash: Digest32,
        final_model_hash: Digest32,
        rounds: Vec<RoundAuditRecord>,
    ) -> Self {
        let mut ledger = Self {
            protocol_id: PROTOCOL_ID.to_string(),
            regime_name: regime_name.into(),
            corpus_pins,
            participants,
            initial_model_hash,
            final_model_hash,
            rounds,
            ledger_digest: [0u8; 32],
        };
        ledger.ledger_digest = ledger.compute_digest();
        ledger
    }

    /// Computes canonical SHA-256 digest over all ledger fields (excluding `ledger_digest`).
    #[must_use]
    pub fn compute_digest(&self) -> Digest32 {
        let mut h = Sha256::new();
        h.update(b"hanzo.fedverify.ledger.v1");
        h.update(&PROTOCOL_VERSION.to_le_bytes());
        h.update_len_prefixed(self.protocol_id.as_bytes());
        h.update_len_prefixed(self.regime_name.as_bytes());
        h.update(&(self.corpus_pins.len() as u64).to_le_bytes());
        for pin in &self.corpus_pins {
            h.update_len_prefixed(pin.relative_path.as_bytes());
            h.update(&pin.byte_len.to_le_bytes());
            h.update(&pin.sha256_digest);
        }
        h.update(&(self.participants.len() as u64).to_le_bytes());
        for p in &self.participants {
            h.update_len_prefixed(p.participant_id.as_bytes());
            h.update(&p.public_key);
            h.update(&p.key_fingerprint);
        }
        h.update(&self.initial_model_hash);
        h.update(&self.final_model_hash);
        h.update(&(self.rounds.len() as u64).to_le_bytes());
        for r in &self.rounds {
            h.update(&r.round_digest);
            h.update(&r.auditor_signature);
        }
        h.finalize()
    }

    /// Encodes the ledger into lossless canonical `FVER` v1 binary wire format.
    #[must_use]
    pub fn encode_wire(&self) -> Vec<u8> {
        let mut w = WireWriter::new();
        w.write_raw(WIRE_MAGIC);
        w.write_u32(PROTOCOL_VERSION);
        w.write_str(&self.protocol_id);
        w.write_str(&self.regime_name);

        w.write_u32(self.corpus_pins.len() as u32);
        for pin in &self.corpus_pins {
            w.write_str(&pin.relative_path);
            w.write_u64(pin.byte_len);
            w.write_raw(&pin.sha256_digest);
        }

        w.write_u32(self.participants.len() as u32);
        for p in &self.participants {
            w.write_str(&p.participant_id);
            w.write_raw(&p.public_key);
            w.write_raw(&p.key_fingerprint);
        }

        w.write_raw(&self.initial_model_hash);
        w.write_raw(&self.final_model_hash);

        w.write_u32(self.rounds.len() as u32);
        for r in &self.rounds {
            w.write_u64(r.round_id);
            w.write_raw(&r.base_model_hash);
            w.write_raw(&r.new_model_hash);
            w.write_f64(r.base_val_loss);
            w.write_f64(r.new_val_loss);
            w.write_u8(u8::from(r.update_committed));

            w.write_u32(r.accepted.len() as u32);
            for a in &r.accepted {
                w.write_str(&a.participant_id);
                w.write_raw(&a.participant_pubkey);
                w.write_raw(&a.envelope_digest);
                w.write_raw(&a.delta_hash);
                w.write_u64(a.sample_count);
                w.write_u64(a.tokens_processed);
                w.write_f64(a.observed_l2_norm);
                w.write_f64(a.observed_cosine);
                w.write_f64(a.observed_val_loss_delta);
                w.write_f64(a.aggregation_weight);
            }

            w.write_u32(r.rejections.len() as u32);
            for rej in &r.rejections {
                w.write_u64(rej.round_id);
                w.write_str(&rej.participant_id);
                w.write_raw(&rej.participant_pubkey);
                w.write_raw(&rej.envelope_digest);
                w.write_raw(&rej.base_model_hash);
                w.write_raw(&rej.delta_hash);
                w.write_u8(rej.reason as u8);
                w.write_f64(rej.observed_l2_norm);
                w.write_f64(rej.observed_cosine);
                w.write_f64(rej.observed_val_loss_delta);
                w.write_f64(rej.claimed_loss_delta);
                w.write_u32(rej.collusion_cluster_size);
                w.write_raw(&rej.auditor_pubkey);
                w.write_raw(&rej.receipt_digest);
                w.write_raw(&rej.auditor_signature);
            }

            w.write_raw(&r.accepted_merkle_root);
            w.write_raw(&r.rejection_merkle_root);
            w.write_raw(&r.round_digest);
            w.write_raw(&r.auditor_pubkey);
            w.write_raw(&r.auditor_signature);
        }

        w.write_raw(&self.ledger_digest);
        w.buf
    }

    /// Decodes a canonical `FVER` v1 binary wire buffer with strict bounds and EOF checks.
    pub fn decode_wire(bytes: &[u8]) -> Result<Self, LedgerError> {
        let mut r = WireReader::new(bytes);
        let magic = r.read_fixed::<4>()?;
        if &magic != WIRE_MAGIC {
            return Err(LedgerError::InvalidHeader);
        }
        let ver = r.read_u32()?;
        if ver != PROTOCOL_VERSION {
            return Err(LedgerError::InvalidHeader);
        }
        let protocol_id = r.read_str()?;
        let regime_name = r.read_str()?;

        let pin_count = r.read_bounded_count()?;
        let mut corpus_pins = Vec::with_capacity(pin_count);
        for _ in 0..pin_count {
            let relative_path = r.read_str()?;
            let byte_len = r.read_u64()?;
            let sha256_digest = r.read_fixed::<32>()?;
            corpus_pins.push(CorpusPin {
                relative_path,
                byte_len,
                sha256_digest,
            });
        }

        let part_count = r.read_bounded_count()?;
        let mut participants = Vec::with_capacity(part_count);
        for _ in 0..part_count {
            let participant_id = r.read_str()?;
            let public_key = r.read_fixed::<32>()?;
            let key_fingerprint = r.read_fixed::<32>()?;
            participants.push(ParticipantIdentity {
                participant_id,
                public_key,
                key_fingerprint,
            });
        }

        let initial_model_hash = r.read_fixed::<32>()?;
        let final_model_hash = r.read_fixed::<32>()?;

        let round_count = r.read_bounded_count()?;
        let mut rounds = Vec::with_capacity(round_count);
        for _ in 0..round_count {
            let round_id = r.read_u64()?;
            let base_model_hash = r.read_fixed::<32>()?;
            let new_model_hash = r.read_fixed::<32>()?;
            let base_val_loss = r.read_f64()?;
            let new_val_loss = r.read_f64()?;
            let committed_byte = r.read_u8()?;
            if committed_byte > 1 {
                return Err(LedgerError::MalformedEncoding("invalid bool byte"));
            }
            let update_committed = committed_byte == 1;

            let acc_count = r.read_bounded_count()?;
            let mut accepted = Vec::with_capacity(acc_count);
            for _ in 0..acc_count {
                accepted.push(AcceptedRecord {
                    participant_id: r.read_str()?,
                    participant_pubkey: r.read_fixed::<32>()?,
                    envelope_digest: r.read_fixed::<32>()?,
                    delta_hash: r.read_fixed::<32>()?,
                    sample_count: r.read_u64()?,
                    tokens_processed: r.read_u64()?,
                    observed_l2_norm: r.read_f64()?,
                    observed_cosine: r.read_f64()?,
                    observed_val_loss_delta: r.read_f64()?,
                    aggregation_weight: r.read_f64()?,
                });
            }

            let rej_count = r.read_bounded_count()?;
            let mut rejections = Vec::with_capacity(rej_count);
            for _ in 0..rej_count {
                let r_id = r.read_u64()?;
                let participant_id = r.read_str()?;
                let participant_pubkey = r.read_fixed::<32>()?;
                let envelope_digest = r.read_fixed::<32>()?;
                let r_base_hash = r.read_fixed::<32>()?;
                let delta_hash = r.read_fixed::<32>()?;
                let reason_tag = r.read_u8()?;
                let reason = RejectionReason::from_u8(reason_tag).ok_or(
                    LedgerError::MalformedEncoding("invalid rejection reason tag"),
                )?;
                rejections.push(RejectionReceipt {
                    round_id: r_id,
                    participant_id,
                    participant_pubkey,
                    envelope_digest,
                    base_model_hash: r_base_hash,
                    delta_hash,
                    reason,
                    observed_l2_norm: r.read_f64()?,
                    observed_cosine: r.read_f64()?,
                    observed_val_loss_delta: r.read_f64()?,
                    claimed_loss_delta: r.read_f64()?,
                    collusion_cluster_size: r.read_u32()?,
                    auditor_pubkey: r.read_fixed::<32>()?,
                    receipt_digest: r.read_fixed::<32>()?,
                    auditor_signature: r.read_fixed::<64>()?,
                });
            }

            let accepted_merkle_root = r.read_fixed::<32>()?;
            let rejection_merkle_root = r.read_fixed::<32>()?;
            let round_digest = r.read_fixed::<32>()?;
            let auditor_pubkey = r.read_fixed::<32>()?;
            let auditor_signature = r.read_fixed::<64>()?;

            rounds.push(RoundAuditRecord {
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
                round_digest,
                auditor_pubkey,
                auditor_signature,
            });
        }

        let ledger_digest = r.read_fixed::<32>()?;
        if r.remaining() != 0 {
            return Err(LedgerError::InvalidLength(
                "trailing bytes after ledger EOF",
            ));
        }

        Ok(Self {
            protocol_id,
            regime_name,
            corpus_pins,
            participants,
            initial_model_hash,
            final_model_hash,
            rounds,
            ledger_digest,
        })
    }

    /// Exports the audit ledger to deterministic, clean-room JSON for sovereign inspection.
    #[must_use]
    pub fn to_json_pretty(&self) -> String {
        let mut s = String::with_capacity(4096);
        s.push_str("{\n");
        s.push_str(&format!(
            "  \"protocol_id\": \"{}\",\n",
            escape_json(&self.protocol_id)
        ));
        s.push_str(&format!(
            "  \"regime_name\": \"{}\",\n",
            escape_json(&self.regime_name)
        ));
        s.push_str(&format!(
            "  \"initial_model_hash\": \"{}\",\n",
            hex_encode(&self.initial_model_hash)
        ));
        s.push_str(&format!(
            "  \"final_model_hash\": \"{}\",\n",
            hex_encode(&self.final_model_hash)
        ));
        s.push_str(&format!(
            "  \"ledger_digest\": \"{}\",\n",
            hex_encode(&self.ledger_digest)
        ));
        s.push_str(&format!(
            "  \"participant_count\": {},\n",
            self.participants.len()
        ));
        s.push_str(&format!("  \"round_count\": {},\n", self.rounds.len()));

        s.push_str("  \"corpus_pins\": [\n");
        for (i, pin) in self.corpus_pins.iter().enumerate() {
            let comma = if i + 1 < self.corpus_pins.len() {
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

        s.push_str("  \"rounds\": [\n");
        for (ri, r) in self.rounds.iter().enumerate() {
            let rcomma = if ri + 1 < self.rounds.len() { "," } else { "" };
            s.push_str("    {\n");
            s.push_str(&format!("      \"round_id\": {},\n", r.round_id));
            s.push_str(&format!(
                "      \"base_model_hash\": \"{}\",\n",
                hex_encode(&r.base_model_hash)
            ));
            s.push_str(&format!(
                "      \"new_model_hash\": \"{}\",\n",
                hex_encode(&r.new_model_hash)
            ));
            s.push_str(&format!(
                "      \"base_val_loss\": {:.6},\n",
                r.base_val_loss
            ));
            s.push_str(&format!("      \"new_val_loss\": {:.6},\n", r.new_val_loss));
            s.push_str(&format!(
                "      \"update_committed\": {},\n",
                r.update_committed
            ));
            s.push_str(&format!(
                "      \"accepted_count\": {},\n",
                r.accepted.len()
            ));
            s.push_str(&format!(
                "      \"rejected_count\": {},\n",
                r.rejections.len()
            ));
            s.push_str(&format!(
                "      \"accepted_merkle_root\": \"{}\",\n",
                hex_encode(&r.accepted_merkle_root)
            ));
            s.push_str(&format!(
                "      \"rejection_merkle_root\": \"{}\",\n",
                hex_encode(&r.rejection_merkle_root)
            ));
            s.push_str(&format!(
                "      \"round_digest\": \"{}\",\n",
                hex_encode(&r.round_digest)
            ));
            s.push_str(&format!(
                "      \"auditor_pubkey\": \"{}\",\n",
                hex_encode(&r.auditor_pubkey)
            ));
            s.push_str(&format!(
                "      \"auditor_signature\": \"{}\",\n",
                hex_encode(&r.auditor_signature)
            ));
            s.push_str("      \"rejections\": [\n");
            for (ji, rej) in r.rejections.iter().enumerate() {
                let jcomma = if ji + 1 < r.rejections.len() { "," } else { "" };
                s.push_str(&format!(
                    "        {{\"participant_id\": \"{}\", \"reason\": \"{}\", \"l2_norm\": {:.6}, \"cosine\": {:.6}, \"val_loss_delta\": {:.6}, \"claimed_loss_delta\": {:.6}, \"cluster_size\": {}, \"receipt_digest\": \"{}\"}}{}\n",
                    escape_json(&rej.participant_id),
                    rej.reason.as_str(),
                    rej.observed_l2_norm,
                    rej.observed_cosine,
                    rej.observed_val_loss_delta,
                    rej.claimed_loss_delta,
                    rej.collusion_cluster_size,
                    hex_encode(&rej.receipt_digest),
                    jcomma
                ));
            }
            s.push_str("      ]\n");
            s.push_str(&format!("    }}{}\n", rcomma));
        }
        s.push_str("  ]\n");
        s.push_str("}\n");
        s
    }
}

/// Summary produced by independent offline verification of a [`FederationAuditLedger`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SovereignAuditSummary {
    /// Number of rounds cryptographically verified.
    pub verified_rounds: usize,
    /// Total accepted update records verified across all rounds.
    pub verified_accepted: usize,
    /// Total Ed25519-signed `RejectionReceipt`s verified across all rounds.
    pub verified_rejections: usize,
    /// Aggregate rejection counts by [`RejectionReason`].
    pub rejections_by_reason: BTreeMap<RejectionReason, usize>,
    /// External Hanzo control-plane calls performed (always `0`).
    pub hanzo_control_plane_calls: u64,
}

/// Independently verifies an entire [`FederationAuditLedger`] offline with **0 Hanzo control-plane calls**.
///
/// Checks participant fingerprints, hash-chain continuity across all rounds, Merkle roots,
/// round auditor Ed25519 signatures, every individual `RejectionReceipt` Ed25519 signature,
/// and the top-level `ledger_digest`.
pub fn verify_ledger_sovereign(
    ledger: &FederationAuditLedger,
) -> Result<SovereignAuditSummary, LedgerError> {
    for p in &ledger.participants {
        if !p.verify_fingerprint() {
            return Err(LedgerError::InvalidParticipantFingerprint(
                p.participant_id.clone(),
            ));
        }
    }

    let mut expected_model_hash = ledger.initial_model_hash;
    let mut verified_accepted = 0usize;
    let mut verified_rejections = 0usize;
    let mut rejections_by_reason: BTreeMap<RejectionReason, usize> = BTreeMap::new();

    for (idx, round) in ledger.rounds.iter().enumerate() {
        if round.round_id != idx as u64 {
            return Err(LedgerError::BrokenModelHashChain(round.round_id));
        }
        if !ct_eq_bytes(&round.base_model_hash, &expected_model_hash) {
            return Err(LedgerError::BrokenModelHashChain(round.round_id));
        }

        // Verify every RejectionReceipt Ed25519 signature and digest.
        for rej in &round.rejections {
            if rej.round_id != round.round_id || rej.verify().is_err() {
                return Err(LedgerError::InvalidRejectionReceipt {
                    round_id: round.round_id,
                    participant_id: rej.participant_id.clone(),
                });
            }
            *rejections_by_reason.entry(rej.reason).or_insert(0) += 1;
            verified_rejections += 1;
        }

        // Verify accepted and rejection Merkle roots.
        let acc_leaves: Vec<Digest32> = round.accepted.iter().map(AcceptedRecord::digest).collect();
        let expected_acc_root = merkle_root(&acc_leaves);
        if !ct_eq_bytes(&round.accepted_merkle_root, &expected_acc_root) {
            return Err(LedgerError::MerkleRootMismatch(round.round_id));
        }
        let rej_leaves: Vec<Digest32> = round.rejections.iter().map(|r| r.receipt_digest).collect();
        let expected_rej_root = merkle_root(&rej_leaves);
        if !ct_eq_bytes(&round.rejection_merkle_root, &expected_rej_root) {
            return Err(LedgerError::MerkleRootMismatch(round.round_id));
        }

        // Verify round digest and auditor Ed25519 signature.
        let expected_round_digest = round.compute_digest();
        if !ct_eq_bytes(&round.round_digest, &expected_round_digest) {
            return Err(LedgerError::InvalidRoundSignature(round.round_id));
        }
        if ed25519_verify(
            &round.auditor_pubkey,
            &round.round_digest,
            &round.auditor_signature,
        )
        .is_err()
        {
            return Err(LedgerError::InvalidRoundSignature(round.round_id));
        }

        verified_accepted += round.accepted.len();
        expected_model_hash = round.new_model_hash;
    }

    if !ct_eq_bytes(&expected_model_hash, &ledger.final_model_hash) {
        return Err(LedgerError::BrokenModelHashChain(ledger.rounds.len() as u64));
    }

    let expected_ledger_digest = ledger.compute_digest();
    if !ct_eq_bytes(&ledger.ledger_digest, &expected_ledger_digest) {
        return Err(LedgerError::LedgerDigestMismatch);
    }

    Ok(SovereignAuditSummary {
        verified_rounds: ledger.rounds.len(),
        verified_accepted,
        verified_rejections,
        rejections_by_reason,
        hanzo_control_plane_calls: 0,
    })
}

/// Escapes a string for JSON emission.
#[must_use]
pub fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

struct WireWriter {
    buf: Vec<u8>,
}

impl WireWriter {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(4096),
        }
    }
    fn write_raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }
    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_f64(&mut self, v: f64) {
        let canonical = if v == 0.0 { 0.0f64 } else { v };
        self.buf.extend_from_slice(&canonical.to_le_bytes());
    }
    fn write_str(&mut self, s: &str) {
        self.write_u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
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
    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }
    fn read_slice(&mut self, n: usize) -> Result<&'a [u8], LedgerError> {
        if self.remaining() < n {
            return Err(LedgerError::InvalidLength("unexpected EOF"));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn read_fixed<const N: usize>(&mut self) -> Result<[u8; N], LedgerError> {
        let s = self.read_slice(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(s);
        Ok(out)
    }
    fn read_u8(&mut self) -> Result<u8, LedgerError> {
        Ok(self.read_fixed::<1>()?[0])
    }
    fn read_u32(&mut self) -> Result<u32, LedgerError> {
        Ok(u32::from_le_bytes(self.read_fixed::<4>()?))
    }
    fn read_u64(&mut self) -> Result<u64, LedgerError> {
        Ok(u64::from_le_bytes(self.read_fixed::<8>()?))
    }
    fn read_f64(&mut self) -> Result<f64, LedgerError> {
        let v = f64::from_le_bytes(self.read_fixed::<8>()?);
        if !v.is_finite() {
            return Err(LedgerError::MalformedEncoding("non-finite f64"));
        }
        Ok(v)
    }
    fn read_bounded_count(&mut self) -> Result<usize, LedgerError> {
        let c = self.read_u32()? as usize;
        if c > MAX_WIRE_ITEMS {
            return Err(LedgerError::InvalidLength("item count exceeds wire limit"));
        }
        Ok(c)
    }
    fn read_str(&mut self) -> Result<String, LedgerError> {
        let len = self.read_u32()? as usize;
        if len > MAX_WIRE_STR_LEN {
            return Err(LedgerError::InvalidLength("string exceeds wire limit"));
        }
        let bytes = self.read_slice(len)?;
        std::str::from_utf8(bytes)
            .map(ToString::to_string)
            .map_err(|_| LedgerError::MalformedEncoding("invalid UTF-8 string"))
    }
}
