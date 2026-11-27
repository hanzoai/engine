// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Cryptographic model-update identity, signed update envelopes, and verifiable
//! rejected-update attribution receipts (`hanzo-fedverify`).

use crate::crypto::{
    ct_eq_bytes, ed25519_verify, sha256, CryptoError, Digest32, Ed25519Keypair, PublicKeyBytes,
    Sha256, SignatureBytes,
};
use std::fmt;

/// Current canonical protocol version for `SignedUpdateEnvelope` and `RejectionReceipt`.
pub const PROTOCOL_VERSION: u32 = 1;

/// Registered public identity of a federated training participant or independent auditor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParticipantIdentity {
    /// Human-readable participant identifier (e.g., `"node-00"`).
    pub participant_id: String,
    /// 32-byte RFC 8032 Ed25519 public key.
    pub public_key: PublicKeyBytes,
    /// Domain-separated SHA-256 fingerprint binding `participant_id` and `public_key`.
    pub key_fingerprint: Digest32,
}

impl ParticipantIdentity {
    /// Constructs a canonical participant identity and computes its fingerprint.
    #[must_use]
    pub fn new(participant_id: impl Into<String>, public_key: PublicKeyBytes) -> Self {
        let participant_id = participant_id.into();
        let key_fingerprint = Self::compute_fingerprint(&participant_id, &public_key);
        Self {
            participant_id,
            public_key,
            key_fingerprint,
        }
    }

    /// Computes `SHA-256("hanzo.fedverify.id.v1" || len(id) || id || public_key)`.
    #[must_use]
    pub fn compute_fingerprint(participant_id: &str, public_key: &PublicKeyBytes) -> Digest32 {
        let mut h = Sha256::new();
        h.update(b"hanzo.fedverify.id.v1");
        h.update_len_prefixed(participant_id.as_bytes());
        h.update(public_key);
        h.finalize()
    }

    /// Verifies that `self.key_fingerprint` matches `(participant_id, public_key)`.
    #[must_use]
    pub fn verify_fingerprint(&self) -> bool {
        let expected = Self::compute_fingerprint(&self.participant_id, &self.public_key);
        ct_eq_bytes(&self.key_fingerprint, &expected)
    }
}

/// Participant signing handle combining a `ParticipantIdentity` and its Ed25519 keypair.
#[derive(Debug, Clone)]
pub struct ParticipantSigner {
    /// Public identity registered for this participant.
    pub identity: ParticipantIdentity,
    keypair: Ed25519Keypair,
}

impl ParticipantSigner {
    /// Deterministically creates a participant signer for simulation and testing.
    #[must_use]
    pub fn from_seed_label(participant_id: impl Into<String>, seed: u64) -> Self {
        let id = participant_id.into();
        let keypair = Ed25519Keypair::from_label(&id, seed);
        let identity = ParticipantIdentity::new(id, keypair.public_key());
        Self { identity, keypair }
    }

    /// Creates a participant signer from an explicit 32-byte Ed25519 seed.
    #[must_use]
    pub fn from_seed_bytes(participant_id: impl Into<String>, seed: [u8; 32]) -> Self {
        let id = participant_id.into();
        let keypair = Ed25519Keypair::from_seed(seed);
        let identity = ParticipantIdentity::new(id, keypair.public_key());
        Self { identity, keypair }
    }

    /// Signs raw message bytes with RFC 8032 Ed25519.
    #[must_use]
    pub fn sign_bytes(&self, msg: &[u8]) -> SignatureBytes {
        self.keypair.sign(msg)
    }

    /// Constructs and signs a `SignedUpdateEnvelope`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn sign_update(
        &self,
        round_id: u64,
        base_model_hash: Digest32,
        update_kind: UpdateKind,
        payload: Vec<f64>,
        claimed_loss_before: f64,
        claimed_loss_after: f64,
        sample_count: u64,
        tokens_processed: u64,
        nonce: u64,
    ) -> SignedUpdateEnvelope {
        let delta_hash = hash_f64_slice(&payload);
        let claimed_loss_delta = claimed_loss_before - claimed_loss_after;
        let header = UpdateHeader {
            protocol_version: PROTOCOL_VERSION,
            round_id,
            participant_id: self.identity.participant_id.clone(),
            participant_pubkey: self.identity.public_key,
            base_model_hash,
            update_kind,
            delta_hash,
            claimed_loss_before,
            claimed_loss_after,
            claimed_loss_delta,
            sample_count,
            tokens_processed,
            nonce,
        };
        let envelope_digest = header.digest();
        let signature = self.keypair.sign(&envelope_digest);
        SignedUpdateEnvelope {
            header,
            payload,
            envelope_digest,
            signature,
        }
    }
}

/// Representation of the submitted model update payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateKind {
    /// Dense parameter delta (`ΔW`) of exact model dimension.
    GradientDelta,
    /// Low-rank adapter (`ΔW = B · A`), serialized as `A [rank * cols]` followed by `B [rows * rank]`.
    LoraAdapter {
        /// LoRA bottleneck rank `r`.
        rank: u32,
        /// Output rows (vocabulary/class count).
        rows: u32,
        /// Input feature dimension.
        cols: u32,
    },
}

impl UpdateKind {
    /// Canonical wire tag for hashing and serialization.
    #[must_use]
    pub fn tag(self) -> u8 {
        match self {
            Self::GradientDelta => 1,
            Self::LoraAdapter { .. } => 2,
        }
    }

    /// Materializes a payload into a dense delta vector of length `expected_dim`.
    ///
    /// Returns `None` if dimensions do not match or any element is non-finite.
    #[must_use]
    pub fn materialize_dense(self, payload: &[f64], expected_dim: usize) -> Option<Vec<f64>> {
        if !payload.iter().all(|v| v.is_finite()) {
            return None;
        }
        match self {
            Self::GradientDelta => {
                if payload.len() != expected_dim {
                    return None;
                }
                Some(payload.to_vec())
            }
            Self::LoraAdapter { rank, rows, cols } => {
                let r = rank as usize;
                let m = rows as usize;
                let k = cols as usize;
                if r == 0 || m.checked_mul(k)? != expected_dim {
                    return None;
                }
                let a_len = r.checked_mul(k)?;
                let b_len = m.checked_mul(r)?;
                if payload.len() != a_len.checked_add(b_len)? {
                    return None;
                }
                let a = &payload[..a_len];
                let b = &payload[a_len..];
                let mut dense = vec![0.0f64; expected_dim];
                for row in 0..m {
                    for col in 0..k {
                        let mut sum = 0.0f64;
                        for t in 0..r {
                            sum += b[row * r + t] * a[t * k + col];
                        }
                        if !sum.is_finite() {
                            return None;
                        }
                        dense[row * k + col] = sum;
                    }
                }
                Some(dense)
            }
        }
    }
}

/// Computes a canonical domain-separated SHA-256 hash over an `f64` parameter slice.
#[must_use]
pub fn hash_f64_slice(values: &[f64]) -> Digest32 {
    let mut h = Sha256::new();
    h.update(b"hanzo.fedverify.tensor.f64.v1");
    h.update(&(values.len() as u64).to_le_bytes());
    for &v in values {
        h.update_f64(v);
    }
    h.finalize()
}

/// Canonical metadata header bound into every participant's Ed25519 signature.
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateHeader {
    /// Protocol version (`PROTOCOL_VERSION`).
    pub protocol_version: u32,
    /// Zero-based federated training round index.
    pub round_id: u64,
    /// Participant identifier.
    pub participant_id: String,
    /// Participant's Ed25519 public key.
    pub participant_pubkey: PublicKeyBytes,
    /// SHA-256 commitment to the global model weights before local training.
    pub base_model_hash: Digest32,
    /// Whether the payload is a dense gradient delta or factorized LoRA adapter.
    pub update_kind: UpdateKind,
    /// SHA-256 digest of the canonical `payload` vector.
    pub delta_hash: Digest32,
    /// Participant-reported local loss before training.
    pub claimed_loss_before: f64,
    /// Participant-reported local loss after training.
    pub claimed_loss_after: f64,
    /// Participant-claimed improvement (`claimed_loss_before - claimed_loss_after`).
    pub claimed_loss_delta: f64,
    /// Number of local training examples processed.
    pub sample_count: u64,
    /// Number of local tokens processed (used for HIP-1350 cost/throughput metrics).
    pub tokens_processed: u64,
    /// Strictly monotonic per-participant anti-replay nonce.
    pub nonce: u64,
}

impl UpdateHeader {
    /// Computes the canonical domain-separated SHA-256 digest signed by the participant.
    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut h = Sha256::new();
        h.update(b"hanzo.fedverify.envelope.v1");
        h.update(&self.protocol_version.to_le_bytes());
        h.update(&self.round_id.to_le_bytes());
        h.update_len_prefixed(self.participant_id.as_bytes());
        h.update(&self.participant_pubkey);
        h.update(&self.base_model_hash);
        h.update(&[self.update_kind.tag()]);
        if let UpdateKind::LoraAdapter { rank, rows, cols } = self.update_kind {
            h.update(&rank.to_le_bytes());
            h.update(&rows.to_le_bytes());
            h.update(&cols.to_le_bytes());
        }
        h.update(&self.delta_hash);
        h.update_f64(self.claimed_loss_before);
        h.update_f64(self.claimed_loss_after);
        h.update_f64(self.claimed_loss_delta);
        h.update(&self.sample_count.to_le_bytes());
        h.update(&self.tokens_processed.to_le_bytes());
        h.update(&self.nonce.to_le_bytes());
        h.finalize()
    }
}

/// Cryptographically signed federated model update submitted by an untrusted participant.
#[derive(Debug, Clone, PartialEq)]
pub struct SignedUpdateEnvelope {
    /// Bound metadata header.
    pub header: UpdateHeader,
    /// Dense gradient/delta or flattened LoRA `(A || B)` parameters.
    pub payload: Vec<f64>,
    /// Canonical header digest (`header.digest()`).
    pub envelope_digest: Digest32,
    /// Ed25519 signature over `envelope_digest` under `header.participant_pubkey`.
    pub signature: SignatureBytes,
}

impl SignedUpdateEnvelope {
    /// Verifies cryptographic integrity of this envelope (header digest, payload hash, Ed25519 signature).
    pub fn verify_cryptography(&self) -> Result<(), CryptoError> {
        let expected_digest = self.header.digest();
        if !ct_eq_bytes(&self.envelope_digest, &expected_digest) {
            return Err(CryptoError::InvalidSignature);
        }
        let expected_payload_hash = hash_f64_slice(&self.payload);
        if !ct_eq_bytes(&self.header.delta_hash, &expected_payload_hash) {
            return Err(CryptoError::InvalidSignature);
        }
        ed25519_verify(
            &self.header.participant_pubkey,
            &self.envelope_digest,
            &self.signature,
        )
    }
}

/// Explicit machine-readable reason code recorded in a signed `RejectionReceipt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum RejectionReason {
    /// Ed25519 signature failed, header digest mismatch, or payload hash mismatch.
    InvalidSignature = 1,
    /// Participant ID / public key is not registered in the round membership set.
    UnknownParticipant = 2,
    /// Nonce is non-monotonic, already consumed, or duplicate submission in the same round.
    ReplayNonce = 3,
    /// `round_id` or `base_model_hash` does not match the active round's global model state.
    StaleBaseModel = 4,
    /// Payload or header metrics contain `NaN`/`±Inf` or invalid tensor/LoRA dimensions.
    NonFiniteOrMalformed = 5,
    /// L2 norm violates hard norm ceiling or adaptive median-norm bound (e.g. `100×` scaling attack).
    NormBoundViolation = 6,
    /// Coordinated Sybil/cartel cluster detected via anomalous pairwise cosine/L2 cloning + quality check.
    CollusionClusterDetected = 7,
    /// Update direction diverges severely from robust consensus direction (e.g. sign-flipping / gradient ascent).
    CosineDirectionAnomaly = 8,
    /// Multi-Krum distance or coordinate-wise MAD outlier score exceeds statistical bound.
    CoordinateOutlier = 9,
    /// Independent held-out validation loss regresses or contradicts `claimed_loss_delta`.
    ValidationLossRegression = 10,
}

impl RejectionReason {
    /// Converts a wire byte tag back into a `RejectionReason`.
    #[must_use]
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::InvalidSignature),
            2 => Some(Self::UnknownParticipant),
            3 => Some(Self::ReplayNonce),
            4 => Some(Self::StaleBaseModel),
            5 => Some(Self::NonFiniteOrMalformed),
            6 => Some(Self::NormBoundViolation),
            7 => Some(Self::CollusionClusterDetected),
            8 => Some(Self::CosineDirectionAnomaly),
            9 => Some(Self::CoordinateOutlier),
            10 => Some(Self::ValidationLossRegression),
            _ => None,
        }
    }

    /// Stable snake_case label for JSON and audit reports.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidSignature => "invalid_signature",
            Self::UnknownParticipant => "unknown_participant",
            Self::ReplayNonce => "replay_nonce",
            Self::StaleBaseModel => "stale_base_model",
            Self::NonFiniteOrMalformed => "non_finite_or_malformed",
            Self::NormBoundViolation => "norm_bound_violation",
            Self::CollusionClusterDetected => "collusion_cluster_detected",
            Self::CosineDirectionAnomaly => "cosine_direction_anomaly",
            Self::CoordinateOutlier => "coordinate_outlier",
            Self::ValidationLossRegression => "validation_loss_regression",
        }
    }
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Cryptographically signed receipt attributing why a specific participant's update was rejected.
///
/// Any sovereign third party can verify `receipt_digest` and `auditor_signature`
/// offline with zero Hanzo control-plane calls.
#[derive(Debug, Clone, PartialEq)]
pub struct RejectionReceipt {
    /// Round index in which the update was rejected.
    pub round_id: u64,
    /// Participant identifier blamed for the rejected update.
    pub participant_id: String,
    /// Participant public key bound to the submission.
    pub participant_pubkey: PublicKeyBytes,
    /// Envelope digest of the rejected submission.
    pub envelope_digest: Digest32,
    /// Base model hash referenced by the rejected submission.
    pub base_model_hash: Digest32,
    /// Payload hash referenced by the rejected submission.
    pub delta_hash: Digest32,
    /// Machine-readable reason code.
    pub reason: RejectionReason,
    /// Observed L2 norm of the materialized delta (`0.0` if rejected before materialization).
    pub observed_l2_norm: f64,
    /// Observed cosine similarity against the robust reference vector (`0.0` if not evaluated).
    pub observed_cosine: f64,
    /// Observed held-out validation loss improvement (`loss_before - loss_after`, `0.0` if not evaluated).
    pub observed_val_loss_delta: f64,
    /// Participant's claimed loss improvement from the envelope header.
    pub claimed_loss_delta: f64,
    /// Size of detected collusion cluster (`1` if not part of a collusion cluster).
    pub collusion_cluster_size: u32,
    /// Auditor/aggregator Ed25519 public key that signed this receipt.
    pub auditor_pubkey: PublicKeyBytes,
    /// Canonical SHA-256 digest of the receipt body.
    pub receipt_digest: Digest32,
    /// Ed25519 signature over `receipt_digest` under `auditor_pubkey`.
    pub auditor_signature: SignatureBytes,
}

impl RejectionReceipt {
    /// Computes the canonical digest over all attribution fields (excluding `receipt_digest` and `auditor_signature`).
    #[must_use]
    pub fn compute_digest(&self) -> Digest32 {
        let mut h = Sha256::new();
        h.update(b"hanzo.fedverify.rejection.v1");
        h.update(&PROTOCOL_VERSION.to_le_bytes());
        h.update(&self.round_id.to_le_bytes());
        h.update_len_prefixed(self.participant_id.as_bytes());
        h.update(&self.participant_pubkey);
        h.update(&self.envelope_digest);
        h.update(&self.base_model_hash);
        h.update(&self.delta_hash);
        h.update(&[self.reason as u8]);
        h.update_f64(self.observed_l2_norm);
        h.update_f64(self.observed_cosine);
        h.update_f64(self.observed_val_loss_delta);
        h.update_f64(self.claimed_loss_delta);
        h.update(&self.collusion_cluster_size.to_le_bytes());
        h.update(&self.auditor_pubkey);
        h.finalize()
    }

    /// Constructs and signs a `RejectionReceipt` using `auditor`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        auditor: &ParticipantSigner,
        envelope: &SignedUpdateEnvelope,
        round_id: u64,
        reason: RejectionReason,
        observed_l2_norm: f64,
        observed_cosine: f64,
        observed_val_loss_delta: f64,
        collusion_cluster_size: u32,
    ) -> Self {
        let mut receipt = Self {
            round_id,
            participant_id: envelope.header.participant_id.clone(),
            participant_pubkey: envelope.header.participant_pubkey,
            envelope_digest: envelope.envelope_digest,
            base_model_hash: envelope.header.base_model_hash,
            delta_hash: envelope.header.delta_hash,
            reason,
            observed_l2_norm: sanitize_finite(observed_l2_norm),
            observed_cosine: sanitize_finite(observed_cosine),
            observed_val_loss_delta: sanitize_finite(observed_val_loss_delta),
            claimed_loss_delta: sanitize_finite(envelope.header.claimed_loss_delta),
            collusion_cluster_size,
            auditor_pubkey: auditor.identity.public_key,
            receipt_digest: [0u8; 32],
            auditor_signature: [0u8; 64],
        };
        receipt.receipt_digest = receipt.compute_digest();
        receipt.auditor_signature = auditor.sign_bytes(&receipt.receipt_digest);
        receipt
    }

    /// Verifies `receipt_digest` and the Ed25519 `auditor_signature` offline.
    pub fn verify(&self) -> Result<(), CryptoError> {
        let expected = self.compute_digest();
        if !ct_eq_bytes(&self.receipt_digest, &expected) {
            return Err(CryptoError::InvalidSignature);
        }
        ed25519_verify(
            &self.auditor_pubkey,
            &self.receipt_digest,
            &self.auditor_signature,
        )
    }
}

/// Cryptographic summary of an accepted update committed to the round's Merkle root.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedRecord {
    /// Participant identifier.
    pub participant_id: String,
    /// Participant Ed25519 public key.
    pub participant_pubkey: PublicKeyBytes,
    /// Envelope digest of the accepted submission.
    pub envelope_digest: Digest32,
    /// Payload hash of the accepted submission.
    pub delta_hash: Digest32,
    /// Local samples contributed.
    pub sample_count: u64,
    /// Local tokens processed.
    pub tokens_processed: u64,
    /// Observed L2 norm of the materialized update.
    pub observed_l2_norm: f64,
    /// Observed cosine similarity against the robust reference vector.
    pub observed_cosine: f64,
    /// Independent held-out validation loss delta (`loss_before - loss_after`).
    pub observed_val_loss_delta: f64,
    /// Aggregation weight assigned to this participant in the round.
    pub aggregation_weight: f64,
}

impl AcceptedRecord {
    /// Computes canonical SHA-256 leaf digest for `accepted_merkle_root`.
    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut h = Sha256::new();
        h.update(b"hanzo.fedverify.accepted.v1");
        h.update_len_prefixed(self.participant_id.as_bytes());
        h.update(&self.participant_pubkey);
        h.update(&self.envelope_digest);
        h.update(&self.delta_hash);
        h.update(&self.sample_count.to_le_bytes());
        h.update(&self.tokens_processed.to_le_bytes());
        h.update_f64(self.observed_l2_norm);
        h.update_f64(self.observed_cosine);
        h.update_f64(self.observed_val_loss_delta);
        h.update_f64(self.aggregation_weight);
        h.finalize()
    }
}

fn sanitize_finite(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Helper to compute an initial canonical digest over raw bytes.
#[must_use]
pub fn digest_bytes(data: &[u8]) -> Digest32 {
    sha256(data)
}
