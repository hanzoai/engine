// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Length-prefixed binary activation transport wire protocol over TCP sockets
//! with BF16/INT8 activation codecs, SHA-256 frame integrity commitments,
//! authenticated HMAC-SHA256 stage receipts, and explicit trust-tier typing.

use crate::model::{decode_bf16, decode_int8, encode_bf16, encode_int8};
use crate::sha256::{constant_time_eq, hmac_sha256, sha256, DIGEST_LEN};
use std::fmt;
use std::io::{self, Read, Write};

/// Four-byte wire magic header (`"HZP2"`).
pub const WIRE_MAGIC: [u8; 4] = *b"HZP2";

/// Current binary wire protocol version.
pub const WIRE_VERSION: u8 = 1;

/// Upper bound on any single wire frame payload (`4 MiB`) to prevent OOM / DoS.
pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

/// Activation wire encoding format across peer-to-peer pipeline hops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActivationCodec {
    /// Lossless IEEE-754 BF16 (`2` bytes per element) for bit-exact stage handoffs.
    Bf16Lossless = 1,
    /// Per-tensor symmetric INT8 (`4`-byte `f32` scale + `1` byte per element, ~2x bandwidth cut).
    Int8PerTensor = 2,
}

impl ActivationCodec {
    /// Decodes a wire byte into an [`ActivationCodec`].
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Bf16Lossless),
            2 => Some(Self::Int8PerTensor),
            _ => None,
        }
    }

    /// Encodes a float hidden-state vector into wire payload bytes.
    #[must_use]
    pub fn encode(&self, hidden: &[f32]) -> Vec<u8> {
        match self {
            Self::Bf16Lossless => encode_bf16(hidden),
            Self::Int8PerTensor => encode_int8(hidden),
        }
    }

    /// Decodes wire payload bytes back into an `f32` hidden-state vector.
    #[must_use]
    pub fn decode(&self, payload: &[u8]) -> Option<Vec<f32>> {
        match self {
            Self::Bf16Lossless => decode_bf16(payload),
            Self::Int8PerTensor => decode_int8(payload),
        }
    }

    /// Canonical string label for telemetry and manifests.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Bf16Lossless => "bf16_lossless",
            Self::Int8PerTensor => "int8_per_tensor",
        }
    }
}

/// Explicit trust and verification tier declaration (never conflating cryptographic
/// commitments, hardware attestations, and economically incentivized spot checks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustTier {
    /// SHA-256 activation/weight commitments + HMAC-SHA256 peer stage receipts +
    /// deterministic or probabilistic spot-check re-execution challenge.
    /// Does NOT claim zero-knowledge soundness or hardware enclave confidentiality.
    CommitmentReceiptAndSpotCheck,
    /// Hardware trusted execution environment attestation (declared separately;
    /// not exercised by pure-CPU TCP loopback tests).
    HardwareTeeAttestation,
    /// Full cryptographic zero-knowledge validity proof over inference execution
    /// (declared separately; not exercised by this transport layer).
    ZeroKnowledgeProof,
}

impl TrustTier {
    /// Canonical string tag for manifests and audit dossiers.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::CommitmentReceiptAndSpotCheck => "commitment_receipt_and_spot_check",
            Self::HardwareTeeAttestation => "hardware_tee_attestation",
            Self::ZeroKnowledgeProof => "zero_knowledge_proof",
        }
    }
}

/// Authenticated peer execution receipt covering one pipeline stage forward pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageReceipt {
    /// Independent operator peer identifier.
    pub peer_id: String,
    /// Globally unique inference session identifier (binds receipt to session).
    pub session_id: u64,
    /// Autoregressive step index (binds receipt to exact step).
    pub step_idx: u32,
    /// Pipeline stage ordinal.
    pub stage_idx: u16,
    /// Inclusive starting layer index executed.
    pub layer_start: u16,
    /// Exclusive ending layer index executed.
    pub layer_end: u16,
    /// SHA-256 digest of the exact stage weights executed.
    pub weights_commitment: [u8; DIGEST_LEN],
    /// SHA-256 digest of the incoming activation wire bytes.
    pub input_digest: [u8; DIGEST_LEN],
    /// SHA-256 digest of the outgoing activation wire bytes.
    pub output_digest: [u8; DIGEST_LEN],
    /// RFC 2104 HMAC-SHA256 signature computed with the peer's secret key.
    pub mac: [u8; DIGEST_LEN],
}

impl StageReceipt {
    /// Computes the canonical signing transcript bytes for this receipt.
    #[must_use]
    pub fn signing_transcript(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(160 + self.peer_id.len());
        buf.extend_from_slice(b"hanzo-p2p-stage-receipt-v1");
        buf.extend_from_slice(&(self.peer_id.len() as u16).to_le_bytes());
        buf.extend_from_slice(self.peer_id.as_bytes());
        buf.extend_from_slice(&self.session_id.to_le_bytes());
        buf.extend_from_slice(&self.step_idx.to_le_bytes());
        buf.extend_from_slice(&self.stage_idx.to_le_bytes());
        buf.extend_from_slice(&self.layer_start.to_le_bytes());
        buf.extend_from_slice(&self.layer_end.to_le_bytes());
        buf.extend_from_slice(&self.weights_commitment);
        buf.extend_from_slice(&self.input_digest);
        buf.extend_from_slice(&self.output_digest);
        buf
    }

    /// Creates and signs a [`StageReceipt`] with `peer_secret`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        peer_id: String,
        session_id: u64,
        step_idx: u32,
        stage_idx: u16,
        layer_start: u16,
        layer_end: u16,
        weights_commitment: [u8; DIGEST_LEN],
        input_digest: [u8; DIGEST_LEN],
        output_digest: [u8; DIGEST_LEN],
        peer_secret: &[u8],
    ) -> Self {
        let mut receipt = Self {
            peer_id,
            session_id,
            step_idx,
            stage_idx,
            layer_start,
            layer_end,
            weights_commitment,
            input_digest,
            output_digest,
            mac: [0u8; DIGEST_LEN],
        };
        let transcript = receipt.signing_transcript();
        receipt.mac = hmac_sha256(peer_secret, &transcript);
        receipt
    }

    /// Verifies the receipt's HMAC-SHA256 signature in constant time.
    #[must_use]
    pub fn verify_mac(&self, peer_secret: &[u8]) -> bool {
        let transcript = self.signing_transcript();
        let expected = hmac_sha256(peer_secret, &transcript);
        constant_time_eq(&self.mac, &expected)
    }

    fn write_to(&self, out: &mut Vec<u8>) {
        write_str(out, &self.peer_id);
        out.extend_from_slice(&self.session_id.to_le_bytes());
        out.extend_from_slice(&self.step_idx.to_le_bytes());
        out.extend_from_slice(&self.stage_idx.to_le_bytes());
        out.extend_from_slice(&self.layer_start.to_le_bytes());
        out.extend_from_slice(&self.layer_end.to_le_bytes());
        out.extend_from_slice(&self.weights_commitment);
        out.extend_from_slice(&self.input_digest);
        out.extend_from_slice(&self.output_digest);
        out.extend_from_slice(&self.mac);
    }

    fn read_from(cur: &mut Cursor<'_>) -> Result<Self, TransportError> {
        let peer_id = cur.read_str()?;
        let session_id = cur.read_u64()?;
        let step_idx = cur.read_u32()?;
        let stage_idx = cur.read_u16()?;
        let layer_start = cur.read_u16()?;
        let layer_end = cur.read_u16()?;
        let weights_commitment = cur.read_digest()?;
        let input_digest = cur.read_digest()?;
        let output_digest = cur.read_digest()?;
        let mac = cur.read_digest()?;
        Ok(Self {
            peer_id,
            session_id,
            step_idx,
            stage_idx,
            layer_start,
            layer_end,
            weights_commitment,
            input_digest,
            output_digest,
            mac,
        })
    }
}

/// Structured protocol error codes returned inside [`WireFrame::ErrorFrame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ProtocolErrorCode {
    /// Incoming or outgoing activation digest did not match payload SHA-256.
    ActivationIntegrityMismatch = 1,
    /// Stage weights commitment did not match caller expectation.
    WeightsCommitmentMismatch = 2,
    /// Requested layer slice is not hosted by this peer.
    StageMismatch = 3,
    /// Peer rejected request due to active-session concurrency backpressure.
    BackpressureRejected = 4,
    /// Frame encoding or payload shape was malformed.
    MalformedFrame = 5,
}

impl ProtocolErrorCode {
    #[must_use]
    const fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::ActivationIntegrityMismatch),
            2 => Some(Self::WeightsCommitmentMismatch),
            3 => Some(Self::StageMismatch),
            4 => Some(Self::BackpressureRejected),
            5 => Some(Self::MalformedFrame),
            _ => None,
        }
    }
}

/// Binary wire messages exchanged over live TCP connections between orchestrator and peers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireFrame {
    /// Liveness and capability heartbeat probe.
    HealthPing {
        /// Monotonic nonce echoed in [`WireFrame::HealthPong`].
        nonce: u64,
    },
    /// Heartbeat response carrying peer identity and hosted layer commitments.
    HealthPong {
        /// Echoed probe nonce.
        nonce: u64,
        /// Peer identifier.
        peer_id: String,
        /// Active inference session count currently resident on this peer.
        active_sessions: u32,
        /// Hosted layer ranges `(layer_start, layer_end)` on this peer.
        hosted_slices: Vec<(u16, u16)>,
    },
    /// Forward-pass activation transfer for a single autoregressive step.
    ForwardRequest {
        /// Session identifier.
        session_id: u64,
        /// Autoregressive step index.
        step_idx: u32,
        /// Pipeline stage ordinal.
        stage_idx: u16,
        /// Requested inclusive start layer.
        layer_start: u16,
        /// Requested exclusive end layer.
        layer_end: u16,
        /// Wire activation encoding format.
        codec: ActivationCodec,
        /// Expected SHA-256 commitment of `[layer_start, layer_end)` weights.
        expected_weights_commitment: [u8; DIGEST_LEN],
        /// SHA-256 digest of `payload_bytes`.
        activation_digest: [u8; DIGEST_LEN],
        /// Encoded hidden-state tensor bytes.
        payload_bytes: Vec<u8>,
    },
    /// Forward-pass activation result plus authenticated [`StageReceipt`].
    ForwardResponse {
        /// Session identifier.
        session_id: u64,
        /// Autoregressive step index.
        step_idx: u32,
        /// Pipeline stage ordinal.
        stage_idx: u16,
        /// Wire activation encoding format.
        codec: ActivationCodec,
        /// SHA-256 digest of `payload_bytes`.
        activation_digest: [u8; DIGEST_LEN],
        /// Peer compute latency in microseconds.
        compute_us: u64,
        /// Authenticated stage execution receipt.
        receipt: StageReceipt,
        /// Encoded output hidden-state tensor bytes.
        payload_bytes: Vec<u8>,
    },
    /// Failover KV-cache reconstruction batch: replays prior stage-input activations
    /// in order onto a surviving/replacement peer so its KV cache matches the lost peer.
    ReplayKvBatch {
        /// Session identifier to rebuild.
        session_id: u64,
        /// Inclusive starting layer to promote/rebuild.
        layer_start: u16,
        /// Exclusive ending layer to promote/rebuild.
        layer_end: u16,
        /// Wire codec used for every replayed activation vector.
        codec: ActivationCodec,
        /// Expected SHA-256 commitment of `[layer_start, layer_end)` weights.
        expected_weights_commitment: [u8; DIGEST_LEN],
        /// Ordered sequence of encoded hidden-state vectors (`0..replayed_tokens`).
        frames: Vec<Vec<u8>>,
    },
    /// Acknowledgement that failover KV-cache reconstruction succeeded.
    ReplayKvAck {
        /// Rebuilt session identifier.
        session_id: u64,
        /// Number of token positions restored into the stage KV cache.
        restored_tokens: u32,
        /// Rebuild compute duration in microseconds.
        rebuild_us: u64,
    },
    /// Resets/evicts a session's KV cache on a peer.
    ResetSession {
        /// Session identifier to clear.
        session_id: u64,
    },
    /// Acknowledgement for [`WireFrame::ResetSession`].
    ResetAck {
        /// Cleared session identifier.
        session_id: u64,
    },
    /// Structured error returned by a peer.
    ErrorFrame {
        /// Machine-readable protocol error code.
        code: ProtocolErrorCode,
        /// Human-readable diagnostic description.
        message: String,
    },
}

/// Errors arising from wire framing, TCP transport, or cryptographic verification.
#[derive(Debug)]
pub enum TransportError {
    /// Underlying OS socket I/O error (e.g., connection reset, refused, EOF, timeout).
    Io(io::Error),
    /// Frame magic header or protocol version was invalid.
    InvalidHeader,
    /// Frame payload exceeded [`MAX_FRAME_BYTES`].
    FrameTooLarge(usize),
    /// Wire payload was truncated or contained invalid discriminant bytes.
    MalformedPayload(String),
    /// Peer returned an explicit [`WireFrame::ErrorFrame`].
    RemoteProtocolError {
        /// Remote protocol error code.
        code: ProtocolErrorCode,
        /// Diagnostic message from peer.
        message: String,
    },
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "tcp transport io error: {e}"),
            Self::InvalidHeader => write!(f, "invalid wire header or version"),
            Self::FrameTooLarge(n) => write!(f, "wire frame too large: {n} bytes"),
            Self::MalformedPayload(s) => write!(f, "malformed wire payload: {s}"),
            Self::RemoteProtocolError { code, message } => {
                write!(f, "remote protocol error ({code:?}): {message}")
            }
        }
    }
}

impl std::error::Error for TransportError {}

impl From<io::Error> for TransportError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl WireFrame {
    /// Serializes a [`WireFrame`] into a complete length-prefixed wire byte buffer.
    #[must_use]
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        let mut body = Vec::new();
        let kind: u8 = match self {
            Self::HealthPing { nonce } => {
                body.extend_from_slice(&nonce.to_le_bytes());
                1
            }
            Self::HealthPong {
                nonce,
                peer_id,
                active_sessions,
                hosted_slices,
            } => {
                body.extend_from_slice(&nonce.to_le_bytes());
                write_str(&mut body, peer_id);
                body.extend_from_slice(&active_sessions.to_le_bytes());
                body.extend_from_slice(&(hosted_slices.len() as u16).to_le_bytes());
                for &(s, e) in hosted_slices {
                    body.extend_from_slice(&s.to_le_bytes());
                    body.extend_from_slice(&e.to_le_bytes());
                }
                2
            }
            Self::ForwardRequest {
                session_id,
                step_idx,
                stage_idx,
                layer_start,
                layer_end,
                codec,
                expected_weights_commitment,
                activation_digest,
                payload_bytes,
            } => {
                body.extend_from_slice(&session_id.to_le_bytes());
                body.extend_from_slice(&step_idx.to_le_bytes());
                body.extend_from_slice(&stage_idx.to_le_bytes());
                body.extend_from_slice(&layer_start.to_le_bytes());
                body.extend_from_slice(&layer_end.to_le_bytes());
                body.push(*codec as u8);
                body.extend_from_slice(expected_weights_commitment);
                body.extend_from_slice(activation_digest);
                write_bytes(&mut body, payload_bytes);
                3
            }
            Self::ForwardResponse {
                session_id,
                step_idx,
                stage_idx,
                codec,
                activation_digest,
                compute_us,
                receipt,
                payload_bytes,
            } => {
                body.extend_from_slice(&session_id.to_le_bytes());
                body.extend_from_slice(&step_idx.to_le_bytes());
                body.extend_from_slice(&stage_idx.to_le_bytes());
                body.push(*codec as u8);
                body.extend_from_slice(activation_digest);
                body.extend_from_slice(&compute_us.to_le_bytes());
                receipt.write_to(&mut body);
                write_bytes(&mut body, payload_bytes);
                4
            }
            Self::ReplayKvBatch {
                session_id,
                layer_start,
                layer_end,
                codec,
                expected_weights_commitment,
                frames,
            } => {
                body.extend_from_slice(&session_id.to_le_bytes());
                body.extend_from_slice(&layer_start.to_le_bytes());
                body.extend_from_slice(&layer_end.to_le_bytes());
                body.push(*codec as u8);
                body.extend_from_slice(expected_weights_commitment);
                body.extend_from_slice(&(frames.len() as u32).to_le_bytes());
                for frame in frames {
                    write_bytes(&mut body, frame);
                }
                5
            }
            Self::ReplayKvAck {
                session_id,
                restored_tokens,
                rebuild_us,
            } => {
                body.extend_from_slice(&session_id.to_le_bytes());
                body.extend_from_slice(&restored_tokens.to_le_bytes());
                body.extend_from_slice(&rebuild_us.to_le_bytes());
                6
            }
            Self::ResetSession { session_id } => {
                body.extend_from_slice(&session_id.to_le_bytes());
                7
            }
            Self::ResetAck { session_id } => {
                body.extend_from_slice(&session_id.to_le_bytes());
                8
            }
            Self::ErrorFrame { code, message } => {
                body.push(*code as u8);
                write_str(&mut body, message);
                9
            }
        };

        let mut out = Vec::with_capacity(10 + body.len());
        out.extend_from_slice(&WIRE_MAGIC);
        out.push(WIRE_VERSION);
        out.push(kind);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    /// Writes a length-prefixed wire frame to `writer`, returning total bytes written.
    pub fn write_to_stream<W: Write>(&self, writer: &mut W) -> Result<usize, TransportError> {
        let bytes = self.to_wire_bytes();
        writer.write_all(&bytes)?;
        writer.flush()?;
        Ok(bytes.len())
    }

    /// Reads and decodes a length-prefixed wire frame from `reader`, returning `(frame, total_bytes)`.
    pub fn read_from_stream<R: Read>(reader: &mut R) -> Result<(Self, usize), TransportError> {
        let mut header = [0u8; 10];
        reader.read_exact(&mut header)?;
        if header[0..4] != WIRE_MAGIC || header[4] != WIRE_VERSION {
            return Err(TransportError::InvalidHeader);
        }
        let kind = header[5];
        let body_len = u32::from_le_bytes([header[6], header[7], header[8], header[9]]) as usize;
        if body_len > MAX_FRAME_BYTES {
            return Err(TransportError::FrameTooLarge(body_len));
        }
        let mut body = vec![0u8; body_len];
        reader.read_exact(&mut body)?;

        let mut cur = Cursor {
            bytes: &body,
            pos: 0,
        };
        let frame = match kind {
            1 => Self::HealthPing {
                nonce: cur.read_u64()?,
            },
            2 => {
                let nonce = cur.read_u64()?;
                let peer_id = cur.read_str()?;
                let active_sessions = cur.read_u32()?;
                let count = cur.read_u16()? as usize;
                let mut hosted_slices = Vec::with_capacity(count);
                for _ in 0..count {
                    let s = cur.read_u16()?;
                    let e = cur.read_u16()?;
                    hosted_slices.push((s, e));
                }
                Self::HealthPong {
                    nonce,
                    peer_id,
                    active_sessions,
                    hosted_slices,
                }
            }
            3 => {
                let session_id = cur.read_u64()?;
                let step_idx = cur.read_u32()?;
                let stage_idx = cur.read_u16()?;
                let layer_start = cur.read_u16()?;
                let layer_end = cur.read_u16()?;
                let codec = ActivationCodec::from_u8(cur.read_u8()?)
                    .ok_or_else(|| TransportError::MalformedPayload("unknown codec".into()))?;
                let expected_weights_commitment = cur.read_digest()?;
                let activation_digest = cur.read_digest()?;
                let payload_bytes = cur.read_bytes()?;
                Self::ForwardRequest {
                    session_id,
                    step_idx,
                    stage_idx,
                    layer_start,
                    layer_end,
                    codec,
                    expected_weights_commitment,
                    activation_digest,
                    payload_bytes,
                }
            }
            4 => {
                let session_id = cur.read_u64()?;
                let step_idx = cur.read_u32()?;
                let stage_idx = cur.read_u16()?;
                let codec = ActivationCodec::from_u8(cur.read_u8()?)
                    .ok_or_else(|| TransportError::MalformedPayload("unknown codec".into()))?;
                let activation_digest = cur.read_digest()?;
                let compute_us = cur.read_u64()?;
                let receipt = StageReceipt::read_from(&mut cur)?;
                let payload_bytes = cur.read_bytes()?;
                Self::ForwardResponse {
                    session_id,
                    step_idx,
                    stage_idx,
                    codec,
                    activation_digest,
                    compute_us,
                    receipt,
                    payload_bytes,
                }
            }
            5 => {
                let session_id = cur.read_u64()?;
                let layer_start = cur.read_u16()?;
                let layer_end = cur.read_u16()?;
                let codec = ActivationCodec::from_u8(cur.read_u8()?)
                    .ok_or_else(|| TransportError::MalformedPayload("unknown codec".into()))?;
                let expected_weights_commitment = cur.read_digest()?;
                let frame_count = cur.read_u32()? as usize;
                let mut frames = Vec::with_capacity(frame_count);
                for _ in 0..frame_count {
                    frames.push(cur.read_bytes()?);
                }
                Self::ReplayKvBatch {
                    session_id,
                    layer_start,
                    layer_end,
                    codec,
                    expected_weights_commitment,
                    frames,
                }
            }
            6 => Self::ReplayKvAck {
                session_id: cur.read_u64()?,
                restored_tokens: cur.read_u32()?,
                rebuild_us: cur.read_u64()?,
            },
            7 => Self::ResetSession {
                session_id: cur.read_u64()?,
            },
            8 => Self::ResetAck {
                session_id: cur.read_u64()?,
            },
            9 => {
                let code = ProtocolErrorCode::from_u8(cur.read_u8()?)
                    .ok_or_else(|| TransportError::MalformedPayload("unknown error code".into()))?;
                let message = cur.read_str()?;
                Self::ErrorFrame { code, message }
            }
            other => {
                return Err(TransportError::MalformedPayload(format!(
                    "unknown frame kind {other}"
                )))
            }
        };
        Ok((frame, 10 + body_len))
    }
}

/// Computes the canonical SHA-256 activation commitment over encoded wire bytes.
#[must_use]
pub fn activation_commitment(payload: &[u8]) -> [u8; DIGEST_LEN] {
    sha256(payload)
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u16).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_bytes(out: &mut Vec<u8>, b: &[u8]) {
    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    out.extend_from_slice(b);
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], TransportError> {
        if self.pos + n > self.bytes.len() {
            return Err(TransportError::MalformedPayload(
                "unexpected end of wire payload".into(),
            ));
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, TransportError> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, TransportError> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, TransportError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, TransportError> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }

    fn read_digest(&mut self) -> Result<[u8; DIGEST_LEN], TransportError> {
        let s = self.take(DIGEST_LEN)?;
        let mut out = [0u8; DIGEST_LEN];
        out.copy_from_slice(s);
        Ok(out)
    }

    fn read_str(&mut self) -> Result<String, TransportError> {
        let len = self.read_u16()? as usize;
        let s = self.take(len)?;
        String::from_utf8(s.to_vec())
            .map_err(|e| TransportError::MalformedPayload(format!("invalid utf8: {e}")))
    }

    fn read_bytes(&mut self) -> Result<Vec<u8>, TransportError> {
        let len = self.read_u32()? as usize;
        Ok(self.take(len)?.to_vec())
    }
}
