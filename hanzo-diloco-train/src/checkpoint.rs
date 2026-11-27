//! Content-addressed VFS checkpoint persistence, cryptographic digest
//! verification, corruption-resilient rollback, and versioned Hanzo stack
//! integration manifest (`hanzo.train.elastic/v1`).
//!
//! Every outer synchronization round writes an immutable content-addressed
//! checkpoint snapshot identified by `vfs://checkpoints/<sha256>`. Because
//! checkpoints are self-verifying and portable across peers, any independent
//! operator can resume training or migrate adapters without a Hanzo-operated
//! coordinator (`hanzo_coordinator_required = false`).

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::sha256::{ct_eq, hmac_sha256_hex, sha256_hex};

/// Protocol schema version for Hanzo elastic training manifests and checkpoints.
pub const HANZO_ELASTIC_TRAIN_PROTOCOL: &str = "hanzo.train.elastic/v1";

/// Binary checkpoint file magic prefix (`"HZCK"`).
const CHECKPOINT_MAGIC: [u8; 4] = *b"HZCK";

/// Errors returned by the content-addressed VFS checkpoint store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointError {
    /// Filesystem I/O failure.
    Io(String),
    /// Checkpoint file header or binary layout was invalid or truncated.
    MalformedFormat(String),
    /// Recomputed SHA-256 did not match the addressed checkpoint digest.
    DigestMismatch {
        /// Expected SHA-256 hex digest.
        expected: String,
        /// Actual SHA-256 hex digest computed from payload bytes.
        actual: String,
    },
    /// Checkpoint HMAC-SHA256 authentication tag failed verification.
    InvalidSignature,
    /// No verified checkpoint was available in the store.
    NoVerifiedCheckpoint,
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "checkpoint I/O error: {msg}"),
            Self::MalformedFormat(msg) => write!(f, "malformed checkpoint: {msg}"),
            Self::DigestMismatch { expected, actual } => {
                write!(
                    f,
                    "checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidSignature => write!(f, "checkpoint HMAC signature verification failed"),
            Self::NoVerifiedCheckpoint => write!(f, "no verified checkpoint available"),
        }
    }
}

impl std::error::Error for CheckpointError {}

/// Immutable training checkpoint snapshot stored in the VFS checkpoint store.
#[derive(Clone, Debug)]
pub struct TrainingCheckpoint {
    /// Protocol version (`hanzo.train.elastic/v1`).
    pub schema_version: String,
    /// Outer synchronization round index (0 = initial base adapter state).
    pub round_id: u64,
    /// Cumulative tokens trained up to this checkpoint.
    pub total_tokens_trained: u64,
    /// SHA-256 hex digest of the parent checkpoint (or `"genesis"` at round 0).
    pub parent_checkpoint_sha256: String,
    /// SHA-256 hex digest of [`crate::model::SmallZenConfig`].
    pub model_config_sha256: String,
    /// Validation cross-entropy loss recorded at this checkpoint.
    pub val_loss: f64,
    /// Validation perplexity recorded at this checkpoint.
    pub val_perplexity: f64,
    /// Flat trainable LoRA parameter vector (`f32`).
    pub params: Vec<f32>,
    /// Outer Nesterov momentum velocity vector (`f32`).
    pub outer_velocity: Vec<f32>,
    /// Content-addressed SHA-256 hex digest over canonical payload bytes.
    pub checkpoint_sha256: String,
    /// HMAC-SHA256 hex authentication tag over `checkpoint_sha256`.
    pub receipt_hmac_sha256: String,
}

impl TrainingCheckpoint {
    /// Canonical VFS URI (`vfs://checkpoints/<sha256>`).
    #[must_use]
    pub fn vfs_uri(&self) -> String {
        format!("vfs://checkpoints/{}", self.checkpoint_sha256)
    }

    /// Creates and signs a new checkpoint snapshot.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn build_signed(
        round_id: u64,
        total_tokens_trained: u64,
        parent_checkpoint_sha256: String,
        model_config_sha256: String,
        val_loss: f64,
        val_perplexity: f64,
        params: Vec<f32>,
        outer_velocity: Vec<f32>,
        signing_key: &[u8],
    ) -> Self {
        let mut ckpt = Self {
            schema_version: HANZO_ELASTIC_TRAIN_PROTOCOL.to_string(),
            round_id,
            total_tokens_trained,
            parent_checkpoint_sha256,
            model_config_sha256,
            val_loss,
            val_perplexity,
            params,
            outer_velocity,
            checkpoint_sha256: String::new(),
            receipt_hmac_sha256: String::new(),
        };
        let payload = ckpt.encode_canonical_payload();
        ckpt.checkpoint_sha256 = sha256_hex(&payload);
        ckpt.receipt_hmac_sha256 = hmac_sha256_hex(signing_key, ckpt.checkpoint_sha256.as_bytes());
        ckpt
    }

    /// Encodes the canonical binary payload covered by `checkpoint_sha256`.
    #[must_use]
    pub fn encode_canonical_payload(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(128 + self.params.len() * 4 + self.outer_velocity.len() * 4);
        out.extend_from_slice(&CHECKPOINT_MAGIC);
        write_str(&mut out, &self.schema_version);
        out.extend_from_slice(&self.round_id.to_le_bytes());
        out.extend_from_slice(&self.total_tokens_trained.to_le_bytes());
        write_str(&mut out, &self.parent_checkpoint_sha256);
        write_str(&mut out, &self.model_config_sha256);
        out.extend_from_slice(&self.val_loss.to_le_bytes());
        out.extend_from_slice(&self.val_perplexity.to_le_bytes());
        out.extend_from_slice(&(self.params.len() as u32).to_le_bytes());
        for &p in &self.params {
            out.extend_from_slice(&p.to_le_bytes());
        }
        out.extend_from_slice(&(self.outer_velocity.len() as u32).to_le_bytes());
        for &v in &self.outer_velocity {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Encodes full disk/wire representation: canonical payload + 64-byte ASCII
    /// `checkpoint_sha256` + 64-byte ASCII `receipt_hmac_sha256`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let payload = self.encode_canonical_payload();
        let mut out = Vec::with_capacity(4 + payload.len() + 128);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&payload);
        out.extend_from_slice(self.checkpoint_sha256.as_bytes());
        out.extend_from_slice(self.receipt_hmac_sha256.as_bytes());
        out
    }

    /// Decodes and cryptographically verifies a checkpoint from raw bytes.
    pub fn from_bytes_verified(
        bytes: &[u8],
        expected_sha256: Option<&str>,
        signing_key: &[u8],
    ) -> Result<Self, CheckpointError> {
        if bytes.len() < 4 + 128 {
            return Err(CheckpointError::MalformedFormat(
                "checkpoint shorter than minimum envelope".to_string(),
            ));
        }
        let payload_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        if bytes.len() != 4 + payload_len + 128 {
            return Err(CheckpointError::MalformedFormat(format!(
                "checkpoint length mismatch: header declares payload {payload_len}, total bytes {}",
                bytes.len()
            )));
        }
        let payload = &bytes[4..4 + payload_len];
        let stored_sha = std::str::from_utf8(&bytes[4 + payload_len..4 + payload_len + 64])
            .map_err(|_| CheckpointError::MalformedFormat("invalid UTF-8 sha256".to_string()))?;
        let stored_hmac = std::str::from_utf8(&bytes[4 + payload_len + 64..])
            .map_err(|_| CheckpointError::MalformedFormat("invalid UTF-8 hmac".to_string()))?;

        let actual_sha = sha256_hex(payload);
        if !ct_eq(actual_sha.as_bytes(), stored_sha.as_bytes()) {
            return Err(CheckpointError::DigestMismatch {
                expected: stored_sha.to_string(),
                actual: actual_sha,
            });
        }
        if let Some(exp) = expected_sha256 {
            if !ct_eq(actual_sha.as_bytes(), exp.as_bytes()) {
                return Err(CheckpointError::DigestMismatch {
                    expected: exp.to_string(),
                    actual: actual_sha,
                });
            }
        }
        let expected_hmac = hmac_sha256_hex(signing_key, actual_sha.as_bytes());
        if !ct_eq(expected_hmac.as_bytes(), stored_hmac.as_bytes()) {
            return Err(CheckpointError::InvalidSignature);
        }

        // Parse payload
        let mut cursor = 0usize;
        if payload.len() < 4 || payload[..4] != CHECKPOINT_MAGIC {
            return Err(CheckpointError::MalformedFormat(
                "bad checkpoint magic".to_string(),
            ));
        }
        cursor += 4;
        let schema_version = read_str(payload, &mut cursor)?;
        let round_id = read_u64(payload, &mut cursor)?;
        let total_tokens_trained = read_u64(payload, &mut cursor)?;
        let parent_checkpoint_sha256 = read_str(payload, &mut cursor)?;
        let model_config_sha256 = read_str(payload, &mut cursor)?;
        let val_loss = read_f64(payload, &mut cursor)?;
        let val_perplexity = read_f64(payload, &mut cursor)?;
        let params_len = read_u32(payload, &mut cursor)? as usize;
        let mut params = Vec::with_capacity(params_len);
        for _ in 0..params_len {
            params.push(read_f32(payload, &mut cursor)?);
        }
        let vel_len = read_u32(payload, &mut cursor)? as usize;
        let mut outer_velocity = Vec::with_capacity(vel_len);
        for _ in 0..vel_len {
            outer_velocity.push(read_f32(payload, &mut cursor)?);
        }
        if cursor != payload.len() {
            return Err(CheckpointError::MalformedFormat(
                "trailing bytes in checkpoint payload".to_string(),
            ));
        }

        Ok(Self {
            schema_version,
            round_id,
            total_tokens_trained,
            parent_checkpoint_sha256,
            model_config_sha256,
            val_loss,
            val_perplexity,
            params,
            outer_velocity,
            checkpoint_sha256: actual_sha,
            receipt_hmac_sha256: stored_hmac.to_string(),
        })
    }
}

/// Content-addressed VFS checkpoint directory store supporting immutable
/// writes, SHA-256 + HMAC verification, and automatic fallback to the newest
/// intact checkpoint if a file is corrupted on disk.
#[derive(Clone, Debug)]
pub struct VfsCheckpointStore {
    root_dir: PathBuf,
    signing_key: Vec<u8>,
    lineage: Vec<String>,
}

impl VfsCheckpointStore {
    /// Initializes a VFS checkpoint directory at `root_dir`.
    pub fn new(root_dir: PathBuf, signing_key: Vec<u8>) -> Result<Self, CheckpointError> {
        fs::create_dir_all(&root_dir).map_err(|e| CheckpointError::Io(e.to_string()))?;
        Ok(Self {
            root_dir,
            signing_key,
            lineage: Vec::new(),
        })
    }

    /// Returns the filesystem path for a content-addressed checkpoint digest.
    #[must_use]
    pub fn path_for_digest(&self, sha256_hex: &str) -> PathBuf {
        self.root_dir.join(format!("{sha256_hex}.hzck"))
    }

    /// Returns the recorded chronological checkpoint digest lineage.
    #[must_use]
    pub fn lineage(&self) -> &[String] {
        &self.lineage
    }

    /// Writes a signed checkpoint to disk under `<digest>.hzck` and records its
    /// digest in the store lineage, returning `(vfs_uri, bytes_written)`.
    pub fn save_checkpoint(
        &mut self,
        checkpoint: &TrainingCheckpoint,
    ) -> Result<(String, usize), CheckpointError> {
        let bytes = checkpoint.to_bytes();
        let path = self.path_for_digest(&checkpoint.checkpoint_sha256);
        fs::write(&path, &bytes).map_err(|e| CheckpointError::Io(e.to_string()))?;
        self.lineage.push(checkpoint.checkpoint_sha256.clone());
        Ok((checkpoint.vfs_uri(), bytes.len()))
    }

    /// Loads and cryptographically verifies a specific checkpoint by SHA-256.
    pub fn load_verified(&self, sha256_hex: &str) -> Result<TrainingCheckpoint, CheckpointError> {
        let path = self.path_for_digest(sha256_hex);
        let bytes = fs::read(&path).map_err(|e| CheckpointError::Io(e.to_string()))?;
        TrainingCheckpoint::from_bytes_verified(&bytes, Some(sha256_hex), &self.signing_key)
    }

    /// Loads raw checkpoint bytes (used by peer checkpoint sync over TCP).
    pub fn load_raw_bytes(&self, sha256_hex: &str) -> Result<Vec<u8>, CheckpointError> {
        let path = self.path_for_digest(sha256_hex);
        fs::read(&path).map_err(|e| CheckpointError::Io(e.to_string()))
    }

    /// Walks the checkpoint lineage from newest to oldest, rejecting corrupted
    /// files and automatically falling back to the latest verified checkpoint.
    /// Returns `(checkpoint, rejected_corrupt_digests)`.
    pub fn load_latest_verified_with_fallback(
        &self,
    ) -> Result<(TrainingCheckpoint, Vec<String>), CheckpointError> {
        let mut rejected = Vec::new();
        for digest in self.lineage.iter().rev() {
            match self.load_verified(digest) {
                Ok(ckpt) => return Ok((ckpt, rejected)),
                Err(_) => rejected.push(digest.clone()),
            }
        }
        Err(CheckpointError::NoVerifiedCheckpoint)
    }

    /// Returns the store directory path.
    #[must_use]
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }
}

/// Enso peer routing entry for a decentralized training worker.
#[derive(Clone, Debug)]
pub struct EnsoPeerRoute {
    /// Worker node identifier.
    pub worker_id: String,
    /// Geographic or administrative region label.
    pub region: String,
    /// Enso peer multiaddress / socket endpoint.
    pub endpoint: String,
    /// Declared inter-region WAN round-trip time (ms).
    pub wan_rtt_ms: f64,
    /// Declared WAN link bandwidth (Mbps).
    pub wan_bandwidth_mbps: f64,
}

/// Visor worker sandbox isolation metadata.
#[derive(Clone, Debug)]
pub struct VisorIsolationSpec {
    /// Worker isolation profile name.
    pub profile: String,
    /// Memory ceiling per worker sandbox in MiB.
    pub memory_limit_mib: u64,
    /// Whether outbound network traffic is restricted to declared Enso peers
    /// and VFS checkpoint mirrors.
    pub egress_allowlist_enforced: bool,
}

/// Versioned Hanzo elastic training manifest (`hanzo.train.elastic/v1`)
/// integrating Enso peer routing, Visor isolation, and VFS checkpoints without
/// any mandatory Hanzo-operated coordinator.
#[derive(Clone, Debug)]
pub struct HanzoTrainingManifest {
    /// Versioned protocol identifier (`hanzo.train.elastic/v1`).
    pub protocol_version: String,
    /// Model configuration SHA-256 fingerprint.
    pub model_config_sha256: String,
    /// Grounded dataset SHA-256 fingerprint.
    pub corpus_sha256: String,
    /// Declared Enso peer routing table.
    pub enso_peer_routes: Vec<EnsoPeerRoute>,
    /// Visor sandbox isolation policy for worker processes.
    pub visor_isolation: VisorIsolationSpec,
    /// VFS immutable checkpoint URI prefix (`vfs://checkpoints/`).
    pub vfs_checkpoint_uri_prefix: String,
    /// Decentralization invariant: must remain `false` so independent operators
    /// can elect any peer or mirror checkpoints without Hanzo infrastructure.
    pub hanzo_coordinator_required: bool,
    /// Exportable open adapter format identifier for sovereignty/forking.
    pub exportable_adapter_format: String,
}

impl HanzoTrainingManifest {
    /// Builds a canonical `hanzo.train.elastic/v1` manifest.
    #[must_use]
    pub fn new(
        model_config_sha256: String,
        corpus_sha256: String,
        enso_peer_routes: Vec<EnsoPeerRoute>,
    ) -> Self {
        Self {
            protocol_version: HANZO_ELASTIC_TRAIN_PROTOCOL.to_string(),
            model_config_sha256,
            corpus_sha256,
            enso_peer_routes,
            visor_isolation: VisorIsolationSpec {
                profile: "visor-sandbox-cpu-v1".to_string(),
                memory_limit_mib: 512,
                egress_allowlist_enforced: true,
            },
            vfs_checkpoint_uri_prefix: "vfs://checkpoints/".to_string(),
            hanzo_coordinator_required: false,
            exportable_adapter_format: "hanzo-lora-v1".to_string(),
        }
    }
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn read_u32(buf: &[u8], cursor: &mut usize) -> Result<u32, CheckpointError> {
    if *cursor + 4 > buf.len() {
        return Err(CheckpointError::MalformedFormat(
            "unexpected EOF u32".to_string(),
        ));
    }
    let v = u32::from_le_bytes([
        buf[*cursor],
        buf[*cursor + 1],
        buf[*cursor + 2],
        buf[*cursor + 3],
    ]);
    *cursor += 4;
    Ok(v)
}

fn read_u64(buf: &[u8], cursor: &mut usize) -> Result<u64, CheckpointError> {
    if *cursor + 8 > buf.len() {
        return Err(CheckpointError::MalformedFormat(
            "unexpected EOF u64".to_string(),
        ));
    }
    let v = u64::from_le_bytes([
        buf[*cursor],
        buf[*cursor + 1],
        buf[*cursor + 2],
        buf[*cursor + 3],
        buf[*cursor + 4],
        buf[*cursor + 5],
        buf[*cursor + 6],
        buf[*cursor + 7],
    ]);
    *cursor += 8;
    Ok(v)
}

fn read_f32(buf: &[u8], cursor: &mut usize) -> Result<f32, CheckpointError> {
    Ok(f32::from_bits(read_u32(buf, cursor)?))
}

fn read_f64(buf: &[u8], cursor: &mut usize) -> Result<f64, CheckpointError> {
    Ok(f64::from_bits(read_u64(buf, cursor)?))
}

fn read_str(buf: &[u8], cursor: &mut usize) -> Result<String, CheckpointError> {
    let len = read_u32(buf, cursor)? as usize;
    if *cursor + len > buf.len() {
        return Err(CheckpointError::MalformedFormat(
            "unexpected EOF string".to_string(),
        ));
    }
    let s = std::str::from_utf8(&buf[*cursor..*cursor + len])
        .map_err(|_| CheckpointError::MalformedFormat("invalid UTF-8 string".to_string()))?
        .to_string();
    *cursor += len;
    Ok(s)
}
