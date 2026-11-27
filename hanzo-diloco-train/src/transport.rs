//! Live loopback TCP framing, simulated inter-region WAN latency & bandwidth
//! shaping, and cryptographically signed [`TrainingRoundReceipt`] messages.
//!
//! All worker-to-aggregator traffic flows over real `std::net::TcpStream`
//! sockets using length-prefixed binary frames (`"HZDL"` magic) so transport
//! serialization, timeout handling, abrupt socket teardown on worker churn,
//! and wire byte accounting are directly measured rather than mocked.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::optimizer::{CompressionMode, TrustTier};
use crate::sha256::{ct_eq, hmac_sha256_hex};

/// Wire frame magic prefix (`"HZDL"`).
const FRAME_MAGIC: [u8; 4] = *b"HZDL";
/// Wire protocol version byte.
const FRAME_VERSION: u8 = 1;
/// Receipt schema identifier.
pub const RECEIPT_VERSION: &str = "hanzo.train.receipt/v1";

/// Wire message type tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameKind {
    /// Round work assignment sent to a worker.
    RoundAssignment = 1,
    /// Signed pseudo-gradient update returned by a worker.
    WorkerUpdate = 2,
    /// Request for raw checkpoint bytes from a checkpoint mirror or peer.
    CheckpointRequest = 3,
    /// Raw checkpoint payload returned over TCP.
    CheckpointResponse = 4,
    /// Graceful worker shutdown signal.
    Shutdown = 5,
}

impl FrameKind {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::RoundAssignment),
            2 => Some(Self::WorkerUpdate),
            3 => Some(Self::CheckpointRequest),
            4 => Some(Self::CheckpointResponse),
            5 => Some(Self::Shutdown),
            _ => None,
        }
    }
}

/// Simulated geographic/administrative WAN link characteristics applied to
/// live loopback TCP exchanges.
#[derive(Clone, Debug)]
pub struct WanLinkProfile {
    /// Region label (e.g., `"us-east"`, `"eu-west"`, `"ap-south"`).
    pub region: String,
    /// Base round-trip latency in milliseconds.
    pub rtt_ms: f64,
    /// Sustained link bandwidth in megabits per second.
    pub bandwidth_mbps: f64,
    /// Deterministic jitter amplitude in milliseconds.
    pub jitter_ms: f64,
    /// Wall-clock sleep scale (`1.0` = real 1:1 simulated WAN RTT sleep;
    /// smaller values may be used in rapid unit tests).
    pub sleep_scale: f64,
}

impl WanLinkProfile {
    /// High-speed intra-datacenter LAN baseline (`0.2 ms` RTT, `25 Gbps`).
    #[must_use]
    pub fn lan_centralized() -> Self {
        Self {
            region: "lan-cluster".to_string(),
            rtt_ms: 0.2,
            bandwidth_mbps: 25_000.0,
            jitter_ms: 0.02,
            sleep_scale: 1.0,
        }
    }

    /// US-East independent operator WAN profile (`45 ms` RTT, `200 Mbps`).
    #[must_use]
    pub fn us_east(sleep_scale: f64) -> Self {
        Self {
            region: "us-east".to_string(),
            rtt_ms: 45.0,
            bandwidth_mbps: 200.0,
            jitter_ms: 3.0,
            sleep_scale,
        }
    }

    /// EU-West independent operator WAN profile (`85 ms` RTT, `150 Mbps`).
    #[must_use]
    pub fn eu_west(sleep_scale: f64) -> Self {
        Self {
            region: "eu-west".to_string(),
            rtt_ms: 85.0,
            bandwidth_mbps: 150.0,
            jitter_ms: 5.0,
            sleep_scale,
        }
    }

    /// AP-South independent operator WAN profile (`120 ms` RTT, `100 Mbps`).
    #[must_use]
    pub fn ap_south(sleep_scale: f64) -> Self {
        Self {
            region: "ap-south".to_string(),
            rtt_ms: 120.0,
            bandwidth_mbps: 100.0,
            jitter_ms: 7.0,
            sleep_scale,
        }
    }

    /// Computes one-way propagation + serialization delay in milliseconds for
    /// transmitting `bytes` across this link, and deterministically applies
    /// the scaled real sleep when `sleep_scale > 0.0`.
    #[must_use]
    pub fn one_way_delay_ms(&self, bytes: usize, round_id: u64) -> f64 {
        let half_rtt = self.rtt_ms * 0.5;
        let bits = (bytes as f64) * 8.0;
        let tx_ms = bits / (self.bandwidth_mbps.max(0.1) * 1_000.0);
        // Deterministic signed jitter phase derived from round_id.
        let phase = ((round_id.wrapping_mul(2_654_435_761) % 200) as f64 / 100.0) - 1.0;
        (half_rtt + tx_ms + phase * self.jitter_ms * 0.5).max(0.05)
    }

    /// Sleeps for the scaled one-way WAN delay and returns the unscaled
    /// simulated WAN transit duration in milliseconds.
    pub fn apply_one_way_shaping(&self, bytes: usize, round_id: u64) -> f64 {
        let sim_ms = self.one_way_delay_ms(bytes, round_id);
        let sleep_ms = sim_ms * self.sleep_scale;
        if sleep_ms >= 0.25 {
            std::thread::sleep(Duration::from_secs_f64(sleep_ms / 1_000.0));
        }
        sim_ms
    }
}

/// Cryptographically authenticated worker training round receipt.
#[derive(Clone, Debug)]
pub struct TrainingRoundReceipt {
    /// Receipt schema version (`hanzo.train.receipt/v1`).
    pub receipt_version: String,
    /// Outer synchronization round index.
    pub round_id: u64,
    /// Worker identifier.
    pub worker_id: String,
    /// Worker geographic/administrative region.
    pub region: String,
    /// SHA-256 hex digest of the base checkpoint from which local steps started.
    pub base_checkpoint_sha256: String,
    /// SHA-256 hex digest of the wire-encoded pseudo-gradient payload.
    pub delta_sha256: String,
    /// Number of inner AdamW steps executed locally.
    pub inner_steps_completed: u32,
    /// Total causal next-token positions processed in this round.
    pub tokens_processed: u64,
    /// Local batch loss before the `H` inner steps.
    pub local_loss_before: f64,
    /// Local batch loss after the `H` inner steps.
    pub local_loss_after: f64,
    /// L2 norm of the submitted pseudo-gradient.
    pub delta_l2_norm: f64,
    /// Wire byte length of the encoded pseudo-gradient.
    pub wire_bytes: u64,
    /// Declared trust tier (`signed_delta_and_loss_spot_check`).
    pub trust_tier: TrustTier,
    /// HMAC-SHA256 hex tag authenticating all receipt fields above.
    pub hmac_sha256: String,
}

impl TrainingRoundReceipt {
    /// Canonical byte string covered by `hmac_sha256`.
    #[must_use]
    pub fn signing_payload(&self) -> Vec<u8> {
        format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{:.8}|{:.8}|{:.8}|{}|{}",
            self.receipt_version,
            self.round_id,
            self.worker_id,
            self.region,
            self.base_checkpoint_sha256,
            self.delta_sha256,
            self.inner_steps_completed,
            self.tokens_processed,
            self.local_loss_before,
            self.local_loss_after,
            self.delta_l2_norm,
            self.wire_bytes,
            self.trust_tier.as_str(),
        )
        .into_bytes()
    }

    /// Computes and attaches the HMAC-SHA256 authentication tag.
    pub fn sign(&mut self, worker_key: &[u8]) {
        self.hmac_sha256 = hmac_sha256_hex(worker_key, &self.signing_payload());
    }

    /// Verifies the receipt's HMAC-SHA256 tag in constant time.
    #[must_use]
    pub fn verify_hmac(&self, worker_key: &[u8]) -> bool {
        let expected = hmac_sha256_hex(worker_key, &self.signing_payload());
        ct_eq(expected.as_bytes(), self.hmac_sha256.as_bytes())
    }
}

/// Round assignment payload sent from the peer aggregator to a worker over TCP.
#[derive(Clone, Debug)]
pub struct RoundAssignmentMsg {
    /// Outer round identifier (1-indexed).
    pub round_id: u64,
    /// Base checkpoint SHA-256 hex digest.
    pub base_checkpoint_sha256: String,
    /// Number of inner AdamW steps `H` to execute locally.
    pub inner_steps: u32,
    /// Batch size (number of sequences per inner step).
    pub batch_size: u32,
    /// Inner AdamW learning rate.
    pub inner_lr: f32,
    /// Requested pseudo-gradient compression mode.
    pub compression: CompressionMode,
    /// Shard index assigned to this worker.
    pub shard_idx: u32,
    /// Total number of shards across the training pool.
    pub num_shards: u32,
    /// Base LoRA parameters `theta^(t)` (`f32`).
    pub base_params: Vec<f32>,
}

impl RoundAssignmentMsg {
    /// Serializes the assignment into binary wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(128 + self.base_params.len() * 4);
        out.extend_from_slice(&self.round_id.to_le_bytes());
        write_str(&mut out, &self.base_checkpoint_sha256);
        out.extend_from_slice(&self.inner_steps.to_le_bytes());
        out.extend_from_slice(&self.batch_size.to_le_bytes());
        out.extend_from_slice(&self.inner_lr.to_le_bytes());
        out.push(self.compression.wire_tag());
        out.extend_from_slice(&self.compression.block_size_u16().to_le_bytes());
        out.push(0u8);
        out.extend_from_slice(&self.shard_idx.to_le_bytes());
        out.extend_from_slice(&self.num_shards.to_le_bytes());
        out.extend_from_slice(&(self.base_params.len() as u32).to_le_bytes());
        for &p in &self.base_params {
            out.extend_from_slice(&p.to_le_bytes());
        }
        out
    }

    /// Deserializes a round assignment from wire bytes.
    pub fn decode(buf: &[u8]) -> Result<Self, String> {
        let mut cur = 0usize;
        let round_id = read_u64(buf, &mut cur)?;
        let base_checkpoint_sha256 = read_str(buf, &mut cur)?;
        let inner_steps = read_u32(buf, &mut cur)?;
        let batch_size = read_u32(buf, &mut cur)?;
        let inner_lr = read_f32(buf, &mut cur)?;
        if cur + 4 > buf.len() {
            return Err("truncated compression header".to_string());
        }
        let comp_tag = buf[cur];
        let comp_bs = u16::from_le_bytes([buf[cur + 1], buf[cur + 2]]);
        cur += 4;
        let compression = CompressionMode::from_wire(comp_tag, comp_bs)
            .ok_or_else(|| format!("invalid compression mode {comp_tag}/{comp_bs}"))?;
        let shard_idx = read_u32(buf, &mut cur)?;
        let num_shards = read_u32(buf, &mut cur)?;
        let p_len = read_u32(buf, &mut cur)? as usize;
        let mut base_params = Vec::with_capacity(p_len);
        for _ in 0..p_len {
            base_params.push(read_f32(buf, &mut cur)?);
        }
        Ok(Self {
            round_id,
            base_checkpoint_sha256,
            inner_steps,
            batch_size,
            inner_lr,
            compression,
            shard_idx,
            num_shards,
            base_params,
        })
    }
}

/// Worker response payload carrying the signed receipt, encoded pseudo-gradient,
/// and measured worker compute/WAN telemetry.
#[derive(Clone, Debug)]
pub struct WorkerUpdateMsg {
    /// Cryptographically signed training round receipt.
    pub receipt: TrainingRoundReceipt,
    /// Wire-encoded pseudo-gradient bytes (FP32 or INT8).
    pub encoded_delta: Vec<u8>,
    /// Measured worker local compute wall time in milliseconds.
    pub compute_ms: f64,
    /// Simulated WAN transit time experienced by this worker in milliseconds.
    pub simulated_wan_ms: f64,
}

impl WorkerUpdateMsg {
    /// Serializes the worker update into binary wire bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(256 + self.encoded_delta.len());
        write_str(&mut out, &self.receipt.receipt_version);
        out.extend_from_slice(&self.receipt.round_id.to_le_bytes());
        write_str(&mut out, &self.receipt.worker_id);
        write_str(&mut out, &self.receipt.region);
        write_str(&mut out, &self.receipt.base_checkpoint_sha256);
        write_str(&mut out, &self.receipt.delta_sha256);
        out.extend_from_slice(&self.receipt.inner_steps_completed.to_le_bytes());
        out.extend_from_slice(&self.receipt.tokens_processed.to_le_bytes());
        out.extend_from_slice(&self.receipt.local_loss_before.to_le_bytes());
        out.extend_from_slice(&self.receipt.local_loss_after.to_le_bytes());
        out.extend_from_slice(&self.receipt.delta_l2_norm.to_le_bytes());
        out.extend_from_slice(&self.receipt.wire_bytes.to_le_bytes());
        write_str(&mut out, &self.receipt.hmac_sha256);
        out.extend_from_slice(&self.compute_ms.to_le_bytes());
        out.extend_from_slice(&self.simulated_wan_ms.to_le_bytes());
        out.extend_from_slice(&(self.encoded_delta.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.encoded_delta);
        out
    }

    /// Deserializes a worker update from wire bytes.
    pub fn decode(buf: &[u8]) -> Result<Self, String> {
        let mut cur = 0usize;
        let receipt_version = read_str(buf, &mut cur)?;
        let round_id = read_u64(buf, &mut cur)?;
        let worker_id = read_str(buf, &mut cur)?;
        let region = read_str(buf, &mut cur)?;
        let base_checkpoint_sha256 = read_str(buf, &mut cur)?;
        let delta_sha256 = read_str(buf, &mut cur)?;
        let inner_steps_completed = read_u32(buf, &mut cur)?;
        let tokens_processed = read_u64(buf, &mut cur)?;
        let local_loss_before = read_f64(buf, &mut cur)?;
        let local_loss_after = read_f64(buf, &mut cur)?;
        let delta_l2_norm = read_f64(buf, &mut cur)?;
        let wire_bytes = read_u64(buf, &mut cur)?;
        let hmac_sha256 = read_str(buf, &mut cur)?;
        let compute_ms = read_f64(buf, &mut cur)?;
        let simulated_wan_ms = read_f64(buf, &mut cur)?;
        let d_len = read_u32(buf, &mut cur)? as usize;
        if cur + d_len > buf.len() {
            return Err("truncated encoded_delta payload".to_string());
        }
        let encoded_delta = buf[cur..cur + d_len].to_vec();

        Ok(Self {
            receipt: TrainingRoundReceipt {
                receipt_version,
                round_id,
                worker_id,
                region,
                base_checkpoint_sha256,
                delta_sha256,
                inner_steps_completed,
                tokens_processed,
                local_loss_before,
                local_loss_after,
                delta_l2_norm,
                wire_bytes,
                trust_tier: TrustTier::SignedDeltaAndLossSpotCheck,
                hmac_sha256,
            },
            encoded_delta,
            compute_ms,
            simulated_wan_ms,
        })
    }
}

/// Writes a framed binary message over a live `TcpStream`, returning the total
/// wire bytes written (10-byte header + payload).
pub fn write_frame(
    stream: &mut TcpStream,
    kind: FrameKind,
    payload: &[u8],
) -> Result<usize, String> {
    let mut header = [0u8; 10];
    header[0..4].copy_from_slice(&FRAME_MAGIC);
    header[4] = FRAME_VERSION;
    header[5] = kind as u8;
    header[6..10].copy_from_slice(&(payload.len() as u32).to_be_bytes());
    stream
        .write_all(&header)
        .map_err(|e| format!("tcp write header failed: {e}"))?;
    stream
        .write_all(payload)
        .map_err(|e| format!("tcp write payload failed: {e}"))?;
    stream
        .flush()
        .map_err(|e| format!("tcp flush failed: {e}"))?;
    Ok(header.len() + payload.len())
}

/// Reads a framed binary message from a live `TcpStream`, returning
/// `(kind, payload, total_wire_bytes)`.
pub fn read_frame(stream: &mut TcpStream) -> Result<(FrameKind, Vec<u8>, usize), String> {
    let mut header = [0u8; 10];
    stream
        .read_exact(&mut header)
        .map_err(|e| format!("tcp read header failed: {e}"))?;
    if header[0..4] != FRAME_MAGIC {
        return Err("bad wire frame magic".to_string());
    }
    if header[4] != FRAME_VERSION {
        return Err(format!("unsupported wire version {}", header[4]));
    }
    let kind =
        FrameKind::from_u8(header[5]).ok_or_else(|| format!("unknown frame kind {}", header[5]))?;
    let len = u32::from_be_bytes([header[6], header[7], header[8], header[9]]) as usize;
    if len > 64 * 1024 * 1024 {
        return Err(format!("frame exceeds 64 MiB limit: {len}"));
    }
    let mut payload = vec![0u8; len];
    stream
        .read_exact(&mut payload)
        .map_err(|e| format!("tcp read payload failed: {e}"))?;
    Ok((kind, payload, 10 + len))
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn read_u32(buf: &[u8], cur: &mut usize) -> Result<u32, String> {
    if *cur + 4 > buf.len() {
        return Err("unexpected EOF reading u32".to_string());
    }
    let v = u32::from_le_bytes([buf[*cur], buf[*cur + 1], buf[*cur + 2], buf[*cur + 3]]);
    *cur += 4;
    Ok(v)
}

fn read_u64(buf: &[u8], cur: &mut usize) -> Result<u64, String> {
    if *cur + 8 > buf.len() {
        return Err("unexpected EOF reading u64".to_string());
    }
    let v = u64::from_le_bytes([
        buf[*cur],
        buf[*cur + 1],
        buf[*cur + 2],
        buf[*cur + 3],
        buf[*cur + 4],
        buf[*cur + 5],
        buf[*cur + 6],
        buf[*cur + 7],
    ]);
    *cur += 8;
    Ok(v)
}

fn read_f32(buf: &[u8], cur: &mut usize) -> Result<f32, String> {
    Ok(f32::from_bits(read_u32(buf, cur)?))
}

fn read_f64(buf: &[u8], cur: &mut usize) -> Result<f64, String> {
    Ok(f64::from_bits(read_u64(buf, cur)?))
}

fn read_str(buf: &[u8], cur: &mut usize) -> Result<String, String> {
    let len = read_u32(buf, cur)? as usize;
    if *cur + len > buf.len() {
        return Err("unexpected EOF reading string".to_string());
    }
    let s = std::str::from_utf8(&buf[*cur..*cur + len])
        .map_err(|_| "invalid UTF-8 in wire string".to_string())?
        .to_string();
    *cur += len;
    Ok(s)
}
