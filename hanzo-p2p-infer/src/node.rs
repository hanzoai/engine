// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Live independent multi-socket TCP worker node (`WorkerNode`).
//!
//! Each worker binds a real loopback `TcpListener` on its own OS ephemeral port,
//! executes deterministic Transformer + MoE `LayerSlice` forward passes with
//! per-session KV caches, signs every response with an HMAC-SHA256 `StageReceipt`,
//! and supports genuine `hard_kill()` socket teardown for disconnect failover tests.

use crate::model::{LayerSlice, MicroModelConfig, StageKvCache, WeightsCommitment};
use crate::sha256::constant_time_eq;
use crate::transport::{
    activation_commitment, ProtocolErrorCode, StageReceipt, TransportError, WireFrame,
};
use std::collections::HashMap;
use std::io;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Adversarial fault injection mode for trust, tamper-detection, and failover testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FaultMode {
    /// Honest execution (default).
    #[default]
    None,
    /// Flips a byte in the outgoing activation payload after computing its digest
    /// (caught immediately by wire SHA-256 integrity verification).
    TamperActivationPayload,
    /// Perturbs outgoing hidden-state values *before* encoding, hashing, and signing
    /// (simulates a malicious peer with valid transport framing; caught by spot-check challenge).
    SilentNumericTamper,
    /// Reports a corrupted weights commitment digest in the response receipt
    /// (caught by `WeightsCommitment` verification).
    WrongWeightsCommitment,
    /// Corrupts one byte of the HMAC-SHA256 tag in `StageReceipt`
    /// (caught by constant-time MAC verification).
    ForgedReceiptMac,
    /// Replays a receipt bound to a different `session_id`
    /// (caught by session/step receipt binding verification).
    ReplayReceiptFromSession(u64),
    /// Sleeps for the specified duration in milliseconds before replying
    /// (simulates a stalled/hung peer to verify socket timeout and failover).
    StallMillis(u64),
}

struct NodeSharedState {
    peer_id: String,
    peer_secret: Vec<u8>,
    model_cfg: MicroModelConfig,
    max_concurrent_sessions: u32,
    slices: Mutex<HashMap<(usize, usize), LayerSlice>>,
    kv_caches: Mutex<HashMap<(u64, usize, usize), StageKvCache>>,
    active_streams: Mutex<Vec<TcpStream>>,
    fault_mode: Mutex<FaultMode>,
    killed: AtomicBool,
    requests_served: AtomicU64,
    bytes_in: AtomicU64,
    bytes_out: AtomicU64,
}

/// Live independent TCP peer worker serving one or more contiguous model layer slices.
pub struct WorkerNode {
    addr: SocketAddr,
    shared: Arc<NodeSharedState>,
    listener_thread: Option<JoinHandle<()>>,
}

impl WorkerNode {
    /// Spawns a live worker node listening on `127.0.0.1:0` hosting `initial_slices`.
    pub fn spawn(
        peer_id: impl Into<String>,
        peer_secret: Vec<u8>,
        model_cfg: MicroModelConfig,
        initial_slices: Vec<(usize, usize)>,
        max_concurrent_sessions: u32,
    ) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;

        let mut slice_map = HashMap::new();
        for (start, end) in initial_slices {
            let slice = LayerSlice::from_config(&model_cfg, start, end);
            slice_map.insert((start, end), slice);
        }

        let shared = Arc::new(NodeSharedState {
            peer_id: peer_id.into(),
            peer_secret,
            model_cfg,
            max_concurrent_sessions,
            slices: Mutex::new(slice_map),
            kv_caches: Mutex::new(HashMap::new()),
            active_streams: Mutex::new(Vec::new()),
            fault_mode: Mutex::new(FaultMode::None),
            killed: AtomicBool::new(false),
            requests_served: AtomicU64::new(0),
            bytes_in: AtomicU64::new(0),
            bytes_out: AtomicU64::new(0),
        });

        let state_bg = Arc::clone(&shared);
        let handle = thread::spawn(move || {
            accept_loop(listener, state_bg);
        });

        Ok(Self {
            addr,
            shared,
            listener_thread: Some(handle),
        })
    }

    /// Returns the live TCP socket address (`127.0.0.1:<port>`) bound by this node.
    #[must_use]
    pub const fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Returns this peer's identifier.
    #[must_use]
    pub fn peer_id(&self) -> &str {
        &self.shared.peer_id
    }

    /// Returns this peer's HMAC receipt secret (used by orchestrators/verifiers in local tests).
    #[must_use]
    pub fn peer_secret(&self) -> &[u8] {
        &self.shared.peer_secret
    }

    /// Returns the SHA-256 [`WeightsCommitment`] for a hosted slice `[layer_start, layer_end)`.
    #[must_use]
    pub fn slice_commitment(
        &self,
        layer_start: usize,
        layer_end: usize,
    ) -> Option<WeightsCommitment> {
        let guard = self.shared.slices.lock().ok()?;
        guard.get(&(layer_start, layer_end)).map(|s| s.commitment)
    }

    /// Pre-loads or dynamically mounts a contiguous layer slice `[layer_start, layer_end)`
    /// (modeling VFS shard fetch during standby promotion / rebalance).
    pub fn mount_slice(&self, layer_start: usize, layer_end: usize) -> WeightsCommitment {
        let slice = LayerSlice::from_config(&self.shared.model_cfg, layer_start, layer_end);
        let commitment = slice.commitment;
        if let Ok(mut guard) = self.shared.slices.lock() {
            guard.insert((layer_start, layer_end), slice);
        }
        commitment
    }

    /// Sets the active adversarial [`FaultMode`] for trust/security testing.
    pub fn set_fault_mode(&self, mode: FaultMode) {
        if let Ok(mut guard) = self.shared.fault_mode.lock() {
            *guard = mode;
        }
    }

    /// Returns whether this node has been hard-killed.
    #[must_use]
    pub fn is_killed(&self) -> bool {
        self.shared.killed.load(Ordering::SeqCst)
    }

    /// Returns total forward/replay requests served by this node.
    #[must_use]
    pub fn requests_served(&self) -> u64 {
        self.shared.requests_served.load(Ordering::Relaxed)
    }

    /// Returns total wire bytes received by this node.
    #[must_use]
    pub fn bytes_in(&self) -> u64 {
        self.shared.bytes_in.load(Ordering::Relaxed)
    }

    /// Returns total wire bytes sent by this node.
    #[must_use]
    pub fn bytes_out(&self) -> u64 {
        self.shared.bytes_out.load(Ordering::Relaxed)
    }

    /// Simulates an abrupt node crash / network disconnect: marks the node dead,
    /// immediately shuts down every active TCP socket (`Shutdown::Both`), and joins
    /// the listener thread so the listening socket is genuinely closed by the OS.
    pub fn hard_kill(&mut self) {
        self.shared.killed.store(true, Ordering::SeqCst);
        if let Ok(mut streams) = self.shared.active_streams.lock() {
            for stream in streams.drain(..) {
                let _ = stream.shutdown(Shutdown::Both);
            }
        }
        if let Some(handle) = self.listener_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for WorkerNode {
    fn drop(&mut self) {
        self.hard_kill();
    }
}

fn accept_loop(listener: TcpListener, state: Arc<NodeSharedState>) {
    while !state.killed.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _peer_addr)) => {
                if state.killed.load(Ordering::SeqCst) {
                    let _ = stream.shutdown(Shutdown::Both);
                    break;
                }
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_nodelay(true);
                let _ = stream.set_read_timeout(Some(Duration::from_millis(200)));
                let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));

                if let Ok(dup) = stream.try_clone() {
                    if let Ok(mut list) = state.active_streams.lock() {
                        list.push(dup);
                    }
                }

                let conn_state = Arc::clone(&state);
                thread::spawn(move || {
                    handle_connection(stream, conn_state);
                });
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(1));
            }
            Err(_) => {
                break;
            }
        }
    }
    // Dropping `listener` here closes the OS listening TCP port immediately.
}

fn handle_connection(mut stream: TcpStream, state: Arc<NodeSharedState>) {
    loop {
        if state.killed.load(Ordering::SeqCst) {
            let _ = stream.shutdown(Shutdown::Both);
            return;
        }

        let (frame, n_in) = match WireFrame::read_from_stream(&mut stream) {
            Ok(v) => v,
            Err(TransportError::Io(ref e))
                if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut =>
            {
                continue;
            }
            Err(_) => return,
        };

        state.bytes_in.fetch_add(n_in as u64, Ordering::Relaxed);
        if state.killed.load(Ordering::SeqCst) {
            let _ = stream.shutdown(Shutdown::Both);
            return;
        }

        let response = dispatch_frame(frame, &state);
        if state.killed.load(Ordering::SeqCst) {
            let _ = stream.shutdown(Shutdown::Both);
            return;
        }

        match response.write_to_stream(&mut stream) {
            Ok(n_out) => {
                state.bytes_out.fetch_add(n_out as u64, Ordering::Relaxed);
            }
            Err(_) => return,
        }
    }
}

fn dispatch_frame(frame: WireFrame, state: &NodeSharedState) -> WireFrame {
    let fault = state
        .fault_mode
        .lock()
        .map(|g| *g)
        .unwrap_or(FaultMode::None);

    if let FaultMode::StallMillis(ms) = fault {
        let mut remaining = ms;
        while remaining > 0 && !state.killed.load(Ordering::SeqCst) {
            let step = remaining.min(10);
            thread::sleep(Duration::from_millis(step));
            remaining -= step;
        }
    }

    match frame {
        WireFrame::HealthPing { nonce } => {
            let hosted_slices = state
                .slices
                .lock()
                .map(|g| {
                    let mut v: Vec<(u16, u16)> =
                        g.keys().map(|&(s, e)| (s as u16, e as u16)).collect();
                    v.sort_unstable();
                    v
                })
                .unwrap_or_default();
            let active_sessions = state.kv_caches.lock().map(|g| g.len() as u32).unwrap_or(0);
            WireFrame::HealthPong {
                nonce,
                peer_id: state.peer_id.clone(),
                active_sessions,
                hosted_slices,
            }
        }
        WireFrame::ResetSession { session_id } => {
            if let Ok(mut kv) = state.kv_caches.lock() {
                kv.retain(|&(sid, _, _), _| sid != session_id);
            }
            WireFrame::ResetAck { session_id }
        }
        WireFrame::ReplayKvBatch {
            session_id,
            layer_start,
            layer_end,
            codec,
            expected_weights_commitment,
            frames,
        } => {
            let t0 = Instant::now();
            let key = (layer_start as usize, layer_end as usize);

            // Dynamically mount shard on demand if the peer does not yet have it mounted
            // (models pulling content-addressed VFS shard on failover promotion).
            let slice = {
                let mut guard = match state.slices.lock() {
                    Ok(g) => g,
                    Err(_) => {
                        return WireFrame::ErrorFrame {
                            code: ProtocolErrorCode::MalformedFrame,
                            message: "slice lock poisoned".into(),
                        }
                    }
                };
                guard
                    .entry(key)
                    .or_insert_with(|| LayerSlice::from_config(&state.model_cfg, key.0, key.1))
                    .clone()
            };

            if !constant_time_eq(&slice.commitment.digest, &expected_weights_commitment) {
                return WireFrame::ErrorFrame {
                    code: ProtocolErrorCode::WeightsCommitmentMismatch,
                    message: "weights commitment mismatch during KV replay".into(),
                };
            }

            let mut fresh_cache = StageKvCache::new(slice.layers.len());
            for payload in &frames {
                let Some(hidden) = codec.decode(payload) else {
                    return WireFrame::ErrorFrame {
                        code: ProtocolErrorCode::MalformedFrame,
                        message: "failed to decode replay activation frame".into(),
                    };
                };
                if hidden.len() != state.model_cfg.d_model {
                    return WireFrame::ErrorFrame {
                        code: ProtocolErrorCode::MalformedFrame,
                        message: "replay activation dimension mismatch".into(),
                    };
                }
                let _ = slice.forward_token(&hidden, &mut fresh_cache);
            }

            if let Ok(mut kv) = state.kv_caches.lock() {
                kv.insert((session_id, key.0, key.1), fresh_cache);
            }
            state.requests_served.fetch_add(1, Ordering::Relaxed);

            WireFrame::ReplayKvAck {
                session_id,
                restored_tokens: frames.len() as u32,
                rebuild_us: t0.elapsed().as_micros() as u64,
            }
        }
        WireFrame::ForwardRequest {
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
            let t0 = Instant::now();
            let actual_in_digest = activation_commitment(&payload_bytes);
            if !constant_time_eq(&actual_in_digest, &activation_digest) {
                return WireFrame::ErrorFrame {
                    code: ProtocolErrorCode::ActivationIntegrityMismatch,
                    message: "incoming activation SHA-256 digest mismatch".into(),
                };
            }

            let key = (layer_start as usize, layer_end as usize);
            let slice = {
                let guard = match state.slices.lock() {
                    Ok(g) => g,
                    Err(_) => {
                        return WireFrame::ErrorFrame {
                            code: ProtocolErrorCode::MalformedFrame,
                            message: "slice lock poisoned".into(),
                        }
                    }
                };
                match guard.get(&key) {
                    Some(s) => s.clone(),
                    None => {
                        return WireFrame::ErrorFrame {
                            code: ProtocolErrorCode::StageMismatch,
                            message: format!("slice [{}, {}) not hosted", key.0, key.1),
                        }
                    }
                }
            };

            if !constant_time_eq(&slice.commitment.digest, &expected_weights_commitment) {
                return WireFrame::ErrorFrame {
                    code: ProtocolErrorCode::WeightsCommitmentMismatch,
                    message: "stage weights commitment mismatch".into(),
                };
            }

            let Some(hidden) = codec.decode(&payload_bytes) else {
                return WireFrame::ErrorFrame {
                    code: ProtocolErrorCode::MalformedFrame,
                    message: "invalid activation payload bytes".into(),
                };
            };
            if hidden.len() != state.model_cfg.d_model {
                return WireFrame::ErrorFrame {
                    code: ProtocolErrorCode::MalformedFrame,
                    message: "activation width != d_model".into(),
                };
            }

            let mut out_hidden = {
                let mut kv_map = match state.kv_caches.lock() {
                    Ok(g) => g,
                    Err(_) => {
                        return WireFrame::ErrorFrame {
                            code: ProtocolErrorCode::MalformedFrame,
                            message: "kv lock poisoned".into(),
                        }
                    }
                };
                let cache_key = (session_id, key.0, key.1);
                if !kv_map.contains_key(&cache_key)
                    && kv_map.len() >= state.max_concurrent_sessions as usize
                {
                    return WireFrame::ErrorFrame {
                        code: ProtocolErrorCode::BackpressureRejected,
                        message: "peer max concurrent sessions exceeded".into(),
                    };
                }
                let cache = kv_map
                    .entry(cache_key)
                    .or_insert_with(|| StageKvCache::new(slice.layers.len()));
                let (next_h, _) = slice.forward_token(&hidden, cache);
                next_h
            };

            if fault == FaultMode::SilentNumericTamper {
                // Deliberately corrupt hidden state before encoding/signing so only
                // spot-check re-execution can catch the dishonest output.
                for (i, v) in out_hidden.iter_mut().enumerate() {
                    *v += if i % 2 == 0 { 0.35 } else { -0.35 };
                }
            }

            let mut out_payload = codec.encode(&out_hidden);
            let out_digest = activation_commitment(&out_payload);

            let mut receipt_weights = slice.commitment.digest;
            if fault == FaultMode::WrongWeightsCommitment {
                receipt_weights[0] ^= 0xff;
            }

            let receipt_session = match fault {
                FaultMode::ReplayReceiptFromSession(other_sid) => other_sid,
                _ => session_id,
            };

            let mut receipt = StageReceipt::sign(
                state.peer_id.clone(),
                receipt_session,
                step_idx,
                stage_idx,
                layer_start,
                layer_end,
                receipt_weights,
                actual_in_digest,
                out_digest,
                &state.peer_secret,
            );

            if fault == FaultMode::ForgedReceiptMac {
                receipt.mac[0] ^= 0x55;
            }

            if fault == FaultMode::TamperActivationPayload && !out_payload.is_empty() {
                out_payload[0] ^= 0xff;
            }

            state.requests_served.fetch_add(1, Ordering::Relaxed);

            WireFrame::ForwardResponse {
                session_id,
                step_idx,
                stage_idx,
                codec,
                activation_digest: out_digest,
                compute_us: t0.elapsed().as_micros() as u64,
                receipt,
                payload_bytes: out_payload,
            }
        }
        _ => WireFrame::ErrorFrame {
            code: ProtocolErrorCode::MalformedFrame,
            message: "unexpected frame kind received by worker".into(),
        },
    }
}
