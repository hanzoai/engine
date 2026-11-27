// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Live multi-node peer-to-peer pipeline orchestrator, automatic TCP disconnect
//! failover engine, KV-cache replay recovery, and cryptographic + spot-check
//! stage receipt verifier.

use crate::model::{LayerSlice, MicroModel, StageKvCache};
use crate::sha256::{constant_time_eq, DIGEST_LEN};
use crate::transport::{
    activation_commitment, ActivationCodec, StageReceipt, TransportError, WireFrame,
};
use std::collections::HashMap;
use std::fmt;
use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant};

/// Connection descriptor for a live TCP peer participating in a P2P pipeline.
#[derive(Debug, Clone)]
pub struct PeerEndpoint {
    /// Unique peer identifier.
    pub peer_id: String,
    /// Live TCP socket address (`127.0.0.1:<port>`).
    pub addr: SocketAddr,
    /// Shared/registered peer HMAC key used to verify [`StageReceipt`] MACs.
    pub peer_secret: Vec<u8>,
}

/// Ordered pipeline stage route entry specifying primary peer and failover candidates.
#[derive(Debug, Clone)]
pub struct LiveStageRoute {
    /// Pipeline stage ordinal (`0..num_stages`).
    pub stage_idx: usize,
    /// Inclusive starting layer index.
    pub layer_start: usize,
    /// Exclusive ending layer index.
    pub layer_end: usize,
    /// Expected SHA-256 commitment over `[layer_start, layer_end)` weights.
    pub expected_weights_commitment: [u8; DIGEST_LEN],
    /// Currently active peer identifier serving this stage.
    pub active_peer_id: String,
    /// Ordered failover peer identifiers capable of promoting this stage on disconnect.
    pub failover_peer_ids: Vec<String>,
}

/// Recorded telemetry for one live mid-stream peer disconnect failover event.
#[derive(Debug, Clone)]
pub struct FailoverEvent {
    /// Stage ordinal that suffered the disconnect.
    pub stage_idx: usize,
    /// Layer range `[layer_start, layer_end)` migrated.
    pub layer_start: usize,
    /// Exclusive ending layer index.
    pub layer_end: usize,
    /// Autoregressive step index where the disconnect was detected.
    pub step_idx: usize,
    /// Disconnected / failed peer identifier.
    pub failed_peer_id: String,
    /// Surviving peer identifier that promoted the stage.
    pub replacement_peer_id: String,
    /// Number of prior token positions replayed to reconstruct stage KV cache.
    pub replayed_tokens: usize,
    /// Wire bytes transmitted during KV-cache replay recovery.
    pub replay_wire_bytes: u64,
    /// Total elapsed wall time (ms) from failed TCP call to completed KV replay.
    pub recovery_latency_ms: f64,
    /// Underlying OS/transport error summary that triggered failover.
    pub trigger_reason: String,
}

/// Verification or execution failure returned by [`P2pPipelineClient`].
#[derive(Debug)]
pub enum ClusterError {
    /// All primary and failover peers for a stage were unreachable or timed out.
    StageUnavailable {
        /// Failed stage ordinal.
        stage_idx: usize,
        /// Last transport error message.
        detail: String,
    },
    /// Wire activation SHA-256 integrity mismatch (bit corruption or tampering in transit).
    ActivationIntegrityMismatch {
        /// Stage ordinal.
        stage_idx: usize,
        /// Peer identifier.
        peer_id: String,
    },
    /// Peer returned a weights commitment that does not match the expected model shard.
    WeightsCommitmentMismatch {
        /// Stage ordinal.
        stage_idx: usize,
        /// Peer identifier.
        peer_id: String,
    },
    /// Peer receipt failed HMAC-SHA256 verification (forged receipt).
    ForgedReceiptMac {
        /// Stage ordinal.
        stage_idx: usize,
        /// Peer identifier.
        peer_id: String,
    },
    /// Peer receipt failed session, step, stage, or input/output digest binding (replay/spoofing).
    ReceiptBindingMismatch {
        /// Stage ordinal.
        stage_idx: usize,
        /// Peer identifier.
        peer_id: String,
        /// Specific mismatch detail.
        detail: String,
    },
    /// Spot-check re-execution detected silent numerical tampering despite valid framing & MAC.
    SpotCheckNumericMismatch {
        /// Stage ordinal.
        stage_idx: usize,
        /// Peer identifier.
        peer_id: String,
        /// Maximum absolute element-wise activation discrepancy detected.
        max_abs_diff: f32,
    },
    /// Remote peer returned a protocol error frame.
    RemoteProtocolError {
        /// Stage ordinal.
        stage_idx: usize,
        /// Error detail.
        detail: String,
    },
    /// Decode or codec mismatch error.
    CodecError(String),
}

impl fmt::Display for ClusterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StageUnavailable { stage_idx, detail } => {
                write!(f, "stage {stage_idx} unavailable after failover attempts: {detail}")
            }
            Self::ActivationIntegrityMismatch { stage_idx, peer_id } => {
                write!(f, "activation SHA-256 integrity mismatch at stage {stage_idx} ({peer_id})")
            }
            Self::WeightsCommitmentMismatch { stage_idx, peer_id } => {
                write!(f, "weights commitment mismatch at stage {stage_idx} ({peer_id})")
            }
            Self::ForgedReceiptMac { stage_idx, peer_id } => {
                write!(f, "forged HMAC stage receipt at stage {stage_idx} ({peer_id})")
            }
            Self::ReceiptBindingMismatch {
                stage_idx,
                peer_id,
                detail,
            } => write!(
                f,
                "receipt binding mismatch at stage {stage_idx} ({peer_id}): {detail}"
            ),
            Self::SpotCheckNumericMismatch {
                stage_idx,
                peer_id,
                max_abs_diff,
            } => write!(
                f,
                "spot-check numerical tamper detected at stage {stage_idx} ({peer_id}): max_abs_diff={max_abs_diff:.6e}"
            ),
            Self::RemoteProtocolError { stage_idx, detail } => {
                write!(f, "remote protocol error at stage {stage_idx}: {detail}")
            }
            Self::CodecError(s) => write!(f, "activation codec error: {s}"),
        }
    }
}

impl std::error::Error for ClusterError {}

/// Complete measured telemetry and outputs from a live multi-node P2P generation run.
#[derive(Debug, Clone)]
pub struct GenerationReport {
    /// Session identifier executed.
    pub session_id: u64,
    /// Generated token IDs (`len == requested_gen_tokens`).
    pub generated_tokens: Vec<u32>,
    /// Final vocabulary logits for every generated token position.
    pub generated_logits: Vec<Vec<f32>>,
    /// All verified [`StageReceipt`] records collected across stages and steps.
    pub receipts: Vec<StageReceipt>,
    /// Recorded mid-stream disconnect failover events.
    pub failover_events: Vec<FailoverEvent>,
    /// Total TCP wire bytes sent by orchestrator.
    pub wire_bytes_sent: u64,
    /// Total TCP wire bytes received by orchestrator.
    pub wire_bytes_recv: u64,
    /// End-to-end wall clock duration in milliseconds (prefill + decode + any failover).
    pub wall_time_ms: f64,
    /// Achieved decode throughput in tokens/sec (`generated_tokens / (decode_time_ms / 1000)`).
    pub decode_tok_per_sec: f64,
    /// Total time spent in prefill in milliseconds (TTFT proxy).
    pub prefill_ttft_ms: f64,
    /// Number of spot-check stage re-executions performed and passed.
    pub spot_checks_passed: usize,
}

/// Live multi-node TCP pipeline orchestrator with persistent sockets, automatic
/// disconnect failover + KV replay, cryptographic receipt verification, and optional
/// spot-check challenge re-execution.
pub struct P2pPipelineClient {
    endpoints: HashMap<String, PeerEndpoint>,
    connections: HashMap<String, TcpStream>,
    routes: Vec<LiveStageRoute>,
    codec: ActivationCodec,
    io_timeout: Duration,
    spot_check_stride: Option<usize>,
}

impl P2pPipelineClient {
    /// Creates a new live P2P pipeline client from `endpoints` and ordered `routes`.
    #[must_use]
    pub fn new(
        endpoints: Vec<PeerEndpoint>,
        routes: Vec<LiveStageRoute>,
        codec: ActivationCodec,
        io_timeout: Duration,
    ) -> Self {
        let ep_map = endpoints
            .into_iter()
            .map(|e| (e.peer_id.clone(), e))
            .collect();
        Self {
            endpoints: ep_map,
            connections: HashMap::new(),
            routes,
            codec,
            io_timeout,
            spot_check_stride: None,
        }
    }

    /// Enables deterministic spot-check verification every `stride` steps (`Some(1)` checks every step).
    pub fn set_spot_check_stride(&mut self, stride: Option<usize>) {
        self.spot_check_stride = stride;
    }

    /// Returns the currently active stage routing table.
    #[must_use]
    pub fn routes(&self) -> &[LiveStageRoute] {
        &self.routes
    }

    /// Sends a live `HealthPing` over TCP to `peer_id` and validates `HealthPong`.
    pub fn health_ping(&mut self, peer_id: &str, nonce: u64) -> Result<WireFrame, TransportError> {
        let req = WireFrame::HealthPing { nonce };
        let (resp, _, _) = self.rpc_once(peer_id, &req)?;
        Ok(resp)
    }

    /// Executes full autoregressive generation (`prompt_tokens` prefill + `gen_tokens` decode)
    /// across live TCP peer stages, invoking `on_after_gen_token(emitted_count)` after each
    /// generated token so callers/tests can inject a real mid-stream `WorkerNode::hard_kill()`.
    pub fn generate_with_hook<F>(
        &mut self,
        client_model: &MicroModel,
        session_id: u64,
        prompt_tokens: &[u32],
        gen_tokens: usize,
        mut on_after_gen_token: F,
    ) -> Result<GenerationReport, ClusterError>
    where
        F: FnMut(usize),
    {
        let run_start = Instant::now();
        let num_stages = self.routes.len();
        let mut stage_input_history: Vec<Vec<Vec<u8>>> = vec![Vec::new(); num_stages];
        let mut receipts = Vec::new();
        let mut failover_events = Vec::new();
        let mut wire_bytes_sent = 0u64;
        let mut wire_bytes_recv = 0u64;
        let mut spot_checks_passed = 0usize;

        // Local spot-check state (only materialized when spot-check challenge triggers).
        let mut spot_slices: Option<Vec<LayerSlice>> = None;
        let mut spot_caches: Option<Vec<StageKvCache>> = None;
        if self.spot_check_stride.is_some() {
            let slices: Vec<LayerSlice> = self
                .routes
                .iter()
                .map(|r| LayerSlice::from_config(&client_model.config, r.layer_start, r.layer_end))
                .collect();
            let caches: Vec<StageKvCache> = slices
                .iter()
                .map(|s| StageKvCache::new(s.layers.len()))
                .collect();
            spot_slices = Some(slices);
            spot_caches = Some(caches);
        }

        let mut step_idx = 0usize;
        let mut last_logits = Vec::new();

        // Prefill phase
        let prefill_start = Instant::now();
        for &tok in prompt_tokens {
            last_logits = self.forward_single_token(
                client_model,
                session_id,
                step_idx,
                tok,
                &mut stage_input_history,
                &mut receipts,
                &mut failover_events,
                &mut wire_bytes_sent,
                &mut wire_bytes_recv,
                &mut spot_slices,
                &mut spot_caches,
                &mut spot_checks_passed,
            )?;
            step_idx += 1;
        }
        let prefill_ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // Decode phase
        let decode_start = Instant::now();
        let mut generated_tokens = Vec::with_capacity(gen_tokens);
        let mut generated_logits = Vec::with_capacity(gen_tokens);

        for emitted in 0..gen_tokens {
            let next_tok = MicroModel::argmax_token(&last_logits);
            generated_tokens.push(next_tok);
            generated_logits.push(last_logits.clone());

            // Invoke mid-stream hook (e.g., hard-kill Node B after emitting token 8).
            on_after_gen_token(emitted + 1);

            last_logits = self.forward_single_token(
                client_model,
                session_id,
                step_idx,
                next_tok,
                &mut stage_input_history,
                &mut receipts,
                &mut failover_events,
                &mut wire_bytes_sent,
                &mut wire_bytes_recv,
                &mut spot_slices,
                &mut spot_caches,
                &mut spot_checks_passed,
            )?;
            step_idx += 1;
        }

        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let wall_time_ms = run_start.elapsed().as_secs_f64() * 1000.0;
        let decode_tok_per_sec = if decode_ms > 0.0 {
            (gen_tokens as f64) * 1000.0 / decode_ms
        } else {
            0.0
        };

        Ok(GenerationReport {
            session_id,
            generated_tokens,
            generated_logits,
            receipts,
            failover_events,
            wire_bytes_sent,
            wire_bytes_recv,
            wall_time_ms,
            decode_tok_per_sec,
            prefill_ttft_ms,
            spot_checks_passed,
        })
    }

    /// Executes full autoregressive generation without any injected mid-stream hook.
    pub fn generate(
        &mut self,
        client_model: &MicroModel,
        session_id: u64,
        prompt_tokens: &[u32],
        gen_tokens: usize,
    ) -> Result<GenerationReport, ClusterError> {
        self.generate_with_hook(client_model, session_id, prompt_tokens, gen_tokens, |_| {})
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_single_token(
        &mut self,
        client_model: &MicroModel,
        session_id: u64,
        step_idx: usize,
        token_id: u32,
        stage_input_history: &mut [Vec<Vec<u8>>],
        receipts: &mut Vec<StageReceipt>,
        failover_events: &mut Vec<FailoverEvent>,
        wire_bytes_sent: &mut u64,
        wire_bytes_recv: &mut u64,
        spot_slices: &mut Option<Vec<LayerSlice>>,
        spot_caches: &mut Option<Vec<StageKvCache>>,
        spot_checks_passed: &mut usize,
    ) -> Result<Vec<f32>, ClusterError> {
        let embed = client_model.embed_token(token_id);
        let mut current_payload = self.codec.encode(&embed);
        let num_stages = self.routes.len();

        for stage_idx in 0..num_stages {
            let input_payload = current_payload.clone();
            let input_digest = activation_commitment(&input_payload);
            let layer_start = self.routes[stage_idx].layer_start;
            let layer_end = self.routes[stage_idx].layer_end;
            let expected_weights = self.routes[stage_idx].expected_weights_commitment;

            let req = WireFrame::ForwardRequest {
                session_id,
                step_idx: step_idx as u32,
                stage_idx: stage_idx as u16,
                layer_start: layer_start as u16,
                layer_end: layer_end as u16,
                codec: self.codec,
                expected_weights_commitment: expected_weights,
                activation_digest: input_digest,
                payload_bytes: input_payload.clone(),
            };

            let (resp_frame, peer_used) = self.send_forward_with_failover(
                stage_idx,
                session_id,
                step_idx,
                &req,
                &stage_input_history[stage_idx],
                failover_events,
                wire_bytes_sent,
                wire_bytes_recv,
            )?;

            match resp_frame {
                WireFrame::ForwardResponse {
                    session_id: r_sid,
                    step_idx: r_step,
                    stage_idx: r_stage,
                    codec: r_codec,
                    activation_digest: r_digest,
                    receipt,
                    payload_bytes,
                    ..
                } => {
                    // 1. Verify outgoing activation SHA-256 wire integrity.
                    let actual_out_digest = activation_commitment(&payload_bytes);
                    if !constant_time_eq(&actual_out_digest, &r_digest) {
                        return Err(ClusterError::ActivationIntegrityMismatch {
                            stage_idx,
                            peer_id: peer_used,
                        });
                    }

                    // 2. Verify WeightsCommitment matches expected shard digest.
                    if !constant_time_eq(&receipt.weights_commitment, &expected_weights) {
                        return Err(ClusterError::WeightsCommitmentMismatch {
                            stage_idx,
                            peer_id: peer_used,
                        });
                    }

                    // 3. Verify receipt session, step, stage, layer range, and digest bindings.
                    if r_sid != session_id
                        || r_step != step_idx as u32
                        || r_stage != stage_idx as u16
                        || r_codec != self.codec
                        || receipt.session_id != session_id
                        || receipt.step_idx != step_idx as u32
                        || receipt.stage_idx != stage_idx as u16
                        || receipt.layer_start != layer_start as u16
                        || receipt.layer_end != layer_end as u16
                        || !constant_time_eq(&receipt.input_digest, &input_digest)
                        || !constant_time_eq(&receipt.output_digest, &actual_out_digest)
                    {
                        return Err(ClusterError::ReceiptBindingMismatch {
                            stage_idx,
                            peer_id: peer_used,
                            detail: format!(
                                "expected sid={session_id} step={step_idx} stage={stage_idx}, got sid={} step={} stage={}",
                                receipt.session_id, receipt.step_idx, receipt.stage_idx
                            ),
                        });
                    }

                    // 4. Verify peer HMAC-SHA256 receipt signature in constant time.
                    let secret = self
                        .endpoints
                        .get(&peer_used)
                        .map(|e| e.peer_secret.as_slice())
                        .unwrap_or(&[]);
                    if !receipt.verify_mac(secret) {
                        return Err(ClusterError::ForgedReceiptMac {
                            stage_idx,
                            peer_id: peer_used,
                        });
                    }

                    // 5. Optional deterministic spot-check re-execution against reference weights.
                    if let (Some(stride), Some(slices), Some(caches)) = (
                        self.spot_check_stride,
                        spot_slices.as_ref(),
                        spot_caches.as_mut(),
                    ) {
                        let decoded_in = self.codec.decode(&input_payload).ok_or_else(|| {
                            ClusterError::CodecError("decode input failed".into())
                        })?;
                        let (ref_out, _) =
                            slices[stage_idx].forward_token(&decoded_in, &mut caches[stage_idx]);
                        if stride > 0 && step_idx.is_multiple_of(stride) {
                            let ref_encoded = self.codec.encode(&ref_out);
                            let ref_decoded = self.codec.decode(&ref_encoded).ok_or_else(|| {
                                ClusterError::CodecError("decode ref output failed".into())
                            })?;
                            let peer_decoded =
                                self.codec.decode(&payload_bytes).ok_or_else(|| {
                                    ClusterError::CodecError("decode peer output failed".into())
                                })?;
                            let mut max_diff = 0.0f32;
                            for (&a, &b) in ref_decoded.iter().zip(peer_decoded.iter()) {
                                let d = (a - b).abs();
                                if d > max_diff {
                                    max_diff = d;
                                }
                            }
                            let tol = match self.codec {
                                ActivationCodec::Bf16Lossless => 0.0,
                                ActivationCodec::Int8PerTensor => 1e-4,
                            };
                            if max_diff > tol {
                                return Err(ClusterError::SpotCheckNumericMismatch {
                                    stage_idx,
                                    peer_id: peer_used,
                                    max_abs_diff: max_diff,
                                });
                            }
                            *spot_checks_passed += 1;
                        }
                    }

                    // Record committed stage input for future failover KV-cache replay.
                    stage_input_history[stage_idx].push(input_payload);
                    receipts.push(receipt);
                    current_payload = payload_bytes;
                }
                WireFrame::ErrorFrame { code, message } => {
                    return Err(ClusterError::RemoteProtocolError {
                        stage_idx,
                        detail: format!("{code:?}: {message}"),
                    });
                }
                other => {
                    return Err(ClusterError::RemoteProtocolError {
                        stage_idx,
                        detail: format!("unexpected wire frame: {other:?}"),
                    });
                }
            }
        }

        let final_hidden = self
            .codec
            .decode(&current_payload)
            .ok_or_else(|| ClusterError::CodecError("failed to decode final activation".into()))?;
        Ok(client_model.project_logits(&final_hidden))
    }

    #[allow(clippy::too_many_arguments)]
    fn send_forward_with_failover(
        &mut self,
        stage_idx: usize,
        session_id: u64,
        step_idx: usize,
        req: &WireFrame,
        prior_inputs: &[Vec<u8>],
        failover_events: &mut Vec<FailoverEvent>,
        wire_bytes_sent: &mut u64,
        wire_bytes_recv: &mut u64,
    ) -> Result<(WireFrame, String), ClusterError> {
        let primary_peer = self.routes[stage_idx].active_peer_id.clone();

        match self.rpc_once(&primary_peer, req) {
            Ok((resp, sent, recv)) => {
                *wire_bytes_sent += sent as u64;
                *wire_bytes_recv += recv as u64;
                Ok((resp, primary_peer))
            }
            Err(first_err) => {
                // Drop broken socket and attempt live failover onto surviving peer.
                self.connections.remove(&primary_peer);
                let recovery_start = Instant::now();
                let trigger_reason = first_err.to_string();
                let candidates = self.routes[stage_idx].failover_peer_ids.clone();
                let layer_start = self.routes[stage_idx].layer_start;
                let layer_end = self.routes[stage_idx].layer_end;
                let expected_weights = self.routes[stage_idx].expected_weights_commitment;

                let mut last_err = trigger_reason.clone();
                for cand in candidates {
                    if cand == primary_peer {
                        continue;
                    }
                    // Step 1: Replay all prior stage inputs into the surviving/replacement peer
                    // so its KV cache for `[layer_start, layer_end)` matches the lost node exactly.
                    let replay_req = WireFrame::ReplayKvBatch {
                        session_id,
                        layer_start: layer_start as u16,
                        layer_end: layer_end as u16,
                        codec: self.codec,
                        expected_weights_commitment: expected_weights,
                        frames: prior_inputs.to_vec(),
                    };

                    match self.rpc_once(&cand, &replay_req) {
                        Ok((
                            WireFrame::ReplayKvAck {
                                restored_tokens, ..
                            },
                            r_sent,
                            r_recv,
                        )) => {
                            *wire_bytes_sent += r_sent as u64;
                            *wire_bytes_recv += r_recv as u64;

                            // Step 2: Execute the interrupted ForwardRequest on the promoted peer.
                            match self.rpc_once(&cand, req) {
                                Ok((resp, f_sent, f_recv)) => {
                                    *wire_bytes_sent += f_sent as u64;
                                    *wire_bytes_recv += f_recv as u64;

                                    let recovery_ms =
                                        recovery_start.elapsed().as_secs_f64() * 1000.0;
                                    self.routes[stage_idx].active_peer_id = cand.clone();
                                    self.routes[stage_idx]
                                        .failover_peer_ids
                                        .retain(|id| id != &cand && id != &primary_peer);

                                    failover_events.push(FailoverEvent {
                                        stage_idx,
                                        layer_start,
                                        layer_end,
                                        step_idx,
                                        failed_peer_id: primary_peer,
                                        replacement_peer_id: cand.clone(),
                                        replayed_tokens: restored_tokens as usize,
                                        replay_wire_bytes: (r_sent + r_recv) as u64,
                                        recovery_latency_ms: recovery_ms,
                                        trigger_reason,
                                    });
                                    return Ok((resp, cand));
                                }
                                Err(e) => {
                                    self.connections.remove(&cand);
                                    last_err = e.to_string();
                                }
                            }
                        }
                        Ok((other, _, _)) => {
                            last_err = format!("unexpected replay ack from {cand}: {other:?}");
                        }
                        Err(e) => {
                            self.connections.remove(&cand);
                            last_err = e.to_string();
                        }
                    }
                }

                Err(ClusterError::StageUnavailable {
                    stage_idx,
                    detail: last_err,
                })
            }
        }
    }

    fn rpc_once(
        &mut self,
        peer_id: &str,
        frame: &WireFrame,
    ) -> Result<(WireFrame, usize, usize), TransportError> {
        let endpoint = self.endpoints.get(peer_id).ok_or_else(|| {
            TransportError::MalformedPayload(format!("unknown peer_id {peer_id}"))
        })?;
        let addr = endpoint.addr;

        if !self.connections.contains_key(peer_id) {
            let stream = TcpStream::connect_timeout(&addr, self.io_timeout)?;
            stream.set_nodelay(true)?;
            stream.set_read_timeout(Some(self.io_timeout))?;
            stream.set_write_timeout(Some(self.io_timeout))?;
            self.connections.insert(peer_id.to_string(), stream);
        }

        let stream = self
            .connections
            .get_mut(peer_id)
            .expect("connection inserted above");
        let sent = match frame.write_to_stream(stream) {
            Ok(n) => n,
            Err(e) => {
                self.connections.remove(peer_id);
                return Err(e);
            }
        };
        let (resp, recv) = match WireFrame::read_from_stream(stream) {
            Ok(v) => v,
            Err(e) => {
                self.connections.remove(peer_id);
                return Err(e);
            }
        };
        Ok((resp, sent, recv))
    }
}
