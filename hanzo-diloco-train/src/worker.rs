//! Independent live loopback TCP training worker node supporting inner AdamW
//! steps, INT8 error-feedback pseudo-gradient compression, abrupt `hard_kill()`
//! churn simulation, live TCP checkpoint recovery, and adversarial fault modes.

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::checkpoint::{CheckpointError, TrainingCheckpoint};
use crate::model::{SmallZenConfig, SmallZenLm};
use crate::optimizer::{
    encode_pseudo_gradient, InnerAdamW, InnerAdamWConfig, OuterNesterovAggregator, TrustTier,
};
use crate::sha256::{sha256_hex, SplitMix64};
use crate::transport::{
    read_frame, write_frame, FrameKind, RoundAssignmentMsg, TrainingRoundReceipt, WanLinkProfile,
    WorkerUpdateMsg, RECEIPT_VERSION,
};

/// Operational behavior of a worker node (honest or adversarial test fault).
#[derive(Clone, Debug)]
pub enum WorkerBehavior {
    /// Honest execution of inner AdamW steps and signed pseudo-gradient return.
    Honest,
    /// Byzantine sign-flip and gradient-explosion attack (`-scale * honest_delta`).
    SignFlipPoison {
        /// Multiplicative inversion factor applied to the honest pseudo-gradient.
        scale: f32,
    },
    /// Free-rider attack submitting an all-zero pseudo-gradient while claiming
    /// full inner step completion.
    FreeRiderZeroDelta,
    /// Stale-round replay attack signing a forged/old `round_id`.
    StaleRoundReplay {
        /// Stale round identifier replayed in the receipt.
        replay_round_id: u64,
    },
    /// Diverged lineage attack submitting a wrong `base_checkpoint_sha256`.
    WrongBaseCheckpoint,
    /// Wire integrity attack corrupting encoded delta bytes after signing the
    /// receipt (causes digest mismatch / tampered payload detection).
    TamperedWireBytes,
    /// Forged receipt HMAC signature attack.
    ForgedReceiptHmac,
}

/// Configuration for spawning a live [`TrainingWorkerNode`].
#[derive(Clone, Debug)]
pub struct WorkerConfig {
    /// Unique worker identifier (e.g., `"worker-us-east"`).
    pub worker_id: String,
    /// Simulated geographic/administrative WAN link profile.
    pub wan_profile: WanLinkProfile,
    /// Shared HMAC signing key bound to this worker identity.
    pub signing_key: Vec<u8>,
    /// Deterministic RNG seed for local mini-batch sampling.
    pub seed: u64,
    /// Model architecture configuration.
    pub model_config: SmallZenConfig,
    /// Worker local training corpus shard (sequences of length `seq_len + 1`).
    pub local_shard: Vec<Vec<u8>>,
    /// Honest or adversarial execution behavior.
    pub behavior: WorkerBehavior,
}

/// Telemetry returned when a freshly spawned replacement worker recovers state
/// from a live TCP checkpoint mirror.
#[derive(Clone, Debug)]
pub struct CheckpointRecoveryReport {
    /// Worker identifier that recovered.
    pub worker_id: String,
    /// Verified checkpoint SHA-256 hex digest.
    pub checkpoint_sha256: String,
    /// Round index recovered to.
    pub recovered_round_id: u64,
    /// Raw checkpoint wire bytes downloaded over TCP.
    pub recovery_wire_bytes: usize,
    /// Measured wall-clock recovery time in milliseconds (including WAN shaping).
    pub recovery_wall_ms: f64,
    /// Simulated unscaled WAN transit time in milliseconds.
    pub simulated_wan_ms: f64,
}

/// Handle to a running independent worker node listening on a live loopback TCP
/// socket in its own OS thread.
pub struct TrainingWorkerNode {
    /// Worker identifier.
    pub worker_id: String,
    /// Bound loopback TCP address (`127.0.0.1:<ephemeral_port>`).
    pub addr: SocketAddr,
    /// WAN profile advertised for this worker.
    pub wan_profile: WanLinkProfile,
    /// Worker HMAC signing key.
    pub signing_key: Vec<u8>,
    alive: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl TrainingWorkerNode {
    /// Spawns a live loopback TCP worker node immediately ready to accept
    /// [`FrameKind::RoundAssignment`] connections.
    pub fn spawn(cfg: WorkerConfig) -> Result<Self, String> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| format!("worker {} bind failed: {e}", cfg.worker_id))?;
        let addr = listener
            .local_addr()
            .map_err(|e| format!("worker {} local_addr failed: {e}", cfg.worker_id))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("worker {} set_nonblocking failed: {e}", cfg.worker_id))?;

        let worker_id = cfg.worker_id.clone();
        let wan_profile = cfg.wan_profile.clone();
        let signing_key = cfg.signing_key.clone();
        let alive = Arc::new(AtomicBool::new(true));
        let alive_bg = Arc::clone(&alive);

        let handle = thread::Builder::new()
            .name(format!("hanzo-worker-{worker_id}"))
            .spawn(move || {
                worker_event_loop(listener, cfg, alive_bg);
            })
            .map_err(|e| format!("failed spawning worker thread: {e}"))?;

        Ok(Self {
            worker_id,
            addr,
            wan_profile,
            signing_key,
            alive,
            handle: Some(handle),
        })
    }

    /// Returns whether this worker has not been killed.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// Abruptly kills the worker mid-training, dropping its `TcpListener` and
    /// closing its listening port so peers experience immediate connection
    /// failure (`ECONNREFUSED`) and must continue via quorum + recovery.
    pub fn hard_kill(&mut self) {
        self.alive.store(false, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Gracefully stops the worker by sending [`FrameKind::Shutdown`] and
    /// joining its thread.
    pub fn shutdown(&mut self) {
        if self.alive.swap(false, Ordering::SeqCst) {
            if let Ok(mut stream) =
                TcpStream::connect_timeout(&self.addr, Duration::from_millis(200))
            {
                let _ = write_frame(&mut stream, FrameKind::Shutdown, &[]);
            }
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Downloads and cryptographically verifies a signed checkpoint over a live
    /// TCP checkpoint mirror connection before a replacement worker joins the
    /// active training pool.
    pub fn fetch_verified_checkpoint_over_tcp(
        mirror_addr: SocketAddr,
        checkpoint_sha256: &str,
        checkpoint_signing_key: &[u8],
        wan_profile: &WanLinkProfile,
        worker_id: &str,
        round_id: u64,
    ) -> Result<(TrainingCheckpoint, CheckpointRecoveryReport), CheckpointError> {
        let start = Instant::now();
        let mut stream = TcpStream::connect_timeout(&mirror_addr, Duration::from_secs(2))
            .map_err(|e| CheckpointError::Io(format!("checkpoint mirror connect failed: {e}")))?;
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));

        let req_bytes = checkpoint_sha256.as_bytes();
        let sim_up = wan_profile.apply_one_way_shaping(req_bytes.len() + 10, round_id);
        let sent = write_frame(&mut stream, FrameKind::CheckpointRequest, req_bytes)
            .map_err(CheckpointError::Io)?;

        let (kind, payload, recv) = read_frame(&mut stream).map_err(CheckpointError::Io)?;
        if kind != FrameKind::CheckpointResponse {
            return Err(CheckpointError::MalformedFormat(format!(
                "unexpected frame kind {kind:?} on checkpoint fetch"
            )));
        }
        let sim_down = wan_profile.apply_one_way_shaping(recv, round_id);

        let ckpt = TrainingCheckpoint::from_bytes_verified(
            &payload,
            Some(checkpoint_sha256),
            checkpoint_signing_key,
        )?;

        let report = CheckpointRecoveryReport {
            worker_id: worker_id.to_string(),
            checkpoint_sha256: ckpt.checkpoint_sha256.clone(),
            recovered_round_id: ckpt.round_id,
            recovery_wire_bytes: sent + recv,
            recovery_wall_ms: start.elapsed().as_secs_f64() * 1_000.0,
            simulated_wan_ms: sim_up + sim_down,
        };
        Ok((ckpt, report))
    }
}

impl Drop for TrainingWorkerNode {
    fn drop(&mut self) {
        self.hard_kill();
    }
}

fn worker_event_loop(listener: TcpListener, cfg: WorkerConfig, alive: Arc<AtomicBool>) {
    let mut model = SmallZenLm::new(cfg.model_config.clone());
    let param_count = model.lora_param_count();
    let mut error_feedback = vec![0.0f32; param_count];
    let mut rng = SplitMix64::new(cfg.seed);

    while alive.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((mut stream, _peer)) => {
                if !alive.load(Ordering::SeqCst) {
                    break;
                }
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));

                let Ok((kind, payload, recv_bytes)) = read_frame(&mut stream) else {
                    continue;
                };
                match kind {
                    FrameKind::Shutdown => {
                        alive.store(false, Ordering::SeqCst);
                        break;
                    }
                    FrameKind::RoundAssignment => {
                        let Ok(assign) = RoundAssignmentMsg::decode(&payload) else {
                            continue;
                        };
                        let sim_down = cfg
                            .wan_profile
                            .apply_one_way_shaping(recv_bytes, assign.round_id);

                        if !alive.load(Ordering::SeqCst) {
                            break;
                        }

                        let update = execute_local_round(
                            &cfg,
                            &mut model,
                            &mut error_feedback,
                            &mut rng,
                            &assign,
                            sim_down,
                        );
                        let encoded = update.encode();
                        let sim_up = cfg
                            .wan_profile
                            .apply_one_way_shaping(encoded.len() + 10, assign.round_id);

                        let mut final_update = update;
                        final_update.simulated_wan_ms = sim_down + sim_up;
                        let final_bytes = final_update.encode();
                        let _ = write_frame(&mut stream, FrameKind::WorkerUpdate, &final_bytes);
                    }
                    _ => {}
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(3));
            }
            Err(_) => {
                thread::sleep(Duration::from_millis(5));
            }
        }
    }
    drop(listener);
}

fn execute_local_round(
    cfg: &WorkerConfig,
    model: &mut SmallZenLm,
    error_feedback: &mut [f32],
    rng: &mut SplitMix64,
    assign: &RoundAssignmentMsg,
    initial_wan_ms: f64,
) -> WorkerUpdateMsg {
    let start = Instant::now();
    let _ = model.load_lora_params(&assign.base_params);

    let mut params = assign.base_params.clone();
    let mut adam = InnerAdamW::new(
        params.len(),
        InnerAdamWConfig {
            learning_rate: assign.inner_lr,
            ..InnerAdamWConfig::default()
        },
    );

    let shard_len = cfg.local_shard.len().max(1);
    let bs = (assign.batch_size as usize).max(1).min(shard_len);

    let mut loss_before = 0.0f64;
    let mut loss_after = 0.0f64;
    let mut total_tokens = 0u64;

    if matches!(cfg.behavior, WorkerBehavior::FreeRiderZeroDelta) {
        // Free-rider skips real training and returns zero pseudo-gradient.
        total_tokens =
            (assign.inner_steps as u64) * (bs as u64) * (cfg.model_config.seq_len as u64);
    } else {
        for step_idx in 0..assign.inner_steps {
            let mut batch: Vec<&[u8]> = Vec::with_capacity(bs);
            let base_offset = rng.next_usize(shard_len);
            for b in 0..bs {
                let s_idx = (base_offset + b * 7) % shard_len;
                batch.push(cfg.local_shard[s_idx].as_slice());
            }

            let (loss, grads, toks) = model.forward_and_backward_batch(&batch);
            if step_idx == 0 {
                loss_before = loss;
            }
            loss_after = loss;
            total_tokens += toks as u64;

            adam.step(&mut params, &grads);
            let _ = model.load_lora_params(&params);
        }
    }

    // Pseudo-gradient delta_k = theta^(t) - theta_k^(t, H)
    let mut delta = vec![0.0f32; params.len()];
    for i in 0..params.len() {
        delta[i] = assign.base_params[i] - params[i];
    }

    // Apply adversarial mutation if configured
    match &cfg.behavior {
        WorkerBehavior::Honest
        | WorkerBehavior::StaleRoundReplay { .. }
        | WorkerBehavior::WrongBaseCheckpoint
        | WorkerBehavior::TamperedWireBytes
        | WorkerBehavior::ForgedReceiptHmac => {}
        WorkerBehavior::SignFlipPoison { scale } => {
            for d in &mut delta {
                *d *= -(*scale);
            }
        }
        WorkerBehavior::FreeRiderZeroDelta => {
            delta.fill(0.0);
        }
    }

    let delta_l2_norm = OuterNesterovAggregator::l2_norm(&delta);
    let mut encoded_delta =
        encode_pseudo_gradient(&delta, assign.compression, Some(error_feedback));
    let delta_sha256 = sha256_hex(&encoded_delta);

    let receipt_round_id = match cfg.behavior {
        WorkerBehavior::StaleRoundReplay { replay_round_id } => replay_round_id,
        _ => assign.round_id,
    };
    let receipt_base_ckpt = match cfg.behavior {
        WorkerBehavior::WrongBaseCheckpoint => {
            "000000000000000000000000000000000000000000000000000000000000dead".to_string()
        }
        _ => assign.base_checkpoint_sha256.clone(),
    };

    let mut receipt = TrainingRoundReceipt {
        receipt_version: RECEIPT_VERSION.to_string(),
        round_id: receipt_round_id,
        worker_id: cfg.worker_id.clone(),
        region: cfg.wan_profile.region.clone(),
        base_checkpoint_sha256: receipt_base_ckpt,
        delta_sha256,
        inner_steps_completed: assign.inner_steps,
        tokens_processed: total_tokens,
        local_loss_before: loss_before,
        local_loss_after: loss_after,
        delta_l2_norm,
        wire_bytes: encoded_delta.len() as u64,
        trust_tier: TrustTier::SignedDeltaAndLossSpotCheck,
        hmac_sha256: String::new(),
    };
    receipt.sign(&cfg.signing_key);

    if matches!(cfg.behavior, WorkerBehavior::ForgedReceiptHmac) {
        receipt.hmac_sha256 =
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_string();
    }
    if matches!(cfg.behavior, WorkerBehavior::TamperedWireBytes) && encoded_delta.len() > 12 {
        // Flip bytes in the wire delta after signing to trigger digest mismatch.
        encoded_delta[10] ^= 0xFF;
        encoded_delta[11] ^= 0x55;
    }

    WorkerUpdateMsg {
        receipt,
        encoded_delta,
        compute_ms: start.elapsed().as_secs_f64() * 1_000.0,
        simulated_wan_ms: initial_wan_ms,
    }
}
