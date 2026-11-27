//! Decentralized elastic multi-node TCP training coordinator with live worker
//! churn, peer-accessible VFS checkpoint mirroring, and Byzantine screening.
//!
//! No Hanzo-operated service is required (`hanzo_coordinator_required = false`):
//! every outer round produces a signed content-addressed VFS checkpoint
//! (`vfs://checkpoints/<sha256>`) served over a stateless loopback TCP mirror so
//! newly joined or replacement workers can bootstrap independently.

use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::checkpoint::{
    EnsoPeerRoute, HanzoTrainingManifest, TrainingCheckpoint, VfsCheckpointStore,
};
use crate::corpus::GroundedCorpus;
use crate::model::{EvalMetrics, SmallZenConfig, SmallZenLm};
use crate::optimizer::{
    decode_pseudo_gradient, CandidateDelta, CompressionMode, OuterNesterovAggregator,
    OuterOptimizerConfig, RejectionReason, TrustTier,
};
use crate::sha256::{ct_eq, sha256_hex};
use crate::transport::{
    read_frame, write_frame, FrameKind, RoundAssignmentMsg, WanLinkProfile, WorkerUpdateMsg,
};
use crate::worker::{CheckpointRecoveryReport, TrainingWorkerNode, WorkerBehavior, WorkerConfig};

/// Optional mid-run worker churn scenario exercised over live TCP sockets.
#[derive(Clone, Debug)]
pub struct ChurnSchedule {
    /// Worker identifier to `hard_kill()` before `kill_before_round`.
    pub victim_worker_id: String,
    /// 1-indexed outer round before which the victim worker is abruptly killed.
    pub kill_before_round: u64,
    /// Fresh replacement worker identifier spawned before `rejoin_before_round`.
    pub replacement_worker_id: String,
    /// 1-indexed outer round before which the replacement fetches the latest
    /// signed VFS checkpoint over TCP, verifies it, and joins training.
    pub rejoin_before_round: u64,
    /// WAN profile assigned to the replacement worker.
    pub replacement_wan_profile: WanLinkProfile,
}

/// Specification of a single worker slot in a cluster run.
#[derive(Clone, Debug)]
pub struct WorkerSlotSpec {
    /// Worker identifier.
    pub worker_id: String,
    /// WAN link profile.
    pub wan_profile: WanLinkProfile,
    /// Honest or adversarial behavior.
    pub behavior: WorkerBehavior,
}

/// Complete configuration for an [`ElasticTrainingCluster`] execution.
#[derive(Clone, Debug)]
pub struct ClusterRunConfig {
    /// Human-readable scenario identifier.
    pub scenario_name: String,
    /// Model configuration.
    pub model_config: SmallZenConfig,
    /// Worker pool definitions.
    pub workers: Vec<WorkerSlotSpec>,
    /// Number of outer synchronization rounds to execute.
    pub outer_rounds: u64,
    /// Number of local inner AdamW steps `H` per outer round.
    pub inner_steps_per_round: u32,
    /// Local mini-batch size (sequences per inner step).
    pub batch_size: u32,
    /// Inner AdamW learning rate.
    pub inner_lr: f32,
    /// Outer Nesterov and Byzantine defense configuration.
    pub outer_config: OuterOptimizerConfig,
    /// Wire pseudo-gradient compression mode.
    pub compression: CompressionMode,
    /// Minimum accepted worker updates required to commit an outer round.
    pub min_quorum: usize,
    /// Number of validation sequences evaluated per round.
    pub eval_sequences: usize,
    /// Number of spot-check validation sequences used for Byzantine screening.
    pub spot_check_sequences: usize,
    /// Deterministic base seed.
    pub seed: u64,
    /// Optional mid-run worker kill + TCP checkpoint recovery schedule.
    pub churn: Option<ChurnSchedule>,
    /// Directory where immutable VFS checkpoints are persisted.
    pub checkpoint_dir: PathBuf,
}

/// Per-round recorded telemetry.
#[derive(Clone, Debug)]
pub struct RoundTelemetry {
    /// 1-indexed outer round number.
    pub round_id: u64,
    /// Number of workers dispatched in this round.
    pub active_workers: usize,
    /// Number of worker updates accepted after cryptographic + Byzantine checks.
    pub accepted_workers: usize,
    /// Number of worker updates rejected or failed due to disconnect.
    pub rejected_or_failed_workers: usize,
    /// Structured rejection reasons observed in this round.
    pub rejection_reasons: Vec<String>,
    /// Held-out validation cross-entropy loss after this outer round.
    pub val_loss: f64,
    /// Held-out validation perplexity after this outer round.
    pub val_perplexity: f64,
    /// Held-out validation top-1 next-token accuracy after this outer round.
    pub val_top1_accuracy: f64,
    /// Measured wall-clock duration of this round (ms).
    pub round_wall_ms: f64,
    /// Simulated unscaled WAN bottleneck duration of this round (ms).
    pub simulated_wan_bottleneck_ms: f64,
    /// Total wire bytes transferred over live TCP in this round (down + up).
    pub round_wire_bytes: u64,
    /// Total pseudo-gradient upload bytes across workers in this round.
    pub delta_upload_bytes: u64,
    /// Total causal next-token positions trained across accepted workers.
    pub tokens_processed: u64,
    /// Verified VFS checkpoint SHA-256 hex digest written at end of round.
    pub checkpoint_sha256: String,
    /// Notable lifecycle events (churn kill, checkpoint recovery, etc.).
    pub events: Vec<String>,
}

/// Complete measured result of an [`ElasticTrainingCluster`] run.
#[derive(Clone, Debug)]
pub struct ClusterRunReport {
    /// Scenario name.
    pub scenario_name: String,
    /// Versioned Hanzo elastic training manifest (`hanzo.train.elastic/v1`).
    pub manifest: HanzoTrainingManifest,
    /// Trust tier used during this run.
    pub trust_tier: TrustTier,
    /// Initial validation metrics before round 1.
    pub initial_eval: EvalMetrics,
    /// Final validation metrics after all completed rounds.
    pub final_eval: EvalMetrics,
    /// Target outer rounds.
    pub target_rounds: u64,
    /// Completed outer rounds.
    pub completed_rounds: u64,
    /// Run completion rate in `[0.0, 1.0]`.
    pub completion_rate: f64,
    /// Total causal tokens trained across all accepted worker rounds.
    pub total_tokens_trained: u64,
    /// Total live TCP wire bytes transferred (assignments + updates + recovery).
    pub total_wire_bytes: u64,
    /// Total pseudo-gradient upload bytes (worker -> aggregator).
    pub total_delta_upload_bytes: u64,
    /// Measured total run wall time in milliseconds.
    pub total_wall_ms: f64,
    /// Time to first committed checkpoint in milliseconds (training TTFT-equivalent).
    pub time_to_first_checkpoint_ms: f64,
    /// Effective simulated wall time (unscaled WAN + real compute) in milliseconds.
    pub effective_wan_wall_ms: f64,
    /// Effective training throughput (useful tokens / second under measured wall time).
    pub throughput_tokens_per_sec: f64,
    /// Effective throughput under unscaled WAN timing (tokens / second).
    pub effective_wan_tokens_per_sec: f64,
    /// Round latency percentiles (ms) measured on live TCP.
    pub round_latency_p50_ms: f64,
    /// 95th percentile round wall latency (ms).
    pub round_latency_p95_ms: f64,
    /// 99th percentile round wall latency (ms).
    pub round_latency_p99_ms: f64,
    /// Number of rejected adversarial/faulty worker updates across the run.
    pub total_rejected_updates: usize,
    /// Breakdown of structured rejection reasons observed.
    pub all_rejection_reasons: Vec<String>,
    /// Live checkpoint recovery report when churn/rejoin occurred.
    pub recovery_report: Option<CheckpointRecoveryReport>,
    /// Final verified checkpoint SHA-256 hex digest.
    pub final_checkpoint_sha256: String,
    /// Chronological per-round telemetry.
    pub rounds: Vec<RoundTelemetry>,
    /// Final trained LoRA parameter vector (exported for sovereignty/verification).
    pub final_lora_params: Vec<f32>,
}

/// Stateless loopback TCP mirror serving immutable VFS checkpoints to newly
/// joined or recovering workers without requiring any Hanzo-hosted endpoint.
struct CheckpointMirrorServer {
    addr: SocketAddr,
    alive: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl CheckpointMirrorServer {
    fn spawn(store: Arc<Mutex<VfsCheckpointStore>>) -> Result<Self, String> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| format!("checkpoint mirror bind failed: {e}"))?;
        let addr = listener
            .local_addr()
            .map_err(|e| format!("checkpoint mirror local_addr failed: {e}"))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("checkpoint mirror nonblocking failed: {e}"))?;
        let alive = Arc::new(AtomicBool::new(true));
        let alive_bg = Arc::clone(&alive);

        let handle = thread::Builder::new()
            .name("hanzo-vfs-mirror".to_string())
            .spawn(move || {
                while alive_bg.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let _ = stream.set_nonblocking(false);
                            let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
                            let _ = stream.set_write_timeout(Some(Duration::from_secs(3)));
                            if let Ok((FrameKind::CheckpointRequest, payload, _)) =
                                read_frame(&mut stream)
                            {
                                if let Ok(digest) = std::str::from_utf8(&payload) {
                                    let bytes_opt = {
                                        let guard = store.lock().expect("store lock");
                                        guard.load_raw_bytes(digest).ok()
                                    };
                                    if let Some(raw) = bytes_opt {
                                        let _ = write_frame(
                                            &mut stream,
                                            FrameKind::CheckpointResponse,
                                            &raw,
                                        );
                                    }
                                }
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
            })
            .map_err(|e| format!("failed spawning checkpoint mirror: {e}"))?;

        Ok(Self {
            addr,
            alive,
            handle: Some(handle),
        })
    }
}

impl Drop for CheckpointMirrorServer {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Decentralized elastic training runner executing live loopback TCP rounds.
pub struct ElasticTrainingCluster;

impl ElasticTrainingCluster {
    /// Executes a complete reproducible multi-node live TCP training run on the
    /// verified [`GroundedCorpus`].
    pub fn run(
        cfg: &ClusterRunConfig,
        corpus: &GroundedCorpus,
    ) -> Result<ClusterRunReport, String> {
        let run_start = Instant::now();
        let ckpt_signing_key = format!("hanzo-vfs-quorum-key-seed-{}", cfg.seed).into_bytes();
        let store = VfsCheckpointStore::new(cfg.checkpoint_dir.clone(), ckpt_signing_key.clone())
            .map_err(|e| format!("failed creating VFS checkpoint store: {e}"))?;
        let shared_store = Arc::new(Mutex::new(store));
        let mirror = CheckpointMirrorServer::spawn(Arc::clone(&shared_store))?;

        let mut global_model = SmallZenLm::new(cfg.model_config.clone());
        let mut global_params = global_model.export_lora_params();
        let mut outer_opt =
            OuterNesterovAggregator::new(global_params.len(), cfg.outer_config.clone());

        let eval_batch = corpus.eval_subset(cfg.eval_sequences);
        let spot_check_batch = corpus.spot_check_subset(cfg.spot_check_sequences);
        let initial_eval = global_model.eval_metrics(&eval_batch);

        // Persist genesis checkpoint (round 0) in VFS.
        let genesis_ckpt = TrainingCheckpoint::build_signed(
            0,
            0,
            "genesis".to_string(),
            cfg.model_config.config_sha256(),
            initial_eval.loss,
            initial_eval.perplexity,
            global_params.clone(),
            outer_opt.velocity.clone(),
            &ckpt_signing_key,
        );
        let mut current_ckpt_sha = genesis_ckpt.checkpoint_sha256.clone();
        {
            let mut guard = shared_store.lock().expect("store lock");
            guard
                .save_checkpoint(&genesis_ckpt)
                .map_err(|e| format!("failed saving genesis checkpoint: {e}"))?;
        }

        // Spawn live loopback TCP worker nodes.
        let num_initial_workers = cfg.workers.len().max(1);
        let mut active_workers: Vec<(usize, TrainingWorkerNode)> =
            Vec::with_capacity(cfg.workers.len());
        let mut enso_routes = Vec::with_capacity(cfg.workers.len());
        let mut worker_keys: HashMap<String, Vec<u8>> = HashMap::new();

        for (idx, spec) in cfg.workers.iter().enumerate() {
            let key = format!("worker-hmac-key-{}-{}", spec.worker_id, cfg.seed).into_bytes();
            worker_keys.insert(spec.worker_id.clone(), key.clone());
            let shard = corpus.shard_for_worker(idx, num_initial_workers);
            let node = TrainingWorkerNode::spawn(WorkerConfig {
                worker_id: spec.worker_id.clone(),
                wan_profile: spec.wan_profile.clone(),
                signing_key: key,
                seed: cfg.seed.wrapping_add((idx as u64 + 1) * 1009),
                model_config: cfg.model_config.clone(),
                local_shard: shard,
                behavior: spec.behavior.clone(),
            })?;
            enso_routes.push(EnsoPeerRoute {
                worker_id: node.worker_id.clone(),
                region: node.wan_profile.region.clone(),
                endpoint: format!("enso://{}", node.addr),
                wan_rtt_ms: node.wan_profile.rtt_ms,
                wan_bandwidth_mbps: node.wan_profile.bandwidth_mbps,
            });
            active_workers.push((idx, node));
        }

        let manifest = HanzoTrainingManifest::new(
            cfg.model_config.config_sha256(),
            corpus.corpus_sha256.clone(),
            enso_routes,
        );

        let mut total_tokens_trained = 0u64;
        let mut total_wire_bytes = 0u64;
        let mut total_delta_upload_bytes = 0u64;
        let mut effective_wan_wall_ms = 0.0f64;
        let mut time_to_first_checkpoint_ms = 0.0f64;
        let mut completed_rounds = 0u64;
        let mut total_rejected_updates = 0usize;
        let mut all_rejection_reasons = Vec::new();
        let mut recovery_report: Option<CheckpointRecoveryReport> = None;
        let mut victim_shard_idx: usize = 1 % num_initial_workers;
        let mut rounds = Vec::with_capacity(cfg.outer_rounds as usize);

        for round_id in 1..=cfg.outer_rounds {
            let round_start = Instant::now();
            let mut round_events = Vec::new();
            let mut round_wire_bytes = 0u64;

            // Optional churn: hard-kill victim worker mid-training.
            if let Some(ref churn) = cfg.churn {
                if round_id == churn.kill_before_round {
                    for (s_idx, node) in &mut active_workers {
                        if node.worker_id == churn.victim_worker_id && node.is_alive() {
                            victim_shard_idx = *s_idx;
                            node.hard_kill();
                            round_events.push(format!(
                                "hard_killed:{} (socket closed abruptly)",
                                churn.victim_worker_id
                            ));
                        }
                    }
                }

                // Optional rejoin: spawn fresh replacement worker that fetches
                // & cryptographically verifies the latest VFS checkpoint over TCP.
                if round_id == churn.rejoin_before_round {
                    let (verified_ckpt, rec_report) =
                        TrainingWorkerNode::fetch_verified_checkpoint_over_tcp(
                            mirror.addr,
                            &current_ckpt_sha,
                            &ckpt_signing_key,
                            &churn.replacement_wan_profile,
                            &churn.replacement_worker_id,
                            round_id,
                        )
                        .map_err(|e| format!("replacement checkpoint recovery failed: {e}"))?;

                    round_wire_bytes += rec_report.recovery_wire_bytes as u64;
                    total_wire_bytes += rec_report.recovery_wire_bytes as u64;
                    effective_wan_wall_ms += rec_report.simulated_wan_ms;

                    round_events.push(format!(
                        "checkpoint_recovered:{} round={} sha256={}.. bytes={} wall_ms={:.2}",
                        rec_report.worker_id,
                        verified_ckpt.round_id,
                        &verified_ckpt.checkpoint_sha256[..12],
                        rec_report.recovery_wire_bytes,
                        rec_report.recovery_wall_ms
                    ));

                    let repl_key = format!(
                        "worker-hmac-key-{}-{}",
                        churn.replacement_worker_id, cfg.seed
                    )
                    .into_bytes();
                    worker_keys.insert(churn.replacement_worker_id.clone(), repl_key.clone());

                    let repl_shard = corpus.shard_for_worker(victim_shard_idx, num_initial_workers);
                    let repl_node = TrainingWorkerNode::spawn(WorkerConfig {
                        worker_id: churn.replacement_worker_id.clone(),
                        wan_profile: churn.replacement_wan_profile.clone(),
                        signing_key: repl_key,
                        seed: cfg.seed.wrapping_add(99_991),
                        model_config: cfg.model_config.clone(),
                        local_shard: repl_shard,
                        behavior: WorkerBehavior::Honest,
                    })?;

                    // Drop dead victim slot and insert live recovered replacement.
                    active_workers.retain(|(_, w)| w.is_alive());
                    active_workers.push((victim_shard_idx, repl_node));
                    recovery_report = Some(rec_report);
                }
            }

            // Dispatch round concurrently over real loopback TCP streams to all
            // slots (including freshly killed slots so the real ECONNREFUSED
            // failure path is exercised and survived!).
            let mut join_handles = Vec::with_capacity(active_workers.len());
            let dispatched_count = active_workers.len();

            for (shard_idx, node) in &active_workers {
                let addr = node.addr;
                let w_id = node.worker_id.clone();
                let assign = RoundAssignmentMsg {
                    round_id,
                    base_checkpoint_sha256: current_ckpt_sha.clone(),
                    inner_steps: cfg.inner_steps_per_round,
                    batch_size: cfg.batch_size,
                    inner_lr: cfg.inner_lr,
                    compression: cfg.compression,
                    shard_idx: *shard_idx as u32,
                    num_shards: num_initial_workers as u32,
                    base_params: global_params.clone(),
                };
                let assign_bytes = assign.encode();

                let h = thread::spawn(
                    move || -> (String, Result<(WorkerUpdateMsg, u64), String>) {
                        let mut stream =
                            match TcpStream::connect_timeout(&addr, Duration::from_millis(250)) {
                                Ok(s) => s,
                                Err(e) => return (w_id, Err(format!("tcp_connect_failed: {e}"))),
                            };
                        let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));
                        let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));

                        let sent = match write_frame(
                            &mut stream,
                            FrameKind::RoundAssignment,
                            &assign_bytes,
                        ) {
                            Ok(n) => n as u64,
                            Err(e) => return (w_id, Err(e)),
                        };
                        let (kind, payload, recv) = match read_frame(&mut stream) {
                            Ok(r) => r,
                            Err(e) => return (w_id, Err(e)),
                        };
                        if kind != FrameKind::WorkerUpdate {
                            return (w_id, Err(format!("unexpected_frame_kind: {kind:?}")));
                        }
                        match WorkerUpdateMsg::decode(&payload) {
                            Ok(msg) => (w_id, Ok((msg, sent + recv as u64))),
                            Err(e) => (w_id, Err(format!("decode_failed: {e}"))),
                        }
                    },
                );
                join_handles.push(h);
            }

            // Collect live TCP responses and verify cryptographic receipts.
            let mut candidates = Vec::new();
            let mut candidate_meta: HashMap<String, (u64, u64)> = HashMap::new();
            let mut round_rejections = Vec::new();
            let mut round_delta_upload = 0u64;
            let mut max_compute_ms = 0.0f64;
            let mut max_sim_wan_ms = 0.0f64;

            for handle in join_handles {
                let (w_id, res) = handle
                    .join()
                    .map_err(|_| "worker dispatch thread panicked".to_string())?;
                match res {
                    Err(err_msg) => {
                        round_events.push(format!("worker_unreachable:{w_id} ({err_msg})"));
                        round_rejections.push(format!("{w_id}:disconnected"));
                    }
                    Ok((msg, wire_bytes)) => {
                        round_wire_bytes += wire_bytes;
                        round_delta_upload += msg.encoded_delta.len() as u64;
                        if msg.compute_ms > max_compute_ms {
                            max_compute_ms = msg.compute_ms;
                        }
                        if msg.simulated_wan_ms > max_sim_wan_ms {
                            max_sim_wan_ms = msg.simulated_wan_ms;
                        }

                        // 1. Verify HMAC-SHA256 receipt signature.
                        let key = worker_keys.get(&msg.receipt.worker_id);
                        if key.is_none_or(|k| !msg.receipt.verify_hmac(k)) {
                            total_rejected_updates += 1;
                            let r = format!(
                                "{}:{}",
                                msg.receipt.worker_id,
                                RejectionReason::InvalidHmac.as_str()
                            );
                            round_rejections.push(r.clone());
                            all_rejection_reasons.push(r);
                            continue;
                        }

                        // 2. Verify round_id matches active outer round.
                        if msg.receipt.round_id != round_id {
                            total_rejected_updates += 1;
                            let r = format!(
                                "{}:{}",
                                msg.receipt.worker_id,
                                RejectionReason::StaleRoundReplay.as_str()
                            );
                            round_rejections.push(r.clone());
                            all_rejection_reasons.push(r);
                            continue;
                        }

                        // 3. Verify base checkpoint lineage matches.
                        if !ct_eq(
                            msg.receipt.base_checkpoint_sha256.as_bytes(),
                            current_ckpt_sha.as_bytes(),
                        ) {
                            total_rejected_updates += 1;
                            let r = format!(
                                "{}:{}",
                                msg.receipt.worker_id,
                                RejectionReason::BaseCheckpointMismatch.as_str()
                            );
                            round_rejections.push(r.clone());
                            all_rejection_reasons.push(r);
                            continue;
                        }

                        // 4. Verify wire delta SHA-256 digest matches signed receipt.
                        let actual_delta_sha = sha256_hex(&msg.encoded_delta);
                        if !ct_eq(
                            actual_delta_sha.as_bytes(),
                            msg.receipt.delta_sha256.as_bytes(),
                        ) {
                            total_rejected_updates += 1;
                            let r = format!(
                                "{}:{}",
                                msg.receipt.worker_id,
                                RejectionReason::DeltaDigestMismatch.as_str()
                            );
                            round_rejections.push(r.clone());
                            all_rejection_reasons.push(r);
                            continue;
                        }

                        // 5. Decode pseudo-gradient payload.
                        match decode_pseudo_gradient(&msg.encoded_delta, global_params.len()) {
                            Ok(delta) => {
                                candidate_meta.insert(
                                    msg.receipt.worker_id.clone(),
                                    (msg.receipt.tokens_processed, msg.encoded_delta.len() as u64),
                                );
                                candidates.push(CandidateDelta {
                                    worker_id: msg.receipt.worker_id,
                                    delta,
                                    inner_steps: msg.receipt.inner_steps_completed,
                                });
                            }
                            Err(_) => {
                                total_rejected_updates += 1;
                                let r = format!(
                                    "{}:{}",
                                    msg.receipt.worker_id,
                                    RejectionReason::MalformedPayload.as_str()
                                );
                                round_rejections.push(r.clone());
                                all_rejection_reasons.push(r);
                            }
                        }
                    }
                }
            }

            // Prune hard-killed workers from active list after observing the disconnect.
            active_workers.retain(|(_, w)| w.is_alive());

            // 6. Statistical + Validation-Loss Spot-Check + Byzantine Aggregation.
            let (agg_opt, verdicts) = outer_opt.screen_and_aggregate(
                &global_model,
                &global_params,
                &spot_check_batch,
                &candidates,
            );

            let mut accepted_count = 0usize;
            let mut round_tokens = 0u64;
            for v in &verdicts {
                if v.accepted {
                    accepted_count += 1;
                    if let Some((toks, _)) = candidate_meta.get(&v.worker_id) {
                        round_tokens += *toks;
                    }
                } else if let Some(ref reason) = v.rejection_reason {
                    total_rejected_updates += 1;
                    let r = format!("{}:{}", v.worker_id, reason.as_str());
                    round_rejections.push(r.clone());
                    all_rejection_reasons.push(r);
                }
            }

            if accepted_count >= cfg.min_quorum {
                if let Some(agg_delta) = agg_opt {
                    outer_opt.apply_outer_nesterov(&mut global_params, &agg_delta);
                    global_model.load_lora_params(&global_params)?;
                    completed_rounds += 1;
                    total_tokens_trained += round_tokens;
                }
            } else {
                round_events.push(format!(
                    "quorum_not_met: accepted={accepted_count} < min_quorum={}",
                    cfg.min_quorum
                ));
            }

            let eval = global_model.eval_metrics(&eval_batch);
            let ckpt = TrainingCheckpoint::build_signed(
                round_id,
                total_tokens_trained,
                current_ckpt_sha.clone(),
                cfg.model_config.config_sha256(),
                eval.loss,
                eval.perplexity,
                global_params.clone(),
                outer_opt.velocity.clone(),
                &ckpt_signing_key,
            );
            current_ckpt_sha = ckpt.checkpoint_sha256.clone();
            {
                let mut guard = shared_store.lock().expect("store lock");
                guard
                    .save_checkpoint(&ckpt)
                    .map_err(|e| format!("failed writing round {round_id} checkpoint: {e}"))?;
            }

            let round_wall_ms = round_start.elapsed().as_secs_f64() * 1_000.0;
            if round_id == 1 {
                time_to_first_checkpoint_ms = run_start.elapsed().as_secs_f64() * 1_000.0;
            }
            total_wire_bytes += round_wire_bytes;
            total_delta_upload_bytes += round_delta_upload;
            effective_wan_wall_ms += max_compute_ms + max_sim_wan_ms;

            rounds.push(RoundTelemetry {
                round_id,
                active_workers: dispatched_count,
                accepted_workers: accepted_count,
                rejected_or_failed_workers: dispatched_count.saturating_sub(accepted_count),
                rejection_reasons: round_rejections,
                val_loss: eval.loss,
                val_perplexity: eval.perplexity,
                val_top1_accuracy: eval.top1_accuracy,
                round_wall_ms,
                simulated_wan_bottleneck_ms: max_sim_wan_ms,
                round_wire_bytes,
                delta_upload_bytes: round_delta_upload,
                tokens_processed: round_tokens,
                checkpoint_sha256: current_ckpt_sha.clone(),
                events: round_events,
            });
        }

        // Gracefully shut down remaining workers.
        for (_, worker) in &mut active_workers {
            worker.shutdown();
        }
        drop(mirror);

        let total_wall_ms = run_start.elapsed().as_secs_f64() * 1_000.0;
        let final_eval = global_model.eval_metrics(&eval_batch);
        let completion_rate = if cfg.outer_rounds > 0 {
            (completed_rounds as f64) / (cfg.outer_rounds as f64)
        } else {
            1.0
        };

        let mut latencies: Vec<f64> = rounds.iter().map(|r| r.round_wall_ms).collect();
        latencies.sort_by(|a, b| a.total_cmp(b));
        let p50 = percentile(&latencies, 0.50);
        let p95 = percentile(&latencies, 0.95);
        let p99 = percentile(&latencies, 0.99);

        let throughput_tokens_per_sec = if total_wall_ms > 0.0 {
            (total_tokens_trained as f64) / (total_wall_ms / 1_000.0)
        } else {
            0.0
        };
        let effective_wan_tokens_per_sec = if effective_wan_wall_ms > 0.0 {
            (total_tokens_trained as f64) / (effective_wan_wall_ms / 1_000.0)
        } else {
            0.0
        };

        Ok(ClusterRunReport {
            scenario_name: cfg.scenario_name.clone(),
            manifest,
            trust_tier: TrustTier::SignedDeltaAndLossSpotCheck,
            initial_eval,
            final_eval,
            target_rounds: cfg.outer_rounds,
            completed_rounds,
            completion_rate,
            total_tokens_trained,
            total_wire_bytes,
            total_delta_upload_bytes,
            total_wall_ms,
            time_to_first_checkpoint_ms,
            effective_wan_wall_ms,
            throughput_tokens_per_sec,
            effective_wan_tokens_per_sec,
            round_latency_p50_ms: p50,
            round_latency_p95_ms: p95,
            round_latency_p99_ms: p99,
            total_rejected_updates,
            all_rejection_reasons,
            recovery_report,
            final_checkpoint_sha256: current_ckpt_sha,
            rounds,
            final_lora_params: global_params,
        })
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
