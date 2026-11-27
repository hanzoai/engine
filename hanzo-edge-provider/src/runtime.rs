// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Binary TCP wire protocol (`HZED` v1) and live independent loopback TCP
//! [`EdgeProviderNode`] workers with real wall-clock TTFT/ITL metering, live RC
//! thermal DVFS pacing, signed [`MeteredWorkReceipt`] issuance, automatic market
//! withdrawal, and adversarial fault injection modes.

use crate::device::{EdgeDeviceClass, WeightPrecision};
use crate::manifest::{
    EdgeProviderAdvertisement, MarketWithdrawalNotice, PrivacyPolicy, TrustTier,
    DEFAULT_ENSO_ROUTE_TOPIC, DEFAULT_VISOR_SANDBOX_PROFILE, EDGE_PROVIDER_SCHEMA_V1,
    ENGINE_COMPAT_V1,
};
use crate::meter::{compute_output_sha256, MeteredWorkReceipt, ZenEdgeMicroModel};
use crate::thermal::{ThermalPowerGovernor, ThrottleLevel, WithdrawalReason};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// 4-byte wire protocol magic prefix (`HZED` = Hanzo Edge Decentralized).
pub const WIRE_MAGIC: [u8; 4] = *b"HZED";

/// Wire protocol major version.
pub const WIRE_VERSION: u8 = 1;

/// Maximum wire frame payload size (`4 MiB` DoS guard).
pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

/// Wire error codes returned by providers and peer routers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WireErrorCode {
    /// Provider has automatically withdrawn from the market and refuses new work.
    RejectedWithdrawn = 1,
    /// Provider does not satisfy the client's required privacy policy.
    RejectedPrivacyViolation = 2,
    /// Provider advertisement exceeds device usable memory headroom.
    RejectedMemoryInfeasible = 3,
    /// Model weights digest mismatch.
    RejectedModelMismatch = 4,
    /// Invalid or forged signature on advertisement or withdrawal notice.
    RejectedInvalidSignature = 5,
    /// No active eligible peer available in the market.
    NoEligiblePeer = 6,
    /// Verification of work receipt failed (fraud detected).
    ReceiptFraudRejected = 7,
    /// Generic protocol/internal error.
    InternalError = 99,
}

impl WireErrorCode {
    const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::RejectedWithdrawn,
            2 => Self::RejectedPrivacyViolation,
            3 => Self::RejectedMemoryInfeasible,
            4 => Self::RejectedModelMismatch,
            5 => Self::RejectedInvalidSignature,
            6 => Self::NoEligiblePeer,
            7 => Self::ReceiptFraudRejected,
            _ => Self::InternalError,
        }
    }

    /// Canonical machine-readable error label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RejectedWithdrawn => "REJECTED_WITHDRAWN",
            Self::RejectedPrivacyViolation => "REJECTED_PRIVACY_VIOLATION",
            Self::RejectedMemoryInfeasible => "REJECTED_MEMORY_INFEASIBLE",
            Self::RejectedModelMismatch => "REJECTED_MODEL_MISMATCH",
            Self::RejectedInvalidSignature => "REJECTED_INVALID_SIGNATURE",
            Self::NoEligiblePeer => "NO_ELIGIBLE_PEER",
            Self::ReceiptFraudRejected => "RECEIPT_FRAUD_REJECTED",
            Self::InternalError => "INTERNAL_ERROR",
        }
    }
}

/// Request payload for an open-weight Zen inference call over the peer TCP network.
#[derive(Debug, Clone, PartialEq)]
pub struct InferZenRequest {
    /// Unique request identifier.
    pub request_id: String,
    /// Session identifier preventing cross-session replay.
    pub session_id: String,
    /// Input prompt token IDs.
    pub prompt_tokens: Vec<u32>,
    /// Maximum completion tokens requested.
    pub max_new_tokens: u32,
    /// Minimum privacy policy required by the client.
    pub required_privacy: PrivacyPolicy,
    /// Expected SHA-256 commitment of the open-weight Zen model.
    pub expected_weights_digest: String,
}

/// Response payload returned by an edge provider after executing [`InferZenRequest`].
#[derive(Debug, Clone, PartialEq)]
pub struct InferZenResponse {
    /// Generated completion token IDs.
    pub output_tokens: Vec<u32>,
    /// Signed metered work receipt with wall-clock, power, thermal, and energy telemetry.
    pub receipt: MeteredWorkReceipt,
    /// Optional signed automatic withdrawal notice if a guardrail fired during execution.
    pub withdrawal_notice: Option<MarketWithdrawalNotice>,
}

/// Live heartbeat telemetry message from an edge provider to the peer market router.
#[derive(Debug, Clone, PartialEq)]
pub struct PeerHeartbeat {
    /// Provider identifier.
    pub provider_id: String,
    /// Current temperature in °C.
    pub temp_c: f64,
    /// Instantaneous power draw in Watts.
    pub active_watts: f64,
    /// Current battery state-of-charge (%).
    pub battery_pct: f64,
    /// Observed free unified memory in GiB.
    pub free_ram_gib: f64,
    /// Active DVFS throttle level.
    pub throttle_level: ThrottleLevel,
}

/// Binary TCP wire messages exchanged across edge providers, peer routers, and clients.
#[derive(Debug, Clone, PartialEq)]
pub enum WireMessage {
    /// Register an edge provider advertisement with the peer router.
    RegisterPeer(EdgeProviderAdvertisement),
    /// Registration acknowledgement from the peer router.
    RegisterAck {
        /// Whether admission succeeded.
        accepted: bool,
        /// Human-readable reason or confirmation.
        detail: String,
    },
    /// Periodic thermal/power/memory heartbeat.
    Heartbeat(PeerHeartbeat),
    /// Heartbeat acknowledgement.
    HeartbeatAck {
        /// Whether heartbeat was accepted.
        ok: bool,
    },
    /// Open-weight Zen inference request.
    InferZenRequest(InferZenRequest),
    /// Open-weight Zen inference response + signed receipt + optional withdrawal notice.
    InferZenResponse(InferZenResponse),
    /// Signed automatic market withdrawal notice.
    WithdrawNotice(MarketWithdrawalNotice),
    /// Withdrawal acknowledgement.
    WithdrawAck {
        /// Whether withdrawal was recorded.
        recorded: bool,
    },
    /// Structured protocol error or policy rejection.
    ErrorFrame {
        /// Machine-readable error code.
        code: WireErrorCode,
        /// Human-readable error detail.
        message: String,
    },
}

/// Writes one length-prefixed `HZED` v1 frame to `stream`.
pub fn write_frame(stream: &mut TcpStream, msg: &WireMessage) -> io::Result<()> {
    let payload = encode_message(msg);
    if payload.len() > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "wire frame exceeds MAX_FRAME_BYTES",
        ));
    }
    let mut header = [0u8; 9];
    header[0..4].copy_from_slice(&WIRE_MAGIC);
    header[4] = WIRE_VERSION;
    header[5..9].copy_from_slice(&(payload.len() as u32).to_be_bytes());
    stream.write_all(&header)?;
    stream.write_all(&payload)?;
    stream.flush()
}

/// Reads one length-prefixed `HZED` v1 frame from `stream`.
pub fn read_frame(stream: &mut TcpStream) -> io::Result<WireMessage> {
    let mut header = [0u8; 9];
    stream.read_exact(&mut header)?;
    if header[0..4] != WIRE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid HZED wire magic",
        ));
    }
    if header[4] != WIRE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported HZED wire version",
        ));
    }
    let len = u32::from_be_bytes([header[5], header[6], header[7], header[8]]) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "wire frame exceeds MAX_FRAME_BYTES",
        ));
    }
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;
    decode_message(&payload)
}

/// Sends a single request-response round-trip over a fresh TCP connection to `addr`.
pub fn rpc_call(addr: SocketAddr, msg: &WireMessage, timeout: Duration) -> io::Result<WireMessage> {
    let mut stream = TcpStream::connect_timeout(&addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    write_frame(&mut stream, msg)?;
    read_frame(&mut stream)
}

/// Adversarial fault modes injected on live [`EdgeProviderNode`] workers for trust testing.
#[derive(Debug, Clone, PartialEq)]
pub enum AdversarialFaultMode {
    /// Honest provider behavior (default).
    None,
    /// Fraud 1: Inflates claimed output tokens inside receipt above emitted tokens.
    InflateOutputTokenCount {
        /// Number of bogus extra tokens claimed in the receipt.
        extra_tokens: u32,
    },
    /// Fraud 2: Executes a cheaper/wrong model seed while optionally spoofing the weights digest.
    SubstituteCheaperModel {
        /// Alternate model seed producing wrong logits/tokens.
        wrong_seed: u64,
        /// If `true`, claims the expected digest so only spot-check re-execution catches it.
        spoof_weights_digest: bool,
    },
    /// Fraud 3a: Corrupts one byte of the receipt HMAC-SHA256 tag.
    TamperReceiptHmac,
    /// Fraud 3b: Reports inconsistent energy (`joules != watts * duration_s`).
    TamperEnergyJoules {
        /// Multiplier applied to true Joules to falsify energy reporting.
        multiplier: f64,
    },
    /// Fraud 4: Continues serving requests even after issuing a signed `MarketWithdrawalNotice`.
    IgnoreWithdrawalAndKeepServing,
}

/// Runtime configuration for spawning a live [`EdgeProviderNode`].
#[derive(Debug, Clone)]
pub struct EdgeProviderConfig {
    /// Unique community provider identifier.
    pub provider_id: String,
    /// Hardware device class profile.
    pub device_class: EdgeDeviceClass,
    /// Canonical hybrid model label (`"4B-Hybrid"`, `"9B-Hybrid"`, `"27B-Hybrid"`).
    pub model_spec_label: String,
    /// Open-weight Zen model identifier.
    pub model_id: String,
    /// Weight precision advertised.
    pub precision: WeightPrecision,
    /// Maximum context length advertised.
    pub max_context_tokens: usize,
    /// Content-addressed VFS weights URI.
    pub vfs_weights_uri: String,
    /// Whether AC wall power is initially connected.
    pub on_ac_power: bool,
    /// Initial battery state-of-charge (%).
    pub initial_battery_pct: f64,
    /// Availability window start Unix timestamp (seconds).
    pub availability_start_unix: u64,
    /// Availability window end Unix timestamp (seconds).
    pub availability_end_unix: u64,
    /// Advertised privacy policy.
    pub privacy_policy: PrivacyPolicy,
    /// Provider HMAC signing secret key.
    pub secret_key: Vec<u8>,
    /// Optional override for usable unified memory cap in GiB.
    pub override_usable_memory_gib: Option<f64>,
    /// Optional thermal resistance override (°C/W) for controlled thermal tests.
    pub override_thermal_resistance: Option<f64>,
    /// Thermal time acceleration factor for live short bursts (`1.0` = real-time).
    pub thermal_time_scale: f64,
    /// Optional decentralized peer market router address to notify on withdrawal.
    pub router_addr: Option<SocketAddr>,
    /// Adversarial fault mode (default `None`).
    pub fault_mode: AdversarialFaultMode,
}

/// Live summary telemetry exported by a running [`EdgeProviderNode`].
#[derive(Debug, Clone)]
pub struct ProviderRuntimeStats {
    /// Provider identifier.
    pub provider_id: String,
    /// Total inference requests served successfully.
    pub requests_served: u64,
    /// Total requests rejected because provider had withdrawn.
    pub requests_rejected_withdrawn: u64,
    /// Current temperature in °C.
    pub current_temp_c: f64,
    /// Peak temperature observed in °C.
    pub peak_temp_c: f64,
    /// Current battery state-of-charge (%).
    pub battery_pct: f64,
    /// Cumulative energy consumed in Joules.
    pub cumulative_joules: f64,
    /// Current DVFS throttle state.
    pub throttle_level: ThrottleLevel,
    /// Highest DVFS throttle level reached during the session.
    pub max_throttle_reached: ThrottleLevel,
    /// Signed withdrawal notice if the provider withdrew from the market.
    pub withdrawal_notice: Option<MarketWithdrawalNotice>,
}

struct ProviderSharedState {
    governor: ThermalPowerGovernor,
    model: ZenEdgeMicroModel,
    substitute_model: Option<ZenEdgeMicroModel>,
    advertisement: EdgeProviderAdvertisement,
    secret_key: Vec<u8>,
    router_addr: Option<SocketAddr>,
    fault_mode: AdversarialFaultMode,
    requests_served: u64,
    requests_rejected_withdrawn: u64,
    max_throttle_reached: ThrottleLevel,
    withdrawal_notice: Option<MarketWithdrawalNotice>,
}

/// Live independent loopback TCP edge inference provider node.
pub struct EdgeProviderNode {
    addr: SocketAddr,
    provider_id: String,
    shared: Arc<Mutex<ProviderSharedState>>,
    stop_flag: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl EdgeProviderNode {
    /// Spawns a live loopback TCP edge provider bound to `127.0.0.1:0`.
    pub fn spawn(cfg: EdgeProviderConfig, model: ZenEdgeMicroModel) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;

        let mut profile = cfg.device_class.profile();
        if let Some(r_th) = cfg.override_thermal_resistance {
            profile.thermal_resistance_c_per_w = r_th;
        }
        let mut governor = ThermalPowerGovernor::new(
            profile.clone(),
            cfg.on_ac_power,
            cfg.initial_battery_pct,
            cfg.availability_start_unix,
            cfg.availability_end_unix,
        );
        governor.thermal_time_scale = cfg.thermal_time_scale;

        let usable_mem = cfg
            .override_usable_memory_gib
            .unwrap_or_else(|| profile.usable_memory_gib());

        let mut advertisement = EdgeProviderAdvertisement {
            schema_version: EDGE_PROVIDER_SCHEMA_V1.to_string(),
            engine_compat: ENGINE_COMPAT_V1.to_string(),
            provider_id: cfg.provider_id.clone(),
            device_class: cfg.device_class,
            endpoint_addr: addr.to_string(),
            usable_memory_gib: usable_mem,
            min_user_headroom_gib: profile.user_reserved_memory_gib,
            model_id: cfg.model_id,
            model_spec_label: cfg.model_spec_label,
            precision: cfg.precision,
            max_context_tokens: cfg.max_context_tokens,
            vfs_weights_uri: cfg.vfs_weights_uri,
            model_weights_digest: model.weights_digest.clone(),
            watts_cap: governor.watts_cap,
            on_ac_power: cfg.on_ac_power,
            min_battery_pct: governor.min_battery_pct,
            availability_start_unix: cfg.availability_start_unix,
            availability_end_unix: cfg.availability_end_unix,
            privacy_policy: cfg.privacy_policy,
            enso_route_topic: DEFAULT_ENSO_ROUTE_TOPIC.to_string(),
            visor_sandbox_profile: DEFAULT_VISOR_SANDBOX_PROFILE.to_string(),
            trust_tier: TrustTier::SignedTelemetryAndSpotCheck,
            hanzo_coordinator_required: false,
            ad_nonce: 1,
            signature_hex: String::new(),
        };
        advertisement.sign(&cfg.secret_key);

        let substitute_model = match &cfg.fault_mode {
            AdversarialFaultMode::SubstituteCheaperModel { wrong_seed, .. } => Some(
                ZenEdgeMicroModel::from_seed("zen-cheaper-substitute", *wrong_seed),
            ),
            _ => None,
        };

        let shared = Arc::new(Mutex::new(ProviderSharedState {
            governor,
            model,
            substitute_model,
            advertisement,
            secret_key: cfg.secret_key,
            router_addr: cfg.router_addr,
            fault_mode: cfg.fault_mode,
            requests_served: 0,
            requests_rejected_withdrawn: 0,
            max_throttle_reached: ThrottleLevel::Full,
            withdrawal_notice: None,
        }));

        let stop_flag = Arc::new(AtomicBool::new(false));
        let thread_shared = Arc::clone(&shared);
        let thread_stop = Arc::clone(&stop_flag);

        let handle = thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _peer)) => {
                        let _ = stream.set_nonblocking(false);
                        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                        let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
                        if let Ok(req) = read_frame(&mut stream) {
                            let resp = handle_provider_message(&thread_shared, req);
                            let _ = write_frame(&mut stream, &resp);
                        }
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(Self {
            addr,
            provider_id: cfg.provider_id,
            shared,
            stop_flag,
            handle: Some(handle),
        })
    }

    /// Returns the provider's bound loopback TCP socket address.
    #[must_use]
    pub const fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Returns the provider's unique identifier.
    #[must_use]
    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    /// Returns a clone of the signed `EdgeProviderAdvertisement v1`.
    #[must_use]
    pub fn advertisement(&self) -> EdgeProviderAdvertisement {
        self.shared
            .lock()
            .expect("provider lock")
            .advertisement
            .clone()
    }

    /// Returns a live snapshot of runtime thermal, power, and request counters.
    #[must_use]
    pub fn stats(&self) -> ProviderRuntimeStats {
        let guard = self.shared.lock().expect("provider lock");
        ProviderRuntimeStats {
            provider_id: self.provider_id.clone(),
            requests_served: guard.requests_served,
            requests_rejected_withdrawn: guard.requests_rejected_withdrawn,
            current_temp_c: guard.governor.temp_c,
            peak_temp_c: guard.governor.peak_temp_c,
            battery_pct: guard.governor.battery_pct,
            cumulative_joules: guard.governor.cumulative_joules,
            throttle_level: guard.governor.throttle_level,
            max_throttle_reached: guard.max_throttle_reached,
            withdrawal_notice: guard.withdrawal_notice.clone(),
        }
    }

    /// Simulates AC power unplug and immediately evaluates withdrawal guardrails.
    pub fn trigger_ac_unplug_and_battery(
        &self,
        battery_pct: f64,
    ) -> Option<MarketWithdrawalNotice> {
        let mut guard = self.shared.lock().expect("provider lock");
        guard.governor.unplug_ac();
        guard.governor.battery_pct = battery_pct.clamp(0.0, 100.0);
        evaluate_and_emit_withdrawal(&mut guard, true)
    }

    /// Simulates user memory pressure (free unified RAM dropping below user headroom).
    pub fn trigger_user_memory_pressure(
        &self,
        free_ram_gib: f64,
    ) -> Option<MarketWithdrawalNotice> {
        let mut guard = self.shared.lock().expect("provider lock");
        guard.governor.set_free_ram_gib(free_ram_gib);
        evaluate_and_emit_withdrawal(&mut guard, true)
    }

    /// Advances the provider's availability clock or marks user foreground activity active.
    pub fn trigger_window_or_foreground(
        &self,
        clock_advance_s: u64,
        user_foreground_active: bool,
    ) -> Option<MarketWithdrawalNotice> {
        let mut guard = self.shared.lock().expect("provider lock");
        guard.governor.advance_clock(clock_advance_s);
        guard
            .governor
            .set_user_foreground_active(user_foreground_active);
        evaluate_and_emit_withdrawal(&mut guard, true)
    }

    /// Stops the TCP listener thread cleanly.
    pub fn shutdown(mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for EdgeProviderNode {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn evaluate_and_emit_withdrawal(
    state: &mut ProviderSharedState,
    notify_router_now: bool,
) -> Option<MarketWithdrawalNotice> {
    let reason = state.governor.check_guardrails()?;
    if state.withdrawal_notice.is_none() {
        let mut notice = MarketWithdrawalNotice {
            provider_id: state.advertisement.provider_id.clone(),
            reason,
            observed_temp_c: state.governor.temp_c,
            observed_battery_pct: state.governor.battery_pct,
            observed_free_ram_gib: state.governor.free_ram_gib,
            withdrawn_at_unix: state.governor.now_unix,
            graceful_drain_in_flight: true,
            signature_hex: String::new(),
        };
        notice.sign(&state.secret_key);
        if notify_router_now {
            if let Some(router_addr) = state.router_addr {
                let _ = rpc_call(
                    router_addr,
                    &WireMessage::WithdrawNotice(notice.clone()),
                    Duration::from_secs(2),
                );
            }
        }
        state.withdrawal_notice = Some(notice);
    }
    state.withdrawal_notice.clone()
}

fn handle_provider_message(
    shared: &Arc<Mutex<ProviderSharedState>>,
    msg: WireMessage,
) -> WireMessage {
    match msg {
        WireMessage::InferZenRequest(req) => execute_infer_request(shared, req),
        _ => WireMessage::ErrorFrame {
            code: WireErrorCode::InternalError,
            message: "unsupported message on edge provider socket".to_string(),
        },
    }
}

fn execute_infer_request(
    shared: &Arc<Mutex<ProviderSharedState>>,
    req: InferZenRequest,
) -> WireMessage {
    let mut state = shared.lock().expect("provider lock");

    // Check guardrails before accepting new work.
    let pre_withdraw = evaluate_and_emit_withdrawal(&mut state, false);
    if pre_withdraw.is_some()
        && !matches!(
            state.fault_mode,
            AdversarialFaultMode::IgnoreWithdrawalAndKeepServing
        )
    {
        state.requests_rejected_withdrawn += 1;
        return WireMessage::ErrorFrame {
            code: WireErrorCode::RejectedWithdrawn,
            message: format!(
                "provider {} has automatically withdrawn from market",
                state.advertisement.provider_id
            ),
        };
    }

    // Enforce advertised privacy policy constraint.
    if !state
        .advertisement
        .privacy_policy
        .satisfies(req.required_privacy)
    {
        return WireMessage::ErrorFrame {
            code: WireErrorCode::RejectedPrivacyViolation,
            message: format!(
                "provider {} advertises {} which does not satisfy required {}",
                state.advertisement.provider_id,
                state.advertisement.privacy_policy,
                req.required_privacy
            ),
        };
    }

    let req_start = Instant::now();
    let mut ttft_ms = 0.0f64;
    let mut prev_tok_instant = req_start;
    let mut itl_sum_ms = 0.0f64;
    let mut itl_count = 0usize;
    let mut watts_sum = 0.0f64;
    let mut watts_samples = 0usize;
    let mut peak_temp_c = state.governor.temp_c;

    // Clone the model handle needed for execution so we can mutably step the governor inside hook.
    let active_model = match &state.substitute_model {
        Some(sub) => sub.clone(),
        None => state.model.clone(),
    };

    // Prefill thermal step (proportional to prompt length).
    let prefill_step_s = (req.prompt_tokens.len().max(1) as f64) * 0.004;
    let prefill_snap = state.governor.step(prefill_step_s, 1.0);
    watts_sum += prefill_snap.active_watts;
    watts_samples += 1;
    peak_temp_c = peak_temp_c.max(prefill_snap.temp_c);
    if prefill_snap.throttle_level > state.max_throttle_reached {
        state.max_throttle_reached = prefill_snap.throttle_level;
    }

    let output_tokens = active_model.generate_with_hook(
        &req.prompt_tokens,
        req.max_new_tokens as usize,
        |step_idx, _tok| {
            // Apply calibrated live DVFS pacing delay so throttled ITL slows measurably on wall clock.
            let pacing_us = state.governor.throttle_level.live_pacing_delay_us();
            if pacing_us > 0 {
                thread::sleep(Duration::from_micros(pacing_us));
            }
            let now = Instant::now();
            if step_idx == 0 {
                ttft_ms = now.duration_since(req_start).as_secs_f64() * 1000.0;
            } else {
                itl_sum_ms += now.duration_since(prev_tok_instant).as_secs_f64() * 1000.0;
                itl_count += 1;
            }
            prev_tok_instant = now;

            // Advance RC thermal & battery integrator for this decode token.
            let snap = state.governor.step(0.015, 1.0);
            watts_sum += snap.active_watts;
            watts_samples += 1;
            if snap.temp_c > peak_temp_c {
                peak_temp_c = snap.temp_c;
            }
            if snap.throttle_level > state.max_throttle_reached {
                state.max_throttle_reached = snap.throttle_level;
            }
        },
    );

    let total_duration_ms = req_start.elapsed().as_secs_f64() * 1000.0;
    let mean_itl_ms = if itl_count > 0 {
        itl_sum_ms / (itl_count as f64)
    } else {
        ttft_ms
    };
    let mean_watts = watts_sum / (watts_samples.max(1) as f64);
    let mut joules_consumed = mean_watts * (total_duration_ms / 1000.0);

    // Check if any withdrawal guardrail fired during this in-flight request; gracefully finish
    // this response and attach signed MarketWithdrawalNotice onto InferZenResponse for the router.
    let withdrawal_notice = evaluate_and_emit_withdrawal(&mut state, false);
    state.requests_served += 1;

    let mut claimed_output_tokens = output_tokens.len() as u32;
    let mut claimed_weights_digest = active_model.weights_digest.clone();

    match &state.fault_mode {
        AdversarialFaultMode::InflateOutputTokenCount { extra_tokens } => {
            claimed_output_tokens = claimed_output_tokens.saturating_add(*extra_tokens);
        }
        AdversarialFaultMode::SubstituteCheaperModel {
            spoof_weights_digest,
            ..
        } => {
            if *spoof_weights_digest {
                claimed_weights_digest = req.expected_weights_digest;
            }
        }
        AdversarialFaultMode::TamperEnergyJoules { multiplier } => {
            joules_consumed *= *multiplier;
        }
        AdversarialFaultMode::None
        | AdversarialFaultMode::TamperReceiptHmac
        | AdversarialFaultMode::IgnoreWithdrawalAndKeepServing => {}
    }

    let mut receipt = MeteredWorkReceipt {
        request_id: req.request_id,
        session_id: req.session_id,
        provider_id: state.advertisement.provider_id.clone(),
        model_weights_digest: claimed_weights_digest,
        prompt_tokens: req.prompt_tokens.len() as u32,
        output_tokens: claimed_output_tokens,
        output_sha256: compute_output_sha256(&req.prompt_tokens, &output_tokens),
        ttft_ms,
        mean_itl_ms,
        total_duration_ms,
        mean_watts,
        joules_consumed,
        peak_temp_c,
        throttle_level: state.governor.throttle_level,
        battery_pct_end: state.governor.battery_pct,
        trust_tier: TrustTier::SignedTelemetryAndSpotCheck,
        telemetry_self_reported: true,
        receipt_hmac_hex: String::new(),
    };
    receipt.sign(&state.secret_key);

    if matches!(state.fault_mode, AdversarialFaultMode::TamperReceiptHmac) {
        let mut bytes = receipt.receipt_hmac_hex.into_bytes();
        if let Some(first) = bytes.first_mut() {
            *first = if *first == b'0' { b'1' } else { b'0' };
        }
        receipt.receipt_hmac_hex = String::from_utf8(bytes).unwrap_or_default();
    }

    WireMessage::InferZenResponse(InferZenResponse {
        output_tokens,
        receipt,
        withdrawal_notice,
    })
}

// ---------------------------------------------------------------------------
// Deterministic Zero-Dependency Binary Codec for WireMessage
// ---------------------------------------------------------------------------

struct BinWriter(Vec<u8>);

impl BinWriter {
    fn new() -> Self {
        Self(Vec::with_capacity(256))
    }

    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }

    fn bool(&mut self, v: bool) {
        self.u8(u8::from(v));
    }

    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }

    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_bits().to_be_bytes());
    }

    fn str(&mut self, s: &str) {
        let bytes = s.as_bytes();
        self.u32(bytes.len() as u32);
        self.0.extend_from_slice(bytes);
    }

    fn u32_vec(&mut self, vals: &[u32]) {
        self.u32(vals.len() as u32);
        for &v in vals {
            self.u32(v);
        }
    }

    fn withdrawal_reason(&mut self, r: &WithdrawalReason) {
        match r {
            WithdrawalReason::ThermalCeilingExceeded { temp_c, limit_c } => {
                self.u8(1);
                self.f64(*temp_c);
                self.f64(*limit_c);
            }
            WithdrawalReason::BatteryFloorReached {
                battery_pct,
                min_battery_pct,
                on_ac_power,
            } => {
                self.u8(2);
                self.f64(*battery_pct);
                self.f64(*min_battery_pct);
                self.bool(*on_ac_power);
            }
            WithdrawalReason::AvailabilityWindowExpired {
                now_unix,
                window_end_unix,
                user_foreground_active,
            } => {
                self.u8(3);
                self.u64(*now_unix);
                self.u64(*window_end_unix);
                self.bool(*user_foreground_active);
            }
            WithdrawalReason::UserMemoryPressure {
                free_ram_gib,
                min_user_headroom_gib,
            } => {
                self.u8(4);
                self.f64(*free_ram_gib);
                self.f64(*min_user_headroom_gib);
            }
        }
    }

    fn withdrawal_notice(&mut self, n: &MarketWithdrawalNotice) {
        self.str(&n.provider_id);
        self.withdrawal_reason(&n.reason);
        self.f64(n.observed_temp_c);
        self.f64(n.observed_battery_pct);
        self.f64(n.observed_free_ram_gib);
        self.u64(n.withdrawn_at_unix);
        self.bool(n.graceful_drain_in_flight);
        self.str(&n.signature_hex);
    }

    fn receipt(&mut self, r: &MeteredWorkReceipt) {
        self.str(&r.request_id);
        self.str(&r.session_id);
        self.str(&r.provider_id);
        self.str(&r.model_weights_digest);
        self.u32(r.prompt_tokens);
        self.u32(r.output_tokens);
        self.str(&r.output_sha256);
        self.f64(r.ttft_ms);
        self.f64(r.mean_itl_ms);
        self.f64(r.total_duration_ms);
        self.f64(r.mean_watts);
        self.f64(r.joules_consumed);
        self.f64(r.peak_temp_c);
        self.str(r.throttle_level.as_str());
        self.f64(r.battery_pct_end);
        self.str(r.trust_tier.as_str());
        self.bool(r.telemetry_self_reported);
        self.str(&r.receipt_hmac_hex);
    }
}

struct BinReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> BinReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated HZED wire frame",
            ));
        }
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn bool(&mut self) -> io::Result<bool> {
        Ok(self.u8()? != 0)
    }

    fn u32(&mut self) -> io::Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> io::Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn str(&mut self) -> io::Result<String> {
        let len = self.u32()? as usize;
        let b = self.take(len)?;
        String::from_utf8(b.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid utf8 in wire frame"))
    }

    fn u32_vec(&mut self) -> io::Result<Vec<u32>> {
        let len = self.u32()? as usize;
        let mut out = Vec::with_capacity(len.min(65536));
        for _ in 0..len {
            out.push(self.u32()?);
        }
        Ok(out)
    }

    fn withdrawal_reason(&mut self) -> io::Result<WithdrawalReason> {
        match self.u8()? {
            1 => Ok(WithdrawalReason::ThermalCeilingExceeded {
                temp_c: self.f64()?,
                limit_c: self.f64()?,
            }),
            2 => Ok(WithdrawalReason::BatteryFloorReached {
                battery_pct: self.f64()?,
                min_battery_pct: self.f64()?,
                on_ac_power: self.bool()?,
            }),
            3 => Ok(WithdrawalReason::AvailabilityWindowExpired {
                now_unix: self.u64()?,
                window_end_unix: self.u64()?,
                user_foreground_active: self.bool()?,
            }),
            4 => Ok(WithdrawalReason::UserMemoryPressure {
                free_ram_gib: self.f64()?,
                min_user_headroom_gib: self.f64()?,
            }),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid withdrawal reason tag",
            )),
        }
    }

    fn withdrawal_notice(&mut self) -> io::Result<MarketWithdrawalNotice> {
        Ok(MarketWithdrawalNotice {
            provider_id: self.str()?,
            reason: self.withdrawal_reason()?,
            observed_temp_c: self.f64()?,
            observed_battery_pct: self.f64()?,
            observed_free_ram_gib: self.f64()?,
            withdrawn_at_unix: self.u64()?,
            graceful_drain_in_flight: self.bool()?,
            signature_hex: self.str()?,
        })
    }

    fn receipt(&mut self) -> io::Result<MeteredWorkReceipt> {
        let request_id = self.str()?;
        let session_id = self.str()?;
        let provider_id = self.str()?;
        let model_weights_digest = self.str()?;
        let prompt_tokens = self.u32()?;
        let output_tokens = self.u32()?;
        let output_sha256 = self.str()?;
        let ttft_ms = self.f64()?;
        let mean_itl_ms = self.f64()?;
        let total_duration_ms = self.f64()?;
        let mean_watts = self.f64()?;
        let joules_consumed = self.f64()?;
        let peak_temp_c = self.f64()?;
        let throttle_str = self.str()?;
        let throttle_level = ThrottleLevel::from_label(&throttle_str)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid throttle level"))?;
        let battery_pct_end = self.f64()?;
        let tier_str = self.str()?;
        let trust_tier = TrustTier::from_label(&tier_str)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid trust tier"))?;
        let telemetry_self_reported = self.bool()?;
        let receipt_hmac_hex = self.str()?;

        Ok(MeteredWorkReceipt {
            request_id,
            session_id,
            provider_id,
            model_weights_digest,
            prompt_tokens,
            output_tokens,
            output_sha256,
            ttft_ms,
            mean_itl_ms,
            total_duration_ms,
            mean_watts,
            joules_consumed,
            peak_temp_c,
            throttle_level,
            battery_pct_end,
            trust_tier,
            telemetry_self_reported,
            receipt_hmac_hex,
        })
    }
}

fn encode_message(msg: &WireMessage) -> Vec<u8> {
    let mut w = BinWriter::new();
    match msg {
        WireMessage::RegisterPeer(ad) => {
            w.u8(1);
            w.str(&ad.schema_version);
            w.str(&ad.engine_compat);
            w.str(&ad.provider_id);
            w.str(ad.device_class.as_str());
            w.str(&ad.endpoint_addr);
            w.f64(ad.usable_memory_gib);
            w.f64(ad.min_user_headroom_gib);
            w.str(&ad.model_id);
            w.str(&ad.model_spec_label);
            w.str(ad.precision.as_str());
            w.u32(ad.max_context_tokens as u32);
            w.str(&ad.vfs_weights_uri);
            w.str(&ad.model_weights_digest);
            w.f64(ad.watts_cap);
            w.bool(ad.on_ac_power);
            w.f64(ad.min_battery_pct);
            w.u64(ad.availability_start_unix);
            w.u64(ad.availability_end_unix);
            w.str(ad.privacy_policy.as_str());
            w.str(&ad.enso_route_topic);
            w.str(&ad.visor_sandbox_profile);
            w.str(ad.trust_tier.as_str());
            w.bool(ad.hanzo_coordinator_required);
            w.u64(ad.ad_nonce);
            w.str(&ad.signature_hex);
        }
        WireMessage::RegisterAck { accepted, detail } => {
            w.u8(2);
            w.bool(*accepted);
            w.str(detail);
        }
        WireMessage::Heartbeat(hb) => {
            w.u8(3);
            w.str(&hb.provider_id);
            w.f64(hb.temp_c);
            w.f64(hb.active_watts);
            w.f64(hb.battery_pct);
            w.f64(hb.free_ram_gib);
            w.str(hb.throttle_level.as_str());
        }
        WireMessage::HeartbeatAck { ok } => {
            w.u8(4);
            w.bool(*ok);
        }
        WireMessage::InferZenRequest(req) => {
            w.u8(5);
            w.str(&req.request_id);
            w.str(&req.session_id);
            w.u32_vec(&req.prompt_tokens);
            w.u32(req.max_new_tokens);
            w.str(req.required_privacy.as_str());
            w.str(&req.expected_weights_digest);
        }
        WireMessage::InferZenResponse(resp) => {
            w.u8(6);
            w.u32_vec(&resp.output_tokens);
            w.receipt(&resp.receipt);
            match &resp.withdrawal_notice {
                Some(n) => {
                    w.bool(true);
                    w.withdrawal_notice(n);
                }
                None => w.bool(false),
            }
        }
        WireMessage::WithdrawNotice(n) => {
            w.u8(7);
            w.withdrawal_notice(n);
        }
        WireMessage::WithdrawAck { recorded } => {
            w.u8(8);
            w.bool(*recorded);
        }
        WireMessage::ErrorFrame { code, message } => {
            w.u8(9);
            w.u8(*code as u8);
            w.str(message);
        }
    }
    w.0
}

fn decode_message(buf: &[u8]) -> io::Result<WireMessage> {
    let mut r = BinReader::new(buf);
    match r.u8()? {
        1 => {
            let schema_version = r.str()?;
            let engine_compat = r.str()?;
            let provider_id = r.str()?;
            let dev_str = r.str()?;
            let device_class = EdgeDeviceClass::from_label(&dev_str).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid device class")
            })?;
            let endpoint_addr = r.str()?;
            let usable_memory_gib = r.f64()?;
            let min_user_headroom_gib = r.f64()?;
            let model_id = r.str()?;
            let model_spec_label = r.str()?;
            let prec_str = r.str()?;
            let precision = WeightPrecision::from_label(&prec_str)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid precision"))?;
            let max_context_tokens = r.u32()? as usize;
            let vfs_weights_uri = r.str()?;
            let model_weights_digest = r.str()?;
            let watts_cap = r.f64()?;
            let on_ac_power = r.bool()?;
            let min_battery_pct = r.f64()?;
            let availability_start_unix = r.u64()?;
            let availability_end_unix = r.u64()?;
            let priv_str = r.str()?;
            let privacy_policy = PrivacyPolicy::from_label(&priv_str).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid privacy policy")
            })?;
            let enso_route_topic = r.str()?;
            let visor_sandbox_profile = r.str()?;
            let tier_str = r.str()?;
            let trust_tier = TrustTier::from_label(&tier_str)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid trust tier"))?;
            let hanzo_coordinator_required = r.bool()?;
            let ad_nonce = r.u64()?;
            let signature_hex = r.str()?;
            Ok(WireMessage::RegisterPeer(EdgeProviderAdvertisement {
                schema_version,
                engine_compat,
                provider_id,
                device_class,
                endpoint_addr,
                usable_memory_gib,
                min_user_headroom_gib,
                model_id,
                model_spec_label,
                precision,
                max_context_tokens,
                vfs_weights_uri,
                model_weights_digest,
                watts_cap,
                on_ac_power,
                min_battery_pct,
                availability_start_unix,
                availability_end_unix,
                privacy_policy,
                enso_route_topic,
                visor_sandbox_profile,
                trust_tier,
                hanzo_coordinator_required,
                ad_nonce,
                signature_hex,
            }))
        }
        2 => Ok(WireMessage::RegisterAck {
            accepted: r.bool()?,
            detail: r.str()?,
        }),
        3 => {
            let provider_id = r.str()?;
            let temp_c = r.f64()?;
            let active_watts = r.f64()?;
            let battery_pct = r.f64()?;
            let free_ram_gib = r.f64()?;
            let thr_str = r.str()?;
            let throttle_level = ThrottleLevel::from_label(&thr_str)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid throttle"))?;
            Ok(WireMessage::Heartbeat(PeerHeartbeat {
                provider_id,
                temp_c,
                active_watts,
                battery_pct,
                free_ram_gib,
                throttle_level,
            }))
        }
        4 => Ok(WireMessage::HeartbeatAck { ok: r.bool()? }),
        5 => {
            let request_id = r.str()?;
            let session_id = r.str()?;
            let prompt_tokens = r.u32_vec()?;
            let max_new_tokens = r.u32()?;
            let priv_str = r.str()?;
            let required_privacy = PrivacyPolicy::from_label(&priv_str)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid privacy"))?;
            let expected_weights_digest = r.str()?;
            Ok(WireMessage::InferZenRequest(InferZenRequest {
                request_id,
                session_id,
                prompt_tokens,
                max_new_tokens,
                required_privacy,
                expected_weights_digest,
            }))
        }
        6 => {
            let output_tokens = r.u32_vec()?;
            let receipt = r.receipt()?;
            let has_notice = r.bool()?;
            let withdrawal_notice = if has_notice {
                Some(r.withdrawal_notice()?)
            } else {
                None
            };
            Ok(WireMessage::InferZenResponse(InferZenResponse {
                output_tokens,
                receipt,
                withdrawal_notice,
            }))
        }
        7 => Ok(WireMessage::WithdrawNotice(r.withdrawal_notice()?)),
        8 => Ok(WireMessage::WithdrawAck {
            recorded: r.bool()?,
        }),
        9 => Ok(WireMessage::ErrorFrame {
            code: WireErrorCode::from_u8(r.u8()?),
            message: r.str()?,
        }),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unknown HZED message tag",
        )),
    }
}
