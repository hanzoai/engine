// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Independent decentralized peer-network market router ([`PeerMarketRouter`]) over
//! live loopback TCP sockets with zero Hanzo-controlled coordinator dependency.
//!
//! Responsibilities:
//! - Admission control of signed [`EdgeProviderAdvertisement`]s, including unclamped
//!   unified-memory residency validation via [`check_model_residency`].
//! - Privacy-policy and thermal-throttle-aware peer selection.
//! - Live `REGISTER` / `HEARTBEAT` / `INFER_ZEN_REQUEST` / `WITHDRAW` / `HANDOVER`
//!   protocol handling.
//! - Immediate recording of signed [`MarketWithdrawalNotice`]s and automatic handover
//!   to surviving peers so in-flight and subsequent bursts complete at `100%`.
//! - Independent [`ReceiptVerifier`] auditing on every served work receipt.

use crate::device::{check_model_residency, HybridModelSpec};
use crate::manifest::{EdgeProviderAdvertisement, MarketWithdrawalNotice, PrivacyPolicy};
use crate::meter::{FraudRejection, MeteredWorkReceipt, ReceiptVerifier, ZenEdgeMicroModel};
use crate::runtime::{
    read_frame, rpc_call, write_frame, InferZenRequest, InferZenResponse, PeerHeartbeat,
    WireErrorCode, WireMessage,
};
use crate::thermal::ThrottleLevel;
use std::collections::HashMap;
use std::io;
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Live entry tracked by [`PeerMarketRouter`] for each registered community edge provider.
#[derive(Debug, Clone)]
pub struct PeerMarketEntry {
    /// Verified signed provider advertisement.
    pub advertisement: EdgeProviderAdvertisement,
    /// Parsed TCP socket address for forwarding inference requests.
    pub socket_addr: SocketAddr,
    /// Registration order index (lower = preferred primary when thermal state is equal).
    pub priority_rank: usize,
    /// Last reported temperature in °C.
    pub last_temp_c: f64,
    /// Last reported active power in Watts.
    pub last_watts: f64,
    /// Last reported battery state-of-charge (%).
    pub last_battery_pct: f64,
    /// Last reported free RAM in GiB.
    pub last_free_ram_gib: f64,
    /// Last reported DVFS throttle state.
    pub throttle_level: ThrottleLevel,
    /// Whether this peer has automatically withdrawn from the market.
    pub withdrawn: bool,
    /// Verified signed withdrawal notice, if withdrawn.
    pub withdrawal_notice: Option<MarketWithdrawalNotice>,
    /// Number of verified requests routed to this peer.
    pub routed_requests: u64,
}

/// Summary telemetry of the decentralized [`PeerMarketRouter`].
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    /// Total peers currently active (eligible for routing).
    pub active_peers: usize,
    /// Total peers that have emitted verified withdrawal notices.
    pub withdrawn_peers: usize,
    /// Total requests routed and verified successfully.
    pub total_routed_verified: u64,
    /// Total automatic peer handovers executed after a peer withdrew.
    pub total_handovers: u64,
    /// Total receipts verified cleanly by the embedded independent verifier.
    pub receipts_verified: u64,
    /// Total deterministic spot checks executed.
    pub spot_checks_executed: u64,
    /// Snapshot of every registered peer entry.
    pub peers: Vec<PeerMarketEntry>,
}

/// Outcome of routing an inference request through [`PeerMarketRouter`] or [`PeerMarketClient`].
#[derive(Debug, Clone)]
pub struct RoutedInferenceOutcome {
    /// Serving provider ID.
    pub provider_id: String,
    /// Generated completion tokens.
    pub output_tokens: Vec<u32>,
    /// Verified signed work receipt.
    pub receipt: MeteredWorkReceipt,
    /// Signed withdrawal notice emitted by the provider on this request, if any.
    pub withdrawal_notice: Option<MarketWithdrawalNotice>,
    /// Whether a handover occurred (primary peer was withdrawn and surviving peer served it).
    pub handed_over: bool,
}

struct RouterSharedState {
    peers: HashMap<String, PeerMarketEntry>,
    peer_keys: HashMap<String, Vec<u8>>,
    verifier: ReceiptVerifier,
    next_priority_rank: usize,
    total_routed_verified: u64,
    total_handovers: u64,
    always_spot_check: bool,
}

/// Live independent decentralized peer-network market router bound to a loopback TCP socket.
pub struct PeerMarketRouter {
    addr: SocketAddr,
    shared: Arc<Mutex<RouterSharedState>>,
    stop_flag: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl PeerMarketRouter {
    /// Spawns a live loopback TCP `PeerMarketRouter` backed by `reference_model` for
    /// deterministic spot-check verification.
    pub fn spawn(reference_model: ZenEdgeMicroModel, always_spot_check: bool) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;

        let shared = Arc::new(Mutex::new(RouterSharedState {
            peers: HashMap::new(),
            peer_keys: HashMap::new(),
            verifier: ReceiptVerifier::new(reference_model),
            next_priority_rank: 0,
            total_routed_verified: 0,
            total_handovers: 0,
            always_spot_check,
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
                            let resp = handle_router_message(&thread_shared, req);
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
            shared,
            stop_flag,
            handle: Some(handle),
        })
    }

    /// Returns the router's bound loopback TCP address.
    #[must_use]
    pub const fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Pre-registers a community peer's HMAC authentication key in the peer key directory.
    pub fn trust_peer_key(&self, provider_id: impl Into<String>, secret_key: Vec<u8>) {
        let id = provider_id.into();
        let mut guard = self.shared.lock().expect("router lock");
        guard.peer_keys.insert(id.clone(), secret_key.clone());
        guard.verifier.register_provider_key(id, secret_key);
    }

    /// Registers a peer advertisement over the live TCP socket.
    pub fn register_peer_via_tcp(
        &self,
        ad: EdgeProviderAdvertisement,
    ) -> Result<String, (WireErrorCode, String)> {
        let reply = rpc_call(
            self.addr,
            &WireMessage::RegisterPeer(ad),
            Duration::from_secs(3),
        )
        .map_err(|e| (WireErrorCode::InternalError, e.to_string()))?;
        match reply {
            WireMessage::RegisterAck {
                accepted: true,
                detail,
            } => Ok(detail),
            WireMessage::RegisterAck {
                accepted: false,
                detail,
            } => Err((WireErrorCode::RejectedMemoryInfeasible, detail)),
            WireMessage::ErrorFrame { code, message } => Err((code, message)),
            other => Err((
                WireErrorCode::InternalError,
                format!("unexpected register reply: {other:?}"),
            )),
        }
    }

    /// Sends a heartbeat over live TCP.
    pub fn send_heartbeat_via_tcp(&self, hb: PeerHeartbeat) -> io::Result<bool> {
        let reply = rpc_call(
            self.addr,
            &WireMessage::Heartbeat(hb),
            Duration::from_secs(2),
        )?;
        Ok(matches!(reply, WireMessage::HeartbeatAck { ok: true }))
    }

    /// Submits an open-weight Zen inference request through the live TCP peer market router.
    pub fn infer_via_tcp(
        &self,
        req: InferZenRequest,
    ) -> Result<InferZenResponse, (WireErrorCode, String)> {
        let reply = rpc_call(
            self.addr,
            &WireMessage::InferZenRequest(req),
            Duration::from_secs(10),
        )
        .map_err(|e| (WireErrorCode::InternalError, e.to_string()))?;
        match reply {
            WireMessage::InferZenResponse(resp) => Ok(resp),
            WireMessage::ErrorFrame { code, message } => Err((code, message)),
            other => Err((
                WireErrorCode::InternalError,
                format!("unexpected infer reply: {other:?}"),
            )),
        }
    }

    /// Returns a live snapshot of peer market state, withdrawal notices, and verification counts.
    #[must_use]
    pub fn snapshot(&self) -> MarketSnapshot {
        let guard = self.shared.lock().expect("router lock");
        let mut peers: Vec<PeerMarketEntry> = guard.peers.values().cloned().collect();
        peers.sort_by_key(|p| p.priority_rank);
        let active_peers = peers.iter().filter(|p| !p.withdrawn).count();
        let withdrawn_peers = peers.iter().filter(|p| p.withdrawn).count();
        MarketSnapshot {
            active_peers,
            withdrawn_peers,
            total_routed_verified: guard.total_routed_verified,
            total_handovers: guard.total_handovers,
            receipts_verified: guard.verifier.receipts_verified,
            spot_checks_executed: guard.verifier.spot_checks_executed,
            peers,
        }
    }

    /// Directly evaluates admission rules for an advertisement (useful for unit/fraud checks).
    pub fn validate_advertisement_residency(
        ad: &EdgeProviderAdvertisement,
    ) -> Result<(), FraudRejection> {
        let spec = HybridModelSpec::from_label(&ad.model_spec_label).ok_or_else(|| {
            FraudRejection::InfeasibleMemoryAdvertisement {
                detail: format!("unknown hybrid model spec: {}", ad.model_spec_label),
            }
        })?;
        let profile = ad.device_class.profile();
        let fit = check_model_residency(
            &profile,
            &spec,
            ad.precision,
            ad.max_context_tokens,
            Some(ad.usable_memory_gib),
        );
        if !fit.fits {
            return Err(FraudRejection::InfeasibleMemoryAdvertisement {
                detail: fit
                    .rejection_reason
                    .unwrap_or_else(|| "insufficient unified memory".to_string()),
            });
        }
        Ok(())
    }

    /// Stops the router TCP listener thread cleanly.
    pub fn shutdown(mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for PeerMarketRouter {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn handle_router_message(shared: &Arc<Mutex<RouterSharedState>>, msg: WireMessage) -> WireMessage {
    match msg {
        WireMessage::RegisterPeer(ad) => handle_register(shared, ad),
        WireMessage::Heartbeat(hb) => handle_heartbeat(shared, hb),
        WireMessage::WithdrawNotice(notice) => handle_withdraw_notice(shared, notice),
        WireMessage::InferZenRequest(req) => handle_routed_infer(shared, req),
        _ => WireMessage::ErrorFrame {
            code: WireErrorCode::InternalError,
            message: "unsupported message on peer market router".to_string(),
        },
    }
}

fn handle_register(
    shared: &Arc<Mutex<RouterSharedState>>,
    ad: EdgeProviderAdvertisement,
) -> WireMessage {
    // 1. Validate unclamped unified-memory residency against device physical/usable cap.
    if let Err(FraudRejection::InfeasibleMemoryAdvertisement { detail }) =
        PeerMarketRouter::validate_advertisement_residency(&ad)
    {
        return WireMessage::ErrorFrame {
            code: WireErrorCode::RejectedMemoryInfeasible,
            message: detail,
        };
    }

    let socket_addr: SocketAddr = match ad.endpoint_addr.parse() {
        Ok(a) => a,
        Err(e) => {
            return WireMessage::ErrorFrame {
                code: WireErrorCode::InternalError,
                message: format!("invalid endpoint address {}: {e}", ad.endpoint_addr),
            };
        }
    };

    let mut guard = shared.lock().expect("router lock");

    // 2. Verify provider HMAC signature.
    let Some(secret) = guard.peer_keys.get(&ad.provider_id) else {
        return WireMessage::ErrorFrame {
            code: WireErrorCode::RejectedInvalidSignature,
            message: format!("unknown peer key for {}", ad.provider_id),
        };
    };
    if !ad.verify(secret) {
        return WireMessage::ErrorFrame {
            code: WireErrorCode::RejectedInvalidSignature,
            message: format!("invalid advertisement signature for {}", ad.provider_id),
        };
    }

    let rank = guard.next_priority_rank;
    guard.next_priority_rank += 1;
    let id = ad.provider_id.clone();
    let profile = ad.device_class.profile();

    guard.peers.insert(
        id.clone(),
        PeerMarketEntry {
            advertisement: ad,
            socket_addr,
            priority_rank: rank,
            last_temp_c: profile.ambient_temp_c + 10.0,
            last_watts: profile.idle_watts,
            last_battery_pct: 100.0,
            last_free_ram_gib: profile.user_reserved_memory_gib + 2.0,
            throttle_level: ThrottleLevel::Full,
            withdrawn: false,
            withdrawal_notice: None,
            routed_requests: 0,
        },
    );

    WireMessage::RegisterAck {
        accepted: true,
        detail: format!("registered community edge peer {id} (rank={rank})"),
    }
}

fn handle_heartbeat(shared: &Arc<Mutex<RouterSharedState>>, hb: PeerHeartbeat) -> WireMessage {
    let mut guard = shared.lock().expect("router lock");
    if let Some(entry) = guard.peers.get_mut(&hb.provider_id) {
        entry.last_temp_c = hb.temp_c;
        entry.last_watts = hb.active_watts;
        entry.last_battery_pct = hb.battery_pct;
        entry.last_free_ram_gib = hb.free_ram_gib;
        entry.throttle_level = hb.throttle_level;
        WireMessage::HeartbeatAck { ok: true }
    } else {
        WireMessage::HeartbeatAck { ok: false }
    }
}

fn handle_withdraw_notice(
    shared: &Arc<Mutex<RouterSharedState>>,
    notice: MarketWithdrawalNotice,
) -> WireMessage {
    let mut guard = shared.lock().expect("router lock");
    let Some(secret) = guard.peer_keys.get(&notice.provider_id) else {
        return WireMessage::ErrorFrame {
            code: WireErrorCode::RejectedInvalidSignature,
            message: format!(
                "unknown peer key for withdrawal from {}",
                notice.provider_id
            ),
        };
    };
    if !notice.verify(secret) {
        return WireMessage::ErrorFrame {
            code: WireErrorCode::RejectedInvalidSignature,
            message: format!("invalid withdrawal signature from {}", notice.provider_id),
        };
    }

    if let Some(entry) = guard.peers.get_mut(&notice.provider_id) {
        entry.withdrawn = true;
        entry.throttle_level = ThrottleLevel::AutoWithdraw;
        entry.last_temp_c = notice.observed_temp_c;
        entry.last_battery_pct = notice.observed_battery_pct;
        entry.last_free_ram_gib = notice.observed_free_ram_gib;
        entry.withdrawal_notice = Some(notice.clone());
    }
    guard
        .verifier
        .mark_provider_withdrawn(notice.provider_id.clone());

    WireMessage::WithdrawAck { recorded: true }
}

fn select_candidates(
    guard: &RouterSharedState,
    req: &InferZenRequest,
) -> Result<Vec<(String, SocketAddr)>, (WireErrorCode, String)> {
    let mut any_active_matching_model = false;
    let mut privacy_rejected_peer: Option<(PrivacyPolicy, PrivacyPolicy)> = None;

    let mut candidates: Vec<&PeerMarketEntry> = Vec::new();
    for entry in guard.peers.values() {
        if entry.withdrawn {
            continue;
        }
        if !req.expected_weights_digest.is_empty()
            && entry.advertisement.model_weights_digest != req.expected_weights_digest
        {
            continue;
        }
        any_active_matching_model = true;
        if !entry
            .advertisement
            .privacy_policy
            .satisfies(req.required_privacy)
        {
            privacy_rejected_peer =
                Some((req.required_privacy, entry.advertisement.privacy_policy));
            continue;
        }
        candidates.push(entry);
    }

    if candidates.is_empty() {
        if any_active_matching_model {
            if let Some((req_pol, adv_pol)) = privacy_rejected_peer {
                return Err((
                    WireErrorCode::RejectedPrivacyViolation,
                    format!(
                        "no active peer satisfies required privacy policy {} (available peer advertises {})",
                        req_pol, adv_pol
                    ),
                ));
            }
        }
        return Err((
            WireErrorCode::NoEligiblePeer,
            "no active eligible community edge provider available".to_string(),
        ));
    }

    // Primary priority order first (keeps primary active through its thermal progression
    // until automatic withdrawal, then immediately hands over to surviving backup peer).
    candidates.sort_by_key(|e| e.priority_rank);
    Ok(candidates
        .into_iter()
        .map(|e| (e.advertisement.provider_id.clone(), e.socket_addr))
        .collect())
}

fn handle_routed_infer(
    shared: &Arc<Mutex<RouterSharedState>>,
    req: InferZenRequest,
) -> WireMessage {
    let candidates = {
        let guard = shared.lock().expect("router lock");
        match select_candidates(&guard, &req) {
            Ok(c) => c,
            Err((code, message)) => return WireMessage::ErrorFrame { code, message },
        }
    };

    let mut attempt_idx = 0usize;
    let mut last_error: Option<(WireErrorCode, String)> = None;

    for (provider_id, peer_addr) in candidates {
        attempt_idx += 1;
        let reply = match rpc_call(
            peer_addr,
            &WireMessage::InferZenRequest(req.clone()),
            Duration::from_secs(8),
        ) {
            Ok(msg) => msg,
            Err(e) => {
                last_error = Some((
                    WireErrorCode::InternalError,
                    format!("peer {provider_id} transport error: {e}"),
                ));
                continue;
            }
        };

        match reply {
            WireMessage::ErrorFrame {
                code: WireErrorCode::RejectedWithdrawn,
                message,
            } => {
                let mut guard = shared.lock().expect("router lock");
                if let Some(entry) = guard.peers.get_mut(&provider_id) {
                    entry.withdrawn = true;
                    entry.throttle_level = ThrottleLevel::AutoWithdraw;
                }
                guard.verifier.mark_provider_withdrawn(provider_id.clone());
                guard.total_handovers += 1;
                last_error = Some((WireErrorCode::RejectedWithdrawn, message));
                continue;
            }
            WireMessage::ErrorFrame { code, message } => {
                return WireMessage::ErrorFrame { code, message };
            }
            WireMessage::InferZenResponse(resp) => {
                let mut guard = shared.lock().expect("router lock");

                // Note: if the provider attached a withdrawal notice on this final gracefully
                // drained request, verify the receipt FIRST (work was accepted before exit),
                // then immediately mark the provider withdrawn for all subsequent requests.
                let spot_check = guard.always_spot_check;
                let verify_res = guard.verifier.verify_work(
                    &req.request_id,
                    &req.session_id,
                    &req.prompt_tokens,
                    &resp.output_tokens,
                    &resp.receipt,
                    spot_check,
                );
                if let Err(fraud) = verify_res {
                    return WireMessage::ErrorFrame {
                        code: WireErrorCode::ReceiptFraudRejected,
                        message: format!("receipt fraud rejected ({}): {fraud:?}", fraud.code()),
                    };
                }

                // Update peer telemetry from verified receipt.
                if let Some(entry) = guard.peers.get_mut(&provider_id) {
                    entry.last_temp_c = resp.receipt.peak_temp_c;
                    entry.last_watts = resp.receipt.mean_watts;
                    entry.last_battery_pct = resp.receipt.battery_pct_end;
                    entry.throttle_level = resp.receipt.throttle_level;
                    entry.routed_requests += 1;
                }
                guard.total_routed_verified += 1;

                // Record verified withdrawal notice emitted on graceful drain.
                if let Some(notice) = &resp.withdrawal_notice {
                    if let Some(secret) = guard.peer_keys.get(&notice.provider_id) {
                        if notice.verify(secret) {
                            if let Some(entry) = guard.peers.get_mut(&notice.provider_id) {
                                entry.withdrawn = true;
                                entry.throttle_level = ThrottleLevel::AutoWithdraw;
                                entry.withdrawal_notice = Some(notice.clone());
                            }
                            guard
                                .verifier
                                .mark_provider_withdrawn(notice.provider_id.clone());
                            if attempt_idx == 1 {
                                guard.total_handovers += 1;
                            }
                        }
                    }
                }

                return WireMessage::InferZenResponse(resp);
            }
            other => {
                last_error = Some((
                    WireErrorCode::InternalError,
                    format!("unexpected reply from {provider_id}: {other:?}"),
                ));
            }
        }
    }

    let (code, message) = last_error.unwrap_or((
        WireErrorCode::NoEligiblePeer,
        "all candidate peers withdrew or failed".to_string(),
    ));
    WireMessage::ErrorFrame { code, message }
}
