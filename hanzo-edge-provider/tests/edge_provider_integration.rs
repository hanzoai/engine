// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive integration test suite for `hanzo-edge-provider` (HIP-1350).

#![forbid(unsafe_code)]

use hanzo_edge_provider::bench::{
    build_memory_fit_matrix, detect_repo_root, evaluate_negative_regimes,
    render_deterministic_canonical_json, render_report_json, run_full_hip1350_benchmark,
};
use hanzo_edge_provider::device::{
    check_model_residency, EdgeDeviceClass, HybridModelSpec, WeightPrecision, MIB,
};
use hanzo_edge_provider::manifest::{
    EdgeProviderAdvertisement, PrivacyPolicy, TrustTier, DEFAULT_ENSO_ROUTE_TOPIC,
    DEFAULT_VISOR_SANDBOX_PROFILE, EDGE_PROVIDER_SCHEMA_V1, ENGINE_COMPAT_V1,
};
use hanzo_edge_provider::market::PeerMarketRouter;
use hanzo_edge_provider::meter::{
    FraudRejection, ReceiptVerifier, ZenEdgeMicroModel, ZEN_MICRO_MODEL_SEED,
};
use hanzo_edge_provider::runtime::{
    rpc_call, AdversarialFaultMode, EdgeProviderConfig, EdgeProviderNode, InferZenRequest,
    PeerHeartbeat, WireErrorCode, WireMessage,
};
use hanzo_edge_provider::thermal::{ThrottleLevel, WithdrawalReason};
use std::time::Duration;

fn default_model() -> ZenEdgeMicroModel {
    ZenEdgeMicroModel::from_seed("zen-4b-hybrid-int4", ZEN_MICRO_MODEL_SEED)
}

fn make_provider_cfg(
    id: &str,
    device_class: EdgeDeviceClass,
    model_spec_label: &str,
    privacy_policy: PrivacyPolicy,
    secret_key: &[u8],
    fault_mode: AdversarialFaultMode,
) -> EdgeProviderConfig {
    let base_ts = 1_760_000_000u64;
    EdgeProviderConfig {
        provider_id: id.to_string(),
        device_class,
        model_spec_label: model_spec_label.to_string(),
        model_id: format!("zen-{}-int4", model_spec_label.to_ascii_lowercase()),
        precision: WeightPrecision::Int4,
        max_context_tokens: 2048,
        vfs_weights_uri: format!(
            "vfs://hanzo.zen/{}/weights.safetensors",
            model_spec_label.to_ascii_lowercase()
        ),
        on_ac_power: true,
        initial_battery_pct: 90.0,
        availability_start_unix: base_ts,
        availability_end_unix: base_ts + 3600,
        privacy_policy,
        secret_key: secret_key.to_vec(),
        override_usable_memory_gib: None,
        override_thermal_resistance: None,
        thermal_time_scale: 1.0,
        router_addr: None,
        fault_mode,
    }
}

#[test]
fn all_four_device_profiles_have_valid_memory_thermal_and_power_envelopes() {
    let classes = EdgeDeviceClass::all();
    assert_eq!(classes.len(), 4);
    for class in classes {
        let p = class.profile();
        assert_eq!(EdgeDeviceClass::from_label(class.as_str()), Some(class));
        assert!(p.unified_memory_gib > p.user_reserved_memory_gib);
        assert!(p.usable_memory_gib() > 0.0);
        assert!(p.soft_throttle_temp_c < p.hard_throttle_temp_c);
        assert!(p.hard_throttle_temp_c < p.withdraw_temp_c);
        assert!(p.ambient_temp_c < p.soft_throttle_temp_c);
        assert!(p.sustained_ac_watts > p.battery_cap_watts);
        assert!(p.battery_cap_watts > p.idle_watts);
        assert!(
            p.effective_bandwidth_bytes_per_sec(1.0) > p.effective_bandwidth_bytes_per_sec(0.7)
        );
        assert!(p.effective_compute_flops_per_sec(1.0) > p.effective_compute_flops_per_sec(0.45));
    }
    assert!(
        EdgeDeviceClass::MobileClass12G
            .profile()
            .is_mobile_skin_limited
    );
    assert!(
        !EdgeDeviceClass::AppleSiliconM4Pro24G
            .profile()
            .is_mobile_skin_limited
    );
}

#[test]
fn canonical_hybrid_4b_9b_27b_matrix_exact_gdn_and_kv_heads() {
    let matrix = HybridModelSpec::canonical_matrix();
    assert_eq!(matrix.len(), 3);
    assert_eq!(matrix[0].label, "4B-Hybrid");
    assert_eq!(matrix[0].num_attention_layers, 9);
    assert_eq!(matrix[0].num_gdn_layers, 27);
    assert_eq!(matrix[0].num_kv_heads, 2);
    assert!(((matrix[0].gdn_state_bytes() as f64) / MIB - 27.0).abs() < 1e-9);

    assert_eq!(matrix[1].label, "9B-Hybrid");
    assert_eq!(matrix[1].num_attention_layers, 10);
    assert_eq!(matrix[1].num_gdn_layers, 30);
    assert_eq!(matrix[1].num_kv_heads, 4);
    assert!(((matrix[1].gdn_state_bytes() as f64) / MIB - 45.0).abs() < 1e-9);

    assert_eq!(matrix[2].label, "27B-Hybrid");
    assert_eq!(matrix[2].num_attention_layers, 16);
    assert_eq!(matrix[2].num_gdn_layers, 48);
    assert_eq!(matrix[2].num_kv_heads, 4);
    assert!(((matrix[2].gdn_state_bytes() as f64) / MIB - 192.0).abs() < 1e-9);
}

#[test]
fn resident_embedding_table_vs_streamed_single_row_accounting() {
    for model in HybridModelSpec::canonical_matrix() {
        let full_embed_bytes = model.embedding_table_bytes_bf16();
        let single_row_bytes = model.embedding_row_bytes_bf16();
        assert_eq!(
            full_embed_bytes,
            (model.vocab_size as u64) * single_row_bytes
        );

        let resident = model.resident_weight_bytes(WeightPrecision::Int4);
        let streamed = model.streamed_weight_bytes_per_step(WeightPrecision::Int4);
        assert_eq!(resident - streamed, full_embed_bytes - single_row_bytes);
        assert!(
            resident - streamed > 700_000_000,
            "resident table must exceed single-row lookup by >700 MB for {}",
            model.label
        );
    }
}

#[test]
fn unclamped_memory_fit_checks_across_all_four_devices_and_three_models() {
    let matrix = build_memory_fit_matrix();
    assert_eq!(matrix.len(), 12);

    // 4B and 9B INT4 fit on all 4 device classes; 27B INT4 fits on M4 Pro/Max/Strix Halo
    // and MUST be rejected without clamping on MobileClass12G.
    for row in &matrix {
        if row.device_class == EdgeDeviceClass::MobileClass12G && row.model_label == "27B-Hybrid" {
            assert!(!row.fits, "27B-Hybrid INT4 must NOT fit on MobileClass12G");
            assert!(row.headroom_gib < -8.0);
            assert!(row.rejection_reason.is_some());
        } else {
            assert!(
                row.fits,
                "{} INT4 should fit on {} (req={:.3} GiB, cap={:.3} GiB)",
                row.model_label, row.device_class, row.total_required_gib, row.usable_memory_gib
            );
            assert!(row.headroom_gib >= 0.0);
        }
    }

    // Verify BF16 27B also cleanly fails on M4 Pro 24G without clamping.
    let m4_pro = EdgeDeviceClass::AppleSiliconM4Pro24G.profile();
    let fit_bf16 = check_model_residency(
        &m4_pro,
        &HybridModelSpec::hybrid_27b(),
        WeightPrecision::Bf16,
        2048,
        None,
    );
    assert!(!fit_bf16.fits);
    assert!(fit_bf16.total_required_gib > 45.0);
}

#[test]
fn advertisement_v1_signing_verification_and_zero_coordinator_invariant() {
    let key = b"owner-sovereign-key-v1";
    let mut ad = EdgeProviderAdvertisement {
        schema_version: EDGE_PROVIDER_SCHEMA_V1.to_string(),
        engine_compat: ENGINE_COMPAT_V1.to_string(),
        provider_id: "mac-m4-max-home".to_string(),
        device_class: EdgeDeviceClass::AppleSiliconM4Max64G,
        endpoint_addr: "127.0.0.1:41001".to_string(),
        usable_memory_gib: 48.0,
        min_user_headroom_gib: 16.0,
        model_id: "zen-27b-hybrid-int4".to_string(),
        model_spec_label: "27B-Hybrid".to_string(),
        precision: WeightPrecision::Int4,
        max_context_tokens: 4096,
        vfs_weights_uri: "vfs://hanzo.zen/27b-hybrid-int4/weights.safetensors".to_string(),
        model_weights_digest: "abcd1234".to_string(),
        watts_cap: 55.0,
        on_ac_power: true,
        min_battery_pct: 25.0,
        availability_start_unix: 1_760_000_000,
        availability_end_unix: 1_760_007_200,
        privacy_policy: PrivacyPolicy::ZeroRetentionEphemeral,
        enso_route_topic: DEFAULT_ENSO_ROUTE_TOPIC.to_string(),
        visor_sandbox_profile: DEFAULT_VISOR_SANDBOX_PROFILE.to_string(),
        trust_tier: TrustTier::SignedTelemetryAndSpotCheck,
        hanzo_coordinator_required: false,
        ad_nonce: 7,
        signature_hex: String::new(),
    };
    ad.sign(key);
    assert!(ad.verify(key));
    assert!(!ad.verify(b"wrong-key"));

    // Reject if someone flips hanzo_coordinator_required = true.
    let mut bad_coord = ad.clone();
    bad_coord.hanzo_coordinator_required = true;
    bad_coord.sign(key);
    assert!(!bad_coord.verify(key));

    // Reject non-VFS weights URI.
    let mut bad_uri = ad;
    bad_uri.vfs_weights_uri = "https://centralized.example/weights.bin".to_string();
    bad_uri.sign(key);
    assert!(!bad_uri.verify(key));
}

#[test]
fn wire_protocol_heartbeat_and_registration_roundtrip_over_tcp() {
    let model = default_model();
    let router = PeerMarketRouter::spawn(model.clone(), true).expect("router");
    let key = b"wire-roundtrip-key";
    router.trust_peer_key("wire-peer-1", key.to_vec());

    let node = EdgeProviderNode::spawn(
        make_provider_cfg(
            "wire-peer-1",
            EdgeDeviceClass::AmdStrixHalo128G,
            "4B-Hybrid",
            PrivacyPolicy::ZeroRetentionEphemeral,
            key,
            AdversarialFaultMode::None,
        ),
        model,
    )
    .expect("node");

    let ack = router
        .register_peer_via_tcp(node.advertisement())
        .expect("register ack");
    assert!(ack.contains("wire-peer-1"));

    let hb_ok = router
        .send_heartbeat_via_tcp(PeerHeartbeat {
            provider_id: "wire-peer-1".to_string(),
            temp_c: 54.2,
            active_watts: 42.0,
            battery_pct: 99.0,
            free_ram_gib: 64.0,
            throttle_level: ThrottleLevel::Full,
        })
        .expect("heartbeat");
    assert!(hb_ok);

    let snap = router.snapshot();
    assert_eq!(snap.active_peers, 1);
    assert!((snap.peers[0].last_temp_c - 54.2).abs() < 1e-6);

    node.shutdown();
    router.shutdown();
}

#[test]
fn live_tcp_mac_and_amd_peer_market_serving_with_verified_receipts() {
    let model = default_model();
    let router = PeerMarketRouter::spawn(model.clone(), true).expect("router");
    let mac_key = b"mac-serving-key";
    let amd_key = b"amd-serving-key";
    router.trust_peer_key("mac-m4-pro-community", mac_key.to_vec());
    router.trust_peer_key("amd-strix-halo-community", amd_key.to_vec());

    let mac = EdgeProviderNode::spawn(
        make_provider_cfg(
            "mac-m4-pro-community",
            EdgeDeviceClass::AppleSiliconM4Pro24G,
            "4B-Hybrid",
            PrivacyPolicy::ZeroRetentionEphemeral,
            mac_key,
            AdversarialFaultMode::None,
        ),
        model.clone(),
    )
    .expect("mac");
    let amd = EdgeProviderNode::spawn(
        make_provider_cfg(
            "amd-strix-halo-community",
            EdgeDeviceClass::AmdStrixHalo128G,
            "4B-Hybrid",
            PrivacyPolicy::ZeroRetentionEphemeral,
            amd_key,
            AdversarialFaultMode::None,
        ),
        model.clone(),
    )
    .expect("amd");

    router
        .register_peer_via_tcp(mac.advertisement())
        .expect("reg mac");
    router
        .register_peer_via_tcp(amd.advertisement())
        .expect("reg amd");

    let prompt = vec![10u32, 22, 35, 99, 141, 7, 64, 12];
    let resp = router
        .infer_via_tcp(InferZenRequest {
            request_id: "live-serve-1".to_string(),
            session_id: "sess-1".to_string(),
            prompt_tokens: prompt.clone(),
            max_new_tokens: 10,
            required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
            expected_weights_digest: model.weights_digest.clone(),
        })
        .expect("infer ok");

    assert_eq!(resp.output_tokens.len(), 10);
    assert_eq!(resp.receipt.provider_id, "mac-m4-pro-community");
    assert!(resp.receipt.verify_hmac(mac_key));
    assert_eq!(
        resp.receipt.trust_tier,
        TrustTier::SignedTelemetryAndSpotCheck
    );
    assert!(resp.receipt.telemetry_self_reported);
    assert!(resp.receipt.ttft_ms > 0.0);
    assert!(resp.receipt.joules_consumed > 0.0);

    let snap = router.snapshot();
    assert_eq!(snap.receipts_verified, 1);
    assert_eq!(snap.spot_checks_executed, 1);

    mac.shutdown();
    amd.shutdown();
    router.shutdown();
}

#[test]
fn live_tcp_progressive_dvfs_throttling_and_wall_itl_increase() {
    let model = default_model();
    let key = b"dvfs-pacing-key";
    let mut cfg = make_provider_cfg(
        "mac-dvfs-test",
        EdgeDeviceClass::AppleSiliconM4Pro24G,
        "4B-Hybrid",
        PrivacyPolicy::ZeroRetentionEphemeral,
        key,
        AdversarialFaultMode::None,
    );
    cfg.override_thermal_resistance = Some(5.5);
    cfg.thermal_time_scale = 105.0;

    let mac = EdgeProviderNode::spawn(cfg, model.clone()).expect("mac");

    let mut full_itl = None;
    let mut t70_itl = None;
    let mut t45_itl = None;

    for idx in 0..8 {
        let reply = rpc_call(
            mac.addr(),
            &WireMessage::InferZenRequest(InferZenRequest {
                request_id: format!("dvfs-req-{idx}"),
                session_id: "dvfs-sess".to_string(),
                prompt_tokens: vec![5u32; 16],
                max_new_tokens: 12,
                required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
                expected_weights_digest: model.weights_digest.clone(),
            }),
            Duration::from_secs(5),
        )
        .expect("rpc");
        if let WireMessage::InferZenResponse(resp) = reply {
            match resp.receipt.throttle_level {
                ThrottleLevel::Full => full_itl = Some(resp.receipt.mean_itl_ms),
                ThrottleLevel::ThermalThrottle70 => t70_itl = Some(resp.receipt.mean_itl_ms),
                ThrottleLevel::ThermalThrottle45 => t45_itl = Some(resp.receipt.mean_itl_ms),
                ThrottleLevel::AutoWithdraw => break,
            }
        }
    }

    let f = full_itl.expect("expected Full state");
    let s70 = t70_itl.expect("expected ThermalThrottle70 state");
    let s45 = t45_itl.expect("expected ThermalThrottle45 state");
    assert!(
        s70 > f,
        "Throttle70 ITL ({s70:.3}ms) must exceed Full ITL ({f:.3}ms)"
    );
    assert!(
        s45 > s70,
        "Throttle45 ITL ({s45:.3}ms) must exceed Throttle70 ITL ({s70:.3}ms)"
    );
    mac.shutdown();
}

#[test]
fn live_tcp_thermal_ceiling_auto_withdrawal_post_rejection_and_handover() {
    let model = default_model();
    let router = PeerMarketRouter::spawn(model.clone(), true).expect("router");
    let mac_key = b"handover-mac-key";
    let amd_key = b"handover-amd-key";
    router.trust_peer_key("mac-m4-pro-community", mac_key.to_vec());
    router.trust_peer_key("amd-strix-halo-community", amd_key.to_vec());

    let mut mac_cfg = make_provider_cfg(
        "mac-m4-pro-community",
        EdgeDeviceClass::AppleSiliconM4Pro24G,
        "4B-Hybrid",
        PrivacyPolicy::ZeroRetentionEphemeral,
        mac_key,
        AdversarialFaultMode::None,
    );
    mac_cfg.override_thermal_resistance = Some(5.5);
    mac_cfg.thermal_time_scale = 105.0;
    mac_cfg.router_addr = Some(router.addr());

    let mut amd_cfg = make_provider_cfg(
        "amd-strix-halo-community",
        EdgeDeviceClass::AmdStrixHalo128G,
        "4B-Hybrid",
        PrivacyPolicy::ZeroRetentionEphemeral,
        amd_key,
        AdversarialFaultMode::None,
    );
    amd_cfg.thermal_time_scale = 105.0;
    amd_cfg.router_addr = Some(router.addr());

    let mac = EdgeProviderNode::spawn(mac_cfg, model.clone()).expect("mac");
    let amd = EdgeProviderNode::spawn(amd_cfg, model.clone()).expect("amd");

    router
        .register_peer_via_tcp(mac.advertisement())
        .expect("reg mac");
    router
        .register_peer_via_tcp(amd.advertisement())
        .expect("reg amd");

    let mut mac_count = 0usize;
    let mut amd_count = 0usize;
    let mut saw_withdraw_notice = false;
    for idx in 0..12 {
        let resp = router
            .infer_via_tcp(InferZenRequest {
                request_id: format!("handover-req-{idx:02}"),
                session_id: "handover-sess".to_string(),
                prompt_tokens: vec![(idx as u32) + 1; 16],
                max_new_tokens: 12,
                required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
                expected_weights_digest: model.weights_digest.clone(),
            })
            .expect("100% completion across thermal handover");
        if resp.receipt.provider_id == "mac-m4-pro-community" {
            mac_count += 1;
        } else if resp.receipt.provider_id == "amd-strix-halo-community" {
            amd_count += 1;
        }
        if let Some(notice) = resp.withdrawal_notice {
            assert!(notice.verify(mac_key));
            assert_eq!(notice.reason.code(), "thermal_ceiling_exceeded");
            saw_withdraw_notice = true;
        }
    }

    assert!(
        saw_withdraw_notice,
        "Mac peer must emit signed thermal withdrawal notice"
    );
    assert!(mac_count > 0 && amd_count > 0);
    assert_eq!(mac_count + amd_count, 12);

    // Direct call to withdrawn Mac must fail with RejectedWithdrawn.
    let probe = rpc_call(
        mac.addr(),
        &WireMessage::InferZenRequest(InferZenRequest {
            request_id: "after-withdraw".to_string(),
            session_id: "handover-sess".to_string(),
            prompt_tokens: vec![1, 2, 3, 4],
            max_new_tokens: 4,
            required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
            expected_weights_digest: model.weights_digest,
        }),
        Duration::from_secs(3),
    )
    .expect("probe");
    assert!(matches!(
        probe,
        WireMessage::ErrorFrame {
            code: WireErrorCode::RejectedWithdrawn,
            ..
        }
    ));

    let snap = router.snapshot();
    assert_eq!(snap.active_peers, 1);
    assert_eq!(snap.withdrawn_peers, 1);
    assert!(snap.total_handovers >= 1);

    mac.shutdown();
    amd.shutdown();
    router.shutdown();
}

#[test]
fn guardrail_1_thermal_ceiling_withdrawal_notice_signed_and_verified() {
    let model = default_model();
    let key = b"guardrail-1-key";
    let mut cfg = make_provider_cfg(
        "thermal-node",
        EdgeDeviceClass::MobileClass12G,
        "4B-Hybrid",
        PrivacyPolicy::ZeroRetentionEphemeral,
        key,
        AdversarialFaultMode::None,
    );
    cfg.override_thermal_resistance = Some(9.0);
    cfg.thermal_time_scale = 200.0;
    let node = EdgeProviderNode::spawn(cfg, model.clone()).expect("node");

    let mut notice = None;
    for i in 0..8 {
        if let Ok(WireMessage::InferZenResponse(r)) = rpc_call(
            node.addr(),
            &WireMessage::InferZenRequest(InferZenRequest {
                request_id: format!("g1-{i}"),
                session_id: "g1".to_string(),
                prompt_tokens: vec![3; 16],
                max_new_tokens: 12,
                required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
                expected_weights_digest: model.weights_digest.clone(),
            }),
            Duration::from_secs(3),
        ) {
            if r.withdrawal_notice.is_some() {
                notice = r.withdrawal_notice;
                break;
            }
        }
    }
    let n = notice.expect("mobile thermal skin ceiling must trigger withdrawal");
    assert!(n.verify(key));
    assert!(matches!(
        n.reason,
        WithdrawalReason::ThermalCeilingExceeded { .. }
    ));
    node.shutdown();
}

#[test]
fn guardrail_2_battery_floor_withdrawal_and_handover_to_surviving_peer() {
    let model = default_model();
    let router = PeerMarketRouter::spawn(model.clone(), true).expect("router");
    let mob_key = b"mob-bat-key";
    let amd_key = b"amd-bat-key";
    router.trust_peer_key("mobile-peer", mob_key.to_vec());
    router.trust_peer_key("amd-backup", amd_key.to_vec());

    let mut mob_cfg = make_provider_cfg(
        "mobile-peer",
        EdgeDeviceClass::MobileClass12G,
        "4B-Hybrid",
        PrivacyPolicy::ZeroRetentionEphemeral,
        mob_key,
        AdversarialFaultMode::None,
    );
    mob_cfg.router_addr = Some(router.addr());
    let amd_cfg = make_provider_cfg(
        "amd-backup",
        EdgeDeviceClass::AmdStrixHalo128G,
        "4B-Hybrid",
        PrivacyPolicy::ZeroRetentionEphemeral,
        amd_key,
        AdversarialFaultMode::None,
    );

    let mob = EdgeProviderNode::spawn(mob_cfg, model.clone()).expect("mob");
    let amd = EdgeProviderNode::spawn(amd_cfg, model.clone()).expect("amd");
    router
        .register_peer_via_tcp(mob.advertisement())
        .expect("reg mob");
    router
        .register_peer_via_tcp(amd.advertisement())
        .expect("reg amd");

    let notice = mob
        .trigger_ac_unplug_and_battery(29.0)
        .expect("battery floor <=30% off AC must withdraw");
    assert!(notice.verify(mob_key));
    assert!(matches!(
        notice.reason,
        WithdrawalReason::BatteryFloorReached { .. }
    ));

    // Next request must route directly to surviving AMD peer.
    let resp = router
        .infer_via_tcp(InferZenRequest {
            request_id: "post-bat-1".to_string(),
            session_id: "bat-sess".to_string(),
            prompt_tokens: vec![7, 8, 9, 10],
            max_new_tokens: 6,
            required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
            expected_weights_digest: model.weights_digest,
        })
        .expect("handover to amd");
    assert_eq!(resp.receipt.provider_id, "amd-backup");

    mob.shutdown();
    amd.shutdown();
    router.shutdown();
}

#[test]
fn guardrail_3_availability_window_expiry_and_foreground_preemption_withdrawal() {
    let model = default_model();
    let key1 = b"win-key-1";
    let key2 = b"win-key-2";

    let n1 = EdgeProviderNode::spawn(
        make_provider_cfg(
            "win-expiry-node",
            EdgeDeviceClass::AppleSiliconM4Pro24G,
            "4B-Hybrid",
            PrivacyPolicy::ZeroRetentionEphemeral,
            key1,
            AdversarialFaultMode::None,
        ),
        model.clone(),
    )
    .expect("n1");
    let notice1 = n1
        .trigger_window_or_foreground(3600, false)
        .expect("availability window expiry must withdraw");
    assert!(notice1.verify(key1));
    assert!(matches!(
        notice1.reason,
        WithdrawalReason::AvailabilityWindowExpired {
            user_foreground_active: false,
            ..
        }
    ));

    let n2 = EdgeProviderNode::spawn(
        make_provider_cfg(
            "fg-preempt-node",
            EdgeDeviceClass::AppleSiliconM4Max64G,
            "4B-Hybrid",
            PrivacyPolicy::ZeroRetentionEphemeral,
            key2,
            AdversarialFaultMode::None,
        ),
        model,
    )
    .expect("n2");
    let notice2 = n2
        .trigger_window_or_foreground(10, true)
        .expect("user foreground preemption must withdraw immediately");
    assert!(notice2.verify(key2));
    assert!(matches!(
        notice2.reason,
        WithdrawalReason::AvailabilityWindowExpired {
            user_foreground_active: true,
            ..
        }
    ));

    n1.shutdown();
    n2.shutdown();
}

#[test]
fn guardrail_4_user_memory_pressure_withdrawal_protects_reserved_ram() {
    let model = default_model();
    let router = PeerMarketRouter::spawn(model.clone(), true).expect("router");
    let key = b"mem-pressure-key";
    router.trust_peer_key("mac-mem-node", key.to_vec());

    let mut cfg = make_provider_cfg(
        "mac-mem-node",
        EdgeDeviceClass::AppleSiliconM4Pro24G,
        "4B-Hybrid",
        PrivacyPolicy::ZeroRetentionEphemeral,
        key,
        AdversarialFaultMode::None,
    );
    cfg.router_addr = Some(router.addr());
    let node = EdgeProviderNode::spawn(cfg, model).expect("node");
    router
        .register_peer_via_tcp(node.advertisement())
        .expect("reg");

    // M4 Pro 24G reserves 8.0 GiB for user/OS; dropping free RAM to 6.0 GiB triggers exit.
    let notice = node
        .trigger_user_memory_pressure(6.0)
        .expect("user memory pressure must trigger withdrawal");
    assert!(notice.verify(key));
    assert!(matches!(
        notice.reason,
        WithdrawalReason::UserMemoryPressure { .. }
    ));
    assert_eq!(router.snapshot().withdrawn_peers, 1);

    node.shutdown();
    router.shutdown();
}

#[test]
fn fraud_1_inflated_output_token_count_rejected_over_live_tcp() {
    let model = default_model();
    let router = PeerMarketRouter::spawn(model.clone(), true).expect("router");
    let key = b"f1-key";
    router.trust_peer_key("f1-peer", key.to_vec());
    let node = EdgeProviderNode::spawn(
        make_provider_cfg(
            "f1-peer",
            EdgeDeviceClass::AppleSiliconM4Pro24G,
            "4B-Hybrid",
            PrivacyPolicy::ZeroRetentionEphemeral,
            key,
            AdversarialFaultMode::InflateOutputTokenCount { extra_tokens: 12 },
        ),
        model.clone(),
    )
    .expect("node");
    router
        .register_peer_via_tcp(node.advertisement())
        .expect("reg");

    let err = router
        .infer_via_tcp(InferZenRequest {
            request_id: "f1".to_string(),
            session_id: "s".to_string(),
            prompt_tokens: vec![1, 2, 3, 4],
            max_new_tokens: 8,
            required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
            expected_weights_digest: model.weights_digest,
        })
        .expect_err("inflated token count must fail");
    assert_eq!(err.0, WireErrorCode::ReceiptFraudRejected);
    assert!(err.1.contains("inflated_output_token_count"));

    node.shutdown();
    router.shutdown();
}

#[test]
fn fraud_2_cheaper_model_substitution_caught_by_deterministic_spot_check() {
    let model = default_model();
    let router = PeerMarketRouter::spawn(model.clone(), true).expect("router");
    let key = b"f2-key";
    router.trust_peer_key("f2-peer", key.to_vec());
    let node = EdgeProviderNode::spawn(
        make_provider_cfg(
            "f2-peer",
            EdgeDeviceClass::AmdStrixHalo128G,
            "4B-Hybrid",
            PrivacyPolicy::ZeroRetentionEphemeral,
            key,
            AdversarialFaultMode::SubstituteCheaperModel {
                wrong_seed: 0xBAD0_0001,
                spoof_weights_digest: true,
            },
        ),
        model.clone(),
    )
    .expect("node");
    router
        .register_peer_via_tcp(node.advertisement())
        .expect("reg");

    let err = router
        .infer_via_tcp(InferZenRequest {
            request_id: "f2".to_string(),
            session_id: "s".to_string(),
            prompt_tokens: vec![11, 22, 33, 44],
            max_new_tokens: 8,
            required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
            expected_weights_digest: model.weights_digest,
        })
        .expect_err("cheaper model substitution must be caught by spot-check");
    assert_eq!(err.0, WireErrorCode::ReceiptFraudRejected);
    assert!(err.1.contains("spot_check_model_substitution"));

    node.shutdown();
    router.shutdown();
}

#[test]
fn fraud_3_forged_receipt_hmac_and_energy_accounting_mismatch_rejected() {
    let model = default_model();
    // 3a: Forged HMAC.
    {
        let router = PeerMarketRouter::spawn(model.clone(), true).expect("router");
        let key = b"f3a-key";
        router.trust_peer_key("f3a-peer", key.to_vec());
        let node = EdgeProviderNode::spawn(
            make_provider_cfg(
                "f3a-peer",
                EdgeDeviceClass::AppleSiliconM4Pro24G,
                "4B-Hybrid",
                PrivacyPolicy::ZeroRetentionEphemeral,
                key,
                AdversarialFaultMode::TamperReceiptHmac,
            ),
            model.clone(),
        )
        .expect("node");
        router
            .register_peer_via_tcp(node.advertisement())
            .expect("reg");
        let err = router
            .infer_via_tcp(InferZenRequest {
                request_id: "f3a".to_string(),
                session_id: "s".to_string(),
                prompt_tokens: vec![1, 2, 3],
                max_new_tokens: 5,
                required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
                expected_weights_digest: model.weights_digest.clone(),
            })
            .expect_err("forged HMAC must fail");
        assert_eq!(err.0, WireErrorCode::ReceiptFraudRejected);
        assert!(err.1.contains("forged_receipt_hmac"));
        node.shutdown();
        router.shutdown();
    }

    // 3b: Energy mismatch.
    {
        let router = PeerMarketRouter::spawn(model.clone(), true).expect("router");
        let key = b"f3b-key";
        router.trust_peer_key("f3b-peer", key.to_vec());
        let node = EdgeProviderNode::spawn(
            make_provider_cfg(
                "f3b-peer",
                EdgeDeviceClass::AppleSiliconM4Pro24G,
                "4B-Hybrid",
                PrivacyPolicy::ZeroRetentionEphemeral,
                key,
                AdversarialFaultMode::TamperEnergyJoules { multiplier: 4.0 },
            ),
            model.clone(),
        )
        .expect("node");
        router
            .register_peer_via_tcp(node.advertisement())
            .expect("reg");
        let err = router
            .infer_via_tcp(InferZenRequest {
                request_id: "f3b".to_string(),
                session_id: "s".to_string(),
                prompt_tokens: vec![1, 2, 3],
                max_new_tokens: 5,
                required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
                expected_weights_digest: model.weights_digest,
            })
            .expect_err("falsified Joules must fail");
        assert_eq!(err.0, WireErrorCode::ReceiptFraudRejected);
        assert!(err.1.contains("energy_accounting_mismatch"));
        node.shutdown();
        router.shutdown();
    }
}

#[test]
fn fraud_4_post_withdrawal_zombie_provider_rejected_by_verifier() {
    let model = default_model();
    let key = b"f4-zombie-key";
    let node = EdgeProviderNode::spawn(
        make_provider_cfg(
            "f4-zombie",
            EdgeDeviceClass::AppleSiliconM4Pro24G,
            "4B-Hybrid",
            PrivacyPolicy::ZeroRetentionEphemeral,
            key,
            AdversarialFaultMode::IgnoreWithdrawalAndKeepServing,
        ),
        model.clone(),
    )
    .expect("node");

    let notice = node
        .trigger_user_memory_pressure(1.0)
        .expect("must emit withdrawal notice");
    let mut verifier = ReceiptVerifier::new(model.clone());
    verifier.register_provider_key("f4-zombie", key.to_vec());
    verifier.mark_provider_withdrawn(notice.provider_id);

    let prompt = vec![4u32, 5, 6, 7];
    let reply = rpc_call(
        node.addr(),
        &WireMessage::InferZenRequest(InferZenRequest {
            request_id: "f4-req".to_string(),
            session_id: "f4-sess".to_string(),
            prompt_tokens: prompt.clone(),
            max_new_tokens: 5,
            required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
            expected_weights_digest: model.weights_digest,
        }),
        Duration::from_secs(3),
    )
    .expect("rpc");

    let WireMessage::InferZenResponse(resp) = reply else {
        panic!("expected zombie response");
    };
    let err = verifier
        .verify_work(
            "f4-req",
            "f4-sess",
            &prompt,
            &resp.output_tokens,
            &resp.receipt,
            true,
        )
        .expect_err("withdrawn zombie receipt must be rejected");
    assert!(matches!(err, FraudRejection::StaleWithdrawnProvider { .. }));
    node.shutdown();
}

#[test]
fn fraud_5_oversized_27b_on_12g_mobile_rejected_at_tcp_admission() {
    let model = default_model();
    let router = PeerMarketRouter::spawn(model.clone(), true).expect("router");
    let key = b"f5-key";
    router.trust_peer_key("f5-mobile-27b", key.to_vec());

    let node = EdgeProviderNode::spawn(
        make_provider_cfg(
            "f5-mobile-27b",
            EdgeDeviceClass::MobileClass12G,
            "27B-Hybrid",
            PrivacyPolicy::ZeroRetentionEphemeral,
            key,
            AdversarialFaultMode::None,
        ),
        model,
    )
    .expect("node");

    let err = router
        .register_peer_via_tcp(node.advertisement())
        .expect_err("27B on 12G mobile must fail admission");
    assert_eq!(err.0, WireErrorCode::RejectedMemoryInfeasible);
    assert!(err.1.contains("InsufficientUnifiedMemory"));
    assert_eq!(router.snapshot().active_peers, 0);

    node.shutdown();
    router.shutdown();
}

#[test]
fn fraud_6_privacy_policy_mismatch_rejected_by_market_router() {
    let model = default_model();
    let router = PeerMarketRouter::spawn(model.clone(), true).expect("router");
    let key = b"f6-key";
    router.trust_peer_key("f6-standard", key.to_vec());

    let node = EdgeProviderNode::spawn(
        make_provider_cfg(
            "f6-standard",
            EdgeDeviceClass::AppleSiliconM4Pro24G,
            "4B-Hybrid",
            PrivacyPolicy::StandardBestEffort,
            key,
            AdversarialFaultMode::None,
        ),
        model.clone(),
    )
    .expect("node");
    router
        .register_peer_via_tcp(node.advertisement())
        .expect("reg");

    let err = router
        .infer_via_tcp(InferZenRequest {
            request_id: "f6".to_string(),
            session_id: "s".to_string(),
            prompt_tokens: vec![9, 8, 7],
            max_new_tokens: 4,
            required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
            expected_weights_digest: model.weights_digest,
        })
        .expect_err("privacy policy violation must be rejected");
    assert_eq!(err.0, WireErrorCode::RejectedPrivacyViolation);

    node.shutdown();
    router.shutdown();
}

#[test]
fn fraud_7_receipt_replay_rejected_across_duplicate_request_session() {
    let model = default_model();
    let router = PeerMarketRouter::spawn(model.clone(), true).expect("router");
    let key = b"f7-key";
    router.trust_peer_key("f7-peer", key.to_vec());

    let node = EdgeProviderNode::spawn(
        make_provider_cfg(
            "f7-peer",
            EdgeDeviceClass::AppleSiliconM4Pro24G,
            "4B-Hybrid",
            PrivacyPolicy::ZeroRetentionEphemeral,
            key,
            AdversarialFaultMode::None,
        ),
        model.clone(),
    )
    .expect("node");
    router
        .register_peer_via_tcp(node.advertisement())
        .expect("reg");

    let req = InferZenRequest {
        request_id: "replay-id-1".to_string(),
        session_id: "replay-sess-1".to_string(),
        prompt_tokens: vec![1, 3, 5, 7],
        max_new_tokens: 6,
        required_privacy: PrivacyPolicy::ZeroRetentionEphemeral,
        expected_weights_digest: model.weights_digest,
    };
    router.infer_via_tcp(req.clone()).expect("first succeeds");
    let err = router
        .infer_via_tcp(req)
        .expect_err("replayed receipt must be rejected");
    assert_eq!(err.0, WireErrorCode::ReceiptFraudRejected);
    assert!(err.1.contains("receipt_replay_detected"));

    node.shutdown();
    router.shutdown();
}

#[test]
fn negative_regimes_all_three_explicitly_rejected() {
    let negs = evaluate_negative_regimes();
    assert_eq!(negs.len(), 3);
    assert_eq!(negs[0].id, "neg_unthrottled_thermal_runaway");
    assert_eq!(negs[1].id, "neg_oversized_27b_on_12g_mobile");
    assert_eq!(negs[2].id, "neg_battery_drain_without_withdrawal");
    for nr in &negs {
        assert_eq!(
            nr.verdict, "REJECTED",
            "negative regime {} must be REJECTED",
            nr.id
        );
    }
}

#[test]
fn full_hip1350_benchmark_acceptance_gate_and_canonical_determinism() {
    let repo_root = detect_repo_root();
    let r1 = run_full_hip1350_benchmark(&repo_root, 1350).expect("bench run 1");
    let r2 = run_full_hip1350_benchmark(&repo_root, 1350).expect("bench run 2");

    assert!(r1.acceptance_gate_passed, "run 1 acceptance gate must pass");
    assert!(r2.acceptance_gate_passed, "run 2 acceptance gate must pass");

    // Verify 100% canonical determinism across repeated runs with identical seed.
    assert_eq!(r1.corpus.combined_sha256_hex, r2.corpus.combined_sha256_hex);
    assert_eq!(r1.model_weights_digest, r2.model_weights_digest);
    assert_eq!(
        r1.live_tcp_market.trace_outputs_sha256,
        r2.live_tcp_market.trace_outputs_sha256
    );
    assert_eq!(
        r1.deterministic_canonical_sha256,
        r2.deterministic_canonical_sha256
    );
    assert_eq!(
        render_deterministic_canonical_json(&r1),
        render_deterministic_canonical_json(&r2)
    );

    // Verify full JSON output is well-formed and carries zero Hanzo coordinator dependency.
    let full_json = render_report_json(&r1);
    assert!(full_json.contains("\"hanzo_coordinator_required\": false"));
    assert!(full_json.contains("\"acceptance_gate_passed\": true"));
    assert!(full_json.contains("\"detected_count\": 7"));
    assert!(full_json.contains("\"neg_unthrottled_thermal_runaway\""));
    assert!(full_json.contains("\"neg_oversized_27b_on_12g_mobile\""));
    assert!(full_json.contains("\"neg_battery_drain_without_withdrawal\""));
}
