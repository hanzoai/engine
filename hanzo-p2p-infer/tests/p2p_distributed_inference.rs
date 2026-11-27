// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive integration test suite for `hanzo-p2p-infer` (HIP-1350):
//! live 3-node loopback TCP execution, mid-stream hard-kill failover, bit-exact
//! single-node oracle parity, adversarial tamper/receipt rejection, heterogeneous
//! partitioning, network-aware MoE routing, and explicit negative regimes.

use hanzo_p2p_infer::bench::{
    detect_repo_root, evaluate_analytical_regimes, render_report_json, run_full_hip1350_benchmark,
};
use hanzo_p2p_infer::cluster::{ClusterError, LiveStageRoute, P2pPipelineClient, PeerEndpoint};
use hanzo_p2p_infer::model::{
    cosine_similarity, decode_bf16, decode_int8, encode_bf16, encode_int8, HybridModelSpec,
    LayerSlice, MicroModel, MicroModelConfig,
};
use hanzo_p2p_infer::node::{FaultMode, WorkerNode};
use hanzo_p2p_infer::partition::{
    plan_heterogeneous_pipeline, replan_after_disconnect, HanzoP2PManifest, NetworkTier,
    PartitionError, PeerCapability,
};
use hanzo_p2p_infer::router::{
    evaluate_expert_routing, ExpertAffinityTracker, ExpertPlacementPolicy,
};
use hanzo_p2p_infer::transport::{
    ActivationCodec, ProtocolErrorCode, StageReceipt, TransportError, TrustTier, WireFrame,
    MAX_FRAME_BYTES, WIRE_MAGIC, WIRE_VERSION,
};
use std::io::Cursor;
use std::time::Duration;

fn spawn_3node_live_cluster(
    cfg: MicroModelConfig,
    codec: ActivationCodec,
    timeout: Duration,
) -> (Vec<WorkerNode>, P2pPipelineClient) {
    let n0 = WorkerNode::spawn("node-a", b"secret-a".to_vec(), cfg, vec![(0, 2)], 16)
        .expect("spawn node-a");
    let n1 = WorkerNode::spawn("node-b", b"secret-b".to_vec(), cfg, vec![(2, 4)], 16)
        .expect("spawn node-b");
    let n2 = WorkerNode::spawn("node-c", b"secret-c".to_vec(), cfg, vec![(4, 6)], 16)
        .expect("spawn node-c");

    let c0 = LayerSlice::from_config(&cfg, 0, 2).commitment.digest;
    let c1 = LayerSlice::from_config(&cfg, 2, 4).commitment.digest;
    let c2 = LayerSlice::from_config(&cfg, 4, 6).commitment.digest;

    let endpoints = vec![
        PeerEndpoint {
            peer_id: n0.peer_id().to_string(),
            addr: n0.addr(),
            peer_secret: n0.peer_secret().to_vec(),
        },
        PeerEndpoint {
            peer_id: n1.peer_id().to_string(),
            addr: n1.addr(),
            peer_secret: n1.peer_secret().to_vec(),
        },
        PeerEndpoint {
            peer_id: n2.peer_id().to_string(),
            addr: n2.addr(),
            peer_secret: n2.peer_secret().to_vec(),
        },
    ];

    let routes = vec![
        LiveStageRoute {
            stage_idx: 0,
            layer_start: 0,
            layer_end: 2,
            expected_weights_commitment: c0,
            active_peer_id: "node-a".to_string(),
            failover_peer_ids: vec!["node-b".to_string(), "node-c".to_string()],
        },
        LiveStageRoute {
            stage_idx: 1,
            layer_start: 2,
            layer_end: 4,
            expected_weights_commitment: c1,
            active_peer_id: "node-b".to_string(),
            failover_peer_ids: vec!["node-c".to_string(), "node-a".to_string()],
        },
        LiveStageRoute {
            stage_idx: 2,
            layer_start: 4,
            layer_end: 6,
            expected_weights_commitment: c2,
            active_peer_id: "node-c".to_string(),
            failover_peer_ids: vec!["node-a".to_string(), "node-b".to_string()],
        },
    ];

    let client = P2pPipelineClient::new(endpoints, routes, codec, timeout);
    (vec![n0, n1, n2], client)
}

#[test]
fn canonical_hybrid_matrix_alignment_and_resident_vs_streamed_weights() {
    let [m4, m9, m27] = HybridModelSpec::canonical_matrix();

    // Exact alignment with hanzo-paged-attn/tests/kivi_kv_quant.rs:450-478
    assert_eq!(m4.label, "4B-Hybrid");
    assert_eq!(m4.num_attention_layers, 9);
    assert_eq!(m4.num_gdn_layers, 27);
    assert_eq!(m4.num_kv_heads, 2);
    assert!((m4.exact_gdn_state_mib() - 27.0).abs() < 1e-9);

    assert_eq!(m9.label, "9B-Hybrid");
    assert_eq!(m9.num_attention_layers, 10);
    assert_eq!(m9.num_gdn_layers, 30);
    assert_eq!(m9.num_kv_heads, 4);
    assert!((m9.exact_gdn_state_mib() - 45.0).abs() < 1e-9);

    assert_eq!(m27.label, "27B-Hybrid");
    assert_eq!(m27.num_attention_layers, 16);
    assert_eq!(m27.num_gdn_layers, 48);
    assert_eq!(m27.num_kv_heads, 4);
    assert!((m27.exact_gdn_state_mib() - 192.0).abs() < 1e-9);

    // Verify strict separation between resident embedding table and single-token row gather.
    for spec in [&m4, &m9, &m27] {
        let expected_emb = (spec.vocab_size as u64) * (spec.d_model as u64) * 2;
        assert_eq!(spec.embedding_table_bytes_bf16(), expected_emb);
        assert_eq!(spec.embedding_row_bytes_bf16(), (spec.d_model as u64) * 2);
        assert_eq!(
            spec.resident_weight_bytes_bf16(),
            spec.streamed_weight_bytes_bf16() + spec.embedding_table_bytes_bf16()
        );
        assert!(spec.embedding_table_bytes_bf16() > spec.embedding_row_bytes_bf16() * 100_000);
    }
}

#[test]
fn bf16_and_int8_codec_roundtrip_accuracy_and_compression() {
    let vals: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.03125).collect();

    let bf16_bytes = encode_bf16(&vals);
    assert_eq!(bf16_bytes.len(), vals.len() * 2);
    let bf16_rt = decode_bf16(&bf16_bytes).expect("decode bf16");
    let bf16_bytes_2 = encode_bf16(&bf16_rt);
    assert_eq!(
        bf16_bytes, bf16_bytes_2,
        "BF16 round-trip must be idempotent"
    );

    let int8_bytes = encode_int8(&vals);
    assert_eq!(int8_bytes.len(), vals.len() + 4);
    let int8_rt = decode_int8(&int8_bytes).expect("decode int8");
    let cos = cosine_similarity(&vals, &int8_rt);
    assert!(cos > 0.9999, "INT8 cosine similarity too low: {cos}");
    assert!(
        (bf16_bytes.len() as f64) / (int8_bytes.len() as f64) > 1.9,
        "INT8 must cut wire size nearly 2x"
    );
}

#[test]
fn wire_frame_roundtrip_and_max_frame_dos_guard() {
    let receipt = StageReceipt::sign(
        "peer-x".to_string(),
        42,
        7,
        1,
        2,
        4,
        [1u8; 32],
        [2u8; 32],
        [3u8; 32],
        b"super-secret-key",
    );
    assert!(receipt.verify_mac(b"super-secret-key"));
    assert!(!receipt.verify_mac(b"wrong-key"));

    let frame = WireFrame::ForwardResponse {
        session_id: 42,
        step_idx: 7,
        stage_idx: 1,
        codec: ActivationCodec::Bf16Lossless,
        activation_digest: [3u8; 32],
        compute_us: 1234,
        receipt,
        payload_bytes: vec![0xaa, 0xbb, 0xcc, 0xdd],
    };

    let mut wire = Vec::new();
    let written = frame.write_to_stream(&mut wire).expect("write frame");
    let mut cur = Cursor::new(wire);
    let (decoded, read) = WireFrame::read_from_stream(&mut cur).expect("read frame");
    assert_eq!(written, read);
    assert_eq!(frame, decoded);

    // Oversized frame guard (DoS protection).
    let mut oversized_header = Vec::new();
    oversized_header.extend_from_slice(&WIRE_MAGIC);
    oversized_header.push(WIRE_VERSION);
    oversized_header.push(1);
    oversized_header.extend_from_slice(&((MAX_FRAME_BYTES as u32) + 1).to_le_bytes());
    let mut bad_cur = Cursor::new(oversized_header);
    assert!(matches!(
        WireFrame::read_from_stream(&mut bad_cur),
        Err(TransportError::FrameTooLarge(_))
    ));

    // Invalid magic header guard.
    let bad_magic = b"NOPE\x01\x01\x00\x00\x00\x00";
    let mut bad_m = Cursor::new(bad_magic.as_slice());
    assert!(matches!(
        WireFrame::read_from_stream(&mut bad_m),
        Err(TransportError::InvalidHeader)
    ));
}

#[test]
fn heterogeneous_pipeline_partitioning_respects_vram_ceilings() {
    let m27 = HybridModelSpec::hybrid_27b();
    let peers = vec![
        PeerCapability::consumer_24gb("peer-0", "op-a", "eu-west", NetworkTier::LocalLanRdma),
        PeerCapability::workstation_32gb("peer-1", "op-b", "eu-west", NetworkTier::LocalLanRdma),
        PeerCapability::consumer_24gb("peer-2", "op-c", "eu-west", NetworkTier::LocalLanRdma),
    ];

    let plan = plan_heterogeneous_pipeline(&m27, &peers, 2048, 1).expect("plan 27B");
    assert_eq!(plan.stages.len(), 3);
    assert_eq!(plan.fault_tolerance_k, 1);

    let mut expected_start = 0usize;
    for (idx, stage) in plan.stages.iter().enumerate() {
        assert_eq!(stage.layer_start, expected_start);
        assert!(stage.layer_end > stage.layer_start);
        assert!(
            stage.vram_required_bytes <= peers[idx].usable_vram_bytes(),
            "stage {idx} exceeded peer VRAM"
        );
        expected_start = stage.layer_end;
    }
    assert_eq!(expected_start, m27.total_layers());

    // Fail-closed insufficient cluster VRAM when only 1x 24 GiB GPU is given for 27B-Hybrid.
    let single_24gb = vec![PeerCapability::consumer_24gb(
        "solo",
        "op-solo",
        "local",
        NetworkTier::LocalLanRdma,
    )];
    assert!(matches!(
        plan_heterogeneous_pipeline(&m27, &single_24gb, 2048, 1),
        Err(PartitionError::InsufficientClusterVram { .. })
    ));
}

#[test]
fn replan_after_disconnect_eliminates_failed_peer_and_preserves_coverage() {
    let m9 = HybridModelSpec::hybrid_9b();
    let peers = vec![
        PeerCapability::consumer_24gb("p0", "op0", "r0", NetworkTier::LocalLanRdma),
        PeerCapability::consumer_24gb("p1", "op1", "r1", NetworkTier::LocalLanRdma),
        PeerCapability::consumer_24gb("p2", "op2", "r2", NetworkTier::LocalLanRdma),
    ];
    let plan = plan_heterogeneous_pipeline(&m9, &peers, 1024, 1).expect("plan 9B");
    let replanned = replan_after_disconnect(&plan, "p1").expect("replan after p1 disconnect");

    assert_eq!(replanned.stages.len(), 3);
    for stage in &replanned.stages {
        assert_ne!(stage.primary_peer_id, "p1");
        assert!(!stage.standby_peer_ids.contains(&"p1".to_string()));
    }

    // Disconnect p2 as well -> p0 still absorbs; then disconnect p0 -> fail closed.
    let after_p2 = replan_after_disconnect(&replanned, "p2").expect("replan after p2");
    assert!(matches!(
        replan_after_disconnect(&after_p2, "p0"),
        Err(PartitionError::NoFailoverCandidate { .. })
    ));
}

#[test]
fn hanzo_p2p_manifest_v1_is_permissionless_and_versioned() {
    let m27 = HybridModelSpec::hybrid_27b();
    let peers = vec![
        PeerCapability::consumer_24gb("p0", "op0", "r0", NetworkTier::LocalLanRdma),
        PeerCapability::consumer_24gb("p1", "op1", "r1", NetworkTier::LocalLanRdma),
        PeerCapability::consumer_24gb("p2", "op2", "r2", NetworkTier::LocalLanRdma),
    ];
    let plan = plan_heterogeneous_pipeline(&m27, &peers, 2048, 1).expect("plan");
    let manifest =
        HanzoP2PManifest::from_plan(&plan, ActivationCodec::Int8PerTensor, "enso://p2p/test");

    assert_eq!(manifest.schema_version, "hanzo.p2p.infer/v1");
    assert_eq!(manifest.engine_compat, "hanzo-engine/1.7");
    assert_eq!(manifest.enso_route_id, "enso://p2p/test");
    assert!(!manifest.hanzo_coordinator_required);
    assert_eq!(
        manifest.trust_tier,
        TrustTier::CommitmentReceiptAndSpotCheck
    );
    assert_eq!(manifest.vfs_shard_uris.len(), 3);
    assert!(manifest
        .vfs_shard_uris
        .iter()
        .all(|u| u.starts_with("vfs://hanzo-p2p/27b-hybrid/")));
}

#[test]
fn network_tier_latency_and_bandwidth_ordering() {
    assert!(NetworkTier::LocalLanRdma.rtt_ms() < NetworkTier::MetroFiber.rtt_ms());
    assert!(NetworkTier::MetroFiber.rtt_ms() < NetworkTier::PublicInternet.rtt_ms());
    assert!(NetworkTier::LocalLanRdma.bandwidth_gbps() > NetworkTier::MetroFiber.bandwidth_gbps());
    assert!(
        NetworkTier::MetroFiber.bandwidth_gbps() > NetworkTier::PublicInternet.bandwidth_gbps()
    );
    assert!(
        NetworkTier::LocalLanRdma.transfer_ms(10_240)
            < NetworkTier::PublicInternet.transfer_ms(10_240)
    );
}

#[test]
fn moe_locality_aware_beats_naive_wan_all_to_all() {
    let m27 = HybridModelSpec::hybrid_27b();
    let local = evaluate_expert_routing(
        &m27,
        ExpertPlacementPolicy::LocalityAwareStageResident,
        NetworkTier::PublicInternet,
        ActivationCodec::Bf16Lossless,
        3,
        1,
    );
    let naive = evaluate_expert_routing(
        &m27,
        ExpertPlacementPolicy::NaiveCrossNetworkAllToAll,
        NetworkTier::PublicInternet,
        ActivationCodec::Bf16Lossless,
        3,
        1,
    );

    assert_eq!(local.remote_expert_hops_per_token, 0);
    assert_eq!(local.expert_wire_bytes_per_token, 0);
    assert_eq!(naive.remote_expert_hops_per_token, 128);
    assert!(naive.added_latency_ms_per_token > 3000.0);
    assert!(naive.network_ceiling_tok_per_sec < 0.35);

    let mut tracker = ExpertAffinityTracker::new(6, 4);
    tracker.record(0, &[1, 3]);
    tracker.record(0, &[3, 2]);
    assert_eq!(tracker.hottest_expert(0), Some((3, 2)));
    assert_eq!(tracker.total_dispatches(), 4);
}

#[test]
fn live_tcp_health_ping_and_backpressure_rejection() {
    let cfg = MicroModelConfig::default_live_6layer();
    let model = MicroModel::from_config(cfg);

    // max_concurrent_sessions = 1 enforces strict backpressure on session 2.
    let node = WorkerNode::spawn("bp-node", b"bp-sec".to_vec(), cfg, vec![(0, 6)], 1)
        .expect("spawn bp-node");
    let c_full = LayerSlice::from_config(&cfg, 0, 6).commitment.digest;

    let mut client = P2pPipelineClient::new(
        vec![PeerEndpoint {
            peer_id: node.peer_id().to_string(),
            addr: node.addr(),
            peer_secret: node.peer_secret().to_vec(),
        }],
        vec![LiveStageRoute {
            stage_idx: 0,
            layer_start: 0,
            layer_end: 6,
            expected_weights_commitment: c_full,
            active_peer_id: "bp-node".to_string(),
            failover_peer_ids: vec![],
        }],
        ActivationCodec::Bf16Lossless,
        Duration::from_millis(250),
    );

    let pong = client.health_ping("bp-node", 777).expect("health ping");
    assert!(matches!(
        pong,
        WireFrame::HealthPong {
            nonce: 777,
            ref peer_id,
            ..
        } if peer_id == "bp-node"
    ));

    // Session 1 succeeds.
    client
        .generate(&model, 1, &[10, 20], 1)
        .expect("session 1 should pass");

    // Session 2 is rejected via ProtocolErrorCode::BackpressureRejected.
    let err = client
        .generate(&model, 2, &[10, 20], 1)
        .expect_err("session 2 must trigger backpressure");
    match err {
        ClusterError::RemoteProtocolError { detail, .. } => {
            assert!(
                detail.contains(&format!("{:?}", ProtocolErrorCode::BackpressureRejected)),
                "expected BackpressureRejected, got: {detail}"
            );
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn live_3node_tcp_bf16_bit_exact_vs_single_node_oracle() {
    let cfg = MicroModelConfig::default_live_6layer();
    let oracle = MicroModel::from_config(cfg);
    let prompt: Vec<u32> = (0..12).map(|i| (i * 17 + 5) % 256).collect();
    let gen_tokens = 16;

    let (ref_tokens, ref_logits) = oracle.generate_reference(&prompt, gen_tokens, &[2, 4, 6]);

    let (_nodes, mut client) = spawn_3node_live_cluster(
        cfg,
        ActivationCodec::Bf16Lossless,
        Duration::from_millis(250),
    );
    client.set_spot_check_stride(Some(2));
    let report = client
        .generate(&oracle, 2001, &prompt, gen_tokens)
        .expect("3-node BF16 run");

    assert_eq!(report.generated_tokens, ref_tokens);
    assert_eq!(report.failover_events.len(), 0);
    assert_eq!(report.receipts.len(), (prompt.len() + gen_tokens) * 3);
    assert!(report.spot_checks_passed > 0);
    assert!(report.wire_bytes_sent > 0 && report.wire_bytes_recv > 0);

    let mut max_diff = 0.0f32;
    for (r, g) in ref_logits.iter().zip(report.generated_logits.iter()) {
        for (&a, &b) in r.iter().zip(g.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
    }
    assert_eq!(
        max_diff, 0.0,
        "BF16 3-node TCP pipeline must be bit-exact vs oracle"
    );
}

#[test]
fn live_3node_tcp_int8_activation_compression_high_fidelity() {
    let cfg = MicroModelConfig::default_live_6layer();
    let oracle = MicroModel::from_config(cfg);
    let prompt: Vec<u32> = (0..10).map(|i| (i * 23 + 3) % 256).collect();
    let gen_tokens = 12;

    let (_ref_tokens, ref_logits) = oracle.generate_reference(&prompt, gen_tokens, &[2, 4, 6]);

    let (_n_bf16, mut c_bf16) = spawn_3node_live_cluster(
        cfg,
        ActivationCodec::Bf16Lossless,
        Duration::from_millis(250),
    );
    let rep_bf16 = c_bf16
        .generate(&oracle, 2101, &prompt, gen_tokens)
        .expect("bf16 run");

    let (_n_int8, mut c_int8) = spawn_3node_live_cluster(
        cfg,
        ActivationCodec::Int8PerTensor,
        Duration::from_millis(250),
    );
    let rep_int8 = c_int8
        .generate(&oracle, 2102, &prompt, gen_tokens)
        .expect("int8 run");

    assert!(
        rep_int8.wire_bytes_sent < rep_bf16.wire_bytes_sent,
        "INT8 wire bytes ({}) must be smaller than BF16 ({})",
        rep_int8.wire_bytes_sent,
        rep_bf16.wire_bytes_sent
    );

    let mean_cos: f64 = ref_logits
        .iter()
        .zip(rep_int8.generated_logits.iter())
        .map(|(r, g)| cosine_similarity(r, g))
        .sum::<f64>()
        / (gen_tokens as f64);
    assert!(
        mean_cos > 0.995,
        "INT8 mean logit cosine too low: {mean_cos}"
    );
}

#[test]
fn live_3node_tcp_hard_kill_node_b_mid_generation_failover_bit_exact() {
    let cfg = MicroModelConfig::default_live_6layer();
    let oracle = MicroModel::from_config(cfg);
    let prompt: Vec<u32> = (0..16).map(|i| (i * 19 + 11) % 256).collect();
    let gen_tokens = 16;

    let (ref_tokens, ref_logits) = oracle.generate_reference(&prompt, gen_tokens, &[2, 4, 6]);

    let (mut nodes, mut client) = spawn_3node_live_cluster(
        cfg,
        ActivationCodec::Bf16Lossless,
        Duration::from_millis(250),
    );

    // Mid-generation after emitting 8 tokens, HARD-KILL Node B (close listener + drop TCP sockets).
    let report = client
        .generate_with_hook(&oracle, 3001, &prompt, gen_tokens, |emitted| {
            if emitted == 8 {
                nodes[1].hard_kill();
            }
        })
        .expect("3-node cluster must tolerate Node B hard-kill");

    assert!(nodes[1].is_killed());
    assert_eq!(report.failover_events.len(), 1);
    let ev = &report.failover_events[0];
    assert_eq!(ev.failed_peer_id, "node-b");
    assert_eq!(ev.replacement_peer_id, "node-c");
    assert_eq!(ev.stage_idx, 1);
    assert_eq!(ev.replayed_tokens, prompt.len() + 7);
    assert!(ev.recovery_latency_ms >= 0.0);

    // Verify 100% completion and bit-exact 0.000e0 match vs monolithic reference oracle!
    assert_eq!(report.generated_tokens, ref_tokens);
    let mut max_diff = 0.0f32;
    for (r, g) in ref_logits.iter().zip(report.generated_logits.iter()) {
        for (&a, &b) in r.iter().zip(g.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
    }
    assert_eq!(
        max_diff, 0.0,
        "post-failover logits must match monolithic oracle bit-exactly"
    );
}

#[test]
fn live_3node_tcp_hard_kill_node_a_stage0_failover_bit_exact() {
    let cfg = MicroModelConfig::default_live_6layer();
    let oracle = MicroModel::from_config(cfg);
    let prompt: Vec<u32> = (0..8).map(|i| (i * 13 + 7) % 256).collect();
    let gen_tokens = 10;

    let (ref_tokens, _) = oracle.generate_reference(&prompt, gen_tokens, &[2, 4, 6]);
    let (mut nodes, mut client) = spawn_3node_live_cluster(
        cfg,
        ActivationCodec::Bf16Lossless,
        Duration::from_millis(250),
    );

    let report = client
        .generate_with_hook(&oracle, 3002, &prompt, gen_tokens, |emitted| {
            if emitted == 4 {
                nodes[0].hard_kill();
            }
        })
        .expect("3-node cluster must tolerate stage-0 Node A hard-kill");

    assert_eq!(report.failover_events.len(), 1);
    assert_eq!(report.failover_events[0].failed_peer_id, "node-a");
    assert_eq!(report.failover_events[0].replacement_peer_id, "node-b");
    assert_eq!(report.generated_tokens, ref_tokens);
}

#[test]
fn live_tcp_rejects_tampered_activation_wire_bytes() {
    let cfg = MicroModelConfig::default_live_6layer();
    let oracle = MicroModel::from_config(cfg);
    let (nodes, mut client) = spawn_3node_live_cluster(
        cfg,
        ActivationCodec::Bf16Lossless,
        Duration::from_millis(250),
    );
    nodes[1].set_fault_mode(FaultMode::TamperActivationPayload);

    let err = client
        .generate(&oracle, 4001, &[1, 2, 3], 2)
        .expect_err("tampered payload must fail");
    assert!(matches!(
        err,
        ClusterError::ActivationIntegrityMismatch { stage_idx: 1, .. }
    ));
}

#[test]
fn live_tcp_rejects_silent_numeric_tamper_via_spot_check() {
    let cfg = MicroModelConfig::default_live_6layer();
    let oracle = MicroModel::from_config(cfg);
    let (nodes, mut client) = spawn_3node_live_cluster(
        cfg,
        ActivationCodec::Bf16Lossless,
        Duration::from_millis(250),
    );
    nodes[1].set_fault_mode(FaultMode::SilentNumericTamper);
    client.set_spot_check_stride(Some(1));

    let err = client
        .generate(&oracle, 4002, &[1, 2, 3], 2)
        .expect_err("silent numerical tamper must be caught by spot-check");
    assert!(matches!(
        err,
        ClusterError::SpotCheckNumericMismatch {
            stage_idx: 1,
            max_abs_diff,
            ..
        } if max_abs_diff > 0.1
    ));
}

#[test]
fn live_tcp_rejects_wrong_weights_commitment() {
    let cfg = MicroModelConfig::default_live_6layer();
    let oracle = MicroModel::from_config(cfg);
    let (nodes, mut client) = spawn_3node_live_cluster(
        cfg,
        ActivationCodec::Bf16Lossless,
        Duration::from_millis(250),
    );
    nodes[1].set_fault_mode(FaultMode::WrongWeightsCommitment);

    let err = client
        .generate(&oracle, 4003, &[1, 2, 3], 2)
        .expect_err("wrong weights commitment must fail");
    assert!(matches!(
        err,
        ClusterError::WeightsCommitmentMismatch { stage_idx: 1, .. }
    ));
}

#[test]
fn live_tcp_rejects_forged_hmac_receipt() {
    let cfg = MicroModelConfig::default_live_6layer();
    let oracle = MicroModel::from_config(cfg);
    let (nodes, mut client) = spawn_3node_live_cluster(
        cfg,
        ActivationCodec::Bf16Lossless,
        Duration::from_millis(250),
    );
    nodes[1].set_fault_mode(FaultMode::ForgedReceiptMac);

    let err = client
        .generate(&oracle, 4004, &[1, 2, 3], 2)
        .expect_err("forged HMAC receipt must fail");
    assert!(matches!(
        err,
        ClusterError::ForgedReceiptMac { stage_idx: 1, .. }
    ));
}

#[test]
fn live_tcp_rejects_cross_session_receipt_replay() {
    let cfg = MicroModelConfig::default_live_6layer();
    let oracle = MicroModel::from_config(cfg);
    let (nodes, mut client) = spawn_3node_live_cluster(
        cfg,
        ActivationCodec::Bf16Lossless,
        Duration::from_millis(250),
    );
    nodes[1].set_fault_mode(FaultMode::ReplayReceiptFromSession(88_888));

    let err = client
        .generate(&oracle, 4005, &[1, 2, 3], 2)
        .expect_err("cross-session receipt replay must fail");
    assert!(matches!(
        err,
        ClusterError::ReceiptBindingMismatch { stage_idx: 1, .. }
    ));
}

#[test]
fn live_tcp_detects_stalled_peer_timeout_and_fails_over() {
    let cfg = MicroModelConfig::default_live_6layer();
    let oracle = MicroModel::from_config(cfg);
    let prompt = [4u32, 9, 15, 22];
    let (ref_tokens, _) = oracle.generate_reference(&prompt, 4, &[2, 4, 6]);

    let (nodes, mut client) = spawn_3node_live_cluster(
        cfg,
        ActivationCodec::Bf16Lossless,
        Duration::from_millis(60),
    );
    nodes[1].set_fault_mode(FaultMode::StallMillis(180));

    let report = client
        .generate(&oracle, 4006, &prompt, 4)
        .expect("stalled node-b should time out and fail over to node-c");
    assert_eq!(report.failover_events.len(), 1);
    assert_eq!(report.failover_events[0].failed_peer_id, "node-b");
    assert_eq!(report.failover_events[0].replacement_peer_id, "node-c");
    assert_eq!(report.generated_tokens, ref_tokens);
}

#[test]
fn analytical_baselines_lan_p2p_beats_pcie_offload_and_rejects_all_three_negative_regimes() {
    let (regimes, negatives, privacy) = evaluate_analytical_regimes();

    let pcie = regimes
        .iter()
        .find(|r| r.regime_id == "conventional_pcie_offload_24gb_b1")
        .expect("pcie baseline");
    let lan = regimes
        .iter()
        .find(|r| r.regime_id == "p2p_3node_local_lan_bf16_b1")
        .expect("lan p2p");
    let h100 = regimes
        .iter()
        .find(|r| r.regime_id == "single_node_h100_80gb_b1")
        .expect("h100 baseline");

    assert!(
        lan.aggregate_decode_tok_per_sec > pcie.aggregate_decode_tok_per_sec * 15.0,
        "Local LAN 3-node P2P ({:.2} tok/s) must beat PCIe offload ({:.2} tok/s) by >15x at B=1",
        lan.aggregate_decode_tok_per_sec,
        pcie.aggregate_decode_tok_per_sec
    );
    let lan_b16 = regimes
        .iter()
        .find(|r| r.regime_id == "p2p_3node_local_lan_int8_b16")
        .expect("lan b16");
    assert!(
        lan_b16.aggregate_decode_tok_per_sec > lan.aggregate_decode_tok_per_sec * 6.0,
        "microbatched B=16 LAN P2P ({:.2} tok/s) must scale >6x over B=1 ({:.2} tok/s)",
        lan_b16.aggregate_decode_tok_per_sec,
        lan.aggregate_decode_tok_per_sec
    );
    assert_eq!(
        h100.completion_rate_pct, 0.0,
        "single-node GPU has 0% 1-node crash survival"
    );
    assert_eq!(lan.completion_rate_pct, 100.0);

    // All 3 explicit negative regimes must be quantified and REJECTED.
    assert_eq!(negatives.len(), 3);
    assert!(negatives.iter().all(|n| n.verdict == "REJECTED"));

    let neg_rtt = negatives
        .iter()
        .find(|n| n.regime_id == "neg_internet_b1_decode_rtt_wall")
        .expect("neg_internet_b1_decode_rtt_wall");
    assert!(
        neg_rtt.per_user_decode_tok_per_sec <= 6.5,
        "internet B=1 must show RTT wall <= 6.5 tok/s, got {:.2}",
        neg_rtt.per_user_decode_tok_per_sec
    );

    let neg_moe = negatives
        .iter()
        .find(|n| n.regime_id == "neg_naive_moe_alltoall_wan")
        .expect("neg_naive_moe_alltoall_wan");
    assert!(neg_moe.per_user_decode_tok_per_sec < 1.0);

    assert_eq!(privacy.verdict, "REJECTED");
    assert!(privacy.activation_inversion_risk_score > 0.8);
}

#[test]
fn full_hip1350_benchmark_passes_acceptance_gate_and_verifies_corpus_sha256() {
    let root = detect_repo_root();
    let report = run_full_hip1350_benchmark(&root, 1350).expect("full HIP-1350 benchmark");
    assert!(report.acceptance_gate_passed);
    assert_eq!(
        report.corpus.combined_sha256_hex,
        "84b45bc2b79cb4612938566807c5eb8b478824036f382307c869cb112a8cd856"
    );
    assert_eq!(report.live_tcp.failover_exact_matches, 16);
    assert_eq!(report.live_tcp.failover_max_abs_logit_diff, 0.0);
    assert_eq!(report.trust_fault_cases.len(), 6);
    assert!(report.trust_fault_cases.iter().all(|c| c.detected));

    let json = render_report_json(&report);
    assert!(json.contains("\"hip\": \"HIP-1350\""));
    assert!(json.contains("\"hanzo_coordinator_required\": false"));
    assert!(json.contains("\"acceptance_gate_passed\": true"));
}
