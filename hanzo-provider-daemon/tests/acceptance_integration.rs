//! End-to-end integration tests for `hanzo-provider-daemon` covering:
//! - Issue #64 acceptance: 3 independently administered providers register without
//!   operator approval, execute paid jobs, gracefully drain while completing in-flight
//!   work, restart from sovereign disk state, and return to service automatically.
//! - Multi-threaded concurrency stress over real loopback TCP verifying zero VRAM
//!   leaks, strict backpressure, and unbroken Ed25519 + SHA-256 ledger integrity.
//! - Durable `fsync` ledger persistence and atomic sovereign export.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use hanzo_provider_daemon::advertise::{
    CapabilityAnnouncement, GpuDevice, HardwareBackend, ModelCapability, PeerDirectory,
};
use hanzo_provider_daemon::daemon::{
    now_unix_ms, DeterministicEngineBackend, JobExecutionResult, ProviderConfig, ProviderDaemon,
    QuoteRequest, SignedJobLease, SignedQuote,
};
use hanzo_provider_daemon::identity::{KeyEnvelope, ProviderKeyPair, TenantKeyPair};
use hanzo_provider_daemon::isolation::SandboxExecutionPolicy;
use hanzo_provider_daemon::json::{parse_json, JsonValue};
use hanzo_provider_daemon::ledger::verify_jsonl_chain;
use hanzo_provider_daemon::rpc::{http_request, RpcServer};

fn make_config(backend: HardwareBackend, vram: u64, conc: u32, price: u64) -> ProviderConfig {
    ProviderConfig {
        endpoint: "http://127.0.0.1:0".into(),
        devices: vec![GpuDevice {
            backend,
            name: format!("{}-gpu", backend.as_str()),
            vram_total_mib: vram,
            vram_usable_mib: vram,
            compute_units: 64,
            device_id: format!("{}:0", backend.as_str()),
        }],
        models: vec![ModelCapability {
            model_id: "zen-8b-q4".into(),
            quantization: "Q4_K_M".into(),
            vram_required_mib: 4_096,
            max_context_tokens: 8_192,
            price_micro_usd_per_1m: price,
        }],
        max_concurrency: conc,
        heartbeat_ttl_secs: 120,
        quote_ttl_secs: 60,
        sandbox_policy: SandboxExecutionPolicy::default(),
    }
}

fn execute_over_tcp(
    addr: std::net::SocketAddr,
    tenant: &TenantKeyPair,
    job_id: &str,
    nonce: u64,
    prompt: &str,
) -> Result<JobExecutionResult, u16> {
    let qreq = QuoteRequest {
        tenant_id: tenant.tenant_id().into(),
        model_id: "zen-8b-q4".into(),
        prompt_tokens_estimate: 16,
        max_tokens: 16,
    };
    let qr = http_request(
        addr,
        "POST",
        "/v1/provider/quote",
        Some(qreq.to_json().to_compact_string().as_bytes()),
    )
    .expect("quote tcp");
    if qr.status != 200 {
        return Err(qr.status);
    }
    let quote = SignedQuote::from_json(&parse_json(qr.body_text().unwrap()).unwrap()).unwrap();
    let lease =
        SignedJobLease::sign_for_quote(tenant, &quote, job_id, nonce, prompt, now_unix_ms());
    let mut body = BTreeMap::new();
    body.insert("quote".into(), quote.to_json());
    body.insert("lease".into(), lease.to_json());
    let jr = http_request(
        addr,
        "POST",
        "/v1/provider/jobs",
        Some(JsonValue::Object(body).to_compact_string().as_bytes()),
    )
    .expect("job tcp");
    if jr.status != 200 {
        return Err(jr.status);
    }
    let res = JobExecutionResult::from_json(&parse_json(jr.body_text().unwrap()).unwrap()).unwrap();
    res.receipt.verify().expect("valid ed25519 receipt");
    Ok(res)
}

#[test]
fn three_independent_providers_register_serve_drain_restart_and_resume() {
    let tmp = std::env::temp_dir().join(format!(
        "hanzo_provider_accept_{}_{}",
        std::process::id(),
        now_unix_ms()
    ));
    fs::create_dir_all(&tmp).unwrap();

    let mut dir = PeerDirectory::new();
    let specs = [
        ("indie-berlin-cuda", HardwareBackend::Cuda, 24_576, 130_000),
        ("indie-tokyo-rocm", HardwareBackend::Rocm, 16_384, 150_000),
        (
            "indie-austin-vulkan",
            HardwareBackend::Vulkan,
            12_288,
            170_000,
        ),
    ];

    let tenant = TenantKeyPair::from_label("paying-tenant-01");
    let mut envelopes = Vec::new();
    let mut ledger_paths = Vec::new();
    let mut configs = Vec::new();
    let mut expected_heads = Vec::new();

    for (idx, (label, hw, vram, price)) in specs.iter().enumerate() {
        let kp = ProviderKeyPair::from_label(label);
        let lp = tmp.join(format!("p{idx}.jsonl"));
        let cfg = make_config(*hw, *vram, 2, *price);
        let daemon = Arc::new(
            ProviderDaemon::open_with_ledger(
                kp.clone(),
                cfg.clone(),
                &lp,
                true, // real fsync durability
                Arc::new(DeterministicEngineBackend {
                    per_token_delay_us: 0,
                    prefill_delay_us: 0,
                }),
            )
            .unwrap(),
        );
        let srv = RpcServer::start("127.0.0.1:0", Arc::clone(&daemon)).unwrap();

        // Register permissionlessly via signed manifest
        let mf = http_request(srv.addr(), "GET", "/v1/provider/manifest", None).unwrap();
        assert_eq!(mf.status, 200);
        let ann = CapabilityAnnouncement::from_json(&parse_json(mf.body_text().unwrap()).unwrap())
            .unwrap();
        dir.register(ann, now_unix_ms()).unwrap();

        // Execute 2 paid jobs on this independent provider
        for j in 0..2u64 {
            let res = execute_over_tcp(
                srv.addr(),
                &tenant,
                &format!("p{idx}-job-{j}"),
                (idx as u64 + 1) * 100 + j,
                "Verify decentralized execution & receipt signing",
            )
            .unwrap();
            assert_eq!(res.receipt.provider_id, daemon.provider_id());
        }

        // Export atomically & verify
        let export_path = tmp.join(format!("p{idx}_export.jsonl"));
        let written = daemon.ledger().export_to_file_atomic(&export_path).unwrap();
        assert!(written > 0);
        let exported_text = fs::read_to_string(&export_path).unwrap();
        let (entries, summary) =
            verify_jsonl_chain(&exported_text, Some(daemon.provider_id())).unwrap();
        assert_eq!(entries.len(), 2);

        envelopes.push(kp.export_envelope());
        ledger_paths.push(lp);
        configs.push(cfg);
        expected_heads.push(summary.head_hash_hex);
        srv.stop();
    }

    assert_eq!(dir.provider_count(), 3);
    assert_eq!(dir.hanzo_control_plane_calls(), 0);

    // Cold-restart all 3 providers from disk & confirm automatic return to service
    for idx in 0..3 {
        let restored_kp = ProviderKeyPair::import_envelope(
            &KeyEnvelope::from_json_str(&envelopes[idx].to_json().to_compact_string()).unwrap(),
        )
        .unwrap();
        let daemon = Arc::new(
            ProviderDaemon::open_with_ledger(
                restored_kp,
                configs[idx].clone(),
                &ledger_paths[idx],
                true,
                Arc::new(DeterministicEngineBackend {
                    per_token_delay_us: 0,
                    prefill_delay_us: 0,
                }),
            )
            .unwrap(),
        );
        assert_eq!(daemon.ledger_summary().entry_count, 2);
        assert_eq!(daemon.ledger_summary().head_hash_hex, expected_heads[idx]);

        let srv = RpcServer::start("127.0.0.1:0", Arc::clone(&daemon)).unwrap();
        let mf = http_request(srv.addr(), "GET", "/v1/provider/manifest", None).unwrap();
        let ann = CapabilityAnnouncement::from_json(&parse_json(mf.body_text().unwrap()).unwrap())
            .unwrap();
        dir.register(ann, now_unix_ms()).unwrap();

        // Pre-restart nonce must still be rejected as 409 Conflict
        let replay = execute_over_tcp(
            srv.addr(),
            &tenant,
            "replay-old",
            (idx as u64 + 1) * 100,
            "replayed",
        );
        assert_eq!(replay, Err(409));

        // Fresh job succeeds immediately after restart
        let res = execute_over_tcp(
            srv.addr(),
            &tenant,
            &format!("p{idx}-post-restart"),
            9_000 + idx as u64,
            "Post-restart paid job",
        )
        .unwrap();
        assert_eq!(res.receipt.receipt_seq, 3);
        srv.stop();
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn graceful_drain_completes_inflight_paid_job_while_rejecting_new_leases() {
    let kp = ProviderKeyPair::from_label("drain-inflight-provider");
    let cfg = make_config(HardwareBackend::Cuda, 16_384, 2, 140_000);
    let daemon = Arc::new(ProviderDaemon::new(
        kp,
        cfg,
        Arc::new(DeterministicEngineBackend {
            per_token_delay_us: 2_000,
            prefill_delay_us: 40_000, // ~72 ms job
        }),
    ));
    let srv = RpcServer::start("127.0.0.1:0", Arc::clone(&daemon)).unwrap();
    let addr = srv.addr();
    let tenant = TenantKeyPair::from_label("drain-tenant");

    // Start slow paid job on background thread
    let tenant_bg = tenant.clone();
    let inflight = thread::spawn(move || {
        execute_over_tcp(
            addr,
            &tenant_bg,
            "inflight-paid-1",
            44_001,
            "long running prompt",
        )
    });

    // Wait briefly until the job is active inside CapacityPool
    for _ in 0..50 {
        if daemon.capacity_snapshot().active_jobs > 0 {
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    assert!(daemon.capacity_snapshot().active_jobs > 0);

    // Initiate graceful drain over HTTP
    let dr = http_request(addr, "POST", "/v1/provider/drain", None).unwrap();
    assert_eq!(dr.status, 200);

    // New quote during drain MUST be rejected with 503
    let new_req = execute_over_tcp(addr, &tenant, "rejected-during-drain", 44_002, "new job");
    assert_eq!(new_req, Err(503));

    // In-flight paid job MUST complete cleanly with a signed receipt!
    assert!(daemon.wait_until_drained(Duration::from_secs(5)));
    let res = inflight
        .join()
        .unwrap()
        .expect("in-flight paid job must complete");
    res.receipt.verify().unwrap();
    assert_eq!(daemon.capacity_snapshot().used_vram_mib, 0);
    assert_eq!(daemon.capacity_snapshot().active_jobs, 0);

    srv.stop();
}

#[test]
fn concurrent_tcp_stress_zero_vram_leaks_and_intact_ledger() {
    let kp = ProviderKeyPair::from_label("stress-provider");
    let cfg = make_config(HardwareBackend::Cuda, 8_192, 2, 140_000);
    let daemon = Arc::new(ProviderDaemon::new(
        kp,
        cfg,
        Arc::new(DeterministicEngineBackend {
            per_token_delay_us: 100,
            prefill_delay_us: 1_500,
        }),
    ));
    let srv = RpcServer::start("127.0.0.1:0", Arc::clone(&daemon)).unwrap();
    let addr = srv.addr();

    let mut handles = Vec::new();
    for t in 0..6u64 {
        let tenant = TenantKeyPair::from_label(format!("stress-tenant-{t}"));
        handles.push(thread::spawn(move || {
            let mut ok = 0u64;
            let mut rejected_429 = 0u64;
            for j in 0..8u64 {
                match execute_over_tcp(
                    addr,
                    &tenant,
                    &format!("s-{t}-{j}"),
                    t * 1_000 + j + 1,
                    "concurrent load test prompt",
                ) {
                    Ok(_) => ok += 1,
                    Err(429) => rejected_429 += 1,
                    Err(other) => panic!("unexpected status {other}"),
                }
            }
            (ok, rejected_429)
        }));
    }

    let mut total_ok = 0u64;
    let mut total_429 = 0u64;
    for h in handles {
        let (ok, r429) = h.join().unwrap();
        total_ok += ok;
        total_429 += r429;
    }

    assert!(total_ok > 0);
    assert_eq!(total_ok + total_429, 48);

    // Zero leaked VRAM, concurrency slots, or tenant scratch buffers
    let snap = daemon.capacity_snapshot();
    assert_eq!(snap.active_jobs, 0);
    assert_eq!(snap.used_vram_mib, 0);
    assert_eq!(snap.active_scratch_slots, 0);
    assert!(snap.peak_vram_mib <= 8_192);
    assert!(snap.peak_concurrency <= 2);

    // Every single completed job is recorded in the unbroken hash-chained ledger
    let exported = daemon.export_ledger_jsonl();
    let (entries, summary) = verify_jsonl_chain(&exported, Some(daemon.provider_id())).unwrap();
    assert_eq!(entries.len() as u64, total_ok);
    assert_eq!(summary.entry_count, total_ok);

    srv.stop();
}
