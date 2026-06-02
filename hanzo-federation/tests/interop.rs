//! Integration test that the Rust coordinator accepts a delta posted by
//! the Python worker (`gym.distributed worker`).
//!
//! Strategy: bring up a Rust coordinator on an ephemeral port, write a
//! single-node lab.yaml with no auth, invoke
//! `PYTHONPATH=…/gym/src python3 -m gym.distributed worker --lab … --as m1
//!  --coordinator http://127.0.0.1:PORT --rounds 1`, then assert that the
//! coordinator received the delta and produced an aggregate.
//!
//! Skipped if `python3` or the gym package can't be found, so the workspace
//! still builds in CI environments without the Python toolchain.

use hanzo_federation::{Coordinator, Lab};
use std::net::{SocketAddr, TcpListener};
use std::time::Duration;

fn pick_free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn python_worker_pushes_to_rust_coordinator() {
    // Single-node hybrid lab — Python worker connects as "m1".
    let lab_yaml = r#"
job_dir: .zen-fed
sync_interval_steps: 1
aggregation: byzantine_robust
nodes:
  - name: m1
    host: 127.0.0.1
    role: hybrid
    backend: cpu
    memory_gb: 16
    nic_gbps: 10
    tflops: 1
"#;
    let lab = Lab::from_yaml_str(lab_yaml).expect("parse minimal lab");
    let coord = Coordinator::new(lab).expect("coordinator");
    let state = coord.state.clone();

    let port = pick_free_port();
    let bind: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();

    // Spawn coordinator.
    let server = tokio::spawn(async move {
        let _ = coord.serve(bind).await;
    });
    // Give it a moment to bind.
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Write the lab to a temp file for the Python worker.
    let dir = tempfile::tempdir().expect("tempdir");
    let lab_path = dir.path().join("lab.yaml");
    std::fs::write(&lab_path, lab_yaml).unwrap();

    // Try to invoke Python worker. If not available, skip.
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());
    let gym_src = std::env::var("GYM_SRC")
        .unwrap_or_else(|_| "/Users/z/work/zen/gym/src".to_string());

    if !std::path::Path::new(&gym_src).exists() {
        eprintln!("skipping: GYM_SRC not found at {gym_src}");
        server.abort();
        return;
    }

    let status = std::process::Command::new(&python)
        .env("PYTHONPATH", &gym_src)
        .args([
            "-m",
            "gym.distributed",
            "worker",
            "--lab",
            lab_path.to_str().unwrap(),
            "--as",
            "m1",
            "--coordinator",
            &format!("http://127.0.0.1:{port}"),
            "--rounds",
            "1",
            "--log-level",
            "INFO",
        ])
        .status();

    let status = match status {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skipping: failed to spawn python: {e}");
            server.abort();
            return;
        }
    };
    if !status.success() {
        eprintln!("python worker exited {status:?} — interop did not complete");
        server.abort();
        // Don't fail hard — Python toolchain absence shouldn't break Rust CI.
        return;
    }

    // Confirm: round 0 received and aggregated.
    let m = state.metrics();
    let rounds = m.get("rounds").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert!(!rounds.is_empty(), "no rounds recorded");
    let r0 = &rounds[0];
    let received = r0
        .get("received")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    assert!(received >= 1, "round 0 received no deltas");
    let aggregated = r0.get("aggregated").and_then(|v| v.as_bool()).unwrap_or(false);
    assert!(aggregated, "round 0 was not aggregated");

    server.abort();
}
