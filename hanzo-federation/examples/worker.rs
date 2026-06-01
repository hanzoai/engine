//! `cargo run --example worker -- --lab lab.yaml --as m1max --coordinator http://spark.local:8443`
//!
//! Dev-mode worker. State lives in a Vec<f32>; the "training step" just adds
//! tiny Gaussian-ish noise so we exercise the full push/pull/apply path
//! without any GPU dependency.

use anyhow::Result;
use hanzo_federation::codec::f32_to_bf16;
use hanzo_federation::{Lab, Worker, WorkerConfig};
use std::env;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> Result<()> {
    let mut lab_path: Option<String> = None;
    let mut name: Option<String> = None;
    let mut coord_url: Option<String> = None;
    let mut rounds: u32 = 10;
    let mut log_level = "info".to_string();

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--lab" => {
                lab_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--as" => {
                name = Some(args[i + 1].clone());
                i += 2;
            }
            "--coordinator" => {
                coord_url = Some(args[i + 1].clone());
                i += 2;
            }
            "--rounds" => {
                rounds = args[i + 1].parse()?;
                i += 2;
            }
            "--log-level" => {
                log_level = args[i + 1].clone();
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: worker --lab lab.yaml --as <name> --coordinator URL [--rounds N]"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }

    let lab_path = lab_path.ok_or_else(|| anyhow::anyhow!("--lab is required"))?;
    let name = name.ok_or_else(|| anyhow::anyhow!("--as is required"))?;
    let coord_url = coord_url.ok_or_else(|| anyhow::anyhow!("--coordinator is required"))?;

    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .init();

    let lab = Lab::from_yaml(&lab_path)?;
    let node = lab.find(&name)?;
    let secret = std::env::var(format!("ZEN_LAB_SECRET_{}", name.to_uppercase()))
        .ok()
        .or_else(|| node.auth_token.clone());

    let config = WorkerConfig {
        coordinator_url: coord_url,
        worker_name: name.clone(),
        secret,
        steps_per_round: lab.sync_interval_steps,
        total_rounds: rounds,
    };

    // Toy state.
    let state: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(vec![0.0; 512]));
    let s_step = state.clone();
    let s_params = state.clone();
    let s_apply = state.clone();

    let worker = Worker::new(config);
    let mut counter: u64 = 0;
    worker
        .run(
            move |_b: &()| {
                let mut s = s_step.lock().unwrap();
                counter = counter.wrapping_add(1);
                for (i, v) in s.iter_mut().enumerate() {
                    *v += 0.0001 * ((counter as f32) + (i as f32) * 0.01).sin();
                }
                0.5
            },
            move || {
                let s = s_params.lock().unwrap();
                let bf16 = f32_to_bf16(&s);
                vec![("toy".to_string(), bf16, vec![s.len() as u64])]
            },
            move |delta| {
                use hanzo_federation::codec::bf16_to_f32;
                let mut s = s_apply.lock().unwrap();
                for (_, _, bytes) in delta {
                    *s = bf16_to_f32(&bytes);
                }
            },
            || Box::new(std::iter::repeat(())),
        )
        .await
}
