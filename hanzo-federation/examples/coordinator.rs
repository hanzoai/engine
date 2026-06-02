//! `cargo run --example coordinator -- --lab lab.yaml --bind 0.0.0.0:8443`
//!
//! Minimal binary entry point. Real deployments use the `hanzod` binary —
//! this exists so the federation crate is testable in isolation.

use anyhow::Result;
use hanzo_federation::{Coordinator, Lab};
use std::env;
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<()> {
    let mut lab_path: Option<String> = None;
    let mut bind: SocketAddr = "0.0.0.0:8443".parse()?;
    let mut log_level = "info".to_string();

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--lab" => {
                lab_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--bind" => {
                bind = args[i + 1].parse()?;
                i += 2;
            }
            "--log-level" => {
                log_level = args[i + 1].clone();
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: coordinator --lab lab.yaml [--bind 0.0.0.0:8443] [--log-level info]"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }

    let lab_path = lab_path.ok_or_else(|| anyhow::anyhow!("--lab is required"))?;

    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .init();

    let lab = Lab::from_yaml(&lab_path)?;
    let coord = Coordinator::new(lab)?;
    coord.serve(bind).await
}
