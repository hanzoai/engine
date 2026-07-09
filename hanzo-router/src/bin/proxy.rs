//! `hanzo-router` proxy: replica load-balancing + prefix-affinity front for N
//! hanzo-engine instances serving the same model. Config via a YAML pool file or
//! inline `--model M --replica URL ...` flags.

use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use hanzo_router::proxy::{serve, ServeConfig, DEFAULT_PROBE_INTERVAL};
use hanzo_router::{Balancer, BalancerConfig, Replica};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "hanzo-router",
    about = "Replica load-balancing proxy for hanzo-engine"
)]
struct Args {
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(short = 'p', long, default_value_t = 1234)]
    port: u16,
    /// YAML pool file: `max_inflight` + `models: {id: [{url}, ...]}`.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Model id for inline `--replica` flags.
    #[arg(long)]
    model: Option<String>,
    /// Replica base URL (repeatable), e.g. `http://127.0.0.1:1234`.
    #[arg(long = "replica")]
    replicas: Vec<String>,
    #[arg(long)]
    max_inflight: Option<usize>,
    /// Background health re-probe cadence, seconds.
    #[arg(long)]
    probe_interval_secs: Option<u64>,
}

fn build_balancer(args: &Args) -> anyhow::Result<Balancer> {
    let mut cfg: BalancerConfig = match &args.config {
        Some(path) => serde_yaml::from_str(&std::fs::read_to_string(path)?)?,
        None => BalancerConfig::default(),
    };
    if let Some(mi) = args.max_inflight {
        cfg.max_inflight = mi;
    }
    if let Some(model) = &args.model {
        let entry = cfg.models.entry(model.clone()).or_default();
        for url in &args.replicas {
            entry.push(Replica::new(url.clone()));
        }
    }
    if cfg.models.is_empty() {
        anyhow::bail!("no replicas: pass --config FILE or --model M --replica URL ...");
    }
    Ok(Balancer::from_config(cfg))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    let probe_interval = args
        .probe_interval_secs
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_PROBE_INTERVAL);
    let balancer = build_balancer(&args)?;
    serve(ServeConfig {
        host: args.host.clone(),
        port: args.port,
        balancer,
        probe_interval,
    })
    .await
}
