//! `router-fit` -- the router-heads fit tool the nightly pipeline drives.
//!
//! Contract (drop-in for enso's own `fit` when it lands; override with $FIT_BIN):
//!   router-fit --events <in.jsonl> --out <heads.safetensors>
//! Optional gate: --holdout <frac> [--incumbent <base.safetensors>] emits a JSON
//! gate report to stdout. The tool always writes the freshly-fit candidate to
//! --out; the pipeline decides whether to PROMOTE it (publish) from gatePassed.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use enso::{ingest, HashFeaturizer};
use enso::EvalSample;
use hanzo_router_retrain::{
    decide_gate, fit_policy, holdout_reward, load_w, save_w, split_indices, transform_reward,
    RewardTuple,
};

#[derive(Parser, Debug)]
#[command(name = "router-fit", about = "Fit + holdout-gate the router heads.")]
struct Args {
    /// Input JSONL: native enso EvalSample lines, or ai reward tuples with --from-rewards.
    #[arg(long)]
    events: PathBuf,
    /// Output heads safetensors (the fresh candidate; always written).
    #[arg(long)]
    out: PathBuf,
    /// Input is ai /v1/export-routing-rewards tuples; transform to EvalSample first.
    #[arg(long, default_value_t = false)]
    from_rewards: bool,
    /// Holdout fraction for the regression gate (0 = fit-all, no gate).
    #[arg(long, default_value_t = 0.0)]
    holdout: f64,
    /// Incumbent heads to beat on the holdout (omit = bootstrap).
    #[arg(long)]
    incumbent: Option<PathBuf>,
    /// Minimum scored holdout rows to trust the gate; below this it never publishes.
    #[arg(long, default_value_t = 50)]
    min_rows: usize,
    /// Ridge strength.
    #[arg(long, default_value_t = 1.0)]
    gamma: f64,
    /// Split seed.
    #[arg(long, default_value_t = 7)]
    seed: u64,
}

fn read_samples(args: &Args) -> Result<Vec<EvalSample>> {
    let text = std::fs::read_to_string(&args.events)
        .with_context(|| format!("read events {}", args.events.display()))?;
    let mut out = Vec::new();
    let mut bad = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed = if args.from_rewards {
            serde_json::from_str::<RewardTuple>(line).map(|r| transform_reward(&r))
        } else {
            serde_json::from_str::<EvalSample>(line)
        };
        match parsed {
            Ok(s) => out.push(s),
            Err(_) => bad += 1,
        }
    }
    if bad > 0 {
        eprintln!("enso-fit: skipped {bad} unparseable line(s)");
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let feat = HashFeaturizer::default();
    let samples = read_samples(&args)?;
    let events = samples.len();

    let (mut gate_value, mut gate_base, mut holdout_rows) = (0.0f64, 0.0f64, 0usize);
    let decision = if args.holdout > 0.0 && !samples.is_empty() {
        let (tr, te) = split_indices(events, args.holdout, args.seed);
        let train: Vec<EvalSample> = tr.iter().map(|&i| samples[i].clone()).collect();
        let test: Vec<EvalSample> = te.iter().map(|&i| samples[i].clone()).collect();
        let table = ingest(&train);
        let cand = fit_policy(&train, &table, &feat, args.gamma);
        let (cv, rows) = holdout_reward(&cand, &table, &feat, &test);
        gate_value = cv;
        holdout_rows = rows;
        let incumbent = match &args.incumbent {
            Some(p) if p.exists() => {
                let inc = load_w(p).with_context(|| format!("load incumbent {}", p.display()))?;
                let (bv, _) = holdout_reward(&inc, &table, &feat, &test);
                gate_base = bv;
                Some(bv)
            }
            _ => None,
        };
        decide_gate(cv, incumbent, rows, args.min_rows)
    } else {
        hanzo_router_retrain::GateDecision {
            passed: false,
            note: if samples.is_empty() {
                "no events: nothing to gate".to_string()
            } else {
                "no holdout requested".to_string()
            },
        }
    };

    let table_all = ingest(&samples);
    let final_policy = fit_policy(&samples, &table_all, &feat, args.gamma);
    save_w(&args.out, &final_policy)
        .with_context(|| format!("write heads {}", args.out.display()))?;

    // Serve bundle the engine loads via ROUTER_HEADS: the base W plus the arm
    // profile vectors it ranks (a `hanzo_router::Heads` -- the router owns the
    // serve-time scorer). Written next to --out as `<stem>.heads.json` so the one
    // fit command produces both the gate artifact and the servable head.
    let bundle_path = args.out.with_extension("heads.json");
    let arm_feats: Vec<hanzo_router::heads::Arm> = table_all
        .profiles
        .iter()
        .map(|p| hanzo_router::heads::Arm { model: p.model.clone(), feat: p.features().to_vec() })
        .collect();
    hanzo_router::Heads::new(final_policy.w, arm_feats)
        .save(&bundle_path)
        .with_context(|| format!("write serve bundle {}", bundle_path.display()))?;

    let report = serde_json::json!({
        "gateKind": "holdout",
        "gateMetric": "holdout_reward",
        "gateValue": gate_value,
        "gateBase": gate_base,
        "gatePassed": decision.passed,
        "events": events,
        "holdoutRows": holdout_rows,
        "note": decision.note,
        "out": args.out.display().to_string(),
        "serveBundle": bundle_path.display().to_string(),
    });
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
