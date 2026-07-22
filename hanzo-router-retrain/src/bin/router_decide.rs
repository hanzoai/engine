//! `router-decide` --- the honest decision-quality gate as a runnable report.
//!
//! Reads the augmented EvalSample JSONL (benchmark + id + EvalSample per line),
//! splits by prompt id stratified by benchmark, fits `enso::Enso` on train, and
//! prints mean realized quality over held-out prompts for four references
//! (head / heuristic / best-single / oracle) plus enso's `holdout_reward`, swept
//! over gammas and averaged across seeds. Nothing is fit to the holdout.

use anyhow::{Context, Result};
use clap::Parser;

use hanzo_router_retrain::decision::{
    mean_std, parse_labeled, score_split, split_by_prompt, to_prompts, Scores,
};

#[derive(Parser, Debug)]
#[command(
    name = "router-decide",
    about = "Decision-quality gate for router heads."
)]
struct Args {
    /// Augmented EvalSample JSONL (benchmark + id + EvalSample fields per line).
    #[arg(long)]
    events: std::path::PathBuf,
    /// Holdout fraction (per benchmark).
    #[arg(long, default_value_t = 0.25)]
    holdout: f64,
    /// Comma-separated ridge strengths to sweep.
    #[arg(long, default_value = "0.1,1.0,10.0")]
    gammas: String,
    /// Comma-separated seeds to average the split over.
    #[arg(long, default_value = "7,8,9,10,11")]
    seeds: String,
}

fn parse_list(s: &str) -> Vec<f64> {
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}
fn parse_seeds(s: &str) -> Vec<u64> {
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}

fn avg_scores(all: &[Scores]) -> (Vec<f64>, Vec<f64>) {
    let pick = |f: fn(&Scores) -> f64| all.iter().map(f).collect::<Vec<_>>();
    let cols: Vec<Vec<f64>> = vec![
        pick(|s| s.head_quality),
        pick(|s| s.head_served),
        pick(|s| s.heuristic),
        pick(|s| s.best_single),
        pick(|s| s.oracle),
        pick(|s| s.holdout_reward),
    ];
    let mut means = Vec::new();
    let mut stds = Vec::new();
    for c in &cols {
        let (m, sd) = mean_std(c);
        means.push(m);
        stds.push(sd);
    }
    (means, stds)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let text = std::fs::read_to_string(&args.events)
        .with_context(|| format!("read {}", args.events.display()))?;
    let rows = parse_labeled(&text)?;
    let prompts = to_prompts(&rows);
    let n_bench = {
        let mut b: Vec<&str> = prompts.iter().map(|p| p.benchmark.as_str()).collect();
        b.sort();
        b.dedup();
        b.into_iter()
            .map(|name| {
                let n = prompts.iter().filter(|p| p.benchmark == name).count();
                format!("{name}={n}")
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let gammas = parse_list(&args.gammas);
    let seeds = parse_seeds(&args.seeds);
    println!(
        "events={} prompts={} [{}]  holdout={} seeds={:?}",
        rows.len(),
        prompts.len(),
        n_bench,
        args.holdout,
        seeds
    );
    // Absolute means (mean +- std across independent seeds).
    println!("--- absolute realized quality (mean +- std over seeds) ---");
    println!(
        "gamma   head@q          head@slo        heuristic       best-single     oracle          holdout_reward"
    );
    let cols = [
        "head@q",
        "head@slo",
        "heuristic",
        "best-single",
        "oracle",
        "holdout_reward",
    ];
    let mut per_gamma: Vec<(f64, Vec<Scores>)> = Vec::new();
    for &gamma in &gammas {
        let mut all = Vec::new();
        let mut arm = String::new();
        for &seed in &seeds {
            let (tr, te) = split_by_prompt(&prompts, args.holdout, seed);
            let s = score_split(&prompts, &tr, &te, gamma);
            arm = s.best_single_arm.clone();
            all.push(s);
        }
        let (means, stds) = avg_scores(&all);
        print!("{gamma:<7.2}");
        for i in 0..cols.len() {
            print!(" {:.4}+-{:.4} ", means[i], stds[i]);
        }
        println!("  (best-single={arm})");
        per_gamma.push((gamma, all));
    }

    // Paired differences on the SAME holdout each seed -- the correct policy
    // comparison (cancels the per-split variance the absolute column carries).
    // A positive diff means the head wins; `wins` is the count of seeds it does.
    println!("\n--- paired differences on identical holdouts (mean +- std; wins/seeds) ---");
    println!("gamma   head@q - heuristic     head@q - best-single   head@slo - heuristic");
    for (gamma, all) in &per_gamma {
        let d_heur: Vec<f64> = all.iter().map(|s| s.head_quality - s.heuristic).collect();
        let d_best: Vec<f64> = all.iter().map(|s| s.head_quality - s.best_single).collect();
        let d_served: Vec<f64> = all.iter().map(|s| s.head_served - s.heuristic).collect();
        let fmt = |d: &[f64]| {
            let (m, sd) = mean_std(d);
            let wins = d.iter().filter(|&&x| x > 1e-9).count();
            format!("{m:+.4}+-{sd:.4} ({wins}/{})", d.len())
        };
        println!(
            "{gamma:<7.2} {:<22} {:<22} {}",
            fmt(&d_heur),
            fmt(&d_best),
            fmt(&d_served)
        );
    }
    println!(
        "\nn.b. head@q = pure-quality SLO (lambda=mu=0); head@slo = default served SLO (0.3/0.3). \
         oracle - best-single is the per-prompt routing headroom any policy could win."
    );
    Ok(())
}
