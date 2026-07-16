//! `enso fit` — the Rust-native warm-start job.
//!
//! Ingests the enso-bench per-item results into the arms `ProfileTable`, ridge-fits
//! the base policy `W` over the bilinear feature `phi = vec(x p^T)` with the crate's
//! own [`enso::fit_base`], and writes a scope artifact via [`enso::persist::save_base`]
//! in exactly the shape `enso_from_path`/the registry load. No Python in the shipped
//! path — the featurizer, ingest, and fit are the same code serving runs, so there is
//! zero train/serve drift.
//!
//! Usage:
//!   enso-fit --results <dir> --out-dir <dir> [--scope global|org=<slug>] [--cost-weight <f>]
//!
//! `--scope global` writes `heads-base.safetensors`; `--scope org=<slug>` writes
//! `heads-<slug>.safetensors` in the same dir (identical format — the per-org retrain
//! is the same code path with a different key).
//!
//! Reward: a per-item outcome — `correct ? 1 : 0` minus a small cost penalty
//! (`cost_weight * cost_norm`). GPQA Diamond is graduate STEM Q&A, so each record is
//! ingested into the KNOWLEDGE task cluster (general / reasoning / math): whatever the
//! serving classifier buckets a knowledge question into, the arm's measured accuracy
//! applies there. The routed meta-systems (enso / enso-ultra) are NOT arms — only the
//! single models the router chooses among.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use enso::{fit_base, persist, Learner, Policy};
use hanzo_router::registry::{Level, Modality, Task};
use serde::Deserialize;

/// The knowledge task cluster GPQA-style records populate. Serving a knowledge
/// question classifies into one of these; the arm's accuracy applies to all.
const KNOWLEDGE_TASKS: [Task; 3] = [Task::General, Task::Reasoning, Task::Math];

/// Meta-systems that ROUTE rather than serve — never arms in the pool.
const NON_ARMS: [&str; 2] = ["enso", "enso-ultra"];

/// Map a benchmark id to the task column(s) its per-item results populate. A bench is
/// evidence about a task: GPQA-Diamond is graduate STEM knowledge (the knowledge
/// cluster); LiveCodeBench / SWE-Bench / Terminal-Bench are code; CharXiv is vision;
/// HLE is hard reasoning. An unrecognized bench is skipped (no guess about its task).
fn bench_tasks(bench: &str) -> &'static [Task] {
    match bench {
        "gpqa_diamond" | "gpqa" | "mmlu" => &KNOWLEDGE_TASKS,
        "livecodebench" | "livecodebench_pro" | "swebench_pro" | "swebench" | "terminal_bench" => {
            &[Task::Code]
        }
        "charxiv" | "charxiv_reasoning" => &[Task::Vision],
        "humanitys_last_exam" | "hle" => &[Task::Reasoning],
        _ => &[],
    }
}

const GAMMA: f64 = 0.01; // ridge strength
const DEFAULT_COST_WEIGHT: f64 = 0.5;

#[derive(Deserialize)]
struct BenchFile {
    #[serde(default)]
    bench: String,
    #[serde(default)]
    system: String,
    #[serde(default)]
    system_kind: String,
    #[serde(default)]
    n: usize,
    #[serde(default)]
    usd_est: f64,
    #[serde(default)]
    max_tokens: usize,
    #[serde(default)]
    records: Vec<Record>,
}

#[derive(Deserialize)]
struct Record {
    #[serde(default)]
    correct: bool,
    #[serde(default)]
    usages: Vec<Usage>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    latency_s: f64,
}

struct Args {
    results: PathBuf,
    out_dir: PathBuf,
    scope: String,
    cost_weight: f64,
}

fn parse_args() -> Result<Args> {
    let mut results = None;
    let mut out_dir = None;
    let mut scope = "global".to_string();
    let mut cost_weight = DEFAULT_COST_WEIGHT;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--results" => results = it.next().map(PathBuf::from),
            "--out-dir" => out_dir = it.next().map(PathBuf::from),
            "--scope" => scope = it.next().unwrap_or_else(|| "global".into()),
            "--cost-weight" => {
                cost_weight = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(DEFAULT_COST_WEIGHT)
            }
            other => bail!("unknown arg {other}"),
        }
    }
    Ok(Args {
        results: results.context("--results <dir> is required")?,
        out_dir: out_dir.context("--out-dir <dir> is required")?,
        scope,
        cost_weight,
    })
}

/// `heads-base.safetensors` for global, `heads-<slug>.safetensors` for an org scope.
fn artifact_name(scope: &str) -> Result<String> {
    if scope == "global" || scope == "base" {
        return Ok("heads-base.safetensors".to_string());
    }
    if let Some(slug) = scope.strip_prefix("org=") {
        let slug = slug.trim();
        if slug.is_empty() {
            bail!("--scope org= requires a slug");
        }
        return Ok(format!("heads-{slug}.safetensors"));
    }
    bail!("--scope must be 'global' or 'org=<slug>'")
}

/// Per (arm, bench) accuracy, for the report.
struct Stat {
    correct: usize,
    n: usize,
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // Each bench record becomes one EvalSample per task the bench maps to. The task,
    // not the file, decides the column — so GPQA (knowledge) and LiveCodeBench (code)
    // for the same arm land in different columns and never dilute each other.
    let mut samples: Vec<enso::EvalSample> = Vec::new();
    let mut stats: BTreeMap<(String, String), Stat> = BTreeMap::new();
    let mut arms: std::collections::BTreeSet<String> = Default::default();
    let mut files = 0usize;
    for entry in std::fs::read_dir(&args.results)
        .with_context(|| format!("read dir {}", args.results.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let text = std::fs::read_to_string(&path)?;
        let bf: BenchFile = match serde_json::from_str(&text) {
            Ok(b) => b,
            Err(_) => continue, // summaries / pending stubs — skip
        };
        if bf.records.is_empty() || bf.system.is_empty() {
            continue;
        }
        if NON_ARMS.contains(&bf.system.as_str())
            || bf.system_kind == "enso"
            || bf.system_kind == "enso-ultra"
        {
            continue; // routed meta-systems are not arms
        }
        let tasks = bench_tasks(&bf.bench);
        if tasks.is_empty() {
            eprintln!("skip {}: unmapped bench {:?}", path.display(), bf.bench);
            continue;
        }
        files += 1;
        arms.insert(bf.system.clone());
        let usd_per_call = if bf.n > 0 { bf.usd_est / bf.n as f64 } else { 0.0 };
        let max_ctx = bf.max_tokens.max(200_000);
        let st = stats
            .entry((bf.system.clone(), bf.bench.clone()))
            .or_insert(Stat { correct: 0, n: 0 });
        for r in &bf.records {
            let lat_ms = r.usages.iter().map(|u| u.latency_s).sum::<f64>() * 1000.0;
            st.correct += r.correct as usize;
            st.n += 1;
            for &task in tasks {
                samples.push(enso::EvalSample {
                    task,
                    modality: if task == Task::Vision {
                        Modality::Vision
                    } else {
                        Modality::Text
                    },
                    approx_tokens: 512,
                    text: String::new(),
                    model: bf.system.clone(),
                    level: Level::Balanced,
                    quality: if r.correct { 1.0 } else { 0.0 },
                    latency_ms: lat_ms,
                    cost: usd_per_call,
                    vram_gb: 0.0,
                    max_context: max_ctx,
                });
            }
        }
    }
    if samples.is_empty() {
        bail!(
            "no mapped arm bench records found in {}",
            args.results.display()
        );
    }

    // Fold into the arms table (per-task mean accuracy + mean latency/cost).
    let table = enso::ingest(&samples);

    // Ridge fit W over the samples. x is the PURE task one-hot (not the full hashing
    // featurizer): the warm-start base must read per-task quality only, with tasks
    // DECOUPLED — the featurizer's task-independent bias term otherwise learns a global
    // arm preference that, once cross-task data (e.g. code) is added, overrides the
    // per-task quality ranking (routing knowledge to a fast-but-weaker arm). The task
    // one-hot leaves W's bias/length/hash rows at zero, so at serving (full featurizer)
    // they contribute nothing and utility = the task row · quality. Online LinUCB then
    // adapts the full feature set per user on top of this clean base.
    // x is the pure task one-hot; p carries the arm's measured quality in ONLY that
    // task's column (zeros elsewhere). This isolates each task, so the ridge fit
    // recovers the clean quality-reading diagonal W[t,t]≈1 — free of the collinearity
    // (equal knowledge columns) and cross-task contamination (an arm's code score
    // bleeding into its reasoning row) that a full-vector fit suffers. At serving the
    // selector scores the arm's FULL quality vector, and W (zero off-diagonal, zero on
    // the bias/length/hash rows) reads exactly the requested task's column, so
    // utility(task, arm) = the arm's measured quality on that task.
    let mut tuples: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::with_capacity(samples.len());
    for s in &samples {
        let ti = s.task.index();
        let mut x = vec![0.0f64; enso::featurize::FEAT_DIM];
        x[ti] = 1.0;
        let profile = table
            .by_model(&s.model)
            .expect("every sample's arm is in the ingested table");
        let mut p = vec![0.0f64; enso::PROFILE_DIM];
        p[ti] = profile.quality[ti];
        let y = s.quality - args.cost_weight * profile.cost_norm();
        tuples.push((x, p, y));
    }
    let policy: Policy = fit_base(
        tuples.iter().map(|(x, p, y)| (x.as_slice(), p.as_slice(), *y)),
        GAMMA,
    );
    let learner = Learner::new(policy, 1.0, 0.5);

    // Write the scope artifact.
    std::fs::create_dir_all(&args.out_dir).ok();
    let out = args.out_dir.join(artifact_name(&args.scope)?);
    persist::save_base(&out, &learner, &table)?;

    // Report.
    eprintln!(
        "enso fit: scope={} files={} arms={} samples={} → {}",
        args.scope,
        files,
        arms.len(),
        samples.len(),
        out.display()
    );
    for ((model, bench), st) in &stats {
        let acc = st.correct as f64 / st.n.max(1) as f64;
        eprintln!("  {model:<18} {bench:<16} n={n:<4} acc={acc:.4}", n = st.n);
    }

    // Self-verify: load the artifact exactly as serving does and print the picks, so
    // the deploy gate can confirm the policy is non-heuristic (a real arm) before ship.
    let (learner, loaded) = persist::load(&out, 1.0, 0.5)?;
    let brain = enso::Enso::new(loaded, learner);
    for (label, text) in [
        ("knowledge", "Explain why the reaction rate depends on temperature; reason step by step."),
        ("code", "fix this ```rust``` compile bug in the function"),
    ] {
        let req = hanzo_router::classify::Request {
            text: text.into(),
            approx_tokens: 400,
            has_media: false,
            task_hint: None,
            modality_hint: None,
        };
        // Quality-first SLO (soft cost/latency weights zeroed) — enso's cloud default:
        // maximize measured quality subject to hard ceilings.
        let slo = hanzo_router::Slo {
            lambda_cost: 0.0,
            mu_latency: 0.0,
            ..Default::default()
        };
        let (route, explain) = brain.route_explained(
            &req,
            &hanzo_router::User::new("verify"),
            &slo,
            &hanzo_router::Registry::new(vec![]),
        );
        eprintln!(
            "  verify[{label}] task={:?} -> arm={:?} conf={:.3} fallback={}",
            explain.task, route.model, route.confidence, explain.used_fallback
        );
        let mut scores = explain.scores.clone();
        scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        for (m, _lvl, obj) in scores.iter().take(6) {
            eprintln!("      {m:<18} objective={obj:.5}");
        }
    }
    Ok(())
}
