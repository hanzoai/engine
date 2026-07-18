//! The honest decision-quality gate --- the piece the `holdout_reward` gate was
//! missing.
//!
//! `holdout_reward` asks *"does `W` predict quality?"*. That is not the question
//! routing turns on. The question that decides routing is *"on held-out prompts,
//! does the learned head PICK arms whose realized quality beats the shipping
//! heuristic?"* --- answered here against **full per-arm counterfactuals**: every
//! arm's observed correctness on every prompt (the enso-bench Single-arm runs),
//! so for any policy's pick we read the quality it would actually have gotten.
//!
//! Leakage protocol: split by **prompt id** (all arms of a prompt stay on one
//! side), **stratified by benchmark**. The profile table and `W` are fit on train
//! prompts only; every number below is over held-out prompts.
//!
//! Four references, one metric (mean realized quality over held-out prompts):
//!   * **head**        --- `enso::Enso` fit on train, its serve-path pick per prompt.
//!   * **heuristic**   --- the shipping `classify`+PREFER incumbent (`enso_prefer`).
//!   * **best-single** --- the single train-best arm applied to every prompt.
//!   * **oracle**      --- the per-prompt max (upper bound on any router).

use std::collections::BTreeMap;

use enso::ingest;
use enso::EvalSample;
use enso::HashFeaturizer;
use hanzo_router::classify::Request;
use hanzo_router::registry::{Backend, ModelCard, Registry, Task};
use hanzo_router::{RoutePolicy, Slo, User};

use crate::fit_policy;

/// One augmented eval row: an `EvalSample` plus the (benchmark, prompt-id) it came
/// from. `EvalSample` ignores the extra keys (no `deny_unknown_fields`), so the
/// same JSONL feeds both `enso-fit` and this evaluator.
#[derive(Debug, Clone)]
pub struct LabeledSample {
    pub benchmark: String,
    pub id: String,
    pub sample: EvalSample,
}

/// Parse augmented JSONL. Each line carries `benchmark` + `id` alongside the
/// `EvalSample` fields; we read the tags, then deserialize the same object as an
/// `EvalSample` (which drops the tags).
pub fn parse_labeled(s: &str) -> anyhow::Result<Vec<LabeledSample>> {
    let mut out = Vec::new();
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let benchmark = v.get("benchmark").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let id = v.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let sample: EvalSample = serde_json::from_value(v)?;
        out.push(LabeledSample { benchmark, id, sample });
    }
    Ok(out)
}

/// One prompt with its full per-arm counterfactual and the rows needed to refit.
#[derive(Debug, Clone)]
pub struct Prompt {
    pub benchmark: String,
    pub id: String,
    pub task: Task,
    pub request: Request,
    /// arm -> mean observed correctness on THIS prompt (the counterfactual cell).
    pub arm_quality: BTreeMap<String, f64>,
    /// every eval row of this prompt (all arms, all runs) -- train side refits from these.
    pub rows: Vec<EvalSample>,
}

/// Fold labeled rows into prompts keyed by (benchmark, id). `arm_quality` averages
/// repeat runs of the same (prompt, arm) so each cell is a single value in [0,1].
pub fn to_prompts(rows: &[LabeledSample]) -> Vec<Prompt> {
    struct Acc {
        benchmark: String,
        task: Task,
        request: Request,
        arm_sum: BTreeMap<String, (f64, u32)>,
        rows: Vec<EvalSample>,
    }
    let mut groups: BTreeMap<(String, String), Acc> = BTreeMap::new();
    for ls in rows {
        let key = (ls.benchmark.clone(), ls.id.clone());
        let acc = groups.entry(key).or_insert_with(|| Acc {
            benchmark: ls.benchmark.clone(),
            task: ls.sample.task,
            request: ls.sample.to_request(),
            arm_sum: BTreeMap::new(),
            rows: Vec::new(),
        });
        let e = acc.arm_sum.entry(ls.sample.model.clone()).or_insert((0.0, 0));
        e.0 += ls.sample.quality;
        e.1 += 1;
        acc.rows.push(ls.sample.clone());
    }
    groups
        .into_iter()
        .map(|((_b, id), acc)| Prompt {
            benchmark: acc.benchmark,
            id,
            task: acc.task,
            request: acc.request,
            arm_quality: acc
                .arm_sum
                .into_iter()
                .map(|(a, (s, n))| (a, s / n as f64))
                .collect(),
            rows: acc.rows,
        })
        .collect()
}

/// splitmix64 finalizer: decorrelates the seed so *consecutive* seed arguments
/// (7, 8, 9, ...) yield independent permutations. Without it, `seed ^ CONST`
/// followed by one LCG step leaves the first swaps correlated across adjacent
/// seeds --- which silently makes a run of seeds sample near-identical holdouts.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Seeded Fisher-Yates shuffle. The seed is mixed (see [`mix`]) so a sweep over
/// adjacent seeds gives statistically independent splits.
fn shuffle(idx: &mut [usize], seed: u64) {
    let mut state = mix(seed ^ 0x9E3779B97F4A7C15);
    for i in (1..idx.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        idx.swap(i, j);
    }
}

/// Split prompt indices into (train, test) by fraction `holdout`, **stratified by
/// benchmark** (each benchmark contributes the same holdout fraction) and grouped
/// by prompt (a prompt is wholly in one side). Deterministic in `seed`.
pub fn split_by_prompt(prompts: &[Prompt], holdout: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut by_bench: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, p) in prompts.iter().enumerate() {
        by_bench.entry(p.benchmark.as_str()).or_default().push(i);
    }
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (_b, mut idx) in by_bench {
        shuffle(&mut idx, seed);
        let n_test = ((idx.len() as f64) * holdout).round() as usize;
        test.extend_from_slice(&idx[..n_test]);
        train.extend_from_slice(&idx[n_test..]);
    }
    (train, test)
}

/// A registry of the candidate arms as always-usable cloud `General` cards --- the
/// pool both the head (fallback only) and the ported heuristic range over.
fn arm_registry(arms: &[String]) -> Registry {
    Registry::new(
        arms.iter()
            .map(|id| ModelCard {
                id: id.clone(),
                backend: Backend::Cloud { provider: "gateway".into() },
                tasks: vec![Task::General],
                max_context: 0,
                vision: false,
                cost_per_1k: 0.0,
            })
            .collect(),
    )
}

/// The four-way realized-quality result on one held-out split.
#[derive(Debug, Clone)]
pub struct Scores {
    pub n_holdout: usize,
    /// head under pure-quality SLO (lambda=mu=0): the fairest test of "did it learn quality routing".
    pub head_quality: f64,
    /// head under the default served SLO (lambda=mu=0.3): what actually ships.
    pub head_served: f64,
    /// shipping incumbent: `classify` + PREFER (`hanzo_router` enso_prefer).
    pub heuristic: f64,
    /// single train-best arm applied to every held-out prompt.
    pub best_single: f64,
    pub best_single_arm: String,
    /// per-prompt max over arms --- upper bound on any router.
    pub oracle: f64,
    /// enso's own metric under the SAME prompt-level split (continuity with the old gate).
    pub holdout_reward: f64,
}

fn pure_slo() -> Slo {
    Slo { lambda_cost: 0.0, mu_latency: 0.0, ..Slo::default() }
}

/// Fit on `train` prompts, score every reference on `test` prompts. `gamma` is the
/// ridge strength. The head is the SAME thing the serve path uses: a
/// `hanzo_router::Policy` with the fitted heads mounted (`with_heads`), routed
/// through `RoutePolicy::route` --- so the eval and production score identically.
pub fn score_split(prompts: &[Prompt], train: &[usize], test: &[usize], gamma: f64) -> Scores {
    let feat = HashFeaturizer::default();
    // Train rows -> profile table + fit W (identical join to the serve path).
    let train_rows: Vec<EvalSample> =
        train.iter().flat_map(|&i| prompts[i].rows.clone()).collect();
    let table = ingest(&train_rows);
    let policy_fit = fit_policy(&train_rows, &table, &feat, gamma);

    let arms: Vec<String> = {
        let mut a: Vec<String> = table.profiles.iter().map(|p| p.model.clone()).collect();
        a.sort();
        a.dedup();
        a
    };
    let reg = arm_registry(&arms);
    let heuristic = hanzo_router::prefer();
    // The learned head as it serves: bilinear W + arm profile vectors, mounted on
    // the one router Policy. Pure vs served SLO is just the `slo` passed to route.
    let arm_feats: Vec<hanzo_router::heads::Arm> = table
        .profiles
        .iter()
        .map(|p| hanzo_router::heads::Arm { model: p.model.clone(), feat: p.features().to_vec() })
        .collect();
    let head =
        hanzo_router::Policy::with_heads(hanzo_router::Heads::new(policy_fit.w.clone(), arm_feats));

    // best-single arm = highest mean train quality.
    let mut arm_train: BTreeMap<String, (f64, u32)> = BTreeMap::new();
    for &i in train {
        for (a, q) in &prompts[i].arm_quality {
            let e = arm_train.entry(a.clone()).or_insert((0.0, 0));
            e.0 += q;
            e.1 += 1;
        }
    }
    let best_single_arm = arm_train
        .iter()
        .max_by(|x, y| (x.1 .0 / x.1 .1 as f64).total_cmp(&(y.1 .0 / y.1 .1 as f64)))
        .map(|(a, _)| a.clone())
        .unwrap_or_default();

    let user = User::anonymous();
    let slo_pure = pure_slo();
    let slo_served = Slo::default();

    let realized = |arm: &str, p: &Prompt| -> Option<f64> { p.arm_quality.get(arm).copied() };

    let (mut hq, mut hs, mut heu, mut bs, mut orc, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0usize);
    for &i in test {
        let p = &prompts[i];
        let head_q = head.route(&p.request, &user, &slo_pure, &reg).model;
        let head_s = head.route(&p.request, &user, &slo_served, &reg).model;
        let heur = heuristic.route(&p.request, &user, &slo_served, &reg).model;
        let (Some(vq), Some(vs), Some(vh)) =
            (realized(&head_q, p), realized(&head_s, p), realized(&heur, p))
        else {
            continue;
        };
        hq += vq;
        hs += vs;
        heu += vh;
        bs += realized(&best_single_arm, p).unwrap_or(0.0);
        orc += p.arm_quality.values().copied().fold(f64::MIN, f64::max);
        n += 1;
    }
    let denom = n.max(1) as f64;

    // enso holdout_reward on the fitted W under this split (continuity with the old gate).
    let test_rows: Vec<EvalSample> =
        test.iter().flat_map(|&i| prompts[i].rows.clone()).collect();
    let (hr, _) = crate::holdout_reward(&policy_fit, &table, &feat, &test_rows);

    Scores {
        n_holdout: n,
        head_quality: hq / denom,
        head_served: hs / denom,
        heuristic: heu / denom,
        best_single: bs / denom,
        best_single_arm,
        oracle: orc / denom,
        holdout_reward: hr,
    }
}

/// Mean of a field across seeds, with sample std-dev, for stability reporting.
pub fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = if xs.len() > 1 {
        xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    (m, v.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(bench: &str, id: &str, model: &str, task: &str, q: f64, text: &str) -> String {
        format!(
            r#"{{"benchmark":"{bench}","id":"{id}","task":"{task}","modality":"text","approx_tokens":50,"text":"{text}","model":"{model}","level":"balanced","quality":{q},"latency_ms":1000.0,"cost":0.01}}"#
        )
    }

    #[test]
    fn parse_and_group_counterfactuals() {
        let s = format!(
            "{}\n{}\n{}\n{}\n",
            line("b", "p1", "opus-4.8", "general", 1.0, "solve x"),
            line("b", "p1", "gpt-5.5", "general", 0.0, "solve x"),
            line("b", "p2", "opus-4.8", "general", 0.0, "why is"),
            line("b", "p2", "gpt-5.5", "general", 1.0, "why is"),
        );
        let rows = parse_labeled(&s).unwrap();
        assert_eq!(rows.len(), 4);
        let prompts = to_prompts(&rows);
        assert_eq!(prompts.len(), 2);
        // p1: opus right, gpt wrong; oracle picks the right one either way -> 1.0.
        for p in &prompts {
            let oracle = p.arm_quality.values().copied().fold(f64::MIN, f64::max);
            assert_eq!(oracle, 1.0, "each prompt has one correct arm -> oracle 1.0");
        }
    }

    #[test]
    fn split_is_prompt_grouped_and_stratified() {
        // 8 prompts in bench A, 4 in bench B; 25% holdout keeps the ratio per bench.
        let mut rows = Vec::new();
        for i in 0..8 {
            rows.push(LabeledSample {
                benchmark: "A".into(),
                id: format!("a{i}"),
                sample: mk_sample(1.0),
            });
        }
        for i in 0..4 {
            rows.push(LabeledSample {
                benchmark: "B".into(),
                id: format!("b{i}"),
                sample: mk_sample(1.0),
            });
        }
        let prompts = to_prompts(&rows);
        let (train, test) = split_by_prompt(&prompts, 0.25, 7);
        assert_eq!(test.len(), 3, "round(8*.25)+round(4*.25)=2+1");
        assert_eq!(train.len(), prompts.len() - 3);
        // stratified: exactly one B in test.
        let b_in_test = test.iter().filter(|&&i| prompts[i].benchmark == "B").count();
        assert_eq!(b_in_test, 1);
    }

    fn mk_sample(q: f64) -> EvalSample {
        EvalSample {
            task: Task::General,
            modality: hanzo_router::registry::Modality::Text,
            approx_tokens: 10,
            text: "t".into(),
            model: "opus-4.8".into(),
            level: hanzo_router::registry::Level::Balanced,
            quality: q,
            latency_ms: 1.0,
            cost: 0.0,
            vram_gb: 0.0,
            max_context: 0,
        }
    }
}
