//! Nightly router-heads retrain, the testable core.
//!
//! One job, four pure pieces the harness (scripts/router-retrain.sh) composes:
//!   1. transform  -- ai /v1/router/rewards tuple -> enso `EvalSample`.
//!   2. fit        -- proof.rs's exact (x,p,quality) join -> `fit_base` -> `Policy`.
//!   3. holdout    -- score a `Policy` on held-out samples (mean quality-prediction).
//!   4. gate       -- publish iff candidate >= incumbent on a trusted holdout.
//! Persist is the enso-base safetensors ("w", f64, [D*K]) the engine loads.

use std::collections::HashMap;
use std::path::Path;

pub mod decision;

use enso::policy::DK;
use enso::profile::PROFILE_DIM;
use enso::{fit_base, Featurizer, HashFeaturizer, Policy, ProfileTable};
use enso::EvalSample;
use hanzo_router::registry::{Level, Modality, Task};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use serde::Deserialize;

/// One line of `GET /v1/router/rewards` (ai controllers/routing_reward.go
/// `rewardTuple`). `features` is the frozen-backbone vector the ledger stored; enso
/// re-featurizes from the request bucket today, so it is carried but not consumed --
/// the binding point to reconcile when the backbone-native head lands.
#[derive(Debug, Clone, Deserialize)]
pub struct RewardTuple {
    #[serde(default)]
    pub features: Option<serde_json::Value>,
    pub model: String,
    #[serde(default)]
    pub task: String,
    pub reward: f64,
    #[serde(default)]
    pub at: String,
}

/// Parse the ai task string (snake_case) into an enso `Task`; unknown -> `General`.
pub fn parse_task(s: &str) -> Task {
    serde_json::from_value::<Task>(serde_json::Value::String(s.to_string())).unwrap_or(Task::General)
}

/// ai reward tuple -> enso `EvalSample`. The reward is the quality label; level and
/// modality are not carried by the export, so they default (Balanced/Text) exactly
/// as `EvalSample`'s own serde defaults would. Cost/latency are unknown here (0).
pub fn transform_reward(r: &RewardTuple) -> EvalSample {
    EvalSample {
        task: parse_task(&r.task),
        modality: Modality::Text,
        approx_tokens: 0,
        text: String::new(),
        model: r.model.clone(),
        level: Level::Balanced,
        quality: r.reward,
        latency_ms: 0.0,
        cost: 0.0,
        vram_gb: 0.0,
        max_context: 0,
    }
}

/// The exact fit join enso's proof.rs uses: featurize each sample's request, look up
/// its pool row `p`, regress `quality` on the bilinear `phi = vec(x p^T)`.
pub fn fit_policy(
    samples: &[EvalSample],
    table: &ProfileTable,
    feat: &HashFeaturizer,
    gamma: f64,
) -> Policy {
    let train: Vec<(Vec<f64>, [f64; PROFILE_DIM], f64)> = samples
        .iter()
        .filter_map(|s| {
            let x = feat.featurize(&s.to_request());
            let p = table.get(&s.model, s.level, s.modality)?.features();
            Some((x, p, s.quality))
        })
        .collect();
    fit_base(
        train.iter().map(|(x, p, y)| (x.as_slice(), p.as_slice(), *y)),
        gamma,
    )
}

/// Score a policy on held-out samples: mean over scorable rows of
/// `clamp(1 - |predicted_quality - observed_quality|, 0, 1)`. Higher is better.
/// Rows whose (model, level, modality) is not in `table` (the train pool) are
/// unscorable and skipped; returns (metric, scored_rows).
pub fn holdout_reward(
    policy: &Policy,
    table: &ProfileTable,
    feat: &HashFeaturizer,
    test: &[EvalSample],
) -> (f64, usize) {
    let mut sum = 0.0;
    let mut n = 0usize;
    for s in test {
        let Some(p) = table.get(&s.model, s.level, s.modality) else {
            continue;
        };
        let x = feat.featurize(&s.to_request());
        let pred = policy.utility(&x, &p.features());
        sum += (1.0 - (pred - s.quality).abs()).clamp(0.0, 1.0);
        n += 1;
    }
    if n == 0 {
        (0.0, 0)
    } else {
        (sum / n as f64, n)
    }
}

/// The publish decision. Fail-closed: too few scored holdout rows never publishes
/// (thin ledger keeps the incumbent). No incumbent = bootstrap publish. Otherwise
/// publish iff the candidate matches or beats the incumbent on the holdout.
#[derive(Debug, Clone, PartialEq)]
pub struct GateDecision {
    pub passed: bool,
    pub note: String,
}

pub fn decide_gate(
    candidate: f64,
    incumbent: Option<f64>,
    holdout_rows: usize,
    min_rows: usize,
) -> GateDecision {
    if holdout_rows < min_rows {
        return GateDecision {
            passed: false,
            note: format!("insufficient holdout rows: {holdout_rows} < {min_rows}"),
        };
    }
    match incumbent {
        None => GateDecision {
            passed: true,
            note: "bootstrap: no incumbent artifact".to_string(),
        },
        Some(base) if candidate >= base => GateDecision {
            passed: true,
            note: format!("candidate {candidate:.4} >= incumbent {base:.4}"),
        },
        Some(base) => GateDecision {
            passed: false,
            note: format!("kept incumbent: candidate {candidate:.4} < {base:.4}"),
        },
    }
}

/// Deterministic 80/20-style split of `n` indices into (train, test) by fraction
/// `holdout`, shuffled by a seeded LCG so a run is reproducible.
pub fn split_indices(n: usize, holdout: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed ^ 0x9E3779B97F4A7C15;
    for i in (1..n).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        idx.swap(i, j);
    }
    let test_n = ((n as f64) * holdout).round() as usize;
    let test = idx[..test_n].to_vec();
    let train = idx[test_n..].to_vec();
    (train, test)
}

const METADATA_FORMAT: &str = "enso-base";

/// Write the base `W` as the enso-base safetensors the engine loads: one f64 tensor
/// "w" of shape [D*K], metadata {"format":"enso-base"}.
pub fn save_w(path: &Path, policy: &Policy) -> anyhow::Result<()> {
    anyhow::ensure!(
        policy.w.len() == DK,
        "policy w len {} != DK {}",
        policy.w.len(),
        DK
    );
    let mut raw = Vec::with_capacity(policy.w.len() * 8);
    for v in &policy.w {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let view = TensorView::new(Dtype::F64, vec![policy.w.len()], &raw)?;
    let mut meta = HashMap::new();
    meta.insert("format".to_string(), METADATA_FORMAT.to_string());
    let bytes = safetensors::serialize([("w".to_string(), view)], Some(meta))?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Load the base `W` written by [`save_w`] (or enso's own persist).
pub fn load_w(path: &Path) -> anyhow::Result<Policy> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let t = st.tensor("w")?;
    anyhow::ensure!(t.dtype() == Dtype::F64, "heads tensor w must be f64");
    let data = t.data();
    anyhow::ensure!(data.len() % 8 == 0, "heads w bytes not f64-aligned");
    let mut w = Vec::with_capacity(data.len() / 8);
    for c in data.chunks_exact(8) {
        w.push(f64::from_le_bytes(c.try_into().unwrap()));
    }
    anyhow::ensure!(w.len() == DK, "heads w len {} != DK {}", w.len(), DK);
    Ok(Policy::from_weights(w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_maps_reward_to_quality_and_task() {
        let r = RewardTuple {
            features: Some(serde_json::json!([0.1, 0.2])),
            model: "claude-sonnet".into(),
            task: "code".into(),
            reward: 0.87,
            at: "2026-07-16T03:00:00Z".into(),
        };
        let s = transform_reward(&r);
        assert_eq!(s.model, "claude-sonnet");
        assert_eq!(s.task, Task::Code);
        assert_eq!(s.quality, 0.87);
        assert_eq!(s.level, Level::Balanced);
        assert_eq!(s.modality, Modality::Text);
    }

    #[test]
    fn transform_unknown_task_falls_back_to_general() {
        let r = RewardTuple {
            features: None,
            model: "m".into(),
            task: "not_a_task".into(),
            reward: 0.5,
            at: String::new(),
        };
        assert_eq!(transform_reward(&r).task, Task::General);
        assert_eq!(parse_task(""), Task::General);
    }

    #[test]
    fn gate_holds_when_holdout_too_thin() {
        let d = decide_gate(0.9, Some(0.1), 10, 50);
        assert!(!d.passed, "thin holdout must never publish");
        assert!(d.note.contains("insufficient"));
    }

    #[test]
    fn gate_bootstraps_without_incumbent() {
        let d = decide_gate(0.42, None, 200, 50);
        assert!(d.passed);
        assert!(d.note.contains("bootstrap"));
    }

    #[test]
    fn gate_publishes_when_candidate_not_worse() {
        assert!(decide_gate(0.80, Some(0.80), 200, 50).passed);
        assert!(decide_gate(0.81, Some(0.80), 200, 50).passed);
    }

    #[test]
    fn gate_keeps_incumbent_on_regression() {
        let d = decide_gate(0.70, Some(0.80), 200, 50);
        assert!(!d.passed);
        assert!(d.note.contains("kept incumbent"));
    }

    #[test]
    fn split_is_deterministic_and_partitions() {
        let (tr, te) = split_indices(100, 0.2, 7);
        assert_eq!(te.len(), 20);
        assert_eq!(tr.len(), 80);
        let (tr2, te2) = split_indices(100, 0.2, 7);
        assert_eq!(te, te2);
        assert_eq!(tr, tr2);
        let mut all: Vec<usize> = tr.iter().chain(te.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..100).collect::<Vec<_>>(), "split must be a partition");
    }

    #[test]
    fn save_load_w_roundtrips() {
        let mut w = vec![0.0f64; DK];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (i as f64) * 0.01 - 0.5;
        }
        let policy = Policy::from_weights(w.clone());
        let dir = std::env::temp_dir();
        let path = dir.join(format!("enso-fit-test-{}.safetensors", std::process::id()));
        save_w(&path, &policy).unwrap();
        let back = load_w(&path).unwrap();
        assert_eq!(back.w, w);
        std::fs::remove_file(&path).ok();
    }

    // A trained policy must not do WORSE than a zero policy on the data it was fit on,
    // proving fit + holdout_reward form a coherent gate on real (sample, quality) rows.
    #[test]
    fn fit_beats_zero_policy_on_holdout() {
        let lineup = enso::synth::lineup();
        let samples = enso::synth::gen_eval_samples(&lineup, 800, 7);
        let (tr, te) = split_indices(samples.len(), 0.2, 7);
        let train: Vec<EvalSample> = tr.iter().map(|&i| samples[i].clone()).collect();
        let test: Vec<EvalSample> = te.iter().map(|&i| samples[i].clone()).collect();
        let table = enso::ingest(&train);
        let feat = HashFeaturizer::default();
        let fitted = fit_policy(&train, &table, &feat, 1.0);
        let zero = Policy::zeros();
        let (cand, rows) = holdout_reward(&fitted, &table, &feat, &test);
        let (base, _) = holdout_reward(&zero, &table, &feat, &test);
        assert!(rows > 0, "must score some holdout rows");
        assert!(
            cand >= base,
            "fitted policy ({cand:.4}) must not underperform zero ({base:.4}) on holdout"
        );
        assert!(decide_gate(cand, Some(base), rows, 1).passed);
    }
}
