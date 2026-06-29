//! Piece 2/6: the model pool as the brain sees it. A [`Profile`] is one
//! (model, level, modality) row carrying its eval-measured feature vector
//! `p = [quality-by-task | latency | cost | vram | ctx]`. [`ingest`] folds raw
//! bench tuples ([`EvalSample`]) into the [`ProfileTable`]. "Add a model = add a
//! row" -- the registry stays the same shape across LLMs, dub generators and
//! image models; only the rows differ.
//!
//! This is where real eval data plugs in: point [`parse_jsonl`] at a bench's
//! JSONL output (one [`EvalSample`] per line) and the rest is unchanged.

use std::collections::BTreeMap;

use hanzo_router::classify::Request;
use hanzo_router::registry::{Level, Modality, Task};
use serde::{Deserialize, Serialize};

use crate::featurize::NUM_TASKS;

/// The one place a routing [`Request`] is built from its routing fields, so eval
/// samples and generated requests featurize identically.
pub fn build_request(
    text: String,
    approx_tokens: usize,
    task: Task,
    modality: Modality,
) -> Request {
    Request {
        text,
        approx_tokens,
        has_media: modality == Modality::Vision,
        task_hint: Some(task),
        modality_hint: Some(modality),
    }
}

/// `p` layout: [quality-by-task (8) | latency_norm | cost_norm | vram_norm | ctx_norm].
pub const PROFILE_DIM: usize = NUM_TASKS + 4;

const LAT_NORM_MS: f64 = 20_000.0;
const COST_NORM: f64 = 30.0;
const VRAM_NORM_GB: f64 = 200.0;
const CTX_NORM: f64 = 1_048_576.0;

/// One raw eval observation -- the shape a bench emits: a request bucket, the
/// (model, level, modality) it ran, and the measured quality/latency/cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSample {
    pub task: Task,
    pub modality: Modality,
    #[serde(default)]
    pub approx_tokens: usize,
    #[serde(default)]
    pub text: String,
    pub model: String,
    pub level: Level,
    pub quality: f64,
    pub latency_ms: f64,
    pub cost: f64,
    #[serde(default)]
    pub vram_gb: f64,
    #[serde(default)]
    pub max_context: usize,
}

impl EvalSample {
    pub fn to_request(&self) -> Request {
        build_request(
            self.text.clone(),
            self.approx_tokens,
            self.task,
            self.modality,
        )
    }
}

/// One aggregated pool row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub model: String,
    pub level: Level,
    pub modality: Modality,
    pub quality: [f64; NUM_TASKS],
    pub latency_ms: f64,
    pub cost: f64,
    pub vram_gb: f64,
    pub max_context: usize,
    pub samples: u32,
}

impl Profile {
    pub fn key(&self) -> (String, Level, Modality) {
        (self.model.clone(), self.level, self.modality)
    }

    pub fn latency_norm(&self) -> f64 {
        self.latency_ms / LAT_NORM_MS
    }

    pub fn cost_norm(&self) -> f64 {
        self.cost / COST_NORM
    }

    /// The eval feature vector `p` consumed by the bilinear policy.
    pub fn features(&self) -> [f64; PROFILE_DIM] {
        let mut p = [0.0; PROFILE_DIM];
        p[..NUM_TASKS].copy_from_slice(&self.quality);
        p[NUM_TASKS] = self.latency_norm();
        p[NUM_TASKS + 1] = self.cost_norm();
        p[NUM_TASKS + 2] = self.vram_gb / VRAM_NORM_GB;
        p[NUM_TASKS + 3] = self.max_context as f64 / CTX_NORM;
        p
    }
}

/// The pool. Lookups are by (model, level, modality); candidate enumeration is
/// by modality (the pool the request must be served from).
#[derive(Debug, Clone, Default)]
pub struct ProfileTable {
    pub profiles: Vec<Profile>,
}

impl ProfileTable {
    pub fn get(&self, model: &str, level: Level, modality: Modality) -> Option<&Profile> {
        self.profiles
            .iter()
            .find(|p| p.model == model && p.level == level && p.modality == modality)
    }

    pub fn for_modality(&self, modality: Modality) -> impl Iterator<Item = &Profile> {
        self.profiles.iter().filter(move |p| p.modality == modality)
    }
}

/// Fold raw eval tuples into aggregated profile rows: per (model, level,
/// modality) the per-task quality is the mean over that task's samples, and
/// latency/cost/vram are means over all the group's samples.
pub fn ingest(samples: &[EvalSample]) -> ProfileTable {
    struct Acc {
        modality: Modality,
        q_sum: [f64; NUM_TASKS],
        q_cnt: [u32; NUM_TASKS],
        lat: f64,
        cost: f64,
        vram: f64,
        max_ctx: usize,
        n: u32,
    }
    let mut groups: BTreeMap<(String, Level), Acc> = BTreeMap::new();
    for s in samples {
        let a = groups
            .entry((s.model.clone(), s.level))
            .or_insert_with(|| Acc {
                modality: s.modality,
                q_sum: [0.0; NUM_TASKS],
                q_cnt: [0; NUM_TASKS],
                lat: 0.0,
                cost: 0.0,
                vram: 0.0,
                max_ctx: 0,
                n: 0,
            });
        let ti = s.task.index();
        a.q_sum[ti] += s.quality;
        a.q_cnt[ti] += 1;
        a.lat += s.latency_ms;
        a.cost += s.cost;
        a.vram += s.vram_gb;
        a.max_ctx = a.max_ctx.max(s.max_context);
        a.n += 1;
    }
    let mut profiles: Vec<Profile> = groups
        .into_iter()
        .map(|((model, level), a)| {
            let mut quality = [0.0; NUM_TASKS];
            for (q, (sum, cnt)) in quality.iter_mut().zip(a.q_sum.iter().zip(a.q_cnt.iter())) {
                if *cnt > 0 {
                    *q = sum / *cnt as f64;
                }
            }
            let n = a.n as f64;
            Profile {
                model,
                level,
                modality: a.modality,
                quality,
                latency_ms: a.lat / n,
                cost: a.cost / n,
                vram_gb: a.vram / n,
                max_context: a.max_ctx,
                samples: a.n,
            }
        })
        .collect();
    profiles.sort_by_key(|p| p.key());
    ProfileTable { profiles }
}

/// Parse a bench's JSONL output (one [`EvalSample`] per non-empty line) -- the
/// real-data entry point.
pub fn parse_jsonl(s: &str) -> Result<Vec<EvalSample>, serde_json::Error> {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(task: Task, quality: f64, latency_ms: f64) -> EvalSample {
        EvalSample {
            task,
            modality: Modality::Text,
            approx_tokens: 50,
            text: String::new(),
            model: "m".into(),
            level: Level::Balanced,
            quality,
            latency_ms,
            cost: 1.0,
            vram_gb: 8.0,
            max_context: 4096,
        }
    }

    #[test]
    fn ingest_aggregates_per_task_and_means_perf() {
        let samples = vec![
            sample(Task::Code, 0.8, 1000.0),
            sample(Task::Code, 0.6, 2000.0),
            sample(Task::Math, 0.4, 1500.0),
        ];
        let table = ingest(&samples);
        let p = table.get("m", Level::Balanced, Modality::Text).unwrap();
        assert!((p.quality[Task::Code.index()] - 0.7).abs() < 1e-9); // mean of code samples
        assert!((p.quality[Task::Math.index()] - 0.4).abs() < 1e-9);
        assert_eq!(p.quality[Task::Reasoning.index()], 0.0); // unobserved task
        assert!((p.latency_ms - 1500.0).abs() < 1e-9); // mean over the group
        assert_eq!(p.samples, 3);
    }

    #[test]
    fn jsonl_roundtrips() {
        let line = serde_json::to_string(&sample(Task::Code, 0.9, 800.0)).unwrap();
        let parsed = parse_jsonl(&format!("{line}\n\n")).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].task, Task::Code);
    }
}
