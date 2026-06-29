//! SYNTHETIC eval data -- clearly labeled, for proving the mechanism before real
//! benches land. It encodes the lineup's *known* strengths (zen-eco is cheap and
//! weak at hard tasks, Zen-Ultra is the top profile, dub/image are their own
//! pools) and provides the [`oracle`] (noiseless best pick) the proof scores
//! against. Replace this with [`crate::profile::parse_jsonl`] on real bench
//! output and nothing downstream changes.

use hanzo_router::classify::Request;
use hanzo_router::registry::{Level, Modality, Task};
use hanzo_router::Slo;

use crate::featurize::NUM_TASKS;
use crate::profile::{build_request, EvalSample, Profile};
use crate::selector::feasible;

const QUALITY_NOISE: f64 = 0.03;
const PERF_JITTER: f64 = 0.05;

/// A model's true, noiseless characteristics -- the data-generating process.
#[derive(Debug, Clone)]
pub struct TrueProfile {
    pub model: &'static str,
    pub level: Level,
    pub modality: Modality,
    pub quality: [f64; NUM_TASKS],
    pub latency_ms: f64,
    pub cost: f64,
    pub vram_gb: f64,
    pub max_context: usize,
}

impl TrueProfile {
    pub fn to_profile(&self) -> Profile {
        Profile {
            model: self.model.to_string(),
            level: self.level,
            modality: self.modality,
            quality: self.quality,
            latency_ms: self.latency_ms,
            cost: self.cost,
            vram_gb: self.vram_gb,
            max_context: self.max_context,
            samples: 0,
        }
    }
}

/// The cross-modal roster: LLMs, a vision model, image-gen and dub generators --
/// all just rows. Add a model = add a row. `quality` columns are in [`Task::ALL`]
/// order: Code, Reasoning, Math, Creative, Vision, LongContext, CheapChat, General.
pub fn lineup() -> Vec<TrueProfile> {
    use Level::*;
    use Modality::*;
    vec![
        TrueProfile {
            model: "zen-eco",
            level: Fast,
            modality: Text,
            quality: [0.45, 0.45, 0.40, 0.55, 0.0, 0.35, 0.92, 0.72],
            latency_ms: 400.0,
            cost: 0.1,
            vram_gb: 6.0,
            max_context: 32_768,
        },
        TrueProfile {
            model: "zen-coder",
            level: Balanced,
            modality: Text,
            quality: [0.90, 0.80, 0.72, 0.60, 0.0, 0.62, 0.70, 0.78],
            latency_ms: 1_500.0,
            cost: 1.0,
            vram_gb: 24.0,
            max_context: 131_072,
        },
        TrueProfile {
            model: "zen-coder",
            level: Max,
            modality: Text,
            quality: [0.93, 0.85, 0.78, 0.63, 0.0, 0.66, 0.68, 0.80],
            latency_ms: 4_500.0,
            cost: 2.5,
            vram_gb: 24.0,
            max_context: 131_072,
        },
        TrueProfile {
            model: "zen-ultra",
            level: Max,
            modality: Text,
            quality: [0.95, 0.97, 0.95, 0.92, 0.0, 0.90, 0.66, 0.93],
            latency_ms: 6_000.0,
            cost: 8.0,
            vram_gb: 80.0,
            max_context: 262_144,
        },
        TrueProfile {
            model: "claude-sonnet",
            level: Balanced,
            modality: Text,
            quality: [0.90, 0.86, 0.80, 0.85, 0.0, 0.84, 0.70, 0.88],
            latency_ms: 2_200.0,
            cost: 6.0,
            vram_gb: 0.0,
            max_context: 200_000,
        },
        TrueProfile {
            model: "zen-omni",
            level: Balanced,
            modality: Vision,
            quality: [0.0, 0.0, 0.0, 0.0, 0.88, 0.0, 0.0, 0.60],
            latency_ms: 1_800.0,
            cost: 1.2,
            vram_gb: 20.0,
            max_context: 32_768,
        },
        TrueProfile {
            model: "zen-omni",
            level: Max,
            modality: Vision,
            quality: [0.0, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.66],
            latency_ms: 3_200.0,
            cost: 2.4,
            vram_gb: 28.0,
            max_context: 32_768,
        },
        TrueProfile {
            model: "zen-image",
            level: Fast,
            modality: Image,
            quality: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78],
            latency_ms: 2_500.0,
            cost: 1.0,
            vram_gb: 12.0,
            max_context: 0,
        },
        TrueProfile {
            model: "zen-image",
            level: Balanced,
            modality: Image,
            quality: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88],
            latency_ms: 5_000.0,
            cost: 2.5,
            vram_gb: 16.0,
            max_context: 0,
        },
        TrueProfile {
            model: "musetalk",
            level: Fast,
            modality: Video,
            quality: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.74],
            latency_ms: 1_200.0,
            cost: 0.5,
            vram_gb: 8.0,
            max_context: 0,
        },
        TrueProfile {
            model: "echomimic",
            level: Max,
            modality: Video,
            quality: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90],
            latency_ms: 7_000.0,
            cost: 3.0,
            vram_gb: 24.0,
            max_context: 0,
        },
    ]
}

pub fn find<'a>(
    lineup: &'a [TrueProfile],
    model: &str,
    level: Level,
    modality: Modality,
) -> Option<&'a TrueProfile> {
    lineup
        .iter()
        .find(|tp| tp.model == model && tp.level == level && tp.modality == modality)
}

/// A deterministic LCG so every run -- and every CI run -- is identical.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e3779b97f4a7c15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in `[-a, a]`.
    pub fn sym(&mut self, a: f64) -> f64 {
        (self.unit() * 2.0 - 1.0) * a
    }
    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[(self.unit() * xs.len() as f64) as usize % xs.len()]
    }
}

/// What the request is about. The (task, modality) pair and a length range.
struct Category {
    task: Task,
    modality: Modality,
    weight: f64,
    tokens: (usize, usize),
    texts: &'static [&'static str],
}

fn categories() -> Vec<Category> {
    use Modality as M;
    use Task as T;
    vec![
        Category {
            task: T::Code,
            modality: M::Text,
            weight: 1.4,
            tokens: (40, 1200),
            texts: &[
                "fix this rust bug",
                "refactor the function",
                "why does this code panic",
                "compile error in the loop",
            ],
        },
        Category {
            task: T::Reasoning,
            modality: M::Text,
            weight: 1.2,
            tokens: (60, 1500),
            texts: &[
                "analyze the trade-off step by step",
                "explain how this works and why",
                "reason about the failure mode",
            ],
        },
        Category {
            task: T::Math,
            modality: M::Text,
            weight: 1.0,
            tokens: (30, 800),
            texts: &[
                "prove this theorem",
                "solve for x in the equation",
                "calculate the integral",
            ],
        },
        Category {
            task: T::Creative,
            modality: M::Text,
            weight: 0.8,
            tokens: (20, 600),
            texts: &[
                "write a short story",
                "brainstorm a poem",
                "imagine a creative scene",
            ],
        },
        Category {
            task: T::CheapChat,
            modality: M::Text,
            weight: 1.3,
            tokens: (1, 12),
            texts: &["hi", "thanks", "what time is it", "ok"],
        },
        Category {
            task: T::General,
            modality: M::Text,
            weight: 1.4,
            tokens: (20, 1500),
            texts: &[
                "summarize this",
                "what is the capital",
                "give me an overview",
            ],
        },
        Category {
            task: T::LongContext,
            modality: M::Text,
            weight: 0.8,
            tokens: (60_000, 220_000),
            texts: &[
                "summarize this very long document",
                "answer over the whole transcript",
            ],
        },
        Category {
            task: T::Vision,
            modality: M::Vision,
            weight: 1.0,
            tokens: (20, 400),
            texts: &[
                "describe this image",
                "what is in the photo",
                "read the chart",
            ],
        },
        Category {
            task: T::General,
            modality: M::Image,
            weight: 0.7,
            tokens: (10, 60),
            texts: &[
                "generate an image of a city",
                "render a logo",
                "draw a landscape",
            ],
        },
        Category {
            task: T::General,
            modality: M::Video,
            weight: 0.7,
            tokens: (10, 60),
            texts: &[
                "dub this clip",
                "lip-sync the footage",
                "animate this avatar",
            ],
        },
    ]
}

/// A request to be routed: its routing fields and the resolved `(task, modality)`.
#[derive(Debug, Clone)]
pub struct RequestSpec {
    pub task: Task,
    pub modality: Modality,
    pub approx_tokens: usize,
    pub text: String,
}

impl RequestSpec {
    pub fn to_request(&self) -> Request {
        build_request(
            self.text.clone(),
            self.approx_tokens,
            self.task,
            self.modality,
        )
    }
}

pub fn gen_request_specs(n: usize, seed: u64) -> Vec<RequestSpec> {
    let cats = categories();
    let total: f64 = cats.iter().map(|c| c.weight).sum();
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| {
            let mut pick = rng.unit() * total;
            let cat = cats
                .iter()
                .find(|c| {
                    pick -= c.weight;
                    pick <= 0.0
                })
                .unwrap_or(&cats[cats.len() - 1]);
            let (lo, hi) = cat.tokens;
            let approx_tokens = lo + (rng.unit() * (hi - lo) as f64) as usize;
            RequestSpec {
                task: cat.task,
                modality: cat.modality,
                approx_tokens,
                text: cat.pick_text(&mut rng),
            }
        })
        .collect()
}

impl Category {
    fn pick_text(&self, rng: &mut Lcg) -> String {
        rng.pick(self.texts).to_string()
    }
}

/// Expand each request into one eval sample per profile that serves its
/// modality, with observed quality = true + noise (an offline benchmark sweep).
pub fn gen_eval_samples(lineup: &[TrueProfile], n: usize, seed: u64) -> Vec<EvalSample> {
    let specs = gen_request_specs(n, seed);
    let mut rng = Lcg::new(seed ^ 0xdead_beef);
    let mut out = Vec::new();
    for spec in &specs {
        let ti = spec.task.index();
        for tp in lineup.iter().filter(|tp| tp.modality == spec.modality) {
            let quality = (tp.quality[ti] + rng.sym(QUALITY_NOISE)).clamp(0.0, 1.0);
            out.push(EvalSample {
                task: spec.task,
                modality: spec.modality,
                approx_tokens: spec.approx_tokens,
                text: spec.text.clone(),
                model: tp.model.to_string(),
                level: tp.level,
                quality,
                latency_ms: tp.latency_ms * (1.0 + rng.sym(PERF_JITTER)),
                cost: tp.cost * (1.0 + rng.sym(PERF_JITTER)),
                vram_gb: tp.vram_gb,
                max_context: tp.max_context,
            });
        }
    }
    out
}

fn objective(p: &Profile, task: Task, slo: &Slo) -> f64 {
    p.quality[task.index()]
        - slo.lambda_cost as f64 * p.cost_norm()
        - slo.mu_latency as f64 * p.latency_norm()
}

/// The ground truth: the SLO-feasible profile that maximizes the *noiseless*
/// objective for `(task, modality)`. Same objective and feasibility the selector
/// uses, so a perfect policy matches it exactly.
pub fn oracle(
    lineup: &[TrueProfile],
    task: Task,
    modality: Modality,
    slo: &Slo,
) -> Option<(String, Level)> {
    lineup
        .iter()
        .filter(|tp| tp.modality == modality)
        .map(TrueProfile::to_profile)
        .filter(|p| feasible(p, task, slo))
        .max_by(|a, b| {
            objective(a, task, slo)
                .partial_cmp(&objective(b, task, slo))
                .unwrap()
        })
        .map(|p| (p.model, p.level))
}

/// A user's latent taste: how much they personally penalize cost and latency.
#[derive(Debug, Clone, Copy)]
pub struct UserPref {
    pub lambda: f64,
    pub mu: f64,
}

/// The reward a user reports for being served `tp` on a `task` request: their
/// taste-adjusted satisfaction. This is what the online bandit learns per user.
pub fn satisfaction(tp: &TrueProfile, task: Task, pref: &UserPref, noise: f64) -> f64 {
    let p = tp.to_profile();
    p.quality[task.index()] - pref.lambda * p.cost_norm() - pref.mu * p.latency_norm() + noise
}
