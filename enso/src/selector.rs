//! Piece 5/6: the decision. Over the profiles in the request's modality pool
//! that satisfy the SLO ceilings, pick `argmax[ utility - lambda*cost -
//! mu*latency ]`. Confidence is the squashed margin to the runner-up. The safety
//! gate is applied by the caller ([`crate::Enso`]) before this runs -- the
//! selector is the constrained optimum, the guard is orthogonal.

use hanzo_router::registry::{Level, Modality, Task};
use hanzo_router::Slo;

use crate::policy::Policy;
use crate::profile::ProfileTable;

const MARGIN_SCALE: f64 = 8.0;
const SOLE_CONFIDENCE: f64 = 0.85;

/// The org's cost/quality knee. Each operating point is `(lambda, quality, cost)` --
/// the mean quality and mean cost the org's own traffic gets when its requests carry
/// that `lambda_cost`. Returns the `lambda` at the Pareto knee: normalize both axes
/// and take the point furthest above the cheapest->best chord (Kneedle), i.e. where
/// quality stops being worth its marginal cost. This is how "highest quality at lowest
/// cost" is *found* per org, not hand-dialed; one org lands thrifty (large lambda,
/// flash tiers), another premium (small lambda), from the same fit. Degenerate sweeps:
/// flat cost -> best quality (smallest lambda); flat quality -> cheapest (largest lambda).
pub fn knee_lambda(points: &[(f64, f64, f64)]) -> f64 {
    if points.len() < 2 {
        return points.first().map(|p| p.0).unwrap_or(0.0);
    }
    let (mut c_lo, mut c_hi, mut q_lo, mut q_hi) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for &(_, q, c) in points {
        c_lo = c_lo.min(c);
        c_hi = c_hi.max(c);
        q_lo = q_lo.min(q);
        q_hi = q_hi.max(q);
    }
    let (cd, qd) = (c_hi - c_lo, q_hi - q_lo);
    if qd <= 0.0 {
        return points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    }
    if cd <= 0.0 {
        return points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    }
    let mut best = (f64::NEG_INFINITY, points[0].0);
    for &(lambda, q, c) in points {
        let gap = (q - q_lo) / qd - (c - c_lo) / cd;
        if gap > best.0 {
            best = (gap, lambda);
        }
    }
    best.1
}

/// Coarse model family: the vendor/architecture prefix at which two arms stop being
/// independent. `claude-opus-4.8 -> claude`, `gpt-5.5 -> gpt`, `zen5 -> zen`,
/// `deepseek-v4-pro -> deepseek`. Diversity is scored over families, not model ids,
/// because two SKUs from one provider share failure modes; a second opinion worth
/// paying for comes from a different family.
pub fn model_family(model: &str) -> &str {
    let m = model.trim();
    let end = m
        .find(|c: char| {
            c == '-' || c == '/' || c == ':' || c == '.' || c == '_' || c.is_ascii_digit()
        })
        .unwrap_or(m.len());
    if end == 0 {
        m
    } else {
        &m[..end]
    }
}

#[derive(Debug, Clone)]
pub struct Choice {
    pub model: String,
    pub level: Level,
    pub modality: Modality,
    pub utility: f64,
    pub objective: f64,
    pub confidence: f64,
}

#[derive(Debug, Default)]
pub struct Selector;

/// The decision inputs the selector ranges over: the request features `x`, the
/// effective weights `w`, the pool, the request's modality/task, and the SLO.
pub struct SelectCtx<'a> {
    pub x: &'a [f64],
    pub w: &'a [f64],
    pub table: &'a ProfileTable,
    pub want: Modality,
    pub task: Task,
    pub slo: &'a Slo,
}

/// SLO feasibility for one profile under a task. Shared with the oracle so the
/// learned selector and the ground truth gate identically.
pub fn feasible(p: &crate::profile::Profile, task: Task, slo: &Slo) -> bool {
    (slo.max_latency_ms <= 0.0 || p.latency_ms <= slo.max_latency_ms as f64)
        && (slo.max_cost <= 0.0 || p.cost <= slo.max_cost as f64)
        && (slo.min_quality <= 0.0 || p.quality[task.index()] >= slo.min_quality as f64)
}

impl Selector {
    /// `ctx.w` is the effective (base or per-user) weight matrix. When `scores`
    /// is `Some`, every feasible candidate's objective is recorded into it for
    /// the observability channel; the hot path passes `None` and allocates
    /// nothing beyond the per-candidate profile vector.
    pub fn select(
        &self,
        ctx: &SelectCtx,
        mut scores: Option<&mut Vec<(String, Level, f64)>>,
    ) -> Option<Choice> {
        let mut best: Option<Choice> = None;
        let mut runner_up = f64::NEG_INFINITY;
        for p in ctx.table.for_modality(ctx.want) {
            if !feasible(p, ctx.task, ctx.slo) {
                continue;
            }
            let pv = p.features();
            let utility = Policy::utility_with(ctx.w, ctx.x, &pv);
            let objective = utility
                - ctx.slo.lambda_cost as f64 * p.cost_norm()
                - ctx.slo.mu_latency as f64 * p.latency_norm();
            if let Some(s) = scores.as_deref_mut() {
                s.push((p.model.clone(), p.level, objective));
            }
            match &best {
                Some(b) if objective <= b.objective => {
                    if objective > runner_up {
                        runner_up = objective;
                    }
                }
                _ => {
                    if let Some(b) = best.take() {
                        runner_up = b.objective;
                    }
                    best = Some(Choice {
                        model: p.model.clone(),
                        level: p.level,
                        modality: ctx.want,
                        utility,
                        objective,
                        confidence: 0.0,
                    });
                }
            }
        }
        best.map(|mut c| {
            c.confidence = if runner_up.is_finite() {
                1.0 / (1.0 + (-(c.objective - runner_up) * MARGIN_SCALE).exp())
            } else {
                SOLE_CONFIDENCE
            };
            c
        })
    }

    /// A family-diverse top-`k`. The head is byte-identical to [`Selector::select`]
    /// (constrained argmax of the objective); the tail is a greedy MMR that demotes a
    /// candidate by `beta` when its [`model_family`] is already represented. This is
    /// what the fan-out SKU folds and what failover walks: a second opinion from a
    /// *different* family resolves an escalating request in fewer total steps than a
    /// near-duplicate from the same family. `beta = 0` is plain top-k by objective;
    /// `k = 1` returns exactly `select`'s pick, so this is a strict extension.
    pub fn select_diverse(&self, ctx: &SelectCtx, k: usize, beta: f64) -> Vec<Choice> {
        let mut cands: Vec<Choice> = Vec::new();
        for p in ctx.table.for_modality(ctx.want) {
            if !feasible(p, ctx.task, ctx.slo) {
                continue;
            }
            let pv = p.features();
            let utility = Policy::utility_with(ctx.w, ctx.x, &pv);
            let objective = utility
                - ctx.slo.lambda_cost as f64 * p.cost_norm()
                - ctx.slo.mu_latency as f64 * p.latency_norm();
            cands.push(Choice {
                model: p.model.clone(),
                level: p.level,
                modality: ctx.want,
                utility,
                objective,
                confidence: 0.0,
            });
        }
        let k = k.min(cands.len());
        let mut picked: Vec<Choice> = Vec::with_capacity(k);
        let mut families: Vec<String> = Vec::new();
        while picked.len() < k {
            let mut best_i = 0usize;
            let mut best_score = f64::NEG_INFINITY;
            for (i, c) in cands.iter().enumerate() {
                let redundant = families.iter().any(|f| f == model_family(&c.model));
                let score = c.objective - if redundant { beta } else { 0.0 };
                if score > best_score {
                    best_score = score;
                    best_i = i;
                }
            }
            let mut chosen = cands.remove(best_i);
            let runner = cands
                .iter()
                .map(|c| c.objective)
                .fold(f64::NEG_INFINITY, f64::max);
            chosen.confidence = if runner.is_finite() {
                1.0 / (1.0 + (-(chosen.objective - runner) * MARGIN_SCALE).exp())
            } else {
                SOLE_CONFIDENCE
            };
            families.push(model_family(&chosen.model).to_string());
            picked.push(chosen);
        }
        picked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{D, K};
    use crate::profile::{Profile, ProfileTable};

    fn prof(model: &str, level: Level, quality_general: f64, latency_ms: f64) -> Profile {
        let mut quality = [0.0; crate::featurize::NUM_TASKS];
        quality[Task::General.index()] = quality_general;
        Profile {
            model: model.into(),
            level,
            modality: Modality::Text,
            quality,
            latency_ms,
            cost: 1.0,
            vram_gb: 8.0,
            max_context: 4096,
            samples: 1,
        }
    }

    // Identity-ish weights: utility picks out the General quality column.
    fn quality_reading_w() -> Vec<f64> {
        let mut w = vec![0.0; D * K];
        let row = Task::General.index(); // the active x one-hot row
        w[row * K + Task::General.index()] = 1.0;
        w
    }

    #[test]
    fn picks_max_objective_and_gates_on_latency() {
        let table = ProfileTable {
            profiles: vec![
                prof("fast", Level::Fast, 0.6, 500.0),
                prof("good", Level::Max, 0.9, 4000.0),
            ],
        };
        let mut x = vec![0.0; D];
        x[Task::General.index()] = 1.0;
        let w = quality_reading_w();
        let sel = Selector;
        let ctx = |slo, want| SelectCtx {
            x: &x,
            w: &w,
            table: &table,
            want,
            task: Task::General,
            slo,
        };

        // No ceilings, tiny penalties: the higher-quality profile wins.
        let slo = Slo {
            lambda_cost: 0.01,
            mu_latency: 0.01,
            ..Default::default()
        };
        assert_eq!(
            sel.select(&ctx(&slo, Modality::Text), None).unwrap().model,
            "good"
        );

        // Latency cap removes the slow one.
        let tight = Slo {
            max_latency_ms: 1000.0,
            ..slo
        };
        assert_eq!(
            sel.select(&ctx(&tight, Modality::Text), None)
                .unwrap()
                .model,
            "fast"
        );

        // Nothing in the pool -> no choice.
        assert!(sel.select(&ctx(&slo, Modality::Video), None).is_none());
    }

    fn profc(model: &str, quality_general: f64, cost: f64) -> Profile {
        let mut quality = [0.0; crate::featurize::NUM_TASKS];
        quality[Task::General.index()] = quality_general;
        Profile {
            model: model.into(),
            level: Level::Max,
            modality: Modality::Text,
            quality,
            latency_ms: 100.0,
            cost,
            vram_gb: 8.0,
            max_context: 4096,
            samples: 1,
        }
    }

    fn ctx_general<'a>(
        table: &'a ProfileTable,
        x: &'a [f64],
        w: &'a [f64],
        slo: &'a Slo,
    ) -> SelectCtx<'a> {
        SelectCtx {
            x,
            w,
            table,
            want: Modality::Text,
            task: Task::General,
            slo,
        }
    }

    // The cost dial (lambda_cost) shifts WHICH tier is targeted: at lambda 0 the
    // higher-quality pro wins; raise it and the cheaper flash wins -- same family, so
    // this is the cost axis, orthogonal to diversity, and never blocked by it.
    #[test]
    fn cost_dial_shifts_tier_within_family() {
        let table = ProfileTable {
            profiles: vec![
                profc("deepseek-v4-pro", 0.90, 15.0),
                profc("deepseek-4-flash", 0.80, 3.0),
            ],
        };
        let mut x = vec![0.0; D];
        x[Task::General.index()] = 1.0;
        let w = quality_reading_w();
        let sel = Selector;
        let premium = Slo {
            lambda_cost: 0.0,
            mu_latency: 0.0,
            ..Default::default()
        };
        let cheap = Slo {
            lambda_cost: 1.0,
            mu_latency: 0.0,
            ..Default::default()
        };
        assert_eq!(
            sel.select(&ctx_general(&table, &x, &w, &premium), None)
                .unwrap()
                .model,
            "deepseek-v4-pro"
        );
        assert_eq!(
            sel.select(&ctx_general(&table, &x, &w, &cheap), None)
                .unwrap()
                .model,
            "deepseek-4-flash"
        );
    }

    // Diversity acts on FAMILY among substitutes in one modality: the head equals
    // select's argmax (unchanged), and the tail prefers a different family over a
    // near-duplicate. k=1 is exactly select; beta=0 is plain top-k.
    #[test]
    fn diverse_tail_prefers_a_second_family() {
        let table = ProfileTable {
            profiles: vec![
                profc("claude-opus", 0.90, 10.0),
                profc("claude-sonnet", 0.82, 5.0),
                profc("gpt-5.5", 0.80, 5.0),
            ],
        };
        let mut x = vec![0.0; D];
        x[Task::General.index()] = 1.0;
        let w = quality_reading_w();
        let sel = Selector;
        let slo = Slo {
            lambda_cost: 0.001,
            mu_latency: 0.0,
            ..Default::default()
        };
        let ctx = ctx_general(&table, &x, &w, &slo);

        let head = sel.select(&ctx, None).unwrap();
        let d1 = sel.select_diverse(&ctx, 1, 0.1);
        assert_eq!(d1.len(), 1);
        assert_eq!(d1[0].model, head.model);
        assert_eq!(head.model, "claude-opus");

        let flat: Vec<String> = sel
            .select_diverse(&ctx, 2, 0.0)
            .iter()
            .map(|c| model_family(&c.model).to_string())
            .collect();
        assert_eq!(flat, vec!["claude".to_string(), "claude".to_string()]);

        let div = sel.select_diverse(&ctx, 2, 0.1);
        assert_eq!(div[0].model, "claude-opus");
        assert_eq!(div[1].model, "gpt-5.5");
    }

    #[test]
    fn model_family_splits_vendor_from_tier() {
        assert_eq!(model_family("deepseek-4-flash"), "deepseek");
        assert_eq!(model_family("deepseek-v4-pro"), "deepseek");
        assert_eq!(model_family("claude-opus-4.8"), "claude");
        assert_eq!(model_family("gpt-5.5"), "gpt");
        assert_eq!(model_family("zen5"), "zen");
        assert_eq!(model_family("gemini-3.1-pro"), "gemini");
        // Generic: any vendor, never a whitelist.
        assert_eq!(model_family("acme-2"), "acme");
        assert_eq!(model_family("standalone"), "standalone");
    }

    // The knee finder works on numbers only -- no model identities. A sweep where
    // quality saturates while cost keeps climbing has its knee at the elbow lambda.
    #[test]
    fn knee_lambda_finds_the_elbow() {
        let points = vec![
            (2.0, 0.60, 1.0),
            (1.0, 0.78, 2.0),
            (0.5, 0.83, 4.0),
            (0.0, 0.85, 10.0),
        ];
        assert_eq!(knee_lambda(&points), 1.0);
        // Flat quality -> take the cheapest (largest lambda).
        assert_eq!(knee_lambda(&[(0.0, 0.8, 10.0), (3.0, 0.8, 1.0)]), 3.0);
        // Flat cost -> take the best quality (smallest lambda).
        assert_eq!(knee_lambda(&[(0.0, 0.9, 5.0), (3.0, 0.6, 5.0)]), 0.0);
        assert_eq!(knee_lambda(&[(1.5, 0.7, 2.0)]), 1.5);
    }
}
