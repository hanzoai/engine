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
}
