//! Workload-aware draft source and verification-shape selector (`E[useful_tok / ms]`).
//!
//! Dynamically selects per step/turn between:
//! 1. `PlainDecode` (zero draft overhead — default when drafting does not win)
//! 2. `PromptLookup` (zero neural compute suffix lookup — wins on repetitive code edits / diffs)
//! 3. `Eagle3Linear(k)` (linear chain feature draft of depth `k`)
//! 4. `Eagle3Tree(depth, fanout, max_nodes)` (tree-structured feature draft — wins on larger
//!    variants and moderate-acceptance mixed prose/code)
//!
//! Enforces complexity-preserving tie-breaking (`PlainDecode < PromptLookup < Eagle3Linear < Eagle3Tree`),
//! hysteresis to prevent thrashing, compute-bound batch-size guards, and explicit stand-down
//! with cooldown on high-entropy / low-acceptance turns.

/// Selectable draft source and verification shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DraftStrategy {
    PlainDecode,
    PromptLookup {
        max_draft: usize,
    },
    Eagle3Linear {
        depth: usize,
    },
    Eagle3Tree {
        depth: usize,
        fanout: usize,
        max_nodes: usize,
    },
}

impl DraftStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            Self::PlainDecode => "plain_decode",
            Self::PromptLookup { .. } => "prompt_lookup",
            Self::Eagle3Linear { .. } => "eagle3_linear",
            Self::Eagle3Tree { .. } => "eagle3_tree",
        }
    }

    /// Complexity rank (`0` = simplest). A higher-rank strategy must beat a lower-rank
    /// strategy by at least `hysteresis_margin` to be preferred ("keep the simpler path as default").
    pub fn complexity_rank(&self) -> u8 {
        match self {
            Self::PlainDecode => 0,
            Self::PromptLookup { .. } => 1,
            Self::Eagle3Linear { .. } => 2,
            Self::Eagle3Tree { .. } => 3,
        }
    }

    pub fn index(&self) -> usize {
        self.complexity_rank() as usize
    }
}

/// Configuration for the online workload-aware selector.
#[derive(Clone, Debug)]
pub struct SelectorConfig {
    pub lookup_max_draft: usize,
    pub linear_depth: usize,
    pub tree_depth: usize,
    pub tree_fanout: usize,
    pub tree_max_nodes: usize,
    /// EWMA smoothing factor `alpha` in `(0, 1]`.
    pub ewma_alpha: f64,
    /// Minimum relative speedup margin over a simpler strategy before escalating complexity.
    pub hysteresis_margin: f64,
    /// Minimum suffix match length to prioritize zero-compute `PromptLookup`.
    pub min_lookup_match_len: usize,
    /// Target entropy threshold (nats) above which speculative acceptance is monitored tightly.
    pub high_entropy_threshold: f64,
    /// Number of consecutive low-gain speculative steps before standing down to `PlainDecode`.
    pub stand_down_patience: usize,
    /// Number of steps to remain in `PlainDecode` stand-down before re-probing.
    pub cooldown_steps: usize,
    /// Batch size at or above which extra verification tokens saturate compute -> force `PlainDecode`.
    pub compute_bound_batch_threshold: usize,
}

impl Default for SelectorConfig {
    fn default() -> Self {
        Self {
            lookup_max_draft: 6,
            linear_depth: 4,
            tree_depth: 4,
            tree_fanout: 2,
            tree_max_nodes: 10,
            ewma_alpha: 0.35,
            hysteresis_margin: 0.04,
            min_lookup_match_len: 3,
            high_entropy_threshold: 2.35,
            stand_down_patience: 2,
            cooldown_steps: 10,
            compute_bound_batch_threshold: 16,
        }
    }
}

/// Runtime context signals available before selecting the draft strategy for the next step.
#[derive(Clone, Copy, Debug)]
pub struct StepSignals {
    /// Active decode batch size (`1` for single-user coding agent, `>= 16` for batched serving).
    pub batch_size: usize,
    /// Length of live suffix match found by `PromptLookup` (`0` if no match).
    pub lookup_match_len: usize,
    /// Number of candidate tokens `PromptLookup` can currently propose.
    pub lookup_draft_avail: usize,
    /// Target logit Shannon entropy (nats) observed on the previous step.
    pub recent_entropy_nats: f64,
}

/// Observed outcome of one verify step fed back into the online EWMA selector.
#[derive(Clone, Copy, Debug)]
pub struct StepFeedback {
    pub strategy: DraftStrategy,
    pub accepted_drafts: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub target_entropy_nats: f64,
}

/// Per-strategy online EWMA statistics.
#[derive(Clone, Debug)]
pub struct StrategyArmStats {
    pub strategy: DraftStrategy,
    pub trials: u64,
    pub ewma_useful_tokens: f64,
    pub ewma_draft_ms: f64,
    pub ewma_verify_ms: f64,
    pub total_useful_tokens: u64,
    pub total_draft_tokens: u64,
}

impl StrategyArmStats {
    fn new(
        strategy: DraftStrategy,
        prior_useful: f64,
        prior_draft_ms: f64,
        prior_verify_ms: f64,
    ) -> Self {
        Self {
            strategy,
            trials: 0,
            ewma_useful_tokens: prior_useful,
            ewma_draft_ms: prior_draft_ms,
            ewma_verify_ms: prior_verify_ms,
            total_useful_tokens: 0,
            total_draft_tokens: 0,
        }
    }

    /// Expected useful tokens per millisecond: `(1 + E[accepted]) / (T_draft + T_verify)`.
    pub fn expected_useful_tok_per_ms(&self) -> f64 {
        let denom = (self.ewma_draft_ms + self.ewma_verify_ms).max(1e-6);
        self.ewma_useful_tokens / denom
    }
}

/// Summary of selector decisions and stand-down events across a run.
#[derive(Clone, Debug, Default)]
pub struct SelectorTelemetry {
    pub plain_steps: u64,
    pub lookup_steps: u64,
    pub linear_steps: u64,
    pub tree_steps: u64,
    pub stand_down_events: u64,
    pub compute_bound_stand_downs: u64,
    pub switches: u64,
}

/// Online EWMA workload-aware draft source & verification-shape selector.
#[derive(Clone, Debug)]
pub struct WorkloadAwareSelector {
    pub config: SelectorConfig,
    arms: [StrategyArmStats; 4],
    active: DraftStrategy,
    consecutive_low_gain: usize,
    cooldown_remaining: usize,
    step_counter: u64,
    pub telemetry: SelectorTelemetry,
}

impl WorkloadAwareSelector {
    pub fn new(config: SelectorConfig) -> Self {
        let plain = DraftStrategy::PlainDecode;
        let lookup = DraftStrategy::PromptLookup {
            max_draft: config.lookup_max_draft,
        };
        let linear = DraftStrategy::Eagle3Linear {
            depth: config.linear_depth,
        };
        let tree = DraftStrategy::Eagle3Tree {
            depth: config.tree_depth,
            fanout: config.tree_fanout,
            max_nodes: config.tree_max_nodes,
        };

        // Calibrated initial priors (quickly overwritten by measured EWMA within 1-2 steps!)
        let arms = [
            StrategyArmStats::new(plain, 1.0, 0.0, 1.0),
            StrategyArmStats::new(lookup, 3.2, 0.005, 1.35),
            StrategyArmStats::new(linear, 2.2, 0.09, 1.35),
            StrategyArmStats::new(tree, 2.7, 0.16, 1.55),
        ];

        Self {
            config,
            arms,
            active: plain,
            consecutive_low_gain: 0,
            cooldown_remaining: 0,
            step_counter: 0,
            telemetry: SelectorTelemetry::default(),
        }
    }

    /// Reset transient turn-level cooldown/hysteresis state at a new conversation turn
    /// while preserving calibrated hardware timing ratios (`ewma_verify_ms`)!
    pub fn begin_turn(&mut self) {
        self.consecutive_low_gain = 0;
        self.cooldown_remaining = 0;
        // Re-center acceptance priors mildly toward neutral so a new turn adapts immediately
        // without discarding PromptLookup's strong code-edit prior.
        self.arms[1].ewma_useful_tokens = self.arms[1].ewma_useful_tokens.max(4.2);
        self.arms[2].ewma_useful_tokens = 0.5 * self.arms[2].ewma_useful_tokens + 0.5 * 2.1;
        self.arms[3].ewma_useful_tokens = 0.5 * self.arms[3].ewma_useful_tokens + 0.5 * 2.5;
    }

    pub fn arm_stats(&self) -> &[StrategyArmStats; 4] {
        &self.arms
    }

    /// Select the optimal draft strategy for the upcoming step using expected useful tokens/ms,
    /// complexity-preserving tie-breaking, compute-bound batch guard, and stand-down hysteresis.
    pub fn select(&mut self, signals: StepSignals) -> DraftStrategy {
        self.step_counter += 1;

        // 1. Compute-bound large-batch guard: extra verification tokens saturate GPU FLOPs.
        if signals.batch_size >= self.config.compute_bound_batch_threshold {
            self.telemetry.compute_bound_stand_downs += 1;
            return self.record_choice(DraftStrategy::PlainDecode);
        }

        // 2. Zero-compute PromptLookup fast path: when a live exact suffix match exists in context
        //    (>= min_lookup_match_len), check whether zero-compute PromptLookup beats or matches
        //    PlainDecode; for long exact matches (>= 4 tokens), always prefer the simpler zero-compute path!
        let has_strong_lookup = signals.lookup_match_len >= self.config.min_lookup_match_len
            && signals.lookup_draft_avail > 0;
        if has_strong_lookup {
            let plain_rate = self.arms[0].expected_useful_tok_per_ms();
            let lookup_rate = self.arms[1].expected_useful_tok_per_ms();
            if signals.lookup_match_len >= 4
                || lookup_rate >= plain_rate * (1.0 + self.config.hysteresis_margin * 0.5)
            {
                return self.record_choice(self.arms[1].strategy);
            }
        }

        // 3. Explicit stand-down cooldown: stay on `PlainDecode` without thrashing until cooldown ends.
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            return self.record_choice(DraftStrategy::PlainDecode);
        }

        // 4. Initial cold-start calibration: ensure we have at least 1 real timing measurement of
        //    `PlainDecode` before judging neural speedups, then probe neural arms cleanly.
        if self.arms[0].trials == 0 {
            return self.record_choice(DraftStrategy::PlainDecode);
        }

        // 5. Compare all eligible strategies by `E[useful_tok / ms]` with complexity-preserving
        //    hysteresis: a more complex arm must exceed a simpler arm by `hysteresis_margin`.
        let plain_rate = self.arms[0].expected_useful_tok_per_ms();
        let mut best_strategy = DraftStrategy::PlainDecode;
        let mut best_rate = plain_rate;
        let mut best_rank = 0u8;

        // Evaluate candidates in ascending complexity order: PromptLookup -> Eagle3Linear -> Eagle3Tree
        for idx in 1..4 {
            // Skip PromptLookup if there is no live suffix match on this step (0 draft tokens).
            if idx == 1 && !has_strong_lookup {
                continue;
            }
            let arm = &self.arms[idx];
            // Small UCB exploration nudge only during the first 2 trials so the selector tries
            // both linear and tree shapes early before locking onto the winner.
            let explore_bonus = if arm.trials < 2
                && signals.recent_entropy_nats < self.config.high_entropy_threshold
            {
                0.08 * plain_rate
            } else {
                0.0
            };
            let cand_rate = arm.expected_useful_tok_per_ms() + explore_bonus;
            let cand_rank = arm.strategy.complexity_rank();

            // Require strict hysteresis margin when escalating complexity rank OR switching away
            // from the currently active strategy.
            let required_margin = if cand_rank > best_rank {
                1.0 + self.config.hysteresis_margin
            } else {
                1.0
            };

            if cand_rate >= best_rate * required_margin {
                best_strategy = arm.strategy;
                best_rate = cand_rate;
                best_rank = cand_rank;
            }
        }

        // Extra hysteresis against rapid switching between Eagle3Linear and Eagle3Tree
        if best_strategy != self.active
            && best_strategy.complexity_rank() >= 2
            && self.active.complexity_rank() >= 2
        {
            let active_rate = self.arms[self.active.index()].expected_useful_tok_per_ms();
            if best_rate < active_rate * (1.0 + self.config.hysteresis_margin) {
                best_strategy = self.active;
            }
        }

        self.record_choice(best_strategy)
    }

    fn record_choice(&mut self, chosen: DraftStrategy) -> DraftStrategy {
        if self.step_counter > 1 && chosen != self.active {
            self.telemetry.switches += 1;
        }
        self.active = chosen;
        match chosen {
            DraftStrategy::PlainDecode => self.telemetry.plain_steps += 1,
            DraftStrategy::PromptLookup { .. } => self.telemetry.lookup_steps += 1,
            DraftStrategy::Eagle3Linear { .. } => self.telemetry.linear_steps += 1,
            DraftStrategy::Eagle3Tree { .. } => self.telemetry.tree_steps += 1,
        }
        chosen
    }

    /// Update online EWMA estimators and stand-down state after verifying a step.
    pub fn observe(&mut self, fb: StepFeedback) {
        let idx = fb.strategy.index();
        let useful = (1 + fb.accepted_drafts) as f64;
        let alpha = self.config.ewma_alpha;

        let arm = &mut self.arms[idx];
        if arm.trials == 0 {
            arm.ewma_useful_tokens = useful;
            arm.ewma_draft_ms = fb.draft_ms;
            arm.ewma_verify_ms = fb.verify_ms.max(1e-5);
        } else {
            arm.ewma_useful_tokens = alpha * useful + (1.0 - alpha) * arm.ewma_useful_tokens;
            arm.ewma_draft_ms = alpha * fb.draft_ms + (1.0 - alpha) * arm.ewma_draft_ms;
            arm.ewma_verify_ms =
                alpha * fb.verify_ms.max(1e-5) + (1.0 - alpha) * arm.ewma_verify_ms;
        }
        arm.trials += 1;
        arm.total_useful_tokens += (1 + fb.accepted_drafts) as u64;
        arm.total_draft_tokens += fb.accepted_drafts as u64;

        // Stand-down detector for speculative arms:
        // If step throughput fails to beat plain decode (or acceptance collapses under high entropy),
        // increment consecutive_low_gain and stand down cleanly after `stand_down_patience` steps.
        if idx >= 1 {
            let plain_rate = self.arms[0].expected_useful_tok_per_ms();
            let step_rate = useful / (fb.draft_ms + fb.verify_ms).max(1e-6);
            let low_gain = step_rate < plain_rate * (1.0 + self.config.hysteresis_margin * 0.5)
                || (fb.accepted_drafts == 0
                    && fb.target_entropy_nats >= self.config.high_entropy_threshold);

            if low_gain {
                self.consecutive_low_gain += 1;
                if self.consecutive_low_gain >= self.config.stand_down_patience {
                    self.cooldown_remaining = self.config.cooldown_steps;
                    self.consecutive_low_gain = 0;
                    self.telemetry.stand_down_events += 1;
                }
            } else {
                self.consecutive_low_gain = 0;
            }
        }
    }
}
