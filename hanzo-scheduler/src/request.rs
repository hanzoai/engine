// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Request definitions, per-sequence runtime tracking, and latency percentile helpers.

/// Workload class used for SLO attribution and mixed long/short scheduling analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestClass {
    /// Short interactive chat (`~128..512` prompt tokens, `~64..256` output tokens).
    ShortInteractive,
    /// Long repository / document prefill (`8K..32K` prompt tokens).
    LongRepoPrefill,
    /// Multi-turn coding-agent turn (warm prefix reuse + tool/diff delta prefill + code decode).
    CodingAgentTurn,
}

impl RequestClass {
    /// Stable lowercase identifier.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::ShortInteractive => "short_interactive",
            Self::LongRepoPrefill => "long_repo_prefill",
            Self::CodingAgentTurn => "coding_agent_turn",
        }
    }
}

/// Lifecycle state of a sequence inside the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    /// Waiting in the admission queue (not yet holding GPU KV/GDN memory).
    Waiting,
    /// Resident on GPU and still processing prompt prefill chunks.
    Prefilling,
    /// Resident on GPU and generating output tokens (1 token per decode iteration).
    Decoding,
    /// Preempted by recomputation; returned to waiting queue for re-admission.
    Preempted,
    /// Swapped out to host RAM; awaiting GPU KV/GDN space for swap-in.
    SwappedOut,
    /// Completed generation and freed all KV blocks and GDN slots.
    Finished,
}

/// Immutable request specification submitted to [`crate::scheduler::SchedulerEngine`].
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceRequest {
    /// Unique request identifier (`0..N`).
    pub id: usize,
    /// Workload classification.
    pub class: RequestClass,
    /// Arrival timestamp in milliseconds from simulation start.
    pub arrival_ms: f64,
    /// Total prompt tokens to prefill (`prefix_cached_tokens + uncached_prefill_tokens`).
    pub prompt_tokens: usize,
    /// Warm prefix tokens already cached (e.g., prior turns of a 20-turn agent loop).
    /// Only `prompt_tokens - warm_prefix_tokens` need active prefill compute, while KV
    /// attention attends over the full prefix + active context.
    pub warm_prefix_tokens: usize,
    /// Exact number of output tokens to generate.
    pub output_tokens: usize,
    /// Per-request TTFT SLO target in milliseconds.
    pub slo_ttft_ms: f64,
    /// Per-request ITL SLO target in milliseconds.
    pub slo_itl_ms: f64,
    /// Optional agent session / turn index (for 20-turn loop dependency chaining).
    pub session_id: Option<usize>,
    /// Turn number within an agent session (`0` for cold turn, `1..=20` for warm turns).
    pub turn_index: usize,
    /// Optional dependency request ID that must complete before this turn arrives.
    pub depends_on: Option<usize>,
}

impl InferenceRequest {
    /// Number of uncached prompt tokens that must actually be prefilled.
    #[must_use]
    pub const fn uncached_prompt_tokens(&self) -> usize {
        self.prompt_tokens.saturating_sub(self.warm_prefix_tokens)
    }

    /// True if this is a warm-cache turn (`warm_prefix_tokens > 0`).
    #[must_use]
    pub const fn is_warm(&self) -> bool {
        self.warm_prefix_tokens > 0
    }
}

/// Mutable execution state and telemetry for an [`InferenceRequest`].
#[derive(Debug, Clone)]
pub struct SequenceRuntime {
    /// Immutable request specification.
    pub req: InferenceRequest,
    /// Effective arrival timestamp (set once any `depends_on` predecessor finishes).
    pub ready_ms: f64,
    /// Current scheduling state.
    pub state: SequenceState,
    /// Number of uncached prompt tokens currently materialized in GPU KV/GDN (`0..=req.uncached_prompt_tokens()`).
    pub prefilled_uncached: usize,
    /// Number of output tokens emitted to the caller so far (`0..=req.output_tokens`).
    pub generated_tokens: usize,
    /// Number of previously emitted output tokens currently materialized in GPU KV/GDN (`0..=generated_tokens`).
    pub materialized_generated: usize,
    /// Timestamp when the first output token was produced (ms).
    pub first_token_ms: Option<f64>,
    /// Timestamp when the most recent output token was produced (ms).
    pub last_token_ms: Option<f64>,
    /// Timestamp when the final output token was produced (ms).
    pub completion_ms: Option<f64>,
    /// Exact inter-token latencies (ms) observed between consecutive generated tokens.
    pub itl_samples_ms: Vec<f64>,
    /// Number of times this sequence was preempted (recompute or host swap).
    pub preemption_count: usize,
    /// Total tokens recomputed due to recomputation preemptions.
    pub recomputed_tokens: usize,
    /// Total time spent waiting in admission queue before first admission (ms).
    pub queue_wait_ms: f64,
    /// Whether this sequence has ever been admitted to GPU memory.
    pub admitted_once: bool,
}

impl SequenceRuntime {
    /// Creates a runtime wrapper for `req`.
    #[must_use]
    pub fn new(req: InferenceRequest) -> Self {
        let ready_ms = req.arrival_ms;
        let cap = req.output_tokens.saturating_sub(1);
        Self {
            req,
            ready_ms,
            state: SequenceState::Waiting,
            prefilled_uncached: 0,
            generated_tokens: 0,
            materialized_generated: 0,
            first_token_ms: None,
            last_token_ms: None,
            completion_ms: None,
            itl_samples_ms: Vec::with_capacity(cap),
            preemption_count: 0,
            recomputed_tokens: 0,
            queue_wait_ms: 0.0,
            admitted_once: false,
        }
    }

    /// Remaining uncached prompt tokens plus any recompute-evicted generated suffix tokens
    /// that must be materialized before decoding can proceed.
    #[must_use]
    pub fn remaining_prefill(&self) -> usize {
        self.req
            .uncached_prompt_tokens()
            .saturating_sub(self.prefilled_uncached)
            + self
                .generated_tokens
                .saturating_sub(self.materialized_generated)
    }

    /// Advances materialized prefill and generated-history replay counters by `chunk_len` tokens.
    pub fn advance_prefill_chunk(&mut self, chunk_len: usize) {
        let prompt_rem = self
            .req
            .uncached_prompt_tokens()
            .saturating_sub(self.prefilled_uncached);
        let prompt_take = chunk_len.min(prompt_rem);
        self.prefilled_uncached += prompt_take;
        let replay_take = chunk_len.saturating_sub(prompt_take);
        let gen_rem = self
            .generated_tokens
            .saturating_sub(self.materialized_generated);
        self.materialized_generated += replay_take.min(gen_rem);
    }

    /// Resets materialized GPU KV/GDN progress on a recompute preemption while preserving
    /// `generated_tokens` already streamed to the caller so they are replayed without
    /// double-counting output tokens.
    pub fn reset_materialized_for_recompute(&mut self) {
        self.prefilled_uncached = 0;
        self.materialized_generated = 0;
        self.state = SequenceState::Preempted;
    }

    /// Total KV context length currently materialized on GPU (warm prefix + materialized prompt + materialized generated).
    #[must_use]
    pub fn current_context_len(&self) -> usize {
        self.req.warm_prefix_tokens + self.prefilled_uncached + self.materialized_generated
    }

    /// Final context length upon completion (`prompt_tokens + output_tokens`).
    #[must_use]
    pub const fn final_context_len(&self) -> usize {
        self.req.prompt_tokens + self.req.output_tokens
    }

    /// Observed TTFT in milliseconds (`first_token_ms - ready_ms`), if first token emitted.
    #[must_use]
    pub fn ttft_ms(&self) -> Option<f64> {
        self.first_token_ms.map(|t| (t - self.ready_ms).max(0.0))
    }

    /// Observed end-to-end latency in milliseconds (`completion_ms - ready_ms`), if finished.
    #[must_use]
    pub fn e2e_latency_ms(&self) -> Option<f64> {
        self.completion_ms.map(|t| (t - self.ready_ms).max(0.0))
    }

    /// Observed `p95` ITL for this individual sequence.
    #[must_use]
    pub fn p95_itl_ms(&self) -> f64 {
        percentile(&self.itl_samples_ms, 0.95)
    }

    /// Remaining ITL slack in milliseconds at `now_ms` (smaller/negative means more urgent).
    #[must_use]
    pub fn itl_slack_ms(&self, now_ms: f64) -> f64 {
        match self.last_token_ms {
            Some(last) => self.req.slo_itl_ms - (now_ms - last),
            None => self.req.slo_itl_ms,
        }
    }

    /// Remaining TTFT slack in milliseconds at `now_ms` (smaller/negative means more urgent).
    #[must_use]
    pub fn ttft_slack_ms(&self, now_ms: f64) -> f64 {
        self.req.slo_ttft_ms - (now_ms - self.ready_ms)
    }

    /// Records emission of one output token at `now_ms` and updates state/ITL telemetry.
    pub fn record_generated_token(&mut self, now_ms: f64) {
        if let Some(prev) = self.last_token_ms {
            self.itl_samples_ms.push((now_ms - prev).max(0.0));
        } else {
            self.first_token_ms = Some(now_ms);
        }
        self.last_token_ms = Some(now_ms);
        self.generated_tokens += 1;
        self.materialized_generated = self.generated_tokens;
        if self.generated_tokens >= self.req.output_tokens {
            self.completion_ms = Some(now_ms);
            self.state = SequenceState::Finished;
        } else {
            self.state = SequenceState::Decoding;
        }
    }
}

/// Computes the nearest-rank percentile `q` (`0.0..=1.0`) of `samples` (returns `0.0` if empty).
#[must_use]
pub fn percentile(samples: &[f64], q: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = samples.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(f64::total_cmp);
    let q_clamped = q.clamp(0.0, 1.0);
    let idx = ((sorted.len() as f64 - 1.0) * q_clamped).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Computes arithmetic mean of `samples` (returns `0.0` if empty).
#[must_use]
pub fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().copied().filter(|v| v.is_finite()).sum();
    sum / (samples.len() as f64)
}
