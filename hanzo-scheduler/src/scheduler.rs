// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Discrete-event and iteration-level continuous batching scheduler implementing 5 matched
//! policies over [`PagedKvPool`] and [`StepCostModel`].

use std::time::Instant;

use crate::config::{PreemptionMode, SchedulerConfig, SchedulingPolicy};
use crate::kv_pool::{PagedKvPool, SchedulerError};
use crate::model::{PrefillChunkCostInput, StepCostModel};
use crate::request::{
    mean, percentile, InferenceRequest, RequestClass, SequenceRuntime, SequenceState,
};

/// Specification of one prefill chunk scheduled inside a [`BatchStepPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillChunkSpec {
    /// Request / sequence identifier.
    pub seq_id: usize,
    /// Prefix length already materialized prior to this chunk.
    pub prefix_len: usize,
    /// Number of new prompt tokens processed in this chunk.
    pub chunk_len: usize,
    /// True if this chunk completes the sequence's prompt prefill and emits token 0.
    pub completes_prefill: bool,
}

/// Exact execution plan emitted for one scheduler iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchStepPlan {
    /// Monotonic iteration counter (`0, 1, 2, ...`).
    pub step_index: usize,
    /// Prefill chunks executed in this iteration.
    pub prefill_chunks: Vec<PrefillChunkSpec>,
    /// Active decoding sequences advancing by 1 output token in this iteration.
    pub decode_seqs: Vec<usize>,
    /// Sequences newly admitted into GPU KV+GDN memory on this iteration.
    pub admitted_seqs: Vec<usize>,
    /// Sequences preempted (recompute or host swap) on this iteration.
    pub preempted_seqs: Vec<usize>,
    /// Sequences restored from host swap on this iteration.
    pub swapped_in_seqs: Vec<usize>,
    /// True when the zero-overhead `C == 1` single-user fast path was selected.
    pub single_user_fast_path: bool,
    /// Total tokens (prefill chunk tokens + decode tokens) processed in this step.
    pub total_step_tokens: usize,
}

/// Comprehensive end-to-end metrics produced by running [`SchedulerEngine::run`].
///
/// ## Honesty Tagging
/// - `cpu_scheduler_wall_ms` and `cpu_us_per_step` are **directly measured** on the host CPU.
/// - `peak_allocated_mib`, `peak_total_vram_gib`, `peak_unbounded_demand_gib`, and `kv_pool_capacity_gib`
///   are **exact unclamped byte accounting**.
/// - `cold_ttft_*`, `warm_ttft_*`, `itl_*`, `makespan_s`, `output_tok_per_s`, `total_tok_per_s`,
///   and `energy_mj_per_output_tok` are **analytical GPU roofline projections** from [`StepCostModel`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleMetrics {
    /// Policy / regime label.
    pub regime_label: &'static str,
    /// Operating profile label.
    pub profile_label: &'static str,
    /// Underlying scheduling policy label.
    pub policy_label: &'static str,
    /// Total requests submitted.
    pub total_requests: usize,
    /// Total requests completed.
    pub completed_requests: usize,
    /// Requests rejected because a single request exceeded physical GPU pool capacity.
    pub oom_rejected_requests: usize,
    /// Total forward iterations executed.
    pub total_steps: usize,
    /// Count of iterations that used the `C == 1` single-user fast path.
    pub single_user_fast_path_steps: usize,
    /// Analytical cold-request TTFT p50 (ms).
    pub cold_ttft_p50_ms: f64,
    /// Analytical cold-request TTFT p95 (ms).
    pub cold_ttft_p95_ms: f64,
    /// Analytical cold-request TTFT mean (ms).
    pub cold_ttft_mean_ms: f64,
    /// Analytical warm-request TTFT p50 (ms) (`0.0` if workload has no warm turns).
    pub warm_ttft_p50_ms: f64,
    /// Analytical warm-request TTFT p95 (ms).
    pub warm_ttft_p95_ms: f64,
    /// Analytical short-interactive TTFT p95 (ms).
    pub short_ttft_p95_ms: f64,
    /// Analytical overall TTFT p95 (ms).
    pub overall_ttft_p95_ms: f64,
    /// Analytical inter-token latency p50 (ms).
    pub itl_p50_ms: f64,
    /// Analytical inter-token latency p95 (ms).
    pub itl_p95_ms: f64,
    /// Analytical inter-token latency p99 (ms).
    pub itl_p99_ms: f64,
    /// Analytical maximum observed single inter-token latency spike (ms).
    pub itl_max_ms: f64,
    /// Analytical end-to-end workload makespan (seconds).
    pub makespan_s: f64,
    /// Analytical mean per-request latency (ms).
    pub mean_e2e_ms: f64,
    /// Analytical generated output throughput per GPU (output tokens / second).
    pub output_tok_per_s: f64,
    /// Analytical total processed throughput per GPU ((prefill + output) tokens / second).
    pub total_tok_per_s: f64,
    /// Percentage of completed requests meeting BOTH TTFT and ITL SLOs (`0.0..=100.0`).
    pub slo_attainment_pct: f64,
    /// Percentage of completed requests meeting their TTFT SLO (`0.0..=100.0`).
    pub ttft_slo_pct: f64,
    /// Percentage of completed requests meeting their ITL SLO (`0.0..=100.0`).
    pub itl_slo_pct: f64,
    /// Total preemption events (recompute + swap-out).
    pub total_preemptions: usize,
    /// Total prompt tokens discarded and recomputed due to preemptions.
    pub total_recomputed_tokens: usize,
    /// Total bytes transferred to/from host RAM during swap preemptions.
    pub total_swap_bytes: u64,
    /// Mean admission queue wait time before first GPU admission (ms).
    pub mean_queue_wait_ms: f64,
    /// Exact peak allocated KV + GDN pool memory on GPU (MiB).
    pub peak_kv_gdn_mib: f64,
    /// Exact peak total GPU VRAM footprint (weights + workspace + peak KV/GDN) in GiB.
    pub peak_total_vram_gib: f64,
    /// Exact physical KV + GDN pool capacity on GPU (GiB).
    pub kv_pool_capacity_gib: f64,
    /// Exact unclamped peak simultaneous KV + GDN demand of all arrived unfinished requests (GiB).
    pub peak_unbounded_demand_gib: f64,
    /// Analytical GPU energy per generated output token (`mJ / output token`).
    pub energy_mj_per_output_tok: f64,
    /// Analytical GPU energy per total processed token (`mJ / total token`).
    pub energy_mj_per_total_tok: f64,
    /// Directly measured CPU scheduler execution wall time (ms).
    pub cpu_scheduler_wall_ms: f64,
    /// Directly measured CPU scheduler overhead per iteration (`µs / step`).
    pub cpu_us_per_step: f64,
    /// True if all KV blocks and GDN slots returned cleanly (`0` leaks).
    pub kv_pool_clean: bool,
    /// Total generated output tokens across all completed requests.
    pub total_output_tokens: usize,
    /// Total uncached prefill tokens processed (excluding recomputed tokens).
    pub total_prefill_tokens: usize,
}

/// Complete simulation result containing aggregate [`ScheduleMetrics`] and per-request telemetry.
#[derive(Debug, Clone)]
pub struct SimulationOutput {
    /// Aggregate schedule scorecard metrics.
    pub metrics: ScheduleMetrics,
    /// Final runtime state of every submitted request (ordered by `req.id`).
    pub sequences: Vec<SequenceRuntime>,
    /// Optional per-step trace (retained when `record_trace` is true).
    pub trace: Vec<BatchStepPlan>,
}

/// Core continuous batching and chunked prefill scheduler engine.
pub struct SchedulerEngine {
    config: SchedulerConfig,
    cost_model: StepCostModel,
    pool: PagedKvPool,
    record_trace: bool,
}

impl SchedulerEngine {
    /// Creates a new [`SchedulerEngine`] with a [`PagedKvPool`] automatically sized for
    /// `cost_model.model` and `cost_model.hardware`.
    ///
    /// # Errors
    /// Returns [`SchedulerError`] if configuration or VRAM pool sizing is invalid.
    pub fn new(config: SchedulerConfig, cost_model: StepCostModel) -> Result<Self, SchedulerError> {
        config.validate()?;
        let pool = PagedKvPool::from_model_and_hardware(
            &cost_model.model,
            &cost_model.hardware,
            config.block_size,
            config.max_num_seqs,
        )?;
        Ok(Self {
            config,
            cost_model,
            pool,
            record_trace: false,
        })
    }

    /// Creates a [`SchedulerEngine`] with an explicitly sized [`PagedKvPool`] (for controlled
    /// oversubscription experiments and unit tests).
    ///
    /// # Errors
    /// Returns [`SchedulerError`] if configuration validation fails.
    pub fn with_pool(
        config: SchedulerConfig,
        cost_model: StepCostModel,
        pool: PagedKvPool,
    ) -> Result<Self, SchedulerError> {
        config.validate()?;
        Ok(Self {
            config,
            cost_model,
            pool,
            record_trace: false,
        })
    }

    /// Enables or disables capturing the full per-iteration [`BatchStepPlan`] trace.
    #[must_use]
    pub const fn with_trace(mut self, enabled: bool) -> Self {
        self.record_trace = enabled;
        self
    }

    /// Runs the configured scheduling policy over `requests` until all admissible requests finish.
    ///
    /// # Errors
    /// Returns [`SchedulerError`] only on internal invariant violations; normal KV pressure is
    /// handled gracefully via admission control or preemption.
    pub fn run(
        mut self,
        requests: Vec<InferenceRequest>,
    ) -> Result<SimulationOutput, SchedulerError> {
        let cpu_start = Instant::now();
        let total_requests = requests.len();
        let mut seqs: Vec<SequenceRuntime> =
            requests.into_iter().map(SequenceRuntime::new).collect();
        // Sort by id so `seqs[id]` matches `req.id` when ids are dense `0..N`; maintain lookup map.
        seqs.sort_by_key(|s| s.req.id);
        let id_to_idx: Vec<(usize, usize)> = seqs
            .iter()
            .enumerate()
            .map(|(idx, s)| (s.req.id, idx))
            .collect();
        let lookup_idx = |req_id: usize| -> Option<usize> {
            id_to_idx
                .binary_search_by_key(&req_id, |&(id, _)| id)
                .ok()
                .map(|pos| id_to_idx[pos].1)
        };

        // Track which requests are released (arrival_ms <= now_ms AND dependency completed).
        let mut released = vec![false; total_requests];
        let mut waiting: Vec<usize> = Vec::new();
        let mut active: Vec<usize> = Vec::new();
        let mut swapped: Vec<usize> = Vec::new();
        let mut oom_rejected: usize = 0;

        let mut now_ms: f64 = seqs
            .iter()
            .filter(|s| s.req.depends_on.is_none())
            .map(|s| s.req.arrival_ms)
            .fold(f64::INFINITY, f64::min);
        if !now_ms.is_finite() {
            now_ms = 0.0;
        }
        let sim_start_ms = now_ms;

        let mut step_index: usize = 0;
        let mut single_user_fast_path_steps: usize = 0;
        let mut consecutive_decode_steps: usize = 0;
        let mut total_energy_joules: f64 = 0.0;
        let mut total_swap_bytes: u64 = 0;
        let mut peak_unbounded_demand_bytes: u64 = 0;
        let mut trace: Vec<BatchStepPlan> = Vec::new();

        // Hard safety guard against pathological livelock (e.g., extreme thrashing).
        let max_iterations: usize = (total_requests.max(1) * 4000).max(200_000);

        loop {
            // 1. Release all requests whose dependencies are finished and arrival_ms <= now_ms.
            release_ready_requests(&mut seqs, &mut released, &mut waiting, now_ms, &lookup_idx);

            // Track exact unclamped KV + GDN demand of all currently arrived unfinished requests.
            let unbounded_demand = compute_unbounded_demand_bytes(
                &seqs,
                &waiting,
                &active,
                &swapped,
                &self.cost_model.model,
                self.pool.block_size(),
            );
            if unbounded_demand > peak_unbounded_demand_bytes {
                peak_unbounded_demand_bytes = unbounded_demand;
            }

            // If nothing is active, swapped, or waiting, advance clock to next future arrival.
            if active.is_empty() && swapped.is_empty() && waiting.is_empty() {
                let next_arrival = next_unreleased_arrival_ms(&seqs, &released, &lookup_idx);
                match next_arrival {
                    Some(t) => {
                        now_ms = now_ms.max(t);
                        continue;
                    }
                    None => break,
                }
            }

            // Static batching formation window: when idle, allow briefly arriving requests to
            // coalesce into the static batch if another independent request arrives soon.
            if self.config.policy == SchedulingPolicy::StaticBatchingFcfs
                && active.is_empty()
                && swapped.is_empty()
                && !waiting.is_empty()
                && waiting.len() < self.config.max_num_seqs
                && self.config.static_batch_wait_ms > 0.0
            {
                if let Some(next_t) = next_unreleased_arrival_ms(&seqs, &released, &lookup_idx) {
                    let window_end = now_ms + self.config.static_batch_wait_ms;
                    if next_t <= window_end {
                        now_ms = next_t;
                        release_ready_requests(
                            &mut seqs,
                            &mut released,
                            &mut waiting,
                            now_ms,
                            &lookup_idx,
                        );
                    }
                }
            }

            // Reject any single request that physically cannot fit even in an empty GPU pool.
            if active.is_empty() && swapped.is_empty() && !waiting.is_empty() {
                let mut kept = Vec::with_capacity(waiting.len());
                for idx in waiting.drain(..) {
                    let needed_initial = self.initial_admission_tokens(&seqs[idx]);
                    if !self.pool.can_admit(needed_initial, 0)
                        || self.pool.blocks_for_tokens(seqs[idx].final_context_len())
                            > self.pool.total_kv_blocks()
                    {
                        seqs[idx].state = SequenceState::Finished;
                        oom_rejected += 1;
                    } else {
                        kept.push(idx);
                    }
                }
                waiting = kept;
                if waiting.is_empty() {
                    continue;
                }
            }

            if step_index >= max_iterations {
                for idx in active.drain(..).chain(swapped.drain(..)) {
                    self.pool.free_sequence(seqs[idx].req.id);
                }
                let unfinished_requests = seqs
                    .iter()
                    .filter(|s| s.state != SequenceState::Finished)
                    .count();
                let completed_requests = seqs.iter().filter(|s| s.completion_ms.is_some()).count();
                return Err(SchedulerError::SimulationStepLimitExceeded {
                    max_iterations,
                    completed_requests,
                    unfinished_requests,
                });
            }

            // 2. Restore swapped-out sequences if space is available before admitting new work.
            let mut swapped_in_seqs: Vec<usize> = Vec::new();
            let mut swap_stall_ms: f64 = 0.0;
            if !swapped.is_empty() {
                let headroom = self.active_decode_headroom_blocks(&seqs, &active);
                let mut still_swapped = Vec::new();
                for idx in swapped.drain(..) {
                    let seq_id = seqs[idx].req.id;
                    if active.len() < self.config.max_num_seqs
                        && self.pool.can_swap_in(seq_id, headroom)
                    {
                        let bytes = self.pool.swap_in(seq_id)?;
                        total_swap_bytes += bytes;
                        swap_stall_ms += self.cost_model.estimate_swap_ms(bytes);
                        seqs[idx].state = if seqs[idx].remaining_prefill() > 0 {
                            SequenceState::Prefilling
                        } else {
                            SequenceState::Decoding
                        };
                        active.push(idx);
                        swapped_in_seqs.push(seq_id);
                    } else {
                        still_swapped.push(idx);
                    }
                }
                swapped = still_swapped;
            }

            // 3. Policy-specific admission & step planning.
            let mut admitted_seqs: Vec<usize> = Vec::new();
            let mut preempted_seqs: Vec<usize> = Vec::new();
            let mut prefill_chunks: Vec<PrefillChunkSpec> = Vec::new();
            let mut decode_seqs: Vec<usize> = Vec::new();
            let mut single_user_fast_path = false;

            match self.config.policy {
                SchedulingPolicy::StaticBatchingFcfs => {
                    // Only admit a fresh batch when the previous batch has completely drained.
                    if active.is_empty() && swapped.is_empty() {
                        self.admit_fcfs_batch(
                            &mut seqs,
                            &mut waiting,
                            &mut active,
                            now_ms,
                            &mut admitted_seqs,
                        )?;
                    }
                    // If any sequence in the static batch still needs prefill, run full unchunked prefill for all of them.
                    let need_prefill: Vec<usize> = active
                        .iter()
                        .copied()
                        .filter(|&i| seqs[i].remaining_prefill() > 0)
                        .collect();
                    if !need_prefill.is_empty() {
                        for idx in need_prefill {
                            let rem = seqs[idx].remaining_prefill();
                            let prefix_len = seqs[idx].current_context_len();
                            let grown = self.ensure_kv_for_step(
                                idx,
                                prefix_len + rem,
                                &mut seqs,
                                &mut active,
                                &mut waiting,
                                &mut swapped,
                                &mut prefill_chunks,
                                &mut decode_seqs,
                                &mut preempted_seqs,
                                &mut total_swap_bytes,
                                &mut swap_stall_ms,
                            )?;
                            if grown {
                                prefill_chunks.push(PrefillChunkSpec {
                                    seq_id: seqs[idx].req.id,
                                    prefix_len,
                                    chunk_len: rem,
                                    completes_prefill: true,
                                });
                            }
                        }
                    } else {
                        // All sequences in static batch decode together until every sequence finishes.
                        let active_snapshot = active.clone();
                        for idx in active_snapshot {
                            if seqs[idx].state == SequenceState::Decoding {
                                let target_len = seqs[idx].current_context_len() + 1;
                                let grown = self.ensure_kv_for_step(
                                    idx,
                                    target_len,
                                    &mut seqs,
                                    &mut active,
                                    &mut waiting,
                                    &mut swapped,
                                    &mut prefill_chunks,
                                    &mut decode_seqs,
                                    &mut preempted_seqs,
                                    &mut total_swap_bytes,
                                    &mut swap_stall_ms,
                                )?;
                                if grown {
                                    decode_seqs.push(seqs[idx].req.id);
                                }
                            }
                        }
                    }
                }

                SchedulingPolicy::ContinuousBatchingPrefillFirst => {
                    // Always try to admit waiting requests immediately and run full unchunked prefill first.
                    self.admit_fcfs_batch(
                        &mut seqs,
                        &mut waiting,
                        &mut active,
                        now_ms,
                        &mut admitted_seqs,
                    )?;
                    let need_prefill: Vec<usize> = active
                        .iter()
                        .copied()
                        .filter(|&i| seqs[i].remaining_prefill() > 0)
                        .collect();
                    if !need_prefill.is_empty() {
                        // Prefill-first stalls active decodes until every waiting/new prefill finishes.
                        for idx in need_prefill {
                            let rem = seqs[idx].remaining_prefill();
                            let prefix_len = seqs[idx].current_context_len();
                            let grown = self.ensure_kv_for_step(
                                idx,
                                prefix_len + rem,
                                &mut seqs,
                                &mut active,
                                &mut waiting,
                                &mut swapped,
                                &mut prefill_chunks,
                                &mut decode_seqs,
                                &mut preempted_seqs,
                                &mut total_swap_bytes,
                                &mut swap_stall_ms,
                            )?;
                            if grown {
                                prefill_chunks.push(PrefillChunkSpec {
                                    seq_id: seqs[idx].req.id,
                                    prefix_len,
                                    chunk_len: rem,
                                    completes_prefill: true,
                                });
                            }
                        }
                    } else {
                        let active_snapshot = active.clone();
                        for idx in active_snapshot {
                            if seqs[idx].state == SequenceState::Decoding {
                                let target_len = seqs[idx].current_context_len() + 1;
                                let grown = self.ensure_kv_for_step(
                                    idx,
                                    target_len,
                                    &mut seqs,
                                    &mut active,
                                    &mut waiting,
                                    &mut swapped,
                                    &mut prefill_chunks,
                                    &mut decode_seqs,
                                    &mut preempted_seqs,
                                    &mut total_swap_bytes,
                                    &mut swap_stall_ms,
                                )?;
                                if grown {
                                    decode_seqs.push(seqs[idx].req.id);
                                }
                            }
                        }
                    }
                }

                SchedulingPolicy::ContinuousBatchingDecodePriority => {
                    let has_decoding = active
                        .iter()
                        .any(|&i| seqs[i].state == SequenceState::Decoding);
                    let must_run_prefill = !has_decoding
                        || (consecutive_decode_steps >= self.config.decode_priority_starvation_cap
                            && (!waiting.is_empty()
                                || active.iter().any(|&i| seqs[i].remaining_prefill() > 0)));
                    if must_run_prefill {
                        consecutive_decode_steps = 0;
                        // Admit at most 1 waiting prefill when breaking starvation (or a batch if idle).
                        if !has_decoding {
                            self.admit_fcfs_batch(
                                &mut seqs,
                                &mut waiting,
                                &mut active,
                                now_ms,
                                &mut admitted_seqs,
                            )?;
                        } else if !waiting.is_empty() {
                            self.admit_up_to_k(
                                1,
                                &mut seqs,
                                &mut waiting,
                                &mut active,
                                now_ms,
                                &mut admitted_seqs,
                            )?;
                        }
                        let need_prefill: Vec<usize> = active
                            .iter()
                            .copied()
                            .filter(|&i| seqs[i].remaining_prefill() > 0)
                            .collect();
                        for idx in need_prefill {
                            let rem = seqs[idx].remaining_prefill();
                            let prefix_len = seqs[idx].current_context_len();
                            let grown = self.ensure_kv_for_step(
                                idx,
                                prefix_len + rem,
                                &mut seqs,
                                &mut active,
                                &mut waiting,
                                &mut swapped,
                                &mut prefill_chunks,
                                &mut decode_seqs,
                                &mut preempted_seqs,
                                &mut total_swap_bytes,
                                &mut swap_stall_ms,
                            )?;
                            if grown {
                                prefill_chunks.push(PrefillChunkSpec {
                                    seq_id: seqs[idx].req.id,
                                    prefix_len,
                                    chunk_len: rem,
                                    completes_prefill: true,
                                });
                            }
                        }
                    } else {
                        consecutive_decode_steps += 1;
                        let active_snapshot = active.clone();
                        for idx in active_snapshot {
                            if seqs[idx].state == SequenceState::Decoding {
                                let target_len = seqs[idx].current_context_len() + 1;
                                let grown = self.ensure_kv_for_step(
                                    idx,
                                    target_len,
                                    &mut seqs,
                                    &mut active,
                                    &mut waiting,
                                    &mut swapped,
                                    &mut prefill_chunks,
                                    &mut decode_seqs,
                                    &mut preempted_seqs,
                                    &mut total_swap_bytes,
                                    &mut swap_stall_ms,
                                )?;
                                if grown {
                                    decode_seqs.push(seqs[idx].req.id);
                                }
                            }
                        }
                    }
                }

                SchedulingPolicy::ChunkedPrefillTokenBudget => {
                    // Admit waiting requests first (either KV-guarded or unguarded for Neg 2).
                    self.admit_fcfs_batch(
                        &mut seqs,
                        &mut waiting,
                        &mut active,
                        now_ms,
                        &mut admitted_seqs,
                    )?;

                    let mut budget_left = self.config.token_budget;
                    // 1. Pack active decoding sequences first (up to budget, leaving room for prefill progress
                    //    if budget is ultra-small like 64 and there are many decodes).
                    let active_decodes: Vec<usize> = active
                        .iter()
                        .copied()
                        .filter(|&i| seqs[i].state == SequenceState::Decoding)
                        .collect();
                    let has_prefill = active.iter().any(|&i| seqs[i].remaining_prefill() > 0);
                    let max_decodes_this_step =
                        if has_prefill && active_decodes.len() >= budget_left {
                            (budget_left / 2).max(1)
                        } else {
                            budget_left
                        };
                    for idx in active_decodes.into_iter().take(max_decodes_this_step) {
                        if !self.pool.is_resident(seqs[idx].req.id) {
                            continue;
                        }
                        let target_len = seqs[idx].current_context_len() + 1;
                        let grown = self.ensure_kv_for_step(
                            idx,
                            target_len,
                            &mut seqs,
                            &mut active,
                            &mut waiting,
                            &mut swapped,
                            &mut prefill_chunks,
                            &mut decode_seqs,
                            &mut preempted_seqs,
                            &mut total_swap_bytes,
                            &mut swap_stall_ms,
                        )?;
                        if grown && budget_left > 0 {
                            decode_seqs.push(seqs[idx].req.id);
                            budget_left -= 1;
                        }
                    }

                    // 2. Piggyback prefill chunks into remaining token budget.
                    // When KV admission control is disabled (unguarded oversubscription regime),
                    // distribute prefill tokens across admitted sequences in round-robin slices
                    // so concurrent long prompts grow their KV tables simultaneously.
                    let mut active_prefills: Vec<usize> = active
                        .iter()
                        .copied()
                        .filter(|&i| seqs[i].remaining_prefill() > 0)
                        .collect();
                    let per_seq_cap = if !self.config.kv_admission_control
                        && active_prefills.len() > 1
                    {
                        let rot = (step_index * 16) % active_prefills.len();
                        active_prefills.rotate_left(rot);
                        (budget_left / active_prefills.len().min(16)).max(self.pool.block_size())
                    } else {
                        budget_left
                    };
                    for idx in active_prefills {
                        if budget_left == 0 {
                            break;
                        }
                        if !self.pool.is_resident(seqs[idx].req.id) {
                            continue;
                        }
                        let rem = seqs[idx].remaining_prefill();
                        let chunk_len = rem.min(budget_left).min(per_seq_cap);
                        let prefix_len = seqs[idx].current_context_len();
                        let grown = self.ensure_kv_for_step(
                            idx,
                            prefix_len + chunk_len,
                            &mut seqs,
                            &mut active,
                            &mut waiting,
                            &mut swapped,
                            &mut prefill_chunks,
                            &mut decode_seqs,
                            &mut preempted_seqs,
                            &mut total_swap_bytes,
                            &mut swap_stall_ms,
                        )?;
                        if grown {
                            prefill_chunks.push(PrefillChunkSpec {
                                seq_id: seqs[idx].req.id,
                                prefix_len,
                                chunk_len,
                                completes_prefill: chunk_len == rem,
                            });
                            budget_left -= chunk_len;
                        }
                    }
                }

                SchedulingPolicy::LatencyAwareChunkedKvScheduler => {
                    // Check automatic C == 1 single-user zero-overhead bypass.
                    let is_single_user = self.config.enable_single_user_bypass
                        && swapped.is_empty()
                        && (active.len() + waiting.len() == 1);

                    if is_single_user {
                        if active.is_empty() {
                            self.admit_latency_aware(
                                &mut seqs,
                                &mut waiting,
                                &mut active,
                                now_ms,
                                &mut admitted_seqs,
                            )?;
                        }
                        if let Some(&idx) = active.first() {
                            single_user_fast_path = true;
                            single_user_fast_path_steps += 1;
                            if seqs[idx].remaining_prefill() > 0 {
                                let rem = seqs[idx].remaining_prefill();
                                let prefix_len = seqs[idx].current_context_len();
                                self.pool
                                    .allocate_for_len(seqs[idx].req.id, prefix_len + rem)?;
                                prefill_chunks.push(PrefillChunkSpec {
                                    seq_id: seqs[idx].req.id,
                                    prefix_len,
                                    chunk_len: rem,
                                    completes_prefill: true,
                                });
                            } else {
                                let target_len = seqs[idx].current_context_len() + 1;
                                self.pool.allocate_for_len(seqs[idx].req.id, target_len)?;
                                decode_seqs.push(seqs[idx].req.id);
                            }
                        }
                    } else {
                        // 1. KV-pressure-aware & slack/anti-HoL admission control.
                        self.admit_latency_aware(
                            &mut seqs,
                            &mut waiting,
                            &mut active,
                            now_ms,
                            &mut admitted_seqs,
                        )?;

                        // 2. EDF / slack-ordered active decodes (bounded by token_budget).
                        let mut active_decodes: Vec<usize> = active
                            .iter()
                            .copied()
                            .filter(|&i| seqs[i].state == SequenceState::Decoding)
                            .collect();
                        active_decodes.sort_by(|&a, &b| {
                            seqs[a]
                                .itl_slack_ms(now_ms)
                                .total_cmp(&seqs[b].itl_slack_ms(now_ms))
                        });
                        let has_prefill = active.iter().any(|&i| seqs[i].remaining_prefill() > 0);
                        let max_decodes_this_step =
                            if has_prefill && active_decodes.len() >= self.config.token_budget {
                                (self.config.token_budget / 2).max(1)
                            } else {
                                self.config.token_budget
                            };

                        let mut decode_ctx_lens: Vec<usize> =
                            Vec::with_capacity(active_decodes.len().min(max_decodes_this_step));
                        let mut min_decode_slack_ms = f64::INFINITY;
                        for idx in active_decodes.into_iter().take(max_decodes_this_step) {
                            if !self.pool.is_resident(seqs[idx].req.id) {
                                continue;
                            }
                            let target_len = seqs[idx].current_context_len() + 1;
                            let grown = self.ensure_kv_for_step(
                                idx,
                                target_len,
                                &mut seqs,
                                &mut active,
                                &mut waiting,
                                &mut swapped,
                                &mut prefill_chunks,
                                &mut decode_seqs,
                                &mut preempted_seqs,
                                &mut total_swap_bytes,
                                &mut swap_stall_ms,
                            )?;
                            if grown {
                                min_decode_slack_ms =
                                    min_decode_slack_ms.min(seqs[idx].itl_slack_ms(now_ms));
                                decode_ctx_lens.push(seqs[idx].current_context_len());
                                decode_seqs.push(seqs[idx].req.id);
                            }
                        }

                        // Re-sync decode_ctx_lens with decode_seqs in case any earlier decode was rolled back.
                        if decode_ctx_lens.len() != decode_seqs.len() {
                            decode_ctx_lens = decode_seqs
                                .iter()
                                .filter_map(|&id| lookup_idx(id))
                                .map(|i| seqs[i].current_context_len())
                                .collect();
                        }

                        // 3. Dynamic roofline-bounded prefill chunk sizing protecting p95 ITL SLO,
                        //    strictly honoring token_budget whether decodes are active or idle.
                        let mut active_prefills: Vec<usize> = active
                            .iter()
                            .copied()
                            .filter(|&i| seqs[i].remaining_prefill() > 0)
                            .collect();
                        // Prioritize short interactive prefills and tightest TTFT slack first.
                        active_prefills.sort_by(|&a, &b| {
                            let a_short = seqs[a].remaining_prefill() <= 1024;
                            let b_short = seqs[b].remaining_prefill() <= 1024;
                            b_short
                                .cmp(&a_short)
                                .then_with(|| {
                                    seqs[a]
                                        .ttft_slack_ms(now_ms)
                                        .total_cmp(&seqs[b].ttft_slack_ms(now_ms))
                                })
                                .then_with(|| {
                                    seqs[a]
                                        .remaining_prefill()
                                        .cmp(&seqs[b].remaining_prefill())
                                })
                        });

                        // If an active decode is dangerously close to breaching ITL SLO, run decode-only
                        // for this single step to protect p95 ITL unless decode-only already exceeds it.
                        let decode_only_ms = if decode_ctx_lens.is_empty() {
                            0.0
                        } else {
                            self.cost_model
                                .estimate_step(&[], &decode_ctx_lens, false)
                                .step_ms
                        };
                        let urgent_decode_guard = !decode_ctx_lens.is_empty()
                            && min_decode_slack_ms < (decode_only_ms + 4.0)
                            && min_decode_slack_ms < 0.45 * self.config.slo.itl_p95_ms;

                        let budget_cap = self
                            .config
                            .token_budget
                            .saturating_sub(decode_seqs.len())
                            .min(self.config.max_prefill_chunk);

                        if !urgent_decode_guard && !active_prefills.is_empty() && budget_cap > 0 {
                            let mut chunk_budget = if decode_ctx_lens.is_empty() {
                                // No active decodes: still strictly honor token_budget and max_prefill_chunk.
                                budget_cap
                            } else {
                                let target_step_ms = (0.88 * self.config.slo.itl_p95_ms).min(
                                    min_decode_slack_ms.max(0.65 * self.config.slo.itl_p95_ms),
                                );
                                let lead = active_prefills[0];
                                self.cost_model.max_chunk_for_step_budget(
                                    seqs[lead].current_context_len(),
                                    seqs[lead].remaining_prefill(),
                                    &decode_ctx_lens,
                                    target_step_ms,
                                    self.config.min_prefill_chunk.min(budget_cap).max(1),
                                    budget_cap,
                                )
                            };
                            chunk_budget = chunk_budget.min(budget_cap);

                            for idx in active_prefills {
                                if chunk_budget == 0 {
                                    break;
                                }
                                if !self.pool.is_resident(seqs[idx].req.id) {
                                    continue;
                                }
                                let rem = seqs[idx].remaining_prefill();
                                let chunk_len = rem.min(chunk_budget);
                                let prefix_len = seqs[idx].current_context_len();
                                let grown = self.ensure_kv_for_step(
                                    idx,
                                    prefix_len + chunk_len,
                                    &mut seqs,
                                    &mut active,
                                    &mut waiting,
                                    &mut swapped,
                                    &mut prefill_chunks,
                                    &mut decode_seqs,
                                    &mut preempted_seqs,
                                    &mut total_swap_bytes,
                                    &mut swap_stall_ms,
                                )?;
                                if grown {
                                    prefill_chunks.push(PrefillChunkSpec {
                                        seq_id: seqs[idx].req.id,
                                        prefix_len,
                                        chunk_len,
                                        completes_prefill: chunk_len == rem,
                                    });
                                    chunk_budget -= chunk_len;
                                }
                            }
                        }
                    }
                }
            }

            // 4. Evaluate roofline step cost and advance simulation clock.
            let prefill_cost_inputs: Vec<PrefillChunkCostInput> = prefill_chunks
                .iter()
                .map(|c| PrefillChunkCostInput {
                    prefix_len: c.prefix_len,
                    chunk_len: c.chunk_len,
                })
                .collect();
            let decode_ctx_lens: Vec<usize> = decode_seqs
                .iter()
                .filter_map(|&id| lookup_idx(id))
                .map(|idx| seqs[idx].current_context_len())
                .collect();
            let est = self.cost_model.estimate_step(
                &prefill_cost_inputs,
                &decode_ctx_lens,
                single_user_fast_path,
            );
            now_ms += est.step_ms + swap_stall_ms;
            total_energy_joules += est.energy_joules;

            // 5. Commit prefill chunk advances (including recompute history replay) and decode token emissions.
            let mut finished_indices: Vec<usize> = Vec::new();
            for chunk in &prefill_chunks {
                if let Some(idx) = lookup_idx(chunk.seq_id) {
                    let is_recompute_replay = seqs[idx].generated_tokens > 0;
                    seqs[idx].advance_prefill_chunk(chunk.chunk_len);
                    if chunk.completes_prefill {
                        if is_recompute_replay {
                            // Rebuilt prompt + consumed generated suffix and GDN state without
                            // double-counting already emitted tokens; resume decoding next step.
                            seqs[idx].state = SequenceState::Decoding;
                        } else {
                            // Completing initial prompt prefill emits output token 0 (TTFT).
                            seqs[idx].record_generated_token(now_ms);
                            if seqs[idx].state == SequenceState::Finished {
                                finished_indices.push(idx);
                            }
                        }
                    }
                }
            }
            for &seq_id in &decode_seqs {
                if let Some(idx) = lookup_idx(seq_id) {
                    seqs[idx].record_generated_token(now_ms);
                    if seqs[idx].state == SequenceState::Finished {
                        finished_indices.push(idx);
                    }
                }
            }

            // 6. Free KV blocks and GDN slots for completed sequences.
            // StaticBatchingFcfs holds batch slots until all sequences in the batch finish!
            if self.config.policy == SchedulingPolicy::StaticBatchingFcfs {
                let all_done = active
                    .iter()
                    .all(|&i| seqs[i].state == SequenceState::Finished);
                if all_done {
                    for idx in active.drain(..) {
                        self.pool.free_sequence(seqs[idx].req.id);
                    }
                }
            } else if !finished_indices.is_empty() {
                for &idx in &finished_indices {
                    self.pool.free_sequence(seqs[idx].req.id);
                }
                active.retain(|&i| seqs[i].state != SequenceState::Finished);
            }

            let total_step_tokens =
                prefill_chunks.iter().map(|c| c.chunk_len).sum::<usize>() + decode_seqs.len();
            if self.record_trace {
                trace.push(BatchStepPlan {
                    step_index,
                    prefill_chunks,
                    decode_seqs,
                    admitted_seqs,
                    preempted_seqs,
                    swapped_in_seqs,
                    single_user_fast_path,
                    total_step_tokens,
                });
            }
            step_index += 1;
        }

        // Cleanup any remaining allocations.
        for idx in active.drain(..).chain(swapped.drain(..)) {
            self.pool.free_sequence(seqs[idx].req.id);
        }

        let cpu_scheduler_wall_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;
        let metrics = self.finalize_metrics(
            &seqs,
            total_requests,
            oom_rejected,
            step_index,
            single_user_fast_path_steps,
            sim_start_ms,
            now_ms,
            total_energy_joules,
            total_swap_bytes,
            peak_unbounded_demand_bytes,
            cpu_scheduler_wall_ms,
        );

        Ok(SimulationOutput {
            metrics,
            sequences: seqs,
            trace,
        })
    }

    /// Computes initial KV token reservation for admission of `seq`.
    fn initial_admission_tokens(&self, seq: &SequenceRuntime) -> usize {
        if self.config.kv_admission_control {
            // Reserve full prompt + any generated suffix to replay + modest decode reserve
            // so admitted sequences do not immediately thrash KV blocks.
            let remaining_outputs = seq.req.output_tokens.saturating_sub(seq.generated_tokens);
            let decode_reserve = remaining_outputs.min(32);
            seq.req.prompt_tokens + seq.generated_tokens + decode_reserve
        } else {
            // Unguarded admission (Negative Regime 2): only checks first chunk + warm prefix!
            let first_chunk = seq.remaining_prefill().min(self.config.token_budget).max(1);
            seq.req.warm_prefix_tokens + first_chunk
        }
    }

    fn active_decode_headroom_blocks(&self, seqs: &[SequenceRuntime], active: &[usize]) -> usize {
        if !self.config.kv_admission_control {
            return 0;
        }
        let decoding_count = active
            .iter()
            .filter(|&&i| seqs[i].state == SequenceState::Decoding)
            .count();
        decoding_count * self.config.decode_headroom_blocks_per_seq
    }

    /// Standard FCFS admission helper used by Policies 1–4.
    fn admit_fcfs_batch(
        &mut self,
        seqs: &mut [SequenceRuntime],
        waiting: &mut Vec<usize>,
        active: &mut Vec<usize>,
        now_ms: f64,
        admitted_seqs: &mut Vec<usize>,
    ) -> Result<(), SchedulerError> {
        self.admit_up_to_k(
            self.config.max_num_seqs.saturating_sub(active.len()),
            seqs,
            waiting,
            active,
            now_ms,
            admitted_seqs,
        )
    }

    fn admit_up_to_k(
        &mut self,
        limit: usize,
        seqs: &mut [SequenceRuntime],
        waiting: &mut Vec<usize>,
        active: &mut Vec<usize>,
        now_ms: f64,
        admitted_seqs: &mut Vec<usize>,
    ) -> Result<(), SchedulerError> {
        let mut admitted_now = 0usize;
        while admitted_now < limit && active.len() < self.config.max_num_seqs && !waiting.is_empty()
        {
            let idx = waiting[0];
            let headroom = if active.is_empty() {
                0
            } else {
                self.active_decode_headroom_blocks(seqs, active)
            };
            let check_tokens = self.initial_admission_tokens(&seqs[idx]);
            let util_ok = active.is_empty()
                || !self.config.kv_admission_control
                || self.pool.kv_utilization() < self.config.kv_high_watermark;
            if util_ok && self.pool.can_admit(check_tokens, headroom) {
                waiting.remove(0);
                let alloc_tokens = seqs[idx].req.warm_prefix_tokens.saturating_add(1).max(1);
                self.pool.admit_sequence(seqs[idx].req.id, alloc_tokens)?;
                if !seqs[idx].admitted_once {
                    seqs[idx].admitted_once = true;
                    seqs[idx].queue_wait_ms = (now_ms - seqs[idx].ready_ms).max(0.0);
                }
                seqs[idx].state = if seqs[idx].remaining_prefill() > 0 {
                    SequenceState::Prefilling
                } else {
                    SequenceState::Decoding
                };
                active.push(idx);
                admitted_seqs.push(seqs[idx].req.id);
                admitted_now += 1;
            } else {
                // Head-of-line blocks under pure FCFS.
                break;
            }
        }
        Ok(())
    }

    /// Latency-aware, KV-pressure-guarded, anti-HoL admission control (Policy 5).
    fn admit_latency_aware(
        &mut self,
        seqs: &mut [SequenceRuntime],
        waiting: &mut Vec<usize>,
        active: &mut Vec<usize>,
        now_ms: f64,
        admitted_seqs: &mut Vec<usize>,
    ) -> Result<(), SchedulerError> {
        if waiting.is_empty() || active.len() >= self.config.max_num_seqs {
            return Ok(());
        }
        // Order candidates: prioritize short interactive requests and tightest TTFT slack
        // so short chats never get head-of-line blocked behind a 32K repo prefill.
        waiting.sort_by(|&a, &b| {
            let a_short = seqs[a].req.uncached_prompt_tokens() <= 1024;
            let b_short = seqs[b].req.uncached_prompt_tokens() <= 1024;
            b_short
                .cmp(&a_short)
                .then_with(|| {
                    seqs[a]
                        .ttft_slack_ms(now_ms)
                        .total_cmp(&seqs[b].ttft_slack_ms(now_ms))
                })
                .then_with(|| seqs[a].ready_ms.total_cmp(&seqs[b].ready_ms))
        });

        let mut remaining_waiting = Vec::with_capacity(waiting.len());
        let mut active_long_prefills = active
            .iter()
            .filter(|&&i| seqs[i].remaining_prefill() > 1024)
            .count();

        for idx in waiting.drain(..) {
            if active.len() >= self.config.max_num_seqs {
                remaining_waiting.push(idx);
                continue;
            }
            let is_long = seqs[idx].req.uncached_prompt_tokens() > 1024;
            // Limit concurrent in-flight 32K prefills to avoid fragmenting KV pool across
            // half-prefilled prompts that cannot complete before hitting watermark.
            if is_long && active_long_prefills >= 4 && !active.is_empty() {
                remaining_waiting.push(idx);
                continue;
            }
            let headroom = if active.is_empty() {
                0
            } else {
                self.active_decode_headroom_blocks(seqs, active)
            };
            let check_tokens = self.initial_admission_tokens(&seqs[idx]);
            let wm = if is_long {
                self.config.kv_high_watermark
            } else {
                (self.config.kv_high_watermark + 0.08).min(0.95)
            };
            let util_ok = active.is_empty() || self.pool.kv_utilization() < wm;
            if util_ok && self.pool.can_admit(check_tokens, headroom) {
                let alloc_tokens = seqs[idx].req.warm_prefix_tokens.saturating_add(1).max(1);
                self.pool.admit_sequence(seqs[idx].req.id, alloc_tokens)?;
                if !seqs[idx].admitted_once {
                    seqs[idx].admitted_once = true;
                    seqs[idx].queue_wait_ms = (now_ms - seqs[idx].ready_ms).max(0.0);
                }
                seqs[idx].state = if seqs[idx].remaining_prefill() > 0 {
                    SequenceState::Prefilling
                } else {
                    SequenceState::Decoding
                };
                if is_long {
                    active_long_prefills += 1;
                }
                active.push(idx);
                admitted_seqs.push(seqs[idx].req.id);
            } else {
                remaining_waiting.push(idx);
            }
        }
        *waiting = remaining_waiting;
        Ok(())
    }

    /// Ensures `target_idx` has enough KV blocks for `target_len`, preempting lower-priority
    /// active victims if needed.
    ///
    /// Already-planned sequences in the current iteration (`prefill_chunks` / `decode_seqs`) are
    /// pinned ahead of unplanned victims; if an already-planned sequence must still be evicted to
    /// guarantee forward progress, its planned entry is rolled back before its KV/GDN state is
    /// freed so no step plan can ever reference a preempted sequence.
    ///
    /// Returns `Ok(true)` iff `target_idx` is resident with `target_len` tokens allocated.
    #[allow(clippy::too_many_arguments)]
    fn ensure_kv_for_step(
        &mut self,
        target_idx: usize,
        target_len: usize,
        seqs: &mut [SequenceRuntime],
        active: &mut Vec<usize>,
        waiting: &mut Vec<usize>,
        swapped: &mut Vec<usize>,
        prefill_chunks: &mut Vec<PrefillChunkSpec>,
        decode_seqs: &mut Vec<usize>,
        preempted_seqs: &mut Vec<usize>,
        total_swap_bytes: &mut u64,
        swap_stall_ms: &mut f64,
    ) -> Result<bool, SchedulerError> {
        let target_id = seqs[target_idx].req.id;
        if !self.pool.is_resident(target_id) {
            return Ok(false);
        }
        while !self.pool.can_grow(target_id, target_len) {
            let is_planned = |id: usize| {
                prefill_chunks.iter().any(|c| c.seq_id == id) || decode_seqs.contains(&id)
            };
            // Prefer evicting unplanned sequences first (prefilling before decoding) so already-
            // planned work in the current iteration stays pinned whenever possible.
            let victim_pos = active
                .iter()
                .rposition(|&i| {
                    i != target_idx
                        && !is_planned(seqs[i].req.id)
                        && seqs[i].state == SequenceState::Prefilling
                })
                .or_else(|| {
                    active
                        .iter()
                        .rposition(|&i| i != target_idx && !is_planned(seqs[i].req.id))
                })
                .or_else(|| {
                    active.iter().rposition(|&i| {
                        i != target_idx && seqs[i].state == SequenceState::Prefilling
                    })
                })
                .or_else(|| active.iter().rposition(|&i| i != target_idx));
            let Some(pos) = victim_pos else {
                // No other sequence to evict; cannot grow further.
                return Ok(false);
            };
            let victim_idx = active.remove(pos);
            let victim_id = seqs[victim_idx].req.id;
            // Roll back any already-planned work for victim_id before freeing its KV/GDN state.
            prefill_chunks.retain(|c| c.seq_id != victim_id);
            decode_seqs.retain(|&id| id != victim_id);

            seqs[victim_idx].preemption_count += 1;
            preempted_seqs.push(victim_id);
            match self.config.preemption_mode {
                PreemptionMode::Recompute => {
                    let recomputed = self.pool.preempt_recompute(victim_id)?;
                    seqs[victim_idx].recomputed_tokens += recomputed;
                    seqs[victim_idx].reset_materialized_for_recompute();
                    waiting.insert(0, victim_idx);
                }
                PreemptionMode::SwapToHost => {
                    let bytes = self.pool.swap_out(victim_id)?;
                    *total_swap_bytes += bytes;
                    *swap_stall_ms += self.cost_model.estimate_swap_ms(bytes);
                    seqs[victim_idx].state = SequenceState::SwappedOut;
                    swapped.push(victim_idx);
                }
            }
        }
        self.pool.allocate_for_len(target_id, target_len)?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn finalize_metrics(
        &self,
        seqs: &[SequenceRuntime],
        total_requests: usize,
        oom_rejected_requests: usize,
        total_steps: usize,
        single_user_fast_path_steps: usize,
        sim_start_ms: f64,
        end_ms: f64,
        total_energy_joules: f64,
        total_swap_bytes: u64,
        peak_unbounded_demand_bytes: u64,
        cpu_scheduler_wall_ms: f64,
    ) -> ScheduleMetrics {
        let mut cold_ttfts = Vec::new();
        let mut warm_ttfts = Vec::new();
        let mut short_ttfts = Vec::new();
        let mut all_ttfts = Vec::new();
        let mut all_itls = Vec::new();
        let mut all_e2es = Vec::new();
        let mut queue_waits = Vec::new();

        let mut completed_requests = 0usize;
        let mut ttft_slo_ok = 0usize;
        let mut itl_slo_ok = 0usize;
        let mut both_slo_ok = 0usize;
        let mut total_preemptions = 0usize;
        let mut total_recomputed_tokens = 0usize;
        let mut total_output_tokens = 0usize;
        let mut total_prefill_tokens = 0usize;

        for s in seqs {
            total_preemptions += s.preemption_count;
            total_recomputed_tokens += s.recomputed_tokens;
            if let (Some(ttft), Some(e2e)) = (s.ttft_ms(), s.e2e_latency_ms()) {
                completed_requests += 1;
                all_ttfts.push(ttft);
                all_e2es.push(e2e);
                queue_waits.push(s.queue_wait_ms);
                if s.req.is_warm() {
                    warm_ttfts.push(ttft);
                } else {
                    cold_ttfts.push(ttft);
                }
                if s.req.class == RequestClass::ShortInteractive {
                    short_ttfts.push(ttft);
                }
                all_itls.extend_from_slice(&s.itl_samples_ms);
                let ok_ttft = ttft <= s.req.slo_ttft_ms;
                let ok_itl = s.itl_samples_ms.is_empty() || s.p95_itl_ms() <= s.req.slo_itl_ms;
                if ok_ttft {
                    ttft_slo_ok += 1;
                }
                if ok_itl {
                    itl_slo_ok += 1;
                }
                if ok_ttft && ok_itl {
                    both_slo_ok += 1;
                }
                total_output_tokens += s.generated_tokens;
                total_prefill_tokens += s.req.uncached_prompt_tokens();
            }
        }

        let makespan_s = ((end_ms - sim_start_ms).max(1e-6)) / 1000.0;
        let output_tok_per_s = total_output_tokens as f64 / makespan_s;
        let total_tokens = total_output_tokens + total_prefill_tokens;
        let total_tok_per_s = total_tokens as f64 / makespan_s;

        let denom = completed_requests.max(1) as f64;
        let slo_attainment_pct = (both_slo_ok as f64 / denom) * 100.0;
        let ttft_slo_pct = (ttft_slo_ok as f64 / denom) * 100.0;
        let itl_slo_pct = (itl_slo_ok as f64 / denom) * 100.0;

        let itl_max_ms = all_itls.iter().copied().fold(0.0f64, f64::max);
        let peak_kv_gdn_mib = self.pool.peak_allocated_bytes() as f64 / (1024.0 * 1024.0);
        let base_vram_bytes =
            self.cost_model.model.weight_bytes_bf16() + self.cost_model.hardware.workspace_bytes;
        let peak_total_vram_gib = (base_vram_bytes + self.pool.peak_allocated_bytes()) as f64
            / (1024.0 * 1024.0 * 1024.0);
        let kv_pool_capacity_gib =
            self.pool.total_capacity_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
        let peak_unbounded_demand_gib =
            peak_unbounded_demand_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        let energy_mj_per_output_tok = if total_output_tokens > 0 {
            (total_energy_joules * 1000.0) / (total_output_tokens as f64)
        } else {
            0.0
        };
        let energy_mj_per_total_tok = if total_tokens > 0 {
            (total_energy_joules * 1000.0) / (total_tokens as f64)
        } else {
            0.0
        };
        let cpu_us_per_step = if total_steps > 0 {
            (cpu_scheduler_wall_ms * 1000.0) / (total_steps as f64)
        } else {
            0.0
        };

        ScheduleMetrics {
            regime_label: self.config.label(),
            profile_label: self.config.profile.as_str(),
            policy_label: self.config.policy.as_str(),
            total_requests,
            completed_requests,
            oom_rejected_requests,
            total_steps,
            single_user_fast_path_steps,
            cold_ttft_p50_ms: percentile(&cold_ttfts, 0.50),
            cold_ttft_p95_ms: percentile(&cold_ttfts, 0.95),
            cold_ttft_mean_ms: mean(&cold_ttfts),
            warm_ttft_p50_ms: percentile(&warm_ttfts, 0.50),
            warm_ttft_p95_ms: percentile(&warm_ttfts, 0.95),
            short_ttft_p95_ms: percentile(&short_ttfts, 0.95),
            overall_ttft_p95_ms: percentile(&all_ttfts, 0.95),
            itl_p50_ms: percentile(&all_itls, 0.50),
            itl_p95_ms: percentile(&all_itls, 0.95),
            itl_p99_ms: percentile(&all_itls, 0.99),
            itl_max_ms,
            makespan_s,
            mean_e2e_ms: mean(&all_e2es),
            output_tok_per_s,
            total_tok_per_s,
            slo_attainment_pct,
            ttft_slo_pct,
            itl_slo_pct,
            total_preemptions,
            total_recomputed_tokens,
            total_swap_bytes,
            mean_queue_wait_ms: mean(&queue_waits),
            peak_kv_gdn_mib,
            peak_total_vram_gib,
            kv_pool_capacity_gib,
            peak_unbounded_demand_gib,
            energy_mj_per_output_tok,
            energy_mj_per_total_tok,
            cpu_scheduler_wall_ms,
            cpu_us_per_step,
            kv_pool_clean: self.pool.is_clean(),
            total_output_tokens,
            total_prefill_tokens,
        }
    }
}

fn release_ready_requests<F>(
    seqs: &mut [SequenceRuntime],
    released: &mut [bool],
    waiting: &mut Vec<usize>,
    now_ms: f64,
    lookup_idx: &F,
) where
    F: Fn(usize) -> Option<usize>,
{
    let mut newly_ready: Vec<usize> = Vec::new();
    for idx in 0..seqs.len() {
        if released[idx] {
            continue;
        }
        let dep_ready_ms = match seqs[idx].req.depends_on {
            None => Some(seqs[idx].req.arrival_ms),
            Some(dep_id) => lookup_idx(dep_id)
                .and_then(|dep_idx| seqs[dep_idx].completion_ms)
                .map(|done_t| done_t.max(seqs[idx].req.arrival_ms)),
        };
        if let Some(ready_t) = dep_ready_ms {
            if ready_t <= now_ms + 1e-9 {
                seqs[idx].ready_ms = ready_t;
                released[idx] = true;
                newly_ready.push(idx);
            }
        }
    }
    // Enqueue newly released requests chronologically by readiness timestamp (tie-breaking by req.id)
    // so FCFS admission never inverts arrival order across multi-arrival forward steps.
    newly_ready.sort_by(|&a, &b| {
        seqs[a]
            .ready_ms
            .total_cmp(&seqs[b].ready_ms)
            .then_with(|| seqs[a].req.id.cmp(&seqs[b].req.id))
    });
    waiting.extend(newly_ready);
}

fn next_unreleased_arrival_ms<F>(
    seqs: &[SequenceRuntime],
    released: &[bool],
    lookup_idx: &F,
) -> Option<f64>
where
    F: Fn(usize) -> Option<usize>,
{
    let mut best = f64::INFINITY;
    for idx in 0..seqs.len() {
        if released[idx] {
            continue;
        }
        let dep_ready_ms = match seqs[idx].req.depends_on {
            None => Some(seqs[idx].req.arrival_ms),
            Some(dep_id) => lookup_idx(dep_id)
                .and_then(|dep_idx| seqs[dep_idx].completion_ms)
                .map(|done_t| done_t.max(seqs[idx].req.arrival_ms)),
        };
        if let Some(t) = dep_ready_ms {
            if t < best {
                best = t;
            }
        }
    }
    if best.is_finite() {
        Some(best)
    } else {
        None
    }
}

fn compute_unbounded_demand_bytes(
    seqs: &[SequenceRuntime],
    waiting: &[usize],
    active: &[usize],
    swapped: &[usize],
    model: &crate::model::HybridModelSpec,
    block_size: usize,
) -> u64 {
    let mut total = 0u64;
    for &idx in waiting.iter().chain(active.iter()).chain(swapped.iter()) {
        if seqs[idx].state != SequenceState::Finished {
            total += model.sequence_state_bytes(seqs[idx].final_context_len(), block_size);
        }
    }
    total
}
