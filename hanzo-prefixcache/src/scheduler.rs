// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Prefix-aware multi-worker request scheduler with strict compatibility filtering
//! and hot-spot load-imbalance protection.

use crate::error::PrefixCacheError;
use crate::key::{CompatibilityProfile, ScopedTokenBlock, TenantAuthorization};
use crate::radix::PrefixMatchResult;
use crate::tier::TieredPrefixStore;

/// Scheduling policy weights and load-bound parameters.
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Utility weight for each token matched in Tier 0 (Hot HBM).
    pub tier0_token_weight: f64,
    /// Utility weight for each token matched in Tier 1 (Warm DRAM/SSD).
    pub tier1_token_weight: f64,
    /// Utility weight for each token matched in Tier 2 (Cold Shared Object Store).
    pub tier2_token_weight: f64,
    /// Penalty weight applied to `active_requests / max_concurrent_requests`.
    pub load_penalty_weight: f64,
    /// Hard upper bound on `worker.active_requests - min_compatible_active_requests`.
    /// Prevents routing every turn of a hot session onto one overloaded worker.
    pub max_load_imbalance: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            tier0_token_weight: 1.00,
            tier1_token_weight: 0.72,
            tier2_token_weight: 0.35,
            load_penalty_weight: 512.0,
            max_load_imbalance: 2,
        }
    }
}

/// A single inference worker participating in prefix-aware routing.
#[derive(Debug)]
pub struct WorkerNode {
    /// Unique worker identifier (e.g., `"worker-0"`).
    pub worker_id: String,
    /// Active model/tokenizer/quant/RoPE/runtime profile loaded on this worker.
    pub compat: CompatibilityProfile,
    /// Currently active in-flight requests on this worker.
    pub active_requests: u32,
    /// Hard concurrency ceiling for this worker.
    pub max_concurrent_requests: u32,
    /// Currently queued prefill tokens on this worker.
    pub queued_tokens: usize,
    /// Worker-local tiered prefix cache (local Tier 0 + Tier 1, shared Tier 2).
    pub store: TieredPrefixStore,
}

impl WorkerNode {
    /// Create a new worker node with its compatibility profile and tiered store.
    #[must_use]
    pub fn new(
        worker_id: impl Into<String>,
        compat: CompatibilityProfile,
        max_concurrent_requests: u32,
        store: TieredPrefixStore,
    ) -> Self {
        Self {
            worker_id: worker_id.into(),
            compat,
            active_requests: 0,
            max_concurrent_requests,
            queued_tokens: 0,
            store,
        }
    }

    /// Fraction of request capacity currently occupied (`0.0..=1.0`).
    #[must_use]
    pub fn load_ratio(&self) -> f64 {
        if self.max_concurrent_requests == 0 {
            return 1.0;
        }
        f64::from(self.active_requests) / f64::from(self.max_concurrent_requests)
    }

    /// Returns `true` if this worker has spare request concurrency capacity.
    #[must_use]
    pub const fn has_capacity(&self) -> bool {
        self.active_requests < self.max_concurrent_requests
    }
}

/// Candidate evaluation score produced during prefix-aware routing.
#[derive(Clone, Debug)]
pub struct WorkerRouteDecision {
    /// Index of the selected worker inside [`PrefixAwareScheduler`].
    pub worker_index: usize,
    /// Identifier of the selected worker.
    pub worker_id: String,
    /// Prefix match found on the selected worker prior to execution.
    pub prefix_match: PrefixMatchResult,
    /// Composite routing score (higher is better).
    pub score: f64,
    /// `true` if the highest-prefix worker was bypassed due to load cap/imbalance protection.
    pub load_guard_rerouted: bool,
}

/// Prefix-aware scheduler coordinating a pool of inference workers.
#[derive(Debug)]
pub struct PrefixAwareScheduler {
    config: SchedulerConfig,
    workers: Vec<WorkerNode>,
}

impl PrefixAwareScheduler {
    /// Construct a new prefix-aware scheduler over `workers`.
    #[must_use]
    pub fn new(config: SchedulerConfig, workers: Vec<WorkerNode>) -> Self {
        Self { config, workers }
    }

    /// Immutable slice of workers managed by this scheduler.
    #[must_use]
    pub fn workers(&self) -> &[WorkerNode] {
        &self.workers
    }

    /// Mutable slice of workers managed by this scheduler.
    pub fn workers_mut(&mut self) -> &mut [WorkerNode] {
        &mut self.workers
    }

    /// Route an incoming multi-turn request to the best compatible, under-capacity worker.
    ///
    /// Enforces:
    /// 1. Exact 5-field compatibility profile matching (`worker.compat == compat`).
    /// 2. Full tenant read authorization on every matched prefix block.
    /// 3. Hard capacity ceiling (`active_requests < max_concurrent_requests`) and
    ///    hot-spot load-imbalance protection (`active_requests <= min_active + max_load_imbalance`).
    pub fn route(
        &mut self,
        compat: &CompatibilityProfile,
        auth: &TenantAuthorization,
        blocks: &[ScopedTokenBlock<'_>],
    ) -> Result<WorkerRouteDecision, PrefixCacheError> {
        let target_compat = compat.canonical_digest();
        let mut eligible_indices = Vec::new();
        for (idx, w) in self.workers.iter().enumerate() {
            if w.compat.canonical_digest().ct_eq(&target_compat) && w.has_capacity() {
                eligible_indices.push(idx);
            }
        }
        if eligible_indices.is_empty() {
            return Err(PrefixCacheError::NoEligibleWorker {
                reason: "no compatible worker has remaining concurrency capacity".to_string(),
            });
        }

        let min_active = eligible_indices
            .iter()
            .map(|&idx| self.workers[idx].active_requests)
            .min()
            .unwrap_or(0);

        let block_tokens = compat.runtime_config.block_size as f64;
        let mut raw_best_prefix_idx: Option<usize> = None;
        let mut raw_best_prefix_tokens: usize = 0;

        let mut best_decision: Option<WorkerRouteDecision> = None;

        for &idx in &eligible_indices {
            let match_res = self.workers[idx].store.match_prefix(compat, auth, blocks)?;
            let matched = match_res.matched_tokens;
            if matched > raw_best_prefix_tokens {
                raw_best_prefix_tokens = matched;
                raw_best_prefix_idx = Some(idx);
            }

            let active = self.workers[idx].active_requests;
            if active.saturating_sub(min_active) > self.config.max_load_imbalance {
                // Skip hot-spotted worker to preserve cluster-wide tail latency.
                continue;
            }

            let prefix_utility = (match_res.tier0_blocks as f64
                * block_tokens
                * self.config.tier0_token_weight)
                + (match_res.tier1_blocks as f64 * block_tokens * self.config.tier1_token_weight)
                + (match_res.tier2_blocks as f64 * block_tokens * self.config.tier2_token_weight);

            let load_penalty = self.workers[idx].load_ratio() * self.config.load_penalty_weight
                + (self.workers[idx].queued_tokens as f64 * 0.05);
            let score = prefix_utility - load_penalty;

            let replace = match &best_decision {
                None => true,
                Some(prev) => score > prev.score + 1e-9,
            };
            if replace {
                best_decision = Some(WorkerRouteDecision {
                    worker_index: idx,
                    worker_id: self.workers[idx].worker_id.clone(),
                    prefix_match: match_res,
                    score,
                    load_guard_rerouted: false,
                });
            }
        }

        let mut chosen = best_decision.ok_or_else(|| PrefixCacheError::NoEligibleWorker {
            reason: "all compatible workers exceeded load imbalance cap".to_string(),
        })?;

        if let Some(prefix_idx) = raw_best_prefix_idx {
            if prefix_idx != chosen.worker_index
                && self.workers[prefix_idx]
                    .active_requests
                    .saturating_sub(min_active)
                    > self.config.max_load_imbalance
            {
                chosen.load_guard_rerouted = true;
            }
        }
        Ok(chosen)
    }

    /// Increment active request load on `worker_index`.
    pub fn acquire_worker(&mut self, worker_index: usize, queued_tokens: usize) {
        if let Some(w) = self.workers.get_mut(worker_index) {
            w.active_requests = w.active_requests.saturating_add(1);
            w.queued_tokens = w.queued_tokens.saturating_add(queued_tokens);
        }
    }

    /// Decrement active request load on `worker_index`.
    pub fn release_worker(&mut self, worker_index: usize, queued_tokens: usize) {
        if let Some(w) = self.workers.get_mut(worker_index) {
            w.active_requests = w.active_requests.saturating_sub(1);
            w.queued_tokens = w.queued_tokens.saturating_sub(queued_tokens);
        }
    }

    /// Simulate a crash/restart of a single worker (wiping its Tier 0 HBM while preserving
    /// Tier 1 local SSD and Tier 2 shared object store).
    pub fn restart_worker(&mut self, worker_index: usize) -> usize {
        self.workers
            .get_mut(worker_index)
            .map_or(0, |w| w.store.simulate_worker_restart())
    }
}
