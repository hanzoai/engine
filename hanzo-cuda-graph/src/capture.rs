// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Breakable piecewise CUDA graph partitioning, capture lifecycle, and bounded LRU cache.
//!
//! Aligns with Hanzo Engine's `CUDA_DECODE_GRAPH_CACHE_CAPACITY = 32` and
//! `capture_cuda_graph_with_output` semantics while adding breakable graph partitioning
//! for hybrid Linear-Attention (GDN/LSA) + Paged-Attention decode and speculative verification.

use std::collections::VecDeque;
use std::fmt;

use crate::bucket::{BucketPolicy, CudaDecodeGraphKey, ShapeBucket};
use crate::lifetime::{
    CaptureEpoch, CapturedBufferRef, GraphMemoryPool, GraphSanitizer, PersistentGraphBuffers,
    SanitizerReport, SanitizerViolation,
};

/// Default bounded LRU cache capacity for captured decode graphs, matching
/// `hanzo-engine/src/pipeline/cuda_graph.rs::CUDA_DECODE_GRAPH_CACHE_CAPACITY`.
pub const CUDA_DECODE_GRAPH_CACHE_CAPACITY: usize = 32;

/// Classification of a piecewise segment inside a breakable hybrid execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphPieceKind {
    /// Static capturable subgraph covering Pre-Attention RMSNorm + GDN/LSA Recurrent Step + SwiGLU MLP.
    LsaRecurrentAndMlp,
    /// Static capturable subgraph covering Paged KV Attention + Speculative Verification logits.
    PagedAttentionAndVerify,
    /// Eager host/dynamic segment executed outside CUDA graph capture at a graph-break boundary.
    HostDynamicFallback,
}

impl GraphPieceKind {
    /// Stable short tag used in [`CudaDecodeGraphKey`].
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::LsaRecurrentAndMlp => "lsa_recurrent_mlp",
            Self::PagedAttentionAndVerify => "paged_attn_verify",
            Self::HostDynamicFallback => "host_dynamic_fallback",
        }
    }
}

/// Explicit reason why a breakable execution plan inserts a graph break or eager fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphBreakReason {
    /// No break required; standard decode runs fully inside captured pieces.
    None,
    /// Dynamic-length prompt prefill exceeds decode bucketing and runs eagerly.
    PrefillDynamicShape,
    /// Host-side speculative rejection sampling and GDN state checkpoint commit/rollback.
    SpeculativeRejectionSampling,
    /// Active batch exceeds maximum configured batch bucket (`> 32`).
    BatchExceedsMaxBucket,
    /// Active context length exceeds maximum configured context bucket (`> 64K`).
    ContextExceedsMaxBucket,
    /// Capture explicitly disabled (e.g., active GPU profiler session).
    CaptureDisabledByProfiler,
}

/// One ordered execution piece inside a [`BreakableExecutionPlan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphPiece {
    /// Segment classification.
    pub kind: GraphPieceKind,
    /// True if this segment is eligible for CUDA graph capture & replay.
    pub capturable: bool,
    /// Reason for a graph break before/around this piece, if any.
    pub break_reason: GraphBreakReason,
}

/// Ordered piecewise execution plan for one hybrid forward iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BreakableExecutionPlan {
    /// Resolved bounded shape bucket (`None` if forced fully eager).
    pub bucket: Option<ShapeBucket>,
    /// Ordered pieces executed to complete the step.
    pub pieces: Vec<GraphPiece>,
    /// Primary graph-break reason (if any).
    pub primary_break: GraphBreakReason,
}

impl BreakableExecutionPlan {
    /// Partitions a hybrid LSA/GDN + Paged-Attention step into capturable static subgraphs
    /// separated by clean graph-break boundaries whenever dynamic shapes or host-side
    /// speculative rejection sampling require eager control flow.
    #[must_use]
    pub fn plan_step(
        policy: &BucketPolicy,
        active_batch: usize,
        active_verify_width: usize,
        active_context: usize,
        is_prefill: bool,
        is_speculative_verify: bool,
        profiler_active: bool,
    ) -> Self {
        if profiler_active {
            return Self::fully_eager(GraphBreakReason::CaptureDisabledByProfiler);
        }
        if is_prefill {
            return Self::fully_eager(GraphBreakReason::PrefillDynamicShape);
        }
        let Some(bucket) = policy.resolve(active_batch, active_verify_width, active_context) else {
            let reason = if policy.max_batch().is_some_and(|m| active_batch > m) {
                GraphBreakReason::BatchExceedsMaxBucket
            } else {
                GraphBreakReason::ContextExceedsMaxBucket
            };
            return Self::fully_eager(reason);
        };

        let mut pieces = vec![
            GraphPiece {
                kind: GraphPieceKind::LsaRecurrentAndMlp,
                capturable: true,
                break_reason: GraphBreakReason::None,
            },
            GraphPiece {
                kind: GraphPieceKind::PagedAttentionAndVerify,
                capturable: true,
                break_reason: GraphBreakReason::None,
            },
        ];

        let primary_break = if is_speculative_verify {
            pieces.push(GraphPiece {
                kind: GraphPieceKind::HostDynamicFallback,
                capturable: false,
                break_reason: GraphBreakReason::SpeculativeRejectionSampling,
            });
            GraphBreakReason::SpeculativeRejectionSampling
        } else {
            GraphBreakReason::None
        };

        Self {
            bucket: Some(bucket),
            pieces,
            primary_break,
        }
    }

    fn fully_eager(reason: GraphBreakReason) -> Self {
        Self {
            bucket: None,
            pieces: vec![GraphPiece {
                kind: GraphPieceKind::HostDynamicFallback,
                capturable: false,
                break_reason: reason,
            }],
            primary_break: reason,
        }
    }

    /// Number of capturable CUDA graph pieces in this plan.
    #[must_use]
    pub fn capturable_piece_count(&self) -> usize {
        self.pieces.iter().filter(|p| p.capturable).count()
    }

    /// Returns true if the step uses CUDA graph capture for its GPU compute pieces.
    #[must_use]
    pub fn uses_graph_capture(&self) -> bool {
        self.capturable_piece_count() > 0
    }
}

/// Errors returned during stream capture or graph replay validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureError {
    /// Dynamic unpooled allocation attempted inside active capture.
    DynamicAllocationForbidden(SanitizerViolation),
    /// Host synchronization attempted inside active capture.
    HostSyncForbidden(SanitizerViolation),
    /// One or more sanitizer violations detected at `end_capture` or before replay.
    SanitizerFailed(Vec<SanitizerViolation>),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DynamicAllocationForbidden(v) => write!(f, "DynamicAllocationForbidden: {v}"),
            Self::HostSyncForbidden(v) => write!(f, "HostSyncForbidden: {v}"),
            Self::SanitizerFailed(vs) => {
                write!(f, "SanitizerFailed ({} violations):", vs.len())?;
                for v in vs {
                    write!(f, " [{v}]")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for CaptureError {}

/// Single kernel node recorded inside a [`CapturedGraph`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedNode {
    /// Sequential node identifier inside the graph.
    pub node_id: usize,
    /// Kernel name (e.g., `"gdn_recurrent_step"`, `"paged_attn_decode"`).
    pub kernel_name: String,
    /// Graph piece kind owning this node.
    pub piece_kind: GraphPieceKind,
    /// Fixed buffer bindings accessed by this kernel on every replay.
    pub bindings: Vec<CapturedBufferRef>,
}

/// Captured CUDA graph artifact tracking pinned buffer lifetimes and private pool bytes.
#[derive(Debug, Clone)]
pub struct CapturedGraph {
    /// Canonical decode-graph cache key.
    pub key: CudaDecodeGraphKey,
    /// Capture epoch in which this graph was instantiated.
    pub epoch: CaptureEpoch,
    /// Recorded kernel nodes and their exact buffer bindings.
    pub nodes: Vec<CapturedNode>,
    /// Private graph pool bytes held by this instantiated graph until teardown.
    pub private_pool_bytes: usize,
    /// Analytical/recorded capture + instantiate + upload duration in milliseconds.
    pub capture_duration_ms: f64,
    /// Total successful replays executed by this graph.
    pub replay_count: u64,
    /// Whether `teardown` has already released refcounts and pool bytes.
    torn_down: bool,
}

impl CapturedGraph {
    /// Captures a graph inside `pool`, retaining buffer refcounts, validating that zero
    /// capture-local scratch leaks past `end_capture`, and accounting for `private_pool_bytes`.
    pub fn capture(
        pool: &mut GraphMemoryPool,
        key: CudaDecodeGraphKey,
        private_pool_bytes: usize,
        capture_duration_ms: f64,
        build_nodes: impl FnOnce(&mut GraphMemoryPool) -> Result<Vec<CapturedNode>, SanitizerViolation>,
    ) -> Result<Self, CaptureError> {
        let epoch = pool.begin_capture().map_err(|v| match v {
            SanitizerViolation::HostSyncDuringCapture { .. } => CaptureError::HostSyncForbidden(v),
            other => CaptureError::SanitizerFailed(vec![other]),
        })?;

        let nodes = match build_nodes(pool) {
            Ok(n) => n,
            Err(v) => {
                pool.abort_capture();
                return Err(match v {
                    SanitizerViolation::DynamicAllocationDuringCapture { .. } => {
                        CaptureError::DynamicAllocationForbidden(v)
                    }
                    SanitizerViolation::HostSyncDuringCapture { .. } => {
                        CaptureError::HostSyncForbidden(v)
                    }
                    other => CaptureError::SanitizerFailed(vec![other]),
                });
            }
        };

        if let Err(leaks) = pool.end_capture() {
            return Err(CaptureError::SanitizerFailed(leaks));
        }

        // Pin unique referenced buffers so freeing them while this graph is alive marks them Poisoned.
        let mut retained = Vec::new();
        for node in &nodes {
            for b in &node.bindings {
                if !retained.contains(&b.buffer_id) {
                    retained.push(b.buffer_id);
                    pool.retain_for_graph(b.buffer_id);
                }
            }
        }
        pool.account_graph_pool_alloc(private_pool_bytes);

        Ok(Self {
            key,
            epoch,
            nodes,
            private_pool_bytes,
            capture_duration_ms,
            replay_count: 0,
            torn_down: false,
        })
    }

    /// Validates all captured node buffer bindings against `pool` via [`GraphSanitizer`]
    /// and increments `replay_count` on clean pass.
    pub fn validate_and_replay(
        &mut self,
        pool: &GraphMemoryPool,
    ) -> Result<SanitizerReport, CaptureError> {
        let view: Vec<(usize, &[CapturedBufferRef])> = self
            .nodes
            .iter()
            .map(|n| (n.node_id, n.bindings.as_slice()))
            .collect();
        let report =
            GraphSanitizer::validate_replay(pool, &view).map_err(CaptureError::SanitizerFailed)?;
        self.replay_count += 1;
        Ok(report)
    }

    /// Explicitly tears down the captured graph, releasing all buffer refcounts and
    /// returning `private_pool_bytes` back to `pool` with zero leaked bytes.
    pub fn teardown(&mut self, pool: &mut GraphMemoryPool) {
        if self.torn_down {
            return;
        }
        self.torn_down = true;
        let mut released = Vec::new();
        for node in &self.nodes {
            for b in &node.bindings {
                if !released.contains(&b.buffer_id) {
                    released.push(b.buffer_id);
                    pool.release_from_graph(b.buffer_id);
                }
            }
        }
        pool.account_graph_pool_free(self.private_pool_bytes);
    }

    /// Returns whether this graph has been cleanly torn down.
    #[must_use]
    pub const fn is_torn_down(&self) -> bool {
        self.torn_down
    }
}

/// Entry stored inside [`DecodeGraphCache`].
#[derive(Debug, Clone)]
pub struct CachedGraphEntry {
    /// Captured graph handle.
    pub graph: CapturedGraph,
    /// Persistent static I/O & metadata buffers bound to this graph.
    pub buffers: PersistentGraphBuffers,
}

/// Telemetry counters for [`DecodeGraphCache`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeGraphCacheStats {
    /// Maximum cached graph entries permitted before LRU eviction.
    pub capacity: usize,
    /// Current number of live cached graphs.
    pub live_entries: usize,
    /// Cache hits (graph replayed without recapture).
    pub hits: u64,
    /// Cache misses (new graph captured).
    pub misses: u64,
    /// LRU evictions cleanly torn down.
    pub evictions: u64,
    /// Cumulative capture duration in milliseconds.
    pub total_capture_ms: f64,
    /// Current private graph pool bytes held by cached graphs.
    pub current_graph_pool_bytes: usize,
    /// Peak private graph pool bytes observed.
    pub peak_graph_pool_bytes: usize,
}

/// Bounded LRU cache of captured decode & verification graphs with clean teardown on eviction.
#[derive(Debug, Clone)]
pub struct DecodeGraphCache {
    capacity: usize,
    /// Front = least-recently-used, Back = most-recently-used.
    entries: VecDeque<(CudaDecodeGraphKey, CachedGraphEntry)>,
    hits: u64,
    misses: u64,
    evictions: u64,
    total_capture_ms: f64,
    current_graph_pool_bytes: usize,
    peak_graph_pool_bytes: usize,
}

impl Default for DecodeGraphCache {
    fn default() -> Self {
        Self::new(CUDA_DECODE_GRAPH_CACHE_CAPACITY)
    }
}

impl DecodeGraphCache {
    /// Creates a bounded LRU decode-graph cache with `capacity` slots (`>= 1`).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: VecDeque::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
            total_capture_ms: 0.0,
            current_graph_pool_bytes: 0,
            peak_graph_pool_bytes: 0,
        }
    }

    /// Looks up `key`, promoting it to most-recently-used on hit.
    pub fn get_mut(&mut self, key: &CudaDecodeGraphKey) -> Option<&mut CachedGraphEntry> {
        let pos = self.entries.iter().position(|(k, _)| k == key)?;
        self.hits += 1;
        if pos + 1 != self.entries.len() {
            if let Some(item) = self.entries.remove(pos) {
                self.entries.push_back(item);
            }
        }
        self.entries.back_mut().map(|(_, entry)| entry)
    }

    /// Inserts a newly captured `(key, graph, buffers)` entry, evicting and cleanly tearing
    /// down the least-recently-used entry if the cache is at capacity.
    pub fn insert(
        &mut self,
        pool: &mut GraphMemoryPool,
        key: CudaDecodeGraphKey,
        graph: CapturedGraph,
        buffers: PersistentGraphBuffers,
    ) -> &mut CachedGraphEntry {
        self.misses += 1;
        self.total_capture_ms += graph.capture_duration_ms;

        // If key already existed, tear down the old entry cleanly first.
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == &key) {
            if let Some((_, mut old)) = self.entries.remove(pos) {
                self.current_graph_pool_bytes = self
                    .current_graph_pool_bytes
                    .saturating_sub(old.graph.private_pool_bytes);
                old.graph.teardown(pool);
                old.buffers.release(pool);
            }
        }

        while self.entries.len() >= self.capacity {
            if let Some((_, mut victim)) = self.entries.pop_front() {
                self.evictions += 1;
                self.current_graph_pool_bytes = self
                    .current_graph_pool_bytes
                    .saturating_sub(victim.graph.private_pool_bytes);
                victim.graph.teardown(pool);
                victim.buffers.release(pool);
            }
        }

        self.current_graph_pool_bytes += graph.private_pool_bytes;
        if self.current_graph_pool_bytes > self.peak_graph_pool_bytes {
            self.peak_graph_pool_bytes = self.current_graph_pool_bytes;
        }
        self.entries
            .push_back((key, CachedGraphEntry { graph, buffers }));
        // Safety: we just pushed an element above.
        &mut self.entries.back_mut().expect("entry just pushed").1
    }

    /// Cleanly tears down and releases all cached graphs and their persistent buffers from `pool`.
    pub fn clear(&mut self, pool: &mut GraphMemoryPool) {
        while let Some((_, mut entry)) = self.entries.pop_front() {
            self.current_graph_pool_bytes = self
                .current_graph_pool_bytes
                .saturating_sub(entry.graph.private_pool_bytes);
            entry.graph.teardown(pool);
            entry.buffers.release(pool);
        }
    }

    /// Returns a snapshot of cache hit/miss/eviction and memory statistics.
    #[must_use]
    pub fn stats(&self) -> DecodeGraphCacheStats {
        DecodeGraphCacheStats {
            capacity: self.capacity,
            live_entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            total_capture_ms: self.total_capture_ms,
            current_graph_pool_bytes: self.current_graph_pool_bytes,
            peak_graph_pool_bytes: self.peak_graph_pool_bytes,
        }
    }
}
