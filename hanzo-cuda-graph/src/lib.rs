// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Clean-room breakable CUDA graph capture, memory-lifetime verification, bounded shape
//! bucketing, and graph-compatible speculative verification for Hanzo Engine's hybrid
//! Linear-Attention (GDN/LSA) + Paged-Attention decode path (HIP-1346).
//!
//! ## Architecture Overview
//!
//! 1. **Graph Memory-Lifetime & Pool Sanitizer ([`lifetime`])**:
//!    Enforces hard GPU graph memory-lifetime invariants before every graph replay:
//!    - Zero use-after-free or dangling buffer references across captured nodes
//!    - Monotonic buffer generation tracking (`BufferGeneration`) to detect reallocated slots
//!    - Zero dynamic unpooled allocations or host synchronizations inside active stream capture
//!    - Persistent fixed-address input/output and metadata buffers (`PersistentGraphBuffers`)
//!      updated strictly in-place via `copy_into_persistent`, matching Hanzo Engine's
//!      `CudaDecodeGraphMetadataBuffers` and block-table signature skip semantics
//!    - Explicit refcounted graph teardown and LRU eviction with zero leaked pool bytes
//!    - Address-range write-alias race detection between GDN recurrent state write-back and
//!      attention KV scratch regions
//!
//! 2. **Bounded Shape Bucketing & Breakable Graph Capture ([`bucket`], [`capture`])**:
//!    - Deterministic bounded shape buckets for batch size (`B ∈ {1, 2, 4, 8, 16, 32}`),
//!      speculative verification width (`K ∈ {1, 2, 4, 8}`), and paged KV context length
//!      (`8K, 16K, 30K, 64K`)
//!    - Breakable piecewise graph partitioning separating static capturable subgraphs
//!      (`LsaRecurrentAndMlp` and `PagedAttentionAndVerify`) from dynamic host boundaries
//!      (unbucketed prefill, speculative rejection sampling, or out-of-range shapes)
//!    - Bounded LRU decode-graph cache (`CUDA_DECODE_GRAPH_CACHE_CAPACITY = 32`) with exact
//!      capture overhead and private pool VRAM tracking
//!
//! 3. **Executable Pure-Rust Hybrid LSA/GDN Decode & Speculative Verification ([`lsa_exec`])**:
//!    - Deterministic pure-Rust hybrid Gated DeltaNet (GDN/LSA) recurrent state update
//!      (`S_t = alpha_t * S_{t-1} + beta_t * (k_t outer (v_t - S_{t-1}^T k_t))`) + Paged
//!      Attention forward engine
//!    - Runs eager, graph-captured bucketed decode, and `K`-token speculative verification
//!      with pre-replay GDN state checkpointing and bit-exact rollback on draft rejection
//!
//! 4. **HIP-1346 Benchmark & End-to-End Scorecard ([`model`], [`bench`])**:
//!    - Exact shared `4B-Hybrid` / `9B-Hybrid` / `27B-Hybrid` matrix (`num_kv_heads = 2, 4, 4`,
//!      `gdn_state_mib = 27.0, 45.0, 192.0`)
//!    - Strict honesty separation between `cpu_directly_measured` verification/sanitizer results
//!      and `analytical_gpu_h100_sxm5` roofline projections across `8K / 16K / 30K / 64K`
//!      contexts and the 20-turn coding-agent scorecard, including three explicit negative regimes.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod bench;
pub mod bucket;
pub mod capture;
pub mod lifetime;
pub mod lsa_exec;
pub mod model;
pub mod sha256;

pub use bench::{
    evaluate_end_to_end_gate, run_full_benchmark, BenchmarkReport, ContextSweepRow,
    EndToEndGateDecision, NegativeRegimeRow, ScorecardRow,
};
pub use bucket::{
    BucketPolicy, CudaDecodeGraphKey, ShapeBucket, DEFAULT_BATCH_BUCKETS, DEFAULT_CONTEXT_BUCKETS,
    DEFAULT_VERIFY_BUCKETS,
};
pub use capture::{
    BreakableExecutionPlan, CaptureError, CapturedGraph, CapturedNode, DecodeGraphCache,
    GraphBreakReason, GraphPiece, GraphPieceKind, CUDA_DECODE_GRAPH_CACHE_CAPACITY,
};
pub use lifetime::{
    BufferAccessKind, BufferGeneration, BufferId, CaptureEpoch, CapturedBufferRef, DeviceLocation,
    GraphMemoryPool, GraphSanitizer, PersistentGraphBuffers, PoolBlockState, SanitizerReport,
    SanitizerViolation,
};
pub use lsa_exec::{
    HybridLsaConfig, HybridLsaEngine, HybridLsaState, SanitizerFaultSummary, SpeculativeVerifyOut,
};
pub use model::{HardwareSpec, HybridModelSpec, QuantFormat};
pub use sha256::{sha256_bytes, sha256_hex};
