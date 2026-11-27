// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! # `hanzo-cascade-attn` — Hardware-Specialised Tiled Paged, Cascade, and Fused Ragged Attention
//!
//! Clean-room Apache-2.0, zero-dependency, `#![forbid(unsafe_code)]` Rust reference kernels,
//! numerical verifiers, and per-architecture serving scorecards for Hanzo Engine (`HIP-1346`).
//!
//! ## Modules
//! - [`model`]: Canonical `4B-Hybrid` / `9B-Hybrid` / `27B-Hybrid` Attention + GDN/LSA model
//!   geometry with strict separation of resident VRAM weights (`vocab_size * d_model * 2`) vs
//!   per-step streamed weights (`embed_tokens` single-row gather `d_model * 2`).
//! - [`arch`]: Four strictly separated hardware profiles (`HopperH100Sxm5`, `BlackwellB200`,
//!   `GraceBlackwellGb10`, `ConsumerRtx4090`) enforcing zero cross-architecture extrapolation.
//! - [`cascade`]: Pure-Rust tiled online-softmax paged grouped-query attention (GQA) plus
//!   two-level and three-level shared-prefix cascade attention with exact Log-Sum-Exp (LSE)
//!   state merging and byte-accurate KV read accounting.
//! - [`ragged`]: Pure-Rust fused ragged prefill + decode paged attention over unified CSR-like
//!   jagged batch descriptors (`qo_indptr`, `kv_indptr`, `kv_page_indices`, `kv_last_page_len`).
//! - [`hybrid`]: Full hybrid Attention + GDN/LSA recurrent + dense SwiGLU MLP step verifier
//!   directly measuring Amdahl's-law component shares and full-step numerical equivalence.
//! - [`bench`]: Complete 20-turn coding-agent per-architecture benchmark harness,
//!   [`bench::evaluate_end_to_end_gate`] acceptance gate, explicit negative controls, and
//!   deterministic JSON serializer.
//! - [`sha256`]: Clean-room FIPS 180-4 SHA-256 digest generator for verifiable receipts.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod arch;
pub mod bench;
pub mod cascade;
pub mod hybrid;
pub mod model;
pub mod ragged;
pub mod sha256;

pub use arch::{HardwareArch, HardwareProfile};
pub use bench::{
    evaluate_end_to_end_gate, evaluate_serving_scorecard, run_full_benchmark, AgentWorkloadConfig,
    AttentionBackend, BenchmarkReport, GateDecision, NegativeRegimeRow, ServingScorecardRow,
};
pub use cascade::{
    merge_lse_multi_level, merge_lse_states, run_cpu_cascade_verification,
    three_level_cascade_gqa_decode, two_level_cascade_gqa_decode, unshared_paged_gqa_decode,
    AttentionOutput, CpuCascadeBenchRow, CpuCascadeVerificationSuite, KvReadStats, Lcg64,
    PagedKvPool,
};
pub use hybrid::{run_cpu_hybrid_step_verification, CpuHybridStepReport, HybridStepConfig};
pub use model::{HybridModelSpec, QuantFormat};
pub use ragged::{
    fused_ragged_prefill_decode_gqa, run_cpu_ragged_verification,
    separate_prefill_and_decode_paged_gqa, CpuRaggedVerificationReport, RaggedAttentionResult,
    RaggedBatchDescriptor, RaggedDescriptorError,
};
pub use sha256::{sha256_bytes, sha256_hex};
