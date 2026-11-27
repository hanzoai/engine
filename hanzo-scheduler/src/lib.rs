// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Clean-room Apache-2.0 continuous batching, chunked prefill, paged KV/GDN memory pool,
//! and latency-aware SLO scheduler for hybrid Transformer + Gated DeltaNet (GDN) inference
//! (`hanzoai/engine#56`, HIP-1346).
//!
//! ## Key Capabilities
//! - Zero external crate dependencies (`[dependencies]` empty), `#![forbid(unsafe_code)]`,
//!   and `#![deny(missing_docs)]`.
//! - Exact HIP-1346 `4B / 9B / 27B` hybrid model matrix aligned with
//!   `hanzo-paged-attn/tests/kivi_kv_quant.rs` (`27.0 / 45.0 / 192.0 MiB` FP32 GDN state).
//! - Exact paged KV block + GDN recurrent slot pool ([`kv_pool::PagedKvPool`]) with strict
//!   VRAM OOM protection, recomputation/host-swap preemption, and zero-leak accounting.
//! - Separate operating profiles ([`config::SchedulerProfile`]) and 5 matched scheduling
//!   policies ([`config::SchedulingPolicy`]).
//! - Real deterministic CPU token-level output equivalence verifier ([`verify::HybridMicroModel`])
//!   proving `100%` bit-identical output (`0.000 pp` quality regression) across chunked prefill,
//!   interleaved continuous batching, and preemption + recompute.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod bench;
pub mod config;
pub mod corpus;
pub mod kv_pool;
pub mod model;
pub mod request;
pub mod scheduler;
pub mod sha256;
pub mod verify;

pub use config::{PreemptionMode, SchedulerConfig, SchedulerProfile, SchedulingPolicy, SloTargets};
pub use kv_pool::{PagedKvPool, SchedulerError, SequenceKvTable};
pub use model::{
    HardwareSpec, HybridModelSpec, PrefillChunkCostInput, StepCostEstimate, StepCostModel,
};
pub use request::{InferenceRequest, RequestClass, SequenceRuntime, SequenceState};
pub use scheduler::{
    BatchStepPlan, PrefillChunkSpec, ScheduleMetrics, SchedulerEngine, SimulationOutput,
};
pub use verify::{HybridMicroModel, MicroModelConfig, VerificationReport, VerifySequenceSpec};
