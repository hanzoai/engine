// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! # `hanzo-gcr` — Clean-Room Incremental GPU Checkpoint/Restore (HIP-1346 / engine#91)
//!
//! Pure-Rust (`std`-only, zero external crate dependencies, `#![forbid(unsafe_code)]`)
//! implementation of page-level dirty-tracked incremental GPU checkpointing and restore for
//! paused and preempted LLM inference and fine-tuning jobs.
//!
//! ## Core Components
//! - [`vram`]: Simulated GPU VRAM page-table allocator supporting configurable logical page sizes
//!   (`4 KiB`, `64 KiB`, `2 MiB`), shared content-addressed immutable read-only model weights,
//!   exact lock-free [`vram::AtomicDirtyBitmap`] tracking, and strict per-`(tenant_id, job_id)`
//!   namespace isolation.
//! - [`store`]: Crash-safe content-addressed storage ([`store::FsCheckpointStore`] with temp-file +
//!   `sync_all()` + atomic rename and [`store::MemoryCheckpointStore`]) with CRC32C + SHA-256
//!   verification, chunk garbage collection, and NVMe / Hanzo S3-VFS transfer-time models.
//! - [`checkpoint`]: Unified [`checkpoint::CheckpointEngine`] implementing all three matched
//!   baselines (`full_naive_checkpoint`, `full_mutable_only_checkpoint`, `gcr_incremental_delta`),
//!   bounded chain compaction, parallel/pipelined verified restore, and explicit rejection of
//!   corrupted/truncated chunks, missing parents, cyclic chains, and cross-tenant restores.
//! - [`bench`]: Deterministic, corpus-grounded benchmark harness (`--seed 1346`) covering agent
//!   inference pause/resume, spot-GPU LoRA preemption, 20-pause compaction sweep, and the explicit
//!   100%-dirty negative regime.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bench;
pub mod checkpoint;
pub mod crypto;
pub mod error;
pub mod store;
pub mod vram;

pub use checkpoint::{
    CheckpointConfig, CheckpointEngine, CheckpointManifest, CheckpointMode, CheckpointReceipt,
    PageChunkRef, RestoreReceipt,
};
pub use crypto::{crc32c, sha256, Digest, Sha256};
pub use error::{GcrError, GcrResult};
pub use store::{
    CheckpointStore, FsCheckpointStore, IoStats, MemoryCheckpointStore, StorageTierModel,
};
pub use vram::{
    AtomicDirtyBitmap, GpuMemorySpace, Namespace, PageRegion, PageSize, SharedWeightBase,
};
