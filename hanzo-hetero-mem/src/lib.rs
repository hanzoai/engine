//! Clean-room Apache-2.0 heterogeneous CPU/GPU memory offload, PowerInfer hot/cold
//! neuron & expert partitioning, zero-copy unified memory views, compressed 4 KiB-paged
//! weight residency, and asynchronous lookahead prefetch for hybrid Transformer + Gated
//! DeltaNet (GDN) inference (`hanzoai/engine#57`, HIP-1346).
//!
//! # Key Properties
//! - Zero external crate dependencies (`[dependencies]` empty)
//! - `#![forbid(unsafe_code)]` and `#![deny(missing_docs)]`
//! - Exact HIP-1346 `4B-Hybrid` / `9B-Hybrid` / `27B-Hybrid` model matrix matching
//!   `hanzo-paged-attn/tests/kivi_kv_quant.rs:450-478`
//! - Explicit separation of directly measured host CPU/file I/O metrics, exact unclamped
//!   memory footprints, and analytical discrete/unified hardware projections

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod bench;
pub mod corpus;
pub mod hardware;
pub mod mmap_store;
pub mod model;
pub mod partition;
pub mod sha256;
pub mod verify;
