//! Clean-room Apache-2.0 permissionless internet-scale elastic low-communication
//! LoRA training runtime for Hanzo Engine (`hanzo.train.elastic/v1`, HIP-1350).
//!
//! # Architecture
//! - **Two-level low-communication optimization** ([`optimizer`]): each worker
//!   runs `H` independent inner AdamW updates over its local grounded corpus
//!   shard with zero cross-WAN traffic, then transmits an INT8 error-feedback
//!   compressed LoRA pseudo-gradient `delta_k = theta^(t) - theta_k^(t, H)`
//!   aggregated via Byzantine-robust coordinate-wise trimmed mean and outer
//!   Nesterov momentum.
//! - **Executable small Zen causal model + LoRA adapters** ([`model`], [`lora`]):
//!   exact forward evaluation and analytical backpropagation verified against
//!   central finite differences, trained directly on the SHA-256 verified
//!   grounded repository corpus ([`corpus`]).
//! - **Live multi-socket TCP cluster & VFS checkpoint recovery** ([`transport`],
//!   [`worker`], [`cluster`], [`checkpoint`]): independent worker nodes bound
//!   on live loopback TCP sockets with simulated inter-region WAN latency and
//!   bandwidth shaping, abrupt mid-run `hard_kill()` churn survival, and live
//!   TCP checkpoint recovery from content-addressed `vfs://checkpoints/<sha256>`
//!   snapshots with `hanzo_coordinator_required = false`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod bench;
pub mod checkpoint;
pub mod cluster;
pub mod corpus;
pub mod lora;
pub mod model;
pub mod optimizer;
pub mod sha256;
pub mod transport;
pub mod worker;
