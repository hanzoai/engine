//! Vendor-neutral federated-training transport, scheduler, and coordinator.
//!
//! Pure-Rust port of `zen/gym/src/gym/distributed`. The wire format (canonical
//! BF16 delta blob + HMAC-SHA256 auth) is byte-identical to the Python
//! implementation so workers and coordinators in either language interoperate.
//!
//! ## Module map
//!
//! * [`topology`] — `Lab`, `Node`, `NodeRole`, YAML loader with `${VAR}` expansion.
//! * [`scheduler`] — capacity-weighted data sharding and best-fit expert pinning.
//! * [`codec`] — canonical BF16 delta encoder/decoder; byte-identical to Python.
//! * [`auth`] — HMAC-SHA256 over `method|path|ts|sha256(body)`.
//! * [`transport`] — axum HTTP server + reqwest client.
//! * [`coordinator`] — round bookkeeping + DeltaSoup trim-mean aggregation.
//! * [`worker`] — local training loop with `step / push / pull / apply`.
//! * [`selftest`] — round-trip check + coordinator handshake.

#![deny(rust_2018_idioms)]
#![warn(missing_debug_implementations)]

pub mod auth;
pub mod codec;
#[cfg(feature = "compression")]
pub mod codec_bitdelta;
pub mod coordinator;
pub mod scheduler;
pub mod selftest;
pub mod topology;
pub mod transport;
pub mod worker;

pub use coordinator::{Coordinator, CoordinatorState};
pub use scheduler::{Assignment, Scheduler};
pub use topology::{Lab, Node, NodeRole};
pub use transport::client::TransportClient;
pub use transport::server::serve;
pub use worker::{Worker, WorkerConfig};
