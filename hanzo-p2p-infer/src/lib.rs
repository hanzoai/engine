// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! # `hanzo-p2p-infer`
//!
//! Clean-room, zero-dependency pure-Rust peer-to-peer distributed LLM inference runtime
//! and HIP-1350 benchmark suite for Hanzo Engine.
//!
//! ## Architecture
//! - [`sha256`]: FIPS 180-4 SHA-256, RFC 2104 HMAC-SHA256, constant-time equality,
//!   and SplitMix64 deterministic PRNG.
//! - [`model`]: Canonical `4B / 9B / 27B` hybrid Transformer + GDN model matrix (with
//!   explicit resident input-embedding vs streamed non-embedding weight accounting) plus
//!   a live executable deterministic Transformer + MoE micro-model with KV caches.
//! - [`transport`]: Length-prefixed binary TCP activation wire protocol (`BF16` lossless
//!   and `INT8` per-tensor quantization), SHA-256 frame integrity commitments, authenticated
//!   HMAC-SHA256 stage receipts, and explicit [`transport::TrustTier`] separation.
//! - [`partition`]: Capability-aware heterogeneous pipeline partitioning, interconnect
//!   modeling (`LocalLanRdma`, `MetroFiber`, `PublicInternet`, `PcieGen4Offload`),
//!   disconnect re-planning, and versioned [`partition::HanzoP2PManifest`] (`v1`).
//! - [`router`]: Network-aware MoE expert placement comparing locality-aware stage-resident
//!   experts against naive cross-WAN all-to-all dispatch.
//! - [`node`]: Live independent loopback TCP worker nodes (`WorkerNode`) with real socket
//!   listeners, per-session KV caches, `hard_kill()` crash teardown, and adversarial fault
//!   injection modes for trust testing.
//! - [`cluster`]: Live multi-node pipeline orchestrator (`P2pPipelineClient`) implementing
//!   automatic OS TCP disconnect detection, failover stage promotion, deterministic KV-cache
//!   replay, cryptographic receipt verification, and spot-check challenge re-execution.
//! - [`bench`]: End-to-end HIP-1350 benchmark runner, grounded corpus verifier, negative
//!   regimes, privacy exposure quantification, and deterministic JSON receipt generator.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod bench;
pub mod cluster;
pub mod model;
pub mod node;
pub mod partition;
pub mod router;
pub mod sha256;
pub mod transport;
