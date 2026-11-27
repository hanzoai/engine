// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! # `hanzo-prefixcache`
//!
//! Persistent, shareable multi-tier prefix cache engine for multi-turn agents (`hanzoai/engine#51`, HIP-1346).
//!
//! ## Architecture
//! - [`crypto`]: Clean-room, zero-dependency FIPS 180-4 SHA-256, IEEE 802.3 CRC32, and constant-time digest equality.
//! - [`key`]: Strict 7-field composite [`key::PrefixCacheKey`] binding model/checkpoint, tokenizer, quantization,
//!   position encoding, runtime configuration, tenant scope ACL, and incremental token prefix chain, with
//!   monotonic scope-boundary enforcement preventing cross-tenant private state exposure.
//! - [`snapshot`]: Versioned binary KV + unquantized `FP32` Gated DeltaNet (GDN) block-boundary checkpoints
//!   with fail-closed header/KV/GDN CRC32 and combined SHA-256 verification.
//! - [`radix`]: Block-aligned RadixAttention prefix index (`arXiv:2312.07104`) with reference-counted pinning
//!   and leaf-first LRU eviction across 3 residency tiers.
//! - [`tier`]: 3-tier persistent storage hierarchy (Tier 0 Hot HBM, Tier 1 Warm local SSD directory with
//!   cold-restart reload, Tier 2 Cold shared object-store directory with bandwidth crossover admission guard).
//! - [`scheduler`]: Prefix-aware multi-worker router balancing deepest hot/warm prefix overlap against
//!   per-worker load ceilings and hot-spot imbalance caps.
//! - [`model`]: Shared `4B / 9B / 27B` hybrid Attention + GDN matrix and transparent analytical latency/energy/crossover model.
//! - [`bench`]: Reproducible 20-turn developer-agent benchmark harness grounded in real repository files.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod bench;
pub mod crypto;
pub mod error;
pub mod key;
pub mod model;
pub mod radix;
pub mod scheduler;
pub mod snapshot;
pub mod tier;
