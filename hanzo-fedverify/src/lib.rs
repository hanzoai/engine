// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! # `hanzo-fedverify` — Verifiable Federated Updates & Byzantine-Robust Training (`hanzoai/engine#73`)
//!
//! Clean-room Apache-2.0 workspace crate with **zero external crate dependencies**,
//! `#![forbid(unsafe_code)]`, and `#![deny(missing_docs)]` implementing HIP-1350
//! verifiable federated learning:
//!
//! - **Cryptographic Model-Update Identity & Attribution ([`crypto`], [`identity`])**:
//!   Pure-Rust FIPS 180-4 SHA-256/SHA-512 and RFC 8032 Ed25519 binding
//!   `(round_id, participant_id, base_model_hash, update_kind, delta_hash, claimed_loss_delta, sample_count, nonce)`
//!   into a [`identity::SignedUpdateEnvelope`], plus Ed25519-signed [`identity::RejectionReceipt`]s
//!   for every rejected update.
//! - **Independent Quality Evaluation & Multi-Layer Byzantine Defense ([`model`], [`aggregator`])**:
//!   Real deterministic language model trained on pinned versioned `hanzoai/engine`
//!   repository text (`calibration_data/calibration_datav3_small.txt`,
//!   `chat_templates/llama3.json`, `orderings/xlora-paper-ordering.json`) with
//!   strictly held-out validation & test splits and 4 matched aggregation regimes:
//!   `naive_fedavg_baseline`, `trimmed_mean_krum`, `hanzo_fedverify_dso`, and
//!   `oracle_honest_only_ceiling`.
//! - **Sovereign Audit Ledger Export & Offline Replay ([`ledger`])**:
//!   Lossless `FVER` v1 binary wire codec, JSON export, and independent third-party
//!   verification requiring **0 Hanzo control-plane calls**.
//! - **Multi-Participant Adversarial Simulation Harness ([`simulation`])**:
//!   Deterministic simulations across `>= 24` participants covering sign-flipping,
//!   `100×` norm scaling, coordinated Sybil collusion cartels, fake claimed improvements,
//!   cryptographic tamper/replay, super-majority (`>= 50%`) Byzantine stress, and
//!   extreme Non-IID honest heterogeneity.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod aggregator;
pub mod crypto;
pub mod identity;
pub mod ledger;
pub mod model;
pub mod simulation;
