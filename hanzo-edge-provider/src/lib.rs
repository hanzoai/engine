// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! # `hanzo-edge-provider`
//!
//! Clean-room, zero-dependency pure-Rust decentralized edge and mobile inference provider
//! runtime and HIP-1350 benchmark suite for Hanzo Engine.
//!
//! Authorized laptops, Apple Silicon systems, AMD Strix Halo machines, and mobile-class
//! devices contribute bounded open-weight Zen inference work while enforcing strict owner
//! guardrails: user-reserved unified-memory headroom, progressive RC thermal DVFS
//! throttling, battery floor and power caps, availability windows, privacy constraints,
//! accurate cryptographic work metering, and automatic market withdrawal with graceful
//! peer handover.
//!
//! ## Module Layout
//! - [`sha256`]: FIPS 180-4 SHA-256, RFC 2104 HMAC-SHA256, constant-time digest equality,
//!   hex encoding, and deterministic SplitMix64 PRNG.
//! - [`device`]: Canonical `4B / 9B / 27B` hybrid Attention + Gated DeltaNet (GDN) model
//!   matrix (separating resident input-embedding table capacity from per-step single-row
//!   embedding lookup bandwidth) and unified-memory hardware profiles
//!   (`AppleSiliconM4Pro24G`, `AppleSiliconM4Max64G`, `AmdStrixHalo128G`, `MobileClass12G`).
//! - [`thermal`]: First-principles RC thermal network integrator
//!   (`dT/dt = (P - (T - T_amb) / R_th) / C_th`), progressive DVFS throttling states
//!   (`Full -> ThermalThrottle70 -> ThermalThrottle45 -> AutoWithdraw`), battery
//!   state-of-charge tracking, and all four deterministic automatic market withdrawal
//!   guardrails.
//! - [`manifest`]: Versioned `EdgeProviderAdvertisement v1` (`hanzo.edge.provider/v1`),
//!   privacy policy contracts, explicit [`manifest::TrustTier`] separation, and signed
//!   `MarketWithdrawalNotice` notices.
//! - [`meter`]: Deterministic executable open-weight Zen hybrid micro-model, signed
//!   `MeteredWorkReceipt` generation, independent receipt/fraud verification
//!   (`7/7` adversarial fault classes), and spot-check re-execution.
//! - [`runtime`]: Length-prefixed binary TCP peer wire protocol (`HZED` v1) and live
//!   independent loopback TCP `EdgeProviderNode` servers with live thermal/power telemetry,
//!   progressive DVFS pacing, automatic market withdrawal, and adversarial fault injection.
//! - [`market`]: Independent decentralized `PeerMarketRouter` (zero Hanzo-controlled
//!   coordinator dependency) implementing admission residency checks, privacy- and
//!   thermal-aware peer routing, live withdrawal handling, and seamless peer handover.
//! - [`bench`]: End-to-end HIP-1350 benchmark runner, grounded corpus verifier, negative
//!   regime evaluator, 5-axis roofline/economics comparison, and deterministic JSON
//!   receipt emitter.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod bench;
pub mod device;
pub mod manifest;
pub mod market;
pub mod meter;
pub mod runtime;
pub mod sha256;
pub mod thermal;
