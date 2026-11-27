// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! # `hanzo-mixed-quant`
//!
//! Clean-room Apache-2.0 Rust implementation of mixed-precision per-layer quantization
//! (`Ternary 1.58-bit`, `INT4` symmetric & asymmetric, `INT8`, `BF16` IEEE-754 RNE),
//! Activation-Aware Quantization (AWQ), SpinQuant orthogonal rotations (Walsh-Hadamard
//! and learned Cayley-parameterized orthogonal matrices), LLM Compressor sensitivity-driven
//! equal-memory bit allocation + 2:4 semi-structured pruning + Quantization-Aware Distillation
//! (QAD), and the HIP-1346 `4B / 9B / 27B` hybrid 20-turn coding-agent scorecard
//! (`hanzoai/engine#49`).
//!
//! Zero external dependencies (`std` only) and `#![forbid(unsafe_code)]`.

#![forbid(unsafe_code)]

pub mod awq;
pub mod bench;
pub mod compressor;
pub mod corpus;
pub mod error;
pub mod formats;
pub mod matrix;
pub mod model;
pub mod rotation;
pub mod sha256;
