#![forbid(unsafe_code)]
//! Clean-room Apache-2.0 Rust implementation of next-generation speculative decoding
//! for `hanzoai/engine#55` (HIP-1346):
//!
//! - **EAGLE-3 feature-level draft head** ([`eagle3`]): fuses multi-layer hidden states
//!   `[h_low; h_mid; h_high]` via learned projection + lightweight causal draft Transformer
//!   trained with multi-step unrolled Training-Time Test (TTT) cross-entropy (zero feature
//!   regression loss; arXiv 2503.01840). Supports both linear chains and top-k beam draft trees.
//! - **Lossless verification & rejection sampling** ([`verify`]): bit-exact greedy linear
//!   and tree verification (100.0% bit-identical to plain decoding), exact Leviathan/Chen
//!   stochastic rejection sampling with residual resampling, and exact multi-branch tree
//!   stochastic verification.
//! - **Zero-compute suffix / prompt lookup** ([`lookup`]): incremental Suffix Automaton (SAM)
//!   with exact most-recent n-gram continuation matching for repetitive code-edit and tool-diff
//!   workloads.
//! - **Workload-aware draft source & verification-shape selector** ([`selector`]): online EWMA
//!   expected-useful-tokens-per-millisecond (`E[useful_tok / ms]`) selector choosing between
//!   `PlainDecode`, `PromptLookup`, `Eagle3Linear`, and `Eagle3Tree` with complexity-preserving
//!   tie-breaking, stand-down, hysteresis, and compute-bound batch-size guards.
//! - **Shared HIP-1346 `4B / 9B / 27B` 20-turn coding-agent scorecard** ([`matrix`], [`bench`]).

pub mod bench;
pub mod corpus;
pub mod eagle3;
pub mod lookup;
pub mod matrix;
pub mod rng;
pub mod selector;
pub mod target;
pub mod verify;
