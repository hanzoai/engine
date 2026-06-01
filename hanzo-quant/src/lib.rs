//! Pure-Rust port of `zen/gym/src/gym/quantization/` (Python -> Rust).
//!
//! Three primitives, one trait:
//!
//! - [`bitdelta`]   — 1-bit deltas (~10x compression). Per-tensor scale + packed sign bits.
//! - [`deltaquant`] — INT2/INT4/INT8 grouped symmetric quantization (default group_size=128).
//! - [`deltasoup`]  — Byzantine-robust aggregation: Mean, Median, TrimmedMean, Krum, MultiKrum.
//!
//! The [`unified::Quantize`] trait gives a single dispatch surface across backends.
//!
//! # Invariants matched against the Python originals
//!
//! - BitDelta uses a **single per-tensor scale** = mean(|delta|), not per-group/per-channel.
//!   Signs are packed 8 bits per byte, little-endian (bit 0 = element 0).
//! - DeltaQuant INT4 uses **group_size=128** by default and **symmetric** per-group quant:
//!   `scale = max(|x|)/7`, two int4s packed per byte (low nibble = even idx).
//! - Trim-mean: for N >= 4, sort each coordinate across workers, drop the top and bottom
//!   element, mean the rest. For N < 4, fall back to plain mean.
//!
//! Reputation / DP / reward distribution are **out of scope for v1** — only the
//! aggregation math is ported. Filed as future work.

pub mod bitdelta;
pub mod deltaquant;
pub mod deltasoup;
pub mod unified;

pub use bitdelta::BitDelta;
pub use deltaquant::{Bits, DeltaQuant};
pub use deltasoup::{aggregate, Method};
pub use unified::{Backend, Quantize, QuantizedDelta};

/// Public error type for the crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("tensor op: {0}")]
    Tensor(#[from] candle_core::Error),
    #[error("shape mismatch: base={base:?} vs weight={weight:?}")]
    ShapeMismatch { base: Vec<usize>, weight: Vec<usize> },
    #[error("invalid bit width: {0} (expected 2, 4, or 8)")]
    InvalidBits(u8),
    #[error("empty input: {0}")]
    Empty(&'static str),
    #[error("aggregation: need >= {needed} deltas, got {got}")]
    NotEnoughDeltas { needed: usize, got: usize },
    #[error("serialization: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
