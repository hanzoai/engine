//! hanzo-zen5 — Rust API for the Zen5 model family.
//!
//! Two implementations sit behind the same [`engine::Zen5Engine`] trait:
//!
//! * [`ffi`] — wraps the vendored zen5-engine C runtime (default). Three
//!   backends: Metal on macOS, CUDA on Linux Spark/H100, CPU elsewhere.
//!   Handles GGUF load, KV cache, MTP speculative decode, snapshot
//!   serialization, sampling. This is the production path today.
//!
//! * [`native`] — pure-Rust scaffold on top of `candle-core` /
//!   `candle-transformers`. Targets DeepSeek V4 Flash (MLA + sparse MoE).
//!   Not feature-complete; gated behind `--features=native` so the crate
//!   builds standalone without a C toolchain.
//!
//! Both implementations satisfy the same trait so hanzod can swap them at
//! runtime once the native backend reaches parity.
//!
//! # Example
//!
//! ```no_run
//! use hanzo_zen5::{engine::{Zen5Engine, GenOpts}, ffi::Engine};
//! use std::path::Path;
//!
//! # tokio_test::block_on(async {
//! let engine = Engine::load(Path::new("zen-5-flash.gguf"), Default::default())?;
//! let mut stream = engine.complete("Hello, ", GenOpts::default()).await?;
//! # Ok::<(), anyhow::Error>(())
//! # });
//! ```

#![deny(rust_2018_idioms)]
#![warn(missing_debug_implementations)]

pub mod engine;

#[cfg(feature = "ffi")]
pub mod ffi;

#[cfg(feature = "native")]
pub mod native;

pub use engine::{GenOpts, ThinkMode, Token, Zen5Engine, Zen5Error};
