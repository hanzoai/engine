//! HTTP transport — axum server + reqwest client.
//!
//! Carries canonical BF16 LoRA-delta blobs between workers and coordinator.
//! HTTP intentional: works over any L3 the user has (Thunderbolt 5 networking,
//! mDNS .local, plain Ethernet, Tailscale). The wire format is byte-identical
//! to the Python implementation; HMAC headers `X-Zen-Worker`, `X-Zen-Sig`,
//! `X-Zen-Ts` match transport.py exactly.

pub mod client;
pub mod server;

pub use client::TransportClient;
pub use server::serve;

/// Public read endpoints that bypass auth (matches Python `public` set).
pub const PUBLIC_PATHS: &[&str] = &["/", "/v1/healthz", "/v1/metrics", "/v1/topology"];

/// HTTP header names — kept centralized so the client and server agree.
pub const HDR_WORKER: &str = "x-zen-worker";
pub const HDR_SIG: &str = "x-zen-sig";
pub const HDR_TS: &str = "x-zen-ts";
/// Optional diagnostic hint: `bf16` (default) or `bitdelta`. The body's
/// per-tensor `codec` field is authoritative — this header only helps logging.
pub const HDR_CODEC: &str = "x-zen-codec";
