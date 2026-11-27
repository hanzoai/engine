//! # `hanzo-provider-daemon`
//!
//! Clean-room, permissionless GPU provider daemon for Hanzo Engine (`hanzoai/engine#64`,
//! tracked under HIP-1350).
//!
//! Turns any independently administered machine (CUDA, ROCm, Metal, Vulkan, or CPU)
//! into a verifiable, multi-tenant inference worker with **zero external crate dependencies**,
//! **`#![forbid(unsafe_code)]`**, and **zero mandatory Hanzo-operated infrastructure**:
//!
//! - [`crypto`]: Pure-Rust FIPS 180-4 SHA-256 / SHA-512 and RFC 8032 Ed25519 over Curve25519
//! - [`identity`]: Self-sovereign Ed25519 provider & tenant keypairs + portable `KeyEnvelope`
//! - [`advertise`]: Hardware capability probe, signed `CapabilityAnnouncement`, and permissionless `PeerDirectory`
//! - [`isolation`]: Strict `TenantNamespace` validation, RAII VRAM/concurrency `CapacityPool`, graceful drain, and zeroized per-tenant KV/scratch buffers
//! - [`ledger`]: Signed `CompletionReceipt` and hash-chained JSONL `SovereignLedger` with 100% lossless export & cold-restart replay
//! - [`daemon`]: `ProviderDaemon` quote/lease verification pipeline and pluggable `InferenceBackend` trait
//! - [`rpc`]: Pure `std::net::TcpListener` multi-threaded HTTP/1.1 JSON server & client
//! - [`bench`]: Reproducible HIP-1350 5-axis benchmark, 15-vector adversarial trust suite, and 2 negative-regime controls

#![forbid(unsafe_code)]

pub mod advertise;
pub mod bench;
pub mod crypto;
pub mod daemon;
pub mod identity;
pub mod isolation;
pub mod json;
pub mod ledger;
pub mod rpc;
