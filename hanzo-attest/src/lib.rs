//! # `hanzo-attest` — Provider-Independent Confidential GPU Inference Attestation Client
//!
//! Pure-Rust, zero-dependency (`#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`,
//! Apache-2.0) remote attestation verifier and authenticated request binder for
//! confidential GPU inference (`hanzoai/engine#69`, HIP-1350).
//!
//! ## Threat Model Honesty
//!
//! **Trusted execution is not a trustless proof.** Remote hardware attestation
//! relies on:
//! 1. Silicon vendor manufacturing roots of trust (e.g. NVIDIA Confidential
//!    Computing H100/Blackwell, AMD SEV-SNP, Intel TDX),
//! 2. Firmware/microcode TCB integrity and timely revocation updates, and
//! 3. Physical and side-channel resistance of the host deployment.
//!
//! `hanzo-attest` explicitly models these hardware trust assumptions locally
//! inside a client-sovereign [`TrustStore`], [`VerificationPolicy`], and
//! [`RevocationList`] — with **zero mandatory Hanzo control-plane calls** — and
//! binds every sensitive prompt to the verified attestation quote, model weights
//! commitment, workload policy, and enclave ephemeral X25519 key via
//! X25519 ECDH + HKDF-SHA256 + ChaCha20-Poly1305 AEAD.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod attestation;
pub mod channel;
pub mod crypto;
pub mod error;
pub mod json;
pub mod policy;

pub use attestation::{
    compute_report_data, verify_attestation, AttestationBundle, AttestationQuote, EnvironmentState,
    PlatformCert, VerifiedAttestation,
};
pub use channel::{
    bind_and_encrypt_request, canonical_binding_header, derive_session_key, AttestedSession,
    EnclaveChannel, EncryptedBoundRequest, BOUND_REQUEST_PROTOCOL_VERSION,
};
pub use error::{AttestError, ErrorCode, Result};
pub use policy::{
    HardwareRootCert, ModelCommitment, NonceReplayCache, RevocationList, SoftwareMeasurement,
    SovereignBundle, TeePlatform, TrustStore, VerificationPolicy,
};
