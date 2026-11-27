//! Structured error types and canonical error codes shared across independent
//! Rust and Python `hanzo-attest` implementations.

use std::fmt;

/// Canonical error codes produced identically by both the Rust and Python
/// attestation verification and encrypted request binding clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// Hardware root-of-trust identifier is not present in the client's trust store.
    UntrustedRoot,
    /// Hardware root-of-trust identifier appears on the client's revocation list.
    RootRevoked,
    /// Hardware root-of-trust certificate is expired or not yet valid.
    RootExpired,
    /// TEE platform mismatch across root certificate, platform certificate, quote, or policy.
    PlatformMismatch,
    /// Ed25519 signature by the hardware root over the platform certificate failed verification.
    InvalidPlatformCertSignature,
    /// Platform endorsement certificate identifier has been revoked.
    PlatformCertRevoked,
    /// Underlying hardware device identifier has been revoked.
    DeviceRevoked,
    /// Platform endorsement certificate is past its expiration timestamp.
    PlatformCertExpired,
    /// Platform endorsement certificate `issued_at_unix` is in the future beyond clock skew.
    PlatformCertNotYetValid,
    /// Quote `platform_cert_id` does not match the attached platform certificate.
    QuoteCertIdMismatch,
    /// Ed25519 signature by the platform certificate key over the attestation quote failed.
    InvalidQuoteSignature,
    /// Enclave or TEE reported debug mode enabled (`debug_disabled == false`).
    DebugModeEnabled,
    /// Host or TEE reported secure boot disabled (`secure_boot == false`).
    SecureBootDisabled,
    /// Confidential GPU / TEE memory encryption is disabled (`memory_encryption == false`).
    MemoryEncryptionDisabled,
    /// Reported TCB version is below the policy minimum floor for the TEE platform.
    TcbVersionTooLow,
    /// Quote TCB version disagrees with the platform endorsement certificate TCB version.
    TcbCertMismatch,
    /// Reported RTMR kernel, RTMR runtime, or enclave code digest is not allowlisted.
    MeasurementMismatch,
    /// Quote `model_id` does not match the policy's expected model identifier.
    ModelIdMismatch,
    /// Quote `model_weights_sha256` does not match the expected model commitment.
    ModelWeightsMismatch,
    /// Quote `tokenizer_sha256` does not match the expected tokenizer commitment.
    TokenizerMismatch,
    /// Quote `workload_policy_hash` does not match the expected workload policy commitment.
    WorkloadPolicyMismatch,
    /// Quote `client_nonce` does not match the caller's fresh challenge nonce.
    NonceMismatch,
    /// Quote `client_nonce` was already consumed in the client's replay cache.
    NonceReplayed,
    /// Attestation quote is past its `expires_at_unix` deadline.
    QuoteExpired,
    /// Attestation quote `issued_at_unix` is in the future beyond allowed clock skew.
    QuoteNotYetValid,
    /// Attestation quote age exceeds the policy's maximum quote lifetime.
    QuoteTooOld,
    /// Attestation quote `revocation_epoch` is older than the client's minimum epoch.
    RevocationEpochStale,
    /// Quote `report_data` does not match `SHA256(domain || nonce || x25519_pub || model_commit || workload_policy)`.
    ReportDataMismatch,
    /// Ephemeral X25519 public key is low-order / produces an all-zero shared secret.
    InvalidEphemeralKey,
    /// ChaCha20-Poly1305 AEAD authentication tag verification failed.
    AeadAuthFailed,
    /// Request envelope binding header disagrees with expected quote/model/workload/nonce.
    BindingHeaderMismatch,
    /// Session quota or lifetime has been exhausted; caller must re-attest.
    SessionExpired,
    /// Malformed JSON structure or missing/invalid field.
    InvalidJson,
    /// Invalid hexadecimal or binary encoding length.
    InvalidEncoding,
}

impl ErrorCode {
    /// Returns the stable, uppercase ASCII code shared across Rust and Python clients.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UntrustedRoot => "UNTRUSTED_ROOT",
            Self::RootRevoked => "ROOT_REVOKED",
            Self::RootExpired => "ROOT_EXPIRED",
            Self::PlatformMismatch => "PLATFORM_MISMATCH",
            Self::InvalidPlatformCertSignature => "INVALID_PLATFORM_CERT_SIGNATURE",
            Self::PlatformCertRevoked => "PLATFORM_CERT_REVOKED",
            Self::DeviceRevoked => "DEVICE_REVOKED",
            Self::PlatformCertExpired => "PLATFORM_CERT_EXPIRED",
            Self::PlatformCertNotYetValid => "PLATFORM_CERT_NOT_YET_VALID",
            Self::QuoteCertIdMismatch => "QUOTE_CERT_ID_MISMATCH",
            Self::InvalidQuoteSignature => "INVALID_QUOTE_SIGNATURE",
            Self::DebugModeEnabled => "DEBUG_MODE_ENABLED",
            Self::SecureBootDisabled => "SECURE_BOOT_DISABLED",
            Self::MemoryEncryptionDisabled => "MEMORY_ENCRYPTION_DISABLED",
            Self::TcbVersionTooLow => "TCB_VERSION_TOO_LOW",
            Self::TcbCertMismatch => "TCB_CERT_MISMATCH",
            Self::MeasurementMismatch => "MEASUREMENT_MISMATCH",
            Self::ModelIdMismatch => "MODEL_ID_MISMATCH",
            Self::ModelWeightsMismatch => "MODEL_WEIGHTS_MISMATCH",
            Self::TokenizerMismatch => "TOKENIZER_MISMATCH",
            Self::WorkloadPolicyMismatch => "WORKLOAD_POLICY_MISMATCH",
            Self::NonceMismatch => "NONCE_MISMATCH",
            Self::NonceReplayed => "NONCE_REPLAYED",
            Self::QuoteExpired => "QUOTE_EXPIRED",
            Self::QuoteNotYetValid => "QUOTE_NOT_YET_VALID",
            Self::QuoteTooOld => "QUOTE_TOO_OLD",
            Self::RevocationEpochStale => "REVOCATION_EPOCH_STALE",
            Self::ReportDataMismatch => "REPORT_DATA_MISMATCH",
            Self::InvalidEphemeralKey => "INVALID_EPHEMERAL_KEY",
            Self::AeadAuthFailed => "AEAD_AUTH_FAILED",
            Self::BindingHeaderMismatch => "BINDING_HEADER_MISMATCH",
            Self::SessionExpired => "SESSION_EXPIRED",
            Self::InvalidJson => "INVALID_JSON",
            Self::InvalidEncoding => "INVALID_ENCODING",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Detailed attestation verification, cryptography, or codec error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestError {
    /// Stable cross-language error code.
    pub code: ErrorCode,
    /// Human-readable diagnostic message.
    pub message: String,
}

impl AttestError {
    /// Constructs a new [`AttestError`] with the given [`ErrorCode`] and message.
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for AttestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for AttestError {}

/// Convenience result alias for `hanzo-attest` operations.
pub type Result<T> = std::result::Result<T, AttestError>;
