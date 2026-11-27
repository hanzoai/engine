//! Provider-independent hardware attestation evidence, certificate chains,
//! canonical preimages, and strict verification producing an unforgeable
//! [`VerifiedAttestation`] capability token.

use std::collections::BTreeMap;

use crate::crypto::{
    ct_eq, ed25519_sign, ed25519_verify, hex_decode_array, hex_encode, sha256,
    x25519_diffie_hellman, Sha256,
};
use crate::error::{AttestError, ErrorCode, Result};
use crate::json::{parse_json, JsonValue};
use crate::policy::{
    ModelCommitment, NonceReplayCache, RevocationList, SoftwareMeasurement, TeePlatform,
    TrustStore, VerificationPolicy,
};

fn put_len_prefixed(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_be_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Computes the canonical 32-byte `report_data` bound inside a signed hardware
/// attestation quote, cryptographically linking the fresh challenge nonce,
/// enclave ephemeral X25519 channel public key, model commitment, and workload policy.
#[must_use]
pub fn compute_report_data(
    client_nonce: &[u8; 32],
    enclave_x25519_pubkey: &[u8; 32],
    model: &ModelCommitment,
) -> [u8; 32] {
    let model_commit_digest = model.model_commitment_digest();
    let mut h = Sha256::new();
    h.update(b"hanzo-attest/report-data/v1");
    h.update(client_nonce);
    h.update(enclave_x25519_pubkey);
    h.update(&model_commit_digest);
    h.update(&model.workload_policy_hash);
    h.finalize()
}

/// Hardware platform / TEE device endorsement certificate signed by a pinned
/// silicon vendor [`crate::policy::HardwareRootCert`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformCert {
    /// Unique identifier for this platform endorsement certificate.
    pub cert_id: String,
    /// Identifier of the signing hardware root certificate.
    pub root_id: String,
    /// TEE platform endorsed by this certificate.
    pub platform: TeePlatform,
    /// Physical device/chip identifier (used for device-level revocation).
    pub device_id: String,
    /// Ed25519 public key used by this platform/TEE to sign attestation quotes.
    pub platform_ed25519_pubkey: [u8; 32],
    /// Endorsed firmware/microcode TCB version.
    pub tcb_version: u64,
    /// Issuance timestamp (Unix seconds).
    pub issued_at_unix: u64,
    /// Expiration timestamp (Unix seconds).
    pub expires_at_unix: u64,
    /// Ed25519 signature by the hardware root over [`Self::signing_preimage`].
    pub root_signature: [u8; 64],
}

impl PlatformCert {
    /// Returns the deterministic domain-separated signing preimage for this certificate.
    #[must_use]
    pub fn signing_preimage(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(192);
        buf.extend_from_slice(b"hanzo-attest/platform-cert/v1\0");
        put_len_prefixed(&mut buf, &self.cert_id);
        put_len_prefixed(&mut buf, &self.root_id);
        put_len_prefixed(&mut buf, self.platform.as_str());
        put_len_prefixed(&mut buf, &self.device_id);
        buf.extend_from_slice(&self.platform_ed25519_pubkey);
        buf.extend_from_slice(&self.tcb_version.to_be_bytes());
        buf.extend_from_slice(&self.issued_at_unix.to_be_bytes());
        buf.extend_from_slice(&self.expires_at_unix.to_be_bytes());
        buf
    }

    /// Issues and signs a [`PlatformCert`] using the hardware root's 32-byte Ed25519 seed.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        cert_id: impl Into<String>,
        root_id: impl Into<String>,
        platform: TeePlatform,
        device_id: impl Into<String>,
        platform_ed25519_pubkey: [u8; 32],
        tcb_version: u64,
        issued_at_unix: u64,
        expires_at_unix: u64,
        root_ed25519_seed: &[u8; 32],
    ) -> Self {
        let mut cert = Self {
            cert_id: cert_id.into(),
            root_id: root_id.into(),
            platform,
            device_id: device_id.into(),
            platform_ed25519_pubkey,
            tcb_version,
            issued_at_unix,
            expires_at_unix,
            root_signature: [0u8; 64],
        };
        let preimage = cert.signing_preimage();
        cert.root_signature = ed25519_sign(root_ed25519_seed, &preimage);
        cert
    }

    /// Serializes this certificate into a [`JsonValue`].
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("cert_id".to_string(), JsonValue::Str(self.cert_id.clone()));
        m.insert("root_id".to_string(), JsonValue::Str(self.root_id.clone()));
        m.insert(
            "platform".to_string(),
            JsonValue::Str(self.platform.as_str().to_string()),
        );
        m.insert(
            "device_id".to_string(),
            JsonValue::Str(self.device_id.clone()),
        );
        m.insert(
            "platform_ed25519_pubkey".to_string(),
            JsonValue::Str(hex_encode(&self.platform_ed25519_pubkey)),
        );
        m.insert("tcb_version".to_string(), JsonValue::Uint(self.tcb_version));
        m.insert(
            "issued_at_unix".to_string(),
            JsonValue::Uint(self.issued_at_unix),
        );
        m.insert(
            "expires_at_unix".to_string(),
            JsonValue::Uint(self.expires_at_unix),
        );
        m.insert(
            "root_signature".to_string(),
            JsonValue::Str(hex_encode(&self.root_signature)),
        );
        JsonValue::Object(m)
    }

    /// Deserializes a [`PlatformCert`] from a [`JsonValue`].
    pub fn from_json_value(v: &JsonValue) -> Result<Self> {
        let obj = v.as_object()?;
        Ok(Self {
            cert_id: JsonValue::req(obj, "cert_id")?.as_str()?.to_string(),
            root_id: JsonValue::req(obj, "root_id")?.as_str()?.to_string(),
            platform: TeePlatform::parse(JsonValue::req(obj, "platform")?.as_str()?)?,
            device_id: JsonValue::req(obj, "device_id")?.as_str()?.to_string(),
            platform_ed25519_pubkey: hex_decode_array(
                JsonValue::req(obj, "platform_ed25519_pubkey")?.as_str()?,
            )?,
            tcb_version: JsonValue::req(obj, "tcb_version")?.as_u64()?,
            issued_at_unix: JsonValue::req(obj, "issued_at_unix")?.as_u64()?,
            expires_at_unix: JsonValue::req(obj, "expires_at_unix")?.as_u64()?,
            root_signature: hex_decode_array(JsonValue::req(obj, "root_signature")?.as_str()?)?,
        })
    }
}

/// Hardware-reported security posture flags inside the attested TEE environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvironmentState {
    /// True iff hardware debugging / JTAG / enclave inspection is permanently disabled.
    pub debug_disabled: bool,
    /// True iff measured hardware secure boot verified the boot chain.
    pub secure_boot: bool,
    /// True iff confidential GPU/CPU memory encryption (e.g. H100 CC / SEV-SNP / TDX) is active.
    pub memory_encryption: bool,
    /// Reported hardware/firmware TCB security version number.
    pub tcb_version: u64,
}

impl EnvironmentState {
    /// Serializes this environment state into a [`JsonValue`].
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "debug_disabled".to_string(),
            JsonValue::Bool(self.debug_disabled),
        );
        m.insert("secure_boot".to_string(), JsonValue::Bool(self.secure_boot));
        m.insert(
            "memory_encryption".to_string(),
            JsonValue::Bool(self.memory_encryption),
        );
        m.insert("tcb_version".to_string(), JsonValue::Uint(self.tcb_version));
        JsonValue::Object(m)
    }

    /// Deserializes an [`EnvironmentState`] from a [`JsonValue`].
    pub fn from_json_value(v: &JsonValue) -> Result<Self> {
        let obj = v.as_object()?;
        Ok(Self {
            debug_disabled: JsonValue::req(obj, "debug_disabled")?.as_bool()?,
            secure_boot: JsonValue::req(obj, "secure_boot")?.as_bool()?,
            memory_encryption: JsonValue::req(obj, "memory_encryption")?.as_bool()?,
            tcb_version: JsonValue::req(obj, "tcb_version")?.as_u64()?,
        })
    }
}

/// Provider-independent remote attestation quote binding hardware environment,
/// measured software stack, model/workload commitment, freshness, and the
/// enclave's ephemeral X25519 channel public key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationQuote {
    /// Unique identifier for this quote instance.
    pub quote_id: String,
    /// TEE hardware platform that generated this quote.
    pub platform: TeePlatform,
    /// Identifier of the signing [`PlatformCert`].
    pub platform_cert_id: String,
    /// Hardware-enforced environment flags and TCB level.
    pub environment: EnvironmentState,
    /// Measured guest kernel, runtime, and enclave binary digests.
    pub measurement: SoftwareMeasurement,
    /// Committed model weights, tokenizer, and workload policy.
    pub model: ModelCommitment,
    /// Fresh 32-byte client challenge nonce echoed by the TEE.
    pub client_nonce: [u8; 32],
    /// Quote issuance timestamp (Unix seconds).
    pub issued_at_unix: u64,
    /// Quote expiration timestamp (Unix seconds).
    pub expires_at_unix: u64,
    /// Monotonic revocation epoch observed when generating this quote.
    pub revocation_epoch: u64,
    /// Ephemeral X25519 public key generated inside the attested enclave.
    pub enclave_x25519_pubkey: [u8; 32],
    /// Hardware report data digest binding nonce, X25519 key, model, and workload policy.
    pub report_data: [u8; 32],
    /// Ed25519 signature by the platform certificate key over [`Self::signing_preimage`].
    pub platform_signature: [u8; 64],
}

impl AttestationQuote {
    /// Returns the deterministic domain-separated signing preimage for this quote.
    #[must_use]
    pub fn signing_preimage(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(512);
        buf.extend_from_slice(b"hanzo-attest/quote/v1\0");
        put_len_prefixed(&mut buf, &self.quote_id);
        put_len_prefixed(&mut buf, self.platform.as_str());
        put_len_prefixed(&mut buf, &self.platform_cert_id);
        buf.push(u8::from(self.environment.debug_disabled));
        buf.push(u8::from(self.environment.secure_boot));
        buf.push(u8::from(self.environment.memory_encryption));
        buf.extend_from_slice(&self.environment.tcb_version.to_be_bytes());
        buf.extend_from_slice(&self.measurement.rtmr_kernel);
        buf.extend_from_slice(&self.measurement.rtmr_runtime);
        buf.extend_from_slice(&self.measurement.enclave_code_hash);
        put_len_prefixed(&mut buf, &self.model.model_id);
        buf.extend_from_slice(&self.model.model_weights_sha256);
        buf.extend_from_slice(&self.model.tokenizer_sha256);
        buf.extend_from_slice(&self.model.workload_policy_hash);
        buf.extend_from_slice(&self.client_nonce);
        buf.extend_from_slice(&self.issued_at_unix.to_be_bytes());
        buf.extend_from_slice(&self.expires_at_unix.to_be_bytes());
        buf.extend_from_slice(&self.revocation_epoch.to_be_bytes());
        buf.extend_from_slice(&self.enclave_x25519_pubkey);
        buf.extend_from_slice(&self.report_data);
        buf
    }

    /// Computes the canonical 32-byte digest of this quote's signing preimage.
    #[must_use]
    pub fn quote_hash(&self) -> [u8; 32] {
        sha256(&self.signing_preimage())
    }

    /// Issues and signs a fresh [`AttestationQuote`] with valid `report_data`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        quote_id: impl Into<String>,
        platform: TeePlatform,
        platform_cert_id: impl Into<String>,
        environment: EnvironmentState,
        measurement: SoftwareMeasurement,
        model: ModelCommitment,
        client_nonce: [u8; 32],
        issued_at_unix: u64,
        expires_at_unix: u64,
        revocation_epoch: u64,
        enclave_x25519_pubkey: [u8; 32],
        platform_ed25519_seed: &[u8; 32],
    ) -> Self {
        let report_data = compute_report_data(&client_nonce, &enclave_x25519_pubkey, &model);
        let mut quote = Self {
            quote_id: quote_id.into(),
            platform,
            platform_cert_id: platform_cert_id.into(),
            environment,
            measurement,
            model,
            client_nonce,
            issued_at_unix,
            expires_at_unix,
            revocation_epoch,
            enclave_x25519_pubkey,
            report_data,
            platform_signature: [0u8; 64],
        };
        let preimage = quote.signing_preimage();
        quote.platform_signature = ed25519_sign(platform_ed25519_seed, &preimage);
        quote
    }

    /// Serializes this quote into a [`JsonValue`].
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "quote_id".to_string(),
            JsonValue::Str(self.quote_id.clone()),
        );
        m.insert(
            "platform".to_string(),
            JsonValue::Str(self.platform.as_str().to_string()),
        );
        m.insert(
            "platform_cert_id".to_string(),
            JsonValue::Str(self.platform_cert_id.clone()),
        );
        m.insert("environment".to_string(), self.environment.to_json_value());
        m.insert("measurement".to_string(), self.measurement.to_json_value());
        m.insert("model".to_string(), self.model.to_json_value());
        m.insert(
            "client_nonce".to_string(),
            JsonValue::Str(hex_encode(&self.client_nonce)),
        );
        m.insert(
            "issued_at_unix".to_string(),
            JsonValue::Uint(self.issued_at_unix),
        );
        m.insert(
            "expires_at_unix".to_string(),
            JsonValue::Uint(self.expires_at_unix),
        );
        m.insert(
            "revocation_epoch".to_string(),
            JsonValue::Uint(self.revocation_epoch),
        );
        m.insert(
            "enclave_x25519_pubkey".to_string(),
            JsonValue::Str(hex_encode(&self.enclave_x25519_pubkey)),
        );
        m.insert(
            "report_data".to_string(),
            JsonValue::Str(hex_encode(&self.report_data)),
        );
        m.insert(
            "platform_signature".to_string(),
            JsonValue::Str(hex_encode(&self.platform_signature)),
        );
        JsonValue::Object(m)
    }

    /// Deserializes an [`AttestationQuote`] from a [`JsonValue`].
    pub fn from_json_value(v: &JsonValue) -> Result<Self> {
        let obj = v.as_object()?;
        Ok(Self {
            quote_id: JsonValue::req(obj, "quote_id")?.as_str()?.to_string(),
            platform: TeePlatform::parse(JsonValue::req(obj, "platform")?.as_str()?)?,
            platform_cert_id: JsonValue::req(obj, "platform_cert_id")?
                .as_str()?
                .to_string(),
            environment: EnvironmentState::from_json_value(JsonValue::req(obj, "environment")?)?,
            measurement: SoftwareMeasurement::from_json_value(JsonValue::req(obj, "measurement")?)?,
            model: ModelCommitment::from_json_value(JsonValue::req(obj, "model")?)?,
            client_nonce: hex_decode_array(JsonValue::req(obj, "client_nonce")?.as_str()?)?,
            issued_at_unix: JsonValue::req(obj, "issued_at_unix")?.as_u64()?,
            expires_at_unix: JsonValue::req(obj, "expires_at_unix")?.as_u64()?,
            revocation_epoch: JsonValue::req(obj, "revocation_epoch")?.as_u64()?,
            enclave_x25519_pubkey: hex_decode_array(
                JsonValue::req(obj, "enclave_x25519_pubkey")?.as_str()?,
            )?,
            report_data: hex_decode_array(JsonValue::req(obj, "report_data")?.as_str()?)?,
            platform_signature: hex_decode_array(
                JsonValue::req(obj, "platform_signature")?.as_str()?,
            )?,
        })
    }
}

/// Wire bundle presented by an untrusted GPU inference provider during handshake.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationBundle {
    /// Endorsement certificate chain linking the TEE platform key to a hardware root.
    pub platform_cert: PlatformCert,
    /// Freshly signed hardware attestation quote.
    pub quote: AttestationQuote,
}

impl AttestationBundle {
    /// Serializes this bundle into a [`JsonValue`].
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "schema_version".to_string(),
            JsonValue::Str("hanzo-attest/bundle/v1".to_string()),
        );
        m.insert(
            "platform_cert".to_string(),
            self.platform_cert.to_json_value(),
        );
        m.insert("quote".to_string(), self.quote.to_json_value());
        JsonValue::Object(m)
    }

    /// Deserializes an [`AttestationBundle`] from a [`JsonValue`].
    pub fn from_json_value(v: &JsonValue) -> Result<Self> {
        let obj = v.as_object()?;
        let version = JsonValue::req(obj, "schema_version")?.as_str()?;
        if version != "hanzo-attest/bundle/v1" {
            return Err(AttestError::new(
                ErrorCode::InvalidJson,
                format!("unsupported attestation bundle schema_version '{version}'"),
            ));
        }
        Ok(Self {
            platform_cert: PlatformCert::from_json_value(JsonValue::req(obj, "platform_cert")?)?,
            quote: AttestationQuote::from_json_value(JsonValue::req(obj, "quote")?)?,
        })
    }

    /// Exports this bundle as canonical JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        self.to_json_value().to_canonical_string()
    }

    /// Parses an attestation bundle from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        let v = parse_json(s)?;
        Self::from_json_value(&v)
    }
}

/// Unforgeable proof-carrying token produced exclusively by [`verify_attestation`].
///
/// Only a [`VerifiedAttestation`] may be passed to [`crate::channel::bind_and_encrypt_request`]
/// or [`crate::channel::AttestedSession::open`], enforcing at compile time that
/// sensitive prompts are never encrypted or transmitted before remote attestation passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedAttestation {
    quote_hash: [u8; 32],
    platform: TeePlatform,
    root_id: String,
    platform_cert_id: String,
    device_id: String,
    tcb_version: u64,
    measurement: SoftwareMeasurement,
    model: ModelCommitment,
    client_nonce: [u8; 32],
    enclave_x25519_pubkey: [u8; 32],
    issued_at_unix: u64,
    expires_at_unix: u64,
    revocation_epoch: u64,
}

impl VerifiedAttestation {
    /// Returns the SHA-256 hash of the verified quote's canonical signing preimage.
    #[must_use]
    pub const fn quote_hash(&self) -> &[u8; 32] {
        &self.quote_hash
    }

    /// Returns the verified TEE hardware platform.
    #[must_use]
    pub const fn platform(&self) -> TeePlatform {
        self.platform
    }

    /// Returns the verified hardware root identifier.
    #[must_use]
    pub fn root_id(&self) -> &str {
        &self.root_id
    }

    /// Returns the verified platform certificate identifier.
    #[must_use]
    pub fn platform_cert_id(&self) -> &str {
        &self.platform_cert_id
    }

    /// Returns the verified hardware device identifier.
    #[must_use]
    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Returns the verified TCB security version.
    #[must_use]
    pub const fn tcb_version(&self) -> u64 {
        self.tcb_version
    }

    /// Returns the verified enclave software measurement.
    #[must_use]
    pub const fn measurement(&self) -> &SoftwareMeasurement {
        &self.measurement
    }

    /// Returns the verified model and workload commitment.
    #[must_use]
    pub const fn model(&self) -> &ModelCommitment {
        &self.model
    }

    /// Returns the verified client challenge nonce.
    #[must_use]
    pub const fn client_nonce(&self) -> &[u8; 32] {
        &self.client_nonce
    }

    /// Returns the verified enclave ephemeral X25519 public key.
    #[must_use]
    pub const fn enclave_x25519_pubkey(&self) -> &[u8; 32] {
        &self.enclave_x25519_pubkey
    }

    /// Returns the quote issuance timestamp (Unix seconds).
    #[must_use]
    pub const fn issued_at_unix(&self) -> u64 {
        self.issued_at_unix
    }

    /// Returns the quote expiration timestamp (Unix seconds).
    #[must_use]
    pub const fn expires_at_unix(&self) -> u64 {
        self.expires_at_unix
    }

    /// Returns the quote revocation epoch.
    #[must_use]
    pub const fn revocation_epoch(&self) -> u64 {
        self.revocation_epoch
    }
}

/// Performs complete offline verification of a provider's [`AttestationBundle`]
/// against the client's sovereign [`TrustStore`], [`VerificationPolicy`],
/// [`RevocationList`], expected fresh `client_nonce`, current wall-clock time
/// `now_unix`, and optional [`NonceReplayCache`].
///
/// Check ordering is canonical and mirrored identically in the independent
/// Python client so differential tests confirm exact verdict and error-code parity.
pub fn verify_attestation(
    bundle: &AttestationBundle,
    trust_store: &TrustStore,
    policy: &VerificationPolicy,
    revocations: &RevocationList,
    expected_client_nonce: &[u8; 32],
    now_unix: u64,
    replay_cache: Option<&mut NonceReplayCache>,
) -> Result<VerifiedAttestation> {
    let cert = &bundle.platform_cert;
    let quote = &bundle.quote;

    // 1. Hardware root trust & revocation
    let root = trust_store.get(&cert.root_id).ok_or_else(|| {
        AttestError::new(
            ErrorCode::UntrustedRoot,
            format!(
                "hardware root '{}' is not in client trust store",
                cert.root_id
            ),
        )
    })?;
    if revocations.revoked_root_ids.contains(&root.root_id) {
        return Err(AttestError::new(
            ErrorCode::RootRevoked,
            format!("hardware root '{}' is revoked", root.root_id),
        ));
    }
    if now_unix < root.not_before_unix || now_unix > root.not_after_unix {
        return Err(AttestError::new(
            ErrorCode::RootExpired,
            format!(
                "hardware root '{}' outside validity window [{}..={}] at {}",
                root.root_id, root.not_before_unix, root.not_after_unix, now_unix
            ),
        ));
    }

    // 2. Platform consistency and allowlist
    if root.platform != cert.platform
        || cert.platform != quote.platform
        || !policy.allowed_platforms.contains(&quote.platform)
    {
        return Err(AttestError::new(
            ErrorCode::PlatformMismatch,
            format!(
                "TEE platform mismatch or disallowed platform (root={}, cert={}, quote={})",
                root.platform.as_str(),
                cert.platform.as_str(),
                quote.platform.as_str()
            ),
        ));
    }

    // 3. Platform certificate revocation, validity, and root signature
    if revocations
        .revoked_platform_cert_ids
        .contains(&cert.cert_id)
    {
        return Err(AttestError::new(
            ErrorCode::PlatformCertRevoked,
            format!("platform cert '{}' is revoked", cert.cert_id),
        ));
    }
    if revocations.revoked_device_ids.contains(&cert.device_id) {
        return Err(AttestError::new(
            ErrorCode::DeviceRevoked,
            format!("hardware device '{}' is revoked", cert.device_id),
        ));
    }
    if now_unix > cert.expires_at_unix {
        return Err(AttestError::new(
            ErrorCode::PlatformCertExpired,
            format!(
                "platform cert '{}' expired at {} (now={})",
                cert.cert_id, cert.expires_at_unix, now_unix
            ),
        ));
    }
    if cert.issued_at_unix > now_unix.saturating_add(policy.max_clock_skew_secs) {
        return Err(AttestError::new(
            ErrorCode::PlatformCertNotYetValid,
            format!(
                "platform cert '{}' issued_at {} exceeds now {} + skew {}",
                cert.cert_id, cert.issued_at_unix, now_unix, policy.max_clock_skew_secs
            ),
        ));
    }
    let cert_preimage = cert.signing_preimage();
    if !ed25519_verify(
        &root.ed25519_public_key,
        &cert_preimage,
        &cert.root_signature,
    ) {
        return Err(AttestError::new(
            ErrorCode::InvalidPlatformCertSignature,
            format!(
                "hardware root '{}' Ed25519 signature invalid for platform cert '{}'",
                root.root_id, cert.cert_id
            ),
        ));
    }

    // 4. Quote linkage and platform signature
    if quote.platform_cert_id != cert.cert_id {
        return Err(AttestError::new(
            ErrorCode::QuoteCertIdMismatch,
            format!(
                "quote platform_cert_id '{}' does not match cert '{}'",
                quote.platform_cert_id, cert.cert_id
            ),
        ));
    }
    let quote_preimage = quote.signing_preimage();
    if !ed25519_verify(
        &cert.platform_ed25519_pubkey,
        &quote_preimage,
        &quote.platform_signature,
    ) {
        return Err(AttestError::new(
            ErrorCode::InvalidQuoteSignature,
            format!(
                "platform Ed25519 signature invalid for quote '{}'",
                quote.quote_id
            ),
        ));
    }

    // 5. Attested environment security flags and TCB floor
    if policy.require_debug_disabled && !quote.environment.debug_disabled {
        return Err(AttestError::new(
            ErrorCode::DebugModeEnabled,
            "attested TEE environment reports debug mode enabled",
        ));
    }
    if policy.require_secure_boot && !quote.environment.secure_boot {
        return Err(AttestError::new(
            ErrorCode::SecureBootDisabled,
            "attested TEE environment reports secure boot disabled",
        ));
    }
    if policy.require_memory_encryption && !quote.environment.memory_encryption {
        return Err(AttestError::new(
            ErrorCode::MemoryEncryptionDisabled,
            "attested TEE environment reports memory encryption disabled",
        ));
    }
    if quote.environment.tcb_version != cert.tcb_version {
        return Err(AttestError::new(
            ErrorCode::TcbCertMismatch,
            format!(
                "quote tcb_version {} disagrees with platform cert tcb_version {}",
                quote.environment.tcb_version, cert.tcb_version
            ),
        ));
    }
    let min_tcb = policy
        .min_tcb_versions
        .get(&quote.platform)
        .copied()
        .unwrap_or(0);
    if quote.environment.tcb_version < min_tcb {
        return Err(AttestError::new(
            ErrorCode::TcbVersionTooLow,
            format!(
                "reported TCB version {} is below policy floor {}",
                quote.environment.tcb_version, min_tcb
            ),
        ));
    }

    // 6. Software measurement allowlist
    if !policy.allowed_measurements.contains(&quote.measurement) {
        return Err(AttestError::new(
            ErrorCode::MeasurementMismatch,
            "attested RTMR kernel/runtime/enclave_code_hash is not allowlisted by policy",
        ));
    }

    // 7. Model & workload commitment
    if quote.model.model_id != policy.expected_model.model_id {
        return Err(AttestError::new(
            ErrorCode::ModelIdMismatch,
            format!(
                "attested model_id '{}' != expected '{}'",
                quote.model.model_id, policy.expected_model.model_id
            ),
        ));
    }
    if !ct_eq(
        &quote.model.model_weights_sha256,
        &policy.expected_model.model_weights_sha256,
    ) {
        return Err(AttestError::new(
            ErrorCode::ModelWeightsMismatch,
            "attested model_weights_sha256 does not match policy commitment",
        ));
    }
    if !ct_eq(
        &quote.model.tokenizer_sha256,
        &policy.expected_model.tokenizer_sha256,
    ) {
        return Err(AttestError::new(
            ErrorCode::TokenizerMismatch,
            "attested tokenizer_sha256 does not match policy commitment",
        ));
    }
    if !ct_eq(
        &quote.model.workload_policy_hash,
        &policy.expected_model.workload_policy_hash,
    ) {
        return Err(AttestError::new(
            ErrorCode::WorkloadPolicyMismatch,
            "attested workload_policy_hash does not match policy commitment",
        ));
    }

    // 8. Freshness, nonce matching, and revocation epoch
    if !ct_eq(&quote.client_nonce, expected_client_nonce) {
        return Err(AttestError::new(
            ErrorCode::NonceMismatch,
            "attested quote client_nonce does not match expected challenge nonce",
        ));
    }
    if now_unix > quote.expires_at_unix {
        return Err(AttestError::new(
            ErrorCode::QuoteExpired,
            format!(
                "quote '{}' expired at {} (now={})",
                quote.quote_id, quote.expires_at_unix, now_unix
            ),
        ));
    }
    if quote.issued_at_unix > now_unix.saturating_add(policy.max_clock_skew_secs) {
        return Err(AttestError::new(
            ErrorCode::QuoteNotYetValid,
            format!(
                "quote '{}' issued_at {} exceeds now {} + skew {}",
                quote.quote_id, quote.issued_at_unix, now_unix, policy.max_clock_skew_secs
            ),
        ));
    }
    if now_unix.saturating_sub(quote.issued_at_unix) > policy.max_quote_age_secs {
        return Err(AttestError::new(
            ErrorCode::QuoteTooOld,
            format!(
                "quote '{}' age {}s exceeds max_quote_age_secs {}s",
                quote.quote_id,
                now_unix.saturating_sub(quote.issued_at_unix),
                policy.max_quote_age_secs
            ),
        ));
    }
    if quote.revocation_epoch < revocations.min_revocation_epoch {
        return Err(AttestError::new(
            ErrorCode::RevocationEpochStale,
            format!(
                "quote revocation_epoch {} < client minimum {}",
                quote.revocation_epoch, revocations.min_revocation_epoch
            ),
        ));
    }

    // 9. Secure-channel ephemeral X25519 key & report_data binding
    let expected_report_data = compute_report_data(
        &quote.client_nonce,
        &quote.enclave_x25519_pubkey,
        &quote.model,
    );
    if !ct_eq(&quote.report_data, &expected_report_data) {
        return Err(AttestError::new(
            ErrorCode::ReportDataMismatch,
            "quote report_data does not match bound (nonce, enclave_x25519_pubkey, model, workload)",
        ));
    }
    // Reject low-order / degenerate enclave ephemeral X25519 public keys upfront.
    let probe_scalar = [0x42u8; 32];
    let _ = x25519_diffie_hellman(&probe_scalar, &quote.enclave_x25519_pubkey)?;

    // 10. Nonce replay defense (applied only after cryptographic & policy validation passes)
    if let Some(cache) = replay_cache {
        cache.check_and_record(quote.client_nonce, quote.expires_at_unix, now_unix)?;
    }

    Ok(VerifiedAttestation {
        quote_hash: sha256(&quote_preimage),
        platform: quote.platform,
        root_id: root.root_id.clone(),
        platform_cert_id: cert.cert_id.clone(),
        device_id: cert.device_id.clone(),
        tcb_version: quote.environment.tcb_version,
        measurement: quote.measurement.clone(),
        model: quote.model.clone(),
        client_nonce: quote.client_nonce,
        enclave_x25519_pubkey: quote.enclave_x25519_pubkey,
        issued_at_unix: quote.issued_at_unix,
        expires_at_unix: quote.expires_at_unix,
        revocation_epoch: quote.revocation_epoch,
    })
}
