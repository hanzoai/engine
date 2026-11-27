//! Sovereign hardware roots of trust, verification policies, revocation lists,
//! and replay-defense caches for provider-independent TEE attestation.
//!
//! All policy artifacts serialize losslessly to canonical JSON so users can
//! inspect, export, migrate, or fork their trust roots and measurement allowlists
//! with zero Hanzo control-plane dependencies.

use std::collections::{BTreeMap, BTreeSet};

use crate::crypto::{hex_decode_array, hex_encode, Sha256};
use crate::error::{AttestError, ErrorCode, Result};
use crate::json::{parse_json, JsonValue};

/// Supported Confidential GPU / CPU Trusted Execution Environment platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TeePlatform {
    /// NVIDIA H100 / Blackwell Confidential Computing GPU enclave.
    NvidiaCcH100,
    /// AMD EPYC SEV-SNP confidential VM.
    AmdSevSnp,
    /// Intel Trust Domain Extensions (TDX) confidential VM.
    IntelTdx,
}

impl TeePlatform {
    /// Returns the canonical wire/JSON identifier for this TEE platform.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NvidiaCcH100 => "nvidia_cc_h100",
            Self::AmdSevSnp => "amd_sev_snp",
            Self::IntelTdx => "intel_tdx",
        }
    }

    /// Parses a canonical TEE platform identifier.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "nvidia_cc_h100" => Ok(Self::NvidiaCcH100),
            "amd_sev_snp" => Ok(Self::AmdSevSnp),
            "intel_tdx" => Ok(Self::IntelTdx),
            _ => Err(AttestError::new(
                ErrorCode::PlatformMismatch,
                format!("unknown TEE platform '{s}'"),
            )),
        }
    }
}

/// Hardware root-of-trust public key anchor pinned locally by the client.
///
/// Note on trust model: trusting a [`HardwareRootCert`] expresses reliance on
/// the silicon vendor's manufacturing root, firmware TCB, and physical security —
/// distinct from trustless cryptographic proofs such as zkML or Freivalds PoAI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareRootCert {
    /// Unique identifier for this hardware root key (e.g. `nvidia-h100-cc-root-v1`).
    pub root_id: String,
    /// Hardware TEE platform endorsed by this root.
    pub platform: TeePlatform,
    /// Ed25519 public key of the hardware root of trust.
    pub ed25519_public_key: [u8; 32],
    /// Start of validity window (inclusive, Unix timestamp seconds).
    pub not_before_unix: u64,
    /// End of validity window (inclusive, Unix timestamp seconds).
    pub not_after_unix: u64,
}

impl HardwareRootCert {
    /// Encodes this root certificate into a [`JsonValue`].
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("root_id".to_string(), JsonValue::Str(self.root_id.clone()));
        m.insert(
            "platform".to_string(),
            JsonValue::Str(self.platform.as_str().to_string()),
        );
        m.insert(
            "ed25519_public_key".to_string(),
            JsonValue::Str(hex_encode(&self.ed25519_public_key)),
        );
        m.insert(
            "not_before_unix".to_string(),
            JsonValue::Uint(self.not_before_unix),
        );
        m.insert(
            "not_after_unix".to_string(),
            JsonValue::Uint(self.not_after_unix),
        );
        JsonValue::Object(m)
    }

    /// Decodes a [`HardwareRootCert`] from a [`JsonValue`].
    pub fn from_json_value(v: &JsonValue) -> Result<Self> {
        let obj = v.as_object()?;
        Ok(Self {
            root_id: JsonValue::req(obj, "root_id")?.as_str()?.to_string(),
            platform: TeePlatform::parse(JsonValue::req(obj, "platform")?.as_str()?)?,
            ed25519_public_key: hex_decode_array(
                JsonValue::req(obj, "ed25519_public_key")?.as_str()?,
            )?,
            not_before_unix: JsonValue::req(obj, "not_before_unix")?.as_u64()?,
            not_after_unix: JsonValue::req(obj, "not_after_unix")?.as_u64()?,
        })
    }
}

/// Client-local sovereign store of pinned hardware root certificates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrustStore {
    /// Map from `root_id` to pinned [`HardwareRootCert`].
    pub roots: BTreeMap<String, HardwareRootCert>,
}

impl TrustStore {
    /// Creates an empty trust store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a hardware root certificate by its `root_id`.
    pub fn insert(&mut self, root: HardwareRootCert) {
        self.roots.insert(root.root_id.clone(), root);
    }

    /// Looks up a hardware root certificate by `root_id`.
    #[must_use]
    pub fn get(&self, root_id: &str) -> Option<&HardwareRootCert> {
        self.roots.get(root_id)
    }

    /// Serializes this trust store into a [`JsonValue`].
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        let arr = self
            .roots
            .values()
            .map(HardwareRootCert::to_json_value)
            .collect();
        m.insert("roots".to_string(), JsonValue::Array(arr));
        JsonValue::Object(m)
    }

    /// Deserializes a [`TrustStore`] from a [`JsonValue`].
    pub fn from_json_value(v: &JsonValue) -> Result<Self> {
        let obj = v.as_object()?;
        let arr = JsonValue::req(obj, "roots")?.as_array()?;
        let mut store = Self::new();
        for item in arr {
            store.insert(HardwareRootCert::from_json_value(item)?);
        }
        Ok(store)
    }

    /// Exports this trust store as canonical JSON.
    #[must_use]
    pub fn export_json(&self) -> String {
        self.to_json_value().to_canonical_string()
    }

    /// Imports a trust store from JSON.
    pub fn import_json(s: &str) -> Result<Self> {
        let v = parse_json(s)?;
        Self::from_json_value(&v)
    }
}

/// Local revocation state evaluated completely offline by the client.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RevocationList {
    /// Minimum acceptable `revocation_epoch` carried by attestation quotes.
    pub min_revocation_epoch: u64,
    /// Revoked hardware root identifiers.
    pub revoked_root_ids: BTreeSet<String>,
    /// Revoked platform endorsement certificate identifiers.
    pub revoked_platform_cert_ids: BTreeSet<String>,
    /// Revoked physical hardware device identifiers.
    pub revoked_device_ids: BTreeSet<String>,
}

impl RevocationList {
    /// Serializes this revocation list into a [`JsonValue`].
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "min_revocation_epoch".to_string(),
            JsonValue::Uint(self.min_revocation_epoch),
        );
        m.insert(
            "revoked_root_ids".to_string(),
            JsonValue::Array(
                self.revoked_root_ids
                    .iter()
                    .cloned()
                    .map(JsonValue::Str)
                    .collect(),
            ),
        );
        m.insert(
            "revoked_platform_cert_ids".to_string(),
            JsonValue::Array(
                self.revoked_platform_cert_ids
                    .iter()
                    .cloned()
                    .map(JsonValue::Str)
                    .collect(),
            ),
        );
        m.insert(
            "revoked_device_ids".to_string(),
            JsonValue::Array(
                self.revoked_device_ids
                    .iter()
                    .cloned()
                    .map(JsonValue::Str)
                    .collect(),
            ),
        );
        JsonValue::Object(m)
    }

    /// Deserializes a [`RevocationList`] from a [`JsonValue`].
    pub fn from_json_value(v: &JsonValue) -> Result<Self> {
        let obj = v.as_object()?;
        let parse_set = |key: &str| -> Result<BTreeSet<String>> {
            let arr = JsonValue::req(obj, key)?.as_array()?;
            let mut set = BTreeSet::new();
            for item in arr {
                set.insert(item.as_str()?.to_string());
            }
            Ok(set)
        };
        Ok(Self {
            min_revocation_epoch: JsonValue::req(obj, "min_revocation_epoch")?.as_u64()?,
            revoked_root_ids: parse_set("revoked_root_ids")?,
            revoked_platform_cert_ids: parse_set("revoked_platform_cert_ids")?,
            revoked_device_ids: parse_set("revoked_device_ids")?,
        })
    }

    /// Exports this revocation list as canonical JSON.
    #[must_use]
    pub fn export_json(&self) -> String {
        self.to_json_value().to_canonical_string()
    }

    /// Imports a revocation list from JSON.
    pub fn import_json(s: &str) -> Result<Self> {
        let v = parse_json(s)?;
        Self::from_json_value(&v)
    }
}

/// Measured software stack inside the confidential GPU/CPU enclave.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SoftwareMeasurement {
    /// Runtime Measurement Register (RTMR) for guest kernel and firmware.
    pub rtmr_kernel: [u8; 32],
    /// RTMR for the confidential inference runtime / container image.
    pub rtmr_runtime: [u8; 32],
    /// SHA-256 digest of the modelwrap / enclave inference binary.
    pub enclave_code_hash: [u8; 32],
}

impl SoftwareMeasurement {
    /// Serializes this measurement into a [`JsonValue`].
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "rtmr_kernel".to_string(),
            JsonValue::Str(hex_encode(&self.rtmr_kernel)),
        );
        m.insert(
            "rtmr_runtime".to_string(),
            JsonValue::Str(hex_encode(&self.rtmr_runtime)),
        );
        m.insert(
            "enclave_code_hash".to_string(),
            JsonValue::Str(hex_encode(&self.enclave_code_hash)),
        );
        JsonValue::Object(m)
    }

    /// Deserializes a [`SoftwareMeasurement`] from a [`JsonValue`].
    pub fn from_json_value(v: &JsonValue) -> Result<Self> {
        let obj = v.as_object()?;
        Ok(Self {
            rtmr_kernel: hex_decode_array(JsonValue::req(obj, "rtmr_kernel")?.as_str()?)?,
            rtmr_runtime: hex_decode_array(JsonValue::req(obj, "rtmr_runtime")?.as_str()?)?,
            enclave_code_hash: hex_decode_array(
                JsonValue::req(obj, "enclave_code_hash")?.as_str()?,
            )?,
        })
    }
}

/// Cryptographic commitment to the exact inference model, tokenizer, and workload policy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelCommitment {
    /// Canonical model identifier (e.g. `hanzo/zen-9b-instruct-q4`).
    pub model_id: String,
    /// SHA-256 commitment over the model weight manifest / tensor shards.
    pub model_weights_sha256: [u8; 32],
    /// SHA-256 commitment over the tokenizer configuration and vocabulary.
    pub tokenizer_sha256: [u8; 32],
    /// SHA-256 commitment over the workload policy (sampling bounds, privacy rules, egress policy).
    pub workload_policy_hash: [u8; 32],
}

impl ModelCommitment {
    /// Computes the domain-separated digest of `(model_id, model_weights_sha256, tokenizer_sha256)`
    /// bound inside the quote's `report_data`.
    #[must_use]
    pub fn model_commitment_digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"hanzo-attest/model-commit/v1");
        h.update(&(self.model_id.len() as u32).to_be_bytes());
        h.update(self.model_id.as_bytes());
        h.update(&self.model_weights_sha256);
        h.update(&self.tokenizer_sha256);
        h.finalize()
    }

    /// Serializes this commitment into a [`JsonValue`].
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "model_id".to_string(),
            JsonValue::Str(self.model_id.clone()),
        );
        m.insert(
            "model_weights_sha256".to_string(),
            JsonValue::Str(hex_encode(&self.model_weights_sha256)),
        );
        m.insert(
            "tokenizer_sha256".to_string(),
            JsonValue::Str(hex_encode(&self.tokenizer_sha256)),
        );
        m.insert(
            "workload_policy_hash".to_string(),
            JsonValue::Str(hex_encode(&self.workload_policy_hash)),
        );
        JsonValue::Object(m)
    }

    /// Deserializes a [`ModelCommitment`] from a [`JsonValue`].
    pub fn from_json_value(v: &JsonValue) -> Result<Self> {
        let obj = v.as_object()?;
        Ok(Self {
            model_id: JsonValue::req(obj, "model_id")?.as_str()?.to_string(),
            model_weights_sha256: hex_decode_array(
                JsonValue::req(obj, "model_weights_sha256")?.as_str()?,
            )?,
            tokenizer_sha256: hex_decode_array(JsonValue::req(obj, "tokenizer_sha256")?.as_str()?)?,
            workload_policy_hash: hex_decode_array(
                JsonValue::req(obj, "workload_policy_hash")?.as_str()?,
            )?,
        })
    }
}

/// Provider-independent verification policy enforced before sending any sensitive input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationPolicy {
    /// Set of acceptable TEE hardware platforms.
    pub allowed_platforms: BTreeSet<TeePlatform>,
    /// Minimum required firmware/microcode TCB version per TEE platform.
    pub min_tcb_versions: BTreeMap<TeePlatform, u64>,
    /// Require `debug_disabled == true`.
    pub require_debug_disabled: bool,
    /// Require `secure_boot == true`.
    pub require_secure_boot: bool,
    /// Require `memory_encryption == true`.
    pub require_memory_encryption: bool,
    /// Allowlisted enclave software measurements (kernel RTMR, runtime RTMR, binary digest).
    pub allowed_measurements: Vec<SoftwareMeasurement>,
    /// Expected model, tokenizer, and workload commitment.
    pub expected_model: ModelCommitment,
    /// Maximum permissible age of a quote in seconds (`now_unix - issued_at_unix <= max_quote_age_secs`).
    pub max_quote_age_secs: u64,
    /// Maximum permissible forward clock skew in seconds (`issued_at_unix <= now_unix + max_clock_skew_secs`).
    pub max_clock_skew_secs: u64,
}

impl VerificationPolicy {
    /// Serializes this verification policy into a [`JsonValue`].
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "allowed_platforms".to_string(),
            JsonValue::Array(
                self.allowed_platforms
                    .iter()
                    .map(|p| JsonValue::Str(p.as_str().to_string()))
                    .collect(),
            ),
        );
        let mut tcb_map = BTreeMap::new();
        for (p, v) in &self.min_tcb_versions {
            tcb_map.insert(p.as_str().to_string(), JsonValue::Uint(*v));
        }
        m.insert("min_tcb_versions".to_string(), JsonValue::Object(tcb_map));
        m.insert(
            "require_debug_disabled".to_string(),
            JsonValue::Bool(self.require_debug_disabled),
        );
        m.insert(
            "require_secure_boot".to_string(),
            JsonValue::Bool(self.require_secure_boot),
        );
        m.insert(
            "require_memory_encryption".to_string(),
            JsonValue::Bool(self.require_memory_encryption),
        );
        m.insert(
            "allowed_measurements".to_string(),
            JsonValue::Array(
                self.allowed_measurements
                    .iter()
                    .map(SoftwareMeasurement::to_json_value)
                    .collect(),
            ),
        );
        m.insert(
            "expected_model".to_string(),
            self.expected_model.to_json_value(),
        );
        m.insert(
            "max_quote_age_secs".to_string(),
            JsonValue::Uint(self.max_quote_age_secs),
        );
        m.insert(
            "max_clock_skew_secs".to_string(),
            JsonValue::Uint(self.max_clock_skew_secs),
        );
        JsonValue::Object(m)
    }

    /// Deserializes a [`VerificationPolicy`] from a [`JsonValue`].
    pub fn from_json_value(v: &JsonValue) -> Result<Self> {
        let obj = v.as_object()?;
        let mut allowed_platforms = BTreeSet::new();
        for item in JsonValue::req(obj, "allowed_platforms")?.as_array()? {
            allowed_platforms.insert(TeePlatform::parse(item.as_str()?)?);
        }
        let mut min_tcb_versions = BTreeMap::new();
        for (k, val) in JsonValue::req(obj, "min_tcb_versions")?.as_object()? {
            min_tcb_versions.insert(TeePlatform::parse(k)?, val.as_u64()?);
        }
        let mut allowed_measurements = Vec::new();
        for item in JsonValue::req(obj, "allowed_measurements")?.as_array()? {
            allowed_measurements.push(SoftwareMeasurement::from_json_value(item)?);
        }
        Ok(Self {
            allowed_platforms,
            min_tcb_versions,
            require_debug_disabled: JsonValue::req(obj, "require_debug_disabled")?.as_bool()?,
            require_secure_boot: JsonValue::req(obj, "require_secure_boot")?.as_bool()?,
            require_memory_encryption: JsonValue::req(obj, "require_memory_encryption")?
                .as_bool()?,
            allowed_measurements,
            expected_model: ModelCommitment::from_json_value(JsonValue::req(
                obj,
                "expected_model",
            )?)?,
            max_quote_age_secs: JsonValue::req(obj, "max_quote_age_secs")?.as_u64()?,
            max_clock_skew_secs: JsonValue::req(obj, "max_clock_skew_secs")?.as_u64()?,
        })
    }

    /// Exports this policy as canonical JSON.
    #[must_use]
    pub fn export_json(&self) -> String {
        self.to_json_value().to_canonical_string()
    }

    /// Imports a policy from JSON.
    pub fn import_json(s: &str) -> Result<Self> {
        let v = parse_json(s)?;
        Self::from_json_value(&v)
    }
}

/// Complete portable sovereignty bundle containing roots, policy, and revocations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SovereignBundle {
    /// Pinned hardware roots of trust.
    pub trust_store: TrustStore,
    /// Verification policy enforced by the client.
    pub policy: VerificationPolicy,
    /// Offline revocation list.
    pub revocations: RevocationList,
}

impl SovereignBundle {
    /// Serializes this bundle into a [`JsonValue`].
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "schema_version".to_string(),
            JsonValue::Str("hanzo-attest/sovereign-bundle/v1".to_string()),
        );
        m.insert("trust_store".to_string(), self.trust_store.to_json_value());
        m.insert("policy".to_string(), self.policy.to_json_value());
        m.insert("revocations".to_string(), self.revocations.to_json_value());
        JsonValue::Object(m)
    }

    /// Deserializes a [`SovereignBundle`] from a [`JsonValue`].
    pub fn from_json_value(v: &JsonValue) -> Result<Self> {
        let obj = v.as_object()?;
        let version = JsonValue::req(obj, "schema_version")?.as_str()?;
        if version != "hanzo-attest/sovereign-bundle/v1" {
            return Err(AttestError::new(
                ErrorCode::InvalidJson,
                format!("unsupported sovereign bundle schema_version '{version}'"),
            ));
        }
        Ok(Self {
            trust_store: TrustStore::from_json_value(JsonValue::req(obj, "trust_store")?)?,
            policy: VerificationPolicy::from_json_value(JsonValue::req(obj, "policy")?)?,
            revocations: RevocationList::from_json_value(JsonValue::req(obj, "revocations")?)?,
        })
    }

    /// Exports this sovereign bundle as canonical JSON for migration or forking.
    #[must_use]
    pub fn export_json(&self) -> String {
        self.to_json_value().to_canonical_string()
    }

    /// Imports a sovereign bundle from JSON.
    pub fn import_json(s: &str) -> Result<Self> {
        let v = parse_json(s)?;
        Self::from_json_value(&v)
    }
}

/// In-memory nonce replay-defense cache preventing quote replay attacks.
#[derive(Debug, Clone, Default)]
pub struct NonceReplayCache {
    seen: BTreeMap<[u8; 32], u64>,
}

impl NonceReplayCache {
    /// Creates an empty nonce replay cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Removes entries whose expiration timestamp is strictly earlier than `now_unix`.
    pub fn prune_expired(&mut self, now_unix: u64) {
        self.seen.retain(|_, &mut exp| exp >= now_unix);
    }

    /// Checks whether `nonce` has already been consumed; if not, records it
    /// until `expires_at_unix`.
    pub fn check_and_record(
        &mut self,
        nonce: [u8; 32],
        expires_at_unix: u64,
        now_unix: u64,
    ) -> Result<()> {
        self.prune_expired(now_unix);
        if self.seen.contains_key(&nonce) {
            return Err(AttestError::new(
                ErrorCode::NonceReplayed,
                "client nonce has already been consumed in replay cache",
            ));
        }
        self.seen.insert(nonce, expires_at_unix);
        Ok(())
    }

    /// Returns the number of active nonces tracked in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Returns `true` if no active nonces are tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}
