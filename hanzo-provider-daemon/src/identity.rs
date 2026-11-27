//! Self-sovereign cryptographic provider & tenant identities using RFC 8032 Ed25519.
//!
//! Every provider and tenant owns its own Ed25519 keypair with zero dependency on
//! any centralized registry or Hanzo-operated KMS. All signatures use explicit
//! protocol domain separation so a signature over one message type (e.g., quote)
//! can never be replayed as another (e.g., lease or receipt).

use std::collections::BTreeMap;
use std::fmt;

use crate::crypto::{ed25519_verify, hex_decode, hex_encode, sha256, Ed25519Keypair};
use crate::json::{parse_json, JsonValue};

/// Domain separator for signed provider capability announcements.
pub const DOMAIN_ANNOUNCEMENT: &[u8] = b"hanzo-provider-daemon:announcement:v1\0";
/// Domain separator for signed provider capacity/price quotes.
pub const DOMAIN_QUOTE: &[u8] = b"hanzo-provider-daemon:quote:v1\0";
/// Domain separator for tenant-signed job execution leases.
pub const DOMAIN_LEASE: &[u8] = b"hanzo-provider-daemon:lease:v1\0";
/// Domain separator for provider-signed job completion receipts.
pub const DOMAIN_RECEIPT: &[u8] = b"hanzo-provider-daemon:receipt:v1\0";

/// Self-sovereign provider identifier derived deterministically from its Ed25519 public key:
/// `prov_<first 16 bytes of SHA-256(pubkey) in hex>`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProviderId(String);

impl ProviderId {
    pub fn from_public_key(public_key: &[u8; 32]) -> Self {
        let digest = sha256(public_key);
        Self(format!("prov_{}", hex_encode(&digest[..16])))
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        if !s.starts_with("prov_") || s.len() != 37 {
            return Err(format!("invalid provider_id format: `{s}`"));
        }
        let hex_part = &s[5..];
        if !hex_part.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("invalid provider_id hex: `{s}`"));
        }
        Ok(Self(s.to_ascii_lowercase()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Self-sovereign provider Ed25519 keypair.
#[derive(Clone, Debug)]
pub struct ProviderKeyPair {
    inner: Ed25519Keypair,
    provider_id: ProviderId,
}

impl ProviderKeyPair {
    /// Creates a provider keypair from a 32-byte Ed25519 seed.
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let inner = Ed25519Keypair::from_seed(seed);
        let provider_id = ProviderId::from_public_key(&inner.public_key());
        Self { inner, provider_id }
    }

    /// Derives a deterministic provider keypair from an operator label & seed material
    /// (useful for reproducible tests, benchmarks, and deterministic key derivation).
    pub fn from_label(label: &str) -> Self {
        let mut input = Vec::with_capacity(32 + label.len());
        input.extend_from_slice(b"hanzo-provider-seed:v1\0");
        input.extend_from_slice(label.as_bytes());
        Self::from_seed(sha256(&input))
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.inner.public_key()
    }

    pub fn public_key_hex(&self) -> String {
        hex_encode(&self.inner.public_key())
    }

    /// Signs a domain-separated payload using RFC 8032 Ed25519.
    pub fn sign_domain(&self, domain: &[u8], payload: &[u8]) -> [u8; 64] {
        let mut msg = Vec::with_capacity(domain.len() + payload.len());
        msg.extend_from_slice(domain);
        msg.extend_from_slice(payload);
        self.inner.sign(&msg)
    }

    /// Exports the keypair in a sovereign, portable JSON envelope so an operator
    /// can migrate or fork their provider daemon onto another host without lock-in.
    pub fn export_envelope(&self) -> KeyEnvelope {
        KeyEnvelope {
            version: 1,
            algorithm: "ed25519-rfc8032".into(),
            provider_id: self.provider_id.as_str().to_owned(),
            public_key_hex: self.public_key_hex(),
            seed_hex: hex_encode(&self.inner.seed()),
        }
    }

    /// Restores a provider keypair from an exported [`KeyEnvelope`], validating
    /// that the derived public key and `provider_id` match the envelope.
    pub fn import_envelope(env: &KeyEnvelope) -> Result<Self, String> {
        if env.version != 1 || env.algorithm != "ed25519-rfc8032" {
            return Err("unsupported key envelope version or algorithm".into());
        }
        let seed_vec = hex_decode(&env.seed_hex)?;
        let seed: [u8; 32] = seed_vec
            .try_into()
            .map_err(|_| "key envelope seed must be 32 bytes".to_owned())?;
        let kp = Self::from_seed(seed);
        if kp.public_key_hex() != env.public_key_hex || kp.provider_id.as_str() != env.provider_id {
            return Err("key envelope public_key/provider_id mismatch".into());
        }
        Ok(kp)
    }
}

/// Self-sovereign tenant Ed25519 keypair used to sign paid job leases.
#[derive(Clone, Debug)]
pub struct TenantKeyPair {
    tenant_id: String,
    inner: Ed25519Keypair,
}

impl TenantKeyPair {
    pub fn from_seed(tenant_id: impl Into<String>, seed: [u8; 32]) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            inner: Ed25519Keypair::from_seed(seed),
        }
    }

    pub fn from_label(tenant_id: impl Into<String>) -> Self {
        let tid = tenant_id.into();
        let mut input = Vec::with_capacity(32 + tid.len());
        input.extend_from_slice(b"hanzo-tenant-seed:v1\0");
        input.extend_from_slice(tid.as_bytes());
        Self::from_seed(tid, sha256(&input))
    }

    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.inner.public_key()
    }

    pub fn public_key_hex(&self) -> String {
        hex_encode(&self.inner.public_key())
    }

    pub fn sign_domain(&self, domain: &[u8], payload: &[u8]) -> [u8; 64] {
        let mut msg = Vec::with_capacity(domain.len() + payload.len());
        msg.extend_from_slice(domain);
        msg.extend_from_slice(payload);
        self.inner.sign(&msg)
    }
}

/// Verifies a domain-separated Ed25519 signature given hex-encoded public key and signature.
pub fn verify_domain_hex(
    public_key_hex: &str,
    domain: &[u8],
    payload: &[u8],
    signature_hex: &str,
) -> bool {
    let Ok(pk_vec) = hex_decode(public_key_hex) else {
        return false;
    };
    let Ok(sig_vec) = hex_decode(signature_hex) else {
        return false;
    };
    let Ok(pk): Result<[u8; 32], _> = pk_vec.try_into() else {
        return false;
    };
    let Ok(sig): Result<[u8; 64], _> = sig_vec.try_into() else {
        return false;
    };
    let mut msg = Vec::with_capacity(domain.len() + payload.len());
    msg.extend_from_slice(domain);
    msg.extend_from_slice(payload);
    ed25519_verify(&pk, &msg, &sig)
}

/// Verifies that `provider_id` cryptographically binds to `public_key_hex`.
pub fn verify_provider_id_binding(provider_id: &str, public_key_hex: &str) -> bool {
    let Ok(pk_vec) = hex_decode(public_key_hex) else {
        return false;
    };
    let Ok(pk): Result<[u8; 32], _> = pk_vec.try_into() else {
        return false;
    };
    ProviderId::from_public_key(&pk).as_str() == provider_id
}

/// Portable sovereign key export format (Apache-2.0, zero vendor lock-in).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyEnvelope {
    pub version: u64,
    pub algorithm: String,
    pub provider_id: String,
    pub public_key_hex: String,
    pub seed_hex: String,
}

impl KeyEnvelope {
    pub fn to_json(&self) -> JsonValue {
        let mut map = BTreeMap::new();
        map.insert("version".into(), JsonValue::Uint(self.version));
        map.insert("algorithm".into(), JsonValue::Str(self.algorithm.clone()));
        map.insert(
            "provider_id".into(),
            JsonValue::Str(self.provider_id.clone()),
        );
        map.insert(
            "public_key_hex".into(),
            JsonValue::Str(self.public_key_hex.clone()),
        );
        map.insert("seed_hex".into(), JsonValue::Str(self.seed_hex.clone()));
        JsonValue::Object(map)
    }

    pub fn from_json_str(s: &str) -> Result<Self, String> {
        let v = parse_json(s).map_err(|e| e.to_string())?;
        Ok(Self {
            version: v.get_u64("version").map_err(|e| e.to_string())?,
            algorithm: v
                .get_str("algorithm")
                .map_err(|e| e.to_string())?
                .to_owned(),
            provider_id: v
                .get_str("provider_id")
                .map_err(|e| e.to_string())?
                .to_owned(),
            public_key_hex: v
                .get_str("public_key_hex")
                .map_err(|e| e.to_string())?
                .to_owned(),
            seed_hex: v.get_str("seed_hex").map_err(|e| e.to_string())?.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_envelope_roundtrip_and_domain_separation() {
        let kp = ProviderKeyPair::from_label("operator-berlin-gpu-01");
        let env = kp.export_envelope();
        let json = env.to_json().to_compact_string();
        let parsed = KeyEnvelope::from_json_str(&json).unwrap();
        let restored = ProviderKeyPair::import_envelope(&parsed).unwrap();

        assert_eq!(restored.provider_id(), kp.provider_id());
        assert_eq!(restored.public_key(), kp.public_key());

        let sig = kp.sign_domain(DOMAIN_QUOTE, b"payload");
        let sig_hex = hex_encode(&sig);
        assert!(verify_domain_hex(
            &kp.public_key_hex(),
            DOMAIN_QUOTE,
            b"payload",
            &sig_hex
        ));
        // Cross-domain replay must fail.
        assert!(!verify_domain_hex(
            &kp.public_key_hex(),
            DOMAIN_RECEIPT,
            b"payload",
            &sig_hex
        ));
    }
}
