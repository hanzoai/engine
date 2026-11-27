//! Secure-channel key agreement and authenticated request binding.
//!
//! Binds every encrypted prompt to the verified attestation quote, model
//! weights commitment, workload policy, and fresh client nonce using
//! X25519 ECDH -> HKDF-SHA256 -> ChaCha20-Poly1305 AEAD.

use std::collections::BTreeMap;

use crate::attestation::VerifiedAttestation;
use crate::crypto::{
    chacha20_poly1305_decrypt, chacha20_poly1305_encrypt, ct_eq, hex_decode, hex_decode_array,
    hex_encode, hkdf_sha256_32, x25519_diffie_hellman, x25519_public_key, zeroize_bytes,
};
use crate::error::{AttestError, ErrorCode, Result};
use crate::json::{parse_json, JsonValue};
use crate::policy::ModelCommitment;

/// Protocol version tag embedded in every [`EncryptedBoundRequest`].
pub const BOUND_REQUEST_PROTOCOL_VERSION: &str = "hanzo-attest/bound-req/v1";

/// Computes the canonical 152-byte AEAD Associated Data (AAD) binding header:
/// `b"hanzo-attest/req-aad/v1\0" || quote_hash || model_weights_sha256 || workload_policy_hash || client_nonce`.
#[must_use]
pub fn canonical_binding_header(
    quote_hash: &[u8; 32],
    model_weights_sha256: &[u8; 32],
    workload_policy_hash: &[u8; 32],
    client_nonce: &[u8; 32],
) -> [u8; 152] {
    let mut out = [0u8; 152];
    out[0..24].copy_from_slice(b"hanzo-attest/req-aad/v1\0");
    out[24..56].copy_from_slice(quote_hash);
    out[56..88].copy_from_slice(model_weights_sha256);
    out[88..120].copy_from_slice(workload_policy_hash);
    out[120..152].copy_from_slice(client_nonce);
    out
}

/// Derives the 32-byte ChaCha20-Poly1305 session key via HKDF-SHA256 bound to
/// the quote hash, client nonce, both ephemeral X25519 public keys, and AAD header.
#[must_use]
pub fn derive_session_key(
    shared_secret: &[u8; 32],
    quote_hash: &[u8; 32],
    client_nonce: &[u8; 32],
    client_x25519_pubkey: &[u8; 32],
    enclave_x25519_pubkey: &[u8; 32],
    binding_aad: &[u8; 152],
) -> [u8; 32] {
    let mut salt = [0u8; 64];
    salt[0..32].copy_from_slice(quote_hash);
    salt[32..64].copy_from_slice(client_nonce);

    let mut info = Vec::with_capacity(27 + 32 + 32 + 152);
    info.extend_from_slice(b"hanzo-attest/session-key/v1");
    info.extend_from_slice(client_x25519_pubkey);
    info.extend_from_slice(enclave_x25519_pubkey);
    info.extend_from_slice(binding_aad);

    hkdf_sha256_32(&salt, shared_secret, &info)
}

/// Authenticated encrypted inference request cryptographically bound to a
/// specific verified TEE quote, model commitment, and workload policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptedBoundRequest {
    /// Wire protocol version identifier.
    pub protocol_version: String,
    /// SHA-256 digest of the verified attestation quote preimage.
    pub quote_hash: [u8; 32],
    /// Intended model identifier.
    pub model_id: String,
    /// Intended model weights SHA-256 commitment.
    pub model_weights_sha256: [u8; 32],
    /// Intended workload policy SHA-256 commitment.
    pub workload_policy_hash: [u8; 32],
    /// Fresh client challenge nonce bound into the quote.
    pub client_nonce: [u8; 32],
    /// Client's ephemeral X25519 public key for ECDH session derivation.
    pub client_x25519_pubkey: [u8; 32],
    /// Enclave's attested ephemeral X25519 public key.
    pub enclave_x25519_pubkey: [u8; 32],
    /// 96-bit ChaCha20-Poly1305 AEAD nonce.
    pub aead_nonce: [u8; 12],
    /// Encrypted request payload bytes.
    pub ciphertext: Vec<u8>,
    /// 128-bit Poly1305 authentication tag over AAD + ciphertext.
    pub tag: [u8; 16],
}

impl EncryptedBoundRequest {
    /// Serializes this encrypted bound request into a [`JsonValue`].
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "protocol_version".to_string(),
            JsonValue::Str(self.protocol_version.clone()),
        );
        m.insert(
            "quote_hash".to_string(),
            JsonValue::Str(hex_encode(&self.quote_hash)),
        );
        m.insert(
            "model_id".to_string(),
            JsonValue::Str(self.model_id.clone()),
        );
        m.insert(
            "model_weights_sha256".to_string(),
            JsonValue::Str(hex_encode(&self.model_weights_sha256)),
        );
        m.insert(
            "workload_policy_hash".to_string(),
            JsonValue::Str(hex_encode(&self.workload_policy_hash)),
        );
        m.insert(
            "client_nonce".to_string(),
            JsonValue::Str(hex_encode(&self.client_nonce)),
        );
        m.insert(
            "client_x25519_pubkey".to_string(),
            JsonValue::Str(hex_encode(&self.client_x25519_pubkey)),
        );
        m.insert(
            "enclave_x25519_pubkey".to_string(),
            JsonValue::Str(hex_encode(&self.enclave_x25519_pubkey)),
        );
        m.insert(
            "aead_nonce".to_string(),
            JsonValue::Str(hex_encode(&self.aead_nonce)),
        );
        m.insert(
            "ciphertext_hex".to_string(),
            JsonValue::Str(hex_encode(&self.ciphertext)),
        );
        m.insert("tag".to_string(), JsonValue::Str(hex_encode(&self.tag)));
        JsonValue::Object(m)
    }

    /// Deserializes an [`EncryptedBoundRequest`] from a [`JsonValue`].
    pub fn from_json_value(v: &JsonValue) -> Result<Self> {
        let obj = v.as_object()?;
        let protocol_version = JsonValue::req(obj, "protocol_version")?
            .as_str()?
            .to_string();
        if protocol_version != BOUND_REQUEST_PROTOCOL_VERSION {
            return Err(AttestError::new(
                ErrorCode::InvalidJson,
                format!("unsupported bound request protocol_version '{protocol_version}'"),
            ));
        }
        Ok(Self {
            protocol_version,
            quote_hash: hex_decode_array(JsonValue::req(obj, "quote_hash")?.as_str()?)?,
            model_id: JsonValue::req(obj, "model_id")?.as_str()?.to_string(),
            model_weights_sha256: hex_decode_array(
                JsonValue::req(obj, "model_weights_sha256")?.as_str()?,
            )?,
            workload_policy_hash: hex_decode_array(
                JsonValue::req(obj, "workload_policy_hash")?.as_str()?,
            )?,
            client_nonce: hex_decode_array(JsonValue::req(obj, "client_nonce")?.as_str()?)?,
            client_x25519_pubkey: hex_decode_array(
                JsonValue::req(obj, "client_x25519_pubkey")?.as_str()?,
            )?,
            enclave_x25519_pubkey: hex_decode_array(
                JsonValue::req(obj, "enclave_x25519_pubkey")?.as_str()?,
            )?,
            aead_nonce: hex_decode_array(JsonValue::req(obj, "aead_nonce")?.as_str()?)?,
            ciphertext: hex_decode(JsonValue::req(obj, "ciphertext_hex")?.as_str()?)?,
            tag: hex_decode_array(JsonValue::req(obj, "tag")?.as_str()?)?,
        })
    }

    /// Exports this request as canonical JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        self.to_json_value().to_canonical_string()
    }

    /// Parses an [`EncryptedBoundRequest`] from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        let v = parse_json(s)?;
        Self::from_json_value(&v)
    }
}

/// Derives an X25519 + HKDF-SHA256 session key from a [`VerifiedAttestation`]
/// and encrypts `plaintext` with ChaCha20-Poly1305 under the canonical binding AAD.
pub fn bind_and_encrypt_request(
    verified: &VerifiedAttestation,
    client_x25519_secret: &[u8; 32],
    aead_nonce: [u8; 12],
    plaintext: &[u8],
) -> Result<EncryptedBoundRequest> {
    let client_x25519_pubkey = x25519_public_key(client_x25519_secret);
    let mut shared_secret =
        x25519_diffie_hellman(client_x25519_secret, verified.enclave_x25519_pubkey())?;
    let aad = canonical_binding_header(
        verified.quote_hash(),
        &verified.model().model_weights_sha256,
        &verified.model().workload_policy_hash,
        verified.client_nonce(),
    );
    let mut session_key = derive_session_key(
        &shared_secret,
        verified.quote_hash(),
        verified.client_nonce(),
        &client_x25519_pubkey,
        verified.enclave_x25519_pubkey(),
        &aad,
    );
    let (ciphertext, tag) = chacha20_poly1305_encrypt(&session_key, &aead_nonce, &aad, plaintext);
    zeroize_bytes(&mut shared_secret);
    zeroize_bytes(&mut session_key);

    Ok(EncryptedBoundRequest {
        protocol_version: BOUND_REQUEST_PROTOCOL_VERSION.to_string(),
        quote_hash: *verified.quote_hash(),
        model_id: verified.model().model_id.clone(),
        model_weights_sha256: verified.model().model_weights_sha256,
        workload_policy_hash: verified.model().workload_policy_hash,
        client_nonce: *verified.client_nonce(),
        client_x25519_pubkey,
        enclave_x25519_pubkey: *verified.enclave_x25519_pubkey(),
        aead_nonce,
        ciphertext,
        tag,
    })
}

/// Amortized attested session caching a verified X25519/HKDF-SHA256 session key
/// across multiple confidential inference requests with monotonic per-request
/// AEAD nonce derivation and strict expiry/request-count bounds.
pub struct AttestedSession {
    quote_hash: [u8; 32],
    model: ModelCommitment,
    client_nonce: [u8; 32],
    client_x25519_pubkey: [u8; 32],
    enclave_x25519_pubkey: [u8; 32],
    binding_aad: [u8; 152],
    session_key: [u8; 32],
    base_nonce: [u8; 12],
    expires_at_unix: u64,
    max_requests: u64,
    request_seq: u64,
}

impl Drop for AttestedSession {
    fn drop(&mut self) {
        zeroize_bytes(&mut self.session_key);
    }
}

impl AttestedSession {
    /// Opens an amortized [`AttestedSession`] from an already-verified attestation.
    pub fn open(
        verified: &VerifiedAttestation,
        client_x25519_secret: &[u8; 32],
        base_nonce: [u8; 12],
        max_requests: u64,
    ) -> Result<Self> {
        let client_x25519_pubkey = x25519_public_key(client_x25519_secret);
        let mut shared_secret =
            x25519_diffie_hellman(client_x25519_secret, verified.enclave_x25519_pubkey())?;
        let binding_aad = canonical_binding_header(
            verified.quote_hash(),
            &verified.model().model_weights_sha256,
            &verified.model().workload_policy_hash,
            verified.client_nonce(),
        );
        let session_key = derive_session_key(
            &shared_secret,
            verified.quote_hash(),
            verified.client_nonce(),
            &client_x25519_pubkey,
            verified.enclave_x25519_pubkey(),
            &binding_aad,
        );
        zeroize_bytes(&mut shared_secret);

        Ok(Self {
            quote_hash: *verified.quote_hash(),
            model: verified.model().clone(),
            client_nonce: *verified.client_nonce(),
            client_x25519_pubkey,
            enclave_x25519_pubkey: *verified.enclave_x25519_pubkey(),
            binding_aad,
            session_key,
            base_nonce,
            expires_at_unix: verified.expires_at_unix(),
            max_requests,
            request_seq: 0,
        })
    }

    /// Encrypts a prompt within the active session, deriving a unique nonce
    /// from `base_nonce XOR seq_be` to prevent ChaCha20 nonce reuse.
    pub fn encrypt_request(
        &mut self,
        now_unix: u64,
        plaintext: &[u8],
    ) -> Result<EncryptedBoundRequest> {
        if now_unix > self.expires_at_unix || self.request_seq >= self.max_requests {
            return Err(AttestError::new(
                ErrorCode::SessionExpired,
                "attested session expired or reached max_requests; re-attestation required",
            ));
        }
        let mut nonce = self.base_nonce;
        let seq_bytes = self.request_seq.to_be_bytes();
        for i in 0..8 {
            nonce[4 + i] ^= seq_bytes[i];
        }
        self.request_seq += 1;

        let (ciphertext, tag) =
            chacha20_poly1305_encrypt(&self.session_key, &nonce, &self.binding_aad, plaintext);

        Ok(EncryptedBoundRequest {
            protocol_version: BOUND_REQUEST_PROTOCOL_VERSION.to_string(),
            quote_hash: self.quote_hash,
            model_id: self.model.model_id.clone(),
            model_weights_sha256: self.model.model_weights_sha256,
            workload_policy_hash: self.model.workload_policy_hash,
            client_nonce: self.client_nonce,
            client_x25519_pubkey: self.client_x25519_pubkey,
            enclave_x25519_pubkey: self.enclave_x25519_pubkey,
            aead_nonce: nonce,
            ciphertext,
            tag,
        })
    }

    /// Returns the number of requests encrypted on this session so far.
    #[must_use]
    pub const fn requests_sent(&self) -> u64 {
        self.request_seq
    }
}

/// Enclave-side confidential channel endpoint holding the attested ephemeral
/// X25519 private key and expected quote/model/workload binding state.
pub struct EnclaveChannel {
    enclave_x25519_secret: [u8; 32],
    enclave_x25519_pubkey: [u8; 32],
    quote_hash: [u8; 32],
    model: ModelCommitment,
    client_nonce: [u8; 32],
}

impl Drop for EnclaveChannel {
    fn drop(&mut self) {
        zeroize_bytes(&mut self.enclave_x25519_secret);
    }
}

impl EnclaveChannel {
    /// Constructs an enclave channel bound to the exact quote, model, and nonce
    /// attested by this enclave.
    #[must_use]
    pub fn new(
        enclave_x25519_secret: [u8; 32],
        quote_hash: [u8; 32],
        model: ModelCommitment,
        client_nonce: [u8; 32],
    ) -> Self {
        let enclave_x25519_pubkey = x25519_public_key(&enclave_x25519_secret);
        Self {
            enclave_x25519_secret,
            enclave_x25519_pubkey,
            quote_hash,
            model,
            client_nonce,
        }
    }

    /// Returns the enclave's ephemeral X25519 public key.
    #[must_use]
    pub const fn public_key(&self) -> &[u8; 32] {
        &self.enclave_x25519_pubkey
    }

    /// Verifies the request's binding header against the enclave's attested
    /// quote, model commitment, workload policy, and nonce, then decrypts and
    /// authenticates the prompt payload via ChaCha20-Poly1305.
    pub fn decrypt_bound_request(&self, req: &EncryptedBoundRequest) -> Result<Vec<u8>> {
        if !ct_eq(&req.quote_hash, &self.quote_hash)
            || req.model_id != self.model.model_id
            || !ct_eq(&req.model_weights_sha256, &self.model.model_weights_sha256)
            || !ct_eq(&req.workload_policy_hash, &self.model.workload_policy_hash)
            || !ct_eq(&req.client_nonce, &self.client_nonce)
            || !ct_eq(&req.enclave_x25519_pubkey, &self.enclave_x25519_pubkey)
        {
            return Err(AttestError::new(
                ErrorCode::BindingHeaderMismatch,
                "encrypted request binding header disagrees with attested enclave quote/model/workload/nonce",
            ));
        }

        let mut shared_secret =
            x25519_diffie_hellman(&self.enclave_x25519_secret, &req.client_x25519_pubkey)?;
        let aad = canonical_binding_header(
            &self.quote_hash,
            &self.model.model_weights_sha256,
            &self.model.workload_policy_hash,
            &self.client_nonce,
        );
        let mut session_key = derive_session_key(
            &shared_secret,
            &self.quote_hash,
            &self.client_nonce,
            &req.client_x25519_pubkey,
            &self.enclave_x25519_pubkey,
            &aad,
        );
        let plaintext = chacha20_poly1305_decrypt(
            &session_key,
            &req.aead_nonce,
            &aad,
            &req.ciphertext,
            &req.tag,
        );
        zeroize_bytes(&mut shared_secret);
        zeroize_bytes(&mut session_key);
        plaintext
    }
}
