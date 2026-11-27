// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Versioned `HanzoProofEnvelope v1` for interoperability with Enso routing, Visor
//! execution orchestration, VFS content-addressed weight/program artifacts, and
//! completely independent stateless verifiers (zero mandatory Hanzo service dependency).

use crate::circuit::{CircuitProof, CircuitVerifyingKey};
use crate::sha256::{hex_decode_32, hex_encode, sha256, Sha256};
use crate::zkvm::{ZkVmReceipt, ZkVmVerifyingKey};
use crate::{TrustTier, ZkmlError};

/// Schema identifier for `HanzoProofEnvelope v1`.
pub const ENVELOPE_SCHEMA_VERSION: &str = "hanzo.proof.envelope.v1";

/// Proof backend discriminator inside a [`HanzoProofEnvelope`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ProofBackend {
    /// Path A: model-specific R1CS circuit proved via Spartan + Ligero.
    ModelSpecificCircuit,
    /// Path B: general deterministic register zkVM proved via `ImageID` + trace constraints.
    GeneralZkVm,
}

impl ProofBackend {
    /// Canonical stable string identifier.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModelSpecificCircuit => "model_specific_circuit",
            Self::GeneralZkVm => "general_zkvm",
        }
    }

    /// Parses canonical backend string.
    pub fn from_str_canonical(s: &str) -> Option<Self> {
        match s {
            "model_specific_circuit" => Some(Self::ModelSpecificCircuit),
            "general_zkvm" => Some(Self::GeneralZkVm),
            _ => None,
        }
    }
}

/// VFS content-addressed references for model weights and circuit/program definitions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VfsArtifactRefs {
    /// VFS content URI for circuit digest or zkVM `ImageID` (`vfs://sha256/<hex>`).
    pub artifact_cid: String,
    /// VFS content URI for raw quantized weights (`vfs://sha256/<hex>`).
    pub weights_cid: String,
    /// Semantic version string of the circuit or zkVM ISA.
    pub artifact_version: String,
}

/// Optional Enso / Visor routing metadata carried alongside the proof envelope.
///
/// Verification NEVER depends on Enso, Visor, or any Hanzo-operated service.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnsoVisorContext {
    /// Enso route identifier (may be any self-hosted or third-party route ID).
    pub enso_route_id: String,
    /// Visor job identifier (may be any local or decentralized scheduler job ID).
    pub visor_job_id: String,
    /// Explicitly records whether any Hanzo-controlled service is required for verification.
    /// Must always be `false` under HIP-1350 Rule 2.
    pub hanzo_service_required: bool,
}

/// Canonical, self-authenticating, versioned proof envelope (`hanzo.proof.envelope.v1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HanzoProofEnvelope {
    /// Schema version (`"hanzo.proof.envelope.v1"`).
    pub schema_version: String,
    /// Explicit HIP-1350 trust tier (always [`TrustTier::CryptographicProof`] here).
    pub trust_tier: TrustTier,
    /// Proof architecture backend.
    pub proof_backend: ProofBackend,
    /// Protocol domain separation tag.
    pub domain_tag: String,
    /// Content-addressed VFS references.
    pub artifact_refs: VfsArtifactRefs,
    /// Enso/Visor routing correlation metadata.
    pub routing_context: EnsoVisorContext,
    /// SHA-256 digest of canonical public inputs.
    pub public_inputs_digest: [u8; 32],
    /// SHA-256 digest of canonical public outputs.
    pub public_outputs_digest: [u8; 32],
    /// SHA-256 digest of the canonical proof / receipt payload bytes.
    pub proof_payload_digest: [u8; 32],
    /// Exact byte length of the serialized proof / receipt payload.
    pub proof_bytes_len: usize,
    /// Canonical SHA-256 over all fields above.
    pub envelope_digest: [u8; 32],
}

/// Computes canonical SHA-256 digest of a signed `i64` public vector.
pub fn digest_i64_slice(tag: &str, values: &[i64]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(tag.as_bytes());
    h.update(&(values.len() as u64).to_le_bytes());
    for &v in values {
        h.update(&v.to_le_bytes());
    }
    h.finalize()
}

impl HanzoProofEnvelope {
    /// Wraps a Path A [`CircuitProof`] into a canonical `HanzoProofEnvelope v1`.
    pub fn from_circuit_proof(
        vk: &CircuitVerifyingKey,
        proof: &CircuitProof,
        public_inputs: &[i64],
        public_outputs: &[i64],
        enso_route_id: impl Into<String>,
        visor_job_id: impl Into<String>,
    ) -> Self {
        let proof_bytes = proof.to_bytes();
        let mut env = Self {
            schema_version: ENVELOPE_SCHEMA_VERSION.to_string(),
            trust_tier: TrustTier::CryptographicProof,
            proof_backend: ProofBackend::ModelSpecificCircuit,
            domain_tag: vk.domain_tag.clone(),
            artifact_refs: VfsArtifactRefs {
                artifact_cid: format!("vfs://sha256/{}", hex_encode(&vk.circuit_digest)),
                weights_cid: format!("vfs://sha256/{}", hex_encode(&vk.weights_sha256)),
                artifact_version: format!("{}@{}", vk.circuit_id, vk.version),
            },
            routing_context: EnsoVisorContext {
                enso_route_id: enso_route_id.into(),
                visor_job_id: visor_job_id.into(),
                hanzo_service_required: false,
            },
            public_inputs_digest: digest_i64_slice("hanzo.zkml.pub_in.v1", public_inputs),
            public_outputs_digest: digest_i64_slice("hanzo.zkml.pub_out.v1", public_outputs),
            proof_payload_digest: sha256(&proof_bytes),
            proof_bytes_len: proof_bytes.len(),
            envelope_digest: [0u8; 32],
        };
        env.envelope_digest = env.compute_envelope_digest();
        env
    }

    /// Wraps a Path B [`ZkVmReceipt`] into a canonical `HanzoProofEnvelope v1`.
    pub fn from_zkvm_receipt(
        vk: &ZkVmVerifyingKey,
        receipt: &ZkVmReceipt,
        journal_input: &[i64],
        journal_output: &[i64],
        enso_route_id: impl Into<String>,
        visor_job_id: impl Into<String>,
    ) -> Self {
        let receipt_bytes = receipt.to_bytes();
        let mut env = Self {
            schema_version: ENVELOPE_SCHEMA_VERSION.to_string(),
            trust_tier: TrustTier::CryptographicProof,
            proof_backend: ProofBackend::GeneralZkVm,
            domain_tag: vk.domain_tag.clone(),
            artifact_refs: VfsArtifactRefs {
                artifact_cid: format!("vfs://sha256/{}", hex_encode(&vk.image_id)),
                weights_cid: format!("vfs://sha256/{}", hex_encode(&vk.trace_vk.weights_sha256)),
                artifact_version: format!("{}@{}", vk.program.program_id, vk.vm_version),
            },
            routing_context: EnsoVisorContext {
                enso_route_id: enso_route_id.into(),
                visor_job_id: visor_job_id.into(),
                hanzo_service_required: false,
            },
            public_inputs_digest: digest_i64_slice("hanzo.zkml.pub_in.v1", journal_input),
            public_outputs_digest: digest_i64_slice("hanzo.zkml.pub_out.v1", journal_output),
            proof_payload_digest: sha256(&receipt_bytes),
            proof_bytes_len: receipt_bytes.len(),
            envelope_digest: [0u8; 32],
        };
        env.envelope_digest = env.compute_envelope_digest();
        env
    }

    /// Computes the canonical SHA-256 digest over every envelope field except `envelope_digest`.
    pub fn compute_envelope_digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"hanzo.proof.envelope.digest.v1");
        for s in [
            self.schema_version.as_str(),
            self.trust_tier.as_str(),
            self.proof_backend.as_str(),
            self.domain_tag.as_str(),
            self.artifact_refs.artifact_cid.as_str(),
            self.artifact_refs.weights_cid.as_str(),
            self.artifact_refs.artifact_version.as_str(),
            self.routing_context.enso_route_id.as_str(),
            self.routing_context.visor_job_id.as_str(),
        ] {
            h.update(&(s.len() as u64).to_le_bytes());
            h.update(s.as_bytes());
        }
        h.update(&[u8::from(self.routing_context.hanzo_service_required)]);
        h.update(&self.public_inputs_digest);
        h.update(&self.public_outputs_digest);
        h.update(&self.proof_payload_digest);
        h.update(&(self.proof_bytes_len as u64).to_le_bytes());
        h.finalize()
    }

    /// Statelessly validates envelope integrity, HIP-1350 sovereignty invariants,
    /// and cryptographic binding to `(proof_bytes, public_inputs, public_outputs)`.
    pub fn verify_binding(
        &self,
        proof_bytes: &[u8],
        public_inputs: &[i64],
        public_outputs: &[i64],
    ) -> Result<(), ZkmlError> {
        if self.schema_version != ENVELOPE_SCHEMA_VERSION {
            return Err(ZkmlError::EnvelopeError {
                detail: format!("unsupported schema_version '{}'", self.schema_version),
            });
        }
        if self.trust_tier != TrustTier::CryptographicProof {
            return Err(ZkmlError::EnvelopeError {
                detail: format!(
                    "hanzo-zkml envelope must have trust_tier=cryptographic_proof, got {}",
                    self.trust_tier.as_str()
                ),
            });
        }
        if self.routing_context.hanzo_service_required {
            return Err(ZkmlError::EnvelopeError {
                detail: "hanzo_service_required must be false for decentralized verification"
                    .to_string(),
            });
        }
        if self.compute_envelope_digest() != self.envelope_digest {
            return Err(ZkmlError::EnvelopeError {
                detail: "envelope_digest mismatch".to_string(),
            });
        }
        if proof_bytes.len() != self.proof_bytes_len
            || sha256(proof_bytes) != self.proof_payload_digest
        {
            return Err(ZkmlError::EnvelopeError {
                detail: "proof payload digest or byte length mismatch".to_string(),
            });
        }
        if digest_i64_slice("hanzo.zkml.pub_in.v1", public_inputs) != self.public_inputs_digest {
            return Err(ZkmlError::PublicInputMismatch {
                detail: "envelope public_inputs_digest mismatch".to_string(),
            });
        }
        if digest_i64_slice("hanzo.zkml.pub_out.v1", public_outputs) != self.public_outputs_digest {
            return Err(ZkmlError::PublicOutputMismatch {
                detail: "envelope public_outputs_digest mismatch".to_string(),
            });
        }
        Ok(())
    }

    /// Encodes the envelope into deterministic flat canonical JSON (zero external crates).
    pub fn to_canonical_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\": \"{}\",\n",
                "  \"trust_tier\": \"{}\",\n",
                "  \"proof_backend\": \"{}\",\n",
                "  \"domain_tag\": \"{}\",\n",
                "  \"artifact_cid\": \"{}\",\n",
                "  \"weights_cid\": \"{}\",\n",
                "  \"artifact_version\": \"{}\",\n",
                "  \"enso_route_id\": \"{}\",\n",
                "  \"visor_job_id\": \"{}\",\n",
                "  \"hanzo_service_required\": {},\n",
                "  \"public_inputs_digest\": \"{}\",\n",
                "  \"public_outputs_digest\": \"{}\",\n",
                "  \"proof_payload_digest\": \"{}\",\n",
                "  \"proof_bytes_len\": {},\n",
                "  \"envelope_digest\": \"{}\"\n",
                "}}"
            ),
            escape_json(&self.schema_version),
            self.trust_tier.as_str(),
            self.proof_backend.as_str(),
            escape_json(&self.domain_tag),
            escape_json(&self.artifact_refs.artifact_cid),
            escape_json(&self.artifact_refs.weights_cid),
            escape_json(&self.artifact_refs.artifact_version),
            escape_json(&self.routing_context.enso_route_id),
            escape_json(&self.routing_context.visor_job_id),
            self.routing_context.hanzo_service_required,
            hex_encode(&self.public_inputs_digest),
            hex_encode(&self.public_outputs_digest),
            hex_encode(&self.proof_payload_digest),
            self.proof_bytes_len,
            hex_encode(&self.envelope_digest),
        )
    }

    /// Parses a [`HanzoProofEnvelope`] emitted by [`Self::to_canonical_json`].
    pub fn from_canonical_json(input: &str) -> Result<Self, ZkmlError> {
        let get_str = |key: &str| -> Result<String, ZkmlError> {
            extract_json_string(input, key).ok_or_else(|| ZkmlError::EnvelopeError {
                detail: format!("missing or invalid string field '{key}'"),
            })
        };
        let get_hex32 = |key: &str| -> Result<[u8; 32], ZkmlError> {
            let s = get_str(key)?;
            hex_decode_32(&s).ok_or_else(|| ZkmlError::EnvelopeError {
                detail: format!("invalid 32-byte hex field '{key}'"),
            })
        };
        let get_bool = |key: &str| -> Result<bool, ZkmlError> {
            extract_json_raw(input, key)
                .and_then(|v| match v {
                    "true" => Some(true),
                    "false" => Some(false),
                    _ => None,
                })
                .ok_or_else(|| ZkmlError::EnvelopeError {
                    detail: format!("missing or invalid bool field '{key}'"),
                })
        };
        let get_usize = |key: &str| -> Result<usize, ZkmlError> {
            extract_json_raw(input, key)
                .and_then(|v| v.parse::<usize>().ok())
                .ok_or_else(|| ZkmlError::EnvelopeError {
                    detail: format!("missing or invalid integer field '{key}'"),
                })
        };

        let trust_tier_str = get_str("trust_tier")?;
        let trust_tier = TrustTier::from_str_canonical(&trust_tier_str).ok_or_else(|| {
            ZkmlError::EnvelopeError {
                detail: format!("unknown trust_tier '{trust_tier_str}'"),
            }
        })?;
        let backend_str = get_str("proof_backend")?;
        let proof_backend = ProofBackend::from_str_canonical(&backend_str).ok_or_else(|| {
            ZkmlError::EnvelopeError {
                detail: format!("unknown proof_backend '{backend_str}'"),
            }
        })?;

        Ok(Self {
            schema_version: get_str("schema_version")?,
            trust_tier,
            proof_backend,
            domain_tag: get_str("domain_tag")?,
            artifact_refs: VfsArtifactRefs {
                artifact_cid: get_str("artifact_cid")?,
                weights_cid: get_str("weights_cid")?,
                artifact_version: get_str("artifact_version")?,
            },
            routing_context: EnsoVisorContext {
                enso_route_id: get_str("enso_route_id")?,
                visor_job_id: get_str("visor_job_id")?,
                hanzo_service_required: get_bool("hanzo_service_required")?,
            },
            public_inputs_digest: get_hex32("public_inputs_digest")?,
            public_outputs_digest: get_hex32("public_outputs_digest")?,
            proof_payload_digest: get_hex32("proof_payload_digest")?,
            proof_bytes_len: get_usize("proof_bytes_len")?,
            envelope_digest: get_hex32("envelope_digest")?,
        })
    }
}

/// Escapes special characters for canonical JSON string emission.
pub fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn extract_json_string(doc: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\": \"");
    let start = doc.find(&needle)? + needle.len();
    let rest = &doc[start..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => {
                let esc = chars.next()?;
                match esc {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    _ => return None,
                }
            }
            other => out.push(other),
        }
    }
    None
}

fn extract_json_raw<'a>(doc: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\": ");
    let start = doc.find(&needle)? + needle.len();
    let rest = &doc[start..];
    let end = rest.find([',', '\n', '}'])?;
    Some(rest[..end].trim())
}
