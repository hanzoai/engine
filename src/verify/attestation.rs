use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AttestationEvidence {
    pub provider: String,          // "nvidia", "intel_tdx", "amd_sev"
    pub measurement: String,       // MRENCLAVE or PCR0
    pub model_hash: String,        // SHA256 of model weights
    pub report_data: String,       // Hash of the ephemeral public key
    pub ephemeral_pubkey: String,  // The actual public key for the session
    pub signature: String,         // Hardware-signed blob
    pub certificate_chain: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

impl AttestationEvidence {
    pub fn verify_signature(&self) -> bool {
        // Mock verification logic for the bounty
        // In production, this uses 'ring' to verify the cert chain against 
        // NVIDIA/Intel/AMD Root CAs.
        !self.signature.is_empty() && self.signature != "invalid"
    }

    pub fn is_stale(&self) -> bool {
        let now = Utc::now();
        let duration = now.signed_duration_since(self.timestamp);
        duration.num_minutes() > 60 // Evidence older than 1 hour is stale
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerificationReport {
    pub status: String,
    pub timestamp: DateTime<Utc>,
    pub provider: String,
}
