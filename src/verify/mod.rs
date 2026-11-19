pub mod attestation;
pub mod crypto;
pub mod providers;

use anyhow::{Result, bail};
pub use attestation::{AttestationEvidence, VerificationReport};

pub struct Verifier {
    pub expected_measurement: String,
    pub expected_model_hash: String,
}

impl Verifier {
    pub fn new(measurement: &str, model_hash: &str) -> Self {
        Self {
            expected_measurement: measurement.to_string(),
            expected_model_hash: model_hash.to_string(),
        }
    }

    pub fn verify(&self, evidence: &AttestationEvidence) -> Result<VerificationReport> {
        // 1. Verify Hardware Signature (Root of Trust)
        // In a real scenario, this calls provider-specific SDKs (NVIDIA/Intel/AMD)
        if !evidence.verify_signature() {
            bail!("Invalid hardware signature: Evidence is forged or tampered.");
        }

        // 2. Check Freshness
        if evidence.is_stale() {
            bail!("Stale evidence: Attestation report has expired.");
        }

        // 3. Verify Software Measurement (MRENCLAVE / PCRs)
        if evidence.measurement != self.expected_measurement {
            bail!("Software measurement mismatch! Found: {}, Expected: {}", 
                evidence.measurement, self.expected_measurement);
        }

        // 4. Verify Model Commitment
        if evidence.model_hash != self.expected_model_hash {
            bail!("Model commitment mismatch! The TEE is running a different model.");
        }

        // 5. Verify Secure Channel Binding
        // The report_data must contain the hash of the ephemeral public key
        let pubkey_hash = crypto::hash_public_key(&evidence.ephemeral_pubkey);
        if evidence.report_data != pubkey_hash {
            bail!("Secure channel binding failed: Public key does not match attestation data.");
        }

        Ok(VerificationReport {
            status: "success".to_string(),
            timestamp: chrono::Utc::now(),
            provider: evidence.provider.clone(),
        })
    }
}
