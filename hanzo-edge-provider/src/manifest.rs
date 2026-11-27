// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! Versioned `EdgeProviderAdvertisement v1` (`hanzo.edge.provider/v1`), privacy policy
//! constraints, explicit [`TrustTier`] separation, and signed `MarketWithdrawalNotice`.
//!
//! All messages are self-sovereign and independently verifiable using provider HMAC-SHA256
//! signatures with `hanzo_coordinator_required = false`.

use crate::device::{EdgeDeviceClass, WeightPrecision};
use crate::sha256::{hmac_sha256_hex, verify_hmac_sha256_hex};
use crate::thermal::WithdrawalReason;
use std::fmt;

/// Canonical versioned schema string for edge/mobile provider advertisements.
pub const EDGE_PROVIDER_SCHEMA_V1: &str = "hanzo.edge.provider/v1";

/// Compatible Hanzo Engine release line.
pub const ENGINE_COMPAT_V1: &str = "hanzo-engine/1.7";

/// Default Enso decentralized routing topic for open-weight Zen edge providers.
pub const DEFAULT_ENSO_ROUTE_TOPIC: &str = "enso://hanzo.edge/zen-open-weight/v1";

/// Default Visor sandbox profile isolating ephemeral edge inference workers.
pub const DEFAULT_VISOR_SANDBOX_PROFILE: &str = "visor-edge-ephemeral-zero-retention-v1";

/// Privacy constraint tier advertised by an edge provider and requested by clients.
///
/// Ordered from least restrictive (`StandardBestEffort`) to strictest (`LocalOnlyTiers`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrivacyPolicy {
    /// Standard best-effort execution (may retain operational telemetry logs).
    StandardBestEffort,
    /// Disables prompt/completion disk logging on the provider host.
    NoPromptLogging,
    /// Ephemeral RAM-only KV/GDN state wiped immediately after each request; zero disk persistence.
    ZeroRetentionEphemeral,
    /// Strictest tier: ephemeral zero-retention restricted to trusted local/LAN peer clusters.
    LocalOnlyTiers,
}

impl PrivacyPolicy {
    /// Canonical wire/manifest string label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StandardBestEffort => "standard_best_effort",
            Self::NoPromptLogging => "no_prompt_logging",
            Self::ZeroRetentionEphemeral => "zero_retention_ephemeral",
            Self::LocalOnlyTiers => "local_only_tiers",
        }
    }

    /// Parses a wire/manifest privacy policy label.
    #[must_use]
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "standard_best_effort" => Some(Self::StandardBestEffort),
            "no_prompt_logging" => Some(Self::NoPromptLogging),
            "zero_retention_ephemeral" => Some(Self::ZeroRetentionEphemeral),
            "local_only_tiers" => Some(Self::LocalOnlyTiers),
            _ => None,
        }
    }

    /// Returns `true` if `self` (advertised by provider) satisfies `required` (demanded by client).
    #[must_use]
    pub const fn satisfies(self, required: Self) -> bool {
        (self as u8) >= (required as u8)
    }
}

impl fmt::Display for PrivacyPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Explicit trust tier distinguishing verifiable metering from hardware attestations and zkML.
///
/// Never conflates self-reported thermal/power telemetry with cryptographic validity proofs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrustTier {
    /// Authenticated HMAC-SHA256 receipts + output SHA-256 commitments + deterministic
    /// spot-check re-execution + wall-clock/energy plausibility checks. Thermal/power values
    /// are explicitly tagged as self-reported or OS-sensor telemetry (NOT zkML / NOT TEE).
    SignedTelemetryAndSpotCheck,
    /// Hardware TEE + hardware-signed telemetry sensor attestation (future extension; NOT claimed here).
    HardwareTeeAttested,
    /// Zero-knowledge validity proof over inference execution (future extension; NOT claimed here).
    ZkValidityProof,
}

impl TrustTier {
    /// Canonical wire/manifest string label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SignedTelemetryAndSpotCheck => "signed_telemetry_and_spot_check",
            Self::HardwareTeeAttested => "hardware_tee_attested",
            Self::ZkValidityProof => "zk_validity_proof",
        }
    }

    /// Parses a wire/manifest trust tier label.
    #[must_use]
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "signed_telemetry_and_spot_check" => Some(Self::SignedTelemetryAndSpotCheck),
            "hardware_tee_attested" => Some(Self::HardwareTeeAttested),
            "zk_validity_proof" => Some(Self::ZkValidityProof),
            _ => None,
        }
    }
}

impl fmt::Display for TrustTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Signed `EdgeProviderAdvertisement v1` (`hanzo.edge.provider/v1`) broadcast by an
/// independent community edge/mobile device to the decentralized peer market.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeProviderAdvertisement {
    /// Versioned schema identifier (`"hanzo.edge.provider/v1"`).
    pub schema_version: String,
    /// Compatible Hanzo Engine version (`"hanzo-engine/1.7"`).
    pub engine_compat: String,
    /// Unique community provider identifier (e.g. `"mac-m4-pro-community"`).
    pub provider_id: String,
    /// Advertised hardware device class.
    pub device_class: EdgeDeviceClass,
    /// Peer TCP listen address (`host:port`).
    pub endpoint_addr: String,
    /// Advertised usable unified memory cap in GiB (excluding user-reserved headroom).
    pub usable_memory_gib: f64,
    /// Mandatory user-reserved memory headroom floor in GiB.
    pub min_user_headroom_gib: f64,
    /// Open-weight Zen model identifier (e.g. `"zen-4b-hybrid-int4"`).
    pub model_id: String,
    /// Canonical hybrid model geometry label (`"4B-Hybrid"`, `"9B-Hybrid"`, `"27B-Hybrid"`).
    pub model_spec_label: String,
    /// Weight precision served by this provider.
    pub precision: WeightPrecision,
    /// Maximum context length advertised for KV cache admission.
    pub max_context_tokens: usize,
    /// Content-addressed VFS URI for the open-weight Zen model artifact.
    pub vfs_weights_uri: String,
    /// SHA-256 digest committing the loaded model weights.
    pub model_weights_digest: String,
    /// Owner-configured power budget cap in Watts.
    pub watts_cap: f64,
    /// Whether the provider is currently on AC wall power.
    pub on_ac_power: bool,
    /// Owner-configured minimum battery state-of-charge floor (%).
    pub min_battery_pct: f64,
    /// Availability window start timestamp (Unix seconds).
    pub availability_start_unix: u64,
    /// Availability window end timestamp (Unix seconds).
    pub availability_end_unix: u64,
    /// Advertised privacy policy enforced by Visor / runtime.
    pub privacy_policy: PrivacyPolicy,
    /// Enso decentralized routing topic URI.
    pub enso_route_topic: String,
    /// Visor sandbox profile isolating this worker.
    pub visor_sandbox_profile: String,
    /// Explicit trust tier (never conflates telemetry with zkML/TEE).
    pub trust_tier: TrustTier,
    /// Always `false`: protocol operates fully without any Hanzo-controlled coordinator.
    pub hanzo_coordinator_required: bool,
    /// Monotonic advertisement nonce.
    pub ad_nonce: u64,
    /// Lowercase hex HMAC-SHA256 signature over [`Self::canonical_payload`].
    pub signature_hex: String,
}

impl EdgeProviderAdvertisement {
    /// Deterministic canonical byte payload covered by `signature_hex`.
    #[must_use]
    pub fn canonical_payload(&self) -> Vec<u8> {
        format!(
            "{}|{}|{}|{}|{}|{:.4}|{:.4}|{}|{}|{}|{}|{}|{}|{:.3}|{}|{:.2}|{}|{}|{}|{}|{}|{}|{}|{}",
            self.schema_version,
            self.engine_compat,
            self.provider_id,
            self.device_class.as_str(),
            self.endpoint_addr,
            self.usable_memory_gib,
            self.min_user_headroom_gib,
            self.model_id,
            self.model_spec_label,
            self.precision.as_str(),
            self.max_context_tokens,
            self.vfs_weights_uri,
            self.model_weights_digest,
            self.watts_cap,
            self.on_ac_power,
            self.min_battery_pct,
            self.availability_start_unix,
            self.availability_end_unix,
            self.privacy_policy.as_str(),
            self.enso_route_topic,
            self.visor_sandbox_profile,
            self.trust_tier.as_str(),
            self.hanzo_coordinator_required,
            self.ad_nonce,
        )
        .into_bytes()
    }

    /// Signs the advertisement in place using `provider_secret_key`.
    pub fn sign(&mut self, provider_secret_key: &[u8]) {
        let payload = self.canonical_payload();
        self.signature_hex = hmac_sha256_hex(provider_secret_key, &payload);
    }

    /// Verifies the advertisement's HMAC-SHA256 signature and schema invariants.
    #[must_use]
    pub fn verify(&self, provider_secret_key: &[u8]) -> bool {
        if self.schema_version != EDGE_PROVIDER_SCHEMA_V1 {
            return false;
        }
        if self.hanzo_coordinator_required {
            return false;
        }
        if !self.vfs_weights_uri.starts_with("vfs://") {
            return false;
        }
        if self.availability_end_unix <= self.availability_start_unix {
            return false;
        }
        let payload = self.canonical_payload();
        verify_hmac_sha256_hex(provider_secret_key, &payload, &self.signature_hex)
    }
}

/// Signed notice emitted immediately by an edge provider upon tripping any automatic
/// withdrawal guardrail (thermal ceiling, battery floor, window expiry, or memory pressure).
#[derive(Debug, Clone, PartialEq)]
pub struct MarketWithdrawalNotice {
    /// Withdrawing provider identifier.
    pub provider_id: String,
    /// Guardrail reason that triggered withdrawal.
    pub reason: WithdrawalReason,
    /// Observed package/skin temperature at withdrawal (°C).
    pub observed_temp_c: f64,
    /// Observed battery state-of-charge at withdrawal (%).
    pub observed_battery_pct: f64,
    /// Observed free unified memory at withdrawal (GiB).
    pub observed_free_ram_gib: f64,
    /// Logical Unix timestamp of withdrawal (seconds).
    pub withdrawn_at_unix: u64,
    /// Whether any in-flight request is being gracefully drained/handed over.
    pub graceful_drain_in_flight: bool,
    /// Lowercase hex HMAC-SHA256 signature over [`Self::canonical_payload`].
    pub signature_hex: String,
}

impl MarketWithdrawalNotice {
    /// Deterministic canonical payload covered by `signature_hex`.
    #[must_use]
    pub fn canonical_payload(&self) -> Vec<u8> {
        format!(
            "WITHDRAW|{}|{}|{:.3}|{:.3}|{:.3}|{}|{}",
            self.provider_id,
            self.reason.code(),
            self.observed_temp_c,
            self.observed_battery_pct,
            self.observed_free_ram_gib,
            self.withdrawn_at_unix,
            self.graceful_drain_in_flight,
        )
        .into_bytes()
    }

    /// Signs the withdrawal notice in place using `provider_secret_key`.
    pub fn sign(&mut self, provider_secret_key: &[u8]) {
        let payload = self.canonical_payload();
        self.signature_hex = hmac_sha256_hex(provider_secret_key, &payload);
    }

    /// Verifies the withdrawal notice's HMAC-SHA256 signature.
    #[must_use]
    pub fn verify(&self, provider_secret_key: &[u8]) -> bool {
        let payload = self.canonical_payload();
        verify_hmac_sha256_hex(provider_secret_key, &payload, &self.signature_hex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn privacy_policy_ordering_enforces_zero_retention() {
        assert!(
            PrivacyPolicy::ZeroRetentionEphemeral.satisfies(PrivacyPolicy::ZeroRetentionEphemeral)
        );
        assert!(PrivacyPolicy::LocalOnlyTiers.satisfies(PrivacyPolicy::ZeroRetentionEphemeral));
        assert!(!PrivacyPolicy::StandardBestEffort.satisfies(PrivacyPolicy::ZeroRetentionEphemeral));
        assert!(!PrivacyPolicy::NoPromptLogging.satisfies(PrivacyPolicy::ZeroRetentionEphemeral));
    }
}
