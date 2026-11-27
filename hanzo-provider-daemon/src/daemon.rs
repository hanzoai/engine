//! Core permissionless GPU provider daemon:
//! - Quote issuance (`SignedQuote`)
//! - Pre-execution lease & replay verification (`SignedJobLease`)
//! - Strict VRAM/concurrency admission control & tenant KV/scratch isolation
//! - Pluggable `InferenceBackend` trait (clean seam for `hanzo-engine`)
//! - Signed `CompletionReceipt` generation + hash-chained `SovereignLedger`
//! - Graceful drain (`initiate_drain` / `wait_until_drained`) & cold restart replay

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::advertise::{CapabilityAnnouncement, GpuDevice, ModelCapability};
use crate::crypto::{hex_encode, sha256_hex};
use crate::identity::{
    verify_domain_hex, verify_provider_id_binding, ProviderKeyPair, TenantKeyPair, DOMAIN_LEASE,
    DOMAIN_QUOTE,
};
use crate::isolation::{
    validate_identifier, CapacityGuard, CapacityPool, CapacitySnapshot, IsolationError,
    SandboxExecutionPolicy, TenantNamespace,
};
use crate::json::{JsonError, JsonValue};
use crate::ledger::{CompletionReceipt, LedgerSummary, SovereignLedger};

/// Current wall-clock time in Unix milliseconds.
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Deterministic token estimator (~4 UTF-8 bytes per token, minimum 1 for non-empty text).
pub fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    (text.len() as u64).div_ceil(4)
}

/// Request from a tenant (or decentralized router like Enso) for a price & capacity quote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuoteRequest {
    pub tenant_id: String,
    pub model_id: String,
    pub prompt_tokens_estimate: u64,
    pub max_tokens: u32,
}

impl QuoteRequest {
    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("tenant_id".into(), JsonValue::Str(self.tenant_id.clone()));
        m.insert("model_id".into(), JsonValue::Str(self.model_id.clone()));
        m.insert(
            "prompt_tokens_estimate".into(),
            JsonValue::Uint(self.prompt_tokens_estimate),
        );
        m.insert("max_tokens".into(), JsonValue::Uint(self.max_tokens as u64));
        JsonValue::Object(m)
    }

    pub fn from_json(v: &JsonValue) -> Result<Self, JsonError> {
        Ok(Self {
            tenant_id: v.get_str("tenant_id")?.to_owned(),
            model_id: v.get_str("model_id")?.to_owned(),
            prompt_tokens_estimate: v.get_u64("prompt_tokens_estimate")?,
            max_tokens: v.get_u64("max_tokens")? as u32,
        })
    }
}

/// Provider-signed capacity & price quote (Ed25519 over `DOMAIN_QUOTE`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedQuote {
    pub protocol_version: u64,
    pub quote_id: String,
    pub provider_id: String,
    pub provider_public_key_hex: String,
    pub tenant_id: String,
    pub model_id: String,
    pub max_tokens: u32,
    pub price_micro_usd_per_1m: u64,
    pub vram_reserved_mib: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub signature_hex: String,
}

impl SignedQuote {
    pub fn canonical_payload_bytes(&self) -> Vec<u8> {
        let mut m = BTreeMap::new();
        m.insert(
            "protocol_version".into(),
            JsonValue::Uint(self.protocol_version),
        );
        m.insert("quote_id".into(), JsonValue::Str(self.quote_id.clone()));
        m.insert(
            "provider_id".into(),
            JsonValue::Str(self.provider_id.clone()),
        );
        m.insert(
            "provider_public_key_hex".into(),
            JsonValue::Str(self.provider_public_key_hex.clone()),
        );
        m.insert("tenant_id".into(), JsonValue::Str(self.tenant_id.clone()));
        m.insert("model_id".into(), JsonValue::Str(self.model_id.clone()));
        m.insert("max_tokens".into(), JsonValue::Uint(self.max_tokens as u64));
        m.insert(
            "price_micro_usd_per_1m".into(),
            JsonValue::Uint(self.price_micro_usd_per_1m),
        );
        m.insert(
            "vram_reserved_mib".into(),
            JsonValue::Uint(self.vram_reserved_mib),
        );
        m.insert("issued_at_ms".into(), JsonValue::Uint(self.issued_at_ms));
        m.insert("expires_at_ms".into(), JsonValue::Uint(self.expires_at_ms));
        JsonValue::Object(m).to_compact_string().into_bytes()
    }

    pub fn digest_hex(&self) -> String {
        sha256_hex(&self.canonical_payload_bytes())
    }

    pub fn verify(&self, now_ms: u64) -> Result<(), String> {
        if self.protocol_version != 1 {
            return Err("unsupported quote protocol_version".into());
        }
        if !verify_provider_id_binding(&self.provider_id, &self.provider_public_key_hex) {
            return Err("quote provider_id does not match provider_public_key_hex".into());
        }
        if now_ms > self.expires_at_ms {
            return Err("quote expired".into());
        }
        if !verify_domain_hex(
            &self.provider_public_key_hex,
            DOMAIN_QUOTE,
            &self.canonical_payload_bytes(),
            &self.signature_hex,
        ) {
            return Err("invalid Ed25519 signature on quote".into());
        }
        Ok(())
    }

    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "protocol_version".into(),
            JsonValue::Uint(self.protocol_version),
        );
        m.insert("quote_id".into(), JsonValue::Str(self.quote_id.clone()));
        m.insert(
            "provider_id".into(),
            JsonValue::Str(self.provider_id.clone()),
        );
        m.insert(
            "provider_public_key_hex".into(),
            JsonValue::Str(self.provider_public_key_hex.clone()),
        );
        m.insert("tenant_id".into(), JsonValue::Str(self.tenant_id.clone()));
        m.insert("model_id".into(), JsonValue::Str(self.model_id.clone()));
        m.insert("max_tokens".into(), JsonValue::Uint(self.max_tokens as u64));
        m.insert(
            "price_micro_usd_per_1m".into(),
            JsonValue::Uint(self.price_micro_usd_per_1m),
        );
        m.insert(
            "vram_reserved_mib".into(),
            JsonValue::Uint(self.vram_reserved_mib),
        );
        m.insert("issued_at_ms".into(), JsonValue::Uint(self.issued_at_ms));
        m.insert("expires_at_ms".into(), JsonValue::Uint(self.expires_at_ms));
        m.insert(
            "signature_hex".into(),
            JsonValue::Str(self.signature_hex.clone()),
        );
        JsonValue::Object(m)
    }

    pub fn from_json(v: &JsonValue) -> Result<Self, JsonError> {
        Ok(Self {
            protocol_version: v.get_u64("protocol_version")?,
            quote_id: v.get_str("quote_id")?.to_owned(),
            provider_id: v.get_str("provider_id")?.to_owned(),
            provider_public_key_hex: v.get_str("provider_public_key_hex")?.to_owned(),
            tenant_id: v.get_str("tenant_id")?.to_owned(),
            model_id: v.get_str("model_id")?.to_owned(),
            max_tokens: v.get_u64("max_tokens")? as u32,
            price_micro_usd_per_1m: v.get_u64("price_micro_usd_per_1m")?,
            vram_reserved_mib: v.get_u64("vram_reserved_mib")?,
            issued_at_ms: v.get_u64("issued_at_ms")?,
            expires_at_ms: v.get_u64("expires_at_ms")?,
            signature_hex: v.get_str("signature_hex")?.to_owned(),
        })
    }
}

/// Tenant-signed job execution lease bound to a specific [`SignedQuote`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedJobLease {
    pub protocol_version: u64,
    pub job_id: String,
    pub tenant_id: String,
    pub tenant_public_key_hex: String,
    pub provider_id: String,
    pub quote_digest_hex: String,
    pub model_id: String,
    pub max_tokens: u32,
    pub price_micro_usd_per_1m: u64,
    pub lease_nonce: u64,
    pub prompt: String,
    pub issued_at_ms: u64,
    pub tenant_signature_hex: String,
}

impl SignedJobLease {
    pub fn canonical_payload_bytes(&self) -> Vec<u8> {
        let mut m = BTreeMap::new();
        m.insert(
            "protocol_version".into(),
            JsonValue::Uint(self.protocol_version),
        );
        m.insert("job_id".into(), JsonValue::Str(self.job_id.clone()));
        m.insert("tenant_id".into(), JsonValue::Str(self.tenant_id.clone()));
        m.insert(
            "tenant_public_key_hex".into(),
            JsonValue::Str(self.tenant_public_key_hex.clone()),
        );
        m.insert(
            "provider_id".into(),
            JsonValue::Str(self.provider_id.clone()),
        );
        m.insert(
            "quote_digest_hex".into(),
            JsonValue::Str(self.quote_digest_hex.clone()),
        );
        m.insert("model_id".into(), JsonValue::Str(self.model_id.clone()));
        m.insert("max_tokens".into(), JsonValue::Uint(self.max_tokens as u64));
        m.insert(
            "price_micro_usd_per_1m".into(),
            JsonValue::Uint(self.price_micro_usd_per_1m),
        );
        m.insert("lease_nonce".into(), JsonValue::Uint(self.lease_nonce));
        m.insert(
            "prompt_sha256".into(),
            JsonValue::Str(sha256_hex(self.prompt.as_bytes())),
        );
        m.insert("issued_at_ms".into(), JsonValue::Uint(self.issued_at_ms));
        JsonValue::Object(m).to_compact_string().into_bytes()
    }

    pub fn sign_for_quote(
        tenant: &TenantKeyPair,
        quote: &SignedQuote,
        job_id: impl Into<String>,
        lease_nonce: u64,
        prompt: impl Into<String>,
        issued_at_ms: u64,
    ) -> Self {
        let mut lease = Self {
            protocol_version: 1,
            job_id: job_id.into(),
            tenant_id: tenant.tenant_id().to_owned(),
            tenant_public_key_hex: tenant.public_key_hex(),
            provider_id: quote.provider_id.clone(),
            quote_digest_hex: quote.digest_hex(),
            model_id: quote.model_id.clone(),
            max_tokens: quote.max_tokens,
            price_micro_usd_per_1m: quote.price_micro_usd_per_1m,
            lease_nonce,
            prompt: prompt.into(),
            issued_at_ms,
            tenant_signature_hex: String::new(),
        };
        let sig = tenant.sign_domain(DOMAIN_LEASE, &lease.canonical_payload_bytes());
        lease.tenant_signature_hex = hex_encode(&sig);
        lease
    }

    pub fn verify_tenant_signature(&self) -> Result<(), String> {
        if self.protocol_version != 1 {
            return Err("unsupported lease protocol_version".into());
        }
        if !verify_domain_hex(
            &self.tenant_public_key_hex,
            DOMAIN_LEASE,
            &self.canonical_payload_bytes(),
            &self.tenant_signature_hex,
        ) {
            return Err("invalid Ed25519 tenant signature on job lease".into());
        }
        Ok(())
    }

    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "protocol_version".into(),
            JsonValue::Uint(self.protocol_version),
        );
        m.insert("job_id".into(), JsonValue::Str(self.job_id.clone()));
        m.insert("tenant_id".into(), JsonValue::Str(self.tenant_id.clone()));
        m.insert(
            "tenant_public_key_hex".into(),
            JsonValue::Str(self.tenant_public_key_hex.clone()),
        );
        m.insert(
            "provider_id".into(),
            JsonValue::Str(self.provider_id.clone()),
        );
        m.insert(
            "quote_digest_hex".into(),
            JsonValue::Str(self.quote_digest_hex.clone()),
        );
        m.insert("model_id".into(), JsonValue::Str(self.model_id.clone()));
        m.insert("max_tokens".into(), JsonValue::Uint(self.max_tokens as u64));
        m.insert(
            "price_micro_usd_per_1m".into(),
            JsonValue::Uint(self.price_micro_usd_per_1m),
        );
        m.insert("lease_nonce".into(), JsonValue::Uint(self.lease_nonce));
        m.insert("prompt".into(), JsonValue::Str(self.prompt.clone()));
        m.insert("issued_at_ms".into(), JsonValue::Uint(self.issued_at_ms));
        m.insert(
            "tenant_signature_hex".into(),
            JsonValue::Str(self.tenant_signature_hex.clone()),
        );
        JsonValue::Object(m)
    }

    pub fn from_json(v: &JsonValue) -> Result<Self, JsonError> {
        Ok(Self {
            protocol_version: v.get_u64("protocol_version")?,
            job_id: v.get_str("job_id")?.to_owned(),
            tenant_id: v.get_str("tenant_id")?.to_owned(),
            tenant_public_key_hex: v.get_str("tenant_public_key_hex")?.to_owned(),
            provider_id: v.get_str("provider_id")?.to_owned(),
            quote_digest_hex: v.get_str("quote_digest_hex")?.to_owned(),
            model_id: v.get_str("model_id")?.to_owned(),
            max_tokens: v.get_u64("max_tokens")? as u32,
            price_micro_usd_per_1m: v.get_u64("price_micro_usd_per_1m")?,
            lease_nonce: v.get_u64("lease_nonce")?,
            prompt: v.get_str("prompt")?.to_owned(),
            issued_at_ms: v.get_u64("issued_at_ms")?,
            tenant_signature_hex: v.get_str("tenant_signature_hex")?.to_owned(),
        })
    }
}

/// Output of an inference execution backend.
#[derive(Clone, Debug)]
pub struct BackendOutput {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub ttft_us: u64,
}

/// Pluggable inference engine seam (`hanzo-engine` hooks in here with zero protocol churn).
pub trait InferenceBackend: Send + Sync {
    fn execute(
        &self,
        model_id: &str,
        prompt: &str,
        max_tokens: u32,
        guard: &CapacityGuard,
    ) -> Result<BackendOutput, String>;
}

/// Deterministic, CPU-safe inference backend for reproducible testing and
/// loopback-network benchmarking without requiring proprietary GPU weights.
#[derive(Clone, Debug)]
pub struct DeterministicEngineBackend {
    /// Simulated per-token decode time in microseconds (0 for max-speed throughput tests).
    pub per_token_delay_us: u64,
    /// Simulated prefill delay in microseconds.
    pub prefill_delay_us: u64,
}

impl Default for DeterministicEngineBackend {
    fn default() -> Self {
        Self {
            per_token_delay_us: 15,
            prefill_delay_us: 200,
        }
    }
}

impl InferenceBackend for DeterministicEngineBackend {
    fn execute(
        &self,
        model_id: &str,
        prompt: &str,
        max_tokens: u32,
        guard: &CapacityGuard,
    ) -> Result<BackendOutput, String> {
        let start = Instant::now();
        let prompt_tokens = estimate_tokens(prompt).max(1);
        let gen_tokens = (max_tokens as u64).clamp(1, 256);

        // Write tenant-scoped prompt state into isolated scratch buffer and verify ownership.
        let kv_seed = sha256_hex(
            format!("{}|{}|{}", guard.namespace().cache_salt(), model_id, prompt).as_bytes(),
        );
        guard
            .write_scratch(kv_seed.as_bytes())
            .map_err(|e| e.to_string())?;
        let readback = guard.read_scratch().map_err(|e| e.to_string())?;
        if !readback.starts_with(kv_seed.as_bytes()) {
            return Err("isolated scratch readback integrity failure".into());
        }

        if self.prefill_delay_us > 0 {
            std::thread::sleep(Duration::from_micros(self.prefill_delay_us));
        }

        let mut tokens = Vec::with_capacity(gen_tokens as usize);
        let mut ttft_us = 0u64;
        let mut rolling = kv_seed;
        for step in 0..gen_tokens {
            rolling = sha256_hex(format!("{rolling}:{step}").as_bytes());
            tokens.push(format!("tok_{}", &rolling[..6]));
            if step == 0 {
                ttft_us = start.elapsed().as_micros().max(1) as u64;
            }
            if self.per_token_delay_us > 0 && step < 16 {
                std::thread::sleep(Duration::from_micros(self.per_token_delay_us));
            }
        }
        let text = tokens.join(" ");

        Ok(BackendOutput {
            text,
            prompt_tokens,
            completion_tokens: gen_tokens,
            ttft_us,
        })
    }
}

/// Complete response returned by [`ProviderDaemon::execute_job`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobExecutionResult {
    pub output_text: String,
    pub cache_salt: String,
    pub ledger_entry_hash_hex: String,
    pub receipt: CompletionReceipt,
}

impl JobExecutionResult {
    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "output_text".into(),
            JsonValue::Str(self.output_text.clone()),
        );
        m.insert("cache_salt".into(), JsonValue::Str(self.cache_salt.clone()));
        m.insert(
            "ledger_entry_hash_hex".into(),
            JsonValue::Str(self.ledger_entry_hash_hex.clone()),
        );
        m.insert("receipt".into(), self.receipt.to_json());
        JsonValue::Object(m)
    }

    pub fn from_json(v: &JsonValue) -> Result<Self, JsonError> {
        let receipt_v = v
            .get("receipt")
            .ok_or_else(|| JsonError("missing receipt".into()))?;
        Ok(Self {
            output_text: v.get_str("output_text")?.to_owned(),
            cache_salt: v.get_str("cache_salt")?.to_owned(),
            ledger_entry_hash_hex: v.get_str("ledger_entry_hash_hex")?.to_owned(),
            receipt: CompletionReceipt::from_json(receipt_v)?,
        })
    }
}

/// Structured provider error mapped cleanly onto HTTP status codes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DaemonError {
    BadRequest(String),
    Unauthorized(String),
    Forbidden(String),
    ConflictReplay(String),
    PayloadTooLarge(String),
    CapacityExhausted(String),
    Draining(String),
    Internal(String),
}

impl DaemonError {
    pub fn http_status(&self) -> (u16, &'static str) {
        match self {
            DaemonError::BadRequest(_) => (400, "Bad Request"),
            DaemonError::Unauthorized(_) => (401, "Unauthorized"),
            DaemonError::Forbidden(_) => (403, "Forbidden"),
            DaemonError::ConflictReplay(_) => (409, "Conflict"),
            DaemonError::PayloadTooLarge(_) => (413, "Payload Too Large"),
            DaemonError::CapacityExhausted(_) => (429, "Too Many Requests"),
            DaemonError::Draining(_) => (503, "Service Unavailable"),
            DaemonError::Internal(_) => (500, "Internal Server Error"),
        }
    }
}

impl fmt::Display for DaemonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DaemonError::BadRequest(s)
            | DaemonError::Unauthorized(s)
            | DaemonError::Forbidden(s)
            | DaemonError::ConflictReplay(s)
            | DaemonError::PayloadTooLarge(s)
            | DaemonError::CapacityExhausted(s)
            | DaemonError::Draining(s)
            | DaemonError::Internal(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for DaemonError {}

/// Configuration for launching a permissionless [`ProviderDaemon`].
#[derive(Clone, Debug)]
pub struct ProviderConfig {
    pub endpoint: String,
    pub devices: Vec<GpuDevice>,
    pub models: Vec<ModelCapability>,
    pub max_concurrency: u32,
    pub heartbeat_ttl_secs: u64,
    pub quote_ttl_secs: u64,
    pub sandbox_policy: SandboxExecutionPolicy,
}

/// Production-grade permissionless GPU provider daemon.
pub struct ProviderDaemon {
    keypair: ProviderKeyPair,
    config: ProviderConfig,
    capacity: CapacityPool,
    ledger: SovereignLedger,
    backend: Arc<dyn InferenceBackend>,
    announcement_seq: AtomicU64,
    quote_counter: AtomicU64,
    in_flight_nonces: Mutex<BTreeSet<u64>>,
    hanzo_control_plane_calls: AtomicU64,
}

impl ProviderDaemon {
    /// Creates a new provider daemon backed by an in-memory sovereign ledger.
    pub fn new(
        keypair: ProviderKeyPair,
        config: ProviderConfig,
        backend: Arc<dyn InferenceBackend>,
    ) -> Self {
        let total_vram: u64 = config.devices.iter().map(|d| d.vram_usable_mib).sum();
        let capacity = CapacityPool::new(total_vram.max(1_024), config.max_concurrency);
        Self {
            keypair,
            config,
            capacity,
            ledger: SovereignLedger::in_memory(),
            backend,
            announcement_seq: AtomicU64::new(0),
            quote_counter: AtomicU64::new(0),
            in_flight_nonces: Mutex::new(BTreeSet::new()),
            hanzo_control_plane_calls: AtomicU64::new(0),
        }
    }

    /// Creates or restarts a provider daemon from a persistent sovereign JSONL ledger on disk.
    ///
    /// Automatically replays and verifies all historical receipts & nonces so the
    /// provider returns to service after a restart with zero state loss and full replay protection.
    pub fn open_with_ledger(
        keypair: ProviderKeyPair,
        config: ProviderConfig,
        ledger_path: impl AsRef<Path>,
        fsync_on_append: bool,
        backend: Arc<dyn InferenceBackend>,
    ) -> Result<Self, String> {
        let ledger = SovereignLedger::open_or_create(
            ledger_path,
            Some(keypair.provider_id().as_str()),
            fsync_on_append,
        )?;
        let summary = ledger.summary();
        let total_vram: u64 = config.devices.iter().map(|d| d.vram_usable_mib).sum();
        let capacity = CapacityPool::new(total_vram.max(1_024), config.max_concurrency);
        // Resume announcement sequence monotonically above recovered entry count.
        let initial_seq = summary.entry_count;
        Ok(Self {
            keypair,
            config,
            capacity,
            ledger,
            backend,
            announcement_seq: AtomicU64::new(initial_seq),
            quote_counter: AtomicU64::new(summary.entry_count),
            in_flight_nonces: Mutex::new(BTreeSet::new()),
            hanzo_control_plane_calls: AtomicU64::new(0),
        })
    }

    pub fn provider_id(&self) -> &str {
        self.keypair.provider_id().as_str()
    }

    pub fn public_key_hex(&self) -> String {
        self.keypair.public_key_hex()
    }

    pub fn keypair(&self) -> &ProviderKeyPair {
        &self.keypair
    }

    pub fn capacity_pool(&self) -> &CapacityPool {
        &self.capacity
    }

    pub fn capacity_snapshot(&self) -> CapacitySnapshot {
        self.capacity.snapshot()
    }

    pub fn ledger(&self) -> &SovereignLedger {
        &self.ledger
    }

    pub fn ledger_summary(&self) -> LedgerSummary {
        self.ledger.summary()
    }

    pub fn export_ledger_jsonl(&self) -> String {
        self.ledger.export_jsonl()
    }

    pub fn hanzo_control_plane_calls(&self) -> u64 {
        self.hanzo_control_plane_calls.load(Ordering::Relaxed)
    }

    /// Generates a freshly signed capability announcement with a monotonic sequence number.
    pub fn signed_announcement(&self, now_ms: u64) -> CapabilityAnnouncement {
        let seq = self.announcement_seq.fetch_add(1, Ordering::SeqCst) + 1;
        let nonce = now_ms ^ (seq.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let draining = self.capacity.snapshot().draining;
        CapabilityAnnouncement::sign_new(
            &self.keypair,
            self.config.endpoint.clone(),
            self.config.devices.clone(),
            self.config.models.clone(),
            self.config.max_concurrency,
            self.config.heartbeat_ttl_secs,
            seq,
            nonce,
            now_ms,
            draining,
        )
    }

    /// Initiates graceful drain: rejects new quotes & leases while completing all in-flight work.
    pub fn initiate_drain(&self) {
        self.capacity.initiate_drain();
    }

    /// Waits for all in-flight jobs to complete during graceful drain.
    pub fn wait_until_drained(&self, timeout: Duration) -> bool {
        self.capacity.wait_until_drained(timeout)
    }

    /// Resumes service after drain or maintenance.
    pub fn resume_service(&self) {
        self.capacity.resume_service();
    }

    /// Issues a signed capacity & pricing quote to a tenant.
    pub fn issue_quote(&self, req: &QuoteRequest, now_ms: u64) -> Result<SignedQuote, DaemonError> {
        if self.capacity.snapshot().draining {
            return Err(DaemonError::Draining(
                "provider is draining; refusing new quote".into(),
            ));
        }
        validate_identifier("tenant_id", &req.tenant_id)
            .map_err(|e| DaemonError::BadRequest(e.to_string()))?;
        let Some(model) = self
            .config
            .models
            .iter()
            .find(|m| m.model_id == req.model_id)
        else {
            return Err(DaemonError::BadRequest(format!(
                "unsupported model_id `{}`",
                req.model_id
            )));
        };
        if req.max_tokens == 0 || req.max_tokens > self.config.sandbox_policy.max_output_tokens {
            return Err(DaemonError::BadRequest(format!(
                "max_tokens {} outside policy bounds 1..={}",
                req.max_tokens, self.config.sandbox_policy.max_output_tokens
            )));
        }
        if req
            .prompt_tokens_estimate
            .saturating_add(req.max_tokens as u64)
            > model.max_context_tokens as u64
        {
            return Err(DaemonError::BadRequest(format!(
                "requested context exceeds model max_context_tokens ({})",
                model.max_context_tokens
            )));
        }

        let seq = self.quote_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let quote_id = format!("q_{}_{seq}", &self.provider_id()[5..13]);
        let expires_at_ms = now_ms.saturating_add(self.config.quote_ttl_secs.saturating_mul(1_000));
        let mut quote = SignedQuote {
            protocol_version: 1,
            quote_id,
            provider_id: self.provider_id().to_owned(),
            provider_public_key_hex: self.public_key_hex(),
            tenant_id: req.tenant_id.clone(),
            model_id: model.model_id.clone(),
            max_tokens: req.max_tokens,
            price_micro_usd_per_1m: model.price_micro_usd_per_1m,
            vram_reserved_mib: model.vram_required_mib,
            issued_at_ms: now_ms,
            expires_at_ms,
            signature_hex: String::new(),
        };
        let sig = self
            .keypair
            .sign_domain(DOMAIN_QUOTE, &quote.canonical_payload_bytes());
        quote.signature_hex = hex_encode(&sig);
        Ok(quote)
    }

    /// Executes a tenant-signed job lease after strict cryptographic, budget,
    /// replay, isolation, and capacity checks, returning a signed [`CompletionReceipt`].
    pub fn execute_job(
        &self,
        quote: &SignedQuote,
        lease: &SignedJobLease,
        now_ms: u64,
    ) -> Result<JobExecutionResult, DaemonError> {
        let start = Instant::now();

        // 1. Drain gate
        if self.capacity.snapshot().draining {
            return Err(DaemonError::Draining(
                "provider is draining; rejecting new job lease".into(),
            ));
        }

        // 2. Strict tenant & job namespace validation (zero path traversal / control chars)
        let ns = TenantNamespace::new(&lease.tenant_id, &lease.job_id)
            .map_err(|e| DaemonError::BadRequest(e.to_string()))?;

        // 3. Prompt size & NUL-byte safety guard
        if lease.prompt.len() > self.config.sandbox_policy.max_prompt_bytes {
            return Err(DaemonError::PayloadTooLarge(format!(
                "prompt exceeds max_prompt_bytes ({} > {})",
                lease.prompt.len(),
                self.config.sandbox_policy.max_prompt_bytes
            )));
        }
        if lease.prompt.as_bytes().contains(&0) {
            return Err(DaemonError::BadRequest(
                "prompt contains disallowed NUL byte".into(),
            ));
        }

        // 4. Target provider binding (blocks cross-provider lease redirection)
        if quote.provider_id != self.provider_id() || lease.provider_id != self.provider_id() {
            return Err(DaemonError::Forbidden(
                "quote or lease is bound to a different provider_id".into(),
            ));
        }

        // 5. Cryptographic quote verification (Ed25519 + TTL)
        quote.verify(now_ms).map_err(DaemonError::Unauthorized)?;

        // 6. Strict quote-to-lease field binding (blocks tampered price/model/max_tokens)
        if lease.tenant_id != quote.tenant_id
            || lease.model_id != quote.model_id
            || lease.max_tokens != quote.max_tokens
            || lease.price_micro_usd_per_1m != quote.price_micro_usd_per_1m
            || lease.quote_digest_hex != quote.digest_hex()
        {
            return Err(DaemonError::BadRequest(
                "job lease does not match signed quote terms or digest".into(),
            ));
        }

        // 7. Cryptographic tenant lease signature verification (Ed25519)
        lease
            .verify_tenant_signature()
            .map_err(DaemonError::Unauthorized)?;

        // 8. Model capability & context budget enforcement
        let Some(model) = self
            .config
            .models
            .iter()
            .find(|m| m.model_id == lease.model_id)
        else {
            return Err(DaemonError::BadRequest(format!(
                "model `{}` not loaded on provider",
                lease.model_id
            )));
        };
        let prompt_tokens_est = estimate_tokens(&lease.prompt).max(1);
        if lease.max_tokens == 0
            || lease.max_tokens > self.config.sandbox_policy.max_output_tokens
            || prompt_tokens_est.saturating_add(lease.max_tokens as u64)
                > model.max_context_tokens as u64
        {
            return Err(DaemonError::BadRequest(
                "prompt + max_tokens exceeds model context or sandbox budget".into(),
            ));
        }

        // 9. Replay protection (atomic check across persisted ledger + in-flight leases)
        {
            let mut inflight = self
                .in_flight_nonces
                .lock()
                .expect("in_flight_nonces mutex poisoned");
            if self.ledger.has_seen_nonce(lease.lease_nonce)
                || inflight.contains(&lease.lease_nonce)
            {
                return Err(DaemonError::ConflictReplay(format!(
                    "replayed lease_nonce {}",
                    lease.lease_nonce
                )));
            }
            inflight.insert(lease.lease_nonce);
        }

        // Helper closure to release in-flight nonce on early error before ledger commit.
        let release_inflight = |nonce: u64| {
            let mut inflight = self
                .in_flight_nonces
                .lock()
                .expect("in_flight_nonces mutex poisoned");
            inflight.remove(&nonce);
        };

        // 10. Strict VRAM + concurrency + per-tenant scratch reservation (zero OOM thrash)
        let scratch_bytes = 256;
        let guard =
            match self
                .capacity
                .try_reserve(ns.clone(), model.vram_required_mib, scratch_bytes)
            {
                Ok(g) => g,
                Err(IsolationError::CapacityExhausted { .. }) => {
                    release_inflight(lease.lease_nonce);
                    return Err(DaemonError::CapacityExhausted(
                        "provider VRAM or concurrency capacity exhausted".into(),
                    ));
                }
                Err(IsolationError::ProviderDraining) => {
                    release_inflight(lease.lease_nonce);
                    return Err(DaemonError::Draining("provider entered drain mode".into()));
                }
                Err(e) => {
                    release_inflight(lease.lease_nonce);
                    return Err(DaemonError::BadRequest(e.to_string()));
                }
            };

        // 11. Isolated execution
        let input_digest_hex = sha256_hex(lease.prompt.as_bytes());
        let output =
            match self
                .backend
                .execute(&lease.model_id, &lease.prompt, lease.max_tokens, &guard)
            {
                Ok(out) => out,
                Err(e) => {
                    drop(guard);
                    release_inflight(lease.lease_nonce);
                    return Err(DaemonError::Internal(format!(
                        "inference backend error: {e}"
                    )));
                }
            };
        let output_digest_hex = sha256_hex(output.text.as_bytes());
        let total_us = start.elapsed().as_micros().max(output.ttft_us as u128) as u64;

        // 12. Sign CompletionReceipt and append durably into sovereign hash-chained ledger
        let receipt_seq = self.ledger.next_receipt_seq();
        let receipt = CompletionReceipt::sign_new(
            &self.keypair,
            receipt_seq,
            lease.job_id.clone(),
            lease.tenant_id.clone(),
            lease.model_id.clone(),
            quote.digest_hex(),
            input_digest_hex,
            output_digest_hex,
            output.prompt_tokens,
            output.completion_tokens,
            quote.price_micro_usd_per_1m,
            output.ttft_us,
            total_us,
            now_unix_ms().max(now_ms),
        );
        let entry = match self.ledger.append(lease.lease_nonce, receipt.clone()) {
            Ok(e) => e,
            Err(e) => {
                drop(guard);
                release_inflight(lease.lease_nonce);
                return Err(DaemonError::Internal(format!("ledger append failed: {e}")));
            }
        };

        // Release in-flight tracker (nonce is now permanently recorded in ledger) & zeroize scratch.
        release_inflight(lease.lease_nonce);
        let cache_salt = guard.namespace().cache_salt();
        drop(guard);

        Ok(JobExecutionResult {
            output_text: output.text,
            cache_salt,
            ledger_entry_hash_hex: entry.entry_hash_hex,
            receipt,
        })
    }
}

/// Negative-regime control #2: Unverified, unmetered execution daemon.
///
/// Deliberately omits Ed25519 quote/lease verification, replay protection, and
/// signed receipt issuance in order to measure both the exact verification CPU
/// overhead AND the resulting 100% revenue/fraud leakage under adversarial traffic.
#[derive(Debug)]
pub struct UnverifiedControlDaemon {
    backend: DeterministicEngineBackend,
    pool: CapacityPool,
    jobs_executed: AtomicU64,
    unpaid_or_forged_executed: AtomicU64,
}

impl UnverifiedControlDaemon {
    pub fn new(
        total_vram_mib: u64,
        max_concurrency: u32,
        backend: DeterministicEngineBackend,
    ) -> Self {
        Self {
            backend,
            pool: CapacityPool::new(total_vram_mib, max_concurrency),
            jobs_executed: AtomicU64::new(0),
            unpaid_or_forged_executed: AtomicU64::new(0),
        }
    }

    /// Executes any incoming prompt without checking signatures, quotes, or replay nonces.
    pub fn execute_unverified(
        &self,
        job_id: &str,
        model_id: &str,
        prompt: &str,
        max_tokens: u32,
        is_adversarial_sample: bool,
    ) -> Result<(String, u64, u64), String> {
        let start = Instant::now();
        let ns = TenantNamespace::new("unverified", job_id).map_err(|e| e.to_string())?;
        let guard = self
            .pool
            .try_reserve(ns, 1_024, 128)
            .map_err(|e| e.to_string())?;
        let out = self.backend.execute(model_id, prompt, max_tokens, &guard)?;
        self.jobs_executed.fetch_add(1, Ordering::Relaxed);
        if is_adversarial_sample {
            self.unpaid_or_forged_executed
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok((out.text, out.ttft_us, start.elapsed().as_micros() as u64))
    }

    pub fn jobs_executed(&self) -> u64 {
        self.jobs_executed.load(Ordering::Relaxed)
    }

    pub fn unpaid_or_forged_executed(&self) -> u64 {
        self.unpaid_or_forged_executed.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advertise::HardwareBackend;

    fn sample_daemon(label: &str) -> ProviderDaemon {
        let kp = ProviderKeyPair::from_label(label);
        let cfg = ProviderConfig {
            endpoint: "http://127.0.0.1:0".into(),
            devices: vec![GpuDevice {
                backend: HardwareBackend::Cuda,
                name: "RTX 4090".into(),
                vram_total_mib: 24_576,
                vram_usable_mib: 22_528,
                compute_units: 128,
                device_id: "cuda:0".into(),
            }],
            models: vec![ModelCapability {
                model_id: "zen-8b-q4".into(),
                quantization: "Q4_K_M".into(),
                vram_required_mib: 6_144,
                max_context_tokens: 8_192,
                price_micro_usd_per_1m: 150_000,
            }],
            max_concurrency: 4,
            heartbeat_ttl_secs: 60,
            quote_ttl_secs: 30,
            sandbox_policy: SandboxExecutionPolicy::default(),
        };
        ProviderDaemon::new(
            kp,
            cfg,
            Arc::new(DeterministicEngineBackend {
                per_token_delay_us: 0,
                prefill_delay_us: 0,
            }),
        )
    }

    #[test]
    fn happy_path_quote_lease_execution_and_replay_rejection() {
        let daemon = sample_daemon("unit-test-provider");
        let tenant = TenantKeyPair::from_label("tenant-alpha");
        let now = 1_700_000_000_000;

        let quote = daemon
            .issue_quote(
                &QuoteRequest {
                    tenant_id: tenant.tenant_id().into(),
                    model_id: "zen-8b-q4".into(),
                    prompt_tokens_estimate: 16,
                    max_tokens: 32,
                },
                now,
            )
            .unwrap();
        let lease = SignedJobLease::sign_for_quote(
            &tenant,
            &quote,
            "job-001",
            42_001,
            "Explain decentralized GPU inference.",
            now,
        );

        let res = daemon.execute_job(&quote, &lease, now + 100).unwrap();
        res.receipt.verify().unwrap();
        assert_eq!(res.receipt.completion_tokens, 32);
        assert_eq!(res.receipt.tenant_id, "tenant-alpha");
        assert_eq!(daemon.hanzo_control_plane_calls(), 0);

        // Replaying the exact same lease nonce must return ConflictReplay (409).
        assert!(matches!(
            daemon.execute_job(&quote, &lease, now + 200),
            Err(DaemonError::ConflictReplay(_))
        ));
    }

    #[test]
    fn rejects_tampered_price_expired_quote_and_forged_signature() {
        let daemon = sample_daemon("unit-test-adversarial");
        let tenant = TenantKeyPair::from_label("tenant-beta");
        let now = 1_700_000_000_000;

        let quote = daemon
            .issue_quote(
                &QuoteRequest {
                    tenant_id: tenant.tenant_id().into(),
                    model_id: "zen-8b-q4".into(),
                    prompt_tokens_estimate: 16,
                    max_tokens: 32,
                },
                now,
            )
            .unwrap();

        // 1. Expired quote
        let lease = SignedJobLease::sign_for_quote(&tenant, &quote, "j-exp", 1, "hello", now);
        assert!(matches!(
            daemon.execute_job(&quote, &lease, quote.expires_at_ms + 1),
            Err(DaemonError::Unauthorized(_))
        ));

        // 2. Tampered price on lease
        let mut tampered_lease =
            SignedJobLease::sign_for_quote(&tenant, &quote, "j-price", 2, "hello", now);
        tampered_lease.price_micro_usd_per_1m = 1;
        assert!(matches!(
            daemon.execute_job(&quote, &tampered_lease, now + 10),
            Err(DaemonError::BadRequest(_))
        ));

        // 3. Forged tenant signature
        let mut forged =
            SignedJobLease::sign_for_quote(&tenant, &quote, "j-forged", 3, "hello", now);
        forged.prompt = "injected prompt".into();
        assert!(matches!(
            daemon.execute_job(&quote, &forged, now + 10),
            Err(DaemonError::Unauthorized(_))
        ));
    }
}
