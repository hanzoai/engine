//! Sovereign hash-chained JSONL receipt ledger with Ed25519 verification,
//! crash-safe persistence (`sync_all`), and 100% lossless export/migration/restart replay.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::crypto::{hex_encode, sha256_hex};
use crate::identity::{
    verify_domain_hex, verify_provider_id_binding, ProviderKeyPair, DOMAIN_RECEIPT,
};
use crate::json::{parse_json, JsonError, JsonValue};

/// Genesis previous-entry hash for the first entry in any sovereign ledger.
pub const GENESIS_HASH_HEX: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Signed completion receipt proving metered execution of a leased inference job.
///
/// Bound cryptographically via RFC 8032 Ed25519 to the provider's identity,
/// the pre-agreed quote digest, input/output SHA-256 digests, token counts,
/// and latency telemetry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionReceipt {
    pub protocol_version: u64,
    pub receipt_seq: u64,
    pub job_id: String,
    pub tenant_id: String,
    pub provider_id: String,
    pub provider_public_key_hex: String,
    pub model_id: String,
    pub quote_digest_hex: String,
    pub input_digest_hex: String,
    pub output_digest_hex: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub price_micro_usd_per_1m: u64,
    pub total_cost_micro_usd: u64,
    pub ttft_us: u64,
    pub total_us: u64,
    pub completed_at_ms: u64,
    pub signature_hex: String,
}

impl CompletionReceipt {
    /// Canonical JSON bytes signed by the provider (excludes `signature_hex`).
    pub fn canonical_payload_bytes(&self) -> Vec<u8> {
        let mut m = BTreeMap::new();
        m.insert(
            "protocol_version".into(),
            JsonValue::Uint(self.protocol_version),
        );
        m.insert("receipt_seq".into(), JsonValue::Uint(self.receipt_seq));
        m.insert("job_id".into(), JsonValue::Str(self.job_id.clone()));
        m.insert("tenant_id".into(), JsonValue::Str(self.tenant_id.clone()));
        m.insert(
            "provider_id".into(),
            JsonValue::Str(self.provider_id.clone()),
        );
        m.insert(
            "provider_public_key_hex".into(),
            JsonValue::Str(self.provider_public_key_hex.clone()),
        );
        m.insert("model_id".into(), JsonValue::Str(self.model_id.clone()));
        m.insert(
            "quote_digest_hex".into(),
            JsonValue::Str(self.quote_digest_hex.clone()),
        );
        m.insert(
            "input_digest_hex".into(),
            JsonValue::Str(self.input_digest_hex.clone()),
        );
        m.insert(
            "output_digest_hex".into(),
            JsonValue::Str(self.output_digest_hex.clone()),
        );
        m.insert("prompt_tokens".into(), JsonValue::Uint(self.prompt_tokens));
        m.insert(
            "completion_tokens".into(),
            JsonValue::Uint(self.completion_tokens),
        );
        m.insert(
            "price_micro_usd_per_1m".into(),
            JsonValue::Uint(self.price_micro_usd_per_1m),
        );
        m.insert(
            "total_cost_micro_usd".into(),
            JsonValue::Uint(self.total_cost_micro_usd),
        );
        m.insert("ttft_us".into(), JsonValue::Uint(self.ttft_us));
        m.insert("total_us".into(), JsonValue::Uint(self.total_us));
        m.insert(
            "completed_at_ms".into(),
            JsonValue::Uint(self.completed_at_ms),
        );
        JsonValue::Object(m).to_compact_string().into_bytes()
    }

    /// Computes metered cost in micro-USD (ceiling integer arithmetic, zero float drift).
    pub fn compute_cost_micro_usd(
        prompt_tokens: u64,
        completion_tokens: u64,
        price_micro_usd_per_1m: u64,
    ) -> u64 {
        let total_tokens = (prompt_tokens as u128) + (completion_tokens as u128);
        let num = total_tokens
            .saturating_mul(price_micro_usd_per_1m as u128)
            .saturating_add(999_999);
        (num / 1_000_000) as u64
    }

    /// Signs a new receipt with the provider's Ed25519 keypair.
    #[allow(clippy::too_many_arguments)]
    pub fn sign_new(
        keypair: &ProviderKeyPair,
        receipt_seq: u64,
        job_id: impl Into<String>,
        tenant_id: impl Into<String>,
        model_id: impl Into<String>,
        quote_digest_hex: impl Into<String>,
        input_digest_hex: impl Into<String>,
        output_digest_hex: impl Into<String>,
        prompt_tokens: u64,
        completion_tokens: u64,
        price_micro_usd_per_1m: u64,
        ttft_us: u64,
        total_us: u64,
        completed_at_ms: u64,
    ) -> Self {
        let total_cost_micro_usd =
            Self::compute_cost_micro_usd(prompt_tokens, completion_tokens, price_micro_usd_per_1m);
        let mut r = Self {
            protocol_version: 1,
            receipt_seq,
            job_id: job_id.into(),
            tenant_id: tenant_id.into(),
            provider_id: keypair.provider_id().as_str().to_owned(),
            provider_public_key_hex: keypair.public_key_hex(),
            model_id: model_id.into(),
            quote_digest_hex: quote_digest_hex.into(),
            input_digest_hex: input_digest_hex.into(),
            output_digest_hex: output_digest_hex.into(),
            prompt_tokens,
            completion_tokens,
            price_micro_usd_per_1m,
            total_cost_micro_usd,
            ttft_us,
            total_us,
            completed_at_ms,
            signature_hex: String::new(),
        };
        let sig = keypair.sign_domain(DOMAIN_RECEIPT, &r.canonical_payload_bytes());
        r.signature_hex = hex_encode(&sig);
        r
    }

    /// Independently verifies provider identity binding, cost arithmetic, and Ed25519 signature.
    pub fn verify(&self) -> Result<(), String> {
        if self.protocol_version != 1 {
            return Err("unsupported receipt protocol_version".into());
        }
        if !verify_provider_id_binding(&self.provider_id, &self.provider_public_key_hex) {
            return Err("receipt provider_id does not match provider_public_key_hex".into());
        }
        let expected_cost = Self::compute_cost_micro_usd(
            self.prompt_tokens,
            self.completion_tokens,
            self.price_micro_usd_per_1m,
        );
        if self.total_cost_micro_usd != expected_cost {
            return Err(format!(
                "receipt total_cost_micro_usd mismatch ({} != {})",
                self.total_cost_micro_usd, expected_cost
            ));
        }
        if !verify_domain_hex(
            &self.provider_public_key_hex,
            DOMAIN_RECEIPT,
            &self.canonical_payload_bytes(),
            &self.signature_hex,
        ) {
            return Err("invalid Ed25519 signature on completion receipt".into());
        }
        Ok(())
    }

    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "protocol_version".into(),
            JsonValue::Uint(self.protocol_version),
        );
        m.insert("receipt_seq".into(), JsonValue::Uint(self.receipt_seq));
        m.insert("job_id".into(), JsonValue::Str(self.job_id.clone()));
        m.insert("tenant_id".into(), JsonValue::Str(self.tenant_id.clone()));
        m.insert(
            "provider_id".into(),
            JsonValue::Str(self.provider_id.clone()),
        );
        m.insert(
            "provider_public_key_hex".into(),
            JsonValue::Str(self.provider_public_key_hex.clone()),
        );
        m.insert("model_id".into(), JsonValue::Str(self.model_id.clone()));
        m.insert(
            "quote_digest_hex".into(),
            JsonValue::Str(self.quote_digest_hex.clone()),
        );
        m.insert(
            "input_digest_hex".into(),
            JsonValue::Str(self.input_digest_hex.clone()),
        );
        m.insert(
            "output_digest_hex".into(),
            JsonValue::Str(self.output_digest_hex.clone()),
        );
        m.insert("prompt_tokens".into(), JsonValue::Uint(self.prompt_tokens));
        m.insert(
            "completion_tokens".into(),
            JsonValue::Uint(self.completion_tokens),
        );
        m.insert(
            "price_micro_usd_per_1m".into(),
            JsonValue::Uint(self.price_micro_usd_per_1m),
        );
        m.insert(
            "total_cost_micro_usd".into(),
            JsonValue::Uint(self.total_cost_micro_usd),
        );
        m.insert("ttft_us".into(), JsonValue::Uint(self.ttft_us));
        m.insert("total_us".into(), JsonValue::Uint(self.total_us));
        m.insert(
            "completed_at_ms".into(),
            JsonValue::Uint(self.completed_at_ms),
        );
        m.insert(
            "signature_hex".into(),
            JsonValue::Str(self.signature_hex.clone()),
        );
        JsonValue::Object(m)
    }

    pub fn from_json(v: &JsonValue) -> Result<Self, JsonError> {
        Ok(Self {
            protocol_version: v.get_u64("protocol_version")?,
            receipt_seq: v.get_u64("receipt_seq")?,
            job_id: v.get_str("job_id")?.to_owned(),
            tenant_id: v.get_str("tenant_id")?.to_owned(),
            provider_id: v.get_str("provider_id")?.to_owned(),
            provider_public_key_hex: v.get_str("provider_public_key_hex")?.to_owned(),
            model_id: v.get_str("model_id")?.to_owned(),
            quote_digest_hex: v.get_str("quote_digest_hex")?.to_owned(),
            input_digest_hex: v.get_str("input_digest_hex")?.to_owned(),
            output_digest_hex: v.get_str("output_digest_hex")?.to_owned(),
            prompt_tokens: v.get_u64("prompt_tokens")?,
            completion_tokens: v.get_u64("completion_tokens")?,
            price_micro_usd_per_1m: v.get_u64("price_micro_usd_per_1m")?,
            total_cost_micro_usd: v.get_u64("total_cost_micro_usd")?,
            ttft_us: v.get_u64("ttft_us")?,
            total_us: v.get_u64("total_us")?,
            completed_at_ms: v.get_u64("completed_at_ms")?,
            signature_hex: v.get_str("signature_hex")?.to_owned(),
        })
    }
}

/// One hash-chained entry in the provider's sovereign JSONL ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerEntry {
    pub index: u64,
    pub prev_hash_hex: String,
    pub entry_hash_hex: String,
    pub lease_nonce: u64,
    pub receipt: CompletionReceipt,
}

impl LedgerEntry {
    pub fn compute_entry_hash(
        index: u64,
        prev_hash_hex: &str,
        lease_nonce: u64,
        receipt: &CompletionReceipt,
    ) -> String {
        let preimage = format!(
            "hanzo-ledger-entry:v1\0{index}\0{prev_hash_hex}\0{lease_nonce}\0{}",
            receipt.to_json().to_compact_string()
        );
        sha256_hex(preimage.as_bytes())
    }

    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("index".into(), JsonValue::Uint(self.index));
        m.insert(
            "prev_hash_hex".into(),
            JsonValue::Str(self.prev_hash_hex.clone()),
        );
        m.insert(
            "entry_hash_hex".into(),
            JsonValue::Str(self.entry_hash_hex.clone()),
        );
        m.insert("lease_nonce".into(), JsonValue::Uint(self.lease_nonce));
        m.insert("receipt".into(), self.receipt.to_json());
        JsonValue::Object(m)
    }

    pub fn from_json(v: &JsonValue) -> Result<Self, JsonError> {
        let receipt_val = v
            .get("receipt")
            .ok_or_else(|| JsonError("missing receipt object".into()))?;
        Ok(Self {
            index: v.get_u64("index")?,
            prev_hash_hex: v.get_str("prev_hash_hex")?.to_owned(),
            entry_hash_hex: v.get_str("entry_hash_hex")?.to_owned(),
            lease_nonce: v.get_u64("lease_nonce")?,
            receipt: CompletionReceipt::from_json(receipt_val)?,
        })
    }
}

/// Summary of verified sovereign ledger state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerSummary {
    pub entry_count: u64,
    pub head_hash_hex: String,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_earnings_micro_usd: u64,
    pub seen_nonces: Vec<u64>,
}

/// Verifies an exported JSONL sovereign ledger end-to-end:
/// - contiguous 0..N index ordering
/// - unbroken SHA-256 hash chain from `GENESIS_HASH_HEX`
/// - valid RFC 8032 Ed25519 signature & cost arithmetic on every receipt
/// - optional single-provider binding (`expected_provider_id`)
pub fn verify_jsonl_chain(
    jsonl: &str,
    expected_provider_id: Option<&str>,
) -> Result<(Vec<LedgerEntry>, LedgerSummary), String> {
    let mut entries = Vec::new();
    let mut prev_hash = GENESIS_HASH_HEX.to_owned();
    let mut total_prompt = 0u64;
    let mut total_completion = 0u64;
    let mut total_earnings = 0u64;
    let mut seen_nonces = Vec::new();

    for (line_no, raw_line) in jsonl.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let val = parse_json(line).map_err(|e| format!("line {}: {e}", line_no + 1))?;
        let entry =
            LedgerEntry::from_json(&val).map_err(|e| format!("line {}: {e}", line_no + 1))?;
        let expected_index = entries.len() as u64;
        if entry.index != expected_index {
            return Err(format!(
                "ledger index discontinuity at line {} ({} != {})",
                line_no + 1,
                entry.index,
                expected_index
            ));
        }
        if entry.prev_hash_hex != prev_hash {
            return Err(format!(
                "ledger prev_hash mismatch at index {}",
                entry.index
            ));
        }
        let recomputed = LedgerEntry::compute_entry_hash(
            entry.index,
            &entry.prev_hash_hex,
            entry.lease_nonce,
            &entry.receipt,
        );
        if entry.entry_hash_hex != recomputed {
            return Err(format!(
                "ledger entry_hash tampered at index {}",
                entry.index
            ));
        }
        entry
            .receipt
            .verify()
            .map_err(|e| format!("receipt invalid at index {}: {e}", entry.index))?;
        if let Some(pid) = expected_provider_id {
            if entry.receipt.provider_id != pid {
                return Err(format!(
                    "provider_id mismatch at index {} ({} != {pid})",
                    entry.index, entry.receipt.provider_id
                ));
            }
        }

        prev_hash = entry.entry_hash_hex.clone();
        total_prompt = total_prompt.saturating_add(entry.receipt.prompt_tokens);
        total_completion = total_completion.saturating_add(entry.receipt.completion_tokens);
        total_earnings = total_earnings.saturating_add(entry.receipt.total_cost_micro_usd);
        seen_nonces.push(entry.lease_nonce);
        entries.push(entry);
    }

    let summary = LedgerSummary {
        entry_count: entries.len() as u64,
        head_hash_hex: prev_hash,
        total_prompt_tokens: total_prompt,
        total_completion_tokens: total_completion,
        total_earnings_micro_usd: total_earnings,
        seen_nonces,
    };
    Ok((entries, summary))
}

#[derive(Debug)]
struct LedgerInner {
    path: Option<PathBuf>,
    fsync_on_append: bool,
    entries: Vec<LedgerEntry>,
    head_hash_hex: String,
    seen_nonces: BTreeSet<u64>,
    total_prompt_tokens: u64,
    total_completion_tokens: u64,
    total_earnings_micro_usd: u64,
}

/// Thread-safe, append-only hash-chained JSONL sovereign receipt ledger.
#[derive(Debug)]
pub struct SovereignLedger {
    inner: Mutex<LedgerInner>,
}

impl SovereignLedger {
    /// Creates an in-memory sovereign ledger (for fast ephemeral runs or unit tests).
    pub fn in_memory() -> Self {
        Self {
            inner: Mutex::new(LedgerInner {
                path: None,
                fsync_on_append: false,
                entries: Vec::new(),
                head_hash_hex: GENESIS_HASH_HEX.to_owned(),
                seen_nonces: BTreeSet::new(),
                total_prompt_tokens: 0,
                total_completion_tokens: 0,
                total_earnings_micro_usd: 0,
            }),
        }
    }

    /// Opens an existing sovereign JSONL ledger on disk (verifying every Ed25519
    /// signature and SHA-256 chain link) or initializes a new ledger file.
    pub fn open_or_create(
        path: impl AsRef<Path>,
        expected_provider_id: Option<&str>,
        fsync_on_append: bool,
    ) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| format!("create ledger dir failed: {e}"))?;
            }
        }

        if path.exists() {
            let file = File::open(&path).map_err(|e| format!("open ledger failed: {e}"))?;
            let reader = BufReader::new(file);
            let mut content = String::new();
            for line in reader.lines() {
                let line = line.map_err(|e| format!("read ledger failed: {e}"))?;
                content.push_str(&line);
                content.push('\n');
            }
            let (entries, summary) = verify_jsonl_chain(&content, expected_provider_id)?;
            let seen_nonces: BTreeSet<u64> = summary.seen_nonces.into_iter().collect();
            Ok(Self {
                inner: Mutex::new(LedgerInner {
                    path: Some(path),
                    fsync_on_append,
                    entries,
                    head_hash_hex: summary.head_hash_hex,
                    seen_nonces,
                    total_prompt_tokens: summary.total_prompt_tokens,
                    total_completion_tokens: summary.total_completion_tokens,
                    total_earnings_micro_usd: summary.total_earnings_micro_usd,
                }),
            })
        } else {
            Ok(Self {
                inner: Mutex::new(LedgerInner {
                    path: Some(path),
                    fsync_on_append,
                    entries: Vec::new(),
                    head_hash_hex: GENESIS_HASH_HEX.to_owned(),
                    seen_nonces: BTreeSet::new(),
                    total_prompt_tokens: 0,
                    total_completion_tokens: 0,
                    total_earnings_micro_usd: 0,
                }),
            })
        }
    }

    /// Appends a verified receipt into the hash chain and persists it durably to disk.
    pub fn append(
        &self,
        lease_nonce: u64,
        receipt: CompletionReceipt,
    ) -> Result<LedgerEntry, String> {
        receipt.verify()?;
        let mut st = self.inner.lock().expect("ledger mutex poisoned");
        let index = st.entries.len() as u64;
        let prev_hash_hex = st.head_hash_hex.clone();
        let entry_hash_hex =
            LedgerEntry::compute_entry_hash(index, &prev_hash_hex, lease_nonce, &receipt);
        let entry = LedgerEntry {
            index,
            prev_hash_hex,
            entry_hash_hex: entry_hash_hex.clone(),
            lease_nonce,
            receipt: receipt.clone(),
        };

        if let Some(ref path) = st.path {
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| format!("open ledger for append failed: {e}"))?;
            let line = entry.to_json().to_compact_string();
            writeln!(f, "{line}").map_err(|e| format!("write ledger line failed: {e}"))?;
            if st.fsync_on_append {
                f.sync_all()
                    .map_err(|e| format!("fsync ledger failed: {e}"))?;
            }
        }

        st.head_hash_hex = entry_hash_hex;
        st.seen_nonces.insert(lease_nonce);
        st.total_prompt_tokens = st.total_prompt_tokens.saturating_add(receipt.prompt_tokens);
        st.total_completion_tokens = st
            .total_completion_tokens
            .saturating_add(receipt.completion_tokens);
        st.total_earnings_micro_usd = st
            .total_earnings_micro_usd
            .saturating_add(receipt.total_cost_micro_usd);
        st.entries.push(entry.clone());
        Ok(entry)
    }

    pub fn next_receipt_seq(&self) -> u64 {
        let st = self.inner.lock().expect("ledger mutex poisoned");
        st.entries.len() as u64 + 1
    }

    pub fn has_seen_nonce(&self, nonce: u64) -> bool {
        let st = self.inner.lock().expect("ledger mutex poisoned");
        st.seen_nonces.contains(&nonce)
    }

    pub fn export_jsonl(&self) -> String {
        let st = self.inner.lock().expect("ledger mutex poisoned");
        let mut out = String::new();
        for e in &st.entries {
            out.push_str(&e.to_json().to_compact_string());
            out.push('\n');
        }
        out
    }

    pub fn summary(&self) -> LedgerSummary {
        let st = self.inner.lock().expect("ledger mutex poisoned");
        LedgerSummary {
            entry_count: st.entries.len() as u64,
            head_hash_hex: st.head_hash_hex.clone(),
            total_prompt_tokens: st.total_prompt_tokens,
            total_completion_tokens: st.total_completion_tokens,
            total_earnings_micro_usd: st.total_earnings_micro_usd,
            seen_nonces: st.seen_nonces.iter().copied().collect(),
        }
    }

    /// Exports the full sovereign JSONL ledger atomically to `dest_path`
    /// (temp file + `sync_all()` + rename) for zero-lock-in migration/forking.
    pub fn export_to_file_atomic(&self, dest_path: impl AsRef<Path>) -> Result<usize, String> {
        let dest = dest_path.as_ref();
        if let Some(parent) = dest.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| format!("create export dir failed: {e}"))?;
            }
        }
        let data = self.export_jsonl();
        let tmp = dest.with_extension("jsonl.tmp");
        let mut f = File::create(&tmp).map_err(|e| format!("create tmp export failed: {e}"))?;
        f.write_all(data.as_bytes())
            .map_err(|e| format!("write tmp export failed: {e}"))?;
        f.sync_all()
            .map_err(|e| format!("sync tmp export failed: {e}"))?;
        drop(f);
        fs::rename(&tmp, dest).map_err(|e| format!("atomic rename export failed: {e}"))?;
        Ok(data.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_chain_verification_and_tamper_detection() {
        let kp = ProviderKeyPair::from_label("ledger-test-provider");
        let ledger = SovereignLedger::in_memory();

        for i in 1..=3u64 {
            let receipt = CompletionReceipt::sign_new(
                &kp,
                i,
                format!("job-{i}"),
                "tenant-acme",
                "zen-8b-q4",
                sha256_hex(format!("quote-{i}").as_bytes()),
                sha256_hex(format!("in-{i}").as_bytes()),
                sha256_hex(format!("out-{i}").as_bytes()),
                128,
                64,
                200_000,
                1_200,
                4_800,
                1_700_000_000_000 + i,
            );
            ledger.append(10_000 + i, receipt).unwrap();
        }

        let exported = ledger.export_jsonl();
        let (entries, summary) =
            verify_jsonl_chain(&exported, Some(kp.provider_id().as_str())).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(summary.total_prompt_tokens, 384);
        assert_eq!(summary.total_completion_tokens, 192);

        // Tamper a single digit in line 2 -> hash chain check MUST fail.
        let tampered = exported.replacen("\"completion_tokens\":64", "\"completion_tokens\":99", 1);
        assert!(verify_jsonl_chain(&tampered, Some(kp.provider_id().as_str())).is_err());
    }
}
