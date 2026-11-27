//! Multi-tenant isolation, strict VRAM/concurrency admission control,
//! graceful drain coordination, and zeroized per-tenant KV/scratch buffers.
//!
//! Aligns with `hanzo-server-core/src/tenant.rs` (`X-Org-Id` / `cache_salt`)
//! and `hanzo-sandbox` execution policies so provider nodes integrate cleanly
//! with canonical Hanzo inference isolation boundaries.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::crypto::sha256_hex;

/// Maximum identifier length for `tenant_id` and `job_id`.
pub const MAX_ID_LEN: usize = 64;

/// Strictly validated `(tenant_id, job_id)` execution namespace.
///
/// Rejects path traversal (`..`, `/`, `\`), NUL bytes, whitespace, and control
/// characters so untrusted client identifiers can never escape tenant isolation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TenantNamespace {
    tenant_id: String,
    job_id: String,
    namespace_digest: String,
}

impl TenantNamespace {
    pub fn new(tenant_id: &str, job_id: &str) -> Result<Self, IsolationError> {
        validate_identifier("tenant_id", tenant_id)?;
        validate_identifier("job_id", job_id)?;
        let preimage = format!(
            "hanzo-tenant-ns:v1\0len_t={}:{tenant_id}\0len_j={}:{job_id}",
            tenant_id.len(),
            job_id.len()
        );
        let namespace_digest = sha256_hex(preimage.as_bytes());
        Ok(Self {
            tenant_id: tenant_id.to_owned(),
            job_id: job_id.to_owned(),
            namespace_digest,
        })
    }

    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    /// Domain-separated SHA-256 namespace key for KV/scratch isolation.
    pub fn namespace_digest(&self) -> &str {
        &self.namespace_digest
    }

    /// Cache salt compatible with `hanzo-server-core/src/tenant.rs` (`[org, job_salt]`),
    /// preventing any tenant's requested salt from colliding with another tenant's partition.
    pub fn cache_salt(&self) -> String {
        format!("[\"{}\",\"{}\"]", self.tenant_id, self.job_id)
    }
}

pub fn validate_identifier(field: &str, value: &str) -> Result<(), IsolationError> {
    if value.is_empty() || value.len() > MAX_ID_LEN {
        return Err(IsolationError::InvalidIdentifier(format!(
            "{field} must be 1..={MAX_ID_LEN} ASCII bytes"
        )));
    }
    if value.contains("..") || value.contains('/') || value.contains('\\') {
        return Err(IsolationError::InvalidIdentifier(format!(
            "{field} contains forbidden path sequence"
        )));
    }
    if !value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
    {
        return Err(IsolationError::InvalidIdentifier(format!(
            "{field} contains disallowed characters"
        )));
    }
    if value.starts_with('.') {
        return Err(IsolationError::InvalidIdentifier(format!(
            "{field} must not start with '.'"
        )));
    }
    Ok(())
}

/// Declarative sandbox contract aligned with `hanzo-sandbox`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxExecutionPolicy {
    pub network_disabled: bool,
    pub max_prompt_bytes: usize,
    pub max_output_tokens: u32,
    pub max_wall_ms: u64,
    pub zeroize_scratch_on_drop: bool,
}

impl Default for SandboxExecutionPolicy {
    fn default() -> Self {
        Self {
            network_disabled: true,
            max_prompt_bytes: 64 * 1024,
            max_output_tokens: 4_096,
            max_wall_ms: 60_000,
            zeroize_scratch_on_drop: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IsolationError {
    InvalidIdentifier(String),
    CapacityExhausted {
        requested_vram_mib: u64,
        available_vram_mib: u64,
        active_jobs: u32,
        max_concurrency: u32,
    },
    ProviderDraining,
    CrossTenantAccessDenied {
        caller_tenant: String,
        target_namespace: String,
    },
    PayloadRejected(String),
}

impl fmt::Display for IsolationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IsolationError::InvalidIdentifier(msg) => write!(f, "invalid identifier: {msg}"),
            IsolationError::CapacityExhausted {
                requested_vram_mib,
                available_vram_mib,
                active_jobs,
                max_concurrency,
            } => write!(
                f,
                "capacity exhausted (requested {requested_vram_mib} MiB, available {available_vram_mib} MiB, active {active_jobs}/{max_concurrency})"
            ),
            IsolationError::ProviderDraining => write!(f, "provider is draining"),
            IsolationError::CrossTenantAccessDenied {
                caller_tenant,
                target_namespace,
            } => write!(
                f,
                "cross-tenant access denied for `{caller_tenant}` on `{target_namespace}`"
            ),
            IsolationError::PayloadRejected(msg) => write!(f, "payload rejected: {msg}"),
        }
    }
}

impl std::error::Error for IsolationError {}

/// Per-tenant isolated KV/scratch buffer that overwrites its memory with zeroes
/// on drop so residual prompt/KV bytes can never leak to a subsequent tenant.
#[derive(Debug)]
struct ScratchBuffer {
    owner_tenant: String,
    data: Vec<u8>,
}

impl Drop for ScratchBuffer {
    fn drop(&mut self) {
        self.data.fill(0);
        self.data.clear();
    }
}

#[derive(Debug)]
struct PoolState {
    total_vram_mib: u64,
    used_vram_mib: u64,
    max_concurrency: u32,
    active_jobs: u32,
    draining: bool,
    peak_vram_mib: u64,
    peak_concurrency: u32,
    scratch_slots: BTreeMap<String, ScratchBuffer>,
}

/// Thread-safe strict admission-control and per-tenant KV/scratch isolation pool.
#[derive(Clone, Debug)]
pub struct CapacityPool {
    inner: Arc<(Mutex<PoolState>, Condvar)>,
}

impl CapacityPool {
    pub fn new(total_vram_mib: u64, max_concurrency: u32) -> Self {
        Self {
            inner: Arc::new((
                Mutex::new(PoolState {
                    total_vram_mib,
                    used_vram_mib: 0,
                    max_concurrency: max_concurrency.max(1),
                    active_jobs: 0,
                    draining: false,
                    peak_vram_mib: 0,
                    peak_concurrency: 0,
                    scratch_slots: BTreeMap::new(),
                }),
                Condvar::new(),
            )),
        }
    }

    /// Atomically reserves VRAM + a concurrency slot + an isolated tenant KV/scratch slot.
    ///
    /// Returns immediately with `ProviderDraining` or `CapacityExhausted` instead of
    /// overcommitting VRAM or letting queues grow unboundedly.
    pub fn try_reserve(
        &self,
        ns: TenantNamespace,
        vram_mib: u64,
        scratch_bytes: usize,
    ) -> Result<CapacityGuard, IsolationError> {
        let mut st = self.inner.0.lock().expect("capacity pool mutex poisoned");
        if st.draining {
            return Err(IsolationError::ProviderDraining);
        }
        let available = st.total_vram_mib.saturating_sub(st.used_vram_mib);
        if st.active_jobs >= st.max_concurrency || vram_mib > available {
            return Err(IsolationError::CapacityExhausted {
                requested_vram_mib: vram_mib,
                available_vram_mib: available,
                active_jobs: st.active_jobs,
                max_concurrency: st.max_concurrency,
            });
        }
        if st.scratch_slots.contains_key(ns.namespace_digest()) {
            return Err(IsolationError::PayloadRejected(
                "duplicate active namespace".into(),
            ));
        }

        st.used_vram_mib += vram_mib;
        st.active_jobs += 1;
        if st.used_vram_mib > st.peak_vram_mib {
            st.peak_vram_mib = st.used_vram_mib;
        }
        if st.active_jobs > st.peak_concurrency {
            st.peak_concurrency = st.active_jobs;
        }
        st.scratch_slots.insert(
            ns.namespace_digest().to_owned(),
            ScratchBuffer {
                owner_tenant: ns.tenant_id().to_owned(),
                data: vec![0u8; scratch_bytes],
            },
        );
        Ok(CapacityGuard {
            pool: self.clone(),
            ns,
            vram_mib,
        })
    }

    /// Writes bytes into a tenant's active isolated KV/scratch slot.
    /// Rejects any cross-tenant access attempt deterministically.
    pub fn write_scratch(
        &self,
        caller_tenant: &str,
        namespace_digest: &str,
        payload: &[u8],
    ) -> Result<(), IsolationError> {
        let mut st = self.inner.0.lock().expect("capacity pool mutex poisoned");
        let Some(slot) = st.scratch_slots.get_mut(namespace_digest) else {
            return Err(IsolationError::CrossTenantAccessDenied {
                caller_tenant: caller_tenant.to_owned(),
                target_namespace: namespace_digest.to_owned(),
            });
        };
        if slot.owner_tenant != caller_tenant {
            return Err(IsolationError::CrossTenantAccessDenied {
                caller_tenant: caller_tenant.to_owned(),
                target_namespace: namespace_digest.to_owned(),
            });
        }
        let copy_len = payload.len().min(slot.data.len());
        slot.data[..copy_len].copy_from_slice(&payload[..copy_len]);
        Ok(())
    }

    /// Reads bytes from an active isolated KV/scratch slot ONLY if `caller_tenant` owns it.
    pub fn read_scratch(
        &self,
        caller_tenant: &str,
        namespace_digest: &str,
    ) -> Result<Vec<u8>, IsolationError> {
        let st = self.inner.0.lock().expect("capacity pool mutex poisoned");
        let Some(slot) = st.scratch_slots.get(namespace_digest) else {
            return Err(IsolationError::CrossTenantAccessDenied {
                caller_tenant: caller_tenant.to_owned(),
                target_namespace: namespace_digest.to_owned(),
            });
        };
        if slot.owner_tenant != caller_tenant {
            return Err(IsolationError::CrossTenantAccessDenied {
                caller_tenant: caller_tenant.to_owned(),
                target_namespace: namespace_digest.to_owned(),
            });
        }
        Ok(slot.data.clone())
    }

    /// Transitions the provider into graceful drain mode: rejects all new leases
    /// while letting in-flight reservations finish cleanly.
    pub fn initiate_drain(&self) {
        let mut st = self.inner.0.lock().expect("capacity pool mutex poisoned");
        st.draining = true;
        self.inner.1.notify_all();
    }

    /// Resumes accepting new leases (e.g., after maintenance or restart).
    pub fn resume_service(&self) {
        let mut st = self.inner.0.lock().expect("capacity pool mutex poisoned");
        st.draining = false;
        self.inner.1.notify_all();
    }

    /// Waits until all in-flight jobs have finished or `timeout` elapses.
    /// Returns `true` if fully drained (`active_jobs == 0`).
    pub fn wait_until_drained(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut st = self.inner.0.lock().expect("capacity pool mutex poisoned");
        while st.active_jobs > 0 {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let (guard, _) = self
                .inner
                .1
                .wait_timeout(st, deadline - now)
                .expect("condvar mutex poisoned");
            st = guard;
        }
        true
    }

    pub fn snapshot(&self) -> CapacitySnapshot {
        let st = self.inner.0.lock().expect("capacity pool mutex poisoned");
        CapacitySnapshot {
            total_vram_mib: st.total_vram_mib,
            used_vram_mib: st.used_vram_mib,
            max_concurrency: st.max_concurrency,
            active_jobs: st.active_jobs,
            draining: st.draining,
            peak_vram_mib: st.peak_vram_mib,
            peak_concurrency: st.peak_concurrency,
            active_scratch_slots: st.scratch_slots.len(),
        }
    }

    fn release(&self, namespace_digest: &str, vram_mib: u64) {
        let mut st = self.inner.0.lock().expect("capacity pool mutex poisoned");
        // Drop triggers zeroization of the tenant's scratch memory.
        st.scratch_slots.remove(namespace_digest);
        st.used_vram_mib = st.used_vram_mib.saturating_sub(vram_mib);
        st.active_jobs = st.active_jobs.saturating_sub(1);
        if st.active_jobs == 0 {
            self.inner.1.notify_all();
        }
    }
}

/// Snapshot of live capacity & drain metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacitySnapshot {
    pub total_vram_mib: u64,
    pub used_vram_mib: u64,
    pub max_concurrency: u32,
    pub active_jobs: u32,
    pub draining: bool,
    pub peak_vram_mib: u64,
    pub peak_concurrency: u32,
    pub active_scratch_slots: usize,
}

/// RAII capacity & isolated KV/scratch guard.
/// Releases VRAM, concurrency slot, and zeroizes tenant scratch automatically on drop.
#[derive(Debug)]
pub struct CapacityGuard {
    pool: CapacityPool,
    ns: TenantNamespace,
    vram_mib: u64,
}

impl CapacityGuard {
    pub fn namespace(&self) -> &TenantNamespace {
        &self.ns
    }

    pub fn write_scratch(&self, payload: &[u8]) -> Result<(), IsolationError> {
        self.pool
            .write_scratch(self.ns.tenant_id(), self.ns.namespace_digest(), payload)
    }

    pub fn read_scratch(&self) -> Result<Vec<u8>, IsolationError> {
        self.pool
            .read_scratch(self.ns.tenant_id(), self.ns.namespace_digest())
    }
}

impl Drop for CapacityGuard {
    fn drop(&mut self) {
        self.pool.release(self.ns.namespace_digest(), self.vram_mib);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_path_traversal_and_cross_tenant_bleed() {
        let up_path = ["..", "var", "restricted"].join("/");
        let nested_up = ["job", "..", "..", "outside"].join("/");
        assert!(TenantNamespace::new(&up_path, "job-1").is_err());
        assert!(TenantNamespace::new("tenant-a", &nested_up).is_err());
        assert!(TenantNamespace::new("tenant\0a", "job-1").is_err());

        let pool = CapacityPool::new(16_384, 2);
        let ns_a = TenantNamespace::new("tenant-alpha", "job-001").unwrap();
        let guard_a = pool.try_reserve(ns_a.clone(), 8_192, 128).unwrap();
        guard_a.write_scratch(b"alpha-isolated-kv-state").unwrap();

        // Tenant B cannot read or overwrite Tenant A's scratch slot.
        assert!(matches!(
            pool.read_scratch("tenant-beta", ns_a.namespace_digest()),
            Err(IsolationError::CrossTenantAccessDenied { .. })
        ));
        assert!(matches!(
            pool.write_scratch("tenant-beta", ns_a.namespace_digest(), b"pwn"),
            Err(IsolationError::CrossTenantAccessDenied { .. })
        ));

        // Dropping releases VRAM and destroys the scratch slot.
        drop(guard_a);
        let snap = pool.snapshot();
        assert_eq!(snap.used_vram_mib, 0);
        assert_eq!(snap.active_jobs, 0);
        assert_eq!(snap.active_scratch_slots, 0);
    }

    #[test]
    fn enforces_vram_backpressure_and_graceful_drain() {
        let pool = CapacityPool::new(12_000, 4);
        let g1 = pool
            .try_reserve(TenantNamespace::new("t1", "j1").unwrap(), 8_000, 64)
            .unwrap();
        // Second 8,000 MiB job exceeds 12,000 MiB -> immediate CapacityExhausted.
        assert!(matches!(
            pool.try_reserve(TenantNamespace::new("t2", "j2").unwrap(), 8_000, 64),
            Err(IsolationError::CapacityExhausted { .. })
        ));

        // Drain rejects new jobs even when VRAM is available, while g1 remains active.
        pool.initiate_drain();
        assert!(matches!(
            pool.try_reserve(TenantNamespace::new("t3", "j3").unwrap(), 1_000, 64),
            Err(IsolationError::ProviderDraining)
        ));
        assert!(!pool.wait_until_drained(Duration::from_millis(5)));
        drop(g1);
        assert!(pool.wait_until_drained(Duration::from_millis(50)));
    }
}
