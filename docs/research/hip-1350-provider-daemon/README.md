# HIP-1350 Research Dossier: Permissionless GPU Provider Daemon (`hanzoai/engine#64`)

- **Canonical Owning Repository**: `hanzoai/engine` (crate: `hanzo-provider-daemon/`, license: `Apache-2.0`)
- **Tracking Issue**: [`hanzoai/engine#64`](https://github.com/hanzoai/engine/issues/64) (`p0`, `bounty`, `research`, `decentralized`, `supply`, `hip-1350`)
- **Raw Benchmark Receipt**: [`results.json`](./results.json)
- **PR Disposition Target**: `GATED` for real-funds settlement (external security audit required before real funds); `MERGE` for opt-in permissionless provider daemon & protocol library on `main`.

---

## 1. Problem Statement, Threat Model, and Defensible Proof Claims

### Problem
Turning independently administered machines (NVIDIA CUDA, AMD ROCm, Apple Metal, Vulkan, or CPU hosts) into a permissionless decentralized inference supply network requires solving five tightly coupled systems problems without introducing any mandatory Hanzo-operated control plane:
1. **Self-sovereign identity & permissionless discovery**: Any operator must join and advertise hardware/model capabilities with zero centralized approval or vendor-hosted KMS.
2. **Pre-execution metering & anti-fraud binding**: A provider must bind quoted price, model, context budget, tenant authorization, and replay nonces cryptographically before committing GPU VRAM.
3. **Multi-tenant host & KV isolation**: Malicious or malformed workloads must never escape tenant namespaces, read sibling tenants' KV/scratch buffers, or OOM-thrash the provider host.
4. **Graceful lifecycle & crash-resilient state**: Operators must drain in-flight paid work without dropping jobs, restart cleanly from disk, and return to service automatically with replay-nonce continuity.
5. **Sovereignty & zero lock-in**: Operators must be able to export keys and hash-chained usage receipts to migrate, audit, or fork freely.

### Threat Model & Honest Proof-Claim Boundary
Per `hanzoai/engine#64`, we **strictly separate** cryptographic signatures, hardware attestations, and economic verification:
- **What `hanzo-provider-daemon` cryptographically proves (RFC 8032 Ed25519 + FIPS 180-4 SHA-256)**:
  - **Authentication & non-repudiation**: A capability announcement, quote, or completion receipt was signed by the holder of the provider's Ed25519 secret key; a job lease was authorized by the holder of the tenant's Ed25519 secret key and bound to a specific quote digest and unique nonce.
  - **Tamper-evident auditability**: Every receipt's `(job_id, quote_digest, input_digest, output_digest, token_counts, price, total_cost, ttft_us, total_us)` is sealed in a provider-signed Ed25519 envelope and linked in a SHA-256 hash-chained JSONL ledger (`SovereignLedger`).
- **What `hanzo-provider-daemon` does NOT claim**:
  - Ed25519 receipts alone do **not** prove hardware TEE integrity (e.g., NVIDIA H100 Confidential Computing / AMD SEV-SNP attestation) or prevent a colluding malicious provider from running a cheaper model behind the signed digest. Those guarantees belong to orthogonal layers (hardware attestation + PoAI/Freivalds verification + economic slashing, which require independent external security review before holding real funds).

---

## 2. Methodology, Public Corpus Grounding, and Reproduction

### Public Corpus Grounding (`PR_CONVERGENCE.md` Compliant)
All benchmark prompts in `provider-daemon-bench` are deterministically sliced (`--seed 1350`, xorshift64*) from **8 real, versioned `hanzoai/engine` repository files** (`190,451` bytes total):
- `README.md`, `LLM.md`, `LICENSE-APACHE`, `HANZO_ENGINE_INTEGRATION.md`, `hanzo-server-core/src/tenant.rs`, `hanzo-sandbox/src/lib.rs`, `hanzo-cli/src/cluster.rs`, `hanzo-server/src/zap_server.rs`
- **Verified Corpus SHA-256**: `3e65974a4937b8a852775864dd13e4810189e737d7c2e0af4937d72b1228fbf0`

### Exact Reproduction Commands
```bash
# 1. Formatting & zero-warning Clippy checks
cargo fmt -p hanzo-provider-daemon -- --check
cargo clippy -p hanzo-provider-daemon --all-targets --offline -- -D warnings

# 2. Unit tests (incl. RFC 8032 §7.1 Ed25519 test vectors 1-3) & TCP integration tests
cargo test -p hanzo-provider-daemon --offline --release

# 3. Reproducible 5-axis HIP-1350 benchmark + 15-vector trust suite + 2 negative regimes
cargo run -p hanzo-provider-daemon --release --offline --bin provider-daemon-bench -- \
  --seed 1350 --jobs-per-provider 20 \
  --output docs/research/hip-1350-provider-daemon/results.json
```

### Measurement Scope & Private Qualification Note
- **Real local execution**: All benchmarks execute over real multi-threaded `std::net::TcpListener` HTTP/1.1 loopback sockets with genuine RFC 8032 Ed25519 signing/verification, real OS file persistence (`SovereignLedger`), and real multi-tenant `CapacityPool` contention.
- **Deterministic inference backend**: Public CI runs use `DeterministicEngineBackend` (implementing the pluggable `InferenceBackend` trait) so any contributor can reproduce exact numbers offline without proprietary GPU weights. Production GPU hardware runs against confidential Hanzo routing corpora remain internal to authorized Hanzo runners per `PR_CONVERGENCE.md`.

---

## 3. Falsifiable Hypotheses, Metrics, and Failure Modes

### Hypotheses
- **H1 (Permissionless Supply & Decentralization)**: 3 independently administered provider daemons (distinct Ed25519 keys, distinct hardware profiles, distinct loopback sockets & ledgers) register without operator approval, serve multi-tenant paid jobs, drain cleanly, restart from disk, and return to service automatically with **`0` Hanzo control-plane calls (`100.0%` decentralization share)**.
- **H2 (Adversarial Trust & Isolation)**: Across **15 live malicious/malformed/fraudulent vectors** (forged signatures, tampered price/model/max-tokens, expired quotes, replayed nonces, cross-provider redirection, over-budget contexts, oversized HTTP DoS bodies, path traversal, NUL injection, cross-tenant KV/scratch probes, tampered receipts, tampered ledger chains, drain-state submissions), rejection rate is **`100.0%`** and cross-tenant data bleed is **`0.0%`**.
- **H3 (Sovereign Migration & Restart Continuity)**: Exporting `KeyEnvelope` + hash-chained JSONL ledger and cold-restarting all 3 providers recovers **`100.0%`** of receipt count, earnings history, chain head digest, and pre-restart replay-nonce protection.
- **H4 (Explicit Negative Regime 1 — 2× Overload Saturation)**: Under `2×` offered concurrency over a provider's VRAM budget (`12,288 MiB`, `max_concurrency = 2`), strict admission control keeps VRAM strictly `<= 12,288 MiB` (`0` oversubscription events) and protects accepted job p99 latency vs an unthrottled oversubscribed control, at the explicit trade-off of fast-rejecting excess burst jobs with `HTTP 429`.
- **H5 (Explicit Negative Regime 2 — Unverified/Unmetered Control)**: Removing Ed25519 quote/lease/receipt verification saves measurable per-job CPU microseconds (documented candidly) but collapses adversarial fraud detection from **`100.0%` to `0.0%`** and increases unauthorized compute/revenue leakage to **`100.0%`**.

### Failure Modes & Mitigations
| Failure Mode | Trigger | Mitigation in `hanzo-provider-daemon` |
|---|---|---|
| Single-node burst saturation (`2×` load) | Offered concurrency exceeds VRAM/slot budget without peer spillover | Immediate `HTTP 429 CapacityExhausted` (< 9 ms p50) so Enso/clients fail over to another permissionless provider; prevents host OOM collapse. |
| Pure-Rust Ed25519 CPU overhead on tiny jobs | Very short prompts with zero GPU compute latency | Amortized verification cost is `$0.0005 / 1M` tokens (`0.29%` of `$0.1645 / 1M`); negligible compared to real GPU inference time (`> 20 ms`). |
| TCP RST race on oversized HTTP `413` bodies | Client sends `> 256 KiB` body while server closes immediately | `graceful_close()` half-closes write (`Shutdown::Write`) and drains strictly bounded buffer briefly (`50 ms` cap) so `413` JSON reaches the client cleanly. |
| Cross-protocol signature replay | Reusing a quote signature as a receipt or announcement | Explicit null-terminated domain separators (`DOMAIN_ANNOUNCEMENT`, `DOMAIN_QUOTE`, `DOMAIN_LEASE`, `DOMAIN_RECEIPT`). |
| Signature malleability / non-canonical curve points | Attacker adds group order `L` to `S` or sends `y >= p` | Strict RFC 8032 canonical checks (`S < L`, `y < p`, zero-x sign bit check) verified against RFC 8032 §7.1 vectors. |
| Clock skew between tenant and provider | Unsynchronized host clocks | Short-lived quote TTLs evaluated against provider's monotonic/wall clock; announcements carry monotonic `sequence` numbers. |

---

## 4. Bounded Implementation and Canonical Hanzo Architecture Fit

### Crate Layout (`hanzo-provider-daemon/` — Zero External Dependencies, `#![forbid(unsafe_code)]`, `Apache-2.0`)
- [`src/crypto.rs`](../../../hanzo-provider-daemon/src/crypto.rs): Pure-Rust FIPS 180-4 SHA-256 & SHA-512, constant-time `ct_eq`, and RFC 8032 Ed25519 over Curve25519 (5-limb radix-$2^{51}$ `u128` field arithmetic, verified against RFC 8032 §7.1 Test Vectors 1, 2, and 3).
- [`src/json.rs`](../../../hanzo-provider-daemon/src/json.rs): Zero-dependency deterministic JSON encoder & DoS-bounded parser (`MAX_JSON_DEPTH = 32`, `MAX_JSON_BYTES = 1 MiB`).
- [`src/identity.rs`](../../../hanzo-provider-daemon/src/identity.rs): Self-sovereign `ProviderKeyPair`, `TenantKeyPair`, `ProviderId`, domain-separated signing/verification, and portable `KeyEnvelope`.
- [`src/advertise.rs`](../../../hanzo-provider-daemon/src/advertise.rs): Local hardware probe (`CUDA`, `ROCm`, `Metal`, `Vulkan`, `CPU`), signed `CapabilityAnnouncement`, and permissionless `PeerDirectory`.
- [`src/isolation.rs`](../../../hanzo-provider-daemon/src/isolation.rs): Validated `TenantNamespace` (`cache_salt` compatible with `hanzo-server-core/src/tenant.rs`), `SandboxExecutionPolicy` aligned with `hanzo-sandbox`, RAII VRAM/concurrency `CapacityPool` + `CapacityGuard`, graceful drain, and zeroized per-tenant scratch buffers.
- [`src/ledger.rs`](../../../hanzo-provider-daemon/src/ledger.rs): Signed `CompletionReceipt`, hash-chained JSONL `SovereignLedger` with `sync_all()` durability, atomic export, and full cold-restart replay.
- [`src/daemon.rs`](../../../hanzo-provider-daemon/src/daemon.rs): `ProviderDaemon` 12-step pre-execution verification pipeline, pluggable `InferenceBackend` trait, and `UnverifiedControlDaemon` baseline.
- [`src/rpc.rs`](../../../hanzo-provider-daemon/src/rpc.rs): Pure `std::net::TcpListener` multi-threaded HTTP/1.1 JSON server (`/health`, `/v1/provider/manifest`, `/v1/provider/quote`, `/v1/provider/jobs`, `/v1/provider/verify-receipt`, `/v1/provider/ledger/export`, `/v1/provider/drain`) with strict header (`16 KiB`), body (`256 KiB`), and socket timeout (`5 s`) caps.
- [`src/bench.rs`](../../../hanzo-provider-daemon/src/bench.rs) & [`src/bin/provider-daemon-bench.rs`](../../../hanzo-provider-daemon/src/bin/provider-daemon-bench.rs): Reproducible benchmark & acceptance harness.

### Canonical Architecture Alignment (Zero Parallel AI Stacks)
- **`hanzo-engine`**: Hooks directly into the `InferenceBackend` trait (`execute(model_id, prompt, max_tokens, &CapacityGuard)`) without duplicating model loaders, tokenizers, or attention kernels.
- **`hanzo-server-core` & `hanzo-kv-fabric`**: `TenantNamespace::cache_salt()` produces `["<tenant_id>","<job_id>"]`, matching [`hanzo-server-core/src/tenant.rs`](../../../hanzo-server-core/src/tenant.rs) prefix-cache partitioning verbatim.
- **`hanzo-sandbox`**: `SandboxExecutionPolicy` mirrors [`hanzo-sandbox/src/lib.rs`](../../../hanzo-sandbox/src/lib.rs) (`NetworkMode::None`, memory/CPU/output bounds) for subprocess isolation when executing untrusted worker extensions.
- **Enso routing, Visor, and VFS**: Enso routers consume `GET /v1/provider/manifest` announcements without any central coordinator; Visor/telemetry scrapes `/health` and receipt latency (`ttft_us`, `total_us`); `SovereignLedger::export_to_file_atomic` syncs signed JSONL batches to VFS/S3 optionally without making VFS mandatory.
- **Rollout & Rollback Plan**: Opt-in workspace crate (`hanzo-provider-daemon`) with zero behavioral impact on existing `hanzo-server` or `hanzo-cli` code paths. Rollout connects `hanzo-engine` via an opt-in adapter behind a feature flag; rollback is a single feature-flag toggle or crate removal with zero schema migration risk.

---

## 5. Measured Results (Exact Values from `results.json`, `--seed 1350`)

### Summary Across All 5 HIP-1350 Axes

| HIP-1350 Axis | Metric | Measured Result |
|---|---|---|
| **1. Cost** | Weighted compute price | `\$0.1641 / 1M` tokens |
| **1. Cost** | Amortized Ed25519 + ledger CPU cost (`\$0.05/vCPU-hr`) | `\$0.0005 / 1M` tokens (`0.29%` overhead) |
| **1. Cost** | **Total verified cost per 1M useful tokens** | **`\$0.1645 / 1M` tokens** |
| **2. Performance** | 3-provider concurrent loopback TCP jobs (`4,800` prompt + `2,880` completion = `7,680` tokens) | `60 / 60` (**`100.0%`** completion rate) |
| **2. Performance** | Time-To-First-Token (TTFT) p50 / p95 / p99 | **`2.044 ms` / `8.000 ms` / `8.361 ms`** |
| **2. Performance** | End-to-end TCP round-trip latency p50 / p95 / p99 | **`17.701 ms` / `25.946 ms` / `28.733 ms`** |
| **2. Performance** | Aggregate throughput (`0.384 s` wall time) | **`19,998.521 tok/s`** (`156.238 jobs/s`) |
| **3. Trust** | Live adversarial vectors rejected (`15 / 15`) | **`100.0%`** |
| **3. Trust** | Cross-tenant KV/scratch data bleed | **`0.0%`** |
| **4. Decentralization** | Independent providers registered permissionlessly (`0` approvals, `0` Hanzo calls) | **`3 / 3`** (**`100.0%`** decentralized) |
| **5. Sovereignty** | Providers exported (`KeyEnvelope` + JSONL), cold-restarted, replay-checked & resumed | **`3 / 3`** (**`100.0%`** lossless recovery) |

### Live Adversarial Trust Suite (`15 / 15` = `100.0%` Rejected)
| # | Adversarial Vector | Observed Enforcement |
|---|---|---|
| 1 | Forged tenant Ed25519 signature (mutated prompt) | `HTTP 401` |
| 2 | Tampered `max_tokens` vs signed quote | `HTTP 400` |
| 3 | Tampered `price_micro_usd_per_1m` (underpayment) | `HTTP 400` |
| 4 | Tampered `model_id` (32B requested on 8B quote) | `HTTP 400` |
| 5 | Expired quote TTL | `HTTP 401` |
| 6 | Replayed `lease_nonce` (duplicate execution) | `HTTP 409` |
| 7 | Cross-provider lease redirection (foreign provider quote) | `HTTP 403` |
| 8 | Over-budget context request (`> max_context_tokens`) | `HTTP 400` |
| 9 | Oversized HTTP payload DoS (`> 256 KiB`) | `HTTP 413` |
| 10 | Path-traversal `tenant_id` (`../etc/passwd`) | `HTTP 400` |
| 11 | NUL-byte injection inside signed prompt | `HTTP 400` |
| 12 | Cross-tenant KV/scratch read probe (Tenant B -> Tenant A) | `CrossTenantAccessDenied` (`0.0%` bleed) |
| 13 | Tampered `CompletionReceipt` (inflated token count) | `HTTP 422` via `/v1/provider/verify-receipt` |
| 14 | Tampered sovereign JSONL ledger chain | `HashChainViolationDetected` |
| 15 | New lease submitted while provider is draining | `HTTP 503` |

### Explicit Negative Regime 1: `2×` Overload Saturation (`12,288 MiB` VRAM, `max_concurrency = 2`, 4 Concurrent Threads, 24 Offered Jobs)
| Regime | Accepted / Admitted | Fast-Rejected (`429`) | p50 Latency | p99 Latency | Peak VRAM | VRAM Oversubscription Events |
|---|---|---|---|---|---|---|
| **`admission_controlled` (`hanzo-provider-daemon`)** | `8` | `16` (`8.538 ms` p50) | **`55.601 ms`** | **`60.601 ms`** | **`12,288 MiB`** | **`0`** |
| **`unthrottled_oversubscribed_control`** | `24` | `0` | `82.910 ms` | `92.007 ms` | `24,576 MiB` (`2.0×`) | `22` |

- **Result**: Admission control delivers **`1.518×` p99 tail-latency protection** and eliminates **`100%`** of VRAM oversubscription/OOM faults (`0` vs `22`), returning fast `HTTP 429` signals (`8.538 ms` p50) so decentralized routers immediately route excess demand to healthy peers.
- **Honest trade-off**: On a single isolated machine with no peer spillover, strict admission control drops excess load (`16/24` rejected with `429`) rather than queueing indefinitely.

### Explicit Negative Regime 2: Verified/Metered vs Unverified/Unmetered Control (`40` Legit + `10` Adversarial Jobs)
| Regime | Avg Legit Job Time (Zero-Delay Backend) | Adversarial Fraud Rejected | Revenue / Compute Leakage | Auditable Signed Receipts |
|---|---|---|---|---|
| **`verified_metered` (`hanzo-provider-daemon`)** | `4,515.300 µs` (`+4,158.450 µs` crypto + ledger overhead) | **`10 / 10` (`100.0%`)** | **`0.0%`** | **`40 / 40`** |
| **`unverified_unmetered_control`** | `356.850 µs` | `0 / 10` (`0.0%`) | `100.0%` | `0 / 40` |

- **Result**: Pure-Rust RFC 8032 Ed25519 quote/lease/receipt signing + verification + hash-chained ledgering costs `4.16 ms` CPU per full job cycle (`\$0.0005 / 1M` tokens amortized), in exchange for **`100.0%` vs `0.0%`** fraud rejection and **`100%`** non-repudiable settlement receipts.

---

## 6. Security Review & PR Convergence Verification

1. **Canonical Owning Repository**: `hanzoai/engine` (`hanzo-provider-daemon/`). Zero forks, zero `[patch.crates-io]` overrides, zero external crate dependencies.
2. **Qualification Criteria**:
   - `#![forbid(unsafe_code)]` across library, CLI binary, and integration tests.
   - Pure-Rust RFC 8032 Ed25519 verified against official RFC 8032 §7.1 Test Vectors 1, 2, and 3 + malleability checks (`S >= L`, `y >= p`).
   - Strict DoS bounds (`MAX_HTTP_HEADER_BYTES = 16 KiB`, `MAX_HTTP_BODY_BYTES = 256 KiB`, `MAX_JSON_DEPTH = 32`, `SOCKET_IO_TIMEOUT = 5 s`).
   - Zero leaked VRAM/concurrency slots/scratch buffers under 6-thread TCP concurrency stress.
3. **Verification Evidence**:
   - `cargo fmt -p hanzo-provider-daemon -- --check`: **PASS**
   - `cargo clippy -p hanzo-provider-daemon --all-targets --offline -- -D warnings`: **PASS (0 warnings)**
   - `cargo test -p hanzo-provider-daemon --offline --release`: **20 / 20 PASS** (`17` unit tests + `3` integration tests, `0.44 s`)
   - `provider-daemon-bench --seed 1350`: **PASS** (raw receipt in [`results.json`](./results.json))
4. **Security & Regulatory Gate**:
   - Auth/IAM: Self-sovereign Ed25519 keys with domain separation, zero hardcoded secrets, zero Hanzo KMS lock-in.
   - Tenant Isolation: Validated `TenantNamespace`, `0.0%` cross-tenant KV/scratch bleed, deterministic zeroization on `Drop`.
   - External Audit Gate: Settlement contracts or escrow mechanisms consuming `CompletionReceipt` MUST undergo independent external security audit before holding real funds (`GATED` per `PR_CONVERGENCE.md` and issue #64).
5. **Rollout & Rollback Plan**: Opt-in workspace crate with zero impact on shipping paths; rollback is a clean single-commit revert.
