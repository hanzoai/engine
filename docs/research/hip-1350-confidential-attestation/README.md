# HIP-1350 Research Dossier: Provider-Independent Confidential GPU Inference & Remote Attestation Client (`hanzoai/engine#69`)

**Issue:** `hanzoai/engine#69` (`[p1, decentralized, proof-of-ai, hip-1350]`)
**Implementations:**
1. **Rust Client (`hanzo-attest`)** — Pure Rust `std`, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`, Apache-2.0, **0 external crate dependencies**.
2. **Python Client (`hanzo-attest/clients/python/hanzo_attest`)** — Independent clean-room Python 3.12 stdlib implementation, Apache-2.0, **0 external pip dependencies**.

**Reproducibility Command:**
```bash
cargo run --release --offline -p hanzo-attest --bin attest-bench -- \
  --seed 1350 \
  --output docs/research/hip-1350-confidential-attestation/results.json
```

---

## 1. Threat-Model Honesty: Hardware TEE Attestation vs. Trustless Proofs

> **Trusted execution is not a trustless proof.**

Before interpreting any latency or rejection metric in this dossier, we state the trust boundary explicitly:

| Security Mechanism | What Is Cryptographically Proven | Residual Trust Assumptions |
|---|---|---|
| **Confidential GPU/CPU TEE Attestation (`hanzo-attest`)** | That a device endorsed by a pinned hardware root key signed a fresh quote binding `(environment_flags, TCB_version, RTMR_measurements, model_weights_sha256, tokenizer_sha256, workload_policy_hash, client_nonce, enclave_x25519_pubkey)`, and that the client's prompt ciphertext is authenticated and bound to those exact commitments via ChaCha20-Poly1305 AAD. | **Relies on silicon vendor manufacturing roots of trust** (NVIDIA H100/Blackwell CC, AMD SEV-SNP, Intel TDX), firmware/microcode TCB integrity, timely revocation propagation, and physical/side-channel resistance of the host deployment. A compromised vendor root key or unpatched hardware microarchitectural leak breaks confidentiality and integrity. |
| **Trustless Proof-of-AI / zkML (`hanzo-poai` / SNARKs)** | Mathematical correctness of matrix/inference computation directly over committed inputs and weights. | Cryptographic soundness assumptions only (no silicon vendor root trust), but does **not** natively provide prompt confidentiality without MPC/FHE and incurs higher proving cost. |

`hanzo-attest` treats hardware roots as **explicit, client-pinned, revocable, locally controlled policy anchors** (`TrustStore` + `VerificationPolicy` + `RevocationList`) rather than implicit cloud trust, and enforces at compile time (`VerifiedAttestation` capability token) that no sensitive prompt can be encrypted or dispatched before full attestation verification succeeds.

---

## 2. Cryptographic & Protocol Architecture

Both independent implementations (Rust and Python 3.12 stdlib) implement identical wire formats, domain-separated signing preimages, canonical JSON codecs, and cryptographic primitives verified against official RFC test vectors:

- **FIPS 180-4 SHA-256 & SHA-512**
- **RFC 2104 HMAC-SHA256 & RFC 5869 HKDF-SHA256** (verified against RFC 5869 Test Case 1)
- **RFC 8032 Ed25519** (verified against §7.1 Test Vectors 1–3, plus strict canonical point `y < p`, `x == 0 => sign == 0`, and canonical scalar `S < L` malleability rejection)
- **RFC 7748 X25519** (verified against §5.2 vectors including 1,000-iteration ladder and §6.1 Alice/Bob ECDH, rejecting low-order all-zero shared secrets)
- **RFC 8439 ChaCha20-Poly1305 AEAD** (verified against §2.3.2, §2.5.2, and §2.8.2 test vectors with constant-time tag verification)

### Secure-Channel & Model/Workload Binding

1. **Quote `report_data` commitment:**
   ```
   model_commit = SHA256("hanzo-attest/model-commit/v1" || len_be32(model_id) || model_id || model_weights_sha256 || tokenizer_sha256)
   report_data  = SHA256("hanzo-attest/report-data/v1" || client_nonce || enclave_x25519_pubkey || model_commit || workload_policy_hash)
   ```
2. **Authenticated Request AAD (`canonical_binding_header`, 152 bytes):**
   ```
   aad = "hanzo-attest/req-aad/v1\0" || quote_hash || model_weights_sha256 || workload_policy_hash || client_nonce
   ```
3. **Session Key Derivation & AEAD Encryption:**
   ```
   shared_secret = X25519(client_x25519_secret, verified.enclave_x25519_pubkey)
   session_key   = HKDF-SHA256(
                     salt = quote_hash || client_nonce,
                     ikm  = shared_secret,
                     info = "hanzo-attest/session-key/v1" || client_x25519_pubkey || enclave_x25519_pubkey || aad
                   )
   (ct, tag)     = ChaCha20-Poly1305(session_key, aead_nonce, aad, plaintext)
   ```

---

## 3. Grounded Versioned Corpus (`--seed 1350`)

All benchmark measurements, model commitments, RTMR digests, and prompt payloads in `results.json` are deterministically derived from 8 versioned files in `hanzoai/engine` (combined corpus SHA-256: `87b2eed8504ef25d8837ca4338e4d06eafcab4d8d31c60e7c81e5dfc50e0d1af`):

| Repository File | Bytes | SHA-256 Prefix |
|---|---:|---|
| `Cargo.toml` | `7,599` | `4d08171065e8a543...` |
| `README.md` | `17,522` | `7377fe2c3636a05c...` |
| `hanzo.yml` | `2,123` | `b2d7f0c1d940be70...` |
| `HANZO_ENGINE_INTEGRATION.md` | `5,962` | `a7fd060907d9ca24...` |
| `hanzo-engine/src/lib.rs` | `91,742` | `80dc78ffb9493ed0...` |
| `hanzo-engine/src/distributed.rs` | `16,942` | `16ab2495b39700fc...` |
| `hanzo-engine/src/kv_cache/phase_policy.rs` | `79,949` | `83c154794e9b3cc9...` |
| `hanzo-engine/src/models/llama.rs` | `24,834` | `34dd4d6df92201d4...` |

---

## 4. Measured Results Across All 5 HIP-1350 Axes

### 4.1 Raw Measured Client & Enclave Latency (`1 KiB` / ~256-Token Prompt)

| Operation (Compiled Rust `--release`, Pure `std`) | Mean (`µs`) | `p50` (`µs`) | `p95` (`µs`) | `p99` (`µs`) | Throughput |
|---|---:|---:|---:|---:|---:|
| Cold attestation verification (2× Ed25519 + X25519 probe + full policy) | `572.013` | `554.512` | `647.516` | `724.125` | `1,748.211 verify/s` |
| Cold client verify + X25519 + HKDF + ChaCha20-Poly1305 encrypt | `711.128` | — | — | — | `1,406.217 req/s` |
| Cold end-to-end (client verify + encrypt + enclave decrypt) | `785.912` | — | — | `1,097.326` | `1,272.407 req/s` |
| Amortized `AttestedSession` client encrypt (cached session key) | `5.421` | — | — | — | `184,467.8 req/s` |
| Amortized session E2E (session encrypt + stateless enclave decrypt) | `78.166` | — | — | `130.293` | `12,793.3 req/s` |

### 4.2 Five-Axis Model-Class Comparison (`4B` / `9B` / `27B`, 512 Useful Tokens/Req)

GPU inference cost combines an explicit H100 reference profile ($\$2.49/\text{hr}$ plaintext vs. $\$2.65/\text{hr}$ H100 Confidential Computing + `3.1%–4.5%` CC enclave compute tax across `4B/9B/27B`) with live wall-clock attestation and AEAD binding latency measured by `attest-bench`:

| Model | Mode | Cost (`$/1M useful tok`) | TTFT (`ms`) | Effective Throughput (`tok/s`) | Adversarial Rejected | Completion Rate | Mandatory Hanzo Calls |
|---|---|---:|---:|---:|---:|---:|---:|
| `4B` | `plaintext_unattested_control` | `\$1.6711` | `18.000` | `413.889` | `0.0%` | `100.0%` | `0` |
| `4B` | **`hanzo_attest_rust_session_1000x`** | **`\$1.8587`** | **`18.889`** | **`396.042`** | **`100.0%`** | **`100.0%`** | **`0`** |
| `4B` | `hanzo_attest_rust_cold_per_req` | `\$1.8597` | `19.596` | `395.825` | `100.0%` | `100.0%` | `0` |
| `4B` | `hanzo_attest_python_cold_per_req` | `\$1.8923` | `42.279` | `389.004` | `100.0%` | `100.0%` | `0` |
| `9B` | `plaintext_unattested_control` | `\$2.6077` | `34.000` | `265.244` | `0.0%` | `100.0%` | `0` |
| `9B` | **`hanzo_attest_rust_session_1000x`** | **`\$2.8808`** | **`35.371`** | **`255.524`** | **`100.0%`** | **`100.0%`** | **`0`** |
| `9B` | `hanzo_attest_rust_cold_per_req` | `\$2.8818` | `36.078` | `255.434` | `100.0%` | `100.0%` | `0` |
| `9B` | `hanzo_attest_python_cold_per_req` | `\$2.9144` | `58.761` | `252.576` | `100.0%` | `100.0%` | `0` |
| `27B` | `plaintext_unattested_control` | `\$6.0680` | `78.000` | `113.986` | `0.0%` | `100.0%` | `0` |
| `27B` | **`hanzo_attest_rust_session_1000x`** | **`\$6.6582`** | **`80.497`** | **`110.556`** | **`100.0%`** | **`100.0%`** | **`0`** |
| `27B` | `hanzo_attest_rust_cold_per_req` | `\$6.6593` | `81.204` | `110.540` | `100.0%` | `100.0%` | `0` |
| `27B` | `hanzo_attest_python_cold_per_req` | `\$6.6919` | `103.887` | `110.001` | `100.0%` | `100.0%` | `0` |

### 4.3 Trust, Cross-Language Differential Parity & Sovereignty Summary

| Audit Dimension | Result | Parity / Notes |
|---|---:|---|
| Genuine multi-TEE attestations accepted (`nvidia_cc_h100`, `amd_sev_snp`, `intel_tdx`) | `3 / 3` (`100.0%`) | Identical `quote_hash` across Rust & Python |
| Adversarial vectors rejected — Rust (`hanzo-attest`) | `30 / 30` (`100.0%`) | Zero false accepts; invalid quotes do not poison replay cache |
| Adversarial vectors rejected — Python (`hanzo_attest`) | `30 / 30` (`100.0%`) | Clean-room independent stdlib verifier |
| Cross-language exact verdict + error-code parity | `33 / 33` (`100.0%`) | Identical canonical error code on every single vector |
| Bidirectional Rust ↔ Python AEAD request interop | `2 / 2` (`100.0%`) | Byte-for-byte deterministic ciphertext & Poly1305 tag parity |
| Sovereign `SovereignBundle` JSON export/import/fork round-trip | `100.0%` lossless | `0` mandatory Hanzo control-plane calls |

#### 30-Vector Adversarial Trust Matrix (`results.json:differential_matrix`)

| # | Attack Vector | Expected & Observed Error Code | Rust | Python | Parity |
|---:|---|---|:---:|:---:|:---:|
| 1 | `forged_root_signature` | `INVALID_PLATFORM_CERT_SIGNATURE` | Rejected | Rejected | Exact |
| 2 | `untrusted_root_id` | `UNTRUSTED_ROOT` | Rejected | Rejected | Exact |
| 3 | `revoked_root_key` | `ROOT_REVOKED` | Rejected | Rejected | Exact |
| 4 | `expired_root_cert` | `ROOT_EXPIRED` | Rejected | Rejected | Exact |
| 5 | `disallowed_tee_platform` | `PLATFORM_MISMATCH` | Rejected | Rejected | Exact |
| 6 | `revoked_platform_cert` | `PLATFORM_CERT_REVOKED` | Rejected | Rejected | Exact |
| 7 | `revoked_device_id` | `DEVICE_REVOKED` | Rejected | Rejected | Exact |
| 8 | `expired_platform_cert` | `PLATFORM_CERT_EXPIRED` | Rejected | Rejected | Exact |
| 9 | `quote_cert_id_mismatch` | `QUOTE_CERT_ID_MISMATCH` | Rejected | Rejected | Exact |
| 10 | `forged_quote_signature` | `INVALID_QUOTE_SIGNATURE` | Rejected | Rejected | Exact |
| 11 | `malleable_ed25519_s_ge_l` | `INVALID_QUOTE_SIGNATURE` | Rejected | Rejected | Exact |
| 12 | `debug_mode_enabled` | `DEBUG_MODE_ENABLED` | Rejected | Rejected | Exact |
| 13 | `secure_boot_disabled` | `SECURE_BOOT_DISABLED` | Rejected | Rejected | Exact |
| 14 | `memory_encryption_disabled` | `MEMORY_ENCRYPTION_DISABLED` | Rejected | Rejected | Exact |
| 15 | `tcb_downgrade` | `TCB_VERSION_TOO_LOW` | Rejected | Rejected | Exact |
| 16 | `tcb_cert_mismatch` | `TCB_CERT_MISMATCH` | Rejected | Rejected | Exact |
| 17 | `rtmr_runtime_mismatch` | `MEASUREMENT_MISMATCH` | Rejected | Rejected | Exact |
| 18 | `swapped_model_id` | `MODEL_ID_MISMATCH` | Rejected | Rejected | Exact |
| 19 | `swapped_model_weights` | `MODEL_WEIGHTS_MISMATCH` | Rejected | Rejected | Exact |
| 20 | `swapped_tokenizer` | `TOKENIZER_MISMATCH` | Rejected | Rejected | Exact |
| 21 | `swapped_workload_policy` | `WORKLOAD_POLICY_MISMATCH` | Rejected | Rejected | Exact |
| 22 | `stale_expired_quote` | `QUOTE_EXPIRED` | Rejected | Rejected | Exact |
| 23 | `quote_exceeds_max_age` | `QUOTE_TOO_OLD` | Rejected | Rejected | Exact |
| 24 | `wrong_client_nonce` | `NONCE_MISMATCH` | Rejected | Rejected | Exact |
| 25 | `replayed_client_nonce` | `NONCE_REPLAYED` | Rejected | Rejected | Exact |
| 26 | `stale_revocation_epoch` | `REVOCATION_EPOCH_STALE` | Rejected | Rejected | Exact |
| 27 | `mitm_unbound_ephemeral_x25519` | `REPORT_DATA_MISMATCH` | Rejected | Rejected | Exact |
| 28 | `low_order_x25519_point` | `INVALID_EPHEMERAL_KEY` | Rejected | Rejected | Exact |
| 29 | `tampered_aead_ciphertext` | `AEAD_AUTH_FAILED` | Rejected | Rejected | Exact |
| 30 | `swapped_aad_model_binding` | `BINDING_HEADER_MISMATCH` | Rejected | Rejected | Exact |

---

## 5. Explicit Honest Negative Regimes

Remote attestation and pure-software AEAD encryption are **not** free in every workload regime. `attest-bench` explicitly measures and reports three negative regimes:

### 5.1 Negative Regime 1 — Cold Per-Request Re-Attestation on Tiny 16-Token Prompts

Running a full cold quote verification + X25519 handshake on every tiny 16-token (`64 B`, `8.0 ms` base TTFT) prompt without session caching imposes a steep relative TTFT penalty:

| Client Mode (16-Token / `64 B` Prompt, `8.0 ms` Base TTFT) | Measured Overhead | Slowdown vs. Rust Session | TTFT Inflation |
|---|---:|---:|---:|
| Rust `AttestedSession` (amortized session channel) | `70.524 µs` (`0.071 ms`) | `1.000×` | `+0.88%` |
| **Rust cold per-request re-attestation** | **`799.948 µs` (`0.800 ms`)** | **`11.343×` slower** | **`+9.999%`** |
| **Python 3.12 stdlib cold per-request re-attestation** | **`23,468.5 µs` (`23.469 ms`)** | **`332.8×` slower** | **`+293.356%`** |

*Operational guidance:* Always reuse a time- and request-bounded `AttestedSession` for short interactive prompts; reserve per-request cold re-attestation for high-value single-shot sensitive payloads or session rotation boundaries.

### 5.2 Negative Regime 2 — Large Multi-Megabyte Context Encryption Overhead (`1–4 MiB`)

Because `hanzo-attest` uses portable, zero-dependency, `#![forbid(unsafe_code)]` scalar ChaCha20-Poly1305 (no hand-written AVX2/NEON assembly), encrypting and decrypting multi-megabyte long-context payloads incurs measurable millisecond latency compared to plaintext zero-copy dispatch:

| Context Payload | Pure-Rust Encrypt (`ms`) | Pure-Rust Decrypt (`ms`) | Round-Trip AEAD Tax (`ms`) | Encrypt Throughput (`MiB/s`) |
|---:|---:|---:|---:|---:|
| `1 MiB` (`1,048,576 B`) | `6.759` | `4.959` | **`11.719`** | `147.941` |
| `2 MiB` (`2,097,152 B`) | `12.199` | `10.393` | **`22.592`** | `163.942` |
| `4 MiB` (`4,194,304 B`) | `23.961` | `20.778` | **`44.739`** | `166.939` |

*Operational guidance:* For multi-megabyte RAG/document contexts on latency-critical paths, stream chunked AEAD frames overlapped with network I/O or negotiate an optional hardware-accelerated AES-GCM/SIMD cipher suite after the `hanzo-attest` handshake.

### 5.3 Negative Regime 3 — Pure-Python Stdlib Client Throughput vs. Compiled Rust Client

The clean-room Python 3.12 stdlib client provides maximum auditability and independent verification diversity (zero pip dependencies), but pays a substantial interpreted-bytecode throughput penalty relative to compiled Rust (`--release`):

| Operation (`1 KiB` Prompt) | Compiled Rust (`µs`) | Pure Python 3.12 Stdlib (`µs`) | Rust Speedup |
|---|---:|---:|---:|
| Cold attestation verification | `572.013` | `15,467.802` | **`27.041×`** |
| Cold verify + X25519 + HKDF + AEAD encrypt | `711.128` | `23,468.504` | **`33.002×`** |
| Session-cached ChaCha20-Poly1305 encrypt | `5.421` | `3,169.366` | **`584.684×`** |

*Operational guidance:* Use the pure-Python client for independent auditing, CLI inspection, CI differential verification, and low-QPS scripts; use the compiled Rust `hanzo-attest` crate on latency-sensitive or high-QPS production inference paths.
