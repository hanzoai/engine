# HIP-1346 Research Dossier: Position-Shifted Prompt Reuse for Tool & System Prompts (`hanzo-cacheslide`)

- **Tracking Issue**: `hanzoai/engine#85` (`p1`, `bounty`, `perf`, `research`, `hip-1346`)
- **Canonical Owning Repository**: `hanzoai/engine`
- **Crate**: [`hanzo-cacheslide/`](../../../hanzo-cacheslide/) (`Apache-2.0`, `#![forbid(unsafe_code)]`, zero external crate dependencies)
- **Raw Receipts**: [`results.json`](./results.json)
- **Canonical Architecture Boundary**:
  > `vfs -> Hanzo S3 -> Visor`; `Cloud` owns lifecycle and billing; `Enso` routes models; **`Zen` generates**; **`Kai` decides**.
  >
  > This research improves **KV cache reuse and prefill TTFT/GPU-memory efficiency inside Hanzo Zen (`hanzo-engine` / `hanzo-kv-fabric` / `hanzo-paged-attn`)** without introducing a parallel scheduler, router, or KV fabric. It provides the missing **RoPE position-shift and attention-path selective recomputation kernel** complementary to `hanzo-kv-fabric/src/composition.rs` (HIP-1345 §2.7 planning layer).

---

## 1. Primary References, Technical Summary & License Audit

### Primary Literature
1. **CacheSlide**: Liu, Yang et al. — *CacheSlide: Position-Shifted Prompt Reuse for LLM Serving* (USENIX FAST 2026), <https://www.usenix.org/conference/fast26/presentation/liu-yang>.
2. **Related Prior Work in Hanzo**: CacheBlend (Yao et al., EuroSys 2025, <https://arxiv.org/abs/2405.16444>), whose fabric-level planning layer already ships in [`hanzo-kv-fabric/src/composition.rs`](../../../hanzo-kv-fabric/src/composition.rs) (HIP-1345 §2.7), explicitly leaving RoPE translation, KV shifting, and attention-path quality measurement to Phase 2.

### Core Technical Mechanism
Ordinary prefix caching ([`hanzo-engine/src/prefix_cacher.rs`](../../../hanzo-engine/src/prefix_cacher.rs) and [`hanzo-engine/src/paged_attention/block_hash.rs`](../../../hanzo-engine/src/paged_attention/block_hash.rs)) chain-hashes KV blocks sequentially from token 0. Whenever an earlier prompt segment changes — e.g., dynamic user metadata/timestamps in a system prompt, reordered MCP tool schemas, mid-prompt state reminders, or reranked RAG chunks — **100% of downstream KV blocks miss**, forcing redundant prefill FLOPs and duplicating identical tool-schema KV blocks across GPU memory for every distinct session header.

`hanzo-cacheslide` eliminates rigid prefix coupling via four clean-room mechanisms:
1. **Exact RoPE Relative Rotation $R(\Delta p)$ (`src/rope.rs`)**:
   Because 2D Rotary Position Embeddings form an abelian group $SO(2)^{d_r/2}$ with $R(a)\,R(b) = R(a + b)$, every token $t \in [0, L)$ in a segment cached at start offset $p_{\text{orig}}$ can be shifted to a new start offset $p_{\text{new}}$ with a **single constant delta** $\Delta p = p_{\text{new}} - p_{\text{orig}}$:
   $$K_{\text{shifted}}[t] = R(\Delta p)\,K_{\text{orig}}[t] = R(p_{\text{new}} + t)\,K_{\text{raw}}[t], \qquad V_{\text{shifted}}[t] = V_{\text{orig}}[t].$$
   A single precomputed `RopeShiftTable` of $d_r / 2$ `(cos, sin)` pairs rotates an entire segment across all tokens, KV heads, and layers without re-running Q/K/V or SwiGLU projections. Both canonical `hanzo-engine/src/layers.rs` layouts are supported (`GptNeoX` half-split and `Interleaved`), along with partial RoPE (`rotary_dim <= head_dim`) and linear frequency scaling.
2. **Content-Addressed, Tenant/ACL-Isolated Segment KV Pool (`src/segment.rs`, `src/cache.rs`)**:
   Segments are keyed by length-prefixed, domain-separated FIPS 180-4 SHA-256 plus exact structural equality on `(tenant_id, acl_group, model_id, kv_dtype, segment_sha256)`, enforcing `0.0%` cross-tenant/cross-ACL reuse, strict `current_bytes <= max_bytes` LRU eviction, instant `evict_tenant(tenant_id)`, optional SHA-256 payload integrity checks, and a configurable `min_segment_tokens` admission gate (`32` default).
3. **Selective Boundary & Cross-Segment Recomputation + Layer-1 Drift Gate (`src/attention.rs`)**:
   Layer-0 shifted KV is mathematically exact (`< 1e-5` f32 error), whereas Layers $\ge 1$ reflect the segment's original preceding context. `CacheSlideSelectiveRecompute` recomputes leading/trailing boundary sink tokens plus the highest Layer-0 cross-segment query-affinity tokens to trade off TTFT vs. KL divergence smoothly, and includes an optional **automatic Layer-1 relative K-drift probe** (`||K_new^(1) - K_shifted^(1)||_2 / ||K_new^(1)||_2`) that automatically escalates recomputation when cross-segment coupling is high.
4. **No Cross-Turn Stale-Approximation Compounding (`exact_prefix_valid`)**:
   Any segment computed downstream of a shifted/approximated segment is tagged `exact_prefix_valid = false` inside `SegmentKvPayload` (and covered by the payload SHA-256 digest), preventing approximate KV from ever masquerading as an exact prefix block on subsequent turns.

### License, Dependency & Patent Audit
| Artifact | License | Audit Disposition |
|---|---|---|
| USENIX FAST '26 CacheSlide paper (conceptual reference only) | Academic publication | **Clear** — clean-room Rust implementation from first-principles RoPE group algebra; zero third-party source code copied. |
| `hanzo-cacheslide` (this PR) | Apache-2.0 (compatible with workspace MIT / Apache-2.0) | **Clear** — **0 external crate dependencies** (pure Rust `std` including clean-room FIPS 180-4 SHA-256), `#![forbid(unsafe_code)]`, offline release build in `< 3s`. |
| Benchmark prompt corpora | Derived deterministically from this repository's own MIT/Apache-2.0 files (`chat_templates/*.jinja`, `docs/*.md`, `hanzo-kv-fabric/src/composition.rs`, `hanzo-engine/src/paged_attention/*.rs`) | **Clear** — no proprietary corpora, no restricted datasets, no external network calls. |
| Patents | No patent claims identified on RoPE group composition or content-addressed KV block caching | **Clear**. |

> Published CacheSlide paper figures are NOT claimed as Hanzo metrics. All numbers below were measured directly on this machine via `cacheslide-bench --release --offline --seed 1346 --trials 6`.

---

## 2. Reproducible Baseline & Hardware Environment

### Host Hardware & Toolchain
- **CPU**: `Intel(R) Xeon(R) CPU @ 2.30GHz` (2 vCPUs, x86_64 KVM, AVX2/FMA/BMI2, 32 KiB L1d, 256 KiB L2, 45 MiB L3)
- **Toolchain**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `cargo 1.100.0 (3acc79c0c 2026-11-05)`
- **Model Evaluated**: Real pure-Rust `f32` 6-layer causal GQA Transformer (`d_model=64`, `num_heads=4`, `num_kv_heads=2`, `head_dim=16`, `d_ff=128` SwiGLU, `vocab_size=1024`, NeoX RoPE $\theta = 10{,}000$, Pre-Norm RMSNorm, `FP16` GPU KV accounting, deterministic seed `1346`).
- **Sampling**: `6` full trials × `5` distinct eval prompts per workload (`30` timed forward passes per mode), `1` unmeasured warmup pass, `--release --offline`.

### Matched Baselines
1. **`full_prefill_baseline`**: Exact causal forward pass over all prompt tokens from scratch (`KL = 0.0`, `Top1 = 100.0%`).
2. **`prefix_cache_only`**: Matched chain-hashed prefix cache (`PrefixKvCache`) under identical `IsolationScope` and `64 MiB` byte budget (mirrors `hanzo-engine/src/paged_attention/block_hash.rs`).
3. **`cacheslide_pure_shift`**: Non-prefix `SegmentKvCache` reuse with exact $R(\Delta p)$ RoPE shift and zero boundary recomputation.
4. **`cacheslide_selective_recompute`**: Non-prefix `SegmentKvCache` reuse with exact $R(\Delta p)$ RoPE shift + `20%` selective boundary/cross-segment recomputation (plus automatic Layer-1 K-drift escalation gate `threshold = 0.08` on `neg1_high_cross_segment_coupling`).

### Exact Reproducible Commands
```bash
# Format, clippy (-D warnings), and release unit/security/concurrency test suite (offline)
cargo fmt -p hanzo-cacheslide -- --check
cargo clippy -p hanzo-cacheslide --all-targets --offline -- -D warnings
cargo test -p hanzo-cacheslide --offline --release

# Reproduce full benchmark, Pareto sweep, GPU memory dedup, isolation audit & results.json
cargo run -p hanzo-cacheslide --release --offline --bin cacheslide-bench -- --seed 1346 --trials 6
```

---

## 3. Falsifiable Hypothesis & Explicitly Tested Negative Regimes

### Falsifiable Hypotheses
- **(H1 — RoPE Exactness & Zero-Leak Isolation)**: Direct key shifting $R(a) R(b) = R(a + b)$ holds to **$< 10^{-5}$ `f32` max absolute error** across `GptNeoX` and `Interleaved` layouts, and cross-scope probes across `(tenant_id, acl_group, model_id, kv_dtype)` incur **`0.0%` leaks (`0 / 100`)**.
- **(H2 — Non-Prefix TTFT & GPU Memory Gain over Prefix Caching)**: On realistic multi-turn tool-calling and shuffled RAG workloads where preceding context shifts downstream blocks, `cacheslide_pure_shift` and `cacheslide_selective_recompute (20%)` achieve **$\ge 2.0\times$ p50 TTFT speedup** and **$> 50\%$ FLOP savings** where `prefix_cache_only` stalls near `0.96x–1.11x`, while deduplicating shared tool KV across 16 same-tenant sessions by **$> 3.0\times$ GPU KV memory**.
- **(H3 — Monotonic TTFT vs. KL Pareto Frontier)**: Increasing selective recomputation ratio $r \in [0.0, 1.0]$ smoothly reduces KL divergence toward `0.000000` (`CosSim -> 1.000000`, `Top1 -> 100.0%` at $r = 1.0$) while preserving $> 1.8\times$ TTFT speedup at $r \le 0.25$.

### Explicitly Tested Negative Regimes (Expected Failure Modes)
- **(F1 — High Cross-Segment Semantic Coupling / Order-Sensitive Context, `neg1_high_cross_segment_coupling`)**:
  A 96-token segment originally cached at `pos = 0` (zero prior context) is moved to `pos = 256` after a dense contrasting 256-token prefix, followed by an 8-token query directly sensitive to the coupled segment.
  - **Expected & Observed Failure**: `cacheslide_pure_shift` suffers elevated KL divergence (**`0.13534` nats > `0.10` threshold**) and drops Top-1 agreement to **`40.0%`**.
  - **Mitigation Verified**: Enabling the automatic Layer-1 K-drift gate (`threshold = 0.08`) measures `max_layer1_k_drift = 0.884973`, fires on **`5 / 5`** prompts, escalates recomputation to `100%`, and restores **`KL = 0.000000`, `CosSim = 1.000000`, `Top1 = 100.0%`** at `0.95x` (`-0.25%` probe FLOP overhead).
- **(F2 — Tiny / Unique Micro-Segments `< 32` tokens, `neg2_tiny_unique_segments`)**:
  24 ephemeral 12-token segments (`288` tokens total) evaluated with an ungated cache (`min_segment_tokens = 0`).
  - **Expected & Observed Failure**: SHA-256 hashing, mutex lookup, and KV block extraction/insertion deliver `0.0%` reuse and **regress p50 TTFT to `0.89x`** (`103.78 ms` vs `92.63 ms` direct prefill).
  - **Mitigation Verified**: Enforcing `min_segment_tokens = 32` bypasses tiny segments cleanly (`short_segment_skips`), preventing cache pollution and overhead.

---

## 4. Bounded Implementation Summary

`hanzo-cacheslide/` is registered cleanly in workspace [`Cargo.toml`](../../../Cargo.toml) (`Apache-2.0`, `#![forbid(unsafe_code)]`, zero external crate dependencies):
- [`src/sha256.rs`](../../../hanzo-cacheslide/src/sha256.rs): Clean-room FIPS 180-4 SHA-256 with length-prefixed domain separation (verified against NIST test vectors).
- [`src/rope.rs`](../../../hanzo-cacheslide/src/rope.rs): Exact `RopeConfig`, `RopeCache`, `RopeShiftTable` (`R(Δp)` in `f64` trig / `f32` apply), `GptNeoX` + `Interleaved` layouts, partial RoPE, linear frequency scaling, and fail-closed `checked_position_shift`.
- [`src/segment.rs`](../../../hanzo-cacheslide/src/segment.rs): `KvDtype` (`F32`/`F16`/`BF16`/`Fp8E4M3`/`Int4Asym`), `IsolationScope`, `SegmentKey`, `PrefixChainKey`, `PromptSegment`, `SegmentKvPayload` (with `exact_prefix_valid` taint bit and SHA-256 integrity digest), and deterministic `CorpusTokenizer`.
- [`src/cache.rs`](../../../hanzo-cacheslide/src/cache.rs): Thread-safe bounded LRU `SegmentKvCache` and matched `PrefixKvCache` enforcing `current_bytes <= max_bytes`, `min_segment_tokens`, `evict_tenant`, and optional integrity verification.
- [`src/attention.rs`](../../../hanzo-cacheslide/src/attention.rs): Pure-Rust multi-layer causal GQA + RoPE + SwiGLU `TransformerModel` executing all four modes, Layer-0 cross-segment affinity selection, Layer-1 boundary K-drift probing, exact FLOP counting, and `evaluate_quality` (`KL`, `JS`, cosine similarity, `Top-1`, `Top-5`, entropy).
- [`src/bin/cacheslide-bench.rs`](../../../hanzo-cacheslide/src/bin/cacheslide-bench.rs): Deterministic CLI benchmark emitting [`results.json`](./results.json).

---

## 5. Correctness, Concurrency, Security & Regression Evidence

- **Test Suite & Static Checks**: **`20 / 20` passing tests** (`cargo test -p hanzo-cacheslide --offline --release`), **`0` clippy warnings** (`cargo clippy -p hanzo-cacheslide --all-targets --offline -- -D warnings`), clean `cargo fmt -p hanzo-cacheslide -- --check`, zero `unsafe` (`#![forbid(unsafe_code)]`), zero external dependencies, offline release build in `~2.6–2.9s`.
- **RoPE Mathematical Verification (`results.json#rope_group_audit`)**:
  - `R(a) R(b) == R(a + b)` max absolute `f32` error: **`2.384e-7`** (`GptNeoX`) and **`1.192e-7`** (`Interleaved`), both well under `1e-5` across positive, negative, and multi-hop shifts (`+256`, `-192`, `+512`, `-3072`).
  - Relative inner-product invariance `<R(p+d)q, R(p)k> == <R(p'+d)q, R(p')k>`, L2 norm preservation, `R(-p) R(p) == I`, partial RoPE unrotated-dim invariance, and fail-closed `checked_position_shift` overflow checks all verified.
- **No Stale-Approximation Compounding**: Verified by `exact_prefix_taint_prevents_stale_approximation_compounding` — downstream segments computed after pure-shifted blocks are tagged `exact_prefix_valid = false`, and `100%` selective recomputation matches exact `FullPrefill` to `< 1e-6` KL divergence.
- **Security & Multi-Tenant Isolation (`results.json#tenant_isolation_audit`)**:
  - **`0 / 100` cross-scope leaks (`0.0%`)** across `25` cross-tenant, `25` cross-ACL, `25` cross-model, and `25` cross-dtype probes.
  - `evict_tenant("tenant_alpha")` purged all `3` entries for `tenant_alpha` while retaining `tenant_beta` intact (`beta_retained_after_alpha_eviction = true`).
  - `integrity_check_mode_rejects_corrupted_payload` verifies tampered KV floats are rejected and evicted on read.
- **Concurrency & Bounded Memory**:
  - 4-thread concurrent stress test (`concurrent_multithreaded_cache_stress`) and adversarial LRU eviction test (`lru_enforces_exact_byte_budget_and_evicts_oldest`) confirm `current_bytes <= peak_bytes <= max_bytes` and fail-closed rejection (`InsertOutcome::RejectedExceedsBudget`) of oversized payloads.

---

## 6. Before/After Measurements, Raw Results & Canonical Integration Plan

All figures below are copied directly from [`results.json`](./results.json) (`--seed 1346 --trials 6`, 5 distinct eval prompts × 6 trials = 30 runs/mode, `--release --offline`).

### Table A — End-to-End Workload Results (4 Positive + 2 Negative Regimes)

| Workload | Regime | Mode | p50 (ms) | p95 (ms) | p99 (ms) | Speedup vs Full | Reuse % | FLOPs Saved % | KL (nats) | CosSim | Top-1 % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tool_schema_shifted_system` | Positive (`Δp=+64`) | `full_prefill_baseline` | 150.62 | 154.94 | 155.31 | 1.00x | 0.0% | 0.0% | 0.00000 | 1.00000 | 100.0% |
| `tool_schema_shifted_system` | Positive (`Δp=+64`) | `prefix_cache_only` | 154.09 | 169.55 | 169.82 | 0.98x | 0.0% | 0.0% | 0.00000 | 1.00000 | 100.0% |
| `tool_schema_shifted_system` | Positive (`Δp=+64`) | `cacheslide_pure_shift` | **53.31** | 59.48 | 59.58 | **2.83x** | **61.5%** | **67.4%** | 0.15210 | 0.96624 | 40.0% |
| `tool_schema_shifted_system` | Positive (`Δp=+64`) | `cacheslide_selective_recompute` | **75.00** | 78.33 | 78.55 | **2.01x** | 49.0% | **53.1%** | **0.11189** | **0.97601** | **60.0%** |
| `multi_tool_subset_reorder` | Positive (reordered MCP) | `full_prefill_baseline` | 117.39 | 122.47 | 122.68 | 1.00x | 0.0% | 0.0% | 0.00000 | 1.00000 | 100.0% |
| `multi_tool_subset_reorder` | Positive (reordered MCP) | `prefix_cache_only` | 105.74 | 115.79 | 124.67 | 1.11x | 29.0% | 21.9% | 0.00000 | 1.00000 | 100.0% |
| `multi_tool_subset_reorder` | Positive (reordered MCP) | `cacheslide_pure_shift` | **14.91** | 42.52 | 43.61 | **7.87x** | **87.2%** | **83.2%** | 0.11805 | 0.97554 | **80.0%** |
| `multi_tool_subset_reorder` | Positive (reordered MCP) | `cacheslide_selective_recompute` | **34.65** | 58.33 | 59.55 | **3.39x** | 72.7% | **67.9%** | **0.10656** | **0.97803** | **80.0%** |
| `rag_shuffled_chunks` | Positive (reranked docs) | `full_prefill_baseline` | 200.49 | 210.16 | 211.00 | 1.00x | 0.0% | 0.0% | 0.00000 | 1.00000 | 100.0% |
| `rag_shuffled_chunks` | Positive (reranked docs) | `prefix_cache_only` | 208.07 | 215.14 | 224.97 | 0.96x | 0.0% | 0.0% | 0.00000 | 1.00000 | 100.0% |
| `rag_shuffled_chunks` | Positive (reranked docs) | `cacheslide_pure_shift` | **43.76** | 46.53 | 46.70 | **4.58x** | **75.0%** | **79.3%** | 0.16523 | 0.95268 | **80.0%** |
| `rag_shuffled_chunks` | Positive (reranked docs) | `cacheslide_selective_recompute` | **78.07** | 82.72 | 84.94 | **2.57x** | 59.8% | **62.6%** | **0.15977** | **0.95706** | **80.0%** |
| `multiturn_sliding_history` | Positive (mid-insert `+64`) | `full_prefill_baseline` | 163.44 | 168.38 | 172.89 | 1.00x | 0.0% | 0.0% | 0.00000 | 1.00000 | 100.0% |
| `multiturn_sliding_history` | Positive (mid-insert `+64`) | `prefix_cache_only` | 152.78 | 159.25 | 162.71 | 1.07x | 14.3% | 8.9% | 0.00000 | 1.00000 | 100.0% |
| `multiturn_sliding_history` | Positive (mid-insert `+64`) | `cacheslide_pure_shift` | **37.50** | 40.40 | 40.43 | **4.36x** | **78.6%** | **79.2%** | **0.03909** | **0.98877** | 60.0% |
| `multiturn_sliding_history` | Positive (mid-insert `+64`) | `cacheslide_selective_recompute` | **62.08** | 65.58 | 67.38 | **2.63x** | 65.6% | **64.6%** | **0.03324** | **0.99079** | 60.0% |
| `neg1_high_cross_segment_coupling` | **Negative #1 (coupled)** | `full_prefill_baseline` | 122.92 | 133.25 | 134.51 | 1.00x | 0.0% | 0.0% | 0.00000 | 1.00000 | 100.0% |
| `neg1_high_cross_segment_coupling` | **Negative #1 (coupled)** | `prefix_cache_only` | 128.15 | 132.87 | 137.35 | 0.96x | 0.0% | 0.0% | 0.00000 | 1.00000 | 100.0% |
| `neg1_high_cross_segment_coupling` | **Negative #1 (coupled)** | `cacheslide_pure_shift` | 86.45 | 88.94 | 90.77 | 1.42x | 26.7% | 33.7% | **0.13534** | 0.96840 | **40.0%** |
| `neg1_high_cross_segment_coupling` | **Negative #1 (drift-gated)** | `cacheslide_selective_recompute` | 129.30 | 132.86 | 133.79 | 0.95x | 0.0% | -0.25% | **0.00000** | **1.00000** | **100.0%** |
| `neg2_tiny_unique_segments` | **Negative #2 (`<32` tok)** | `full_prefill_baseline` | 92.63 | 98.05 | 98.05 | 1.00x | 0.0% | 0.0% | 0.00000 | 1.00000 | 100.0% |
| `neg2_tiny_unique_segments` | **Negative #2 (`<32` tok)** | `prefix_cache_only` | 97.56 | 101.14 | 101.14 | 0.95x | 0.0% | 0.0% | 0.00000 | 1.00000 | 100.0% |
| `neg2_tiny_unique_segments` | **Negative #2 (`<32` tok)** | `cacheslide_pure_shift` (ungated) | **103.78** | 116.77 | 116.77 | **0.89x** | 0.0% | 0.0% | 0.00000 | 1.00000 | 100.0% |
| `neg2_tiny_unique_segments` | **Negative #2 (`<32` tok)** | `cacheslide_selective_recompute` | 97.62 | 112.72 | 112.72 | **0.95x** | 0.0% | 0.0% | 0.00000 | 1.00000 | 100.0% |

### Table B — Empirical TTFT vs. Accuracy Pareto Frontier (`tool_schema_shifted_system` & `rag_shuffled_chunks`)

| Recompute Ratio $r$ | Tool p50 (ms) | Tool Speedup | Tool FLOPs Saved | Tool KL (nats) | Tool CosSim | Tool Top-1 | RAG p50 (ms) | RAG Speedup | RAG FLOPs Saved | RAG KL (nats) | RAG CosSim | RAG Top-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.00` | 53.37 | **2.82x** | 67.41% | 0.152096 | 0.966237 | 40.0% | 48.92 | **4.10x** | 79.33% | 0.165234 | 0.952685 | 0.64 |
| `0.05` | 61.60 | 2.45x | 63.23% | 0.142122 | 0.968688 | 40.0% | 60.50 | 3.31x | 74.40% | 0.156718 | 0.955270 | 0.68 |
| `0.10` | 67.12 | 2.24x | 59.83% | 0.133071 | 0.970608 | 60.0% | 64.69 | 3.10x | 70.65% | 0.150618 | 0.957482 | 0.64 |
| `0.15` | 72.35 | 2.08x | 56.45% | 0.118739 | 0.973988 | 60.0% | 71.17 | 2.82x | 66.31% | 0.164761 | 0.955752 | 0.60 |
| `0.25` | 82.05 | 1.84x | 50.11% | 0.102270 | 0.978253 | 60.0% | 86.43 | 2.32x | 58.84% | 0.154284 | 0.959351 | 0.72 |
| `0.40` | 94.57 | 1.59x | 40.07% | 0.080227 | 0.982963 | 60.0% | 108.83 | 1.84x | 46.32% | 0.123491 | 0.970446 | 0.72 |
| `0.60` | 112.95 | 1.33x | 26.91% | 0.047096 | 0.990100 | 80.0% | 140.18 | 1.43x | 30.67% | 0.042175 | 0.988155 | 0.88 |
| `1.00` | 151.19 | 1.00x | -0.05% | **0.000000** | **1.000000** | **100.0%** | 199.62 | 1.00x | -0.05% | **0.000000** | **1.000000** | **1.00** |

### Table C — Multi-Session GPU KV Memory Deduplication (16 Same-Tenant Sessions × Shared 256-Token Tool Schema)

| Metric | `PrefixKvCache` (Baseline) | `SegmentKvCache` (CacheSlide) | Reduction |
|---|---:|---:|---:|
| Cached KV Entries (16 sessions) | 48 | **33** | 1.45x fewer entries |
| Measured Benchmark GPU KV Bytes (`FP16`, 6L) | 4,325,376 B | **1,376,256 B** | **3.14x less GPU RAM** |
| Projected Llama-3-8B GPU KV (`FP16`, 32L/8KVH/128D) | 704.0 MB | **224.0 MB** | **3.14x** (480.0 MB saved) |
| Projected Llama-3-8B GPU KV (`KIVI Int4Asym`) | 704.0 MB *(FP16)* | **70.0 MB** | **10.06x** combined |
| Strict Byte Budget Respected (`peak <= max`) | `true` | `true` | Zero budget overruns |

### Canonical Hanzo Zen Integration & Rollout Plan (HIP-1346)
1. **Integration Seams in Canonical Repositories (No Parallel Engines)**:
   - **KV Fabric Planner -> Engine Attention Seam**: Consume `CompositionDecision::Reuse { position_shift, recompute_window, sha256 }` already emitted by [`hanzo-kv-fabric/src/composition.rs`](../../../hanzo-kv-fabric/src/composition.rs) directly inside `hanzo-engine`'s prefill path.
   - **RoPE Kernel Integration**: Add a `shift_cached_keys(k_cache, delta_pos, is_gpt_neox)` kernel alongside [`hanzo-engine/src/layers.rs`](../../../hanzo-engine/src/layers.rs) (`RotaryEmbedding`, `apply_rotary_positions_qk`) and `hanzo-paged-attn` CUDA/Metal/ROCm block-gather kernels (note: a single `RopeShiftTable(delta_pos)` applies across every token, head, and layer in a shifted segment).
   - **Tenant/ACL Isolation & Block Hashing**: Pair `SegmentKey(IsolationScope, segment_sha256)` with `ExtraHashKey::CacheSalt` in [`hanzo-engine/src/paged_attention/block_hash.rs`](../../../hanzo-engine/src/paged_attention/block_hash.rs) and extend [`hanzo-engine/src/paged_attention/kv_isolation.rs`](../../../hanzo-engine/src/paged_attention/kv_isolation.rs) to assert `0.0%` cross-tenant shifted reuse and `exact_prefix_valid` taint invariants.
   - **Mandatory Production Guards (Derived from Negative Regimes #1 & #2)**:
     1. Gate admission with `min_segment_tokens >= 32` (prevents Negative Regime #2 `0.89x` latency regression).
     2. Gate shifted reuse behind the Layer-1 relative K-drift probe (`auto_escalate_drift_threshold`) so order-sensitive / high-coupling contexts automatically escalate to full recomputation (prevents Negative Regime #1 accuracy drop).
     3. Never mark KV blocks computed downstream of shifted segments as `exact_prefix_valid`.
2. **PR Convergence Metadata & Rollout / Rollback Plan (`PR_CONVERGENCE.md`)**:
   - **Canonical Owning Repository**: `hanzoai/engine`
   - **Qualification Criteria**: Zero `unsafe`, zero external deps, `20/20` tests green, `clippy -D warnings` clean, `RoPE` group error `< 1e-5`, `0/100` cross-tenant leaks, reproducible `--seed 1346` receipts in `results.json`.
   - **Phase 1 (This PR — Research Bounty Deliverable)**: Standalone `hanzo-cacheslide` crate + reproducible benchmark + HIP-1346 dossier (`0` behavioral changes to shipping `hanzo-engine` paths).
   - **Phase 2 (Follow-Up Implementation Bounty)**: Wire `RopeShiftTable` + Layer-1 drift gate into `hanzo-paged-attn` GPU kernels and `hanzo-engine/src/prefix_cacher.rs` behind opt-in `HANZO_CACHESLIDE=1` flag, then qualify on real Zen (Qwen/Llama/Gemma4) weights and confidential Hanzo internal tool-calling corpora (aggregate receipts only).
   - **Rollback**: Single runtime flag (`HANZO_CACHESLIDE=0`) reverts to `prefix_cache_only`; `git revert` removes `hanzo-cacheslide/` cleanly with zero storage or schema migration impact.
