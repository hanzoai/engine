# HIP-1346 Research & Implementation: Persistent, Shareable Multi-Tier Prefix Caches (`hanzoai/engine#51`)

## 1. Primary References & Clean-Room License Audit

- **RadixAttention**: Lianmin Zheng et al., *"SGLang: Efficient Execution of Structured Language Model Programs"*, NeurIPS 2024 (`https://arxiv.org/abs/2312.07104`).
- **Tiered KV Persistence**: Hierarchical HBM / Host-DRAM-SSD / Shared Object-Store KV offloading with bandwidth-crossover admission gating.
- **Hybrid Attention + GDN Safety**: `docs/HIP-1346-kivi-kv-compression.md` — recurrent Gated DeltaNet (GDN) states are checkpointed at exact block boundaries in unquantized `FP32` precision alongside Attention KV pages.
- **License & Dependency Audit**: 100% clean-room Rust workspace crate `hanzo-prefixcache` (`Apache-2.0`, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`, **0 external crate dependencies**).

## 2. Reproducibility & Hardware Environment

Reproduce exact artifacts from repository root:

```bash
cargo test -p hanzo-prefixcache --offline --release
cargo run --release --offline -p hanzo-prefixcache --bin prefixcache-bench -- --seed 1346 --out-dir docs/research/hip-1346-persistent-prefix-cache
```

- **CPU Architecture**: `Intel(R) Xeon(R) CPU @ 2.30GHz (x86_64)`
- **OS / Kernel**: `Linux 6.18.57`
- **GPU / Driver**: `none (host-measured CPU/IO control plane + analytical 4B/9B/27B GPU model)`
- **Git Commit**: `df1722952421`
- **Seed / Block Size**: `--seed 1346`, `block_size = 64` tokens
- **Versioned Corpus Provenance (SHA-256)**:
  - `calibration_data/calibration_datav3_small.txt`: `aabd956dba34ff31f40e78b9de4e234fdd4865f90ba231e8e3deec8dfb22cbc0`
  - `chat_templates/llama3.json`: `67d1d402738727aa3059d28590fa3488c44740a358be4b5c2135c60f58ec9d7a`
  - `orderings/xlora-paper-ordering.json`: `8d0650ea231eeade0b97e5c026411365ff9feed41b2b07ecf10853c2a399043a`

> **Methodology Transparency**: Radix lookup, 7-field SHA-256 key derivation, CRC32 + SHA-256 snapshot verification, binary Tier-1/Tier-2 filesystem persistence, cold-restart reload, multi-worker prefix-aware scheduling, 200-probe tenant isolation, and bit-exact KV+GDN state equivalence execute physically on the host (`host_control_plane_ms`). End-to-end GPU prefill/decode timings and `analytical_mj_per_token` combine those real host-measured control-plane timings with an explicit analytical 4B/9B/27B compute + HBM/PCIe/NIC transfer + TDP power model.

## 3. Shared `4B / 9B / 27B` 20-Turn Developer-Agent Scorecard

Workload: 20-turn developer-agent loop (`512`-token `PublicSystem` prompt + `1,536`-token `OrgShared` repo reads + growing `TenantPrivate` edits with 4 tail tool-result mutations on turns 6/11/16/19, burst-load Worker 0 -> Worker 1 handoff on turn 8, and Worker 0 cold restart wiping HBM on turn 13).

| Model | Quant | Regime | Cold TTFT (ms) | Warm TTFT p50 / p95 (ms) | ITL p50 / p95 (ms) | Recomputed Tokens (Saved %) | 20-Turn Wall Time (ms) | Peak Mem (MiB) | Analytical Energy (mJ/tok) | Bit-Exact / Quality Reg | Tier Hits (T0 / T1 / T2) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `4B-Hybrid` | `fp16` | `no_cache_baseline` | 97.28 | 191.34 / 285.74 | 4.79 / 5.05 | 85120 (0.0%) | 12852.2 | 136.1 | 27.052 | 20/20 (0.0) | 0 / 0 / 0 |
| `4B-Hybrid` | `fp16` | `hbm_only_radix_cache` | 101.28 | 144.93 / 289.81 | 4.79 / 5.05 | 71680 (15.8%) | 12363.0 | 136.1 | 25.922 | 20/20 (0.0) | 210 / 0 / 0 |
| `4B-Hybrid` | `fp16` | `naive_disk_kv_every_turn` | 101.45 | 27.76 / 236.41 | 4.79 / 5.05 | 14464 (83.0%) | 10774.9 | 136.1 | 21.040 | 20/20 (0.0) | 461 / 643 / 0 |
| `4B-Hybrid` | `fp16` | `hanzo_tiered_prefix_cache` | 109.28 | 28.35 / 60.34 | 4.79 / 5.05 | 6208 (92.7%) | 9761.1 | 136.1 | 20.150 | 20/20 (0.0) | 487 / 691 / 55 |
| `9B-Hybrid` | `fp16` | `no_cache_baseline` | 188.16 | 362.40 / 540.42 | 9.62 / 10.05 | 85120 (0.0%) | 25393.4 | 287.5 | 84.138 | 20/20 (0.0) | 0 / 0 / 0 |
| `9B-Hybrid` | `fp16` | `hbm_only_radix_cache` | 187.59 | 272.63 / 545.04 | 9.62 / 10.05 | 71680 (15.8%) | 24409.9 | 287.5 | 80.695 | 20/20 (0.0) | 210 / 0 / 0 |
| `9B-Hybrid` | `fp16` | `naive_disk_kv_every_turn` | 192.09 | 47.69 / 427.07 | 9.62 / 10.05 | 14464 (83.0%) | 21448.6 | 287.5 | 65.760 | 20/20 (0.0) | 461 / 643 / 0 |
| `9B-Hybrid` | `fp16` | `hanzo_tiered_prefix_cache` | 199.88 | 48.07 / 115.06 | 9.62 / 10.05 | 6208 (92.7%) | 19394.8 | 287.5 | 63.197 | 20/20 (0.0) | 486 / 692 / 55 |
| `27B-Hybrid` | `fp16` | `no_cache_baseline` | 504.77 | 993.38 / 1480.18 | 23.96 / 25.02 | 85120 (0.0%) | 65033.6 | 968.0 | 387.765 | 20/20 (0.0) | 0 / 0 / 0 |
| `27B-Hybrid` | `fp16` | `hbm_only_radix_cache` | 504.77 | 743.04 / 1488.80 | 23.96 / 25.02 | 71680 (15.8%) | 62212.8 | 968.0 | 370.496 | 20/20 (0.0) | 210 / 0 / 0 |
| `27B-Hybrid` | `fp16` | `naive_disk_kv_every_turn` | 512.78 | 125.88 / 1130.74 | 23.96 / 25.02 | 14464 (83.0%) | 55107.1 | 968.0 | 295.397 | 20/20 (0.0) | 461 / 643 / 0 |
| `27B-Hybrid` | `fp16` | `hanzo_tiered_prefix_cache` | 516.78 | 125.40 / 344.29 | 23.96 / 25.02 | 6208 (92.7%) | 48529.7 | 968.0 | 282.982 | 20/20 (0.0) | 489 / 689 / 55 |

## 4. Zero-Leak Cross-Tenant Isolation & Integrity Audit (`200` Adversarial Probes)

| Attack Surface | Probes | Rejected | Leaks |
|---|---:|---:|---:|
| Cross-tenant private prefix read (`dev-bob` -> `dev-alice`) | 40 | 40 | 0 |
| Cross-org shared prefix read (`other-org` -> `hanzo-eng`) | 40 | 40 | 0 |
| Private-to-shared scope blending injection | 40 | 40 | 0 |
| 5-field compatibility spoofing (model/tok/quant/RoPE/runtime) | 40 | 40 | 0 |
| Snapshot header/KV/GDN bit-flip & version mismatch | 40 | 40 | 0 |
| **Total** | **200** | **200** | **0** |

## 5. Honest Negative Regimes & Crossover Boundaries

Per HIP-1346, we explicitly evaluate regimes where persistent prefix caching regresses end-to-end performance rather than winning:

1. **Unique Zero-Reuse Single-Turn Prompts (`0%` prefix overlap, `9B-Hybrid`)**:
   - `no_cache_baseline` wall time: `20545.63 ms` (`0.0%` tokens saved)
   - `hanzo_tiered_prefix_cache` wall time: `20809.79 ms` (`0.0%` tokens saved, host control-plane overhead `319.45 ms`)
   - **Measured regression**: **`+1.29%`** wall-time overhead from SHA-256 keying, radix indexing, and disk serialization with zero hits. Mitigation: bypass eager persistence for stateless single-turn endpoints (`private_persist_min_hits >= 2`).

2. **Early-Prefix System/Tool Mutation at Block 0 (`9B-Hybrid`)**:
   - Mutating a single dynamic token (e.g., timestamp/nonce) at position 3 inside Block 0 invalidates the entire downstream `TokenPrefixChain`.
   - `no_cache_baseline` wall time: `25381.50 ms`
   - `hanzo_tiered_prefix_cache` wall time: `25970.68 ms` (`0.0%` tokens saved, `0` warm hits)
   - **Measured regression**: **`+2.32%`** wall-time overhead vs baseline (and losing ~92.7% token savings vs stable system prompts). Mitigation: hoist dynamic timestamps/nonces to the end of the prompt tail, never inside Block 0.

3. **Slow Cold Object-Store Link / Over-Aggressive Tier-2 Offload on `27B-Hybrid`**:
   - Restoring `32` blocks (`2,048` tokens) of `27B-Hybrid` FP16 KV (`256 MiB`) + unquantized FP32 GDN checkpoint (`192 MiB`) transfers `448 MiB` (`469.8 MB`).
   - **Derived analytical crossover bandwidth**: **`1198.8 MB/s`** (`~9.6 Gbps`). Below this threshold, cold Tier-2 fetches take longer than local prefill recompute.
   - At a constrained **`100 MB/s`** (`0.8 Gbps`) cold link:
     - Local prefill recompute TTFT: **`444.08 ms`**
     - Unguarded Tier-2 fetch TTFT: **`4722.36 ms`** (**`10.63x` TTFT regression!**)
     - Bandwidth-guarded Tier-2 TTFT (`enable_tier2_bandwidth_guard = true`): **`444.08 ms`** (`32` slow Tier-2 fetches cleanly bypassed -> regression eliminated).
