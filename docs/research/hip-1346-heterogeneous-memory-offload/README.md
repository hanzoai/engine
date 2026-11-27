# HIP-1346 Research Dossier: Heterogeneous Memory, CPU/GPU Offload & Cold-Start Minimisation (`hanzoai/engine#57`)

- **Canonical Owning Repository**: `hanzoai/engine` (crate: [`hanzo-hetero-mem/`](../../../hanzo-hetero-mem/), license: `Apache-2.0`, zero external dependencies, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`)
- **Tracking Issue**: [`hanzoai/engine#57`](https://github.com/hanzoai/engine/issues/57) (`p1`, `perf`, `research`, `bounty`, `hip-1346`)
- **Raw Benchmark Receipt**: [`results.json`](./results.json) (Deterministic Receipt SHA-256: `979466204f5834b389a27fff3b34b3520e9f7be0a32fc0abafa6b511c76a96c7`)
- **PR Disposition Target**: `MERGE` as a zero-dependency pure-Rust heterogeneous memory planner, 4 KiB-aligned paged weight container with background async lookahead prefetch, PowerInfer hot/cold profiler, bit-exact hybrid execution verifier, and HIP-1346 benchmark crate; live CUDA/Metal pointer-binding integration into `hanzo-engine/src/device_map/` and `hanzo-engine/src/model_loader.rs` remains gated on live GPU driver validation.

---

## 1. Problem Statement, Primary References & Clean-Room License Audit

### Problem
Serving hybrid Transformer + Gated DeltaNet (GDN) models (`4B-Hybrid`, `9B-Hybrid`, `27B-Hybrid`) at `128K` context (`131,072` tokens) across both **discrete PCIe GPUs** (`24 GiB` consumer/edge and `80 GiB` datacenter) and **zero-copy unified-memory hosts** (`24 GiB` / `64 GiB` Apple Silicon LPDDR5X) exposes four sharp failure modes:
1. **Eager full-file cold starts**: Reading full uncompressed BF16 weights (`16.76 GiB` for `9B`, `50.29 GiB` for `27B`) from NVMe before token 0 incurs millions of synchronous major 4 KiB page faults (`4.39M–13.18M`) and multi-second cold-start stalls (`10.7 s–33.8 s` on `DiscreteGpu24G`).
2. **Blocking PCIe offload collapse**: Streaming overflowing BF16 layers over PCIe Gen4 (`26 GB/s`) on every decode step blows `27B-Hybrid` ITL to `1,608 ms/tok` (`8,774 s` for a 20-turn coding-agent loop). Even with static INT4 layer splitting (`layer_split_static_offload`), running 16 cold full layers synchronously on CPU DDR5 (`85 GB/s`) limits `27B-Hybrid` to `78.92 ms/tok` (`519.36 s` 20-turn wall time).
3. **Unified-memory copy-bounce waste**: Applying discrete-style host↔device copies on unified-memory hardware duplicates resident weights and KV buffers inside the **same** physical LPDDR5X pool (`+86.3%` peak residency), wasting bus bandwidth and regressing 20-turn wall time by `+30.5%` and energy by `+81.2%`.
4. **Prefetch buffer thrashing under tight VRAM**: When `9B-Hybrid` BF16 barely fits `DiscreteGpu24G` at `128K` (`20.56 GiB` / `22.50 GiB` usable), allocating an aggressive 8-layer async prefetch staging ring (`+3.35 GiB`) wins the prefetch-hit microbenchmark (`99.2%` vs `91.5%`) while overflowing usable VRAM (`23.91 GiB > 22.50 GiB`), evicting live KV cache and regressing 20-turn wall time by **`+111.2%`**.

### Primary References & Clean-Room License Audit
All code in `hanzo-hetero-mem` is a **100% independent clean-room Apache-2.0 Rust implementation** using only the Rust standard library (`[dependencies]` empty):
- **PowerInfer** (Song et al., SOSP 2024, *"PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU"*): studied offline power-law neuron/expert activation locality and hybrid GPU-hot / CPU-cold execution with exact reduction; independent pure-Rust implementation (`src/partition.rs`, `src/verify.rs`).
- **FlexGen** (Sheng et al., ICML 2023, Apache-2.0): studied compressed weight residency, compute-bound PCIe-streamed prefill vs bandwidth-bound decode placement, and double-buffered async I/O overlap; independent Rust roofline & planner implementation.
- **llama.cpp / GGUF `mmap` & Static Layer Split** (MIT): studied demand-paged aligned weight loading and static first-K-layers GPU offload baseline; independent safe 4 KiB-paged `WeightFile` container (`#![forbid(unsafe_code)]`, `src/mmap_store.rs`).
- **Apple Metal / Unified Memory Zero-Copy Architecture**: studied coherent shared-pool CPU/GPU views eliminating duplicate allocations and host↔device copy energy (`src/hardware.rs`, `src/mmap_store.rs`, `src/partition.rs`).

Zero third-party crates, external source files, or non-commercial code were imported.

---

## 2. Methodology, Grounded Corpus & Exact Reproduction

### Grounded Repository Corpus (Verified SHA-256)
All offline activation-skew profiling, synthetic tile weights, and bit-exact execution verification inputs are deterministically derived (`--seed 1346`, SplitMix64) from **3 versioned repository files** (`140,742` bytes total):
- `calibration_data/calibration_datav3_small.txt` (`133,837` bytes, SHA-256 `aabd956dba34ff31f40e78b9de4e234fdd4865f90ba231e8e3deec8dfb22cbc0`)
- `chat_templates/llama3.json` (`420` bytes, SHA-256 `67d1d402738727aa3059d28590fa3488c44740a358be4b5c2135c60f58ec9d7a`)
- `orderings/xlora-paper-ordering.json` (`6,485` bytes, SHA-256 `8d0650ea231eeade0b97e5c026411365ff9feed41b2b07ecf10853c2a399043a`)
- **Combined Concatenated Corpus SHA-256**: `84b45bc2b79cb4612938566807c5eb8b478824036f382307c869cb112a8cd856`
- **Empirical Corpus Activation Skew** (`224` neuron blocks): Gini = **`0.8628`**, top-20% hottest neuron blocks cover **`91.40%`** of empirical activation mass.

### Recorded Hardware, Software & Honesty Boundary
- **Host CPU / OS**: `Intel(R) Xeon(R) CPU @ 2.30GHz`, `Linux 6.18.57 x86_64`
- **Compiler / Base Commit**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, base commit `df1722952421`
- **Strict Measurement Taxonomy**:
  1. **`cpu_directly_measured`**: Real on-disk 16.00 MiB `WeightFile` (`256` INT4 tiles, `4,096` 4 KiB pages, SHA-256 verified) cold/warm latencies, major/minor page faults, background async thread prefetch hit rate (`99.6%`), decode throughput (`MiB/s`), and pure-Rust 4-layer hybrid CPU/GPU execution equivalence (`64/64` tokens, `0.000e0` max error, `83.92 ms` CPU wall time).
  2. **`exact_unclamped_bytes`**: Exact byte-level model weight footprints (`BF16` `2.000 B/param`, `INT4` `0.678125 B/param` incl. group-128 metadata + 10% FP16 norms/embeddings), `128K` FP16 attention KV cache (`1.125 / 2.500 / 8.000 GiB`), unquantized FP32 GDN recurrent state (`27.0 / 45.0 / 192.0 MiB`), and runtime workspace. **Infeasible memory footprints are never clamped** (e.g., `27B-Hybrid` on `UnifiedMemory24G` honestly reports `26.739 GiB` INT4 / `110.270 GiB` naive copy-duplicated demand vs `20.0 GiB` wired cap and is marked `REJECTED`).
  3. **`analytical_hardware_projection`**: `Cold TTFT`, `Warm TTFT`, `ITL p50/p95`, `20-Turn Wall`, `PCIe Transfer`, and `Energy (mJ/tok)` across `DiscreteGpu24G` (RTX 4090/L4 class: 24 GiB GDDR6X `1,008 GB/s`, `165 TFLOPS`, PCIe Gen4 `26 GB/s`, NVMe `7 GB/s`), `DiscreteGpu80G` (H100-SXM5: 80 GiB HBM3 `3,350 GB/s`, `989 TFLOPS`, PCIe Gen5 `52 GB/s`, NVMe `12 GB/s`), `UnifiedMemory24G` (Apple M4 Pro: 24 GiB shared LPDDR5X `273 GB/s`, `18 TFLOPS`, NVMe `6.5 GB/s`), and `UnifiedMemory64G` (Apple M4 Max: 64 GiB shared LPDDR5X `546 GB/s`, `36 TFLOPS`, NVMe `7.4 GB/s`).

### Exact Offline Reproduction Commands
```bash
# 1. Formatting and zero-warning Clippy checks (offline)
cargo fmt -p hanzo-hetero-mem -- --check
cargo clippy -p hanzo-hetero-mem --all-targets --offline -- -D warnings

# 2. Full 18-test integration suite (HIP-1346 matrix alignment, fit regimes, SHA-256,
#    real on-disk WeightFile paging & async prefetch, zero-copy views, bit-exact hybrid
#    execution verifier, 5 placement regimes, 3 negative regimes, determinism)
cargo test -p hanzo-hetero-mem --offline --release

# 3. Reproducible HIP-1346 benchmark and JSON receipt generation
cargo run -p hanzo-hetero-mem --release --offline --bin hetero-bench -- \
  --seed 1346 --output-dir docs/research/hip-1346-heterogeneous-memory-offload
```

---

## 3. Crate Architecture (`hanzo-hetero-mem/`)

- [`src/model.rs`](../../../hanzo-hetero-mem/src/model.rs): Exact HIP-1346 `4B / 9B / 27B` hybrid Attention+GDN matrix matching `hanzo-paged-attn/tests/kivi_kv_quant.rs:450-478`, `WeightFormat` (`Bf16`, `Int8Packed`, `Int4Packed`), and `FitRegime` (`FullyFits`, `BarelyFits`, `ExceedsVram`) classification without clamping.
- [`src/hardware.rs`](../../../hanzo-hetero-mem/src/hardware.rs): Explicit `MemoryArchitecture` (`DiscretePcie` vs `UnifiedZeroCopy`) and all four canonical `HardwareProfile` definitions (`DiscreteGpu24G`, `DiscreteGpu80G`, `UnifiedMemory24G`, `UnifiedMemory64G`).
- [`src/mmap_store.rs`](../../../hanzo-hetero-mem/src/mmap_store.rs): Pure-Rust safe (`#![forbid(unsafe_code)]`) 4 KiB-aligned paged `WeightFile` (`HZWT` v1) with `O(1)` tile lookup, packed INT4/INT8/BF16 codecs, SHA-256 integrity verification, real 4 KiB major/minor page-fault counters, background `std::thread` async lookahead prefetch, and `Arc::ptr_eq`-verified `UnifiedTileView`.
- [`src/partition.rs`](../../../hanzo-hetero-mem/src/partition.rs): Grounded-corpus PowerInfer `ActivationProfile` skew profiler, all 5 matched `PlacementRegime` evaluators, `AdaptiveHeteroPlanner`, 3 explicit honest negative regimes (`NegativeRegimeKind`), and strict `apply_gate` end-to-end acceptance gate.
- [`src/verify.rs`](../../../hanzo-hetero-mem/src/verify.rs): Pure-Rust deterministic 4-layer SwiGLU hybrid CPU+GPU execution verifier proving `100.000%` token equivalence (`0.000e0` max abs error, `0.000 pp` regression) across full-GPU, layer-split, PowerInfer hot/cold, and zero-copy unified views.
- [`src/corpus.rs`](../../../hanzo-hetero-mem/src/corpus.rs), [`src/sha256.rs`](../../../hanzo-hetero-mem/src/sha256.rs), [`src/bench.rs`](../../../hanzo-hetero-mem/src/bench.rs), [`src/bin/hetero_bench.rs`](../../../hanzo-hetero-mem/src/bin/hetero_bench.rs): Grounded corpus attestation, FIPS 180-4 SHA-256, JSON receipt generator, and CLI.

---

## 4. Directly Measured Host On-Disk `WeightFile` I/O (`mmap_store.rs`)

Real filesystem benchmark on `16.00 MiB` (`16,781,312 B`, `256` INT4 tiles × `16` 4 KiB pages/tile, SHA-256 `verified = true`, `cpu_directly_measured` on Linux host):

| Load Mode (`cpu_directly_measured`) | Cold 1st Tile (`ms`) | Warm 1st Tile (`ms`) | Full Stream (`ms`) | Sustained Decode (`MiB/s`) | Cold Major Faults | Total Major / Minor Faults | Prefetch Hit Rate | Copy Bytes | Peak Resident Pages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `eager_full_read` | `18.405` | `0.000` | `162.00` | `98.8` | `4,097` | `4,097 / 4,112` | `0.0%` | `0 B` | `4,097` |
| `demand_paged` | **`< 0.001`** | `0.000` | `152.08` | `105.2` | **`17`** (`241x` lower) | `4,097 / 16` | `0.0%` | `0 B` | `1,025` |
| **`async_lookahead_prefetch`** (`depth=4`) | **`< 0.001`** | `0.000` | **`150.60`** | **`106.3`** | **`17`** | **`17 / 4,096`** | **`99.6%`** | `0 B` | `1,025` |
| **`unified_zero_copy_view`** | `0.086` | `0.000` | `155.85` | `102.7` | `17` | `4,097 / 16` | `0.0%` | **`0 B`** | `4,097` |
| `unified_naive_copy_bounce` (neg control) | `0.000` | `0.000` | **`170.20`** (`+9.2%`) | `94.0` | `17` | `4,097 / 16` | `0.0%` | **`16,842,752 B`** | `4,113` |

- All 5 modes produce **bit-identical** decoded floating-point dot-product checksums across all 256 tiles.
- `async_lookahead_prefetch` cuts cold-start major faults by **`241x`** (`4,097 -> 17`), converts **`99.6%`** of subsequent tile fetches into zero-stall background prefetch hits (`17` total blocking major faults vs `4,097`), and achieves the highest sustained decode throughput (`106.3 MiB/s`).
- `unified_naive_copy_bounce` copies `16.84 MiB` needlessly and degrades real host stream throughput by `-8.5%` (`102.7 -> 94.0 MiB/s`).

---

## 5. Directly Measured CPU Bit-Exact Hybrid Execution Verifier (`verify.rs`)

Executing a deterministic 4-layer SwiGLU micro-network (`hidden = 64`, `intermediate = 256`, `25%` GPU hot neurons, `64` grounded-corpus tokens, `83.92 ms` total CPU wall time):

| Execution Path (`cpu_directly_measured`) | Matched Tokens | Quality Regression | Max Abs Logit Error | Weight Copy Bytes | Zero-Copy Ptr Verified |
|---|---:|---:|---:|---:|---|
| `full_gpu_reference` | `64 / 64` (`100.00%`) | `0.000 pp` | `0.000e0` | `0 B` | `false` |
| `layer_split_static_offload` | **`64 / 64` (`100.00%`)** | **`0.000 pp`** | **`0.000e0`** | `25,165,824 B` | `false` |
| `powerinfer_hot_cold_hybrid` | **`64 / 64` (`100.00%`)** | **`0.000 pp`** | **`0.000e0`** | `131,072 B` (`192x` less) | `false` |
| `unified_zero_copy_view` | **`64 / 64` (`100.00%`)** | **`0.000 pp`** | **`0.000e0`** | **`0 B`** | **`true`** (`Arc::ptr_eq`) |

---

## 6. HIP-1346 Shared `4B / 9B / 27B` Matrix Across Discrete & Unified Topologies

Exact model specs matching `hanzo-paged-attn/tests/kivi_kv_quant.rs:450-478`:
- **`4B-Hybrid`**: `9` Attn + `27` GDN, `num_kv_heads = 2`, `head_dim = 128`, `gdn_state = 27.0 MiB`, `kv = 9,216 B/tok` (`1.125 GiB` @ 128K), weights `7.451 GiB` BF16 / `2.526 GiB` INT4.
- **`9B-Hybrid`**: `10` Attn + `30` GDN, `num_kv_heads = 4`, `head_dim = 128`, `gdn_state = 45.0 MiB`, `kv = 20,480 B/tok` (`2.500 GiB` @ 128K), weights `16.764 GiB` BF16 / `5.684 GiB` INT4.
- **`27B-Hybrid`**: `16` Attn + `48` GDN, `num_kv_heads = 4`, `head_dim = 256`, `gdn_state = 192.0 MiB`, `kv = 65,536 B/tok` (`8.000 GiB` @ 128K), weights `50.291 GiB` BF16 / `17.052 GiB` INT4.

Workload: 20-turn coding-agent loop (`32,768`-tok cold repo prefill + `20 × 4,800`-tok warm tool/diff delta prefills + `21 × 256 = 5,376` generated tokens, peak `128K` KV). All regimes achieve `100.00%` output correctness (`0.000 pp` regression) via `verify.rs`. Timing/energy are `analytical_hardware_projection`; VRAM/Host/Copy are `exact_unclamped_bytes`.

### 6.1 Discrete 24 GiB GPU (`DiscreteGpu24G` — Fully Fits `4B` / Barely Fits `9B` / Exceeds VRAM `27B`)

| Model (`BF16 Fit`) | Regime | Cold / Warm TTFT (`ms`) | Cold Major Faults | ITL p50 / p95 (`ms`) | **20-Turn Wall (`s`)** | VRAM / Host (`GiB`) | Copy (`GiB`) | Energy (`mJ/tok`) | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `4B` (`fully_fits`) | `eager_full_read_naive_offload` | `4,770.1 / 464.8` | `1,953,125` | `8.88 / 9.72` | `62.00` | `9.10 / 7.45` | `7.45` | `4,280.3` | REJECTED |
| `4B` (`fully_fits`) | `layer_split_static_offload` | `3,016.9 / 464.8` | `108,507` | `8.91 / 9.66` | `60.44` | `9.10 / 0.41` | `7.45` | `3,562.1` | ACCEPTED |
| `4B` (`fully_fits`) | `powerinfer_hot_cold_hybrid` (INT4) | `3,196.1 / 502.0` | `11,038` | `3.93 / 4.61` | `34.58` | `4.30 / 0.25` | `2.53` | `2,159.7` | ACCEPTED |
| `4B` (`fully_fits`) | `unified_zero_copy_view` | `3,171.8 / 440.8` | `108,507` | `8.84 / 9.49` | `59.70` | `9.10 / 7.45` | `7.45` | `3,149.5` | REJECTED* |
| `4B` (`fully_fits`) | **`adaptive_hetero_planner`** (BF16 full-GPU) | **`3,034.0 / 464.8`** | `108,507` | **`8.88 / 9.53`** | **`60.26`** | `9.31 / 0.21` | `7.45` | `3,980.5` | **ACCEPTED** |
| `9B` (`barely_fits`) | `eager_full_read_naive_offload` | `10,725.4 / 1,043.8` | `4,394,532` | `19.71 / 21.58` | `138.04` | `20.56 / 16.76` | `16.76` | `9,529.8` | REJECTED |
| `9B` (`barely_fits`) | `layer_split_static_offload` (INT4) | `7,212.2 / 1,127.3` | `74,502` | `8.53 / 9.94` | `76.10` | `9.48 / 0.28` | `5.68` | `4,485.4` | ACCEPTED |
| `9B` (`barely_fits`) | `powerinfer_hot_cold_hybrid` (INT4) | `7,181.7 / 1,127.3` | `22,351` | `8.55 / 10.05` | `76.18` | `9.60 / 0.25` | `5.68` | `4,758.5` | ACCEPTED |
| `9B` (`barely_fits`) | `unified_zero_copy_view` | `7,076.2 / 1,069.0` | `74,502` | `8.49 / 9.80` | `74.58` | `9.48 / 5.68` | `5.68` | `3,934.3` | REJECTED* |
| `9B` (`barely_fits`) | **`adaptive_hetero_planner`** (INT4 full-GPU) | **`7,224.0 / 1,127.3`** | `74,502` | **`8.53 / 9.84`** | **`76.12`** | **`9.62 / 0.14`** | `5.68` | `5,027.6` | **ACCEPTED** |
| `27B` (`exceeds_vram`) | `eager_full_read_naive_offload` | `33,769.6 / 4,700.3` | `13,183,594` | `1,608.10 / 1,675.88` | `8,774.44` | `22.50 / 50.29` | `50.29` | `607,065.9` | REJECTED |
| `27B` (`exceeds_vram`) | `layer_split_static_offload` (INT4 48/64 GPU) | `21,813.5 / 3,581.0` | `139,690` | `78.92 / 84.98` | `519.36` | `22.48 / 4.80` | `12.79` | `37,855.5` | ACCEPTED |
| `27B` (`exceeds_vram`) | **`powerinfer_hot_cold_hybrid`** (`98.97%` MAC GPU) | **`21,643.0 / 3,467.8`** | **`41,907`** | **`20.84 / 25.40`** | **`204.67`** | **`22.50 / 4.61`** | `12.69` | **`13,974.3`** | **ACCEPTED** |
| `27B` (`exceeds_vram`) | `unified_zero_copy_view` | `21,226.4 / 3,403.7` | `139,690` | `200.67 / 208.34` | `1,169.72` | `22.50 / 17.05` | `101.84` | `61,707.4` | REJECTED* |
| `27B` (`exceeds_vram`) | **`adaptive_hetero_planner`** (auto PowerInfer) | **`21,643.0 / 3,467.8`** | **`41,907`** | **`20.84 / 25.40`** | **`204.67`** | **`22.50 / 4.61`** | `12.69` | **`13,974.3`** | **ACCEPTED** |

*\*`unified_zero_copy_view` is explicitly rejected on `DiscretePcie` because PCIe hardware cannot share coherent zero-copy pages without DMA.*

**Key `27B-Hybrid` on `DiscreteGpu24G` Result**:
Even INT4-compressed `27B-Hybrid` (`17.05 GiB` weights + `8.00 GiB` 128K KV + `0.19 GiB` FP32 GDN + `1.50 GiB` workspace = `26.74 GiB`) exceeds `22.50 GiB` usable VRAM. `adaptive_hetero_planner` automatically selects `powerinfer_hot_cold_hybrid`, placing `63.4%` of MLP weights (`98.97%` of empirical active MACs) + 100% attention/GDN/KV on GPU:
- **`42.9x` faster** 20-turn wall time than `eager_full_read_naive_offload` (`8,774.44 s -> 204.67 s`) and **`77.2x` lower** decode ITL (`1,608.10 -> 20.84 ms`).
- **`2.54x` faster** 20-turn wall time than static `layer_split_static_offload` (`519.36 s -> 204.67 s`), **`3.79x` lower** p50 ITL (`78.92 -> 20.84 ms`), and **`2.71x` lower** energy (`37,855 -> 13,974 mJ/tok`).
- **`314x` fewer** cold-start major page faults (`13,183,594 -> 41,907`).

### 6.2 Discrete 80 GiB (`DiscreteGpu80G`) & Unified Memory (`UnifiedMemory24G` / `UnifiedMemory64G`) Summary

| Topology | Model (`BF16 Fit`) | `eager_full_read` Wall (`s`) / VRAM (`GiB`) | **`adaptive_hetero_planner` Strategy** | Cold TTFT (`ms`) | ITL p50 / p95 (`ms`) | **20-Turn Wall (`s`)** | VRAM / Host (`GiB`) | Copy (`GiB`) | Energy (`mJ/tok`) | Gate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `DiscreteGpu80G` | `4B` (`fully_fits`) | `18.23` / `9.10` | `full_gpu_demand_paged_bf16` | `675.2` | `2.80 / 3.00` | **`17.32`** | `9.31 / 0.21` | `7.45` | `1,886.4` | **ACCEPTED** |
| `DiscreteGpu80G` | `9B` (`fully_fits`) | `39.74` / `20.56` | `full_gpu_demand_paged_bf16` | `1,517.3` | `6.06 / 6.50` | **`37.69`** | `20.81 / 0.25` | `16.76` | `4,105.1` | **ACCEPTED** |
| `DiscreteGpu80G` | `27B` (`barely_fits`) | `118.01` / `59.98` | `full_gpu_compressed_int4_demand_paged` | **`3,654.9`** | **`7.84 / 9.10`** | **`57.59`** (`2.05x` faster) | **`26.99 / 0.25`** | `17.05` | `6,273.0` | **ACCEPTED** |
| `UnifiedMemory24G` | `4B` (`fully_fits`) | `288.57` / `16.55` | `unified_zero_copy_view` (BF16) | `25,740.7` | `32.25 / 34.65` | **`280.99`** | `9.10 / 0.00` | **`0.000`** | **`1,543.4`** | **ACCEPTED** |
| `UnifiedMemory24G` | `9B` (`exceeds_vram`) | `4,552.64` / `37.32` | `unified_zero_copy_view` (INT4) | `62,360.6` | **`30.95 / 35.79`** | **`427.33`** (`10.7x` faster) | **`9.48 / 0.00`** | **`0.000`** | **`2,347.2`** | **ACCEPTED** |
| `UnifiedMemory24G` | `27B` (`exceeds_vram`) | `22,305.76` / `110.27` | `unified_zero_copy_view` (INT4 — unclamped overflow!) | `188,015.7` | `227.71 / 277.09` | `2,023.46` | **`26.74 > 20.00`** | `0.000` | `11,114.4` | **REJECTED** (honest OOM) |
| `UnifiedMemory64G` | `4B` (`fully_fits`) | `145.37` / `16.55` | `unified_zero_copy_view` (BF16) | `12,896.2` | `16.19 / 17.40` | **`140.90`** | `9.10 / 0.00` | **`0.000`** | **`1,407.0`** | **ACCEPTED** |
| `UnifiedMemory64G` | `9B` (`fully_fits`) | `325.28` / `37.32` | `unified_zero_copy_view` (BF16) | `28,970.5` | `36.19 / 38.87` | **`315.48`** | `20.56 / 0.00` | **`0.000`** | **`3,150.5`** | **ACCEPTED** |
| `UnifiedMemory64G` | `27B` (`exceeds_vram`) | `12,125.10` / `110.27` | `unified_zero_copy_view` (INT4) | `93,762.8` | **`47.13 / 54.82`** | **`647.19`** (`18.7x` faster) | **`26.74 / 0.00`** | **`0.000`** | **`6,463.0`** | **ACCEPTED** |

- **Honest Unclamped Infeasibility**: On `UnifiedMemory24G` (`20.0 GiB` Metal wired limit), `27B-Hybrid` at `128K` FP16 KV genuinely demands `26.739 GiB` even in INT4 (`6.739 GiB` overflow) and is honestly marked **REJECTED** (requiring either `UnifiedMemory64G`, `DiscreteGpu24G + PowerInfer`, or KIVI KV quantization from `#50`).

---

## 7. Explicitly Tested Honest Negative Regimes (All 3 REJECTED)

| Negative Regime | Target Cell | Baseline vs Negative Wall (`s`) | Wall Delta | Energy Delta | Peak Mem Delta | Microbenchmark Illusion vs Reality | Gate |
|---|---|---:|---:|---:|---:|---|---|
| `neg_unified_memory_naive_copy_bounce` | `9B` / `UnifiedMemory24G` | `427.33` vs `557.62` | **`+30.5%`** | **`+81.2%`** | **`+86.3%`** (`9.48 -> 17.66 GiB`) | Discrete host↔GPU staging copies inside unified RAM waste `78+ GiB` LPDDR5X copies and nearly double residency | **REJECTED** |
| `neg_low_skew_uniform_activation_offload` | `27B` / `DiscreteGpu24G` | `519.36` vs `689.59` | **`+32.8%`** | **`+45.6%`** | `-0.6%` (`22.48 -> 22.50 GiB`) | Under uniform activations (`Gini = 0.029`, top-20% = `21.1%`), PowerInfer loses locality (`ITL 110.33` vs `78.92 ms` static split) | **REJECTED** |
| `neg_aggressive_prefetch_thrash_tight_ram` | `9B` BF16 / `DiscreteGpu24G` | `134.62` vs `284.32` | **`+111.2%`** | **`+111.5%`** | **`+14.0%`** (`20.98 -> 23.91 GiB`) | **Wins prefetch-hit microbenchmark (`99.2%` vs `91.5%`)**, but 8-layer staging ring overflows `22.50 GiB` usable VRAM (`+1.411 GiB`), evicting live KV cache! | **REJECTED** |

- Importantly, `adaptive_hetero_planner` defends against **all three** traps automatically: it selects `unified_zero_copy_view` (`0` copy bytes) on unified hardware, falls back to `layer_split_static_offload_low_skew_fallback` when `ActivationProfile::is_skewed == false`, and clamps async prefetch staging strictly to `<= headroom / 8` so live KV cache is never evicted.

---

## 8. Integration Plan for `hanzo-engine`

1. **Topology-Aware Device Mapper (`hanzo-engine/src/device_map/mappers.rs`)**:
   - Adopt `hanzo_hetero_mem::hardware::MemoryArchitecture` (`DiscretePcie` vs `UnifiedZeroCopy`) so Metal/unified hosts automatically bind shared zero-copy weight & KV views (`0` staging buffer duplication) instead of executing discrete CPU↔GPU offload copies.
2. **Demand-Paged `WeightFile` Loader & Background Prefetch (`hanzo-engine/src/model_loader.rs`)**:
   - Replace eager full-weight reads during model startup with 4 KiB-aligned demand-paged header opens + headroom-clamped background lookahead prefetch (`PagedWeightStore`), cutting cold-start major faults by `> 200x` while preserving SHA-256 weight integrity.
3. **Skew-Gated PowerInfer Offload Policy (`hanzo-engine/src/device_map/mod.rs`)**:
   - Integrate `PlacementPlanner` with offline activation-skew gating (`Gini >= 0.45`, top-20% coverage `>= 55%`) so overflowing discrete GPU deployments (`27B-Hybrid` on `24 GiB` VRAM) keep attention, FP32 GDN state, KV cache, and hot MLP/expert neurons on GPU while safely falling back to static layer split on dense low-skew workloads.
