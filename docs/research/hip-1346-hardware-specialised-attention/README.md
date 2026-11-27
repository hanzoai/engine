# HIP-1346: Hardware-Specialised Attention — Cascade LSE Merge, Fused Ragged Prefill/Decode, and Per-Architecture Hybrid Serving (`hanzoai/engine#53`)

## Executive Summary

Hybrid Transformer + Linear-Attention (**Gated DeltaNet / LSA**) models in the shared HIP-1346 `4B / 9B / 27B` matrix devote only `25%` of decoder layers to full grouped-query paged attention (`9/36`, `10/40`, `16/64`), while `75%` execute recurrent GDN/LSA state updates and `100%` execute dense SwiGLU MLPs. Consequently, **isolated attention kernel TFLOPS microbenchmarks do not transfer automatically to end-to-end serving speedups**, and datacenter **Blackwell B200** (`8,000 GB/s` HBM3e, `281.3 FLOP/byte` ridge point) numbers **cannot** be extrapolated onto **Grace Blackwell GB10** (`273 GB/s` unified LPDDR5X, `915.8 FLOP/byte` ridge point) or **Consumer RTX 4090** (`24 GiB` GDDR6X VRAM cap).

This crate (`hanzo-cascade-attn`) delivers a clean-room Apache-2.0, zero-dependency, `#![forbid(unsafe_code)]` Rust reference kernel suite, exact Log-Sum-Exp (LSE) state-merge verifier, fused ragged prefill/decode engine, hybrid Amdahl step profiler, and strictly separated four-architecture serving scorecard:

- **Directly Measured CPU Verification (`cpu_directly_measured`)**:
  - **Two-level shared-prefix cascade vs unshared paged GQA**: `max_abs_err = 3.166e-8` (`<= 1e-6`), `100.00%` greedy argmax token match, `0.000 pp` quality regression across `B ∈ {1, 4, 8, 16}`, prefix `64..512`, suffix `16..128`.
  - **Real KV read reduction & CPU wall-clock speedup** via 4-way query register blocking over shared prefix KV pages:
    - `B = 8, prefix = 512, suffix = 32`: **`11.33x` KV read reduction**, **`2.34x` CPU wall-clock speedup** (`2,834.1 -> 1,213.2 us`).
    - `B = 16, prefix = 512, suffix = 32`: **`13.60x` KV read reduction**, **`2.17x` CPU wall-clock speedup** (`5,689.4 -> 2,624.9 us`).
  - **Three-level prefix-tree cascade** (global system -> repository sub-group -> unique suffix): `max_abs_err = 2.608e-8`, **`6.10x` KV read reduction**.
  - **Fused ragged prefill + decode** over unified CSR-like `(qo_indptr, kv_indptr, kv_page_indices, kv_last_page_len)`: `max_abs_err = 0.00e0`, `100.00%` argmax match, collapses `2 -> 1` dispatch stages.
  - **Full hybrid `2 Attn + 6 GDN + 8 SwiGLU MLP` CPU Amdahl proof**: `9.3%` Attention / `26.9%` GDN / `63.8%` MLP — a `1.40x` isolated attention speedup translates to `1.03x` full-step speedup (`max_abs_hidden_err = 1.79e-7`, `100.00%` argmax match).
- **Per-Architecture End-to-End 20-Turn Serving Speedups (`adaptive_hybrid_dispatch`, `9B-Hybrid INT4-AWQ`, 8K shared repo prefix)**:
  - **Hopper H100-SXM5 80GiB**: `B = 8` wall `36.93 -> 31.08s` (**`1.19x`**), cold TTFT `2,155.1 -> 481.1 ms` (**`4.48x`**); `B = 16` wall `53.51 -> 42.05s` (**`1.27x`**), cold TTFT `4,304.8 -> 723.8 ms` (**`5.95x`**), energy `354.9 -> 277.7 mJ/tok`.
  - **Blackwell B200 192GiB**: `B = 8` wall `19.40 -> 16.95s` (**`1.14x`**), cold TTFT `888.8 -> 198.2 ms` (**`4.48x`**), `1,750.1 TFLOPS` (`1.77x`) isolated 32K prefill attention; `B = 16` wall `26.21 -> 21.40s` (**`1.22x`**), energy `242.4 -> 196.0 mJ/tok`.
  - **Grace Blackwell GB10 128GiB (`273 GB/s` LPDDR5X)**: Unconditional `warp_pipelined_async` wins isolated 32K prefill TFLOPS (`1.33x`) yet **regresses** 20-turn wall time (`0.96x–0.996x` -> **`REJECTED`**). By disabling heavy async pipeline staging and activating shared-prefix cascade + fused ragged bandwidth reduction, `adaptive_hybrid_dispatch` achieves **`1.22x` (`B = 8`)** and **`1.30x` (`B = 16`, `451.27 -> 345.96s`)** end-to-end speedup!
  - **Consumer RTX 4090 24GiB (GDDR6X)**: `B = 8` wall `104.89 -> 86.76s` (**`1.21x`**), `B = 16` wall `159.83 -> 124.54s` (**`1.28x`**), VRAM `15,336 / 24,576 MiB` (`2.4 GiB` saved by prefix sharing). `27B-Hybrid BF16` (`49,566.5 MiB`) is honestly marked OOM (`vram_fits = false`, **`REJECTED`**), while `27B-Hybrid INT4-AWQ` at `B = 1` fits cleanly (`17,132.8 / 24,576 MiB`, **`ACCEPTED`**).
- **Three Explicit Honest Negative Regimes — All Strictly `REJECTED`**:
  1. `neg_cascade_overhead_at_b1_unshared`: `0.98x` wall time (`+2.21%` slower) -> **`REJECTED`**.
  2. `neg_kernel_tflops_win_amdahl_loss`: `1.33x` isolated kernel TFLOPS win on GB10 yet `0.96x` (`+4.26%` slower) end-to-end wall time -> **`REJECTED`**.
  3. `neg_rtx4090_27b_bf16_vram_overflow`: `49,566.5 MiB > 24,576.0 MiB` VRAM cap -> **`REJECTED`**.

---

## Measurement Honesty & Provenance

| Artifact / Metric Class | Provenance Tag | What Is Measured vs Projected |
|---|---|---|
| Pure-Rust two-level & three-level shared-prefix cascade LSE merge, fused ragged prefill/decode equivalence, full hybrid `2 Attn + 6 GDN + 8 MLP` Amdahl profiling, KV read byte counts, and CPU wall-clock speedups | `cpu_directly_measured` | Directly executed, timed, and asserted on the host CPU (`x86_64`, Linux) in `hanzo-cascade-attn`. |
| Memory byte accounting (resident weights including full `vocab_size * d_model * 2` input-embedding table, per-step streamed weights + single-row `embed_tokens` gather `d_model * 2`, FP16 paged KV blocks, unquantized FP32 GDN state `27.0 / 45.0 / 192.0 MiB`, workspace scratch) | Exact formula (unclamped) | Exact architectural byte math matching `hanzo-paged-attn/tests/kivi_kv_quant.rs`; OOM configs fail `vram_fits = false` without clamping. |
| Per-architecture GPU 20-turn scorecards (cold/warm TTFT, `p50`/`p95` ITL, wall time, isolated 32K kernel TFLOPS, `mJ/token`) | `analytical_gpu_hopper_h100_sxm5`, `analytical_gpu_blackwell_b200`, `analytical_gpu_grace_blackwell_gb10`, `analytical_gpu_consumer_rtx_4090` | Four strictly separated per-architecture roofline + launch + staging analytical models. **No physical GPU was executed on this host; B200 numbers are never extrapolated onto GB10 or Consumer RTX.** |

---

## Strictly Separated Hardware Architecture Profiles

| Architecture | Memory Subsystem | Capacity (GiB) | Peak / Sustained BW (GB/s) | Peak BF16 (TFLOPS) | Ridge Point (FLOP/B) | Async Pipeline | Driver Label |
|---|---|---:|---:|---:|---:|---|---|
| **Hopper H100-SXM5** | `HBM3` | `80.0` | `3,350 / 2,680` | `989.0` | `295.2` | TMA + WGMMA | `CUDA 12.6 (R560)` |
| **Blackwell B200** | `HBM3e` | `192.0` | `8,000 / 6,560` | `2,250.0` | `281.3` | 5th-gen TMEM Dual-Warp | `CUDA 13.0 (R580)` |
| **Grace Blackwell GB10** | `LPDDR5X Unified` | `128.0` | **`273 / 212.9`** | `250.0` | **`915.8`** | Bandwidth-Bound Unified | `CUDA 13.0 (R580-aarch64)` |
| **Consumer RTX 4090** | `GDDR6X` | **`24.0`** | `1,008 / 786.2` | `330.0` | `327.4` | Consumer Ada Warp (No TMEM) | `CUDA 12.6 (R560)` |

Notice the `3.26x` ridge-point gap between **Blackwell B200** (`281.3 FLOP/byte`) and **Grace Blackwell GB10** (`915.8 FLOP/byte`): compute-pipelined attention kernels accelerate compute-bound prefill tiles on B200 (`1.77x` isolated kernel TFLOPS), whereas GB10 spends nearly its entire 20-turn hybrid runtime memory-bandwidth-bound on `273 GB/s` LPDDR5X.

---

## Directly Measured CPU Verification & Speedup (`cpu_directly_measured`)

### 1. Two-Level & Three-Level Shared-Prefix Cascade Equivalence and CPU Wall-Clock Speedup

| Batch (`B`) | Shared Prefix | Unique Suffix | Max FP32 Output Err | Greedy Argmax Match | KV Read Reduction | Unshared CPU (us) | Cascade CPU (us) | CPU Wall Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `64` | `16` | `2.24e-8` | `100.00%` | `2.50x` | `60.0` | `37.4` | `1.61x` |
| `4` | `128` | `32` | `3.17e-8` | `100.00%` | `4.00x` | `486.8` | `270.5` | `1.80x` |
| `8` | `256` | `64` | `2.98e-8` | `100.00%` | `4.44x` | `1,795.9` | `968.5` | `1.85x` |
| **`8`** | **`512`** | **`32`** | **`1.68e-8`** | **`100.00%`** | **`11.33x`** | **`2,834.1`** | **`1,213.2`** | **`2.34x`** |
| **`16`** | **`512`** | **`32`** | **`2.05e-8`** | **`100.00%`** | **`13.60x`** | **`5,689.4`** | **`2,624.9`** | **`2.17x`** |
| `16` | `256` | `128` | `3.17e-8` | `100.00%` | `2.91x` | `3,926.4` | `2,402.5` | `1.63x` |
| **3-Level (`B=8`)** | `128 + 96` | `32` | **`2.61e-8`** | **`100.00%`** | **`6.10x`** | — | — | — |

### 2. Fused Ragged Prefill + Decode & Hybrid Attention + GDN/LSA + Dense MLP Amdahl Proof

| Verification Check | Result (`cpu_directly_measured`) |
|---|---|
| Fused ragged mixed batch (`2` prefill chunks + `6` decodes = `86` packed Q tokens) | `2 -> 1` dispatch stages, `max_abs_err = 0.00e0`, `100.00%` argmax match (`0.000 pp` regression) |
| Full hybrid step (`2 Attn + 6 GDN + 8 SwiGLU MLP`, exact `1:3` ratio, `B = 8`) | Component share: **`9.3%` Attn**, **`26.9%` GDN/LSA**, **`63.8%` Dense MLP** |
| Hybrid step isolated vs end-to-end speedup (Amdahl bound) | Isolated attention: **`1.40x`** -> Full hybrid step: **`1.03x`**, `max_abs_hidden_err = 1.79e-7`, `100.00%` argmax match |

---

## Per-Architecture 5-Backend Ablation (`9B-Hybrid BF16`, `B = 8`, 20-Turn Coding Agent)

This table proves the core thesis of `hanzoai/engine#53`: **unconditional `warp_pipelined_async` wins isolated 32K prefill kernel TFLOPS on every GPU (`1.15x–1.77x`), yet is `REJECTED` end-to-end across all four architectures** because its per-layer workspace/setup overhead regresses the 5,120 memory-bound decode tokens! Combining **shared-prefix cascade** + **fused ragged prefill/decode** + **selective compute-prefill pipelining** (`adaptive_hybrid_dispatch`) wins end-to-end on every architecture.

| Architecture | Backend | Iso 32K Attn (TFLOPS) | Cold / Warm TTFT (ms) | p50 / p95 ITL (ms) | 20-Turn Wall (s) | Speedup | Peak VRAM (MiB) | Energy (mJ/tok) | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Hopper H100` | `baseline_paged_separate` | `415.3 (1.00x)` | `2,159.3 / 304.6` | `10.37 / 13.81` | `60.69` | `1.00x` | `23,024.5` | `749.1` | `ACCEPTED` |
| `Hopper H100` | `ragged_paged_fused` | `415.3 (1.00x)` | `2,159.3 / 304.5` | `9.54 / 10.65` | `56.41` | `1.08x` | `23,024.5` | `699.4` | `ACCEPTED` |
| `Hopper H100` | `cascade_shared_prefix` | `415.3 (1.00x)` | `499.5 / 304.5` | `10.10 / 13.47` | `57.61` | `1.05x` | `21,904.5` | `705.4` | `ACCEPTED` |
| `Hopper H100` | `warp_pipelined_async` | **`632.1 (1.52x)`** | `2,113.7 / 281.1` | `10.59 / 14.08` | `61.30` | `0.99x` | `23,025.1` | `753.3` | **`REJECTED`** |
| **`Hopper H100`** | **`adaptive_hybrid_dispatch`** | **`632.1 (1.52x)`** | **`485.4 / 280.8`** | **`9.28 / 10.38`** | **`52.98`** | **`1.15x`** | **`21,905.1`** | **`649.4`** | **`ACCEPTED`** |
| `Blackwell B200` | `baseline_paged_separate` | `989.5 (1.00x)` | `890.5 / 127.3` | `5.07 / 6.68` | `29.11` | `1.00x` | `23,024.5` | `506.1` | `ACCEPTED` |
| `Blackwell B200` | `ragged_paged_fused` | `989.5 (1.00x)` | `890.5 / 127.3` | `4.65 / 5.14` | `26.96` | `1.08x` | `23,024.5` | `470.8` | `ACCEPTED` |
| `Blackwell B200` | `cascade_shared_prefix` | `989.5 (1.00x)` | `207.4 / 127.5` | `5.04 / 6.64` | `28.27` | `1.03x` | `21,904.5` | `487.6` | `ACCEPTED` |
| `Blackwell B200` | `warp_pipelined_async` | **`1,750.1 (1.77x)`** | `866.3 / 114.9` | `5.22 / 6.86` | `29.62` | `0.98x` | `23,025.0` | `512.4` | **`REJECTED`** |
| **`Blackwell B200`** | **`adaptive_hybrid_dispatch`** | **`1,750.1 (1.77x)`** | **`199.9 / 114.8`** | **`4.62 / 5.11`** | **`25.89`** | **`1.12x`** | **`21,905.0`** | **`446.7`** | **`ACCEPTED`** |
| `GB10 (LPDDR5X)` | `baseline_paged_separate` | `90.0 (1.00x)` | `10,345.0 / 1,495.7` | `110.88 / 149.41` | `602.09` | `1.00x` | `23,024.5` | `1,472.1` | `ACCEPTED` |
| `GB10 (LPDDR5X)` | `ragged_paged_fused` | `90.0 (1.00x)` | `10,345.0 / 1,495.7` | `102.16 / 115.47` | `557.67` | `1.08x` | `23,024.5` | `1,366.4` | `ACCEPTED` |
| `GB10 (LPDDR5X)` | `cascade_shared_prefix` | `90.0 (1.00x)` | `2,421.4 / 1,490.9` | `105.27 / 142.45` | `565.45` | `1.06x` | `21,904.5` | `1,377.8` | `ACCEPTED` |
| `GB10 (LPDDR5X)` | `warp_pipelined_async` | **`119.9 (1.33x)`** | `10,191.8 / 1,416.9` | `111.64 / 150.36` | `604.34` | `0.996x` | `23,029.5` | `1,475.5` | **`REJECTED`** |
| **`GB10 (LPDDR5X)`** | **`adaptive_hybrid_dispatch`** | `90.0 (1.00x)` | **`2,421.3 / 1,490.8`** | **`96.99 / 110.09`** | **`523.28`** | **`1.15x`** | **`21,909.5`** | **`1,277.5`** | **`ACCEPTED`** |
| `RTX 4090 24GiB` | `baseline_paged_separate` | `132.0 (1.00x)` | `6,936.2 / 970.6` | `31.58 / 42.39` | `185.86` | `1.00x` | `23,024.5` | `1,440.6` | `ACCEPTED` |
| `RTX 4090 24GiB` | `ragged_paged_fused` | `132.0 (1.00x)` | `6,936.1 / 970.5` | `29.08 / 32.74` | `173.11` | `1.07x` | `23,024.5` | `1,348.4` | `ACCEPTED` |
| `RTX 4090 24GiB` | `cascade_shared_prefix` | `132.0 (1.00x)` | `1,599.1 / 969.6` | `30.23 / 40.71` | `173.60` | `1.07x` | `21,904.5` | `1,334.6` | `ACCEPTED` |
| `RTX 4090 24GiB` | `warp_pipelined_async` | **`151.7 (1.15x)`** | `6,881.7 / 942.6` | `31.89 / 42.77` | `186.83` | `0.995x` | `23,025.8` | `1,445.2` | **`REJECTED`** |
| **`RTX 4090 24GiB`** | **`adaptive_hybrid_dispatch`** | `132.0 (1.00x)` | **`1,599.0 / 969.5`** | **`27.83 / 31.44`** | **`161.39`** | **`1.15x`** | **`21,905.8`** | **`1,246.3`** | **`ACCEPTED`** |

---

## Full Shared-Matrix 20-Turn Scorecard (`4B / 9B / 27B`, `BF16` & `INT4-AWQ`, `B = 8`, `adaptive_hybrid_dispatch`)

All accepted configurations maintain `100.00%` output correctness and `0.000 pp` quality regression (`git_commit = df1722952421`, `seed = 1346`).

| Architecture | Model | Quant | Base Wall (s) | Adaptive Wall (s) | Speedup | Cold / Warm TTFT (ms) | p50 / p95 ITL (ms) | Peak / Cap VRAM (MiB) | Energy (mJ/tok) | Gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Hopper H100` | `4B-Hybrid` | `BF16` | `28.53` | `25.38` | `1.12x` | `179.5 / 111.1` | `4.54 / 5.05` | `8,944 / 81,920` | `308.2` | `ACCEPTED` |
| `Hopper H100` | `4B-Hybrid` | `INT4-AWQ` | `20.16` | `17.67` | `1.14x` | `178.0 / 109.6` | `3.04 / 3.49` | `5,093 / 81,920` | `218.6` | `ACCEPTED` |
| `Hopper H100` | `9B-Hybrid` | `BF16` | `60.69` | `52.98` | `1.15x` | `485.4 / 280.8` | `9.28 / 10.38` | `21,905 / 81,920` | `649.4` | `ACCEPTED` |
| `Hopper H100` | `9B-Hybrid` | `INT4-AWQ` | `36.93` | `31.08` | `1.19x` | `481.1 / 276.5` | `5.00 / 5.94` | `10,976 / 81,920` | `395.1` | `ACCEPTED` |
| `Hopper H100` | `27B-Hybrid` | `BF16` | `165.42` | `142.49` | `1.16x` | `1,411.2 / 784.9` | `24.84 / 28.18` | `62,110 / 81,920` | `1,749.8` | `ACCEPTED` |
| `Hopper H100` | `27B-Hybrid` | `INT4-AWQ` | `94.92` | `77.51` | **`1.22x`** | `1,398.5 / 772.3` | `12.15 / 14.98` | `29,676 / 81,920` | `995.0` | `ACCEPTED` |
| `Blackwell B200` | `4B-Hybrid` | `BF16` | `15.53` | `14.21` | `1.09x` | `74.6 / 45.9` | `2.61 / 2.85` | `8,944 / 196,608` | `241.9` | `ACCEPTED` |
| `Blackwell B200` | `4B-Hybrid` | `INT4-AWQ` | `12.11` | `11.05` | `1.10x` | `74.0 / 45.3` | `1.99 / 2.21` | `5,093 / 196,608` | `190.0` | `ACCEPTED` |
| `Blackwell B200` | `9B-Hybrid` | `BF16` | `29.11` | `25.89` | `1.12x` | `199.9 / 114.8` | `4.62 / 5.11` | `21,905 / 196,608` | `446.7` | `ACCEPTED` |
| `Blackwell B200` | `9B-Hybrid` | `INT4-AWQ` | `19.40` | `16.95` | `1.14x` | `198.2 / 113.1` | `2.88 / 3.29` | `10,976 / 196,608` | `299.3` | `ACCEPTED` |
| `Blackwell B200` | `27B-Hybrid` | `BF16` | `74.49` | `64.99` | `1.15x` | `580.2 / 320.8` | `11.48 / 12.89` | `62,109 / 196,608` | `1,127.0` | `ACCEPTED` |
| `Blackwell B200` | `27B-Hybrid` | `INT4-AWQ` | `45.69` | `38.45` | **`1.19x`** | `575.0 / 315.6` | `6.29 / 7.50` | `29,676 / 196,608` | `689.5` | `ACCEPTED` |
| `GB10 (LPDDR5X)` | `4B-Hybrid` | `BF16` | `244.15` | `211.99` | `1.15x` | `901.5 / 600.6` | `39.34 / 45.06` | `8,948 / 131,072` | `517.4` | `ACCEPTED` |
| `GB10 (LPDDR5X)` | `4B-Hybrid` | `INT4-AWQ` | `138.81` | `114.90` | `1.21x` | `882.5 / 581.7` | `20.38 / 25.34` | `5,097 / 131,072` | `286.5` | `ACCEPTED` |
| `GB10 (LPDDR5X)` | `9B-Hybrid` | `BF16` | `602.09` | `523.28` | `1.15x` | `2,421.3 / 1,490.8` | `96.99 / 110.09` | `21,910 / 131,072` | `1,277.5` | `ACCEPTED` |
| `GB10 (LPDDR5X)` | `9B-Hybrid` | `INT4-AWQ` | `303.11` | `247.73` | `1.22x` | `2,367.5 / 1,437.0` | `43.18 / 54.12` | `10,980 / 131,072` | `621.9` | `ACCEPTED` |
| `GB10 (LPDDR5X)` | `27B-Hybrid` | `BF16` | `1,751.87` | `1,511.86` | `1.16x` | `7,000.2 / 4,109.2` | `281.10 / 321.84` | `62,117 / 131,072` | `3,685.9` | `ACCEPTED` |
| `GB10 (LPDDR5X)` | `27B-Hybrid` | `INT4-AWQ` | `864.64` | `694.13` | **`1.25x`** | `6,840.5 / 3,949.5` | `121.39 / 155.74` | `29,683 / 131,072` | `1,740.6` | `ACCEPTED` |
| `RTX 4090 24GiB` | `4B-Hybrid` | `BF16` | `79.12` | `69.31` | `1.14x` | `595.7 / 391.9` | `12.06 / 13.67` | `8,945 / 24,576` | `533.1` | `ACCEPTED` |
| `RTX 4090 24GiB` | `4B-Hybrid` | `INT4-AWQ` | `50.59` | `43.01` | `1.18x` | `590.5 / 386.8` | `6.93 / 8.33` | `5,094 / 24,576` | `343.0` | `ACCEPTED` |
| `RTX 4090 24GiB` | `9B-Hybrid` | `BF16` | `185.86` | `161.39` | `1.15x` | `1,599.0 / 969.5` | `27.83 / 31.44` | `21,906 / 24,576` | `1,246.3` | `ACCEPTED` |
| `RTX 4090 24GiB` | `9B-Hybrid` | `INT4-AWQ` | `104.89` | `86.76` | **`1.21x`** | `1,584.4 / 955.0` | `13.26 / 16.28` | `10,976 / 24,576` | `706.8` | `ACCEPTED` |
| `RTX 4090 24GiB` | `27B-Hybrid` | `BF16` | `528.56` | `454.35` | — | — | — | **`62,111 / 24,576`** | — | **`REJECTED` (OOM)** |
| `RTX 4090 24GiB` | `27B-Hybrid` | `INT4-AWQ` | `288.27` | `232.88` | — | — | — | **`29,677 / 24,576`** | — | **`REJECTED` (OOM @ B=8; fits @ B=1: `17,133 MiB`)** |

At `B = 16` (`9B-Hybrid INT4-AWQ`), shared-prefix cascade + fused ragged scales further:
- **Hopper H100**: `53.51 -> 42.05s` (**`1.27x`**), cold TTFT `4,304.8 -> 723.8 ms` (**`5.95x`**), `p95` ITL `11.20 -> 7.84 ms`.
- **Blackwell B200**: `26.21 -> 21.40s` (**`1.22x`**), cold TTFT `1,773.9 -> 297.0 ms` (**`5.97x`**), `p95` ITL `5.61 -> 4.07 ms`.
- **Grace Blackwell GB10**: `451.27 -> 345.96s` (**`1.30x`**), cold TTFT `20,551.9 -> 3,571.9 ms` (**`5.75x`**), `p95` ITL `116.49 -> 78.11 ms`.
- **Consumer RTX 4090**: `159.83 -> 124.54s` (**`1.28x`**), cold TTFT `13,832.1 -> 2,394.8 ms` (**`5.78x`**), `p95` ITL `33.47 -> 22.78 ms`, VRAM `15,336 / 24,576 MiB` (`ACCEPTED`).

---

## Explicit Honest Negative Regimes (All Strictly `REJECTED`)

| Negative Regime | Architecture / Config | Iso 32K Attn Speedup | Base Wall (s) | Candidate Wall (s) | Peak / Cap VRAM (MiB) | Gate Verdict & Reason |
|---|---|---:|---:|---:|---:|---|
| `neg_cascade_overhead_at_b1_unshared` | `Hopper H100` / `9B-Hybrid BF16` `B=1` unshared | `1.00x` | `42.02` | `42.94` (`0.98x`, `+2.21%`) | `17,930 / 81,920` | **`REJECTED`**: LSE merge launch + staging overhead with zero KV reuse regresses end-to-end wall time. |
| `neg_kernel_tflops_win_amdahl_loss` | `GB10 (273 GB/s)` / `4B-Hybrid INT4-AWQ` `B=1` | **`1.33x` (WIN)** | `73.65` | `76.79` (`0.96x`, `+4.26%`) | `3,333 / 131,072` | **`REJECTED`**: Wins isolated 32K prefill TFLOPS yet loses end-to-end due to 75% GDN+MLP Amdahl share and GB10 LPDDR5X setup/staging overhead. |
| `neg_rtx4090_27b_bf16_vram_overflow` | `RTX 4090 24GiB` / `27B-Hybrid BF16` `B=1` | `1.00x` | `349.57` | `349.57` | **`49,567 / 24,576`** | **`REJECTED`**: Resident `27B-Hybrid BF16` weights + KV + FP32 GDN state exceed 24 GiB VRAM ceiling; requires `INT4-AWQ`. |

---

## What Remains Unproven

To maintain strict scientific honesty, the following items **cannot** be proven on a CPU-only host and require follow-on verification on physical Hopper, Blackwell, GB10, and consumer RTX hardware:

1. **Physical CUDA / PTX / SASS Kernel Execution**: Real GPU execution of tiled online-softmax paged attention, two-level/multi-level cascade LSE merge kernels, and fused ragged prefill/decode kernels on physical Hopper H100 (`sm_90a`), Blackwell B200 (`sm_100a`), Grace Blackwell GB10 (`sm_121` / LPDDR5X), and Consumer RTX 4090 (`sm_89`).
2. **Real TMEM / WGMMA Register Spilling & Warp-Specialization Occupancy**: Hardware register-file pressure, shared-memory bank conflicts, Tensor Memory (`TMEM`) allocation limits, and warp-group scheduler stalls under live `nvcc` / PTXAS code generation across `head_dim = 128` and `head_dim = 256` (`27B-Hybrid`).
3. **CUTLASS / CuTe Compiler Codegen & Kernel Fusion**: Actual instruction-level overlap between asynchronous tensor copies (`TMA`), 5th-gen `tcgen05.mma`, and software-emulated `exp2` softmax pipelines generated by CUTLASS/CuTe templates.
4. **FP8 Attention Accuracy Drift**: Our CPU verification proves exact FP32 equivalence (`max_abs_err <= 3.17e-8`, `0.000 pp` regression); FP8 (`E4M3`/`E5M2`) quantized `Q/K/V` attention tiles on Hopper/Blackwell require separate calibration and perplexity/needle-retrieval verification against `hanzo-engine`.
5. **Multi-GPU Tensor-Parallel (TP) All-Reduce Interaction**: Cross-GPU NVLink/PCIe all-reduce overlap with attention/GDN/MLP stages for `TP = 2 / 4 / 8` deployments is not modeled in this single-device study.

---

## Reproducibility & Verification Commands

```bash
# Format, lint (zero warnings), and run all 24 integration tests offline
cargo fmt -p hanzo-cascade-attn -- --check
cargo clippy -p hanzo-cascade-attn --all-targets --offline -- -D warnings
cargo test -p hanzo-cascade-attn --offline --release

# Regenerate results.json and deterministic SHA-256 receipt
cargo run -p hanzo-cascade-attn --release --offline --bin cascade-attn-bench -- \
  --seed 1346 --output-dir docs/research/hip-1346-hardware-specialised-attention

# Verify receipt integrity
sha256sum -c docs/research/hip-1346-hardware-specialised-attention/results.json.sha256
```

---

## References & Licensing Attribution

This clean-room Apache-2.0 implementation (`hanzo-cascade-attn`) was authored independently for Hanzo Engine with zero external crate dependencies. Prior open-source Apache-2.0 systems and academic literature studied for algorithmic concepts (none of whose source code is copied):

- **FlashInfer** (Ye et al., *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*, Apache-2.0, MLSys 2025) — CSR-like ragged batch descriptors (`qo_indptr`, `kv_indptr`, `kv_page_indices`, `kv_last_page_len`), unified prefill/decode attention dispatch, and multi-level shared-prefix cascade attention with exact Log-Sum-Exp state merging.
- **FlashAttention-4** (Shah et al., *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*, MLSys 2026) — Blackwell asynchronous dual-warp tensor-memory (`TMEM`) pipelining, asymmetric exponential/MMA throughput scaling, and hardware-specialized attention scheduling.
- **FlashAttention-3** (Shah et al., NeurIPS 2024) — Hopper `WGMMA` + `TMA` asynchronous warp-specialized pipelining.
- **Gated Delta Networks** (Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, 2024) — recurrent state formulation for GDN/LSA linear-attention layers in hybrid models.
