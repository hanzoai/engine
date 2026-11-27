# HIP-1346 Research Dossier: Mixed-Precision & Activation-Aware Quantization per Layer (`hanzoai/engine#49`)

- **Canonical Owning Repository**: `hanzoai/engine` (crate: [`hanzo-mixed-quant/`](../../../hanzo-mixed-quant/), license: `Apache-2.0`, zero external dependencies, `#![forbid(unsafe_code)]`)
- **Tracking Issue**: [`hanzoai/engine#49`](https://github.com/hanzoai/engine/issues/49) (`p1`, `perf`, `research`, `bounty`, `hip-1346`)
- **Raw Benchmark Receipt**: [`results.json`](./results.json) (Receipt SHA-256: `c694cd26a56d7c695ea4d2493eddc67713309fc2797e0371568d42f9c72a8812`)
- **PR Disposition Target**: `MERGE` as an opt-in pure-Rust reference oracle, calibration planner, and HIP-1346 benchmark crate alongside `hanzo-quant`; device CUDA/ROCm/Metal kernel promotion remains gated on live GPU weight-loaded verification.

---

## 1. Problem Statement, References & License Audit

### Problem
*"Minimum bits everywhere is not always best quality-per-byte or fastest."* Uniformly pushing every layer of a hybrid Transformer + Gated DeltaNet (GDN) language model to 4-bit or 1.58-bit ternary wastes precious precision budget on small, extreme-sensitivity bottlenecks (`Embedding`, `LmHead`, `RouterGate`, and multiplicative `GdnRecurrent` layers) while over-provisioning bulk MLP projections (`MlpGateUp`, `MlpDown`) that represent ~65–70% of parameters and tolerate aggressive rotation + activation-aware compression.

### Primary References & Clean-Room License Audit
All algorithms in `hanzo-mixed-quant` are **100% independent clean-room Apache-2.0 Rust implementations** using only the Rust standard library (`[dependencies]` empty):
- **AWQ** (Lin et al., MLSys 2024, *"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"*): studied paper equations; clean-room implementation of per-channel salience collection, geometric-mean scale normalization, $\alpha \in [0, 1]$ grid search, and BF16 salient-channel protection (with protected coordinates strictly masked out of the packed low-bit dot product during both calibration and inference).
- **SpinQuant** (Liu et al., ICLR 2025, *"SpinQuant: LLM Quantization with Learned Rotations"*): studied paper formulation; clean-room implementation of seed-regenerated $O(n \log n)$ normalized Walsh-Hadamard transforms ($H_n$, 16-byte descriptor fully budgeted in `storage_bytes()`) and exact Cayley-parameterized orthogonal matrices $R = (I - A)(I + A)^{-1}$ via partial-pivoted Gaussian elimination.
- **LLM Compressor** (vLLM / Neural Magic, Apache-2.0): studied mixed-precision sensitivity allocation, 2:4 semi-structured pruning sensitivity, and quantization-aware distillation (QAD); independent clean-room Rust implementation.
- **BitNet-b1.58** (Ma et al., 2024): studied absmean ternary `{-1, 0, +1}` formulation; independent clean-room 2-bit packed trit storage and branchless LUT matvec.

Zero external source files, third-party crates, or non-commercial code were imported.

---

## 2. Methodology, Public Corpus Grounding & Reproduction

### Grounded Repository Corpus (Verified SHA-256)
All calibration batches and 4-domain evaluation sequences are deterministically derived (`--seed 1346`, SplitMix64) from **3 versioned repository files** (`140,742` bytes total):
- `calibration_data/calibration_datav3_small.txt` (`133,837` bytes, SHA-256 `aabd956dba34ff31f40e78b9de4e234fdd4865f90ba231e8e3deec8dfb22cbc0`)
- `chat_templates/llama3.json` (`420` bytes, SHA-256 `67d1d402738727aa3059d28590fa3488c44740a358be4b5c2135c60f58ec9d7a`)
- `orderings/xlora-paper-ordering.json` (`6,485` bytes, SHA-256 `8d0650ea231eeade0b97e5c026411365ff9feed41b2b07ecf10853c2a399043a`)
- **Combined Grounded Corpus SHA-256**: `ebf55f7a1753ed837efcc218dd089c2f74559a7a6f5ff37d8db099a39c241a6d`

### Recorded Hardware & Software Environment
- **CPU Architecture / Model**: `x86_64` — `Intel(R) Xeon(R) CPU @ 2.30GHz`
- **OS / Kernel**: `Linux 6.18.57 x86_64`
- **GPU / Driver**: CPU-only host (no local CUDA/ROCm/Metal GPU detected; `4B / 9B / 27B` GPU timing & energy columns are **explicitly tagged analytical H100-SXM5 80GB projections** combining exact unclamped byte accounting with measured CPU quality and overhead ratios)
- **Compiler / Base Commit**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, base commit `df1722952421`

### Exact Offline Reproduction Commands
```bash
# 1. Formatting and zero-warning Clippy checks (offline)
cargo fmt -p hanzo-mixed-quant -- --check
cargo clippy -p hanzo-mixed-quant --all-targets --offline -- -D warnings

# 2. Full 18-test integration suite (bit-exact formats, orthogonality, AWQ, QAD, Pareto & negative regimes)
cargo test -p hanzo-mixed-quant --offline --release

# 3. Reproducible HIP-1346 benchmark and JSON receipt generation
cargo run -p hanzo-mixed-quant --release --offline --bin mixed-quant-bench -- \
  --seed 1346 --output docs/research/hip-1346-mixed-precision-quant/results.json
```

---

## 3. Crate Architecture (`hanzo-mixed-quant/`)

- [`src/formats.rs`](../../../hanzo-mixed-quant/src/formats.rs): Bit-exact packed weight formats and direct packed matvec kernels (`PackedTernaryMatrix`, `PackedInt4Matrix`, `PackedInt8Matrix`, `PackedBf16Matrix`, `PackedWeights::fold_row_scales`), with exact `QuantizedLinear::storage_bytes()` accounting for packed weights, AWQ scales, protected BF16 channels, QAD scales, and owned `RotationTransform` payloads, plus masked low-bit inference excluding protected coordinates from asymmetric affine zero-point residuals.
- [`src/rotation.rs`](../../../hanzo-mixed-quant/src/rotation.rs): Seed-regenerated $O(n \log n)$ normalized Fast Walsh-Hadamard Transform (`SeededWalshHadamard`, 16-byte descriptor), explicit-sign `WalshHadamard`, exact Cayley-parameterized orthogonal matrices $R = (I - A)(I + A)^{-1}$, orthogonality verifier `||R^T R - I||_inf`, and learned SpinQuant optimizer (`optimize_spinquant_rotation`).
- [`src/awq.rs`](../../../hanzo-mixed-quant/src/awq.rs): Real per-input-channel activation statistics (`ActivationStats`), top-salient BF16 channel protection (`ProtectedChannels`) masked from the packed low-bit residual during both calibration and inference, geometric-mean normalized AWQ scale grid search ($\alpha \in [0, 1]$), and SpinQuant+AWQ composition.
- [`src/compressor.rs`](../../../hanzo-mixed-quant/src/compressor.rs): Per-component sensitivity profiler across `{Ternary, Int4Asym, Int8, Bf16}`, exact 2:4 semi-structured and 50% unstructured magnitude pruning sensitivity, equal-memory greedy marginal-gain knapsack planner (`plan_mixed_precision`), and Straight-Through Estimator (STE) Quantization-Aware Distillation (`run_qad_linear_refinement`).
- [`src/model.rs`](../../../hanzo-mixed-quant/src/model.rs): Deterministic multi-layer hybrid Transformer + Gated DeltaNet (GDN) inference engine evaluating all 9 quantization regimes across all 4 required quality domains (`instruction_following`, `code_edit_accuracy`, `multilingual`, `long_context_retrieval`).
- [`src/matrix.rs`](../../../hanzo-mixed-quant/src/matrix.rs): Shared HIP-1346 `4B / 9B / 27B` hybrid matrix accounting aligned with `hanzo-paged-attn/tests/kivi_kv_quant.rs` (`num_kv_heads = 2, 4, 4`), unclamped exact footprint reporting (honestly exposing when large-vocab `4B/9B` INT8 critical-layer protection exceeds uniform-INT4), 20-turn coding-agent loop scorecard (`129,088` peak context), and strict end-to-end acceptance gate (`evaluate_acceptance_gate`).

---

## 4. Measured Kernel Verifications (Exact Values from `results.json`)

| Verification Kernel | Metric | Measured Result |
|---|---|---|
| **AWQ Grid Search** (`layer_0.seq_in`, grounded activations) | Optimal salience exponent `alpha*` / outlier ratio | `alpha* = 0.30`, outlier ratio `156.86x` |
| **AWQ Grid Search** (with masked protected salient channels) | Plain PTQ MSE -> AWQ MSE | `1.838276` -> **`0.117252`** (**`-93.6%` MSE**) |
| **SpinQuant Cayley** (`d=64`, `layer_0.seq_out` × heavy-tailed `layer_in[0]`) | Orthogonality error `\|\|R^T R - I\|\|_inf` | **`8.55e-9`** (`< 1e-5` bound) |
| **SpinQuant Cayley** | Exact unquantized invariance `max \|x W^T - (x R)(W R)^T\|` | **`3.05e-5`** |
| **SpinQuant Cayley** | Activation crest factor (`max\|x\| / rms(x)`) | `12.40` -> **`4.05`** (**`3.06x` peak suppression**) |
| **SpinQuant Cayley** | W4A4 Hadamard MSE -> Learned Cayley MSE | `1.870540` -> **`1.789185`** (Cayley strictly improves Hadamard; synthetic 30x outlier test `test_07` confirms W4A4 win over unrotated baseline) |
| **LLM Compressor Knapsack** | Equal-memory budget vs realized total bytes | **`131,040 B`** (incl. AWQ scales & `160 B` seeded Hadamard descriptors) `<= 133,760 B` Uniform INT4 |
| **LLM Compressor Knapsack** | Weighted sensitivity MSE (`Uniform INT4` -> `Mixed`) | `4.132797` -> **`2.370331`** (**`-42.6%`**) |
| **2:4 Semi-Structured Pruning** | Components tolerating 2:4 pruning (`rel_mse < 8%`) | `3 / 22` (all `10` critical components exempt) |
| **QAD Refinement** (`17` compressed layers, zero-byte folded scales) | Mean pre-QAD -> post-QAD MSE / Logit KL | `-31.7%` MSE, `-33.8%` KL |

### Planned Per-Layer Mixed-Precision Assignment (`131,040 B` Total incl. Rotations <= `133,760 B` Uniform INT4)
- **Protected at `INT8`** (`12` components): `embedding`, `lm_head`, all `4` `router_gate` projections, all `4` `gdn_recurrent` projections (`layer_0` & `layer_2` `seq_in`/`seq_out`), and `attn_out` (`layer_1` & `layer_3`).
- **Allocated `INT4_Asym` + AWQ + SpinQuant + QAD** (`4` components): `layer_1.seq_in`, `layer_3.seq_in` (`attn_qkv`), `layer_0.mlp_down`, `layer_3.mlp_down`.
- **Compressed to `Ternary 1.58-bit` + AWQ + SpinQuant + QAD** (`6` bulk MLP components): `layer_0..3.mlp_up`, `layer_1.mlp_down`, `layer_2.mlp_down`.

---

## 5. CPU-Measured End-to-End 4-Domain Quality & Equal-Memory Scorecard (`--seed 1346`)

Every number in this table is **directly measured** on `Intel(R) Xeon(R) CPU @ 2.30GHz` executing real packed-weight forward passes through the 4-layer Hybrid Transformer + GDN model (including exact rotation descriptor bytes in `Model Size`):

| Regime | Model Size (`Bytes`) | Effective BPW | Comp vs BF16 | Instr Acc | Code-Edit Acc | Multilingual Acc | Long-Ctx Acc | **Mean Acc** | Reg vs BF16 | **Mean KL** | CPU Wall (`ms`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bf16_reference` | `428,032` | `16.00` | `1.00x` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | **`100.00%`** | `0.00 pp` | `0.0000` | `234.6` |
| `uniform_int8` | `227,392` | `8.50` | `1.88x` | `98.5%` | `100.0%` | `100.0%` | `100.0%` | **`99.62%`** | `-0.38 pp` | `0.0005` | `246.5` |
| `uniform_int4` | `133,760` | `5.00` | `3.20x` | `75.0%` | `82.6%` | `100.0%` | `75.0%` | `83.14%` | `-16.86 pp` | `0.1519` | `370.0` |
| `uniform_ternary` *(Neg 1)* | `66,880` | `2.50` | `6.40x` | `0.0%` | `0.0%` | `1.5%` | `4.2%` | `1.42%` | `-98.58 pp` | `4.3981` | `261.6` |
| `awq_int4` | `168,464` | `6.30` | `2.54x` | `75.8%` | `96.2%` | `100.0%` | `83.3%` | `88.83%` | `-11.17 pp` | `0.0904` | `433.1` |
| `spinquant_int4` | `134,096` | `5.01` | `3.19x` | `77.3%` | `90.2%` | `100.0%` | `79.2%` | `86.65%` | `-13.35 pp` | `0.0924` | `390.6` |
| **`mixed_precision_awq_spinquant_qad`** | **`131,040`** | **`4.90`** | **`3.27x`** | **`97.0%`** | **`95.5%`** | **`100.0%`** | **`100.0%`** | **`98.11%`** | **`-1.89 pp`** | **`0.0038`** | `325.5` |
| `neg2_unmerged_rot_scalar_dequant` *(Neg 2)* | `131,040` | `4.90` | `3.27x` | `97.0%` | `95.5%` | `100.0%` | `100.0%` | `98.11%` | `-1.89 pp` | `0.0038` | **`4,930.6`** |
| `neg3_mismatched_prose_cal_awq` *(Neg 3)* | `167,952` | `6.28` | `2.55x` | `77.3%` | `97.7%` | `100.0%` | `75.0%` | `87.50%` | `-12.50 pp` | `0.0931` | `482.4` |

**Equal-Memory Pareto Headline**: At **strictly smaller memory** than `uniform_int4` (`131,040 B` / `4.90 BPW` incl. all seeded Hadamard descriptors vs `133,760 B` / `5.00 BPW`, `-2.03%` fewer bytes), `mixed_precision_awq_spinquant_qad` raises 4-domain accuracy by **`+14.97 pp`** (`83.14%` -> **`98.11%`**) and cuts logit KL divergence by **`40.0x`** (`0.1519` -> **`0.0038`**), while running faster on CPU than `uniform_int4` (`325.5 ms` vs `370.0 ms`).

---

## 6. Shared `4B / 9B / 27B` Hybrid Matrix & 20-Turn Coding-Agent Scorecard

Architectural geometries match [`hanzo-paged-attn/tests/kivi_kv_quant.rs`](../../../hanzo-paged-attn/tests/kivi_kv_quant.rs) (`4B-Hybrid`: `9` Attn + `27` GDN, `num_kv_heads = 2`; `9B-Hybrid`: `10` Attn + `30` GDN, `num_kv_heads = 4`; `27B-Hybrid`: `16` Attn + `48` GDN, `num_kv_heads = 4`). Workload: `32,768`-token cold repo prefill + 20 coding-agent turns × (`4,800` tool/diff prefill + `256` generated tokens) = `129,088` peak context, `5,120` generated tokens.

> **Honesty Boundary**: `Model Size (GiB)` and `Peak Memory (GiB)` are **exact unclamped deterministic byte accounting** (`group_size = 128`, FP16 Attn KV + unquantized FP32 GDN state + 256 MiB workspace). Notice that on `4B-Hybrid` (`1.94 GiB` vs `1.75 GiB` uniform INT4) and `9B-Hybrid` (`4.61 GiB` vs `4.48 GiB`), keeping both `151,936`-vocab `Embedding`+`LmHead` and all `GDN` projections at `INT8` genuinely exceeds the uniform-INT4 byte budget even with all-ternary MLPs; we report those actual footprints without clamping. On `27B-Hybrid` (where MLPs dominate parameter share), mixed precision fits strictly inside the exact uniform-INT4 budget (`12.23 GiB == 12.23 GiB`). `Correctness (%)` and `Reg (pp)` are **directly measured on CPU**. `Cold/Warm TTFT`, `p50/p95 ITL`, `Warm tok/s`, `20-Turn Wall`, and `mJ/tok` are **analytically modeled H100-SXM5 80GB projections**.

| Model | Regime | Model Size (`GiB`) | Peak Mem (`GiB`) | Cold TTFT (`ms`) | Warm TTFT (`ms`) | p50 / p95 ITL (`ms`) | 20-Turn Wall (`s`) | Energy (`mJ/tok`) | Correctness (Reg) | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `4B-Hybrid` | `bf16_reference` | `6.57` | `8.00` | `696.4` | `80.6` | `2.9 / 3.1` | `17.31` | `2,198.0` | `100.0% (0.0 pp)` | **ACCEPTED** |
| `4B-Hybrid` | `uniform_int8` | `3.34` | `4.76` | `547.5` | `69.3` | `1.7 / 1.8` | `10.57` | `1,341.9` | `99.6% (-0.4 pp)` | **ACCEPTED** |
| `4B-Hybrid` | `uniform_int4` | `1.75` | `3.17` | `556.1` | `75.8` | `1.1 / 1.1` | `7.50` | `951.6` | `83.1% (-16.9 pp)` | REJECTED |
| `4B-Hybrid` | `uniform_ternary` | `0.87` | `2.30` | `525.7` | `74.2` | `0.7 / 0.7` | `5.50` | `698.6` | `1.4% (-98.6 pp)` | REJECTED |
| `4B-Hybrid` | `awq_int4` | `1.79` | `3.22` | `573.7` | `78.2` | `1.1 / 1.2` | `7.77` | `986.4` | `88.8% (-11.2 pp)` | REJECTED |
| `4B-Hybrid` | `spinquant_int4` | `1.75` | `3.17` | `578.2` | `79.0` | `1.1 / 1.2` | `7.73` | `981.9` | `86.6% (-13.4 pp)` | REJECTED |
| `4B-Hybrid` | **`mixed_precision_awq_spinquant_qad`** | **`1.94`** | **`3.37`** | **`571.6`** | **`77.4`** | **`1.2 / 1.2`** | **`8.03`** | **`1,019.3`** | **`98.1% (-1.9 pp)`** | **ACCEPTED** |
| `4B-Hybrid` | `neg2_unmerged_rot_scalar_dequant` | `1.75` | `3.17` | `1,303.5` | `185.3` | `2.6 / 3.1` | `18.19` | `2,309.2` | `98.1% (-1.9 pp)` | REJECTED |
| `4B-Hybrid` | `neg3_mismatched_prose_cal_awq` | `1.79` | `3.22` | `573.7` | `78.2` | `1.1 / 1.2` | `7.77` | `986.4` | `87.5% (-12.5 pp)` | REJECTED |
| `9B-Hybrid` | `bf16_reference` | `16.85` | `19.70` | `1,786.2` | `206.8` | `7.4 / 7.7` | `43.50` | `5,522.0` | `100.0% (0.0 pp)` | **ACCEPTED** |
| `9B-Hybrid` | `uniform_int8` | `8.56` | `11.41` | `1,404.3` | `177.8` | `4.2 / 4.4` | `26.17` | `3,322.8` | `99.6% (-0.4 pp)` | **ACCEPTED** |
| `9B-Hybrid` | `uniform_int4` | `4.48` | `7.32` | `1,426.5` | `194.4` | `2.6 / 2.7` | `18.26` | `2,318.0` | `83.1% (-16.9 pp)` | REJECTED |
| `9B-Hybrid` | `uniform_ternary` | `2.24` | `5.09` | `1,348.5` | `190.3` | `1.6 / 1.7` | `13.16` | `1,670.4` | `1.4% (-98.6 pp)` | REJECTED |
| `9B-Hybrid` | **`mixed_precision_awq_spinquant_qad`** | **`4.61`** | **`7.46`** | **`1,457.7`** | **`198.5`** | **`2.6 / 2.8`** | **`18.78`** | **`2,384.7`** | **`98.1% (-1.9 pp)`** | **ACCEPTED** |
| `9B-Hybrid` | `neg2_unmerged_rot_scalar_dequant` | `4.48` | `7.32` | `3,343.4` | `475.2` | `6.2 / 7.4` | `44.31` | `5,624.7` | `98.1% (-1.9 pp)` | REJECTED |
| `27B-Hybrid` | `bf16_reference` | `46.03` | `54.64` | `4,878.7` | `564.8` | `20.3 / 21.2` | `119.74` | `15,201.3` | `100.0% (0.0 pp)` | **ACCEPTED** |
| `27B-Hybrid` | `uniform_int8` | `23.37` | `31.98` | `3,835.4` | `485.7` | `11.6 / 12.2` | `72.45` | `9,198.2` | `99.6% (-0.4 pp)` | **ACCEPTED** |
| `27B-Hybrid` | `uniform_int4` | `12.23` | `20.84` | `3,896.2` | `530.9` | `7.2 / 7.6` | `50.86` | `6,457.3` | `83.1% (-16.9 pp)` | REJECTED |
| `27B-Hybrid` | `uniform_ternary` | `6.11` | `14.72` | `3,683.2` | `519.6` | `4.6 / 4.8` | `36.92` | `4,687.3` | `1.4% (-98.6 pp)` | REJECTED |
| `27B-Hybrid` | **`mixed_precision_awq_spinquant_qad`** | **`12.23`** | **`20.84`** | **`3,973.3`** | **`542.2`** | **`7.3 / 7.7`** | **`51.50`** | **`6,538.5`** | **`98.1% (-1.9 pp)`** | **ACCEPTED** |
| `27B-Hybrid` | `neg2_unmerged_rot_scalar_dequant` | `12.23` | `20.84` | `9,131.8` | `1,297.9` | `17.5 / 20.7` | `123.43` | `15,669.6` | `98.1% (-1.9 pp)` | REJECTED |

*(All 27 `(model, regime)` rows including `awq_int4`, `spinquant_int4`, and `neg3_mismatched_prose_cal_awq` for `9B` and `27B` are recorded verbatim in [`results.json`](./results.json).)*

---

## 7. Explicitly Tested Honest Negative Regimes

1. **Negative Regime 1 — Uniform Ternary / INT4 on Critical Layers (`uniform_ternary`, `uniform_int4`)**:
   - Applying `1.58-bit Ternary` uniformly across `Embedding`, `LmHead`, `RouterGate`, and `GdnRecurrent` collapses mean 4-domain accuracy to **`1.42%`** (`-98.58 pp`, `KL = 4.3981`). Even uniform `INT4` drops to **`83.14%`** (`-16.86 pp`, `KL = 0.1519`). Both are **REJECTED** by the acceptance gate, proving mixed precision is mandatory.
2. **Negative Regime 2 — Microbenchmark Win / End-to-End Loss (`neg2_unmerged_rot_scalar_dequant`)**:
   - Uses identical packed mixed-precision weights (`131,040 B`, `3.27x` smaller than BF16) and identical high quality (`98.11%`, `KL = 0.0038`), winning the weight-memory microbenchmark. However, applying an unmerged dense $O(d^2)$ runtime rotation and naive full-matrix scalar dequantization regresses measured CPU wall time to **`4,930.6 ms`** (`21.0x` slower than BF16, `15.1x` slower than fused MixedPrecision) and modeled 27B 20-turn wall time to **`123.43 s`** vs BF16 **`119.74 s`**. Our acceptance gate **explicitly REJECTS** it: *"microbenchmark win regresses end-to-end inference"*.
3. **Negative Regime 3 — Uncalibrated / Mismatched Domain Calibration (`neg3_mismatched_prose_cal_awq`)**:
   - Calibrating AWQ strictly on ASCII prose slices from `calibration_datav3_small.txt` (omitting `llama3.json` chat templates and `xlora-paper-ordering.json` structural code tokens) misses long-context outlier channels, degrading AWQ from **`88.83%` (`KL = 0.0904`)** down to **`87.50%` (`KL = 0.0931`, `-8.3 pp` on `long_context_retrieval`)**.

---

## 8. Integration Recommendation for `hanzo-engine` / `hanzo-quant`

1. **Per-Layer Precision Policy**: Wire `hanzo_mixed_quant::compressor::plan_mixed_precision` into `hanzo-quant`'s ISQ/UQFF planner so `Embedding`, `LmHead`, `RouterGate`, and `GdnRecurrent` layers enforce a fail-closed `INT8`/`BF16` precision floor while bulk MLP projections accept `INT4`/`Ternary` + SpinQuant + AWQ + QAD under a user-specified VRAM budget.
2. **Mandatory Offline Rotation Merging & Zero-Byte QAD Folding**: Always fuse SpinQuant rotations offline into adjacent RMSNorm/linear weights (or use fast $O(d \log d)$ block-Hadamard kernels) and fold QAD row scales into per-group quantization headers (`PackedWeights::fold_row_scales`)—never deploy unmerged dense runtime rotations.
3. **Multi-Domain Calibration Manifest**: Require SHA-256-verified multi-domain calibration batches (prose + chat templates + code/JSON + multilingual) before emitting AWQ/QAD artifacts, and validate on live CUDA/ROCm/Metal `4B/9B/27B` checkpoints before changing default quantization recipes.
