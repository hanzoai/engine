# HIP-1350 Research Dossier: Deterministic AI Execution Profiles (`hanzo-detprofile`, `det.zen-kai.v1`)

## 1. Primary References, Coordination, and License Audit

- **Canonical owning repository & tracking issue**: `hanzoai/engine#65` (`p0`, `bounty`, `research`, `decentralized`, `proof-of-ai`, `hip-1350` — *"Deterministic AI execution profiles"*).
- **Research & protocol references**:
  - Gensyn *RepOps* (*Reproducible Operators for Decentralized Machine Learning*): eliminating cross-hardware floating-point drift via exact fixed-point / integer operator specifications and canonical reduction trees.
  - Zoo *Proof of AI (PoAI)* consensus research (`zoo-apps/papers#13`, `hanzoai/engine#66`): bit-identical operator outputs and cryptographic commitments enabling independent verifier agreement across heterogeneous CPU/GPU providers.
  - NIST FIPS 180-4, *"Secure Hash Standard (SHS)"* (clean-room SHA-256 verified against official NIST test vectors including the 1,000,000-byte `'a'` vector in `hanzo-detprofile/src/commit.rs`).
  - `hanzo-quant` (`Q4_0` 32-weight split-half nibble packing convention in `hanzo-quant/src/gguf/`), mirrored with exact integer `i16` block scales in `PackedInt4Block`.
  - Public calibration corpus anchor: `calibration_data/calibration_datav3.txt` (`279,515` bytes, SHA-256 `200e109bcd2b599fabcceaada7f52bbd1e7c8f9ae030b8dc59c011de039a8026`, sourced from the public Bartowski v3 calibration dataset already versioned in `hanzoai/engine`).
- **License & dependency audit**:
  - `hanzo-detprofile` is a **100% clean-room Apache-2.0** Rust crate using **`std` only** with **zero external crate dependencies** (`[dependencies]` and `[dev-dependencies]` are empty; `#![forbid(unsafe_code)]`; `#![deny(missing_docs)]`).
  - `Cargo.lock` adds only the local `hanzo-detprofile` package entry with zero third-party crates, builds offline (`--offline --release`) in seconds, and imports no proprietary datasets or restricted weights.

---

## 2. Explicit Scope Boundary and Threat Model

> **Mandatory Scope Disclaimer (`hanzoai/engine#65` / `det.zen-kai.v1`)**:
> Execution profile `det.zen-kai.v1` guarantees bit-identical canonical outputs and SHA-256 `ExecutionReceipt` commitments across independent CPU, SIMD-tiled, and GPU-warp implementations **strictly for the 8 declared integer and dyadic fixed-point operators**. Floating-point operators (`Fp16MatMul`, `Bf16MatMul`, `Fp32ParallelReduce`, `AtomicFloatAdd`, `NonDeterministicSoftmax`, `UncappedTopPSampling`, `HardwareFmaDependentOp`, `Fp8DynamicScaleGemm`) are **explicitly rejected** with `DetError::UnsupportedFloatingPointOp` and are **NEVER** silently treated as deterministic.

### Strict Separation of Verification Layers

In compliance with the decentralized Hanzo bounty policy, `hanzo-detprofile` never conflates determinism, cryptographic verification, hardware attestation, or economic settlement:

| Layer | What `hanzo-detprofile` Provides | What Remains Explicitly Out of Scope |
|---|---|---|
| **Deterministic execution semantics** | Bit-identical outputs & SHA-256 commitments (`210 B` wire receipt) across heterogeneous scalar, tiled-SIMD, and GPU-warp reduction topologies for all 8 eligible integer/fixed-point operators. | Any floating-point kernel (`FP16`/`BF16`/`FP32`/`FP8`), vendor `libm` transcendentals, or unprofiled stochastic nucleus sampling. |
| **Cryptographic commitment & verification** | Domain-separated FIPS 180-4 SHA-256 binding `(profile_id, profile_version, op_graph_digest, input_digest, weights_digest, rounding_mode, overflow_mode, reduction_order, silu_lut_version, rmsnorm_eps_q24, output_tensor_bytes)` with constant-time verification (`ct_eq_digest`). | Probabilistic sub-linear matrix proofs (`hanzo-poai` / `#66`) and zero-knowledge succinct proofs are complementary layers. |
| **Hardware attestation** | **None assumed.** Determinism holds purely by algebra (integer associativity + fixed dyadic scaling + total-order tie breaking), requiring zero TEEs, SGX enclaves, or vendor attestation keys. | Physical GPU identity, wall-clock timing, or enclave confidentiality. |
| **Economic / settlement security** | Compact `210 B` little-endian `DZKP` wire receipts suitable for Enso / Zoo PoAI dispute games and audit logs. | On-chain slashing contracts, staking, and challenge periods require independent external security review before holding real funds. |

---

## 3. Mathematical Specification (`det.zen-kai.v1`)

### 3.1 Exact Rational Dyadic Scaling, Rounding, and Overflow Policies (`profile.rs`)

Floating-point scale multiplication is banned from `det.zen-kai.v1`. Every quantization scale is an exact rational dyadic number:
$$S = m \cdot 2^{-s}, \qquad m \in [-2^{31}, 2^{31}-1], \quad s \in \{0, 1, \dots, 62\}.$$

Given an exact accumulator $v \in \mathbb{Z}$ (`i128`), `rescale_i128(v, S, rounding)` forms $p = v \cdot m$ using checked `i128` multiplication, decomposes $p$ into sign and unsigned magnitude $|p| \in [0, 2^{127}]$ (preventing signed negation overflow at `i64::MIN` and `i128::MIN`), and computes $q = \lfloor |p| / 2^s \rfloor$, $r = |p| \bmod 2^s$, $h = 2^{s-1}$:

- **`RoundingMode::RoundHalfToEven` (banker's rounding)**: round magnitude up iff $r > h$, or ($r = h$ and $q \equiv 1 \pmod 2$).
- **`RoundingMode::RoundHalfAwayFromZero`**: round magnitude up iff $r \ge h$ (symmetric around zero: $+0.5 \mapsto +1$, $-0.5 \mapsto -1$).

Narrowing (`i128 -> i32`, `i128 -> i64`, `i128 -> i8`) and `i32` bias addition obey the profile's explicit **`OverflowMode`**:
- **`OverflowMode::Saturating`**: clamps strictly to `[T::MIN, T::MAX]`.
- **`OverflowMode::CheckedError`**: aborts deterministically with `DetError::ArithmeticOverflow` on every backend.

Exact integer division `round_div_i128(num, den, rounding)` compares `rem` against `den - rem` (algebraically equivalent to `2 * rem <=> den` with **zero `u128` overflow** across the full `u128` domain).

### 3.2 Eligible Zen & Kai Operator Set (`ops.rs`)

| # | Canonical Operator | Exact Mathematical Semantics (`det.zen-kai.v1`) | Why Cross-Hardware Bit-Identical |
|---|---|---|---|
| 1 | `Int8Gemm` | $Y_{i,j} = \mathrm{narrow}\!\left(\mathrm{rescale}\!\left(\sum_{k=0}^{K-1} A_{i,k} W_{k,j},\; S\right)\right)$ in exact `i128` | Integer addition is associative & commutative across scalar, `8×8` tiled, and 32-lane warp trees. |
| 2 | `PackedInt4Gemm` | $Y_{i,j} = \mathrm{narrow}\!\left(\mathrm{rescale}\!\left(\sum_{b=0}^{K/32-1} s_{j,b} \sum_{t=0}^{31} A_{i,32b+t} q_{j,32b+t},\; S\right)\right)$, $q \in [-8,7]$ | Distributivity $s\sum a_t q_t = \sum s a_t q_t$ holds exactly over `i128`, matching block-summed CPU tiles and lane-distributed GPU warps bit-for-bit. |
| 3 | `Int32BiasAdd` | $Y_{i,j} = \mathrm{add}_{32}(X_{i,j}, b_j; \text{overflow\_mode})$ | Elementwise exact integer addition under declared overflow mode. |
| 4 | `FixedPointReLU` | $Y_{i,j} = \min(\max(X_{i,j}, 0), \text{cap})$ | Exact integer order comparison. |
| 5 | `FixedPointSiLU_LUT` | Q12 input (`1.0 = 4096`), canonical versioned 257-knot `SIGMOID_LUT_Q15` over $[-8, +8]$ with exact linear interpolation (`>> 8`) and dyadic product (`>> 15`) | Zero runtime floats; symmetric monotone compile-time integer LUT (`LUT[i] + LUT[256-i] == 32768`). |
| 6 | `FixedPointRMSNorm_Int64` | $\mathrm{sum\_sq} = \sum_j |x_j|^2 \in \mathbb{N}_{128}$, exact overflow-free $\mu_{24} = \lfloor \mathrm{sum\_sq}/D \rfloor 2^{24} + \lfloor ((\mathrm{sum\_sq} \bmod D)2^{24})/D \rfloor + \varepsilon_{24}$, bitwise `isqrt_u128`, $y_j = \mathrm{round\_div}((x_j \gamma_j) 2^{12}, \mathrm{isqrt}(\mu_{24}))$ | Zero `f64::sqrt`; exact bitwise integer square root & associative `u128` sum-of-squares across all reduction trees. |
| 7 | `ArgMax_DeterministicTieBreak` | $\arg\max_j x_j$ with associative winner operator: $(v_1, i_1) \succ (v_2, i_2) \iff v_1 > v_2 \lor (v_1 = v_2 \land i_1 < i_2)$ | Total strict order makes parallel warp-tree & tiled-lane reductions bit-identical to left-to-right scan even on ties crossing warp boundaries. |
| 8 | `TopK_DeterministicOrder` | Top-$k$ entries sorted strictly by `(value desc, index asc)` | Bounded sorted merge under total order is associative across tiled chunks and 32-lane warp binary merge trees. |

### 3.3 Explicit Unsupported Floating-Point Exclusion Gate

All 8 ineligible floating-point operators (`Fp16MatMul`, `Bf16MatMul`, `Fp32ParallelReduce`, `AtomicFloatAdd`, `NonDeterministicSoftmax`, `UncappedTopPSampling`, `HardwareFmaDependentOp`, `Fp8DynamicScaleGemm`) are **unconditionally rejected** at both `ExecutionProfile::validate_operator` and `OpGraph::validate` / `DeterministicBackend::execute` with `DetError::UnsupportedFloatingPointOp { op_name, reason }`.

### 3.4 Canonical Serialization & SHA-256 Output Commitments (`wire.rs`, `commit.rs`)

- Wire envelope magic: `b"DZKP"` + versioned tags + little-endian fixed/length-prefixed integers + strict EOF validation (rejecting truncated payloads, trailing bytes, unknown tags, and out-of-bounds length prefixes).
- Master commitment (`ExecutionReceipt`, **`210 B`** serialized wire size; complete profile reconstructible via `ExecutionReceipt::profile()`):
  $$\text{commitment} = \mathrm{SHA256}\!\left(\text{domain} \,\|\, \text{profile\_id} \,\|\, \text{profile\_version} \,\|\, H(\text{graph}) \,\|\, H(\text{inputs}) \,\|\, H(\text{weights}) \,\|\, \text{rounding} \,\|\, \text{overflow} \,\|\, \text{reduction} \,\|\, \text{silu\_lut\_ver} \,\|\, \varepsilon_{24} \,\|\, \text{len}(\text{out}) \,\|\, \text{output\_tensor\_bytes}\right).$$

---

## 4. Bounded Implementation (`hanzo-detprofile`) and Reproducible Commands

### Crate Layout

- `hanzo-detprofile/src/profile.rs`: `ExecutionProfile` (`det.zen-kai.v1`), `RoundingMode`, `OverflowMode`, `ReductionOrder`, `DyadicScale`, exact `rescale_i128` / `round_div_i128` / `isqrt_u128`, `EligibleOp`, `IneligibleFloatingPointOp`, and explicit FP exclusion gate.
- `hanzo-detprofile/src/ops.rs`: `Int8Tensor`, `Int32Tensor`, `PackedInt4Block` / `PackedInt4Weight` (`QUANT_BLOCK_SIZE = 32`), `validate()` boundary invariant checks on tensors and weights, `SIGMOID_LUT_Q15`, `fixed_point_silu_elem`, `RankedEntry`, `OpNode`, and `OpGraph`.
- `hanzo-detprofile/src/wire.rs`: Canonical little-endian `WireWriter` / `WireReader` with anti-DoS bounds and EOF checks.
- `hanzo-detprofile/src/commit.rs`: Clean-room FIPS 180-4 `Sha256`, `ct_eq_digest`, domain-separated commitments, `ReceiptCommitCtx`, and `ExecutionReceipt` (binding the complete `ExecutionProfile` including `silu_lut_version` and `rmsnorm_eps_q24`).
- `hanzo-detprofile/src/backends.rs`:
  - `ScalarCpuReference` (naive strict sequential scalar oracle)
  - `TiledSimdLikeCpuBackend` (`8×8` cache-blocked micro-kernel + `32×8` L1 panel packing + 4-way unrolled lanes + chunked reductions)
  - `SimGpuWarpBackend` (`16×16` threadblocks + `WARP_SIZE = 32` strided lanes + 5-stage butterfly binary tree reduction `16->8->4->2->1`)
  - Explicit negative control: `ScalarFloatRef` vs `UnorderedFloatGpuSim` (`WarpStridedTree32`, `SplitKAtomicReverse`, `FmaWarpTree32`) + exact IEEE-754 `f32_ulp_distance`.
- `hanzo-detprofile/src/bin/detprofile-bench.rs`: Corpus-anchored HIP-1350 acceptance & timing harness outputting `docs/research/hip-1350-deterministic-profiles/results.json`.
- `hanzo-detprofile/tests/acceptance_suite.rs`: Integration, boundary, tie-breaking, overflow, wire-tampering, epsilon-binding, malformed-tensor boundary validation, FP-exclusion, negative-control, and 16-thread concurrency tests.

### Reproducible Commands

```bash
# 1. Offline format check, strict clippy, and release unit + integration + concurrency tests
cargo fmt -p hanzo-detprofile -- --check
cargo clippy -p hanzo-detprofile --all-targets --offline -- -D warnings
cargo test -p hanzo-detprofile --offline --release

# 2. Full HIP-1350 acceptance (512 corpus workloads x 3 backends, 8/8 FP gate,
#    128x4 negative-control divergence runs, and latency/throughput benchmarks)
cargo run -p hanzo-detprofile --release --offline --bin detprofile-bench
```

Reference environment recorded in `docs/research/hip-1350-deterministic-profiles/results.json`:
- **CPU**: `Intel(R) Xeon(R) CPU @ 2.30GHz` (`x86_64`, Linux, single-threaded deterministic measurement)
- **Public Corpus Anchor**: `calibration_data/calibration_datav3.txt` (`279,515 B`, SHA-256 `200e109bcd2b599fabcceaada7f52bbd1e7c8f9ae030b8dc59c011de039a8026`)

---

## 5. Verification Evidence, Real Measurements, and Negative-Control / Negative-Regime Analysis

All numbers below are transcribed verbatim from `docs/research/hip-1350-deterministic-profiles/results.json`.

### 5.1 Acceptance Gate Results (`hanzoai/engine#65`)

| Acceptance Gate | Tested | Passed / Rejected | Agreement / Rejection Rate | Wall Time |
|---|---:|---:|---:|---:|
| Unit + integration + 16-thread concurrency (`cargo test --release`) | `21` tests (`11` unit + `10` integration) | **`21 / 21` PASS** | `100.00%` | `0.04 s` |
| **Shared deterministic corpus across 3 independent backends** (`ScalarCpuReference` == `TiledSimdLikeCpuBackend` == `SimGpuWarpBackend`) | `512` workloads (`1,536` runs) | **`512 / 512` bit-identical bytes & SHA-256 commitments** | **`100.0000%`** (`0` divergences) | `0.5298 s` |
| **Unsupported floating-point exclusion gate** (all 8 FP ops × 3 backends) | `8` ops (`24` invocations) | **`8 / 8` REJECTED (`24 / 24`)** | **`100.00%`** (`DetError::UnsupportedFloatingPointOp`) | `< 0.01 s` |
| Canonical `ExecutionReceipt` wire footprint | `512` receipts | `210 B` fixed envelope | `100.00%` round-trip verified | — |

Per-category breakdown across the 512 corpus-anchored workloads (covering both `RoundHalfToEven` & `RoundHalfAwayFromZero`, both `Saturating` & `CheckedError`, and both `FixedBinaryTree` & `LeftToRight`):

| Workload Category | Tested | Bit-Identical (All 3 Backends) | Sample SHA-256 Commitment (Prefix) |
|---|---:|---:|---|
| `Int8Gemm` | `52` | `52 / 52` | `9cebe6d35d952fb2...` |
| `PackedInt4Gemm` | `52` | `52 / 52` | `b51296c3de0e29c7...` |
| `Int32BiasAdd` | `51` | `51 / 51` | `36dc63460df955c2...` |
| `FixedPointReLU` | `51` | `51 / 51` | `de942af14f3a42e2...` |
| `FixedPointSiLU_LUT` | `51` | `51 / 51` | `0ed262ca2f44727d...` |
| `FixedPointRMSNorm_Int64` | `51` | `51 / 51` | `56d8b693357cf08a...` |
| `ArgMax_DeterministicTieBreak` | `51` | `51 / 51` | `00dcb1b92dbf8e4e...` |
| `TopK_DeterministicOrder` | `51` | `51 / 51` | `39873462b0bb4160...` |
| `ZenKaiMicroTransformer_ArgMax` (7-stage block) | `51` | `51 / 51` | `d33305bc9874fd56...` |
| `ZenKaiMicroTransformer_TopK` (7-stage block) | `51` | `51 / 51` | `ace4b55684a171d6...` |

### 5.2 Explicit Negative Control: Why Unprofiled FP32 GPU Execution Diverges Bit-for-Bit

Running 128 corpus-anchored `4×128×64` GEMM + RMSNorm + SiLU workloads (`32,768` total `f32` outputs per comparison) under `ScalarFloatRef` vs `UnorderedFloatGpuSim` proves empirically why floating-point operators MUST be excluded from deterministic consensus profiles:

| Negative-Control Comparison (`f32`) | Workloads | Diverged SHA-256 Digests | Digest Divergence Rate | Mismatched Elements | Element Mismatch Rate | Max ULP | Mean ULP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ScalarFloatRef` vs `UnorderedFloatGpuSim(warp_strided_tree_32)` | `128` | **`128 / 128`** | **`100.00%`** | `25,125 / 32,768` | `76.68%` | `25,231` | `10.38` |
| `ScalarFloatRef` vs `UnorderedFloatGpuSim(split_k_atomic_reverse)` | `128` | **`128 / 128`** | **`100.00%`** | `25,642 / 32,768` | `78.25%` | `24,857` | `10.35` |
| `ScalarFloatRef` vs `UnorderedFloatGpuSim(fma_warp_tree_32)` | `128` | **`128 / 128`** | **`100.00%`** | `25,156 / 32,768` | `76.77%` | `20,123` | `9.71` |
| `UnorderedFloatGpuSim(warp_strided_tree_32)` vs `UnorderedFloatGpuSim(split_k_atomic_reverse)` | `128` | **`128 / 128`** | **`100.00%`** | `21,666 / 32,768` | `66.12%` | `25,230` | `8.43` |

Even **between two GPU FP32 reduction schedules on the same machine** (`warp_strided_tree_32` vs `split_k_atomic_reverse`), **100.0% of SHA-256 commitments diverge** and `66.12%` of individual `f32` elements differ (up to `25,230` ULPs near zero crossings after RMSNorm + SiLU amplification!). In contrast, `det.zen-kai.v1` achieves **`0 / 512` (`0.00%`) divergence** across scalar, tiled-SIMD, and 32-lane GPU-warp execution.

### 5.3 Throughput, Tail Latency (`p50` / `p95`), and Honest Negative-Regime Overhead Analysis

All deterministic backend timings include **full end-to-end execution PLUS cold SHA-256 commitment hashing** of the graph, inputs, weights, and output bytes (`ExecutionReceipt`), whereas the unordered `f32` baselines hash only outputs:

| Workload & Shape (`M×K×N`) | MAC Ops | `ScalarCpuRef` `p50 / p95` (`us`) | `TiledSimdCpu` `p50 / p95` (`us`) | `SimGpuWarp` `p50 / p95` (`us`) | Unordered `f32` Scalar `p50` (`us`) | Unordered `f32` GPU-Sim `p50` (`us`) | `TiledDet / F32` Ratio | `TiledDet` Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Int8Gemm+RMSNorm+SiLU` (`1×64×64`, *neg regime*) | `4,096` | `42.40 / 67.48` | `52.22 / 162.73` | `47.47 / 65.56` | `6.20` | `8.26` | `8.42x` | `78.44 MMAC/s` |
| `Int8Gemm+RMSNorm+SiLU` (`4×64×64`) | `16,384` | `84.99 / 140.73` | `93.39 / 7223.09` | `95.38 / 153.55` | `26.64` | `31.86` | `3.51x` | `175.44 MMAC/s` |
| `Int8Gemm+RMSNorm+SiLU` (`8×128×128`) | `131,072` | `371.11 / 496.87` | `465.02 / 4473.95` | `504.48 / 6562.89` | `217.26` | `205.83` | `2.14x` | `281.86 MMAC/s` |
| `Int8Gemm+RMSNorm+SiLU` (`16×128×128`) | `262,144` | `756.04 / 14796.03` | `808.90 / 21129.67` | `1045.43 / 11482.67` | `462.82` | `502.20` | `1.75x` | `324.08 MMAC/s` |
| **`PackedInt4+RMSNorm+SiLU` (`16×128×128`)** | `262,144` | `698.57 / 10810.03` | **`444.56 / 7084.93`** | `913.27 / 9847.80` | `419.33` | `387.27` | **`1.06x`** | **`589.67 MMAC/s`** |

**Honest Negative-Regime Findings**:
1. **Tiny single-token decode (`1×64×64`) exhibits `8.42x` end-to-end overhead** (`52.22 us` vs `6.20 us` unordered `f32`) because `execute()` recomputes cold `SHA256(weights)` over the full weight matrix on every single-token invocation plus 128-bit integer `isqrt_u128` RMSNorm overhead. In production inference, `weights_digest` MUST be computed once at model load time and cached per layer rather than re-hashed per decode token.
2. **`PackedInt4Gemm` tiled execution reaches near-parity (`1.06x`, `589.67 MMAC/s`) with unordered `f32` at `M = 16`** (`444.56 us` vs `419.33 us`), because unpacking each 32-weight block once per `8×8` register tile amortizes nibble decoding AND shrinks memory bandwidth by `8x` vs `f32` (`18 B` per 32 weights vs `128 B` for `f32`).
3. **`SimGpuWarpBackend` CPU emulation overhead (`~1.1–2.1x` vs tiled CPU)** reflects software simulation of 32-lane warp shuffles on a scalar CPU; native CUDA/Metal/ROCm warp shuffle (`__shfl_down_sync`) executes the identical 5-stage `i64`/`i128` butterfly tree in 5 warp instructions with zero host memory traffic.

### 5.4 Decentralized Hanzo Bounty 5-Axis Scorecard

- **Cost**: Constant `210 B` canonical wire receipt per verified graph/block; `PackedInt4Gemm` + fixed-point epilogue adds only `~6.0%` (`1.06x`) end-to-end latency over unordered `f32` at `M = 16` prefill while eliminating `100%` of false-dispute slash/re-execution waste caused by FP32 non-determinism.
- **Performance**: `589.67 MMAC/s` (`444.56 us` `p50`) on a single 2.30 GHz Xeon core for deterministic `16×128×128` `PackedInt4 + RMSNorm + SiLU` including full SHA-256 commitments; `100.0%` completion rate (`512 / 512` workloads).
- **Trust**: `100.00%` (`512 / 512`) cross-backend bit-exactness across 3 independent reduction topologies, `100.00%` (`8 / 8`) rejection of non-deterministic floating-point ops, complete profile binding (`silu_lut_version`, `rmsnorm_eps_q24`) in every receipt, and empirical proof that unprofiled FP32 diverges on `128 / 128` (`100.00%`) runs.
- **Decentralization**: `100%` of execution, commitment generation, and receipt verification runs independently with zero Hanzo-controlled servers, zero KMS dependencies, and zero external crate fetches.
- **Sovereignty**: Pure Apache-2.0 `std`-only Rust crate with canonical `DZKP` wire format; any node operator, verifier, or fork can build, audit, migrate, or port `det.zen-kai.v1` to CUDA/Metal/ROCm/WebAssembly without vendor lock-in.

---

## 6. Security Review, Rollout/Rollback, and Integration Roadmap

- **Security review**:
  - `#![forbid(unsafe_code)]` across the library, benchmark CLI, and test suite.
  - Strict input validation enforced at both constructors AND the execution boundary (`GraphInput::validate`, `OpGraph::validate`, `Int8Tensor::validate`, `Int32Tensor::validate`, `PackedInt4Weight::validate`, and `requantize_i32_to_i8`) preventing out-of-bounds panics even if callers construct or mutate public tensor fields directly, alongside packed 4-bit weight range `[-8, 7]`, dyadic shift bounds (`0..=62`), non-zero RMSNorm epsilon, and profile version enforcement (`det.zen-kai.v1` v1 only).
  - Overflow-hardened extreme arithmetic (`unsigned_abs()` in `u128`, `rem.cmp(&(den - rem))` exact midpoint comparison, overflow-free `q << 24 + (r << 24)/d` RMSNorm scaling) verified against `i32::MIN`, `i64::MIN`, and `i128::MIN`.
  - Complete numerical profile binding via `ReceiptCommitCtx` (exhaustively destructuring `ExecutionProfile` to include `silu_lut_version` and `rmsnorm_eps_q24` in both the wire encoding and SHA-256 commitment), constant-time SHA-256 digest comparison (`ct_eq_digest`), and strict wire decoder (`MAX_WIRE_ELEMENTS` anti-OOM limit, magic/tag check, EOF trailing-byte rejection).
- **Disposition, Rollout, and Rollback**:
  - **Crate `hanzo-detprofile`**: Ready for `MERGE` as a standalone opt-in workspace crate (touches only root `Cargo.toml` workspace membership/dependencies and adds zero external crates to `Cargo.lock`; zero impact on existing hot paths). Rollback is a single clean `git revert`.
  - **On-chain dispute / slashing settlement**: Explicitly `GATED` pending independent external cryptographic and smart-contract security audit prior to binding real funds.
- **Integration roadmap (`hanzo-engine`, `hanzo-quant`, `hanzo-router`, Enso / Zoo PoAI)**:
  1. **Cached model-load weight digests**: Compute `OpGraph::weights_digest()` once during `hanzo-quant` GGUF/ISQ load and attach to layer metadata to eliminate per-token decode SHA-256 overhead.
  2. **Native GPU kernels matching `SimGpuWarpBackend`**: Implement CUDA/ROCm/Metal `det.zen-kai.v1` kernels executing the exact 5-stage `i64`/`i128` warp butterfly tree and `SIGMOID_LUT_Q15` table, gated behind `--det-profile det.zen-kai.v1` in `hanzo-engine` and verified bit-for-bit against `ScalarCpuReference` in CI.
  3. **Composable PoAI verification**: Pair `det.zen-kai.v1` `ExecutionReceipt` (`210 B`) with Freivalds matrix proofs (`hanzo-poai`, `#66`) inside Enso routing and Zoo PoAI challenge-response transcripts.
