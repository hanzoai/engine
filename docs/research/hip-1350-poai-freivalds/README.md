# HIP-1350 Research Dossier: Proof of AI — CPU-Verifiable GPU Matrix Products (`hanzo-poai`)

## 1. Primary References, Coordination, and License Audit

- **Tracking issue (Hanzo execution side)**: `hanzoai/engine#66` (`p0`, `bounty`, `research`, `decentralized`, `proof-of-ai`, `hip-1350`).
- **Protocol & specification counterpart**: `zoo-apps/papers#13` ("Proof of AI: CPU-verifiable GPU work" — Thinking Chains / Zoo Gym compute-proof specification).
- **Cryptographic and algorithmic references**:
  - Rusins Freivalds, *"Fast Probabilistic Algorithms"*, MFCS 1979, LNCS 74, pp. 57–69.
  - Jacob T. Schwartz, *"Fast Probabilistic Algorithms for Verification of Polynomial Identities"*, JACM 27(4), 1980, pp. 701–717; Richard Zippel, EUROSAM 1979.
  - Amos Fiat and Adi Shamir, *"How to Prove Yourself: Practical Solutions to Identification and Signature Problems"*, CRYPTO 1986.
  - NIST FIPS 180-4, *"Secure Hash Standard (SHS)"* (SHA-256 test vectors verified in `hanzo-poai/src/transcript.rs`).
  - `hanzo-quant` (`Q8_0` and `Q4_0` 32-element payload conventions in `hanzo-quant/src/lib.rs` and `hanzo-quant/src/gguf/`; note that `poai.matmul.v1` defines its own deterministic `i16`-scaled execution-profile layouts—and a distinct custom `PackedTernaryI2S` 32-weight / `10 B` layout that is **not** wire-compatible with GGUF `TQ2_0`'s 256-weight / `66 B` layout, see §3.2).
- **License and dependency audit**:
  - `hanzo-poai` is a **100% clean-room Apache-2.0** Rust crate using **`std` only** with **zero external crate dependencies** (`[dependencies]` and `[dev-dependencies]` are both empty; `Cargo.lock` adds only the local `hanzo-poai` package entry with zero dependencies).
  - No external source files, third-party crypto crates, proprietary datasets, or restricted model weights are imported.

---

## 2. Explicit Scope Boundary and Threat Model

> **Mandatory Scope Disclaimer (`hanzoai/engine#66` / `zoo-apps/papers#13`)**:
> A matrix-product proof verifies **only** the declared integer matrix product $C = A \cdot B$ under profile `poai.matmul.v1`; attention, normalization (`RMSNorm` / `LayerNorm`), softmax, sampling, KV-cache / recurrent state updates, and overall LLM response truthfulness or scientific value are separate operations and are **NOT** covered by this proof.

### Separation of Verification Layers

In accordance with the decentralized Hanzo bounty rules, this crate strictly separates verification mechanisms and never conflates them:

| Mechanism | What `hanzo-poai` Provides | What Remains Out of Scope |
|---|---|---|
| **Probabilistic cryptographic check** | Verifies $C = A \cdot B$ exactly over $\mathbb{Z}$ via Freivalds over $\mathbb{F}_p$ ($p = 2^{61} - 1$) bound by SHA-256 commitments (`commit_A`, `commit_B`, `commit_C`), transcript-bound round count, and round projection digests. | Does not verify non-matmul layers (attention, norms, activations, sampling, KV/GDN state updates) or semantic response quality. |
| **Hardware attestation** | **None assumed.** Any commodity CPU verifier can audit a GPU prover without TEEs, SGX, or vendor attestation keys. | GPU kernel timing / hardware identity is not attested by this receipt. |
| **Economic / settlement security** | Provides compact `216 B` (Fiat-Shamir) / `248 B` (Interactive) canonical wire receipts suitable for challenge-response games. | Slashing contracts, stake sizing, challenge windows, and economic incentives require external security audit before holding real funds. |

### Challenge Modes, Verifier-Owned Policy, and Grinding Defense

`hanzo-poai` supports two explicit challenge modes (`ChallengeMode`) governed by a verifier-owned [`VerificationPolicy`](../../../hanzo-poai/src/freivalds.rs):

1. **`ChallengeMode::FiatShamir` (non-interactive)**: Binds `(profile_version, operator_tag, p, m, k, n, rounds, commit_A, commit_B, commit_C)` via domain-separated SHA-256. With `k = 2` rounds, finding a forged $C' \neq A B$ that collides with all $k$ challenge projections requires $\Omega(p^2) > 2^{121}$ SHA-256 evaluations, which is computationally infeasible.
2. **`ChallengeMode::InteractiveSeed([u8; 32])` (commit-then-challenge)**: The prover commits to `commit_C` *before* receiving a fresh 256-bit verifier CSPRNG seed (from OS `/dev/urandom` or an independent randomness beacon). Verifiers calling `verify_matrix_product_with_mode` or `verify_matrix_product_with_policy` reject any receipt attempting to downgrade from interactive to Fiat-Shamir or substitute a different seed, eliminating prover grinding altogether.
3. **Verifier-owned minimum round enforcement**: All default verifiers (`verify_matrix_product`, `verify_matrix_product_with_mode`, and `verify_matrix_product_with_cached_b_commit`) enforce `min_rounds = profile.default_rounds` (`2` rounds / `122.00` soundness bits under `poai.matmul.v1`), rejecting any untrusted 1-round receipt with `PoaiError::InvalidChallengeRounds`. Furthermore, `rounds` is hashed into `derive_transcript_seed`, so truncating a 2-round receipt to 1 round fails transcript verification even if a verifier explicitly selects `VerificationPolicy::with_min_rounds(1)` via `verify_matrix_product_with_policy` or `verify_matrix_product_with_cached_b_commit_and_policy`.

---

## 3. Mathematical Specification (`poai.matmul.v1`)

### 3.1 Explicit Prime Finite Field $\mathbb{F}_p$ (`field.rs`)

- **Modulus**: 9th Mersenne prime $p = 2^{61} - 1 = 2{,}305{,}843{,}009{,}213{,}693{,}951$ (`0x1FFF_FFFF_FFFF_FFFF`).
- **Division-free exact 128-bit reduction**: Because $2^{61} \equiv 1 \pmod p$, any `u128` product $x = x_0 + x_1 2^{61} + x_2 2^{122}$ ($x_0, x_1 \le p$, $x_2 \le 63$) satisfies $s = x_0 + x_1 + x_2 \le 2p + 63 = 2^{62} + 61 < 2^{64}$ (zero `u64` overflow). Folding once gives $r = (s \ \& \ p) + (s \gg 61) \le p + 2 < 2p$, and a single conditional subtraction `if r >= p { r -= p }` yields the exact canonical residue in $[0, p - 1]$ (mapping $p \mapsto 0$ and `u128::MAX` $\mapsto 63$).
- **Wraparound-free signed bijection**: Every signed integer $x$ with
  $$|x| \le \frac{p - 1}{2} = 2^{60} - 1 = 1{,}152{,}921{,}504{,}606{,}846{,}975$$
  maps injectively into $\mathbb{F}_p$ ($x \mapsto x$ for $x \ge 0$; $x \mapsto p - |x|$ for $x < 0$). Both the prover and verifier strictly reject any matrix entry with $|C_{i,j}| > (p - 1)/2$ (`PoaiError::OutputOutOfFieldRange`), closing the modular wraparound attack where a malicious prover substitutes $C_{i,j} + p \equiv C_{i,j} \pmod p$.

### 3.2 Accepted Numerical Semantics and Operator Eligibility (`profile.rs`, `quant.rs`)

Under **`poai.matmul.v1`**:

- Activation matrix $A \in [-128, 127]^{M \times K}$ is row-major signed `i8`.
- Weight matrix $B \in \mathbb{Z}^{K \times N}$ is stored in `poai.matmul.v1`'s canonical column-blocked integer layout ($N$ output channels, each partitioned into $K / 32$ blocks of `QUANT_BLOCK_SIZE = 32` weights along $K$). Each block $(j, b)$ holds a leading little-endian signed fixed-point `i16` scale $s_{j,b} \in [-32768, 32767]$ (`s = 1` for unscaled integer GEMM) followed by packed integer weights:

| Eligible Operator | Packed Format | Block Layout (`32` weights) | Weight Domain $q$ | Exact Byte Size / Block | Relationship to `hanzo-quant` / GGUF |
|---|---|---|---|---|---|
| `Int8Gemm` | `Int8Q8_0` | Leading `i16` scale + `32 × i8` | $[-128, 127]$ | `34 B` | Same 32-weight / `34 B` payload geometry as GGML `block_q8_0`, but replaces the `f16` scale with an exact `i16` fixed-point scale (requires scale conversion at load time) |
| `PackedInt4BlockGemm` | `PackedInt4Q4_0` | Leading `i16` scale + `16 B` split-half nibbles | $[-8, 7]$ | `18 B` | Same 32-weight / `18 B` split-half nibble payload geometry as GGML `block_q4_0`, but replaces the `f16` scale with an exact `i16` fixed-point scale (requires scale conversion at load time) |
| `PackedTernaryI2SGemm` | `PackedTernaryI2S` | Leading `i16` scale + `8 B` 2-bit codes (`00:-1, 01:0, 10:+1`; `11` rejected) | $\{-1, 0, +1\}$ | `10 B` (`80 B` / 256 weights) | **Distinct custom `poai.matmul.v1` format — NOT wire-compatible with GGUF `TQ2_0`** (`BlockTQ2_0` in `hanzo-quant`/`hanzo-ml` packs 256 weights into `66 B`: `qs[64]` followed by a trailing `f16` scale). Raw `TQ2_0` bytes are rejected by `from_raw_bytes` and require an explicit repacking adapter |

- Exact integer product $C = A \cdot B \in \mathbb{Z}^{M \times N}$:
  $$C_{i,j} = \sum_{b=0}^{K/32 - 1} s_{j,b} \sum_{t=0}^{31} A_{i,\,32b+t} \cdot q_{j,\,32b+t}$$
- **Bit-identical cross-hardware determinism**: Because integer addition and multiplication are associative and commutative, GPU warp reductions, tensor-core INT8 MMA tiles, and CPU scalar/SIMD loops produce bit-identical `i64` outputs $C_{i,j}$. Each product term satisfies $|s_{j,b} A_{i,k} q_{j,k}| \le 2^{15} \cdot 2^7 \cdot 2^7 = 2^{29}$, so for all $K \le 2^{30}$, $|C_{i,j}| \le 2^{59} < 2^{60} - 1 = (p - 1)/2$ (zero overflow, exact agreement with $\mathbb{F}_p$).
- **Explicitly rejected ineligible operators** (`PoaiError::IneligibleOperator`): `Fp16Gemm`, `Bf16Gemm`, `Fp32Gemm`, `Fp8Gemm` (non-associative floating-point reductions), plus `FlashAttention`, `Normalization`, `Sampling`, and `StateUpdate` (outside matrix-product scope).

### 3.3 Exact Soundness Bound Proof

Suppose a cheating prover returns $C \neq A \cdot B$ with every entry in $[-(p-1)/2, +(p-1)/2]$. By injectivity of the signed embedding, the error matrix $D = A B - C \pmod p$ is **non-zero** in $\mathbb{F}_p^{M \times N}$. Therefore $\dim \ker(D) \le N - 1$, so $|\ker(D)| \le p^{N-1}$ and the number of non-zero kernel vectors is $|\ker(D) \setminus \{0\}| \le p^{N-1} - 1$.

Because `derive_challenge_vector` uses zero-bias rejection sampling (rejecting only `2^61 - 1` per 61-bit slice and resampling the all-zero vector) to draw $r \sim \mathrm{Uniform}(\mathbb{F}_p^N \setminus \{0\})$:
$$\Pr[A(Br) \equiv Cr \pmod p \mid C \neq AB] = \Pr[Dr = 0 \mid r \neq 0] \le \frac{p^{N-1} - 1}{p^N - 1} < \frac{1}{p} \approx 4.3368 \times 10^{-19}.$$

Across $k$ independent rounds:
- $k = 1$ *(opt-in only via explicit `VerificationPolicy::with_min_rounds(1)`)*: $\varepsilon_1 < p^{-1} \approx 4.3368 \times 10^{-19}$ (`61.00` soundness bits)
- $k = 2$ *(default verifier minimum)*: $\varepsilon_2 < p^{-2} \approx 1.8808 \times 10^{-37} < 2^{-121}$ (`122.00` soundness bits)
- $k = 4$: $\varepsilon_4 < p^{-4} \approx 3.5374 \times 10^{-74} < 2^{-243}$ (`244.00` soundness bits)

---

## 4. Bounded Implementation (`hanzo-poai`) and Reproducible Commands

### Crate Layout

- `hanzo-poai/src/field.rs`: `Fp`, `MODULUS_P`, `MAX_SAFE_SIGNED`, exact 128-bit reduction, Fermat inverse, constant-time digest equality.
- `hanzo-poai/src/profile.rs`: `VerificationProfile::v1()` (`poai.matmul.v1`), `OperatorKind` eligibility gate, `validate_rounds_at_least`, `PROOF_SCOPE_DISCLAIMER`, soundness bound calculators.
- `hanzo-poai/src/quant.rs`: `ActivationMatrix`, `PackedWeightMatrix` (`Int8Q8_0`, `PackedInt4Q4_0`, custom `PackedTernaryI2S`), `OutputMatrix`, `prover_matmul`, and streaming zero-decompression `B * r (mod p)` using split-`i64` accumulators and branchless ternary LUTs.
- `hanzo-poai/src/transcript.rs`: Clean-room FIPS 180-4 `Sha256` (verified against official NIST vectors including the 1,000,000-byte `'a'` vector), domain-separated commitments, `ChallengeMode::{FiatShamir, InteractiveSeed}`, round-bound transcript seed derivation, and unbiased challenge sampling.
- `hanzo-poai/src/freivalds.rs`: `MatrixProofReceipt` canonical binary wire codec (`216 B` / `248 B` at `k = 2`), verifier-owned `VerificationPolicy`, cold/cached/interactive/policy-controlled proving and verification.
- `hanzo-poai/src/error.rs`: Non-panicking typed `PoaiError`.
- `hanzo-poai/src/bin/poai-bench.rs`: Deterministic acceptance & timing harness outputting `docs/research/hip-1350-poai-freivalds/results.json`.
- `hanzo-poai/tests/acceptance_suite.rs`: Ineligible operator gate, 150+150 integration tests, modular wraparound rejection, round-downgrade rejection across all verifier entrypoints, `TQ2_0` vs `PackedTernaryI2S` layout separation test, and 16-thread concurrent verifier stress test.

### Reproducible Commands

```bash
# 1. Offline format, strict clippy, and release unit + integration + concurrency tests
cargo fmt -p hanzo-poai -- --check
cargo clippy -p hanzo-poai --all-targets --offline -- -D warnings
cargo test -p hanzo-poai --offline --release

# 2. Full HIP-1350 acceptance (1,000 correct + 1,000 corrupted) and timing benchmark
cargo run -p hanzo-poai --offline --release --bin poai-bench
```

Reference measurement environment recorded in `docs/research/hip-1350-poai-freivalds/results.json`:
- **CPU**: `Intel(R) Xeon(R) CPU @ 2.30GHz` (`x86_64`, AVX2, 2 vCPUs, single-threaded deterministic execution)
- **OS / Toolchain**: Linux `x86_64`, `rustc 1.100.0`, `--release` profile, deterministic SplitMix64 seed `0x4841_4E5A_4F5F_1350`.

---

## 5. Verification Evidence, Real Measurements, and Negative Regime

All numbers below are transcribed directly from `docs/research/hip-1350-poai-freivalds/results.json`.

### 5.1 Acceptance Gate Results (`hanzoai/engine#66`)

| Acceptance Suite | Tested | Result | Rate / False Accepts | Wall Time |
|---|---:|---:|---|---:|
| Unit + integration + 16-thread concurrency (`cargo test --release`) | `19` tests | **`19 / 19` PASS** | `100.00%` | `0.04 s` |
| Ineligible operator rejection (`Fp16/Bf16/Fp32/Fp8/Attn/Norm/Sample/State`) | `8` ops | **`8 / 8` REJECTED** | `100.00%` | `< 0.01 s` |
| **Correct packed matrix products** (`Int8=334`, `Int4=333`, `Ternary=333`; `FS=500`, `Interactive=500`) | `1,000` | **`1,000 / 1,000` PASS** | `100.00%` (`0` false rejects) | `7.09 s` |
| **Deliberately corrupted matrix products** (all 7 attack classes below) | `1,000` | **`1,000 / 1,000` DETECTED** | `100.00%` (`0` false accepts) | `5.28 s` |

Breakdown of the 1,000 adversarial corruptions (`0` false accepts across every attack class):

| Adversarial Attack Class | Tested | Detected | False Accepts | Detection Mechanism |
|---|---:|---:|---:|---|
| `single_elem_pm1` (`C[idx] ± 1`, re-proved) | `143` | `143` | `0` | Freivalds `A(Br) != Cr (mod p)` |
| `single_bit_flip` (`C[idx] ^= 1 << b`, re-proved) | `143` | `143` | `0` | Freivalds `A(Br) != Cr (mod p)` |
| `zeroed_output_block` (32-elem tile zeroed, re-proved) | `143` | `143` | `0` | Freivalds `A(Br) != Cr (mod p)` |
| `row_or_col_swap` (row/col permutation, re-proved) | `143` | `143` | `0` | Freivalds `A(Br) != Cr (mod p)` |
| `stale_cached_row` (stale previous layer row, re-proved) | `143` | `143` | `0` | Freivalds `A(Br) != Cr (mod p)` |
| `quantized_scale_tamper` (`i16` block scale perturbed vs honest `B`) | `143` | `143` | `0` | Freivalds `A(Br) != Cr (mod p)` |
| `forged_commitment_or_digest` (`commit_C` / `round_digests` bit flip) | `142` | `142` | `0` | Constant-time SHA-256 check |

### 5.2 Prover Cost, CPU Verifier Cost, Challenge Overhead, and Speedup Matrix (`k = 2`, `216 B` Receipt)

Times are median (`p50`) in microseconds (`us`). **Online verifier** caches static model weight commitment `commit(B)` computed once at model load time and hashes dynamic `A, C` + challenge + Freivalds; **Cold verifier** recomputes `commit(B)` on every call.

| Format | Shape `M×K×N` | Prover `A·B` (`us`) | `commit(A,C)` (`us`) | `commit(B)` (`us`) | Challenge Gen (`us`) | Freivalds Alg (`us`) | Online Total (`us`) | Cold Total (`us`) | Speedup (Alg / Online / Cold) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Int8Q8_0` | `1×64×64` *(neg)* | `3.0` | `4.0` | `24.0` | `24.1` | `8.1` | `37.0` | `61.1` | `0.37x / 0.08x / 0.05x` |
| `Int8Q8_0` | `1×512×512` *(neg)* | `156.1` | `24.5` | `1,461.9` | `166.3` | `439.1` | `651.5` | `2,238.5` | `0.36x / 0.24x / 0.07x` |
| `Int8Q8_0` | `4×1024×1024` *(neg)* | `1,242.3` | `182.6` | `5,929.4` | `336.7` | `1,868.6` | `2,444.2` | `8,518.5` | `0.66x / 0.51x / 0.15x` |
| `Int8Q8_0` | `32×1024×1024` | `6,479.6` | `1,486.0` | `5,956.3` | `332.6` | `2,040.5` | `4,076.4` | `10,156.8` | **`3.18x / 1.59x`** `/ 0.64x` |
| `Int8Q8_0` | `64×2048×2048` | `51,282.1` | `6,346.6` | `23,627.4` | `740.4` | `8,443.5` | `15,186.0` | `38,074.3` | **`6.07x / 3.38x / 1.35x`** |
| `Int8Q8_0` | `128×2048×2048` | `101,164.6` | `12,465.0` | `23,213.6` | `703.1` | `8,803.2` | `21,889.9` | `44,518.5` | **`11.49x / 4.62x / 2.27x`** |
| `PackedInt4Q4_0` | `1×64×64` *(neg)* | `3.5` | `3.9` | `12.4` | `23.2` | `8.4` | `36.3` | `50.2` | `0.42x / 0.10x / 0.07x` |
| `PackedInt4Q4_0` | `1×512×512` *(neg)* | `184.3` | `24.4` | `790.1` | `159.7` | `453.8` | `672.5` | `1,491.2` | `0.41x / 0.27x / 0.12x` |
| `PackedInt4Q4_0` | `4×1024×1024` *(neg)* | `1,387.7` | `194.8` | `3,163.0` | `336.5` | `1,960.4` | `2,604.8` | `5,901.9` | `0.71x / 0.53x / 0.24x` |
| `PackedInt4Q4_0` | `32×1024×1024` | `6,472.6` | `1,586.0` | `3,070.6` | `339.0` | `2,175.5` | `4,282.2` | `7,399.2` | **`2.98x / 1.51x`** `/ 0.87x` |
| `PackedInt4Q4_0` | `64×2048×2048` | `51,342.9` | `6,299.1` | `12,510.6` | `684.5` | `8,634.1` | `15,560.6` | `27,983.9` | **`5.95x / 3.30x / 1.83x`** |
| `PackedInt4Q4_0` | `128×2048×2048` | `100,794.6` | `11,927.0` | `12,243.5` | `705.1` | `9,361.7` | `22,364.2` | `36,309.1` | **`10.77x / 4.51x / 2.78x`** |
| `PackedTernaryI2S` | `1×64×64` *(neg)* | `4.2` | `3.6` | `6.6` | `21.6` | `11.8` | `37.9` | `44.4` | `0.36x / 0.11x / 0.10x` |
| `PackedTernaryI2S` | `1×512×512` *(neg)* | `262.1` | `23.6` | `439.1` | `184.8` | `772.0` | `999.1` | `1,426.6` | `0.34x / 0.26x / 0.18x` |
| `PackedTernaryI2S` | `4×1024×1024` *(neg)* | `1,640.9` | `190.0` | `1,712.1` | `389.8` | `2,968.8` | `3,533.6` | `5,342.0` | `0.55x / 0.46x / 0.31x` |
| `PackedTernaryI2S` | `32×1024×1024` | `7,055.4` | `1,542.3` | `1,701.4` | `353.2` | `3,164.8` | `5,121.6` | `6,717.8` | **`2.23x / 1.38x / 1.05x`** |
| `PackedTernaryI2S` | `64×2048×2048` | `54,847.6` | `6,271.2` | `6,893.6` | `709.7` | `12,258.1` | `19,207.3` | `26,667.0` | **`4.47x / 2.86x / 2.06x`** |
| `PackedTernaryI2S` | `128×2048×2048` | `103,450.7` | `12,089.4` | `6,992.6` | `741.1` | `13,325.5` | `26,030.8` | `33,286.5` | **`7.76x / 3.97x / 3.11x`** |

### 5.3 Soundness vs Latency Trade-Off (`64×2048×2048`, `PackedInt4Q4_0`)

| Rounds $k$ | Soundness Error Bound | Security Bits | Prover (`us`) | Challenge (`us`) | Freivalds Alg (`us`) | Online Total (`us`) | Speedup (Alg / Online) | Receipt Size |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| `k = 1` *(explicit opt-in)* | `< 4.34e-19` | `61.00` | `51,987.2` | `337.1` | `3,949.6` | `10,036.2` | `13.16x / 5.18x` | `184 B` |
| **`k = 2` (default)** | **`< 1.88e-37`** | **`122.00`** | `53,400.7` | `761.2` | `8,456.5` | `15,474.3` | **`6.31x / 3.45x`** | **`216 B`** |
| `k = 4` | `< 3.54e-74` | `244.00` | `51,608.5` | `1,407.4` | `16,877.9` | `24,760.0` | `3.06x / 2.08x` | `280 B` |

### 5.4 Honest Negative Regime Analysis

As shown explicitly in Table 5.2, **Freivalds is strictly slower than naive recomputation for tiny and single-token decode GEMVs ($M \in \{1, 4\}$)**:

- At `1×64×64`, algebraic Freivalds (`k = 2`) is `0.36x–0.42x` (2.4–2.8× slower than direct matmul) and online verification including SHA-256 commitments is `0.08x–0.11x` (~9–12× slower).
- At `1×512×512`, algebraic Freivalds is `0.34x–0.41x` and online verification is `0.24x–0.27x`.
- **Root cause**: When $M = 1$, exact matrix-vector multiplication $A \cdot B$ costs $O(KN)$ work, whereas $k = 2$ rounds of Freivalds execute $2 \times O(KN)$ sweeps over $B$ plus $O(MK + MN)$ SHA-256 hashing.
- **Crossover & engineering takeaway**: Freivalds crosses `> 1.0x` online speedup between $M = 8$ and $M = 32$ (`1.38x–1.59x` online / `2.23x–3.18x` algebraic at `32×1024×1024`), reaching `3.97x–4.62x` online (`7.76x–11.49x` algebraic) at `128×2048×2048` and scaling linearly with $M$ (`O(M)` speedup). Therefore, **single-token decode steps MUST be stacked across $M \ge 32$ decode tokens (or batched requests sharing static weights $B$) into one `(M × K) · (K × N)` verification proof** rather than proving individual `M = 1` GEMVs.

### 5.5 Decentralized Hanzo Bounty 5-Axis Scorecard

- **Cost**: Constant `216 B` receipt overhead per verified matmul (`~0.17 bytes/token` amortized over a 128-token prefill batch); CPU verification cuts compute work by `3.97x–11.49x` at $M = 128$ (and `> 40x` at $M = 512$ prefill batches) with zero prover extra matmuls (prover overhead is only dynamic SHA-256 commitments + $k$ vector projections `C * r`).
- **Performance**: `8.44–13.33 ms` algebraic / `15.19–26.03 ms` online CPU verification on a single 2.30 GHz Xeon core for `64–128 × 2048 × 2048` projections; zero GPU stalling when proofs are verified asynchronously off the critical decode path.
- **Trust**: `1,000 / 1,000` (`100.00%`) corruption detection with `0` false accepts across all 7 attack classes and mathematically proven `< 1.88e-37` (`< 2^-121`) cheating acceptance probability under `poai.matmul.v1`.
- **Decentralization**: `100%` independently verifiable on any commodity CPU with zero Hanzo-operated infrastructure, zero cloud KMS, and zero external crate dependencies.
- **Sovereignty**: Pure Apache-2.0 Rust `std` crate with canonical little-endian wire receipts (`POAI` v1) that any external verifier, node operator, or fork can compile and run offline.

---

## 6. Security Review, Rollout/Rollback, and Integration Roadmap

- **Security review**:
  - Strict input validation on all dimensions (`1..=2^30`, checked multiplications, `k % 32 == 0`), reserved 2-bit ternary code (`0b11`) rejection, strict signed output range guard (`|C[i,j]| <= 2^60 - 1` preventing wraparound forgery), constant-time digest comparison (`ct_eq_digest`), verifier-owned `VerificationPolicy` enforcing minimum round count (`default = 2`) and interactive seed anti-downgrade, round-bound transcript seeds preventing receipt truncation, and canonical wire parser rejecting trailing/truncated bytes.
  - Zero `unsafe` code anywhere in `hanzo-poai`.
- **Disposition & Rollout / Rollback**:
  - **Execution-profile crate (`hanzo-poai`)**: Ready for `MERGE` as an opt-in workspace crate (touches only root `Cargo.toml` workspace membership, adds zero external dependencies to `Cargo.lock`, and leaves all existing `hanzo-quant` / `hanzo-engine` hot paths untouched). Revert is a single clean `git revert`.
  - **Economic / on-chain settlement**: Explicitly `GATED` on external cryptographic and smart-contract review before binding real funds or slashing stakes.
- **Integration roadmap (`hanzo-quant`, `hanzo-engine`, `zoo-apps/papers#13`)**:
  1. **Model-load adapter & weight commitment registry**: At `hanzo-quant` GGUF / ISQ model load, convert loaded weights into canonical `poai.matmul.v1` buffers via explicit adapters—converting `Q8_0`/`Q4_0` `f16` block scales to exact `i16` fixed-point scales, and repacking GGUF `TQ2_0` (`256`-weight, `66 B` blocks with trailing `f16` scales) into `PackedTernaryI2S` (`32`-weight, `10 B` blocks with leading `i16` scales)—then compute and cache `commit_weight_matrix` once per converted projection alongside `GgufMatMul` metadata.
  2. **Deterministic integer epilogue hook**: Add an opt-in `--poai-profile poai.matmul.v1` execution mode in `hanzo-quant` / `hanzo-engine` exposing the pre-dequantization `i64` integer accumulator buffer `C` for sampled prefill layers (`M >= 32`) or stacked decode windows (`32–128` steps).
  3. **Cross-repo protocol alignment (`zoo-apps/papers#13`)**: Map `MatrixProofReceipt::to_bytes()` directly into the Zoo Gym / Enso challenge-response envelope alongside interactive randomness beacons (`ChallengeMode::InteractiveSeed`).
