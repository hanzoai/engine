# HIP-1350 Research Dossier: zkML and zkVM Proofs for Small Models and Decisions (`hanzo-zkml`)

## 1. Primary References, Coordination, and License Audit

- **Tracking issue**: `hanzoai/engine#68` (*"zkML and zkVM proofs for small models and decisions"*, `p1`, `bounty`, `research`, `decentralized`, `hip-1350`).
- **Sibling HIP-1350 work**:
  - `hanzoai/engine#66` (`hanzo-poai`): CPU-verifiable probabilistic Freivalds proofs for large GPU integer matrix products (`poai.matmul.v1`).
  - `hanzoai/engine#67`: Optimistic AI fraud-proof challenges and dispute games for large autoregressive inference.
- **Cryptographic & systems references**:
  - **EZKL** (Apache-2.0) — fixed-point quantized ML circuit compilation for compact models.
  - **RISC Zero** (Apache-2.0) — general-purpose deterministic zkVM with `ImageID`-bound receipts and public journals.
  - Srinath Setty, *"Spartan: Efficient and General-Purpose zkSNARKs Without Trusted Setup"*, CRYPTO 2020 (two-phase multilinear Sumcheck over R1CS).
  - Scott Ames, Carmit Hazay, Yuval Ishai, Muthuramakrishnan Venkitasubramaniam, *"Ligero: Lightweight Sublinear Arguments Without a Trusted Setup"*, ACM CCS 2017 (Reed-Solomon interleaved Merkle PCS).
  - Amos Fiat & Adi Shamir, *"How to Prove Yourself"*, CRYPTO 1986; NIST FIPS 180-4 SHA-256.
  - Goldilocks prime field $p = 2^{64} - 2^{32} + 1$ with quadratic extension $\mathbb{F}_{p^2} = \mathbb{F}_p[u]/(u^2 - 7)$ (Plonky2 / Plonky3 / RISC Zero ecosystem field).
- **License & dependency audit**:
  - `hanzo-zkml` is a **100% clean-room Apache-2.0** Rust crate (`#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`) using **`std` only** with **zero external dependencies** (`[dependencies]` and `[dev-dependencies]` are empty; `Cargo.lock` adds only the local `hanzo-zkml` package entry).
  - No proprietary model weights, restricted datasets, or third-party cryptographic source files are imported. All reference classifier, Kai policy, and rubric weights are deterministic open fixed-point test vectors defined directly in code.

---

## 2. Explicit Scope Boundary and Separation of Verification Tiers

> **Mandatory Scope Boundary (`hanzoai/engine#68`)**:
> `hanzo-zkml` targets **tractable, high-value ML and control-plane computations**: compact fixed-point classifiers, Kai routing/guardrail decision components, fixed deterministic preprocessing, and deterministic evaluation scorers. Full cryptographic proving of every autoregressive decode token of a 27B-parameter LLM is **explicitly rejected** as an interactive initial deliverable (see §6 for measured CPU extrapolation).

In strict compliance with HIP-1350 Rule 3, `hanzo-zkml` separates the three verification mechanisms via the typed [`TrustTier`](../../../hanzo-zkml/src/lib.rs) enum and refuses to conflate them:

| Trust Tier | `hanzo-zkml` Role | What Remains Explicitly Out of Scope |
|---|---|---|
| **1. Cryptographic Proof** (`TrustTier::CryptographicProof`) | **Provided by `hanzo-zkml`.** Verifies exact R1CS arithmetic satisfiability (Path A) and deterministic register zkVM execution traces (Path B) over Goldilocks $\mathbb{F}_p / \mathbb{F}_{p^2}$ via Spartan Sumcheck + Ligero RS-Merkle commitments. | Does not attest to semantic LLM helpfulness, training provenance, or per-token 27B autoregressive decode. |
| **2. Hardware Attestation** (`TrustTier::HardwareAttestation`) | **Not assumed or produced.** Any commodity CPU can verify every proof statelessly without TEEs, SGX/TDX, Nitro, or vendor keys. `HanzoProofEnvelope::verify_binding` rejects envelopes claiming `HardwareAttestation`. | Physical GPU/CPU identity, enclave measurement, and confidential weight secrecy against a malicious host. |
| **3. Economic / Optimistic Challenge** (`TrustTier::EconomicChallenge`) | **Not implemented in this crate.** `HanzoProofEnvelope v1` provides canonical VFS-addressed proof receipts that external optimistic settlement contracts or dispute games can consume as validity/fraud proofs. | Staking, slashing parameters, liveness windows, and on-chain settlement contracts (require independent security audit before holding real funds). |

---

## 3. Protocol Specification (`hanzo.zkml.v1`)

### 3.1 Field Arithmetic & Transcript (`field.rs`, `sha256.rs`)

- **Base field**: Goldilocks prime $p = 2^{64} - 2^{32} + 1 = \text{0xFFFF\_FFFF\_0000\_0001}$ with 2-adicity $32$ and primitive $2^{32}$-th root of unity $\omega_{32} = 7^{2^{32}-1} \bmod p = 1{,}753{,}635{,}133{,}440{,}165{,}772$.
- **Wraparound-free signed bijection**: Signed `i64` inputs, outputs, weights, and intermediate registers are checked against $|x| \le (p - 1)/2 = 9{,}223{,}372{,}034{,}707{,}292{,}160$; values outside this range are rejected with `ZkmlError::FieldRangeExceeded`.
- **Challenge extension field**: $\mathbb{F}_{p^2} = \mathbb{F}_p[u]/(u^2 - 7)$ (provably irreducible because $7$ is a primitive generator of $\mathbb{F}_p^*$, verified in unit tests via $7^{(p-1)/2} \equiv -1 \pmod p$). Field size $p^2 \approx 2^{128}$ yields $\sim 128$-bit Schwartz-Zippel / Sumcheck challenge entropy per draw.
- **Fiat-Shamir transcript**: Length-prefixed, label-delimited, domain-separated NIST FIPS 180-4 SHA-256 sponge with exact rejection sampling for uniform $\mathbb{F}_p$ / $\mathbb{F}_{p^2}$ challenges and distinct column query sampling.

### 3.2 Path A — Model-Specific R1CS Circuits (`circuit.rs`, `merkle.rs`)

Each compact model compiles via `R1csBuilder` into an immutable `CircuitSpec` with wire vector $z \in \mathbb{F}_p^N$ ($N = 2^\nu \ge 8$) partitioned by the top two address bits:

- **`00` ($0 \dots N/4 - 1$)**: Public wires `[1, public_inputs..., public_outputs...]` (evaluated directly by the verifier).
- **`01` ($N/4 \dots N/2 - 1$)**: Committed model weights (committed offline into `vk.weights_commitment: LigeroCommitment` and VFS digest `vk.weights_sha256`).
- **`1*` ($N/2 \dots N - 1$)**: Private witness wires (committed per proof into `proof.witness_commitment: LigeroCommitment`).

Because the three regions occupy disjoint power-of-two subcubes, the multilinear extension at random inner-sumcheck point $r_y \in \mathbb{F}_{p^2}^\nu$ decomposes cleanly as:

$$\widetilde{Z}(r_y) = (1 - r_{\nu-1})(1 - r_{\nu-2})\widetilde{Z}_{\text{pub}}(r_{0..\nu-2}) + (1 - r_{\nu-1})r_{\nu-2}\widetilde{Z}_w(r_{0..\nu-2}) + r_{\nu-1}\widetilde{Z}_h(r_{0..\nu-1})$$

Proving and stateless verification execute:
1. **Outer degree-3 Spartan Sumcheck** ($\mu = \log_2 M$ rounds over $\mathbb{F}_{p^2}$) verifying $\sum_{x \in \{0,1\}^\mu} \widetilde{\text{eq}}(\tau, x)(\widetilde{A}_z(x)\widetilde{B}_z(x) - \widetilde{C}_z(x)) = 0$.
2. **Inner degree-2 Spartan Sumcheck** ($\nu = \log_2 N$ rounds over $\mathbb{F}_{p^2}$) reducing $(\lambda_A v_A + \lambda_B v_B + \lambda_C v_C)$ to $M_{r_x}(r_y) \cdot \widetilde{Z}(r_y)$, where the verifier computes $M_{r_x}(r_y)$ directly from sparse public R1CS matrices in $O(\text{nnz})$.
3. **Ligero RS-Merkle PCS Openings** (Reed-Solomon rate $\rho = 1/8$, blowup $B = 8$ via radix-2 Goldilocks NTT, $t = 64$ Fiat-Shamir distinct Merkle-authenticated column openings, both proximity check and rank-1 tensor MLE evaluation check) against `vk.weights_commitment` and `proof.witness_commitment`.
4. **Exact non-linear ML operators in R1CS**: 24-bit boolean bit decompositions enforce exact signed ReLU ($y = \max(0, x)$), exact signed comparisons (`a >= b`), and exact one-hot argmax with zero polynomial approximation error.

**Three compiled open circuits**:
- `CompactClassifierV1` (`hanzo.compact_classifier.mlp_8x8x4@1.0.0`): 8-feature fixed preprocessing (`scale * (x - mean)`) + `8 -> 8` exact 24-bit ReLU hidden layer + `8 -> 4` output logits + exact 4-class argmax (`447` R1CS constraints, `124` committed weights).
- `KaiPolicyGateV1` (`hanzo.kai.policy_gate_6f@1.0.0`): 6-feature weighted Kai telemetry score + dual threshold comparisons emitting deterministic 3-way routing (`0 = ROUTE_LOCAL_FAST`, `1 = ROUTE_VERIFIED_ESCROW`, `2 = REJECT_POLICY_VIOLATION`) (`60` R1CS constraints).
- `DeterministicEvalScorerV1` (`hanzo.eval.deterministic_scorer_8r@1.0.0`): 8-dimension rubric weighted sum + quadratic L2 calibration penalty $\sum (s_i - t_i)^2$ + threshold pass/fail gate (`54` R1CS constraints).

### 3.3 Path B — General Deterministic Register zkVM (`zkvm.rs`)

A clean 8-register (`R0..R7`) deterministic zkVM supporting 8 fixed instructions:
`LOAD` (immediate / journal input / committed weight), `ADD`, `SUB`, `MUL`, `SRA` (exact Euclidean floor division $x = 2^s q + r$ with bit-decomposed $0 \le r < 2^s$, supporting negative integers exactly), `CMP_GT` (exact 24-bit signed greater-than), `SELECT` (boolean-gated ternary mux), and `HALT`.

- **`ImageID`** (`[u8; 32]`): `SHA256("hanzo.zkml.zkvm.image_id.v1" || vm_version || domain_tag || program_digest || weights_sha256 || weights_root || max_cycles)`.
- **Dual-bound execution proof**:
  1. Full cycle-by-cycle algebraic AIR/R1CS transition, register-preservation, inter-cycle continuity, input/weight load, `SRA`/`CMP_GT`/`SELECT`, and `HALT` journal-output proof via the same Spartan + Ligero backend.
  2. SHA-256 Merkle commitment (`trace_merkle_root`) over every serialized execution trace row + 8 Fiat-Shamir spot-opened trace rows with Merkle authentication and local transition re-checks.
- **Three compiled open zkVM programs** matching Path A head-to-head:
  - `hanzo.zkvm.compact_classifier_linear_8x4` (`215` cycles, exercises all 8 opcodes)
  - `hanzo.zkvm.kai_policy_gate_6f` (`31` cycles; verified bit-for-bit identical output to Path A `KaiPolicyGateV1`)
  - `hanzo.zkvm.deterministic_eval_scorer_8r` (`73` cycles)

### 3.4 Versioned Hanzo Integration Interface (`envelope.rs`)

`HanzoProofEnvelope` (`schema_version = "hanzo.proof.envelope.v1"`) binds VFS content-addressed artifact URIs (`vfs://sha256/<circuit_digest_or_image_id>`, `vfs://sha256/<weights_sha256>`), optional Enso route ID and Visor job ID metadata, explicit `trust_tier = "cryptographic_proof"`, `hanzo_service_required = false`, and SHA-256 digests of public inputs, public outputs, proof payload, and the envelope itself. Pure-Rust canonical JSON serialization/parsing requires zero external dependencies and zero Hanzo-operated verification infrastructure.

---

## 4. Measured Benchmarks (Path A vs Path B)

Measured with `zkml-bench --seed 1350 --trials 15 --output-dir docs/research/hip-1350-zkml-zkvm-proofs` (`--release --offline`, 2 warmup + 15 measured trials per workload, Ligero `blowup = 8`, `queries = 64`, `43.40`-bit conservative proximity soundness + `128.00`-bit $\mathbb{F}_{p^2}$ Sumcheck field soundness).

- **Deterministic receipt**: [`results.json`](./results.json)
- **Receipt SHA-256**: `e91aff8749c0e03ef67e4ffafc5ea1c6711c86e2fefd7508912398e0b552586b`

| Workload | Backend | Constraints | Cycles | Prove p50 (ms) | Prove p95 (ms) | Verify p50 (ms) | Verify p95 (ms) | Proof Size (B) | Prove / s | Verify / s | Cost (\$/M tok) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `compact_classifier_mlp_8x8x4` | Model-Specific Circuit | 447 | — | **1.575** | 1.597 | **2.045** | 2.100 | 54,125 | 634.35 | 487.85 | \$0.000079 |
| `kai_policy_gate_6f` | Model-Specific Circuit | 60 | — | **0.563** | 0.649 | **1.338** | 1.655 | 36,371 | 1,761.34 | 724.80 | \$0.000042 |
| `deterministic_eval_scorer_8r` | Model-Specific Circuit | 54 | — | **0.568** | 0.631 | **1.365** | 1.476 | 36,381 | 1,573.42 | 678.74 | \$0.000046 |
| `zkvm_compact_classifier_linear_8x4` | General zkVM | 1,826 | 215 | **6.495** | 11.382 | **4.407** | 8.040 | 79,922 | 129.80 | 190.42 | \$0.000281 |
| `zkvm_kai_policy_gate_6f` | General zkVM | 308 | 31 | **1.767** | 3.083 | **2.600** | 3.848 | 56,903 | 543.76 | 390.61 | \$0.000095 |
| `zkvm_deterministic_eval_scorer_8r` | General zkVM | 624 | 73 | **3.289** | 4.000 | **3.000** | 4.000 | 68,289 | 300.62 | 335.68 | \$0.000137 |

**Empirical Path A vs Path B Tradeoff (Head-to-Head on `kai_policy_gate_6f`)**:
- Model-specific circuit uses **5.1x fewer constraints** (`60` vs `308`), proves **3.1x faster** (`0.563 ms` vs `1.767 ms` p50), verifies **1.9x faster** (`1.338 ms` vs `2.600 ms` p50), and produces **1.56x smaller proofs** (`36.4 KB` vs `56.9 KB`).
- General zkVM pays honest register-file and inter-cycle continuity overhead in exchange for universal `ImageID`-bound programmability without compiling a new custom circuit schema per policy update.

### Adversarial Rejection Matrix (`16 / 16 = 100.0%` Rejected)

| # | Attack Vector | Path | Result | Verifier Rejection Error |
|---:|---|---|---|---|
| 1 | `circuit_wrong_weights_vk` | Circuit | **REJECTED** | `weights commitment mismatch` |
| 2 | `circuit_wrong_output_logit` | Circuit | **REJECTED** | `sumcheck verification failed at stage 'outer_round_sum'` |
| 3 | `circuit_wrong_output_class` | Circuit | **REJECTED** | `sumcheck verification failed at stage 'outer_round_sum'` |
| 4 | `circuit_wrong_version` | Circuit | **REJECTED** | `circuit version mismatch: expected '...1.0.0', got '...2.0.0'` |
| 5 | `circuit_wrong_digest` | Circuit | **REJECTED** | `circuit digest mismatch` |
| 6 | `circuit_forged_intermediate_witness` | Circuit | **REJECTED** | `sumcheck verification failed at stage 'outer_round_sum'` |
| 7 | `circuit_wrong_preprocessing_input` | Circuit | **REJECTED** | `sumcheck verification failed at stage 'outer_round_sum'` |
| 8 | `circuit_cross_domain_replay` | Circuit | **REJECTED** | `domain separation mismatch` |
| 9 | `zkvm_wrong_image_id` | zkVM | **REJECTED** | `zkVM ImageID mismatch` |
| 10 | `zkvm_wrong_vm_version` | zkVM | **REJECTED** | `zkVM version mismatch: expected '1.0.0', got '2.0.0'` |
| 11 | `zkvm_wrong_weights` | zkVM | **REJECTED** | `zkVM ImageID mismatch` |
| 12 | `zkvm_wrong_journal_output` | zkVM | **REJECTED** | `sumcheck verification failed at stage 'outer_round_sum'` |
| 13 | `zkvm_wrong_journal_input` | zkVM | **REJECTED** | `sumcheck verification failed at stage 'outer_round_sum'` |
| 14 | `zkvm_corrupted_trace_witness` | zkVM | **REJECTED** | `sumcheck verification failed at stage 'outer_round_sum'` |
| 15 | `zkvm_tampered_merkle_trace_row` | zkVM | **REJECTED** | `zkVM trace verification failed: Merkle proof failed at cycle 160` |
| 16 | `zkvm_cycle_bound_exceeded` | zkVM | **REJECTED** | `zkVM cycle bound exceeded: 224 > 223` |

---

## 5. HIP-1350 Five-Axis Evaluation

| Axis | Measured Result | Notes / Declared Model |
|---|---|---|
| **Cost** (`$/M useful tokens`) | **\$0.000042 – \$0.000281 / M tokens** (tractable targets) | Commodity CPU rate \$0.04 / vCPU-hr, 1 proof amortized per 512-token request/evaluation (includes both proving and verification CPU time). |
| **Performance** | TTFT overhead **0.56 – 6.50 ms** (p50 prove), verify **1.34 – 4.41 ms** (p50), throughput **130 – 1,761 proofs/s**, completion rate **100.0%** | Sub-10 ms overhead is negligible relative to LLM TTFT (`50 – 500 ms`), making real-time Kai gating and compact classification directly deployable on CPU. |
| **Trust** | **100.0% (`16/16`)** adversarial rejection under declared threat model | Covers wrong weights, wrong outputs, wrong argmax, wrong version/digest/`ImageID`, violated internal ReLU/trace constraints, wrong inputs, cycle-bound forgery, and cross-domain replay. |
| **Decentralization** | **100.0%** Hanzo-independent | Stateless verifiers require only `(vk / ImageID, public_inputs, public_outputs, proof)`; zero mandatory Hanzo RPC, KMS, or attestation authority. |
| **Sovereignty** | **100.0%** exportable, verifiable, forkable | Apache-2.0 zero-dependency crate + open canonical `HanzoProofEnvelope v1` JSON with VFS `vfs://sha256/...` CIDs. |

---

## 6. Explicit Negative Regime: Why Per-Token Full zkML on 27B Decode Is Rejected

Calibrating directly from the measured `compact_classifier_mlp_8x8x4` CPU proving rate (`3,526.65 ns / R1CS constraint` mean on this CPU) against the **lower bound** of $5.4 \times 10^{10}$ MACs per single dense 27B autoregressive decode token (ignoring nonlinear attention softmax, RoPE, RMSNorm, and KV-cache commitments):

- **Extrapolated CPU prove time per single 27B token**: `190,439.32 s` (**52.90 hours / token**).
- **Extrapolated CPU proving cost**: **~\$2,115,992.39 / M tokens** at \$0.04 / vCPU-hr.
- **Verdict**: **`REJECTED_FOR_INTERACTIVE_27B_DECODE`**.

**Recommended HIP-1350 Layered Verification Stack**:
1. **`hanzo-zkml` (this crate)**: Full cryptographic proofs for compact classifiers, Kai routing/guardrail gates, fixed preprocessing, and deterministic evaluators (`< 7 ms`, `< $0.0003 / M tokens`).
2. **`hanzo-poai` (`hanzoai/engine#66`)**: Fast CPU-verifiable probabilistic Freivalds matrix-product proofs for large quantized GPU GEMM layers.
3. **Optimistic fraud proofs (`hanzoai/engine#67`)**: Economically bonded challenge games that invoke targeted `hanzo-zkml` / `hanzo-poai` proofs only on disputed checkpoints.

---

## 7. What Remains Unproven (Explicit Engineering & Cryptographic Disclosures)

In accordance with HIP-1350's strict anti-overclaiming discipline, the following properties are **NOT** claimed by `hanzo-zkml v1`:

1. **128-bit succinct PCS & full statistical zero-knowledge witness hiding**: Default Ligero parameters (`blowup = 8`, `queries = 64`) provide `43.40`-bit conservative proximity soundness (alongside `~128`-bit $\mathbb{F}_{p^2}$ Sumcheck soundness) and $O(\sqrt{N})$ proof sizes (`36 – 80 KB`) without degree-$t$ random blinding rows for witness secrecy. Production deployments requiring full 128-bit cryptographic soundness, strict zero-knowledge witness hiding, and constant/polylog succinct on-chain proofs should pair this interface with audited FRI/STARK recursion (e.g., Plonky3 / RISC Zero) or Halo2/KZG backends.
2. **Dynamic control-flow zkVM & memory paging**: Path B proves deterministic straight-line / predicated `ZkVmProgram` bytecode (`LOAD/ADD/SUB/MUL/SRA/CMP_GT/SELECT/HALT`) with fixed cycle schedules; general cyclic RISC-V (`rv32im`) branching, RAM read/write permutation arguments (offline memory checking), and continuation folding are future milestones.
3. **GPU proving acceleration & side-channel hardening**: The pure-Rust CPU prover is unoptimized for AVX-512/CUDA/Metal NTTs and is not hardened against timing/cache side channels on untrusted multi-tenant hosts.
4. **Independent external cryptographic audit**: Must be completed before securing real financial escrow or slashing contracts.

---

## 8. Reproduction Commands

```bash
cargo fmt -p hanzo-zkml -- --check
cargo clippy -p hanzo-zkml --all-targets --offline -- -D warnings
cargo test -p hanzo-zkml --offline --release
cargo run -p hanzo-zkml --release --offline --bin zkml-bench -- \
    --seed 1350 --trials 15 --output-dir docs/research/hip-1350-zkml-zkvm-proofs
sha256sum -c docs/research/hip-1350-zkml-zkvm-proofs/results.json.sha256
```
