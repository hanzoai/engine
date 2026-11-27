# HIP-1350 Research Report — Optimistic AI Execution & Interactive $O(\log_2 T)$ Bisection Fraud-Proof Games (`hanzo-opai`, `hanzoai/engine#67`)

**Status:** Implemented & verified clean-room Apache-2.0 workspace crate (`hanzo-opai`, 0 external dependencies)  
**Target Issue:** [`hanzoai/engine#67`](https://github.com/hanzoai/engine/issues/67) (`[p1, decentralized, proof-of-ai, hip-1350]`)  
**Execution Profile:** `opai.det.zen-kai.v1` (`profile_digest = b873434bc9b87caa2b3967e90aea8ec1480b8004cbd03c826e3d339805780f41`)  
**Benchmark Artifact:** [`results.json`](./results.json) (deterministic `--seed 1350`, pinned to versioned repository inputs)

---

## 1. Executive Summary

`hanzo-opai` implements an end-to-end optimistic AI execution and interactive binary-search bisection fraud-proof engine aligned with HIP-1350 (`hanzo-detprofile` / `hanzo-poai` / `hanzo-provider-daemon` fixed-point deterministic semantics) with **zero external crate dependencies**, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`, and Apache-2.0 clean-room licensing.

Instead of re-executing 100% of inference steps on every claim (`full_reexecution_every_claim`) or trusting providers blindly (`unverified_optimistic_control`), `hanzo-opai`:
1. Binds every bonded `Claim` to an explicit fixed-point `ExecutionProfile`, `QuantizedModel` digest, `InputCommitment`, `start_root`, `end_root`, and `trace_merkle_root` over all `T + 1` canonical SHA-256 `StateRoot` boundaries (`0..=T`).
2. Allows unchallenged honest claims to settle optimistically after `challenge_window_secs` with **0 ML judge steps executed** (`610.03×` lower verification compute at 0% fraud; `44.94×` lower verification compute at 1% fraud).
3. Allows any permissionless `Challenger` to post a counter-bond within `challenge_window_secs`, narrow the first disagreement over `[0, T]` in at most $\lceil \log_2 T \rceil$ interactive bisection rounds, and invoke the deterministic **$O(1)$ single-step judge** (`verify_single_step_with_ctx`) on ONLY step `lo -> lo + 1` (`15.99–19.90 µs` constant judge time from `T = 20` to `T = 1,040`, achieving **729.76×** single-step judge speedup at `T = 1,040`).
4. Enforces strict per-turn liveness timeouts (`turn_deadline_secs`), anti-griefing bond escalation (`1.5×` per concurrent slot), burn-backed slashing (preventing profitable self-challenging), concurrent watchtower tournament refunds, and lossless sovereign binary wire (`OPAI` v1, ~2.2 KB) + JSON transcript export with independent third-party replay (**0 Hanzo control-plane calls**).

---

## 2. Threat Model, Guarantees, and Explicit Scope Boundaries

| Layer | What `hanzo-opai` Provides | What Remains Explicitly Out of Scope |
|---|---|---|
| **Deterministic step-trace execution** | Pure integer & dyadic fixed-point quantized transformer/MLP VM (`12` atomic op kinds: `EmbedToken`, `RmsNorm`, `AttentionMixProj`, `ResidualAdd`, `FfnUpProj`, `SiLU_LUT`, `FfnDownProj`, `FinalRmsNorm`, `LmHeadLogits`, `ArgMaxEmit`) with `i128` accumulators, bitwise `isqrt_u128`, 257-knot symmetric `SIGMOID_LUT_Q15`, and explicit floating-point exclusion gate. | Any floating-point kernel (`FP16`/`BF16`/`FP32`/`FP8`) or unprofiled stochastic sampling. |
| **Interactive fraud proofs** | $O(\log_2 T)$ binary-search bisection + $O(1)$ single-step judge (`verify_single_step`, `verify_single_step_with_ctx`) + 0-round `start_root` forgery check + invalid pre-state witness rejection. | Succinct non-interactive ZK proofs and Freivalds matrix proofs (`hanzo-poai`, `#66`) are complementary verification layers. |
| **Economic & liveness security** | Bonded claims, counter-bond escalation (`griefing_escalation_bps`), strict monotonic timestamps, per-turn timeout forfeits (`DefenderTimeoutForfeit` / `ChallengerTimeoutForfeit`), burn fraction `> 0` (`challenger_reward_bps < 10_000`) preventing self-challenge arbitrage, and exact global bond conservation across concurrent multi-challenger tournaments. | On-chain smart contract deployment, key custody, and cross-chain bridge settlement require independent external security audits before holding real funds. |
| **Sovereignty & decentralization** | Zero external crate dependencies, zero TEE/enclave assumptions, permissionless challengers, canonical `OPAI` v1 wire & JSON transcripts, and `verify_transcript_sovereign` requiring **0 Hanzo control-plane calls**. | P2P gossip networking and Mempool/L1 censorship resistance belong to the transport/chain layer. |

---

## 3. Protocol Architecture (`hanzo-opai`)

- [`hanzo-opai/src/crypto.rs`](../../../hanzo-opai/src/crypto.rs): Clean-room FIPS 180-4 `Sha256`, domain separation tags, constant-time `ct_eq_digest`, hex codecs, `TraceMerkleTree` + `MerkleProof`, and deterministic `SplitMix64`.
- [`hanzo-opai/src/profile.rs`](../../../hanzo-opai/src/profile.rs): `ExecutionProfile` (`opai.det.zen-kai.v1`), `NumericDomain` floating-point exclusion gate, `DyadicScale`, exact overflow-free `rescale_i128` / `round_div_i128` / `isqrt_u128`, symmetric 257-knot `SIGMOID_LUT_Q15`, `fixed_point_silu_q12`, and `InputCommitment`.
- [`hanzo-opai/src/vm.rs`](../../../hanzo-opai/src/vm.rs): `QuantizedModel`, `StepOp` (12 atomic op categories), `VmState::state_root`, `execute_single_step`, `execute_trace`, `execute_trace_with_fault`, `JudgeContext`, `verify_single_step_with_ctx`, and `verify_single_step`.
- [`hanzo-opai/src/game.rs`](../../../hanzo-opai/src/game.rs): `GameConfig`, `Claim`, `DisputeGame` (`open`, `assert_invalid_start_root`, `submit_defender_midpoint`, `submit_challenger_branch`, `submit_single_step_proof`, `claim_timeout_forfeit`), `BondSettlement`, `run_dispute_to_completion`, and multi-tenant/multi-challenger `OptimisticCoordinator`.
- [`hanzo-opai/src/transcript.rs`](../../../hanzo-opai/src/transcript.rs): `DisputeTranscript` canonical `OPAI` v1 binary wire codec (anti-DoS length bounds + strict EOF), clean-room JSON serializer/parser, and `verify_transcript_sovereign`.
- [`hanzo-opai/src/telemetry.rs`](../../../hanzo-opai/src/telemetry.rs): HIP-1350 5-axis telemetry data types and `ns_to_usd_per_1m_tokens` (`$0.05 / vCPU-hour` reference rate).
- [`hanzo-opai/src/bin/opai-bench.rs`](../../../hanzo-opai/src/bin/opai-bench.rs): Reproducible benchmark runner generating [`results.json`](./results.json).

---

## 4. Pinned Repository Inputs & Reproducible Commands

All benchmark prompts and deterministic model seeds are cryptographically pinned to real versioned files in `hanzoai/engine`:

| Versioned Repository File | Size (Bytes) | SHA-256 Digest |
|---|---:|---|
| `calibration_data/calibration_datav3_small.txt` | `133,837` | `aabd956dba34ff31f40e78b9de4e234fdd4865f90ba231e8e3deec8dfb22cbc0` |
| `chat_templates/llama3.json` | `420` | `67d1d402738727aa3059d28590fa3488c44740a358be4b5c2135c60f58ec9d7a` |
| `orderings/xlora-paper-ordering.json` | `6,485` | `8d0650ea231eeade0b97e5c026411365ff9feed41b2b07ecf10853c2a399043a` |

### Exact Reproduction Commands

```bash
cargo fmt -p hanzo-opai -- --check
cargo clippy -p hanzo-opai --all-targets --offline -- -D warnings
cargo test -p hanzo-opai --offline --release
cargo run --offline --release -p hanzo-opai --bin opai-bench -- \
  --seed 1350 \
  --out-json docs/research/hip-1350-optimistic-fraud-proofs/results.json
```

---

## 5. Measured HIP-1350 5-Axis Benchmark Results (`--seed 1350`)

Standard workload: `N = 100` claims per sweep, `QuantizedModel(vocab=64, D=32, F=64, L=4)`, `max_new_tokens = 4` (`T = 144` atomic steps/claim, $\lceil \log_2 144 \rceil = 8$ max bisection rounds), `challenge_window_secs = 600`, `turn_deadline_secs = 60`, reference compute pricing `$0.05 / vCPU-hour`, modeled interactive WAN RTT `80 ms/msg`.

### 5.1 Matched 3-Regime Comparison Across Fraud & Griefing Sweeps

| Scenario | Regime | Axis 1: Total Cost (`$/1M tok`) | Axis 1: Verify Ratio | Axis 2: Throughput (`tok/s`) | Axis 2: TTFT (`ms`) | Axis 2: Verify Speedup vs Full | Axis 2: Rounds (`mean/max`) | Axis 2: Judge (`µs`) | Axis 3: Exact Fraud Pinpoint | Axis 3: Griefers Slashed | Axis 4: Control-Plane Calls | Axis 5: Wire / JSON Transcript (`B`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fraud_0pct_grief_0pct` | `unverified_optimistic_control` | `0.0072` | `0.0000` | `1917.69` | `0.5215` | `∞` | `0.00 / 0` | `0.00` | `100.00%` *(N/A: 0 faults)* | `0.00%` | `0` | `0 / 0` |
| `fraud_0pct_grief_0pct` | `full_reexecution_every_claim` | `0.0140` | `0.9271` | `995.14` | `0.5215` | `1.00×` | `0.00 / 0` | `1933.70` | `100.00%` | `100.00%` | `0` | `0 / 0` |
| `fraud_0pct_grief_0pct` | **`hanzo_opai_bisection_game`** | **`0.0073`** | **`0.0015`** | **`1914.78`** | **`0.5215`** | **`610.03×`** | `0.00 / 8` | `0.00` | **`100.00%`** | **`100.00%`** | **`0`** | `0 / 0` |
| `fraud_1pct_grief_0pct` | `unverified_optimistic_control` | `0.0066` | `0.0000` | `2098.53` | `0.4765` | `∞` | `0.00 / 0` | `0.00` | `0.00%` | `0.00%` | `0` | `0 / 0` |
| `fraud_1pct_grief_0pct` | `full_reexecution_every_claim` | `0.0130` | `0.9676` | `1066.52` | `0.4765` | `1.00×` | `0.00 / 0` | `1844.42` | `100.00%` | `100.00%` | `0` | `0 / 0` |
| `fraud_1pct_grief_0pct` | **`hanzo_opai_bisection_game`** | **`0.0068`** | **`0.0215`** | **`2054.30`** | **`0.4765`** | **`44.94×`** | **`7.00 / 8`** | **`15.35`** | **`100.00%`** | **`100.00%`** | **`0`** | **`2187 / 4794`** |
| `fraud_10pct_grief_0pct` | `unverified_optimistic_control` | `0.0065` | `0.0000` | `2128.18` | `0.4699` | `∞` | `0.00 / 0` | `0.00` | `0.00%` | `0.00%` | `0` | `0 / 0` |
| `fraud_10pct_grief_0pct` | `full_reexecution_every_claim` | `0.0130` | `0.9901` | `1069.37` | `0.4699` | `1.00×` | `0.00 / 0` | `1860.98` | `100.00%` | `100.00%` | `0` | `0 / 0` |
| `fraud_10pct_grief_0pct` | **`hanzo_opai_bisection_game`** | **`0.0079`** | **`0.2043`** | **`1767.18`** | **`0.4699`** | **`4.85×`** | **`7.40 / 8`** | **`16.07`** | **`100.00%`** | **`100.00%`** | **`0`** | **`2221 / 4827`** |
| `fraud_50pct_grief_0pct` | `unverified_optimistic_control` | `0.0067` | `0.0000` | `2073.17` | `0.4824` | `∞` | `0.00 / 0` | `0.00` | `0.00%` | `0.00%` | `0` | `0 / 0` |
| `fraud_50pct_grief_0pct` | `full_reexecution_every_claim` | `0.0132` | `0.9637` | `1055.76` | `0.4824` | `1.00×` | `0.00 / 0` | `1859.32` | `100.00%` | `100.00%` | `0` | `0 / 0` |
| `fraud_50pct_grief_0pct` | **`hanzo_opai_bisection_game`** | **`0.0135`** | **`1.0094`** | **`1031.71`** | **`0.4824`** | **`0.95×`** *(crossover)* | **`7.18 / 8`** | **`16.70`** | **`100.00%`** | **`100.00%`** | **`0`** | **`2203 / 4806`** |
| `fraud_1pct_grief_5pct` | `full_reexecution_every_claim` | `0.0133` | `0.9441` | `1047.57` | `0.4910` | `1.00×` | `0.00 / 0` | `1854.33` | `100.00%` | `100.00%` | `0` | `0 / 0` |
| `fraud_1pct_grief_5pct` | **`hanzo_opai_bisection_game`** | **`0.0073`** | **`0.0719`** | **`1900.01`** | **`0.4910`** | **`13.13×`** | **`7.50 / 8`** | **`15.61`** | **`100.00%`** | **`100.00%`** | **`0`** | **`2227 / 4702`** |
| `fraud_1pct_grief_25pct` | `full_reexecution_every_claim` | `0.0138` | `0.9275` | `1005.24` | `0.5161` | `1.00×` | `0.00 / 0` | `1914.77` | `100.00%` | `100.00%` | `0` | `0 / 0` |
| `fraud_1pct_grief_25pct` | **`hanzo_opai_bisection_game`** | **`0.0091`** | **`0.2720`** | **`1523.31`** | **`0.5161`** | **`3.41×`** | **`7.54 / 8`** | **`17.48`** | **`100.00%`** | **`100.00%`** | **`0`** | **`2227 / 4603`** |

> **Key takeaway on crossover:** Optimistic bisection dominates full re-execution cleanly whenever the aggregate dispute rate (`fraud_rate + griefing_rate`) stays below ~45% (`610×` at 0%, `44.9×` at 1%, `13.1×` at 6%, `4.85×` at 10%, `3.41×` at 26%). At `50%` fraud, end-to-end verification (challenger trace + bisection + judge + sovereign replay) reaches parity/crossover (`0.95×`).

### 5.2 Trace Depth Scaling (`O(log2 T)` Rounds & `O(1)` Single-Step Judge)

| Layers (`L`) | Tokens (`G`) | Total Steps (`T`) | Theoretical Max Rounds ($\lceil \log_2 T \rceil$) | Observed Rounds | Full Trace (`µs`) | Single-Step Judge (`µs`) | Judge Speedup vs Full | Wire / JSON Transcript (`B`) | Exact Pinpoint |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `2` | `1` | `20` | `5` | `5` | `535.14` | `17.34` | **`30.86×`** | `2,006 / 4,077` | `true` |
| `8` | `1` | `68` | `7` | `6` | `1,457.77` | `15.99` | **`91.16×`** | `2,089 / 4,338` | `true` |
| `16` | `2` | `264` | `9` | `8` | `3,849.32` | `16.66` | **`231.08×`** | `2,268 / 5,059` | `true` |
| `32` | `4` | `1,040` | `11` | `10` | `14,521.68` | `19.90` | **`729.76×`** | `2,441 / 5,530` | `true` |

Single-step judge execution remains flat (`15.99–19.90 µs`, $O(1)$) across a **52×** increase in trace depth (`T = 20 -> 1,040`), while bisection rounds grow strictly logarithmically (`5 -> 10`) and wire transcripts grow by only `+435 B` (`2,006 B -> 2,441 B`).

### 5.3 Security, Liveness & Sovereignty Audit (`results.json:security_audit`)

| Audit Dimension | Passed / Tested | Rate | Notes |
|---|---:|---:|---|
| Exact single-step fault pinpointing (`0, 1, T/4, T/2, T-2, T-1` + `start_root` forgery) | `7 / 7` | `100.00%` | `0`-round start-root fraud proof + $\le \lceil\log_2 T\rceil$ step pinpoint |
| Malicious / griefing challengers slashed | `2 / 2` | `100.00%` | Honest provider compensated + burn `> 0` |
| Per-turn timeout forfeits enforced (defender mid-round, defender single-step, challenger mid-round) | `3 / 3` | `100.00%` | Automatic default slashing past `turn_deadline_secs` |
| Transcript tamper rejection (wire bit-flip, forged winner, altered witness, forged bond) | `4 / 4` | `100.00%` | All rejected even when attacker recomputes `transcript_digest` |
| Sovereign lossless wire & JSON round-trip + independent third-party replay | `6 / 6` | `100.00%` | **`0` Hanzo control-plane calls** |

---

## 6. Explicit Honest Negative Regimes

Optimistic bisection fraud proofs are **not** universally optimal. `opai-bench` explicitly measures and reports three regimes where `hanzo-opai` is strictly outperformed by direct single-pass full re-execution or cryptographic PoAI (`hanzo-poai`):

### 6.1 Negative Regime A — High-Dispute / 100% Challenge Storm

When 100% of honest claims are attacked by malicious challengers (`N = 20` claims, `T = 144` steps), interactive bisection incurs challenger recomputation, $\lceil \log_2 T \rceil$ interactive round-trips (`80 ms/msg`), single-step judging, and transcript verification — making BOTH compute cost and latency strictly worse than direct single-pass re-execution:

| Concurrent Challengers / Claim | Full Re-exec Compute (`µs`) | OPAI Storm Compute (`µs`) | Compute Overhead vs Full | Full Re-exec Latency (`ms`) | OPAI Storm Latency (`ms`) | Griefers Slashed | Honest Claims Finalized | Bond Conservation |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `1` | `37,089.13` | `62,278.60` | **`1.679×` worse** | `1.854` | **`1,363.114`** (`735×` slower) | `20 / 20` | `20 / 20` | `true` |
| `4` *(Sybil storm)* | `37,089.13` | `241,761.14` | **`6.518×` worse** | `1.854` | **`1,372.088`** (`740×` slower) | `80 / 80` | `20 / 20` | `true` |

*Mitigation:* Anti-griefing bond escalation (`1.5×` per concurrent slot) plus `50%` slashing burn makes sustained challenge storms exponentially expensive for attackers while guaranteeing `20 / 20` honest claims still settle cleanly with exact bond conservation. Under expected sustained high-dispute rates (`> 45%`), switch to synchronous Freivalds (`hanzo-poai`) or direct re-execution.

### 6.2 Negative Regime B — Challenge-Window Finality Delay

Optimistic settlement requires locking provider bonds and deferring unconditional economic finality for the full `challenge_window_secs`, compared to synchronous millisecond finality under direct re-execution:

| Challenge Window (`s`) | Full Re-exec Finality (`ms`) | Optimistic Finality (`ms`) | Finality Delay Ratio |
|---:|---:|---:|---:|
| `60` | `3.751` | `60,000.0` | **`15,997.8×` slower** |
| `600` | `3.751` | `600,000.0` | **`159,977.9×` slower** |
| `3,600` | `3.751` | `3,600,000.0` | **`959,867.5×` slower** |

*Guidance:* Applications requiring immediate atomic settlement (e.g., sub-second synchronous trading or on-chain oracle execution) cannot rely solely on long challenge windows without an external bonded liquidity/fast-finality underwriter or synchronous `hanzo-poai` verification.

### 6.3 Negative Regime C — Shallow 1–4 Step Workloads (`T ∈ {1, 2, 4}`)

For tiny workloads (`L = 0`, `T = 1, 2, 4` steps, `250` timed iterations after warmup), bisection state setup, commitment hashing, transcript serialization, and sovereign replay overhead strictly exceed direct single-pass re-execution (`speedup < 1.0×`):

| Total Steps (`T`) | Full Re-exec (`µs`) | OPAI Dispute Protocol (`µs`) | Speedup vs Full Re-exec | Wire Transcript (`B`) |
|---:|---:|---:|---:|---:|
| `1` | `36.166` | `119.664` | **`0.302×` (3.3× slower)** | `1,129` |
| `2` | `43.588` | `124.925` | **`0.349×` (2.9× slower)** | `1,215` |
| `4` | `58.215` | `131.927` | **`0.441×` (2.3× slower)** | `1,299` |

*Guidance:* For shallow `T <= 4` workloads, execute direct deterministic verification rather than opening an interactive bisection game.
