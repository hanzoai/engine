# HIP-1350 Research Report — Verifiable Federated Updates & Byzantine-Robust Training (`hanzo-fedverify`, `hanzoai/engine#73`)

**Status:** Implemented & verified clean-room Apache-2.0 workspace crate (`hanzo-fedverify`, 0 external dependencies)  
**Target Issue:** [`hanzoai/engine#73`](https://github.com/hanzoai/engine/issues/73) (`[p1, decentralized, training, hip-1350]`)  
**Protocol Identifier:** `hanzo.fedverify.dso.v1` (RFC 8032 Ed25519 + FIPS 180-4 SHA-256/SHA-512 + `FVER` v1 sovereign wire)  
**Benchmark Artifact:** [`results.json`](./results.json) (deterministic `--seed 1350`, `N = 24` participants, pinned to versioned repository inputs)

---

## 1. Executive Summary

`hanzo-fedverify` implements a verifiable federated update and Byzantine-robust aggregation engine aligned with HIP-1350 (referencing Flower, Gensyn RepOps, and Zoo DSO) with **zero external crate dependencies**, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`, and Apache-2.0 clean-room licensing.

Independent training nodes cannot be trusted when submitting gradients, LoRA adapters, weight deltas, or claimed loss improvements. `hanzo-fedverify` replaces unverified FedAvg and purely statistical defenses with a **5-layer verifiable defense pipeline** (`hanzo_fedverify_dso`):
1. **Layer 1 — Cryptographic Model-Update Identity & Anti-Replay:** Pure-Rust RFC 8032 Ed25519 + FIPS 180-4 SHA-256 binds `(round_id, participant_id, participant_pubkey, base_model_hash, update_kind, delta_hash, claimed_loss_delta, sample_count, tokens_processed, nonce)` into a `SignedUpdateEnvelope` supporting both dense deltas (`UpdateKind::GradientDelta`) and rank-`r` LoRA adapters (`UpdateKind::LoraAdapter`). Immediately blocks unknown keys, forged signatures, payload tampering, stale base models, duplicate submissions, and replayed nonces.
2. **Layer 2 — Hard & Adaptive Norm Bounds:** Rejects `100×` scaling explosions via `min(hard_max_l2_norm, adaptive_norm_multiplier * median_norm)`.
3. **Layer 3 — Sybil / Collusion Cluster Detection + Validation Cross-Check:** Detects coordinated near-clone cartels (`cosine >= 0.992`, relative L2 gap `<= 0.08`, cluster size `>= 3`) that artificially minimize pairwise distances to hijack naive Multi-Krum, cross-checking the cluster centroid on an independent held-out validation split so naturally correlated honest gradients are never falsely blamed.
4. **Layer 4 — Robust Direction & Multi-Krum MAD Filtering:** Rejects sign-flipping / gradient ascent (`cosine < min_cosine_similarity` against the coordinate-wise median) and extreme coordinate outliers.
5. **Layer 5 — Independent Held-Out Quality Gate & Claim Audit + Signed Attribution:** Evaluates each candidate on an independent held-out validation split (`D_val`), rejecting validation regressions and fabricated improvement claims (`> 20×` claim inflation), commits only if the aggregated update passes a post-aggregation zero-regression check, and issues an **Ed25519-signed `RejectionReceipt`** for every rejected update inside a lossless **`FVER` v1** sovereign audit ledger verifiable offline with **0 Hanzo control-plane calls**.

Across **`N = 24` simulated participants** (`12` rounds, `--seed 1350`) trained on real versioned `hanzoai/engine` corpus text:
- From `0%` through `41.7%` mixed Byzantine adversaries (combining sign-flipping, `100×` scaling, Sybil collusion cartels, fake claimed improvements, and tamper/replay), `hanzo_fedverify_dso` achieves **`0.00%` attack success**, **`0.00%` honest false-rejection rate**, and final held-out test loss **`2.1887–2.1908`** — matching `oracle_honest_only_ceiling` (`2.1892–2.1908`), whereas `naive_fedavg_baseline` collapses catastrophically (`test_loss = 48.9667`, `acc = 2.07%`) and pure statistical `trimmed_mean_krum` admits **`16.67%–49.17%`** of malicious updates while falsely rejecting **`13.69%–29.17%`** of honest participants.
- In per-attack-class isolation (`29.2%` pure attack), `hanzo_fedverify_dso` blocks **`100.00%`** (`84/84`) of every attack class with exact reason-code attribution, including **`79.76%`** cartel hijack and **`100.00%`** tamper/replay admission that defeat pure `trimmed_mean_krum`.

---

## 2. Threat Model, Guarantees, and Explicit Scope Boundaries

| Layer | What `hanzo-fedverify` Provides | What Remains Explicitly Out of Scope |
|---|---|---|
| **Cryptographic model-update identity** | Clean-room RFC 8032 Ed25519 (verified against RFC 8032 §7.1 vectors #1 & #2) + FIPS 180-4 SHA-256/SHA-512 binding round, base model hash, payload hash (dense or LoRA), claimed loss delta, sample count, and strictly monotonic anti-replay nonce. | Hardware key custody, PKI/on-chain validator stake registries, and sybil admission bonding live at the chain/economic layer. |
| **Byzantine-robust aggregation & collusion resistance** | 5-layer defense defeating sign-flipping, `100×` norm scaling, Sybil clone cartels, fake claimed improvements, and tamper/replay up to `41.7%` Byzantine ratios; preserves safety (zero poisoning) even at `54.2%` super-majority adversaries. | When `f >= n/2` diverse adversaries control the statistical median, safety holds via the independent validation gate, but liveness/honest throughput stalls (explicitly reported in §6.1). |
| **Independent quality evaluation** | Strictly separated held-out `independent_val` (gating) and `held_out_test` (unbiased final evaluation) splits derived from pinned versioned repository text. | Assumes the verifier holds a representative unpoisoned validation set; targeted covert backdoors on triggers absent from `D_val` require complementary trigger/red-team evaluation. |
| **Attribution & sovereignty** | Ed25519-signed `RejectionReceipt` per rejected update, per-round Merkle roots (`accepted_merkle_root`, `rejection_merkle_root`), hash-chained `FederationAuditLedger`, lossless `FVER` v1 binary + JSON export, and `verify_ledger_sovereign` requiring **0 Hanzo control-plane calls**. | P2P gossip transport, secure aggregation (MPC/FHE privacy of individual gradients), and differential privacy are orthogonal layers. |

---

## 3. Crate Architecture (`hanzo-fedverify`)

- [`hanzo-fedverify/src/crypto.rs`](../../../hanzo-fedverify/src/crypto.rs): Pure-Rust FIPS 180-4 `Sha256` & `Sha512`, RFC 8032 `Ed25519Keypair` & `ed25519_verify` over Curve25519 (`F_{2^255-19}`), constant-time `ct_eq_bytes`, binary `merkle_root`, hex codecs, and `SplitMix64`.
- [`hanzo-fedverify/src/identity.rs`](../../../hanzo-fedverify/src/identity.rs): `ParticipantIdentity`, `ParticipantSigner`, `UpdateKind` (`GradientDelta` & `LoraAdapter`), `UpdateHeader`, `SignedUpdateEnvelope`, 10-code `RejectionReason`, Ed25519-signed `RejectionReceipt`, and `AcceptedRecord`.
- [`hanzo-fedverify/src/model.rs`](../../../hanzo-fedverify/src/model.rs): Pinned versioned corpus loader (`CorpusDataset`, `CorpusPin`), 12-class lexical/structural next-token language model (`MODEL_DIM = 384`), deterministic mini-batch SGD (`batch_size = 32`), rank-`r` LoRA adapter factorization (`compress_to_lora_adapter`), `StratifiedIid` & `ExtremeNonIid` partitioning, and held-out evaluation (`EvalMetrics`).
- [`hanzo-fedverify/src/aggregator.rs`](../../../hanzo-fedverify/src/aggregator.rs): `FederatedAggregator`, `DefenseConfig` (`balanced_default`, `strict_defense`, `non_iid_relaxed`), `RoundAuditRecord`, and all 4 matched regimes (`naive_fedavg_baseline`, `trimmed_mean_krum`, `hanzo_fedverify_dso`, `oracle_honest_only_ceiling`).
- [`hanzo-fedverify/src/ledger.rs`](../../../hanzo-fedverify/src/ledger.rs): `FederationAuditLedger`, lossless `FVER` v1 binary wire codec (anti-DoS length bounds + strict EOF), clean-room JSON export, and offline `verify_ledger_sovereign` (**0 Hanzo control-plane calls**).
- [`hanzo-fedverify/src/simulation.rs`](../../../hanzo-fedverify/src/simulation.rs): Deterministic multi-participant simulation harness (`N >= 24`), 7 participant roles, and HIP-1350 5-axis telemetry.
- [`hanzo-fedverify/src/bin/fedverify-bench.rs`](../../../hanzo-fedverify/src/bin/fedverify-bench.rs): Reproducible benchmark runner emitting [`results.json`](./results.json).
- [`hanzo-fedverify/tests/integration_tests.rs`](../../../hanzo-fedverify/tests/integration_tests.rs): 9 integration tests (plus 5 cryptographic unit tests = **14 passing tests**).

---

## 4. Pinned Repository Inputs & Reproducible Commands

All training shards, independent validation samples (`1,500`), and held-out test samples (`1,500`) are constructed from real versioned files in `hanzoai/engine` (`27,387` parsed tokens total):

| Versioned Repository File | Size (Bytes) | SHA-256 Digest |
|---|---:|---|
| `calibration_data/calibration_datav3_small.txt` | `133,837` | `aabd956dba34ff31f40e78b9de4e234fdd4865f90ba231e8e3deec8dfb22cbc0` |
| `chat_templates/llama3.json` | `420` | `67d1d402738727aa3059d28590fa3488c44740a358be4b5c2135c60f58ec9d7a` |
| `orderings/xlora-paper-ordering.json` | `6,485` | `8d0650ea231eeade0b97e5c026411365ff9feed41b2b07ecf10853c2a399043a` |

### Exact Reproduction Commands

```bash
cargo fmt -p hanzo-fedverify -- --check
cargo clippy -p hanzo-fedverify --all-targets --offline -- -D warnings
cargo test -p hanzo-fedverify --offline --release
cargo run --offline --release -p hanzo-fedverify --bin fedverify-bench -- \
  --seed 1350 \
  --out-json docs/research/hip-1350-verifiable-federated-training/results.json
```

---

## 5. Measured HIP-1350 5-Axis Benchmark Results (`--seed 1350`, `N = 24`, `12` Rounds)

Reference workload: `24` participants (`9,000` train samples = `375`/node, `5` corpus tokens/sample, `1,500` independent validation samples, `1,500` held-out test samples, untrained baseline `test_loss = 2.4846`, `ppl = 12.00`, `acc = 10.80%`), reference compute rate `$0.05 / vCPU-hour`.

### 5.1 Matched 4-Regime Comparison Across Mixed Byzantine Ratios (`0%` to `54.2%`)

| Scenario (Byzantine / 24) | Regime | Axis 1: Cost (`$/1M useful tok`) | Axis 1: Verify Share | Axis 2: Useful Throughput (`tok/s`) | Axis 2: Verify (`ms/round`) | Axis 3: Attack Success | Axis 3: Honest Rejection | Final Test Loss (PPL / Acc) | Axis 4: Control-Plane Calls | Axis 5: Wire / JSON (`B`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `byz_0pct_clean` (`0/24`) | `naive_fedavg_baseline` | `0.000007` | `1.26%` | `2,136,030` | `0.265` | `0.00%` | `0.00%` | `2.1893` (`8.93` / `23.87%`) | `0` | `50,313 / 12,156` |
| `byz_0pct_clean` (`0/24`) | `trimmed_mean_krum` | `0.000010` | `7.38%` | `1,461,169` | `1.609` | `0.00%` | `29.17%` | `2.1891` (`8.93` / `24.40%`) | `0` | `63,497 / 35,408` |
| `byz_0pct_clean` (`0/24`) | **`hanzo_fedverify_dso`** | **`0.000010`** | **`37.94%`** | **`1,358,362`** | **`12.569`** | **`0.00%`** | **`0.00%`** | **`2.1890`** (`8.93` / `24.20%`) | **`0`** | `50,311 / 12,154` |
| `byz_0pct_clean` (`0/24`) | `oracle_honest_only_ceiling` | `0.000007` | `3.64%` | `2,109,261` | `0.777` | `0.00%` | `0.00%` | `2.1893` (`8.93` / `23.87%`) | `0` | `50,318 / 12,161` |
| `byz_17pct_mixed` (`4/24`) | `naive_fedavg_baseline` | `0.000007` | `1.37%` | `1,963,469` | `0.262` | `100.00%` | `0.00%` | `48.9667` *(collapsed)* (`2.07%`) | `0` | `50,313 / 12,156` |
| `byz_17pct_mixed` (`4/24`) | `trimmed_mean_krum` | `0.000010` | `8.38%` | `1,395,099` | `1.839` | `16.67%` | `18.33%` | `2.1905` (`8.94` / `23.67%`) | `0` | `63,497 / 35,420` |
| `byz_17pct_mixed` (`4/24`) | **`hanzo_fedverify_dso`** | **`0.000012`** | **`37.13%`** | **`1,170,460`** | **`11.897`** | **`0.00%`** | **`0.00%`** | **`2.1887`** (`8.92` / `24.27%`) | **`0`** | `57,847 / 25,608` |
| `byz_17pct_mixed` (`4/24`) | `oracle_honest_only_ceiling` | `0.000008` | `3.36%` | `1,750,540` | `0.719` | `0.00%` | `0.00%` | `2.1892` (`8.93` / `24.00%`) | `0` | `42,878 / 12,161` |
| `byz_29pct_mixed` (`7/24`) | `naive_fedavg_baseline` | `0.000008` | `1.51%` | `1,787,135` | `0.269` | `100.00%` | `0.00%` | `48.9667` *(collapsed)* (`2.07%`) | `0` | `50,313 / 12,156` |
| `byz_29pct_mixed` (`7/24`) | `trimmed_mean_krum` | `0.000010` | `8.16%` | `1,374,289` | `1.532` | `46.43%` | `19.12%` | `2.2035` (`9.06` / `23.07%`) | `0` | `63,497 / 35,420` |
| `byz_29pct_mixed` (`7/24`) | **`hanzo_fedverify_dso`** | **`0.000013`** | **`42.76%`** | **`1,061,811`** | **`12.836`** | **`0.00%`** | **`0.00%`** | **`2.1894`** (`8.93` / `23.53%`) | **`0`** | `63,499 / 35,871` |
| `byz_29pct_mixed` (`7/24`) | `oracle_honest_only_ceiling` | `0.000008` | `3.65%` | `1,748,195` | `0.666` | `0.00%` | `0.00%` | `2.1898` (`8.93` / `23.87%`) | `0` | `37,298 / 12,161` |
| `byz_42pct_mixed` (`10/24`) | `naive_fedavg_baseline` | `0.000010` | `1.57%` | `1,449,699` | `0.285` | `100.00%` | `0.00%` | `48.9667` *(collapsed)* (`2.07%`) | `0` | `50,313 / 12,156` |
| `byz_42pct_mixed` (`10/24`) | `trimmed_mean_krum` | `0.000011` | `8.43%` | `1,245,241` | `1.534` | `49.17%` | `13.69%` | `2.2204` (`9.21` / `21.93%`) | `0` | `63,497 / 35,432` |
| `byz_42pct_mixed` (`10/24`) | **`hanzo_fedverify_dso`** | **`0.000015`** | **`42.92%`** | **`898,288`** | **`12.543`** | **`0.00%`** | **`0.00%`** | **`2.1908`** (`8.94` / `23.20%`) | **`0`** | `69,151 / 46,072` |
| `byz_42pct_mixed` (`10/24`) | `oracle_honest_only_ceiling` | `0.000010` | `3.81%` | `1,425,652` | `0.701` | `0.00%` | `0.00%` | `2.1908` (`8.94` / `22.93%`) | `0` | `31,718 / 12,161` |
| `byz_54pct_supermajority` (`13/24`) | `naive_fedavg_baseline` | `0.000008` | `2.42%` | `1,824,139` | `0.274` | `100.00%` | `0.00%` | `2.4981` *(degraded)* (`13.87%`) | `0` | `50,313 / 12,156` |
| `byz_54pct_supermajority` (`13/24`) | `trimmed_mean_krum` | `0.000013` | `11.75%` | `1,105,062` | `1.512` | `72.44%` | `31.06%` | `2.5622` *(corrupted)* (`12.93%`) | `0` | `63,497 / 35,408` |
| `byz_54pct_supermajority` (`13/24`) | **`hanzo_fedverify_dso`** | `0.000000` *(stalled)* | `51.91%` | `0` *(liveness stall)* | `11.935` | **`0.00%`** *(safe)* | **`100.00%`** | **`2.4846`** *(no poisoning)* (`10.80%`) | **`0`** | `95,527 / 94,258` |
| `byz_54pct_supermajority` (`13/24`) | `oracle_honest_only_ceiling` | `0.000008` | `5.41%` | `1,683,237` | `0.663` | `0.00%` | `0.00%` | `2.1936` (`8.97` / `23.07%`) | `0` | `26,138 / 12,161` |

### 5.2 Per-Attack-Class Isolation Matrix (`7/24 = 29.2%` Pure Attack, `84` Malicious Submissions Each)

| Attack Class | `naive_fedavg` Attack Success / Loss | `trimmed_mean_krum` Attack Success / Honest Rej / Loss (`p0` Poison Rate) | **`hanzo_fedverify_dso`** Attack Success / Honest Rej / Loss (`p0`) | Primary Signed `RejectionReceipt` Attribution (`84` Total) |
|---|---:|---:|---:|---|
| `pure_sign_flip_29pct` | `100.00%` / `2.3595` | `0.00%` / `0.00%` / `2.1889` (`6.40%`) | **`0.00%`** / **`0.00%`** / **`2.1894`** (`6.47%`) | `cosine_direction_anomaly: 84` |
| `pure_scaling_100x_29pct` | `100.00%` / `48.9667` | `0.00%` / `0.00%` / `2.1889` (`6.40%`) | **`0.00%`** / **`0.00%`** / **`2.1894`** (`6.47%`) | `norm_bound_violation: 84` |
| `pure_collusion_cartel_29pct` | `100.00%` / `2.2197` | **`79.76%`** / `32.84%` / `2.2331` (**`22.47%`**) | **`0.00%`** / **`0.00%`** / **`2.1894`** (`6.47%`) | `collusion_cluster_detected: 84` |
| `pure_fake_claim_29pct` | `100.00%` / `2.2082` | **`57.14%`** / `23.53%` / `2.2061` (**`16.33%`**) | **`0.00%`** / **`0.00%`** / **`2.1894`** (`6.47%`) | `validation_loss_regression: 77`, `coordinate_outlier: 7` |
| `pure_tamper_replay_29pct` | `100.00%` / `2.2018` | **`100.00%`** / `41.18%` / `2.2124` (**`14.27%`**) | **`0.00%`** / **`0.00%`** / **`2.1894`** (`6.47%`) | `invalid_signature: 42`, `replay_nonce: 21`, `stale_base_model: 21` |

### 5.3 LoRA Rank-4 Adapter Verification (`lora_r4_byz_29pct_mixed`, `176` vs `384` Params = `2.18×` Payload Reduction)

| Regime | Attack Success | Honest Rejection | Final Test Loss | Final Perplexity | Top-1 Accuracy |
|---|---:|---:|---:|---:|---:|
| `naive_fedavg_baseline` | `100.00%` | `0.00%` | `48.9667` | `1.84e21` | `2.07%` |
| `trimmed_mean_krum` | `46.43%` | `19.12%` | `2.2451` | `9.44` | `21.67%` |
| **`hanzo_fedverify_dso`** | **`0.00%`** | **`0.00%`** | **`2.2293`** | **`9.29`** | **`21.53%`** |
| `oracle_honest_only_ceiling` | `0.00%` | `0.00%` | `2.2270` | `9.27` | `21.60%` |

### 5.4 Security & Sovereignty Audit (`results.json:security_and_sovereignty_audit`)

| Audit Check (`byz_29pct_mixed` Ledger: `63,499 B` `FVER` v1 Wire / `35,871 B` JSON) | Result | Notes |
|---|---|---|
| Lossless `FVER` v1 wire encode → decode → independent offline replay | **PASS (`12/12` rounds, `84/84` signed receipts)** | **`0` Hanzo control-plane calls** |
| Wire bit-flip inside serialized ledger | **REJECTED** | Digest & Ed25519 signature mismatch |
| Forged `RejectionReason` (with recomputed top-level `ledger_digest`) | **REJECTED** | Individual `RejectionReceipt` Ed25519 signature fails |
| Deleted `RejectionReceipt` (hiding Byzantine attribution) | **REJECTED** | `rejection_merkle_root` + round auditor signature fail |
| Altered `final_model_hash` (checkpoint substitution) | **REJECTED** | Hash-chain continuity check fails |
| Forged round `auditor_signature` | **REJECTED** | RFC 8032 Ed25519 verification fails |

---

## 6. Explicit Honest Negative Regimes

Verifiable Byzantine-robust federated training involves concrete mathematical and systems tradeoffs. `fedverify-bench` explicitly measures and reports three regimes where `hanzo_fedverify_dso` encounters real limits or overhead taxes:

### 6.1 Super-Majority Colluding / Diverse Byzantine Adversaries (`>= 50%`, `byz_54pct_supermajority`: `13/24 = 54.2%`)
- **Mathematical breakdown of statistical medians (`f >= n/2`):** When `13/24` diverse non-clone adversaries outvote `11/24` honest participants, pure statistical `trimmed_mean_krum` collapses (`72.44%` attack success, `31.06%` honest rejection, test loss **worsens** from `2.4846` to `2.5622`).
- **Safety vs. liveness split in `hanzo_fedverify_dso`:** Because the `54.2%` adversarial majority flips Layer 4's coordinate-wise median, Layer 4 falsely flags all `132/132` honest minority updates (`cosine_direction_anomaly: 132`, **`100.00%` honest rejection**). Simultaneously, Layer 5's independent held-out validation gate catches all `156/156` adversarial updates (`validation_loss_regression: 156`, **`0.00%` attack success**). Thus **safety holds** (model loss stays cleanly at `2.4846` with zero poisoning), while **liveness stalls** (`0/12` rounds committed, `0` useful tokens/s). Recovering liveness above `f >= n/2` requires bypassing statistical median pre-filtering and scoring candidates directly against `D_val` (at higher validation cost) or slashing Sybils economically.

### 6.2 Extreme Non-IID Honest Heterogeneity (`80%` 2-Class Label Skew per Participant)
When honest participants hold heavily skewed domain shards (`PartitionStrategy::ExtremeNonIid`), individual local updates naturally diverge from the global median and can momentarily raise loss on held-out classes not present in that participant's shard. Sweeping `DefenseConfig` profiles directly exposes the **honest false-rejection vs. attack-resistance Pareto tradeoff**:

| Scenario | Defense Profile | Attack Success | Honest False-Rejection | Final Test Loss | Top-1 Accuracy | Useful Throughput (`tok/s`) |
|---|---|---:|---:|---:|---:|---:|
| `extreme_non_iid_0pct_attack` | `strict_defense` | `0.00%` | **`100.00%`** | `2.4846` *(stalled)* | `10.80%` | `0` |
| `extreme_non_iid_0pct_attack` | `balanced_default` | `0.00%` | **`100.00%`** | `2.4846` *(stalled)* | `10.80%` | `0` |
| `extreme_non_iid_0pct_attack` | **`non_iid_relaxed`** | `0.00%` | **`13.89%`** | **`2.2855`** | **`24.13%`** | `1,200,658` |
| `extreme_non_iid_0pct_attack` | `oracle_honest_only_ceiling` | `0.00%` | `0.00%` | `2.2920` | `25.67%` | `2,270,735` |
| `extreme_non_iid_29pct_attack` | `strict_defense` | `0.00%` | `100.00%` | `2.4846` *(stalled)* | `10.80%` | `0` |
| `extreme_non_iid_29pct_attack` | `balanced_default` | `0.00%` | `100.00%` | `2.4846` *(stalled)* | `10.80%` | `0` |
| `extreme_non_iid_29pct_attack` | **`non_iid_relaxed`** | **`14.29%`** | **`25.00%`** | **`2.2865`** | **`20.93%`** | `774,593` |

> **Key takeaway:** Tight IID thresholds (`strict_defense` and `balanced_default`) suffer severe false rejections under extreme 2-class Non-IID skew (`100%` stall). Switching to `non_iid_relaxed` restores convergence (`test_loss = 2.2855`, matching/beating Oracle `2.2920` with `13.89%` honest rejection at `0%` attack), at the explicit cost of admitting `14.29%` of subtle attacks under a `29.2%` mixed adversary (though final test loss remains protected at `2.2865`!).

### 6.3 Clean `0%`-Attack Rounds — Verification Overhead Tax (`byz_0pct_clean`)
When **0 adversaries** are present:
- `naive_fedavg_baseline` spends only **`0.265 ms/round`** in aggregation (`1.26%` of total round compute, `2,136,030` useful tok/s, `$0.000007 / 1M useful tokens`).
- `hanzo_fedverify_dso` spends **`12.569 ms/round`** (`47.4×` aggregator latency vs. naive FedAvg, `37.94%` of total round compute for 24 pure-Rust Ed25519 verifications + `O(N²)` pairwise collusion check + 24 independent `D_val` forward passes + Merkle commitments), reducing end-to-end useful throughput by `36.4%` (`1,358,362` tok/s) and raising end-to-end cost to **`$0.000010 / 1M useful tokens`** (`+43%` total compute cost tax on this small `384`-parameter benchmark model; for larger LLM LoRA training where client local backward passes take seconds/minutes, the same `~12.6 ms` verification cost amortizes to `< 0.1%`).
