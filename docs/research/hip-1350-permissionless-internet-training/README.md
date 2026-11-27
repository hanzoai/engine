# HIP-1350 Research Dossier: Permissionless Internet-Scale Elastic LoRA Training (`hanzoai/engine#72`)

- **Crate**: `hanzo-diloco-train` (clean-room Apache-2.0, zero external dependencies, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`)
- **Protocol Version**: `hanzo.train.elastic/v1` (`hanzo_coordinator_required = false`)
- **Literature & Ecosystem Context**: Builds on two-level low-communication distributed optimization (DiLoCo, [arXiv:2311.08105](https://arxiv.org/abs/2311.08105); Prime DiLoCo) and aligns with the companion training benchmark track (`zoo-apps/gym#4`).
- **Deterministic Seed**: `--seed 1350`
- **Verified Grounded Corpus SHA-256**: `84b45bc2b79cb4612938566807c5eb8b478824036f382307c869cb112a8cd856` (`140,742` bytes across `calibration_data/calibration_datav3_small.txt`, `chat_templates/llama3.json`, and `orderings/xlora-paper-ordering.json`)
- **Artifact Receipt**: [`results.json`](results.json) verified by [`results.json.sha256`](results.json.sha256) (`ab197b63fad73839b2138476a52250d24d420977a7048687f431a2540d884270`)

---

## 1. Executive Summary

`hanzo-diloco-train` implements a reproducible, pure-Rust elastic low-communication LoRA training runtime over live loopback TCP sockets with simulated inter-region WAN latency and bandwidth shaping (`us-east`: `45 ms / 200 Mbps`, `eu-west`: `85 ms / 150 Mbps`, `ap-south`: `120 ms / 100 Mbps`), mid-run abrupt worker failure (`hard_kill()`), live TCP checkpoint recovery from content-addressed VFS snapshots (`vfs://checkpoints/<sha256>`), and Byzantine-robust aggregation.

Every convergence curve in this dossier is produced by executing real forward passes and exact analytical backpropagation through a causal small Zen Transformer (`SmallZenLm`, `vocab=98`, `d_model=48`, `d_ff=96`, `seq_len=16`, `lora_rank=8`, `5,776` trainable LoRA parameters) on the SHA-256 verified repository corpus — zero synthetic or fabricated loss trajectories.

### Key Measured Outcomes (`--seed 1350`)

| Metric | Centralized LAN (`H=1`, FP32) | 3-Node WAN Steady (`H=16`, INT8) | 3-Node WAN Churn + Recovery (`H=16`, INT8) |
| --- | ---: | ---: | ---: |
| Initial Held-Out Val Loss / PPL | `4.7004` / `109.99` | `4.7004` / `109.99` | `4.7004` / `109.99` |
| **Final Held-Out Val Loss / PPL** | **`2.9826` / `19.74`** | **`3.0128` / `20.34`** (`+0.030` nats) | **`3.0568` / `21.26`** (`+0.074` nats) |
| Final Val Top-1 Next-Token Acc | `19.47%` | `20.18%` | `20.25%` |
| Tokens Trained (Useful) | `49,152` | `49,152` | `45,056` (`2` rounds at `2/3` quorum) |
| **Total Live TCP Wire Bytes** | `17,921,664 B` | **`712,984 B` (`25.14x` less)** | **`746,858 B`** (incl. `46,630 B` TCP recovery) |
| **Pseudo-Gradient Upload Bytes** | `8,875,008 B` | **`147,552 B` (`60.15x` less)** | **`135,256 B`** |
| Effective WAN Throughput | `25,767.9 tok/s` | `21,383.7 tok/s` (`83.0%` of LAN) | `20,121.6 tok/s` |
| Time to First Checkpoint (TTFC) | `127.04 ms` | `416.71 ms` | `429.69 ms` |
| Round Wall Latency (`p50 / p95 / p99`) | `66.74 / 74.43 / 75.67 ms` | `375.19 / 396.49 / 396.49 ms` | `373.28 / 459.58 / 459.58 ms` |
| Completion Rate | `100.0%` (`128/128`) | `100.0%` (`8/8`) | **`100.0%` (`8/8`)** |
| False-Positive Honest Rejections | `0` | `0` | `0` |

---

## 2. Five Mandatory Program Metrics

| Dimension | Measured / Modeled Value | Basis & Verification |
| --- | --- | --- |
| **Cost** (`$/1M useful tokens`) | Live CPU small model: `\$0.002090` decentralized vs `\$0.002587` centralized (`-19.2%`). Analytical GPU LoRA-r32 (`H=64`, incl. 2% verification + WAN egress + settlement): **4B** `\$0.0218` vs `\$0.0424` (`-48.6%`); **9B** `\$0.0680` vs `\$0.0848` (`-19.8%`); **27B** `\$0.3037` vs `\$0.2529` (`+20.1%`, centralized NVLink wins on single-node 27B). | `results.json` (`program_metrics_summary` and `analytical_scale_projections`). |
| **Performance** | Steady 3-node WAN: `TTFC = 416.71 ms`, `21,383.7 eff tok/s`, `p50/p95/p99 = 375.19 / 396.49 / 396.49 ms`, `100%` completion. Churn run: `TTFC = 429.69 ms`, `20,121.6 eff tok/s`, `88.41 ms` live TCP checkpoint recovery, `100%` completion. | Measured directly on live `127.0.0.1` TCP sockets with 1:1 simulated WAN RTT sleeps. |
| **Trust** | **`6/6` (`100%`)** adversarial vectors detected & rejected under declared tier `TrustTier::SignedDeltaAndLossSpotCheck`. | Live TCP + VFS corruption suite (`t13`–`t18` and `adversarial_defense_suite`). |
| **Decentralization** | **`100%` (`1.0000`)** of training, aggregation, verification, and checkpoint recovery proceeds with `hanzo_coordinator_required = false`. | Any peer can mirror `vfs://checkpoints/<sha256>` and coordinate outer rounds without Hanzo-hosted services. |
| **Sovereignty** | Self-verifying binary VFS checkpoints (`vfs://checkpoints/<sha256>`) and flat `hanzo-lora-v1` adapter exports enable operators to export, migrate, resume, or fork without lock-in. | Verified by cold-start replacement worker bootstrapping solely from signed VFS checkpoint bytes over TCP. |

---

## 3. Architecture & Hanzo Stack Integration (`hanzo.train.elastic/v1`)

1. **Two-Level Optimization (`src/optimizer.rs`)**:
   - **Inner loop (`H = 16` steps)**: Each worker runs independent decoupled-weight-decay AdamW (`lr = 4.5e-3`, gradient clip `2.0`) on its disjoint grounded corpus shard with **zero cross-WAN traffic**.
   - **Pseudo-gradient compression**: After `H` steps, worker `k` computes $\Delta_k = \theta^{(t)} - \theta_k^{(t, H)}$ and encodes it with block-64 symmetric INT8 quantization + persistent per-worker error-feedback residual accumulation (`~3.76x` smaller than FP32 with `< 1e-3` cumulative drift over 20 rounds).
   - **Outer Nesterov momentum**: Accepted pseudo-gradients are merged via coordinate-wise trimmed mean and applied with outer Nesterov momentum ($\eta_{\text{outer}} = 0.7$, $\mu = 0.65$).
2. **Versioned Stack Integration (`src/checkpoint.rs`, `HanzoTrainingManifest`)**:
   - **Enso Peer Routing**: Every worker registers an `EnsoPeerRoute` (`enso://127.0.0.1:<port>`, region, RTT/bandwidth profile).
   - **Visor Isolation**: `VisorIsolationSpec` records sandbox profile (`visor-sandbox-cpu-v1`), memory cap (`512 MiB`), and peer/VFS egress allowlist policy.
   - **VFS Checkpoints**: Immutable content-addressed snapshots (`vfs://checkpoints/<sha256>`) store LoRA parameters, outer Nesterov velocity, lineage parent digest, and HMAC-SHA256 receipt.
   - **No Mandatory Hanzo Coordinator**: `hanzo_coordinator_required = false` is pinned in both the manifest and integration tests.

---

## 4. Live 3-Node Worker Churn & Checkpoint Recovery Trace

In `wan_3node_churn_recovery_h16_int8` (`results.json`):
- **Rounds 1–2**: All 3 geographic workers (`worker-us-east`, `worker-eu-west`, `worker-ap-south`) converge monotonically (`val_loss: 4.7004 -> 3.4412 -> 3.3387`).
- **Round 3 (Abrupt Failure)**: `worker-eu-west` is terminated via `hard_kill()`, immediately dropping its `TcpListener`. The round dispatcher observes OS-level `Connection refused (os error 111)`, continues uninterrupted with the `2` surviving peers (`min_quorum = 2`), and advances validation loss to `3.1891`.
- **Round 4**: Surviving 2-peer quorum advances to `val_loss = 3.1290` and commits signed VFS checkpoint `46c873db3cf86136...`.
- **Round 5 (Live TCP Recovery & Rejoin)**: A fresh `worker-eu-replacement` connects over live TCP to the stateless VFS checkpoint mirror, downloads `46,630` raw checkpoint bytes across the `eu-west` WAN profile in `88.41 ms` (`82.74 ms` simulated WAN transit), verifies `checkpoint_sha256` and `receipt_hmac_sha256`, loads round-4 parameters, and rejoins round 5 (`accepted = 3`, `val_loss = 3.0867`).
- **Rounds 6–8**: All 3 workers complete the run (`100%` completion rate, final `val_loss = 3.0568`, `val_ppl = 21.26`, `top1_acc = 20.25%`).

---

## 5. Resident vs. Streamed Weight Accounting & 4B / 9B / 27B Scale Projections

Per Hanzo Engine accounting discipline, **full resident input-embedding tables (`vocab_size * d_model * 2` bytes BF16)** are explicitly separated from **single-row embedding lookups (`d_model * 2` bytes BF16)** in both the executable small model (`9,408 B` resident vs `96 B` single-row lookup) and large-model projections:

| Model Profile | Resident Input Embed (`vocab*d*2`) | Single-Row Lookup (`d*2`) | Trainable LoRA Params (`r=32`) | LoRA INT8 Sync Payload | Full FP32 Sync Payload | LoRA INT8 Sync @ 100 Mbps | Full FP32 Sync @ 100 Mbps | WAN Overhead (`H=64`) | Decentralized `$/1M tok` | Centralized H100 `$/1M tok` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zen-4b-lora-r32` | `777.9 MB` | `5.12 KB` | `66.06M` | `70.19 MB` | `16.00 GB` | `5.74 s` | `1,280.1 s` | `9.14%` | `\$0.0218` | `\$0.0424` |
| `zen-9b-lora-r32` | `1,089.1 MB` | `7.17 KB` | `105.51M` | `112.11 MB` | `36.00 GB` | `9.09 s` | `2,880.1 s` | `7.39%` | `\$0.0680` | `\$0.0848` |
| `zen-27b-lora-r32` | `1,555.8 MB` | `10.24 KB` | `241.17M` | `256.25 MB` | `108.00 GB` | `20.62 s` | `8,640.1 s` | `6.27%` | `\$0.3037` | `\$0.2529` |

Notice the honest economic boundary: for 4B and 9B LoRA fine-tuning, INT8 low-communication WAN training keeps communication under `10%` of round wall time and undercuts centralized H100 rates by `20%–49%` even after verification (`2%`), egress (`\$0.02/GB`), and settlement (`\$0.0005/1M tok`). At 27B on single spot GPUs without intra-node NVLink parallelism, spot compute efficiency drops below centralized H100 pricing (`\$0.3037` vs `\$0.2529`).

---

## 6. Adversarial Trust Model & 6/6 Detection Results

**Declared Trust Tier**: `TrustTier::SignedDeltaAndLossSpotCheck`

> **Non-Conflation Guarantee**: Signed worker receipts (HMAC-SHA256 + SHA-256 weight/delta commitments), norm bounds, held-out validation-loss spot-checking, and coordinate-wise trimmed-mean aggregation are **probabilistic/statistical and authenticated-integrity defenses**. They are **explicitly distinct** from hardware TEE attestation (`TrustTier::HardwareTeeAttestation`) and succinct cryptographic zkML execution proofs (`TrustTier::CryptographicZkProof`).

| # | Attack Vector | Status | Observed Runtime Rejection Signal |
| --- | --- | --- | --- |
| 1 | Stealthy Byzantine sign-flip poisoning (`-2x` inverted delta bypassing coarse `3.5x` median norm cap) | **REJECTED** | `adversary-ap:validation_loss_regression` |
| 2 | Free-rider zero-delta submission claiming full `H` inner steps | **REJECTED** | `adversary-ap:free_rider_zero_delta` |
| 3 | Stale-round replay (`round_id = 0`) | **REJECTED** | `adversary-ap:stale_round_replay` |
| 4 | Diverged model lineage (`base_checkpoint_sha256` mismatch) | **REJECTED** | `adversary-ap:base_checkpoint_mismatch` |
| 5 | Tampered wire delta bytes + forged HMAC-SHA256 receipt | **REJECTED** | `adversary-ap:delta_digest_mismatch` + `adversary-ap:invalid_hmac` |
| 6 | Corrupted newest VFS checkpoint file on disk during recovery | **REJECTED** | `checkpoint_digest_mismatch -> fallback_to_round_1` |

---

## 7. Honest Negative Regimes (`REJECTED`)

| Regime ID | Verdict | Empirical Failure Mode (`--seed 1350`) |
| --- | --- | --- |
| `neg_sync_allreduce_over_wan_h1` | **REJECTED** | Synchronous `H=1` FP32 AllReduce over `45/85/120 ms` WAN collapses throughput to `2,919.4 tok/s` vs `21,383.7 tok/s` for `H=16` INT8 (`7.32x` live-socket slowdown; `> 100x` wall collapse at 4B–27B scale). |
| `neg_excessive_local_steps_divergence` | **REJECTED** | Running `H=256` unsynchronized local steps induces severe LoRA client drift and bilinear averaging cross-term interference (`B_avg A_avg != avg(B_k A_k)`): held-out `val_loss = 3.3179` (`ppl = 27.60`) vs `H=16` `val_loss = 3.0128` (`ppl = 20.34`, `1.36x` perplexity degradation). |
| `neg_unprotected_byzantine_mean` | **REJECTED** | Naive unweighted averaging under 1/3 `-8x` sign-flip poisoning explodes held-out loss to `val_loss = 27.2630` (`ppl = 6.92e11`) vs defended `val_loss = 3.0128` (`ppl = 20.34`). |

---

## 8. What Remains Unproven

To keep claims strictly defensible, the following properties are **not** proven by this crate and require follow-on work on multi-node GPU infrastructure:

1. **Real 7B–70B Multi-Day GPU Pretraining Convergence**: The live executable training runs validate exact LoRA backpropagation and two-level convergence on a small causal Zen Transformer (`5,776` LoRA params, `140.7 KB` grounded corpus). Scaling trajectories for 4B/9B/27B are analytical projections, not multi-day GPU pretraining curves.
2. **Severe Non-IID Internet Data Skew**: Our grounded corpus shards are deterministically striped across workers (near-IID). Strongly non-IID domain partitions across untrusted operators increase client drift and can raise false-positive rates on single-batch validation spot checks.
3. **Real Residential NAT Traversal, Packet Loss & QUIC/libp2p Transport**: Live tests run over Linux loopback TCP with simulated RTT/bandwidth shaping rather than real wide-area UDP/QUIC hole-punching, asymmetric ISP upload throttling, or BGP flap conditions.
4. **Covert Gradient-Leakage Privacy**: Unmasked INT8 pseudo-gradients can leak memorized training examples under gradient-inversion attacks. Differential privacy (DP-SGD) or hardware enclave aggregation is required for confidential training data.
5. **Economic Slashing & Sybil Collusion Games**: When > `f` workers collude (e.g., `>= 50%` Sybil identities), coordinate-wise trimmed mean and median norm gates lose statistical robustness; protocol-level stake weighting, bond slashing, and commit-reveal audit committees remain outside this crate.

---

## 9. Reproducibility Commands

```bash
# Formatting & strict zero-warning lint check
cargo fmt -p hanzo-diloco-train -- --check
cargo clippy -p hanzo-diloco-train --all-targets --offline -- -D warnings

# Full unit + integration test suite (30 tests)
cargo test -p hanzo-diloco-train --offline --release

# Regenerate deterministic results.json + results.json.sha256
cargo run -p hanzo-diloco-train --release --offline --bin diloco-bench -- \
  --seed 1350 --output-dir docs/research/hip-1350-permissionless-internet-training

# Verify cryptographic digest of results.json
sha256sum -c docs/research/hip-1350-permissionless-internet-training/results.json.sha256
```
