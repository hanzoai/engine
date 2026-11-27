# HIP-1350: Peer-to-Peer Distributed LLM Inference (`hanzoai/engine#70`)

- **Tracking Issue**: `hanzoai/engine#70` (*"Peer-to-peer distributed LLM inference"*)
- **Implementation Crate**: [`hanzo-p2p-infer`](../../../hanzo-p2p-infer/) (clean-room, zero-dependency, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`, Apache-2.0)
- **Benchmark Receipt**: [`results.json`](./results.json) (SHA-256: `17e01b8cf19f45be7b86c029cbb88b77ee5c1881afefbaa8795885f36f5e9848`, verified in [`results.json.sha256`](./results.json.sha256))
- **Acceptance Gate**: **PASS** (`acceptance_gate_passed: true`) — live 3-node loopback TCP pipeline tolerates mid-generation hard-kill of `peer-node-b` (`8.001 ms` recovery, `23` KV-replayed tokens, `100.0%` completion, `16/16` bit-exact tokens, `0.000000` max abs logit error vs single-node oracle); `6/6` (`100.0%`) adversarial tamper/forgery/stall faults detected; `3/3` negative regimes explicitly quantified and `REJECTED`; `hanzo_coordinator_required: false`.

---

## 1. Problem Statement, Primary References & Clean-Room License Audit

### Problem
Serving large hybrid Transformer + Gated DeltaNet (GDN) + Mixture-of-Experts (MoE) models across independently administered machines requires solving five coupled systems challenges without introducing a mandatory centralized coordinator:
1. **Heterogeneous Pipeline Partitioning**: Consumer/prosumer accelerators (`24–32 GiB` VRAM) cannot hold `27B-Hybrid` (`46.340 GiB` BF16 resident memory at `B = 1, S = 2048`) in a single card. Conventional single-host PCIe Gen4 CPU/RAM layer offload streams `23.84 GiB/token` over `25 GB/s` PCIe, collapsing decode to `0.95 tok/s`.
2. **Interconnect Awareness (LAN vs Metro vs Internet)**: Pipeline and MoE routing behaves fundamentally differently across **Local LAN / RDMA (`0.2 ms` RTT, `25 Gbps`)**, **Metro Fiber (`5 ms` RTT, `10 Gbps`)**, and **Public Internet (`50 ms` RTT, `200 Mbps`)**. Interactive `B = 1` decode across three `50 ms` internet hops hits an unavoidable `>= 150 ms/token` RTT floor (`4.69 tok/s`), while naive cross-WAN per-layer MoE all-to-all routing triggers `128` remote hops/token (`0.30 tok/s`).
3. **Live Mid-Stream Disconnect Fault Tolerance**: Independent peers drop unexpectedly. A 3-node pipeline must detect OS-level TCP socket failure/stalls, promote a surviving peer, reconstruct exact stage KV state via activation replay, and finish generation without corrupting tokens.
4. **Defensible Trust & Tamper Detection**: Untrusted peers can corrupt activations in transit, substitute cheaper weights, forge receipts, replay prior sessions, or return silently tampered floats with valid wire framing. Verification claims must strictly separate cryptographic commitments + authenticated peer receipts + spot-check re-execution from zkML proofs and hardware TEE attestations.
5. **Honest Privacy Boundaries**: Keeping `embed_tokens` and `lm_head` on the client plus encrypting transport links prevents passive wiretapping and raw token-ID logging, but does **not** eliminate white-box activation-inversion risk against an untrusted intermediate-stage peer operator without TEE or FHE.

### Academic & Provenance Attribution (Clean-Room License Audit)
All code in `hanzo-p2p-infer` is a **100% independent clean-room Apache-2.0 Rust implementation** with `[dependencies]` completely empty (Rust `std` only):
- **Petals** (Borzunov et al., ACL 2023, MIT): studied decentralized swarm pipeline partitioning, activation compression, and mid-stream server-failure KV-cache recovery via past-input replay; independent clean-room Rust implementation.
- **Exo** (Apache-2.0): studied heterogeneous device-capability discovery and memory-proportional ring layer partitioning across consumer machines; independent clean-room Rust implementation.
- **GPipe / 1F1B Pipeline Parallelism** (Huang et al., NeurIPS 2019): standard `(S - 1) / (M + S - 1)` microbatch pipeline bubble analysis.

Zero external crates, third-party source files, or non-permissive code were imported.

---

## 2. Methodology, Grounded Corpus & Exact Reproduction

### Grounded Repository Corpus (Verified SHA-256)
All prompt tokens for live multi-socket TCP execution and tamper testing are deterministically derived (`--seed 1350`, SplitMix64) from **3 versioned repository files** (`140,742` bytes total):
- `calibration_data/calibration_datav3_small.txt` (`133,837` bytes, SHA-256 `aabd956dba34ff31f40e78b9de4e234fdd4865f90ba231e8e3deec8dfb22cbc0`)
- `chat_templates/llama3.json` (`420` bytes, SHA-256 `67d1d402738727aa3059d28590fa3488c44740a358be4b5c2135c60f58ec9d7a`)
- `orderings/xlora-paper-ordering.json` (`6,485` bytes, SHA-256 `8d0650ea231eeade0b97e5c026411365ff9feed41b2b07ecf10853c2a399043a`)
- **Combined Grounded Corpus SHA-256** (verified byte-exact against `sha256sum`): `84b45bc2b79cb4612938566807c5eb8b478824036f382307c869cb112a8cd856`

### Recorded Environment & Honesty Boundary
- **Host CPU / OS**: `x86_64` — `Intel(R) Xeon(R) CPU @ 2.30GHz` (2 vCPUs), `Linux 6.18.57 x86_64`
- **Compiler / Base Commit**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, base commit `df172295`
- **Honesty Boundary**:
  - **Directly Measured Live Multi-Socket TCP on Host CPU (`directly_measured_live_tcp_cpu`)**: All 3-node distributed forward passes, BF16 vs INT8 wire byte counts, bit-exact single-node oracle parity (`0.000000` max abs logit diff), genuine `WorkerNode::hard_kill()` socket teardown + KV replay recovery latency (`8.001 ms`), and all `6/6` adversarial fault detections run over real OS loopback `TcpListener` / `TcpStream` sockets executing a real 6-layer deterministic Transformer + Top-2 MoE model with live per-layer KV caches.
  - **Exact Unclamped Resident vs Streamed Accounting**: Aligned verbatim with the shared HIP-1346/1350 hybrid matrix (`hanzo-paged-attn/tests/kivi_kv_quant.rs:450-478`): `4B-Hybrid` (`9` Attn + `27` GDN, `num_kv_heads = 2`, `27.0 MiB` GDN), `9B-Hybrid` (`10` Attn + `30` GDN, `num_kv_heads = 4`, `45.0 MiB` GDN), `27B-Hybrid` (`16` Attn + `48` GDN, `num_kv_heads = 4`, `192.0 MiB` GDN). VRAM capacity charges the full resident input-embedding table (`vocab_size * d_model * 2`), while per-step HBM bandwidth streams non-embedding weights plus one `embed_tokens` row lookup (`d_model * 2`) per token.
  - **Explicitly Tagged Analytical Projections (`analytical_gpu_network_projection`)**: Multi-GPU `TTFT`, `tok/s`, `p95/p99 ITL`, and `$/1M tokens` across H100-SXM5 80 GiB, consumer 24 GiB PCIe Gen4 offload, Local LAN/RDMA, Metro Fiber, and Public Internet are first-principles roofline + network propagation/serialization models.

### Exact Offline Reproduction Commands
```bash
# 1. Formatting and zero-warning Clippy verification (offline)
cargo fmt -p hanzo-p2p-infer -- --check
cargo clippy -p hanzo-p2p-infer --all-targets --offline -- -D warnings

# 2. Full 23-test suite (2 unit + 21 integration: live 3-node TCP execution,
#    mid-stream hard-kill failover, 6 adversarial trust faults, MoE routing, 3 negative regimes)
cargo test -p hanzo-p2p-infer --offline --release

# 3. Reproducible HIP-1350 benchmark and SHA-256 receipt generation
cargo run -p hanzo-p2p-infer --release --offline --bin p2p-infer-bench -- \
  --seed 1350 --output-dir docs/research/hip-1350-p2p-distributed-inference
```

---

## 3. Crate Architecture (`hanzo-p2p-infer/`) & Versioned Hanzo Integration

- [`src/sha256.rs`](../../../hanzo-p2p-infer/src/sha256.rs): Pure-Rust FIPS 180-4 SHA-256, RFC 2104 HMAC-SHA256, constant-time digest equality, hex encoding, and SplitMix64 PRNG.
- [`src/model.rs`](../../../hanzo-p2p-infer/src/model.rs): Canonical `4B / 9B / 27B` `HybridModelSpec` matrix (explicit resident vs streamed weight accounting) plus executable deterministic `MicroModel`, `TransformerMoeLayer`, `LayerSlice`, `StageKvCache`, `WeightsCommitment`, BF16 round-to-nearest-even codec, and symmetric per-tensor INT8 codec.
- [`src/transport.rs`](../../../hanzo-p2p-infer/src/transport.rs): Length-prefixed binary TCP framing (`HZP2` v1, `4 MiB` DoS guard), `ActivationCodec`, `StageReceipt`, `WireFrame` (`HealthPing/Pong`, `ForwardRequest/Response`, `ReplayKvBatch/Ack`, `ResetSession/Ack`, `ErrorFrame`), and explicit `TrustTier` separation.
- [`src/partition.rs`](../../../hanzo-p2p-infer/src/partition.rs): Capability-aware heterogeneous pipeline planner (`plan_heterogeneous_pipeline`), disconnect re-planner (`replan_after_disconnect`), `NetworkTier` physics (`LocalLanRdma`, `MetroFiber`, `PublicInternet`, `PcieGen4Offload`), and versioned `HanzoP2PManifest` (`v1`).
- [`src/router.rs`](../../../hanzo-p2p-infer/src/router.rs): Network-aware MoE expert routing analysis (`LocalityAwareStageResident` vs `NaiveCrossNetworkAllToAll`) and `ExpertAffinityTracker`.
- [`src/node.rs`](../../../hanzo-p2p-infer/src/node.rs): Live independent multi-socket TCP `WorkerNode` (`TcpListener::bind("127.0.0.1:0")`), session backpressure enforcement, genuine `hard_kill()` socket teardown, on-demand shard mounting, and 6 adversarial `FaultMode` variants.
- [`src/cluster.rs`](../../../hanzo-p2p-infer/src/cluster.rs): Live multi-node `P2pPipelineClient` with persistent sockets, OS TCP error/timeout detection, automatic failover + `ReplayKvBatch` reconstruction, cryptographic receipt verification, and configurable `SpotCheck` re-execution.
- [`src/bench.rs`](../../../hanzo-p2p-infer/src/bench.rs) & [`src/bin/p2p-infer-bench.rs`](../../../hanzo-p2p-infer/src/bin/p2p-infer-bench.rs): End-to-end HIP-1350 benchmark driver and deterministic JSON receipt generator.

### Versioned Hanzo Stack Interoperability (`HanzoP2PManifest v1`)
- `schema_version`: `"hanzo.p2p.infer/v1"`
- `engine_compat`: `"hanzo-engine/1.7"`
- `enso_route_id`: `"enso://hanzo.p2p/27b-hybrid-v1"` (Enso decentralized routing topic)
- `visor_sandbox_profile`: `"visor-seccomp-net-egress-allowlist-v1"` (Visor worker isolation)
- `vfs_shard_uris`: `["vfs://hanzo-p2p/27b-hybrid/layers-000-022.safetensors", "vfs://hanzo-p2p/27b-hybrid/layers-022-043.safetensors", "vfs://hanzo-p2p/27b-hybrid/layers-043-064.safetensors"]`
- `trust_tier`: `"commitment_receipt_and_spot_check"`
- `hanzo_coordinator_required`: **`false`** (100% independently operable via permissionless peer discovery & direct TCP transport)

---

## 4. Directly Measured Live Multi-Socket TCP 3-Node Execution & Failover (`directly_measured_live_tcp_cpu`)

Executing the deterministic 6-layer Transformer + Top-2 MoE model (`d_model = 64`, `4` heads, `4` experts, `16` grounded-corpus prompt tokens, `16` generated tokens) across **3 independent live loopback TCP worker nodes** (`peer-node-a: 0..2`, `peer-node-b: 2..4`, `peer-node-c: 4..6`):

| Live TCP Experiment | Wall Time (ms) | Decode (tok/s) | Wire Bytes | Exact Token Match | Max Abs Logit Diff | Extra Verification / Failover Telemetry |
|---|---:|---:|---:|---:|---:|---|
| Single-Node Monolithic Oracle | `11.320` | `1,413.48` | `0` | `16 / 16` (ref) | `0.000000` | Monolithic in-process reference oracle |
| Live 3-Node TCP (`BF16` Lossless) | `52.929` | `828.89` | `55,776` | **`16 / 16`** | **`0.000000`** | `96` HMAC-SHA256 receipts verified, `24` spot checks passed |
| Live 3-Node TCP (`INT8` Per-Tensor) | `32.092` | `994.40` | `44,256` | **`16 / 16`** | cos `0.999958` | `1.882x` activation payload compression vs BF16 |
| **Live 3-Node Hard-Kill `peer-node-b` @ tok 8** | **`32.120`** | **`1,071.86`** | `55,776 + 3,125` | **`16 / 16`** | **`0.000000`** | **`8.001 ms` failover recovery**, `23` tokens KV-replayed onto surviving `peer-node-c`, **`100.0%` completion** |

**Key live finding**: Hard-killing `peer-node-b` mid-generation (genuinely closing its OS TCP listener and shutting down active sockets after token 8) triggers immediate OS socket error detection, dynamic promotion of `layers 2..4` onto surviving `peer-node-c`, deterministic replay of all `23` prior stage-1 input activations (`16` prompt + `7` prior decode steps, `3,125` wire bytes) in `8.001 ms`, and bit-exact completion of all `16/16` tokens (`0.000000` max abs logit error vs the single-node oracle).

---

## 5. Live Adversarial Trust & Tamper Detection Suite (`TrustTier::CommitmentReceiptAndSpotCheck`)

Declared threat model: independent peer operators may drop frames, tamper with activations in transit, silently corrupt output numerics while signing valid frames, load wrong model weights, forge receipts, replay prior sessions, or stall connections. **This tier explicitly does NOT claim zero-knowledge validity proofs or hardware TEE confidentiality.**

| Fault Class Injected on Live TCP `trust-b` | Detected | Observed Runtime Outcome |
|---|:---:|---|
| `tampered_activation_wire_bytes` | YES | `rejected:ActivationIntegrityMismatch` |
| `silent_numeric_tamper_spot_check` | YES | `rejected:SpotCheckNumericMismatch(max_diff=0.352)` |
| `wrong_weights_commitment` | YES | `rejected:WeightsCommitmentMismatch` |
| `forged_hmac_receipt_mac` | YES | `rejected:ForgedReceiptMac` |
| `cross_session_receipt_replay` | YES | `rejected:ReceiptBindingMismatch` |
| `stalled_peer_timeout_failover` (`180 ms` stall vs `60 ms` timeout) | YES | `recovered:TimeoutFailover(replacement=trust-c)` |

- **Detection / Recovery Rate under Declared Threat Model**: **`6 / 6` (`100.00%`)**.

---

## 6. Analytical Multi-Node Network-Aware Roofline Benchmarks (`analytical_gpu_network_projection`)

Evaluated across the canonical `4B / 9B / 27B` hybrid matrix (`seq_len = 2048`), explicitly separating full resident input-embedding VRAM (`vocab_size * d_model * 2`) from streamed per-step non-embedding weights plus single-token `embed_tokens` row gather (`d_model * 2`). P2P costs include `+5%` spot-check verification and `+1%` decentralized settlement overhead (`1.06x`).

| Regime ID | Model | Link Tier | `B` / `M` | Nodes | Resident / Streamed (GiB) | Wire B/tok | Bubble Loss | TTFT (ms) | Agg Decode (tok/s) | Per-User (tok/s) | p95 / p99 ITL (ms) | 1-Node Survival | Cost ($/1M tok) | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `single_node_h100_80gb_b1` | 27B-Hybrid | Single H100 HBM | `1 / 1` | 1 | `46.34 / 44.58` | `0` | `0.0%` | `198.7` | `53.35` | `53.35` | `19.7 / 21.0` | `0.0%` | `$13.017` | ACCEPTED |
| `single_node_h100_80gb_b16` | 27B-Hybrid | Single H100 HBM | `16 / 1` | 1 | `51.03 / 44.58` | `0` | `0.0%` | `2,905.1` | `733.07` | `45.82` | `22.9 / 24.5` | `0.0%` | `$0.947` | ACCEPTED |
| `conventional_pcie_offload_24gb_b1` | 27B-Hybrid | PCIe Gen4 Offload | `1 / 1` | 1 | `46.34 / 44.58` | `23.84 GiB` | `0.0%` | `1,618.1` | **`0.95`** | `0.95` | `1,137 / 1,211` | `0.0%` | `$122.822` | ACCEPTED |
| `p2p_3node_local_lan_bf16_b1` | 27B-Hybrid | Local LAN `0.2 ms / 25 Gbps` | `1 / 1` | 3 | `46.34 / 44.58` | `61,440` | `66.7%` | `581.0` | **`15.94`** | `15.94` | `66.5 / 70.3` | **`100.0%`** | `$23.270` | ACCEPTED |
| `p2p_3node_local_lan_int8_b16` | 27B-Hybrid | Local LAN `0.2 ms / 25 Gbps` | `16 / 4` | 3 | `51.03 / 44.58` | `30,744` | `33.3%` | `560.9` | **`124.09`** | `15.43` | `68.7 / 72.6` | **`100.0%`** | **`$2.990`** | ACCEPTED |
| `p2p_3node_local_lan_9b_b1` | 9B-Hybrid | Local LAN `0.2 ms / 25 Gbps` | `1 / 1` | 3 | `16.94 / 15.69` | `49,152` | `66.7%` | `222.9` | `43.94` | `43.94` | `24.1 / 25.5` | `100.0%` | `$8.444` | ACCEPTED |
| `p2p_3node_local_lan_4b_b1` | 4B-Hybrid | Local LAN `0.2 ms / 25 Gbps` | `1 / 1` | 3 | `6.61 / 5.85` | `30,720` | `66.7%` | `91.6` | `108.36` | `108.36` | `9.8 / 10.3` | `100.0%` | `$3.424` | ACCEPTED |
| `p2p_3node_metro_fiber_int8_b16` | 27B-Hybrid | Metro Fiber `5 ms / 10 Gbps` | `16 / 4` | 3 | `51.03 / 44.58` | `30,744` | `33.3%` | `605.5` | `111.58` | `12.62` | `90.3 / 99.9` | `100.0%` | `$3.325` | ACCEPTED |
| `p2p_3node_public_internet_int8_b16_pipelined` | 27B-Hybrid | Public Internet `50 ms / 200 Mbps` | `16 / 4` | 3 | `51.03 / 44.58` | `30,744` | `33.3%` | `3,208.7` | `56.49` | `4.56` | `289.2 / 346.1` | `100.0%` | `$6.567` | ACCEPTED |
| `p2p_3node_failover_degraded_2surviving_lan` | 27B-Hybrid | Local LAN (2 surviving nodes) | `16 / 4` | 2 | `51.03 / 44.58` | `20,496` | `20.0%` | `553.9` | `99.78` (`-19.6%`) | `15.53` | `68.3 / 72.1` | **`100.0%`** | `$2.479` | ACCEPTED |

**Key quantitative takeaways**:
- **vs Conventional PCIe Offload**: 3x independent `24 GiB` GPUs on Local LAN hold `27B-Hybrid` fully resident in distributed VRAM (`0` bytes PCIe offload), delivering **`15.94 tok/s` vs `0.95 tok/s` (`16.8x` faster at `B = 1`)** and **`124.09 tok/s` (`130x` higher aggregate throughput) at `B = 16, M = 4`**.
- **vs Single-Node H100**: Centralized H100-SXM5 80 GiB remains faster (`53.35 tok/s` at `B = 1`, `733.07 tok/s` at `B = 16`, `$0.947/1M tok`) for centralized datacenter workloads, but has `0%` single-node crash tolerance and requires expensive datacenter-class 80 GiB GPUs. P2P wins specifically when pooling commodity `24 GiB` machines, eliminating single-operator dependency, or operating without centralized cloud infrastructure.
- **Batching & Pipeline Bubble Losses**: At `B = 1` across `S = 3` stages, `66.7%` of cluster GPU capacity sits idle in the pipeline bubble (`(S - 1) / (M + S - 1)`). Microbatch pipelining (`B = 16, M = 4`) cuts bubble loss to `33.3%`, scaling LAN aggregate throughput from `15.94` to `124.09 tok/s` (`7.8x`) and lifting Public Internet aggregate throughput from `4.69` to `56.49 tok/s` (`12.0x`).

---

## 7. Honest Negative Regimes (`3 / 3 REJECTED`)

Peer-to-peer execution does **not** beat centralized inference on every workload. The benchmark explicitly quantifies and rejects three failure regimes:

| Negative Regime ID | Quantified Failure Metric | Verdict | Why P2P Loses |
|---|---|:---:|---|
| `neg_internet_b1_decode_rtt_wall` | `4.69 tok/s`, `p95 = 281.6 ms`, `p99 = 337.1 ms`, `$79.15/1M tok` (`66.7%` bubble) | **REJECTED** | `B = 1` interactive decode over 3 public-internet hops (`50 ms` RTT) hits an unavoidable `>= 150 ms/tok` pure RTT floor (`11.4x` slower than single H100). |
| `neg_naive_moe_alltoall_wan` | `128` remote WAN hops/tok, `1.78 MB/tok`, `0.30 tok/s`, `p95 = 4,122 ms`, `$1,223.52/1M tok` | **REJECTED** | Naive cross-WAN per-layer MoE expert all-to-all dispatch pays `2 * L` WAN hops per token, collapsing throughput by `> 180x` vs locality-aware stage-resident experts. |
| `neg_plaintext_activation_privacy_exposure` | `activation_inversion_risk_score = 0.87` (`shallow_layer_2_of_64`, `d_model = 5120`), `wire_eavesdrop_protection = 1.00` (TLS only) | **REJECTED** | Client-held embeddings/LM-head + TLS stop passive sniffing, but do **not** stop an adversarial intermediate-stage peer from running white-box activation-inversion attacks. Confidential/regulated workloads require TEE or FHE. |

---

## 8. HIP-1350 5-Axis Evaluation Summary

1. **Cost**: `$2.990 / 1M useful tokens` on 3-node Local LAN (`B = 16, M = 4`, including `+5%` spot-check verification + `+1%` settlement overhead) vs `$122.822 / 1M` on conventional single-card PCIe offload (`41x` cheaper) and `$0.947 / 1M` on centralized datacenter H100 (`3.16x` decentralization premium at `B = 16`).
2. **Performance**: Live 3-node TCP `16/16` bit-exact (`0.000000` max abs diff), `8.001 ms` live mid-stream `Node B` hard-kill recovery, `100.0%` completion rate; analytical `124.09 tok/s` (`LAN B=16`), `111.58 tok/s` (`Metro B=16`), `56.49 tok/s` (`Internet B=16`), and honest rejection of `B = 1` Internet (`4.69 tok/s`).
3. **Trust**: `6 / 6` (`100.0%`) detection/recovery of wire bit-flips, silent numeric tampering (caught by spot check), wrong weight commitments, forged HMAC receipts, cross-session receipt replay, and stalled peers under `TrustTier::CommitmentReceiptAndSpotCheck` — without conflating zkML or TEE guarantees.
4. **Decentralization**: `100.0%` of partitioning, peer health probing, activation transport, failover recovery, and receipt verification proceeds with `hanzo_coordinator_required = false`.
5. **Sovereignty**: Zero-dependency clean-room Apache-2.0 crate, versioned `hanzo.p2p.infer/v1` manifest, content-addressed `vfs://` shard URIs, exportable signed stage receipts, and full operator fork/migration independence.

---

## 9. What Remains Unproven

To maintain strict scientific and engineering honesty, the following items are **not** proven by this crate and require future live infrastructure validation:
1. **Real Multi-Continent WAN Jitter, Packet Loss & NAT Traversal**: Loopback TCP validates framing, socket error handling, timeouts, and KV replay, but does not measure real internet BGP jitter, TCP head-of-line blocking under packet loss, QUIC/UDP transport migration, or ICE/STUN/TURN NAT hole-punching across residential firewalls.
2. **Live Multi-GPU Kernel & Communication Overlap (NCCL / UCX / GPUDirect RDMA)**: GPU latency, TTFT, and throughput tables for `4B / 9B / 27B` are analytical roofline projections; real CUDA/Metal/ROCm stream overlap between asynchronous activation send/recv and GDN/attention kernel execution must be benchmarked on physical multi-GPU hardware before wiring into `hanzo-engine`.
3. **White-Box Activation-Inversion Privacy Without TEE/FHE**: Intermediate hidden-state tensors (`d_model = 5120`) retain recoverable semantic and lexical information, particularly at shallow cut points. Defending against curious-but-honest or colluding peer operators on regulated data requires hardware TEE enclaves (confidential GPUs) or homomorphic/MPC split protocols, which are outside this crate's scope.
4. **Sybil-Resistant Economic Slashing & Collusion Resistance**: HMAC-SHA256 stage receipts plus spot-check re-execution detect single-peer dishonesty once challenged, but full game-theoretic incentive design (bonded stakes, committee challenge routing, anti-collusion auditing, and on-chain dispute settlement) requires integration with the external settlement layer.
5. **Direct Integration into `hanzo-engine` Serving Loops**: Wiring `hanzo-p2p-infer` into `hanzo-engine/src/pipeline/` and `hanzo-router` for live GGUF/safetensors streaming weights remains gated on multi-host GPU CI validation.
