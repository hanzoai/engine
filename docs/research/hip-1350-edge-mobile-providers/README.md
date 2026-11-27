# HIP-1350 Research Dossier: Decentralized Edge & Mobile Inference Providers (`hanzoai/engine#71`)

- **Issue**: `hanzoai/engine#71` (*"Decentralized edge and mobile inference providers"*)
- **Crate**: `hanzo-edge-provider` (clean-room, zero external crate dependencies, Apache-2.0, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`)
- **Reproducible Seed**: `1350`
- **Grounded Corpus Combined SHA-256**: `84b45bc2b79cb4612938566807c5eb8b478824036f382307c869cb112a8cd856` (`140,742` bytes across 3 versioned repo files)
- **Canonical Deterministic SHA-256**: `74e117167143b23b1de6a0238f7213bf0cda01f365531d36935847ab1426b7d7`
- **Committed `results.json` SHA-256**: `6bacc0c36c4b36d16c8f561c40e756f0f30416952c49be895f0950160ce3ecde`
- **Acceptance Gate**: **PASS** (`29 / 29` tests passing, `7 / 7` fraud cases rejected, `4 / 4` withdrawal guardrails verified live, `3 / 3` negative regimes rejected)

---

## 1. Reproducing & Verifying This Dossier

```bash
# Run formatting, lints, and release test suite (7 unit + 22 integration = 29 tests)
cargo fmt -p hanzo-edge-provider -- --check
cargo clippy -p hanzo-edge-provider --all-targets --offline -- -D warnings
cargo test -p hanzo-edge-provider --offline --release

# Regenerate results.json and verify SHA-256 receipt
cargo run -p hanzo-edge-provider --release --offline --bin edge-provider-bench -- \
  --seed 1350 --output-dir docs/research/hip-1350-edge-mobile-providers
sha256sum -c docs/research/hip-1350-edge-mobile-providers/results.json.sha256
```

### Grounded Repository Corpus

All live TCP request prompts are deterministically derived from three versioned repository files mixed with `--seed 1350`:

| Repository File | Size (bytes) | SHA-256 |
|---|---:|---|
| `calibration_data/calibration_datav3_small.txt` | `133,837` | `aabd956dba34ff31f40e78b9de4e234fdd4865f90ba231e8e3deec8dfb22cbc0` |
| `chat_templates/llama3.json` | `420` | `67d1d402738727aa3059d28590fa3488c44740a358be4b5c2135c60f58ec9d7a` |
| `orderings/xlora-paper-ordering.json` | `6,485` | `8d0650ea231eeade0b97e5c026411365ff9feed41b2b07ecf10853c2a399043a` |

---

## 2. Protocol & Runtime Architecture

`hanzo-edge-provider` enables owner-authorized laptops, Apple Silicon workstations, AMD Strix Halo unified-memory systems, and mobile-class devices to contribute bounded open-weight Zen inference work over a permissionless peer market without sacrificing device safety, battery life, user memory headroom, or privacy:

1. **Unified-Memory Residency Accounting (`src/device.rs`)**: Separates full resident BF16 input-embedding table capacity (`vocab_size * d_model * 2` bytes) from single-token decode row lookups (`d_model * 2` bytes) alongside quantized non-embedding weights, BF16 attention KV cache, unquantized FP32 Gated DeltaNet (GDN) recurrent state, and mandatory user/OS reserved RAM. Oversized models are rejected cleanly without clamping.
2. **First-Principles RC Thermal & Power Governor (`src/thermal.rs`)**: Integrates the lumped thermal ODE `dT/dt = (P - (T - T_amb) / R_th) / C_th`, enforces progressive DVFS clock/power throttling (`Full -> ThermalThrottle70 -> ThermalThrottle45 -> AutoWithdraw`) with `3°C` cooling hysteresis, tracks battery state-of-charge off AC, and enforces all 4 owner-protection automatic withdrawal guardrails:
   - **Guardrail 1**: Thermal ceiling exceeded (`95°C` die / `46°C` mobile skin)
   - **Guardrail 2**: Battery floor reached off AC (`20%–30%` device-specific floor)
   - **Guardrail 3**: Availability window expired or user foreground activity preemption
   - **Guardrail 4**: Free unified RAM dropped below mandatory user-reserved headroom (`5.5–32.0 GiB`)
3. **Self-Sovereign Manifests & Explicit Trust Separation (`src/manifest.rs`)**: Versioned `EdgeProviderAdvertisement v1` (`hanzo.edge.provider/v1`) and signed `MarketWithdrawalNotice` carry `hanzo_coordinator_required = false`, content-addressed `vfs://` weight URIs, Enso routing topics, Visor sandbox profiles, and explicit `TrustTier::SignedTelemetryAndSpotCheck` — strictly separating cryptographic receipt/spot-check verification from self-reported/OS-sensor thermal and power telemetry (never conflated with zkML or hardware TEEs).
4. **Executable Micro-Model & Independent Receipt Auditor (`src/meter.rs`)**: Deterministic 4-layer hybrid Attention + GDN causal micro-model (`model_weights_digest = 68b7b232789ec5c002b169c3e7b34a214655b5d6578c003b9c07f25d34b58e3b`), RFC 2104 HMAC-SHA256 `MeteredWorkReceipt` signing, physical energy conservation checks (`joules ≈ mean_watts * duration_s`), replay tracking, and deterministic spot-check re-execution.
5. **Live Loopback TCP Wire Protocol & Peer Market Router (`src/runtime.rs`, `src/market.rs`)**: Zero-dependency `HZED` v1 length-prefixed binary TCP framing (`4 MiB` DoS guard), independent `EdgeProviderNode` workers, and decentralized `PeerMarketRouter` performing admission checks, privacy filtering, live receipt verification, withdrawal recording, and automatic handover to surviving peers.

---

## 3. Canonical `4B / 9B / 27B` Unified-Memory Residency Matrix (`INT4`, `ctx = 2048`)

Exact layer counts and unquantized FP32 GDN recurrent state sizes (`27.0 MiB` for `4B-Hybrid`, `45.0 MiB` for `9B-Hybrid`, `192.0 MiB` for `27B-Hybrid`) align with the canonical HIP-1346/1350 hybrid matrix:

| Device Class | Total / Reserved / Usable RAM | Model (`INT4`) | Resident Embed (GiB) | Non-Embed Weights (GiB) | Attn KV + GDN + Workspace (GiB) | Total Required (GiB) | Headroom (GiB) | Admission |
|---|---|---|---:|---:|---:|---:|---:|:---:|
| `apple_silicon_m4_pro_24g` | `24 / 8.0 / 16.0 GiB` | `4B-Hybrid` | `0.7245` | `1.6437` | `0.6387` | `3.0068` | `+12.9932` | **FITS** |
| `apple_silicon_m4_pro_24g` | `24 / 8.0 / 16.0 GiB` | `9B-Hybrid` | `1.1592` | `4.4129` | `0.8059` | `6.3780` | `+9.6220` | **FITS** |
| `apple_silicon_m4_pro_24g` | `24 / 8.0 / 16.0 GiB` | `27B-Hybrid` | `1.4490` | `12.5364` | `1.3719` | `15.3573` | `+0.6427` | **FITS** |
| `apple_silicon_m4_max_64g` | `64 / 16.0 / 48.0 GiB` | `4B-Hybrid` | `0.7245` | `1.6437` | `0.6387` | `3.0068` | `+44.9932` | **FITS** |
| `apple_silicon_m4_max_64g` | `64 / 16.0 / 48.0 GiB` | `9B-Hybrid` | `1.1592` | `4.4129` | `0.8059` | `6.3780` | `+41.6220` | **FITS** |
| `apple_silicon_m4_max_64g` | `64 / 16.0 / 48.0 GiB` | `27B-Hybrid` | `1.4490` | `12.5364` | `1.3719` | `15.3573` | `+32.6427` | **FITS** |
| `amd_strix_halo_128g` | `128 / 32.0 / 96.0 GiB` | `4B-Hybrid` | `0.7245` | `1.6437` | `0.6387` | `3.0068` | `+92.9932` | **FITS** |
| `amd_strix_halo_128g` | `128 / 32.0 / 96.0 GiB` | `9B-Hybrid` | `1.1592` | `4.4129` | `0.8059` | `6.3780` | `+89.6220` | **FITS** |
| `amd_strix_halo_128g` | `128 / 32.0 / 96.0 GiB` | `27B-Hybrid` | `1.4490` | `12.5364` | `1.3719` | `15.3573` | `+80.6427` | **FITS** |
| `mobile_class_12g` | `12 / 5.5 / 6.5 GiB` | `4B-Hybrid` | `0.7245` | `1.6437` | `0.3887` | `2.7568` | `+3.7432` | **FITS** |
| `mobile_class_12g` | `12 / 5.5 / 6.5 GiB` | `9B-Hybrid` | `1.1592` | `4.4129` | `0.5559` | `6.1280` | `+0.3720` | **FITS** |
| `mobile_class_12g` | `12 / 5.5 / 6.5 GiB` | `27B-Hybrid` | `1.4490` | `12.5364` | `1.1219` | `15.1073` | **`-8.6073`** | **REJECTED** |

---

## 4. Live Loopback TCP Serving, Progressive DVFS Throttling & Automatic Handover

Two live independent community peer nodes (`mac-m4-pro-community` on `AppleSiliconM4Pro24G` under restricted-airflow lap thermal resistance `R_th = 5.5 °C/W`, and `amd-strix-halo-community` on `AmdStrixHalo128G` at nominal `R_th = 0.76 °C/W`) register signed `EdgeProviderAdvertisement v1` manifests over live loopback TCP against `PeerMarketRouter` (`hanzo_coordinator_required = false`) and serve a 12-request open-weight Zen burst (`16` prompt tokens, `12` generated tokens each, `ZeroRetentionEphemeral`):

| Metric | Measured Live Value (`results.json`) |
|---|---:|
| Completed / Total Requests | **`12 / 12` (`100.00%`)** |
| Requests Served by `mac-m4-pro-community` -> `amd-strix-halo-community` | **`8` -> `4`** (`1` live automatic handover) |
| Signed Receipts Verified / Deterministic Spot Checks Passed | **`12 / 12`** (`100%`) |
| `mac-m4-pro-community` Peak Temperature / Max Throttle State | **`95.066 °C`** (`auto_withdraw`) |
| `amd-strix-halo-community` Peak Temperature (Surviving Peer) | **`73.837 °C`** (`full`) |
| Signed Withdrawal Reason Emitted | `thermal_ceiling_exceeded` |
| Direct Post-Withdrawal TCP Probe Rejected (`RejectedWithdrawn`) | **`true`** |
| Live Wall-Clock ITL Progression (`Full` -> `Throttle70` -> `Throttle45`) | **`0.0433 ms` -> `0.2792 ms` -> `0.8378 ms`** |
| Live Wall-Clock Mean TTFT / p95 ITL / Total Burst Duration | `0.8129 ms` / `0.6967 ms` / `87.562 ms` |
| Modeled Cumulative Energy (Mac / AMD) | `42.9917 J` / `82.9600 J` |
| Independent Automatic Withdrawal Guardrails Verified Live | **`4 / 4`** (thermal, battery, window/foreground, memory) |

### Live Request-by-Request Thermal & Handover Trace

| Request | Serving Peer | Peak Temp (°C) | Mean Power (W) | DVFS State | Wall TTFT (ms) | Wall Mean ITL (ms) | Withdrawal Notice |
|---|---|---:|---:|---|---:|---:|:---:|
| `zen-edge-req-00` | `mac-m4-pro-community` | `52.405` | `30.000` | `full` | `0.7901` | `0.0438` | `false` |
| `zen-edge-req-01` | `mac-m4-pro-community` | `66.752` | `30.000` | `full` | `0.6095` | `0.0408` | `false` |
| `zen-edge-req-02` | `mac-m4-pro-community` | `79.591` | `30.000` | `full` | `0.5195` | `0.0422` | `false` |
| `zen-edge-req-03` | `mac-m4-pro-community` | `88.252` | `24.037` | `thermal_throttle_70` | `0.6254` | `0.2792` | `false` |
| `zen-edge-req-04` | `mac-m4-pro-community` | `91.201` | `15.995` | `thermal_throttle_45` | `0.9258` | `0.6544` | `false` |
| `zen-edge-req-05` | `mac-m4-pro-community` | `92.635` | `14.700` | `thermal_throttle_45` | `1.2684` | `0.6967` | `false` |
| `zen-edge-req-06` | `mac-m4-pro-community` | `93.918` | `14.700` | `thermal_throttle_45` | `1.2666` | `1.1623` | `false` |
| `zen-edge-req-07` | `mac-m4-pro-community` | **`95.066`** | `14.700` | **`auto_withdraw`** | `1.1862` | `0.6852` | **`true`** |
| `zen-edge-req-08` | `amd-strix-halo-community` | `45.720` | `85.000` | `full` | `0.8276` | `0.0401` | `false` |
| `zen-edge-req-09` | `amd-strix-halo-community` | `58.546` | `85.000` | `full` | `0.5602` | `0.0464` | `false` |
| `zen-edge-req-10` | `amd-strix-halo-community` | `67.536` | `85.000` | `full` | `0.5546` | `0.0431` | `false` |
| `zen-edge-req-11` | `amd-strix-halo-community` | `73.837` | `85.000` | `full` | `0.6213` | `0.0466` | `false` |

---

## 5. Adversarial Fraud & Policy Verification Suite (`7 / 7`, `100.0%`)

Every adversarial fault is injected against live loopback TCP providers or the independent `ReceiptVerifier` under `TrustTier::SignedTelemetryAndSpotCheck`:

| # | Case ID | Attack Vector | Observed Code | Verdict |
|---:|---|---|---|:---:|
| 1 | `fraud_1_inflated_token_count` | Provider inflates receipt `output_tokens` (`24` claimed vs `8` emitted over TCP) | `inflated_output_token_count` | **REJECTED** |
| 2 | `fraud_2_cheaper_model_substitution_spot_check` | Provider runs cheaper weights while spoofing `model_weights_digest` | `spot_check_model_substitution` | **REJECTED** |
| 3 | `fraud_3_tampered_receipt_hmac_and_energy` | Forged HMAC-SHA256 tag and `3.5x` falsified Joules (`J != W * s`) | `forged_receipt_hmac+energy_accounting_mismatch` | **REJECTED** |
| 4 | `fraud_4_post_withdrawal_request_acceptance` | Zombie provider keeps serving after emitting signed `MarketWithdrawalNotice` | `stale_withdrawn_provider` | **REJECTED** |
| 5 | `fraud_5_oversized_27b_on_12g_mobile_admission` | `MobileClass12G` advertises `27B-Hybrid` INT4 (`15.107 GiB > 6.500 GiB` usable) | `infeasible_memory_advertisement` | **REJECTED** |
| 6 | `fraud_6_privacy_policy_mismatch` | Client demands `ZeroRetentionEphemeral` against `StandardBestEffort` peer | `privacy_policy_violation` | **REJECTED** |
| 7 | `fraud_7_receipt_replay_attack` | Duplicate submission of previously verified `(request_id, session_id)` receipt | `receipt_replay_detected` | **REJECTED** |

---

## 6. HIP-1350 5-Axis Evaluation vs Centralized H100 Baseline

Analytical roofline and economics projections across all 4 edge/mobile profiles vs Centralized `H100-SXM5 80 GiB` (`$3.20/hr`, `700 W` TDP), using `ctx = 2048`, `512`-token prefill, `$0.15/kWh` residential electricity, `+5%` spot-check overhead, `$0.002/1M tok` settlement overhead, and 3-year `20%`-duty-cycle amortized consumer capex:

| Scenario ID | Model / Prec | Batch / Power | Res. / Cap RAM (GiB) | TTFT (ms) | Agg / User Decode (`tok/s`) | p95 ITL (ms) | Steady W / Temp | Energy (`J/1k tok`) | Marginal Cost (`$/1M tok`) | Amortized TCO (`$/1M tok`) | Fraud / Indep. | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `centralized_h100_80gb_9b_b1` | `9B` BF16 | `B=1` SXM5 | `17.93 / 80.0` | `15.90` | `153.85 / 153.85` | `7.02` | `700W / 72.0C` | `4,549.81` | `$5.7775` | `$5.7775` | `0% / 0%` | ACCEPTED |
| `centralized_h100_80gb_9b_b8` | `9B` BF16 | `B=8` SXM5 | `18.51 / 80.0` | `127.18` | `1,165.34 / 145.67` | `7.41` | `700W / 72.0C` | `600.68` | `$0.7628` | `$0.7628` | `0% / 0%` | ACCEPTED |
| `centralized_h100_80gb_27b_b8` | `27B` BF16 | `B=8` SXM5 | `49.52 / 80.0` | `361.03` | `400.80 / 50.10` | `21.56` | `700W / 72.0C` | `1,746.52` | `$2.2178` | `$2.2178` | `0% / 0%` | ACCEPTED |
| `edge_m4_pro_24g_4b_int4_ac` | `4B` INT4 | `B=1` AC | `3.01 / 16.0` | `402.71` | `80.99 / 80.99` | `14.57` | `20.3W / 84.8C` | `250.76` | **`$0.0130`** | `$1.3174` | `100% / 100%` | ACCEPTED |
| `edge_m4_pro_24g_9b_int4_ac` | `9B` INT4 | `B=1` AC | `6.38 / 16.0` | `1,079.84` | `30.58 / 30.58` | `38.59` | `20.3W / 84.8C` | `664.20` | **`$0.0311`** | `$3.4860` | `100% / 100%` | ACCEPTED |
| `edge_m4_max_64g_27b_int4_ac` | `27B` INT4 | `B=1` AC | `15.36 / 48.0` | `1,328.34` | `21.84 / 21.84` | `54.02` | `36.8W / 84.6C` | `1,682.88` | **`$0.0756`** | `$9.5091` | `100% / 100%` | ACCEPTED |
| `edge_amd_strix_halo_128g_27b_int4_ac` | `27B` INT4 | `B=1` AC | `15.36 / 96.0` | `1,188.75` | `9.73 / 9.73` | `121.28` | `56.1W / 81.4C` | `5,768.00` | **`$0.2543`** | `$13.8286` | `100% / 100%` | ACCEPTED |
| `edge_mobile_12g_4b_int4_battery_cap` | `4B` INT4 | `B=1` Battery | `2.76 / 6.5` | `1,102.54` | `26.60 / 26.60` | `42.10` | `3.5W / 37.2C` | `131.56` | **`$0.0078`** | `$1.9923` | `100% / 100%` | ACCEPTED |

### 5-Axis Synthesis

1. **Cost**: On **surplus idle capacity of already-owned personal devices** (zero incremental capex), marginal electricity + `5%` spot-check + settlement cost is **`$0.0078–$0.2543 / 1M tok`** (`8.7x–74x` cheaper than batched cloud H100). Conversely, on **full amortized retail TCO** (`$1.32–$13.83 / 1M tok`), dedicated consumer hardware is **`1.7x–6.2x` more expensive** than batched datacenter H100 (`$0.76–$2.22 / 1M tok`). Edge supply is economically compelling strictly as opportunistic harvesting of already-amortized idle devices.
2. **Performance**: `M4 Pro` (`80.99 tok/s` on `4B` INT4, `30.58 tok/s` on `9B` INT4) and `M4 Max` (`21.84 tok/s` on `27B` INT4) deliver interactive `B = 1` decode at low power (`20–37 W`), and live TCP failover achieves **`100.0%` completion** across thermal withdrawal. Datacenter H100 remains superior for high-batch throughput (`1,165 tok/s` at `B = 8`) and sub-second long-prompt TTFT.
3. **Trust**: `7 / 7` (`100.0%`) adversarial fraud and policy violations rejected under `TrustTier::SignedTelemetryAndSpotCheck`. Thermal/power telemetry is explicitly flagged `telemetry_self_reported = true` and never overstated as hardware attestation or zkML.
4. **Decentralization**: `100.0%` Hanzo-independent (`hanzo_coordinator_required = false`) across peer registration, routing, thermal withdrawal, handover, and receipt auditing.
5. **Sovereignty**: Hardware owners retain unconditional control over reserved RAM (`5.5–32.0 GiB`), thermal ceilings, battery floors, availability windows, foreground preemption, and privacy policy tiers (`ZeroRetentionEphemeral` / `LocalOnlyTiers`).

---

## 7. Honest Negative Regimes (`3 / 3 REJECTED`)

| Negative Regime ID | Device Evaluated | Quantified Failure Metric | Verdict | Why This Regime Fails |
|---|---|---|:---:|---|
| `neg_unthrottled_thermal_runaway` | Apple Silicon M4 Pro (24 GiB) | Unthrottled peak **`114.58 °C` > `95.0 °C`** safety ceiling (DVFS-governed stays at `88.35 °C`) | **REJECTED** | Sustained unthrottled burst decode breaches thin-enclosure thermal limits; progressive DVFS throttling and automatic thermal market withdrawal are mandatory. |
| `neg_oversized_27b_on_12g_mobile` | Mobile-Class SoC (12 GiB) | `27B-Hybrid` INT4 requires **`15.107 GiB` > `6.500 GiB`** usable cap (`-8.607 GiB` deficit, `5.5 GiB` reserved for OS/user) | **REJECTED** | Attempting to load `27B-Hybrid` on a 12 GiB mobile device invades OS/user memory and triggers immediate jetsam/OOM; admission control must reject unclamped. |
| `neg_battery_drain_without_withdrawal` | Mobile-Class SoC (12 GiB) | Unguarded SoC drained `45.0% -> 22.95%` (`< 30.0%` owner floor); guarded exited cleanly at **`30.00%`** after `2,855 s` | **REJECTED** | Serving off AC power without automatic battery-floor withdrawal degrades owner usability and accelerates lithium-ion cycle wear. |

---

## 8. What Remains Unproven

To uphold strict scientific and engineering honesty, this crate and dossier do **not** claim to have proven the following items, each of which requires future live platform and network validation:

1. **Native OS Thermal & Power Sensor Drivers**: The RC thermal network and battery integrator are physics-based software models validated over live loopback TCP. Production edge daemons must bind to native host telemetry APIs — **macOS `IOReport` / Apple SMC** (`powermetrics` equivalents), **Linux `hwmon` / Intel & AMD `RAPL` / `amdgpu` sysfs**, **Android Thermal API** (`PowerManager.OnThermalStatusChangedListener` + `BatteryManager`), and **iOS `ProcessInfo.thermalState`** — and handle OS permission sandboxing cleanly.
2. **Hardware-Attested Power & Thermal Sensors**: Under `TrustTier::SignedTelemetryAndSpotCheck`, `mean_watts`, `peak_temp_c`, and `battery_pct_end` are self-reported / OS-sensor telemetry (`telemetry_self_reported = true`). A malicious host OS can spoof temperature or power readings without breaking `J = W * s` conservation; cryptographic proof of physical energy draw or thermal compliance requires hardware-attested sensor enclaves (TEE-signed telemetry) not yet broadly available on consumer laptops or phones.
3. **Residential NAT Traversal & Mobile Network Churn**: Live TCP tests run on `127.0.0.1` loopback. Real residential and cellular edge nodes sit behind symmetric NAT, carrier-grade NAT (CG-NAT), Wi-Fi/5G handoffs, and asymmetric upload links; production routing requires QUIC/UDP hole-punching, ICE/STUN/TURN relay fallback, and real WAN jitter/loss benchmarks.
4. **Idle-Electricity Economics, Battery Wear & Regional Tariffs**: Marginal cost calculations assume `$0.15/kWh` residential power and zero incremental hardware wear. In high-tariff regions (`$0.35–$0.50/kWh`), continuous lithium-ion battery cycling (replacing a `$99–$199` phone/laptop battery after ~500–1,000 full cycles) and elevated fan/thermal wear can erase thin token margins unless participation is restricted to AC-plugged idle windows.
5. **Direct Integration with `hanzo-engine` Metal / ROCm / Vulkan Backends**: Live TCP execution uses the pure-Rust 4-layer hybrid Attention + GDN micro-model for deterministic cross-platform CI verification. Wiring `EdgeProviderNode` and `ThermalPowerGovernor` directly into `hanzo-engine`'s Metal (Apple Silicon), ROCm/Vulkan (`HSA_XNACK=1` AMD Strix Halo), and mobile GPU serving loops — and measuring real `4B / 9B / 27B` GGUF/safetensors TTFT, unified-memory pressure callbacks, and DVFS clock curves on physical hardware — remains future integration work.

---

## 9. Ecosystem & Academic Context

This work draws on published research and open-source architectural concepts in consumer unified-memory inference, decentralized heterogeneous device pools, and mobile thermal/power governance:
- **Unified-Memory Consumer Clusters**: Distributed home-cluster designs such as *exo* (heterogeneous Apple Silicon / consumer LAN pooling) and local edge runtimes (*llama.cpp*, *Ollama*) demonstrated the viability of quantized inference on personal unified-memory devices, while highlighting the need for strict owner memory reservation and thermal withdrawal guardrails.
- **Decentralized & Volunteer Inference Networks**: Volunteer swarm inference (*Petals*, Borzunov et al., 2023) established peer discovery and failover principles for consumer hardware pools, motivating our permissionless `hanzo.edge.provider/v1` advertisement schema and deterministic spot-check receipt auditing.
- **Mobile & Edge Thermal Governance**: Lumped RC thermal modeling and progressive DVFS skin-temperature capping build on established mobile SoC thermal management literature (Skadron et al., *HotSpot*; Android/iOS thermal mitigation frameworks).
