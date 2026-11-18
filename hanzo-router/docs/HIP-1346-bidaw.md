# HIP-1346 Research and Bounded Implementation: KV-Residency-Aware Request Scheduling (Bidaw)

## 1. Primary Reference and License Audit

- Primary paper: Shipeng Hu et al., "Bidaw: KV-Residency-Aware Request Scheduling", USENIX FAST 2026 (`https://www.usenix.org/conference/fast26/presentation/hu-shipeng`).
- Tracking issue: `hanzoai/engine#86` (HIP-1346).
- License audit: Clean-room Rust implementation authored directly against `hanzo-router`'s existing `ReplicaSet` and `SchedulerState` data structures (`MIT OR Apache-2.0`). Zero external paper artifacts, datasets, or third-party dependencies were imported; `hanzo-router/Cargo.toml` retains its existing dependency set (`serde`, `serde_json`, `serde_yaml`, `sha2`).

## 2. Upstream Baseline

In `hanzo-router/src/scheduler.rs`, `ReplicaSet::pick_agent` ranks healthy candidate replicas with a lexicographic tuple led unconditionally by `std::cmp::Reverse(measured)` (GPU-cached prefix tokens reported via `observe_completion`). Two baseline behaviors degrade tail latency and goodput on mixed conversational workloads:

1. Short prefixes (`128` to `1,024` tokens) cached on a heavily loaded replica (`3/4` slots occupied) force subsequent turns onto that congested replica even when cold prefill on an idle higher-weight replica finishes faster than queueing wait.
2. Prefixes evicted from GPU HBM into host DRAM or local NVMe report `gpu_tokens == 0`, causing the baseline router to treat offloaded prefixes as cold misses and route by static weight to an engine that must recompute the full prompt from scratch.

## 3. Falsifiable Hypothesis and Failure Modes

- Hypothesis: Ranking candidate replicas (and overloaded sticky session pins at `inflight >= slots`) by quantized expected TTFT (`kv_load_ms + prefill_compute_ms + queue_wait_ms`) across HBM, host DRAM, and local NVMe tiers reduces tail TTFT and increases `<= 1s` goodput on mixed warm/cold conversational workloads without adding `> 25 us` per routing decision.
- Expected failure mode (bounded by fallback): When a storage tier's page-in bandwidth degrades below prefill compute throughput (`nvme_kv_load_tok_per_s < prefill_tok_per_s`), naively paging KV blocks from storage increases TTFT relative to cold recomputation. `BidawCostModel::estimate` bounds each tier by `min(load_ms, recompute_ms)` so degraded storage tiers never outrank faster compute replicas.

## 4. Bounded Implementation (`hanzo-router`)

- `hanzo-router/src/bidaw.rs`: Introduces `KvTierResidency`, `BidawCostModel`, and `TieredEstimate` with finite/positive input sanitization and deterministic quantum bucketing (`bucket()`).
- `hanzo-router/src/scheduler.rs` and `hanzo-router/src/lib.rs`: Adds opt-in `ReplicaSet::set_bidaw_model`, `ReplicaSet::observe_tiered_residency`, and `ReplicaSet::estimate_replica_ttft`. Default `ReplicaSet::new` leaves `bidaw: None`, preserving exact upstream behavior across all existing router unit tests while respecting `PREFIX_TTL`, `MAX_ENTRIES`, and health-transition cache invalidation.

## 5. Correctness, Concurrency, Security, and Regression Evidence

- `cargo test -p hanzo-router` passes `53/53` tests (`43` library unit tests, `1` rules-path latency benchmark, `4` `bidaw_eval` integration/trace tests, `4` YAML policy tests, `1` doctest).
- Concurrency and bounds: 16-thread barrier stress test racing `observe_tiered_residency` and `pick_agent`, `MAX_ENTRIES` eviction enforcement, health-flip residency invalidation, and `NaN` / negative / infinite parameter sanitization.

## 6. Before and After Measurements

Reproduce locally with `cargo test -p hanzo-router --test bidaw_eval -- --nocapture`:

| Metric (600-request mixed warm/cold deterministic trace) | Upstream Greedy-Prefix | Bidaw Cost-Scheduled |
|---|---|---|
| Median TTFT (`p50`) | `377.2 ms` | `351.2 ms` |
| Tail TTFT (`p99`) | `11,135.2 ms` | `11,118.6 ms` |
| Goodput (`TTFT <= 1,000 ms`) | `70.5%` | `72.0%` |
| Routing decision overhead | `7.52 us/req` | `16.76 us/req` |
| Short-prefix (128 tok) on 3/4-busy Evo (`600 ms` EWMA) | Routes to busy `evo` (`~450 ms` wait) | Spills to idle `spark` (`~200 ms` prefill) |
| Deep 24k prefix in host DRAM (`12k tok/s`) vs cold prefill (`2.55k tok/s`) | Cold recompute (`9,411 ms`) | Host DRAM page-in (`2,000 ms`, `4.7x` faster) |
| Degraded NVMe (`400 tok/s`) negative control | Stalls on slow storage if unguarded | Falls back to compute (`kv_load_ms = 0.0`) on `spark` |

### Integration Recommendation

Keep `BidawCostModel` opt-in on `ReplicaSet` until `hanzo-server` / `hanzo-kv-fabric` emit per-tier (`gpu_tokens`, `host_tokens`, `nvme_tokens`) usage telemetry in completion headers alongside `cached_tokens`, preserving canonical `vfs -> Hanzo S3 -> Visor` and Cloud router boundaries.
