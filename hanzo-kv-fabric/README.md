# `hanzo-kv-fabric`

Epoch-fenced KV block ownership directory and framed TCP transport (`HKV1`) for disaggregating or replicating KV cache blocks across Hanzo Engine workers.

## Features

- **`EpochFencedKvDirectory`**: Monotonic per-worker epoch registration and stale-block fencing so restarted or preempted replicas never serve stale KV pages.
- **`KvBlockTransport`**: Framed `HKV1` async TCP push/pull transport with SHA-256 block integrity verification, epoch fencing, and a pluggable `KvStorageBackend` trait for host RAM or RDMA-pinned arenas.
- **`kv_block_hash`**: Deterministic 64-bit XXH3 content addressing over `(layer_group, token_ids)`.

## Example & Benchmark

```bash
cargo run --release -p hanzo-kv-fabric --example kv_fabric_bench
```
- **`SsdOffloadPool`** (`ssd_offload` module): sparse SSD KV offload policy per HIP-1345 §2.1 (SolidAttention, engine#92) — coarse 64 MiB blocks, tenant-salted block keys (§1.1 CacheSalt), LRU staging under a 512 MiB budget, temporal-locality speculative prefetch, and a per-block completion overlap model. Demand fetch is the matched baseline; a depth-1 oracle probe shows prefetch depth dominates hiding. Sparse attention algorithms and answer quality are out of scope (engine work).
