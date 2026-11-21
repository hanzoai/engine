# `hanzo-kv-fabric`

Epoch-fenced KV block ownership directory and framed TCP transport (`HKV1`) for disaggregating or replicating KV cache blocks across Hanzo Engine workers.

## Features

- **`EpochFencedKvDirectory`**: Monotonic per-worker epoch registration and stale-block fencing so restarted or preempted replicas never serve stale KV pages.
- **`KvBlockTransport`**: Framed `HKV1` async TCP push/pull transport with SHA-256 block integrity verification, epoch fencing, and a pluggable `KvStorageBackend` trait for host RAM or RDMA-pinned arenas.
- **`kv_block_hash`**: Deterministic 64-bit XXH3 content addressing over `(layer_group, token_ids)`.
- **`SegmentDirectory`** (`segment` module): position-shifted segment reuse per HIP-1345 §2.2 (CacheSlide) — salted, unchained content hashes over whole segments keyed by `(cache_salt, segment_hash)` (§1.1), publish-time token offsets for position translation, and fail-closed admission that rejects empty and incomplete (`num_computed_tokens < num_tokens`) segments (§1.2).

## Example & Benchmark

```bash
cargo run --release -p hanzo-kv-fabric --example kv_fabric_bench
```

Position-shifted segment reuse demo (engine#93 / CacheSlide):

```bash
cargo run --release -p hanzo-kv-fabric --example cacheslide_shift_bench
```
