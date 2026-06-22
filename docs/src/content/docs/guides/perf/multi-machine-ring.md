---
title: Run across multiple machines
description: Use the ring backend for distributed inference across hosts.
sidebar:
  order: 9
---

When a model exceeds one machine's GPU memory, hanzo can split it across multiple hosts via a ring backend.

## Build

The `ring` feature must be compiled in:

```bash
cargo install --path hanzo-cli --features "cuda flash-attn ring"
```

If the binary is also built with `nccl`, set `HANZO_NO_NCCL=1` when launching so `Comm::from_device` selects the ring backend.

## Configuration

The ring backend reads its configuration from a JSON file pointed to by the `RING_CONFIG` environment variable. Each participant has its own `RING_CONFIG` with rank-specific values.

Config shape:

```json
{
  "master_ip": "10.0.0.1",
  "master_port": 9000,
  "port": 9001,
  "right_port": 9002,
  "right_ip": "10.0.0.2",
  "rank": 0,
  "world_size": 3
}
```

Non-master ranks (`rank != 0`) must specify `master_ip`. The master rank (`rank = 0`) is reachable via `master_ip`.

## Environment

Ring backend selection is controlled by `RING_CONFIG`:

| Variable | Purpose |
|---|---|
| `RING_CONFIG` | Path to the per-rank ring JSON config. |
| `MN_GLOBAL_WORLD_SIZE` | Total world size across nodes. |
| `MN_LOCAL_WORLD_SIZE` | Local TP size override on the node. |
| `MN_HEAD_NUM_WORKERS` | Number of worker nodes (set on head). |
| `MN_HEAD_PORT` | Head node port. |
| `MN_WORKER_SERVER_ADDR` | Head node address (set on workers). |
| `MN_WORKER_ID` | Worker node id. |
| `NO_NCCL=1` | Disable NCCL fallback. |

Full env var reference: [environment variables](/hanzo/reference/environment-variables/).

## Notes

The ring backend is Linux-only. For single-machine multi-GPU, prefer NCCL-based [tensor parallelism](/hanzo/guides/perf/multi-gpu-tensor-parallel/).
