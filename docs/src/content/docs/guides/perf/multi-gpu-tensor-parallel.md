---
title: Single-machine multi-GPU
description: NCCL tensor parallelism and layer/P2P mapping on one host.
sidebar:
  order: 7
---

When a model exceeds one GPU's memory after quantization, hanzo can split it across multiple GPUs on the same host.

**Tensor parallelism** splits each layer across all GPUs and uses NCCL collectives to combine partial results. This is the preferred CUDA multi-GPU mode when the model supports it.

**Layer mapping** places different layer ranges on different devices. It is the fallback when NCCL is unavailable, disabled, or not suitable for the selected model. CUDA layer mapping enables peer access (P2P) for GPU pairs that support it; otherwise boundary activations are staged through CPU.

## Default selection

With no manual mapping flags:

1. One visible GPU runs the whole model on that GPU.
2. Multiple visible CUDA GPUs use NCCL tensor parallelism when the binary was built with `cuda nccl` and `MISTRALRS_NO_NCCL` is not set.
3. If NCCL is unavailable or disabled, mistral.rs uses layer mapping across the visible GPUs.

The selected layout is printed in the startup logs.

## Build requirements

Linux CUDA installs enable `nccl` when the installer or wheel builder finds `libnccl`.

Manual Linux CUDA build with NCCL:

```bash
hanzo serve -m Qwen/Qwen3-32B --quant 4
```

If NCCL is not installed, omit `nccl`:

```bash
cargo install mistralrs-cli --features "cuda flash-attn cudnn"
```

To force the installer decision, use `MISTRALRS_INSTALL_NCCL=1` or `MISTRALRS_INSTALL_NO_NCCL=1`. To disable NCCL at runtime without rebuilding:

```bash
MISTRALRS_NO_NCCL=1 mistralrs serve -m Qwen/Qwen3-32B --quant 4
```

## Select GPUs

Use `CUDA_VISIBLE_DEVICES` to restrict the GPU set before mistral.rs starts:

```bash
CUDA_VISIBLE_DEVICES=0,1 hanzo serve -m Qwen/Qwen3-32B --quant 4
```

The ordinals in `--device-layers` are the visible ordinals after `CUDA_VISIBLE_DEVICES` is applied.

NCCL tensor parallelism uses all visible CUDA GPUs. The tensor-parallel size must be compatible with the model:

- Attention heads must divide evenly across GPUs.
- KV heads must either divide evenly across GPUs or be replicated evenly when there are fewer KV heads than GPUs.

If the visible GPU count is incompatible, mistral.rs errors instead of selecting a smaller subset.

Use `CUDA_VISIBLE_DEVICES` to choose a compatible subset.

## Manual layer mapping

`-n`/`--device-layers` assigns layer counts to devices. Format:

```bash
hanzo serve -n "0:32;1:32" -m <model>
```

For per-tensor or per-layer placement, see the [topology guide](/hanzo/guides/perf/topology/).

```bash
mistralrs serve -n "0:44;1:20" -m Qwen/Qwen3-32B --quant 4
```

For cross-machine splitting, see the [ring backend guide](/hanzo/guides/perf/multi-machine-ring/).
