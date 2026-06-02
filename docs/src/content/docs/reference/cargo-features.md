---
title: Cargo features
description: Feature flags for the hanzo workspace crates.
sidebar:
  order: 11
---

hanzo uses Cargo features to gate platform-specific and optional functionality.

## Accelerator features

| Feature | Crates | Purpose |
|---|---|---|
| `cuda` | `hanzo-cli`, `hanzo`, `hanzo-engine`, `hanzo-http` | NVIDIA GPU support via CUDA. |
| `cudnn` | as above | cuDNN-accelerated kernels. |
| `flash-attn` | as above | Flash attention v2 (Ampere+, requires `cuda`). |
| `flash-attn-v3` | `hanzo-cli`, `hanzo-engine`, `hanzo-http` | Flash attention v3 (Hopper, requires `cuda`). Not exposed by the top-level `hanzo` crate. |
| `metal` | as above | Apple Silicon GPU support via Metal. |
| `accelerate` | as above | Apple Accelerate framework for CPU math. |
| `mkl` | as above | Intel MKL for CPU math. |
| `nccl` | `hanzo-cli` | NCCL multi-GPU support. |

Typical combinations:

- NVIDIA Hopper: `cuda flash-attn flash-attn-v3 cudnn`
- NVIDIA Ampere or Ada: `cuda flash-attn cudnn`
- NVIDIA older: `cuda cudnn`
- Apple Silicon: `metal accelerate`
- Intel CPU with MKL: `mkl`

## Functional features

| Feature | Crates | Purpose |
|---|---|---|
| `code-execution` | `hanzo-cli`, `hanzo`, `hanzo-engine`, `hanzo-http` | Python code execution tool. In `hanzo-cli` defaults. |
| `ring` | as above | Multi-machine ring distributed inference. |
| `swagger-ui` | `hanzo-http` | Mounts Swagger UI on the HTTP server. |

## Enabling features

From `cargo install`:

```bash
cargo install hanzo-cli --features "cuda flash-attn cudnn"
```

From a source checkout:

```bash
cargo install --path hanzo-cli --features "cuda flash-attn cudnn"
```

In a consumer crate depending on `hanzo`:

```toml
[dependencies]
hanzo = { version = "0.8", features = ["cuda", "flash-attn", "cudnn"] }
```

## Default features

`hanzo-cli`'s default feature is `code-execution`. To exclude it, use `--no-default-features`.

Other crates enable no accelerator features by default. Opt in to the accelerator matching your hardware.

## Feature verification

`hanzo doctor` prints a `Build features:` line listing compiled-in features.
