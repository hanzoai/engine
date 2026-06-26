---
title: Quantization types
description: Every runtime ISQ type hanzo supports, what hardware it works on, and how it compares.
sidebar:
  order: 13
---

ISQ types supported by hanzo. For normal CLI usage, use `--quant`; use `--isq` only when you want to force runtime ISQ and skip the UQFF lookup. For format selection guidance, see the [quantization decision guide](/hanzo/guides/perf/pick-a-quantization/). For underlying tradeoffs, see [the explanation page](/hanzo/explanation/quantization-tradeoffs/).

## Numeric shorthands

Pass `--quant N` for the normal CLI path. If it falls back to runtime ISQ, or if you pass `--isq N` directly, hanzo resolves N to a format based on the detected backend.

| Shorthand | Metal resolves to | CUDA / CPU resolves to |
|---|---|---|
| `2` | AFQ2 | Q2K |
| `3` | AFQ3 | Q3K |
| `4` | AFQ4 | Q4K |
| `5` | Q5K | Q5K |
| `6` | AFQ6 | Q6K |
| `8` | AFQ8 | Q8_0 |

## Format-specific types

### AFQ family (Metal-only)

Adaptive float quantization, Metal backend only.

| Type | Bits |
|---|---|
| `afq2` | 2 |
| `afq3` | 3 |
| `afq4` | 4 |
| `afq6` | 6 |
| `afq8` | 8 |

Loading AFQ on CUDA or CPU returns an error.

### Q*K family (CUDA and CPU)

GGML K-quant formats. Supported on all backends.

| Type | Bits |
|---|---|
| `q2k` | 2 |
| `q3k` | 3 |
| `q4k` | 4 |
| `q5k` | 5 |
| `q6k` | 6 |

### Legacy GGML types

Supported for GGUF compatibility:

| Type | Bits |
|---|---|
| `q4_0`, `q4_1` | 4 |
| `q5_0`, `q5_1` | 5 |
| `q8_0`, `q8_1` | 8 |

GGUF files using these types load correctly.

### ROCm native GGUF decode (RDNA3.5 APUs)

On AMD `gfx1151` (Strix Halo / Radeon 8060S) the **full GGUF quant zoo decodes resident** -- weights stay quantized, no dequant-to-f16 -- through a single unified compute core, every type bit-exact:

| Family | Types |
|---|---|
| K-quants | `q2k` `q3k` `q4k` `q5k` `q6k` |
| Legacy | `q4_0` `q4_1` `q5_0` `q5_1` `q8_0` `q8_1` |
| I-quants | `iq1_s` `iq1_m` `iq2_xxs` `iq2_xs` `iq2_s` `iq3_xxs` `iq3_s` `iq4_xs` `iq4_nl` |
| Ternary | `tq1_0` `tq2_0` |

Adding a format is one decode function + one traits row (no per-format kernel). See [ROCm GGUF decode on RDNA3.5 APUs](/hanzo/guides/perf/rocm-apu-gguf/) for the architecture and measured performance vs llama.cpp.

### FP8

Native FP8 on NVIDIA GPUs with compute capability 8.9+.

| Type | Bits | Layout |
|---|---|---|
| `fp8` | 8 | E4M3 (4-bit exponent, 3-bit mantissa) |
| `f8q8` | 8 | FP8 weights, INT8 activations |

### MXFP4

4-bit microscaling format. Native on Blackwell; emulated elsewhere.

| Type | Bits |
|---|---|
| `mxfp4` | 4 |

### HQQ

Half-quadratic quantization.

| Type | Bits |
|---|---|
| `hqq4` | 4 |
| `hqq8` | 8 |

## GPTQ and AWQ

Not ISQ types, pre-quantized formats. Load directly when a Hugging Face model is available as GPTQ or AWQ:

```bash
hanzo run --format plain -m <gptq-or-awq-repo>
```

hanzo detects the quantization from the model's config. No `--quant` or `--isq` required.

See the [pick-a-quantization guide](/hanzo/guides/perf/pick-a-quantization/) for format selection.
