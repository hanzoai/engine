---
title: ROCm GGUF decode on RDNA3.5 APUs
description: Native-resident GGUF quant decode on AMD Strix Halo (gfx1151) -- the complete quant zoo, the unified compute core, and measured performance vs llama.cpp on the same silicon.
sidebar:
  order: 40
---

Hanzo's ROCm backend runs GGUF-quantized models natively on AMD RDNA3.5 APUs (the Ryzen AI Max+ 395 "Strix Halo", Radeon 8060S iGPU, `gfx1151`) with the weights kept resident in their quantized form -- no dequant-to-f16 round trip. Every supported quant type decodes through one unified, templated compute core, and every format is bit-exact against the CPU reference.

## Complete quant coverage (22 native-resident types)

All of these decode resident on ROCm (matvec + indexed-MoE), bit-exact (`nbad=0` vs the CPU `to_float` oracle):

| Family | Types | Decode path |
|---|---|---|
| K-quants | `Q2_K` `Q3_K` `Q4_K` `Q5_K` `Q6_K` | int8 dp4a |
| Legacy | `Q4_0` `Q4_1` `Q5_0` `Q5_1` `Q8_0` `Q8_1` | dp4a / scalar |
| I-quants | `IQ1_S` `IQ1_M` `IQ2_XXS` `IQ2_XS` `IQ2_S` `IQ3_XXS` `IQ3_S` `IQ4_XS` `IQ4_NL` | codebook int8 dp4a (grid -> int8, `sudot4`) |
| Ternary | `TQ1_0` `TQ2_0` | scalar |

The int8-WMMA prefill GEMM serves the 11 `qmmq_capable` types (`Q8_0` `Q4_0` `Q4_1` `Q5_0` `Q5_1` `Q8_1` `Q4_K` `Q5_K` `Q6_K` plus the dp4a-capable K-quants); decode-only types dequantize-to-f16 for prefill.

## The unified compute core (one way to add a quant)

There is no per-format kernel. Each stage is a single `WTYPE`-templated core:

- **Decode**: `qmatvec_core<WTYPE, XT>` (scalar) and `qdp4a<WTYPE>` (int8), driving both the dense matvec and the batched indexed-MoE matvec.
- **Prefill**: `qmmq_core<WTYPE, MOE, NWAVE_M>` -- the same int8-WMMA machine for dense and fused-expert GEMM.

Adding a format is **one `qdec<WTYPE>::partial` decode function + one `qdw_traits<WTYPE>` row + one `DEFINE_QMATVECU` generation entry** -- zero new kernels. Capability is a single value (`RocmQuantType::qmmq_capable()`), and the type/activation-dtype/MoE/dp4a axes are orthogonal template parameters that compose.

## Performance (Qwen3-30B-A3B, gfx1151, vs llama.cpp on the same GPU)

Measured on a quiet GPU, `pp1024`/`tg128`@d4, native Linux ROCm 7.13. Hanzo numbers are bit-exact.

| | hanzo (ROCm) | llama.cpp HIP | llama.cpp Vulkan |
|---|---|---|---|
| **Prefill** (Q4_K, tok/s) | **1209** | 1071 (1.13x) | 957 (1.26x) |
| Decode Q4_K (tok/s) | 64.7 | 66.2 | 82.8 |
| Decode Q2_K (tok/s) | 75.0 | 79.3 | 91.9 |
| Decode IQ3_XXS (tok/s) | 65.5 | 57.6 | 91.2 |

**Prefill leads both backends** (dense and sparse). **Decode is at llama.cpp-HIP parity** across the zoo. The decode matvec runs at ~90% of the realistic LPDDR5X bandwidth ceiling (~234 GB/s) -- it is memory-bound, not compute-bound, so lower-bit quants decode proportionally faster (the Q4_K -> Q2_K ladder). The residual gap to llama.cpp's Vulkan backend is non-matvec dispatch overhead, not the matvec.

## Running it

```bash
# Force all layers onto the iGPU (the auto device-mapper otherwise offloads to CPU on UMA APUs)
hanzo run -n "0:48" --max-seq-len 4096 --format gguf -m /path/to/models -f Qwen3-30B-A3B-Q4_K_M.gguf -i "..."
```

The whole quant zoo is selected automatically from the GGUF metadata -- no flags. HIP graphs are on by default for the decode loop.
