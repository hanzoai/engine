---
title: ROCm GGUF decode on RDNA3.5 APUs
description: Native-resident GGUF quant decode on AMD Strix Halo (gfx1151) -- the complete quant zoo and the unified compute core behind it.
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

## Measured performance

Throughput figures for this backend are published one claim per page, each rendering
its own run (engine build, flags, host, repetition count, spread and date) from the
benchmark endpoint:
[prefill by prompt length](/hanzo/benchmarks/prefill-by-prompt-length/),
[decode throughput](/hanzo/benchmarks/decode-throughput/),
[time to first token](/hanzo/benchmarks/time-to-first-token/) and
[four concurrent clients](/hanzo/benchmarks/four-client-concurrency/).

Decode here is memory-bound rather than compute-bound, so lower-bit quantizations
decode proportionally faster on the same silicon. That is a property of the memory
system, and the claim pages are where the numbers for it live.

## Running it

```bash
# Force all layers onto the iGPU (the auto device-mapper otherwise offloads to CPU on UMA APUs)
hanzo run -n "0:48" --max-seq-len 4096 --format gguf -m /path/to/models -f Qwen3-30B-A3B-Q4_K_M.gguf -i "..."
```

The whole quant zoo is selected automatically from the GGUF metadata -- no flags. HIP graphs are on by default for the decode loop.
