---
title: Performance
description: Get the most out of the hardware you have. Quantization, attention kernels, multi-GPU, and auto-tuning.
---

Guides for tuning throughput, memory, and latency.

## Choose by constraint

| If you need to... | Start here |
|---|---|
| Fit a model into less memory | [Pick a quantization method](/hanzo/guides/perf/pick-a-quantization/) |
| Let hanzo benchmark the host | [Let the tune command decide for you](/hanzo/guides/perf/auto-tune/) |
| Improve attention throughput on NVIDIA GPUs | [Use flash attention](/hanzo/guides/perf/use-flash-attention/) |
| Improve high-concurrency serving memory use | [Use paged attention](/hanzo/guides/perf/use-paged-attention/) |
| Split one model across local GPUs | [Multi-GPU tensor parallelism](/hanzo/guides/perf/multi-gpu-tensor-parallel/) |
| Split one model across machines | [Multi-machine inference with the ring backend](/hanzo/guides/perf/multi-machine-ring/) |
| Place layers manually | [Topology](/hanzo/guides/perf/topology/) |
| Reduce decode latency with MTP | [Speculative decoding](/hanzo/guides/perf/speculative-decoding/) |
| Use Gemma 4 assistant checkpoints for MTP | [Gemma 4 MTP](/hanzo/guides/perf/gemma4-mtp/) |
| Save an ISQ result for faster reloads | [UQFF for pre-quantized models](/hanzo/guides/perf/use-uqff/) |

Underlying concepts (paged attention design, what quantization changes, MLA) live in the [Explanation](/hanzo/explanation/) section.
