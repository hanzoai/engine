---
title: PagedAttention
description: Block-based KV caching in hanzo.
sidebar:
  order: 5
---

hanzo's paged attention splits the KV cache into fixed-size blocks drawn from a central pool, instead of allocating one contiguous slab per sequence.

## Block size

The default is 32 tokens per block. Supported block sizes on CUDA are 8, 16, and 32, values outside that set fail to load because the attention kernel dispatches on block size explicitly. Override with `--pa-block-size`.

## Memory budget

The pool's size is set via one of three mutually exclusive flags:

- `--pa-memory-mb <n>`: budget in MB.
- `--pa-memory-fraction <f>`: fraction of available VRAM.
- `--pa-context-len <n>`: budget sized to hold the configured maximum concurrent sequences at that context length.

When none are set, the engine defaults to 90% of available VRAM on CUDA.

## Block lifecycle

Each sequence holds a list of block pointers. On each decoding step, the scheduler checks whether the sequence has a free slot in its tail block; if not, it allocates a new block from the pool. When a sequence finishes, its blocks return to the pool.

Shared-prefix optimization: sequences that begin with identical tokens share the blocks holding those tokens. A shared block is reference-counted rather than duplicated. This is the mechanism behind prefix caching when paged attention is on.

### Prefix-cache isolation

Every request carries a cache salt, and every block hash includes it, so two requests share cached KV only when their salts are equal. The server builds the salt from the `X-Org-Id` header, which the gateway sets from the validated caller, together with the optional `cache_salt` body field (`/v1/chat/completions`, `/v1/completions`). The two parts are encoded together, so a body salt narrows sharing within an org and never reaches another org's blocks. `/v1/messages` and `/v1/responses` take the org alone. A request with neither shares only with other unsalted requests. Trust in `X-Org-Id` comes from the gateway, so the engine must not be reachable by clients that could set it themselves.

The sequence-level prefix cache used without paged attention is keyed the same way, and a salted prefix is dropped rather than spilled to disk on eviction, because a spill file is keyed by tokens alone.

A block becomes reusable only once every slot holds KV a forward wrote: the last sampled token's slot is written by the next forward, and a sequence preempted before its first forward caches nothing.

## Cache types

`--pa-cache-type` sets the KV cache's numeric representation:

- `auto` (default): match the model's compute dtype.
- Explicit quantization types reduce the per-block memory cost at some quality cost.

This is separate from model-weight quantization (`--quant` or `--isq`). Weight and cache quantization are chosen independently.

## Composition with flash attention

Paged attention and flash attention are separate layers of the runtime.

For non-paged attention, the `flash-attn` and `flash-attn-v3` Cargo features enable the usual flash attention kernels when the model and GPU support them.

For CUDA paged attention, Hanzo Engine can use a FlashInfer-backed paged KV-cache layout and paged decode kernel directly. Eligible paged prefill chunks also use FlashInfer. When a request falls outside those constraints, the runtime falls back to the generic paged path, which gathers blocks and then dispatches to the available attention backend.

FlashInfer paged kernels are built as part of the `cuda` feature. They do not require the `flash-attn` Cargo feature.

## Prompt chunking

Long CUDA prompts with paged attention are processed in chunks. The default chunk size is 4096 tokens. Chunking keeps prefill memory use stable and lets long prompts use the paged prefill kernels instead of one very large prompt pass.

Chunking is internal and does not change the visible prompt, logits, or generated text.

## CUDA graphs

CUDA decode graphs can replay supported single-token paged decode steps with lower CPU launch overhead. They are enabled by default for supported CUDA paged decode paths. Set `HANZO_CUDA_GRAPHS=0` to disable them.

See [Use CUDA graphs](/hanzo/guides/perf/use-cuda-graphs/) for requirements and benchmarking guidance.

## Default behavior

The engine enables paged attention by default for server workloads and disables it for single-request CLI use. Paged attention has a gather overhead versus a contiguous cache. Explicit control is via `--paged-attn auto|on|off`.

MLA's latent cache is supported through a dedicated kernel path. Some models can opt out of FlashInfer paged cache layout when their cache shape needs a different backend.

## See also

- Guide: [use paged attention](/hanzo/guides/perf/use-paged-attention/).
- Reference: [CLI `--pa-*` flags](/hanzo/reference/cli/).
