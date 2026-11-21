# HIP-1345 Status Update — Sparse SSD KV Offload (Bounty #11, engine#92)

Proposed status text for HIP-1345 §2.1 (KV cache reuse / sparse SSD offload,
Bounty #11 — SolidAttention), per the live HIP text at
`hanzoai/hips/HIPs/hip-1345-agent-aware-model-serving-and-cache-reuse.md`.

## Proposed §2.1 addendum (status note)

> **§2.1 Status — sparse SSD KV offload policy (Bounty #11, fabric layer).**
> A SolidAttention-style offload policy is implemented in `hanzo-kv-fabric`
> (`ssd_offload` module): coarse-grained 64 MiB KV blocks, tenant-salted
> block identity (`salted_block_key`, SplitMix64 — HIP-1345 §1.1 CacheSalt
> isolation holds at the offload tier), an LRU staging pool under a 512 MiB
> budget, a bounded temporal-locality prefetch predictor, and a per-block
> completion overlap model (prefetch I/O issued at step j hides under the
> cumulative decode compute of steps j..i−1; only the residual stalls at
> the consuming step; demand fetches stall in full). On a deterministic
> 128k-token trace (64 blocks × 64 MiB = 4 GiB KV, 6 resident, window-3
> sparse attention with an attention-sink re-read), speculative prefetch
> cuts decode wall time **1.13×** (2 GB/s SSD) / **1.46×** (500 MB/s SSD)
> vs demand fetch, with total SSD bytes identical and exactly one wasted
> 64 MiB prefetch reported. A depth-1 oracle (perfect next-step prediction)
> is *slower* than the predictor — prefetch **depth**, not prediction
> accuracy, dominates I/O hiding in this regime. Raw numbers:
> `results/solid-attention-results.json` in the PR. These are
> fabric-policy-layer simulation figures on the authoring host; the
> upstream SolidAttention figures (up to 3.1× inference speedup, ~98%
> memory-footprint reduction) remain the original authors' own and are NOT
> treated as Hanzo expectations until reproduced on Hanzo hardware.

## What this PR proves

- Speculative prefetch with a deliberately simple recency predictor hides
  SSD block reads under decode compute in the AIPC regime (6 ms of
  attention compute per 64 MiB block vs ~34 ms of NVMe read): decode wall
  time improves 1.13×–1.46× while total SSD bytes stay constant.
- The per-block completion model charges residual I/O latency to the
  consuming step only — a global pending-I/O drain model double-counts
  in-flight I/O as stall and was rejected during implementation
  (documented in the results JSON).
- **Depth beats accuracy**: a depth-1 oracle with *perfect* prediction is
  slower than the imperfect predictor, because one step of lookahead
  cannot hide a 34–134 ms read under one 18 ms step. The predictor wins by
  issuing at window-epoch start (implicit depth up to 8). Engine
  integrations should maximize prefetch depth, not predictor accuracy.
- Waste is measured, not hidden: exactly one 64 MiB block is prefetched
  and never used (block 12, issued at step 88) and is reported in the
  results (`unused_staged_bytes` + eviction accounting).

## Explicit out-of-scope statement

This PR does **not** cover:

1. **Sparse attention algorithms** — which blocks attention actually reads
   is the engine's job; the access trace is an input here.
2. **Answer-quality regression** (a bounty acceptance item) — there is no
   attention path at the fabric policy layer; this requires the engine
   integration and is listed as follow-up work.
3. **SolidAttention leg 3 (fine-grained kernel-level compute/I-O
   orchestration)** — only the bounded per-block overlap model is here.
4. **Real device I/O** — configured rates, not measured; the
   Hanzo-hardware rerun must replace them.
5. **Engine integration** — decode-loop hooks, host paging, CUDA graphs,
   multi-tenant concurrency (CacheSalt keying is provided; the concurrency
   policy is engine work).

## Follow-ups for §2.1 completion

- Engine integration: wire `SsdOffloadPool` into the local Zen decode loop
  with measured device rates and measure decoded tokens/sec, p99 decode
  stalls, and answer-quality regression on a real model.
- Depth-tuned prefetch issue policy (issue at earliest legal point —
  this PR's depth finding suggests the win is in issue timing, not
  predictor sophistication).
