# HIP-1345 Status Update — Model-Loading IO Templates (Bounty #14, engine#95)

Proposed status text for HIP-1345 §2.4 (Model-Loading IO Templates &
Multi-Tenant LoRA Paging, Bounties #14/#15), per the live HIP text at
`hanzoai/hips/HIPs/hip-1345-agent-aware-model-serving-and-cache-reuse.md`.
Companion deliverable: the LoRA paging half of §2.4 (Bounty #15, S-LoRA)
is already covered by engine PR #104 (`adapter-paging-research`); this PR
is the weight-loading template half (Bounty #14, PPC/MAIO).

## Proposed §2.4 addendum (status note)

> **§2.4 Status — weight-loading IO templates (Bounty #14, template
> layer).** A PPC/MAIO-style load-template system is implemented in
> `hanzo-bench` (`load_templates` module): 16 MiB weight-page identity,
> canonical page access orders for three checkpoint layouts
> (contiguous / sharded-strided / mmap-random permutation), a recorded
> `LoadTemplate` structure, and two replay policies — a synchronous
> serial demand baseline and a two-channel pipelined replay (NVMe
> producer stages into a bounded host RAM window while the PCIe consumer
> streams pages onto the device; the producer is gated so staged-but-
> not-copied pages stay within the window), plus a head pre-warm policy
> that stages the first 2 GiB of the learned order in the idle window
> before serving. On a deterministic 8 GiB (512-page) workload across
> three storage/link scenarios, the streaming pipeline improves cold
> load 1.02×–1.85× and head pre-warm 1.35×–2.11× vs demand, with the
> break-even for the pre-warm window at ONE replayed serving session in
> all nine scenario×layout pairs. The occupancy column reports when the
> staging window actually binds (pinned at budget in the link-bound
> contiguous case; a natural 74-page lead under budget for sharded;
> one page when the consumer drains faster than the producer stages).
> Raw numbers: `results/load-templates-results.json` in the PR. These
> are template-layer simulation figures on the authoring host; the
> upstream PPC/MAIO figure (up to 79% model-load latency reduction)
> remains the original authors' own and is NOT treated as a Hanzo
> expectation until reproduced on Hanzo hardware with measured device
> rates.

## What this PR proves

- The two-channel pipeline converts serial NVMe→host→device time
  (Σread+Σcopy) into max(Σread, Σcopy)+startup — the win is real only
  when the two channel times are comparable, and the results include the
  regime map: 1.58× balanced (contiguous, NVMe 7 / PCIe 12 GB/s),
  1.02×–1.06× storage-bound (SATA), 1.43×–1.85× link-bound (PCIe 3).
- Checkpoint layout dominates absolute cold-load latency through its
  NVMe access pattern (configured 1/2 strided and 1/4 random effective
  bandwidth): demand spans 1.94 s–69.79 s across layouts in the SATA
  scenario while the pipeline ratio varies only 1.02×–1.58×. Layout
  choice matters more than pipelining for random-order formats.
- Head pre-warm amortizes immediately: the 2 GiB window costs
  0.31 s–17.18 s depending on scenario and saves 1.02 s–18.25 s per
  replayed session — break-even is one session everywhere measured.
- The staging-window gate is measured, not asserted: occupancy stays
  within budget by construction and the results show exactly the three
  regimes (pinned at budget / natural lead under budget / fully drained).

## Explicit out-of-scope statement

This PR does **not** cover:

1. **The actual IO engine** — no io_uring/Overlapped I/O, CUDA host
   staging, mmap syscalls, or tensor-level placement; this is the
   template + policy layer with a deterministic timeline recurrence.
2. **Learned predictors** — the template replays a recorded order; drift
   and staleness handling are follow-ups.
3. **Quantization-aware page sizing / weight file formats themselves.**
4. **Multi-model concurrent pre-warm scheduling and template eviction.**
5. **Real device measurements** — configured rates until the
   Hanzo-hardware rerun.
6. **LoRA paging (Bounty #15)** — the other half of §2.4, covered by
   engine PR #104.

## Follow-ups for §2.4 completion

- Engine integration: record real cold-load access templates from
  `hanzo-engine` model loads (weight formats actually in use), wire the
  two-channel replay into the loader with measured device rates, and
  report cold-load latency across real weight formats on Hanzo hardware.
- Template drift handling: invalidate/refresh recorded orders when a
  model update changes the access pattern.
