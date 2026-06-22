use std::sync::OnceLock;

const CUDA_GRAPHS_ENV: &str = "HANZO_CUDA_GRAPHS";
#[cfg(feature = "metal")]
const METAL_GRAPHS_ENV: &str = "HANZO_METAL_GRAPHS";
#[cfg(feature = "rocm")]
const ROCM_GRAPHS_ENV: &str = "HANZO_ROCM_GRAPHS";
const FLASHINFER_DECODE_ENV: &str = "HANZO_FLASHINFER_DECODE";

static CUDA_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
#[cfg(feature = "metal")]
static METAL_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
#[cfg(feature = "rocm")]
static ROCM_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
static FLASHINFER_DECODE_ENABLED: OnceLock<bool> = OnceLock::new();

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on") {
                true
            } else if matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off") {
                false
            } else {
                default
            }
        })
        .unwrap_or(default)
}

pub(crate) fn cuda_graphs_enabled() -> bool {
    *CUDA_GRAPHS_ENABLED.get_or_init(|| env_flag(CUDA_GRAPHS_ENV, true))
}

#[cfg(feature = "metal")]
pub(crate) fn metal_graphs_enabled() -> bool {
    *METAL_GRAPHS_ENABLED.get_or_init(|| env_flag(METAL_GRAPHS_ENV, true))
}

#[cfg(feature = "rocm")]
pub(crate) fn rocm_graphs_enabled() -> bool {
    // Default OFF. The async-conversion + capture-safety work is DONE: the ROCm
    // decode forward now captures into a hipGraph cleanly (no HIP 906, no crash),
    // all per-token state is device-resident in stable `Var` buffers refreshed in
    // place between replays (input_ids / slot_mappings / context_lens / block_tables
    // / rope_positions — all verified to advance correctly), and the captured output
    // buffers are reserved out of the caching pool for the graph's lifetime so no
    // later allocation can alias them (see `wrappers::PoolInner`).
    //
    // COHERENCE BUG — RESOLVED 2026-06-08. graphs-ON is now COHERENT and
    // bit-exact (token-for-token identical to graphs-OFF on the same prompt+seed
    // for 707-token Qwen3-8B-Q8_0 decode). Two sub-issues were fixed:
    //
    //  (1) STALE/FROZEN LOGITS — fixed earlier: instantiate with flags=0 (no
    //      `hipGraphInstantiateFlagAutoFreeOnLaunch`); the ROCm backend has no
    //      graph-ordered allocator, so AutoFreeOnLaunch recycled the reserved
    //      logits buffer across replays. See `rocm_graph.rs`.
    //
    //  (2) FROZEN FULL-ATTENTION CONTEXT SPAN — the actual final blocker, fixed
    //      here. Models WITHOUT a sliding window (Qwen3) take the `use_full`
    //      decode path in `paged_attention::layers::paged_attention`, where the
    //      paged-attn v1 kernel reads `full_context_lens` / `full_block_tables`
    //      (NOT the windowed `context_lens` / `block_tables`). The ROCm graph's
    //      `RocmDecodeGraphMetadataBuffers` only refreshed the windowed buffers
    //      between replays and cloned the full ones VERBATIM from the warmup
    //      metadata — so the captured kernel attended a FROZEN context span (the
    //      warmup length, e.g. 17) every token while the windowed buffers (unused
    //      on this path) advanced correctly. A per-token KV/state probe proved
    //      slot/ctx/rope/block_tables all advanced AND the KV was written to the
    //      correct advancing slot, yet replay logits diverged from a fresh eager
    //      forward by ~20 (max-abs) while matching an eager forward run with the
    //      REBOUND metadata exactly — isolating `full_context_lens`/`full_block_tables`
    //      as the frozen piece. Fix: `RocmDecodeGraphMetadataBuffers` now owns
    //      `full_block_tables` + `full_context_lens` stable Vars, refreshes them in
    //      `copy_from` every token (full_context_lens always, full_block_tables on
    //      signature change), aliases them in `metadata_from`, and derives
    //      `full_max_context_len` as the bucketed capacity — mirroring the CUDA
    //      path exactly. Verified replay-vs-eager max_abs_diff = 0.0.
    //
    // Default stays OFF: the fix is correct and non-regressing, but on this box
    // the captured graph gives NO decode speedup (decode ~12.7 t/s graphs-ON ==
    // graphs-OFF, both paged). After the fused-kernel work (rmsnorm/rope/softmax/
    // swiglu/bf16-matvec) the decode is no longer launch-bound, so collapsing the
    // remaining per-token launch overhead via hipGraph buys nothing here; paged
    // decode (12.7) also sits below the non-paged eager floor (~20.5). Flip
    // `HANZO_ROCM_GRAPHS=1` to use the (now-coherent) graph path.
    *ROCM_GRAPHS_ENABLED.get_or_init(|| env_flag(ROCM_GRAPHS_ENV, false))
}

pub(crate) fn flashinfer_decode_enabled() -> bool {
    *FLASHINFER_DECODE_ENABLED.get_or_init(|| env_flag(FLASHINFER_DECODE_ENV, true))
}
