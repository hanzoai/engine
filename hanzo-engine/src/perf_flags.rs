use std::sync::OnceLock;

// Perf toggles resolve on two orthogonal axes: the optimization (the "concept": GRAPHS, FUSED_ATTN,
// PREFILL_GRAPHS, ...) and the backend it runs on (CUDA, METAL, ROCM, VK) through one precedence,
// so a new toggle is one `resolve(...)` line, never another copy of this logic. First hit wins:
//
//   1. `ENGINE_<BACKEND>_<CONCEPT>` (with fallback to `<BACKEND>_<CONCEPT>` / `HANZO_<BACKEND>_<CONCEPT>`)
//   2. `ENGINE_<CONCEPT>`           (with fallback to `<CONCEPT>` / `HANZO_<CONCEPT>`)
//   3. `ENGINE_PERF`                (with fallback to `PERF` / `HANZO_PERF`)
//   4. autotuned default

fn parse_bool_str(s: &str) -> Option<bool> {
    match s.trim() {
        "1" | "true" | "TRUE" | "yes" | "on" => Some(true),
        "0" | "false" | "FALSE" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Parse a boolean env var checking `ENGINE_<name>` first, then bare `<name>`, then `HANZO_<name>`.
/// `None` when unset or unrecognized, so resolution falls through to the next source.
fn env_opt(name: &str) -> Option<bool> {
    let engine_key = format!("ENGINE_{name}");
    if let Ok(v) = std::env::var(&engine_key) {
        return parse_bool_str(&v);
    }
    if let Ok(v) = std::env::var(name) {
        if let Some(b) = parse_bool_str(&v) {
            return Some(b);
        }
    }
    let legacy_key = format!("HANZO_{name}");
    std::env::var(&legacy_key)
        .ok()
        .and_then(|v| parse_bool_str(&v))
}

/// Resolve one perf toggle: its full `<BACKEND>_<CONCEPT>` name, its bare concept (backend-independent
/// override; `None` for opts with no cross-backend sibling), and the autotuned default.
fn resolve(primary: &str, concept: Option<&str>, default: bool) -> bool {
    env_opt(primary)
        .or_else(|| concept.and_then(env_opt))
        .or_else(|| env_opt("PERF"))
        .unwrap_or(default)
}

static CUDA_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
#[cfg(feature = "cuda")]
static CUDA_PREFILL_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
#[cfg(feature = "metal")]
static METAL_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
#[cfg(feature = "rocm")]
static ROCM_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
static FLASHINFER_DECODE_ENABLED: OnceLock<bool> = OnceLock::new();
#[cfg(all(feature = "cuda", target_family = "unix"))]
static FLASHINFER_PREFILL_ENABLED: OnceLock<bool> = OnceLock::new();
static MLA_ABSORB_ENABLED: OnceLock<bool> = OnceLock::new();
#[cfg(feature = "vulkan")]
static VULKAN_FUSED_ATTN_ENABLED: OnceLock<bool> = OnceLock::new();
#[cfg(feature = "vulkan")]
static VULKAN_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();

pub(crate) fn cuda_graphs_enabled() -> bool {
    *CUDA_GRAPHS_ENABLED.get_or_init(|| resolve("CUDA_GRAPHS", Some("GRAPHS"), true))
}

// Fused GQA flash-SDPA decode kernel (sdpa_blk) on Vulkan: one dispatch replaces the repeat_kv + QKᵀ
// bmm + softmax + ·V bmm chain. Default on; VK_FUSED_ATTN=0 (or FUSED_ATTN=0) A/Bs the naive path.
#[cfg(feature = "vulkan")]
pub(crate) fn vulkan_fused_attn_enabled() -> bool {
    *VULKAN_FUSED_ATTN_ENABLED.get_or_init(|| resolve("VK_FUSED_ATTN", Some("FUSED_ATTN"), true))
}

// Vulkan decode command-graph: capture the single-token decode forward once and replay it per token,
// collapsing the eager per-token re-record + resubmit of the full dispatch stream into one queue
// submit — the residual per-token CPU cost once decode is kernel-bound. Default on: the fluent-but-
// stale-output failure mode is guarded by two bit-exact replay tests in hanzo-ml, byte-identical
// greedy decode vs eager in scripts/bench_vk_graph.sh, a self-shaping capture policy that keeps short
// generations on the eager path, and fail-closed fallback on any capture/replay error. VK_GRAPHS=0
// (or GRAPHS=0) forces eager.
#[cfg(feature = "vulkan")]
pub(crate) fn vulkan_graphs_enabled() -> bool {
    *VULKAN_GRAPHS_ENABLED.get_or_init(|| resolve("VK_GRAPHS", Some("GRAPHS"), true))
}

// Dense fixed-shape prefill graph capture (single-sequence, offset-0 first prompt chunk). Default off,
// and additionally gated by cuda_graphs_enabled(): capturing the prefill collapses its eager launches
// into one replay and recovers part of the prefill GPU-idle gap, but it does not close the whole gap,
// and replay is not yet bit-exact for chunked/large prefill (greedy output diverged on a
// long prompt) — so it fails the exactness bar for a default-on path and stays behind
// CUDA_PREFILL_GRAPHS=1 for continued hardening. Eager prefill remains the correct default.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_prefill_graphs_enabled() -> bool {
    *CUDA_PREFILL_GRAPHS_ENABLED.get_or_init(|| {
        cuda_graphs_enabled() && resolve("CUDA_PREFILL_GRAPHS", Some("PREFILL_GRAPHS"), false)
    })
}

#[cfg(feature = "metal")]
pub(crate) fn metal_graphs_enabled() -> bool {
    *METAL_GRAPHS_ENABLED.get_or_init(|| resolve("METAL_GRAPHS", Some("GRAPHS"), true))
}

#[cfg(feature = "rocm")]
pub(crate) fn rocm_graphs_enabled() -> bool {
    // Default on, and coherent/bit-exact (token-for-token identical to graphs-off on the same
    // prompt+seed). The decode forward captures into a hipGraph cleanly; all per-token state is
    // device-resident in stable `Var` buffers refreshed in place between replays (input_ids /
    // slot_mappings / context_lens / block_tables / rope_positions); the captured output buffers are
    // reserved out of the caching pool for the graph's lifetime so no later allocation can alias them
    // (see `wrappers::PoolInner`).
    //
    // Two coherence bugs had to be fixed to reach bit-exactness, both worth remembering:
    //  (1) Stale/frozen logits: instantiate with flags=0 (no `hipGraphInstantiateFlagAutoFreeOnLaunch`)
    //      — the ROCm backend has no graph-ordered allocator, so AutoFreeOnLaunch recycled the reserved
    //      logits buffer across replays (see `rocm_graph.rs`).
    //  (2) Frozen full-attention context span: models WITHOUT a sliding window (Qwen3) take the
    //      `use_full` decode path, where paged-attn v1 reads `full_context_lens`/`full_block_tables`,
    //      NOT the windowed `context_lens`/`block_tables`. The graph metadata only refreshed the
    //      windowed buffers and cloned the full ones verbatim from warmup, so the captured kernel
    //      attended a frozen context span every token. `RocmDecodeGraphMetadataBuffers` now owns and
    //      refreshes the full buffers in `copy_from` (full_context_lens every token, full_block_tables
    //      on signature change), aliases them in `metadata_from`, and derives `full_max_context_len` as
    //      the bucketed capacity — mirroring the CUDA path.
    //
    // Only `model_supports_rocm_decode_graph` variants use this path (mRoPE Qwen35/Qwen3-VL stay
    // eager), and `disable_rocm_decode_graph` falls back to eager on any capture/replay error.
    // ROCM_GRAPHS=0 (or GRAPHS=0) forces eager.
    *ROCM_GRAPHS_ENABLED.get_or_init(|| resolve("ROCM_GRAPHS", Some("GRAPHS"), true))
}

pub(crate) fn flashinfer_decode_enabled() -> bool {
    // No cross-backend sibling: FlashInfer is one decode-kernel selector, so only its own name and the
    // PERF master gate it.
    *FLASHINFER_DECODE_ENABLED.get_or_init(|| resolve("FLASHINFER_DECODE", None, true))
}

// FlashInfer paged prefill (BatchPrefillWithPagedKVCache, MaskMode::kCausal). Replaces the eager
// cutlass-GEMM + softmax_f32 prompt attention for GGUF/quantized models, whose causal mask is Custom
// and therefore never reaches the dense flash path. Reads the paged cache (the current tokens' K/V are
// written first), so it composes with the FlashInfer decode cache layout. Default on: greedy decode is
// token-for-token identical to the eager causal path across diverse prompts -- the online-softmax
// reduction reorders float adds so intermediate scores differ within flash-attention tolerance, but the
// argmax does not move -- and the call-site guards fall through to eager for any uncovered shape.
// FLASHINFER_PREFILL=0 forces eager. Consumed only by the paged-attention prefill path, which is
// cfg-gated to cuda + unix; the accessor tracks the same gate so non-cuda builds carry no dead flag.
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn flashinfer_prefill_enabled() -> bool {
    *FLASHINFER_PREFILL_ENABLED.get_or_init(|| resolve("FLASHINFER_PREFILL", None, true))
}

// Absorbed-MLA decode (DeepSeek weight absorption) for the MLA archs (deepseek2 / glm-dsa). When on,
// the KV cache holds only the compressed latent `[kv_lora + qk_rope]` per token instead of the
// materialized per-head K/V, and `kv_b` folds into the query (`q_nope @ w_uk`) and out of the context
// (`(att @ ckv) @ w_uv_t`). Algebraically identical to the materialized path, so decode is
// token-for-token equal in exact arithmetic -- but absorption reassociates the score/context
// reductions, so float rounding differs and a near-tie argmax can in principle flip (the same
// shape-dependent-kernel caveat as the MTP/CUDA tiers). Device-agnostic: it is a different
// factorization of the same math, not a kernel.
//
// This gates BOTH the model's decode path and the KV-cache shape the engine preallocates
// (`ContentConfig` reports the latent dims when it is on) -- they must agree, so they read one flag.
pub(crate) fn mla_absorb_enabled() -> bool {
    *MLA_ABSORB_ENABLED.get_or_init(|| resolve("MLA_ABSORB", None, false))
}

#[cfg(test)]
mod tests {
    use super::resolve;
    use std::sync::Mutex;

    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn engine_prefix_precedes_bare_and_legacy_hanzo() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let keys = [
            "ENGINE_TEST_GRAPHS",
            "TEST_GRAPHS",
            "HANZO_TEST_GRAPHS",
            "ENGINE_TEST_CONCEPT",
            "TEST_CONCEPT",
            "HANZO_TEST_CONCEPT",
            "ENGINE_PERF",
            "PERF",
            "HANZO_PERF",
        ];
        for k in keys {
            std::env::remove_var(k);
        }

        assert!(resolve("TEST_GRAPHS", Some("TEST_CONCEPT"), true));
        assert!(!resolve("TEST_GRAPHS", Some("TEST_CONCEPT"), false));

        std::env::set_var("HANZO_TEST_GRAPHS", "0");
        assert!(!resolve("TEST_GRAPHS", Some("TEST_CONCEPT"), true));

        std::env::set_var("TEST_GRAPHS", "1");
        assert!(resolve("TEST_GRAPHS", Some("TEST_CONCEPT"), false));

        std::env::set_var("ENGINE_TEST_GRAPHS", "0");
        assert!(!resolve("TEST_GRAPHS", Some("TEST_CONCEPT"), true));

        std::env::remove_var("ENGINE_TEST_GRAPHS");
        std::env::remove_var("TEST_GRAPHS");
        std::env::remove_var("HANZO_TEST_GRAPHS");

        std::env::set_var("HANZO_TEST_CONCEPT", "1");
        assert!(resolve("TEST_GRAPHS", Some("TEST_CONCEPT"), false));
        std::env::set_var("ENGINE_TEST_CONCEPT", "0");
        assert!(!resolve("TEST_GRAPHS", Some("TEST_CONCEPT"), true));

        for k in keys {
            std::env::remove_var(k);
        }
    }
}
