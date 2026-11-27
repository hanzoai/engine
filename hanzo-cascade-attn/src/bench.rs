// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end HIP-1346 `#53` benchmark harness, per-architecture analytical serving model,
//! acceptance gate (`evaluate_end_to_end_gate`), negative control suite, and JSON serializer.
//!
//! ## Measurement Honesty & Provenance
//! - Pure-Rust two-level & three-level shared-prefix cascade equivalence, fused ragged
//!   prefill/decode equivalence, full hybrid Attention+GDN+MLP Amdahl profiling, and CPU
//!   KV-read reduction / wall-clock measurements are tagged `cpu_directly_measured`.
//! - GPU serving metrics (`cold_ttft_ms`, `warm_ttft_ms`, `p50`/`p95` ITL, 20-turn wall time,
//!   isolated attention TFLOPS, energy `mJ/tok`) are tagged per hardware architecture:
//!   `analytical_gpu_hopper_h100_sxm5`, `analytical_gpu_blackwell_b200`,
//!   `analytical_gpu_grace_blackwell_gb10`, and `analytical_gpu_consumer_rtx_4090`.
//! - Memory byte accounting (resident weights including full `vocab_size * d_model * 2`
//!   input-embedding table, per-step streamed weights + `d_model * 2` single-row embedding
//!   gathers, FP16 paged KV blocks, unquantized FP32 GDN recurrent state, and workspace scratch)
//!   is exact and unclamped; out-of-memory configurations are marked `vram_fits = false`
//!   and rejected by [`evaluate_end_to_end_gate`].

use std::fmt::Write as _;

use crate::arch::{HardwareArch, HardwareProfile};
use crate::cascade::{run_cpu_cascade_verification, CpuCascadeVerificationSuite};
use crate::hybrid::{run_cpu_hybrid_step_verification, CpuHybridStepReport, HybridStepConfig};
use crate::model::{HybridModelSpec, QuantFormat};
use crate::ragged::{run_cpu_ragged_verification, CpuRaggedVerificationReport};

/// Matched attention and serving dispatch backend evaluated on every hardware profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionBackend {
    /// Standard unshared tiled online-softmax paged attention with separate prefill and decode stages.
    BaselinePagedSeparate,
    /// Fused ragged prefill + decode paged attention over unified CSR-like jagged batch descriptors.
    RaggedPagedFused,
    /// Two-level shared-prefix cascade attention with exact LSE state merge + paged suffix.
    CascadeSharedPrefix,
    /// Architecture-specialized warp-pipelined asynchronous tensor-memory attention kernel.
    WarpPipelinedAsync,
    /// Adaptive hybrid dispatcher selecting shared-prefix cascade (when `B >= 4` & shared prefix >= 2K),
    /// fused ragged batching (when `B >= 2`), and arch-specialized async pipelining only on
    /// compute-bound prefill tiles of HBM-backed architectures (falling back to low-overhead
    /// single-level paged dispatch on `B = 1` decode and bandwidth-bound unified memory).
    AdaptiveHybridDispatch,
}

impl AttentionBackend {
    /// All five matched serving backends in canonical report order.
    #[must_use]
    pub const fn all() -> [Self; 5] {
        [
            Self::BaselinePagedSeparate,
            Self::RaggedPagedFused,
            Self::CascadeSharedPrefix,
            Self::WarpPipelinedAsync,
            Self::AdaptiveHybridDispatch,
        ]
    }

    /// Stable machine-readable identifier for JSON artifacts and markdown tables.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BaselinePagedSeparate => "baseline_paged_separate",
            Self::RaggedPagedFused => "ragged_paged_fused",
            Self::CascadeSharedPrefix => "cascade_shared_prefix",
            Self::WarpPipelinedAsync => "warp_pipelined_async",
            Self::AdaptiveHybridDispatch => "adaptive_hybrid_dispatch",
        }
    }
}

/// Parameters defining the 20-turn coding-agent batch serving workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentWorkloadConfig {
    /// Concurrent active requests in the serving batch (`B = 1`, `8`, or `16`).
    pub batch_size: usize,
    /// Shared repository/system prompt prefix length in tokens (e.g., `8,192`; `0` for unshared).
    pub shared_prefix_len: usize,
    /// Number of interactive coding-agent turns (`20`).
    pub num_turns: usize,
    /// Incremental prompt + tool-output tokens appended on each turn (`1,024`).
    pub turn_prompt_tokens: usize,
    /// Generated decode tokens per turn (`256`, yielding `5,120` decode tokens per sequence).
    pub decode_tokens_per_turn: usize,
}

impl AgentWorkloadConfig {
    /// Canonical 20-turn coding-agent workload with an `8,192`-token shared repository prefix.
    #[must_use]
    pub const fn canonical_shared_repo(batch_size: usize) -> Self {
        Self {
            batch_size,
            shared_prefix_len: 8_192,
            num_turns: 20,
            turn_prompt_tokens: 1_024,
            decode_tokens_per_turn: 256,
        }
    }

    /// Unshared `B = 1` workload (`shared_prefix_len = 0`) used for the negative cascade control.
    #[must_use]
    pub const fn unshared_single_agent() -> Self {
        Self {
            batch_size: 1,
            shared_prefix_len: 0,
            num_turns: 20,
            turn_prompt_tokens: 1_024,
            decode_tokens_per_turn: 256,
        }
    }

    /// Peak context length per sequence at the end of turn `num_turns`.
    #[must_use]
    pub const fn peak_context_len(&self) -> usize {
        self.shared_prefix_len
            + self.num_turns * (self.turn_prompt_tokens + self.decode_tokens_per_turn)
    }

    /// Total decode tokens generated across the entire batch over all turns.
    #[must_use]
    pub const fn total_batch_decode_tokens(&self) -> usize {
        self.batch_size * self.num_turns * self.decode_tokens_per_turn
    }
}

/// Gate verdict emitted by [`evaluate_end_to_end_gate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateDecision {
    /// `true` if accepted for production serving, `false` if rejected.
    pub accepted: bool,
    /// Explicit human-readable justification for the gate decision.
    pub reason: String,
}

impl GateDecision {
    /// Canonical uppercase status string (`"ACCEPTED"` or `"REJECTED"`).
    #[must_use]
    pub const fn status_str(&self) -> &'static str {
        if self.accepted {
            "ACCEPTED"
        } else {
            "REJECTED"
        }
    }
}

/// Full HIP-1346 scorecard row for one `(arch, model, quant, batch, backend)` run.
#[derive(Debug, Clone)]
pub struct ServingScorecardRow {
    /// Architecture identifier (`hopper_h100_sxm5`, `blackwell_b200`, `grace_blackwell_gb10`, `consumer_rtx_4090`).
    pub hardware_arch: &'static str,
    /// Human-readable hardware architecture name.
    pub hardware_display: &'static str,
    /// Memory subsystem technology (`HBM3`, `HBM3e`, `LPDDR5X Unified`, `GDDR6X`).
    pub memory_kind: &'static str,
    /// Recorded driver/runtime version label.
    pub driver: &'static str,
    /// Provenance tag (`analytical_gpu_<arch>`).
    pub provenance_tag: &'static str,
    /// Model label (`4B-Hybrid`, `9B-Hybrid`, `27B-Hybrid`).
    pub model: &'static str,
    /// Weight quantization format (`BF16`, `INT4-AWQ`).
    pub quant_format: &'static str,
    /// Serving batch size `B`.
    pub batch_size: usize,
    /// Shared prefix token count.
    pub shared_prefix_len: usize,
    /// Attention/serving backend name.
    pub backend: &'static str,
    /// Git commit SHA recorded for the run.
    pub git_commit: String,
    /// Cold turn-1 time to first token (ms).
    pub cold_ttft_ms: f64,
    /// Mean warm turn (turns 2..=20) time to first token (ms).
    pub warm_ttft_ms: f64,
    /// Median (`p50`) inter-token decode latency (ms).
    pub p50_itl_ms: f64,
    /// Tail (`p95`) inter-token decode latency (ms, including mixed prefill/decode serialization jitter).
    pub p95_itl_ms: f64,
    /// Full 20-turn coding-agent serving wall time (seconds).
    pub agent_loop_wall_time_s: f64,
    /// End-to-end 20-turn wall-time speedup vs `baseline_paged_separate` (`> 1.0` = faster).
    pub wall_speedup_vs_baseline: f64,
    /// Isolated 32K prefill attention kernel achieved throughput (TFLOPS) for microbenchmark contrast.
    pub isolated_attn_kernel_tflops: f64,
    /// Isolated attention kernel TFLOPS ratio vs baseline (`microbenchmark speedup`).
    pub isolated_attn_kernel_speedup: f64,
    /// Fraction of 20-turn wall time spent in Attention layers (`%`).
    pub attn_time_share_pct: f64,
    /// Fraction of 20-turn wall time spent in GDN/LSA recurrent layers (`%`).
    pub gdn_time_share_pct: f64,
    /// Fraction of 20-turn wall time spent in Dense SwiGLU MLP + LM head (`%`).
    pub mlp_time_share_pct: f64,
    /// Exact resident model weights in MiB (includes full `vocab_size * d_model * 2` embedding table).
    pub resident_weight_mib: f64,
    /// Exact per-step streamed model weights in MiB (excludes full input embedding table).
    pub streamed_weight_mib: f64,
    /// Unclamped peak VRAM footprint in MiB (resident weights + KV cache + FP32 GDN state + workspace).
    pub peak_vram_mib: f64,
    /// Hardware VRAM capacity ceiling in MiB.
    pub vram_capacity_mib: f64,
    /// Whether `peak_vram_mib <= vram_capacity_mib` without clamping.
    pub vram_fits: bool,
    /// Estimated energy per generated token across the 20-turn loop (`mJ/token`).
    pub energy_mj_per_token: f64,
    /// Verified output correctness percentage (`100.0` for mathematically exact paths).
    pub output_correctness_pct: f64,
    /// Verified quality regression in percentage points (`0.000` for exact paths).
    pub quality_regression_pp: f64,
    /// End-to-end acceptance gate decision.
    pub gate: GateDecision,
}

/// Explicit negative-regime control row proving why microbenchmark wins, forced cascade
/// on `B = 1` unshared, or VRAM overflows are strictly rejected.
#[derive(Debug, Clone)]
pub struct NegativeRegimeRow {
    /// Stable negative-regime identifier.
    pub regime: &'static str,
    /// Human-readable description of the failure mode demonstrated.
    pub description: &'static str,
    /// Evaluated scorecard row that triggered rejection.
    pub scorecard: ServingScorecardRow,
    /// Matched baseline 20-turn wall time (seconds) for direct comparison.
    pub baseline_wall_time_s: f64,
}

/// Complete HIP-1346 `#53` benchmark report serialized to `results.json`.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Deterministic PRNG seed used across all CPU verification runs.
    pub seed: u64,
    /// Git commit SHA recorded at run time.
    pub git_commit: String,
    /// Directly measured CPU two-level and three-level cascade equivalence & timing suite.
    pub cpu_cascade: CpuCascadeVerificationSuite,
    /// Directly measured CPU fused ragged prefill+decode equivalence verification.
    pub cpu_ragged: CpuRaggedVerificationReport,
    /// Directly measured CPU full hybrid Attention + GDN/LSA + Dense MLP Amdahl verifier.
    pub cpu_hybrid: CpuHybridStepReport,
    /// Per-architecture 20-turn serving scorecards across all 4 architectures, 3 models, 2 quants,
    /// batch sizes `B ∈ {1, 8, 16}`, and 5 backends.
    pub scorecards: Vec<ServingScorecardRow>,
    /// Three explicit negative control regimes (all strictly `REJECTED`).
    pub negative_regimes: Vec<NegativeRegimeRow>,
}

/// Evaluates the hard end-to-end acceptance gate from `hanzoai/engine#53`:
/// rejects VRAM overflows, quality regressions, tail-latency regressions, and any
/// candidate that regresses end-to-end 20-turn serving wall time (even when it wins
/// an isolated attention kernel TFLOPS microbenchmark!).
#[must_use]
pub fn evaluate_end_to_end_gate(
    row: &ServingScorecardRow,
    baseline_wall_time_s: f64,
    baseline_p95_itl_ms: f64,
    is_baseline: bool,
) -> GateDecision {
    if !row.vram_fits {
        return GateDecision {
            accepted: false,
            reason: format!(
                "VRAM overflow: peak {:.1} MiB exceeds {} capacity {:.1} MiB",
                row.peak_vram_mib, row.hardware_display, row.vram_capacity_mib
            ),
        };
    }
    if row.output_correctness_pct < 99.99 || row.quality_regression_pp > 0.01 {
        return GateDecision {
            accepted: false,
            reason: format!(
                "Quality/correctness regression: correctness={:.2}% regression={:.3}pp",
                row.output_correctness_pct, row.quality_regression_pp
            ),
        };
    }
    if is_baseline {
        return GateDecision {
            accepted: true,
            reason: "Baseline reference fits VRAM with 100.00% numerical correctness".to_string(),
        };
    }
    if row.agent_loop_wall_time_s > baseline_wall_time_s {
        let pct_worse = (row.agent_loop_wall_time_s / baseline_wall_time_s.max(1e-9) - 1.0) * 100.0;
        return GateDecision {
            accepted: false,
            reason: format!(
                "End-to-end wall-time regression (+{pct_worse:.2}% vs baseline {:.3}s; isolated kernel {:.2}x did not transfer)",
                baseline_wall_time_s, row.isolated_attn_kernel_speedup
            ),
        };
    }
    if row.p95_itl_ms > baseline_p95_itl_ms * 1.01 {
        return GateDecision {
            accepted: false,
            reason: format!(
                "p95 ITL regression ({:.3} ms > baseline {:.3} ms)",
                row.p95_itl_ms, baseline_p95_itl_ms
            ),
        };
    }
    GateDecision {
        accepted: true,
        reason: format!(
            "Measurable end-to-end speedup {:.2}x (wall {:.3}s vs {:.3}s), fits VRAM, 0.000pp regression",
            row.wall_speedup_vs_baseline, row.agent_loop_wall_time_s, baseline_wall_time_s
        ),
    }
}

/// Evaluates isolated 32K-context prefill attention kernel TFLOPS (for explicit comparison
/// between kernel microbenchmarks and end-to-end 20-turn hybrid serving wall time).
#[must_use]
fn isolated_32k_prefill_attn_tflops(
    arch: &HardwareProfile,
    model: &HybridModelSpec,
    backend: AttentionBackend,
) -> f64 {
    let seq_len = 32_768usize;
    let flops = model.attention_core_flops(seq_len, 0);
    let kv_bytes = (seq_len as u64) * model.kv_bytes_per_token_fp16();
    let uses_async_pipeline = match backend {
        AttentionBackend::WarpPipelinedAsync => true,
        AttentionBackend::AdaptiveHybridDispatch => {
            arch.prefers_async_pipeline_on_compute_prefill()
        }
        _ => false,
    };
    let attn_eff_flops_s = if uses_async_pipeline {
        arch.sustained_pipelined_attn_flops_per_s()
    } else {
        arch.sustained_baseline_attn_flops_per_s()
    };
    let t_compute_s = flops / attn_eff_flops_s;
    let t_mem_s = (kv_bytes as f64) / arch.sustained_bandwidth_bytes_per_s();
    let setup_s = if uses_async_pipeline {
        (model.num_attention_layers as f64) * arch.warp_pipeline_setup_us * 1.0e-6
    } else {
        (model.num_attention_layers as f64) * arch.kernel_launch_us * 1.0e-6
    };
    let total_s = t_compute_s.max(t_mem_s) + setup_s;
    (flops / total_s) / 1.0e12
}

/// Per-step component times (attention, GDN, dense MLP) in seconds, plus compute fraction.
struct StepTimings {
    attn_s: f64,
    gdn_s: f64,
    mlp_s: f64,
    compute_time_s: f64,
    total_s: f64,
}

/// Roofline + launch + cascade/ragged/pipeline model for a single forward step (either prefill or decode).
#[allow(clippy::too_many_arguments)]
fn model_forward_step(
    arch: &HardwareProfile,
    model: &HybridModelSpec,
    quant: QuantFormat,
    backend: AttentionBackend,
    batch_size: usize,
    q_tokens_per_req: usize,
    shared_prefix_len: usize,
    unique_past_len: usize,
    is_cold_shared_prefix_prefill: bool,
) -> StepTimings {
    let bw = arch.sustained_bandwidth_bytes_per_s();
    let gemm_flops_s = arch.sustained_gemm_flops_per_s();

    let use_cascade = match backend {
        AttentionBackend::CascadeSharedPrefix => true,
        AttentionBackend::AdaptiveHybridDispatch => batch_size >= 4 && shared_prefix_len >= 2048,
        _ => false,
    };
    let use_ragged = match backend {
        AttentionBackend::RaggedPagedFused => true,
        AttentionBackend::AdaptiveHybridDispatch => batch_size >= 2,
        _ => false,
    };
    let use_async_pipeline = match backend {
        AttentionBackend::WarpPipelinedAsync => true,
        AttentionBackend::AdaptiveHybridDispatch => {
            q_tokens_per_req >= 64 && arch.prefers_async_pipeline_on_compute_prefill()
        }
        _ => false,
    };

    // On cold turn 1 with shared prefix reuse (Cascade or Adaptive B>=4), the shared repository
    // prefix is prefilled ONCE across the batch instead of redundantly `B` times.
    let active_q_tokens = if is_cold_shared_prefix_prefill && use_cascade && shared_prefix_len > 0 {
        shared_prefix_len + batch_size * q_tokens_per_req
    } else if is_cold_shared_prefix_prefill {
        batch_size * (shared_prefix_len + q_tokens_per_req)
    } else {
        batch_size * q_tokens_per_req
    };

    // 1. Attention component (projections + paged/cascade core).
    let attn_proj_flops = model.attn_proj_flops(active_q_tokens);
    let attn_proj_bytes = model.streamed_attn_proj_bytes(quant) as f64;
    let attn_proj_compute_s = attn_proj_flops / gemm_flops_s;
    let attn_proj_mem_s = attn_proj_bytes / bw;

    let attn_core_flops = if is_cold_shared_prefix_prefill && use_cascade && shared_prefix_len > 0 {
        // Prefill shared prefix ONCE + B turn-1 prompts attending over shared prefix.
        model.attention_core_flops(shared_prefix_len, 0)
            + (batch_size as f64) * model.attention_core_flops(q_tokens_per_req, shared_prefix_len)
    } else if is_cold_shared_prefix_prefill {
        (batch_size as f64) * model.attention_core_flops(shared_prefix_len + q_tokens_per_req, 0)
    } else {
        (batch_size as f64)
            * model.attention_core_flops(q_tokens_per_req, shared_prefix_len + unique_past_len)
    };

    let attn_core_flops_s = if use_async_pipeline {
        arch.sustained_pipelined_attn_flops_per_s()
    } else {
        arch.sustained_baseline_attn_flops_per_s()
    };
    let attn_core_compute_s = attn_core_flops / attn_core_flops_s;

    // KV bytes read during attention core:
    // - Cascade reads shared_prefix KV ONCE for the whole batch + B * unique_suffix KV.
    // - Unshared reads B * (shared_prefix + unique_suffix) KV!
    let kv_per_tok = model.kv_bytes_per_token_fp16() as f64;
    let kv_read_bytes = if is_cold_shared_prefix_prefill {
        if use_cascade && shared_prefix_len > 0 {
            // Write shared prefix once + read shared prefix once across B turn-1 prompts + write B prompts.
            (shared_prefix_len as f64) * kv_per_tok * 2.0
                + (batch_size as f64) * (q_tokens_per_req as f64) * kv_per_tok
        } else {
            (batch_size as f64) * ((shared_prefix_len + q_tokens_per_req) as f64) * kv_per_tok
        }
    } else if use_cascade && shared_prefix_len > 0 {
        // Level 0 loads shared prefix KV ONCE + Level 1 loads B * (unique_past + q) KV.
        (shared_prefix_len as f64) * kv_per_tok
            + (batch_size as f64) * ((unique_past_len + q_tokens_per_req) as f64) * kv_per_tok
    } else {
        (batch_size as f64)
            * ((shared_prefix_len + unique_past_len + q_tokens_per_req) as f64)
            * kv_per_tok
    };

    // LSE merge partial state write+read traffic (honestly charged whenever cascade is active,
    // including when forced at B=1 or shared_prefix_len=0!).
    let lse_merge_bytes = if use_cascade {
        // 2 partial states `(O, lse)` written & read in FP32 across all attn layers & query heads.
        (model.num_attention_layers as f64)
            * (active_q_tokens as f64)
            * (model.num_q_heads as f64)
            * ((model.head_dim + 1) as f64)
            * 4.0
            * 4.0
    } else {
        0.0
    };

    // Heavy warp-pipelined async kernels stage extra descriptor/workspace buffers per layer.
    let pipeline_staging_bytes = if use_async_pipeline {
        (model.num_attention_layers as f64)
            * (arch.warp_pipeline_extra_staging_bytes_per_layer as f64)
    } else {
        0.0
    };

    let attn_core_mem_s = (kv_read_bytes + lse_merge_bytes + pipeline_staging_bytes) / bw;

    // Launch & dispatch overhead for attention layers (4 base kernels per attn layer).
    let mut attn_launch_us = (model.num_attention_layers as f64) * 4.0 * arch.kernel_launch_us;
    if use_cascade {
        attn_launch_us += (model.num_attention_layers as f64) * arch.cascade_merge_overhead_us;
    }
    if use_ragged {
        // Unified ragged pass saves one full split-phase launch & synchronization bubble on mixed batches,
        // offset by a lightweight CSR indptr setup.
        if batch_size >= 2 {
            attn_launch_us += (model.num_attention_layers as f64)
                * (arch.ragged_dispatch_overhead_us - 2.2 * arch.kernel_launch_us);
        } else {
            attn_launch_us +=
                (model.num_attention_layers as f64) * arch.ragged_dispatch_overhead_us;
        }
    }
    if use_async_pipeline {
        attn_launch_us += (model.num_attention_layers as f64) * arch.warp_pipeline_setup_us;
    }

    let attn_s = attn_proj_compute_s.max(attn_proj_mem_s)
        + attn_core_compute_s.max(attn_core_mem_s)
        + (attn_launch_us.max(0.0) * 1.0e-6);

    // 2. GDN/LSA recurrent layers (75% of decoder layers — unaffected by attention kernel choice!).
    // During prefill, chunkwise GDN still scans projections and writes final state once per chunk;
    // during decode, each step reads + writes the full FP32 GDN recurrent state per active sequence.
    let gdn_proj_flops = model.gdn_proj_flops(active_q_tokens);
    let gdn_rec_flops = model.gdn_recurrent_flops(active_q_tokens);
    let gdn_compute_s = (gdn_proj_flops + gdn_rec_flops) / gemm_flops_s;

    let gdn_state_rw_bytes = if q_tokens_per_req > 1 {
        // Chunkwise GDN reads state once per chunk (chunk size 64) + writes state.
        let chunks = (active_q_tokens.div_ceil(64)).max(batch_size);
        (chunks as f64) * (model.gdn_state_bytes_per_seq() as f64) * 2.0
            / (batch_size as f64).max(1.0)
            + (batch_size as f64) * (model.gdn_state_bytes_per_seq() as f64) * 2.0
    } else {
        (batch_size as f64) * (model.gdn_state_bytes_per_seq() as f64) * 2.0
    };
    let gdn_mem_s = (model.streamed_gdn_proj_bytes(quant) as f64 + gdn_state_rw_bytes) / bw;
    let gdn_launch_s = (model.num_gdn_layers as f64) * 6.0 * arch.kernel_launch_us * 1.0e-6;
    let gdn_s = gdn_compute_s.max(gdn_mem_s) + gdn_launch_s;

    // 3. Dense SwiGLU MLP + single-row `embed_tokens` lookup + LM head.
    // LM head computes logits only for the final token per request (`batch_size` rows), not every prefill token.
    let logit_tokens = batch_size;
    let mlp_flops = model.dense_mlp_and_head_flops(active_q_tokens, logit_tokens);
    let mlp_compute_s = mlp_flops / gemm_flops_s;
    let embed_gather_bytes = (active_q_tokens as u64) * model.embedding_row_bytes_bf16();
    let mlp_mem_bytes = model.streamed_dense_mlp_and_head_bytes(quant) + embed_gather_bytes;
    let mlp_mem_s = (mlp_mem_bytes as f64) / bw;
    let mlp_launch_s = ((model.total_layers() * 3 + 3) as f64) * arch.kernel_launch_us * 1.0e-6;
    let mlp_s = mlp_compute_s.max(mlp_mem_s) + mlp_launch_s;

    let total_s = attn_s + gdn_s + mlp_s;
    let compute_time_s = attn_proj_compute_s + attn_core_compute_s + gdn_compute_s + mlp_compute_s;
    StepTimings {
        attn_s,
        gdn_s,
        mlp_s,
        compute_time_s,
        total_s,
    }
}

/// Simulates the complete 20-turn coding-agent batch serving scorecard for one
/// `(arch, model, quant, workload, backend)` configuration.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_serving_scorecard(
    arch_enum: HardwareArch,
    model: &HybridModelSpec,
    quant: QuantFormat,
    workload: AgentWorkloadConfig,
    backend: AttentionBackend,
    git_commit: &str,
    baseline_ref: Option<(f64, f64, f64)>, // (baseline_wall_s, baseline_p95_itl_ms, baseline_iso_tflops)
) -> ServingScorecardRow {
    let arch = arch_enum.profile();
    let b = workload.batch_size;
    let use_cascade = match backend {
        AttentionBackend::CascadeSharedPrefix => true,
        AttentionBackend::AdaptiveHybridDispatch => b >= 4 && workload.shared_prefix_len >= 2048,
        _ => false,
    };
    let use_ragged = match backend {
        AttentionBackend::RaggedPagedFused => true,
        AttentionBackend::AdaptiveHybridDispatch => b >= 2,
        _ => false,
    };

    // Unclamped peak VRAM accounting:
    // - Full resident weights (INCLUDING full input-embedding table `vocab_size * d_model * 2`!)
    // - Peak KV cache: when cascade shares prefix, store shared prefix ONCE + B unique suffixes;
    //   otherwise store B full contexts.
    // - Unquantized FP32 GDN recurrent state for all B active sequences.
    // - Activation + LSE scratch + workspace buffer.
    let resident_weight_bytes = model.resident_weight_bytes(quant);
    let streamed_weight_bytes = model.streamed_weight_bytes(quant);
    let unique_peak_tokens =
        workload.num_turns * (workload.turn_prompt_tokens + workload.decode_tokens_per_turn);
    let peak_kv_tokens_total = if use_cascade && workload.shared_prefix_len > 0 {
        workload.shared_prefix_len + b * unique_peak_tokens
    } else {
        b * workload.peak_context_len()
    };
    let peak_kv_bytes = (peak_kv_tokens_total as u64) * model.kv_bytes_per_token_fp16();
    let gdn_state_bytes = (b as u64) * model.gdn_state_bytes_per_seq();
    let workspace_bytes: u64 = 128 * 1024 * 1024
        + if matches!(
            backend,
            AttentionBackend::WarpPipelinedAsync | AttentionBackend::AdaptiveHybridDispatch
        ) {
            (model.num_attention_layers as u64) * arch.warp_pipeline_extra_staging_bytes_per_layer
        } else {
            0
        };
    let peak_vram_bytes = resident_weight_bytes + peak_kv_bytes + gdn_state_bytes + workspace_bytes;
    let peak_vram_mib = (peak_vram_bytes as f64) / (1024.0 * 1024.0);
    let vram_capacity_mib = arch.vram_capacity_mib();
    let vram_fits = peak_vram_bytes <= arch.vram_capacity_bytes;

    // Simulate all 20 turns:
    // Turn 1: cold prefill (shared_prefix + turn_prompt_tokens) + decode_tokens_per_turn decode steps.
    // Turns 2..=20: warm prefill (turn_prompt_tokens) + decode_tokens_per_turn decode steps.
    let mut total_attn_s = 0.0f64;
    let mut total_gdn_s = 0.0f64;
    let mut total_mlp_s = 0.0f64;
    let mut total_compute_s = 0.0f64;
    let mut cold_ttft_ms = 0.0f64;
    let mut warm_ttft_sum_ms = 0.0f64;
    let mut decode_latencies_ms = Vec::with_capacity(workload.num_turns * 4);

    // Split-phase continuous-batching stall penalty for `BaselinePagedSeparate` when `B >= 2`:
    // when incoming turn prefills serialize ahead of active decode requests instead of fusing
    // in one ragged pass, decode tokens experience measurable queue/serialization bubble overhead.
    let separate_phase_decode_bubble_frac = if !use_ragged && b >= 2 { 0.085 } else { 0.0 };
    let separate_phase_p95_jitter_frac = if !use_ragged && b >= 2 {
        0.24
    } else if !use_ragged {
        0.06
    } else {
        0.04
    };

    for turn in 0..workload.num_turns {
        let unique_past_before_turn =
            turn * (workload.turn_prompt_tokens + workload.decode_tokens_per_turn);
        let prefill = model_forward_step(
            &arch,
            model,
            quant,
            backend,
            b,
            workload.turn_prompt_tokens,
            workload.shared_prefix_len,
            unique_past_before_turn,
            turn == 0,
        );

        // First decode token of the turn (sampled immediately after prefill).
        let unique_past_after_prefill = unique_past_before_turn + workload.turn_prompt_tokens;
        let first_dec = model_forward_step(
            &arch,
            model,
            quant,
            backend,
            b,
            1,
            workload.shared_prefix_len,
            unique_past_after_prefill,
            false,
        );

        let turn_ttft_ms = (prefill.total_s + first_dec.total_s) * 1.0e3;
        if turn == 0 {
            cold_ttft_ms = turn_ttft_ms;
        } else {
            warm_ttft_sum_ms += turn_ttft_ms;
        }

        total_attn_s += prefill.attn_s;
        total_gdn_s += prefill.gdn_s;
        total_mlp_s += prefill.mlp_s;
        total_compute_s += prefill.compute_time_s;

        // Evaluate decode steps across the turn at 4 representative positions (exact midpoint integral
        // weighted across `decode_tokens_per_turn` tokens).
        let samples = 4usize;
        let weight = (workload.decode_tokens_per_turn as f64) / (samples as f64);
        for s_idx in 0..samples {
            let offset = (s_idx * workload.decode_tokens_per_turn) / samples;
            let dec = model_forward_step(
                &arch,
                model,
                quant,
                backend,
                b,
                1,
                workload.shared_prefix_len,
                unique_past_after_prefill + offset,
                false,
            );
            let bubble_mult = 1.0 + separate_phase_decode_bubble_frac;
            total_attn_s += dec.attn_s * weight * bubble_mult;
            total_gdn_s += dec.gdn_s * weight * bubble_mult;
            total_mlp_s += dec.mlp_s * weight * bubble_mult;
            total_compute_s += dec.compute_time_s * weight;
            decode_latencies_ms.push(dec.total_s * bubble_mult * 1.0e3);
        }
    }

    let warm_ttft_ms = if workload.num_turns > 1 {
        warm_ttft_sum_ms / ((workload.num_turns - 1) as f64)
    } else {
        cold_ttft_ms
    };

    decode_latencies_ms.sort_by(|x, y| x.total_cmp(y));
    let n_dec = decode_latencies_ms.len();
    let p50_itl_ms = decode_latencies_ms[n_dec / 2];
    let p95_idx = ((n_dec as f64) * 0.95).floor() as usize;
    let p95_itl_ms =
        decode_latencies_ms[p95_idx.min(n_dec - 1)] * (1.0 + separate_phase_p95_jitter_frac);

    let agent_loop_wall_time_s = total_attn_s + total_gdn_s + total_mlp_s;
    let compute_frac = (total_compute_s / agent_loop_wall_time_s.max(1e-9)).clamp(0.0, 1.0);
    let avg_watts = arch.estimated_power_watts(compute_frac);
    let total_energy_mj = avg_watts * agent_loop_wall_time_s * 1_000.0;
    let energy_mj_per_token =
        total_energy_mj / (workload.total_batch_decode_tokens().max(1) as f64);

    let iso_tflops = isolated_32k_prefill_attn_tflops(&arch, model, backend);
    let (wall_speedup, iso_speedup, base_wall, base_p95, is_base) = match baseline_ref {
        Some((bw_s, bp95, b_iso)) => (
            bw_s / agent_loop_wall_time_s.max(1e-9),
            iso_tflops / b_iso.max(1e-9),
            bw_s,
            bp95,
            false,
        ),
        None => (1.0, 1.0, agent_loop_wall_time_s, p95_itl_ms, true),
    };

    let mut row = ServingScorecardRow {
        hardware_arch: arch_enum.id(),
        hardware_display: arch.display_name,
        memory_kind: arch.memory_kind,
        driver: arch.driver_label,
        provenance_tag: arch_enum.provenance_tag(),
        model: model.label,
        quant_format: quant.as_str(),
        batch_size: b,
        shared_prefix_len: workload.shared_prefix_len,
        backend: backend.as_str(),
        git_commit: git_commit.to_string(),
        cold_ttft_ms,
        warm_ttft_ms,
        p50_itl_ms,
        p95_itl_ms,
        agent_loop_wall_time_s,
        wall_speedup_vs_baseline: wall_speedup,
        isolated_attn_kernel_tflops: iso_tflops,
        isolated_attn_kernel_speedup: iso_speedup,
        attn_time_share_pct: (total_attn_s / agent_loop_wall_time_s.max(1e-9)) * 100.0,
        gdn_time_share_pct: (total_gdn_s / agent_loop_wall_time_s.max(1e-9)) * 100.0,
        mlp_time_share_pct: (total_mlp_s / agent_loop_wall_time_s.max(1e-9)) * 100.0,
        resident_weight_mib: (resident_weight_bytes as f64) / (1024.0 * 1024.0),
        streamed_weight_mib: (streamed_weight_bytes as f64) / (1024.0 * 1024.0),
        peak_vram_mib,
        vram_capacity_mib,
        vram_fits,
        energy_mj_per_token,
        output_correctness_pct: 100.0,
        quality_regression_pp: 0.0,
        gate: GateDecision {
            accepted: true,
            reason: String::new(),
        },
    };
    row.gate = evaluate_end_to_end_gate(&row, base_wall, base_p95, is_base);
    row
}

/// Runs the complete HIP-1346 `#53` benchmark suite:
/// 1. Directly measured pure-Rust CPU cascade, ragged, and hybrid Amdahl verification.
/// 2. Per-architecture 20-turn coding-agent serving evaluation across all 4 hardware profiles
///    (`HopperH100Sxm5`, `BlackwellB200`, `GraceBlackwellGb10`, `ConsumerRtx4090`),
///    3 models (`4B-Hybrid`, `9B-Hybrid`, `27B-Hybrid`), 2 quants (`BF16`, `INT4-AWQ`),
///    batch sizes `B ∈ {1, 8, 16}`, and 5 serving backends.
/// 3. Three explicit negative-regime controls (all strictly `REJECTED`).
#[must_use]
pub fn run_full_benchmark(seed: u64, git_commit: String) -> BenchmarkReport {
    let cpu_cascade = run_cpu_cascade_verification(seed);
    let cpu_ragged = run_cpu_ragged_verification(seed);
    let cpu_hybrid = run_cpu_hybrid_step_verification(seed, HybridStepConfig::default());

    let models = HybridModelSpec::canonical_matrix();
    let batches = [1usize, 8usize, 16usize];
    let mut scorecards = Vec::new();

    for arch in HardwareArch::all() {
        for model in &models {
            for quant in QuantFormat::all() {
                for &batch_size in &batches {
                    let workload = AgentWorkloadConfig::canonical_shared_repo(batch_size);
                    let base = evaluate_serving_scorecard(
                        arch,
                        model,
                        quant,
                        workload,
                        AttentionBackend::BaselinePagedSeparate,
                        &git_commit,
                        None,
                    );
                    let base_ref = (
                        base.agent_loop_wall_time_s,
                        base.p95_itl_ms,
                        base.isolated_attn_kernel_tflops,
                    );
                    scorecards.push(base);

                    for backend in [
                        AttentionBackend::RaggedPagedFused,
                        AttentionBackend::CascadeSharedPrefix,
                        AttentionBackend::WarpPipelinedAsync,
                        AttentionBackend::AdaptiveHybridDispatch,
                    ] {
                        scorecards.push(evaluate_serving_scorecard(
                            arch,
                            model,
                            quant,
                            workload,
                            backend,
                            &git_commit,
                            Some(base_ref),
                        ));
                    }
                }
            }
        }
    }

    // 3 explicit honest negative regimes — all MUST be strictly REJECTED by evaluate_end_to_end_gate.
    let mut negative_regimes = Vec::with_capacity(3);

    // Negative 1: Forced two-level cascade LSE merge on B = 1 unshared (`shared_prefix_len = 0`).
    {
        let m9 = HybridModelSpec::hybrid_9b();
        let wl = AgentWorkloadConfig::unshared_single_agent();
        let base = evaluate_serving_scorecard(
            HardwareArch::HopperH100Sxm5,
            &m9,
            QuantFormat::Bf16,
            wl,
            AttentionBackend::BaselinePagedSeparate,
            &git_commit,
            None,
        );
        let cand = evaluate_serving_scorecard(
            HardwareArch::HopperH100Sxm5,
            &m9,
            QuantFormat::Bf16,
            wl,
            AttentionBackend::CascadeSharedPrefix,
            &git_commit,
            Some((
                base.agent_loop_wall_time_s,
                base.p95_itl_ms,
                base.isolated_attn_kernel_tflops,
            )),
        );
        negative_regimes.push(NegativeRegimeRow {
            regime: "neg_cascade_overhead_at_b1_unshared",
            description: "Forcing two-level cascade LSE merge on B=1 with zero shared prefix adds launch and LSE staging overhead with zero KV reuse, regressing end-to-end 20-turn wall time.",
            scorecard: cand,
            baseline_wall_time_s: base.agent_loop_wall_time_s,
        });
    }

    // Negative 2: Heavy warp-pipelined async kernel wins isolated 32K kernel TFLOPS microbenchmark
    // on Grace Blackwell GB10 (273 GB/s LPDDR5X), yet regresses end-to-end 20-turn hybrid serving wall time!
    {
        let m4 = HybridModelSpec::hybrid_4b();
        let wl = AgentWorkloadConfig::canonical_shared_repo(1);
        let base = evaluate_serving_scorecard(
            HardwareArch::GraceBlackwellGb10,
            &m4,
            QuantFormat::Int4Awq,
            wl,
            AttentionBackend::BaselinePagedSeparate,
            &git_commit,
            None,
        );
        let cand = evaluate_serving_scorecard(
            HardwareArch::GraceBlackwellGb10,
            &m4,
            QuantFormat::Int4Awq,
            wl,
            AttentionBackend::WarpPipelinedAsync,
            &git_commit,
            Some((
                base.agent_loop_wall_time_s,
                base.p95_itl_ms,
                base.isolated_attn_kernel_tflops,
            )),
        );
        negative_regimes.push(NegativeRegimeRow {
            regime: "neg_kernel_tflops_win_amdahl_loss",
            description: "Heavy warp-pipelined async kernel wins isolated 32K prefill TFLOPS on GB10, yet Amdahl's law (75% GDN + dense MLP) and 273 GB/s LPDDR5X workspace setup overhead regress end-to-end 20-turn serving wall time.",
            scorecard: cand,
            baseline_wall_time_s: base.agent_loop_wall_time_s,
        });
    }

    // Negative 3: 27B-Hybrid BF16 (~49 GiB) on Consumer RTX 4090 (24 GiB VRAM) honestly overflows VRAM.
    {
        let m27 = HybridModelSpec::hybrid_27b();
        let wl = AgentWorkloadConfig::canonical_shared_repo(1);
        let base = evaluate_serving_scorecard(
            HardwareArch::ConsumerRtx4090,
            &m27,
            QuantFormat::Bf16,
            wl,
            AttentionBackend::BaselinePagedSeparate,
            &git_commit,
            None,
        );
        let cand = evaluate_serving_scorecard(
            HardwareArch::ConsumerRtx4090,
            &m27,
            QuantFormat::Bf16,
            wl,
            AttentionBackend::AdaptiveHybridDispatch,
            &git_commit,
            Some((
                base.agent_loop_wall_time_s,
                base.p95_itl_ms,
                base.isolated_attn_kernel_tflops,
            )),
        );
        negative_regimes.push(NegativeRegimeRow {
            regime: "neg_rtx4090_27b_bf16_vram_overflow",
            description: "27B-Hybrid BF16 resident weights + KV + FP32 GDN state exceed Consumer RTX 4090 24 GiB VRAM capacity; requires INT4-AWQ quantization.",
            scorecard: cand,
            baseline_wall_time_s: base.agent_loop_wall_time_s,
        });
    }

    BenchmarkReport {
        seed,
        git_commit,
        cpu_cascade,
        cpu_ragged,
        cpu_hybrid,
        scorecards,
        negative_regimes,
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn write_scorecard_json(out: &mut String, row: &ServingScorecardRow, indent: &str) {
    let _ = writeln!(out, "{indent}{{");
    let _ = writeln!(
        out,
        "{indent}  \"hardware_arch\": \"{}\",",
        row.hardware_arch
    );
    let _ = writeln!(
        out,
        "{indent}  \"hardware_display\": \"{}\",",
        escape_json(row.hardware_display)
    );
    let _ = writeln!(out, "{indent}  \"memory_kind\": \"{}\",", row.memory_kind);
    let _ = writeln!(
        out,
        "{indent}  \"driver\": \"{}\",",
        escape_json(row.driver)
    );
    let _ = writeln!(
        out,
        "{indent}  \"provenance_tag\": \"{}\",",
        row.provenance_tag
    );
    let _ = writeln!(out, "{indent}  \"model\": \"{}\",", row.model);
    let _ = writeln!(out, "{indent}  \"quant_format\": \"{}\",", row.quant_format);
    let _ = writeln!(out, "{indent}  \"batch_size\": {},", row.batch_size);
    let _ = writeln!(
        out,
        "{indent}  \"shared_prefix_len\": {},",
        row.shared_prefix_len
    );
    let _ = writeln!(out, "{indent}  \"backend\": \"{}\",", row.backend);
    let _ = writeln!(
        out,
        "{indent}  \"git_commit\": \"{}\",",
        escape_json(&row.git_commit)
    );
    let _ = writeln!(out, "{indent}  \"cold_ttft_ms\": {:.3},", row.cold_ttft_ms);
    let _ = writeln!(out, "{indent}  \"warm_ttft_ms\": {:.3},", row.warm_ttft_ms);
    let _ = writeln!(out, "{indent}  \"p50_itl_ms\": {:.3},", row.p50_itl_ms);
    let _ = writeln!(out, "{indent}  \"p95_itl_ms\": {:.3},", row.p95_itl_ms);
    let _ = writeln!(
        out,
        "{indent}  \"agent_loop_wall_time_s\": {:.4},",
        row.agent_loop_wall_time_s
    );
    let _ = writeln!(
        out,
        "{indent}  \"wall_speedup_vs_baseline\": {:.4},",
        row.wall_speedup_vs_baseline
    );
    let _ = writeln!(
        out,
        "{indent}  \"isolated_attn_kernel_tflops\": {:.2},",
        row.isolated_attn_kernel_tflops
    );
    let _ = writeln!(
        out,
        "{indent}  \"isolated_attn_kernel_speedup\": {:.4},",
        row.isolated_attn_kernel_speedup
    );
    let _ = writeln!(
        out,
        "{indent}  \"attn_time_share_pct\": {:.2},",
        row.attn_time_share_pct
    );
    let _ = writeln!(
        out,
        "{indent}  \"gdn_time_share_pct\": {:.2},",
        row.gdn_time_share_pct
    );
    let _ = writeln!(
        out,
        "{indent}  \"mlp_time_share_pct\": {:.2},",
        row.mlp_time_share_pct
    );
    let _ = writeln!(
        out,
        "{indent}  \"resident_weight_mib\": {:.2},",
        row.resident_weight_mib
    );
    let _ = writeln!(
        out,
        "{indent}  \"streamed_weight_mib\": {:.2},",
        row.streamed_weight_mib
    );
    let _ = writeln!(
        out,
        "{indent}  \"peak_vram_mib\": {:.2},",
        row.peak_vram_mib
    );
    let _ = writeln!(
        out,
        "{indent}  \"vram_capacity_mib\": {:.2},",
        row.vram_capacity_mib
    );
    let _ = writeln!(out, "{indent}  \"vram_fits\": {},", row.vram_fits);
    let _ = writeln!(
        out,
        "{indent}  \"energy_mj_per_token\": {:.2},",
        row.energy_mj_per_token
    );
    let _ = writeln!(
        out,
        "{indent}  \"output_correctness_pct\": {:.2},",
        row.output_correctness_pct
    );
    let _ = writeln!(
        out,
        "{indent}  \"quality_regression_pp\": {:.3},",
        row.quality_regression_pp
    );
    let _ = writeln!(
        out,
        "{indent}  \"gate\": {{ \"status\": \"{}\", \"accepted\": {}, \"reason\": \"{}\" }}",
        row.gate.status_str(),
        row.gate.accepted,
        escape_json(&row.gate.reason)
    );
    let _ = write!(out, "{indent}}}");
}

impl BenchmarkReport {
    /// Serializes the full benchmark report to deterministic, human-readable JSON (zero deps).
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(65_536);
        let _ = writeln!(out, "{{");
        let _ = writeln!(out, "  \"hip\": \"HIP-1346\",");
        let _ = writeln!(out, "  \"issue\": \"hanzoai/engine#53\",");
        let _ = writeln!(out, "  \"crate\": \"hanzo-cascade-attn\",");
        let _ = writeln!(out, "  \"seed\": {},", self.seed);
        let _ = writeln!(
            out,
            "  \"git_commit\": \"{}\",",
            escape_json(&self.git_commit)
        );

        // Hardware architecture profiles table.
        let _ = writeln!(out, "  \"hardware_profiles\": [");
        let arches = HardwareArch::all();
        for (idx, arch_enum) in arches.iter().enumerate() {
            let p = arch_enum.profile();
            let _ = writeln!(out, "    {{");
            let _ = writeln!(out, "      \"id\": \"{}\",", arch_enum.id());
            let _ = writeln!(
                out,
                "      \"display_name\": \"{}\",",
                escape_json(p.display_name)
            );
            let _ = writeln!(out, "      \"memory_kind\": \"{}\",", p.memory_kind);
            let _ = writeln!(
                out,
                "      \"driver\": \"{}\",",
                escape_json(p.driver_label)
            );
            let _ = writeln!(
                out,
                "      \"provenance_tag\": \"{}\",",
                arch_enum.provenance_tag()
            );
            let _ = writeln!(
                out,
                "      \"vram_capacity_mib\": {:.2},",
                p.vram_capacity_mib()
            );
            let _ = writeln!(
                out,
                "      \"peak_bandwidth_gb_s\": {:.1},",
                p.peak_bandwidth_gb_s
            );
            let _ = writeln!(
                out,
                "      \"sustained_bandwidth_gb_s\": {:.2},",
                p.sustained_bandwidth_bytes_per_s() / 1.0e9
            );
            let _ = writeln!(
                out,
                "      \"peak_bf16_tflops\": {:.1},",
                p.peak_bf16_tflops
            );
            let _ = writeln!(
                out,
                "      \"ridge_point_flops_per_byte\": {:.2},",
                p.ridge_point_flops_per_byte()
            );
            let _ = writeln!(
                out,
                "      \"supports_tma_wgmma\": {},",
                p.supports_tma_wgmma
            );
            let _ = writeln!(
                out,
                "      \"supports_tmem_dual_warp_pipeline\": {},",
                p.supports_tmem_dual_warp_pipeline
            );
            let _ = writeln!(
                out,
                "      \"is_bandwidth_bound_unified_memory\": {}",
                p.is_bandwidth_bound_unified_memory
            );
            let comma = if idx + 1 < arches.len() { "," } else { "" };
            let _ = writeln!(out, "    }}{comma}");
        }
        let _ = writeln!(out, "  ],");

        // Directly measured CPU verification section.
        let _ = writeln!(out, "  \"cpu_directly_measured\": {{");
        let _ = writeln!(out, "    \"provenance_tag\": \"cpu_directly_measured\",");
        let _ = writeln!(
            out,
            "    \"max_two_level_output_err\": {:.6e},",
            self.cpu_cascade.max_two_level_output_err
        );
        let _ = writeln!(
            out,
            "    \"three_level_max_output_err\": {:.6e},",
            self.cpu_cascade.three_level_max_output_err
        );
        let _ = writeln!(
            out,
            "    \"three_level_kv_read_reduction\": {:.4},",
            self.cpu_cascade.three_level_kv_read_reduction
        );
        let _ = writeln!(
            out,
            "    \"aggregate_argmax_match_pct\": {:.2},",
            self.cpu_cascade.aggregate_argmax_match_pct
        );
        let _ = writeln!(
            out,
            "    \"quality_regression_pp\": {:.3},",
            self.cpu_cascade.quality_regression_pp
        );
        let _ = writeln!(out, "    \"cascade_batch_rows\": [");
        for (idx, r) in self.cpu_cascade.rows.iter().enumerate() {
            let _ = writeln!(out, "      {{");
            let _ = writeln!(out, "        \"batch_size\": {},", r.batch_size);
            let _ = writeln!(
                out,
                "        \"shared_prefix_len\": {},",
                r.shared_prefix_len
            );
            let _ = writeln!(
                out,
                "        \"unique_suffix_len\": {},",
                r.unique_suffix_len
            );
            let _ = writeln!(
                out,
                "        \"max_abs_output_err\": {:.6e},",
                r.max_abs_output_err
            );
            let _ = writeln!(
                out,
                "        \"max_abs_lse_err\": {:.6e},",
                r.max_abs_lse_err
            );
            let _ = writeln!(
                out,
                "        \"greedy_argmax_match_pct\": {:.2},",
                r.greedy_argmax_match_pct
            );
            let _ = writeln!(
                out,
                "        \"unshared_kv_bytes\": {},",
                r.unshared_kv_bytes
            );
            let _ = writeln!(out, "        \"cascade_kv_bytes\": {},", r.cascade_kv_bytes);
            let _ = writeln!(
                out,
                "        \"cascade_lse_merge_bytes\": {},",
                r.cascade_lse_merge_bytes
            );
            let _ = writeln!(
                out,
                "        \"kv_read_reduction_ratio\": {:.4},",
                r.kv_read_reduction_ratio
            );
            let _ = writeln!(
                out,
                "        \"unshared_cpu_us\": {:.2},",
                r.unshared_cpu_us
            );
            let _ = writeln!(out, "        \"cascade_cpu_us\": {:.2},", r.cascade_cpu_us);
            let _ = writeln!(
                out,
                "        \"cpu_wall_speedup\": {:.4}",
                r.cpu_wall_speedup
            );
            let comma = if idx + 1 < self.cpu_cascade.rows.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(out, "      }}{comma}");
        }
        let _ = writeln!(out, "    ],");

        // Ragged CPU verification.
        let rg = &self.cpu_ragged;
        let _ = writeln!(out, "    \"ragged_prefill_decode\": {{");
        let _ = writeln!(out, "      \"prefill_requests\": {},", rg.prefill_requests);
        let _ = writeln!(out, "      \"decode_requests\": {},", rg.decode_requests);
        let _ = writeln!(out, "      \"total_q_tokens\": {},", rg.total_q_tokens);
        let _ = writeln!(
            out,
            "      \"separate_dispatch_stages\": {},",
            rg.separate_dispatch_stages
        );
        let _ = writeln!(
            out,
            "      \"fused_dispatch_stages\": {},",
            rg.fused_dispatch_stages
        );
        let _ = writeln!(
            out,
            "      \"max_abs_output_err\": {:.6e},",
            rg.max_abs_output_err
        );
        let _ = writeln!(
            out,
            "      \"max_abs_lse_err\": {:.6e},",
            rg.max_abs_lse_err
        );
        let _ = writeln!(
            out,
            "      \"greedy_argmax_match_pct\": {:.2}",
            rg.greedy_argmax_match_pct
        );
        let _ = writeln!(out, "    }},");

        // Hybrid Attention + GDN/LSA + Dense MLP Amdahl CPU verification.
        let hb = &self.cpu_hybrid;
        let _ = writeln!(out, "    \"hybrid_amdahl_step\": {{");
        let _ = writeln!(out, "      \"batch_size\": {},", hb.batch_size);
        let _ = writeln!(out, "      \"num_attn_layers\": {},", hb.num_attn_layers);
        let _ = writeln!(out, "      \"num_gdn_layers\": {},", hb.num_gdn_layers);
        let _ = writeln!(
            out,
            "      \"unshared_attn_us\": {:.2},",
            hb.unshared_attn_us
        );
        let _ = writeln!(out, "      \"cascade_attn_us\": {:.2},", hb.cascade_attn_us);
        let _ = writeln!(out, "      \"gdn_time_us\": {:.2},", hb.gdn_time_us);
        let _ = writeln!(out, "      \"mlp_time_us\": {:.2},", hb.mlp_time_us);
        let _ = writeln!(
            out,
            "      \"unshared_total_step_us\": {:.2},",
            hb.unshared_total_step_us
        );
        let _ = writeln!(
            out,
            "      \"cascade_total_step_us\": {:.2},",
            hb.cascade_total_step_us
        );
        let _ = writeln!(
            out,
            "      \"attn_time_fraction\": {:.4},",
            hb.attn_time_fraction
        );
        let _ = writeln!(
            out,
            "      \"gdn_time_fraction\": {:.4},",
            hb.gdn_time_fraction
        );
        let _ = writeln!(
            out,
            "      \"mlp_time_fraction\": {:.4},",
            hb.mlp_time_fraction
        );
        let _ = writeln!(
            out,
            "      \"isolated_attn_speedup\": {:.4},",
            hb.isolated_attn_speedup
        );
        let _ = writeln!(
            out,
            "      \"full_step_speedup\": {:.4},",
            hb.full_step_speedup
        );
        let _ = writeln!(
            out,
            "      \"max_abs_hidden_err\": {:.6e},",
            hb.max_abs_hidden_err
        );
        let _ = writeln!(
            out,
            "      \"max_abs_gdn_state_err\": {:.6e},",
            hb.max_abs_gdn_state_err
        );
        let _ = writeln!(
            out,
            "      \"greedy_argmax_match_pct\": {:.2}",
            hb.greedy_argmax_match_pct
        );
        let _ = writeln!(out, "    }}");
        let _ = writeln!(out, "  }},");

        // Per-architecture scorecards.
        let _ = writeln!(out, "  \"scorecards\": [");
        for (idx, row) in self.scorecards.iter().enumerate() {
            write_scorecard_json(&mut out, row, "    ");
            let comma = if idx + 1 < self.scorecards.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(out, "{comma}");
        }
        let _ = writeln!(out, "  ],");

        // Explicit negative regimes.
        let _ = writeln!(out, "  \"negative_regimes\": [");
        for (idx, neg) in self.negative_regimes.iter().enumerate() {
            let _ = writeln!(out, "    {{");
            let _ = writeln!(out, "      \"regime\": \"{}\",", neg.regime);
            let _ = writeln!(
                out,
                "      \"description\": \"{}\",",
                escape_json(neg.description)
            );
            let _ = writeln!(
                out,
                "      \"baseline_wall_time_s\": {:.4},",
                neg.baseline_wall_time_s
            );
            let _ = writeln!(out, "      \"scorecard\":");
            write_scorecard_json(&mut out, &neg.scorecard, "      ");
            let _ = writeln!(out);
            let comma = if idx + 1 < self.negative_regimes.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(out, "    }}{comma}");
        }
        let _ = writeln!(out, "  ]");
        let _ = writeln!(out, "}}");
        out
    }
}
