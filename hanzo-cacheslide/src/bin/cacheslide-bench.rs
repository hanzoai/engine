// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Reproducible CacheSlide benchmark harness (`HIP-1346` / `hanzoai/engine#85`).
//!
//! Grounds every workload in real `hanzoai/engine` repository files (`chat_templates/`,
//! `docs/`, `hanzo-kv-fabric/`, `hanzo-engine/`), evaluates four execution modes
//! (`full_prefill_baseline`, `prefix_cache_only`, `cacheslide_pure_shift`,
//! `cacheslide_selective_recompute`), exposes two explicit negative regimes, traces the
//! TTFT vs. KL-divergence Pareto frontier, measures multi-session GPU KV deduplication,
//! audits cross-tenant/ACL isolation, and writes raw JSON receipts to
//! `docs/research/hip-1346-cacheslide/results.json`.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hanzo_cacheslide::{
    apply_rope_in_place, evaluate_quality, CorpusTokenizer, ExecutionMode, IsolationScope, KvDtype,
    PrefixKvCache, PromptSegment, QualityMetrics, RopeConfig, RopeLayout, SegmentKey, SegmentKind,
    SegmentKvCache, SelectiveRecomputeConfig, TransformerConfig, TransformerModel,
};

struct CliArgs {
    seed: u64,
    trials: usize,
    out_path: PathBuf,
}

fn parse_args() -> CliArgs {
    let mut seed = 1346u64;
    let mut trials = 6usize;
    let mut out_path = PathBuf::from("docs/research/hip-1346-cacheslide/results.json");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().expect("invalid --seed u64");
                i += 2;
            }
            "--trials" if i + 1 < args.len() => {
                trials = args[i + 1].parse().expect("invalid --trials usize");
                assert!(trials >= 1, "--trials must be >= 1");
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cacheslide-bench [--seed 1346] [--trials 6] [--out docs/research/hip-1346-cacheslide/results.json]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    CliArgs {
        seed,
        trials,
        out_path,
    }
}

fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn load_repo_text(root: &Path, rel_path: &str) -> String {
    let full = root.join(rel_path);
    fs::read_to_string(&full).unwrap_or_else(|e| {
        panic!("Failed to read repo corpus file {}: {e}", full.display());
    })
}

struct LatencyStats {
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    mean_ms: f64,
}

fn summarize_latencies(mut samples_ms: Vec<f64>) -> LatencyStats {
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = samples_ms.len();
    let pct = |p: f64| -> f64 {
        let idx = ((p * (n.saturating_sub(1)) as f64).round() as usize).min(n - 1);
        samples_ms[idx]
    };
    let mean_ms = samples_ms.iter().sum::<f64>() / (n as f64);
    LatencyStats {
        p50_ms: pct(0.50),
        p95_ms: pct(0.95),
        p99_ms: pct(0.99),
        mean_ms,
    }
}

struct ModeBenchResult {
    mode_name: String,
    latency: LatencyStats,
    speedup_vs_full: f64,
    total_tokens: usize,
    tokens_reused: usize,
    tokens_reused_pct: f64,
    tokens_computed: usize,
    tokens_selectively_recomputed: usize,
    segments_shifted: usize,
    segments_exact_hit: usize,
    prefill_flops: u64,
    flops_saved_pct: f64,
    max_layer1_k_drift: f32,
    drift_gate_escalations: usize,
    quality: QualityMetrics,
    top1_agreement_pct: f64,
}

struct WorkloadSpec {
    name: String,
    regime: String,
    description: String,
    source_files: Vec<String>,
    warmup_prompts: Vec<Vec<PromptSegment>>,
    eval_prompts: Vec<Vec<PromptSegment>>,
    selective_cfg: SelectiveRecomputeConfig,
    /// When `Some(min_toks)`, override CacheSlide `min_segment_tokens` (used for Negative Regime #2).
    cache_min_tokens: usize,
}

fn run_workload(
    model: &TransformerModel,
    scope: &IsolationScope,
    spec: &WorkloadSpec,
    trials: usize,
) -> Vec<ModeBenchResult> {
    let budget = 64 * 1024 * 1024; // 64 MiB bounded cache budget

    // Precompute exact full-prefill ground truth for each eval prompt.
    let mut exact_logits_per_prompt = Vec::with_capacity(spec.eval_prompts.len());
    let mut full_flops_sum = 0u64;
    let mut full_samples_ms = Vec::with_capacity(trials * spec.eval_prompts.len());

    // Warmup CPU caches once.
    for prompt in &spec.eval_prompts {
        let out = model.prefill_segments(scope, prompt, ExecutionMode::FullPrefill, None, None);
        exact_logits_per_prompt.push(out.logits);
        full_flops_sum += out.prefill_flops;
    }

    let mut full_last_total = 0usize;
    for _ in 0..trials {
        for prompt in &spec.eval_prompts {
            let t0 = Instant::now();
            let out = model.prefill_segments(scope, prompt, ExecutionMode::FullPrefill, None, None);
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            full_samples_ms.push(dt);
            full_last_total = out.total_tokens;
        }
    }

    let full_latency = summarize_latencies(full_samples_ms);
    let full_mean_flops = full_flops_sum / (spec.eval_prompts.len() as u64);
    let ref_q = evaluate_quality(&exact_logits_per_prompt[0], &exact_logits_per_prompt[0]);

    let mut results = Vec::new();
    results.push(ModeBenchResult {
        mode_name: "full_prefill_baseline".to_string(),
        latency: full_latency,
        speedup_vs_full: 1.0,
        total_tokens: full_last_total,
        tokens_reused: 0,
        tokens_reused_pct: 0.0,
        tokens_computed: full_last_total,
        tokens_selectively_recomputed: 0,
        segments_shifted: 0,
        segments_exact_hit: 0,
        prefill_flops: full_mean_flops,
        flops_saved_pct: 0.0,
        max_layer1_k_drift: 0.0,
        drift_gate_escalations: 0,
        quality: ref_q,
        top1_agreement_pct: 100.0,
    });

    let modes: Vec<(&str, ExecutionMode)> = vec![
        ("prefix_cache_only", ExecutionMode::PrefixCacheOnly),
        ("cacheslide_pure_shift", ExecutionMode::CacheSlidePureShift),
        (
            "cacheslide_selective_recompute",
            ExecutionMode::CacheSlideSelectiveRecompute(spec.selective_cfg),
        ),
    ];

    for (mode_name, mode) in modes {
        let mut samples_ms = Vec::with_capacity(trials * spec.eval_prompts.len());
        let mut sum_reused = 0usize;
        let mut sum_computed = 0usize;
        let mut sum_sel_recomp = 0usize;
        let mut sum_shifted = 0usize;
        let mut sum_exact_hit = 0usize;
        let mut sum_flops = 0u64;
        let mut sum_kl = 0.0f64;
        let mut sum_js = 0.0f64;
        let mut sum_cos = 0.0f64;
        let mut max_logit_diff = 0.0f32;
        let mut top1_hits = 0usize;
        let mut sum_top5 = 0.0f64;
        let mut sum_ent = 0.0f64;
        let mut max_drift = 0.0f32;
        let mut total_escalations = 0usize;

        for trial in 0..trials {
            let pcache = PrefixKvCache::new(budget);
            let scache = SegmentKvCache::new(budget, spec.cache_min_tokens);

            // Populate cache with warmup turns (same-tenant prior turns).
            for wprompt in &spec.warmup_prompts {
                let _ = model.prefill_segments(scope, wprompt, mode, Some(&pcache), Some(&scache));
            }

            for (pidx, prompt) in spec.eval_prompts.iter().enumerate() {
                let t0 = Instant::now();
                let out = model.prefill_segments(scope, prompt, mode, Some(&pcache), Some(&scache));
                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                samples_ms.push(dt);

                if trial == 0 {
                    sum_reused += out.tokens_reused;
                    sum_computed += out.tokens_computed;
                    sum_sel_recomp += out.tokens_selectively_recomputed;
                    sum_shifted += out.segments_shifted;
                    sum_exact_hit += out.segments_exact_hit;
                    sum_flops += out.prefill_flops;
                    max_drift = max_drift.max(out.max_layer1_k_drift);
                    total_escalations += out.drift_gate_escalations;

                    let q = evaluate_quality(&exact_logits_per_prompt[pidx], &out.logits);
                    sum_kl += q.kl_divergence;
                    sum_js += q.js_divergence;
                    sum_cos += q.logit_cosine_similarity;
                    max_logit_diff = max_logit_diff.max(q.max_abs_logit_diff);
                    if q.top1_agree {
                        top1_hits += 1;
                    }
                    sum_top5 += q.top5_overlap;
                    sum_ent += q.ref_entropy_nats;
                }
            }
        }

        let n_eval = spec.eval_prompts.len();
        let latency = summarize_latencies(samples_ms);
        let speedup = results[0].latency.p50_ms / latency.p50_ms.max(1e-9);
        let mean_flops = sum_flops / (n_eval as u64);
        let flops_saved_pct = 100.0 * (1.0 - (mean_flops as f64) / (full_mean_flops.max(1) as f64));
        let mean_reused = sum_reused / n_eval;
        let mean_computed = sum_computed / n_eval;
        let mean_sel_recomp = sum_sel_recomp / n_eval;

        results.push(ModeBenchResult {
            mode_name: mode_name.to_string(),
            latency,
            speedup_vs_full: speedup,
            total_tokens: full_last_total,
            tokens_reused: mean_reused,
            tokens_reused_pct: 100.0 * (mean_reused as f64) / (full_last_total.max(1) as f64),
            tokens_computed: mean_computed,
            tokens_selectively_recomputed: mean_sel_recomp,
            segments_shifted: sum_shifted / n_eval,
            segments_exact_hit: sum_exact_hit / n_eval,
            prefill_flops: mean_flops,
            flops_saved_pct,
            max_layer1_k_drift: max_drift,
            drift_gate_escalations: total_escalations,
            quality: QualityMetrics {
                kl_divergence: sum_kl / (n_eval as f64),
                js_divergence: sum_js / (n_eval as f64),
                logit_cosine_similarity: sum_cos / (n_eval as f64),
                max_abs_logit_diff: max_logit_diff,
                top1_agree: top1_hits == n_eval,
                top5_overlap: sum_top5 / (n_eval as f64),
                ref_entropy_nats: sum_ent / (n_eval as f64),
            },
            top1_agreement_pct: 100.0 * (top1_hits as f64) / (n_eval as f64),
        });
    }

    results
}

struct ParetoPoint {
    recompute_ratio: f32,
    ttft_p50_ms: f64,
    speedup_vs_full: f64,
    flops_saved_pct: f64,
    tokens_reused_pct: f64,
    kl_divergence: f64,
    logit_cosine_similarity: f64,
    top1_agreement_pct: f64,
    top5_overlap: f64,
}

fn run_pareto_sweep(
    model: &TransformerModel,
    scope: &IsolationScope,
    spec: &WorkloadSpec,
    full_p50_ms: f64,
    full_flops: u64,
    trials: usize,
) -> Vec<ParetoPoint> {
    let ratios = [0.0f32, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.0];
    let budget = 64 * 1024 * 1024;
    let exact_logits: Vec<Vec<f32>> = spec
        .eval_prompts
        .iter()
        .map(|p| {
            model
                .prefill_segments(scope, p, ExecutionMode::FullPrefill, None, None)
                .logits
        })
        .collect();

    let mut points = Vec::with_capacity(ratios.len());
    for &ratio in &ratios {
        let mode = if ratio == 0.0 {
            ExecutionMode::CacheSlidePureShift
        } else {
            ExecutionMode::CacheSlideSelectiveRecompute(SelectiveRecomputeConfig {
                recompute_ratio: ratio,
                lead_boundary_tokens: if ratio >= 0.05 { 4 } else { 0 },
                trail_boundary_tokens: if ratio >= 0.05 { 2 } else { 0 },
                auto_escalate_drift_threshold: None,
                escalated_recompute_ratio: 1.0,
            })
        };

        let mut samples = Vec::with_capacity(trials * spec.eval_prompts.len());
        let mut sum_flops = 0u64;
        let mut sum_reused = 0usize;
        let mut total_toks = 0usize;
        let mut sum_kl = 0.0f64;
        let mut sum_cos = 0.0f64;
        let mut top1_hits = 0usize;
        let mut sum_top5 = 0.0f64;

        for trial in 0..trials {
            let scache = SegmentKvCache::new(budget, 32);
            for w in &spec.warmup_prompts {
                let _ = model.prefill_segments(scope, w, mode, None, Some(&scache));
            }
            for (pidx, p) in spec.eval_prompts.iter().enumerate() {
                let t0 = Instant::now();
                let out = model.prefill_segments(scope, p, mode, None, Some(&scache));
                samples.push(t0.elapsed().as_secs_f64() * 1000.0);
                if trial == 0 {
                    sum_flops += out.prefill_flops;
                    sum_reused += out.tokens_reused;
                    total_toks = out.total_tokens;
                    let q = evaluate_quality(&exact_logits[pidx], &out.logits);
                    sum_kl += q.kl_divergence;
                    sum_cos += q.logit_cosine_similarity;
                    if q.top1_agree {
                        top1_hits += 1;
                    }
                    sum_top5 += q.top5_overlap;
                }
            }
        }

        let n = spec.eval_prompts.len();
        let lat = summarize_latencies(samples);
        let mean_flops = sum_flops / (n as u64);
        points.push(ParetoPoint {
            recompute_ratio: ratio,
            ttft_p50_ms: lat.p50_ms,
            speedup_vs_full: full_p50_ms / lat.p50_ms.max(1e-9),
            flops_saved_pct: 100.0 * (1.0 - (mean_flops as f64) / (full_flops.max(1) as f64)),
            tokens_reused_pct: 100.0 * ((sum_reused / n) as f64) / (total_toks.max(1) as f64),
            kl_divergence: sum_kl / (n as f64),
            logit_cosine_similarity: sum_cos / (n as f64),
            top1_agreement_pct: 100.0 * (top1_hits as f64) / (n as f64),
            top5_overlap: sum_top5 / (n as f64),
        });
    }
    points
}

struct RopeAuditReceipt {
    max_group_err_neox: f32,
    max_group_err_interleaved: f32,
    passed_under_1e5: bool,
}

fn run_rope_audit() -> RopeAuditReceipt {
    let check_layout = |layout: RopeLayout| -> f32 {
        let cfg = RopeConfig::standard(64, 500_000.0, layout);
        let raw: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.37).sin()).collect();
        let mut worst = 0.0f32;
        for &(a, b) in &[(128i64, 256i64), (512, -192), (1024, 512), (4096, -3072)] {
            let mut composed = raw.clone();
            apply_rope_in_place(&cfg, &mut composed, b);
            apply_rope_in_place(&cfg, &mut composed, a);

            let mut direct = raw.clone();
            apply_rope_in_place(&cfg, &mut direct, a + b);

            for (x, y) in composed.iter().zip(direct.iter()) {
                worst = worst.max((x - y).abs());
            }
        }
        worst
    };
    let neox = check_layout(RopeLayout::GptNeoX);
    let inter = check_layout(RopeLayout::Interleaved);
    RopeAuditReceipt {
        max_group_err_neox: neox,
        max_group_err_interleaved: inter,
        passed_under_1e5: neox < 1e-5 && inter < 1e-5,
    }
}

struct TenantIsolationReceipt {
    total_cross_scope_probes: usize,
    cross_tenant_leaks: usize,
    cross_acl_leaks: usize,
    cross_model_leaks: usize,
    cross_dtype_leaks: usize,
    evicted_alpha_entries: usize,
    beta_retained_after_alpha_eviction: bool,
    leak_rate_pct: f64,
}

fn run_tenant_isolation_audit(
    model: &TransformerModel,
    shared_tool: &PromptSegment,
    sys: &PromptSegment,
    query: &PromptSegment,
) -> TenantIsolationReceipt {
    let scache = SegmentKvCache::new(32 * 1024 * 1024, 32);
    let alpha_eng = IsolationScope::new("tenant_alpha", "acl_eng", "zen-7b", KvDtype::F16);
    let beta_eng = IsolationScope::new("tenant_beta", "acl_eng", "zen-7b", KvDtype::F16);
    let alpha_fin = IsolationScope::new("tenant_alpha", "acl_finance", "zen-7b", KvDtype::F16);
    let alpha_32b = IsolationScope::new("tenant_alpha", "acl_eng", "zen-32b", KvDtype::F16);
    let alpha_int4 = IsolationScope::new("tenant_alpha", "acl_eng", "zen-7b", KvDtype::Int4Asym);

    // Populate tenant_alpha / acl_eng / zen-7b / F16.
    let prompt = [sys.clone(), shared_tool.clone(), query.clone()];
    let _ = model.prefill_segments(
        &alpha_eng,
        &prompt,
        ExecutionMode::CacheSlidePureShift,
        None,
        Some(&scache),
    );

    let mut cross_tenant_leaks = 0usize;
    let mut cross_acl_leaks = 0usize;
    let mut cross_model_leaks = 0usize;
    let mut cross_dtype_leaks = 0usize;

    for _ in 0..25 {
        if scache
            .get(&SegmentKey::from_segment(beta_eng.clone(), shared_tool))
            .is_some()
        {
            cross_tenant_leaks += 1;
        }
        if scache
            .get(&SegmentKey::from_segment(alpha_fin.clone(), shared_tool))
            .is_some()
        {
            cross_acl_leaks += 1;
        }
        if scache
            .get(&SegmentKey::from_segment(alpha_32b.clone(), shared_tool))
            .is_some()
        {
            cross_model_leaks += 1;
        }
        if scache
            .get(&SegmentKey::from_segment(alpha_int4.clone(), shared_tool))
            .is_some()
        {
            cross_dtype_leaks += 1;
        }
    }

    // Also insert tenant_beta, then evict tenant_alpha and verify tenant_beta survives.
    let _ = model.prefill_segments(
        &beta_eng,
        &prompt,
        ExecutionMode::CacheSlidePureShift,
        None,
        Some(&scache),
    );
    let evicted = scache.evict_tenant("tenant_alpha");
    let beta_ok = scache
        .get(&SegmentKey::from_segment(beta_eng, shared_tool))
        .is_some();

    let total_probes = 100usize;
    let total_leaks = cross_tenant_leaks + cross_acl_leaks + cross_model_leaks + cross_dtype_leaks;
    TenantIsolationReceipt {
        total_cross_scope_probes: total_probes,
        cross_tenant_leaks,
        cross_acl_leaks,
        cross_model_leaks,
        cross_dtype_leaks,
        evicted_alpha_entries: evicted,
        beta_retained_after_alpha_eviction: beta_ok,
        leak_rate_pct: 100.0 * (total_leaks as f64) / (total_probes as f64),
    }
}

struct MemoryDedupReceipt {
    num_sessions: usize,
    shared_tool_tokens: usize,
    prefix_cache_entries: usize,
    cacheslide_entries: usize,
    prefix_cache_gpu_bytes_f16: usize,
    cacheslide_gpu_bytes_f16: usize,
    gpu_memory_reduction_ratio: f64,
    projected_llama8b_prefix_mb_f16: f64,
    projected_llama8b_cacheslide_mb_f16: f64,
    projected_llama8b_cacheslide_mb_int4: f64,
    budget_respected: bool,
}

fn run_memory_dedup_audit(
    model: &TransformerModel,
    scope: &IsolationScope,
    tok: &CorpusTokenizer,
    shared_tool: &PromptSegment,
) -> MemoryDedupReceipt {
    let num_sessions = 16usize;
    let budget = 64 * 1024 * 1024;
    let pcache = PrefixKvCache::new(budget);
    let scache = SegmentKvCache::new(budget, 32);

    for s in 0..num_sessions {
        let header = PromptSegment::new(
            SegmentKind::SystemPrompt,
            format!("session_hdr_{s}"),
            tok.encode_exact(
                &format!("Session {s}: user_id=u{s:04} tz=UTC locale=en_US policy=strict"),
                64,
            ),
        );
        let query = PromptSegment::new(
            SegmentKind::UserQuery,
            format!("session_q_{s}"),
            tok.encode_exact(&format!("Run tool inspection for session {s}"), 32),
        );
        let prompt = [header, shared_tool.clone(), query];
        let _ = model.prefill_segments(
            scope,
            &prompt,
            ExecutionMode::PrefixCacheOnly,
            Some(&pcache),
            None,
        );
        let _ = model.prefill_segments(
            scope,
            &prompt,
            ExecutionMode::CacheSlidePureShift,
            None,
            Some(&scache),
        );
    }

    let ps = pcache.stats();
    let ss = scache.stats();

    // Also project realistic 32-layer Llama-3-8B GQA (8 KV heads, head_dim=128) GPU KV RAM
    // for the exact same 16-session workload (headers + deduplicated vs duplicated 256-tok tool).
    let prefix_total_toks = num_sessions * (64 + shared_tool.len() + 32);
    let slide_total_toks = num_sessions * (64 + 32) + shared_tool.len();
    let proj_prefix_f16 =
        KvDtype::F16.kv_bytes(32, prefix_total_toks, 8, 128) as f64 / (1024.0 * 1024.0);
    let proj_slide_f16 =
        KvDtype::F16.kv_bytes(32, slide_total_toks, 8, 128) as f64 / (1024.0 * 1024.0);
    let proj_slide_int4 =
        KvDtype::Int4Asym.kv_bytes(32, slide_total_toks, 8, 128) as f64 / (1024.0 * 1024.0);

    MemoryDedupReceipt {
        num_sessions,
        shared_tool_tokens: shared_tool.len(),
        prefix_cache_entries: ps.entry_count,
        cacheslide_entries: ss.entry_count,
        prefix_cache_gpu_bytes_f16: ps.current_gpu_kv_bytes,
        cacheslide_gpu_bytes_f16: ss.current_gpu_kv_bytes,
        gpu_memory_reduction_ratio: (ps.current_gpu_kv_bytes as f64)
            / (ss.current_gpu_kv_bytes.max(1) as f64),
        projected_llama8b_prefix_mb_f16: proj_prefix_f16,
        projected_llama8b_cacheslide_mb_f16: proj_slide_f16,
        projected_llama8b_cacheslide_mb_int4: proj_slide_int4,
        budget_respected: ps.peak_bytes <= ps.max_bytes && ss.peak_bytes <= ss.max_bytes,
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
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
    out.push('"');
    out
}

fn main() {
    let args = parse_args();
    let root = repo_root();
    let cfg = TransformerConfig::benchmark_default();
    let model = TransformerModel::new_deterministic(cfg.clone(), args.seed);
    let tok = CorpusTokenizer::new(cfg.vocab_size);
    let scope = IsolationScope::new(
        "tenant_hanzo_bench",
        "acl_eng",
        "zen-bench-6l",
        cfg.kv_dtype,
    );

    // Load real hanzoai/engine repository files for grounded prompt corpora.
    let hermes_jinja = load_repo_text(&root, "chat_templates/nousresearch_hermes3_tool_use.jinja");
    let mistral_jinja = load_repo_text(&root, "chat_templates/mistral_small_tool_call.jinja");
    let deepseek_jinja = load_repo_text(&root, "chat_templates/deepseek_tool_call.jinja");
    let smollm_jinja = load_repo_text(&root, "chat_templates/smollm3.jinja");
    let engine_md = load_repo_text(&root, "docs/ENGINE.md");
    let flash_md = load_repo_text(&root, "docs/FLASH_ATTENTION.md");
    let anthropic_md = load_repo_text(&root, "docs/anthropic-api.md");
    let cli_md = load_repo_text(&root, "docs/CLI_CONFIG.md");
    let composition_rs = load_repo_text(&root, "hanzo-kv-fabric/src/composition.rs");
    let block_hash_rs = load_repo_text(&root, "hanzo-engine/src/paged_attention/block_hash.rs");
    let kv_iso_rs = load_repo_text(&root, "hanzo-engine/src/paged_attention/kv_isolation.rs");
    let agents_md = load_repo_text(&root, "docs/AGENTS.md");

    // Reusable grounded segments.
    let sys_a = PromptSegment::new(
        SegmentKind::SystemPrompt,
        "sys_session_a",
        tok.encode_exact(&agents_md[..agents_md.len().min(900)], 64),
    );
    let sys_a_prime = PromptSegment::new(
        SegmentKind::SystemPrompt,
        "sys_session_a_updated",
        tok.encode_exact(
            &format!(
                "SESSION_ID=2026-11-26-turn2 USER_ROLE=staff_engineer TENANT=hanzo\n{}",
                &agents_md[900..agents_md.len().min(2200)]
            ),
            128,
        ),
    );
    let tool_bundle_b = PromptSegment::new(
        SegmentKind::ToolDefinition,
        "mcp_tool_bundle_hermes_mistral",
        tok.encode_exact(&format!("{hermes_jinja}\n{mistral_jinja}"), 256),
    );

    let tool_1 = PromptSegment::new(
        SegmentKind::ToolDefinition,
        "tool_hermes",
        tok.encode_exact(&hermes_jinja, 96),
    );
    let tool_2 = PromptSegment::new(
        SegmentKind::ToolDefinition,
        "tool_mistral",
        tok.encode_exact(&mistral_jinja, 96),
    );
    let tool_3 = PromptSegment::new(
        SegmentKind::ToolDefinition,
        "tool_deepseek",
        tok.encode_exact(&deepseek_jinja, 96),
    );
    let tool_4_new = PromptSegment::new(
        SegmentKind::ToolDefinition,
        "tool_smollm_new",
        tok.encode_exact(&smollm_jinja, 64),
    );

    let doc_1 = PromptSegment::new(
        SegmentKind::RagChunk,
        "docs_engine_md",
        tok.encode_exact(&engine_md, 128),
    );
    let doc_2 = PromptSegment::new(
        SegmentKind::RagChunk,
        "docs_flash_attn_md",
        tok.encode_exact(&flash_md, 128),
    );
    let doc_3 = PromptSegment::new(
        SegmentKind::RagChunk,
        "docs_anthropic_api_md",
        tok.encode_exact(&anthropic_md, 128),
    );
    let doc_4 = PromptSegment::new(
        SegmentKind::RagChunk,
        "docs_cli_config_md",
        tok.encode_exact(&cli_md, 128),
    );

    let repo_ctx_a = PromptSegment::new(
        SegmentKind::RagChunk,
        "hanzo_kv_fabric_composition_rs",
        tok.encode_exact(&composition_rs, 160),
    );
    let tool_spec_ds = PromptSegment::new(
        SegmentKind::ToolDefinition,
        "deepseek_tool_spec",
        tok.encode_exact(&deepseek_jinja, 128),
    );
    let mid_reminder = PromptSegment::new(
        SegmentKind::ConversationTurn,
        "mid_prompt_state_update",
        tok.encode_exact(&kv_iso_rs[..kv_iso_rs.len().min(800)], 64),
    );

    let q_turn1 = PromptSegment::new(
        SegmentKind::UserQuery,
        "query_turn1",
        tok.encode_exact("Inspect KV cache isolation and emit tool call JSON.", 32),
    );
    let q_eval_1 = PromptSegment::new(
        SegmentKind::UserQuery,
        "query_eval_1",
        tok.encode_exact(
            "Call the paged attention diagnostic tool with block_size 16.",
            32,
        ),
    );
    let q_eval_2 = PromptSegment::new(
        SegmentKind::UserQuery,
        "query_eval_2",
        tok.encode_exact(
            "Summarize RoPE offset translation and fail-closed chunk rules.",
            32,
        ),
    );
    let q_eval_3 = PromptSegment::new(
        SegmentKind::UserQuery,
        "query_eval_3",
        tok.encode_exact(
            "Compare FlashAttention prefill flags with Anthropic API tool schemas.",
            32,
        ),
    );

    let q_eval_4 = PromptSegment::new(
        SegmentKind::UserQuery,
        "query_eval_4",
        tok.encode_exact(
            "Validate tenant cache salt isolation under LRU eviction.",
            32,
        ),
    );
    let q_eval_5 = PromptSegment::new(
        SegmentKind::UserQuery,
        "query_eval_5",
        tok.encode_exact("Explain KIVI asymmetric quantization bytes per token.", 32),
    );

    let make_dyn_sys = |idx: usize, extra: &str| -> PromptSegment {
        PromptSegment::new(
            SegmentKind::SystemPrompt,
            format!("sys_session_dyn_{idx}"),
            tok.encode_exact(
                &format!("SESSION_ID=2026-11-26-s{idx} USER_ID=u{idx:04} ROLE=staff_eng\n{extra}"),
                128,
            ),
        )
    };
    let make_mid_state = |idx: usize, extra: &str| -> PromptSegment {
        PromptSegment::new(
            SegmentKind::ConversationTurn,
            format!("mid_state_turn_{idx}"),
            tok.encode_exact(&format!("TURN={idx} STATE_REV={idx}09\n{extra}"), 64),
        )
    };
    let _ = (&sys_a_prime, &mid_reminder);

    // Build all 6 workloads (4 positive + 2 explicit negative regimes).
    let workloads = vec![
        WorkloadSpec {
            name: "tool_schema_shifted_system".to_string(),
            regime: "positive".to_string(),
            description: "5 distinct dynamic session system prompts (128 tok) shift shared 256-token MCP tool schema bundle by delta_pos=+64".to_string(),
            source_files: vec![
                "docs/AGENTS.md".to_string(),
                "chat_templates/nousresearch_hermes3_tool_use.jinja".to_string(),
                "chat_templates/mistral_small_tool_call.jinja".to_string(),
            ],
            warmup_prompts: vec![vec![
                sys_a.clone(),
                tool_bundle_b.clone(),
                q_turn1.clone(),
            ]],
            eval_prompts: vec![
                vec![make_dyn_sys(1, &agents_md[900..agents_md.len().min(2000)]), tool_bundle_b.clone(), q_eval_1.clone()],
                vec![make_dyn_sys(2, &engine_md[..engine_md.len().min(1100)]), tool_bundle_b.clone(), q_eval_2.clone()],
                vec![make_dyn_sys(3, &cli_md[..cli_md.len().min(1100)]), tool_bundle_b.clone(), q_eval_3.clone()],
                vec![make_dyn_sys(4, &flash_md[..flash_md.len().min(1100)]), tool_bundle_b.clone(), q_eval_4.clone()],
                vec![make_dyn_sys(5, &anthropic_md[..anthropic_md.len().min(1100)]), tool_bundle_b.clone(), q_eval_5.clone()],
            ],
            selective_cfg: SelectiveRecomputeConfig {
                recompute_ratio: 0.20,
                lead_boundary_tokens: 4,
                trail_boundary_tokens: 2,
                auto_escalate_drift_threshold: None,
                escalated_recompute_ratio: 1.0,
            },
            cache_min_tokens: 32,
        },
        WorkloadSpec {
            name: "multi_tool_subset_reorder".to_string(),
            regime: "positive".to_string(),
            description: "Agent router selects 5 distinct reordered subsets of cached MCP tools after system prompt (delta_pos in {-192,-96,+96})".to_string(),
            source_files: vec![
                "chat_templates/nousresearch_hermes3_tool_use.jinja".to_string(),
                "chat_templates/mistral_small_tool_call.jinja".to_string(),
                "chat_templates/deepseek_tool_call.jinja".to_string(),
                "chat_templates/smollm3.jinja".to_string(),
            ],
            warmup_prompts: vec![vec![
                sys_a.clone(),
                tool_1.clone(),
                tool_2.clone(),
                tool_3.clone(),
                q_turn1.clone(),
            ]],
            eval_prompts: vec![
                vec![sys_a.clone(), tool_3.clone(), tool_2.clone(), tool_4_new.clone(), q_eval_1.clone()],
                vec![sys_a.clone(), tool_2.clone(), tool_3.clone(), tool_4_new.clone(), q_eval_2.clone()],
                vec![sys_a.clone(), tool_3.clone(), tool_1.clone(), tool_4_new.clone(), q_eval_3.clone()],
                vec![sys_a.clone(), tool_2.clone(), tool_1.clone(), tool_4_new.clone(), q_eval_4.clone()],
                vec![sys_a.clone(), tool_4_new.clone(), tool_3.clone(), tool_2.clone(), q_eval_5.clone()],
            ],
            selective_cfg: SelectiveRecomputeConfig {
                recompute_ratio: 0.20,
                lead_boundary_tokens: 4,
                trail_boundary_tokens: 2,
                auto_escalate_drift_threshold: None,
                escalated_recompute_ratio: 1.0,
            },
            cache_min_tokens: 32,
        },
        WorkloadSpec {
            name: "rag_shuffled_chunks".to_string(),
            regime: "positive".to_string(),
            description: "Reranked 3x128-token Hanzo docs [Doc_3, Doc_1, Doc_4] after 5 distinct 96-token query preambles (delta_pos=-224,+160,-96)".to_string(),
            source_files: vec![
                "docs/ENGINE.md".to_string(),
                "docs/FLASH_ATTENTION.md".to_string(),
                "docs/anthropic-api.md".to_string(),
                "docs/CLI_CONFIG.md".to_string(),
            ],
            warmup_prompts: vec![vec![
                sys_a.clone(),
                doc_1.clone(),
                doc_2.clone(),
                doc_3.clone(),
                doc_4.clone(),
                q_turn1.clone(),
            ]],
            eval_prompts: vec![
                vec![
                    PromptSegment::new(SegmentKind::SystemPrompt, "rag_preamble_1", tok.encode_exact(&cli_md[..cli_md.len().min(1000)], 96)),
                    doc_3.clone(), doc_1.clone(), doc_4.clone(), q_eval_1.clone(),
                ],
                vec![
                    PromptSegment::new(SegmentKind::SystemPrompt, "rag_preamble_2", tok.encode_exact(&flash_md[..flash_md.len().min(1000)], 96)),
                    doc_3.clone(), doc_1.clone(), doc_4.clone(), q_eval_2.clone(),
                ],
                vec![
                    PromptSegment::new(SegmentKind::SystemPrompt, "rag_preamble_3", tok.encode_exact(&anthropic_md[..anthropic_md.len().min(1000)], 96)),
                    doc_3.clone(), doc_1.clone(), doc_4.clone(), q_eval_3.clone(),
                ],
                vec![
                    PromptSegment::new(SegmentKind::SystemPrompt, "rag_preamble_4", tok.encode_exact(&composition_rs[..composition_rs.len().min(1000)], 96)),
                    doc_3.clone(), doc_1.clone(), doc_4.clone(), q_eval_4.clone(),
                ],
                vec![
                    PromptSegment::new(SegmentKind::SystemPrompt, "rag_preamble_5", tok.encode_exact(&block_hash_rs[..block_hash_rs.len().min(1000)], 96)),
                    doc_3.clone(), doc_1.clone(), doc_4.clone(), q_eval_5.clone(),
                ],
            ],
            selective_cfg: SelectiveRecomputeConfig {
                recompute_ratio: 0.20,
                lead_boundary_tokens: 4,
                trail_boundary_tokens: 2,
                auto_escalate_drift_threshold: None,
                escalated_recompute_ratio: 1.0,
            },
            cache_min_tokens: 32,
        },
        WorkloadSpec {
            name: "multiturn_sliding_history".to_string(),
            regime: "positive".to_string(),
            description: "5 distinct mid-prompt 64-token state reminders inserted between System and [RepoContext(160), ToolSpec(128)], shifting both by +64".to_string(),
            source_files: vec![
                "hanzo-kv-fabric/src/composition.rs".to_string(),
                "hanzo-engine/src/paged_attention/kv_isolation.rs".to_string(),
                "chat_templates/deepseek_tool_call.jinja".to_string(),
            ],
            warmup_prompts: vec![vec![
                sys_a.clone(),
                repo_ctx_a.clone(),
                tool_spec_ds.clone(),
                q_turn1.clone(),
            ]],
            eval_prompts: vec![
                vec![sys_a.clone(), make_mid_state(1, &kv_iso_rs[..kv_iso_rs.len().min(800)]), repo_ctx_a.clone(), tool_spec_ds.clone(), q_eval_1.clone()],
                vec![sys_a.clone(), make_mid_state(2, &block_hash_rs[..block_hash_rs.len().min(800)]), repo_ctx_a.clone(), tool_spec_ds.clone(), q_eval_2.clone()],
                vec![sys_a.clone(), make_mid_state(3, &engine_md[..engine_md.len().min(800)]), repo_ctx_a.clone(), tool_spec_ds.clone(), q_eval_3.clone()],
                vec![sys_a.clone(), make_mid_state(4, &cli_md[..cli_md.len().min(800)]), repo_ctx_a.clone(), tool_spec_ds.clone(), q_eval_4.clone()],
                vec![sys_a.clone(), make_mid_state(5, &smollm_jinja[..smollm_jinja.len().min(800)]), repo_ctx_a.clone(), tool_spec_ds.clone(), q_eval_5.clone()],
            ],
            selective_cfg: SelectiveRecomputeConfig {
                recompute_ratio: 0.20,
                lead_boundary_tokens: 4,
                trail_boundary_tokens: 2,
                auto_escalate_drift_threshold: None,
                escalated_recompute_ratio: 1.0,
            },
            cache_min_tokens: 32,
        },
        // NEGATIVE REGIME #1: High cross-segment semantic coupling / order-sensitive context
        WorkloadSpec {
            name: "neg1_high_cross_segment_coupling".to_string(),
            regime: "negative_1_high_coupling".to_string(),
            description: "Order-sensitive segment originally cached at pos=0 (zero prior context) moved to pos=256 after 5 dense contrasting prefixes; pure shift suffers elevated KL unless Layer-1 drift gate escalates recomputation".to_string(),
            source_files: vec![
                "hanzo-engine/src/paged_attention/block_hash.rs".to_string(),
                "hanzo-engine/src/paged_attention/kv_isolation.rs".to_string(),
            ],
            warmup_prompts: vec![vec![
                PromptSegment::new(
                    SegmentKind::RagChunk,
                    "coupled_block_hash_seg",
                    tok.encode_exact(&block_hash_rs, 96),
                ),
                q_turn1.clone(),
            ]],
            eval_prompts: vec![
                vec![
                    PromptSegment::new(SegmentKind::SystemPrompt, "heavy_override_prefix_1", tok.encode_exact(&kv_iso_rs, 256)),
                    PromptSegment::new(SegmentKind::RagChunk, "coupled_block_hash_seg", tok.encode_exact(&block_hash_rs, 96)),
                    PromptSegment::new(SegmentKind::UserQuery, "short_coupled_q1", tok.encode_exact("Verify salt hash:", 8)),
                ],
                vec![
                    PromptSegment::new(SegmentKind::SystemPrompt, "heavy_override_prefix_2", tok.encode_exact(&composition_rs, 256)),
                    PromptSegment::new(SegmentKind::RagChunk, "coupled_block_hash_seg", tok.encode_exact(&block_hash_rs, 96)),
                    PromptSegment::new(SegmentKind::UserQuery, "short_coupled_q2", tok.encode_exact("Emit verdict:", 8)),
                ],
                vec![
                    PromptSegment::new(SegmentKind::SystemPrompt, "heavy_override_prefix_3", tok.encode_exact(&anthropic_md, 256)),
                    PromptSegment::new(SegmentKind::RagChunk, "coupled_block_hash_seg", tok.encode_exact(&block_hash_rs, 96)),
                    PromptSegment::new(SegmentKind::UserQuery, "short_coupled_q3", tok.encode_exact("Next token:", 8)),
                ],
                vec![
                    PromptSegment::new(SegmentKind::SystemPrompt, "heavy_override_prefix_4", tok.encode_exact(&hermes_jinja, 256)),
                    PromptSegment::new(SegmentKind::RagChunk, "coupled_block_hash_seg", tok.encode_exact(&block_hash_rs, 96)),
                    PromptSegment::new(SegmentKind::UserQuery, "short_coupled_q4", tok.encode_exact("Check block:", 8)),
                ],
                vec![
                    PromptSegment::new(SegmentKind::SystemPrompt, "heavy_override_prefix_5", tok.encode_exact(&mistral_jinja, 256)),
                    PromptSegment::new(SegmentKind::RagChunk, "coupled_block_hash_seg", tok.encode_exact(&block_hash_rs, 96)),
                    PromptSegment::new(SegmentKind::UserQuery, "short_coupled_q5", tok.encode_exact("Return key:", 8)),
                ],
            ],
            // Automatic Layer-1 K-drift gate enabled!
            selective_cfg: SelectiveRecomputeConfig {
                recompute_ratio: 0.20,
                lead_boundary_tokens: 4,
                trail_boundary_tokens: 2,
                auto_escalate_drift_threshold: Some(0.08),
                escalated_recompute_ratio: 1.0,
            },
            cache_min_tokens: 32,
        },
        // NEGATIVE REGIME #2: Tiny / unique micro-segments (len < 32 tokens) with ungated cache
        WorkloadSpec {
            name: "neg2_tiny_unique_segments".to_string(),
            regime: "negative_2_tiny_segments".to_string(),
            description: "24 ephemeral 12-token micro-segments (288 tokens total, all < 32 tokens) evaluated with ungated cache (min_segment_tokens=0); SHA-256 hashing + cache lock/insert overhead yields negligible/negative speedup".to_string(),
            source_files: vec!["hanzo-engine/src/paged_attention/block_hash.rs".to_string()],
            warmup_prompts: vec![(0..24)
                .map(|i| {
                    PromptSegment::new(
                        SegmentKind::ConversationTurn,
                        format!("warm_micro_{i}"),
                        tok.encode_exact(&format!("warmup_turn_{i}_id_{}", i * 17 + 3), 12),
                    )
                })
                .collect()],
            eval_prompts: vec![(0..24)
                .map(|i| {
                    PromptSegment::new(
                        SegmentKind::ConversationTurn,
                        format!("eval_micro_{i}"),
                        tok.encode_exact(&format!("unique_ephemeral_{i}_ts_{}", 9000 + i * 31), 12),
                    )
                })
                .collect()],
            selective_cfg: SelectiveRecomputeConfig {
                recompute_ratio: 0.20,
                lead_boundary_tokens: 2,
                trail_boundary_tokens: 1,
                auto_escalate_drift_threshold: None,
                escalated_recompute_ratio: 1.0,
            },
            // Explicitly set to 0 to expose ungated tiny-segment overhead!
            cache_min_tokens: 0,
        },
    ];

    let mut all_workload_results: Vec<(&WorkloadSpec, Vec<ModeBenchResult>)> = Vec::new();
    println!(
        "=== hanzo-cacheslide HIP-1346 Benchmark (seed={}, trials={}) ===",
        args.seed, args.trials
    );
    println!(
        "{:<34} {:<30} {:>8} {:>7} {:>8} {:>8} {:>9} {:>8} {:>7}",
        "Workload", "Mode", "p50(ms)", "Speedup", "Reuse%", "FLOPs%", "KL(nats)", "CosSim", "Top1%"
    );
    println!("{}", "-".repeat(126));

    for spec in &workloads {
        let res = run_workload(&model, &scope, spec, args.trials);
        for m in &res {
            println!(
                "{:<34} {:<30} {:>8.2} {:>6.2}x {:>7.1}% {:>7.1}% {:>9.5} {:>8.5} {:>6.1}%",
                spec.name,
                m.mode_name,
                m.latency.p50_ms,
                m.speedup_vs_full,
                m.tokens_reused_pct,
                m.flops_saved_pct,
                m.quality.kl_divergence,
                m.quality.logit_cosine_similarity,
                m.top1_agreement_pct
            );
        }
        all_workload_results.push((spec, res));
    }

    // Pareto frontier sweeps on workload #0 (tool_schema_shifted_system) and workload #2 (rag_shuffled_chunks).
    let pareto_tool = run_pareto_sweep(
        &model,
        &scope,
        &workloads[0],
        all_workload_results[0].1[0].latency.p50_ms,
        all_workload_results[0].1[0].prefill_flops,
        args.trials,
    );
    let pareto_rag = run_pareto_sweep(
        &model,
        &scope,
        &workloads[2],
        all_workload_results[2].1[0].latency.p50_ms,
        all_workload_results[2].1[0].prefill_flops,
        args.trials,
    );

    let rope_audit = run_rope_audit();
    let tenant_audit = run_tenant_isolation_audit(&model, &tool_bundle_b, &sys_a, &q_eval_1);
    let mem_audit = run_memory_dedup_audit(&model, &scope, &tok, &tool_bundle_b);

    println!("\n=== Verification Audits ===");
    println!(
        "RoPE R(a)R(b)==R(a+b): max_err_neox={:.3e}, max_err_interleaved={:.3e}, pass={}",
        rope_audit.max_group_err_neox,
        rope_audit.max_group_err_interleaved,
        rope_audit.passed_under_1e5
    );
    println!(
        "Tenant/ACL Isolation:  {}/{} leaks ({:.1}%), evict_alpha={}, beta_retained={}",
        tenant_audit.cross_tenant_leaks
            + tenant_audit.cross_acl_leaks
            + tenant_audit.cross_model_leaks
            + tenant_audit.cross_dtype_leaks,
        tenant_audit.total_cross_scope_probes,
        tenant_audit.leak_rate_pct,
        tenant_audit.evicted_alpha_entries,
        tenant_audit.beta_retained_after_alpha_eviction
    );
    println!(
        "GPU Memory Dedup (16 sessions x 256-tok tool): prefix={} B vs cacheslide={} B ({:.2}x reduction; Llama-3-8B proj: {:.1} MB -> {:.1} MB FP16 / {:.1} MB Int4)",
        mem_audit.prefix_cache_gpu_bytes_f16,
        mem_audit.cacheslide_gpu_bytes_f16,
        mem_audit.gpu_memory_reduction_ratio,
        mem_audit.projected_llama8b_prefix_mb_f16,
        mem_audit.projected_llama8b_cacheslide_mb_f16,
        mem_audit.projected_llama8b_cacheslide_mb_int4
    );

    // Serialize JSON receipts cleanly with zero external crate dependencies.
    let mut json = String::with_capacity(32 * 1024);
    json.push_str("{\n");
    json.push_str("  \"benchmark\": \"hanzo-cacheslide-hip-1346\",\n");
    json.push_str("  \"issue\": \"hanzoai/engine#85\",\n");
    json.push_str(&format!("  \"seed\": {},\n", args.seed));
    json.push_str(&format!("  \"trials\": {},\n", args.trials));
    json.push_str("  \"model_config\": {\n");
    json.push_str(&format!("    \"vocab_size\": {},\n", cfg.vocab_size));
    json.push_str(&format!("    \"num_layers\": {},\n", cfg.num_layers));
    json.push_str(&format!("    \"d_model\": {},\n", cfg.d_model));
    json.push_str(&format!("    \"num_heads\": {},\n", cfg.num_heads));
    json.push_str(&format!("    \"num_kv_heads\": {},\n", cfg.num_kv_heads));
    json.push_str(&format!("    \"head_dim\": {},\n", cfg.head_dim));
    json.push_str(&format!("    \"d_ff\": {},\n", cfg.d_ff));
    json.push_str(&format!("    \"rope_theta\": {:.1},\n", cfg.rope_theta));
    json.push_str("    \"rope_layout\": \"gpt_neox\",\n");
    json.push_str(&format!(
        "    \"kv_dtype\": {}\n",
        json_escape(cfg.kv_dtype.as_str())
    ));
    json.push_str("  },\n");

    // RoPE audit
    json.push_str("  \"rope_group_audit\": {\n");
    json.push_str(&format!(
        "    \"max_abs_err_neox\": {:.8e},\n",
        rope_audit.max_group_err_neox
    ));
    json.push_str(&format!(
        "    \"max_abs_err_interleaved\": {:.8e},\n",
        rope_audit.max_group_err_interleaved
    ));
    json.push_str(&format!(
        "    \"passed_under_1e5\": {}\n",
        rope_audit.passed_under_1e5
    ));
    json.push_str("  },\n");

    // Tenant isolation audit
    json.push_str("  \"tenant_isolation_audit\": {\n");
    json.push_str(&format!(
        "    \"total_cross_scope_probes\": {},\n",
        tenant_audit.total_cross_scope_probes
    ));
    json.push_str(&format!(
        "    \"cross_tenant_leaks\": {},\n",
        tenant_audit.cross_tenant_leaks
    ));
    json.push_str(&format!(
        "    \"cross_acl_leaks\": {},\n",
        tenant_audit.cross_acl_leaks
    ));
    json.push_str(&format!(
        "    \"cross_model_leaks\": {},\n",
        tenant_audit.cross_model_leaks
    ));
    json.push_str(&format!(
        "    \"cross_dtype_leaks\": {},\n",
        tenant_audit.cross_dtype_leaks
    ));
    json.push_str(&format!(
        "    \"leak_rate_pct\": {:.4},\n",
        tenant_audit.leak_rate_pct
    ));
    json.push_str(&format!(
        "    \"evicted_alpha_entries\": {},\n",
        tenant_audit.evicted_alpha_entries
    ));
    json.push_str(&format!(
        "    \"beta_retained_after_alpha_eviction\": {}\n",
        tenant_audit.beta_retained_after_alpha_eviction
    ));
    json.push_str("  },\n");

    // GPU Memory deduplication audit
    json.push_str("  \"gpu_memory_dedup\": {\n");
    json.push_str(&format!(
        "    \"num_sessions\": {},\n",
        mem_audit.num_sessions
    ));
    json.push_str(&format!(
        "    \"shared_tool_tokens\": {},\n",
        mem_audit.shared_tool_tokens
    ));
    json.push_str(&format!(
        "    \"prefix_cache_entries\": {},\n",
        mem_audit.prefix_cache_entries
    ));
    json.push_str(&format!(
        "    \"cacheslide_entries\": {},\n",
        mem_audit.cacheslide_entries
    ));
    json.push_str(&format!(
        "    \"prefix_cache_gpu_bytes_f16\": {},\n",
        mem_audit.prefix_cache_gpu_bytes_f16
    ));
    json.push_str(&format!(
        "    \"cacheslide_gpu_bytes_f16\": {},\n",
        mem_audit.cacheslide_gpu_bytes_f16
    ));
    json.push_str(&format!(
        "    \"gpu_memory_reduction_ratio\": {:.4},\n",
        mem_audit.gpu_memory_reduction_ratio
    ));
    json.push_str(&format!(
        "    \"projected_llama8b_prefix_mb_f16\": {:.3},\n",
        mem_audit.projected_llama8b_prefix_mb_f16
    ));
    json.push_str(&format!(
        "    \"projected_llama8b_cacheslide_mb_f16\": {:.3},\n",
        mem_audit.projected_llama8b_cacheslide_mb_f16
    ));
    json.push_str(&format!(
        "    \"projected_llama8b_cacheslide_mb_int4\": {:.3},\n",
        mem_audit.projected_llama8b_cacheslide_mb_int4
    ));
    json.push_str(&format!(
        "    \"budget_respected\": {}\n",
        mem_audit.budget_respected
    ));
    json.push_str("  },\n");

    // Workloads
    json.push_str("  \"workloads\": [\n");
    for (wi, (spec, modes)) in all_workload_results.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!("      \"name\": {},\n", json_escape(&spec.name)));
        json.push_str(&format!(
            "      \"regime\": {},\n",
            json_escape(&spec.regime)
        ));
        json.push_str(&format!(
            "      \"description\": {},\n",
            json_escape(&spec.description)
        ));
        json.push_str("      \"source_files\": [");
        for (fi, f) in spec.source_files.iter().enumerate() {
            if fi > 0 {
                json.push_str(", ");
            }
            json.push_str(&json_escape(f));
        }
        json.push_str("],\n");
        json.push_str(&format!(
            "      \"cache_min_tokens\": {},\n",
            spec.cache_min_tokens
        ));
        json.push_str("      \"modes\": [\n");
        for (mi, m) in modes.iter().enumerate() {
            json.push_str("        {\n");
            json.push_str(&format!(
                "          \"mode\": {},\n",
                json_escape(&m.mode_name)
            ));
            json.push_str(&format!(
                "          \"ttft_p50_ms\": {:.4},\n",
                m.latency.p50_ms
            ));
            json.push_str(&format!(
                "          \"ttft_p95_ms\": {:.4},\n",
                m.latency.p95_ms
            ));
            json.push_str(&format!(
                "          \"ttft_p99_ms\": {:.4},\n",
                m.latency.p99_ms
            ));
            json.push_str(&format!(
                "          \"ttft_mean_ms\": {:.4},\n",
                m.latency.mean_ms
            ));
            json.push_str(&format!(
                "          \"speedup_vs_full\": {:.4},\n",
                m.speedup_vs_full
            ));
            json.push_str(&format!(
                "          \"total_tokens\": {},\n",
                m.total_tokens
            ));
            json.push_str(&format!(
                "          \"tokens_reused\": {},\n",
                m.tokens_reused
            ));
            json.push_str(&format!(
                "          \"tokens_reused_pct\": {:.2},\n",
                m.tokens_reused_pct
            ));
            json.push_str(&format!(
                "          \"tokens_computed\": {},\n",
                m.tokens_computed
            ));
            json.push_str(&format!(
                "          \"tokens_selectively_recomputed\": {},\n",
                m.tokens_selectively_recomputed
            ));
            json.push_str(&format!(
                "          \"segments_shifted\": {},\n",
                m.segments_shifted
            ));
            json.push_str(&format!(
                "          \"segments_exact_hit\": {},\n",
                m.segments_exact_hit
            ));
            json.push_str(&format!(
                "          \"prefill_flops\": {},\n",
                m.prefill_flops
            ));
            json.push_str(&format!(
                "          \"flops_saved_pct\": {:.2},\n",
                m.flops_saved_pct
            ));
            json.push_str(&format!(
                "          \"max_layer1_k_drift\": {:.6},\n",
                m.max_layer1_k_drift
            ));
            json.push_str(&format!(
                "          \"drift_gate_escalations\": {},\n",
                m.drift_gate_escalations
            ));
            json.push_str(&format!(
                "          \"kl_divergence_nats\": {:.6},\n",
                m.quality.kl_divergence
            ));
            json.push_str(&format!(
                "          \"js_divergence_nats\": {:.6},\n",
                m.quality.js_divergence
            ));
            json.push_str(&format!(
                "          \"logit_cosine_similarity\": {:.6},\n",
                m.quality.logit_cosine_similarity
            ));
            json.push_str(&format!(
                "          \"max_abs_logit_diff\": {:.6},\n",
                m.quality.max_abs_logit_diff
            ));
            json.push_str(&format!(
                "          \"top1_agreement_pct\": {:.2},\n",
                m.top1_agreement_pct
            ));
            json.push_str(&format!(
                "          \"top5_overlap\": {:.4},\n",
                m.quality.top5_overlap
            ));
            json.push_str(&format!(
                "          \"ref_entropy_nats\": {:.4}\n",
                m.quality.ref_entropy_nats
            ));
            json.push_str(if mi + 1 < modes.len() {
                "        },\n"
            } else {
                "        }\n"
            });
        }
        json.push_str("      ]\n");
        json.push_str(if wi + 1 < all_workload_results.len() {
            "    },\n"
        } else {
            "    }\n"
        });
    }
    json.push_str("  ],\n");

    // Pareto sweeps
    let write_pareto = |out: &mut String, name: &str, pts: &[ParetoPoint], trailing_comma: bool| {
        out.push_str(&format!("    {}: [\n", json_escape(name)));
        for (pi, pt) in pts.iter().enumerate() {
            out.push_str("      {\n");
            out.push_str(&format!(
                "        \"recompute_ratio\": {:.2},\n",
                pt.recompute_ratio
            ));
            out.push_str(&format!(
                "        \"ttft_p50_ms\": {:.4},\n",
                pt.ttft_p50_ms
            ));
            out.push_str(&format!(
                "        \"speedup_vs_full\": {:.4},\n",
                pt.speedup_vs_full
            ));
            out.push_str(&format!(
                "        \"flops_saved_pct\": {:.2},\n",
                pt.flops_saved_pct
            ));
            out.push_str(&format!(
                "        \"tokens_reused_pct\": {:.2},\n",
                pt.tokens_reused_pct
            ));
            out.push_str(&format!(
                "        \"kl_divergence_nats\": {:.6},\n",
                pt.kl_divergence
            ));
            out.push_str(&format!(
                "        \"logit_cosine_similarity\": {:.6},\n",
                pt.logit_cosine_similarity
            ));
            out.push_str(&format!(
                "        \"top1_agreement_pct\": {:.2},\n",
                pt.top1_agreement_pct
            ));
            out.push_str(&format!(
                "        \"top5_overlap\": {:.4}\n",
                pt.top5_overlap
            ));
            out.push_str(if pi + 1 < pts.len() {
                "      },\n"
            } else {
                "      }\n"
            });
        }
        out.push_str(if trailing_comma {
            "    ],\n"
        } else {
            "    ]\n"
        });
    };

    json.push_str("  \"pareto_frontier\": {\n");
    write_pareto(&mut json, "tool_schema_shifted_system", &pareto_tool, true);
    write_pareto(&mut json, "rag_shuffled_chunks", &pareto_rag, false);
    json.push_str("  }\n");
    json.push_str("}\n");

    let out_full = if args.out_path.is_absolute() {
        args.out_path
    } else {
        root.join(&args.out_path)
    };
    if let Some(parent) = out_full.parent() {
        fs::create_dir_all(parent).expect("failed to create output directory");
    }
    fs::write(&out_full, &json).expect("failed to write results.json");
    println!("\nWrote JSON receipt -> {}", out_full.display());
}
