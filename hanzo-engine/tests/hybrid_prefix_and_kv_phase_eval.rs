#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_engine::{
    active_kv_phase_policy, evaluate_multiturn_hybrid_prefix_reuse,
    evaluate_phase_aware_kv_precision, set_kv_phase_policy_override,
    verify_hybrid_prefix_provenance_isolation, HybridEvalConfig, KvPhasePolicy,
};
use hanzo_ml::{DType, Device, Result};
#[cfg(feature = "cuda")]
use serde::Serialize;

#[cfg(feature = "cuda")]
const ENV_ENGINE_EVAL_OUT_JSON: &str = "ENGINE_EVAL_OUT_JSON";

#[cfg(feature = "cuda")]
#[derive(Serialize)]
struct EngineTrack3EvalBundle {
    provenance_isolation: hanzo_engine::ProvenanceIsolationReport,
    hybrid_multiturn_f32: hanzo_engine::HybridMultiturnEvalReport,
    hybrid_multiturn_bf16: hanzo_engine::HybridMultiturnEvalReport,
    kv_phase_eval: hanzo_engine::KvPhaseEvalReport,
}

#[test]
fn hybrid_prefix_and_kv_phase_cpu_gate() -> Result<()> {
    set_kv_phase_policy_override(Some(KvPhasePolicy::PhaseAware {
        min_fp8_decode_ctx: 2048,
    }));
    assert_eq!(
        active_kv_phase_policy(),
        KvPhasePolicy::PhaseAware {
            min_fp8_decode_ctx: 2048
        }
    );

    let dev = Device::Cpu;
    let iso = verify_hybrid_prefix_provenance_isolation(&dev)?;
    assert_eq!(iso.owner_hit_tokens, 80);
    assert_eq!(iso.cross_tenant_hits, 0);
    assert_eq!(iso.unsalted_hits_against_salted, 0);
    assert_eq!(iso.cross_model_hits, 0);
    assert_eq!(iso.cross_adapter_hits, 0);
    assert_eq!(iso.base_hits_against_adapter, 0);
    assert_eq!(iso.boundary_collision_hits, 0);
    assert_eq!(iso.disabled_env_hits, 0);
    assert_eq!(iso.post_mutation_checkpoint_rel_err, 0.0);

    let cfg = HybridEvalConfig {
        num_gdn_layers: 3,
        num_attn_layers: 1,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        hidden_size: 256,
        system_tokens: 16,
        repo_tokens: 32,
        tool_schema_tokens: 16,
        user_turn_tokens: 16,
        assistant_turn_tokens: 8,
        num_turns: 5,
        repetitions: 3,
    };
    let rep = evaluate_multiturn_hybrid_prefix_reuse(&dev, DType::F32, &cfg)?;
    assert_eq!(rep.warmup_runs, 1);
    assert_eq!(rep.repetitions, 3);
    assert!(rep.is_synthetic_microbench);
    assert_eq!(rep.cumulative_cold_ms_samples.len(), 3);
    assert_eq!(rep.cumulative_warm_ms_samples.len(), 3);
    assert!(
        rep.cumulative_ttft_speedup_stats.ci95_low <= rep.cumulative_ttft_speedup_stats.mean
            && rep.cumulative_ttft_speedup_stats.mean
                <= rep.cumulative_ttft_speedup_stats.ci95_high
    );
    for t in rep
        .shared_base_turns
        .iter()
        .chain(&rep.cumulative_coding_turns)
    {
        assert_eq!(t.cold_ms_samples.len(), 3);
        assert_eq!(t.warm_ms_samples.len(), 3);
        assert!(t.attn_out_max_rel < 1e-5);
        assert!(t.gdn_out_max_rel < 1e-5);
        assert!(t.gdn_state_max_rel < 1e-5);
        assert_eq!(t.conv_state_max_rel, 0.0);
    }

    // Exercise both BF16-retained short contexts (< 2048) and active FP8 E4M3 decode (ctx = 2048).
    let kv_rep = evaluate_phase_aware_kv_precision(&dev, &[64, 128], &[256, 512, 2048])?;
    assert_eq!(kv_rep.points.len(), 6);
    let mut saw_fp8_active = false;
    for pt in &kv_rep.points {
        assert!(!pt.decode_speedup_is_native);
        assert!((pt.analytic_kv_bytes_ratio - 2.0).abs() < 1e-9);
        assert!(pt.measured_fp8_kv_plus_scale_bytes > 0);
        assert!(pt.measured_bf16_kv_alloc_bytes > pt.measured_fp8_kv_plus_scale_bytes);
        assert!(pt.bf16_prefill_us > 0.0);
        assert!(pt.phase_aware_prefill_us > 0.0);
        assert!(pt.fp8_always_prefill_us > 0.0);
        assert!(pt.phase_aware_prefill_max_rel <= pt.fp8_always_prefill_max_rel);
        assert!(
            pt.fp8_decode_cosine_vs_f32 >= 0.999,
            "d={} ctx={} fp8 decode cosine {} < 0.999",
            pt.head_dim,
            pt.context_len,
            pt.fp8_decode_cosine_vs_f32
        );
        if pt.context_len >= 2048 {
            assert!(pt.phase_aware_used_fp8);
            saw_fp8_active = true;
        } else {
            assert!(!pt.phase_aware_used_fp8);
        }
    }
    assert!(saw_fp8_active);
    set_kv_phase_policy_override(None);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_hybrid_prefix_and_phase_aware_kv_eval() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    set_kv_phase_policy_override(Some(KvPhasePolicy::PhaseAware {
        min_fp8_decode_ctx: 2048,
    }));

    let iso = verify_hybrid_prefix_provenance_isolation(&dev)?;
    assert_eq!(iso.owner_hit_tokens, 80);
    assert_eq!(iso.cross_tenant_hits, 0);
    assert_eq!(iso.unsalted_hits_against_salted, 0);
    assert_eq!(iso.cross_model_hits, 0);
    assert_eq!(iso.cross_adapter_hits, 0);
    assert_eq!(iso.base_hits_against_adapter, 0);
    assert_eq!(iso.boundary_collision_hits, 0);
    assert_eq!(iso.disabled_env_hits, 0);
    assert_eq!(iso.post_mutation_checkpoint_rel_err, 0.0);

    let cfg = HybridEvalConfig::default();
    assert_eq!(cfg.repetitions, 10);
    let rep_f32 = evaluate_multiturn_hybrid_prefix_reuse(&dev, DType::F32, &cfg)?;
    for t in rep_f32
        .shared_base_turns
        .iter()
        .chain(&rep_f32.cumulative_coding_turns)
    {
        assert!(
            t.attn_out_max_rel < 1e-5,
            "CUDA F32 turn {} ({}) attn rel_err {} >= 1e-5",
            t.turn,
            t.mode,
            t.attn_out_max_rel
        );
        assert!(
            t.gdn_out_max_rel < 1e-5,
            "CUDA F32 turn {} ({}) gdn rel_err {} >= 1e-5",
            t.turn,
            t.mode,
            t.gdn_out_max_rel
        );
        assert!(
            t.gdn_state_max_rel < 1e-5,
            "CUDA F32 turn {} ({}) gdn_state rel_err {} >= 1e-5",
            t.turn,
            t.mode,
            t.gdn_state_max_rel
        );
        assert_eq!(t.conv_state_max_rel, 0.0);
    }

    let rep_bf16 = evaluate_multiturn_hybrid_prefix_reuse(&dev, DType::BF16, &cfg)?;
    for t in rep_bf16
        .shared_base_turns
        .iter()
        .chain(&rep_bf16.cumulative_coding_turns)
    {
        assert!(t.attn_out_cosine > 0.9999);
        assert!(t.gdn_state_max_rel < 1e-5);
        assert_eq!(t.conv_state_max_rel, 0.0);
    }

    let kv_rep =
        evaluate_phase_aware_kv_precision(&dev, &[64, 128, 256], &[4096, 8192, 16384, 32768])?;
    for pt in &kv_rep.points {
        assert!(pt.phase_aware_used_fp8);
        assert!(pt.decode_speedup_is_native);
        assert!((pt.analytic_kv_bytes_ratio - 2.0).abs() < 1e-9);
        assert!(pt.measured_fp8_kv_plus_scale_bytes > 0);
        assert!(
            pt.fp8_decode_cosine_vs_f32 >= 0.999,
            "d={} ctx={} fp8 cosine {}",
            pt.head_dim,
            pt.context_len,
            pt.fp8_decode_cosine_vs_f32
        );
        assert!(
            pt.phase_aware_prefill_max_rel <= pt.fp8_always_prefill_max_rel,
            "phase-aware prefill must be at least as accurate as fp8-always"
        );
    }

    eprintln!(
        "\n[CUDA Provenance Isolation] owner_hit={} cross_tenant={} cross_model={} cross_adapter={} boundary={} mut_rel={:.2e}",
        iso.owner_hit_tokens,
        iso.cross_tenant_hits,
        iso.cross_model_hits,
        iso.cross_adapter_hits,
        iso.boundary_collision_hits,
        iso.post_mutation_checkpoint_rel_err
    );
    eprintln!(
        "\n[CUDA Multi-Turn Synthetic Hybrid Prefix Reuse BF16 ({} reps after {} warmup)] median_cold={:.2}ms median_warm={:.2}ms median_speedup={:.2}x mean={:.2}x sd={:.2}x ci95=[{:.2}x, {:.2}x]",
        rep_bf16.repetitions,
        rep_bf16.warmup_runs,
        rep_bf16.cumulative_cold_ms,
        rep_bf16.cumulative_warm_ms,
        rep_bf16.cumulative_ttft_speedup,
        rep_bf16.cumulative_ttft_speedup_stats.mean,
        rep_bf16.cumulative_ttft_speedup_stats.std_dev,
        rep_bf16.cumulative_ttft_speedup_stats.ci95_low,
        rep_bf16.cumulative_ttft_speedup_stats.ci95_high
    );
    for t in &rep_bf16.cumulative_coding_turns {
        eprintln!(
            "  turn={} prompt={} reused={} suffix={} cold_med={:.2}ms warm_med={:.2}ms speedup_med={:.2}x ci95=[{:.2}x,{:.2}x] f32_attn_rel={:.2e} bf16_attn_rel={:.2e} gdn_st_rel={:.2e} conv_rel={:.2e}",
            t.turn,
            t.total_prompt_tokens,
            t.reused_prefix_tokens,
            t.prefilled_suffix_tokens,
            t.cold_reprefill_ms,
            t.warm_prefix_reuse_ms,
            t.ttft_speedup,
            t.ttft_speedup_stats.ci95_low,
            t.ttft_speedup_stats.ci95_high,
            rep_f32.cumulative_coding_turns[t.turn - 1].attn_out_max_rel,
            t.attn_out_max_rel,
            t.gdn_state_max_rel,
            t.conv_state_max_rel
        );
    }

    eprintln!("\n[CUDA Phase-Aware KV Precision (BF16 vs FP8 E4M3)]");
    for pt in &kv_rep.points {
        eprintln!(
            "  d={:3} ctx={:5} | prefill_us(bf16={:.1}, pa={:.1}, fp8={:.1}) pre_rel(pa={:.2e}, fp8={:.2e}) | decode_us(bf16={:.1}, fp8={:.1}, {:.2}x native={}) kv_ratio(analytic={:.2}x, alloc={:.4}x, fp8+scale={}B) dec_rel(bf16={:.2e}, fp8={:.2e}) cos={:.6}",
            pt.head_dim,
            pt.context_len,
            pt.bf16_prefill_us,
            pt.phase_aware_prefill_us,
            pt.fp8_always_prefill_us,
            pt.phase_aware_prefill_max_rel,
            pt.fp8_always_prefill_max_rel,
            pt.bf16_decode_us,
            pt.fp8_decode_us,
            pt.decode_speedup,
            pt.decode_speedup_is_native,
            pt.analytic_kv_bytes_ratio,
            pt.measured_kv_alloc_bytes_ratio,
            pt.measured_fp8_kv_plus_scale_bytes,
            pt.bf16_decode_max_rel_vs_f32,
            pt.fp8_decode_max_rel_vs_f32,
            pt.fp8_decode_cosine_vs_f32
        );
    }

    if let Ok(out_path) = std::env::var(ENV_ENGINE_EVAL_OUT_JSON) {
        let bundle = EngineTrack3EvalBundle {
            provenance_isolation: iso,
            hybrid_multiturn_f32: rep_f32,
            hybrid_multiturn_bf16: rep_bf16,
            kv_phase_eval: kv_rep,
        };
        let json = serde_json::to_string_pretty(&bundle)
            .map_err(|e| hanzo_ml::Error::Msg(format!("json encode failed: {e}")))?;
        std::fs::write(&out_path, json)
            .map_err(|e| hanzo_ml::Error::Msg(format!("write {out_path} failed: {e}")))?;
    }

    set_kv_phase_policy_override(None);
    Ok(())
}
