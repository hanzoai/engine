//! Reproducible HIP-1346 benchmark orchestrator and zero-dependency JSON receipt
//! generator for `hanzoai/engine#57`.

use std::fmt::Write as FmtWrite;
use std::fs;
use std::io;
use std::path::Path;

use crate::corpus::GroundedCorpus;
use crate::hardware::HardwareProfile;
use crate::mmap_store::{run_real_host_io_benchmark, RealHostIoReport};
use crate::model::{HybridModelSpec, GIB};
use crate::partition::{
    ActivationProfile, EvaluationScorecard, NegativeRegimeResult, PlacementPlanner, PlacementRegime,
};
use crate::sha256::Sha256;
use crate::verify::{run_execution_equivalence_verifier, ExecutionVerificationReport};

/// Complete HIP-1346 `#57` benchmark bundle combining:
/// 1. Grounded 3-file corpus SHA-256 attestation
/// 2. Directly measured Linux host `WeightFile` I/O & async prefetch benchmark
/// 3. Directly measured pure-Rust bit-exact CPU/GPU hybrid execution verifier
/// 4. Complete `3 models × 4 topologies × 5 placement regimes` analytical + exact-memory matrix
/// 5. Three explicit honest negative regimes
#[derive(Clone, Debug)]
pub struct BenchmarkBundle {
    /// Deterministic PRNG seed (`1346` by default).
    pub seed: u64,
    /// Grounded repository corpus attestation.
    pub corpus: GroundedCorpus,
    /// Grounded corpus activation skew summary (`9B-Hybrid` 224 neuron blocks).
    pub corpus_skew: ActivationProfile,
    /// Directly measured on-disk `WeightFile` I/O benchmark on this host.
    pub host_io: RealHostIoReport,
    /// Directly measured pure-Rust CPU bit-exact execution verifier.
    pub execution_verify: ExecutionVerificationReport,
    /// Complete `3 × 4 × 5 = 60` analytical + exact-memory matrix scorecards.
    pub matrix_scorecards: Vec<EvaluationScorecard>,
    /// Three explicit honest negative regimes with baseline comparisons.
    pub negative_regimes: Vec<NegativeRegimeResult>,
    /// Deterministic SHA-256 receipt over all deterministic corpus, analytical, exact-memory,
    /// and correctness fields (excluding non-deterministic host wall-clock timings).
    pub deterministic_receipt_sha256: String,
}

/// Runs the complete HIP-1346 `#57` benchmark suite deterministically from `seed`.
pub fn run_full_benchmark(seed: u64) -> io::Result<BenchmarkBundle> {
    let corpus = GroundedCorpus::load_from_workspace()?;
    let corpus_skew = ActivationProfile::from_grounded_corpus(&corpus, 224);

    // Directly measured host on-disk WeightFile benchmark: 256 tiles × 112,640 f32 elems (~16 MiB INT4 on disk).
    let host_io = run_real_host_io_benchmark(seed, 256, 112_640)?;

    // Directly measured pure-Rust CPU bit-exact hybrid execution verifier (64 grounded tokens).
    let execution_verify = run_execution_equivalence_verifier(&corpus, seed, 64);
    let verified_ok_pct = if execution_verify.all_passed() {
        100.0
    } else {
        0.0
    };
    let verified_reg_pp = execution_verify
        .regimes
        .iter()
        .map(|r| r.quality_regression_pp)
        .fold(0.0f64, f64::max);

    let models = HybridModelSpec::hip_1346_matrix();
    let topologies = HardwareProfile::all_profiles();
    let regimes = PlacementRegime::all_five();

    let mut matrix_scorecards = Vec::with_capacity(models.len() * topologies.len() * regimes.len());
    for model in &models {
        let skew = ActivationProfile::from_grounded_corpus(&corpus, model.neuron_blocks_per_layer);
        for hw in &topologies {
            for &regime in &regimes {
                let sc = PlacementPlanner::evaluate_regime(
                    model,
                    hw,
                    regime,
                    &skew,
                    verified_ok_pct,
                    verified_reg_pp,
                );
                matrix_scorecards.push(sc);
            }
        }
    }

    let negative_regimes = PlacementPlanner::evaluate_negative_regimes(
        &corpus,
        seed,
        verified_ok_pct,
        verified_reg_pp,
    );

    let canonical_payload = build_deterministic_hash_payload(
        seed,
        &corpus,
        &corpus_skew,
        &host_io,
        &execution_verify,
        &matrix_scorecards,
        &negative_regimes,
    );
    let deterministic_receipt_sha256 = Sha256::hex_digest(canonical_payload.as_bytes());

    Ok(BenchmarkBundle {
        seed,
        corpus,
        corpus_skew,
        host_io,
        execution_verify,
        matrix_scorecards,
        negative_regimes,
        deterministic_receipt_sha256,
    })
}

fn build_deterministic_hash_payload(
    seed: u64,
    corpus: &GroundedCorpus,
    skew: &ActivationProfile,
    host_io: &RealHostIoReport,
    verify: &ExecutionVerificationReport,
    matrix: &[EvaluationScorecard],
    negatives: &[NegativeRegimeResult],
) -> String {
    let mut s = String::with_capacity(16_384);
    let _ = writeln!(
        s,
        "seed={seed}|corpus={}|gini={:.6}|top20={:.6}|wf_sha={}|wf_bytes={}",
        corpus.combined_sha256_hex,
        skew.gini,
        skew.top_20_pct_coverage,
        host_io.payload_sha256_hex,
        host_io.file_bytes,
    );
    for r in &verify.regimes {
        let _ = writeln!(
            s,
            "verify|{}|{}/{}|{:.6e}|{}",
            r.regime, r.matched_tokens, r.total_tokens, r.max_abs_err, r.weight_copy_bytes
        );
    }
    for sc in matrix {
        let _ = writeln!(
            s,
            "cell|{}|{}|{}|{}|{:.3}|{:.3}|{:.3}|{:.3}|{:.3}|{}|{}|{}|{}|{}",
            sc.model_label,
            sc.hardware_id,
            sc.plan.regime_name,
            sc.plan.resolved_strategy,
            sc.cold_ttft_ms,
            sc.warm_ttft_ms,
            sc.itl_p50_ms,
            sc.itl_p95_ms,
            sc.agent_20_turn_wall_s,
            sc.plan.peak_vram_bytes,
            sc.plan.peak_host_ram_bytes,
            sc.plan.peak_total_mem_bytes,
            sc.gate_accepted,
            sc.gate_reason,
        );
    }
    for neg in negatives {
        let _ = writeln!(
            s,
            "neg|{}|{:.3}|{:.3}|{}|{}",
            neg.kind.as_str(),
            neg.wall_time_delta_pct,
            neg.energy_delta_pct,
            neg.negative.gate_accepted,
            neg.negative.gate_reason,
        );
    }
    s
}

/// Serialises `bundle` to canonical pretty-printed JSON (zero external crate dependencies).
pub fn to_json(bundle: &BenchmarkBundle) -> String {
    let mut j = String::with_capacity(65_536);
    j.push_str("{\n");
    let _ = writeln!(j, "  \"issue\": \"hanzoai/engine#57\",");
    let _ = writeln!(j, "  \"hip\": \"HIP-1346\",");
    let _ = writeln!(j, "  \"crate\": \"hanzo-hetero-mem\",");
    let _ = writeln!(j, "  \"license\": \"Apache-2.0\",");
    let _ = writeln!(j, "  \"seed\": {},", bundle.seed);
    let _ = writeln!(
        j,
        "  \"deterministic_receipt_sha256\": \"{}\",",
        bundle.deterministic_receipt_sha256
    );
    let _ = writeln!(j, "  \"measurement_taxonomy\": {{");
    let _ = writeln!(
        j,
        "    \"host_io_and_verifier\": \"cpu_directly_measured\","
    );
    let _ = writeln!(j, "    \"memory_footprints\": \"exact_unclamped_bytes\",");
    let _ = writeln!(
        j,
        "    \"gpu_and_unified_latency_energy\": \"analytical_hardware_projection\""
    );
    let _ = writeln!(j, "  }},");

    // Corpus section
    let _ = writeln!(j, "  \"grounded_corpus\": {{");
    let _ = writeln!(j, "    \"total_bytes\": {},", bundle.corpus.total_bytes);
    let _ = writeln!(
        j,
        "    \"combined_sha256\": \"{}\",",
        bundle.corpus.combined_sha256_hex
    );
    let _ = writeln!(
        j,
        "    \"activation_skew_gini\": {:.6},",
        bundle.corpus_skew.gini
    );
    let _ = writeln!(
        j,
        "    \"activation_skew_top20_pct_coverage\": {:.6},",
        bundle.corpus_skew.top_20_pct_coverage
    );
    let _ = writeln!(j, "    \"files\": [");
    for (i, f) in bundle.corpus.files.iter().enumerate() {
        let comma = if i + 1 < bundle.corpus.files.len() {
            ","
        } else {
            ""
        };
        let _ = writeln!(
            j,
            "      {{\"path\": \"{}\", \"bytes\": {}, \"sha256\": \"{}\"}}{}",
            escape_json(&f.relative_path),
            f.byte_len,
            f.sha256_hex,
            comma
        );
    }
    let _ = writeln!(j, "    ]");
    let _ = writeln!(j, "  }},");

    // Directly measured host I/O section
    let _ = writeln!(j, "  \"host_io_directly_measured\": {{");
    let _ = writeln!(j, "    \"measurement_tag\": \"cpu_directly_measured\",");
    let _ = writeln!(j, "    \"file_bytes\": {},", bundle.host_io.file_bytes);
    let _ = writeln!(j, "    \"num_tiles\": {},", bundle.host_io.num_tiles);
    let _ = writeln!(
        j,
        "    \"pages_per_tile\": {},",
        bundle.host_io.pages_per_tile
    );
    let _ = writeln!(j, "    \"format\": \"{}\",", bundle.host_io.format.as_str());
    let _ = writeln!(
        j,
        "    \"payload_sha256\": \"{}\",",
        bundle.host_io.payload_sha256_hex
    );
    let _ = writeln!(
        j,
        "    \"sha256_verified\": {},",
        bundle.host_io.sha256_verified
    );
    let _ = writeln!(j, "    \"modes\": [");
    for (i, m) in bundle.host_io.modes.iter().enumerate() {
        let comma = if i + 1 < bundle.host_io.modes.len() {
            ","
        } else {
            ""
        };
        let _ = writeln!(
            j,
            "      {{\"mode\": \"{}\", \"cold_first_tile_ms\": {:.4}, \"warm_first_tile_ms\": {:.4}, \"full_stream_ms\": {:.4}, \"sustained_decode_mibs\": {:.2}, \"cold_start_major_faults\": {}, \"total_major_faults\": {}, \"total_minor_faults\": {}, \"prefetch_hit_rate\": {:.4}, \"copy_bytes\": {}, \"peak_resident_pages\": {}}}{}",
            m.mode_label,
            m.cold_first_tile_ms,
            m.warm_first_tile_ms,
            m.full_stream_ms,
            m.sustained_decode_mibs,
            m.cold_start_major_faults,
            m.total_major_faults,
            m.total_minor_faults,
            m.prefetch_hit_rate,
            m.copy_bytes,
            m.peak_resident_pages,
            comma
        );
    }
    let _ = writeln!(j, "    ]");
    let _ = writeln!(j, "  }},");

    // Directly measured CPU execution equivalence verifier
    let _ = writeln!(j, "  \"execution_verifier_directly_measured\": {{");
    let _ = writeln!(j, "    \"measurement_tag\": \"cpu_directly_measured\",");
    let _ = writeln!(
        j,
        "    \"all_passed\": {},",
        bundle.execution_verify.all_passed()
    );
    let _ = writeln!(
        j,
        "    \"num_layers\": {},",
        bundle.execution_verify.num_layers
    );
    let _ = writeln!(
        j,
        "    \"hidden_dim\": {},",
        bundle.execution_verify.hidden_dim
    );
    let _ = writeln!(
        j,
        "    \"intermediate_dim\": {},",
        bundle.execution_verify.intermediate_dim
    );
    let _ = writeln!(
        j,
        "    \"gpu_hot_neuron_fraction\": {:.4},",
        bundle.execution_verify.gpu_hot_neuron_fraction
    );
    let _ = writeln!(
        j,
        "    \"cpu_wall_time_ms\": {:.3},",
        bundle.execution_verify.cpu_wall_time_ms
    );
    let _ = writeln!(j, "    \"regimes\": [");
    for (i, r) in bundle.execution_verify.regimes.iter().enumerate() {
        let comma = if i + 1 < bundle.execution_verify.regimes.len() {
            ","
        } else {
            ""
        };
        let _ = writeln!(
            j,
            "      {{\"regime\": \"{}\", \"max_abs_err\": {:.6e}, \"matched_tokens\": {}, \"total_tokens\": {}, \"quality_regression_pp\": {:.4}, \"weight_copy_bytes\": {}, \"zero_copy_ptr_verified\": {}}}{}",
            r.regime,
            r.max_abs_err,
            r.matched_tokens,
            r.total_tokens,
            r.quality_regression_pp,
            r.weight_copy_bytes,
            r.zero_copy_ptr_verified,
            comma
        );
    }
    let _ = writeln!(j, "    ]");
    let _ = writeln!(j, "  }},");

    // Shared model matrix
    let _ = writeln!(j, "  \"hip_1346_model_matrix\": [");
    let models = HybridModelSpec::hip_1346_matrix();
    for (i, m) in models.iter().enumerate() {
        let comma = if i + 1 < models.len() { "," } else { "" };
        let _ = writeln!(
            j,
            "    {{\"label\": \"{}\", \"attention_layers\": {}, \"gdn_layers\": {}, \"num_kv_heads\": {}, \"head_dim\": {}, \"gdn_state_mib\": {:.1}, \"kv_bytes_per_token\": {}, \"bf16_weight_gib\": {:.3}, \"int4_weight_gib\": {:.3}}}{}",
            m.label,
            m.num_attention_layers,
            m.num_gdn_layers,
            m.num_kv_heads,
            m.head_dim,
            m.gdn_state_mib(),
            m.kv_bytes_per_token(),
            m.weight_bytes(crate::model::WeightFormat::Bf16) as f64 / GIB as f64,
            m.weight_bytes(crate::model::WeightFormat::Int4Packed) as f64 / GIB as f64,
            comma
        );
    }
    let _ = writeln!(j, "  ],");

    // Analytical matrix scorecards
    let _ = writeln!(j, "  \"matrix_scorecards\": [");
    for (i, sc) in bundle.matrix_scorecards.iter().enumerate() {
        let comma = if i + 1 < bundle.matrix_scorecards.len() {
            ","
        } else {
            ""
        };
        write_scorecard_json(&mut j, sc, "    ", comma);
    }
    let _ = writeln!(j, "  ],");

    // Negative regimes
    let _ = writeln!(j, "  \"negative_regimes\": [");
    for (i, neg) in bundle.negative_regimes.iter().enumerate() {
        let comma = if i + 1 < bundle.negative_regimes.len() {
            ","
        } else {
            ""
        };
        let _ = writeln!(j, "    {{");
        let _ = writeln!(j, "      \"kind\": \"{}\",", neg.kind.as_str());
        let _ = writeln!(
            j,
            "      \"wall_time_delta_pct\": {:.2},",
            neg.wall_time_delta_pct
        );
        let _ = writeln!(
            j,
            "      \"energy_delta_pct\": {:.2},",
            neg.energy_delta_pct
        );
        let _ = writeln!(
            j,
            "      \"peak_mem_delta_pct\": {:.2},",
            neg.peak_mem_delta_pct
        );
        let _ = writeln!(
            j,
            "      \"microbench_note\": \"{}\",",
            escape_json(&neg.microbench_note)
        );
        let _ = writeln!(j, "      \"baseline\": ");
        write_scorecard_json(&mut j, &neg.baseline, "      ", ",");
        let _ = writeln!(j, "      \"negative\": ");
        write_scorecard_json(&mut j, &neg.negative, "      ", "");
        let _ = writeln!(j, "    }}{}", comma);
    }
    let _ = writeln!(j, "  ]");
    j.push_str("}\n");
    j
}

fn write_scorecard_json(out: &mut String, sc: &EvaluationScorecard, indent: &str, comma: &str) {
    let _ = writeln!(out, "{indent}{{");
    let _ = writeln!(
        out,
        "{indent}  \"measurement_tag\": \"analytical_hardware_projection\","
    );
    let _ = writeln!(out, "{indent}  \"memory_tag\": \"exact_unclamped_bytes\",");
    let _ = writeln!(out, "{indent}  \"model\": \"{}\",", sc.model_label);
    let _ = writeln!(out, "{indent}  \"hardware\": \"{}\",", sc.hardware_id);
    let _ = writeln!(out, "{indent}  \"memory_arch\": \"{}\",", sc.memory_arch);
    let _ = writeln!(
        out,
        "{indent}  \"regime\": \"{}\",",
        escape_json(&sc.plan.regime_name)
    );
    let _ = writeln!(
        out,
        "{indent}  \"resolved_strategy\": \"{}\",",
        escape_json(&sc.plan.resolved_strategy)
    );
    let _ = writeln!(
        out,
        "{indent}  \"weight_format\": \"{}\",",
        sc.plan.weight_format.as_str()
    );
    let _ = writeln!(
        out,
        "{indent}  \"bf16_fit_regime\": \"{}\",",
        sc.plan.bf16_fit_regime.as_str()
    );
    let _ = writeln!(
        out,
        "{indent}  \"active_fit_regime\": \"{}\",",
        sc.plan.active_fit_regime.as_str()
    );
    let _ = writeln!(
        out,
        "{indent}  \"zero_copy_active\": {},",
        sc.plan.zero_copy_active
    );
    let _ = writeln!(
        out,
        "{indent}  \"gpu_mlp_weight_fraction\": {:.4},",
        sc.plan.gpu_mlp_weight_fraction
    );
    let _ = writeln!(
        out,
        "{indent}  \"gpu_active_mac_coverage\": {:.4},",
        sc.plan.gpu_active_mac_coverage
    );
    let _ = writeln!(
        out,
        "{indent}  \"prefetch_hit_rate\": {:.4},",
        sc.plan.prefetch_hit_rate
    );
    let _ = writeln!(out, "{indent}  \"cold_ttft_ms\": {:.2},", sc.cold_ttft_ms);
    let _ = writeln!(out, "{indent}  \"warm_ttft_ms\": {:.2},", sc.warm_ttft_ms);
    let _ = writeln!(
        out,
        "{indent}  \"major_page_faults\": {},",
        sc.major_page_faults
    );
    let _ = writeln!(
        out,
        "{indent}  \"minor_page_faults\": {},",
        sc.minor_page_faults
    );
    let _ = writeln!(out, "{indent}  \"itl_p50_ms\": {:.2},", sc.itl_p50_ms);
    let _ = writeln!(out, "{indent}  \"itl_p95_ms\": {:.2},", sc.itl_p95_ms);
    let _ = writeln!(
        out,
        "{indent}  \"agent_20_turn_wall_s\": {:.3},",
        sc.agent_20_turn_wall_s
    );
    let _ = writeln!(
        out,
        "{indent}  \"peak_vram_gib\": {:.3},",
        sc.plan.peak_vram_bytes as f64 / GIB as f64
    );
    let _ = writeln!(
        out,
        "{indent}  \"peak_host_ram_gib\": {:.3},",
        sc.plan.peak_host_ram_bytes as f64 / GIB as f64
    );
    let _ = writeln!(
        out,
        "{indent}  \"peak_total_mem_gib\": {:.3},",
        sc.plan.peak_total_mem_bytes as f64 / GIB as f64
    );
    let _ = writeln!(
        out,
        "{indent}  \"vram_eviction_overflow_gib\": {:.3},",
        sc.plan.vram_eviction_overflow_bytes as f64 / GIB as f64
    );
    let _ = writeln!(
        out,
        "{indent}  \"pcie_transfer_gib\": {:.3},",
        sc.pcie_transfer_gib
    );
    let _ = writeln!(
        out,
        "{indent}  \"copy_bytes_gib\": {:.3},",
        sc.copy_bytes_gib
    );
    let _ = writeln!(
        out,
        "{indent}  \"energy_mj_per_tok\": {:.2},",
        sc.energy_mj_per_tok
    );
    let _ = writeln!(
        out,
        "{indent}  \"output_correctness_pct\": {:.3},",
        sc.output_correctness_pct
    );
    let _ = writeln!(
        out,
        "{indent}  \"quality_regression_pp\": {:.3},",
        sc.quality_regression_pp
    );
    let _ = writeln!(out, "{indent}  \"gate_accepted\": {},", sc.gate_accepted);
    let _ = writeln!(out, "{indent}  \"gate_reason\": \"{}\"", sc.gate_reason);
    let _ = writeln!(out, "{indent}}}{comma}");
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

/// Writes `results.json` into `output_dir` (creating directories if needed).
pub fn write_results_json(bundle: &BenchmarkBundle, output_dir: &Path) -> io::Result<()> {
    fs::create_dir_all(output_dir)?;
    let path = output_dir.join("results.json");
    let json = to_json(bundle);
    fs::write(path, json)
}
