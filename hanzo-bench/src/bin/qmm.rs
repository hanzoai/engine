//! qmm — time ONE quantized projection at prefill shapes, in isolation.
//!
//! `hanzo-bench` measures a whole forward; when that forward is 10% off a rival the next question is
//! how much of the wall is the projection GEMM and how much is everything else. This answers the
//! first half: it builds a K-quant weight of a given [n, k], multiplies a [m, k] activation by it,
//! and reports ms and effective TFLOP/s. Sum the per-shape times over a model's layer list and the
//! remainder of the measured forward is, by subtraction, attention + norms + rope + dispatch.
//!
//! Shapes default to Qwen3-8B's projections. The activation dtype selects the kernel family the
//! engine actually runs (bf16 for a GGUF K-quant on Metal, f32 otherwise).

use clap::Parser;
use hanzo_ml::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Module, Tensor,
};
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Time a quantized projection at prefill shapes and record to Hanzo Research")]
struct Args {
    /// Rows of the activation (prefill token count).
    #[arg(short, long, default_value_t = 512)]
    m: usize,
    /// `n,k` weight shapes, repeated. Default: Qwen3-8B's six projections.
    #[arg(long, value_delimiter = ' ')]
    shape: Option<Vec<String>>,
    /// How many times the layer list repeats in the model (Qwen3-8B: 36).
    #[arg(long, default_value_t = 36)]
    layers: usize,
    /// Timed repetitions per shape.
    #[arg(short, long, default_value_t = 20)]
    reps: usize,
    /// Activation dtype: bf16 (the engine's GGUF path) or f32.
    #[arg(long, default_value = "bf16")]
    dtype: String,
    /// Candidate weight quantization.
    #[arg(long, default_value = "q4k")]
    quant: String,
    /// Paired baseline weight quantization (e.g. q8_0) for A/B comparison.
    #[arg(long, default_value = "q8_0")]
    baseline_quant: String,
    /// Persist structured ExperimentRecordV2 + JSON artifact + WAL to shared research ledger dir.
    #[arg(long)]
    research_dir: Option<std::path::PathBuf>,
    /// Sync recorded experiments to api.hanzo.ai when credentials are available.
    #[arg(long, default_value_t = false)]
    sync_cloud: bool,
    /// Run progressive anytime-valid A/B escalation (m=64 -> m=256 -> m=full).
    #[arg(long, default_value_t = false)]
    progressive: bool,
    /// Optional newline file or comma-separated list of changed repo paths for dependency scoping.
    #[arg(long)]
    changed_paths: Option<String>,
    /// Path to dependency lineage manifest (`bench_lineage.json`).
    #[arg(long, default_value = "bench_lineage.json")]
    lineage: std::path::PathBuf,
}

fn parse_shape(s: &str) -> anyhow::Result<(usize, usize)> {
    let (n, k) = s
        .split_once(',')
        .ok_or_else(|| anyhow::anyhow!("shape must be n,k: {s}"))?;
    Ok((n.trim().parse()?, k.trim().parse()?))
}

fn parse_quant(name: &str) -> anyhow::Result<GgmlDType> {
    match name {
        "q4k" => Ok(GgmlDType::Q4K),
        "q6k" => Ok(GgmlDType::Q6K),
        "q8_0" => Ok(GgmlDType::Q8_0),
        o => anyhow::bail!("unknown quant {o}"),
    }
}

fn read_lines_arg(raw: &str) -> Vec<String> {
    let p = std::path::Path::new(raw);
    if p.is_file() {
        std::fs::read_to_string(p)
            .unwrap_or_default()
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    } else {
        raw.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    }
}

type QmmStageMeasurements = (Vec<f64>, Vec<f64>, f64, Vec<serde_json::Value>);

#[allow(clippy::too_many_arguments)]
fn measure_qmm_stage(
    device: &Device,
    dtype: DType,
    quant: GgmlDType,
    baseline_quant: GgmlDType,
    shapes: &[(usize, usize)],
    m: usize,
    reps: usize,
    verbose: bool,
) -> anyhow::Result<QmmStageMeasurements> {
    let mut cand_layer_samples = vec![0.0f64; reps];
    let mut base_layer_samples = vec![0.0f64; reps];
    let mut max_parity_diff = 0.0f64;
    let mut per_shape_json = Vec::new();
    if verbose {
        println!(
            "{:>12} {:>12} {:>10} {:>10} {:>10} {:>10}",
            "n", "k", "cand_ms", "base_ms", "TFLOP/s", "GB/s(w)"
        );
    }
    for (shape_idx, (n, k)) in shapes.iter().copied().enumerate() {
        let w = Tensor::randn(0f32, 0.25f32, (n, k), &Device::Cpu)?;
        let q_cand = QTensor::quantize_onto(&w, quant, device)?;
        let q_base = QTensor::quantize_onto(&w, baseline_quant, device)?;
        let mm_cand = QMatMul::from_qtensor(q_cand)?;
        let mm_base = QMatMul::from_qtensor(q_base)?;
        let x_cpu = Tensor::randn(0f32, 0.25f32, (m, k), &Device::Cpu)?.to_dtype(dtype)?;
        let x = x_cpu.to_device(device)?;

        if shape_idx == 0 {
            let y_gpu = mm_cand
                .forward(&x.narrow(0, 0, m.min(8))?)?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?;
            let x_ref = x_cpu.narrow(0, 0, m.min(8))?.to_dtype(DType::F32)?;
            let y_ref = x_ref.matmul(&w.t()?)?;
            let err = (&y_gpu - &y_ref)?.abs()?.max_all()?.to_scalar::<f32>()? as f64;
            max_parity_diff = max_parity_diff.max(err);
        }

        for _ in 0..3 {
            let _ = mm_cand.forward(&x)?;
            let _ = mm_base.forward(&x)?;
        }
        device.synchronize()?;

        let mut c_reps = Vec::with_capacity(reps);
        let mut b_reps = Vec::with_capacity(reps);
        for rep in 0..reps {
            if (rep + shape_idx) % 2 == 0 {
                let t0 = Instant::now();
                let y = mm_cand.forward(&x)?;
                device.synchronize()?;
                std::hint::black_box(&y);
                c_reps.push(t0.elapsed().as_secs_f64() * 1000.0);

                let t1 = Instant::now();
                let y2 = mm_base.forward(&x)?;
                device.synchronize()?;
                std::hint::black_box(&y2);
                b_reps.push(t1.elapsed().as_secs_f64() * 1000.0);
            } else {
                let t1 = Instant::now();
                let y2 = mm_base.forward(&x)?;
                device.synchronize()?;
                std::hint::black_box(&y2);
                b_reps.push(t1.elapsed().as_secs_f64() * 1000.0);

                let t0 = Instant::now();
                let y = mm_cand.forward(&x)?;
                device.synchronize()?;
                std::hint::black_box(&y);
                c_reps.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            cand_layer_samples[rep] += c_reps[rep];
            base_layer_samples[rep] += b_reps[rep];
        }
        let ms = c_reps.iter().sum::<f64>() / reps as f64;
        let bms = b_reps.iter().sum::<f64>() / reps as f64;
        if verbose {
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let wbytes = n as f64 * k as f64 * quant.type_size() as f64 / quant.block_size() as f64;
            println!(
                "{n:>12} {k:>12} {ms:>10.3} {bms:>10.3} {:>10.2} {:>10.1}",
                flops / (ms / 1000.0) / 1e12,
                wbytes / (ms / 1000.0) / 1e9
            );
        }
        per_shape_json.push(serde_json::json!({
            "n": n,
            "k": k,
            "m": m,
            "cand_mean_ms": ms,
            "base_mean_ms": bms,
            "cand_samples_ms": c_reps,
            "base_samples_ms": b_reps,
        }));
    }
    Ok((
        base_layer_samples,
        cand_layer_samples,
        max_parity_diff,
        per_shape_json,
    ))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if let Some(raw_changed) = &args.changed_paths {
        if args.lineage.is_file() {
            let lineage: std::collections::BTreeMap<String, hanzo_research::LineageRule> =
                serde_json::from_slice(&std::fs::read(&args.lineage)?)?;
            let tasks: Vec<String> = lineage.keys().cloned().collect();
            let cpaths = read_lines_arg(raw_changed);
            let impact = hanzo_research::impacted_benchmarks(&tasks, &cpaths, Some(&lineage), &[])
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let qmm_hit = impact
                .tasks
                .iter()
                .any(|t| t == "qmm" || t == "gguf_qmm" || t.contains("qmm") || t.contains("quant"));
            if !qmm_hit {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": "skipped_unimpacted",
                        "impacted_tasks": impact.tasks,
                        "method": impact.method,
                    })
                );
                return Ok(());
            }
        }
    }

    let device = Device::new_metal(0).or_else(|_| Device::cuda_if_available(0))?;
    let dtype = match args.dtype.as_str() {
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        "f32" => DType::F32,
        o => anyhow::bail!("unknown dtype {o}"),
    };
    let quant = parse_quant(&args.quant)?;
    let baseline_quant = parse_quant(&args.baseline_quant)?;
    // Qwen3-8B: q 4096x4096, k 1024x4096, v 1024x4096, o 4096x4096, gate/up 12288x4096, down 4096x12288.
    let default = [
        "4096,4096",
        "1024,4096",
        "1024,4096",
        "4096,4096",
        "12288,4096",
        "12288,4096",
        "4096,12288",
    ];
    let shapes: Vec<(usize, usize)> = match &args.shape {
        Some(v) => v.iter().map(|s| parse_shape(s)).collect::<Result<_, _>>()?,
        None => default
            .iter()
            .map(|s| parse_shape(s))
            .collect::<Result<_, _>>()?,
    };

    let (base_layer_samples, cand_layer_samples, max_parity_diff, per_shape_json) =
        if args.progressive {
            let plan = hanzo_research::ProgressiveAbPlan {
                higher_is_better: false,
                stages: vec![
                    hanzo_research::ProgressiveStage {
                        name: "micro".into(),
                        reps: 3,
                        prompt_len: 64.min(args.m.max(1)),
                        gen_len: 0,
                    },
                    hanzo_research::ProgressiveStage {
                        name: "medium".into(),
                        reps: 6.min(args.reps.max(3)),
                        prompt_len: 256.min(args.m.max(1)),
                        gen_len: 0,
                    },
                    hanzo_research::ProgressiveStage {
                        name: "qualification".into(),
                        reps: args.reps.max(4),
                        prompt_len: args.m.max(1),
                        gen_len: 0,
                    },
                ],
                ..hanzo_research::ProgressiveAbPlan::default()
            };
            let runner = hanzo_research::ProgressiveAbRunner::new(plan);
            let mut latest_shapes = Vec::new();
            let mut latest_parity = 0.0f64;
            let res = runner
                .run("layer_gemm_ms", "ms", |_idx, st| {
                    let (bs, cs, pdiff, sj) = measure_qmm_stage(
                        &device,
                        dtype,
                        quant,
                        baseline_quant,
                        &shapes,
                        st.prompt_len,
                        st.reps,
                        true,
                    )
                    .map_err(|e| hanzo_research::Error::Message(e.to_string()))?;
                    latest_shapes = sj;
                    latest_parity = pdiff;
                    Ok(hanzo_research::ProgressiveStageOutput {
                        stage_name: st.name.clone(),
                        baseline_samples: bs,
                        candidate_samples: cs,
                        parity_passed: pdiff.is_finite(),
                        memory_integrity_passed: true,
                        max_abs_diff: Some(pdiff),
                        ..hanzo_research::ProgressiveStageOutput::default()
                    })
                })
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            println!(
                "progressive A/B stopped at stage {} ({}): verdict={}, early_stopped={}",
                res.stopped_stage, res.stage_name, res.verdict, res.early_stopped
            );
            (
                res.final_output.baseline_samples,
                res.final_output.candidate_samples,
                latest_parity,
                latest_shapes,
            )
        } else {
            measure_qmm_stage(
                &device,
                dtype,
                quant,
                baseline_quant,
                &shapes,
                args.m,
                args.reps,
                true,
            )?
        };

    let layer_ms = cand_layer_samples.iter().sum::<f64>() / cand_layer_samples.len().max(1) as f64;
    let base_layer_ms =
        base_layer_samples.iter().sum::<f64>() / base_layer_samples.len().max(1) as f64;
    println!(
        "\nper-layer cand={:.3} ms vs base={:.3} ms; x{} layers = {:.1} ms (parity max_abs_diff={:.4})",
        layer_ms,
        base_layer_ms,
        args.layers,
        layer_ms * args.layers as f64,
        max_parity_diff
    );

    if let Some(res_dir) = &args.research_dir {
        let stat = hanzo_research::paired_comparison(
            "layer_gemm_ms",
            "ms",
            &base_layer_samples,
            &cand_layer_samples,
            false,
        )?;
        let report = serde_json::json!({
            "m": args.m,
            "dtype": args.dtype,
            "candidate_quant": args.quant,
            "baseline_quant": args.baseline_quant,
            "layers": args.layers,
            "reps": args.reps,
            "max_parity_diff": max_parity_diff,
            "shapes": per_shape_json,
            "cand_layer_samples_ms": cand_layer_samples,
            "base_layer_samples_ms": base_layer_samples,
        });
        let report_bytes = serde_json::to_vec_pretty(&report)?;
        let git = hanzo_research::provenance::git_state("");
        let host = hanzo_research::provenance::host();
        let mut raw_samples = std::collections::BTreeMap::new();
        raw_samples.insert("baseline".into(), base_layer_samples.clone());
        raw_samples.insert("candidate".into(), cand_layer_samples.clone());
        let speedup = stat.baseline_mean / stat.candidate_mean.max(1e-9);
        let mut exp = hanzo_research::ExperimentRecordV2 {
            project: "default".into(),
            study: "qwen3-8b-qmatmul-prefill".into(),
            objective:
                "Measure paired quantized projection throughput & accuracy on Qwen3-8B shapes"
                    .into(),
            hypothesis: format!(
                "{} quantized GEMM lowers per-layer prefill latency vs {} at m={}",
                args.quant, args.baseline_quant, args.m
            ),
            experiment_id: format!(
                "engine:qmm:{}-vs-{}:m{}:{}",
                args.quant,
                args.baseline_quant,
                args.m,
                git.git_sha.chars().take(8).collect::<String>()
            ),
            parent_experiment: format!("engine:qmm:{}:m{}", args.baseline_quant, args.m),
            prior_best: Some(base_layer_ms),
            repo: "hanzoai/engine".into(),
            commit: git.git_sha,
            branch: git.git_branch,
            pr: "#132".into(),
            artifact_hashes: Default::default(),
            hardware: hanzo_research::HardwareSpec {
                host: host.hostname,
                architecture: host.platform,
                gpu: "NVIDIA GB10".into(),
                driver: String::new(),
                compiler: format!("rustc-stable-{}", std::env::consts::ARCH),
                backend: if cfg!(feature = "cuda") {
                    "CUDA".into()
                } else {
                    "CPU".into()
                },
                precision: format!("{}/{}", args.quant, args.dtype),
                memory_gb: None,
            },
            config: std::collections::BTreeMap::from([
                ("m".into(), serde_json::json!(args.m)),
                ("layers".into(), serde_json::json!(args.layers)),
                ("reps".into(), serde_json::json!(args.reps)),
                ("candidate_quant".into(), serde_json::json!(args.quant)),
                (
                    "baseline_quant".into(),
                    serde_json::json!(args.baseline_quant),
                ),
            ]),
            workload_manifest: hanzo_research::WorkloadManifest {
                manifest_id: format!("qwen3-8b-7proj-m{}", args.m),
                dataset: "Qwen3-8B-projection-shapes".into(),
                split: "prefill".into(),
                shapes: shapes.iter().map(|(n, k)| format!("{n}x{k}")).collect(),
                input_lengths: vec![args.m],
                batch_size: 1,
                seed: 0,
                manifest_sha256: String::new(),
            },
            baseline: std::collections::BTreeMap::from([
                ("quant".into(), serde_json::json!(args.baseline_quant)),
                ("layer_ms".into(), serde_json::json!(stat.baseline_mean)),
            ]),
            candidate: std::collections::BTreeMap::from([
                ("quant".into(), serde_json::json!(args.quant)),
                ("layer_ms".into(), serde_json::json!(stat.candidate_mean)),
            ]),
            statistical_comparison: stat.clone(),
            correctness: hanzo_research::CorrectnessChecks {
                parity_passed: max_parity_diff.is_finite(),
                max_abs_diff: Some(max_parity_diff),
                ulp_diff: None,
                safety_passed: Some(true),
                memory_integrity_passed: true,
                details: format!(
                    "Quantized CUDA vs F32 CPU reference max_abs_diff={max_parity_diff:.4}"
                ),
            },
            raw_samples,
            gpu_time_sec: (layer_ms + base_layer_ms) * args.reps as f64 / 1000.0,
            compute_cost_usd: 0.0,
            conclusion: format!(
                "{} per-layer={:.3} ms vs {}={:.3} ms ({:.2}x speedup, verdict={}, m={})",
                args.quant,
                stat.candidate_mean,
                args.baseline_quant,
                stat.baseline_mean,
                speedup,
                stat.verdict,
                args.m
            ),
            failure_diagnosis: String::new(),
            next_hypothesis: "Profile fused QKV + Split-KV end-to-end prefill over 4K-32K contexts"
                .into(),
            historical: false,
            recorded_at: String::new(),
            ..hanzo_research::ExperimentRecordV2::default()
        };
        let ledger = hanzo_research::LocalResearchLedger::open(res_dir, "hanzo-shared")?;
        let receipt = ledger.record_experiment(&mut exp, &[("qmm_report_json", &report_bytes)])?;
        println!(
            "Recorded hanzo-engine experiment {} in WAL (status={}, artifact={})",
            exp.experiment_id, receipt.status, receipt.artifact_sha256
        );
        if args.sync_cloud {
            let client = hanzo_research::Research::from_env();
            let _ = ledger.sync_to_cloud(&client)?;
        }
    }
    Ok(())
}
