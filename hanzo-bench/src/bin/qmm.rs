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
#[command(about = "Time a quantized projection at prefill shapes")]
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
    /// Weight quantization.
    #[arg(long, default_value = "q4k")]
    quant: String,
}

fn parse_shape(s: &str) -> anyhow::Result<(usize, usize)> {
    let (n, k) = s
        .split_once(',')
        .ok_or_else(|| anyhow::anyhow!("shape must be n,k: {s}"))?;
    Ok((n.trim().parse()?, k.trim().parse()?))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::new_metal(0).or_else(|_| Device::cuda_if_available(0))?;
    let dtype = match args.dtype.as_str() {
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        "f32" => DType::F32,
        o => anyhow::bail!("unknown dtype {o}"),
    };
    let quant = match args.quant.as_str() {
        "q4k" => GgmlDType::Q4K,
        "q6k" => GgmlDType::Q6K,
        "q8_0" => GgmlDType::Q8_0,
        o => anyhow::bail!("unknown quant {o}"),
    };
    // Qwen3-8B: q 4096x4096, k 1024x4096, v 1024x4096, o 4096x4096, gate/up 12288x4096, down 4096x12288.
    let default = [
        "4096,4096", "1024,4096", "1024,4096", "4096,4096", "12288,4096", "12288,4096", "4096,12288",
    ];
    let shapes: Vec<(usize, usize)> = match &args.shape {
        Some(v) => v.iter().map(|s| parse_shape(s)).collect::<Result<_, _>>()?,
        None => default.iter().map(|s| parse_shape(s)).collect::<Result<_, _>>()?,
    };

    let mut layer_ms = 0.0f64;
    println!(
        "{:>12} {:>12} {:>10} {:>10} {:>10}",
        "n", "k", "ms", "TFLOP/s", "GB/s(w)"
    );
    for (n, k) in shapes {
        // `quantize_onto` quantizes on the CPU and uploads, so the source must be a CPU tensor.
        let w = Tensor::randn(0f32, 1f32, (n, k), &Device::Cpu)?;
        let q = QTensor::quantize_onto(&w, quant, &device)?;
        let mm = QMatMul::from_qtensor(q)?;
        let x = Tensor::randn(0f32, 1f32, (args.m, k), &device)?.to_dtype(dtype)?;

        for _ in 0..3 {
            let _ = mm.forward(&x)?;
        }
        device.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..args.reps {
            let y = mm.forward(&x)?;
            std::hint::black_box(&y);
        }
        device.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / args.reps as f64;

        let flops = 2.0 * args.m as f64 * n as f64 * k as f64;
        let wbytes = n as f64 * k as f64 * quant.type_size() as f64 / quant.block_size() as f64;
        println!(
            "{n:>12} {k:>12} {ms:>10.3} {:>10.2} {:>10.1}",
            flops / (ms / 1000.0) / 1e12,
            wbytes / (ms / 1000.0) / 1e9
        );
        layer_ms += ms;
    }
    println!(
        "\nper-layer {:.3} ms; x{} layers = {:.1} ms of GEMM for m={}",
        layer_ms,
        args.layers,
        layer_ms * args.layers as f64,
        args.m
    );
    Ok(())
}
