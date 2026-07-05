// Prefill micro-benchmark for the unified native Q4_K int8-WMMA GEMM (qmmq_core<Q4_K>). Times
// QMatMul::forward(x) with rows>1 (the prefill path) on a production Q4_K shape and reports us/launch.
// Not a correctness gate (that's qmmq_unified_numeric); this isolates the native prefill kernel cost.
// Lives in hanzo-cli so it links the HIP runtime via hanzo-cli's build.rs.
#![cfg(feature = "rocm")]

use hanzo_ml::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use hanzo_ml::{DType, Device, Module, Tensor};
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

fn time_forward(qm: &QMatMul, x: &Tensor, dev: &Device, iters: usize) -> f64 {
    for _ in 0..3 {
        let _ = qm.forward(x).expect("forward warmup");
    }
    dev.synchronize().expect("sync");
    let t0 = Instant::now();
    for _ in 0..iters {
        let y = qm.forward(x).expect("forward");
        std::mem::drop(y);
    }
    dev.synchronize().expect("sync");
    t0.elapsed().as_secs_f64() / iters as f64 * 1e6 // us/launch
}

#[test]
fn qmmq_q4k_prefill_perf() {
    let dev = Device::new_rocm(0).expect("rocm device");
    // Production-ish Q4_K prefill: M = prompt tokens, K = hidden, N = FFN width (Qwen3-14B: 5120 ->
    // 17408). K and N are multiples of 256 (the Q4_K super-block). gate/up share this shape.
    let m = 512usize;
    let k = 5120usize;
    let n = 17408usize;

    // Build a Q4_K weight [n, k]: quantize on CPU (Q4_K from_float is CPU-only), then upload the raw
    // GGML bytes to the rocm device via from_data -- exactly the GGUF model load path. from_arc then
    // recognizes the rocm Q4_K QTensor and routes decode/prefill through the native unified core.
    let wsrc = Tensor::randn(0f32, 1.0, (n, k), &Device::Cpu).expect("wsrc");
    let qt_cpu = QTensor::quantize(&wsrc, GgmlDType::Q4K).expect("quantize Q4_K (cpu)");
    let bytes = qt_cpu.data().expect("q4k bytes");
    let storage = QStorage::from_data(bytes, &dev, GgmlDType::Q4K).expect("upload Q4_K to rocm");
    let qt = QTensor::new(storage, (n, k)).expect("rocm Q4_K qtensor");
    let qm = QMatMul::from_arc(Arc::new(qt)).expect("qmatmul");

    // Activation [m, k] f16 on device.
    let x = Tensor::randn(0f32, 1.0, (m, k), &Device::Cpu)
        .expect("x")
        .to_dtype(DType::F16)
        .expect("f16")
        .to_device(&dev)
        .expect("to dev");

    let iters = 30;

    // Native unified int8 WMMA prefill (qmmq_core<Q4_K>).
    let native = time_forward(&qm, &x, &dev, iters);

    let line = format!("Q4_K prefill m={m} k={k} n={n}: native={native:.1}us\n");
    eprintln!("{line}");
    let _ = std::fs::File::create("C:\\qmmq-q4k-prefill-perf.txt")
        .and_then(|mut f| f.write_all(line.as_bytes()));

    // Sanity: native path produces the right shape (correctness is the numeric gate).
    let y = qm.forward(&x).expect("forward final");
    assert_eq!(y.dims(), &[m, n]);
}
