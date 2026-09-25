//! Block-FP8 GEMM timing on the shapes Qwen3.8-Flash-Next serves, for the
//! side-by-side with vLLM's kernel (`scripts/blockwise_fp8_bench.py` reads
//! this output and prints the ratio table).
//!
//! Method, identical in both harnesses: weights rotate across copies that
//! together exceed 2x L2 (48 MiB, at most 8), one CUDA graph per shape
//! replays one GEMM per copy, 3 warm replays, then 5 rounds of 4 timed
//! replays between CUDA events. Reported per GEMM: min and median µs.
//!
//!   cargo run -p hanzo-quant --features cuda --release --example blockwise_fp8_bench -- \
//!       --out target/blockwise_fp8.hanzo.jsonl

#[cfg(has_blockwise_fp8_cutlass_kernels)]
fn main() -> hanzo_ml::Result<()> {
    bench::run()
}

#[cfg(not(has_blockwise_fp8_cutlass_kernels))]
fn main() {
    eprintln!("built without the sm_121 block-FP8 CUTLASS lib: rebuild with --features cuda and CUDA_COMPUTE_CAP=121");
    std::process::exit(2);
}

#[cfg(has_blockwise_fp8_cutlass_kernels)]
mod bench {
    use std::io::Write;

    use float8::F8E4M3;
    use hanzo_ml::cuda::cudarc::driver::sys;
    use hanzo_ml::{Device, Result, Tensor};
    use hanzo_quant::blockwise_fp8::cutlass::{self, Tile, BLOCK};

    const L2_ROTATION: usize = 48 << 20;
    const MAX_COPIES: usize = 8;
    const MAX_OUTPUT: usize = 256 << 20;
    const WARM: usize = 3;
    const ROUNDS: usize = 5;
    const REPLAYS: usize = 4;

    /// (N, K): vLLM's fused projections, then hanzo's unfused ones.
    pub const SHAPES: [(usize, usize); 10] = [
        (16384, 2560),
        (13312, 2560),
        (1280, 2560),
        (2560, 6144),
        (2560, 640),
        (10240, 2560),
        (6144, 2560),
        (12288, 2560),
        (512, 2560),
        (640, 2560),
    ];
    pub const ROWS: [usize; 20] = [
        1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 257, 512, 1024, 2048, 4096, 4097, 8192,
    ];

    pub fn copies(m: usize, n: usize, k: usize) -> usize {
        let by_l2 = L2_ROTATION.div_ceil(n * k).clamp(1, MAX_COPIES);
        by_l2.min((MAX_OUTPUT / (m * n * 2)).max(1))
    }

    fn wrap<T>(r: std::result::Result<T, hanzo_ml::cuda::cudarc::driver::DriverError>) -> Result<T> {
        r.map_err(hanzo_ml::Error::wrap)
    }

    fn codes(len: usize, seed: u64) -> Vec<F8E4M3> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| loop {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let b = (s >> 24) as u8;
                // Finite, and small enough that nothing saturates: timing only.
                if b & 0x7F < 0x60 {
                    break F8E4M3::from_bits(b);
                }
            })
            .collect()
    }

    fn utilization() -> String {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    }

    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    /// Device-to-device copy of the same traffic, same timing method: the bandwidth ceiling.
    fn copy_ceiling(dev: &hanzo_ml::CudaDevice, bytes: usize) -> Result<f64> {
        let stream = dev.cuda_stream();
        let src = unsafe { dev.alloc::<u8>(bytes)? };
        let mut dst = unsafe { dev.alloc::<u8>(bytes)? };
        let ctx = stream.context();
        let start = wrap(ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)))?;
        let end = wrap(ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)))?;
        let mut best = f64::MAX;
        for _ in 0..ROUNDS {
            wrap(start.record(&stream))?;
            for _ in 0..REPLAYS {
                wrap(stream.memcpy_dtod(&src, &mut dst))?;
            }
            wrap(end.record(&stream))?;
            best = best.min(wrap(start.elapsed_ms(&end))? as f64 * 1e3 / REPLAYS as f64);
        }
        Ok(2.0 * bytes as f64 / (best * 1e-6) / 1e9)
    }

    pub fn run() -> Result<()> {
        let mut out_path = std::path::PathBuf::from("target/blockwise_fp8.hanzo.jsonl");
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--out" => out_path = args.next().expect("--out takes a path").into(),
                other => panic!("unknown argument {other}"),
            }
        }
        let device = Device::new_cuda(0)?;
        let Device::Cuda(dev) = device.clone() else { unreachable!() };
        if !cutlass::device_supported(&dev) {
            hanzo_ml::bail!("the block-FP8 CUTLASS lib runs on sm_121 only");
        }
        // The engine's capture paths run without cross-stream event tracking.
        unsafe { dev.disable_event_tracking() };
        let stream = dev.cuda_stream();
        let ctx = stream.context();
        let mut out = std::fs::File::create(&out_path)?;

        for &(n, k) in &SHAPES {
            let m_max = *ROWS.iter().max().unwrap();
            let qa_all = Tensor::from_vec(codes(m_max * k, 11), (m_max, k), &Device::Cpu)?.to_device(&device)?;
            let sa_all = Tensor::full(1e-2f32, (m_max, k / BLOCK), &device)?;
            let copies_max = copies(1, n, k);
            let weights: Vec<(Tensor, Tensor)> = (0..copies_max)
                .map(|c| -> Result<_> {
                    Ok((
                        Tensor::from_vec(codes(n * k, 101 + c as u64), (n, k), &Device::Cpu)?.to_device(&device)?,
                        Tensor::full(1e-3f32, (n / BLOCK, k / BLOCK), &device)?,
                    ))
                })
                .collect::<Result<_>>()?;
            let copy_gbps = copy_ceiling(&dev, n * k)?;

            for &m in &ROWS {
                let c = copies(m, n, k);
                let qa = qa_all.narrow(0, 0, m)?;
                let sa = sa_all.narrow(0, 0, m)?;
                for (qw, sw) in &weights[..c] {
                    cutlass::matmul(&qa, &sa, qw, sw)?;
                }
                wrap(stream.synchronize())?;
                wrap(stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED))?;
                let captured: Result<Vec<Tensor>> =
                    weights[..c].iter().map(|(qw, sw)| cutlass::matmul(&qa, &sa, qw, sw)).collect();
                let graph = wrap(stream.end_capture(
                    sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                ))?
                .expect("capture produced a graph");
                let outputs = captured?;
                for _ in 0..WARM {
                    wrap(graph.launch())?;
                }
                let start = wrap(ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)))?;
                let end = wrap(ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)))?;
                let mut per_gemm = Vec::with_capacity(ROUNDS);
                for _ in 0..ROUNDS {
                    wrap(start.record(&stream))?;
                    for _ in 0..REPLAYS {
                        wrap(graph.launch())?;
                    }
                    wrap(end.record(&stream))?;
                    per_gemm.push(wrap(start.elapsed_ms(&end))? as f64 * 1e3 / (REPLAYS * c) as f64);
                }
                wrap(stream.synchronize())?;
                drop(outputs);
                drop(graph);

                let min_us = per_gemm.iter().cloned().fold(f64::MAX, f64::min);
                let median_us = median(&mut per_gemm);
                let bytes = (n * k + m * k + 2 * m * n + 4 * (n / BLOCK + m) * (k / BLOCK)) as f64;
                let line = serde_json::json!({
                    "engine": "hanzo",
                    "n": n, "k": k, "m": m,
                    "tile": format!("{:?}", Tile::for_rows(m)),
                    "copies": c,
                    "min_us": min_us,
                    "median_us": median_us,
                    "gbps": bytes / (min_us * 1e-6) / 1e9,
                    "tflops": 2.0 * (m * n * k) as f64 / (min_us * 1e-6) / 1e12,
                    "copy_gbps": copy_gbps,
                    "util": utilization(),
                });
                writeln!(out, "{line}")?;
                println!("{line}");
            }
        }
        println!("wrote {}", out_path.display());
        Ok(())
    }
}
