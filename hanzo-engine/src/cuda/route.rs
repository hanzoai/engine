//! Fused MoE router: vLLM v0.29.0's `topkGating`, ported verbatim to `exact/route.cu` and built
//! without fast math, so ids and weights equal vLLM's bit for bit (score, top-k with ties to the
//! lower index, renormalize, routed scaling factor) in one launch.

use std::ffi::c_void;
use std::sync::OnceLock;

use hanzo_ml::backend::BackendStorage;
use hanzo_ml::cuda_backend::cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut};
use hanzo_ml::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
use hanzo_ml::{DType, Result, Shape, Storage, Tensor};

use super::ffi;
use crate::ops::{MoeRouterScoreFunction, MoeRouterSelectedWeight, MoeRouterTopKConfig, TopKOutput};

/// The expert counts `route.cu` instantiates: vLLM's fused set (1-512 powers of two, 192, 320,
/// 384, 448, 576). Others take the portable path, as they take vLLM's unfused one.
pub(crate) fn experts() -> &'static [usize] {
    static TABLE: OnceLock<Vec<usize>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table: *const i32 = std::ptr::null();
        let n = unsafe { ffi::route_experts(&mut table) };
        unsafe { std::slice::from_raw_parts(table, n as usize) }
            .iter()
            .map(|&e| e as usize)
            .collect()
    })
}

pub(crate) fn supports(experts: usize) -> bool {
    self::experts().contains(&experts)
}

/// Whether `cfg` is one `route.cu` computes. A sigmoid of the selected logit exists only for the
/// raw score (llama4); every other combination is vLLM's or an extension of it.
pub(crate) fn serves(experts: usize, top_k: usize, cfg: &MoeRouterTopKConfig) -> bool {
    supports(experts)
        && top_k >= 1
        && top_k <= experts
        && (matches!(cfg.selected_weight, MoeRouterSelectedWeight::Score)
            || matches!(cfg.score_function, MoeRouterScoreFunction::Raw))
}

/// Raw launch on `stream` into preallocated outputs, for callers that own their buffers (graph
/// capture, a Lane). `logits` is `rows x experts` row-major, 16-byte aligned; `bias` and `scale`
/// are f32 `[experts]` or null.
///
/// # Safety
/// Every pointer must be a live device allocation of the stated size on `stream`'s context.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch(
    dtype: DType,
    logits: *const c_void,
    weights: *mut f32,
    ids: *mut u32,
    bias: *const f32,
    scale: *const f32,
    rows: usize,
    experts: usize,
    cfg: &MoeRouterTopKConfig,
    stream: &CudaStream,
) -> Result<()> {
    let (clip, lo, hi) = match cfg.logit_clip {
        Some((lo, hi)) => (true, lo, hi),
        None => (false, 0.0, 0.0),
    };
    let score = match cfg.score_function {
        MoeRouterScoreFunction::Raw => 0,
        MoeRouterScoreFunction::Softmax => 1,
        MoeRouterScoreFunction::Sigmoid => 2,
    };
    let weight = match cfg.selected_weight {
        MoeRouterSelectedWeight::Score => 0,
        MoeRouterSelectedWeight::Sigmoid => 1,
    };
    let f = match dtype {
        DType::F32 => ffi::route_f32,
        DType::BF16 => ffi::route_bf16,
        DType::F16 => ffi::route_f16,
        dt => hanzo_ml::bail!("route: unsupported logits dtype {dt:?}"),
    };
    let rc = f(
        logits,
        weights,
        ids,
        bias,
        scale,
        rows as i32,
        experts as i32,
        cfg.top_k as i32,
        score,
        weight,
        cfg.renormalize,
        clip,
        lo,
        hi,
        cfg.norm_min,
        cfg.output_scale,
        stream.cu_stream() as i64,
    );
    match rc {
        0 => Ok(()),
        1 => hanzo_ml::bail!("route: {experts} experts is not a fused count"),
        _ => hanzo_ml::bail!("route: kernel launch failed"),
    }
}

fn f32_vector<'a>(t: Option<&'a Tensor>, experts: usize, what: &str) -> Result<Option<Tensor>> {
    let Some(t) = t else { return Ok(None) };
    if t.dtype() != DType::F32 || t.elem_count() != experts {
        hanzo_ml::bail!("route: {what} must be F32 [{experts}]");
    }
    Ok(Some(t.contiguous()?))
}

/// Top-k routing of `logits` (`[..., experts]`, f32/bf16/f16, CUDA) on the logits' stream.
/// Returns f32 weights and u32 ids shaped `[..., top_k]`.
pub(crate) fn topk(
    logits: &Tensor,
    cfg: MoeRouterTopKConfig,
    bias: Option<&Tensor>,
    scale: Option<&Tensor>,
) -> Result<TopKOutput> {
    let experts = *logits
        .dims()
        .last()
        .ok_or_else(|| hanzo_ml::Error::Msg("route: scalar logits".into()))?;
    if !serves(experts, cfg.top_k, &cfg) {
        hanzo_ml::bail!("route: {experts} experts, top_k {} is not served", cfg.top_k);
    }
    let bias = f32_vector(bias, experts, "bias")?;
    let scale = f32_vector(scale, experts, "expert_scale")?;
    let logits = logits.contiguous()?;
    let rows = logits.elem_count() / experts;
    let mut out_dims = logits.dims().to_vec();
    *out_dims.last_mut().unwrap() = cfg.top_k;

    let (storage, layout) = logits.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        hanzo_ml::bail!("route: logits must be on CUDA");
    };
    let dev = storage.device().clone();
    let stream = dev.cuda_stream();
    let bias_sl = bias.as_ref().map(|t| t.storage_and_layout());
    let scale_sl = scale.as_ref().map(|t| t.storage_and_layout());
    let f32_ptr = |sl: &Option<(
        std::sync::RwLockReadGuard<'_, Storage>,
        &hanzo_ml::Layout,
    )>|
     -> Result<*const f32> {
        let Some((s, l)) = sl else {
            return Ok(std::ptr::null());
        };
        let Storage::Cuda(CudaStorage {
            slice: CudaStorageSlice::F32(s),
            ..
        }) = &**s
        else {
            hanzo_ml::bail!("route: bias and expert_scale must be CUDA f32");
        };
        let (p, _) = s.device_ptr(&stream);
        Ok((p as usize + l.start_offset() * 4) as *const f32)
    };
    let bias_ptr = f32_ptr(&bias_sl)?;
    let scale_ptr = f32_ptr(&scale_sl)?;

    let n = rows * cfg.top_k;
    let mut weights = unsafe { dev.alloc::<f32>(n) }?;
    let mut ids = unsafe { dev.alloc::<u32>(n) }?;
    {
        let (w_ptr, _wg) = weights.device_ptr_mut(&stream);
        let (i_ptr, _ig) = ids.device_ptr_mut(&stream);
        macro_rules! go {
            ($variant:ident, $t:ty) => {{
                let CudaStorageSlice::$variant(src) = &storage.slice else {
                    hanzo_ml::bail!("route: logits dtype mismatch");
                };
                let bytes = std::mem::size_of::<$t>();
                let len = rows * experts;
                let (base, _g) = src.device_ptr(&stream);
                let at = base as usize + layout.start_offset() * bytes;
                // vLLM loads a row in 16-byte vectors: a view that starts off that alignment is
                // copied out first rather than read misaligned.
                let copy = if at % 16 != 0 {
                    let view = src.slice(layout.start_offset()..layout.start_offset() + len);
                    Some(stream.clone_dtod(&view).w()?)
                } else {
                    None
                };
                let guard = copy.as_ref().map(|c| c.device_ptr(&stream));
                let ptr = guard.as_ref().map_or(at, |(p, _)| *p as usize);
                unsafe {
                    launch(
                        logits.dtype(),
                        ptr as *const c_void,
                        w_ptr as *mut f32,
                        i_ptr as *mut u32,
                        bias_ptr,
                        scale_ptr,
                        rows,
                        experts,
                        &cfg,
                        &stream,
                    )?;
                }
            }};
        }
        match logits.dtype() {
            DType::F32 => go!(F32, f32),
            DType::BF16 => go!(BF16, half::bf16),
            DType::F16 => go!(F16, half::f16),
            dt => hanzo_ml::bail!("route: unsupported logits dtype {dt:?}"),
        }
    }
    let wrap = |slice| {
        Tensor::from((
            Storage::Cuda(CudaStorage {
                slice,
                device: dev.clone(),
            }),
            Shape::from_dims(&out_dims),
        ))
    };
    Ok(TopKOutput {
        values: wrap(CudaStorageSlice::F32(weights)),
        indices: wrap(CudaStorageSlice::U32(ids)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;
    use std::collections::HashMap;

    pub(crate) const FIXTURES: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen4exp_moe");

    /// The golden script's configs: (name, score, renormalize, bias, routed scaling factor).
    fn configs() -> Vec<(String, MoeRouterScoreFunction, bool, bool, f32)> {
        let mut v = vec![
            ("softmax.renorm".into(), MoeRouterScoreFunction::Softmax, true, false, 1.0),
            ("softmax.plain".into(), MoeRouterScoreFunction::Softmax, false, false, 1.0),
            ("softmax.bias".into(), MoeRouterScoreFunction::Softmax, true, true, 1.0),
        ];
        for renorm in [true, false] {
            for bias in [false, true] {
                for rsf in [1.0f32, 2.5] {
                    let name = format!(
                        "sigmoid.{}.{}.rsf{}",
                        if renorm { "renorm" } else { "plain" },
                        if bias { "bias" } else { "nobias" },
                        rsf
                    );
                    v.push((name, MoeRouterScoreFunction::Sigmoid, renorm, bias, rsf));
                }
            }
        }
        v
    }

    pub(crate) fn cfg(score: MoeRouterScoreFunction, k: usize, renorm: bool, rsf: f32) -> MoeRouterTopKConfig {
        MoeRouterTopKConfig {
            top_k: k,
            score_function: score,
            selected_weight: MoeRouterSelectedWeight::Score,
            renormalize: renorm,
            norm_min: 0.0,
            output_scale: rsf,
            logit_clip: None,
        }
    }

    pub(crate) fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    pub(crate) fn ids(t: &Tensor) -> Vec<u32> {
        match t.dtype() {
            DType::U32 => t.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
            DType::I32 => t
                .flatten_all()
                .unwrap()
                .to_vec1::<i32>()
                .unwrap()
                .iter()
                .map(|&v| v as u32)
                .collect(),
            dt => panic!("ids dtype {dt:?}"),
        }
    }

    pub(crate) fn load(name: &str, dev: &Device) -> HashMap<String, Tensor> {
        hanzo_ml::safetensors::load(format!("{FIXTURES}/{name}.safetensors"), dev).unwrap()
    }

    /// Every (count, dtype, config) route.cu instantiates, bitwise against vLLM's topkGating.
    #[test]
    fn matches_vllm() {
        let dev = Device::new_cuda(0).unwrap();
        let g = load("route", &dev);
        let (mut cases, mut words) = (0usize, 0usize);
        for &e in experts() {
            for dt in ["f32", "bf16", "f16"] {
                let x = g
                    .get(&format!("x.{e}.{dt}"))
                    .unwrap_or_else(|| panic!("no golden for {e} experts {dt}"));
                let bias = &g[&format!("bias.{e}")];
                for (name, score, renorm, use_bias, rsf) in configs() {
                    let k = e.min(10);
                    let out = topk(
                        x,
                        cfg(score, k, renorm, rsf),
                        use_bias.then_some(bias),
                        None,
                    )
                    .unwrap();
                    let wv = &g[&format!("w.{e}.{dt}.{name}")];
                    let iv = &g[&format!("i.{e}.{dt}.{name}")];
                    let (a, b) = (bits(&out.values), bits(wv));
                    let (c, d) = (ids(&out.indices), ids(iv));
                    let bad = a.iter().zip(&b).filter(|(x, y)| x != y).count()
                        + c.iter().zip(&d).filter(|(x, y)| x != y).count();
                    assert_eq!(a.len(), b.len());
                    assert_eq!(
                        bad, 0,
                        "{e} experts {dt} {name}: {bad} words differ from vLLM"
                    );
                    cases += 1;
                    words += a.len() + c.len();
                }
            }
        }
        println!("matches_vllm: {cases} cases, {words} words, 0 differ");
    }

    /// A row-narrowed view, aligned or not, routes the same as its own copy.
    #[test]
    fn view_offset() {
        let dev = Device::new_cuda(0).unwrap();
        let g = load("route", &dev);
        for (e, dt) in [(512usize, "bf16"), (64, "f32"), (8, "f16"), (192, "bf16")] {
            let x = &g[&format!("x.{e}.{dt}")];
            let c = cfg(MoeRouterScoreFunction::Softmax, e.min(10), true, 1.0);
            // Row 1 starts off 16 bytes for 8 f16 experts only; the others stay aligned.
            let view = x.narrow(0, 1, 40).unwrap();
            let copy = Tensor::from_vec(
                view.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap(),
                view.dims(),
                &dev,
            )
            .unwrap()
            .to_dtype(x.dtype())
            .unwrap();
            let a = topk(&view, c, None, None).unwrap();
            let b = topk(&copy, c, None, None).unwrap();
            assert_eq!(bits(&a.values), bits(&b.values), "{e} {dt}");
            assert_eq!(ids(&a.indices), ids(&b.indices), "{e} {dt}");
        }
    }

    /// The engine's extensions: raw score with the taken mask, clip, expert scale and norm_min
    /// against the portable reference path.
    #[test]
    fn extensions_match_portable() {
        let dev = Device::new_cuda(0).unwrap();
        let g = load("route", &dev);
        let x = g["x.128.f32"].narrow(0, 0, 64).unwrap();
        let scale = (Tensor::arange(0f32, 128., &dev).unwrap() / 64.).unwrap();
        let mut raw = cfg(MoeRouterScoreFunction::Raw, 4, false, 1.0);
        raw.selected_weight = MoeRouterSelectedWeight::Sigmoid;
        let mut clip = cfg(MoeRouterScoreFunction::Softmax, 8, true, 1.0);
        clip.logit_clip = Some((-1.0, 3.5));
        let mut floor = cfg(MoeRouterScoreFunction::Sigmoid, 6, true, 2.5);
        floor.norm_min = 1e-20;
        for (name, c, s) in [("raw", raw, None), ("clip", clip, Some(&scale)), ("floor", floor, None)] {
            let fused = topk(&x, c, None, s).unwrap();
            let cpu = x.to_device(&Device::Cpu).unwrap();
            let s_cpu = s.map(|t| t.to_device(&Device::Cpu).unwrap());
            let port = crate::ops::moe_router_topk(&cpu, c, None, s_cpu.as_ref()).unwrap();
            assert_eq!(ids(&fused.indices), ids(&port.indices), "{name} ids");
            let (a, b) = (
                fused.values.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                port.values.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            );
            for (u, v) in a.iter().zip(&b) {
                assert!((u - v).abs() <= 2e-6 * v.abs().max(1e-30), "{name}: {u} vs {v}");
            }
        }
    }

    /// Paired device-time protocol shared by the MoE microbenchmarks: each side is one CUDA
    /// graph of `launches` calls; the graphs replay interleaved; a pair counts only when
    /// nvidia-smi shows <= 5% utilization before and after it.
    pub(crate) mod paired {
        use hanzo_ml::cuda_backend::cudarc::driver::sys::{
            CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode,
        };
        use hanzo_ml::cuda_backend::cudarc::driver::{CudaGraph, CudaStream};
        use std::sync::Arc;

        pub const PAIRS: usize = 20;
        pub const ATTEMPTS: usize = 200;

        pub fn capture(s: &Arc<CudaStream>, launches: usize, mut f: impl FnMut()) -> CudaGraph {
            for _ in 0..3 {
                f();
            }
            s.synchronize().unwrap();
            s.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .unwrap();
            for _ in 0..launches {
                f();
            }
            let g = s
                .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                .unwrap()
                .unwrap();
            g.launch().unwrap();
            s.synchronize().unwrap();
            g
        }

        /// Device microseconds per call for one replay.
        pub fn replay_us(s: &Arc<CudaStream>, g: &CudaGraph, launches: usize) -> f64 {
            let ctx = s.context();
            let a = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
            let b = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
            a.record(s).unwrap();
            g.launch().unwrap();
            b.record(s).unwrap();
            a.elapsed_ms(&b).unwrap() as f64 * 1000.0 / launches as f64
        }

        pub fn gpu_util() -> u32 {
            let out = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stdout).trim().lines().next().unwrap().trim().parse().unwrap()
        }

        pub struct Paired {
            pub a: Vec<f64>,
            pub b: Vec<f64>,
            pub ratios: Vec<f64>,
            pub attempts: usize,
        }

        pub fn median(v: &[f64]) -> f64 {
            if v.is_empty() {
                return f64::NAN;
            }
            let mut v = v.to_vec();
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        }

        impl Paired {
            pub fn conclusive(&self) -> bool {
                self.ratios.len() >= PAIRS
            }
            pub fn ratio(&self) -> f64 {
                median(&self.ratios)
            }
        }

        pub fn run(s: &Arc<CudaStream>, ga: &CudaGraph, gb: &CudaGraph, launches: usize) -> Paired {
            let mut p = Paired { a: vec![], b: vec![], ratios: vec![], attempts: 0 };
            while p.ratios.len() < PAIRS && p.attempts < ATTEMPTS {
                p.attempts += 1;
                let before = gpu_util();
                let x = replay_us(s, ga, launches);
                let y = replay_us(s, gb, launches);
                std::thread::sleep(std::time::Duration::from_millis(1000));
                let after = gpu_util();
                if before > 5 || after > 5 {
                    continue;
                }
                p.a.push(x);
                p.b.push(y);
                p.ratios.push(x / y);
            }
            p
        }

        /// Eager host microseconds per call.
        pub fn host_us(s: &Arc<CudaStream>, mut f: impl FnMut()) -> f64 {
            for _ in 0..10 {
                f();
            }
            s.synchronize().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..200 {
                f();
            }
            s.synchronize().unwrap();
            t.elapsed().as_secs_f64() * 1e6 / 200.0
        }
    }

    /// W8c-4: device time of (a) route alone, (c) the path it replaces (f32 cast, f32 GEMM,
    /// softmax_last_dim, topk, renorm) and (d) main.linear + route, E=512, k=10, bf16. Bar:
    /// (d) < (c) at every T. The (a)-vs-vLLM bar is scripts/qwen4exp_moe_bench.py's.
    #[test]
    #[ignore]
    fn bench() {
        use crate::cuda::lane::tests::{device, slice, stream, HIDDEN};
        use crate::cuda::lane::Lane;
        use crate::ops::TopKLastDimOp;
        use paired::*;
        const LAUNCHES: usize = 100;
        let dev = device();
        let s = stream(&dev);
        let main = Lane::new(s.clone()).unwrap();
        let g = load("gate", &dev);
        let wg = slice(&g["wg"]);
        let wg32 = g["wg"].to_dtype(DType::F32).unwrap();
        let c = cfg(MoeRouterScoreFunction::Softmax, 10, true, 1.0);
        let mut ok = true;
        println!(
            "{:>5} | {:>9} {:>9} {:>9} {:>6} {:>9} | {:>9} {:>9} {:>9} | bar (d) < (c)",
            "T", "(a) us", "(c) us", "(d) us", "d/c", "pairs", "(a) host", "(c) host", "(d) host"
        );
        for t in [1usize, 8, 64, 512, 4096] {
            let x_t = Tensor::randn(0f32, 1., (t, HIDDEN), &dev).unwrap().to_dtype(DType::BF16).unwrap();
            let x = slice(&x_t);
            let mut logits = s.alloc_zeros::<half::bf16>(t * 512).unwrap();
            let mut w = s.alloc_zeros::<f32>(t * 10).unwrap();
            let mut ids = s.alloc_zeros::<u32>(t * 10).unwrap();
            let (lp, _) = logits.device_ptr_mut(&s);
            let (wp, _) = w.device_ptr_mut(&s);
            let (ip, _) = ids.device_ptr_mut(&s);
            let route_only = || unsafe {
                launch(DType::BF16, lp as *const c_void, wp as *mut f32, ip as *mut u32,
                       std::ptr::null(), std::ptr::null(), t, 512, &c, &s).unwrap();
            };
            let mut logits2 = s.alloc_zeros::<half::bf16>(t * 512).unwrap();
            let (lp2, _) = logits2.device_ptr_mut(&s);
            let mut fused = || {
                main.linear(&x, &wg, &mut logits2, t, 512, HIDDEN).unwrap();
                unsafe {
                    launch(DType::BF16, lp2 as *const c_void, wp as *mut f32, ip as *mut u32,
                           std::ptr::null(), std::ptr::null(), t, 512, &c, &s).unwrap();
                }
            };
            let old = || {
                let l = x_t.to_dtype(DType::F32).unwrap().matmul(&wg32.t().unwrap()).unwrap();
                let p = hanzo_nn::ops::softmax_last_dim(&l).unwrap();
                let TopKOutput { values, indices } = p.topk(10).unwrap();
                let v = values.broadcast_div(&values.sum_keepdim(1).unwrap()).unwrap();
                drop((v, indices));
            };
            let ga = capture(&s, LAUNCHES, route_only);
            let gc = capture(&s, LAUNCHES, old);
            let gd = capture(&s, LAUNCHES, &mut fused);
            let a_us = median(&(0..20).map(|_| replay_us(&s, &ga, LAUNCHES)).collect::<Vec<_>>());
            let p = run(&s, &gd, &gc, LAUNCHES);
            let (ha, hc, hd) = (host_us(&s, route_only), host_us(&s, old), host_us(&s, &mut fused));
            let verdict = if !p.conclusive() {
                ok = false;
                "INCONCLUSIVE"
            } else if p.ratio() < 1.0 {
                "PASS"
            } else {
                ok = false;
                "FAIL"
            };
            println!(
                "{t:5} | {a_us:9.2} {:9.2} {:9.2} {:6.3} {:>4}/{:<4} | {ha:9.2} {hc:9.2} {hd:9.2} | {verdict}",
                median(&p.b), median(&p.a), p.ratio(), p.ratios.len(), p.attempts
            );
            drop(fused);
            drop((logits, logits2));
        }
        assert!(ok, "bench bar not met (numbers above)");
    }
}
