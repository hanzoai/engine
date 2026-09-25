//! A Lane: one CUDA stream with its own cuBLAS handle and torch's 32 MiB workspace.
//!
//! `Lane::linear` is torch's `F.linear` for bf16 (what vLLM's `ReplicatedLinear` gate runs):
//! `cublasGemmEx(T, N, out, rows, inp, COMPUTE_32F, DEFAULT_TENSOR_OP)` in the default math
//! mode. cuBLAS picks its kernel by M and by the workspace it may use: with torch's 32 MiB the
//! gate's logits equal vLLM's bitwise at every M; hanzo-ml's own handle carries no workspace and
//! differs at M 98-101, 126-160, 184-256 and 289-320 (N=512, GB10, cuBLAS 13.1.1).
//!
//! Each Lane owns its handle and workspace, since one workspace shared by two concurrent
//! streams is a race.

use std::sync::{Arc, Once};

use half::bf16;
use hanzo_ml::cuda_backend::cudarc::cublas::{result as blas, sys, CudaBlas};
use hanzo_ml::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use hanzo_ml::cuda_backend::WrapErr;
use hanzo_ml::Result;

/// The cuBLAS vLLM 0.29.0 ships (its wheel's nvidia-cublas 13.1.1.3). The gate's kernel choice,
/// and so its bits, follow the cuBLAS build: a different one breaks parity, not correctness.
pub const CUBLAS: (i32, i32, i32) = (13, 1, 1);

/// torch's cuBLAS workspace on sm_90 and newer.
pub const WORKSPACE: usize = 32 << 20;

pub struct Lane {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    _workspace: CudaSlice<u8>,
}

/// The loaded cuBLAS version, from `cublasGetProperty`.
pub fn cublas_version() -> Result<(i32, i32, i32)> {
    let get = |p| {
        let mut v = 0;
        unsafe { sys::cublasGetProperty(p, &mut v) }.result().w()?;
        Ok::<_, hanzo_ml::Error>(v)
    };
    Ok((
        get(sys::libraryPropertyType_t::MAJOR_VERSION)?,
        get(sys::libraryPropertyType_t::MINOR_VERSION)?,
        get(sys::libraryPropertyType_t::PATCH_LEVEL)?,
    ))
}

impl Lane {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        static SEEN: Once = Once::new();
        let mut version = Ok(CUBLAS);
        SEEN.call_once(|| {
            version = cublas_version();
            if let Ok(v) = version {
                tracing::info!("cuBLAS {}.{}.{}", v.0, v.1, v.2);
                if v != CUBLAS {
                    tracing::warn!(
                        "cuBLAS {}.{}.{} is loaded; vLLM parity is pinned to {}.{}.{}: gate logits may differ",
                        v.0, v.1, v.2, CUBLAS.0, CUBLAS.1, CUBLAS.2
                    );
                }
            }
        });
        version?;
        let blas = CudaBlas::new(stream.clone()).w()?;
        let mut workspace = stream.alloc_zeros::<u8>(WORKSPACE).w()?;
        {
            let (ptr, _g) = workspace.device_ptr_mut(&stream);
            unsafe { sys::cublasSetWorkspace_v2(*blas.handle(), ptr as *mut _, WORKSPACE) }
                .result()
                .w()?;
        }
        Ok(Self {
            stream,
            blas,
            _workspace: workspace,
        })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    /// `y[rows, out] = x[rows, inp] @ w[out, inp]^T` in bf16 with f32 accumulation, exactly as
    /// torch's `F.linear(x, w)`. `x`, `w` and `y` are contiguous and row-major.
    pub fn linear(
        &self,
        x: &impl DevicePtr<bf16>,
        w: &impl DevicePtr<bf16>,
        y: &mut impl DevicePtrMut<bf16>,
        rows: usize,
        out: usize,
        inp: usize,
    ) -> Result<()> {
        if x.len() < rows * inp || w.len() < out * inp || y.len() < rows * out {
            hanzo_ml::bail!(
                "Lane::linear: x {} w {} y {} for rows {rows} out {out} inp {inp}",
                x.len(),
                w.len(),
                y.len()
            );
        }
        let (xp, _xg) = x.device_ptr(&self.stream);
        let (wp, _wg) = w.device_ptr(&self.stream);
        let (yp, _yg) = y.device_ptr_mut(&self.stream);
        let (alpha, beta) = (1.0f32, 0.0f32);
        unsafe {
            blas::gemm_ex(
                *self.blas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                out as i32,
                rows as i32,
                inp as i32,
                &alpha as *const f32 as *const _,
                wp as *const _,
                sys::cudaDataType::CUDA_R_16BF,
                inp as i32,
                xp as *const _,
                sys::cudaDataType::CUDA_R_16BF,
                inp as i32,
                &beta as *const f32 as *const _,
                yp as *mut _,
                sys::cudaDataType::CUDA_R_16BF,
                out as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .w()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::cuda::route;
    use crate::ops::MoeRouterScoreFunction;
    use hanzo_ml::backend::BackendStorage;
    use hanzo_ml::cuda_backend::cudarc::driver::sys::{
        CUgraphInstantiate_flags, CUstreamCaptureMode,
    };
    use hanzo_ml::cuda_backend::CudaStorageSlice;
    use hanzo_ml::{DType, Device, Storage, Tensor};
    use std::collections::HashMap;

    pub(crate) const HIDDEN: usize = 2560;

    /// A CUDA device with cudarc event tracking off, as graph capture and Branch need.
    pub(crate) fn device() -> Device {
        let dev = Device::new_cuda(0).unwrap();
        if let Device::Cuda(d) = &dev {
            unsafe { d.cuda_stream().context().disable_event_tracking() };
        }
        dev
    }

    pub(crate) fn stream(dev: &Device) -> Arc<CudaStream> {
        match dev {
            Device::Cuda(d) => d.cuda_stream(),
            _ => unreachable!(),
        }
    }

    /// The bf16 device slice under a contiguous CUDA tensor.
    pub(crate) fn slice(t: &Tensor) -> CudaSlice<bf16> {
        let t = t.contiguous().unwrap();
        let (s, l) = t.storage_and_layout();
        assert_eq!(l.start_offset(), 0);
        let Storage::Cuda(s) = &*s else { unreachable!() };
        let CudaStorageSlice::BF16(s) = &s.slice else {
            panic!("not bf16")
        };
        s.try_clone().unwrap()
    }

    pub(crate) fn host(dev_stream: &Arc<CudaStream>, s: &CudaSlice<bf16>) -> Vec<u16> {
        dev_stream
            .clone_dtoh(s)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    pub(crate) fn tensor_bits(t: &Tensor) -> Vec<u16> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<bf16>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    pub(crate) fn gate_meta() -> HashMap<String, String> {
        let bytes = std::fs::read(format!("{}/gate.safetensors", route::tests::FIXTURES)).unwrap();
        let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
        meta.metadata().clone().unwrap()
    }

    pub(crate) fn gate_ms() -> Vec<usize> {
        serde_json::from_str(&gate_meta()["m"]).unwrap()
    }

    fn run(lane: &Lane, x: &CudaSlice<bf16>, w: &CudaSlice<bf16>, m: usize, n: usize) -> Vec<u16> {
        let mut y = lane.stream().alloc_zeros::<bf16>(m * n).unwrap();
        lane.linear(&x.slice(..m * HIDDEN), w, &mut y, m, n, HIDDEN).unwrap();
        lane.stream().synchronize().unwrap();
        host(lane.stream(), &y)
    }

    fn captured(lane: &Lane, x: &CudaSlice<bf16>, w: &CudaSlice<bf16>, m: usize, n: usize) -> Vec<u16> {
        let s = lane.stream().clone();
        let mut y = s.alloc_zeros::<bf16>(m * n).unwrap();
        s.synchronize().unwrap();
        s.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        lane.linear(&x.slice(..m * HIDDEN), w, &mut y, m, n, HIDDEN).unwrap();
        let g = s
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .unwrap()
            .unwrap();
        g.launch().unwrap();
        s.synchronize().unwrap();
        host(&s, &y)
    }

    /// Both gates, every fixture M, three ways: the main Lane eager, the main Lane captured and
    /// replayed, and a second Lane on a new stream. All equal vLLM's F.linear bitwise.
    #[test]
    fn gate_matches_vllm() {
        let dev = device();
        let main = Lane::new(stream(&dev)).unwrap();
        let side = Lane::new(stream(&dev).context().new_stream().unwrap()).unwrap();
        let g = route::tests::load("gate", &dev);
        let x = slice(&g["x"]);
        for (gate, n) in [("router", 512usize), ("shared", 1)] {
            let w = slice(&g[if n == 1 { "ws" } else { "wg" }]);
            for m in gate_ms() {
                let want = tensor_bits(&g[&format!("{gate}.{m}")]);
                for (how, got) in [
                    ("main", run(&main, &x, &w, m, n)),
                    ("captured", captured(&main, &x, &w, m, n)),
                    ("side", run(&side, &x, &w, m, n)),
                ] {
                    let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                    assert_eq!(bad, 0, "{gate} gate M={m} {how}: {bad} logits differ from vLLM");
                }
            }
        }
        println!("gate_matches_vllm: M {:?}, both gates, main/captured/side: 0 differ", gate_ms());
    }

    /// Lane::linear then route::topk equals vLLM's F.linear then topk_softmax, bitwise.
    #[test]
    fn gate_routes_like_vllm() {
        let dev = device();
        let main = Lane::new(stream(&dev)).unwrap();
        let g = route::tests::load("gate", &dev);
        let x = slice(&g["x"]);
        let w = slice(&g["wg"]);
        for m in gate_ms() {
            let mut y = main.stream().alloc_zeros::<bf16>(m * 512).unwrap();
            main.linear(&x.slice(..m * HIDDEN), &w, &mut y, m, 512, HIDDEN).unwrap();
            let logits = Tensor::from((
                Storage::Cuda(hanzo_ml::cuda_backend::CudaStorage {
                    slice: CudaStorageSlice::BF16(y),
                    device: match &dev {
                        Device::Cuda(d) => d.clone(),
                        _ => unreachable!(),
                    },
                }),
                hanzo_ml::Shape::from_dims(&[m, 512]),
            ));
            let out = route::topk(
                &logits,
                route::tests::cfg(MoeRouterScoreFunction::Softmax, 10, true, 1.0),
                None,
                None,
            )
            .unwrap();
            assert_eq!(
                route::tests::ids(&out.indices),
                route::tests::ids(&g[&format!("i.{m}")]),
                "ids at M={m}"
            );
            assert_eq!(
                route::tests::bits(&out.values),
                route::tests::bits(&g[&format!("w.{m}")]),
                "weights at M={m}"
            );
        }
    }

    fn padded(t: usize, sizes: &[usize]) -> usize {
        *sizes.iter().find(|&&s| s >= t).unwrap_or(&t)
    }

    fn runs(ms: &[usize]) -> String {
        let mut out = Vec::new();
        let mut i = 0;
        while i < ms.len() {
            let mut j = i;
            while j + 1 < ms.len() && ms[j + 1] == ms[j] + 1 {
                j += 1;
            }
            out.push(if i == j {
                format!("{}", ms[i])
            } else {
                format!("{}-{}", ms[i], ms[j])
            });
            i = j + 1;
        }
        out.join(",")
    }

    /// vLLM pads a batch of T up to its capture size P; hanzo's gate runs at T. The first T rows
    /// must not depend on which of the two M cuBLAS saw, under both capture lists the parity
    /// configs use. Prints the class map over M=1..512 and vLLM's 512-grid crossings.
    #[test]
    fn pad_classes() {
        let dev = device();
        let main = Lane::new(stream(&dev)).unwrap();
        let g = route::tests::load("gate", &dev);
        let x = slice(&g["x"]);
        let meta = gate_meta();
        for (gate, n) in [("router", 512usize), ("shared", 1)] {
            let w = slice(&g[if n == 1 { "ws" } else { "wg" }]);
            let out: Vec<Vec<u16>> = (0..=512).map(|m| if m == 0 { vec![] } else { run(&main, &x, &w, m, n) }).collect();
            for list in ["capture_nospec", "capture_production"] {
                let sizes: Vec<usize> = serde_json::from_str(&meta[list]).unwrap();
                for t in 1..=*sizes.last().unwrap() {
                    let p = padded(t, &sizes);
                    assert_eq!(
                        out[t][..t * n],
                        out[p][..t * n],
                        "{gate}: rows at T={t} differ from rows at vLLM's padded M={p} ({list})"
                    );
                }
            }
            // Classes: M whose shared rows are bitwise equal.
            let mut reps: Vec<(usize, Vec<usize>)> = Vec::new();
            for m in 1..=512 {
                match reps.iter_mut().find(|(r, _)| {
                    let k = (*r).min(m) * n;
                    out[*r][..k] == out[m][..k]
                }) {
                    Some((_, mem)) => mem.push(m),
                    None => reps.push((m, vec![m])),
                }
            }
            let class = |m: usize| reps.iter().position(|(_, mem)| mem.contains(&m)).unwrap();
            let grid: Vec<usize> = [1, 2, 4]
                .into_iter()
                .chain((8..256).step_by(8))
                .chain((256..=512).step_by(16))
                .collect();
            let cross: Vec<usize> = (1..=512)
                .filter(|&t| class(t) != class(padded(t, &grid)))
                .collect();
            println!(
                "{gate} gate (N={n}): {} classes: {}; vLLM 512-grid crossings: T={}",
                reps.len(),
                reps.iter().map(|(_, m)| runs(m)).collect::<Vec<_>>().join(" | "),
                if cross.is_empty() { "none".into() } else { runs(&cross) }
            );
        }
    }

    /// The loaded cuBLAS, the one the goldens were cut with, and the pin all agree.
    #[test]
    fn cublas_version() {
        let loaded = super::cublas_version().unwrap();
        let fixture: Vec<i32> = serde_json::from_str(&gate_meta()["cublas"]).unwrap();
        let fixture = (fixture[0], fixture[1], fixture[2]);
        assert!(
            loaded == fixture && fixture == CUBLAS,
            "cuBLAS loaded {loaded:?}, fixture {fixture:?}, pinned {CUBLAS:?}"
        );
    }

    /// Each rejected way to compute the gate fails gate_matches_vllm's comparison, and where.
    #[test]
    #[ignore]
    fn gate_mutations() {
        let dev = device();
        let s = stream(&dev);
        let g = route::tests::load("gate", &dev);
        let x = slice(&g["x"]);
        let w = slice(&g["wg"]);
        let bare = CudaBlas::new(s.clone()).unwrap();
        let mut fails: HashMap<&str, Vec<usize>> = HashMap::new();
        for m in gate_ms() {
            let want = tensor_bits(&g[&format!("router.{m}")]);
            // No user workspace: hanzo-ml's own handle shape.
            let mut y = s.alloc_zeros::<bf16>(m * 512).unwrap();
            {
                let (xp, _a) = x.device_ptr(&s);
                let (wp, _b) = w.device_ptr(&s);
                let (yp, _c) = y.device_ptr_mut(&s);
                let (alpha, beta) = (1.0f32, 0.0f32);
                unsafe {
                    blas::gemm_ex(
                        *bare.handle(),
                        sys::cublasOperation_t::CUBLAS_OP_T,
                        sys::cublasOperation_t::CUBLAS_OP_N,
                        512,
                        m as i32,
                        HIDDEN as i32,
                        &alpha as *const f32 as *const _,
                        wp as *const _,
                        sys::cudaDataType::CUDA_R_16BF,
                        HIDDEN as i32,
                        xp as *const _,
                        sys::cudaDataType::CUDA_R_16BF,
                        HIDDEN as i32,
                        &beta as *const f32 as *const _,
                        yp as *mut _,
                        sys::cudaDataType::CUDA_R_16BF,
                        512,
                        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                }
                .unwrap();
            }
            s.synchronize().unwrap();
            if host(&s, &y) != want {
                fails.entry("no workspace").or_default().push(m);
            }
            let xm = g["x"].narrow(0, 0, m).unwrap();
            // hanzo-quant's custom GEMV (UnquantLinear's path at batch <= 8).
            if m <= 8 && tensor_bits(&hanzo_quant::gemv(&xm, &g["wg"], None).unwrap()) != want {
                fails.entry("hanzo_quant::gemv").or_default().push(m);
            }
            // An f32 GEMM, then one cast to bf16.
            let f = xm
                .to_dtype(DType::F32)
                .unwrap()
                .matmul(&g["wg"].to_dtype(DType::F32).unwrap().t().unwrap())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            if tensor_bits(&f) != want {
                fails.entry("f32 gemm + cast").or_default().push(m);
            }
        }
        let mut keys: Vec<_> = fails.keys().copied().collect();
        keys.sort();
        for k in keys {
            println!("mutation '{k}' differs from vLLM at M {:?}", fails[k]);
        }
        assert!(fails.contains_key("no workspace"));
        assert!(fails.contains_key("f32 gemm + cast"));
    }

    /// W8c-11, window only: vLLM's dumped gate inputs (scripts/qwen4exp_moe_dump.py) through
    /// Lane::linear and route::topk at the same M. Bar: logits, ids and weights bitwise on 100%
    /// of (token, layer) pairs at every concurrency dumped.
    #[test]
    #[ignore]
    fn teacher_forced() {
        const SNAP: &str = "/home/z/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47-fp8hybrid";
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../target/qwen4exp_moe_window");
        let dev = device();
        let main = Lane::new(stream(&dev)).unwrap();
        let shards: Vec<_> = std::fs::read_dir(SNAP)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        let ckpt = unsafe { hanzo_ml::safetensors::MmapedSafetensors::multi(&shards) }.unwrap();
        let mut weights: HashMap<usize, CudaSlice<bf16>> = HashMap::new();
        let mut total = (0usize, 0usize, 0usize, 0usize);
        for c in [1, 3, 4] {
            let path = format!("{dir}/dump_c{c}.safetensors");
            let d = hanzo_ml::safetensors::load(&path, &dev).unwrap();
            let (mut pairs, mut logit_ok, mut id_ok, mut w_ok) = (0usize, 0usize, 0usize, 0usize);
            for key in d.keys().filter(|k| k.starts_with("x.")) {
                let (n, layer) = key[2..].split_once('.').unwrap();
                let l: usize = layer.parse().unwrap();
                let w = weights.entry(l).or_insert_with(|| {
                    slice(&ckpt.load(&format!("model.language_model.layers.{l}.mlp.gate.weight"), &dev).unwrap())
                });
                let x = &d[key];
                let m = x.dims()[0];
                let mut y = main.stream().alloc_zeros::<bf16>(m * 512).unwrap();
                main.linear(&slice(x), w, &mut y, m, 512, HIDDEN).unwrap();
                let got = host(main.stream(), &y);
                let want = tensor_bits(&d[&format!("logits.{n}.{l}")]);
                let logits = Tensor::from_vec(
                    got.iter().map(|b| bf16::from_bits(*b)).collect::<Vec<_>>(),
                    (m, 512),
                    &dev,
                )
                .unwrap();
                let out = route::topk(
                    &logits,
                    route::tests::cfg(MoeRouterScoreFunction::Softmax, 10, true, 1.0),
                    None,
                    None,
                )
                .unwrap();
                let (ids, wts) = (route::tests::ids(&out.indices), route::tests::bits(&out.values));
                let (vids, vwts) = (
                    route::tests::ids(&d[&format!("i.{n}.{l}")]),
                    route::tests::bits(&d[&format!("w.{n}.{l}")]),
                );
                for t in 0..m {
                    pairs += 1;
                    logit_ok += (got[t * 512..(t + 1) * 512] == want[t * 512..(t + 1) * 512]) as usize;
                    id_ok += (ids[t * 10..(t + 1) * 10] == vids[t * 10..(t + 1) * 10]) as usize;
                    w_ok += (wts[t * 10..(t + 1) * 10] == vwts[t * 10..(t + 1) * 10]) as usize;
                }
            }
            println!("c={c}: {pairs} (token, layer) pairs; logits {logit_ok}, ids {id_ok}, weights {w_ok} bitwise");
            total = (total.0 + pairs, total.1 + logit_ok, total.2 + id_ok, total.3 + w_ok);
        }
        assert!(total.0 > 0, "no dumps under {dir}");
        assert!(
            total.1 == total.0 && total.2 == total.0 && total.3 == total.0,
            "teacher-forced routing not bitwise on every pair: {total:?}"
        );
    }
}
