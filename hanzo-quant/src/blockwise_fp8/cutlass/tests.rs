//! GPU tests for the CUTLASS block-FP8 lib. They fail, never skip, when no
//! sm_121 device can be opened: a blocked GPU run is reported as blocked.

use super::*;

fn gb10() -> Result<(Device, CudaDevice)> {
    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = device.clone() else {
        unreachable!()
    };
    if !device_supported(&cuda) {
        hanzo_ml::bail!("these tests need an sm_121 device, got {:?}", compute_cap(&cuda)?);
    }
    Ok((device, cuda))
}

#[test]
fn tile_for_rows_matches_vllm() {
    for m in [1, 63, 64] {
        assert_eq!(Tile::for_rows(m), Tile::SwapAb, "M={m}");
    }
    for m in [65, 255, 256] {
        assert_eq!(Tile::for_rows(m), Tile::Pingpong, "M={m}");
    }
    for m in [257, 4097, 8193] {
        assert_eq!(Tile::for_rows(m), Tile::Cooperative, "M={m}");
    }
}

#[test]
fn kernels_prepare_on_sm121() -> Result<()> {
    const SMEM_OPT_IN: usize = 101_376;
    let (_, cuda) = gb10()?;
    for tile in Tile::ALL {
        let r = prepare(&cuda, tile)?;
        println!(
            "{tile:?}: cc {}.{}, {} threads, {} registers/thread, {} B shared, {} B local",
            r.major, r.minor, r.threads, r.registers_per_thread, r.shared_bytes, r.local_bytes
        );
        assert_eq!(r.threads, 384, "{tile:?}: 1 producer + 2 math warpgroups");
        assert!(r.shared_bytes <= SMEM_OPT_IN, "{tile:?}: {} B shared", r.shared_bytes);
        for (m, n, k) in [(5, 16384, 2560), (200, 2560, 6144), (4096, 1280, 2560)] {
            assert_eq!(workspace_size(&cuda, tile, m, n, k)?, 0, "{tile:?} at ({m},{n},{k})");
        }
    }
    Ok(())
}

fn bits(t: &Tensor) -> Result<Vec<u16>> {
    Ok(t.to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<bf16>()?
        .into_iter()
        .map(|v| v.to_bits())
        .collect())
}

/// Distance in bf16 steps, across zero.
fn ulps(a: u16, b: u16) -> i32 {
    let ordered = |v: u16| {
        if v & 0x8000 != 0 {
            -((v & 0x7FFF) as i32)
        } else {
            v as i32
        }
    };
    (ordered(a) - ordered(b)).abs()
}

/// T3: every fixture case at every M, on the tile vLLM picks, against the
/// bf16 bits vLLM's cutlass_scaled_mm produced. Zero differing elements.
#[test]
fn matches_vllm_bitwise() -> Result<()> {
    use crate::blockwise_fp8::fixture;
    let (dev, _) = gb10()?;
    let fx = fixture::load()?;
    let mut failures = 0usize;
    for case in &fx.cases {
        let qa = case.qa.to_device(&dev)?;
        let sa = case.sa.to_device(&dev)?;
        let qw = case.qw.to_device(&dev)?;
        let sw = case.sw.to_device(&dev)?;
        let want = bits(&case.out)?;
        for &m in &case.m {
            let got = bits(&matmul(&qa.narrow(0, 0, m)?, &sa.narrow(0, 0, m)?, &qw, &sw)?)?;
            let want = &want[..m * case.n];
            let diff: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
            if diff.is_empty() {
                continue;
            }
            failures += 1;
            let worst = diff.iter().map(|&i| ulps(got[i], want[i])).max().unwrap();
            let first: Vec<(usize, usize)> = diff.iter().take(5).map(|&i| (i / case.n, i % case.n)).collect();
            println!(
                "{} M={m} {:?}: {} of {} differ, max {worst} ulp, first {first:?}",
                case.name,
                Tile::for_rows(m),
                diff.len(),
                got.len()
            );
        }
        println!("{}: N={} K={} M={:?}: checked", case.name, case.n, case.k, case.m);
    }
    if failures > 0 {
        let meta: Vec<_> = fx.metadata.iter().filter(|(k, _)| k.as_str() != "cases").collect();
        println!("fixture metadata: {meta:?}");
    }
    assert_eq!(failures, 0, "outputs differ from vLLM's");
    Ok(())
}

/// Seeded operands: e4m3 codes over the whole finite codebook, log-uniform scales.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn codes(&mut self, len: usize) -> Vec<F8E4M3> {
        (0..len)
            .map(|_| loop {
                let b = (self.next() >> 24) as u8;
                if b & 0x7F != 0x7F {
                    break F8E4M3::from_bits(b);
                }
            })
            .collect()
    }

    fn scales(&mut self, len: usize, lo: f64, hi: f64) -> Vec<f32> {
        (0..len)
            .map(|_| {
                let u = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
                (lo.ln() + u * (hi.ln() - lo.ln())).exp() as f32
            })
            .collect()
    }
}

struct Operands {
    qa: Tensor,
    sa: Tensor,
    qw: Tensor,
    sw: Tensor,
}

/// `rows` activation rows (row 2 all zero codes) and an `[n, k]` weight, on CPU.
fn operands(rows: usize, n: usize, k: usize, seed: u64) -> Result<Operands> {
    let mut rng = Rng(seed | 1);
    let mut a = rng.codes(rows * k);
    for v in &mut a[2 * k..3 * k] {
        *v = F8E4M3::from_bits(0);
    }
    let cpu = Device::Cpu;
    Ok(Operands {
        qa: Tensor::from_vec(a, (rows, k), &cpu)?,
        sa: Tensor::from_vec(rng.scales(rows * k / BLOCK, 1e-4, 1e-1), (rows, k / BLOCK), &cpu)?,
        qw: Tensor::from_vec(rng.codes(n * k), (n, k), &cpu)?,
        sw: Tensor::from_vec(rng.scales(n / BLOCK * k / BLOCK, 1e-5, 1e-2), (n / BLOCK, k / BLOCK), &cpu)?,
    })
}

impl Operands {
    fn to(&self, dev: &Device) -> Result<Operands> {
        Ok(Operands {
            qa: self.qa.to_device(dev)?,
            sa: self.sa.to_device(dev)?,
            qw: self.qw.to_device(dev)?,
            sw: self.sw.to_device(dev)?,
        })
    }

    fn rows(&self, m: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.qa.narrow(0, 0, m)?, self.sa.narrow(0, 0, m)?))
    }
}

const T4_ROWS: [usize; 11] = [1, 3, 5, 7, 13, 64, 65, 100, 256, 257, 300];
const T4_SHAPES: [(usize, usize); 4] = [(128, 128), (384, 640), (1280, 2560), (2560, 6144)];

/// T4: all three tiles and the dispatching entry give the same bits at every
/// M, within the fp64 bound; an all-zero activation row is +0.0.
#[test]
fn tiles_agree_bitwise() -> Result<()> {
    use crate::blockwise_fp8::fixture;
    let (dev, _) = gb10()?;
    for (i, &(n, k)) in T4_SHAPES.iter().enumerate() {
        let cpu = operands(300, n, k, 0x5EED + i as u64)?;
        let reference = fixture::reference(&cpu.qa, &cpu.sa, &cpu.qw, &cpu.sw, 300)?;
        let gpu = cpu.to(&dev)?;
        for &m in &T4_ROWS {
            let (qa, sa) = gpu.rows(m)?;
            let dispatched = bits(&matmul(&qa, &sa, &gpu.qw, &gpu.sw)?)?;
            for tile in Tile::ALL {
                let got = bits(&matmul_tile(&qa, &sa, &gpu.qw, &gpu.sw, tile)?)?;
                let diff = got.iter().zip(&dispatched).filter(|(a, b)| a != b).count();
                assert_eq!(diff, 0, "N={n} K={k} M={m}: {tile:?} differs from {:?} in {diff}", Tile::for_rows(m));
            }
            for (idx, &b) in dispatched.iter().enumerate() {
                let y = bf16::from_bits(b).to_f64();
                let (y64, c) = (reference.y[idx], reference.c[idx]);
                assert!(
                    fixture::within_bound(y, y64, c),
                    "N={n} K={k} M={m} [{}, {}]: {y} vs fp64 {y64} (C {c})",
                    idx / n,
                    idx % n
                );
            }
            if m > 2 {
                assert!(dispatched[2 * n..3 * n].iter().all(|&b| b == 0), "N={n} K={k} M={m}: zero row is not +0.0");
            }
        }
        println!("N={n} K={k}: 3 tiles agree and hold the fp64 bound at M={T4_ROWS:?}");
    }
    Ok(())
}

/// T4: a row's output does not depend on how many rows ride with it.
#[test]
fn rows_are_independent() -> Result<()> {
    let (dev, _) = gb10()?;
    for (i, &(n, k)) in T4_SHAPES.iter().enumerate() {
        let gpu = operands(300, n, k, 0xA11 + i as u64)?.to(&dev)?;
        let full = bits(&matmul(&gpu.qa, &gpu.sa, &gpu.qw, &gpu.sw)?)?;
        for &m in &T4_ROWS {
            let (qa, sa) = gpu.rows(m)?;
            let part = bits(&matmul(&qa, &sa, &gpu.qw, &gpu.sw)?)?;
            assert_eq!(part, full[..m * n], "N={n} K={k}: rows at M={m} differ from M=300");
        }
    }
    Ok(())
}

/// T4: strided and offset views read the same values as packed copies.
#[test]
fn views_match_contiguous() -> Result<()> {
    let (dev, _) = gb10()?;
    let (n, k) = (384, 640);
    for m in [1, 5, 65, 300] {
        // Row views: 3 leading rows before the operand, and 256 weight rows / 2 scale rows.
        let big = operands(m + 7, n + 256, k, 0xB16 + m as u64)?.to(&dev)?;
        let qa = big.qa.narrow(0, 3, m)?;
        let sa = big.sa.narrow(0, 3, m)?;
        let qw = big.qw.narrow(0, 128, n)?;
        let sw = big.sw.narrow(0, 1, n / BLOCK)?;
        assert!(qa.layout().start_offset() > 0 && qw.layout().start_offset() > 0);
        let packed = bits(&matmul(&qa.copy()?, &sa.copy()?, &qw.copy()?, &sw.copy()?)?)?;
        let viewed = bits(&matmul(&qa, &sa, &qw, &sw)?)?;
        assert_eq!(viewed, packed, "M={m}: row-offset views");

        // Column-major activation scales.
        let sa_cm = sa.copy()?.t()?.contiguous()?.t()?;
        assert!(!sa_cm.is_contiguous() || m == 1);
        let strided = bits(&matmul(&qa.copy()?, &sa_cm, &qw.copy()?, &sw.copy()?)?)?;
        assert_eq!(strided, packed, "M={m}: column-major activation scales");
    }
    Ok(())
}

/// T4: every unsupported input is an Err naming the constraint, not a panic,
/// a CUTLASS status or a bad address; M=0 is an empty result.
#[test]
fn rejects_unsupported_inputs() -> Result<()> {
    let (dev, _) = gb10()?;
    let expect_err = |r: Result<Tensor>, needle: &str, what: &str| {
        let e = r.expect_err(what).to_string();
        assert!(e.contains(needle), "{what}: error `{e}` does not name `{needle}`");
    };
    let good = operands(8, 256, 256, 7)?.to(&dev)?;
    let (qa, sa, qw, sw) = (&good.qa, &good.sa, &good.qw, &good.sw);

    let odd_k = operands(8, 256, 384, 9)?.to(&dev)?;
    expect_err(
        matmul(&odd_k.qa.narrow(1, 0, 320)?, &odd_k.sa, &odd_k.qw.narrow(1, 0, 320)?, &odd_k.sw),
        "multiples of 128",
        "K=320",
    );
    expect_err(matmul(qa, sa, &qw.narrow(0, 0, 192)?, sw), "multiples of 128", "N=192");
    expect_err(matmul(qa, &sa.narrow(0, 0, 4)?, qw, sw), "activation scales [8, 2]", "short sa");
    expect_err(matmul(qa, sa, qw, &sw.narrow(1, 0, 1)?), "weight scales [2, 2]", "narrow sw");
    expect_err(matmul(&Tensor::zeros(qa.shape(), DType::BF16, &dev)?, sa, qw, sw), "activations as F8E4M3", "bf16 qa");
    expect_err(matmul(qa, sa, &Tensor::zeros(qw.shape(), DType::BF16, &dev)?, sw), "weights as F8E4M3", "bf16 qw");
    expect_err(matmul(qa, &Tensor::zeros(sa.shape(), DType::BF16, &dev)?, qw, sw), "activation scales as F32", "bf16 sa");
    expect_err(matmul(qa, sa, qw, &Tensor::zeros(sw.shape(), DType::BF16, &dev)?), "weight scales as F32", "bf16 sw");
    let cpu = operands(8, 256, 256, 7)?;
    expect_err(matmul(&cpu.qa, &cpu.sa, &cpu.qw, &cpu.sw), "CUDA device", "CPU tensors");
    expect_err(matmul(qa, &cpu.sa, qw, sw), "CUDA device", "CPU activation scales");

    // A contiguous view whose first element sits one byte into the buffer.
    let flat = Tensor::from_vec(Rng(3).codes(8 * 256 + 1), 8 * 256 + 1, &Device::Cpu)?.to_device(&dev)?;
    let shifted = flat.narrow(0, 1, 8 * 256)?.reshape((8, 256))?;
    assert!(shifted.is_contiguous() && shifted.layout().start_offset() == 1);
    expect_err(matmul(&shifted, sa, qw, sw), "16-byte aligned", "misaligned activations");

    let empty = matmul(&qa.narrow(0, 0, 0)?, &sa.narrow(0, 0, 0)?, qw, sw)?;
    assert_eq!(empty.dims2()?, (0, 256));
    assert_eq!(empty.dtype(), DType::BF16);
    Ok(())
}

/// Copy `src` into `dst`'s device memory, in place: the captured graph keeps
/// reading `dst`'s address.
fn overwrite(dst: &Tensor, src: &Tensor, stream: &hanzo_ml::cuda::cudarc::driver::CudaStream) -> Result<()> {
    use hanzo_ml::cuda::cudarc::driver::result;
    assert_eq!((dst.dims(), dst.dtype()), (src.dims(), src.dtype()));
    let bytes = dst.elem_count() * dst.dtype().size_in_bytes();
    let ptr = |t: &Tensor| -> Result<u64> {
        let (storage, layout) = t.storage_and_layout();
        let Storage::Cuda(s) = &*storage else { unreachable!() };
        let (p, _g) = match t.dtype() {
            DType::F8E4M3 => slice_ptr(s.as_cuda_slice::<F8E4M3>()?, layout.start_offset()),
            _ => slice_ptr(s.as_cuda_slice::<f32>()?, layout.start_offset()),
        };
        Ok(p)
    };
    unsafe { result::memcpy_dtod_async(ptr(dst)?, ptr(src)?, bytes, stream.cu_stream()) }
        .map_err(hanzo_ml::Error::wrap)
}

/// T5: three calls captured into one graph replay with fresh inputs and give
/// the eager bits; the graph holds exactly 3 kernels and no memset.
#[test]
fn capturable_in_cuda_graph() -> Result<()> {
    use hanzo_ml::cuda::cudarc::driver::sys;
    let (dev, cuda) = gb10()?;
    // As the engine's capture paths do: no cross-stream event waits inside capture.
    unsafe { cuda.disable_event_tracking() };
    let stream = cuda.cuda_stream();
    let shapes = [(5usize, 2560usize, 6144usize), (40, 16384, 2560), (300, 1280, 2560)];
    let inputs: Vec<Operands> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(m, n, k))| operands(m, n, k, 0xC0DE + i as u64)?.to(&dev))
        .collect::<Result<_>>()?;
    // Warm-up runs prepare outside capture.
    for x in &inputs {
        matmul(&x.qa, &x.sa, &x.qw, &x.sw)?;
    }
    stream.synchronize().map_err(hanzo_ml::Error::wrap)?;

    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(hanzo_ml::Error::wrap)?;
    let captured: Result<Vec<Tensor>> = inputs.iter().map(|x| matmul(&x.qa, &x.sa, &x.qw, &x.sw)).collect();
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(hanzo_ml::Error::wrap)?
        .expect("capture produced a graph");
    let outputs = captured?;

    let (mut kernels, mut memsets, mut total) = (0, 0, 0usize);
    unsafe {
        let mut count = 0usize;
        sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count).result().map_err(hanzo_ml::Error::wrap)?;
        let mut nodes = vec![std::ptr::null_mut(); count];
        sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count).result().map_err(hanzo_ml::Error::wrap)?;
        for node in nodes {
            let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            sys::cuGraphNodeGetType(node, &mut kind).result().map_err(hanzo_ml::Error::wrap)?;
            total += 1;
            match kind {
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => kernels += 1,
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => memsets += 1,
                _ => {}
            }
        }
    }
    println!("graph: {total} nodes, {kernels} kernels, {memsets} memsets");
    assert_eq!(kernels, 3, "one kernel per call");
    assert_eq!(memsets, 0, "no per-call memset");

    for round in 0..2u64 {
        for (i, (x, &(m, n, k))) in inputs.iter().zip(&shapes).enumerate() {
            let fresh = operands(m, n, k, 0xF00D + 16 * round + i as u64)?.to(&dev)?;
            overwrite(&x.qa, &fresh.qa, &stream)?;
            overwrite(&x.sa, &fresh.sa, &stream)?;
            overwrite(&x.qw, &fresh.qw, &stream)?;
            overwrite(&x.sw, &fresh.sw, &stream)?;
        }
        graph.launch().map_err(hanzo_ml::Error::wrap)?;
        stream.synchronize().map_err(hanzo_ml::Error::wrap)?;
        for (i, (x, out)) in inputs.iter().zip(&outputs).enumerate() {
            let eager = bits(&matmul(&x.qa.copy()?, &x.sa.copy()?, &x.qw.copy()?, &x.sw.copy()?)?)?;
            assert_eq!(bits(out)?, eager, "replay {round}, call {i}: graph differs from eager");
        }
    }
    Ok(())
}
