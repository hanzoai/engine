//! Kernel tests against CPU references and vLLM's own outputs (`tests/fixtures/moe_*`).
//!
//! GPU runs follow the box's memory rule: the first CUDA action creates the device and
//! allocates the test's declared peak; any CUDA out-of-memory there panics with
//! BLOCKED-ON-MEMORY and a memory snapshot, which is a blocked run, never a kernel failure.

use half::bf16;
use hanzo_ml::{DType, Device, Result, Tensor};

use super::*;

/// Largest device allocation any test here makes, with headroom.
const PEAK: usize = 1 << 30;

fn meminfo() -> String {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let field = |name: &str| {
        text.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .unwrap_or("?")
            .to_string()
    };
    format!(
        "MemAvailable={} kB SwapFree={} kB",
        field("MemAvailable:"),
        field("SwapFree:")
    )
}

fn blocked(what: &str, e: impl std::fmt::Display) -> ! {
    panic!("BLOCKED-ON-MEMORY: {what}: {e} ({})", meminfo())
}

/// The test device, with the declared peak proven allocatable.
pub(super) fn device() -> Device {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            let s = e.to_string();
            if s.contains("OUT_OF_MEMORY") || s.contains("out of memory") {
                blocked("context", s)
            }
            panic!("no CUDA device: {s}")
        }
    };
    match Tensor::zeros(PEAK, DType::U8, &dev) {
        Ok(t) => drop(t),
        Err(e) => blocked("peak allocation", e),
    }
    dev
}

/// xorshift64*: reproducible inputs without a dependency.
pub(super) struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn normal(&mut self) -> f32 {
        let u = self.unit().max(1e-7);
        let v = self.unit();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }
}

// ------------------------------------------------------------------------------------------
// route
// ------------------------------------------------------------------------------------------

/// k distinct experts per token.
fn distinct(rng: &mut Rng, m: usize, k: usize, e: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(m * k);
    for _ in 0..m {
        let mut row: Vec<u32> = Vec::with_capacity(k);
        while row.len() < k {
            let x = rng.below(e) as u32;
            if !row.contains(&x) {
                row.push(x);
            }
        }
        ids.extend(row);
    }
    ids
}

struct RouteRef {
    offsets: Vec<i32>,
    src: Vec<i32>,
    dst: Vec<i32>,
    group: Vec<i32>,
    active: Vec<i32>,
    nactive: i32,
}

fn route_ref(ids: &[u32], e: usize) -> RouteRef {
    let r = ids.len();
    let mut order: Vec<usize> = (0..r).collect();
    order.sort_by_key(|&f| (ids[f], f));
    let mut counts = vec![0i32; e];
    for &x in ids {
        counts[x as usize] += 1;
    }
    let mut offsets = vec![0i32; e + 1];
    for i in 0..e {
        offsets[i + 1] = offsets[i] + counts[i];
    }
    let src: Vec<i32> = order.iter().map(|&f| f as i32).collect();
    let mut dst = vec![0i32; r];
    for (p, &f) in order.iter().enumerate() {
        dst[f] = p as i32;
    }
    let mut group = vec![-1i32; e];
    let mut active = vec![-1i32; r.min(e)];
    let mut n = 0;
    for i in 0..e {
        if counts[i] > 0 {
            group[i] = n;
            active[n as usize] = i as i32;
            n += 1;
        }
    }
    RouteRef {
        offsets,
        src,
        dst,
        group,
        active,
        nactive: n,
    }
}

fn v(t: &Tensor) -> Vec<i32> {
    t.to_vec1::<i32>().unwrap()
}

fn check_route(dev: &Device, ids: &[u32], m: usize, k: usize, e: usize, what: &str) -> Result<()> {
    let t = Tensor::from_slice(ids, (m, k), dev)?;
    let want = route_ref(ids, e);
    let mut first: Option<Vec<Vec<i32>>> = None;
    for run in 0..3 {
        let got = route(&t, e)?;
        let all = vec![
            v(&got.offsets),
            v(&got.src),
            v(&got.dst),
            v(&got.group),
            v(&got.active),
            v(&got.nactive),
        ];
        assert_eq!(all[0], want.offsets, "{what}: offsets");
        assert_eq!(all[1], want.src, "{what}: src");
        assert_eq!(all[2], want.dst, "{what}: dst");
        assert_eq!(all[3], want.group, "{what}: group");
        assert_eq!(all[4], want.active, "{what}: active");
        assert_eq!(all[5], vec![want.nactive], "{what}: nactive");
        let r = m * k;
        let expect = if r <= 4096 { 1 } else { 3 };
        assert_eq!(got.launches, expect, "{what}: launches at R={r}");
        match &first {
            None => first = Some(all),
            Some(f) => assert_eq!(f, &all, "{what}: run {run} differs"),
        }
    }
    Ok(())
}

#[test]
fn route_matches_stable_sort() -> Result<()> {
    let dev = device();
    let mut rng = Rng::new(7);
    for &m in &[1usize, 2, 7, 64, 409, 410, 4096, 8192] {
        let (k, e) = (10, 512);
        let uniform = distinct(&mut rng, m, k, e);
        check_route(&dev, &uniform, m, k, e, &format!("uniform M={m}"))?;
        // Every token on the same ten experts.
        let hot: Vec<u32> = (0..m * k).map(|f| (37 + 41 * (f % k)) as u32 % e as u32).collect();
        check_route(&dev, &hot, m, k, e, &format!("hot M={m}"))?;
        // Round robin: every expert once M*k >= E.
        let rr: Vec<u32> = (0..m * k).map(|f| (f % e) as u32).collect();
        check_route(&dev, &rr, m, k, e, &format!("round-robin M={m}"))?;
    }
    for &m in &[1usize, 5, 3000] {
        let ids = distinct(&mut rng, m, 2, 4);
        check_route(&dev, &ids, m, 2, 4, &format!("E=4 k=2 M={m}"))?;
    }
    // The launch count flips exactly past one chunk.
    let a = distinct(&mut rng, 4096, 1, 512);
    check_route(&dev, &a, 4096, 1, 512, "R=4096")?;
    let b = distinct(&mut rng, 410, 10, 512);
    check_route(&dev, &b, 410, 10, 512, "R=4100")?;
    println!("route: all cases exact ({})", meminfo());
    Ok(())
}

// ------------------------------------------------------------------------------------------
// combine
// ------------------------------------------------------------------------------------------

fn flush(x: f32) -> f32 {
    if x.is_subnormal() {
        0.0f32.copysign(x)
    } else {
        x
    }
}

/// FFMA.FTZ: subnormal inputs and result flush to signed zero.
fn fma_ftz(a: f32, b: f32, c: f32) -> f32 {
    flush(flush(a).mul_add(flush(b), flush(c)))
}

fn combine_ref(
    y: &[bf16],
    dst: &[i32],
    w: &[f32],
    addend: Option<&[bf16]>,
    m: usize,
    k: usize,
    h: usize,
    rule: Rule,
) -> Vec<u16> {
    let mut out = Vec::with_capacity(m * h);
    for t in 0..m {
        for c in 0..h {
            let mut s = 0.0f32;
            for j in 0..k {
                let yv = y[dst[t * k + j] as usize * h + c].to_f32();
                s = match rule {
                    Rule::Finalize => fma_ftz(w[t * k + j], yv, s),
                    Rule::Sum => s + yv,
                };
            }
            let mut o = bf16::from_f32(s);
            if let Some(a) = addend {
                o = bf16::from_f32(o.to_f32() + a[t * h + c].to_f32());
            }
            out.push(o.to_bits());
        }
    }
    out
}

fn special(rng: &mut Rng) -> f32 {
    match rng.below(12) {
        0 => 0.0,
        1 => -0.0,
        2 => f32::from_bits(0x0004_0000) * if rng.below(2) == 0 { 1.0 } else { -1.0 }, // subnormal
        3 => 1.0e30 * if rng.below(2) == 0 { 1.0 } else { -1.0 },
        4 => 1.2e-38,
        _ => rng.normal() * 4.0,
    }
}

#[test]
fn combine_rules_are_exact() -> Result<()> {
    let dev = device();
    let mut rng = Rng::new(11);
    for &m in &[1usize, 5, 4096] {
        for &h in &[128usize, 2560] {
            for &k in &[2usize, 10] {
                if m == 4096 && h == 2560 && k == 10 {
                    continue; // 210 MB of y; the other 4096 cases cover the grid stride
                }
                let r = m * k;
                let y: Vec<bf16> = (0..r * h).map(|_| bf16::from_f32(special(&mut rng))).collect();
                // A random permutation for dst.
                let mut dst: Vec<i32> = (0..r as i32).collect();
                for i in (1..r).rev() {
                    dst.swap(i, rng.below(i + 1));
                }
                let w: Vec<f32> = (0..r)
                    .map(|i| match i % 7 {
                        0 => 0.0,
                        1 => 1.0,
                        _ => rng.unit(),
                    })
                    .collect();
                let addend: Vec<bf16> =
                    (0..m * h).map(|_| bf16::from_f32(rng.normal())).collect();
                let yt = Tensor::from_slice(&y, (r, h), &dev)?;
                let dt = Tensor::from_slice(&dst, r, &dev)?;
                let wt = Tensor::from_slice(&w, (m, k), &dev)?;
                let at = Tensor::from_slice(&addend, (m, h), &dev)?;
                for rule in [Rule::Finalize, Rule::Sum] {
                    for with in [false, true] {
                        let got = combine(&yt, &dt, &wt, with.then_some(&at), rule)?
                            .flatten_all()?
                            .to_vec1::<bf16>()?;
                        let got: Vec<u16> = got.iter().map(|x| x.to_bits()).collect();
                        let want = combine_ref(
                            &y,
                            &dst,
                            &w,
                            with.then_some(&addend[..]),
                            m,
                            k,
                            h,
                            rule,
                        );
                        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                        assert_eq!(
                            bad, 0,
                            "combine {rule:?} addend={with} M={m} H={h} k={k}: {bad} of {} differ",
                            want.len()
                        );
                    }
                }
            }
        }
    }
    // An all -0 row gives +0 under both rules.
    let y = Tensor::from_slice(&[bf16::from_f32(-0.0); 16], (2, 8), &dev)?;
    let dst = Tensor::from_slice(&[0i32, 1], 2, &dev)?;
    let w = Tensor::from_slice(&[1.0f32, 1.0], (1, 2), &dev)?;
    for rule in [Rule::Finalize, Rule::Sum] {
        let out = combine(&y, &dst, &w, None, rule)?.flatten_all()?.to_vec1::<bf16>()?;
        assert!(out.iter().all(|x| x.to_bits() == 0), "{rule:?}: -0 row");
    }
    println!("combine: all cases exact ({})", meminfo());
    Ok(())
}

// ------------------------------------------------------------------------------------------
// shared helpers for the fixture-driven tests
// ------------------------------------------------------------------------------------------

use std::collections::HashMap;

use hanzo_ml::cuda::cudarc::driver::result as cu;

pub(super) fn fixture(dev: &Device, name: &str) -> HashMap<String, Tensor> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    hanzo_ml::safetensors::load(&path, dev).unwrap_or_else(|e| panic!("fixture {path}: {e}"))
}

fn cuda_dev(dev: &Device) -> hanzo_ml::CudaDevice {
    match dev {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    }
}

/// Reads `n` values of `T` at byte `off` of a device buffer.
pub(super) fn read<T: Copy + Default>(dev: &Device, buf: &Tensor, off: usize, n: usize) -> Vec<T> {
    let d = cuda_dev(dev);
    let stream = d.cuda_stream();
    let base = super::ffi::ptr(buf).unwrap();
    let mut out = vec![T::default(); n];
    unsafe { cu::memcpy_dtoh_async(&mut out, base + off as u64, stream.cu_stream()) }.unwrap();
    stream.synchronize().unwrap();
    out
}

/// Writes `data` at byte `off` of a device buffer.
pub(super) fn write<T: Copy>(dev: &Device, buf: &Tensor, off: usize, data: &[T]) {
    let d = cuda_dev(dev);
    let stream = d.cuda_stream();
    let base = super::ffi::ptr(buf).unwrap();
    unsafe { cu::memcpy_htod_async(base + off as u64, data, stream.cu_stream()) }.unwrap();
    stream.synchronize().unwrap();
}

pub(super) fn fill(dev: &Device, buf: &Tensor, off: usize, bytes: usize, byte: u8) {
    let d = cuda_dev(dev);
    let stream = d.cuda_stream();
    let base = super::ffi::ptr(buf).unwrap();
    unsafe { cu::memset_d8_async(base + off as u64, byte, bytes, stream.cu_stream()) }.unwrap();
    stream.synchronize().unwrap();
}

pub(super) fn host<T: hanzo_ml::WithDType>(t: &Tensor) -> Vec<T> {
    t.flatten_all().unwrap().to_vec1::<T>().unwrap()
}

pub(super) fn bytes(t: &Tensor) -> Vec<u8> {
    match t.dtype() {
        DType::U8 => host::<u8>(t),
        DType::F8E4M3 => host::<float8::F8E4M3>(t).iter().map(|v| v.to_bits()).collect(),
        d => panic!("no byte view of {d:?}"),
    }
}

pub(super) fn bf16s(t: &Tensor) -> Vec<f32> {
    host::<bf16>(&t.to_dtype(DType::BF16).unwrap())
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

/// Distance of bf16 results from a reference, M1's metric: |got - want| in bf16 ulps at
/// |want|, with a floor of 2^-10 * rms(want) per row.
#[derive(Debug, Clone, Copy)]
pub(super) struct Gap {
    pub bit_exact: f64,
    pub max_ulp: f64,
    pub p999_ulp: f64,
    pub min_cos: f64,
    pub err: f64,
}

pub(super) fn gap(got: &[f32], want: &[f32], cols: usize) -> Gap {
    assert_eq!(got.len(), want.len());
    let mut ulps = Vec::with_capacity(got.len());
    let mut exact = 0usize;
    let mut min_cos = 1.0f64;
    let mut err = 0.0f64;
    for (g, w) in got.chunks(cols).zip(want.chunks(cols)) {
        let rms = (w.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / cols as f64).sqrt();
        let (mut dot, mut ng, mut nw) = (0.0f64, 0.0f64, 0.0f64);
        for (a, b) in g.iter().zip(w) {
            if a.to_bits() == b.to_bits() {
                exact += 1;
            }
            let e = (b.abs().max(1e-30) as f64).log2().floor();
            let ulp = 2f64.powf(e - 7.0) + rms * 2f64.powi(-10);
            ulps.push(((*a as f64) - (*b as f64)).abs() / ulp);
            dot += *a as f64 * *b as f64;
            ng += (*a as f64).powi(2);
            nw += (*b as f64).powi(2);
            err += ((*a as f64) - (*b as f64)).powi(2);
        }
        let cos = if ng == 0.0 && nw == 0.0 { 1.0 } else { dot / (ng.sqrt() * nw.sqrt()) };
        min_cos = min_cos.min(cos);
    }
    ulps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ulps.len();
    let p999 = ulps[((n as f64 - 1.0) * 0.999).round() as usize];
    Gap {
        bit_exact: exact as f64 / n as f64,
        max_ulp: ulps[n - 1],
        p999_ulp: p999,
        min_cos,
        err: err.sqrt(),
    }
}

pub(super) fn norm(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((*x as f64) - (*y as f64)).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Places the stored experts at their global ids in an `e`-expert stack; the rest are `pad`.
pub(super) fn scatter_experts(t: &Tensor, experts: &[i32], e: usize, pad: f32) -> Tensor {
    let dims = t.dims().to_vec();
    let per: usize = dims[1..].iter().product();
    let mut shape = dims.clone();
    shape[0] = e;
    match t.dtype() {
        DType::U8 | DType::F8E4M3 => {
            let src = bytes(t);
            let mut out = vec![pad as u8; e * per];
            for (n, &x) in experts.iter().enumerate() {
                out[x as usize * per..(x as usize + 1) * per]
                    .copy_from_slice(&src[n * per..(n + 1) * per]);
            }
            Tensor::from_vec(out, shape, t.device()).unwrap()
        }
        DType::F32 => {
            let src = host::<f32>(t);
            let mut out = vec![pad; e * per];
            for (n, &x) in experts.iter().enumerate() {
                out[x as usize * per..(x as usize + 1) * per]
                    .copy_from_slice(&src[n * per..(n + 1) * per]);
            }
            Tensor::from_vec(out, shape, t.device()).unwrap()
        }
        d => panic!("scatter of {d:?}"),
    }
}

// ------------------------------------------------------------------------------------------
// NVFP4 experts against FlashInfer (tests/fixtures/moe_nvfp4.safetensors)
// ------------------------------------------------------------------------------------------

mod nvfp4_cases {
    use super::*;
    use crate::experts::nvfp4::{sf_base_row, sf_offset, Experts, Stage};
    use crate::experts::{Layout, Tile};

    pub struct Case {
        pub tag: &'static str,
        pub ex: Experts,
        pub x: Tensor,
        pub ids: Tensor,
        pub w: Tensor,
        pub fx: HashMap<String, Tensor>,
        pub m: usize,
        pub k: usize,
        pub h: usize,
        pub i: usize,
    }

    impl Case {
        pub fn get(&self, name: &str) -> &Tensor {
            self.fx
                .get(&format!("{}.{name}", self.tag))
                .unwrap_or_else(|| panic!("fixture has no {}.{name}", self.tag))
        }
    }

    /// `r`: layer 3's three busiest experts, M=128, k=2. `s`: E=512, k=10, H=I=128, M=4.
    pub fn load(dev: &Device, tag: &'static str) -> Case {
        let fx = fixture(dev, "moe_nvfp4.safetensors");
        let g = |n: &str| fx[&format!("{tag}.{n}")].clone();
        let (ex, e) = if tag == "s" {
            let experts = host::<i32>(&g("experts"));
            let e = 512;
            let sc = |n: &str, pad: f32| scatter_experts(&g(n), &experts, e, pad);
            let ex = Experts::new(
                &sc("w13", 0.0),
                &sc("w13_scale", 0.0),
                &sc("w13_global", 0.01),
                &sc("w13_input", 0.05),
                &sc("w2", 0.0),
                &sc("w2_scale", 0.0),
                &sc("w2_global", 0.01),
                &sc("w2_input", 0.02),
            )
            .unwrap();
            (ex, e)
        } else {
            let ex = Experts::new(
                &g("w13"),
                &g("w13_scale"),
                &g("w13_global"),
                &g("w13_input"),
                &g("w2"),
                &g("w2_scale"),
                &g("w2_global"),
                &g("w2_input"),
            )
            .unwrap();
            (ex, 3)
        };
        assert_eq!(ex.experts(), e);
        let x = g("x");
        let ids = g("ids");
        let w = g("weights");
        let (m, k) = ids.dims2().unwrap();
        let (h, i) = (ex.hidden(), ex.inter());
        Case { tag, ex, x, ids, w, fx, m, k, h, i }
    }

    pub struct Run {
        pub ws: Tensor,
        pub out: Tensor,
        pub layout: Layout,
        pub dst: Vec<i32>,
        pub offsets: Vec<i32>,
        pub group: Vec<i32>,
        pub bound: crate::experts::nvfp4::Bound,
    }

    /// A workspace filled with 0xFF, bound, with route and expand run.
    pub fn expand(dev: &Device, c: &mut Case, tile: Tile) -> Run {
        c.ex.set_tile(Some(tile));
        let layout = c.ex.layout(c.m, c.k).unwrap();
        let ws = Tensor::zeros(layout.total, DType::U8, dev).unwrap();
        fill(dev, &ws, 0, layout.total, 0xFF);
        let out = Tensor::zeros((c.m, c.h), DType::BF16, dev).unwrap();
        let bound = c.ex.bind(&c.x, &c.ids, &c.w, None, &ws, &out).unwrap();
        c.ex.exec(&bound, Stage::Route, Stage::Expand).unwrap();
        let r = c.m * c.k;
        let e = c.ex.experts();
        Run {
            dst: read::<i32>(dev, &ws, layout.dst, r),
            offsets: read::<i32>(dev, &ws, layout.offsets, e + 1),
            group: read::<i32>(dev, &ws, layout.group, e),
            ws,
            out,
            layout,
            bound,
        }
    }

    pub fn ids(c: &Case) -> Vec<usize> {
        host::<i32>(&c.ids).iter().map(|&v| v as usize).collect()
    }

    /// (group base byte, row within group) of sorted position `pos` of expert `e`.
    pub fn sf_at(run: &Run, e: usize, pos: usize, cols: usize) -> (usize, usize) {
        let off = run.offsets[e] as usize;
        let base = sf_base_row(off, run.group[e] as usize) * cols;
        (base, pos - off)
    }

    pub fn sf_byte(base: usize, row: usize, col: usize, cols: usize) -> usize {
        base + sf_offset(row, col, cols)
    }
}

/// T5: codes and scale bytes of every routed row equal vLLM's scaled_fp4_quant, in each group's
/// swizzled block; the grouped arguments carry the route's row counts.
#[test]
fn nvfp4_expand() -> Result<()> {
    use crate::experts::Tile;
    use nvfp4_cases::*;
    let dev = device();
    for tag in ["r", "s"] {
        let mut c = load(&dev, tag);
        let xq = bytes(c.get("xq"));
        let xs = bytes(c.get("xs"));
        let ids = ids(&c);
        let (h, k, m) = (c.h, c.k, c.m);
        let cols = h / 16;
        for tile in Tile::ALL {
            let run = expand(&dev, &mut c, tile);
            let r = m * k;
            let a1 = read::<u8>(&dev, &run.ws, run.layout.a1, r * h / 2);
            let s1 = read::<u8>(&dev, &run.ws, run.layout.s1, run.layout.y1 - run.layout.s1);
            let mut bad = 0;
            for f in 0..r {
                let (t, e, pos) = (f / k, ids[f], run.dst[f] as usize);
                if a1[pos * h / 2..(pos + 1) * h / 2] != xq[t * h / 2..(t + 1) * h / 2] {
                    bad += 1;
                }
                let (base, row) = sf_at(&run, e, pos, cols);
                for b in 0..cols {
                    if s1[sf_byte(base, row, b, cols)] != xs[t * cols + b] {
                        bad += 1;
                    }
                }
            }
            assert_eq!(bad, 0, "{tag} {tile:?}: {bad} rows/scales differ from scaled_fp4_quant");
            // Grouped arguments: problem shapes first, three ints per group.
            let groups = r.min(c.ex.experts());
            let shapes = read::<i32>(&dev, &run.ws, run.layout.args1, 3 * groups);
            let active = read::<i32>(&dev, &run.ws, run.layout.active, groups);
            for g in 0..groups {
                let rows = if active[g] >= 0 {
                    let e = active[g] as usize;
                    run.offsets[e + 1] - run.offsets[e]
                } else {
                    0
                };
                let (mm, nn, kk) = match tile {
                    Tile::P => (shapes[3 * g], shapes[3 * g + 1], shapes[3 * g + 2]),
                    _ => (shapes[3 * g + 1], shapes[3 * g], shapes[3 * g + 2]),
                };
                assert_eq!((mm, nn, kk), (rows, 2 * c.i as i32, h as i32), "{tag} {tile:?} group {g}");
            }
        }
        // Edge rows of the synthetic case: zeros give scale 0 and code 0; saturating rows reach
        // codes 7 and 15.
        if tag == "s" {
            let row = |t: usize| &xq[t * h / 2..(t + 1) * h / 2];
            assert!(row(1).iter().all(|&b| b == 0) && xs[cols..2 * cols].iter().all(|&b| b == 0));
            assert!(row(2).iter().all(|&b| (b & 7) == 7 && (b >> 4) & 7 == 7));
        }
        println!("nvfp4_expand {tag}: exact on P, D32, D64");
    }
    Ok(())
}

/// Sorted-order rows of `y` (bf16 at byte `off`, `cols` wide) back in flat order.
fn unsort(dev: &Device, ws: &Tensor, off: usize, dst: &[i32], cols: usize) -> Vec<f32> {
    let r = dst.len();
    let raw = read::<u16>(dev, ws, off, r * cols);
    let mut out = vec![0f32; r * cols];
    for (f, &p) in dst.iter().enumerate() {
        for c in 0..cols {
            out[f * cols + c] = bf16::from_bits(raw[p as usize * cols + c]).to_f32();
        }
    }
    out
}

/// T6: both grouped GEMMs against vLLM's dense NVFP4 GEMM per expert, on every tile.
#[test]
fn nvfp4_gemm() -> Result<()> {
    use crate::experts::nvfp4::Stage;
    use crate::experts::Tile;
    use nvfp4_cases::*;
    let dev = device();
    for tag in ["r", "s"] {
        let mut c = load(&dev, tag);
        let want1 = bf16s(c.get("gemm1"));
        let want2 = bf16s(c.get("gemm2"));
        let codes = bytes(c.get("act_codes"));
        let scales = bytes(c.get("act_scales"));
        let ids = ids(&c);
        let (h, i, r) = (c.h, c.i, c.m * c.k);
        for tile in Tile::ALL {
            let run = expand(&dev, &mut c, tile);
            c.ex.exec(&run.bound, Stage::Gemm1, Stage::Gemm1)?;
            let raw = read::<u16>(&dev, &run.ws, run.layout.y1, r * 2 * i);
            assert!(raw.iter().all(|&v| v != 0xFFFF), "{tag} {tile:?}: GEMM1 left rows unwritten");
            let got1 = unsort(&dev, &run.ws, run.layout.y1, &run.dst, 2 * i);
            let g1 = gap(&got1, &want1, 2 * i);
            println!("nvfp4_gemm {tag} {tile:?} GEMM1 {g1:?}");
            assert!(g1.bit_exact >= 0.999 && g1.max_ulp <= 1.0, "{tag} {tile:?} GEMM1 {g1:?}");

            // GEMM2 on the probe's activation, written into the act output region.
            let cols = i / 16;
            let mut a2 = read::<u8>(&dev, &run.ws, run.layout.a2, r * i / 2);
            let mut s2 = read::<u8>(&dev, &run.ws, run.layout.s2, run.layout.a1 - run.layout.s2);
            for f in 0..r {
                let (e, pos) = (ids[f], run.dst[f] as usize);
                a2[pos * i / 2..(pos + 1) * i / 2].copy_from_slice(&codes[f * i / 2..(f + 1) * i / 2]);
                let (base, row) = sf_at(&run, e, pos, cols);
                for b in 0..cols {
                    s2[sf_byte(base, row, b, cols)] = scales[f * cols + b];
                }
            }
            write(&dev, &run.ws, run.layout.a2, &a2);
            write(&dev, &run.ws, run.layout.s2, &s2);
            c.ex.exec(&run.bound, Stage::Gemm2, Stage::Gemm2)?;
            let got2 = unsort(&dev, &run.ws, run.layout.y2, &run.dst, h);
            let g2 = gap(&got2, &want2, h);
            println!("nvfp4_gemm {tag} {tile:?} GEMM2 {g2:?}");
            assert!(g2.bit_exact >= 0.999 && g2.max_ulp <= 1.0, "{tag} {tile:?} GEMM2 {g2:?}");
        }
    }
    Ok(())
}

fn decode(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let m = MAG[(code & 7) as usize];
    if code & 8 != 0 {
        -m
    } else {
        m
    }
}

/// T7: act (FlashInfer's fast-math SwiGLU, bf16, FP4 requantization) on the fixture's GEMM1
/// output equals FlashInfer's own FC2 input, value for value.
#[test]
fn nvfp4_act() -> Result<()> {
    use crate::experts::nvfp4::Stage;
    use crate::experts::Tile;
    use nvfp4_cases::*;
    let dev = device();
    for tag in ["r", "s"] {
        let mut c = load(&dev, tag);
        let gemm1 = host::<bf16>(c.get("gemm1"));
        let want = bf16s(c.get("act"));
        let ids = ids(&c);
        let (i, r) = (c.i, c.m * c.k);
        for tile in [Tile::P, Tile::D32] {
            let run = expand(&dev, &mut c, tile);
            let mut y1 = vec![bf16::ZERO; r * 2 * i];
            for f in 0..r {
                let pos = run.dst[f] as usize;
                y1[pos * 2 * i..(pos + 1) * 2 * i].copy_from_slice(&gemm1[f * 2 * i..(f + 1) * 2 * i]);
            }
            write(&dev, &run.ws, run.layout.y1, &y1);
            c.ex.exec(&run.bound, Stage::Act, Stage::Act)?;
            let a2 = read::<u8>(&dev, &run.ws, run.layout.a2, r * i / 2);
            let s2 = read::<u8>(&dev, &run.ws, run.layout.s2, run.layout.a1 - run.layout.s2);
            let cols = i / 16;
            let mut bad = 0usize;
            for f in 0..r {
                let (e, pos) = (ids[f], run.dst[f] as usize);
                let (base, row) = sf_at(&run, e, pos, cols);
                for col in 0..i {
                    let byte = a2[pos * i / 2 + col / 2];
                    let code = if col % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    let sc = float8::F8E4M3::from_bits(s2[sf_byte(base, row, col / 16, cols)]).to_f32();
                    let got = decode(code) * sc;
                    if got != want[f * i + col] {
                        bad += 1;
                    }
                }
            }
            assert_eq!(bad, 0, "{tag} {tile:?}: {bad} of {} act values differ from FlashInfer", r * i);
            println!("nvfp4_act {tag} {tile:?}: {} values exact", r * i);
        }
    }
    Ok(())
}

/// T8: the whole forward against FlashInfer's cutlass_fused_moe, deterministic, graph-safe.
#[test]
fn nvfp4_forward() -> Result<()> {
    use crate::experts::nvfp4::Stage;
    use crate::experts::Tile;
    use nvfp4_cases::*;
    let dev = device();
    for tag in ["r", "s"] {
        let mut c = load(&dev, tag);
        let want = bf16s(c.get("out"));
        let ref32 = host::<f32>(c.get("ref32"));
        let fi_err = norm(&want, &ref32);
        for tile in Tile::ALL {
            c.ex.set_tile(Some(tile));
            let mut first: Option<Vec<u16>> = None;
            for _ in 0..3 {
                let out = c.ex.forward(&c.x, &c.ids, &c.w, None)?;
                let bits: Vec<u16> = host::<bf16>(&out).iter().map(|v| v.to_bits()).collect();
                match &first {
                    None => first = Some(bits),
                    Some(f) => assert_eq!(f, &bits, "{tag} {tile:?}: runs differ"),
                }
            }
            let got: Vec<f32> = first.unwrap().iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
            let g = gap(&got, &want, c.h);
            let ratio = norm(&got, &want) / fi_err;
            println!("nvfp4_forward {tag} {tile:?} {g:?} ||ours-FI||/||FI-ref32|| = {ratio:.4}");
            assert!(g.bit_exact >= 0.99, "{tag} {tile:?} {g:?}");
            assert!(g.p999_ulp <= 2.0, "{tag} {tile:?} {g:?}");
            assert!(g.min_cos >= 0.99995, "{tag} {tile:?} {g:?}");
            assert!(ratio <= 0.02, "{tag} {tile:?} ratio {ratio}");
        }
    }

    // CUDA graph: capture at M=4 over caller-owned buffers, rewrite the inputs, replay.
    let mut c = load(&dev, "s");
    c.ex.set_tile(None);
    let (m, k, h) = (c.m, c.k, c.h);
    let layout = c.ex.layout(m, k)?;
    let x = c.x.copy()?;
    let ids = c.ids.copy()?;
    let w = c.w.copy()?;
    let ws = Tensor::zeros(layout.total, DType::U8, &dev)?;
    let out = Tensor::zeros((m, h), DType::BF16, &dev)?;
    let bound = c.ex.bind(&x, &ids, &w, None, &ws, &out)?;
    c.ex.exec(&bound, Stage::Route, Stage::Combine)?; // warm
    let cd = cuda_dev(&dev);
    let stream = cd.cuda_stream();
    stream.synchronize().unwrap();
    use hanzo_ml::cuda::cudarc::driver::sys;
    stream
        .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
        .unwrap();
    c.ex.exec(&bound, Stage::Route, Stage::Combine)?;
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .unwrap()
        .expect("captured graph");
    let mut rng = Rng::new(99);
    let x2: Vec<bf16> = (0..m * h).map(|_| bf16::from_f32(rng.normal())).collect();
    let ids2: Vec<i32> = distinct(&mut rng, m, k, 512).iter().map(|&v| v as i32).collect();
    let w2: Vec<f32> = (0..m * k).map(|_| rng.unit()).collect();
    write(&dev, &x, 0, &x2);
    write(&dev, &ids, 0, &ids2);
    write(&dev, &w, 0, &w2);
    graph.launch().unwrap();
    stream.synchronize().unwrap();
    let replay: Vec<u16> = host::<bf16>(&out).iter().map(|v| v.to_bits()).collect();
    let eager = c.ex.forward(&x, &ids, &w, None)?;
    let eager: Vec<u16> = host::<bf16>(&eager).iter().map(|v| v.to_bits()).collect();
    assert_eq!(replay, eager, "graph replay differs from eager");
    println!("nvfp4_forward: graph replay == eager at M={m}");

    // The published arena formula.
    for m in [1usize, 64, 4096] {
        let l = c.ex.layout(m, 10)?;
        let fixed = l.a2; // route, arguments and CUTLASS workspace come first
        let r = m * 10;
        let g = r.min(512);
        let al = |b: usize| b.div_ceil(256) * 256;
        let sf_rows = |_: usize| (r + g * 127).div_ceil(128) * 128 + 128;
        let (hh, ii) = (c.ex.hidden(), c.ex.inter());
        let region_a = al(r * ii / 2) + al(sf_rows(0) * ii / 16);
        let region_s = (al(r * hh / 2) + al(sf_rows(0) * hh / 16) + al(r * 2 * ii * 2)).max(al(r * hh * 2));
        assert_eq!(l.total, fixed + region_a + region_s, "arena formula at M={m}");
        println!("nvfp4 arena M={m}: {} bytes ({} fixed + {} act + {} gemm)", l.total, fixed, region_a, region_s);
    }

    // new() refuses a gate/up weight_scale_2 mismatch and a hidden size off the 128 grid.
    let g = |n: &str| c.fx[&format!("s.{n}")].clone();
    let e = g("w13").dim(0)?;
    let bad = Tensor::from_vec(
        (0..2 * e).map(|v| if v == 1 { 0.5f32 } else { 0.01 }).collect::<Vec<_>>(),
        (e, 2),
        &dev,
    )?;
    let r = crate::experts::nvfp4::Experts::new(
        &g("w13"),
        &g("w13_scale"),
        &bad,
        &g("w13_input"),
        &g("w2"),
        &g("w2_scale"),
        &g("w2_global"),
        &g("w2_input"),
    );
    assert!(r.is_err(), "gate/up weight_scale_2 mismatch accepted");
    let w13 = g("w13").narrow(2, 0, 32)?.contiguous()?; // H = 64
    let r = crate::experts::nvfp4::Experts::new(
        &w13,
        &g("w13_scale").narrow(2, 0, 4)?.contiguous()?,
        &g("w13_global"),
        &g("w13_input"),
        &g("w2").narrow(1, 0, 64)?.contiguous()?,
        &g("w2_scale").narrow(1, 0, 64)?.contiguous()?,
        &g("w2_global"),
        &g("w2_input"),
    );
    assert!(r.is_err(), "H % 128 != 0 accepted");
    Ok(())
}
