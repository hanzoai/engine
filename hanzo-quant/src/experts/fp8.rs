//! Block-FP8 routed experts with BF16 (or F32) block scales: Flash-Next's MTP drafter MoE (E4),
//! as vLLM's TritonExperts serves it on sm_121 with VLLM_USE_DEEP_GEMM=0.

use std::ffi::c_void;

use half::bf16;
use hanzo_ml::{DType, Result, Tensor};

use super::ffi::{self, check, Stream};
use super::nvfp4::Stage;
use super::{Layout, Shape, Tile};

const GROUP: usize = 128;
/// M1's quantizer mode for vLLM's per_token_group_quant_fp8 (eps 1e-10, IEEE divisions).
const GATHER: i32 = 2;

#[repr(C)]
struct Launch {
    shape: Shape,
    xq: *const u8,
    xs: *const f32,
    ids: *const i32,
    weights: *const f32,
    addend: *const c_void,
    w13: *const u8,
    w13_s: *const f32,
    w2: *const u8,
    w2_s: *const f32,
    out: *mut c_void,
    workspace: *mut c_void,
    stream: Stream,
}

extern "C" {
    fn hanzo_moe_fp8_prepare(device: i32) -> i32;
    fn hanzo_moe_fp8_layout(shape: *const Shape, layout: *mut Layout) -> i32;
    fn hanzo_moe_fp8_run(launch: *const Launch, first: i32, last: i32) -> i32;
    fn hanzo_quantize_fp8_bf16(
        x: *const bf16,
        q: *mut u8,
        s: *mut f32,
        m: i32,
        k: i32,
        mode: i32,
        stream: Stream,
    );
}

/// One layer's block-FP8 routed experts, prepared once at load.
#[derive(Debug)]
pub struct Experts {
    w13: Tensor,
    w13_s: Tensor,
    w2: Tensor,
    w2_s: Tensor,
    experts: usize,
    hidden: usize,
    inter: usize,
    sm_count: i32,
}

fn widen(s: &Tensor, dims: (usize, usize, usize), what: &str) -> Result<Tensor> {
    if s.dims3()? != dims {
        hanzo_ml::bail!("{what}: expected {dims:?}, got {:?}", s.shape());
    }
    match s.dtype() {
        // BF16 -> F32 is exact.
        DType::BF16 | DType::F32 => s.to_dtype(DType::F32)?.contiguous(),
        d => hanzo_ml::bail!("{what}: block scales must be BF16 or F32, got {d:?}"),
    }
}

impl Experts {
    /// `w13` F8E4M3 `[E, 2I, H]` (gate rows then up rows), `w13_scale` `[E, 2I/128, H/128]`;
    /// `w2` F8E4M3 `[E, H, I]`, `w2_scale` `[E, H/128, I/128]`; scales BF16 (the checkpoint's
    /// `weight_scale_inv`) or F32, widened to F32 once.
    pub fn new(w13: &Tensor, w13_scale: &Tensor, w2: &Tensor, w2_scale: &Tensor) -> Result<Self> {
        let dev = ffi::cuda(w13.device())?;
        ffi::prepare(&dev)?;
        prepare(&dev)?;
        let (e, n13, hidden) = w13.dims3()?;
        if n13 % 2 != 0 {
            hanzo_ml::bail!("w13 rows {n13} are not gate + up");
        }
        let inter = n13 / 2;
        if hidden % GROUP != 0 || inter % GROUP != 0 {
            hanzo_ml::bail!("block-FP8 experts need H and I multiples of 128, got {hidden}, {inter}");
        }
        if w2.dims3()? != (e, hidden, inter) {
            hanzo_ml::bail!("w2 {:?} does not match w13 {:?}", w2.shape(), w13.shape());
        }
        if w13.dtype() != DType::F8E4M3 || w2.dtype() != DType::F8E4M3 {
            hanzo_ml::bail!("block-FP8 expert weights must be F8E4M3");
        }
        if e > 1024 {
            hanzo_ml::bail!("at most 1024 experts, got {e}");
        }
        Ok(Self {
            w13: w13.contiguous()?,
            w13_s: widen(w13_scale, (e, n13 / GROUP, hidden / GROUP), "w13 scale")?,
            w2: w2.contiguous()?,
            w2_s: widen(w2_scale, (e, hidden / GROUP, inter / GROUP), "w2 scale")?,
            experts: e,
            hidden,
            inter,
            sm_count: ffi::sm_count(&dev)?,
        })
    }

    pub fn experts(&self) -> usize {
        self.experts
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn inter(&self) -> usize {
        self.inter
    }

    fn shape(&self, m: usize, k: usize) -> Shape {
        Shape {
            m: m as i32,
            k: k as i32,
            e: self.experts as i32,
            h: self.hidden as i32,
            i: self.inter as i32,
            tile: Tile::P as i32,
            sm_count: self.sm_count,
        }
    }

    /// Byte layout of the transient workspace a forward of `m` tokens at top-`k` uses.
    pub fn layout(&self, m: usize, k: usize) -> Result<Layout> {
        let shape = self.shape(m, k);
        let mut l = Layout::default();
        check(unsafe { hanzo_moe_fp8_layout(&shape, &mut l) }, "fp8 layout")?;
        Ok(l)
    }

    /// `x` BF16 `[M, H]`, `ids` U32/I32 `[M, k]`, `weights` F32 `[M, k]` (applied in GEMM2, as
    /// Triton does), `addend` BF16 `[M, H]`. Returns BF16 `[M, H]`.
    pub fn forward(
        &self,
        x: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
        addend: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dev = ffi::cuda(x.device())?;
        let (m, k) = ids.dims2()?;
        let ws = ffi::empty::<u8>(&dev, self.layout(m, k)?.total)?;
        let out = ffi::empty::<bf16>(&dev, (m, self.hidden))?;
        let b = self.bind(x, ids, weights, addend, &ws, &out)?;
        self.exec(&b, Stage::Route, Stage::Combine)?;
        Ok(out)
    }

    /// Binds a forward to caller-owned buffers (see [`super::nvfp4::Experts::bind`]).
    pub fn bind(
        &self,
        x: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
        addend: Option<&Tensor>,
        ws: &Tensor,
        out: &Tensor,
    ) -> Result<Bound> {
        let dev = ffi::cuda(x.device())?;
        let (m, h) = x.dims2()?;
        let (mi, k) = ids.dims2()?;
        if m != mi || h != self.hidden || weights.dims2()? != (m, k) || out.dims2()? != (m, h) {
            hanzo_ml::bail!(
                "fp8 experts: x {:?}, ids {:?}, weights {:?}, out {:?} for H={}",
                x.shape(),
                ids.shape(),
                weights.shape(),
                out.shape(),
                self.hidden
            );
        }
        if x.dtype() != DType::BF16 || weights.dtype() != DType::F32 || out.dtype() != DType::BF16
        {
            hanzo_ml::bail!("fp8 experts take BF16 x and out, F32 router weights");
        }
        if !matches!(ids.dtype(), DType::U32 | DType::I32) {
            hanzo_ml::bail!("fp8 experts take U32 or I32 ids");
        }
        let layout = self.layout(m, k)?;
        if ws.dtype() != DType::U8 || ws.elem_count() < layout.total {
            hanzo_ml::bail!("fp8 experts need a U8 workspace of {} bytes", layout.total);
        }
        let addend = match addend {
            Some(a) => {
                if a.dims2()? != (m, h) || a.dtype() != DType::BF16 {
                    hanzo_ml::bail!("fp8 experts addend must be BF16 [{m},{h}]");
                }
                ffi::ptr(a)? as *const c_void
            }
            None => std::ptr::null(),
        };
        let base = ffi::ptr(ws)?;
        Ok(Bound {
            launch: Launch {
                shape: self.shape(m, k),
                xq: (base + layout.xq as u64) as *const u8,
                xs: (base + layout.xs as u64) as *const f32,
                ids: ffi::ptr(ids)? as *const i32,
                weights: ffi::ptr(weights)? as *const f32,
                addend,
                w13: ffi::ptr(&self.w13)? as *const u8,
                w13_s: ffi::ptr(&self.w13_s)? as *const f32,
                w2: ffi::ptr(&self.w2)? as *const u8,
                w2_s: ffi::ptr(&self.w2_s)? as *const f32,
                out: ffi::ptr(out)? as *mut c_void,
                workspace: base as *mut c_void,
                stream: ffi::stream(&dev),
            },
            x: ffi::ptr(x)? as *const bf16,
        })
    }

    /// Enqueues stages `first..=last`. Expand is preceded by M1's per-token group quantizer.
    pub fn exec(&self, b: &Bound, first: Stage, last: Stage) -> Result<()> {
        let l = &b.launch;
        for stage in first as i32..=last as i32 {
            if stage == Stage::Expand as i32 {
                unsafe {
                    hanzo_quantize_fp8_bf16(
                        b.x,
                        l.xq as *mut u8,
                        l.xs as *mut f32,
                        l.shape.m,
                        l.shape.h,
                        GATHER,
                        l.stream,
                    )
                };
            }
            check(unsafe { hanzo_moe_fp8_run(l, stage, stage) }, "fp8 forward")?;
        }
        Ok(())
    }
}

/// A forward bound to its buffers' device addresses.
pub struct Bound {
    launch: Launch,
    x: *const bf16,
}

fn prepare(dev: &hanzo_ml::CudaDevice) -> Result<()> {
    use std::sync::{Mutex, OnceLock};
    static DONE: OnceLock<Mutex<Vec<usize>>> = OnceLock::new();
    let ordinal = dev.cuda_stream().context().ordinal();
    let mut done = DONE.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap();
    if !done.contains(&ordinal) {
        check(unsafe { hanzo_moe_fp8_prepare(ordinal as i32) }, "fp8 prepare")?;
        done.push(ordinal);
    }
    Ok(())
}
