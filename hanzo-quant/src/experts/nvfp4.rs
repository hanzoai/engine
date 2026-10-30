//! NVFP4 routed experts (Flash-Next E1): FlashInfer's `cutlass_fused_moe` as vLLM serves it on
//! sm_121, in six launches with no host synchronization.

use std::ffi::c_void;

use half::bf16;
use hanzo_ml::{DType, Result, Tensor};

use super::ffi::{self, check, Stream};
use super::{Layout, Shape, Tile};

#[repr(C)]
struct Launch {
    shape: Shape,
    x: *const c_void,
    ids: *const i32,
    weights: *const f32,
    addend: *const c_void,
    w13: *const u8,
    w13_sf: *const u8,
    alpha1: *const f32,
    gs1: f32,
    w2: *const u8,
    w2_sf: *const u8,
    alpha2: *const f32,
    gs2: f32,
    out: *mut c_void,
    workspace: *mut c_void,
    stream: Stream,
}

extern "C" {
    fn hanzo_moe_nvfp4_prepare(device: i32) -> i32;
    fn hanzo_moe_nvfp4_layout(shape: *const Shape, layout: *mut Layout) -> i32;
    fn hanzo_moe_nvfp4_run(launch: *const Launch, first: i32, last: i32) -> i32;
    fn hanzo_nvfp4_swizzle_cuda(
        source: *const c_void,
        dest: *mut c_void,
        rows: i32,
        k: i32,
        dest_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

/// The stages of a forward, in launch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    Route = 0,
    Expand = 1,
    Gemm1 = 2,
    Act = 3,
    Gemm2 = 4,
    Combine = 5,
}

/// One layer's NVFP4 routed experts, prepared once at load.
#[derive(Debug)]
pub struct Experts {
    w13: Tensor,
    w13_sf: Tensor,
    alpha1: Tensor,
    gs1: f32,
    w2: Tensor,
    w2_sf: Tensor,
    alpha2: Tensor,
    gs2: f32,
    experts: usize,
    hidden: usize,
    inter: usize,
    sm_count: i32,
    tile: Option<Tile>,
}

fn per_expert(t: &Tensor, e: usize, what: &str) -> Result<Vec<f32>> {
    let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if v.len() != e && v.len() != 2 * e {
        hanzo_ml::bail!("{what}: expected [{e}] or [{e},2], got {:?}", t.shape());
    }
    Ok(v)
}

fn swizzle(scale: &Tensor, rows: usize, k: usize) -> Result<Tensor> {
    let dev = ffi::cuda(scale.device())?;
    let scale = scale.contiguous()?;
    let bytes = rows * k / 16;
    let out = ffi::empty::<u8>(&dev, bytes)?;
    check(
        unsafe {
            hanzo_nvfp4_swizzle_cuda(
                ffi::ptr(&scale)? as *const c_void,
                ffi::ptr(&out)? as *mut c_void,
                rows as i32,
                k as i32,
                bytes,
                ffi::stream(&dev),
            )
        },
        "weight scale swizzle",
    )?;
    Ok(out)
}

impl Experts {
    /// `w13` U8 `[E, 2I, H/2]` (gate rows then up rows), `w13_scale` F8E4M3 `[E, 2I, H/16]`,
    /// `w13_global` (weight_scale_2) and `w13_input` (input_scale) F32 `[E]` or `[E, 2]`;
    /// `w2` U8 `[E, H, I/2]`, `w2_scale` F8E4M3 `[E, H, I/16]`, `w2_global` and `w2_input` F32
    /// `[E]`. Scales are the checkpoint's linear E4M3 bytes.
    ///
    /// As vLLM does for FlashInfer: one activation scale per GEMM, the max `input_scale` over
    /// every expert; `alpha = weight_scale_2 * max_input`, `gs = 1 / max_input`, both f32 RN.
    /// Where vLLM warns on a gate/up `weight_scale_2` mismatch this is an error.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        w13: &Tensor,
        w13_scale: &Tensor,
        w13_global: &Tensor,
        w13_input: &Tensor,
        w2: &Tensor,
        w2_scale: &Tensor,
        w2_global: &Tensor,
        w2_input: &Tensor,
    ) -> Result<Self> {
        let dev = ffi::cuda(w13.device())?;
        ffi::prepare(&dev)?;
        prepare(&dev)?;
        let (e, n13, h2) = w13.dims3()?;
        let hidden = h2 * 2;
        if n13 % 2 != 0 {
            hanzo_ml::bail!("w13 rows {n13} are not gate + up");
        }
        let inter = n13 / 2;
        if hidden % 128 != 0 || inter % 128 != 0 {
            hanzo_ml::bail!("NVFP4 experts need H and I multiples of 128, got H={hidden} I={inter}");
        }
        if w2.dims3()? != (e, hidden, inter / 2) {
            hanzo_ml::bail!("w2 {:?} does not match w13 {:?}", w2.shape(), w13.shape());
        }
        if w13_scale.dims3()? != (e, n13, hidden / 16) || w2_scale.dims3()? != (e, hidden, inter / 16) {
            hanzo_ml::bail!("NVFP4 expert scales have the wrong shape");
        }
        if e > 1024 {
            hanzo_ml::bail!("at most 1024 experts, got {e}");
        }
        let g13 = per_expert(w13_global, e, "w13 weight_scale_2")?;
        let g13: Vec<f32> = if g13.len() == 2 * e {
            let mut v = Vec::with_capacity(e);
            for x in 0..e {
                let (gate, upv) = (g13[2 * x], g13[2 * x + 1]);
                if gate.to_bits() != upv.to_bits() {
                    hanzo_ml::bail!(
                        "expert {x}: gate weight_scale_2 {gate} != up weight_scale_2 {upv}"
                    );
                }
                v.push(gate);
            }
            v
        } else {
            g13
        };
        let g2 = per_expert(w2_global, e, "w2 weight_scale_2")?;
        if g2.len() != e {
            hanzo_ml::bail!("w2 weight_scale_2 must be [{e}]");
        }
        let a13 = per_expert(w13_input, e, "w13 input_scale")?
            .into_iter()
            .fold(f32::MIN, f32::max);
        let a2 = per_expert(w2_input, e, "w2 input_scale")?
            .into_iter()
            .fold(f32::MIN, f32::max);
        if !(a13 > 0.0 && a2 > 0.0) {
            hanzo_ml::bail!("NVFP4 experts need positive input scales");
        }
        let alpha1: Vec<f32> = g13.iter().map(|w| w * a13).collect();
        let alpha2: Vec<f32> = g2.iter().map(|w| w * a2).collect();
        let device = w13.device();
        let bytes = |t: &Tensor| -> Result<Tensor> {
            match t.dtype() {
                DType::F8E4M3 | DType::U8 => Ok(t.contiguous()?),
                d => hanzo_ml::bail!("NVFP4 scales must be F8E4M3 bytes, got {d:?}"),
            }
        };
        let w13_sf = swizzle(&bytes(w13_scale)?, e * n13, hidden)?;
        let w2_sf = swizzle(&bytes(w2_scale)?, e * hidden, inter)?;
        Ok(Self {
            w13: w13.contiguous()?,
            w13_sf,
            alpha1: Tensor::from_vec(alpha1, e, device)?,
            gs1: 1.0 / a13,
            w2: w2.contiguous()?,
            w2_sf,
            alpha2: Tensor::from_vec(alpha2, e, device)?,
            gs2: 1.0 / a2,
            experts: e,
            hidden,
            inter,
            sm_count: ffi::sm_count(&dev)?,
            tile: None,
        })
    }

    /// Pins the GEMM tile (benchmarks and tests); `None` picks it from the rows per expert.
    pub fn set_tile(&mut self, tile: Option<Tile>) {
        self.tile = tile;
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

    pub(crate) fn shape(&self, m: usize, k: usize) -> Shape {
        let tile = self.tile.unwrap_or_else(|| Tile::pick(m * k, self.experts));
        Shape {
            m: m as i32,
            k: k as i32,
            e: self.experts as i32,
            h: self.hidden as i32,
            i: self.inter as i32,
            tile: tile as i32,
            sm_count: self.sm_count,
        }
    }

    /// Byte layout of the transient workspace a forward of `m` tokens at top-`k` uses.
    pub fn layout(&self, m: usize, k: usize) -> Result<Layout> {
        let shape = self.shape(m, k);
        let mut l = Layout::default();
        check(unsafe { hanzo_moe_nvfp4_layout(&shape, &mut l) }, "nvfp4 layout")?;
        Ok(l)
    }

    /// `x` BF16 `[M, H]`, `ids` U32/I32 `[M, k]`, `weights` F32 `[M, k]`, `addend` BF16
    /// `[M, H]` (the shared expert, added as vLLM adds it). Returns BF16 `[M, H]`.
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

    /// Binds a forward to caller-owned buffers: `ws` holds [`Self::layout`]`.total` bytes and
    /// `out` is BF16 `[M, H]`. The result can be run any number of times, and captured in a
    /// CUDA graph, as long as the buffers live and keep their addresses.
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
                "nvfp4 experts: x {:?}, ids {:?}, weights {:?}, out {:?} for H={}",
                x.shape(),
                ids.shape(),
                weights.shape(),
                out.shape(),
                self.hidden
            );
        }
        if x.dtype() != DType::BF16 || weights.dtype() != DType::F32 || out.dtype() != DType::BF16
        {
            hanzo_ml::bail!("nvfp4 experts take BF16 x and out, F32 router weights");
        }
        if !matches!(ids.dtype(), DType::U32 | DType::I32) {
            hanzo_ml::bail!("nvfp4 experts take U32 or I32 ids");
        }
        let shape = self.shape(m, k);
        let need = self.layout(m, k)?.total;
        if ws.dtype() != DType::U8 || ws.elem_count() < need {
            hanzo_ml::bail!("nvfp4 experts need a U8 workspace of {need} bytes");
        }
        let addend = match addend {
            Some(a) => {
                if a.dims2()? != (m, h) || a.dtype() != DType::BF16 {
                    hanzo_ml::bail!("nvfp4 experts addend must be BF16 [{m},{h}]");
                }
                ffi::ptr(a)? as *const c_void
            }
            None => std::ptr::null(),
        };
        Ok(Bound(Launch {
            shape,
            x: ffi::ptr(x)? as *const c_void,
            ids: ffi::ptr(ids)? as *const i32,
            weights: ffi::ptr(weights)? as *const f32,
            addend,
            w13: ffi::ptr(&self.w13)? as *const u8,
            w13_sf: ffi::ptr(&self.w13_sf)? as *const u8,
            alpha1: ffi::ptr(&self.alpha1)? as *const f32,
            gs1: self.gs1,
            w2: ffi::ptr(&self.w2)? as *const u8,
            w2_sf: ffi::ptr(&self.w2_sf)? as *const u8,
            alpha2: ffi::ptr(&self.alpha2)? as *const f32,
            gs2: self.gs2,
            out: ffi::ptr(out)? as *mut c_void,
            workspace: ffi::ptr(ws)? as *mut c_void,
            stream: ffi::stream(&dev),
        }))
    }

    /// Enqueues stages `first..=last` of a bound forward on its stream.
    pub fn exec(&self, b: &Bound, first: Stage, last: Stage) -> Result<()> {
        check(
            unsafe { hanzo_moe_nvfp4_run(&b.0, first as i32, last as i32) },
            "nvfp4 forward",
        )
    }
}

/// A forward bound to its buffers' device addresses (see [`Experts::bind`]).
pub struct Bound(Launch);

fn prepare(dev: &hanzo_ml::CudaDevice) -> Result<()> {
    use std::sync::{Mutex, OnceLock};
    static DONE: OnceLock<Mutex<Vec<usize>>> = OnceLock::new();
    let ordinal = dev.cuda_stream().context().ordinal();
    let mut done = DONE.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap();
    if !done.contains(&ordinal) {
        check(unsafe { hanzo_moe_nvfp4_prepare(ordinal as i32) }, "nvfp4 prepare")?;
        done.push(ordinal);
    }
    Ok(())
}

/// Linear E4M3 scale bytes as the MMA wants them, for tests: `rows x k/16`, 128x4 atoms.
pub(crate) fn sf_offset(row: usize, col: usize, cols: usize) -> usize {
    (((row / 128) * (cols / 4) + col / 4) * 32 + row % 32) * 16 + (row % 128) / 32 * 4 + col % 4
}

/// First swizzled scale row of compact group `g` whose rows start at sorted position `offset`.
pub(crate) fn sf_base_row(offset: usize, g: usize) -> usize {
    (offset + g * 127).div_ceil(128) * 128
}
