//! The k-way reduction of routed expert rows back into token order.

use std::ffi::c_void;

use half::bf16;
use hanzo_ml::{DType, Result, Tensor};

use super::ffi::{self, check};

/// How the slots of a token are summed. Both start from +0, walk the slots in order and round
/// to bf16 once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// `s = fma.rn.ftz(w, y, s)`: FlashInfer's finalize (NVFP4 experts).
    Finalize,
    /// `s = s + y`: vLLM's moe_sum, the weight already applied by GEMM2 (block-FP8 experts).
    Sum,
}

impl Rule {
    fn code(self) -> i32 {
        match self {
            Self::Finalize => ffi::FINALIZE,
            Self::Sum => ffi::SUM,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    y: u64,
    dst: u64,
    weights: u64,
    addend: u64,
    out: u64,
    m: usize,
    k: usize,
    h: usize,
    rule: Rule,
    stream: ffi::Stream,
) -> Result<()> {
    check(
        unsafe {
            ffi::hanzo_moe_combine(
                y as *const c_void,
                dst as *const i32,
                weights as *const f32,
                addend as *const c_void,
                out as *mut c_void,
                m as i32,
                k as i32,
                h as i32,
                rule.code(),
                stream,
            )
        },
        "combine",
    )
}

/// `out[t] = sum_j y[dst[t*k+j]]` under `rule`, plus `addend` when given.
///
/// `y` BF16 `[R,H]`, `dst` I32 `[R]` (from [`super::route`]), `weights` F32 `[M,k]` (used by
/// [`Rule::Finalize`]), `addend` BF16 `[M,H]`. Returns BF16 `[M,H]`.
pub fn combine(
    y: &Tensor,
    dst: &Tensor,
    weights: &Tensor,
    addend: Option<&Tensor>,
    rule: Rule,
) -> Result<Tensor> {
    let dev = ffi::cuda(y.device())?;
    let (m, k) = weights.dims2()?;
    let (r, h) = y.dims2()?;
    if r != m * k || dst.elem_count() != r {
        hanzo_ml::bail!("combine: y has {r} rows for {m} tokens x {k} slots");
    }
    if y.dtype() != DType::BF16 || weights.dtype() != DType::F32 || dst.dtype() != DType::I32 {
        hanzo_ml::bail!("combine takes BF16 rows, F32 weights and I32 positions");
    }
    let out = ffi::empty::<bf16>(&dev, (m, h))?;
    let addend_ptr = match addend {
        Some(a) => {
            if a.dims2()? != (m, h) || a.dtype() != DType::BF16 {
                hanzo_ml::bail!("combine addend must be BF16 [{m},{h}]");
            }
            ffi::ptr(a)?
        }
        None => 0,
    };
    launch(
        ffi::ptr(y)?,
        ffi::ptr(dst)?,
        ffi::ptr(weights)?,
        addend_ptr,
        ffi::ptr(&out)?,
        m,
        k,
        h,
        rule,
        ffi::stream(&dev),
    )?;
    Ok(out)
}
