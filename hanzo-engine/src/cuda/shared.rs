//! Shared-expert kernels for the Flash-Next MoE block, launched on an explicit Lane so the whole
//! shared expert can run on Branch's side stream. The shared gate's logit is `Lane::linear`
//! (N=1), torch's F.linear, so there is no separate logit kernel.

use std::ffi::c_void;

use half::bf16;
use hanzo_ml::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use hanzo_ml::Result;

use super::ffi;
use super::lane::Lane;

fn launched(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        hanzo_ml::bail!("shared::{what}: kernel launch failed");
    }
    Ok(())
}

/// `h[t, j] = bf16(silu(g) * u)` over vLLM's merged layout `gu[t] = [g (inter) | u (inter)]`,
/// in f32 with one rounding.
pub fn act(
    lane: &Lane,
    gu: &impl DevicePtr<bf16>,
    h: &mut impl DevicePtrMut<bf16>,
    rows: usize,
    inter: usize,
) -> Result<()> {
    if gu.len() < rows * 2 * inter || h.len() < rows * inter {
        hanzo_ml::bail!("shared::act: gu {} h {} for rows {rows} inter {inter}", gu.len(), h.len());
    }
    let s = lane.stream();
    let (a, _a) = gu.device_ptr(s);
    let (b, _b) = h.device_ptr_mut(s);
    let rc = unsafe {
        ffi::shared_act_bf16(
            a as *const c_void,
            b as *mut c_void,
            rows as i64,
            inter as i64,
            s.cu_stream() as i64,
        )
    };
    launched(rc, "act")
}

/// `y[t, :] = sigmoid(g[t]) * d[t, :]`, each op rounding to bf16 as torch's eager ops do.
pub fn gate(
    lane: &Lane,
    g: &impl DevicePtr<bf16>,
    d: &impl DevicePtr<bf16>,
    y: &mut impl DevicePtrMut<bf16>,
    rows: usize,
    hidden: usize,
) -> Result<()> {
    if g.len() < rows || d.len() < rows * hidden || y.len() < rows * hidden {
        hanzo_ml::bail!("shared::gate: g {} d {} y {} for rows {rows} hidden {hidden}", g.len(), d.len(), y.len());
    }
    let s = lane.stream();
    let (gp, _g) = g.device_ptr(s);
    let (dp, _d) = d.device_ptr(s);
    let (yp, _y) = y.device_ptr_mut(s);
    let rc = unsafe {
        ffi::shared_gate_bf16(
            gp as *const c_void,
            dp as *const c_void,
            yp as *mut c_void,
            rows as i64,
            hidden as i64,
            s.cu_stream() as i64,
        )
    };
    launched(rc, "gate")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::lane::tests::{device, host, slice, stream, tensor_bits, HIDDEN};
    use crate::cuda::route;

    const TS: [usize; 3] = [1, 16, 64];

    fn ulps(a: u16, b: u16) -> u32 {
        // bf16 bits are sign-magnitude; map to a monotone integer line.
        let key = |v: u16| -> i32 {
            if v & 0x8000 != 0 {
                -((v & 0x7fff) as i32)
            } else {
                v as i32
            }
        };
        (key(a) - key(b)).unsigned_abs()
    }

    /// act against torch's f32 reference (bitwise) and vLLM's compiled SiluAndMul (tolerance).
    #[test]
    fn act_matches() {
        let dev = device();
        let s = stream(&dev);
        let lane = Lane::new(s.clone()).unwrap();
        let f = route::tests::load("shared", &dev);
        for t in TS {
            let gu = &f[&format!("gu.{t}")];
            let inter = gu.dims()[1] / 2;
            let gu = slice(gu);
            let mut h = s.alloc_zeros::<bf16>(t * inter).unwrap();
            act(&lane, &gu, &mut h, t, inter).unwrap();
            let got = host(&s, &h);
            let reference = tensor_bits(&f[&format!("h_ref.{t}")]);
            let bad = got.iter().zip(&reference).filter(|(a, b)| a != b).count();
            assert_eq!(bad, 0, "T={t}: {bad} of {} differ from the torch f32 reference", got.len());
            let vllm = tensor_bits(&f[&format!("h.{t}")]);
            let same = got.iter().zip(&vllm).filter(|(a, b)| a == b).count();
            let worst = got.iter().zip(&vllm).map(|(a, b)| ulps(*a, *b)).max().unwrap();
            let rate = same as f64 / got.len() as f64;
            println!("act T={t}: torch f32 ref 100.000% bitwise; vLLM compiled SiluAndMul {:.3}% bitwise, max {worst} ulp", 100. * rate);
            assert!(rate >= 0.999 && worst <= 1, "T={t}: {:.3}% bitwise, {worst} ulp", 100. * rate);
        }
    }

    /// The shared gate's logit, lane.linear with the [1, 2560] weight on a side Lane, equals
    /// vLLM's F.linear bitwise.
    #[test]
    fn logit_matches() {
        let dev = device();
        let s = stream(&dev);
        let side = Lane::new(s.context().new_stream().unwrap()).unwrap();
        let g = route::tests::load("gate", &dev);
        let f = route::tests::load("shared", &dev);
        let x = slice(&g["x"]);
        let w = slice(&g["ws"]);
        for t in TS {
            let mut y = side.stream().alloc_zeros::<bf16>(t).unwrap();
            side.linear(&x.slice(..t * HIDDEN), &w, &mut y, t, 1, HIDDEN).unwrap();
            side.stream().synchronize().unwrap();
            assert_eq!(host(side.stream(), &y), tensor_bits(&f[&format!("g.{t}")]), "T={t}");
        }
    }

    /// gate applied to vLLM's (g, d) equals vLLM's eager sigmoid(g) * d bitwise.
    #[test]
    fn gate_matches() {
        let dev = device();
        let s = stream(&dev);
        let lane = Lane::new(s.clone()).unwrap();
        let f = route::tests::load("shared", &dev);
        for t in TS {
            let g = slice(&f[&format!("g.{t}")]);
            let d = slice(&f[&format!("d.{t}")]);
            let mut y = s.alloc_zeros::<bf16>(t * HIDDEN).unwrap();
            gate(&lane, &g, &d, &mut y, t, HIDDEN).unwrap();
            let got = host(&s, &y);
            let want = tensor_bits(&f[&format!("y.{t}")]);
            let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(bad, 0, "T={t}: {bad} of {} differ", got.len());
        }
    }
}
