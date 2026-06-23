//! MXFP4 int8 / dp4a matmul path (scaffold).
//!
//! MXFP4 (OCP microscaling FP4) stores, per 32-element block: one shared E8M0 power-of-two scale
//! (a raw u8 exponent, bias 127) and 16 packed bytes -> 32 E2M1 4-bit codes. The 16 distinct
//! magnitudes the E2M1 LUT can take are {0,.5,1,1.5,2,3,4,6} (+/-), all multiples of 0.5, so each
//! code is EXACTLY representable as an int8 in [-12, 12] via `code_i8 = round(value * 2)` (the
//! `KVALUES` table below; identical to llama.cpp `kvalues_mxfp4`). That exactness is the whole point:
//! it lets us route MXFP4 through the same hardware int8 dot-product (dp4a / IDP4A / DP4a, i.e. 4x
//! int8 MAC per instruction) that the Q8_0 prefill path already uses, instead of expanding to f32.
//!
//! Plan (mirrors the Q8_0 dp4a prefill lever in hanzo-ml `matmul_q8_dp4a_gpu`):
//!   1. weight side: unpack each FP4 nibble -> int8 code in [-12,12] (`unpack_block_to_i8`). The
//!      per-block f32 multiplier is `0.5 * 2^(e-127)` (the 0.5 undoes the *2 baked into the codes).
//!   2. activation side: symmetric int8 quantize each 32-wide activation block: `xq = round(x*127/amax)`,
//!      with per-block scale `sx = amax/127` (same quantizer as hanzo-ml `quantize_act_q8.comp`).
//!   3. accumulate `dp4a(xq_block, wq_block)` -> i32 per block, then
//!      `acc_f32 += i32 * sx * (0.5 * 2^(e-127))`. Sum over blocks = the row dot product.
//!
//! This module currently provides the int8 unpack + a portable CPU int8 GEMM reference
//! (`matmul_i8_reference`) used to validate the numerics. The GPU dp4a kernels are NOT wired:
//! `mxfp4_matmul_dp4a` falls back to the f32 dequant matmul. The CUDA hook is where a real
//! `BlockMXFP4 -> q8 bank -> dp4a` kernel would slot in (see TODO); on Vulkan the equivalent is to
//! upload the unpacked int8 codes + per-block scales and reuse the existing `mul_mat_q8_dp4a` shader.

use hanzo_ml::{DType, Device, Result, Tensor};

use super::MXFP4_BLOCK_SIZE;

/// E2M1 nibble -> int8 code = value*2 (value in {0,.5,1,1.5,2,3,4,6} +/-). Exact; bit-for-bit the
/// llama.cpp `kvalues_mxfp4` table. The *2 is undone by the 0.5 in the per-block multiplier.
pub(crate) const KVALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Per-block f32 multiplier so that `code_i8 * mult == dequantized weight`. `0.5` cancels the *2 in
/// KVALUES; `2^(e-127)` is the E8M0 scale. Matches hanzo-ml `e8m0_to_fp32_half`.
#[inline]
pub(crate) fn block_multiplier(e: u8) -> f32 {
    // 2^(e-127) via raw exponent bits; e==0 would be 2^-127 (subnormal-ish) but MXFP4 never emits 0.
    0.5 * f32::from_bits((e as u32) << 23)
}

/// Unpack one MXFP4 block's 16 packed bytes -> 32 int8 codes in [-12, 12].
/// Layout matches `BlockMXFP4.qs`: byte j low nibble -> code j, high nibble -> code j+16.
#[inline]
pub(crate) fn unpack_block_to_i8(packed: &[u8], out: &mut [i8; MXFP4_BLOCK_SIZE]) {
    let half = MXFP4_BLOCK_SIZE / 2;
    for j in 0..half {
        let b = packed[j];
        out[j] = KVALUES[(b & 0x0F) as usize];
        out[j + half] = KVALUES[((b >> 4) & 0x0F) as usize];
    }
}

/// Symmetric int8 quantize one activation block (CPU reference for the dp4a activation side).
/// Returns the int8 codes and the per-block scale `amax/127` (so `x ~= code * scale`).
#[inline]
pub(crate) fn quantize_act_block_i8(x: &[f32], out: &mut [i8; MXFP4_BLOCK_SIZE]) -> f32 {
    let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if amax == 0.0 {
        out.iter_mut().for_each(|v| *v = 0);
        return 1.0;
    }
    let inv = 127.0 / amax;
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    amax / 127.0
}

/// Portable int8 block dot (the dp4a primitive: sum of 32 int8*int8 -> i32). On GPU this is one
/// dp4a per 4 lanes; here it's the scalar reference used to verify the kernel path.
#[inline]
pub(crate) fn dp4a_block(a: &[i8; MXFP4_BLOCK_SIZE], b: &[i8; MXFP4_BLOCK_SIZE]) -> i32 {
    let mut acc = 0i32;
    for i in 0..MXFP4_BLOCK_SIZE {
        acc += a[i] as i32 * b[i] as i32;
    }
    acc
}

/// CPU int8-path reference matmul: `out[m,n] = x[m,k] @ W[n,k]^T` where W is MXFP4
/// (`blocks` = [N, K/2] packed bytes, `scales` = [N, K/32] E8M0). Numerically equivalent to the
/// f32 dequant matmul up to the activation int8 rounding; exists to validate a future dp4a kernel.
pub(crate) fn matmul_i8_reference(x: &Tensor, blocks: &Tensor, scales: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let (m, k) = x.dims2()?;
    let bdims = blocks.dims();
    if bdims.len() != 2 {
        hanzo_ml::bail!("matmul_i8_reference expects 2D MXFP4 weight, got {bdims:?}");
    }
    let (n, k_half) = (bdims[0], bdims[1]);
    if k_half * 2 != k {
        hanzo_ml::bail!("MXFP4 weight K/2 ({k_half}) does not match activation K ({k})");
    }
    let nblk = k / MXFP4_BLOCK_SIZE;
    let half = MXFP4_BLOCK_SIZE / 2;

    let blocks_data: Vec<u8> = blocks.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
    let scales_data: Vec<u8> = scales.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
    let x_data: Vec<f32> = x.flatten_all()?.to_vec1()?;

    // Pre-quantize activations once per (token, block): reused across all N rows (the dp4a win).
    let mut xq = vec![[0i8; MXFP4_BLOCK_SIZE]; m * nblk];
    let mut sx = vec![0f32; m * nblk];
    for t in 0..m {
        for blk in 0..nblk {
            let c0 = blk * MXFP4_BLOCK_SIZE;
            sx[t * nblk + blk] = quantize_act_block_i8(
                &x_data[t * k + c0..t * k + c0 + MXFP4_BLOCK_SIZE],
                &mut xq[t * nblk + blk],
            );
        }
    }

    let mut out = vec![0f32; m * n];
    let mut wq = [0i8; MXFP4_BLOCK_SIZE];
    for row in 0..n {
        for blk in 0..nblk {
            let e = scales_data[row * nblk + blk];
            let wmul = block_multiplier(e);
            unpack_block_to_i8(&blocks_data[row * k_half + blk * half..], &mut wq);
            for t in 0..m {
                let dot = dp4a_block(&xq[t * nblk + blk], &wq);
                out[t * n + row] += dot as f32 * sx[t * nblk + blk] * wmul;
            }
        }
    }
    Tensor::from_vec(out, (m, n), &Device::Cpu)
}

/// Opt-in env flag to exercise the int8 path's CPU reference (`HANZO_MXFP4_DP4A=1`). Off by default.
const ENV_MXFP4_DP4A: &str = "HANZO_MXFP4_DP4A";

/// MXFP4 matmul via the int8/dp4a path. SCAFFOLD: no GPU dp4a kernel is wired yet. By default returns
/// `Ok(None)` so the caller uses the f32 dequant fallback. With `HANZO_MXFP4_DP4A=1` it runs the CPU
/// int8 reference (`matmul_i8_reference`) so the unpack+quantize+dp4a numerics are exercised on real
/// weights. When a kernel lands, gate it here on `Device::Cuda` (BlockMXFP4 -> resident q8 bank ->
/// dp4a) or `Device::Vulkan` (upload unpacked int8 + scales, dispatch `mul_mat_q8_dp4a`).
pub(crate) fn mxfp4_matmul_dp4a(
    x: &Tensor,
    blocks: &Tensor,
    scales: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    if !std::env::var(ENV_MXFP4_DP4A).is_ok_and(|v| v == "1") {
        // TODO(deepseek-fp4): wire CUDA dp4a (reuse hanzo-ml matmul_q8_dp4a_gpu) + Vulkan
        // mul_mat_q8_dp4a. Until then, signal "no fast path" so forward_dequantize runs.
        return Ok(None);
    }
    let orig_dims = x.dims().to_vec();
    let x_2d = if orig_dims.len() > 2 {
        let feats = orig_dims[orig_dims.len() - 1];
        let bs: usize = orig_dims[..orig_dims.len() - 1].iter().product();
        x.reshape((bs, feats))?
    } else {
        x.clone()
    };
    let mut out = matmul_i8_reference(&x_2d, blocks, scales)?
        .to_device(x.device())?
        .to_dtype(x.dtype())?;
    if let Some(bias) = bias {
        out = out.broadcast_add(bias)?;
    }
    if orig_dims.len() > 2 {
        let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
        new_dims.push(out.dim(1)?);
        out = out.reshape(new_dims)?;
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    // FP4 codes round-trip through the int8 LUT exactly (the property the dp4a path relies on).
    #[test]
    fn kvalues_are_value_times_two() {
        let fp4: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for (i, &v) in fp4.iter().enumerate() {
            assert_eq!(KVALUES[i] as f32, v * 2.0, "nibble {i}");
        }
    }

    // The int8 path matches a direct f32 dequant dot to within activation rounding error.
    #[test]
    fn int8_path_matches_f32_dot() {
        // One block, scale e=127 (2^0) -> weight code*0.5. Activations are exact multiples here.
        let e = 127u8;
        let wmul = block_multiplier(e);
        assert_eq!(wmul, 0.5);
        let mut wq = [0i8; MXFP4_BLOCK_SIZE];
        let mut packed = [0u8; MXFP4_BLOCK_SIZE / 2];
        // nibble 5 = code 6 = value 3.0 in both halves.
        packed.iter_mut().for_each(|b| *b = 0x55);
        unpack_block_to_i8(&packed, &mut wq);
        let x = vec![1.0f32; MXFP4_BLOCK_SIZE];
        let mut xq = [0i8; MXFP4_BLOCK_SIZE];
        let sx = quantize_act_block_i8(&x, &mut xq);
        let got = dp4a_block(&xq, &wq) as f32 * sx * wmul;
        // f32 reference: 32 * (1.0 * 3.0) = 96.
        assert!((got - 96.0).abs() < 1e-3, "got {got}");
    }
}
