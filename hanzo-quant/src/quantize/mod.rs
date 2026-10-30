//! Activation quantizers bit-exact with the kernels vLLM serves.
//!
//! [`fp8`] quantizes per token and per 128-wide group to E4M3 with an F32 scale, in the three
//! forms the served graph uses ([`Fp8Mode`]). [`nvfp4`] quantizes per 16 to E2M1 codes with an
//! E4M3 scale over a global scale, with the served fast-math formula. On CUDA both reproduce the
//! served approximate instructions (div.full.f32, rcp.approx.ftz.f32) as inline PTX, in kernels
//! built without fast math (`kernels/quantize`). On the CPU the same formulas run with IEEE
//! division and reciprocals, which differ from the served ones only in the last bit.

use hanzo_ml::{DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
mod ffi;

/// Values per FP8 activation scale.
pub const FP8_GROUP: usize = 128;
/// Values per NVFP4 activation scale.
pub const NVFP4_BLOCK: usize = 16;

const RCP448: f32 = 0.002_232_142_857_142_857;
const FLOOR: f32 = 4.359_654_017_857_143e-6;

/// Which served FP8 activation quantizer to reproduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Mode {
    /// The fused inductor quantizer in front of every W8A8 GEMM of the compiled graph:
    /// `s = max(amax * RN(1/448), 1/229376)`, `q = e4m3(clamp(x div.full s, +-448))`.
    Linear,
    /// The standalone QuantFP8 kernel that runs inside custom ops (the shared expert inside
    /// `moe_forward_shared`): `s = max(amax div.full 448, 1/229376)`, `q` as above.
    Eager,
    /// vLLM's `per_token_group_fp8_quant` (eps 1e-10) for fused-MoE gathers:
    /// `s = max(amax, 1e-10) / 448`, `q = e4m3(clamp(x / s, +-448))`, IEEE divisions.
    Gather,
}

impl Fp8Mode {
    fn code(self) -> i32 {
        match self {
            Self::Linear => 0,
            Self::Eager => 1,
            Self::Gather => 2,
        }
    }

    /// The scale of a group whose largest magnitude is `amax`, with IEEE division.
    pub fn scale(self, amax: f32) -> f32 {
        match self {
            Self::Linear => (amax * RCP448).max(FLOOR),
            Self::Eager => (amax / 448.0).max(FLOOR),
            Self::Gather => amax.max(1e-10) / 448.0,
        }
    }
}

/// E4M3 quantization of `x` `[M, K]` (bf16, f16 or f32; K a multiple of 128) per row and
/// 128-wide group. Returns the codes F8E4M3 `[M, K]` and the scales F32 `[M, K/128]`.
pub fn fp8(x: &Tensor, mode: Fp8Mode) -> Result<(Tensor, Tensor)> {
    let (m, k) = x.dims2()?;
    if k % FP8_GROUP != 0 {
        hanzo_ml::bail!("fp8 activation quantization needs K divisible by 128, got {k}");
    }
    match x.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => ffi::fp8(&x.contiguous()?, mode),
        _ => fp8_cpu(x, m, k, mode),
    }
}

/// NVFP4 quantization of `x` `[M, K]` (bf16 or f16; K a multiple of 16) with global scale `gs`
/// (`1 / input_scale`). Returns the codes U8 `[M, K/2]` (low nibble first) and the block scales
/// F8E4M3 `[M, K/16]` in linear layout.
pub fn nvfp4(x: &Tensor, gs: f32) -> Result<(Tensor, Tensor)> {
    let (m, k) = x.dims2()?;
    if k % NVFP4_BLOCK != 0 {
        hanzo_ml::bail!("nvfp4 activation quantization needs K divisible by 16, got {k}");
    }
    match x.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => ffi::nvfp4(&x.contiguous()?, gs),
        _ => nvfp4_cpu(x, m, k, gs),
    }
}

/// Round to the nearest E2M1 code, ties to the even code, saturating at 6 (NaN too), keeping the
/// sign bit of the input even when the result is zero: `cvt.rn.satfinite.e2m1x2.f32`.
pub fn e2m1_encode(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 8 } else { 0 };
    let a = v.abs();
    let code = if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    code | sign
}

/// The value of an E2M1 code.
pub fn e2m1_decode(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let m = MAG[(code & 7) as usize];
    if code & 8 != 0 {
        -m
    } else {
        m
    }
}

/// The E4M3 scale byte of a block whose largest magnitude is `vmax`, and the multiplier its
/// values are encoded with: `SF = gs * (vmax * rcp(6))`, `out = rcp(f32(e4m3(SF)) * rcp(gs))`,
/// or 0 for an all-zero block. IEEE reciprocals here; `rcp.approx.ftz` on CUDA.
pub fn nvfp4_block(vmax: f32, gs: f32) -> (u8, f32) {
    let sf = gs * (vmax * (1.0 / 6.0f32));
    let sf8 = float8::F8E4M3::from_f32(sf);
    let out = if vmax != 0.0 {
        1.0 / (sf8.to_f32() * (1.0 / gs))
    } else {
        0.0
    };
    (sf8.to_bits(), out)
}

fn rows_f32(x: &Tensor) -> Result<Vec<f32>> {
    x.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

fn fp8_cpu(x: &Tensor, m: usize, k: usize, mode: Fp8Mode) -> Result<(Tensor, Tensor)> {
    let dev = x.device().clone();
    let v = rows_f32(x)?;
    let groups = k / FP8_GROUP;
    let mut q = Vec::with_capacity(m * k);
    let mut s = Vec::with_capacity(m * groups);
    for g in v.chunks(FP8_GROUP) {
        let amax = g.iter().fold(0f32, |a, x| a.max(x.abs()));
        let scale = mode.scale(amax);
        s.push(scale);
        q.extend(
            g.iter()
                .map(|x| float8::F8E4M3::from_f32((x / scale).clamp(-448.0, 448.0))),
        );
    }
    Ok((
        Tensor::from_vec(q, (m, k), &Device::Cpu)?.to_device(&dev)?,
        Tensor::from_vec(s, (m, groups), &Device::Cpu)?.to_device(&dev)?,
    ))
}

fn nvfp4_cpu(x: &Tensor, m: usize, k: usize, gs: f32) -> Result<(Tensor, Tensor)> {
    let dev = x.device().clone();
    let v = rows_f32(x)?;
    let mut codes = Vec::with_capacity(m * k / 2);
    let mut scales = Vec::with_capacity(m * k / NVFP4_BLOCK);
    for b in v.chunks(NVFP4_BLOCK) {
        let vmax = b.iter().fold(0f32, |a, x| a.max(x.abs()));
        let (sf8, out) = nvfp4_block(vmax, gs);
        scales.push(float8::F8E4M3::from_bits(sf8));
        for pair in b.chunks(2) {
            codes.push(e2m1_encode(pair[0] * out) | (e2m1_encode(pair[1] * out) << 4));
        }
    }
    Ok((
        Tensor::from_vec(codes, (m, k / 2), &Device::Cpu)?.to_device(&dev)?,
        Tensor::from_vec(scales, (m, k / NVFP4_BLOCK), &Device::Cpu)?.to_device(&dev)?,
    ))
}

/// The CUDA encoder over `v`, one code per value (for tests).
#[cfg(feature = "cuda")]
pub fn e2m1_encode_cuda(v: &Tensor) -> Result<Tensor> {
    ffi::e2m1(&v.to_dtype(DType::F32)?.contiguous()?)
}

#[cfg(test)]
mod tests;
