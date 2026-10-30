/**
 * NVFP4 building blocks shared by every activation quantizer: the served approximate
 * instructions, the E2M1 encoder and decoder, and the block-scale formula.
 *
 * The served quantizers (vLLM scaled_fp4_quant, FlashInfer cvt_warp_fp16_to_fp4 with fast math
 * on, TensorRT-LLM quantization_utils.cuh) compute, per block of 16 with global scale gs:
 *
 *   SF    = gs * (vecMax * rcp.approx.ftz(6))
 *   sf8   = e4m3 RN satfinite of SF
 *   out   = vecMax != 0 ? rcp.approx.ftz(f32(sf8) * rcp.approx.ftz(gs)) : 0
 *   code  = e2m1 RN satfinite of x * out
 *
 * The approximate reciprocal is inline PTX so no compiler flag can change it; files that include
 * this header are built without --use_fast_math and with --fmad=false.
 *
 * Plain sm_121 has no cvt.rn.satfinite.e2m1x2.f32, so the encoder is software: round to nearest,
 * ties to the even code, saturate at 6, and keep the sign of a zero result, as the instruction
 * does.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <cuda_fp8.h>

namespace e2m1 {

__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float b;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ float div_full(float a, float b) {
  float c;
  asm("div.full.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

/// E2M1 magnitudes by code: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
__host__ __device__ __forceinline__ float decode(uint8_t code) {
  const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float m = mag[code & 7];
  return (code & 8) ? -m : m;
}

/// Round to nearest E2M1, ties to the even code, saturating; NaN saturates. Returns the 4-bit
/// code with the input's sign bit, so -0 and negatives rounding to zero encode as 0b1000.
__host__ __device__ __forceinline__ uint8_t encode(float v) {
#ifdef __CUDA_ARCH__
  const uint32_t bits = __float_as_uint(v);
#else
  uint32_t bits;
  __builtin_memcpy(&bits, &v, 4);
#endif
  const uint8_t sign = (bits >> 31) ? 8 : 0;
  const float a = fabsf(v);
  uint8_t code;
  if (a <= 0.25f)
    code = 0;
  else if (a < 0.75f)
    code = 1;
  else if (a <= 1.25f)
    code = 2;
  else if (a < 1.75f)
    code = 3;
  else if (a <= 2.5f)
    code = 4;
  else if (a < 3.5f)
    code = 5;
  else if (a <= 5.0f)
    code = 6;
  else
    code = 7; // also +inf and NaN
  return code | sign;
}

__device__ __forceinline__ float e4m3_to_float(uint8_t v) {
  __nv_fp8_e4m3 f;
  f.__x = v;
  return static_cast<float>(f);
}

/// The block's E4M3 scale byte and the multiplier its values are encoded with.
struct Block {
  uint8_t scale;
  float out;
};

__device__ __forceinline__ Block block(float vec_max, float gs) {
  const float sf = gs * (vec_max * rcp_approx_ftz(6.0f));
  const uint8_t sf8 = __nv_cvt_float_to_fp8(sf, __NV_SATFINITE, __NV_E4M3);
  const float out =
      vec_max != 0.0f ? rcp_approx_ftz(e4m3_to_float(sf8) * rcp_approx_ftz(gs)) : 0.0f;
  return {sf8, out};
}

} // namespace e2m1
