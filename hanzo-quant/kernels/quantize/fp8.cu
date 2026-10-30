/**
 * Per-token, per-128-group E4M3 activation quantization, as the served kernels compute it.
 *
 * Modes (one scale s per row and group, q = e4m3 RN satfinite of clamp(x / s, +-448)):
 *   0 linear: s = max(amax * RN(1/448), 1/229376), x / s by div.full.f32. The inductor kernel that
 *             quantizes every W8A8 input in the compiled graph (it folds the /448 into a multiply).
 *   1 eager:  s = max(amax div.full 448, 1/229376). The standalone QuantFP8 kernel that runs inside
 *             custom ops (the shared expert inside moe_forward_shared), divisor passed at runtime.
 *   2 gather: vLLM's per_token_group_fp8_quant (eps 1e-10), used by the fused-MoE gathers:
 *             s = max(amax, 1e-10) / 448 (IEEE), q = x / s (IEEE).
 *
 * Built without --use_fast_math and with --fmad=false: div.full.f32 is written as inline PTX where
 * the served kernel uses it, and every other operation is the IEEE one.
 */

#include "e2m1.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int GROUP = 128;
constexpr float RCP448 = 0.002232142857142857f;
constexpr float FLOOR = 4.359654017857143e-06f;

template <typename T> __device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// One warp per (row, group): 32 lanes x 4 values.
template <typename T>
__global__ void fp8_kernel(const T *__restrict__ x, uint8_t *__restrict__ q,
                           float *__restrict__ s, int m, int k, int mode) {
  const int groups = k / GROUP;
  const long warp = ((long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (warp >= (long)m * groups)
    return;
  const long row = warp / groups;
  const int g = warp % groups;
  const T *src = x + row * k + g * GROUP;
  float v[4];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    v[j] = to_f32<T>(src[lane * 4 + j]);
    amax = fmaxf(amax, fabsf(v[j]));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));

  float scale;
  if (mode == 0)
    scale = fmaxf(amax * RCP448, FLOOR);
  else if (mode == 1)
    scale = fmaxf(e2m1::div_full(amax, 448.0f), FLOOR);
  else
    scale = fmaxf(amax, 1e-10f) / 448.0f;

  uint8_t *dst = q + row * k + g * GROUP + lane * 4;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float r = mode == 2 ? v[j] / scale : e2m1::div_full(v[j], scale);
    r = fminf(fmaxf(r, -448.0f), 448.0f);
    dst[j] = __nv_cvt_float_to_fp8(r, __NV_SATFINITE, __NV_E4M3);
  }
  if (lane == 0)
    s[row * groups + g] = scale;
}

template <typename T>
void launch(const T *x, uint8_t *q, float *s, int m, int k, int mode, cudaStream_t stream) {
  const long warps = (long)m * (k / GROUP);
  const int threads = 256;
  const long blocks = (warps * 32 + threads - 1) / threads;
  fp8_kernel<T><<<(unsigned)blocks, threads, 0, stream>>>(x, q, s, m, k, mode);
}

} // namespace

extern "C" void hanzo_quantize_fp8_f32(const float *x, uint8_t *q, float *s, int m, int k,
                                       int mode, cudaStream_t stream) {
  launch<float>(x, q, s, m, k, mode, stream);
}

extern "C" void hanzo_quantize_fp8_f16(const __half *x, uint8_t *q, float *s, int m, int k,
                                       int mode, cudaStream_t stream) {
  launch<__half>(x, q, s, m, k, mode, stream);
}

extern "C" void hanzo_quantize_fp8_bf16(const __nv_bfloat16 *x, uint8_t *q, float *s, int m,
                                        int k, int mode, cudaStream_t stream) {
  launch<__nv_bfloat16>(x, q, s, m, k, mode, stream);
}
