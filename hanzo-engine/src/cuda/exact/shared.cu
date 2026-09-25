// Shared-expert epilogue kernels for the Flash-Next (qwen4_exp) MoE block, in the exact lane
// (no --use_fast_math): IEEE expf and division, as torch computes them.
//
// act:  h[t, j] = bf16(g / (1 + expf(-g)) * u), g = gu[t, j], u = gu[t, I + j]: SiluAndMul over
//       vLLM's merged gate_up layout, f32 math, one rounding (torch's f32 reference exactly).
// gate: y[t, :] = bf16(float(bf16(1 / (1 + expf(-g[t])))) * float(d[t, :])): torch's eager
//       sigmoid(g) * d, each op rounding to bf16 as torch's does.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace shared {

struct alignas(16) Vec8 {
  __nv_bfloat16 v[8];
};

__device__ __forceinline__ __nv_bfloat16 silu_mul(__nv_bfloat16 gb, __nv_bfloat16 ub) {
  const float g = __bfloat162float(gb);
  const float u = __bfloat162float(ub);
  const float s = g / (1.0f + expf(-g));
  return __float2bfloat16(s * u);
}

template <bool VEC>
__global__ void act_bf16(const __nv_bfloat16 *__restrict__ gu, __nv_bfloat16 *__restrict__ h,
                         const int64_t rows, const int64_t inter) {
  if constexpr (VEC) {
    const int64_t per_row = inter / 8;
    const int64_t n = rows * per_row;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
      const int64_t t = i / per_row;
      const int64_t j = (i % per_row) * 8;
      const Vec8 g = *reinterpret_cast<const Vec8 *>(gu + t * 2 * inter + j);
      const Vec8 u = *reinterpret_cast<const Vec8 *>(gu + t * 2 * inter + inter + j);
      Vec8 o;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        o.v[k] = silu_mul(g.v[k], u.v[k]);
      }
      *reinterpret_cast<Vec8 *>(h + t * inter + j) = o;
    }
  } else {
    const int64_t n = rows * inter;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
      const int64_t t = i / inter;
      const int64_t j = i % inter;
      h[i] = silu_mul(gu[t * 2 * inter + j], gu[t * 2 * inter + inter + j]);
    }
  }
}

__device__ __forceinline__ __nv_bfloat16 sigmoid_bf16(__nv_bfloat16 g) {
  return __float2bfloat16(1.0f / (1.0f + expf(-__bfloat162float(g))));
}

template <bool VEC>
__global__ void gate_bf16(const __nv_bfloat16 *__restrict__ g, const __nv_bfloat16 *__restrict__ d,
                          __nv_bfloat16 *__restrict__ y, const int64_t rows,
                          const int64_t hidden) {
  if constexpr (VEC) {
    const int64_t per_row = hidden / 8;
    const int64_t n = rows * per_row;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
      const int64_t t = i / per_row;
      const float s = __bfloat162float(sigmoid_bf16(g[t]));
      const Vec8 x = reinterpret_cast<const Vec8 *>(d)[i];
      Vec8 o;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        o.v[k] = __float2bfloat16(s * __bfloat162float(x.v[k]));
      }
      reinterpret_cast<Vec8 *>(y)[i] = o;
    }
  } else {
    const int64_t n = rows * hidden;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
      const float s = __bfloat162float(sigmoid_bf16(g[i / hidden]));
      y[i] = __float2bfloat16(s * __bfloat162float(d[i]));
    }
  }
}

inline bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

inline int blocks(int64_t n, int threads) {
  const int64_t b = (n + threads - 1) / threads;
  return (int)(b < 1024 ? (b < 1 ? 1 : b) : 1024);
}

} // namespace shared

// Both return 0, or 2 for a launch error.
extern "C" int shared_act_bf16(const void *gu, void *h, int64_t rows, int64_t inter,
                               int64_t stream) {
  const auto *a = static_cast<const __nv_bfloat16 *>(gu);
  auto *b = static_cast<__nv_bfloat16 *>(h);
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  constexpr int T = 256;
  if (inter % 8 == 0 && shared::aligned16(a) && shared::aligned16(b)) {
    shared::act_bf16<true><<<shared::blocks(rows * inter / 8, T), T, 0, s>>>(a, b, rows, inter);
  } else {
    shared::act_bf16<false><<<shared::blocks(rows * inter, T), T, 0, s>>>(a, b, rows, inter);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int shared_gate_bf16(const void *g, const void *d, void *y, int64_t rows,
                                int64_t hidden, int64_t stream) {
  const auto *gp = static_cast<const __nv_bfloat16 *>(g);
  const auto *dp = static_cast<const __nv_bfloat16 *>(d);
  auto *yp = static_cast<__nv_bfloat16 *>(y);
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  constexpr int T = 256;
  if (hidden % 8 == 0 && shared::aligned16(dp) && shared::aligned16(yp)) {
    shared::gate_bf16<true><<<shared::blocks(rows * hidden / 8, T), T, 0, s>>>(gp, dp, yp, rows,
                                                                            hidden);
  } else {
    shared::gate_bf16<false><<<shared::blocks(rows * hidden, T), T, 0, s>>>(gp, dp, yp, rows,
                                                                             hidden);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
