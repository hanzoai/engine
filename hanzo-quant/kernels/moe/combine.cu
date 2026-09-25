/**
 * combine: the k-way weighted reduction of the routed experts' rows back into token order.
 *
 * Two rules, each the reference's own arithmetic, both starting from +0.0f, walking slots
 * j = 0..k-1 in order and rounding to bf16 once:
 *   FINALIZE  s = fma.rn.ftz(w[t,j], f32(y), s)   FlashInfer finalizeMoeRoutingKernel, whose
 *             `thread_output + row_scale * expert_result` compiles to FFMA.FTZ under
 *             -use_fast_math (fused_moe_120.so, sm_121a).
 *   SUM       s = add.rn(s, f32(y))              vLLM moe_sum (moe_align_sum_kernels.cu), the
 *             router weight having been applied in the GEMM2 epilogue.
 * An addend (the shared expert) joins as vLLM adds it: bf16_rn(f32(bf16_rn(s)) + f32(addend)).
 *
 * One CTA per token, 8 bf16 (16 bytes) per thread per slot. No atomics: deterministic.
 * The FTZ operation is inline PTX so neither --fmad nor fast math can change it.
 */

#include "moe.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int VEC = 8;
constexpr int MAX_K = 32;

__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

__device__ __forceinline__ float add_rn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

union Pack {
  uint4 raw;
  __nv_bfloat16 v[VEC];
};

template <int RULE, bool ADDEND>
__global__ void combine_kernel(const __nv_bfloat16 *__restrict__ y, const int32_t *__restrict__ dst,
                               const float *__restrict__ weights,
                               const __nv_bfloat16 *__restrict__ addend,
                               __nv_bfloat16 *__restrict__ out, int k, int h) {
  __shared__ int rows[MAX_K];
  __shared__ float w[MAX_K];
  const long t = blockIdx.x;
  if (threadIdx.x < k) {
    rows[threadIdx.x] = dst[t * k + threadIdx.x];
    w[threadIdx.x] = RULE == HANZO_MOE_FINALIZE ? weights[t * k + threadIdx.x] : 1.0f;
  }
  __syncthreads();
  const int vecs = h / VEC;
  for (int c = threadIdx.x; c < vecs; c += blockDim.x) {
    float s[VEC];
#pragma unroll
    for (int i = 0; i < VEC; i++)
      s[i] = 0.0f;
    for (int j = 0; j < k; j++) {
      Pack p;
      p.raw = reinterpret_cast<const uint4 *>(y + (long)rows[j] * h)[c];
#pragma unroll
      for (int i = 0; i < VEC; i++) {
        const float v = __bfloat162float(p.v[i]);
        s[i] = RULE == HANZO_MOE_FINALIZE ? fma_ftz(w[j], v, s[i]) : add_rn(s[i], v);
      }
    }
    Pack o;
    if (ADDEND) {
      Pack a;
      a.raw = reinterpret_cast<const uint4 *>(addend + t * h)[c];
#pragma unroll
      for (int i = 0; i < VEC; i++)
        o.v[i] = __float2bfloat16_rn(
            add_rn(__bfloat162float(__float2bfloat16_rn(s[i])), __bfloat162float(a.v[i])));
    } else {
#pragma unroll
      for (int i = 0; i < VEC; i++)
        o.v[i] = __float2bfloat16_rn(s[i]);
    }
    reinterpret_cast<uint4 *>(out + t * h)[c] = o.raw;
  }
}

template <int RULE>
void launch(const __nv_bfloat16 *y, const int32_t *dst, const float *weights,
            const __nv_bfloat16 *addend, __nv_bfloat16 *out, int m, int k, int h,
            cudaStream_t stream) {
  int threads = h / VEC;
  threads = threads > 512 ? 512 : ((threads + 31) / 32) * 32;
  if (addend)
    combine_kernel<RULE, true><<<m, threads, 0, stream>>>(y, dst, weights, addend, out, k, h);
  else
    combine_kernel<RULE, false><<<m, threads, 0, stream>>>(y, dst, weights, addend, out, k, h);
}

bool aligned16(const void *p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

} // namespace

extern "C" int hanzo_moe_combine(const __nv_bfloat16 *y, const int32_t *dst, const float *weights,
                                 const __nv_bfloat16 *addend, __nv_bfloat16 *out, int m, int k,
                                 int h, int rule, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || k > MAX_K || h <= 0 || h % VEC != 0)
    return HANZO_MOE_BAD_ARGUMENT;
  if (!aligned16(y) || !aligned16(out) || (addend && !aligned16(addend)))
    return HANZO_MOE_UNALIGNED;
  if (rule == HANZO_MOE_FINALIZE) {
    if (!weights)
      return HANZO_MOE_BAD_ARGUMENT;
    launch<HANZO_MOE_FINALIZE>(y, dst, weights, addend, out, m, k, h, stream);
  } else if (rule == HANZO_MOE_SUM) {
    launch<HANZO_MOE_SUM>(y, dst, weights, addend, out, m, k, h, stream);
  } else {
    return HANZO_MOE_BAD_ARGUMENT;
  }
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : -(int)e;
}
