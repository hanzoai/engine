/**
 * @brief Fused MoE expert-combine kernel (prefill + decode).
 *
 * Replaces the two-op `routed.broadcast_mul(scores).sum(dim=topk)` combine,
 * which materialized a [T, topk, N] product and then ran a STRIDED candle
 * reduce over the middle (topk) axis. That reduce (`fast_sum_f32`) gathered
 * `topk` values `N` apart per output element -- one uncoalesced cache line per
 * add, ~15% of prefill GPU time on Qwen3.6-35B (measured: 216ms vs llama's
 * 1ms). llama fuses the same weighted-accumulate; so do we now.
 *
 * out[t, :] = sum_{e in [0,topk)} scores[t, e] * routed[t, e, :]
 *
 * Layout: routed [T, topk, N] contiguous (row-major, the reshape of the
 * indexed-MoE output [T*topk, N]); scores [T, topk] contiguous (always f32);
 * out [T, N]. Accumulation is always f32; routed/out are f32 or bf16 to match
 * the model dtype exactly (same result as the old bf16-mul + f32-reduce path).
 * Grid: one block per token t; threads stride over N (fully coalesced).
 */

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#ifndef NO_BF16_KERNEL
#include <cuda_bf16.h>
#endif

__device__ __forceinline__ float ld_f(const float *p) { return *p; }
__device__ __forceinline__ void st_f(float *p, float v) { *p = v; }
#ifndef NO_BF16_KERNEL
__device__ __forceinline__ float ld_f(const __nv_bfloat16 *p) {
  return __bfloat162float(*p);
}
__device__ __forceinline__ void st_f(__nv_bfloat16 *p, float v) {
  *p = __float2bfloat16(v);
}
#endif

template <typename T, int BLOCK>
__global__ void moe_combine_kernel(const T *__restrict__ routed,     // [T, topk, N]
                                   const float *__restrict__ scores, // [T, topk]
                                   T *__restrict__ out,              // [T, N]
                                   int topk, int n) {
  const int t = blockIdx.x;
  const T *routed_t = routed + (size_t)t * topk * n;

  __shared__ float sc[32];
  if (threadIdx.x < topk)
    sc[threadIdx.x] = scores[(size_t)t * topk + threadIdx.x];
  __syncthreads();

  for (int col = threadIdx.x; col < n; col += BLOCK) {
    float acc = 0.0f;
    const T *p = routed_t + col;
#pragma unroll 4
    for (int e = 0; e < topk; e++)
      acc = __fmaf_rn(sc[e], ld_f(p + (size_t)e * n), acc);
    st_f(out + (size_t)t * n + col, acc);
  }
}

// dtype: 0 = f32, 1 = bf16 (routed/out; scores always f32).
extern "C" void moe_combine_f32(const void *routed, const float *scores, void *out,
                                int num_tokens, int topk, int n, int dtype,
                                cudaStream_t stream) {
  constexpr int BLOCK = 256;
  dim3 grid(num_tokens);
  dim3 block(BLOCK);
  if (dtype == 0) {
    moe_combine_kernel<float, BLOCK><<<grid, block, 0, stream>>>(
        reinterpret_cast<const float *>(routed), scores,
        reinterpret_cast<float *>(out), topk, n);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) {
    moe_combine_kernel<__nv_bfloat16, BLOCK><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(routed), scores,
        reinterpret_cast<__nv_bfloat16 *>(out), topk, n);
  }
#endif
  else {
    fprintf(stderr, "moe_combine_f32: unsupported dtype %d\n", dtype);
  }
}
