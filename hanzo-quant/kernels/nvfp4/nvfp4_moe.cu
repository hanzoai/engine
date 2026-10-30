/**
 * NVFP4 W4A4 GEMMs in the scaled domain, for stacked expert banks (and a dense layer as a bank of
 * one).
 *
 * Both operands arrive quantized: activations as E2M1 codes [rows, K/2] with E4M3 block scales
 * [rows, K/16] (from kernels/quantize/nvfp4.cu), weights as codes [E, N, K/2] with E4M3 scales
 * [E, N, K/16]. A code times its E4M3 scale is exact in bf16 and f16 (at most 5 significant bits,
 * exponents 2^-10 .. 2^12), so both expand to exact 16-bit tiles; the products accumulate in f32,
 * are multiplied once by the expert's alpha (activation input_scale x weight_scale_2), and are
 * stored in the output dtype. This equals an FP4 tensor-core multiply up to summation order.
 *
 *   indexed vecmat: one warp per output, for decode (few rows).
 *   grouped WMMA:   one block per (N tile, expert) over the rows routed to that expert, found
 *                   through moe_dispatch_build's expert bounds and sorted assignment ids.
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda::wmma;

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

namespace nvfp4_moe {

constexpr int BLOCK = 16; // values per E4M3 scale

__device__ __forceinline__ float e2m1(uint8_t code) {
  const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float m = mag[code & 7];
  return (code & 8) ? -m : m;
}

__device__ __forceinline__ float e4m3(uint8_t v) {
  __nv_fp8_e4m3 f;
  f.__x = v;
  return static_cast<float>(f);
}

template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ half from_f32<half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }

/// The row of activations an assignment reads: its own row when the input carries the topk dim.
__device__ __forceinline__ long input_row(long work, int topk, bool input_has_topk_dim) {
  return input_has_topk_dim ? work : work / topk;
}

// ---------------------------------------------------------------------------
// Indexed vecmat: one warp per (assignment, n). Lane l takes blocks l, l+32, ...
// ---------------------------------------------------------------------------
template <typename T>
__global__ void vecmat(const uint8_t *__restrict__ a_codes, const uint8_t *__restrict__ a_scales,
                       const uint8_t *__restrict__ w_codes, const uint8_t *__restrict__ w_scales,
                       const float *__restrict__ alpha, const uint32_t *__restrict__ indices,
                       T *__restrict__ out, int work_items, int topk, int experts, int N, int K,
                       bool input_has_topk_dim) {
  const long warp = ((long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  const int n = warp % N;
  const long work = warp / N;
  if (work >= work_items)
    return;
  const uint32_t e = __ldg(&indices[work]);
  if (e >= (uint32_t)experts)
    return;
  const long row = input_row(work, topk, input_has_topk_dim);
  const int blocks = K / BLOCK;
  const uint8_t *ac = a_codes + row * (K / 2);
  const uint8_t *as = a_scales + row * blocks;
  const uint8_t *wc = w_codes + ((size_t)e * N + n) * (K / 2);
  const uint8_t *ws = w_scales + ((size_t)e * N + n) * blocks;
  float acc = 0.0f;
  for (int b = lane; b < blocks; b += 32) {
    const float sa = e4m3(__ldg(&as[b]));
    const float sw = e4m3(__ldg(&ws[b]));
    const uint2 av = __ldg(reinterpret_cast<const uint2 *>(ac + b * (BLOCK / 2)));
    const uint2 wv = __ldg(reinterpret_cast<const uint2 *>(wc + b * (BLOCK / 2)));
#pragma unroll
    for (int j = 0; j < BLOCK; j++) {
      const uint32_t aw = j < 8 ? av.x : av.y;
      const uint32_t ww = j < 8 ? wv.x : wv.y;
      const int sh = (j % 8) * 4;
      acc += (e2m1((aw >> sh) & 15) * sa) * (e2m1((ww >> sh) & 15) * sw);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off /= 2)
    acc += __shfl_down_sync(0xffffffff, acc, off);
  if (lane == 0)
    out[work * N + n] = from_f32<T>(acc * __ldg(&alpha[e]));
}

// ---------------------------------------------------------------------------
// Grouped WMMA: 64x64 output tiles, K staged 32 at a time (two scale blocks), 8 warps (4x2),
// each warp two 16x16 fragments along N.
// ---------------------------------------------------------------------------
constexpr int M_BLK = 64;
constexpr int N_BLK = 64;
constexpr int K_BLK = 32;
constexpr int WARPS_N = 2;
constexpr int THREADS = 256;
constexpr int W = 16; // WMMA tile

// Tiles are bf16 whatever the output dtype: a code times its scale is exact there.
template <typename Out>
__launch_bounds__(THREADS) __global__
    void grouped(const uint8_t *__restrict__ a_codes, const uint8_t *__restrict__ a_scales,
                 const uint8_t *__restrict__ w_codes, const uint8_t *__restrict__ w_scales,
                 const float *__restrict__ alpha, const uint32_t *__restrict__ bounds,
                 const uint32_t *__restrict__ sorted_work, Out *__restrict__ out, int topk, int N,
                 int K, bool input_has_topk_dim) {
  using T = __nv_bfloat16;
  __shared__ T A_sh[M_BLK * K_BLK];
  __shared__ T B_sh[N_BLK * K_BLK];
  __shared__ float C_sh[M_BLK * N_BLK];

  const int e = blockIdx.y;
  const int n_base = blockIdx.x * N_BLK;
  const uint32_t lo = __ldg(&bounds[e]);
  const int rows = (int)(__ldg(&bounds[e + 1]) - lo);
  if (rows == 0)
    return;
  const int blocks = K / BLOCK;
  const uint8_t *wc = w_codes + (size_t)e * N * (K / 2);
  const uint8_t *ws = w_scales + (size_t)e * N * blocks;
  const float a_e = __ldg(&alpha[e]);
  const int warp = threadIdx.x / 32;
  const int wm = warp / WARPS_N;
  const int wn = warp % WARPS_N;

  for (int m_base = 0; m_base < rows; m_base += M_BLK) {
    fragment<accumulator, W, W, W, float> c[2];
    fill_fragment(c[0], 0.0f);
    fill_fragment(c[1], 0.0f);
    for (int k0 = 0; k0 < K; k0 += K_BLK) {
      for (int i = threadIdx.x; i < M_BLK * K_BLK; i += THREADS) {
        const int lm = i / K_BLK, lk = i % K_BLK;
        float v = 0.0f;
        if (m_base + lm < rows) {
          const long row = input_row(__ldg(&sorted_work[lo + m_base + lm]), topk,
                                     input_has_topk_dim);
          const int k = k0 + lk;
          const uint8_t byte = __ldg(&a_codes[row * (K / 2) + k / 2]);
          v = e2m1((k & 1) ? byte >> 4 : byte & 15) *
              e4m3(__ldg(&a_scales[row * blocks + k / BLOCK]));
        }
        A_sh[i] = from_f32<T>(v);
      }
      for (int i = threadIdx.x; i < N_BLK * K_BLK; i += THREADS) {
        const int ln = i / K_BLK, lk = i % K_BLK;
        const int gn = n_base + ln;
        float v = 0.0f;
        if (gn < N) {
          const int k = k0 + lk;
          const uint8_t byte = __ldg(&wc[(size_t)gn * (K / 2) + k / 2]);
          v = e2m1((k & 1) ? byte >> 4 : byte & 15) *
              e4m3(__ldg(&ws[(size_t)gn * blocks + k / BLOCK]));
        }
        B_sh[i] = from_f32<T>(v);
      }
      __syncthreads();
#pragma unroll
      for (int ks = 0; ks < K_BLK / W; ks++) {
        fragment<matrix_a, W, W, W, T, row_major> a;
        load_matrix_sync(a, A_sh + wm * W * K_BLK + ks * W, K_BLK);
#pragma unroll
        for (int s = 0; s < 2; s++) {
          fragment<matrix_b, W, W, W, T, col_major> b;
          load_matrix_sync(b, B_sh + (wn * 2 + s) * W * K_BLK + ks * W, K_BLK);
          mma_sync(c[s], a, b, c[s]);
        }
      }
      __syncthreads();
    }
    for (int s = 0; s < 2; s++)
      store_matrix_sync(C_sh + wm * W * N_BLK + (wn * 2 + s) * W, c[s], N_BLK, mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < M_BLK * N_BLK; i += THREADS) {
      const int lm = i / N_BLK, ln = i % N_BLK;
      const int gn = n_base + ln;
      if (m_base + lm < rows && gn < N) {
        const long work = __ldg(&sorted_work[lo + m_base + lm]);
        out[work * N + gn] = from_f32<Out>(C_sh[i] * a_e);
      }
    }
    __syncthreads();
  }
}

} // namespace nvfp4_moe

#define NVFP4_MOE_API(NAME, T)                                                                   \
  extern "C" void launch_nvfp4_moe_vecmat_##NAME(                                                \
      const uint8_t *a_codes, const uint8_t *a_scales, const uint8_t *w_codes,                   \
      const uint8_t *w_scales, const float *alpha, const uint32_t *indices, T *out,             \
      int work_items, int topk, int experts, int N, int K, bool input_has_topk_dim,             \
      cudaStream_t stream) {                                                                     \
    const long threads = (long)work_items * N * 32;                                              \
    nvfp4_moe::vecmat<T><<<CEILDIV(threads, 256), 256, 0, stream>>>(                             \
        a_codes, a_scales, w_codes, w_scales, alpha, indices, out, work_items, topk, experts, N, \
        K, input_has_topk_dim);                                                                  \
    CUDA_CHECK(cudaGetLastError());                                                              \
  }

NVFP4_MOE_API(f16, half)
NVFP4_MOE_API(bf16, __nv_bfloat16)
NVFP4_MOE_API(f32, float)

#define NVFP4_GROUPED_API(NAME, T)                                                               \
  extern "C" void launch_nvfp4_moe_grouped_##NAME(                                               \
      const uint8_t *a_codes, const uint8_t *a_scales, const uint8_t *w_codes,                   \
      const uint8_t *w_scales, const float *alpha, const uint32_t *bounds,                      \
      const uint32_t *sorted_work, T *out, int experts, int topk, int N, int K,                 \
      bool input_has_topk_dim, cudaStream_t stream) {                                            \
    dim3 grid(CEILDIV(N, nvfp4_moe::N_BLK), experts);                                            \
    nvfp4_moe::grouped<T><<<grid, nvfp4_moe::THREADS, 0, stream>>>(                              \
        a_codes, a_scales, w_codes, w_scales, alpha, bounds, sorted_work, out, topk, N, K,       \
        input_has_topk_dim);                                                                     \
    CUDA_CHECK(cudaGetLastError());                                                              \
  }

NVFP4_GROUPED_API(f16, half)
NVFP4_GROUPED_API(bf16, __nv_bfloat16)
NVFP4_GROUPED_API(f32, float)
