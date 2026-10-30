/**
 * @brief W8A8 GEMM kernels for block-FP8 weights and per-group FP8 activations.
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace fp8_gemm {

// ============================================================================
// Helper functions
// ============================================================================

__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 val) {
  return __half2float(__nv_cvt_fp8_to_halfraw(val.__x, __NV_E4M3));
}

__device__ __forceinline__ float get_scale(const float *__restrict__ scale,
                                           int n, int k, int scale_stride,
                                           int block_size_y, int block_size_x) {
  int sr = n / block_size_y;
  int sc = k / block_size_x;
  return __ldg(&scale[sr * scale_stride + sc]);
}

// ============================================================================
// W8A8: E4M3 activations (per token, per 128-wide group, scale s_a) times E4M3
// weights (128x128 blocks, scale s_w). Every 128-deep K block is an f32 dot of
// the codes, scaled once by s_a * s_w and added into an f32 accumulator, as
// the served cutlass_scaled_mm does; one store in the output dtype.
// ============================================================================

constexpr int KB = 128; // K per activation group and per weight block

template <typename T> __device__ __forceinline__ void store(T *p, float v);
template <> __device__ __forceinline__ void store<float>(float *p, float v) { *p = v; }
template <> __device__ __forceinline__ void store<half>(half *p, float v) {
  *p = __float2half(v);
}
template <> __device__ __forceinline__ void store<__nv_bfloat16>(__nv_bfloat16 *p, float v) {
  *p = __float2bfloat16(v);
}

__device__ __forceinline__ float code(uint8_t v) {
  __nv_fp8_e4m3 f;
  f.__x = v;
  return fp8_to_float(f);
}

/// One output per thread, 32x32 tiles; K staged 32 at a time through shared memory.
template <typename T>
__global__ void w8a8_tiled(const uint8_t *__restrict__ qa, const float *__restrict__ sa,
                           const uint8_t *__restrict__ w, const float *__restrict__ sw,
                           T *__restrict__ out, int M, int N, int K, int sw_stride,
                           int block_y) {
  constexpr int TILE = 32;
  __shared__ float s_a[TILE][TILE + 1];
  __shared__ float s_w[TILE][TILE + 1];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;
  const int groups = K / KB;
  float acc = 0.0f;
  for (int kb = 0; kb < groups; kb++) {
    float partial = 0.0f;
    for (int kt = 0; kt < KB; kt += TILE) {
      const int k0 = kb * KB + kt;
      const int ar = blockIdx.y * TILE + ty;
      const int wr = blockIdx.x * TILE + ty;
      s_a[ty][tx] = ar < M ? code(qa[(size_t)ar * K + k0 + tx]) : 0.0f;
      s_w[ty][tx] = wr < N ? code(w[(size_t)wr * K + k0 + tx]) : 0.0f;
      __syncthreads();
#pragma unroll
      for (int k = 0; k < TILE; k++)
        partial += s_a[ty][k] * s_w[tx][k];
      __syncthreads();
    }
    if (row < M && col < N)
      acc += partial * (sa[(size_t)row * groups + kb] * sw[(col / block_y) * sw_stride + kb]);
  }
  if (row < M && col < N)
    store<T>(&out[(size_t)row * N + col], acc);
}

/// One warp per output: lane l holds k = 4l..4l+3 of every 128 block.
template <typename T>
__device__ __forceinline__ float w8a8_warp_dot(const uint8_t *__restrict__ a_row,
                                               const float *__restrict__ a_scale,
                                               const uint8_t *__restrict__ w_row,
                                               const float *__restrict__ w_scale, int K,
                                               int lane) {
  const int groups = K / KB;
  float acc = 0.0f;
  for (int kb = 0; kb < groups; kb++) {
    const int k = kb * KB + lane * 4;
    const uint32_t a4 = __ldg(reinterpret_cast<const uint32_t *>(&a_row[k]));
    const uint32_t w4 = __ldg(reinterpret_cast<const uint32_t *>(&w_row[k]));
    float partial = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++)
      partial += code((a4 >> (8 * j)) & 0xFF) * code((w4 >> (8 * j)) & 0xFF);
    acc += partial * (__ldg(&a_scale[kb]) * __ldg(&w_scale[kb]));
  }
#pragma unroll
  for (int off = 16; off > 0; off /= 2)
    acc += __shfl_down_sync(0xffffffff, acc, off);
  return acc;
}

template <typename T>
__global__ void w8a8_warp(const uint8_t *__restrict__ qa, const float *__restrict__ sa,
                          const uint8_t *__restrict__ w, const float *__restrict__ sw,
                          T *__restrict__ out, int M, int N, int K, int sw_stride, int block_y) {
  const long warp = ((long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (warp >= (long)M * N)
    return;
  const int row = warp / N;
  const int col = warp % N;
  const int groups = K / KB;
  const float acc =
      w8a8_warp_dot<T>(qa + (size_t)row * K, sa + (size_t)row * groups, w + (size_t)col * K,
                       sw + (size_t)(col / block_y) * sw_stride, K, lane);
  if (lane == 0)
    store<T>(&out[(size_t)row * N + col], acc);
}

/// Indexed MoE: output [tokens, topk, N]; activation row `token` (or `token * topk + slot`
/// when the input carries the topk dim), expert `indices[token, slot]`.
template <typename T>
__global__ void w8a8_moe(const uint8_t *__restrict__ qa, const float *__restrict__ sa,
                         const uint8_t *__restrict__ w, const float *__restrict__ sw,
                         const uint32_t *__restrict__ indices, T *__restrict__ out,
                         int tokens, int topk, int experts, int N, int K, int sw_stride,
                         int block_y, bool input_has_topk_dim) {
  const long warp = ((long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  const int n = warp % N;
  const long t = warp / N;
  const int slot = t % topk;
  const long token = t / topk;
  if (token >= tokens)
    return;
  const uint32_t e = __ldg(&indices[token * topk + slot]);
  if (e >= (uint32_t)experts)
    return;
  const long a_row = input_has_topk_dim ? token * topk + slot : token;
  const int groups = K / KB;
  const size_t sw_expert = (size_t)((N + block_y - 1) / block_y) * sw_stride;
  const float acc = w8a8_warp_dot<T>(
      qa + a_row * K, sa + a_row * groups, w + ((size_t)e * N + n) * K,
      sw + e * sw_expert + (size_t)(n / block_y) * sw_stride, K, lane);
  if (lane == 0)
    store<T>(&out[((size_t)token * topk + slot) * N + n], acc);
}

} // namespace fp8_gemm

// ============================================================================
// C API
// ============================================================================

namespace {
template <typename T>
void launch_matmul(const uint8_t *qa, const float *sa, const uint8_t *w, const float *sw, T *out,
                   int M, int N, int K, int sw_stride, int block_y, cudaStream_t stream) {
  if (M <= 16) {
    const long threads = (long)M * N * 32;
    fp8_gemm::w8a8_warp<T><<<CEILDIV(threads, 256), 256, 0, stream>>>(qa, sa, w, sw, out, M, N,
                                                                      K, sw_stride, block_y);
  } else {
    dim3 block(32, 32);
    dim3 grid(CEILDIV(N, 32), CEILDIV(M, 32));
    fp8_gemm::w8a8_tiled<T><<<grid, block, 0, stream>>>(qa, sa, w, sw, out, M, N, K, sw_stride,
                                                        block_y);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_moe(const uint8_t *qa, const float *sa, const uint8_t *w, const float *sw,
                const uint32_t *indices, T *out, int tokens, int topk, int experts, int N, int K,
                int sw_stride, int block_y, bool input_has_topk_dim, cudaStream_t stream) {
  const long threads = (long)tokens * topk * N * 32;
  fp8_gemm::w8a8_moe<T><<<CEILDIV(threads, 512), 512, 0, stream>>>(
      qa, sa, w, sw, indices, out, tokens, topk, experts, N, K, sw_stride, block_y,
      input_has_topk_dim);
  CUDA_CHECK(cudaGetLastError());
}
} // namespace

#define W8A8_API(NAME, T)                                                                      \
  extern "C" void launch_fp8_matmul_##NAME(const uint8_t *qa, const float *sa, const uint8_t *w, \
                                           const float *sw, T *out, int M, int N, int K,        \
                                           int sw_stride, int block_y, cudaStream_t stream) {   \
    launch_matmul<T>(qa, sa, w, sw, out, M, N, K, sw_stride, block_y, stream);                 \
  }                                                                                            \
  extern "C" void launch_fp8_indexed_moe_gemm_##NAME(                                          \
      const uint8_t *qa, const float *sa, const uint8_t *w, const float *sw,                   \
      const uint32_t *indices, T *out, int tokens, int topk, int experts, int N, int K,        \
      int sw_stride, int block_y, bool input_has_topk_dim, cudaStream_t stream) {              \
    launch_moe<T>(qa, sa, w, sw, indices, out, tokens, topk, experts, N, K, sw_stride, block_y, \
                  input_has_topk_dim, stream);                                                 \
  }

W8A8_API(f32, float)
W8A8_API(f16, half)
W8A8_API(bf16, __nv_bfloat16)
