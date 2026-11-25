/*
 * AFQ (Affine Fast Quantization) Fused GEMM CUDA Kernels
 *
 * Implements fused dequantization + matrix multiplication for optimal
 * performance. These kernels dequantize on-the-fly during the matmul to save
 * memory bandwidth.
 */

#include "afq_utils.cuh"
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================================
// Configuration
// ============================================================================

// QMV (quantized matrix-vector) kernel configuration
#define QMV_BLOCK_SIZE 256
#define QMV_ROWS_PER_BLOCK 4

// QMM (quantized matrix-matrix) kernel configuration
#define QMM_TILE_M 64
#define QMM_TILE_N 64
#define QMM_TILE_K 32

// ============================================================================
// Quantized Matrix-Vector Multiply (QMV)
// Computes: y = x @ W^T where W is quantized
// x: [M, K], W: [N, K] (quantized), y: [M, N]
// Optimized for small M (generation phase)
// ============================================================================

template <typename T> __device__ __forceinline__ float afq_to_f32(T v) {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(v);
  } else {
    return __bfloat162float(v);
  }
}

template <typename T> __device__ __forceinline__ T afq_from_f32(float v) {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __float2half(v);
  } else {
    return __float2bfloat16(v);
  }
}

// QMV kernel for power-of-2 bit widths
template <typename T, int bits, int group_size>
__global__ void
afq_qmv_kernel(const T *__restrict__ x, const uint32_t *__restrict__ w_q,
               const T *__restrict__ scales, const T *__restrict__ biases,
               T *__restrict__ y, int M, int N, int K) {
  constexpr int values_per_u32 = 32 / bits;
  const int packed_K = K * bits / 32;
  const int groups_per_row = K / group_size;

  int64_t warp_id =
      ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / AFQ_WARP_SIZE;
  int lane = threadIdx.x % AFQ_WARP_SIZE;

  int64_t total_outputs = (int64_t)M * N;
  if (warp_id >= total_outputs)
    return;

  int m = (int)(warp_id / N);
  int n = (int)(warp_id % N);

  const T *x_row = x + (size_t)m * K;
  const uint32_t *w_row = w_q + (size_t)n * packed_K;
  const T *scales_row = scales + (size_t)n * groups_per_row;
  const T *biases_row = biases + (size_t)n * groups_per_row;

  float acc = 0.0f;

  for (int p = lane; p < packed_K; p += AFQ_WARP_SIZE) {
    uint32_t packed = w_row[p];
    int base_k = p * values_per_u32;
    int group_idx = base_k / group_size;
    float scale = afq_to_f32(scales_row[group_idx]);
    float bias = afq_to_f32(biases_row[group_idx]);

    if constexpr (sizeof(T) == 2 && values_per_u32 == 16) {
      if ((((uintptr_t)(x_row + base_k)) & 15u) == 0u) {
        uint4 v0 = *reinterpret_cast<const uint4 *>(x_row + base_k);
        uint4 v1 = *reinterpret_cast<const uint4 *>(x_row + base_k + 8);
        T xv[16];
        *reinterpret_cast<uint4 *>(xv) = v0;
        *reinterpret_cast<uint4 *>(xv + 8) = v1;
#pragma unroll
        for (int i = 0; i < 16; i++) {
          uint32_t q = (packed >> (2 * i)) & 3u;
          acc = fmaf(afq_to_f32(xv[i]), (float)q * scale + bias, acc);
        }
        continue;
      }
    }

#pragma unroll
    for (int i = 0; i < values_per_u32; i++) {
      uint32_t q = extract_bits<bits>(packed, i);
      float x_val = afq_to_f32(x_row[base_k + i]);
      acc = fmaf(x_val, (float)q * scale + bias, acc);
    }
  }

  acc = warp_reduce_sum(acc);

  if (lane == 0) {
    y[(size_t)m * N + n] = afq_from_f32<T>(acc);
  }
}

// QMV kernel for 3-bit (non-power-of-2)
template <typename T, int group_size>
__global__ void
afq_qmv_3bit_kernel(const T *__restrict__ x, const uint8_t *__restrict__ w_q,
                    const T *__restrict__ scales, const T *__restrict__ biases,
                    T *__restrict__ y, int M, int N, int K) {
  const int packed_K = (K * 3 + 7) / 8;
  const int groups_per_row = K / group_size;

  int64_t warp_id =
      ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / AFQ_WARP_SIZE;
  int lane = threadIdx.x % AFQ_WARP_SIZE;

  int64_t total_outputs = (int64_t)M * N;
  if (warp_id >= total_outputs)
    return;

  int m = (int)(warp_id / N);
  int n = (int)(warp_id % N);

  const T *x_row = x + (size_t)m * K;
  const uint8_t *w_row = w_q + (size_t)n * packed_K;
  const T *scales_row = scales + (size_t)n * groups_per_row;
  const T *biases_row = biases + (size_t)n * groups_per_row;

  float acc = 0.0f;

  for (int k = lane; k < K; k += AFQ_WARP_SIZE) {
    uint32_t q = extract_3bit(w_row, k);

    int group_idx = k / group_size;
    float scale = afq_to_f32(scales_row[group_idx]);
    float bias = afq_to_f32(biases_row[group_idx]);
    float w = (float)q * scale + bias;
    float x_val = afq_to_f32(x_row[k]);

    acc = fmaf(x_val, w, acc);
  }

  acc = warp_reduce_sum(acc);

  if (lane == 0) {
    y[(size_t)m * N + n] = afq_from_f32<T>(acc);
  }
}

// QMV kernel for 6-bit (non-power-of-2)
template <typename T, int group_size>
__global__ void
afq_qmv_6bit_kernel(const T *__restrict__ x, const uint8_t *__restrict__ w_q,
                    const T *__restrict__ scales, const T *__restrict__ biases,
                    T *__restrict__ y, int M, int N, int K) {
  const int packed_K = (K * 6 + 7) / 8;
  const int groups_per_row = K / group_size;

  int64_t warp_id =
      ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / AFQ_WARP_SIZE;
  int lane = threadIdx.x % AFQ_WARP_SIZE;

  int64_t total_outputs = (int64_t)M * N;
  if (warp_id >= total_outputs)
    return;

  int m = (int)(warp_id / N);
  int n = (int)(warp_id % N);

  const T *x_row = x + (size_t)m * K;
  const uint8_t *w_row = w_q + (size_t)n * packed_K;
  const T *scales_row = scales + (size_t)n * groups_per_row;
  const T *biases_row = biases + (size_t)n * groups_per_row;

  float acc = 0.0f;

  for (int k = lane; k < K; k += AFQ_WARP_SIZE) {
    uint32_t q = extract_6bit(w_row, k);

    int group_idx = k / group_size;
    float scale = afq_to_f32(scales_row[group_idx]);
    float bias = afq_to_f32(biases_row[group_idx]);
    float w = (float)q * scale + bias;
    float x_val = afq_to_f32(x_row[k]);

    acc = fmaf(x_val, w, acc);
  }

  acc = warp_reduce_sum(acc);

  if (lane == 0) {
    y[(size_t)m * N + n] = afq_from_f32<T>(acc);
  }
}

// ============================================================================
// Quantized Matrix-Matrix Multiply (QMM) - Tiled version
// Computes: y = x @ W^T where W is quantized
// x: [M, K], W: [N, K] (quantized), y: [M, N]
// Optimized for larger M (prefill phase)
// ============================================================================

// Tiled QMM kernel using shared memory
template <typename T, int bits, int group_size, int TILE_M = 32,
          int TILE_N = 32, int TILE_K = 32>
__global__ void
afq_qmm_kernel(const T *__restrict__ x, const uint32_t *__restrict__ w_q,
               const T *__restrict__ scales, const T *__restrict__ biases,
               T *__restrict__ y, int M, int N, int K) {
  constexpr int values_per_u32 = 32 / bits;
  const int packed_K = K * bits / 32;
  const int groups_per_row = K / group_size;

  // Shared memory for tiles
  __shared__ float x_tile[TILE_M][TILE_K];
  __shared__ float w_tile[TILE_N][TILE_K];

  int bm = blockIdx.y * TILE_M;
  int bn = blockIdx.x * TILE_N;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Accumulator
  float acc = 0.0f;

  // Process K dimension in tiles
  for (int bk = 0; bk < K; bk += TILE_K) {
    int m_idx = bm + ty;
    int k_idx = bk + tx;
    if (m_idx < M && k_idx < K) {
      x_tile[ty][tx] = afq_to_f32(x[(size_t)m_idx * K + k_idx]);
    } else {
      x_tile[ty][tx] = 0.0f;
    }

    int n_idx = bn + ty;
    if (n_idx < N && k_idx < K) {
      int packed_idx = k_idx / values_per_u32;
      int value_idx = k_idx % values_per_u32;
      uint32_t packed = w_q[(size_t)n_idx * packed_K + packed_idx];
      uint32_t q = extract_bits<bits>(packed, value_idx);

      int group_idx = k_idx / group_size;
      float scale = afq_to_f32(scales[(size_t)n_idx * groups_per_row + group_idx]);
      float bias = afq_to_f32(biases[(size_t)n_idx * groups_per_row + group_idx]);

      w_tile[ty][tx] = (float)q * scale + bias;
    } else {
      w_tile[ty][tx] = 0.0f;
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_K; k++) {
      acc = fmaf(x_tile[ty][k], w_tile[tx][k], acc);
    }

    __syncthreads();
  }

  int m_out = bm + ty;
  int n_out = bn + tx;
  if (m_out < M && n_out < N) {
    y[(size_t)m_out * N + n_out] = afq_from_f32<T>(acc);
  }
}

#define DEFINE_QMV_LAUNCHER(bits, gs, dtype, dtype_name)                       \
  extern "C" void afq_qmv_##bits##bit_gs##gs##_##dtype_name(                   \
      const dtype *x, const uint32_t *w_q, const dtype *scales,                \
      const dtype *biases, dtype *y, int M, int N, int K) {                    \
    constexpr int warps_per_block = QMV_BLOCK_SIZE / AFQ_WARP_SIZE;            \
    int64_t total_outputs = (int64_t)M * N;                                    \
    int blocks =                                                               \
        (int)((total_outputs + warps_per_block - 1) / warps_per_block);        \
    afq_qmv_kernel<dtype, bits, gs>                                            \
        <<<blocks, QMV_BLOCK_SIZE>>>(x, w_q, scales, biases, y, M, N, K);      \
  }

#define DEFINE_QMV_3BIT_LAUNCHER(gs, dtype, dtype_name)                        \
  extern "C" void afq_qmv_3bit_gs##gs##_##dtype_name(                          \
      const dtype *x, const uint8_t *w_q, const dtype *scales,                 \
      const dtype *biases, dtype *y, int M, int N, int K) {                    \
    constexpr int warps_per_block = QMV_BLOCK_SIZE / AFQ_WARP_SIZE;            \
    int64_t total_outputs = (int64_t)M * N;                                    \
    int blocks =                                                               \
        (int)((total_outputs + warps_per_block - 1) / warps_per_block);        \
    afq_qmv_3bit_kernel<dtype, gs>                                             \
        <<<blocks, QMV_BLOCK_SIZE>>>(x, w_q, scales, biases, y, M, N, K);      \
  }

#define DEFINE_QMV_6BIT_LAUNCHER(gs, dtype, dtype_name)                        \
  extern "C" void afq_qmv_6bit_gs##gs##_##dtype_name(                          \
      const dtype *x, const uint8_t *w_q, const dtype *scales,                 \
      const dtype *biases, dtype *y, int M, int N, int K) {                    \
    constexpr int warps_per_block = QMV_BLOCK_SIZE / AFQ_WARP_SIZE;            \
    int64_t total_outputs = (int64_t)M * N;                                    \
    int blocks =                                                               \
        (int)((total_outputs + warps_per_block - 1) / warps_per_block);        \
    afq_qmv_6bit_kernel<dtype, gs>                                             \
        <<<blocks, QMV_BLOCK_SIZE>>>(x, w_q, scales, biases, y, M, N, K);      \
  }

// 2-bit QMV launchers
DEFINE_QMV_LAUNCHER(2, 32, float, f32)
DEFINE_QMV_LAUNCHER(2, 64, float, f32)
DEFINE_QMV_LAUNCHER(2, 128, float, f32)
DEFINE_QMV_LAUNCHER(2, 32, __half, f16)
DEFINE_QMV_LAUNCHER(2, 64, __half, f16)
DEFINE_QMV_LAUNCHER(2, 128, __half, f16)

// 3-bit QMV launchers
DEFINE_QMV_3BIT_LAUNCHER(32, float, f32)
DEFINE_QMV_3BIT_LAUNCHER(64, float, f32)
DEFINE_QMV_3BIT_LAUNCHER(128, float, f32)
DEFINE_QMV_3BIT_LAUNCHER(32, __half, f16)
DEFINE_QMV_3BIT_LAUNCHER(64, __half, f16)
DEFINE_QMV_3BIT_LAUNCHER(128, __half, f16)

// 4-bit QMV launchers
DEFINE_QMV_LAUNCHER(4, 32, float, f32)
DEFINE_QMV_LAUNCHER(4, 64, float, f32)
DEFINE_QMV_LAUNCHER(4, 128, float, f32)
DEFINE_QMV_LAUNCHER(4, 32, __half, f16)
DEFINE_QMV_LAUNCHER(4, 64, __half, f16)
DEFINE_QMV_LAUNCHER(4, 128, __half, f16)

// 6-bit QMV launchers
DEFINE_QMV_6BIT_LAUNCHER(32, float, f32)
DEFINE_QMV_6BIT_LAUNCHER(64, float, f32)
DEFINE_QMV_6BIT_LAUNCHER(128, float, f32)
DEFINE_QMV_6BIT_LAUNCHER(32, __half, f16)
DEFINE_QMV_6BIT_LAUNCHER(64, __half, f16)
DEFINE_QMV_6BIT_LAUNCHER(128, __half, f16)

// 8-bit QMV launchers
DEFINE_QMV_LAUNCHER(8, 32, float, f32)
DEFINE_QMV_LAUNCHER(8, 64, float, f32)
DEFINE_QMV_LAUNCHER(8, 128, float, f32)
DEFINE_QMV_LAUNCHER(8, 32, __half, f16)
DEFINE_QMV_LAUNCHER(8, 64, __half, f16)
DEFINE_QMV_LAUNCHER(8, 128, __half, f16)

// BFloat16 QMV
DEFINE_QMV_LAUNCHER(2, 32, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(2, 64, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(2, 128, __nv_bfloat16, bf16)
DEFINE_QMV_3BIT_LAUNCHER(32, __nv_bfloat16, bf16)
DEFINE_QMV_3BIT_LAUNCHER(64, __nv_bfloat16, bf16)
DEFINE_QMV_3BIT_LAUNCHER(128, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(4, 32, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(4, 64, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(4, 128, __nv_bfloat16, bf16)
DEFINE_QMV_6BIT_LAUNCHER(32, __nv_bfloat16, bf16)
DEFINE_QMV_6BIT_LAUNCHER(64, __nv_bfloat16, bf16)
DEFINE_QMV_6BIT_LAUNCHER(128, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(8, 32, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(8, 64, __nv_bfloat16, bf16)
DEFINE_QMV_LAUNCHER(8, 128, __nv_bfloat16, bf16)

// ============================================================================
// Extern "C" Launch Functions - QMM (for larger batch sizes)
// ============================================================================

#define DEFINE_QMM_LAUNCHER(bits, gs, dtype, dtype_name)                       \
  extern "C" void afq_qmm_##bits##bit_gs##gs##_##dtype_name(                   \
      const dtype *x, const uint32_t *w_q, const dtype *scales,                \
      const dtype *biases, dtype *y, int M, int N, int K) {                    \
    constexpr int TILE_M = 32;                                                 \
    constexpr int TILE_N = 32;                                                 \
    constexpr int TILE_K = 32;                                                 \
    dim3 grid(cdiv(N, TILE_N), cdiv(M, TILE_M));                               \
    dim3 block(TILE_N, TILE_M);                                                \
    afq_qmm_kernel<dtype, bits, gs, TILE_M, TILE_N, TILE_K>                    \
        <<<grid, block>>>(x, w_q, scales, biases, y, M, N, K);                 \
  }

// 4-bit QMM launchers (most common)
DEFINE_QMM_LAUNCHER(4, 32, float, f32)
DEFINE_QMM_LAUNCHER(4, 64, float, f32)
DEFINE_QMM_LAUNCHER(4, 128, float, f32)
DEFINE_QMM_LAUNCHER(4, 32, __half, f16)
DEFINE_QMM_LAUNCHER(4, 64, __half, f16)
DEFINE_QMM_LAUNCHER(4, 128, __half, f16)

// 8-bit QMM launchers
DEFINE_QMM_LAUNCHER(8, 32, float, f32)
DEFINE_QMM_LAUNCHER(8, 64, float, f32)
DEFINE_QMM_LAUNCHER(8, 128, float, f32)
DEFINE_QMM_LAUNCHER(8, 32, __half, f16)
DEFINE_QMM_LAUNCHER(8, 64, __half, f16)
DEFINE_QMM_LAUNCHER(8, 128, __half, f16)

// 2-bit QMM launchers
DEFINE_QMM_LAUNCHER(2, 32, float, f32)
DEFINE_QMM_LAUNCHER(2, 64, float, f32)
DEFINE_QMM_LAUNCHER(2, 128, float, f32)
DEFINE_QMM_LAUNCHER(2, 32, __half, f16)
DEFINE_QMM_LAUNCHER(2, 64, __half, f16)
DEFINE_QMM_LAUNCHER(2, 128, __half, f16)

// BFloat16 QMM
DEFINE_QMM_LAUNCHER(4, 32, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(4, 64, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(4, 128, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(8, 32, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(8, 64, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(8, 128, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(2, 32, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(2, 64, __nv_bfloat16, bf16)
DEFINE_QMM_LAUNCHER(2, 128, __nv_bfloat16, bf16)
