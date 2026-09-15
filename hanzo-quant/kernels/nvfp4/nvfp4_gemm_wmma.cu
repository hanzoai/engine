/**
 * NVFP4 GEMM with WMMA tensor cores (compute >= 80).
 *
 * Same shape as the MXFP4 WMMA kernel: dequantize a weight tile into shared
 * memory with vectorized uint4 loads and the E2M1 LUT, then run 16x16x16 MMA.
 * NVFP4 scales are FP8 E4M3 per 16 weights with one FP32 scale for the tensor,
 * so each uint4 row load covers two blocks and reads two scales.
 *
 * Block tile 64x64x32, 8 warps (4x2), 256 threads; each warp owns one 16-row
 * M sub-tile and two 16-column N sub-tiles.
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <type_traits>

using namespace nvcuda::wmma;

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define NVFP4_BLOCK_SIZE 16

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

namespace nvfp4_wmma {

__device__ __forceinline__ float e4m3_to_float(uint8_t v) {
  const uint32_t sign = (uint32_t)(v & 0x80) << 24;
  const uint32_t exp = (uint32_t)((v >> 3) & 0x0F);
  const uint32_t man = (uint32_t)(v & 0x07);
  if (exp == 0) {
    const float sub = (float)man * 0.001953125f; // 2^-9
    return (v & 0x80) ? -sub : sub;
  }
  return __uint_as_float(sign | ((exp + 120u) << 23) | (man << 20));
}

__device__ __forceinline__ int2 get_int_from_table_16(const int q4,
                                                      const uint32_t table0,
                                                      const uint32_t table1,
                                                      const uint32_t table2,
                                                      const uint32_t table3) {
  uint32_t tmp[2];
  const uint32_t low_high_selection = 0x32103210 | ((q4 & 0x88888888) >> 1);

#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t shift = 16 * i;
    const uint32_t low = __byte_perm(table0, table1, q4 >> shift);
    const uint32_t high = __byte_perm(table2, table3, q4 >> shift);
    tmp[i] = __byte_perm(low, high, low_high_selection >> shift);
  }

  return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420),
                   __byte_perm(tmp[0], tmp[1], 0x7531));
}

__device__ __forceinline__ void dequant_store_8_f16(int q4, float scale,
                                                    uint32_t L0, uint32_t L1,
                                                    uint32_t L2, uint32_t L3,
                                                    half *dst) {
  int2 w = get_int_from_table_16(q4, L0, L1, L2, L3);
  dst[0] = __float2half((float)(int8_t)(w.x) * scale);
  dst[1] = __float2half((float)(int8_t)(w.y) * scale);
  dst[2] = __float2half((float)(int8_t)(w.x >> 8) * scale);
  dst[3] = __float2half((float)(int8_t)(w.y >> 8) * scale);
  dst[4] = __float2half((float)(int8_t)(w.x >> 16) * scale);
  dst[5] = __float2half((float)(int8_t)(w.y >> 16) * scale);
  dst[6] = __float2half((float)(int8_t)(w.x >> 24) * scale);
  dst[7] = __float2half((float)(int8_t)(w.y >> 24) * scale);
}

__device__ __forceinline__ void dequant_store_8_bf16(int q4, float scale,
                                                     uint32_t L0, uint32_t L1,
                                                     uint32_t L2, uint32_t L3,
                                                     __nv_bfloat16 *dst) {
  int2 w = get_int_from_table_16(q4, L0, L1, L2, L3);
  dst[0] = __float2bfloat16((float)(int8_t)(w.x) * scale);
  dst[1] = __float2bfloat16((float)(int8_t)(w.y) * scale);
  dst[2] = __float2bfloat16((float)(int8_t)(w.x >> 8) * scale);
  dst[3] = __float2bfloat16((float)(int8_t)(w.y >> 8) * scale);
  dst[4] = __float2bfloat16((float)(int8_t)(w.x >> 16) * scale);
  dst[5] = __float2bfloat16((float)(int8_t)(w.y >> 16) * scale);
  dst[6] = __float2bfloat16((float)(int8_t)(w.x >> 24) * scale);
  dst[7] = __float2bfloat16((float)(int8_t)(w.y >> 24) * scale);
}

template <typename T>
__device__ __forceinline__ void
dequant_store_8(int q4, float scale, uint32_t L0, uint32_t L1, uint32_t L2,
                uint32_t L3, T *dst);

template <>
__device__ __forceinline__ void
dequant_store_8<half>(int q4, float scale, uint32_t L0, uint32_t L1,
                      uint32_t L2, uint32_t L3, half *dst) {
  dequant_store_8_f16(q4, scale, L0, L1, L2, L3, dst);
}

template <>
__device__ __forceinline__ void
dequant_store_8<__nv_bfloat16>(int q4, float scale, uint32_t L0, uint32_t L1,
                               uint32_t L2, uint32_t L3, __nv_bfloat16 *dst) {
  dequant_store_8_bf16(q4, scale, L0, L1, L2, L3, dst);
}

constexpr int WMMA_M_DIM = 16;
constexpr int WMMA_N_DIM = 16;
constexpr int WMMA_K_DIM = 16;

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;  // 8
constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * 32; // 256

constexpr int M_BLK = WARPS_M * WMMA_M_DIM;      // 64
constexpr int N_BLK = WARPS_N * 2 * WMMA_N_DIM;  // 64
// One uint4 covers 32 weights. Two per row would halve the A-tile reloads and the barriers,
// and measured 15-22% SLOWER in-engine on a GB10 (254 vs 297 T/s at pp178, 280 vs 358 at
// pp1291, 255 vs 310 at pp12452): the extra 16 KB of shared memory costs more occupancy than
// the reuse buys. Keep one.
constexpr int PAIRS_PER_ROW = 1;
constexpr int K_BLK = NVFP4_BLOCK_SIZE * 2 * PAIRS_PER_ROW; // 32
constexpr int WMMA_K_STEPS = K_BLK / WMMA_K_DIM;            // 2

using VecT = float4;
constexpr int VEC_SIZE = 8; // float4 = 16 bytes = 8 fp16/bf16 values

template <typename T>
__launch_bounds__(BLOCK_THREADS) __global__
    void nvfp4_matmul_wmma_kernel(const T *__restrict__ input,
                                  const uint8_t *__restrict__ weight,
                                  const uint8_t *__restrict__ weight_scale,
                                  float global_scale, const T *__restrict__ bias,
                                  T *__restrict__ output, int M, int N, int K,
                                  bool has_bias) {
  // The table holds 2x-scaled codebook values; the 0.5f goes back into the scale.
  const uint32_t LUT0 = 0x03020100;
  const uint32_t LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00;
  const uint32_t LUT3 = 0xF4F8FAFC;

  const int scale_stride = K / NVFP4_BLOCK_SIZE;

  extern __shared__ uint8_t smem_bytes[];

  T *A_sh = reinterpret_cast<T *>(smem_bytes);
  T *B_sh = A_sh + M_BLK * K_BLK;
  uint8_t *C_raw = reinterpret_cast<uint8_t *>(B_sh + N_BLK * K_BLK);
  size_t align_off = reinterpret_cast<uintptr_t>(C_raw) % alignof(float);
  if (align_off != 0)
    C_raw += (alignof(float) - align_off);
  float *C_sh = reinterpret_cast<float *>(C_raw);

  const int threadId = threadIdx.x;
  const int warpId = threadId / 32;
  const int warp_m_idx = warpId / WARPS_N;
  const int warp_n_idx = warpId % WARPS_N;

  const int m_base = blockIdx.y * M_BLK;
  const int n_base = blockIdx.x * N_BLK;

  VecT zero_vec;
  zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

  fragment<accumulator, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, float> c_frag[2];
  fill_fragment(c_frag[0], 0.0f);
  fill_fragment(c_frag[1], 0.0f);

  for (int k_base = 0; k_base < K; k_base += K_BLK) {
    constexpr int A_VEC_ELEMS = M_BLK * K_BLK / VEC_SIZE;
    for (int i = threadId; i < A_VEC_ELEMS; i += BLOCK_THREADS) {
      const int idx = i * VEC_SIZE;
      const int lm = idx / K_BLK;
      const int lk = idx % K_BLK;
      const int gm = m_base + lm;
      const int gk = k_base + lk;

      if (gm < M && gk < K) {
        *reinterpret_cast<VecT *>(&A_sh[lm * K_BLK + lk]) =
            *reinterpret_cast<const VecT *>(&input[(size_t)gm * K + gk]);
      } else {
        *reinterpret_cast<VecT *>(&A_sh[lm * K_BLK + lk]) = zero_vec;
      }
    }

    for (int ln = threadId; ln < N_BLK; ln += BLOCK_THREADS) {
      const int gn = n_base + ln;
      T *dst = &B_sh[ln * K_BLK];
      if (gn < N) {
        const size_t w_row = (size_t)gn * (K / 2) + k_base / 2;
        const size_t s_row =
            (size_t)gn * scale_stride + k_base / NVFP4_BLOCK_SIZE;
#pragma unroll
        for (int p = 0; p < PAIRS_PER_ROW; p++) {
          uint4 w_vec =
              *reinterpret_cast<const uint4 *>(&weight[w_row + p * 16]);
          const float s0 = e4m3_to_float(__ldg(&weight_scale[s_row + p * 2])) *
                           global_scale * 0.5f;
          const float s1 =
              e4m3_to_float(__ldg(&weight_scale[s_row + p * 2 + 1])) *
              global_scale * 0.5f;
          T *d = dst + p * 32;
          dequant_store_8<T>(w_vec.x, s0, LUT0, LUT1, LUT2, LUT3, d);
          dequant_store_8<T>(w_vec.y, s0, LUT0, LUT1, LUT2, LUT3, d + 8);
          dequant_store_8<T>(w_vec.z, s1, LUT0, LUT1, LUT2, LUT3, d + 16);
          dequant_store_8<T>(w_vec.w, s1, LUT0, LUT1, LUT2, LUT3, d + 24);
        }
      } else {
#pragma unroll
        for (int k = 0; k < K_BLK; k++)
          dst[k] = T(0);
      }
    }

    __syncthreads();

#pragma unroll
    for (int k_step = 0; k_step < WMMA_K_STEPS; k_step++) {
      fragment<matrix_a, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, T, row_major>
          a_frag;
      const T *A_ptr =
          A_sh + warp_m_idx * WMMA_M_DIM * K_BLK + k_step * WMMA_K_DIM;
      load_matrix_sync(a_frag, A_ptr, K_BLK);

#pragma unroll
      for (int n_sub = 0; n_sub < 2; n_sub++) {
        fragment<matrix_b, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, T, col_major>
            b_frag;
        const T *B_ptr = B_sh + (warp_n_idx * 2 + n_sub) * WMMA_N_DIM * K_BLK +
                         k_step * WMMA_K_DIM;
        load_matrix_sync(b_frag, B_ptr, K_BLK);
        mma_sync(c_frag[n_sub], a_frag, b_frag, c_frag[n_sub]);
      }
    }

    __syncthreads();
  }

  for (int n_sub = 0; n_sub < 2; n_sub++) {
    float *C_ptr = C_sh + warp_m_idx * WMMA_M_DIM * N_BLK +
                   (warp_n_idx * 2 + n_sub) * WMMA_N_DIM;
    store_matrix_sync(C_ptr, c_frag[n_sub], N_BLK, mem_row_major);
  }
  __syncthreads();

  constexpr int C_ELEMS = M_BLK * N_BLK;
  for (int i = threadId; i < C_ELEMS; i += BLOCK_THREADS) {
    const int lm = i / N_BLK;
    const int ln = i % N_BLK;
    const int gm = m_base + lm;
    const int gn = n_base + ln;

    if (gm < M && gn < N) {
      float val = C_sh[lm * N_BLK + ln];
      if (has_bias && bias != nullptr) {
        if constexpr (std::is_same_v<T, half>) {
          val += __half2float(__ldg(&bias[gn]));
        } else {
          val += __bfloat162float(__ldg(&bias[gn]));
        }
      }
      if constexpr (std::is_same_v<T, half>) {
        output[(size_t)gm * N + gn] = __float2half(val);
      } else {
        output[(size_t)gm * N + gn] = __float2bfloat16(val);
      }
    }
  }
}

} // namespace nvfp4_wmma

static size_t nvfp4_wmma_smem_bytes() {
  using namespace nvfp4_wmma;
  size_t AB = (M_BLK * K_BLK + N_BLK * K_BLK) * 2;
  size_t pad = (16 - (AB % 16)) % 16;
  size_t C = M_BLK * N_BLK * sizeof(float);
  return AB + pad + C;
}

// Decode is bandwidth-bound and wastes 15 of 16 MMA slots, so it stays on the
// vecmat kernel in nvfp4_gemm.cu.
extern "C" void launch_nvfp4_matmul_f16(const __half *, const uint8_t *,
                                        const uint8_t *, float, const __half *,
                                        __half *, int, int, int, bool,
                                        cudaStream_t);
extern "C" void launch_nvfp4_matmul_bf16(const __nv_bfloat16 *, const uint8_t *,
                                         const uint8_t *, float,
                                         const __nv_bfloat16 *,
                                         __nv_bfloat16 *, int, int, int, bool,
                                         cudaStream_t);

extern "C" void
launch_nvfp4_matmul_wmma_f16(const __half *input, const uint8_t *weight,
                             const uint8_t *weight_scale, float global_scale,
                             const __half *bias, __half *output, int M, int N,
                             int K, bool has_bias, cudaStream_t stream) {
  if (M <= 4 || K % nvfp4_wmma::K_BLK != 0) {
    launch_nvfp4_matmul_f16(input, weight, weight_scale, global_scale, bias,
                            output, M, N, K, has_bias, stream);
    return;
  }
  using namespace nvfp4_wmma;

  dim3 grid(CEILDIV(N, N_BLK), CEILDIV(M, M_BLK));
  dim3 block(BLOCK_THREADS);
  nvfp4_wmma::nvfp4_matmul_wmma_kernel<half>
      <<<grid, block, nvfp4_wmma_smem_bytes(), stream>>>(
          input, weight, weight_scale, global_scale, bias, output, M, N, K,
          has_bias);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_nvfp4_matmul_wmma_bf16(
    const __nv_bfloat16 *input, const uint8_t *weight,
    const uint8_t *weight_scale, float global_scale, const __nv_bfloat16 *bias,
    __nv_bfloat16 *output, int M, int N, int K, bool has_bias,
    cudaStream_t stream) {
  if (M <= 4 || K % nvfp4_wmma::K_BLK != 0) {
    launch_nvfp4_matmul_bf16(input, weight, weight_scale, global_scale, bias,
                             output, M, N, K, has_bias, stream);
    return;
  }
  using namespace nvfp4_wmma;

  dim3 grid(CEILDIV(N, N_BLK), CEILDIV(M, M_BLK));
  dim3 block(BLOCK_THREADS);
  nvfp4_wmma::nvfp4_matmul_wmma_kernel<__nv_bfloat16>
      <<<grid, block, nvfp4_wmma_smem_bytes(), stream>>>(
          input, weight, weight_scale, global_scale, bias, output, M, N, K,
          has_bias);
  CUDA_CHECK(cudaGetLastError());
}
