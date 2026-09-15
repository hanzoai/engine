/**
 * NVFP4 GEMM kernels with LUT-based dequantization.
 *
 * NVFP4 (NVIDIA ModelOpt) is MXFP4's codebook with a finer, richer scale:
 * - FP4 E2M1 weights, 2 per byte, same 16-entry codebook as MXFP4
 * - Block size 16 (MXFP4 uses 32)
 * - Per-block scale in FP8 E4M3 (MXFP4 uses power-of-two E8M0)
 * - One FP32 scale for the whole tensor, folded into the block scale here
 *
 * The weights stay packed in memory and are expanded in registers, so decode
 * reads 4.5 bits per weight instead of the 16 a dequantized copy would cost.
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <type_traits>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define NVFP4_BLOCK_SIZE 16
#define NVFP4_PAIR 32 // two blocks: one uint4 weight load
#define WARP_SIZE 32
#define VECMAT_COLS 4
#define VECMAT_THREADS 256
#define VECMAT_WARPS (VECMAT_THREADS / WARP_SIZE)

namespace nvfp4_gemm {

// E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa. No infinity; 0x7F/0xFF are NaN.
// Normals map onto f32 by re-biasing the exponent; subnormals are m * 2^-9.
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

// Vectorized LUT lookup: 8 packed nibbles -> 8 int8 codebook entries.
// The table holds 2x-scaled values; callers fold the 0.5f back into the scale.
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

__device__ __forceinline__ void dequant_store_8(int q4, float scale,
                                                uint32_t LUT0, uint32_t LUT1,
                                                uint32_t LUT2, uint32_t LUT3,
                                                float *dst) {
  int2 w = get_int_from_table_16(q4, LUT0, LUT1, LUT2, LUT3);
  dst[0] = (float)(int8_t)(w.x) * scale;
  dst[1] = (float)(int8_t)(w.y) * scale;
  dst[2] = (float)(int8_t)(w.x >> 8) * scale;
  dst[3] = (float)(int8_t)(w.y >> 8) * scale;
  dst[4] = (float)(int8_t)(w.x >> 16) * scale;
  dst[5] = (float)(int8_t)(w.y >> 16) * scale;
  dst[6] = (float)(int8_t)(w.x >> 24) * scale;
  dst[7] = (float)(int8_t)(w.y >> 24) * scale;
}

#define NVFP4_LUT0 0x03020100
#define NVFP4_LUT1 0x0C080604
#define NVFP4_LUT2 0xFDFEFF00
#define NVFP4_LUT3 0xF4F8FAFC

// ============================================================================
// Decode path (M small): one row against VECMAT_COLS output columns.
// Batch-1 decode is memory-bound, so the win is reading packed bytes.
// ============================================================================

template <typename T>
__global__ void nvfp4_vecmat(const T *__restrict__ input,
                             const uint8_t *__restrict__ weight,
                             const uint8_t *__restrict__ weight_scale,
                             float global_scale, const T *__restrict__ bias,
                             T *__restrict__ output, int M, int N, int K,
                             bool has_bias) {
  const int row = blockIdx.y;
  const int col_base = blockIdx.x * VECMAT_COLS;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int scale_stride = K / NVFP4_BLOCK_SIZE;
  const int k_half = K / 2;

  __shared__ float s_reduce[VECMAT_COLS][VECMAT_WARPS];

  float acc[VECMAT_COLS];
#pragma unroll
  for (int c = 0; c < VECMAT_COLS; c++)
    acc[c] = 0.0f;

  const int num_pairs = K / NVFP4_PAIR;

  for (int pair = tid; pair < num_pairs; pair += VECMAT_THREADS) {
    const int k_start = pair * NVFP4_PAIR;

    float in_vals[NVFP4_PAIR];
    const T *in_ptr = input + (size_t)row * K + k_start;
#pragma unroll
    for (int i = 0; i < NVFP4_PAIR; i += 4) {
      if constexpr (std::is_same_v<T, half>) {
        in_vals[i + 0] = __half2float(__ldg(&in_ptr[i + 0]));
        in_vals[i + 1] = __half2float(__ldg(&in_ptr[i + 1]));
        in_vals[i + 2] = __half2float(__ldg(&in_ptr[i + 2]));
        in_vals[i + 3] = __half2float(__ldg(&in_ptr[i + 3]));
      } else {
        in_vals[i + 0] = __bfloat162float(__ldg(&in_ptr[i + 0]));
        in_vals[i + 1] = __bfloat162float(__ldg(&in_ptr[i + 1]));
        in_vals[i + 2] = __bfloat162float(__ldg(&in_ptr[i + 2]));
        in_vals[i + 3] = __bfloat162float(__ldg(&in_ptr[i + 3]));
      }
    }

#pragma unroll
    for (int c = 0; c < VECMAT_COLS; c++) {
      const int col = col_base + c;
      if (col >= N)
        continue;

      // 16 bytes = 32 packed weights = two NVFP4 blocks, so two scales.
      uint4 w_vec = *reinterpret_cast<const uint4 *>(
          &weight[(size_t)col * k_half + k_start / 2]);
      const size_t s_off = (size_t)col * scale_stride + pair * 2;
      const float s0 =
          e4m3_to_float(__ldg(&weight_scale[s_off])) * global_scale * 0.5f;
      const float s1 =
          e4m3_to_float(__ldg(&weight_scale[s_off + 1])) * global_scale * 0.5f;

      float w_vals[NVFP4_PAIR];
      dequant_store_8(w_vec.x, s0, NVFP4_LUT0, NVFP4_LUT1, NVFP4_LUT2,
                      NVFP4_LUT3, &w_vals[0]);
      dequant_store_8(w_vec.y, s0, NVFP4_LUT0, NVFP4_LUT1, NVFP4_LUT2,
                      NVFP4_LUT3, &w_vals[8]);
      dequant_store_8(w_vec.z, s1, NVFP4_LUT0, NVFP4_LUT1, NVFP4_LUT2,
                      NVFP4_LUT3, &w_vals[16]);
      dequant_store_8(w_vec.w, s1, NVFP4_LUT0, NVFP4_LUT1, NVFP4_LUT2,
                      NVFP4_LUT3, &w_vals[24]);

      float dot = 0.0f;
#pragma unroll
      for (int i = 0; i < NVFP4_PAIR; i++)
        dot = fmaf(in_vals[i], w_vals[i], dot);
      acc[c] += dot;
    }
  }

#pragma unroll
  for (int c = 0; c < VECMAT_COLS; c++) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
      acc[c] += __shfl_down_sync(0xffffffff, acc[c], offset);
  }

  if (lane_id == 0) {
#pragma unroll
    for (int c = 0; c < VECMAT_COLS; c++)
      s_reduce[c][warp_id] = acc[c];
  }
  __syncthreads();

  if (warp_id == 0) {
#pragma unroll
    for (int c = 0; c < VECMAT_COLS; c++) {
      float val = (lane_id < VECMAT_WARPS) ? s_reduce[c][lane_id] : 0.0f;
#pragma unroll
      for (int offset = VECMAT_WARPS / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

      if (lane_id == 0) {
        const int col = col_base + c;
        if (col < N) {
          if (has_bias && bias != nullptr) {
            if constexpr (std::is_same_v<T, half>) {
              val += __half2float(__ldg(&bias[col]));
            } else {
              val += __bfloat162float(__ldg(&bias[col]));
            }
          }
          if constexpr (std::is_same_v<T, half>) {
            output[(size_t)row * N + col] = __float2half(val);
          } else {
            output[(size_t)row * N + col] = __float2bfloat16(val);
          }
        }
      }
    }
  }
}

// ============================================================================
// Prefill path: register-tiled GEMM, weights expanded into shared memory.
// BLOCK_K is 32, so each row load covers two NVFP4 blocks.
// ============================================================================

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int TM, int TN>
__global__ void nvfp4_matmul_tiled(const T *__restrict__ input,
                                   const uint8_t *__restrict__ weight,
                                   const uint8_t *__restrict__ weight_scale,
                                   float global_scale,
                                   const T *__restrict__ bias,
                                   T *__restrict__ output, int M, int N, int K,
                                   bool has_bias) {
  constexpr int THREADS_N = BLOCK_N / TN;
  constexpr int THREADS_M = BLOCK_M / TM;
  constexpr int NUM_THREADS = THREADS_N * THREADS_M;
  constexpr int BK_PAD = BLOCK_K + 1;

  __shared__ float s_input[BLOCK_M][BK_PAD];
  __shared__ float s_weight[BLOCK_N][BK_PAD];

  const int tid = threadIdx.y * THREADS_N + threadIdx.x;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int scale_stride = K / NVFP4_BLOCK_SIZE;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++)
      acc[i][j] = 0.0f;

  for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += NUM_THREADS) {
      const int lm = idx / BLOCK_K;
      const int lk = idx % BLOCK_K;
      const int gm = by * BLOCK_M + lm;
      const int gk = k_tile + lk;

      float val = 0.0f;
      if (gm < M && gk < K) {
        if constexpr (std::is_same_v<T, half>) {
          val = __half2float(__ldg(&input[(size_t)gm * K + gk]));
        } else {
          val = __bfloat162float(__ldg(&input[(size_t)gm * K + gk]));
        }
      }
      s_input[lm][lk] = val;
    }

    for (int ln = tid; ln < BLOCK_N; ln += NUM_THREADS) {
      const int gn = bx * BLOCK_N + ln;
      if (gn < N) {
        uint4 w_vec = *reinterpret_cast<const uint4 *>(
            &weight[(size_t)gn * (K / 2) + k_tile / 2]);
        const size_t s_off =
            (size_t)gn * scale_stride + k_tile / NVFP4_BLOCK_SIZE;
        const float s0 =
            e4m3_to_float(__ldg(&weight_scale[s_off])) * global_scale * 0.5f;
        const float s1 = e4m3_to_float(__ldg(&weight_scale[s_off + 1])) *
                         global_scale * 0.5f;

        dequant_store_8(w_vec.x, s0, NVFP4_LUT0, NVFP4_LUT1, NVFP4_LUT2,
                        NVFP4_LUT3, &s_weight[ln][0]);
        dequant_store_8(w_vec.y, s0, NVFP4_LUT0, NVFP4_LUT1, NVFP4_LUT2,
                        NVFP4_LUT3, &s_weight[ln][8]);
        dequant_store_8(w_vec.z, s1, NVFP4_LUT0, NVFP4_LUT1, NVFP4_LUT2,
                        NVFP4_LUT3, &s_weight[ln][16]);
        dequant_store_8(w_vec.w, s1, NVFP4_LUT0, NVFP4_LUT1, NVFP4_LUT2,
                        NVFP4_LUT3, &s_weight[ln][24]);
      } else {
#pragma unroll
        for (int k = 0; k < BLOCK_K; k++)
          s_weight[ln][k] = 0.0f;
      }
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_K; k++) {
      float a_frag[TM];
      float b_frag[TN];

#pragma unroll
      for (int i = 0; i < TM; i++)
        a_frag[i] = s_input[threadIdx.y * TM + i][k];

#pragma unroll
      for (int j = 0; j < TN; j++)
        b_frag[j] = s_weight[threadIdx.x * TN + j][k];

#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int row = by * BLOCK_M + threadIdx.y * TM + i;
    if (row < M) {
#pragma unroll
      for (int j = 0; j < TN; j++) {
        const int col = bx * BLOCK_N + threadIdx.x * TN + j;
        if (col < N) {
          float val = acc[i][j];
          if (has_bias && bias != nullptr) {
            if constexpr (std::is_same_v<T, half>) {
              val += __half2float(__ldg(&bias[col]));
            } else {
              val += __bfloat162float(__ldg(&bias[col]));
            }
          }
          if constexpr (std::is_same_v<T, half>) {
            output[(size_t)row * N + col] = __float2half(val);
          } else {
            output[(size_t)row * N + col] = __float2bfloat16(val);
          }
        }
      }
    }
  }
}

} // namespace nvfp4_gemm

// ============================================================================
// C API. K must be a multiple of 32; the caller checks and otherwise keeps the
// dequantized path.
// ============================================================================

extern "C" void launch_nvfp4_matmul_f16(const __half *input,
                                        const uint8_t *weight,
                                        const uint8_t *weight_scale,
                                        float global_scale, const __half *bias,
                                        __half *output, int M, int N, int K,
                                        bool has_bias, cudaStream_t stream) {
  if (M <= 4) {
    dim3 block(VECMAT_THREADS);
    dim3 grid(CEILDIV(N, VECMAT_COLS), M);
    nvfp4_gemm::nvfp4_vecmat<half><<<grid, block, 0, stream>>>(
        input, weight, weight_scale, global_scale, bias, output, M, N, K,
        has_bias);
  } else {
    constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
    dim3 block(BN / TN, BM / TM);
    dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
    nvfp4_gemm::nvfp4_matmul_tiled<half, BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(input, weight, weight_scale, global_scale,
                                     bias, output, M, N, K, has_bias);
  }
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void
launch_nvfp4_matmul_bf16(const __nv_bfloat16 *input, const uint8_t *weight,
                         const uint8_t *weight_scale, float global_scale,
                         const __nv_bfloat16 *bias, __nv_bfloat16 *output,
                         int M, int N, int K, bool has_bias,
                         cudaStream_t stream) {
  if (M <= 4) {
    dim3 block(VECMAT_THREADS);
    dim3 grid(CEILDIV(N, VECMAT_COLS), M);
    nvfp4_gemm::nvfp4_vecmat<__nv_bfloat16><<<grid, block, 0, stream>>>(
        input, weight, weight_scale, global_scale, bias, output, M, N, K,
        has_bias);
  } else {
    constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
    dim3 block(BN / TN, BM / TM);
    dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
    nvfp4_gemm::nvfp4_matmul_tiled<__nv_bfloat16, BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(input, weight, weight_scale, global_scale,
                                     bias, output, M, N, K, has_bias);
  }
  CUDA_CHECK(cudaGetLastError());
}
