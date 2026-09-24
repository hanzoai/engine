/**
 * Activation quantization for the block-scaled NVFP4 GEMM.
 *
 * ModelOpt's scheme: one FP8 E4M3 scale per 16 values on top of the FP32
 * `input_scale` the checkpoint calibrated, so a value reconstructs as
 * code * block_scale * global. The block scale is written straight into the
 * swizzled layout the MMA reads, which saves a second pass over the scales.
 * Padding is left untouched, so the destination arrives zeroed.
 *
 * The scale and the codes are the served quantizer's (../quantize/e2m1.cuh), so
 * this path and the linear-layout quantizer produce identical bytes.
 */

#include "../quantize/e2m1.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
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

namespace nvfp4_quantize {

constexpr int BLOCK_SIZE = 16;
constexpr int SCALE_ROW_TILE = 128;
constexpr int SCALE_COL_TILE = 4;
constexpr int QUANT_THREADS = 256;
constexpr int MAX_BLOCKS = 65535;

// Byte index of scale (row, col) in the layout the block-scaled MMA reads.
__device__ __forceinline__ size_t swizzled_offset(size_t row, size_t col,
                                                  size_t padded_cols) {
  return (((row / SCALE_ROW_TILE) * (padded_cols / SCALE_COL_TILE) +
           col / SCALE_COL_TILE) *
              32 +
          row % 32) *
             16 +
         (row % SCALE_ROW_TILE) / 32 * SCALE_COL_TILE + col % SCALE_COL_TILE;
}

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<half>(half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__global__ void quantize_kernel(const T *__restrict__ input,
                                uint8_t *__restrict__ packed,
                                uint8_t *__restrict__ scale_swizzled, int M,
                                int K, float gs, size_t padded_cols) {
  const int blocks_per_row = K / BLOCK_SIZE;
  const size_t total = (size_t)M * blocks_per_row;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t row = i / blocks_per_row;
    const size_t col = i % blocks_per_row;
    const size_t base = i * BLOCK_SIZE;

    float v[BLOCK_SIZE];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < BLOCK_SIZE; j++) {
      v[j] = to_float<T>(input[base + j]);
      amax = fmaxf(amax, fabsf(v[j]));
    }

    const e2m1::Block blk = e2m1::block(amax, gs);
    scale_swizzled[swizzled_offset(row, col, padded_cols)] = blk.scale;

    uint8_t bytes[BLOCK_SIZE / 2];
#pragma unroll
    for (int j = 0; j < BLOCK_SIZE / 2; j++) {
      const uint8_t lo = e2m1::encode(v[j * 2] * blk.out);
      const uint8_t hi = e2m1::encode(v[j * 2 + 1] * blk.out);
      bytes[j] = (uint8_t)(lo | (hi << 4));
    }
    *reinterpret_cast<uint2 *>(&packed[base / 2]) =
        *reinterpret_cast<const uint2 *>(bytes);
  }
}

} // namespace nvfp4_quantize

extern "C" size_t nvfp4_quantize_scale_bytes(int rows, int k) {
  using namespace nvfp4_quantize;
  const size_t cols = (size_t)k / BLOCK_SIZE;
  const size_t padded_cols =
      (cols + SCALE_COL_TILE - 1) / SCALE_COL_TILE * SCALE_COL_TILE;
  const size_t padded_rows =
      ((size_t)rows + SCALE_ROW_TILE - 1) / SCALE_ROW_TILE * SCALE_ROW_TILE;
  return padded_rows * padded_cols;
}

namespace {
template <typename T>
void launch(const T *input, uint8_t *packed, uint8_t *scale_swizzled, int M,
            int K, float global, cudaStream_t stream) {
  using namespace nvfp4_quantize;
  const size_t cols = (size_t)K / BLOCK_SIZE;
  const size_t padded_cols =
      (cols + SCALE_COL_TILE - 1) / SCALE_COL_TILE * SCALE_COL_TILE;
  size_t blocks = ((size_t)M * cols + QUANT_THREADS - 1) / QUANT_THREADS;
  if (blocks > MAX_BLOCKS)
    blocks = MAX_BLOCKS;
  nvfp4_quantize::quantize_kernel<T>
      <<<(unsigned)blocks, QUANT_THREADS, 0, stream>>>(
          input, packed, scale_swizzled, M, K, 1.0f / global, padded_cols);
  // gs = 1 / input_scale, an IEEE division as the server's (1.0 / a13_scale on the device).
  CUDA_CHECK(cudaGetLastError());
}
} // namespace

extern "C" void nvfp4_quantize_activations_f16(const __half *input,
                                               uint8_t *packed,
                                               uint8_t *scale_swizzled, int M,
                                               int K, float global,
                                               cudaStream_t stream) {
  launch<half>(input, packed, scale_swizzled, M, K, global, stream);
}

extern "C" void nvfp4_quantize_activations_bf16(const __nv_bfloat16 *input,
                                                uint8_t *packed,
                                                uint8_t *scale_swizzled, int M,
                                                int K, float global,
                                                cudaStream_t stream) {
  launch<__nv_bfloat16>(input, packed, scale_swizzled, M, K, global, stream);
}
