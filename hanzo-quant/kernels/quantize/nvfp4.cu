/**
 * NVFP4 activation quantization with the served fast-math formula (e2m1.cuh), linear scale
 * layout: codes [M, K/2] (low nibble first), E4M3 scales [M, K/16].
 *
 * The same formula serves vLLM's scaled_fp4_quant (the fc1 input of the NVFP4 MoE and the dense
 * NVFP4 layers) and FlashInfer's cvt_warp_fp16_to_fp4 (the fc2 input, after the bf16 SwiGLU).
 */

#include "e2m1.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int BLOCK = 16;

template <typename T> __device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// One thread per block of 16.
template <typename T>
__global__ void nvfp4_kernel(const T *__restrict__ x, uint8_t *__restrict__ codes,
                             uint8_t *__restrict__ scales, long blocks, float gs) {
  for (long b = (long)blockIdx.x * blockDim.x + threadIdx.x; b < blocks;
       b += (long)gridDim.x * blockDim.x) {
    const T *src = x + b * BLOCK;
    float v[BLOCK];
    float vmax = 0.0f;
#pragma unroll
    for (int j = 0; j < BLOCK; j++) {
      v[j] = to_f32<T>(src[j]);
      vmax = fmaxf(vmax, fabsf(v[j]));
    }
    const e2m1::Block blk = e2m1::block(vmax, gs);
    scales[b] = blk.scale;
    uint8_t *dst = codes + b * (BLOCK / 2);
#pragma unroll
    for (int j = 0; j < BLOCK / 2; j++) {
      const uint8_t lo = e2m1::encode(v[2 * j] * blk.out);
      const uint8_t hi = e2m1::encode(v[2 * j + 1] * blk.out);
      dst[j] = lo | (hi << 4);
    }
  }
}

template <typename T>
void launch(const T *x, uint8_t *codes, uint8_t *scales, int m, int k, float gs,
            cudaStream_t stream) {
  const long blocks = (long)m * (k / BLOCK);
  const int threads = 256;
  long grid = (blocks + threads - 1) / threads;
  if (grid > 65535)
    grid = 65535;
  nvfp4_kernel<T><<<(unsigned)grid, threads, 0, stream>>>(x, codes, scales, blocks, gs);
}

// Round-trip probes for the encoder tests: one code per input.
__global__ void e2m1_encode_kernel(const float *v, uint8_t *codes, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    codes[i] = e2m1::encode(v[i]);
}

} // namespace

extern "C" void hanzo_quantize_nvfp4_f16(const __half *x, uint8_t *codes, uint8_t *scales, int m,
                                         int k, float gs, cudaStream_t stream) {
  launch<__half>(x, codes, scales, m, k, gs, stream);
}

extern "C" void hanzo_quantize_nvfp4_bf16(const __nv_bfloat16 *x, uint8_t *codes,
                                          uint8_t *scales, int m, int k, float gs,
                                          cudaStream_t stream) {
  launch<__nv_bfloat16>(x, codes, scales, m, k, gs, stream);
}

extern "C" void hanzo_e2m1_encode(const float *v, uint8_t *codes, int n, cudaStream_t stream) {
  e2m1_encode_kernel<<<(n + 255) / 256, 256, 0, stream>>>(v, codes, n);
}
