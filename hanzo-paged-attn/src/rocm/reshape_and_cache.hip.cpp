// reshape_and_cache for ROCm / HIP (RDNA3.5, gfx1151).
//
// Port of ../cuda/reshape_and_cache_kernel.cu (CORE, non-FP8 path). Writes new
// per-token K/V into the paged KV cache at the slots given by `slot_mapping`.
//
//   key/value:   [num_tokens, num_heads, head_size]   (key_stride/value_stride)
//   key_cache:   [num_blocks, num_heads, head_size/x, block_size, x]
//   value_cache: [num_blocks, num_heads, head_size, block_size]
//   slot_mapping:[num_tokens]   (int64; negative slot => padding, skipped)

#include "hip/hip_runtime.h"
#include "hip/hip_bf16.h"
#include "hip/hip_fp16.h"
#include <stdint.h>

namespace hanzo_rocm_pa {

template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t *__restrict__ key, const scalar_t *__restrict__ value,
    scalar_t *__restrict__ key_cache, scalar_t *__restrict__ value_cache,
    const int64_t *__restrict__ slot_mapping, const int key_stride,
    const int value_stride, const int num_heads, const int head_size,
    const int block_size, const int x) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return; // padding token
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;

    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

template <typename scalar_t>
void launch_reshape(void *key, void *value, void *key_cache, void *value_cache,
                    const int64_t *slot_mapping, int num_tokens, int num_heads,
                    int head_size, int block_size, int x, int key_stride,
                    int value_stride, hipStream_t stream) {
  dim3 grid((unsigned)num_tokens);
  int threads = num_heads * head_size;
  if (threads > 512)
    threads = 512;
  dim3 block((unsigned)threads);
  reshape_and_cache_kernel<scalar_t><<<grid, block, 0, stream>>>(
      reinterpret_cast<const scalar_t *>(key),
      reinterpret_cast<const scalar_t *>(value),
      reinterpret_cast<scalar_t *>(key_cache),
      reinterpret_cast<scalar_t *>(value_cache), slot_mapping, key_stride,
      value_stride, num_heads, head_size, block_size, x);
}

} // namespace hanzo_rocm_pa

// Mirrors src/cuda/ffi.rs `reshape_and_cache` signature exactly.
extern "C" void reshape_and_cache(
    void *key, void *value, void *key_cache, void *value_cache,
    int64_t *slot_mapping, int32_t num_tokens, int32_t num_heads,
    int32_t head_size, int32_t block_size, int32_t x, int32_t key_stride,
    int32_t value_stride, int64_t stream, uint32_t dtype, uint32_t cache_dtype,
    const float *k_scale, const float *v_scale) {
  (void)k_scale;
  (void)v_scale;
  if (cache_dtype == 3) {
    return; // FP8 cache deferred for stage 1.
  }
  hipStream_t st = (hipStream_t)stream;
  if (dtype == 0) {
    hanzo_rocm_pa::launch_reshape<__half>(
        key, value, key_cache, value_cache, slot_mapping, num_tokens, num_heads,
        head_size, block_size, x, key_stride, value_stride, st);
  } else if (dtype == 1) {
    hanzo_rocm_pa::launch_reshape<__hip_bfloat16>(
        key, value, key_cache, value_cache, slot_mapping, num_tokens, num_heads,
        head_size, block_size, x, key_stride, value_stride, st);
  }
  // dtype == 2 (f32) intentionally omitted for stage 1 (decode is f16/bf16).
}
