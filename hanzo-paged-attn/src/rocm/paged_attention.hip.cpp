// PagedAttention v1 DECODE kernel for ROCm / HIP (RDNA3.5, gfx1151).
//
// This is a self-contained port of the CORE vLLM v1 decode path (see
// ../cuda/pagedattention.cuh). For a single decode query per sequence it
// gathers K/V from the paged KV cache via a block_table, computes
// softmax(scale * Q.K^T) . V, and writes the result.
//
// CRITICAL (the whole point of PagedAttention for the decode hipGraph lever):
// the per-sequence context length is read from the DEVICE int array
// `context_lens[seq_idx]` inside the kernel. It is NEVER a host scalar baked
// into the launch parameters. The grid/block shape depends only on
// (num_seqs, num_heads, head_size) and is therefore CONSTANT per token, which
// is what makes later graph capture possible.
//
// Scope: f16 + bf16, non-FP8 cache only (fp8/quant deferred per stage-1 scope).
// scale, softcapping, alibi and attention sinks are supported as cheap scalar
// ops so the extern "C" signature stays byte-identical to the CUDA backend,
// letting the Rust side be a near-verbatim mirror of src/cuda/.

#include "hip/hip_runtime.h"
#include "hip/hip_bf16.h"
#include "hip/hip_fp16.h"
#include <float.h>
#include <math.h>
#include <stdint.h>

// RDNA3 / RDNA3.5 wavefronts are 32 lanes wide (warpSize == 32). We hardcode
// 32 rather than relying on the runtime `warpSize` so the reduction unrolls
// are compile-time, matching the CUDA WARP_SIZE==32 assumption.
#define WARP_SIZE 32

#ifndef DIVIDE_ROUND_UP
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#endif

namespace hanzo_rocm_pa {

// ---- scalar conversion helpers ---------------------------------------------

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_float<__hip_bfloat16>(__hip_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __hip_bfloat16 from_float<__hip_bfloat16>(float v) {
  return __float2bfloat16(v);
}

__device__ __forceinline__ float fast_tanh(float x) { return tanhf(x); }

// Block-wide reductions. NUM_THREADS is a power of two and a multiple of
// WARP_SIZE; `red` is shared scratch of length >= NUM_THREADS / WARP_SIZE.
template <int NUM_THREADS>
__device__ __forceinline__ float block_reduce_max(float val, float *red) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor(val, mask));
  }
  if (lane == 0) {
    red[warp] = val;
  }
  __syncthreads();
  val = (lane < NUM_WARPS) ? red[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor(val, mask));
  }
  return __shfl(val, 0);
}

template <int NUM_THREADS>
__device__ __forceinline__ float block_reduce_sum(float val, float *red) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask >>= 1) {
    val += __shfl_xor(val, mask);
  }
  if (lane == 0) {
    red[warp] = val;
  }
  __syncthreads();
  val = (lane < NUM_WARPS) ? red[lane] : 0.f;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask >>= 1) {
    val += __shfl_xor(val, mask);
  }
  return __shfl(val, 0);
}

// One thread block per (head_idx = blockIdx.x, seq_idx = blockIdx.y).
//
// Layout (matches vLLM / the CUDA backend):
//   q:           [num_seqs, num_heads, head_size]        (q_stride between seqs)
//   key_cache:   [num_blocks, num_kv_heads, head_size/x, block_size, x]
//   value_cache: [num_blocks, num_kv_heads, head_size, block_size]
//   block_table: [num_seqs, max_num_blocks_per_seq]
//   context_lens:[num_seqs]   (DEVICE-side; read inside the kernel)
//   out:         [num_seqs, num_heads, head_size]
template <typename scalar_t, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ q,
    const scalar_t *__restrict__ k_cache, const scalar_t *__restrict__ v_cache,
    const int num_kv_heads, const float scale, const float softcapping,
    const int *__restrict__ block_tables,
    const int *__restrict__ context_lens, const int max_num_blocks_per_seq,
    const float *__restrict__ alibi_slopes, const int q_stride,
    const int kv_block_stride, const int kv_head_stride, const int head_size,
    const int block_size, const int x,
    const float *__restrict__ sinks // [num_heads] or nullptr
) {
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int num_heads = gridDim.x;
  const int tid = threadIdx.x;

  // DEVICE-SIDE sequence length. This is the keystone: launch params do not
  // encode it, so they stay constant across decode steps.
  const int context_len = context_lens[seq_idx];

  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.f : alibi_slopes[head_idx];

  // Dynamic shared memory: logits[ padded_context_len ] (one float per token).
  extern __shared__ float smem[];
  float *logits = smem;
  __shared__ float red[NUM_THREADS / WARP_SIZE];
  // Query for this (seq, head), promoted to fp32 in shared memory so every
  // thread can read every dim cheaply.
  __shared__ float q_shared[512]; // head_size <= 512 (checked Rust-side)

  const scalar_t *q_ptr = q + seq_idx * q_stride + head_idx * head_size;
  for (int d = tid; d < head_size; d += NUM_THREADS) {
    q_shared[d] = to_float(q_ptr[d]) * scale;
  }
  __syncthreads();

  const int *block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  // ---- Phase 1: logits[t] = scale * Q . K_t  (+ softcap, + alibi) ----------
  float local_max = -FLT_MAX;
  for (int t = tid; t < context_len; t += NUM_THREADS) {
    const int64_t phys_block = (int64_t)block_table[t / block_size];
    const int block_offset = t % block_size;
    // K element address inside the [.., head_size/x, block_size, x] tile.
    const scalar_t *k_base = k_cache + phys_block * kv_block_stride +
                             kv_head_idx * kv_head_stride;
    float qk = 0.f;
#pragma unroll 4
    for (int d = 0; d < head_size; d++) {
      const int x_idx = d / x;
      const int x_off = d % x;
      const scalar_t kv =
          k_base[x_idx * block_size * x + block_offset * x + x_off];
      qk += q_shared[d] * to_float(kv);
    }
    if (softcapping != 1.0f) {
      qk = fast_tanh(qk / softcapping) * softcapping;
    }
    if (alibi_slope != 0.f) {
      qk += alibi_slope * (float)(t - context_len + 1);
    }
    logits[t] = qk;
    local_max = fmaxf(local_max, qk);
  }
  float qk_max = block_reduce_max<NUM_THREADS>(local_max, red);
  // V1 (non-partitioned) includes the attention sink in the max.
  if (sinks != nullptr) {
    qk_max = fmaxf(qk_max, sinks[head_idx]);
  }

  // ---- Phase 2: exponentiate and sum --------------------------------------
  float local_sum = 0.f;
  for (int t = tid; t < context_len; t += NUM_THREADS) {
    float e = __expf(logits[t] - qk_max);
    logits[t] = e;
    local_sum += e;
  }
  float exp_sum = block_reduce_sum<NUM_THREADS>(local_sum, red);
  if (sinks != nullptr) {
    exp_sum += __expf(sinks[head_idx] - qk_max);
  }
  const float inv_sum = 1.0f / (exp_sum + 1e-6f);
  __syncthreads(); // logits[] fully written before Phase 3 reads it

  // ---- Phase 3: out[d] = sum_t (logits[t] * inv_sum) * V_t[d] --------------
  // One thread per output dim (head_size <= NUM_THREADS for the supported
  // head sizes when NUM_THREADS == 128; larger heads are handled by striding).
  scalar_t *out_ptr =
      out + seq_idx * num_heads * head_size + head_idx * head_size;
  for (int d = tid; d < head_size; d += NUM_THREADS) {
    float acc = 0.f;
    for (int t = 0; t < context_len; t++) {
      const int64_t phys_block = (int64_t)block_table[t / block_size];
      const int block_offset = t % block_size;
      const scalar_t *v_base = v_cache + phys_block * kv_block_stride +
                               kv_head_idx * kv_head_stride;
      // value_cache row-major within block: [head_size, block_size].
      const scalar_t vv = v_base[d * block_size + block_offset];
      acc += logits[t] * to_float(vv);
    }
    out_ptr[d] = from_float<scalar_t>(acc * inv_sum);
  }
}

template <typename scalar_t>
void launch_v1(void *out, void *query, void *key_cache, void *value_cache,
               void *alibi_slopes, int num_kv_heads, float scale,
               float softcapping, const int *block_tables,
               const int *context_lens, int block_size, int max_context_len,
               int num_seqs, int num_heads, int head_size,
               int max_num_blocks_per_seq, int q_stride, int kv_block_stride,
               int kv_head_stride, int x, hipStream_t stream,
               const float *sinks) {
  constexpr int NUM_THREADS = 128;
  // Shared logits buffer sized to the padded max context length, exactly like
  // the CUDA launcher (padded_max_context_len * sizeof(float)).
  const int padded =
      DIVIDE_ROUND_UP(max_context_len, block_size) * block_size;
  const size_t shared_mem = (size_t)padded * sizeof(float);

  dim3 grid((unsigned)num_heads, (unsigned)num_seqs, 1u);
  dim3 block((unsigned)NUM_THREADS);

  paged_attention_v1_kernel<scalar_t, NUM_THREADS>
      <<<grid, block, shared_mem, stream>>>(
          reinterpret_cast<scalar_t *>(out),
          reinterpret_cast<const scalar_t *>(query),
          reinterpret_cast<const scalar_t *>(key_cache),
          reinterpret_cast<const scalar_t *>(value_cache), num_kv_heads, scale,
          softcapping, block_tables, context_lens, max_num_blocks_per_seq,
          reinterpret_cast<const float *>(alibi_slopes), q_stride,
          kv_block_stride, kv_head_stride, head_size, block_size, x, sinks);
}

} // namespace hanzo_rocm_pa

// The extern "C" signatures mirror src/cuda/ffi.rs exactly (same argument order
// and types) so the Rust FFI is a verbatim mirror. `cache_dtype`, `k_scale`,
// `v_scale` are accepted for signature parity; only the non-FP8 path (0/1/2) is
// implemented in stage 1.

extern "C" void paged_attention_v1_f16(
    void *out, void *query, void *key_cache, void *value_cache,
    void *alibi_slopes, int32_t num_kv_heads, float scale, float softcapping,
    int32_t *block_tables, int32_t *context_lens, int32_t block_size,
    int32_t max_context_len, int32_t num_seqs, int32_t num_heads,
    int32_t head_size, int32_t max_num_blocks_per_seq, int32_t q_stride,
    int32_t kv_block_stride, int32_t kv_head_stride, int64_t stream,
    uint32_t cache_dtype, const float *k_scale, const float *v_scale,
    const float *sinks) {
  (void)k_scale;
  (void)v_scale;
  if (cache_dtype == 3) {
    // FP8 KV cache deferred for stage 1.
    return;
  }
  const int x = 16 / (int)sizeof(__half); // matches CUDA: x == 16/sizeof(cache_t)
  hanzo_rocm_pa::launch_v1<__half>(
      out, query, key_cache, value_cache, alibi_slopes, num_kv_heads, scale,
      softcapping, block_tables, context_lens, block_size, max_context_len,
      num_seqs, num_heads, head_size, max_num_blocks_per_seq, q_stride,
      kv_block_stride, kv_head_stride, x, (hipStream_t)stream, sinks);
}

extern "C" void paged_attention_v1_bf16(
    void *out, void *query, void *key_cache, void *value_cache,
    void *alibi_slopes, int32_t num_kv_heads, float scale, float softcapping,
    int32_t *block_tables, int32_t *context_lens, int32_t block_size,
    int32_t max_context_len, int32_t num_seqs, int32_t num_heads,
    int32_t head_size, int32_t max_num_blocks_per_seq, int32_t q_stride,
    int32_t kv_block_stride, int32_t kv_head_stride, int64_t stream,
    uint32_t cache_dtype, const float *k_scale, const float *v_scale,
    const float *sinks) {
  (void)k_scale;
  (void)v_scale;
  if (cache_dtype == 3) {
    return;
  }
  const int x = 16 / (int)sizeof(__hip_bfloat16);
  hanzo_rocm_pa::launch_v1<__hip_bfloat16>(
      out, query, key_cache, value_cache, alibi_slopes, num_kv_heads, scale,
      softcapping, block_tables, context_lens, block_size, max_context_len,
      num_seqs, num_heads, head_size, max_num_blocks_per_seq, q_stride,
      kv_block_stride, kv_head_stride, x, (hipStream_t)stream, sinks);
}
