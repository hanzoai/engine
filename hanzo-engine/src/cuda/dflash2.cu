#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int DFLASH2_CONV_THREADS = 256;
constexpr int DFLASH2_TOPK_THREADS = 256;
constexpr int DFLASH2_TOPK_BUF = 512;
constexpr int DFLASH2_MAX_TOPK = 128;
constexpr int DFLASH2_WALK_THREADS = 256;
constexpr int DFLASH2_WARP_SIZE = 32;
constexpr uint32_t DFLASH2_INVALID_IDX = 0xffffffffu;

template <typename T> __device__ __forceinline__ float dflash2_to_f32(T v) {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(v);
  } else {
    return __bfloat162float(v);
  }
}

template <typename T> __device__ __forceinline__ T dflash2_from_f32(float v) {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __float2half(v);
  } else {
    return __float2bfloat16(v);
  }
}

__device__ __forceinline__ bool dflash2_better(float va, uint32_t ia, float vb,
                                               uint32_t ib) {
  if (ia == DFLASH2_INVALID_IDX) {
    return false;
  }
  if (ib == DFLASH2_INVALID_IDX) {
    return true;
  }
  return (va > vb) || (va == vb && ia < ib);
}

__device__ __forceinline__ void dflash2_bitonic_sort_512(float *sv,
                                                         uint32_t *si,
                                                         int tid) {
  for (int size = 2; size <= DFLASH2_TOPK_BUF; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      int pos = 2 * tid - (tid & (stride - 1));
      bool asc = ((pos & size) != 0);
      bool swap = asc ? dflash2_better(sv[pos], si[pos], sv[pos + stride],
                                       si[pos + stride])
                      : dflash2_better(sv[pos + stride], si[pos + stride],
                                       sv[pos], si[pos]);
      if (swap) {
        float tv = sv[pos];
        sv[pos] = sv[pos + stride];
        sv[pos + stride] = tv;
        uint32_t ti = si[pos];
        si[pos] = si[pos + stride];
        si[pos + stride] = ti;
      }
      __syncthreads();
    }
  }
}

template <typename T>
__global__ void dflash2_block_conv_kernel(
    const T *__restrict__ x, const T *__restrict__ dynamic,
    const T *__restrict__ base_kernel, const T *__restrict__ residual,
    T *__restrict__ out, int slots, int hidden, int kernel_size, int group_size,
    int half_idx) {
  const int total = slots * hidden;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int slot = idx / hidden;
  const int ch = idx % hidden;
  const int groups = hidden / group_size;
  const int group = ch / group_size;

  float acc = 0.0f;
  for (int offset = 0; offset < kernel_size; ++offset) {
    if (offset > slot) {
      break;
    }
    float base_val = dflash2_to_f32(
        base_kernel[((size_t)half_idx * kernel_size + offset) * hidden + ch]);
    float dyn_val = dflash2_to_f32(
        dynamic[(((size_t)slot * 2 + half_idx) * kernel_size + offset) *
                    groups +
                group]);
    float w = __fadd_rn(base_val, dyn_val);
    float x_val = dflash2_to_f32(x[(size_t)(slot - offset) * hidden + ch]);
    float term = __fmul_rn(w, x_val);
    acc = offset == 0 ? term : __fadd_rn(acc, term);
  }
  if (residual != nullptr) {
    acc = __fadd_rn(acc, dflash2_to_f32(residual[(size_t)slot * hidden + ch]));
  }
  out[(size_t)slot * hidden + ch] = dflash2_from_f32<T>(acc);
}

template <typename T>
__global__ void dflash2_topk_kernel(const T *__restrict__ logits,
                                    float *__restrict__ values_out,
                                    uint32_t *__restrict__ indices_out,
                                    int nrows, int ncols, int k) {
  const int row = blockIdx.x;
  if (row >= nrows) {
    return;
  }
  const int tid = threadIdx.x;
  const T *row_ptr = logits + (size_t)row * ncols;

  __shared__ float sv[DFLASH2_TOPK_BUF];
  __shared__ uint32_t si[DFLASH2_TOPK_BUF];
  __shared__ int count;
  __shared__ int chunk_hits;
  __shared__ float kth_val;
  __shared__ uint32_t kth_idx;

  sv[tid] = -INFINITY;
  si[tid] = DFLASH2_INVALID_IDX;
  sv[tid + DFLASH2_TOPK_THREADS] = -INFINITY;
  si[tid + DFLASH2_TOPK_THREADS] = DFLASH2_INVALID_IDX;
  if (tid == 0) {
    count = 0;
    kth_val = -INFINITY;
    kth_idx = DFLASH2_INVALID_IDX;
  }
  __syncthreads();

  for (int base = 0; base < ncols; base += DFLASH2_TOPK_THREADS) {
    int col = base + tid;
    float v = -INFINITY;
    bool valid = false;
    if (col < ncols) {
      v = dflash2_to_f32(row_ptr[col]);
      valid = !isnan(v);
    }
    bool pass = valid && dflash2_better(v, (uint32_t)col, kth_val, kth_idx);

    if (tid == 0) {
      chunk_hits = 0;
    }
    __syncthreads();

    uint32_t mask = __ballot_sync(0xffffffffu, pass);
    if (mask != 0u && (tid & (DFLASH2_WARP_SIZE - 1)) == 0) {
      atomicAdd(&chunk_hits, __popc(mask));
    }
    __syncthreads();

    if (chunk_hits == 0) {
      continue;
    }

    if (count + chunk_hits > DFLASH2_TOPK_BUF) {
      for (int i = tid; i < DFLASH2_TOPK_BUF; i += DFLASH2_TOPK_THREADS) {
        if (i >= count) {
          sv[i] = -INFINITY;
          si[i] = DFLASH2_INVALID_IDX;
        }
      }
      __syncthreads();
      dflash2_bitonic_sort_512(sv, si, tid);
      if (tid == 0) {
        count = k;
        kth_val = sv[k - 1];
        kth_idx = si[k - 1];
      }
      __syncthreads();
      pass = valid && dflash2_better(v, (uint32_t)col, kth_val, kth_idx);
    }

    if (pass) {
      int pos = atomicAdd(&count, 1);
      if (pos < DFLASH2_TOPK_BUF) {
        sv[pos] = v;
        si[pos] = (uint32_t)col;
      }
    }
    __syncthreads();

    if (kth_idx == DFLASH2_INVALID_IDX && count >= k) {
      for (int i = tid; i < DFLASH2_TOPK_BUF; i += DFLASH2_TOPK_THREADS) {
        if (i >= count) {
          sv[i] = -INFINITY;
          si[i] = DFLASH2_INVALID_IDX;
        }
      }
      __syncthreads();
      dflash2_bitonic_sort_512(sv, si, tid);
      if (tid == 0) {
        count = k;
        kth_val = sv[k - 1];
        kth_idx = si[k - 1];
      }
      __syncthreads();
    }
  }

  for (int i = tid; i < DFLASH2_TOPK_BUF; i += DFLASH2_TOPK_THREADS) {
    if (i >= count) {
      sv[i] = -INFINITY;
      si[i] = DFLASH2_INVALID_IDX;
    }
  }
  __syncthreads();
  dflash2_bitonic_sort_512(sv, si, tid);

  if (tid < k) {
    values_out[(size_t)row * k + tid] = sv[tid];
    indices_out[(size_t)row * k + tid] = si[tid];
  }
}

template <typename T>
__global__ void dflash2_greedy_walk_kernel(
    const float *__restrict__ topk_vals, const uint32_t *__restrict__ topk_ids,
    const T *__restrict__ gate_hidden, const T *__restrict__ pred_codebook,
    const T *__restrict__ succ_codebook, uint32_t *__restrict__ out_tokens,
    uint32_t anchor_token, int slots, int top_k, int rank) {
  extern __shared__ float s_gate[];
  __shared__ float s_scores[DFLASH2_MAX_TOPK];
  __shared__ uint32_t s_prev_token;

  const int tid = threadIdx.x;
  const int warp_id = tid / DFLASH2_WARP_SIZE;
  const int lane = tid % DFLASH2_WARP_SIZE;
  constexpr int num_warps = DFLASH2_WALK_THREADS / DFLASH2_WARP_SIZE;

  if (tid == 0) {
    s_prev_token = anchor_token;
  }
  __syncthreads();

  for (int slot = 0; slot < slots; ++slot) {
    const uint32_t prev_tok = s_prev_token;
    if (prev_tok == DFLASH2_INVALID_IDX) {
      if (tid == 0) {
        out_tokens[slot] = DFLASH2_INVALID_IDX;
      }
      __syncthreads();
      continue;
    }
    const T *pred_row = pred_codebook + (size_t)prev_tok * rank;
    const T *gh_row = gate_hidden + (size_t)slot * rank;

    for (int r = tid; r < rank; r += DFLASH2_WALK_THREADS) {
      s_gate[r] = dflash2_to_f32(pred_row[r]) * dflash2_to_f32(gh_row[r]);
    }
    __syncthreads();

    for (int cand = warp_id; cand < top_k; cand += num_warps) {
      const uint32_t cand_id = topk_ids[(size_t)slot * top_k + cand];
      const float unary = topk_vals[(size_t)slot * top_k + cand];
      float score = -INFINITY;
      if (cand_id != DFLASH2_INVALID_IDX && isfinite(unary)) {
        const T *succ_row = succ_codebook + (size_t)cand_id * rank;
        float dot = 0.0f;
        for (int r = lane; r < rank; r += DFLASH2_WARP_SIZE) {
          dot = fmaf(s_gate[r], dflash2_to_f32(succ_row[r]), dot);
        }
        for (int offset = DFLASH2_WARP_SIZE / 2; offset > 0; offset >>= 1) {
          dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        score = unary + dot;
      }
      if (lane == 0) {
        s_scores[cand] = score;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      float best_score = -INFINITY;
      int best_cand = 0;
      bool has_cand = false;
      for (int c = lane; c < top_k; c += DFLASH2_WARP_SIZE) {
        float sc = s_scores[c];
        if (isfinite(sc) &&
            (!has_cand || sc > best_score || (sc == best_score && c < best_cand))) {
          best_score = sc;
          best_cand = c;
          has_cand = true;
        }
      }
      for (int offset = DFLASH2_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float osc = __shfl_down_sync(0xffffffffu, best_score, offset);
        int oc = __shfl_down_sync(0xffffffffu, best_cand, offset);
        int ohas = __shfl_down_sync(0xffffffffu, (int)has_cand, offset);
        if (ohas && (!has_cand || osc > best_score ||
                     (osc == best_score && oc < best_cand))) {
          best_score = osc;
          best_cand = oc;
          has_cand = true;
        }
      }
      if (lane == 0) {
        if (has_cand) {
          uint32_t chosen = topk_ids[(size_t)slot * top_k + best_cand];
          s_prev_token = chosen;
          out_tokens[slot] = chosen;
        } else {
          s_prev_token = DFLASH2_INVALID_IDX;
          out_tokens[slot] = DFLASH2_INVALID_IDX;
        }
      }
    }
    __syncthreads();
  }
}

template <typename T>
void launch_block_conv(const void *x, const void *dynamic,
                       const void *base_kernel, const void *residual, void *out,
                       int slots, int hidden, int kernel_size, int group_size,
                       int half_idx, int64_t stream) {
  const int total = slots * hidden;
  if (total <= 0) {
    return;
  }
  const int blocks = (total + DFLASH2_CONV_THREADS - 1) / DFLASH2_CONV_THREADS;
  const cudaStream_t custream = (cudaStream_t)stream;
  dflash2_block_conv_kernel<T><<<blocks, DFLASH2_CONV_THREADS, 0, custream>>>(
      (const T *)x, (const T *)dynamic, (const T *)base_kernel,
      (const T *)residual, (T *)out, slots, hidden, kernel_size, group_size,
      half_idx);
}

template <typename T>
void launch_topk(const void *logits, float *values_out, uint32_t *indices_out,
                 int nrows, int ncols, int k, int64_t stream) {
  if (nrows <= 0 || ncols <= 0 || k <= 0) {
    return;
  }
  const cudaStream_t custream = (cudaStream_t)stream;
  dflash2_topk_kernel<T><<<nrows, DFLASH2_TOPK_THREADS, 0, custream>>>(
      (const T *)logits, values_out, indices_out, nrows, ncols, k);
}

template <typename T>
void launch_greedy_walk(const float *topk_vals, const uint32_t *topk_ids,
                        const void *gate_hidden, const void *pred_codebook,
                        const void *succ_codebook, uint32_t *out_tokens,
                        uint32_t anchor_token, int slots, int top_k, int rank,
                        int64_t stream) {
  if (slots <= 0 || top_k <= 0) {
    return;
  }
  const cudaStream_t custream = (cudaStream_t)stream;
  const size_t smem = (size_t)rank * sizeof(float);
  dflash2_greedy_walk_kernel<T><<<1, DFLASH2_WALK_THREADS, smem, custream>>>(
      topk_vals, topk_ids, (const T *)gate_hidden, (const T *)pred_codebook,
      (const T *)succ_codebook, out_tokens, anchor_token, slots, top_k, rank);
}

} // namespace

extern "C" void dflash2_block_conv_f16(const void *x, const void *dynamic,
                                       const void *base_kernel,
                                       const void *residual, void *out,
                                       int slots, int hidden, int kernel_size,
                                       int group_size, int half_idx,
                                       int64_t stream) {
  launch_block_conv<__half>(x, dynamic, base_kernel, residual, out, slots,
                            hidden, kernel_size, group_size, half_idx, stream);
}

extern "C" void dflash2_block_conv_bf16(const void *x, const void *dynamic,
                                        const void *base_kernel,
                                        const void *residual, void *out,
                                        int slots, int hidden, int kernel_size,
                                        int group_size, int half_idx,
                                        int64_t stream) {
  launch_block_conv<__nv_bfloat16>(x, dynamic, base_kernel, residual, out,
                                   slots, hidden, kernel_size, group_size,
                                   half_idx, stream);
}

extern "C" void dflash2_block_conv_f32(const void *x, const void *dynamic,
                                       const void *base_kernel,
                                       const void *residual, void *out,
                                       int slots, int hidden, int kernel_size,
                                       int group_size, int half_idx,
                                       int64_t stream) {
  launch_block_conv<float>(x, dynamic, base_kernel, residual, out, slots,
                           hidden, kernel_size, group_size, half_idx, stream);
}

extern "C" void dflash2_topk_f16(const void *logits, float *values_out,
                                 uint32_t *indices_out, int nrows, int ncols,
                                 int k, int64_t stream) {
  launch_topk<__half>(logits, values_out, indices_out, nrows, ncols, k, stream);
}

extern "C" void dflash2_topk_bf16(const void *logits, float *values_out,
                                  uint32_t *indices_out, int nrows, int ncols,
                                  int k, int64_t stream) {
  launch_topk<__nv_bfloat16>(logits, values_out, indices_out, nrows, ncols, k,
                             stream);
}

extern "C" void dflash2_topk_f32(const void *logits, float *values_out,
                                 uint32_t *indices_out, int nrows, int ncols,
                                 int k, int64_t stream) {
  launch_topk<float>(logits, values_out, indices_out, nrows, ncols, k, stream);
}

extern "C" void dflash2_greedy_walk_f16(
    const float *topk_vals, const uint32_t *topk_ids, const void *gate_hidden,
    const void *pred_codebook, const void *succ_codebook, uint32_t *out_tokens,
    uint32_t anchor_token, int slots, int top_k, int rank, int64_t stream) {
  launch_greedy_walk<__half>(topk_vals, topk_ids, gate_hidden, pred_codebook,
                             succ_codebook, out_tokens, anchor_token, slots,
                             top_k, rank, stream);
}

extern "C" void dflash2_greedy_walk_bf16(
    const float *topk_vals, const uint32_t *topk_ids, const void *gate_hidden,
    const void *pred_codebook, const void *succ_codebook, uint32_t *out_tokens,
    uint32_t anchor_token, int slots, int top_k, int rank, int64_t stream) {
  launch_greedy_walk<__nv_bfloat16>(topk_vals, topk_ids, gate_hidden,
                                    pred_codebook, succ_codebook, out_tokens,
                                    anchor_token, slots, top_k, rank, stream);
}

extern "C" void dflash2_greedy_walk_f32(
    const float *topk_vals, const uint32_t *topk_ids, const void *gate_hidden,
    const void *pred_codebook, const void *succ_codebook, uint32_t *out_tokens,
    uint32_t anchor_token, int slots, int top_k, int rank, int64_t stream) {
  launch_greedy_walk<float>(topk_vals, topk_ids, gate_hidden, pred_codebook,
                            succ_codebook, out_tokens, anchor_token, slots,
                            top_k, rank, stream);
}
