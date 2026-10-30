/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2024, The vLLM team.
 *
 * Ported from vLLM v0.29.0 csrc/libtorch_stable/moe/topk_softmax_kernels.cu
 * (topkGating, TopkConstants, topkGatingLauncherHelper, topkGatingKernelLauncher),
 * itself adapted from TensorRT-LLM v0.7.1
 * cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu.
 *
 * Modified by Hanzo AI: the torch bindings, finished rows, expert-parallel
 * ranges, source_rows and is_padding are removed; a Raw score, a logit clip, a
 * selected-weight sigmoid, norm_min and a per-expert output scale are added as
 * extensions that leave the upstream arithmetic untouched when off.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy
 * of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * Built in the exact lane (no --use_fast_math), as vLLM builds _moe_C: IEEE
 * expf and division, no FTZ. Fast math changes 14-54% of the weight bits.
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width)                        \
  __shfl_xor_sync(uint32_t(-1), (var), (lane_mask), (width))

namespace route {

template <typename T, int N, int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray {
  T data[N];
};

// Score functions. SOFTMAX and SIGMOID are vLLM's; RAW is the engine's
// (llama4): the logit itself, NaN read as -inf, knockout by a taken mask.
enum Score { RAW = 0, SOFTMAX = 1, SIGMOID = 2 };

// Per-launch options outside vLLM's signature. All off reproduces vLLM.
struct Extra {
  const float *expert_scale; // weight *= expert_scale[id] after vLLM's scale
  float lo, hi;              // clip logits to [lo, hi] before scoring
  float norm_min;            // denom = sum > 0 ? max(sum, norm_min) : 1
  bool clip;
  bool sigmoid_weight; // RAW only: weight = 1 / (1 + expf(-logit))
};

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,
          int WARP_SIZE_PARAM, typename InputType, Score SF>
__launch_bounds__(WARPS_PER_CTA *WARP_SIZE_PARAM) __global__
    void topkGating(const InputType *input, float *output, const int num_rows,
                    uint32_t *indices, const int k, const bool renormalize,
                    const float *bias, const float routed_scaling_factor,
                    const Extra extra) {
  static_assert(std::is_same_v<InputType, float> ||
                    std::is_same_v<InputType, __nv_bfloat16> ||
                    std::is_same_v<InputType, __half>,
                "InputType must be float, __nv_bfloat16, or __half");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  if constexpr (std::is_same_v<InputType, __nv_bfloat16> ||
                std::is_same_v<InputType, __half>) {
    static_assert(ELTS_PER_LDG == 1 || ELTS_PER_LDG % 2 == 0,
                  "ELTS_PER_LDG must be 1 or even for 16-bit conversion");
  }
  static_assert(VPT % ELTS_PER_LDG == 0,
                "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE_PARAM % THREADS_PER_ROW == 0,
                "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
                "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE_PARAM,
                "THREADS_PER_ROW can be at most warp size");

  static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0,
                "The elts per row must cleanly divide the total elt per warp");

  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;
  if (thread_row >= num_rows) {
    return;
  }

  const InputType *thread_row_ptr = input + thread_row * ELTS_PER_ROW;
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const InputType *thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  float row_chunk[VPT];

  if constexpr (std::is_same_v<InputType, float>) {
    using VecType = AlignedArray<float, ELTS_PER_LDG>;
    VecType *row_chunk_vec_ptr = reinterpret_cast<VecType *>(&row_chunk);
    const VecType *vec_thread_read_ptr =
        reinterpret_cast<const VecType *>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }
  } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
    if constexpr (ELTS_PER_LDG >= 2) {
      using VecType = AlignedArray<__nv_bfloat16, ELTS_PER_LDG>;
      float2 *row_chunk_f2 = reinterpret_cast<float2 *>(row_chunk);
      const VecType *vec_thread_read_ptr =
          reinterpret_cast<const VecType *>(thread_read_ptr);
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
          row_chunk_f2[base_idx_f2 + jj] = __bfloat1622float2(
              *reinterpret_cast<const __nv_bfloat162 *>(vec.data + jj * 2));
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        const __nv_bfloat16 *scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
        row_chunk[ii] = __bfloat162float(*scalar_ptr);
      }
    }
  } else if constexpr (std::is_same_v<InputType, __half>) {
    if constexpr (ELTS_PER_LDG >= 2) {
      using VecType = AlignedArray<__half, ELTS_PER_LDG>;
      float2 *row_chunk_f2 = reinterpret_cast<float2 *>(row_chunk);
      const VecType *vec_thread_read_ptr =
          reinterpret_cast<const VecType *>(thread_read_ptr);
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
          row_chunk_f2[base_idx_f2 + jj] = __half22float2(
              *reinterpret_cast<const __half2 *>(vec.data + jj * 2));
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        const __half *scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
        row_chunk[ii] = __half2float(*scalar_ptr);
      }
    }
  }

  // Extension: clip the logits before scoring.
  if (extra.clip) {
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = fminf(fmaxf(row_chunk[ii], extra.lo), extra.hi);
    }
  }

  if constexpr (SF == SOFTMAX) {
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii) {
      thread_max = max(thread_max, row_chunk[ii]);
    }
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      thread_max =
          max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
    }
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = expf(row_chunk[ii] - thread_max);
      row_sum += row_chunk[ii];
    }
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }
    const float reciprocal_row_sum = 1.f / row_sum;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }
  } else if constexpr (SF == SIGMOID) {
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = 1.0f / (1.0f + __expf(-row_chunk[ii]));
    }
  }

  if constexpr (SF == RAW) {
    // Extension: a NaN logit can never be chosen ahead of a number.
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      if (isnan(row_chunk[ii])) {
        row_chunk[ii] = -INFINITY;
      }
    }
  } else {
    // vLLM: NaN/Inf probabilities are 0, so argmax falls back to index order.
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      if (isnan(row_chunk[ii]) || isinf(row_chunk[ii])) {
        row_chunk[ii] = 0.f;
      }
    }
  }

  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  float row_chunk_for_choice[VPT];
  if (bias != nullptr) {
#pragma unroll
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        const int expert = first_elt_read_by_thread + ldg * COLS_PER_GROUP_LDG + ii;
        float bias_val = expert < NUM_EXPERTS ? bias[expert] : 0.0f;
        row_chunk_for_choice[ldg * ELTS_PER_LDG + ii] =
            row_chunk[ldg * ELTS_PER_LDG + ii] + bias_val;
      }
    }
  } else {
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk_for_choice[ii] = row_chunk[ii];
    }
  }

  // Extension (RAW): chosen experts are masked out instead of overwritten,
  // since a raw logit has no floor to knock it below.
  static_assert(SF != RAW || VPT <= 32, "taken mask holds 32 values");
  uint32_t taken = 0;

  int start_col = first_elt_read_by_thread;

  float selected_sum = 0.f;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    float max_val_for_choice = row_chunk_for_choice[0];
    float max_val = row_chunk[0];
    int expert = start_col;
    if constexpr (SF == RAW) {
      max_val_for_choice = -INFINITY;
      max_val = -INFINITY;
      expert = NUM_EXPERTS;
    }
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
         ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        float val_for_choice = row_chunk_for_choice[ldg * ELTS_PER_LDG + ii];
        float val = row_chunk[ldg * ELTS_PER_LDG + ii];
        if constexpr (SF == RAW) {
          if (!(taken >> (ldg * ELTS_PER_LDG + ii) & 1u) &&
              (expert == NUM_EXPERTS || val_for_choice > max_val_for_choice)) {
            max_val_for_choice = val_for_choice;
            max_val = val;
            expert = col + ii;
          }
        } else {
          if (val_for_choice > max_val_for_choice) {
            max_val_for_choice = val_for_choice;
            max_val = val;
            expert = col + ii;
          }
        }
      }
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max_for_choice =
          VLLM_SHFL_XOR_SYNC_WIDTH(max_val_for_choice, mask, THREADS_PER_ROW);
      float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
      int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);
      if (other_max_for_choice > max_val_for_choice ||
          (other_max_for_choice == max_val_for_choice && other_expert < expert)) {
        max_val_for_choice = other_max_for_choice;
        max_val = other_max;
        expert = other_expert;
      }
    }

    if (thread_group_idx == 0) {
      const int idx = k * thread_row + k_idx;
      float weight = max_val;
      if constexpr (SF == RAW) {
        if (extra.sigmoid_weight) {
          weight = 1.0f / (1.0f + expf(-max_val));
        }
      }
      output[idx] = weight;
      indices[idx] = static_cast<uint32_t>(expert);
      if (renormalize) {
        selected_sum += weight;
      }
    }

    if (k_idx + 1 < k) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        if constexpr (SF == RAW) {
          taken |= 1u << (ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert);
        } else {
          row_chunk_for_choice[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
              -10000.f;
        }
      }
    }
  }

  if (thread_group_idx == 0) {
    float scale = routed_scaling_factor;
    if (renormalize) {
      const float denom = selected_sum > 0.f ? fmaxf(selected_sum, extra.norm_min) : 1.f;
      scale /= denom;
    }
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int idx = k * thread_row + k_idx;
      output[idx] = output[idx] * scale;
      if (extra.expert_scale != nullptr) {
        output[idx] = output[idx] * extra.expert_scale[indices[idx]];
      }
    }
  }
}

template <int EXPERTS, int BYTES_PER_LDG, int WARP_SIZE_PARAM, typename InputType>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 ||
                    EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0,
                "");
  static constexpr int VECs_PER_THREAD =
      MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static const int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};

template <int EXPERTS, int WARPS_PER_TB, int WARP_SIZE_PARAM, int MAX_BYTES_PER_LDG,
          typename InputType, Score SF>
void launch(const InputType *input, float *output, uint32_t *indices,
            const int num_rows, const int k, const bool renormalize,
            const float *bias, const float routed_scaling_factor, const Extra &extra,
            cudaStream_t stream) {
  static constexpr int BYTES_PER_LDG =
      MIN(MAX_BYTES_PER_LDG, sizeof(InputType) * EXPERTS);
  using Constants = TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(WARP_SIZE_PARAM, WARPS_PER_TB);
  topkGating<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType, SF>
      <<<num_blocks, block_dim, 0, stream>>>(input, output, num_rows, indices, k,
                                             renormalize, bias,
                                             routed_scaling_factor, extra);
}

// vLLM's fused expert counts, in its switch order. Anything else takes vLLM's
// unfused moeSoftmax + moeTopK path upstream, and the engine's portable path.
constexpr int EXPERT_COUNTS[] = {1,   2,   4,   8,   16,  32,  64,  128,
                                 256, 512, 192, 320, 384, 448, 576};

template <typename InputType, Score SF>
int dispatch(const InputType *gating_output, float *topk_weights,
             uint32_t *topk_indices, const int num_tokens, const int num_experts,
             const int topk, const bool renormalize, const float *bias,
             const float routed_scaling_factor, const Extra &extra,
             cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;
  static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;
  // for bfloat16 dtype, we need 4 bytes loading to make sure num_experts
  // elements can be loaded by a warp
  static constexpr int BYTES_PER_LDG_MULTIPLE_64 =
      (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>)
          ? 4
          : 8;
#define LAUNCH_TOPK(NUM_EXPERTS, MAX_BYTES)                                    \
  launch<NUM_EXPERTS, WARPS_PER_TB, 32, MAX_BYTES, InputType, SF>(             \
      gating_output, topk_weights, topk_indices, num_tokens, topk,             \
      renormalize, bias, routed_scaling_factor, extra, stream);                \
  return 0
  switch (num_experts) {
  case 1: LAUNCH_TOPK(1, BYTES_PER_LDG_POWER_OF_2);
  case 2: LAUNCH_TOPK(2, BYTES_PER_LDG_POWER_OF_2);
  case 4: LAUNCH_TOPK(4, BYTES_PER_LDG_POWER_OF_2);
  case 8: LAUNCH_TOPK(8, BYTES_PER_LDG_POWER_OF_2);
  case 16: LAUNCH_TOPK(16, BYTES_PER_LDG_POWER_OF_2);
  case 32: LAUNCH_TOPK(32, BYTES_PER_LDG_POWER_OF_2);
  case 64: LAUNCH_TOPK(64, BYTES_PER_LDG_POWER_OF_2);
  case 128: LAUNCH_TOPK(128, BYTES_PER_LDG_POWER_OF_2);
  case 256: LAUNCH_TOPK(256, BYTES_PER_LDG_POWER_OF_2);
  case 512: LAUNCH_TOPK(512, BYTES_PER_LDG_POWER_OF_2);
  case 192: LAUNCH_TOPK(192, BYTES_PER_LDG_MULTIPLE_64);
  case 320: LAUNCH_TOPK(320, BYTES_PER_LDG_MULTIPLE_64);
  case 384: LAUNCH_TOPK(384, BYTES_PER_LDG_MULTIPLE_64);
  case 448: LAUNCH_TOPK(448, BYTES_PER_LDG_MULTIPLE_64);
  case 576: LAUNCH_TOPK(576, BYTES_PER_LDG_MULTIPLE_64);
  default: return 1;
  }
#undef LAUNCH_TOPK
}

template <typename InputType>
int route(const void *logits, float *weights, uint32_t *ids, const float *bias,
          const float *expert_scale, int rows, int experts, int k, int score,
          int weight, bool renormalize, bool clip, float lo, float hi,
          float norm_min, float scale, int64_t stream) {
  const Extra extra{expert_scale, lo, hi, norm_min, clip, weight == 1};
  const InputType *x = static_cast<const InputType *>(logits);
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  int rc;
  switch (score) {
  case RAW:
    rc = dispatch<InputType, RAW>(x, weights, ids, rows, experts, k, renormalize,
                                  bias, scale, extra, s);
    break;
  case SOFTMAX:
    rc = dispatch<InputType, SOFTMAX>(x, weights, ids, rows, experts, k,
                                      renormalize, bias, scale, extra, s);
    break;
  case SIGMOID:
    rc = dispatch<InputType, SIGMOID>(x, weights, ids, rows, experts, k,
                                      renormalize, bias, scale, extra, s);
    break;
  default:
    return 1;
  }
  if (rc != 0) {
    return rc;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

} // namespace route

// The expert counts route_* serves, for the caller's dispatch and its tests.
extern "C" int route_experts(const int **table) {
  *table = route::EXPERT_COUNTS;
  return sizeof(route::EXPERT_COUNTS) / sizeof(route::EXPERT_COUNTS[0]);
}

// score: 0 raw, 1 softmax, 2 sigmoid. weight: 0 the score, 1 sigmoid(logit).
// Returns 0, 1 for an unsupported expert count or score, 2 for a launch error.
#define ROUTE_ENTRY(NAME, T)                                                   \
  extern "C" int NAME(const void *logits, float *weights, uint32_t *ids,       \
                      const float *bias, const float *expert_scale, int rows,  \
                      int experts, int k, int score, int weight,               \
                      bool renormalize, bool clip, float lo, float hi,         \
                      float norm_min, float scale, int64_t stream) {           \
    return route::route<T>(logits, weights, ids, bias, expert_scale, rows,     \
                           experts, k, score, weight, renormalize, clip, lo,   \
                           hi, norm_min, scale, stream);                       \
  }
ROUTE_ENTRY(route_f32, float)
ROUTE_ENTRY(route_bf16, __nv_bfloat16)
ROUTE_ENTRY(route_f16, __half)
