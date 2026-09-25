/**
 * @brief Dummy FP8 GEMM kernels for GPUs that don't support FP8 (CC < 8.0).
 *
 * These are stub implementations that will never be called but are needed
 * for linking when FP8 is not supported.
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Dummy FP8 type for compilation
struct __nv_fp8_e4m3_dummy {
  unsigned char __x;
};

#define W8A8_DUMMY(NAME, T)                                                                    \
  extern "C" void launch_fp8_matmul_##NAME(const uint8_t *, const float *, const uint8_t *,      \
                                           const float *, T *, int, int, int, int, int,          \
                                           cudaStream_t) {                                       \
    fprintf(stderr, "FP8 matmul not supported on this GPU (requires compute "                   \
                    "capability >= 8.0)\n");                                                     \
  }                                                                                              \
  extern "C" void launch_fp8_indexed_moe_gemm_##NAME(                                            \
      const uint8_t *, const float *, const uint8_t *, const float *, const uint32_t *, T *,     \
      int, int, int, int, int, int, int, bool, cudaStream_t) {                                   \
    fprintf(stderr, "FP8 indexed MoE GEMM not supported on this GPU (requires "                 \
                    "compute capability >= 8.0)\n");                                             \
  }

W8A8_DUMMY(f32, float)
W8A8_DUMMY(f16, __half)
W8A8_DUMMY(bf16, __nv_bfloat16)
