#ifndef HANZO_BLOCKWISE_FP8_CUTLASS_H
#define HANZO_BLOCKWISE_FP8_CUTLASS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* vLLM v0.29.0's sm_12x tiles, picked by M: <=64, <=256, larger. */
enum {
  HANZO_BLOCKWISE_FP8_SWAP_AB = 0,     /* 128x32x128 cooperative, A/B swapped */
  HANZO_BLOCKWISE_FP8_PINGPONG = 1,    /* 64x128x128 pingpong */
  HANZO_BLOCKWISE_FP8_COOPERATIVE = 2, /* 128x128x128 cooperative */
};

typedef struct hanzo_blockwise_fp8_context {
  int32_t device;
  int32_t sm_count;
  int32_t tile;
} hanzo_blockwise_fp8_context;

typedef struct hanzo_blockwise_fp8_shape {
  int32_t m;
  int32_t n;
  int32_t k;
} hanzo_blockwise_fp8_shape;

/*
 * a: e4m3 [M,K] row-major.        a_scale: f32 [M,K/128] row-major.
 * w: e4m3 [N,K] row-major.        w_scale: f32 [N/128,K/128] row-major.
 * output: bf16 [M,N] row-major = RN_bf16(sum_kb RN(sa*sw) * (a.w)_kb).
 */
typedef struct hanzo_blockwise_fp8_launch {
  hanzo_blockwise_fp8_shape shape;
  hanzo_blockwise_fp8_context context;
  const void *a;
  const float *a_scale;
  const void *w;
  const float *w_scale;
  void *output;
  void *workspace;
  size_t workspace_bytes;
  void *stream;
} hanzo_blockwise_fp8_launch;

typedef struct hanzo_blockwise_fp8_resources {
  hanzo_blockwise_fp8_context context;
  int32_t major;
  int32_t minor;
  int32_t threads;
  int32_t registers_per_thread;
  size_t shared_bytes;
  size_t local_bytes;
} hanzo_blockwise_fp8_resources;

/* Success is zero, CUDA errors are negative, CUTLASS errors are positive. */
const char *hanzo_blockwise_fp8_error_string(int status);

/* Prepare outside capture; the requested device must already be current. */
int hanzo_blockwise_fp8_prepare(int32_t device, int32_t tile,
                                hanzo_blockwise_fp8_resources *resources);
int hanzo_blockwise_fp8_workspace_size(const hanzo_blockwise_fp8_context *context,
                                       const hanzo_blockwise_fp8_shape *shape,
                                       size_t *bytes);

/* M >= 1, N and K multiples of 128. Operands and output 16-byte aligned,
 * scales 4-byte aligned. Buffers must outlive eager work and graph replay. */
int hanzo_blockwise_fp8_gemm(const hanzo_blockwise_fp8_launch *launch);

#ifdef __cplusplus
}
#endif
#endif
