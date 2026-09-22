#ifndef HANZO_NVFP4_CUTLASS_H
#define HANZO_NVFP4_CUTLASS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { HANZO_NVFP4_BF16 = 0, HANZO_NVFP4_F16 = 1 };
enum { HANZO_NVFP4_PREFILL = 0, HANZO_NVFP4_DECODE_DP32 = 1 };

typedef struct hanzo_nvfp4_context {
  int32_t device;
  int32_t sm_count;
  int32_t dtype;
  int32_t kernel;
} hanzo_nvfp4_context;

typedef struct hanzo_nvfp4_shape {
  int32_t m;
  int32_t n;
  int32_t k;
} hanzo_nvfp4_shape;

typedef struct hanzo_nvfp4_launch {
  hanzo_nvfp4_shape shape;
  hanzo_nvfp4_context context;
  const void *a_packed;
  const void *w_packed;
  const void *a_scale_swizzled;
  const void *w_scale_swizzled;
  const float *weight_global;
  const float *activation_global;
  void *output;
  void *workspace;
  size_t workspace_bytes;
  void *stream;
} hanzo_nvfp4_launch;

typedef struct hanzo_nvfp4_resources {
  hanzo_nvfp4_context context;
  int32_t major;
  int32_t minor;
  int32_t threads;
  int32_t registers_per_thread;
  size_t shared_bytes;
  size_t local_bytes;
} hanzo_nvfp4_resources;

/* Success is zero, CUDA errors are negative, CUTLASS errors are positive. */
const char *hanzo_nvfp4_error_string(int status);

/* Prepare outside capture; the requested device must already be current. */
int hanzo_nvfp4_prepare(int32_t device, int32_t dtype, int32_t kernel,
                            hanzo_nvfp4_resources *resources);
int hanzo_nvfp4_workspace_size(const hanzo_nvfp4_context *context,
                                   const hanzo_nvfp4_shape *shape,
                                   size_t *bytes);

// The device, stream, and buffers must match the prepared context.
// Operands and workspace must outlive eager work and graph replay.
// Align A/W/scales/output/workspace to 16 bytes and global scales to 4 bytes.
// Output is RN_dtype((FP32_acc * weight_global[n]) * activation_global[0]).
// Bias is applied separately.
int hanzo_nvfp4_gemm(const hanzo_nvfp4_launch *launch);

// Swizzling preserves bytes and zeroes padding; the buffers must not overlap.
int hanzo_nvfp4_scale_bytes(int32_t rows, int32_t k, size_t *bytes);
int hanzo_nvfp4_swizzle_host(const void *source, void *dest, int32_t rows,
                                 int32_t k, size_t dest_bytes);
int hanzo_nvfp4_swizzle_cuda(const void *source, void *dest, int32_t rows,
                                 int32_t k, size_t dest_bytes, void *stream);

#ifdef __cplusplus
}
#endif
#endif
