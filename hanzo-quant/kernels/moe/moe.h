/**
 * Routed-expert MoE on sm_121a: route, expand, grouped GEMM, act, grouped GEMM, combine.
 *
 * Every size a launch depends on is fixed by (M, k, E), and every per-expert size lives only on
 * the device, so a forward never synchronizes with the host and can be captured in a graph.
 *
 * Status: 0 ok, negative a CUDA error (-cudaError_t), positive a CUTLASS status, 1000+ a bad
 * argument.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HANZO_MOE_BAD_ARGUMENT 1000
#define HANZO_MOE_UNALIGNED 1001

/* Largest expert count route supports (one CTA scans the histogram). */
#define HANZO_MOE_MAX_EXPERTS 1024
/* Flat rows one routing CTA ranks. */
#define HANZO_MOE_ROUTE_CHUNK 4096

/* Combine rules. */
#define HANZO_MOE_FINALIZE 0 /* s = fma.rn.ftz(w, y, s): FlashInfer finalizeMoeRoutingKernel */
#define HANZO_MOE_SUM 1      /* s = add.rn(s, y): vLLM moe_sum; the weight is already in y */

/* GEMM tiles. P is 128x128x128; D32 and D64 swap A and B (weights as A) with N = 32, 64. */
#define HANZO_MOE_TILE_P 0
#define HANZO_MOE_TILE_D32 1
#define HANZO_MOE_TILE_D64 2
#define HANZO_MOE_TILES 3

const char *hanzo_moe_error_string(int status);

/* Sets every kernel's shared-memory attribute; call once per device before capture. */
int hanzo_moe_prepare(int device);

/* ---------------------------------------------------------------------------------------- */
/* route: stable counting sort of the R = M*k flat assignments f = t*k + j by expert.         */
/* ---------------------------------------------------------------------------------------- */

/* Scratch bytes route needs for R rows over E experts (0 when one CTA suffices). */
size_t hanzo_moe_route_scratch(int r, int e);

/* ids [M,k] int32 in [0,E). Outputs:
 *   offsets [E+1]  rows of expert e are sorted positions [offsets[e], offsets[e+1])
 *   src     [R]    sorted position -> flat row f
 *   dst     [R]    flat row f -> sorted position
 *   group   [E]    compact index of expert e among the active ones, or -1
 *   active  [G]    G = min(E, R): expert of compact group g, -1 past nactive
 *   nactive [1]
 * launches, when not null, receives how many kernels ran (1 when R <= 4096, else 3). */
int hanzo_moe_route(const int32_t *ids, int m, int k, int e, int32_t *offsets, int32_t *src,
                    int32_t *dst, int32_t *group, int32_t *active, int32_t *nactive,
                    void *scratch, int *launches, cudaStream_t stream);

/* ---------------------------------------------------------------------------------------- */
/* combine: out[t] = bf16_rn(sum over slots j = 0..k-1, from +0, of y[dst[t*k+j]])            */
/* ---------------------------------------------------------------------------------------- */

/* y [R,H] bf16; weights [M,k] f32 (FINALIZE only; ignored by SUM); addend [M,H] bf16 or null:
 * out = bf16_rn(f32(bf16_rn(s)) + f32(addend)). */
int hanzo_moe_combine(const __nv_bfloat16 *y, const int32_t *dst, const float *weights,
                      const __nv_bfloat16 *addend, __nv_bfloat16 *out, int m, int k, int h,
                      int rule, cudaStream_t stream);

/* ---------------------------------------------------------------------------------------- */
/* One forward's workspace: route outputs, both GEMMs' operands, CUTLASS arguments.          */
/* ---------------------------------------------------------------------------------------- */

typedef struct {
  int m;        /* tokens */
  int k;        /* experts per token */
  int e;        /* experts */
  int h;        /* hidden size (GEMM1 K, GEMM2 N) */
  int i;        /* intermediate size (GEMM1 N = 2i, GEMM2 K) */
  int tile;     /* HANZO_MOE_TILE_* */
  int sm_count; /* persistent CTAs */
} hanzo_moe_shape;

/* Byte offsets into one workspace. Region S (a1, s1, y1) is dead once act has read y1, so
 * GEMM2's output y2 is written over it. */
typedef struct {
  size_t offsets, src, dst, group, active, nactive, route;
  size_t a1, s1, y1; /* expanded input codes, their scales, GEMM1 output [R, 2i] bf16 */
  size_t a2, s2;     /* act output codes and scales */
  size_t y2;         /* GEMM2 output [R, h] bf16 (aliases a1) */
  size_t w;          /* router weights in sorted order (block FP8 only) */
  size_t xq, xs;     /* per-token quantized input before the k-way scatter (block FP8 only) */
  size_t args1, args2, gemm;
  size_t total;
} hanzo_moe_layout;

/* Stages, in order. A forward runs ROUTE..COMBINE; tests stop after any one. */
#define HANZO_MOE_ROUTE 0
#define HANZO_MOE_EXPAND 1
#define HANZO_MOE_GEMM1 2
#define HANZO_MOE_ACT 3
#define HANZO_MOE_GEMM2 4
#define HANZO_MOE_COMBINE 5

/* ---------------------------------------------------------------------------------------- */
/* NVFP4 experts (E1): block-scaled E2M1 x E2M1 with E4M3 scales per 16, one alpha per      */
/* expert, FlashInfer's SwiGLU and finalize.                                                  */
/* ---------------------------------------------------------------------------------------- */

typedef struct {
  hanzo_moe_shape shape;
  const __nv_bfloat16 *x; /* [m, h] */
  const int32_t *ids;     /* [m, k] */
  const float *weights;   /* [m, k] router weights */
  const __nv_bfloat16 *addend; /* [m, h] or null */
  const uint8_t *w13;     /* [e, 2i, h/2] gate rows then up rows */
  const uint8_t *w13_sf;  /* [e, 2i, h/16] swizzled per expert */
  const float *alpha1;    /* [e] */
  float gs1;              /* 1 / max(w13 input_scale) */
  const uint8_t *w2;      /* [e, h, i/2] */
  const uint8_t *w2_sf;   /* [e, h, i/16] swizzled per expert */
  const float *alpha2;    /* [e] */
  float gs2;              /* 1 / max(w2 input_scale) */
  __nv_bfloat16 *out;     /* [m, h] */
  void *workspace;        /* hanzo_moe_nvfp4_layout(...).total bytes, 256-byte aligned */
  cudaStream_t stream;
} hanzo_moe_nvfp4_launch;

int hanzo_moe_nvfp4_prepare(int device);
int hanzo_moe_nvfp4_layout(const hanzo_moe_shape *shape, hanzo_moe_layout *layout);
/* Runs stages first..last (inclusive). */
int hanzo_moe_nvfp4_run(const hanzo_moe_nvfp4_launch *launch, int first, int last);

/* ---------------------------------------------------------------------------------------- */
/* Block-FP8 experts (E4, the MTP drafter): E4M3 weights with a scale per 128x128 block,      */
/* activations quantized per token and 128-wide group, vLLM's Triton path.                    */
/* ---------------------------------------------------------------------------------------- */

typedef struct {
  hanzo_moe_shape shape;
  const uint8_t *xq;      /* [m, h] E4M3: vLLM per_token_group_quant_fp8 of x (M1's gather mode) */
  const float *xs;        /* [m, h/128] */
  const int32_t *ids;     /* [m, k] */
  const float *weights;   /* [m, k] router weights, applied in GEMM2's epilogue */
  const __nv_bfloat16 *addend; /* [m, h] or null */
  const uint8_t *w13;     /* [e, 2i, h] E4M3, gate rows then up rows */
  const float *w13_s;     /* [e, 2i/128, h/128] */
  const uint8_t *w2;      /* [e, h, i] E4M3 */
  const float *w2_s;      /* [e, h/128, i/128] */
  __nv_bfloat16 *out;     /* [m, h] */
  void *workspace;
  cudaStream_t stream;
} hanzo_moe_fp8_launch;

int hanzo_moe_fp8_prepare(int device);
int hanzo_moe_fp8_layout(const hanzo_moe_shape *shape, hanzo_moe_layout *layout);
int hanzo_moe_fp8_run(const hanzo_moe_fp8_launch *launch, int first, int last);

#ifdef __cplusplus
}
#endif
