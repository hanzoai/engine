/**
 * NVFP4 routed experts (Flash-Next E1) on sm_121a, as FlashInfer's cutlass_fused_moe computes
 * them at vLLM's production tactic, in six launches:
 *
 *   route    stable counting sort by expert (route.cu)
 *   expand   quantize each token once (e2m1.cuh, the served scaled_fp4_quant formula), write its
 *            codes to its k sorted rows and its scales into each group's swizzled block; the
 *            trailing CTAs fill both GEMMs' grouped arguments from the device-side route
 *   gemm1    CUTLASS sm120 pointer-array block-scaled grouped GEMM, D = bf16(alpha1[e] * acc)
 *   act      FlashInfer's fast-math SwiGLU (transcribed from fused_moe_120.so SASS) rounded to
 *            bf16, then requantized with e2m1.cuh against gs2
 *   gemm2    the same GEMM, D = bf16(alpha2[e] * acc)
 *   combine  FlashInfer's finalize: fma.rn.ftz over slots from +0, one bf16 rounding
 *
 * Group g of the grouped GEMMs is the g-th active expert; its activation scales start at row
 * roundup(offsets[e] + g*127, 128), FlashInfer's getOffsetActivationSF with a compact index, so
 * no group's padded 128-row blocks overlap the next one's. Nothing synchronizes with the host.
 *
 * Built without fast math and with --fmad=false; every non-IEEE instruction a reference uses is
 * inline PTX.
 */

#include "moe.h"

#include "../quantize/e2m1.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

static_assert(CUDART_VERSION >= 13000);

namespace moe_nvfp4 {
using namespace cute;

constexpr int VEC = 16;       // values per E4M3 block scale
constexpr int ALIGN = 256;    // workspace region alignment
constexpr int SF_ROWS = 128;  // swizzled scale atom rows
constexpr int SF_COLS = 4;    // swizzled scale atom columns
constexpr int ARG_THREADS = 128;

// ------------------------------------------------------------------------------------------
// The grouped GEMM: FlashInfer's SM120 TMA warp-specialized MoE GEMM at tactic 0 (P), plus the
// swap-AB decode tiles (weights as A, tokens as B, D column-major = row-major [tokens, N]).
// ------------------------------------------------------------------------------------------

template <class TileShape, bool Swap> struct Gemm {
  static constexpr bool kSwap = Swap;
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using Element = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using Packed = cutlass::float_e2m1_t;
  using SF = cutlass::float_ue4m3_t;
  using D = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD =
      std::conditional_t<Swap, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  static constexpr int kAlignAB = 32;
  static constexpr int kAlignD = 8;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape,
      Shape<_1, _1, _1>, cutlass::epilogue::collective::EpilogueTileAuto, float, float, D,
      LayoutD *, kAlignD, D, LayoutD *, kAlignD,
      cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Element, LayoutA *,
      kAlignAB, Element, LayoutB *, kAlignAB, float, TileShape, Shape<_1, _1, _1>,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue, void, void>;
  using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using Shape3 = typename ProblemShape::UnderlyingProblemShape;
  using StrideA = typename Kernel::InternalStrideA;
  using StrideB = typename Kernel::InternalStrideB;
  using StrideD = typename Kernel::InternalStrideD;
  using LayoutSFA = typename Mainloop::InternalLayoutSFA;
  using LayoutSFB = typename Mainloop::InternalLayoutSFB;
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<VEC>;
};

using TileP = Gemm<Shape<_128, _128, _128>, false>;
using TileD32 = Gemm<Shape<_128, _32, _128>, true>;
using TileD64 = Gemm<Shape<_128, _64, _128>, true>;

size_t up(size_t x, size_t a = ALIGN) { return (x + a - 1) / a * a; }

/// Device arrays of one grouped GEMM's per-group arguments, carved from the workspace.
template <class G> struct Args {
  typename G::Shape3 *shape;
  const typename G::Packed **a;
  const typename G::Packed **b;
  const typename G::SF **sfa;
  const typename G::SF **sfb;
  typename G::D **d;
  const float **alpha;
  typename G::StrideA *da;
  typename G::StrideB *db;
  typename G::StrideD *dd;
  typename G::LayoutSFA *lsfa;
  typename G::LayoutSFB *lsfb;

  static size_t bytes(int groups) {
    size_t n = groups;
    return up(n * sizeof(typename G::Shape3)) + 6 * up(n * sizeof(void *)) +
           up(n * sizeof(typename G::StrideA)) + up(n * sizeof(typename G::StrideB)) +
           up(n * sizeof(typename G::StrideD)) + up(n * sizeof(typename G::LayoutSFA)) +
           up(n * sizeof(typename G::LayoutSFB));
  }

  static Args carve(char *base, int groups) {
    size_t n = groups;
    Args r;
    auto take = [&](size_t bytes) {
      char *p = base;
      base += up(bytes);
      return p;
    };
    r.shape = reinterpret_cast<typename G::Shape3 *>(take(n * sizeof(typename G::Shape3)));
    r.a = reinterpret_cast<const typename G::Packed **>(take(n * sizeof(void *)));
    r.b = reinterpret_cast<const typename G::Packed **>(take(n * sizeof(void *)));
    r.sfa = reinterpret_cast<const typename G::SF **>(take(n * sizeof(void *)));
    r.sfb = reinterpret_cast<const typename G::SF **>(take(n * sizeof(void *)));
    r.d = reinterpret_cast<typename G::D **>(take(n * sizeof(void *)));
    r.alpha = reinterpret_cast<const float **>(take(n * sizeof(void *)));
    r.da = reinterpret_cast<typename G::StrideA *>(take(n * sizeof(typename G::StrideA)));
    r.db = reinterpret_cast<typename G::StrideB *>(take(n * sizeof(typename G::StrideB)));
    r.dd = reinterpret_cast<typename G::StrideD *>(take(n * sizeof(typename G::StrideD)));
    r.lsfa = reinterpret_cast<typename G::LayoutSFA *>(take(n * sizeof(typename G::LayoutSFA)));
    r.lsfb = reinterpret_cast<typename G::LayoutSFB *>(take(n * sizeof(typename G::LayoutSFB)));
    return r;
  }
};

/// One GEMM of the pair, as the args kernel sees it: the activation side (sorted rows starting
/// at `act`, their scales at `act_sf`), the expert side, and the output.
struct Operand {
  const uint8_t *act;
  const uint8_t *act_sf;
  const uint8_t *w;
  const uint8_t *w_sf;
  const float *alpha;
  __nv_bfloat16 *d;
  int n, k;
};

__host__ __device__ inline long sf_cols(int k) { return (k / VEC + SF_COLS - 1) / SF_COLS * SF_COLS; }

/// First swizzled scale row of compact group g whose rows start at sorted position `offset`.
__host__ __device__ inline long sf_base_row(long offset, int g) {
  return (offset + (long)g * (SF_ROWS - 1) + SF_ROWS - 1) / SF_ROWS * SF_ROWS;
}

/// Byte of (row, col) in a 128x4-atom swizzled scale block with `cols` (padded) columns.
__host__ __device__ inline long sf_offset(long row, long col, long cols) {
  return (((row / SF_ROWS) * (cols / SF_COLS) + col / SF_COLS) * 32 + row % 32) * 16 +
         (row % SF_ROWS) / 32 * 4 + col % 4;
}

/// Scale rows a workspace must hold for R sorted rows over G groups.
inline long sf_rows(long r, int g) { return sf_base_row(r, g) + SF_ROWS; }

template <class G>
__device__ void fill(const Args<G> &args, const Operand &op, int g, int e, int rows, int off) {
  using Packed = typename G::Packed;
  using SF = typename G::SF;
  const int n = op.n, k = op.k;
  const bool live = e >= 0;
  const int ee = live ? e : 0;
  const auto *act = reinterpret_cast<const Packed *>(op.act + (long)off * (k / 2));
  const auto *act_sf =
      reinterpret_cast<const SF *>(op.act_sf + sf_base_row(off, g) * sf_cols(k));
  const auto *w = reinterpret_cast<const Packed *>(op.w + (long)ee * n * (k / 2));
  const auto *w_sf = reinterpret_cast<const SF *>(op.w_sf + (long)ee * n * (k / VEC));
  if constexpr (G::kSwap) {
    args.shape[g] = typename G::Shape3(n, rows, k);
    args.a[g] = w;
    args.b[g] = act;
    args.sfa[g] = w_sf;
    args.sfb[g] = act_sf;
    args.da[g] = cutlass::make_cute_packed_stride(typename G::StrideA{}, make_shape(n, k, 1));
    args.db[g] = cutlass::make_cute_packed_stride(typename G::StrideB{}, make_shape(rows, k, 1));
    args.dd[g] = cutlass::make_cute_packed_stride(typename G::StrideD{}, make_shape(n, rows, 1));
    args.lsfa[g] = G::Config::tile_atom_to_shape_SFA(make_shape(n, rows, k, 1));
    args.lsfb[g] = G::Config::tile_atom_to_shape_SFB(make_shape(n, rows, k, 1));
  } else {
    args.shape[g] = typename G::Shape3(rows, n, k);
    args.a[g] = act;
    args.b[g] = w;
    args.sfa[g] = act_sf;
    args.sfb[g] = w_sf;
    args.da[g] = cutlass::make_cute_packed_stride(typename G::StrideA{}, make_shape(rows, k, 1));
    args.db[g] = cutlass::make_cute_packed_stride(typename G::StrideB{}, make_shape(n, k, 1));
    args.dd[g] = cutlass::make_cute_packed_stride(typename G::StrideD{}, make_shape(rows, n, 1));
    args.lsfa[g] = G::Config::tile_atom_to_shape_SFA(make_shape(rows, n, k, 1));
    args.lsfb[g] = G::Config::tile_atom_to_shape_SFB(make_shape(rows, n, k, 1));
  }
  args.d[g] = reinterpret_cast<typename G::D *>(op.d + (long)off * n);
  args.alpha[g] = op.alpha + ee;
}

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

/// Quantizes 16 values with e2m1.cuh; returns the scale byte and writes 8 code bytes.
__device__ __forceinline__ uint8_t quant16(const float (&v)[VEC], float gs, uint2 &codes) {
  float vmax = 0.0f;
#pragma unroll
  for (int j = 0; j < VEC; j++)
    vmax = fmaxf(vmax, fabsf(v[j]));
  const e2m1::Block blk = e2m1::block(vmax, gs);
  uint32_t w[2] = {0, 0};
#pragma unroll
  for (int j = 0; j < VEC; j++) {
    const uint32_t c = e2m1::encode(v[j] * blk.out);
    w[j / 8] |= c << ((j % 8) * 4);
  }
  codes = make_uint2(w[0], w[1]);
  return blk.scale;
}

/// expand: CTAs [0, m) quantize one token each and scatter it to its k rows; CTAs past m fill
/// the grouped arguments of both GEMMs.
template <class G>
__global__ void expand_kernel(const __nv_bfloat16 *__restrict__ x, const int32_t *__restrict__ ids,
                              const int32_t *__restrict__ offsets, const int32_t *__restrict__ dst,
                              const int32_t *__restrict__ group, const int32_t *__restrict__ active,
                              int m, int k, int h, float gs, uint8_t *a1, uint8_t *s1,
                              Args<G> args1, Operand op1, Args<G> args2, Operand op2, int groups) {
  if (blockIdx.x >= m) {
    const int g = (blockIdx.x - m) * blockDim.x + threadIdx.x;
    if (g >= groups)
      return;
    const int e = active[g];
    const int off = e >= 0 ? offsets[e] : 0;
    const int rows = e >= 0 ? offsets[e + 1] - off : 0;
    fill<G>(args1, op1, g, e, rows, off);
    fill<G>(args2, op2, g, e, rows, off);
    return;
  }
  const long t = blockIdx.x;
  const long cols = sf_cols(h);
  __shared__ long pos[32], row[32], base[32];
  if (threadIdx.x < k) {
    const int f = t * k + threadIdx.x;
    const int e = ids[f];
    pos[threadIdx.x] = dst[f];
    row[threadIdx.x] = dst[f] - offsets[e];
    base[threadIdx.x] = sf_base_row(offsets[e], group[e]) * cols;
  }
  __syncthreads();
  for (int b = threadIdx.x; b < h / VEC; b += blockDim.x) {
    const uint4 *src = reinterpret_cast<const uint4 *>(x + t * h + b * VEC);
    const uint4 lo = src[0], hi = src[1];
    const __nv_bfloat16 *p = reinterpret_cast<const __nv_bfloat16 *>(&lo);
    const __nv_bfloat16 *q = reinterpret_cast<const __nv_bfloat16 *>(&hi);
    float v[VEC];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      v[j] = bf(p[j]);
      v[8 + j] = bf(q[j]);
    }
    uint2 codes;
    const uint8_t scale = quant16(v, gs, codes);
    for (int j = 0; j < k; j++) {
      reinterpret_cast<uint2 *>(a1 + pos[j] * (h / 2))[b] = codes;
      s1[base[j] + sf_offset(row[j], b, cols)] = scale;
    }
  }
}

/// FlashInfer's GLUAdaptor<SiLu> under -use_fast_math on sm_121a (fused_moe_120.so,
/// doActivationKernel<fp4, bf16, bf16, GLUAdaptor<SiLu>, NVFP4>), instruction for instruction:
///   FMUL.FTZ t, x, -log2(e); MUFU.EX2; FADD.FTZ +1; MUFU.RCP; FMUL.FTZ r*x; FMUL.FTZ *u;
///   FMUL.FTZ *quant_scale (1.0); F2F.BF16.F32.
__device__ __forceinline__ float swiglu_bf16(float x, float u) {
  float t, s;
  asm("mul.ftz.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(x));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(t) : "f"(t));
  asm("add.ftz.f32 %0, %1, 0f3F800000;" : "=f"(t) : "f"(t));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(t) : "f"(t));
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(s) : "f"(t), "f"(x));
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(s) : "f"(s), "f"(u));
  asm("mul.ftz.f32 %0, %1, 0f3F800000;" : "=f"(s) : "f"(s));
  return bf(__float2bfloat16_rn(s));
}

/// act: one thread per (sorted row, 16-block of I). Gate is y1[:, :I], up is y1[:, I:].
__global__ void act_kernel(const __nv_bfloat16 *__restrict__ y1, const int32_t *__restrict__ ids,
                           const int32_t *__restrict__ src, const int32_t *__restrict__ offsets,
                           const int32_t *__restrict__ group, long rows, int inter, float gs,
                           uint8_t *a2, uint8_t *s2) {
  const int blocks = inter / VEC;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * blocks)
    return;
  const long r = idx / blocks;
  const int b = idx % blocks;
  const __nv_bfloat16 *row = y1 + r * 2 * inter;
  const uint4 *gp = reinterpret_cast<const uint4 *>(row + b * VEC);
  const uint4 *up_ = reinterpret_cast<const uint4 *>(row + inter + b * VEC);
  uint4 graw[2] = {gp[0], gp[1]};
  uint4 uraw[2] = {up_[0], up_[1]};
  const __nv_bfloat16 *gv = reinterpret_cast<const __nv_bfloat16 *>(graw);
  const __nv_bfloat16 *uv = reinterpret_cast<const __nv_bfloat16 *>(uraw);
  float v[VEC];
#pragma unroll
  for (int j = 0; j < VEC; j++)
    v[j] = swiglu_bf16(bf(gv[j]), bf(uv[j]));
  uint2 codes;
  const uint8_t scale = quant16(v, gs, codes);
  reinterpret_cast<uint2 *>(a2 + r * (inter / 2))[b] = codes;
  const int e = ids[src[r]];
  const long cols = sf_cols(inter);
  const long base = sf_base_row(offsets[e], group[e]) * cols;
  s2[base + sf_offset(r - offsets[e], b, cols)] = scale;
}

int status(cudaError_t e) { return e == cudaSuccess ? 0 : -(int)e; }

template <class G> int prepare_tile() {
  if constexpr (G::Kernel::SharedStorageSize >= 48 * 1024) {
    auto e = cudaFuncSetAttribute(cutlass::device_kernel<typename G::Kernel>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  G::Kernel::SharedStorageSize);
    if (e != cudaSuccess)
      return status(e);
  }
  return 0;
}

template <class G>
typename G::Adapter::Arguments arguments(const Args<G> &a, int groups, int sm_count) {
  typename G::Adapter::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
  args.problem_shape = {groups, a.shape, nullptr};
  args.mainloop.ptr_A = a.a;
  args.mainloop.dA = a.da;
  args.mainloop.ptr_B = a.b;
  args.mainloop.dB = a.db;
  args.mainloop.ptr_SFA = a.sfa;
  args.mainloop.layout_SFA = a.lsfa;
  args.mainloop.ptr_SFB = a.sfb;
  args.mainloop.layout_SFB = a.lsfb;
  auto &fusion = args.epilogue.thread;
  fusion.alpha = 1.0f;
  fusion.beta = 0.0f;
  fusion.alpha_ptr_array = a.alpha;
  fusion.dAlpha = {_0{}, _0{}, 1};
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = nullptr;
  args.epilogue.ptr_D = a.d;
  args.epilogue.dD = a.dd;
  args.hw_info.device_id = 0;
  args.hw_info.sm_count = sm_count;
  args.scheduler.max_swizzle_size = 1;
  args.scheduler.raster_order = G::Kernel::TileScheduler::RasterOrderOptions::AlongN;
  return args;
}

template <class G> size_t gemm_workspace(int groups, int sm_count) {
  Args<G> a{};
  auto args = arguments<G>(a, groups, sm_count);
  return G::Adapter::get_workspace_size(args);
}

template <class G>
int run_gemm(const Args<G> &a, int groups, int sm_count, void *ws, cudaStream_t stream) {
  auto args = arguments<G>(a, groups, sm_count);
  auto st = G::Kernel::initialize_workspace(args, ws, stream, nullptr);
  if (st != cutlass::Status::kSuccess)
    return (int)st;
  auto params = G::Kernel::to_underlying_arguments(args, ws);
  st = G::Adapter::run(params, stream, nullptr, false);
  if (st != cutlass::Status::kSuccess)
    return (int)st;
  return status(cudaGetLastError());
}

bool valid(const hanzo_moe_shape &s) {
  return s.m > 0 && s.k > 0 && s.k <= 32 && s.e > 0 && s.e <= HANZO_MOE_MAX_EXPERTS &&
         s.h > 0 && s.h % 128 == 0 && s.i > 0 && s.i % 128 == 0 && s.sm_count > 0 &&
         s.tile >= 0 && s.tile < HANZO_MOE_TILES && (long)s.m * s.k < (1L << 30);
}

template <class G> int layout_for(const hanzo_moe_shape &s, hanzo_moe_layout *l) {
  const long R = (long)s.m * s.k;
  const int groups = (int)(R < s.e ? R : s.e);
  size_t at = 0;
  auto take = [&](size_t bytes) {
    size_t p = at;
    at += up(bytes);
    return p;
  };
  *l = hanzo_moe_layout{};
  l->offsets = take((size_t)(s.e + 1) * 4);
  l->src = take(R * 4);
  l->dst = take(R * 4);
  l->group = take((size_t)s.e * 4);
  l->active = take((size_t)groups * 4);
  l->nactive = take(4);
  l->route = take(hanzo_moe_route_scratch((int)R, s.e));
  l->args1 = take(Args<G>::bytes(groups));
  l->args2 = take(Args<G>::bytes(groups));
  l->gemm = take(gemm_workspace<G>(groups, s.sm_count));
  // Region A: the act output, live through GEMM2.
  l->a2 = take(R * (s.i / 2));
  l->s2 = take(sf_rows(R, groups) * sf_cols(s.i));
  // Region S: GEMM1's operands and output, then GEMM2's output over them.
  const size_t s_start = at;
  l->a1 = take(R * (s.h / 2));
  l->s1 = take(sf_rows(R, groups) * sf_cols(s.h));
  l->y1 = take(R * 2 * (size_t)s.i * 2);
  const size_t s_end = at;
  l->y2 = s_start;
  const size_t y2_end = s_start + up(R * (size_t)s.h * 2);
  at = s_end > y2_end ? s_end : y2_end;
  l->w = 0;
  l->xq = 0;
  l->xs = 0;
  l->total = at;
  return 0;
}

int layout(const hanzo_moe_shape &s, hanzo_moe_layout *l) {
  switch (s.tile) {
  case HANZO_MOE_TILE_P:
    return layout_for<TileP>(s, l);
  case HANZO_MOE_TILE_D32:
    return layout_for<TileD32>(s, l);
  default:
    return layout_for<TileD64>(s, l);
  }
}

template <class G> int run(const hanzo_moe_nvfp4_launch &L, int first, int last) {
  const hanzo_moe_shape &s = L.shape;
  hanzo_moe_layout l;
  layout_for<G>(s, &l);
  char *ws = static_cast<char *>(L.workspace);
  const long R = (long)s.m * s.k;
  const int groups = (int)(R < s.e ? R : s.e);
  auto *offsets = reinterpret_cast<int32_t *>(ws + l.offsets);
  auto *src = reinterpret_cast<int32_t *>(ws + l.src);
  auto *dst = reinterpret_cast<int32_t *>(ws + l.dst);
  auto *group = reinterpret_cast<int32_t *>(ws + l.group);
  auto *active = reinterpret_cast<int32_t *>(ws + l.active);
  auto *nactive = reinterpret_cast<int32_t *>(ws + l.nactive);
  auto *a1 = reinterpret_cast<uint8_t *>(ws + l.a1);
  auto *s1 = reinterpret_cast<uint8_t *>(ws + l.s1);
  auto *y1 = reinterpret_cast<__nv_bfloat16 *>(ws + l.y1);
  auto *a2 = reinterpret_cast<uint8_t *>(ws + l.a2);
  auto *s2 = reinterpret_cast<uint8_t *>(ws + l.s2);
  auto *y2 = reinterpret_cast<__nv_bfloat16 *>(ws + l.y2);
  Args<G> args1 = Args<G>::carve(ws + l.args1, groups);
  Args<G> args2 = Args<G>::carve(ws + l.args2, groups);
  const cudaStream_t st = L.stream;
  int rc = 0;
  for (int stage = first; stage <= last && rc == 0; stage++) {
    switch (stage) {
    case HANZO_MOE_ROUTE:
      rc = hanzo_moe_route(L.ids, s.m, s.k, s.e, offsets, src, dst, group, active, nactive,
                           ws + l.route, nullptr, st);
      break;
    case HANZO_MOE_EXPAND: {
      const Operand op1{a1, s1, L.w13, L.w13_sf, L.alpha1, y1, 2 * s.i, s.h};
      const Operand op2{a2, s2, L.w2, L.w2_sf, L.alpha2, y2, s.h, s.i};
      const int arg_blocks = (groups + ARG_THREADS - 1) / ARG_THREADS;
      expand_kernel<G><<<s.m + arg_blocks, ARG_THREADS, 0, st>>>(
          L.x, L.ids, offsets, dst, group, active, s.m, s.k, s.h, L.gs1, a1, s1, args1, op1,
          args2, op2, groups);
      rc = status(cudaGetLastError());
      break;
    }
    case HANZO_MOE_GEMM1:
      rc = run_gemm<G>(args1, groups, s.sm_count, ws + l.gemm, st);
      break;
    case HANZO_MOE_ACT: {
      const long threads = R * (s.i / VEC);
      act_kernel<<<(unsigned)((threads + 255) / 256), 256, 0, st>>>(
          y1, L.ids, src, offsets, group, R, s.i, L.gs2, a2, s2);
      rc = status(cudaGetLastError());
      break;
    }
    case HANZO_MOE_GEMM2:
      rc = run_gemm<G>(args2, groups, s.sm_count, ws + l.gemm, st);
      break;
    case HANZO_MOE_COMBINE:
      rc = hanzo_moe_combine(y2, dst, L.weights, L.addend, L.out, s.m, s.k, s.h,
                             HANZO_MOE_FINALIZE, st);
      break;
    default:
      rc = HANZO_MOE_BAD_ARGUMENT;
    }
  }
  return rc;
}

} // namespace moe_nvfp4

using namespace moe_nvfp4;

extern "C" int hanzo_moe_nvfp4_prepare(int device) {
  auto e = cudaSetDevice(device);
  if (e != cudaSuccess)
    return status(e);
  int rc = prepare_tile<TileP>();
  if (!rc)
    rc = prepare_tile<TileD32>();
  if (!rc)
    rc = prepare_tile<TileD64>();
  return rc;
}

extern "C" int hanzo_moe_nvfp4_layout(const hanzo_moe_shape *shape, hanzo_moe_layout *layout) {
  if (!shape || !layout || !valid(*shape))
    return HANZO_MOE_BAD_ARGUMENT;
  return moe_nvfp4::layout(*shape, layout);
}

extern "C" int hanzo_moe_nvfp4_run(const hanzo_moe_nvfp4_launch *launch, int first, int last) {
  if (!launch || !valid(launch->shape) || first < 0 || last > HANZO_MOE_COMBINE || first > last)
    return HANZO_MOE_BAD_ARGUMENT;
  if (reinterpret_cast<uintptr_t>(launch->workspace) % ALIGN != 0)
    return HANZO_MOE_UNALIGNED;
  switch (launch->shape.tile) {
  case HANZO_MOE_TILE_P:
    return run<TileP>(*launch, first, last);
  case HANZO_MOE_TILE_D32:
    return run<TileD32>(*launch, first, last);
  default:
    return run<TileD64>(*launch, first, last);
  }
}
