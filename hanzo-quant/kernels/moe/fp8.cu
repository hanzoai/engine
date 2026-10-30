/**
 * Block-FP8 routed experts (Flash-Next E4, the MTP drafter's MoE) on sm_121a, as vLLM's
 * TritonExperts computes them with VLLM_USE_DEEP_GEMM=0:
 *
 *   route    stable counting sort by expert (route.cu)
 *   (quant)  per_token_group_quant_fp8 of x, M1's gather-mode quantizer, launched by the caller
 *   expand   scatter each token's codes and group scales to its k sorted rows, gather the router
 *            weights into sorted order; trailing CTAs fill both GEMMs' grouped arguments
 *   gemm1    CUTLASS sm120 pointer-array blockwise-scaled grouped GEMM (A per token and 128-group,
 *            B per 128x128 block), D = bf16(acc)
 *   act      vLLM silu_and_mul_per_block_quant: r = (g * (1 / (1 + expf(-g)))) * u in f32,
 *            s = max(amax / 448, 1 / (448 * 512)), q = e4m3(clamp(r / s, +-448)), IEEE throughout
 *   gemm2    the same GEMM with an epilogue visitor D = bf16_rn(acc * w[row]): Triton's
 *            `accumulator * moe_weight[:, None]` before the cast
 *   combine  vLLM moe_sum: add.rn over slots from +0, one bf16 rounding
 *
 * Built without fast math and with --fmad=false: every operation is the IEEE one vLLM runs.
 */

#include "moe.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

static_assert(CUDART_VERSION >= 13000);

namespace moe_fp8 {
using namespace cute;
namespace fusion = cutlass::epilogue::fusion;

constexpr int GROUP = 128;
constexpr int ALIGN = 256;
constexpr int ARG_THREADS = 128;
constexpr float FP8_MAX = 448.0f;

template <class TileShape> struct Gemm {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using E = cutlass::float_e4m3_t;
  using D = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ScaleConfig =
      cutlass::detail::Sm120BlockwiseScaleConfig<1, GROUP, GROUP, UMMA::Major::K, UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  // D = bf16_rn(acc * w[row]); w is a per-group pointer into the sorted router weights, or
  // null (GEMM1) for a constant 1.0.
  using Weight = fusion::Sm90ColBroadcast<0, TileShape, float *, float, Stride<_1, _0, int64_t>,
                                          1, true>;
  using Fusion = fusion::Sm90EVT<
      fusion::Sm90Compute<cutlass::multiplies, D, float, cutlass::FloatRoundStyle::round_to_nearest>,
      fusion::Sm90AccFetch, Weight>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD *, 8, D,
      LayoutD *, 8, cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, E, cute::tuple<LayoutA *, LayoutSFA *>,
      16, E, cute::tuple<LayoutB *, LayoutSFB *>, 16, float, TileShape, Shape<_1, _1, _1>,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue, void, void>;
  using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using Shape3 = typename ProblemShape::UnderlyingProblemShape;
  using StrideA = typename Kernel::InternalStrideA;
  using StrideB = typename Kernel::InternalStrideB;
  using StrideD = typename Kernel::InternalStrideD;
};

using TileP = Gemm<Shape<_128, _128, _128>>;

size_t up(size_t x, size_t a = ALIGN) { return (x + a - 1) / a * a; }
int status(cudaError_t e) { return e == cudaSuccess ? 0 : -(int)e; }

template <class G> struct Args {
  typename G::Shape3 *shape;
  const typename G::E **a;
  const typename G::E **b;
  const float **sfa;
  const float **sfb;
  typename G::D **d;
  const float **w;
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
    r.a = reinterpret_cast<const typename G::E **>(take(n * sizeof(void *)));
    r.b = reinterpret_cast<const typename G::E **>(take(n * sizeof(void *)));
    r.sfa = reinterpret_cast<const float **>(take(n * sizeof(void *)));
    r.sfb = reinterpret_cast<const float **>(take(n * sizeof(void *)));
    r.d = reinterpret_cast<typename G::D **>(take(n * sizeof(void *)));
    r.w = reinterpret_cast<const float **>(take(n * sizeof(void *)));
    r.da = reinterpret_cast<typename G::StrideA *>(take(n * sizeof(typename G::StrideA)));
    r.db = reinterpret_cast<typename G::StrideB *>(take(n * sizeof(typename G::StrideB)));
    r.dd = reinterpret_cast<typename G::StrideD *>(take(n * sizeof(typename G::StrideD)));
    r.lsfa = reinterpret_cast<typename G::LayoutSFA *>(take(n * sizeof(typename G::LayoutSFA)));
    r.lsfb = reinterpret_cast<typename G::LayoutSFB *>(take(n * sizeof(typename G::LayoutSFB)));
    return r;
  }
};

struct Operand {
  const uint8_t *act;   // sorted rows, k bytes each
  const float *act_s;   // sorted rows, k/128 scales each
  const uint8_t *w;     // [e, n, k]
  const float *w_s;     // [e, n/128, k/128]
  const float *weights; // sorted router weights, or null
  __nv_bfloat16 *d;
  int n, k;
};

template <class G>
__device__ void fill(const Args<G> &args, const Operand &op, int g, int e, int rows, int off) {
  const int n = op.n, k = op.k;
  const int ee = e >= 0 ? e : 0;
  args.shape[g] = typename G::Shape3(rows, n, k);
  args.a[g] = reinterpret_cast<const typename G::E *>(op.act + (long)off * k);
  args.b[g] = reinterpret_cast<const typename G::E *>(op.w + (long)ee * n * k);
  args.sfa[g] = op.act_s + (long)off * (k / GROUP);
  args.sfb[g] = op.w_s + (long)ee * (n / GROUP) * (k / GROUP);
  args.d[g] = reinterpret_cast<typename G::D *>(op.d + (long)off * n);
  args.w[g] = op.weights ? op.weights + off : nullptr;
  args.da[g] = cutlass::make_cute_packed_stride(typename G::StrideA{}, make_shape(rows, k, 1));
  args.db[g] = cutlass::make_cute_packed_stride(typename G::StrideB{}, make_shape(n, k, 1));
  args.dd[g] = cutlass::make_cute_packed_stride(typename G::StrideD{}, make_shape(rows, n, 1));
  args.lsfa[g] = G::ScaleConfig::tile_atom_to_shape_SFA(make_shape(rows, n, k, 1));
  args.lsfb[g] = G::ScaleConfig::tile_atom_to_shape_SFB(make_shape(rows, n, k, 1));
}

/// expand: CTAs [0, m) copy one token's codes and scales to its k sorted rows and place its
/// router weights; CTAs past m fill both GEMMs' grouped arguments.
template <class G>
__global__ void expand_kernel(const uint8_t *__restrict__ xq, const float *__restrict__ xs,
                              const float *__restrict__ weights, const int32_t *__restrict__ ids,
                              const int32_t *__restrict__ offsets, const int32_t *__restrict__ dst,
                              const int32_t *__restrict__ active, int m, int k, int h,
                              uint8_t *a1, float *s1, float *ws, Args<G> args1, Operand op1,
                              Args<G> args2, Operand op2, int groups) {
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
  const int vecs = h / 16, groups_h = h / GROUP;
  const uint4 *src = reinterpret_cast<const uint4 *>(xq + t * h);
  for (int j = 0; j < k; j++) {
    const long f = t * k + j;
    const long pos = dst[f];
    uint4 *d = reinterpret_cast<uint4 *>(a1 + pos * h);
    for (int c = threadIdx.x; c < vecs; c += blockDim.x)
      d[c] = src[c];
    for (int c = threadIdx.x; c < groups_h; c += blockDim.x)
      s1[pos * groups_h + c] = xs[t * groups_h + c];
    if (threadIdx.x == 0)
      ws[pos] = weights[f];
  }
}

/// act: vLLM silu_and_mul_per_block_quant, one 128-thread CTA per (sorted row, group).
__global__ void __launch_bounds__(GROUP)
    act_kernel(const __nv_bfloat16 *__restrict__ y1, int inter, uint8_t *a2, float *s2) {
  __shared__ float red[GROUP];
  const long r = blockIdx.x;
  const int g = blockIdx.y, tid = threadIdx.x;
  const int groups = inter / GROUP;
  const __nv_bfloat16 *row = y1 + r * 2 * inter;
  const float gate = __bfloat162float(row[g * GROUP + tid]);
  const float upv = __bfloat162float(row[inter + g * GROUP + tid]);
  const float sig = 1.0f / (1.0f + expf(-gate));
  const float result = (gate * sig) * upv;
  red[tid] = fabsf(result);
  __syncthreads();
#pragma unroll
  for (int s = GROUP / 2; s > 0; s >>= 1) {
    if (tid < s)
      red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  float scale = red[0] / FP8_MAX;
  scale = fmaxf(scale, 1.0f / (FP8_MAX * 512.0f));
  if (tid == 0)
    s2[r * groups + g] = scale;
  const float q = fmaxf(-FP8_MAX, fminf(result / scale, FP8_MAX));
  a2[r * inter + g * GROUP + tid] = __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
}

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
typename G::Adapter::Arguments arguments(const Args<G> &a, int groups, int sm_count,
                                         bool weighted) {
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
  // {AccFetch, Weight{ptr_col, null_default, dCol}, Compute}
  args.epilogue.thread = {{}, {weighted ? a.w : nullptr, 1.0f, {}}, {}};
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
  auto args = arguments<G>(a, groups, sm_count, false);
  return G::Adapter::get_workspace_size(args);
}

template <class G>
int run_gemm(const Args<G> &a, int groups, int sm_count, bool weighted, void *ws,
             cudaStream_t stream) {
  auto args = arguments<G>(a, groups, sm_count, weighted);
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
         s.h > 0 && s.h % GROUP == 0 && s.i > 0 && s.i % GROUP == 0 && s.sm_count > 0 &&
         (long)s.m * s.k < (1L << 30);
}

template <class G> void layout_for(const hanzo_moe_shape &s, hanzo_moe_layout *l) {
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
  l->w = take(R * 4);
  // Region A: the act output, live through GEMM2.
  l->a2 = take(R * (size_t)s.i);
  l->s2 = take(R * (size_t)(s.i / GROUP) * 4);
  // Region S: the per-token input, GEMM1's operands and output; GEMM2's output over them.
  const size_t s_start = at;
  l->a1 = take(R * (size_t)s.h);
  l->s1 = take(R * (size_t)(s.h / GROUP) * 4);
  l->y1 = take(R * 2 * (size_t)s.i * 2);
  l->xq = take((size_t)s.m * s.h);
  l->xs = take((size_t)s.m * (s.h / GROUP) * 4);
  const size_t s_end = at;
  l->y2 = s_start;
  const size_t y2_end = s_start + up(R * (size_t)s.h * 2);
  l->total = s_end > y2_end ? s_end : y2_end;
}

int run(const hanzo_moe_fp8_launch &L, int first, int last) {
  using G = TileP;
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
  auto *s1 = reinterpret_cast<float *>(ws + l.s1);
  auto *y1 = reinterpret_cast<__nv_bfloat16 *>(ws + l.y1);
  auto *a2 = reinterpret_cast<uint8_t *>(ws + l.a2);
  auto *s2 = reinterpret_cast<float *>(ws + l.s2);
  auto *y2 = reinterpret_cast<__nv_bfloat16 *>(ws + l.y2);
  auto *wsorted = reinterpret_cast<float *>(ws + l.w);
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
      const Operand op1{a1, s1, L.w13, L.w13_s, nullptr, y1, 2 * s.i, s.h};
      const Operand op2{a2, s2, L.w2, L.w2_s, wsorted, y2, s.h, s.i};
      const int arg_blocks = (groups + ARG_THREADS - 1) / ARG_THREADS;
      expand_kernel<G><<<s.m + arg_blocks, ARG_THREADS, 0, st>>>(
          L.xq, L.xs, L.weights, L.ids, offsets, dst, active, s.m, s.k, s.h, a1, s1, wsorted,
          args1, op1, args2, op2, groups);
      rc = status(cudaGetLastError());
      break;
    }
    case HANZO_MOE_GEMM1:
      rc = run_gemm<G>(args1, groups, s.sm_count, false, ws + l.gemm, st);
      break;
    case HANZO_MOE_ACT: {
      dim3 grid((unsigned)R, s.i / GROUP);
      act_kernel<<<grid, GROUP, 0, st>>>(y1, s.i, a2, s2);
      rc = status(cudaGetLastError());
      break;
    }
    case HANZO_MOE_GEMM2:
      rc = run_gemm<G>(args2, groups, s.sm_count, true, ws + l.gemm, st);
      break;
    case HANZO_MOE_COMBINE:
      rc = hanzo_moe_combine(y2, dst, nullptr, L.addend, L.out, s.m, s.k, s.h, HANZO_MOE_SUM,
                             st);
      break;
    default:
      rc = HANZO_MOE_BAD_ARGUMENT;
    }
  }
  return rc;
}

} // namespace moe_fp8

using namespace moe_fp8;

extern "C" int hanzo_moe_fp8_prepare(int device) {
  auto e = cudaSetDevice(device);
  if (e != cudaSuccess)
    return status(e);
  return prepare_tile<TileP>();
}

extern "C" int hanzo_moe_fp8_layout(const hanzo_moe_shape *shape, hanzo_moe_layout *layout) {
  if (!shape || !layout || !valid(*shape))
    return HANZO_MOE_BAD_ARGUMENT;
  layout_for<TileP>(*shape, layout);
  return 0;
}

extern "C" int hanzo_moe_fp8_run(const hanzo_moe_fp8_launch *launch, int first, int last) {
  if (!launch || !valid(launch->shape) || first < 0 || last > HANZO_MOE_COMBINE || first > last)
    return HANZO_MOE_BAD_ARGUMENT;
  if (reinterpret_cast<uintptr_t>(launch->workspace) % ALIGN != 0)
    return HANZO_MOE_UNALIGNED;
  return moe_fp8::run(*launch, first, last);
}
