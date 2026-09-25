/**
 * Block-FP8 GEMM on sm_12x tensor cores: the kernels vLLM v0.29.0 serves.
 *
 * Port of vLLM's cutlass_3x_gemm_fp8_blockwise
 * (csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/
 *  scaled_mm_blockwise_sm120_fp8_dispatch.cuh @ 98dff2a81) without torch.
 * Same element types, tiles, schedules, epilogue and stage carveout, so the
 * mainloop is CUTLASS's sm120 blockwise collective in both engines: per 128-K
 * block, mma.sync f8f6f4 into a zeroed FP32 temp, then one FFMA folds it into
 * the accumulator with RN(s_a * s_w).
 *
 * One deliberate difference: both scale operands are K-major. Activation
 * scales are row-major [M, K/128] (what hanzo's quantizer writes); weight
 * scales are [N/128, K/128] as on disk. vLLM reads activation scales M-major.
 * The major changes only where a scale is fetched from, never the arithmetic.
 */
#pragma once

#include "blockwise_fp8_cutlass.h"

#include <cstdint>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

static_assert(CUDART_VERSION >= 13000);

namespace hanzo_blockwise_fp8 {
using namespace cute;

constexpr int kBlock = 128;
constexpr uintptr_t kOperandAlignment = 16;
constexpr uintptr_t kScaleAlignment = sizeof(float);
constexpr size_t kDefaultSharedBytes = 48 * 1024;

// The sm_12x family guard vLLM wraps its kernel in (cutlass_extensions/common.hpp).
template <typename Kernel> struct enable_sm120_family : Kernel {
  template <typename... Args> CUTLASS_DEVICE void operator()(Args &&...args) {
#if defined __CUDA_ARCH__
#if (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
    Kernel::operator()(std::forward<Args>(args)...);
#else
    printf("This kernel only supports sm120f.\n");
    asm("trap;");
#endif
#endif
  }
};

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK,
          class MmaTileShape, class EpilogueScheduler, class MainloopScheduler,
          bool SwapAb>
struct Gemm {
  static constexpr bool kSwapAb = SwapAb;
  using ClusterShape = Shape<_1, _1, _1>;
  using ElementAB = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementAB>::value;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementAB>::value;

  using ElementD = cutlass::bfloat16_t;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ElementC = void;
  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float;

  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
      ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
      UMMA::Major::K, UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC,
          conditional_t<SwapAb, LayoutC_Transpose, LayoutC>, AlignmentC,
          ElementD, conditional_t<SwapAb, LayoutD_Transpose, LayoutD>,
          AlignmentD, EpilogueScheduler, DefaultOperation>::CollectiveOp;

  using Carveout = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
  using CollectiveMainloop = conditional_t<
      SwapAb,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementAB,
          cute::tuple<LayoutB_Transpose, LayoutSFA>, AlignmentB, ElementAB,
          cute::tuple<LayoutA_Transpose, LayoutSFB>, AlignmentA,
          ElementAccumulator, MmaTileShape, ClusterShape, Carveout,
          MainloopScheduler>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementAB, cute::tuple<LayoutA, LayoutSFA>,
          AlignmentA, ElementAB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
          ElementAccumulator, MmaTileShape, ClusterShape, Carveout,
          MainloopScheduler>::CollectiveOp>;

  using KernelType = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;
  struct Kernel : public KernelType {};
};

// vLLM's sm120_blockwise_fp8_config_swapab: M <= 64.
using SwapAbGemm =
    Gemm<128, 1, 128, Shape<_128, _32, _128>,
         cutlass::epilogue::collective::EpilogueScheduleAuto,
         cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120, true>;
// vLLM's sm120_blockwise_fp8_config_pingpong: 64 < M <= 256.
using PingpongGemm =
    Gemm<1, 128, 128, Shape<_64, _128, _128>,
         cutlass::epilogue::collective::EpilogueScheduleAuto,
         cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120, false>;
// vLLM's sm120_blockwise_fp8_config_default: M > 256.
using CooperativeGemm =
    Gemm<1, 128, 128, Shape<_128, _128, _128>,
         cutlass::epilogue::collective::EpilogueScheduleAuto,
         cutlass::gemm::collective::KernelScheduleAuto, false>;

template <class G>
typename G::Kernel::Arguments make_arguments(const hanzo_blockwise_fp8_launch &launch) {
  using Kernel = typename G::Kernel;
  using ElementAB = typename G::ElementAB;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  const int m = launch.shape.m, n = launch.shape.n, k = launch.shape.k;

  auto a_stride = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  auto b_stride = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  auto c_stride = cutlass::make_cute_packed_stride(
      StrideC{}, G::kSwapAb ? make_shape(n, m, 1) : make_shape(m, n, 1));
  auto problem = G::kSwapAb ? make_shape(n, m, k, 1) : make_shape(m, n, k, 1);

  auto a = static_cast<const ElementAB *>(launch.a);
  auto w = static_cast<const ElementAB *>(launch.w);
  typename Kernel::MainloopArguments mainloop{};
  mainloop.layout_SFA = G::ScaleConfig::tile_atom_to_shape_SFA(problem);
  mainloop.layout_SFB = G::ScaleConfig::tile_atom_to_shape_SFB(problem);
  if constexpr (G::kSwapAb) {
    // Weights take the M role: the tile's 128 rows are one weight-scale row.
    mainloop.ptr_A = w;
    mainloop.dA = b_stride;
    mainloop.ptr_B = a;
    mainloop.dB = a_stride;
    mainloop.ptr_SFA = launch.w_scale;
    mainloop.ptr_SFB = launch.a_scale;
  } else {
    mainloop.ptr_A = a;
    mainloop.dA = a_stride;
    mainloop.ptr_B = w;
    mainloop.dB = b_stride;
    mainloop.ptr_SFA = launch.a_scale;
    mainloop.ptr_SFB = launch.w_scale;
  }
  auto out = static_cast<typename G::ElementD *>(launch.output);
  typename Kernel::EpilogueArguments epilogue{{}, out, c_stride, out, c_stride};
  cutlass::KernelHardwareInfo hw_info{launch.context.device,
                                      launch.context.sm_count};
  // Default scheduler arguments, as vLLM's cutlass_gemm_caller passes.
  return {cutlass::gemm::GemmUniversalMode::kGemm, problem, mainloop, epilogue,
          hw_info, typename Kernel::TileSchedulerArguments{}};
}

template <class G>
int prepare(const hanzo_blockwise_fp8_context &context, const cudaDeviceProp &props,
            hanzo_blockwise_fp8_resources *resources) {
  using Kernel = typename G::Kernel;
  if constexpr (Kernel::SharedStorageSize >= kDefaultSharedBytes) {
    auto error = cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      Kernel::SharedStorageSize);
    if (error != cudaSuccess)
      return -static_cast<int>(error);
  }
  cudaFuncAttributes attributes{};
  auto error = cudaFuncGetAttributes(&attributes, cutlass::device_kernel<Kernel>);
  if (error != cudaSuccess)
    return -static_cast<int>(error);
  *resources = {context,
                props.major,
                props.minor,
                static_cast<int32_t>(Kernel::MaxThreadsPerBlock),
                attributes.numRegs,
                static_cast<size_t>(Kernel::SharedStorageSize),
                attributes.localSizeBytes};
  return 0;
}

template <class G>
int workspace_size(const hanzo_blockwise_fp8_context &context,
                   const hanzo_blockwise_fp8_shape &shape, size_t *bytes) {
  using Adapter = cutlass::gemm::device::GemmUniversalAdapter<typename G::Kernel>;
  hanzo_blockwise_fp8_launch launch{};
  launch.shape = shape;
  launch.context = context;
  auto arguments = make_arguments<G>(launch);
  auto status = Adapter::can_implement(arguments);
  if (status != cutlass::Status::kSuccess)
    return static_cast<int>(status);
  *bytes = Adapter::get_workspace_size(arguments);
  return 0;
}

template <class G> int gemm(const hanzo_blockwise_fp8_launch &launch) {
  using Kernel = typename G::Kernel;
  using Adapter = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  auto arguments = make_arguments<G>(launch);
  auto status = Adapter::can_implement(arguments);
  if (status != cutlass::Status::kSuccess)
    return static_cast<int>(status);
  size_t bytes = Adapter::get_workspace_size(arguments);
  if (bytes > launch.workspace_bytes || (bytes && !launch.workspace))
    return static_cast<int>(cutlass::Status::kErrorWorkspaceNull);
  if (bytes && reinterpret_cast<uintptr_t>(launch.workspace) % kOperandAlignment)
    return -static_cast<int>(cudaErrorInvalidValue);
  auto stream = static_cast<cudaStream_t>(launch.stream);
  status = Kernel::initialize_workspace(arguments, launch.workspace, stream, nullptr);
  if (status != cutlass::Status::kSuccess)
    return static_cast<int>(status);
  auto params = Kernel::to_underlying_arguments(arguments, launch.workspace);
  return static_cast<int>(Adapter::run(params, stream, nullptr, false));
}

// One tile per translation unit: each .cu instantiates exactly one of these.
int prepare_swap_ab(const hanzo_blockwise_fp8_context &, const cudaDeviceProp &,
                    hanzo_blockwise_fp8_resources *);
int workspace_swap_ab(const hanzo_blockwise_fp8_context &,
                      const hanzo_blockwise_fp8_shape &, size_t *);
int gemm_swap_ab(const hanzo_blockwise_fp8_launch &);
int prepare_pingpong(const hanzo_blockwise_fp8_context &, const cudaDeviceProp &,
                     hanzo_blockwise_fp8_resources *);
int workspace_pingpong(const hanzo_blockwise_fp8_context &,
                       const hanzo_blockwise_fp8_shape &, size_t *);
int gemm_pingpong(const hanzo_blockwise_fp8_launch &);
int prepare_cooperative(const hanzo_blockwise_fp8_context &, const cudaDeviceProp &,
                        hanzo_blockwise_fp8_resources *);
int workspace_cooperative(const hanzo_blockwise_fp8_context &,
                          const hanzo_blockwise_fp8_shape &, size_t *);
int gemm_cooperative(const hanzo_blockwise_fp8_launch &);
} // namespace hanzo_blockwise_fp8

#define HANZO_BLOCKWISE_FP8_TILE(name, G)                                      \
  namespace hanzo_blockwise_fp8 {                                              \
  int prepare_##name(const hanzo_blockwise_fp8_context &context,               \
                     const cudaDeviceProp &props,                              \
                     hanzo_blockwise_fp8_resources *resources) {               \
    return prepare<G>(context, props, resources);                              \
  }                                                                            \
  int workspace_##name(const hanzo_blockwise_fp8_context &context,             \
                       const hanzo_blockwise_fp8_shape &shape, size_t *bytes) { \
    return workspace_size<G>(context, shape, bytes);                           \
  }                                                                            \
  int gemm_##name(const hanzo_blockwise_fp8_launch &launch) {                  \
    return gemm<G>(launch);                                                    \
  }                                                                            \
  }
