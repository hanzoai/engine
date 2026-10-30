/**
 * C ABI for the three sm_12x block-FP8 tiles (gemm.cuh): validation and
 * dispatch by the tile the caller prepared. The caller picks the tile by M,
 * exactly as vLLM's cutlass_gemm_blockwise_sm120_fp8_dispatch does.
 */
#include "gemm.cuh"

using namespace hanzo_blockwise_fp8;

namespace {
bool valid_tile(int32_t tile) {
  return tile == HANZO_BLOCKWISE_FP8_SWAP_AB || tile == HANZO_BLOCKWISE_FP8_PINGPONG ||
         tile == HANZO_BLOCKWISE_FP8_COOPERATIVE;
}

bool valid_context(const hanzo_blockwise_fp8_context *context) {
  return context && context->device >= 0 && context->sm_count > 0 &&
         valid_tile(context->tile);
}

bool valid_shape(const hanzo_blockwise_fp8_shape *shape) {
  return shape && shape->m > 0 && shape->n > 0 && shape->k > 0 &&
         shape->n % kBlock == 0 && shape->k % kBlock == 0;
}

bool aligned(const void *pointer, uintptr_t alignment) {
  return pointer && reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

constexpr int invalid() { return -static_cast<int>(cudaErrorInvalidValue); }
} // namespace

extern "C" const char *hanzo_blockwise_fp8_error_string(int status) {
  return status < 0 ? cudaGetErrorString(static_cast<cudaError_t>(-status))
                    : cutlass::cutlassGetStatusString(static_cast<cutlass::Status>(status));
}

extern "C" int hanzo_blockwise_fp8_prepare(int32_t device, int32_t tile,
                                           hanzo_blockwise_fp8_resources *resources) {
  if (!resources || device < 0 || !valid_tile(tile))
    return invalid();
  int current = 0;
  auto error = cudaGetDevice(&current);
  if (error != cudaSuccess)
    return -static_cast<int>(error);
  if (current != device)
    return -static_cast<int>(cudaErrorInvalidDevice);
  cudaDeviceProp props{};
  error = cudaGetDeviceProperties(&props, device);
  if (error != cudaSuccess)
    return -static_cast<int>(error);
  if (props.major != 12 || props.minor != 1)
    return -static_cast<int>(cudaErrorNotSupported);
  hanzo_blockwise_fp8_context context{device, props.multiProcessorCount, tile};
  switch (tile) {
  case HANZO_BLOCKWISE_FP8_SWAP_AB:
    return prepare_swap_ab(context, props, resources);
  case HANZO_BLOCKWISE_FP8_PINGPONG:
    return prepare_pingpong(context, props, resources);
  default:
    return prepare_cooperative(context, props, resources);
  }
}

extern "C" int hanzo_blockwise_fp8_workspace_size(const hanzo_blockwise_fp8_context *context,
                                                  const hanzo_blockwise_fp8_shape *shape,
                                                  size_t *bytes) {
  if (!valid_context(context) || !valid_shape(shape) || !bytes)
    return invalid();
  switch (context->tile) {
  case HANZO_BLOCKWISE_FP8_SWAP_AB:
    return workspace_swap_ab(*context, *shape, bytes);
  case HANZO_BLOCKWISE_FP8_PINGPONG:
    return workspace_pingpong(*context, *shape, bytes);
  default:
    return workspace_cooperative(*context, *shape, bytes);
  }
}

extern "C" int hanzo_blockwise_fp8_gemm(const hanzo_blockwise_fp8_launch *launch) {
  if (!launch || !valid_context(&launch->context) || !valid_shape(&launch->shape) ||
      !aligned(launch->a, kOperandAlignment) || !aligned(launch->w, kOperandAlignment) ||
      !aligned(launch->output, kOperandAlignment) ||
      !aligned(launch->a_scale, kScaleAlignment) ||
      !aligned(launch->w_scale, kScaleAlignment))
    return invalid();
  switch (launch->context.tile) {
  case HANZO_BLOCKWISE_FP8_SWAP_AB:
    return gemm_swap_ab(*launch);
  case HANZO_BLOCKWISE_FP8_PINGPONG:
    return gemm_pingpong(*launch);
  default:
    return gemm_cooperative(*launch);
  }
}
