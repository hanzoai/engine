#pragma once

#ifndef USE_ROCM
#define HANZO_LDG(arg) __ldg(arg)
#else
#define HANZO_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
#define HANZO_SHFL_XOR_SYNC(var, lane_mask)                                     \
  __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#else
#define HANZO_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#endif

#ifndef USE_ROCM
#define HANZO_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
#else
#define HANZO_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif

#ifndef USE_ROCM
#define HANZO_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL)        \
  cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)
#else
#define HANZO_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL)        \
  hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#endif
