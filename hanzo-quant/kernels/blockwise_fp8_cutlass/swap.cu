// The A/B-swapped cooperative 128x32x128 tile (M <= 64): one tile per translation unit.
#include "gemm.cuh"

HANZO_BLOCKWISE_FP8_TILE(swap_ab, hanzo_blockwise_fp8::SwapAbGemm)
