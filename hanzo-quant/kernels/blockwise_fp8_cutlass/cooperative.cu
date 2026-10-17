// The cooperative 128x128x128 tile (M > 256): one tile per translation unit.
#include "gemm.cuh"

HANZO_BLOCKWISE_FP8_TILE(cooperative, hanzo_blockwise_fp8::CooperativeGemm)
