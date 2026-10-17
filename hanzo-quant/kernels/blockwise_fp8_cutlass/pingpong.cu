// The pingpong 64x128x128 tile (64 < M <= 256): one tile per translation unit.
#include "gemm.cuh"

HANZO_BLOCKWISE_FP8_TILE(pingpong, hanzo_blockwise_fp8::PingpongGemm)
