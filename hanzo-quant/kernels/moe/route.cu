/**
 * route: a stable counting sort of the R = M*k routed assignments by expert.
 *
 * Flat row f = t*k + j (token t, slot j) goes to sorted position
 *   dst[f] = offsets[e] + #{f' < f : ids[f'] = e},
 * so rows of one expert keep flat order. Placement never uses atomics: a warp ranks its rows
 * with __match_any_sync, warps are ordered by an exclusive scan of per-warp counts, and chunks
 * of 4096 rows by an exclusive scan of per-chunk counts. The result is a pure function of ids.
 *
 * R <= 4096: one 1024-thread CTA does everything. Larger R: rank per chunk, one CTA scans the
 * [chunks x E] counts, then a scatter. Nothing synchronizes with the host.
 */

#include "moe.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int THREADS = 1024;
constexpr int WARPS = THREADS / 32;
constexpr int CHUNK = HANZO_MOE_ROUTE_CHUNK;
constexpr int STEPS = CHUNK / THREADS; // 32-row steps per warp slice

size_t rank_smem(int e) { return (size_t)WARPS * e * sizeof(uint16_t) + (size_t)e * sizeof(int); }

/// Ranks rows [f0, f0 + n) (n <= CHUNK) of ids within their expert. On return (after a
/// __syncthreads) cnt[w*E + e] holds the number of rows of expert e in warp slices before w, and
/// tot[e] the chunk's count of e; each lane's rank[s] is its row's rank in the chunk.
__device__ void rank_chunk(const int32_t *__restrict__ ids, int f0, int n, int E,
                           uint16_t *cnt, int *tot, int (&ex)[STEPS], int (&rank)[STEPS]) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int i = threadIdx.x; i < WARPS * E; i += THREADS)
    cnt[i] = 0;
  __syncthreads();

  const int slice = (n + WARPS - 1) / WARPS;
  const int lo = warp * slice;
  const int hi = min(n, lo + slice);
  uint16_t *mine = cnt + warp * E;
  const unsigned lt = (1u << lane) - 1u;
#pragma unroll
  for (int s = 0; s < STEPS; s++) {
    const int f = lo + s * 32 + lane;
    const bool valid = f < hi;
    const int e = valid ? ids[f0 + f] : -1 - lane; // invalid lanes never match a valid one
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    int r = 0;
    if (valid)
      r = mine[e] + __popc(peers & lt);
    __syncwarp();
    if (valid && (peers & lt) == 0)
      mine[e] += __popc(peers);
    __syncwarp();
    ex[s] = e;
    rank[s] = r;
  }
  __syncthreads();
  for (int e = threadIdx.x; e < E; e += THREADS) {
    int run = 0;
    for (int w = 0; w < WARPS; w++) {
      const int c = cnt[w * E + e];
      cnt[w * E + e] = (uint16_t)run;
      run += c;
    }
    tot[e] = run;
  }
  __syncthreads();
  // Fold the warp prefix into each rank.
#pragma unroll
  for (int s = 0; s < STEPS; s++)
    if (ex[s] >= 0)
      rank[s] += cnt[warp * E + ex[s]];
}

/// Exclusive scan of v over the CTA (one value per thread). Returns the thread's prefix and
/// writes the total to *total.
__device__ int block_exclusive(int v, int *scratch, int *total) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o)
      x += y;
  }
  if (lane == 31)
    scratch[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = scratch[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o)
        w += y;
    }
    scratch[lane] = w; // inclusive over warps
  }
  __syncthreads();
  const int before = warp ? scratch[warp - 1] : 0;
  *total = scratch[WARPS - 1];
  __syncthreads();
  return before + x - v;
}

/// From per-expert totals tot[E]: offsets, group, active (G entries), nactive.
__device__ void finish(const int *tot, int E, int G, int32_t *offsets, int32_t *group,
                       int32_t *active, int32_t *nactive, int *scan, int *base) {
  const int e = threadIdx.x;
  const int n = e < E ? tot[e] : 0;
  int rows, count;
  const int off = block_exclusive(n, scan, &rows);
  const int g = block_exclusive(n > 0 ? 1 : 0, scan, &count);
  if (e < E) {
    offsets[e] = off;
    base[e] = off;
    group[e] = n > 0 ? g : -1;
    if (n > 0)
      active[g] = e;
  }
  if (e == 0) {
    offsets[E] = rows;
    *nactive = count;
  }
  for (int i = count + e; i < G; i += THREADS)
    active[i] = -1;
  __syncthreads();
}

__global__ void __launch_bounds__(THREADS)
    route_one(const int32_t *__restrict__ ids, int R, int E, int G, int32_t *offsets,
              int32_t *src, int32_t *dst, int32_t *group, int32_t *active, int32_t *nactive) {
  extern __shared__ __align__(16) unsigned char smem[];
  int *tot = reinterpret_cast<int *>(smem);
  uint16_t *cnt = reinterpret_cast<uint16_t *>(tot + E);
  __shared__ int scan[WARPS];
  int ex[STEPS], rank[STEPS];
  rank_chunk(ids, 0, R, E, cnt, tot, ex, rank);
  // tot becomes the expert base after finish.
  finish(tot, E, G, offsets, group, active, nactive, scan, tot);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int slice = (R + WARPS - 1) / WARPS;
#pragma unroll
  for (int s = 0; s < STEPS; s++) {
    if (ex[s] < 0)
      continue;
    const int f = warp * slice + s * 32 + lane;
    const int pos = tot[ex[s]] + rank[s];
    dst[f] = pos;
    src[pos] = f;
  }
}

/// Multi-chunk pass 1: per-chunk counts [chunks x E] and each row's rank in its chunk (in dst).
__global__ void __launch_bounds__(THREADS)
    route_rank(const int32_t *__restrict__ ids, int R, int E, int32_t *counts, int32_t *dst) {
  extern __shared__ __align__(16) unsigned char smem[];
  int *tot = reinterpret_cast<int *>(smem);
  uint16_t *cnt = reinterpret_cast<uint16_t *>(tot + E);
  const int f0 = blockIdx.x * CHUNK;
  const int n = min(CHUNK, R - f0);
  int ex[STEPS], rank[STEPS];
  rank_chunk(ids, f0, n, E, cnt, tot, ex, rank);
  for (int e = threadIdx.x; e < E; e += THREADS)
    counts[blockIdx.x * E + e] = tot[e];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int slice = (n + WARPS - 1) / WARPS;
#pragma unroll
  for (int s = 0; s < STEPS; s++)
    if (ex[s] >= 0)
      dst[f0 + warp * slice + s * 32 + lane] = rank[s];
}

/// Multi-chunk pass 2: one CTA turns counts into absolute chunk bases.
__global__ void __launch_bounds__(THREADS)
    route_scan(int chunks, int E, int G, int32_t *counts, int32_t *offsets, int32_t *group,
               int32_t *active, int32_t *nactive) {
  __shared__ int tot[HANZO_MOE_MAX_EXPERTS];
  __shared__ int base[HANZO_MOE_MAX_EXPERTS];
  __shared__ int scan[WARPS];
  const int e = threadIdx.x;
  if (e < E) {
    int run = 0;
    for (int c = 0; c < chunks; c++) {
      const int v = counts[c * E + e];
      counts[c * E + e] = run;
      run += v;
    }
    tot[e] = run;
  }
  __syncthreads();
  finish(tot, E, G, offsets, group, active, nactive, scan, base);
  if (e < E)
    for (int c = 0; c < chunks; c++)
      counts[c * E + e] += base[e];
}

/// Multi-chunk pass 3: dst[f] = chunk base + rank; src[dst[f]] = f.
__global__ void route_place(const int32_t *__restrict__ ids, int R, int E,
                            const int32_t *__restrict__ counts, int32_t *dst, int32_t *src) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= R)
    return;
  const int pos = counts[(f / CHUNK) * E + ids[f]] + dst[f];
  dst[f] = pos;
  src[pos] = f;
}

int status(cudaError_t e) { return e == cudaSuccess ? 0 : -(int)e; }

} // namespace

extern "C" size_t hanzo_moe_route_scratch(int r, int e) {
  if (r <= CHUNK)
    return 0;
  const size_t chunks = (r + CHUNK - 1) / CHUNK;
  return chunks * e * sizeof(int32_t);
}

extern "C" const char *hanzo_moe_error_string(int s) {
  if (s == 0)
    return "success";
  if (s == HANZO_MOE_BAD_ARGUMENT)
    return "bad argument";
  if (s == HANZO_MOE_UNALIGNED)
    return "pointer or size not aligned for the kernel";
  if (s < 0)
    return cudaGetErrorString((cudaError_t)(-s));
  return "CUTLASS status (see cutlass::Status)";
}

extern "C" int hanzo_moe_prepare(int device) {
  cudaError_t e = cudaSetDevice(device);
  const int bytes = (int)rank_smem(HANZO_MOE_MAX_EXPERTS);
  if (e == cudaSuccess)
    e = cudaFuncSetAttribute(route_one, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (e == cudaSuccess)
    e = cudaFuncSetAttribute(route_rank, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  return status(e);
}

extern "C" int hanzo_moe_route(const int32_t *ids, int m, int k, int e, int32_t *offsets,
                               int32_t *src, int32_t *dst, int32_t *group, int32_t *active,
                               int32_t *nactive, void *scratch, int *launches,
                               cudaStream_t stream) {
  if (m <= 0 || k <= 0 || e <= 0 || e > HANZO_MOE_MAX_EXPERTS)
    return HANZO_MOE_BAD_ARGUMENT;
  const long r64 = (long)m * k;
  if (r64 > (1L << 30))
    return HANZO_MOE_BAD_ARGUMENT;
  const int R = (int)r64;
  const int G = R < e ? R : e;
  const size_t smem = rank_smem(e);
  if (R <= CHUNK) {
    route_one<<<1, THREADS, smem, stream>>>(ids, R, e, G, offsets, src, dst, group, active,
                                            nactive);
    if (launches)
      *launches = 1;
    return status(cudaGetLastError());
  }
  if (!scratch)
    return HANZO_MOE_BAD_ARGUMENT;
  const int chunks = (R + CHUNK - 1) / CHUNK;
  int32_t *counts = static_cast<int32_t *>(scratch);
  route_rank<<<chunks, THREADS, smem, stream>>>(ids, R, e, counts, dst);
  route_scan<<<1, THREADS, 0, stream>>>(chunks, e, G, counts, offsets, group, active, nactive);
  route_place<<<(R + 255) / 256, 256, 0, stream>>>(ids, R, e, counts, dst, src);
  if (launches)
    *launches = 3;
  return status(cudaGetLastError());
}
