# Perf-regression guards

`run_perf_guards.sh` runs three real workloads on the correct hardware and fails (non-zero exit,
loud `FAIL measured vs floor`) if any of the hard-won speedups silently regresses. Floors are set
with margin below the known-good number and well above the regressed number, so normal run-to-run
variance never trips them but a real regression does.

## The guards

| guard          | box         | workload                                  | metric            | floor | known-good | regressed |
|----------------|-------------|-------------------------------------------|-------------------|-------|------------|-----------|
| `cuda-prefill` | spark/GB10  | `hanzo bench` Qwen3-8B-Q8_0 pp512         | prefill t/s       | 1000  | ~1485      | ~120      |
| `vulkan-mmq`   | evo/8060S   | `hanzo.exe bench` Qwen3-8B-Q8_0 pp512     | route + prefill t/s | 150 | ~210       | ~113 (dp4a)|
| `musetalk-fps` | spark/GB10  | `musetalk-bench bench` (f16, framework)   | single-frame fps  | 3.5   | ~4.5-5.0   | 2.51      |

## What each guard protects (PROVENANCE)

1. **cuda-prefill** -- the 12.4x CUDA prefill win.
   - Fix: `hanzo-engine/src/attention/backends/naive.rs` `maybe_synchronize` skips the per-attention-
     layer `MemoryUsage::query` / `System::new_all()` sysinfo scan on unified-memory CUDA (GB10). That
     scan is a ~110ms host stall called once per layer; on a unified-memory GPU it left the GPU idle
     ~94% of prefill and the low-VRAM guard it implements is meaningless (no separate VRAM pool).
   - Commit: `5616fbdae perf(cuda): skip per-layer sysinfo scan on unified-memory GPUs (12.4x prefill)`.
   - Regression signature: prefill collapses to ~120 t/s (the per-layer-stall-bound number).

2. **vulkan-mmq** -- the 1.95x Vulkan dense-Q8 prefill win.
   - Fix: `hanzo-ml/src/vulkan/shaders/mul_mm_q8_mmq.comp` (+ `_body.glsl`), a llama.cpp RDNA3
     warp-tiled int8-dp4a Q8 GEMM, is the DEFAULT Q8 prefill route on int8-dot devices
     (`hanzo-ml/src/quantized/mod.rs` -> `VulkanBackend::matmul_q8_mmq_gpu`).
   - Commit: `c9d241d2 perf(vulkan): mul_mm_q8_mmq -- llama RDNA3 warp-tiled int8-dp4a Q8 prefill GEMM (1.95x)`.
   - The guard asserts BOTH: (a) `mul_mm_q8_mmq` shows up in the `VK_PROFILE=1` per-op census (the
     route actually fired -- so a future routing change that silently falls back to dp4a/coopmat is
     caught even if t/s happens to stay above the floor), and (b) pp512 t/s > 150.
   - Regression signature: route falls back to dp4a (~113 t/s) or the kernel itself regresses.

3. **musetalk-fps** -- the MuseTalk GB10 framework keystones (2.51 -> ~4.5 fps, GPU util 15-22% -> ~93%).
   - Fixes (all in `hanzo-ml`): pinned async-mempool release threshold
     (`e89067b5`, 2.51->3.09), f16-native fused GroupNorm kernel (`d32145b3`, 3.09->4.25), and the
     conv bias-add fused into the im2col NHWC->NCHW epilogue (`fab7ecf7`).
   - The guard measures the **framework-only** single-frame path (full VAE-encode + framework UNet +
     full VAE-decode) at f16 -- the path those keystones accelerate. (The harness also prints a
     faster COMBINED path with cached-ref encode + TAESD decode; the guard does not gate on it.)
   - Regression signature: fps falls back toward the 2.51 pre-keystone baseline.

## Running

```bash
# everything reachable (spark guards over ssh, evo guard via local cmd.exe):
scripts/perf-guard/run_perf_guards.sh

# one box / one guard:
scripts/perf-guard/run_perf_guards.sh --box spark
scripts/perf-guard/run_perf_guards.sh --guard cuda-prefill
```

Exit code is non-zero iff a guard FAILED. Unreachable boxes / missing binaries are reported as SKIP
(not failures) -- see the box-reachability notes below.

## Box reachability + build prerequisites

- **spark** (GB10, CUDA): reached over `ssh spark`. Needs a CUDA `hanzo` binary built from a tree
  with fix #1 at `$SPARK_ENGINE_DIR/target/release/hanzo`, and a CUDA `musetalk-bench` at
  `$SPARK_MUSETALK_DIR/target/release/musetalk-bench`. Sync + build with the repo's `sync-spark.sh`
  then `cargo build --release -p hanzo-cli --features cuda` (engine) /
  `cargo build --release --features cuda` (musetalk-bench), with
  `CUDA_COMPUTE_CAP=121` + the sbsa nvcc on PATH.
- **evo** (8060S, Vulkan): this WSL box. The native Windows Vulkan `hanzo.exe` is the fast path; WSL
  Vulkan is not. The guard invokes `$EVO_HANZO_EXE` through `cmd.exe`. Build it natively with
  `C:\Users\z\work\hanzo-native\build-engine-vulkan-quick.bat` (or `...-vulkan.bat` for release);
  the `mul_mm_q8_mmq` win requires a build AFTER the kernel's committed SPIR-V
  (`src/vulkan/spv/mul_mm_q8_mmq.spv`) -- the build embeds the committed `.spv` since the Windows
  box has no `glslc`. If the exe is absent the guard SKIPs.

All paths/knobs are overridable by env var (see the top of `run_perf_guards.sh`):
`SPARK_HOST`, `SPARK_ENGINE_DIR`, `SPARK_MUSETALK_DIR`, `SPARK_GGUF_DIR/FILE`,
`EVO_HANZO_EXE`, `EVO_GGUF_DIR/FILE`, `PROMPT_LEN`, `ITERS`, `WARMUP`, `MUSETALK_ITERS`.
