# Hanzo Engine — multi-backend + distributed perf roadmap

Status legend: ✅ done/committed · 🟡 partial/HW-bound · 🔴 not started · ⏳ program (weeks+)

## Done this cycle (on `main`, both repos)
- ✅ **Qwen3.6-35B-A3B GGUF (`qwen35moe`, hybrid GDN+attn MoE)** loads + generates coherently on CUDA/Blackwell (sm_121). Fixes: tiled V-head q/k regroup (16k/32v), array-tolerant `head_count_kv`, `ssm_dt` tensor name, hybrid-cache scatter dtype, f32 GDN forward, single-device pin.
- ✅ **Fused CUDA GDN recurrence** wired into the quantized loader (`chunked_gated_delta_rule_recurrence_cuda`): 1.1 → 1.4 T/s decode.
- ✅ **flash-attn builds on sm_121** (authored `hanzo-flash-attn-build` + cudarc 0.17 fix). ~0 gain on 3B-active MoE (attention isn't the bottleneck); will help dense/long-context.
- ✅ **CPU AVX-512/VNNI** vec_dot kernels (cfg-gated) + adaptive decode threading.
- ✅ **Vulkan stack** (subgroup Q8 shader, staging allocator, Q4_K-in-VRAM, push_descriptor, hazard barriers, native copy2d).
- ✅ **CI matrix** (`.github/workflows/build-test-matrix.yml`) — cuda/vulkan/metal/cpu over self-hosted auto-labels.

## P0 — Native AMD/Vulkan on unified memory (Strix Halo / 128 GB APUs on Windows)
CORRECTION (2026-06-02): native Windows Vulkan on the 8060S (gfx1151) is FAST, NOT 0.7 T/s. `vkbench` real-driver: coopmat fp16 GEMM up to **7701 GFLOPS** (1024³), 6883 @ 4096³; **Q8 decode matvec 146 GB/s**; NT-matmul = 30x forward; engine has native Q8 `VulkanQuant` (weights kept quantized in VRAM). Reference: llama.cpp ~242 gen t/s on 0.6B-Q8. The earlier "0.7 T/s" was a BAD bench (Q4_K dequant-to-f32 + per-op-flush harness), not the engine.
Real remaining work (tractable, NOT HW-bound):
1. **End-to-end model decode is dispatch-cost-bound** (hundreds of dispatches x N layers, re-recorded per token), not kernel/BW-bound. Frontier = push-descriptors default-on + op fusion + ensuring every layer hits the native Q8 matvec (not a dequant fallback). The kernels are already fast.
2. **Native Q4_K/Q5_K/Q6_K matvec** (only Q8_0 has the keep-quantized-in-VRAM path today). Q4_K currently dequants to f32 → a 30B Q4_K = ~74 GB > the 60 GB Vulkan heap = OOM. Either use **Q8_0** (35 GB, fits, native path) or add native Q4_K matvec. (Memory note: the 8060S exposes ONE ~60 GB DEVICE_LOCAL heap where host-visible == device-local, so staging is N/A; a bigger BIOS UMA carveout raises the ceiling.)
3. Target: end-to-end decode near the 146 GB/s / llama.cpp parity the kernels already prove. ⏳ days, not weeks.

## P1 — Distributed serving via hanzo-node (3 boxes → ~3× aggregate)
Goal: dbc(M4 Max) + evo(8060S) + spark(GB10), each loads a full Qwen3.6 replica in its 128 GB unified mem; a router interleaves requests → ~3× single-box throughput (data-parallel, NOT tensor-parallel — the 2.5GbE/LAN link is too slow for TP).
- Prereq: per-box single-replica throughput must be worth multiplying (today: spark 1.4 / evo ~CPU / dbc TBD). **Fix per-box perf first** (P0 + CUDA decode).
- Build: `hanzo serve` on each box (bound to its fast backend) + a round-robin/least-loaded router (hanzo-node) over the cluster. Health-checked, model-pinned.
- Note: 3× only holds if the 3 replicas are throughput-comparable; they are NOT today (asymmetric HW). Realistic near-term: spark dominates; true 3× needs P0 making evo competitive. ⏳.

## P2 — Backend completeness (cuda/vulkan/rocm/hip/metal/wgsl)
- ✅ CUDA (sm_121), ✅ **Vulkan (FAST — 7701 GFLOPS coopmat, 146 GB/s Q8; engine `rocm`+`vulkan` both compile green on main)**, ✅ Metal (feature wiring verified; needs dbc runtime bench), ✅ CPU.
- 🟡 **ROCm/HIP**: VIABLE on Windows for gfx1151 (ROCm 7.1 installed; `build_winhip.bat`; ~192 gen t/s per measured landscape). Engine `rocm` feature compiles green (HIP kernels incl. sort.hip.cpp). Runtime bench on native Windows is the open item. CORRECTION: earlier "not viable on Windows" was wrong.
- 🔴 **WGSL/WebGPU**: not started — new backend (wgpu; `ash` drives the GPU today, wgpu only sees llvmpipe on WSL). Largest lift; native-Windows wgpu is the path.
- Per-arch model matrix (every QWEN3* + the MoE/GDN archs) verified in CI. ⏳ months for rocm/hip/wgsl.

## Infra blockers
- CI matrix jobs **queued, not running** — self-hosted runners online (evo×2, spark) but a **runner-group/permission** issue (the `engine` repo isn't granted the org runner group). Needs org-admin. dbc Actions runner offline.
