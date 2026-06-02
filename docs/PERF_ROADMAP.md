# Hanzo Engine — multi-backend + distributed perf roadmap

Status legend: ✅ done/committed · 🟡 partial/HW-bound · 🔴 not started · ⏳ program (weeks+)

## Done this cycle (on `main`, both repos)
- ✅ **Qwen3.6-35B-A3B GGUF (`qwen35moe`, hybrid GDN+attn MoE)** loads + generates coherently on CUDA/Blackwell (sm_121). Fixes: tiled V-head q/k regroup (16k/32v), array-tolerant `head_count_kv`, `ssm_dt` tensor name, hybrid-cache scatter dtype, f32 GDN forward, single-device pin.
- ✅ **Fused CUDA GDN recurrence** wired into the quantized loader (`chunked_gated_delta_rule_recurrence_cuda`): 1.1 → 1.4 T/s decode.
- ✅ **flash-attn builds on sm_121** (authored `hanzo-flash-attn-build` + cudarc 0.17 fix). ~0 gain on 3B-active MoE (attention isn't the bottleneck); will help dense/long-context.
- ✅ **CPU AVX-512/VNNI** vec_dot kernels (cfg-gated) + adaptive decode threading.
- ✅ **Vulkan stack** (subgroup Q8 shader, staging allocator, Q4_K-in-VRAM, push_descriptor, hazard barriers, native copy2d).
- ✅ **CI matrix** (`.github/workflows/build-test-matrix.yml`) — cuda/vulkan/metal/cpu over self-hosted auto-labels.

## P0 — Native AMD/Vulkan high-perf on unified memory (the product need: Strix Halo / 128 GB APUs on Windows)
Current: **0.7 T/s (0.6B), 30B OOMs** on the Radeon 8060S despite committed fixes. 🟡
1. **Use the 128 GB GTT for weights, not the small VRAM carveout.** The staging allocator compiles but doesn't engage for large weight buffers. Needs: real DEVICE_LOCAL-on-GTT allocation verified against `VkPhysicalDeviceMemoryProperties` heap sizes on the 8060S; likely also a **BIOS UMA Frame Buffer Size** bump. Until weights live in GTT, >carveout models OOM.
2. **Decode throughput.** Subgroup q8 matvec didn't move ~0.7 T/s → profile with `HANZO_VK_PROFILE`; the cost is dispatch/barrier + unfused per-op, not BW. Needs cooperative-matrix matmul on RDNA3.5 + fused dequant+matvec + a **Vulkan GDN kernel** (branch `perf/vulkan-gdn-kernel`).
3. Target: ≥10–20 T/s on a 30B-A3B-class MoE, large models loadable. ⏳ weeks.

## P1 — Distributed serving via hanzo-node (3 boxes → ~3× aggregate)
Goal: dbc(M4 Max) + evo(8060S) + spark(GB10), each loads a full Qwen3.6 replica in its 128 GB unified mem; a router interleaves requests → ~3× single-box throughput (data-parallel, NOT tensor-parallel — the 2.5GbE/LAN link is too slow for TP).
- Prereq: per-box single-replica throughput must be worth multiplying (today: spark 1.4 / evo ~CPU / dbc TBD). **Fix per-box perf first** (P0 + CUDA decode).
- Build: `hanzo serve` on each box (bound to its fast backend) + a round-robin/least-loaded router (hanzo-node) over the cluster. Health-checked, model-pinned.
- Note: 3× only holds if the 3 replicas are throughput-comparable; they are NOT today (asymmetric HW). Realistic near-term: spark dominates; true 3× needs P0 making evo competitive. ⏳.

## P2 — Backend completeness (cuda/vulkan/rocm/hip/metal/wgsl)
- ✅ CUDA (sm_121), 🟡 Vulkan (works, slow), ✅ Metal (feature wiring verified; needs dbc runtime bench), ✅ CPU.
- 🔴 **ROCm/HIP**: not viable on Strix-Halo-Windows today (ROCm Windows iGPU support immature) — Linux ROCm path is the realistic target, or skip in favor of Vulkan.
- 🔴 **WGSL/WebGPU**: not started — new backend (wgpu); largest lift.
- Per-arch model matrix (every QWEN3* + the MoE/GDN archs) verified in CI. ⏳ months for rocm/hip/wgsl.

## Infra blockers
- CI matrix jobs **queued, not running** — self-hosted runners online (evo×2, spark) but a **runner-group/permission** issue (the `engine` repo isn't granted the org runner group). Needs org-admin. dbc Actions runner offline.
