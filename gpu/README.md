# hanzo-engine (GPU) — native Windows GPU inference, Vulkan + ROCm

Native, **bridge-free** GPU inference for the hanzo stack on Windows, built on the
llama.cpp/ggml runtime. Two selectable backends wired in:

| Backend | Use | Measured on AMD Radeon 8060S (Strix Halo, gfx1151), Qwen3-0.6B |
|---|---|---|
| **vulkan** (default) | any GPU (AMD/NVIDIA/Intel), Win+Linux | **~350 tok/s decode**, ~11k tok/s prefill (real AMD driver, `KHR_coopmat`) |
| **rocm** (opt-in) | AMD, long-context fast-path | ties Vulkan at std ctx, ~3× at 130K ctx (tuned rocWMMA) |

For comparison, the WSL2 mistral.rs/ROCm path capped at **~1.4 tok/s** — the
WSL→D3D12 bridge synchronizes every GPU call. Native = **~250× faster**. This is
why the engine is built on ggml's mature native backends, not a hand-ported one.

Serves the **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/embeddings`) on
`127.0.0.1:<port>` — a drop-in for the hanzo node (point a provider's `external_url`
here). Runs **GGUF** models (the format we already produce for the Zen models).

## Build & run (from the hanzo tree)

```powershell
cd hanzo\engine\gpu

# 1. Assemble backends from upstream optimized binaries (+ a demo model)
.\setup.ps1 -Backend all -PullModel        # vulkan (always) + rocm (AMD) into .\dist\

# 2. Run — Vulkan (default, cross-vendor)
.\hanzo-engine.ps1 -Backend vulkan -Port 36920
#    or ROCm (AMD opt-in, long-context)
.\hanzo-engine.ps1 -Backend rocm   -Port 36920
#    add -Embedding to expose /v1/embeddings (pooling)
```

Then point the hanzo node at it:
```
POST /v2/add_llm_provider { external_url:"http://127.0.0.1:36920", model:"openai:hanzo", api_key:"x" }
```

## Build from source (optional — needs SDKs)

`setup.ps1` fetches the same optimized binaries AMD/ggml publish, so from-source
gives no perf gain. If you want to compile anyway:

```powershell
# Vulkan: winget install Kitware.CMake KhronosGroup.VulkanSDK Ninja-build.Ninja
.\build-from-source.ps1 -Backend vulkan
# ROCm: install HIP SDK for Windows 7.1.1+ (gfx1151 officially Supported)
.\build-from-source.ps1 -Backend rocm
```

## Notes
- `dist\` (binaries) is gitignored; commit only the scripts.
- The WSL mistral.rs ROCm backend (`../`, `hanzo-ml` `feature/rocm-backend`) stays as
  the **native-Linux ROCm** / research track — correct but bridge-capped on Windows.
- Model gotcha: the Ollama `zen-nano:0.6b` blob is a degenerate quant (loops, no EOS).
  Use a properly-quantized GGUF (canonical `ggml-org/Qwen3-0.6B-GGUF`, or re-quantize Zen).
