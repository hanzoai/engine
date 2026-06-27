<a name="top"></a>
<!--
<h1 align="center">
  hanzo
</h1>
-->

<div align="center">
  <img src="https://raw.githubusercontent.com/hanzoai/engine/master/res/banner.png" alt="hanzo" width="100%" style="max-width: 800px;">
</div>

<h3 align="center">
Fast, flexible LLM inference.
</h3>

<p align="center">
  | <a href="https://hanzoai.github.io/engine/"><b>Documentation</b></a> | <a href="https://crates.io/crates/hanzo"><b>Rust SDK</b></a> | <a href="https://hanzoai.github.io/engine/tutorials/03-python-sdk/"><b>Python SDK</b></a> | <a href="https://discord.gg/SZrecqK8qw"><b>Discord</b></a> |
</p>

<p align="center">
  <a href="https://github.com/hanzoai/engine/stargazers">
    <img src="https://img.shields.io/github/stars/hanzoai/engine?style=social&label=Star" alt="GitHub stars">
  </a>
</p>

## Latest

- **Qwen3-Omni**: native end-to-end omni-modal model (understand → think → speak) — text/image/video/audio in, text + 24kHz speech out, through one extensible modality pipeline. Validated against the reference weights.
- **New frontier models**: MiniMax-M2 (sparse-MoE) and DeepSeek-V3.2, alongside the existing DeepSeek-V3, Kimi-K2, GLM-4, and Qwen3 families. [Supported models](https://hanzoai.github.io/engine/reference/supported-models/)
- **Paged-attention serving** for the omni Thinker, plus a **disk-first KV cache** (cross-restart sessions + agent prefix reuse) for cheap long-context serving.
- **Anthropic Messages API**: `hanzo serve` now exposes an Anthropic-compatible `POST /v1/messages` endpoint (streaming, tool use, and Claude Code harness support) alongside the OpenAI-compatible `/v1` API. [Examples](examples/server/)
- **Agentic runtime**: web search, local Python code execution with model feedback, session management, and custom tool hooks. [Guide](https://hanzoai.github.io/engine/tutorials/05-build-an-agent/)
- **Gemma 4**: full multimodal: text, image, video, and audio input. [Guide](https://hanzoai.github.io/engine/reference/supported-models/) | [Video setup](https://hanzoai.github.io/engine/guides/models/video-setup/)
- **MXFP4 ISQ quantization**: MXFP4 with optimized decode kernels for faster, smaller models. [Quantization docs](https://hanzoai.github.io/engine/reference/quantization-types/)

## Why hanzo?

- **Any Hugging Face model, zero config**: Just `hanzo run -m user/model`. Architecture, quantization format, and chat template are auto-detected.
- **True multimodality**: Text, vision, video, and audio, speech generation, image generation, and embeddings in one engine.
- **Smart quantization**: `--quant` automatically selects the best quantization format at that level: using a prebuilt UQFF if one is published, otherwise applying ISQ. [Docs](https://hanzoai.github.io/engine/tutorials/06-quantize-a-model/)
- **OpenAI + Anthropic compatible serving**: The same `hanzo serve` process exposes OpenAI-compatible `/v1` endpoints and an Anthropic-compatible Messages endpoint.
- **Built-in web UI**: Served at `/ui` by default. Shows reasoning, code execution, plots, and files inline. Edit any message and the new branch runs with its own Python state. Pass `--no-ui` to disable.
- **Hardware-aware**: `hanzo tune` benchmarks your system and picks optimal quantization + device mapping.
- **Flexible SDKs**: Python package and Rust crate to build your projects.
- **Native agentic support**: built-in [agentic loop](https://hanzoai.github.io/engine/guides/agents/) with web search, local Python code execution with model feedback, session management, and custom tool hooks.

## Quick Start

### Install

**Linux/macOS:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/hanzoai/engine/master/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/hanzoai/engine/master/install.ps1 | iex
```

[Manual installation & other platforms](https://hanzoai.github.io/engine/guides/install/)

### Run Your First Model

```bash
# Interactive chat
hanzo run -m Qwen/Qwen3-4B

# One-shot prompt (no interactive session)
hanzo run -m Qwen/Qwen3-4B -i "What is the capital of France?"

# One-shot with an image
hanzo run -m google/gemma-4-E4B-it --image photo.jpg -i "Describe this image"

# Agentic REPL: search + code execution from the terminal
hanzo run --agent -m Qwen/Qwen3-4B

# Start an API server with the built-in web UI
hanzo serve -m google/gemma-4-E4B-it
```

For the server command, visit `http://localhost:1234/ui` for the web chat interface. OpenAI-compatible clients use `http://localhost:1234/v1`; Anthropic-compatible clients use `http://localhost:1234`.

### The `hanzo` CLI

The CLI is designed to be **zero-config**: just point it at a model and go.

- **Auto-detection**: Automatically detects model architecture, quantization format, and chat template
- **All-in-one**: Single binary for chat, server, benchmarks, and web UI (`run`, `serve`, `bench`)
- **Hardware tuning**: Run `hanzo tune` to automatically benchmark and configure optimal settings for your hardware
- **Format-agnostic**: Works with Hugging Face models, GGUF files, and [UQFF quantizations](https://hanzoai.github.io/engine/reference/uqff-format/) seamlessly

```bash
# Auto-tune for your hardware and emit a config file
hanzo tune -m Qwen/Qwen3-4B --emit-config config.toml

# Run using the generated config
hanzo from-config -f config.toml

# Diagnose system issues (CUDA, Metal, HuggingFace connectivity)
hanzo doctor
```

[Full CLI documentation](https://hanzoai.github.io/engine/reference/cli/)

<details open>
  <summary><b>UI Demo</b></summary>
  <br>
  <img src="https://raw.githubusercontent.com/hanzoai/engine/master/res/chat.gif" alt="Web Chat UI Demo" />
</details>

## What Makes It Fast

**Performance**
- Continuous batching support by default on all devices.
- CUDA with [FlashAttention](https://hanzoai.github.io/engine/guides/perf/use-flash-attention/) V2/V3, Metal, [multi-GPU tensor parallelism](https://hanzoai.github.io/engine/guides/perf/multi-gpu-tensor-parallel/)
- [PagedAttention](https://hanzoai.github.io/engine/guides/perf/use-paged-attention/) for high throughput continuous batching on CUDA or Apple Silicon, prefix caching (including multimodal)

**Quantization** ([full docs](https://hanzoai.github.io/engine/reference/quantization-types/))
- [In-situ quantization (ISQ)](https://hanzoai.github.io/engine/guides/perf/pick-a-quantization/) of any Hugging Face model
- GGUF (2-8 bit), GPTQ, AWQ, HQQ, FP8, BNB support
- ⭐ [Per-layer topology](https://hanzoai.github.io/engine/guides/perf/topology/): Fine-tune quantization per layer for optimal quality/speed
- ⭐ Auto-select fastest quant method for your hardware

**Flexibility**
- [LoRA & X-LoRA](https://hanzoai.github.io/engine/guides/customize/lora-adapters/) with weight merging
- [AnyMoE](https://hanzoai.github.io/engine/guides/customize/anymoe/): Create mixture-of-experts on any base model
- [Multiple models](https://hanzoai.github.io/engine/guides/serve/multiple-models/): Load/unload at runtime

**Agentic Features**
- Integrated [tool calling](https://hanzoai.github.io/engine/guides/agents/tool-calling-basics/) with grammar enforcement and strict schema mode
- ⭐ Server-side [agentic loop](https://hanzoai.github.io/engine/guides/agents/configure-tool-loop/): auto-execute tools and feed results back
- ⭐ [Python code execution](https://hanzoai.github.io/engine/guides/agents/enable-code-execution/): persistent Jupyter-like sessions with matplotlib capture and multimodal feedback
- ⭐ [Web search integration](https://hanzoai.github.io/engine/guides/agents/web-search/) with embedding-based ranking
- ⭐ [Tool dispatch URL](https://hanzoai.github.io/engine/guides/agents/configure-tool-loop/): POST tool calls to your own endpoint
- ⭐ [MCP client](https://hanzoai.github.io/engine/guides/agents/connect-mcp-server/): Connect to external tools via Process, HTTP, or WebSocket
- Python/Rust [tool callbacks](https://hanzoai.github.io/engine/guides/agents/tool-calling-basics/) for custom execution

[Full feature documentation](https://hanzoai.github.io/engine/)

## Supported Models

<details>
<summary><b>Text Models</b></summary>

- Granite 4.0
- SmolLM 3
- DeepSeek V3
- GPT-OSS
- DeepSeek V2
- Qwen 3 Next
- Qwen 3 MoE
- Phi 3.5 MoE
- Qwen 3
- GLM 4
- GLM-4.7-Flash
- GLM-4.7 (MoE)
- Gemma 2
- Qwen 2
- Starcoder 2
- Phi 3
- Mixtral
- Phi 2
- Gemma
- Llama
- Mistral
</details>

<details>
<summary><b>Multimodal Models</b></summary>

- Qwen 3.5
- Qwen 3.5 MoE
- Qwen 3-VL
- Qwen 3-VL MoE
- Gemma 3n
- Llama 4
- Gemma 3
- Mistral 3
- Phi 4 multimodal
- Qwen 2.5-VL
- MiniCPM-O
- Llama 3.2 Vision
- Qwen 2-VL
- Idefics 3
- Idefics 2
- LLaVA Next
- LLaVA
- Phi 3V
</details>

<details>
<summary><b>Speech Models</b></summary>

- Voxtral (ASR/speech-to-text)
- Dia
</details>

<details>
<summary><b>Image Generation Models</b></summary>

- FLUX
</details>

<details>
<summary><b>Embedding Models</b></summary>

- Embedding Gemma
- Qwen 3 Embedding
</details>

[Request a new model](https://github.com/hanzoai/engine/issues/156) | [Full compatibility tables](https://hanzoai.github.io/engine/reference/supported-models/)

## Python SDK

```bash
pip install hanzo  # or hanzo-cuda, hanzo-metal, hanzo-mkl, hanzo-accelerate
```

```python
from hanzo import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)
```

[Python SDK](https://hanzoai.github.io/engine/tutorials/03-python-sdk/) | [Installation](https://hanzoai.github.io/engine/guides/install/) | [Examples](examples/python) | [Cookbook](examples/python/cookbook.ipynb)

## Rust SDK

```bash
cargo add hanzo
```

```rust
use anyhow::Result;
use hanzo::{IsqType, TextMessageRole, TextMessages, MultimodalModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = MultimodalModelBuilder::new("google/gemma-4-E4B-it")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "Hello!",
    );

    let response = model.send_chat_request(messages).await?;

    println!("{:?}", response.choices[0].message.content);

    Ok(())
}
```

[API Docs](https://docs.rs/hanzo) | [Crate](https://crates.io/crates/hanzo) | [Examples](hanzo/examples)

## Docker

For quick containerized deployment:

```bash
docker pull ghcr.io/hanzoai/engine:latest
docker run --gpus all -p 1234:1234 ghcr.io/hanzoai/engine:latest \
  serve -m Qwen/Qwen3-4B
```

[Docker images](https://github.com/hanzoai/engine/pkgs/container/hanzo)

> For production use, we recommend installing the CLI directly for maximum flexibility.

## Documentation

For complete documentation, see the **[Documentation](https://hanzoai.github.io/engine/)**.

**Quick Links:**
- [CLI Reference](https://hanzoai.github.io/engine/reference/cli/) - All commands and options
- [HTTP API](https://hanzoai.github.io/engine/reference/http-api/) - OpenAI-compatible endpoints
- [Quantization](https://hanzoai.github.io/engine/reference/quantization-types/) - ISQ, GGUF, GPTQ, and more
- [Device Mapping](https://hanzoai.github.io/engine/explanation/device-mapping/) - Multi-GPU and CPU offloading
- [MCP Integration](https://hanzoai.github.io/engine/guides/agents/connect-mcp-server/) - MCP integration documentation
- [Troubleshooting](https://hanzoai.github.io/engine/reference/troubleshooting/) - Common issues and solutions
- [Configuration](https://hanzoai.github.io/engine/reference/environment-variables/) - Environment variables for configuration

## Contributing

Contributions welcome! Please [open an issue](https://github.com/hanzoai/engine/issues) to discuss new features or report bugs. If you want to add a new model, please contact us via an issue and we can coordinate.

## Credits

This project would not be possible without the excellent work at [Hanzo](https://github.com/hanzoai/ml). Thank you to all [contributors](https://github.com/hanzoai/engine/graphs/contributors)!

hanzo is not affiliated with Mistral AI.

<p align="right">
  <a href="#top">Back to Top</a>
</p>
