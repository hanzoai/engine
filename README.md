<a name="top"></a>

<h1 align="center">
  Hanzo Engine
</h1>

<h3 align="center">
Production AI inference at any scale.
</h3>

[![CI](https://github.com/hanzoai/engine/actions/workflows/ci.yml/badge.svg)](https://github.com/hanzoai/engine/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.88+-orange)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![GHCR](https://img.shields.io/badge/ghcr.io-hanzoai%2Fengine-2496ED)](https://github.com/hanzoai/engine/pkgs/container/engine)
[![Crates.io](https://img.shields.io/crates/v/hanzo-engine)](https://crates.io/crates/hanzo-engine)
[![PyPI](https://img.shields.io/pypi/v/hanzo-engine)](https://pypi.org/project/hanzo-engine)
[![Discord](https://img.shields.io/badge/discord-hanzo-7289DA)](https://discord.gg/SZrecqK8qw)

<p align="center">
| <a href="https://github.com/hanzoai/engine"><b>GitHub</b></a> | <a href="https://docs.hanzo.ai/docs/services/engine"><b>Documentation</b></a> | <a href="https://engine.hanzo.ai/docs/rust-sdk"><b>Rust SDK</b></a> | <a href="https://engine.hanzo.ai/docs/python-sdk"><b>Python SDK</b></a> | <a href="https://discord.gg/SZrecqK8qw"><b>Discord</b></a> |
</p>

High-performance cloud inference engine serving [Zen models](https://zenlm.org) and 60+ model architectures. CUDA, Metal, and CPU backends with paged attention, continuous batching, speculative decoding, and tensor parallelism. Built in Rust on [Hanzo ML](https://github.com/hanzoai/ml).

- **Multimodal**: text, vision, audio, speech, image generation, embeddings, agents
- **APIs**: Rust SDK, Python SDK, OpenAI-compatible HTTP server, MCP server and client
- **Performance**: PagedAttention, FlashAttention V2/V3, in-situ quantization, per-layer topology
- **Scale**: Multi-GPU via NCCL, multi-node via TCP ring, continuous batching
- **Ecosystem**: integrates with [Hanzo Edge](https://github.com/hanzoai/edge), [Hanzo Gateway](https://github.com/hanzoai/gateway), [Hanzo Cloud](https://github.com/hanzoai/cloud), and [Hanzo MCP](https://github.com/hanzoai/mcp)

For full documentation, see [docs.hanzo.ai/docs/services/engine](https://docs.hanzo.ai/docs/services/engine).

---

## Quick Start

### Install

**Linux / macOS:**

```bash
curl -sSL https://engine.hanzo.ai/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://engine.hanzo.ai/install.ps1 | iex
```

**Via Cargo:**

```bash
cargo install hanzo-engine
```

[Manual installation & other platforms](docs/INSTALLATION.md)

### Run Your First Model

```bash
# Interactive chat with a Zen model
hanzo-engine run -m zenlm/zen4-mini

# Start an OpenAI-compatible server with web UI
hanzo-engine serve --ui -m zenlm/zen4-mini --port 8000

# Serve any Hugging Face model
hanzo-engine serve -m google/gemma-3-4b-it --port 8000
```

Visit `http://localhost:8000/ui` for the web chat interface.

---

## Docker

Pre-built images are published to `ghcr.io/hanzoai/engine` for every release.

| Tag | Backend | Use Case |
|-----|---------|----------|
| `latest` | CPU | Development, CI, ARM64 servers |
| `cuda` | NVIDIA CUDA | Production GPU serving |
| `cuda-<version>` | NVIDIA CUDA (pinned) | Reproducible deployments |
| `metal` | Apple Metal | macOS GPU serving |

### Quick Run

```bash
# CPU -- good for small models and testing
docker run -p 8000:8000 \
  -v hanzo-models:/root/.cache/huggingface \
  ghcr.io/hanzoai/engine:latest \
  serve -m zenlm/zen4-mini --port 8000

# NVIDIA GPU -- production serving
docker run -p 8000:8000 --gpus all \
  -v hanzo-models:/root/.cache/huggingface \
  ghcr.io/hanzoai/engine:cuda \
  serve -m zenlm/zen4 --port 8000

# NVIDIA GPU with quantization -- reduce VRAM usage
docker run -p 8000:8000 --gpus all \
  -v hanzo-models:/root/.cache/huggingface \
  ghcr.io/hanzoai/engine:cuda \
  serve -m zenlm/zen4 --isq Q4K --port 8000

# Apple Silicon (Metal)
docker run -p 8000:8000 \
  -v hanzo-models:/root/.cache/huggingface \
  ghcr.io/hanzoai/engine:metal \
  serve -m zenlm/zen4-mini --port 8000

# Multi-GPU with tensor parallelism
docker run -p 8000:8000 --gpus all \
  -e HANZO_ENGINE_LOCAL_WORLD_SIZE=4 \
  -v hanzo-models:/root/.cache/huggingface \
  ghcr.io/hanzoai/engine:cuda \
  serve -m zenlm/zen4-max --port 8000

# With HuggingFace token for gated models
docker run -p 8000:8000 --gpus all \
  -e HF_TOKEN=hf_your_token_here \
  -v hanzo-models:/root/.cache/huggingface \
  ghcr.io/hanzoai/engine:cuda \
  serve -m zenlm/zen4 --port 8000
```

### Docker Compose

```yaml
services:
  engine:
    image: ghcr.io/hanzoai/engine:cuda
    ports:
      - "8000:8000"
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - HANZO_ENGINE_LOCAL_WORLD_SIZE=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - model-cache:/root/.cache/huggingface
    command: serve -m zenlm/zen4 --port 8000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

volumes:
  model-cache:
```

---

## Kubernetes Deployment

### Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hanzo-engine
  labels:
    app: hanzo-engine
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hanzo-engine
  template:
    metadata:
      labels:
        app: hanzo-engine
    spec:
      nodeSelector:
        nvidia.com/gpu.present: "true"
      containers:
        - name: engine
          image: ghcr.io/hanzoai/engine:cuda
          command: ["hanzo-engine", "serve", "-m", "zenlm/zen4", "--port", "8000"]
          ports:
            - name: http
              containerPort: 8000
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hanzo-engine-secrets
                  key: hf-token
            - name: HANZO_ENGINE_LOCAL_WORLD_SIZE
              value: "1"
          resources:
            requests:
              memory: "32Gi"
              cpu: "4"
              nvidia.com/gpu: "1"
            limits:
              memory: "64Gi"
              nvidia.com/gpu: "1"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 30
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: hanzo-engine
spec:
  selector:
    app: hanzo-engine
  ports:
    - name: http
      port: 8000
      targetPort: 8000
  type: ClusterIP
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hanzo-engine-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hanzo-engine
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: engine_requests_in_flight
        target:
          type: AverageValue
          averageValue: "10"
```

### Multi-GPU StatefulSet

For large models requiring tensor parallelism across multiple GPUs:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: hanzo-engine-multi-gpu
spec:
  serviceName: hanzo-engine-multi-gpu
  replicas: 1
  selector:
    matchLabels:
      app: hanzo-engine-multi-gpu
  template:
    metadata:
      labels:
        app: hanzo-engine-multi-gpu
    spec:
      nodeSelector:
        nvidia.com/gpu.count: "8"
      containers:
        - name: engine
          image: ghcr.io/hanzoai/engine:cuda
          command: ["hanzo-engine", "serve", "-m", "zenlm/zen4-max", "--port", "8000"]
          env:
            - name: HANZO_ENGINE_LOCAL_WORLD_SIZE
              value: "8"
          resources:
            limits:
              nvidia.com/gpu: "8"
              memory: "640Gi"
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 64Gi
  volumeClaimTemplates:
    - metadata:
        name: model-cache
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 500Gi
```

---

## Zen Models

Hanzo Engine is the inference backend for the [Zen model family](https://zenlm.org). First-class support for all 14 production Zen models.

| Model | Parameters | Active Params | Context | Architecture | Modality | Use Case |
|-------|-----------|---------------|---------|--------------|----------|----------|
| **zen4** | 744B MoE | 40B | 202K | Transformer MoE | Text | Flagship reasoning and generation |
| **zen4-max** | 1.04T MoE | 32B | 256K | Transformer MoE | Text | Maximum capability, longest context |
| **zen4-ultra** | 744B MoE + CoT | 40B | 202K | Transformer MoE | Text | Extended chain-of-thought reasoning |
| **zen4-pro** | 80B MoE | 3B | 131K | Transformer MoE | Text | High quality, efficient serving |
| **zen4-mini** | 8B dense | 8B | 40K | Transformer | Text | Fast inference, edge deployment |
| **zen4-coder** | 480B MoE | 35B | 262K | Transformer MoE | Code | Code generation and analysis |
| **zen4-coder-flash** | 30B MoE | 3B | 262K | Transformer MoE | Code | Fast code completion |
| **zen4-coder-pro** | 480B dense BF16 | 480B | 262K | Transformer | Code | Maximum code quality |
| **zen3-vl** | 30B MoE | 3B | 131K | Vision-Language MoE | Text + Vision | Multimodal understanding |
| **zen3-omni** | ~200B | ~200B | 202K | Multimodal Transformer | Text + Vision + Audio | Unified multimodal |
| **zen3-nano** | 4B dense | 4B | 40K | Transformer | Text | Ultra-lightweight, embedded |
| **zen3-guard** | 4B dense | 4B | - | Classifier | Safety | Content filtering, guardrails |
| **zen3-embedding** | - | - | - | Embedding (3072-dim) | Embedding | Search, retrieval, RAG |
| **zen-agent** | - | - | - | Agent Framework | Agents | Autonomous tool use, planning |

```bash
# Serve any Zen model
hanzo-engine serve -m zenlm/zen4-mini --port 8000
hanzo-engine serve -m zenlm/zen4 --port 8000 --isq Q4K
hanzo-engine serve -m zenlm/zen4-coder --port 8000
hanzo-engine serve -m zenlm/zen4-max --port 8000

# Vision model
hanzo-engine serve -m zenlm/zen3-vl --port 8000

# Embedding model
hanzo-engine serve -m zenlm/zen3-embedding --port 8000
```

See all Zen model weights at [@zenlm on HuggingFace](https://huggingface.co/zenlm).

---

## Performance Features

| Category | Feature | Description |
|----------|---------|-------------|
| **Attention** | PagedAttention | High-throughput KV cache management on CUDA and Metal with prefix caching |
| **Attention** | FlashAttention V2 | Memory-efficient attention for Ampere+ GPUs (CC >= 8.0) |
| **Attention** | FlashAttention V3 | Optimized attention for Hopper GPUs (CC >= 9.0, H100/H200) |
| **Batching** | Continuous Batching | Dynamic request batching across all backends, enabled by default |
| **Batching** | Prompt Caching | Block-level prefix caching across requests sharing common prefixes |
| **Decoding** | Speculative Decoding | Draft-model acceleration with rejection sampling for 2-3x speedup |
| **Decoding** | Constrained Decoding | Regex, Lark grammar, JSON schema, llguidance |
| **Memory** | KV Cache Quantization | FP8 (E4M3) cache compression, halving memory with minimal quality loss |
| **Quantization** | ISQ | In-situ quantization of any HuggingFace model at load time |
| **Quantization** | GGUF | Pre-quantized 2-8 bit model loading |
| **Quantization** | GPTQ / AWQ / HQQ | Pre-quantized model formats |
| **Quantization** | FP8 / BNB | 8-bit and bitsandbytes quantization |
| **Quantization** | Per-Layer Topology | Fine-tune quantization per layer for optimal quality/speed tradeoff |
| **Quantization** | UQFF | Universal Quantized File Format for portable quantized models |
| **Parallelism** | NCCL Tensor Parallelism | Multi-GPU on NVIDIA (recommended for CUDA) |
| **Parallelism** | Ring Tensor Parallelism | TCP-based, cross-device, cross-machine (Metal + CUDA + CPU) |
| **Adapters** | LoRA / X-LoRA | Runtime adapter loading with weight merging |
| **Adapters** | AnyMoE | Create mixture-of-experts on any base model |
| **Serving** | Multi-Model | Load and unload models at runtime via API |
| **Serving** | Auto-Detection | Automatically detects architecture, quantization, and chat template |

### Compute Backends

| Backend | Platform | Hardware |
|---------|----------|----------|
| **CUDA** | Linux, Windows (WSL) | NVIDIA GPUs (all generations) |
| **cuDNN** | Linux, Windows (WSL) | NVIDIA GPUs (optimized primitives) |
| **Metal** | macOS | Apple Silicon, AMD GPUs |
| **Accelerate** | macOS | Apple CPU optimization |
| **MKL** | Linux, Windows | Intel CPU optimization |
| **CPU** | All platforms | Any x86_64 or ARM64 processor |

---

## API Reference

Hanzo Engine exposes a fully OpenAI-compatible HTTP API. Interactive docs are available at `http://localhost:<port>/docs` (Swagger UI).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (streaming and non-streaming) |
| POST | `/v1/completions` | Text completion |
| POST | `/v1/embeddings` | Generate embeddings |
| POST | `/v1/images/generations` | Image generation (FLUX models) |
| GET | `/v1/models` | List loaded models |
| POST | `/v1/models/load` | Load a model at runtime |
| POST | `/v1/models/unload` | Unload a model at runtime |
| GET | `/health` | Health check (returns 200 when ready) |
| GET | `/docs` | Interactive API documentation (Swagger UI) |
| GET | `/ui` | Built-in web chat interface (when `--ui` is enabled) |

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in one paragraph."}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Embeddings

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": ["Search query", "Document passage to embed"]
  }'
```

### Image Generation

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "A futuristic cityscape at sunset, photorealistic",
    "n": 1,
    "size": "1024x1024"
  }'
```

### Extended Parameters

Hanzo Engine extends the OpenAI API with additional parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `top_k` | int | Top-K sampling |
| `min_p` | float | Minimum probability threshold |
| `grammar` | object | Constrained decoding (regex, Lark, JSON schema, llguidance) |
| `enable_thinking` | bool | Enable chain-of-thought for supported models (zen4-ultra) |
| `web_search_options` | object | Enable web search integration |
| `reasoning_effort` | string | Control reasoning depth: low, medium, high |
| `repetition_penalty` | float | Multiplicative penalty for repeated tokens |
| `truncate_sequence` | bool | Truncate overlong prompts instead of rejecting |

[Full HTTP API documentation](docs/HTTP.md)

---

## Python SDK

### Installation

```bash
pip install hanzo-engine               # CPU
pip install hanzo-engine-cuda          # NVIDIA GPU
pip install hanzo-engine-metal         # Apple Silicon
pip install hanzo-engine-mkl           # Intel CPU
```

### OpenAI SDK (recommended for HTTP server)

Any OpenAI-compatible client works out of the box:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

# Basic chat completion
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Write a haiku about Rust."}
    ],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain paged attention."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Vision (multimodal)

```python
import base64

with open("image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ],
    }],
)
print(response.choices[0].message.content)
```

### Embeddings

```python
response = client.embeddings.create(
    model="default",
    input=["Search query", "Document to embed"],
)
for item in response.data:
    print(f"Embedding dimension: {len(item.embedding)}")
```

### Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    },
}]

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto",
)

if response.choices[0].message.tool_calls:
    for call in response.choices[0].message.tool_calls:
        print(f"Function: {call.function.name}")
        print(f"Arguments: {call.function.arguments}")
```

### Native Python Bindings

For direct in-process inference without an HTTP server:

```python
from hanzo_engine import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(model_id="zenlm/zen4-mini"),
    in_situ_quant="4",
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=256,
    )
)
print(response.choices[0].message.content)
```

[Python SDK docs](docs/PYTHON_SDK.md) | [Installation](docs/PYTHON_INSTALLATION.md) | [Examples](examples/python) | [Cookbook](examples/python/cookbook.ipynb)

---

## Rust SDK

```bash
cargo add hanzo-engine
```

```rust
use anyhow::Result;
use hanzo_engine::{IsqType, TextMessageRole, TextMessages, TextModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("zenlm/zen4-mini")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(TextMessageRole::User, "Hello!");

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
```

[Rust API docs](https://docs.rs/hanzo-engine) | [Examples](hanzo-engine/examples)

---

## CLI Reference

The `hanzo-engine` CLI is designed to be **zero-config**: point it at a model and go.

```bash
# Interactive chat
hanzo-engine run -m zenlm/zen4-mini

# HTTP server with web UI
hanzo-engine serve --ui -m zenlm/zen4 --port 8000

# Auto-tune for your hardware
hanzo-engine tune -m zenlm/zen4-mini --emit-config config.toml

# Run from generated config
hanzo-engine from-config -f config.toml

# Benchmark a model
hanzo-engine bench -m zenlm/zen4-mini

# Generate quantized UQFF file
hanzo-engine quantize -m zenlm/zen4 --isq Q4K -o zen4-q4k.uqff

# System diagnostics
hanzo-engine doctor

# Manage model cache
hanzo-engine cache list
hanzo-engine cache clean
```

### Commands

| Command | Description |
|---------|-------------|
| `run` | Interactive chat mode |
| `serve` | Start OpenAI-compatible HTTP/MCP server |
| `from-config` | Run from TOML configuration file |
| `quantize` | Generate UQFF quantized model file |
| `tune` | Auto-benchmark and recommend settings for your hardware |
| `doctor` | System diagnostics (CUDA, Metal, HuggingFace connectivity) |
| `login` | Authenticate with HuggingFace Hub |
| `cache` | Manage the model cache |
| `bench` | Performance benchmarking |

[Full CLI documentation](docs/CLI.md)

---

## Supported Architectures

Hanzo Engine supports 60+ model architectures across five modalities.

### Text Models

| Architecture | GGUF | ISQ | LoRA | AnyMoE |
|-------------|------|-----|------|--------|
| Llama (1/2/3/3.1/3.3) | Yes | Yes | Yes | Yes |
| Mistral (7B/Nemo) | Yes | Yes | Yes | Yes |
| Mixtral | Yes | Yes | Yes | - |
| Qwen 2 | - | Yes | - | Yes |
| Qwen 3 | Yes | Yes | - | - |
| Qwen 3 MoE | - | Yes | - | - |
| Qwen 3 Next | - | Yes | - | - |
| Gemma | - | Yes | Yes | Yes |
| Gemma 2 | - | Yes | Yes | Yes |
| Phi 2 | Yes | Yes | Yes | Yes |
| Phi 3 | Yes | Yes | Yes | Yes |
| Phi 3.5 MoE | - | Yes | - | - |
| Starcoder 2 | - | Yes | Yes | Yes |
| DeepSeek V2 | - | Yes | - | - |
| DeepSeek V3 | - | Yes | - | - |
| GLM 4 | - | Yes | Yes | - |
| GLM-4.7-Flash (MoE) | - | Yes | - | - |
| GLM-4.7 (MoE) | - | Yes | - | - |
| SmolLM 3 | - | Yes | Yes | Yes |
| Granite 4.0 | - | Yes | - | - |
| GPT-OSS | - | Yes | - | - |

### Vision Models

| Architecture | ISQ | LoRA | AnyMoE |
|-------------|-----|------|--------|
| Qwen 3-VL | Yes | - | - |
| Qwen 3-VL MoE | Yes | - | - |
| Qwen 2.5-VL | Yes | - | - |
| Qwen 2-VL | Yes | - | - |
| Gemma 3 | Yes | - | Yes |
| Gemma 3n | Yes | - | - |
| Llama 4 | Yes | - | - |
| Llama 3.2 Vision | Yes | - | - |
| Mistral 3 | Yes | - | Yes |
| Phi 3V | Yes | - | - |
| Phi 4 Multimodal | Yes | - | - |
| MiniCPM-O 2.6 | Yes | - | - |
| Idefics 2 | Yes | - | - |
| Idefics 3 | Yes | - | Yes |
| LLaVA | Yes | - | Yes |
| LLaVA Next | Yes | - | Yes |

### Speech Models

| Architecture | ISQ | Description |
|-------------|-----|-------------|
| Voxtral | Yes | ASR / speech-to-text |
| Dia | Yes | Text-to-speech |

### Image Generation Models

| Architecture | Description |
|-------------|-------------|
| FLUX | High-quality image generation |

### Embedding Models

| Architecture | ISQ | Description |
|-------------|-----|-------------|
| Embedding Gemma | Yes | Text embeddings |
| Qwen 3 Embedding | Yes | Text embeddings |

---

## Deployment

### Multi-GPU (NCCL)

```bash
# 2x GPU tensor parallelism
HANZO_ENGINE_LOCAL_WORLD_SIZE=2 hanzo-engine serve \
  -m zenlm/zen4 --port 8000

# 4x GPU
HANZO_ENGINE_LOCAL_WORLD_SIZE=4 hanzo-engine serve \
  -m zenlm/zen4-max --port 8000
```

### Multi-Node (Ring)

```bash
# Node 0 (master)
RING_CONFIG=ring_node0.json hanzo-engine serve -m zenlm/zen4 --port 8000

# Node 1
RING_CONFIG=ring_node1.json hanzo-engine serve -m zenlm/zen4 --port 8001
```

Ring config example (`ring_node0.json`):

```json
{
  "master_ip": "0.0.0.0",
  "master_port": 1234,
  "port": 12345,
  "right_port": 12346,
  "rank": 0,
  "world_size": 2
}
```

Ring supports heterogeneous setups: mix Metal, CUDA, and CPU nodes in a single inference cluster.

[Distributed inference docs](docs/DISTRIBUTED/DISTRIBUTED.md)

---

## Performance

Benchmarks measured on representative hardware with continuous batching enabled.

| Model | Hardware | Quantization | Throughput (tok/s) | Latency (TTFT) | Memory |
|-------|----------|-------------|-------------------|----------------|--------|
| zen4-mini (8B) | 1x A100 80GB | FP16 | ~2,400 | 28ms | 16 GB |
| zen4-mini (8B) | 1x A100 80GB | Q4K ISQ | ~3,800 | 18ms | 5 GB |
| zen4-mini (8B) | M3 Max 64GB | Metal | ~85 | 120ms | 16 GB |
| zen4-mini (8B) | M3 Max 64GB | Q4K ISQ | ~110 | 80ms | 5 GB |
| zen4-pro (80B MoE) | 1x A100 80GB | Q4K ISQ | ~950 | 65ms | 42 GB |
| zen4 (744B MoE) | 4x H100 | FP8 + NCCL | ~1,200 | 180ms | 280 GB |
| zen4 (744B MoE) | 8x A100 80GB | Q4K + NCCL | ~800 | 250ms | 320 GB |

Run your own benchmarks:

```bash
hanzo-engine bench -m zenlm/zen4-mini --isq Q4K
hanzo-engine tune -m zenlm/zen4-mini --emit-config optimal.toml
```

---

## Building from Source

```bash
git clone https://github.com/hanzoai/engine.git
cd engine
```

### Feature Flags

| Feature | Description | Requires |
|---------|-------------|----------|
| `cuda` | NVIDIA GPU acceleration | CUDA toolkit |
| `cudnn` | cuDNN optimized primitives | CUDA + cuDNN |
| `flash-attn` | FlashAttention V2 | CUDA, CC >= 8.0 (Ampere+) |
| `flash-attn-v3` | FlashAttention V3 | CUDA, CC >= 9.0 (Hopper) |
| `metal` | Apple Metal GPU | macOS |
| `accelerate` | Apple CPU optimization | macOS |
| `mkl` | Intel MKL CPU optimization | Intel MKL |
| `nccl` | Multi-GPU tensor parallelism | CUDA + NCCL |
| `ring` | Multi-node ring topology | TCP networking |

### Build by Hardware

```bash
# NVIDIA GPU (Ampere or newer, recommended)
cargo build --release --features "cuda cudnn flash-attn"

# NVIDIA Hopper (H100)
cargo build --release --features "cuda cudnn flash-attn-v3"

# NVIDIA Multi-GPU
cargo build --release --features "cuda cudnn flash-attn nccl"

# Apple Silicon
cargo build --release --features "metal accelerate"

# Intel CPU
cargo build --release --features "mkl"

# CPU only (no features needed)
cargo build --release
```

### Requirements

- Rust 1.88+
- For CUDA: NVIDIA CUDA toolkit, `nvcc` in PATH
- For Flash Attention V2: GPU compute capability >= 8.0
- For Flash Attention V3: GPU compute capability >= 9.0
- For MKL: Intel oneAPI or standalone MKL installation
- For NCCL: NVIDIA NCCL library

[Full cargo features reference](docs/CARGO_FEATURES.md)

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `HANZO_ENGINE_LOCAL_WORLD_SIZE` | Number of GPUs for NCCL tensor parallelism |
| `HANZO_ENGINE_NO_NCCL=1` | Disable NCCL, use device mapping instead |
| `RING_CONFIG` | Path to ring topology JSON config |
| `KEEP_ALIVE_INTERVAL` | SSE keep-alive interval in ms |
| `HF_TOKEN` | HuggingFace Hub authentication token |

### TOML Configuration

For complex setups, use a TOML config file:

```toml
[model]
model_id = "zenlm/zen4"
isq = "Q4K"

[server]
port = 8000
log = "info"

[paged_attention]
gpu_mem_fraction = 0.9
block_size = 32
cache_type = "f8e4m3"

[speculative]
draft_model = "zenlm/zen4-mini"
gamma = 16
```

```bash
hanzo-engine from-config -f config.toml
```

[Configuration reference](docs/CONFIGURATION.md)

---

## Documentation

| Topic | Link |
|-------|------|
| Full Documentation | [docs.hanzo.ai/docs/services/engine](https://docs.hanzo.ai/docs/services/engine) |
| CLI Reference | [docs/CLI.md](docs/CLI.md) |
| HTTP API | [docs/HTTP.md](docs/HTTP.md) |
| Quantization Guide | [docs/QUANTS.md](docs/QUANTS.md) |
| ISQ (In-Situ Quantization) | [docs/ISQ.md](docs/ISQ.md) |
| PagedAttention | [docs/PAGED_ATTENTION.md](docs/PAGED_ATTENTION.md) |
| FlashAttention | [docs/FLASH_ATTENTION.md](docs/FLASH_ATTENTION.md) |
| Speculative Decoding | [docs/SPECULATIVE_DECODING.md](docs/SPECULATIVE_DECODING.md) |
| Distributed Inference | [docs/DISTRIBUTED/DISTRIBUTED.md](docs/DISTRIBUTED/DISTRIBUTED.md) |
| Device Mapping | [docs/DEVICE_MAPPING.md](docs/DEVICE_MAPPING.md) |
| Per-Layer Topology | [docs/TOPOLOGY.md](docs/TOPOLOGY.md) |
| LoRA & X-LoRA | [docs/ADAPTER_MODELS.md](docs/ADAPTER_MODELS.md) |
| AnyMoE | [docs/ANYMOE.md](docs/ANYMOE.md) |
| Tool Calling | [docs/TOOL_CALLING.md](docs/TOOL_CALLING.md) |
| Web Search | [docs/WEB_SEARCH.md](docs/WEB_SEARCH.md) |
| MCP Integration | [docs/MCP/README.md](docs/MCP/README.md) |
| Cargo Features | [docs/CARGO_FEATURES.md](docs/CARGO_FEATURES.md) |
| Python SDK | [docs/PYTHON_SDK.md](docs/PYTHON_SDK.md) |
| Rust SDK | [docs/RUST_SDK.md](docs/RUST_SDK.md) |
| Troubleshooting | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| Configuration | [docs/CONFIGURATION.md](docs/CONFIGURATION.md) |

---

## Related Projects

| Project | Description |
|---------|-------------|
| [Hanzo Edge](https://github.com/hanzoai/edge) | On-device inference for mobile, web, and embedded (WASM, Metal, CPU) |
| [Hanzo Gateway](https://github.com/hanzoai/gateway) | API gateway with rate limiting, auth, and circuit breakers |
| [Hanzo Ingress](https://github.com/hanzoai/ingress) | L7 reverse proxy with automatic TLS and Kubernetes-native routing |
| [Hanzo ML](https://github.com/hanzoai/ml) | Rust ML framework (tensor ops, neural networks, GPU kernels) |
| [Hanzo Cloud](https://github.com/hanzoai/cloud) | Cloud API gateway for AI inference |
| [Hanzo LLM Gateway](https://github.com/hanzoai/llm) | Unified proxy for 100+ LLM providers |
| [Hanzo Node](https://github.com/hanzoai/node) | Decentralized compute node for AI workloads |
| [Hanzo MCP](https://github.com/hanzoai/mcp) | Model Context Protocol tools (260+) |
| [Zen Models](https://zenlm.org) | Zen model family documentation and weights |
| [@zenlm](https://huggingface.co/zenlm) | Zen model weights on HuggingFace |

---

## License

MIT

<p align="right">
  <a href="#top">Back to Top</a>
</p>
