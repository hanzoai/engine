# Installation Guide

## Quick Install (Recommended)

Downloads a **prebuilt, cosign-signed `hanzoai` binary** for your platform — no
Rust, no compiler, no CUDA toolkit required. The server speaks the OpenAI API
(`/v1/chat/completions`) and the Anthropic API (`/v1/messages`) natively.

**Linux/macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/hanzoai/engine/main/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/hanzoai/engine/main/install.ps1 | iex
```

Prebuilt targets: **linux** `amd64`/`arm64`, **macos** `arm64`, **windows** `amd64`/`arm64`.
The installer auto-detects your OS + CPU, downloads the matching bundle from the
latest release, verifies it (cosign signature, or `SHA256SUMS` fallback), and puts
`hanzoai` on your `PATH`.

### Installer options (env vars)

| Variable | Effect |
|----------|--------|
| `HANZOAI_VERSION=v1.7.6` | Install a specific tag instead of `latest` |
| `HANZOAI_INSTALL_DIR=/opt/bin` | Install location (default: first writable of `/usr/local/bin`, `~/.local/bin`, `~/.hanzo/bin`) |
| `HANZOAI_BASE_URL=https://mirror/…` | Air-gapped / self-hosted release mirror |
| `HANZOAI_NO_VERIFY=1` | Skip signature/checksum verification |

## Test on a fresh machine

The exact commands to go from nothing to a working local LLM endpoint on a new box:

```bash
# 1. install the prebuilt engine (no dependencies)
curl -fsSL https://raw.githubusercontent.com/hanzoai/engine/main/install.sh | sh

# 2. it works
hanzoai --version

# 3. serve a small model on :1234 (downloads weights from HF on first run)
hanzoai --port 1234 run -m Qwen/Qwen3-4B &

# 4a. OpenAI-compatible endpoint
curl localhost:1234/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"say hi in 3 words"}]}'

# 4b. Anthropic-compatible endpoint (same engine, same weights)
curl localhost:1234/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"default","max_tokens":64,"messages":[{"role":"user","content":"say hi in 3 words"}]}'
```

> Prefer to join it to your cloud fleet instead of a bare port?
> `hanzo gpu connect --serve-engine` installs + serves this same binary and
> registers the node with your Hanzo Cloud account (see the `hanzo` CLI).

## Verifying signatures

Every bundle is cosign-signed keyless (Sigstore / GitHub OIDC). Each `<bundle>`
ships an adjacent `<bundle>.sig` + `<bundle>.pem`, and the release carries a
`SHA256SUMS`:

```bash
cosign verify-blob \
  --certificate hanzoai-linux-amd64.tar.gz.pem \
  --signature   hanzoai-linux-amd64.tar.gz.sig \
  --certificate-identity-regexp "https://github.com/hanzoai/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  hanzoai-linux-amd64.tar.gz
```

The installer performs this check automatically when `cosign` is on your `PATH`.

---

The rest of this guide covers **building from source** (for a GPU-accelerated
build, an unsupported platform, or local development). Building from source needs
the Rust toolchain:

- OpenSSL: `sudo apt install libssl-dev` (Ubuntu) · pkg-config: `sudo apt install pkg-config`
- Rust from https://rustup.rs/ — `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source $HOME/.cargo/env`
- (Optional) [FFmpeg](https://ffmpeg.org/) for video input — see [Video Input](VIDEO.md)
- (Optional) HuggingFace auth: `hanzo login` (or `huggingface-cli login`)

## Supported Accelerators

| Accelerator              | Feature Flag  | Additional Flags       |
|--------------------------|---------------|------------------------|
| NVIDIA GPUs (CUDA)       | `cuda`        | `flash-attn`, `flash-attn-v3`, `cudnn`  |
| Apple Silicon GPU (Metal)| `metal`       |                        |
| CPU (Intel)              | `mkl`         |                        |
| CPU (Apple Accelerate)   | `accelerate`  |                        |
| Generic CPU (ARM/AVX)    | _none_        | ARM NEON / AVX enabled by default |

> **Note for Linux users:** The `metal` feature is macOS-only. Use `--features "cuda flash-attn cudnn"` for NVIDIA GPUs or `--features mkl` for Intel CPUs instead of `--all-features`.

## Feature Detection

Determine which features to enable based on your hardware:

| Hardware | Features |
|----------|----------|
| NVIDIA GPU (Ampere+, compute >=80) | `cuda cudnn flash-attn` |
| NVIDIA GPU (Hopper, compute 90) | `cuda cudnn flash-attn flash-attn-v3` |
| NVIDIA GPU (older) | `cuda cudnn` |
| Apple Silicon (macOS) | `metal accelerate` |
| Intel CPU with MKL | `mkl` |
| CPU only | (no features needed) |

## Install from crates.io

```bash
cargo install hanzo-cli --features "<your-features>"
```

Example:
```bash
cargo install hanzo-cli --features "cuda flash-attn cudnn"
```

## Build from Source

```bash
git clone https://github.com/hanzoai/engine.git
cd engine
cargo install --path hanzo-cli --features "<your-features>"
```

Example:
```bash
cargo build --release --features "cuda flash-attn cudnn"
```

## Docker

Docker images are available for quick deployment:

```bash
docker pull ghcr.io/hanzoai/engine:latest
docker run --gpus all -p 1234:1234 ghcr.io/hanzoai/engine:latest \
  serve -m Qwen/Qwen3-4B
```

[Docker images on GitHub Container Registry](https://github.com/hanzoai/engine/pkgs/container/engine)

Learn more about running Docker containers: https://docs.docker.com/engine/reference/run/

## Python SDK

Install the Python package:

```bash
pip install hanzo-cuda    # For NVIDIA GPUs
pip install hanzo-metal   # For Apple Silicon
pip install hanzo-mkl     # For Intel CPUs
pip install hanzo         # CPU-only
```

- [Full installation instructions](PYTHON_INSTALLATION.md)
- [SDK documentation](PYTHON_SDK.md)

## Verify Installation

After installation, verify everything works:

```bash
# Check CLI is installed
hanzo --help

# Run system diagnostics
hanzo doctor

# Test with a small model
hanzo run -m Qwen/Qwen3-0.6B
```

## Getting Models

### From Hugging Face Hub (Default)

Models download automatically from Hugging Face Hub:

```bash
hanzo run -m meta-llama/Llama-3.2-3B-Instruct
```

For gated models, authenticate first:
```bash
hanzo login
# Or: hanzo run --token-source env:HF_TOKEN -m <model>
```

### From Local Files

Pass a path to a downloaded model:

```bash
hanzo run -m /path/to/model
```

### Running GGUF Models

```bash
hanzo run --format gguf -m author/model-repo -f model-quant.gguf
```

Specify tokenizer if needed:
```bash
hanzo run --format gguf -m author/model-repo -f file.gguf -t author/official-tokenizer
```

## Next Steps

- [CLI Reference](CLI.md): All commands and options
- [HTTP API](HTTP.md): Run as an OpenAI-compatible server
- [Python SDK](PYTHON_SDK.md): Python package documentation
- [Troubleshooting](TROUBLESHOOTING.md): Common issues and solutions
