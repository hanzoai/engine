---
title: Run hanzo in Docker
description: Build and run the published container images, with and without CUDA.
sidebar:
  order: 1
---

The hanzo repository ships several Dockerfiles for common deployment targets.

- `Dockerfile`: default. Multi-stage build producing a Debian-based CPU-only image with the server binary.
- `Dockerfile.cuda`: CUDA variant for NVIDIA GPUs with flash attention (multi-arch, Blackwell `sm_120`/`sm_121` included).
- `Dockerfile.manylinux`: for producing the Python wheels published to PyPI.

Apple Silicon (Metal) is not containerizable -- macOS gives containers no GPU access -- so the Metal build ships as a native Homebrew binary (`brew install hanzoai/hanzo/hanzo`), not an image. AMD (ROCm/HIP) and Vulkan build from source today: the AMD APU (`gfx1151`) path runs natively rather than in a container.

## Building an image

From a repository checkout:

```bash
# CPU
docker build -t hanzo:latest -f Dockerfile .

# CUDA
docker build -t hanzo:cuda -f Dockerfile.cuda .
```

The CUDA build is slower the first time because flash-attention compilation takes a while. Subsequent builds use the Docker layer cache.

## Running the CPU image

```bash
docker run --rm -it \
  -p 1234:80 \
  -v $HOME/.cache/huggingface:/data \
  hanzo:latest \
  hanzo-server -m Qwen/Qwen3-4B
```

Notes:

The image binds port 80 internally by default, controlled by the `PORT` environment variable. `-p 1234:80` publishes it as 1234 on the host, matching the standard hanzo default.

The container's Hugging Face cache directory is `/data`. Mounting the host cache there avoids re-downloading weights on first run.

## Running the CUDA image

```bash
docker run --rm -it --gpus all \
  -p 1234:80 \
  -v $HOME/.cache/huggingface:/data \
  hanzo:cuda \
  hanzo-server -m Qwen/Qwen3-4B
```

`--gpus all` exposes all detected NVIDIA GPUs. To pin a specific GPU: `--gpus '"device=0"'`. Without the flag, the CUDA image falls back to CPU inference.

The host requires the NVIDIA Container Toolkit. See [NVIDIA's documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Production deployment notes

**Persist the cache.** Hugging Face weights are large enough that re-downloading on every container restart is wasteful. Mount a persistent volume at `/data`.

**Pin model versions.** `-m Qwen/Qwen3-4B` resolves to whatever revision is tagged `main` at download time. For reproducible deployments, pin to a specific Hugging Face revision (the Rust SDK's `ModelBuilder::with_hf_revision` accepts a revision string; the CLI does not currently expose this flag).

**Health check.** `/health` returns 200 when the server is up. Add a Docker healthcheck:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD curl -fsS http://localhost:80/health || exit 1
```

The generous `--start-period` matters, first-run model loading can take minutes.

**Resource limits.** Set `--memory` and `--gpus` on `docker run` to bound the container's resources.

**Video input.** Install FFmpeg inside the image when serving video-capable models. See [Set up video input](/hanzo/guides/models/video-setup/) for the Docker snippet and runtime check.

## Kubernetes

The pieces above translate directly:

- Use a Deployment with a readiness probe hitting `/health`.
- Mount a PersistentVolumeClaim at `/data` for the Hugging Face cache.
- Use the NVIDIA device plugin and a `nvidia.com/gpu` resource request for CUDA.
- Use an initContainer to pre-download weights for fast pod startup.

There is no official Helm chart. Contributions welcome.

## See also

- [Production checklist](/hanzo/guides/deploy/production-checklist/): operational concerns regardless of container layer.
- [HTTP server guide](/hanzo/guides/serve/http-server/): config options.
