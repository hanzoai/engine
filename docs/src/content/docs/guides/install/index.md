---
title: Install
description: Platform-specific install steps and source builds.
---

The install script in [Tutorial 1](/hanzo/tutorials/01-install-and-run/) works on Linux, macOS, and Windows, detects the accelerator, and selects the matching feature flags. For manual installs, specific driver versions, or source builds, use the guides below.

## Install options

| Situation | Guide |
|---|---|
| Linux with an NVIDIA GPU | [Linux with CUDA](/hanzo/guides/install/linux-cuda/) |
| Apple Silicon Mac | [macOS with Metal](/hanzo/guides/install/macos-metal/) |
| Windows (native or WSL) | [Windows](/hanzo/guides/install/windows/) |
| Build from source | [Build from source](/hanzo/guides/install/from-source/) |

The [cargo features reference](/hanzo/reference/cargo-features/) maps GPU generations to feature flags. For containerised or production deployment, see the [Deploy guides](/hanzo/guides/deploy/).

Video input uses FFmpeg at runtime. The install commands and runtime checks are in [Set up video input](/hanzo/guides/models/video-setup/).
