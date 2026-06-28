# Release Process

This document describes how Hanzo Engine is versioned, gated, and shipped per platform.

## Repository layout (load-bearing)

The release pipeline assumes three sibling checkouts:

```
<root>/
  engine/   this repo (hanzoai/engine)
  ml/       hanzoai/ml      (provides hanzo-ml / hanzo-nn / flash-attn / metal-kernels via PATH deps)
  node/     hanzoai/node    (consumes engine; checked out by release.yml)
```

`engine/Cargo.toml` consumes ml through path deps (`hanzo-ml = { path = "../ml/hanzo-ml" }`, etc.).
The crates.io versions are present but commented out ("for reference, not used"). `cargo check`,
`cargo build`, and the release workflow all require `ml/` to exist as a sibling of `engine/`.
Release order is **ml -> engine -> node** (ml is the leaf, node depends on engine).

## Versioning and the tag scheme

- All workspace crates share one version via `[workspace.package] version` in `Cargo.toml`; every
  member inherits it with `version.workspace = true`. The inter-crate path deps in
  `[workspace.dependencies]` also pin that same version, so the version lives in exactly two places
  in the root `Cargo.toml` (the package version and the `path = ... , version = ...` dep lines) and
  in `hanzo-pyo3/pyproject.toml` (the Python wheel version). Bump all of them together.
- Tags are `v<semver>`, e.g. `v1.0.2`. `release.yml` fires on any `v*` tag.
- **Tags must be monotonic and must not regress below an already-cut release.** `v1.0.0` and `v1.0.1`
  are already ancestors of `main`, so the next tag must be **>= v1.0.2**. The current workspace
  version is `1.0.2`; tag it `v1.0.2`.
- The Docker-build workflows (`build_cpu.yaml`, `build_cuda.yaml`, `build_rocm.yaml`) trigger on tags
  matching `**[0-9]+.[0-9]+.[0-9]+*` as well as on GitHub `release: published`.

To cut a release:

1. Set the version in `Cargo.toml` (`[workspace.package] version` + the inter-crate dep pins) and in
   `hanzo-pyo3/pyproject.toml`. Keep all of them equal.
2. `cargo check` (with the `ml/` sibling present) so `Cargo.lock` is refreshed; commit the lockfile.
3. Merge to `main`. Confirm CI (`ci.yml`) is green on `main` (see below).
4. `git tag vX.Y.Z && git push origin vX.Y.Z`. This drives `release.yml` (binaries + GitHub release +
   Docker) and the `build_cpu.yaml` / `build_cuda.yaml` / `build_rocm.yaml` image builds.
5. For the Homebrew cask, run `brew_release.yml` (`workflow_dispatch`) with the new tag.

## CI gate (`ci.yml`)

Triggers on push and PR to `main` (plus a weekly cron and manual dispatch). Jobs:

- **Check** - matrix over `ubuntu-latest`, `windows-latest`, `macOS-latest`.
- **Check (metal)** - `cargo check --workspace --features metal` on macOS.
- **Test Suite** - matrix over the same three OSes.
- **Rustfmt**, **Clippy** (`-D warnings`), **Docs**, **Typos**, **Doc Links**.
- **MSRV Check** - pinned to Rust `1.90.0`.

Note: the MSRV is declared in two places that currently disagree - `Cargo.toml` says
`rust-version = "1.88"`, while the `ci.yml` MSRV job pins `1.90.0`. Reconcile to a single number
before relying on the MSRV gate.

`ci.yml` is the primary correctness gate; it must be green on `main` before any tag is trusted.

## Per-platform release matrix (`release.yml`)

`release.yml` runs `build-and-test` (check/test/fmt/clippy on `hanzo-engine --no-default-features`),
then a 5-target binary matrix, then `create-release` and `publish-docker`. It checks out `engine/`,
`hanzoai/ml`, and `hanzoai/node` as siblings. Every target builds `--package hanzo-engine
--no-default-features` (plus the per-target features below), renames the binary to `hanzoai`,
and packages it (`.tar.gz` on unix, `.zip` on Windows).

| Target triple | Runner | Features | Artifact | Notes |
|---|---|---|---|---|
| `x86_64-unknown-linux-gnu` | `hanzo-build-linux-amd64` | (none) | `hanzoai-linux-amd64.tar.gz` | |
| `aarch64-unknown-linux-gnu` | `hanzo-build-linux-amd64` | `vendored-openssl` | `hanzoai-linux-arm64.tar.gz` | cross-compiled (gcc-aarch64), `continue-on-error` |
| `x86_64-apple-darwin` | `macos-latest` | `metal` | `hanzoai-macos-amd64.tar.gz` | |
| `aarch64-apple-darwin` | `macos-14` | `metal` | `hanzoai-macos-arm64.tar.gz` | Apple Silicon |
| `x86_64-pc-windows-msvc` | `windows-latest` | (none) | `hanzoai-windows-amd64.exe.zip` | |

After the matrix:

- **create-release** - downloads all artifacts, generates a changelog from git log since the last
  tag, and publishes a non-draft GitHub Release with the binaries attached.
- **publish-docker** - calls the reusable `hanzoai/.github/.github/workflows/docker-build.yml` to push
  `ghcr.io/hanzoai/engine`.

### Other build/publish workflows

- **`build_cpu.yaml`** - CPU Docker image (`linux/amd64`) on release/tag.
- **`build_cuda.yaml`** - CUDA Docker images for compute capabilities `80, 86, 89, 90, 120, 121`
  (Blackwell `sm_120`/`sm_121` included) on release/tag (tagged `cuda-<cc>-<version>`).
- **`build_rocm.yaml`** - ROCm Docker images for AMD `gfx942`/`gfx90a` on release/tag (tagged
  `rocm-<gfx>-<version>`); `gfx1151` builds natively or via a ROCm 7.x base override.
- **`brew_release.yml`** - `workflow_dispatch` only; builds `-p hanzo-cli --features metal` on
  `macos-15` and updates the Homebrew cask for the given tag.
- **`ci_cuda.yaml`** - CUDA build/test on a self-hosted `[self-hosted, Linux, ARM64, gpu, cuda]`
  runner (expects `/usr/local/cuda-13.0`); PR + manual dispatch only.

## Feature flags by platform

Features are fanned out consistently across `hanzo-engine`, `hanzo-server`, `hanzo-cli`, `hanzo`, and
`hanzo-bench`:

- **CPU** - default; no feature needed.
- **Metal** (Apple) - `--features metal`. Covered by `ci.yml` (check-metal) and `release.yml` macOS
  targets. Requires a macOS runner.
- **CUDA** (NVIDIA) - `--features "cuda flash-attn cudnn"` (also `nccl`, `flash-attn-v3`). Requires
  `nvcc` / a CUDA GPU runner (`ci_cuda.yaml`). Not built by the default `release.yml` matrix.
- **Vulkan** - `--features vulkan` (fans out to `hanzo-ml/vulkan`, `hanzo-nn/vulkan`,
  `hanzo-quant/vulkan`). Needs `glslc` on the runner. Confirmed to compile locally; not yet in any CI
  job - add `cargo check -p hanzo-engine --features vulkan` to gate it.
- **ROCm** (AMD) - `--features rocm`. Needs `/opt/rocm`. Confirmed to compile in ml; not yet in any
  engine CI job - add `cargo check -p hanzo-engine --features rocm` to gate it.

## Local verification before tagging

```bash
# with the ml/ sibling checked out next to engine/
cargo check                                   # default features
cargo check -p hanzo-engine --features vulkan # if glslc is installed
cargo fmt --all -- --check
cargo clippy --workspace --tests --examples -- -D warnings
cargo test -p hanzo-engine -p hanzo-quant -p hanzo-vision
```

CUDA and Metal cannot be verified without the respective toolchain/runner; rely on `ci_cuda.yaml` and
the macOS CI/release jobs for those.
