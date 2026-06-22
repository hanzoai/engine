#!/usr/bin/env bash
# Build the Hanzo engine FFI cdylib with the right native backend for THIS host.
# One script, every platform: macOS (Metal/Accelerate), Linux (CUDA/ROCm/CPU).
# CI (arcd, one job per <org>-<os>-<arch>) runs this on each platform; locally it
# auto-picks. Override with HANZO_FFI_FEATURES.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> hanzo/engine (workspace root)

OS="$(uname -s)"
ARCH="$(uname -m)"
FEATURES="${HANZO_FFI_FEATURES:-}"

if [ -z "$FEATURES" ]; then
  case "$OS" in
    Darwin)
      # Metal needs the Xcode Metal Toolchain; Accelerate (CPU BLAS) always works.
      # Use metal when available, else accelerate. Override: HANZO_FFI_FEATURES=metal,accelerate
      if xcrun metal -v >/dev/null 2>&1; then FEATURES="metal,accelerate"; else FEATURES="accelerate"; fi
      export SDKROOT="$(xcrun --show-sdk-path)"; export CPATH="$SDKROOT/usr/include"
      ;;
    Linux)
      if command -v nvidia-smi >/dev/null 2>&1; then FEATURES="cuda";
      elif command -v rocminfo >/dev/null 2>&1; then FEATURES="rocm";
      else FEATURES=""; fi   # portable CPU build
      ;;
    *) FEATURES="" ;;
  esac
fi

echo "==> hanzo-engine-ffi  os=$OS arch=$ARCH features=[${FEATURES:-cpu}]"
if [ -n "$FEATURES" ]; then
  cargo build --release -p hanzo-engine-ffi --features "$FEATURES"
else
  cargo build --release -p hanzo-engine-ffi
fi

echo "==> artifacts:"
ls -lh target/release/libhanzo_engine_ffi.* 2>/dev/null || true
