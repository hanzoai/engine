#!/usr/bin/env bash
# Build the Hanzo engine FFI cdylib with the right native backend for THIS host.
# One script, every platform: macOS (Metal/Accelerate), Linux (CUDA/ROCm/CPU).
# iOS: ENGINE_FFI_TARGET=aarch64-apple-ios (device) or aarch64-apple-ios-sim (simulator) builds the Metal
# staticlib an app links (apps/mobile/modules/hanzo-engine/build.sh wraps both into an xcframework).
# CI (arcd, one job per <org>-<os>-<arch>) runs this on each platform; locally it
# auto-picks. Override with ENGINE_FFI_FEATURES.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> hanzo/engine (workspace root)

OS="$(uname -s)"
ARCH="$(uname -m)"
FEATURES="${ENGINE_FFI_FEATURES:-${HANZO_FFI_FEATURES:-}}"

if [ -z "$FEATURES" ]; then
  case "$OS" in
    Darwin)
      # Metal needs the Xcode Metal Toolchain; Accelerate (CPU BLAS) always works.
      # Use metal when available, else accelerate. Override: ENGINE_FFI_FEATURES=metal,accelerate
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

TARGET="${ENGINE_FFI_TARGET:-${HANZO_FFI_TARGET:-}}"
case "$TARGET" in
  aarch64-apple-ios|aarch64-apple-ios-sim)
    SDK=iphoneos; [ "$TARGET" = aarch64-apple-ios-sim ] && SDK=iphonesimulator
    export SDKROOT="$(xcrun --sdk $SDK --show-sdk-path)" IPHONEOS_DEPLOYMENT_TARGET=18.0
    FEATURES="${ENGINE_FFI_FEATURES:-${HANZO_FFI_FEATURES:-metal}}"
    ;;
  "") ;;
  *) echo "unsupported ENGINE_FFI_TARGET: $TARGET" >&2; exit 2 ;;
esac

echo "==> hanzo-engine-ffi  os=$OS arch=$ARCH target=${TARGET:-host} features=[${FEATURES:-cpu}]"
cargo build --release -p hanzo-engine-ffi ${FEATURES:+--features "$FEATURES"} ${TARGET:+--target "$TARGET"}

echo "==> artifacts:"
ls -lh "${CARGO_TARGET_DIR:-target}/${TARGET}/release/"libhanzo_engine_ffi.* 2>/dev/null || true
