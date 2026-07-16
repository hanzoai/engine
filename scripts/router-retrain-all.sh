#!/usr/bin/env bash
# The nightly composition the scheduler fires: fit the shared base heads on opted-in
# rows, then per-org heads for each opted-in org (the gate holds any org under
# MIN_ROWS). One script, scope as a flag -- this only sequences scopes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"

# Build the native fit fresh so it tracks the enso crate (override FIT_BIN to use
# enso's own `fit` once the learning-loop agent lands it -- identical flags).
cargo build --release -p hanzo-router-retrain --manifest-path "$ROOT/Cargo.toml" >&2

"$SCRIPT_DIR/router-retrain.sh" --scope global

if [[ -n "${CONTRIB_ORGS:-}" ]]; then
  IFS=',' read -r -a ORGS <<< "$CONTRIB_ORGS"
  for o in "${ORGS[@]}"; do
    o="$(echo "$o" | xargs)"; [[ -n "$o" ]] || continue
    "$SCRIPT_DIR/router-retrain.sh" --scope "org=$o"
  done
fi
