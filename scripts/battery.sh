#!/usr/bin/env bash
# battery.sh — the model-matrix battery: run every model in a set through scripts/dossier.sh
# for ONE backend on ONE box, then summarise. Each model is a separate dossier run (its own
# pinned manifest, raw per-rep JSON, quiet-gate log, derived board), so concerns stay split;
# this file only iterates and folds the per-model boards into one table.
#
#   battery.sh --backend rocm|vulkan|metal|cuda \
#     --hanzo-bench PATH --llama-bench PATH [--llama-dir DIR --engine-dir DIR] \
#     [-p 512,500,2048,4096 -n 128 -r 7 --concurrency 1] \
#     --models "tag1=/path/one.gguf,tag2=/path/two.gguf,..."
#
# Decode is greedy-vs-greedy by default (dossier.sh -> hanzo-bench greedy); sampling parity is
# recorded per run. Intended to run ONCE on the new engine cut, not to re-measure old versions.
set -euo pipefail
ARGS=(); BACKEND=""; MODELS=""
while [[ $# -gt 0 ]]; do case "$1" in
  --models) MODELS="$2"; shift 2;;
  --backend) BACKEND="$2"; ARGS+=(--backend "$2"); shift 2;;
  *) ARGS+=("$1"); shift;;
esac; done
: "${BACKEND:?--backend required}" "${MODELS:?--models required (tag=path,...)}"

RUNS=()
IFS=',' read -ra PAIRS <<< "$MODELS"
for pair in "${PAIRS[@]}"; do
  tag="${pair%%=*}"; path="${pair#*=}"
  [[ -f "$path" ]] || { echo "SKIP $tag: no file $path" >&2; continue; }
  echo "=== battery: $BACKEND / $tag ===" >&2
  # dossier.sh prints the run dir as its last stdout line
  d="$(bash "$(dirname "$0")/dossier.sh" "${ARGS[@]}" --model "$path" --model-tag "$tag" | tail -1)"
  [[ -d "$d" ]] && RUNS+=("$d") && python3 "$(dirname "$0")/bench_stats.py" "$d" >/dev/null 2>&1 || true
done

echo "" >&2
echo "=== combined board: $BACKEND ($(hostname -s)) ==="
head -2 "${RUNS[0]}/board.md" 2>/dev/null
for d in "${RUNS[@]}"; do tail -n +3 "$d/board.md" 2>/dev/null; done
echo ""
printf '%s\n' "${RUNS[@]}"
