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
  --fail-fast) FAILFAST=1; shift;;
  *) ARGS+=("$1"); shift;;
esac; done
: "${BACKEND:?--backend required}" "${MODELS:?--models required (tag=path,...)}"
FAILFAST="${FAILFAST:-0}"

# Failure policy is EXPLICIT: default is continue-on-error -- a long matrix must not lose the
# models it already measured to one bad GGUF; each failure (missing file OR a nonzero dossier
# run) is recorded and skipped. --fail-fast makes any failure abort. Both paths are uniform.
RUNS=(); FAILED=()
IFS=',' read -ra PAIRS <<< "$MODELS"
for pair in "${PAIRS[@]}"; do
  tag="${pair%%=*}"; path="${pair#*=}"
  fail() { echo "$1" >&2; FAILED+=("$2"); if [[ "$FAILFAST" == 1 ]]; then echo "fail-fast: aborting" >&2; exit 4; fi; return 0; }
  if [[ ! -f "$path" ]]; then fail "SKIP $tag: no file $path" "$tag:missing"; continue; fi
  echo "=== battery: $BACKEND / $tag ===" >&2
  out=""; rc=0
  out="$(bash "$(dirname "$0")/dossier.sh" "${ARGS[@]}" --model "$path" --model-tag "$tag")" || rc=$?
  d="$(printf '%s\n' "$out" | tail -1)"          # dossier prints the run dir as its last line
  if [[ "$rc" -ne 0 || ! -d "$d" ]]; then fail "FAILED $tag (rc=$rc)" "$tag:rc$rc"; continue; fi
  RUNS+=("$d")
  python3 "$(dirname "$0")/bench_stats.py" "$d" >/dev/null 2>&1 || echo "warn: stats failed for $tag" >&2
done

# Empty-RUNS guard (set -u would otherwise trip on ${RUNS[0]}).
if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "no successful model runs (failed/skipped: ${FAILED[*]:-none})" >&2; exit 1
fi
echo "" >&2
echo "=== combined board: $BACKEND ($(hostname -s)) ==="
head -2 "${RUNS[0]}/board.md" 2>/dev/null || true
for d in "${RUNS[@]}"; do tail -n +3 "$d/board.md" 2>/dev/null || true; done
[[ ${#FAILED[@]} -gt 0 ]] && echo "failed/skipped: ${FAILED[*]}" >&2 || true
echo ""
printf '%s\n' "${RUNS[@]}"
