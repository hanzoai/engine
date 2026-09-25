#!/usr/bin/env bash
# Q4 end to end: Qwen3.8-Flash-Next (fp8hybrid snapshot) served by hanzo with
# --fp8-block-gemm-backend auto (the default) and scalar, no speculation.
#
#   scripts/blockwise_fp8_e2e.sh --out DIR [--plan]
#
# It needs the box to itself (~75 GiB resident): the user stops hanzo-vllm first and restarts it
# afterwards. This script never touches that unit, and refuses to run while :18300 listens.
#
# Per backend: serve, then
#   - W0's harness (scripts/bench_http.py): c=1 decode of 512 tokens x5, warm prefill at ~8K and ~32K;
#   - nsys over 20 decode steps: block-FP8 kernel µs per step;
#   - greedy output identical across 3 runs;
#   - the startup log's resolution line.
# Exits 1 unless: auto logs exactly one "fp8-block-gemm-backend auto -> cutlass" line and no
# fallback WARN; cutlass block-FP8 GEMM µs per decode step <= 1.05 x the vLLM M=1 sum
# (VLLM_M1_SUM_US, from T6's minima over the model's GEMM shapes); auto decode beats scalar;
# greedy output is identical across 3 runs per backend.
#
# Inputs: HANZO_ENGINE_BIN (default <repo>/target/release/hanzo-engine), PORT (default 30000),
# SNAPSHOT (default the fp8hybrid snapshot below), VLLM_M1_SUM_US (required).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
BIN=${HANZO_ENGINE_BIN:-$REPO/target/release/hanzo-engine}
PORT=${PORT:-30000}
SNAPSHOT=${SNAPSHOT:-$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47-fp8hybrid}
URL=http://127.0.0.1:$PORT/v1/chat/completions
MIN_GIB=100

OUT=""
PLAN=0
while [ $# -gt 0 ]; do
  case $1 in
    --out) OUT=$2; shift 2 ;;
    --plan) PLAN=1; shift ;;
    *) echo "usage: blockwise_fp8_e2e.sh --out DIR [--plan]" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "usage: blockwise_fp8_e2e.sh --out DIR [--plan]" >&2; exit 2; }

serve_cmd() { # serve_cmd <backend> [extra...]
  local backend=$1
  shift
  printf '%s\n' "$BIN" serve -m "$SNAPSHOT" --port "$PORT" --fp8-block-gemm-backend "$backend" "$@"
}

if [ "$PLAN" = 1 ]; then
  for b in auto scalar; do
    mapfile -t argv < <(serve_cmd "$b")
    printf '%q ' "${argv[@]}"
    printf '\n'
  done
  exit 0
fi

# Preflight: every refusal happens before anything starts.
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ':18300$'; then
  echo "e2e: :18300 is listening (hanzo-vllm); a full model needs it stopped" >&2
  exit 75
fi
mem=$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
[ "$mem" -ge "$MIN_GIB" ] || { echo "e2e: MemAvailable $mem GiB < $MIN_GIB GiB" >&2; exit 75; }
[ -x "$BIN" ] || { echo "e2e: no binary at $BIN" >&2; exit 1; }
[ -d "$SNAPSHOT" ] || { echo "e2e: no snapshot at $SNAPSHOT" >&2; exit 1; }
[ -f "$HERE/bench_http.py" ] || { echo "e2e: scripts/bench_http.py (W0's harness) is missing" >&2; exit 1; }
[ -n "${VLLM_M1_SUM_US:-}" ] || { echo "e2e: set VLLM_M1_SUM_US from T6's vLLM minima" >&2; exit 2; }
command -v nsys >/dev/null || { echo "e2e: nsys not on PATH" >&2; exit 1; }
mkdir -p "$OUT"

PID=""
stop() { [ -z "$PID" ] || { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; PID=""; }; }
trap stop EXIT

wait_ready() {
  for _ in $(seq 1 900); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && return 0
    kill -0 "$PID" 2>/dev/null || { echo "e2e: server exited during load" >&2; return 1; }
    sleep 2
  done
  echo "e2e: server not ready after 30 min" >&2
  return 1
}

greedy() { # greedy <file>: one deterministic completion
  curl -sf "$URL" -H 'content-type: application/json' -d '{
    "model": "default", "temperature": 0, "max_tokens": 256, "seed": 0,
    "messages": [{"role": "user", "content": "List the first twelve prime numbers and explain the sieve of Eratosthenes."}]
  }' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' >"$1"
}

fail=0
declare -A DECODE
for backend in auto scalar; do
  dir=$OUT/$backend
  mkdir -p "$dir"
  mapfile -t argv < <(serve_cmd "$backend")
  "${argv[@]}" >"$dir/serve.log" 2>&1 &
  PID=$!
  wait_ready

  for i in 1 2 3; do greedy "$dir/greedy.$i.txt"; done
  if ! cmp -s "$dir/greedy.1.txt" "$dir/greedy.2.txt" || ! cmp -s "$dir/greedy.1.txt" "$dir/greedy.3.txt"; then
    echo "FAIL $backend: greedy output differs across 3 runs" >&2
    fail=1
  fi

  for i in 1 2 3 4 5; do
    python3 "$HERE/bench_http.py" "$URL" --gen 512 --sizes 4 --label "$backend-decode-$i" --json "$dir/decode.jsonl"
  done
  # ~8K and ~32K prompt tokens; the second pass is the warm one.
  for pass in cold warm; do
    python3 "$HERE/bench_http.py" "$URL" --gen 16 --sizes 260,1030 --label "$backend-prefill-$pass" --json "$dir/prefill.jsonl"
  done
  DECODE[$backend]=$(python3 -c '
import json, sys
rates = [json.loads(l).get("decode_tps") for l in open(sys.argv[1])]
rates = [r for r in rates if r]
print(max(rates) if rates else 0)' "$dir/decode.jsonl")
  stop

  if [ "$backend" = auto ]; then
    lines=$(grep -c 'fp8-block-gemm-backend auto -> cutlass' "$dir/serve.log" || true)
    warns=$(grep -c 'cutlass lib not built for sm_121' "$dir/serve.log" || true)
    if [ "$lines" != 1 ] || [ "$warns" != 0 ]; then
      echo "FAIL auto: $lines 'auto -> cutlass' lines, $warns fallback WARNs" >&2
      fail=1
    fi
  fi

  # 20 decode steps under nsys: block-FP8 GEMM µs per step.
  mapfile -t argv < <(serve_cmd "$backend" --profiler cuda --profiler-max-iterations 20)
  nsys profile --trace=cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o "$dir/decode" "${argv[@]}" >"$dir/nsys.log" 2>&1 &
  PID=$!
  wait_ready
  python3 "$HERE/bench_http.py" "$URL" --gen 64 --sizes 4 --label "$backend-nsys" >/dev/null
  stop
  nsys stats --report cuda_gpu_kern_sum --format csv -o "$dir/kern" "$dir/decode.nsys-rep" >/dev/null
  per_step=$(python3 -c '
import csv, sys
total = 0.0
for row in csv.DictReader(open(sys.argv[1])):
    name = row["Name"]
    if "hanzo_blockwise_fp8" in name or "Sm120Blockwise" in name or "w8a8" in name or "BlockwiseScaling" in name:
        total += float(row["Total Time (ns)"])
print(total / 1e3 / 20)' "$dir/kern_cuda_gpu_kern_sum.csv")
  echo "$backend: block-FP8 GEMM $per_step us per decode step"
  if [ "$backend" = auto ] && python3 -c "import sys; sys.exit(0 if $per_step <= 1.05 * $VLLM_M1_SUM_US else 1)"; then :; elif [ "$backend" = auto ]; then
    echo "FAIL auto: $per_step us per step > 1.05 x vLLM $VLLM_M1_SUM_US us" >&2
    fail=1
  fi
done

echo "decode tok/s: auto ${DECODE[auto]}, scalar ${DECODE[scalar]}"
if ! python3 -c "import sys; sys.exit(0 if ${DECODE[auto]} > ${DECODE[scalar]} else 1)"; then
  echo "FAIL: auto decode is not faster than scalar" >&2
  fail=1
fi
exit $fail
