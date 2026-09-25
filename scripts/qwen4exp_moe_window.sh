#!/usr/bin/env bash
# W8c full-model checks for the Flash-Next MoE router (E2) and shared-expert overlap (E3).
#
# Run ONLY in a window the user schedules with hanzo-vllm stopped: every step loads the 75 GB
# model. This script never stops, restarts or signals a server; it refuses to start while
# anything answers on :18300.
#
#   1. Teacher-forced routing (bar: 100% bitwise). vLLM runs eager with a hook on every
#      layer's mlp.gate (qwen4exp_moe_dump.py) at c=1, 3, 4; hanzo replays the dumped x through
#      Lane::linear + route::topk at the same M (lane::tests::teacher_forced).
#   2. hanzo 2x2 A/B: --moe-router fused|split x --shared-expert-overlap 256|0. For each: decode
#      tok/s at c=1, 3, 4 (bench_http.py), three greedy runs checked identical, and an nsys trace
#      of one decode step. Bars: overlap >= serial, fused >= split, runs identical, and the trace
#      shows shared-expert kernels overlapping routed-expert kernels (checked by eye in nsys).
#   3. vLLM baseline without speculation, max-num-seqs 8 (W8's bar config) on the same prompts.
#
# Usage: scripts/qwen4exp_moe_window.sh <hanzo binary>
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=$root/target/qwen4exp_moe_window
snap=/home/z/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47-fp8hybrid
py=/home/z/vllm-env/bin/python
hanzo=${1:?usage: $0 <hanzo binary>}
port=18399
mkdir -p "$out"

if curl -s -m 2 -o /dev/null http://127.0.0.1:18300/v1/models; then
  echo "hanzo-vllm is serving on :18300; this needs the window with it stopped" >&2
  exit 1
fi
for flag in --moe-router --shared-expert-overlap; do
  if ! "$hanzo" serve --help | grep -q -- "$flag"; then
    echo "$hanzo has no $flag (W8c-9 lands after W0's flag surface)" >&2
    exit 1
  fi
done
[ -f "$root/scripts/bench_http.py" ] || { echo "scripts/bench_http.py (W0) is missing" >&2; exit 1; }

echo "== 1. teacher-forced routing"
VLLM_ALLOW_INSECURE_SERIALIZATION=1 nice -n19 "$py" "$root/scripts/qwen4exp_moe_dump.py"
/data/engine-build.sh "$root" cargo test -p hanzo-engine --features cuda --lib \
  cuda::lane::tests::teacher_forced -- --ignored --nocapture --test-threads=1

wait_up() {
  for _ in $(seq 1 600); do
    curl -s -m 2 -o /dev/null "http://127.0.0.1:$1/v1/models" && return 0
    sleep 2
  done
  echo "server on :$1 never came up" >&2
  return 1
}

bench() { # label port
  for c in 1 3 4; do
    "$py" "$root/scripts/bench_http.py" "http://127.0.0.1:$2/v1/chat/completions" --label "$1" \
      --sizes 4,40 --gen 256 --concurrency "$c" --json "$out/bench.jsonl"
  done
}

greedy() { # label port
  for run in 1 2 3; do
    curl -s "http://127.0.0.1:$2/v1/chat/completions" -H 'content-type: application/json' \
      -d '{"model":"default","messages":[{"role":"user","content":"Count from one to forty in words."}],"max_tokens":200,"temperature":0}' \
      | "$py" -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' \
      > "$out/greedy_$1_$run.txt"
  done
  if cmp -s "$out/greedy_$1_1.txt" "$out/greedy_$1_2.txt" && cmp -s "$out/greedy_$1_1.txt" "$out/greedy_$1_3.txt"; then
    echo "$1: 3 greedy runs identical"
  else
    echo "$1: greedy runs DIFFER"
  fi
}

echo "== 2. hanzo A/B"
for router in fused split; do
  for overlap in 256 0; do
    label="router=$router,overlap=$overlap"
    "$hanzo" serve --port "$port" --moe-router "$router" --shared-expert-overlap "$overlap" \
      --max-seqs 8 -m "$snap" > "$out/serve_${router}_${overlap}.log" 2>&1 &
    pid=$!
    wait_up "$port"
    bench "$label" "$port"
    greedy "${router}_${overlap}" "$port"
    kill "$pid"; wait "$pid" || true
  done
done
# One decode step under nsys, fused router with overlap on.
nsys profile -o "$out/decode_overlap" --force-overwrite true --trace cuda,nvtx --duration 60 \
  "$hanzo" serve --port "$port" --moe-router fused --shared-expert-overlap 256 --max-seqs 8 -m "$snap" \
  > "$out/serve_nsys.log" 2>&1 &
pid=$!
wait_up "$port"
"$py" "$root/scripts/bench_http.py" "http://127.0.0.1:$port/v1/chat/completions" --sizes 4 --gen 16 --concurrency 1
kill -INT "$pid"; wait "$pid" || true

echo "== 3. vLLM baseline (no speculation, max-num-seqs 8)"
/home/z/vllm-env/bin/vllm serve "$snap" --port "$port" --max-num-seqs 8 --gpu-memory-utilization 0.78 \
  > "$out/vllm_baseline.log" 2>&1 &
pid=$!
wait_up "$port"
bench "vllm" "$port"
greedy vllm "$port"
kill "$pid"; wait "$pid" || true

echo "== results in $out (bench.jsonl, greedy_*.txt, decode_overlap.nsys-rep)"
