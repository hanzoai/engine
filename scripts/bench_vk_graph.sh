#!/usr/bin/env bash
# bench_vk_graph — the Vulkan decode command-graph gate: correctness + perf, reproducibly.
#
# Runs hanzo-server twice on one GGUF — graph default-ON (env unset) and forced eager
# (VK_GRAPHS=0) — over a fixed greedy prompt matrix, then enforces:
#   1. IDENTITY: every completion byte-identical ON vs OFF (greedy, temp 0). The graph's
#      failure mode is fluent-but-stale output, so this is the correctness gate.
#   2. NO REGRESSION: no sustained prompt may lose more than REGRESS_PCT vs eager. The graph
#      removes CPU record/submit overhead, so a memory-bound generation (wide MoE-expert
#      spread) is neutral rather than faster, but must never be slower.
#   3. SUSTAINED WIN: the MEAN sustained-prompt speedup must clear MIN_WIN_PCT — the graph
#      demonstrably helps CPU-exposed decode, averaged over the matrix, without demanding
#      every prompt clear it (the most memory-bound one legitimately won't).
#   4. AUTOSHAPE: a short-generation-only workload must produce ZERO captures — the self-
#      shaping capture policy learns the generations are too short to repay a graph and keeps
#      them on the eager path. This is the structural proof that short workloads never regress
#      (a 24-token timing is startup-dominated noise; identity + zero-capture is the real bar).
# Exit is nonzero on any gate failure — wire it into CI on a Vulkan GPU runner and perf
# is tracked per commit instead of living in comments. JSON results land in the out dir
# (bench_vk_graph.json) for trend dashboards.
#
#   bench_vk_graph.sh <model.gguf> [-o out-dir] [--server path] [--port port]
#                     [--gen N] [--min-win pct] [--max-short-loss pct]
set -uo pipefail

MODEL=""
OUT=""
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${HANZO_SERVER:-$HERE/../target/release/hanzo-server}"
PORT=1250
GEN=200
MIN_WIN_PCT=5
REGRESS_PCT=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUT="$2"; shift 2 ;;
    --server) SERVER="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gen) GEN="$2"; shift 2 ;;
    --min-win) MIN_WIN_PCT="$2"; shift 2 ;;
    --max-short-loss) REGRESS_PCT="$2"; shift 2 ;;
    -*) echo "unknown arg: $1" >&2; exit 2 ;;
    *) MODEL="$1"; shift ;;
  esac
done
[[ -n "$MODEL" && -f "$MODEL" ]] || { echo "usage: bench_vk_graph.sh <model.gguf> [flags]" >&2; exit 2; }
OUT="${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/bench_vk_graph.XXXXXX")}"
mkdir -p "$OUT"
echo "out-dir: $OUT" >&2

# name|prompt|max_tokens — long prompts amortize a capture, `short` must stay pure eager.
PROMPTS=(
  "prose|Write a detailed paragraph about the ocean.|$GEN"
  "numbers|Count from 1 to 40, writing each number in words, one per line.|$GEN"
  "code|Write a Rust function that parses an IPv4 address string into [u8;4] with error handling.|$GEN"
  "short|Reply with exactly: The quick brown fox jumps over the lazy dog.|24"
)

serve() { # mode: on|off
  local mode=$1 log=$OUT/server_$1.log
  fuser -k "${PORT}/tcp" 2>/dev/null; sleep 1
  local -a env_prefix=()
  [[ "$mode" == off ]] && env_prefix=(env VK_GRAPHS=0)
  "${env_prefix[@]}" "$SERVER" --port "$PORT" gguf \
    -m "$(dirname "$MODEL")" -f "$(basename "$MODEL")" >"$log" 2>&1 &
  local ready=0
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    pgrep -f "hanzo-server.*$(basename "$MODEL")" >/dev/null || break
    sleep 1
  done
  [[ $ready == 1 ]] || { echo "FATAL: server ($mode) never became ready" >&2; tail -5 "$log" >&2; exit 1; }
}

ask() { # name prompt max_tokens mode -> writes .txt/.json, prints t/s
  local name=$1 prompt=$2 max=$3 mode=$4
  curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(python3 - "$prompt" "$max" <<'PY'
import json, sys
print(json.dumps({"model": "default", "messages": [{"role": "user", "content": sys.argv[1]}],
                  "max_tokens": int(sys.argv[2]), "temperature": 0.0, "stream": False}))
PY
)" -o "$OUT/${name}_${mode}.json" || { echo "FATAL: request $name ($mode) failed" >&2; exit 1; }
  python3 - "$OUT/${name}_${mode}.json" "$OUT/${name}_${mode}.txt" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
open(sys.argv[2], "w").write(d["choices"][0]["message"]["content"])
print(f'{d["usage"]["avg_compl_tok_per_sec"]:.2f}')
PY
}

declare -A TPS
for mode in on off; do
  serve $mode
  # Warm up before measuring: the first generation after a model load pays one-time GPU/driver
  # warmup (and, graph-on, the first capture), which would otherwise corrupt whichever prompt runs
  # first. Discarded — its only job is to leave the device warm so the matrix is comparable.
  ask warmup "Warm up." 32 "$mode" >/dev/null
  for row in "${PROMPTS[@]}"; do
    IFS='|' read -r name prompt max <<<"$row"
    TPS[${name}_${mode}]=$(ask "$name" "$prompt" "$max" "$mode")
    echo "[$mode] $name: ${TPS[${name}_${mode}]} t/s"
  done
  fuser -k "${PORT}/tcp" 2>/dev/null; sleep 1
done

FAIL=0
echo "== identity gate =="
for row in "${PROMPTS[@]}"; do
  IFS='|' read -r name _ _ <<<"$row"
  if cmp -s "$OUT/${name}_on.txt" "$OUT/${name}_off.txt"; then
    echo "  $name: identical"
  else
    echo "  $name: DIVERGED — graph output != eager output"; FAIL=1
  fi
done

echo "== perf gates (no sustained prompt worse than -${REGRESS_PCT}%, mean sustained win >= ${MIN_WIN_PCT}%) =="
python3 - "$OUT/bench_vk_graph.json" "$MIN_WIN_PCT" "$REGRESS_PCT" <<PY || FAIL=1
import json, sys
tps = { $(for k in "${!TPS[@]}"; do printf '"%s": %s, ' "$k" "${TPS[$k]}"; done) }
min_win, max_loss = float(sys.argv[2]), float(sys.argv[3])
fail = False
rows, sustained = {}, []
for name in sorted({k.rsplit("_", 1)[0] for k in tps}):
    on, off = tps[f"{name}_on"], tps[f"{name}_off"]
    delta = (on - off) / off * 100
    rows[name] = {"graph_on_tps": on, "eager_tps": off, "delta_pct": round(delta, 1)}
    if name == "short":
        # Sub-breakeven timing is startup-dominated noise; identity + the autoshape gate below
        # (zero captures) are the real proof it runs the eager path. Reported, not gated.
        rows[name]["gate"] = "reported (see autoshape gate)"
    else:
        ok = delta >= -max_loss
        rows[name]["gate"] = "ok" if ok else f"REGRESSION beyond {max_loss}%"
        sustained.append(delta)
        fail |= not ok
    print(f"  {name}: on {on:.2f} / off {off:.2f} = {delta:+.1f}%  [{rows[name]['gate']}]")
mean_win = sum(sustained) / len(sustained) if sustained else 0.0
win_ok = mean_win >= min_win
print(f"  mean sustained win: {mean_win:+.1f}%  [{'ok' if win_ok else f'BELOW {min_win}%'}]")
fail |= not win_ok
rows["_mean_sustained_win_pct"] = round(mean_win, 1)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
sys.exit(1 if fail else 0)
PY

# Autoshape gate: a short-only workload must teach the policy to stop capturing. Fresh graph-on
# server, a burst of short generations; the server logs one line per capture, so zero lines proves
# the self-shaping policy kept every short generation on the eager path.
echo "== autoshape gate (short-only workload -> zero captures) =="
serve on
for _ in $(seq 1 8); do ask burst "Say a random short greeting." 24 on >/dev/null; done
# grep -c prints 0 and exits 1 on no match, so read its stdout and normalize; never `|| echo 0`
# (that would append a second line and break the integer compare).
CAPTURES=$(grep -c 'vulkan decode graph: captured' "$OUT/server_on.log" 2>/dev/null); CAPTURES=${CAPTURES:-0}
fuser -k "${PORT}/tcp" 2>/dev/null; sleep 1
if [[ "$CAPTURES" -eq 0 ]]; then
  echo "  8 short generations -> $CAPTURES captures  [ok — policy suppressed capture]"
else
  echo "  8 short generations -> $CAPTURES captures  [FAIL — policy captured a short workload]"; FAIL=1
fi

[[ $FAIL == 0 ]] && echo "PASS" || { echo "FAIL"; exit 1; }
