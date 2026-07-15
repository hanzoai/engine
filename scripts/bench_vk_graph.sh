#!/usr/bin/env bash
# bench_vk_graph — the Vulkan decode command-graph gate: correctness + perf, reproducibly.
#
# Runs hanzo-server twice on one GGUF — graph default-ON (env unset) and forced eager
# (VK_GRAPHS=0) — over a fixed greedy prompt matrix, then enforces:
#   1. IDENTITY: every completion byte-identical ON vs OFF (greedy, temp 0). The graph's
#      failure mode is fluent-but-stale output, so this is the correctness gate.
#   2. NO SHORT-GEN REGRESSION: generations below the capture threshold must not lose more
#      than REGRESS_PCT vs eager (lazy capture means they run the same path).
#   3. SUSTAINED WIN: long generations must beat eager by at least MIN_WIN_PCT.
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
REGRESS_PCT=10

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

echo "== perf gates (min sustained win ${MIN_WIN_PCT}%, max short loss ${REGRESS_PCT}%) =="
python3 - "$OUT/bench_vk_graph.json" "$MIN_WIN_PCT" "$REGRESS_PCT" <<PY || FAIL=1
import json, sys
tps = { $(for k in "${!TPS[@]}"; do printf '"%s": %s, ' "$k" "${TPS[$k]}"; done) }
min_win, max_loss = float(sys.argv[2]), float(sys.argv[3])
fail = False
rows = {}
for name in {k.rsplit("_", 1)[0] for k in tps}:
    on, off = tps[f"{name}_on"], tps[f"{name}_off"]
    delta = (on - off) / off * 100
    rows[name] = {"graph_on_tps": on, "eager_tps": off, "delta_pct": round(delta, 1)}
    if name == "short":
        ok = delta >= -max_loss
        verdict = "ok" if ok else f"REGRESSION beyond {max_loss}%"
    else:
        ok = delta >= min_win
        verdict = "ok" if ok else f"BELOW {min_win}% win"
    rows[name]["gate"] = verdict
    print(f"  {name}: on {on:.2f} / off {off:.2f} = {delta:+.1f}%  [{verdict}]")
    fail |= not ok
json.dump(rows, open(sys.argv[1], "w"), indent=1)
sys.exit(1 if fail else 0)
PY

[[ $FAIL == 0 ]] && echo "PASS" || { echo "FAIL"; exit 1; }
