#!/usr/bin/env bash
# bench_vs_llama — head-to-head proof harness: llama.cpp vs hanzo-engine, one GGUF.
#
# Given a GGUF path it (1) runs llama-bench (pp<P>/tg<N>, -fa 1, -r reps) and
# parses prompt/generation tok/s from its JSON; (2) launches OUR hanzo-server on
# the same file, proves the bind, measures prefill tok/s (P-token prompt,
# usage.prompt_tokens over wall) and steady decode tok/s (N new tokens,
# prefill-excluded via the engine's usage superset, wall-minus-prefill fallback);
# (3) folds both into ONE table — model, metric, llama.cpp, hanzo-engine, ratio,
# winner — plus bench_vs_llama.json.
#
# Concerns are split, not braided (same shape as bench_moe.sh): this file does
# lifecycle + serialization; inline python blocks each own exactly one concern
# (parse rival, drive ours, fold + render). The engines NEVER coexist in memory:
# the rival runs to completion first, and free RAM is re-proved before each load.
# Rival errors and server errors are captured into the JSON — never fatal to the
# other side's numbers — so a flaky engine still yields an honest report.
#
# Build/run env (REQUIRED — the sandbox cc shim rejects -m64):
#   cd ~/work/hanzo/engine
#   PATH="/usr/bin:$PATH:$HOME/.cargo/bin:/opt/rocm/bin" \
#     CC=/usr/bin/gcc CXX=/usr/bin/g++ \
#     CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc \
#     cargo build --release -p hanzo-server --features rocm
# Runtime adds LD_LIBRARY_PATH=/opt/rocm/lib and HSA_XNACK=1 (gfx1151 unified
# memory — hipMalloc otherwise OOMs in the 1 GB carve-out).
#
#   bench_vs_llama.sh <model.gguf> [-o out-dir] [--server path] [--llama-bench path]
#                     [-p prompt-tokens] [-n gen-tokens] [-r reps] [--port port]
#                     [--llama-json path]   reuse a prior llama-bench JSON (skip rerun)
#                     [--skip-ours]         rival-only run
#
# Env overrides: HANZO_SERVER, LLAMA_BENCH, LD_LIBRARY_PATH.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=""
OUT=""
SERVER="${HANZO_SERVER:-$HERE/../target/release/hanzo-server}"
LLAMA="${LLAMA_BENCH:-/home/z/llama.cpp/build/bin/llama-bench}"
LD_ROCM="${LD_LIBRARY_PATH:-/opt/rocm/lib}"
P_TOK=512
N_TOK=128
REPS=3
PORT=1234
MAX_CTX=8192
LLAMA_JSON=""
SKIP_OURS=0
MIN_FREE_GB=25

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUT="$2"; shift 2 ;;
    --server) SERVER="$2"; shift 2 ;;
    --llama-bench) LLAMA="$2"; shift 2 ;;
    -p) P_TOK="$2"; shift 2 ;;
    -n) N_TOK="$2"; shift 2 ;;
    -r) REPS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --max-ctx) MAX_CTX="$2"; shift 2 ;;
    --llama-json) LLAMA_JSON="$2"; shift 2 ;;
    --skip-ours) SKIP_OURS=1; shift ;;
    -*) echo "unknown arg: $1" >&2; exit 2 ;;
    *) MODEL="$1"; shift ;;
  esac
done

[[ -n "$MODEL" && -f "$MODEL" ]] || { echo "usage: bench_vs_llama.sh <model.gguf> [flags]" >&2; exit 2; }
OUT="${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/bench_vs_llama.XXXXXX")}"
mkdir -p "$OUT"
echo "out-dir: $OUT" >&2
MODEL_DIR="$(dirname "$MODEL")"
MODEL_FILE="$(basename "$MODEL")"
SERVER_PID=""

# One trap owns cleanup: graceful TERM by the recorded PID, KILL only if it
# lingers. Never a blind pkill across the box.
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || return 0; sleep 0.5; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# --- RAM discipline: prove headroom before every engine load -------------------
require_free_gb() { # $1=floor
  local free_gb
  free_gb="$(free -g | awk '/^Mem:/{print $7}')"
  if (( free_gb < $1 )); then
    echo "refusing to load: ${free_gb} GB available < ${1} GB floor (another model resident?)" >&2
    exit 3
  fi
  echo "free check: ${free_gb} GB available >= ${1} GB floor" >&2
}

# --- reproducibility manifest ---------------------------------------------------
write_manifest() {
  python3 - "$OUT/manifest.json" "$HERE" "$MODEL" "$SERVER" "$LLAMA" "$P_TOK" "$N_TOK" "$REPS" <<'PY'
import json, os, platform, subprocess, sys, time
out, here, model, server, llama, p, n, r = sys.argv[1:9]
def sh(*a):
    try: return subprocess.check_output(a, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
m = {
  "git_sha": sh("git", "-C", here, "rev-parse", "HEAD"),
  "host": platform.node(),
  "uname": " ".join(os.uname()),
  "model_path": model,
  "server_bin": server,
  "llama_bench_bin": llama,
  "params": {"prompt_tokens": int(p), "gen_tokens": int(n), "reps": int(r)},
  "env": {k: os.environ.get(k) for k in ("LD_LIBRARY_PATH", "HSA_XNACK")},
  "timestamp": time.time(),
}
json.dump(m, open(out, "w"), indent=2)
print("manifest ->", out, file=sys.stderr)
PY
}

# --- side 1: the rival ----------------------------------------------------------
run_llama() { # -> $OUT/llama.json (raw llama-bench output or {"error": ...})
  if [[ -n "$LLAMA_JSON" ]]; then
    [[ -f "$LLAMA_JSON" ]] || { echo "no such --llama-json: $LLAMA_JSON" >&2; exit 2; }
    cp "$LLAMA_JSON" "$OUT/llama.json"
    echo "llama.cpp: reusing $LLAMA_JSON" >&2
    return 0
  fi
  if [[ ! -x "$LLAMA" ]]; then
    echo "{\"error\": \"no llama-bench binary: $LLAMA\"}" > "$OUT/llama.json"
    echo "llama.cpp: missing binary, recorded as error" >&2
    return 0
  fi
  require_free_gb "$MIN_FREE_GB"
  echo "llama.cpp: llama-bench pp$P_TOK/tg$N_TOK fa=1 r=$REPS ..." >&2
  if ! "$LLAMA" -m "$MODEL" -p "$P_TOK" -n "$N_TOK" -fa 1 -r "$REPS" -o json \
      > "$OUT/llama.json" 2> "$OUT/llama.err"; then
    echo "{\"error\": \"llama-bench exited nonzero; see llama.err\"}" > "$OUT/llama.json"
    echo "llama.cpp: FAILED (see $OUT/llama.err)" >&2
  fi
}

# --- side 2: ours ---------------------------------------------------------------
start_server() { # bind-poll /health with pid-liveness bail; sets SERVER_PID
  require_free_gb "$MIN_FREE_GB"
  local log="$OUT/server.log"
  echo "hanzo-engine: launching $SERVER on port $PORT ..." >&2
  # --pa-ctxt-len caps the PagedAttention KV cache at MAX_CTX tokens: without it
  # the 0.9 memory-usage heuristic grabs ~84 GB of the 128 GB unified pool.
  env HSA_XNACK=1 LD_LIBRARY_PATH="$LD_ROCM" \
    "$SERVER" --port "$PORT" --max-seqs 4 --prefix-cache-n 0 \
    --pa-ctxt-len "$MAX_CTX" \
    gguf -m "$MODEL_DIR" -f "$MODEL_FILE" --max-seq-len "$MAX_CTX" > "$log" 2>&1 &
  SERVER_PID=$!
  local waited=0
  until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      SERVER_PID=""
      echo "hanzo-engine: server exited before bind; tail of $log:" >&2
      tail -5 "$log" >&2 || true
      return 1
    fi
    sleep 1; waited=$((waited + 1))
    if (( waited > 600 )); then
      echo "hanzo-engine: no bind after ${waited}s; giving up" >&2
      return 1
    fi
  done
  echo "hanzo-engine: ready in ~${waited}s (pid $SERVER_PID)" >&2
}

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}

run_ours() { # -> $OUT/hanzo.json ({"prefill":..,"decode":..} or {"error":..})
  if ! start_server; then
    echo "{\"error\": \"server failed to start; see server.log\"}" > "$OUT/hanzo.json"
    return 0
  fi
  # The client owns measurement; stdlib-only so the harness has no pip deps.
  # Prefill: P-token prompt, max_tokens=1 -> usage.prompt_tokens / wall (the
  # engine's usage.avg_prompt_tok_per_sec preferred when present). Decode: tiny
  # prompt, N new tokens -> usage.avg_compl_tok_per_sec (prefill-excluded at the
  # source); fallback completion_tokens / (wall - measured tiny-prompt prefill).
  # Any per-request error is recorded, not raised — flaky server, honest report.
  python3 - "http://localhost:$PORT" "$P_TOK" "$N_TOK" "$REPS" "$OUT/hanzo.json" <<'PY' >&2 || \
    echo '{"error": "bench client crashed"}' > "$OUT/hanzo.json"
import json, statistics, sys, time, urllib.error, urllib.request, uuid

base, p_tok, n_tok, reps, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
URL = base + "/v1/chat/completions"

def filler(n):
    # ~1 token per short word; usage.prompt_tokens is the ground truth anyway.
    head = f"bench {uuid.uuid4().hex} "
    return head + " ".join(f"w{i}" for i in range(max(0, n - 8)))

def chat(prompt, max_tokens):
    """One request -> (wall_s, usage) or raises with the server's own words."""
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "seed": 42,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    return time.perf_counter() - t0, data.get("usage") or {}

def attempt(fn, errors, phase):
    try:
        return fn()
    except urllib.error.HTTPError as e:
        errors.append({"phase": phase, "http": e.code, "body": e.read().decode(errors="replace")[:500]})
    except Exception as e:
        errors.append({"phase": phase, "error": f"{type(e).__name__}: {e}"})
    return None

errors, prefill, decode = [], [], []

# Unscored warmup (graph capture, allocator steady-state).
attempt(lambda: chat("warmup " + uuid.uuid4().hex, 8), errors, "warmup")

# Tiny-prompt prefill cost, measured once: the decode fallback subtracts it.
tiny_prefill_s = 0.0
r = attempt(lambda: chat(filler(8), 1), errors, "tiny-prefill")
if r:
    tiny_prefill_s = r[0]

for i in range(reps):
    r = attempt(lambda: chat(filler(p_tok), 1), errors, f"prefill[{i}]")
    if r:
        wall, u = r
        n = u.get("prompt_tokens")
        rate = u.get("avg_prompt_tok_per_sec") or (n / wall if n else None)
        if rate:
            prefill.append({"tok_s": rate, "wall_s": wall, "prompt_tokens": n})
    r = attempt(lambda: chat(filler(8), n_tok), errors, f"decode[{i}]")
    if r:
        wall, u = r
        n = u.get("completion_tokens")
        rate = u.get("avg_compl_tok_per_sec")
        src = "usage"
        if not rate and n and wall > tiny_prefill_s:
            rate, src = n / (wall - tiny_prefill_s), "wall-minus-prefill"
        if rate:
            decode.append({"tok_s": rate, "wall_s": wall, "completion_tokens": n, "source": src})

def fold(recs):
    if not recs:
        return None
    ts = [x["tok_s"] for x in recs]
    return {"tok_s": statistics.mean(ts),
            "stddev": statistics.stdev(ts) if len(ts) > 1 else 0.0,
            "runs": recs}

result = {"prefill": fold(prefill), "decode": fold(decode), "errors": errors}
if not prefill and not decode:
    result["error"] = "no successful requests; see errors[]"
json.dump(result, open(out, "w"), indent=2)
print("hanzo-engine ->", out, file=sys.stderr)
PY
  stop_server
}

# --- fold both sides into the table + bench_vs_llama.json ------------------------
report() {
  python3 - "$OUT" "$MODEL_FILE" "$P_TOK" "$N_TOK" <<'PY'
import json, os, sys
out, model, p_tok, n_tok = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def load(p):
    try:
        return json.load(open(os.path.join(out, p)))
    except Exception as e:
        return {"error": f"unreadable {p}: {e}"}

llama, ours = load("llama.json"), load("hanzo.json")

def llama_metric(rows, want_prompt):
    """llama-bench JSON row -> (tok_s, stddev): pp has n_prompt>0/n_gen==0, tg the inverse."""
    if not isinstance(rows, list):
        return None
    for row in rows:
        pp, tg = row.get("n_prompt", 0), row.get("n_gen", 0)
        if (pp > 0 and tg == 0) if want_prompt else (tg > 0 and pp == 0):
            return row.get("avg_ts"), row.get("stddev_ts")
    return None

def ours_metric(phase):
    v = ours.get(phase) if isinstance(ours, dict) else None
    return (v["tok_s"], v.get("stddev", 0.0)) if v else None

metrics = [(f"pp{p_tok} tok/s", llama_metric(llama, True), ours_metric("prefill")),
           (f"tg{n_tok} tok/s", llama_metric(llama, False), ours_metric("decode"))]

rows, folded = [], []
for name, lm, om in metrics:
    lv, ov = (lm[0] if lm else None), (om[0] if om else None)
    ratio = (ov / lv) if lv and ov else None
    winner = "hanzo-engine" if (ratio or 0) > 1 else ("llama.cpp" if ratio else
             "hanzo-engine" if ov else "llama.cpp" if lv else "neither")
    fmt = lambda m: f"{m[0]:.1f} ±{m[1]:.1f}" if m and m[0] is not None else "ERROR"
    rows.append((name, fmt(lm), fmt(om), f"{ratio:.2f}x" if ratio else "-", winner))
    folded.append({"metric": name,
                   "llama_cpp": {"tok_s": lv, "stddev": lm[1] if lm else None},
                   "hanzo": {"tok_s": ov, "stddev": om[1] if om else None},
                   "ratio": ratio, "winner": winner})

widths = [max(len(model), 5), 12, 15, 15, 7, 12]
hdr = ("model", "metric", "llama.cpp", "hanzo-engine", "ratio", "winner")
def line(cells):
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))
print(line(hdr))
print(line(["-" * w for w in widths]))
for i, (name, l, o, r, w) in enumerate(rows):
    print(line((model if i == 0 else "", name, l, o, r, w)))
for side, blob in (("llama.cpp", llama), ("hanzo-engine", ours)):
    err = blob.get("error") if isinstance(blob, dict) else None
    if err:
        print(f"note: {side}: {err}")
    if isinstance(blob, dict) and blob.get("errors"):
        print(f"note: hanzo-engine request errors: {len(blob['errors'])} (see bench_vs_llama.json)")

final = {"manifest": load("manifest.json"), "model": model,
         "results": folded, "llama_cpp_raw": llama, "hanzo_raw": ours}
path = os.path.join(out, "bench_vs_llama.json")
json.dump(final, open(path, "w"), indent=2)
print("json ->", path, file=sys.stderr)
PY
}

# --- main: strictly serialized, never both engines resident ---------------------
write_manifest
run_llama
if [[ "$SKIP_OURS" == 1 ]]; then
  echo '{"error": "skipped (--skip-ours)"}' > "$OUT/hanzo.json"
else
  run_ours
fi
report
echo "done: $OUT" >&2
