#!/usr/bin/env bash
# spec_decode.sh — the ARCHITECTURE row: prompt-lookup speculative-decode multiple, per backend.
#
# Faithful port of the Metal lane's protocol (bench-protocols/spec-decode-metal.md): three
# verbatim workload classes, ngram_max=3, a gamma sweep, greedy + lossless (target verifies every
# draft -> byte-identical to baseline greedy). Reported BESIDE the kernel board, never blended --
# this is an algorithmic multiple, orthogonal to per-token kernel efficiency. Run the SAME thing
# per backend so the acceptance/multiple is byte-comparable.
#
#   spec_decode.sh --model GGUF --hanzo HANZO_CLI --llama-bench PATH \
#     [--backend-env "K=V ..."] [--gammas 4,8,16] [--ngram 3] [--reps 7] [--maxtok 200] \
#     [--port 8613] [--out DIR]
#
# multiple = spec t/s / baseline t/s (same server, spec flags off = baseline). vs-llama = spec
# t/s / llama-bench greedy t/s. Lossless is asserted by SHA(completion) with lookup ON == OFF.
set -euo pipefail
MODEL="" HANZO="" LB="" BENV="" GAMMAS="4,8,16" NGRAM=3 REPS=7 MAXTOK=200 PORT=8613 OUT=""
while [[ $# -gt 0 ]]; do case "$1" in
  --model) MODEL="$2"; shift 2;; --hanzo) HANZO="$2"; shift 2;; --llama-bench) LB="$2"; shift 2;;
  --backend-env) BENV="$2"; shift 2;; --gammas) GAMMAS="$2"; shift 2;; --ngram) NGRAM="$2"; shift 2;;
  --reps) REPS="$2"; shift 2;; --maxtok) MAXTOK="$2"; shift 2;; --port) PORT="$2"; shift 2;;
  --out) OUT="$2"; shift 2;; *) echo "unknown arg: $1" >&2; exit 2;;
esac; done
: "${MODEL:?--model required}" "${HANZO:?--hanzo (CLI binary) required}"
OUT="${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/spec.XXXXXX")}"; mkdir -p "$OUT"
MDIR="$(cd "$(dirname "$MODEL")" && pwd)"; MFILE="$(basename "$MODEL")"

# Verbatim prompts (heredocs preserve newlines/spacing exactly -- byte-comparable across backends).
cat > "$OUT/A.txt" <<'PA'
The mission log recorded the same entry every day. Day one: all systems nominal, crew resting, oxygen stable, course unchanged. Day two: all systems nominal, crew resting, oxygen stable, course unchanged. Day three: all systems nominal, crew resting, oxygen stable, course unchanged. Day four: all systems nominal, crew resting, oxygen stable, course unchanged. Day five:
PA
cat > "$OUT/B.txt" <<'PB'
Reproduce this file exactly with no changes:
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def fact(n):
    r = 1
    for i in range(1, n+1):
        r *= i
    return r

Exact reproduction:
def fib(n):
PB
cat > "$OUT/C.txt" <<'PC'
Write an original science fiction story about the first human colony on a distant exoplanet. Begin now:
PC

SPID=""
cleanup() { [[ -n "$SPID" ]] && kill -TERM "$SPID" 2>/dev/null && for _ in $(seq 1 20); do kill -0 "$SPID" 2>/dev/null || break; sleep 0.5; done; kill -KILL "$SPID" 2>/dev/null || true; SPID=""; }
trap cleanup EXIT
wait_health() { local waited=0; until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do kill -0 "$SPID" 2>/dev/null || return 1; sleep 1; waited=$((waited+1)); ((waited>600)) && return 1; done; }

# Run one server config over all three workloads. $1=label, $2..=extra serve flags (spec on/off).
run_cfg() {
  local label="$1"; shift
  env $BENV "$HANZO" serve "$@" --format gguf -m "$MDIR" -f "$MFILE" -p "$PORT" > "$OUT/serve-$label.log" 2>&1 &
  SPID=$!; wait_health || { echo "server '$label' no bind" >&2; tail -5 "$OUT/serve-$label.log" >&2; cleanup; return 1; }
  for wl in A B C; do
    python3 - "$PORT" "$OUT/$wl.txt" "$REPS" "$MAXTOK" "$label" "$wl" "$OUT/spec_raw.jsonl" <<'PY'
import json, sys, time, hashlib, urllib.request
port, pf, reps, maxtok, label, wl, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7]
prompt = open(pf).read()
url = f"http://localhost:{port}/v1/completions"
ts, txt = [], ""
for i in range(reps):
    body = json.dumps({"model":"default","prompt":prompt,"max_tokens":maxtok,"temperature":0}).encode()
    t0 = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers={"content-type":"application/json"}), timeout=600) as r:
        d = json.load(r)
    dt = time.perf_counter() - t0
    n = (d.get("usage") or {}).get("completion_tokens") or 0
    txt = d["choices"][0].get("text","")
    if i > 0 and dt > 0 and n > 0:            # drop the first (cold) rep
        ts.append(n/dt)
mean = sum(ts)/len(ts) if ts else float("nan")
rec = {"label":label,"workload":wl,"tok_s":mean,"n_scored":len(ts),
       "completion_sha256":hashlib.sha256(txt.encode()).hexdigest()}
open(out,"a").write(json.dumps(rec)+"\n")
print(f"  {label} {wl}: {mean:.1f} t/s (sha {rec['completion_sha256'][:8]})", file=sys.stderr)
PY
  done
  cleanup
}

echo "spec-decode: $MFILE (ngram=$NGRAM gammas=$GAMMAS)" >&2
: > "$OUT/spec_raw.jsonl"
LLAMA_TS="nan"
if [[ -n "$LB" && -x "$LB" ]]; then
  LLAMA_TS="$(env $BENV "$LB" -m "$MODEL" -p 0 -n 256 -o json 2>/dev/null | python3 -c "import sys,json; r=[x for x in json.load(sys.stdin) if x.get('n_gen',0)>0]; print(f'{r[0][\"avg_ts\"]:.1f}' if r else 'nan')" 2>/dev/null || echo nan)"
fi
run_cfg baseline || true
IFS=',' read -ra GS <<< "$GAMMAS"
for g in "${GS[@]}"; do run_cfg "g$g" --prompt-lookup-ngram "$NGRAM" --gamma "$g" || true; done

python3 - "$OUT" "$LLAMA_TS" <<'PY'
import json, os, sys, math
out, llama = sys.argv[1], float(sys.argv[2]) if sys.argv[2] not in ("nan","") else float("nan")
rows = [json.loads(l) for l in open(os.path.join(out,"spec_raw.jsonl"))]
by = {(r["label"], r["workload"]): r for r in rows}
names = {"A":"sustained-reuse","B":"code-copy","C":"novel"}
res = {"llama_greedy_ts": llama, "workloads": []}
labels = sorted({r["label"] for r in rows if r["label"]!="baseline"})
for wl in ["A","B","C"]:
    base = by.get(("baseline",wl))
    if not base: continue
    bt = base["tok_s"]
    entry = {"workload": names[wl], "baseline_ts": bt, "gammas": {}}
    for lab in labels:
        r = by.get((lab,wl))
        if not r: continue
        mult = r["tok_s"]/bt if bt else float("nan")
        vsll = r["tok_s"]/llama if llama and not math.isnan(llama) else float("nan")
        lossless = (r["completion_sha256"] == base["completion_sha256"])
        entry["gammas"][lab] = {"ts": r["tok_s"], "multiple": mult, "vs_llama": vsll, "lossless": lossless}
    res["workloads"].append(entry)
    best = max(entry["gammas"].values(), key=lambda x:x["multiple"], default=None)
    if best:
        print(f'{names[wl]:16} baseline {bt:6.1f}  best mult {best["multiple"]:.2f}x  vs-llama {best["vs_llama"]:.2f}x  lossless={all(g["lossless"] for g in entry["gammas"].values())}')
json.dump(res, open(os.path.join(out,"spec_decode.json"),"w"), indent=2)
print("-> "+os.path.join(out,"spec_decode.json"))
PY
