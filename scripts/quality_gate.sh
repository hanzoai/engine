#!/usr/bin/env bash
# quality_gate.sh — perplexity parity: is hanzo-engine as CORRECT as llama.cpp on the SAME
# GGUF at the SAME quant? A throughput win is void if it is a win by being subtly wrong; this
# is the guard. Run per model, beside the throughput battery.
#
# Method parity is the whole point: BOTH engines score the SAME fixed text by teacher-forced
# per-token logprobs via /v1/completions (echo=true, logprobs, max_tokens=0, temperature=0),
# and perplexity = exp(-mean token_logprob). Same tokens, same method, same quant -> the delta
# is the engine's numerical fidelity, not a windowing artifact. |ppl_ours - ppl_llama| /
# ppl_llama <= tol (default 0.02) PASSES. At identical weights+quant the two should agree to
# well under a percent; a larger gap means our kernels drift the distribution.
#
#   quality_gate.sh --model GGUF --hanzo-server PATH --llama-server PATH \
#     [--backend-env "K=V ..."] [--text FILE] [--tol 0.02] [--ctx 1024] \
#     [--port-h 8611 --port-l 8612] [--out DIR]
set -euo pipefail
MODEL="" HS="" LS="" TEXT="" TOL=0.02 CTX=1024 PORTH=8611 PORTL=8612 OUT="" BENV=""
while [[ $# -gt 0 ]]; do case "$1" in
  --model) MODEL="$2"; shift 2;;
  --hanzo-server) HS="$2"; shift 2;;
  --llama-server) LS="$2"; shift 2;;
  --backend-env) BENV="$2"; shift 2;;
  --text) TEXT="$2"; shift 2;;
  --tol) TOL="$2"; shift 2;;
  --ctx) CTX="$2"; shift 2;;
  --port-h) PORTH="$2"; shift 2;;
  --port-l) PORTL="$2"; shift 2;;
  --out) OUT="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done
: "${MODEL:?--model required}" "${HS:?--hanzo-server required}" "${LS:?--llama-server required}"
OUT="${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/qgate.XXXXXX")}"; mkdir -p "$OUT"
MDIR="$(cd "$(dirname "$MODEL")" && pwd)"; MFILE="$(basename "$MODEL")"

# A fixed, self-contained corpus so the gate is deterministic and needs no network. Natural
# English prose the model should model well; the ABSOLUTE ppl varies by model, only the
# ours-vs-llama DELTA is judged. Override with --text for a standard set (e.g. wikitext-2).
if [[ -z "$TEXT" ]]; then
  TEXT="$OUT/corpus.txt"
  cat > "$TEXT" <<'CORPUS'
The history of computing hardware spans from early mechanical calculators to modern general
purpose processors. Charles Babbage designed the analytical engine, a proposed mechanical
computer, and Ada Lovelace wrote what is regarded as the first algorithm intended for such a
machine. Electronic digital computers emerged in the middle of the twentieth century, using
vacuum tubes and later transistors. The invention of the integrated circuit allowed enormous
numbers of components to be placed on a single chip, and the resulting exponential growth in
capability reshaped science, industry, and daily life. Today the same fundamental principles
of stored programs and binary arithmetic underlie devices from the smallest sensors to the
largest supercomputers, and the field continues to advance through parallelism, specialized
accelerators, and increasingly sophisticated software.
CORPUS
fi

SPID=""
cleanup() { [[ -n "$SPID" ]] && kill -TERM "$SPID" 2>/dev/null && for _ in $(seq 1 20); do kill -0 "$SPID" 2>/dev/null || break; sleep 0.5; done; kill -KILL "$SPID" 2>/dev/null || true; SPID=""; }
trap cleanup EXIT

wait_health() { local url="$1" waited=0; until curl -sf "$url" >/dev/null 2>&1; do kill -0 "$SPID" 2>/dev/null || { echo "server died" >&2; return 1; }; sleep 1; waited=$((waited+1)); ((waited>600)) && return 1; done; }

# Score the corpus on a running OpenAI-compatible server -> perplexity via echo+logprobs.
score() { # $1=port  -> prints ppl
  python3 - "$1" "$TEXT" <<'PY'
import json, sys, urllib.request
port, textf = sys.argv[1], sys.argv[2]
prompt = open(textf).read()
body = json.dumps({"model":"default","prompt":prompt,"echo":True,"logprobs":1,
                   "max_tokens":0,"temperature":0}).encode()
req = urllib.request.Request(f"http://localhost:{port}/v1/completions", data=body,
                            headers={"content-type":"application/json"})
with urllib.request.urlopen(req, timeout=600) as r: d = json.load(r)
lp = d["choices"][0].get("logprobs",{}) or {}
toks = lp.get("token_logprobs") or []
vals = [x for x in toks if isinstance(x,(int,float))]   # first prompt token has null logprob
if not vals: print("nan"); sys.exit()
import math
print(f"{math.exp(-sum(vals)/len(vals)):.4f}")
PY
}

echo "quality-gate: $MFILE" >&2
# --- ours ---
env $BENV "$HS" --port "$PORTH" --prefix-cache-n 0 gguf -m "$MDIR" -f "$MFILE" --max-seq-len "$CTX" > "$OUT/hanzo-server.log" 2>&1 &
SPID=$!; wait_health "http://localhost:$PORTH/health" || { echo "hanzo server no bind" >&2; tail -5 "$OUT/hanzo-server.log" >&2; }
PPL_H="$(score "$PORTH" || echo nan)"; cleanup
# --- llama ---
env $BENV "$LS" -m "$MODEL" --port "$PORTL" -c "$CTX" -ngl 999 > "$OUT/llama-server.log" 2>&1 &
SPID=$!; wait_health "http://localhost:$PORTL/health" || { echo "llama server no bind" >&2; tail -5 "$OUT/llama-server.log" >&2; }
PPL_L="$(score "$PORTL" || echo nan)"; cleanup

python3 - "$OUT/quality.json" "$MFILE" "$PPL_H" "$PPL_L" "$TOL" <<'PY'
import json, sys, math
out, model, h, l, tol = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
h=float(h) if h not in ("nan","") else float("nan"); l=float(l) if l not in ("nan","") else float("nan")
rel = abs(h-l)/l if l and not math.isnan(h) and not math.isnan(l) else float("nan")
verdict = "PASS" if (not math.isnan(rel) and rel <= tol) else ("FAIL" if not math.isnan(rel) else "ERROR")
r={"model":model,"ppl_hanzo":h,"ppl_llama":l,"rel_delta":rel,"tol":tol,"verdict":verdict,
   "method":"teacher-forced echo+logprobs perplexity, same GGUF/quant, same corpus"}
json.dump(r, open(out,"w"), indent=2)
print(f"{model}: ppl ours={h:.3f} llama={l:.3f} rel={rel:.4f} tol={tol} -> {verdict}")
PY
