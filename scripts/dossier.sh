#!/usr/bin/env bash
# dossier.sh — collect count-based prefill/decode samples for the SOTA dossier.
#
# Runs OUR hanzo-bench (--json: raw per-rep [wall_s, tokens], immune to any
# self-reported-rate bug) and llama.cpp's llama-bench (-o json) on the SAME GGUF,
# the SAME shapes, on a VERIFIED-QUIET box, and records a pinned manifest so the
# numbers are reproducible. Timing is wall-clock for BOTH engines (identical
# method => no cross-engine instrument bias); repetition + Student-t CIs quantify
# residual noise; the quiet-gate removes contention. Analysis is bench_stats.py.
#
#   dossier.sh --backend rocm|vulkan|metal|cuda \
#              --hanzo-bench PATH --llama-bench PATH \
#              --model GGUF --model-tag qwen3-1p7b \
#              [-p 512,500 -n 128 -r 7] [--concurrency 1] [--max-ctx 4096] \
#              [--engine-dir DIR --llama-dir DIR] [--out DIR] [--force]
#
# One GPU workload per box: the quiet-gate HARD-FAILS if another GPU job
# (cargo/rustc/ncu/nsys/rocprof/metal-capture/hanzo-bench/hanzo-server/llama-bench/
# ollama/vllm) is already running. Desktop CPU load is recorded, not fatal.
set -euo pipefail

BACKEND="" HANZO="" LLAMA="" MODEL="" MTAG="" OUT="" FORCE=0 STOCH=0
PLIST="512" NGEN=128 REPS=7 CONC=1 MAXCTX=4096
ENGINE_DIR="$HOME/work/hanzo/engine" LLAMA_DIR=""
while [[ $# -gt 0 ]]; do case "$1" in
  --backend) BACKEND="$2"; shift 2;;
  --hanzo-bench) HANZO="$2"; shift 2;;
  --llama-bench) LLAMA="$2"; shift 2;;
  --model) MODEL="$2"; shift 2;;
  --model-tag) MTAG="$2"; shift 2;;
  -p) PLIST="$2"; shift 2;;
  -n) NGEN="$2"; shift 2;;
  -r) REPS="$2"; shift 2;;
  --concurrency) CONC="$2"; shift 2;;
  --max-ctx) MAXCTX="$2"; shift 2;;
  --engine-dir) ENGINE_DIR="$2"; shift 2;;
  --llama-dir) LLAMA_DIR="$2"; shift 2;;
  --out) OUT="$2"; shift 2;;
  --force) FORCE=1; shift;;
  --stochastic) STOCH=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done
: "${BACKEND:?--backend required}" "${HANZO:?--hanzo-bench required}" "${MODEL:?--model required}" "${MTAG:?--model-tag required}"
# Sampler is an explicit, recorded choice. Default greedy argmax = parity with llama-bench;
# --stochastic reproduces the full-vocab multinomial only to MEASURE its tax.
SAMPLER="$([[ "$STOCH" == 1 ]] && echo stochastic-temp1-fullvocab || echo greedy-argmax)"
[[ -f "$MODEL" ]] || { echo "no model: $MODEL" >&2; exit 2; }
[[ -x "$HANZO" ]] || { echo "no hanzo-bench: $HANZO" >&2; exit 2; }
OUT="${OUT:-$ENGINE_DIR/bench-runs/$(hostname -s)-$BACKEND-$MTAG-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
MDIR="$(cd "$(dirname "$MODEL")" && pwd)"; MFILE="$(basename "$MODEL")"

# --- backend runtime env (one place) -------------------------------------------
backend_env() { case "$BACKEND" in
  rocm)   echo "LD_LIBRARY_PATH=/opt/rocm/lib HSA_XNACK=1";;
  vulkan) echo "";;
  *)      echo "";;
esac; }

# --- GPU busy % (best-effort, per backend) -------------------------------------
gpu_busy() { case "$BACKEND" in
  rocm|vulkan) rocm-smi --showuse 2>/dev/null | grep -oE "GPU use \(%\): [0-9]+" | grep -oE "[0-9]+$" | head -1;;
  cuda)        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1;;
  *)           echo "na";;
esac; }

# --- quiet gate: one GPU workload per box --------------------------------------
COMPET='ncu|nsys|rocprof|rocprofv3|metal-capture|xctrace|cargo|rustc|hanzo-bench|hanzo-server|llama-bench|llama-server|ollama|vllm|sglang'
quiet_gate() {
  # The real GPU-contention signals are (a) GPU busy% and (b) a true GPU-compute
  # process. Match the competitor list against the COMMAND NAME, not the full
  # argv, so a guard daemon (earlyoom, whose --prefer string names these tools) or
  # a network-bound eval (python calling an API) that merely mentions them does not
  # false-positive. Desktop/CPU load is recorded as advisory, never fatal -- a
  # shared workstation is never fully idle, and CV>5% flagging in bench_stats.py
  # surfaces any timing noise it induces.
  local hits busy load
  hits="$(ps -eo pid=,comm= 2>/dev/null | grep -wE "$COMPET" | grep -vw "$$" || true)"
  busy="$(gpu_busy)"; load="$(uptime | grep -oE 'load average.*' || true)"
  { echo "quiet-gate @ $(date +%Y-%m-%dT%H:%M:%S%z)"; echo "gpu_busy%: $busy"; echo "$load";
    echo "gpu/compile competitors (by command name):"; echo "${hits:-  (none)}"; } | tee "$OUT/quiet-gate.txt"
  if [[ -n "$hits" && "$FORCE" != 1 ]]; then
    echo "REFUSING: a GPU/compile workload is running (see above). Re-run when quiet or --force." >&2; exit 3; fi
  # A numeric-expected backend (rocm/vulkan/cuda) whose busy% cannot be read is a gate
  # FAILURE, not a silent degrade to the process gate: an unreadable accelerator is exactly
  # when a stale or contending job hides. Metal has no CLI busy% and is process-gated by design.
  if [[ "$BACKEND" != metal && ! "$busy" =~ ^[0-9]+$ && "$FORCE" != 1 ]]; then
    echo "REFUSING: GPU busy%% unreadable on $BACKEND (got '$busy'); cannot verify idle. --force to override." >&2; exit 3; fi
  if [[ "$busy" =~ ^[0-9]+$ && "$busy" -gt 15 && "$FORCE" != 1 ]]; then
    echo "REFUSING: GPU busy ${busy}% > 15%. Re-run when idle or --force." >&2; exit 3; fi
}

# --- pinned manifest ------------------------------------------------------------
ver_of() { grep -A1 "^name = \"$1\"" "$ENGINE_DIR/Cargo.lock" 2>/dev/null | grep version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+'; }
dirty_of() { [[ -n "$(git -C "$1" status --porcelain 2>/dev/null)" ]] && echo true || echo false; }
manifest() {
  local ever egit gpu
  ever="$(grep -m1 '^version' "$ENGINE_DIR/Cargo.toml" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
  egit="$(git -C "$ENGINE_DIR" rev-parse HEAD 2>/dev/null || true)"
  gpu="$(rocminfo 2>/dev/null | grep -m1 'Marketing Name' | sed 's/^ *//' || nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  # Values pass via argv (key=value), never interpolated into the Python source: a model path
  # with a space or quote cannot break or inject the manifest builder. The binaries are hashed
  # and each tree's dirty state recorded -- a binary built from an uncommitted tree carries a
  # clean commit SHA and would otherwise misreport its provenance.
  python3 - "$OUT/manifest.json" \
    backend="$BACKEND" engine_dir="$ENGINE_DIR" engine_git_sha="$egit" \
    engine_dirty="$(dirty_of "$ENGINE_DIR")" engine_version="$ever" \
    hanzo_ml="$(ver_of hanzo-ml)" hanzo_rocm_kernels="$(ver_of hanzo-rocm-kernels)" \
    hanzo_metal_kernels="$(ver_of hanzo-metal-kernels)" \
    llama_dir="$LLAMA_DIR" llama_sha="$( [[ -n "$LLAMA_DIR" ]] && git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || true)" \
    llama_dirty="$( [[ -n "$LLAMA_DIR" ]] && dirty_of "$LLAMA_DIR" || echo false)" \
    hanzo_bench_bin="$HANZO" llama_bench_bin="$LLAMA" model_path="$MODEL" \
    prompt_sizes="$PLIST" n_gen="$NGEN" reps="$REPS" concurrency="$CONC" max_ctx="$MAXCTX" \
    backend_env="$(backend_env)" sampler="$SAMPLER" gpu="$gpu" <<'PY'
import json, hashlib, os, sys, time
out = sys.argv[1]
kv = dict(a.split("=", 1) for a in sys.argv[2:])
def sha256(p):
    if not p or not os.path.exists(p): return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def size(p): return os.path.getsize(p) if p and os.path.exists(p) else None
m = {
  "host": os.uname().nodename, "uname": " ".join(os.uname()), "backend": kv.get("backend"),
  "timestamp": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "sampler": kv.get("sampler"),
  "engine_git_sha": kv.get("engine_git_sha") or None, "engine_git_dirty": kv.get("engine_dirty") == "true",
  "engine_version": kv.get("engine_version"),
  "hanzo_ml": kv.get("hanzo_ml"), "hanzo_rocm_kernels": kv.get("hanzo_rocm_kernels"),
  "hanzo_metal_kernels": kv.get("hanzo_metal_kernels"),
  "llama_sha": kv.get("llama_sha") or None, "llama_git_dirty": kv.get("llama_dirty") == "true",
  "hanzo_bench_bin": kv.get("hanzo_bench_bin"), "hanzo_bench_bin_sha256": sha256(kv.get("hanzo_bench_bin")),
  "llama_bench_bin": kv.get("llama_bench_bin"), "llama_bench_bin_sha256": sha256(kv.get("llama_bench_bin")),
  "model_path": kv.get("model_path"), "model_bytes": size(kv.get("model_path")),
  "model_sha256": sha256(kv.get("model_path")),
  "params": {"prompt_sizes": kv.get("prompt_sizes"), "n_gen": int(kv.get("n_gen") or 0),
             "reps": int(kv.get("reps") or 0), "concurrency": int(kv.get("concurrency") or 1),
             "max_ctx": int(kv.get("max_ctx") or 0)},
  "env": {"backend_env": kv.get("backend_env")},
  "gpu": kv.get("gpu") or None,
}
json.dump(m, open(out, "w"), indent=2)
print("manifest ->", out)
PY
}

run_hanzo() { # $1=phase(pp|tg) $2=n $3=tag
  # hanzo-bench flags: -p n_prompt, -g n_gen. (-n is --num-device-layers -- never pass it;
  # leaving it unset selects the production auto device map, all layers on GPU here.)
  local p g; if [[ "$1" == pp ]]; then p="$2"; g=0; else p=0; g="$2"; fi
  echo ">> hanzo-bench $BACKEND $3 (p=$p g=$g r=$REPS c=$CONC)" >&2
  env $(backend_env) "$HANZO" -p "$p" -g "$g" -r "$REPS" --concurrency "$CONC" \
      $([[ "$STOCH" == 1 ]] && echo --stochastic) \
      --json "$OUT/hanzo_${3}.json" \
      gguf -m "$MDIR" -f "$MFILE" --max-seq-len "$MAXCTX" \
      > "$OUT/hanzo_${3}.log" 2>&1 || { echo "hanzo-bench FAILED ($3); tail:" >&2; tail -8 "$OUT/hanzo_${3}.log" >&2; }
}
run_llama() { # $1=phase(pp|tg) $2=n $3=tag
  [[ -x "$LLAMA" ]] || { echo '{"skipped":"no llama-bench"}' > "$OUT/llama_${3}.json"; return; }
  local p n; if [[ "$1" == pp ]]; then p="$2"; n=0; else p=0; n="$2"; fi
  echo ">> llama-bench $BACKEND $3 (p=$p n=$n r=$REPS)" >&2
  env $(backend_env) "$LLAMA" -m "$MODEL" -p "$p" -n "$n" -fa 1 -r "$REPS" -o json \
      > "$OUT/llama_${3}.json" 2> "$OUT/llama_${3}.err" || { echo "llama-bench FAILED ($3)" >&2; tail -5 "$OUT/llama_${3}.err" >&2; }
}

echo "=== dossier: $BACKEND / $MTAG -> $OUT ===" >&2
quiet_gate
manifest || echo "manifest step failed (non-fatal); continuing" >&2
# decode (one shared run), then each prefill size (ragged included via -p list)
run_hanzo tg "$NGEN" "${MTAG}_${BACKEND}_tg${NGEN}"
run_llama tg "$NGEN" "${MTAG}_${BACKEND}_tg${NGEN}"
IFS=',' read -ra PS <<< "$PLIST"
for P in "${PS[@]}"; do
  run_hanzo pp "$P" "${MTAG}_${BACKEND}_pp${P}"
  run_llama pp "$P" "${MTAG}_${BACKEND}_pp${P}"
done
echo "=== done: $OUT ===" >&2
echo "$OUT"
