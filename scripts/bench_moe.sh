#!/usr/bin/env bash
# bench_moe — server-driven MoE bench orchestrator (pure executor of bench_moe.toml).
#
# One arg = a config file. For each row of the sweep matrix it launches `hanzo
# serve` with the row's env + flags, proves the server is up, samples resident
# memory, drives the load generator (bench_client.py) for prefill/decode/steady/
# batch, reads the expert-cache + peak RSS off the live process, then terminates
# the server gracefully (SIGTERM by the recorded PID — never pkill / kill -9). A
# roofline probe runs once at the end; bench_report.py folds every row + the probe
# into one results.json plus a rendered table.
#
# Concerns are split, not braided: this file does lifecycle + wiring; bench_client
# drives load; roofline measures bandwidth; bench_report owns the schema + math.
#
# Build/run env (REQUIRED — the sandbox cc shim rejects -m64):
#   cd ~/work/hanzo/engine
#   PATH="/usr/bin:$PATH:$HOME/.cargo/bin:/opt/rocm/bin" \
#     CC=/usr/bin/gcc CXX=/usr/bin/g++ \
#     CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc \
#     cargo build --release -p hanzo --bin hanzo --features rocm
#   cargo build --release -p hanzo-bench --bin roofline --features rocm
# Runtime adds LD_LIBRARY_PATH=/opt/rocm/lib. Engine main is v1.7.37.
#
#   bench_moe.sh [-c config.toml] [-o out-dir] [--hanzo path] [--roofline path]
#                [--no-hash] [--dry-run]
#
# --dry-run wires manifest -> synthetic row -> report without launching a server,
# so the harness is self-testing on everything the ROCm serve path doesn't gate.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HERE/bench_moe.toml"
OUT=""
HANZO="${HANZO:-$HERE/../target/release/hanzo}"
ROOFLINE="${ROOFLINE:-$HERE/../target/release/roofline}"
LD_ROCM="${LD_LIBRARY_PATH:-/opt/rocm/lib}"
DRY_RUN=0
DO_HASH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) CONFIG="$2"; shift 2 ;;
    -o) OUT="$2"; shift 2 ;;
    --hanzo) HANZO="$2"; shift 2 ;;
    --roofline) ROOFLINE="$2"; shift 2 ;;
    --no-hash) DO_HASH=0; shift ;;
    --dry-run) DRY_RUN=1; DO_HASH=0; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -f "$CONFIG" ]] || { echo "no config: $CONFIG" >&2; exit 2; }
OUT="${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/bench_moe.XXXXXX")}"
mkdir -p "$OUT"
echo "out-dir: $OUT" >&2

# --- config accessors (python tomllib; no jq/toml shell dep) -------------------
cfg() { python3 -c "import tomllib,sys;d=tomllib.load(open(sys.argv[1],'rb'));print(d.get(sys.argv[2],''))" "$CONFIG" "$1"; }

MODEL="$(cfg model)"
PORT="$(cfg port)"; PORT="${PORT:-1234}"
WARMUP="$(cfg warmup)"; WARMUP="${WARMUP:-1}"
PROMPT_TOKENS="$(cfg prompt_tokens)"
MAX_TOKENS="$(cfg max_tokens)"
BATCH="$(cfg batch)"
MODEL_DIR="$(dirname "$MODEL")"
MODEL_FILE="$(basename "$MODEL")"

# --- reproducibility manifest --------------------------------------------------
write_manifest() {
  local sha="skipped"
  if [[ "$DO_HASH" == 1 && -f "$MODEL" ]]; then
    sha="$(sha256sum "$MODEL" | awk '{print $1}')"
  fi
  local gpu="unknown"
  if command -v rocminfo >/dev/null 2>&1; then
    gpu="$(rocminfo 2>/dev/null | grep -i 'gfx' | head -1 | awk '{print $NF}' || true)"
    gpu="${gpu:-unknown}"
  fi
  local rocm_ver="unknown"
  if command -v hipconfig >/dev/null 2>&1; then
    rocm_ver="$(hipconfig --version 2>/dev/null || echo unknown)"
  fi
  python3 - "$OUT/manifest.json" <<PY
import json, os, platform, subprocess, sys, time
def sh(*a):
    try: return subprocess.check_output(a, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
m = {
  "git_sha": sh("git","-C","$HERE","rev-parse","HEAD"),
  "host": platform.node(),
  "uname": " ".join(os.uname()),
  "gpu": "$gpu",
  "rocm_ver": "$rocm_ver",
  "model_path": "$MODEL",
  "model_sha256": "$sha",
  "build_features": "rocm",
  "serve_cmdline": "hanzo serve --port $PORT --isq <q> --max-seqs <n> --prefix-cache-n 0 -m $MODEL_DIR -f $MODEL_FILE --format gguf",
  "env": {k: os.environ.get(k) for k in ("STREAM_EXPERTS","STREAM_EXPERTS_RAM_GB","LD_LIBRARY_PATH")},
  "timestamp": time.time(),
}
json.dump(m, open(sys.argv[1],"w"), indent=2)
print("manifest ->", sys.argv[1], file=sys.stderr)
PY
}

# --- resident-memory sampler ---------------------------------------------------
rss_sampler() { # $1=pid $2=outfile ; VmRSS every 250ms while the pid lives
  local pid="$1" out="$2"
  while kill -0 "$pid" 2>/dev/null; do
    awk -v t="$(date +%s.%N)" '/VmRSS/{print t, $2}' "/proc/$pid/status" >> "$out" 2>/dev/null || true
    sleep 0.25
  done
}
peak_rss_gb() { # $1=pid -> VmHWM (kB) as GB, kernel high-water mark
  awk '/VmHWM/{printf "%.3f", $2/1048576}' "/proc/$1/status" 2>/dev/null || echo ""
}

# --- one sweep row -------------------------------------------------------------
run_row() { # $1=idx $2=quant $3=stream(0/1) $4=ram_gb $5=max_seqs
  local idx="$1" quant="$2" se="$3" ram_gb="$4" max_seqs="$5"
  local log="$OUT/row-$idx.serve.log" rss="$OUT/row-$idx.rss.tsv"
  local recs="$OUT/row-$idx.records.json" sysinfo="$OUT/row-$idx.sysinfo.json"

  local -a envp=(LD_LIBRARY_PATH="$LD_ROCM")
  [[ "$se" == 1 ]] && envp+=(STREAM_EXPERTS=1)
  [[ "$se" == 1 && -n "$ram_gb" ]] && envp+=(STREAM_EXPERTS_RAM_GB="$ram_gb")

  echo "[row $idx] quant=$quant stream=$se ram_gb=${ram_gb:-inf} max_seqs=$max_seqs" >&2
  local t_load0 t_load1
  t_load0="$(date +%s.%N)"
  env "${envp[@]}" "$HANZO" serve --port "$PORT" --isq "$quant" \
    --max-seqs "$max_seqs" --prefix-cache-n 0 \
    -m "$MODEL_DIR" -f "$MODEL_FILE" --format gguf > "$log" 2>&1 &
  local pid=$!

  # Ready-poll /health; bail if the process dies.
  until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "[row $idx] server exited early; see $log" >&2; return 1; }
    sleep 0.5
  done
  t_load1="$(date +%s.%N)"
  echo "[row $idx] ready in $(echo "$t_load1 - $t_load0" | bc)s (pid $pid)" >&2

  : > "$rss"
  rss_sampler "$pid" "$rss" &
  local sampler=$!

  # Unscored warmup, then the scored phases.
  local i
  for ((i=0; i<WARMUP; i++)); do
    curl -sf "http://localhost:$PORT/v1/chat/completions" \
      -H 'content-type: application/json' \
      -d '{"model":"default","messages":[{"role":"user","content":"warmup"}],"max_tokens":8,"temperature":0}' \
      >/dev/null 2>&1 || true
  done

  python3 "$HERE/bench_client.py" --base-url "http://localhost:$PORT" \
    --max-seqs "$max_seqs" --prompt-tokens "$PROMPT_TOKENS" \
    --max-tokens "$MAX_TOKENS" --batch "$BATCH" --out "$recs" >&2

  # Read the expert-cache + peak RSS off the live process before termination.
  curl -sf "http://localhost:$PORT/v1/system/info" -o "$sysinfo" 2>/dev/null || echo '{}' > "$sysinfo"
  local peak; peak="$(peak_rss_gb "$pid")"

  # Graceful terminate by the recorded PID; never pkill / kill -9.
  kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true

  [[ -f "$MODEL_DIR/.hanzo_experts_usage" ]] || echo "[row $idx] note: no .hanzo_experts_usage sidecar" >&2

  emit_row "$idx" "$quant" "$se" "$ram_gb" "$max_seqs" "$recs" "$sysinfo" "$rss" "$peak"
}

# --- fold one row's artifacts into row-<idx>.json ------------------------------
emit_row() { # idx quant se ram_gb max_seqs recs sysinfo rss peak
  python3 - "$@" <<'PY'
import json, sys
idx, quant, se, ram_gb, max_seqs, recs_p, sys_p, rss_p, peak = sys.argv[1:10]
def load(p, default):
    try: return json.load(open(p))
    except Exception: return default
records = load(recs_p, [])
sysinfo = load(sys_p, {})
ec = sysinfo.get("expert_cache")  # None on an engine without the readout surface
resident = ec.get("resident_gb") if ec else None
row = {
  "config": {"quant": quant, "stream_experts": se == "1",
             "ram_gb": (float(ram_gb) if ram_gb else None), "max_seqs": int(max_seqs)},
  "records": records,
  "peak_rss_gb": (float(peak) if peak else None),
  "resident_gb": resident,
  "expert_cache": ec,
}
out = f"{__import__('os').path.dirname(recs_p)}/row-{idx}.json"
json.dump(row, open(out, "w"), indent=2)
print("row ->", out, file=sys.stderr)
PY
}

# --- dry-run: synthesize a row + roofline, exercise the assembler --------------
dry_run() {
  echo "[dry-run] no server launch; wiring manifest -> synthetic row -> report" >&2
  python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
records = [
  {"phase":"prefill","config":{"prompt_tokens":512},"usage_prompt_tok_s":1800.0,"prompt_tokens":512},
  {"phase":"decode","config":{"max_tokens":256},"usage_compl_tok_s":95.0,"completion_tokens":256},
  {"phase":"steady","config":{"max_tokens":256},"usage_compl_tok_s":95.0,
   "itl":{"count":248,"p50_ms":10.4,"p90_ms":11.1,"p99_ms":13.0,"steady_tok_s":96.1}},
  {"phase":"batch","config":{"n":1,"max_tokens":256},"wall_s":2.7,"aggregate_tok_s":95.0,"per_stream_tok_s":95.0,"completion_tokens":256},
  {"phase":"batch","config":{"n":8,"max_tokens":256},"wall_s":5.0,"aggregate_tok_s":410.0,"per_stream_tok_s":51.0,"completion_tokens":2048},
]
# Illustrative resident-mode numbers (hit_rate 1.0 -> no NVMe miss path): with
# ~1.5 GB active weights/token served from RAM, the ceiling sits near ~120 t/s and
# the measured decode lands under it. Forced-small streamed budgets in a real run
# drive hit_rate down and expose the brutal NVMe-miss ceiling instead.
row = {"config":{"quant":"q4k","stream_experts":True,"ram_gb":4.0,"max_seqs":32},
       "records":records,"peak_rss_gb":9.8,"resident_gb":4.1,
       "expert_cache":{"hit_rate":1.0,"active_weight_bytes":1.5e9,
                       "resident_dense_bytes":4.0e8,"kv_read_bytes":5.0e7}}
json.dump(row, open(os.path.join(out,"row-0.json"),"w"), indent=2)
roof = {"ram_bw_gbs":{"single_thread":41.0,"all_thread":248.0,"threads":32,"buffer_gb":2.0,"reps":5},
        "nvme_bw_gbs":{"value":3.31,"direct":True,"bytes_read":4294967296,"chunk_mb":4,"file":"model.gguf"}}
json.dump(roof, open(os.path.join(out,"roofline.json"),"w"), indent=2)
print("[dry-run] wrote synthetic row-0.json + roofline.json", file=sys.stderr)
PY
}

# --- main ----------------------------------------------------------------------
write_manifest

if [[ "$DRY_RUN" == 1 ]]; then
  dry_run
else
  [[ -x "$HANZO" ]] || { echo "no hanzo binary: $HANZO (build it first)" >&2; exit 2; }
  idx=0
  while IFS=$'\t' read -r quant se ram_gb max_seqs; do
    [[ -z "$quant" ]] && continue
    run_row "$idx" "$quant" "$se" "$ram_gb" "$max_seqs" || echo "[row $idx] failed" >&2
    idx=$((idx + 1))
  done < <(python3 -c "import tomllib,sys
d=tomllib.load(open('$CONFIG','rb'))
for c in d.get('configs',[]):
    print('\t'.join([c.get('quant','q4k'),'1' if c.get('stream_experts') else '0',str(c.get('ram_gb','')),str(c.get('max_seqs',32))]))")

  if [[ -x "$ROOFLINE" ]]; then
    echo "roofline probe ..." >&2
    LD_LIBRARY_PATH="$LD_ROCM" "$ROOFLINE" --file "$MODEL" > "$OUT/roofline.json" 2>>"$OUT/roofline.log" \
      || echo "roofline probe failed; see $OUT/roofline.log" >&2
  else
    echo "no roofline binary: $ROOFLINE (skipping probe)" >&2
  fi
fi

python3 "$HERE/bench_report.py" --in-dir "$OUT"
echo "done: $OUT" >&2
