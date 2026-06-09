#!/usr/bin/env bash
# Perf-regression guards for the three hard-won speedups. Each guard runs the REAL workload on the
# correct box, parses the measured number, and asserts it stays above a floor set with margin below
# the known-good number (and well above the regressed number). A guard FAILS loudly (measured vs
# floor) if perf regresses. Exit code is non-zero if ANY selected guard fails.
#
# GUARDS (see PROVENANCE in README.md for the commits + measured baselines):
#   1. cuda-prefill  (spark/GB10): Qwen3-8B-Q8_0 pp512 t/s. Fix = naive.rs maybe_synchronize skips
#                    the per-attention-layer sysinfo (MemoryUsage::query / System::new_all) stall on
#                    unified-memory CUDA. Known good ~1485 (regressed ~120). FLOOR 1000.
#   2. vulkan-mmq    (evo/8060S, native Windows Vulkan): Qwen3-8B-Q8_0 pp512 routes through the
#                    mul_mm_q8_mmq GEMM (llama RDNA3 warp-tile, 1.95x over dp4a). Known good ~210
#                    (dp4a ~113). FLOOR: route fires (mul_mm_q8_mmq in VK_PROFILE) AND t/s > 150.
#   3. musetalk-fps  (spark/GB10): MuseTalk framework-only single-frame fps (f16). Keystones =
#                    pinned async-mempool threshold + f16-native GroupNorm + fused conv-bias.
#                    Known good ~4.5 (regressed/baseline 2.51). FLOOR 3.5.
#
# Usage:
#   run_perf_guards.sh [--box spark|evo|all] [--guard cuda-prefill|vulkan-mmq|musetalk-fps] ...
#   (default: --box all -- runs every guard whose box is reachable; skips+reports unreachable ones)
set -uo pipefail

# ---- Floors (the regression thresholds). Real measured baselines are in README.md. ----
CUDA_PREFILL_FLOOR="${CUDA_PREFILL_FLOOR:-1000}"   # t/s; known-good ~1485, regressed ~120
VULKAN_MMQ_FLOOR="${VULKAN_MMQ_FLOOR:-150}"        # t/s; known-good ~210, dp4a-fallback ~113
MUSETALK_FPS_FLOOR="${MUSETALK_FPS_FLOOR:-3.5}"    # fps; known-good ~4.5-5.0, baseline 2.51

# ---- Box / workload locations (override via env). ----
SPARK_HOST="${SPARK_HOST:-spark}"
SPARK_ENGINE_DIR="${SPARK_ENGINE_DIR:-/home/z/work/sw-perf/engine}"
SPARK_MUSETALK_DIR="${SPARK_MUSETALK_DIR:-/home/z/work/sw-perf/musetalk-bench}"
SPARK_GGUF_DIR="${SPARK_GGUF_DIR:-/home/z/models}"
SPARK_GGUF_FILE="${SPARK_GGUF_FILE:-Qwen3-8B-Q8_0.gguf}"

# evo native Windows hanzo.exe + gguf (paths are Windows paths used by the .exe; the script invokes
# it through cmd.exe so stderr/stdout is captured cleanly -- PowerShell mis-flags engine stderr).
EVO_HANZO_EXE="${EVO_HANZO_EXE:-C:\\Users\\z\\work\\hanzo-native\\engine\\target\\quick\\hanzo.exe}"
EVO_GGUF_DIR="${EVO_GGUF_DIR:-C:\\llama}"
EVO_GGUF_FILE="${EVO_GGUF_FILE:-Qwen3-8B-Q8_0.gguf}"

# Bench knobs.
PROMPT_LEN="${PROMPT_LEN:-512}"
ITERS="${ITERS:-5}"          # >=4 keeps the high-variance pp512 mean stable
WARMUP="${WARMUP:-1}"
MUSETALK_ITERS="${MUSETALK_ITERS:-40}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; BOLD=$'\033[1m'; RST=$'\033[0m'
PASS_N=0; FAIL_N=0; SKIP_N=0
RESULTS=()

log()  { printf '%s\n' "$*" >&2; }
hdr()  { log ""; log "${BOLD}=== $* ===${RST}"; }

# float compare: ok if $1 (measured) >= $2 (floor)
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 >= b+0)}'; }

record_pass() { PASS_N=$((PASS_N+1)); RESULTS+=("${GRN}PASS${RST}  $1"); log "${GRN}PASS${RST}  $1"; }
record_fail() { FAIL_N=$((FAIL_N+1)); RESULTS+=("${RED}FAIL${RST}  $1"); log "${RED}${BOLD}FAIL${RST}  $1"; }
record_skip() { SKIP_N=$((SKIP_N+1)); RESULTS+=("${YEL}SKIP${RST}  $1"); log "${YEL}SKIP${RST}  $1"; }

reachable_spark() { timeout 12 ssh -o ConnectTimeout=8 -o BatchMode=yes "$SPARK_HOST" true >/dev/null 2>&1; }
# evo = this box; native build is reachable iff cmd.exe + the exe exist.
reachable_evo() {
  command -v cmd.exe >/dev/null 2>&1 || return 1
  local winexe="${EVO_HANZO_EXE//\\//}"   # C:\..  -> C:/..
  winexe="/mnt/c/${winexe#C:/}"
  [ -f "$winexe" ]
}

# Pull the prefill t/s out of the hanzo bench result table. The row reads:
#   | Prefill (512 tokens)  | 1485.0 +- 30.0 | 394.00 ms (TTFT) |
# Strip ANSI, find the Prefill row, take the first float after the test-name cell.
parse_prefill_tps() {
  sed -E 's/\x1b\[[0-9;]*m//g' \
    | grep -E "Prefill \(${PROMPT_LEN} tokens\)" \
    | grep -oE '[0-9]+\.[0-9]+' | head -1
}

# Pull the framework-only fps(mean) from the musetalk-bench output. The section is:
#   -- framework-only (full VAE enc + UNet + full VAE dec) --
#   ...
#   fps(mean):          5.02
# Take the FIRST fps(mean) (framework-only block prints before COMBINED).
parse_musetalk_fps() {
  sed -E 's/\x1b\[[0-9;]*m//g' \
    | grep -E 'fps\(mean\):' | head -1 \
    | grep -oE '[0-9]+\.[0-9]+' | head -1
}

# ---------------------------------------------------------------------------------------------
guard_cuda_prefill() {
  hdr "GUARD 1/3  cuda-prefill  (spark GB10, Qwen3-8B-Q8_0 pp${PROMPT_LEN})"
  if ! reachable_spark; then
    record_skip "cuda-prefill: spark ($SPARK_HOST) unreachable"; return
  fi
  local out tps
  out=$(timeout 600 ssh "$SPARK_HOST" \
    "cd '$SPARK_ENGINE_DIR' && ./target/release/hanzo bench --format gguf -m '$SPARK_GGUF_DIR' \
     -f '$SPARK_GGUF_FILE' --dtype auto --prompt-len $PROMPT_LEN --gen-len 0 \
     --iterations $ITERS --warmup $WARMUP 2>&1" 2>&1)
  tps=$(printf '%s' "$out" | parse_prefill_tps)
  if [ -z "$tps" ]; then
    log "$(printf '%s' "$out" | tail -15)"
    record_fail "cuda-prefill: could not parse Prefill t/s from bench output (workload failed?)"; return
  fi
  log "  measured pp${PROMPT_LEN} = ${BOLD}${tps} t/s${RST}   floor = ${CUDA_PREFILL_FLOOR} t/s"
  if ge "$tps" "$CUDA_PREFILL_FLOOR"; then
    record_pass "cuda-prefill: ${tps} t/s  >= ${CUDA_PREFILL_FLOOR} (known-good ~1485, regressed ~120)"
  else
    record_fail "cuda-prefill: ${tps} t/s  <  ${CUDA_PREFILL_FLOOR}  -- sysinfo-stall regression suspected (naive.rs maybe_synchronize)"
  fi
}

# ---------------------------------------------------------------------------------------------
guard_vulkan_mmq() {
  hdr "GUARD 2/3  vulkan-mmq  (evo 8060S native Vulkan, Qwen3-8B-Q8_0 pp${PROMPT_LEN})"
  if ! reachable_evo; then
    record_skip "vulkan-mmq: evo native Vulkan exe not found ($EVO_HANZO_EXE) -- needs the native Windows build"
    return
  fi
  # VK_PROFILE=1 makes the engine print a per-op census ("ops: ... mul_mm_q8_mmq=N ...") to stderr;
  # we assert the mmq kernel actually fired for the dense Q8 prefill AND that pp512 cleared the floor.
  # Run through a generated .bat (one statement per line) -- a single `cmd /c "set X=Y&& exe %X%"`
  # line expands %X% at parse time, BEFORE the set, so inline env vars never take effect. cmd captures
  # stdout+stderr together (engine writes the table to stdout, profile/tracing to stderr); PowerShell
  # would mis-treat the stderr as a NativeCommandError.
  local bat_win bat_wsl log_win log_wsl
  bat_win="C:\\Users\\z\\perfguard-vk-mmq.bat"; bat_wsl="/mnt/c/Users/z/perfguard-vk-mmq.bat"
  log_win="C:\\Users\\z\\perfguard-vk-mmq.log"; log_wsl="/mnt/c/Users/z/perfguard-vk-mmq.log"
  rm -f "$log_wsl" 2>/dev/null
  {
    printf '@echo off\r\n'
    printf 'set HANZO_VK_Q8_PREFILL=mmq\r\n'
    printf 'set VK_PROFILE=1\r\n'
    printf '"%s" bench --format gguf -m "%s" -f "%s" --dtype auto --prompt-len %s --gen-len 0 --iterations %s --warmup %s\r\n' \
      "$EVO_HANZO_EXE" "$EVO_GGUF_DIR" "$EVO_GGUF_FILE" "$PROMPT_LEN" "$ITERS" "$WARMUP"
  } > "$bat_wsl"
  cmd.exe /c "$bat_win > $log_win 2>&1" >/dev/null 2>&1
  if [ ! -f "$log_wsl" ]; then
    record_fail "vulkan-mmq: bench produced no output ($log_wsl missing)"; return
  fi
  local out tps routed
  out=$(cat "$log_wsl")
  tps=$(printf '%s' "$out" | parse_prefill_tps)
  routed=$(printf '%s' "$out" | sed -E 's/\x1b\[[0-9;]*m//g' | grep -c 'mul_mm_q8_mmq')
  log "  routed-to-mmq dispatches seen in VK_PROFILE: ${routed}"
  if [ -z "$tps" ]; then
    log "$(printf '%s' "$out" | tail -20)"
    record_fail "vulkan-mmq: could not parse Prefill t/s (workload failed? OOM?)"; return
  fi
  log "  measured pp${PROMPT_LEN} = ${BOLD}${tps} t/s${RST}   floor = ${VULKAN_MMQ_FLOOR} t/s"
  if [ "$routed" -lt 1 ]; then
    record_fail "vulkan-mmq: mul_mm_q8_mmq did NOT appear in VK_PROFILE -- Q8 prefill is NOT routing to the MMQ kernel"
    return
  fi
  if ge "$tps" "$VULKAN_MMQ_FLOOR"; then
    record_pass "vulkan-mmq: routed to mul_mm_q8_mmq AND ${tps} t/s >= ${VULKAN_MMQ_FLOOR} (known-good ~210, dp4a ~113)"
  else
    record_fail "vulkan-mmq: routed to mul_mm_q8_mmq but ${tps} t/s < ${VULKAN_MMQ_FLOOR} -- MMQ GEMM regressed"
  fi
}

# ---------------------------------------------------------------------------------------------
guard_musetalk_fps() {
  hdr "GUARD 3/3  musetalk-fps  (spark GB10, framework-only single-frame f16)"
  if ! reachable_spark; then
    record_skip "musetalk-fps: spark ($SPARK_HOST) unreachable"; return
  fi
  local out fps
  out=$(timeout 600 ssh "$SPARK_HOST" \
    "cd '$SPARK_MUSETALK_DIR' && MUSETALK_DEV=cuda MUSETALK_DTYPE=f16 MUSETALK_ITERS=$MUSETALK_ITERS \
     ./target/release/musetalk-bench bench 2>&1" 2>&1)
  fps=$(printf '%s' "$out" | parse_musetalk_fps)
  if [ -z "$fps" ]; then
    log "$(printf '%s' "$out" | tail -15)"
    record_fail "musetalk-fps: could not parse framework-only fps(mean) (workload failed?)"; return
  fi
  log "  measured framework-only fps(mean) = ${BOLD}${fps}${RST}   floor = ${MUSETALK_FPS_FLOOR}"
  if ge "$fps" "$MUSETALK_FPS_FLOOR"; then
    record_pass "musetalk-fps: ${fps} fps >= ${MUSETALK_FPS_FLOOR} (known-good ~4.5, baseline 2.51)"
  else
    record_fail "musetalk-fps: ${fps} fps < ${MUSETALK_FPS_FLOOR} -- keystone regression (mempool/GroupNorm/conv-bias)"
  fi
}

# ---------------------------------------------------------------------------------------------
main() {
  local box="all"; local guards=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --box) box="$2"; shift 2 ;;
      --guard) guards+=("$2"); shift 2 ;;
      -h|--help) grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
      *) log "unknown arg: $1"; exit 2 ;;
    esac
  done
  if [ "${#guards[@]}" -eq 0 ]; then
    case "$box" in
      spark) guards=(cuda-prefill musetalk-fps) ;;
      evo)   guards=(vulkan-mmq) ;;
      all|*) guards=(cuda-prefill vulkan-mmq musetalk-fps) ;;
    esac
  fi
  log "${BOLD}perf-regression guards${RST}  (box=$box)  guards: ${guards[*]}"
  for g in "${guards[@]}"; do
    case "$g" in
      cuda-prefill) guard_cuda_prefill ;;
      vulkan-mmq)   guard_vulkan_mmq ;;
      musetalk-fps) guard_musetalk_fps ;;
      *) log "unknown guard: $g"; exit 2 ;;
    esac
  done
  hdr "SUMMARY"
  for r in "${RESULTS[@]}"; do log "  $r"; done
  log ""
  log "${BOLD}${PASS_N} passed, ${FAIL_N} failed, ${SKIP_N} skipped${RST}"
  [ "$FAIL_N" -eq 0 ]
}
main "$@"
