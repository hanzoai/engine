#!/bin/sh
# TDD for shim.sh. Stubs `gh` on PATH so every case runs offline. Asserts the
# behaviour that matters: newest-tag selection, the make_latest-safe skip of
# superseded gaps, dispatch on a dropped tag, idempotence, dry-run, and the
# ml `--ref` / publish-run path.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
SHIM="$HERE/shim.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# --- fake gh: scenario-driven, offline -------------------------------------
BIN="$TMP/bin"; mkdir -p "$BIN"
cat >"$BIN/gh" <<'GH'
#!/bin/sh
case "$1 $2" in
  "api "*)          printf '%s\n' "${FAKE_TAGS:-}" ;;         # repos/<r>/tags
  "release view")   tag=$3
                    for t in ${FAKE_RELEASES:-}; do [ "$t" = "$tag" ] && exit 0; done
                    exit 1 ;;
  "run list")       printf '%s\n' "${FAKE_RUN_BRANCHES:-}" ;;
  "workflow run")   shift; printf 'workflow run %s\n' "$*" >>"$GH_CALLS" ;;
  *) : ;;
esac
GH
chmod +x "$BIN/gh"
export PATH="$BIN:$PATH"

GH_CALLS="$TMP/calls"; export GH_CALLS

pass=0; fail=0
ok()   { pass=$((pass+1)); printf '  ok   %s\n' "$1"; }
bad()  { fail=$((fail+1)); printf '  FAIL %s\n' "$1"; }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (want [$3] got [$2])"; fi; }

# run one scenario: RESET state + calls, invoke shim with injected fixtures
run() {
  desc=$1; shift
  : >"$GH_CALLS"
  STATE="$TMP/state.$pass.$fail"; rm -rf "$STATE"
  env HANZO_SHIM_STATE="$STATE" HANZO_SHIM_REPOS="$ROW" \
      FAKE_TAGS="$FAKE_TAGS" FAKE_RELEASES="${FAKE_RELEASES:-}" \
      FAKE_RUN_BRANCHES="${FAKE_RUN_BRANCHES:-}" GH_CALLS="$GH_CALLS" \
      sh "$SHIM" "$@" >"$TMP/out" 2>&1 || { bad "$desc: shim exited nonzero"; cat "$TMP/out"; }
}
calls() { cat "$GH_CALLS"; }
ncalls(){ [ -s "$GH_CALLS" ] && grep -c . "$GH_CALLS" || echo 0; }

echo "== shim_test =="

# T1 -- newest_tag picks highest SEMVER, not lexical; ignores non-v / ml-v tags.
# shellcheck source=/dev/null
( HANZO_SHIM_LIB=1 . "$SHIM"
  FAKE_TAGS="v1.7.9
v1.7.89
v1.7.91
v1.7.54
ml-v0.11.93
nightly"
  export FAKE_TAGS
  got=$(newest_tag hanzoai/engine)
  [ "$got" = "v1.7.91" ] && echo T1PASS || echo "T1FAIL:$got"
) >"$TMP/t1"; check "T1 newest_tag = highest semver" "$(cat "$TMP/t1")" "T1PASS"

# T2 -- newest tag already released -> no dispatch (steady state).
ROW='hanzoai/engine|release.yml|input|ci.yml|release'
FAKE_TAGS="v1.7.90
v1.7.91"; FAKE_RELEASES="v1.7.91 v1.7.90"; FAKE_RUN_BRANCHES=""
run "T2" --once
check "T2 released newest -> 0 dispatches" "$(ncalls)" "0"

# T3 -- newest tag has NO release -> dispatch release (-f version=) + ci (--ref main).
FAKE_TAGS="v1.7.91
v1.7.92"; FAKE_RELEASES="v1.7.91"; FAKE_RUN_BRANCHES=""
run "T3" --once
check "T3 dropped tag -> 2 dispatches" "$(ncalls)" "2"
if calls | grep -q 'release.yml .* version=v1.7.92'; then ok "T3 release dispatch carries version=v1.7.92"; else bad "T3 release dispatch"; fi
if calls | grep -q 'ci.yml .* --ref main'; then ok "T3 ci dispatch on main"; else bad "T3 ci dispatch"; fi

# T4 -- make_latest safety: superseded gaps (v1.7.89, v1.7.54 unreleased) below a
# released newest (v1.7.91) must NOT be dispatched (would flip Latest backward).
FAKE_TAGS="v1.7.54
v1.7.89
v1.7.90
v1.7.91"; FAKE_RELEASES="v1.7.91 v1.7.90"; FAKE_RUN_BRANCHES=""
run "T4" --once
check "T4 superseded gaps -> 0 dispatches" "$(ncalls)" "0"

# T5 -- idempotence: a 2nd pass over the SAME still-unreleased tag does not re-fire
# (state latch covers the window before the release row exists). Reuse one STATE.
: >"$GH_CALLS"; S5="$TMP/state5"; rm -rf "$S5"
FAKE_TAGS="v1.7.91
v1.7.92"; FAKE_RELEASES="v1.7.91"
env HANZO_SHIM_STATE="$S5" HANZO_SHIM_REPOS="$ROW" FAKE_TAGS="$FAKE_TAGS" \
    FAKE_RELEASES="$FAKE_RELEASES" FAKE_RUN_BRANCHES="" GH_CALLS="$GH_CALLS" sh "$SHIM" --once >/dev/null 2>&1
first=$(ncalls)
env HANZO_SHIM_STATE="$S5" HANZO_SHIM_REPOS="$ROW" FAKE_TAGS="$FAKE_TAGS" \
    FAKE_RELEASES="$FAKE_RELEASES" FAKE_RUN_BRANCHES="" GH_CALLS="$GH_CALLS" sh "$SHIM" --once >/dev/null 2>&1
second=$(ncalls)
check "T5 first pass dispatched" "$first" "2"
check "T5 second pass no re-dispatch (still 2 total)" "$second" "2"

# T6 -- dry-run dispatches nothing even when a tag is un-released.
FAKE_TAGS="v1.7.91
v1.7.92"; FAKE_RELEASES="v1.7.91"; FAKE_RUN_BRANCHES=""
run "T6" --once --dry-run
check "T6 dry-run -> 0 real dispatches" "$(ncalls)" "0"

# T7 -- ml path: run-style handled + --ref dispatch (not -f version).
ROW='hanzoai/ml|publish.yml|ref|rust-ci.yml|run'
# 7a: newest tag already has a publish run -> handled -> no dispatch.
FAKE_TAGS="v0.11.29
v0.11.30"; FAKE_RELEASES=""; FAKE_RUN_BRANCHES="v0.11.30"
run "T7a" --once
check "T7a ml publish-run exists -> 0 dispatches" "$(ncalls)" "0"
# 7b: no publish run -> dispatch publish.yml at --ref <tag> + ci.
FAKE_RUN_BRANCHES=""
run "T7b" --once
check "T7b ml no run -> 2 dispatches" "$(ncalls)" "2"
if calls | grep -q 'publish.yml .* --ref v0.11.30'; then ok "T7b ml dispatch uses --ref (not version=)"; else bad "T7b ml --ref dispatch"; fi
if calls | grep -q 'version='; then bad "T7b ml wrongly used version="; else ok "T7b ml did not use version="; fi

echo "== $pass passed, $fail failed =="
[ "$fail" = 0 ]
