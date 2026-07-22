#!/bin/sh
# =============================================================================
# TEMPORARY -- tag -> release trigger shim.
#
# RETIRE THIS when the native forge (git.hanzo.ai) becomes the canonical remote
# (migration Phase B). It exists ONLY to work around a GitHub Actions fault where
# push / tag-push events fire zero workflow runs (batched tag-push events get
# dropped), so a `git push --tags` can land a `vX.Y.Z` tag that never triggers
# release.yml. v1.7.90 / v1.7.91 shipped only because someone ran
# `gh workflow run` by hand. This shim automates exactly that recovery.
#
# WHAT IT DOES, once per run (`--once`, meant for a 10-min cron / systemd timer):
#   for each configured repo, look at the NEWEST vX.Y.Z tag; if that tag has no
#   release yet AND we have not already dispatched it, fire
#     gh workflow run <release-wf> (recover the release)
#     gh workflow run <ci-wf> --ref main (recover the paired main CI run)
#
# WHY "newest tag only" (not "every un-released tag"): release.yml sets
# `make_latest: true`, so re-dispatching an OLD superseded tag (e.g. v1.7.54,
# whose push-event was also dropped) would wrongly flip the "Latest" release
# pointer BACKWARD. The newest tag is the only one that both wants a release and
# is safe to mark latest. Superseded gaps are reported, never auto-released.
#
# IDEMPOTENT -- never double-fires a tag:
#   1. per-repo state file latches a tag the moment it is dispatched (covers the
#      window before the release row exists), and
#   2. `gh release view` (engine) / an existing publish run (ml) is the remote
#      truth once the target workflow has started.
#
# DEPENDENCIES: gh (authenticated), sort -V, standard POSIX sh. No repo clone
# needed -- everything is a `gh api` / `gh release` / `gh run` call.
#
# DEPLOY: this repo cannot self-host it (the GitHub event fault is the very thing
# it routes around). Run it from any box with an authenticated `gh` on a short
# timer. See hanzo-release-shim.{service,timer} for the systemd unit, or:
#   */10 * * * * /path/to/shim.sh --once >>/var/log/hanzo-release-shim.log 2>&1
# =============================================================================
set -eu

# --- config -----------------------------------------------------------------
# One row per repo: repo | release-workflow | dispatch-style | ci-workflow | handled-style
#   dispatch-style: input  -> release-wf takes `-f version=<tag>` (engine release.yml)
#                   ref    -> release-wf runs at `--ref <tag>` (ml publish.yml)
#   handled-style:  release-> a GitHub Release exists for the tag (engine)
#                   run    -> a release-wf run exists for the tag ref (ml, crates.io)
# Overridable via HANZO_SHIM_REPOS (used by the tests to inject a single repo).
#
# ml is supported by the code (handled-style `run`, `--ref` dispatch; see the T7
# tests) but is OFF by default: ml releases to crates.io, not GitHub Releases, and
# its tags carry two schemes (`v0.11.x` and `ml-v0.11.x`) while publish.yml still
# triggers on a bare `N.N.N` tag that matches NEITHER. Reconcile ml's canonical
# tag scheme + publish.yml's trigger, then add the row below (cargo's
# already-uploaded skip is the publish-side idempotence backstop):
#   hanzoai/ml|publish.yml|ref|rust-ci.yml|run
DEFAULT_REPOS='hanzoai/engine|release.yml|input|ci.yml|release'
REPOS="${HANZO_SHIM_REPOS:-$DEFAULT_REPOS}"

STATE_DIR="${HANZO_SHIM_STATE:-${XDG_STATE_HOME:-$HOME/.local/state}/hanzo-release-shim}"
TAG_GLOB='^v[0-9]'   # vX.Y.Z scheme; ml's `ml-v*` tags are intentionally ignored
DRY_RUN=0
ONLY_REPO=''

log() { printf '%s shim: %s\n' "$(date -u +%H:%M:%S)" "$*"; }

usage() {
  # Print the header banner (line 2 through the closing `# ===` rule), de-commented.
  sed -n '2,/^# ===/p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

Usage: shim.sh [--once] [--dry-run] [--repo <owner/name>]
  --once      single reconcile pass (the default; for cron / systemd timer)
  --dry-run   log the decisions and the exact gh commands, dispatch nothing
  --repo R    restrict to one configured repo (default: all)
USAGE
}

# --- primitives (thin gh wrappers; the tests stub `gh`) ---------------------

# Highest vX.Y.Z tag in the repo, or empty. `sort -V` orders 1.7.9 < 1.7.89 < 1.7.91.
newest_tag() {
  gh api "repos/$1/tags" --paginate -q '.[].name' 2>/dev/null \
    | grep -E "$TAG_GLOB" | sort -V | tail -1
}

# 0 if the tag's release already exists (or its release-wf already ran for it).
# POSIX sh has no function scope: every param var is `_`-prefixed and private so
# it can never clobber a caller's variable (repo/rel_wf/style/... in process_repo).
tag_handled() {
  _repo=$1; _tag=$2; _hstyle=$3; _relwf=$4
  case "$_hstyle" in
    release)
      gh release view "$_tag" -R "$_repo" >/dev/null 2>&1 ;;
    run)
      # A publish run whose head ref is this tag means it already fired.
      gh run list --workflow "$_relwf" -R "$_repo" -L 50 \
        --json headBranch -q '.[].headBranch' 2>/dev/null \
        | grep -qx "$_tag" ;;
    *) return 1 ;;
  esac
}

# state-file latch: prevents a re-dispatch in the window before the release row exists.
state_file() { printf '%s/%s.dispatched' "$STATE_DIR" "$(echo "$1" | tr '/' '_')"; }
tag_dispatched() { _f=$(state_file "$1"); [ -f "$_f" ] && grep -qx "$2" "$_f"; }
record_dispatch() { _f=$(state_file "$1"); mkdir -p "$STATE_DIR"; printf '%s\n' "$2" >>"$_f"; }

dispatch_release() {
  _repo=$1; _wf=$2; _dstyle=$3; _tag=$4
  if [ "$_dstyle" = input ]; then
    set -- gh workflow run "$_wf" -R "$_repo" -f "version=$_tag"
  else
    set -- gh workflow run "$_wf" -R "$_repo" --ref "$_tag"
  fi
  log "dispatch release: $*"
  [ "$DRY_RUN" = 1 ] && return 0
  "$@"
}

dispatch_ci() {
  _repo=$1; _wf=$2
  log "dispatch ci: gh workflow run $_wf -R $_repo --ref main"
  [ "$DRY_RUN" = 1 ] && return 0
  gh workflow run "$_wf" -R "$_repo" --ref main
}

# --- one repo ----------------------------------------------------------------
process_repo() {
  repo=$1; rel_wf=$2; style=$3; ci_wf=$4; handled=$5
  tag=$(newest_tag "$repo") || tag=''
  if [ -z "$tag" ]; then log "$repo: no ${TAG_GLOB} tags -- nothing to do"; return 0; fi

  if tag_dispatched "$repo" "$tag"; then
    log "$repo: newest tag $tag already dispatched (state latch) -- skip"; return 0
  fi
  if tag_handled "$repo" "$tag" "$handled" "$rel_wf"; then
    log "$repo: newest tag $tag already released -- up to date"; return 0
  fi

  log "$repo: newest tag $tag has NO release -- recovering (dropped tag-push event)"
  dispatch_release "$repo" "$rel_wf" "$style" "$tag"
  dispatch_ci "$repo" "$ci_wf"
  [ "$DRY_RUN" = 1 ] || record_dispatch "$repo" "$tag"
}

# --- main --------------------------------------------------------------------
main() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --once) : ;;
      --dry-run) DRY_RUN=1 ;;
      --repo) ONLY_REPO=$2; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
  done

  command -v gh >/dev/null 2>&1 || { echo "shim: gh not found on PATH" >&2; exit 127; }

  printf '%s\n' "$REPOS" | while IFS='|' read -r repo rel_wf style ci_wf handled; do
    [ -z "${repo:-}" ] && continue
    [ -n "$ONLY_REPO" ] && [ "$repo" != "$ONLY_REPO" ] && continue
    process_repo "$repo" "$rel_wf" "$style" "$ci_wf" "$handled"
  done
}

# Sourced by shim_test.sh with HANZO_SHIM_LIB=1 to exercise the functions in
# isolation; run directly otherwise.
[ "${HANZO_SHIM_LIB:-0}" = 1 ] || main "$@"
