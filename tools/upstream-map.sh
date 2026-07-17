#!/usr/bin/env bash
# Route an upstream (EricLBuehler/mistral.rs) path or commit onto our tree.
#
# We forked at 7483b396f (2024-02-27) and renamed every crate, so `git log` cannot follow a change
# across the boundary: rename detection finds ONE rename between the merge-base and HEAD because the
# files diverged far past any similarity threshold. That makes upstream unreadable-by-tooling and is
# why "what did they do here?" is hard to answer. This is the missing dictionary.
#
# This exists to STUDY upstream, not to merge it. A wholesale merge is a measured downgrade: on an
# M4 Max, identical Qwen3-30B-A3B Q4_K_M GGUF and identical bench flags, upstream decodes at 0.20 t/s
# (Metal-mapped, one CPU core pinned -- a host-side per-expert MoE loop) against our 94.4. Their tree
# also has zero Vulkan and zero ROCm. Harvest correctness fixes and new architectures; never their
# MoE/expert dispatch or perf work.
#
#   ./tools/upstream-map.sh path  mistralrs-core/src/foo.rs   # -> our path
#   ./tools/upstream-map.sh show  <upstream-sha>              # what a commit touches, routed to ours
#   ./tools/upstream-map.sh diff  <subsystem>                 # their file vs ours, side by side
#   ./tools/upstream-map.sh harvest                           # correctness commits we may not have
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

UP=upstream/master

# The dictionary. Mostly mistralrs-X -> hanzo-X; `core` is the exception that matters most.
map_path() {
  sed -E \
    -e 's|^mistralrs-core/|hanzo-engine/|' \
    -e 's|^mistralrs-cli/|hanzo-cli/|' \
    -e 's|^mistralrs-server-core/|hanzo-server-core/|' \
    -e 's|^mistralrs-quant/|hanzo-quant/|' \
    -e 's|^mistralrs-vision/|hanzo-vision/|' \
    -e 's|^mistralrs-audio/|hanzo-audio/|' \
    -e 's|^mistralrs-paged-attn/|hanzo-paged-attn/|' \
    -e 's|^mistralrs-flash-attn/|hanzo-flash-attn/|' \
    -e 's|^mistralrs-code-exec/|hanzo-code-exec/|' \
    -e 's|^mistralrs-mcp/|hanzo-llm-mcp/|' \
    -e 's|^mistralrs-macros/|hanzo-macros/|' \
    -e 's|^mistralrs-bench/|hanzo-bench/|' \
    -e 's|^mistralrs/|hanzo/|'
}

case "${1:-}" in
  path) echo "$2" | map_path ;;

  show)
    sha="$2"
    git log -1 --format='%h %ad %s' --date=short "$sha"
    echo
    git show --stat --format= "$sha" | sed -E 's/\|.*//' | awk 'NF' | while read -r f; do
      ours=$(echo "$f" | map_path)
      if [ -e "$ours" ]; then st="EXISTS"; else st="ABSENT (we deleted/rewrote it -> likely a NO-OP)"; fi
      printf '  %-52s -> %-52s %s\n' "$f" "$ours" "$st"
    done
    ;;

  diff)
    sub="$2"
    their=$(git ls-tree -r --name-only "$UP" | grep -i "$sub" | head -1)
    [ -n "$their" ] || { echo "no upstream file matching '$sub'"; exit 1; }
    ours=$(echo "$their" | map_path)
    echo "theirs: $their"
    echo "ours:   $ours $([ -e "$ours" ] || echo '(ABSENT)')"
    [ -e "$ours" ] || exit 0
    diff -u <(git show "$UP:$their") "$ours" || true
    ;;

  harvest)
    # Correctness-signal commits since the fork. Perf/MoE deliberately excluded: we are ~470x ahead
    # of their MoE path, so importing it is the regression, not the fix.
    git fetch upstream --quiet
    git log --oneline HEAD.."$UP" \
      --grep='fix' --grep='correctness' --grep='regression' --grep='panic' --grep='unsound' \
      --grep='overflow' --grep='incorrect' --grep='wrong' --grep='leak' --grep='race' \
      --regexp-ignore-case \
      | grep -viE 'moe|expert|perf|speed|faster|optimi' \
      | head -60
    echo
    echo "Run monthly. Route each with: $0 show <sha>. ABSENT => we rewrote that code; skip."
    ;;

  *) sed -n '2,26p' "$0"; exit 1 ;;
esac
