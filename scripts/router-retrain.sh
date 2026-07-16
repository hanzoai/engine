#!/usr/bin/env bash
# Nightly router-heads retrain, one scope per invocation. Spark-local: the engine
# `fit` is built + run natively here; only the ledger pull and the meta publish
# cross the network. One mechanism, N scopes:
#   router-retrain.sh --scope global        # opted-in orgs -> heads-base.safetensors  (owner "*")
#   router-retrain.sh --scope org=<slug>    # that org's rows -> heads-<slug>.safetensors (owner "<slug>")
#
# Cycle: pull last-24h rewards -> enso-fit (ridge) -> holdout gate -> publish-or-hold
#        -> local log record -> POST /v1/router/publish-artifact-meta.
# Fail-closed: a thin/absent ledger or a failed gate keeps the incumbent serving.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SCOPE="global"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope) SCOPE="${2:?--scope needs a value}"; shift 2 ;;
    --scope=*) SCOPE="${1#--scope=}"; shift ;;
    *) echo "router-retrain: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

AI_BASE="${AI_BASE:-https://api.hanzo.ai}"
FIT_BIN="${FIT_BIN:-$ROOT/target/release/enso-fit}"
HEADS_DIR="${HEADS_DIR:-$ROOT/.router-retrain/heads}"
RETRAIN_LOG="${RETRAIN_LOG:-$ROOT/.router-retrain/log.jsonl}"
WINDOW_HOURS="${WINDOW_HOURS:-24}"
HOLDOUT="${HOLDOUT:-0.2}"
MIN_ROWS="${MIN_ROWS:-200}"
ENGINE_DEPLOY="${ENGINE_DEPLOY:-engine}"
ENGINE_NS="${ENGINE_NS:-hanzo}"
KCTX="${KCTX:-do-sfo3-hanzo-k8s}"
# Opt-in gate for the two prod-mutating side effects (off by default: the engine does
# not yet mount heads and publish-artifact-meta is not yet on prod -- see LLM notes).
DO_RELOAD="${DO_RELOAD:-0}"
DO_PUBLISH_META="${DO_PUBLISH_META:-1}"
# Comma-separated opted-in orgs for the GLOBAL fit (object.ListTrainingContributorOrgs).
CONTRIB_ORGS="${CONTRIB_ORGS:-}"
# ROUTER_ADMIN_TOKEN: super-admin bearer (IAM admin-org). Blank -> HTTP export/publish skipped.
ROUTER_ADMIN_TOKEN="${ROUTER_ADMIN_TOKEN:-}"

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$HEADS_DIR" "$(dirname "$RETRAIN_LOG")"

SINCE="$(date -u -d "$WINDOW_HOURS hours ago" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v-"${WINDOW_HOURS}"H +%Y-%m-%dT%H:%M:%SZ)"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Resolve scope -> (owner, heads filename, per-org query).
case "$SCOPE" in
  global)      OWNER="*"; HEADS_NAME="heads-base.safetensors"; ORG_Q="" ;;
  org=*)       SLUG="${SCOPE#org=}"; OWNER="$SLUG"; HEADS_NAME="heads-${SLUG}.safetensors"; ORG_Q="&org=${SLUG}" ;;
  *) echo "router-retrain: bad --scope '$SCOPE' (want 'global' or 'org=<slug>')" >&2; exit 2 ;;
esac
CURRENT="$HEADS_DIR/$HEADS_NAME"
REWARDS="$WORK/rewards.jsonl"; : > "$REWARDS"

# --- pull the rewarded routing tuples for this scope --------------------------
pull_http() { # $1=url ; appends ndjson to $REWARDS ; returns curl-ok
  [[ -n "$ROUTER_ADMIN_TOKEN" ]] || return 1
  local code
  code=$(curl -sS -m 60 -o "$WORK/pull.out" -w '%{http_code}' \
      -H "Authorization: Bearer $ROUTER_ADMIN_TOKEN" "$1" 2>>"$WORK/pull.err" || echo 000)
  if [[ "$code" == "200" ]]; then cat "$WORK/pull.out" >> "$REWARDS"; return 0; fi
  echo "router-retrain: export $1 -> HTTP $code" >&2; return 1
}

# A pre-fetched rewards file wins over HTTP: the seam for the kubectl DB-read
# fallback (export not on prod) and for local self-checks. Same ndjson line shape.
if [[ -n "${ROUTER_REWARDS_FILE:-}" && -f "${ROUTER_REWARDS_FILE:-}" ]]; then
  cp "$ROUTER_REWARDS_FILE" "$REWARDS"
elif [[ "$SCOPE" == "global" && -n "$CONTRIB_ORGS" ]]; then
  # Opt-in: the reward export line carries no org, so fetch per opted-in org and concat.
  IFS=',' read -r -a ORGS <<< "$CONTRIB_ORGS"
  for o in "${ORGS[@]}"; do
    o="$(echo "$o" | xargs)"; [[ -n "$o" ]] || continue
    pull_http "$AI_BASE/v1/export-routing-rewards?since=$SINCE&org=$o" || true
  done
else
  [[ "$SCOPE" == "global" ]] && echo "router-retrain: CONTRIB_ORGS empty -> unfiltered global pull (set it once ListTrainingContributorOrgs is exposed)" >&2
  pull_http "$AI_BASE/v1/export-routing-rewards?since=$SINCE$ORG_Q" || true
fi

ROWS=$(awk 'NF{n++} END{print n+0}' "$REWARDS")
echo "router-retrain[$SCOPE]: pulled $ROWS rewarded rows since $SINCE" >&2

# --- fit + holdout gate (native) ---------------------------------------------
CAND="$WORK/candidate.safetensors"
INC_ARG=(); [[ -f "$CURRENT" ]] && INC_ARG=(--incumbent "$CURRENT")
[[ -x "$FIT_BIN" ]] || { echo "router-retrain: FIT_BIN not built at $FIT_BIN (cargo build --release -p hanzo-router-retrain)" >&2; exit 3; }

REPORT="$("$FIT_BIN" --from-rewards --events "$REWARDS" --out "$CAND" \
           --holdout "$HOLDOUT" --min-rows "$MIN_ROWS" "${INC_ARG[@]}")"
echo "router-retrain[$SCOPE]: $REPORT" >&2

get() { echo "$REPORT" | jq -r "$1"; }
GATE_PASSED=$(get '.gatePassed'); GATE_VALUE=$(get '.gateValue'); GATE_BASE=$(get '.gateBase')
GATE_KIND=$(get '.gateKind'); GATE_METRIC=$(get '.gateMetric'); NOTE=$(get '.note')
EVENTS=$(get '.events'); HOLDOUT_ROWS=$(get '.holdoutRows')
ARTIFACT_SHA=$(sha256sum "$CAND" | cut -d' ' -f1)
VERSION="${NOW}-${ARTIFACT_SHA:0:12}"

# --- publish-or-hold ----------------------------------------------------------
PUBLISHED=false
if [[ "$GATE_PASSED" == "true" ]]; then
  cp "$CAND" "$HEADS_DIR/${HEADS_NAME%.safetensors}-${VERSION}.safetensors"
  cp "$CAND" "$CURRENT"                       # the "current" pointer the engine loads
  echo "router-retrain[$SCOPE]: promoted -> $CURRENT (v=$VERSION)" >&2
  if [[ "$DO_RELOAD" == "1" ]]; then
    kubectl --context "$KCTX" rollout restart "deployment/$ENGINE_DEPLOY" -n "$ENGINE_NS"
    PUBLISHED=true
  else
    echo "router-retrain[$SCOPE]: DO_RELOAD=0 -> wrote artifact, skipped engine reload (engine heads-mount not wired yet)" >&2
  fi
else
  echo "router-retrain[$SCOPE]: gate held -> kept incumbent ($NOTE)" >&2
fi

# --- durable local log record -------------------------------------------------
LOG_JSON=$(jq -nc \
  --arg ts "$NOW" --arg scope "$SCOPE" --arg owner "$OWNER" --arg ver "$VERSION" \
  --argjson rows "${EVENTS:-0}" --argjson hrows "${HOLDOUT_ROWS:-0}" \
  --argjson passed "${GATE_PASSED:-false}" --arg kind "$GATE_KIND" --arg metric "$GATE_METRIC" \
  --argjson val "${GATE_VALUE:-0}" --argjson base "${GATE_BASE:-0}" \
  --arg sha "$ARTIFACT_SHA" --argjson published "$PUBLISHED" --arg note "$NOTE" \
  '{ts:$ts,scope:$scope,owner:$owner,version:$ver,rows:$rows,holdoutRows:$hrows,gatePassed:$passed,gateKind:$kind,gateMetric:$metric,gateValue:$val,gateBase:$base,artifact:$sha,published:$published,note:$note}')
echo "$LOG_JSON" >> "$RETRAIN_LOG"
echo "$LOG_JSON"

# --- publish outcome to ai (the panel read path) ------------------------------
if [[ "$DO_PUBLISH_META" == "1" && -n "$ROUTER_ADMIN_TOKEN" ]]; then
  BODY=$(jq -nc \
    --arg owner "$OWNER" --arg version "$VERSION" --arg trainedTime "$NOW" \
    --argjson events "${EVENTS:-0}" --argjson gatePassed "${GATE_PASSED:-false}" \
    --argjson published "$PUBLISHED" --arg gateKind "$GATE_KIND" --arg gateMetric "$GATE_METRIC" \
    --argjson gateValue "${GATE_VALUE:-0}" --argjson gateBase "${GATE_BASE:-0}" --arg note "$NOTE" \
    '{owner:$owner,version:$version,trainedTime:$trainedTime,events:$events,gatePassed:$gatePassed,published:$published,gateKind:$gateKind,gateMetric:$gateMetric,gateValue:$gateValue,gateBase:$gateBase,note:$note}')
  code=$(curl -sS -m 30 -o "$WORK/meta.out" -w '%{http_code}' -X POST \
      -H "Authorization: Bearer $ROUTER_ADMIN_TOKEN" -H 'Content-Type: application/json' \
      -d "$BODY" "$AI_BASE/v1/router/publish-artifact-meta" 2>>"$WORK/pull.err" || echo 000)
  echo "router-retrain[$SCOPE]: publish-artifact-meta -> HTTP $code" >&2
else
  echo "router-retrain[$SCOPE]: publish-artifact-meta skipped (no ROUTER_ADMIN_TOKEN or DO_PUBLISH_META=0)" >&2
fi
