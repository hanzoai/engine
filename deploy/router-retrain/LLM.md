# Router-heads nightly retrain (Enso router)

Spark-local job that refits the Enso routing heads from the ai routing ledger each
night, gates the candidate on a holdout, and publishes only a non-regressing head.
Enso is the learned policy (the `enso` crate, `W = D*K` bilinear utility); this job
produces the `heads-{scope}.safetensors` it loads.

## Pieces (one job, orthogonal parts)

- `hanzo-router-retrain/` crate -- the tested core.
  - `src/lib.rs`: `transform_reward` (ai reward tuple -> enso `EvalSample`),
    `fit_policy` (proof.rs's exact (x,p,quality) join -> `fit_base`), `holdout_reward`
    (score a `Policy` on held-out rows), `decide_gate` (fail-closed publish decision),
    `split_indices` (seeded), `save_w`/`load_w` (enso-base safetensors: one f64 tensor
    `w`, shape `[D*K]=192`, metadata `format=enso-base`). 9 unit tests.
  - `src/main.rs`: `enso-fit` binary. Contract, a drop-in for enso's own `fit`:
    `enso-fit --events <in.jsonl> --out <heads.safetensors>` (+ `--from-rewards`,
    `--holdout`, `--incumbent`, `--min-rows`). Emits a gate JSON report to stdout.
- `scripts/router-retrain.sh` -- one scope: pull last-24h rewards -> `enso-fit` ->
  holdout gate -> promote-or-hold -> local log record -> POST publish-artifact-meta.
- `scripts/router-retrain-all.sh` -- the nightly composition: `--scope global` then a
  `--scope org=<slug>` per opted-in org.
- `deploy/router-retrain/` -- the scheduler (systemd user timer) + the canonical
  `hanzoai/tasks` cron ConfigMap (convergence reference).

## Scope model (one mechanism, N scopes)

- `--scope global` -> `heads-base.safetensors`, publish `owner:"*"` (shared base).
  Rows come from opted-in orgs only (`CONTRIB_ORGS` = `object.ListTrainingContributorOrgs`);
  the reward export carries no `org`, so global fetches per opted-in org and concats.
- `--scope org=<slug>` -> `heads-<slug>.safetensors`, publish `owner:"<slug>"`, using
  `GET /v1/export-routing-rewards?org=<slug>`. The gate holds any org under `MIN_ROWS`.

Filenames align with the engine adapter-cache fallback chain (org -> base -> rules).

## Gate

Holdout (RouterBench replication is absent in-repo). Split 80/20, fit the candidate on
80%, score candidate + incumbent on the held-out 20% with `holdout_reward`
(`mean clamp(1-|predicted-quality - reward|, 0, 1)`), publish iff
`candidate >= incumbent` AND `holdout_rows >= MIN_ROWS`. `gateKind:"holdout"`,
`gateMetric:"holdout_reward"`. No incumbent -> bootstrap publish. Thin ledger -> hold.

## Scheduler decision (systemd timer, with evidence)

The job MUST run on spark: it builds + runs the enso Rust `fit` natively on the GB10
box. The in-cluster `hanzoai/tasks` durable cron is LIVE but its `RunJobActivity`
creates k8s Jobs IN-cluster; it cannot place work on spark without an unbuilt
spark<->tasks bridge. By the "exists AND fits a nightly spark job" test the systemd
user timer wins: present on the box, `Persistent=true` catches missed runs on wake,
`OnCalendar=... America/Los_Angeles` tracks DST. History = the local `log.jsonl` plus
the ai `router_artifact_meta` panel. `tasks-cron.configmap.yaml` is the documented
in-cluster convergence for when a spark receiver/worker exists.

Install: `deploy/router-retrain/install.sh` (builds fit, enables linger, installs the
`--user` timer). Env/secrets: `~/.config/router-retrain/env` (`ROUTER_ADMIN_TOKEN`
from KMS `hanzo/prod/router-admin-token`, `CONTRIB_ORGS`, `DO_RELOAD`).

## Blockers (wiring this job depends on, none owned here)

1. `fit` binary: enso ships only `fit_base`/`ingest`/`parse_jsonl` as lib fns; the
   `fit` CLI + `persist.rs` are uncommitted WIP in the learning-loop worktree. This job
   ships a reference `enso-fit` on the SAME `--events/--out` contract, reusing committed
   primitives; point `FIT_BIN` at enso's `fit` when it merges. Reconcile the fit INPUT
   schema: enso re-featurizes from the request bucket, so the ledger's stored backbone
   `features` vector is carried but not consumed yet.
2. Export not on prod: `GET /v1/export-routing-rewards` 404s on api.hanzo.ai (deployed
   cloud image predates the routes; they exist on ai main). `publish-artifact-meta` is
   on ai branch `feat/router-stats-observability` (PR #99), not yet deployed.
3. Super-admin auth: exports need IAM `admin`-org identity; no in-cluster/service
   principal mints one. `ROUTER_ADMIN_TOKEN` must be provisioned in KMS. Documented
   fallback: read the cluster store directly (kubectl `do-sfo3-hanzo-k8s`).
4. Engine heads-mount: the engine App CR mounts only `hf-cache emptyDir`; nothing
   ingests `heads-*.safetensors` and CI RBAC does not whitelist `engine` for restart.
   `DO_RELOAD=0` until an initContainer/PVC/S3 pull + reload path lands. Publishing
   writes the versioned artifact + `current` pointer; it does not restart prod.
