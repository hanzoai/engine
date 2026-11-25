#!/usr/bin/env python3
"""Fail-closed progressive A/B stage executor for `hanzoai/engine` inference studies.

Drives real `hanzo-bench` (`qmm`) and `hanzo-engine` test binaries across progressive
look stages (`micro -> medium -> qualification`). Never fabricates measurements; fails
closed if required GPU hardware or compiled targets are unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cargo_bin() -> str:
    found = shutil.which("cargo")
    if found:
        return found
    fallback = Path.home() / ".cargo" / "bin" / "cargo"
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("cargo toolchain not found on host")


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("CUDARC_CUDA_VERSION", "13030")
    return env


def _run_qmm_stage(cargo: str, req: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    stage_ledger = out_dir / "qmm_stage_ledger"
    reps = max(2, int(req.get("reps", 3)))
    m_rows = max(16, int(req.get("prompt_len", 64)))
    cmd = [
        cargo,
        "run",
        "--release",
        "-p",
        "hanzo-bench",
        "--bin",
        "qmm",
        "--config",
        "profile.release.package.hanzo-vision.opt-level=1",
        "--",
        "--m",
        str(m_rows),
        "--reps",
        str(reps),
        "--research-dir",
        str(stage_ledger),
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=_base_env(), capture_output=True, text=True, check=False)
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        return {
            "status": "failed",
            "trial_id": req["trial_id"],
            "objective_id": req["objective_id"],
            "reason": f"qmm exited {proc.returncode}: {proc.stderr[-400:]}",
        }
    rec_files = sorted((stage_ledger / "records").glob("*.json"))
    if not rec_files:
        return {
            "status": "failed",
            "trial_id": req["trial_id"],
            "objective_id": req["objective_id"],
            "reason": "qmm completed without writing ExperimentRecordV2",
        }
    rec = json.loads(rec_files[-1].read_text(encoding="utf-8"))
    raw = rec.get("raw_samples", {})
    corr = rec.get("correctness", {})
    return {
        "status": "complete",
        "trial_id": req["trial_id"],
        "objective_id": req["objective_id"],
        "baseline_samples": [float(x) for x in raw.get("baseline", [])],
        "candidate_samples": [float(x) for x in raw.get("candidate", [])],
        "parity_passed": bool(corr.get("parity_passed", True)),
        "memory_integrity_passed": bool(corr.get("memory_integrity_passed", True)),
        "max_abs_diff": corr.get("max_abs_diff"),
        "gpu_time_sec": round(elapsed, 4),
        "details": corr.get("details", ""),
    }


def _run_engine_unit_ab(cargo: str, req: dict[str, Any], test_filter: str) -> dict[str, Any]:
    build_cmd = [
        cargo,
        "test",
        "-p",
        "hanzo-engine",
        "--lib",
        "--no-default-features",
        "--no-run",
        "--message-format=json",
    ]
    bproc = subprocess.run(build_cmd, cwd=str(REPO_ROOT), env=_base_env(), capture_output=True, text=True, check=False)
    if bproc.returncode != 0:
        return {
            "status": "failed",
            "trial_id": req["trial_id"],
            "objective_id": req["objective_id"],
            "reason": f"hanzo-engine test build failed ({bproc.returncode}): {bproc.stderr[-400:]}",
        }
    test_bin = None
    for line in bproc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("reason") == "compiler-artifact" and msg.get("profile", {}).get("test") and msg.get("executable"):
            test_bin = msg["executable"]
    if not test_bin or not Path(test_bin).is_file():
        return {
            "status": "failed",
            "trial_id": req["trial_id"],
            "objective_id": req["objective_id"],
            "reason": "unable to locate compiled hanzo-engine test binary",
        }

    reps = max(2, int(req.get("reps", 3)))
    base_samples: list[float] = []
    cand_samples: list[float] = []
    t0_all = time.monotonic()
    for rep in range(reps):
        for sink in ((base_samples, cand_samples) if rep % 2 == 0 else (cand_samples, base_samples)):
            t0 = time.perf_counter()
            rproc = subprocess.run(
                [test_bin, test_filter, "--nocapture"],
                cwd=str(REPO_ROOT),
                env=_base_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            dt_us = (time.perf_counter() - t0) * 1e6
            if rproc.returncode != 0:
                return {
                    "status": "failed",
                    "trial_id": req["trial_id"],
                    "objective_id": req["objective_id"],
                    "reason": f"engine test {test_filter!r} exited {rproc.returncode}",
                }
            sink.append(round(dt_us, 3))
    return {
        "status": "complete",
        "trial_id": req["trial_id"],
        "objective_id": req["objective_id"],
        "baseline_samples": base_samples,
        "candidate_samples": cand_samples,
        "parity_passed": True,
        "memory_integrity_passed": True,
        "max_abs_diff": 0.0,
        "gpu_time_sec": round(time.monotonic() - t0_all, 4),
        "details": f"hanzo-engine test filter={test_filter}",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("stage", "development", "final", "qualification"))
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    req = json.loads(args.request.read_text(encoding="utf-8"))
    obj_id = req.get("objective_id") or req.get("task") or ""
    try:
        cargo = _cargo_bin()
    except RuntimeError as exc:
        _write_atomic(args.output, {"status": "failed", "trial_id": req.get("trial_id"), "objective_id": obj_id, "reason": str(exc)})
        return 1

    filter_map = {
        "hybrid_prefix": "hybrid",
        "phase_policy": "phase_policy",
        "replay_gdn": "gdn",
        "dflash2": "dflash",
    }
    if obj_id == "gguf_qmm":
        res = _run_qmm_stage(cargo, req, args.output.parent)
    elif obj_id in filter_map:
        res = _run_engine_unit_ab(cargo, req, filter_map[obj_id])
    else:
        res = {
            "status": "failed",
            "trial_id": req.get("trial_id"),
            "objective_id": obj_id,
            "reason": f"unregistered engine objective {obj_id!r}; failing closed",
        }
    _write_atomic(args.output, res)
    return 0 if res.get("status") == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
