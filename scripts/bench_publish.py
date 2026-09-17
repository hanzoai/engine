#!/usr/bin/env python3
"""bench_publish — file a run as evidence in Hanzo Research.

A run directory (scripts/dossier.sh, scored by scripts/bench_stats.py) becomes one batch for
POST /v1/research/experiments: every cell is two kernel-perf experiments, ours and llama.cpp's,
each carrying the commit that produced it. Research keeps versions under a stable id and is
idempotent by content, so a later engine revision measuring the same cell appends beside the
earlier one, and filing the same run twice adds nothing.

    bench_publish.py RUN_DIR                       print the batch
    bench_publish.py RUN_DIR --post URL            send it; the key is read from $HANZO_API_KEY

The stable id is kernel-perf:<engine>/<backend>/<host>/<model>:<phase>-<n>. The value is the
harness's mean tokens per second; the samples, the interval, and for our row the ratio against
llama.cpp travel in meta. Nothing here measures or re-scores.

stdlib only.
"""
import argparse
import json
import os
import sys
import urllib.request

KIND = "kernel-perf"
NOISY_CV = 5.0
BATCH = 200


def load(path):
    with open(path) as f:
        return json.load(f)


def model_stem(path):
    return os.path.splitext(os.path.basename(path or ""))[0]


def experiments(run_dir):
    run = os.path.basename(os.path.normpath(run_dir))
    manifest = load(os.path.join(run_dir, "manifest.json"))
    params = manifest.get("params") or {}
    model = model_stem(manifest.get("model_path"))
    where = f"{manifest.get('backend')}/{manifest.get('host')}/{model}"
    common = {
        "run": run,
        "model_sha256": manifest.get("model_sha256"),
        "model_bytes": manifest.get("model_bytes"),
        "gpu": manifest.get("gpu"),
        "sampler": manifest.get("sampler"),
        "reps": params.get("reps"),
        "concurrency": params.get("concurrency"),
    }
    engines = {
        "hanzo": {
            "subject": f"hanzo-engine/{where}",
            "git_sha": manifest.get("engine_git_sha") or "",
            "git_dirty": bool(manifest.get("engine_git_dirty")),
            "lib_versions": {
                k: v
                for k, v in {
                    "hanzo-engine": manifest.get("engine_version"),
                    "hanzo-ml": manifest.get("hanzo_ml"),
                    "hanzo-rocm-kernels": manifest.get("hanzo_rocm_kernels"),
                    "hanzo-metal-kernels": manifest.get("hanzo_metal_kernels"),
                }.items()
                if v
            },
            "extra": {"args": params.get("hanzo_args") or ""},
        },
        "llama": {
            "subject": f"llama.cpp/{where}",
            "git_sha": manifest.get("llama_sha") or "",
            "git_dirty": bool(manifest.get("llama_git_dirty")),
            "lib_versions": {},
            "extra": {},
        },
    }
    out = []
    for cell in load(os.path.join(run_dir, "board.json")):
        task = f"{cell['phase']}-{cell['n']}"
        noisy = any((cell.get(k) or {}).get("cv", 0) > NOISY_CV for k in ("hanzo", "llama"))
        for key, engine in engines.items():
            scored = cell.get(key) or {}
            if not isinstance(scored.get("mean"), (int, float)):
                continue
            meta = {
                **common,
                **engine["extra"],
                "ci95": scored.get("ci"),
                "cv_pct": scored.get("cv"),
                "std": scored.get("std"),
                "samples": scored.get("samples"),
                "noisy": noisy,
            }
            ratio = cell.get("ratio")
            if key == "hanzo" and isinstance((ratio or {}).get("ratio"), (int, float)):
                meta["versus"] = {"subject": engines["llama"]["subject"], **ratio}
            out.append(
                {
                    "id": f"{KIND}:{engine['subject']}:{task}",
                    "kind": KIND,
                    "subject": engine["subject"],
                    "task": task,
                    "metric": "tok/s",
                    "value": scored["mean"],
                    "n": scored.get("n") or 0,
                    "meta": meta,
                    "git_sha": engine["git_sha"],
                    "git_dirty": engine["git_dirty"],
                    "lib_versions": engine["lib_versions"],
                    "ts": int(manifest.get("timestamp") or 0),
                }
            )
    return out


def post(url, key, batch):
    req = urllib.request.Request(
        url,
        data=json.dumps({"experiments": batch, "attempts": []}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--post", metavar="URL", help="e.g. https://api.hanzo.ai/v1/research/experiments")
    args = ap.parse_args()

    rows = experiments(args.run_dir)
    if not args.post:
        json.dump({"experiments": rows, "attempts": []}, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return
    key = os.environ.get("HANZO_API_KEY")
    if not key:
        sys.exit("bench_publish: --post needs $HANZO_API_KEY")
    for i in range(0, len(rows), BATCH):
        print(json.dumps(post(args.post, key, rows[i : i + BATCH])))


if __name__ == "__main__":
    main()
