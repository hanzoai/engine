#!/usr/bin/env python3
"""bench_ledger — every committed run, as one document.

A run directory under bench-runs/ is what scripts/dossier.sh wrote and scripts/bench_stats.py
scored: raw per-repetition samples, a manifest pinning both engines, the model and the box, and
board.json. This reads all of them and emits the rows a reader wants side by side, each still
traceable to its run directory.

    bench_ledger.py [--runs bench-runs] [--source TEXT] [--branch NAME] [--commit SHA] > ledger.json

Nothing here measures or re-scores. A figure is the harness's own float, rounded for transport:
tokens per second to four places, ratios to six. `noisy` repeats the harness's flag, a
coefficient of variation above 5% in either engine, which is how it says the box was not quiet.

stdlib only: it runs on a bare box next to the harness.
"""
import argparse
import json
import os
import re
import subprocess
import sys

NOISY_CV = 5.0


def model_name(path):
    """`/x/Qwen_Qwen3-1.7B-Q4_K_M.gguf` -> `Qwen3-1.7B-Q4_K_M`: the file's stem, less a vendor
    prefix that only repeats the family name."""
    stem = os.path.splitext(os.path.basename(path or ""))[0]
    return re.sub(r"^([A-Za-z0-9]+)_(?=\1)", "", stem)


def git_head(path):
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load(path):
    with open(path) as f:
        return json.load(f)


def run_entry(run, manifest):
    params = manifest.get("params") or {}
    return {
        "run": run,
        "host": manifest.get("host"),
        "backend": manifest.get("backend"),
        "iso": manifest.get("iso"),
        "model": model_name(manifest.get("model_path")),
        "model_sha256": manifest.get("model_sha256"),
        "engine": manifest.get("engine_version"),
        "engine_sha": manifest.get("engine_git_sha"),
        "llama_sha": manifest.get("llama_sha"),
        "reps": params.get("reps"),
    }


def row_entry(run, model, cell):
    hanzo, llama, ratio = cell["hanzo"], cell["llama"], cell["ratio"]
    return {
        "run": run,
        "backend": cell["backend"],
        "model": model,
        "phase": cell["phase"],
        "n": cell["n"],
        "hanzo": round(hanzo["mean"], 4),
        "hanzo_ci": round(hanzo["ci"], 4),
        "hanzo_cv": round(hanzo["cv"], 4),
        "llama": round(llama["mean"], 4),
        "llama_ci": round(llama["ci"], 4),
        "llama_cv": round(llama["cv"], 4),
        "ratio": round(ratio["ratio"], 6),
        "ci": round(ratio["ci"], 6),
        "lo": round(ratio["lo"], 6),
        "hi": round(ratio["hi"], 6),
        "verdict": ratio["verdict"],
        "noisy": hanzo["cv"] > NOISY_CV or llama["cv"] > NOISY_CV,
        **best(hanzo, llama, ratio),
    }


def best(hanzo, llama, ratio):
    """Best-of-N, where the run recorded it. Runs scored before it existed carry none."""
    if hanzo.get("best") is None or llama.get("best") is None:
        return {}
    return {
        "hanzo_best": round(hanzo["best"], 4),
        "llama_best": round(llama["best"], 4),
        "best_ratio": round(ratio["best"], 6) if ratio.get("best") else None,
    }


def scored(cell):
    """A cell both engines completed. One that failed carries no mean, and has no ratio."""
    return all(isinstance((cell.get(k) or {}).get("mean"), (int, float)) for k in ("hanzo", "llama"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", default="bench-runs", help="directory of run directories")
    ap.add_argument("--source", default="github.com/hanzoai/engine · branch bench/dossier · bench-runs/*/board.json")
    ap.add_argument("--branch", default="bench/dossier")
    ap.add_argument("--commit", default=None, help="the ledger commit; default: HEAD of --runs' repository")
    args = ap.parse_args()

    runs, rows, skipped = [], [], []
    for run in sorted(os.listdir(args.runs)):
        base = os.path.join(args.runs, run)
        board, manifest = os.path.join(base, "board.json"), os.path.join(base, "manifest.json")
        if not (os.path.isfile(board) and os.path.isfile(manifest)):
            continue
        entry = run_entry(run, load(manifest))
        cells = [c for c in load(board) if scored(c)]
        if not cells:
            skipped.append(run)
            continue
        runs.append(entry)
        rows.extend(row_entry(run, entry["model"], c) for c in cells)

    for run in skipped:
        print(f"bench_ledger: {run} has no cell both engines completed; left out", file=sys.stderr)
    json.dump(
        {
            "source": args.source,
            "branch": args.branch,
            "commit": args.commit or git_head(args.runs),
            "runs": runs,
            "rows": rows,
        },
        sys.stdout,
        indent=2,
        ensure_ascii=False,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
