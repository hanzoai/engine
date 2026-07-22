#!/usr/bin/env python3
"""bench_stats — turn raw per-repetition bench samples into a defensible board.

Input is a run directory produced by scripts/dossier.sh: for each (model, backend)
one `hanzo_<tag>.json` (from `hanzo-bench --json`, raw per-rep [wall_s, tokens]) and
one `llama_<tag>.json` (from `llama-bench -o json`). Every published statistic is a
pure function of those raw samples, computed here and nowhere else:

  * per-rep tok/s  = tokens / wall_s / concurrency          (hanzo, matches the binary)
                   = samples_ts[i]                           (llama, native)
  * mean, sample stddev (ddof=1), n
  * 95% CI         = mean +/- t(0.975, n-1) * s / sqrt(n)    (Student t, small n)
  * CV%            = 100 * s / mean                          (flagged when > 5%)
  * ratio          = mean_hanzo / mean_llama, with the relative CIs added in
                     quadrature (delta method, independent means)

Outputs, all in the run dir:
  board.md          human table (model x backend x phase: hanzo, llama, ratio, verdict)
  board.json        the same, machine-readable, every field traceable to a sample file
  results-data.tex  \\renewcommand macros for the LaTeX paper (numbers-as-data contract)

No third-party deps: this must run on a bare box next to the harness. stdlib only.
"""
import glob
import json
import math
import os
import sys

# Student t two-sided 0.975 critical values by degrees of freedom (n-1). For the
# small n a benchmark actually runs, the normal 1.96 understates the interval; the
# table is exact for df<=30 and falls back to 1.96 (asymptotic) above.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
        27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def tcrit(n):
    return T975.get(n - 1, 1.96) if n > 1 else float("nan")


def stats(samples):
    """(mean, ci_halfwidth, cv_percent, n) from raw per-rep tok/s samples."""
    n = len(samples)
    if n == 0:
        return None
    mean = sum(samples) / n
    if n == 1:
        return {"mean": mean, "ci": float("nan"), "cv": float("nan"), "n": 1,
                "samples": samples}
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)
    s = math.sqrt(var)
    ci = tcrit(n) * s / math.sqrt(n)
    cv = 100.0 * s / mean if mean else float("nan")
    return {"mean": mean, "ci": ci, "cv": cv, "n": n, "std": s, "samples": samples}


def hanzo_samples(rec):
    """per-rep tok/s for one hanzo result record: tokens/wall/concurrency."""
    c = rec["concurrency"]
    return [toks / secs / c for secs, toks in rec["per_rep"] if secs > 0 and toks > 0]


def load_hanzo(path):
    d = json.load(open(path))
    out = {}
    for r in d["results"]:
        key = (r["phase"], r["n"], r["concurrency"])
        out[key] = {"stats": stats(hanzo_samples(r)),
                    "backend": d["backend"], "model": d.get("model_id", "?"),
                    "engine_version": d.get("engine_version"), "src": os.path.basename(path)}
    return out


def load_llama(path):
    """llama-bench -o json -> {(phase,n,1): stats}. Prefer raw samples_ts; else
    reconstruct the interval from avg_ts+stddev_ts+n (same t-formula)."""
    rows = json.load(open(path))
    out = {}
    for row in rows:
        pp, tg = row.get("n_prompt", 0), row.get("n_gen", 0)
        phase, n = ("prefill", pp) if (pp > 0 and tg == 0) else ("decode", tg)
        samp = row.get("samples_ts")
        if samp:
            st = stats(samp)
        else:  # reconstruct from reported moments
            n_rep = row.get("reps") or len(row.get("samples_ns", []) or []) or 1
            mean, sd = row.get("avg_ts"), row.get("stddev_ts", 0.0) or 0.0
            ci = tcrit(n_rep) * sd / math.sqrt(n_rep) if n_rep > 1 else float("nan")
            st = {"mean": mean, "ci": ci, "cv": (100 * sd / mean if mean else float("nan")),
                  "n": n_rep, "std": sd, "samples": None}
        out[(phase, n, 1)] = {"stats": st, "src": os.path.basename(path)}
    return out


def ratio(h, l):
    """hanzo/llama mean ratio with a delta-method 95% CI and a verdict."""
    if not h or not l or not h["mean"] or not l["mean"]:
        return None
    r = h["mean"] / l["mean"]
    rh = (h["ci"] / h["mean"]) if h["mean"] and not math.isnan(h.get("ci", float("nan"))) else 0.0
    rl = (l["ci"] / l["mean"]) if l["mean"] and not math.isnan(l.get("ci", float("nan"))) else 0.0
    rel = math.sqrt(rh * rh + rl * rl)
    ci = r * rel
    lo, hi = r - ci, r + ci
    verdict = "WIN" if lo > 1.0 else "LOSS" if hi < 1.0 else "PARITY"
    return {"ratio": r, "ci": ci, "lo": lo, "hi": hi, "verdict": verdict}


# LaTeX control sequences are letters ONLY (no digits), so every measured cell gets
# a letter-only macro key: backend title + a spelled role. Canonical shapes map to
# words (Decode / Prefill / PrefillRagged / PrefillLong); anything else falls back to
# a digit->letter spelling so the key stays valid and deterministic.
BACKEND_TITLE = {"ROCm": "Rocm", "CUDA": "Cuda", "Metal": "Metal", "Vulkan": "Vulkan", "CPU": "Cpu"}
_DIGIT = "ZOTHFVSNEI"  # 0..9 -> letters


def spell(n):
    return "".join(_DIGIT[int(d)] for d in str(n))


def role(phase, n, conc):
    r = "Decode" if phase == "decode" else {512: "Prefill", 500: "PrefillRagged",
                                             2048: "PrefillLong", 4096: "PrefillMax"}.get(n, "Prefill" + spell(n))
    return r + (f"C{spell(conc)}" if conc != 1 else "")


def macro_base(backend, phase, n, conc):
    return "r" + BACKEND_TITLE.get(backend, "".join(c for c in backend if c.isalpha())) + role(phase, n, conc)


def fmt(st):
    if not st:
        return "--"
    if math.isnan(st.get("ci", float("nan"))):
        return f"{st['mean']:.1f}"
    flag = "!" if (not math.isnan(st["cv"]) and st["cv"] > 5.0) else ""
    return f"{st['mean']:.1f}±{st['ci']:.1f}{flag}"


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "."
    hanzo, llama = {}, {}
    for p in sorted(glob.glob(os.path.join(run, "hanzo_*.json"))):
        tag = os.path.basename(p)[len("hanzo_"):-len(".json")]
        hanzo[tag] = load_hanzo(p)
    for p in sorted(glob.glob(os.path.join(run, "llama_*.json"))):
        tag = os.path.basename(p)[len("llama_"):-len(".json")]
        try:
            llama[tag] = load_llama(p)
        except Exception as e:
            print(f"warn: unreadable {p}: {e}", file=sys.stderr)

    board, tex, texrows = [], [], []
    md = ["| model | backend | phase | n | conc | hanzo t/s | llama t/s | ratio | verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for tag in sorted(hanzo):
        for key in sorted(hanzo[tag]):
            phase, n, conc = key
            h = hanzo[tag][key]
            hs = h["stats"]
            lrec = llama.get(tag, {}).get(key)
            ls = lrec["stats"] if lrec else None
            rr = ratio(hs, ls) if ls else None
            model, backend = h["model"], h["backend"]
            md.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                model.split("/")[-1][:20], backend, phase, n, conc, fmt(hs), fmt(ls),
                (f"{rr['ratio']:.3f}±{rr['ci']:.3f}" if rr else "--"),
                (rr["verdict"] if rr else "--")))
            board.append({"tag": tag, "model": model, "backend": backend, "phase": phase,
                          "n": n, "concurrency": conc, "hanzo": hs, "llama": ls, "ratio": rr,
                          "engine_version": h.get("engine_version"),
                          "hanzo_src": h.get("src"), "llama_src": lrec.get("src") if lrec else None})
            base = macro_base(backend, phase, n, conc)  # letters only, e.g. rRocmDecode
            if hs:
                tex.append(f"\\renewcommand{{\\{base}H}}{{{hs['mean']:.1f}}}")
                if not math.isnan(hs.get("ci", float("nan"))):
                    tex.append(f"\\renewcommand{{\\{base}Hci}}{{{hs['ci']:.1f}}}")
            if ls:
                tex.append(f"\\renewcommand{{\\{base}L}}{{{ls['mean']:.1f}}}")
            if rr:
                tex.append(f"\\renewcommand{{\\{base}R}}{{{rr['ratio']:.2f}}}")
                tex.append(f"\\renewcommand{{\\{base}V}}{{{rr['verdict']}}}")
            # one booktabs row per measured cell, for \input into the paper's results table
            def cell(st):
                if not st:
                    return "--"
                if math.isnan(st.get("ci", float("nan"))):
                    return f"{st['mean']:.1f}"
                return f"{st['mean']:.1f}$\\pm${st['ci']:.1f}"
            texrows.append("{} & {} & {} & {} & {} & {} & {} \\\\".format(
                backend, phase, n, cell(hs), cell(ls),
                (f"{rr['ratio']:.2f}$\\pm${rr['ci']:.2f}" if rr else "--"),
                (f"\\textsc{{{rr['verdict'].lower()}}}" if rr else "--")))

    open(os.path.join(run, "board.md"), "w").write("\n".join(md) + "\n")
    json.dump(board, open(os.path.join(run, "board.json"), "w"), indent=2)
    header = ("% GENERATED by scripts/bench_stats.py -- do not hand-edit.\n"
              "% Every number is a pure function of the raw per-rep samples in this run dir.\n"
              "% '!' in board.md marks CV>5% (timing variance; box was not fully quiet).\n")
    open(os.path.join(run, "results-data.tex"), "w").write(header + "\n".join(sorted(set(tex))) + "\n")
    # rows.tex: bare booktabs rows, for merging several backends' runs into one paper
    # table. board.tex: a complete standalone tabular (\input at FLOAT level only --
    # never inside an open alignment, which misplaces \noalign).
    open(os.path.join(run, "rows.tex"), "w").write("\n".join(texrows) + "\n")
    open(os.path.join(run, "board.tex"), "w").write(
        "\\begin{tabular}{@{}llrrrrl@{}}\n\\toprule\n"
        "backend & phase & $n$ & hanzo t/s & llama t/s & ratio & verdict \\\\\n\\midrule\n"
        + "\n".join(texrows) + "\n\\bottomrule\n\\end{tabular}\n")
    print("\n".join(md))
    print(f"\n-> {run}/board.md  board.json  results-data.tex  rows.tex  board.tex")


if __name__ == "__main__":
    main()
