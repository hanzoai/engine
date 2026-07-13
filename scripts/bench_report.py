#!/usr/bin/env python3
"""bench_report — assemble one results.json and render the tables (one writer, DRY).

Input: a directory of per-row artifacts written by the orchestrator
  manifest.json          reproducibility header (git/host/gpu/model/env/serve cmd)
  roofline.json          measured RAM BW + NVMe BW (from the roofline probe)
  row-<i>.json           { config, records:[client records], peak_rss_gb,
                           resident_gb, expert_cache }

Output: results.json (machine) + a console/markdown table (human), both derived
from the same in-memory rows. This is the single home for the roofline arithmetic:
bytes/token, predicted ceiling, %-of-ceiling — computed here where the row data is,
not in the bandwidth probe (which only measures) or the load generator (which only
drives). Ceiling is emitted only when the expert-cache readout is present; on an
engine without that surface the column degrades to null (no invented numbers).
"""

import argparse
import json
import os
import sys

GIB = 1024.0**3


def load(path: str) -> dict | list | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pick(records: list[dict], phase: str) -> list[dict]:
    return [r for r in records if r.get("phase") == phase]


def prefill_tok_s(records: list[dict]) -> float | None:
    vals = [r["usage_prompt_tok_s"] for r in pick(records, "prefill") if r.get("usage_prompt_tok_s")]
    return max(vals) if vals else None


def decode_usage_tok_s(records: list[dict]) -> float | None:
    vals = [r["usage_compl_tok_s"] for r in pick(records, "decode") if r.get("usage_compl_tok_s")]
    return max(vals) if vals else None


def steady(records: list[dict]) -> dict | None:
    best = None
    for r in pick(records, "steady"):
        itl = r.get("itl")
        if itl and itl.get("steady_tok_s"):
            if best is None or itl["steady_tok_s"] > best["steady_tok_s"]:
                best = itl
    return best


def batch_curve(records: list[dict]) -> list[dict]:
    return [
        {"n": r["config"]["n"], "aggregate_tok_s": r.get("aggregate_tok_s"),
         "per_stream_tok_s": r.get("per_stream_tok_s")}
        for r in pick(records, "batch")
    ]


def roofline_for_row(row: dict, roof: dict | None, decode_tok_s: float | None) -> dict | None:
    """Predicted decode ceiling from measured BW + the row's expert-cache readout.

    hit_bytes serviced from RAM, miss_bytes from NVMe:
      hit_bytes  = active_weight_bytes*hit_rate + resident_dense_bytes
      miss_bytes = active_weight_bytes*(1-hit_rate)
      ceiling    = 1 / (hit_bytes/RAM_BW + miss_bytes/NVMe_BW + kv_bytes/RAM_BW)
    STREAM_EXPERTS-off collapses to RAM_BW/bytes_per_token (hit_rate treated as 1).
    Requires the expert-cache readout; returns None otherwise (no guessing).
    """
    ec = row.get("expert_cache")
    if not roof or not ec:
        return None
    ram_bw = (roof.get("ram_bw_gbs") or {}).get("all_thread")
    nvme_bw = (roof.get("nvme_bw_gbs") or {}).get("value")
    if not ram_bw:
        return None
    active = ec.get("active_weight_bytes")
    if not active:
        return None
    hit_rate = ec.get("hit_rate", 1.0)
    resident_dense = ec.get("resident_dense_bytes", 0.0)
    kv_bytes = ec.get("kv_read_bytes", 0.0)
    hit_bytes = active * hit_rate + resident_dense
    miss_bytes = active * (1.0 - hit_rate)
    ram_bw_bps = ram_bw * 1e9
    secs = (hit_bytes + kv_bytes) / ram_bw_bps
    if miss_bytes > 0 and nvme_bw:
        secs += miss_bytes / (nvme_bw * 1e9)
    ceiling = (1.0 / secs) if secs > 0 else None
    pct = (100.0 * decode_tok_s / ceiling) if (ceiling and decode_tok_s) else None
    return {
        "bytes_per_token": hit_bytes + miss_bytes + kv_bytes,
        "predicted_ceiling_tok_s": ceiling,
        "pct_of_ceiling": pct,
    }


def assemble(rows_in: list[dict], roof: dict | None) -> list[dict]:
    rows = []
    for row in rows_in:
        recs = row.get("records", [])
        dec_usage = decode_usage_tok_s(recs)
        st = steady(recs)
        dec_steady = st["steady_tok_s"] if st else None
        rows.append({
            "config": row.get("config", {}),
            "prefill_tok_s": prefill_tok_s(recs),
            "decode_tok_s_usage": dec_usage,
            "decode_tok_s_steady": dec_steady,
            "itl_p50_ms": st["p50_ms"] if st else None,
            "itl_p90_ms": st["p90_ms"] if st else None,
            "peak_rss_gb": row.get("peak_rss_gb"),
            "resident_gb": row.get("resident_gb"),
            "expert_cache": row.get("expert_cache"),
            "batch": batch_curve(recs),
            "roofline": roofline_for_row(row, roof, dec_steady or dec_usage),
        })
    return rows


def fmt(v, prec=1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def cfg_label(c: dict) -> str:
    q = c.get("quant", "?")
    se = "on" if c.get("stream_experts") else "off"
    ram = c.get("ram_gb")
    ram = "inf" if ram is None else fmt(ram, 0)
    return f"{q}/se={se}/ram={ram}"


def render_table(rows: list[dict]) -> str:
    cols = ["config", "prefill t/s", "decode t/s", "steady t/s", "p90 ITL ms",
            "peak RSS", "resident", "hit%", "ceiling t/s", "%ceil"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in sorted(rows, key=lambda x: cfg_label(x["config"])):
        ec = r.get("expert_cache") or {}
        roof = r.get("roofline") or {}
        hit = ec.get("hit_rate")
        lines.append("| " + " | ".join([
            cfg_label(r["config"]),
            fmt(r["prefill_tok_s"]),
            fmt(r["decode_tok_s_usage"]),
            fmt(r["decode_tok_s_steady"]),
            fmt(r["itl_p90_ms"], 2),
            fmt(r["peak_rss_gb"]),
            fmt(r["resident_gb"]),
            fmt(hit * 100 if hit is not None else None),
            fmt(roof.get("predicted_ceiling_tok_s")),
            fmt(roof.get("pct_of_ceiling")),
        ]) + " |")
    return "\n".join(lines)


def render_roofline(roof: dict | None, rows: list[dict]) -> str:
    if not roof:
        return "roofline: (no probe)"
    ram = roof.get("ram_bw_gbs") or {}
    nvme = roof.get("nvme_bw_gbs") or {}
    out = [
        "roofline probe:",
        f"  RAM BW  : {fmt(ram.get('single_thread'))} GB/s (1t), "
        f"{fmt(ram.get('all_thread'))} GB/s ({ram.get('threads')}t)",
        f"  NVMe BW : {fmt(nvme.get('value'))} GB/s "
        f"(direct={nvme.get('direct')}, {fmt(nvme.get('bytes_read', 0) / GIB)} GiB read)",
    ]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Assemble bench results.json + table")
    p.add_argument("--in-dir", required=True, help="dir with manifest/roofline/row-*.json")
    p.add_argument("--out", default="", help="results.json path (default <in-dir>/results.json)")
    a = p.parse_args(argv)

    manifest = load(os.path.join(a.in_dir, "manifest.json")) or {}
    roof = load(os.path.join(a.in_dir, "roofline.json"))
    rows_in = []
    for name in sorted(os.listdir(a.in_dir)):
        if name.startswith("row-") and name.endswith(".json"):
            rows_in.append(load(os.path.join(a.in_dir, name)))

    rows = assemble([r for r in rows_in if r], roof)
    if roof:
        manifest["ram_bw_gbs"] = (roof.get("ram_bw_gbs") or {}).get("all_thread")
        manifest["nvme_bw_gbs"] = (roof.get("nvme_bw_gbs") or {}).get("value")
    results = {"manifest": manifest, "rows": rows}

    out = a.out or os.path.join(a.in_dir, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(render_roofline(roof, rows))
    print()
    print(render_table(rows))
    print()
    print(f"results.json -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
