#!/usr/bin/env python3
"""Bytes one decode step moves, two ways, never substituted for each other.

  weights  COMPUTED weight + KV + recurrent-state bytes per token, from the safetensors headers
           (only the header bytes are read) and config.json. The method behind SGLang's 19.44 GB
           and hanzo's 26.65 GB in LLM.md section 3b.
  nsys     MEASURED SM-to-L2 bytes per token ('VidL2 Total from L1TEX [Bytes]', GB10's gb20y-top
           metric set) and GPU seconds per decode step (kernel time inside NVTX `decode` ranges),
           from an `nsys export --type sqlite` file. GB10 exposes no DRAM counter, so this is L2
           traffic, a cross-check on the computed figure and a finder of redundant reads.

Standard library only.
"""
import argparse
import glob
import json
import os
import re
import sqlite3
import struct
import sys

DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2, "I16": 2,
    "U16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BOOL": 1,
}
NAMED = {"bf16": 2, "f16": 2, "fp16": 2, "f32": 4, "fp32": 4, "fp8": 1, "f8": 1, "fp8_e4m3": 1,
         "fp4": 0.5, "nvfp4": 0.5625}
SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")
SKIPPED = re.compile(r"(^|\.)(visual|mtp)\.|^mtp\.")


def headers(checkpoint):
    """{tensor name: {dtype, shape, bytes}} over every shard's header."""
    tensors = {}
    for path in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            tensors[name] = {"dtype": info["dtype"], "shape": info["shape"], "bytes": end - start}
    if not tensors:
        raise SystemExit(f"no safetensors under {checkpoint}")
    return tensors


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def logical_numel(name, info, tensors):
    """Elements a weight represents: NVFP4 packs two per U8 byte (it has an F8 block scale)."""
    base = name[: -len(".weight")] if name.endswith(".weight") else None
    if info["dtype"] == "U8" and base and base + ".weight_scale" in tensors:
        return 2 * numel(info["shape"])
    return numel(info["shape"])


def text_config(checkpoint):
    with open(os.path.join(checkpoint, "config.json")) as f:
        config = json.load(f)
    return config.get("text_config", config)


def weights(args):
    tensors = headers(args.checkpoint)
    cfg = text_config(args.checkpoint)
    resident = []
    for spec in args.resident:
        pattern, _, dtype = spec.rpartition("=")
        if not pattern or dtype.lower() not in NAMED:
            raise SystemExit(f"--resident wants REGEX=DTYPE with DTYPE in {sorted(NAMED)}: {spec}")
        resident.append((re.compile(pattern), NAMED[dtype.lower()]))

    weight_bytes = 0.0
    embed_row = 0.0
    held = {}
    for name, info in tensors.items():
        if SKIPPED.search(name):
            continue
        if "embed_tokens" in name:
            embed_row += info["bytes"] / info["shape"][0]
            continue
        rule = next((r for r in resident if r[0].search(name)), None)
        if rule:
            if name.endswith(SCALE_SUFFIXES):
                continue  # the module is held dequantized; its scales are not read
            size = logical_numel(name, info, tensors) * rule[1]
            held[rule[0].pattern] = held.get(rule[0].pattern, 0) + size
            weight_bytes += size
        else:
            weight_bytes += info["bytes"]

    layer_types = cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"]
    full = layer_types.count("full_attention")
    linear = layer_types.count("linear_attention")
    kv_dtype = {"bf16": 2, "fp8": 1}[args.kv_dtype]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    kv = full * cfg["num_key_value_heads"] * head_dim * 2 * kv_dtype * args.context
    state = 0
    if linear:
        v_heads, k_heads = cfg["linear_num_value_heads"], cfg["linear_num_key_heads"]
        k_dim, v_dim = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
        ssm = linear * v_heads * k_dim * v_dim * 4 * 2  # fp32 state, read and written
        conv_dim = 2 * k_heads * k_dim + v_heads * v_dim
        conv = linear * conv_dim * (cfg["linear_conv_kernel_dim"] - 1) * 2 * 2  # bf16, read + write
        state = ssm + conv
    total = weight_bytes + embed_row + kv + state
    out = {
        "kind": "computed",
        "checkpoint": os.path.abspath(args.checkpoint),
        "context": args.context,
        "kv_dtype": args.kv_dtype,
        "weights_bytes": int(weight_bytes),
        "embed_row_bytes": int(embed_row),
        "kv_bytes": int(kv),
        "state_bytes": int(state),
        "resident": {k: int(v) for k, v in held.items()},
        "bytes_per_token": int(total),
        "gb_per_token": round(total / 1e9, 3),
    }
    print(json.dumps(out))
    return out


# ---------------------------------------------------------------------------------------------
# nsys export. Every table and column name the tool depends on is in this one function.

L2_METRIC = "VidL2 Total from L1TEX [Bytes]"


def read_capture(path, range_name="decode"):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    capture = {"l2_bytes": None, "ranges": 0, "kernel_ns": None}
    if {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
        (capture["l2_bytes"],) = db.execute(
            "select coalesce(sum(g.value), 0) from GPU_METRICS g join TARGET_INFO_GPU_METRICS t"
            " on g.typeId = t.typeId and g.metricId = t.metricId where t.metricName = ?",
            (L2_METRIC,)).fetchone()
    if "NVTX_EVENTS" in tables:
        text = ("coalesce(n.text, (select s.value from StringIds s where s.id = n.textId))"
                if "StringIds" in tables else "n.text")
        ranges = db.execute(
            f"select n.start, n.end, n.globalTid from NVTX_EVENTS n"
            f" where n.end is not null and {text} = ?", (range_name,)).fetchall()
        capture["ranges"] = len(ranges)
        work = [t for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
                if t in tables]
        if ranges and work and "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
            db.execute("create temp table ranges (start integer, end integer, tid integer)")
            db.executemany("insert into ranges values (?, ?, ?)", ranges)
            total = 0
            for table in work:
                (ns,) = db.execute(
                    f"select coalesce(sum(k.end - k.start), 0) from {table} k"
                    " join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = k.correlationId"
                    " join ranges g on r.globalTid = g.tid and r.start >= g.start and r.start <= g.end"
                ).fetchone()
                total += ns
            capture["kernel_ns"] = total
    return capture


def nsys(args):
    capture = read_capture(args.sqlite, args.range)
    out = {"kind": "measured", "sqlite": os.path.abspath(args.sqlite), "tokens": args.tokens}
    if capture["l2_bytes"] is not None:
        out["l2_bytes"] = capture["l2_bytes"]
        out["l2_bytes_per_token"] = capture["l2_bytes"] / args.tokens
        out["l2_gb_per_token"] = round(capture["l2_bytes"] / args.tokens / 1e9, 3)
    out["decode_ranges"] = capture["ranges"]
    if capture["kernel_ns"] is not None and capture["ranges"]:
        out["gpu_s_per_step"] = capture["kernel_ns"] / capture["ranges"] / 1e9
        if args.computed_bytes:
            out["achieved_gb_s"] = args.computed_bytes / out["gpu_s_per_step"] / 1e9
    print(json.dumps(out))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weights", help="computed bytes per token")
    w.add_argument("--checkpoint", required=True)
    w.add_argument("--context", type=int, required=True, help="tokens in the KV cache")
    w.add_argument("--kv-dtype", choices=["bf16", "fp8"], required=True)
    w.add_argument("--resident", action="append", default=[], metavar="REGEX=DTYPE",
                   help="tensors an engine holds at another dtype (their scales are dropped)")
    n = sub.add_parser("nsys", help="measured bytes per token from an nsys sqlite export")
    n.add_argument("--sqlite", required=True)
    n.add_argument("--tokens", type=int, required=True, help="tokens decoded inside the capture")
    n.add_argument("--range", default="decode", help="NVTX range name of a decode step")
    n.add_argument("--computed-bytes", type=float,
                   help="computed bytes per step, to report achieved GB/s over GPU time")
    args = ap.parse_args(argv)
    {"weights": weights, "nsys": nsys}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
