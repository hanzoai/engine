"""Time vLLM's block-FP8 CUTLASS GEMM on Flash-Next's shapes and compare with hanzo.

Same method as hanzo-quant/examples/blockwise_fp8_bench.rs: weights rotate
across copies that together exceed 2x L2 (48 MiB, at most 8), one CUDA graph
per shape replays one ops.cutlass_scaled_mm per copy, 3 warm replays, then 5
rounds of 4 timed replays between CUDA events.

  flock /tmp/hanzo-engine-build.lock systemd-run --user --scope --quiet \
    -p MemoryMax=6G -p MemorySwapMax=0 choom -n 1000 -- nice -n19 ionice -c3 \
    /home/z/vllm-env/bin/python scripts/blockwise_fp8_bench.py \
    --hanzo target/blockwise_fp8.hanzo.jsonl --out target/blockwise_fp8.vllm.jsonl

With several --hanzo/--vllm files (alternating runs), per-shape minima are
compared. Exits 1 if any shape has hanzo_min > 1.05 x vllm_min, or if
t(4097) > 1.1 x t(4096) x 4097/4096 on either engine's kernel (the M%4 slow path).
"""

import argparse
import json
import subprocess
import sys

import torch

L2_ROTATION = 48 << 20
MAX_COPIES = 8
MAX_OUTPUT = 256 << 20
WARM, ROUNDS, REPLAYS = 3, 5, 4
BLOCK = 128
SHAPES = [
    (16384, 2560), (13312, 2560), (1280, 2560), (2560, 6144), (2560, 640),
    (10240, 2560), (6144, 2560), (12288, 2560), (512, 2560), (640, 2560),
]
ROWS = [1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 257, 512, 1024, 2048, 4096, 4097, 8192]
GATE = 1.05
M4_GATE = 1.1


def copies(m, n, k):
    by_l2 = min(max(-(-L2_ROTATION // (n * k)), 1), MAX_COPIES)
    return min(by_l2, max(MAX_OUTPUT // (m * n * 2), 1))


def utilization():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def codes(shape, gen, dev):
    # Finite and unsaturated, like the Rust harness: timing only.
    b = torch.randint(0, 0x60, shape, generator=gen, dtype=torch.int32)
    sign = torch.randint(0, 2, shape, generator=gen, dtype=torch.int32) << 7
    return (b | sign).to(torch.uint8).view(torch.float8_e4m3fn).to(dev)


def copy_ceiling(nbytes, dev):
    src = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(src)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(ROUNDS):
        start.record()
        for _ in range(REPLAYS):
            dst.copy_(src)
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / REPLAYS)
    return 2 * nbytes / (best * 1e-6) / 1e9


def measure(out_path):
    from vllm import _custom_ops as ops

    dev = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(11)
    rows = []
    with open(out_path, "w") as out:
        for n, k in SHAPES:
            m_max = max(ROWS)
            qa_all = codes((m_max, k), gen, dev)
            weights = [(codes((n, k), gen, dev), torch.full((n // BLOCK, k // BLOCK), 1e-3, device=dev))
                       for _ in range(copies(1, n, k))]
            copy_gbps = copy_ceiling(n * k, dev)
            for m in ROWS:
                c = copies(m, n, k)
                qa = qa_all[:m]
                # vLLM's kernel reads activation scales M-major, as its quantizer writes them.
                sa = torch.full((k // BLOCK, m), 1e-2, device=dev).t()
                for qw, sw in weights[:c]:
                    ops.cutlass_scaled_mm(qa, qw.t(), sa, sw.t(), torch.bfloat16)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    outs = [ops.cutlass_scaled_mm(qa, qw.t(), sa, sw.t(), torch.bfloat16)
                            for qw, sw in weights[:c]]
                for _ in range(WARM):
                    g.replay()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                per = []
                for _ in range(ROUNDS):
                    start.record()
                    for _ in range(REPLAYS):
                        g.replay()
                    end.record()
                    end.synchronize()
                    per.append(start.elapsed_time(end) * 1e3 / (REPLAYS * c))
                del outs, g
                min_us = min(per)
                median_us = sorted(per)[len(per) // 2]
                nbytes = n * k + m * k + 2 * m * n + 4 * (n // BLOCK + m) * (k // BLOCK)
                row = {
                    "engine": "vllm", "n": n, "k": k, "m": m, "copies": c,
                    "min_us": min_us, "median_us": median_us,
                    "gbps": nbytes / (min_us * 1e-6) / 1e9,
                    "tflops": 2 * m * n * k / (min_us * 1e-6) / 1e12,
                    "copy_gbps": copy_gbps, "util": utilization(),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps(row), flush=True)
                rows.append(row)
            del weights, qa_all
            torch.cuda.empty_cache()
    print(f"peak device alloc {torch.cuda.max_memory_allocated(dev) / 2**20:.0f} MiB")
    return rows


def minima(paths):
    best = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                key = (r["n"], r["k"], r["m"])
                if key not in best or r["min_us"] < best[key]["min_us"]:
                    best[key] = r
    return best


def compare(hanzo, vllm):
    fails = []
    print(f"{'N':>6} {'K':>5} {'M':>5} {'hanzo us':>10} {'vllm us':>10} {'ratio':>6} "
          f"{'hanzo GB/s':>10} {'TFLOPS':>7} {'copy GB/s':>9}")
    for key in sorted(vllm):
        if key not in hanzo:
            fails.append(f"{key}: no hanzo measurement")
            continue
        h, v = hanzo[key], vllm[key]
        ratio = h["min_us"] / v["min_us"]
        mark = "" if ratio <= GATE else "  FAIL"
        print(f"{key[0]:>6} {key[1]:>5} {key[2]:>5} {h['min_us']:>10.2f} {v['min_us']:>10.2f} "
              f"{ratio:>6.3f} {h['gbps']:>10.1f} {h['tflops']:>7.1f} {h['copy_gbps']:>9.1f}{mark}")
        if ratio > GATE:
            fails.append(f"{key}: hanzo {h['min_us']:.2f} us > {GATE} x vllm {v['min_us']:.2f} us")
    for name, rows in (("hanzo", hanzo), ("vllm", vllm)):
        for n, k in SHAPES:
            a, b = rows.get((n, k, 4096)), rows.get((n, k, 4097))
            if a and b and b["min_us"] > M4_GATE * a["min_us"] * 4097 / 4096:
                fails.append(f"{name} ({n},{k}): t(4097)={b['min_us']:.1f} us > {M4_GATE} x t(4096)={a['min_us']:.1f} us")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hanzo", action="append", default=[], help="hanzo JSONL (repeatable)")
    ap.add_argument("--vllm", action="append", default=[], help="existing vLLM JSONL (repeatable)")
    ap.add_argument("--out", help="measure vLLM now and write its JSONL here")
    args = ap.parse_args()
    vllm_files = list(args.vllm)
    if args.out:
        measure(args.out)
        vllm_files.append(args.out)
    if not args.hanzo:
        return
    fails = compare(minima(args.hanzo), minima(vllm_files))
    for f in fails:
        print("FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
