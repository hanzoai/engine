#!/usr/bin/env python3
"""Router device time: hanzo's route.cu (the engine's own exact-lane archive) vs vLLM's
topk_softmax, E=512, k=10, bf16, T in {1, 8, 64, 512, 4096}.

Device time comes from CUDA-graph replay: each side captures 100 launches in one graph, the two
graphs replay interleaved for 20 pairs, and the bar is on the median paired ratio (hanzo/vLLM in
[0.95, 1.05] at every T). A pair counts only if nvidia-smi reports <= 5% GPU utilization before
and after it (the GB10 serves live traffic); fewer than 20 clean pairs in 200 attempts is
INCONCLUSIVE and reported with its numbers. Eager host time per call is reported without a bar.

    systemd-run --user --scope -p MemoryMax=3G nice -n19 ionice -c3 \\
        /home/z/vllm-env/bin/python scripts/qwen4exp_moe_bench.py
"""
import argparse
import ctypes
import glob
import os
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLLM = "/home/z/vllm-env/lib/python3.12/site-packages/vllm"
TS = [1, 8, 64, 512, 4096]
E, K = 512, 10
PAIRS, ATTEMPTS, LAUNCHES = 20, 200, 100


def gpu_util():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


def engine_lib():
    archives = sorted(glob.glob(os.path.join(ROOT, "target", "*", "build", "hanzo-engine-*", "out",
                                             "libhanzocudaexact.a")), key=os.path.getmtime)
    if not archives:
        sys.exit("no libhanzocudaexact.a: build hanzo-engine with --features cuda first")
    so = os.path.join(tempfile.mkdtemp(prefix="route-bench-"), "exact.so")
    subprocess.run(["g++", "-shared", "-o", so, "-Wl,--whole-archive", archives[-1], "-Wl,--no-whole-archive",
                    "-L/usr/local/cuda-13.0/lib64", "-lcudart"], check=True)
    lib = ctypes.CDLL(so)
    lib.route_bf16.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_bool] * 2 + \
                              [ctypes.c_float] * 4 + [ctypes.c_int64]
    lib.route_bf16.restype = ctypes.c_int
    return lib, archives[-1]


def graph(torch, fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(LAUNCHES):
            fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def replay_us(torch, g):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / LAUNCHES


def paired(torch, ga, gb, gate):
    ratios, ta, tb, attempts = [], [], [], 0
    while len(ratios) < PAIRS and attempts < ATTEMPTS:
        attempts += 1
        before = gpu_util() if gate else 0
        x = replay_us(torch, ga)
        y = replay_us(torch, gb)
        time.sleep(1.0)
        after = gpu_util() if gate else 0
        if before > 5 or after > 5:
            continue
        ta.append(x)
        tb.append(y)
        ratios.append(x / y)
    return ratios, ta, tb, attempts


def eager_us(torch, fn, n=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e6 / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-gate", action="store_true", help="skip the nvidia-smi utilization gate")
    args = ap.parse_args()
    import torch
    torch.ops.load_library(os.path.join(VLLM, "_moe_C_stable_libtorch.abi3.so"))
    lib, archive = engine_lib()
    print(f"hanzo: {os.path.relpath(archive, ROOT)}; vLLM: _moe_C_stable_libtorch topk_softmax")
    print(f"{'T':>5} | {'hanzo us':>9} {'vLLM us':>9} {'ratio':>6} {'pairs':>9} | {'hanzo host':>10} {'vLLM host':>10} | bar")
    ok = True
    torch.manual_seed(0)
    for T in TS:
        x = torch.randn(T, E, device="cuda").to(torch.bfloat16)
        wa = torch.empty(T, K, device="cuda")
        ia = torch.empty(T, K, device="cuda", dtype=torch.int32)
        wb, ib, sb = torch.empty_like(wa), torch.empty_like(ia), torch.empty_like(ia)

        def a():
            rc = lib.route_bf16(x.data_ptr(), wa.data_ptr(), ia.data_ptr(), None, None, T, E, K, 1, 0,
                                True, False, 0.0, 0.0, 0.0, 1.0, torch.cuda.current_stream().cuda_stream)
            assert rc == 0

        def b():
            torch.ops._moe_C.topk_softmax(wb, ib, sb, x, True, None, None)

        a(); b(); torch.cuda.synchronize()
        assert torch.equal(ia, ib) and torch.equal(wa.view(torch.int32), wb.view(torch.int32))
        ga, gb = graph(torch, a), graph(torch, b)
        ratios, ta, tb, attempts = paired(torch, ga, gb, not args.no_gate)
        ha, hb = eager_us(torch, a), eager_us(torch, b)
        if len(ratios) < PAIRS:
            verdict = "INCONCLUSIVE"
            ok = False
        else:
            r = statistics.median(ratios)
            verdict = "PASS" if 0.95 <= r <= 1.05 else "FAIL"
            ok &= verdict == "PASS"
        med = lambda v: statistics.median(v) if v else float("nan")
        print(f"{T:5d} | {med(ta):9.2f} {med(tb):9.2f} {med(ratios):6.3f} {len(ratios):3d}/{attempts:<5d} | "
              f"{ha:10.2f} {hb:10.2f} | {verdict}")
    print("PASS" if ok else "NOT PASSED")


if __name__ == "__main__":
    main()
