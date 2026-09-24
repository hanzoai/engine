#!/usr/bin/env python3
"""vLLM goldens for the Flash-Next (qwen4_exp) MoE router, gate and shared expert.

Writes hanzo-engine/tests/fixtures/qwen4exp_moe/{route,gate,shared}.safetensors from
vLLM 0.29.0's own kernels, and checks them against hanzo's exact-lane router.

    route   every fused expert count x {f32,bf16,f16} through topk_softmax/topk_sigmoid
    gate    layer 23's router gate (N=512) and shared gate (N=1) through F.linear at
            every cuBLAS class and workspace-sensitive M, plus vLLM's routing of them
    shared  layer 23's shared expert: QuantFP8 + cutlass_scaled_mm, SiluAndMul, gate
    check   compiles hanzo-engine/src/cuda/exact/route.cu exact and with --use_fast_math
            and holds the goldens to it

The GB10 serves live traffic, so run one at a time, small, and at the lowest priority:

    for i in 1 2 3; do systemd-run --user --scope -p MemoryMax=3G nice -n19 ionice -c3 \\
        /home/z/vllm-env/bin/python scripts/qwen4exp_moe_golden.py route && break; done

It refuses to start below 6 GiB MemAvailable. A CUDA OOM at context creation is the box
being busy, not a bug; the loop above retries it at most three times.
"""
import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "hanzo-engine", "tests", "fixtures", "qwen4exp_moe")
ROUTE_CU = os.path.join(ROOT, "hanzo-engine", "src", "cuda", "exact", "route.cu")
VLLM = "/home/z/vllm-env/lib/python3.12/site-packages/vllm"
SNAP = ("/home/z/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
        "fc694b54fb0174e0913e6adf86691ef85a4ead47-fp8hybrid")
LAYER = 23
VLLM_LOG = "/home/z/spark-vllm.log"

# vLLM's fused topkGating counts (topk_softmax_kernels.cu switch); others go unfused.
COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 192, 320, 384, 448, 576]
DTYPES = ["f32", "bf16", "f16"]
# (name, score, renormalize, bias, routed_scaling_factor)
CONFIGS = [("softmax.renorm", "softmax", True, False, 1.0),
           ("softmax.plain", "softmax", False, False, 1.0),
           ("softmax.bias", "softmax", True, True, 1.0)]
for renorm in (True, False):
    for bias in (False, True):
        for rsf in (1.0, 2.5):
            CONFIGS.append((f"sigmoid.{'renorm' if renorm else 'plain'}.{'bias' if bias else 'nobias'}.rsf{rsf:g}",
                            "sigmoid", renorm, bias, rsf))
# Every cuBLAS class of the N=512 gate on GB10 ({1,17-183,257-320}, {2-16}, {184-256,321-})
# and every M range where a handle without torch's 32 MiB workspace differs (98-101,
# 126-160, 184-256, 289-320).
GATE_M = [1, 2, 3, 4, 5, 8, 16, 17, 64, 100, 128, 184, 256, 300, 512]
# vLLM's capture list with no speculation at max-num-seqs 8 (config/vllm.py:1996-2146).
NOSPEC = [1, 2, 4, 8, 16]
SHARED_T = [1, 16, 64]


def mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return 0.0


def torch_setup():
    if mem_available_gib() < 6:
        sys.exit(f"MemAvailable {mem_available_gib():.1f} GiB < 6 GiB; not starting")
    import torch
    torch.ops.load_library(os.path.join(VLLM, "_moe_C_stable_libtorch.abi3.so"))
    torch.cuda.init()
    return torch


def to_dtype(torch, x, dt):
    return x.to({"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[dt])


def route_inputs(torch, E):
    """Logit rows for E experts, in f32; each dtype rounds them its own way."""
    g = torch.Generator(device="cpu").manual_seed(1000 + E)
    rn = lambda n, s: torch.randn(n, E, generator=g) * s
    if E == 512:
        parts = [rn(256, 0.2), rn(256, 1.0), rn(256, 3.0)]
        parts.append(torch.randint(-4, 5, (256, E), generator=g).float() * 0.25)  # ties
        parts.append(torch.zeros(16, E))  # all equal
        under = rn(64, 1.0)
        under[:, :5] += 120.0  # only 5 normal probabilities; the rest underflow to 0
        parts.append(under)
        nan = rn(32, 1.0)
    else:
        parts = [rn(64, 1.0), torch.randint(-4, 5, (16, E), generator=g).float() * 0.25,
                 torch.zeros(4, E)]
        nan = rn(4, 1.0)
    nan[0::2, 0] = float("nan")
    nan[1::2, E - 1] = float("inf")
    if E > 2:
        nan[0::4, 1] = float("-inf")
    parts.append(nan)
    return torch.cat(parts)


def run_route(torch, x, cfg, bias):
    name, score, renorm, use_bias, rsf = cfg
    n, E = x.shape
    k = min(10, E)
    w = torch.empty(n, k, device="cuda", dtype=torch.float32)
    i = torch.empty(n, k, device="cuda", dtype=torch.int32)
    s = torch.empty(n, k, device="cuda", dtype=torch.int32)
    b = bias if use_bias else None
    if score == "softmax":
        torch.ops._moe_C.topk_softmax(w, i, s, x, renorm, b, None)
    else:
        torch.ops._moe_C.topk_sigmoid(w, i, s, x, renorm, b, rsf, None)
    torch.cuda.synchronize()
    return w.cpu(), i.cpu()


def build_route(torch):
    out = {}
    for E in COUNTS:
        g = torch.Generator(device="cpu").manual_seed(2000 + E)
        bias = torch.randn(E, generator=g) * 0.01
        out[f"bias.{E}"] = bias
        xf = route_inputs(torch, E)
        for dt in DTYPES:
            x = to_dtype(torch, xf, dt).contiguous()
            out[f"x.{E}.{dt}"] = x
            xd = x.cuda()
            for cfg in CONFIGS:
                w, i = run_route(torch, xd, cfg, bias.cuda())
                out[f"w.{E}.{dt}.{cfg[0]}"] = w
                out[f"i.{E}.{dt}.{cfg[0]}"] = i
    return out


def digest(tensors):
    h = hashlib.sha256()
    for key in sorted(tensors):
        t = tensors[key].contiguous()
        h.update(key.encode())
        h.update(t.view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def cublas_info(torch):
    torch.nn.functional.linear(torch.ones(1, 8, device="cuda", dtype=torch.bfloat16),
                               torch.ones(8, 8, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    path = None
    with open("/proc/self/maps") as f:
        for line in f:
            if "libcublas.so" in line and "Lt" not in line:
                path = line.split()[-1]
                break
    lib = ctypes.CDLL(path)
    ver = []
    for prop in range(3):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
        v = ctypes.c_int()
        assert lib.cublasGetProperty(prop, ctypes.byref(v)) == 0
        ver.append(v.value)
    return path, ver


def production_capture_list():
    """cudagraph_capture_sizes from the last engine init hanzo-vllm logged."""
    last = None
    with open(VLLM_LOG, errors="replace") as f:
        for line in f:
            if "Initializing a V1 LLM engine" in line or "'cudagraph_capture_sizes'" in line:
                m = re.search(r"'cudagraph_capture_sizes': \[([0-9, ]*)\]", line)
                if m:
                    last = [int(v) for v in m.group(1).split(",") if v.strip()]
    return last


def metadata(torch, extra):
    import importlib.metadata as md
    path, ver = cublas_info(torch)
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    meta = {
        "vllm": md.version("vllm"),
        "torch": torch.__version__,
        "nvidia-cublas": md.version("nvidia-cublas"),
        "cublas": ver,
        "libcublas": path,
        "gpu": torch.cuda.get_device_name(),
        "driver": drv,
    }
    meta.update(extra)
    return {k: v if isinstance(v, str) else json.dumps(v) for k, v in meta.items()}


def cmd_route(args):
    torch = torch_setup()
    from safetensors.torch import save_file
    out = build_route(torch)
    covered = [[E, dt] for E in COUNTS for dt in DTYPES if f"x.{E}.{dt}" in out]
    meta = metadata(torch, {"covered": covered, "configs": [c[0] for c in CONFIGS],
                            "k": "min(10, experts)", "sha256": digest(out)})
    os.makedirs(FIX, exist_ok=True)
    save_file(out, os.path.join(FIX, "route.safetensors"), metadata=meta)
    print(f"route: {len(covered)} (count, dtype) pairs x {len(CONFIGS)} configs, sha256 {meta['sha256'][:16]}")


def gate_inputs(torch):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
    p = f"model.language_model.layers.{LAYER}.mlp."
    names = {"wg": p + "gate.weight", "ws": p + "shared_expert_gate.weight",
             "embed": "model.language_model.embed_tokens.weight"}
    t = {}
    for key, name in names.items():
        with safe_open(os.path.join(SNAP, idx[name]), "pt") as f:
            s = f.get_slice(name)
            t[key] = s[:] if key != "embed" else None
            if key == "embed":
                g = torch.Generator(device="cpu").manual_seed(7)
                ids = torch.randperm(s.get_shape()[0], generator=g)[:512].sort().values
                t["ids"] = ids
                t[key] = torch.stack([s[int(r):int(r) + 1][0] for r in ids])
    x = t["embed"].float()
    x = x / x.pow(2).mean(-1, keepdim=True).sqrt()
    return x.to(torch.bfloat16).contiguous(), t["wg"].contiguous(), t["ws"].contiguous(), t["ids"]


def linear_graph_equals_eager(torch, x, w):
    F = torch.nn.functional
    eager = F.linear(x, w)
    xs = x.clone()
    ys = torch.empty_like(eager)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ys.copy_(F.linear(xs, w))
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ys.copy_(F.linear(xs, w))
    ys.zero_()
    g.replay()
    torch.cuda.synchronize()
    return eager, torch.equal(ys.view(torch.int16), eager.view(torch.int16))


def cmd_gate(args):
    torch = torch_setup()
    from safetensors.torch import save_file
    x, wg, ws, ids = gate_inputs(torch)
    xd, wgd, wsd = x.cuda(), wg.cuda(), ws.cuda()
    out = {"x": x, "wg": wg, "ws": ws, "token_ids": ids.to(torch.int32)}
    for M in GATE_M:
        lg, ok_g = linear_graph_equals_eager(torch, xd[:M], wgd)
        ls, ok_s = linear_graph_equals_eager(torch, xd[:M], wsd)
        assert ok_g and ok_s, f"graph replay != eager at M={M}"
        w, i = run_route(torch, lg, ("softmax.renorm", "softmax", True, False, 1.0), None)
        out[f"router.{M}"] = lg.cpu()
        out[f"shared.{M}"] = ls.cpu()
        out[f"w.{M}"] = w
        out[f"i.{M}"] = i
    meta = metadata(torch, {"layer": LAYER, "m": GATE_M, "graph_equals_eager": True,
                            "capture_nospec": NOSPEC,
                            "capture_production": production_capture_list(),
                            "x": "512 embed_tokens rows (token_ids), unit RMS, bf16",
                            "route": "topk_softmax k=10 renormalize, no bias"})
    os.makedirs(FIX, exist_ok=True)
    save_file(out, os.path.join(FIX, "gate.safetensors"), metadata=meta)
    print(f"gate: M {GATE_M}; graph == eager at every M; production capture {meta['capture_production']}")


def cmd_shared(args):
    torch = torch_setup()
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    gate = load_file(os.path.join(FIX, "gate.safetensors"))
    idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
    p = f"model.language_model.layers.{LAYER}.mlp.shared_expert."
    t = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("weight", "weight_scale_inv"):
            name = p + proj + "." + suffix
            with safe_open(os.path.join(SNAP, idx[name]), "pt") as f:
                t[proj + "." + suffix] = f.get_tensor(name)
    # vLLM's merged gate_up layout: gate rows then up rows, scales likewise.
    gu_w = torch.cat([t["gate_proj.weight"], t["up_proj.weight"]]).cuda()
    gu_s = torch.cat([t["gate_proj.weight_scale_inv"], t["up_proj.weight_scale_inv"]]).cuda()
    dn_w = t["down_proj.weight"].cuda()
    dn_s = t["down_proj.weight_scale_inv"].cuda()
    quant = QuantFP8(static=False, group_shape=GroupShape(1, 128), column_major_scales=True)

    def mm(a, w, s):
        aq, as_ = quant.forward_cuda(a)
        # CutlassFp8BlockScaledMMKernel: weight is [N, K]; cutlass takes K-major B.
        return ops.cutlass_scaled_mm(aq, w.t(), as_, s.t(), torch.bfloat16)

    act = SiluAndMul()
    compiled = torch.compile(act.forward_native, dynamic=False, fullgraph=True)
    out = {"gate_up.weight": gu_w.cpu(), "gate_up.weight_scale_inv": gu_s.cpu(),
           "down.weight": dn_w.cpu(), "down.weight_scale_inv": dn_s.cpu()}
    for T in SHARED_T:
        x = gate["x"][:T].cuda()
        gu = mm(x, gu_w, gu_s)
        h_c = compiled(gu)
        h_e = act.forward_native(gu)
        I = gu.shape[-1] // 2
        g32, u32 = gu[:, :I].float(), gu[:, I:].float()
        h_ref = (g32 / (1.0 + torch.exp(-g32)) * u32).to(torch.bfloat16)
        d = mm(h_c, dn_w, dn_s)
        g = gate[f"shared.{T}"].cuda()
        y = torch.sigmoid(g) * d
        torch.cuda.synchronize()
        out.update({f"gu.{T}": gu.cpu(), f"h.{T}": h_c.cpu(), f"h_eager.{T}": h_e.cpu(),
                    f"h_ref.{T}": h_ref.cpu(), f"d.{T}": d.cpu(), f"g.{T}": g.cpu(), f"y.{T}": y.cpu()})
    meta = metadata(torch, {"layer": LAYER, "t": SHARED_T,
                            "h": "SiluAndMul.forward_native under torch.compile (custom_ops none)",
                            "h_ref": "torch f32 g/(1+exp(-g))*u, one bf16 rounding",
                            "mm": "QuantFP8 group (1,128) column-major scales + cutlass_scaled_mm",
                            "y": "torch.sigmoid(g) * d, eager"})
    save_file(out, os.path.join(FIX, "shared.safetensors"), metadata=meta)
    print(f"shared: T {SHARED_T}")


def nvcc_build(dst, fast):
    flags = ["-std=c++17", "-O3", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
             "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
             "--expt-relaxed-constexpr", "--expt-extended-lambda", "-arch=sm_121"]
    if fast:
        flags.append("--use_fast_math")
    subprocess.run(["/usr/local/cuda-13.0/bin/nvcc", "-shared", "-Xcompiler", "-fPIC", *flags,
                    ROUTE_CU, "-o", dst], check=True)
    lib = ctypes.CDLL(dst)
    for dt in DTYPES:
        fn = getattr(lib, f"route_{dt}")
        fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_bool] * 2 + \
                      [ctypes.c_float] * 4 + [ctypes.c_int64]
        fn.restype = ctypes.c_int
    return lib


def lib_route(torch, lib, dt, x, cfg, bias):
    name, score, renorm, use_bias, rsf = cfg
    n, E = x.shape
    k = min(10, E)
    w = torch.empty(n, k, device="cuda", dtype=torch.float32)
    i = torch.empty(n, k, device="cuda", dtype=torch.int32)
    rc = getattr(lib, f"route_{dt}")(
        x.data_ptr(), w.data_ptr(), i.data_ptr(), bias.data_ptr() if use_bias else None, None,
        n, E, k, 1 if score == "softmax" else 2, 0, renorm, False, 0.0, 0.0, 0.0, rsf,
        torch.cuda.current_stream().cuda_stream)
    assert rc == 0, f"route_{dt} E={E} rc={rc}"
    torch.cuda.synchronize()
    return w.cpu(), i.cpu()


def cmd_check(args):
    torch = torch_setup()
    from safetensors import safe_open
    from safetensors.torch import load_file
    ok = True
    route = load_file(os.path.join(FIX, "route.safetensors"))
    with safe_open(os.path.join(FIX, "route.safetensors"), "pt") as f:
        meta = f.metadata()
    covered = {tuple(c) for c in json.loads(meta["covered"])}
    want = {(E, dt) for E in COUNTS for dt in DTYPES}
    if covered != want:
        print(f"FAIL coverage: missing {sorted(want - covered)}")
        ok = False
    tmp = tempfile.mkdtemp(prefix="route-check-")
    exact = nvcc_build(os.path.join(tmp, "exact.so"), False)
    fast = nvcc_build(os.path.join(tmp, "fast.so"), True)
    diff = {"exact": 0, "fast": 0}
    cases = 0
    for E in COUNTS:
        bias = route[f"bias.{E}"].cuda()
        for dt in DTYPES:
            x = route[f"x.{E}.{dt}"].cuda()
            for cfg in CONFIGS:
                wv, iv = route[f"w.{E}.{dt}.{cfg[0]}"], route[f"i.{E}.{dt}.{cfg[0]}"]
                if wv.shape != (x.shape[0], min(10, E)):
                    print(f"FAIL truncated golden E={E} {dt} {cfg[0]} {tuple(wv.shape)}")
                    ok = False
                cases += 1
                for name, lib in (("exact", exact), ("fast", fast)):
                    w, i = lib_route(torch, lib, dt, x, cfg, bias)
                    bad = int((w.view(torch.int32) != wv.view(torch.int32)).sum()) + int((i != iv).sum())
                    diff[name] += bad
                    if name == "exact" and bad:
                        print(f"FAIL exact E={E} {dt} {cfg[0]}: {bad} differing words")
    print(f"route: {cases} cases; exact build differing words {diff['exact']}, fast-math build {diff['fast']}")
    if diff["exact"] != 0:
        ok = False
    if diff["fast"] == 0:
        print("FAIL fast-math build matched vLLM bit for bit; the exact lane would be untested")
        ok = False
    again = build_route(torch)
    sha = digest(again)
    print(f"route rerun sha256 {sha[:16]} vs file {meta['sha256'][:16]} vs file bytes {digest(route)[:16]}")
    if not (sha == meta["sha256"] == digest(route)):
        print("FAIL route golden is not reproducible")
        ok = False
    with safe_open(os.path.join(FIX, "gate.safetensors"), "pt") as f:
        gm = f.metadata()
    if gm.get("graph_equals_eager") != "true" or json.loads(gm["m"]) != GATE_M:
        print("FAIL gate golden lacks the graph == eager assertion over every M")
        ok = False
    for name in ("route", "gate", "shared"):
        p = os.path.join(FIX, f"{name}.safetensors")
        if os.path.exists(p):
            print(f"{name}.safetensors {os.path.getsize(p) / 2**20:.1f} MiB")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("route", "gate", "shared", "check"):
        sub.add_parser(name)
    args = ap.parse_args()
    {"route": cmd_route, "gate": cmd_gate, "shared": cmd_shared, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
