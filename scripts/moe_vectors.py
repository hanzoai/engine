#!/usr/bin/env python3
"""Routed-expert (MoE) vectors and benchmarks from vLLM's own kernels, for hanzo_quant::experts.

Subcommands:
  gate --peak GIB     Can this box run a GPU step of that peak now? MemAvailable >= 12 GiB, then a
                      real CUDA context and a peak-sized allocation through libcuda. Exit 0, or 75
                      (BLOCKED-ON-MEMORY) with MemAvailable, SwapFree and /proc/buddyinfo. No torch.
  vectors             Golden stage vectors for the NVFP4 (E1) and block-FP8 (E4) experts into
                      hanzo-quant/tests/fixtures/moe_{nvfp4,fp8}.safetensors.
  bench               vLLM production-equivalent timings on the shapes examples/moe.rs times.

Every GPU step runs as

  scripts/moe_vectors.py gate --peak 1.5 && \\
  systemd-run --user --scope --quiet -p MemoryMax=6G -p MemorySwapMax=0 choom -n 1000 -- \\
    env MAX_JOBS=1 FLASHINFER_DISABLE_JIT=1 nice -n19 ionice -c3 \\
    /home/z/vllm-env/bin/python scripts/moe_vectors.py vectors --snapshot <fp8hybrid>

It never touches the served vLLM: at most 1.5 GiB of device memory (1.8 for bench), one layer's
tensors at a time, and FlashInfer's cached modules only.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

BLOCKED = 75
MIN_AVAILABLE_GIB = 12.0
ROOT = Path(__file__).resolve().parent.parent


def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemAvailable", "MemFree", "SwapFree"):
            out[k] = int(v.split()[0]) / (1 << 20)  # GiB
    return out


def vllm_up() -> bool:
    import subprocess

    r = subprocess.run(["systemctl", "--user", "is-active", "hanzo-vllm"], capture_output=True, text=True)
    s = subprocess.run(["systemctl", "is-active", "hanzo-vllm"], capture_output=True, text=True)
    return r.stdout.strip() == "active" or s.stdout.strip() == "active"


def snapshot() -> dict:
    m = meminfo()
    m["hanzo_vllm"] = vllm_up()
    try:
        m["buddyinfo"] = Path("/proc/buddyinfo").read_text().strip().splitlines()
    except OSError:
        pass
    return m


def blocked(why: str) -> None:
    print(f"BLOCKED-ON-MEMORY: {why} {json.dumps(snapshot())}", flush=True)
    sys.exit(BLOCKED)


def gate(peak_gib: float) -> None:
    m = meminfo()
    if m["MemAvailable"] < MIN_AVAILABLE_GIB:
        blocked(f"MemAvailable {m['MemAvailable']:.1f} GiB < {MIN_AVAILABLE_GIB}")
    cuda = ctypes.CDLL("libcuda.so.1")
    ok = lambda r: r == 0  # noqa: E731
    if not ok(cuda.cuInit(0)):
        print("gate: cuInit failed", flush=True)
        sys.exit(1)
    dev = ctypes.c_int()
    ctx = ctypes.c_void_p()
    if not ok(cuda.cuDeviceGet(ctypes.byref(dev), 0)):
        print("gate: no device", flush=True)
        sys.exit(1)
    r = cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    if r != 0:
        blocked(f"cuDevicePrimaryCtxRetain -> {r}")
    cuda.cuCtxSetCurrent(ctx)
    ptr = ctypes.c_uint64()
    size = ctypes.c_size_t(int(peak_gib * (1 << 30)))
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    r = cuda.cuMemAlloc_v2(ctypes.byref(ptr), size)
    if r != 0:
        cuda.cuDevicePrimaryCtxRelease(dev)
        blocked(f"cuMemAlloc({peak_gib} GiB) -> {r}")
    cuda.cuMemFree_v2(ptr)
    cuda.cuDevicePrimaryCtxRelease(dev)
    s = snapshot()
    s.pop("buddyinfo", None)
    print(f"gate: ok peak={peak_gib} GiB {json.dumps(s)}", flush=True)


# --------------------------------------------------------------------------------------------
# GPU helpers (torch is imported only by the GPU subcommands)
# --------------------------------------------------------------------------------------------

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def claim(peak_gib: float):
    """The first CUDA action: a context and the declared peak, or BLOCKED-ON-MEMORY (exit 75)."""
    import torch

    try:
        t = torch.empty(int(peak_gib * (1 << 30)), dtype=torch.uint8, device="cuda")
        del t
        torch.cuda.empty_cache()
    except (torch.OutOfMemoryError, RuntimeError) as e:  # context creation raises RuntimeError
        if "out of memory" in str(e).lower() or "OUT_OF_MEMORY" in str(e):
            blocked(f"torch claim of {peak_gib} GiB: {e}")
        raise
    torch.cuda.reset_peak_memory_stats()


def e2m1_table(device):
    import torch

    mag = torch.tensor(E2M1, dtype=torch.float32, device=device)
    return torch.cat([mag, -mag])


def dequant_nvfp4(codes, scales, global_scale):
    """[.., N, K/2] u8 codes (low nibble first) x [.., N, K/16] e4m3 x global -> f32 [.., N, K]."""
    import torch

    table = e2m1_table(codes.device)
    lo = table[(codes & 0xF).long()]
    hi = table[(codes >> 4).long()]
    v = torch.stack([lo, hi], dim=-1).reshape(*codes.shape[:-1], codes.shape[-1] * 2)
    s = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return v * s * global_scale


def decompose(values):
    """Exact (codes, e4m3 scale bytes) for block-scaled products code * scale, 16 per block.

    Every probe block is code * scale with code in E2M1 and scale in E4M3, so an exact
    decomposition exists; any exact one gives the same block-scaled GEMM products. Returns
    ([R, K/2] u8, [R, K/16] u8)."""
    import torch

    r, k = values.shape
    blocks = values.reshape(-1, 16).double()
    cand = torch.arange(1, 127, dtype=torch.uint8)  # positive finite e4m3
    cval = cand.view(torch.float8_e4m3fn).double().to(values.device)
    mag = torch.tensor(E2M1, dtype=torch.float64, device=values.device)
    q = blocks.abs().unsqueeze(1) / cval.view(1, -1, 1)  # [B, C, 16]
    near = (q.unsqueeze(-1) - mag).abs().amin(-1)
    ok = (near == 0).all(-1)  # [B, C]
    zero = (blocks == 0).all(-1)
    # Prefer the scale whose largest code is 6 (what the quantizer writes), else the first fit.
    top = (blocks.abs().amax(-1, keepdim=True) / cval.view(1, -1)) == 6.0
    pick = torch.where((ok & top).any(-1), (ok & top).float().argmax(-1), ok.float().argmax(-1))
    assert bool((ok.any(-1) | zero).all()), "a probe block has no exact decomposition"
    scale = torch.where(zero, torch.zeros_like(pick), cand.to(values.device)[pick].long())
    sval = torch.where(zero, torch.ones_like(cval[pick]), cval[pick])
    qv = blocks / sval.unsqueeze(-1)
    idx = (qv.abs().unsqueeze(-1) - mag).abs().argmin(-1)
    code = idx | torch.where(qv < 0, 8, 0)
    code = code.reshape(r, k).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, scale.reshape(r, k // 16).to(torch.uint8)


# --------------------------------------------------------------------------------------------
# NVFP4 experts (E1): FlashInfer cutlass_fused_moe as vLLM serves it
# --------------------------------------------------------------------------------------------


class Nvfp4Case:
    """One NVFP4 MoE problem in the loader's layout (w13 = gate rows then up rows, linear scales).

    `in13`/`in2` are the max input_scale over every expert of the layer (vLLM's one global
    activation scale), which the fixture stores per stored expert so a subset reproduces it."""

    def __init__(self, x, ids, weights, w13, s13, g13, in13, w2, s2, g2, in2):
        self.x, self.ids, self.weights = x, ids, weights
        self.w13, self.s13, self.g13, self.in13 = w13, s13, g13, in13
        self.w2, self.s2, self.g2, self.in2 = w2, s2, g2, in2

    @property
    def dims(self):
        e, n13, h2 = self.w13.shape
        return e, h2 * 2, n13 // 2


def fi_weights(c: Nvfp4Case, w2=None, s2=None, g2=None):
    """vLLM's FlashInfer transform: w13 to [up; gate], scales swizzled, alphas and gscales."""
    import torch
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

    e, h, i = c.dims
    w2 = c.w2 if w2 is None else w2
    s2 = c.s2 if s2 is None else s2
    w13 = torch.cat([c.w13[:, i:], c.w13[:, :i]], dim=1).contiguous()
    s13 = torch.cat([c.s13[:, i:], c.s13[:, :i]], dim=1).contiguous()
    s13 = swizzle_blockscale(s13.view(torch.float8_e4m3fn))
    s2s = swizzle_blockscale(s2.view(torch.float8_e4m3fn))
    in13 = torch.tensor(c.in13, dtype=torch.float32, device=c.x.device)
    in2 = torch.tensor(c.in2, dtype=torch.float32, device=c.x.device)
    g1 = c.g13 * in13
    g2 = (c.g2 * in2) if g2 is None else g2
    return dict(
        w13=w13, s13=s13, w2=w2, s2=s2s, g1=g1, g2=g2, a1g=1.0 / in13, a2g=1.0 / in2
    )


def fi_moe(c: Nvfp4Case, ids, weights, fw, profile_ids=None):
    import torch
    import vllm._custom_ops as ops
    from flashinfer.fused_moe import cutlass_fused_moe

    xq, xs = ops.scaled_fp4_quant(c.x, fw["a1g"], is_sf_swizzled_layout=True)
    out = torch.empty_like(c.x)
    cutlass_fused_moe(
        input=xq,
        token_selected_experts=ids.to(torch.int32).contiguous(),
        token_final_scales=weights.contiguous(),
        fc1_expert_weights=fw["w13"].view(torch.long),
        fc1_expert_biases=None,
        fc2_expert_weights=fw["w2"].view(torch.long),
        fc2_expert_biases=None,
        output_dtype=torch.bfloat16,
        quant_scales=[fw["a1g"], fw["s13"].view(torch.int32), fw["g1"], fw["a2g"],
                      fw["s2"].view(torch.int32), fw["g2"]],
        input_sf=xs,
        output=out,
        profile_ids=profile_ids,
    )
    return out


def nvfp4_stages(c: Nvfp4Case, tag: str, vec: dict, report: dict) -> None:
    """Stage vectors for one case, flat row f = t*k + j."""
    import torch
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

    dev = c.x.device
    e, h, i = c.dims
    m, k = c.ids.shape
    fw = fi_weights(c)

    # expand: the served input quantizer (linear scales)
    xq, xs = ops.scaled_fp4_quant(c.x, fw["a1g"], is_sf_swizzled_layout=False)
    vec[f"{tag}.xq"] = xq
    vec[f"{tag}.xs"] = xs.view(torch.uint8).reshape(m, h // 16)

    # GEMM1 per routed row: vLLM's dense NVFP4 GEMM against each expert, gate then up columns.
    flat = c.ids.reshape(-1).long()
    tok = torch.arange(m * k, device=dev) // k
    gemm1 = torch.zeros(m * k, 2 * i, dtype=torch.bfloat16, device=dev)
    for ex in flat.unique().tolist():
        rows = (flat == ex).nonzero().squeeze(1)
        xr = c.x[tok[rows]]
        q, s = ops.scaled_fp4_quant(xr, fw["a1g"], is_sf_swizzled_layout=True)
        wsf = swizzle_blockscale(c.s13[ex].view(torch.float8_e4m3fn))
        alpha = (c.g13[ex] * torch.tensor(c.in13, dtype=torch.float32, device=dev)).reshape(1)
        gemm1[rows] = ops.cutlass_scaled_fp4_mm(q, c.w13[ex], s, wsf, alpha, torch.bfloat16)
    vec[f"{tag}.gemm1"] = gemm1

    # act: FlashInfer's own fc2 input, read back through an identity w2, one slot at a time.
    assert i <= h, "the identity probe needs I <= H"
    ident = torch.zeros(e, h, i // 2, dtype=torch.uint8, device=dev)
    hh = torch.arange(i, device=dev)
    ident[:, hh, hh // 2] = torch.where(hh % 2 == 0, 0x02, 0x20).to(torch.uint8)
    ident_s = torch.full((e, h, i // 16), 0x38, dtype=torch.uint8, device=dev)
    fwp = fi_weights(c, w2=ident, s2=ident_s, g2=torch.ones(e, dtype=torch.float32, device=dev))
    act = torch.zeros(m * k, i, dtype=torch.float32, device=dev)
    ones = torch.ones(m, 1, dtype=torch.float32, device=dev)
    for j in range(k):
        o = fi_moe(c, c.ids[:, j : j + 1], ones, fwp)
        assert bool((o[:, i:] == 0).all()), "probe columns past I are not zero"
        act[j::k] = o[:, :i].float()
    vec[f"{tag}.act"] = act.to(torch.bfloat16)
    codes, scales = decompose(act)
    vec[f"{tag}.act_codes"] = codes
    vec[f"{tag}.act_scales"] = scales

    # GEMM2 per routed row, on the probe's activation.
    gemm2 = torch.zeros(m * k, h, dtype=torch.bfloat16, device=dev)
    for ex in flat.unique().tolist():
        rows = (flat == ex).nonzero().squeeze(1)
        s = swizzle_blockscale(scales[rows].view(torch.float8_e4m3fn))
        wsf = swizzle_blockscale(c.s2[ex].view(torch.float8_e4m3fn))
        alpha = (c.g2[ex] * torch.tensor(c.in2, dtype=torch.float32, device=dev)).reshape(1)
        gemm2[rows] = ops.cutlass_scaled_fp4_mm(codes[rows], c.w2[ex], s, wsf, alpha, torch.bfloat16)
    vec[f"{tag}.gemm2"] = gemm2

    # Final: the production call, and the proof that it runs tactic 0 of each GEMM.
    out = fi_moe(c, c.ids, c.weights, fw)
    vec[f"{tag}.out"] = out
    n1 = None
    for cand in range(1, 256):
        try:
            o2 = fi_moe(c, c.ids, c.weights, fw, profile_ids=[0, cand])
        except Exception:  # noqa: BLE001 - FlashInfer rejects gemm2 ids inside the gemm1 range
            continue
        n1 = cand
        break
    assert n1 is not None, "no gemm2 tactic accepted"
    assert torch.equal(out.view(torch.int16), o2.view(torch.int16)), "default != [0, gemm1_count]"
    report[f"{tag}.gemm1_tactics"] = n1

    # Composition: finalize (fma.ftz, slot order, from +0) of the per-expert GEMM2 rows.
    y = gemm2.float().reshape(m, k, h)
    s = torch.zeros(m, h, dtype=torch.float32, device=dev)
    for j in range(k):
        s = torch.addcmul(s, c.weights[:, j : j + 1], y[:, j])
    comp = s.to(torch.bfloat16)
    g = gap(comp, out)
    g["bit_exact"] = float((comp.view(torch.int16) == out.view(torch.int16)).float().mean())
    report[f"{tag}.composition"] = g
    assert g["max_ulp"] <= 1.0 and g["bit_exact"] >= 0.999, g

    # fp32 reference: dequantized weights, unquantized activations.
    w13f = dequant_nvfp4(c.w13, c.s13, c.g13.view(-1, 1, 1))
    w2f = dequant_nvfp4(c.w2, c.s2, c.g2.view(-1, 1, 1))
    ref = torch.zeros(m, h, dtype=torch.float32, device=dev)
    xf = c.x.float()
    for j in range(k):
        for ex in c.ids[:, j].unique().tolist():
            rows = (c.ids[:, j] == ex).nonzero().squeeze(1)
            gu = xf[rows] @ w13f[ex].T
            a = torch.nn.functional.silu(gu[:, :i]) * gu[:, i:]
            ref[rows] += c.weights[rows, j : j + 1] * (a @ w2f[ex].T)
    vec[f"{tag}.ref32"] = ref
    dist = float((out.float() - ref).norm() / ref.norm())
    report[f"{tag}.fi_vs_ref32"] = dist
    assert 0.01 <= dist <= 0.3, dist


def nvfp4_real(ck, golden, report) -> tuple[dict, Nvfp4Case]:
    """Layer 3, the three experts its router picks most often, M=128, k=2."""
    import torch
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

    import qwen4exp_golden as G

    dev = torch.device("cuda")
    ids = golden["l3.ids"].reshape(-1)
    top = torch.bincount(ids.long(), minlength=512).argsort(descending=True)[:3].sort().values
    sel = top.tolist()
    p = G.PREFIX + "layers.3.mlp"
    u = golden["l3.mlp.u"].to(dev)
    gen = torch.Generator(device="cpu").manual_seed(83)
    sigma = float(u.float().std())
    noise = (torch.randn(128 - u.shape[0], u.shape[1], generator=gen) * sigma).to(torch.bfloat16)
    x = torch.cat([u, noise.to(dev)]).contiguous()
    gate = ck.get(f"{p}.gate.weight").to(dev)[top.to(dev)]
    logits = x @ gate.T
    weights, lids, _ = fused_topk(x, logits, 2, True)
    in13 = max(float(ck.get(f"{p}.experts.{e}.{n}_proj.input_scale")) for e in range(512) for n in ("gate", "up"))
    in2 = max(float(ck.get(f"{p}.experts.{e}.down_proj.input_scale")) for e in range(512))
    w13, s13, g13, w2, s2, g2 = [], [], [], [], [], []
    for e in sel:
        q = f"{p}.experts.{e}"
        w13.append(torch.cat([ck.get(f"{q}.gate_proj.weight"), ck.get(f"{q}.up_proj.weight")]))
        s13.append(torch.cat([ck.get(f"{q}.gate_proj.weight_scale"), ck.get(f"{q}.up_proj.weight_scale")]).view(torch.uint8))
        gg, gu = float(ck.get(f"{q}.gate_proj.weight_scale_2")), float(ck.get(f"{q}.up_proj.weight_scale_2"))
        assert gg == gu, (e, gg, gu)
        g13.append(gg)
        w2.append(ck.get(f"{q}.down_proj.weight"))
        s2.append(ck.get(f"{q}.down_proj.weight_scale").view(torch.uint8))
        g2.append(float(ck.get(f"{q}.down_proj.weight_scale_2")))
    t = lambda v: torch.stack(v).to(dev).contiguous()  # noqa: E731
    f = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)  # noqa: E731
    c = Nvfp4Case(x, lids.to(torch.int32), weights, t(w13), t(s13), f(g13), in13, t(w2), t(s2), f(g2), in2)
    report["r.experts"] = sel
    vec = {
        "r.x": x, "r.ids": c.ids, "r.weights": c.weights,
        "r.w13": c.w13, "r.w13_scale": c.s13, "r.w13_global": c.g13,
        "r.w13_input": torch.full((3,), in13, dtype=torch.float32, device=dev),
        "r.w2": c.w2, "r.w2_scale": c.s2, "r.w2_global": c.g2,
        "r.w2_input": torch.full((3,), in2, dtype=torch.float32, device=dev),
    }
    return vec, c


def nvfp4_synthetic(report) -> tuple[dict, Nvfp4Case]:
    """E=512, k=10, H=I=128, M=4 (a normal row, zeros, saturating, tiny); only active experts
    are stored, under their global ids (`s.experts`)."""
    import torch

    dev = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(512)
    e, h, i, m, k = 512, 128, 128, 4, 10
    x = torch.randn(m, h, generator=gen)
    x[1] = 0.0
    x[2] = torch.sign(torch.randn(h, generator=gen)) * 3.0e4
    x[3] = torch.randn(h, generator=gen) * 1e-6
    x = x.to(torch.bfloat16).to(dev)
    ids = torch.stack([torch.randperm(e, generator=gen)[:k] for _ in range(m)]).to(torch.int32)
    w = torch.rand(m, k, generator=gen) + 0.05
    w = (w / w.sum(-1, keepdim=True)).to(dev)
    codes13 = torch.randint(0, 256, (e, 2 * i, h // 2), generator=gen, dtype=torch.uint8)
    codes2 = torch.randint(0, 256, (e, h, i // 2), generator=gen, dtype=torch.uint8)
    sc = lambda *s: torch.randint(0x28, 0x40, s, generator=gen, dtype=torch.uint8)  # noqa: E731
    s13, s2 = sc(e, 2 * i, h // 16), sc(e, h, i // 16)
    g13 = (torch.rand(e, generator=gen) * 0.02 + 0.01)
    g2 = (torch.rand(e, generator=gen) * 0.02 + 0.01)
    c = Nvfp4Case(x, ids.to(dev), w, codes13.to(dev), s13.to(dev), g13.to(dev), 0.05,
                  codes2.to(dev), s2.to(dev), g2.to(dev), 0.02)
    act = ids.reshape(-1).unique().long()
    report["s.active"] = int(act.numel())
    vec = {
        "s.x": x, "s.ids": c.ids, "s.weights": w, "s.experts": act.to(torch.int32).to(dev),
        "s.w13": c.w13[act], "s.w13_scale": c.s13[act], "s.w13_global": c.g13[act],
        "s.w13_input": torch.full((act.numel(),), 0.05, dtype=torch.float32, device=dev),
        "s.w2": c.w2[act], "s.w2_scale": c.s2[act], "s.w2_global": c.g2[act],
        "s.w2_input": torch.full((act.numel(),), 0.02, dtype=torch.float32, device=dev),
    }
    return vec, c


def vectors(args) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import qwen4exp_golden as G
    import qwen4exp_vectors as QV

    claim(1.5)
    QV.forbid_jit()
    global gap
    gap = QV.gap
    report: dict = {"memory": snapshot()}
    report["memory"].pop("buddyinfo", None)
    golden = load_file(str(args.golden))
    ck = G.Checkpoint(args.snapshot)
    vec: dict = {}

    v, c = nvfp4_real(ck, golden, report)
    vec.update(v)
    nvfp4_stages(c, "r", vec, report)
    del c
    torch.cuda.empty_cache()
    print("nvfp4 real done", flush=True)
    v, c = nvfp4_synthetic(report)
    vec.update(v)
    nvfp4_stages(c, "s", vec, report)
    del c
    torch.cuda.empty_cache()
    print("nvfp4 synthetic done", flush=True)

    peak = torch.cuda.max_memory_allocated()
    report["device_peak_gib"] = peak / (1 << 30)
    assert peak <= int(1.5 * (1 << 30)), peak
    meta = {
        "vllm": __import__("vllm").__version__,
        "flashinfer": __import__("flashinfer").__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "golden_sha256": G.sha256(args.golden),
        "report": json.dumps(report),
    }
    out = ROOT / "hanzo-quant/tests/fixtures/moe_nvfp4.safetensors"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: t.detach().contiguous().cpu() for k, t in vec.items()}, str(out), metadata=meta)
    size = out.stat().st_size
    print(f"wrote {out} ({size / 1e6:.2f} MB, {len(vec)} tensors) sha256 {G.sha256(out)}")
    print(json.dumps(report, indent=1))
    assert size <= 30e6, size


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--peak", type=float, required=True, help="GiB the step will allocate")
    v = sub.add_parser("vectors")
    v.add_argument("--snapshot", required=True, type=Path)
    v.add_argument("--golden", type=Path, default=Path("/data/engine-qwen4/hanzo-engine/tests/fixtures/qwen4exp.safetensors"))
    args = ap.parse_args()
    if args.cmd == "gate":
        gate(args.peak)
    elif args.cmd == "vectors":
        try:
            vectors(args)
        except Exception as e:  # noqa: BLE001
            if "out of memory" in str(e).lower():
                blocked(f"vectors: {e}")
            raise


if __name__ == "__main__":
    main()
