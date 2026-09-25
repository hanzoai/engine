#!/usr/bin/env python3
"""Served-kernel vectors for qwen4exp: vLLM's own GPU kernels on the golden's inputs.

The CPU golden (qwen4exp_golden.py) transcribes the served kernels with IEEE arithmetic. This
script runs the served kernels themselves, on the golden's inputs and on synthetic edge rows, so the
engine's tests can hold its quantizers to bit equality with the server and its blocks to the
server's own outputs. It records how far the golden sits from the server, stage by stage.

It never touches the running vLLM server: it loads one layer's tensors at a time from the
checkpoint, uses at most 1.2 GiB of device memory, and refuses to JIT-compile FlashInfer (the
cached fused_moe_120 and fp4_quantization_120f modules must load as they are).

  systemd-run --user --scope --quiet -p MemoryMax=6G -p MemorySwapMax=0 choom -n 1000 -- \
    env MAX_JOBS=1 nice -n19 ionice -c3 ~/vllm-env/bin/python \
    scripts/qwen4exp_vectors.py --snapshot <fp8hybrid> \
      --golden hanzo-engine/tests/fixtures/qwen4exp.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MAX_JOBS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen4exp_golden as G  # noqa: E402

DEV = torch.device("cuda")
BF16 = torch.bfloat16
F32 = torch.float32
LIMIT = int(1.2 * (1 << 30))
ROOT = Path(__file__).resolve().parent.parent


def forbid_jit() -> None:
    """FlashInfer must load its cached modules; building one is an error, not a side effect.

    A JIT module is loaded after `build()`, where ninja no-ops on an up-to-date cache. This
    `build()` leaves the served build.ninja untouched and only asks ninja, dry-run, whether
    anything would compile: nothing (or a cached .so of the same version) loads it, anything else
    is an error.
    (FLASHINFER_DISABLE_JIT refuses before that check, so it cannot be used here.)"""
    import subprocess

    from flashinfer.jit import core

    def check(self, *a, **k):
        ninja = Path(self.ninja_path)
        if not ninja.exists():
            raise RuntimeError(f"FlashInfer module {self.name} is not in the JIT cache")
        out = subprocess.run(
            ["ninja", "-n", "-C", str(Path(self.build_dir).resolve()), "-f", str(ninja.resolve())],
            capture_output=True, text=True, check=True,
        ).stdout
        if "no work to do" in out:
            return
        # A cached .so whose build.ninja another environment wrote (same FlashInfer version, so
        # the same sources): load it as it is rather than rebuild.
        if Path(self.jit_library_path).exists():
            print(f"FlashInfer {self.name}: ninja is stale, loading the cached {self.jit_library_path}")
            return
        raise RuntimeError(f"FlashInfer would JIT-build {self.name}:\n{out[:2000]}")

    for cls in (core.JitSpec, *core.JitSpec.__subclasses__()):
        cls.build = check


# --------------------------------------------------------------------------------------------
# inductor kernels, verbatim from the live compile cache
# --------------------------------------------------------------------------------------------


def inductor_kernel(path: Path, name: str):
    """The @triton.jit body of `name` from an inductor output module, compiled standalone."""
    import triton
    import triton.language as tl
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice
    from torch._inductor.runtime.triton_helpers import math as tl_math

    text = path.read_text()
    m = re.search(rf"{name} = async_compile\.triton\('{name}', '''(.*?)''', device_str", text, re.S)
    if m:
        src = m.group(1)
        body = src[src.index("@triton.jit") :]
    else:
        # a standalone kernel module: the file is the kernel
        at = text.index(f"@triton.jit\ndef {name}(")
        body = text[at:]
    scope = {
        "triton": triton,
        "tl": tl,
        "triton_helpers": triton_helpers,
        "libdevice": libdevice,
        "tl_math": tl_math,
    }
    # a real file, so triton can read the source back
    cache = ROOT / "target" / "qwen4exp-vectors"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{name}_{path.stem[:8]}.py"
    f.write_text(
        "import triton\nimport triton.language as tl\n"
        "from torch._inductor.runtime import triton_helpers\n"
        "from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math\n\n" + body
    )
    code = compile(f.read_text(), str(f), "exec")
    exec(code, scope)
    return scope[name]


INDUCTOR = G.INDUCTOR
FUSED = INDUCTOR / "d7/cd76kp4tkxhvavtr4sv37bs4zyaxtb3ozm22umsosh3ggoncajyl.py"
GATED = INDUCTOR / "7w/c7wgpo357q2hrlrcgvyaumoyoqeqjzziqapm4jzorvvnkqhetdic.py"
EAGER = INDUCTOR / "7l/c7lvd6lba4iozfx55hisw2larwxcmcpvff34nriwmo3gvhsvtg2j.py"
KERN = "triton_per_fused__to_copy_abs_clamp_clone_cutlass_scaled_mm_div_max_permute_squeeze_transpose_view"
KERN_G = "triton_per_fused__to_copy_abs_clamp_clone_cutlass_scaled_mm_div_max_mul_permute_sigmoid_squeeze_transpose_view_0"
KERN_E = "triton_per_fused__to_copy_abs_clamp_div_max_view_0"


def fp8_linear(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused W8A8 input quantizer as served: kernel _2 for 2560-wide rows (hc_gate_mix out),
    kernel _0 for 6144-wide rows (the delta-net core into out_proj)."""
    t, k = x.shape
    kern = inductor_kernel(FUSED, f"{KERN}_{2 if k == 2560 else 0}")
    assert k in (2560, 6144), k
    s = torch.empty((t, k // 128), dtype=F32, device=DEV).t().contiguous().t()  # stride (1, t)
    q = torch.empty((t, k), dtype=torch.float8_e4m3fn, device=DEV)
    kern[(t * (k // 128),)](x.contiguous(), s, q, t, t * (k // 128), 128, XBLOCK=1, num_warps=2)
    return q, s.contiguous()


def fp8_gated(core: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The o_proj quantizer fused with attn * sigmoid(gate) (f32 product, never stored)."""
    t, k = core.shape
    kern = inductor_kernel(GATED, KERN_G)
    s = torch.empty((t, k // 128), dtype=F32, device=DEV).t().contiguous().t()
    q = torch.empty((t, k), dtype=torch.float8_e4m3fn, device=DEV)
    kern[(t * (k // 128),)](core.contiguous(), gate.contiguous(), s, q, t, t * (k // 128), 128, XBLOCK=1, num_warps=2)
    return q, s.contiguous()


def fp8_eager(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The standalone QuantFP8 kernel (amax / 448 with a runtime divisor), run inside
    moe_forward_shared for the shared expert."""
    t, k = x.shape
    groups = k // 128
    quant = inductor_kernel(EAGER, KERN_E)
    scale = inductor_kernel(EAGER, "triton_poi_fused__to_copy_clamp_clone_squeeze_transpose_1")
    q = torch.empty((t, k), dtype=torch.float8_e4m3fn, device=DEV)
    amax = torch.empty((t, groups), dtype=BF16, device=DEV)
    quant[(t * groups,)](x.contiguous(), 448.0, amax, q, t * groups, 128, XBLOCK=1, num_warps=2)
    # the graph's second kernel: s = max(amax / 448, 1/229376), stored column major
    s = torch.empty((groups, t), dtype=F32, device=DEV)
    scale[(t, groups, 1)](amax, 448.0, s, k, t, groups, t, YBLOCK=1, XBLOCK=1, num_warps=1)
    return q, s.t().contiguous()


# --------------------------------------------------------------------------------------------
# synthetic edge rows
# --------------------------------------------------------------------------------------------


def synthetic() -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    rows = []
    k = 2560
    rows.append(torch.zeros(k))  # an all-zero row
    rows.append(torch.full((k,), -0.0))  # negative zeros
    for scale in (1e-3, 5e-4, 1.9e-3):  # amax below 1/512
        rows.append(torch.randn(k, generator=g) * scale / 4)
    for p in range(-6, 7, 3):  # amax on powers of two
        r = torch.randn(k, generator=g).clamp(-1.9, 1.9) / 2
        r[::128] = 2.0**p
        rows.append(r * 2.0**p)
    for v in (448.0, 1000.0, 3.0e4):  # saturating rows
        r = torch.randn(k, generator=g)
        r[5::128] = v
        rows.append(r)
    # e4m3 ties: values half-way between codes after scaling by amax/448
    r = torch.zeros(k)
    r[::128] = 448.0
    r[1::128] = 1.0625
    r[2::128] = 17.0 / 16 * 2.0**-7
    r[3::128] = 2.0**-9 * 1.5
    rows.append(r)
    # e2m1 ties: block max 6 so the scale is 1, then values on the midpoints
    r = torch.zeros(k)
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 6.0, -6.0])
    r[:] = ties.repeat(k // 16)
    rows.append(r)
    while len(rows) < 64:
        scale = [1e-4, 1e-2, 1.0, 1e2][len(rows) % 4]
        rows.append(torch.randn(k, generator=g) * scale)
    return torch.stack(rows).to(BF16)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def ulps(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """|got - want| in bf16 ulps at |want|, with a floor of 2^-10 * rms(want) per row."""
    got = got.float().cpu()
    want = want.float().cpu()
    exp = torch.floor(torch.log2(want.abs().clamp(min=1e-30)))
    ulp = torch.pow(2.0, exp - 7)
    rms = want.pow(2).mean(-1, keepdim=True).sqrt()
    return (got - want).abs() / (ulp + rms * 2.0**-10)


def gap(got: torch.Tensor, want: torch.Tensor) -> dict:
    u = ulps(got, want).reshape(-1)
    g2 = got.float().cpu().reshape(want.shape[0], -1)
    w2 = want.float().cpu().reshape(want.shape[0], -1)
    cos = torch.nn.functional.cosine_similarity(g2, w2, dim=-1)
    cos = torch.where(torch.isnan(cos), torch.ones_like(cos), cos)
    return {
        "max_ulp": float(u.max()),
        "p999_ulp": float(torch.quantile(u.double(), 0.999)),
        "min_cos": float(cos.min()),
    }


def code_gap(got: torch.Tensor, want: torch.Tensor) -> dict:
    """Quantizer codes: fraction of elements that differ, and the largest step."""
    g = got.view(torch.uint8).cpu().to(torch.int32) if got.dtype != torch.uint8 else got.cpu().to(torch.int32)
    w = want.view(torch.uint8).cpu().to(torch.int32) if want.dtype != torch.uint8 else want.cpu().to(torch.int32)
    diff = (g != w)
    return {"differ": float(diff.float().mean()), "count": int(diff.sum())}


# --------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--golden", required=True, type=Path)
    ap.add_argument("--quant-out", type=Path, default=ROOT / "hanzo-quant/tests/fixtures/vllm_vectors.safetensors")
    ap.add_argument("--out", type=Path, default=ROOT / "hanzo-engine/tests/fixtures/qwen4exp_vectors.safetensors")
    args = ap.parse_args()

    G.check_kernels()
    free, _ = torch.cuda.mem_get_info()
    assert free >= 4 << 30, f"only {free / 2**30:.1f} GiB free on the device"
    forbid_jit()
    torch.cuda.reset_peak_memory_stats()

    from safetensors.torch import load_file, save_file

    gold = {k: v for k, v in load_file(str(args.golden)).items()}

    def gd(name: str) -> torch.Tensor:
        return gold[name].to(DEV)

    ck = G.Checkpoint(args.snapshot)
    qv: dict[str, torch.Tensor] = {}  # quantizer vectors (inputs and outputs)
    ev: dict[str, torch.Tensor] = {}  # engine vectors (outputs)
    report: dict[str, dict] = {}

    import vllm._custom_ops as ops

    # ---------------- quantizers ----------------
    syn = synthetic().to(DEV)
    qv["syn.x"] = syn
    for tag, x in [("syn", syn), ("l0u", gd("l0.attn.u")), ("l3u", gd("l3.attn.u")), ("l3mu", gd("l3.mlp.u"))]:
        if tag != "syn":
            qv[f"x.{tag}"] = x
        q, s = fp8_linear(x)
        qv[f"fp8.linear.{tag}.q"], qv[f"fp8.linear.{tag}.s"] = q, s
        if tag != "syn":
            gq, gs = G.fp8_quant(x.float().cpu(), "linear")
            report[f"fp8.linear.{tag}.golden"] = code_gap(q, gq)
    x = gd("l0.gdn.normed")
    qv["fp8.linear.l0n.x"] = x
    q, s = fp8_linear(x)
    qv["fp8.linear.l0n.q"], qv["fp8.linear.l0n.s"] = q, s
    report["fp8.linear.l0n.golden"] = code_gap(q, G.fp8_quant(x.float().cpu(), "linear")[0])

    core, gate = gd("l3.attn.core"), gd("l3.attn.gate")
    q, s = fp8_gated(core, gate)
    qv["fp8.gated.core"], qv["fp8.gated.gate"] = core, gate
    qv["fp8.gated.q"], qv["fp8.gated.s"] = q, s
    report["fp8.gated.golden"] = code_gap(q, gold["l3.attn.oq"])

    for tag, x in [("syn", syn), ("l3mu", gd("l3.mlp.u")), ("l3act", gd("l3.shared.act"))]:
        q, s = fp8_eager(x)
        qv[f"fp8.eager.{tag}.q"], qv[f"fp8.eager.{tag}.s"] = q, s
        if tag != "syn":
            report[f"fp8.eager.{tag}.golden"] = code_gap(q, G.fp8_quant(x.float().cpu(), "eager")[0])
    qv["fp8.eager.l3act.x"] = gd("l3.shared.act")

    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    q, s = per_token_group_quant_fp8(syn, 128, eps=1e-10, dtype=torch.float8_e4m3fn, use_ue8m0=False)
    qv["fp8.group.syn.q"], qv["fp8.group.syn.s"] = q, s

    gs1, gs2 = [float(v) for v in gold["l3.mlp.gs"]]
    qv["fp4.gs"] = torch.tensor([gs1, gs2], dtype=F32)
    for tag, x in [("syn", syn), ("l3mu", gd("l3.mlp.u"))]:
        q, s = ops.scaled_fp4_quant(x, torch.tensor([gs1], dtype=F32, device=DEV), is_sf_swizzled_layout=False)
        qv[f"fp4.input.{tag}.q"], qv[f"fp4.input.{tag}.s"] = q, s.view(torch.float8_e4m3fn).reshape(x.shape[0], -1)
    gq, gs_ = G.nvfp4_quant(gold["l3.mlp.u"].float(), gs1)
    packed = (gq[:, 0::2] | (gq[:, 1::2] << 4)).to(torch.uint8)
    report["fp4.input.l3mu.golden"] = {
        "codes": code_gap(qv["fp4.input.l3mu.q"], packed),
        "scales": code_gap(qv["fp4.input.l3mu.s"], gs_),
    }

    import flashinfer

    inter = gd("l3.mlp.swiglu").reshape(-1, 640)
    qv["fp4.inter.l3.x"] = inter
    syn640 = syn[:, :640].contiguous()
    for tag, x in [("syn", syn640), ("l3", inter)]:
        q, s = flashinfer.fp4_quantize(
            x, torch.tensor([gs2], dtype=F32, device=DEV), sf_vec_size=16, sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
        qv[f"fp4.inter.{tag}.q"], qv[f"fp4.inter.{tag}.s"] = q, s.view(torch.float8_e4m3fn).reshape(x.shape[0], -1)
    gq = gold["l3.mlp.iq"].reshape(-1, 640).to(torch.int32)
    packed = (gq[:, 0::2] | (gq[:, 1::2] << 4)).to(torch.uint8)
    report["fp4.inter.l3.golden"] = {
        "codes": code_gap(qv["fp4.inter.l3.q"], packed),
        "scales": code_gap(qv["fp4.inter.l3.s"], gold["l3.mlp.is"].reshape(-1, 40)),
    }
    print("quantizers done", flush=True)

    # ---------------- W8A8 (cutlass_scaled_mm, as CutlassFp8BlockScaledMMKernel calls it) ----------
    def scaled_mm(q, s, w, ws):
        n, k = w.shape
        out = torch.empty((q.shape[0], n), dtype=BF16, device=DEV)
        a_s = s.t().contiguous().t()  # [M, K/128], column major
        torch.ops._C.cutlass_scaled_mm(out, q, w.t(), a_s, ws.t(), None)
        return out

    p = G.PREFIX + "layers.3.self_attn.o_proj"
    w = ck.get(p + ".weight").to(DEV)
    ws = ck.get(p + ".weight_scale_inv").to(DEV)
    ev["w8a8.o_proj"] = scaled_mm(qv["fp8.gated.q"], qv["fp8.gated.s"], w, ws)
    report["w8a8.o_proj.golden"] = gap(ev["w8a8.o_proj"], gold["l3.attn.y"])
    del w, ws
    p = G.PREFIX + "layers.3.mlp.shared_expert.down_proj"
    w = ck.get(p + ".weight").to(DEV)
    ws = ck.get(p + ".weight_scale_inv").to(DEV)
    ev["w8a8.shared_down"] = scaled_mm(qv["fp8.eager.l3act.q"], qv["fp8.eager.l3act.s"], w, ws)
    report["w8a8.shared_down.golden"] = gap(ev["w8a8.shared_down"], gold["l3.shared.down"])
    del w, ws
    torch.cuda.empty_cache()
    print("w8a8 done", flush=True)

    # ---------------- MoE: FlashInfer cutlass_fused_moe over layer 3's selected experts ----------
    ev["moe.routed"] = moe(ck, gold, report)
    torch.cuda.empty_cache()
    print("moe done", flush=True)

    # ---------------- GDN prefill and decode (layer 0) ----------------
    gdn(ck, gold, ev, report)
    torch.cuda.empty_cache()
    print("gdn done", flush=True)

    # ---------------- attention (layer 3) ----------------
    attention(ck, gold, ev, report)
    print("attention done", flush=True)

    peak = torch.cuda.max_memory_allocated()
    meta = {
        "vllm": __import__("vllm").__version__,
        "flashinfer": flashinfer.__version__,
        "triton": __import__("triton").__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "golden_sha256": G.sha256(args.golden),
        "kernels": json.dumps({k: v[2] for k, v in G.KERNELS.items()}),
        "gaps": json.dumps(report),
        "device_peak": str(peak),
    }
    for path, tensors in ((args.quant_out, qv), (args.out, ev)):
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().contiguous().cpu() for k, v in tensors.items()}, str(path), metadata=meta)
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(tensors)} tensors), sha256 {G.sha256(path)}")
    print(json.dumps(report, indent=1))
    print(f"device peak {peak / 2**30:.3f} GiB")
    assert peak <= LIMIT, peak


def moe(ck: "G.Checkpoint", gold: dict, report: dict) -> torch.Tensor:
    from flashinfer.fused_moe import cutlass_fused_moe
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

    import vllm._custom_ops as ops

    ids = gold["l3.ids"]
    wts = gold["l3.weights"]
    sel = sorted(set(ids.reshape(-1).tolist()))
    remap = {e: n for n, e in enumerate(sel)}
    local = torch.tensor([[remap[int(e)] for e in row] for row in ids.tolist()], dtype=torch.int32, device=DEV)
    p = G.PREFIX + "layers.3.mlp.experts"
    in13 = max(float(ck.get(f"{p}.{e}.{n}_proj.input_scale")) for e in range(512) for n in ("gate", "up"))
    in2 = max(float(ck.get(f"{p}.{e}.down_proj.input_scale")) for e in range(512))
    w13, s13, w2, s2, a1, a2 = [], [], [], [], [], []
    for e in sel:
        g = ck.get(f"{p}.{e}.gate_proj.weight")
        u = ck.get(f"{p}.{e}.up_proj.weight")
        gs = ck.get(f"{p}.{e}.gate_proj.weight_scale")
        us = ck.get(f"{p}.{e}.up_proj.weight_scale")
        # FlashInfer takes [w3; w1] = [up; gate]
        w13.append(torch.cat([u, g]))
        s13.append(torch.cat([us, gs]))
        w2.append(ck.get(f"{p}.{e}.down_proj.weight"))
        s2.append(ck.get(f"{p}.{e}.down_proj.weight_scale"))
        a1.append(float(ck.get(f"{p}.{e}.gate_proj.weight_scale_2")))
        a2.append(float(ck.get(f"{p}.{e}.down_proj.weight_scale_2")))
    w13 = torch.stack(w13).to(DEV)
    w2 = torch.stack(w2).to(DEV)
    s13 = swizzle_blockscale(torch.stack(s13).to(DEV))
    s2 = swizzle_blockscale(torch.stack(s2).to(DEV))
    in13_t = torch.tensor(in13, dtype=F32, device=DEV)
    in2_t = torch.tensor(in2, dtype=F32, device=DEV)
    g1 = torch.tensor(a1, dtype=F32, device=DEV) * in13_t
    g2 = torch.tensor(a2, dtype=F32, device=DEV) * in2_t
    a1g = 1.0 / in13_t
    a2g = 1.0 / in2_t
    x = gold["l3.mlp.u"].to(DEV)
    xq, xs = ops.scaled_fp4_quant(x, a1g, is_sf_swizzled_layout=True)
    out = torch.empty_like(x)
    cutlass_fused_moe(
        input=xq,
        token_selected_experts=local,
        token_final_scales=wts.to(DEV),
        fc1_expert_weights=w13.view(torch.long),
        fc2_expert_weights=w2.view(torch.long),
        output_dtype=BF16,
        quant_scales=[a1g, s13.view(torch.int32), g1, a2g, s2.view(torch.int32), g2],
        input_sf=xs,
        output=out,
    )
    report["moe.routed.golden"] = gap(out, gold["l3.routed"])
    del w13, w2, s13, s2
    return out


def gdn(ck: "G.Checkpoint", gold: dict, ev: dict, report: dict) -> None:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd

    import vllm._custom_ops as ops

    p = G.PREFIX + "layers.0.linear_attn"
    qkvz = gold["l0.gdn.qkvz"].to(DEV)
    ba = gold["l0.gdn.ba"].to(DEV)
    t = qkvz.shape[0]
    dim = 2 * G.KEY_DIM + G.VAL_DIM
    mixed = qkvz[:, :dim].contiguous()
    z = qkvz[:, dim:].contiguous().reshape(t, G.NV, G.DV)
    b, a = ba[:, : G.NV].contiguous(), ba[:, G.NV :].contiguous()
    w = ck.get(p + ".conv1d.weight").to(DEV).reshape(dim, 4)
    a_log = ck.get(p + ".A_log").to(DEV).float()
    dt = ck.get(p + ".dt_bias").to(DEV)
    nw = ck.get(p + ".norm.weight").to(DEV)

    def prefill(n: int, conv_state: torch.Tensor, ssm: torch.Tensor):
        """causal_conv1d_fn -> fused_post_conv_prep -> FLA chunk -> gated norm, over tokens [0, n)."""
        x = mixed[:n]
        qsl = torch.tensor([0, n], dtype=torch.int32, device=DEV)
        conv = causal_conv1d_fn(
            x.transpose(0, 1),
            w,
            None,
            activation="silu",
            conv_states=conv_state,
            has_initial_state=torch.zeros(1, dtype=torch.bool, device=DEV),
            cache_indices=torch.zeros(1, dtype=torch.int32, device=DEV),
            query_start_loc=qsl,
        ).transpose(0, 1)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=conv.contiguous(), a=a[:n], b=b[:n], A_log=a_log, dt_bias=dt,
            num_k_heads=G.NK, head_k_dim=G.DK, head_v_dim=G.DV, apply_l2norm=True, output_g_exp=False,
        )
        o, state = chunk_gated_delta_rule(
            q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0), g=g.unsqueeze(0), beta=beta.unsqueeze(0),
            initial_state=torch.zeros(1, G.NV, G.DV, G.DK, dtype=F32, device=DEV),
            output_final_state=True, cu_seqlens=qsl.long(), use_qk_l2norm_in_kernel=False,
        )
        core = o.squeeze(0).to(BF16)
        ssm[0] = state[0].to(ssm.dtype)
        normed = torch.empty_like(core)
        layer_norm_fwd(
            core.reshape(-1, G.DV), nw.contiguous(), None, 1e-6, z=z[:n].reshape(-1, G.DV),
            out=normed.reshape(-1, G.DV), group_size=G.DV, norm_before_gate=True, is_rms_norm=True,
            activation="sigmoid",
        )
        return conv, q, k, v, g, beta, core, normed

    conv_state = torch.zeros(1, dim, 3 + 4, dtype=BF16, device=DEV)
    ssm = torch.zeros(1, G.NV, G.DV, G.DK, dtype=F32, device=DEV)
    conv, q, k, v, g, beta, core, normed = prefill(t, conv_state, ssm)
    names = {"conv": conv, "q": q.reshape(t, -1), "k": k.reshape(t, -1), "v": v.reshape(t, -1), "g": g, "beta": beta, "core": core.reshape(t, -1), "normed": normed.reshape(t, -1)}
    for n, val in names.items():
        ev[f"gdn.prefill.{n}"] = val
        report[f"gdn.prefill.{n}.golden"] = gap(val, gold[f"l0.gdn.{n}"])
    # served state is [HV, V, K]; the golden keeps every fourth head as [K, V]
    ev["gdn.prefill.state"] = ssm[0].contiguous()
    gs = gold["l0.gdn.state"]
    served = ssm[0][::4].transpose(-1, -2).cpu()
    report["gdn.prefill.state.golden_rel"] = float((served - gs).abs().max() / gs.abs().max())

    # decode: one step for the last token from the (t-1)-token state, and one 5-token step over
    # the last five from the (t-5)-token state, through the fused MTP decode op
    for n, tag in ((1, "decode1"), (5, "decode5")):
        conv_state = torch.zeros(1, dim, 3 + 4, dtype=BF16, device=DEV)
        ssm = torch.zeros(1, G.NV, G.DV, G.DK, dtype=F32, device=DEV)
        prefill(t - n, conv_state, ssm)
        rows = slice(t - n, t)
        qsl = torch.tensor([0, n], dtype=torch.int32, device=DEV)
        idx = torch.zeros(1, n, dtype=torch.int32, device=DEV)
        acc = torch.ones(1, dtype=torch.int32, device=DEV)
        xs = causal_conv1d_update(
            mixed[rows].contiguous(), conv_state, w, None, "silu", conv_state_indices=idx[:, 0],
            num_accepted_tokens=acc, query_start_loc=qsl, max_query_len=n, validate_data=False,
        )
        out = torch.zeros(n, G.NV, G.DV, dtype=BF16, device=DEV)
        ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=xs, a=a[rows].contiguous(), b=b[rows].contiguous(), A_log=a_log, dt_bias=dt,
            state_indices=idx, cu_seqlens=qsl, num_accepted_tokens=acc, state=ssm,
            output_gate=z[rows].contiguous(), norm_weight=nw, out=out, scale=G.DK**-0.5, norm_eps=1e-6,
            output_gate_activation="sigmoid",
        )
        ev[f"gdn.{tag}.normed"] = out.reshape(n, -1)
        ev[f"gdn.{tag}.state"] = ssm[0].contiguous()
        report[f"gdn.{tag}.normed.golden"] = gap(out.reshape(n, -1), gold["l0.gdn.normed"][rows])


def attention(ck: "G.Checkpoint", gold: dict, ev: dict, report: dict) -> None:
    from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    cfg = json.loads((Path(ck.snapshot) / "config.json").read_text())["text_config"]
    p = G.PREFIX + "layers.3.self_attn"
    qkv = gold["l3.attn.qkv"].to(DEV)
    t = qkv.shape[0]
    from vllm.config import VllmConfig, set_current_vllm_config

    # a CustomOp reads the current config when it is built; the defaults are what matters here
    with torch.device(DEV), set_current_vllm_config(VllmConfig()):
        rope = get_rope(head_size=G.HD, max_position=4096, rope_parameters=cfg["rope_parameters"], dtype=BF16)
    ev["attn.cos_sin"] = rope.cos_sin_cache[:t].to(BF16)
    pos = torch.arange(t, device=DEV)
    positions = pos.unsqueeze(0).expand(3, t).contiguous()
    qg = qkv[:, : G.NQ * 2 * G.HD]
    k = qkv[:, G.NQ * 2 * G.HD : G.NQ * 2 * G.HD + G.NKV * G.HD]
    v = qkv[:, G.NQ * 2 * G.HD + G.NKV * G.HD :]
    q, k, gate = fused_qk_rmsnorm_rope_gate(
        qg.contiguous(), k.contiguous(), ck.get(p + ".q_norm.weight").to(DEV), ck.get(p + ".k_norm.weight").to(DEV),
        rope.cos_sin_cache, positions, 1e-6, G.NQ, G.NKV, G.HD, rope.rotary_dim, norm_beta=1.0,
        mrope_section=getattr(rope, "mrope_section", None),
    )
    for n, val in {"q": q, "k": k, "gate": gate}.items():
        ev[f"attn.normrope.{n}"] = val
        report[f"attn.normrope.{n}.golden"] = gap(val, gold[f"l3.attn.{n}"])

    # dense causal indices over a paged bf16 cache holding the 24 tokens
    width = 2051
    page = 16
    pages = (t + page - 1) // page
    kc = torch.zeros(pages, page, G.NKV, G.HD, dtype=BF16, device=DEV)
    vc = torch.zeros_like(kc)
    kc.reshape(-1, G.NKV, G.HD)[:t] = k.reshape(t, G.NKV, G.HD)
    vc.reshape(-1, G.NKV, G.HD)[:t] = v.reshape(t, G.NKV, G.HD)
    table = torch.arange(pages, dtype=torch.int32, device=DEV).unsqueeze(0)
    idx = torch.full((t, width), -1, dtype=torch.int32, device=DEV)
    for r in range(t):
        idx[r, : r + 1] = torch.arange(r + 1, dtype=torch.int32, device=DEV)
    qh = q.reshape(t, G.NQ, G.HD)
    out = qsa_sparse_paged_attention(qh, kc, vc, idx, table, torch.zeros(t, dtype=torch.int32, device=DEV))
    ev["attn.core24"] = out.reshape(t, -1)
    report["attn.core24.golden"] = gap(out.reshape(t, -1), gold["l3.attn.core"])
    one = qsa_sparse_paged_attention(
        qh[t - 1 :].contiguous(), kc, vc, idx[t - 1 :].contiguous(), table, torch.zeros(1, dtype=torch.int32, device=DEV)
    )
    ev["attn.core1"] = one.reshape(1, -1)
    report["attn.core1.golden"] = gap(one.reshape(1, -1), gold["l3.attn.core"][t - 1 :])


if __name__ == "__main__":
    main()
