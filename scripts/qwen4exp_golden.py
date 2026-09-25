#!/usr/bin/env python3
"""CPU golden for Qwen3.8-Flash-Next (qwen4exp) layers 0-3, as vLLM serves them on a GB10.

Runs 24 real tokens through the embedding, layers 0-3 (three gated delta-nets, then QSA attention),
the final hyper-connection mixer and lm_head rows [0, 8192), and writes every intermediate the
engine's tests compare against to one safetensors fixture.

The reference is vLLM as served, not its eager module code. The live server compiles its glue with
inductor (custom_ops none), and inside a fused kernel inductor drops the eager bf16 casts. So a
value is rounded to bf16 here only where a served kernel stores it: a tl.store or an extern op's
output in the live run's compiled graph, or a store in a custom kernel's source. Between store
points arithmetic is f32. The served approximate instructions (div.full.f32, rcp.approx.ftz.f32,
libdevice rsqrt) are IEEE here; the GPU vectors (qwen4exp_vectors.py) measure that gap.

Every compiled or custom kernel this script transcribes is listed in KERNELS with its sha256, and
the script refuses to run when one differs: a regenerated compile cache changes the rounding model.

Run on CPU only, in a capped scope:
  systemd-run --user --scope --quiet -p MemoryMax=3G -p MemorySwapMax=0 choom -n 1000 -- \
    env VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES= nice -n19 ionice -c3 \
    ~/vllm-env/bin/python scripts/qwen4exp_golden.py --snapshot <fp8hybrid> \
      --out hanzo-engine/tests/fixtures/qwen4exp.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path.home()
SITE = Path(torch.__file__).resolve().parent.parent
INDUCTOR = (
    HOME
    / ".cache/vllm/torch_compile_cache/torch_aot_compile"
    / "b5ce9057ef3e69282ab304d6c174cf070fc41e1671ab7f4ca5aabfe3c742a8aa/inductor_cache"
)

# stage -> (file, lines, sha256). Inductor artifacts belong to compile cache 92affc475f.
KERNELS = {
    "fp8.linear (fused W8A8 input quantizer, amax*RN(1/448), div.full)": (
        INDUCTOR / "d7/cd76kp4tkxhvavtr4sv37bs4zyaxtb3ozm22umsosh3ggoncajyl.py",
        "59-110",
        "f529f4e6edc28635ce7f97767cdf5e197a5cd32de3dcf37b4f7313a3dec5bbe1",
    ),
    "fp8.gated (o_proj quantizer over attn*sigmoid(gate) in f32)": (
        INDUCTOR / "7w/c7wgpo357q2hrlrcgvyaumoyoqeqjzziqapm4jzorvvnkqhetdic.py",
        "35-62",
        "b417413a75a5bb067a5a5a4573913e5ae4f6541dfde081ce782e398e9271664f",
    ),
    "fp8.eager (standalone QuantFP8 inside moe_forward_shared, amax div.full 448)": (
        INDUCTOR / "7l/c7lvd6lba4iozfx55hisw2larwxcmcpvff34nriwmo3gvhsvtg2j.py",
        "97-118",
        "914f9e6597889a67745b7e00fe6f057eab4c4957f0b3a8401cf0aadffcc5b2ab",
    ),
    "shared.act (SiluAndMul: g/(1+exp(-g))*u in f32, one store)": (
        INDUCTOR / "ca/ccaihrq327fpbfzxgik6lpdltwgwcgrg4il4pyy6u2u2352oqmmb.py",
        "45-56",
        "7274bb492bc3302ee2852f7c09ee7281115bcd15870af82aa2d565e461350810",
    ),
    "moe.add (routed + shared, f32 add, one store)": (
        INDUCTOR / "jv/cjvnrjize3dsih2pizz7jltdu5a23ptopgpo2raxrvzuvn2yygmo.py",
        "1-26",
        "23d9f6a8abbf48e1e04d2917508af711441d1ad4a761f11bcc2f556f1e4a9e0f",
    ),
    "ple.gate (norms, dot, gate in f32; gv stored; ssq from unrounded gv)": (
        INDUCTOR / "u7/cu7z4zyhdcnzrxnzdllkyk2a4elu2vutujq5ydckri66ni6n5ziw.py",
        "46-359",
        "c6dc25bc85b0e56e663a476587a152aa6cd06c6188c4cbc3613ab47cb58fa02b",
    ),
    "ple.add (X + (gv + conv) in f32, one store)": (
        INDUCTOR / "l4/cl44yah5z5izcetowiay42gryoh4maea2jd6ujac66tyjqwdn4ly.py",
        "40-59",
        "fbb34228f9c856c166e09f0dfaa12fb5d7d0a077ab4c68725c5847219972b52d",
    ),
    "hc (grouped Gemma norm, silu, gate mix, combine[+norm])": (
        SITE / "vllm/models/qwen4_exp/nvidia/ops/hc.py",
        "11-360",
        "1de6e4780d25aecbaf3e36394cef4ac8d58375c708cc41891be36577341393fc",
    ),
    "attn.core (QSA sparse paged GQA: exp2, P bf16, f32 normalizer)": (
        SITE / "vllm/models/qwen4_exp/nvidia/ops/qsa.py",
        "193-359",
        "1038cd6744b0b27ef73bd3f6915d89332c017e298fd49b520e12723bfc2f0d74",
    ),
    "attn.normrope (fused q/k RMSNorm + NeoX partial RoPE + gate copy)": (
        SITE / "vllm/model_executor/layers/fused_qk_norm_rope.py",
        "16-131",
        "182a13b20ac5857674a368d030268161ae0d4f358a38114cded804c1791abcaa",
    ),
    "gdn.post_conv (l2norm q/k, v copy, g, beta)": (
        SITE / "vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py",
        "100-157",
        "590717b2107abc90d72548de06b5d315229eb2775c94614f3f545df11cc2b5df",
    ),
    "gdn.norm (gated RMSNorm, norm before sigmoid gate)": (
        SITE / "vllm/third_party/flash_linear_attention/ops/layernorm_guard.py",
        "66-168",
        "0ca5e46aab408b80d673561b91cf7bff7987c71d798d3bdacac75b7e29416f49",
    ),
    "gdn.conv (causal conv: bf16 products into an f32 acc, silu)": (
        SITE / "vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
        "398-478",
        "044d005cfe59fd0818ed421274e04f3e8dd679b8cdb64fca9f4422f2643484b2",
    ),
    "gdn.forward (fused-norm packed path, prefill through FLA)": (
        SITE / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "889-965,1259-1582,1784-1911",
        "cb819264c1e791177d866f484ca0579a7d57141926705a1f75dd4a79de4d0125",
    ),
    "ple.conv (bf16 conv1d then bf16 silu, eager)": (
        SITE / "vllm/models/qwen4_exp/nvidia/ple_layer.py",
        "730-830,1155-1188",
        "601484d26dcf24637e142f770da46ca848c8e5f4b311d8801fc74d7ce843b14e",
    ),
    "fp4.inter (FlashInfer fast-math cvt_warp_fp16_to_fp4)": (
        SITE / "flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh",
        "430-510",
        "c2a860da4407f70c281c84981db796ec111520d141a3753d77fe66c6c11f3928",
    ),
    "moe.swiglu (fc1 out bf16, SwiGLU rounded to bf16 before quantizing)": (
        SITE / "flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh",
        "1075-1090,2515-2533",
        "fd9e2e976496ab318bda6d133d2b68f45b3451a6482978f452acd7a86e029841",
    ),
    "model (decoder order: delayed combine, PLE before attn mix)": (
        SITE / "vllm/models/qwen4_exp/nvidia/model.py",
        "274-327,474-540",
        "e9ee63d7921d1fba937a3674320225a1f914e668d5a2f91dbde11dfc70e2214e",
    ),
    "attn.module (qkv, norm-rope, QSA, gate, o_proj)": (
        SITE / "vllm/models/qwen4_exp/nvidia/qsa.py",
        "440-470",
        "3d5eab88a90abf3dece855c7b13e40b67db793a8857b90c49d2c3fd20420ba92",
    ),
}

PREFIX = "model.language_model."
T_TEXT = (
    "The quick brown fox jumps over the lazy dog. Numbers like 3.14159 and 2718 sit beside "
    "words such as hyper-connection, delta-net and attention in this short test passage."
)
EOS = 248044
EOS_POS = 9
NTOK = 24
HEAD_ROWS = 8192

F32 = torch.float32
BF16 = torch.bfloat16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_kernels() -> dict:
    table = {}
    bad = []
    for stage, (path, lines, want) in KERNELS.items():
        got = sha256(path) if path.exists() else "missing"
        if got != want:
            bad.append(f"{stage}: {path} is {got}, expected {want}")
        table[stage] = {"path": str(path), "lines": lines, "sha256": want}
    if bad:
        raise SystemExit("served kernels changed; re-derive the rounding model:\n" + "\n".join(bad))
    return table


# --------------------------------------------------------------------------------------------
# rounding helpers: tensors are f32 throughout; bf() is a bf16 store
# --------------------------------------------------------------------------------------------


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(BF16).to(F32)


def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """a*b + c rounded once to f32, as a fused multiply-add (Triton contracts these)."""
    return (a.double() * b.double() + c.double()).to(F32)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


# --------------------------------------------------------------------------------------------
# checkpoint access, one tensor at a time
# --------------------------------------------------------------------------------------------


ST_DTYPES = {
    "BF16": (np.int16, torch.bfloat16),
    "F32": (np.float32, torch.float32),
    "F8_E4M3": (np.uint8, torch.float8_e4m3fn),
    "U8": (np.uint8, torch.uint8),
    "I64": (np.int64, torch.int64),
}


class Checkpoint:
    """Reads tensors by header offset with pread, one at a time: no mmap, so pages read once
    do not stay in this process's RSS."""

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.headers = {}
        self.index = {}
        for f in sorted(snapshot.glob("*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                self.index[name] = f
                self.headers[name] = (meta, 8 + n)
        self._fds = {}

    def _read(self, name: str, lo_byte: int, nbytes: int) -> bytes:
        path = self.index[name]
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        meta, base = self.headers[name]
        start = base + meta["data_offsets"][0] + lo_byte
        buf = os.pread(fd, nbytes, start)
        assert len(buf) == nbytes, (name, len(buf), nbytes)
        return buf

    def _tensor(self, name: str, buf: bytes, shape: list[int]) -> torch.Tensor:
        np_dtype, t_dtype = ST_DTYPES[self.headers[name][0]["dtype"]]
        arr = np.frombuffer(buf, dtype=np_dtype).copy()
        return torch.from_numpy(arr).view(t_dtype).reshape(shape)

    def get(self, name: str) -> torch.Tensor:
        meta, _ = self.headers[name]
        lo, hi = meta["data_offsets"]
        return self._tensor(name, self._read(name, 0, hi - lo), meta["shape"])

    def rows(self, name: str, lo: int, hi: int) -> torch.Tensor:
        meta, _ = self.headers[name]
        shape = meta["shape"]
        row = (meta["data_offsets"][1] - meta["data_offsets"][0]) // shape[0]
        return self._tensor(name, self._read(name, lo * row, (hi - lo) * row), [hi - lo, *shape[1:]])

    def take(self, name: str, ids: list[int]) -> torch.Tensor:
        return torch.cat([self.rows(name, i, i + 1) for i in ids])

    def lm(self, name: str) -> torch.Tensor:
        return self.get(PREFIX + name)

    def f(self, name: str) -> torch.Tensor:
        return self.lm(name).to(F32)


# --------------------------------------------------------------------------------------------
# quantizers
# --------------------------------------------------------------------------------------------

FP8_RCP448 = torch.tensor(0.002232142857142857, dtype=F32)
FP8_FLOOR = torch.tensor(4.359654017857143e-06, dtype=F32)


def fp8_quant(x: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token, per-128 group e4m3 quantization of f32 `x` [T, K] (bf16 or f32 values).

    linear: the fused inductor kernel, s = max(amax * RN(1/448), 1/229376), q = x / s.
    eager:  the standalone QuantFP8 kernel, s = max(amax / 448, 1/229376), q = x / s.
    The served divisions are div.full.f32; IEEE here.
    """
    t, k = x.shape
    g = x.reshape(t, k // 128, 128)
    amax = g.abs().amax(-1, keepdim=True)
    if mode == "linear":
        s = torch.maximum(amax * FP8_RCP448, FP8_FLOOR)
    elif mode == "eager":
        s = torch.maximum(amax / torch.tensor(448.0, dtype=F32), FP8_FLOOR)
    else:
        raise ValueError(mode)
    q = (g / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(t, k), s.squeeze(-1)


def w8a8(q: torch.Tensor, s: torch.Tensor, w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """cutlass_scaled_mm with 128x128 weight blocks: per K block an f32 dot of the codes, scaled
    by s_a * s_w and accumulated in f32; one bf16 store."""
    t, k = q.shape
    n = w.shape[0]
    qf = q.to(F32)
    acc = torch.zeros(t, n, dtype=F32)
    wsx = ws.to(F32).repeat_interleave(128, dim=0)[:n]  # [N, K/128]
    for kb in range(k // 128):
        part = qf[:, kb * 128 : (kb + 1) * 128] @ w[:, kb * 128 : (kb + 1) * 128].to(F32).T
        acc += part * s[:, kb : kb + 1] * wsx[:, kb].unsqueeze(0)
    return bf(acc)


def gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A bf16 GEMM: f32 matmul of the bf16 values, rounded once."""
    return bf(x @ w.to(F32).T)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=F32)


def e2m1_encode(v: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even, satfinite, sign kept on zero (cvt.rn.satfinite.e2m1x2.f32)."""
    a = v.abs().clamp(max=6.0)
    # candidates ordered; ties go to the even code (low mantissa bit 0)
    diff = (a.unsqueeze(-1) - E2M1).abs()
    best = diff.min(-1, keepdim=True).values
    tie = diff == best
    codes = torch.arange(8)
    # among tied candidates prefer the even code
    score = torch.where(tie, (codes % 2) * 1 + 0, torch.full_like(codes, 9))
    code = score.argmin(-1)
    sign = (torch.signbit(v)).to(torch.int64) << 3
    return (code | sign).to(torch.uint8)


def e2m1_decode(c: torch.Tensor) -> torch.Tensor:
    mag = E2M1[(c & 7).long()]
    return torch.where((c & 8) != 0, -mag, mag)


def nvfp4_quant(x: torch.Tensor, gs: float) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 activation quantization with global scale `gs` (vLLM scaled_fp4_quant and FlashInfer
    fast-math cvt_warp_fp16_to_fp4 share the formula):
      SF = gs * (vecMax * rcp(6)); sf8 = e4m3(SF); out = rcp(f32(sf8) * rcp(gs)) or 0 when the
      block is all zero; code = e2m1(x * out).
    rcp.approx.ftz is IEEE here. Returns codes [T, K] (uint8 0..15) and scales e4m3 [T, K/16]."""
    t, k = x.shape
    g = x.reshape(t, k // 16, 16)
    gs32 = torch.tensor(gs, dtype=F32)
    one = torch.tensor(1.0, dtype=F32)
    vmax = g.abs().amax(-1, keepdim=True)
    sf = gs32 * (vmax * (one / torch.tensor(6.0, dtype=F32)))
    sf8 = sf.to(torch.float8_e4m3fn)
    out = torch.where(vmax != 0, one / (sf8.to(F32) * (one / gs32)), torch.zeros_like(vmax))
    codes = e2m1_encode(g * out)
    return codes.reshape(t, k), sf8.squeeze(-1)


def nvfp4_dequant_codes(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """code * e4m3 scale, exact in f32."""
    t, k = codes.shape
    v = e2m1_decode(codes).reshape(t, k // 16, 16) * scales.to(F32).unsqueeze(-1)
    return v.reshape(t, k)


def unpack_fp4(w: torch.Tensor) -> torch.Tensor:
    """U8 [N, K/2] -> codes [N, K], low nibble first."""
    lo = w & 0x0F
    hi = w >> 4
    return torch.stack([lo, hi], dim=-1).reshape(w.shape[0], -1)


# --------------------------------------------------------------------------------------------
# hyper-connections (nvidia/ops/hc.py)
# --------------------------------------------------------------------------------------------

HC = 4
H = 2560
EPS = 1e-6


def hc_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """_grouped_gemma_rmsnorm_kernel: y = x*rrms; y += y*w (an FMA); one store."""
    t = x.shape[0]
    g = x.reshape(t, HC, H)
    rrms = torch.rsqrt((g * g).sum(-1, keepdim=True) / H + EPS)
    y = g * rrms
    y = fma(y, w.to(F32).reshape(1, HC, H).expand_as(y), y)
    return bf(y.reshape(t, HC * H))


def hc_silu(x: torch.Tensor) -> torch.Tensor:
    v = x / HC
    return bf(v * sigmoid(v))


def hc_gate_mix(xn: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    t = xn.shape[0]
    acc = torch.zeros(t, H, dtype=F32)
    for s in range(HC):
        acc = fma(sigmoid(gate[:, s * H : (s + 1) * H]), xn[:, s * H : (s + 1) * H], acc)
    return bf(acc / HC)


def hc_combine(res: torch.Tensor, block: torch.Tensor, inj: torch.Tensor) -> torch.Tensor:
    """out = res + block * 2σ(inj/HC), an FMA, one store (both _hc_combine and _hc_combine_norm)."""
    t = res.shape[0]
    w = 2.0 * sigmoid(inj / HC)  # [T, HC]
    r = res.reshape(t, HC, H)
    return bf(fma(block.unsqueeze(1).expand_as(r), w.unsqueeze(-1).expand_as(r), r)).reshape(t, HC * H)


class Branch:
    """GatedResidual: hc_norm, merged [down; inject; pad] GEMM, hc_silu, up GEMM, gate mix."""

    def __init__(self, ck: Checkpoint, prefix: str, inject: bool = True):
        self.norm = ck.lm(f"{prefix}.hc_norm.weight")
        self.down = ck.lm(f"{prefix}.input_mix_weight_down.weight")
        self.up = ck.lm(f"{prefix}.input_mix_weight_up.weight")
        self.inject = ck.lm(f"{prefix}.block_inject_weight.weight") if inject else None

    def mix(self, x: torch.Tensor) -> dict:
        xn = hc_norm(x, self.norm)
        r = gemm(xn, self.down)
        inj = gemm(xn, self.inject) if self.inject is not None else None
        lora = hc_silu(r)
        gate = gemm(lora, self.up)
        u = hc_gate_mix(xn, gate)
        return {"xn": xn, "u": u, "inj": inj}


# --------------------------------------------------------------------------------------------
# gated delta-net (qwen_gdn_linear_attn.py fused-norm packed path, prefill)
# --------------------------------------------------------------------------------------------

NK, NV, DK, DV = 16, 48, 128, 128
KEY_DIM, VAL_DIM = NK * DK, NV * DV


def gdn(ck: Checkpoint, i: int, u: torch.Tensor, out: dict, store: bool) -> torch.Tensor:
    p = f"layers.{i}.linear_attn"
    t = u.shape[0]
    wq = torch.cat([ck.lm(f"{p}.in_proj_qkv.weight"), ck.lm(f"{p}.in_proj_z.weight")])
    wqs = torch.cat([ck.lm(f"{p}.in_proj_qkv.weight_scale_inv"), ck.lm(f"{p}.in_proj_z.weight_scale_inv")])
    q8, s8 = fp8_quant(u, "linear")
    qkvz = w8a8(q8, s8, wq, wqs)
    del wq
    ba = gemm(u, torch.cat([ck.lm(f"{p}.in_proj_b.weight"), ck.lm(f"{p}.in_proj_a.weight")]))
    b, a = ba[:, :NV], ba[:, NV:]
    mixed = qkvz[:, : 2 * KEY_DIM + VAL_DIM]
    z = qkvz[:, 2 * KEY_DIM + VAL_DIM :]

    # causal conv + silu: bf16 products into an f32 accumulator, silu, one store
    w = ck.lm(f"{p}.conv1d.weight").reshape(-1, 4)
    hist = torch.cat([torch.zeros(3, mixed.shape[1]), mixed])
    acc = torch.zeros_like(mixed)
    for j in range(4):
        acc = acc + bf(hist[j : j + t] * w[:, j].to(F32).unsqueeze(0))
    conv = bf(acc / (1.0 + torch.exp(-acc)))

    # fused post-conv prep: l2norm q/k in f32, one store; v copied; g, beta f32
    q = conv[:, :KEY_DIM].reshape(t, NK, DK)
    k = conv[:, KEY_DIM : 2 * KEY_DIM].reshape(t, NK, DK)
    v = conv[:, 2 * KEY_DIM :].reshape(t, NV, DV)
    q = bf(q * (1.0 / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)))
    k = bf(k * (1.0 / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)))
    a_log = ck.f(f"{p}.A_log")
    dt = ck.f(f"{p}.dt_bias")
    x = a + dt
    sp = torch.where(x > 0, x + torch.log(1.0 + torch.exp(-x)), torch.log(1.0 + torch.exp(x)))
    sp = torch.where(x <= 20.0, sp, x)
    g = -torch.exp(a_log) * sp
    beta = sigmoid(b)

    # core: FLA chunked gated delta rule (f32 here), grouped GQA: v head h reads k head h // 3
    from transformers.models.qwen4_exp.modeling_qwen4_exp import torch_chunk_gated_delta_rule

    rep = NV // NK
    core, state = torch_chunk_gated_delta_rule(
        q.repeat_interleave(rep, dim=1).unsqueeze(0),
        k.repeat_interleave(rep, dim=1).unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    core = bf(core.squeeze(0))  # [T, NV, DV]

    # gated RMSNorm (norm before gate, sigmoid), one store
    nw = ck.f(f"{p}.norm.weight")
    rstd = torch.rsqrt((core * core).sum(-1, keepdim=True) / DV + EPS)
    y = core * rstd * nw
    normed = bf(y * sigmoid(z.reshape(t, NV, DV))).reshape(t, VAL_DIM)

    oq, os_ = fp8_quant(normed, "linear")
    y = w8a8(oq, os_, ck.lm(f"{p}.out_proj.weight"), ck.lm(f"{p}.out_proj.weight_scale_inv"))
    if store:
        out.update(
            {
                f"l{i}.gdn.qkvz": qkvz,
                f"l{i}.gdn.ba": ba,
                f"l{i}.gdn.conv": conv,
                f"l{i}.gdn.q": q.reshape(t, -1),
                f"l{i}.gdn.k": k.reshape(t, -1),
                f"l{i}.gdn.v": v.reshape(t, -1),
                f"l{i}.gdn.g": g,
                f"l{i}.gdn.beta": beta,
                f"l{i}.gdn.core": core.reshape(t, -1),
                f"l{i}.gdn.normed": normed,
                # every fourth v head (all three group residues), f32, grouped order
                f"l{i}.gdn.state": state.squeeze(0)[::4].contiguous(),
            }
        )
    return y


# --------------------------------------------------------------------------------------------
# QSA attention (dense through 2,051 tokens)
# --------------------------------------------------------------------------------------------

NQ, NKV, HD, ROT = 24, 2, 256, 64
THETA = 10_000_000.0
SCALE_LOG2 = torch.tensor((HD**-0.5) * 1.4426950408889634, dtype=F32)


def rope_table(n: int) -> torch.Tensor:
    """vLLM RotaryEmbedding cache: [cos | sin] over rotary_dim/2 frequencies, f32 then bf16."""
    inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, dtype=torch.float) / ROT))
    t = torch.arange(n, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(BF16).to(F32)


def norm_rope(x: torch.Tensor, w: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
    """_fused_qk_rmsnorm_rope_gate_kernel for one head set: x [T, heads, 256]."""
    var = (x * x).sum(-1, keepdim=True) / HD
    inv = torch.rsqrt(var + EPS)
    wf = w.to(F32) + 1.0
    xn = bf(x * inv * wf)
    half = ROT // 2
    cos = cs[:, None, :half]
    sin = cs[:, None, half:]
    x1, x2 = xn[..., :half], xn[..., half:ROT]
    o1 = bf(x1 * cos - x2 * sin)
    o2 = bf(x2 * cos + x1 * sin)
    return torch.cat([o1, o2, xn[..., ROT:]], dim=-1)


def attention(ck: Checkpoint, i: int, u: torch.Tensor, out: dict) -> torch.Tensor:
    p = f"layers.{i}.self_attn"
    t = u.shape[0]
    w = torch.cat([ck.lm(f"{p}.{n}_proj.weight") for n in "qkv"])
    ws = torch.cat([ck.lm(f"{p}.{n}_proj.weight_scale_inv") for n in "qkv"])
    q8, s8 = fp8_quant(u, "linear")
    qkv = w8a8(q8, s8, w, ws)
    del w
    qg = qkv[:, : NQ * 2 * HD].reshape(t, NQ, 2 * HD)
    k = qkv[:, NQ * 2 * HD : NQ * 2 * HD + NKV * HD].reshape(t, NKV, HD)
    v = qkv[:, NQ * 2 * HD + NKV * HD :].reshape(t, NKV, HD)
    cs = rope_table(t)
    q = norm_rope(qg[..., :HD], ck.lm(f"{p}.q_norm.weight"), cs)
    k = norm_rope(k, ck.lm(f"{p}.k_norm.weight"), cs)
    gate = qg[..., HD:]

    # QSA kernel math, one tile: scores f32 * 256^-0.5*log2e, exp2 against the row max, P rounded
    # to bf16 for PV, normalizer from unrounded P, one division, one store.
    group = NQ // NKV
    core = torch.zeros(t, NQ, HD, dtype=F32)
    for h in range(NQ):
        kv = h // group
        s = (q[:, h] @ k[:, kv].T) * SCALE_LOG2  # [T, T]
        mask = torch.ones(t, t, dtype=torch.bool).tril()
        s = torch.where(mask, s, torch.tensor(-1.0e20))
        m = s.amax(-1, keepdim=True)
        pr = torch.where(mask, torch.exp2(s - m), torch.zeros(()))
        lsum = pr.sum(-1, keepdim=True)
        acc = bf(pr) @ v[:, kv]
        core[:, h] = bf(acc / torch.clamp(lsum, min=1.0e-20))

    prod = core.reshape(t, -1) * sigmoid(gate.reshape(t, -1))  # f32, never rounded
    oq, os_ = fp8_quant(prod, "linear")
    y = w8a8(oq, os_, ck.lm(f"{p}.o_proj.weight"), ck.lm(f"{p}.o_proj.weight_scale_inv"))
    out.update(
        {
            f"l{i}.attn.qkv": qkv,
            f"l{i}.attn.q": q.reshape(t, -1),
            f"l{i}.attn.k": k.reshape(t, -1),
            f"l{i}.attn.v": v.reshape(t, -1),
            f"l{i}.attn.gate": gate.reshape(t, -1).contiguous(),
            f"l{i}.attn.core": core.reshape(t, -1),
            f"l{i}.attn.oq": oq,
            f"l{i}.attn.os": os_,
        }
    )
    return y


# --------------------------------------------------------------------------------------------
# MoE (router, NVFP4 routed experts through FlashInfer CUTLASS, block-FP8 shared expert)
# --------------------------------------------------------------------------------------------

E, TOPK, FF = 512, 10, 640


def route(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """topk_softmax: softmax in f32 over bf16 logits, top-10 by (p desc, index asc), renorm."""
    prob = torch.softmax(logits, dim=-1)
    ids, wts = [], []
    for row in prob:
        order = np.lexsort((np.arange(E), -row.numpy()))[:TOPK]
        sel = row[torch.from_numpy(order)]
        ids.append(torch.from_numpy(order.astype(np.int64)))
        wts.append(sel / sel.sum())
    return torch.stack(ids), torch.stack(wts)


def moe(ck: Checkpoint, i: int, u: torch.Tensor, out: dict, store: bool) -> torch.Tensor:
    p = f"layers.{i}.mlp"
    t = u.shape[0]
    logits = gemm(u, ck.lm(f"{p}.gate.weight"))
    ids, wts = route(logits)

    # global activation scales: max over experts (gate and up share fc1's)
    in13 = max(
        float(ck.lm(f"{p}.experts.{e}.{n}_proj.input_scale")) for e in range(E) for n in ("gate", "up")
    )
    in2 = max(float(ck.lm(f"{p}.experts.{e}.down_proj.input_scale")) for e in range(E))
    gs1 = float(torch.tensor(1.0, dtype=F32) / torch.tensor(in13, dtype=F32))
    gs2 = float(torch.tensor(1.0, dtype=F32) / torch.tensor(in2, dtype=F32))

    uq, us = nvfp4_quant(u, gs1)
    xd = nvfp4_dequant_codes(uq, us)  # exact code * scale

    # one expert at a time: every (token, slot) routed to it, then its weights are dropped
    a1 = torch.tensor(in13, dtype=F32)
    a2 = torch.tensor(in2, dtype=F32)
    fc1 = torch.zeros(t, TOPK, 2 * FF)
    act_all = torch.zeros(t, TOPK, FF)
    iq_all = torch.zeros(t, TOPK, FF, dtype=torch.uint8)
    is_all = torch.zeros(t, TOPK, FF // 16, dtype=torch.float8_e4m3fn)
    fc2 = torch.zeros(t, TOPK, H)
    for e in sorted(set(ids.reshape(-1).tolist())):
        def load(n):
            pre = f"{p}.experts.{e}.{n}_proj"
            codes = unpack_fp4(ck.lm(f"{pre}.weight"))
            wd = nvfp4_dequant_codes(codes, ck.lm(f"{pre}.weight_scale"))
            return wd, float(ck.lm(f"{pre}.weight_scale_2"))

        (gw, g2), (uw, u2), (dw, d2) = load("gate"), load("up"), load("down")
        if g2 != u2:
            raise SystemExit(f"layer {i} expert {e}: gate weight_scale_2 {g2} != up {u2}")
        alpha1 = a1 * torch.tensor(g2, dtype=F32)
        alpha2 = a2 * torch.tensor(d2, dtype=F32)
        rows, slots = (ids == e).nonzero(as_tuple=True)
        xe = xd[rows]
        gate = bf((xe @ gw.T) * alpha1)
        up = bf((xe @ uw.T) * alpha1)
        act = bf(gate / (1.0 + torch.exp(-gate)) * up)
        aq, as_ = nvfp4_quant(act, gs2)
        y = bf((nvfp4_dequant_codes(aq, as_) @ dw.T) * alpha2)
        fc1[rows, slots] = torch.cat([gate, up], dim=-1)
        act_all[rows, slots] = act
        iq_all[rows, slots] = aq
        is_all[rows, slots] = as_
        fc2[rows, slots] = y
        del gw, uw, dw
    routed = torch.zeros(t, H, dtype=F32)
    for r in range(t):
        acc = torch.zeros(H, dtype=F32)
        for j in range(TOPK):
            acc = acc + wts[r, j] * fc2[r, j]
        routed[r] = bf(acc)

    # shared expert, inside moe_forward_shared: standalone QuantFP8 (amax / 448) and SiluAndMul
    sp = f"{p}.shared_expert"
    gq, gs_ = fp8_quant(u, "eager")
    gu = w8a8(
        gq,
        gs_,
        torch.cat([ck.lm(f"{sp}.gate_proj.weight"), ck.lm(f"{sp}.up_proj.weight")]),
        torch.cat([ck.lm(f"{sp}.gate_proj.weight_scale_inv"), ck.lm(f"{sp}.up_proj.weight_scale_inv")]),
    )
    g, up = gu[:, :FF], gu[:, FF:]
    act = bf(g / (1.0 + torch.exp(-g)) * up)
    aq, as_ = fp8_quant(act, "eager")
    down = w8a8(aq, as_, ck.lm(f"{sp}.down_proj.weight"), ck.lm(f"{sp}.down_proj.weight_scale_inv"))
    sg = gemm(u, ck.lm(f"{p}.shared_expert_gate.weight"))
    shared = bf(bf(sigmoid(sg)) * down)
    y = bf(routed + shared)

    out.update({f"l{i}.router": logits, f"l{i}.ids": ids, f"l{i}.weights": wts, f"l{i}.routed": routed, f"l{i}.shared": shared, f"l{i}.moe": y})
    if store:
        out.update(
            {
                f"l{i}.mlp.uq": uq,
                f"l{i}.mlp.us": us,
                f"l{i}.mlp.swiglu": act_all,
                f"l{i}.mlp.iq": iq_all,
                f"l{i}.mlp.is": is_all,
                f"l{i}.mlp.gs": torch.tensor([gs1, gs2], dtype=F32),
                f"l{i}.shared.gu": gu,
                f"l{i}.shared.act": act,
                f"l{i}.shared.q": aq,
                f"l{i}.shared.s": as_,
                f"l{i}.shared.down": down,
                f"l{i}.shared.sg": sg,
            }
        )
    return y


# --------------------------------------------------------------------------------------------
# PLE (layer 1): n-gram ids, FP8 table rows from disk, compiled gate kernel, eager conv
# --------------------------------------------------------------------------------------------


def ngram_module(ck: Checkpoint, cfg: dict):
    """vLLM's Qwen4ExpNGramEmbedding.compute_ngram_ids bound to the checkpoint's I64 buffers."""
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding as N

    pre = "layers.1.ple.ple_embedding"
    mod = SimpleNamespace(
        ngram_size=cfg["ngram_size"],
        heads_per_ngram=cfg["heads_per_ngram"],
        eos_token_id=cfg["eos_token_id"],
        layer_multipliers=ck.lm(f"{pre}.layer_multipliers"),
        ngram_heads_vocab_sizes=ck.lm(f"{pre}.ngram_heads_vocab_sizes"),
        ngram_heads_offsets=ck.lm(f"{pre}.ngram_heads_offsets"),
        positions_buffer=torch.arange(4096, dtype=torch.int64),
        padded_buffer=torch.full((8, 4096), cfg["eos_token_id"], dtype=torch.int64),
        _shift_precompute=N._shift_precompute,
        _shift_apply=N._shift_apply,
    )

    def ids(tokens: list[int]) -> torch.Tensor:
        return N.compute_ngram_ids(
            mod,
            torch.tensor(tokens, dtype=torch.int64),
            torch.tensor([0, len(tokens)], dtype=torch.int32),
            torch.full((1, cfg["ngram_size"] - 1), cfg["eos_token_id"], dtype=torch.int64),
        )

    return ids


def ple_rows(ck: Checkpoint, ids: torch.Tensor) -> torch.Tensor:
    """Raw fp8 bytes of each id's row, read at the header offsets: row r is shard
    r // rows_per_shard, byte (r % rows_per_shard) * width of that shard's data."""
    pre = PREFIX + "layers.1.ple.ple_embedding.ngram_embedding"
    rows_per, width = ck.headers[f"{pre}.shard_0.weight"][0]["shape"]
    out = np.empty((ids.numel(), width), dtype=np.uint8)
    for n, r in enumerate(ids.reshape(-1).tolist()):
        shard, off = divmod(int(r), rows_per)
        out[n] = np.frombuffer(ck._read(f"{pre}.shard_{shard}.weight", off * width, width), np.uint8)
    return torch.from_numpy(out).reshape(*ids.shape, width)


def ple(ck: Checkpoint, x: torch.Tensor, ids: torch.Tensor, out: dict) -> torch.Tensor:
    p = "layers.1.ple"
    t = x.shape[0]
    rows = ple_rows(ck, ids)  # [T, 16, 160] u8
    scale = bf(ck.f(f"{p}.ple_embedding.ngram_embedding.weight_scale").reshape(()))
    e = bf(rows.view(torch.float8_e4m3fn).to(F32) * scale).reshape(t, -1)
    key = gemm(e, ck.lm(f"{p}.key_proj.weight"))
    value = gemm(e, ck.lm(f"{p}.value_proj.weight"))

    wk = ck.f(f"{p}.norm_key.weight").reshape(HC, H)
    wq = ck.f(f"{p}.norm_query.weight").reshape(HC, H)
    wc = ck.f(f"{p}.norm_conv.weight").reshape(HC, H)
    kk = key.reshape(t, HC, H)
    xq = x.reshape(t, HC, H)
    kn = kk * torch.rsqrt((kk * kk).sum(-1, keepdim=True) / H + EPS) * (wk + 1.0)
    qn = xq * torch.rsqrt((xq * xq).sum(-1, keepdim=True) / H + EPS) * (wq + 1.0)
    dot = (kn * qn).sum(-1, keepdim=True)
    gl = dot * torch.tensor(0.01976423537605237, dtype=F32)
    gate = sigmoid(torch.sign(gl) * torch.sqrt(torch.clamp(gl.abs(), min=1e-6)))
    gvf = gate * value.unsqueeze(1)  # f32
    gv = bf(gvf)
    ssq = (gvf * gvf).sum(-1, keepdim=True)
    nrm = bf(gv * torch.rsqrt(ssq / H + EPS) * (wc + 1.0)).reshape(t, HC * H)

    # eager bf16 conv1d (kernel 4, dilation = ngram_size 3: 9 zeros of history), then bf16 silu
    w = ck.lm(f"{p}.conv1d.weight")  # [10240, 1, 4] bf16
    hist = torch.cat([torch.zeros(HC * H, 9, dtype=BF16), nrm.to(BF16).T], dim=-1).unsqueeze(0)
    c = F.conv1d(hist.to(F32), w.to(F32), groups=HC * H, dilation=3)
    conv = bf(F.silu(bf(c)).squeeze(0).T)
    delta = gv.reshape(t, -1) + conv  # f32, fused with the stream add: never stored
    xn = bf(x + delta)
    out.update(
        {
            "l1.ple.ids": ids,
            "l1.ple.rows": rows,
            "l1.ple.e": e,
            "l1.ple.key": key,
            "l1.ple.value": value,
            "l1.ple.gate": gate.reshape(t, HC),
            "l1.ple.gv": gv.reshape(t, -1),
            "l1.ple.nrm": nrm,
            "l1.ple.conv": conv,
        }
    )
    return xn


# --------------------------------------------------------------------------------------------
# self-checks
# --------------------------------------------------------------------------------------------


def self_checks(ck: Checkpoint, cfg: dict, ngram_ids) -> dict:
    results = {}
    torch.manual_seed(0)

    # 1. vLLM GatedResidual vs transformers Qwen4ExpTextGatedResidual (f32)
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual
    from vllm.models.qwen4_exp.common.hyperconnection import GatedResidual, HyperConnectionConfig

    small = dict(hc_count=4, hidden_size=64, hc_lowrank=8, rms_norm_eps=1e-6)
    tcfg = Qwen4ExpTextConfig(**small)
    tr = Qwen4ExpTextGatedResidual(tcfg).float()
    vr = GatedResidual(
        HyperConnectionConfig(
            hc_count=4, hidden_size=64, params_dtype=F32, hc_lowrank=8, rms_norm_eps=1e-6, hc_per_branch_norm=True
        )
    )
    with torch.no_grad():
        for prm in tr.parameters():
            prm.uniform_(-0.3, 0.3)
        vr.hc_norm.weight.copy_(tr.hc_norm.weight)
        vr.input_mix_weight_down.weight.copy_(tr.input_mix_weight_down.weight)
        vr.input_mix_weight_up.weight.copy_(tr.input_mix_weight_up.weight)
        vr.block_inject_weight.weight.copy_(tr.block_inject_weight.weight)
        x = torch.randn(5, 256)
        mixed_t, _, inj_t = tr(x)
        blk = torch.randn(5, 64)
        comb_t = x.unflatten(-1, (4, 64)) + blk.unsqueeze(-2) * inj_t.unsqueeze(-1)
        mixed_v, res = vr.mix(x)
        comb_v = vr.combine(blk, res)
    err = max((mixed_t - mixed_v).abs().max().item(), (comb_t.flatten(-2) - comb_v).abs().max().item())
    assert err <= 1e-6, f"GatedResidual mismatch {err}"
    results["gated_residual_vs_transformers"] = err

    # 2. vLLM n-gram ids equal transformers' Qwen4ExpTextNGramEmbedding ids
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    full = Qwen4ExpTextConfig(
        vocab_size=cfg["vocab_size"],
        ngram_size=cfg["ngram_size"],
        heads_per_ngram=cfg["heads_per_ngram"],
        ngram_vocab_size_base=cfg["ngram_vocab_size_base"],
        make_ngram_vocab_size_divisible_by=cfg["make_ngram_vocab_size_divisible_by"],
        seed=cfg["seed"],
        eos_token_id=cfg["eos_token_id"],
    )
    ref = Qwen4ExpTextNGramEmbedding.__new__(Qwen4ExpTextNGramEmbedding)
    torch.nn.Module.__init__(ref)
    ref.ngram_size = full.ngram_size
    ref.context_len = full.ngram_size - 1
    ref.heads_per_ngram = full.heads_per_ngram
    ref.ngram_heads = (full.ngram_size - 1) * full.heads_per_ngram
    ref.eos_token_id = cfg["eos_token_id"]
    pre = "layers.1.ple.ple_embedding"
    ref.layer_multipliers = ck.lm(f"{pre}.layer_multipliers")
    ref.ngram_heads_vocab_sizes = ck.lm(f"{pre}.ngram_heads_vocab_sizes")
    ref.ngram_heads_offsets = ck.lm(f"{pre}.ngram_heads_offsets")
    captured = {}

    class Grab(torch.nn.Module):
        def forward(self, idx):
            captured["ids"] = idx.clone()
            return torch.zeros(*idx.shape, 1)

    ref.ngram_embedding = Grab()
    ref.ngram_embedding.weight = torch.zeros(1, device="cpu")
    toks = [5, 17, cfg["eos_token_id"], 9, 9, 250, 11, cfg["eos_token_id"], 3, 4, 5]
    ref.forward(torch.tensor([toks]), None)
    same = torch.equal(captured["ids"][0], ngram_ids(toks))
    assert same, "vLLM and transformers n-gram ids differ"
    results["ngram_ids_vs_transformers"] = bool(same)

    # 3. chunked vs recurrent gated delta rule (f32)
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    qq = F.normalize(torch.randn(1, 70, 3, 16), dim=-1)
    kk = F.normalize(torch.randn(1, 70, 3, 16), dim=-1)
    vv = torch.randn(1, 70, 3, 16)
    gg = -torch.rand(1, 70, 3)
    bb = torch.rand(1, 70, 3)
    oc, sc = torch_chunk_gated_delta_rule(qq, kk, vv, gg, bb, output_final_state=True)
    orr, sr = torch_recurrent_gated_delta_rule(qq, kk, vv, gg, bb, output_final_state=True)
    err = max((oc - orr).abs().max().item(), (sc - sr).abs().max().item())
    assert err <= 1e-5, f"chunked vs recurrent {err}"
    results["chunk_vs_recurrent"] = err

    # 4. the engine's ngram.rs hash_real_constants
    want = [6380558, 26411572, 56460672, 78566983, 94693008, 106742196, 124822692, 148942556,
            164226950, 190352573, 210933682, 238908951, 242182004, 265475238, 299910982, 312121804]
    got = ngram_ids([9707, 11, 1879])[2].tolist()
    assert got == want, f"hash constants {got}"
    results["hash_real_constants"] = True

    # 5. every kernel-table sha256 (checked before running; recorded here)
    results["kernel_table"] = len(KERNELS)
    return results


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if os.environ.get("VLLM_TARGET_DEVICE") != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "":
        raise SystemExit("run with VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=")
    torch.set_num_threads(4)
    torch.manual_seed(0)

    table = check_kernels()
    ck = Checkpoint(args.snapshot)
    cfg = json.loads((args.snapshot / "config.json").read_text())["text_config"]

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
    ids = tok.encode(T_TEXT).ids
    assert len(ids) >= NTOK, len(ids)
    tokens = ids[:NTOK]
    tokens[EOS_POS] = EOS

    ngram_ids = ngram_module(ck, cfg)
    checks = self_checks(ck, cfg, ngram_ids)

    out: dict[str, torch.Tensor] = {"tokens": torch.tensor(tokens, dtype=torch.int64)}
    emb = ck.take(PREFIX + "embed_tokens.weight", tokens).to(F32)
    out["embed"] = emb
    x = emb.repeat(1, HC)  # [T, 4*2560], stream outer

    pending = None  # (block output, injection) of the previous MLP
    for i in range(4):
        attn = Branch(ck, f"layers.{i}.attn_hyper_connection")
        mlp = Branch(ck, f"layers.{i}.mlp_hyper_connection")
        if pending is not None:
            x = hc_combine(x, *pending)
            out[f"l{i}.x"] = x
        if i == 1:
            x = ple(ck, x, ngram_ids(tokens), out)
            out["l1.xp"] = x
        m = attn.mix(x)
        if i % 4 == 3:
            y = attention(ck, i, m["u"], out)
        else:
            y = gdn(ck, i, m["u"], out, store=(i == 0))
        out.update({f"l{i}.attn.u": m["u"], f"l{i}.attn.inj": m["inj"], f"l{i}.attn.y": y})
        if i == 0:
            out["l0.attn.xn"] = m["xn"]
        x = hc_combine(x, y, m["inj"])
        if i in (0, 3):
            out[f"l{i}.mid"] = x
        n = mlp.mix(x)
        out.update({f"l{i}.mlp.u": n["u"], f"l{i}.mlp.inj": n["inj"]})
        yb = moe(ck, i, n["u"], out, store=(i == 3))
        pending = (yb, n["inj"])
        print(f"layer {i} done", flush=True)

    x = hc_combine(x, *pending)
    out["l4.x"] = x
    head = Branch(ck, "hyper_connection_mixer", inject=False)
    h = head.mix(x)["u"]
    out["head.h"] = h
    out["head.logits"] = gemm(h, ck.rows("lm_head.weight", 0, HEAD_ROWS))

    meta = {
        "vllm": __import__("vllm").__version__,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "snapshot": str(args.snapshot),
        "tokens": json.dumps(tokens),
        "rope": "default",
        "script_sha256": sha256(Path(__file__)),
        "kernels": json.dumps({k: v["sha256"] for k, v in table.items()}),
        "checks": json.dumps(checks),
    }
    from safetensors.torch import save_file

    tensors = {}
    for k, v in out.items():
        v = v.contiguous()
        if v.dtype == F32 and k not in ("l0.gdn.g", "l0.gdn.beta", "l0.gdn.state", "l3.attn.os",
                                        "l3.shared.s", "l3.mlp.gs", "l1.ple.gate",
                                        "l0.weights", "l1.weights", "l2.weights", "l3.weights"):
            # every other f32 tensor holds bf16 values: store it as bf16
            assert torch.equal(bf(v), v), f"{k} is not bf16-exact"
            v = v.to(BF16)
        tensors[k] = v
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata=meta)
    size = args.out.stat().st_size
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
    print(f"checks: {checks}")
    print(f"wrote {args.out} ({size / 1e6:.2f} MB, {len(tensors)} tensors), sha256 {sha256(args.out)}")
    print(f"peak RSS {rss:.2f} GiB")
    assert size <= 16_000_000, size
    assert rss < 2.0, rss


if __name__ == "__main__":
    main()
