"""Golden block-FP8 GEMM outputs from vLLM's own CUTLASS kernel.

Writes one safetensors file of synthetic operands and vLLM's bf16 outputs for
hanzo-quant's `blockwise_fp8` tests. Operands are seeded, never read from a
checkpoint; weight-scale statistics are the ones measured on the
Qwen3.8-Flash-Next fp8hybrid tensors, hardcoded here.

Layout stored (hanzo's): qa [M,K] e4m3, sa [M,K/128] f32 row-major,
qw [N,K] e4m3, sw [N/128,K/128] f32, out [Mmax,N] bf16. vLLM's kernel reads
activation scales M-major, so only the call below transposes them.

Run on the GB10 under the build lock (< 300 MiB of device memory):
  flock /tmp/hanzo-engine-build.lock systemd-run --user --scope --quiet \
    -p MemoryMax=6G -p MemorySwapMax=0 choom -n 1000 -- nice -n19 ionice -c3 \
    /home/z/vllm-env/bin/python scripts/blockwise_fp8_golden.py \
    --out hanzo-quant/tests/fixtures/blockwise_fp8.safetensors
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import save_file

import vllm
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)

FP8 = torch.float8_e4m3fn
BLOCK = 128
MAX_BYTES = 16 << 20
VLLM_COMMIT = "98dff2a81d747d1dba01a47f939f48c3526d4206"

M_LIST = [1, 2, 3, 4, 5, 7, 8, 13, 16, 31, 32, 33, 63, 64, 65, 100, 128, 255, 256, 257, 300]
M_DOWN = M_LIST + [512, 1024, 2048]

# (name, N, K, log2 mean, log2 sd) of weight_scale_inv, measured on six Q4 tensors.
CASES = [
    ("o", 256, 6144, -11.28, 0.63),
    ("k", 512, 2560, -11.53, 0.50),
    ("qkv", 384, 2560, -11.25, 0.63),
    ("down", 512, 640, -11.86, 0.46),
]
# Adjacent-K correlation of log2 scales: gives a median |ratio-1| near 0.2.
RHO = 0.85


def dispatch(m):
    if m <= 64:
        return "swap_ab_128x32x128"
    if m <= 256:
        return "pingpong_64x128x128"
    return "cooperative_128x128x128"


def block_weights(n, k, gen):
    """E4M3 codes of N(0,1) blocks, each block's amax mapped to 448."""
    w = torch.randn(n, k, generator=gen)
    blocks = w.view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
    return (blocks * (448.0 / amax)).view(n, k).to(FP8)


def weight_scales(n, k, mu, sd, gen):
    rows, cols = n // BLOCK, k // BLOCK
    z = torch.empty(rows, cols, dtype=torch.float64)
    z[:, 0] = torch.randn(rows, generator=gen, dtype=torch.float64)
    for c in range(1, cols):
        e = torch.randn(rows, generator=gen, dtype=torch.float64)
        z[:, c] = RHO * z[:, c - 1] + (1 - RHO * RHO) ** 0.5 * e
    return torch.exp2(mu + sd * z).to(torch.float32)


def extreme(gen):
    n = k = 384
    finite = [b for b in range(256) if (b & 0x7F) != 0x7F]  # drop the two NaNs
    assert len(finite) == 254
    lut = torch.tensor(finite, dtype=torch.uint8)
    codes = lut[torch.randint(0, 254, (n, k), generator=gen)]
    # Pin +-0 and +-448 so they are present regardless of the draw.
    codes[0, :4] = torch.tensor([0x00, 0x80, 0x7E, 0xFE], dtype=torch.uint8)
    qw = codes.view(FP8)
    u = torch.rand(n // BLOCK, k // BLOCK, generator=gen, dtype=torch.float64)
    sw = torch.exp(torch.log(torch.tensor(1e-6)) + u * (torch.log(torch.tensor(1e-1)) - torch.log(torch.tensor(1e-6)))).float()
    return qw, sw


def activations(m, k, gen):
    x = torch.randn(m, k, generator=gen)
    outliers = torch.randperm(k, generator=gen)[:2]
    x[:, outliers] *= 60.0
    if m > 7:
        x[7] = 0.0
    if m > 11:
        x[11] = 0.0
        x[11, int(torch.randint(0, k, (1,), generator=gen))] = 1e4
    return x.to(torch.bfloat16)


def stats(sw):
    l2 = torch.log2(sw.double())
    out = {"log2_mean": float(l2.mean()), "log2_sd": float(l2.std())}
    if sw.shape[1] > 1:
        ratio = (sw[:, 1:].double() / sw[:, :-1].double() - 1).abs().flatten()
        out["adjacent_k_ratio_median"] = float(ratio.median())
        out["adjacent_k_identical"] = float((sw[:, 1:] == sw[:, :-1]).double().mean())
    return out


def vllm_mm(qa, sa, qw, sw):
    # vLLM's CutlassFp8BlockScaledMMKernel: A, B.T, As column-major, Bs.T.
    out = torch.empty(qa.shape[0], qw.shape[0], dtype=torch.bfloat16, device=qa.device)
    sa_cm = sa.t().contiguous().t()
    torch.ops._C.cutlass_scaled_mm(out, qa, qw.t(), sa_cm, sw.t(), None)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    cc = torch.cuda.get_device_capability(dev)
    assert cc == (12, 1), f"expected a GB10 (cc 12.1), got {cc}"
    gen = torch.Generator().manual_seed(0x51A8)

    tensors = {}
    meta_cases = {}
    acts = {}
    specs = CASES + [("extreme", 384, 384, None, None)]
    for name, n, k, mu, sd in specs:
        m_list = M_DOWN if name == "down" else M_LIST
        m_max = max(m_list)
        if name == "extreme":
            qw, sw = extreme(gen)
        else:
            qw = block_weights(n, k, gen)
            sw = weight_scales(n, k, mu, sd, gen)
        key = (k, m_max)
        if key not in acts:
            x = activations(m_max, k, gen).to(dev)
            qa, sa = per_token_group_quant_fp8(x, BLOCK, column_major_scales=False, use_ue8m0=False)
            assert sa.is_contiguous() and sa.shape == (m_max, k // BLOCK)
            acts[key] = (qa, sa)
            tensors[f"act.k{k}.m{m_max}.qa"] = qa.cpu()
            tensors[f"act.k{k}.m{m_max}.sa"] = sa.cpu()
        qa, sa = acts[key]
        qw_d, sw_d = qw.to(dev), sw.to(dev)
        full = vllm_mm(qa, sa, qw_d, sw_d)
        for m in m_list:
            part = vllm_mm(qa[:m].contiguous(), sa[:m].contiguous(), qw_d, sw_d)
            same = torch.equal(part.view(torch.int16), full[:m].view(torch.int16))
            assert same, f"{name}: vLLM output at M={m} differs from rows of M={m_max}"
        torch.cuda.synchronize()
        tensors[f"{name}.qw"] = qw
        tensors[f"{name}.sw"] = sw
        tensors[f"{name}.out"] = full.cpu()
        meta_cases[name] = {
            "n": n,
            "k": k,
            "act": f"act.k{k}.m{m_max}",
            "m": m_list,
            "dispatch": {str(m): dispatch(m) for m in m_list},
            "weight_scale": stats(sw),
            "target_log2_mean": mu,
            "target_log2_sd": sd,
        }
        print(f"{name}: N={n} K={k} Mmax={m_max} rows independent of M: ok", flush=True)

    so = os.path.join(os.path.dirname(vllm.__file__), "_C_stable_libtorch.abi3.so")
    metadata = {
        "synthetic": "true",
        "vllm_version": vllm.__version__,
        "vllm_commit": VLLM_COMMIT,
        "vllm_c_sha256": sha256(so),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(dev),
        "cc": f"{cc[0]}.{cc[1]}",
        "cutlass": "v4.4.2",
        "op": "torch.ops._C.cutlass_scaled_mm(out, qa, qw.t(), sa_colmajor, sw.t(), None)",
        "dispatch_rule": "M<=64 swap_ab 128x32x128 cooperative; M<=256 64x128x128 pingpong; else 128x128x128 cooperative",
        "activation_quant": "per_token_group_quant_fp8(x, 128, column_major_scales=False, use_ue8m0=False)",
        "cases": json.dumps(meta_cases),
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(tensors, args.out, metadata=metadata)
    size = os.path.getsize(args.out)
    print(f"wrote {args.out}: {size / 1e6:.2f} MB, peak device alloc "
          f"{torch.cuda.max_memory_allocated(dev) / 2**20:.1f} MiB")
    if size > MAX_BYTES:
        print(f"fixture is {size} bytes, above the {MAX_BYTES} byte cap", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
