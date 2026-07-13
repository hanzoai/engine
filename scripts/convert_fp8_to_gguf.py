"""
FP8 (GLM-5.2) -> standard llama.cpp-style `glm-dsa` GGUF, shard-by-shard, disk-safe.

Studied from colibri's `tools/convert_fp8_to_int4.py`: download ONE ~5 GB safetensors
shard, convert its tensors, DELETE the shard, move on. Peak disk = one shard + the growing
output. The difference: colibri writes its OWN int4 container; this writes a STANDARD GGUF
that hanzo-engine's `glm-dsa` loader (models/quantized_deepseek2.rs, arch "glm-dsa") reads.

What "standard glm-dsa GGUF" means here, matched byte-for-byte to that loader:
  - general.architecture = "glm-dsa"; split-MLA metadata (key_length = qk_nope, key_length_mla =
    q_head_dim, value_length_mla = v_head_dim, rope.dimension_count = qk_rope)
  - routed experts stacked into rank-3 banks `blk.{i}.ffn_{gate,up,down}_exps.weight` (Q4_K)
  - router `blk.{i}.ffn_gate_inp.weight` + no-aux bias `blk.{i}.exp_probs_b.bias` kept F32
  - expert_gating_func = 2 (sigmoid), leading_dense_block_count, expert_group_count = 1 (noaux)
  - the trailing MTP / nextn block is emitted at Q8_0 (an int4 draft head measures ~0% acceptance,
    colibri issue #8 -- speculation never starts), and nextn_predict_layers records it so the
    text-forward loader drops it from block_count.

GGUF is a single file whose whole tensor directory (name, shape, type, byte offset) precedes the
data blob, so it can't be appended shard-by-shard like per-shard safetensors. We split the two:
  1. PLAN from config.json -> the complete tensor directory + metadata (no weights). Write the
     header + KV + tensor-info, preallocate the file. Every tensor's absolute offset is now fixed.
  2. STREAM shards; each source tensor (or one expert's slice of a bank) is dequantized from FP8,
     requantized to its GGUF type, and pwrite()n straight to its slot. Delete the shard. A sidecar
     records finished slots so an interrupted run resumes.

Quantization uses gguf-py's canonical routines where it has them (Q8_0, F32) and a faithful
llama.cpp `quantize_row_q4_K_ref` for Q4_K (gguf-py 0.19 dequantizes Q4_K but does not quantize
it). `--selftest` proves the Q4_K blocks decode through gguf-py's own dequantizer and that a tiny
synthetic glm-dsa GGUF round-trips every key and tensor the engine loader requires.

USAGE
  python3 scripts/convert_fp8_to_gguf.py --selftest                       # no checkpoint needed
  python3 scripts/convert_fp8_to_gguf.py --indir glm52_fp8 --out glm52.gguf
  python3 scripts/convert_fp8_to_gguf.py --repo zai-org/GLM-5.2-FP8 --out glm52.gguf   # stream+delete
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

import gguf
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as GT, GGUFValueType as VT

QK_K = 256  # K-quant super-block

# ------------------------------------------------------------------ FP8 dequant --

def _dequant_fp8_block(weight, scale_inv):
    """FP8 e4m3 weight with a 128x128 block scale grid -> f32, matching colibri's dequant."""
    import torch

    w = weight.to(torch.float32)
    o, i = w.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:o, :i]
    return (w * s).numpy()


def iter_safetensors(path: str) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, f32 ndarray) for a shard: FP8+block-scale dequantized, BF16/F16/F32 upcast.

    `*.weight_scale_inv` companions are consumed with their weight, never yielded on their own.
    """
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for name in f.keys():
            if name.endswith("_scale_inv"):
                continue
            dt = f.get_slice(name).get_dtype()
            if dt in ("F8_E4M3", "float8_e4m3fn"):
                sname = name + "_scale_inv"
                if sname not in keys:
                    raise ValueError(f"{name}: FP8 tensor without {sname}")
                yield name, _dequant_fp8_block(f.get_tensor(name), f.get_tensor(sname))
            else:
                import torch

                yield name, f.get_tensor(name).to(torch.float32).numpy()


# ------------------------------------------------------------------ quantizers --
# gguf-py 0.19 implements quantize() for F32/F16/Q8_0/Q4_0/... but NOT the K-quants, while its
# dequantize() covers Q4_K. So Q8_0 and F32 go through gguf-py; Q4_K is the faithful llama.cpp
# reference below, validated by round-tripping through gguf-py's dequantize() in --selftest.

def _nearest_int(f: np.ndarray) -> np.ndarray:
    """llama.cpp nearest_int: round-half-to-even via the 2^23 float trick (|f| < 2^22)."""
    v = np.ascontiguousarray(f, dtype=np.float32) + np.float32(12582912.0)
    return (v.view(np.int32) & 0x007FFFFF) - 0x00400000


def _make_qkx2(x: np.ndarray, w: np.ndarray, nmax: int, rmin: float, rdelta: float,
               nstep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized llama.cpp make_qkx2_quants (use_mad=false) over B independent 32-vectors.

    Returns (scale[B], the_min[B] = -min, L[B, n] in [0, nmax]).
    """
    B, n = x.shape
    xmin = np.minimum(x.min(1), 0.0)
    xmax = x.max(1)
    sum_w = w.sum(1)
    sum_x = (w * x).sum(1)
    rng = xmax - xmin
    valid = rng > 0.0

    denom = np.where(valid, rng, 1.0)
    iscale = np.where(valid, nmax / denom, 0.0)
    scale = np.where(valid, denom / nmax, 0.0)  # 1/iscale
    L = np.clip(_nearest_int(iscale[:, None] * (x - xmin[:, None])), 0, nmax)
    diff = scale[:, None] * L + xmin[:, None] - x
    best_mad = (w * (diff * diff)).sum(1)

    cur_min = xmin.copy()
    cur_scale = scale.copy()
    bestL = L.copy()
    for is_ in range(nstep + 1):
        iscale = np.where(valid, (rmin + rdelta * is_ + nmax) / np.where(valid, xmax - cur_min, 1.0), 0.0)
        Laux = np.clip(_nearest_int(iscale[:, None] * (x - cur_min[:, None])), 0, nmax).astype(np.float64)
        sl = (w * Laux).sum(1)
        sl2 = (w * Laux * Laux).sum(1)
        sxl = (w * Laux * x).sum(1)
        D = sum_w * sl2 - sl * sl
        good = valid & (D > 0)
        Dsafe = np.where(good, D, 1.0)
        this_scale = (sum_w * sxl - sum_x * sl) / Dsafe
        this_min = (sl2 * sum_x - sl * sxl) / Dsafe
        # min>0 is disallowed: pin min=0 and refit scale from sum_xl/sum_l2.
        pos = this_min > 0.0
        sl2safe = np.where(sl2 > 0, sl2, 1.0)
        this_scale = np.where(pos, sxl / sl2safe, this_scale)
        this_min = np.where(pos, 0.0, this_min)
        pred = this_scale[:, None] * Laux + this_min[:, None] - x
        mad = (w * (pred * pred)).sum(1)
        take = good & (mad < best_mad)
        if take.any():
            bestL[take] = Laux[take].astype(bestL.dtype)
            best_mad = np.where(take, mad, best_mad)
            cur_scale = np.where(take, this_scale, cur_scale)
            cur_min = np.where(take, this_min, cur_min)
    return cur_scale.astype(np.float32), (-cur_min).astype(np.float32), bestL.astype(np.int32)


def quantize_q4_k(x2d: np.ndarray) -> np.ndarray:
    """Faithful llama.cpp quantize_row_q4_K_ref. x2d [rows, cols], cols % 256 == 0 -> uint8 blocks."""
    x = np.ascontiguousarray(x2d, dtype=np.float32)
    rows, cols = x.shape
    if cols % QK_K:
        raise ValueError(f"Q4_K needs cols %% {QK_K} == 0, got {cols}")
    nsb = rows * (cols // QK_K)
    sb = x.reshape(nsb, QK_K)
    sub = sb.reshape(nsb * 8, 32)

    sum_x2 = (sub * sub).sum(1)
    av_x = np.sqrt(sum_x2 / 32.0)
    weights = av_x[:, None] + np.abs(sub)
    scales, mins, _ = _make_qkx2(sub, weights, 15, -1.0, 0.1, 20)
    scales = scales.reshape(nsb, 8)
    mins = mins.reshape(nsb, 8)

    max_scale = scales.max(1)
    max_min = mins.max(1)
    inv_scale = np.where(max_scale > 0, 63.0 / np.where(max_scale > 0, max_scale, 1.0), 0.0)
    inv_min = np.where(max_min > 0, 63.0 / np.where(max_min > 0, max_min, 1.0), 0.0)
    ls = np.minimum(_nearest_int(inv_scale[:, None] * scales), 63).astype(np.uint8)
    lm = np.minimum(_nearest_int(inv_min[:, None] * mins), 63).astype(np.uint8)

    sc_bytes = np.zeros((nsb, 12), np.uint8)
    for j in range(4):
        sc_bytes[:, j] = ls[:, j]
        sc_bytes[:, j + 4] = lm[:, j]
    for j in range(4, 8):
        sc_bytes[:, j + 4] = (ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)
        sc_bytes[:, j - 4] |= (ls[:, j] >> 4) << 6
        sc_bytes[:, j] |= (lm[:, j] >> 4) << 6

    d = (max_scale / 63.0).astype(np.float16)
    dmin = (max_min / 63.0).astype(np.float16)

    # Recover per-sub-block (sc, m) exactly as get_scale_min_k4 does, then quantize the nibbles.
    sc = np.zeros((nsb, 8), np.float32)
    m = np.zeros((nsb, 8), np.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = sc_bytes[:, j] & 63
            m[:, j] = sc_bytes[:, j + 4] & 63
        else:
            sc[:, j] = (sc_bytes[:, j + 4] & 0xF) | ((sc_bytes[:, j - 4] >> 6) << 4)
            m[:, j] = (sc_bytes[:, j + 4] >> 4) | ((sc_bytes[:, j] >> 6) << 4)
    d1 = d.astype(np.float32)[:, None] * sc
    dm1 = dmin.astype(np.float32)[:, None] * m
    d1safe = np.where(d1 != 0, d1, 1.0)
    subq = sb.reshape(nsb, 8, 32)
    L = np.where(d1[:, :, None] != 0,
                 np.clip(_nearest_int((subq + dm1[:, :, None]) / d1safe[:, :, None]), 0, 15), 0)
    L = L.astype(np.uint8).reshape(nsb, 8, 32)

    qs = np.zeros((nsb, 128), np.uint8)
    for k in range(4):
        qs[:, k * 32:(k + 1) * 32] = L[:, 2 * k, :] | (L[:, 2 * k + 1, :] << 4)

    block = np.concatenate([d.view(np.uint8).reshape(nsb, 2),
                            dmin.view(np.uint8).reshape(nsb, 2),
                            sc_bytes, qs], axis=1)  # [nsb, 144]
    return block.reshape(rows, (cols // QK_K) * 144)


def quantize(x2d: np.ndarray, t: GT) -> np.ndarray:
    if t == GT.F32:
        return np.ascontiguousarray(x2d, dtype=np.float32)
    if t == GT.F16:
        return np.ascontiguousarray(x2d, dtype=np.float16)
    if t == GT.Q4_K:
        return quantize_q4_k(x2d)
    if t in (GT.Q8_0, GT.Q4_0, GT.Q5_0, GT.Q4_1, GT.Q5_1):
        return gguf.quants.quantize(np.ascontiguousarray(x2d, dtype=np.float32), t)
    raise ValueError(f"unsupported quant type {t.name}")


def type_bytes(shape: tuple[int, ...], t: GT) -> int:
    bs, ts = GGML_QUANT_SIZES[t]
    n = int(np.prod(shape))
    return n // bs * ts


def choose_type(shape: tuple[int, ...], want: GT) -> GT:
    """Fall back when the innermost dim can't tile the block: K-quant -> Q8_0 -> F16 -> F32."""
    last = shape[-1]
    order = [want, GT.Q8_0, GT.F16, GT.F32]
    for t in order:
        bs = GGML_QUANT_SIZES[t][0]
        if last % bs == 0:
            return t
    return GT.F32


# ------------------------------------------------------------------ GLM-5.2 plan --

@dataclass
class Route:
    target: str          # GGUF tensor name
    ggml_type: GT
    slice_index: Optional[int] = None   # expert index within a rank-3 bank, else None
    n_slices: int = 1                    # expert count for the bank (for the slot byte-stride)


@dataclass
class Plan:
    meta: list[tuple[str, object, VT]]                 # (key, value, gguf value type)
    tensors: list[tuple[str, tuple[int, ...], GT]]     # ordered directory (name, shape, type)
    routes: dict[str, Route]                           # hf tensor name -> where it lands
    arch: str = "glm-dsa"


def _cfg(c: dict, *keys, default=None):
    for k in keys:
        if k in c:
            return c[k]
    if default is not None:
        return default
    raise KeyError(keys)


def build_plan(config: dict, name: str, expert_type: GT, attn_type: GT,
               embed_type: GT, mtp_type: GT) -> Plan:
    H = _cfg(config, "hidden_size")
    NL = _cfg(config, "num_hidden_layers")
    NH = _cfg(config, "num_attention_heads")
    QL = _cfg(config, "q_lora_rank", default=0)
    KVL = _cfg(config, "kv_lora_rank")
    QKN = _cfg(config, "qk_nope_head_dim")
    QKR = _cfg(config, "qk_rope_head_dim")
    VH = _cfg(config, "v_head_dim")
    NE = _cfg(config, "n_routed_experts")
    TOPK = _cfg(config, "num_experts_per_tok")
    NS = _cfg(config, "n_shared_experts", default=0)
    MOEI = _cfg(config, "moe_intermediate_size")
    DENSEI = _cfg(config, "intermediate_size")
    FKD = _cfg(config, "first_k_dense_replace", default=0)
    NEXTN = _cfg(config, "num_nextn_predict_layers", default=0)
    VOCAB = _cfg(config, "vocab_size")
    CTX = _cfg(config, "max_position_embeddings")
    EPS = float(_cfg(config, "rms_norm_eps", default=1e-5))
    rope = config.get("rope_parameters") or {}
    THETA = float(rope.get("rope_theta", config.get("rope_theta", 10000.0)))
    SCALE = float(_cfg(config, "routed_scaling_factor", default=1.0))
    NORM = bool(_cfg(config, "norm_topk_prob", default=True))
    NGROUP = _cfg(config, "n_group", default=1)
    TOPKG = _cfg(config, "topk_group", default=1)
    sigmoid = _cfg(config, "scoring_func", default="sigmoid") == "sigmoid"
    q_head = QKN + QKR
    tied = bool(config.get("tie_word_embeddings", False))

    meta: list[tuple[str, object, VT]] = [
        ("general.name", name, VT.STRING),
        ("general.quantization_version", 2, VT.UINT32),
        ("general.file_type", 15, VT.UINT32),  # MOSTLY_Q4_K_M
        ("glm-dsa.context_length", CTX, VT.UINT32),
        ("glm-dsa.embedding_length", H, VT.UINT32),
        ("glm-dsa.block_count", NL + NEXTN, VT.UINT32),
        ("glm-dsa.feed_forward_length", DENSEI, VT.UINT32),
        ("glm-dsa.expert_feed_forward_length", MOEI, VT.UINT32),
        ("glm-dsa.attention.head_count", NH, VT.UINT32),
        ("glm-dsa.attention.head_count_kv", NH, VT.UINT32),
        ("glm-dsa.attention.layer_norm_rms_epsilon", EPS, VT.FLOAT32),
        ("glm-dsa.attention.kv_lora_rank", KVL, VT.UINT32),
        ("glm-dsa.attention.key_length", QKN, VT.UINT32),
        ("glm-dsa.attention.key_length_mla", q_head, VT.UINT32),
        ("glm-dsa.attention.value_length", VH, VT.UINT32),
        ("glm-dsa.attention.value_length_mla", VH, VT.UINT32),
        ("glm-dsa.rope.dimension_count", QKR, VT.UINT32),
        ("glm-dsa.rope.freq_base", THETA, VT.FLOAT32),
        ("glm-dsa.expert_count", NE, VT.UINT32),
        ("glm-dsa.expert_used_count", TOPK, VT.UINT32),
        ("glm-dsa.expert_shared_count", NS, VT.UINT32),
        ("glm-dsa.expert_gating_func", 2 if sigmoid else 1, VT.UINT32),
        ("glm-dsa.expert_weights_scale", SCALE, VT.FLOAT32),
        ("glm-dsa.expert_weights_norm", NORM, VT.BOOL),
        ("glm-dsa.expert_group_count", NGROUP, VT.UINT32),
        ("glm-dsa.expert_group_used_count", TOPKG, VT.UINT32),
        ("glm-dsa.leading_dense_block_count", FKD, VT.UINT32),
        ("glm-dsa.nextn_predict_layers", NEXTN, VT.UINT32),
        ("glm-dsa.vocab_size", VOCAB, VT.UINT32),
        ("glm-dsa.attention.q_lora_rank", QL, VT.UINT32),
    ]

    tensors: list[tuple[str, tuple[int, ...], GT]] = []
    routes: dict[str, Route] = {}

    def add(target, shape, want, hf_name=None, slice_index=None, n_slices=1):
        t = choose_type(shape, want)
        tensors.append((target, tuple(shape), t))
        if hf_name is not None:
            routes[hf_name] = Route(target, t, slice_index, n_slices)
        return t

    def add_bank(target, shape, want, n_slices):
        # rank-3 bank registered once; per-expert routes point at the shared type.
        t = choose_type(shape, want)
        tensors.append((target, tuple(shape), t))
        return t

    # embeddings + head
    add("token_embd.weight", (VOCAB, H), embed_type, "model.embed_tokens.weight")
    add("output_norm.weight", (H,), GT.F32, "model.norm.weight")
    if tied:
        routes["model.embed_tokens.weight"] = Route("token_embd.weight", choose_type((VOCAB, H), embed_type))
    else:
        add("output.weight", (VOCAB, H), embed_type, "lm_head.weight")

    def emit_layer(i: int, prefix_layer: int, moe: bool, dense: bool, w_attn: GT, w_ffn: GT):
        p = f"blk.{i}"
        hp = f"model.layers.{prefix_layer}"
        add(f"{p}.attn_norm.weight", (H,), GT.F32, f"{hp}.input_layernorm.weight")
        add(f"{p}.ffn_norm.weight", (H,), GT.F32, f"{hp}.post_attention_layernorm.weight")
        if QL:
            add(f"{p}.attn_q_a.weight", (QL, H), w_attn, f"{hp}.self_attn.q_a_proj.weight")
            add(f"{p}.attn_q_a_norm.weight", (QL,), GT.F32, f"{hp}.self_attn.q_a_layernorm.weight")
            add(f"{p}.attn_q_b.weight", (NH * q_head, QL), w_attn, f"{hp}.self_attn.q_b_proj.weight")
        else:
            add(f"{p}.attn_q.weight", (NH * q_head, H), w_attn, f"{hp}.self_attn.q_proj.weight")
        add(f"{p}.attn_kv_a_mqa.weight", (KVL + QKR, H), w_attn, f"{hp}.self_attn.kv_a_proj_with_mqa.weight")
        add(f"{p}.attn_kv_a_norm.weight", (KVL,), GT.F32, f"{hp}.self_attn.kv_a_layernorm.weight")
        add(f"{p}.attn_kv_b.weight", (NH * (QKN + VH), KVL), w_attn, f"{hp}.self_attn.kv_b_proj.weight")
        add(f"{p}.attn_output.weight", (H, NH * VH), w_attn, f"{hp}.self_attn.o_proj.weight")

        if dense:
            add(f"{p}.ffn_gate.weight", (DENSEI, H), w_ffn, f"{hp}.mlp.gate_proj.weight")
            add(f"{p}.ffn_up.weight", (DENSEI, H), w_ffn, f"{hp}.mlp.up_proj.weight")
            add(f"{p}.ffn_down.weight", (H, DENSEI), w_ffn, f"{hp}.mlp.down_proj.weight")
        if moe:
            add(f"{p}.ffn_gate_inp.weight", (NE, H), GT.F32, f"{hp}.mlp.gate.weight")
            add(f"{p}.exp_probs_b.bias", (NE,), GT.F32, f"{hp}.mlp.gate.e_score_correction_bias")
            gt = add_bank(f"{p}.ffn_gate_exps.weight", (NE, MOEI, H), expert_type, NE)
            ut = add_bank(f"{p}.ffn_up_exps.weight", (NE, MOEI, H), expert_type, NE)
            dt = add_bank(f"{p}.ffn_down_exps.weight", (NE, H, MOEI), expert_type, NE)
            for e in range(NE):
                he = f"{hp}.mlp.experts.{e}"
                routes[f"{he}.gate_proj.weight"] = Route(f"{p}.ffn_gate_exps.weight", gt, e, NE)
                routes[f"{he}.up_proj.weight"] = Route(f"{p}.ffn_up_exps.weight", ut, e, NE)
                routes[f"{he}.down_proj.weight"] = Route(f"{p}.ffn_down_exps.weight", dt, e, NE)
            if NS:
                add(f"{p}.ffn_gate_shexp.weight", (NS * MOEI, H), w_ffn, f"{hp}.mlp.shared_experts.gate_proj.weight")
                add(f"{p}.ffn_up_shexp.weight", (NS * MOEI, H), w_ffn, f"{hp}.mlp.shared_experts.up_proj.weight")
                add(f"{p}.ffn_down_shexp.weight", (H, NS * MOEI), w_ffn, f"{hp}.mlp.shared_experts.down_proj.weight")

    for i in range(NL):
        dense = i < FKD
        emit_layer(i, i, moe=not dense, dense=dense, w_attn=attn_type, w_ffn=attn_type)

    # MTP / nextn block: full transformer layer + MTP heads, all Q8_0 (int4 head => 0% acceptance).
    for k in range(NEXTN):
        i = NL + k
        emit_layer(i, i, moe=True, dense=False, w_attn=mtp_type, w_ffn=mtp_type)
        hp = f"model.layers.{i}"
        p = f"blk.{i}"
        add(f"{p}.nextn.enorm.weight", (H,), GT.F32, f"{hp}.enorm.weight")
        add(f"{p}.nextn.hnorm.weight", (H,), GT.F32, f"{hp}.hnorm.weight")
        add(f"{p}.nextn.shared_head_norm.weight", (H,), GT.F32, f"{hp}.shared_head.norm.weight")
        add(f"{p}.nextn.eh_proj.weight", (H, 2 * H), mtp_type, f"{hp}.eh_proj.weight")
        add(f"{p}.nextn.embed_tokens.weight", (VOCAB, H), embed_type, f"{hp}.embed_tokens.weight")
        add(f"{p}.nextn.shared_head_head.weight", (VOCAB, H), embed_type, f"{hp}.shared_head.head.weight")

    return Plan(meta=meta, tensors=tensors, routes=routes)


# ------------------------------------------------------------------ GGUF writer --
# gguf-py's GGUFWriter emits a correct header/KV/tensor-info directory; we then compute each
# tensor's absolute offset (identical algorithm to its write_ti_data_to_file) and pwrite() data
# into slots. This is the only way to fill a single GGUF from streamed, out-of-order shards.

class GgufSlotWriter:
    def __init__(self, out_path: str, plan: Plan):
        self.out_path = out_path
        self.plan = plan
        self.w = gguf.GGUFWriter(path=None, arch=plan.arch)
        for key, val, vt in plan.meta:
            self.w.add_key_value(key, val, vt)
        for name, shape, t in plan.tensors:
            nbytes = type_bytes(shape, t)
            if t in (GT.F32, GT.F16):
                dt = np.float32 if t == GT.F32 else np.float16
                self.w.add_tensor_info(name, shape, np.dtype(dt), nbytes)
            else:
                # Pass the logical element shape with a non-uint8 sentinel dtype: add_tensor_info
                # only re-derives the shape from bytes when tensor_dtype is uint8, which we avoid.
                self.w.add_tensor_info(name, shape, np.dtype(np.float32), nbytes, raw_dtype=t)
        self.offset: dict[str, int] = {}
        self.nbytes: dict[str, int] = {}
        self.fd = -1

    def open(self):
        """Non-destructive open: the header/KV/tensor-info prefix is deterministic, so serialize it
        to a temp file, then pwrite it into the real file WITHOUT O_TRUNC. A resumed run reuses the
        same prefix and keeps every tensor slot already on disk (O_TRUNC/"wb" would zero them)."""
        pad = gguf.GGUFWriter.ggml_pad
        align = self.w.data_alignment
        hdr = self.out_path + ".hdr.tmp"
        self.w.open_output_file(Path(hdr))
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_ti_data_to_file()
        fout = self.w.fout[0]
        cur = fout.tell()
        fout.write(b"\x00" * (pad(cur, align) - cur))  # align the data section start
        fout.flush()
        fout.close()
        prefix = Path(hdr).read_bytes()
        os.remove(hdr)

        data_start = len(prefix)
        off = 0
        for name, ti in self.w.tensors[0].items():
            self.offset[name] = data_start + off
            self.nbytes[name] = ti.nbytes
            off += pad(ti.nbytes, align)
        self.total = data_start + off

        self.fd = os.open(self.out_path, os.O_RDWR | os.O_CREAT, 0o644)
        os.pwrite(self.fd, prefix, 0)          # header only touches [0, data_start); never tensor data
        os.ftruncate(self.fd, self.total)      # same size on resume -> preserves written slots

    def _pwrite_all(self, b: bytes, off: int):
        """Write every byte at `off`. A single os.pwrite is capped by Linux at ~2 GB and
        short-writes, which would silently truncate a large tensor (an F32 token_embd is
        ~3.8 GB) while its slot is marked done. Loop until the whole buffer lands."""
        mv = memoryview(b)
        written = 0
        while written < len(mv):
            n = os.pwrite(self.fd, mv[written:], off + written)
            if n <= 0:
                raise IOError(f"pwrite made no progress at offset {off + written}")
            written += n

    def write_full(self, name: str, data: np.ndarray):
        b = data.tobytes()
        if len(b) != self.nbytes[name]:
            raise ValueError(f"{name}: {len(b)} bytes, slot wants {self.nbytes[name]}")
        self._pwrite_all(b, self.offset[name])

    def write_slice(self, name: str, index: int, n_slices: int, data: np.ndarray):
        b = data.tobytes()
        stride = self.nbytes[name] // n_slices
        if len(b) != stride:
            raise ValueError(f"{name}[{index}]: {len(b)} bytes, per-expert stride {stride}")
        self._pwrite_all(b, self.offset[name] + index * stride)

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


# ------------------------------------------------------------------ conversion --

def _as_2d(arr: np.ndarray) -> np.ndarray:
    return arr if arr.ndim == 2 else arr.reshape(1, -1)


def convert_source(w: GgufSlotWriter, plan: Plan, src: Iterator[tuple[str, np.ndarray]],
                   done: set[str], on_done: Callable[[str], None]) -> int:
    """Route every tensor a source yields into its GGUF slot. Returns count written this pass."""
    written = 0
    for hf_name, arr in src:
        r = plan.routes.get(hf_name)
        if r is None:
            continue  # DSA indexer / bias / unused: not part of the glm-dsa forward
        slot_key = r.target if r.slice_index is None else f"{r.target}#{r.slice_index}"
        if slot_key in done:
            continue
        if arr.ndim == 1:
            q = quantize(arr.reshape(1, -1) if r.ggml_type not in (GT.F32, GT.F16) else arr, r.ggml_type)
        else:
            q = quantize(_as_2d(arr), r.ggml_type)
        if r.slice_index is None:
            w.write_full(r.target, q)
        else:
            w.write_slice(r.target, r.slice_index, r.n_slices, q)
        done.add(slot_key)
        on_done(slot_key)
        written += 1
    return written


def expected_slots(plan: Plan) -> set[str]:
    slots = set()
    for hf_name, r in plan.routes.items():
        slots.add(r.target if r.slice_index is None else f"{r.target}#{r.slice_index}")
    return slots


# ------------------------------------------------------------------ shard driver --

def load_config(indir: Optional[str], repo: Optional[str], workdir: str) -> dict:
    if indir:
        return json.loads(Path(indir, "config.json").read_text())
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "config.json", local_dir=str(Path(workdir, "_meta")))
    return json.loads(Path(p).read_text())


def shard_list(indir: Optional[str], repo: Optional[str]):
    if indir:
        return sorted(glob.glob(str(Path(indir, "*.safetensors"))))
    from huggingface_hub import HfApi
    info = HfApi().repo_info(repo, files_metadata=True)
    return sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors"))


def run_convert(args):
    workdir = str(Path(args.out).with_suffix("")) + "_work"
    os.makedirs(workdir, exist_ok=True)
    config = load_config(args.indir, args.repo, workdir)
    name = args.name or (args.repo or Path(args.indir or ".").name).split("/")[-1]
    plan = build_plan(config, name, GT[args.expert_type], GT[args.attn_type],
                      GT[args.embed_type], GT[args.mtp_type])

    w = GgufSlotWriter(args.out, plan)
    w.open()
    sidecar = args.out + ".progress.json"
    # Resume only if the persisted signature matches this plan; a different --*-type or config
    # shifts every tensor offset, so a mismatched sidecar would corrupt the file -- start fresh.
    sig = {"total": w.total, "n": len(plan.tensors)}
    done: set[str] = set()
    if os.path.exists(sidecar):
        state = json.loads(Path(sidecar).read_text())
        if state.get("sig") == sig:
            done = set(state.get("done", []))
            print(f"resume: {len(done)} slots already written")
        else:
            print("sidecar plan signature mismatch; restarting from scratch")
    if not done:
        print(f"planned {len(plan.tensors)} tensors, file preallocated to {w.total/1e9:.2f} GB")

    def flush_progress():
        Path(sidecar).write_text(json.dumps({"sig": sig, "done": sorted(done)}))

    def checkpoint(_slot):
        if len(done) % 512 == 0:
            flush_progress()

    shards = shard_list(args.indir, args.repo)
    tmp = Path(workdir, "_inflight")
    tmp.mkdir(exist_ok=True)
    total = 0
    for idx, sh in enumerate(shards):
        if args.indir:
            path = sh
        else:
            if shutil.disk_usage(workdir).free / 1e9 < args.min_free_gb:
                print(f"STOP: < {args.min_free_gb} GB free; free space and rerun to resume")
                break
            from huggingface_hub import hf_hub_download
            print(f"[{idx+1}/{len(shards)}] downloading {sh}", flush=True)
            path = hf_hub_download(args.repo, sh, local_dir=str(tmp))
        n = convert_source(w, plan, iter_safetensors(path), done, checkpoint)
        total += n
        if not args.indir:
            os.remove(path)
        flush_progress()
        print(f"    shard {idx+1}: +{n} slots ({len(done)} total)", flush=True)

    w.close()
    # The main text model (blk.0..NL-1 + embeddings/head) is the loader-visible model; the
    # trailing MTP/nextn block (blk.{NL..NL+NEXTN-1}) is a speculative draft head the text loader
    # drops. DeepSeek/GLM checkpoints frequently tie or omit the MTP embed_tokens/shared_head.head,
    # so a missing MTP slot must NOT fail an otherwise-complete conversion: the file is loadable.
    NL = int(_cfg(config, "num_hidden_layers"))
    NEXTN = int(_cfg(config, "num_nextn_predict_layers", default=0))
    mtp_prefixes = tuple(f"blk.{NL + k}." for k in range(NEXTN))
    missing = expected_slots(plan) - done
    missing_mtp = {m for m in missing if mtp_prefixes and m.startswith(mtp_prefixes)}
    missing_main = missing - missing_mtp
    if missing_main:
        print(f"INCOMPLETE: {len(missing_main)} slots unfilled (rerun to resume). e.g. {sorted(missing_main)[:3]}")
        return 1
    if missing_mtp:
        print(f"WARNING: {len(missing_mtp)} MTP/nextn slot(s) unfilled -- source omitted/renamed them. "
              f"The text model (blk.0..{NL - 1}) is complete and loadable; a future speculative loader "
              f"would need these. e.g. {sorted(missing_mtp)[:3]}")
    os.path.exists(sidecar) and os.remove(sidecar)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"DONE: {args.out} ({w.total/1e9:.2f} GB, {len(plan.tensors)} tensors)")
    return 0


# ------------------------------------------------------------------ selftest --

TINY_CONFIG = {
    "hidden_size": 512, "num_hidden_layers": 3, "num_attention_heads": 4,
    "q_lora_rank": 256, "kv_lora_rank": 128, "qk_nope_head_dim": 64, "qk_rope_head_dim": 64,
    "v_head_dim": 64, "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
    "moe_intermediate_size": 256, "intermediate_size": 512, "first_k_dense_replace": 1,
    "num_nextn_predict_layers": 1, "vocab_size": 512, "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5, "rope_parameters": {"rope_theta": 8000000.0},
    "routed_scaling_factor": 2.5, "norm_topk_prob": True, "n_group": 1, "topk_group": 1,
    "scoring_func": "sigmoid", "tie_word_embeddings": False,
}

# The exact keys/tensors models/quantized_deepseek2.rs (arch "glm-dsa") + utils/gguf_metadata.rs
# require to load; --selftest asserts every one is present and typed correctly.
REQUIRED_META = [
    "glm-dsa.attention.head_count", "glm-dsa.block_count", "glm-dsa.embedding_length",
    "glm-dsa.attention.layer_norm_rms_epsilon", "glm-dsa.attention.kv_lora_rank",
    "glm-dsa.attention.key_length", "glm-dsa.rope.dimension_count", "glm-dsa.expert_count",
    "glm-dsa.expert_used_count", "glm-dsa.expert_feed_forward_length", "glm-dsa.feed_forward_length",
    "glm-dsa.context_length",
]


def synth_source(config: dict) -> Iterator[tuple[str, np.ndarray]]:
    """Deterministic in-memory 'checkpoint' with GLM-5.2 HF names, matching TINY_CONFIG shapes."""
    rng = np.random.default_rng(0)
    H = config["hidden_size"]; NL = config["num_hidden_layers"]; NH = config["num_attention_heads"]
    QL = config["q_lora_rank"]; KVL = config["kv_lora_rank"]
    QKN = config["qk_nope_head_dim"]; QKR = config["qk_rope_head_dim"]; VH = config["v_head_dim"]
    NE = config["n_routed_experts"]; NS = config["n_shared_experts"]
    MOEI = config["moe_intermediate_size"]; DENSEI = config["intermediate_size"]
    FKD = config["first_k_dense_replace"]; NEXTN = config["num_nextn_predict_layers"]
    VOCAB = config["vocab_size"]; qh = QKN + QKR

    def r(*s):
        return (rng.standard_normal(s) * 0.1).astype(np.float32)

    yield "model.embed_tokens.weight", r(VOCAB, H)
    yield "model.norm.weight", r(H)
    yield "lm_head.weight", r(VOCAB, H)
    for i in range(NL + NEXTN):
        hp = f"model.layers.{i}"
        dense = i < FKD
        yield f"{hp}.input_layernorm.weight", r(H)
        yield f"{hp}.post_attention_layernorm.weight", r(H)
        yield f"{hp}.self_attn.q_a_proj.weight", r(QL, H)
        yield f"{hp}.self_attn.q_a_layernorm.weight", r(QL)
        yield f"{hp}.self_attn.q_b_proj.weight", r(NH * qh, QL)
        yield f"{hp}.self_attn.kv_a_proj_with_mqa.weight", r(KVL + QKR, H)
        yield f"{hp}.self_attn.kv_a_layernorm.weight", r(KVL)
        yield f"{hp}.self_attn.kv_b_proj.weight", r(NH * (QKN + VH), KVL)
        yield f"{hp}.self_attn.o_proj.weight", r(H, NH * VH)
        if dense:
            yield f"{hp}.mlp.gate_proj.weight", r(DENSEI, H)
            yield f"{hp}.mlp.up_proj.weight", r(DENSEI, H)
            yield f"{hp}.mlp.down_proj.weight", r(H, DENSEI)
        else:
            yield f"{hp}.mlp.gate.weight", r(NE, H)
            yield f"{hp}.mlp.gate.e_score_correction_bias", r(NE)
            for e in range(NE):
                yield f"{hp}.mlp.experts.{e}.gate_proj.weight", r(MOEI, H)
                yield f"{hp}.mlp.experts.{e}.up_proj.weight", r(MOEI, H)
                yield f"{hp}.mlp.experts.{e}.down_proj.weight", r(H, MOEI)
            if NS:
                yield f"{hp}.mlp.shared_experts.gate_proj.weight", r(NS * MOEI, H)
                yield f"{hp}.mlp.shared_experts.up_proj.weight", r(NS * MOEI, H)
                yield f"{hp}.mlp.shared_experts.down_proj.weight", r(H, NS * MOEI)
        if i >= NL:
            yield f"{hp}.enorm.weight", r(H)
            yield f"{hp}.hnorm.weight", r(H)
            yield f"{hp}.shared_head.norm.weight", r(H)
            yield f"{hp}.eh_proj.weight", r(H, 2 * H)
            yield f"{hp}.embed_tokens.weight", r(VOCAB, H)
            yield f"{hp}.shared_head.head.weight", r(VOCAB, H)


def selftest() -> int:
    fails = []

    # 1. Q4_K blocks must decode through gguf-py's canonical dequantizer within tolerance.
    a = (np.random.default_rng(1).standard_normal((6, 512)) * 0.3).astype(np.float32)
    q = quantize_q4_k(a)
    d = gguf.quants.dequantize(q, GT.Q4_K).astype(np.float32)
    rel = float(np.abs(d - a).mean() / np.abs(a).mean())
    ok = rel < 0.10
    print(f"[q4_k] gguf-py dequant round-trip relerr={rel:.4f} {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append("q4_k relerr too high")

    # 2. Build a tiny glm-dsa GGUF through the real writer, read it back, verify the contract.
    out = str(Path(_scratch(), "glm_dsa_tiny.gguf"))
    plan = build_plan(TINY_CONFIG, "glm-dsa-tiny", GT.Q4_K, GT.Q4_K, GT.Q8_0, GT.Q8_0)
    w = GgufSlotWriter(out, plan)
    w.open()
    done: set[str] = set()
    convert_source(w, plan, synth_source(TINY_CONFIG), done, lambda s: None)
    w.close()

    missing = expected_slots(plan) - done
    if missing:
        fails.append(f"unfilled slots: {sorted(missing)[:5]}")
        print(f"[write] FAIL unfilled: {sorted(missing)[:5]}")
    else:
        print(f"[write] all {len(done)} tensor slots filled")

    reader = gguf.GGUFReader(out)
    rmeta = {f.name: f for f in reader.fields.values()}
    arch = _field_str(rmeta.get("general.architecture"))
    print(f"[read] general.architecture = {arch!r}")
    if arch != "glm-dsa":
        fails.append(f"arch {arch!r}")

    for k in REQUIRED_META:
        if k not in rmeta:
            fails.append(f"missing meta {k}")
            print(f"[read] MISSING meta {k}")
    # value + type spot checks against the reader's expectations
    checks = {
        "glm-dsa.block_count": TINY_CONFIG["num_hidden_layers"] + TINY_CONFIG["num_nextn_predict_layers"],
        "glm-dsa.nextn_predict_layers": TINY_CONFIG["num_nextn_predict_layers"],
        "glm-dsa.expert_count": TINY_CONFIG["n_routed_experts"],
        "glm-dsa.expert_gating_func": 2,
        "glm-dsa.leading_dense_block_count": TINY_CONFIG["first_k_dense_replace"],
        "glm-dsa.attention.key_length": TINY_CONFIG["qk_nope_head_dim"],
        "glm-dsa.attention.key_length_mla": TINY_CONFIG["qk_nope_head_dim"] + TINY_CONFIG["qk_rope_head_dim"],
        "glm-dsa.rope.dimension_count": TINY_CONFIG["qk_rope_head_dim"],
    }
    for k, want in checks.items():
        got = _field_num(rmeta.get(k))
        if got != want:
            fails.append(f"{k}={got} want {want}")
            print(f"[read] {k}={got} want {want} FAIL")
    eps = _field_num(rmeta.get("glm-dsa.attention.layer_norm_rms_epsilon"))
    if not (abs(eps - 1e-5) < 1e-9):
        fails.append(f"rms_eps={eps}")
    # reader-visible transformer layer count = block_count - nextn
    n_layers = checks["glm-dsa.block_count"] - checks["glm-dsa.nextn_predict_layers"]
    print(f"[read] loader-visible layers = {n_layers} (block_count {checks['glm-dsa.block_count']} - nextn {checks['glm-dsa.nextn_predict_layers']})")

    # tensors the loader dereferences by name for layer 0 (dense) and layer 1 (moe)
    names = {t.name for t in reader.tensors}
    needed = [
        "token_embd.weight", "output_norm.weight", "output.weight",
        "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight", "blk.0.attn_q_a.weight",
        "blk.0.attn_q_a_norm.weight", "blk.0.attn_q_b.weight", "blk.0.attn_kv_a_mqa.weight",
        "blk.0.attn_kv_a_norm.weight", "blk.0.attn_kv_b.weight", "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
        "blk.1.ffn_gate_inp.weight", "blk.1.exp_probs_b.bias",
        "blk.1.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight",
        "blk.1.ffn_gate_shexp.weight", "blk.1.ffn_up_shexp.weight", "blk.1.ffn_down_shexp.weight",
        "blk.3.nextn.eh_proj.weight",  # MTP block (NL=3)
    ]
    for nm in needed:
        if nm not in names:
            fails.append(f"missing tensor {nm}")
            print(f"[read] MISSING tensor {nm}")

    # rank-3 expert bank shape + type as the fused-MoE path expects: ne = [hidden, moe_inter, n_expert]
    exps = next(t for t in reader.tensors if t.name == "blk.1.ffn_gate_exps.weight")
    want_ne = [TINY_CONFIG["hidden_size"], TINY_CONFIG["moe_intermediate_size"], TINY_CONFIG["n_routed_experts"]]
    got_ne = list(int(x) for x in exps.shape)
    print(f"[read] ffn_gate_exps ne={got_ne} type={exps.tensor_type.name}")
    if got_ne != want_ne:
        fails.append(f"expert ne {got_ne} want {want_ne}")
    if exps.tensor_type != GT.Q4_K:
        fails.append(f"expert type {exps.tensor_type.name} want Q4_K")

    mtp = next(t for t in reader.tensors if t.name == "blk.3.nextn.eh_proj.weight")
    print(f"[read] MTP eh_proj type={mtp.tensor_type.name} (want Q8_0 -- int4-head trap)")
    if mtp.tensor_type != GT.Q8_0:
        fails.append(f"mtp type {mtp.tensor_type.name} want Q8_0")

    # expert bank data round-trips: decode expert 0's gate slice, compare to source
    src = dict(synth_source(TINY_CONFIG))
    e0 = src["model.layers.1.mlp.experts.0.gate_proj.weight"]
    bank = gguf.quants.dequantize(np.asarray(exps.data), GT.Q4_K).astype(np.float32)
    bank = bank.reshape(want_ne[2], want_ne[1], want_ne[0])  # [n_expert, moe_inter, hidden]
    rel_e = float(np.abs(bank[0] - e0).mean() / np.abs(e0).mean())
    print(f"[read] expert-0 gate bank round-trip relerr={rel_e:.4f} {'OK' if rel_e < 0.12 else 'FAIL'}")
    if rel_e >= 0.12:
        fails.append(f"expert bank relerr {rel_e}")

    print("=" * 60)
    if fails:
        print(f"SELFTEST FAILED ({len(fails)}):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("SELFTEST PASSED: glm-dsa GGUF header + metadata + tensors round-trip.")
    return 0


def _scratch() -> str:
    d = os.environ.get("SCRATCH", "/tmp")
    os.makedirs(d, exist_ok=True)
    return d


def _field_str(field) -> Optional[str]:
    if field is None:
        return None
    return field.contents()


def _field_num(field):
    if field is None:
        return None
    v = field.contents()
    return v


# ------------------------------------------------------------------ CLI --

def main():
    ap = argparse.ArgumentParser(description="FP8 GLM-5.2 -> glm-dsa GGUF (disk-safe, resumable)")
    ap.add_argument("--indir", help="local dir of FP8 safetensors + config.json")
    ap.add_argument("--repo", help="HF repo, e.g. zai-org/GLM-5.2-FP8 (stream+delete each shard)")
    ap.add_argument("--out", help="output .gguf path")
    ap.add_argument("--name", help="general.name (default: repo/dir basename)")
    ap.add_argument("--expert-type", default="Q4_K", help="routed expert bank type (default Q4_K)")
    ap.add_argument("--attn-type", default="Q4_K", help="attention + dense/shared FFN type (default Q4_K)")
    ap.add_argument("--embed-type", default="Q8_0", help="token_embd/output type (default Q8_0)")
    ap.add_argument("--mtp-type", default="Q8_0", help="MTP/nextn block type (default Q8_0)")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true", help="synthetic round-trip, no checkpoint")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.out or not (a.indir or a.repo):
        ap.error("need --out and one of --indir/--repo (or --selftest)")
    sys.exit(run_convert(a))


if __name__ == "__main__":
    main()
