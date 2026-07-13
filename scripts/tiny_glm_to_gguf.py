"""
Tiny GLM-5.2 (arch glm_moe_dsa) F32 safetensors -> standard `glm-dsa` GGUF, full precision.

The oracle model is already F32, so there is nothing to dequantize: every tensor lands as F32
(token-exactness needs full precision, not Q4_K). We reuse the FP8 converter's `build_plan` +
`GgufSlotWriter` + `convert_source` (the canonical HF-name -> glm-dsa-GGUF-name mapping and slot
writer) with expert_type = attn_type = embed_type = mtp_type = F32. Indexer tensors
(`self_attn.indexer.*`) and any nextn/MTP block are omitted: `convert_source` drops any HF name not
in `plan.routes`, and with `num_nextn_predict_layers` absent (=> 0) the plan emits no MTP block.

Then VERIFY the emitted GGUF against what the loader (models/quantized_deepseek2.rs + gguf_moe.rs,
arch glm-dsa) actually dereferences: architecture, required metadata, and every tensor name/shape.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
from safetensors import safe_open

import gguf
from gguf.constants import GGMLQuantizationType as GT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_fp8_to_gguf import (
    GgufSlotWriter,
    build_plan,
    convert_source,
    expected_slots,
)

SRC_DIR = Path(
    "/tmp/claude-1000/-home-z/0715b931-9f69-4f7d-9442-e431e9674360/scratchpad/oracle/glm_tiny"
)
OUT = Path(
    "/tmp/claude-1000/-home-z/0715b931-9f69-4f7d-9442-e431e9674360/scratchpad/oracle/glm_tiny_f32.gguf"
)


def st_source(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (hf_name, f32 ndarray) for every tensor in the safetensors file (already F32)."""
    with safe_open(str(path), framework="numpy") as f:
        for name in f.keys():
            yield name, np.ascontiguousarray(f.get_tensor(name), dtype=np.float32)


def convert() -> tuple[dict, object]:
    config = json.loads((SRC_DIR / "config.json").read_text())
    plan = build_plan(config, "glm-tiny-oracle", GT.F32, GT.F32, GT.F32, GT.F32)

    w = GgufSlotWriter(str(OUT), plan)
    w.open()
    done: set[str] = set()
    convert_source(w, plan, st_source(SRC_DIR / "model.safetensors"), done, lambda s: None)
    w.close()

    missing = expected_slots(plan) - done
    if missing:
        raise SystemExit(f"INCOMPLETE: unfilled slots {sorted(missing)}")
    return config, plan


# tensor names the loader dereferences (models/quantized_deepseek2.rs + gguf_moe.rs, arch glm-dsa).
def loader_expected_tensors(config: dict) -> list[str]:
    H = config["hidden_size"]
    NL = config["num_hidden_layers"]
    FKD = config["first_k_dense_replace"]
    names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    for i in range(NL):
        p = f"blk.{i}"
        names += [
            f"{p}.attn_norm.weight", f"{p}.ffn_norm.weight",
            f"{p}.attn_q_a.weight", f"{p}.attn_q_a_norm.weight", f"{p}.attn_q_b.weight",
            f"{p}.attn_kv_a_mqa.weight", f"{p}.attn_kv_a_norm.weight",
            f"{p}.attn_kv_b.weight", f"{p}.attn_output.weight",
        ]
        if i < FKD:
            names += [f"{p}.ffn_gate.weight", f"{p}.ffn_up.weight", f"{p}.ffn_down.weight"]
        else:
            names += [
                f"{p}.ffn_gate_inp.weight", f"{p}.exp_probs_b.bias",
                f"{p}.ffn_gate_exps.weight", f"{p}.ffn_up_exps.weight", f"{p}.ffn_down_exps.weight",
                f"{p}.ffn_gate_shexp.weight", f"{p}.ffn_up_shexp.weight", f"{p}.ffn_down_shexp.weight",
            ]
    return names


# metadata keys the loader reads via ContentMetadata.get_value (glm-dsa. prefix stripped there).
LOADER_META = [
    "glm-dsa.embedding_length", "glm-dsa.attention.head_count", "glm-dsa.expert_count",
    "glm-dsa.expert_gating_func", "glm-dsa.rope.dimension_count", "glm-dsa.attention.key_length_mla",
    "glm-dsa.attention.value_length_mla", "glm-dsa.block_count",
    "glm-dsa.attention.layer_norm_rms_epsilon", "glm-dsa.context_length", "glm-dsa.rope.freq_base",
    "glm-dsa.attention.q_lora_rank", "glm-dsa.attention.kv_lora_rank", "glm-dsa.expert_shared_count",
    "glm-dsa.expert_feed_forward_length", "glm-dsa.feed_forward_length", "glm-dsa.expert_used_count",
    "glm-dsa.leading_dense_block_count",
]


def verify(config: dict, plan) -> bool:
    reader = gguf.GGUFReader(str(OUT))
    fails: list[str] = []

    fields = {f.name: f for f in reader.fields.values()}
    arch = fields.get("general.architecture")
    arch_val = arch.contents() if arch else None
    if arch_val != "glm-dsa":
        fails.append(f"general.architecture={arch_val!r} want 'glm-dsa'")

    for k in LOADER_META:
        if k not in fields:
            fails.append(f"missing metadata {k}")

    # value spot-checks against config
    def num(k):
        f = fields.get(k)
        return f.contents() if f else None

    checks = {
        "glm-dsa.embedding_length": config["hidden_size"],
        "glm-dsa.attention.head_count": config["num_attention_heads"],
        "glm-dsa.expert_count": config["n_routed_experts"],
        "glm-dsa.expert_gating_func": 2,  # sigmoid/noaux
        "glm-dsa.rope.dimension_count": config["qk_rope_head_dim"],
        "glm-dsa.attention.key_length_mla": config["qk_nope_head_dim"] + config["qk_rope_head_dim"],
        "glm-dsa.attention.value_length_mla": config["v_head_dim"],
        "glm-dsa.block_count": config["num_hidden_layers"],  # NEXTN=0
        "glm-dsa.attention.q_lora_rank": config["q_lora_rank"],
        "glm-dsa.attention.kv_lora_rank": config["kv_lora_rank"],
        "glm-dsa.expert_shared_count": config["n_shared_experts"],
        "glm-dsa.expert_feed_forward_length": config["moe_intermediate_size"],
        "glm-dsa.feed_forward_length": config["intermediate_size"],
        "glm-dsa.expert_used_count": config["num_experts_per_tok"],
        "glm-dsa.leading_dense_block_count": config["first_k_dense_replace"],
    }
    for k, want in checks.items():
        got = num(k)
        if got != want:
            fails.append(f"{k}={got} want {want}")
    eps = num("glm-dsa.attention.layer_norm_rms_epsilon")
    if eps is None or abs(float(eps) - float(config["rms_norm_eps"])) > 1e-12:
        fails.append(f"rms_eps={eps} want {config['rms_norm_eps']}")
    # nextn must be 0/absent -> no MTP block
    nextn = num("glm-dsa.nextn_predict_layers")
    if nextn not in (None, 0):
        fails.append(f"nextn_predict_layers={nextn} want 0/absent")

    # tensors: every loader-expected name present, all F32, no indexer/MTP leaked.
    tensors = {t.name: t for t in reader.tensors}
    for nm in loader_expected_tensors(config):
        if nm not in tensors:
            fails.append(f"missing tensor {nm}")
    for nm, t in tensors.items():
        if t.tensor_type != GT.F32:
            fails.append(f"{nm} type {t.tensor_type.name} want F32")
        if "indexer" in nm or ".nextn." in nm:
            fails.append(f"leaked tensor {nm}")

    # shape spot-checks (GGUF reader returns dims reversed vs the logical (out,in) plan shape).
    H = config["hidden_size"]; NE = config["n_routed_experts"]; MOEI = config["moe_intermediate_size"]
    QL = config["q_lora_rank"]; KVL = config["kv_lora_rank"]
    NH = config["num_attention_heads"]
    QKN = config["qk_nope_head_dim"]; QKR = config["qk_rope_head_dim"]; VH = config["v_head_dim"]
    FKD = config["first_k_dense_replace"]
    q_head = QKN + QKR

    def shape(nm):
        return list(int(x) for x in tensors[nm].shape) if nm in tensors else None

    shape_expect = {
        "token_embd.weight": [H, config["vocab_size"]],
        "output.weight": [H, config["vocab_size"]],
        "blk.0.attn_q_a.weight": [H, QL],
        "blk.0.attn_q_b.weight": [QL, NH * q_head],
        "blk.0.attn_kv_a_mqa.weight": [H, KVL + QKR],
        "blk.0.attn_kv_b.weight": [KVL, NH * (QKN + VH)],
        "blk.0.attn_output.weight": [NH * VH, H],
        f"blk.{FKD}.ffn_gate_inp.weight": [H, NE],
        f"blk.{FKD}.exp_probs_b.bias": [NE],
        # rank-3 expert bank: GGUF dims reversed -> [hidden, moe_inter, n_expert]
        f"blk.{FKD}.ffn_gate_exps.weight": [H, MOEI, NE],
        f"blk.{FKD}.ffn_down_exps.weight": [MOEI, H, NE],
        f"blk.{FKD}.ffn_gate_shexp.weight": [H, MOEI],
    }
    for nm, want in shape_expect.items():
        got = shape(nm)
        if got != want:
            fails.append(f"shape {nm}={got} want {want}")

    # expert bank data round-trips exactly (F32) vs source expert 0.
    with safe_open(str(SRC_DIR / "model.safetensors"), framework="numpy") as f:
        e0 = np.ascontiguousarray(
            f.get_tensor(f"model.layers.{FKD}.mlp.experts.0.gate_proj.weight"), dtype=np.float32
        )
    bank = np.asarray(tensors[f"blk.{FKD}.ffn_gate_exps.weight"].data).astype(np.float32)
    bank = bank.reshape(NE, MOEI, H)  # [n_expert, moe_inter, hidden]
    if not np.array_equal(bank[0], e0):
        fails.append("expert-0 gate bank does NOT byte-match source (F32 must be exact)")

    ok = not fails
    print("=" * 60)
    print(f"arch = {arch_val!r}")
    print(f"tensors = {len(tensors)}  metadata fields = {len(fields)}")
    print(f"F32-only = {all(t.tensor_type == GT.F32 for t in tensors.values())}")
    print(f"indexer/MTP leaked = {any('indexer' in n or '.nextn.' in n for n in tensors)}")
    if fails:
        print(f"VERIFY FAILED ({len(fails)}):")
        for x in fails:
            print(f"  - {x}")
    else:
        print("VERIFY PASSED: names + shapes + metadata match the glm-dsa loader; experts exact.")
    return ok


def main():
    config, plan = convert()
    ok = verify(config, plan)

    reader = gguf.GGUFReader(str(OUT))
    tensor_names = [t.name for t in reader.tensors]
    meta_keys = sorted(f.name for f in reader.fields.values())
    # names_match_loader: every loader-required name present AND nothing extraneous (indexer/MTP) leaked.
    required = set(loader_expected_tensors(config))
    present = set(tensor_names)
    names_match = required.issubset(present) and not any(
        "indexer" in n or ".nextn." in n for n in present
    )

    result = {
        "gguf_path": str(OUT),
        "tensor_count": len(tensor_names),
        "metadata_keys": meta_keys,
        "names_match_loader": bool(names_match and ok),
        "arch": "glm-dsa",
    }
    print("\nRESULT " + json.dumps(result))
    sys.exit(0 if (ok and names_match) else 1)


if __name__ == "__main__":
    main()
