#!/usr/bin/env python3
"""bytes_per_token.py on synthetic safetensors headers and a real gb20y-top nsys export."""
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bytes_per_token as bpt  # noqa: E402


def write_shard(path, tensors):
    """A safetensors file whose data section is zeros; only the header matters."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = bpt.numel(shape) * bpt.DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob + bytes(offset))


CONFIG = {"text_config": {
    "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_num_value_heads": 4, "linear_num_key_heads": 2, "linear_key_head_dim": 8,
    "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
}}


def run(argv):
    out = io.StringIO()
    with redirect_stdout(out):
        bpt.main(argv)
    return json.loads(out.getvalue())


class Weights(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        write_shard(os.path.join(self.dir, "model-00001-of-00002.safetensors"), {
            "model.language_model.embed_tokens.weight": ("BF16", [100, 64]),   # 1 row: 128 B
            "model.language_model.layers.0.mlp.up_proj.weight": ("U8", [128, 32]),  # 4096 B
            "model.language_model.layers.0.mlp.up_proj.weight_scale": ("F8_E4M3", [128, 4]),  # 512
            "model.language_model.layers.0.mlp.up_proj.weight_scale_2": ("F32", []),  # 4
            "model.visual.blocks.0.weight": ("BF16", [1000, 1000]),  # skipped
        })
        write_shard(os.path.join(self.dir, "model-00002-of-00002.safetensors"), {
            "model.language_model.layers.3.self_attn.q_proj.weight": ("F8_E4M3", [64, 64]),  # 4096
            "lm_head.weight": ("BF16", [100, 64]),  # 12800
            "mtp.fc.weight": ("BF16", [64, 128]),  # skipped
        })
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(CONFIG, f)

    def test_known_sum(self):
        out = run(["weights", "--checkpoint", self.dir, "--context", "10", "--kv-dtype", "bf16"])
        self.assertEqual(out["weights_bytes"], 4096 + 512 + 4 + 4096 + 12800)
        self.assertEqual(out["embed_row_bytes"], 128)
        # 1 full layer x 2 KV heads x 16 x (K, V) x 2 B x 10 tokens
        self.assertEqual(out["kv_bytes"], 1 * 2 * 16 * 2 * 2 * 10)
        ssm = 3 * 4 * 8 * 8 * 4 * 2
        conv = 3 * (2 * 2 * 8 + 4 * 8) * 3 * 2 * 2
        self.assertEqual(out["state_bytes"], ssm + conv)
        self.assertEqual(out["bytes_per_token"],
                         out["weights_bytes"] + 128 + out["kv_bytes"] + ssm + conv)
        fp8 = run(["weights", "--checkpoint", self.dir, "--context", "10", "--kv-dtype", "fp8"])
        self.assertEqual(fp8["kv_bytes"], out["kv_bytes"] // 2)

    def test_resident_counts_logical_elements_and_drops_scales(self):
        out = run(["weights", "--checkpoint", self.dir, "--context", "0", "--kv-dtype", "bf16",
                   "--resident", r"mlp\.up_proj=bf16"])
        # 128 x 32 packed bytes are 128 x 64 FP4 values, held as bf16; the scales are not read.
        self.assertEqual(out["resident"], {r"mlp\.up_proj": 128 * 64 * 2})
        self.assertEqual(out["weights_bytes"], 128 * 64 * 2 + 4096 + 12800)


class Nsys(unittest.TestCase):
    def test_gb20y_top_fixture(self):
        # 50 GPU_METRICS rows copied with their exact schema from a real dgx capture
        # (nsys 2026.3.2, --gpu-metrics-set=gb20y-top); 25 are the L2 byte counter.
        path = os.path.join(HERE, "testdata", "gb20y-top.sqlite")
        out = run(["nsys", "--sqlite", path, "--tokens", "2"])
        self.assertEqual(out["l2_bytes"], 1314448)
        self.assertEqual(out["l2_bytes_per_token"], 1314448 / 2)
        self.assertEqual(out["decode_ranges"], 0)
        self.assertNotIn("gpu_s_per_step", out)


if __name__ == "__main__":
    unittest.main()
