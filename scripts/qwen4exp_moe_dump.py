#!/usr/bin/env python3
"""Dump vLLM's Flash-Next router gate inputs and logits on real activations, for the
teacher-forced routing bar (hanzo's Lane::linear + route::topk on vLLM's x must reproduce
vLLM's logits, ids and weights bitwise on every (token, layer) pair).

Needs the whole GPU: run only in a window with hanzo-vllm stopped (scripts/qwen4exp_moe_window.sh).
The run is eager: the gate's graph replay equals eager (gate.safetensors asserts it per M).

For each concurrency c in {1, 3, 4} it generates greedily for the section-3b prompts (a uuid
nonce, WORDS x size, "Summarize in one sentence."; size 4 is ~220 tokens, size 40 ~1,335) and
writes target/qwen4exp_moe_window/dump_c{c}.safetensors:

    x.{call}.{layer}       gate input  [M, 2560] bf16 (M is that forward's token count)
    logits.{call}.{layer}  F.linear     [M, 512]  bf16
    w.{call}.{layer}       topk_softmax weights [M, 10] f32 (k=10, renormalize), vLLM's kernel
    i.{call}.{layer}       topk_softmax ids     [M, 10] i32
"""
import argparse
import os
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "target", "qwen4exp_moe_window")
SNAP = ("/home/z/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
        "fc694b54fb0174e0913e6adf86691ef85a4ead47-fp8hybrid")
WORDS = ("The bandwidth of a decode step is the only number that matters here. "
         "Every token read is a byte moved, and the arithmetic is idle waiting for it. ")
RUNS = {1: [4, 40], 3: [4, 4, 4], 4: [4, 4, 4, 4]}


def prompt(size):
    return f"Session {uuid.uuid4()}. " + WORDS * size + "\n\nSummarize in one sentence."


def install(model):
    """Forward hooks on every layer's mlp.gate; calls are numbered per layer."""
    import torch
    store = {}
    count = {}

    def hook(layer):
        def fn(mod, inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            n = count.get(layer, 0)
            count[layer] = n + 1
            store[f"x.{n}.{layer}"] = inputs[0].detach().to("cpu", copy=True).contiguous()
            store[f"logits.{n}.{layer}"] = logits.detach().to("cpu", copy=True).contiguous()
        return fn

    handles = []
    for name, mod in model.named_modules():
        if name.endswith("mlp.gate") and ".layers." in name:
            layer = int(name.split(".layers.")[1].split(".")[0])
            handles.append(mod.register_forward_hook(hook(layer)))
    model._hanzo_dump = (store, count, handles)
    return len(handles)


def take(model):
    store, count, handles = model._hanzo_dump
    out = dict(store)
    store.clear()
    count.clear()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--concurrency", type=int, nargs="*", default=[1, 3, 4])
    args = ap.parse_args()
    import torch
    from safetensors.torch import save_file
    from vllm import LLM, SamplingParams
    torch.ops.load_library(os.path.join(os.path.dirname(__import__("vllm").__file__),
                                        "_moe_C_stable_libtorch.abi3.so"))
    os.makedirs(OUT, exist_ok=True)
    # W8's bar config: no speculation, max-num-seqs 8, vLLM's 0.78 memory share.
    llm = LLM(model=SNAP, enforce_eager=True, max_num_seqs=8, gpu_memory_utilization=0.78,
              max_model_len=16384)
    hooked = llm.apply_model(install)
    print(f"hooked {hooked} gates")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    for c in args.concurrency:
        llm.generate([prompt(s) for s in RUNS[c]], sp)
        dump = llm.apply_model(take)[0]
        for key in [k for k in dump if k.startswith("logits.")]:
            _, n, layer = key.split(".")
            logits = dump[key].cuda()
            M = logits.shape[0]
            w = torch.empty(M, 10, device="cuda")
            i = torch.empty(M, 10, device="cuda", dtype=torch.int32)
            s = torch.empty_like(i)
            torch.ops._moe_C.topk_softmax(w, i, s, logits, True, None, None)
            dump[f"w.{n}.{layer}"] = w.cpu()
            dump[f"i.{n}.{layer}"] = i.cpu()
        path = os.path.join(OUT, f"dump_c{c}.safetensors")
        save_file(dump, path, metadata={"concurrency": str(c), "max_tokens": str(args.max_tokens)})
        calls = len({k.split(".")[1] for k in dump if k.startswith("x.")})
        print(f"c={c}: {calls} forwards x {hooked} layers -> {path} ({os.path.getsize(path) / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
