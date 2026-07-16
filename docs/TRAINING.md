# Training (LoRA fine-tuning)

The engine trains natively in Rust. `hanzo-train` implements the four
Tinker-shaped primitives on the engine's ML stack (hanzo-ml autograd +
hanzo-nn AdamW), and the same loop is exposed three ways — CLI, HTTP, and the
Python SDK. The base model stays frozen; LoRA factors on the attention + MLP
projections are the trainable parameters. Saved adapters are standard PEFT
(`adapter_model.safetensors` + `adapter_config.json`) and load straight back
into the engine for inference.

Supported today: Llama / Qwen2-family decoders (SmolLM2, Qwen2.5, Llama),
CPU F32. The primitives:

| Primitive | What it does |
| --- | --- |
| `create_lora_training_client` | load base frozen, inject LoRA `Var`s |
| `forward_backward` | forward with grad → masked next-token CE → backward (grads accumulate) |
| `optim_step` | apply AdamW to the accumulated gradients |
| `sample` | decode from the current base+LoRA weights |
| `save_weights_and_get_sampling_client` | write the PEFT adapter for inference |

## CLI

Datasets are JSONL, one `{"prompt": ..., "completion": ...}` per line. The
loss mask covers exactly the completion (+ EOS), never the prompt.

```bash
hanzo train -m HuggingFaceTB/SmolLM2-135M --data sft.jsonl \
    --lora-rank 16 --lr 1e-4 --steps 100 --out ./adapter \
    --sample-prompt "2+2="
```

`--lora-alpha` defaults to `2 * rank`. The run logs per-step loss and prints
the adapter path, trainable-parameter count, and final loss.

## HTTP API

`hanzo serve` exposes the same primitives under `/v1/training` — the server
that does inference also trains. Model load is async: create returns
immediately with `status: "loading"`; poll the client until `ready`.

| Route | Method | Purpose |
| --- | --- | --- |
| `/v1/training/clients` | POST | create: `{"base_model", "lora_config": {"rank", "alpha", "target_modules"}}` |
| `/v1/training/clients` | GET | list clients |
| `/v1/training/clients/{id}` | GET | status, counters, `loss_history` |
| `/v1/training/clients/{id}` | DELETE | drop the client, free its memory |
| `/v1/training/clients/{id}/forward_backward` | POST | `{"data": [...]}` → `{"loss", "num_tokens", "metrics"}` |
| `/v1/training/clients/{id}/optim_step` | POST | `{"adam_params": {"lr", ...}}` |
| `/v1/training/clients/{id}/sample` | POST | `{"prompt"` or `"tokens", "sampling_params", "num_samples"}` |
| `/v1/training/clients/{id}/save_weights` | POST | `{"name", "dir"?}` → `{"path", "format": "peft"}` |

`data` entries are either raw text — `{"prompt": "...", "completion": "..."}`,
tokenized server-side with the client's tokenizer — or a pre-tokenized datum:
`{"model_input": {"tokens": [...]}, "target_tokens": [...], "weights": [...]}`
(`weights` is the per-position loss mask). Errors: `400` bad input, `404`
unknown client, `409` while loading or after a failed load.

The routes are in the OpenAPI doc at `/docs` on a running server.

## Python SDK

`pip install hanzo-train` — the Tinker-shaped client over the HTTP API:

```python
from hanzo_train import ServiceClient, LoraConfig, AdamParams, SamplingParams

sc = ServiceClient(base_url="http://localhost:1234")
tc = sc.create_lora_training_client("HuggingFaceTB/SmolLM2-135M",
                                    lora_config=LoraConfig(rank=16))
for batch in batches:  # [{"prompt": ..., "completion": ...}, ...]
    out = tc.forward_backward(batch).result()
    tc.optim_step(AdamParams(lr=1e-4)).result()
print(tc.sample(prompt="2+2=", sampling_params=SamplingParams(max_tokens=8),
                num_samples=1).result().sequences[0].text)
adapter = tc.save_weights_and_get_sampling_client(name="my-adapter").result().path
```

## Rust

Use the crate directly for in-process training — `hanzo_train::run_sft` is
the one-shot loop behind `hanzo train`, and `hanzo_train::TrainingClient`
is the step-by-step API (see the crate docs).

## Using the adapter

The saved directory is a normal PEFT adapter — pass it to the engine's LoRA
inference loading (see [Adapter Models](ADAPTER_MODELS.md)) to serve the
fine-tuned model.
