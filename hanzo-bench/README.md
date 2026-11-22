# `hanzo-bench`

> **Deprecated:** The standalone `hanzo-bench` package is deprecated. Use `hanzo bench` instead for the same functionality.
>
> **Migration:**
> ```bash
> # Old
> cargo run --release --features cuda --package hanzo-bench -- plain -m model-id
>
> # New
> hanzo bench -m model-id
> ```

This is our official benchmarking application, which allows you to collect structured information about the speed of `hanzo`.

> [!NOTE]
> You should replace `--features ...` with one of the features specified [here](../README.md#supported-accelerators), or remove it for pure CPU inference.

To run: `cargo run --release --features ... --package hanzo-bench`

```bash
Fast and easy LLM serving.

Usage: hanzo-bench [OPTIONS] <COMMAND>

Commands:
  plain        Select a plain model
  x-lora       Select an X-LoRA architecture
  lora         Select a LoRA architecture
  gguf         Select a GGUF model
  x-lora-gguf  Select a GGUF model with X-LoRA
  lora-gguf    Select a GGUF model with LoRA
  ggml         Select a GGML model
  x-lora-ggml  Select a GGML model with X-LoRA
  lora-ggml    Select a GGML model with LoRA
  help         Print this message or the help of the given subcommand(s)

Options:
  -p, --n-prompt <N_PROMPT>
          Number of prompt tokens to run [default: 512]
  -g, --n-gen <N_GEN>
          Number of generations tokens to run [default: 128]
  -c, --concurrency <CONCURRENCY>
          Number of concurrent requests to run. Default is 1
  -r, --repetitions <REPETITIONS>
          Number of times to repeat each test [default: 5]
  -n, --num-device-layers <NUM_DEVICE_LAYERS>
          Number of device layers to load and run on the device. All others will be on the CPU
  -h, --help
          Print help
  -V, --version
          Print version
```

## Unified Hanzo Research WAL & Ledger Integration

`hanzo-bench` (`board.rs` and `bin/qmm.rs`) records structured `ExperimentRecordV2` entries
via `hanzo-research` (`LocalResearchLedger`, WAL-first under `HANZO_RESEARCH_LEDGER` /
`~/.cache/hanzo/research/shared`) alongside the JSON/markdown benchmark board:

```bash
cargo run --release --features cuda -p hanzo-bench --bin qmm -- \
  --m 128 --reps 12 --quant q4k --baseline-quant q8_0 \
  --research-dir "${HANZO_RESEARCH_LEDGER:-$HOME/.cache/hanzo/research/shared}"
```

Pass `--sync-cloud` (with `HANZO_API_KEY` set) to also upload and SHA-256 read-back verify
records against `https://api.hanzo.ai`; offline runs persist `pending_sync` receipts that
can later be replayed with `hanzo-research sync-wal`.