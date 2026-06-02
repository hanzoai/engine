---
title: Start here
description: Choose the right entry point for the task you are building.
---

Use this page to pick the first document to read. Most workflows start with auto-detection and add flags only when the model, hardware, or deployment requires them.

## Choose by task

| If you need to... | Start here | Then read |
|---|---|---|
| Chat with a model on one machine | [Your first model](/hanzo/tutorials/01-install-and-run/) | [Pick a quantization method](/hanzo/guides/perf/pick-a-quantization/) |
| Verify install, GPU support, or Hugging Face access | [Your first model](/hanzo/tutorials/01-install-and-run/) | [Troubleshooting](/hanzo/reference/troubleshooting/) |
| Expose an OpenAI-compatible endpoint | [Serve a model as an API](/hanzo/tutorials/02-serve-an-api/) | [Configure the HTTP server](/hanzo/guides/serve/http-server/) |
| Use the built-in browser UI | [Serve a model as an API](/hanzo/tutorials/02-serve-an-api/) | [Use the built-in web UI](/hanzo/guides/serve/with-web-ui/) |
| Call hanzo from Python in-process | [Call a model from Python](/hanzo/tutorials/03-python-sdk/) | [Python API reference](/hanzo/reference/python/) |
| Embed hanzo in Rust | [Call a model from Rust](/hanzo/tutorials/04-rust-sdk/) | [Rust API on docs.rs](https://docs.rs/hanzo) |
| Build a local agent app with tools, code execution, web search, multimodal inputs, or session state | [Build an agent](/hanzo/tutorials/05-build-an-agent/) | [Agentic runtime for apps](/hanzo/guides/agents/agentic-runtime/) |
| Fit a larger model on the same hardware | [Quantize a model](/hanzo/tutorials/06-quantize-a-model/) | [Auto-tune with hanzo tune](/hanzo/guides/perf/auto-tune/) |
| Split a model across GPUs or machines | [Performance](/hanzo/guides/perf/) | [Split a model across multiple GPUs](/hanzo/guides/perf/multi-gpu-tensor-parallel/) |
| Run a server for real traffic | [Run hanzo in Docker](/hanzo/guides/deploy/docker/) | [Production checklist](/hanzo/guides/deploy/production-checklist/) |

## Choose by runtime mode

| Mode | Use when | Entry point |
|---|---|---|
| CLI | You want local interactive use, quick tests, or benchmarking. | `hanzo run`, `hanzo bench`, `hanzo tune` |
| HTTP server | You want OpenAI-compatible clients, a web UI, or a process boundary around inference. | `hanzo serve` |
| Config file | You need repeatable multi-model startup or a deployment config checked into source control. | `hanzo from-config -f config.toml` |
| Diagnostics | You want to check hardware detection, build features, or Hugging Face connectivity. | `hanzo doctor` |
| Python package | You want in-process access from Python without running a server. | `hanzo.Runner` |
| Rust crate | You want inference embedded inside a Rust service. | `hanzo` crate |

## If unsure

Start with [Your first model](/hanzo/tutorials/01-install-and-run/), then [Serve a model as an API](/hanzo/tutorials/02-serve-an-api/). Those two pages exercise the default local and server paths and make later choices easier to evaluate.
