---
title: Python API
description: "The hanzo Python package."
sidebar:
  order: 6
---

The `hanzo` Python package exposes the same engine that powers the `hanzo` CLI.

## Install

One wheel per accelerator. All wheels expose the same `hanzo` module.

| Accelerator | Package |
| --- | --- |
| CPU (or Intel CPU with MKL) | `pip install hanzo` |
| NVIDIA GPU | `pip install hanzo-cuda` |
| Apple Silicon | `pip install hanzo-metal` |
| Intel MKL (pinned) | `pip install hanzo-mkl` |
| macOS Accelerate | `pip install hanzo-accelerate` |

## Pages

| Page | Covers |
| --- | --- |
| [Runner](/hanzo/reference/python/runner/) | The main entry point. Load a model and send requests. |
| [Which](/hanzo/reference/python/which/) | Variants that select which kind of model to load. |
| [Requests](/hanzo/reference/python/requests/) | Request dataclasses passed to Runner methods. |
| [Responses](/hanzo/reference/python/responses/) | Response and streaming types returned by the engine. |
| [Enums](/hanzo/reference/python/enums/) | Architecture, dtype, and option enums. |
| [Search](/hanzo/reference/python/search/) | Types for web-search tool configuration. |
| [AnyMoE](/hanzo/reference/python/anymoe/) | AnyMoE expert and config types. |
| [Code execution](/hanzo/reference/python/code-execution/) | Configuration for the built-in Python code executor. |
| [Agent approvals](/hanzo/reference/python/agent-approvals/) | Request and decision types for agent action approval callbacks. |
| [Files](/hanzo/reference/python/files/) | First-class output files surfaced from agentic runs. |
| [MCP](/hanzo/reference/python/mcp/) | MCP client configuration types. |
| [Auto-mapping](/hanzo/reference/python/automap/) | Hints for automatic device mapping. |

See [Tutorial 3](/hanzo/tutorials/03-python-sdk/) for a walkthrough and the [Python guides](/hanzo/guides/python/) for task-oriented recipes.

---

<small>Generated from [`hanzo-pyo3/hanzo.pyi`](https://github.com/hanzoai/engine/blob/master/hanzo-pyo3/hanzo.pyi).</small>
