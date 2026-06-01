---
title: Reference
description: Exhaustive lookup pages for flags, endpoints, schemas, and APIs.
---

Reference pages are short, complete, and lookup-oriented. For narrative or motivation, see the [Guides](/hanzo/guides/) or [Explanation](/hanzo/explanation/) sections.

## Contents

**CLI.** Every subcommand and flag of the `hanzo` binary: `run`, `serve`, `bench`, `tune`, `login`, `from-config`, and the rest.

**TOML configuration.** The schema for the config file `hanzo from-config` reads.

**HTTP API.** Endpoint-by-endpoint server documentation, with request and response schemas.

**OpenAI compatibility.** Which parts of OpenAI's Chat Completions and Responses surface are implemented, and which are not.

**[Python API](/hanzo/reference/python/).** The public surface of the `hanzo` Python package, generated from the type stub: `Runner`, `Which`, request and response types.

**Rust API.** Canonical reference at [docs.rs/hanzo](https://docs.rs/hanzo).

**MCP configuration schema.** The JSON schema for MCP client configuration files.

**Supported models.** Every supported architecture, the modalities each accepts, and the compatible quantization methods.

**Model notes.** A short FAQ for models with non-standard behavior.

**Cargo features.** Build-from-source feature flags.

**Environment variables.** Every env var the binary or its build scripts read.

**UQFF format.** The on-disk binary layout of the UQFF quantization format.

**Quantization types.** Bit counts, hardware requirements, and relative quality per supported method.

**Troubleshooting.** Symptom-to-cause index for common errors.
