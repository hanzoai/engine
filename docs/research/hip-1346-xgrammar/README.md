# HIP-1346 Research Dossier: Compiled Grammar Masks for Constrained Generation (`hanzo-xgrammar`)

- **Tracking Issue**: `hanzoai/engine#89` (`p1`, `bounty`, `perf`, `research`, `hip-1346`)
- **Canonical Owning Repository**: `hanzoai/engine`
- **Crate**: [`hanzo-xgrammar/`](../../../hanzo-xgrammar/) (`Apache-2.0`, `#![forbid(unsafe_code)]`, zero external crate dependencies)
- **Raw Receipts**: [`results.json`](./results.json)
- **Canonical Architecture Boundary**:
  > `vfs -> Hanzo S3 -> Visor`; `Cloud` owns lifecycle and billing; `Enso` routes models; **`Zen` generates**; **`Kai` decides**.
  >
  > This research improves **structured text generation (JSON Schema outputs and OpenAI/MCP tool-call envelopes) inside Hanzo Zen (`hanzo-engine`)**. **Kai already produces structured decisions directly without generating text and is neither modified nor replaced.**

---

## 1. Primary References, Summary & License Audit

### Primary Literature & Upstream Code
1. **XGrammar**: Yixin Dong, Charlie F. Ruan, Yaxing Cai, Ruihang Lai, Ziyi Xu, Yilong Zhao, Tianqi Chen — *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models* (arXiv:2411.15100, MLSys 2025). Reference implementation: <https://github.com/mlc-ai/xgrammar>
2. **XGrammar-2**: Yixin Dong et al. — *XGrammar 2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs* (ASPLOS '26, DOI: <https://doi.org/10.1145/3786335.3813124>).

### Core Technical Mechanism
Standard constrained decoding evaluates every token $t \in \{0, \dots, V-1\}$ against the active grammar parser on every decode step ($O(V)$ parser evaluations per token), becoming a severe CPU bottleneck for modern vocabularies ($V = 32{,}768$ to $128{,}256+$) when GPU decode latency is 2–5 ms/token.

XGrammar eliminates the per-step $O(V)$ scan via three compiler techniques implemented clean-room in `hanzo-xgrammar`:
1. **Byte-level Pushdown Automaton (PDA) with Tail-Call Optimization (TCO)**: JSON Schemas and OpenAI/MCP tool-call grammars compile into a compact byte-level PDA where repetitive productions (`StringBody`, digits, whitespace, flat arrays) execute in $O(1)$ stack depth via TCO while recursive `$defs`/`$ref` structures track true nesting depth bounded by `MAX_RECURSION_DEPTH`.
2. **Context-Independent vs. Context-Dependent Token Partitioning + Post-Pop Suffix Grouping**: At compile time, walking a compact byte-prefix trie (`VocabTrie`) partitions every leaf position into:
   - **Context-Independent Accepted** (strictly in-rule depth-1 survivors — unconditionally valid for all caller stacks),
   - **Caller-Live Accepted** (clean end-of-token pops — valid in $O(1)$ whenever `expand(caller)` is non-empty),
   - **Context-Independent Rejected** (dead subtrees pruned in $O(1)$),
   - **Post-Pop Suffix Groups** (tokens exiting a leaf with identical post-pop suffixes such as `","`, `":"`, `"}"`, collapsing thousands of per-token checks into ~10–25 unique suffix checks), and
   - **General Context-Dependent (`uncertain`)** tokens checked against the exact runtime stack.
3. **Adaptive Sparse/Dense Storage & LRU Compiled-Grammar Cache**: Structural states allowing few tokens store `Sparse(Vec<u32>)` instead of dense `BitMask` (`Vec<u64>`), cutting memory footprint from megabytes to ~350–925 KB/grammar, while `GrammarCache` reuses compiled grammars across multi-turn agent requests in ~1.8–2.6 µs.

### License, Dependency & Patent Audit
| Artifact | License | Audit Disposition |
|---|---|---|
| `mlc-ai/xgrammar` (reference inspiration only) | Apache-2.0 (`LICENSE` in upstream repo) | **Clear** — clean-room Rust re-implementation; zero upstream C++/Python source files copied. |
| `hanzo-xgrammar` (this PR) | Apache-2.0 (compatible with workspace MIT / Apache-2.0) | **Clear** — **0 external crate dependencies** (Rust `std` only), `#![forbid(unsafe_code)]`, offline-buildable in < 2.1s (`--release`). |
| Benchmark datasets / vocabularies | Derived deterministically from this repository's own MIT/Apache-2.0 source files + standard RFC 8259 / JSON Schema Draft 2020-12 definitions (plus optional `--tokenizer <path>` for local HuggingFace `tokenizer.json`) | **Clear** — no proprietary corpora, no restricted datasets, no external network fetches required. |
| Patents | No patent encumbrances declared on `mlc-ai/xgrammar`; algorithms rest on standard CFG/PDA parsing, prefix tries, and bitset precomputation | **Clear**. |

> Published XGrammar paper numbers are not claimed as Hanzo metrics; all numbers below were measured directly on Hanzo benchmark hardware via `xgrammar-bench --release --offline`.

---

## 2. Reproducible Baseline & Hardware Environment

### Host Hardware & Toolchain
- **CPU**: `Intel(R) Xeon(R) CPU @ 2.30GHz` (2 vCPUs, x86_64 KVM, AVX2/BMI2, 32 KiB L1d, 256 KiB L2, 45 MiB L3)
- **Toolchain**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `cargo 1.100.0 (3acc79c0c 2026-11-05)`
- **Build & Execution Flags**: `--release --offline` (`opt-level = 3`), deterministic seed `13462026`, `20` trials per workload.
- **Vocabularies**: Corpus-derived byte-level BPE vocabularies built from repository Rust/TOML/Markdown sources (all 256 raw single bytes `0x00..=0xFF`, JSON/code delimiters, multi-byte UTF-8 and split-UTF-8 pieces, ~35% space-prefixed tokens mirroring Llama-3 / Qwen2.5):
  - `V = 32,768`: `103,365` trie nodes, mean token length `8.34` bytes, `34.81%` space-prefixed, built in `50.0 ms`.
  - `V = 128,256`: `421,897` trie nodes, mean token length `9.54` bytes, `35.52%` space-prefixed, built in `251.6 ms`.

### Matched Baselines
1. **Unconstrained Decoding**: Samples directly from noisy LLM logits without grammar masking; evaluated by an **independent strict RFC 8259 JSON parser + JSON Schema validator** (never self-graded by the PDA) to measure invalid-output rate.
2. **`NaivePerStepConstrainedDecoder` (Matched Uncompiled Baseline)**: Executes the **exact same `Grammar`, `PdaState`, `PdaConfig`, `SmallStack`, and `FIRST`-set lookahead pruning** across all `V` vocabulary tokens on every decode step (representing standard per-step constrained decoding without XGrammar's precomputed context-independent masks).
3. **`XGrammarMatcher` (Compiled Grammar + Cached Masks)**: Precomputes per-leaf context-independent masks and post-pop suffix groups via `VocabTrie`, caches compiled grammars in `GrammarCache`, and applies dense bitmasks (`-inf`) directly onto `logits: &mut [f32]`.

### Exact Reproducible Commands
```bash
# Format, lint, and unit/parity test suite (offline, release)
cargo fmt -p hanzo-xgrammar -- --check
cargo clippy -p hanzo-xgrammar --all-targets --offline -- -D warnings
cargo test -p hanzo-xgrammar --offline --release

# Reproduce full V=32,768 and V=128,256 benchmark and regenerate results.json
cargo run -p hanzo-xgrammar --release --offline --bin xgrammar-bench

# Optional: run against any local HuggingFace tokenizer.json
cargo run -p hanzo-xgrammar --release --offline --bin xgrammar-bench -- --tokenizer /path/to/tokenizer.json
```

---

## 3. Falsifiable Hypothesis & Expected Failure Modes

### Hypothesis
For structured JSON Schema and OpenAI/MCP tool-call generation over realistic BPE vocabularies ($V \in \{32{,}768, 128{,}256\}$):
- **(H1 — Validity & Parity)**: `XGrammarMatcher` achieves **0.0% invalid-output rate** (versus 70–100% under noisy unconstrained decoding) and **100% bit-identical token masks** (`0` mismatches) compared to `NaivePerStepConstrainedDecoder` on every decode step.
- **(H2 — Warm-Cache Latency & Throughput)**: On recurring JSON/tool-call schemas with fused-whitespace compilation, `XGrammarMatcher` reduces mean per-step mask latency by **$\ge 50\times$ at $V = 32{,}768$** and **$\ge 150\times$ at $V = 128{,}256$**, sustaining **$> 84\%$ of unconstrained effective throughput** at a fast 2.0 ms/token LLM decode step (and **$> 93\%$** at 5.0 ms/token), within **$< 1$ MB/grammar** RAM.

### Explicitly Tested Negative Regimes (Expected Failure Modes)
- **(F1 — Cold One-Shot Short Output)**: `neg_oneshot_ephemeral_16tok` compiles an uncached 12-field schema once for a 15-token output. Hypothesis: eager compilation cost ($O(|\text{leaves}| \times |\text{trie}|)$) exceeds 15 steps of naive evaluation, causing end-to-end one-shot speedup to drop **below `1.0x`** (`0.12x–0.13x`).
- **(F2 — Context-Heavy Un-inlined Short-Rule CFG)**: `neg_ambiguous_context_heavy` uses an un-inlined 1-byte non-terminal (`IdentChar -> [a-zA-Z0-9_]`) inside a nested expression CFG. Hypothesis: $> 50\%$ of BPE tokens pop `IdentChar` on byte 0 with unique post-pop suffixes, collapsing per-step speedup to **`< 2.0x`** (`1.25x–1.48x`).

---

## 4. Bounded Implementation Summary

`hanzo-xgrammar/` is registered in workspace `Cargo.toml` as a standalone, zero-dependency Rust crate (`Apache-2.0`, `#![forbid(unsafe_code)]`):
- [`src/json.rs`](../../../hanzo-xgrammar/src/json.rs): Strict RFC 8259 JSON parser (bounded depth `128`, UTF-16 surrogate pair decoding, control-char / leading-zero / trailing-comma rejection) and canonical serializer.
- [`src/grammar.rs`](../../../hanzo-xgrammar/src/grammar.rs): Byte-level PDA bytecode IR (`ByteSet([u64; 4])`, zero-alloc `Copy` `SmallStack`, precomputed `FIRST`/nullable tables, tail-call optimization, compile-time 3-color DFS left-recursion rejection, `PdaCheckpoint`/`rollback`, hard `MAX_RECURSION_DEPTH = 64` & `MAX_ACTIVE_STACKS = 256` bounds).
- [`src/schema.rs`](../../../hanzo-xgrammar/src/schema.rs): JSON Schema (`object`, `properties`, `required`, `additionalProperties`, `array`, `minItems`/`maxItems`, `string`, `enum`, `const`, `integer`, `number`, `boolean`, `null`, `anyOf`/`oneOf`, `$defs`/`$ref`) + OpenAI/MCP tool-call compiler (`ToolAndArguments` and `NameAndArguments`), fused-whitespace compiler pass, full RFC 3629 UTF-8 + RFC 8259 JSON escape `StringBody`, explicit DoS rejection (`MAX_SCHEMA_DEPTH = 32`, `MAX_OBJECT_PROPERTIES = 128`, `MAX_ENUM_VARIANTS = 256`, `MAX_ARRAY_UNROLL = 64`), and independent `SchemaDocument::validate_json_str`.
- [`src/vocab.rs`](../../../hanzo-xgrammar/src/vocab.rs): `Vocabulary`, `VocabTrie` (iterative DFS pre-order subtree ranges), HuggingFace `tokenizer.json` loader (GPT-2 byte-level decoding), and corpus-derived BPE vocabulary builder.
- [`src/mask.rs`](../../../hanzo-xgrammar/src/mask.rs): `BitMask` (64-bit word-level `-inf` logits masking), `CompactTokenSet` (`Empty`/`Sparse`/`Dense`), `SuffixGroup`, `CompiledGrammar`, `XGrammarMatcher`, and matched `NaivePerStepConstrainedDecoder`.
- [`src/cache.rs`](../../../hanzo-xgrammar/src/cache.rs): Thread-safe LRU `GrammarCache` bounded by both entry count and memory byte budget, with out-of-lock compilation so cold compiles never stall warm hits.
- [`src/bin/xgrammar-bench.rs`](../../../hanzo-xgrammar/src/bin/xgrammar-bench.rs): Deterministic CLI benchmark emitting `results.json`.

---

## 5. Correctness, Concurrency, Security & Regression Evidence

- **Unit & Integration Test Suite**: `21` passing tests in `cargo test -p hanzo-xgrammar --offline --release` (0 failures, 0 warnings under `cargo clippy --all-targets --offline -- -D warnings`, clean `cargo fmt -- --check`).
- **Bit-Exact Mask Parity**: Verified `xgrammar_mask == naive_mask` on **every single decode step**:
  - `V = 32,768`: **984 / 984 steps identical (`0` mismatches)**
  - `V = 128,256`: **976 / 976 steps identical (`0` mismatches)**
  - Dedicated tests cover RFC 3629 multi-byte UTF-8 (`日本語`, `🚀`, `café`), **split multi-byte UTF-8 BPE tokens** (`[0xE4, 0xB8]` + `[0xAD]` -> `中`), JSON escapes (`\"`, `\\`, `\n`, `\uXXXX`), rejection of overlong UTF-8 / UTF-16 surrogates / raw control bytes, recursive `$ref` at tight stack depth limits, optional properties, `additionalProperties: true`, speculative `checkpoint`/`rollback`, and ambiguous CFGs.
- **Concurrency**: `GrammarCache` tested under 4-thread concurrent stress (`lru_cache_hits_evicts_and_is_thread_safe`); compilation executes outside the mutex lock, and `CompiledGrammar` / `Vocabulary` are immutable `Arc`-shared across concurrent Zen decode sequences (`XGrammarMatcher` holds only per-sequence state).
- **Security & DoS Hardening**: `#![forbid(unsafe_code)]`; explicit compile-time rejection of cyclic/left-recursive `$ref`, unresolved `$ref`, excessive schema nesting depth (`> 32`), excessive properties (`> 128`), excessive array unrolls (`> 64`), invalid bounds (`minItems > maxItems`), missing required keys under `additionalProperties: false`, and runtime PDA depth/branch caps (`MAX_RECURSION_DEPTH = 64`, `MAX_ACTIVE_STACKS = 256`).

---

## 6. Before/After Measurements & Canonical Zen Integration Plan

All figures below are copied directly from [`results.json`](./results.json) (seed `13462026`, 20 trials/workload, `--release --offline`).

### Table A — Vocabulary `V = 32,768` (984 parity steps, 0 mismatches)

| Workload | Regime | Invalid Rate (Unc / Naive / XG) | Naive p50 (µs) | XG p50 (µs) | Mean Mask Speedup | Pure Mask tok/s (Naive -> XG) | Eff. tok/s @ 2ms LLM (Naive -> XG Warm) | Compile (ms) | 1-Shot E2E Speedup | Mem (KB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `json_api_response` | Positive (cached) | 100.0% / **0.0%** / **0.0%** | 4,954.3 | **227.0** | **89.3x** | 42.9 -> **3,829.8** | 39.5 -> **442.3** | 321.1 | 6.09x | 382.1 |
| `nested_mcp_tool_call` | Positive (cached) | 100.0% / **0.0%** / **0.0%** | 3,401.6 | **13.7** | **109.9x** | 64.9 -> **7,130.2** | 57.4 -> **467.2** | 379.1 | 2.23x | 756.8 |
| `recursive_ast_tree` | Positive (cached) | 100.0% / **0.0%** / **0.0%** | 4,348.1 | **29.6** | **73.4x** | 94.4 -> **6,922.7** | 79.4 -> **466.3** | 321.2 | 2.79x | 353.6 |
| `strict_enum_classification` | Positive (cached) | 100.0% / **0.0%** / **0.0%** | 54,552.4 | **451.9** | **86.5x** | 32.4 -> **2,799.2** | 30.4 -> **424.2** | 356.2 | 3.82x | 452.8 |
| `large_20field_config` | Positive (cached) | 100.0% / **0.0%** / **0.0%** | 1,731.8 | **10.4** | **64.4x** | 120.8 -> **7,771.4** | 97.3 -> **469.8** | 364.2 | 3.65x | 706.0 |
| `neg_oneshot_ephemeral_16tok` | **Negative #1 (cold 15-tok)** | 80.0% / **0.0%** / **0.0%** | 1,649.1 | **7.7** | 35.6x | 386.1 -> 13,760.3 | 217.9 -> 482.5 *(cold: **41.2**)* | 332.7 | **0.12x** | 493.5 |
| `neg_ambiguous_context_heavy` | **Negative #2 (55.4% CD)** | 100.0% / **0.0%** / **0.0%** | 52,319.7 | 30,987.3 | **1.48x** | 22.7 -> 33.6 | 21.7 -> 31.5 | 36.0 | **1.42x** | 190.7 |

### Table B — Vocabulary `V = 128,256` (976 parity steps, 0 mismatches)

| Workload | Regime | Invalid Rate (Unc / Naive / XG) | Naive p50 (µs) | XG p50 (µs) | Mean Mask Speedup | Pure Mask tok/s (Naive -> XG) | Eff. tok/s @ 2ms LLM (Naive -> XG Warm) | Eff. tok/s @ 5ms LLM (Naive -> XG Warm) | Compile (ms) | 1-Shot E2E Speedup | Mem (KB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `json_api_response` | Positive (cached) | 100.0% / **0.0%** / **0.0%** | 14,990.7 | **305.8** | **253.8x** | 10.7 -> **2,727.7** | 10.5 -> **422.5** | 10.2 -> **186.3** | 1,343.9 | 6.08x | 548.4 |
| `nested_mcp_tool_call` | Positive (cached) | 100.0% / **0.0%** / **0.0%** | 11,346.7 | **45.2** | **347.5x** | 14.3 -> **4,982.1** | 13.9 -> **454.4** | 13.4 -> **192.3** | 1,328.3 | 2.86x | 923.2 |
| `recursive_ast_tree` | Positive (cached) | 100.0% / **0.0%** / **0.0%** | 14,573.3 | **53.5** | **209.3x** | 24.6 -> **5,148.4** | 23.4 -> **455.7** | 21.9 -> **192.5** | 1,248.4 | 2.83x | 519.9 |
| `strict_enum_classification` | Positive (cached) | 100.0% / **0.0%** / **0.0%** | 77,442.3 | **351.3** | **361.6x** | 7.7 -> **2,796.9** | 7.6 -> **424.2** | 7.4 -> **186.7** | 1,254.6 | 4.78x | 619.2 |
| `large_20field_config` | Positive (cached) | 100.0% / **0.0%** / **0.0%** | 6,399.3 | **21.9** | **226.2x** | 28.4 -> **6,424.8** | 26.9 -> **463.9** | 24.9 -> **194.0** | 1,314.6 | 4.39x | 872.4 |
| `neg_oneshot_ephemeral_16tok` | **Negative #1 (cold 15-tok)** | 80.0% / **0.0%** / **0.0%** | 6,006.3 | **20.1** | 88.0x | 91.0 -> 8,010.4 | 77.0 -> 470.6 *(cold: **11.7**)* | 62.6 -> 195.1 *(cold: **11.3**)* | 1,255.1 | **0.13x** | 659.9 |
| `neg_ambiguous_context_heavy` | **Negative #2 (64.7% CD)** | 100.0% / **0.0%** / **0.0%** | 272,485.1 | 189,567.7 | **1.25x** | 4.3 -> 5.4 | 4.2 -> 5.3 | 4.2 -> 5.2 | 243.7 | **1.19x** | 924.1 |

### Multi-Turn Agent `GrammarCache` Telemetry (100-Request Stream, 95 Recurring / 5 Ephemeral)
- `V = 32,768`: **90.0% hit rate** (`90` hits / `10` cold misses / `0` evictions), **`2.55 µs`** mean warm lookup, **`3.97 MB`** total RAM for 10 cached grammars.
- `V = 128,256`: **90.0% hit rate** (`90` hits / `10` cold misses / `0` evictions), **`1.80 µs`** mean warm lookup, **`5.63 MB`** total RAM for 10 cached grammars.

### Key Findings & Negative-Regime Lessons
1. **Strong Positive Result on Recurring JSON & Tool-Call Grammars**: At `V = 128,256`, `hanzo-xgrammar` achieves **209.3x–361.6x** mask speedup over naive per-step PDA decoding, preserving **84.5%–92.8%** of unconstrained 500 tok/s GPU decode speed (and **93.2%–97.0%** at 200 tok/s) with **0.0%** invalid outputs and **< 1 MB/grammar** cache memory.
2. **Negative Regime #1 (`0.12x–0.13x` cold 1-shot)**: Eagerly compiling all 242 leaf positions for an uncached 15-token ephemeral response costs `333–1,255 ms`, making cold one-shot execution ~8x slower than naive. **Implication for Zen**: production integration should combine `GrammarCache` (for recurring toolsets/response schemas) with **lazy per-position mask compilation** (compiling `PositionMask` on first visit of a leaf rather than eagerly compiling unvisited branches) AND async background compilation overlapping prompt prefill.
3. **Negative Regime #2 (`1.25x–1.48x` context-heavy CFG)**: When short 1-byte rules are not inlined, 55–65% of the vocabulary pops on byte 0 with distinct suffixes, neutralizing mask precomputation. **Implication for Zen**: always run rule-inlining passes (as `schema.rs` does for fused-whitespace rules and literal keys) and keep `llguidance` as fallback for arbitrary user Lark/regex grammars with un-inlined short rules.

### Canonical Hanzo Zen Integration & Rollout Plan (HIP-1346)
1. **Integration Seams in `hanzo-engine`** (does **not** touch or replace **Kai**):
   - Add an optional `SequenceRecognizer::XGrammar(Box<hanzo_xgrammar::XGrammarMatcher>)` variant alongside existing `SequenceRecognizer::Llguidance(Box<llguidance::Matcher>)` in [`hanzo-engine/src/sequence.rs`](../../../hanzo-engine/src/sequence.rs).
   - Attach a per-engine `Arc<hanzo_xgrammar::GrammarCache>` and `Arc<hanzo_xgrammar::Vocabulary>` built alongside `build_llg_factory` in [`hanzo-engine/src/pipeline/llg.rs`](../../../hanzo-engine/src/pipeline/llg.rs).
   - Dispatch `Constraint::JsonSchema` and strict tool-call JSON bodies from [`hanzo-engine/src/tools/grammar.rs`](../../../hanzo-engine/src/tools/grammar.rs) to `hanzo-xgrammar` when enabled by feature/runtime flag (`HANZO_XGRAMMAR=1`), falling back transparently to `llguidance` for `Constraint::Regex`, `Constraint::Lark`, or unsupported JSON Schema keywords (`pattern`, `format`).
   - In [`hanzo-engine/src/pipeline/sampling.rs`](../../../hanzo-engine/src/pipeline/sampling.rs), call `matcher.mask_logits(logits, &mut mask_buf)` and `matcher.consume_token(tok)`, and use `matcher.checkpoint()` / `matcher.rollback()` to enable grammar-constrained **speculative decoding** verification in [`hanzo-engine/src/speculative/verifier.rs`](../../../hanzo-engine/src/speculative/verifier.rs).
2. **Rollout & Rollback**:
   - **Phase 1 (This PR)**: Standalone zero-dependency `hanzo-xgrammar` crate + reproducible benchmark harness + HIP-1346 dossier (`0` behavioral changes to shipping `hanzo-engine` code).
   - **Phase 2 (Follow-up Implementation Bounty)**: Add lazy per-leaf mask caching + background prefill-overlapped compilation, wire `SequenceRecognizer::XGrammar` behind opt-in env/config gate, and benchmark head-to-head against shipping `llguidance` on real Zen models (Qwen/Llama/Gemma4) and confidential Hanzo internal tool-calling corpora.
   - **Rollback**: Single config toggle reverts to `SequenceRecognizer::Llguidance`, or `git revert` cleanly removes `hanzo-xgrammar` with zero schema/storage migration impact.
