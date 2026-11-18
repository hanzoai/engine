# HIP-1346 Research and Bounded Implementation: Asymmetric 2-Bit and 4-Bit KV Cache Compression (KIVI)

## 1. Primary Reference and License Audit

- Primary paper: Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu, "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024 (`https://arxiv.org/abs/2402.02750`).
- Complementary baseline reference: Coleman Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization", NeurIPS 2024 (`https://arxiv.org/abs/2401.18079`), cited in `hanzo-paged-attn/src/quant.rs`.
- Tracking issue: `hanzoai/engine#50` (HIP-1346).
- License audit: 100% independent clean-room Rust implementation (`MIT OR Apache-2.0`, matching the workspace license) authored directly against `hanzo-paged-attn`'s `u32` word-packing conventions and existing crate dependencies (`half`, `thiserror`). The KIVI authors' reference Python/CUDA repository (`jy-yuan/KIVI`) is MIT-licensed; zero external source files, kernels, calibration datasets, or new Cargo dependencies were imported.

## 2. Upstream Baseline

`hanzo-paged-attn/src/quant.rs` implements symmetric per-token `int8` KV cache quantization (`PagedCacheType::Int8` in `hanzo-engine/src/paged_attention/cache_engine.rs`), packing 4 signed `i8` codes per `u32` word alongside one co-located `f32` scale word (`words_per_token(head_size) = head_size / 4 + 1`). At 8 bits (`255` symmetric levels), scale-relative error is bounded by `1/254` (`~0.00394`), which absorbs persistent key channel outliers within a single per-token dynamic range while halving `FP16` KV memory (`2.0x` vs `FP16` asymptotically, `1.94x` at `head_size = 128`).

Two limitations arise when pushing long-context KV compression (`5K` to `128K` tokens) to 4-bit and 2-bit:

1. **Outlier channel collapse under per-token symmetric quantization**: Transformer key states exhibit fixed feature channels whose magnitudes are `30x-80x` larger than normal semantic channels across all sequence positions. Under uniform symmetric per-token `INT4` (`[-7, 7]`) or `INT2` (`{-3, -1, +1, +3}`), a single `+92.0` outlier channel sets `scale = 13.14` (`INT4`) or `30.67` (`INT2`), quantizing normal semantic channels in `[-2.8, +2.8]` directly to zero and collapsing retrieval fidelity.
2. **Hybrid recurrent Gated DeltaNet (GDN) safety**: Hybrid architectures in `hanzo-engine` (such as Qwen3-Next, Qwen3.5, and Zen) interleave paged attention layers with recurrent GDN state layers. Applying low-bit KV recipes to recurrent GDN state matrices is numerically unsafe because state updates recur multiplicatively at every token step.

## 3. Falsifiable Hypothesis and Failure Modes

- **Falsifiable hypothesis**: Quantizing paged attention **Key** blocks per-channel across completed token groups of `G` (`G = 32, 64, 128`) with co-located `(f16 scale, f16 zero_point)` words, quantizing **Value** blocks per-token, retaining the most recent `R` (`R = 32, 64`) tokens in an unquantized `FP16`/`FP32` sliding residual buffer that flushes atomically on `G`-token boundaries, and executing online page-streaming fused attention (`Q @ K_quant^T` + `softmax @ V_quant`) reduces attention KV memory by `3.54x-3.71x` (`INT4`) and `6.66x-7.31x` (`INT2`) vs `FP16` while preserving `> 0.995` (`INT4`) and `> 0.928` (`INT2`) attention cosine similarity and `100%` (`4/4`) multi-needle retrieval across `5K`, `30K`, and `128K` contexts.
- **Expected failure mode 1 (Uniform symmetric baseline)**: Uniform symmetric per-token `INT4` and `INT2` on identical outlier-heavy key distributions should fail semantic needle retrieval (`0/4` on `INT4` due to zero-clamping of normal channels) and suffer severe cosine collapse on `INT2` (`cosine < 0.05`).
- **Expected failure mode 2 (Recurrent GDN state quantization negative control)**: Applying 2-bit row-wise affine quantization inside a 384-step recurrent DeltaNet state recurrence `S_t = alpha S_{t-1} + beta (v_t - S_{t-1} k_t) k_t^T` compounds rounding error multiplicatively (`relative L2 drift > 20%`, observed `828.4%`), proving that `LayerStateKind::RecurrentGdn` must remain unquantized (`LayerCompressionPlan::UnquantizedRecurrentGdn`).

## 4. Bounded Implementation (`hanzo-paged-attn`)

- `hanzo-paged-attn/src/kivi.rs` (exported via `pub mod kivi;` in `hanzo-paged-attn/src/lib.rs`, leaving `src/quant.rs` and all CUDA/Metal/ROCm/Vulkan backends untouched):
  - `KiviBits::{Int4, Int2}`: Little-endian bit-packing (`8` unsigned 4-bit nibbles per `u32` word for `Int4`, `16` unsigned 2-bit crumbs per `u32` word for `Int2`) compatible with `DType::U32` paged storage across backends lacking sub-byte tensors.
  - `KiviKeyPage`: Per-channel group quantization across `G` tokens (`channel_params[0..head_size]` storing packed `(f16 scale_c, f16 zero_c)` in one `u32` word per channel). `KiviKeyPage::score_query` hoists `sum_c (softmax_scale * q[c] * zero_c)` and `q_scaled[c]` once per `G`-token page and computes logits directly from packed `u32` words without materializing dequantized key matrices.
  - `KiviValuePage`: Per-token affine quantization (whole-token or configurable `value_group_size` subgroup) with `KiviValuePage::accumulate_weighted` hoisting `sum_t (w_t * zero_v[t])` per subgroup and accumulating weighted codes in place.
  - `KiviKvCache`: Multi-head / GQA paged cache with configurable `FP16` or `FP32` sliding residual window (`R = 32/64`), atomic `G`-token flush on both streaming `append_token` and zero-spike prefill `append_batch`, and zero-allocation FlashAttention-style online rescaling `fused_decode_head` / `fused_gqa_decode`.
  - `LayerStateKind::{AttentionKv, RecurrentGdn}`, `resolve_layer_plan`, `KiviKvCache::new_for_layer`, and `project_hybrid_model_memory`: Enforces full-precision retention on recurrent GDN layers (`Err(KiviError::RecurrentLayerExempt)` if quantization is requested on a GDN layer).
- `hanzo-paged-attn/tests/kivi_kv_quant.rs`: End-to-end validation, numerical bound proofs, `5K`/`30K`/`128K` outlier + multi-needle benchmarks, 20-turn coding-agent prefix-reuse loop, `4B`/`9B`/`27B` hybrid KV+GDN memory projections, GDN negative control, and 16-thread concurrency stress test.

## 5. Correctness, Concurrency, Security, and Regression Evidence

- **Exact mathematical error bounds**: `affine_roundtrip_respects_tight_int4_and_int2_bounds` verifies worst-case span-relative error `<= 1/30 + 1e-6` (`Int4`) and `<= 1/6 + 1e-6` (`Int2`) across all supported head dimensions (`64, 80, 96, 112, 128, 192, 256, 512`) and dynamic ranges `1e-6..1e6`.
- **Online fused vs materialized equivalence**: `online_fused_attention_matches_materialized_reference` verifies that zero-allocation page-streaming `fused_decode_head` agrees with two-pass materialized dequantized attention within `1e-5` max absolute difference across multi-page + partial-residual states.
- **Streaming vs batched equivalence**: `append_batch_matches_token_by_token_streaming_exactly` proves bit-identical page counts, residual lengths, and attention outputs between single-token decode appends and batched prefill ingestion.
- **Input sanitization & non-panicking bounds**: `NaN`, `+Inf`, and `-Inf` inputs are clamped cleanly without poisoning `scale`, `zero_point`, or softmax denominators; all invalid dimensions, head ratios, and recurrent-layer requests return typed `Result<_, KiviError>` errors.
- **Multi-thread concurrency**: `multithreaded_concurrent_gqa_decode_is_deterministic` races 16 OS threads behind a `std::sync::Barrier` (12 concurrent GQA readers sharing `Arc<KiviKvCache>` + 4 concurrent cache builders/readers), confirming deterministic bit-exact outputs with zero data races.

## 6. Before and After Measurements, Scorecard, and Integration Recommendation

Reproduce with:

```bash
cargo test -p hanzo-paged-attn --lib
cargo test -p hanzo-paged-attn --test kivi_kv_quant --release -- --nocapture
```

Reference hardware environment recorded for raw baseline and KIVI oracle measurements:
- CPU: `Intel(R) Xeon(R) CPU E5-1650 v4 @ 3.60GHz` (`x86_64`, single-thread deterministic execution)
- OS / Runtime: `Linux x86_64 (glibc 2.39)`, IEEE-754 `f16`/`f32` exact reference harness (`head_size = 128`, GQA `4:1`, 4 persistent outlier channels `[+68, -54, +92, -76]`, 4 orthogonal Walsh-Hadamard needles planted at `10%`, `50%`, `85%`, and `L - 12`)

### 6.1 Long-Context KV Compression, Fidelity, Retrieval, and Latency Scorecard (`5K / 30K / 128K`)

| Context | Format | Resident KV Bytes | Ratio vs FP16 (vs FP32) | Output Cosine | Output MSE | Needle Retrieval (Val Cosine) | Cold Prefill Quant (`ms`) | Decode `p50 / p95` (`us`) |
|---|---|---|---|---|---|---|---|---|
| `5K` (`5,120`) | `FP16` Baseline | `5,242,880` | `1.00x` (`2.00x`) | `1.000000` | `0.000e+00` | `4/4` (`1.0000`) | `0.00` | `5,878.4 / --` |
| `5K` (`5,120`) | Uniform Sym `INT4` | `1,392,640` | `3.76x` (`7.53x`) | `0.081013` | `4.875e-02` | `0/4` (collapsed) | `--` | `--` |
| `5K` (`5,120`) | Uniform Sym `INT2` | `737,280` | `7.11x` (`14.22x`) | `-0.019279` | `8.427e-01` | `4/4` (noisy) | `--` | `--` |
| `5K` (`5,120`) | **KIVI `INT4` (`G=64, R=64`)** | `1,481,216` | **`3.54x`** (`7.08x`) | **`0.997848`** | **`2.122e-04`** | **`4/4` (`0.9984`)** | `88.79` | `6,671.9 / 12,732.7` |
| `5K` (`5,120`) | **KIVI `INT2` (`G=128, R=32`)** | `737,280` | **`7.11x`** (`14.22x`) | **`0.955679`** | **`4.640e-03`** | **`4/4` (`0.9565`)** | `31.61` | `3,451.0 / 6,942.2` |
| `30K` (`30,720`) | `FP16` Baseline | `31,457,280` | `1.00x` (`2.00x`) | `1.000000` | `0.000e+00` | `4/4` (`1.0000`) | `0.00` | `40,944.2 / --` |
| `30K` (`30,720`) | Uniform Sym `INT4` | `8,355,840` | `3.76x` (`7.53x`) | `0.112972` | `3.388e-03` | `0/4` (collapsed) | `--` | `--` |
| `30K` (`30,720`) | Uniform Sym `INT2` | `4,423,680` | `7.11x` (`14.22x`) | `-0.010172` | `8.608e-01` | `4/4` (noisy) | `--` | `--` |
| `30K` (`30,720`) | **KIVI `INT4` (`G=64, R=64`)** | `8,649,216` | **`3.64x`** (`7.27x`) | **`0.997856`** | **`1.479e-05`** | **`4/4` (`0.9984`)** | `530.16` | `40,058.7 / 55,625.8` |
| `30K` (`30,720`) | **KIVI `INT2` (`G=64, R=64`)** | `4,725,248` | **`6.66x`** (`13.31x`) | **`0.956293`** | **`3.176e-04`** | **`4/4` (`0.9674`)** | `508.69` | `41,795.9 / 43,279.5` |
| `30K` (`30,720`) | **KIVI `INT2` (`G=128, R=32`)** | `4,423,680` | **`7.11x`** (`14.22x`) | **`0.955884`** | **`3.174e-04`** | **`4/4` (`0.9565`)** | `631.54` | `47,322.1 / 50,723.5` |
| `128K` (`131,072`, 1 KV head) | `FP16` Baseline | `67,108,864` | `1.00x` (`2.00x`) | `1.000000` | `0.000e+00` | `4/4` (`1.0000`) | `0.00` | `57,008.5 / --` |
| `128K` (`131,072`, 1 KV head) | Uniform Sym `INT4` | `17,825,792` | `3.76x` (`7.53x`) | `0.707992` | `5.653e-06` | `0/4` (collapsed) | `--` | `--` |
| `128K` (`131,072`, 1 KV head) | Uniform Sym `INT2` | `9,437,184` | `7.11x` (`14.22x`) | `0.032949` | `9.116e-01` | `4/4` (noisy) | `--` | `--` |
| `128K` (`131,072`, 1 KV head) | **KIVI `INT4` (`G=64, R=64`)** | `18,373,888` | **`3.65x`** (`7.30x`) | **`0.996927`** | **`7.125e-08`** | **`4/4` (`0.9984`)** | `1,965.77` | `55,010.2 / 62,270.5` |
| `128K` (`131,072`, 1 KV head) | **KIVI `INT2` (`G=64, R=64`)** | `9,989,376` | **`6.72x`** (`13.44x`) | **`0.930943`** | **`2.287e-06`** | **`4/4` (`0.9674`)** | `2,124.20` | `55,138.7 / 60,265.9` |
| `128K` (`131,072`, 1 KV head) | **KIVI `INT2` (`G=128, R=32`)** | `9,437,184` | **`7.11x`** (`14.22x`) | **`0.928964`** | **`2.407e-06`** | **`4/4` (`0.9565`)** | `2,128.58` | `52,465.3 / 58,870.0` |

### 6.2 Shared `4B / 9B / 27B` Hybrid Attention + Unquantized GDN Resident State Matrix (`128K` Context)

| Hybrid Architecture (`131,072` tokens) | Baseline (`FP16` KV + `FP32` GDN) | KIVI `INT4` Total (`Attn Ratio / Total Ratio`) | KIVI `INT2` Total (`Attn Ratio / Total Ratio`) | Unquantized `FP32` GDN State |
|---|---|---|---|---|
| `4B-Hybrid` (`9` Attn + `27` GDN, `d=128`) | `1,179.00 MiB` | `342.41 MiB` (`3.65x / 3.44x`) | `189.00 MiB` (`7.11x / 6.24x`) | `27.00 MiB` (untouched) |
| `9B-Hybrid` (`10` Attn + `30` GDN, `d=128`) | `2,605.00 MiB` | `745.91 MiB` (`3.65x / 3.49x`) | `405.00 MiB` (`7.11x / 6.43x`) | `45.00 MiB` (untouched) |
| `27B-Hybrid` (`16` Attn + `48` GDN, `d=256`) | `8,384.00 MiB` | `2,402.92 MiB` (`3.71x / 3.49x`) | `1,312.00 MiB` (`7.31x / 6.39x`) | `192.00 MiB` (untouched) |

### 6.3 Honesty Boundaries and Integration Recommendation

- **Scope boundary**: The measurements above validate the backend-agnostic CPU reference/packing oracle and fused online paged-attention kernel in `hanzo-paged-attn::kivi`. End-to-end GPU TTFT, energy per token (`J/tok`), and live `4B/9B/27B` weight-loaded generation throughput on CUDA/ROCm/Vulkan/Metal hardware require wiring `PagedCacheType::{KiviInt4, KiviInt2}` device shaders against this exact `u32` page layout and re-measuring on target GPUs before promoting beyond opt-in.
- **Integration roadmap**:
  1. Adopt `hanzo_paged_attn::kivi` as the canonical packing and numerical oracle alongside `hanzo_paged_attn::quant` (mirroring the staged rollout used for `PagedCacheType::Int8` in commit `07f71598`).
  2. Wire `PagedCacheType::KiviInt4` first (`3.65x-3.71x` compression with `> 0.997` cosine fidelity), keeping `block_size = 64` aligned to `key_group_size = 64` so each physical `DType::U32` cache block in `CacheEngine` maps 1:1 to a `KiviKeyPage` / `KiviValuePage` pair while keeping the active unsealed block in `FP16` residual storage.
  3. Expose `PagedCacheType::KiviInt2` (`6.72x-7.31x` compression) behind an explicit opt-in flag (`--pa-cache-type kivi-int2`) for memory-bound `30K-128K` single-GPU sessions, while enforcing `LayerStateKind::RecurrentGdn` exemption across all hybrid model loaders.
