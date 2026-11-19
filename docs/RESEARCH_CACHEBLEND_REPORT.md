# Research Report: Non-prefix RAG KV Fusion (CacheBlend)

## 1. Paper & License Audit
- **Reference**: [CacheBlend: Fast Prompt-Adaptive LLM Serving (arXiv:2405.16444)](https://arxiv.org/abs/2405.16444)
- **License**: The implementation follows the Apache-2.0 license compatible with `hanzo-engine`.
- **Design**: We utilize a block-level fusion strategy that avoids full re-prefill by treating retrieved RAG documents as independent KV blocks.

## 2. Falsifiable Hypothesis
- **Hypothesis**: By reusing KV blocks for retrieved documents, we expect a reduction in TTFT by ~30-40% for RAG prompts with >50% document overlap compared to full prefill.
- **Failure Mode**: If the `fusion_threshold` is too low, cross-block attention misalignment will cause perplexity degradation > 0.05.

## 3. Measurements
- **Baseline**: Full prefill (no cache).
- **Target**: TTFT < 200ms for 10k token context with 4k cached document blocks.
- **Methodology**: Use `hanzo bench` with the `calibration_datav3_small.txt` dataset.
