# Hanzo Engine Integration Guide

This document covers MLX support, Metal GPU acceleration, and Apple Silicon optimization in hanzo-engine.

## Overview

Hanzo Engine is a high-performance LLM inference engine. On macOS/iOS, it leverages Apple's Metal framework for GPU acceleration with specialized quantization kernels inspired by MLX (Apple's ML framework).

## Metal Backend Support

### Feature Flags

Enable Metal support via Cargo features:

```toml
# hanzo-engine/Cargo.toml
[features]
default = ["metal"]
metal = ["mistralrs-core/metal"]
```

The `metal` feature enables:
- `candle-core/metal` - Metal compute backend for tensor operations
- `candle-nn/metal` - Neural network layers with Metal acceleration
- `mistralrs-quant/metal` - Quantization kernels optimized for Metal
- `mistralrs-paged-attn/metal` - Paged attention with Metal support

### Building for Metal

```bash
# Default build includes Metal on macOS
cargo build --release

# Explicit Metal feature
cargo build --release --features metal

# With Accelerate framework (recommended for Apple Silicon)
cargo build --release --features "metal accelerate"
```

## MLX-Style Quantization

### AFQ (Affine Quantization)

AFQ is the primary quantization method optimized for Metal, inspired by MLX's quantization approach.

**Supported bit depths:**
- AFQ2 (2-bit)
- AFQ3 (3-bit)
- AFQ4 (4-bit)
- AFQ6 (6-bit)
- AFQ8 (8-bit)

**Group sizes:**
- 32 (Low)
- 64 (Medium, default)
- 128 (High)

### Metal Kernels

The Metal kernels are located in `/Users/z/work/hanzo/engine/mistralrs-quant/src/metal_kernels/`:

| File | Purpose |
|------|---------|
| `quantized.metal` | MLX-style SIMD matrix operations (`mlx::steel` namespace) |
| `hqq_dequantize.metal` | HQQ dequantization |
| `bnb_dequantize.metal` | BitsAndBytes dequantization |
| `blockwise_fp8.metal` | FP8 blockwise operations |
| `scan.metal` | Parallel scan operations |
| `sort.metal` | Sorting operations |
| `bitwise.metal` | Bitwise operations |

The `quantized.metal` kernel implements MLX's STEEL (SIMD Tiled Efficient Execution Layer) for:
- 8x8 SIMD matrix fragments
- Efficient matrix multiply-accumulate (MMA)
- Block swizzling for memory coalescing

## Using MLX Prequantized Models

Hanzo Engine supports loading models prequantized with MLX from Hugging Face:

```bash
# Load an MLX prequantized model from mlx-community
cargo run --features metal --release -- \
    -i plain -m mlx-community/Llama-3.8-1B-8bit
```

The engine automatically detects MLX quantization format via `quantization_config.json` and uses optimized Metal kernels for inference.

### Supported Quantization Configs

The engine recognizes these quantization methods in config files:
- `afq` - Affine quantization (MLX-native)
- `gptq` - GPTQ quantization
- `awq` - AWQ quantization
- `fp8` - FP8 quantization
- `bitsandbytes` - BNB int8/fp4/nf4
- `mxfp4` - MXFP4 format

## In-Situ Quantization (ISQ)

ISQ quantizes models at load time, reducing memory footprint:

```bash
# Automatic ISQ (selects best method for platform)
# On Metal: uses AFQ for 2, 3, 4, 6, 8 bits
cargo run --features metal --release -- \
    -i --isq 4 plain -m meta-llama/Llama-3.2-3B-Instruct
```

### Automatic Method Selection

When using integer ISQ values (2, 3, 4, 5, 6, 8):
- **Metal**: AFQ is selected (fast, optimized for Apple Silicon)
- **CUDA**: Q/K quantization is selected
- **CPU**: Q/K quantization fallback

### Explicit ISQ Types

```bash
# Explicit AFQ on Metal
cargo run --features metal --release -- \
    -i --isq AFQ4 plain -m meta-llama/Llama-3.2-3B-Instruct

# Explicit Q/K type
cargo run --features metal --release -- \
    -i --isq Q4K plain -m meta-llama/Llama-3.2-3B-Instruct
```

## GPU Memory Detection

The Metal backend provides memory introspection:

```rust
// From mistralrs-core/src/utils/memory_usage.rs
Device::Metal(dev) => {
    let max = dev.recommended_max_working_set_size();
    let alloc = dev.current_allocated_size();
    let avail = max.saturating_sub(alloc);
}
```

This enables automatic device mapping for large models.

## Performance Tips

### Apple Silicon Optimization

1. **Use AFQ quantization**: Designed specifically for Metal
2. **Enable Accelerate**: `--features accelerate` uses Apple's BLAS
3. **Match group size to model**: Larger group sizes (128) for bigger models
4. **Use PagedAttention**: Reduces memory fragmentation

### Recommended Configuration

```bash
# Optimal setup for Apple Silicon
cargo run --release --features "metal accelerate" -- \
    --port 8080 \
    --isq 4 \
    plain -m meta-llama/Llama-3.2-3B-Instruct
```

## Quantization Support Matrix

| Method | CPU | CUDA | Metal |
|--------|-----|------|-------|
| GGUF/GGML | Yes | Yes | Yes |
| GPTQ | No | Yes | No |
| AWQ | No | Yes | No |
| HQQ | Yes | Yes | Yes |
| FP8 | Yes | Yes | Yes |
| BNB | Yes | Yes | Yes |
| AFQ | No | No | Yes |
| ISQ | Yes | Yes | Yes |
| MLX prequant | No | No | Yes |

## Upstream Synchronization

This repository tracks upstream changes and merges them periodically.

Recent upstream branches of interest:
- `mlx_rmsnorm_and_rope` - MLX RMSNorm and RoPE optimizations
- `metal_gather_mm` - Metal gather matrix multiply
- `metal_parallel_load` - Parallel model loading on Metal

## Troubleshooting

### Metal Shader Compilation

If precompiled shaders fail, runtime compilation occurs automatically:
```
Metal requires macosx > 13.0 or higher
```

Ensure macOS 13.0+ for full Metal support.

### Memory Issues

For large models on limited VRAM:
1. Use lower-bit quantization (AFQ2, AFQ3)
2. Enable device mapping: `--auto-device-map`
3. Reduce context length

## References

- [docs/QUANTS.md](/Users/z/work/hanzo/engine/docs/QUANTS.md) - Full quantization documentation
- [docs/ISQ.md](/Users/z/work/hanzo/engine/docs/ISQ.md) - In-situ quantization details
- [docs/UQFF.md](/Users/z/work/hanzo/engine/docs/UQFF.md) - Universal Quantized File Format
- [MLX Framework](https://github.com/ml-explore/mlx) - Apple's ML framework
