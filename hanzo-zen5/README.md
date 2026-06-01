# hanzo-zen5

Rust crate giving hanzod a clean `Zen5Engine` API for loading Zen5 GGUF models
and streaming inference.

## Backends

| Feature flag | Status | What it does |
|---|---|---|
| `ffi` (default) | scaffold + wiring | Wraps the vendored `zen5-engine` C runtime (Metal / CUDA / CPU). |
| `native` | scaffold | Pure-Rust DeepSeek V4 Flash on `candle-core` / `candle-transformers`. |

Build the native path standalone without a C toolchain:

```sh
cargo check -p hanzo-zen5 --no-default-features --features=native
```

Build the FFI path (requires the submodule and a C compiler):

```sh
git submodule add https://github.com/zenlm/zen5-engine \
    hanzo-libs/hanzo-zen5/zen5-engine-src
cargo build -p hanzo-zen5 --features=metal     # macOS
cargo build -p hanzo-zen5 --features=cuda      # Linux H100/Spark
```

## Layout

```
hanzo-zen5/
  Cargo.toml
  build.rs                  cc::Build + bindgen for ds4.h
  src/
    lib.rs                  re-exports
    engine.rs               Zen5Engine trait, GenOpts, ThinkMode, Zen5Error
    ffi/
      mod.rs                RAII Engine wrapper, blocking-thread bridge
      sys.rs                bindgen output (or stub when submodule missing)
    native/
      mod.rs                candle Zen5Model scaffold + V4 Flash 284B config
      attention.rs          MLA (Multi-head Latent Attention) scaffold
      moe.rs                Sparse MoE 256 experts top-8 scaffold
  examples/
    inference.rs            load gguf + stream one prompt
  zen5-engine-src/          (gitsubmodule — not vendored in tree)
```

See `src/engine.rs` for the trait both backends satisfy.
