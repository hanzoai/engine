//! End-to-end example: load a Zen5 GGUF and stream one completion.
//!
//! Run with:
//!
//! ```text
//! cargo run --example inference --release -- \
//!     /path/to/zen-5-flash.gguf "Why is the sky blue?"
//! ```
//!
//! Requires the FFI backend (default). For the native backend swap
//! `Engine::load(...)?` for `hanzo_zen5::native::Engine::load(...)?`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use futures::StreamExt;
use hanzo_zen5::engine::{GenOpts, Zen5Engine};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model: PathBuf = args
        .next()
        .context("usage: inference <model.gguf> <prompt>")?
        .into();
    let prompt: String = args.next().unwrap_or_else(|| "Hello, ".into());

    #[cfg(feature = "ffi")]
    let engine = hanzo_zen5::ffi::Engine::load(&model, Default::default())
        .context("load Zen5 engine")?;

    #[cfg(all(not(feature = "ffi"), feature = "native"))]
    let engine = hanzo_zen5::native::Engine::load(&model).context("load Zen5 engine")?;

    #[cfg(not(any(feature = "ffi", feature = "native")))]
    {
        let _ = model;
        let _ = prompt;
        anyhow::bail!("no backend enabled — build with --features=ffi or --features=native");
    }

    #[cfg(any(feature = "ffi", feature = "native"))]
    {
        println!("backend: {}", engine.backend());
        let opts = GenOpts {
            max_tokens: 256,
            ..Default::default()
        };
        let mut stream = engine.complete(&prompt, opts).await?;
        while let Some(token) = stream.next().await {
            let token = token?;
            print!("{}", token.text);
        }
        println!();
    }

    Ok(())
}
