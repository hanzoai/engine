//! Load + run the DeepSeek-V4 GGUF (antirez ds4 80GB IQ2 build) — the V4 load-test.
//!
//! Run (GB10, earlyoom paused):
//!   cargo run --release --features cuda --example deepseek_v4_gguf -p hanzo
//!
//! The GGUF carries its own tokenizer + chat template, so no external files needed.

use anyhow::Result;
use hanzo::{GgufModelBuilder, RequestBuilder, TextMessageRole};

#[tokio::main]
async fn main() -> Result<()> {
    // Tight activation budget for the 86GB load on unified memory: a single
    // sequence (default 32) shrinks the KV reservation ~32x, leaving the pool for
    // weights. Pair with IGPU_MEMORY_FRACTION=0.92 on a freed GB10.
    let model = GgufModelBuilder::new(
        "/home/z/work/zen/hf/ds4-flash-gguf",
        vec!["DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf".to_string()],
    )
    .with_max_num_seqs(1)
    .with_logging()
    .build()
    .await?;

    // Correctness check: a factual short-answer prompt that naturally stops.
    let messages = RequestBuilder::new().add_message(
        TextMessageRole::User,
        "In one sentence, what is the capital of France?",
    );

    let response = model.send_chat_request(messages).await?;
    println!("\n=== DEEPSEEK-V4 OUTPUT ===");
    println!(
        "{}",
        response.choices[0].message.content.as_deref().unwrap_or("<none>")
    );
    println!(
        "=== tok/s: prompt {:?} compl {:?} | completion_tokens {} ===",
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec,
        response.usage.completion_tokens,
    );
    Ok(())
}
