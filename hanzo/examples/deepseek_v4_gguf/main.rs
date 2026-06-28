//! Load + run the DeepSeek-V4 GGUF (antirez ds4 80GB IQ2 build) — the V4 load-test.
//!
//! Run (GB10, earlyoom paused):
//!   cargo run --release --features cuda --example deepseek_v4_gguf -p hanzo
//!
//! The GGUF carries its own tokenizer + chat template, so no external files needed.

use anyhow::Result;
use hanzo::{GgufModelBuilder, TextMessageRole, TextMessages};

#[tokio::main]
async fn main() -> Result<()> {
    let model = GgufModelBuilder::new(
        "/home/z/work/zen/hf/ds4-flash-gguf",
        vec!["DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf".to_string()],
    )
    .with_logging()
    .build()
    .await?;

    // Keep it <= window_size (128 tokens) so the v1 dense path is the correct one.
    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "In one sentence, what is the capital of France?",
    );

    let response = model.send_chat_request(messages).await?;
    println!("\n=== DEEPSEEK-V4 OUTPUT ===");
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    println!("=== tok/s: prompt {:?} compl {:?} ===",
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec);
    Ok(())
}
