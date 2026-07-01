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
    // The MTP self-speculative draft head is available via `.with_mtp_model(path, n)`
    // (fully wired + greedy-identical). It's left off here because naive depth-1 MTP
    // is net-negative single-stream; the net-positive path is EAGLE-style multi-draft.
    let model = GgufModelBuilder::new(
        "/home/z/work/zen/hf/ds4-flash-gguf",
        vec!["DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf".to_string()],
    )
    .with_max_num_seqs(1)
    .with_logging()
    .build()
    .await?;

    // Correctness check: a factual short-answer prompt, capped so a working kernel
    // completes fast (a deadlocked one still hangs — the decisive distinction).
    let messages = RequestBuilder::new()
        .add_message(
            TextMessageRole::User,
            "In one sentence, what is the capital of France?",
        )
        .set_sampler_max_len(48);

    let response = model.send_chat_request(messages).await?;
    let msg = &response.choices[0].message;
    // V4 is a reasoning model: the answer may arrive as `content` (on natural stop)
    // or still be in `reasoning_content` when a max_len cap cuts the chain short.
    let text = msg
        .content
        .as_deref()
        .or(msg.reasoning_content.as_deref())
        .unwrap_or("<none>");
    println!("\n=== DEEPSEEK-V4 OUTPUT ===\n{text}");
    println!(
        "=== tok/s: prompt {:?} compl {:?} | completion_tokens {} ===",
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec,
        response.usage.completion_tokens,
    );
    Ok(())
}
