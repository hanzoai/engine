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
    // Bench knobs (measure-don't-guess the MTP draft depth):
    //   argv[1]  = MTP n_predict MAX draft depth (default 2); 0 disables MTP entirely.
    //   argv[2]  = sampler max_len in tokens (default 48 = quick correctness check;
    //             pass ~256 for a stable steady-state decode-rate measurement).
    //   argv[3..] = prompt override (joined by spaces); default = a coherent-long
    //             list prompt (sustained generation with mixed easy/hard spans, so
    //             it exercises adaptive draft length without degenerating early).
    //   env MTP_CONF_THRESHOLD = adaptive draft-length confidence gate (0.0 = fixed
    //             depth = old behavior; ~0.6 = extend the chain only while the draft
    //             head's top token prob stays above the gate).
    let args: Vec<String> = std::env::args().collect();
    let n_predict: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let max_len: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(48);
    let prompt: String = if args.len() > 3 {
        args[3..].join(" ")
    } else {
        "List 20 widely used programming languages. For each, give the language \
         name followed by a colon and a short phrase describing its primary use."
            .to_string()
    };

    // Tight activation budget for the 86GB load on unified memory: a single
    // sequence (default 32) shrinks the KV reservation ~32x, leaving the pool for
    // weights. Pair with IGPU_MEMORY_FRACTION=0.92 on a freed GB10.
    // MTP self-speculative decode is net-positive here via the non-paged
    // NormalSpeculativeCacheAccess; greedy-identical output preserved (the target
    // verify decides every token). n_predict=0 drops the MTP head for the baseline.
    let mut builder = GgufModelBuilder::new(
        "/home/z/work/zen/hf/ds4-flash-gguf",
        vec!["DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf".to_string()],
    )
    .with_max_num_seqs(1)
    .with_logging();
    if n_predict > 0 {
        builder = builder.with_mtp_model(
            "/home/z/work/zen/hf/ds4-flash-gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf",
            Some(n_predict),
        );
    }
    println!("=== V4 bench: MTP n_predict={n_predict} max_len={max_len} ===");
    let model = builder.build().await?;

    // A prompt that elicits sustained generation so the reported decode rate is
    // steady-state, not dominated by the first few tokens.
    let messages = RequestBuilder::new()
        .add_message(TextMessageRole::User, prompt)
        .set_sampler_max_len(max_len);

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
