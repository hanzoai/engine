//! DSpark parallel-block speculative decoding on Qwen3-4B — end-to-end t/s benchmark.
//!
//! Loads Qwen3-4B (bf16, non-paged, `max_num_seqs = 1`) twice: once plain, once with a DSpark
//! draft attached (`SpeculativeConfig::Dspark`). Runs the SAME greedy coding prompt for 256
//! tokens each way and reports decode t/s, the DSpark accept rate, and the generated text.
//! Because greedy speculative decoding is exact, the two outputs must be byte-identical — that
//! doubles as the correctness check.
//!
//! Run:
//!   DSPARK_TARGET=/home/z/work/zen/hf/Qwen3-4B \
//!   DSPARK_DRAFT=/home/z/work/zen/hf/dspark/dspark_qwen3_4b_block7 \
//!   CUDA_COMPUTE_CAP=121 cargo run --release --features cuda \
//!     --example dspark_qwen3 -p hanzo

use anyhow::Result;
use hanzo::{ModelDType, RequestBuilder, TextMessageRole, TextModelBuilder};
use hanzo_engine::speculative::stats;

const PROMPT: &str =
    "Write a Python function to compute the nth Fibonacci number, then explain how it works.";
const MAX_TOKENS: usize = 256;

fn target_path() -> String {
    std::env::var("DSPARK_TARGET").unwrap_or_else(|_| "/home/z/work/zen/hf/Qwen3-4B".to_string())
}

fn draft_path() -> String {
    std::env::var("DSPARK_DRAFT")
        .unwrap_or_else(|_| "/home/z/work/zen/hf/dspark/dspark_qwen3_4b_block7".to_string())
}

/// A greedy 256-token request for the coding prompt.
fn request() -> RequestBuilder {
    RequestBuilder::new()
        .add_message(TextMessageRole::User, PROMPT)
        .set_deterministic_sampler()
        .set_sampler_max_len(MAX_TOKENS)
}

struct RunResult {
    text: String,
    compl_toks: usize,
    tok_per_sec: f32,
}

async fn run(model: &hanzo::Model, label: &str) -> Result<RunResult> {
    let response = model.send_chat_request(request()).await?;
    let choice = &response.choices[0];
    let text = choice.message.content.clone().unwrap_or_default();
    let compl_toks = response.usage.completion_tokens;
    let tok_per_sec = response.usage.avg_compl_tok_per_sec;
    println!("[{label}] completion_tokens={compl_toks}  decode={tok_per_sec:.2} tok/s");
    Ok(RunResult {
        text,
        compl_toks,
        tok_per_sec,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let target = target_path();
    let draft = draft_path();
    println!("target: {target}\ndraft : {draft}\n");

    // ---- Baseline: plain Qwen3-4B (no draft) ------------------------------------------------
    let baseline_model = TextModelBuilder::new(&target)
        .with_dtype(ModelDType::BF16)
        .with_max_num_seqs(1)
        .with_logging()
        .build()
        .await?;
    // Warm up (kernel autotune / graph capture) so the measured run is steady-state.
    let _ = baseline_model.send_chat_request(request()).await?;
    let baseline = run(&baseline_model, "baseline").await?;
    drop(baseline_model);

    // ---- DSpark: Qwen3-4B + parallel-block draft --------------------------------------------
    let dspark_model = TextModelBuilder::new(&target)
        .with_dtype(ModelDType::BF16)
        .with_max_num_seqs(1)
        .with_dspark(&draft, 0.0)
        .with_logging()
        .build()
        .await?;
    let _ = dspark_model.send_chat_request(request()).await?; // warmup
    stats::reset();
    let dspark = run(&dspark_model, "dspark").await?;
    let spec = stats::snapshot();
    drop(dspark_model);

    // ---- Report -----------------------------------------------------------------------------
    let speedup = if baseline.tok_per_sec > 0.0 {
        dspark.tok_per_sec / baseline.tok_per_sec
    } else {
        f32::NAN
    };
    println!("\n================ DSpark Qwen3-4B speculative benchmark ================");
    println!(
        "baseline : {:.2} tok/s  ({} tokens)",
        baseline.tok_per_sec, baseline.compl_toks
    );
    println!(
        "dspark   : {:.2} tok/s  ({} tokens)",
        dspark.tok_per_sec, dspark.compl_toks
    );
    println!("speedup  : {speedup:.2}x");
    println!(
        "accept   : {:.2} accepted + 1 verified per round over {} verify rounds \
         (mean staged {:.2}/block)",
        spec.mean_accepted(),
        spec.verify_rounds,
        spec.mean_proposed(),
    );
    let identical = baseline.text == dspark.text;
    println!(
        "outputs  : {} (greedy speculative decoding is exact)",
        if identical {
            "IDENTICAL ✓"
        } else {
            "DIFFERENT ✗ — verification bug"
        }
    );
    println!("======================================================================\n");
    println!("---- generated text (dspark) ----\n{}\n", dspark.text);

    if !identical {
        anyhow::bail!("dspark output diverged from greedy baseline — verification is unsound");
    }
    Ok(())
}
