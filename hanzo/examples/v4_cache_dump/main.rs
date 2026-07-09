//! DeepSeek-V4 DSpark training-cache dump — teacher-force each text sample through
//! the 86GB IQ2 GGUF and write per-sample hidden-state shards for the DSpark
//! draft-head trainer.
//!
//! The capture itself lives in the model (`models/quantized_deepseek4.rs`, driven by
//! env `V4_CAPTURE_DIR`); this is a thin driver that streams samples through prefill.
//! Each request uses `max_len = 1`, so the only forward that carries signal is the
//! teacher-forced prefill over the full prompt — the in-model capture hooks collect
//! per-position hiddens during that pass and write one safetensors + an `index.jsonl`
//! line per sample.
//!
//! Run (GB10, earlyoom paused, ~90GB free):
//!   cargo run --release --features cuda --example v4_cache_dump -p hanzo -- \
//!       samples.jsonl out_dir/ [max_samples] [max_seq_len=2048]
//!
//! Input JSONL: one object per line with a `"text"` field, or a `"messages"` array
//! of `{role, content}` objects (their contents are concatenated).
//!
//! Output (in `out_dir/`):
//!   sample_<n>.safetensors  — input_ids [s] u32, target_hidden_states [s,5,4096]
//!                             bf16, target_last_hidden_states [s,4096] bf16
//!   index.jsonl             — {"idx","file","seq_len","ids_hash"} per sample

use std::io::BufRead;
use std::time::Instant;

use anyhow::{Context, Result};
use hanzo::{GgufModelBuilder, ModelDType, RequestBuilder, TextMessageRole};

/// Extract the sample text from one JSONL line: `"text"` verbatim, else the
/// concatenated `"messages"[].content`. Returns `None` for blank/unparseable lines.
fn sample_text(line: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
        let t = t.trim();
        return (!t.is_empty()).then(|| t.to_string());
    }
    if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
        let joined = msgs
            .iter()
            .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
            .collect::<Vec<_>>()
            .join("\n");
        let joined = joined.trim();
        return (!joined.is_empty()).then(|| joined.to_string());
    }
    None
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: {} <input.jsonl> <out_dir> [max_samples] [max_seq_len=2048]",
            args[0]
        );
        std::process::exit(2);
    }
    let input = args[1].clone();
    let out_dir = args[2].clone();
    let max_samples: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let max_seq_len: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2048);

    // Route the in-model capture: models/quantized_deepseek4.rs latches this on the
    // first prefill. Set it BEFORE build so the model's OnceLock reads the right dir.
    std::env::set_var("V4_CAPTURE_DIR", &out_dir);
    std::fs::create_dir_all(&out_dir).with_context(|| format!("mkdir {out_dir}"))?;

    // One sequence + NO MTP (capture is a pure teacher-forced prefill) + NO prefix
    // caching so EVERY sample prefills from offset 0 (is_prefill = true). With prefix
    // caching on, a shared chat-template prefix would be cache-reused and the second
    // sample onward would decode-continue instead of prefilling — skipping capture.
    // IGPU_MEMORY_FRACTION is honored by the allocator via env (unchanged here).
    println!("=== V4 cache dump: input={input} out={out_dir} max_samples={max_samples} max_seq_len={max_seq_len} ===");
    let model = GgufModelBuilder::new(
        "/home/z/work/zen/hf/ds4-flash-gguf",
        vec!["DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf".to_string()],
    )
    .with_max_num_seqs(1)
    .with_prefix_cache_n(None)
    .with_dtype(ModelDType::Auto)
    .with_logging()
    .build()
    .await?;

    let file = std::fs::File::open(&input).with_context(|| format!("open {input}"))?;
    let reader = std::io::BufReader::new(file);

    let start = Instant::now();
    let mut done = 0usize;
    let mut total_prompt_tokens = 0usize;
    for line in reader.lines() {
        if done >= max_samples {
            break;
        }
        let line = line?;
        let Some(mut text) = sample_text(&line) else {
            continue;
        };
        // Char-heuristic truncation (~4 chars/token) — an exact token cap is
        // unnecessary for a corpus dump.
        let cap = max_seq_len.saturating_mul(4);
        if text.chars().count() > cap {
            text = text.chars().take(cap).collect();
        }

        // max_len = 1: the prefill over the full prompt IS the teacher-forced pass.
        let req = RequestBuilder::new()
            .add_message(TextMessageRole::User, text)
            .set_sampler_max_len(1);
        let response = model.send_chat_request(req).await?;
        total_prompt_tokens += response.usage.prompt_tokens;
        done += 1;

        if done % 10 == 0 {
            let secs = start.elapsed().as_secs_f64();
            println!(
                "[{done}] running prefill tok/s {:.1} | last prompt_tokens {} avg_prompt_tok/s {:.1}",
                total_prompt_tokens as f64 / secs.max(1e-9),
                response.usage.prompt_tokens,
                response.usage.avg_prompt_tok_per_sec,
            );
        }
    }

    let secs = start.elapsed().as_secs_f64();
    println!(
        "=== done: {done} samples, {total_prompt_tokens} prompt tokens in {secs:.1}s | prefill tok/s {:.1} | shards -> {out_dir} ===",
        total_prompt_tokens as f64 / secs.max(1e-9),
    );
    Ok(())
}
