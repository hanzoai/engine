//! LOSSLESSNESS + acceptance regression test for prompt-lookup (n-gram) speculative decoding.
//!
//! Prompt-lookup drafts from the sequence's OWN token history and returns `logits: None`, so the
//! target verifier accepts a draft only when it matches the token the target would itself emit
//! (verifier.rs exact-match path). Greedy (argmax) decoding therefore MUST produce the byte-for-byte
//! identical token sequence whether prompt-lookup is on or off. This codifies that invariant and, on
//! a grounded/repetitive prompt, also proves drafts are actually being accepted (mean accepted > 1).
//!
//! Unlike the classic draft path, prompt-lookup needs NO draft model and runs on the non-paged
//! (normal KV) backend, so this test runs on plain CPU. It is a clean skip unless a local GGUF is
//! provided:
//!   TEST_GGUF=/abs/path/to/zen-eco-4b.gguf \
//!     cargo test -p hanzo-cli --test prompt_lookup_lossless -- --nocapture --test-threads=1

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use hanzo_engine::{
    Constraint, Hanzo, ModelDType, ModelSelected, NormalRequest, Request, RequestMessage, Response,
    SamplingParams, TokenSource,
};
use hanzo_server_core::server::ServerBuilder;
use tokio::sync::mpsc::channel;

const MAX_SEQ_LEN: usize = 4096;
const NGRAM_MAX: usize = 3; // longest tail n-gram to match
const GAMMA: usize = 8; // draft tokens proposed per verify step

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("TEST_GGUF").ok()?;
    let pb = PathBuf::from(&p);
    if !pb.is_file() {
        panic!("TEST_GGUF={p} is not a file");
    }
    Some(pb)
}

fn split_dir_file(path: &Path) -> (String, String) {
    let dir = path
        .parent()
        .map(|d| {
            if d.as_os_str().is_empty() {
                ".".to_string()
            } else {
                d.to_string_lossy().to_string()
            }
        })
        .unwrap_or_else(|| ".".to_string());
    let file = path
        .file_name()
        .expect("gguf path has a filename")
        .to_string_lossy()
        .to_string();
    (dir, file)
}

fn gguf(path: &Path) -> ModelSelected {
    let (dir, file) = split_dir_file(path);
    ModelSelected::GGUF {
        tok_model_id: None, // use the GGUF's embedded tokenizer (fully offline)
        quantized_model_id: dir,
        quantized_filename: file,
        dtype: ModelDType::Auto,
        topology: None,
        max_seq_len: MAX_SEQ_LEN,
        max_batch_size: 1,
    }
}

/// Build a CPU engine over the GGUF with paged attention off (prompt-lookup drives the non-paged
/// normal-cache path). `prompt_lookup` enables the n-gram proposer; `None` is the plain baseline.
async fn load(path: &Path, prompt_lookup: Option<usize>) -> Arc<Hanzo> {
    ServerBuilder::new()
        .with_model(gguf(path))
        .with_max_seqs(1)
        .with_no_kv_cache(false)
        .with_token_source(TokenSource::None)
        .with_interactive_mode(false)
        .with_prefix_cache_n(0) // determinism: never serve a cached prefix
        .with_cpu(true)
        .set_paged_attn(Some(false)) // prompt-lookup uses the non-paged backend
        .with_prompt_lookup_optional(prompt_lookup, GAMMA)
        .build()
        .await
        .expect("load GGUF model (+ optional prompt-lookup) on CPU")
}

/// Greedy (argmax) raw completion. Returns the per-step generated token strings (the token-level
/// fingerprint), the detokenized text, and the wall time. `RequestMessage::Completion` skips the
/// chat template so the model continues the prompt verbatim and the run is reproducible.
async fn greedy(
    hanzo: &Arc<Hanzo>,
    prompt: &str,
    max_tokens: usize,
) -> (Vec<String>, String, Duration) {
    let mut sp = SamplingParams::deterministic();
    sp.max_len = Some(max_tokens);
    sp.top_n_logprobs = 1; // expose the chosen token per step

    let sender = hanzo.get_sender(None).expect("sender");
    let (tx, mut rx) = channel(64);

    let req = Request::Normal(Box::new(NormalRequest {
        id: hanzo.next_request_id(),
        messages: RequestMessage::Completion {
            text: prompt.to_string(),
            echo_prompt: false,
            best_of: None,
        },
        sampling_params: sp,
        response: tx,
        return_logprobs: true,
        is_streaming: false,
        constraint: Constraint::None,
        suffix: None,
        tools: None,
        tool_choice: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        enable_code_execution: false,
        code_execution_permission: None,
        code_execution_approval_notifier: None,
        agent_permission: None,
        agent_approval_handler: None,
        agent_approval_notifier: None,
        session_id: None,
        max_tool_rounds: None,
        tool_dispatch_url: None,
        model_id: None,
        truncate_sequence: false,
        files: None,
    }));

    let start = Instant::now();
    sender.send(req).await.expect("send request");

    loop {
        match rx.recv().await {
            Some(Response::CompletionDone(resp)) => {
                let elapsed = start.elapsed();
                let choice = resp.choices.into_iter().next().expect("one choice");
                let toks = choice
                    .logprobs
                    .and_then(|l| l.content)
                    .map(|c| c.into_iter().map(|r| r.token).collect())
                    .unwrap_or_default();
                return (toks, choice.text, elapsed);
            }
            Some(Response::CompletionModelError(e, _)) => panic!("completion model error: {e}"),
            Some(Response::InternalError(e)) => panic!("internal error: {e:?}"),
            Some(Response::ValidationError(e)) => panic!("validation error: {e:?}"),
            Some(Response::ModelError(e, _)) => panic!("model error: {e}"),
            Some(Response::AgenticToolCallProgress { .. }) | Some(Response::File(_)) => continue,
            Some(_) => panic!("unexpected non-terminal response variant for a completion"),
            None => panic!("response channel closed with no completion"),
        }
    }
}

/// A grounded/repetitive prompt: a function defined verbatim several times primes the model to keep
/// reproducing it, so long spans recur in-history and prompt-lookup can draft them. This is the
/// canonical code/agentic scenario where n-gram speculation wins.
const GROUNDED: &str = "def add(a, b):\n    return a + b\n\ndef add(a, b):\n    return a + b\n\ndef add(a, b):\n    return a + b\n\ndef add(a, b):\n    return";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prompt_lookup_is_lossless_and_accepts() {
    let Some(path) = model_path() else {
        eprintln!("SKIP prompt_lookup_is_lossless_and_accepts: set TEST_GGUF to a local GGUF");
        return;
    };

    let baseline = load(&path, None).await; // OFF
    let drafted = load(&path, Some(NGRAM_MAX)).await; // ON

    // ---- Losslessness + coherence: greedy output identical ON vs OFF -------------------------
    let coherence_prompt = "The capital of France is";
    let lossless_prompts = [coherence_prompt, "1 2 3 4 5 6 7 8 9", GROUNDED];
    for prompt in lossless_prompts {
        let (off_toks, off_text, _) = greedy(&baseline, prompt, 24).await;
        let (on_toks, on_text, _) = greedy(&drafted, prompt, 24).await;
        assert!(
            !on_toks.is_empty(),
            "prompt-lookup produced no tokens for {prompt:?}"
        );
        assert_eq!(
            on_toks, off_toks,
            "LOSSLESSNESS VIOLATED for {prompt:?}: prompt-lookup greedy tokens differ from baseline\n  off: {off_toks:?}\n  on:  {on_toks:?}"
        );
        assert_eq!(
            on_text, off_text,
            "LOSSLESSNESS VIOLATED for {prompt:?}: prompt-lookup greedy text differs from baseline"
        );
    }
    // Coherence: the model actually answers, and identically with the proposer attached.
    let (_, paris_text, _) = greedy(&drafted, coherence_prompt, 8).await;
    assert!(
        paris_text.contains("Paris"),
        "coherence check failed: {coherence_prompt:?} -> {paris_text:?} (expected to contain \"Paris\")"
    );

    // ---- Acceptance: drafts are actually accepted on the grounded prompt ----------------------
    hanzo_engine::speculative::stats::reset();
    let (_, _, on_grounded) = greedy(&drafted, GROUNDED, 48).await;
    let stats = hanzo_engine::speculative::stats::snapshot();
    eprintln!(
        "prompt-lookup acceptance: verify_rounds={}, mean_accepted={:.2}, mean_proposed={:.2}",
        stats.verify_rounds,
        stats.mean_accepted(),
        stats.mean_proposed()
    );
    assert!(
        stats.verify_rounds > 0,
        "no speculative verify rounds ran — prompt-lookup did not engage"
    );
    assert!(
        stats.mean_accepted() > 1.0,
        "expected mean accepted drafts > 1 on a grounded prompt, got {:.2}",
        stats.mean_accepted()
    );

    // ---- Speedup: measured and reported, not asserted -----------------------------------------
    // Batched verify emits (1 + accepted) tokens from ONE forward of width (1 + gamma). The win is
    // memory-bandwidth-bound: when decode streams the weights once per forward (GPU, or a large
    // model on CPU), the wider verify forward is nearly free and throughput scales with acceptance.
    // On a compute-bound decode (this small model on this CPU box) the wider forward instead costs
    // ~(1 + gamma)x a width-1 forward, so throughput can be net-negative even at high acceptance —
    // exactly why the proposer auto-gates on batch. The deterministic gates above (byte-identical
    // output + accepted drafts > 1) are the correctness contract; the raw throughput is diagnostic.
    let (off_toks, _, off_grounded) = greedy(&baseline, GROUNDED, 48).await;
    let (on_toks, _, _) = greedy(&drafted, GROUNDED, 48).await;
    let n = on_toks.len().min(off_toks.len()) as f64;
    let tps_off = n / off_grounded.as_secs_f64();
    let tps_on = n / on_grounded.as_secs_f64();
    eprintln!(
        "grounded decode: off={:.2} tok/s ({:.2}s), on={:.2} tok/s ({:.2}s), throughput_ratio={:.2}x (regime-dependent; see comment)",
        tps_off,
        off_grounded.as_secs_f64(),
        tps_on,
        on_grounded.as_secs_f64(),
        tps_on / tps_off
    );
}
