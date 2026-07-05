//! End-to-end CORRECTNESS tests for the core LLM inference path:
//! tokenize -> prefill -> decode -> detokenize.
//!
//! These exercise the REAL pipeline (GGUF load + quant dequant + attention + sampling) through the
//! same `ServerBuilder` -> `Request::Normal` -> `Response::CompletionDone` flow the CLI/server
//! use, and assert on the actual generated text/logprobs. They are NOT "it didn't crash" smoke tests:
//! every test makes a hard assertion on inference output (exact strings, contained substrings, or a
//! numeric ceiling).
//!
//! Default device is CPU so the suite is portable and deterministic; build with `--features cuda` to
//! exercise the CUDA path instead (greedy/top_k=1 output is expected to match the CPU reference for a
//! correct backend, though tiny float differences can in principle perturb a late token; the verbatim
//! reference is captured on the same backend you bless on).
//!
//! Env-gated on the model path so CI without a model is a no-op:
//!   TEST_GGUF=/abs/path/to/Qwen3-0.6B-Q8_0.gguf  (a local Qwen3/zen-eco GGUF file)
//! Optional:
//!   TEST_BLESS=1   print the captured greedy completions/logits then PASS (use to (re)generate
//!                        the committed reference strings below from current HEAD), e.g.
//!                        TEST_GGUF=... TEST_BLESS=1 cargo test -p hanzo-cli \
//!                          --test llm_e2e_correctness -- --nocapture --test-threads=1
//!
//! Keep --test-threads=1: each test loads the model into its own engine; running them serially keeps
//! peak memory ~one model and avoids contention.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use hanzo_engine::{
    Constraint, Hanzo, ModelDType, ModelSelected, NormalRequest, Request, RequestMessage, Response,
    SamplingParams, TokenSource,
};
use hanzo_server_core::server::ServerBuilder;
use tokio::sync::mpsc::channel;

// ---- Committed reference outputs (captured from HEAD on CPU via TEST_BLESS=1) ----
// Model: Qwen3-0.6B-Q8_0.gguf. Raw-completion path (no chat template), greedy (top_k=1).
// If a deliberate, understood change to the inference math lands, re-bless and update these in the
// same commit. An UNEXPECTED change here means the inference path regressed (wrong dequant, wrong
// attention/RoPE, wrong sampling) -> investigate before re-blessing.

const PROMPT_FRANCE: &str = "The capital of France is";
// 16 greedy tokens of completion appended after PROMPT_FRANCE (no leading space: GGUF byte-BPE).
const REF_FRANCE_COMPLETION: &str =
    "Paris. The capital of Italy is Rome. The capital of Spain is Madrid.";

const PROMPT_COUNT: &str = "1 2 3 4 5 6 7 8 9";
// 12 greedy tokens after PROMPT_COUNT: the model should keep counting.
const REF_COUNT_COMPLETION: &str = "10 11 12 13";

const PROMPT_MATH: &str = "2 + 2 =";

// Perplexity ceiling for a trivial, in-distribution English sentence under greedy logprobs.
// A correct small model is highly confident here; a broken dequant/attention path blows this up.
const PPL_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const PPL_CEILING: f64 = 30.0;

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("TEST_GGUF").ok()?;
    let pb = PathBuf::from(&p);
    if !pb.is_file() {
        panic!("TEST_GGUF={p} is not a file");
    }
    Some(pb)
}

fn bless() -> bool {
    std::env::var("TEST_BLESS")
        .map(|v| v == "1")
        .unwrap_or(false)
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

/// Build a single-model CPU (or feature-selected) engine over the local GGUF. Mirrors the
/// loader/builder wiring used by `hanzo run`/`hanzo bench`, so this is the real production path.
async fn load(path: &Path) -> Arc<Hanzo> {
    let (dir, file) = split_dir_file(path);
    let model = ModelSelected::GGUF {
        tok_model_id: None, // use the tokenizer + chat template embedded in the GGUF (fully offline)
        quantized_model_id: dir,
        quantized_filename: file,
        dtype: ModelDType::Auto,
        topology: None,
        max_seq_len: 4096,
        max_batch_size: 1,
    };

    // CPU unless a GPU feature is enabled; greedy decode of a 0.6B model is fast enough on CPU.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "rocm",
        feature = "vulkan"
    ))]
    let cpu = false;
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "rocm",
        feature = "vulkan"
    )))]
    let cpu = true;

    ServerBuilder::new()
        .with_model(model)
        .with_max_seqs(1)
        .with_no_kv_cache(false)
        .with_token_source(TokenSource::None)
        .with_interactive_mode(false)
        .with_prefix_cache_n(0) // determinism: never serve a cached prefix
        .with_cpu(cpu)
        .build()
        .await
        .expect("load GGUF model")
}

struct Generation {
    text: String,
    logprobs: Vec<(String, f32)>,
}

/// Greedy (top_k=1) raw-completion. `RequestMessage::Completion` skips the chat template, so the
/// model continues the prompt verbatim -> reproducible. EOS stop is left ON; we cap with `max_len`.
async fn greedy_complete(
    hanzo: &Arc<Hanzo>,
    prompt: &str,
    max_tokens: usize,
    want_logprobs: bool,
) -> Generation {
    let mut sp = SamplingParams::deterministic();
    sp.max_len = Some(max_tokens);
    if want_logprobs {
        sp.top_n_logprobs = 1;
    }

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
        return_logprobs: want_logprobs,
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

    sender.send(req).await.expect("send request");

    loop {
        match rx.recv().await {
            Some(Response::CompletionDone(resp)) => {
                let choice = resp.choices.into_iter().next().expect("one choice");
                let logprobs = choice
                    .logprobs
                    .and_then(|l| l.content)
                    .map(|c| c.into_iter().map(|r| (r.token, r.logprob)).collect())
                    .unwrap_or_default();
                return Generation {
                    text: choice.text,
                    logprobs,
                };
            }
            Some(Response::CompletionModelError(e, _)) => panic!("completion model error: {e}"),
            Some(Response::InternalError(e)) => panic!("internal error: {e:?}"),
            Some(Response::ValidationError(e)) => panic!("validation error: {e:?}"),
            Some(Response::ModelError(e, _)) => panic!("model error: {e}"),
            Some(Response::AgenticToolCallProgress { .. }) | Some(Response::File(_)) => continue,
            Some(_) => panic!(
                "unexpected non-terminal response variant for a non-streaming completion request"
            ),
            None => panic!("response channel closed with no completion"),
        }
    }
}

fn assert_finite(label: &str, logprobs: &[(String, f32)]) {
    assert!(!logprobs.is_empty(), "{label}: expected logprobs, got none");
    for (i, (tok, lp)) in logprobs.iter().enumerate() {
        assert!(
            lp.is_finite(),
            "{label}: non-finite logprob {lp} at step {i} (token {tok:?}) -> NaN/inf in the dequant/attention path"
        );
        // A real next-token logprob is <= 0 (log of a probability) and not absurdly negative.
        assert!(
            *lp <= 1e-3 && *lp > -50.0,
            "{label}: implausible logprob {lp} at step {i} (token {tok:?})"
        );
    }
}

/// (1) DETERMINISTIC OUTPUT + semantic correctness.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deterministic_greedy_output() {
    let Some(path) = model_path() else {
        eprintln!(
            "SKIP deterministic_greedy_output: set TEST_GGUF to a local Qwen3/zen-eco GGUF"
        );
        return;
    };
    let hanzo = load(&path).await;

    let france = greedy_complete(&hanzo, PROMPT_FRANCE, 16, false).await;
    let count = greedy_complete(&hanzo, PROMPT_COUNT, 12, false).await;

    if bless() {
        eprintln!("\n==== BLESS: paste these into the REF_* consts ====");
        eprintln!("REF_FRANCE_COMPLETION = {:?}", france.text);
        eprintln!("REF_COUNT_COMPLETION  = {:?}", count.text);
        eprintln!("==================================================\n");
        return;
    }

    // Semantic correctness: the world capital fact must surface.
    assert!(
        france.text.contains("Paris"),
        "expected 'Paris' in completion of {PROMPT_FRANCE:?}, got {:?}",
        france.text
    );

    // Determinism / stability: greedy output must match the captured reference byte-for-byte.
    assert_eq!(
        france.text, REF_FRANCE_COMPLETION,
        "greedy completion drifted from committed reference (inference path changed?)"
    );
    assert_eq!(
        count.text, REF_COUNT_COMPLETION,
        "greedy counting completion drifted from committed reference"
    );

    // Determinism within a run: same prompt + greedy must reproduce exactly.
    let france2 = greedy_complete(&hanzo, PROMPT_FRANCE, 16, false).await;
    assert_eq!(
        france.text, france2.text,
        "greedy decode is not deterministic across identical requests"
    );
}

/// (2) QUANT PATH: GGUF (Q8_0 here) loads + runs the dequant path with no NaN/inf and coherent output.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn quant_path_no_nan_coherent() {
    let Some(path) = model_path() else {
        eprintln!("SKIP quant_path_no_nan_coherent: set TEST_GGUF");
        return;
    };
    let hanzo = load(&path).await;

    // 24 tokens with per-token logprobs so we can prove the quantized matmul output is finite.
    let gen = greedy_complete(&hanzo, PROMPT_FRANCE, 24, true).await;

    if bless() {
        eprintln!(
            "\n[bless] quant path produced {} tokens, text = {:?}",
            gen.logprobs.len(),
            gen.text
        );
        return;
    }

    assert_finite("quant_path", &gen.logprobs);

    let text = gen.text.trim();
    assert!(!text.is_empty(), "quant path produced empty output");
    // Coherent decode = printable text, not replacement chars / control-byte garbage.
    assert!(
        text.chars()
            .all(|c| c.is_ascii_graphic() || c == ' ' || c == '\n'),
        "quant path produced non-printable output (garbled decode?): {text:?}"
    );
    assert!(
        !text.contains('\u{FFFD}'),
        "quant path produced U+FFFD replacement chars (broken detokenize): {text:?}"
    );

    // Repetition-collapse guard: a blown-up quant path often degenerates into one repeated token.
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() >= 6 {
        let distinct = words.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(
            distinct >= 3,
            "quant path output looks degenerate (only {distinct} distinct tokens in {:?})",
            words
        );
    }
}

/// (3) SANITY: trivial-arithmetic top-1 expectation + a perplexity ceiling on a fixed English text.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sanity_logits_and_perplexity() {
    let Some(path) = model_path() else {
        eprintln!("SKIP sanity_logits_and_perplexity: set TEST_GGUF");
        return;
    };
    let hanzo = load(&path).await;

    // 3a. Trivial arithmetic: greedy continuation of "2 + 2 =" must contain "4".
    let math = greedy_complete(&hanzo, PROMPT_MATH, 4, true).await;
    if bless() {
        eprintln!(
            "[bless] math: text={:?} first_tok={:?}",
            math.text,
            math.logprobs.first()
        );
    }
    assert_finite("math", &math.logprobs);
    if !bless() {
        assert!(
            math.text.contains('4'),
            "expected '4' in greedy completion of {PROMPT_MATH:?}, got {:?}",
            math.text
        );
    }

    // 3b. Perplexity ceiling on a fixed in-distribution sentence.
    // ppl = exp(-mean(log p(chosen token))). Each step here is the greedy (argmax) token, so its
    // logprob is the max next-token logprob; a correct model is confident -> low ppl. A broken
    // dequant/attention/softmax path flattens the distribution -> ppl explodes past the ceiling.
    let ppl_gen = greedy_complete(&hanzo, PPL_TEXT, 12, true).await;
    assert_finite("ppl", &ppl_gen.logprobs);

    let n = ppl_gen.logprobs.len() as f64;
    let sum_lp: f64 = ppl_gen.logprobs.iter().map(|(_, lp)| *lp as f64).sum();
    let ppl = (-sum_lp / n).exp();

    if bless() {
        eprintln!(
            "[bless] perplexity over {} greedy steps = {:.4}",
            n as usize, ppl
        );
        return;
    }

    assert!(ppl.is_finite(), "perplexity is non-finite: {ppl}");
    assert!(
        ppl < PPL_CEILING,
        "greedy perplexity {ppl:.4} exceeded ceiling {PPL_CEILING} on {PPL_TEXT:?} (logits sanity failed)"
    );
}
