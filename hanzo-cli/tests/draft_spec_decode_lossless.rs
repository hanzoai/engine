//! LOSSLESSNESS regression test for classic draft+target speculative decoding.
//!
//! The verifier corrects every draft token against the target's own distribution, so greedy (argmax)
//! decoding MUST produce the byte-for-byte identical token sequence whether or not a draft model is
//! attached. This codifies that invariant: it greedily completes the same prompts with and without a
//! self-draft (the draft IS the target GGUF, gamma>=2 so multiple drafts are verified per target
//! step) and asserts the generated token sequences are identical. This is the gate that proves
//! losslessness forever; an inequality here means the verify/accept path regressed.
//!
//! The classic draft path requires PagedAttention, which is GPU-only, so this test is a no-op unless
//! built with a GPU backend AND given a local GGUF (so CPU CI is a clean skip):
//!   TEST_GGUF=/abs/path/to/Qwen3-0.6B-Q8_0.gguf \
//!     cargo test -p hanzo-cli --features rocm --test draft_spec_decode_lossless \
//!       -- --nocapture --test-threads=1

use std::path::{Path, PathBuf};
use std::sync::Arc;

use hanzo_engine::{
    Constraint, Hanzo, ModelDType, ModelSelected, NormalRequest, Request, RequestMessage, Response,
    SamplingParams, TokenSource,
};
use hanzo_server_core::server::ServerBuilder;
use tokio::sync::mpsc::channel;

const MAX_SEQ_LEN: usize = 4096;
const GAMMA: usize = 4; // >= 2 so several draft tokens are verified per target step.
const GEN_TOKENS: usize = 48;

/// PagedAttention (required by the classic draft path) is only available on a GPU backend.
const GPU: bool = cfg!(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan"
));

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
        tok_model_id: None, // use the GGUF's embedded tokenizer (fully offline, draft==target vocab)
        quantized_model_id: dir,
        quantized_filename: file,
        dtype: ModelDType::Auto,
        topology: None,
        max_seq_len: MAX_SEQ_LEN,
        max_batch_size: 1,
    }
}

/// Build a GPU engine over the GGUF with PagedAttention forced on. With `with_draft`, the same GGUF
/// is attached as a self-draft (identical tokenizer + weights), so every draft token is exactly what
/// the target would greedily emit -> high acceptance that exercises the propose/verify/cache wiring
/// while the verifier still owns correctness.
async fn load(path: &Path, with_draft: bool) -> Arc<Hanzo> {
    let draft = with_draft.then(|| gguf(path));
    ServerBuilder::new()
        .with_model(gguf(path))
        .with_max_seqs(1)
        .with_no_kv_cache(false)
        .with_token_source(TokenSource::None)
        .with_interactive_mode(false)
        .with_prefix_cache_n(0) // determinism: never serve a cached prefix
        .with_cpu(false)
        .set_paged_attn(Some(true)) // classic draft path requires PagedAttention
        .with_paged_ctxt_len_optional(Some(MAX_SEQ_LEN)) // bound the KV cache so self-draft fits too
        .with_draft_model_optional(draft, GAMMA)
        .build()
        .await
        .expect("load GGUF model (+ optional self-draft) with paged attention")
}

/// Greedy (argmax) raw completion. Returns the per-step generated token strings (the token-level
/// fingerprint) and the detokenized text. `RequestMessage::Completion` skips the chat template so the
/// model continues the prompt verbatim and the run is reproducible.
async fn greedy_tokens(
    hanzo: &Arc<Hanzo>,
    prompt: &str,
    max_tokens: usize,
) -> (Vec<String>, String) {
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

    sender.send(req).await.expect("send request");

    loop {
        match rx.recv().await {
            Some(Response::CompletionDone(resp)) => {
                let choice = resp.choices.into_iter().next().expect("one choice");
                let toks = choice
                    .logprobs
                    .and_then(|l| l.content)
                    .map(|c| c.into_iter().map(|r| r.token).collect())
                    .unwrap_or_default();
                return (toks, choice.text);
            }
            Some(Response::CompletionModelError(e, _)) => panic!("completion model error: {e}"),
            Some(Response::InternalError(e)) => panic!("internal error: {e:?}"),
            Some(Response::ValidationError(e)) => panic!("validation error: {e:?}"),
            Some(Response::ModelError(e, _)) => panic!("model error: {e}"),
            Some(Response::AgenticToolCallProgress { .. }) | Some(Response::File(_)) => continue,
            Some(_) => {
                panic!("unexpected non-terminal response variant for a non-streaming completion")
            }
            None => panic!("response channel closed with no completion"),
        }
    }
}

/// Greedy output must be identical with and without a (self-)draft. This is the losslessness gate.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn draft_spec_decode_is_lossless_greedy() {
    if !GPU {
        eprintln!(
            "SKIP draft_spec_decode_is_lossless_greedy: classic draft decoding needs PagedAttention \
             (GPU-only); build with --features cuda|metal|rocm|vulkan"
        );
        return;
    }
    let Some(path) = model_path() else {
        eprintln!("SKIP draft_spec_decode_is_lossless_greedy: set TEST_GGUF to a local GGUF");
        return;
    };

    let prompts = [
        "The capital of France is",
        "1 2 3 4 5 6 7 8 9",
        "Once upon a time,",
    ];

    // Baseline: target only (paged attention, no draft).
    let baseline = load(&path, false).await;
    let mut without = Vec::new();
    for p in prompts {
        without.push(greedy_tokens(&baseline, p, GEN_TOKENS).await);
    }
    drop(baseline);

    // With self-draft (gamma=GAMMA) over the same target + paged attention.
    let drafted = load(&path, true).await;
    for (p, (ref_toks, ref_text)) in prompts.into_iter().zip(without.into_iter()) {
        let (toks, text) = greedy_tokens(&drafted, p, GEN_TOKENS).await;
        assert!(!toks.is_empty(), "draft run produced no tokens for {p:?}");
        assert_eq!(
            toks, ref_toks,
            "LOSSLESSNESS VIOLATED for {p:?}: draft greedy tokens differ from no-draft\n  no-draft: {ref_toks:?}\n  draft:    {toks:?}"
        );
        assert_eq!(
            text, ref_text,
            "LOSSLESSNESS VIOLATED for {p:?}: draft greedy text differs from no-draft"
        );
    }

    // All assertions held (a failure would have panicked above). Skip the racy ROCm-runtime atexit
    // teardown (a cosmetic post-success "pure virtual method called" SIGABRT in the HIP runtime, not
    // our code) by exiting cleanly here, mirroring the CLI one-shot path. A real failure still aborts
    // non-zero via the panic before reaching this point.
    use std::io::Write;
    let _ = std::io::stdout().flush();
    std::process::exit(0);
}
