//! Count-based prefill and decode throughput of an engine as it is served.
//!
//! The caller builds the model with the server's own builder, so a run measures what a server
//! gets: its device, paged attention, graphs and speculation. Tokens are the engine's usage
//! counts and the clock is this process's wall clock -- llama-bench's method, so the two engines
//! compare like for like.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use cli_table::{format::Justify, print_stdout, Cell, CellStruct, Style, Table};
use hanzo_engine::{
    Constraint, Hanzo, NormalRequest, Request, RequestMessage, Response, SamplingParams,
};
use tokio::sync::mpsc::{channel, Sender};
use tracing::info;

use crate::board::{Phase, Samples, Shape, Timed};

/// What to measure. Prefill at each prompt length and decode of `n_gen` tokens, at each
/// concurrency, `repetitions` times.
pub struct Spec {
    pub model_id: String,
    pub n_prompt: Vec<usize>,
    pub n_gen: usize,
    pub concurrency: Vec<usize>,
    pub repetitions: usize,
    /// Sample from the full vocabulary at temperature 1 instead of greedily: the sampler's tax
    /// measured, never a rate reported.
    pub stochastic: bool,
    /// Where the raw per-repetition samples go. Every published statistic is computed from them.
    pub json: Option<PathBuf>,
}

/// The backend this instrument was built for; the server's device choice follows the same order.
pub const BACKEND: &str = if cfg!(feature = "vulkan") {
    "Vulkan"
} else if cfg!(feature = "rocm") {
    "ROCm"
} else if cfg!(feature = "metal") {
    "Metal"
} else if cfg!(feature = "cuda") {
    "CUDA"
} else {
    "CPU"
};

/// Greedy is sampling parity with llama-bench: top-k 1 takes the device argmax. The stochastic
/// sampler draws from the full vocabulary on the host, a per-token tax llama-bench does not pay.
fn sampling(max_len: usize, greedy: bool) -> SamplingParams {
    if greedy {
        return SamplingParams {
            max_len: Some(max_len),
            ..SamplingParams::deterministic()
        };
    }
    SamplingParams {
        temperature: None,
        top_k: None,
        top_p: None,
        min_p: None,
        top_n_logprobs: 0,
        frequency_penalty: None,
        presence_penalty: None,
        repetition_penalty: None,
        max_len: Some(max_len),
        stop_toks: None,
        logits_bias: None,
        n_choices: 1,
        dry_params: None,
    }
}

fn request(
    hanzo: &Hanzo,
    messages: RequestMessage,
    sampling_params: SamplingParams,
    response: Sender<Response>,
) -> Request {
    Request::Normal(Box::new(NormalRequest {
        id: hanzo.next_request_id(),
        messages,
        sampling_params,
        response,
        return_logprobs: false,
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
        max_tool_rounds: None,
        tool_dispatch_url: None,
        model_id: None,
        truncate_sequence: false,
        session_id: None,
        files: None,
    }))
}

/// Time `repetitions` of one shape: each sends `concurrency` copies of the request and waits for
/// all of them. A failed forward is the run's failure, in the engine's own words.
async fn time(
    hanzo: &Hanzo,
    messages: RequestMessage,
    max_len: usize,
    shape: Shape,
    repetitions: usize,
    greedy: bool,
) -> Result<Timed> {
    let sender = hanzo.get_sender(None)?;
    let (tx, mut rx) = channel(10_000);
    let req = request(hanzo, messages, sampling(max_len, greedy), tx);
    let mut per_rep = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let t0 = Instant::now();
        for _ in 0..shape.concurrency {
            sender.send(req.clone()).await?;
        }
        let mut tokens = 0;
        let mut finished = 0;
        while finished < shape.concurrency {
            let usage = match rx.recv().await {
                Some(Response::Done(res)) => res.usage,
                Some(Response::CompletionDone(res)) => res.usage,
                Some(Response::AgenticToolCallProgress { .. })
                | Some(Response::AgenticToolApprovalRequired { .. })
                | Some(Response::File(_)) => continue,
                Some(Response::InternalError(e)) => anyhow::bail!("internal error: {e}"),
                Some(Response::ModelError(e, _)) => anyhow::bail!("model error: {e}"),
                Some(Response::CompletionModelError(e, _)) => anyhow::bail!("model error: {e}"),
                Some(Response::ValidationError(e)) => anyhow::bail!("validation error: {e}"),
                Some(_) => anyhow::bail!("unexpected response during a benchmark"),
                None => anyhow::bail!("response channel closed before a terminal response"),
            };
            tokens += match shape.phase {
                Phase::Prefill => usage.prompt_tokens,
                Phase::Decode => usage.completion_tokens,
            };
            finished += 1;
        }
        per_rep.push((t0.elapsed().as_secs_f64(), tokens));
    }
    Ok(Timed { shape, per_rep })
}

fn print_table(model: &str, results: &[Timed]) {
    let rows: Vec<Vec<CellStruct>> = results
        .iter()
        .filter_map(|t| Some((t.shape, t.stats()?)))
        .map(|(shape, st)| {
            let test = match shape.phase {
                Phase::Prefill => format!("pp {}", shape.n),
                Phase::Decode => format!("tg {}", shape.n),
            };
            let std = st.spread.as_ref().map_or(0.0, |s| s.std);
            let right = |s: String| s.cell().justify(Justify::Right);
            vec![
                model.cell(),
                BACKEND.cell(),
                test.cell(),
                right(format!("{:.3}±{std:.3}", st.mean)),
                right(format!("{:.3}", 1000.0 / st.mean)),
                right(shape.concurrency.to_string()),
                right(format!("{:.3}", st.mean * shape.concurrency as f64)),
                right(format!("{:.3}", st.best)),
            ]
        })
        .collect();
    let heads = [
        "model", "backend", "test", "t/s", "ms/t", "concurrency", "throughput/s", "best t/s",
    ];
    let table = rows.table().title(heads.map(|h| h.cell().bold(true))).bold(true);
    print_stdout(table).expect("print table");
}

/// Warm each shape, time every test at every concurrency, print the table, and write the
/// samples where `spec.json` says.
pub async fn run(hanzo: &Hanzo, spec: &Spec) -> Result<Samples> {
    let decode = RequestMessage::Completion {
        text: "Rust".to_string(),
        echo_prompt: false,
        best_of: None,
    };
    // A shape's phase and length, the request that exercises it, and the tokens it may generate.
    let mut tests = Vec::new();
    if spec.n_gen > 0 {
        tests.push((Phase::Decode, spec.n_gen, decode, spec.n_gen - 1));
    }
    for &n in spec.n_prompt.iter().filter(|n| **n > 0) {
        let prefill = RequestMessage::CompletionTokens((1000..1000 + n as u32).collect());
        tests.push((Phase::Prefill, n, prefill, 1));
    }

    // Warm each test at its own shape, as llama-bench does before it times anything: a short
    // prompt reaches only the decode matvec, and the prefill kernels would otherwise compile
    // inside the first timed repetition.
    for (phase, n, messages, _) in &tests {
        let shape = Shape {
            phase: *phase,
            n: *n,
            concurrency: 1,
        };
        time(hanzo, messages.clone(), 1, shape, 1, true).await?;
    }
    info!("Finished warmup run.");

    let mut results = Vec::new();
    for &concurrency in &spec.concurrency {
        let from = results.len();
        for (phase, n, messages, max_len) in &tests {
            let shape = Shape {
                phase: *phase,
                n: *n,
                concurrency,
            };
            let greedy = !spec.stochastic;
            let timed = time(hanzo, messages.clone(), *max_len, shape, spec.repetitions, greedy);
            results.push(timed.await?);
        }
        print_table(&spec.model_id, &results[from..]);
    }

    let samples = Samples {
        engine_version: env!("CARGO_PKG_VERSION").into(),
        backend: BACKEND.into(),
        sampler: match spec.stochastic {
            true => "stochastic-temp1-fullvocab".into(),
            false => "greedy-argmax".into(),
        },
        model_id: spec.model_id.clone(),
        results,
    };
    if let Some(path) = &spec.json {
        std::fs::write(path, serde_json::to_string_pretty(&samples)?)?;
        info!("Wrote raw samples to {}", path.display());
    }
    Ok(samples)
}
