//! hanzo-bench: count-based prefill and decode throughput of the engine as it is served, and
//! the scoring of what was measured (`board`).
//!
//! The model is built by the server's own builder, so a run measures what a server gets: its
//! device, paged attention, graphs and speculation. Tokens are the engine's usage counts and
//! the clock is this process's wall clock -- llama-bench's method, so the two compare like for
//! like.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use cli_table::{format::Justify, print_stdout, Cell, CellStruct, Style, Table};
use hanzo_bench::board::{self, Phase, Samples, Shape, Timed};
use hanzo_engine::{
    initialize_logging, Constraint, Hanzo, ModelSelected, MtpConfig, NormalRequest, PagedCacheType,
    Request, RequestMessage, Response, SamplingParams, TokenSource,
};
use hanzo_server_core::server::ServerBuilder;
use tokio::sync::mpsc::{channel, Sender};
use tracing::info;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,

    /// Integer seed to ensure reproducible random number generation.
    #[arg(short, long)]
    seed: Option<u64>,

    /// Prompt tokens of the prefill test; 0 skips it.
    #[arg(long, short = 'p', default_value_t = 512)]
    n_prompt: usize,

    /// Generated tokens of the decode test; 0 skips it.
    #[arg(long, short = 'g', default_value_t = 128)]
    n_gen: usize,

    /// Concurrent requests per repetition; each value is its own set of tests.
    #[arg(short, long, value_delimiter = ',', default_value = "1")]
    concurrency: Vec<usize>,

    /// Repetitions of each test. With three or more, the first is scored as warmup.
    #[arg(long, short, default_value_t = 5)]
    repetitions: usize,

    /// Device layers, as a count or `ORD:NUM;...`; omitted, the automatic device map.
    #[arg(short, long, value_delimiter = ';')]
    num_device_layers: Option<Vec<String>>,

    /// In-situ quantization to apply.
    #[arg(long = "isq")]
    in_situ_quant: Option<String>,

    /// KV cache budget in MB. Priority: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    #[arg(long = "pa-gpu-mem")]
    paged_attn_gpu_mem: Option<usize>,

    /// KV cache budget as a fraction of device memory, 0 to 1.
    #[arg(long = "pa-gpu-mem-usage")]
    paged_attn_gpu_mem_usage: Option<f32>,

    /// KV cache budget as the tokens it must hold; default, the model's `max-seq-len`.
    #[arg(long = "pa-ctxt-len")]
    paged_ctxt_len: Option<usize>,

    /// KV cache type (auto or f8e4m3).
    #[arg(long = "pa-cache-type", value_parser = |s: &str| s.parse::<PagedCacheType>())]
    cache_type: Option<PagedCacheType>,

    /// Tokens per KV cache block.
    #[arg(long = "pa-blk-size")]
    paged_attn_block_size: Option<usize>,

    /// Turn PagedAttention off where the server would turn it on.
    #[arg(long = "no-paged-attn")]
    no_paged_attn: bool,

    /// Turn PagedAttention on where the server would leave it off.
    #[arg(long = "paged-attn", conflicts_with = "no_paged_attn")]
    paged_attn: bool,

    /// Draft with a DFlash checkpoint directory.
    #[arg(long)]
    dflash: Option<String>,

    /// Draft a block shorter than the DFlash checkpoint's trained block; 0 drafts the full block.
    #[arg(long, default_value_t = 0)]
    dflash_block_size: usize,

    /// Draft with a multi-token-prediction head.
    #[arg(long)]
    mtp_model: Option<String>,

    #[arg(long, requires = "mtp_model")]
    mtp_n_predict: Option<usize>,

    /// Write the raw per-repetition samples (wall seconds, scored tokens) of every test here.
    /// Every published statistic is computed from this file, so its uncertainty is auditable.
    #[arg(long = "json")]
    json: Option<PathBuf>,

    /// Sample from the full vocabulary at temperature 1 instead of greedily. It exists to
    /// measure the sampler's tax, not to report a rate.
    #[arg(long)]
    stochastic: bool,
}

// Parsed once per process; the model selection is the large variant.
#[allow(clippy::large_enum_variant)]
#[derive(clap::Subcommand)]
enum Command {
    #[command(flatten)]
    Measure(ModelSelected),
    /// Score a run directory from the raw samples in it: board.md, board.json, and the paper's
    /// results-data.tex and board.tex.
    Score { run: PathBuf },
    /// Runs as Hanzo Research evidence: printed, or filed with --to (bearer `$HANZO_API_KEY`).
    Publish {
        #[arg(required = true)]
        runs: Vec<PathBuf>,
        #[arg(long, value_name = "URL")]
        to: Option<String>,
    },
    /// Pin a run before its first sample: write its manifest.
    Manifest {
        out: PathBuf,
        #[command(flatten)]
        pins: board::Pins,
    },
}

const BACKEND: &str = if cfg!(feature = "vulkan") {
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
/// all of them. A failed forward is the run's failure, with the engine's own words.
async fn time(
    hanzo: &Hanzo,
    messages: RequestMessage,
    max_len: usize,
    shape: Shape,
    repetitions: usize,
    greedy: bool,
) -> anyhow::Result<Timed> {
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
        "model",
        "backend",
        "test",
        "t/s",
        "ms/t",
        "concurrency",
        "throughput/s",
        "best t/s",
    ];
    let table = rows
        .table()
        .title(heads.map(|h| h.cell().bold(true)))
        .bold(true);
    print_stdout(table).expect("print table");
}

async fn measure(args: &Args, model: ModelSelected) -> anyhow::Result<()> {
    initialize_logging();
    let paged_attn = match (args.paged_attn, args.no_paged_attn) {
        (true, _) => Some(true),
        (_, true) => Some(false),
        _ => None,
    };
    let mtp = args
        .mtp_model
        .clone()
        .map(|m| MtpConfig::new(m, args.mtp_n_predict));
    let hanzo = ServerBuilder::new()
        .with_model(model)
        .with_max_seqs(args.concurrency.iter().copied().max().unwrap_or(1))
        .with_token_source(TokenSource::CacheToken)
        .with_interactive_mode(false)
        .with_prefix_cache_n(0)
        .with_disable_eos_stop(true)
        .with_seed_optional(args.seed)
        .with_num_device_layers_optional(args.num_device_layers.clone())
        .with_in_situ_quant_optional(args.in_situ_quant.clone())
        .set_paged_attn(paged_attn)
        .with_paged_attn_gpu_mem_optional(args.paged_attn_gpu_mem)
        .with_paged_attn_gpu_mem_usage_optional(args.paged_attn_gpu_mem_usage)
        .with_paged_ctxt_len_optional(args.paged_ctxt_len)
        .with_paged_attn_block_size_optional(args.paged_attn_block_size)
        .with_paged_attn_cache_type(args.cache_type.unwrap_or_default())
        .with_mtp_config_optional(mtp)
        .with_dflash_optional(args.dflash.clone(), args.dflash_block_size)
        .build()
        .await?;
    let model_id = hanzo
        .get_default_model_id()
        .ok()
        .flatten()
        .unwrap_or_default();
    info!("Model loaded.");

    let decode = RequestMessage::Completion {
        text: "Rust".to_string(),
        echo_prompt: false,
        best_of: None,
    };
    let prefill = RequestMessage::CompletionTokens((1000..1000 + args.n_prompt as u32).collect());
    // A shape's phase and length, the request that exercises it, and the tokens it may generate.
    let tests = [
        (
            Phase::Decode,
            args.n_gen,
            decode,
            args.n_gen.saturating_sub(1),
        ),
        (Phase::Prefill, args.n_prompt, prefill, 1),
    ];
    let tests: Vec<_> = tests.into_iter().filter(|t| t.1 > 0).collect();

    // Warm each test at its own shape, as llama-bench does before it times anything: a short
    // prompt reaches only the decode matvec, and the prefill kernels would otherwise compile
    // inside the first timed repetition.
    for (phase, n, messages, _) in &tests {
        let shape = Shape {
            phase: *phase,
            n: *n,
            concurrency: 1,
        };
        time(&hanzo, messages.clone(), 1, shape, 1, true).await?;
    }
    info!("Finished warmup run.");

    let mut results = Vec::new();
    for &concurrency in &args.concurrency {
        let from = results.len();
        for (phase, n, messages, max_len) in &tests {
            let shape = Shape {
                phase: *phase,
                n: *n,
                concurrency,
            };
            let greedy = !args.stochastic;
            results.push(
                time(
                    &hanzo,
                    messages.clone(),
                    *max_len,
                    shape,
                    args.repetitions,
                    greedy,
                )
                .await?,
            );
        }
        print_table(&model_id, &results[from..]);
    }

    if let Some(path) = &args.json {
        let sampler = match args.stochastic {
            true => "stochastic-temp1-fullvocab",
            false => "greedy-argmax",
        };
        let samples = Samples {
            engine_version: env!("CARGO_PKG_VERSION").into(),
            backend: BACKEND.into(),
            sampler: sampler.into(),
            model_id,
            results,
        };
        std::fs::write(path, serde_json::to_string_pretty(&samples)?)?;
        info!("Wrote raw samples to {}", path.display());
    }
    Ok(())
}

fn print_json<T: serde::Serialize>(value: &T) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn publish(runs: &[PathBuf], to: Option<&String>) -> anyhow::Result<()> {
    let runs = runs
        .iter()
        .map(|r| board::read(r))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let evidence = board::evidence(&runs)?;
    match to {
        Some(url) => board::post(url, &evidence),
        None => print_json(&serde_json::json!({"experiments": evidence, "attempts": []})),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match &args.command {
        Command::Score { run } => print!("{}", board::markdown(&board::score(run)?)),
        Command::Publish { runs, to } => publish(runs, to.as_ref())?,
        Command::Manifest { out, pins } => {
            std::fs::write(
                out,
                serde_json::to_string_pretty(&board::pin(pins.clone())?)?,
            )?;
            println!("manifest -> {}", out.display());
        }
        Command::Measure(model) => {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(measure(&args, model.clone()))?;
            // The samples are written. A GPU runtime's exit-time teardown can abort after the
            // fact (ROCm on gfx1151 does), and a harness would read that as a failed run.
            std::io::stdout().flush()?;
            std::process::exit(0);
        }
    }
    Ok(())
}
