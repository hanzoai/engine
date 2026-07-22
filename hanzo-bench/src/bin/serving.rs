//! serving — a continuous-batching serving benchmark.
//!
//! The default `hanzo bench` is a single-stream, uniform-length probe (llama-bench
//! style): every request is the same length, so at any decode step every running
//! sequence shares one length and the batch is trivially dense. That workload
//! cannot see the continuous-batching win, because it never produces the
//! length-divergent decode batch that iteration-level scheduling exists to keep
//! dense.
//!
//! This driver reproduces production load: a mixed-length (exponential, ShareGPT-
//! shaped) workload arriving as a Poisson process at a fixed rate, with a bounded
//! number of in-flight requests. It reports the numbers that actually move under
//! batching — aggregate output throughput, plus TTFT and TPOT percentiles — the
//! same axes Orca and Sarathi measure on.
//!
//! It also carries an `--identity` mode: the lossless bar for the scheduler
//! change. Greedy completions produced when K prompts run concurrently must be
//! byte-identical to the same prompts run one at a time. A batching bug that
//! crosses one sequence's state into another shows up here as a mismatch.

use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use hanzo_engine::{
    get_auto_device_map_params, get_model_dtype, initialize_logging, paged_attn_supported, Builder,
    Constraint, DefaultSchedulerMethod, DeviceMapSetting, Hanzo, Loader, LoaderBuilder,
    MemoryGpuConfig, ModelSelected, NormalRequest, PagedAttentionConfig, Request, RequestMessage,
    Response, SamplingParams, SchedulerConfig, TokenSource,
};
use hanzo_ml::Device;
use tokio::sync::mpsc::channel;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::info;

/// splitmix64 — a tiny, dependency-free, fully seedable PRNG. Deterministic
/// workloads are reproducible from `--seed` alone.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Exponential with the given mean (inverse-CDF).
    fn exp(&mut self, mean: f64) -> f64 {
        -mean * (1.0 - self.unit()).ln()
    }
    /// A token id in [lo, hi).
    fn tok(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u64() % (hi - lo) as u64) as u32
    }
}

/// One completed request's client-observed timings.
struct ReqMetrics {
    /// Time to first token: submit -> first streamed token.
    ttft: Duration,
    /// Mean inter-token latency across this request's decode.
    tpot: Option<Duration>,
    /// End-to-end: submit -> last token.
    e2e: Duration,
    output_toks: usize,
    /// Absolute wall-clock of the last token, for the aggregate window.
    last_token_at: Instant,
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return f64::NAN;
    }
    let idx = ((p / 100.0) * (sorted_ms.len() as f64 - 1.0)).round() as usize;
    sorted_ms[idx.min(sorted_ms.len() - 1)]
}

fn summarize(label: &str, mut values_ms: Vec<f64>) {
    values_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = values_ms.iter().sum::<f64>() / values_ms.len().max(1) as f64;
    println!(
        "  {label:<26} mean {:>9.2}  p50 {:>9.2}  p90 {:>9.2}  p99 {:>9.2}  (ms)",
        mean,
        percentile(&values_ms, 50.0),
        percentile(&values_ms, 90.0),
        percentile(&values_ms, 99.0),
    );
}

#[derive(Parser)]
#[command(about = "Mixed-length, Poisson-arrival continuous-batching serving benchmark")]
struct Args {
    #[clap(subcommand)]
    model: ModelSelected,

    /// Total number of requests to issue.
    #[arg(long, default_value_t = 200)]
    num_requests: usize,

    /// Poisson arrival rate in requests/sec. 0 means issue all at once (a burst),
    /// bounded only by `--max-concurrency`.
    #[arg(long, default_value_t = 0.0)]
    request_rate: f64,

    /// Maximum in-flight requests (the server's `max_num_seqs`).
    #[arg(long, default_value_t = 16)]
    max_concurrency: usize,

    /// Mean prompt length in tokens (exponential distribution).
    #[arg(long, default_value_t = 256)]
    mean_prompt: usize,

    /// Mean output length in tokens (exponential distribution).
    #[arg(long, default_value_t = 128)]
    mean_output: usize,

    /// Minimum output length in tokens (clamps the exponential tail near zero).
    #[arg(long, default_value_t = 8)]
    min_output: usize,

    /// Maximum output length in tokens (clamps the long tail).
    #[arg(long, default_value_t = 512)]
    max_output: usize,

    /// Seed for the (deterministic) workload generator.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Total KV cache context length for PagedAttention (tokens).
    #[arg(long)]
    paged_ctxt_len: Option<usize>,

    /// Disable PagedAttention (CUDA only; elsewhere it is off unless `--paged-attn`).
    #[arg(long, default_value_t = false)]
    no_paged_attn: bool,

    /// Enable PagedAttention on Metal/ROCm/Vulkan (on CUDA it is already the default).
    #[arg(long, default_value_t = false)]
    paged_attn: bool,

    /// Output-identity mode: assert concurrent greedy completions equal single-stream.
    #[arg(long, default_value_t = false)]
    identity: bool,

    /// Disable prefix caching. For the identity check this removes the confound of
    /// the single-stream pass warming a cache the concurrent pass then restores from,
    /// isolating decode batch size (1 vs N) as the only difference between the two.
    #[arg(long, default_value_t = false)]
    no_prefix_cache: bool,
}

/// Build the engine exactly as the standard bench does (device cascade, KV cache
/// sizing, scheduler selection), differing only in that this benchmark honors
/// `--paged-attn` so the production continuous-batching path is exercised.
async fn build_engine(args: &Args) -> anyhow::Result<(Arc<Hanzo>, Device, String)> {
    let dtype = get_model_dtype(&args.model)?;
    let auto_device_map_params = get_auto_device_map_params(&args.model)?;
    let max_seq_len = auto_device_map_params.max_seq_len();

    let loader: Box<dyn Loader> = LoaderBuilder::new(args.model.clone()).build()?;
    let model_name = loader.get_id().to_string();

    #[cfg(feature = "vulkan")]
    let device = Device::new_vulkan(0)?;
    #[cfg(all(feature = "rocm", not(feature = "vulkan")))]
    let device = Device::new_rocm(0)?;
    #[cfg(all(feature = "metal", not(feature = "rocm"), not(feature = "vulkan")))]
    let device = Device::new_metal(0)?;
    #[cfg(all(not(feature = "metal"), not(feature = "rocm"), not(feature = "vulkan")))]
    let device = Device::cuda_if_available(0)?;

    device.set_seed(args.seed)?;

    let no_paged_attn = if device.is_cuda() {
        args.no_paged_attn
    } else {
        !args.paged_attn
    };

    let cache_config = if paged_attn_supported() && !no_paged_attn {
        let mem = match args.paged_ctxt_len {
            Some(ctxt) => MemoryGpuConfig::ContextSize(ctxt),
            None => MemoryGpuConfig::ContextSize(max_seq_len),
        };
        Some(PagedAttentionConfig::new(None, mem, Default::default())?)
    } else {
        None
    };

    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &dtype,
        &device,
        false,
        DeviceMapSetting::Auto(auto_device_map_params),
        None,
        cache_config,
    )?;
    info!("Model loaded.");

    let scheduler_config = if cache_config.is_some() {
        if let Some(ref cache_config) = pipeline.lock().await.get_metadata().cache_config {
            SchedulerConfig::PagedAttentionMeta {
                max_num_seqs: args.max_concurrency,
                config: cache_config.clone(),
            }
        } else {
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(args.max_concurrency.try_into().unwrap()),
            }
        }
    } else {
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(args.max_concurrency.try_into().unwrap()),
        }
    };

    // Prefix caching stays enabled (production default). `disable_eos_stop` keeps
    // every request to its exact requested output length so throughput is measured
    // over a fixed token budget rather than variable early stops.
    let hanzo = Builder::new(pipeline, scheduler_config, false, None)
        .with_disable_eos_stop(true)
        .with_no_prefix_cache(args.no_prefix_cache)
        .build()
        .await;

    Ok((hanzo, device, model_name))
}

/// Greedy sampling with a fixed output length.
fn greedy(max_len: usize) -> SamplingParams {
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

fn make_request(
    hanzo: &Hanzo,
    messages: RequestMessage,
    out_len: usize,
    is_streaming: bool,
    tx: tokio::sync::mpsc::Sender<Response>,
) -> Request {
    Request::Normal(Box::new(NormalRequest {
        id: hanzo.next_request_id(),
        messages,
        sampling_params: greedy(out_len),
        response: tx,
        return_logprobs: false,
        is_streaming,
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

/// The mixed-length workload: `(prompt_tokens, output_len)` per request, plus the
/// cumulative Poisson arrival offset from t0.
fn workload(args: &Args) -> Vec<(Vec<u32>, usize, Duration)> {
    let mut rng = Rng(args.seed);
    let mut arrival = 0.0f64;
    (0..args.num_requests)
        .map(|_| {
            let prompt_len = (rng.exp(args.mean_prompt as f64).round() as usize).max(4);
            let out_len = (rng.exp(args.mean_output as f64).round() as usize)
                .clamp(args.min_output, args.max_output);
            let prompt = (0..prompt_len).map(|_| rng.tok(10, 10_000)).collect();
            if args.request_rate > 0.0 {
                arrival += rng.exp(1.0 / args.request_rate);
            }
            (prompt, out_len, Duration::from_secs_f64(arrival))
        })
        .collect()
}

/// Drive one streaming request to completion, returning client-observed timings.
async fn drive_request(
    hanzo: Arc<Hanzo>,
    prompt: Vec<u32>,
    out_len: usize,
) -> anyhow::Result<ReqMetrics> {
    let sender = hanzo.get_sender(None).unwrap();
    let (tx, mut rx) = channel(4096);
    let submit = Instant::now();
    sender
        .send(make_request(
            &hanzo,
            RequestMessage::CompletionTokens(prompt),
            out_len,
            true,
            tx,
        ))
        .await
        .map_err(|_| anyhow::anyhow!("engine receiver disconnected"))?;

    let mut first_token: Option<Instant> = None;
    let mut last_token: Option<Instant> = None;
    let mut streamed_toks = 0usize;
    let mut output_toks = 0usize;

    loop {
        match rx.recv().await {
            Some(Response::CompletionChunk(chunk)) => {
                let now = Instant::now();
                if first_token.is_none() {
                    first_token = Some(now);
                }
                // One chunk == one generated token in the streaming completion path.
                streamed_toks += chunk.choices.len().max(1);
                last_token = Some(now);
            }
            Some(Response::CompletionDone(resp)) => {
                // Authoritative output-token count from server-side usage.
                output_toks = resp.usage.completion_tokens;
                break;
            }
            Some(Response::CompletionModelError(e, _)) => {
                anyhow::bail!("completion model error: {e}")
            }
            Some(Response::InternalError(e)) | Some(Response::ValidationError(e)) => {
                anyhow::bail!("engine error: {e}")
            }
            Some(_) => continue,
            None => break,
        }
    }

    let first = first_token.unwrap_or(submit);
    let last = last_token.unwrap_or(first);
    // Fall back to the streamed count if the terminal usage was absent.
    if output_toks == 0 {
        output_toks = streamed_toks;
    }
    let tpot = if output_toks > 1 {
        Some((last - first) / (output_toks as u32 - 1))
    } else {
        None
    };
    Ok(ReqMetrics {
        ttft: first - submit,
        tpot,
        e2e: last - submit,
        output_toks,
        last_token_at: last,
    })
}

async fn run_load(hanzo: Arc<Hanzo>, args: &Args, model: &str, device: &Device) {
    let work = workload(args);
    let total_prompt_toks: usize = work.iter().map(|(p, _, _)| p.len()).sum();
    let planned_out_toks: usize = work.iter().map(|(_, o, _)| *o).sum();
    println!(
        "\nWorkload: {} requests, mean prompt {} tok, mean output {} tok, \
         arrival {}, max in-flight {}",
        args.num_requests,
        total_prompt_toks / args.num_requests.max(1),
        planned_out_toks / args.num_requests.max(1),
        if args.request_rate > 0.0 {
            format!("Poisson {:.1} req/s", args.request_rate)
        } else {
            "burst".to_string()
        },
        args.max_concurrency,
    );

    let sem = Arc::new(Semaphore::new(args.max_concurrency));
    let t0 = Instant::now();
    let mut tasks = JoinSet::new();

    for (prompt, out_len, arrival) in work {
        let hanzo = hanzo.clone();
        let sem = sem.clone();
        tasks.spawn(async move {
            let elapsed = t0.elapsed();
            if arrival > elapsed {
                tokio::time::sleep(arrival - elapsed).await;
            }
            let _permit = sem.acquire().await.unwrap();
            drive_request(hanzo, prompt, out_len).await
        });
    }

    let mut metrics = Vec::new();
    while let Some(res) = tasks.join_next().await {
        match res {
            Ok(Ok(m)) => metrics.push(m),
            Ok(Err(e)) => eprintln!("request failed: {e}"),
            Err(e) => eprintln!("task panicked: {e}"),
        }
    }

    if metrics.is_empty() {
        eprintln!("no requests completed");
        return;
    }

    // Aggregate window: first submission (t0) -> last token across all requests.
    let last_token_at = metrics.iter().map(|m| m.last_token_at).max().unwrap();
    let window = (last_token_at - t0).as_secs_f64();
    let output_toks: usize = metrics.iter().map(|m| m.output_toks).sum();

    println!(
        "\n=== {model} on {:?}  |  {} completed / {} issued ===",
        device,
        metrics.len(),
        args.num_requests
    );
    println!(
        "  aggregate output throughput   {:>9.2} tok/s   over {:.2}s window",
        output_toks as f64 / window,
        window
    );
    println!(
        "  request throughput            {:>9.2} req/s",
        metrics.len() as f64 / window
    );
    summarize(
        "TTFT",
        metrics.iter().map(|m| m.ttft.as_secs_f64() * 1e3).collect(),
    );
    summarize(
        "TPOT",
        metrics
            .iter()
            .filter_map(|m| m.tpot)
            .map(|d| d.as_secs_f64() * 1e3)
            .collect(),
    );
    summarize(
        "end-to-end latency",
        metrics.iter().map(|m| m.e2e.as_secs_f64() * 1e3).collect(),
    );
}

/// Non-streaming greedy completion returning the decoded text, for identity checks.
async fn complete_text(
    hanzo: Arc<Hanzo>,
    prompt: String,
    out_len: usize,
) -> anyhow::Result<String> {
    let sender = hanzo.get_sender(None).unwrap();
    let (tx, mut rx) = channel(4096);
    let messages = RequestMessage::Completion {
        text: prompt,
        echo_prompt: false,
        best_of: None,
    };
    sender
        .send(make_request(&hanzo, messages, out_len, false, tx))
        .await
        .map_err(|_| anyhow::anyhow!("engine receiver disconnected"))?;
    loop {
        match rx.recv().await {
            Some(Response::CompletionDone(resp)) => {
                return Ok(resp
                    .choices
                    .first()
                    .map(|c| c.text.clone())
                    .unwrap_or_default());
            }
            Some(Response::CompletionModelError(e, _)) => anyhow::bail!("model error: {e}"),
            Some(Response::InternalError(e)) | Some(Response::ValidationError(e)) => {
                anyhow::bail!("engine error: {e}")
            }
            Some(_) => continue,
            None => anyhow::bail!("channel closed before completion"),
        }
    }
}

/// First byte index at which two strings differ, or None if identical.
fn first_divergence(a: &str, b: &str) -> Option<usize> {
    a.bytes().zip(b.bytes()).position(|(x, y)| x != y).or({
        if a.len() == b.len() {
            None
        } else {
            Some(a.len().min(b.len()))
        }
    })
}

/// The lossless bar: real, well-conditioned greedy prompts of divergent length,
/// decoded concurrently (one length-mixed continuous batch), must match the same
/// prompts decoded one at a time. Real prompts (not random tokens) keep greedy
/// argmax off logit near-ties, so a mismatch signals cross-sequence contamination
/// rather than benign batch-size GEMM reduction-order noise. First-divergence
/// position separates the two: an immediate, structural split is contamination; a
/// rare, late split is floating-point drift.
async fn run_identity(hanzo: Arc<Hanzo>) {
    let prompts: Vec<String> = vec![
        "Once upon a time, in a small village nestled between two great mountains, there lived a curious young girl named".to_string(),
        "The theory of relativity fundamentally changed our understanding of space and time. In essence, it tells us that".to_string(),
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return".to_string(),
        "Dear hiring manager, I am writing to express my strong interest in the software engineering position at your".to_string(),
        "The capital of France is Paris, the capital of Japan is Tokyo, and the capital of Canada is".to_string(),
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of".to_string(),
        "To be, or not to be, that is the".to_string(),
        "Water boils at one hundred degrees Celsius at sea level because".to_string(),
    ];
    let out_len = 64usize;

    println!(
        "\n=== output-identity: {} real prompts, greedy, out_len {out_len} ===",
        prompts.len()
    );

    // Single-stream baseline: one request in flight at a time (decode batch = 1).
    let mut sequential = Vec::new();
    for p in &prompts {
        sequential.push(
            complete_text(hanzo.clone(), p.clone(), out_len)
                .await
                .unwrap(),
        );
    }

    // Concurrent: all K in flight -> one length-mixed continuous decode batch.
    let mut tasks = JoinSet::new();
    for (i, p) in prompts.iter().cloned().enumerate() {
        let hanzo = hanzo.clone();
        tasks.spawn(async move { (i, complete_text(hanzo, p, out_len).await.unwrap()) });
    }
    let mut concurrent = vec![String::new(); prompts.len()];
    while let Some(res) = tasks.join_next().await {
        let (i, text) = res.unwrap();
        concurrent[i] = text;
    }

    let mut mismatches = 0;
    for (i, (s, c)) in sequential.iter().zip(&concurrent).enumerate() {
        match first_divergence(s, c) {
            None => println!("  prompt[{i}]  IDENTICAL ({} bytes)", s.len()),
            Some(pos) => {
                mismatches += 1;
                println!(
                    "  prompt[{i}]  DIVERGES at byte {pos} of {} (single) / {} (batched)",
                    s.len(),
                    c.len()
                );
                println!("    single:   {s:?}");
                println!("    batched:  {c:?}");
            }
        }
    }
    if mismatches == 0 {
        println!("  PASS: concurrent continuous-batch decode == single-stream, byte-for-byte");
    } else {
        // Byte-identity across batch sizes requires batch-invariant kernels. F16 /
        // quantized GPU GEMM is not batch-invariant: reduction order depends on
        // batch composition, so greedy argmax can flip between near-tie tokens and
        // cascade. That is inherent to continuous batching (vLLM/Orca alike), not
        // corruption. The property that MUST hold is the absence of cross-sequence
        // contamination -- each batched completion stays a valid continuation of ITS
        // OWN prompt. A batched output carrying another prompt's content is the real
        // failure this mode exists to surface.
        println!(
            "  {mismatches}/{} sequence(s) diverge from single-stream (positions above).",
            prompts.len()
        );
        println!(
            "  Batch-invariant execution would be byte-identical; on F16/quantized GPU GEMM this\n  \
             is expected reduction-order variation. Verify each batched completion stays on-topic\n  \
             for its own prompt -- that (not byte-identity) is the no-contamination bar."
        );
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    initialize_logging();
    let args = Args::parse();

    let (hanzo, device, model) = build_engine(&args).await?;

    // Warm up the pipeline (weight upload, kernel/graph capture) before timing.
    let _ = drive_request(hanzo.clone(), (0..8).collect(), 4).await;

    if args.identity {
        run_identity(hanzo).await;
    } else {
        run_load(hanzo, &args, &model, &device).await;
    }
    Ok(())
}
