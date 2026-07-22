use clap::Parser;
use cli_table::{format::Justify, print_stdout, Cell, CellStruct, Style, Table};
use hanzo_engine::{
    get_auto_device_map_params, get_model_dtype, initialize_logging, paged_attn_supported,
    parse_isq_value, Builder, Constraint, DefaultSchedulerMethod, DeviceLayerMapMetadata,
    DeviceMapMetadata, DeviceMapSetting, Hanzo, Loader, LoaderBuilder,
    MemoryGpuConfig, ModelSelected, NormalRequest, PagedAttentionConfig, PagedCacheType, Request,
    RequestMessage, Response, SamplingParams, SchedulerConfig, TokenSource, Usage,
};
use hanzo_ml::Device;
use std::fmt::Display;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::channel;
use tracing::{info, warn};

#[derive(Clone, Copy)]
enum TestName {
    Prompt(usize),
    Gen(usize),
}

impl Display for TestName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            TestName::Prompt(n) => format!("pp {n}"),
            TestName::Gen(n) => format!("tg {n}"),
        };
        write!(f, "{name}")
    }
}

// Per repetition: (wall seconds for the whole concurrency batch, tokens scored that batch). Tokens
// scored are prompt_tokens for a Prompt test and completion_tokens for a Gen test -- llama-bench's
// definition. Wall-clock is measured here rather than read from the response's self-reported rate,
// which the completion path does not populate from real prefill timing.
struct BenchResult {
    per_rep: Vec<(f64, usize)>,
    concurrency: usize,
    test_name: TestName,
}

struct UncertainTokSec {
    mean: f32,
    std_dev: f32,
}

impl Display for UncertainTokSec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3}±{:.3}", self.mean, self.std_dev)
    }
}

async fn run_bench(
    hanzo: Arc<Hanzo>,
    prompt: RequestMessage,
    n_gen: usize,
    concurrency: usize,
    repetitions: usize,
    test_name: TestName,
    greedy: bool,
) -> anyhow::Result<BenchResult> {
    // Sampling PARITY with llama-bench (which decodes greedily) is a required control.
    // greedy == deterministic(): top_k=Some(1) engages the device top-1 (argmax) path.
    // A None temperature is forced to 1.0 downstream (engine add_request), and top_k=None
    // then falls into a full-vocabulary CPU multinomial that idles the GPU -- a per-token
    // sampler tax NOT present in llama-bench, which silently inflated the apparent decode
    // gap. The stochastic branch reproduces that artifact so the tax can be measured.
    let sampling_params = if greedy {
        SamplingParams {
            max_len: Some(n_gen),
            ..SamplingParams::deterministic()
        }
    } else {
        SamplingParams {
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            top_n_logprobs: 0,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            max_len: Some(n_gen),
            stop_toks: None,
            logits_bias: None,
            n_choices: 1,
            dry_params: None,
        }
    };
    let sender = hanzo.get_sender(None).unwrap();
    let (tx, mut rx) = channel(10_000);

    let req = Request::Normal(Box::new(NormalRequest {
        id: hanzo.next_request_id(),
        messages: prompt,
        sampling_params: sampling_params.clone(),
        response: tx,
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
    }));

    // Scored token count for one finished request: prompt tokens for a prefill test, generated
    // (completion) tokens for a decode test -- llama-bench's t/s convention. The response's own
    // rate fields are ignored (see BenchResult); only the token COUNTS are trusted.
    let scored = |u: &Usage| -> usize {
        match test_name {
            TestName::Prompt(_) => u.prompt_tokens,
            TestName::Gen(_) => u.completion_tokens,
        }
    };

    let mut per_rep: Vec<(f64, usize)> = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let t0 = Instant::now();
        for _ in 0..concurrency {
            if sender.send(req.clone()).await.is_err() {
                eprintln!("Receiver disconnected");
            }
        }
        let mut toks = 0usize;
        for _ in 0..concurrency {
            loop {
                match rx.recv().await {
                    Some(Response::AgenticToolCallProgress { .. }) => continue,
                    Some(Response::AgenticToolApprovalRequired { .. }) => continue,
                    Some(Response::File(_)) => continue,
                    Some(Response::Done(res)) => {
                        toks += scored(&res.usage);
                        break;
                    }
                    Some(Response::CompletionDone(res)) => {
                        toks += scored(&res.usage);
                        break;
                    }
                    // A benchmark must surface a failed forward, not panic: report the engine's own
                    // error so the cause (e.g. a device/storage mismatch) is legible.
                    Some(Response::InternalError(e)) => anyhow::bail!("internal error: {e}"),
                    Some(Response::ModelError(e, _)) => anyhow::bail!("model error: {e}"),
                    Some(Response::CompletionModelError(e, _)) => anyhow::bail!("model error: {e}"),
                    Some(Response::ValidationError(e)) => anyhow::bail!("validation error: {e}"),
                    Some(_) => anyhow::bail!("unexpected response variant during benchmark"),
                    None => anyhow::bail!("response channel closed before a terminal response"),
                }
            }
        }
        per_rep.push((t0.elapsed().as_secs_f64(), toks));
    }

    Ok(BenchResult {
        per_rep,
        concurrency,
        test_name,
    })
}

fn uncertain(v: &[f32]) -> UncertainTokSec {
    if v.is_empty() {
        return UncertainTokSec { mean: 0.0, std_dev: 0.0 };
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let variance = v.iter().map(|e| (mean - e).powf(2.)).sum::<f32>() / v.len() as f32;
    UncertainTokSec { mean, std_dev: variance.sqrt() }
}

// Per-stream throughput: scored tokens over the batch wall, divided by concurrency (so t/s is the
// single-stream rate and the throughput column's `t/s * concurrency` recovers the aggregate).
fn get_tok_s(result: &BenchResult) -> UncertainTokSec {
    let rates: Vec<f32> = result
        .per_rep
        .iter()
        .filter(|(secs, toks)| *secs > 0.0 && *toks > 0)
        .map(|(secs, toks)| *toks as f32 / *secs as f32 / result.concurrency as f32)
        .collect();
    uncertain(&rates)
}

fn get_ms_tok(result: &BenchResult) -> UncertainTokSec {
    let ms: Vec<f32> = result
        .per_rep
        .iter()
        .filter(|(secs, toks)| *secs > 0.0 && *toks > 0)
        .map(|(secs, toks)| *secs as f32 * 1000. * result.concurrency as f32 / *toks as f32)
        .collect();
    uncertain(&ms)
}

fn print_usage(model: &str, device: &Device, results: Vec<BenchResult>) {
    let backend = match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        Device::Metal(_) => "Metal",
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => "ROCm",
        #[cfg(feature = "vulkan")]
        Device::Vulkan(_) => "Vulkan",
    };
    let results: Vec<Vec<CellStruct>> = results
        .into_iter()
        .map(|r| {
            vec![
                model.cell(),
                backend.cell(),
                r.test_name.to_string().cell(),
                get_tok_s(&r).cell().justify(Justify::Right),
                get_ms_tok(&r).cell().justify(Justify::Right),
                r.concurrency.cell().justify(Justify::Right),
                (get_tok_s(&r).mean * r.concurrency as f32)
                    .cell()
                    .justify(Justify::Right),
            ]
        })
        .collect();

    let table = results
        .table()
        .title(vec![
            "model".cell().bold(true),
            // "size".cell().bold(true),
            // "params".cell().bold(true),
            "backend".cell().bold(true),
            // "ngl".cell().bold(true),
            "test".cell().bold(true),
            "t/s".cell().bold(true),
            "ms/t".cell().bold(true),
            "concurrency".cell().bold(true),
            "throughput/s".cell().bold(true),
        ])
        .bold(true);
    print_stdout(table).expect("print table");
}

async fn warmup_run(hanzo: Arc<Hanzo>) {
    let sampling_params = SamplingParams {
        max_len: Some(1),
        ..SamplingParams::deterministic()
    };
    let sender = hanzo.get_sender(None).unwrap();
    let (tx, mut rx) = channel(10_000);

    let req = Request::Normal(Box::new(NormalRequest {
        id: hanzo.next_request_id(),
        messages: RequestMessage::Completion {
            text: "Hello!".to_string(),
            echo_prompt: false,
            best_of: None,
        },
        sampling_params: sampling_params.clone(),
        response: tx,
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
    }));

    if sender.send(req.clone()).await.is_err() {
        eprintln!("Receiver disconnected");
    }

    let _ = rx.recv().await;
}

fn parse_cache_type(s: &str) -> Result<PagedCacheType, String> {
    s.parse()
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Model
    #[clap(subcommand)]
    model: ModelSelected,

    /// Integer seed to ensure reproducible random number generation.
    #[arg(short, long)]
    seed: Option<u64>,

    /// Number of prompt tokens to run.
    #[arg(long, short = 'p', default_value_t = 512)]
    n_prompt: usize,

    /// Number of generations tokens to run.
    #[arg(long, short = 'g', default_value_t = 128)]
    n_gen: usize,

    /// Number of concurrent requests to run. Default is 1
    #[clap(short, long, value_parser, value_delimiter = ',')]
    concurrency: Option<Vec<usize>>,

    /// Number of times to repeat each test.
    #[arg(long, short, default_value_t = 5)]
    repetitions: usize,

    /// NOTE: This can be omitted to use automatic device mapping!
    /// Number of device layers to load and run on GPU(s). All others will be on the CPU.
    /// If one GPU is used, then this value should be an integer. Otherwise, it follows the following pattern:
    /// ORD:NUM;... Where ORD is a unique device ordinal and NUM is the number of layers for that device.
    #[arg(short, long, value_parser, value_delimiter = ';')]
    num_device_layers: Option<Vec<String>>,

    /// In-situ quantization to apply.
    #[arg(long = "isq")]
    in_situ_quant: Option<String>,

    /// GPU memory to allocate for KV cache with PagedAttention in MBs.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    #[arg(long = "pa-gpu-mem")]
    paged_attn_gpu_mem: Option<usize>,

    /// Percentage of GPU memory to utilize after allocation of KV cache with PagedAttention, from 0 to 1.
    /// If this is not set and the device is CUDA, it will default to `0.9`.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    #[arg(long = "pa-gpu-mem-usage")]
    paged_attn_gpu_mem_usage: Option<f32>,

    /// Total context length to allocate the KV cache for (total number of tokens which the KV cache can hold).
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    /// This is the default setting, and it defaults to the `max-seq-len` specified in after the model type.
    #[arg(long = "pa-ctxt-len")]
    paged_ctxt_len: Option<usize>,

    /// PagedAttention KV cache type (auto or f8e4m3).
    /// Defaults to `auto`.
    #[arg(long = "pa-cache-type", value_parser = parse_cache_type)]
    cache_type: Option<PagedCacheType>,

    /// Block size (number of tokens per block) for PagedAttention. If this is not set and the device is CUDA, it will default to 32.
    /// PagedAttention is only supported on CUDA and is always automatically activated.
    #[arg(long = "pa-blk-size")]
    paged_attn_block_size: Option<usize>,

    /// Disable PagedAttention on CUDA. Because PagedAttention is already disabled on Metal, this is only applicable on CUDA.
    #[arg(long = "no-paged-attn", default_value_t = false)]
    no_paged_attn: bool,

    /// Enable PagedAttention on Metal. Because PagedAttention is already enabled on CUDA, this is only applicable on Metal.
    #[arg(long = "paged-attn", default_value_t = false)]
    paged_attn: bool,

    /// Emit the raw per-repetition samples (wall seconds, scored tokens) for every test to this
    /// path as JSON. Downstream statistics (mean, 95% CI, coefficient of variation) are computed
    /// from these samples rather than a pre-aggregated mean/stddev, so the reported uncertainty is
    /// auditable and reproducible. The token COUNTS are the engine's own usage figures; the timing
    /// is wall-clock, identical in method to llama-bench, so the two engines compare like-for-like.
    #[arg(long = "json")]
    json: Option<String>,

    /// Decode with the full-vocabulary stochastic sampler (temperature 1.0, no top-k) instead
    /// of greedy argmax. Default is greedy, at sampling parity with llama-bench; this flag
    /// exists only to MEASURE the sampler tax (the artifact greedy avoids), not to report a rate.
    #[arg(long)]
    stochastic: bool,
}

fn backend_str(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        Device::Metal(_) => "Metal",
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => "ROCm",
        #[cfg(feature = "vulkan")]
        Device::Vulkan(_) => "Vulkan",
    }
}

// Serialize one test's raw samples. `per_rep` is the ground truth; every published statistic is a
// pure function of this array, so a reviewer can recompute the board from the JSON alone.
fn result_json(r: &BenchResult) -> serde_json::Value {
    serde_json::json!({
        "test": r.test_name.to_string(),
        "phase": match r.test_name { TestName::Prompt(_) => "prefill", TestName::Gen(_) => "decode" },
        "n": match r.test_name { TestName::Prompt(n) | TestName::Gen(n) => n },
        "concurrency": r.concurrency,
        "per_rep": r.per_rep.iter().map(|(secs, toks)| serde_json::json!([secs, toks])).collect::<Vec<_>>(),
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();
    initialize_logging();

    warn!("hanzo-bench is deprecated. Please use `hanzo bench` from hanzo-cli instead.");

    args.concurrency = Some(args.concurrency.unwrap_or(vec![1]));

    let dtype = get_model_dtype(&args.model)?;
    let auto_device_map_params = get_auto_device_map_params(&args.model)?;

    let max_seq_len = auto_device_map_params.max_seq_len();

    let loader: Box<dyn Loader> = LoaderBuilder::new(args.model).build()?;
    let model_name = loader.get_id();

    // Device selection mirrors the accelerator cascade in hanzo-server-core's `init_device`
    // (vulkan > rocm > metal > cuda/cpu): the bench must run on the same backend it was
    // compiled for, otherwise `--features rocm` silently falls through to CPU.
    #[cfg(feature = "vulkan")]
    let device = Device::new_vulkan(0)?;
    #[cfg(all(feature = "rocm", not(feature = "vulkan")))]
    let device = Device::new_rocm(0)?;
    #[cfg(all(feature = "metal", not(feature = "rocm"), not(feature = "vulkan")))]
    let device = Device::new_metal(0)?;
    #[cfg(all(not(feature = "metal"), not(feature = "rocm"), not(feature = "vulkan")))]
    let device = if hanzo_engine::distributed::use_nccl() {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    if let Some(seed) = args.seed {
        device.set_seed(seed)?;
    }

    let token_source = TokenSource::CacheToken;
    info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        hanzo_ml::utils::with_avx(),
        hanzo_ml::utils::with_neon(),
        hanzo_ml::utils::with_simd128(),
        hanzo_ml::utils::with_f16c()
    );
    info!("Sampling method: penalties -> temperature -> topk -> topp -> minp -> multinomial");
    info!("Model kind is: {}", loader.get_kind().to_string());

    // Parse device mapper
    let mapper = if let Some(device_layers) = args.num_device_layers {
        if device_layers.len() == 1 && device_layers[0].parse::<usize>().is_ok() {
            let layers = device_layers[0].parse::<usize>().unwrap();
            DeviceMapSetting::Map(DeviceMapMetadata::from_num_device_layers(vec![
                DeviceLayerMapMetadata { ordinal: 0, layers },
            ]))
        } else {
            let mut mapping = Vec::new();
            for layer in device_layers {
                let split = layer.splitn(2, ':').collect::<Vec<_>>();
                if split.len() < 2 {
                    panic!("Expected layer to be of format ORD:NUM, got {layer}");
                }
                let ord = split[0]
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("Failed to parse {} as integer.", split[0]));
                let num = split[1]
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("Failed to parse {} as integer.", split[1]));
                for DeviceLayerMapMetadata { ordinal, layers: _ } in &mapping {
                    if *ordinal == ord {
                        panic!("Duplicate ordinal {ord}");
                    }
                }
                mapping.push(DeviceLayerMapMetadata {
                    ordinal: ord,
                    layers: num,
                });
            }
            DeviceMapSetting::Map(DeviceMapMetadata::from_num_device_layers(mapping))
        }
    } else {
        DeviceMapSetting::Auto(auto_device_map_params)
    };

    let no_paged_attn = if device.is_cuda() || hanzo_engine::distributed::use_nccl() {
        args.no_paged_attn
    } else if device.is_metal() {
        !args.paged_attn
    } else {
        // ROCm/Vulkan support PagedAttention (server-core enables it): default off to match
        // llama-bench's contiguous KV, but honor `--paged-attn` so the production path is benchable.
        !args.paged_attn
    };

    let cache_config = match (
        args.paged_attn_block_size,
        args.paged_attn_gpu_mem,
        args.paged_attn_gpu_mem_usage,
        args.paged_ctxt_len,
        paged_attn_supported(),
        no_paged_attn,
    ) {
        (block_size, None, None, None, true, false) => Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::ContextSize(max_seq_len),
            args.cache_type.unwrap_or_default(),
        )?),
        (block_size, None, None, Some(ctxt), true, false) => Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::ContextSize(ctxt),
            args.cache_type.unwrap_or_default(),
        )?),
        (block_size, None, Some(f), None, true, false) => Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::Utilization(f),
            args.cache_type.unwrap_or_default(),
        )?),
        (block_size, Some(m), None, None, true, false) => Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::MbAmount(m),
            args.cache_type.unwrap_or_default(),
        )?),
        (block_size, Some(_m), Some(f), None, true, false) => {
            info!("Both memory size, and usage were specified, defaulting to the usage value.");
            Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::Utilization(f),
                args.cache_type.unwrap_or_default(),
            )?)
        }
        (block_size, Some(_m), None, Some(ctxt), true, false) => {
            info!("All memory size and ctxt len, defaulting to the context len value.");
            Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::ContextSize(ctxt),
                args.cache_type.unwrap_or_default(),
            )?)
        }
        (block_size, None, Some(f), Some(_ctxt), true, false) => {
            info!("Both ctxt len and usage were specified, defaulting to the usage value.");
            Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::Utilization(f),
                args.cache_type.unwrap_or_default(),
            )?)
        }
        (_, _, _, _, _, _) => None,
    };

    let isq = args
        .in_situ_quant
        .as_ref()
        .map(|isq| parse_isq_value(isq, Some(&device)).map_err(|e| anyhow::anyhow!("{e}")))
        .transpose()?;

    let pipeline = loader.load_model_from_hf(
        None,
        token_source,
        &dtype,
        &device,
        false,
        mapper,
        isq,
        cache_config,
    )?;
    info!("Model loaded.");

    let scheduler_config = if cache_config.is_some() {
        // Handle case where we may have device mapping
        if let Some(ref cache_config) = pipeline.lock().await.get_metadata().cache_config {
            SchedulerConfig::PagedAttentionMeta {
                max_num_seqs: *args.concurrency.as_ref().unwrap().iter().max().unwrap(),
                config: cache_config.clone(),
            }
        } else {
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(
                    (*args.concurrency.as_ref().unwrap().iter().max().unwrap())
                        .try_into()
                        .unwrap(),
                ),
            }
        }
    } else {
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(
                (*args.concurrency.as_ref().unwrap().iter().max().unwrap())
                    .try_into()
                    .unwrap(),
            ),
        }
    };
    let hanzo = Builder::new(pipeline, scheduler_config, false, None)
        .with_no_prefix_cache(true)
        .with_disable_eos_stop(true)
        .build()
        .await;

    info!("Starting warmup run.");
    warmup_run(hanzo.clone()).await;
    info!("Finished warmup run.");
    info!("Starting benchmarks.");

    let mut json_records = Vec::new();
    for concurrency in args.concurrency.as_ref().unwrap() {
        let mut results = vec![];
        if args.n_gen > 0 {
            let r = run_bench(
                hanzo.clone(),
                RequestMessage::Completion {
                    text: "Rust".to_string(),
                    echo_prompt: false,
                    best_of: None,
                },
                args.n_gen - 1,
                *concurrency,
                args.repetitions,
                TestName::Gen(args.n_gen),
                !args.stochastic,
            )
            .await?;
            results.push(r);
        }

        if args.n_prompt > 0 {
            let tks = (1000..1000 + args.n_prompt as u32).collect();
            let r = run_bench(
                hanzo.clone(),
                RequestMessage::CompletionTokens(tks),
                1,
                *concurrency,
                args.repetitions,
                TestName::Prompt(args.n_prompt),
                !args.stochastic,
            )
            .await?;

            results.push(r);
        }

        if args.json.is_some() {
            json_records.extend(results.iter().map(result_json));
        }
        print_usage(&model_name, &device, results);
    }

    if let Some(path) = &args.json {
        let doc = serde_json::json!({
            "engine_version": env!("CARGO_PKG_VERSION"),
            "backend": backend_str(&device),
            "sampler": if args.stochastic { "stochastic-temp1-fullvocab" } else { "greedy-argmax" },
            "model_id": model_name,
            "n_prompt": args.n_prompt,
            "n_gen": args.n_gen,
            "repetitions": args.repetitions,
            "results": json_records,
        });
        std::fs::write(path, serde_json::to_string_pretty(&doc)?)?;
        info!("Wrote raw samples to {path}");
    }

    Ok(())
}
