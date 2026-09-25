//! `hanzo-engine bench`: the count-based instrument, on the model the server would build.

use anyhow::Result;
use hanzo_bench::measure::{self, Spec};
use hanzo_server_core::server::ServerBuilder;

use crate::args::{BenchRuntimeOptions, GlobalOptions, ModelType};

use super::serve::{
    apply_quant_resolution, convert_to_model_selected, extract_device_settings,
    extract_isq_setting, extract_paged_attn_settings,
};

/// The model's id, for the table and the samples file.
fn model_id(model_type: &ModelType) -> String {
    match model_type {
        ModelType::Auto { model, .. }
        | ModelType::Text { model, .. }
        | ModelType::Multimodal { model, .. }
        | ModelType::Diffusion { model, .. }
        | ModelType::Speech { model, .. }
        | ModelType::Embedding { model, .. } => model.model_id.clone(),
        ModelType::Animation { model_id, .. } => model_id.clone(),
    }
}

pub struct BenchArgs {
    pub prompt_len: Vec<usize>,
    pub gen_len: usize,
    pub repetitions: usize,
    pub concurrency: Vec<usize>,
    pub stochastic: bool,
    pub json: Option<std::path::PathBuf>,
}

pub async fn run_bench(
    mut model_type: ModelType,
    runtime: BenchRuntimeOptions,
    global: GlobalOptions,
    args: BenchArgs,
) -> Result<()> {
    hanzo_engine::initialize_logging();
    let spec = Spec {
        model_id: model_id(&model_type),
        n_prompt: args.prompt_len,
        n_gen: args.gen_len,
        concurrency: args.concurrency,
        repetitions: args.repetitions,
        stochastic: args.stochastic,
        json: args.json,
    };
    if spec.concurrency.is_empty() {
        anyhow::bail!("--concurrency needs at least one value");
    }

    let matformer = runtime.matformer_selection();
    apply_quant_resolution(&mut model_type, &global.token_source, &matformer).await?;
    let model_selected = convert_to_model_selected(&model_type, &matformer)?;
    let (draft_max_seq_len, draft_max_batch_size) = model_selected.max_dims();
    let (
        paged_attn,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_block_size,
        paged_cache_type,
    ) = extract_paged_attn_settings(&model_type);
    let (cpu, device_layers) = extract_device_settings(&model_type);
    let isq = extract_isq_setting(&model_type);

    let hanzo = ServerBuilder::new()
        .with_model(model_selected)
        .with_max_seqs(spec.concurrency.iter().copied().max().unwrap_or(1))
        .with_no_kv_cache(runtime.no_kv_cache)
        .with_token_source(global.token_source)
        .with_interactive_mode(false)
        .with_prefix_cache_n(0)
        .with_disable_eos_stop(true)
        .with_mtp_config_optional(runtime.mtp_config())
        .with_draft_model_optional(
            runtime.draft_model_selected(draft_max_seq_len, draft_max_batch_size),
            runtime.gamma(),
        )
        .with_prompt_lookup_optional(runtime.prompt_lookup_ngram, runtime.gamma())
        .with_dflash_optional(runtime.dflash.clone(), runtime.dflash_block_size)
        .set_paged_attn(paged_attn)
        .with_cpu(cpu)
        .with_seed_optional(global.seed)
        .with_num_device_layers_optional(device_layers)
        .with_in_situ_quant_optional(isq)
        .with_paged_attn_gpu_mem_optional(paged_attn_gpu_mem)
        .with_paged_attn_gpu_mem_usage_optional(paged_attn_gpu_mem_usage)
        .with_paged_ctxt_len_optional(paged_ctxt_len)
        .with_paged_attn_block_size_optional(paged_attn_block_size)
        .with_paged_attn_cache_type(paged_cache_type)
        .build()
        .await?;

    measure::run(&hanzo, &spec).await?;

    // The samples are written. A GPU runtime's exit-time teardown can abort after the fact
    // (ROCm on gfx1151 does), and a harness would read that as a failed run.
    use std::io::Write;
    std::io::stdout().flush()?;
    std::process::exit(0)
}
