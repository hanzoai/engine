use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use hanzo_ml::{Device, Tensor};
use hanzo_quant::IsqType;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use indicatif::MultiProgress;
use rand_isaac::Isaac64Rng;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use crate::lora::Ordering;
use crate::models::llada::{self, GenParams};
use crate::pipeline::chat_template::{calculate_eos_tokens, GenerationConfig};
use crate::pipeline::llg::build_llg_factory;
use crate::pipeline::sampling::finish_or_add_toks_to_seq;
use crate::pipeline::{
    get_chat_template, get_model_paths, get_xlora_paths, AnyMoePipelineMixin, Cache,
    CacheBackendMetadata, CacheManagerMixin, ChatTemplate, EitherCache, ForwardInputsResult,
    GeneralMetadata, IsqPipelineMixin, Loader, LocalModelPaths, MetadataMixin, Modalities,
    ModelCategory, ModelKind, ModelPaths, Pipeline, PreProcessingMixin, Processor,
    SupportedModality, TokenSource,
};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sampler::Logprobs;
use crate::sequence::{Sequence, SequenceState, StopReason};
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::tokens::get_token;
use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
use crate::{get_paths, DeviceMapSetting, PagedAttentionConfig, TryIntoDType};

const DEFAULT_GEN_LENGTH: usize = 128;
const DEFAULT_BLOCK_LENGTH: usize = 32;

pub struct DiffusionLmPipeline {
    model: llada::Model,
    tokenizer: Arc<Tokenizer>,
    chat_template: Arc<ChatTemplate>,
    metadata: Arc<GeneralMetadata>,
    model_id: String,
    dummy_cache: EitherCache,
    // seq id -> remaining generated tokens (denoised on the prompt step, drained one per step).
    buffers: HashMap<usize, VecDeque<u32>>,
}

pub struct DiffusionLmLoader {
    pub model_id: String,
    pub tokenizer_json: Option<String>,
    pub chat_template: Option<String>,
    pub xlora_model_id: Option<String>,
    pub lora_adapter_ids: Option<Vec<String>>,
    pub xlora_order: Option<Ordering>,
}

impl DiffusionLmLoader {
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            tokenizer_json: None,
            chat_template: None,
            xlora_model_id: None,
            lora_adapter_ids: None,
            xlora_order: None,
        }
    }
}

impl Loader for DiffusionLmLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths!(
            LocalModelPaths,
            &token_source,
            revision.clone(),
            self,
            None,
            None,
            silent,
            false
        );
        self.load_model_from_path(
            &paths?,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        _mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        if in_situ_quant.is_some() {
            anyhow::bail!("ISQ is not supported for diffusion LMs.");
        }
        if paged_attn_config.is_some() {
            anyhow::bail!("PagedAttention is not supported for diffusion LMs (no KV cache).");
        }

        let config: llada::Config =
            serde_json::from_str(&std::fs::read_to_string(paths.get_config_filename())?)?;
        let dtype = dtype.try_into_dtype(&[device])?;

        let vb = from_mmaped_safetensors(
            paths.get_weight_filenames().to_vec(),
            Vec::new(),
            Some(dtype),
            device,
            vec![None],
            silent,
            None,
            |_| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )?;
        let model = llada::Model::new(&config, vb, device, &Arc::new(MultiProgress::new()))?;

        let tokenizer = get_tokenizer(paths.get_tokenizer_filename(), None)?;
        let gen_conf: Option<GenerationConfig> = paths
            .get_gen_conf_filename()
            .map(|f| serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap());
        let chat_template = get_chat_template(paths, None, None, None, None);
        let llg_factory = build_llg_factory(tokenizer.clone())?;
        let eos = calculate_eos_tokens(&chat_template, gen_conf.as_ref(), &tokenizer);

        let max_seq_len = model.max_seq_len();
        Ok(Arc::new(Mutex::new(DiffusionLmPipeline {
            model,
            tokenizer: tokenizer.into(),
            chat_template: Arc::new(chat_template),
            model_id: self.model_id.clone(),
            dummy_cache: EitherCache::Full(Cache::new(0, false)),
            buffers: HashMap::new(),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: Some(llg_factory),
                is_xlora: false,
                no_prefix_cache: true,
                num_hidden_layers: config.n_layers,
                eos_tok: eos,
                kind: ModelKind::Normal,
                no_kv_cache: true,
                activation_dtype: dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Text],
                },
            }),
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::Normal
    }
}

impl DiffusionLmPipeline {
    // gen_length = requested completion (or default) rounded up to a block multiple and capped by the
    // remaining context; steps = gen_length (one token unmasked per traversal, LLaDA's default).
    fn gen_params_for(&self, seq: &Sequence) -> GenParams {
        let block = DEFAULT_BLOCK_LENGTH;
        let requested = seq.max_len().unwrap_or(DEFAULT_GEN_LENGTH).max(1);
        let room = self
            .metadata
            .max_seq_len
            .saturating_sub(seq.get_toks().len())
            .max(block);
        let gen = (requested.div_ceil(block) * block)
            .min(room / block * block)
            .max(block);
        GenParams {
            gen_length: gen,
            steps: gen,
            block_length: block,
        }
    }
}

impl PreProcessingMixin for DiffusionLmPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(crate::pipeline::processing::BasicProcessor)
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for DiffusionLmPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!("Diffusion LMs do not support ISQ.")
    }
}

impl CacheManagerMixin for DiffusionLmPipeline {
    fn clone_in_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn clone_out_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn set_none_cache(
        &self,
        _seqs: &mut [&mut Sequence],
        _reset_non_granular: bool,
        _modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
    }
    fn cache(&self) -> &EitherCache {
        &self.dummy_cache
    }
}

impl MetadataMixin for DiffusionLmPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {}
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn device_mapper(&self) -> Option<&dyn crate::device_map::DeviceMapper> {
        None
    }
}

#[async_trait::async_trait]
impl Pipeline for DiffusionLmPipeline {
    fn forward_inputs(
        &mut self,
        _inputs: Box<dyn Any>,
        _return_raw_logits: bool,
    ) -> std::result::Result<ForwardInputsResult, hanzo_ml::Error> {
        hanzo_ml::bail!("DiffusionLmPipeline drives generation from `step`, not `forward_inputs`.")
    }

    // One denoise pass fills a per-sequence token buffer on the prompt step; each step emits the next
    // buffered token through the shared AR finalizer (streaming/stop/detokenization reused verbatim).
    async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        _return_raw_logits: bool,
        prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        _backend_metadata: CacheBackendMetadata,
    ) -> std::result::Result<Duration, hanzo_ml::Error> {
        let eos = self.metadata.eos_tok.clone();
        let mut exec = Duration::ZERO;

        if is_prompt {
            for seq in input_seqs.iter() {
                let prompt = seq.get_toks().to_vec();
                let params = self.gen_params_for(seq);
                let start = Instant::now();
                let toks = self.model.generate(&prompt, &params)?;
                exec += start.elapsed();
                self.buffers.insert(*seq.id(), toks.into());
            }
        }

        for seq in input_seqs.iter_mut() {
            let next = self.buffers.get_mut(seq.id()).and_then(|b| b.pop_front());
            match next {
                Some(token) => {
                    let logprobs = Logprobs {
                        token,
                        logprob: 0.0,
                        bytes: None,
                        top_logprobs: None,
                    };
                    finish_or_add_toks_to_seq(
                        &*self,
                        prefix_cacher,
                        seq,
                        logprobs,
                        Some(eos.as_slice()),
                        false,
                    )
                    .await?;
                }
                None => {
                    seq.set_state(SequenceState::Done(StopReason::Length(
                        self.metadata.max_seq_len,
                    )));
                }
            }
        }
        input_seqs
            .iter()
            .filter(|s| self.buffers.get(s.id()).is_some_and(VecDeque::is_empty))
            .map(|s| *s.id())
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|id| {
                self.buffers.remove(&id);
            });

        Ok(exec)
    }

    async fn sample_causal_gen(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: Vec<Tensor>,
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> std::result::Result<(), hanzo_ml::Error> {
        hanzo_ml::bail!("`sample_causal_gen` is not used by DiffusionLmPipeline.")
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }
}

impl AnyMoePipelineMixin for DiffusionLmPipeline {}
