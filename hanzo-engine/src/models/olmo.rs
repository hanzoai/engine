#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{Embedding, LayerNorm, Module};
use hanzo_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding, Activation, CausalMasker, Mlp, RotaryEmbedding, Sdpa},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalLoadingMetadata,
        NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

// OLMo's `OlmoLayerNorm` is non-parametric (no weight, no bias) and always computes in f32 with a
// fixed epsilon; the checkpoint therefore carries no norm tensors.
const NORM_EPS: f64 = 1e-5;

serde_default_fn!(bool, word_emb_default, false);
serde_default_fn!(bool, attention_bias_default, false);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Config {
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub clip_qkv: Option<f64>,
    #[serde(default = "attention_bias_default")]
    pub attention_bias: bool,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

/// A non-parametric layer norm (ones weight, no bias) matching HF `OlmoLayerNorm`.
fn olmo_norm(size: usize, dtype: DType, device: &Device) -> Result<LayerNorm> {
    Ok(LayerNorm::new_no_bias(
        Tensor::ones((size,), dtype, device)?,
        NORM_EPS,
    ))
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    clip_qkv: Option<f64>,
    rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<hanzo_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden_sz / num_heads;
        let b = cfg.attention_bias;
        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            b,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard = hanzo_quant::compute_kv_shard(num_kv_heads, head_dim, comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            b,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            b,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            b,
            comm,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            clip_qkv: cfg.clip_qkv,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: hanzo_quant::compute_n_kv_groups(num_kv_heads, num_heads, comm)?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let (mut q, mut k, mut v) =
            crate::ops::qkv_projections(xs, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        if let Some(clip) = self.clip_qkv {
            q = q.clamp(-clip, clip)?;
            k = k.clamp(-clip, clip)?;
            v = v.clamp(-clip, clip)?;
        }

        let (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        let rope_positions = ctx
            .rope_positions(q.device())?
            .ok_or_else(|| hanzo_ml::Error::msg("missing RoPE positions"))?;
        let (q, k) = self.rotary_emb.forward_positions(&q, &k, rope_positions)?;
        let metadata = ctx.paged_layer(layer_idx);

        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(ctx.flash_params()),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(!matches!(attention_mask, AttentionMask::None));
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(ctx.flash_params()),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(ctx.flash_params()),
                    &self.sdpa_params,
                )?
            }
        };

        attn_output = if !matches!(attention_mask, AttentionMask::None) {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        self.o_proj.forward(&attn_output)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<hanzo_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
            comm,
        )?;
        let mlp = Mlp::new(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_act,
            comm,
        )?;
        let device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or(Device::Cpu);
        let input_layernorm = olmo_norm(cfg.hidden_size, vb.dtype(), &device)?;
        let post_attention_layernorm = olmo_norm(cfg.hidden_size, vb.dtype(), &device)?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = (self
            .self_attn
            .forward(&xs, attention_mask, kv_cache, ctx, layer_idx)?
            + residual)?;
        let residual = &xs;
        let xs = (self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?
            + residual)?;
        Ok(xs)
    }
}

pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    lm_head: Arc<dyn QuantMethod>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?),
            );
        }

        let vb_l = vb_m.pp("layers");
        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(
                    PagedAttention::new(head_dim, device, None)
                        .expect("PagedAttention creation failed"),
                ),
            };
            let comm = mapper
                .get_comm_for(layer_idx)
                .expect("Failed to get comm for layer");
            DecoderLayer::new(
                rotary_emb,
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                &comm,
            )
        })?;
        let norm = olmo_norm(
            cfg.hidden_size,
            vb_m.dtype(),
            &normal_loading_metadata.real_device,
        )?;
        let lm_head = if cfg.tie_word_embeddings {
            ReplicatedLayer::from_linear(hanzo_nn::Linear::new(
                mapper.cast_nm_device(
                    embed_tokens.embeddings(),
                    normal_loading_metadata.loading_isq,
                )?,
                None,
            ))?
        } else {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: None,
                k_head_dim: head_dim,
                v_head_dim: head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;

        let cache = &mut self.cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig::default(),
        )?;
        let attention_mask = if ctx.is_first_prompt_chunk() {
            attention_mask
        } else {
            AttentionMask::None
        };
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(&xs, &attention_mask.get(xs.device()), &mut cache[i], ctx, i)?;
        }
        let xs = xs.to_device(&self.device)?.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        self.lm_head.forward(&xs)
    }
}

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.q_proj, Some(i)));
            tensors.push((&mut layer.self_attn.k_proj, Some(i)));
            tensors.push((&mut layer.self_attn.v_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            tensors.extend(
                layer
                    .mlp
                    .get_isq_layers()
                    .into_iter()
                    .map(|m| (m, Some(i)))
                    .collect::<Vec<_>>(),
            );
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        // OLMo's norms are non-parametric, so only the embeddings are residual (non-ISQ) tensors.
        let uvb = UnVarBuilder::new();
        uvb.pp("model").pp("embed_tokens").add(&self.embed_tokens);
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn supports_cuda_decode_graphs(&self) -> bool {
        true
    }
}

impl AnyMoeBaseModelMixin for Model {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::ModelForwardContext;
    use crate::DeviceMapSetting;
    use hanzo_ml::safetensors;
    use indicatif::MultiProgress;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    // Byte-for-byte the arch block of the real `allenai/OLMo-1B-hf` config.json (tie + non-parametric
    // norm + null clip_qkv + silu SwiGLU). Embedded so the test is hermetic.
    const REAL_CONFIG: &str = r#"{
        "architectures": ["OlmoForCausalLM"],
        "attention_bias": false,
        "clip_qkv": null,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "max_position_embeddings": 2048,
        "model_type": "olmo",
        "num_attention_heads": 16,
        "num_hidden_layers": 16,
        "num_key_value_heads": 16,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
        "vocab_size": 50304
    }"#;

    // Tiny config exercising the same wiring (GQA + non-parametric norm + clip_qkv + tied lm_head).
    const TOY_CONFIG: &str = r#"{
        "hidden_act": "silu",
        "hidden_size": 32,
        "intermediate_size": 64,
        "vocab_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rope_theta": 10000.0,
        "max_position_embeddings": 32,
        "clip_qkv": 8.0,
        "tie_word_embeddings": true
    }"#;

    fn loading_metadata(device: &Device, num_layers: usize) -> Result<NormalLoadingMetadata> {
        let mapper = DeviceMapSetting::dummy().into_mapper(
            num_layers,
            device,
            None,
            std::slice::from_ref(device),
        )?;
        Ok(NormalLoadingMetadata {
            mapper,
            loading_isq: false,
            real_device: device.clone(),
            multi_progress: Arc::new(MultiProgress::new()),
            matformer_slicing_config: None,
        })
    }

    // Random-init checkpoint written to `dir` with exactly the real OLMo tensor keys (no norm/lm_head
    // tensors — they are non-parametric / tied).
    fn write_checkpoint(cfg: &Config, dir: &std::path::Path) -> Result<std::path::PathBuf> {
        let mut rng = StdRng::seed_from_u64(0x01);
        let mut t = |rows: usize, cols: usize| -> Result<Tensor> {
            let data: Vec<f32> = (0..rows * cols)
                .map(|_| rng.random_range(-0.08f32..0.08))
                .collect();
            Tensor::from_vec(data, (rows, cols), &Device::Cpu)
        };
        let h = cfg.hidden_size;
        let head_dim = h / cfg.num_attention_heads;
        let kv = cfg.num_key_value_heads * head_dim;
        let mut ts = std::collections::HashMap::new();
        ts.insert(
            "model.embed_tokens.weight".to_string(),
            t(cfg.vocab_size, h)?,
        );
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            ts.insert(format!("{p}.self_attn.q_proj.weight"), t(h, h)?);
            ts.insert(format!("{p}.self_attn.k_proj.weight"), t(kv, h)?);
            ts.insert(format!("{p}.self_attn.v_proj.weight"), t(kv, h)?);
            ts.insert(format!("{p}.self_attn.o_proj.weight"), t(h, h)?);
            ts.insert(
                format!("{p}.mlp.gate_proj.weight"),
                t(cfg.intermediate_size, h)?,
            );
            ts.insert(
                format!("{p}.mlp.up_proj.weight"),
                t(cfg.intermediate_size, h)?,
            );
            ts.insert(
                format!("{p}.mlp.down_proj.weight"),
                t(h, cfg.intermediate_size)?,
            );
        }
        let path = dir.join("model.safetensors");
        safetensors::save(&ts, &path)?;
        Ok(path)
    }

    fn load_toy(path: &std::path::Path, cfg: &Config, device: &Device) -> Result<Model> {
        let vb = crate::utils::varbuilder_utils::from_mmaped_safetensors(
            vec![path.to_path_buf()],
            Vec::new(),
            Some(DType::F32),
            device,
            vec![None],
            true,
            None,
            |_| true,
            Arc::new(|_| crate::utils::varbuilder_utils::DeviceForLoadTensor::Base),
        )?;
        let metadata = loading_metadata(device, cfg.num_hidden_layers)?;
        Model::new(cfg, vb, true, metadata, AttentionImplementation::Eager)
    }

    fn forward_ids(model: &Model, ids: &[u32], device: &Device) -> Result<Tensor> {
        let seq_len = ids.len();
        let input_ids = Tensor::from_vec(ids.to_vec(), (1, seq_len), device)?;
        let seqlen_offsets = vec![0usize];
        let context_lens = vec![(0usize, seq_len)];
        let position_ids: Vec<usize> = (0..seq_len).collect();
        let flash_params = FlashParams::empty(true);
        let mut ctx = ModelForwardContext::new(
            &seqlen_offsets,
            &context_lens,
            &position_ids,
            None,
            &flash_params,
        );
        model.forward(&input_ids, &mut ctx)
    }

    #[test]
    fn olmo_config_parses() {
        let cfg: Config = serde_json::from_str(REAL_CONFIG).unwrap();
        assert_eq!(cfg.num_hidden_layers, 16);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 16);
        assert!(cfg.tie_word_embeddings);
        assert!(!cfg.attention_bias);
        assert!(cfg.clip_qkv.is_none());
    }

    #[test]
    fn olmo_registry_dispatch() {
        use crate::pipeline::NormalLoaderType;
        assert_eq!(
            NormalLoaderType::from_causal_lm_name("OlmoForCausalLM").unwrap(),
            NormalLoaderType::Olmo
        );
    }

    #[test]
    fn olmo_toy_forward_causal() -> Result<()> {
        let device = Device::Cpu;
        let cfg: Config = serde_json::from_str(TOY_CONFIG).unwrap();
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::wrap)?;
        let path = write_checkpoint(&cfg, dir.path())?;

        let model = load_toy(&path, &cfg, &device)?;
        let logits = forward_ids(&model, &[1u32, 2, 3, 4, 5], &device)?;
        assert_eq!(logits.dims3()?, (1, 5, cfg.vocab_size));
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(flat.iter().all(|x| x.is_finite()), "logits must be finite");

        // Causality: changing the LAST token must not change logits at earlier positions. A fresh
        // model (fresh KV cache) is used per sequence.
        let model_a = load_toy(&path, &cfg, &device)?;
        let a = forward_ids(&model_a, &[1u32, 2, 3, 4, 5], &device)?;
        let model_b = load_toy(&path, &cfg, &device)?;
        let b = forward_ids(&model_b, &[1u32, 2, 3, 4, 9], &device)?;
        let a_past = a.narrow(1, 0, 4)?.flatten_all()?.to_vec1::<f32>()?;
        let b_past = b.narrow(1, 0, 4)?.flatten_all()?.to_vec1::<f32>()?;
        for (x, y) in a_past.iter().zip(b_past.iter()) {
            assert!((x - y).abs() < 1e-4, "past logits changed: {x} vs {y}");
        }
        Ok(())
    }
}
