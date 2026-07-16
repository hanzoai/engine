#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use hanzo_ml::{Device, Module, Result, Tensor};
use hanzo_nn::{Embedding, LayerNorm};
use hanzo_quant::{
    QuantMethod, QuantMethodConfig, QuantizedConfig, ShardedVarBuilder, UnquantLinear,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding, layer_norm, Activation, CausalMasker, Sdpa},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalLoadingMetadata,
        NormalModel,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

fn default_act() -> Activation {
    Activation::NewGelu
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_positions: usize,
    #[serde(default)]
    pub n_inner: Option<usize>,
    #[serde(default = "default_act")]
    pub activation_function: Activation,
    pub layer_norm_epsilon: f64,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    fn intermediate_size(&self) -> usize {
        self.n_inner.unwrap_or(4 * self.n_embd)
    }
}

/// GPT-2 stores its projections as `Conv1D` layers whose weight is transposed relative to
/// `nn.Linear` (shape `[in, out]`). We load the raw weight, transpose it into the `[out, in]`
/// layout the engine's linear layers expect, and wrap it so ISQ still applies.
fn conv1d(in_f: usize, out_f: usize, vb: ShardedVarBuilder) -> Result<Arc<dyn QuantMethod>> {
    let w = vb.get((in_f, out_f), "weight")?.t()?.contiguous()?;
    let b = vb.get(out_f, "bias")?;
    Ok(Arc::new(UnquantLinear::new(
        QuantMethodConfig::Unquantized(hanzo_nn::Linear::new(w, Some(b))),
    )?))
}

struct Attention {
    c_attn: Arc<dyn QuantMethod>,
    c_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let hidden_size = cfg.n_embd;
        let num_heads = cfg.n_head;
        let head_dim = hidden_size / num_heads;
        let c_attn = conv1d(hidden_size, 3 * hidden_size, vb.pp("c_attn"))?;
        let c_proj = conv1d(hidden_size, hidden_size, vb.pp("c_proj"))?;
        Ok(Self {
            c_attn,
            c_proj,
            num_heads,
            head_dim,
            hidden_size,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: 1,
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

        let qkv = self.c_attn.forward(xs)?;
        let q = qkv.narrow(2, 0, self.hidden_size)?;
        let k = qkv.narrow(2, self.hidden_size, self.hidden_size)?;
        let v = qkv.narrow(2, 2 * self.hidden_size, self.hidden_size)?;

        let (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q.contiguous()?, k.contiguous()?, v.contiguous()?)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

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
        self.c_proj.forward(&attn_output)
    }
}

struct Mlp {
    c_fc: Arc<dyn QuantMethod>,
    c_proj: Arc<dyn QuantMethod>,
    act: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden_size = cfg.n_embd;
        let inner = cfg.intermediate_size();
        Ok(Self {
            c_fc: conv1d(hidden_size, inner, vb.pp("c_fc"))?,
            c_proj: conv1d(inner, hidden_size, vb.pp("c_proj"))?,
            act: cfg.activation_function,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.c_proj
            .forward(&self.c_fc.forward(xs)?.apply(&self.act)?)
    }
}

struct Block {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let attn = Attention::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("attn"), loading_isq),
            paged_attn,
        )?;
        let mlp = Mlp::new(cfg, mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq))?;
        let ln_1 = layer_norm(
            cfg.n_embd,
            cfg.layer_norm_epsilon,
            mapper.set_device(layer_idx, vb.pp("ln_1"), false),
        )?;
        let ln_2 = layer_norm(
            cfg.n_embd,
            cfg.layer_norm_epsilon,
            mapper.set_device(layer_idx, vb.pp("ln_2"), false),
        )?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
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
        let xs = self.ln_1.forward(xs)?;
        let xs = (self
            .attn
            .forward(&xs, attention_mask, kv_cache, ctx, layer_idx)?
            + residual)?;
        let residual = &xs;
        let xs = (self.mlp.forward(&self.ln_2.forward(&xs)?)? + residual)?;
        Ok(xs)
    }
}

pub struct Model {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
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
        _is_gptx: bool,
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

        let wte = embedding(
            cfg.vocab_size,
            cfg.n_embd,
            mapper.set_nm_device(vb.pp("wte"), false),
            &cfg.quantization_config,
        )?;
        let wpe = embedding(
            cfg.n_positions,
            cfg.n_embd,
            mapper.set_nm_device(vb.pp("wpe"), false),
            &cfg.quantization_config,
        )?;
        let head_dim = cfg.n_embd / cfg.n_head;

        let vb_h = vb.pp("h");
        let blocks: Vec<Block> = NiceProgressBar::<_, 'b'>(
            0..cfg.n_layer,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(
                    PagedAttention::new(head_dim, device, None)
                        .expect("PagedAttention creation failed"),
                ),
            };
            Block::new(
                cfg,
                vb_h.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
            )
        })?;
        let ln_f = layer_norm(
            cfg.n_embd,
            cfg.layer_norm_epsilon,
            mapper.set_nm_device(vb.pp("ln_f"), false),
        )?;
        // GPT-2 ties the output head to the token embedding.
        let lm_head =
            mapper.cast_nm_device(wte.embeddings(), normal_loading_metadata.loading_isq)?;
        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head: Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                hanzo_nn::Linear::new(lm_head, None),
            ))?),
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::new(cfg.n_layer, cfg.n_positions)),
            max_seq_len: cfg.n_positions,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.n_positions,
                num_layers: cfg.n_layer,
                hidden_size: cfg.n_embd,
                num_attn_heads: cfg.n_head,
                num_kv_heads: cfg.n_head,
                sliding_window: None,
                k_head_dim: head_dim,
                v_head_dim: head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
        })
    }

    fn position_ids(&self, b_sz: usize, q_len: usize, ctx: &ModelForwardContext<'_>) -> Vec<u32> {
        let offsets = ctx.seqlen_offsets();
        let mut positions = Vec::with_capacity(b_sz * q_len);
        for b in 0..b_sz {
            let offset = offsets.get(b).copied().unwrap_or(0);
            for j in 0..q_len {
                positions.push((offset + j) as u32);
            }
        }
        positions
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let (b_sz, q_len) = input_ids.dims2()?;
        let tok = self.wte.forward(input_ids)?;
        let positions = self.position_ids(b_sz, q_len, ctx);
        let position_ids = Tensor::from_vec(positions, (b_sz, q_len), tok.device())?;
        let mut xs = (tok + self.wpe.forward(&position_ids)?)?;

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

        for (i, block) in self.blocks.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = block.forward(&xs, &attention_mask.get(xs.device()), &mut cache[i], ctx, i)?;
        }
        let xs = xs.to_device(&self.device)?.apply(&self.ln_f)?;
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
        for (i, block) in self.blocks.iter_mut().enumerate() {
            tensors.push((&mut block.attn.c_attn, Some(i)));
            tensors.push((&mut block.attn.c_proj, Some(i)));
            tensors.push((&mut block.mlp.c_fc, Some(i)));
            tensors.push((&mut block.mlp.c_proj, Some(i)));
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("wte").add(&self.wte);
        uvb.pp("wpe").add(&self.wpe);
        uvb.pp("ln_f").add(&self.ln_f);
        for (i, block) in self.blocks.iter().enumerate() {
            let uvb_l = uvb.pp("h").pp(i);
            uvb_l.pp("ln_1").add(&block.ln_1);
            uvb_l.pp("ln_2").add(&block.ln_2);
        }
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
    use hanzo_ml::{safetensors, DType};
    use indicatif::MultiProgress;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    // The arch block of the real `openai-community/gpt2` config.json (gelu_new + learned positions
    // + tied head). Embedded for a hermetic test.
    const REAL_CONFIG: &str = r#"{
        "activation_function": "gelu_new",
        "architectures": ["GPT2LMHeadModel"],
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": 1024,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "n_positions": 1024,
        "vocab_size": 50257
    }"#;

    const TOY_CONFIG: &str = r#"{
        "activation_function": "gelu_new",
        "layer_norm_epsilon": 1e-05,
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 2,
        "n_positions": 32,
        "vocab_size": 64
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

    // Random-init checkpoint with the exact real GPT-2 tensor keys, including the Conv1D `[in, out]`
    // weight layout (so the loader's transpose is exercised).
    fn write_checkpoint(cfg: &Config, dir: &std::path::Path) -> Result<std::path::PathBuf> {
        let mut rng = StdRng::seed_from_u64(0x67707432);
        let mut t = |shape: (usize, usize)| -> Result<Tensor> {
            let data: Vec<f32> = (0..shape.0 * shape.1)
                .map(|_| rng.random_range(-0.08f32..0.08))
                .collect();
            Tensor::from_vec(data, shape, &Device::Cpu)
        };
        let one = |n: usize| Tensor::ones((n,), DType::F32, &Device::Cpu);
        let zero = |n: usize| Tensor::zeros((n,), DType::F32, &Device::Cpu);
        let h = cfg.n_embd;
        let inner = cfg.intermediate_size();
        let mut ts = std::collections::HashMap::new();
        ts.insert("wte.weight".to_string(), t((cfg.vocab_size, h))?);
        ts.insert("wpe.weight".to_string(), t((cfg.n_positions, h))?);
        ts.insert("ln_f.weight".to_string(), one(h)?);
        ts.insert("ln_f.bias".to_string(), zero(h)?);
        for i in 0..cfg.n_layer {
            let p = format!("h.{i}");
            ts.insert(format!("{p}.ln_1.weight"), one(h)?);
            ts.insert(format!("{p}.ln_1.bias"), zero(h)?);
            ts.insert(format!("{p}.ln_2.weight"), one(h)?);
            ts.insert(format!("{p}.ln_2.bias"), zero(h)?);
            ts.insert(format!("{p}.attn.c_attn.weight"), t((h, 3 * h))?);
            ts.insert(format!("{p}.attn.c_attn.bias"), zero(3 * h)?);
            ts.insert(format!("{p}.attn.c_proj.weight"), t((h, h))?);
            ts.insert(format!("{p}.attn.c_proj.bias"), zero(h)?);
            ts.insert(format!("{p}.mlp.c_fc.weight"), t((h, inner))?);
            ts.insert(format!("{p}.mlp.c_fc.bias"), zero(inner)?);
            ts.insert(format!("{p}.mlp.c_proj.weight"), t((inner, h))?);
            ts.insert(format!("{p}.mlp.c_proj.bias"), zero(h)?);
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
        let metadata = loading_metadata(device, cfg.n_layer)?;
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
    fn gpt2_config_parses() {
        let cfg: Config = serde_json::from_str(REAL_CONFIG).unwrap();
        assert_eq!(cfg.n_layer, 12);
        assert_eq!(cfg.n_embd, 768);
        assert_eq!(cfg.n_head, 12);
        assert_eq!(cfg.vocab_size, 50257);
        assert_eq!(cfg.intermediate_size(), 3072);
        assert!(matches!(cfg.activation_function, Activation::NewGelu));
    }

    #[test]
    fn gpt2_registry_dispatch() {
        use crate::pipeline::NormalLoaderType;
        assert_eq!(
            NormalLoaderType::from_causal_lm_name("GPT2LMHeadModel").unwrap(),
            NormalLoaderType::GPT2
        );
    }

    #[test]
    fn gpt2_toy_forward_causal() -> Result<()> {
        let device = Device::Cpu;
        let cfg: Config = serde_json::from_str(TOY_CONFIG).unwrap();
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::wrap)?;
        let path = write_checkpoint(&cfg, dir.path())?;

        let model = load_toy(&path, &cfg, &device)?;
        let logits = forward_ids(&model, &[1u32, 2, 3, 4, 5], &device)?;
        assert_eq!(logits.dims3()?, (1, 5, cfg.vocab_size));
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(flat.iter().all(|x| x.is_finite()), "logits must be finite");

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
