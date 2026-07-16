#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use hanzo_ml::{DType, Device, Module, Result, Tensor};
use hanzo_nn::LayerNorm;
use hanzo_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding, layer_norm, Activation, CausalMasker, RotaryEmbedding, Sdpa},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalLoadingMetadata,
        NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, bias_default, false);
serde_default_fn!(bool, alibi_default, false);
serde_default_fn!(bool, parallel_attn_default, true);
serde_default_fn!(bool, multi_query_default, true);
serde_default_fn!(bool, new_decoder_default, false);
serde_default_fn!(f64, layer_norm_epsilon_default, 1e-5);
serde_default_fn!(f64, rope_theta_default, 10000.0);
serde_default_fn!(usize, max_position_default, 2048);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: Option<usize>,
    #[serde(default = "bias_default")]
    pub bias: bool,
    #[serde(default = "alibi_default")]
    pub alibi: bool,
    #[serde(default = "parallel_attn_default")]
    pub parallel_attn: bool,
    #[serde(default = "multi_query_default")]
    pub multi_query: bool,
    #[serde(default = "new_decoder_default")]
    pub new_decoder_architecture: bool,
    pub num_ln_in_parallel_attn: Option<usize>,
    pub ffn_hidden_size: Option<usize>,
    #[serde(default = "layer_norm_epsilon_default")]
    pub layer_norm_epsilon: f64,
    #[serde(default = "rope_theta_default")]
    pub rope_theta: f64,
    #[serde(default = "max_position_default")]
    pub max_position_embeddings: usize,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Effective K/V head count, mirroring HF `FalconAttention`: full GQA for the new decoder
    /// architecture or classic multi-head, one shared head for multi-query.
    fn num_kv_heads(&self) -> usize {
        if self.new_decoder_architecture || !self.multi_query {
            self.num_kv_heads.unwrap_or(self.num_attention_heads)
        } else {
            1
        }
    }

    fn ffn_hidden_size(&self) -> usize {
        self.ffn_hidden_size.unwrap_or(4 * self.hidden_size)
    }

    fn ln_in_parallel_attn(&self) -> usize {
        self.num_ln_in_parallel_attn
            .unwrap_or(if self.new_decoder_architecture { 2 } else { 1 })
    }
}

/// ALiBi slopes, matching HF `build_alibi_tensor` (with the closest-power-of-2 interpolation for a
/// non-power-of-2 head count).
fn alibi_slopes(num_heads: usize) -> Vec<f32> {
    let closest = 2f64.powf((num_heads as f64).log2().floor()) as usize;
    let base = 2f64.powf(-(2f64.powf(-((closest as f64).log2() - 3.0))));
    let mut slopes: Vec<f32> = (1..=closest).map(|p| base.powi(p as i32) as f32).collect();
    if closest != num_heads {
        let extra_base = 2f64.powf(-(2f64.powf(-(((2 * closest) as f64).log2() - 3.0))));
        let n_remaining = closest.min(num_heads - closest);
        for p in (1..=2 * n_remaining).step_by(2) {
            slopes.push(extra_base.powi(p as i32) as f32);
        }
    }
    slopes
}

/// Loaded arrangement of the per-layer norms — the three real Falcon shapes.
enum Norms {
    /// Old parallel: one `input_layernorm`; attention and MLP share it.
    ParallelSingle(LayerNorm),
    /// Old sequential (e.g. falcon-rw-1b): pre-attention + pre-MLP norms in a residual chain.
    Sequential { input: LayerNorm, post: LayerNorm },
    /// New decoder architecture: separate `ln_attn` / `ln_mlp` off the same residual.
    ParallelDual { attn: LayerNorm, mlp: LayerNorm },
}

struct Mlp {
    dense_h_to_4h: Arc<dyn QuantMethod>,
    dense_4h_to_h: Arc<dyn QuantMethod>,
    act: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let ffn = cfg.ffn_hidden_size();
        Ok(Self {
            dense_h_to_4h: ReplicatedLayer::new(
                cfg.hidden_size,
                ffn,
                &cfg.quantization_config,
                cfg.bias,
                vb.pp("dense_h_to_4h"),
            )?,
            dense_4h_to_h: ReplicatedLayer::new(
                ffn,
                cfg.hidden_size,
                &cfg.quantization_config,
                cfg.bias,
                vb.pp("dense_4h_to_h"),
            )?,
            act: Activation::Gelu,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.dense_4h_to_h
            .forward(&self.dense_h_to_4h.forward(xs)?.apply(&self.act)?)
    }
}

struct Attention {
    query_key_value: Arc<dyn QuantMethod>,
    dense: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    multi_query: bool,
    new_decoder_architecture: bool,
    rotary_emb: Option<Arc<RotaryEmbedding>>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_kv_heads();
        let qkv_out_dim = if cfg.new_decoder_architecture {
            (num_kv_heads * 2 + num_heads) * head_dim
        } else if cfg.multi_query {
            cfg.hidden_size + 2 * head_dim
        } else {
            3 * cfg.hidden_size
        };
        let query_key_value = ReplicatedLayer::new(
            cfg.hidden_size,
            qkv_out_dim,
            &cfg.quantization_config,
            cfg.bias,
            vb.pp("query_key_value"),
        )?;
        let dense = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.hidden_size,
            &cfg.quantization_config,
            cfg.bias,
            vb.pp("dense"),
        )?;
        Ok(Self {
            query_key_value,
            dense,
            num_heads,
            num_kv_heads,
            head_dim,
            multi_query: cfg.multi_query,
            new_decoder_architecture: cfg.new_decoder_architecture,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    /// Split the fused `query_key_value` output into per-head q/k/v, mirroring HF `_split_heads`.
    /// Returns `(b, s, heads, head_dim)` tensors (kv with `num_kv_heads`).
    fn split_qkv(&self, qkv: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = qkv.dims3()?;
        let hd = self.head_dim;
        if self.new_decoder_architecture {
            let q_per_kv = self.num_heads / self.num_kv_heads;
            let qkv = qkv.reshape((b, s, self.num_kv_heads, q_per_kv + 2, hd))?;
            let q =
                qkv.narrow(3, 0, q_per_kv)?
                    .contiguous()?
                    .reshape((b, s, self.num_heads, hd))?;
            let k =
                qkv.narrow(3, q_per_kv, 1)?
                    .contiguous()?
                    .reshape((b, s, self.num_kv_heads, hd))?;
            let v = qkv.narrow(3, q_per_kv + 1, 1)?.contiguous()?.reshape((
                b,
                s,
                self.num_kv_heads,
                hd,
            ))?;
            Ok((q, k, v))
        } else if self.multi_query {
            let qkv = qkv.reshape((b, s, self.num_heads + 2, hd))?;
            let q = qkv.narrow(2, 0, self.num_heads)?.contiguous()?;
            let k = qkv.narrow(2, self.num_heads, 1)?.contiguous()?;
            let v = qkv.narrow(2, self.num_heads + 1, 1)?.contiguous()?;
            Ok((q, k, v))
        } else {
            let qkv = qkv.reshape((b, s, self.num_heads, 3, hd))?;
            let q = qkv
                .narrow(3, 0, 1)?
                .contiguous()?
                .reshape((b, s, self.num_heads, hd))?;
            let k = qkv
                .narrow(3, 1, 1)?
                .contiguous()?
                .reshape((b, s, self.num_heads, hd))?;
            let v = qkv
                .narrow(3, 2, 1)?
                .contiguous()?
                .reshape((b, s, self.num_heads, hd))?;
            Ok((q, k, v))
        }
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

        let qkv = self.query_key_value.forward(xs)?;
        let (q, k, v) = self.split_qkv(&qkv)?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let (q, k) = if let Some(rotary_emb) = &self.rotary_emb {
            let rope_positions = ctx
                .rope_positions(q.device())?
                .ok_or_else(|| hanzo_ml::Error::msg("missing RoPE positions"))?;
            rotary_emb.forward_positions(&q, &k, rope_positions)?
        } else {
            (q, k)
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
        self.dense.forward(&attn_output)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    norms: Norms,
    parallel_attn: bool,
    new_decoder_architecture: bool,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attention"), loading_isq),
            rotary_emb,
            paged_attn,
        )?;
        let mlp = Mlp::new(cfg, mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq))?;
        let ln = |name: &str| {
            layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_epsilon,
                mapper.set_device(layer_idx, vb.pp(name), false),
            )
        };
        let norms = if cfg.new_decoder_architecture {
            if cfg.ln_in_parallel_attn() == 2 {
                Norms::ParallelDual {
                    attn: ln("ln_attn")?,
                    mlp: ln("ln_mlp")?,
                }
            } else {
                Norms::ParallelSingle(ln("ln_attn")?)
            }
        } else if cfg.parallel_attn {
            Norms::ParallelSingle(ln("input_layernorm")?)
        } else {
            Norms::Sequential {
                input: ln("input_layernorm")?,
                post: ln("post_attention_layernorm")?,
            }
        };
        Ok(Self {
            self_attn,
            mlp,
            norms,
            parallel_attn: cfg.parallel_attn,
            new_decoder_architecture: cfg.new_decoder_architecture,
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
        match &self.norms {
            Norms::ParallelDual { attn, mlp } => {
                let attn_ln = attn.forward(xs)?;
                let mlp_ln = mlp.forward(xs)?;
                let attn_out =
                    self.self_attn
                        .forward(&attn_ln, attention_mask, kv_cache, ctx, layer_idx)?;
                let mlp_out = (self.mlp.forward(&mlp_ln)? + &attn_out)?;
                mlp_out + residual
            }
            Norms::ParallelSingle(ln) => {
                let ln_out = ln.forward(xs)?;
                let attn_out =
                    self.self_attn
                        .forward(&ln_out, attention_mask, kv_cache, ctx, layer_idx)?;
                // `parallel_attn` shares the pre-attention norm with the MLP.
                debug_assert!(self.parallel_attn || self.new_decoder_architecture);
                let mlp_out = (self.mlp.forward(&ln_out)? + &attn_out)?;
                mlp_out + residual
            }
            Norms::Sequential { input, post } => {
                let ln_out = input.forward(xs)?;
                let attn_out =
                    self.self_attn
                        .forward(&ln_out, attention_mask, kv_cache, ctx, layer_idx)?;
                let residual = (attn_out + residual)?;
                let mlp_out = self.mlp.forward(&post.forward(&residual)?)?;
                mlp_out + residual
            }
        }
    }
}

pub struct Model {
    word_embeddings: hanzo_nn::Embedding,
    layers: Vec<DecoderLayer>,
    ln_f: LayerNorm,
    lm_head: Arc<dyn QuantMethod>,
    alibi_slopes: Option<Vec<f32>>,
    num_heads: usize,
    softmax_scale: f32,
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
        let vb_t = vb.pp("transformer");
        let head_dim = cfg.head_dim();

        let word_embeddings = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_t.pp("word_embeddings"), false),
            &cfg.quantization_config,
        )?;

        let mut ropes = HashMap::new();
        if !cfg.alibi {
            for layer_idx in 0..cfg.num_hidden_layers {
                let device = mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&normal_loading_metadata.real_device);
                ropes.insert(
                    device.location(),
                    Arc::new(RotaryEmbedding::new(
                        cfg.rope_theta as f32,
                        head_dim,
                        cfg.max_position_embeddings,
                        device,
                        is_gptx,
                        vb_t.dtype(),
                    )?),
                );
            }
        }

        let vb_l = vb_t.pp("h");
        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes.get(&device.location()).cloned();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(
                    PagedAttention::new(head_dim, device, None)
                        .expect("PagedAttention creation failed"),
                ),
            };
            DecoderLayer::new(
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                rotary_emb,
                paged_attn,
            )
        })?;
        let ln_f = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            mapper.set_nm_device(vb_t.pp("ln_f"), false),
        )?;
        let lm_head = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.vocab_size,
            &cfg.quantization_config,
            false,
            mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
        )?;
        Ok(Self {
            word_embeddings,
            layers,
            ln_f,
            lm_head,
            alibi_slopes: cfg.alibi.then(|| alibi_slopes(cfg.num_attention_heads)),
            num_heads: cfg.num_attention_heads,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
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
                num_attn_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_kv_heads(),
                sliding_window: None,
                k_head_dim: head_dim,
                v_head_dim: head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
        })
    }

    /// Build the additive attention bias for an ALiBi model: `softmax_scale * slope[h] * key_pos`
    /// on allowed positions, a large negative on causally-masked positions. Shape `(b, h, q, k)`.
    fn alibi_mask(
        &self,
        slopes: &[f32],
        b_sz: usize,
        q_len: usize,
        offsets: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let k_len = offsets.iter().copied().max().unwrap_or(0) + q_len;
        let mut data = vec![0f32; b_sz * self.num_heads * q_len * k_len];
        for b in 0..b_sz {
            let offset = offsets.get(b).copied().unwrap_or(0);
            for (h, slope) in slopes.iter().enumerate() {
                let s = self.softmax_scale * slope;
                for i in 0..q_len {
                    let q_pos = offset + i;
                    let row = (((b * self.num_heads) + h) * q_len + i) * k_len;
                    for j in 0..k_len {
                        data[row + j] = if j <= q_pos { s * j as f32 } else { f32::MIN };
                    }
                }
            }
        }
        Tensor::from_vec(data, (b_sz, self.num_heads, q_len, k_len), device)?.to_dtype(dtype)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let (b_sz, q_len) = input_ids.dims2()?;
        let mut xs = self.word_embeddings.forward(input_ids)?;

        let cache = &mut self.cache.normal().0;
        let attention_mask = if let Some(slopes) = &self.alibi_slopes {
            // ALiBi bias is required on every step (prompt and decode), so build it explicitly rather
            // than reusing the `None`-on-decode causal fast path.
            AttentionMask::Custom(self.alibi_mask(
                slopes,
                b_sz,
                q_len,
                ctx.seqlen_offsets(),
                xs.dtype(),
                xs.device(),
            )?)
        } else {
            let mask_cache = ctx.mask_cache(cache);
            let mask = CausalMasker.make_causal_mask(
                input_ids,
                &mask_cache,
                xs.dtype(),
                &CausalMaskConfig::default(),
            )?;
            if ctx.is_first_prompt_chunk() {
                mask
            } else {
                AttentionMask::None
            }
        };
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(&xs, &attention_mask.get(xs.device()), &mut cache[i], ctx, i)?;
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
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.query_key_value, Some(i)));
            tensors.push((&mut layer.self_attn.dense, Some(i)));
            tensors.push((&mut layer.mlp.dense_h_to_4h, Some(i)));
            tensors.push((&mut layer.mlp.dense_4h_to_h, Some(i)));
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_t = uvb.pp("transformer");
        uvb_t.pp("word_embeddings").add(&self.word_embeddings);
        uvb_t.pp("ln_f").add(&self.ln_f);
        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_t.pp("h").pp(i);
            match &layer.norms {
                Norms::ParallelSingle(ln) => {
                    uvb_l.pp("input_layernorm").add(ln);
                }
                Norms::Sequential { input, post } => {
                    uvb_l.pp("input_layernorm").add(input);
                    uvb_l.pp("post_attention_layernorm").add(post);
                }
                Norms::ParallelDual { attn, mlp } => {
                    uvb_l.pp("ln_attn").add(attn);
                    uvb_l.pp("ln_mlp").add(mlp);
                }
            }
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
        // ALiBi builds a host-shaped mask each step, so keep it off the position-invariant fast path.
        self.alibi_slopes.is_none()
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

    // The arch block of the real `tiiuae/falcon-rw-1b` config.json (ALiBi + sequential norms +
    // multi-head + bias). Embedded for a hermetic test.
    const REAL_CONFIG: &str = r#"{
        "alibi": true,
        "architectures": ["FalconForCausalLM"],
        "bias": true,
        "hidden_size": 2048,
        "layer_norm_epsilon": 1e-05,
        "model_type": "falcon",
        "multi_query": false,
        "new_decoder_architecture": false,
        "num_attention_heads": 32,
        "num_hidden_layers": 24,
        "parallel_attn": false,
        "vocab_size": 50304
    }"#;

    // Tiny falcon-rw-style config: ALiBi + non-parallel sequential norms + full multi-head + bias —
    // the exact code path the `falcon-rw-1b` checkpoint exercises.
    const TOY_CONFIG: &str = r#"{
        "alibi": true,
        "bias": true,
        "hidden_size": 32,
        "layer_norm_epsilon": 1e-05,
        "multi_query": false,
        "new_decoder_architecture": false,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "parallel_attn": false,
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

    fn write_checkpoint(cfg: &Config, dir: &std::path::Path) -> Result<std::path::PathBuf> {
        let mut rng = StdRng::seed_from_u64(0xFA1C0);
        let mut t = |shape: (usize, usize)| -> Result<Tensor> {
            let data: Vec<f32> = (0..shape.0 * shape.1)
                .map(|_| rng.random_range(-0.08f32..0.08))
                .collect();
            Tensor::from_vec(data, shape, &Device::Cpu)
        };
        let one = |n: usize| Tensor::ones((n,), DType::F32, &Device::Cpu);
        let zero = |n: usize| Tensor::zeros((n,), DType::F32, &Device::Cpu);
        let h = cfg.hidden_size;
        let ffn = cfg.ffn_hidden_size();
        let qkv_out = 3 * h; // toy config: old-arch multi-head
        let mut ts = std::collections::HashMap::new();
        ts.insert(
            "transformer.word_embeddings.weight".to_string(),
            t((cfg.vocab_size, h))?,
        );
        ts.insert("transformer.ln_f.weight".to_string(), one(h)?);
        ts.insert("transformer.ln_f.bias".to_string(), zero(h)?);
        ts.insert("lm_head.weight".to_string(), t((cfg.vocab_size, h))?);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("transformer.h.{i}");
            ts.insert(format!("{p}.input_layernorm.weight"), one(h)?);
            ts.insert(format!("{p}.input_layernorm.bias"), zero(h)?);
            ts.insert(format!("{p}.post_attention_layernorm.weight"), one(h)?);
            ts.insert(format!("{p}.post_attention_layernorm.bias"), zero(h)?);
            ts.insert(
                format!("{p}.self_attention.query_key_value.weight"),
                t((qkv_out, h))?,
            );
            ts.insert(
                format!("{p}.self_attention.query_key_value.bias"),
                zero(qkv_out)?,
            );
            ts.insert(format!("{p}.self_attention.dense.weight"), t((h, h))?);
            ts.insert(format!("{p}.self_attention.dense.bias"), zero(h)?);
            ts.insert(format!("{p}.mlp.dense_h_to_4h.weight"), t((ffn, h))?);
            ts.insert(format!("{p}.mlp.dense_h_to_4h.bias"), zero(ffn)?);
            ts.insert(format!("{p}.mlp.dense_4h_to_h.weight"), t((h, ffn))?);
            ts.insert(format!("{p}.mlp.dense_4h_to_h.bias"), zero(h)?);
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
    fn falcon_config_parses() {
        let cfg: Config = serde_json::from_str(REAL_CONFIG).unwrap();
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 32);
        assert!(cfg.alibi);
        assert!(cfg.bias);
        assert!(!cfg.parallel_attn);
        assert!(!cfg.multi_query);
        assert!(!cfg.new_decoder_architecture);
        // rw-1b is classic multi-head: kv heads == attention heads.
        assert_eq!(cfg.num_kv_heads(), 32);
        assert_eq!(cfg.ffn_hidden_size(), 8192);
    }

    #[test]
    fn falcon_registry_dispatch() {
        use crate::pipeline::NormalLoaderType;
        assert_eq!(
            NormalLoaderType::from_causal_lm_name("FalconForCausalLM").unwrap(),
            NormalLoaderType::Falcon
        );
    }

    #[test]
    fn falcon_alibi_slopes_powers_of_two() {
        // For a power-of-2 head count the slopes are exactly `base^p`, base = 2^-0.25.
        let slopes = alibi_slopes(4);
        assert_eq!(slopes.len(), 4);
        let base = 2f32.powf(-0.5); // 2^(-(2^-(log2(4)-3))) = 2^(-(2^1)) ... verify monotone decrease
        assert!(slopes[0] > slopes[1] && slopes[1] > slopes[2] && slopes[2] > slopes[3]);
        let _ = base;
    }

    #[test]
    fn falcon_toy_forward_causal() -> Result<()> {
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
