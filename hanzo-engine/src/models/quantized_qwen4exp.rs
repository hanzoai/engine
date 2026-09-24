#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Qwen3.8-Flash-Next from its GGUF (arch `qwen4exp`).
//!
//! Qwen3.5's blocks (gated attention, gated delta-net, 512-expert MoE with a shared expert) joined
//! by `n`-stream hyper-connections in place of RMSNorm residuals, and a hashed n-gram memory added
//! to the streams before one layer. The head's hyper-connection mixer is the output norm.
//!
//! Every attention layer carries a sparse-attention (QSA) indexer that keeps at most its token
//! budget of keys per query. Up to that budget it keeps them all, so dense attention is exact, and
//! the context is capped there until the sparse path lands.
//!
//! Source: vLLM `models/qwen4_exp/nvidia/model.py`.

use std::sync::{Arc, Mutex};

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{QuantMethod, ShardedVarBuilder};

use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::kv_cache::{
    HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
};
use crate::layers::{CausalMaskConfig, CausalMasker, Qwen3VLRotaryEmbedding};
use crate::layers_masker::PastKvLenCache;
use crate::models::gdn::{forward_pooled, PoolSlots};
use crate::models::hyper::{expand, Branch, Mixer};
use crate::models::ngram::{Hash, Ngram};
use crate::models::quantized_qwen3_5_moe::{
    gguf_qmm, text_mrope, verify_arch, FusedMoe, GatedFullAttention, LayerImpl, PropsGGUF,
    QGatedDeltaNet,
};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

const ARCH: &str = "qwen4exp";

/// One decoder layer: a mixing block (attention or gated delta-net) and the MoE, each behind its
/// own hyper-connection branch, and on one layer the n-gram block in front of them.
struct Layer {
    mix: LayerImpl,
    attn: Branch,
    ffn: Branch,
    moe: FusedMoe,
    ngram: Option<Ngram>,
}

/// What one forward hands every layer.
struct Step<'a> {
    cache: &'a mut HybridCache,
    slots: &'a dyn Fn() -> Result<PoolSlots<'a>>,
    trail: bool,
    mask: &'a DeviceMappedMask,
    cos_sin: &'a (Tensor, Tensor),
    paged: Option<&'a (Vec<(Tensor, Tensor)>, &'a PagedAttentionInputMetadata)>,
    /// The next attention layer's ordinal in the paged cache.
    kv_layer: usize,
    /// The n-gram embeddings of this forward, and the side pool holding the conv history.
    ngram: Option<&'a Tensor>,
    side: usize,
}

impl Layer {
    /// Layer `i` at `lvb` (`...layers.{i}`) of a safetensors snapshot.
    fn new(
        i: usize,
        lvb: &ShardedVarBuilder,
        cfg: &crate::models::qwen4exp::Config,
        weights: &[std::path::PathBuf],
        rotary: Arc<Qwen3VLRotaryEmbedding>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        use crate::models::ngram::Fp8Table;
        use crate::models::qwen4exp::PREFIX;
        let text = cfg.text();
        let props = cfg.props();
        let quant = cfg.quant()?;
        let eps = props.rms_norm_eps;
        let dev = lvb.device().clone();
        let mix = if text.attention(i) {
            let paged = match attention_mechanism {
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(props.head_dim, &dev, None)?)
                }
                AttentionImplementation::Eager => None,
            };
            LayerImpl::FullAttention(GatedFullAttention::new(
                &lvb.pp("self_attn"),
                &props,
                &quant,
                rotary,
                paged,
                &dev,
                dtype,
            )?)
        } else {
            LayerImpl::LinearAttention(QGatedDeltaNet::new(&lvb.pp("linear_attn"), &props, &quant)?)
        };
        let ngram = if i == text.ple_layer()? {
            let hash = Hash::new(
                &lvb.pp("ple.ple_embedding"),
                text.eos_token_id,
                text.heads_per_ngram,
            )?;
            let table = Fp8Table::open(
                weights,
                &format!("{PREFIX}layers.{i}.ple.ple_embedding.ngram_embedding"),
            )?;
            Some(Ngram::new(&lvb.pp("ple"), hash, table, f64::from(eps))?)
        } else {
            None
        };
        Ok(Self {
            mix,
            attn: Branch::new(&lvb.pp("attn_hyper_connection"), text.hc_count, eps)?,
            ffn: Branch::new(&lvb.pp("mlp_hyper_connection"), text.hc_count, eps)?,
            moe: FusedMoe::new(&lvb.pp("mlp"), &props, &quant, dtype)?,
            ngram,
        })
    }

    /// This layer over the streams `x` `[b, s, n, h]`.
    fn forward(&self, i: usize, x: Tensor, step: &mut Step<'_>) -> Result<Tensor> {
        let mut x = x;
        if let Some(ngram) = &self.ngram {
            let Some(HybridLayerCache::Recurrent(pool)) = step.cache.get_mut(step.side) else {
                hanzo_ml::bail!("hybrid cache has no n-gram pool at {}", step.side);
            };
            let Some(g) = step.ngram else {
                hanzo_ml::bail!("layer {i} needs the n-gram embeddings");
            };
            let g = g.to_device(x.device())?;
            let delta = forward_pooled(pool, (step.slots)()?, step.side, step.trail, |cache| {
                ngram.forward(&x, &g, cache)
            })?;
            // The served graph adds the f32 delta inside one kernel and rounds only the sum.
            x = (x.to_dtype(DType::F32)? + delta)?.to_dtype(x.dtype())?;
        }
        x = match &self.mix {
            LayerImpl::FullAttention(attn) => {
                let paged = step
                    .paged
                    .map(|(kv_cache, meta)| (kv_cache[step.kv_layer].clone(), *meta));
                step.kv_layer += 1;
                let Some(HybridLayerCache::Attention(kv_cache)) = step.cache.get_mut(i) else {
                    hanzo_ml::bail!("hybrid cache layer {i} is not attention");
                };
                let mask = step.mask.get(x.device());
                let cos_sin = step.cos_sin;
                self.attn
                    .apply(&x, |u| attn.forward(u, &mask, cos_sin, kv_cache, paged))?
            }
            LayerImpl::LinearAttention(gdn) => {
                let Some(HybridLayerCache::Recurrent(pool)) = step.cache.get_mut(i) else {
                    hanzo_ml::bail!("hybrid cache layer {i} is not recurrent");
                };
                let slots = (step.slots)()?;
                let trail = step.trail;
                self.attn.apply(&x, |u| {
                    forward_pooled(pool, slots, i, trail, |cache| gdn.forward(u, cache))
                })?
            }
        };
        self.ffn.apply(&x, |v| self.moe.forward(v))
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    /// Residual streams.
    streams: usize,
    /// The layer whose input the n-gram delta joins.
    ngram_layer: usize,
    layers: Vec<Layer>,
    head: Mixer,
    output: Arc<dyn QuantMethod>,
    rotary: Arc<Qwen3VLRotaryEmbedding>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    /// The attention layers' shape and the layers that hold paged K/V (safetensors path).
    meta: Option<(crate::paged_attention::ModelConfigMetadata, Vec<usize>)>,
}

/// The hybrid cache: one pool per gated delta-net layer (f32 state, conv history in `dtype`),
/// then the n-gram history as a side pool.
fn hybrid_cache(
    props: &PropsGGUF,
    attention: &dyn Fn(usize) -> bool,
    ngram: &Ngram,
    dtype: DType,
    device: &Device,
) -> Result<EitherCache> {
    let layer_types: Vec<HybridLayerType> = (0..props.block_count)
        .map(|i| {
            if attention(i) {
                HybridLayerType::Attention
            } else {
                HybridLayerType::Recurrent
            }
        })
        .collect();
    let key_dim = props.num_k_heads * props.head_k_dim;
    let gdn = RecurrentLayerConfig {
        conv_dim: key_dim * 2 + props.num_v_heads * props.head_v_dim,
        conv_width: props.conv_kernel,
        state_dims: vec![props.num_v_heads, props.head_k_dim, props.head_v_dim],
        conv_dtype: dtype,
        state_dtype: DType::F32,
    };
    let recurrent = layer_types
        .iter()
        .filter(|t| **t == HybridLayerType::Recurrent)
        .count();
    let mut pools = vec![gdn; recurrent];
    pools.push(ngram.pool(dtype)?);
    let cache = HybridCache::new(
        HybridCacheConfig {
            layer_types,
            max_seq_len: props.max_seq_len,
            pools,
        },
        device,
    )
    .map_err(|e| hanzo_ml::Error::Msg(format!("Failed to create hybrid cache: {e}")))?;
    Ok(EitherCache::Hybrid(Arc::new(Mutex::new(cache))))
}

impl ModelWeights {
    /// The text model of a Qwen3.8-Flash-Next safetensors snapshot (`vb` at its root, `weights`
    /// its files, which the n-gram table maps itself): block-FP8 side layers, NVFP4 experts,
    /// bf16 elsewhere, with the served numerics. The context is the QSA indexer's budget.
    pub fn new(
        cfg: &crate::models::qwen4exp::Config,
        vb: ShardedVarBuilder,
        weights: &[std::path::PathBuf],
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        use crate::models::ngram::Fp8Table;
        use crate::models::qwen4exp::PREFIX;
        let text = cfg.text();
        let props = cfg.props();
        let quant = cfg.quant()?;
        let device = vb.device().clone();
        let eps = props.rms_norm_eps;
        let streams = text.hc_count;
        let ngram_layer = text.ple_layer()?;
        let attention = |i: usize| text.attention(i);
        let lm = vb.pp(PREFIX.trim_end_matches('.'));

        let tok_embeddings = mapper
            .set_nm_device(lm.pp("embed_tokens"), false)
            .get((text.vocab_size, text.hidden_size), "weight")?;
        let output = hanzo_quant::linear_no_bias(
            text.hidden_size,
            text.vocab_size,
            &None,
            mapper.set_nm_device(vb.pp("lm_head"), false),
        )?;
        let head = Mixer::new(
            &mapper.set_nm_device(lm.pp("hyper_connection_mixer"), false),
            streams,
            eps,
        )?;
        let rotary = Arc::new(Qwen3VLRotaryEmbedding::new(
            props.rope_freq_base,
            props.rot_dim,
            &device,
            props.mrope_section.clone(),
        )?);

        let mut layers = Vec::with_capacity(props.block_count);
        for i in NiceProgressBar::<_, 'b'>(
            0..props.block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let lvb = mapper.set_device(i, lm.pp(format!("layers.{i}")), false);
            let dev = lvb.device().clone();
            let rotary = if dev.same_device(&device) {
                rotary.clone()
            } else {
                Arc::new(Qwen3VLRotaryEmbedding::new(
                    props.rope_freq_base,
                    props.rot_dim,
                    &dev,
                    props.mrope_section.clone(),
                )?)
            };
            layers.push(Layer::new(
                i,
                &lvb,
                cfg,
                weights,
                rotary,
                attention_mechanism,
                dtype,
            )?);
        }
        let Some(ngram) = layers[ngram_layer].ngram.as_ref() else {
            hanzo_ml::bail!("layer {ngram_layer} has no n-gram block");
        };
        let cache = hybrid_cache(&props, &attention, ngram, dtype, &device)?;
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, text.hidden_size),
            streams,
            ngram_layer,
            layers,
            head,
            output,
            rotary,
            device,
            cache,
            max_seq_len: props.max_seq_len,
            mapper: Some(mapper),
            dtype,
            meta: Some((
                crate::paged_attention::ModelConfigMetadata {
                    max_seq_len: props.max_seq_len,
                    num_layers: props.block_count,
                    hidden_size: text.hidden_size,
                    num_kv_heads: props.head_count_kv,
                    num_attn_heads: props.head_count,
                    sliding_window: None,
                    k_head_dim: props.head_dim,
                    v_head_dim: props.head_dim,
                    kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
                },
                (0..props.block_count).filter(|&i| text.attention(i)).collect(),
            )),
        })
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        let meta = ct.get_metadata();
        verify_arch(meta, &[ARCH])?;
        let md = ContentMetadata {
            path_prefix: ARCH,
            metadata: meta,
        };
        let get = |key: &str| {
            md.get_value::<u32>(key)
                .map(|v| v as usize)
                .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))
        };
        let mut props = PropsGGUF::try_from(&md, true)?;
        let streams = get("hyper_connection.count")?;
        // The indexer keeps every key up to its budget, where dense attention is exact.
        props.max_seq_len = props.max_seq_len.min(get("attention.indexer.top_k")?);
        // An INT32 array in the file, as the converter writes Python ints.
        let ngram_layer = match md.get_value::<Vec<i32>>("ple.layers") {
            Ok(layers) if layers.len() == 1 && layers[0] >= 0 => layers[0] as usize,
            other => hanzo_ml::bail!("qwen4exp needs exactly one n-gram layer, got {other:?}"),
        };
        let hash = Hash::from_gguf(&md)?;
        let eps = props.rms_norm_eps;

        let attention = |i: usize| (i + 1) % props.full_attention_interval == 0;
        let key_dim = props.num_k_heads * props.head_k_dim;
        let conv_dim = key_dim * 2 + props.num_v_heads * props.head_v_dim;

        let tok_embeddings = ct.tensor("token_embd.weight", device)?.dequantize(device)?;
        let output = gguf_qmm(ct.tensor("output.weight", device)?)?;
        let head = Mixer::from_gguf(&mut ct, "output_hc", streams, eps, device)?;
        let ngram_dev = mapper.device_for(ngram_layer, false).unwrap_or(device);
        let mut ngram = Some(Ngram::load(&mut ct, hash, ngram_layer, f64::from(eps), ngram_dev)?);

        let rotary = Arc::new(Qwen3VLRotaryEmbedding::new(
            props.rope_freq_base,
            props.rot_dim,
            device,
            props.mrope_section.clone(),
        )?);

        let mut layers = Vec::with_capacity(props.block_count);
        for i in NiceProgressBar::<_, 'b'>(
            0..props.block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{i}");
            let dev = mapper.device_for(i, false).unwrap_or(device);
            let rotary = if dev.same_device(device) {
                rotary.clone()
            } else {
                Arc::new(Qwen3VLRotaryEmbedding::new(
                    props.rope_freq_base,
                    props.rot_dim,
                    dev,
                    props.mrope_section.clone(),
                )?)
            };
            let mix = if attention(i) {
                let paged = match attention_mechanism {
                    AttentionImplementation::PagedAttention => {
                        Some(PagedAttention::new(props.head_dim, dev, None)?)
                    }
                    AttentionImplementation::Eager => None,
                };
                LayerImpl::FullAttention(
                    GatedFullAttention::load(&mut ct, &prefix, &props, rotary, paged, dev, dtype)?
                        .qsa(),
                )
            } else {
                // The output gate is a sigmoid here, where Qwen3.5 uses silu.
                LayerImpl::LinearAttention(
                    QGatedDeltaNet::load(&mut ct, &prefix, &props, dev)?.sigmoid(),
                )
            };
            layers.push(Layer {
                mix,
                attn: Branch::from_gguf(&mut ct, &format!("{prefix}.hc_attn"), streams, eps, dev)?,
                ffn: Branch::from_gguf(&mut ct, &format!("{prefix}.hc_ffn"), streams, eps, dev)?,
                moe: FusedMoe::from_gguf(&mut ct, &prefix, dev, props.num_experts_per_tok)?,
                ngram: if i == ngram_layer { ngram.take() } else { None },
            });
        }

        let Some(ngram) = layers[ngram_layer].ngram.as_ref() else {
            hanzo_ml::bail!("layer {ngram_layer} has no n-gram block");
        };
        let cache = hybrid_cache(&props, &attention, ngram, dtype, device)?;

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, props.embedding_length),
            streams,
            ngram_layer,
            layers,
            head,
            output,
            rotary,
            device: device.clone(),
            cache,
            max_seq_len: props.max_seq_len,
            mapper: Some(mapper),
            dtype,
            meta: None,
        })
    }
}

/// At trace level, a fingerprint of `t`: its sum and absolute sum, and for its last token the
/// norm and a 64-value sketch (signed sums over a fixed pseudo-random ±1 pattern, scaled by
/// 1/√len). The sketch is linear, so two runs' sketches differ by the sketch of their difference,
/// whose norm estimates the difference's: runs on two backends line up stage by stage.
fn trace(stage: &str, layer: usize, t: &Tensor) -> Result<()> {
    const SKETCH: usize = 64;
    if tracing::enabled!(tracing::Level::TRACE) {
        let t = t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let sum = t.sum_all()?.to_scalar::<f32>()?;
        let abs = t.abs()?.sum_all()?.to_scalar::<f32>()?;
        // The last token: position `seq - 1` of `[batch, seq, ..]`, or the last row of logits.
        let last = if t.rank() >= 3 {
            t.narrow(1, t.dim(1)? - 1, 1)?
        } else {
            t.narrow(0, t.dim(0)? - 1, 1)?
        };
        let v = last.flatten_all()?.to_vec1::<f32>()?;
        let norm = v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let scale = (v.len() as f64).sqrt().recip();
        let sketch: Vec<f32> = (0..SKETCH as u64)
            .map(|j| {
                let dot: f64 = v
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        // splitmix64 of (i, j): one sign bit per entry.
                        let mut z = (i as u64).wrapping_mul(SKETCH as u64).wrapping_add(j);
                        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
                        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                        if (z ^ (z >> 31)) & 1 == 0 {
                            f64::from(*x)
                        } else {
                            -f64::from(*x)
                        }
                    })
                    .sum();
                (dot * scale) as f32
            })
            .collect();
        tracing::trace!(
            "{stage} {layer}: sum {sum:.6e} abs {abs:.6e} norm {norm:.6e} sketch {sketch:?}"
        );
    }
    Ok(())
}

impl ModelWeights {
    /// Logits for `input_ids` (batch, seq). `prior[b]` holds the tokens before sequence b's chunk
    /// that the n-gram hash reads (at most `ngram_size - 1`, fewer at the sequence start).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        prior: &[Vec<u32>],
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        let e = self.tok_embeddings.forward(input_ids)?;
        let Some(ngram) = self.layers[self.ngram_layer].ngram.as_ref() else {
            hanzo_ml::bail!("layer {} has no n-gram block", self.ngram_layer);
        };
        let g = ngram.embed(prior, &input_ids.to_vec2::<u32>()?, e.device())?;
        trace("embed", 0, &e)?;
        trace("ngram-rows", 0, &g)?;
        let mut x = expand(&e, self.streams)?;

        let mut hybrid_cache = self.cache.hybrid();
        let trail = hybrid_cache.records_trail(seq_len);
        let state_indices = hybrid_cache.state_indices().cloned();
        let state_indices_host: Option<Vec<u32>> =
            hybrid_cache.state_indices_host().map(|s| s.to_vec());
        let offset = seqlen_offsets.first().copied().unwrap_or(0);
        let slots = || -> Result<PoolSlots<'_>> {
            if b_sz == 1 {
                // One sequence reads its slot on the host, with no device sync (as Qwen3.5).
                let slot = state_indices_host
                    .as_ref()
                    .and_then(|s| s.first().copied())
                    .ok_or_else(|| hanzo_ml::Error::msg("missing host recurrent state index"))?;
                Ok(PoolSlots::One {
                    slot: slot as usize,
                    offset,
                })
            } else {
                state_indices
                    .as_ref()
                    .map(PoolSlots::Many)
                    .ok_or_else(|| hanzo_ml::Error::msg("missing recurrent state indices"))
            }
        };

        let mask = CausalMasker.make_causal_mask(
            input_ids,
            match metadata.as_ref() {
                Some(_) => &seqlen_offsets as &dyn PastKvLenCache,
                None => &*hybrid_cache as &dyn PastKvLenCache,
            },
            self.dtype,
            &CausalMaskConfig::gguf(),
        )?;
        let mask = crate::layers_masker::paged_chunk_mask(
            mask,
            metadata.as_ref().map(|(_, meta)| *meta),
            input_ids,
        )?;
        let mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(mask)
        };
        let rope_positions = metadata
            .as_ref()
            .and_then(|(_, meta)| meta.rope_positions.as_ref())
            .and_then(|rp| rp.get(&self.device.location()));
        let cos_sin = text_mrope(
            &self.rotary,
            &self.device,
            seqlen_offsets,
            seq_len,
            e.dtype(),
            rope_positions,
        )?;

        let side = self.layers.len();
        let mut step = Step {
            cache: &mut *hybrid_cache,
            slots: &slots,
            trail,
            mask: &mask,
            cos_sin: &cos_sin,
            paged: metadata.as_ref(),
            kv_layer: 0,
            ngram: Some(&g),
            side,
        };
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                x = mapper.map(x, i)?;
            }
            x = layer.forward(i, x, &mut step)?;
            trace("layer", i, &x)?;
            // Metal recycles pooled buffers without a completion check; drain each prefill layer
            // as Qwen3.5 does.
            if seq_len > 1 && x.device().is_metal() {
                x.device().synchronize()?;
            }
        }

        let x = x.to_device(&self.device)?;
        let h = self.head.mix(&self.head.norm(&x)?)?;
        trace("head", 0, &h)?;
        let h = extract_logits(&h, context_lens)?;
        let logits = self.output.forward(&h.contiguous()?)?;
        trace("logits", 0, &logits)?;
        Ok(logits)
    }
}

impl crate::pipeline::IsqModel for ModelWeights {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mapper = self.mapper.as_deref().expect("a device mapper");
        (Vec::new(), mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }
}

impl crate::amoe::AnyMoeBaseModelMixin for ModelWeights {}

impl crate::speculative::SpeculativeTargetMixin for ModelWeights {}

impl crate::pipeline::NormalModel for ModelWeights {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let metadata = ctx.paged_metadata();
        self.forward(
            input_ids,
            ctx.prior(),
            ctx.seqlen_offsets(),
            ctx.context_lens_vec(),
            metadata,
        )
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
        _flash_params: &crate::pipeline::text_models_inputs_processor::FlashParams,
        _flash_params_full: &crate::pipeline::text_models_inputs_processor::FlashParams,
    ) -> Result<Tensor> {
        hanzo_ml::bail!("qwen4exp does not support X-LoRA")
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
    fn config(&self) -> &crate::paged_attention::ModelConfigMetadata {
        &self.meta.as_ref().expect("a safetensors qwen4exp").0
    }
    fn model_config(&self) -> Arc<dyn crate::paged_attention::ModelConfigLike + Send + Sync> {
        let (meta, layers) = self.meta.as_ref().expect("a safetensors qwen4exp");
        Arc::new(crate::paged_attention::KvLayers::new(meta.clone(), layers.clone()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::Cursor;

    use hanzo_ml::quantized::{gguf_file, GgmlDType, QTensor};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::device_map::DummyDeviceMapper;
    use crate::utils::model_config::FromGGUF;

    // A tiny qwen4exp: 4 layers (three gated delta-nets, then attention), 2 streams, and the
    // n-gram block at layer 1 with a hand-sized hash (order 3, two heads per order).
    const VOCAB: usize = 24;
    // Hidden and FFN widths are Q8_0 blocks (32): expert banks are quantized, as in real files.
    const HIDDEN: usize = 32;
    const STREAMS: usize = 2;
    const RANK: usize = 4;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const GDN_V_HEADS: usize = 2;
    const GDN_DIM: usize = 4;
    const CONV: usize = 4;
    const EXPERTS: usize = 4;
    const FFN: usize = 32;
    const ROW: usize = 2;
    const EOS: u32 = 9;
    const MULT: [u64; 3] = [3, 5, 7];
    const SIZE: [u64; 4] = [11, 13, 17, 19];
    const OFFSET: [u64; 4] = [0, 11, 24, 41];

    fn metadata() -> HashMap<String, gguf_file::Value> {
        use gguf_file::Value::{Array, F32, I32, U32, U64};
        // Value types as the real file stores them: INT32 arrays, u32 scalars, u64 hash tables.
        let i32s = |v: &[i32]| Array(v.iter().map(|&x| I32(x)).collect());
        let u64s = |v: &[u64]| Array(v.iter().map(|&x| U64(x)).collect());
        let n = |x: usize| U32(x as u32);
        [
            ("attention.head_count", n(HEADS)),
            ("attention.head_count_kv", n(1)),
            ("attention.key_length", n(HEAD_DIM)),
            ("attention.layer_norm_rms_epsilon", F32(1e-6)),
            ("attention.indexer.top_k", n(32)),
            ("block_count", n(4)),
            ("context_length", n(64)),
            ("embedding_length", n(HIDDEN)),
            ("full_attention_interval", n(4)),
            ("rope.dimension_count", n(4)),
            ("rope.dimension_sections", i32s(&[1, 1, 0, 0])),
            ("rope.freq_base", F32(10_000.0)),
            ("ssm.conv_kernel", n(CONV)),
            ("ssm.state_size", n(GDN_DIM)),
            ("ssm.group_count", n(1)),
            ("ssm.time_step_rank", n(GDN_V_HEADS)),
            ("ssm.inner_size", n(GDN_V_HEADS * GDN_DIM)),
            ("expert_count", n(EXPERTS)),
            ("expert_used_count", n(2)),
            ("expert_feed_forward_length", n(FFN)),
            ("hyper_connection.count", n(STREAMS)),
            ("ple.layers", i32s(&[1])),
            ("ple.ngram_size", n(MULT.len())),
            ("ple.heads_per_ngram", n(2)),
            ("ple.eos_token_id", U32(EOS)),
            ("ple.layer_multipliers", u64s(&MULT)),
            ("ple.head_offsets", u64s(&OFFSET)),
            ("ple.head_vocab_sizes", u64s(&SIZE)),
        ]
        .into_iter()
        .map(|(k, v)| (format!("{ARCH}.{k}"), v))
        .collect()
    }

    /// Every tensor the loader reads, in torch layout, seeded. Norm-like vectors sit near 1.
    fn tensors() -> Vec<(String, Vec<usize>)> {
        let c = STREAMS * HIDDEN;
        let width = SIZE.len() * ROW;
        let rows = (OFFSET[3] + SIZE[3]) as usize;
        let conv_dim = 2 * GDN_DIM + GDN_V_HEADS * GDN_DIM;
        let mut t: Vec<(String, Vec<usize>)> = vec![
            ("token_embd".into(), vec![VOCAB, HIDDEN]),
            ("output".into(), vec![VOCAB, HIDDEN]),
            ("output_hc_norm".into(), vec![c]),
            ("output_hc_down".into(), vec![RANK, c]),
            ("output_hc_up".into(), vec![c, RANK]),
            ("per_layer_token_embd".into(), vec![rows, ROW]),
            ("blk.1.ple_key".into(), vec![c, width]),
            ("blk.1.ple_value".into(), vec![HIDDEN, width]),
            ("blk.1.ple_norm_query".into(), vec![c]),
            ("blk.1.ple_norm_key".into(), vec![c]),
            ("blk.1.ple_norm_conv".into(), vec![c]),
            ("blk.1.ple_conv1d".into(), vec![c, CONV]),
        ];
        for i in 0..4 {
            let b = |name: &str, shape: &[usize]| (format!("blk.{i}.{name}"), shape.to_vec());
            for hc in ["hc_attn", "hc_ffn"] {
                t.push(b(&format!("{hc}_norm"), &[c]));
                t.push(b(&format!("{hc}_down"), &[RANK, c]));
                t.push(b(&format!("{hc}_up"), &[c, RANK]));
                t.push(b(&format!("{hc}_inject"), &[STREAMS, c]));
            }
            if i == 3 {
                t.push(b("attn_q", &[2 * HEADS * HEAD_DIM, HIDDEN]));
                t.push(b("attn_k", &[HEAD_DIM, HIDDEN]));
                t.push(b("attn_v", &[HEAD_DIM, HIDDEN]));
                t.push(b("attn_output", &[HIDDEN, HEADS * HEAD_DIM]));
                t.push(b("attn_q_norm", &[HEAD_DIM]));
                t.push(b("attn_k_norm", &[HEAD_DIM]));
            } else {
                t.push(b("attn_qkv", &[conv_dim, HIDDEN]));
                t.push(b("attn_gate", &[GDN_V_HEADS * GDN_DIM, HIDDEN]));
                t.push(b("ssm_beta", &[GDN_V_HEADS, HIDDEN]));
                t.push(b("ssm_alpha", &[GDN_V_HEADS, HIDDEN]));
                t.push(b("ssm_out", &[HIDDEN, GDN_V_HEADS * GDN_DIM]));
                t.push(b("ssm_conv1d", &[conv_dim, CONV]));
                t.push((format!("blk.{i}.ssm_dt.bias"), vec![GDN_V_HEADS]));
                t.push((format!("blk.{i}.ssm_a"), vec![GDN_V_HEADS]));
                t.push(b("ssm_norm", &[GDN_DIM]));
            }
            t.push(b("ffn_gate_inp", &[EXPERTS, HIDDEN]));
            t.push(b("ffn_gate_exps", &[EXPERTS, FFN, HIDDEN]));
            t.push(b("ffn_up_exps", &[EXPERTS, FFN, HIDDEN]));
            t.push(b("ffn_down_exps", &[EXPERTS, HIDDEN, FFN]));
            t.push(b("ffn_gate_inp_shexp", &[HIDDEN]));
            t.push(b("ffn_gate_shexp", &[FFN, HIDDEN]));
            t.push(b("ffn_up_shexp", &[FFN, HIDDEN]));
            t.push(b("ffn_down_shexp", &[HIDDEN, FFN]));
        }
        t
    }

    /// The fixture's GGUF bytes. `ssm_a` is negative (the file stores -exp(A_log)); vectors named
    /// as norms sit near 1; everything else is uniform around 0.
    fn gguf(seed: u64) -> Result<Vec<u8>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let quantized = tensors()
            .into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                let (lo, hi) = if name.ends_with("ssm_a") {
                    (-1.0, -0.2)
                } else if shape.len() == 1 && !name.contains("dt.bias") && !name.ends_with("shexp")
                {
                    (0.7, 1.3)
                } else {
                    (-0.5, 0.5)
                };
                let data: Vec<f32> = (0..n).map(|_| rng.random_range(lo..hi)).collect();
                let t = Tensor::from_vec(data, shape, &Device::Cpu)?;
                let key = if name.ends_with(".bias") || name.ends_with("ssm_a") {
                    name
                } else {
                    format!("{name}.weight")
                };
                // The indexed MoE matmul runs on quantized banks only.
                let dtype = if key.contains("_exps.") {
                    GgmlDType::Q8_0
                } else {
                    GgmlDType::F32
                };
                Ok((key, QTensor::quantize(&t, dtype)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let arch = gguf_file::Value::String(ARCH.into());
        let md = metadata();
        let mut kv: Vec<(&str, &gguf_file::Value)> = vec![("general.architecture", &arch)];
        kv.extend(md.iter().map(|(k, v)| (k.as_str(), v)));
        let named: Vec<(&str, &QTensor)> = quantized.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let mut file = Cursor::new(Vec::new());
        gguf_file::write(&mut file, &kv, &named)?;
        Ok(file.into_inner())
    }

    /// A model loaded from `bytes`, its cache holding one sequence at slot 0.
    fn model(bytes: &[u8]) -> Result<ModelWeights> {
        let mut file = Cursor::new(bytes.to_vec());
        let mut readers = [&mut file];
        let ct = Content::from_readers(&mut readers)?;
        let model = ModelWeights::from_gguf(
            ct,
            &Device::Cpu,
            Box::new(DummyDeviceMapper {
                nm_device: Device::Cpu,
            }),
            AttentionImplementation::Eager,
            DType::F32,
        )?;
        {
            let mut cache = model.cache.hybrid();
            let slot = cache.allocate_seq().expect("a free recurrent slot") as u32;
            cache.set_state_indices(Some(Tensor::new(&[slot], &Device::Cpu)?));
            cache.set_state_indices_host(Some(vec![slot]));
        }
        Ok(model)
    }

    /// Logits of the last token after feeding `toks` in chunks of the given lengths.
    fn last_logits(bytes: &[u8], toks: &[u32], chunks: &[usize]) -> Result<Vec<f32>> {
        let model = model(bytes)?;
        let (mut at, mut logits) = (0, None);
        for &len in chunks {
            let ids = Tensor::new(&toks[at..at + len], &Device::Cpu)?.unsqueeze(0)?;
            let prior = vec![toks[at.saturating_sub(2)..at].to_vec()];
            logits = Some(model.forward(&ids, &prior, &[at], vec![(len - 1, 1)], None)?);
            at += len;
        }
        logits
            .expect("at least one chunk")
            .flatten_all()?
            .to_vec1::<f32>()
    }

    // The EOS token (9) sits mid-sequence, so the hash's segment rule is crossed on the way.
    const TOKS: [u32; 7] = [1, 5, EOS, 3, 7, 2, 11];

    #[test]
    fn loads_and_yields_finite_logits() -> Result<()> {
        let bytes = gguf(7)?;
        let logits = last_logits(&bytes, &TOKS, &[TOKS.len()])?;
        assert_eq!(logits.len(), VOCAB);
        assert!(logits.iter().all(|x| x.is_finite()), "{logits:?}");
        // The indexer's budget caps the context.
        assert_eq!(model(&bytes)?.max_seq_len, 32);
        Ok(())
    }

    // One pass, chunks, and a token at a time carry the same state through every cache: the
    // n-gram history and its hash prior, the delta-net pools, and attention's KV. Prefill chunks
    // agree to float noise. A single-token forward takes the quantized decode kernels, whose
    // rounding differs from prefill's, so runs with such steps get a looser bound; a state carried
    // wrongly misses either by orders of magnitude.
    #[test]
    fn chunked_and_stepped_forwards_match_one_pass() -> Result<()> {
        let bytes = gguf(11)?;
        let whole = last_logits(&bytes, &TOKS, &[TOKS.len()])?;
        let runs: [(&[usize], f32); 5] = [
            (&[3, 4], 1e-5),
            (&[2, 2, 3], 1e-5),
            (&[2, 1, 4], 2e-3),
            (&[6, 1], 2e-3),
            (&[1; 7], 2e-3),
        ];
        for (chunks, bound) in runs {
            let got = last_logits(&bytes, &TOKS, chunks)?;
            let worst = got
                .iter()
                .zip(&whole)
                .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
                .fold(0f32, f32::max);
            assert!(
                worst <= bound,
                "chunks {chunks:?}: worst relative error {worst:e}"
            );
        }
        Ok(())
    }

    // ---------------------------------------------------------------------------------------
    // The safetensors constructor
    // ---------------------------------------------------------------------------------------

    use crate::models::qwen4exp::manifest;
    use crate::models::qwen4exp::tests::{
        assert_close, config, fixture, gpu, rows, snapshot, tiny_checkpoint, tiny_config,
        vb as snapshot_vb, weights, Recorder, Tol, CHUNKS,
    };

    fn tiny_model(dev: &Device, dtype: DType) -> Result<(ModelWeights, std::collections::BTreeSet<String>, tempfile::TempDir)> {
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let files = tiny_checkpoint(dir.path())?;
        let (vb, names) = Recorder::over(&files, dtype, dev)?;
        let model = ModelWeights::new(
            &tiny_config(),
            vb,
            &files,
            Box::new(DummyDeviceMapper { nm_device: dev.clone() }),
            AttentionImplementation::Eager,
            dtype,
        )?;
        {
            let mut cache = model.cache.hybrid();
            let slot = cache.allocate_seq().expect("a free recurrent slot") as u32;
            cache.set_state_indices(Some(Tensor::new(&[slot], dev)?));
            cache.set_state_indices_host(Some(vec![slot]));
        }
        let read = names.lock().unwrap().clone();
        Ok((model, read, dir))
    }

    /// The loader reads exactly the manifest, apart from the table the n-gram block maps itself.
    #[test]
    fn loader_takes_the_manifest() -> Result<()> {
        let mut devs = vec![Device::Cpu];
        if let Ok(d) = Device::new_cuda(0) {
            devs.push(d);
        }
        for dev in devs {
            let (_, read, _dir) = tiny_model(&dev, DType::BF16)?;
            let want: std::collections::BTreeSet<String> = manifest(&tiny_config().text_config)?
                .into_iter()
                .map(|e| e.name)
                .filter(|n| !n.contains(".shard_") && !n.ends_with("ngram_embedding.weight_scale"))
                .collect();
            let missing: Vec<_> = want.difference(&read).collect();
            let extra: Vec<_> = read.difference(&want).collect();
            assert!(missing.is_empty() && extra.is_empty(), "{dev:?}: missing {missing:?}, extra {extra:?}");
        }
        Ok(())
    }

    fn tiny_last_logits(toks: &[u32], chunks: &[usize], dtype: DType) -> Result<Vec<f32>> {
        let (model, _, _dir) = tiny_model(&Device::Cpu, dtype)?;
        let (mut at, mut logits) = (0, None);
        for &len in chunks {
            let ids = Tensor::new(&toks[at..at + len], &Device::Cpu)?.unsqueeze(0)?;
            let prior = vec![toks[at.saturating_sub(2)..at].to_vec()];
            logits = Some(model.forward(&ids, &prior, &[at], vec![(len - 1, 1)], None)?);
            at += len;
        }
        logits.expect("a chunk").to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    }

    /// Chunks and single steps carry every cache of the safetensors model as one pass does.
    #[test]
    fn tiny_chunked_matches_one_pass() -> Result<()> {
        const T: [u32; 7] = [1, 5, 9, 3, 7, 2, 11];
        let whole = tiny_last_logits(&T, &[T.len()], DType::F32)?;
        assert!(whole.iter().all(|v| v.is_finite()));
        let runs: [(&[usize], f32); 4] = [(&[3, 4], 1e-5), (&[2, 2, 3], 1e-5), (&[6, 1], 2e-3), (&[1; 7], 2e-3)];
        for (chunks, bound) in runs {
            let got = tiny_last_logits(&T, chunks, DType::F32)?;
            let worst = got
                .iter()
                .zip(&whole)
                .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
                .fold(0f32, f32::max);
            assert!(worst <= bound, "chunks {chunks:?}: worst relative error {worst:e}");
        }
        Ok(())
    }

    fn tiny_json() -> String {
        serde_json::to_string(&tiny_config()).expect("config serializes")
    }

    #[test]
    fn from_causal_lm_name_maps_qwen4exp() {
        use crate::pipeline::NormalLoaderType;
        let t = NormalLoaderType::from_causal_lm_name("Qwen4ExpForConditionalGeneration").unwrap();
        assert_eq!(t, NormalLoaderType::Qwen4Exp);
        assert_eq!(t.to_string(), "qwen4exp");
        assert_eq!("qwen4exp".parse::<NormalLoaderType>().unwrap(), NormalLoaderType::Qwen4Exp);
    }

    /// The registered loader builds the tiny checkpoint from its config.json and runs it through
    /// `NormalModel::forward` to the same logits as the model built directly.
    #[test]
    fn normal_loader_runs_tiny_checkpoint() -> Result<()> {
        use crate::pipeline::loaders::{NormalLoadingMetadata, NormalModelLoader};
        use crate::pipeline::text_models_inputs_processor::FlashParams;
        use crate::pipeline::{ModelForwardContext, NormalModel};
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let files = tiny_checkpoint(dir.path())?;
        let vb = unsafe {
            hanzo_quant::ShardedSafeTensors::sharded(&files, DType::F32, &dev, None, Arc::new(|_| true))?
        };
        let meta = NormalLoadingMetadata {
            weights: files.clone(),
            mapper: Box::new(DummyDeviceMapper { nm_device: dev.clone() }),
            loading_isq: false,
            real_device: dev.clone(),
            multi_progress: Arc::new(indicatif::MultiProgress::new()),
            matformer_slicing_config: None,
        };
        let model = crate::pipeline::Qwen4ExpLoader
            .load(&tiny_json(), vb, meta, AttentionImplementation::Eager)
            .map_err(hanzo_ml::Error::msg)?;
        assert_eq!(model.config().num_layers, 4);
        assert_eq!(model.model_config().kv_layers(), vec![3]);
        {
            let mut cache = model.cache().hybrid();
            let slot = cache.allocate_seq().expect("a free recurrent slot") as u32;
            cache.set_state_indices(Some(Tensor::new(&[slot], &dev)?));
            cache.set_state_indices_host(Some(vec![slot]));
        }
        const T: [u32; 5] = [1, 5, 9, 3, 7];
        let ids = Tensor::new(&T, &dev)?.unsqueeze(0)?;
        let (offsets, lens, pos, prior) = ([0usize], [(T.len() - 1, 1)], [0usize], [vec![]]);
        let flash = FlashParams::empty(true);
        let mut ctx = ModelForwardContext::new(&offsets, &lens, &pos, None, &flash).with_prior(&prior);
        let got = NormalModel::forward(model.as_ref(), &ids, &mut ctx)?;
        let want = tiny_last_logits(&T, &[T.len()], DType::F32)?;
        let got = got.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(got, want);
        Ok(())
    }

    /// Only the attention layers hold paged K/V, and the device charge is the manifest's device
    /// entries: the n-gram table and hash are host-side and never charged.
    #[test]
    fn hybrid_kv_charges_attention_layers_only() -> Result<()> {
        use crate::pipeline::loaders::DeviceMappedModelLoader;
        let json = tiny_json();
        let loader = crate::pipeline::Qwen4ExpLoader;
        let kv = loader.model_config(&json).map_err(hanzo_ml::Error::msg)?.kv_layers();
        let want: Vec<usize> = (0..4).filter(|&i| tiny_config().text().attention(i)).collect();
        assert_eq!(kv, want);
        assert_eq!(kv, vec![3]);
        Ok(())
    }

    /// The loader's charge for the real snapshot, from its config alone: the manifest's device
    /// bytes plus what is held wider on the device (f32 norms and GDN side tensors, the banks'
    /// alpha and global scale), the attention layers' KV per token, and one sequence's state.
    #[test]
    #[ignore = "reads the Qwen3.8-Flash-Next snapshot's config"]
    fn qwen4exp_accounting_matches_checkpoint() -> Result<()> {
        use crate::pipeline::loaders::DeviceMappedModelLoader;
        let json = std::fs::read_to_string(snapshot().join("config.json")).map_err(hanzo_ml::Error::wrap)?;
        let (layers, rest) = crate::pipeline::Qwen4ExpLoader::sizes(&json).map_err(hanzo_ml::Error::msg)?;
        let estimate = layers.iter().sum::<usize>() + rest;
        println!("estimate {estimate} B ({:.3} GiB)", estimate as f64 / (1u64 << 30) as f64);
        assert!(
            (74_895_296_000..=74_895_296_000 + (32 << 20)).contains(&estimate),
            "estimate {estimate}"
        );
        let per_token = crate::pipeline::Qwen4ExpLoader
            .model_config(&json)
            .map_err(hanzo_ml::Error::msg)?
            .kv_cache_elements_per_token();
        assert_eq!(per_token * DType::BF16.size_in_bytes(), 24_576);
        assert_eq!(per_token * DType::F8E4M3.size_in_bytes(), 12_288);
        assert_eq!(config(&snapshot()).text().state_bytes_per_sequence(), 116_379_648);
        Ok(())
    }

    #[test]
    fn table_never_charged() -> Result<()> {
        use crate::models::qwen4exp::Role;
        let json = tiny_json();
        let (layers, rest) = crate::pipeline::Qwen4ExpLoader::sizes(&json).map_err(hanzo_ml::Error::msg)?;
        let entries = manifest(tiny_config().text())?;
        let host: usize = entries.iter().filter(|e| e.role == Role::Host).map(|e| e.bytes()).sum();
        let table: usize = entries
            .iter()
            .filter(|e| e.name.contains("ngram_embedding"))
            .map(|e| e.bytes())
            .sum();
        assert!(table > 0 && host >= table);
        let pure: Vec<_> = entries.into_iter().filter(|e| e.role == Role::Device).collect();
        let charged: usize = layers.iter().sum::<usize>() + rest;
        let device: usize = pure.iter().map(crate::pipeline::Qwen4ExpLoader::device_bytes).sum();
        assert_eq!(charged, device, "every charged byte is a device entry's");
        Ok(())
    }

    /// The loader's per-layer estimate is what loading the tiny model on CUDA allocates: the
    /// default mempool's used bytes after the load, less before, within allocator rounding.
    #[test]
    #[cfg(feature = "cuda")]
    fn loader_estimate_equals_allocation() -> Result<()> {
        use hanzo_ml::cuda_backend::cudarc::driver::sys;
        let Ok(dev) = Device::new_cuda(0) else { return Ok(()) };
        let Device::Cuda(cu) = &dev else { unreachable!() };
        let used = || -> u64 {
            cu.synchronize().expect("sync");
            let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
            let mut v = 0u64;
            unsafe {
                sys::cuDeviceGetDefaultMemPool(&mut pool, cu.cuda_stream().context().cu_device());
                sys::cuMemPoolGetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                    (&mut v as *mut u64).cast(),
                );
            }
            v
        };
        let json = tiny_json();
        let (layers, rest) = crate::pipeline::Qwen4ExpLoader::sizes(&json).map_err(hanzo_ml::Error::msg)?;
        let estimate = (layers.iter().sum::<usize>() + rest) as u64;
        let before = used();
        let (model, _, _dir) = tiny_model(&dev, DType::BF16)?;
        let cache = cache_bytes(&model)?;
        let after = used();
        let weights = after - before - cache;
        let tensors = manifest(tiny_config().text())?.len() as u64;
        println!("estimate {estimate}, allocated {weights} (+{cache} cache) over {tensors} tensors");
        // the pool rounds each allocation up; nothing beyond that may differ
        assert!(weights >= estimate, "allocated {weights} under the estimate {estimate}");
        assert!(weights - estimate <= tensors * 512, "allocated {weights}, estimate {estimate}");
        drop(model);
        Ok(())
    }

    /// Device bytes the model's recurrent pools hold (K/V grows on first use).
    #[cfg(feature = "cuda")]
    fn cache_bytes(model: &ModelWeights) -> Result<u64> {
        let cache = model.cache.hybrid();
        let bytes = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        Ok(cache
            .caches
            .iter()
            .filter_map(|c| c.as_recurrent_pool())
            .map(|p| bytes(&p.conv_state) + bytes(&p.recurrent_state))
            .sum())
    }

    /// Layers 0-3 of the real snapshot one at a time, each over [16, 7, 1] and dropped before
    /// the next loads, then the head: the streams after each layer and the logits.
    #[test]
    #[ignore = "reads the Qwen3.8-Flash-Next snapshot; needs a CUDA device"]
    fn golden_chain() -> Result<()> {
        let g = gpu(1.55);
        let dev = g.dev.clone();
        let cfg = config(&snapshot());
        let props = cfg.props();
        let text = cfg.text();
        let files = weights(&snapshot());
        let lm = snapshot_vb(&dev)?.pp("model.language_model");
        let rotary = Arc::new(Qwen3VLRotaryEmbedding::new(props.rope_freq_base, props.rot_dim, &dev, props.mrope_section.clone())?);
        let layers = 4;
        let attention = |i: usize| text.attention(i);
        let gdn = RecurrentLayerConfig {
            conv_dim: 2 * props.num_k_heads * props.head_k_dim + props.num_v_heads * props.head_v_dim,
            conv_width: props.conv_kernel,
            state_dims: vec![props.num_v_heads, props.head_k_dim, props.head_v_dim],
            conv_dtype: DType::BF16,
            state_dtype: DType::F32,
        };
        let side = RecurrentLayerConfig {
            conv_dim: text.hc_count * text.hidden_size,
            conv_width: (text.ple_conv_kernel_size - 1) * text.ngram_size,
            state_dims: vec![],
            conv_dtype: DType::BF16,
            state_dtype: DType::BF16,
        };
        let types: Vec<_> = (0..layers)
            .map(|i| if attention(i) { HybridLayerType::Attention } else { HybridLayerType::Recurrent })
            .collect();
        let mut pools = vec![gdn; types.iter().filter(|t| **t == HybridLayerType::Recurrent).count()];
        pools.push(side);
        let mut cache = HybridCache::new(HybridCacheConfig { layer_types: types, max_seq_len: props.max_seq_len, pools }, &dev)
            .map_err(|e| hanzo_ml::Error::Msg(e.to_string()))?;
        let slot = cache.allocate_seq().expect("slot");
        let toks = crate::models::qwen4exp::tests::tokens();
        let mut x = expand(&fixture("embed").to_device(&dev)?.unsqueeze(0)?, text.hc_count)?;
        for i in 0..layers {
            let layer = Layer::new(i, &lm.pp(format!("layers.{i}")), &cfg, &files, rotary.clone(), AttentionImplementation::Eager, DType::BF16)?;
            let mut outs = vec![];
            let mut at = 0;
            for len in CHUNKS {
                let xi = x.narrow(1, at, len)?;
                let ids = Tensor::new(&toks[at..at + len], &dev)?.unsqueeze(0)?;
                let offsets = [at];
                let mask = CausalMasker.make_causal_mask(&ids, &offsets as &dyn PastKvLenCache, DType::BF16, &CausalMaskConfig::gguf())?;
                let mask = DeviceMappedMask::from_single(mask);
                let cos_sin = text_mrope(&rotary, &dev, &offsets, len, DType::BF16, None)?;
                let g = match &layer.ngram {
                    Some(ng) => Some(ng.embed(&[toks[at.saturating_sub(2)..at].to_vec()], &[toks[at..at + len].to_vec()], &dev)?),
                    None => None,
                };
                let slots = move || Ok(PoolSlots::One { slot, offset: at });
                let mut step = Step {
                    cache: &mut cache,
                    slots: &slots,
                    trail: false,
                    mask: &mask,
                    cos_sin: &cos_sin,
                    paged: None,
                    kv_layer: 0,
                    ngram: g.as_ref(),
                    side: layers,
                };
                outs.push(layer.forward(i, xi, &mut step)?);
                at += len;
            }
            x = Tensor::cat(&outs, 1)?;
            let want = fixture(&format!("l{}.x", i + 1));
            assert_close(&format!("streams after layer {i}"), &x.reshape((24, ()))?, &want, Tol::most(8.0, 0.99, 0.9995));
            drop(layer);
        }
        let head = Mixer::new(&lm.pp("hyper_connection_mixer"), text.hc_count, props.rms_norm_eps)?;
        let h = head.mix(&head.norm(&x)?)?.squeeze(0)?;
        assert_close("head.h", &h, &fixture("head.h"), Tol::ulps(4.0));
        let w = rows("lm_head.weight", 0..8192).to_device(&dev)?;
        let lm_head = hanzo_quant::UnquantLinear::new(hanzo_quant::QuantMethodConfig::Unquantized(hanzo_nn::Linear::new(w, None)))?;
        let logits = lm_head.forward(&h)?;
        assert_close("head.logits", &logits, &fixture("head.logits"), Tol::ulps(4.0));
        Ok(())
    }
}
