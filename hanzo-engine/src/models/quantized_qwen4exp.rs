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
use hanzo_quant::QuantMethod;

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
/// own hyper-connection branch.
struct Layer {
    mix: LayerImpl,
    attn: Branch,
    ffn: Branch,
    moe: FusedMoe,
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    /// Residual streams.
    streams: usize,
    ngram: Ngram,
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
        let ngram = Ngram::load(&mut ct, hash, ngram_layer, f64::from(eps), ngram_dev)?;

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
                LayerImpl::FullAttention(GatedFullAttention::load(
                    &mut ct, &prefix, &props, rotary, paged, dev, dtype,
                )?)
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
            });
        }

        // One pool per gated delta-net layer, then the n-gram history as a side pool. The
        // recurrent state is f32 (the config's `mamba_ssm_dtype`); the conv history is not.
        let layer_types: Vec<HybridLayerType> = (0..props.block_count)
            .map(|i| {
                if attention(i) {
                    HybridLayerType::Attention
                } else {
                    HybridLayerType::Recurrent
                }
            })
            .collect();
        let gdn = RecurrentLayerConfig {
            conv_dim,
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

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, props.embedding_length),
            streams,
            ngram,
            ngram_layer,
            layers,
            head,
            output,
            rotary,
            device: device.clone(),
            cache: EitherCache::Hybrid(Arc::new(Mutex::new(cache))),
            max_seq_len: props.max_seq_len,
            mapper: Some(mapper),
            dtype,
        })
    }
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
        let g = self
            .ngram
            .embed(prior, &input_ids.to_vec2::<u32>()?, e.device())?;
        let mut x = expand(&e, self.streams)?;

        let mut hybrid_cache = self.cache.hybrid();
        let trail = hybrid_cache.records_trail(seq_len);
        let state_indices = hybrid_cache.state_indices().cloned();
        let state_indices_host: Option<Vec<u32>> =
            hybrid_cache.state_indices_host().map(|s| s.to_vec());
        let slots = || -> Result<PoolSlots<'_>> {
            if b_sz == 1 {
                // One sequence reads its slot on the host, with no device sync (as Qwen3.5).
                let slot = state_indices_host
                    .as_ref()
                    .and_then(|s| s.first().copied())
                    .ok_or_else(|| hanzo_ml::Error::msg("missing host recurrent state index"))?;
                Ok(PoolSlots::One {
                    slot: slot as usize,
                    offset: seqlen_offsets.first().copied().unwrap_or(0),
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

        // The paged cache holds one K/V pair per attention layer, read at its ordinal.
        let mut kv_layer = 0;
        let side = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                x = mapper.map(x, i)?;
            }
            if i == self.ngram_layer {
                let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(side) else {
                    hanzo_ml::bail!("hybrid cache has no n-gram pool at {side}");
                };
                let g = g.to_device(x.device())?;
                let delta = forward_pooled(pool, slots()?, side, trail, |cache| {
                    self.ngram.forward(&x, &g, cache)
                })?;
                x = (x + delta)?;
            }
            x = match &layer.mix {
                LayerImpl::FullAttention(attn) => {
                    let paged = metadata
                        .as_ref()
                        .map(|(kv_cache, meta)| (kv_cache[kv_layer].clone(), *meta));
                    kv_layer += 1;
                    let Some(HybridLayerCache::Attention(kv_cache)) = hybrid_cache.get_mut(i)
                    else {
                        hanzo_ml::bail!("hybrid cache layer {i} is not attention");
                    };
                    let mask = mask.get(x.device());
                    layer
                        .attn
                        .apply(&x, |u| attn.forward(u, &mask, &cos_sin, kv_cache, paged))?
                }
                LayerImpl::LinearAttention(gdn) => {
                    let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(i) else {
                        hanzo_ml::bail!("hybrid cache layer {i} is not recurrent");
                    };
                    let slots = slots()?;
                    layer.attn.apply(&x, |u| {
                        forward_pooled(pool, slots, i, trail, |cache| gdn.forward(u, cache))
                    })?
                }
            };
            x = layer.ffn.apply(&x, |v| layer.moe.forward(v))?;
            // Metal recycles pooled buffers without a completion check; drain each prefill layer
            // as Qwen3.5 does.
            if seq_len > 1 && x.device().is_metal() {
                x.device().synchronize()?;
            }
        }

        let x = x.to_device(&self.device)?;
        let h = self.head.mix(&self.head.norm(&x)?)?;
        let h = extract_logits(&h, context_lens)?;
        self.output.forward(&h.contiguous()?)
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
}
