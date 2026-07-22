#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::attention::{sdpa_naive_cache, AttentionMask, SdpaParams, VkGraphAttn};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMaskConfig, CausalMasker, QRmsNorm, RotaryEmbedding};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

struct Mlp {
    feed_forward_w1: Arc<dyn QuantMethod>,
    feed_forward_w2: Arc<dyn QuantMethod>,
    feed_forward_w3: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        let y = crate::ops::mul_and_act(&w1, &w3, crate::layers::Activation::Silu)?;
        self.feed_forward_w2.forward(&y)
    }
}

struct LayerWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    attention_norm: QRmsNorm,
    q_norm: Option<QRmsNorm>,
    k_norm: Option<QRmsNorm>,
    mlp: Mlp,
    ffn_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl LayerWeights {
    #[allow(clippy::too_many_arguments)]
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        positions: &Tensor,
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        vk_graph: Option<&VkGraphAttn>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k, v)
        };

        // Decode (single new token) reads the RoPE index from a DEVICE positions
        // tensor rather than baking a host `start_offsets` scalar into the launch —
        // the prerequisite for capturing the decode step into a CUDA/ROCm graph
        // (paged attention already reads context_lens on device). Prompt/prefill keeps
        // the host-offset form, which also feeds the same offsets into the causal mask.
        let (q, k) = if seq_len == 1 {
            let positions = if positions.device().same_device(q.device()) {
                positions.clone()
            } else {
                positions.to_device(q.device())?
            };
            match (&self.q_norm, &self.k_norm) {
                (Some(q_norm), Some(k_norm)) => self.rotary.forward_qk_norm_positions(
                    &q,
                    &k,
                    q_norm.weight(),
                    k_norm.weight(),
                    q_norm.eps(),
                    k_norm.eps(),
                    &positions,
                )?,
                _ => self.rotary.forward_positions(&q, &k, &positions)?,
            }
        } else {
            match (&self.q_norm, &self.k_norm) {
                (Some(q_norm), Some(k_norm)) => self.rotary.forward_qk_norm(
                    &q,
                    &k,
                    q_norm.weight(),
                    k_norm.weight(),
                    q_norm.eps(),
                    k_norm.eps(),
                    start_offsets,
                )?,
                _ => self.rotary.forward(&q, &k, start_offsets)?,
            }
        };

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        let y = match &self.paged_attn {
            Some(paged_attn) => {
                let ((key_cache, value_cache), input_metadata) = metadata.unwrap();
                paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    None,
                )?
            }
            None => sdpa_naive_cache(
                &q,
                &k,
                &v,
                mask,
                kv_cache,
                positions,
                &self.sdpa_params,
                vk_graph,
                b_sz,
                self.n_head,
                self.head_dim,
            )?,
        };

        let y = if mask.is_custom() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        let y = self.attention_wo.forward(&y.to_dtype(x.dtype())?)?;
        Ok(y)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

// qwen2 `llm` fields:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#llm
// NOTE: Types here do not match spec
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub key_length: usize,
    pub value_length: usize,
}

fn verify_qwen_arch(
    metadata: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>,
    expected_archs: &[&str],
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;

    if !expected_archs.contains(&actual_arch.as_str()) {
        hanzo_ml::bail!(
            "Expected `{:?}` architecture, got `{actual_arch}`.",
            expected_archs
        );
    }
    Ok(actual_arch)
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let _ = verify_qwen_arch(c.metadata, &["qwen2", "qwen3"])?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;

        // NOTE: Values are not aligned with GGUFv3 types
        // TODO: Normalize value types to spec
        let props = Self {
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
            key_length: c
                .get_value::<u32>("attention.key_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            value_length: c
                .get_value::<u32>("attention.value_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
        };

        Ok(props)
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
        // Parameter extraction from metadata.
        let meta = ct.get_metadata();
        let actual_arch = verify_qwen_arch(meta, &["qwen2", "qwen3"])?;

        let metadata = ContentMetadata {
            path_prefix: &actual_arch,
            metadata: meta,
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            key_length,
            value_length,
        } = PropsGGUF::try_from(metadata).or_else(|err| hanzo_ml::bail!("{err}"))?;

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new_dtype(
            ct.tensor("output_norm.weight", device)?,
            rms_norm_eps,
            dtype,
        )?;
        let output = if !ct.has_tensor("output.weight") {
            ct.tensor("token_embd.weight", device)?
        } else {
            ct.tensor("output.weight", device)?
        };
        let mut layers = Vec::with_capacity(block_count);

        let head_dim = key_length;
        if key_length != value_length {
            hanzo_ml::bail!(
                "Expected key_length == value_length, got {key_length} != {value_length}"
            );
        }

        let mut ropes = HashMap::new();
        for layer_idx in 0..block_count {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_freq_base,
                    head_dim,
                    max_seq_len,
                    device,
                    true,
                    dtype,
                )?),
            );
        }

        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();

            let attention_wq = ct.tensor(&format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(&format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(&format!("{prefix}.attn_v.weight"), device)?;

            let attention_bq = ct.tensor(&format!("{prefix}.attn_q.bias"), device);
            let attention_bk = ct.tensor(&format!("{prefix}.attn_k.bias"), device);
            let attention_bv = ct.tensor(&format!("{prefix}.attn_v.bias"), device);

            let attention_bq = if let Ok(bq) = attention_bq {
                Some(bq.dequantize(device)?)
            } else {
                None
            };

            let attention_bk = if let Ok(bk) = attention_bk {
                Some(bk.dequantize(device)?)
            } else {
                None
            };

            let attention_bv = if let Ok(bv) = attention_bv {
                Some(bv.dequantize(device)?)
            } else {
                None
            };

            let attention_wo = ct.tensor(&format!("{prefix}.attn_output.weight"), device)?;

            let feed_forward_w1 = ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_w2 = ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
            let feed_forward_w3 = ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
            let mlp = Mlp {
                feed_forward_w1: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(feed_forward_w1),
                    b: None,
                })?),
                feed_forward_w2: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(feed_forward_w2),
                    b: None,
                })?),
                feed_forward_w3: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(feed_forward_w3),
                    b: None,
                })?),
            };

            let q_norm = ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device);
            let k_norm = ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device);

            let (q_norm, k_norm) = match (q_norm, k_norm) {
                (Ok(q), Ok(k)) => {
                    let q_norm = QRmsNorm::new_dtype(q, rms_norm_eps, dtype)?;
                    let k_norm = QRmsNorm::new_dtype(k, rms_norm_eps, dtype)?;
                    (Some(q_norm), Some(k_norm))
                }
                _ => (None, None),
            };

            let attention_norm = ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?;
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            layers.push(LayerWeights {
                attention_wq: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wq),
                    b: attention_bq,
                })?),
                attention_wk: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wk),
                    b: attention_bk,
                })?),
                attention_wv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wv),
                    b: attention_bv,
                })?),
                attention_wo: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wo),
                    b: None,
                })?),
                attention_norm: QRmsNorm::new_dtype(attention_norm, rms_norm_eps, dtype)?,
                q_norm,
                k_norm,
                mlp,
                ffn_norm: QRmsNorm::new_dtype(ffn_norm, rms_norm_eps, dtype)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary: rotary.clone(),
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: head_count / head_count_kv,
                    softcap: None,
                    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                    sliding_window: None,
                    sinks: None,
                },
                dtype,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(output),
                b: None,
            })?),
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(block_count, max_seq_len)),
            max_seq_len,
            mapper: Some(mapper),
            dtype,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        self.forward_inner(x, start_offsets, context_lens, metadata, None, None)
    }

    /// Decode command-graph capture entry: runs the forward reading RoPE from the stable `positions`
    /// buffer and appending K/V at the device-offset slot + attending the shared span via `vk_graph`,
    /// so a capture in flight records a replayable single-token forward. Naive KV cache path only.
    #[cfg(feature = "vulkan")]
    pub fn forward_vk_graph(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        positions: &Tensor,
        vk_graph: &VkGraphAttn,
    ) -> Result<Tensor> {
        self.forward_inner(
            x,
            start_offsets,
            context_lens,
            None,
            Some(positions),
            Some(vk_graph),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        positions_override: Option<&Tensor>,
        vk_graph: Option<&VkGraphAttn>,
    ) -> Result<Tensor> {
        // Single-dtype residual: cast the F32 embedding output to the compute dtype so norms/matmul/
        // attention stay one dtype (drops the F32<->half attention casts, engages the fused qk-norm).
        let mut layer_in = self.tok_embeddings.forward(x)?.to_dtype(self.dtype)?;
        let cache = &mut self.cache.normal().0;
        let mask = CausalMasker.make_causal_mask(
            x,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.dtype,
            &CausalMaskConfig::gguf(),
        )?;
        let mask = if metadata
            .as_ref()
            .map(|(_, meta)| meta.is_first_prompt_chunk)
            .unwrap_or(true)
        {
            mask
        } else {
            AttentionMask::None
        };
        let mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(mask)
        };
        // RoPE positions come from a STABLE device tensor when a caller supplies one: the Vulkan
        // command-graph (`positions_override`) or the ROCm/CUDA decode graph (`metadata.rope_positions`)
        // refresh that buffer in place between replays, so the captured decode advances the rotation
        // instead of freezing at the warmup offset. Otherwise build a fresh one from the host offsets.
        // Only the seq_len==1 attention path reads it.
        let positions = match positions_override {
            Some(positions) => positions.clone(),
            None => match metadata
                .as_ref()
                .and_then(|(_, meta)| meta.rope_positions.as_ref())
                .and_then(|positions| positions.get(&self.device.location()))
            {
                Some(positions) => positions.clone(),
                None => {
                    let pos = start_offsets
                        .iter()
                        .copied()
                        .map(u32::try_from)
                        .collect::<std::result::Result<Vec<_>, _>>()
                        .map_err(hanzo_ml::Error::wrap)?;
                    Tensor::from_vec(pos, start_offsets.len(), &self.device)?
                }
            },
        };
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(
                &x,
                &mask.get(x.device()),
                start_offsets,
                &positions,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                vk_graph,
            )?;
            // Fused residual-add + ffn norm: sum = attn + residual (the new residual stream), x =
            // ffn_norm(sum). One dispatch on backends with the fused kernel (Vulkan/CUDA/ROCm), else a
            // separate add + rms_norm. Mirrors quantized_qwen3.
            let (sum, x) = layer.ffn_norm.forward_of_sum(&attn, residual)?;

            // MLP
            let residual = &sum;
            let x = layer.mlp.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.forward(&x.contiguous()?)
    }

    /// Build the shared decode command-graph attention buffers (scale + meta) for the current KV cache
    /// geometry. Every layer has the identical head/kv/head_dim shape and one contiguous cache, so a
    /// single [`VkGraphAttn`] serves the whole forward; `seq_k` is the initial attended span (advanced
    /// per replay via [`VkGraphAttn::set_seq_k`]).
    #[cfg(feature = "vulkan")]
    pub fn vk_build_graph_attn(&self, seq_k: usize) -> Result<VkGraphAttn> {
        let Device::Vulkan(dev) = &self.device else {
            hanzo_ml::bail!("vulkan decode graph: model is not on a vulkan device");
        };
        let l = self
            .layers
            .first()
            .ok_or_else(|| hanzo_ml::Error::msg("vulkan decode graph: model has no layers"))?;
        let capacity = self
            .vk_kv_capacity()
            .ok_or_else(|| hanzo_ml::Error::msg("vulkan decode graph: KV cache not allocated"))?;
        dev.new_graph_attn(
            l.n_head,
            l.n_kv_head,
            l.head_dim,
            capacity,
            l.sdpa_params.softmax_scale,
            false,
            seq_k,
        )
    }

    /// The contiguous KV cache capacity (max cached tokens) the graph is captured against; `None` until
    /// an eager warmup step allocates the cache. A captured graph is valid only while the write slot
    /// stays below this — a grow reallocates the cache buffer and invalidates the graph.
    #[cfg(feature = "vulkan")]
    pub fn vk_kv_capacity(&self) -> Option<usize> {
        let guard = self.cache.normal();
        match guard.0.first() {
            Some(KvCache::Normal { k, .. }) if k.all_data.is_some() => Some(k.capacity_seq_len),
            _ => None,
        }
    }

    /// Advance every layer's host KV length after a graph replay wrote the token's K/V on-device, so the
    /// cache stays consistent for a later eager fallback or capacity-boundary recapture.
    #[cfg(feature = "vulkan")]
    pub fn vk_advance_kv_len(&self, len: usize) -> Result<()> {
        let mut guard = self.cache.normal();
        for kv in guard.0.iter_mut() {
            if let KvCache::Normal { k, v } = kv {
                k.set_len(len)?;
                v.set_len(len)?;
            }
        }
        Ok(())
    }
}
