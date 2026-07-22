#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{GgufMatMul, QuantMethod, QuantMethodConfig, RingPipeline};

use crate::attention::{sdpa_naive_cache, AttentionMask, SdpaParams, VkGraphAttn};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMaskConfig, CausalMasker, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::pipeline_parallel::{
    pp_head_forward, use_pipeline_parallel, PipelineParallelModel, RingLayout,
};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

// Default fallback for models that don't specify context_length
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
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
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
        flash_params: &FlashParams,
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

        // Decode (single new token) uses a DEVICE positions tensor so the RoPE
        // index is read from device memory rather than baked into the launch
        // from a host `start_offsets` scalar — the prerequisite for capturing
        // the decode step (paged attention already reads context_lens on the
        // device). The prompt/prefill path keeps the host-offset form, which
        // feeds the same offsets into the causal mask.
        let (q, k) = if seq_len == 1 {
            // Keep positions on the same device as q/k (handles device mapping).
            let positions = if positions.device().same_device(q.device()) {
                positions.clone()
            } else {
                positions.to_device(q.device())?
            };
            self.rotary.forward_qk_norm_positions(
                &q,
                &k,
                self.q_norm.weight(),
                self.k_norm.weight(),
                self.q_norm.eps(),
                self.k_norm.eps(),
                &positions,
            )?
        } else {
            self.rotary.forward_qk_norm(
                &q,
                &k,
                self.q_norm.weight(),
                self.k_norm.weight(),
                self.q_norm.eps(),
                self.k_norm.eps(),
                start_offsets,
            )?
        };

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        let y = if let Some(g) = vk_graph {
            // Decode command-graph capture: drive the naive KV cache with the device-offset K/V
            // append + shared-span fused SDPA (`sdpa_naive_cache` graph path), so the recorded
            // single-token forward replays with the advancing write slot and attended span.
            sdpa_naive_cache(
                &q,
                &k,
                &v,
                mask,
                kv_cache,
                positions,
                &self.sdpa_params,
                Some(g),
                b_sz,
                self.n_head,
                self.head_dim,
            )?
        } else {
            match &self.paged_attn {
            Some(paged_attn) => {
                // One-time confirmation that the Qwen3 GGUF decode is running
                // through PagedAttention (not the SingleCache + Sdpa eager path)
                // and that the decode RoPE uses device positions. Emitted once.
                if seq_len == 1 {
                    static PAGED_DECODE_ONCE: std::sync::Once = std::sync::Once::new();
                    PAGED_DECODE_ONCE.call_once(|| {
                        eprintln!(
                            "[hanzo] qwen3 GGUF decode: PagedAttention ACTIVE; RoPE via device positions (graph-safe)"
                        );
                    });
                }
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
                    Some(flash_params),
                )?
            }
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(&q, &k, &v, mask, Some(flash_params), &self.sdpa_params)?
            }
            }
        };

        // CausalFlash and eager both return [b, heads, seq, dim]; only decode (None) needs no transpose.
        let y = if !matches!(mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        let y = self.attention_wo.forward(&y.to_dtype(x.dtype())?)?;
        Ok(y)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_block(
        &self,
        x: Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        positions: &Tensor,
        kv_cache: &mut KvCache,
        flash_params: &FlashParams,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        vk_graph: Option<&VkGraphAttn>,
    ) -> Result<Tensor> {
        let residual = &x;
        let xn = self.attention_norm.forward(&x)?;
        let attn = self.forward_attn(
            &xn,
            mask,
            start_offsets,
            positions,
            kv_cache,
            flash_params,
            metadata,
            vk_graph,
        )?;
        let (sum, xn) = self.ffn_norm.forward_of_sum(&attn, residual)?;
        let residual = &sum;
        let xn = self.mlp.forward(&xn)?;
        xn + residual
    }
}

pub struct ModelWeights {
    tok_embeddings: Option<Embedding>,
    layers: Vec<LayerWeights>,
    norm: Option<QRmsNorm>,
    output: Option<Arc<dyn QuantMethod>>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    pp: Option<Arc<RingPipeline>>,
}

// qwen3 `llm` fields:
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

fn verify_qwen3_arch(
    metadata: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>,
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;

    // `qwen3vl` is the dense Qwen3-VL text backbone: identical text architecture to
    // `qwen3` (same tensors, q/k-norm, GQA). Its LLM GGUF is loaded here image-blind
    // (interleaved-MRoPE collapses to standard RoPE for text-only positions).
    if actual_arch != "qwen3" && actual_arch != "qwen3vl" {
        hanzo_ml::bail!("Expected `qwen3` or `qwen3vl` architecture, got `{actual_arch}`.");
    }
    Ok(actual_arch)
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let _ = verify_qwen3_arch(c.metadata)?;

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
        let actual_arch = verify_qwen3_arch(meta)?;

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

        let pp = if use_pipeline_parallel() {
            let config = hanzo_quant::RingConfig::load();
            Some(Arc::new(RingPipeline::from_config(&config)?))
        } else {
            None
        };
        let layout = match &pp {
            Some(_) => Some(RingLayout::new(block_count)?),
            None => None,
        };
        let local_range = layout.as_ref().map_or(0..block_count, RingLayout::local);
        let is_head = layout.as_ref().is_none_or(RingLayout::is_head);
        let local_len = local_range.end - local_range.start;

        let (tok_embeddings, norm, output) = if is_head {
            let qtok = ct.tensor("token_embd.weight", device)?;
            let tok = Embedding::new(qtok.dequantize(device)?, embedding_length);
            let norm = QRmsNorm::new_dtype(
                ct.tensor("output_norm.weight", device)?,
                rms_norm_eps,
                dtype,
            )?;
            let out = if !ct.has_tensor("output.weight") {
                ct.tensor("token_embd.weight", device)?
            } else {
                ct.tensor("output.weight", device)?
            };
            let out = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(out),
                b: None,
            })?) as Arc<dyn QuantMethod>;
            (Some(tok), Some(norm), Some(out))
        } else {
            (None, None, None)
        };
        let mut layers = Vec::with_capacity(local_len);

        let head_dim = key_length;
        if key_length != value_length {
            hanzo_ml::bail!(
                "Expected key_length == value_length, got {key_length} != {value_length}"
            );
        }

        let mut ropes = HashMap::new();
        for layer_idx in local_range.clone() {
            let ldev = if pp.is_some() {
                device
            } else {
                mapper.device_for(layer_idx, false).unwrap_or(device)
            };
            if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(ldev.location()) {
                e.insert(Arc::new(RotaryEmbedding::new(
                    rope_freq_base,
                    head_dim,
                    max_seq_len,
                    ldev,
                    true,
                    dtype,
                )?));
            }
        }

        for layer_idx in NiceProgressBar::<_, 'b'>(
            local_range.clone(),
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = if pp.is_some() {
                device
            } else {
                mapper.device_for(layer_idx, false).unwrap_or(device)
            };
            let rotary = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();

            let attention_wq = ct.tensor(&format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(&format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(&format!("{prefix}.attn_v.weight"), device)?;
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

            // Qwen3 always has q_norm and k_norm
            let q_norm = QRmsNorm::new_dtype(
                ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
                dtype,
            )?;
            let k_norm = QRmsNorm::new_dtype(
                ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
                dtype,
            )?;

            let attention_norm = ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?;
            let paged_attn = match &attention_mechanism {
                _ if pp.is_some() => None,
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            layers.push(LayerWeights {
                attention_wq: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wq),
                    b: None,
                })?),
                attention_wk: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wk),
                    b: None,
                })?),
                attention_wv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(attention_wv),
                    b: None,
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
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(local_len, max_seq_len)),
            max_seq_len,
            mapper: if pp.is_some() { None } else { Some(mapper) },
            dtype,
            pp,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        flash_params: &FlashParams,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        self.forward_inner(x, start_offsets, context_lens, flash_params, metadata, None, None)
    }

    /// Decode command-graph capture entry (Vulkan): run the single-token forward reading RoPE from the
    /// stable `positions` buffer and appending K/V at the device-offset slot + attending the shared span
    /// via `vk_graph`, so a capture in flight records a replayable single-token forward. Naive KV cache
    /// path only (no paged metadata). Mirrors `quantized_qwen3_moe::forward_vk_graph`.
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
            &FlashParams::empty(true),
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
        flash_params: &FlashParams,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        positions_override: Option<&Tensor>,
        vk_graph: Option<&VkGraphAttn>,
    ) -> Result<Tensor> {
        if self.pp.is_some() {
            return pp_head_forward(self, x, start_offsets, context_lens);
        }
        // Run the residual stream in the model compute dtype (not the F32 the embedding dequantizes
        // to) so the per-layer norm/matmul/attention chain stays single-dtype: this drops the
        // F32<->half round-trip casts around attention and lets the fused qk-norm-rope kernel accept
        // the (now matching-dtype) norm weights.
        let mut layer_in = self
            .tok_embeddings
            .as_ref()
            .unwrap()
            .forward(x)?
            .to_dtype(self.dtype)?;
        // Build the RoPE positions tensor once per step on the model device from
        // the host `start_offsets` (one position per sequence). The decode path
        // (seq_len == 1) reads these from device memory instead of baking the
        // offset into the launch; mirrors ModelForwardContext::rope_positions_from_offsets.
        //
        // When the caller supplies `metadata.rope_positions` (the ROCm/CUDA decode
        // graph path), use that *stable* device tensor verbatim instead of
        // synthesizing a fresh allocation here: a graph captures the kernels reading
        // a fixed buffer, and the graph runner refreshes that buffer's contents in
        // place between replays (see pipeline/rocm_graph.rs). Synthesizing a new
        // tensor each forward would freeze the captured positions at the warmup
        // value and corrupt every replayed token. Falls back to the host-offset
        // form for prefill and the non-graph decode path.
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
        let cache = &mut self.cache.normal().0;
        let mask = CausalMasker.make_causal_mask(
            x,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.dtype,
            &CausalMaskConfig::default(),
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
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }
            let dmask = mask.get(layer_in.device());
            layer_in = layer.forward_block(
                layer_in,
                &dmask,
                start_offsets,
                &positions,
                &mut cache[i],
                flash_params,
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                vk_graph,
            )?;
        }
        let x = self.norm.as_ref().unwrap().forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.as_ref().unwrap().forward(&x.contiguous()?)
    }

    fn run_local_layers(&self, h: &Tensor, offsets: &[usize]) -> Result<Tensor> {
        let positions = {
            let pos = offsets
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(hanzo_ml::Error::wrap)?;
            Tensor::from_vec(pos, offsets.len(), &self.device)?
        };
        let ids2d = h.narrow(2, 0, 1)?.squeeze(2)?;
        let mask = CausalMasker.make_causal_mask(
            &ids2d,
            &offsets as &dyn PastKvLenCache,
            self.dtype,
            &CausalMaskConfig::gguf(),
        )?;
        let mask = DeviceMappedMask::from_single(mask);
        let cache = &mut self.cache.normal().0;
        if !self.pp.as_ref().unwrap().is_head()
            && cache
                .first()
                .is_some_and(|c| c.current_seq_len() != offsets[0])
        {
            for c in cache.iter_mut() {
                c.reset();
            }
        }
        let mut layer_in = h.clone();
        let flash_params = FlashParams::empty(true);
        for (i, layer) in self.layers.iter().enumerate() {
            let dmask = mask.get(layer_in.device());
            layer_in = layer.forward_block(
                layer_in,
                &dmask,
                offsets,
                &positions,
                &mut cache[i],
                &flash_params,
                None,
                None,
            )?;
        }
        Ok(layer_in)
    }

    /// Build the shared decode command-graph attention buffers (scale + meta) for the current KV cache
    /// geometry. Every layer has the identical head/kv/head_dim shape and one contiguous cache, so a
    /// single [`VkGraphAttn`] serves the whole forward; `seq_k` is the initial attended span (advanced
    /// per replay via [`VkGraphAttn::set_seq_k`]). Mirrors `quantized_qwen3_moe`.
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
    /// an eager warmup step allocates the cache. A grow reallocates the buffer and invalidates the graph.
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

impl PipelineParallelModel for ModelWeights {
    fn ring(&self) -> &Arc<RingPipeline> {
        self.pp.as_ref().expect("pipeline parallel not enabled")
    }
    fn pp_device(&self) -> &Device {
        &self.device
    }
    fn pp_dtype(&self) -> DType {
        self.dtype
    }
    fn pp_embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.as_ref().unwrap().forward(tokens)
    }
    fn pp_run_local(&self, h: &Tensor, offsets: &[usize]) -> Result<Tensor> {
        self.run_local_layers(h, offsets)
    }
    fn pp_norm_head(&self, h: &Tensor, context_lens: Vec<(usize, usize)>) -> Result<Tensor> {
        let x = self.norm.as_ref().unwrap().forward(h)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.as_ref().unwrap().forward(&x.contiguous()?)
    }
}
