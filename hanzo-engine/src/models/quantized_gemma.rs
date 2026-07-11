#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF Gemma / Gemma2 (`gemma`, `gemma2` archs) quantized dense models.
//!
//! Mirrors the safetensors `gemma`/`gemma2` math: `(1 + w)` RMSNorm gating (baked into the GGUF
//! norm weights by llama.cpp's converter, so the plain `QRmsNorm` applies), `sqrt(hidden)` embedding
//! scale, GeGLU (`gelu_pytorch_tanh`), and `head_dim = key_length` (decoupled from `hidden/heads`).
//! Gemma2 additionally runs post-attention and post-feedforward norms, alternating sliding/full
//! attention, an attention-logit soft-cap, and a final-logit soft-cap. Tensor/metadata names follow
//! llama.cpp's `LLM_ARCH_GEMMA`/`LLM_ARCH_GEMMA2`.
//!
//! Single-dtype: the residual stream runs in the compute dtype (norm weights held in `dtype` via
//! `QRmsNorm::new_dtype`, embedding cast to `dtype`), so the per-layer chain stays half-precision and
//! avoids the F32<->half prefill casts.

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{Activation, CausalMaskConfig, CausalMasker, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache, NormalCacheType};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{softcap, GgufMatMul, QuantMethod, QuantMethodConfig};

const DEFAULT_MAX_SEQ_LEN: u32 = 8192;
const GEMMA_ROPE_FREQ_BASE: f32 = 10_000.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Which {
    Gemma,
    Gemma2,
}

fn gguf_linear(w: hanzo_ml::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(w),
        b: None,
    })?))
}

struct Mlp {
    ffn_gate: Arc<dyn QuantMethod>,
    ffn_up: Arc<dyn QuantMethod>,
    ffn_down: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.ffn_gate.forward(xs)?;
        let up = self.ffn_up.forward(xs)?;
        let y = crate::ops::mul_and_act(&gate, &up, Activation::GeluPytorchTanh)?;
        self.ffn_down.forward(&y)
    }
}

struct LayerWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    attention_norm: QRmsNorm,
    ffn_norm: QRmsNorm,
    post_attention_norm: Option<QRmsNorm>,
    post_ffn_norm: Option<QRmsNorm>,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    is_sliding: bool,
    dtype: DType,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        positions: &Tensor,
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
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

        // RoPE runs in F32 (its cos/sin table dtype) for precision and to match the plain rope op;
        // the single-dtype residual keeps the rest of the layer in the compute dtype. No fused
        // qk-norm-rope kernel here (Gemma has no q/k norm), so the F32 cast is explicit.
        let (q, k) = (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?);
        // Decode reads the RoPE index from the device positions tensor; prefill uses host offsets.
        let (q, k) = if seq_len == 1 {
            let positions = if positions.device().same_device(q.device()) {
                positions.clone()
            } else {
                positions.to_device(q.device())?
            };
            self.rotary.forward_positions(&q, &k, &positions)?
        } else {
            self.rotary.forward(&q, &k, start_offsets)?
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
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa_params)?
            }
        };

        let y = if !matches!(mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        self.attention_wo.forward(&y.to_dtype(x.dtype())?)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    embed_scale: f64,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    final_logit_softcap: Option<f32>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    sliding_window: usize,
}

#[allow(dead_code)]
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub head_dim: usize,
    pub value_length: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub query_pre_attn_scalar: usize,
    pub sliding_window: usize,
    pub attn_logit_softcap: Option<f32>,
    pub final_logit_softcap: Option<f32>,
}

fn verify_arch(metadata: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>) -> Result<Which> {
    use crate::utils::gguf_metadata::TryValueInto;
    let arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    match arch.as_str() {
        "gemma" => Ok(Which::Gemma),
        "gemma2" => Ok(Which::Gemma2),
        other => hanzo_ml::bail!("Expected `gemma` or `gemma2` architecture, got `{other}`."),
    }
}

impl PropsGGUF {
    fn new(c: &ContentMetadata) -> std::result::Result<Self, anyhow::Error> {
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
        let head_dim = c
            .get_value::<u32>("attention.key_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(embed_len / head_count);

        Ok(Self {
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            head_dim,
            value_length: c
                .get_value::<u32>("attention.value_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(head_dim),
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            query_pre_attn_scalar: c
                .get_value::<u32>("attention.query_pre_attn_scalar")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(head_dim),
            sliding_window: c
                .get_value::<u32>("attention.sliding_window")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
            attn_logit_softcap: c.get_value::<f32>("attn_logit_softcapping").ok(),
            final_logit_softcap: c.get_value::<f32>("final_logit_softcapping").ok(),
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
        let which = verify_arch(ct.get_metadata())?;
        let arch = match which {
            Which::Gemma => "gemma",
            Which::Gemma2 => "gemma2",
        };
        let metadata = ContentMetadata {
            path_prefix: arch,
            metadata: ct.get_metadata(),
        };
        let props = PropsGGUF::new(&metadata).or_else(|err| hanzo_ml::bail!("{err}"))?;

        if props.head_dim != props.value_length {
            hanzo_ml::bail!(
                "Expected key_length == value_length, got {} != {}",
                props.head_dim,
                props.value_length
            );
        }

        let head_dim = props.head_dim;
        let tok = ct.tensor("token_embd.weight", device)?.dequantize(device)?;
        let tok_embeddings = Embedding::new(tok, props.embedding_length);
        let norm = QRmsNorm::new_dtype(
            ct.tensor("output_norm.weight", device)?,
            props.rms_norm_eps,
            dtype,
        )?;
        // Gemma ties the output head to the input embedding (no `output.weight`).
        let output = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };

        let mut ropes = HashMap::new();
        for layer_idx in 0..props.block_count {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(device.location()) {
                e.insert(Arc::new(RotaryEmbedding::new(
                    GEMMA_ROPE_FREQ_BASE,
                    head_dim,
                    props.max_seq_len,
                    device,
                    true,
                    DType::F32,
                )?));
            }
        }

        let mut layers = Vec::with_capacity(props.block_count);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..props.block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();

            let attention_wq = gguf_linear(ct.tensor(&format!("{prefix}.attn_q.weight"), device)?)?;
            let attention_wk = gguf_linear(ct.tensor(&format!("{prefix}.attn_k.weight"), device)?)?;
            let attention_wv = gguf_linear(ct.tensor(&format!("{prefix}.attn_v.weight"), device)?)?;
            let attention_wo =
                gguf_linear(ct.tensor(&format!("{prefix}.attn_output.weight"), device)?)?;
            let mlp = Mlp {
                ffn_gate: gguf_linear(ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?)?,
                ffn_up: gguf_linear(ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?)?,
                ffn_down: gguf_linear(ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?)?,
            };
            let attention_norm = QRmsNorm::new_dtype(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                props.rms_norm_eps,
                dtype,
            )?;
            let ffn_norm = QRmsNorm::new_dtype(
                ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?,
                props.rms_norm_eps,
                dtype,
            )?;
            // Gemma2 wraps each sublayer output in a post-norm before the residual add.
            let (post_attention_norm, post_ffn_norm) = if which == Which::Gemma2 {
                (
                    Some(QRmsNorm::new_dtype(
                        ct.tensor(&format!("{prefix}.post_attention_norm.weight"), device)?,
                        props.rms_norm_eps,
                        dtype,
                    )?),
                    Some(QRmsNorm::new_dtype(
                        ct.tensor(&format!("{prefix}.post_ffw_norm.weight"), device)?,
                        props.rms_norm_eps,
                        dtype,
                    )?),
                )
            } else {
                (None, None)
            };

            // Gemma2 alternates sliding/full attention starting with sliding (layer 0).
            let is_sliding =
                which == Which::Gemma2 && props.sliding_window > 0 && layer_idx % 2 == 0;
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            layers.push(LayerWeights {
                attention_wq,
                attention_wk,
                attention_wv,
                attention_wo,
                attention_norm,
                ffn_norm,
                post_attention_norm,
                post_ffn_norm,
                mlp,
                n_head: props.head_count,
                n_kv_head: props.head_count_kv,
                head_dim,
                rotary: rotary.clone(),
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: props.head_count / props.head_count_kv,
                    softcap: props.attn_logit_softcap,
                    softmax_scale: 1.0 / (props.query_pre_attn_scalar as f32).sqrt(),
                    sliding_window: if is_sliding {
                        Some(props.sliding_window)
                    } else {
                        None
                    },
                    sinks: None,
                },
                is_sliding,
                dtype,
            });
        }

        let cache_types: Vec<NormalCacheType> = (0..props.block_count)
            .map(|layer_idx| {
                if which == Which::Gemma2 && props.sliding_window > 0 && layer_idx % 2 == 0 {
                    NormalCacheType::SlidingWindow {
                        window: props.sliding_window,
                    }
                } else {
                    NormalCacheType::Normal {
                        max_seq_len: props.max_seq_len,
                    }
                }
            })
            .collect();

        Ok(Self {
            tok_embeddings,
            embed_scale: (props.embedding_length as f64).sqrt(),
            layers,
            norm,
            output: gguf_linear(output)?,
            final_logit_softcap: props.final_logit_softcap,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            max_seq_len: props.max_seq_len,
            mapper: Some(mapper),
            dtype,
            sliding_window: props.sliding_window,
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
        let mut layer_in =
            (self.tok_embeddings.forward(x)?.to_dtype(self.dtype)? * self.embed_scale)?;
        let cache = &mut self.cache.normal().0;
        let positions = match metadata
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
        };
        let past_len = metadata
            .as_ref()
            .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
            .unwrap_or(cache as &dyn PastKvLenCache);
        let is_first = metadata
            .as_ref()
            .map(|(_, meta)| meta.is_first_prompt_chunk)
            .unwrap_or(true);

        let causal_mask =
            CausalMasker.make_causal_mask(x, past_len, self.dtype, &CausalMaskConfig::default())?;
        let sliding_mask = CausalMasker.make_causal_mask(
            x,
            past_len,
            self.dtype,
            &CausalMaskConfig {
                sliding_window: Some(self.sliding_window),
                ..Default::default()
            },
        )?;
        let (causal_mask, sliding_mask) = if is_first {
            (causal_mask, sliding_mask)
        } else {
            (AttentionMask::None, AttentionMask::None)
        };
        let (causal_mask, sliding_mask) = if let Some(ref mapper) = self.mapper {
            (
                DeviceMappedMask::new(causal_mask, &**mapper)?,
                DeviceMappedMask::new(sliding_mask, &**mapper)?,
            )
        } else {
            (
                DeviceMappedMask::from_single(causal_mask),
                DeviceMappedMask::from_single(sliding_mask),
            )
        };

        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }
            let residual = layer_in;
            let xn = layer.attention_norm.forward(&residual)?;
            let dev = xn.device();
            let mask = if layer.is_sliding {
                sliding_mask.get(dev)
            } else {
                causal_mask.get(dev)
            };
            let attn = layer.forward_attn(
                &xn,
                &mask,
                start_offsets,
                &positions,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
            )?;
            let attn = match &layer.post_attention_norm {
                Some(n) => n.forward(&attn)?,
                None => attn,
            };
            let x = (attn.to_dtype(residual.dtype())? + &residual)?;

            let residual = x;
            let xn = layer.ffn_norm.forward(&residual)?;
            let xn = layer.mlp.forward(&xn)?;
            let xn = match &layer.post_ffn_norm {
                Some(n) => n.forward(&xn)?,
                None => xn,
            };
            layer_in = (xn.to_dtype(residual.dtype())? + &residual)?;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        let logits = self.output.forward(&x.contiguous()?)?;
        match self.final_logit_softcap {
            Some(cap) => {
                let dt = logits.dtype();
                softcap(&logits, cap)?.to_dtype(dt)
            }
            None => Ok(logits),
        }
    }
}
