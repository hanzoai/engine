#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF GLM-4.5/4.6 MoE (`glm4moe` arch) quantized model.
//!
//! Standard GQA attention (q/k/v biases, optional q/k RMSNorm, partial NeoX RoPE) with the shared
//! DeepSeek-style sigmoid/group MoE gate (`models::gguf_moe`). Covers GLM-4.5-Air, GLM-4.5/4.6 and
//! GLM-4.7 (355B). The trailing `nextn_predict_layers` MTP block is skipped for text generation.
//! Tensor/metadata names follow llama.cpp's `LLM_ARCH_GLM4_MOE`.
//!
//! GLM-4.7-Flash is a *different* GGUF arch (`deepseek2`, MLA) — see `quantized_deepseek2`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMaskConfig, CausalMasker, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::models::gguf_moe::{build_moe_or_mlp, gguf_linear, MoeOrMlp, MoeParams};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use hanzo_ml::quantized::QTensor;
use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

const DEFAULT_MAX_SEQ_LEN: u32 = 131072;
const EXPERT_GATING_SIGMOID: u32 = 2;

fn gguf_linear_b(w: QTensor, bias: Option<Tensor>) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(w),
        b: bias,
    })?))
}

/// Load an optional bias tensor as F32 — matches the F32 activations from the F32-weighted RMSNorm
/// and GgufMatMul output, so the fused bias-add never hits a dtype mismatch.
fn opt_bias<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    name: &str,
    device: &Device,
    _dtype: DType,
) -> Result<Option<Tensor>> {
    if ct.has_tensor(name) {
        Ok(Some(
            ct.tensor(name, device)?
                .dequantize(device)?
                .to_dtype(DType::F32)?,
        ))
    } else {
        Ok(None)
    }
}

struct LayerWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    attention_norm: QRmsNorm,
    post_attention_norm: QRmsNorm,
    q_norm: Option<QRmsNorm>,
    k_norm: Option<QRmsNorm>,
    mlp: MoeOrMlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
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

        let (mut q, mut k, v) = if seq_len != 1 {
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

        // Optional q/k RMSNorm over head_dim (GLM-4.6), applied before RoPE.
        if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            q = q_norm.forward(&q.contiguous()?)?;
            k = k_norm.forward(&k.contiguous()?)?;
        }

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

        let y = if mask.is_custom() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        self.attention_wo.forward(&y.to_dtype(x.dtype())?)
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

#[allow(dead_code)]
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    pub head_dim: usize,
    pub moe: MoeParams,
    pub nextn_predict_layers: usize,
}

fn verify_arch(metadata: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>) -> Result<()> {
    use crate::utils::gguf_metadata::TryValueInto;
    let arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    if arch != "glm4moe" {
        hanzo_ml::bail!("Expected `glm4moe` architecture, got `{arch}`.");
    }
    Ok(())
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
            "rope.dimension_count",
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;
        let sigmoid_scoring =
            c.get_value::<u32>("expert_gating_func").unwrap_or(0) == EXPERT_GATING_SIGMOID;

        let moe = MoeParams {
            n_routed_experts: c.get_value::<u32>("expert_count")? as usize,
            num_experts_per_tok: c.get_value::<u32>("expert_used_count")? as usize,
            n_group: c
                .get_value::<u32>("expert_group_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(1),
            topk_group: c
                .get_value::<u32>("expert_group_used_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(1),
            routed_scaling_factor: c
                .get_value::<f32>("expert_weights_scale")
                .ok()
                .map(|x| x as f64)
                .unwrap_or(1.0),
            norm_topk_prob: c.get_value::<bool>("expert_weights_norm").ok().unwrap_or(true),
            sigmoid_scoring,
            n_shared_experts: c
                .get_value::<u32>("expert_shared_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
            leading_dense_block_count: c
                .get_value::<u32>("leading_dense_block_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
        };

        Ok(Self {
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
            rope_dim: c.get_value::<u32>("rope.dimension_count")? as usize,
            head_dim: c
                .get_value::<u32>("attention.key_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            moe,
            nextn_predict_layers: c
                .get_value::<u32>("nextn_predict_layers")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
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
        verify_arch(meta)?;
        let metadata = ContentMetadata {
            path_prefix: "glm4moe",
            metadata: meta,
        };
        let props = PropsGGUF::try_from(metadata).or_else(|err| hanzo_ml::bail!("{err}"))?;

        // Run all transformer blocks; the trailing MTP (nextn) block is not part of the LM forward.
        let n_layers = props.block_count.saturating_sub(props.nextn_predict_layers);
        let head_dim = props.head_dim;

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, props.rms_norm_eps)?;
        let output = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };

        let mut ropes = HashMap::new();
        for layer_idx in 0..n_layers {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new_partial(
                    props.rope_freq_base,
                    props.rope_dim,
                    props.max_seq_len,
                    device,
                    true,
                    DType::F32,
                )?),
            );
        }

        let mut layers = Vec::with_capacity(n_layers);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..n_layers,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();

            let wq_b = opt_bias(&mut ct, &format!("{prefix}.attn_q.bias"), device, dtype)?;
            let wk_b = opt_bias(&mut ct, &format!("{prefix}.attn_k.bias"), device, dtype)?;
            let wv_b = opt_bias(&mut ct, &format!("{prefix}.attn_v.bias"), device, dtype)?;
            let wo_b = opt_bias(&mut ct, &format!("{prefix}.attn_output.bias"), device, dtype)?;

            let attention_wq = gguf_linear_b(
                ct.tensor(&format!("{prefix}.attn_q.weight"), device)?,
                wq_b,
            )?;
            let attention_wk = gguf_linear_b(
                ct.tensor(&format!("{prefix}.attn_k.weight"), device)?,
                wk_b,
            )?;
            let attention_wv = gguf_linear_b(
                ct.tensor(&format!("{prefix}.attn_v.weight"), device)?,
                wv_b,
            )?;
            let attention_wo = gguf_linear_b(
                ct.tensor(&format!("{prefix}.attn_output.weight"), device)?,
                wo_b,
            )?;

            let (q_norm, k_norm) = if ct.has_tensor(&format!("{prefix}.attn_q_norm.weight")) {
                (
                    Some(QRmsNorm::new(
                        ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
                        props.rms_norm_eps,
                    )?),
                    Some(QRmsNorm::new(
                        ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?,
                        props.rms_norm_eps,
                    )?),
                )
            } else {
                (None, None)
            };

            let mlp = build_moe_or_mlp(&mut ct, layer_idx, device, &props.moe)?;

            let attention_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                props.rms_norm_eps,
            )?;
            let post_attention_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.post_attention_norm.weight"), device)?,
                props.rms_norm_eps,
            )?;

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
                post_attention_norm,
                q_norm,
                k_norm,
                mlp,
                n_head: props.head_count,
                n_kv_head: props.head_count_kv,
                head_dim,
                rotary: rotary.clone(),
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: props.head_count / props.head_count_kv,
                    softcap: None,
                    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                    sliding_window: None,
                    sinks: None,
                },
                dtype,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, props.embedding_length),
            layers,
            norm,
            output: gguf_linear(output)?,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(n_layers, props.max_seq_len)),
            max_seq_len: props.max_seq_len,
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
        let mut layer_in = self.tok_embeddings.forward(x)?;
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
            let x = layer_in;
            let residual = &x;
            let xn = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(
                &xn,
                &mask.get(x.device()),
                start_offsets,
                &positions,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
            )?;
            let x = (attn.to_dtype(residual.dtype())? + residual)?;

            let residual = &x;
            let xn = layer.post_attention_norm.forward(&x)?;
            let xn = layer.mlp.forward(&xn)?;
            let x = (xn.to_dtype(residual.dtype())? + residual)?;
            layer_in = x;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.forward(&x.contiguous()?)
    }
}
