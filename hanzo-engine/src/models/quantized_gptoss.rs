#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF GPT-OSS (`gpt-oss` arch) quantized model.
//!
//! Mirrors the safetensors `gpt_oss` math — clamped-SwiGLU MoE with per-expert biases, per-head
//! attention sinks, alternating sliding-window attention, and YaRN RoPE — but loads weights from a
//! `gpt-oss` GGUF (Q8_0 attention/embeddings, MXFP4 routed experts) through the quantized path.
//! Tensor/metadata names follow llama.cpp's `LLM_ARCH_OPENAI_MOE`.
//!
//! MXFP4 note: hanzo-ml decodes MXFP4 (ggml type 39) on every backend; the fused CUDA indexed-MoE
//! kernels do not cover MXFP4, so routed-expert matmuls take the generic per-expert QMatMul path
//! (correct, unfused). Decode-graph capture is intentionally not wired (eager-first).

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMaskConfig, CausalMasker, GptOssRotaryEmbedding, QRmsNorm, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache, NormalCacheType};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use hanzo_ml::quantized::{QMatMul, QTensor};
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

const DEFAULT_MAX_SEQ_LEN: u32 = 131072;
const GPTOSS_ALPHA: f32 = 1.702;
const GPTOSS_SWIGLU_LIMIT: f32 = 7.0;

fn gguf_linear(w: QTensor, bias: Option<Tensor>) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(w),
        b: bias,
    })?))
}

/// Load a bias tensor as F32 — matches the F32 activations produced by the F32-weighted RMSNorm and
/// the GgufMatMul output, so the fused bias-add never hits a dtype mismatch.
fn load_bias<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    name: &str,
    device: &Device,
    _dtype: DType,
) -> Result<Tensor> {
    ct.tensor(name, device)?
        .dequantize(device)?
        .to_dtype(DType::F32)
}

/// Clamped SwiGLU: `(clamp(up,-l,l) + 1) * g * sigmoid(alpha * g)` with `g = min(gate, l)`.
/// Portable (any dtype/device) — the fused CUDA kernel is F16/BF16-only, but the MoE runs in F32.
fn gptoss_swiglu(gate: &Tensor, up: &Tensor, alpha: f32, limit: f32) -> Result<Tensor> {
    let gate_clamped = gate.clamp(f32::MIN as f64, limit as f64)?;
    let up_clamped = up.clamp(-(limit as f64), limit as f64)?;
    let glu = (&gate_clamped * &hanzo_nn::ops::sigmoid(&(&gate_clamped * alpha as f64)?)?)?;
    ((&up_clamped + 1.0)?).mul(&glu)
}

/// Gather one row per routed (token, slot) from a per-expert bias `[n_experts, out]` -> `[t, topk, out]`.
fn gather_expert_bias(bias: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (t, topk) = ids.dims2()?;
    let out = bias.dim(1)?;
    let ids_flat = ids.reshape((t * topk,))?.to_dtype(DType::U32)?;
    bias.index_select(&ids_flat, 0)?.reshape((t, topk, out))
}

struct GptOssMoe {
    router_weight: Tensor, // [n_experts, hidden], f32
    router_bias: Tensor,   // [n_experts], f32
    gate_experts: QMatMul,
    up_experts: QMatMul,
    down_experts: QMatMul,
    gate_bias: Tensor, // [n_experts, ff]
    up_bias: Tensor,   // [n_experts, ff]
    down_bias: Tensor, // [n_experts, hidden]
    num_experts_per_tok: usize,
}

impl GptOssMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let out_dtype = xs_flat.dtype();
        let num_tokens = xs_flat.dim(0)?;

        // Router: top-k over raw (logit + bias), softmax over the selected logits (gpt-oss routing).
        let router_logits = xs_flat
            .to_dtype(DType::F32)?
            .matmul(&self.router_weight.t()?)?
            .broadcast_add(&self.router_bias.unsqueeze(0)?)?;
        let (topk_weight, topk_idx) = {
            use crate::ops::TopKLastDimOp;
            let topk = router_logits.topk(self.num_experts_per_tok)?;
            (hanzo_nn::ops::softmax_last_dim(&topk.values)?, topk.indices)
        };

        let xs3 = xs_flat.reshape((num_tokens, 1, hidden_dim))?;
        let gate = self
            .gate_experts
            .indexed_moe_forward(&xs3, &topk_idx)?
            .broadcast_add(&gather_expert_bias(&self.gate_bias, &topk_idx)?.to_dtype(out_dtype)?)?;
        let up = self
            .up_experts
            .indexed_moe_forward(&xs3, &topk_idx)?
            .broadcast_add(&gather_expert_bias(&self.up_bias, &topk_idx)?.to_dtype(out_dtype)?)?;
        let activated = gptoss_swiglu(&gate, &up, GPTOSS_ALPHA, GPTOSS_SWIGLU_LIMIT)?;
        let down = self
            .down_experts
            .indexed_moe_forward(&activated, &topk_idx)?
            .broadcast_add(&gather_expert_bias(&self.down_bias, &topk_idx)?.to_dtype(out_dtype)?)?;

        down.broadcast_mul(&topk_weight.to_dtype(down.dtype())?.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .to_dtype(out_dtype)?
            .reshape((batch, seq_len, hidden_dim))
    }
}

struct LayerWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    attention_norm: QRmsNorm,
    post_attention_norm: QRmsNorm,
    mlp: GptOssMoe,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<GptOssRotaryEmbedding>,
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
    sliding_window: usize,
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
    pub head_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub sliding_window: usize,
    pub rope_scaling_factor: f64,
    pub rope_yarn_orig_ctx: usize,
}

fn verify_arch(metadata: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>) -> Result<()> {
    use crate::utils::gguf_metadata::TryValueInto;
    let arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    if arch != "gpt-oss" {
        hanzo_ml::bail!("Expected `gpt-oss` architecture, got `{arch}`.");
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
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;

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
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(150_000_f32),
            head_dim: c
                .get_value::<u32>("attention.key_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            num_experts: c.get_value::<u32>("expert_count")? as usize,
            num_experts_per_tok: c.get_value::<u32>("expert_used_count")? as usize,
            sliding_window: c
                .get_value::<u32>("attention.sliding_window")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
            rope_scaling_factor: c
                .get_value::<f32>("rope.scaling.factor")
                .ok()
                .map(|x| x as f64)
                .unwrap_or(1.0),
            rope_yarn_orig_ctx: c
                .get_value::<u32>("rope.scaling.original_context_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(4096),
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
            path_prefix: "gpt-oss",
            metadata: meta,
        };
        let props = PropsGGUF::try_from(metadata).or_else(|err| hanzo_ml::bail!("{err}"))?;

        let head_dim = props.head_dim;
        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new_dtype(
            ct.tensor("output_norm.weight", device)?,
            props.rms_norm_eps,
            dtype,
        )?;
        let output = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };

        // YaRN RoPE (gpt-oss ships rope.scaling.type = yarn). beta_fast/slow default 32/1, truncate off.
        let mut ropes = HashMap::new();
        for layer_idx in 0..props.block_count {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                device.location(),
                Arc::new(GptOssRotaryEmbedding::new(
                    props.rope_freq_base as f64,
                    head_dim,
                    props.max_seq_len,
                    props.rope_scaling_factor,
                    props.rope_yarn_orig_ctx,
                    32.0,
                    1.0,
                    false,
                    device,
                    DType::F32,
                )?),
            );
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

            let wq_b = load_bias(&mut ct, &format!("{prefix}.attn_q.bias"), device, dtype)?;
            let wk_b = load_bias(&mut ct, &format!("{prefix}.attn_k.bias"), device, dtype)?;
            let wv_b = load_bias(&mut ct, &format!("{prefix}.attn_v.bias"), device, dtype)?;
            let wo_b = load_bias(
                &mut ct,
                &format!("{prefix}.attn_output.bias"),
                device,
                dtype,
            )?;
            let attention_wq = gguf_linear(
                ct.tensor(&format!("{prefix}.attn_q.weight"), device)?,
                Some(wq_b),
            )?;
            let attention_wk = gguf_linear(
                ct.tensor(&format!("{prefix}.attn_k.weight"), device)?,
                Some(wk_b),
            )?;
            let attention_wv = gguf_linear(
                ct.tensor(&format!("{prefix}.attn_v.weight"), device)?,
                Some(wv_b),
            )?;
            let attention_wo = gguf_linear(
                ct.tensor(&format!("{prefix}.attn_output.weight"), device)?,
                Some(wo_b),
            )?;

            let sinks = ct
                .tensor(&format!("{prefix}.attn_sinks.weight"), device)?
                .dequantize(device)?
                .to_dtype(dtype)?;

            // Routed experts (MXFP4) + per-expert biases; router is f32 weight + bias.
            let router_weight = ct
                .tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?
                .dequantize(device)?
                .to_dtype(DType::F32)?;
            let router_bias = ct
                .tensor(&format!("{prefix}.ffn_gate_inp.bias"), device)?
                .dequantize(device)?
                .to_dtype(DType::F32)?;
            let gate_bias = load_bias(
                &mut ct,
                &format!("{prefix}.ffn_gate_exps.bias"),
                device,
                dtype,
            )?;
            let up_bias = load_bias(
                &mut ct,
                &format!("{prefix}.ffn_up_exps.bias"),
                device,
                dtype,
            )?;
            let down_bias = load_bias(
                &mut ct,
                &format!("{prefix}.ffn_down_exps.bias"),
                device,
                dtype,
            )?;
            let mlp = GptOssMoe {
                router_weight,
                router_bias,
                gate_experts: QMatMul::from_qtensor(
                    ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), device)?,
                )?,
                up_experts: QMatMul::from_qtensor(
                    ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), device)?,
                )?,
                down_experts: QMatMul::from_qtensor(
                    ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), device)?,
                )?,
                gate_bias,
                up_bias,
                down_bias,
                num_experts_per_tok: props.num_experts_per_tok,
            };

            let attention_norm = QRmsNorm::new_dtype(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                props.rms_norm_eps,
                dtype,
            )?;
            let post_attention_norm = QRmsNorm::new_dtype(
                ct.tensor(&format!("{prefix}.post_attention_norm.weight"), device)?,
                props.rms_norm_eps,
                dtype,
            )?;

            // gpt-oss alternates sliding/full attention; layer 0 is sliding.
            let is_sliding = props.sliding_window > 0 && layer_idx % 2 == 0;
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
                    sliding_window: if is_sliding {
                        Some(props.sliding_window)
                    } else {
                        None
                    },
                    sinks: Some(sinks),
                },
                is_sliding,
                dtype,
            });
        }

        let cache_types: Vec<NormalCacheType> = (0..props.block_count)
            .map(|layer_idx| {
                if props.sliding_window > 0 && layer_idx % 2 == 0 {
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
            tok_embeddings: Embedding::new(tok_embeddings, props.embedding_length),
            layers,
            norm,
            output: gguf_linear(output, None)?,
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
        // Single-dtype residual: cast the F32 embedding output to the compute dtype so the
        // norm/attention/MoE chain stays one dtype (drops F32<->half casts; the MoE expert path
        // restores the input dtype so no F32 round-trip is reintroduced).
        let mut layer_in = self.tok_embeddings.forward(x)?.to_dtype(self.dtype)?;
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

        let causal_mask = CausalMasker.make_causal_mask(
            x,
            past_len,
            self.dtype,
            &CausalMaskConfig {
                force_custom: true,
                ..Default::default()
            },
        )?;
        let sliding_mask = CausalMasker.make_causal_mask(
            x,
            past_len,
            self.dtype,
            &CausalMaskConfig {
                sliding_window: Some(self.sliding_window),
                force_custom: true,
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
            let x = layer_in;
            let residual = &x;
            let xn = layer.attention_norm.forward(&x)?;
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
