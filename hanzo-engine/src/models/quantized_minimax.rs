#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF MiniMax-M2 (`minimax-m2` arch) quantized sparse-MoE model.
//!
//! Mirrors the safetensors `minimax_m2` math: every layer is MoE (sigmoid router with an additive
//! `exp_probs_b` correction bias used for selection only, no shared experts), GQA attention with a
//! per-layer RMSNorm over the *flattened* q/k projection (before the head reshape, `qk_norm_type =
//! per_layer`, distinct from qwen3's per-head qk-norm), and partial RoPE (only the first
//! `rope.dimension_count` channels of each head are rotated). The MoE gate/expert math is the shared
//! `gguf_moe` block (same as deepseek/glm4moe); tensor/metadata names follow llama.cpp's
//! `LLM_ARCH_MINIMAX_M2`.
//!
//! Single-dtype: norm weights held in `dtype` (`QRmsNorm::new_dtype`) and the embedding is cast to
//! `dtype`, so the residual stream stays in the compute dtype; RoPE runs in F32 for precision.

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMaskConfig, CausalMasker, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::models::gguf_moe::{build_moe_or_mlp, MoeOrMlp, MoeParams};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

const DEFAULT_MAX_SEQ_LEN: u32 = 196608;
const EXPERT_GATING_SIGMOID: u32 = 2;

fn gguf_linear(w: hanzo_ml::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(w),
        b: None,
    })?))
}

struct LayerWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    attention_norm: QRmsNorm,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    ffn_norm: QRmsNorm,
    mlp: MoeOrMlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rot_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl LayerWeights {
    /// Partial NeoX RoPE: rotate the leading `rot_dim` channels of each head, pass the rest through.
    /// Runs in F32 (the cos/sin table dtype) then casts back to the input dtype.
    fn apply_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        start_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        if self.rot_dim == self.head_dim {
            let (q, k) = (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?);
            return self.rotary.forward(&q, &k, start_offsets);
        }
        let pass = self.head_dim - self.rot_dim;
        let q_rot = q.narrow(D::Minus1, 0, self.rot_dim)?.to_dtype(DType::F32)?;
        let k_rot = k.narrow(D::Minus1, 0, self.rot_dim)?.to_dtype(DType::F32)?;
        let q_pass = q.narrow(D::Minus1, self.rot_dim, pass)?;
        let k_pass = k.narrow(D::Minus1, self.rot_dim, pass)?;
        let (q_rot, k_rot) = self.rotary.forward(&q_rot, &k_rot, start_offsets)?;
        let q = Tensor::cat(
            &[q_rot.to_dtype(q_pass.dtype())?, q_pass.clone()],
            D::Minus1,
        )?
        .contiguous()?;
        let k = Tensor::cat(
            &[k_rot.to_dtype(k_pass.dtype())?, k_pass.clone()],
            D::Minus1,
        )?
        .contiguous()?;
        Ok((q, k))
    }

    fn forward_attn(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        // Per-layer qk-norm over the full flat projection, before the head reshape.
        let q = self.q_norm.forward(&self.attention_wq.forward(x)?)?;
        let k = self.k_norm.forward(&self.attention_wk.forward(x)?)?;
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

        let (q, k) = self.apply_rope(&q, &k, start_offsets)?;
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

    fn forward_block(
        &self,
        x: Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let residual = &x;
        let xn = self.attention_norm.forward(&x)?;
        let attn = self.forward_attn(&xn, mask, start_offsets, kv_cache, metadata)?;
        let (sum, xn) = self.ffn_norm.forward_of_sum(&attn, residual)?;
        let residual = &sum;
        let xn = self.mlp.forward(&xn)?;
        xn + residual
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
    pub head_dim: usize,
    pub value_length: usize,
    pub rot_dim: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub sigmoid_scoring: bool,
}

fn verify_arch(metadata: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>) -> Result<()> {
    use crate::utils::gguf_metadata::TryValueInto;
    let arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    if arch != "minimax-m2" {
        hanzo_ml::bail!("Expected `minimax-m2` architecture, got `{arch}`.");
    }
    Ok(())
}

impl PropsGGUF {
    fn new(c: &ContentMetadata) -> std::result::Result<Self, anyhow::Error> {
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
            rot_dim: c
                .get_value::<u32>("rope.dimension_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(head_dim),
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
            expert_count: c.get_value::<u32>("expert_count")? as usize,
            expert_used_count: c.get_value::<u32>("expert_used_count")? as usize,
            sigmoid_scoring: c
                .get_value::<u32>("expert_gating_func")
                .ok()
                .map(|f| f == EXPERT_GATING_SIGMOID)
                .unwrap_or(true),
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
        verify_arch(ct.get_metadata())?;
        let metadata = ContentMetadata {
            path_prefix: "minimax-m2",
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
        let output = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };

        let mut ropes = HashMap::new();
        for layer_idx in 0..props.block_count {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(device.location()) {
                e.insert(Arc::new(RotaryEmbedding::new_partial(
                    props.rope_freq_base,
                    props.rot_dim,
                    props.max_seq_len,
                    device,
                    true,
                    dtype,
                )?));
            }
        }

        let moe_params = MoeParams {
            n_routed_experts: props.expert_count,
            num_experts_per_tok: props.expert_used_count,
            n_group: 1,
            topk_group: 1,
            routed_scaling_factor: 1.0,
            norm_topk_prob: true,
            sigmoid_scoring: props.sigmoid_scoring,
            n_shared_experts: 0,
            leading_dense_block_count: 0,
        };

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
            let q_norm = QRmsNorm::new_dtype(
                ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
                props.rms_norm_eps,
                dtype,
            )?;
            let k_norm = QRmsNorm::new_dtype(
                ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?,
                props.rms_norm_eps,
                dtype,
            )?;
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
            let mlp = build_moe_or_mlp(&mut ct, layer_idx, device, &moe_params)?;

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
                q_norm,
                k_norm,
                ffn_norm,
                mlp,
                n_head: props.head_count,
                n_kv_head: props.head_count_kv,
                head_dim,
                rot_dim: props.rot_dim,
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
            tok_embeddings,
            layers,
            norm,
            output: gguf_linear(output)?,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(props.block_count, props.max_seq_len)),
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
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }
            let dmask = mask.get(layer_in.device());
            layer_in = layer.forward_block(
                layer_in,
                &dmask,
                start_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
            )?;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.forward(&x.contiguous()?)
    }
}
