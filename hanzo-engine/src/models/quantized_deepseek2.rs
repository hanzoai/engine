#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GGUF DeepSeek-V2/V3 (MLA) quantized model.
//!
//! Mirrors the safetensors `deepseek2`/`deepseek3` attention (Multi-head Latent Attention) but loads
//! weights from a GGUF `deepseek2` checkpoint and runs them through the quantized `GgufMatMul` path.
//! The same arch covers V3/V4-class checkpoints (the V3 no-aux-loss gate is scaffolded behind the
//! `expert_weights_norm`/`expert_gating_func` metadata; see TODOs). Layout/tensor names follow
//! llama.cpp's `LLM_ARCH_DEEPSEEK2` convention.
//!
//! NOTE: this is a scope+scaffold pass. MLA here uses the un-absorbed (materialized K/V) eager path
//! that the safetensors model uses on the no-paged branch; the CUDA paged "absorbed" MLA decode in
//! `crate::mla` is NOT wired here yet (it needs paged-attn MLA cache layout for GGUF). See the report.

use std::collections::HashMap;
use std::sync::Arc;

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{
    CausalMaskConfig, CausalMasker, DeepSeekV2RopeConfig, DeepSeekV2RopeScaling,
    DeepSeekV2RotaryEmbedding, QRmsNorm, ScaledRopeType, Sdpa,
};
use crate::layers_masker::PastKvLenCache;
use crate::ops::{SplitOp, TopKLastDimOp, TopKOutput};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use hanzo_ml::quantized::QMatMul;
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

// llama.cpp expert gating funcs (LLM_EXPERT_GATING_FUNC_TYPE). V2 = softmax, V3 = sigmoid.
const EXPERT_GATING_SOFTMAX: u32 = 1;
const EXPERT_GATING_SIGMOID: u32 = 2;

struct Mlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(xs)?;
        let up = self.up.forward(xs)?;
        let y = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        self.down.forward(&y)
    }
}

struct MoeGate {
    weight: Tensor,
    e_score_correction_bias: Option<Tensor>,
    top_k: usize,
    n_routed_experts: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f64,
    norm_topk_prob: bool,
    sigmoid_scoring: bool,
}

impl MoeGate {
    // (topk_idx, topk_weight). Greedy softmax (V2) or grouped sigmoid no-aux (V3) selection.
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (bs, seq_len, h) = xs.dims3()?;
        let xs = xs.reshape(((), h))?;
        let logits = xs
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.weight.t()?.to_dtype(DType::F32)?)?;
        let scores = if self.sigmoid_scoring {
            hanzo_nn::ops::sigmoid(&logits)?
        } else {
            hanzo_nn::ops::softmax_last_dim(&logits)?
        };

        let mut topk_weight;
        let topk_idx;
        if let Some(bias) = &self.e_score_correction_bias {
            // V3 noaux_tc: group-limited greedy on (scores + bias), gathered weights are raw scores.
            let scores_for_choice = scores
                .reshape((bs * seq_len, ()))?
                .broadcast_add(&bias.unsqueeze(0)?)?;
            let group_scores = scores_for_choice
                .reshape((bs * seq_len, self.n_group, ()))?
                .topk(2)?
                .values
                .sum(D::Minus1)?;
            let group_idx = group_scores.topk(self.topk_group)?.indices;
            let mut group_mask = group_scores.zeros_like()?;
            group_mask = group_mask.scatter_add(
                &group_idx,
                &group_idx.ones_like()?.to_dtype(group_mask.dtype())?,
                1,
            )?;
            let score_mask = group_mask
                .unsqueeze(D::Minus1)?
                .expand((
                    bs * seq_len,
                    self.n_group,
                    self.n_routed_experts / self.n_group,
                ))?
                .reshape((bs * seq_len, ()))?;
            let tmp_scores = scores_for_choice.broadcast_mul(&score_mask)?;
            topk_idx = tmp_scores.topk(self.top_k)?.indices;
            topk_weight = scores.gather(&topk_idx, 1)?;
        } else {
            let TopKOutput { values, indices } = scores.topk(self.top_k)?;
            topk_weight = values;
            topk_idx = indices;
        }

        if self.norm_topk_prob {
            let denom = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weight = topk_weight.broadcast_div(&denom)?;
        }
        topk_weight = (topk_weight * self.routed_scaling_factor)?;
        Ok((topk_idx, topk_weight))
    }
}

// GGUF fused MoE: ffn_gate_exps / ffn_up_exps / ffn_down_exps stacked across experts.
struct FusedMoe {
    gate: MoeGate,
    gate_experts: QMatMul,
    up_experts: QMatMul,
    down_experts: QMatMul,
    shared: Option<Mlp>,
}

impl FusedMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let identity = xs.clone();
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let original_dtype = xs_flat.dtype();
        let (num_tokens, hidden_dim) = xs_flat.dims2()?;

        let (topk_idx, topk_weight) = self.gate.forward(xs)?;

        let ys = {
            let xs3 = xs_flat.reshape((num_tokens, 1, hidden_dim))?;
            let gate = self.gate_experts.indexed_moe_forward(&xs3, &topk_idx)?;
            let up = self.up_experts.indexed_moe_forward(&xs3, &topk_idx)?;
            let activated = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
            self.down_experts.indexed_moe_forward(&activated, &topk_idx)?
        };
        let mut y = ys
            .broadcast_mul(&topk_weight.to_dtype(ys.dtype())?.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((batch, seq_len, hidden_dim))?
            .to_dtype(original_dtype)?;

        if let Some(shared) = &self.shared {
            y = (y + shared.forward(&identity)?)?;
        }
        Ok(y)
    }
}

enum MoeOrMlp {
    FusedMoe(Box<FusedMoe>),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs),
        }
    }
}

struct LayerWeights {
    // MLA projections. q is either plain (q_proj) or low-rank (q_a/q_b with q_a_norm).
    q_a_proj: Option<Arc<dyn QuantMethod>>,
    q_a_norm: Option<QRmsNorm>,
    q_b_proj: Option<Arc<dyn QuantMethod>>,
    q_proj: Option<Arc<dyn QuantMethod>>,
    kv_a_proj_with_mqa: Arc<dyn QuantMethod>,
    kv_a_norm: QRmsNorm,
    kv_b_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    attn_norm: QRmsNorm,
    ffn_norm: QRmsNorm,
    mlp: MoeOrMlp,
    rotary: Arc<DeepSeekV2RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    n_head: usize,
    q_head_dim: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    kv_lora_rank: usize,
    v_head_dim: usize,
    dtype: DType,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?;

        let q = match (&self.q_proj, &self.q_a_proj, &self.q_b_proj, &self.q_a_norm) {
            (Some(q_proj), _, _, _) => q_proj.forward(x)?,
            (None, Some(a), Some(b), Some(norm)) => b.forward(&norm.forward(&a.forward(x)?)?)?,
            _ => hanzo_ml::bail!("deepseek2 gguf: inconsistent q projection weights"),
        };
        let q = q
            .reshape((bs, seq_len, self.n_head, self.q_head_dim))?
            .transpose(1, 2)?;
        let q_split = q.split(&[self.qk_nope_head_dim, self.qk_rope_head_dim], D::Minus1)?;
        let q_nope = q_split[0].clone();
        let mut q_pe = q_split[1].clone();

        let compressed_kv = self.kv_a_proj_with_mqa.forward(x)?;
        let ckv_split =
            compressed_kv.split(&[self.kv_lora_rank, self.qk_rope_head_dim], D::Minus1)?;
        let compressed_kv = ckv_split[0].clone();
        let mut k_pe = ckv_split[1].clone();
        k_pe = k_pe
            .reshape((bs, seq_len, 1, self.qk_rope_head_dim))?
            .transpose(1, 2)?;
        let ckv = self.kv_a_norm.forward(&compressed_kv)?;

        (q_pe, k_pe) = self.rotary.forward(&q_pe, &k_pe, start_offsets)?;

        let mut kv = self.kv_b_proj.forward(&ckv)?;
        kv = kv
            .reshape((
                bs,
                seq_len,
                self.n_head,
                self.qk_nope_head_dim + self.v_head_dim,
            ))?
            .transpose(1, 2)?;
        let kv_split = kv.split(&[self.qk_nope_head_dim, self.v_head_dim], D::Minus1)?;
        let k_nope = kv_split[0].clone();
        let v = kv_split[1].clone();

        let q = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?.contiguous()?;
        let k = Tensor::cat(&[&k_nope, &k_pe.repeat((1, self.n_head, 1, 1))?], D::Minus1)?
            .contiguous()?;

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        let y = match &self.paged_attn {
            Some(paged_attn) => {
                let ((key_cache, value_cache), input_metadata) = metadata.unwrap();
                // Pad V to q_head_dim so the paged cache K/V share head_dim, then slice back.
                let v = v
                    .pad_with_zeros(D::Minus1, 0, self.q_head_dim - self.v_head_dim)?
                    .contiguous()?;
                paged_attn
                    .forward(
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
                    .narrow(D::Minus1, 0, self.v_head_dim)?
            }
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa_params)?
            }
        };

        let y = if mask.is_custom() {
            y.transpose(1, 2)?.reshape((bs, seq_len, ()))?
        } else {
            y.reshape((bs, seq_len, ()))?
        };
        self.o_proj.forward(&y.to_dtype(x.dtype())?)
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

// Some fields mirror the full deepseek2 GGUF metadata surface but aren't needed at load time (GGUF
// tensors carry their own shapes); kept for clarity and future paged-MLA wiring.
#[allow(dead_code)]
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    // MLA dims.
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    // MoE.
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub moe_intermediate_size: usize,
    pub dense_intermediate_size: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub leading_dense_block_count: usize,
    pub expert_weights_scale: f64,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub sigmoid_scoring: bool,
    // YaRN rope scaling (optional).
    pub rope_scaling_factor: Option<f32>,
    pub rope_yarn_orig_ctx: Option<usize>,
}

fn verify_arch(metadata: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>) -> Result<()> {
    use crate::utils::gguf_metadata::TryValueInto;
    let arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    if arch != "deepseek2" {
        hanzo_ml::bail!("Expected `deepseek2` architecture, got `{arch}`.");
    }
    Ok(())
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let required = [
            "attention.head_count",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
            "attention.kv_lora_rank",
            "attention.key_length", // qk_nope_head_dim (llama.cpp sets key_length = nope dim)
            "rope.dimension_count",  // qk_rope_head_dim
            "expert_count",
            "expert_used_count",
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;
        let n_routed_experts = c.get_value::<u32>("expert_count")? as usize;

        let sigmoid_scoring = match c.get_value::<u32>("expert_gating_func") {
            Ok(EXPERT_GATING_SIGMOID) => true,
            Ok(EXPERT_GATING_SOFTMAX) | Err(_) => false,
            Ok(other) => anyhow::bail!("unknown deepseek2 expert_gating_func {other}"),
        };

        Ok(Self {
            head_count,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
            q_lora_rank: c
                .get_value::<u32>("attention.q_lora_rank")
                .ok()
                .map(|x| x as usize),
            kv_lora_rank: c.get_value::<u32>("attention.kv_lora_rank")? as usize,
            qk_nope_head_dim: c.get_value::<u32>("attention.key_length")? as usize,
            qk_rope_head_dim: c.get_value::<u32>("rope.dimension_count")? as usize,
            v_head_dim: c
                .get_value::<u32>("attention.value_length")
                .ok()
                .map(|x| x as usize)
                // V-head dim falls back to nope dim when not present.
                .unwrap_or(c.get_value::<u32>("attention.key_length")? as usize),
            n_routed_experts,
            n_shared_experts: c
                .get_value::<u32>("expert_shared_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
            moe_intermediate_size: c.get_value::<u32>("expert_feed_forward_length")? as usize,
            dense_intermediate_size: c.get_value::<u32>("feed_forward_length")? as usize,
            num_experts_per_tok: c.get_value::<u32>("expert_used_count")? as usize,
            first_k_dense_replace: c
                .get_value::<u32>("leading_dense_block_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
            leading_dense_block_count: c
                .get_value::<u32>("leading_dense_block_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(0),
            expert_weights_scale: c
                .get_value::<f32>("expert_weights_scale")
                .ok()
                .map(|x| x as f64)
                .unwrap_or(1.0),
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
            norm_topk_prob: c.get_value::<bool>("expert_weights_norm").ok().unwrap_or(true),
            sigmoid_scoring,
            rope_scaling_factor: c.get_value::<f32>("rope.scaling.factor").ok(),
            rope_yarn_orig_ctx: c
                .get_value::<u32>("rope.scaling.original_context_length")
                .ok()
                .map(|x| x as usize),
        })
    }
}

fn gguf_linear(q: hanzo_ml::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(q),
        b: None,
    })?))
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
            path_prefix: "deepseek2",
            metadata: meta,
        };
        let props = PropsGGUF::try_from(metadata).or_else(|err| hanzo_ml::bail!("{err}"))?;

        let q_head_dim = props.qk_nope_head_dim + props.qk_rope_head_dim;
        // softmax_scale = 1/sqrt(q_head_dim), optionally YaRN-mscaled (see deepseek3.rs softmax_scale).
        let mut softmax_scale = 1.0 / (q_head_dim as f32).sqrt();
        if let (Some(factor), Some(_orig)) = (props.rope_scaling_factor, props.rope_yarn_orig_ctx) {
            // mscale_all_dim defaults; replicate DeepSeekV2RotaryEmbedding::yarn_get_mscale(factor, 1.0).
            let mscale = DeepSeekV2RotaryEmbedding::yarn_get_mscale(factor, 1.0);
            softmax_scale = softmax_scale * mscale * mscale;
        }

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, props.rms_norm_eps)?;
        let output = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };

        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: props.rope_scaling_factor.and_then(|factor| {
                props.rope_yarn_orig_ctx.map(|orig| DeepSeekV2RopeScaling::Yarn {
                    original_max_position_embeddings: orig,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                    factor,
                    mscale: 1.0,
                    mscale_all_dim: 1.0,
                    scaling_type: ScaledRopeType::Yarn,
                })
            }),
            max_position_embeddings: props.max_seq_len,
            rope_theta: props.rope_freq_base,
            qk_rope_head_dim: props.qk_rope_head_dim,
        };

        let mut ropes = HashMap::new();
        for layer_idx in 0..props.block_count {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                device.location(),
                Arc::new(DeepSeekV2RotaryEmbedding::new(&rope_cfg, DType::F32, device)?),
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

            // Q: low-rank (attn_q_a/attn_q_b) if q_lora_rank set, else plain attn_q.
            let (q_a_proj, q_a_norm, q_b_proj, q_proj) = if props.q_lora_rank.is_some() {
                (
                    Some(gguf_linear(ct.tensor(&format!("{prefix}.attn_q_a.weight"), device)?)?),
                    Some(QRmsNorm::new(
                        ct.tensor(&format!("{prefix}.attn_q_a_norm.weight"), device)?,
                        props.rms_norm_eps,
                    )?),
                    Some(gguf_linear(ct.tensor(&format!("{prefix}.attn_q_b.weight"), device)?)?),
                    None,
                )
            } else {
                (
                    None,
                    None,
                    None,
                    Some(gguf_linear(ct.tensor(&format!("{prefix}.attn_q.weight"), device)?)?),
                )
            };

            let kv_a_proj_with_mqa =
                gguf_linear(ct.tensor(&format!("{prefix}.attn_kv_a_mqa.weight"), device)?)?;
            let kv_a_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_kv_a_norm.weight"), device)?,
                props.rms_norm_eps,
            )?;
            let kv_b_proj =
                gguf_linear(ct.tensor(&format!("{prefix}.attn_kv_b.weight"), device)?)?;
            let o_proj =
                gguf_linear(ct.tensor(&format!("{prefix}.attn_output.weight"), device)?)?;

            let attn_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                props.rms_norm_eps,
            )?;
            let ffn_norm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?,
                props.rms_norm_eps,
            )?;

            // Layer is MoE when idx >= leading_dense_block_count and experts exist.
            let is_moe = props.n_routed_experts > 0
                && layer_idx >= props.leading_dense_block_count;
            let mlp = if is_moe {
                let gate = ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let gate_experts =
                    ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), device)?;
                let up_experts = ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), device)?;
                let down_experts =
                    ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), device)?;
                let e_score_correction_bias = if props.sigmoid_scoring
                    && ct.has_tensor(&format!("{prefix}.exp_probs_b.bias"))
                {
                    Some(
                        ct.tensor(&format!("{prefix}.exp_probs_b.bias"), device)?
                            .dequantize(device)?
                            .to_dtype(DType::F32)?,
                    )
                } else {
                    None
                };
                let shared = if props.n_shared_experts > 0 {
                    Some(Mlp {
                        gate: gguf_linear(
                            ct.tensor(&format!("{prefix}.ffn_gate_shexp.weight"), device)?,
                        )?,
                        up: gguf_linear(
                            ct.tensor(&format!("{prefix}.ffn_up_shexp.weight"), device)?,
                        )?,
                        down: gguf_linear(
                            ct.tensor(&format!("{prefix}.ffn_down_shexp.weight"), device)?,
                        )?,
                    })
                } else {
                    None
                };
                MoeOrMlp::FusedMoe(Box::new(FusedMoe {
                    gate: MoeGate {
                        weight: gate.dequantize(device)?,
                        e_score_correction_bias,
                        top_k: props.num_experts_per_tok,
                        n_routed_experts: props.n_routed_experts,
                        n_group: props.n_group,
                        topk_group: props.topk_group,
                        routed_scaling_factor: props.expert_weights_scale,
                        norm_topk_prob: props.norm_topk_prob,
                        sigmoid_scoring: props.sigmoid_scoring,
                    },
                    gate_experts: QMatMul::from_qtensor(gate_experts)?,
                    up_experts: QMatMul::from_qtensor(up_experts)?,
                    down_experts: QMatMul::from_qtensor(down_experts)?,
                    shared,
                }))
            } else {
                MoeOrMlp::Mlp(Mlp {
                    gate: gguf_linear(ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?)?,
                    up: gguf_linear(ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?)?,
                    down: gguf_linear(ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?)?,
                })
            };

            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(q_head_dim, device, None)?)
                }
            };

            layers.push(LayerWeights {
                q_a_proj,
                q_a_norm,
                q_b_proj,
                q_proj,
                kv_a_proj_with_mqa,
                kv_a_norm,
                kv_b_proj,
                o_proj,
                attn_norm,
                ffn_norm,
                mlp,
                rotary,
                paged_attn,
                sdpa_params: SdpaParams {
                    n_kv_groups: 1,
                    softcap: None,
                    softmax_scale,
                    sliding_window: None,
                    sinks: None,
                },
                n_head: props.head_count,
                q_head_dim,
                qk_nope_head_dim: props.qk_nope_head_dim,
                qk_rope_head_dim: props.qk_rope_head_dim,
                kv_lora_rank: props.kv_lora_rank,
                v_head_dim: props.v_head_dim,
                dtype,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, props.embedding_length),
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
        let mut layer_in = self.tok_embeddings.forward(x)?;
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
            let x = layer_in;
            let residual = &x;
            let xn = layer.attn_norm.forward(&x)?;
            let attn = layer.forward_attn(
                &xn,
                &mask.get(x.device()),
                start_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
            )?;
            let x = (attn + residual)?;

            let residual = &x;
            let xn = layer.ffn_norm.forward(&x)?;
            let xn = layer.mlp.forward(&xn)?;
            let x = (xn + residual)?;
            layer_in = x;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.forward(&x.contiguous()?)
    }
}
