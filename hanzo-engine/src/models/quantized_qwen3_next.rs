#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Quantized (GGUF) loader for Qwen3-Next (80B-A3B) hybrid linear-attention MoE.
//!
//! Architecture string: `qwen3next`. 48 layers on a 3:1 schedule — every
//! `full_attention_interval`-th layer (default 4) is a gated full-attention layer (partial-rotary
//! NEOX RoPE + qk-RMSNorm + sigmoid output gate); the other three are Gated-DeltaNet (GDN)
//! linear-attention layers (causal conv1d kernel=4, gated RMSNorm, recurrent state). The FFN is a
//! sparse MoE (512 experts, 10 routed + 1 shared with a sigmoid gate).
//!
//! The GDN math is shared with `models::gdn` / `models::qwen3_next` (safetensors) and
//! `models::quantized_qwen3_5_moe` (the sibling GGUF hybrid). The MoE block (`FusedMoe`) and the
//! `gguf_qmm` helper are shared verbatim with `quantized_qwen3_5_moe` — the sparse-MoE layout is
//! identical between Qwen3.5-MoE and Qwen3-Next. The recurrent state flows through the pipeline
//! `HybridCache` exactly like both siblings.
//!
//! ===================== Qwen3-Next vs Qwen3.5 GGUF differences =====================
//! (source: llama.cpp `src/models/qwen3next.cpp`, `src/llama-arch.cpp`, `src/llama-model.cpp`)
//!
//!  1. RoPE: Qwen3-Next uses plain NEOX **partial** rotary (`LLAMA_ROPE_TYPE_NEOX`, rot_dim =
//!     head_dim * partial_rotary_factor), NOT the interleaved mRoPE that Qwen3.5 uses. So this
//!     loader uses `RotaryEmbedding::new_partial` + `forward_qk_norm{,_positions}` (the same path
//!     the safetensors `qwen3_next` model uses), never `Qwen3VLRotaryEmbedding`.
//!  2. beta/alpha projection: MERGED into a single `ssm_ba` tensor ([2*num_v_heads, hidden]) split
//!     per key-head group into (beta | alpha), vs Qwen3.5's separate `ssm_beta` + `ssm_alpha`.
//!  3. V-head order: GROUPED `[k0_v0, k0_v1, k1_v2, k1_v3, ...]` (HF-native), so V head j pairs with
//!     K head `j / v_per_group`. The q/k repeat is therefore a GROUPED repeat
//!     (`unsqueeze(3).repeat(..v_per_group..)`), not Qwen3.5's TILED repeat (`% num_k_heads`).
//!  4. Experts: 512 routed / 10 per token (vs 256 / 8). Read straight from GGUF metadata.
//!
//! ===================== GGUF tensor-name mapping =====================
//! Per layer `blk.{i}`:
//!   GDN (linear-attention) layers — two possible input-projection encodings:
//!     optimized: attn_qkv.weight (merged q|k|v, key_dim*2+value_dim) + attn_gate.weight (z, value_dim)
//!     legacy:    ssm_in.weight (merged q|k|v|z, key_dim*2+value_dim*2, grouped by key-head)
//!     ssm_ba.weight    <- merged beta|alpha (2*num_v_heads, hidden), grouped by key-head
//!     ssm_conv1d.weight<- causal conv1d (conv_dim, kernel) [squeezed to 2D]
//!     ssm_dt.bias      <- dt_bias (num_v_heads, f32)
//!     ssm_a            <- -exp(A_log) precomputed at conversion (num_v_heads, f32) [no `.weight`]
//!     ssm_norm.weight  <- gated RMSNorm (head_v_dim)
//!     ssm_out.weight   <- out_proj (value_dim, hidden)
//!   Full-attention layers:
//!     attn_q.weight (DOUBLED: num_heads*head_dim*2 = query + gate), attn_k/attn_v/attn_output,
//!     attn_q_norm, attn_k_norm (standard GQA + qk-norm).
//!   Shared: attn_norm (input_layernorm), post_attention_norm (post-attn layernorm).
//!   MoE FFN: ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps,
//!            ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp.
//!   Global: token_embd, output_norm, output (tied to token_embd when absent).
//!
//! Metadata keys (prefixed with `qwen3next.` by ContentMetadata):
//!   attention.head_count / head_count_kv / key_length / layer_norm_rms_epsilon,
//!   block_count, context_length, rope.freq_base, rope.dimension_count (= rot_dim),
//!   full_attention_interval, ssm.conv_kernel (=conv width), ssm.state_size (=head_k_dim=head_v_dim),
//!   ssm.group_count (=num_k_heads), ssm.time_step_rank (=num_v_heads), ssm.inner_size (=value_dim),
//!   expert_count, expert_used_count, expert_feed_forward_length.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMaskConfig, CausalMasker, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::models::gdn::{
    gated_delta_rule_recurrence, l2_norm, softplus, GdnLayerCache, RmsNormGated,
};
use crate::models::quantized_qwen3_5_moe::{gguf_qmm, FusedMoe};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache};
use crate::pipeline_parallel::{
    pp_head_forward, use_pipeline_parallel, PipelineParallelModel, RingLayout,
};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{QuantMethod, RingPipeline};

use crate::kv_cache::{
    HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
};

const DEFAULT_MAX_SEQ_LEN: u32 = 4096;
const DEFAULT_FULL_ATTENTION_INTERVAL: usize = 4;
const DEFAULT_PARTIAL_ROTARY_FACTOR: f64 = 0.25;
const L2_NORM_EPS: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerType {
    FullAttention,
    LinearAttention,
}

// ===================== Gated full-attention layer =====================

struct QGatedFullAttention {
    // attn_q output is doubled: first head_dim per head is q, second head_dim is the output gate.
    attn_q: Arc<dyn QuantMethod>,
    attn_k: Arc<dyn QuantMethod>,
    attn_v: Arc<dyn QuantMethod>,
    attn_o: Arc<dyn QuantMethod>,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl QGatedFullAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        positions: &Tensor,
        kv_cache: &mut KvCache,
        paged: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q_gate = self.attn_q.forward(x)?;
        let k = self.attn_k.forward(x)?;
        let v = self.attn_v.forward(x)?;

        // Split q_gate into q and gate: per head, first head_dim is q, second head_dim is the gate.
        let q_gate = q_gate.reshape((b_sz, seq_len, self.n_head, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        let gate = gate.reshape((b_sz, seq_len, self.n_head * self.head_dim))?;

        let (q, k, v) = if seq_len != 1 {
            let q = q.transpose(1, 2)?;
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

        // qk-RMSNorm + partial NEOX RoPE. Decode (seq_len == 1) reads positions from the device
        // buffer (graph-safe, position-invariant); prefill uses the host seqlen_offsets. Mirrors
        // quantized_qwen3.rs, which also drives the safetensors qwen3_next partial-rope path.
        let (q, k) = if seq_len == 1 {
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

        let y = match (&self.paged_attn, paged) {
            (Some(paged_attn), Some(((key_cache, value_cache), input_metadata))) => paged_attn
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
                )?,
            (Some(paged_attn), None) => {
                let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    mask,
                    None,
                    None,
                    &input_metadata,
                    &self.sdpa_params,
                    None,
                )?
            }
            (None, _) => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa_params)?
            }
        };

        let y = if mask.is_custom() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        // Output gate: y = y * sigmoid(gate).
        let gate = hanzo_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        let y = y.broadcast_mul(&gate)?;

        self.attn_o.forward(&y.to_dtype(x.dtype())?)
    }
}

// ===================== Gated DeltaNet (linear-attention) layer =====================

/// Input QKVZ encoding. Modern GGUFs split the projection into `attn_qkv` (merged q|k|v) + `attn_gate`
/// (z); legacy GGUFs keep a single merged `ssm_in` (q|k|v|z) that must be split per key-head group.
enum QkvzProj {
    Split {
        qkv: Arc<dyn QuantMethod>,
        z: Arc<dyn QuantMethod>,
    },
    Merged(Arc<dyn QuantMethod>),
}

struct QGatedDeltaNet {
    qkvz: QkvzProj,
    in_proj_ba: Arc<dyn QuantMethod>, // merged beta|alpha -> ssm_ba
    conv1d_weight: Tensor,            // (conv_dim, kernel) f32
    dt_bias: Tensor,                  // (num_v_heads,) f32
    a: Tensor,                        // -exp(A_log), (num_v_heads,) f32, precomputed in GGUF
    norm: RmsNormGated,
    out_proj: Arc<dyn QuantMethod>,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_kernel_size: usize,
    key_dim: usize,
    value_dim: usize,
}

impl QGatedDeltaNet {
    fn forward(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        // Run the GDN recurrence + gates in f32 end-to-end (matches quantized_qwen3_5_moe); lift the
        // input here and cast the out_proj result back to the model dtype.
        let orig_dtype = x.dtype();
        let x = &x.to_dtype(DType::F32)?;
        let (batch_size, seq_len, _hidden) = x.dims3()?;
        let v_per_group = self.num_v_heads / self.num_k_heads;

        // 1. Input projections. Produce the concatenated [q|k|v] conv input (v in GROUPED order) and
        //    the z gate reshaped to (b, s, num_v_heads, head_v_dim).
        let (mixed_qkv, z) = match &self.qkvz {
            QkvzProj::Split { qkv, z } => {
                let mixed_qkv = qkv.forward(x)?;
                let z = z.forward(x)?.reshape((
                    batch_size,
                    seq_len,
                    self.num_v_heads,
                    self.head_v_dim,
                ))?;
                (mixed_qkv, z)
            }
            QkvzProj::Merged(ssm_in) => {
                // Legacy grouped layout: [q(head_k) | k(head_k) | v(v_per_group*head_v) | z(same)]
                // per key-head group. Split within the group, then flatten q,k,v and concat.
                let group_size = 2 * self.head_k_dim + 2 * v_per_group * self.head_v_dim;
                let mixed = ssm_in.forward(x)?.reshape((
                    batch_size,
                    seq_len,
                    self.num_k_heads,
                    group_size,
                ))?;
                let mut offset = 0;
                let q = mixed.narrow(D::Minus1, offset, self.head_k_dim)?;
                offset += self.head_k_dim;
                let k = mixed.narrow(D::Minus1, offset, self.head_k_dim)?;
                offset += self.head_k_dim;
                let v = mixed.narrow(D::Minus1, offset, v_per_group * self.head_v_dim)?;
                offset += v_per_group * self.head_v_dim;
                let z = mixed.narrow(D::Minus1, offset, v_per_group * self.head_v_dim)?;

                let q = q.reshape((batch_size, seq_len, self.key_dim))?;
                let k = k.reshape((batch_size, seq_len, self.key_dim))?;
                let v = v.reshape((batch_size, seq_len, self.value_dim))?;
                let z = z.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
                let mixed_qkv = Tensor::cat(&[&q, &k, &v], D::Minus1)?;
                (mixed_qkv, z)
            }
        };

        // 2. beta|alpha: grouped [b(v_per_group) | a(v_per_group)] per key-head group.
        let mixed_ba = self.in_proj_ba.forward(x)?;
        let mixed_ba =
            mixed_ba.reshape((batch_size, seq_len, self.num_k_heads, 2 * v_per_group))?;
        let b = mixed_ba.narrow(D::Minus1, 0, v_per_group)?.reshape((
            batch_size,
            seq_len,
            self.num_v_heads,
        ))?;
        let a = mixed_ba
            .narrow(D::Minus1, v_per_group, v_per_group)?
            .reshape((batch_size, seq_len, self.num_v_heads))?;

        // 3. Causal conv1d over the concatenated qkv (includes silu).
        let mixed_qkv = if cache.seqlen_offset > 0 && seq_len == 1 {
            self.causal_conv1d_update(&mixed_qkv, cache)?
        } else {
            self.causal_conv1d_full(&mixed_qkv, cache)?
        };

        // 4. Split conv output back into per-head q, k, v.
        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let q = q.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // 5. beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias). GGUF `ssm_a` already stores
        //    -exp(A_log), so multiply directly (no neg/exp here).
        let beta = hanzo_nn::ops::sigmoid(&b)?;
        let dt_bias = self
            .dt_bias
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let g = self
            .a
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_mul(&softplus(
                &a.to_dtype(DType::F32)?.broadcast_add(&dt_bias)?,
            )?)?;

        // 6. GROUPED repeat q,k to V-head count: V head j -> K head j / v_per_group. The GGUF lays out
        //    every per-V-head tensor (v, z, beta, g) grouped by key-head, so inserting the repeat axis
        //    AFTER the K axis reproduces the `j / v_per_group` pairing (a tiled `% num_k_heads` repeat
        //    would mismatch). When num_k_heads == num_v_heads this collapses to identity.
        let (q, k) = if v_per_group > 1 {
            let q = q
                .unsqueeze(3)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            let k = k
                .unsqueeze(3)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            (q, k)
        } else {
            (q, k)
        };

        // 7. L2-normalize q and k.
        let q = l2_norm(&q, L2_NORM_EPS)?;
        let k = l2_norm(&k, L2_NORM_EPS)?;

        // 8. Recurrent gated delta rule (dispatches to the fused per-backend kernel internally).
        let y = gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut cache.recurrent_state)?;
        cache.seqlen_offset += seq_len;

        // 9. Gated RMSNorm with z, then output projection.
        let z_shape = z.shape().clone();
        let y = y.reshape(((), self.head_v_dim))?;
        let z = z.reshape(((), self.head_v_dim))?;
        let y = self.norm.forward(&y, &z)?;
        let y = y.reshape(z_shape)?;
        let y = y.reshape((batch_size, seq_len, self.value_dim))?;

        self.out_proj.forward(&y)?.to_dtype(orig_dtype)
    }

    fn causal_conv1d_update(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (_batch, seq_len, _conv_dim) = x.dims3()?;
        let x_t = x.transpose(1, 2)?.contiguous()?;

        let state_len = cache.conv_state.dim(2)?;
        let conv_state = cache.conv_state.to_dtype(x_t.dtype())?;
        let hidden_new = Tensor::cat(&[conv_state, x_t], 2)?;
        let new_len = hidden_new.dim(2)?;
        cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;

        let weight = self.conv1d_weight.to_dtype(hidden_new.dtype())?;
        let mut conv_outputs = Vec::with_capacity(seq_len);
        let total_len = hidden_new.dim(2)?;
        for i in (total_len - seq_len)..total_len {
            let window =
                hidden_new.narrow(2, i + 1 - self.conv_kernel_size, self.conv_kernel_size)?;
            let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        let out = Tensor::stack(&conv_outputs, 2)?;
        let out = hanzo_nn::ops::silu(&out)?;
        out.transpose(1, 2)
    }

    fn causal_conv1d_full(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, conv_dim) = x.dims3()?;
        let x_t = x.transpose(1, 2)?.contiguous()?;

        #[cfg(feature = "cuda")]
        if x_t.device().is_cuda() {
            let weight = self.conv1d_weight.to_dtype(x_t.dtype())?.contiguous()?;
            let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
                &x_t,
                &weight,
                &cache.conv_state,
                self.conv_kernel_size,
                false,
            )?;
            cache.conv_state = new_conv_state;
            return output.transpose(1, 2);
        }

        let pad_width = self.conv_kernel_size.saturating_sub(seq_len);
        cache.conv_state = if pad_width > 0 {
            let zeros =
                Tensor::zeros((batch_size, conv_dim, pad_width), x_t.dtype(), x_t.device())?;
            Tensor::cat(&[zeros, x_t.clone()], 2)?
        } else {
            x_t.narrow(2, seq_len - self.conv_kernel_size, self.conv_kernel_size)?
        };

        let padded_t = Tensor::cat(
            &[
                Tensor::zeros(
                    (batch_size, conv_dim, self.conv_kernel_size - 1),
                    x_t.dtype(),
                    x_t.device(),
                )?,
                x_t,
            ],
            2,
        )?;

        let weight = self.conv1d_weight.to_dtype(padded_t.dtype())?;
        let mut conv_outputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let window = padded_t.narrow(2, i, self.conv_kernel_size)?;
            let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        let out = Tensor::stack(&conv_outputs, 2)?;
        let out = hanzo_nn::ops::silu(&out)?;
        out.transpose(1, 2)
    }
}

// ===================== Decoder layer =====================

enum LayerImpl {
    FullAttention(QGatedFullAttention),
    LinearAttention(QGatedDeltaNet),
}

struct DecoderLayer {
    layer_impl: LayerImpl,
    input_layernorm: QRmsNorm,
    post_attention_layernorm: QRmsNorm,
    mlp: FusedMoe,
}

// ===================== Config extraction =====================

#[allow(dead_code)]
struct PropsGGUF {
    head_count: usize,
    head_count_kv: usize,
    block_count: usize,
    embedding_length: usize,
    rms_norm_eps: f32,
    max_seq_len: usize,
    rope_freq_base: f32,
    head_dim: usize,
    rot_dim: usize,
    full_attention_interval: usize,
    // GDN
    conv_kernel: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    // MoE
    num_experts: usize,
    num_experts_per_tok: usize,
}

fn verify_arch(metadata: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>) -> Result<()> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    if actual_arch != "qwen3next" {
        hanzo_ml::bail!("Expected `qwen3next` architecture, got `{actual_arch}`.");
    }
    Ok(())
}

impl PropsGGUF {
    fn try_from(c: &ContentMetadata) -> Result<Self> {
        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.state_size",
            "ssm.group_count",
            "ssm.time_step_rank",
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
        ];
        c.has_required_keys(&required)
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?;

        let embed_len = c
            .get_value::<u32>("embedding_length")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        let head_count = c
            .get_value::<u32>("attention.head_count")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;

        // head_dim from attention.key_length (Qwen3-Next has head_dim != embed/head_count).
        let head_dim = c
            .get_value::<u32>("attention.key_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(embed_len / head_count);

        // Partial-rotary width. Prefer rope.dimension_count; else head_dim * 0.25.
        let rot_dim = c
            .get_value::<u32>("rope.dimension_count")
            .ok()
            .map(|x| x as usize)
            .unwrap_or((head_dim as f64 * DEFAULT_PARTIAL_ROTARY_FACTOR) as usize);

        let head_k_dim = c
            .get_value::<u32>("ssm.state_size")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        let num_k_heads = c
            .get_value::<u32>("ssm.group_count")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        let num_v_heads = c
            .get_value::<u32>("ssm.time_step_rank")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        // head_v_dim = ssm.inner_size / num_v_heads, else == head_k_dim (state_size).
        let head_v_dim = c
            .get_value::<u32>("ssm.inner_size")
            .ok()
            .map(|x| x as usize / num_v_heads)
            .unwrap_or(head_k_dim);

        Ok(Self {
            head_count,
            head_count_kv: {
                // hybrid layers may store head_count_kv as a per-layer array; take the max.
                let key = "attention.head_count_kv";
                c.get_value::<u32>(key)
                    .map(|n| n as usize)
                    .or_else(|_| {
                        c.get_value::<Vec<u32>>(key)
                            .map(|v| v.into_iter().max().unwrap_or(0) as usize)
                    })
                    .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?
            },
            // block_count may include trailing MTP (multi-token-prediction) layers; the transformer
            // depth is block_count - nextn_predict_layers. MTP is ignored for text-only inference.
            block_count: (c
                .get_value::<u32>("block_count")
                .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?
                .saturating_sub(c.get_value::<u32>("nextn_predict_layers").unwrap_or(0)))
                as usize,
            embedding_length: embed_len,
            rms_norm_eps: c
                .get_value("attention.layer_norm_rms_epsilon")
                .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_000_f32),
            head_dim,
            rot_dim,
            full_attention_interval: c
                .get_value::<u32>("full_attention_interval")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(DEFAULT_FULL_ATTENTION_INTERVAL),
            conv_kernel: c
                .get_value::<u32>("ssm.conv_kernel")
                .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?
                as usize,
            head_k_dim,
            head_v_dim,
            num_k_heads,
            num_v_heads,
            num_experts: c
                .get_value::<u32>("expert_count")
                .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?
                as usize,
            num_experts_per_tok: c
                .get_value::<u32>("expert_used_count")
                .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?
                as usize,
        })
    }
}

// ===================== Model =====================

pub struct ModelWeights {
    tok_embeddings: Option<Embedding>,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<LayerType>,
    norm: Option<QRmsNorm>,
    output: Option<Arc<dyn QuantMethod>>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    pp: Option<Arc<RingPipeline>>,
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
            path_prefix: "qwen3next",
            metadata: meta,
        };
        let props = PropsGGUF::try_from(&metadata)?;

        let key_dim = props.num_k_heads * props.head_k_dim;
        let value_dim = props.num_v_heads * props.head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;

        if props.num_v_heads % props.num_k_heads != 0 {
            hanzo_ml::bail!(
                "qwen3next GDN requires num_v_heads ({}) to be a multiple of num_k_heads ({}).",
                props.num_v_heads,
                props.num_k_heads
            );
        }

        // PP: each rank loads only its layer range; rank 0 also owns embed/norm/lm_head. The GDN
        // recurrent + KV state lives on the owning rank. Layer type keys off the GLOBAL index, so
        // the hybrid schedule survives the split.
        let pp = if use_pipeline_parallel() {
            let config = hanzo_quant::RingConfig::load();
            Some(Arc::new(RingPipeline::from_config(&config)))
        } else {
            None
        };
        let layout = match &pp {
            Some(_) => Some(RingLayout::new(props.block_count)?),
            None => None,
        };
        let local_range = layout
            .as_ref()
            .map_or(0..props.block_count, RingLayout::local);
        let is_head = layout.as_ref().is_none_or(RingLayout::is_head);
        let local_start = local_range.start;

        let layer_types: Vec<LayerType> = local_range
            .clone()
            .map(|i| {
                if (i + 1) % props.full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();

        let (tok_embeddings, norm, output) = if is_head {
            let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
            let tok_embeddings =
                Embedding::new(qtok_embeddings.dequantize(device)?, props.embedding_length);
            let norm =
                QRmsNorm::new(ct.tensor("output_norm.weight", device)?, props.rms_norm_eps)?;
            let output = if ct.has_tensor("output.weight") {
                ct.tensor("output.weight", device)?
            } else {
                ct.tensor("token_embd.weight", device)?
            };
            (Some(tok_embeddings), Some(norm), Some(gguf_qmm(output)?))
        } else {
            (None, None, None)
        };

        // One partial NEOX RoPE per device location (shared by the full-attention layers).
        let mut ropes = HashMap::new();
        for layer_idx in local_range.clone() {
            let dev = if pp.is_some() {
                device
            } else {
                mapper.device_for(layer_idx, false).unwrap_or(device)
            };
            if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(dev.location()) {
                e.insert(Arc::new(RotaryEmbedding::new_partial(
                    props.rope_freq_base,
                    props.rot_dim,
                    props.max_seq_len,
                    dev,
                    true,
                    DType::F32,
                )?));
            }
        }

        let mut layers = Vec::with_capacity(layer_types.len());
        for layer_idx in NiceProgressBar::<_, 'b'>(
            local_range.clone(),
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let dev = if pp.is_some() {
                device
            } else {
                mapper.device_for(layer_idx, false).unwrap_or(device)
            };
            let rotary = ropes
                .get(&dev.location())
                .expect("No RoPE for device location!")
                .clone();

            let input_layernorm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), dev)?,
                props.rms_norm_eps,
            )?;
            let post_attention_layernorm = QRmsNorm::new(
                ct.tensor(&format!("{prefix}.post_attention_norm.weight"), dev)?,
                props.rms_norm_eps,
            )?;

            let layer_impl = match layer_types[layer_idx - local_start] {
                LayerType::FullAttention => {
                    let attn_q = gguf_qmm(ct.tensor(&format!("{prefix}.attn_q.weight"), dev)?)?;
                    let attn_k = gguf_qmm(ct.tensor(&format!("{prefix}.attn_k.weight"), dev)?)?;
                    let attn_v = gguf_qmm(ct.tensor(&format!("{prefix}.attn_v.weight"), dev)?)?;
                    let attn_o =
                        gguf_qmm(ct.tensor(&format!("{prefix}.attn_output.weight"), dev)?)?;
                    let q_norm = QRmsNorm::new(
                        ct.tensor(&format!("{prefix}.attn_q_norm.weight"), dev)?,
                        props.rms_norm_eps,
                    )?;
                    let k_norm = QRmsNorm::new(
                        ct.tensor(&format!("{prefix}.attn_k_norm.weight"), dev)?,
                        props.rms_norm_eps,
                    )?;
                    let paged_attn = match attention_mechanism {
                        _ if pp.is_some() => None,
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(props.head_dim, dev, None)?)
                        }
                        AttentionImplementation::Eager => None,
                    };
                    LayerImpl::FullAttention(QGatedFullAttention {
                        attn_q,
                        attn_k,
                        attn_v,
                        attn_o,
                        q_norm,
                        k_norm,
                        n_head: props.head_count,
                        n_kv_head: props.head_count_kv,
                        head_dim: props.head_dim,
                        rotary,
                        paged_attn,
                        sdpa_params: SdpaParams {
                            n_kv_groups: props.head_count / props.head_count_kv,
                            softcap: None,
                            softmax_scale: 1.0 / (props.head_dim as f32).sqrt(),
                            sliding_window: None,
                            sinks: None,
                        },
                        dtype,
                    })
                }
                LayerType::LinearAttention => {
                    // Optimized (attn_qkv + attn_gate) vs legacy (ssm_in) input encoding.
                    let qkvz = if ct.has_tensor(&format!("{prefix}.attn_qkv.weight")) {
                        QkvzProj::Split {
                            qkv: gguf_qmm(ct.tensor(&format!("{prefix}.attn_qkv.weight"), dev)?)?,
                            z: gguf_qmm(ct.tensor(&format!("{prefix}.attn_gate.weight"), dev)?)?,
                        }
                    } else {
                        QkvzProj::Merged(gguf_qmm(
                            ct.tensor(&format!("{prefix}.ssm_in.weight"), dev)?,
                        )?)
                    };
                    let in_proj_ba = gguf_qmm(ct.tensor(&format!("{prefix}.ssm_ba.weight"), dev)?)?;
                    let out_proj = gguf_qmm(ct.tensor(&format!("{prefix}.ssm_out.weight"), dev)?)?;

                    // conv1d / dt / a are small f32 params kept dequantized.
                    let mut conv1d_weight = ct
                        .tensor(&format!("{prefix}.ssm_conv1d.weight"), dev)?
                        .dequantize(dev)?;
                    if conv1d_weight.rank() == 3 {
                        conv1d_weight = conv1d_weight.squeeze(1)?;
                    }
                    let dt_bias = ct
                        .tensor(&format!("{prefix}.ssm_dt.bias"), dev)
                        .or_else(|_| ct.tensor(&format!("{prefix}.ssm_dt"), dev))?
                        .dequantize(dev)?
                        .to_dtype(DType::F32)?;
                    let a = ct
                        .tensor(&format!("{prefix}.ssm_a"), dev)?
                        .dequantize(dev)?
                        .to_dtype(DType::F32)?;

                    let ssm_norm_w = ct
                        .tensor(&format!("{prefix}.ssm_norm.weight"), dev)?
                        .dequantize(dev)?;
                    let norm = RmsNormGated::from_weight(ssm_norm_w, props.rms_norm_eps as f64);

                    LayerImpl::LinearAttention(QGatedDeltaNet {
                        qkvz,
                        in_proj_ba,
                        conv1d_weight,
                        dt_bias,
                        a,
                        norm,
                        out_proj,
                        num_k_heads: props.num_k_heads,
                        num_v_heads: props.num_v_heads,
                        head_k_dim: props.head_k_dim,
                        head_v_dim: props.head_v_dim,
                        conv_kernel_size: props.conv_kernel,
                        key_dim,
                        value_dim,
                    })
                }
            };

            let mlp = FusedMoe::from_gguf(&mut ct, &prefix, dev, props.num_experts_per_tok)?;

            layers.push(DecoderLayer {
                layer_impl,
                input_layernorm,
                post_attention_layernorm,
                mlp,
            });
        }

        // Pipeline hybrid cache (recurrent pool for GDN layers + KV cache for attention layers).
        let pipeline_layer_types: Vec<HybridLayerType> = layer_types
            .iter()
            .map(|lt| match lt {
                LayerType::FullAttention => HybridLayerType::Attention,
                LayerType::LinearAttention => HybridLayerType::Recurrent,
            })
            .collect();
        let hybrid_cache_config = HybridCacheConfig {
            layer_types: pipeline_layer_types,
            max_seq_len: props.max_seq_len,
            recurrent: RecurrentLayerConfig {
                conv_dim,
                conv_width: props.conv_kernel,
                state_dims: vec![props.num_v_heads, props.head_k_dim, props.head_v_dim],
            },
        };
        let pipeline_cache = Arc::new(Mutex::new(
            HybridCache::new(hybrid_cache_config, dtype, device)
                .map_err(|e| hanzo_ml::Error::Msg(format!("Failed to create hybrid cache: {e}")))?,
        ));

        Ok(Self {
            tok_embeddings,
            layers,
            layer_types,
            norm,
            output,
            device: device.clone(),
            cache: EitherCache::Hybrid(pipeline_cache),
            max_seq_len: props.max_seq_len,
            mapper: if pp.is_some() { None } else { Some(mapper) },
            dtype,
            pp,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        if self.pp.is_some() {
            return pp_head_forward(self, input_ids, seqlen_offsets, context_lens);
        }
        let (b_sz, _seq_len) = input_ids.dims2()?;
        let mut x = self.tok_embeddings.as_ref().unwrap().forward(input_ids)?;

        let mut hybrid_cache = self.cache.hybrid();
        let state_indices = hybrid_cache.state_indices().cloned();
        let state_indices_host: Option<Vec<u32>> =
            hybrid_cache.state_indices_host().map(|s| s.to_vec());
        if self
            .layer_types
            .iter()
            .any(|lt| matches!(lt, LayerType::LinearAttention))
            && state_indices.is_none()
        {
            hanzo_ml::bail!(
                "Hybrid recurrent state indices are required for linear-attention layers."
            );
        }

        // Past-kv length: with PagedAttention the running context lives in the paged pool, so it
        // comes from the host `seqlen_offsets`; decode (non-first chunk) needs no mask. Without
        // paging (CPU/eager) fall back to the hybrid cache. Mirrors quantized_qwen3_5_moe.
        let mask = CausalMasker.make_causal_mask(
            input_ids,
            match metadata.as_ref() {
                Some(_) => &seqlen_offsets as &dyn PastKvLenCache,
                None => &*hybrid_cache as &dyn PastKvLenCache,
            },
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

        // RoPE positions: prefer the stable device buffer from decode-graph metadata (refreshed in
        // place across replays); else synthesize from the host `seqlen_offsets`. Mirrors
        // quantized_qwen3.rs. One position per sequence.
        let positions = match metadata
            .as_ref()
            .and_then(|(_, meta)| meta.rope_positions.as_ref())
            .and_then(|positions| positions.get(&self.device.location()))
        {
            Some(positions) => positions.clone(),
            None => {
                let pos = seqlen_offsets
                    .iter()
                    .copied()
                    .map(u32::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(hanzo_ml::Error::wrap)?;
                Tensor::from_vec(pos, seqlen_offsets.len().max(1), &self.device)?
            }
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                x = mapper.map(x, layer_idx)?;
            }
            let residual = x.clone();
            let normed = layer.input_layernorm.forward(&x)?;

            let attn_out = match &layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    let paged = metadata
                        .as_ref()
                        .map(|(kv_cache, meta)| (kv_cache[layer_idx].clone(), *meta));
                    let Some(HybridLayerCache::Attention(kv_cache)) =
                        hybrid_cache.get_mut(layer_idx)
                    else {
                        hanzo_ml::bail!("Hybrid cache layer {layer_idx} not attention.");
                    };
                    let positions = if positions.device().same_device(normed.device()) {
                        positions.clone()
                    } else {
                        positions.to_device(normed.device())?
                    };
                    attn.forward(
                        &normed,
                        &mask.get(normed.device()),
                        seqlen_offsets,
                        &positions,
                        kv_cache,
                        paged,
                    )?
                }
                LayerImpl::LinearAttention(gdn) => {
                    let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer_idx)
                    else {
                        hanzo_ml::bail!("Hybrid cache layer {layer_idx} not recurrent.");
                    };
                    if b_sz == 1 {
                        // Single sequence: constant-offset narrow/slice_set on the host slot, no
                        // to_vec1 sync (mirrors quantized_qwen3_5_moe's graph-safe fast path).
                        let slot = state_indices_host
                            .as_ref()
                            .and_then(|s| s.first().copied())
                            .ok_or_else(|| {
                                hanzo_ml::Error::msg("missing host recurrent state index")
                            })? as usize;
                        let mut gdn_cache = GdnLayerCache {
                            conv_state: pool.conv_state.narrow(0, slot, 1)?,
                            recurrent_state: pool.recurrent_state.narrow(0, slot, 1)?,
                            seqlen_offset: seqlen_offsets.first().copied().unwrap_or(0),
                        };
                        let out = gdn.forward(&normed, &mut gdn_cache)?;
                        let conv_dt = pool.conv_state.dtype();
                        let rec_dt = pool.recurrent_state.dtype();
                        pool.conv_state.slice_set(
                            &gdn_cache.conv_state.to_dtype(conv_dt)?.contiguous()?,
                            0,
                            slot,
                        )?;
                        pool.recurrent_state.slice_set(
                            &gdn_cache.recurrent_state.to_dtype(rec_dt)?.contiguous()?,
                            0,
                            slot,
                        )?;
                        pool.set_seqlen_offset(slot, gdn_cache.seqlen_offset);
                        out
                    } else {
                        let indices = state_indices
                            .as_ref()
                            .expect("checked above: recurrent indices required");
                        let indices_vec: Vec<u32> = indices.to_vec1()?;
                        if indices_vec.is_empty() {
                            hanzo_ml::bail!("Hybrid recurrent state indices are empty.");
                        }
                        let first_offset = pool.get_seqlen_offset(indices_vec[0] as usize);
                        if indices_vec
                            .iter()
                            .any(|&idx| pool.get_seqlen_offset(idx as usize) != first_offset)
                        {
                            hanzo_ml::bail!(
                                "Hybrid recurrent seqlen offsets diverged within a batch for layer {layer_idx}."
                            );
                        }
                        let conv_state = pool.gather_conv_state(indices)?;
                        let recurrent_state = pool.gather_recurrent_state(indices)?;
                        let mut gdn_cache = GdnLayerCache {
                            conv_state,
                            recurrent_state,
                            seqlen_offset: first_offset,
                        };
                        let out = gdn.forward(&normed, &mut gdn_cache)?;
                        pool.scatter_conv_state(indices, &gdn_cache.conv_state)?;
                        pool.scatter_recurrent_state(indices, &gdn_cache.recurrent_state)?;
                        let delta = gdn_cache.seqlen_offset.saturating_sub(first_offset);
                        for &idx in &indices_vec {
                            let updated = pool.get_seqlen_offset(idx as usize) + delta;
                            pool.set_seqlen_offset(idx as usize, updated);
                        }
                        out
                    }
                }
            };

            let x_mid = (attn_out + residual)?;
            let residual = &x_mid;
            let normed = layer.post_attention_layernorm.forward(&x_mid)?;
            let ffn_out = layer.mlp.forward(&normed)?;
            x = (ffn_out + residual)?;
        }

        let x = x.to_device(&self.device)?;
        let x = self.norm.as_ref().unwrap().forward(&x)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.as_ref().unwrap().forward(&x.contiguous()?)
    }

    fn run_local_layers(&self, h: &Tensor, offsets: &[usize]) -> Result<Tensor> {
        let mut hybrid_cache = self.cache.hybrid();
        let is_head = self.pp.as_ref().unwrap().is_head();

        // Worker ranks are stateless across requests: a fresh prompt (past-kv len 0) zeroes the
        // local KV + recurrent state. The head slot is (re)zeroed by the HybridCacheManager.
        if !is_head && offsets.first().copied().unwrap_or(0) == 0 {
            hybrid_cache.reset();
        }
        // Head reuses its manager-allocated slot; a worker always drives slot 0 (batch is 1).
        let slot = hybrid_cache
            .state_indices_host()
            .and_then(|s| s.first().copied())
            .unwrap_or(0) as usize;

        let ids2d = h.narrow(2, 0, 1)?.squeeze(2)?;
        let mask = CausalMasker.make_causal_mask(
            &ids2d,
            &offsets as &dyn PastKvLenCache,
            self.dtype,
            &CausalMaskConfig::default(),
        )?;
        let mask = DeviceMappedMask::from_single(mask);
        let positions = {
            let pos = offsets
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(hanzo_ml::Error::wrap)?;
            Tensor::from_vec(pos, offsets.len().max(1), &self.device)?
        };

        let mut x = h.clone();
        for (local_idx, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let normed = layer.input_layernorm.forward(&x)?;
            let attn_out = match &layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    let Some(HybridLayerCache::Attention(kv_cache)) =
                        hybrid_cache.get_mut(local_idx)
                    else {
                        hanzo_ml::bail!("Hybrid cache layer {local_idx} not attention.");
                    };
                    attn.forward(
                        &normed,
                        &mask.get(normed.device()),
                        offsets,
                        &positions,
                        kv_cache,
                        None,
                    )?
                }
                LayerImpl::LinearAttention(gdn) => {
                    let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(local_idx)
                    else {
                        hanzo_ml::bail!("Hybrid cache layer {local_idx} not recurrent.");
                    };
                    let mut gdn_cache = GdnLayerCache {
                        conv_state: pool.conv_state.narrow(0, slot, 1)?,
                        recurrent_state: pool.recurrent_state.narrow(0, slot, 1)?,
                        seqlen_offset: offsets.first().copied().unwrap_or(0),
                    };
                    let out = gdn.forward(&normed, &mut gdn_cache)?;
                    let conv_dt = pool.conv_state.dtype();
                    let rec_dt = pool.recurrent_state.dtype();
                    pool.conv_state.slice_set(
                        &gdn_cache.conv_state.to_dtype(conv_dt)?.contiguous()?,
                        0,
                        slot,
                    )?;
                    pool.recurrent_state.slice_set(
                        &gdn_cache.recurrent_state.to_dtype(rec_dt)?.contiguous()?,
                        0,
                        slot,
                    )?;
                    pool.set_seqlen_offset(slot, gdn_cache.seqlen_offset);
                    out
                }
            };
            let x_mid = (attn_out + residual)?;
            let residual = &x_mid;
            let normed = layer.post_attention_layernorm.forward(&x_mid)?;
            let ffn_out = layer.mlp.forward(&normed)?;
            x = (ffn_out + residual)?;
        }
        Ok(x)
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
