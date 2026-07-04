#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Quantized (GGUF) loader for Qwen3.5 / Qwen3.6 hybrid models.
//!
//! Architecture string: `qwen35moe` (35B-A3B MoE, primary target) or `qwen35` (27B dense).
//! Hybrid per-layer schedule: every `full_attention_interval`-th layer (default 4) is a gated
//! full-attention layer (partial-rotary interleaved mRoPE + sigmoid output gate); the rest are
//! Gated-DeltaNet (GDN) linear-attention layers (causal conv1d kernel=4, gated RMSNorm, recurrent
//! state). MoE variant uses 256 experts (8 routed + 1 shared); dense variant uses a plain SwiGLU MLP.
//!
//! The GDN math mirrors `models::gdn` / `models::qwen3_next` / `vision_models::qwen3_5_moe::text`.
//! The recurrent state is driven through the pipeline `HybridCache` exactly like `qwen3_next`.
//!
//! NOT YET CORRECTNESS-VERIFIED on a GPU with a real GGUF. The GGUF tensor-name mapping and the
//! GDN V-head ordering are documented inline; every assumption is called out. MTP and vision are
//! ignored (text-only).
//!
//! ===================== GGUF tensor-name mapping (source: llama.cpp) =====================
//! Verified against llama.cpp `gguf-py/gguf/constants.py` (MODEL_ARCH.QWEN35 / QWEN35MOE),
//! `gguf-py/gguf/tensor_mapping.py`, `conversion/qwen.py` (Qwen3_5MoeTextModel ->
//! _LinearAttentionVReorderBase -> Qwen3NextModel) and `src/models/qwen35moe.cpp`.
//!
//! Per layer `blk.{i}`:
//!   GDN (linear-attention) layers:
//!     attn_qkv.weight   <- linear_attn.in_proj_qkv  (merged q,k,v; key_dim*2 + value_dim, hidden)
//!     attn_gate.weight  <- linear_attn.in_proj_z    (the z gate; value_dim, hidden)
//!     ssm_conv1d.weight <- linear_attn.conv1d squeezed to 2D (conv_dim, kernel)
//!     ssm_dt.bias       <- linear_attn.dt_bias      (1D, num_v_heads, f32) [note: GGUF suffix is `bias`]
//!     ssm_a             <- -exp(A_log) precomputed at conversion (1D, num_v_heads, f32) [no `.weight`]
//!     ssm_beta.weight   <- linear_attn.in_proj_b    (num_v_heads, hidden)
//!     ssm_alpha.weight  <- linear_attn.in_proj_a    (num_v_heads, hidden)
//!     ssm_norm.weight   <- linear_attn.norm         (gated RMSNorm; head_v_dim)
//!     ssm_out.weight    <- linear_attn.out_proj     (value_dim, hidden)
//!   Full-attention layers:
//!     attn_q.weight     <- self_attn.q_proj  (DOUBLED: num_heads*head_dim*2 = query + gate)
//!     attn_k/attn_v/attn_output, attn_q_norm, attn_k_norm  (standard GQA + qk-norm)
//!   Shared (both): attn_norm (input_layernorm), post_attention_norm (post-attn layernorm).
//!   MoE FFN: ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps,
//!            ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp.
//!   Dense FFN: ffn_gate, ffn_up, ffn_down.
//!   Global: token_embd, output_norm, output (tied to token_embd when absent).
//!
//! Metadata keys (prefixed with the arch string by ContentMetadata):
//!   attention.head_count / head_count_kv / key_length / value_length / layer_norm_rms_epsilon
//!   block_count, context_length, rope.freq_base, rope.dimension_count (= rot_dim),
//!   rope.dimension_sections (mrope [t,h,w,0]), full_attention_interval,
//!   ssm.conv_kernel (=linear_conv_kernel_dim), ssm.state_size (=linear_key_head_dim, also head_v_dim),
//!   ssm.group_count (=linear_num_key_heads), ssm.time_step_rank (=linear_num_value_heads),
//!   ssm.inner_size (=value_dim), expert_count, expert_used_count, expert_feed_forward_length.
//!
//! V-head ordering: llama.cpp's qwen3.5 converter (_LinearAttentionVReorderBase) REORDERS the V
//! heads of in_proj_qkv(v part) / in_proj_z / in_proj_a / in_proj_b / out_proj / conv1d(v part) /
//! A_log / dt_bias from HF grouped order [K0_v0..v{r-1}, K1_v0..v{r-1}, ...] into TILED order
//! [v0_K0..v0_K{K-1}, v1_K0..v1_K{K-1}, ...]. We do NOT undo this at load; instead the recurrence
//! consumes tiled order natively. Every per-V-head tensor (v, z, beta, g, conv V-channels) comes
//! straight from a GGUF projection in tiled order, and out_proj's input columns are tiled too, so
//! the layer is self-consistent end-to-end. The only place that mixes K-indexed and V-indexed
//! tensors is the q/k repeat in `QGatedDeltaNet::forward`, which TILES q/k (V head j -> K head
//! j % num_k_heads) to match. When num_k_heads == num_v_heads the reorder is a no-op and the tile
//! collapses to identity, leaving the original (verified) path unchanged. The shared safetensors
//! `gdn.rs` recurrence instead consumes HF grouped order (it repeats grouped: j -> j / v_per_group);
//! that path is untouched and still correct for its grouped inputs.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMaskConfig, CausalMasker, QRmsNorm, Qwen3VLRotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::models::gdn::{
    gated_delta_rule_recurrence, l2_norm, softplus, GdnLayerCache, RmsNormGated,
};
use crate::ops::{TopKLastDimOp, TopKOutput};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use hanzo_ml::quantized::QMatMul;
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

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

// ===================== MoE / dense FFN =====================

struct FusedMoe {
    gate: QMatMul,
    gate_experts: QMatMul,
    up_experts: QMatMul,
    down_experts: QMatMul,
    shared_gate: QMatMul,
    shared_gate_proj: Arc<dyn QuantMethod>,
    shared_up_proj: Arc<dyn QuantMethod>,
    shared_down_proj: Arc<dyn QuantMethod>,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl FusedMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let original_dtype = xs.dtype();
        let (num_tokens, hidden_dim) = xs.dims2()?;

        let router_logits = self.gate.forward(&xs.to_dtype(DType::F32)?)?;
        let routing_weights = hanzo_nn::ops::softmax_last_dim(&router_logits)?;

        let TopKOutput {
            values: mut scores,
            indices,
        } = routing_weights.topk(self.num_experts_per_tok)?;
        if self.norm_topk_prob {
            scores = scores.broadcast_div(&scores.sum_keepdim(D::Minus1)?)?;
        }

        let routed = {
            let xs_e = xs.reshape((num_tokens, 1, hidden_dim))?;
            let gate = self.gate_experts.indexed_moe_forward(&xs_e, &indices)?;
            let up = self.up_experts.indexed_moe_forward(&xs_e, &indices)?;
            let activated = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
            self.down_experts
                .indexed_moe_forward(&activated, &indices)?
        };
        let routed = routed
            .broadcast_mul(&scores.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?;

        // Shared expert with sigmoid gating, matching qwen3_next SparseMoeBlock.
        let shared_g = self.shared_gate_proj.forward(&xs)?;
        let shared_u = self.shared_up_proj.forward(&xs)?;
        let shared_act =
            crate::ops::mul_and_act(&shared_g, &shared_u, crate::layers::Activation::Silu)?;
        let shared_out = self.shared_down_proj.forward(&shared_act)?;
        let shared_gate =
            hanzo_nn::ops::sigmoid(&self.shared_gate.forward(&xs.to_dtype(DType::F32)?)?)?
                .to_dtype(shared_out.dtype())?;
        let shared_out = shared_out.broadcast_mul(&shared_gate)?;

        (routed + shared_out)?
            .reshape((batch, seq_len, hidden_dim))?
            .to_dtype(original_dtype)
    }
}

struct DenseMlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

impl DenseMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(xs)?;
        let up = self.up.forward(xs)?;
        let y = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        self.down.forward(&y)
    }
}

enum MoeOrMlp {
    FusedMoe(Box<FusedMoe>),
    Mlp(DenseMlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs),
        }
    }
}

// ===================== Gated full-attention layer =====================

struct GatedFullAttention {
    // q_proj output is doubled: first head_dim is q, second head_dim is the output gate.
    attn_q: Arc<dyn QuantMethod>,
    attn_k: Arc<dyn QuantMethod>,
    attn_v: Arc<dyn QuantMethod>,
    attn_o: Arc<dyn QuantMethod>,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<Qwen3VLRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl GatedFullAttention {
    fn forward(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: &mut KvCache,
        paged: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q_gate = self.attn_q.forward(x)?;
        let k = self.attn_k.forward(x)?;
        let v = self.attn_v.forward(x)?;

        // Split q_gate into q and gate (interleaved per head: [q_head, gate_head] * n_head).
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

        // Partial-rotary interleaved mRoPE + qk RMSNorm.
        let (q, k) = self.rotary.forward_qk_norm(
            cos_sin,
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
        )?;

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        // PagedAttention keeps the decode KV write/read position-invariant (device slot_mappings +
        // bucketed context_lens), which is what makes the whole decode forward capturable by a CUDA
        // graph. The plain hybrid `KvCache::append` path (host-offset write, frozen under capture) is
        // kept only for the eager / no-paged (CPU) case. Mirrors quantized_qwen3_moe / qwen3_next.
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
                paged_attn.forward(&q, &k, &v, mask, None, None, &input_metadata, &self.sdpa_params, None)?
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

struct QGatedDeltaNet {
    in_proj_qkv: Arc<dyn QuantMethod>, // merged q,k,v (no z) -> attn_qkv
    in_proj_z: Arc<dyn QuantMethod>,   // z gate -> attn_gate
    in_proj_b: Arc<dyn QuantMethod>,   // beta -> ssm_beta
    in_proj_a: Arc<dyn QuantMethod>,   // alpha -> ssm_alpha
    conv1d_weight: Tensor,             // (conv_dim, kernel) f32
    dt_bias: Tensor,                   // (num_v_heads,) f32
    a: Tensor,                         // -exp(A_log), (num_v_heads,) f32, precomputed in GGUF
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
        // GDN recurrence + gates run in f32 end-to-end to avoid bf16/f32 boundary mismatches;
        // input is lifted to f32 here and the out_proj result cast back to the model dtype.
        let orig_dtype = x.dtype();
        let x = &x.to_dtype(DType::F32)?;
        let (batch_size, seq_len, _hidden) = x.dims3()?;
        let dtype = x.dtype();
        let v_per_group = self.num_v_heads / self.num_k_heads;

        // 1. Projections. qkv is already merged [q | k | v]; z is the gate.
        let mixed_qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?;
        let b = self.in_proj_b.forward(x)?;
        let a = self.in_proj_a.forward(x)?;

        let z = z.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let b = b.reshape((batch_size, seq_len, self.num_v_heads))?;
        let a = a.reshape((batch_size, seq_len, self.num_v_heads))?;

        // 2. Causal conv1d over the concatenated qkv (includes silu).
        let mixed_qkv = if cache.seqlen_offset > 0 && seq_len == 1 {
            self.causal_conv1d_update(&mixed_qkv, cache)?
        } else {
            self.causal_conv1d_full(&mixed_qkv, cache)?
        };

        // 3. Split conv output back into per-head q, k, v.
        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let q = q.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // 4. beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias).
        //    The GGUF `ssm_a` already stores -exp(A_log), so we multiply directly (no neg/exp here).
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
            )?)?
            .to_dtype(dtype)?;

        // 5. If num_v_heads > num_k_heads, tile q,k to V-head count. The GGUF lays out every per-V-head
        //    tensor (v, z, beta, g, conv V-channels, out_proj columns) in tiled order [v0_K0..v0_K{K-1},
        //    v1_K0..], so V head j pairs with K head j % num_k_heads. Tiling K (insert axis BEFORE the
        //    K axis, repeat, flatten) reproduces that j -> j % num_k_heads pairing; a grouped repeat
        //    (j -> j / v_per_group) would mismatch. The whole layer then stays in tiled order through
        //    out_proj, so no weights need re-permuting at load.
        let (q, k) = if v_per_group > 1 {
            let q = q
                .unsqueeze(2)?
                .repeat((1, 1, v_per_group, 1, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            let k = k
                .unsqueeze(2)?
                .repeat((1, 1, v_per_group, 1, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            (q, k)
        } else {
            (q, k)
        };

        // 6. L2-normalize q and k.
        let q = l2_norm(&q, L2_NORM_EPS)?;
        let k = l2_norm(&k, L2_NORM_EPS)?;

        // 7. Recurrent gated delta rule (dispatches to the fused per-backend kernel internally).
        let y = gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut cache.recurrent_state)?;
        cache.seqlen_offset += seq_len;

        // 8. Gated RMSNorm with z, then output projection.
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

        // Vulkan conv1d single-step kernel (gdn_conv1d_step_vulkan) isn't ported to canonical
        // hanzo-ml yet; the portable candle path below runs correctly on the Vulkan device.

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

    // Single decode step (seq_len==1, batch==1) of the causal conv1d on Vulkan. conv_state is
    // (1, conv_dim, k) -- it stores k columns; the step drops the oldest and appends x, exactly as
    // the CPU causal_conv1d_update does. conv_state is updated in place in VRAM (aliases the pool
    // buffer); x is (1, 1, conv_dim). Returns silu(conv) as (1, 1, conv_dim). The GGUF conv1d_weight
    // is (conv_dim, k) with no bias.
    fn causal_conv1d_update_vulkan(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let conv_dim = self.conv1d_weight.dim(0)?;
        let x_flat = x.reshape(conv_dim)?.to_dtype(DType::F32)?.contiguous()?;
        let weight = self.conv1d_weight.to_dtype(DType::F32)?.contiguous()?;
        let mut conv_state = cache
            .conv_state
            .reshape((conv_dim, self.conv_kernel_size))?;
        let out = crate::vulkan::gdn::gdn_conv1d_step_vulkan(&mut conv_state, &x_flat, &weight)?;
        cache.conv_state = conv_state.reshape((1, conv_dim, self.conv_kernel_size))?;
        out.reshape((1, 1, conv_dim))
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
    FullAttention(GatedFullAttention),
    LinearAttention(QGatedDeltaNet),
}

struct DecoderLayer {
    layer_impl: LayerImpl,
    input_layernorm: QRmsNorm,
    post_attention_layernorm: QRmsNorm,
    mlp: MoeOrMlp,
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
    mrope_section: Vec<usize>,
    full_attention_interval: usize,
    // GDN
    conv_kernel: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    // MoE (None for dense)
    num_experts: Option<usize>,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    is_moe: bool,
}

fn verify_arch(
    metadata: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>,
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    if actual_arch != "qwen35moe" && actual_arch != "qwen35" {
        hanzo_ml::bail!("Expected `qwen35moe`/`qwen35` architecture, got `{actual_arch}`.");
    }
    Ok(actual_arch)
}

impl PropsGGUF {
    fn try_from(c: &ContentMetadata, is_moe: bool) -> Result<Self> {
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
        ];
        c.has_required_keys(&required)
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?;

        let embed_len = c
            .get_value::<u32>("embedding_length")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        let head_count = c
            .get_value::<u32>("attention.head_count")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;

        let head_dim = c
            .get_value::<u32>("attention.key_length")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(embed_len / head_count);

        // rope.dimension_count is the rotary (partial) dim; fall back to head_dim * 0.25.
        let rot_dim = c
            .get_value::<u32>("rope.dimension_count")
            .ok()
            .map(|x| x as usize)
            .unwrap_or((head_dim as f64 * DEFAULT_PARTIAL_ROTARY_FACTOR) as usize);

        let mrope_section = c
            .get_value::<Vec<u32>>("rope.dimension_sections")
            .ok()
            .map(|v| v.into_iter().map(|x| x as usize).collect::<Vec<_>>())
            // mrope_section sums to rot_dim/2; default [t,h,w,0] from llama.cpp is [11,11,10,0].
            .unwrap_or_else(|| vec![11, 11, 10, 0]);

        let head_k_dim = c
            .get_value::<u32>("ssm.state_size")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        let num_k_heads = c
            .get_value::<u32>("ssm.group_count")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        let num_v_heads = c
            .get_value::<u32>("ssm.time_step_rank")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        // head_v_dim: ssm.inner_size / num_v_heads, else equal to head_k_dim (state_size).
        let head_v_dim = c
            .get_value::<u32>("ssm.inner_size")
            .ok()
            .map(|x| x as usize / num_v_heads)
            .unwrap_or(head_k_dim);

        let (num_experts, num_experts_per_tok, moe_intermediate_size) = if is_moe {
            (
                Some(
                    c.get_value::<u32>("expert_count")
                        .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?
                        as usize,
                ),
                c.get_value::<u32>("expert_used_count")
                    .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize,
                c.get_value::<u32>("expert_feed_forward_length")
                    .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize,
            )
        } else {
            (None, 0, 0)
        };

        Ok(Self {
            head_count,
            head_count_kv: {
                // hybrid layers store head_count_kv as a per-layer array; take the max (attention layers)
                let key = "attention.head_count_kv";
                c.get_value::<u32>(key)
                    .map(|n| n as usize)
                    .or_else(|_| {
                        c.get_value::<Vec<u32>>(key)
                            .map(|v| v.into_iter().max().unwrap_or(0) as usize)
                    })
                    .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))?
            },
            // block_count includes trailing MTP (multi-token-prediction) layers in some exports
            // (e.g. the MXFP4 gguf: block_count=41, nextn_predict_layers=1). MTP is ignored for
            // text-only inference, so the transformer depth is block_count - nextn_predict_layers.
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
            mrope_section,
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
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            is_moe,
        })
    }
}

// ===================== Model =====================

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<LayerType>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    rotary: Arc<Qwen3VLRotaryEmbedding>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

fn gguf_qmm(q: hanzo_ml::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
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
        let actual_arch = verify_arch(meta)?;
        let is_moe = actual_arch == "qwen35moe";

        let metadata = ContentMetadata {
            path_prefix: &actual_arch,
            metadata: meta,
        };
        let props = PropsGGUF::try_from(&metadata, is_moe)?;

        let key_dim = props.num_k_heads * props.head_k_dim;
        let value_dim = props.num_v_heads * props.head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;

        // GGUF stores V heads in tiled order (converter's _LinearAttentionVReorderBase). The K==V
        // case is a no-op reorder; for V = m*K we consume tiled order directly (QGatedDeltaNet
        // tiles q/k to match). Only a non-integer V/K split is genuinely unsupported.
        if props.num_v_heads % props.num_k_heads != 0 {
            hanzo_ml::bail!(
                "qwen35 GGUF GDN requires num_v_heads ({}) to be a multiple of num_k_heads ({}).",
                props.num_v_heads,
                props.num_k_heads
            );
        }

        let layer_types: Vec<LayerType> = (0..props.block_count)
            .map(|i| {
                if (i + 1) % props.full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, props.rms_norm_eps)?;
        let output = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };

        // One mRoPE per device location. head_dim arg = rot_dim so cos/sin width = rot_dim/2.
        let mut ropes = HashMap::new();
        for layer_idx in 0..props.block_count {
            let dev = mapper.device_for(layer_idx, false).unwrap_or(device);
            if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(dev.location()) {
                e.insert(Arc::new(Qwen3VLRotaryEmbedding::new(
                    props.rope_freq_base,
                    props.rot_dim,
                    dev,
                    props.mrope_section.clone(),
                )?));
            }
        }
        let default_rotary = ropes
            .get(&device.location())
            .cloned()
            .unwrap_or_else(|| ropes.values().next().unwrap().clone());

        let mut layers = Vec::with_capacity(props.block_count);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..props.block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let dev = mapper.device_for(layer_idx, false).unwrap_or(device);
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

            let layer_impl = match layer_types[layer_idx] {
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
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(props.head_dim, dev, None)?)
                        }
                        AttentionImplementation::Eager => None,
                    };
                    LayerImpl::FullAttention(GatedFullAttention {
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
                    let in_proj_qkv =
                        gguf_qmm(ct.tensor(&format!("{prefix}.attn_qkv.weight"), dev)?)?;
                    let in_proj_z =
                        gguf_qmm(ct.tensor(&format!("{prefix}.attn_gate.weight"), dev)?)?;
                    let in_proj_b =
                        gguf_qmm(ct.tensor(&format!("{prefix}.ssm_beta.weight"), dev)?)?;
                    let in_proj_a =
                        gguf_qmm(ct.tensor(&format!("{prefix}.ssm_alpha.weight"), dev)?)?;
                    let out_proj = gguf_qmm(ct.tensor(&format!("{prefix}.ssm_out.weight"), dev)?)?;

                    // conv1d / dt / a are small f32 params kept dequantized.
                    let mut conv1d_weight = ct
                        .tensor(&format!("{prefix}.ssm_conv1d.weight"), dev)?
                        .dequantize(dev)?;
                    // GGUF squeezes conv1d to 2D (conv_dim, kernel); ensure 2D.
                    if conv1d_weight.rank() == 3 {
                        conv1d_weight = conv1d_weight.squeeze(1)?;
                    }
                    // GGUF conversions name this `ssm_dt.bias` (Unsloth/llama.cpp) or `ssm_dt`; accept both.
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
                        in_proj_qkv,
                        in_proj_z,
                        in_proj_b,
                        in_proj_a,
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

            let mlp = if is_moe {
                let gate = ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), dev)?;
                let gate_experts = ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), dev)?;
                let up_experts = ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), dev)?;
                let down_experts = ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), dev)?;
                let shared_gate = ct.tensor(&format!("{prefix}.ffn_gate_inp_shexp.weight"), dev)?;
                let shared_gate_proj =
                    gguf_qmm(ct.tensor(&format!("{prefix}.ffn_gate_shexp.weight"), dev)?)?;
                let shared_up_proj =
                    gguf_qmm(ct.tensor(&format!("{prefix}.ffn_up_shexp.weight"), dev)?)?;
                let shared_down_proj =
                    gguf_qmm(ct.tensor(&format!("{prefix}.ffn_down_shexp.weight"), dev)?)?;
                MoeOrMlp::FusedMoe(Box::new(FusedMoe {
                    gate: QMatMul::from_qtensor(gate)?,
                    gate_experts: QMatMul::from_qtensor(gate_experts)?,
                    up_experts: QMatMul::from_qtensor(up_experts)?,
                    down_experts: QMatMul::from_qtensor(down_experts)?,
                    // ffn_gate_inp_shexp is a hidden->1 shared-expert gate stored 1D [hidden]; the
                    // matmul needs 2D, so dequantize and reshape to [1, hidden].
                    shared_gate: {
                        let w = shared_gate.dequantize(dev)?;
                        QMatMul::Tensor(if w.rank() == 1 { w.unsqueeze(0)? } else { w })
                    },
                    shared_gate_proj,
                    shared_up_proj,
                    shared_down_proj,
                    norm_topk_prob: true,
                    num_experts_per_tok: props.num_experts_per_tok,
                }))
            } else {
                let gate = gguf_qmm(ct.tensor(&format!("{prefix}.ffn_gate.weight"), dev)?)?;
                let up = gguf_qmm(ct.tensor(&format!("{prefix}.ffn_up.weight"), dev)?)?;
                let down = gguf_qmm(ct.tensor(&format!("{prefix}.ffn_down.weight"), dev)?)?;
                MoeOrMlp::Mlp(DenseMlp { gate, up, down })
            };

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
            tok_embeddings: Embedding::new(tok_embeddings, props.embedding_length),
            layers,
            layer_types,
            norm,
            output: gguf_qmm(output)?,
            rotary: default_rotary,
            device: device.clone(),
            cache: EitherCache::Hybrid(pipeline_cache),
            max_seq_len: props.max_seq_len,
            mapper: Some(mapper),
            dtype,
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
        let (b_sz, seq_len) = input_ids.dims2()?;
        let mut x = self.tok_embeddings.forward(input_ids)?;

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

        // With PagedAttention the running context lives in the paged KV pool, so the past-kv length
        // comes from the host `seqlen_offsets`, not the (unused) hybrid attention KvCache; decode
        // (non-first chunk) needs no mask, as paged attention enforces causality via context_lens.
        // Without paging (CPU/eager) fall back to the hybrid cache. Mirrors quantized_qwen3_moe.
        let mask = CausalMasker.make_causal_mask(
            input_ids,
            match metadata.as_ref() {
                Some(_) => &seqlen_offsets as &dyn PastKvLenCache,
                None => &*hybrid_cache as &dyn PastKvLenCache,
            },
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

        // Text-only 3D mRoPE position ids: all three rows equal the linear position per sequence.
        // The decode-graph path threads a STABLE device `rope_positions` buffer (refreshed in place
        // by the graph runner) so the captured cos/sin advance with the replayed token instead of
        // freezing at the warmup position; the eager path synthesizes them from `seqlen_offsets`.
        let rope_positions = metadata
            .as_ref()
            .and_then(|(_, meta)| meta.rope_positions.as_ref())
            .and_then(|rp| rp.get(&self.device.location()));
        let cos_sin = self.compute_text_mrope(seqlen_offsets, seq_len, x.dtype(), rope_positions)?;

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
                    attn.forward(&normed, &mask.get(normed.device()), &cos_sin, kv_cache, paged)?
                }
                LayerImpl::LinearAttention(gdn) => {
                    let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer_idx)
                    else {
                        hanzo_ml::bail!("Hybrid cache layer {layer_idx} not recurrent.");
                    };
                    if b_sz == 1 && seq_len == 1 {
                        // Single-sequence decode: gather/scatter the recurrent + conv state via
                        // constant-offset `narrow`/`slice_set` using the HOST slot, so there is no
                        // `to_vec1` device->host sync and the pool buffers (stable VRAM) are updated
                        // in place. This is exactly what makes the GDN decode step capturable by a
                        // CUDA graph: the baked slot offset is constant for the sequence's lifetime,
                        // and the recurrence evolves the live pool state across serialized replays.
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
        let x = self.norm.forward(&x)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.forward(&x.contiguous()?)
    }

    /// Build text-only mRoPE cos/sin. position_ids shape (3, batch, seq) with all three temporal/
    /// height/width rows equal to the absolute token position; this collapses interleaved mRoPE to
    /// plain partial RoPE, which is correct for text-only generation.
    fn compute_text_mrope(
        &self,
        seqlen_offsets: &[usize],
        seq_len: usize,
        dtype: DType,
        rope_positions: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // (3, batch, seq) position ids -> interleaved mRoPE collapses to partial RoPE for text.
        let position_ids = match rope_positions {
            // Decode-graph path: read the advancing position from the stable device buffer the graph
            // runner refreshes in place (no host Tensor::from_vec, so the captured cos/sin advance).
            Some(rp) if seq_len == 1 => {
                let batch = rp.dim(0)?;
                let pos = rp.reshape((1, batch, 1))?;
                Tensor::cat(&[&pos, &pos, &pos], 0)?
            }
            _ => {
                let batch = seqlen_offsets.len().max(1);
                let mut positions = Vec::with_capacity(batch * seq_len);
                for &off in seqlen_offsets.iter() {
                    for p in 0..seq_len {
                        positions.push((off + p) as u32);
                    }
                }
                if seqlen_offsets.is_empty() {
                    for p in 0..seq_len {
                        positions.push(p as u32);
                    }
                }
                let pos_1d = Tensor::from_vec(positions, (batch, seq_len), &self.device)?;
                Tensor::stack(&[&pos_1d, &pos_1d, &pos_1d], 0)?
            }
        };
        self.rotary.compute_cos_sin(&position_ids, dtype)
    }
}
