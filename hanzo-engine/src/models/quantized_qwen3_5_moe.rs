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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMaskConfig, CausalMasker, QRmsNorm, Qwen3VLRotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::models::gdn::{
    forward_pooled, l2_norm, sigmoid, softplus, GdnLayerCache, PoolSlots, RmsNormGated,
};
use crate::ops::{TopKLastDimOp, TopKOutput};
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache};
use crate::utils::gguf_metadata::{ContentMetadata, DEFAULT_FULL_ATTENTION_INTERVAL};
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
const DEFAULT_PARTIAL_ROTARY_FACTOR: f64 = 0.25;
const L2_NORM_EPS: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerType {
    FullAttention,
    LinearAttention,
}

// ===================== MoE / dense FFN =====================

/// A router or gate GEMM: GGUF's QMatMul, or a quantization-aware linear.
pub(crate) enum Gemm {
    Gguf(QMatMul),
    Quant(Arc<dyn QuantMethod>),
}

impl Gemm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gguf(m) => m.forward(x),
            Self::Quant(m) => m.forward(x),
        }
    }
}

/// The routed experts' gate, up and down banks.
pub(crate) enum Banks {
    /// GGUF stacked banks for the indexed matmul.
    Gguf([QMatMul; 3]),
    /// Quantized stacked banks (NVFP4 experts), run through `gather_forward`.
    Quant([Arc<dyn QuantMethod>; 3]),
}

pub(crate) struct FusedMoe {
    gate: Gemm,
    banks: Banks,
    shared_gate: Gemm,
    pub(crate) shared_gate_proj: Arc<dyn QuantMethod>,
    pub(crate) shared_up_proj: Arc<dyn QuantMethod>,
    pub(crate) shared_down_proj: Arc<dyn QuantMethod>,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    /// The dtype router logits are rounded to before the f32 softmax: F32 for GGUF, the
    /// activation dtype (bf16) for vLLM's served router GEMM.
    router_dtype: DType,
}

/// `silu(g) · u` in f32 with one rounding to `dtype` (`triton_poi_fused_mul_silu_slice_0`, and
/// FlashInfer's SwiGLU before its fc2 quantizer).
fn swiglu_rounded(g: &Tensor, u: &Tensor, dtype: DType) -> Result<Tensor> {
    let g = g.to_dtype(DType::F32)?;
    let u = u.to_dtype(DType::F32)?;
    ((g.clone() / (g.neg()?.exp()? + 1.0)?)? * u)?.to_dtype(dtype)
}

impl FusedMoe {
    /// Load the shared sparse-MoE block (routed experts + sigmoid-gated shared expert) from GGUF.
    /// Identical tensor layout for Qwen3.5-MoE and Qwen3-Next: `ffn_gate_inp`, `ffn_{gate,up,down}_exps`,
    /// and the `*_shexp` shared-expert tensors. `norm_topk_prob` is always true for both.
    pub(crate) fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        dev: &Device,
        num_experts_per_tok: usize,
    ) -> Result<Self> {
        let gate = ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), dev)?;
        let gate_experts = ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), dev)?;
        let up_experts = ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), dev)?;
        let down_experts = ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), dev)?;
        let shared_gate = ct.tensor(&format!("{prefix}.ffn_gate_inp_shexp.weight"), dev)?;
        let shared_gate_proj =
            gguf_qmm(ct.tensor(&format!("{prefix}.ffn_gate_shexp.weight"), dev)?)?;
        let shared_up_proj = gguf_qmm(ct.tensor(&format!("{prefix}.ffn_up_shexp.weight"), dev)?)?;
        let shared_down_proj =
            gguf_qmm(ct.tensor(&format!("{prefix}.ffn_down_shexp.weight"), dev)?)?;
        Ok(Self {
            gate: Gemm::Gguf(QMatMul::from_qtensor(gate)?),
            banks: Banks::Gguf([
                QMatMul::from_qtensor(gate_experts)?,
                QMatMul::from_qtensor(up_experts)?,
                QMatMul::from_qtensor(down_experts)?,
            ]),
            // ffn_gate_inp_shexp is a hidden->1 shared-expert gate stored 1D [hidden]; the
            // matmul needs 2D, so dequantize and reshape to [1, hidden].
            shared_gate: Gemm::Gguf({
                let w = shared_gate.dequantize(dev)?;
                QMatMul::Tensor(if w.rank() == 1 { w.unsqueeze(0)? } else { w })
            }),
            shared_gate_proj,
            shared_up_proj,
            shared_down_proj,
            norm_topk_prob: true,
            num_experts_per_tok,
            router_dtype: DType::F32,
        })
    }

    /// The MoE block at `vb` (`...mlp`) of a safetensors checkpoint: a bf16 router `gate`
    /// `[E, h]`, NVFP4 expert banks (gate and up quantize one input, so they share its
    /// activation scale), the shared expert through `quant` (block FP8), and a bf16
    /// `shared_expert_gate` `[1, h]`.
    pub(crate) fn new(
        vb: &hanzo_quant::ShardedVarBuilder,
        props: &PropsGGUF,
        quant: &Option<hanzo_quant::QuantizedConfig>,
        dtype: DType,
    ) -> Result<Self> {
        let h = props.embedding_length;
        let e = props.num_experts.unwrap_or(0);
        let ff = props.moe_intermediate_size;
        let lin = |i: usize, o: usize, q: &Option<hanzo_quant::QuantizedConfig>, name: &str| {
            hanzo_quant::linear_no_bias(i, o, q, vb.pp(name))
        };
        let experts = vb.pp("experts");
        let bank = |proj: &str, share: &[&str], n: usize, k: usize| -> Result<Arc<dyn QuantMethod>> {
            Ok(Arc::new(hanzo_quant::NVFP4Layer::experts(experts.clone(), proj, share, e, n, k, dtype)?))
        };
        // The shared expert runs inside vLLM's moe_forward_shared custom op, where the
        // standalone QuantFP8 kernel quantizes its inputs (amax div.full 448): a block-FP8
        // projection takes the Eager quantizer, anything else loads through `quant`.
        let shared = |i: usize, o: usize, name: &str| -> Result<Arc<dyn QuantMethod>> {
            let p = vb.pp(name);
            if p.contains_tensor("weight_scale_inv") {
                let w = p.get_with_hints_dtype((o, i), "weight", Default::default(), DType::F8E4M3)?;
                let sc = p.get_with_hints_dtype(
                    (o.div_ceil(128), i.div_ceil(128)),
                    "weight_scale_inv",
                    Default::default(),
                    DType::F32,
                )?;
                hanzo_quant::blockwise_fp8_act(
                    w,
                    sc,
                    vec![128, 128],
                    dtype,
                    hanzo_quant::quantize::Fp8Mode::Eager,
                )
            } else {
                lin(i, o, quant, name)
            }
        };
        // The shared expert's intermediate width is its gate_proj's rows.
        let sff = vb
            .clone()
            .set_device(Device::Cpu)
            .get_unchecked_dtype("shared_expert.gate_proj.weight_scale_inv", DType::F32)
            .map(|s| s.dim(0).map(|b| b * 128))
            .unwrap_or(Ok(ff))?;
        Ok(Self {
            gate: Gemm::Quant(lin(h, e, &None, "gate")?),
            banks: Banks::Quant([
                bank("gate_proj", &["up_proj"], ff, h)?,
                bank("up_proj", &["gate_proj"], ff, h)?,
                bank("down_proj", &[], h, ff)?,
            ]),
            shared_gate: Gemm::Quant(lin(h, 1, &None, "shared_expert_gate")?),
            shared_gate_proj: shared(h, sff, "shared_expert.gate_proj")?,
            shared_up_proj: shared(h, sff, "shared_expert.up_proj")?,
            shared_down_proj: shared(sff, h, "shared_expert.down_proj")?,
            norm_topk_prob: true,
            num_experts_per_tok: props.num_experts_per_tok,
            router_dtype: dtype,
        })
    }

    /// Router logits `[t, E]`, rounded to the router dtype.
    pub(crate) fn logits(&self, xs: &Tensor) -> Result<Tensor> {
        match self.router_dtype {
            DType::F32 => self.gate.forward(&xs.to_dtype(DType::F32)?),
            d => self.gate.forward(&xs.to_dtype(d)?),
        }
    }

    /// `(ids [t, k], weights [t, k] f32)`: softmax in f32 over the logits, the top k by
    /// probability (ties to the lower index), renormalized in f32 (vLLM `topk_softmax`).
    pub(crate) fn route(&self, logits: &Tensor) -> Result<(Tensor, Tensor)> {
        let routing_weights = hanzo_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
        let TopKOutput {
            values: mut scores,
            indices,
        } = routing_weights.topk(self.num_experts_per_tok)?;
        if self.norm_topk_prob {
            scores = scores.broadcast_div(&scores.sum_keepdim(D::Minus1)?)?;
        }
        Ok((indices, scores))
    }

    /// The routed experts over `xs` `[t, h]` for `ids`/`weights` `[t, k]`: `[t, h]` in `xs`'s dtype.
    pub(crate) fn experts(&self, xs: &Tensor, ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let xs_e = xs.reshape((num_tokens, 1, hidden_dim))?;
        let routed = match &self.banks {
            Banks::Gguf([gate, up, down]) => {
                let g = gate.indexed_moe_forward(&xs_e, ids)?;
                let u = up.indexed_moe_forward(&xs_e, ids)?;
                let act = crate::ops::mul_and_act(&g, &u, crate::layers::Activation::Silu)?;
                down.indexed_moe_forward(&act, ids)?
            }
            Banks::Quant([gate, up, down]) => {
                let ids = ids.to_dtype(DType::U32)?;
                let g = gate.gather_forward(&xs_e, &ids)?;
                let u = up.gather_forward(&xs_e, &ids)?;
                let act = swiglu_rounded(&g, &u, xs.dtype())?;
                down.gather_forward(&act, &ids)?
            }
        };
        // Weighted expert-combine: out[t,:] = sum_e scores[t,e] * routed[t,e,:], accumulated in
        // f32 and rounded once. The fused CUDA kernel does it in one coalesced pass.
        #[cfg(feature = "cuda")]
        if routed.device().is_cuda() {
            return crate::cuda::moe::moe_combine_cuda(&routed, weights);
        }
        routed
            .to_dtype(DType::F32)?
            .broadcast_mul(&weights.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .to_dtype(routed.dtype())
    }

    /// The sigmoid-gated shared expert over `xs` `[t, h]`.
    pub(crate) fn shared(&self, xs: &Tensor) -> Result<Tensor> {
        let g = self.shared_gate_proj.forward(xs)?;
        let u = self.shared_up_proj.forward(xs)?;
        if self.router_dtype == DType::F32 {
            // GGUF: matching qwen3_next SparseMoeBlock.
            let act = crate::ops::mul_and_act(&g, &u, crate::layers::Activation::Silu)?;
            let out = self.shared_down_proj.forward(&act)?;
            let gate = sigmoid(&self.shared_gate.forward(&xs.to_dtype(DType::F32)?)?)?
                .to_dtype(out.dtype())?;
            return out.broadcast_mul(&gate);
        }
        // Served: SiluAndMul in f32 with one rounding; `sigmoid(gate)` and its product with the
        // expert each rounded, as the eager module inside moe_forward_shared computes them.
        let dtype = xs.dtype();
        let act = swiglu_rounded(&g.to_dtype(dtype)?, &u.to_dtype(dtype)?, dtype)?;
        let out = self.shared_down_proj.forward(&act)?.to_dtype(dtype)?;
        let gate = sigmoid(&self.shared_gate.forward(xs)?.to_dtype(DType::F32)?)?.to_dtype(dtype)?;
        out.broadcast_mul(&gate)
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let original_dtype = xs.dtype();
        let (ids, weights) = self.route(&self.logits(&xs)?)?;
        let routed = self.experts(&xs, &ids, &weights)?;
        let shared = self.shared(&xs)?;
        let out = if self.router_dtype == DType::F32 {
            (routed + shared)?
        } else {
            // routed + shared in f32, one rounding (the compiled add kernel)
            (routed.to_dtype(DType::F32)? + shared.to_dtype(DType::F32)?)?
        };
        out.reshape((batch, seq_len, hidden_dim))?.to_dtype(original_dtype)
    }
}

pub(crate) struct DenseMlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

impl DenseMlp {
    pub(crate) fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            gate: gguf_qmm(ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?)?,
            up: gguf_qmm(ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?)?,
            down: gguf_qmm(ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(xs)?;
        let up = self.up.forward(xs)?;
        let y = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        self.down.forward(&y)
    }
}

pub(crate) enum MoeOrMlp {
    FusedMoe(Box<FusedMoe>),
    Mlp(DenseMlp),
}

impl MoeOrMlp {
    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs),
        }
    }
}

// ===================== Gated full-attention layer =====================

pub(crate) struct GatedFullAttention {
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
    /// Attend with the QSA kernel's math (qwen4exp) instead of SDPA.
    qsa: bool,
}

impl GatedFullAttention {
    /// The attention at `vb` (`...self_attn`) of a safetensors checkpoint: `q_proj` `[2·hq·d, h]`
    /// with the per-head `[q | gate]` layout GGUF `attn_q` uses, `k_proj`/`v_proj` `[hkv·d, h]`,
    /// `o_proj` `[h, hq·d]`, all through `quant` (block FP8 in a ModelOpt checkpoint), and the
    /// Gemma q/k norms as `1 + w` in f32. Attends with the QSA kernel's math.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        vb: &hanzo_quant::ShardedVarBuilder,
        props: &PropsGGUF,
        quant: &Option<hanzo_quant::QuantizedConfig>,
        rotary: Arc<Qwen3VLRotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (h, hq, hkv, d) = (
            props.embedding_length,
            props.head_count,
            props.head_count_kv,
            props.head_dim,
        );
        let lin = |i: usize, o: usize, name: &str| {
            hanzo_quant::linear_no_bias(i, o, quant, vb.pp(name))
        };
        let norm = |name: &str| -> Result<QRmsNorm> {
            let w = vb.get_with_hints_dtype((d,), &format!("{name}.weight"), Default::default(), DType::F32)?;
            Ok(QRmsNorm::from_weight((w.to_device(device)? + 1.0)?, props.rms_norm_eps as f64))
        };
        Ok(Self {
            attn_q: lin(h, 2 * hq * d, "q_proj")?,
            attn_k: lin(h, hkv * d, "k_proj")?,
            attn_v: lin(h, hkv * d, "v_proj")?,
            attn_o: lin(hq * d, h, "o_proj")?,
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            n_head: hq,
            n_kv_head: hkv,
            head_dim: d,
            rotary,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: hq / hkv,
                softcap: None,
                softmax_scale: 1.0 / (d as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
            dtype,
            qsa: true,
        })
    }

    /// Attend with the QSA kernel's math (qwen4exp's attention in both formats).
    pub(crate) fn qsa(self) -> Self {
        Self { qsa: true, ..self }
    }

    /// The attention of one full-attention block at `prefix`. `paged_attn` is `None` for a block
    /// that keeps its own `KvCache` instead of a slot in the paged cache.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        props: &PropsGGUF,
        rotary: Arc<Qwen3VLRotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            attn_q: gguf_qmm(ct.tensor(&format!("{prefix}.attn_q.weight"), device)?)?,
            attn_k: gguf_qmm(ct.tensor(&format!("{prefix}.attn_k.weight"), device)?)?,
            attn_v: gguf_qmm(ct.tensor(&format!("{prefix}.attn_v.weight"), device)?)?,
            attn_o: gguf_qmm(ct.tensor(&format!("{prefix}.attn_output.weight"), device)?)?,
            q_norm: QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
                props.rms_norm_eps,
            )?,
            k_norm: QRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?,
                props.rms_norm_eps,
            )?,
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
            qsa: false,
        })
    }

    /// This block's MRoPE cos/sin for `[3, batch, seq]` position ids.
    pub(crate) fn rotary_cos_sin(
        &self,
        positions: &Tensor,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        self.rotary.compute_cos_sin(positions, dtype)
    }

    /// Project, attend, then gate and project out: vLLM `Qwen3NextAttention.forward`
    /// (`qwen3_next.py:446-457`).
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: &mut KvCache,
        paged: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (q, k, v, gate) = self.project(x, cos_sin)?;
        let y = self.attend(&q, &k, &v, mask, kv_cache, paged)?;
        self.output(&y, &gate, x.dtype())
    }

    /// `(q, k, v, gate)`: q, k and v as `[batch, heads, seq, head_dim]` in the compute dtype, with
    /// q and k normed and roped, and the pre-sigmoid output gate as `[batch, seq, heads·head_dim]`.
    /// vLLM `_project_qkv_gate` (`qwen3_next.py:384-444`).
    pub(crate) fn project(
        &self,
        x: &Tensor,
        cos_sin: &(Tensor, Tensor),
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
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

        // qk RMSNorm then partial NeoX RoPE with the served kernel's roundings.
        let (q, k) = self.rotary.forward_qk_norm_served(
            cos_sin,
            &q.contiguous()?,
            &k.contiguous()?,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
        )?;

        Ok((
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
            gate,
        ))
    }

    /// `project`'s q attended over this block's cache once k and v are written to it, as
    /// `[batch, seq, heads·head_dim]`. vLLM `self.attn(q, k, v)` (`qwen3_next.py:453`).
    pub(crate) fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &AttentionMask,
        kv_cache: &mut KvCache,
        paged: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, _, seq_len, _) = q.dims4()?;

        // PagedAttention keeps the decode KV write/read position-invariant (device slot_mappings +
        // bucketed context_lens), which is what makes the whole decode forward capturable by a CUDA
        // graph. The plain hybrid `KvCache::append` path (host-offset write, frozen under capture) is
        // kept only for the eager / no-paged (CPU) case. Mirrors quantized_qwen3_moe / qwen3_next.
        let y = match (&self.paged_attn, paged) {
            (Some(paged_attn), Some(((key_cache, value_cache), input_metadata))) => paged_attn
                .forward(
                    q,
                    k,
                    v,
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
                    q,
                    k,
                    v,
                    mask,
                    None,
                    None,
                    &input_metadata,
                    &self.sdpa_params,
                    None,
                )?
            }
            (None, _) if self.qsa => {
                let (k, v) = kv_cache.append(k, v)?;
                // [b, hq, s, d] -> [b, s, hq·d] below, as SDPA's output
                crate::models::qsa::attend(q, &k, &v)?.transpose(1, 2)?
            }
            (None, _) => {
                let (k, v) = kv_cache.append(k, v)?;
                Sdpa.run_attention(q, &k, &v, mask, None, &self.sdpa_params)?
            }
        };
        if self.qsa && self.paged_attn.is_none() {
            return y.contiguous()?.reshape((b_sz, seq_len, ()));
        }

        if mask.is_custom() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))
        } else {
            y.reshape((b_sz, seq_len, ()))
        }
    }

    /// `o_proj(y * sigmoid(gate))` in `dtype`, the residual stream's. The product is formed in f32
    /// and handed over unrounded: the served graph fuses it into o_proj's activation quantizer
    /// (`…mul_permute_sigmoid…`), which reads it from registers. vLLM `qwen3_next.py:454-456`.
    pub(crate) fn output(&self, y: &Tensor, gate: &Tensor, dtype: DType) -> Result<Tensor> {
        let gate = sigmoid(&gate.to_dtype(DType::F32)?)?;
        let y = y.to_dtype(DType::F32)?.broadcast_mul(&gate)?;
        self.attn_o.forward(&y)?.to_dtype(dtype)
    }
}

// ===================== Gated DeltaNet (linear-attention) layer =====================

pub(crate) struct QGatedDeltaNet {
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

/// Rows produced by the conv state spliced onto the left of a continuation are context, not output.
fn trim_carried(out: &Tensor, carried: usize) -> Result<Tensor> {
    if carried == 0 {
        return Ok(out.clone());
    }
    let len = out.dim(1)?;
    out.narrow(1, carried, len - carried)
}

impl QGatedDeltaNet {
    /// The GDN block at `prefix`. Its output norm gates with silu(z), vLLM's default
    /// `output_gate_type` (`qwen_gdn_linear_attn.py:471`).
    pub(crate) fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        props: &PropsGGUF,
        dev: &Device,
    ) -> Result<Self> {
        let in_proj_qkv = gguf_qmm(ct.tensor(&format!("{prefix}.attn_qkv.weight"), dev)?)?;
        let in_proj_z = gguf_qmm(ct.tensor(&format!("{prefix}.attn_gate.weight"), dev)?)?;
        let in_proj_b = gguf_qmm(ct.tensor(&format!("{prefix}.ssm_beta.weight"), dev)?)?;
        let in_proj_a = gguf_qmm(ct.tensor(&format!("{prefix}.ssm_alpha.weight"), dev)?)?;
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

        Ok(Self {
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
            key_dim: props.num_k_heads * props.head_k_dim,
            value_dim: props.num_v_heads * props.head_v_dim,
        })
    }

    /// The GDN block at `vb` (`...linear_attn`) of a safetensors checkpoint, in the tiled V-head
    /// order the recurrence consumes.
    ///
    /// The checkpoint keeps V heads grouped by key head (`[K0_v0..v{g-1}, K1_v0.., ...]`); tiled
    /// order puts tiled head `r·K + k` where grouped head `k·g + r` was (π(r·K+k) = k·g+r, K key
    /// heads, g V heads per key head), so V head j reads key head j % K as the q/k tiling does.
    /// The permutation moves whole 128-row heads, which are whole FP8 blocks: it applies to the V
    /// rows of `in_proj_qkv` (rows ≥ 2·key_dim) and all rows of `in_proj_z` with their
    /// `weight_scale_inv` rows, to `out_proj`'s columns and scale columns, to `in_proj_a`/`b`
    /// rows, the conv's V channels, `A_log` and `dt_bias`. `norm` is used as-is (no `+1`).
    pub(crate) fn new(
        vb: &hanzo_quant::ShardedVarBuilder,
        props: &PropsGGUF,
        quant: &Option<hanzo_quant::QuantizedConfig>,
    ) -> Result<Self> {
        let dev = vb.device().clone();
        let host = vb.clone().set_device(Device::Cpu);
        let (kh, vh, dk, dv) = (
            props.num_k_heads,
            props.num_v_heads,
            props.head_k_dim,
            props.head_v_dim,
        );
        let (key_dim, value_dim, h) = (kh * dk, vh * dv, props.embedding_length);
        let g = vh / kh;
        // tiled head p reads grouped head perm[p]
        let perm: Vec<usize> = (0..vh).map(|p| (p % kh) * g + p / kh).collect();
        let heads = |t: &Tensor, dim: usize, start: usize, width: usize| -> Result<Tensor> {
            let parts = perm
                .iter()
                .map(|&j| t.narrow(dim, start + j * width, width))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&parts, dim)
        };
        let rows = |name: &str| host.get_unchecked_dtype(name, DType::F32);
        // A projection with its V heads moved to tiled order: rows (dim 0) from `start` for the
        // input projections, columns (dim 1) for out_proj. Block FP8 when the checkpoint carries
        // `weight_scale_inv` (one scale row or column per 128-wide head), else a float linear.
        let proj = |name: &str, dim: usize, start: usize| -> Result<Arc<dyn QuantMethod>> {
            let permute = |t: &Tensor, width: usize, start: usize| -> Result<Tensor> {
                let len = t.dim(dim)?;
                let lead = t.narrow(dim, 0, start)?;
                let moved = heads(t, dim, start, width)?;
                if start == 0 {
                    Ok(moved)
                } else {
                    debug_assert_eq!(start + vh * width, len);
                    Tensor::cat(&[lead, moved], dim)
                }
            };
            if host.contains_tensor(&format!("{name}.weight_scale_inv")) {
                if dv != 128 || start % 128 != 0 {
                    hanzo_ml::bail!("{name}: tiling block-FP8 heads needs 128-wide V heads");
                }
                let w = host.get_unchecked_dtype(&format!("{name}.weight"), DType::F8E4M3)?;
                let sc = rows(&format!("{name}.weight_scale_inv"))?;
                hanzo_quant::blockwise_fp8_moe(
                    permute(&w, dv, start)?.to_device(&dev)?,
                    permute(&sc, 1, start / 128)?.to_device(&dev)?,
                    vec![128, 128],
                    DType::BF16,
                )
            } else {
                let w = host.get_unchecked_dtype(&format!("{name}.weight"), vb.dtype())?;
                let w = permute(&w, dv, start)?.to_device(&dev)?;
                Ok(Arc::new(hanzo_quant::UnquantLinear::new(QuantMethodConfig::Unquantized(
                    hanzo_nn::Linear::new(w, None),
                ))?))
            }
        };
        let _ = (quant, h);
        let (in_proj_qkv, in_proj_z, out_proj) = (
            proj("in_proj_qkv", 0, 2 * key_dim)?,
            proj("in_proj_z", 0, 0)?,
            proj("out_proj", 1, 0)?,
        );
        // in_proj_a / in_proj_b run on the f32-lifted input: f32 weights holding the bf16 values
        let f32_lin = |name: &str| -> Result<Arc<dyn QuantMethod>> {
            let w = heads(&rows(&format!("{name}.weight"))?, 0, 0, 1)?.to_device(&dev)?;
            Ok(Arc::new(hanzo_quant::UnquantLinear::new(QuantMethodConfig::Unquantized(
                hanzo_nn::Linear::new(w, None),
            ))?))
        };
        let conv = rows("conv1d.weight")?.squeeze(1)?;
        let conv = Tensor::cat(&[conv.narrow(0, 0, 2 * key_dim)?, heads(&conv, 0, 2 * key_dim, dv)?], 0)?;
        let a_log = heads(&rows("A_log")?, 0, 0, 1)?;
        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b: f32_lin("in_proj_b")?,
            in_proj_a: f32_lin("in_proj_a")?,
            conv1d_weight: conv.to_device(&dev)?,
            dt_bias: heads(&rows("dt_bias")?, 0, 0, 1)?.to_device(&dev)?,
            a: a_log.exp()?.neg()?.to_device(&dev)?,
            norm: RmsNormGated::from_weight(rows("norm.weight")?.to_device(&dev)?, props.rms_norm_eps as f64)
                .sigmoid(),
            out_proj,
            num_k_heads: kh,
            num_v_heads: vh,
            head_k_dim: dk,
            head_v_dim: dv,
            conv_kernel_size: props.conv_kernel,
            key_dim,
            value_dim,
        })
    }

    /// Gate the output norm with sigmoid(z), `output_gate_type = "sigmoid"`
    /// (vLLM `qwen_gdn_linear_attn.py:471-484`).
    pub(crate) fn sigmoid(self) -> Self {
        Self {
            norm: self.norm.sigmoid(),
            ..self
        }
    }

    /// The block over `x` `[b, s, h]`. Arithmetic is f32, rounded to `x`'s dtype at the six points
    /// vLLM's prefill path stores (`qwen_gdn_linear_attn.py:889-965, 1259-1545`,
    /// `fused_gdn_prefill_post_conv.py:100-215`): the projection outputs (qkv and z, then b and a),
    /// the causal conv + silu, the l2-normed q and k (and v), the recurrence output, and the gated
    /// norm. g, beta and the state stay f32. With f32 input every rounding is a no-op.
    pub(crate) fn forward(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        self.forward_probed(x, cache, None)
    }

    /// [`Self::forward`], noting each stored stage in `probe` when given (for the goldens).
    pub(crate) fn forward_probed(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        mut probe: Option<&mut Vec<(&'static str, Tensor)>>,
    ) -> Result<Tensor> {
        let mut note = |name: &'static str, t: &Tensor| {
            if let Some(p) = probe.as_mut() {
                p.push((name, t.clone()));
            }
        };
        let rd = x.dtype();
        let store = |t: Tensor| -> Result<Tensor> { t.to_dtype(rd)?.to_dtype(DType::F32) };
        let x = &x.to_dtype(DType::F32)?;
        let (batch_size, seq_len, _hidden) = x.dims3()?;
        let v_per_group = self.num_v_heads / self.num_k_heads;

        // 1-2. Projections: qkv is merged [q | k | v]; z is the gate; b and a are bf16 GEMMs.
        let mixed_qkv = store(self.in_proj_qkv.forward(x)?)?;
        let z = store(self.in_proj_z.forward(x)?)?;
        let b = store(self.in_proj_b.forward(x)?)?;
        let a = store(self.in_proj_a.forward(x)?)?;

        let z = z.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let b = b.reshape((batch_size, seq_len, self.num_v_heads))?;
        let a = a.reshape((batch_size, seq_len, self.num_v_heads))?;

        cache.trail_conv(&mixed_qkv)?;

        // 3. Causal conv1d over the concatenated qkv, then silu.
        let mixed_qkv = if rd != DType::F32 {
            self.causal_conv1d_served(&mixed_qkv, cache, rd)?
        } else if cache.seqlen_offset > 0 && seq_len == 1 {
            self.causal_conv1d_update(&mixed_qkv, cache)?
        } else {
            self.causal_conv1d_full(&mixed_qkv, cache)?
        };
        let mixed_qkv = store(mixed_qkv)?;
        note("conv", &mixed_qkv);

        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;
        let q = q.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // 4. beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias), softplus as the served
        //    kernel forms it (identity past 20).
        let beta = sigmoid(&b)?;
        let dt_bias = self.dt_bias.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?;
        let xa = a.broadcast_add(&dt_bias)?;
        let sp = {
            let pos = (xa.clone() + (xa.neg()?.exp()? + 1.0)?.log()?)?;
            let neg = (xa.exp()? + 1.0)?.log()?;
            let zero = xa.zeros_like()?;
            let sp = xa.gt(&zero)?.where_cond(&pos, &neg)?;
            xa.le(20.0)?.where_cond(&sp, &xa)?
        };
        let g = self
            .a
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_mul(&sp)?;
        note("g", &g);
        note("beta", &beta);

        // 5. Tile q, k to the V-head count: V head j reads key head j % num_k_heads.
        let (q, k) = if v_per_group > 1 {
            let tile = |t: &Tensor| {
                t.unsqueeze(2)?
                    .repeat((1, 1, v_per_group, 1, 1))?
                    .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))
            };
            (tile(&q)?, tile(&k)?)
        } else {
            (q, k)
        };

        // 6. L2-normalize q and k, one rounding.
        let q = store(l2_norm(&q, L2_NORM_EPS)?)?;
        let k = store(l2_norm(&k, L2_NORM_EPS)?)?;
        note("q", &q);
        note("k", &k);
        note("v", &v);

        // 7. Recurrent gated delta rule in f32; its output is stored.
        let y = store(cache.recurrence(&q, &k, &v, &g, &beta)?)?;
        note("core", &y);
        cache.seqlen_offset += seq_len;

        // 8. Gated RMSNorm with z (stored), then the output projection.
        let z_shape = z.shape().clone();
        let y = y.reshape(((), self.head_v_dim))?;
        let z = z.reshape(((), self.head_v_dim))?;
        let y = store(self.norm.forward(&y, &z)?)?;
        let y = y.reshape(z_shape)?.reshape((batch_size, seq_len, self.value_dim))?;
        note("normed", &y);

        self.out_proj.forward(&y)?.to_dtype(rd)
    }

    /// vLLM's causal conv (`causal_conv1d.py:398-478`): each tap's product rounded to `rd` (the
    /// kernel multiplies two bf16 tensors), accumulated in f32 over the four taps in order, then
    /// silu in f32. The conv state (the last `kernel` inputs) advances as in the other paths.
    fn causal_conv1d_served(&self, x: &Tensor, cache: &mut GdnLayerCache, rd: DType) -> Result<Tensor> {
        let (_, seq_len, _) = x.dims3()?;
        let kw = self.conv_kernel_size;
        let x_t = x.transpose(1, 2)?.contiguous()?;
        let left = if cache.seqlen_offset > 0 {
            cache.conv_state.narrow(D::Minus1, 1, kw - 1)?.to_dtype(DType::F32)?
        } else {
            let (b, c, _) = cache.conv_state.dims3()?;
            Tensor::zeros((b, c, kw - 1), DType::F32, x.device())?
        };
        let hist = Tensor::cat(&[&left, &x_t], D::Minus1)?.contiguous()?;
        let total = hist.dim(D::Minus1)?;
        cache.conv_state = hist.narrow(D::Minus1, total - kw, kw)?.to_dtype(cache.conv_state.dtype())?;
        let weight = self.conv1d_weight.to_dtype(DType::F32)?;
        let mut acc: Option<Tensor> = None;
        for j in 0..kw {
            let tap = hist
                .narrow(D::Minus1, j, seq_len)?
                .broadcast_mul(&weight.narrow(1, j, 1)?.unsqueeze(0)?)?
                .to_dtype(rd)?
                .to_dtype(DType::F32)?;
            acc = Some(match acc {
                None => tap,
                Some(a) => (a + tap)?,
            });
        }
        let acc = acc.expect("kernel width >= 1");
        let out = (acc.clone() / (acc.neg()?.exp()? + 1.0)?)?;
        out.transpose(1, 2)
    }

    fn causal_conv1d_update(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (_batch, seq_len, _conv_dim) = x.dims3()?;

        // Vulkan conv1d single-step kernel (gdn_conv1d_step_vulkan) isn't ported to canonical
        // hanzo-ml yet; the portable hanzo-ml path below runs correctly on the Vulkan device.

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
            let out = window
                .broadcast_mul(&weight.unsqueeze(0)?)?
                .sum(D::Minus1)?;
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
    #[allow(dead_code)]
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
        // The full kernel has no conv_state argument and zero-pads its left edge, which is right
        // only at the start of a sequence. A continuation (a later prefill chunk, or a speculative
        // replay) must see the previous tokens, so splice them on and drop their outputs after.
        let carried = if cache.seqlen_offset > 0 {
            self.conv_kernel_size - 1
        } else {
            0
        };
        let x_t = if carried > 0 {
            let left = cache
                .conv_state
                .narrow(D::Minus1, 1, carried)?
                .to_dtype(x.dtype())?;
            Tensor::cat(&[&left, &x.transpose(1, 2)?], D::Minus1)?.contiguous()?
        } else {
            x.transpose(1, 2)?.contiguous()?
        };
        let seq_len = seq_len + carried;

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
            return trim_carried(&output.transpose(1, 2)?, carried);
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
            let out = window
                .broadcast_mul(&weight.unsqueeze(0)?)?
                .sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        let out = Tensor::stack(&conv_outputs, 2)?;
        let out = hanzo_nn::ops::silu(&out)?;
        trim_carried(&out.transpose(1, 2)?, carried)
    }
}

// ===================== Decoder layer =====================

pub(crate) enum LayerImpl {
    FullAttention(GatedFullAttention),
    LinearAttention(QGatedDeltaNet),
}

pub(crate) struct DecoderLayer {
    pub(crate) layer_impl: LayerImpl,
    pub(crate) input_layernorm: QRmsNorm,
    pub(crate) post_attention_layernorm: QRmsNorm,
    pub(crate) mlp: MoeOrMlp,
}

impl DecoderLayer {
    /// This block's MRoPE cos/sin for `[3, batch, seq]` position ids.
    pub(crate) fn rotary_cos_sin(
        &self,
        positions: &Tensor,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let LayerImpl::FullAttention(attn) = &self.layer_impl else {
            hanzo_ml::bail!("expected a full-attention block");
        };
        attn.rotary_cos_sin(positions, dtype)
    }

    /// One full-attention block over its own `KvCache`: attention, then the MLP, both residual.
    pub(crate) fn forward_attention(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let LayerImpl::FullAttention(attn) = &self.layer_impl else {
            hanzo_ml::bail!("expected a full-attention block");
        };
        let residual = x.clone();
        let normed = self.input_layernorm.forward(x)?;
        let x = (attn.forward(&normed, mask, cos_sin, kv_cache, None)? + residual)?;
        let residual = x.clone();
        let normed = self.post_attention_layernorm.forward(&x)?;
        self.mlp.forward(&normed)? + residual
    }
}

// ===================== Config extraction =====================

#[allow(dead_code)]
pub(crate) struct PropsGGUF {
    pub(crate) head_count: usize,
    pub(crate) head_count_kv: usize,
    pub(crate) block_count: usize,
    pub(crate) embedding_length: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) max_seq_len: usize,
    pub(crate) rope_freq_base: f32,
    pub(crate) head_dim: usize,
    pub(crate) rot_dim: usize,
    pub(crate) mrope_section: Vec<usize>,
    pub(crate) full_attention_interval: usize,
    // GDN
    pub(crate) conv_kernel: usize,
    pub(crate) head_k_dim: usize,
    pub(crate) head_v_dim: usize,
    pub(crate) num_k_heads: usize,
    pub(crate) num_v_heads: usize,
    // MoE (None for dense)
    pub(crate) num_experts: Option<usize>,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) is_moe: bool,
    /// Trailing multi-token-prediction blocks, excluded from `block_count`; the first sits at
    /// `blk.{block_count}`.
    pub(crate) nextn_predict_layers: usize,
}

/// The file's `general.architecture`, which must be one of `allowed`.
pub(crate) fn verify_arch(
    metadata: &HashMap<String, hanzo_ml::quantized::gguf_file::Value>,
    allowed: &[&str],
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    if !allowed.contains(&actual_arch.as_str()) {
        let expected = allowed
            .iter()
            .map(|arch| format!("`{arch}`"))
            .collect::<Vec<_>>()
            .join("/");
        hanzo_ml::bail!("Expected {expected} architecture, got `{actual_arch}`.");
    }
    Ok(actual_arch)
}

impl PropsGGUF {
    pub(crate) fn try_from(c: &ContentMetadata, is_moe: bool) -> Result<Self> {
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

        // An INT32 array in the file (the converter writes Python ints).
        let mrope_section = c
            .get_value::<Vec<i32>>("rope.dimension_sections")
            .ok()
            .map(|v| v.into_iter().map(|x| x.max(0) as usize).collect::<Vec<_>>())
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
            // block_count includes the trailing multi-token-prediction blocks, so the transformer
            // depth is block_count - nextn_predict_layers; the head loads from `blk.{depth}` when
            // `--mtp-model` asks for it.
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
                .filter(|i| *i > 0)
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
            nextn_predict_layers: c.get_value::<u32>("nextn_predict_layers").unwrap_or(0) as usize,
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
    props: PropsGGUF,
    /// Hidden-state capture for a parallel-block draft (DFlash). Off until a draft names layers.
    pub(crate) spec_capture: crate::speculative::HiddenPrefixCapture,
    /// What the MTP head reads from the last forward. Off by default.
    last_spec: Mutex<Option<SpecRows>>,
    store_spec: AtomicBool,
    /// The MRoPE positions of the target rows the MTP head drafts from, handed over as those
    /// rows are selected.
    mtp_anchors: crate::models::qwen3_5_mtp::AnchorPositions,
}

/// The rows the last forward took its logits from: their final-norm hidden state,
/// `[batch, rows, hidden]`, and the absolute position of each.
pub(crate) struct SpecRows {
    pub(crate) hidden: Tensor,
    pub(crate) positions: Vec<Vec<usize>>,
}

pub(crate) fn gguf_qmm(q: hanzo_ml::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
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
        let actual_arch = verify_arch(meta, &["qwen35moe", "qwen35"])?;
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
                    let paged_attn = match attention_mechanism {
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(props.head_dim, dev, None)?)
                        }
                        AttentionImplementation::Eager => None,
                    };
                    LayerImpl::FullAttention(GatedFullAttention::load(
                        &mut ct, &prefix, &props, rotary, paged_attn, dev, dtype,
                    )?)
                }
                LayerType::LinearAttention => {
                    LayerImpl::LinearAttention(QGatedDeltaNet::load(&mut ct, &prefix, &props, dev)?)
                }
            };

            let mlp = if is_moe {
                MoeOrMlp::FusedMoe(Box::new(FusedMoe::from_gguf(
                    &mut ct,
                    &prefix,
                    dev,
                    props.num_experts_per_tok,
                )?))
            } else {
                MoeOrMlp::Mlp(DenseMlp::load(&mut ct, &prefix, dev)?)
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
        let hybrid_cache_config = HybridCacheConfig::uniform(
            pipeline_layer_types,
            props.max_seq_len,
            RecurrentLayerConfig {
                conv_dim,
                conv_width: props.conv_kernel,
                state_dims: vec![props.num_v_heads, props.head_k_dim, props.head_v_dim],
                conv_dtype: dtype,
                state_dtype: dtype,
            },
        );
        let pipeline_cache = Arc::new(Mutex::new(
            HybridCache::new(hybrid_cache_config, device)
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
            props,
            spec_capture: crate::speculative::HiddenPrefixCapture::default(),
            last_spec: Mutex::new(None),
            store_spec: AtomicBool::new(false),
            mtp_anchors: Arc::new(Mutex::new(None)),
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
        let trail = hybrid_cache.records_trail(x.dim(1)?);
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

        // Text-only 3D mRoPE position ids: all three rows equal the linear position per sequence.
        // The decode-graph path threads a STABLE device `rope_positions` buffer (refreshed in place
        // by the graph runner) so the captured cos/sin advance with the replayed token instead of
        // freezing at the warmup position; the eager path synthesizes them from `seqlen_offsets`.
        let rope_positions = metadata
            .as_ref()
            .and_then(|(_, meta)| meta.rope_positions.as_ref())
            .and_then(|rp| rp.get(&self.device.location()));
        let cos_sin = text_mrope(
            &self.rotary,
            &self.device,
            seqlen_offsets,
            seq_len,
            x.dtype(),
            rope_positions,
        )?;

        let capture_layers = self.spec_capture.layers_for(b_sz);
        let mut captured: Vec<Tensor> = Vec::with_capacity(capture_layers.len());
        // The paged cache holds one K/V pair per attention layer, so an attention layer reads it
        // at its ordinal among attention layers, not at its decoder index.
        let mut kv_layer = 0;
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
                        .map(|(kv_cache, meta)| (kv_cache[kv_layer].clone(), *meta));
                    kv_layer += 1;
                    let Some(HybridLayerCache::Attention(kv_cache)) =
                        hybrid_cache.get_mut(layer_idx)
                    else {
                        hanzo_ml::bail!("Hybrid cache layer {layer_idx} not attention.");
                    };
                    attn.forward(
                        &normed,
                        &mask.get(normed.device()),
                        &cos_sin,
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
                        // Single sequence: one slot, so gather/scatter is constant-offset
                        // `narrow`/`slice_set` on the HOST slot with no `to_vec1` sync. That sync-free
                        // form is what makes the decode step CUDA-graph capturable (constant baked slot,
                        // recurrence evolves the live pool state in place across replays). b_sz>1 gathers
                        // below.
                        let slot = state_indices_host
                            .as_ref()
                            .and_then(|s| s.first().copied())
                            .ok_or_else(|| {
                                hanzo_ml::Error::msg("missing host recurrent state index")
                            })? as usize;
                        let slots = PoolSlots::One {
                            slot,
                            offset: seqlen_offsets.first().copied().unwrap_or(0),
                        };
                        forward_pooled(pool, slots, layer_idx, trail, |cache| {
                            gdn.forward(&normed, cache)
                        })?
                    } else {
                        let indices = state_indices
                            .as_ref()
                            .expect("checked above: recurrent indices required");
                        forward_pooled(pool, PoolSlots::Many(indices), layer_idx, trail, |cache| {
                            gdn.forward(&normed, cache)
                        })?
                    }
                }
            };

            let x_mid = (attn_out + residual)?;
            let residual = &x_mid;
            let normed = layer.post_attention_layernorm.forward(&x_mid)?;
            let ffn_out = layer.mlp.forward(&normed)?;
            x = (ffn_out + residual)?;

            // Metal prefill hazard drain. The GDN + generic-MoE prefill path churns many
            // short-lived pooled Metal buffers, and the device buffer pool recycles on
            // `Arc::strong_count == 1` with no GPU-completion check (find_available_buffer): a
            // buffer can be handed to a later op while an earlier, recorded-but-not-yet-executed
            // op still needs to read its previous contents -> corrupted intermediates -> NaN
            // logits (which surfaced downstream as the sampler's bad Metal top-k normalizer). A
            // completion drain per prefill layer forces each layer's ops to finish before its
            // buffers can be recycled. Gated to prefill (seq_len > 1) and Metal only: the
            // single-token decode step is short and race-free, so decode keeps the fast resident
            // MoE path with no per-token drain, and CUDA/other backends are untouched.
            if seq_len > 1 && x.device().is_metal() {
                x.device().synchronize()?;
            }
            if capture_layers.contains(&layer_idx) {
                captured.push(x.clone());
            }
        }
        if !capture_layers.is_empty() {
            let start_pos = seqlen_offsets.first().copied().unwrap_or(0);
            self.spec_capture.fold(start_pos, captured)?;
        }

        let x = x.to_device(&self.device)?;
        let x = self.norm.forward(&x)?;
        let x = extract_logits(&x, context_lens.clone())?;
        if self.store_spec.load(Ordering::Relaxed) {
            // The rows `extract_logits` kept, and where each sits in its sequence, so a row index
            // means the same position in both.
            let positions = context_lens
                .iter()
                .enumerate()
                .map(|(seq, (start, len))| {
                    let offset = seqlen_offsets.get(seq).copied().unwrap_or(0) + start;
                    (offset..offset + len).collect()
                })
                .collect();
            if let Ok(mut slot) = self.last_spec.lock() {
                *slot = Some(SpecRows {
                    hidden: x.clone(),
                    positions,
                });
            }
        }
        self.output.forward(&x.contiguous()?)
    }

    pub(crate) fn mtp_anchors(&self) -> crate::models::qwen3_5_mtp::AnchorPositions {
        self.mtp_anchors.clone()
    }

    pub(crate) fn props(&self) -> &PropsGGUF {
        &self.props
    }

    pub(crate) fn rotary(&self) -> Arc<Qwen3VLRotaryEmbedding> {
        self.rotary.clone()
    }

    pub(crate) fn compute_dtype(&self) -> DType {
        self.dtype
    }

    /// Stash the final-norm hidden state and positions of each forward for the MTP head.
    pub(crate) fn set_store_spec(&self, store: bool) {
        self.store_spec.store(store, Ordering::Relaxed);
        if !store {
            if let Ok(mut slot) = self.last_spec.lock() {
                *slot = None;
            }
        }
    }

    /// What the last forward stashed, if any.
    pub(crate) fn last_spec(&self) -> Option<(Tensor, Vec<Vec<usize>>)> {
        let slot = self.last_spec.lock().ok()?;
        slot.as_ref()
            .map(|rows| (rows.hidden.clone(), rows.positions.clone()))
    }

    /// The embedding and output head, lent to a draft that carries neither. The head takes the
    /// residual stream's dtype, which is the embedding's.
    pub(crate) fn shared_heads(&self) -> crate::speculative::SpeculativeSharedHeads {
        let embed_tokens = self.tok_embeddings.clone();
        let output = Arc::clone(&self.output);
        let dtype = self.tok_embeddings.embeddings().dtype();
        crate::speculative::SpeculativeSharedHeads {
            embed: Arc::new(move |ids: &Tensor| embed_tokens.forward(ids)),
            lm_head: Arc::new(move |hidden: &Tensor| {
                output.forward(&hidden.to_dtype(dtype)?.contiguous()?)
            }),
        }
    }
}

/// Build text-only mRoPE cos/sin. position_ids shape (3, batch, seq) with all three temporal/
/// height/width rows equal to the absolute token position; this collapses interleaved mRoPE to
/// plain partial RoPE, which is correct for text-only generation.
pub(crate) fn text_mrope(
    rotary: &Qwen3VLRotaryEmbedding,
    device: &Device,
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
            let pos_1d = Tensor::from_vec(positions, (batch, seq_len), device)?;
            Tensor::stack(&[&pos_1d, &pos_1d, &pos_1d], 0)?
        }
    };
    rotary.compute_cos_sin(&position_ids, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gdn::{GatedDeltaNet, GdnInProj};
    use hanzo_ml::quantized::{gguf_file, GgmlDType, QTensor};
    use hanzo_nn::Linear;
    use hanzo_quant::UnquantLinear;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    const HIDDEN: usize = 12;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 16;
    const ROT_DIM: usize = 8;
    const THETA: f32 = 10_000.0;
    const EPS: f32 = 1e-6;
    const GDN_HEADS: usize = 2;
    const GDN_DIM: usize = 4;
    const CONV: usize = 4;

    fn props() -> PropsGGUF {
        PropsGGUF {
            head_count: HEADS,
            head_count_kv: KV_HEADS,
            block_count: 4,
            embedding_length: HIDDEN,
            rms_norm_eps: EPS,
            max_seq_len: 64,
            rope_freq_base: THETA,
            head_dim: HEAD_DIM,
            rot_dim: ROT_DIM,
            mrope_section: vec![2, 1, 1],
            full_attention_interval: 4,
            conv_kernel: CONV,
            head_k_dim: GDN_DIM,
            head_v_dim: GDN_DIM,
            num_k_heads: GDN_HEADS,
            num_v_heads: GDN_HEADS,
            num_experts: None,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            is_moe: false,
            nextn_predict_layers: 0,
        }
    }

    /// Seeded tensors in torch layout (`[out, in]`, or `[n]` for a vector), named as in the GGUF.
    fn draw(shapes: &[(&str, &[usize])], seed: u64) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut rng = StdRng::seed_from_u64(seed);
        shapes
            .iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                // Vectors sit around 1, as a trained norm weight does.
                let shift = if shape.len() == 1 { 1.0 } else { 0.0 };
                let data = (0..n)
                    .map(|_| shift + rng.random_range(-0.5f32..0.5))
                    .collect();
                (format!("blk.0.{name}"), shape.to_vec(), data)
            })
            .collect()
    }

    fn open(
        dir: &std::path::Path,
        tensors: &[(String, Vec<usize>, Vec<f32>)],
    ) -> Result<std::fs::File> {
        let path = dir.join("block.gguf");
        let owned = tensors
            .iter()
            .map(|(name, shape, data)| {
                let t = Tensor::from_vec(data.clone(), shape.as_slice(), &Device::Cpu)?;
                Ok((name.as_str(), QTensor::quantize(&t, GgmlDType::F32)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let refs = owned.iter().map(|(n, t)| (*n, t)).collect::<Vec<_>>();
        let arch = gguf_file::Value::String("qwen35".to_string());
        let mut file = std::fs::File::create(&path).map_err(hanzo_ml::Error::msg)?;
        gguf_file::write(&mut file, &[("general.architecture", &arch)], &refs)?;
        std::fs::File::open(&path).map_err(hanzo_ml::Error::msg)
    }

    fn metadata(arch: &str) -> HashMap<String, gguf_file::Value> {
        HashMap::from([(
            "general.architecture".to_string(),
            gguf_file::Value::String(arch.to_string()),
        )])
    }

    #[test]
    fn verify_arch_accepts_only_the_listed() -> Result<()> {
        let allowed = ["qwen35moe", "qwen35"];
        assert_eq!(verify_arch(&metadata("qwen35"), &allowed)?, "qwen35");
        let err = verify_arch(&metadata("qwen4exp"), &allowed)
            .expect_err("qwen4exp is not listed")
            .to_string();
        assert!(
            err.contains("Expected `qwen35moe`/`qwen35` architecture, got `qwen4exp`."),
            "{err}"
        );
        assert_eq!(
            verify_arch(&metadata("qwen4exp"), &["qwen4exp"])?,
            "qwen4exp"
        );
        Ok(())
    }

    /// One position of vLLM `Qwen3NextAttention.forward` in f64: `q`, `k`, `v` and `gate` are
    /// `_project_qkv_gate` (`qwen3_next.py:384-444`), `y` is `self.attn` (`:453`) and `out` is
    /// `o_proj(y * sigmoid(gate))` (`:454-456`).
    struct Row {
        q: Vec<f64>,
        k: Vec<f64>,
        v: Vec<f64>,
        gate: Vec<f64>,
        y: Vec<f64>,
        out: Vec<f64>,
    }

    /// The reference over one sequence at positions `0..`: per-head q/k RMSNorm, NeoX RoPE on the
    /// first `ROT_DIM` dims (text MRoPE has equal planes, so it is plain partial RoPE), and causal
    /// `softmax(q·k/√d)·v` with each KV head serving `HEADS / KV_HEADS` query heads.
    fn attention_reference(x: &[f64], w: &HashMap<String, Vec<f64>>) -> Vec<Row> {
        let linear = |m: &[f64], u: &[f64]| -> Vec<f64> {
            m.chunks(u.len())
                .map(|row| row.iter().zip(u).map(|(a, b)| a * b).sum())
                .collect()
        };
        let norm = |u: &mut [f64], g: &[f64]| {
            let mean = u.iter().map(|a| a * a).sum::<f64>() / u.len() as f64;
            let inv = 1.0 / (mean + f64::from(EPS)).sqrt();
            for (a, g) in u.iter_mut().zip(g) {
                *a *= inv * g;
            }
        };
        let rope = |u: &mut [f64], p: usize| {
            let (lo, hi) = u[..ROT_DIM].split_at_mut(ROT_DIM / 2);
            for (i, (a, c)) in lo.iter_mut().zip(hi).enumerate() {
                let theta = p as f64 * f64::from(THETA).powf(-((2 * i) as f64) / ROT_DIM as f64);
                let (x0, x1) = (*a, *c);
                *a = x0 * theta.cos() - x1 * theta.sin();
                *c = x1 * theta.cos() + x0 * theta.sin();
            }
        };
        let mut rows = x
            .chunks(HIDDEN)
            .enumerate()
            .map(|(p, xt)| {
                // q_proj interleaves `[q_h | gate_h]` per head (`qwen3_next.py:424-432`).
                let qg = linear(&w["blk.0.attn_q.weight"], xt);
                let (mut q, mut gate) = (vec![], vec![]);
                for head in qg.chunks(2 * HEAD_DIM) {
                    let mut qh = head[..HEAD_DIM].to_vec();
                    norm(&mut qh, &w["blk.0.attn_q_norm.weight"]);
                    rope(&mut qh, p);
                    q.extend(qh);
                    gate.extend_from_slice(&head[HEAD_DIM..]);
                }
                let mut k = linear(&w["blk.0.attn_k.weight"], xt);
                for kh in k.chunks_mut(HEAD_DIM) {
                    norm(kh, &w["blk.0.attn_k_norm.weight"]);
                    rope(kh, p);
                }
                let v = linear(&w["blk.0.attn_v.weight"], xt);
                Row {
                    q,
                    k,
                    v,
                    gate,
                    y: vec![],
                    out: vec![],
                }
            })
            .collect::<Vec<_>>();
        for t in 0..rows.len() {
            let mut y = vec![0f64; HEADS * HEAD_DIM];
            for h in 0..HEADS {
                let kv = (h / (HEADS / KV_HEADS)) * HEAD_DIM;
                let qh = &rows[t].q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                let scores = rows[..=t]
                    .iter()
                    .map(|row| {
                        let dot: f64 = qh.iter().zip(&row.k[kv..]).map(|(a, b)| a * b).sum();
                        dot / (HEAD_DIM as f64).sqrt()
                    })
                    .collect::<Vec<_>>();
                let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e = scores.iter().map(|s| (s - top).exp()).collect::<Vec<_>>();
                let z: f64 = e.iter().sum();
                for (row, e) in rows[..=t].iter().zip(&e) {
                    for (yj, vj) in y[h * HEAD_DIM..(h + 1) * HEAD_DIM]
                        .iter_mut()
                        .zip(&row.v[kv..])
                    {
                        *yj += e / z * vj;
                    }
                }
            }
            let gated = y
                .iter()
                .zip(&rows[t].gate)
                .map(|(a, g)| a / (1.0 + (-g).exp()))
                .collect::<Vec<_>>();
            rows[t].out = linear(&w["blk.0.attn_output.weight"], &gated);
            rows[t].y = y;
        }
        rows
    }

    /// `got` against `want` at position `at`, each value relative to its reference or, where that
    /// is below 0.1, to 0.1: the values are sums that can cancel to near zero, where the f32
    /// rounding of their O(1) terms is no longer small relative to the result.
    fn check(stage: &str, at: usize, got: &[f32], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "{stage} at {at}");
        let worst = got
            .iter()
            .zip(want)
            .map(|(g, r)| (f64::from(*g) - r).abs() / r.abs().max(0.1))
            .fold(0f64, f64::max);
        assert!(
            worst < tol,
            "{stage} at {at}: max relative error {worst:.3e}"
        );
    }

    /// Position `i` of head `h` in the values of a `[1, heads, len, HEAD_DIM]` tensor.
    fn head(t: &[f32], len: usize, h: usize, i: usize) -> &[f32] {
        &t[(h * len + i) * HEAD_DIM..][..HEAD_DIM]
    }

    /// A prompt, then one decode step over the block's own `KvCache`, each through `project`,
    /// `attend` and `output`: every stage equals the reference over the whole sequence.
    #[test]
    fn gated_attention_matches_reference() -> Result<()> {
        // The CPU's eager SDPA, which serves the prompt, multiplies in f16 (hanzo-quant
        // `MatMul::matmul` without `accelerate`): 2^-11 rounding per operand through the score and
        // value products. The single-query kernel that serves the decode step is f32.
        const F16_ATTENTION: f64 = 1e-2;
        let dev = Device::Cpu;
        let tensors = draw(
            &[
                ("attn_q.weight", &[2 * HEADS * HEAD_DIM, HIDDEN]),
                ("attn_k.weight", &[KV_HEADS * HEAD_DIM, HIDDEN]),
                ("attn_v.weight", &[KV_HEADS * HEAD_DIM, HIDDEN]),
                ("attn_output.weight", &[HIDDEN, HEADS * HEAD_DIM]),
                ("attn_q_norm.weight", &[HEAD_DIM]),
                ("attn_k_norm.weight", &[HEAD_DIM]),
            ],
            0x6174_746e,
        );
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let mut files = [open(dir.path(), &tensors)?];
        let mut readers: Vec<&mut std::fs::File> = files.iter_mut().collect();
        let mut ct = Content::from_readers(&mut readers)?;
        let props = props();
        let rotary = Arc::new(Qwen3VLRotaryEmbedding::new(
            THETA,
            ROT_DIM,
            &dev,
            props.mrope_section.clone(),
        )?);
        let attn =
            GatedFullAttention::load(&mut ct, "blk.0", &props, rotary, None, &dev, DType::F32)?;
        let w: HashMap<String, Vec<f64>> = tensors
            .iter()
            .map(|(name, _, data)| (name.clone(), data.iter().map(|&a| f64::from(a)).collect()))
            .collect();

        let (prompt, total) = (5usize, 6usize);
        let mut rng = StdRng::seed_from_u64(0x7870);
        let x: Vec<f32> = (0..total * HIDDEN)
            .map(|_| rng.random_range(-1f32..1.0))
            .collect();
        let want = attention_reference(&x.iter().map(|&a| f64::from(a)).collect::<Vec<_>>(), &w);
        let xs = Tensor::from_vec(x, (1, total, HIDDEN), &dev)?;

        let mut cache = KvCache::new_normal(2, total, total);
        for (start, len) in [(0, prompt), (prompt, 1)] {
            let ids = Tensor::zeros((1, len), DType::U32, &dev)?;
            let offsets: &[usize] = &[start];
            let mask = CausalMasker.make_causal_mask(
                &ids,
                &offsets,
                DType::F32,
                &CausalMaskConfig::gguf(),
            )?;
            let pos = Tensor::arange(start as u32, (start + len) as u32, &dev)?.unsqueeze(0)?;
            let cos_sin =
                attn.rotary_cos_sin(&Tensor::stack(&[&pos, &pos, &pos], 0)?, DType::F32)?;
            let chunk = xs.narrow(1, start, len)?.contiguous()?;
            let (q, k, v, gate) = attn.project(&chunk, &cos_sin)?;
            let y = attn.attend(&q, &k, &v, &mask, &mut cache, None)?;
            let out = attn.output(&y, &gate, DType::F32)?;

            let tol = if len > 1 { F16_ATTENTION } else { 1e-5 };
            let values = |t: &Tensor| t.flatten_all()?.to_vec1::<f32>();
            let (q, k, v) = (values(&q)?, values(&k)?, values(&v)?);
            let (gate, y, out) = (values(&gate)?, values(&y)?, values(&out)?);
            let width = HEADS * HEAD_DIM;
            for (i, row) in want[start..start + len].iter().enumerate() {
                let at = start + i;
                for h in 0..HEADS {
                    let own = h * HEAD_DIM..(h + 1) * HEAD_DIM;
                    check("q", at, head(&q, len, h, i), &row.q[own], 1e-5);
                }
                for h in 0..KV_HEADS {
                    let own = h * HEAD_DIM..(h + 1) * HEAD_DIM;
                    check("k", at, head(&k, len, h, i), &row.k[own.clone()], 1e-5);
                    check("v", at, head(&v, len, h, i), &row.v[own], 1e-5);
                }
                check("gate", at, &gate[i * width..][..width], &row.gate, 1e-5);
                check("y", at, &y[i * width..][..width], &row.y, tol);
                check("out", at, &out[i * HIDDEN..][..HIDDEN], &row.out, tol);
            }
        }
        Ok(())
    }

    /// `QGatedDeltaNet::load` reads a Qwen3.5 GGUF block into the layer the safetensors
    /// `GatedDeltaNet` computes from the same weights. With as many V heads as K heads the GGUF's
    /// tiled V order is the checkpoint's grouped order, so the two agree over a prompt and a step.
    #[test]
    fn gdn_load_matches_safetensors_layer() -> Result<()> {
        let dev = Device::Cpu;
        let (key_dim, value_dim) = (GDN_HEADS * GDN_DIM, GDN_HEADS * GDN_DIM);
        let conv_dim = 2 * key_dim + value_dim;
        let mut tensors = draw(
            &[
                ("attn_qkv.weight", &[conv_dim, HIDDEN]),
                ("attn_gate.weight", &[value_dim, HIDDEN]),
                ("ssm_beta.weight", &[GDN_HEADS, HIDDEN]),
                ("ssm_alpha.weight", &[GDN_HEADS, HIDDEN]),
                ("ssm_out.weight", &[HIDDEN, value_dim]),
                ("ssm_conv1d.weight", &[conv_dim, CONV]),
                ("ssm_dt.bias", &[GDN_HEADS]),
                ("ssm_norm.weight", &[GDN_DIM]),
            ],
            0x6764_6e6c,
        );
        let a_log = Tensor::new(&[-0.3f32, 0.4], &dev)?;
        // The converter stores -exp(A_log).
        let ssm_a = a_log.exp()?.neg()?;
        tensors.push(("blk.0.ssm_a".to_string(), vec![GDN_HEADS], ssm_a.to_vec1()?));
        let tensor = |name: &str| -> Result<Tensor> {
            let (_, shape, data) = tensors
                .iter()
                .find(|(n, _, _)| n == &format!("blk.0.{name}"))
                .expect("drawn above");
            Tensor::from_vec(data.clone(), shape.as_slice(), &dev)
        };
        let dense = |name: &str| -> Result<Arc<dyn QuantMethod>> {
            Ok(Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(tensor(name)?, None)),
            )?))
        };
        let twin = GatedDeltaNet {
            in_proj: GdnInProj::Split {
                qkv: dense("attn_qkv.weight")?,
                z: dense("attn_gate.weight")?,
                b: dense("ssm_beta.weight")?,
                a: dense("ssm_alpha.weight")?,
            },
            conv1d_weight: tensor("ssm_conv1d.weight")?.unsqueeze(1)?,
            dt_bias: tensor("ssm_dt.bias")?,
            a_log,
            norm: RmsNormGated::from_weight(tensor("ssm_norm.weight")?, f64::from(EPS)),
            out_proj: dense("ssm_out.weight")?,
            num_k_heads: GDN_HEADS,
            num_v_heads: GDN_HEADS,
            head_k_dim: GDN_DIM,
            head_v_dim: GDN_DIM,
            conv_kernel_size: CONV,
            key_dim,
            value_dim,
        };

        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let mut files = [open(dir.path(), &tensors)?];
        let mut readers: Vec<&mut std::fs::File> = files.iter_mut().collect();
        let mut ct = Content::from_readers(&mut readers)?;
        let gdn = QGatedDeltaNet::load(&mut ct, "blk.0", &props(), &dev)?;

        let fresh = || -> Result<GdnLayerCache> {
            Ok(GdnLayerCache {
                conv_state: Tensor::zeros((1, conv_dim, CONV), DType::F32, &dev)?,
                recurrent_state: Tensor::zeros((1, GDN_HEADS, GDN_DIM, GDN_DIM), DType::F32, &dev)?,
                seqlen_offset: 0,
                trail: None,
            })
        };
        let (mut ours, mut theirs) = (fresh()?, fresh()?);
        let mut rng = StdRng::seed_from_u64(0x6764_6e78);
        let mut at = 0;
        for len in [5usize, 1] {
            let x: Vec<f32> = (0..len * HIDDEN)
                .map(|_| rng.random_range(-1f32..1.0))
                .collect();
            let x = Tensor::from_vec(x, (1, len, HIDDEN), &dev)?;
            let got = gdn
                .forward(&x, &mut ours)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let want = twin
                .forward(&x, &mut theirs)?
                .flatten_all()?
                .to_dtype(DType::F64)?
                .to_vec1::<f64>()?;
            check("gdn", at, &got, &want, 1e-5);
            at += len;
        }
        Ok(())
    }
}
