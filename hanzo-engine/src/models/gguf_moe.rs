#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Shared GGUF fused-MoE block for the DeepSeek-V2/V3 and GLM-4.5/4.7 families.
//!
//! Both `deepseek2` and `glm4moe` GGUFs stack the routed experts identically
//! (`ffn_gate_inp`, `ffn_{gate,up,down}_exps`, `ffn_{gate,up,down}_shexp`, `exp_probs_b.bias`)
//! and select them with the same sigmoid/softmax + optional group-limited no-aux-loss gate. This is
//! the one place that math lives; the arch models supply only their per-family params + attention.

use std::sync::Arc;

use crate::gguf::Content;
use crate::ops::{TopKLastDimOp, TopKOutput};
use hanzo_ml::quantized::QMatMul;
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

pub(crate) fn gguf_linear(q: hanzo_ml::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(q),
        b: None,
    })?))
}

pub(crate) struct Mlp {
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

pub(crate) struct MoeGate {
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
    // (topk_idx, topk_weight). Greedy softmax (V2) or grouped sigmoid no-aux (V3/GLM) selection.
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (bs, seq_len, h) = xs.dims3()?;
        let xs = xs.reshape(((), h))?;
        let logits = xs
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.weight.t()?.to_dtype(DType::F32)?)?;
        let scores = if self.sigmoid_scoring {
            // 1/(1+exp(-x)) via standard unaries; the fused Sigmoid custom-op has no ROCm kernel.
            (logits.neg()?.exp()? + 1.0)?.recip()?
        } else {
            hanzo_nn::ops::softmax_last_dim(&logits)?
        };

        let mut topk_weight;
        let topk_idx;
        if let Some(bias) = &self.e_score_correction_bias {
            let scores_for_choice = scores
                .reshape((bs * seq_len, ()))?
                .broadcast_add(&bias.unsqueeze(0)?)?;
            // With a single expert group the group-limited mask is all-ones (identity), so skip the
            // scatter entirely (also sidesteps ROCm's missing scatter_add). n_group > 1 keeps the
            // DeepSeek-V3 grouped no-aux-loss selection.
            let tmp_scores = if self.n_group <= 1 {
                scores_for_choice
            } else {
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
                scores_for_choice.broadcast_mul(&score_mask)?
            };
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

pub(crate) struct FusedMoe {
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
            self.down_experts
                .indexed_moe_forward(&activated, &topk_idx)?
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

pub(crate) enum MoeOrMlp {
    FusedMoe(Box<FusedMoe>),
    Mlp(Mlp),
}

impl MoeOrMlp {
    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs),
        }
    }
}

/// Per-family routing/expert config, all sourced from GGUF metadata.
pub(crate) struct MoeParams {
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
    pub norm_topk_prob: bool,
    pub sigmoid_scoring: bool,
    pub n_shared_experts: usize,
    pub leading_dense_block_count: usize,
}

/// Build the layer's feed-forward block: a dense SwiGLU MLP for the leading dense layers, else the
/// fused routed+shared MoE. `blk.{layer_idx}` tensors are read from `ct`.
pub(crate) fn build_moe_or_mlp<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    layer_idx: usize,
    device: &Device,
    p: &MoeParams,
) -> Result<MoeOrMlp> {
    let prefix = format!("blk.{layer_idx}");
    let is_moe = p.n_routed_experts > 0 && layer_idx >= p.leading_dense_block_count;
    if !is_moe {
        return Ok(MoeOrMlp::Mlp(Mlp {
            gate: gguf_linear(ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?)?,
            up: gguf_linear(ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?)?,
            down: gguf_linear(ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?)?,
        }));
    }

    let gate = ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?;
    let gate_experts = ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), device)?;
    let up_experts = ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), device)?;
    let down_experts = ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), device)?;
    let e_score_correction_bias =
        if p.sigmoid_scoring && ct.has_tensor(&format!("{prefix}.exp_probs_b.bias")) {
            Some(
                ct.tensor(&format!("{prefix}.exp_probs_b.bias"), device)?
                    .dequantize(device)?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };
    let shared = if p.n_shared_experts > 0 {
        Some(Mlp {
            gate: gguf_linear(ct.tensor(&format!("{prefix}.ffn_gate_shexp.weight"), device)?)?,
            up: gguf_linear(ct.tensor(&format!("{prefix}.ffn_up_shexp.weight"), device)?)?,
            down: gguf_linear(ct.tensor(&format!("{prefix}.ffn_down_shexp.weight"), device)?)?,
        })
    } else {
        None
    };
    Ok(MoeOrMlp::FusedMoe(Box::new(FusedMoe {
        gate: MoeGate {
            weight: gate.dequantize(device)?,
            e_score_correction_bias,
            top_k: p.num_experts_per_tok,
            n_routed_experts: p.n_routed_experts,
            n_group: p.n_group,
            topk_group: p.topk_group,
            routed_scaling_factor: p.routed_scaling_factor,
            norm_topk_prob: p.norm_topk_prob,
            sigmoid_scoring: p.sigmoid_scoring,
        },
        gate_experts: QMatMul::from_qtensor(gate_experts)?,
        up_experts: QMatMul::from_qtensor(up_experts)?,
        down_experts: QMatMul::from_qtensor(down_experts)?,
        shared,
    })))
}
