//! Sparse Mixture-of-Experts routing for DeepSeek V4 Flash.
//!
//! Architecture:
//! * 256 routed experts (`n_routed_experts`).
//! * Top-8 per token (`num_experts_per_tok`).
//! * 1 always-on shared expert (`n_shared_experts`).
//! * First 3 layers are dense MLP (`first_k_dense_replace`); MoE starts at L4.
//! * Sigmoid scoring (DeepSeek V3 changed from softmax to sigmoid + bias).
//! * Auxiliary load-balancing loss with alpha=0.001.
//!
//! Status: scaffold. Translate from
//! `mistralrs-core/src/models/deepseek3.rs::DeepseekV3MoE`.

use super::{ScoringFunc, Zen5Config};

#[cfg(feature = "native")]
use candle_core::{Result, Tensor};
#[cfg(feature = "native")]
use candle_nn::{Module, VarBuilder};

/// One MoE block. Holds the gate, the routed experts, and the shared expert.
#[cfg(feature = "native")]
#[derive(Debug)]
pub struct MoeBlock {
    pub cfg: Zen5Config,
    // TODO(native):
    // pub gate: Linear,                  // [hidden_size, n_routed_experts]
    // pub gate_bias: Tensor,             // V3+: bias added before topk (sigmoid path)
    // pub experts: Vec<ExpertFfn>,       // 256 experts
    // pub shared_expert: ExpertFfn,
}

#[cfg(feature = "native")]
#[derive(Debug)]
pub struct ExpertFfn {
    // pub gate_proj: Linear,
    // pub up_proj: Linear,
    // pub down_proj: Linear,
}

#[cfg(feature = "native")]
impl MoeBlock {
    pub fn new(cfg: &Zen5Config, _vb: VarBuilder<'_>) -> Result<Self> {
        // TODO(native):
        //   1. Build `gate` Linear from vb.pp("gate")
        //   2. Build `gate_bias` from vb.pp("gate.e_score_correction_bias")
        //      (only present in V3+/V4 with sigmoid scoring)
        //   3. Build N=n_routed_experts ExpertFfn from vb.pp("experts.{i}")
        //   4. Build shared_expert from vb.pp("shared_experts")
        Ok(Self { cfg: cfg.clone() })
    }

    /// Score → topk → dispatch → weighted sum.
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        // TODO(native):
        //   logits = gate(x) + gate_bias
        //   scores = match self.cfg.scoring_func {
        //       Softmax => softmax(logits, -1),
        //       Sigmoid => sigmoid(logits),
        //   }
        //   (topk_w, topk_idx) = topk(scores, num_experts_per_tok)
        //   if norm_topk_prob: topk_w /= topk_w.sum(-1, keepdim=true)
        //   y = zeros_like(x)
        //   for each token t, for each picked expert e:
        //       y[t] += topk_w[t, e] * experts[e](x[t])
        //   y += shared_expert(x)
        //   return y
        unimplemented!("MoeBlock::forward — scaffold")
    }
}

#[cfg(feature = "native")]
impl Module for MoeBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        MoeBlock::forward(self, xs)
    }
}

#[cfg(not(feature = "native"))]
#[derive(Debug)]
pub struct MoeBlock {
    _cfg: Zen5Config,
}

// Used by `Zen5Config::scoring_func` callers in the scaffold.
#[cfg(feature = "native")]
fn _scoring_branch(s: ScoringFunc) -> &'static str {
    match s {
        ScoringFunc::Softmax => "softmax",
        ScoringFunc::Sigmoid => "sigmoid",
    }
}
