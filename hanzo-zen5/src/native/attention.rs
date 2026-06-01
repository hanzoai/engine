//! Multi-head Latent Attention (MLA) for DeepSeek V4.
//!
//! MLA compresses KV via a low-rank projection (kv_lora_rank=512) and
//! decouples RoPE from the value head. Diagram:
//!
//! ```text
//! q  = down_q(x) · up_q       split → q_nope (qk_nope_head_dim)
//!                                   ↘ q_rope (qk_rope_head_dim) → RoPE
//! kv = down_kv(x)             split → c_kv (kv_lora_rank)        → k_nope, v
//!                                   ↘ k_rope (qk_rope_head_dim) → RoPE (shared)
//! attn = softmax(q · kᵀ / √d) · v
//! ```
//!
//! Cache shape: `[batch, seq, kv_lora_rank + qk_rope_head_dim]` instead of
//! the usual `[batch, seq, n_kv_heads * head_dim]` — savings are ~5–15× at
//! 256K context.
//!
//! Status: scaffold. Translate from
//! `mistralrs-core/src/models/deepseek3.rs::mla` and
//! `candle-transformers/src/models/deepseek2.rs::AttentionLayer` once those
//! versions are pinned in the workspace.

use super::Zen5Config;

#[cfg(feature = "native")]
use candle_core::{Result, Tensor};
#[cfg(feature = "native")]
use candle_nn::{Module, VarBuilder};

#[cfg(feature = "native")]
#[derive(Debug)]
pub struct Mla {
    pub cfg: Zen5Config,
    // TODO(native): wire up these projections from a candle VarBuilder.
    // pub q_a_proj: Linear,
    // pub q_a_layernorm: candle_nn::RmsNorm,
    // pub q_b_proj: Linear,
    // pub kv_a_proj_with_mqa: Linear,
    // pub kv_a_layernorm: candle_nn::RmsNorm,
    // pub kv_b_proj: Linear,
    // pub o_proj: Linear,
    // pub rotary: DeepSeekV2RotaryEmbedding,
}

#[cfg(feature = "native")]
impl Mla {
    pub fn new(cfg: &Zen5Config, _vb: VarBuilder<'_>) -> Result<Self> {
        // TODO(native): construct each Linear, the RmsNorms, and the
        // YARN-scaled rotary embedding. See mistral.rs deepseek3.rs lines
        // ~120-260 for the reference wiring.
        Ok(Self { cfg: cfg.clone() })
    }
}

#[cfg(feature = "native")]
impl Module for Mla {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // TODO(native): MLA forward:
        //   1. q_compressed   = q_a_layernorm(q_a_proj(xs))
        //   2. q              = q_b_proj(q_compressed) → split (nope, rope)
        //   3. kv_compressed  = kv_a_layernorm(slice(kv_a_proj_with_mqa(xs), :kv_lora_rank))
        //   4. k_rope         = slice(kv_a_proj_with_mqa(xs), kv_lora_rank:)
        //   5. kv             = kv_b_proj(kv_compressed)  → split (k_nope, v)
        //   6. q_rope, k_rope ← rotary
        //   7. attn = softmax(q · kᵀ * softmax_scale) · v
        //   8. o_proj(attn)
        let _ = xs;
        unimplemented!("Mla::forward — scaffold")
    }
}

#[cfg(not(feature = "native"))]
#[derive(Debug)]
pub struct Mla {
    _cfg: Zen5Config,
}
