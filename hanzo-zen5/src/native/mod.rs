//! Pure-Rust scaffold of DeepSeek V4 Flash on top of `candle-core` /
//! `candle-transformers`.
//!
//! ## State of the world (2026-05)
//! * `candle-transformers/src/models/deepseek2.rs` exists in candle main.
//!   V3 and V4 are NOT yet upstream in candle (only V2 / DeepSeek-VL2).
//! * `mistralrs-core/src/models/deepseek3.rs` ships in mistral.rs — DeepSeek
//!   V3 with full MLA + MoE. V4 Flash is architecturally identical to V3
//!   (same MLA, same sparse MoE; the "Flash" change is the 4B routing-only
//!   dense expert + MTP head). So porting V3 forward to V4 is small.
//! * `mistralrs-core` also has `qwen3.rs`, `qwen3_moe.rs`, `qwen3_next.rs`
//!   (Gated DeltaNet). zen-5-flash itself is a Qwen3-4B base, so
//!   `qwen3.rs` is the direct fit for the 4B SKU.
//! * `candle-transformers` has `qwen3.rs`, `qwen3_moe.rs`, `qwen3_vl/`,
//!   plus `quantized_qwen3.rs` and `quantized_qwen3_moe.rs` for GGUF.
//!
//! ## Approach
//! Until DeepSeek V4 lands in candle, the native path will either:
//!   (a) port mistral.rs `deepseek3.rs` → V4 (rename a few constants, add
//!       the MTP head); or
//!   (b) compose `candle-transformers::models::deepseek2` (MLA + MoE) with
//!       custom router constants matching V4 Flash (256 experts, top-8).
//!
//! This module is the (b) scaffold today. Submodules:
//! * [`attention`] — MLA (Multi-head Latent Attention)
//! * [`moe`] — sparse MoE routing, top-k expert dispatch

pub mod attention;
pub mod moe;

use std::path::Path;

use async_trait::async_trait;

use crate::engine::{GenOpts, Token, TokenStream, Zen5Engine, Zen5Error};

/// Native candle-rs engine for zen5 DeepSeek V4 Flash. Scaffold only —
/// load + forward pass are not yet implemented.
#[derive(Debug)]
pub struct Engine {
    #[cfg(feature = "native")]
    _model: Zen5Model,
    // Without the `native` feature the struct is empty; lib.rs gates
    // construction so this path isn't reachable.
}

impl Engine {
    /// Load a GGUF or safetensors checkpoint.
    pub fn load(path: &Path) -> Result<Self, Zen5Error> {
        let _ = path;
        // TODO(native):
        //   1. Detect safetensors vs gguf by file magic.
        //   2. For gguf, drive `candle-transformers::models::quantized_qwen3_moe`
        //      (zen-5-flash is Qwen3-4B base — use quantized_qwen3 directly).
        //   3. For safetensors, build `Zen5Model` via candle's `VarBuilder`.
        //   4. Move the tokenizer load to a shared `tokenizer.rs` so it can
        //      be reused by FFI when we drop the C tokenizer.
        Err(Zen5Error::Backend(
            "native::Engine::load not implemented yet — use the ffi backend".into(),
        ))
    }
}

#[cfg(feature = "native")]
#[derive(Debug)]
pub struct Zen5Model {
    pub cfg: Zen5Config,
    // TODO(native): hold the candle weights + tokenizer here.
    // pub embed: candle_nn::Embedding,
    // pub layers: Vec<Layer>,
    // pub norm: candle_nn::RmsNorm,
    // pub lm_head: candle_nn::Linear,
}

/// DeepSeek V4 Flash architectural constants. Values mirror DeepSeek V3
/// public config (V4 Flash is the same family at smaller scale + MTP head).
/// Override per-checkpoint when loading.
#[derive(Debug, Clone)]
pub struct Zen5Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,

    // MLA-specific
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,

    // MoE
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub first_k_dense_replace: usize,
    pub norm_topk_prob: bool,
    pub scoring_func: ScoringFunc,
    pub aux_loss_alpha: f32,

    // V4 Flash adds Multi-Token Prediction head
    pub mtp_num_layers: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum ScoringFunc {
    Softmax,
    Sigmoid,
}

impl Zen5Config {
    /// DeepSeek V4 Flash 284B IQ2_XXS preset.
    /// Matches `https://github.com/zenlm/zen5-engine` defaults.
    pub fn v4_flash_284b() -> Self {
        Self {
            vocab_size: 129_280,
            hidden_size: 7_168,
            intermediate_size: 18_432,
            num_hidden_layers: 61,
            num_attention_heads: 128,
            num_key_value_heads: 128,
            max_position_embeddings: 163_840,
            rope_theta: 10_000.0,
            q_lora_rank: 1_536,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            qk_nope_head_dim: 128,
            v_head_dim: 128,
            n_routed_experts: 256,
            n_shared_experts: 1,
            num_experts_per_tok: 8,
            moe_intermediate_size: 2_048,
            first_k_dense_replace: 3,
            norm_topk_prob: true,
            scoring_func: ScoringFunc::Sigmoid,
            aux_loss_alpha: 0.001,
            mtp_num_layers: 1,
        }
    }
}

#[async_trait]
impl Zen5Engine for Engine {
    fn backend(&self) -> &'static str {
        "native/candle"
    }

    async fn complete(&self, _prompt: &str, _opts: GenOpts) -> Result<TokenStream, Zen5Error> {
        Err(Zen5Error::Backend(
            "native::Engine::complete not implemented yet".into(),
        ))
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>, Zen5Error> {
        Err(Zen5Error::Backend(
            "native::Engine::embed not implemented yet".into(),
        ))
    }
}

// Unused import warning silencer when the cfg gate hides the field.
#[allow(dead_code)]
fn _touch<T>(_: T) {}

#[allow(dead_code)]
fn _touch_token(t: Token) -> Token {
    t
}
