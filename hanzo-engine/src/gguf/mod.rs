mod chat_template;
mod content;
mod gguf_tokenizer;
use strum::EnumString;

use anyhow::{Context, Result};
pub(crate) use chat_template::get_gguf_chat_template;
pub(crate) use content::Content;
pub(crate) use gguf_tokenizer::{convert_gguf_to_hf_tokenizer, GgufTokenizerConversion};
use std::str::FromStr;

pub const GGUF_MULTI_FILE_DELIMITER: &str = ";";

#[derive(Debug, EnumString, Clone, Copy, strum::Display)]
#[strum(serialize_all = "lowercase")]
pub enum GGUFArchitecture {
    Llama,
    Mpt,
    Gptneox,
    Gptj,
    Gpt2,
    Bloom,
    Falcon,
    Mamba,
    Rwkv,
    Phi2,
    Phi3,
    Starcoder2,
    Qwen2,
    Qwen3,
    Qwen3MoE,
    /// Qwen3-VL text backbone (dense 2/4/8/32B). Structurally identical to `Qwen3`
    /// (same `blk.*` tensor names, q/k-norm, GQA); the only VL-specific bits —
    /// interleaved-MRoPE and DeepStack vision injection — collapse to the plain
    /// Qwen3 path for text-only input (all mrope sections carry the same position
    /// when t==h==w, so the interleaved partition is a no-op). Reuses `QQwen3`.
    Qwen3Vl,
    /// Qwen3-VL MoE text backbone (30B-A3B, 235B-A22B). Same relationship to
    /// `Qwen3MoE` as `Qwen3Vl` is to `Qwen3`. Reuses `QQwen3MoE`.
    Qwen3VlMoE,
    Qwen35,
    Qwen35MoE,
    Mistral3,
    #[strum(serialize = "gpt-oss")]
    GptOss,
    Glm4Moe,
    Deepseek2,
    Deepseek4,
    /// The DeepSeek-V4 MTP draft-head GGUF (a separate file). Never loaded as a
    /// standalone model — opened only by the MTP speculative loader, which reads its
    /// `mtp.0.*` tensors directly.
    #[strum(serialize = "deepseek4_mtp_support")]
    Deepseek4MtpSupport,
}

// Wraps from_str() for some convenience:
// - Case-insensitive variant matching (TODO: is this desirable?)
// - Customized error until potential upstream support: https://github.com/Peternator7/strum/issues/332
impl GGUFArchitecture {
    pub fn from_value<T: AsRef<str> + std::fmt::Display>(value: T) -> Result<Self> {
        Self::from_str(&value.as_ref().to_ascii_lowercase())
            .with_context(|| format!("Unknown GGUF architecture `{value}`"))
            .map_err(anyhow::Error::msg)
    }
}
