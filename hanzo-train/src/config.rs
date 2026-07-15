//! Hugging Face `config.json` parsing for the Llama / Qwen2 decoder family
//! (SmolLM2, Qwen2.5, Llama). Only the fields the training forward needs are read.

use serde::Deserialize;

fn default_rope_theta() -> f64 {
    10000.0
}
fn default_rms_eps() -> f64 {
    1e-6
}
fn default_max_pos() -> usize {
    4096
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Absent => same as `num_attention_heads` (no GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// Absent => `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_smollm2_style_config_with_gqa_defaults() {
        // SmolLM2-135M shape, kv heads present, tied embeddings, no explicit head_dim.
        let json = r#"{
            "vocab_size": 49152,
            "hidden_size": 576,
            "intermediate_size": 1536,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.num_kv_heads(), 3);
        assert_eq!(cfg.num_kv_groups(), 3);
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn kv_heads_default_to_attention_heads() {
        let json = r#"{
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.num_kv_heads(), 8);
        assert_eq!(cfg.num_kv_groups(), 1);
        assert_eq!(cfg.head_dim(), 8);
        assert_eq!(cfg.rope_theta, 10000.0);
    }
}
