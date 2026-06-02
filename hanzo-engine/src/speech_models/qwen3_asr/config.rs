#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use serde::{Deserialize, Serialize};

use crate::serde_default_fn;

serde_default_fn!(bool, default_true, true);
serde_default_fn!(f64, default_rms_eps, 1e-6);
serde_default_fn!(f64, default_rope_theta, 1_000_000.0);
serde_default_fn!(usize, default_audio_token_id, 151_676);
serde_default_fn!(usize, default_sampling_rate, 16_000);
serde_default_fn!(usize, default_n_mels, 128);
serde_default_fn!(usize, default_hop_length, 160);
serde_default_fn!(usize, default_window_size, 400);
serde_default_fn!(usize, default_conv_channels, 480);

/// AuT audio encoder configuration (`thinker.audio_tower`).
///
/// Conv2d stem (3 layers, k=3 s=2 p=1, 8x downsampling) feeding a bidirectional
/// transformer with sinusoidal absolute position embeddings and LayerNorm.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AudioEncoderConfig {
    #[serde(alias = "d_model", alias = "hidden_size")]
    pub d_model: usize,
    #[serde(alias = "encoder_layers", alias = "num_hidden_layers")]
    pub num_layers: usize,
    #[serde(alias = "encoder_attention_heads", alias = "num_attention_heads")]
    pub num_heads: usize,
    #[serde(alias = "encoder_ffn_dim", alias = "intermediate_size")]
    pub ffn_dim: usize,
    #[serde(default = "default_n_mels", alias = "num_mel_bins")]
    pub n_mels: usize,
    #[serde(default = "default_conv_channels")]
    pub conv_channels: usize,
    /// Projection target dim feeding the LM (`proj2` out features == decoder hidden).
    #[serde(alias = "output_dim", alias = "d_out")]
    pub output_dim: usize,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: usize,
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
}

/// Generous cap for the precomputed sinusoidal table: 8x Conv2d downsampling of
/// 30s @ 16kHz / hop 160 yields ~375 frames; round up for longer clips.
const MAX_AUDIO_POSITIONS: usize = 4096;

impl AudioEncoderConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    pub fn max_audio_positions(&self) -> usize {
        MAX_AUDIO_POSITIONS
    }

    /// Frequency-axis length after one `k=3 s=2 p=1` Conv2d: `floor((f-1)/2)+1`.
    fn conv_freq_step(f: usize) -> usize {
        (f - 1) / 2 + 1
    }

    /// Flattened conv-stem feature width feeding `conv_out`: `conv_channels *`
    /// freq length after the 3 stride-2 Conv2d layers over `n_mels`.
    pub fn conv_feature_dim(&self) -> usize {
        let f = Self::conv_freq_step(Self::conv_freq_step(Self::conv_freq_step(self.n_mels)));
        self.conv_channels * f
    }
}

/// Qwen3 text decoder configuration (`thinker.model`).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TextDecoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
}

/// Top-level Qwen3-ASR config. HF nests the encoder under `audio_config` /
/// `audio_tower_config` and the decoder under `text_config`; aliases cover both.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen3AsrConfig {
    #[serde(alias = "audio_tower_config", alias = "audio_encoder")]
    pub audio_config: AudioEncoderConfig,
    #[serde(alias = "decoder_config", alias = "llm_config")]
    pub text_config: TextDecoderConfig,
    /// `<|audio_pad|>` placeholder id whose embedding is replaced by audio features.
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: usize,
}
