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
serde_default_fn!(usize, default_audio_start_token_id, 151_669);
serde_default_fn!(usize, default_audio_end_token_id, 151_670);
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
    // HF audio configs duplicate every dim under both the encoder name and the
    // generic name (e.g. `encoder_layers` AND `num_hidden_layers`), so aliasing
    // both would make serde see a "duplicate field". Bind only the canonical
    // `encoder_*` / `d_model` keys the Qwen3-ASR audio_config actually uses.
    pub d_model: usize,
    #[serde(alias = "encoder_layers")]
    pub num_layers: usize,
    #[serde(alias = "encoder_attention_heads")]
    pub num_heads: usize,
    #[serde(alias = "encoder_ffn_dim")]
    pub ffn_dim: usize,
    #[serde(default = "default_n_mels", alias = "num_mel_bins")]
    pub n_mels: usize,
    #[serde(
        default = "default_conv_channels",
        alias = "downsample_hidden_size",
        alias = "conv_channels"
    )]
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

/// The audio + text + token-id fields, shared between the flat layout and the
/// `thinker_config`-nested layout that the published HF checkpoints actually use.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ThinkerConfig {
    #[serde(alias = "audio_tower_config", alias = "audio_encoder")]
    pub audio_config: AudioEncoderConfig,
    #[serde(alias = "decoder_config", alias = "llm_config")]
    pub text_config: TextDecoderConfig,
    /// `<|audio_pad|>` placeholder id whose embedding is replaced by audio features.
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: usize,
    #[serde(default = "default_audio_start_token_id")]
    pub audio_start_token_id: usize,
    #[serde(default = "default_audio_end_token_id")]
    pub audio_end_token_id: usize,
}

/// Top-level Qwen3-ASR config. The HF `Qwen3ASRForConditionalGeneration` config
/// wraps everything in `thinker_config`; older/flat configs put the fields at the
/// root. `#[serde(flatten)]` on the optional wrapper accepts both shapes.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen3AsrConfigRaw {
    pub thinker_config: Option<ThinkerConfig>,
    #[serde(flatten)]
    pub flat: Option<ThinkerConfig>,
}

#[derive(Debug, Clone)]
pub struct Qwen3AsrConfig {
    pub audio_config: AudioEncoderConfig,
    pub text_config: TextDecoderConfig,
    pub audio_token_id: usize,
    pub audio_start_token_id: usize,
    pub audio_end_token_id: usize,
}

impl<'de> Deserialize<'de> for Qwen3AsrConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = Qwen3AsrConfigRaw::deserialize(deserializer)?;
        let inner = raw.thinker_config.or(raw.flat).ok_or_else(|| {
            serde::de::Error::custom("missing `thinker_config` and no flat audio/text config")
        })?;
        Ok(Self {
            audio_config: inner.audio_config,
            text_config: inner.text_config,
            audio_token_id: inner.audio_token_id,
            audio_start_token_id: inner.audio_start_token_id,
            audio_end_token_id: inner.audio_end_token_id,
        })
    }
}
