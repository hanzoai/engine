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
serde_default_fn!(usize, default_n_window, 50);
serde_default_fn!(usize, default_n_window_infer, 800);

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
    /// Half the conv-chunk size, in raw mel frames. The mel is split into
    /// `n_window * 2`-frame chunks before the conv stem (HF `n_window`).
    #[serde(default = "default_n_window")]
    pub n_window: usize,
    /// Inference attention-window size, in raw mel frames. Post-CNN features are
    /// attended block-diagonally in windows of this size (HF `n_window_infer`).
    #[serde(default = "default_n_window_infer")]
    pub n_window_infer: usize,
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

    /// Post-CNN time length for a single conv chunk of `frames` raw mel frames.
    /// One `k=3 s=2 p=1` Conv2d gives `floor((L-1)/2)+1`; applied 3x.
    pub fn conv_time_len(frames: usize) -> usize {
        Self::conv_freq_step(Self::conv_freq_step(Self::conv_freq_step(frames)))
    }

    /// Raw chunk size in mel frames (`n_window * 2`).
    pub fn chunk_size(&self) -> usize {
        self.n_window * 2
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

// ---- structural CI tests: parse the real zen-3 configs and assert dub-critical invariants.
// Cheap (JSON only, no weights); skips if the config files are absent so it is CI-safe.
#[cfg(test)]
mod zen3_struct {
    use super::*;

    fn asr_cfg_path() -> String {
        std::env::var("ZEN3_ASR_CONFIG")
            .unwrap_or_else(|_| "/home/z/work/zen/hf/zen-3-asr-0.6B/config.json".to_string())
    }
    fn tts_cfg_path() -> String {
        std::env::var("ZEN3_TTS_CONFIG")
            .unwrap_or_else(|_| "/home/z/work/zen/hf/zen-3-tts-0.6B/config.json".to_string())
    }

    #[test]
    fn asr_config_parses_and_is_sane() {
        let p = asr_cfg_path();
        let Ok(s) = std::fs::read_to_string(&p) else {
            eprintln!("ASR config {p} absent; skipping");
            return;
        };
        let cfg: Qwen3AsrConfig = serde_json::from_str(&s).expect("parse ASR config.json");
        // audio encoder feeds 80-mel 16kHz frames into the thinker.
        assert!(cfg.audio_config.n_mels >= 80, "expected >=80 mel bins");
        assert!(cfg.audio_config.d_model > 0 && cfg.audio_config.num_layers > 0);
        assert_eq!(cfg.audio_config.sampling_rate, 16000, "ASR ingests 16kHz");
        // text decoder must have a real vocab + matching head geometry.
        assert!(cfg.text_config.vocab_size > 1000);
        assert!(cfg.text_config.num_attention_heads >= cfg.text_config.num_key_value_heads);
        assert!(cfg.text_config.head_dim > 0);
    }

    #[test]
    fn tts_config_parses_and_is_sane() {
        use crate::speech_models::qwen3_tts::Qwen3TtsConfig;
        let p = tts_cfg_path();
        let Ok(s) = std::fs::read_to_string(&p) else {
            eprintln!("TTS config {p} absent; skipping");
            return;
        };
        let cfg: Qwen3TtsConfig = serde_json::from_str(&s).expect("parse TTS config.json");
        // the talker drives 16 residual code groups into the codec (zen-3-tts-0.6B topology).
        assert_eq!(cfg.talker_config.num_code_groups, 16, "expected 16 code groups");
        assert_eq!(
            cfg.talker_config.code_predictor_config.num_code_groups,
            cfg.talker_config.num_code_groups,
            "talker and code_predictor must agree on group count"
        );
        assert!(cfg.talker_config.hidden_size > 0);
        assert!(cfg.talker_config.vocab_size > 0 && cfg.talker_config.text_vocab_size > 0);
        // special-token ids the prefill builder depends on are present and distinct.
        let ids = [
            cfg.tts_bos_token_id,
            cfg.tts_eos_token_id,
            cfg.tts_pad_token_id,
        ];
        assert!(ids.iter().all(|&x| x > 0));
        assert!(ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2]);
    }
}
