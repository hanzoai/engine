#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, dead_code)]

use serde::Deserialize;

use crate::speech_models::Qwen3TtsConfig;
use crate::vision_models::qwen3_5_moe::Config as ThinkerConfig;
use crate::vision_models::voxtral::config::WhisperEncoderArgs;

use crate::serde_default_fn;

serde_default_fn!(u32, default_audio_token_id, 151646);
serde_default_fn!(u32, default_audio_start_token_id, 151647);
serde_default_fn!(u32, default_audio_end_token_id, 151648);

/// Aggregate config for the fused Qwen3.6-MoE-Omni model.
///
/// Omni is understand -> translate -> speak: a shared Thinker (Qwen3.5/3.6 MoE, the same
/// hybrid-attention text+vision backbone as `qwen3_5_moe`) consumes interleaved text, image,
/// video and audio embeddings and emits both text logits and a hidden-state stream that the
/// Talker (the `qwen3_tts` autoregressive codec backbone) renders to a waveform.
///
/// The three sub-configs below correspond to the three weight namespaces that a published
/// `zenlm/zen-omni` checkpoint would expose: `thinker.*`, `audio_encoder.*`, `talker.*`. No such
/// checkpoint exists yet, so this struct only defines the assembled shape.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3OmniConfig {
    /// The MoE Thinker: text + vision tower + hybrid (full/linear) attention decoder.
    pub thinker_config: ThinkerConfig,
    /// Whisper-style causal audio encoder (mel -> hidden), reused from Voxtral.
    pub audio_encoder_config: WhisperEncoderArgs,
    /// The TTS Talker + MTP code predictor + neural codec decoder.
    pub talker_config: Qwen3TtsConfig,

    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: u32,
    #[serde(default = "default_audio_start_token_id")]
    pub audio_start_token_id: u32,
    #[serde(default = "default_audio_end_token_id")]
    pub audio_end_token_id: u32,

    #[serde(default)]
    pub tie_word_embeddings: bool,
}
