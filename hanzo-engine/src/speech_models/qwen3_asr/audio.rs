#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#![allow(dead_code)]

//! Whisper-style log-mel frontend for Qwen3-ASR. The HF processor is a
//! `WhisperFeatureExtractor` (n_fft=400, hop=160, 128 Slaney mel bins, centered
//! STFT, `log10` magnitudes clamped to `max - 8` then mapped to `(x + 4) / 4`).
//! The STFT/mel/normalize core and the resampler live in `speech_models::utils`
//! (shared with the openai-whisper-tiny frontend); this just wires the config.
//! Output is `[B, n_mels, T]`, the layout the AuT conv stem expects.

use hanzo_audio::AudioInput;
use hanzo_ml::{Device, Result, Tensor};

use super::config::AudioEncoderConfig;
use crate::speech_models::utils;

pub struct Qwen3AsrAudioProcessor {
    sampling_rate: u32,
    n_mels: usize,
    hop_length: usize,
    n_fft: usize,
}

impl Qwen3AsrAudioProcessor {
    pub fn new(cfg: &AudioEncoderConfig) -> Self {
        Self {
            sampling_rate: cfg.sampling_rate as u32,
            n_mels: cfg.n_mels,
            hop_length: cfg.hop_length,
            n_fft: cfg.window_size,
        }
    }

    /// Audio -> log-mel `[1, n_mels, T]` on `device`.
    pub fn process(&self, audio: &AudioInput, device: &Device) -> Result<Tensor> {
        let mono = audio.to_mono();
        let samples = if audio.sample_rate != self.sampling_rate {
            utils::resample(&mono, audio.sample_rate, self.sampling_rate)?
        } else {
            mono
        };
        utils::log_mel_tensor(
            &samples,
            self.n_fft,
            self.hop_length,
            self.n_mels,
            self.sampling_rate,
            device,
        )?
        .unsqueeze(0)
    }
}
