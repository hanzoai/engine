use hanzo_ml::{Device, Result, Tensor};

use crate::speech_models::utils;

pub const WHISPER_N_FFT: usize = 400;
pub const WHISPER_HOP: usize = 160;
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// openai-whisper log-mel `[n_mels, T]` for 16 kHz mono PCM. Thin preset over the
/// shared `utils::log_mel_tensor` so ASR and whisper-tiny share one frontend.
pub fn log_mel_spectrogram(samples: &[f32], n_mels: usize, device: &Device) -> Result<Tensor> {
    utils::log_mel_tensor(
        samples,
        WHISPER_N_FFT,
        WHISPER_HOP,
        n_mels,
        WHISPER_SAMPLE_RATE,
        device,
    )
}
