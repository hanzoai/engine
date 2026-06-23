#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#![allow(dead_code)]

//! Whisper-style log-mel frontend for Qwen3-ASR. The HF processor is a
//! `WhisperFeatureExtractor` (n_fft=400, hop=160, 128 Slaney mel bins, centered
//! STFT, `log10` magnitudes clamped to `max - 8` then mapped to `(x + 4) / 4`).
//! Output is `[B, n_mels, T]`, the layout the AuT conv stem expects.

use hanzo_audio::AudioInput;
use hanzo_ml::{Device, Result, Tensor};
use rubato::Resampler;
use rustfft::{num_complex::Complex32, FftPlanner};

use super::config::AudioEncoderConfig;

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
            self.resample(&mono, audio.sample_rate, self.sampling_rate)?
        } else {
            mono
        };
        let mel = self.compute_mel(&samples)?; // [T][n_mels]
        let t = mel.len();
        if t == 0 {
            hanzo_ml::bail!("audio too short to produce mel frames");
        }
        // [T][n_mels] -> [n_mels, T] -> [1, n_mels, T].
        let mut data = vec![0f32; self.n_mels * t];
        for (ti, frame) in mel.iter().enumerate() {
            for (mi, &v) in frame.iter().enumerate() {
                data[mi * t + ti] = v;
            }
        }
        Tensor::from_vec(data, (1, self.n_mels, t), device).map_err(hanzo_ml::Error::msg)
    }

    fn resample(&self, samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
        if from == to {
            return Ok(samples.to_vec());
        }
        let params = rubato::SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: rubato::SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: rubato::WindowFunction::BlackmanHarris2,
        };
        let mut rs =
            rubato::SincFixedIn::<f32>::new(to as f64 / from as f64, 2.0, params, samples.len(), 1)
                .map_err(hanzo_ml::Error::msg)?;
        let out = rs
            .process(&[samples.to_vec()], None)
            .map_err(hanzo_ml::Error::msg)?;
        Ok(out[0].clone())
    }

    /// Centered STFT (`torch.stft(center=True)`: reflect-pad n_fft/2 each side,
    /// drop the last frame) + Slaney mel + Whisper log normalization.
    fn compute_mel(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let n_fft = self.n_fft;
        let hop = self.hop_length;
        let n_freqs = n_fft / 2 + 1;
        let pad = n_fft / 2;

        let padded_len = pad + samples.len() + pad;
        let mut padded = vec![0f32; padded_len];
        for (i, p) in padded.iter_mut().enumerate().take(pad) {
            p.clone_from(&samples[(pad - i).min(samples.len() - 1)]);
        }
        padded[pad..pad + samples.len()].copy_from_slice(samples);
        for i in 0..pad {
            padded[pad + samples.len() + i] = samples[samples.len().saturating_sub(2 + i)];
        }

        let total = (padded_len - n_fft) / hop + 1;
        let num_frames = total.saturating_sub(1);

        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos()))
            .collect();
        let filters = self.mel_filterbank(n_freqs);

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        let mut frames: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
        let mut frame_max = f32::MIN;
        for fi in 0..num_frames {
            let start = fi * hop;
            let mut buf: Vec<Complex32> = padded[start..start + n_fft]
                .iter()
                .zip(&window)
                .map(|(&s, &w)| Complex32::new(s * w, 0.0))
                .collect();
            fft.process(&mut buf);
            let power: Vec<f32> = buf[..n_freqs].iter().map(|c| c.norm_sqr()).collect();

            let mut mel = vec![0f32; self.n_mels];
            for (mi, filt) in filters.iter().enumerate() {
                let mut sum = 0f32;
                for (k, &c) in filt.iter().enumerate() {
                    sum += power[k] * c;
                }
                let logv = sum.max(1e-10).log10();
                mel[mi] = logv;
                if logv > frame_max {
                    frame_max = logv;
                }
            }
            frames.push(mel);
        }

        // Whisper: clamp to global max - 8, then (x + 4) / 4.
        let floor = frame_max - 8.0;
        for frame in &mut frames {
            for v in frame {
                *v = (v.max(floor) + 4.0) / 4.0;
            }
        }
        Ok(frames)
    }

    fn hertz_to_mel(f: f32) -> f32 {
        const MIN_LOG_HZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = 15.0;
        const LOGSTEP: f32 = 27.0 / 1.856_298;
        if f >= MIN_LOG_HZ {
            MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() * LOGSTEP
        } else {
            3.0 * f / 200.0
        }
    }

    fn mel_to_hertz(m: f32) -> f32 {
        const MIN_LOG_HZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = 15.0;
        const LOGSTEP: f32 = 1.856_298 / 27.0;
        if m >= MIN_LOG_MEL {
            MIN_LOG_HZ * (LOGSTEP * (m - MIN_LOG_MEL)).exp()
        } else {
            200.0 * m / 3.0
        }
    }

    /// Slaney-normalized triangular mel filterbank `[n_mels][n_freqs]`.
    fn mel_filterbank(&self, n_freqs: usize) -> Vec<Vec<f32>> {
        let sr = self.sampling_rate as f32;
        let n_mels = self.n_mels;
        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * (sr / 2.0) / (n_freqs - 1) as f32)
            .collect();
        let mel_min = Self::hertz_to_mel(0.0);
        let mel_max = Self::hertz_to_mel(sr / 2.0);
        let pts: Vec<f32> = (0..n_mels + 2)
            .map(|i| {
                Self::mel_to_hertz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            })
            .collect();
        let diff: Vec<f32> = pts.windows(2).map(|w| w[1] - w[0]).collect();

        let mut fb = vec![vec![0f32; n_freqs]; n_mels];
        for m in 0..n_mels {
            for (j, &f) in fft_freqs.iter().enumerate() {
                let down = (f - pts[m]) / diff[m];
                let up = (pts[m + 2] - f) / diff[m + 1];
                fb[m][j] = 0f32.max(down.min(up));
            }
            let enorm = 2.0 / (pts[m + 2] - pts[m]);
            for v in &mut fb[m] {
                *v *= enorm;
            }
        }
        fb
    }
}
