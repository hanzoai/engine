#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::io::Write;

use hanzo_ml::{Device, Result, Tensor};
use rubato::Resampler;
use rustfft::{num_complex::Complex32, FftPlanner};

use super::bs1770;

const MEL_LOG_FLOOR: f32 = 1e-10;
const MEL_DYNAMIC_RANGE: f32 = 8.0;

/// Sinc resample `samples` from `from` Hz to `to` Hz. The one resampler in the
/// speech stack; Whisper/ASR frontends and the dub speak->animate hop all route
/// PCM through here so rate conversion is identical everywhere.
pub(crate) fn resample(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
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

/// openai-Whisper / HF-WhisperFeatureExtractor log-mel: centered Hann STFT
/// (`torch.stft(center=True)`: reflect-pad n_fft/2 each side, drop the last
/// frame), Slaney triangular mel, `log10` magnitudes clamped to `max - 8` then
/// mapped to `(x + 4) / 4`. Returns `[T][n_mels]`. Both Qwen3-ASR (128 bins) and
/// openai-whisper-tiny (80 bins) are this exact frontend at different bin counts.
pub(crate) fn log_mel(
    samples: &[f32],
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    sample_rate: u32,
) -> Vec<Vec<f32>> {
    if samples.is_empty() {
        return Vec::new();
    }
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
    let filters = mel_filterbank(n_mels, n_freqs, sample_rate);

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

        let mut mel = vec![0f32; n_mels];
        for (mi, filt) in filters.iter().enumerate() {
            let mut sum = 0f32;
            for (k, &c) in filt.iter().enumerate() {
                sum += power[k] * c;
            }
            let logv = sum.max(MEL_LOG_FLOOR).log10();
            mel[mi] = logv;
            if logv > frame_max {
                frame_max = logv;
            }
        }
        frames.push(mel);
    }

    let floor = frame_max - MEL_DYNAMIC_RANGE;
    for frame in &mut frames {
        for v in frame {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
    }
    frames
}

/// `log_mel` packed into a `[n_mels, T]` tensor on `device` (the layout a conv
/// stem expects). Errors only when the clip is too short to yield any frame.
pub(crate) fn log_mel_tensor(
    samples: &[f32],
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    sample_rate: u32,
    device: &Device,
) -> Result<Tensor> {
    let mel = log_mel(samples, n_fft, hop, n_mels, sample_rate);
    let t = mel.len();
    if t == 0 {
        hanzo_ml::bail!("audio too short to produce mel frames");
    }
    let mut data = vec![0f32; n_mels * t];
    for (ti, frame) in mel.iter().enumerate() {
        for (mi, &v) in frame.iter().enumerate() {
            data[mi * t + ti] = v;
        }
    }
    Tensor::from_vec(data, (n_mels, t), device)
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
fn mel_filterbank(n_mels: usize, n_freqs: usize, sample_rate: u32) -> Vec<Vec<f32>> {
    let sr = sample_rate as f32;
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * (sr / 2.0) / (n_freqs - 1) as f32)
        .collect();
    let mel_min = hertz_to_mel(0.0);
    let mel_max = hertz_to_mel(sr / 2.0);
    let pts: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hertz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
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

pub(crate) fn normalize_loudness(
    wav: &Tensor,
    sample_rate: u32,
    loudness_compressor: bool,
) -> Result<Tensor> {
    let energy = wav.sqr()?.mean_all()?.sqrt()?.to_vec0::<f32>()?;
    if energy < 2e-3 {
        return Ok(wav.clone());
    }
    let wav_array = wav.to_vec1::<f32>()?;
    let mut meter = bs1770::ChannelLoudnessMeter::new(sample_rate);
    meter.push(wav_array.into_iter());
    let power = meter.as_100ms_windows();
    let loudness = match bs1770::gated_mean(power) {
        None => return Ok(wav.clone()),
        Some(gp) => gp.loudness_lkfs() as f64,
    };
    let delta_loudness = -14. - loudness;
    let gain = 10f64.powf(delta_loudness / 20.);
    let wav = (wav * gain)?;
    if loudness_compressor {
        wav.tanh()
    } else {
        Ok(wav)
    }
}

pub trait Sample {
    fn to_i16(&self) -> i16;
}

impl Sample for f32 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}

impl Sample for f64 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}

impl Sample for i16 {
    fn to_i16(&self) -> i16 {
        *self
    }
}

pub fn write_pcm_as_wav<W: Write, S: Sample>(
    w: &mut W,
    samples: &[S],
    sample_rate: u32,
    n_channels: u16,
) -> std::io::Result<()> {
    let len = 12u32; // header
    let len = len + 24u32; // fmt
    let len = len + samples.len() as u32 * 2 + 8; // data
    let bytes_per_second = sample_rate * 2 * n_channels as u32;
    w.write_all(b"RIFF")?;
    w.write_all(&(len - 8).to_le_bytes())?; // total length minus 8 bytes
    w.write_all(b"WAVE")?;

    // Format block
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // block len minus 8 bytes
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&n_channels.to_le_bytes())?; // one channel
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&bytes_per_second.to_le_bytes())?;
    let block_align = 2 * n_channels;
    w.write_all(&block_align.to_le_bytes())?; // 2 bytes of data per sample
    w.write_all(&16u16.to_le_bytes())?; // bits per sample

    // Data block
    w.write_all(b"data")?;
    w.write_all(&(samples.len() as u32 * 2).to_le_bytes())?;
    for sample in samples.iter() {
        w.write_all(&sample.to_i16().to_le_bytes())?
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, secs: f32, sr: u32) -> Vec<f32> {
        let n = (secs * sr as f32) as usize;
        (0..n)
            .map(|i| (i as f32 * freq * 2.0 * std::f32::consts::PI / sr as f32).sin() * 0.5)
            .collect()
    }

    #[test]
    fn resample_changes_length_and_passes_through() {
        let x = sine(440.0, 0.5, 24_000);
        let y = resample(&x, 24_000, 16_000).unwrap();
        assert!(!y.is_empty());
        // 24k -> 16k is a 2/3 ratio; allow sinc edge slack.
        let expected = x.len() * 16_000 / 24_000;
        assert!((y.len() as i64 - expected as i64).abs() < 512);
        // identity rate is a no-copy pass-through.
        let z = resample(&x, 16_000, 16_000).unwrap();
        assert_eq!(z, x);
    }

    #[test]
    fn log_mel_concentrates_sine_energy() {
        let sr = 16_000u32;
        let mel = log_mel(&sine(1000.0, 1.0, sr), 400, 160, 80, sr);
        assert!(!mel.is_empty());
        assert!(mel.iter().all(|f| f.len() == 80));
        for f in &mel {
            assert!(f.iter().all(|v| v.is_finite()));
        }
        // average each bin across time; the peak bin must sit near 1 kHz, not at
        // a low/high extreme -- this catches a transposed or mis-scaled filterbank.
        let t = mel.len() as f32;
        let avg: Vec<f32> = (0..80)
            .map(|m| mel.iter().map(|f| f[m]).sum::<f32>() / t)
            .collect();
        let peak = avg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            (20..=45).contains(&peak),
            "1 kHz sine peaked at mel bin {peak}, expected mid-band (20..=45)"
        );
    }
}
