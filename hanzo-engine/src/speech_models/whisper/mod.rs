#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! openai-whisper-tiny audio encoder used as MuseTalk's lip-sync condition.
//! MuseTalk's UNet was trained on the stacked encoder embeddings (post-conv map
//! + every block output) of openai-whisper-tiny, sliced into a per-video-frame
//! window. `WhisperFeatureExtractor::features` reproduces that exact tensor.

pub mod encoder;
pub mod mel;

pub use encoder::WhisperEncoder;

use hanzo_ml::{Device, Result, Tensor};
use hanzo_quant::ShardedVarBuilder;

use crate::speech_models::utils;

pub const WHISPER_SAMPLE_RATE: u32 = 16_000;
const FEATURE_FPS: usize = 50; // post-conv whisper feature frames per second (100 mel fps / 2)
const AUDIO_FEAT_PAD: usize = 2; // MuseTalk audio_feat_length = [2, 2]
const SECONDS_PER_WINDOW: usize = 30; // openai whisper encoder window length

pub struct WhisperConfig {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
}

impl WhisperConfig {
    pub fn tiny() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
        }
    }

    /// Stacked encoder maps per feature frame: the post-conv embedding plus one
    /// per transformer block (5 for tiny).
    pub fn stack(&self) -> usize {
        self.n_audio_layer + 1
    }

    /// MuseTalk window width in feature frames: `[2,2]` -> 10 (4 left + 6 right).
    pub fn window_len(&self) -> usize {
        4 * AUDIO_FEAT_PAD + 2
    }

    /// Per-frame audio-feature sequence length the UNet cross-attends to
    /// (`window_len * stack`; 50 for tiny).
    pub fn audio_seq_len(&self) -> usize {
        self.window_len() * self.stack()
    }
}

pub struct WhisperFeatureExtractor {
    encoder: WhisperEncoder,
    cfg: WhisperConfig,
    device: Device,
}

impl WhisperFeatureExtractor {
    pub fn new(cfg: WhisperConfig, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        let encoder = WhisperEncoder::new(&cfg, vb)?;
        Ok(Self {
            encoder,
            cfg,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.cfg
    }

    /// MuseTalk audio features `[T, audio_seq_len, n_audio_state]` with
    /// `T = ceil(secs * fps)`. `pcm` at any rate is resampled to 16 kHz; `secs`
    /// is taken from the input pcm so it is resample-invariant.
    pub fn features(&self, pcm: &[f32], sample_rate: usize, fps: f64) -> Result<Tensor> {
        let secs = pcm.len() as f64 / sample_rate as f64;
        let n_frames = ((secs * fps).ceil() as usize).max(1);

        let mut pcm16 = utils::resample(pcm, sample_rate as u32, WHISPER_SAMPLE_RATE)?;
        // openai pads audio to whole 30 s windows; the zero tail becomes mel-floor silence.
        let samples_per_window = SECONDS_PER_WINDOW * WHISPER_SAMPLE_RATE as usize;
        let n_win = pcm16.len().div_ceil(samples_per_window).max(1);
        pcm16.resize(n_win * samples_per_window, 0.0);

        let mel = mel::log_mel_spectrogram(&pcm16, self.cfg.n_mels, &self.device)?;
        let mel_per_window = 2 * self.cfg.n_audio_ctx;

        // Each 30 s window is encoded independently (faithful to openai's chunking).
        let mut windows = Vec::with_capacity(n_win);
        for w in 0..n_win {
            let win = mel.narrow(1, w * mel_per_window, mel_per_window)?.unsqueeze(0)?;
            windows.push(self.encoder.encode_window(&win)?);
        }
        let feats = if windows.len() == 1 {
            windows.pop().unwrap()
        } else {
            Tensor::cat(&windows, 1)?
        };

        // Trim padded-tail features to real content; boundary windows clamp-repeat.
        let total_feat = feats.dim(1)?;
        let n_content = ((secs * FEATURE_FPS as f64).ceil() as usize).clamp(1, total_feat);
        let feats = feats.narrow(1, 0, n_content)?.transpose(0, 1)?.contiguous()?;

        // MuseTalk get_sliced_feature, vectorized: gather a clamped `window_len`
        // run of feature frames centered on `vid * 50 / fps` for every video frame.
        let stack = self.cfg.stack();
        let d = self.cfg.n_audio_state;
        let window_len = self.cfg.window_len();
        let left_off = (2 * AUDIO_FEAT_PAD) as i64;
        let max_idx = n_content as i64 - 1;
        let mut idx = Vec::with_capacity(n_frames * window_len);
        for vid in 0..n_frames {
            let center = (vid as f64 * FEATURE_FPS as f64 / fps) as i64;
            for k in 0..window_len as i64 {
                idx.push((center - left_off + k).clamp(0, max_idx) as u32);
            }
        }
        let idx = Tensor::from_vec(idx, n_frames * window_len, &self.device)?;
        let gathered = feats.index_select(&idx, 0)?;
        gathered.reshape((n_frames, window_len * stack, d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::{DType, Shape};
    use hanzo_nn::var_builder::SimpleBackend;
    use hanzo_nn::Init;
    use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};

    struct RandnBackend;
    impl SimpleBackend for RandnBackend {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _h: Init,
            dtype: DType,
            dev: &Device,
        ) -> Result<Tensor> {
            if name.ends_with("bias") {
                Tensor::zeros(s, dtype, dev)
            } else if name.ends_with("weight") && s.rank() == 1 {
                Tensor::ones(s, dtype, dev)
            } else {
                Tensor::randn(0f64, 0.02, s, dev)?.to_dtype(dtype)
            }
        }
        fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
            hanzo_ml::bail!("RandnBackend requires an explicit shape")
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    fn random_vb(dtype: DType, dev: &Device) -> ShardedVarBuilder {
        ShardedSafeTensors::wrap(Box::new(RandnBackend), dtype, dev.clone())
    }

    fn sine(freq: f32, secs: f32, sr: usize) -> Vec<f32> {
        let n = (secs * sr as f32) as usize;
        (0..n)
            .map(|i| (i as f32 * freq * 2.0 * std::f32::consts::PI / sr as f32).sin() * 0.3)
            .collect()
    }

    #[test]
    fn feature_shape_finite() -> Result<()> {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let cfg = WhisperConfig::tiny();
        let seq = cfg.audio_seq_len();
        let d = cfg.n_audio_state;
        let extractor = WhisperFeatureExtractor::new(cfg, random_vb(dtype, &dev), &dev)?;

        // 1 s of 24 kHz (omni rate) -> resampled to 16 k -> features at 25 fps.
        let pcm = sine(220.0, 1.0, 24_000);
        let fps = 25.0;
        let feats = extractor.features(&pcm, 24_000, fps)?;
        let t = (1.0_f64 * fps).ceil() as usize;
        assert_eq!(feats.dims(), &[t, seq, d], "whisper feature tensor shape");

        let v = feats.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            v.iter().all(|x| x.is_finite()),
            "non-finite whisper features"
        );
        let (mn, mx) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        assert!(mx != mn, "whisper features are constant (degenerate)");

        let feats2 = extractor.features(&pcm, 24_000, fps)?;
        assert_eq!(
            v,
            feats2.flatten_all()?.to_vec1::<f32>()?,
            "whisper features not deterministic"
        );
        Ok(())
    }
}
