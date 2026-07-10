//! ACE-Step music generation (Apache-2.0): text -> song with vocals.
//! DiT over DCAE music latents + HiFi-GAN vocoder. This module ports the
//! latent -> waveform tail (DCAE decode + vocoder) first; the text encoder,
//! DiT and flow sampler land alongside as they parity-gate.

pub mod dcae;
pub mod pipeline;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vocoder;

use hanzo_ml::{Result, Tensor};
use hanzo_quant::ShardedVarBuilder;

use dcae::DcaeDecoder;
use vocoder::Vocoder;

// MusicDCAE latent scaling and log-mel normalisation constants
// (acestep/music_dcae/music_dcae_pipeline.py).
const SCALE_FACTOR: f64 = 0.1786;
const SHIFT_FACTOR: f64 = -1.9091;
const MIN_MEL: f64 = -11.0;
const MAX_MEL: f64 = 3.0;

/// DCAE decoder + vocoder: scaled music latent -> stereo 44.1kHz waveform.
#[derive(Debug, Clone)]
pub struct MusicDcae {
    dcae: DcaeDecoder,
    vocoder: Vocoder,
}

impl MusicDcae {
    /// `dcae_vb` is the root of `music_dcae_f8c8` (keys `decoder.*`/`encoder.*`);
    /// `vocoder_vb` is the root of `music_vocoder` (keys `backbone.*`/`head.*`).
    pub fn new(dcae_vb: ShardedVarBuilder, vocoder_vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            dcae: DcaeDecoder::new(dcae_vb.pp("decoder"))?,
            vocoder: Vocoder::new(vocoder_vb)?,
        })
    }

    /// latents (B, 8, H, W), scaled -> waveform (B, 2, samples) at 44.1kHz.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let latents = ((latents / SCALE_FACTOR)? + SHIFT_FACTOR)?;
        let mels = self.dcae.decode(&latents)?;
        let mels = ((mels * 0.5)? + 0.5)?;
        let mels = ((mels * (MAX_MEL - MIN_MEL))? + MIN_MEL)?;
        let ch0 = mels.narrow(1, 0, 1)?.squeeze(1)?;
        let ch1 = mels.narrow(1, 1, 1)?.squeeze(1)?;
        let w0 = self.vocoder.decode(&ch0)?;
        let w1 = self.vocoder.decode(&ch1)?;
        Tensor::cat(&[&w0, &w1], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use hanzo_ml::{DType, Device};
    use std::path::PathBuf;
    use std::sync::Arc;

    const LATENT_DIMS: (usize, usize, usize, usize) = (1, 8, 16, 32);

    fn read_f32_le(p: &str) -> Vec<f32> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn corr(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for i in 0..n {
            d += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        (d / (na.sqrt() * nb.sqrt())) as f32
    }

    // Decode a fixed latent with the real DCAE + vocoder weights and assert the stereo waveform
    // matches the ACE-Step MusicDCAE reference (correlation > 0.99). Env-gated on the fixtures.
    #[test]
    fn music_dcae_decode_matches_reference() {
        let dcae = std::env::var("ACE_DCAE_ST")
            .unwrap_or_else(|_| "/data/ace-fixtures/dcae.safetensors".to_string());
        let voc = std::env::var("ACE_VOC_ST")
            .unwrap_or_else(|_| "/data/ace-fixtures/vocoder.safetensors".to_string());
        let fix = std::env::var("ACE_FIX_DIR").unwrap_or_else(|_| "/data/ace-fixtures".to_string());
        let latent_f = format!("{fix}/latent.f32");
        let wav_f = format!("{fix}/wav.f32");
        if !PathBuf::from(&dcae).is_file() || !PathBuf::from(&wav_f).is_file() {
            eprintln!("ACE-Step DCAE fixtures absent; skipping decode validation");
            return;
        }
        let device = Device::Cpu;
        let load = |p: &str, pref: &'static str| {
            from_mmaped_safetensors(
                vec![PathBuf::from(p)],
                Vec::new(),
                Some(DType::F32),
                &device,
                vec![None],
                true,
                None,
                move |n: String| n.starts_with(pref),
                Arc::new(|_| DeviceForLoadTensor::Base),
            )
            .unwrap()
        };
        let dcae_vb = load(&dcae, "decoder.");
        let voc_vb = load(&voc, "");
        let model = MusicDcae::new(dcae_vb, voc_vb).unwrap();

        let latent = Tensor::from_vec(read_f32_le(&latent_f), LATENT_DIMS, &device).unwrap();
        let wav = model.decode(&latent).unwrap();
        let got: Vec<f32> = wav.flatten_all().unwrap().to_vec1().unwrap();
        let want = read_f32_le(&wav_f);
        let c = corr(&got, &want);
        println!(
            "music_dcae decode correlation = {c:.5} (n_got={}, n_want={})",
            got.len(),
            want.len()
        );
        assert!(c > 0.99, "waveform correlation {c} below 0.99");
    }
}
