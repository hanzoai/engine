//! ACE-Step music generation (Apache-2.0): text -> song with vocals.
//! DiT over DCAE music latents + HiFi-GAN vocoder. This module ports the
//! latent -> waveform tail (DCAE decode + vocoder) first; the text encoder,
//! DiT and flow sampler land alongside as they parity-gate.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
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

    // GB10 profile of the DCAE decode path: isolate the grouped-conv DcaeDecoder (stage3 storm) from
    // the vocoder, report the decode-time split, and dump the reference-latent waveform for A/B
    // correlation across a rebuild. ACE_BENCH=1 gated; needs a CUDA build + the DCAE/vocoder fixtures.
    #[test]
    fn dcae_profile_cuda() {
        use std::time::Instant;
        if std::env::var("ACE_BENCH").is_err() {
            eprintln!("dcae_profile_cuda: set ACE_BENCH=1 to run");
            return;
        }
        let dcae_p = std::env::var("ACE_DCAE_ST")
            .unwrap_or_else(|_| "/data/ace-fixtures/dcae.safetensors".to_string());
        let voc_p = std::env::var("ACE_VOC_ST")
            .unwrap_or_else(|_| "/data/ace-fixtures/vocoder.safetensors".to_string());
        let fix = std::env::var("ACE_FIX_DIR").unwrap_or_else(|_| "/data/ace-fixtures".to_string());
        if !PathBuf::from(&dcae_p).is_file() {
            eprintln!("DCAE fixtures absent; skipping profile");
            return;
        }
        let device = Device::new_cuda(0).unwrap();
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
        let model = MusicDcae::new(load(&dcae_p, "decoder."), load(&voc_p, "")).unwrap();

        let frames: usize = std::env::var("ACE_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(108);
        let iters: usize = std::env::var("ACE_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        let latent = Tensor::randn(0f32, 1.0, (1, 8, 16, frames), &device).unwrap();
        let scaled = ((&latent / SCALE_FACTOR).unwrap() + SHIFT_FACTOR).unwrap();

        for _ in 0..3 {
            let _ = model.decode(&latent).unwrap();
        }
        device.synchronize().unwrap();

        let t = Instant::now();
        let mut mel = model.dcae.decode(&scaled).unwrap();
        for _ in 1..iters {
            mel = model.dcae.decode(&scaled).unwrap();
        }
        device.synchronize().unwrap();
        let dcae_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let m = ((mel * 0.5).unwrap() + 0.5).unwrap();
        let m = ((m * (MAX_MEL - MIN_MEL)).unwrap() + MIN_MEL).unwrap();
        let ch0 = m.narrow(1, 0, 1).unwrap().squeeze(1).unwrap();
        let ch1 = m.narrow(1, 1, 1).unwrap().squeeze(1).unwrap();
        device.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            let _ = model.vocoder.decode(&ch0).unwrap();
            let _ = model.vocoder.decode(&ch1).unwrap();
        }
        device.synchronize().unwrap();
        let voc_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        device.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            let _ = model.decode(&latent).unwrap();
        }
        device.synchronize().unwrap();
        let full_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let audio_s = (frames * 8 * 512) as f64 / 44100.0;
        println!(
            "DCAE-PROFILE frames={frames} audio={audio_s:.3}s iters={iters} | dcae_decoder={dcae_ms:.2}ms vocoder={voc_ms:.2}ms full={full_ms:.2}ms | dcae_share={:.1}%",
            100.0 * dcae_ms / full_ms
        );

        if let Ok(ab) = std::env::var("ACE_AB_OUT") {
            let rl = Tensor::from_vec(read_f32_le(&format!("{fix}/latent.f32")), (1, 8, 16, 32), &device)
                .unwrap();
            let wav = model.decode(&rl).unwrap();
            let flat: Vec<f32> = wav.flatten_all().unwrap().to_vec1().unwrap();
            let bytes: Vec<u8> = flat.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(&ab, bytes).unwrap();
            println!("dcae_profile_cuda wrote A/B waveform {ab} ({} samples)", flat.len());
        }
    }
}
