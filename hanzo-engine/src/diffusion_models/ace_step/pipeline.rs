#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
//! ACE-Step end-to-end: prompt tokens -> DiT flow-match sampling -> DCAE -> stereo waveform.

use hanzo_ml::{Device, Result, Tensor};

use super::scheduler::{apg_forward, FlowMatchScheduler, MomentumBuffer};
use super::text_encoder::Umt5TextEncoder;
use super::transformer::AceStepTransformer;
use super::MusicDcae;

const OMEGA_SCALE: f64 = 10.0;
const GUIDANCE_INTERVAL: f64 = 0.5;
const SPEAKER_DIM: usize = 512;
const LATENT_CHANNELS: usize = 8;
const LATENT_HEIGHT: usize = 16;

pub struct AceStepPipeline {
    text_encoder: Umt5TextEncoder,
    transformer: AceStepTransformer,
    dcae: MusicDcae,
    device: Device,
}

impl AceStepPipeline {
    pub fn new(
        text_encoder: Umt5TextEncoder,
        transformer: AceStepTransformer,
        dcae: MusicDcae,
        device: Device,
    ) -> Self {
        Self {
            text_encoder,
            transformer,
            dcae,
            device,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// input_ids (1, T_text) u32 -> stereo waveform (1, 2, samples) at 44.1kHz.
    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        frames: usize,
        steps: usize,
        guidance_scale: f64,
    ) -> Result<Tensor> {
        let dev = &self.device;
        let text_hidden = self.text_encoder.encode(input_ids)?;
        let (_, t_text, _) = text_hidden.dims3()?;
        let speaker = Tensor::zeros((1, SPEAKER_DIM), text_hidden.dtype(), dev)?;
        let ehs = self.transformer.encode(&text_hidden, &speaker)?;
        let text_null = Tensor::zeros((1, t_text, text_hidden.dim(2)?), text_hidden.dtype(), dev)?;
        let ehs_null = self.transformer.encode(&text_null, &speaker)?;

        let sched = FlowMatchScheduler::new(steps, OMEGA_SCALE);
        let mut latent =
            Tensor::randn(0f32, 1.0, (1, LATENT_CHANNELS, LATENT_HEIGHT, frames), dev)?;
        let mut mb = MomentumBuffer::new();

        let g_start = (steps as f64 * (1.0 - GUIDANCE_INTERVAL) / 2.0) as usize;
        let g_end = (steps as f64 * (GUIDANCE_INTERVAL / 2.0 + 0.5)) as usize;

        for i in 0..steps {
            let t = Tensor::from_vec(vec![sched.timesteps[i]], (1,), dev)?;
            let v = if i >= g_start && i < g_end {
                let v_cond = self.transformer.decode(&latent, &ehs, None, &t)?;
                let v_uncond = self.transformer.decode(&latent, &ehs_null, None, &t)?;
                apg_forward(&v_cond, &v_uncond, guidance_scale, &mut mb)?
            } else {
                self.transformer.decode(&latent, &ehs, None, &t)?
            };
            latent = sched.step(&v, &latent, i)?;
        }

        self.dcae.decode(&latent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion_models::ace_step::MusicDcae;
    use crate::diffusion_models::t5::Config;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use hanzo_ml::{DType, IndexOp};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn read_i64_le(p: &str) -> Vec<i64> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn write_wav16(path: &str, stereo: &[Vec<f32>], sr: u32) {
        let n = stereo[0].len();
        let bytes_per = 2u32;
        let ch = stereo.len() as u16;
        let data_len = (n as u32) * bytes_per * ch as u32;
        let mut out = Vec::with_capacity(44 + data_len as usize);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&ch.to_le_bytes());
        out.extend_from_slice(&sr.to_le_bytes());
        out.extend_from_slice(&(sr * bytes_per as u32 * ch as u32).to_le_bytes());
        out.extend_from_slice(&(bytes_per as u16 * ch).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for i in 0..n {
            for c in stereo {
                let s = (c[i].clamp(-1.0, 1.0) * 32767.0) as i16;
                out.extend_from_slice(&s.to_le_bytes());
            }
        }
        std::fs::write(path, out).unwrap();
    }

    // Full prompt->music path (short clip) with the real weights: non-NaN + saved .wav artifact.
    #[test]
    fn ace_step_generate_e2e() {
        let fx = std::env::var("ACE_FIX_DIR")
            .unwrap_or_else(|_| "/home/z/work/hanzo/ace-fixtures".to_string());
        let need = [
            "umt5.safetensors",
            "dit.safetensors",
            "dcae.safetensors",
            "vocoder.safetensors",
            "umt5_ids.i64",
            "umt5_config.json",
        ];
        if need
            .iter()
            .any(|f| !PathBuf::from(format!("{fx}/{f}")).is_file())
        {
            eprintln!("ACE-Step e2e fixtures absent; skipping generation");
            return;
        }
        let device = Device::Cpu;
        let load = |p: String, keep: Arc<dyn Fn(&str) -> bool + Send + Sync>| {
            from_mmaped_safetensors(
                vec![PathBuf::from(p)],
                Vec::new(),
                Some(DType::F32),
                &device,
                vec![None],
                true,
                None,
                move |n: String| keep(&n),
                Arc::new(|_| DeviceForLoadTensor::Base),
            )
            .unwrap()
        };
        let all: Arc<dyn Fn(&str) -> bool + Send + Sync> = Arc::new(|_| true);
        let dec: Arc<dyn Fn(&str) -> bool + Send + Sync> = Arc::new(|n| n.starts_with("decoder."));
        let dit_keep: Arc<dyn Fn(&str) -> bool + Send + Sync> = Arc::new(|n| {
            !n.contains(".add_")
                && !n.contains(".to_add_out")
                && !n.starts_with("lyric")
                && !n.starts_with("projectors")
        });

        let mut cfg: Config = serde_json::from_str(
            &std::fs::read_to_string(format!("{fx}/umt5_config.json")).unwrap(),
        )
        .unwrap();
        cfg.umt5 = true;
        let text_encoder = Umt5TextEncoder::new(
            &cfg,
            load(format!("{fx}/umt5.safetensors"), all.clone()),
            &device,
        )
        .unwrap();
        let transformer =
            AceStepTransformer::new(load(format!("{fx}/dit.safetensors"), dit_keep)).unwrap();
        let dcae = MusicDcae::new(
            load(format!("{fx}/dcae.safetensors"), dec),
            load(format!("{fx}/vocoder.safetensors"), all),
        )
        .unwrap();
        let mut pipe = AceStepPipeline::new(text_encoder, transformer, dcae, device.clone());

        let ids: Vec<u32> = read_i64_le(&format!("{fx}/umt5_ids.i64"))
            .iter()
            .map(|&v| v as u32)
            .collect();
        let n = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, n), &device).unwrap();

        let wav = pipe.generate(&input_ids, 32, 8, 15.0).unwrap();
        let flat: Vec<f32> = wav.flatten_all().unwrap().to_vec1().unwrap();
        let finite = flat.iter().all(|x| x.is_finite());
        let rms =
            (flat.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / flat.len() as f64).sqrt();
        println!(
            "ace e2e wav shape {:?} finite={finite} rms={rms:.5}",
            wav.dims()
        );
        assert!(finite, "generated waveform has NaN/inf");
        assert!(rms > 1e-5, "generated waveform is silent");

        let (_, ch, s) = wav.dims3().unwrap();
        let stereo: Vec<Vec<f32>> = (0..ch)
            .map(|c| wav.i((0, c)).unwrap().to_vec1::<f32>().unwrap())
            .collect();
        let _ = s;
        let out = std::env::var("ACE_WAV_OUT").unwrap_or_else(|_| format!("{fx}/ace_e2e.wav"));
        write_wav16(&out, &stereo, 44100);
        println!("wrote {out}");
    }
}
