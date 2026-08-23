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

        // Constants shared by every forward + all step timesteps, materialised once off the hot loop.
        let ctx = self.transformer.sample_ctx(frames, ehs.dim(1)?, dev)?;
        let timesteps = Tensor::from_vec(sched.timesteps.clone(), (steps,), dev)?;

        let g_start = (steps as f64 * (1.0 - GUIDANCE_INTERVAL) / 2.0) as usize;
        let g_end = (steps as f64 * (GUIDANCE_INTERVAL / 2.0 + 0.5)) as usize;

        for i in 0..steps {
            let t = timesteps.narrow(0, i, 1)?;
            let v = if i >= g_start && i < g_end {
                let v_cond = self
                    .transformer
                    .decode_with_ctx(&latent, &ehs, None, &t, &ctx)?;
                let v_uncond = self
                    .transformer
                    .decode_with_ctx(&latent, &ehs_null, None, &t, &ctx)?;
                apg_forward(&v_cond, &v_uncond, guidance_scale, &mut mb)?
            } else {
                self.transformer
                    .decode_with_ctx(&latent, &ehs, None, &t, &ctx)?
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
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
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
        out.extend_from_slice(&(sr * bytes_per * ch as u32).to_le_bytes());
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

    // GB10 end-to-end music-gen real-time factor. Seeded so the same config is bit-reproducible for an
    // A/B waveform correlation across a rebuild. ACE_BENCH=1 gated; needs a CUDA build + full fixtures.
    #[test]
    fn music_generate_cuda_bench() {
        use std::time::Instant;
        if std::env::var("ACE_BENCH").is_err() {
            eprintln!("music_generate_cuda_bench: set ACE_BENCH=1 to run");
            return;
        }
        let fx = std::env::var("ACE_FIX_DIR").unwrap_or_else(|_| "/data/ace-bench-fix".to_string());
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
            eprintln!("ACE-Step fixtures absent; skipping generate bench");
            return;
        }
        let device = Device::new_cuda(0).unwrap();
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
        let mut cfg: Config = serde_json::from_str(
            &std::fs::read_to_string(format!("{fx}/umt5_config.json")).unwrap(),
        )
        .unwrap();
        cfg.umt5 = true;
        let mut text_encoder = Umt5TextEncoder::new(
            &cfg,
            load(format!("{fx}/umt5.safetensors"), all.clone()),
            &device,
        )
        .unwrap();
        let dit_dtype = match std::env::var("ACE_DIT_DTYPE")
            .unwrap_or_else(|_| "f32".to_string())
            .as_str()
        {
            "f32" => DType::F32,
            "bf16" => DType::BF16,
            "f16" => DType::F16,
            o => panic!("ACE_DIT_DTYPE must be f32|bf16|f16, got {o}"),
        };
        let dit_vb = from_mmaped_safetensors(
            vec![PathBuf::from(format!("{fx}/dit.safetensors"))],
            Vec::new(),
            Some(dit_dtype),
            &device,
            vec![None],
            true,
            None,
            move |n: String| {
                !n.contains(".add_")
                    && !n.contains(".to_add_out")
                    && !n.starts_with("lyric")
                    && !n.starts_with("projectors")
            },
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap();
        let transformer = AceStepTransformer::new(dit_vb).unwrap();
        let dcae = MusicDcae::new(
            load(format!("{fx}/dcae.safetensors"), dec),
            load(format!("{fx}/vocoder.safetensors"), all),
        )
        .unwrap();

        let ids: Vec<u32> = read_i64_le(&format!("{fx}/umt5_ids.i64"))
            .iter()
            .map(|&v| v as u32)
            .collect();
        let n = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, n), &device).unwrap();

        let frames: usize = std::env::var("ACE_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(108);
        let steps: usize = std::env::var("ACE_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(27);

        // Per-DiT-forward microbench (the sampler's hot inner call), isolated from text-enc + DCAE.
        {
            let text_hidden = text_encoder.encode(&input_ids).unwrap();
            let speaker = Tensor::zeros((1, SPEAKER_DIM), text_hidden.dtype(), &device).unwrap();
            let ehs = transformer.encode(&text_hidden, &speaker).unwrap();
            let ctx = transformer
                .sample_ctx(frames, ehs.dim(1).unwrap(), &device)
                .unwrap();
            device.set_seed(1234).unwrap();
            let latent = Tensor::randn(
                0f32,
                1.0,
                (1, LATENT_CHANNELS, LATENT_HEIGHT, frames),
                &device,
            )
            .unwrap();
            let ts = Tensor::from_vec(vec![500f32], (1,), &device).unwrap();
            for _ in 0..5 {
                let _ = transformer
                    .decode_with_ctx(&latent, &ehs, None, &ts, &ctx)
                    .unwrap();
            }
            device.synchronize().unwrap();
            let mut ms = Vec::with_capacity(41);
            let mut vel0: Option<Vec<f32>> = None;
            for i in 0..41 {
                let t0 = Instant::now();
                let v = transformer
                    .decode_with_ctx(&latent, &ehs, None, &ts, &ctx)
                    .unwrap();
                device.synchronize().unwrap();
                ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                if i == 0 {
                    vel0 = Some(v.flatten_all().unwrap().to_vec1().unwrap());
                }
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = ms.iter().sum::<f64>() / ms.len() as f64;
            println!(
                "DIT-BENCH dtype={dit_dtype:?} per_forward_ms median={:.2} mean={mean:.2} min={:.2} max={:.2}",
                ms[ms.len() / 2],
                ms[0],
                ms[ms.len() - 1]
            );
            if let Ok(p) = std::env::var("ACE_AB_VEL") {
                let flat = vel0.unwrap();
                let bytes: Vec<u8> = flat.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(&p, bytes).unwrap();
                println!("wrote DiT velocity A/B {p} ({} f32)", flat.len());
            }
        }

        let mut pipe = AceStepPipeline::new(text_encoder, transformer, dcae, device.clone());

        device.set_seed(1234).unwrap();
        let _ = pipe.generate(&input_ids, 32, 4, 15.0).unwrap();
        device.synchronize().unwrap();

        device.set_seed(1234).unwrap();
        device.synchronize().unwrap();
        let t = Instant::now();
        let wav = pipe.generate(&input_ids, frames, steps, 15.0).unwrap();
        device.synchronize().unwrap();
        let wall = t.elapsed().as_secs_f64();
        let (_, _, s) = wav.dims3().unwrap();
        let audio_s = s as f64 / 44100.0;
        println!(
            "GEN-BENCH dtype={dit_dtype:?} frames={frames} steps={steps} audio={audio_s:.3}s wall={wall:.3}s realtime={:.3}x",
            audio_s / wall
        );

        if let Ok(ab) = std::env::var("ACE_AB_GEN") {
            let flat: Vec<f32> = wav.flatten_all().unwrap().to_vec1().unwrap();
            let bytes: Vec<u8> = flat.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(&ab, bytes).unwrap();
            println!(
                "music_generate_cuda_bench wrote A/B waveform {ab} ({} samples)",
                flat.len()
            );
        }
    }
}
