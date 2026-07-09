//! GPU (Vulkan/ROCm) vs CPU wall-clock benchmark for the native MuseTalk dub.
//!
//! Loads the REAL staged checkpoints (MuseTalk v1.5 UNet, sd-vae-ft-mse, openai-whisper-tiny,
//! S3FD) onto the selected device, builds the `MuseTalkAnimator`, and times `animate()` over a
//! fixed number of output frames. The device is picked by cargo feature:
//!   default            -> CPU
//!   --features vulkan  -> Vulkan:0
//!   --features rocm    -> ROCm:0
//!   --features cuda    -> CUDA:0
//!
//! Env-gated on the weights so a box without them is a clean no-op (CI-safe). Override any path:
//!   MT_DIR VAE_DIR WHISPER_ST S3FD_ST DEMO_PNG SUN_WAV DUB_FRAMES
//!
//! Run (this box):
//!   CPU:     cargo test -p hanzo-engine --test dub_gpu_bench --release -- --nocapture --test-threads=1
//!   Vulkan:  cargo test -p hanzo-engine --features vulkan --test dub_gpu_bench --release -- --nocapture --test-threads=1
//!   ROCm:    cargo test -p hanzo-engine --features rocm --test dub_gpu_bench --release -- --nocapture --test-threads=1

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use hanzo_engine::diffusion_models::animation::{
    AnimationRequest, DrivingAudio, FacialAnimator, VisualSource,
};
use hanzo_engine::diffusion_models::musetalk::{
    AnimatorOptions, MuseTalk, MuseTalkAnimator, MuseTalkConfig, UNetConfig, VaeConfig,
};
use hanzo_engine::speech_models::whisper::{WhisperConfig, WhisperFeatureExtractor};
use hanzo_ml::{DType, Device};
use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};

const WHISPER_ENCODER_PREFIX: &str = "encoder";
const RESIZED_IMG: usize = 256;
const FPS: f64 = 25.0;
const DEFAULT_FRAMES: usize = 13; // matches the measured CPU-floor clip (256-res, 13 frames)
const WARMUP_FRAMES: usize = 2; // first GPU forward compiles shaders/allocates; don't time it

fn env_path(key: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| default.to_string()))
}

fn device() -> Result<Device> {
    #[cfg(feature = "vulkan")]
    {
        eprintln!("[dub-gpu] device = Vulkan:0");
        return Ok(Device::new_vulkan(0)?);
    }
    #[cfg(all(feature = "rocm", not(feature = "vulkan")))]
    {
        eprintln!("[dub-gpu] device = ROCm:0");
        return Ok(Device::new_rocm(0)?);
    }
    #[cfg(all(feature = "cuda", not(feature = "vulkan"), not(feature = "rocm")))]
    {
        eprintln!("[dub-gpu] device = CUDA:0");
        return Ok(Device::new_cuda(0)?);
    }
    #[cfg(not(any(feature = "vulkan", feature = "rocm", feature = "cuda")))]
    {
        eprintln!("[dub-gpu] device = CPU");
        Ok(Device::Cpu)
    }
}

fn load_vb(path: &PathBuf, dtype: DType, dev: &Device) -> Result<ShardedVarBuilder> {
    let predicate: Arc<dyn Fn(String) -> bool + Send + Sync> = Arc::new(|_| true);
    let vb = unsafe { ShardedSafeTensors::sharded(&[path.clone()], dtype, dev, None, predicate)? };
    Ok(vb)
}

// PCM of just enough samples for `frames` output frames: the extractor yields ceil(secs * fps)
// frames, so secs = frames / fps. Uses sun.wav content when present (resampled internally), else a
// sine. Returns the pcm and its sample rate.
fn pcm_for_frames(frames: usize) -> (Arc<Vec<f32>>, usize) {
    let wav = env_path(
        "SUN_WAV",
        "/home/z/work/zen-dub-run/zen-dub/data/audio/sun.wav",
    );
    if let Ok(audio) = hanzo_audio::AudioInput::read_wav(&wav.to_string_lossy()) {
        let rate = audio.sample_rate.max(1) as usize;
        let want = ((frames as f64 / FPS) * rate as f64).ceil() as usize + 1;
        if audio.samples.len() >= want {
            return (Arc::new(audio.samples[..want].to_vec()), rate);
        }
    }
    let rate = 24_000usize;
    let n = ((frames as f64 / FPS) * rate as f64).ceil() as usize + 1;
    let pcm = (0..n)
        .map(|i| (i as f32 * 220.0 * std::f32::consts::PI / rate as f32).sin() * 0.3)
        .collect();
    (Arc::new(pcm), rate)
}

fn build_animator(dev: &Device) -> Result<MuseTalkAnimator> {
    let mt_dir = env_path("MT_DIR", "/home/z/models/MuseTalk/musetalkV15");
    let vae_dir = env_path("VAE_DIR", "/home/z/models/MuseTalk/sd-vae-ft-mse");
    let whisper_st = env_path("WHISPER_ST", "/home/z/models/whisper/tiny.safetensors");
    let s3fd_st = env_path("S3FD_ST", "/home/z/models/s3fd.safetensors");

    let unet: UNetConfig = serde_json::from_str(
        &std::fs::read_to_string(mt_dir.join("musetalk.json")).context("read musetalk.json")?,
    )?;
    let vae: VaeConfig = serde_json::from_str(
        &std::fs::read_to_string(vae_dir.join("config.json")).context("read vae config.json")?,
    )?;
    let cfg = MuseTalkConfig {
        unet,
        vae,
        resized_img: RESIZED_IMG,
    };

    let vae_vb = load_vb(
        &vae_dir.join("diffusion_pytorch_model.safetensors"),
        DType::F32,
        dev,
    )?;
    let unet_vb = load_vb(&mt_dir.join("unet.safetensors"), DType::F32, dev)?;
    let musetalk =
        MuseTalk::new(cfg, vae_vb, unet_vb, dev, DType::F32).context("build MuseTalk")?;

    let whisper_vb = load_vb(&whisper_st, DType::F32, dev)?;
    let whisper = WhisperFeatureExtractor::new(
        WhisperConfig::tiny(),
        whisper_vb.pp(WHISPER_ENCODER_PREFIX),
        dev,
    )
    .context("build whisper")?;

    let s3fd_vb = load_vb(&s3fd_st, DType::F32, dev)?;
    MuseTalkAnimator::new(musetalk, whisper, s3fd_vb, AnimatorOptions::default())
        .context("build animator")
}

#[test]
fn dub_gpu_bench() -> Result<()> {
    let mt_dir = env_path("MT_DIR", "/home/z/models/MuseTalk/musetalkV15");
    let demo = env_path(
        "DEMO_PNG",
        "/home/z/work/echomimic-test/echomimic_v3/datasets/echomimicv3_demos/imgs/demo_ch_woman_04.png",
    );
    if !mt_dir.join("unet.safetensors").is_file() || !demo.is_file() {
        eprintln!("[dub-gpu] weights/footage absent; skipping (set MT_DIR/DEMO_PNG)");
        return Ok(());
    }

    let dev = device()?;

    let t_load = Instant::now();
    let mut animator = build_animator(&dev)?;
    eprintln!(
        "[dub-gpu] weights loaded in {:.2}s",
        t_load.elapsed().as_secs_f64()
    );

    let footage = vec![image::open(&demo).context("open demo png")?];
    let frames: usize = std::env::var("DUB_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_FRAMES);

    let mk_req = |n: usize| {
        let (pcm, rate) = pcm_for_frames(n);
        AnimationRequest {
            driving: DrivingAudio {
                pcm,
                sample_rate: rate,
            },
            visual: VisualSource::Footage {
                frames: footage.clone(),
            },
            fps: FPS,
        }
    };

    // Warmup: first forward compiles the GPU compute pipelines; exclude it from timing.
    let warm = animator
        .animate(&mk_req(WARMUP_FRAMES))
        .context("warmup animate")?;
    eprintln!("[dub-gpu] warmup produced {} frames", warm.frames.len());

    let t0 = Instant::now();
    let out = animator.animate(&mk_req(frames)).context("timed animate")?;
    let dt = t0.elapsed().as_secs_f64();
    let n = out.frames.len();

    let (dw, dh) = (footage[0].width(), footage[0].height());
    for (i, f) in out.frames.iter().enumerate() {
        assert_eq!(f.width(), dw, "frame {i} width preserved");
        assert_eq!(f.height(), dh, "frame {i} height preserved");
    }

    eprintln!(
        "[dub-gpu] RESULT: {n} frames @ {RESIZED_IMG}-res in {dt:.3}s = {:.4} frames/s ({:.3} s/frame)",
        n as f64 / dt,
        dt / n as f64
    );
    assert!(n >= frames.min(1), "no frames produced");
    Ok(())
}
