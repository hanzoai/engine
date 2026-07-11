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
const CORE_FRAMES: usize = 64; // core-fps averaging window
const AUDIO_SEQ_LEN: usize = 50; // MuseTalk whisper chunk length per output frame
const TAESD_DIR_DEFAULT: &str = "/home/z/models/MuseTalk/taesd";

fn env_path(key: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| default.to_string()))
}

// Per-branch `return` keeps the cfg device selector readable; whichever branch is active would
// otherwise trip needless_return under that feature set.
#[allow(clippy::needless_return)]
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
    let vb = unsafe {
        ShardedSafeTensors::sharded(std::slice::from_ref(path), dtype, dev, None, predicate)?
    };
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

    let dtype = dtype_from_env();
    let vae_vb = load_vb(
        &vae_dir.join("diffusion_pytorch_model.safetensors"),
        dtype,
        dev,
    )?;
    let unet_vb = load_vb(&mt_dir.join("unet.safetensors"), dtype, dev)?;
    let mut musetalk = MuseTalk::new(cfg, vae_vb, unet_vb, dev, dtype).context("build MuseTalk")?;
    let td = env_path("TAESD_DIR", TAESD_DIR_DEFAULT);
    if taesd_enabled() && td.join("taesd_encoder.safetensors").is_file() {
        let enc = load_vb(&td.join("taesd_encoder.safetensors"), dtype, dev)?;
        let dec = load_vb(&td.join("taesd_decoder.safetensors"), dtype, dev)?;
        musetalk = musetalk.with_taesd(enc, dec).context("attach TAESD")?;
        eprintln!("[dub-gpu] TAESD encoder+decoder attached");
    }

    // S3FD + whisper stay F32; the animator casts whisper features to the core dtype.
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

fn dtype_from_env() -> DType {
    match std::env::var("DUB_DTYPE").as_deref() {
        Ok("f16") | Ok("F16") | Ok("half") => DType::F16,
        _ => DType::F32,
    }
}

fn taesd_enabled() -> bool {
    std::env::var("DUB_TAESD")
        .map(|v| {
            let v = v.trim().to_string();
            !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

// Just the MuseTalk core (VAE/TAESD + UNet), the realtime-critical inner loop. No S3FD/whisper.
fn build_core(dev: &Device, dtype: DType, with_taesd: bool) -> Result<MuseTalk> {
    let mt_dir = env_path("MT_DIR", "/home/z/models/MuseTalk/musetalkV15");
    let vae_dir = env_path("VAE_DIR", "/home/z/models/MuseTalk/sd-vae-ft-mse");
    let unet: UNetConfig =
        serde_json::from_str(&std::fs::read_to_string(mt_dir.join("musetalk.json"))?)?;
    let vae: VaeConfig =
        serde_json::from_str(&std::fs::read_to_string(vae_dir.join("config.json"))?)?;
    let cfg = MuseTalkConfig {
        unet,
        vae,
        resized_img: RESIZED_IMG,
    };
    let vae_vb = load_vb(
        &vae_dir.join("diffusion_pytorch_model.safetensors"),
        dtype,
        dev,
    )?;
    let unet_vb = load_vb(&mt_dir.join("unet.safetensors"), dtype, dev)?;
    let mt = MuseTalk::new(cfg, vae_vb, unet_vb, dev, dtype).context("build MuseTalk core")?;
    if with_taesd {
        let td = env_path("TAESD_DIR", TAESD_DIR_DEFAULT);
        let enc = load_vb(&td.join("taesd_encoder.safetensors"), dtype, dev)?;
        let dec = load_vb(&td.join("taesd_decoder.safetensors"), dtype, dev)?;
        return mt.with_taesd(enc, dec).context("attach TAESD");
    }
    Ok(mt)
}

fn bench_core(
    label: &str,
    mt: &MuseTalk,
    dev: &Device,
    dtype: DType,
    frames: usize,
) -> Result<f64> {
    let sz = mt.resized_img();
    let cdim = mt.cross_attention_dim();
    let face = hanzo_ml::Tensor::rand(0f32, 1f32, (1, 3, sz, sz), dev)?.to_dtype(dtype)?;
    let audio =
        hanzo_ml::Tensor::randn(0f64, 1.0, (1, AUDIO_SEQ_LEN, cdim), dev)?.to_dtype(dtype)?;
    let ts = hanzo_ml::Tensor::zeros(1, DType::F32, dev)?;

    for _ in 0..WARMUP_FRAMES {
        mt.forward(&face, &audio)?;
    }
    dev.synchronize()?;

    let t = Instant::now();
    let lat = mt.latents_for_unet(&face)?;
    dev.synchronize()?;
    let t_enc = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let pred = mt.unet_forward(&lat, &ts, &audio)?;
    dev.synchronize()?;
    let t_unet = t.elapsed().as_secs_f64();
    let t = Instant::now();
    mt.decode_latents(&pred)?;
    dev.synchronize()?;
    let t_dec = t.elapsed().as_secs_f64();

    let t0 = Instant::now();
    for _ in 0..frames {
        mt.forward(&face, &audio)?;
    }
    dev.synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    let fps = frames as f64 / dt;
    eprintln!(
        "[core] {label:9} encode {:6.2}ms  unet {:6.2}ms  decode {:6.2}ms | {:6.2} ms/frame = {:6.2} fps",
        t_enc * 1e3,
        t_unet * 1e3,
        t_dec * 1e3,
        dt / frames as f64 * 1e3,
        fps
    );
    Ok(fps)
}

// Diagnostic: compare TAESD vs full-VAE latent scale + isolate the decoder, on real weights.
// Set MUSETALK_DUBIN to a dir with face_000000.npy. Prints the std ratio (the encoder scale bug
// signature) and the TAESD-decode-of-correct-UNet-pred PSNR (isolates the decoder).
#[test]
fn taesd_scale_diagnostic() -> Result<()> {
    let Ok(dubin) = std::env::var("MUSETALK_DUBIN") else {
        eprintln!("[diag] set MUSETALK_DUBIN; skipping");
        return Ok(());
    };
    let dev = device()?;
    let face = hanzo_ml::Tensor::read_npy(format!("{dubin}/face_000000.npy"))?.to_device(&dev)?;
    let audio = hanzo_ml::Tensor::read_npy(format!("{dubin}/audio_000000.npy"))?.to_device(&dev)?;
    let full = build_core(&dev, DType::F32, false)?;
    let taesd = build_core(&dev, DType::F32, true)?;

    let std = |t: &hanzo_ml::Tensor| -> Result<f64> {
        let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let m = v.iter().sum::<f32>() as f64 / v.len() as f64;
        Ok((v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt())
    };
    let psnr = |a: &hanzo_ml::Tensor, b: &hanzo_ml::Tensor| -> Result<f64> {
        let (a, b) = (
            a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
        );
        let mse = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).powi(2) as f64)
            .sum::<f64>()
            / a.len() as f64;
        Ok(if mse > 0.0 {
            10.0 * (1.0 / mse).log10()
        } else {
            99.0
        })
    };

    let lat_full = full.latents_for_unet(&face)?;
    let lat_taesd = taesd.latents_for_unet(&face)?;
    eprintln!(
        "[diag] latent std: full={:.4} taesd={:.4} ratio(taesd/full)={:.4}",
        std(&lat_full)?,
        std(&lat_taesd)?,
        std(&lat_taesd)? / std(&lat_full)?
    );
    // Isolate the decoder: same correct UNet pred through both decoders.
    let ts = hanzo_ml::Tensor::zeros(1, DType::F32, &dev)?;
    let pred = full.unet_forward(&lat_full, &ts, &audio)?;
    let img_full = full.decode_latents(&pred)?;
    let img_taesd = taesd.decode_latents(&pred)?;
    eprintln!(
        "[diag] decoder-only PSNR(TAESD vs fullVAE, same pred)={:.2}dB",
        psnr(&img_taesd, &img_full)?
    );
    Ok(())
}

// Renders the LSE-C quality-gate mouths: read the pre-dumped standard-clip faces + whisper feats
// from MUSETALK_DUBIN, run the real-weight core (full-VAE or DUB_TAESD), write [3,256,256] mouths
// to MUSETALK_DUBOUT. Python reblend + syncnet then score LSE-C. Same npy contract as the dub tool.
#[test]
fn dub_render_mouths() -> Result<()> {
    let (Ok(dubin), Ok(dubout)) = (
        std::env::var("MUSETALK_DUBIN"),
        std::env::var("MUSETALK_DUBOUT"),
    ) else {
        eprintln!("[render] set MUSETALK_DUBIN + MUSETALK_DUBOUT to render mouths; skipping");
        return Ok(());
    };
    let dev = device()?;
    let dtype = dtype_from_env();
    let with_taesd = taesd_enabled();
    let mt = build_core(&dev, dtype, with_taesd)?;
    std::fs::create_dir_all(&dubout)?;

    let t0 = Instant::now();
    let mut n = 0usize;
    loop {
        let face_p = format!("{dubin}/face_{n:06}.npy");
        if !std::path::Path::new(&face_p).is_file() {
            break;
        }
        let face = hanzo_ml::Tensor::read_npy(&face_p)?
            .to_device(&dev)?
            .to_dtype(dtype)?;
        let audio = hanzo_ml::Tensor::read_npy(format!("{dubin}/audio_{n:06}.npy"))?
            .to_device(&dev)?
            .to_dtype(dtype)?;
        let mouth = mt.forward(&face, &audio)?;
        mouth
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .write_npy(format!("{dubout}/mouth_{n:06}.npy"))?;
        n += 1;
    }
    dev.synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "[render] {n} mouths -> {dubout} in {dt:.2}s ({:.2} fps) taesd={with_taesd} dtype={dtype:?}",
        n as f64 / dt
    );
    assert!(n > 0, "no face_*.npy found in {dubin}");
    Ok(())
}

// Realtime core benchmark: per-frame encode/unet/decode, full-VAE vs TAESD encoder, at DUB_DTYPE.
#[test]
fn dub_musetalk_core_fps() -> Result<()> {
    let mt_dir = env_path("MT_DIR", "/home/z/models/MuseTalk/musetalkV15");
    if !mt_dir.join("unet.safetensors").is_file() {
        eprintln!("[core] weights absent; skipping (set MT_DIR)");
        return Ok(());
    }
    let dev = device()?;
    let dtype = dtype_from_env();
    let frames = std::env::var("DUB_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CORE_FRAMES);
    eprintln!("[core] dtype = {dtype:?}, frames = {frames}");

    let full_fps = {
        let full = build_core(&dev, dtype, false)?;
        bench_core("full-VAE", &full, &dev, dtype, frames)?
    };

    let td = env_path("TAESD_DIR", TAESD_DIR_DEFAULT);
    if td.join("taesd_encoder.safetensors").is_file() {
        let taesd = build_core(&dev, dtype, true)?;
        let taesd_fps = bench_core("TAESD", &taesd, &dev, dtype, frames)?;
        eprintln!(
            "[core] SPEEDUP TAESD/full = {:.2}x  ({:.2} -> {:.2} fps @ {dtype:?})",
            taesd_fps / full_fps,
            full_fps,
            taesd_fps
        );
    } else {
        eprintln!(
            "[core] TAESD weights absent at {}; full-VAE only",
            td.display()
        );
    }
    Ok(())
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
