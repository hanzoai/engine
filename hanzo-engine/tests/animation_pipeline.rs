//! Env-gated end-to-end load of a real MuseTalk animation model through the
//! `AnimationLoader`. A clean no-op when weights are absent (CI), exactly like
//! `tests/dub_e2e.rs`. When the weights are present this exercises the full load
//! path: config parse -> 4 VarBuilders (UNet/VAE/whisper/S3FD, incl. .pth/.bin)
//! -> M1 encoder-scoping -> animator construction. If any weight key or shape
//! mismatches a model's `vb.get`, loading fails -- so a green run validates the
//! whole real-weight wiring.
//!
//!   MUSETALK_UNET_CONFIG=/abs/musetalkV15/musetalk.json
//!   MUSETALK_UNET_WEIGHTS=/abs/musetalkV15/unet.pth
//!   VAE_CONFIG=/abs/sd-vae-ft-mse/config.json
//!   VAE_WEIGHTS=/abs/sd-vae-ft-mse/diffusion_pytorch_model.safetensors
//!   WHISPER_TINY=/abs/whisper/tiny.pt
//!   S3FD_WEIGHTS=/abs/s3fd.pth

use std::path::PathBuf;

use hanzo_engine::diffusion_models::musetalk::AnimatorOptions;
use hanzo_engine::{
    AnimationLoader, AnimationLoaderType, AnimationModelPaths, DeviceMapSetting, Loader,
    ModelCategory, ModelDType, ModelPaths,
};
use hanzo_ml::Device;

fn file(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
}

#[test]
fn animation_loader_builds_pipeline() -> anyhow::Result<()> {
    let (
        Some(unet_config),
        Some(unet_weights),
        Some(vae_config),
        Some(vae_weights),
        Some(whisper_weights),
        Some(s3fd_weights),
    ) = (
        file("MUSETALK_UNET_CONFIG"),
        file("MUSETALK_UNET_WEIGHTS"),
        file("VAE_CONFIG"),
        file("VAE_WEIGHTS"),
        file("WHISPER_TINY"),
        file("S3FD_WEIGHTS"),
    )
    else {
        eprintln!("[animation] MuseTalk weights absent; skipping");
        return Ok(());
    };

    let paths = AnimationModelPaths::new(
        unet_config,
        unet_weights,
        vae_config,
        vae_weights,
        whisper_weights,
        s3fd_weights,
    );
    let loader = AnimationLoader {
        model_id: "musetalk".to_string(),
        arch: AnimationLoaderType::MuseTalk,
        options: AnimatorOptions::default(),
    };

    let pipeline = loader.load_model_from_path(
        &(Box::new(paths) as Box<dyn ModelPaths>),
        &ModelDType::Auto,
        &Device::Cpu,
        true,
        DeviceMapSetting::dummy(),
        None,
        None,
    )?;

    let guard = pipeline.blocking_lock();
    assert_eq!(guard.category(), ModelCategory::Animation);
    eprintln!("[animation] loaded MuseTalk animation pipeline (real weights)");
    Ok(())
}
