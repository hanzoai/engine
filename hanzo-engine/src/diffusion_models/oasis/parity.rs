//! Parity harness: compares the Rust Oasis port against the Python reference (`oracle_dump.py`).
//! Gated on the weights + oracle safetensors existing under `OASIS_DIR` (default the evo scratch
//! dir); skips with a notice otherwise so CI without the 3GB weights stays green. Runs on CPU in f32
//! to match the CPU f32 oracle apples-to-apples.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use hanzo_ml::{DType, Device, Result, Tensor};

use super::{SampleParams, WorldModel};

const OASIS_DIR_DEFAULT: &str = "/home/z/work/hanzo/oasis-weights";

fn oasis_dir() -> PathBuf {
    PathBuf::from(std::env::var("OASIS_DIR").unwrap_or_else(|_| OASIS_DIR_DEFAULT.to_string()))
}

fn as_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

fn cosine(a: &Tensor, b: &Tensor) -> Result<f64> {
    let (a, b) = (as_vec(a)?, as_vec(b)?);
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(&b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    Ok(dot / (na.sqrt() * nb.sqrt()))
}

fn max_abs_err(a: &Tensor, b: &Tensor) -> Result<f64> {
    let (a, b) = (as_vec(a)?, as_vec(b)?);
    Ok(a.iter()
        .zip(&b)
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0f64, f64::max))
}

// PSNR in dB against peak 1.0 (frames are in [0, 1]).
fn psnr(a: &Tensor, b: &Tensor) -> Result<f64> {
    let (a, b) = (as_vec(a)?, as_vec(b)?);
    let mse: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| {
            let d = f64::from(x - y);
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    Ok(if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (1.0 / mse).log10()
    })
}

fn load_oracle(path: &Path, dev: &Device) -> Result<HashMap<String, Tensor>> {
    hanzo_ml::safetensors::load(path, dev)
}

#[test]
fn oasis_parity() -> Result<()> {
    let dir = oasis_dir();
    let vae_p = dir.join("vit-l-20.safetensors");
    let dit_p = dir.join("oasis500m.safetensors");
    let oracle_p = dir.join("oracle.safetensors");
    if !vae_p.exists() || !dit_p.exists() || !oracle_p.exists() {
        eprintln!(
            "SKIP oasis_parity: weights/oracle not found under {}",
            dir.display()
        );
        return Ok(());
    }

    let dev = Device::Cpu;
    let wm = WorldModel::load(vae_p, dit_p, DType::F32, &dev)?;
    let o = load_oracle(&oracle_p, &dev)?;
    let get = |k: &str| o.get(k).cloned().unwrap();

    // --- VAE encode: cosine > 0.999 ---
    let frame_in = get("frame_in");
    let rust_lat = wm.encode_frames(&frame_in)?[0].clone();
    let enc_cos = cosine(&rust_lat, &get("vae_latent"))?;
    let enc_err = max_abs_err(&rust_lat, &get("vae_latent"))?;
    println!("VAE encode : cosine {enc_cos:.6}  maxerr {enc_err:.3e}");

    // --- VAE decode round-trip: PSNR > 35 dB ---
    let rust_dec = wm.decode_frames(&[get("vae_latent")])?;
    let dec_psnr = psnr(&rust_dec, &get("vae_decoded"))?;
    let dec_err = max_abs_err(&rust_dec, &get("vae_decoded"))?;
    println!("VAE decode : psnr {dec_psnr:.2} dB  maxerr {dec_err:.3e}");

    // --- DiT single denoise step: cosine ~1, tight maxerr ---
    let rust_v = wm
        .dit()
        .forward(&get("dit_x"), &get("dit_t"), &get("dit_act"))?;
    let dit_cos = cosine(&rust_v, &get("dit_v"))?;
    let dit_err = max_abs_err(&rust_v, &get("dit_v"))?;
    println!("DiT step   : cosine {dit_cos:.6}  maxerr {dit_err:.3e}");

    assert!(enc_cos > 0.999, "VAE encode cosine {enc_cos} <= 0.999");
    assert!(dec_psnr > 35.0, "VAE decode PSNR {dec_psnr} <= 35 dB");
    assert!(dit_cos > 0.999, "DiT step cosine {dit_cos} <= 0.999");
    Ok(())
}

// A tiny 4-frame rollout must stay finite and produce the right shape (no NaN, no shape drift).
#[test]
fn oasis_rollout_smoke() -> Result<()> {
    let dir = oasis_dir();
    let (vae_p, dit_p) = (
        dir.join("vit-l-20.safetensors"),
        dir.join("oasis500m.safetensors"),
    );
    if !vae_p.exists() || !dit_p.exists() {
        eprintln!("SKIP oasis_rollout_smoke: weights not found");
        return Ok(());
    }
    let dev = Device::Cpu;
    let wm = WorldModel::load(vae_p, dit_p, DType::F32, &dev)?;
    let prompt = Tensor::rand(0f32, 1f32, (1, 3, super::FRAME_H, super::FRAME_W), &dev)?;
    let actions = Tensor::zeros((1, 4, super::ACTION_KEYS.len()), DType::F32, &dev)?;
    let params = SampleParams {
        total_frames: 4,
        ddim_steps: 4,
        seed: 0,
    };
    let out = wm.generate(&prompt, &actions, &params)?;
    assert_eq!(out.dims(), &[4, 3, super::FRAME_H, super::FRAME_W]);
    let flat = as_vec(&out)?;
    assert!(
        flat.iter().all(|x| x.is_finite()),
        "rollout produced non-finite pixels"
    );
    Ok(())
}
