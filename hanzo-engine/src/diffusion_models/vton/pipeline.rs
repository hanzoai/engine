//! VTON serving: weight loading (cached), image preprocessing (resize/pad/normalize), and the
//! end-to-end `vton_generate` seam used by the `/v1/images/tryon` route.
//!
//! Pose conditioning: the reference derives a grayscale skeleton via DWPose (ONNX). That detector
//! is not yet ported, so poses are optional user-supplied grayscale images; when absent they are
//! blank (out-of-distribution for the person stream, so quality degrades) - DWPose is the follow-up.
//! Segmentation-free maskless mode only (the person image is passed through unmasked, exactly as the
//! reference default `segmentation_free=True`); masked mode needs the human parser (also a follow-up).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Result};
use hanzo_ml::{DType, Device, Tensor};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use image::{DynamicImage, RgbImage};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_isaac::Isaac64Rng;

use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};

use super::config::VtonConfig;
use super::model::TryOnModel;
use super::sampling::{denoise, Conditioning, SampleParams};

pub const DEFAULT_MODEL_ID: &str = "fashn-ai/fashn-vton-1.5";
const WEIGHTS_FILE: &str = "model.safetensors";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Tops,
    Bottoms,
    OnePieces,
}

impl Category {
    /// Class id fed to the category embedding (0 is the null/unconditional class).
    pub fn label(self) -> u32 {
        match self {
            Category::Tops => 1,
            Category::Bottoms => 2,
            Category::OnePieces => 3,
        }
    }
}

impl FromStr for Category {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "tops" => Ok(Category::Tops),
            "bottoms" => Ok(Category::Bottoms),
            "one-pieces" | "one_pieces" | "onepieces" => Ok(Category::OnePieces),
            other => Err(anyhow!(
                "unknown category '{other}' (tops|bottoms|one-pieces)"
            )),
        }
    }
}

/// A try-on request resolved to decoded images. `person_pose` / `garment_pose` are optional
/// grayscale skeleton maps; when `None` a blank pose is used (see module docs).
pub struct TryOnInputs {
    pub model_id: String,
    pub person: DynamicImage,
    pub garment: DynamicImage,
    pub person_pose: Option<DynamicImage>,
    pub garment_pose: Option<DynamicImage>,
    pub category: Category,
    pub num_samples: usize,
    pub seed: u64,
    pub sample: SampleParams,
}

type ModelCache = Mutex<HashMap<String, Arc<TryOnModel>>>;

fn model_cache() -> &'static ModelCache {
    static CACHE: OnceLock<ModelCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn resolve_weights(model_id: &str) -> Result<PathBuf> {
    let p = PathBuf::from(model_id);
    if p.is_file() {
        return Ok(p);
    }
    let local = p.join(WEIGHTS_FILE);
    if local.is_file() {
        return Ok(local);
    }
    let api = ApiBuilder::from_env().with_progress(true).build()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
    Ok(repo.get(WEIGHTS_FILE)?)
}

/// Load (and cache) the VTON model for `model_id` on `device`, in F32.
pub fn load_model(model_id: &str, device: &Device) -> Result<Arc<TryOnModel>> {
    let key = format!("{model_id}@{device:?}");
    if let Some(m) = model_cache().lock().unwrap().get(&key) {
        return Ok(m.clone());
    }
    let weights = resolve_weights(model_id)?;
    let vb = from_mmaped_safetensors(
        vec![weights],
        Vec::new(),
        Some(DType::F32),
        device,
        vec![None],
        false,
        None,
        |_| true,
        Arc::new(|_| DeviceForLoadTensor::Base),
    )?;
    let model = Arc::new(TryOnModel::new(&VtonConfig::default(), vb)?);
    model_cache().lock().unwrap().insert(key, model.clone());
    Ok(model)
}

/// Aspect-preserving fit into (w, h) then center-pad with black. Returns the padded RGB image and
/// the (left, top) pad offsets so the output can be cropped back.
fn resize_pad(img: &DynamicImage, w: u32, h: u32) -> (RgbImage, u32, u32) {
    let fitted = img
        .resize(w, h, image::imageops::FilterType::Lanczos3)
        .to_rgb8();
    let (fw, fh) = (fitted.width(), fitted.height());
    let (left, top) = ((w - fw) / 2, (h - fh) / 2);
    let mut canvas = RgbImage::new(w, h);
    image::imageops::replace(&mut canvas, &fitted, left as i64, top as i64);
    (canvas, left, top)
}

fn resize_pad_gray(img: &DynamicImage, w: u32, h: u32) -> image::GrayImage {
    let fitted = img
        .resize(w, h, image::imageops::FilterType::Nearest)
        .to_luma8();
    let (fw, fh) = (fitted.width(), fitted.height());
    let (left, top) = ((w - fw) / 2, (h - fh) / 2);
    let mut canvas = image::GrayImage::new(w, h);
    image::imageops::replace(&mut canvas, &fitted, left as i64, top as i64);
    canvas
}

/// uint8 [0,255] -> [-1,1], HWC -> CHW, shape (1, C, H, W).
fn rgb_to_tensor(img: &RgbImage, device: &Device) -> Result<Tensor> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let data: Vec<f32> = img
        .as_raw()
        .iter()
        .map(|&p| p as f32 / 127.5 - 1.0)
        .collect();
    Ok(Tensor::from_vec(data, (h, w, 3), device)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?)
}

fn pose_to_tensor(pose: Option<&DynamicImage>, w: u32, h: u32, device: &Device) -> Result<Tensor> {
    let data: Vec<f32> = match pose {
        Some(p) => resize_pad_gray(p, w, h)
            .as_raw()
            .iter()
            .map(|&p| p as f32 / 127.5 - 1.0)
            .collect(),
        None => vec![-1.0f32; (w * h) as usize],
    };
    Ok(Tensor::from_vec(
        data,
        (1, 1, h as usize, w as usize),
        device,
    )?)
}

/// [-1,1] CHW f32 tensor -> RGB image, cropping the padding region.
fn tensor_to_rgb(
    t: &Tensor,
    crop_left: u32,
    crop_top: u32,
    crop_w: u32,
    crop_h: u32,
) -> Result<DynamicImage> {
    let (c, h, w) = t.dims3()?;
    if c != 3 {
        return Err(anyhow!("expected 3 channels, got {c}"));
    }
    let hwc = t.permute((1, 2, 0))?.flatten_all()?.to_vec1::<f32>()?;
    let pixels: Vec<u8> = hwc
        .iter()
        .map(|&v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let full = RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow!("invalid image buffer"))?;
    let cropped = image::imageops::crop_imm(&full, crop_left, crop_top, crop_w, crop_h).to_image();
    Ok(DynamicImage::ImageRgb8(cropped))
}

/// Run virtual try-on, returning `num_samples` images.
pub fn vton_generate(inputs: &TryOnInputs, device: &Device) -> Result<Vec<DynamicImage>> {
    let model = load_model(&inputs.model_id, device)?;
    let cfg = model.config();
    let (w, h) = (cfg.width as u32, cfg.height as u32);

    let (person_rgb, pad_left, pad_top) = resize_pad(&inputs.person, w, h);
    let fitted = inputs
        .person
        .resize(w, h, image::imageops::FilterType::Lanczos3);
    let (crop_w, crop_h) = (fitted.width(), fitted.height());
    let (garment_rgb, _, _) = resize_pad(&inputs.garment, w, h);

    let ca = rgb_to_tensor(&person_rgb, device)?;
    let garment = rgb_to_tensor(&garment_rgb, device)?;
    let person_pose = pose_to_tensor(inputs.person_pose.as_ref(), w, h, device)?;
    let garment_pose = pose_to_tensor(inputs.garment_pose.as_ref(), w, h, device)?;
    let categories = Tensor::from_vec(vec![inputs.category.label()], 1, device)?;

    let cond = Conditioning {
        ca_images: ca,
        garment_images: garment,
        person_poses: person_pose,
        garment_poses: garment_pose,
        categories,
    };

    // Host-seeded Gaussian init noise: device-agnostic and deterministic across
    // CPU/CUDA/Metal (device.set_seed is unimplemented on the CPU backend).
    let normal = Normal::new(0f32, 1.0).map_err(|e| anyhow!("vton noise dist: {e}"))?;
    let mut rng = Isaac64Rng::seed_from_u64(inputs.seed);
    let numel = cfg.channels_in * cfg.height * cfg.width;
    let mut out = Vec::with_capacity(inputs.num_samples);
    for _ in 0..inputs.num_samples {
        let noise: Vec<f32> = (0..numel).map(|_| normal.sample(&mut rng)).collect();
        let init = Tensor::from_vec(noise, (1, cfg.channels_in, cfg.height, cfg.width), device)?;
        let image = denoise(&model, &cond, &init, &inputs.sample)?;
        let image = image.squeeze(0)?;
        out.push(tensor_to_rgb(&image, pad_left, pad_top, crop_w, crop_h)?);
    }
    Ok(out)
}
