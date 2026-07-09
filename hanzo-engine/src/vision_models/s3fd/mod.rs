#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! S3FD (Single Shot Scale-invariant Face Detector), the detector MuseTalk drives
//! its crops from (vendored from face_alignment's `sfd`). VGG16-SSD backbone, 6
//! anchor scales, max-out background on the finest scale. `detect` returns image-
//! space `FaceBox`es sorted by score.

mod model;
mod nms;

pub use nms::FaceBox;

use hanzo_ml::{Device, Result, Tensor};
use hanzo_nn::ops::softmax;
use hanzo_quant::ShardedVarBuilder;
use image::{imageops::FilterType, DynamicImage, GenericImageView};

use model::S3fdModel;
use nms::{decode_box, nms};

const COLLECT_THRESHOLD: f32 = 0.05; // candidate cutoff before NMS (face_alignment)
const BGR_MEAN: [f32; 3] = [104.0, 117.0, 123.0];
const ANCHOR_SCALE: f32 = 4.0; // anchor side = stride * 4
                               // Below this the VGG stem's 5 max-pools collapse a spatial dim to 0 (the conv would bail).
                               // Nothing detectable that small anyway, so report "no faces" rather than erroring.
const MIN_DETECT_SIDE: u32 = 32;

#[derive(Clone, Copy, Debug)]
pub struct S3fdConfig {
    /// Final score to keep a detection after NMS.
    pub score_threshold: f32,
    /// IoU above which a lower-scoring box is suppressed.
    pub nms_iou: f32,
    /// Longest side the detector runs at; larger images are downscaled then boxes
    /// scaled back (S3FD is scale-invariant, so this only bounds compute).
    pub max_side: u32,
}

impl Default for S3fdConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            nms_iou: 0.3,
            max_side: 640,
        }
    }
}

pub struct S3fd {
    model: S3fdModel,
    device: Device,
    cfg: S3fdConfig,
}

impl S3fd {
    pub fn new(vb: ShardedVarBuilder, device: &Device, cfg: S3fdConfig) -> Result<Self> {
        Ok(Self {
            model: S3fdModel::new(vb)?,
            device: device.clone(),
            cfg,
        })
    }

    pub fn detect(&self, image: &DynamicImage) -> Result<Vec<FaceBox>> {
        let (w0, h0) = image.dimensions();
        let long = w0.max(h0);
        let scale = if long > self.cfg.max_side {
            self.cfg.max_side as f32 / long as f32
        } else {
            1.0
        };
        let det_img = if scale < 1.0 {
            let nw = ((w0 as f32 * scale).round() as u32).max(1);
            let nh = ((h0 as f32 * scale).round() as u32).max(1);
            image.resize_exact(nw, nh, FilterType::Triangle)
        } else {
            image.clone()
        };

        let (dw, dh) = det_img.dimensions();
        if dw.min(dh) < MIN_DETECT_SIDE {
            return Ok(Vec::new());
        }

        let input = self.preprocess(&det_img)?;
        let scales = self.model.forward(&input)?;

        let mut candidates = Vec::new();
        for s in &scales {
            let conf = softmax(&s.conf, 1)?;
            let (_, _, h, w) = conf.dims4()?;
            let face = conf
                .narrow(1, 1, 1)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let loc = s.loc.squeeze(0)?.reshape((4, h * w))?.to_vec2::<f32>()?;
            let stride = s.stride as f32;
            let anchor = stride * ANCHOR_SCALE;
            for p in 0..h * w {
                let score = face[p];
                if score <= COLLECT_THRESHOLD {
                    continue;
                }
                let cx = stride / 2.0 + (p % w) as f32 * stride;
                let cy = stride / 2.0 + (p / w) as f32 * stride;
                let l = [loc[0][p], loc[1][p], loc[2][p], loc[3][p]];
                candidates.push(decode_box(l, cx, cy, anchor, score));
            }
        }

        let inv = 1.0 / scale;
        let mut faces: Vec<FaceBox> = nms(candidates, self.cfg.nms_iou)
            .into_iter()
            .filter(|b| b.score >= self.cfg.score_threshold)
            .map(|b| b.scaled(inv))
            .collect();
        faces.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(faces)
    }

    fn preprocess(&self, image: &DynamicImage) -> Result<Tensor> {
        let rgb = image.to_rgb8();
        let (w, h) = rgb.dimensions();
        let (w, h) = (w as usize, h as usize);
        let raw = rgb.into_raw();
        let plane = w * h;
        let mut data = vec![0f32; 3 * plane];
        for i in 0..plane {
            data[i] = raw[i * 3 + 2] as f32 - BGR_MEAN[0]; // B
            data[plane + i] = raw[i * 3 + 1] as f32 - BGR_MEAN[1]; // G
            data[2 * plane + i] = raw[i * 3] as f32 - BGR_MEAN[2]; // R
        }
        Tensor::from_vec(data, (1, 3, h, w), &self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::{DType, Shape};
    use hanzo_nn::var_builder::SimpleBackend;
    use hanzo_nn::Init;
    use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};
    use std::path::PathBuf;

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

    fn random_vb(dev: &Device) -> ShardedVarBuilder {
        ShardedSafeTensors::wrap(Box::new(RandnBackend), DType::F32, dev.clone())
    }

    // Exercises the full VGG-SSD forward + decode + NMS wiring (catches conv-dim,
    // max-out, and head shape mismatches). Random weights -> garbage scores, so we
    // only assert the pipeline runs and emits finite, well-formed boxes.
    #[test]
    fn detect_graph_runs_random_weights() -> Result<()> {
        let dev = Device::Cpu;
        let s3fd = S3fd::new(random_vb(&dev), &dev, S3fdConfig::default())?;
        let img = DynamicImage::new_rgb8(128, 128);
        let faces = s3fd.detect(&img)?;
        for f in &faces {
            for v in [f.x1, f.y1, f.x2, f.y2, f.score] {
                assert!(v.is_finite(), "non-finite box coordinate");
            }
            assert!(f.x2 >= f.x1 && f.y2 >= f.y1, "degenerate box ordering");
        }
        Ok(())
    }

    #[test]
    fn detect_too_small_returns_empty() -> Result<()> {
        let dev = Device::Cpu;
        let s3fd = S3fd::new(random_vb(&dev), &dev, S3fdConfig::default())?;
        // 16 < MIN_DETECT_SIDE: must report no faces, not bail in the conv stem.
        assert!(s3fd.detect(&DynamicImage::new_rgb8(16, 16))?.is_empty());
        Ok(())
    }

    // Real-weights accuracy check. Provide S3FD_WEIGHTS (safetensors), a test image,
    // and the ground-truth bbox to activate; otherwise a clean no-op (no weights in
    // CI), mirroring tests/dub_e2e.rs.
    #[test]
    fn detect_known_bbox() -> Result<()> {
        let (Some(weights), Some(img_path), Some(bbox)) = (
            std::env::var("S3FD_WEIGHTS").ok().map(PathBuf::from),
            std::env::var("S3FD_TEST_IMAGE").ok().map(PathBuf::from),
            std::env::var("S3FD_TEST_BBOX").ok(),
        ) else {
            eprintln!("[s3fd] S3FD_WEIGHTS/S3FD_TEST_IMAGE/S3FD_TEST_BBOX unset; skipping");
            return Ok(());
        };
        let coords: Vec<f32> = bbox
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        assert_eq!(coords.len(), 4, "S3FD_TEST_BBOX must be x1,y1,x2,y2");
        let truth = FaceBox {
            x1: coords[0],
            y1: coords[1],
            x2: coords[2],
            y2: coords[3],
            score: 1.0,
        };

        let dev = Device::Cpu;
        let predicate: std::sync::Arc<dyn Fn(String) -> bool + Send + Sync> =
            std::sync::Arc::new(|_| true);
        let vb =
            unsafe { ShardedSafeTensors::sharded(&[weights], DType::F32, &dev, None, predicate)? };
        let s3fd = S3fd::new(vb, &dev, S3fdConfig::default())?;
        let img = image::open(&img_path).map_err(|e| hanzo_ml::Error::msg(e.to_string()))?;
        let faces = s3fd.detect(&img)?;
        assert!(!faces.is_empty(), "no faces detected");
        let iou = faces[0].iou(&truth);
        eprintln!("[s3fd] top face {:?} IoU={iou:.3} vs {truth:?}", faces[0]);
        assert!(iou >= 0.7, "top face IoU {iou:.3} < 0.7");
        Ok(())
    }
}
