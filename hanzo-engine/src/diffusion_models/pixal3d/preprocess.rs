//! Image preprocessing for the DINOv2 conditioner.
//!
//! Mirrors TRELLIS `preprocess_image` + `encode_image` for the already-segmented (RGBA) case:
//! alpha-bbox crop with 1.2x margin, resize to 518, premultiply by alpha, ImageNet-normalize. For a
//! plain RGB image (no alpha) TRELLIS runs rembg/u2net background removal; hanzo-ml has no such model
//! yet, so an RGB input is resized/normalized whole (document this - pass a segmented RGBA for a
//! faithful result).

use hanzo_ml::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, RgbaImage};

const RES: usize = 518;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const ALPHA_FG: u8 = 204; // 0.8 * 255

/// `image` -> [1, 3, 518, 518] ImageNet-normalized conditioning tensor.
pub fn preprocess(image: &DynamicImage, device: &Device) -> Result<Tensor> {
    let rgba = image.to_rgba8();
    let has_alpha = rgba.pixels().any(|p| p[3] != 255);
    let (canvas, premultiply) = if has_alpha {
        (crop_to_object(&rgba), true)
    } else {
        (rgba, false)
    };
    let resized = image::imageops::resize(&canvas, RES as u32, RES as u32, FilterType::Lanczos3);

    let mut data = vec![0f32; 3 * RES * RES];
    for (x, y, px) in resized.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        let a = if premultiply { px[3] as f32 / 255.0 } else { 1.0 };
        for c in 0..3 {
            let v = (px[c] as f32 / 255.0) * a;
            data[c * RES * RES + y * RES + x] = (v - MEAN[c]) / STD[c];
        }
    }
    Tensor::from_vec(data, (1, 3, RES, RES), device)
}

/// Square crop around the alpha foreground with a 1.2x margin, out-of-bounds padded transparent.
fn crop_to_object(rgba: &RgbaImage) -> RgbaImage {
    let (w, h) = (rgba.width() as i32, rgba.height() as i32);
    let (mut minx, mut miny, mut maxx, mut maxy) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for (x, y, px) in rgba.enumerate_pixels() {
        if px[3] > ALPHA_FG {
            let (x, y) = (x as i32, y as i32);
            minx = minx.min(x);
            miny = miny.min(y);
            maxx = maxx.max(x);
            maxy = maxy.max(y);
        }
    }
    if minx > maxx {
        return rgba.clone(); // no foreground; use as-is
    }
    let cx = (minx + maxx) as f32 / 2.0;
    let cy = (miny + maxy) as f32 / 2.0;
    let size = (((maxx - minx).max(maxy - miny) as f32) * 1.2).round() as i32;
    let size = size.max(1);
    let (x0, y0) = ((cx - size as f32 / 2.0).round() as i32, (cy - size as f32 / 2.0).round() as i32);

    let mut out = RgbaImage::new(size as u32, size as u32);
    for oy in 0..size {
        for ox in 0..size {
            let (sx, sy) = (x0 + ox, y0 + oy);
            if sx >= 0 && sy >= 0 && sx < w && sy < h {
                out.put_pixel(ox as u32, oy as u32, *rgba.get_pixel(sx as u32, sy as u32));
            }
        }
    }
    out
}
