use image::{imageops::FilterType, DynamicImage};

/// Crop the `(x, y, w, h)` rectangle, clamped to the image bounds.
pub fn crop(image: &DynamicImage, x: u32, y: u32, w: u32, h: u32) -> DynamicImage {
    image.crop_imm(x, y, w, h)
}

/// Resize `patch` to `(w, h)` (bilinear) and overlay it onto `base` at `(x, y)`.
/// The inverse of `crop`: a face patch generated at model resolution is pasted
/// back into the frame at its detected box.
pub fn paste_resized(
    base: &mut DynamicImage,
    patch: &DynamicImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) {
    let resized = patch.resize_exact(w.max(1), h.max(1), FilterType::Triangle);
    image::imageops::overlay(base, &resized, x as i64, y as i64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, Rgb};

    #[test]
    fn crop_then_paste_roundtrip_dimensions() {
        let mut base = DynamicImage::new_rgb8(64, 48);
        let patch =
            DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(16, 16, Rgb([255, 0, 0])));
        let region = crop(&base, 8, 8, 16, 16);
        assert_eq!(region.dimensions(), (16, 16));
        paste_resized(&mut base, &patch, 8, 8, 16, 16);
        assert_eq!(base.get_pixel(8, 8), image::Rgba([255, 0, 0, 255]));
        assert_eq!(base.get_pixel(0, 0), image::Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn crop_clamps_out_of_bounds() {
        let base = DynamicImage::new_rgb8(32, 32);
        let region = crop(&base, 24, 24, 32, 32);
        assert_eq!(region.dimensions(), (8, 8));
    }
}
