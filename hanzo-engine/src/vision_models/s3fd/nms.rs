//! S3FD anchor decode + greedy NMS. Pure geometry, no weights, so it is unit
//! tested directly. Matches face_alignment's `bbox.decode` (variances 0.1/0.2)
//! and the standard IoU NMS the SFD detector applies post-softmax.

const VAR_CENTER: f32 = 0.1;
const VAR_SIZE: f32 = 0.2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl FaceBox {
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn iou(&self, other: &FaceBox) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// Scale corner coords by `s` (used to map detection-resolution boxes back to
    /// the original image when the input was downscaled).
    pub fn scaled(&self, s: f32) -> FaceBox {
        FaceBox {
            x1: self.x1 * s,
            y1: self.y1 * s,
            x2: self.x2 * s,
            y2: self.y2 * s,
            score: self.score,
        }
    }
}

/// Decode one anchor: `loc` is the 4-vector head output, the anchor is a square
/// of side `anchor` centered at `(cx, cy)`. Faithful to face_alignment's decode:
/// center offsets use variance 0.1, size uses exp() with variance 0.2.
pub(crate) fn decode_box(loc: [f32; 4], cx: f32, cy: f32, anchor: f32, score: f32) -> FaceBox {
    let pcx = cx + loc[0] * VAR_CENTER * anchor;
    let pcy = cy + loc[1] * VAR_CENTER * anchor;
    let pw = anchor * (loc[2] * VAR_SIZE).exp();
    let ph = anchor * (loc[3] * VAR_SIZE).exp();
    FaceBox {
        x1: pcx - pw / 2.0,
        y1: pcy - ph / 2.0,
        x2: pcx + pw / 2.0,
        y2: pcy + ph / 2.0,
        score,
    }
}

/// Greedy IoU NMS: keep the highest-scoring box, drop any later box overlapping a
/// kept box by more than `iou_thresh`.
pub(crate) fn nms(mut boxes: Vec<FaceBox>, iou_thresh: f32) -> Vec<FaceBox> {
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<FaceBox> = Vec::new();
    for cand in boxes {
        if keep.iter().all(|k| k.iou(&cand) <= iou_thresh) {
            keep.push(cand);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> FaceBox {
        FaceBox {
            x1,
            y1,
            x2,
            y2,
            score,
        }
    }

    #[test]
    fn iou_known_values() {
        let a = b(0.0, 0.0, 10.0, 10.0, 1.0);
        assert!((a.iou(&a) - 1.0).abs() < 1e-6);
        let half = b(5.0, 0.0, 15.0, 10.0, 1.0); // inter 50, union 150 -> 1/3
        assert!((a.iou(&half) - 1.0 / 3.0).abs() < 1e-4);
        let disjoint = b(100.0, 100.0, 110.0, 110.0, 1.0);
        assert_eq!(a.iou(&disjoint), 0.0);
    }

    #[test]
    fn decode_zero_loc_recovers_anchor() {
        let fb = decode_box([0.0; 4], 50.0, 60.0, 16.0, 0.9);
        assert!((fb.x1 - 42.0).abs() < 1e-4);
        assert!((fb.y1 - 52.0).abs() < 1e-4);
        assert!((fb.x2 - 58.0).abs() < 1e-4);
        assert!((fb.y2 - 68.0).abs() < 1e-4);
        assert!((fb.width() - 16.0).abs() < 1e-4);
    }

    #[test]
    fn nms_suppresses_overlap_keeps_disjoint() {
        let a = b(0.0, 0.0, 10.0, 10.0, 0.9);
        let dup = b(1.0, 1.0, 11.0, 11.0, 0.8); // high IoU with a
        let far = b(100.0, 100.0, 110.0, 110.0, 0.7);
        let keep = nms(vec![dup, a, far], 0.3);
        assert_eq!(keep.len(), 2);
        assert!(
            (keep[0].score - 0.9).abs() < 1e-6,
            "highest score kept first"
        );
        assert!((keep[1].score - 0.7).abs() < 1e-6, "disjoint box survives");
    }
}
