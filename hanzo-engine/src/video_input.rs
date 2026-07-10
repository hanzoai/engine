use image::DynamicImage;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// qwen-vl-utils frame-sampling constants (github.com/QwenLM/Qwen2.5-VL qwen_vl_utils/vision_process.py).
pub const DEFAULT_SAMPLE_FPS: f64 = 2.0;
const FRAME_FACTOR: usize = 2;
const FPS_MIN_FRAMES: usize = 4;
const FPS_MAX_FRAMES: usize = 768;

/// Decoded video input: a sequence of frames with metadata for timestamp generation.
///
/// Create from pre-decoded frames with [`VideoInput::from_frames`], or use the
/// server-core `parse_video_url` helper to decode from a video file (requires FFmpeg
/// for non-GIF formats).
#[derive(Clone, PartialEq)]
pub struct VideoInput {
    /// Decoded video frames (RGB images).
    pub frames: Vec<DynamicImage>,
    /// Frames per second of the *original* video. Used to compute per-frame
    /// timestamps for the prompt (e.g. `"00:05"`). Defaults to 24.0.
    pub fps: f64,
    /// Total number of frames in the original video before sampling.
    pub total_num_frames: usize,
    /// Indices of the frames that were sampled from the original video.
    /// Length must equal `frames.len()`.
    pub sampled_indices: Vec<usize>,
}

impl std::fmt::Debug for VideoInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoInput")
            .field("num_frames", &self.frames.len())
            .field("fps", &self.fps)
            .field("total_num_frames", &self.total_num_frames)
            .finish()
    }
}

impl VideoInput {
    /// Create a `VideoInput` from pre-decoded frames.
    ///
    /// `fps` is the original video frame rate (used for timestamp generation).
    /// If the frames were not sampled (i.e. all frames are provided), pass `None`
    /// for `sampled_indices` and they will default to `0..frames.len()`.
    pub fn from_frames(
        frames: Vec<DynamicImage>,
        fps: f64,
        sampled_indices: Option<Vec<usize>>,
    ) -> Self {
        let n = frames.len();
        let sampled_indices = sampled_indices.unwrap_or_else(|| (0..n).collect());
        Self {
            frames,
            fps,
            total_num_frames: *sampled_indices.last().unwrap_or(&0) + 1,
            sampled_indices,
        }
    }

    /// Compute per-frame timestamps in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn timestamps_secs(&self) -> Vec<f64> {
        self.sampled_indices
            .iter()
            .map(|&idx| idx as f64 / self.fps)
            .collect()
    }

    /// Format timestamps as `"mm:ss"` strings.
    #[allow(clippy::cast_possible_truncation)]
    pub fn timestamp_strings(&self) -> Vec<String> {
        self.timestamps_secs()
            .iter()
            .map(|&secs| {
                let minutes = (secs / 60.0) as u32;
                let seconds = (secs % 60.0) as u32;
                format!("{minutes:02}:{seconds:02}")
            })
            .collect()
    }

    /// Effective sampling rate of the retained frames, in the original video's time base.
    /// `second_per_grid_t = temporal_patch_size / sampled_fps` for Qwen VL temporal M-RoPE.
    #[allow(clippy::cast_precision_loss)]
    pub fn sampled_fps(&self) -> f64 {
        if self.total_num_frames == 0 {
            return self.fps;
        }
        self.frames.len() as f64 * self.fps / self.total_num_frames as f64
    }

    /// Compute a content hash for each frame (for prefix caching).
    pub fn frame_hashes(&self) -> Vec<u64> {
        self.frames
            .iter()
            .map(|img| {
                let mut hasher = DefaultHasher::new();
                img.as_bytes().hash(&mut hasher);
                hasher.finish()
            })
            .collect()
    }

    /// Compute a single hash representing the entire video (for prefix caching).
    pub fn video_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        for frame in &self.frames {
            frame.as_bytes().hash(&mut hasher);
        }
        self.fps.to_bits().hash(&mut hasher);
        hasher.finish()
    }
}

/// Number of frames to sample for a `target_fps` playback, matching qwen-vl-utils `smart_nframes`:
/// clamp `total/video_fps*target_fps` to `[FPS_MIN_FRAMES, min(FPS_MAX_FRAMES, total)]`, floor to `FRAME_FACTOR`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn smart_nframes(total_frames: usize, video_fps: f64, target_fps: f64) -> usize {
    if total_frames == 0 {
        return 0;
    }
    let min_frames = FPS_MIN_FRAMES.div_ceil(FRAME_FACTOR) * FRAME_FACTOR;
    let max_frames = (FPS_MAX_FRAMES.min(total_frames) / FRAME_FACTOR) * FRAME_FACTOR;
    let ideal = (total_frames as f64 / video_fps.max(f64::MIN_POSITIVE) * target_fps) as usize;
    let clamped = ideal
        .clamp(min_frames, max_frames.max(min_frames))
        .min(total_frames);
    (clamped / FRAME_FACTOR * FRAME_FACTOR).max(FRAME_FACTOR.min(total_frames))
}

/// Sample `num_frames` frame indices from a video with `total_frames` frames.
///
/// Matches the HF reference: `torch.linspace(0, total_frames - 1, num_frames).round().long()`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn sample_frame_indices(total_frames: usize, num_frames: usize) -> Vec<usize> {
    if num_frames == 0 || total_frames == 0 {
        return Vec::new();
    }
    let n = num_frames.min(total_frames);
    if n == 1 {
        return vec![0];
    }
    let step = (total_frames - 1) as f64 / (n - 1) as f64;
    (0..n).map(|i| (i as f64 * step).round() as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_frame_indices() {
        let indices = sample_frame_indices(96, 32);
        assert_eq!(indices.len(), 32);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 3);
        assert_eq!(indices[31], 95); // linspace spans the endpoint
    }

    #[test]
    fn test_smart_nframes() {
        // 30s @ 30fps, target 2fps -> 60 frames.
        assert_eq!(smart_nframes(900, 30.0, 2.0), 60);
        // short clip clamps up to FPS_MIN_FRAMES (4), floored to FRAME_FACTOR.
        assert_eq!(smart_nframes(10, 30.0, 2.0), 4);
        // always a multiple of FRAME_FACTOR.
        assert_eq!(smart_nframes(101, 30.0, 2.0) % 2, 0);
        // never exceeds total.
        assert!(smart_nframes(3, 30.0, 2.0) <= 3);
    }

    #[test]
    fn test_sampled_fps() {
        let vi = VideoInput {
            frames: vec![DynamicImage::new_rgb8(1, 1); 60],
            fps: 30.0,
            total_num_frames: 900,
            sampled_indices: (0..60).map(|i| i * 15).collect(),
        };
        assert!((vi.sampled_fps() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_sample_frame_indices_equal() {
        let indices = sample_frame_indices(5, 5);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_sample_frame_indices_more_than_total() {
        let indices = sample_frame_indices(3, 10);
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_timestamp_strings() {
        let vi = VideoInput {
            frames: Vec::new(),
            fps: 24.0,
            total_num_frames: 2880,
            sampled_indices: vec![0, 720, 1440, 2160],
        };
        let ts = vi.timestamp_strings();
        assert_eq!(ts, vec!["00:00", "00:30", "01:00", "01:30"]);
    }
}
