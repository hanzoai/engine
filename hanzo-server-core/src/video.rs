//! Video frame extraction: FFmpeg for general formats, `image` crate for GIFs.
//!
//! Both the HTTP server and the CLI use this module to decode video files
//! into frames suitable for multimodal model input.
//!
//! ## FFmpeg requirement
//!
//! For non-GIF formats (mp4, avi, mov, mkv, webm, etc.) the `ffmpeg` binary
//! must be installed and available on `$PATH`. If it is not found the module
//! falls back to GIF-only support via the `image` crate.
//!
//! Install FFmpeg:
//! - **Linux**: `apt install ffmpeg` / `dnf install ffmpeg`
//! - **macOS**: `brew install ffmpeg`
//! - **Windows**: <https://ffmpeg.org/download.html>
//!
//! See <https://hanzoai.github.io/engine/guides/models/video-setup/> for full details.

use anyhow::{bail, Context, Result};
use hanzo_engine::{sample_frame_indices, VideoInput};
use image::codecs::gif::GifDecoder;
use image::{AnimationDecoder, DynamicImage};
use std::io::Cursor;
use std::path::Path;
use tokio::{
    fs::{self, File},
    io::AsyncReadExt,
};

/// Default frames-per-second assumed when metadata is unavailable (e.g. GIF).
const DEFAULT_FPS: f64 = 24.0;

const FFMPEG_INSTALL_HELP: &str = "\
FFmpeg is required for video input (non-GIF formats). Install it:
  - Linux:   apt install ffmpeg  /  dnf install ffmpeg
  - macOS:   brew install ffmpeg
  - Windows: https://ffmpeg.org/download.html
See https://hanzoai.github.io/engine/guides/models/video-setup/ for details.";

/// Fetch video bytes from a URL, file path, or data URL, then decode into
/// a [`VideoInput`] (sampled frames + metadata).
///
/// Supports:
/// - HTTP/HTTPS URLs
/// - Local file paths (absolute or relative)
/// - `file://` URLs
/// - `data:video/...;base64,...` data URLs
///
/// GIF files are decoded with the `image` crate. All other formats require
/// FFmpeg.
/// Fetch raw bytes from an http(s) URL, a `file://` URL, an absolute file path,
/// or a `data:` URL. Shared by the video/image/audio decoders.
pub async fn fetch_bytes(url_unparsed: &str) -> Result<Vec<u8>> {
    let url = if let Ok(url) = url::Url::parse(url_unparsed) {
        url
    } else if File::open(url_unparsed).await.is_ok() {
        url::Url::from_file_path(std::path::absolute(url_unparsed)?)
            .map_err(|_| anyhow::anyhow!("Could not parse file path: {}", url_unparsed))?
    } else {
        bail!(
            "Invalid source '{}': not a valid URL (http/https/data) and file not found. \
             Use a full URL, a data URL, or an absolute file path.",
            url_unparsed
        )
    };

    if url.scheme() == "http" || url.scheme() == "https" {
        let resp = reqwest::get(url.clone())
            .await
            .context(format!("Failed to fetch: {url}"))?;
        Ok(resp.bytes().await?.to_vec())
    } else if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file path: {}", url))?;
        let mut f = File::open(&path)
            .await
            .context(format!("Could not open file: {}", path.display()))?;
        let metadata = fs::metadata(&path).await?;
        let mut buffer = vec![0; metadata.len() as usize];
        f.read_exact(&mut buffer).await?;
        Ok(buffer)
    } else if url.scheme() == "data" {
        let data_url = data_url::DataUrl::process(url.as_str())?;
        Ok(data_url.decode_to_vec()?.0)
    } else {
        bail!("Unsupported URL scheme: {}", url.scheme());
    }
}

pub async fn parse_video_url(url_unparsed: &str, num_frames: Option<usize>) -> Result<VideoInput> {
    let bytes = fetch_bytes(url_unparsed).await?;

    // Detect format
    let lower = url_unparsed.to_lowercase();
    let is_gif = lower.ends_with(".gif")
        || lower.contains("image/gif")
        || (bytes.len() >= 6 && &bytes[..6] == b"GIF89a")
        || (bytes.len() >= 6 && &bytes[..6] == b"GIF87a");

    if is_gif {
        decode_gif_frames(&bytes, num_frames)
    } else {
        decode_video_ffmpeg(&bytes, num_frames, url_unparsed).await
    }
}

/// Decode a GIF into frames using the `image` crate.
fn decode_gif_frames(bytes: &[u8], num_frames: Option<usize>) -> Result<VideoInput> {
    let decoder = GifDecoder::new(Cursor::new(bytes)).context("Failed to decode GIF")?;

    let raw_frames: Vec<_> = decoder.into_frames().collect::<Result<Vec<_>, _>>()?;
    let total = raw_frames.len();
    if total == 0 {
        bail!("GIF contains no frames");
    }

    // Estimate FPS from average frame delay
    let total_delay_ms: u32 = raw_frames
        .iter()
        .map(|f| {
            let (num, den) = f.delay().numer_denom_ms();
            (num * 1000).checked_div(den).unwrap_or(100)
        })
        .sum();
    let fps = if total_delay_ms > 0 {
        (total as f64 * 1000.0) / total_delay_ms as f64
    } else {
        DEFAULT_FPS
    };

    let indices = num_frames
        .map(|n| sample_frame_indices(total, n))
        .unwrap_or_else(|| (0..total).collect());
    let frames: Vec<DynamicImage> = indices
        .iter()
        .map(|&i| DynamicImage::ImageRgba8(raw_frames[i].buffer().clone()))
        .collect();

    Ok(VideoInput {
        frames,
        fps,
        total_num_frames: total,
        sampled_indices: indices,
    })
}

/// Decode a video file using FFmpeg subprocess.
///
/// 1. Probe with `ffprobe` for FPS and total frame count.
/// 2. Extract frames with `ffmpeg`.
/// 3. Load frames as images.
async fn decode_video_ffmpeg(
    bytes: &[u8],
    num_frames: Option<usize>,
    source_hint: &str,
) -> Result<VideoInput> {
    // Check ffmpeg availability
    let ffmpeg_ok = tokio::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .is_ok();

    if !ffmpeg_ok {
        bail!(
            "Cannot decode video '{}': FFmpeg not found.\n{}",
            source_hint,
            FFMPEG_INSTALL_HELP
        );
    }

    // Write to temp file
    let tmp_dir = std::env::temp_dir().join("hanzo_video");
    fs::create_dir_all(&tmp_dir).await?;
    let video_id = uuid::Uuid::new_v4();
    let input_path = tmp_dir.join(format!("{video_id}.video"));
    fs::write(&input_path, bytes).await?;

    // Probe video metadata with ffprobe
    let (fps, total_frames) = probe_video(&input_path).await.unwrap_or((DEFAULT_FPS, 0));

    // Create output directory
    let out_dir = tmp_dir.join(format!("{video_id}_frames"));
    fs::create_dir_all(&out_dir).await?;
    let output_pattern = format!("{}/frame_%010d.png", out_dir.display());

    let mut requested_indices = None;
    let effective_total = if let Some(num_frames) = num_frames {
        let effective_total = if total_frames > 0 {
            total_frames
        } else {
            num_frames
        };
        let indices = sample_frame_indices(effective_total, num_frames);
        let select_expr = indices
            .iter()
            .map(|i| format!("eq(n\\,{i})"))
            .collect::<Vec<_>>()
            .join("+");
        let mut command = tokio::process::Command::new("ffmpeg");
        command
            .arg("-i")
            .arg(input_path.to_str().unwrap())
            .arg("-vf")
            .arg(format!("select='{select_expr}'"))
            .arg("-vsync")
            .arg("vfr")
            .arg("-frames:v")
            .arg(indices.len().to_string())
            .arg(&output_pattern);
        requested_indices = Some(indices);
        let status = command
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .context("Failed to run ffmpeg")?;
        if !status.success() {
            let _ = fs::remove_file(&input_path).await;
            let _ = fs::remove_dir_all(&out_dir).await;
            bail!(
                "FFmpeg failed to extract frames from '{}' (exit code: {:?})",
                source_hint,
                status.code()
            );
        }
        effective_total
    } else {
        let mut command = tokio::process::Command::new("ffmpeg");
        command
            .arg("-i")
            .arg(input_path.to_str().unwrap())
            .arg("-vsync")
            .arg("vfr")
            .arg(&output_pattern);
        let status = command
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .context("Failed to run ffmpeg")?;
        if !status.success() {
            let _ = fs::remove_file(&input_path).await;
            let _ = fs::remove_dir_all(&out_dir).await;
            bail!(
                "FFmpeg failed to extract frames from '{}' (exit code: {:?})",
                source_hint,
                status.code()
            );
        }
        total_frames
    };

    let mut frame_paths = Vec::new();
    let mut entries = fs::read_dir(&out_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|ext| ext.to_string_lossy().eq_ignore_ascii_case("png"))
        {
            frame_paths.push(path);
        }
    }
    frame_paths.sort();

    // Load extracted frame images
    let mut frames = Vec::with_capacity(frame_paths.len());
    for frame_path in frame_paths {
        let frame_bytes = fs::read(&frame_path).await?;
        let img = image::load_from_memory(&frame_bytes).context(format!(
            "Failed to load extracted frame {}",
            frame_path.display()
        ))?;
        frames.push(img);
    }

    // Cleanup temp files
    let _ = fs::remove_file(&input_path).await;
    let _ = fs::remove_dir_all(&out_dir).await;

    if frames.is_empty() {
        bail!(
            "FFmpeg extracted 0 frames from '{}'. The file may be corrupt or empty.",
            source_hint
        );
    }

    let total_num_frames = if effective_total > 0 {
        effective_total
    } else {
        frames.len()
    };
    let actual_indices = match requested_indices {
        Some(indices) if frames.len() < indices.len() => {
            sample_frame_indices(total_num_frames, frames.len())
        }
        Some(indices) => indices,
        None => (0..frames.len()).collect(),
    };

    Ok(VideoInput {
        frames,
        fps,
        total_num_frames,
        sampled_indices: actual_indices,
    })
}

/// Use `ffprobe` to get FPS and total frame count for a video file.
async fn probe_video(path: &Path) -> Result<(f64, usize)> {
    // Get FPS
    let fps_output = tokio::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.to_str().unwrap(),
        ])
        .output()
        .await
        .context("Failed to run ffprobe for FPS")?;

    let fps_str = String::from_utf8_lossy(&fps_output.stdout);
    let fps = parse_fps_fraction(fps_str.trim()).unwrap_or(DEFAULT_FPS);

    // Get total frame count
    let count_output = tokio::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.to_str().unwrap(),
        ])
        .output()
        .await
        .context("Failed to run ffprobe for frame count")?;

    let count_str = String::from_utf8_lossy(&count_output.stdout);
    let total_frames: usize = count_str.trim().parse().unwrap_or(0);

    Ok((fps, total_frames))
}

/// Parse a fractional FPS string like "30000/1001" or "30" into f64.
fn parse_fps_fraction(s: &str) -> Option<f64> {
    if let Some((num, den)) = s.split_once('/') {
        let n: f64 = num.parse().ok()?;
        let d: f64 = den.parse().ok()?;
        if d > 0.0 {
            Some(n / d)
        } else {
            None
        }
    } else {
        s.parse().ok()
    }
}

/// Mux rendered frames + mono PCM into an H.264/AAC MP4, returning the bytes.
///
/// The inverse of `decode_video_ffmpeg`: write frames as a PNG sequence and the
/// PCM as raw `f32le`, then let `ffmpeg` encode. This lives in the handler layer,
/// never a pipeline (a pipeline emits frames; the handler containers them).
pub async fn mux(
    frames: &[DynamicImage],
    fps: f64,
    pcm: &[f32],
    sample_rate: u32,
) -> Result<Vec<u8>> {
    if frames.is_empty() {
        bail!("mux: no frames to encode");
    }
    let ffmpeg_ok = tokio::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .is_ok();
    if !ffmpeg_ok {
        bail!(
            "Cannot mux video: FFmpeg not found.\n{}",
            FFMPEG_INSTALL_HELP
        );
    }

    let tmp_dir = std::env::temp_dir().join("hanzo_mux");
    fs::create_dir_all(&tmp_dir).await?;
    let job = uuid::Uuid::new_v4();
    let frame_dir = tmp_dir.join(format!("{job}_frames"));
    fs::create_dir_all(&frame_dir).await?;

    for (i, frame) in frames.iter().enumerate() {
        let path = frame_dir.join(format!("frame_{i:010}.png"));
        frame
            .to_rgb8()
            .save_with_format(&path, image::ImageFormat::Png)
            .with_context(|| format!("mux: writing frame {i}"))?;
    }

    let has_audio = !pcm.is_empty();
    let audio_path = tmp_dir.join(format!("{job}.f32le"));
    if has_audio {
        let mut raw = Vec::with_capacity(pcm.len() * 4);
        for &s in pcm {
            raw.extend_from_slice(&s.to_le_bytes());
        }
        fs::write(&audio_path, &raw).await?;
    }

    let out_path = tmp_dir.join(format!("{job}.mp4"));
    let frame_pattern = frame_dir.join("frame_%010d.png");

    let mut command = tokio::process::Command::new("ffmpeg");
    command
        .arg("-y")
        .arg("-framerate")
        .arg(format!("{fps}"))
        .arg("-start_number")
        .arg("0")
        .arg("-i")
        .arg(&frame_pattern);
    if has_audio {
        command
            .arg("-f")
            .arg("f32le")
            .arg("-ar")
            .arg(sample_rate.to_string())
            .arg("-ac")
            .arg("1")
            .arg("-i")
            .arg(&audio_path)
            .arg("-c:a")
            .arg("aac")
            .arg("-shortest");
    }
    command
        .arg("-c:v")
        .arg("libx264")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(&out_path);

    let status = command
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .context("Failed to run ffmpeg (mux)")?;

    let frame_dir_c = frame_dir.clone();
    let audio_path_c = audio_path.clone();
    let out_path_c = out_path.clone();
    let cleanup = || async {
        let _ = fs::remove_dir_all(&frame_dir_c).await;
        let _ = fs::remove_file(&audio_path_c).await;
        let _ = fs::remove_file(&out_path_c).await;
    };

    if !status.success() {
        cleanup().await;
        bail!(
            "FFmpeg failed to mux video (exit code: {:?})",
            status.code()
        );
    }

    let bytes = fs::read(&out_path)
        .await
        .context("mux: reading output mp4")?;
    cleanup().await;
    if bytes.is_empty() {
        bail!("mux: ffmpeg produced an empty mp4");
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ffmpeg_present() -> bool {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn mux_frames_and_pcm_to_mp4() {
        if !ffmpeg_present() {
            eprintln!("[mux] ffmpeg absent; skipping");
            return;
        }
        let frames: Vec<DynamicImage> = (0..6)
            .map(|i| {
                let v = (i * 40) as u8;
                DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                    64,
                    64,
                    image::Rgb([v, 255 - v, v]),
                ))
            })
            .collect();
        let sr = 16_000u32;
        let pcm: Vec<f32> = (0..(sr / 5))
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sr as f32).sin() * 0.3)
            .collect();
        let mp4 = mux(&frames, 25.0, &pcm, sr).await.expect("mux");
        assert!(mp4.len() > 256, "mp4 too small ({} bytes)", mp4.len());
        // ISO-BMFF: bytes 4..8 are the 'ftyp' box type.
        assert_eq!(&mp4[4..8], b"ftyp", "missing ftyp box -> not a valid mp4");
    }

    #[test]
    fn test_parse_fps_fraction() {
        assert!((parse_fps_fraction("30000/1001").unwrap() - 29.97).abs() < 0.01);
        assert!((parse_fps_fraction("30").unwrap() - 30.0).abs() < 0.01);
        assert!((parse_fps_fraction("24/1").unwrap() - 24.0).abs() < 0.01);
        assert!(parse_fps_fraction("").is_none());
        assert!(parse_fps_fraction("abc").is_none());
    }
}
