//! Thin, typed wrappers over the `ffmpeg`/`ffprobe` CLIs. Every media byte the
//! pipeline emits is produced here; nothing else spawns a process. Async via
//! tokio so many shot renders run concurrently.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;
use tokio::process::Command;

/// Locate a TrueType font for `drawtext`; None disables text overlays gracefully.
fn font() -> Option<&'static str> {
    const CANDIDATES: &[&str] = &[
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ];
    CANDIDATES.iter().copied().find(|p| Path::new(p).exists())
}

/// Whether this ffmpeg build ships the `drawtext` filter (static builds often
/// omit it — it needs freetype). Probed once and cached.
async fn has_drawtext() -> bool {
    static CACHE: tokio::sync::OnceCell<bool> = tokio::sync::OnceCell::const_new();
    *CACHE
        .get_or_init(|| async {
            Command::new("ffmpeg")
                .args(["-hide_banner", "-filters"])
                .output()
                .await
                .map(|o| String::from_utf8_lossy(&o.stdout).contains("drawtext"))
                .unwrap_or(false)
        })
        .await
}

async fn run(bin: &str, args: &[String]) -> Result<()> {
    let out = Command::new(bin)
        .args(args)
        .output()
        .await
        .with_context(|| format!("spawning {bin}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        bail!(
            "{bin} failed ({}):\n{}",
            out.status,
            err.lines().rev().take(6).collect::<Vec<_>>().join("\n")
        );
    }
    Ok(())
}

fn s(v: impl ToString) -> String {
    v.to_string()
}

/// Escape a string for use inside an ffmpeg filter `drawtext=text='...'`.
fn esc_text(t: &str) -> String {
    t.replace('\\', "\\\\")
        .replace(':', "\\:")
        .replace('\'', "\u{2019}")
        .replace('%', "\\%")
        .chars()
        .take(120)
        .collect()
}

/// Deterministic dark background color (hex) derived from a label, so an entity's
/// procedural card is stable across runs.
pub fn color_for(label: &str) -> String {
    let mut h: u32 = 2166136261;
    for b in label.bytes() {
        h = (h ^ b as u32).wrapping_mul(16777619);
    }
    // Keep it dark so overlaid white text is legible.
    let r = 24 + (h & 0x3f);
    let g = 24 + ((h >> 8) & 0x3f);
    let b = 24 + ((h >> 16) & 0x3f);
    format!("0x{r:02x}{g:02x}{b:02x}")
}

/// Draw a static procedural card (solid color + optional wrapped caption) as a PNG.
/// The dependency-free stand-in for the image-gen endpoint on a GPU-less box.
pub async fn procedural_card(
    text: &str,
    label: &str,
    width: usize,
    height: usize,
    out: &Path,
) -> Result<()> {
    let mut vf = String::new();
    if let (Some(f), true) = (font(), has_drawtext().await) {
        vf = format!(
            "drawtext=fontfile={f}:text='{}':fontcolor=white:fontsize={}:x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.4:boxborderw=20",
            esc_text(text),
            (width / 24).max(18)
        );
    }
    let mut args = vec![
        s("-y"),
        s("-f"),
        s("lavfi"),
        s("-i"),
        format!("color=c={}:s={}x{}", color_for(label), width, height),
        s("-frames:v"),
        s(1),
    ];
    if !vf.is_empty() {
        args.push(s("-vf"));
        args.push(vf);
    }
    args.push(out.to_string_lossy().into_owned());
    run("ffmpeg", &args).await
}

/// Ken-Burns a still into a silent clip of exactly `dur_s` seconds. Optional
/// `grade` is an extra filter fragment (the bible's color grade).
pub async fn kenburns_clip(
    image: &Path,
    dur_s: f32,
    width: usize,
    height: usize,
    fps: usize,
    grade: Option<&str>,
    out: &Path,
) -> Result<()> {
    let frames = ((dur_s * fps as f32).round() as i64).max(1);
    // zoompan works on a single input frame; loop it and drive a slow push-in.
    // Upscale first (integer dims) for zoom headroom, then render back to WxH.
    let mut chain = format!(
        "scale={sw}:{sh},zoompan=z='min(zoom+0.0006,1.4)':d={d}:s={w}x{h}:fps={fps}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',format=yuv420p",
        sw = width * 2, sh = height * 2, w = width, h = height, d = frames, fps = fps
    );
    if let Some(g) = grade.filter(|g| !g.trim().is_empty()) {
        chain.push(',');
        chain.push_str(g);
    }
    let args = vec![
        s("-y"),
        s("-loop"),
        s(1),
        s("-i"),
        image.to_string_lossy().into_owned(),
        s("-t"),
        s(dur_s),
        s("-r"),
        s(fps),
        s("-vf"),
        chain,
        s("-c:v"),
        s("libx264"),
        s("-pix_fmt"),
        s("yuv420p"),
        s("-preset"),
        s("veryfast"),
        s("-an"),
        out.to_string_lossy().into_owned(),
    ];
    run("ffmpeg", &args).await
}

/// Extract the last frame of a clip as a PNG (conditioning for `continue` shots).
pub async fn tail_frame(clip: &Path, out: &Path) -> Result<()> {
    let args = vec![
        s("-y"),
        s("-sseof"),
        s("-0.25"),
        s("-i"),
        clip.to_string_lossy().into_owned(),
        s("-update"),
        s(1),
        s("-frames:v"),
        s(1),
        out.to_string_lossy().into_owned(),
    ];
    run("ffmpeg", &args).await
}

/// Extract one frame at timestamp `t_s` as a PNG (coherence sampling).
pub async fn frame_at(clip: &Path, t_s: f32, out: &Path) -> Result<()> {
    let args = vec![
        s("-y"),
        s("-ss"),
        s(t_s.max(0.0)),
        s("-i"),
        clip.to_string_lossy().into_owned(),
        s("-update"),
        s(1),
        s("-frames:v"),
        s(1),
        out.to_string_lossy().into_owned(),
    ];
    run("ffmpeg", &args).await
}

/// Re-encode arbitrary (already-decoded) mp4 bytes to the pipeline's uniform
/// codec/geometry so the final concat can stream-copy. Used for WAN clips.
pub async fn normalize_clip(
    input: &Path,
    width: usize,
    height: usize,
    fps: usize,
    out: &Path,
) -> Result<()> {
    let args = vec![
        s("-y"),
        s("-i"), input.to_string_lossy().into_owned(),
        s("-vf"), format!("scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"),
        s("-r"), s(fps),
        s("-c:v"), s("libx264"),
        s("-preset"), s("veryfast"),
        s("-an"),
        out.to_string_lossy().into_owned(),
    ];
    run("ffmpeg", &args).await
}

/// A silent WAV of `dur_s` seconds (mono 44.1k) — placeholder audio bed.
pub async fn silence(dur_s: f32, out: &Path) -> Result<()> {
    let args = vec![
        s("-y"),
        s("-f"),
        s("lavfi"),
        s("-i"),
        s("anullsrc=r=44100:cl=mono"),
        s("-t"),
        s(dur_s),
        s("-c:a"),
        s("pcm_s16le"),
        out.to_string_lossy().into_owned(),
    ];
    run("ffmpeg", &args).await
}

/// A low, quiet sine of `dur_s` seconds — audible placeholder for a spoken line,
/// so mix placement (adelay/amix) is demonstrable without a TTS model.
pub async fn tone(dur_s: f32, freq: f32, out: &Path) -> Result<()> {
    let args = vec![
        s("-y"),
        s("-f"),
        s("lavfi"),
        s("-i"),
        format!("sine=frequency={freq}:sample_rate=44100:duration={dur_s}"),
        s("-af"),
        s("volume=0.12"),
        s("-c:a"),
        s("pcm_s16le"),
        out.to_string_lossy().into_owned(),
    ];
    run("ffmpeg", &args).await
}

/// Transcode raw audio bytes (e.g. engine TTS wav/mp3) into canonical PCM WAV.
pub async fn to_wav(input: &Path, out: &Path) -> Result<()> {
    let args = vec![
        s("-y"),
        s("-i"),
        input.to_string_lossy().into_owned(),
        s("-ar"),
        s(44100),
        s("-ac"),
        s(1),
        s("-c:a"),
        s("pcm_s16le"),
        out.to_string_lossy().into_owned(),
    ];
    run("ffmpeg", &args).await
}

/// One dialogue placement on the shot's timeline.
pub struct Placement {
    pub wav: std::path::PathBuf,
    pub start_s: f32,
}

/// Mix a scene music bed (optional) with delayed dialogue placements into a single
/// WAV of `total_s` seconds. Empty inputs -> silence.
pub async fn mix(bed: Option<&Path>, lines: &[Placement], total_s: f32, out: &Path) -> Result<()> {
    if bed.is_none() && lines.is_empty() {
        return silence(total_s, out).await;
    }
    let mut args = vec![s("-y")];
    let mut filter = String::new();
    let mut labels: Vec<String> = Vec::new();
    let mut idx = 0usize;

    if let Some(b) = bed {
        args.push(s("-i"));
        args.push(b.to_string_lossy().into_owned());
        filter.push_str(&format!("[{idx}:a]volume=0.35[bed];"));
        labels.push("[bed]".into());
        idx += 1;
    }
    for l in lines {
        args.push(s("-i"));
        args.push(l.wav.to_string_lossy().into_owned());
        let delay_ms = (l.start_s.max(0.0) * 1000.0).round() as i64;
        filter.push_str(&format!("[{idx}:a]adelay={delay_ms}:all=1[d{idx}];"));
        labels.push(format!("[d{idx}]"));
        idx += 1;
    }
    filter.push_str(&labels.join(""));
    filter.push_str(&format!(
        "amix=inputs={}:duration=longest:normalize=0,apad,atrim=0:{total_s},asetpts=N/SR/TB[out]",
        labels.len()
    ));

    args.extend([
        s("-filter_complex"),
        filter,
        s("-map"),
        s("[out]"),
        s("-ar"),
        s(44100),
        s("-ac"),
        s(1),
        s("-c:a"),
        s("pcm_s16le"),
        out.to_string_lossy().into_owned(),
    ]);
    run("ffmpeg", &args).await
}

/// Concat uniform clips in order via the demuxer (stream copy).
pub async fn concat(clips: &[std::path::PathBuf], out: &Path) -> Result<()> {
    if clips.is_empty() {
        bail!("nothing to concat");
    }
    let list = out.with_extension("concat.txt");
    let body: String = clips
        .iter()
        .map(|c| {
            format!(
                "file '{}'\n",
                c.canonicalize().unwrap_or_else(|_| c.clone()).display()
            )
        })
        .collect();
    std::fs::write(&list, body)?;
    let r = run(
        "ffmpeg",
        &[
            s("-y"),
            s("-f"),
            s("concat"),
            s("-safe"),
            s(0),
            s("-i"),
            list.to_string_lossy().into_owned(),
            s("-c"),
            s("copy"),
            out.to_string_lossy().into_owned(),
        ],
    )
    .await;
    let _ = std::fs::remove_file(&list);
    r
}

/// Mux a silent video with an audio track into the final mp4.
pub async fn mux(video: &Path, audio: &Path, out: &Path) -> Result<()> {
    let args = vec![
        s("-y"),
        s("-i"),
        video.to_string_lossy().into_owned(),
        s("-i"),
        audio.to_string_lossy().into_owned(),
        s("-c:v"),
        s("copy"),
        s("-c:a"),
        s("aac"),
        s("-b:a"),
        s("192k"),
        s("-map"),
        s("0:v:0"),
        s("-map"),
        s("1:a:0"),
        s("-shortest"),
        out.to_string_lossy().into_owned(),
    ];
    run("ffmpeg", &args).await
}

#[derive(Debug, Deserialize)]
pub struct Probe {
    pub format: ProbeFormat,
    #[serde(default)]
    pub streams: Vec<ProbeStream>,
}

#[derive(Debug, Deserialize)]
pub struct ProbeFormat {
    #[serde(default)]
    pub duration: String,
}

#[derive(Debug, Deserialize)]
pub struct ProbeStream {
    pub codec_type: String,
    #[serde(default)]
    pub codec_name: String,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
}

impl Probe {
    pub fn duration_s(&self) -> f32 {
        self.format.duration.parse().unwrap_or(0.0)
    }
}

/// ffprobe -> parsed format+streams.
pub async fn probe(path: &Path) -> Result<Probe> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            &path.to_string_lossy(),
        ])
        .output()
        .await
        .context("spawning ffprobe")?;
    if !out.status.success() {
        bail!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    Ok(serde_json::from_slice(&out.stdout)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_is_stable_and_dark() {
        let a = color_for("hero");
        assert_eq!(a, color_for("hero"));
        assert_ne!(a, color_for("villain"));
        assert!(a.starts_with("0x"));
    }

    #[test]
    fn esc_text_neutralizes_specials() {
        let e = esc_text("a:b'c%d");
        assert!(!e.contains(":") || e.contains("\\:"));
        assert!(!e.contains('\''));
    }
}
