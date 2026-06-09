//! Milestone-1 smoke test for the Qwen3-TTS (zen3-tts) port.
//!
//! Run with: `cargo run --release --example zentts -p hanzo`

use std::time::Instant;

use anyhow::Result;
use hanzo::{speech_utils, SpeechLoaderType, SpeechModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = SpeechModelBuilder::new("zenlm/zen-3-tts-0.6B", SpeechLoaderType::Qwen3Tts)
        .with_logging()
        .build()
        .await?;

    let start = Instant::now();
    let text = "Hello, this is a test.";
    let (pcm, rate, channels) = model.generate_speech(text).await?;
    let dur = Instant::now().duration_since(start).as_secs_f32();

    // Waveform sanity stats.
    let n = pcm.len();
    let mut nan = 0usize;
    let mut peak = 0f32;
    let mut sumsq = 0f64;
    for &s in pcm.iter() {
        if s.is_nan() || s.is_infinite() {
            nan += 1;
        } else {
            peak = peak.max(s.abs());
            sumsq += (s as f64) * (s as f64);
        }
    }
    let rms = if n > 0 {
        (sumsq / n as f64).sqrt()
    } else {
        0.0
    };
    let secs = n as f32 / (rate as f32 * channels as f32);

    let clamped: Vec<f32> = pcm.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
    let mut out = std::fs::File::create("zentts_out.wav")?;
    speech_utils::write_pcm_as_wav(&mut out, &clamped, rate as u32, channels as u16)?;

    println!("=== zen3-tts milestone-1 smoke ===");
    println!("text          : {text:?}");
    println!("gen time      : {dur:.2}s");
    println!("samples       : {n}  ({secs:.2}s @ {rate} Hz, {channels}ch)");
    println!("NaN/Inf       : {nan}");
    println!("peak |amp|    : {peak:.4}");
    println!("RMS           : {rms:.4}");
    println!("wav written   : zentts_out.wav");
    Ok(())
}
