#![allow(clippy::too_many_arguments)]

//! Standalone CPU harness for the engine's zen3 speech stack: zen3-ASR (audio -> text)
//! and zen3-TTS (text -> 24 kHz PCM). It links `hanzo-engine`'s `speech_models` directly
//! (no full serve/Loader path) so the pipelines can be exercised on real safetensors weights
//! without the CUDA engine build. Mirrors the musetalk-bench "link the engine modules" pattern.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use hanzo_audio::AudioInput;
use hanzo_ml::{DType, Device};
use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};
use hanzo_engine::speech_models::{
    Qwen3AsrConfig, Qwen3AsrPipeline, Qwen3TtsCodecConfig, Qwen3TtsConfig, Qwen3TtsPipeline,
    SpeechGenerationConfig, SpeechLoaderType,
};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(name = "zen3-serving", about = "zen3-ASR + zen3-TTS standalone harness")]
struct Cli {
    /// Run on CUDA (GB10 etc.). Requires the `cuda` build feature; falls back to CPU if absent.
    #[arg(long, global = true)]
    cuda: bool,
    /// CUDA device ordinal (0-based) when `--cuda` is set.
    #[arg(long, global = true, default_value_t = 0)]
    cuda_device: usize,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Transcribe a wav/audio file with zen3-ASR.
    Asr {
        /// zen-3-asr-0.6B model directory (config.json + model.safetensors + vocab/merges).
        #[arg(long)]
        model: PathBuf,
        /// Input audio file (wav or anything `AudioInput::from_bytes` decodes).
        #[arg(long)]
        audio: PathBuf,
        /// Max new tokens to decode.
        #[arg(long, default_value_t = 256)]
        max_new: usize,
        /// Force the output language (e.g. "Chinese", "English"); teacher-forces the
        /// model's `language <Lang><asr_text>` prefix so it can't mis-ID and decode
        /// the wrong language. Omit to let the model auto-detect.
        #[arg(long)]
        lang: Option<String>,
    },
    /// Synthesize text to a 24 kHz wav with zen3-TTS.
    Tts {
        /// zen-3-tts-0.6B model directory (config.json + model.safetensors + speech_tokenizer/).
        #[arg(long)]
        model: PathBuf,
        /// Text to synthesize.
        #[arg(long)]
        text: String,
        /// Output wav path.
        #[arg(long)]
        out: PathBuf,
        /// Max codec frames to generate.
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
    },
}

fn device(cuda: bool, ordinal: usize) -> Result<Device> {
    if !cuda {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        let dev = Device::new_cuda(ordinal)?;
        eprintln!("[device] CUDA:{ordinal}");
        Ok(dev)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
        anyhow::bail!("--cuda requested but binary was built without the `cuda` feature");
    }
}

/// mmap a set of safetensors files into a `ShardedVarBuilder` at the requested dtype.
fn load_vb(paths: &[PathBuf], dtype: DType, dev: &Device) -> Result<ShardedVarBuilder> {
    let predicate: Arc<dyn Fn(String) -> bool + Send + Sync> = Arc::new(|_| true);
    // Safety: inherited from memmap2; files are read-only for the process lifetime.
    let vb = unsafe { ShardedSafeTensors::sharded(paths, dtype, dev, None, predicate)? };
    Ok(vb)
}

/// Load `tokenizer.json` if present, else build a Qwen2 byte-level BPE from the repo's
/// `vocab.json` + `merges.txt` + `tokenizer_config.json` specials (zen3 repos ship the latter).
fn load_tokenizer(dir: &Path) -> Result<Tokenizer> {
    use ahash::AHashMap;
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDec;
    use tokenizers::models::bpe::BpeBuilder;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
    use tokenizers::processors::byte_level::ByteLevel as ByteLevelPost;
    use tokenizers::AddedToken;

    let tj = dir.join("tokenizer.json");
    if tj.exists() {
        return Tokenizer::from_file(tj).map_err(anyhow::Error::msg);
    }

    let vocab_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("vocab.json"))?)?;
    let mut vocab: AHashMap<String, u32> = AHashMap::new();
    for (k, v) in vocab_json.as_object().context("vocab.json not an object")? {
        vocab.insert(k.clone(), v.as_u64().context("vocab id not u64")? as u32);
    }
    let merges: Vec<(String, String)> = std::fs::read_to_string(dir.join("merges.txt"))?
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| {
            let mut it = l.splitn(2, ' ');
            Some((it.next()?.to_string(), it.next()?.to_string()))
        })
        .collect();

    let bpe = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(anyhow::Error::msg)?;
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevelPre::new(false, false, false)));
    tok.with_decoder(Some(ByteLevelDec::new(false, false, false)));
    tok.with_post_processor(Some(ByteLevelPost::new(false, false, false)));

    let tc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json"))?)?;
    if let Some(added) = tc.get("added_tokens_decoder").and_then(|v| v.as_object()) {
        let specials: Vec<AddedToken> = added
            .values()
            .filter_map(|v| v.get("content").and_then(|c| c.as_str()))
            .map(|c| AddedToken::from(c.to_string(), true))
            .collect();
        tok.add_special_tokens(&specials);
    }
    Ok(tok)
}

fn read_audio(path: &Path) -> Result<AudioInput> {
    let p = path.to_string_lossy();
    if p.ends_with(".wav") {
        AudioInput::read_wav(&p).map_err(|e| anyhow::anyhow!("read_wav: {e}"))
    } else {
        AudioInput::from_bytes(&std::fs::read(path)?).map_err(|e| anyhow::anyhow!("decode: {e}"))
    }
}

fn write_wav(path: &Path, pcm: &[f32], rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}

fn run_asr(model: &Path, audio: &Path, max_new: usize, lang: Option<&str>, dev: &Device) -> Result<()> {
    let cfg: Qwen3AsrConfig =
        serde_json::from_str(&std::fs::read_to_string(model.join("config.json"))?)
            .context("parse ASR config.json")?;
    eprintln!(
        "[asr] audio d_model={} layers={} heads={} | text hidden={} layers={} vocab={}",
        cfg.audio_config.d_model,
        cfg.audio_config.num_layers,
        cfg.audio_config.num_heads,
        cfg.text_config.hidden_size,
        cfg.text_config.num_hidden_layers,
        cfg.text_config.vocab_size,
    );

    let vb = load_vb(&[model.join("model.safetensors")], DType::F32, dev)?;
    let pipeline = Qwen3AsrPipeline::new(&cfg, vb).context("build ASR pipeline")?;
    let tok = load_tokenizer(model)?;

    let audio = read_audio(audio)?;
    eprintln!(
        "[asr] audio: {} samples sr={} ch={} ({:.2}s)",
        audio.samples.len(),
        audio.sample_rate,
        audio.channels,
        audio.samples.len() as f32 / (audio.sample_rate as f32 * audio.channels as f32),
    );

    if let Some(l) = lang {
        eprintln!("[asr] forcing language={l:?}");
    }
    let t0 = std::time::Instant::now();
    let text = pipeline
        .transcribe_with_language(&audio, &tok, None, lang, Some(max_new))
        .context("transcribe")?;
    eprintln!("[asr] decode took {:.1}s", t0.elapsed().as_secs_f32());
    println!("{text}");
    Ok(())
}

fn run_tts(model: &Path, text: &str, out: &Path, max_tokens: usize, dev: &Device) -> Result<()> {
    let cfg: Qwen3TtsConfig =
        serde_json::from_str(&std::fs::read_to_string(model.join("config.json"))?)
            .context("parse TTS config.json")?;
    let codec_cfg: Qwen3TtsCodecConfig = serde_json::from_str(&std::fs::read_to_string(
        model.join("speech_tokenizer").join("config.json"),
    )?)
    .context("parse codec speech_tokenizer/config.json")?;

    let vb = load_vb(&[model.join("model.safetensors")], DType::F32, dev)?;
    let codec_vb = load_vb(
        &[model.join("speech_tokenizer").join("model.safetensors")],
        DType::F32,
        dev,
    )?;
    let pipeline =
        Qwen3TtsPipeline::new(&cfg, &codec_cfg, vb, codec_vb).context("build TTS pipeline")?;

    // The pipeline `generate` entry expects the already-tokenized prompt as a space-separated
    // u32 id stream (it leaves BPE tokenization to the caller). Build the chat-template prompt
    // `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n` and tokenize it here.
    let tok = load_tokenizer(model)?;
    let prompt = format!(
        "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    );
    let enc = tok.encode(prompt, false).map_err(anyhow::Error::msg)?;
    let id_stream: String = enc
        .get_ids()
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("[tts] prompt tokenized to {} ids", enc.get_ids().len());

    let gen = match SpeechGenerationConfig::default(SpeechLoaderType::Qwen3Tts) {
        SpeechGenerationConfig::Qwen3Tts { temperature, top_p, top_k, .. } => {
            SpeechGenerationConfig::Qwen3Tts {
                max_tokens: Some(max_tokens),
                temperature,
                top_p,
                top_k,
            }
        }
        other => other,
    };

    let t0 = std::time::Instant::now();
    let outp = pipeline.generate(&id_stream, &gen).context("tts generate")?;
    eprintln!(
        "[tts] generated {} samples @ {} Hz in {:.1}s",
        outp.pcm.len(),
        outp.rate,
        t0.elapsed().as_secs_f32()
    );
    write_wav(out, &outp.pcm, outp.rate as u32)?;
    eprintln!("[tts] wrote {}", out.display());
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let dev = device(cli.cuda, cli.cuda_device)?;
    match cli.cmd {
        Cmd::Asr { model, audio, max_new, lang } => {
            run_asr(&model, &audio, max_new, lang.as_deref(), &dev)
        }
        Cmd::Tts { model, text, out, max_tokens } => run_tts(&model, &text, &out, max_tokens, &dev),
    }
}
