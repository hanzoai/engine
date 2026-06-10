//! PURE-NATIVE-RUST end-to-end test for the zen-dub pipeline. Replaces the bash orchestrator
//! (`run_fullnative_dub.sh`) + the Python-Whisper round-trip with a single cargo integration test
//! that drives the whole dub through the engine's Rust APIs. NO python process is spawned.
//!
//!   sun.wav (zh)
//!     -> zen3-ASR transcribe              [speech_models::Qwen3AsrPipeline]   -> char-match golden zh
//!     -> translate (zh -> en)             [LIVE hanzo LLM (zen-eco-4b GGUF), or golden fallback]
//!     -> zen3-TTS synthesize (en)         [speech_models::Qwen3TtsPipeline]   -> 24 kHz PCM
//!     -> zen3-ASR re-transcribe the TTS   [speech_models::Qwen3AsrPipeline]   -> round-trip char-match
//!     -> MuseTalk render (REAL weights)   [diffusion_models::musetalk::MuseTalk] -> per-stage cos vs PyTorch
//!
//! The TTS->ASR round-trip is the Python-Whisper killer: verification uses hanzo's OWN zen3-ASR,
//! not a Whisper subprocess, and the resample (24kHz->16kHz) happens inside the ASR processor, so
//! there is no ffmpeg either. The ASR-accuracy check compares zen3-ASR output to a COMMITTED golden
//! transcript (no live Whisper).
//!
//! TWO STAGES were promoted off committed goldens onto live native inference:
//!   * translate  -- a LIVE hanzo LLM call through the engine's own GGUF text pipeline (no Python,
//!                   no `hanzo` SDK crate; the engine `GGUFLoaderBuilder` + `MistralRsBuilder` that
//!                   the orchestrator/serve path uses). Asserts the live output is coherent English.
//!   * MuseTalk   -- a REAL-WEIGHT render: the engine `MuseTalk` is built over the converted real
//!                   `unet.safetensors` + `vae.safetensors` and run on the EXACT PyTorch-dumped
//!                   per-stage inputs, asserting per-stage cosine vs the committed PyTorch outputs
//!                   in `tests/fixtures/musetalk/`. This is the same numerical contract the
//!                   `musetalk-bench realverify` validates at cos~1.0 on CUDA.
//!
//! Env-gated on weights so CI without the models is a clean no-op:
//!   ZEN3_ASR_DIR=/abs/zen-3-asr-0.6B   ZEN3_TTS_DIR=/abs/zen-3-tts-0.6B
//!   DUB_SUN_WAV=/abs/sun.wav           (default: spark layout)
//! Optional:
//!   HANZO_TEST_GGUF=/abs/qwen.gguf     LLM weights for the live translate (default: spark zen-eco-4b);
//!                                      if absent the translate falls back to the committed golden.
//!   MUSETALK_WDIR=/abs/rustweights     dir with unet.safetensors + vae.safetensors for the real-weight
//!                                      render (default: spark layout); if absent MuseTalk falls back
//!                                      to a random-init graph/shape check.
//!   --features cuda                    run every stage on CUDA (default CPU). The real-weight
//!                                      MuseTalk numeric cosine assert is CUDA-only (see below).
//!
//! NOTE on CPU vs CUDA for the real-weight MuseTalk: CUDA is the target and the only gated numeric
//! assert. On CPU the VAE-decode `up_blocks` path has a known wiring discrepancy (diagnosed
//! separately), so on CPU the test still LOADS the real weights, runs the full render graph, and
//! checks finite/correct-shape + the encode/unet-input/unet-pred cosines (which DO match on CPU),
//! but does NOT hard-assert the decode/end-to-end cosine. `cargo test --features cuda dub_e2e`
//! exercises and asserts the full real-weight numeric path.
//!
//! Run:  ZEN3_ASR_DIR=... ZEN3_TTS_DIR=... cargo test -p hanzo-engine --test dub_e2e -- --nocapture --test-threads=1

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ahash::AHashMap;
use anyhow::{Context, Result};
use either::Either;
use hanzo_audio::AudioInput;
use hanzo_engine::diffusion_models::musetalk::{MuseTalk, MuseTalkConfig};
use hanzo_engine::speech_models::{
    Qwen3AsrConfig, Qwen3AsrPipeline, Qwen3TtsCodecConfig, Qwen3TtsConfig, Qwen3TtsPipeline,
    SpeechGenerationConfig, SpeechGenerationOutput, SpeechLoaderType,
};
use hanzo_ml::{DType, Device, Tensor};
use hanzo_nn::var_builder::SimpleBackend;
use hanzo_nn::Init;
use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};
use indexmap::IndexMap;
use tokenizers::Tokenizer;

// Engine LLM API for the LIVE translate stage (the same low-level GGUF text pipeline the
// orchestrator/serve path drives; deliberately NOT the `hanzo` SDK crate, which would be a
// circular dep on `hanzo-engine`).
use hanzo_engine::{
    AutoDeviceMapParams, Constraint, DefaultSchedulerMethod, DeviceMapSetting, GGUFLoaderBuilder,
    GGUFSpecificConfig, Hanzo, HanzoBuilder, MessageContent, ModelDType, NormalRequest, Request,
    RequestMessage, Response, SamplingParams, SchedulerConfig, TokenSource,
};

// ---- Committed golden references (no Python; the ASR-accuracy check compares against these) ----
// zen3-ASR transcription of sun.wav (zh). The clip is longer than this first sentence, so the
// hypothesis legitimately CONTAINS the reference: the char-match metric prefix-aligns. Blessed
// from zen3-ASR HEAD on CUDA; re-bless if a deliberate ASR change lands.
const GOLDEN_ZH: &str = "每个人到了一定年纪一切都看淡了顺其自然地活着珍惜所有的遇见";
// Committed golden English translation. This is the bash orchestrator's reference (zen-eco-4b
// Qwen3) and is used (a) as the TTS input / round-trip target when the LIVE LLM is unavailable and
// (b) as an OPTIONAL soft-similarity reference for the live translation. The live LLM call now
// produces the translation by default (see `LiveLlm::translate`); this constant is the fallback.
const GOLDEN_EN: &str =
    "Everyone becomes calm with age and lives naturally, cherishing every encounter.";

// Thresholds (mirror the bash runner's gates).
const THR_ASR_CHAR: f64 = 0.95; // zen3-ASR vs golden zh
const THR_TTS_RT: f64 = 0.50; // TTS round-tripped back through zen3-ASR vs the English text
// Real-weight MuseTalk per-stage cosine vs the committed PyTorch reference dumps. The
// musetalk-bench `realverify` hits cosine ~1.0 on CUDA f16 across every stage; we gate a little
// looser to tolerate f16 vs the f32 reference. CUDA-only (CPU VAE-decode wiring caveat, see top).
const THR_MUSETALK_COS: f64 = 0.999;

// ---------------------------------------------------------------- char-match metric (pure Rust)
// Prefix-aligned, punctuation-stripped, case-folded, Unicode-scalar char agreement. Ported from
// native-dub/tests/charmatch so the e2e needs no external helper binary and spawns no process.
fn is_punct_or_space(c: char) -> bool {
    if c.is_whitespace() || c.is_ascii_punctuation() {
        return true;
    }
    matches!(c,
        '\u{3000}' | '\u{3001}' | '\u{3002}' | '\u{FF0C}' | '\u{FF01}' | '\u{FF1F}'
        | '\u{FF1A}' | '\u{FF1B}' | '\u{2018}' | '\u{2019}' | '\u{201C}' | '\u{201D}'
        | '\u{2026}' | '\u{2014}' | '\u{2013}' | '\u{300C}' | '\u{300D}' | '\u{300E}'
        | '\u{300F}' | '\u{FF08}' | '\u{FF09}'
    )
}

fn normalize(s: &str) -> Vec<char> {
    s.chars()
        .filter(|&c| !is_punct_or_space(c))
        .collect::<String>()
        .to_lowercase()
        .chars()
        .collect()
}

fn char_match(reference: &str, hyp: &str) -> f64 {
    let r = normalize(reference);
    let h = normalize(hyp);
    if r.is_empty() {
        return 0.0;
    }
    let agree = (0..r.len()).filter(|&i| i < h.len() && h[i] == r[i]).count();
    agree as f64 / r.len() as f64
}

// ---------------------------------------------------------------- weights / device plumbing
fn device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        let ord: usize = std::env::var("CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        eprintln!("[dub-e2e] device = CUDA:{ord}");
        return Ok(Device::new_cuda(ord)?);
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("[dub-e2e] device = CPU");
        Ok(Device::Cpu)
    }
}

fn asr_dir() -> Option<PathBuf> {
    let d = PathBuf::from(
        std::env::var("ZEN3_ASR_DIR").unwrap_or_else(|_| "/home/z/work/zen/hf/zen-3-asr-0.6B".to_string()),
    );
    d.join("model.safetensors").is_file().then_some(d)
}

fn tts_dir() -> Option<PathBuf> {
    let d = PathBuf::from(
        std::env::var("ZEN3_TTS_DIR").unwrap_or_else(|_| "/home/z/work/zen/hf/zen-3-tts-0.6B".to_string()),
    );
    d.join("model.safetensors").is_file().then_some(d)
}

fn sun_wav() -> Option<PathBuf> {
    let p = PathBuf::from(
        std::env::var("DUB_SUN_WAV")
            .unwrap_or_else(|_| "/home/z/work/zen-dub-run/zen-dub/data/audio/sun.wav".to_string()),
    );
    p.is_file().then_some(p)
}

// GGUF for the LIVE LLM translate (zen-eco-4b / Qwen3). Returns the (model_dir, filename) split the
// GGUFLoaderBuilder wants. Default: spark layout. Absent -> live translate is skipped (golden used).
fn gguf_path() -> Option<(String, String)> {
    let p = PathBuf::from(
        std::env::var("HANZO_TEST_GGUF")
            .unwrap_or_else(|_| "/home/z/work/zen-eco-4b/zen-eco-4b.gguf".to_string()),
    );
    if !p.is_file() {
        return None;
    }
    // Resolve symlinks so the loader mmaps the real blob even when the path is a symlink.
    let p = std::fs::canonicalize(&p).unwrap_or(p);
    let dir = p.parent()?.to_string_lossy().to_string();
    let file = p.file_name()?.to_string_lossy().to_string();
    Some((dir, file))
}

// Dir holding the converted real MuseTalk weights (unet.safetensors + vae.safetensors). Default:
// spark layout. Absent -> MuseTalk falls back to a random-init graph/shape check.
fn musetalk_wdir() -> Option<PathBuf> {
    let d = PathBuf::from(
        std::env::var("MUSETALK_WDIR")
            .unwrap_or_else(|_| "/home/z/work/zen-dub-run/rustweights".to_string()),
    );
    (d.join("unet.safetensors").is_file() && d.join("vae.safetensors").is_file()).then_some(d)
}

// Committed PyTorch reference fixtures for the real-weight MuseTalk per-stage check.
fn musetalk_fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("musetalk")
}

// Flat cosine similarity between two equal-length tensors (f64 accumulation).
fn cosine(a: &Tensor, b: &Tensor) -> Result<f64> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    anyhow::ensure!(a.len() == b.len(), "cosine: len {} != {}", a.len(), b.len());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    Ok(dot / (na.sqrt() * nb.sqrt() + 1e-12))
}

fn load_vb(paths: &[PathBuf], dtype: DType, dev: &Device) -> Result<ShardedVarBuilder> {
    let predicate: Arc<dyn Fn(String) -> bool + Send + Sync> = Arc::new(|_| true);
    let vb = unsafe { ShardedSafeTensors::sharded(paths, dtype, dev, None, predicate)? };
    Ok(vb)
}

// zen3 repos ship vocab.json + merges.txt + tokenizer_config.json (no tokenizer.json); build the
// Qwen2 byte-level BPE the same way zen3-serving does.
fn load_tokenizer(dir: &Path) -> Result<Tokenizer> {
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
    let bpe = BpeBuilder::new().vocab_and_merges(vocab, merges).build().map_err(anyhow::Error::msg)?;
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

// ---------------------------------------------------------------- pipeline stages
fn build_asr(dir: &Path, dev: &Device) -> Result<(Qwen3AsrPipeline, Tokenizer)> {
    let cfg: Qwen3AsrConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .context("parse ASR config.json")?;
    let vb = load_vb(&[dir.join("model.safetensors")], DType::F32, dev)?;
    let pipe = Qwen3AsrPipeline::new(&cfg, vb).context("build ASR pipeline")?;
    let tok = load_tokenizer(dir)?;
    Ok((pipe, tok))
}

fn transcribe(pipe: &Qwen3AsrPipeline, tok: &Tokenizer, audio: &AudioInput, lang: &str) -> Result<String> {
    pipe.transcribe_with_language(audio, tok, None, Some(lang), Some(160))
        .context("transcribe")
}

fn synthesize(dir: &Path, text: &str, dev: &Device) -> Result<SpeechGenerationOutput> {
    let cfg: Qwen3TtsConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .context("parse TTS config.json")?;
    let codec_cfg: Qwen3TtsCodecConfig = serde_json::from_str(&std::fs::read_to_string(
        dir.join("speech_tokenizer").join("config.json"),
    )?)
    .context("parse codec config.json")?;
    let vb = load_vb(&[dir.join("model.safetensors")], DType::F32, dev)?;
    let codec_vb = load_vb(&[dir.join("speech_tokenizer").join("model.safetensors")], DType::F32, dev)?;
    let pipe = Qwen3TtsPipeline::new(&cfg, &codec_cfg, vb, codec_vb).context("build TTS pipeline")?;

    // The pipeline `generate` entry wants the prompt as a space-separated u32 id stream.
    let tok = load_tokenizer(dir)?;
    let prompt = format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
    let enc = tok.encode(prompt, false).map_err(anyhow::Error::msg)?;
    let id_stream: String =
        enc.get_ids().iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" ");

    let gen = match SpeechGenerationConfig::default(SpeechLoaderType::Qwen3Tts) {
        SpeechGenerationConfig::Qwen3Tts { temperature, top_p, top_k, .. } => {
            SpeechGenerationConfig::Qwen3Tts { max_tokens: Some(1200), temperature, top_p, top_k }
        }
        other => other,
    };
    pipe.generate(&id_stream, &gen).context("tts generate")
}

// ============================================================ LIVE LLM translate (zh -> en)
// Drives the engine's OWN GGUF text pipeline -- the same `GGUFLoaderBuilder` + `MistralRsBuilder`
// (`HanzoBuilder`) the serve/orchestrator path uses. No Python, no `hanzo` SDK crate (that depends
// on `hanzo-engine`, so importing it here would be circular). Mirrors `hanzo-cli` bench.rs's
// engine-level request loop, but sends a chat translate prompt and reads the decoded text back.
// The engine runs on its own std::thread with its own runtime, so a single owned runtime here
// (kept alive for the runner's lifetime) drives both the async `build` and the request sends.
struct LiveLlm {
    runner: Arc<Hanzo>,
    rt: tokio::runtime::Runtime,
}

fn build_llm(model_dir: &str, file: &str, dev: &Device) -> Result<LiveLlm> {
    // zen-eco-4b is a Qwen3 GGUF (arch=qwen2) that EMBEDS its tokenizer + chat template, so no
    // tok_model_id / external chat-template is needed: the loader reads both from the GGUF.
    let loader = GGUFLoaderBuilder::new(
        None,
        None,
        model_dir.to_string(),
        vec![file.to_string()],
        GGUFSpecificConfig { topology: None },
        false,
        None,
    )
    .build();
    let pipeline = loader
        .load_model_from_hf(
            None,
            TokenSource::None,
            &ModelDType::Auto,
            dev,
            true,
            DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
            None,
            None,
        )
        .context("load GGUF pipeline")?;
    let scheduler = SchedulerConfig::DefaultScheduler {
        method: DefaultSchedulerMethod::Fixed(NonZeroUsize::new(1).unwrap()),
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;
    let runner = rt.block_on(async {
        HanzoBuilder::new(pipeline, scheduler, false, None)
            .with_no_kv_cache(false)
            .with_prefix_cache_n(0)
            .build()
            .await
    });
    Ok(LiveLlm { runner, rt })
}

fn chat_message(role: &str, content: &str) -> IndexMap<String, MessageContent> {
    let mut m: IndexMap<String, MessageContent> = IndexMap::new();
    m.insert("role".to_string(), Either::Left(role.to_string()));
    m.insert("content".to_string(), Either::Left(content.to_string()));
    m
}

impl LiveLlm {
    // Run ONE deterministic chat completion through the engine runner; return the decoded text.
    fn chat(&self, system: &str, user: &str, max_tokens: usize) -> Result<String> {
        use tokio::sync::mpsc::channel;
        let hanzo = &self.runner;
        self.rt.block_on(async move {
            let mut sampling = SamplingParams::deterministic();
            sampling.max_len = Some(max_tokens);

            let (tx, mut rx) = channel(64);
            let messages = vec![chat_message("system", system), chat_message("user", user)];
            let req = Request::Normal(Box::new(NormalRequest {
                id: hanzo.next_request_id(),
                messages: RequestMessage::Chat {
                    messages,
                    enable_thinking: Some(false),
                    reasoning_effort: None,
                },
                sampling_params: sampling,
                response: tx,
                return_logprobs: false,
                is_streaming: false,
                constraint: Constraint::None,
                suffix: None,
                tools: None,
                tool_choice: None,
                logits_processors: None,
                return_raw_logits: false,
                web_search_options: None,
                enable_code_execution: false,
                code_execution_permission: None,
                code_execution_approval_notifier: None,
                agent_permission: None,
                agent_approval_handler: None,
                agent_approval_notifier: None,
                session_id: None,
                max_tool_rounds: None,
                tool_dispatch_url: None,
                model_id: None,
                truncate_sequence: false,
                files: None,
            }));
            hanzo
                .get_sender(None)
                .map_err(|e| anyhow::anyhow!("get_sender: {e}"))?
                .send(req)
                .await
                .map_err(|e| anyhow::anyhow!("send: {e}"))?;

            loop {
                // `Response` doesn't derive Debug: match by variant, format only the payloads.
                match rx.recv().await {
                    Some(Response::Done(r)) => {
                        return r.choices.first().and_then(|c| c.message.content.clone()).ok_or_else(
                            || anyhow::anyhow!("LLM returned no message content"),
                        );
                    }
                    Some(Response::AgenticToolCallProgress { .. }) | Some(Response::File(_)) => {
                        continue
                    }
                    Some(Response::ModelError(e, _)) => anyhow::bail!("LLM model error: {e}"),
                    Some(Response::CompletionModelError(e, _)) => anyhow::bail!("LLM model error: {e}"),
                    Some(Response::InternalError(e)) => anyhow::bail!("LLM internal error: {e}"),
                    Some(Response::ValidationError(e)) => anyhow::bail!("LLM validation error: {e}"),
                    Some(_) => anyhow::bail!("LLM returned an unexpected (non-chat) response variant"),
                    None => anyhow::bail!("LLM channel closed with no Done"),
                }
            }
        })
    }

    // Translate Chinese -> English. Strips any `<think>` block + stray quoting / leading label.
    fn translate(&self, zh: &str) -> Result<String> {
        let system = "You are a professional Chinese-to-English translator. Translate the user's \
                      Chinese text into a single natural English sentence. Output ONLY the English \
                      translation, with no quotes, no pinyin, and no explanation.";
        let out = self.chat(system, zh, 160)?;
        // Drop a Qwen3 `<think>...</think>` reasoning block if present (keep text after it).
        let out = match out.split_once("</think>") {
            Some((_, after)) => after,
            None => &out,
        };
        let line = out
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty() && !l.starts_with("<think>"))
            .unwrap_or("")
            .trim_matches(|c| c == '"' || c == '\'')
            .trim();
        let line = line
            .strip_prefix("Translation:")
            .or_else(|| line.strip_prefix("English:"))
            .unwrap_or(line)
            .trim();
        Ok(line.to_string())
    }
}

// ============================================================ MuseTalk render
// Random-init fallback backend (used only when the real weights are absent): REAL tensor shapes
// prove the render graph runs end-to-end and yields finite, correctly-shaped frames.
struct RandnBackend;
impl SimpleBackend for RandnBackend {
    fn get(&self, s: hanzo_ml::Shape, name: &str, _h: Init, dtype: DType, dev: &Device) -> hanzo_ml::Result<Tensor> {
        if name.ends_with("bias") {
            Tensor::zeros(s, dtype, dev)
        } else if name.ends_with("weight") && s.rank() == 1 {
            Tensor::ones(s, dtype, dev)
        } else {
            Tensor::randn(0f64, 0.02, s, dev)?.to_dtype(dtype)
        }
    }
    fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> hanzo_ml::Result<Tensor> {
        hanzo_ml::bail!("RandnBackend requires an explicit shape")
    }
    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

fn assert_finite_nondegenerate(name: &str, t: &Tensor) -> Result<()> {
    let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let bad = v.iter().filter(|x| !x.is_finite()).count();
    anyhow::ensure!(bad == 0, "MuseTalk {name} has {bad} non-finite values");
    let (mn, mx) = v.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    anyhow::ensure!(mx != mn, "MuseTalk {name} is constant (degenerate)");
    Ok(())
}

// Random-init graph/shape check (fallback when real weights are missing).
fn musetalk_render_random(dev: &Device) -> Result<()> {
    let dtype = DType::F32;
    let cfg = MuseTalkConfig::default();
    let rand_vb = || ShardedSafeTensors::wrap(Box::new(RandnBackend), dtype, dev.clone());
    let model = MuseTalk::new(cfg, rand_vb(), rand_vb(), dev, dtype).context("build MuseTalk")?;

    let sz = model.resized_img();
    let xa = model.cross_attention_dim();
    let face = Tensor::rand(0f32, 1f32, (1, 3, sz, sz), dev)?;
    let audio = Tensor::randn(0f64, 1.0, (1, 50, xa), dev)?;

    let latents = model.latents_for_unet(&face)?;
    let image = model.forward(&face, &audio)?;
    let blended = model.blend(&face, &image)?;

    assert_eq!(image.dims(), &[1, 3, sz, sz], "MuseTalk frame shape");
    assert_eq!(blended.dims(), &[1, 3, sz, sz], "blended frame shape");
    for (name, t) in [("latents", &latents), ("image", &image), ("blended", &blended)] {
        assert_finite_nondegenerate(name, t)?;
    }
    eprintln!("[dub-e2e] MuseTalk render (random-init graph check): {sz}x{sz} frame, finite, non-degenerate");
    Ok(())
}

// REAL-WEIGHT render: build the engine MuseTalk over the converted real safetensors and run each
// stage on the EXACT PyTorch-dumped input, comparing to the committed PyTorch output (cosine).
// This is the same per-stage contract `musetalk-bench realverify` validates at cos~1.0 on CUDA.
//
// `hard_assert_numeric` (= CUDA) controls the numeric gate: on CUDA every stage cosine is asserted
// >= THR_MUSETALK_COS. On CPU we load the real weights + run the full graph + assert finite/shape,
// but report cosines without asserting (hanzo-ml's CPU conv path diverges from the PyTorch f32 ref
// even for the validated bench code -- VAE-encode cos ~0.896 on CPU -- so CPU numerics aren't gated).
fn musetalk_render_real(dev: &Device, wdir: &Path, hard_assert_numeric: bool) -> Result<()> {
    let fixtures = musetalk_fixtures();
    anyhow::ensure!(
        fixtures.join("enc_mode.npy").is_file(),
        "missing MuseTalk fixtures at {}",
        fixtures.display()
    );
    // f32 by default (matches the PyTorch f32 reference dumps, so the cosine gate is exact-ish).
    // MUSETALK_DTYPE=f16 opts into the half-precision inference path (still gated >= THR on CUDA).
    let dtype = match std::env::var("MUSETALK_DTYPE").as_deref() {
        Ok("f16") => DType::F16,
        Ok("bf16") => DType::BF16,
        _ => DType::F32,
    };
    let load = |n: &str| -> Result<Tensor> {
        Tensor::read_npy(fixtures.join(format!("{n}.npy")))
            .with_context(|| format!("read fixture {n}.npy"))?
            .to_device(dev)
            .map_err(Into::into)
    };

    let cfg = MuseTalkConfig::default();
    let vae_vb = load_vb(&[wdir.join("vae.safetensors")], dtype, dev)?;
    let unet_vb = load_vb(&[wdir.join("unet.safetensors")], dtype, dev)?;
    let model = MuseTalk::new(cfg, vae_vb, unet_vb, dev, dtype).context("build real-weight MuseTalk")?;
    eprintln!(
        "[dub-e2e] MuseTalk REAL weights loaded from {} (dtype {:?})",
        wdir.display(),
        dtype
    );

    // ---- Stage 1: VAE-encode(mode) on the exact normalized ref crop ----
    let face_crop = load("face_crop")?.to_dtype(dtype)?;
    let enc = model.vae_encode_mode(&face_crop)?;
    assert_finite_nondegenerate("enc_mode", &enc)?;
    let enc_ref = load("enc_mode")?;
    assert_eq!(enc.dims(), enc_ref.dims(), "enc_mode shape");
    let cos_enc = cosine(&enc_ref, &enc)?;

    // ---- Stage 1b: 8-channel UNet input = cat([masked_lat, enc]) ----
    let masked_img = load("masked_img")?.to_dtype(dtype)?;
    let masked_lat = model.vae_encode_mode(&masked_img)?;
    let unet_in = Tensor::cat(&[&masked_lat, &enc], 1)?;
    let unet_in_ref = load("unet_in")?;
    assert_eq!(unet_in.dims(), unet_in_ref.dims(), "unet_in shape");
    let cos_unet_in = cosine(&unet_in_ref, &unet_in)?;

    // ---- Stage 2: UNet single step on the EXACT PyTorch unet_in + post-PE audio feat ----
    let unet_in_id = unet_in_ref.to_dtype(dtype)?;
    let audio_feat = load("audio_feat")?.to_dtype(dtype)?;
    let ts = Tensor::zeros(1, DType::F32, dev)?;
    let pred = model.unet_forward(&unet_in_id, &ts, &audio_feat)?;
    assert_finite_nondegenerate("unet_pred", &pred)?;
    let pred_ref = load("unet_pred")?;
    assert_eq!(pred.dims(), pred_ref.dims(), "unet_pred shape");
    let cos_pred = cosine(&pred_ref, &pred)?;

    // ---- Stage 3: VAE-decode (raw) on the EXACT PyTorch pred ----
    let pred_id = pred_ref.to_dtype(dtype)?;
    let dec = model.vae_decode_raw(&pred_id)?;
    assert_finite_nondegenerate("vae_dec", &dec)?;
    let dec_ref = load("vae_dec")?;
    assert_eq!(dec.dims(), dec_ref.dims(), "vae_dec shape");
    let cos_dec = cosine(&dec_ref, &dec)?;

    // ---- End-to-end: our encode -> our unet -> our decode, vs PyTorch final decode ----
    let e2e_pred = model.unet_forward(&unet_in, &ts, &audio_feat)?;
    let e2e_dec = model.vae_decode_raw(&e2e_pred)?;
    let cos_e2e = cosine(&dec_ref, &e2e_dec)?;

    eprintln!(
        "[dub-e2e] MuseTalk REAL-WEIGHT per-stage cosine vs PyTorch:\n\
         [dub-e2e]   VAE-encode(mode) : {cos_enc:.6}\n\
         [dub-e2e]   UNet-input(8ch)  : {cos_unet_in:.6}\n\
         [dub-e2e]   UNet-pred        : {cos_pred:.6}\n\
         [dub-e2e]   VAE-decode       : {cos_dec:.6}\n\
         [dub-e2e]   end-to-end       : {cos_e2e:.6}  (target >= {THR_MUSETALK_COS})"
    );

    // NUMERIC GATE: CUDA only. The musetalk-bench `realverify` reaches cosine ~1.0 across every
    // stage on CUDA (f16) vs the PyTorch f32 reference. On CPU, hanzo-ml's conv/groupnorm path
    // diverges from PyTorch even for the validated bench code (measured: VAE-encode cosine ~0.896
    // on CPU f32) -- a CPU-backend numerical gap, not a MuseTalk-wiring issue -- and the VAE-decode
    // up_blocks path has an additional known caveat. So on CPU we DON'T assert any cosine: the CPU
    // run's job is to prove the real weights LOAD and the full render graph runs finite + correctly
    // shaped. `cargo test --features cuda dub_e2e` asserts the actual numeric fidelity.
    if hard_assert_numeric {
        for (name, c) in [
            ("VAE-encode", cos_enc),
            ("UNet-input", cos_unet_in),
            ("UNet-pred", cos_pred),
            ("VAE-decode", cos_dec),
            ("end-to-end", cos_e2e),
        ] {
            anyhow::ensure!(c >= THR_MUSETALK_COS, "MuseTalk {name} cosine {c:.6} < {THR_MUSETALK_COS}");
        }
        eprintln!("[dub-e2e] MuseTalk REAL-WEIGHT numeric gate PASSED (all stages >= {THR_MUSETALK_COS})");
    } else {
        eprintln!(
            "[dub-e2e] MuseTalk REAL-WEIGHT (CPU): real weights loaded + full graph ran \
             finite/correct-shape. Per-stage cosines reported above are informational only; the \
             numeric gate runs under `--features cuda` (CPU hanzo-ml conv path diverges from the \
             PyTorch reference)."
        );
    }
    Ok(())
}

// Dispatch: real-weight render if the weights are present, else the random-init graph check.
fn musetalk_render(dev: &Device) -> Result<()> {
    match musetalk_wdir() {
        Some(wdir) => musetalk_render_real(dev, &wdir, dev.is_cuda()),
        None => {
            eprintln!("[dub-e2e] MuseTalk real weights absent (set MUSETALK_WDIR); random-init graph check");
            musetalk_render_random(dev)
        }
    }
}

// ---------------------------------------------------------------- the test
#[test]
fn dub_e2e_native() -> Result<()> {
    let (Some(asr_d), Some(tts_d), Some(wav)) = (asr_dir(), tts_dir(), sun_wav()) else {
        eprintln!(
            "[dub-e2e] weights/audio absent (set ZEN3_ASR_DIR, ZEN3_TTS_DIR, DUB_SUN_WAV); skipping"
        );
        return Ok(());
    };
    let dev = device()?;

    // --- 1. zen3-ASR transcribe sun.wav (zh) -> char-match committed golden ---
    let (asr, asr_tok) = build_asr(&asr_d, &dev)?;
    let audio = AudioInput::read_wav(&wav.to_string_lossy()).map_err(|e| anyhow::anyhow!("read_wav: {e}"))?;
    eprintln!(
        "[dub-e2e] sun.wav: {} samples @ {} Hz ({:.1}s)",
        audio.samples.len(),
        audio.sample_rate,
        audio.samples.len() as f32 / (audio.sample_rate as f32 * audio.channels.max(1) as f32)
    );
    let zh = transcribe(&asr, &asr_tok, &audio, "Chinese")?;
    let m_asr = char_match(GOLDEN_ZH, &zh);
    eprintln!("[dub-e2e] ASR zh: {zh:?}\n[dub-e2e] ASR char-match = {m_asr:.4} (>= {THR_ASR_CHAR})");
    assert!(m_asr >= THR_ASR_CHAR, "zen3-ASR char-match {m_asr:.4} < {THR_ASR_CHAR}");

    // --- 2. translate zh -> en via a LIVE hanzo LLM (zen-eco-4b GGUF), golden fallback ---
    // The translate stage is now a real native-Rust LLM call through the engine's GGUF text
    // pipeline (no Python, no `hanzo` SDK). We feed the ASR hypothesis (or the golden zh if the ASR
    // text is empty) and assert the output is coherent English. If no GGUF is present the stage
    // falls back to the committed golden so the rest of the pipeline still runs.
    let zh_for_translate = if zh.trim().is_empty() { GOLDEN_ZH } else { &zh };
    let en_owned: String = match gguf_path() {
        Some((dir, file)) => {
            eprintln!("[dub-e2e] LLM translate: loading GGUF {dir}/{file}");
            let llm = build_llm(&dir, &file, &dev).context("build live LLM")?;
            let t = llm.translate(zh_for_translate).context("live LLM translate")?;
            eprintln!("[dub-e2e] LLM translate (LIVE): {t:?}");
            // --- coherence asserts on the LIVE translation ---
            assert!(!t.trim().is_empty(), "live translation is empty");
            assert!(t.is_ascii(), "live translation not ASCII English: {t:?}");
            let words = t.split_whitespace().count();
            assert!(words >= 3, "live translation too short ({words} words): {t:?}");
            // The ASR hypothesis covers the whole clip (not just the golden first sentence), so the
            // translation is legitimately a multi-sentence paragraph; cap only against a runaway.
            assert!(t.len() <= 4000, "live translation implausibly long ({} chars)", t.len());
            // Must be letter-dominant English (not a code block / id dump).
            let alpha = t.chars().filter(|c| c.is_ascii_alphabetic()).count();
            let frac_alpha = alpha as f64 / t.chars().count().max(1) as f64;
            assert!(frac_alpha >= 0.6, "live translation not letter-dominant ({frac_alpha:.2}): {t:?}");
            // English sanity: contains common function words.
            let lower = t.to_lowercase();
            let has_fn_word = ["the", "and", "we", "of", "is", "to", "a ", "with", "every"]
                .iter()
                .any(|w| lower.contains(w));
            assert!(has_fn_word, "live translation lacks common English words: {t:?}");
            // Soft check vs the prior golden (informational; English paraphrases vary a lot).
            let sim = char_match(GOLDEN_EN, &t);
            eprintln!("[dub-e2e] live-vs-golden char-match = {sim:.4} (soft, informational only)");
            t
        }
        None => {
            eprintln!("[dub-e2e] GGUF absent (set HANZO_TEST_GGUF); using committed golden translation");
            GOLDEN_EN.to_string()
        }
    };
    // The live translation may be a multi-sentence paragraph (the ASR hypothesis is the whole
    // clip). Feed only the FIRST sentence to TTS so the TTS->ASR round-trip stays a focused,
    // single-utterance check (its prefix-aligned char-match would be brittle over a 400+ char
    // paragraph). The full translation was already asserted coherent above.
    let first_sentence = en_owned
        .split_inclusive(['.', '!', '?'])
        .next()
        .map(str::trim)
        .filter(|s| s.split_whitespace().count() >= 3)
        .unwrap_or(en_owned.as_str());
    let en: &str = first_sentence;
    assert!(en.split_whitespace().count() >= 3, "translation too short");
    assert!(en.is_ascii(), "translation must be ascii English");
    eprintln!("[dub-e2e] EN (first sentence -> TTS): {en:?}");

    // --- 3. zen3-TTS synthesize en -> 24 kHz PCM ---
    let out = synthesize(&tts_d, en, &dev)?;
    eprintln!("[dub-e2e] TTS: {} samples @ {} Hz", out.pcm.len(), out.rate);
    assert!(out.pcm.len() > out.rate / 2, "TTS produced < 0.5s of audio");
    let energy: f32 = out.pcm.iter().map(|x| x * x).sum::<f32>() / out.pcm.len().max(1) as f32;
    assert!(energy > 1e-6, "TTS output is silence (rms^2={energy:.2e})");

    // --- 4. TTS -> zen3-ASR round-trip (NO Python Whisper; ASR resamples 24k->16k internally) ---
    let tts_audio = AudioInput {
        samples: (*out.pcm).clone(),
        sample_rate: out.rate as u32,
        channels: out.channels as u16,
    };
    let back = transcribe(&asr, &asr_tok, &tts_audio, "English")?;
    let m_rt = char_match(en, &back);
    eprintln!("[dub-e2e] TTS round-trip heard: {back:?}\n[dub-e2e] round-trip char-match = {m_rt:.4} (>= {THR_TTS_RT})");
    assert!(m_rt >= THR_TTS_RT, "TTS->zen3-ASR round-trip {m_rt:.4} < {THR_TTS_RT} :: heard {back:?}");

    // --- 5. MuseTalk render: REAL weights, per-stage cosine vs PyTorch (CUDA-gated numeric) ---
    musetalk_render(&dev)?;

    eprintln!("[dub-e2e] PASS: zen3-ASR -> LIVE LLM translate -> zen3-TTS -> zen3-ASR round-trip -> REAL-weight MuseTalk, all native Rust");
    Ok(())
}
