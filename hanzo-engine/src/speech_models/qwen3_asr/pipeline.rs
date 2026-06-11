#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! End-to-end Qwen3-ASR transcription: mel frontend -> AuT encoder -> audio
//! features spliced into a Qwen3 prompt -> greedy decode to text. Mirrors the
//! `Qwen3TtsPipeline` shape (own `ShardedVarBuilder`, self-contained `forward`).
//!
//! Decode here re-runs the full prefill per step (no KV-cache reuse yet), so it
//! is O(n^2) in output length. That is fine for verification and short clips;
//! KV-cache streaming is the remaining perf work.

use hanzo_audio::AudioInput;
use hanzo_ml::{DType, Device, IndexOp, Result, Tensor, D};
use hanzo_quant::ShardedVarBuilder;
use tokenizers::Tokenizer;

use super::audio::Qwen3AsrAudioProcessor;
use super::config::Qwen3AsrConfig;
use super::Qwen3AsrModel;

const IM_START: u32 = 151_644;
const IM_END: u32 = 151_645;
const ENDOFTEXT: u32 = 151_643;
/// `<asr_text>` — the model emits `language <LANG><asr_text><transcription>`;
/// everything up to and including this marker is metadata, not transcript.
const ASR_TEXT: u32 = 151_704;
const DEFAULT_SYSTEM: &str = "You are a speech recognition model.";
const DEFAULT_MAX_NEW_TOKENS: usize = 440;

pub struct Qwen3AsrPipeline {
    model: Qwen3AsrModel,
    processor: Qwen3AsrAudioProcessor,
    cfg: Qwen3AsrConfig,
    device: Device,
}

impl Qwen3AsrPipeline {
    pub fn new(cfg: &Qwen3AsrConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let processor = Qwen3AsrAudioProcessor::new(&cfg.audio_config);
        let model = Qwen3AsrModel::new(cfg, vb)?;
        Ok(Self {
            model,
            processor,
            cfg: cfg.clone(),
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Build the chat prompt ids: system block + user block (with `n_audio`
    /// `<|audio_pad|>` placeholders between audio start/end) + assistant cue.
    fn build_prompt(&self, tok: &Tokenizer, system: &str, n_audio: usize) -> Result<Vec<u32>> {
        let enc = |s: &str| -> Result<Vec<u32>> {
            Ok(tok.encode(s, false).map_err(hanzo_ml::Error::msg)?.get_ids().to_vec())
        };
        let mut ids = Vec::new();
        ids.push(IM_START);
        ids.extend(enc("system\n")?);
        ids.extend(enc(system)?);
        ids.push(IM_END);
        ids.extend(enc("\n")?);
        ids.push(IM_START);
        ids.extend(enc("user\n")?);
        ids.push(self.cfg.audio_start_token_id as u32);
        ids.extend(std::iter::repeat_n(self.cfg.audio_token_id as u32, n_audio));
        ids.push(self.cfg.audio_end_token_id as u32);
        ids.push(IM_END);
        ids.extend(enc("\n")?);
        ids.push(IM_START);
        ids.extend(enc("assistant\n")?);
        Ok(ids)
    }

    /// Per-batch RoPE start offset. The decoder's `forward_qk_norm_positions`
    /// kernel takes one starting offset per batch row (length == batch) and walks
    /// `offset..offset+seq_len` internally; we prefill the full prompt from 0.
    fn positions(&self) -> Result<Tensor> {
        Tensor::from_vec(vec![0u32], 1, &self.device).map_err(hanzo_ml::Error::msg)
    }

    /// Transcribe `audio` to text. Greedy decode (argmax) capped at
    /// `max_new_tokens` (default `DEFAULT_MAX_NEW_TOKENS`), stopping on EOS.
    pub fn transcribe(
        &self,
        audio: &AudioInput,
        tok: &Tokenizer,
        system: Option<&str>,
        max_new_tokens: Option<usize>,
    ) -> Result<String> {
        self.transcribe_with_language(audio, tok, system, None, max_new_tokens)
    }

    /// Like [`Self::transcribe`] but optionally teacher-forces the output
    /// language. The model emits `language <Lang><asr_text><transcript>`, doing
    /// its own (autoregressive) language ID first; on some clips that ID is wrong
    /// (e.g. a Chinese clip decoded as `language English`), which derails the
    /// whole transcript. Passing `language` (e.g. `"Chinese"`) seeds the assistant
    /// turn with `language <Lang><asr_text>` so decode is pinned to that language.
    pub fn transcribe_with_language(
        &self,
        audio: &AudioInput,
        tok: &Tokenizer,
        system: Option<&str>,
        language: Option<&str>,
        max_new_tokens: Option<usize>,
    ) -> Result<String> {
        let mel = self.processor.process(audio, &self.device)?;
        let audio_embeds = self.model.encode_audio(&mel)?;
        let n_audio = audio_embeds.dim(1)?;

        let system = system.unwrap_or(DEFAULT_SYSTEM);
        let max_new = max_new_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        let mut ids = self.build_prompt(tok, system, n_audio)?;

        // Teacher-force `language <Lang><asr_text>` so the model can't mis-ID the
        // language and decode the wrong-language transcript. These prefix tokens
        // are not part of the returned transcript (the `<asr_text>` split drops them).
        if let Some(lang) = language {
            let enc = |s: &str| -> Result<Vec<u32>> {
                Ok(tok.encode(s, false).map_err(hanzo_ml::Error::msg)?.get_ids().to_vec())
            };
            ids.extend(enc(&format!("language {lang}"))?);
            ids.push(ASR_TEXT);
        }

        let mut generated = Vec::new();
        for _ in 0..max_new {
            let seq = ids.len();
            let input_ids = Tensor::from_vec(ids.clone(), (1, seq), &self.device)?;
            let positions = self.positions()?;
            // No KV-cache reuse: every step re-runs the full prompt, so the audio
            // features must be re-merged into the placeholder span each time.
            // Encoder already ran once above; reuse `audio_embeds`.
            let logits = self
                .model
                .forward_with_audio(&input_ids, Some(&audio_embeds), &positions)?;
            let last = logits.i((0, seq - 1))?;
            let next = last.to_dtype(DType::F32)?.argmax(D::Minus1)?.to_scalar::<u32>()?;
            if next == IM_END || next == ENDOFTEXT {
                break;
            }
            generated.push(next);
            ids.push(next);
        }

        if std::env::var("ZEN3_ASR_DEBUG").is_ok() {
            eprintln!("[asr-dbg] n_audio={} generated_ids={:?}", n_audio, generated);
            eprintln!(
                "[asr-dbg] raw_decode={:?}",
                tok.decode(&generated, false).unwrap_or_default()
            );
        }
        // The model prefixes the transcript with `language <LANG><asr_text>`.
        // Return only the transcript: drop everything up to and including the
        // last `<asr_text>` marker (if the model emitted one).
        let transcript = match generated.iter().rposition(|&t| t == ASR_TEXT) {
            Some(i) => &generated[i + 1..],
            None => &generated[..],
        };
        tok.decode(transcript, true).map_err(hanzo_ml::Error::msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use std::path::PathBuf;
    use std::sync::Arc;

    /// Load `tokenizer.json` if present, else build a Qwen2 byte-level BPE from
    /// the repo's `vocab.json` + `merges.txt` + `tokenizer_config.json` specials.
    fn load_tokenizer(dir: &std::path::Path) -> anyhow::Result<Tokenizer> {
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
        for (k, v) in vocab_json.as_object().unwrap() {
            vocab.insert(k.clone(), v.as_u64().unwrap() as u32);
        }
        let merges: Vec<(String, String)> = std::fs::read_to_string(dir.join("merges.txt"))?
            .lines()
            .filter(|l| !l.starts_with("#") && !l.trim().is_empty())
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

    fn synthetic_tone(sample_rate: u32, secs: f32, hz: f32) -> AudioInput {
        let n = (sample_rate as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * hz * i as f32 / sample_rate as f32).sin())
            .collect();
        AudioInput {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    /// Loads the real zen-3-asr weights from `$ZEN3_ASR_DIR` and runs an
    /// end-to-end transcription on a short clip ($ZEN3_ASR_WAV or a synthetic
    /// tone). Asserts the forward graph yields finite logits and decode runs.
    /// Ignored by default; run with:
    ///   ZEN3_ASR_DIR=/path cargo test -p hanzo-engine qwen3_asr_e2e -- --ignored --nocapture
    #[test]
    #[ignore]
    fn qwen3_asr_e2e() {
        let dir = match std::env::var("ZEN3_ASR_DIR") {
            Ok(d) => PathBuf::from(d),
            Err(_) => {
                eprintln!("ZEN3_ASR_DIR not set; skipping");
                return;
            }
        };
        let device = Device::Cpu;
        let cfg: Qwen3AsrConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .expect("parse config.json");
        eprintln!(
            "config: audio d_model={} layers={} heads={} | text hidden={} layers={} vocab={}",
            cfg.audio_config.d_model,
            cfg.audio_config.num_layers,
            cfg.audio_config.num_heads,
            cfg.text_config.hidden_size,
            cfg.text_config.num_hidden_layers,
            cfg.text_config.vocab_size,
        );

        let vb = from_mmaped_safetensors(
            vec![dir.join("model.safetensors")],
            Vec::new(),
            Some(DType::F32),
            &device,
            vec![None],
            true,
            None,
            |_| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .expect("load safetensors");

        let pipeline = Qwen3AsrPipeline::new(&cfg, vb).expect("build pipeline");
        let tok = load_tokenizer(&dir).expect("load/build tokenizer");

        // Tokenizer cross-check: the prompt-piece ids must match HF.
        for s in ["system\n", "user\n", "assistant\n", "\n"] {
            let ids = tok.encode(s, false).unwrap().get_ids().to_vec();
            eprintln!("tok {s:?} -> {ids:?}");
        }

        let audio = match std::env::var("ZEN3_ASR_WAV") {
            Ok(p) => {
                if p.ends_with(".wav") {
                    AudioInput::read_wav(&p).expect("read wav")
                } else {
                    AudioInput::from_bytes(&std::fs::read(&p).expect("read file"))
                        .expect("decode audio")
                }
            }
            Err(_) => synthetic_tone(16_000, 1.5, 440.0),
        };
        eprintln!(
            "audio: {} samples sr={} ch={}",
            audio.samples.len(),
            audio.sample_rate,
            audio.channels
        );

        // Mel frontend stats.
        let mel = pipeline.processor.process(&audio, &device).unwrap();
        let (mb, mn, mt) = mel.dims3().unwrap();
        let melf: Vec<f32> = mel.flatten_all().unwrap().to_vec1().unwrap();
        let mmean = melf.iter().sum::<f32>() / melf.len() as f32;
        let mmin = melf.iter().cloned().fold(f32::INFINITY, f32::min);
        let mmax = melf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("mel: [{mb}, {mn}, {mt}] mean={mmean:.4} min={mmin:.3} max={mmax:.3}");

        // Encoder sanity: features must be finite and shaped [1, T/8-ish, hidden].
        let feats = pipeline.model.encode_audio(&mel).unwrap();
        let (b, t, h) = feats.dims3().unwrap();
        let flat: Vec<f32> = feats.flatten_all().unwrap().to_vec1().unwrap();
        let finite = flat.iter().all(|x| x.is_finite());
        let rms = (flat.iter().map(|x| x * x).sum::<f32>() / flat.len() as f32).sqrt();
        eprintln!("audio features: [{b}, {t}, {h}] finite={finite} rms={rms:.4}");
        assert!(finite, "audio features contain NaN/Inf");
        assert_eq!(h, cfg.text_config.hidden_size);
        if let Ok(p) = std::env::var("ZEN3_ASR_DUMP_EMBEDS") {
            let bytes: Vec<u8> = flat.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(&p, bytes).unwrap();
            eprintln!("dumped rust embeds [{b},{t},{h}] -> {p}");
        }

        // Step-0 logits / argmax cross-check against the HF reference.
        let n_audio = feats.dim(1).unwrap();
        let ids = pipeline.build_prompt(&tok, DEFAULT_SYSTEM, n_audio).unwrap();
        eprintln!("prompt len {} head {:?} tail {:?}", ids.len(), &ids[..12.min(ids.len())], &ids[ids.len().saturating_sub(12)..]);
        let seq = ids.len();
        let input_ids = Tensor::from_vec(ids.clone(), (1, seq), &device).unwrap();
        let positions = pipeline.positions().unwrap();
        let logits = pipeline
            .model
            .forward_with_audio(&input_ids, Some(&feats), &positions)
            .unwrap();
        let last = logits.i((0, seq - 1)).unwrap().to_dtype(DType::F32).unwrap();
        let lv: Vec<f32> = last.to_vec1().unwrap();
        let mut idx: Vec<usize> = (0..lv.len()).collect();
        idx.sort_by(|&a, &b| lv[b].partial_cmp(&lv[a]).unwrap());
        let top10: Vec<(usize, f32)> = idx[..10].iter().map(|&i| (i, lv[i])).collect();
        eprintln!("STEP0 top10: {top10:?}");

        let text = pipeline
            .transcribe(&audio, &tok, None, Some(48))
            .expect("transcribe");
        eprintln!("transcription: {text:?}");
    }

    /// Decoder-isolation cross-check: feed the *reference* audio embeds
    /// (dumped from the HF model as raw little-endian f32 [n_audio, hidden] at
    /// $ZEN3_ASR_REF_EMBEDS) straight into the Rust decoder and greedily decode.
    /// If the decoder is correct, the generated ids must match the HF reference.
    #[test]
    #[ignore]
    fn qwen3_asr_decoder_vs_ref() {
        let (dir, embeds_path) = match (
            std::env::var("ZEN3_ASR_DIR"),
            std::env::var("ZEN3_ASR_REF_EMBEDS"),
        ) {
            (Ok(d), Ok(e)) => (PathBuf::from(d), e),
            _ => {
                eprintln!("ZEN3_ASR_DIR / ZEN3_ASR_REF_EMBEDS not set; skipping");
                return;
            }
        };
        let device = Device::Cpu;
        let cfg: Qwen3AsrConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let vb = from_mmaped_safetensors(
            vec![dir.join("model.safetensors")],
            Vec::new(),
            Some(DType::F32),
            &device,
            vec![None],
            true,
            None,
            |_| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap();
        let pipeline = Qwen3AsrPipeline::new(&cfg, vb).unwrap();
        let tok = load_tokenizer(&dir).unwrap();

        let hidden = cfg.text_config.hidden_size;
        let raw = std::fs::read(&embeds_path).unwrap();
        let floats: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(floats.len() % hidden, 0);
        let n_audio = floats.len() / hidden;
        let ref_embeds = Tensor::from_vec(floats, (1, n_audio, hidden), &device).unwrap();
        eprintln!("ref embeds: [1, {n_audio}, {hidden}]");

        let ids = pipeline.build_prompt(&tok, DEFAULT_SYSTEM, n_audio).unwrap();
        let mut cur = ids.clone();
        let mut generated: Vec<u32> = Vec::new();
        for step in 0..48 {
            let seq = cur.len();
            let input_ids = Tensor::from_vec(cur.clone(), (1, seq), &device).unwrap();
            let positions = pipeline.positions().unwrap();
            let logits = pipeline
                .model
                .forward_with_audio(&input_ids, Some(&ref_embeds), &positions)
                .unwrap();
            let last = logits.i((0, seq - 1)).unwrap().to_dtype(DType::F32).unwrap();
            let lv: Vec<f32> = last.to_vec1().unwrap();
            if step == 0 {
                let mut idx: Vec<usize> = (0..lv.len()).collect();
                idx.sort_by(|&a, &b| lv[b].partial_cmp(&lv[a]).unwrap());
                eprintln!(
                    "STEP0 top5: {:?}",
                    idx[..5].iter().map(|&i| (i, lv[i])).collect::<Vec<_>>()
                );
            }
            let next = lv
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            if next == IM_END || next == ENDOFTEXT {
                break;
            }
            generated.push(next);
            cur.push(next);
        }
        eprintln!("RUST(decoder, ref-embeds) generated ids: {generated:?}");
        eprintln!(
            "transcription: {:?}",
            tok.decode(&generated, true).unwrap()
        );
    }
}
