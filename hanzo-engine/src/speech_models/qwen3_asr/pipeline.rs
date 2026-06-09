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
        let mel = self.processor.process(audio, &self.device)?;
        let audio_embeds = self.model.encode_audio(&mel)?;
        let n_audio = audio_embeds.dim(1)?;

        let system = system.unwrap_or(DEFAULT_SYSTEM);
        let max_new = max_new_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        let mut ids = self.build_prompt(tok, system, n_audio)?;

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

        tok.decode(&generated, true).map_err(hanzo_ml::Error::msg)
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

        let audio = match std::env::var("ZEN3_ASR_WAV") {
            Ok(p) => AudioInput::read_wav(&p).expect("read wav"),
            Err(_) => synthetic_tone(16_000, 1.5, 440.0),
        };

        // Encoder sanity: features must be finite and shaped [1, T/8-ish, hidden].
        let mel = pipeline.processor.process(&audio, &device).unwrap();
        let feats = pipeline.model.encode_audio(&mel).unwrap();
        let (b, t, h) = feats.dims3().unwrap();
        let flat: Vec<f32> = feats.flatten_all().unwrap().to_vec1().unwrap();
        let finite = flat.iter().all(|x| x.is_finite());
        let rms = (flat.iter().map(|x| x * x).sum::<f32>() / flat.len() as f32).sqrt();
        eprintln!("audio features: [{b}, {t}, {h}] finite={finite} rms={rms:.4}");
        assert!(finite, "audio features contain NaN/Inf");
        assert_eq!(h, cfg.text_config.hidden_size);

        let text = pipeline
            .transcribe(&audio, &tok, None, Some(48))
            .expect("transcribe");
        eprintln!("transcription: {text:?}");
    }
}
