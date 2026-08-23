#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use hanzo_ml::{DType, Device, IndexOp, Result, Tensor, D};
use hanzo_quant::ShardedVarBuilder;
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    SeedableRng,
};
use rand_isaac::Isaac64Rng;

mod config;
mod model;

pub use config::{CodecConfig, Qwen3TtsConfig};
#[cfg(test)]
pub(crate) use model::CodecDecoder;
use model::{causal_mask, Qwen3TtsModel};

use super::{SpeechGenerationConfig, SpeechGenerationOutput};

const CHANNELS: usize = 1;

pub struct Qwen3TtsPipeline {
    model: Qwen3TtsModel,
    cfg: Qwen3TtsConfig,
    device: Device,
    dtype: DType,
}

/// All the special-token embeddings + role/text/codec pieces needed to build the talker prefill.
struct PrefillParts {
    tts_bos: Tensor,
    tts_eos: Tensor,
    tts_pad: Tensor,
}

impl Qwen3TtsPipeline {
    pub fn new(
        cfg: &Qwen3TtsConfig,
        codec_cfg: &CodecConfig,
        vb: ShardedVarBuilder,
        codec_vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let model = Qwen3TtsModel::new(cfg, codec_cfg, vb, codec_vb)?;
        Ok(Self {
            model,
            cfg: cfg.clone(),
            device,
            dtype,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn codec_sample_rate(&self) -> usize {
        self.model.codec.sample_rate()
    }

    fn ids(&self, v: &[u32]) -> Result<Tensor> {
        Tensor::new(v, &self.device)?.unsqueeze(0)
    }

    /// `text_projection(text_embedding(ids))` for a (1, n) id tensor -> (1, n, hidden).
    fn text_proj(&self, ids: &Tensor) -> Result<Tensor> {
        self.model.talker.embed_text(ids)
    }

    /// `codec_embedding(ids)` for a (1, n) id tensor -> (1, n, hidden).
    fn codec_embed(&self, ids: &Tensor) -> Result<Tensor> {
        self.model.talker.embed_codec(ids)
    }

    fn special_embeds(&self) -> Result<PrefillParts> {
        let ids = self.ids(&[
            self.cfg.tts_bos_token_id,
            self.cfg.tts_eos_token_id,
            self.cfg.tts_pad_token_id,
        ])?;
        let e = self.text_proj(&ids)?;
        Ok(PrefillParts {
            tts_bos: e.i((.., 0..1, ..))?,
            tts_eos: e.i((.., 1..2, ..))?,
            tts_pad: e.i((.., 2..3, ..))?,
        })
    }

    /// Builds the non-streaming talker prefill for the base model with no speaker embedding,
    /// mirroring `Qwen3TTSForConditionalGeneration.generate` (non_streaming_mode=True, speaker None).
    /// `text_ids` is the full tokenized prompt `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`.
    /// Returns (inputs_embeds (1, T, H), trailing_text_hidden (1, 1, H) = tts_pad).
    #[cfg_attr(test, allow(dead_code))]
    pub(crate) fn build_prefill(
        &self,
        text_ids: &[u32],
        language_id: Option<u32>,
    ) -> Result<(Tensor, Tensor)> {
        let tc = &self.cfg.talker_config;
        let sp = self.special_embeds()?;

        let codec_prefill_0: Vec<u32> = match language_id {
            None => vec![
                tc.codec_nothink_id,
                tc.codec_think_bos_id,
                tc.codec_think_eos_id,
            ],
            Some(lid) => vec![
                tc.codec_think_id,
                tc.codec_think_bos_id,
                lid,
                tc.codec_think_eos_id,
            ],
        };
        let codec_prefill_1: Vec<u32> = vec![tc.codec_pad_id, tc.codec_bos_id];
        let mut codec_all = codec_prefill_0.clone();
        codec_all.extend_from_slice(&codec_prefill_1);
        let codec_input_embedding = self.codec_embed(&self.ids(&codec_all)?)?; // (1, n, H)
        let n = codec_input_embedding.dim(1)?;

        // role: first 3 text tokens (<|im_start|>, assistant, \n)
        let role = self.text_proj(&self.ids(&text_ids[0..3])?)?;

        // [tts_pad*(n-2), tts_bos] + codec_input_embedding[:, :-1]
        let pad_rep = sp
            .tts_pad
            .broadcast_as((1, n - 2, sp.tts_pad.dim(2)?))?
            .contiguous()?;
        let text_head = Tensor::cat(&[&pad_rep, &sp.tts_bos], 1)?;
        let codec_head = codec_input_embedding.narrow(1, 0, n - 1)?;
        let body = (text_head + codec_head)?;

        let mut talker_input = Tensor::cat(&[&role, &body], 1)?;

        // non-streaming, no-speaker branch:
        // text part: text_projection(text[3:-5]) ++ tts_eos ; codec part: codec_pad*(len+1)
        let text_len = text_ids.len();
        let body_text_ids = &text_ids[3..text_len - 5];
        let body_text = self.text_proj(&self.ids(body_text_ids)?)?;
        let text_with_eos = Tensor::cat(&[&body_text, &sp.tts_eos], 1)?;
        let pad_count = body_text_ids.len() + 1;
        let codec_pad_ids = vec![tc.codec_pad_id; pad_count];
        let codec_pad_emb = self.codec_embed(&self.ids(&codec_pad_ids)?)?;
        let chunk_a = (text_with_eos + codec_pad_emb)?;

        // (tts_pad + codec_bos)
        let codec_bos_emb = self.codec_embed(&self.ids(&[tc.codec_bos_id])?)?;
        let chunk_b = sp.tts_pad.broadcast_add(&codec_bos_emb)?;

        talker_input = Tensor::cat(&[&talker_input, &chunk_a, &chunk_b], 1)?;

        Ok((talker_input, sp.tts_pad))
    }

    fn sample(&self, logits: &Tensor, temperature: f32, rng: &mut Isaac64Rng) -> Result<u32> {
        if temperature == 0. {
            return logits.argmax(D::Minus1)?.to_scalar::<u32>();
        }
        let probs =
            hanzo_nn::ops::softmax_last_dim(&(logits.to_dtype(DType::F32)? / temperature as f64)?)?;
        let probs: Vec<f32> = probs.to_vec1()?;
        let distr = WeightedIndex::new(&probs).map_err(hanzo_ml::Error::msg)?;
        Ok(distr.sample(rng) as u32)
    }

    /// Suppress the last 1024 talker-vocab ids except codec_eos (matches the reference suppress_tokens).
    fn suppress_specials(&self, logits: &Tensor) -> Result<Tensor> {
        let tc = &self.cfg.talker_config;
        let v = tc.vocab_size;
        let lo = v - 1024;
        let mut row: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
        for (i, x) in row.iter_mut().enumerate().take(v).skip(lo) {
            if i as u32 != tc.codec_eos_token_id {
                *x = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(row, v, logits.device())
    }

    /// Given the talker hidden state for the current frame (group 0) and the sampled code0,
    /// autoregressively predict codes for groups 1..num_code_groups via the sub-talker.
    fn predict_groups(
        &self,
        talker_hidden_last: &Tensor,
        code0: u32,
        temperature: f32,
        rng: &mut Isaac64Rng,
    ) -> Result<Vec<u32>> {
        let groups = self.cfg.talker_config.num_code_groups;
        let mut codes = vec![code0];
        // sub-talker sequence starts with [talker_hidden, embed_group(1, code0)]
        let mut seq = Tensor::cat(
            &[
                talker_hidden_last,
                &self.model.talker.embed_codec(&self.ids(&[code0])?)?,
            ],
            1,
        )?;
        for group in 1..groups {
            let mask = causal_mask(seq.dim(1)?, self.dtype, &self.device)?;
            let logits = self.model.code_predictor.step(&seq, Some(&mask), group)?;
            let last = logits.i((0, logits.dim(1)? - 1, ..))?;
            let id = self.sample(&last, temperature, rng)?;
            codes.push(id);
            if group < groups - 1 {
                let emb = self
                    .model
                    .code_predictor
                    .embed_group(group, &self.ids(&[id])?)?;
                seq = Tensor::cat(&[&seq, &emb], 1)?;
            }
        }
        Ok(codes)
    }

    /// Full synthesis from already-tokenized prompt ids. Returns flattened codes (num_groups, T).
    pub fn generate_codes(
        &self,
        text_ids: &[u32],
        language_id: Option<u32>,
        max_tokens: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Vec<Vec<u32>>> {
        let tc = &self.cfg.talker_config;
        let groups = tc.num_code_groups;
        let eos = tc.codec_eos_token_id;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let (mut inputs_embeds, trailing_pad) = self.build_prefill(text_ids, language_id)?;
        let mut all_codes: Vec<Vec<u32>> = Vec::new();

        for _ in 0..max_tokens {
            let mask = causal_mask(inputs_embeds.dim(1)?, self.dtype, &self.device)?;
            let (hidden, logits) = self
                .model
                .talker
                .forward(&inputs_embeds, &[0], Some(&mask))?;
            let last_hidden = hidden.i((.., hidden.dim(1)? - 1.., ..))?;
            let last_logits = logits.i((0, logits.dim(1)? - 1, ..))?;
            let last_logits = self.suppress_specials(&last_logits)?;
            let code0 = self.sample(&last_logits, temperature, &mut rng)?;
            if code0 == eos {
                break;
            }
            let frame = self.predict_groups(&last_hidden, code0, temperature, &mut rng)?;

            // next talker input = sum of all group embeddings + trailing_text_hidden (tts_pad here)
            let mut summed = self.model.talker.embed_codec(&self.ids(&[code0])?)?;
            for (group, &id) in frame.iter().enumerate().skip(1) {
                summed = (summed
                    + self
                        .model
                        .code_predictor
                        .embed_group(group, &self.ids(&[id])?)?)?;
            }
            let next = summed.broadcast_add(&trailing_pad)?;
            all_codes.push(frame);
            inputs_embeds = Tensor::cat(&[&inputs_embeds, &next], 1)?;
        }

        // transpose to (num_groups, T)
        let t = all_codes.len();
        let mut grouped = vec![Vec::with_capacity(t); groups];
        for frame in &all_codes {
            for (g, &c) in frame.iter().enumerate() {
                grouped[g].push(c);
            }
        }
        Ok(grouped)
    }

    /// Decode codes (num_groups, T) -> PCM f32 vector at the codec sample rate.
    pub fn decode_codes(&self, grouped: &[Vec<u32>]) -> Result<Vec<f32>> {
        if grouped.is_empty() || grouped[0].is_empty() {
            return Ok(Vec::new());
        }
        let groups = grouped.len();
        let t = grouped[0].len();
        let mut flat = Vec::with_capacity(groups * t);
        for g in grouped {
            flat.extend_from_slice(g);
        }
        let codes = Tensor::from_vec(flat, (1, groups, t), &self.device)?;
        let pcm = self.model.codec.decode(&codes)?;
        pcm.i((0, 0))?.to_dtype(DType::F32)?.to_vec1::<f32>()
    }

    pub fn generate(
        &self,
        text: &str,
        gen_cfg: &SpeechGenerationConfig,
    ) -> Result<SpeechGenerationOutput> {
        let SpeechGenerationConfig::Qwen3Tts {
            max_tokens,
            temperature,
            top_p: _,
            top_k: _,
        } = gen_cfg
        else {
            hanzo_ml::bail!("Qwen3TtsPipeline requires a SpeechGenerationConfig::Qwen3Tts");
        };

        let max_tokens = max_tokens.unwrap_or(2048);
        // The pipeline-level entry expects `text` to already be the tokenized prompt ids encoded as
        // a space-separated list of u32 (the SpeechPipeline tokenizes via the Qwen3 BPE before this).
        let text_ids: Vec<u32> = text
            .split_whitespace()
            .filter_map(|s| s.parse::<u32>().ok())
            .collect();
        if text_ids.len() < 9 {
            hanzo_ml::bail!(
                "Qwen3TtsPipeline: prompt id stream too short ({})",
                text_ids.len()
            );
        }
        let grouped = self.generate_codes(&text_ids, None, max_tokens, *temperature, 0)?;
        let pcm = self.decode_codes(&grouped)?;
        Ok(SpeechGenerationOutput {
            pcm: Arc::new(pcm),
            rate: self.model.codec.sample_rate(),
            channels: CHANNELS,
        })
    }
}

#[cfg(test)]
mod codec_validation {
    use super::*;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use std::path::PathBuf;

    fn read_f32(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }
    fn read_i64(path: &str) -> Vec<i64> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect()
    }
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    // ---- committed PyTorch reference fixtures (no python at test time) ----
    // These tensors were dumped ONCE, offline, by reference/dump_tts_ref.py running the real
    // QwenLM/Qwen3-TTS PyTorch model on a fixed prompt+seed with a fully greedy decode, then
    // committed under tests/fixtures/zen3_tts/. The tests load the committed bytes and compare;
    // nothing spawns python. To re-bless after a deliberate math change, re-run dump_tts_ref.py
    // and re-copy its output here. The dims below are from that dump's meta.txt.
    const FX_TALKER_T: usize = 19; // talker prefill length (T_ids=17 + 2 role/codec-bos)
    const FX_GEN_T: usize = 48; // greedy frames generated (== codec T)
    const FX_Q: usize = 16; // residual code groups
    const FX_LANG_ID: u32 = 2050; // english language token id used to build the prefill

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/zen3_tts")
    }
    /// Resolve a reference tensor: prefer an explicit `$env` override (used by the offline
    /// re-bless flow), else the committed fixture `tests/fixtures/zen3_tts/<file>`.
    fn ref_path(env: &str, file: &str) -> String {
        std::env::var(env)
            .unwrap_or_else(|_| fixture_dir().join(file).to_string_lossy().into_owned())
    }
    /// The zen-3-tts model dir (1.8GB safetensors -- env-gated, NOT committed). Defaults to the
    /// spark layout. Returns None (test skips) if the weights are not on disk.
    fn tts_dir() -> Option<PathBuf> {
        let d = PathBuf::from(
            std::env::var("ZEN3_TTS_DIR")
                .unwrap_or_else(|_| "/home/z/work/zen/hf/zen-3-tts-0.6B".to_string()),
        );
        d.join("model.safetensors").is_file().then_some(d)
    }
    /// `$ZEN3_MAIN_WEIGHTS` override (re-bless flow) else `<tts_dir>/model.safetensors`.
    fn main_weights(dir: &std::path::Path) -> String {
        std::env::var("ZEN3_MAIN_WEIGHTS")
            .unwrap_or_else(|_| dir.join("model.safetensors").to_string_lossy().into_owned())
    }
    fn main_config(dir: &std::path::Path) -> String {
        std::env::var("ZEN3_MAIN_CONFIG")
            .unwrap_or_else(|_| dir.join("config.json").to_string_lossy().into_owned())
    }
    fn codec_weights(dir: &std::path::Path) -> String {
        std::env::var("ZEN3_CODEC_WEIGHTS").unwrap_or_else(|_| {
            dir.join("speech_tokenizer/model.safetensors")
                .to_string_lossy()
                .into_owned()
        })
    }
    fn codec_config(dir: &std::path::Path) -> String {
        std::env::var("ZEN3_CODEC_CONFIG").unwrap_or_else(|_| {
            dir.join("speech_tokenizer/config.json")
                .to_string_lossy()
                .into_owned()
        })
    }
    fn load_vb(path: &str, device: &Device) -> ShardedVarBuilder {
        from_mmaped_safetensors(
            vec![PathBuf::from(path)],
            Vec::new(),
            Some(DType::F32),
            device,
            vec![None],
            true,
            None,
            |_| true,
            std::sync::Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap()
    }

    /// Validates the Rust codec decoder against committed PyTorch reference tensors captured from
    /// the real zen-3-tts-0.6B speech_tokenizer. Reference tensors are committed fixtures (no
    /// python); only the codec weights are loaded from disk (env-gated via ZEN3_TTS_DIR).
    #[test]
    fn codec_matches_reference() {
        let Some(dir) = tts_dir() else {
            eprintln!("zen-3-tts weights absent (set ZEN3_TTS_DIR); skipping codec validation");
            return;
        };
        let weights = codec_weights(&dir);
        let cfg_path = codec_config(&dir);
        let codes_path = ref_path("ZEN3_CODES_QT", "codes_QT.i64");
        let q: usize = FX_Q;
        let t: usize = FX_GEN_T;

        let device = Device::Cpu;
        let codec_cfg: CodecConfig =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        let codec = CodecDecoder::new(&codec_cfg, load_vb(&weights, &device)).unwrap();

        let codes_i64 = read_i64(&codes_path);
        assert_eq!(codes_i64.len(), q * t, "codes len mismatch");
        let codes_u32: Vec<u32> = codes_i64.iter().map(|&v| v.max(0) as u32).collect();
        let codes = Tensor::from_vec(codes_u32, (1, q, t), &device).unwrap();

        let (quant, _pc, pretrans, up, wav) = codec.decode_debug(&codes).unwrap();

        // Every stage is checked against its committed fixture (the offline re-bless flow may
        // override the path via the matching ZEN3_REF_* env).
        let check = |name: &str, env: &str, file: &str, got: &Tensor| {
            let refv = read_f32(&ref_path(env, file));
            let gotv: Vec<f32> = got
                .flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(
                gotv.len(),
                refv.len(),
                "{name} len {} vs ref {}",
                gotv.len(),
                refv.len()
            );
            let cos = cosine(&gotv, &refv);
            let mad = max_abs_diff(&gotv, &refv);
            eprintln!(
                "[codec] {name}: cos={cos:.6} max_abs_diff={mad:.6} (n={})",
                gotv.len()
            );
            assert!(cos > 0.99, "{name} cosine {cos} below 0.99");
        };
        check("quant", "ZEN3_REF_QUANT", "ref_quant.f32", &quant);
        check(
            "pretrans",
            "ZEN3_REF_PRETRANS",
            "ref_pretrans.f32",
            &pretrans,
        );
        check("upsample", "ZEN3_REF_UPSAMPLE", "ref_upsample.f32", &up);
        check("wav", "ZEN3_REF_WAV", "ref_wav.f32", &wav);

        if let Ok(out) = std::env::var("ZEN3_OUT_WAV") {
            let samples: Vec<f32> = wav.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut bytes = Vec::with_capacity(samples.len() * 4);
            for s in samples {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            std::fs::write(out, bytes).unwrap();
        }
    }

    /// Validates the Rust talker + code_predictor forward passes against PyTorch reference tensors
    /// captured from zen-3-tts-0.6B. Feeds the exact prefill embeds PyTorch built and checks the
    /// talker hidden state, codebook-0 logits, and the greedy frame-0 codes (all groups).
    ///   ZEN3_MAIN_WEIGHTS=.../model.safetensors  ZEN3_MAIN_CONFIG=.../config.json
    ///   ZEN3_TK_PREFILL=.../tk_prefill.f32  ZEN3_TK_T=20
    ///   ZEN3_TK_HIDDEN/ZEN3_TK_LOGITS (raw f32, optional)
    ///   ZEN3_TK_FRAME0=.../tk_frame0_codes.i64 (16 i64, optional)
    #[test]
    fn talker_matches_reference() {
        let Some(dir) = tts_dir() else {
            eprintln!("zen-3-tts weights absent (set ZEN3_TTS_DIR); skipping talker validation");
            return;
        };
        let weights = main_weights(&dir);
        let cfg_path = main_config(&dir);
        let prefill_path = ref_path("ZEN3_TK_PREFILL", "tk_prefill.f32");
        let pt: usize = FX_TALKER_T;

        let device = Device::Cpu;
        let cfg: Qwen3TtsConfig =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        let h = cfg.talker_config.hidden_size;
        let vb = load_vb(&weights, &device);
        let talker = model::Talker::new(&cfg.talker_config, vb.pp("talker")).unwrap();
        let code_predictor = model::CodePredictor::new(
            &cfg.talker_config.code_predictor_config,
            vb.pp("talker").pp("code_predictor"),
        )
        .unwrap();

        let prefill_data = read_f32(&prefill_path);
        assert_eq!(prefill_data.len(), pt * h);
        let prefill = Tensor::from_vec(prefill_data, (1, pt, h), &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        let mask = causal_mask(pt, DType::F32, &device).unwrap();
        let (hidden, logits) = talker.forward(&prefill, &[0], Some(&mask)).unwrap();

        let cmp = |name: &str, env: &str, file: &str, got: &Tensor| {
            let refv = read_f32(&ref_path(env, file));
            let gotv: Vec<f32> = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(gotv.len(), refv.len(), "{name} len");
            let cos = cosine(&gotv, &refv);
            let mad = max_abs_diff(&gotv, &refv);
            eprintln!("[talker] {name}: cos={cos:.6} max_abs_diff={mad:.6}");
            assert!(cos > 0.999, "{name} cosine {cos}");
        };
        cmp("hidden", "ZEN3_TK_HIDDEN", "tk_hidden.f32", &hidden);
        cmp("logits", "ZEN3_TK_LOGITS", "tk_logits.f32", &logits);

        // greedy frame-0: argmax code0, then greedy groups 1..G via code_predictor.
        let groups = cfg.talker_config.num_code_groups;
        let last_logits = logits.i((0, pt - 1, ..)).unwrap();
        let code0 = last_logits
            .argmax(D::Minus1)
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        let last_hidden = hidden.i((.., pt - 1.., ..)).unwrap();
        let mut frame = vec![code0];
        let ids = |v: u32| Tensor::new(&[v], &device).unwrap().unsqueeze(0).unwrap();
        let mut seq = Tensor::cat(
            &[&last_hidden, &talker.embed_codec(&ids(code0)).unwrap()],
            1,
        )
        .unwrap();
        for group in 1..groups {
            let m = causal_mask(seq.dim(1).unwrap(), DType::F32, &device).unwrap();
            let lg = code_predictor.step(&seq, Some(&m), group).unwrap();
            let last = lg.i((0, lg.dim(1).unwrap() - 1, ..)).unwrap();
            let id = last.argmax(D::Minus1).unwrap().to_scalar::<u32>().unwrap();
            frame.push(id);
            if group < groups - 1 {
                let emb = code_predictor.embed_group(group, &ids(id)).unwrap();
                seq = Tensor::cat(&[&seq, &emb], 1).unwrap();
            }
        }
        eprintln!("[talker] rust greedy frame0: {frame:?}");
        let refc: Vec<u32> = read_i64(&ref_path("ZEN3_TK_FRAME0", "tk_frame0_codes.i64"))
            .iter()
            .map(|&v| v as u32)
            .collect();
        assert_eq!(frame, refc, "frame0 codes mismatch");
        eprintln!("[talker] frame0 codes MATCH reference");
    }

    /// Validates the Rust prefill construction against PyTorch's `tk_prefill`. Builds the full
    /// pipeline and compares `build_prefill(input_ids, english)` to the reference embeds.
    ///   ZEN3_MAIN_WEIGHTS, ZEN3_MAIN_CONFIG, ZEN3_CODEC_WEIGHTS, ZEN3_CODEC_CONFIG
    ///   ZEN3_INPUT_IDS=.../input_ids.i64  ZEN3_LANG_ID=2050
    ///   ZEN3_TK_PREFILL=.../tk_prefill.f32  ZEN3_TK_T=20
    #[test]
    fn prefill_matches_reference() {
        let Some(dir) = tts_dir() else {
            eprintln!("zen-3-tts weights absent (set ZEN3_TTS_DIR); skipping prefill validation");
            return;
        };
        let weights = main_weights(&dir);
        let cfg_path = main_config(&dir);
        let codec_w = codec_weights(&dir);
        let codec_c = codec_config(&dir);
        let ids_path = ref_path("ZEN3_INPUT_IDS", "input_ids.i64");
        let lang_id: u32 = FX_LANG_ID;
        let prefill_path = ref_path("ZEN3_TK_PREFILL", "tk_prefill.f32");
        let pt: usize = FX_TALKER_T;

        let device = Device::Cpu;
        let cfg: Qwen3TtsConfig =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        let codec_cfg: CodecConfig =
            serde_json::from_str(&std::fs::read_to_string(&codec_c).unwrap()).unwrap();
        let vb = load_vb(&weights, &device);
        let cvb = from_mmaped_safetensors(
            vec![PathBuf::from(&codec_w)],
            Vec::new(),
            Some(DType::F32),
            &device,
            vec![None],
            true,
            None,
            |_| true,
            std::sync::Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap();
        let pipe = Qwen3TtsPipeline::new(&cfg, &codec_cfg, vb, cvb).unwrap();

        let ids: Vec<u32> = read_i64(&ids_path).iter().map(|&v| v as u32).collect();
        let (prefill, _trailing) = pipe.build_prefill(&ids, Some(lang_id)).unwrap();
        let h = cfg.talker_config.hidden_size;
        assert_eq!(prefill.dims3().unwrap(), (1, pt, h), "prefill shape");

        let refv = read_f32(&prefill_path);
        let gotv: Vec<f32> = prefill.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cos = cosine(&gotv, &refv);
        let mad = max_abs_diff(&gotv, &refv);
        eprintln!("[prefill] cos={cos:.6} max_abs_diff={mad:.6}");
        assert!(cos > 0.9999, "prefill cosine {cos}");
    }

    /// End-to-end greedy generation: runs the full Rust talker+code_predictor loop and compares the
    /// produced code sequence to PyTorch's greedy reference (codebook-0 column), validating the
    /// autoregressive feedback, EOS handling, and trailing-pad embedding.
    ///   ZEN3_MAIN_WEIGHTS, ZEN3_MAIN_CONFIG, ZEN3_CODEC_WEIGHTS, ZEN3_CODEC_CONFIG
    ///   ZEN3_INPUT_IDS, ZEN3_LANG_ID, ZEN3_GREEDY_TQ (T*Q i64), ZEN3_GREEDY_T, ZEN3_GREEDY_Q
    #[test]
    fn full_generation_greedy_matches_reference() {
        let Some(dir) = tts_dir() else {
            eprintln!(
                "zen-3-tts weights absent (set ZEN3_TTS_DIR); skipping full-generation validation"
            );
            return;
        };
        let weights = main_weights(&dir);
        let cfg_path = main_config(&dir);
        let codec_w = codec_weights(&dir);
        let codec_c = codec_config(&dir);
        let ids_path = ref_path("ZEN3_INPUT_IDS", "input_ids.i64");
        let lang_id: u32 = FX_LANG_ID;
        let greedy_path = ref_path("ZEN3_GREEDY_TQ", "greedy_TQ.i64");
        let rt: usize = FX_GEN_T;
        let rq: usize = FX_Q;

        let device = Device::Cpu;
        let cfg: Qwen3TtsConfig =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        let codec_cfg: CodecConfig =
            serde_json::from_str(&std::fs::read_to_string(&codec_c).unwrap()).unwrap();
        let pipe = Qwen3TtsPipeline::new(
            &cfg,
            &codec_cfg,
            load_vb(&weights, &device),
            load_vb(&codec_w, &device),
        )
        .unwrap();

        let ids: Vec<u32> = read_i64(&ids_path).iter().map(|&v| v as u32).collect();
        // temperature 0 = greedy
        let grouped = pipe
            .generate_codes(&ids, Some(lang_id), rt + 8, 0.0, 0)
            .unwrap();
        let got_t = grouped[0].len();
        eprintln!("[fullgen] rust produced {} frames (ref {})", got_t, rt);

        let refc = read_i64(&greedy_path);
        // ref is (T, Q) row-major; compare codebook-0 column over min length.
        let n = got_t.min(rt);
        let mut col0_matches = 0;
        for f in 0..n {
            let r0 = refc[f * rq] as u32;
            if grouped[0][f] == r0 {
                col0_matches += 1;
            }
        }
        eprintln!("[fullgen] codebook-0 matches {col0_matches}/{n}");
        // also full-frame match on the first few frames
        let mut full_frame_ok = 0;
        for f in 0..n.min(8) {
            let mut ok = true;
            for g in 0..rq {
                if grouped[g][f] != refc[f * rq + g] as u32 {
                    ok = false;
                    break;
                }
            }
            if ok {
                full_frame_ok += 1;
            }
        }
        eprintln!(
            "[fullgen] full-frame (all 16 codes) matches {full_frame_ok}/{}",
            n.min(8)
        );
        // The first frames must match bit-exact (validates the loop, feedback embed, trailing pad,
        // and code_predictor). Beyond that, greedy argmax is sensitive to f32 accumulation order on
        // near-tie logits (this degenerate greedy sequence has long runs of repeated codes), so we
        // allow a small number of late tie-break flips rather than requiring a perfect prefix.
        assert_eq!(
            full_frame_ok,
            n.min(8),
            "early full-frame codes diverged (logic error)"
        );
        assert!(
            col0_matches as f32 >= 0.9 * n as f32,
            "codebook-0 diverged too early: {col0_matches}/{n}"
        );
    }
}
