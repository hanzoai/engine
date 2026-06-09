#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
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
pub use model::{CodePredictor, Qwen3TtsModel, Talker};
use model::causal_mask;

use super::{SpeechGenerationConfig, SpeechGenerationOutput};

const CHANNELS: usize = 1;

/// Codec frames per emitted streaming chunk. The codec runs at 12.5 Hz so 8 frames ~= 0.64 s of
/// audio; large enough that the conv-upsampler overhead amortizes, small enough to feel realtime.
const STREAM_CHUNK_FRAMES: usize = 8;

/// Resolved per-request sampling controls for one logits row.
#[derive(Clone, Copy)]
struct SamplingParams {
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repetition_penalty: f32,
}

pub struct Qwen3TtsPipeline {
    model: Qwen3TtsModel,
    cfg: Qwen3TtsConfig,
    device: Device,
    dtype: DType,
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

    /// Samples one token id from a 1-D logits row with temp/top-k/top-p/repetition-penalty.
    /// `counts` holds how many times each id was already emitted in the current stream and drives
    /// the repetition penalty (Hugging Face CTRL form: divide positives, multiply negatives).
    fn sample(
        &self,
        logits: &Tensor,
        sp: &SamplingParams,
        counts: &HashMap<u32, usize>,
        rng: &mut Isaac64Rng,
    ) -> Result<u32> {
        let mut row: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;

        if sp.repetition_penalty != 1.0 {
            for (&id, &c) in counts {
                if c == 0 {
                    continue;
                }
                let i = id as usize;
                if i >= row.len() {
                    continue;
                }
                row[i] = if row[i] > 0.0 {
                    row[i] / sp.repetition_penalty
                } else {
                    row[i] * sp.repetition_penalty
                };
            }
        }

        if sp.temperature <= 0.0 {
            let mut best = 0usize;
            for i in 1..row.len() {
                if row[i] > row[best] {
                    best = i;
                }
            }
            return Ok(best as u32);
        }

        let logits = Tensor::from_vec(row, logits.dims().to_vec(), &self.device)?;
        let probs =
            hanzo_nn::ops::softmax_last_dim(&(logits / sp.temperature as f64)?)?;
        let mut probs: Vec<f32> = probs.to_vec1()?;

        let mut argsort: Vec<usize> = (0..probs.len()).collect();
        argsort.sort_unstable_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

        if let Some(k) = sp.top_k {
            for (rank, &idx) in argsort.iter().enumerate() {
                if rank >= k {
                    probs[idx] = 0.0;
                }
            }
        }

        if sp.top_p < 1.0 {
            let mut cumsum = 0.0;
            for &idx in &argsort {
                if cumsum >= sp.top_p {
                    probs[idx] = 0.0;
                } else {
                    cumsum += probs[idx];
                }
            }
        }

        let distr = WeightedIndex::new(&probs).map_err(hanzo_ml::Error::msg)?;
        Ok(distr.sample(rng) as u32)
    }

    /// Builds the prefill embedding sequence from the text token ids.
    /// This is a structural placeholder: the full prefill interleaves role tokens, tts pad/bos, a
    /// speaker embedding, and per-frame codec pad embeddings. Speaker conditioning is not wired:
    /// `SpeakerEncoderConfig` exists but no SpeakerEncoder module/weights are loaded and there is no
    /// reference-audio input path, so a voice embedding cannot be produced or injected here yet.
    fn prefill_embeds(&self, text_ids: &Tensor) -> Result<Tensor> {
        let text_embeds = self.model.talker.embed_text(text_ids)?;
        let bos = Tensor::new(&[self.cfg.talker_config.codec_bos_id], &self.device)?;
        let bos_embed = self.model.talker.embed_codec(&bos)?.unsqueeze(0)?;
        Tensor::cat(&[text_embeds, bos_embed], 1)
    }

    /// Runs one MTP pass to predict codebooks 1..num_code_groups from the talker hidden state
    /// for the current frame, given the already-sampled codebook-0 id. Uses a fresh per-frame
    /// KV-cache for the small MTP transformer.
    fn predict_codes(
        &self,
        talker_hidden: &Tensor,
        code0: u32,
        sp: &SamplingParams,
        rng: &mut Isaac64Rng,
    ) -> Result<Vec<u32>> {
        let num_groups = self.cfg.talker_config.num_code_groups;
        let mut codes = Vec::with_capacity(num_groups);
        codes.push(code0);

        let mut cache = self.model.code_predictor.make_cache();
        let mut cur = self.model.code_predictor.project(talker_hidden)?;
        let mut prev = Tensor::new(&[code0], &self.device)?;
        let no_counts = HashMap::new();
        for g in 0..num_groups.saturating_sub(1) {
            let prev_embed = self.model.code_predictor.embed_codec(&prev)?.unsqueeze(0)?;
            let inp = (&cur + &prev_embed)?;
            let offset = cache[0].current_seq_len();
            let logits = self
                .model
                .code_predictor
                .step(&inp, &[offset], None, g, &mut cache)?;
            let last = logits.i((0, logits.dim(1)? - 1, ..))?;
            let id = self.sample(&last, sp, &no_counts, rng)?;
            codes.push(id);
            prev = Tensor::new(&[id], &self.device)?;
            cur = inp;
        }
        Ok(codes)
    }

    /// Decodes a window of frames (each `num_groups` codes) into PCM samples via the codec.
    fn decode_frames(&self, frames: &[Vec<u32>], num_groups: usize) -> Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let t = frames.len();
        let mut flat: Vec<u32> = Vec::with_capacity(t * num_groups);
        for g in 0..num_groups {
            for frame in frames {
                flat.push(*frame.get(g).unwrap_or(&0));
            }
        }
        let codes = Tensor::from_vec(flat, (1, num_groups, t), &self.device)?;
        let pcm = self.model.codec.decode(&codes)?;
        pcm.i((0, 0))?.to_dtype(DType::F32)?.to_vec1::<f32>()
    }

    pub fn generate(
        &self,
        text: &str,
        gen_cfg: &SpeechGenerationConfig,
    ) -> Result<SpeechGenerationOutput> {
        self.generate_streaming(text, gen_cfg, |_| {})
    }

    /// Autoregressive generation with a KV-cached talker and per-frame KV-cached MTP head.
    /// `on_chunk` is invoked with freshly decoded PCM every `STREAM_CHUNK_FRAMES` frames so callers
    /// can stream audio out as it is produced; the full PCM is also returned in the output.
    pub fn generate_streaming(
        &self,
        text: &str,
        gen_cfg: &SpeechGenerationConfig,
        mut on_chunk: impl FnMut(&[f32]),
    ) -> Result<SpeechGenerationOutput> {
        let SpeechGenerationConfig::Qwen3Tts {
            max_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        } = gen_cfg
        else {
            hanzo_ml::bail!("Qwen3TtsPipeline requires a SpeechGenerationConfig::Qwen3Tts");
        };

        let max_tokens = max_tokens.unwrap_or(2048);
        let sp = SamplingParams {
            temperature: *temperature,
            top_p: *top_p,
            top_k: *top_k,
            repetition_penalty: *repetition_penalty,
        };

        // Tokenization of `text` into talker text-vocab ids is handled by the
        // speech processor/tokenizer upstream; here we use byte ids as a stand-in
        // so the forward path is exercised end-to-end and compiles.
        let bytes: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        let text_ids = Tensor::new(bytes.as_slice(), &self.device)?.unsqueeze(0)?;

        let eos = self.cfg.talker_config.codec_eos_token_id;
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let num_groups = self.cfg.talker_config.num_code_groups;

        let mut cache = self.model.talker.make_cache();
        let mut counts: HashMap<u32, usize> = HashMap::new();

        // Prefill: run the whole prompt once through the cached backbone with a causal mask.
        let prefill = self.prefill_embeds(&text_ids)?;
        let prefill_len = prefill.dim(1)?;
        let mask = causal_mask(prefill_len, self.dtype, &self.device)?;
        let (mut hidden, mut logits) =
            self.model
                .talker
                .forward(&prefill, &[0], Some(&mask), &mut cache)?;

        let mut all_codes: Vec<Vec<u32>> = Vec::new();
        let mut emitted_frames = 0usize;
        for _ in 0..max_tokens {
            let last_hidden = hidden.i((.., hidden.dim(1)? - 1.., ..))?;
            let last_logits = logits.i((0, logits.dim(1)? - 1, ..))?;
            let code0 = self.sample(&last_logits, &sp, &counts, &mut rng)?;
            if code0 == eos {
                break;
            }
            *counts.entry(code0).or_insert(0) += 1;

            let frame = self.predict_codes(&last_hidden, code0, &sp, &mut rng)?;
            all_codes.push(frame);

            if all_codes.len() - emitted_frames >= STREAM_CHUNK_FRAMES {
                let chunk = self.decode_frames(&all_codes[emitted_frames..], num_groups)?;
                if !chunk.is_empty() {
                    on_chunk(&chunk);
                }
                emitted_frames = all_codes.len();
            }

            // Decode step: feed only the just-emitted codebook-0 embedding back in.
            let code0_t = Tensor::new(&[code0], &self.device)?;
            let next_embed = self.model.talker.embed_codec(&code0_t)?.unsqueeze(0)?;
            let offset = cache[0].current_seq_len();
            let (h, l) = self
                .model
                .talker
                .forward(&next_embed, &[offset], None, &mut cache)?;
            hidden = h;
            logits = l;
        }

        if emitted_frames < all_codes.len() {
            let chunk = self.decode_frames(&all_codes[emitted_frames..], num_groups)?;
            if !chunk.is_empty() {
                on_chunk(&chunk);
            }
        }

        let pcm = self.decode_frames(&all_codes, num_groups)?;

        Ok(SpeechGenerationOutput {
            pcm: Arc::new(pcm),
            rate: self.model.codec.sample_rate(),
            channels: CHANNELS,
        })
    }
}

#[cfg(test)]
mod tests;
