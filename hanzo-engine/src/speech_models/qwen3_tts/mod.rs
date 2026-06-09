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
pub use model::{CodePredictor, Qwen3TtsModel, Talker};
use model::causal_mask;

use super::{SpeechGenerationConfig, SpeechGenerationOutput};

const CHANNELS: usize = 1;

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

    /// Greedily/temperature-samples a single token id from a 1-D logits row.
    fn sample(&self, logits: &Tensor, temperature: f32, rng: &mut Isaac64Rng) -> Result<u32> {
        if temperature == 0. {
            return Ok(logits.argmax(D::Minus1)?.to_scalar::<u32>()?);
        }
        let probs = hanzo_nn::ops::softmax_last_dim(
            &(logits.to_dtype(DType::F32)? / temperature as f64)?,
        )?;
        let probs: Vec<f32> = probs.to_vec1()?;
        let distr = WeightedIndex::new(&probs).map_err(hanzo_ml::Error::msg)?;
        Ok(distr.sample(rng) as u32)
    }

    /// Builds the prefill embedding sequence from the text token ids.
    /// This is a structural placeholder: the full prefill interleaves role tokens,
    /// tts pad/bos, speaker embedding, and per-frame codec pad embeddings.
    fn prefill_embeds(&self, text_ids: &Tensor) -> Result<Tensor> {
        let text_embeds = self.model.talker.embed_text(text_ids)?;
        let bos = Tensor::new(&[self.cfg.talker_config.codec_bos_id], &self.device)?;
        let bos_embed = self.model.talker.embed_codec(&bos)?.unsqueeze(0)?;
        Tensor::cat(&[text_embeds, bos_embed], 1)
    }

    /// Runs one MTP pass to predict codebooks 1..num_code_groups from the talker hidden state
    /// for the current frame, given the already-sampled codebook-0 id.
    fn predict_codes(
        &self,
        talker_hidden: &Tensor,
        code0: u32,
        temperature: f32,
        rng: &mut Isaac64Rng,
    ) -> Result<Vec<u32>> {
        let num_groups = self.cfg.talker_config.num_code_groups;
        let mut codes = Vec::with_capacity(num_groups);
        codes.push(code0);

        let mut cur = self.model.code_predictor.project(talker_hidden)?;
        let mut prev = Tensor::new(&[code0], &self.device)?;
        for g in 0..num_groups.saturating_sub(1) {
            let prev_embed = self.model.code_predictor.embed_codec(&prev)?.unsqueeze(0)?;
            let inp = (&cur + &prev_embed)?;
            let mask = causal_mask(inp.dim(1)?, self.dtype, &self.device)?;
            let logits = self
                .model
                .code_predictor
                .step(&inp, &[0], Some(&mask), g)?;
            let last = logits.i((0, logits.dim(1)? - 1, ..))?;
            let id = self.sample(&last, temperature, rng)?;
            codes.push(id);
            prev = Tensor::new(&[id], &self.device)?;
            cur = inp;
        }
        Ok(codes)
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
        let temperature = *temperature;

        // Tokenization of `text` into talker text-vocab ids is handled by the
        // speech processor/tokenizer upstream; here we use byte ids as a stand-in
        // so the forward path is exercised end-to-end and compiles.
        let bytes: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        let text_ids = Tensor::new(bytes.as_slice(), &self.device)?.unsqueeze(0)?;

        let mut inputs_embeds = self.prefill_embeds(&text_ids)?;

        let eos = self.cfg.talker_config.codec_eos_token_id;
        let mut rng = Isaac64Rng::seed_from_u64(0);
        let num_groups = self.cfg.talker_config.num_code_groups;

        let mut all_codes: Vec<Vec<u32>> = Vec::new();
        for _ in 0..max_tokens {
            let mask = causal_mask(inputs_embeds.dim(1)?, self.dtype, &self.device)?;
            let (hidden, logits) = self.model.talker.forward(&inputs_embeds, &[0], Some(&mask))?;

            let last_hidden = hidden.i((.., hidden.dim(1)? - 1.., ..))?;
            let last_logits = logits.i((0, logits.dim(1)? - 1, ..))?;
            let code0 = self.sample(&last_logits, temperature, &mut rng)?;
            if code0 == eos {
                break;
            }

            let frame = self.predict_codes(&last_hidden, code0, temperature, &mut rng)?;
            all_codes.push(frame);

            // Feed the codebook-0 embedding of the just-emitted frame back in.
            let code0_t = Tensor::new(&[code0], &self.device)?;
            let next_embed = self.model.talker.embed_codec(&code0_t)?.unsqueeze(0)?;
            inputs_embeds = Tensor::cat(&[inputs_embeds, next_embed], 1)?;
        }

        let pcm = if all_codes.is_empty() {
            Vec::new()
        } else {
            let t = all_codes.len();
            let mut flat: Vec<u32> = Vec::with_capacity(t * num_groups);
            for g in 0..num_groups {
                for frame in &all_codes {
                    flat.push(*frame.get(g).unwrap_or(&0));
                }
            }
            let codes = Tensor::from_vec(flat, (1, num_groups, t), &self.device)?;
            let pcm = self.model.codec.decode(&codes)?;
            pcm.i((0, 0))?.to_dtype(DType::F32)?.to_vec1::<f32>()?
        };

        Ok(SpeechGenerationOutput {
            pcm: Arc::new(pcm),
            rate: self.model.codec.sample_rate(),
            channels: CHANNELS,
        })
    }
}
