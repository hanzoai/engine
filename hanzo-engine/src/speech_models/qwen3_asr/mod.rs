#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
// Some encoder/decoder helper methods are only exercised by the isolation tests
// (decoder-vs-ref, ref-embeds), so keep a crate-local dead_code allow.
#![allow(dead_code)]

//! Qwen3-ASR (audio -> text). ASR modality for the engine, served via the ASR
//! pipeline (`pipeline::asr`) at `/v1/audio/transcriptions`.
//!
//! Architecture (per the HF `Qwen3-ASR` release): an AuT audio encoder
//! (Conv2d 8x-downsample stem + sinusoidal-position bidirectional transformer +
//! 2-layer output projection) feeds a standard Qwen3 text decoder. Audio
//! features are merged into the decoder by *replacing* the embeddings of
//! `<|audio_pad|>` placeholder tokens (id 151676) with the projected encoder
//! outputs, mirroring how the vision models splice image embeddings into the
//! LM input sequence.
//!
//! Weight layout (safetensors):
//!   `thinker.audio_tower.*`  -> [`encoder::Qwen3AsrAudioEncoder`]
//!   `thinker.model.*`        -> [`decoder::Qwen3AsrTextDecoder`]
//!   `thinker.lm_head.weight` -> tied with `thinker.model.embed_tokens.weight`
//!
//! End-to-end path (mel frontend -> AuT encoder -> audio-embed splice -> greedy
//! decode) lives in [`pipeline::Qwen3AsrPipeline`] and is covered by the
//! `qwen3_asr_e2e` test. Streaming incremental (KV-cache) decode is the one
//! remaining perf item; decode currently re-runs prefill per step.

mod audio;
pub mod config;
mod decoder;
mod encoder;
mod pipeline;

use hanzo_ml::{Result, Tensor};
use hanzo_quant::ShardedVarBuilder;

pub use audio::Qwen3AsrAudioProcessor;
pub use config::{AudioEncoderConfig, Qwen3AsrConfig, TextDecoderConfig};
use decoder::Qwen3AsrTextDecoder;
use encoder::Qwen3AsrAudioEncoder;
pub use pipeline::Qwen3AsrPipeline;

const AUDIO_TOWER_PREFIX: &str = "thinker.audio_tower";
const TEXT_MODEL_PREFIX: &str = "thinker.model";

pub struct Qwen3AsrModel {
    encoder: Qwen3AsrAudioEncoder,
    decoder: Qwen3AsrTextDecoder,
    audio_token_id: u32,
}

impl Qwen3AsrModel {
    pub fn new(cfg: &Qwen3AsrConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let encoder = Qwen3AsrAudioEncoder::new(&cfg.audio_config, vb.pp(AUDIO_TOWER_PREFIX))?;
        let decoder = Qwen3AsrTextDecoder::new(&cfg.text_config, vb.pp(TEXT_MODEL_PREFIX))?;
        Ok(Self {
            encoder,
            decoder,
            audio_token_id: cfg.audio_token_id as u32,
        })
    }

    /// Encode log-mel features `[B, n_mels, T]` to LM-space audio embeddings
    /// `[B, T/8, hidden_size]`.
    pub fn encode_audio(&self, mel: &Tensor) -> Result<Tensor> {
        self.encoder.forward(mel)
    }

    /// Embed `input_ids` and replace `<|audio_pad|>` placeholder positions with
    /// `audio_embeds`. Audio occupies a single contiguous span per batch row.
    fn embed_and_merge(&self, input_ids: &Tensor, audio_embeds: Option<&Tensor>) -> Result<Tensor> {
        let mut input_embeds = self.decoder.embed_tokens(input_ids)?;
        let Some(audio_embeds) = audio_embeds else {
            return Ok(input_embeds);
        };
        let audio_embeds = audio_embeds.to_dtype(input_embeds.dtype())?;
        let hidden = self.decoder.hidden_size();

        let ids: Vec<Vec<u32>> = input_ids.to_dtype(hanzo_ml::DType::U32)?.to_vec2::<u32>()?;
        for (batch, row) in ids.iter().enumerate() {
            let Some(start) = row.iter().position(|&t| t == self.audio_token_id) else {
                continue;
            };
            let len = row[start..]
                .iter()
                .take_while(|&&t| t == self.audio_token_id)
                .count();
            let avail = audio_embeds.dim(1)?;
            let len = len.min(avail);
            let chunk = audio_embeds.narrow(0, batch.min(audio_embeds.dim(0)? - 1), 1)?;
            let chunk = chunk.narrow(1, 0, len)?;
            input_embeds = input_embeds
                .slice_assign(&[batch..batch + 1, start..start + len, 0..hidden], &chunk)?;
        }
        Ok(input_embeds)
    }

    /// Prefill forward: merge audio into the prompt and run the decoder.
    /// Returns logits `[B, seq_len, vocab]`. `positions` is `[seq_len]` u32.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        mel_features: Option<&Tensor>,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let audio_embeds = mel_features.map(|m| self.encode_audio(m)).transpose()?;
        self.forward_with_audio(input_ids, audio_embeds.as_ref(), positions)
    }

    /// Like [`Self::forward`] but takes already-encoded audio embeddings so the
    /// AuT encoder runs once per clip instead of once per decode step.
    pub fn forward_with_audio(
        &self,
        input_ids: &Tensor,
        audio_embeds: Option<&Tensor>,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let input_embeds = self.embed_and_merge(input_ids, audio_embeds)?;
        self.decoder.forward_embeds(&input_embeds, positions)
    }
}
