#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use hanzo_ml::{DType, Device, Result};
use hanzo_quant::ShardedVarBuilder;
use tokenizers::Tokenizer;

mod config;
mod model;

pub use config::{Qwen3TtsCodecConfig, Qwen3TtsConfig};
use model::{CodecDecoder, Talker};

use super::SpeechGenerationOutput;

const SAMPLE_RATE: usize = 24000;
const CHANNELS: usize = 1;
// 12.5 Hz frame rate -> ~80ms/frame; cap synthesis length for milestone-1.
const DEFAULT_MAX_FRAMES: usize = 512;

pub struct Qwen3TtsPipeline {
    talker: Talker,
    codec: CodecDecoder,
    tokenizer: Tokenizer,
    cfg: Qwen3TtsConfig,
}

impl Qwen3TtsPipeline {
    pub fn new(
        cfg: &Qwen3TtsConfig,
        codec_cfg: &Qwen3TtsCodecConfig,
        vb: ShardedVarBuilder,
        codec_vb: ShardedVarBuilder,
        tokenizer: Tokenizer,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let talker = Talker::new(cfg, dtype, device, vb)?;
        let codec = CodecDecoder::new(codec_cfg, DType::F32, device, codec_vb)?;
        Ok(Self {
            talker,
            codec,
            tokenizer,
            cfg: cfg.clone(),
        })
    }

    // Encode the text prompt with the Qwen3 chat template framing the TTS task. Milestone-1 uses a
    // minimal framing: <|im_start|>user\n<text><|im_end|>\n<|im_start|>assistant\n<tts_bos>.
    fn encode_prompt(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(hanzo_ml::Error::msg)?;
        let mut ids = Vec::new();
        ids.push(self.cfg.im_start_token_id);
        ids.extend(enc.get_ids().iter().copied());
        ids.push(self.cfg.im_end_token_id);
        ids.push(self.cfg.tts_bos_token_id);
        Ok(ids)
    }

    pub fn generate(
        &self,
        text: &str,
        max_frames: Option<usize>,
    ) -> Result<SpeechGenerationOutput> {
        let ids = self.encode_prompt(text)?;
        let max_frames = max_frames.unwrap_or(DEFAULT_MAX_FRAMES);
        let codes = self.talker.generate_codes(&ids, max_frames)?;
        if std::env::var("ZENTTS_DEBUG").is_ok() {
            let (nc, nf) = (codes.dim(0)?, codes.dim(1)?);
            eprintln!("[zentts] prompt_ids={ids:?}");
            eprintln!("[zentts] codes shape = [{nc} codebooks, {nf} frames]");
            if nf > 0 {
                let v: Vec<Vec<u32>> = codes.to_vec2()?;
                for (ci, row) in v.iter().enumerate().take(3) {
                    let head: Vec<u32> = row.iter().take(12).copied().collect();
                    let uniq: std::collections::HashSet<u32> = row.iter().copied().collect();
                    eprintln!("[zentts]   cb{ci}: uniq={} head={head:?}", uniq.len());
                }
            }
        }
        let pcm = if codes.dim(1)? == 0 {
            Vec::new()
        } else {
            self.codec.decode(&codes)?
        };
        Ok(SpeechGenerationOutput {
            pcm: Arc::new(pcm),
            rate: SAMPLE_RATE,
            channels: CHANNELS,
        })
    }

    pub fn device(&self) -> &Device {
        self.talker.device()
    }
}
