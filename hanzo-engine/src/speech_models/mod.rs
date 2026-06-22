mod bs1770;
mod dia;
pub mod qwen3_asr;
pub mod qwen3_tts;
pub mod utils;

use std::{str::FromStr, sync::Arc};

pub use dia::{DiaConfig, DiaPipeline};
pub use qwen3_asr::{Qwen3AsrConfig, Qwen3AsrModel, Qwen3AsrPipeline};
pub use qwen3_tts::{CodecConfig as Qwen3TtsCodecConfig, Qwen3TtsConfig, Qwen3TtsPipeline};
use serde::{Deserialize, Serialize};

/// Audio-understanding (speech -> text) model families. Distinct from
/// [`SpeechLoaderType`], which covers speech *generation* (text -> PCM): ASR
/// emits token logits through the LM, not a waveform.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub enum AsrLoaderType {
    #[serde(rename = "qwen3_asr")]
    Qwen3Asr,
}

impl FromStr for AsrLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "qwen3_asr" | "qwen3-asr" => Ok(Self::Qwen3Asr),
            a => Err(format!(
                "Unknown ASR architecture `{a}`. Possible architectures: `qwen3_asr`."
            )),
        }
    }
}

impl AsrLoaderType {
    /// Auto-detect an ASR loader type from a config.json string.
    /// Extend this when adding new ASR pipelines.
    pub fn auto_detect_from_config(config: &str) -> Option<Self> {
        if serde_json::from_str::<Qwen3AsrConfig>(config).is_ok() {
            return Some(Self::Qwen3Asr);
        }
        None
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub enum SpeechLoaderType {
    #[serde(rename = "dia")]
    Dia,
    #[serde(rename = "qwen3_tts")]
    Qwen3Tts,
}

impl FromStr for SpeechLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dia" => Ok(Self::Dia),
            "qwen3_tts" | "qwen3-tts" | "zen3_tts" | "zen3-tts" => Ok(Self::Qwen3Tts),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `dia`, `qwen3_tts`."
            )),
        }
    }
}

impl SpeechLoaderType {
    /// Auto-detect speech loader type from a config.json string.
    /// Extend this when adding new speech pipelines.
    pub fn auto_detect_from_config(config: &str) -> Option<Self> {
        // qwen3_tts is checked first: its config has a distinct `talker_config`/`model_type`.
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(config) {
            if v.get("model_type").and_then(|m| m.as_str()) == Some("qwen3_tts") {
                return Some(Self::Qwen3Tts);
            }
        }
        if serde_json::from_str::<Qwen3TtsConfig>(config).is_ok() {
            return Some(Self::Qwen3Tts);
        }
        if serde_json::from_str::<DiaConfig>(config).is_ok() {
            return Some(Self::Dia);
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SpeechGenerationConfig {
    Dia {
        max_tokens: Option<usize>,
        cfg_scale: f32,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
    },
    Qwen3Tts {
        max_tokens: Option<usize>,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
    },
}

impl SpeechGenerationConfig {
    pub fn default(ty: SpeechLoaderType) -> Self {
        match ty {
            SpeechLoaderType::Dia => Self::Dia {
                max_tokens: None,
                cfg_scale: 3.,
                temperature: 1.3,
                top_p: 0.95,
                top_k: Some(35),
            },
            SpeechLoaderType::Qwen3Tts => Self::Qwen3Tts {
                max_tokens: Some(2048),
                temperature: 0.9,
                top_p: 1.0,
                top_k: Some(50),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeechGenerationOutput {
    pub pcm: Arc<Vec<f32>>,
    pub rate: usize,
    pub channels: usize,
}
