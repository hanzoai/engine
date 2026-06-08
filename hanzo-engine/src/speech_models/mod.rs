mod bs1770;
mod dia;
mod qwen3_tts;
pub mod utils;

use std::{str::FromStr, sync::Arc};

pub use dia::{DiaConfig, DiaPipeline};
pub use qwen3_tts::{Qwen3TtsCodecConfig, Qwen3TtsConfig, Qwen3TtsPipeline};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub enum SpeechLoaderType {
    #[serde(rename = "dia")]
    Dia,
    #[serde(rename = "qwen3-tts")]
    Qwen3Tts,
}

impl FromStr for SpeechLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dia" => Ok(Self::Dia),
            "qwen3-tts" | "qwen3_tts" => Ok(Self::Qwen3Tts),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `dia`, `qwen3-tts`."
            )),
        }
    }
}

impl SpeechLoaderType {
    /// Auto-detect speech loader type from a config.json string.
    /// Extend this when adding new speech pipelines.
    pub fn auto_detect_from_config(config: &str) -> Option<Self> {
        if let Ok(cfg) = serde_json::from_str::<Qwen3TtsConfig>(config) {
            if cfg.is_qwen3_tts() {
                return Some(Self::Qwen3Tts);
            }
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
        max_frames: Option<usize>,
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
            SpeechLoaderType::Qwen3Tts => Self::Qwen3Tts { max_frames: None },
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeechGenerationOutput {
    pub pcm: Arc<Vec<f32>>,
    pub rate: usize,
    pub channels: usize,
}
