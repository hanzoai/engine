mod bs1770;
mod dia;
pub mod qwen3_asr;
pub mod qwen3_tts;
pub mod utils;
pub mod whisper;

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
    /// Auto-detect an ASR loader type from a config.json string. Gated on the
    /// `model_type`/`architectures` name so a Qwen3-Omni/VL config (which also
    /// nests an audio+text config) is never mis-routed here; the shape must also
    /// parse as a `Qwen3AsrConfig`. Extend this when adding new ASR pipelines.
    pub fn auto_detect_from_config(config: &str) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(config).ok()?;
        let name_signal = v.get("model_type").and_then(|m| m.as_str()) == Some("qwen3_asr")
            || v.get("architectures")
                .and_then(|a| a.as_array())
                .is_some_and(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str())
                        .any(|n| n.contains("Qwen3ASR") || n.contains("Qwen3Asr"))
                });
        if name_signal && serde_json::from_str::<Qwen3AsrConfig>(config).is_ok() {
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
    /// ACE-Step: text/tag prompt -> song (UMT5 encoder + DiT flow-match + DCAE + vocoder).
    #[serde(rename = "ace_step")]
    AceStep,
}

impl FromStr for SpeechLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dia" => Ok(Self::Dia),
            "qwen3_tts" | "qwen3-tts" | "zen3_tts" | "zen3-tts" => Ok(Self::Qwen3Tts),
            "ace_step" | "ace-step" | "acestep" => Ok(Self::AceStep),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `dia`, `qwen3_tts`, `ace_step`."
            )),
        }
    }
}

impl SpeechLoaderType {
    /// Auto-detect speech loader type from a config.json string.
    /// Extend this when adding new speech pipelines.
    pub fn auto_detect_from_config(config: &str) -> Option<Self> {
        // ace_step is checked first: the DiT config names itself `ACEStepTransformer2DModel`.
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(config) {
            if v.get("_class_name").and_then(|m| m.as_str()) == Some("ACEStepTransformer2DModel") {
                return Some(Self::AceStep);
            }
            // qwen3_tts is checked next: its config has a distinct `talker_config`/`model_type`.
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
    /// ACE-Step music: `frames` is the DiT latent time dimension (44100/4096 ~= 10.77
    /// latent frames per second of stereo audio); `steps` is the flow-match sampler
    /// count; `guidance_scale` is APG classifier-free guidance.
    AceStep {
        frames: usize,
        steps: usize,
        guidance_scale: f64,
    },
}

/// Samples of stereo audio produced by one ACE-Step latent frame (DCAE f8 over a
/// mel with hop 512 at 44.1 kHz). `frames = round(seconds * 44100 / 4096)`.
pub const ACE_STEP_SAMPLES_PER_FRAME: usize = 4096;
/// ACE-Step output sample rate (Hz).
pub const ACE_STEP_SAMPLE_RATE: usize = 44100;

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
            // ~10 s clip, ACE-Step's recommended 27 sampler steps, APG guidance 15.
            SpeechLoaderType::AceStep => Self::AceStep {
                frames: 10 * ACE_STEP_SAMPLE_RATE / ACE_STEP_SAMPLES_PER_FRAME,
                steps: 27,
                guidance_scale: 15.0,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ace_step_arch_registration() {
        for s in ["ace_step", "ace-step", "acestep"] {
            assert_eq!(
                s.parse::<SpeechLoaderType>().unwrap(),
                SpeechLoaderType::AceStep
            );
        }
        // The ACE-Step DiT/top-level config names itself; auto-detect must route to AceStep.
        let cfg = r#"{"_class_name":"ACEStepTransformer2DModel","in_channels":8}"#;
        assert_eq!(
            SpeechLoaderType::auto_detect_from_config(cfg),
            Some(SpeechLoaderType::AceStep)
        );
        let SpeechGenerationConfig::AceStep { frames, steps, .. } =
            SpeechGenerationConfig::default(SpeechLoaderType::AceStep)
        else {
            panic!("default config for AceStep must be the AceStep variant");
        };
        assert!(steps > 0);
        // ~10 s at 44100/4096 latent fps.
        assert_eq!(
            frames,
            10 * ACE_STEP_SAMPLE_RATE / ACE_STEP_SAMPLES_PER_FRAME
        );
    }
}
