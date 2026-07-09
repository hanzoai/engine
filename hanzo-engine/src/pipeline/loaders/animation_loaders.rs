use std::fmt::Debug;
use std::str::FromStr;

use anyhow::Result;
use hanzo_ml::{DType, Device};
use hanzo_quant::ShardedVarBuilder;
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;
use serde::{Deserialize, Serialize};

use crate::diffusion_models::animation::FacialAnimator;
use crate::diffusion_models::musetalk::{
    AnimatorOptions, MuseTalk, MuseTalkAnimator, MuseTalkConfig,
};
use crate::diffusion_models::wan::{
    AutoencoderKLWan, EchoMimicAnimator, EchoMimicConfig, EchoMimicDiT, EchoMimicGenerator,
    EchoMimicOptions, WanVaeConfig,
};
use crate::speech_models::whisper::{WhisperConfig, WhisperFeatureExtractor};

/// openai-whisper-tiny ships a full checkpoint; the encoder weights are nested
/// under this prefix (`encoder.conv1.weight`, ...). We scope the VarBuilder here
/// so `WhisperEncoder` keys resolve without renaming.
const WHISPER_ENCODER_PREFIX: &str = "encoder";

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum AnimationLoaderType {
    #[serde(rename = "musetalk")]
    MuseTalk,
    #[serde(rename = "echomimicv3")]
    EchoMimicV3,
}

impl FromStr for AnimationLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "musetalk" => Ok(Self::MuseTalk),
            "echomimicv3" | "echomimic-v3" | "echomimic_v3" => Ok(Self::EchoMimicV3),
            a => Err(format!(
                "Unknown animation architecture `{a}`. Possible: `musetalk`, `echomimicv3`."
            )),
        }
    }
}

/// The component weights + configs an animator is built from. Frames/PCM are request-time inputs.
/// Arch-tagged so DiT backbones (Wan family) carry their own slots instead of borrowing MuseTalk's.
pub enum AnimationComponents {
    MuseTalk(MuseTalkComponents),
    Dit(DitComponents),
}

pub struct MuseTalkComponents {
    pub musetalk_config: MuseTalkConfig,
    pub vae_vb: ShardedVarBuilder,
    pub unet_vb: ShardedVarBuilder,
    pub whisper_config: WhisperConfig,
    pub whisper_vb: ShardedVarBuilder,
    pub s3fd_vb: ShardedVarBuilder,
    pub device: Device,
    pub dtype: DType,
    pub options: AnimatorOptions,
}

/// Wan-family DiT backbone (EchoMimic, InfiniteTalk, LongCat). audio/clip/text encoder var
/// builders are optional: wav2vec2/CLIP-ViT-H/umT5 are not ported yet, so the pipeline takes
/// precomputed embeddings and these slots stay None until those encoders land.
pub struct DitComponents {
    pub dit_config: EchoMimicConfig,
    pub vae_config: WanVaeConfig,
    pub dit_vb: ShardedVarBuilder,
    pub vae_vb: ShardedVarBuilder,
    pub audio_vb: Option<ShardedVarBuilder>,
    pub clip_vb: Option<ShardedVarBuilder>,
    pub text_vb: Option<ShardedVarBuilder>,
    pub options: EchoMimicOptions,
    pub device: Device,
    pub dtype: DType,
}

/// Turns loaded component weights into a `FacialAnimator`. New backbones drop in as new
/// `AnimationLoaderType` variants + `AnimationComponents` arms with zero pipeline changes.
pub trait AnimationModelLoader: Send + Sync {
    fn load(&self, components: AnimationComponents) -> Result<Box<dyn FacialAnimator>>;
}

pub struct MuseTalkAnimationLoader;

impl AnimationModelLoader for MuseTalkAnimationLoader {
    fn load(&self, components: AnimationComponents) -> Result<Box<dyn FacialAnimator>> {
        let AnimationComponents::MuseTalk(c) = components else {
            anyhow::bail!("MuseTalk loader requires MuseTalk components");
        };
        let musetalk = MuseTalk::new(c.musetalk_config, c.vae_vb, c.unet_vb, &c.device, c.dtype)?;
        let whisper = WhisperFeatureExtractor::new(
            c.whisper_config,
            c.whisper_vb.pp(WHISPER_ENCODER_PREFIX),
            &c.device,
        )?;
        let animator = MuseTalkAnimator::new(musetalk, whisper, c.s3fd_vb, c.options)?;
        Ok(Box::new(animator))
    }
}

pub struct EchoMimicV3Loader;

impl AnimationModelLoader for EchoMimicV3Loader {
    fn load(&self, components: AnimationComponents) -> Result<Box<dyn FacialAnimator>> {
        let AnimationComponents::Dit(c) = components else {
            anyhow::bail!("EchoMimicV3 loader requires DiT components");
        };
        let vae = AutoencoderKLWan::new(&c.vae_config, c.vae_vb, &c.device)?;
        let dit = EchoMimicDiT::new(c.dit_config, c.dit_vb, c.device.clone(), c.dtype)?;
        let generator = EchoMimicGenerator::new(vae, dit, c.options, c.device, c.dtype);
        Ok(Box::new(EchoMimicAnimator::new(generator)))
    }
}

pub fn animation_loader(kind: &AnimationLoaderType) -> Box<dyn AnimationModelLoader> {
    match kind {
        AnimationLoaderType::MuseTalk => Box::new(MuseTalkAnimationLoader),
        AnimationLoaderType::EchoMimicV3 => Box::new(EchoMimicV3Loader),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::{Result as MlResult, Shape, Tensor};
    use hanzo_nn::var_builder::SimpleBackend;
    use hanzo_nn::Init;
    use hanzo_quant::ShardedSafeTensors;
    use std::sync::{Arc, Mutex};

    // Records every weight key the model requests, so we can assert the M1 scoping
    // (everything under `encoder.`) and faithful openai-whisper-tiny key naming.
    struct RecordingBackend(Arc<Mutex<Vec<String>>>);
    impl SimpleBackend for RecordingBackend {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _h: Init,
            dtype: DType,
            dev: &Device,
        ) -> MlResult<Tensor> {
            self.0.lock().unwrap().push(name.to_string());
            Tensor::zeros(s, dtype, dev)
        }
        fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> MlResult<Tensor> {
            hanzo_ml::bail!("RecordingBackend requires an explicit shape")
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    #[test]
    fn whisper_vb_scopes_under_encoder() {
        let dev = Device::Cpu;
        let keys = Arc::new(Mutex::new(Vec::new()));
        let vb = ShardedSafeTensors::wrap(
            Box::new(RecordingBackend(keys.clone())),
            DType::F32,
            dev.clone(),
        );
        // Mirror the loader: scope to the encoder before constructing the extractor.
        WhisperFeatureExtractor::new(WhisperConfig::tiny(), vb.pp(WHISPER_ENCODER_PREFIX), &dev)
            .unwrap();

        let keys = keys.lock().unwrap();
        assert!(!keys.is_empty());
        assert!(
            keys.iter().all(|k| k.starts_with("encoder.")),
            "all whisper keys must be scoped under encoder.; got {keys:?}"
        );
        for expected in [
            "encoder.conv1.weight",
            "encoder.conv2.weight",
            "encoder.positional_embedding",
            "encoder.blocks.0.attn.query.weight",
            "encoder.blocks.0.attn.key.weight",
            "encoder.blocks.0.attn_ln.weight",
            "encoder.blocks.0.mlp.0.weight",
            "encoder.blocks.0.mlp.2.weight",
            "encoder.blocks.3.attn.out.weight", // tiny has 4 encoder blocks
        ] {
            assert!(keys.iter().any(|k| k == expected), "missing key {expected}");
        }
        // openai-whisper attention key projection has no bias.
        assert!(
            !keys.iter().any(|k| k == "encoder.blocks.0.attn.key.bias"),
            "whisper attn key projection must not have a bias"
        );
        // MuseTalk conditions on the pre-ln_post embeddings, so ln_post is never loaded.
        assert!(
            !keys.iter().any(|k| k.starts_with("encoder.ln_post")),
            "MuseTalk whisper features must not use ln_post"
        );
    }

    #[test]
    fn echomimic_v3_arch_parses() {
        assert_eq!(
            "echomimicv3".parse::<AnimationLoaderType>().unwrap(),
            AnimationLoaderType::EchoMimicV3
        );
        assert_eq!(
            "echomimic-v3".parse::<AnimationLoaderType>().unwrap(),
            AnimationLoaderType::EchoMimicV3
        );
        assert!("nope".parse::<AnimationLoaderType>().is_err());
    }
}
