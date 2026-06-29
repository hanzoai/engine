//! Wan-family video backbones. The shared 3D causal VAE (`vae.rs`) and conv3d decomposition
//! (`conv3d.rs`) are owned by the EchoMimic port and reused by every backbone (EchoMimic v3,
//! LongCat avatar).

pub mod conv3d;
pub mod echomimic;
pub mod echomimic_dit;
pub mod longcat;
pub mod vae;

pub use conv3d::{CausalConv3d, FeatCache};
pub use echomimic::{
    EchoMimicAnimator, EchoMimicGenerator, EchoMimicOptions, FlowMatchScheduler, Wav2Vec2Encoder,
};
pub use echomimic_dit::{EchoMimicConfig, EchoMimicDiT};
pub use longcat::{
    LongCatAvatarAnimator, LongCatAvatarDiT, LongCatConfig, LongCatGenerator, LongCatOptions,
    WhisperLargeEncoder,
};
pub use vae::{AutoencoderKLWan, WanVaeConfig};
