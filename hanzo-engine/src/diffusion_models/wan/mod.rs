//! Wan2.1 family video backbones. The shared 3D causal VAE (`vae.rs`), conv3d decomposition
//! (`conv3d.rs`), and DiT primitives (`dit_common.rs`) are reused by every backbone: EchoMimic v3
//! (`echomimic*`), LongCat avatar (`longcat`), and InfiniteTalk (`infinitetalk`).

pub mod conv3d;
pub mod echomimic;
pub mod echomimic_dit;
pub mod infinitetalk;
pub mod longcat;
pub mod vae;

pub use conv3d::{CausalConv3d, FeatCache};
pub use echomimic::{
    EchoMimicAnimator, EchoMimicGenerator, EchoMimicOptions, FlowMatchScheduler, Wav2Vec2Encoder,
};
pub use echomimic_dit::{EchoMimicConfig, EchoMimicDiT};
pub use infinitetalk::{InfiniteTalkConfig, InfiniteTalkGenerator};
pub use longcat::{
    LongCatAvatarAnimator, LongCatAvatarDiT, LongCatConfig, LongCatGenerator, LongCatOptions,
    WhisperLargeEncoder,
};
pub use vae::{AutoencoderKLWan, WanVaeConfig};
