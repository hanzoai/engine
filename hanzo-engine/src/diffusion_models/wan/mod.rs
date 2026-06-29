//! Wan-family video backbones. The shared 3D causal VAE (`vae.rs`) and conv3d decomposition are
//! owned by the EchoMimic port; LongCat reuses them via the `longcat::WanVae` trait.

pub mod longcat;
