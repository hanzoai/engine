//! Wan2.1 family: the shared `AutoencoderKLWan` VAE + Wan DiT backbone, plus the per-product audio /
//! streaming overlays (EchoMimic, InfiniteTalk, LongCat). The VAE and DiT are shared components owned
//! here; each overlay adds only its novel value. `pub mod vae;` / the Wan DiT module land with the
//! EchoMimic port (Wan DiT @ 1.3B), which InfiniteTalk reuses at the 14B config.

pub mod infinitetalk;
