//! Pixal3D: image-to-3D in the TRELLIS family (structured-latent SLAT -> rectified-flow decode).
//!
//! The forward pass is not yet wired: it reuses the Flux / Qwen-Image diffusion backbone for the
//! SLAT sampler and decodes into a [`hanzo_3d::Mesh`]. Only the entry point exists here; the
//! sampler, the rectified-flow transformer, and the mesh decoder are the follow-up.

pub mod dinov2;
pub mod glb;
pub mod ss_flow;
pub mod transformer;

use anyhow::{bail, Result};
use hanzo_3d::Mesh;
use image::DynamicImage;

/// Run Pixal3D image-to-3D, returning the decoded mesh.
///
/// `image` is the conditioning image, `seed` the sampler seed, `steps` the number of
/// rectified-flow steps. The SLAT sampler + RFT decode are not yet wired.
pub fn pixal3d_generate(image: &DynamicImage, seed: u64, steps: usize) -> Result<Mesh> {
    let _ = (image, seed, steps);
    bail!("Pixal3D forward pending model wiring")
}
