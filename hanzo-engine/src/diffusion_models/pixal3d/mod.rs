//! Pixal3D: image-to-3D in the TRELLIS family (MIT, microsoft/TRELLIS-image-large).
//!
//! Coarse-geometry path implemented end to end: DINOv2 conditioner -> sparse-structure rectified-flow
//! sample -> 3D-conv decode to a 64^3 occupancy grid -> surface mesh -> GLB. The sparse SLAT flow +
//! FlexiCubes mesh refinement is the remaining stage.

pub mod dinov2;
pub mod flexicubes;
pub mod flexicubes_tables;
pub mod glb;
pub mod mesh;
pub mod pipeline;
pub mod preprocess;
pub mod sampler;
pub mod slat_decoder;
pub mod slat_flow;
pub mod sparse;
pub mod ss_decoder;
pub mod ss_flow;
pub mod transformer;

use anyhow::Result;
use hanzo_3d::Mesh;
use image::DynamicImage;

/// Run Pixal3D image-to-3D, returning the coarse occupancy mesh. `seed` seeds the sampler, `steps` is
/// the number of rectified-flow steps. Weights are resolved from `PIXAL3D_MODEL`.
pub fn pixal3d_generate(image: &DynamicImage, seed: u64, steps: usize) -> Result<Mesh> {
    pipeline::generate(image, seed, steps)
}

/// Run the full Pixal3D pipeline (coarse occupancy -> SLAT flow -> mesh decoder -> FlexiCubes),
/// returning the fine textured mesh (vertices + faces + per-vertex RGB).
pub fn pixal3d_generate_textured(
    image: &DynamicImage,
    seed: u64,
    steps: usize,
) -> Result<flexicubes::TexturedMesh> {
    pipeline::generate_textured(image, seed, steps)
}
