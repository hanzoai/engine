//! Pixal3D image-to-3D pipeline (coarse-geometry path).
//!
//! image -> DINOv2 tokens -> sparse-structure rectified-flow sample -> 3D-conv decode to a 64^3
//! occupancy grid -> boundary surface mesh. This is the dense half of TRELLIS end to end; the sparse
//! SLAT flow + FlexiCubes mesh refinement is the remaining stage.
//!
//! Weights live under `PIXAL3D_MODEL` (the microsoft/TRELLIS-image-large snapshot, plus a converted
//! `dinov2_vitl14_reg.safetensors`).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, bail, Result};
use hanzo_3d::Mesh;
use hanzo_ml::{DType, Device, Tensor};
use hanzo_quant::ShardedVarBuilder;
use image::DynamicImage;

use super::dinov2::{DinoV2, DinoV2Config};
use super::slat_decoder::{SlatDecoderConfig, SlatMeshDecoder};
use super::slat_flow::{SlatFlow, SlatFlowConfig};
use super::sparse::Sparse;
use super::ss_decoder::{SsDecoder, SsDecoderConfig};
use super::ss_flow::{SsFlow, SsFlowConfig};
use super::{flexicubes, mesh, preprocess, sampler};
use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};

const DINOV2_FILE: &str = "dinov2_vitl14_reg.safetensors";
const SS_FLOW_FILE: &str = "ss_flow_img_dit_L_16l8_fp16.safetensors";
const SS_DEC_FILE: &str = "ss_dec_conv3d_16l8_fp16.safetensors";
const SLAT_FLOW_FILE: &str = "slat_flow_img_dit_L_64l8p2_fp16.safetensors";
const SLAT_DEC_FILE: &str = "slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors";
const SS_RES: usize = 16;
const SS_CH: usize = 8;
const OCC_RES: usize = 64;
const SLAT_CH: usize = 8;
const MESH_RES: i32 = 256; // FlexiCubes grid = decoder resolution (64) * 4

// slat_flow_img_dit_L_64l8p2 dataset normalization (denormalize the sampled latent before decoding).
const SLAT_MEAN: [f32; 8] = [
    -2.168_754_6,
    -0.004_347_046,
    -0.133_523_5,
    -0.084_180_73,
    -0.527_120_65,
    0.723_868_9,
    -1.141_445,
    1.203_936_3,
];
const SLAT_STD: [f32; 8] = [
    2.377_650_7,
    2.386_378_3,
    2.124_418,
    2.174_855_2,
    2.663_944_7,
    2.371_192_2,
    2.621_744_6,
    2.684_523,
];

struct Models {
    dinov2: DinoV2,
    ss_flow: SsFlow,
    ss_dec: SsDecoder,
    slat_flow: SlatFlow,
    slat_dec: SlatMeshDecoder,
}

fn load_vb(path: PathBuf, dev: &Device) -> Result<ShardedVarBuilder> {
    Ok(from_mmaped_safetensors(
        vec![path],
        Vec::new(),
        Some(DType::F32),
        dev,
        vec![None],
        true,
        None,
        |_| true,
        Arc::new(|_| DeviceForLoadTensor::Base),
    )?)
}

/// Deterministic standard-normal noise (splitmix64 + Box-Muller). candle's CPU RNG isn't seedable
/// via `Device::set_seed`; the sample only needs to be reproducible, not to match torch's RNG.
fn seeded_gaussian(shape: &[usize], seed: u64, dev: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next_unit = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (((z >> 40) as f32) / ((1u32 << 24) as f32)).max(1e-7) // (0, 1]
    };
    let mut data = Vec::with_capacity(n);
    while data.len() < n {
        let (u1, u2) = (next_unit(), next_unit());
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        data.push(r * theta.cos());
        if data.len() < n {
            data.push(r * theta.sin());
        }
    }
    Ok(Tensor::from_vec(data, shape.to_vec(), dev)?)
}

/// A weight file may sit directly in the model dir or under `ckpts/` (the HF snapshot layout).
fn resolve(dir: &Path, name: &str) -> Result<PathBuf> {
    for cand in [dir.join(name), dir.join("ckpts").join(name)] {
        if cand.exists() {
            return Ok(cand);
        }
    }
    bail!("missing weight {name} under {}", dir.display())
}

impl Models {
    fn load(dir: &Path, dev: &Device) -> Result<Self> {
        Ok(Self {
            dinov2: DinoV2::new(
                &DinoV2Config::default(),
                load_vb(resolve(dir, DINOV2_FILE)?, dev)?,
            )?,
            ss_flow: SsFlow::new(
                &SsFlowConfig::default(),
                load_vb(resolve(dir, SS_FLOW_FILE)?, dev)?,
            )?,
            ss_dec: SsDecoder::new(
                &SsDecoderConfig::default(),
                load_vb(resolve(dir, SS_DEC_FILE)?, dev)?,
            )?,
            slat_flow: SlatFlow::new(
                &SlatFlowConfig::default(),
                load_vb(resolve(dir, SLAT_FLOW_FILE)?, dev)?,
            )?,
            slat_dec: SlatMeshDecoder::new(
                &SlatDecoderConfig::default(),
                load_vb(resolve(dir, SLAT_DEC_FILE)?, dev)?,
            )?,
        })
    }
}

type Cache = Mutex<HashMap<String, Arc<Models>>>;

fn models(dir: &Path, dev: &Device) -> Result<Arc<Models>> {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = format!("{}@{dev:?}", dir.display());
    if let Some(m) = cache.lock().unwrap().get(&key) {
        return Ok(m.clone());
    }
    let m = Arc::new(Models::load(dir, dev)?);
    cache.lock().unwrap().insert(key, m.clone());
    Ok(m)
}

/// Front half shared by both quality levels: image -> DINOv2 cond -> SS rectified-flow sample ->
/// 3D-conv decode to occupancy logits [1,1,64,64,64], plus the conditioning for the SLAT stage.
fn sample_occupancy(
    m: &Models,
    image: &DynamicImage,
    seed: u64,
    steps: usize,
    dev: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let img = preprocess::preprocess(image, dev)?;
    let cond = m.dinov2.forward(&img)?; // [1, 1374, 1024]
    let neg = cond.zeros_like()?;

    let noise = seeded_gaussian(&[1, SS_CH, SS_RES, SS_RES, SS_RES], seed, dev)?;
    let params = sampler::FlowEulerParams {
        steps: steps.max(1),
        ..Default::default()
    };
    let ss_flow = &m.ss_flow;
    let latent = sampler::sample(&noise, &cond, &neg, &params, |x, t, c| {
        let tt = Tensor::full(t as f32, 1, dev)?;
        ss_flow.forward(x, &tt, c)
    })?;
    let logits = m.ss_dec.forward(&latent)?; // [1, 1, 64, 64, 64]
    Ok((logits, cond, neg))
}

/// Active occupancy voxels (logit > 0) as (z,y,x) coords in C-order (matches `torch.argwhere`).
fn active_coords(logits: &Tensor) -> Result<Vec<[i32; 3]>> {
    let v = logits.flatten_all()?.to_vec1::<f32>()?;
    let r = OCC_RES as i32;
    let mut coords = Vec::new();
    for z in 0..r {
        for y in 0..r {
            for x in 0..r {
                if v[((z * r + y) * r + x) as usize] > 0.0 {
                    coords.push([z, y, x]);
                }
            }
        }
    }
    Ok(coords)
}

/// Run the coarse image-to-3D pipeline, returning the occupancy surface mesh.
pub fn generate(image: &DynamicImage, seed: u64, steps: usize) -> Result<Mesh> {
    let dir = std::env::var("PIXAL3D_MODEL")
        .map(PathBuf::from)
        .map_err(|_| anyhow!("set PIXAL3D_MODEL to the TRELLIS ckpts + dinov2 dir"))?;
    let dev = Device::Cpu;
    let m = models(&dir, &dev)?;

    let (logits, _cond, _neg) = sample_occupancy(&m, image, seed, steps, &dev)?;
    let occ: Vec<bool> = logits
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|v| v > 0.0)
        .collect();
    let mesh = mesh::occupancy_to_mesh(&occ, OCC_RES);
    if mesh.faces.is_empty() {
        bail!("sparse-structure produced no occupied voxels");
    }
    Ok(mesh)
}

/// Run the full image-to-3D pipeline: coarse occupancy -> SLAT rectified-flow on the active voxels
/// -> mesh decoder -> FlexiCubes, returning the fine textured mesh (vertices + faces + vertex RGB).
pub fn generate_textured(
    image: &DynamicImage,
    seed: u64,
    steps: usize,
) -> Result<flexicubes::TexturedMesh> {
    let dir = std::env::var("PIXAL3D_MODEL")
        .map(PathBuf::from)
        .map_err(|_| anyhow!("set PIXAL3D_MODEL to the TRELLIS ckpts + dinov2 dir"))?;
    let dev = Device::Cpu;
    let m = models(&dir, &dev)?;

    let (logits, cond, neg) = sample_occupancy(&m, image, seed, steps, &dev)?;
    let coords = active_coords(&logits)?;
    if coords.is_empty() {
        bail!("sparse-structure produced no occupied voxels");
    }

    // SLAT rectified flow: denoise 8-ch latent on the fixed active-voxel coords.
    let noise = seeded_gaussian(&[coords.len(), SLAT_CH], seed ^ 0x51A7_51A7, &dev)?;
    let params = sampler::FlowEulerParams {
        steps: steps.max(1),
        ..Default::default()
    };
    let slat_flow = &m.slat_flow;
    let coords_c = coords.clone();
    let feats = sampler::sample(&noise, &cond, &neg, &params, |x, t, c| {
        let tt = Tensor::full(t as f32, 1, &dev)?;
        Ok(slat_flow
            .forward(&Sparse::new(coords_c.clone(), x.clone()), &tt, c)?
            .feats)
    })?;
    // denormalize.
    let std = Tensor::from_vec(SLAT_STD.to_vec(), (1, SLAT_CH), &dev)?;
    let mean = Tensor::from_vec(SLAT_MEAN.to_vec(), (1, SLAT_CH), &dev)?;
    let feats = feats.broadcast_mul(&std)?.broadcast_add(&mean)?;

    // decode to per-cube FlexiCubes features and extract the surface.
    let dec = m.slat_dec.forward(&Sparse::new(coords, feats))?; // [M, 101] @ res 256
    let tm = flexicubes::extract(&dec.coords, &dec.feats, MESH_RES)?;
    if tm.mesh.faces.is_empty() {
        bail!("SLAT/FlexiCubes produced no surface");
    }
    Ok(tm)
}

#[cfg(test)]
mod tests {
    use super::*;

    // PIXAL3D_MODEL=/dir PIXAL3D_TEST_IMAGE=/img.png [PIXAL3D_OUT=/out.glb PIXAL3D_STEPS=12] \
    //   cargo test -p hanzo-engine pixal3d::pipeline -- --ignored --nocapture
    #[test]
    #[ignore = "needs PIXAL3D_MODEL + PIXAL3D_TEST_IMAGE"]
    fn end_to_end_image_to_glb() {
        let img_path = std::env::var("PIXAL3D_TEST_IMAGE").expect("set PIXAL3D_TEST_IMAGE");
        let out = std::env::var("PIXAL3D_OUT").unwrap_or_else(|_| "/tmp/pixal3d_out.glb".into());
        let steps: usize = std::env::var("PIXAL3D_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12);
        let image = image::open(&img_path).expect("load test image");

        let t0 = std::time::Instant::now();
        let mesh = generate(&image, 42, steps).unwrap();
        println!(
            "pixal3d: {} verts, {} faces in {:.1}s ({steps} steps)",
            mesh.vertices.len(),
            mesh.faces.len(),
            t0.elapsed().as_secs_f32()
        );
        assert!(!mesh.faces.is_empty());

        let glb = super::super::glb::mesh_to_glb(&mesh);
        assert_eq!(&glb[0..4], b"glTF");
        std::fs::write(&out, &glb).unwrap();
        println!("wrote {out} ({} bytes)", glb.len());
    }

    // Full image -> textured GLB. PIXAL3D_MODEL=/model PIXAL3D_TEST_IMAGE=/img.png \
    //   PIXAL3D_OUT=/out.glb PIXAL3D_STEPS=12 cargo test -p hanzo-engine \
    //   pixal3d::pipeline::end_to_end_textured -- --ignored --nocapture
    #[test]
    #[ignore = "needs full PIXAL3D_MODEL (dinov2 + ss + slat) + PIXAL3D_TEST_IMAGE"]
    fn end_to_end_textured_glb() {
        let img_path = std::env::var("PIXAL3D_TEST_IMAGE").expect("set PIXAL3D_TEST_IMAGE");
        let out =
            std::env::var("PIXAL3D_OUT").unwrap_or_else(|_| "/tmp/pixal3d_textured.glb".into());
        let steps: usize = std::env::var("PIXAL3D_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12);
        let image = image::open(&img_path).expect("load test image");

        let t0 = std::time::Instant::now();
        let tm = generate_textured(&image, 42, steps).unwrap();
        println!(
            "pixal3d textured: {} verts, {} faces, {} colors in {:.1}s ({steps} steps)",
            tm.mesh.vertices.len(),
            tm.mesh.faces.len(),
            tm.colors.len(),
            t0.elapsed().as_secs_f32()
        );
        assert!(!tm.mesh.faces.is_empty());
        let glb = super::super::glb::mesh_to_glb_colored(&tm.mesh, Some(&tm.colors));
        assert_eq!(&glb[0..4], b"glTF");
        std::fs::write(&out, &glb).unwrap();
        println!("wrote {out} ({} bytes)", glb.len());
    }
}
