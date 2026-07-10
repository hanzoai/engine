//! Pixal3D image-to-3D pipeline (coarse-geometry path).
//!
//! image -> DINOv2 tokens -> sparse-structure rectified-flow sample -> 3D-conv decode to a 64^3
//! occupancy grid -> boundary surface mesh. This is the dense half of TRELLIS end to end; the sparse
//! SLAT flow + FlexiCubes mesh refinement is the remaining stage.
//!
//! Weights live under `PIXAL3D_MODEL` (the microsoft/TRELLIS-image-large snapshot, plus a converted
//! `dinov2_vitl14_reg.safetensors`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, bail, Result};
use hanzo_3d::Mesh;
use hanzo_ml::{DType, Device, Tensor};
use hanzo_quant::ShardedVarBuilder;
use image::DynamicImage;

use super::dinov2::{DinoV2, DinoV2Config};
use super::ss_decoder::{SsDecoder, SsDecoderConfig};
use super::ss_flow::{SsFlow, SsFlowConfig};
use super::{mesh, preprocess, sampler};
use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};

const DINOV2_FILE: &str = "dinov2_vitl14_reg.safetensors";
const SS_FLOW_FILE: &str = "ss_flow_img_dit_L_16l8_fp16.safetensors";
const SS_DEC_FILE: &str = "ss_dec_conv3d_16l8_fp16.safetensors";
const SS_RES: usize = 16;
const SS_CH: usize = 8;
const OCC_RES: usize = 64;

struct Models {
    dinov2: DinoV2,
    ss_flow: SsFlow,
    ss_dec: SsDecoder,
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

/// Run the coarse image-to-3D pipeline, returning the occupancy surface mesh.
pub fn generate(image: &DynamicImage, seed: u64, steps: usize) -> Result<Mesh> {
    let dir = std::env::var("PIXAL3D_MODEL")
        .map(PathBuf::from)
        .map_err(|_| anyhow!("set PIXAL3D_MODEL to the TRELLIS ckpts + dinov2 dir"))?;
    let dev = Device::Cpu;
    let m = models(&dir, &dev)?;

    let img = preprocess::preprocess(image, &dev)?;
    let cond = m.dinov2.forward(&img)?; // [1, 1374, 1024]
    let neg = cond.zeros_like()?;

    dev.set_seed(seed)?;
    let noise = Tensor::randn(0f32, 1.0, (1, SS_CH, SS_RES, SS_RES, SS_RES), &dev)?;
    let params = sampler::FlowEulerParams {
        steps: steps.max(1),
        ..Default::default()
    };
    let ss_flow = &m.ss_flow;
    let latent = sampler::sample(&noise, &cond, &neg, &params, |x, t, c| {
        let tt = Tensor::full(t as f32, 1, &dev)?;
        ss_flow.forward(x, &tt, c)
    })?;

    let logits = m.ss_dec.forward(&latent)?; // [1, 1, 64, 64, 64]
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
}
