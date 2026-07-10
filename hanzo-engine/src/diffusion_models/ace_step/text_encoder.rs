//! UMT5-base text encoder for ACE-Step tag/genre conditioning (768-dim last hidden state).
//! Reuses the shared T5 encoder with `Config.umt5 = true` (per-layer relative attention bias).

use hanzo_ml::{Device, Result, Tensor};
use hanzo_quant::ShardedVarBuilder;

use crate::diffusion_models::t5::{Config, T5EncoderModel};

#[derive(Debug, Clone)]
pub struct Umt5TextEncoder {
    model: T5EncoderModel,
}

impl Umt5TextEncoder {
    pub fn new(cfg: &Config, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        Ok(Self {
            model: T5EncoderModel::load(vb, cfg, device, false)?,
        })
    }

    /// input_ids (B, T) u32 -> last hidden state (B, T, 768).
    pub fn encode(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use hanzo_ml::DType;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn read_f32_le(p: &str) -> Vec<f32> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_i64_le(p: &str) -> Vec<i64> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for i in 0..n {
            d += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        (d / (na.sqrt() * nb.sqrt())) as f32
    }

    #[test]
    fn umt5_encode_matches_reference() {
        let st = std::env::var("ACE_UMT5_ST")
            .unwrap_or_else(|_| "/home/z/work/hanzo/ace-fixtures/umt5.safetensors".to_string());
        let cfgp = std::env::var("ACE_UMT5_CONFIG")
            .unwrap_or_else(|_| "/home/z/work/hanzo/ace-fixtures/umt5_config.json".to_string());
        let fix = std::env::var("ACE_FIX_DIR")
            .unwrap_or_else(|_| "/home/z/work/hanzo/ace-fixtures".to_string());
        if !PathBuf::from(&st).is_file()
            || !PathBuf::from(format!("{fix}/umt5_hidden.f32")).is_file()
        {
            eprintln!("UMT5 fixtures absent; skipping text-encoder validation");
            return;
        }
        let device = Device::Cpu;
        let mut cfg: Config =
            serde_json::from_str(&std::fs::read_to_string(&cfgp).unwrap()).unwrap();
        cfg.umt5 = true;

        let vb = from_mmaped_safetensors(
            vec![PathBuf::from(&st)],
            Vec::new(),
            Some(DType::F32),
            &device,
            vec![None],
            true,
            None,
            |_: String| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap();
        let mut enc = Umt5TextEncoder::new(&cfg, vb, &device).unwrap();

        let ids_u32: Vec<u32> = read_i64_le(&format!("{fix}/umt5_ids.i64"))
            .iter()
            .map(|&v| v as u32)
            .collect();
        let n = ids_u32.len();
        let ids = Tensor::from_vec(ids_u32, (1, n), &device).unwrap();
        let hidden = enc.encode(&ids).unwrap();
        let got: Vec<f32> = hidden.flatten_all().unwrap().to_vec1().unwrap();
        let want = read_f32_le(&format!("{fix}/umt5_hidden.f32"));
        let c = cosine(&got, &want);
        println!(
            "umt5 encode cosine = {c:.6} (n_got={}, n_want={})",
            got.len(),
            want.len()
        );
        assert!(c > 0.999, "umt5 encode cosine {c} below 0.999");
    }
}
