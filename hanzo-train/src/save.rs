//! Write a trained adapter in PEFT layout so the engine loads it for inference.
//!
//! Tensor names match `hanzo_quant::lora::merge_lora_weights`, which looks up
//! `base_model.model.<module_prefix>.lora_A.weight` / `.lora_B.weight`. The
//! `adapter_config.json` deserializes into the engine's `hanzo_quant::lora::LoraConfig`
//! (`r`, `lora_alpha`, `target_modules`).

use std::{collections::HashMap, fs, path::Path, path::PathBuf};

use hanzo_ml::Tensor;

use crate::lora::LoraDelta;
use crate::types::{LoraConfig, PeftAdapterConfig};

/// PEFT tensor key for a LoRA factor. `factor` is `"lora_A"` or `"lora_B"`.
pub fn adapter_key(module_prefix: &str, factor: &str) -> String {
    format!("base_model.model.{module_prefix}.{factor}.weight")
}

/// Write `adapter_model.safetensors` + `adapter_config.json` into `dir`. Returns `dir`.
pub fn write_adapter(
    dir: impl AsRef<Path>,
    adapters: &[LoraDelta],
    lora: &LoraConfig,
) -> anyhow::Result<PathBuf> {
    if adapters.is_empty() {
        anyhow::bail!("no LoRA adapters to save");
    }
    let dir = dir.as_ref().to_path_buf();
    fs::create_dir_all(&dir)?;

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for delta in adapters {
        tensors.insert(adapter_key(&delta.name, "lora_A"), delta.a.detach().contiguous()?);
        tensors.insert(adapter_key(&delta.name, "lora_B"), delta.b.detach().contiguous()?);
    }
    hanzo_ml::safetensors::save(&tensors, dir.join("adapter_model.safetensors"))?;

    let cfg = PeftAdapterConfig::from(lora);
    fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&cfg)?,
    )?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::{DType, Device, Var};

    #[test]
    fn adapter_round_trips_in_engine_layout() {
        let dev = Device::Cpu;
        let rank = 4;
        let (in_dim, out_dim) = (16usize, 12usize);
        let a = Var::randn(0f32, 0.02f32, (rank, in_dim), &dev)
            .unwrap()
            .as_tensor()
            .clone();
        let b = Var::zeros((out_dim, rank), DType::F32, &dev)
            .unwrap()
            .as_tensor()
            .clone();
        let delta = LoraDelta {
            name: "model.layers.0.self_attn.q_proj".into(),
            a,
            b,
            scale: 2.0,
        };
        let lora = LoraConfig {
            rank,
            alpha: 8.0,
            target_modules: vec!["q_proj".into()],
        };

        let dir = std::env::temp_dir().join(format!("hanzo-train-save-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_adapter(&dir, std::slice::from_ref(&delta), &lora).unwrap();

        // Weights: exact engine-facing keys + shapes.
        let loaded = hanzo_ml::safetensors::load(dir.join("adapter_model.safetensors"), &dev).unwrap();
        let ka = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let kb = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight";
        assert_eq!(loaded.get(ka).unwrap().dims(), &[rank, in_dim]);
        assert_eq!(loaded.get(kb).unwrap().dims(), &[out_dim, rank]);

        // Config: PEFT field names the engine deserializes.
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("adapter_config.json")).unwrap())
                .unwrap();
        assert_eq!(cfg["r"], 4);
        assert_eq!(cfg["lora_alpha"], 8.0);
        assert_eq!(cfg["target_modules"][0], "q_proj");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
