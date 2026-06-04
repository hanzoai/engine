use std::{
    fmt::Debug,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result};
use hanzo_ml::{Device, Tensor};

use hanzo_quant::ShardedVarBuilder;
use hf_hub::api::sync::ApiRepo;
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use regex::Regex;
use serde::Deserialize;

use super::{ModelPaths, NormalLoadingMetadata};
use crate::{
    api_dir_list, api_get_file,
    diffusion_models::{
        flux::{
            self,
            stepper::{FluxStepper, FluxStepperConfig},
        },
        qwen_image::{
            self,
            stepper::{QwenImageStepper, QwenImageStepperConfig},
        },
        DiffusionGenerationParams,
    },
    paged_attention::AttentionImplementation,
    pipeline::{paths::AdapterPaths, EmbeddingModulePaths},
};

pub trait DiffusionModel {
    /// This returns a tensor of shape (bs, c, h, w), with values in [0, 255].
    fn forward(
        &mut self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
    ) -> hanzo_ml::Result<Tensor>;
    fn device(&self) -> &Device;
    fn max_seq_len(&self) -> usize;
}

pub trait DiffusionModelLoader: Send + Sync {
    /// If the model is being loaded with `load_model_from_hf` (so manual paths not provided), this will be called.
    fn get_model_paths(
        &self,
        api: &ApiRepo,
        model_id: &Path,
        revision: &str,
    ) -> Result<Vec<PathBuf>>;
    /// If the model is being loaded with `load_model_from_hf` (so manual paths not provided), this will be called.
    fn get_config_filenames(
        &self,
        api: &ApiRepo,
        model_id: &Path,
        revision: &str,
    ) -> Result<Vec<PathBuf>>;
    fn force_cpu_vb(&self) -> Vec<bool>;
    /// Groups the flat `get_model_paths` output into one safetensors set per VarBuilder.
    /// Default: one file per VarBuilder (matches `force_cpu_vb().len()` 1:1 with `filenames`).
    /// Sharded models (e.g. Qwen-Image) override this to merge shards into a single VB.
    fn group_model_paths(&self, filenames: &[PathBuf]) -> Vec<Vec<PathBuf>> {
        filenames.iter().map(|f| vec![f.clone()]).collect()
    }
    // `configs` and `vbs` should be corresponding. It is up to the implementer to maintain this invaraint.
    fn load(
        &self,
        configs: Vec<String>,
        vbs: Vec<ShardedVarBuilder>,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
        silent: bool,
    ) -> Result<Box<dyn DiffusionModel + Send + Sync>>;
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq)]
/// The architecture to load the diffusion model as.
pub enum DiffusionLoaderType {
    #[serde(rename = "flux")]
    Flux,
    #[serde(rename = "flux-offloaded")]
    FluxOffloaded,
    #[serde(rename = "qwen-image")]
    QwenImage,
}

impl FromStr for DiffusionLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "flux" => Ok(Self::Flux),
            "flux-offloaded" => Ok(Self::FluxOffloaded),
            "qwen-image" => Ok(Self::QwenImage),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `flux`, `flux-offloaded`, `qwen-image`."
            )),
        }
    }
}

impl DiffusionLoaderType {
    /// Auto-detect diffusion loader type from a repo file listing.
    /// Extend this when adding new diffusion pipelines.
    pub fn auto_detect_from_files(files: &[String]) -> Option<Self> {
        if Self::matches_qwen_image(files) {
            return Some(Self::QwenImage);
        }
        if Self::matches_flux(files) {
            return Some(Self::Flux);
        }
        None
    }

    fn matches_qwen_image(files: &[String]) -> bool {
        let has_transformer = files.iter().any(|f| f == "transformer/config.json");
        let has_vae = files.iter().any(|f| f == "vae/config.json");
        let has_text_encoder = files
            .iter()
            .any(|f| f.starts_with("text_encoder/") && f.ends_with(".json"));
        // Qwen-Image ships a Qwen2.5-VL text encoder dir and a `vae/`; it has no flux `ae.safetensors`.
        let has_flux_ae = files.iter().any(|f| f == "ae.safetensors");
        has_transformer && has_vae && has_text_encoder && !has_flux_ae
    }

    fn matches_flux(files: &[String]) -> bool {
        let flux_regex = Regex::new(r"^flux\\d+-(schnell|dev)\\.safetensors$");
        let Ok(flux_regex) = flux_regex else {
            return false;
        };
        let has_transformer = files.iter().any(|f| f == "transformer/config.json");
        let has_vae = files.iter().any(|f| f == "vae/config.json");
        let has_ae = files.iter().any(|f| f == "ae.safetensors");
        let has_flux = files.iter().any(|f| {
            let name = f.rsplit('/').next().unwrap_or(f);
            flux_regex.is_match(name)
        });

        has_transformer && has_vae && has_ae && has_flux
    }
}

#[derive(Clone, Debug)]
pub struct DiffusionModelPathsInner {
    pub config_filenames: Vec<PathBuf>,
    pub filenames: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct DiffusionModelPaths(pub DiffusionModelPathsInner);

impl ModelPaths for DiffusionModelPaths {
    fn get_config_filename(&self) -> &PathBuf {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_template_filename(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_chat_template_explicit(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_adapter_paths(&self) -> &AdapterPaths {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]> {
        unreachable!("Use `std::any::Any`.")
    }
}

// ======================== Flux loader

/// [`DiffusionLoader`] for a Flux Diffusion model.
///
/// [`DiffusionLoader`]: https://docs.rs/hanzo/latest/hanzo/struct.DiffusionLoader.html
pub struct FluxLoader {
    pub(crate) offload: bool,
}

impl DiffusionModelLoader for FluxLoader {
    fn get_model_paths(
        &self,
        api: &ApiRepo,
        model_id: &Path,
        revision: &str,
    ) -> Result<Vec<PathBuf>> {
        let regex = Regex::new(r"^flux\d+-(schnell|dev)\.safetensors$")?;
        let flux_name = api_dir_list!(api, model_id, true, revision)
            .filter(|x| regex.is_match(x))
            .nth(0)
            .with_context(|| "Expected at least 1 .safetensors file matching the FLUX regex, please raise an issue.")?;
        let flux_file = api_get_file!(api, &flux_name, model_id, revision);
        let ae_file = api_get_file!(api, "ae.safetensors", model_id, revision);

        // NOTE(hanzoai): disgusting way of doing this but the 0th path is the flux, 1 is ae
        Ok(vec![flux_file, ae_file])
    }
    fn get_config_filenames(
        &self,
        api: &ApiRepo,
        model_id: &Path,
        revision: &str,
    ) -> Result<Vec<PathBuf>> {
        let flux_file = api_get_file!(api, "transformer/config.json", model_id, revision);
        let ae_file = api_get_file!(api, "vae/config.json", model_id, revision);

        // NOTE(hanzoai): disgusting way of doing this but the 0th path is the flux, 1 is ae
        Ok(vec![flux_file, ae_file])
    }
    fn force_cpu_vb(&self) -> Vec<bool> {
        vec![self.offload, false]
    }
    fn load(
        &self,
        mut configs: Vec<String>,
        mut vbs: Vec<ShardedVarBuilder>,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
        silent: bool,
    ) -> Result<Box<dyn DiffusionModel + Send + Sync>> {
        let (vae_cfg, vae_vb) = (configs.remove(1), vbs.remove(1));
        let (flux_cfg, flux_vb) = (configs.remove(0), vbs.remove(0));

        let vae_cfg: flux::autoencoder::Config = serde_json::from_str(&vae_cfg)?;
        let flux_cfg: flux::model::Config = serde_json::from_str(&flux_cfg)?;

        let flux_dtype = flux_vb.dtype();
        if flux_dtype != vae_vb.dtype() {
            anyhow::bail!(
                "Expected VAE and FLUX model VBs to be the same dtype, got {:?} and {flux_dtype:?}",
                vae_vb.dtype()
            );
        }

        Ok(Box::new(FluxStepper::new(
            FluxStepperConfig::default_for_guidance(flux_cfg.guidance_embeds),
            (flux_vb, &flux_cfg),
            (vae_vb, &vae_cfg),
            flux_dtype,
            &normal_loading_metadata.real_device,
            silent,
            self.offload,
        )?))
    }
}

// ======================== Qwen-Image loader

/// [`DiffusionLoader`] for a Qwen-Image (`QwenImageTransformer2DModel`) diffusion model.
pub struct QwenImageLoader;

impl DiffusionModelLoader for QwenImageLoader {
    fn get_model_paths(
        &self,
        api: &ApiRepo,
        model_id: &Path,
        revision: &str,
    ) -> Result<Vec<PathBuf>> {
        let transformer_regex = Regex::new(r"^transformer/.*\.safetensors$")?;
        let vae_regex = Regex::new(r"^vae/.*\.safetensors$")?;
        let mut transformer_files = Vec::new();
        let mut vae_files = Vec::new();
        for f in api_dir_list!(api, model_id, true, revision) {
            if transformer_regex.is_match(&f) {
                transformer_files.push(api_get_file!(api, &f, model_id, revision));
            } else if vae_regex.is_match(&f) {
                vae_files.push(api_get_file!(api, &f, model_id, revision));
            }
        }
        if transformer_files.is_empty() {
            anyhow::bail!("Expected at least 1 transformer/*.safetensors for Qwen-Image.");
        }
        if vae_files.is_empty() {
            anyhow::bail!("Expected at least 1 vae/*.safetensors for Qwen-Image.");
        }
        // Convention: transformer shards first, then VAE shards. Counts are recovered via configs.
        let mut out = transformer_files;
        out.extend(vae_files);
        Ok(out)
    }
    fn get_config_filenames(
        &self,
        api: &ApiRepo,
        model_id: &Path,
        revision: &str,
    ) -> Result<Vec<PathBuf>> {
        let transformer = api_get_file!(api, "transformer/config.json", model_id, revision);
        let vae = api_get_file!(api, "vae/config.json", model_id, revision);
        Ok(vec![transformer, vae])
    }
    fn force_cpu_vb(&self) -> Vec<bool> {
        // One VB per loaded safetensors group: transformer + vae.
        vec![false, false]
    }
    fn group_model_paths(&self, filenames: &[PathBuf]) -> Vec<Vec<PathBuf>> {
        // Group shards by their HF subfolder so transformer/* and vae/* each form one VB.
        let is_vae = |p: &PathBuf| {
            p.parent()
                .and_then(|d| d.file_name())
                .map(|n| n == "vae")
                .unwrap_or(false)
        };
        let transformer: Vec<PathBuf> = filenames.iter().filter(|p| !is_vae(p)).cloned().collect();
        let vae: Vec<PathBuf> = filenames.iter().filter(|p| is_vae(p)).cloned().collect();
        vec![transformer, vae]
    }
    fn load(
        &self,
        mut configs: Vec<String>,
        mut vbs: Vec<ShardedVarBuilder>,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
        _silent: bool,
    ) -> Result<Box<dyn DiffusionModel + Send + Sync>> {
        let (vae_cfg, vae_vb) = (configs.remove(1), vbs.remove(1));
        let (transformer_cfg, transformer_vb) = (configs.remove(0), vbs.remove(0));

        let vae_cfg: qwen_image::autoencoder::Config = serde_json::from_str(&vae_cfg)?;
        let transformer_cfg: qwen_image::model::Config = serde_json::from_str(&transformer_cfg)?;

        let dtype = transformer_vb.dtype();

        Ok(Box::new(QwenImageStepper::new(
            QwenImageStepperConfig::default(),
            (transformer_vb, &transformer_cfg),
            (vae_vb, &vae_cfg),
            dtype,
            &normal_loading_metadata.real_device,
        )?))
    }
}
