pub mod animation;
pub(crate) mod clip;
pub(crate) mod flux;
pub mod musetalk;
pub mod pixal3d;
pub(crate) mod processor;
pub(crate) mod qwen_image;
pub(crate) mod t5;
pub mod vton;
pub mod wan;

macro_rules! generate_repr {
    ($t:ident) => {
        #[cfg(feature = "pyo3_macros")]
        #[pyo3::pymethods]
        impl $t {
            fn __repr__(&self) -> String {
                format!("{self:#?}")
            }
        }
    };
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffusionGenerationParams {
    pub height: usize,
    pub width: usize,
}

generate_repr!(DiffusionGenerationParams);

impl Default for DiffusionGenerationParams {
    /// Image dimensions will be 720x1280.
    fn default() -> Self {
        Self {
            height: 720,
            width: 1280,
        }
    }
}
