use pyo3::{pyclass, pymethods};

/// Python wrapper for [`hanzo_engine::RequestedFile`].
#[pyclass]
#[derive(Clone, Debug)]
pub struct RequestedFile {
    pub(crate) name: String,
    pub(crate) format: Option<String>,
    pub(crate) description: Option<String>,
}

#[pymethods]
impl RequestedFile {
    #[new]
    #[pyo3(signature = (name, format = None, description = None))]
    fn new(name: String, format: Option<String>, description: Option<String>) -> Self {
        Self {
            name,
            format,
            description,
        }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    #[getter]
    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "RequestedFile(name={:?}, format={:?}, description={:?})",
            self.name, self.format, self.description
        )
    }
}

impl From<RequestedFile> for hanzo_engine::RequestedFile {
    fn from(f: RequestedFile) -> Self {
        hanzo_engine::RequestedFile {
            name: f.name,
            format: f.format,
            description: f.description,
        }
    }
}
