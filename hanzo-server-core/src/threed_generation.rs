//! Async image-to-3D generation: `POST /v1/3d`, `GET /v1/3d/{id}`,
//! `GET /v1/3d/{id}/content` (OpenAI-Sora async-job shape).
//!
//! Image-to-3D is long-running, so the create call returns a job id immediately and the work runs
//! in a spawned task. The job store is a process-local map; a job holds its params, live
//! status/progress, and (once done) the serialized mesh bytes. `texture=false` runs the coarse
//! occupancy forward (`pixal3d_generate`); `texture=true` runs the full SLAT + FlexiCubes stage
//! (`pixal3d_generate_textured`) for a fine textured GLB. The `demo` flag runs a GPU-free path
//! that serializes [`hanzo_3d::unit_cube`], proving the job -> mesh -> bytes path.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Json, Path, State},
    http::{header, StatusCode},
    response::IntoResponse,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use hanzo_3d::{io, Mesh};
use hanzo_engine::diffusion_models::pixal3d::flexicubes::TexturedMesh;
use hanzo_engine::diffusion_models::pixal3d::{glb, pixal3d_generate, pixal3d_generate_textured};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::ExtractedState;
use crate::video::fetch_bytes;

const DEFAULT_STEPS: usize = 25;

fn default_steps() -> usize {
    DEFAULT_STEPS
}

/// Output container for a decoded mesh.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ThreeDFormat {
    #[default]
    Glb,
    Ply,
    Obj,
}

impl ThreeDFormat {
    fn mime(self) -> &'static str {
        match self {
            ThreeDFormat::Glb => "model/gltf-binary",
            ThreeDFormat::Ply => "application/ply",
            ThreeDFormat::Obj => "model/obj",
        }
    }

    fn ext(self) -> &'static str {
        match self {
            ThreeDFormat::Glb => "glb",
            ThreeDFormat::Ply => "ply",
            ThreeDFormat::Obj => "obj",
        }
    }

    /// Serialize a decoded mesh into this container's bytes.
    fn serialize_mesh(self, mesh: &Mesh) -> Vec<u8> {
        match self {
            ThreeDFormat::Glb => glb::mesh_to_glb(mesh),
            ThreeDFormat::Ply => io::mesh_to_ply(mesh).into_bytes(),
            ThreeDFormat::Obj => io::mesh_to_obj(mesh).into_bytes(),
        }
    }

    /// Serialize a textured mesh. Only GLB carries the per-vertex colours (glTF `COLOR_0`); PLY/OBJ
    /// fall back to geometry only.
    fn serialize_textured(&self, mesh: &Mesh, colors: &[[f32; 3]]) -> Vec<u8> {
        match self {
            ThreeDFormat::Glb => glb::mesh_to_glb_colored(mesh, Some(colors)),
            _ => self.serialize_mesh(mesh),
        }
    }
}

/// An image-to-3D generation request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ThreeDGenerationRequest {
    /// Conditioning image: an http(s) URL, a `data:` URL, an absolute file path, or raw base64.
    pub image: String,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub format: ThreeDFormat,
    #[serde(default = "default_steps")]
    pub steps: usize,
    /// GPU-free demo path: serialize `unit_cube()` instead of running Pixal3D.
    #[serde(default)]
    pub demo: bool,
    /// Run the full SLAT + FlexiCubes stage for a fine textured mesh (vs the coarse occupancy mesh).
    #[serde(default)]
    pub texture: bool,
}

/// Lifecycle of a 3D job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ThreeDJobStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

/// A single 3D job: its identity, live status, and (when done) the serialized mesh.
#[derive(Debug, Clone)]
pub struct ThreeDJob {
    pub id: String,
    pub status: ThreeDJobStatus,
    pub progress: f32,
    pub created: u64,
    pub error: Option<String>,
    pub result: Option<Vec<u8>>,
    pub params: ThreeDGenerationRequest,
}

impl ThreeDJob {
    fn queued(id: String, created: u64, params: ThreeDGenerationRequest) -> Self {
        Self {
            id,
            status: ThreeDJobStatus::Queued,
            progress: 0.0,
            created,
            error: None,
            result: None,
            params,
        }
    }
}

/// The public status view (no raw mesh bytes; those go through `/content`).
#[derive(Debug, Clone, Serialize)]
pub struct ThreeDJobResponse {
    pub id: String,
    pub object: &'static str,
    pub status: ThreeDJobStatus,
    pub progress: f32,
    pub format: ThreeDFormat,
    pub created: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl From<&ThreeDJob> for ThreeDJobResponse {
    fn from(job: &ThreeDJob) -> Self {
        Self {
            id: job.id.clone(),
            object: "threed.job",
            status: job.status,
            progress: job.progress,
            format: job.params.format,
            created: job.created,
            error: job.error.clone(),
        }
    }
}

type ThreeDJobStore = Arc<Mutex<HashMap<String, ThreeDJob>>>;

/// Process-local job store, shared across all requests. The engine `Hanzo` state is per-model and
/// not the right home for cross-request job bookkeeping, so this lives beside it as a singleton.
fn job_store() -> &'static ThreeDJobStore {
    static STORE: OnceLock<ThreeDJobStore> = OnceLock::new();
    STORE.get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `POST /v1/3d` - queue a job and return its id immediately.
pub async fn create_3d(
    State(_state): ExtractedState,
    Json(req): Json<ThreeDGenerationRequest>,
) -> axum::response::Response {
    if req.image.trim().is_empty() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "message": "image must not be empty" })),
        )
            .into_response();
    }

    let id = format!("threed-{}", Uuid::new_v4());
    let job = ThreeDJob::queued(id.clone(), now_unix(), req);
    let response = ThreeDJobResponse::from(&job);
    job_store()
        .lock()
        .expect("3d job store poisoned")
        .insert(id.clone(), job);

    tokio::spawn(run_job(id));

    (StatusCode::ACCEPTED, Json(response)).into_response()
}

/// `GET /v1/3d/{id}` - return the job's status JSON.
pub async fn get_3d(
    State(_state): ExtractedState,
    Path(id): Path<String>,
) -> axum::response::Response {
    match job_store().lock().expect("3d job store poisoned").get(&id) {
        Some(job) => (StatusCode::OK, Json(ThreeDJobResponse::from(job))).into_response(),
        None => not_found(&id),
    }
}

/// `GET /v1/3d/{id}/content` - the mesh bytes (200 when done, 202 while pending).
pub async fn get_3d_content(
    State(_state): ExtractedState,
    Path(id): Path<String>,
) -> axum::response::Response {
    let store = job_store().lock().expect("3d job store poisoned");
    let Some(job) = store.get(&id) else {
        return not_found(&id);
    };
    match job.status {
        ThreeDJobStatus::Completed => match &job.result {
            Some(bytes) => {
                let disposition = format!(
                    "inline; filename=\"{}.{}\"",
                    job.id,
                    job.params.format.ext()
                );
                (
                    StatusCode::OK,
                    [
                        (header::CONTENT_TYPE, job.params.format.mime().to_string()),
                        (header::CONTENT_LENGTH, bytes.len().to_string()),
                        (header::CONTENT_DISPOSITION, disposition),
                    ],
                    bytes.clone(),
                )
                    .into_response()
            }
            None => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "message": "completed job has no content" })),
            )
                .into_response(),
        },
        ThreeDJobStatus::Failed => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "message": job.error.clone().unwrap_or_else(|| "3d generation failed".into()),
            })),
        )
            .into_response(),
        ThreeDJobStatus::Queued | ThreeDJobStatus::Running => {
            (StatusCode::ACCEPTED, Json(ThreeDJobResponse::from(job))).into_response()
        }
    }
}

fn not_found(id: &str) -> axum::response::Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "message": format!("no such 3d job: {id}") })),
    )
        .into_response()
}

fn set_status(id: &str, status: ThreeDJobStatus, progress: f32) {
    if let Some(job) = job_store()
        .lock()
        .expect("3d job store poisoned")
        .get_mut(id)
    {
        job.status = status;
        job.progress = progress;
    }
}

/// The generation task: mark running, run the Pixal3D forward (or the demo cube), serialize the
/// mesh, store the result.
async fn run_job(id: String) {
    set_status(&id, ThreeDJobStatus::Running, 0.1);

    let params = {
        let store = job_store().lock().expect("3d job store poisoned");
        match store.get(&id) {
            Some(job) => job.params.clone(),
            None => return,
        }
    };

    match generate(&params).await {
        Ok(out) => {
            let bytes = match &out {
                GenOutput::Coarse(mesh) => params.format.serialize_mesh(mesh),
                GenOutput::Fine(tm) => params.format.serialize_textured(&tm.mesh, &tm.colors),
            };
            let mut store = job_store().lock().expect("3d job store poisoned");
            if let Some(job) = store.get_mut(&id) {
                job.result = Some(bytes);
                job.progress = 1.0;
                job.status = ThreeDJobStatus::Completed;
            }
        }
        Err(e) => {
            let msg = crate::util::sanitize_error_message(e.as_ref());
            let mut store = job_store().lock().expect("3d job store poisoned");
            if let Some(job) = store.get_mut(&id) {
                job.error = Some(msg);
                job.status = ThreeDJobStatus::Failed;
            }
        }
    }
}

/// The forward output: a coarse occupancy mesh or a fine textured mesh (SLAT + FlexiCubes).
enum GenOutput {
    Coarse(Mesh),
    Fine(Box<TexturedMesh>),
}

/// Decode the conditioning image, then run the model forward. `texture` selects the fine SLAT +
/// FlexiCubes path (textured mesh); otherwise the coarse occupancy mesh.
async fn generate(req: &ThreeDGenerationRequest) -> anyhow::Result<GenOutput> {
    if req.demo {
        return Ok(GenOutput::Coarse(hanzo_3d::unit_cube()));
    }
    let bytes = decode_image_input(&req.image).await?;
    let image = image::load_from_memory(&bytes)?;
    if req.texture {
        Ok(GenOutput::Fine(Box::new(pixal3d_generate_textured(&image, req.seed, req.steps)?)))
    } else {
        Ok(GenOutput::Coarse(pixal3d_generate(&image, req.seed, req.steps)?))
    }
}

/// Resolve the request `image` field to raw image bytes: URL / data-URL / path via
/// [`fetch_bytes`], else treat it as raw base64.
async fn decode_image_input(image: &str) -> anyhow::Result<Vec<u8>> {
    match fetch_bytes(image).await {
        Ok(bytes) => Ok(bytes),
        Err(_) => Ok(STANDARD.decode(image.trim())?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_cube_serializes_to_ply() {
        let mesh = hanzo_3d::unit_cube();
        let ply = String::from_utf8(ThreeDFormat::Ply.serialize_mesh(&mesh)).unwrap();
        assert!(
            ply.starts_with("ply"),
            "expected PLY header, got: {ply:.16}"
        );
        assert!(ply.contains("element vertex 8"));
    }

    #[test]
    fn demo_cube_serializes_to_obj() {
        let mesh = hanzo_3d::unit_cube();
        let obj = String::from_utf8(ThreeDFormat::Obj.serialize_mesh(&mesh)).unwrap();
        assert!(obj.contains("v "), "expected OBJ vertices");
        assert!(obj.contains("f "), "expected OBJ faces");
    }

    #[test]
    fn glb_serializes_to_binary_gltf() {
        let mesh = hanzo_3d::unit_cube();
        let glb = ThreeDFormat::Glb.serialize_mesh(&mesh);
        assert_eq!(&glb[0..4], b"glTF", "expected GLB magic");
        assert_eq!(glb.len() % 4, 0, "GLB must be 4-byte aligned");
    }
}
