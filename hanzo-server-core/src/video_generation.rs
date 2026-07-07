//! Async text-to-video generation: `POST /v1/videos`, `GET /v1/videos/{id}`,
//! `GET /v1/videos/{id}/content` (OpenAI-Sora async-job shape).
//!
//! Video generation is long-running, so the create call returns a job id immediately and the work
//! runs in a spawned task. The job store is a process-local map; a job holds its params, live
//! status/progress, and (once done) the encoded mp4 bytes. The model forward (`wan_t2v_generate`)
//! is the one piece pending weights; the job lifecycle around it is real end to end.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Json, Path, State},
    http::{header, StatusCode},
    response::IntoResponse,
};
use hanzo_engine::diffusion_models::wan::t2v::{wan_t2v_generate, WanT2vParams, DEFAULT_FPS};
use hanzo_engine::Hanzo;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{ExtractedState, SharedState};
use crate::video::mux;

const DEFAULT_NUM_FRAMES: usize = 81;
const DEFAULT_WIDTH: usize = 1280;
const DEFAULT_HEIGHT: usize = 720;
const DEFAULT_STEPS: usize = 30;

fn default_num_frames() -> usize {
    DEFAULT_NUM_FRAMES
}
fn default_width() -> usize {
    DEFAULT_WIDTH
}
fn default_height() -> usize {
    DEFAULT_HEIGHT
}
fn default_steps() -> usize {
    DEFAULT_STEPS
}

/// A text-to-video generation request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VideoGenerationRequest {
    pub prompt: String,
    #[serde(default = "default_num_frames")]
    pub num_frames: usize,
    #[serde(default = "default_width")]
    pub width: usize,
    #[serde(default = "default_height")]
    pub height: usize,
    #[serde(default = "default_steps")]
    pub steps: usize,
}

/// Lifecycle of a video job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoJobStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

/// A single video job: its identity, live status, and (when done) the encoded result.
#[derive(Debug, Clone)]
pub struct VideoJob {
    pub id: String,
    pub status: VideoJobStatus,
    pub progress: f32,
    pub created: u64,
    pub error: Option<String>,
    pub result: Option<Vec<u8>>,
    pub params: VideoGenerationRequest,
}

impl VideoJob {
    fn queued(id: String, created: u64, params: VideoGenerationRequest) -> Self {
        Self {
            id,
            status: VideoJobStatus::Queued,
            progress: 0.0,
            created,
            error: None,
            result: None,
            params,
        }
    }
}

/// The public status view (no raw mp4 bytes; those go through `/content`).
#[derive(Debug, Clone, Serialize)]
pub struct VideoJobResponse {
    pub id: String,
    pub object: &'static str,
    pub status: VideoJobStatus,
    pub progress: f32,
    pub created: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl From<&VideoJob> for VideoJobResponse {
    fn from(job: &VideoJob) -> Self {
        Self {
            id: job.id.clone(),
            object: "video",
            status: job.status,
            progress: job.progress,
            created: job.created,
            error: job.error.clone(),
        }
    }
}

type VideoJobStore = Arc<Mutex<HashMap<String, VideoJob>>>;

/// Process-local job store, shared across all requests. The engine `Hanzo` state is per-model and
/// not the right home for cross-request job bookkeeping, so this lives beside it as a singleton.
fn job_store() -> &'static VideoJobStore {
    static STORE: OnceLock<VideoJobStore> = OnceLock::new();
    STORE.get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `POST /v1/videos` - queue a job and return its id immediately.
pub async fn create_video(
    State(state): ExtractedState,
    Json(req): Json<VideoGenerationRequest>,
) -> axum::response::Response {
    if req.prompt.trim().is_empty() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "message": "prompt must not be empty" })),
        )
            .into_response();
    }

    let id = format!("video-{}", Uuid::new_v4());
    let job = VideoJob::queued(id.clone(), now_unix(), req.clone());
    let response = VideoJobResponse::from(&job);
    job_store()
        .lock()
        .expect("video job store poisoned")
        .insert(id.clone(), job);

    tokio::spawn(run_job(state, id));

    (StatusCode::ACCEPTED, Json(response)).into_response()
}

/// `GET /v1/videos/{id}` - return the job's status JSON.
pub async fn get_video(
    State(_state): ExtractedState,
    Path(id): Path<String>,
) -> axum::response::Response {
    match job_store().lock().expect("video job store poisoned").get(&id) {
        Some(job) => (StatusCode::OK, Json(VideoJobResponse::from(job))).into_response(),
        None => not_found(&id),
    }
}

/// `GET /v1/videos/{id}/content` - the mp4 bytes (200 when done, 202 while pending).
pub async fn get_video_content(
    State(_state): ExtractedState,
    Path(id): Path<String>,
) -> axum::response::Response {
    let store = job_store().lock().expect("video job store poisoned");
    let Some(job) = store.get(&id) else {
        return not_found(&id);
    };
    match job.status {
        VideoJobStatus::Completed => match &job.result {
            Some(bytes) => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "video/mp4")],
                bytes.clone(),
            )
                .into_response(),
            None => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "message": "completed job has no content" })),
            )
                .into_response(),
        },
        VideoJobStatus::Failed => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "message": job.error.clone().unwrap_or_else(|| "video generation failed".into()),
            })),
        )
            .into_response(),
        VideoJobStatus::Queued | VideoJobStatus::Running => (
            StatusCode::ACCEPTED,
            Json(VideoJobResponse::from(job)),
        )
            .into_response(),
    }
}

fn not_found(id: &str) -> axum::response::Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "message": format!("no such video job: {id}") })),
    )
        .into_response()
}

fn set_status(id: &str, status: VideoJobStatus, progress: f32) {
    if let Some(job) = job_store()
        .lock()
        .expect("video job store poisoned")
        .get_mut(id)
    {
        job.status = status;
        job.progress = progress;
    }
}

/// The generation task: mark running, run the WAN t2v forward, mux frames to mp4, store the result.
async fn run_job(state: SharedState, id: String) {
    set_status(&id, VideoJobStatus::Running, 0.0);

    let params = {
        let store = job_store().lock().expect("video job store poisoned");
        match store.get(&id) {
            Some(job) => job.params.clone(),
            None => return,
        }
    };

    match generate(&params).await {
        Ok(mp4) => {
            let mut store = job_store().lock().expect("video job store poisoned");
            if let Some(job) = store.get_mut(&id) {
                job.result = Some(mp4);
                job.progress = 1.0;
                job.status = VideoJobStatus::Completed;
            }
        }
        Err(e) => {
            let msg = crate::util::sanitize_error_message(e.as_ref());
            Hanzo::maybe_log_error(state, e.as_ref());
            let mut store = job_store().lock().expect("video job store poisoned");
            if let Some(job) = store.get_mut(&id) {
                job.error = Some(msg);
                job.status = VideoJobStatus::Failed;
            }
        }
    }
}

/// Run the model forward then container the frames. `wan_t2v_generate` is the seam pending weights;
/// everything downstream (mux -> mp4 bytes) is real and exercised the moment the forward lands.
async fn generate(req: &VideoGenerationRequest) -> anyhow::Result<Vec<u8>> {
    let params = WanT2vParams {
        prompt: req.prompt.clone(),
        num_frames: req.num_frames,
        width: req.width,
        height: req.height,
        steps: req.steps,
    };
    let rendered = wan_t2v_generate(&params)?;
    mux(&rendered.frames, rendered.fps, &[], 0).await
}

/// Frame rate used when muxing generated video (mirrors the generator default).
pub const VIDEO_FPS: f64 = DEFAULT_FPS;
