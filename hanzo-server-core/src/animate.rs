//! Unified animation route handler: `/v1/animate` (alias `/v1/video/lipsync`).
//!
//! One endpoint, audio + visual in, dubbed mp4 out. The visual decides the kind:
//! a video becomes `VisualSource::Footage` (lip-sync over real frames), a still
//! image becomes `VisualSource::Portrait` (avatar generation). The pipeline picks
//! the animator via `accepts()`; muxing the rendered frames + driving audio into
//! a container is a handler concern, never the pipeline's.

use std::error::Error;
use std::sync::Arc;

use anyhow::{bail, Result};
use axum::{
    extract::{Json, State},
    http::{header, StatusCode},
    response::IntoResponse,
};
use hanzo_engine::{
    diffusion_models::animation::VisualKind, AudioInput, Constraint, Hanzo, NormalRequest, Request,
    RequestMessage, Response, SamplingParams,
};
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::{
    handler_core::{create_response_channel, send_request, ErrorToResponse, JsonError},
    media::{self, Origin},
    types::{ExtractedState, SharedState},
    util::{sanitize_error_message, validate_model_name},
    video::{mux, parse_video_url},
};

const IMAGE_EXTS: [&str; 6] = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"];
const PORTRAIT_DEFAULT_FPS: f64 = 25.0;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnimateRequest {
    pub model: String,
    /// Driving audio: http(s)/file/data URL or absolute path.
    pub audio: String,
    /// Visual source: a video -> Footage (lip-sync); a still image -> Portrait (avatar).
    pub visual: String,
    /// Output fps; defaults to the source video's fps, or 25 for a still image.
    pub fps: Option<f64>,
}

pub enum AnimateResponder {
    Mp4(Vec<u8>),
    InternalError(Box<dyn Error>),
    ValidationError(Box<dyn Error>),
}

impl IntoResponse for AnimateResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            AnimateResponder::Mp4(bytes) => {
                ([(header::CONTENT_TYPE, "video/mp4")], bytes).into_response()
            }
            AnimateResponder::InternalError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(StatusCode::INTERNAL_SERVER_ERROR)
            }
            AnimateResponder::ValidationError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(StatusCode::UNPROCESSABLE_ENTITY)
            }
        }
    }
}

fn looks_like_image(src: &str) -> bool {
    let lower = src.to_lowercase();
    IMAGE_EXTS.iter().any(|e| lower.contains(e))
        || (lower.contains("image/") && !lower.contains("image/gif"))
}

/// Decode the visual into frames + kind + output fps.
async fn decode_visual(
    visual: &str,
    fps_override: Option<f64>,
) -> Result<(Vec<DynamicImage>, VisualKind, f64)> {
    if looks_like_image(visual) {
        let media = media::load(visual, Origin::Network).await?;
        let img = image::load_from_memory(&media.bytes)?;
        Ok((
            vec![img],
            VisualKind::Portrait,
            fps_override.unwrap_or(PORTRAIT_DEFAULT_FPS),
        ))
    } else {
        let video = parse_video_url(visual, None, Origin::Network).await?;
        let fps = fps_override.unwrap_or(video.fps);
        Ok((video.frames, VisualKind::Footage, fps))
    }
}

async fn decode_audio(audio: &str) -> Result<(Arc<Vec<f32>>, usize)> {
    let media = media::load(audio, Origin::Network).await?;
    let input =
        AudioInput::from_bytes(&media.bytes).map_err(|e| anyhow::anyhow!("decode audio: {e}"))?;
    let sample_rate = input.sample_rate as usize;
    Ok((Arc::new(input.to_mono()), sample_rate))
}

pub async fn animate(
    State(state): ExtractedState,
    Json(req): Json<AnimateRequest>,
) -> AnimateResponder {
    match animate_inner(state.clone(), req).await {
        Ok(mp4) => AnimateResponder::Mp4(mp4),
        Err(e) => {
            Hanzo::maybe_log_error(state, e.as_ref());
            AnimateResponder::InternalError(
                anyhow::Error::msg(sanitize_error_message(e.as_ref())).into(),
            )
        }
    }
}

async fn animate_inner(state: SharedState, req: AnimateRequest) -> Result<Vec<u8>> {
    validate_model_name(&req.model, state.clone())?;

    let (frames, kind, fps) = decode_visual(&req.visual, req.fps).await?;
    if frames.is_empty() {
        bail!("animate: visual produced no frames");
    }
    let (pcm, sample_rate) = decode_audio(&req.audio).await?;

    let (tx, mut rx) = create_response_channel(None);
    let request = Request::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        messages: RequestMessage::Animation {
            frames,
            pcm: pcm.clone(),
            sample_rate,
            kind,
            fps,
        },
        sampling_params: SamplingParams::deterministic(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        suffix: None,
        constraint: Constraint::None,
        tool_choice: None,
        tools: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        enable_code_execution: false,
        code_execution_permission: None,
        code_execution_approval_notifier: None,
        agent_permission: None,
        agent_approval_handler: None,
        agent_approval_notifier: None,
        max_tool_rounds: None,
        tool_dispatch_url: None,
        model_id: if req.model == "default" {
            None
        } else {
            Some(req.model.clone())
        },
        truncate_sequence: false,
        session_id: None,
        files: None,
    }));
    send_request(&state, request).await?;

    // Receive rendered frames, then mux with the driving audio (handler-side, async).
    match rx.recv().await {
        Some(Response::Frames { frames, fps }) => {
            mux(&frames[..], fps, &pcm[..], sample_rate as u32).await
        }
        Some(Response::InternalError(e)) | Some(Response::ValidationError(e)) => bail!("{e}"),
        Some(Response::ModelError(m, _)) => bail!("{m}"),
        Some(_) => bail!("animate: unexpected response type"),
        None => bail!("animate: no response from model"),
    }
}
