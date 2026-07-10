//! OpenAI-compatible `/v1/audio/transcriptions` (ASR: audio -> text).
//!
//! Accepts either `multipart/form-data` (the OpenAI shape: a `file` part plus a
//! `model` field) or `application/json` with base64 audio in `file`. The audio
//! bytes are decoded (wav/mp3/flac/... via symphonia), transcribed by the loaded
//! ASR pipeline, and returned as `{ "text": ... }`.

use std::error::Error;

use anyhow::{anyhow, bail, Result};
use axum::{
    body::Bytes,
    extract::{FromRequest, Multipart, Request, State},
    http::{header::CONTENT_TYPE, StatusCode},
    response::IntoResponse,
    Json,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use hanzo_engine::{
    AudioInput, Constraint, Hanzo, NormalRequest, Request as EngineRequest, RequestMessage,
    Response, SamplingParams,
};
use serde::{Deserialize, Serialize};

use crate::{
    handler_core::{create_response_channel, send_request, ErrorToResponse, JsonError},
    types::{ExtractedState, SharedState},
    util::{sanitize_error_message, validate_model_name},
};

const MAX_JSON_BODY: usize = 64 * 1024 * 1024;

/// JSON-body form: base64 (optionally a `data:` URL) audio in `file`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranscriptionRequest {
    #[serde(default = "default_model")]
    pub model: String,
    /// Base64-encoded audio bytes (raw or a `data:audio/...;base64,` URL).
    pub file: String,
    /// Optional teacher-forced output language (e.g. `"Chinese"`).
    #[serde(default)]
    pub language: Option<String>,
}

fn default_model() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

pub enum TranscriptionResponder {
    Json(TranscriptionResponse),
    InternalError(Box<dyn Error>),
    ValidationError(Box<dyn Error>),
}

impl IntoResponse for TranscriptionResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            TranscriptionResponder::Json(resp) => (StatusCode::OK, Json(resp)).into_response(),
            TranscriptionResponder::InternalError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(StatusCode::INTERNAL_SERVER_ERROR)
            }
            TranscriptionResponder::ValidationError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(StatusCode::UNPROCESSABLE_ENTITY)
            }
        }
    }
}

/// A parsed transcription request: which model, the raw audio bytes, and language.
struct ParsedRequest {
    model: String,
    audio: Vec<u8>,
    language: Option<String>,
}

pub async fn audio_transcription(
    State(state): ExtractedState,
    req: Request,
) -> TranscriptionResponder {
    match transcribe_inner(state.clone(), req).await {
        Ok(text) => TranscriptionResponder::Json(TranscriptionResponse { text }),
        Err(e) => {
            Hanzo::maybe_log_error(state, e.as_ref());
            TranscriptionResponder::InternalError(
                anyhow!(sanitize_error_message(e.as_ref())).into(),
            )
        }
    }
}

async fn transcribe_inner(state: SharedState, req: Request) -> Result<String> {
    let is_multipart = req
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("multipart/form-data"));

    let ParsedRequest {
        model,
        audio,
        language,
    } = if is_multipart {
        parse_multipart(req).await?
    } else {
        parse_json(req).await?
    };

    validate_model_name(&model, state.clone())?;

    let input = AudioInput::from_bytes(&audio).map_err(|e| anyhow!("decode audio: {e}"))?;

    let (tx, mut rx) = create_response_channel(None);
    let request = EngineRequest::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        messages: RequestMessage::AudioTranscription {
            audio: input,
            language,
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
        model_id: if model == "default" {
            None
        } else {
            Some(model.clone())
        },
        truncate_sequence: false,
        session_id: None,
        files: None,
    }));
    send_request(&state, request).await?;

    match rx.recv().await {
        Some(Response::Transcription { text }) => Ok(text),
        Some(Response::InternalError(e)) | Some(Response::ValidationError(e)) => bail!("{e}"),
        Some(Response::ModelError(m, _)) => bail!("{m}"),
        Some(_) => bail!("transcription: unexpected response type"),
        None => bail!("transcription: no response from model"),
    }
}

async fn parse_multipart(req: Request) -> Result<ParsedRequest> {
    // `Multipart` ignores the state generic, so parse against `()`.
    let mut multipart = Multipart::from_request(req, &())
        .await
        .map_err(|e| anyhow!("invalid multipart body: {e}"))?;
    let mut model = None;
    let mut audio = None;
    let mut language = None;
    while let Some(field) = multipart.next_field().await? {
        match field.name() {
            Some("file") => audio = Some(field.bytes().await?.to_vec()),
            Some("model") => model = Some(field.text().await?),
            Some("language") => language = Some(field.text().await?),
            // Drain unused OpenAI fields (prompt, response_format, temperature, ...).
            _ => {
                let _ = field.bytes().await;
            }
        }
    }
    let audio = audio.ok_or_else(|| anyhow!("multipart body missing `file` part"))?;
    Ok(ParsedRequest {
        model: model.unwrap_or_else(default_model),
        audio,
        language: language.filter(|s| !s.is_empty()),
    })
}

async fn parse_json(req: Request) -> Result<ParsedRequest> {
    let bytes = Bytes::from_request(req, &())
        .await
        .map_err(|e| anyhow!("read body: {e}"))?;
    if bytes.len() > MAX_JSON_BODY {
        bail!("request body exceeds {MAX_JSON_BODY} bytes");
    }
    let parsed: TranscriptionRequest = serde_json::from_slice(&bytes)?;
    let b64 = parsed
        .file
        .split_once(";base64,")
        .map_or(parsed.file.as_str(), |(_, data)| data);
    let audio = STANDARD
        .decode(b64.trim())
        .map_err(|e| anyhow!("decode base64 audio: {e}"))?;
    Ok(ParsedRequest {
        model: parsed.model,
        audio,
        language: parsed.language.filter(|s| !s.is_empty()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;

    const AUDIO: &[u8] = &[0x52, 0x49, 0x46, 0x46, 0xDE, 0xAD, 0xBE, 0xEF];

    /// Exercises the real axum `Multipart` extractor: a form-data body with
    /// `model`, `language`, and a binary `file` part parses to the audio bytes
    /// + fields that feed `transcribe()`.
    #[tokio::test]
    async fn parse_multipart_extracts_file_and_fields() {
        let boundary = "TESTBOUNDARY";
        let mut body = Vec::new();
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nzen-asr\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nEnglish\r\n").as_bytes(),
        );
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(AUDIO);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let req = HttpRequest::builder()
            .method("POST")
            .header(
                CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap();

        let parsed = parse_multipart(req).await.unwrap();
        assert_eq!(parsed.model, "zen-asr");
        assert_eq!(parsed.audio, AUDIO);
        assert_eq!(parsed.language.as_deref(), Some("English"));
    }

    /// Missing `file` part is a client error, not a panic.
    #[tokio::test]
    async fn parse_multipart_missing_file_errors() {
        let boundary = "B";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n--{boundary}--\r\n"
        );
        let req = HttpRequest::builder()
            .method("POST")
            .header(
                CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap();
        assert!(parse_multipart(req).await.is_err());
    }

    /// Exercises the JSON/base64 seam through a real request body, including the
    /// `data:` URL prefix strip. "hi" -> "aGk=".
    #[tokio::test]
    async fn parse_json_decodes_base64_data_url() {
        let json = serde_json::json!({
            "model": "zen-asr",
            "file": "data:audio/wav;base64,aGk=",
            "language": "English",
        })
        .to_string();
        let req = HttpRequest::builder()
            .method("POST")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(json))
            .unwrap();

        let parsed = parse_json(req).await.unwrap();
        assert_eq!(parsed.model, "zen-asr");
        assert_eq!(parsed.audio, b"hi");
        assert_eq!(parsed.language.as_deref(), Some("English"));
    }

    /// Raw (non-data-URL) base64 also decodes; model defaults; empty language drops.
    #[tokio::test]
    async fn parse_json_defaults_model_and_raw_base64() {
        let req = HttpRequest::builder()
            .method("POST")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"file":"aGk="}"#))
            .unwrap();
        let parsed = parse_json(req).await.unwrap();
        assert_eq!(parsed.model, "default");
        assert_eq!(parsed.audio, b"hi");
        assert!(parsed.language.is_none());
    }
}
