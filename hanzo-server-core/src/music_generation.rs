//! ## Music generation route handler (`POST /v1/audio/music`).
//!
//! Music models (e.g. ACE-Step) generate a stereo waveform and reuse the speech
//! response plane (`Response::Speech` -> WAV/PCM). This handler is the typed,
//! music-specific front door; the parse + response processing are shared with
//! `speech_generation` so there is one code path from model to bytes.

use std::sync::Arc;

use axum::extract::{Json, State};
use hanzo_engine::Hanzo;

use crate::{
    handler_core::{create_response_channel, send_request, JsonError},
    openai::{AudioResponseFormat, MusicGenerationRequest, SpeechGenerationRequest},
    speech_generation::{
        handle_error, parse_request, process_non_streaming_response, SpeechGenerationResponder,
    },
};

/// Music generation endpoint handler.
#[utoipa::path(
    post,
    tag = "Hanzo",
    path = "/v1/audio/music",
    request_body = MusicGenerationRequest,
    responses((status = 200, description = "Music generation"))
)]
pub async fn music_generation(
    State(state): State<Arc<Hanzo>>,
    Json(oairequest): Json<MusicGenerationRequest>,
) -> SpeechGenerationResponder {
    let (tx, mut rx) = create_response_channel(None);

    let MusicGenerationRequest {
        model,
        input,
        response_format,
    } = oairequest;
    let speech_request = SpeechGenerationRequest {
        model,
        input,
        response_format,
    };

    let (request, response_format) = match parse_request(speech_request, state.clone(), tx) {
        Ok(x) => x,
        Err(e) => return handle_error(state, e.into()),
    };

    if !matches!(
        response_format,
        AudioResponseFormat::Wav | AudioResponseFormat::Pcm
    ) {
        return SpeechGenerationResponder::ValidationError(Box::new(JsonError::new(
            "Only support wav/pcm response format.".to_string(),
        )));
    }

    if let Err(e) = send_request(&state, request).await {
        return handle_error(state, e.into());
    }

    process_non_streaming_response(&mut rx, state, response_format).await
}
