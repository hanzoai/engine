//! Core functionality for completions.

use std::error::Error;

use anyhow::Result;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Sse},
    Json,
};
use hanzo_engine::{
    DrySamplingParams, Hanzo, StopTokens as InternalStopTokens, FORCED_CALL_DECLINED,
};
use serde_json::Value;

use crate::{error::ApiError, openai::StopTokens, types::SharedState};

/// Generic responder enum for different completion types.
#[derive(Debug)]
pub enum BaseCompletionResponder<R, S> {
    /// Server-Sent Events streaming response
    Sse(Sse<S>),
    /// Complete JSON response for non-streaming requests
    Json(R),
    /// Model error with partial response data
    ModelError(String, R),
    /// Internal server error
    InternalError(Box<dyn Error>),
    /// Request validation error
    ValidationError(Box<dyn Error>),
}

/// Generic function to handle completion errors and logging them.
pub(crate) fn handle_completion_error<R, S>(
    state: SharedState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> BaseCompletionResponder<R, S> {
    Hanzo::maybe_log_error(state, &*e);
    BaseCompletionResponder::InternalError(Box::new(ApiError::from_error(
        &*e,
        StatusCode::INTERNAL_SERVER_ERROR,
    )))
}

/// A request that could not be turned into an engine request. An `ApiError` keeps its status;
/// anything else was caused by the request, so it is a 400.
pub(crate) fn handle_request_error<R, S>(
    state: SharedState,
    e: anyhow::Error,
) -> BaseCompletionResponder<R, S> {
    Hanzo::maybe_log_error(state, e.as_ref());
    BaseCompletionResponder::ValidationError(Box::new(ApiError::from_request(&e)))
}

/// An error variant as an HTTP response in OpenAI's envelope: `status` unless the error is an
/// `ApiError` carrying its own.
pub(crate) fn error_response(
    e: &(dyn Error + 'static),
    status: StatusCode,
) -> axum::response::Response {
    ApiError::from_error(e, status).into_response()
}

/// The error a model error is: a forced tool call the model declined is a 502, any other failure
/// mid-generation a 500.
pub(crate) fn model_error(message: &str) -> ApiError {
    let status = if message == FORCED_CALL_DECLINED {
        StatusCode::BAD_GATEWAY
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    ApiError::new(status, message)
}

/// A model error: the envelope, plus what was generated before it as `partial_response`.
pub(crate) fn model_error_response(message: &str, partial: Value) -> axum::response::Response {
    let error = model_error(message);
    let mut body = error.body();
    body["partial_response"] = partial;
    (error.status, Json(body)).into_response()
}

/// Converts OpenAI stop strings to the engine's, skipping empty ones; `None` when none is left.
pub(crate) fn convert_stop_tokens(stop_seqs: Option<StopTokens>) -> Option<InternalStopTokens> {
    let seqs: Vec<String> = match stop_seqs? {
        StopTokens::Multi(sequences) => sequences,
        StopTokens::Single(sequence) => vec![sequence],
    }
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect();
    (!seqs.is_empty()).then_some(InternalStopTokens::Seqs(seqs))
}

/// Helper function to get the dry sampling params.
pub(crate) fn get_dry_sampling_params(
    dry_multiplier: Option<f32>,
    dry_sequence_breakers: Option<Vec<String>>,
    dry_base: Option<f32>,
    dry_allowed_length: Option<usize>,
) -> Result<Option<DrySamplingParams>> {
    match dry_multiplier {
        Some(multiplier) => {
            let params = DrySamplingParams::new_with_defaults(
                multiplier,
                dry_sequence_breakers,
                dry_base,
                dry_allowed_length,
            )?;
            Ok(Some(params))
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seqs(stop: Option<StopTokens>) -> Option<Vec<String>> {
        match convert_stop_tokens(stop)? {
            InternalStopTokens::Seqs(s) => Some(s),
            InternalStopTokens::Ids(_) => unreachable!(),
        }
    }

    async fn body_of(response: axum::response::Response) -> (StatusCode, Value) {
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("a body");
        (status, serde_json::from_slice(&bytes).expect("JSON"))
    }

    #[tokio::test]
    async fn errors_render_in_the_openai_envelope() {
        let refused = ApiError::invalid("tool_choice 'required' needs tools");
        let (status, body) =
            body_of(error_response(&refused, StatusCode::INTERNAL_SERVER_ERROR)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            body,
            serde_json::json!({"error": {
                "message": "tool_choice 'required' needs tools",
                "type": "invalid_request_error",
                "param": null,
                "code": null,
            }})
        );

        let foreign = std::io::Error::other("engine gone");
        let (status, body) =
            body_of(error_response(&foreign, StatusCode::INTERNAL_SERVER_ERROR)).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "server_error");

        let (status, body) = body_of(model_error_response(
            "decode failed",
            serde_json::json!({"id": "1"}),
        ))
        .await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"]["message"], "decode failed");
        assert_eq!(body["partial_response"]["id"], "1");

        let (status, body) = body_of(model_error_response(
            FORCED_CALL_DECLINED,
            serde_json::json!({}),
        ))
        .await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body["error"]["message"], FORCED_CALL_DECLINED);
        assert_eq!(body["error"]["type"], "server_error");

        let json_error = crate::handler_core::JsonError::new("not found".to_string());
        let (status, body) = body_of(crate::handler_core::ErrorToResponse::to_response(
            &json_error,
            StatusCode::NOT_FOUND,
        ))
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["type"], "not_found_error");
    }

    #[test]
    fn empty_stop_strings_are_skipped() {
        assert_eq!(seqs(None), None);
        assert_eq!(seqs(Some(StopTokens::Single(String::new()))), None);
        assert_eq!(
            seqs(Some(StopTokens::Multi(vec![String::new(), String::new()]))),
            None
        );
        assert_eq!(
            seqs(Some(StopTokens::Multi(vec![
                String::new(),
                "END".to_string()
            ]))),
            Some(vec!["END".to_string()])
        );
    }
}
