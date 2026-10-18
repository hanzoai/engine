//! Errors in OpenAI's envelope: `{"error":{"message","type","param","code"}}`.

use axum::{
    extract::rejection::JsonRejection,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

/// A request the server refuses, or a failure it reports, with the HTTP status it carries.
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    /// A 400: the request is malformed or asks for something this server refuses.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    /// The error a request that failed to parse turns into. An `ApiError` inside the chain keeps
    /// its status; anything else was caused by the request, so it is a 400.
    pub fn from_request(e: &anyhow::Error) -> Self {
        match e.downcast_ref::<ApiError>() {
            Some(api) => api.clone(),
            None => Self::invalid(crate::util::sanitize_error_message(e.as_ref())),
        }
    }

    /// The error an arbitrary error value turns into; `status` applies unless it is an `ApiError`.
    pub fn from_error(e: &(dyn std::error::Error + 'static), status: StatusCode) -> Self {
        match e.downcast_ref::<ApiError>() {
            Some(api) => api.clone(),
            None => Self::new(status, crate::util::sanitize_error_message(e)),
        }
    }

    /// The JSON body: `{"error":{...}}`.
    pub fn body(&self) -> Value {
        envelope(self.status, &self.message)
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

impl From<JsonRejection> for ApiError {
    /// A body that is not JSON, or does not fit the request type, is a 400.
    fn from(rejection: JsonRejection) -> Self {
        Self::invalid(rejection.body_text())
    }
}

/// The error `type` for a status. Client errors outside the named ones are
/// `invalid_request_error`; server errors are `server_error`.
pub fn error_type(status: StatusCode) -> &'static str {
    match status.as_u16() {
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        400..=499 => "invalid_request_error",
        _ => "server_error",
    }
}

/// The error `code`: `engine_busy` for 503, `server_error` for an internal failure (500), else null.
pub fn error_code(status: StatusCode) -> Value {
    match status.as_u16() {
        503 => json!("engine_busy"),
        500 => json!("server_error"),
        _ => Value::Null,
    }
}

/// `{"error":{"message","type","param","code"}}` for a status and message.
pub fn envelope(status: StatusCode, message: &str) -> Value {
    json!({
        "error": {
            "message": message,
            "type": error_type(status),
            "param": Value::Null,
            "code": error_code(status),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_and_code_follow_the_status() {
        let cases = [
            (400, "invalid_request_error", Value::Null),
            (401, "authentication_error", Value::Null),
            (403, "permission_error", Value::Null),
            (404, "not_found_error", Value::Null),
            (422, "invalid_request_error", Value::Null),
            (429, "rate_limit_error", Value::Null),
            (500, "server_error", json!("server_error")),
            (502, "server_error", Value::Null),
            (503, "server_error", json!("engine_busy")),
            (504, "server_error", Value::Null),
        ];
        for (status, tp, code) in cases {
            let status = StatusCode::from_u16(status).unwrap();
            let body = envelope(status, "m");
            assert_eq!(body["error"]["type"], tp, "{status}");
            assert_eq!(body["error"]["code"], code, "{status}");
            assert_eq!(body["error"]["message"], "m");
            assert!(body["error"]["param"].is_null());
        }
    }

    #[test]
    fn an_api_error_keeps_its_status_through_anyhow() {
        let e = anyhow::Error::new(ApiError::new(StatusCode::NOT_FOUND, "no such model"));
        assert_eq!(
            ApiError::from_request(&e),
            ApiError::new(StatusCode::NOT_FOUND, "no such model")
        );
        let plain = anyhow::anyhow!("Expected `text` key in input message.");
        assert_eq!(
            ApiError::from_request(&plain),
            ApiError::invalid("Expected `text` key in input message.")
        );
    }

    #[test]
    fn from_error_uses_the_fallback_status_for_foreign_errors() {
        let e = std::io::Error::other("boom");
        let api = ApiError::from_error(&e, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(api.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(api.message, "boom");
    }
}
