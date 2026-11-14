//! ## General hanzo server route handlers.

use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::extract::Path;
use axum::extract::{Json, State};
use axum::http::{
    header::{AUTHORIZATION, CONTENT_TYPE},
    HeaderMap, StatusCode,
};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use hanzo_engine::{
    auto_tune, collect_system_info, ledger, parse_isq_value, run_doctor, AutoDeviceMapParams,
    AutoTuneRequest, AutoTuneResult, Error, Hanzo, ModelDType, ModelSelected,
    ModelStatus as CoreModelStatus, Request, SerializedSession, TokenSource, TuneProfile,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use utoipa::ToSchema;

use crate::{
    defaults::Defaults,
    observe::{self, Snapshot, ToolFormat},
    openai::{ModelObject, ModelObjects},
    route_registry::v1_paths,
    types::ExtractedState,
};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum TuneProfileRequest {
    Quality,
    Balanced,
    Fast,
}

impl From<TuneProfileRequest> for TuneProfile {
    fn from(value: TuneProfileRequest) -> Self {
        match value {
            TuneProfileRequest::Quality => TuneProfile::Quality,
            TuneProfileRequest::Balanced => TuneProfile::Balanced,
            TuneProfileRequest::Fast => TuneProfile::Fast,
        }
    }
}

#[utoipa::path(
  get,
  tag = "Hanzo",
  path = "/v1/models",
  responses((status = 200, description = "Served model info", body = ModelObjects))
)]
pub async fn models(
    State(state): ExtractedState,
    Extension(defaults): Extension<Defaults>,
) -> Json<ModelObjects> {
    let mut model_objects = Vec::new();

    // Add "default" as a special model option. With `--served-model-name` the served id is the
    // one model listed, so a client that takes the first id gets it.
    if defaults.served_name.is_none() {
        model_objects.push(ModelObject {
            id: "default".to_string(),
            object: "model",
            created: state.get_creation_time(),
            owned_by: "local",
            status: None,
            tools_available: None,
            mcp_tools_count: None,
            mcp_servers_connected: None,
        });
    }

    // Get all models with their status (loaded, unloaded, reloading)
    let models_with_status = state.list_models_with_status().unwrap_or_default();

    for (model_id, status) in models_with_status {
        // Get model-specific information (only available for loaded models)
        let (tools_available, mcp_tools_count, mcp_servers_connected) =
            if status == CoreModelStatus::Loaded {
                let tools_count = state.get_tools_count(Some(&model_id)).unwrap_or(0);
                let has_mcp = state.has_mcp_client(Some(&model_id)).unwrap_or(false);

                if has_mcp || tools_count > 0 {
                    (Some(tools_count > 0), Some(tools_count), Some(1)) // Simplified MCP info
                } else {
                    (None, None, None)
                }
            } else {
                (None, None, None)
            };

        model_objects.push(ModelObject {
            id: model_id,
            object: "model",
            created: state.get_creation_time(),
            owned_by: "local",
            status: Some(status.to_string()),
            tools_available,
            mcp_tools_count,
            mcp_servers_connected,
        });
    }

    Json(ModelObjects {
        object: "list",
        data: model_objects,
    })
}

/// `GET /`: the server is up.
pub async fn root() -> &'static str {
    "OK"
}

/// What the server and the default engine are, read from the engine's published state without
/// waiting on it.
fn snapshot(state: &Hanzo, model: String) -> Result<Snapshot, String> {
    let config = state.config(None)?;
    let logger = state.get_logger(None).map_err(|e| e.to_string())?;
    let alive = state.engine_alive(None).map_err(|e| e.to_string())?;
    let tool_format = match config.chat_template.as_deref() {
        Some(template) if template.uses_xml_tool_calls() => ToolFormat::QwenXml,
        Some(template) if template.is_harmony_format() => ToolFormat::Harmony,
        _ => ToolFormat::Json,
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
    Ok(Snapshot {
        model,
        endpoints: v1_paths(),
        alive,
        beat: logger.last_beat(),
        load: logger.load(),
        now,
        context: config.context,
        slots: config.slots,
        drafter: config.drafter,
        indexer_budget: config.indexer_budget,
        checkpoint_format: config.kind.checkpoint_format(),
        template_effort: config
            .chat_template
            .as_deref()
            .map_or("medium", |template| template.default_reasoning_effort()),
        tool_format,
        prefix: logger.prefix_stats(),
    })
}

/// `GET /health`: what this server serves and whether its engine answers.
/// 200 when it does; 503 with the same body, `status: engine_unresponsive`, when it does not.
#[utoipa::path(
  get,
  tag = "Hanzo",
  path = "/health",
  responses(
    (status = 200, description = "The server, its engine and its request surface"),
    (status = 503, description = "The engine does not answer; the same body")
  )
)]
pub async fn health(
    State(state): ExtractedState,
    Extension(defaults): Extension<Defaults>,
) -> (StatusCode, Json<Value>) {
    let model = defaults
        .served_name
        .clone()
        .or_else(|| state.get_default_model_id().ok().flatten())
        .unwrap_or_else(|| "default".to_string());
    match snapshot(&state, model.clone()) {
        Ok(snapshot) => {
            let (responds, body) = observe::health(&snapshot, &defaults);
            let status = if responds {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            };
            (status, Json(body))
        }
        Err(detail) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "engine_unresponsive",
                "model": model,
                "engine": {"responds": false, "detail": detail},
            })),
        ),
    }
}

/// The ledger totals at the last `/metrics` scrape; the rate gauges cover what finished since.
static LAST_SCRAPE: Mutex<Option<ledger::Totals>> = Mutex::new(None);

/// `GET /metrics`: Prometheus text. Never waits on an engine.
pub async fn metrics(State(state): ExtractedState) -> impl IntoResponse {
    let loads: Vec<_> = state
        .list_models()
        .unwrap_or_default()
        .iter()
        .filter_map(|id| state.get_logger(Some(id)).ok())
        .map(|logger| logger.load())
        .collect();
    let totals = ledger::totals();
    let previous = LAST_SCRAPE
        .lock()
        .map(|mut last| last.replace(totals))
        .unwrap_or_default()
        .unwrap_or_default();
    (
        [(CONTENT_TYPE, observe::METRICS_CONTENT_TYPE)],
        observe::metrics(&totals, &previous, &loads),
    )
}

/// `GET /cache`: the default engine's prefix-cache counters; 501 when prefix caching is off.
pub async fn cache(State(state): ExtractedState) -> Response {
    let prefix = match state.get_logger(None) {
        Ok(logger) => logger.prefix_stats(),
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(observe::error(&format!("no engine is loaded: {e}"))),
            )
                .into_response()
        }
    };
    match observe::cache(&prefix, &ledger::totals()) {
        Some(body) => Json(body).into_response(),
        None => (
            StatusCode::NOT_IMPLEMENTED,
            Json(observe::error("this engine reports no prompt cache")),
        )
            .into_response(),
    }
}

pub async fn system_info() -> Json<hanzo_engine::SystemInfo> {
    Json(collect_system_info())
}

pub async fn system_doctor() -> Json<hanzo_engine::DoctorReport> {
    Json(run_doctor())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LsaStatusResponse {
    pub enabled: bool,
    /// Attention calls LSA has served since the process started.
    pub dispatches: u64,
    pub config: Option<hanzo_engine::LsaConfig>,
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn authorize_lsa_admin(headers: &HeaderMap) -> std::result::Result<(), (StatusCode, Json<Value>)> {
    let expected = std::env::var("HANZO_LSA_ADMIN_TOKEN")
        .or_else(|_| std::env::var("HANZO_ADMIN_TOKEN"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let Some(expected) = expected else {
        return Err((
            StatusCode::FORBIDDEN,
            Json(json!({
                "error": {
                    "message": "LSA runtime mutations are disabled unless HANZO_LSA_ADMIN_TOKEN is configured; configure LSA at startup via --lsa",
                    "type": "forbidden"
                }
            })),
        ));
    };

    let provided = headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            v.strip_prefix("Bearer ")
                .or_else(|| v.strip_prefix("bearer "))
        })
        .map(str::trim)
        .or_else(|| {
            headers
                .get("x-hanzo-admin-token")
                .and_then(|v| v.to_str().ok())
                .map(str::trim)
        });

    match provided {
        Some(tok) if ct_eq(expected.as_bytes(), tok.as_bytes()) => Ok(()),
        _ => Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({
                "error": {
                    "message": "Unauthorized: valid admin bearer token required for /v1/lsa mutation",
                    "type": "unauthorized"
                }
            })),
        )),
    }
}

/// GET `/v1/lsa` — inspect active Latent Sparse Attention (D-LSA / N-LSA) configuration.
pub async fn get_lsa_config() -> Json<LsaStatusResponse> {
    let config = hanzo_engine::LsaConfig::global().map(|c| (*c).clone());
    Json(LsaStatusResponse {
        enabled: config.is_some(),
        dispatches: hanzo_engine::LsaConfig::dispatches(),
        config,
    })
}

/// POST `/v1/lsa` — enable or update Latent Sparse Attention (D-LSA / N-LSA) live at runtime.
///
/// Requires admin authorization via `HANZO_LSA_ADMIN_TOKEN` (`Authorization: Bearer <token>`)
/// and validates configuration bounds before installation. In-flight sequences pin their
/// initial configuration until sequence completion.
pub async fn set_lsa_config(
    headers: HeaderMap,
    Json(config): Json<hanzo_engine::LsaConfig>,
) -> std::result::Result<Json<LsaStatusResponse>, (StatusCode, Json<Value>)> {
    authorize_lsa_admin(&headers)?;
    if let Err(msg) = config.validate() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!("Invalid LSA configuration: {msg}"),
                    "type": "invalid_request_error"
                }
            })),
        ));
    }
    config.clone().install();
    Ok(Json(LsaStatusResponse {
        enabled: true,
        dispatches: hanzo_engine::LsaConfig::dispatches(),
        config: Some(config),
    }))
}

/// DELETE `/v1/lsa` — disable Latent Sparse Attention and revert to standard attention.
///
/// Requires admin authorization via `HANZO_LSA_ADMIN_TOKEN` (`Authorization: Bearer <token>`).
pub async fn delete_lsa_config(
    headers: HeaderMap,
) -> std::result::Result<Json<LsaStatusResponse>, (StatusCode, Json<Value>)> {
    authorize_lsa_admin(&headers)?;
    hanzo_engine::LsaConfig::clear();
    Ok(Json(LsaStatusResponse {
        enabled: false,
        dispatches: hanzo_engine::LsaConfig::dispatches(),
        config: None,
    }))
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ReIsqRequest {
    #[schema(example = "Q4K")]
    ggml_type: String,
}

#[utoipa::path(
  post,
  tag = "Hanzo",
  path = "/re_isq",
  request_body = ReIsqRequest,
  responses((status = 200, description = "Reapply ISQ to a non GGUF or GGML model."))
)]
pub async fn re_isq(
    State(state): ExtractedState,
    Json(request): Json<ReIsqRequest>,
) -> Result<String, String> {
    let repr = format!("Re ISQ: {:?}", request.ggml_type);
    Hanzo::maybe_log_request(state.clone(), repr.clone());
    let request = Request::ReIsq(parse_isq_value(&request.ggml_type, None)?);
    state.get_sender(None).unwrap().send(request).await.unwrap();
    Ok(repr)
}

/// Request for model operations (unload, reload, status)
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ModelOperationRequest {
    #[schema(example = "my-model")]
    pub model_id: String,
}

/// Model status enum
#[derive(Debug, Clone, Copy, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    Loaded,
    Unloaded,
    Reloading,
    NotFound,
    /// Model doesn't have loader config for reload
    NoLoaderConfig,
    /// Internal error (e.g., lock poisoned)
    InternalError,
}

/// Response for model status operations
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ModelStatusResponse {
    #[schema(example = "my-model")]
    pub model_id: String,
    pub status: ModelStatus,
    /// Error message when status indicates an error condition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[utoipa::path(
  post,
  tag = "Hanzo",
  path = "/v1/models/unload",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model unloaded successfully", body = ModelStatusResponse),
    (status = 400, description = "Failed to unload model", body = ModelStatusResponse)
  )
)]
pub async fn unload_model(
    State(state): ExtractedState,
    Json(request): Json<ModelOperationRequest>,
) -> Json<ModelStatusResponse> {
    let model_id = request.model_id;
    match state.unload_model(&model_id) {
        Ok(()) => Json(ModelStatusResponse {
            model_id,
            status: ModelStatus::Unloaded,
            error: None,
        }),
        Err(e) => {
            let (status, error) = match &e {
                Error::ModelNotFound(_) => (ModelStatus::NotFound, None),
                Error::ModelAlreadyUnloaded(_) => (ModelStatus::Unloaded, None),
                Error::NoLoaderConfig(_) => (ModelStatus::NoLoaderConfig, None),
                _ => (ModelStatus::InternalError, Some(e.to_string())),
            };
            Json(ModelStatusResponse {
                model_id,
                status,
                error,
            })
        }
    }
}

#[utoipa::path(
  post,
  tag = "Hanzo",
  path = "/v1/models/reload",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model reloaded successfully", body = ModelStatusResponse),
    (status = 400, description = "Failed to reload model", body = ModelStatusResponse)
  )
)]
pub async fn reload_model(
    State(state): ExtractedState,
    Json(request): Json<ModelOperationRequest>,
) -> Json<ModelStatusResponse> {
    let model_id = request.model_id;
    match state.reload_model(&model_id).await {
        Ok(()) => Json(ModelStatusResponse {
            model_id,
            status: ModelStatus::Loaded,
            error: None,
        }),
        Err(e) => {
            let (status, error) = match &e {
                Error::ModelNotFound(_) => (ModelStatus::NotFound, None),
                Error::ModelReloading(_) => (ModelStatus::Reloading, None),
                Error::ModelAlreadyLoaded(_) => (ModelStatus::Loaded, None),
                Error::ReloadFailed(msg) => (ModelStatus::InternalError, Some(msg.clone())),
                _ => (ModelStatus::InternalError, Some(e.to_string())),
            };
            Json(ModelStatusResponse {
                model_id,
                status,
                error,
            })
        }
    }
}

#[utoipa::path(
  post,
  tag = "Hanzo",
  path = "/v1/models/status",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model status", body = ModelStatusResponse),
    (status = 404, description = "Model not found", body = ModelStatusResponse)
  )
)]
pub async fn get_model_status(
    State(state): ExtractedState,
    Json(request): Json<ModelOperationRequest>,
) -> Json<ModelStatusResponse> {
    let model_id = request.model_id;
    match state.get_model_status(&model_id) {
        Ok(Some(core_status)) => {
            let status = match core_status {
                CoreModelStatus::Loaded => ModelStatus::Loaded,
                CoreModelStatus::Unloaded => ModelStatus::Unloaded,
                CoreModelStatus::Reloading => ModelStatus::Reloading,
            };
            Json(ModelStatusResponse {
                model_id,
                status,
                error: None,
            })
        }
        Ok(None) => Json(ModelStatusResponse {
            model_id,
            status: ModelStatus::NotFound,
            error: None,
        }),
        Err(e) => Json(ModelStatusResponse {
            model_id,
            status: ModelStatus::InternalError,
            error: Some(e.to_string()),
        }),
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct TuneModelRequest {
    #[schema(example = "meta-llama/Llama-3.2-3B-Instruct")]
    pub model_id: String,
    /// Optional model dtype (auto, f16, bf16, etc)
    #[serde(default)]
    pub dtype: Option<String>,
    /// Optional max sequence length for tuning
    #[serde(default)]
    pub max_seq_len: Option<usize>,
    /// Optional max batch size for tuning
    #[serde(default)]
    pub max_batch_size: Option<usize>,
    /// Optional max num images (multimodal)
    #[serde(default)]
    pub max_num_images: Option<usize>,
    /// Optional max image length (multimodal)
    #[serde(default)]
    pub max_image_length: Option<usize>,
    /// Optional tuning profile
    #[serde(default)]
    pub profile: Option<TuneProfileRequest>,
    /// Optional fixed ISQ level to test (e.g., Q4K)
    #[serde(default)]
    pub requested_isq: Option<String>,
    /// Optional HF token source
    #[serde(default)]
    pub token_source: Option<String>,
    /// Optional HF revision
    #[serde(default)]
    pub hf_revision: Option<String>,
    /// Force CPU-only tuning
    #[serde(default)]
    pub cpu: Option<bool>,
}

pub async fn tune_model(
    Json(request): Json<TuneModelRequest>,
) -> Result<Json<AutoTuneResult>, String> {
    let token_source = match request.token_source {
        Some(value) => value
            .parse()
            .map_err(|err| format!("Invalid token_source: {err}"))?,
        None => TokenSource::CacheToken,
    };

    let dtype = request
        .dtype
        .as_deref()
        .unwrap_or("auto")
        .parse::<ModelDType>()
        .map_err(|err| format!("Invalid dtype: {err}"))?;

    let max_seq_len = request
        .max_seq_len
        .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN);
    let max_batch_size = request
        .max_batch_size
        .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE);

    let model_selected = ModelSelected::Run {
        model_id: request.model_id.clone(),
        tokenizer_json: None,
        dtype,
        topology: None,
        organization: None,
        write_uqff: None,
        from_uqff: None,
        imatrix: None,
        calibration_file: None,
        max_edge: None,
        max_seq_len,
        max_batch_size,
        max_num_images: request.max_num_images,
        max_image_length: request.max_image_length,
        hf_cache_path: None,
        matformer_config_path: None,
        matformer_slice_name: None,
    };

    let requested_isq = match request.requested_isq {
        Some(value) => {
            Some(parse_isq_value(&value, None).map_err(|err| format!("Invalid isq value: {err}"))?)
        }
        None => None,
    };

    let tune_request = AutoTuneRequest {
        model: model_selected,
        token_source,
        hf_revision: request.hf_revision,
        force_cpu: request.cpu.unwrap_or(false),
        profile: request
            .profile
            .map(Into::into)
            .unwrap_or(TuneProfile::Balanced),
        requested_isq,
    };

    auto_tune(tune_request)
        .map(Json)
        .map_err(|err| err.to_string())
}

/// GET `/v1/sessions/{session_id}`. 404 if the session doesn't exist.
#[utoipa::path(
    get,
    tag = "Hanzo",
    path = "/v1/sessions/{session_id}",
    params(("session_id" = String, Path, description = "Session ID to export")),
    responses(
        (status = 200, description = "Serialized agentic session", body = SerializedSession),
        (status = 404, description = "Session not found"),
    )
)]
pub async fn get_session(
    State(state): ExtractedState,
    Path(session_id): Path<String>,
) -> Result<Json<SerializedSession>, (StatusCode, String)> {
    match state.export_session(None, &session_id) {
        Ok(Some(session)) => Ok(Json(session)),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            format!("Session {session_id} not found"),
        )),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

/// PUT `/v1/sessions/{session_id}`. Replaces any existing session.
#[utoipa::path(
    put,
    tag = "Hanzo",
    path = "/v1/sessions/{session_id}",
    params(("session_id" = String, Path, description = "Session ID to import as")),
    request_body = SerializedSession,
    responses(
        (status = 200, description = "Session imported"),
        (status = 400, description = "Invalid session payload"),
    )
)]
pub async fn put_session(
    State(state): ExtractedState,
    Path(session_id): Path<String>,
    Json(session): Json<SerializedSession>,
) -> Result<StatusCode, (StatusCode, String)> {
    state
        .import_session(None, session_id, session)
        .map(|()| StatusCode::OK)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))
}

/// DELETE `/v1/sessions/{session_id}`. Idempotent: returns 200 either way.
#[utoipa::path(
    delete,
    tag = "Hanzo",
    path = "/v1/sessions/{session_id}",
    params(("session_id" = String, Path, description = "Session ID to delete")),
    responses((status = 200, description = "Session deleted (or did not exist)"))
)]
pub async fn delete_session(
    State(state): ExtractedState,
    Path(session_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    state
        .delete_session(None, &session_id)
        .map(|_| StatusCode::OK)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[tokio::test]
    async fn lsa_mutations_enforce_admin_token_and_bounds() {
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();

        hanzo_engine::LsaConfig::clear();
        std::env::remove_var("HANZO_LSA_ADMIN_TOKEN");
        std::env::remove_var("HANZO_ADMIN_TOKEN");

        let valid_cfg = hanzo_engine::LsaConfig::zen_27b();

        // 1. Without HANZO_LSA_ADMIN_TOKEN configured, mutations return 403 Forbidden.
        let err = set_lsa_config(HeaderMap::new(), Json(valid_cfg.clone()))
            .await
            .expect_err("unconfigured admin token must forbid POST /v1/lsa");
        assert_eq!(err.0, StatusCode::FORBIDDEN);
        assert!(hanzo_engine::LsaConfig::global().is_none());

        let err_del = delete_lsa_config(HeaderMap::new())
            .await
            .expect_err("unconfigured admin token must forbid DELETE /v1/lsa");
        assert_eq!(err_del.0, StatusCode::FORBIDDEN);

        // 2. With HANZO_LSA_ADMIN_TOKEN set, missing or wrong bearer token returns 401 Unauthorized.
        std::env::set_var("HANZO_LSA_ADMIN_TOKEN", "test-secret-admin-token");

        let err_missing = set_lsa_config(HeaderMap::new(), Json(valid_cfg.clone()))
            .await
            .expect_err("missing bearer token must return 401");
        assert_eq!(err_missing.0, StatusCode::UNAUTHORIZED);
        assert!(hanzo_engine::LsaConfig::global().is_none());

        let mut bad_headers = HeaderMap::new();
        bad_headers.insert(
            AUTHORIZATION,
            HeaderValue::from_static("Bearer wrong-token"),
        );
        let err_wrong = set_lsa_config(bad_headers.clone(), Json(valid_cfg.clone()))
            .await
            .expect_err("wrong bearer token must return 401");
        assert_eq!(err_wrong.0, StatusCode::UNAUTHORIZED);
        assert!(hanzo_engine::LsaConfig::global().is_none());

        // 3. Valid admin token with out-of-bounds config returns 400 Bad Request.
        let mut auth_headers = HeaderMap::new();
        auth_headers.insert(
            AUTHORIZATION,
            HeaderValue::from_static("Bearer test-secret-admin-token"),
        );
        let invalid_cfg = valid_cfg.clone().with_window_size(0);
        let err_bounds = set_lsa_config(auth_headers.clone(), Json(invalid_cfg))
            .await
            .expect_err("window_size=0 must return 400");
        assert_eq!(err_bounds.0, StatusCode::BAD_REQUEST);
        assert!(hanzo_engine::LsaConfig::global().is_none());

        // 4. Valid admin token + valid config succeeds (200 OK) and installs config; DELETE clears it.
        let Json(status_on) = set_lsa_config(auth_headers.clone(), Json(valid_cfg.clone()))
            .await
            .expect("valid admin token + valid config must succeed");
        assert!(status_on.enabled);
        assert_eq!(status_on.config, Some(valid_cfg));

        let err_del_unauth = delete_lsa_config(bad_headers)
            .await
            .expect_err("wrong bearer token must fail DELETE /v1/lsa");
        assert_eq!(err_del_unauth.0, StatusCode::UNAUTHORIZED);
        assert!(hanzo_engine::LsaConfig::global().is_some());

        let Json(status_off) = delete_lsa_config(auth_headers)
            .await
            .expect("valid admin token must allow DELETE /v1/lsa");
        assert!(!status_off.enabled);
        assert!(hanzo_engine::LsaConfig::global().is_none());

        std::env::remove_var("HANZO_LSA_ADMIN_TOKEN");
    }
}
