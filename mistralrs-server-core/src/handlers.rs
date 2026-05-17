//! ## General mistral.rs server route handlers.

use anyhow::Result;
use axum::extract::{Json, State};
use mistralrs_core::{
    auto_tune, collect_system_info, parse_isq_value, run_doctor, AutoDeviceMapParams,
    AutoTuneRequest, AutoTuneResult, MistralRs, MistralRsError, ModelDType, ModelSelected,
    ModelStatus as CoreModelStatus, Request, TokenSource, TuneProfile,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    model_registry,
    openai::{ModelObject, ModelObjects},
    types::ExtractedMistralRsState,
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
  tag = "Mistral.rs",
  path = "/v1/models",
  responses((status = 200, description = "Served model info", body = ModelObjects))
)]
pub async fn models(State(state): ExtractedMistralRsState) -> Json<ModelObjects> {
    // Collect the base "loaded model" view from the in-process pipeline. This
    // is the legacy data that mistralrs-server has always returned and must
    // stay byte-stable for single-model parity (M1 task #5).
    let models_with_status = state.list_models_with_status().unwrap_or_default();
    let created = state.get_creation_time();

    // If a ModelRegistry is installed (i.e. hanzo-engine started with at
    // least one --register flag, including the auto-registered "default"),
    // build the listing from the registry so /v1/models reflects every
    // expert plus its modality bits. Otherwise fall back to the historical
    // shape used by raw mistralrs-server.
    let mut model_objects: Vec<ModelObject> = Vec::new();

    if let Some(registry) = model_registry::global() {
        // Stable sort by id so the response order is deterministic across
        // restarts. (HashMap iteration order is not.)
        let mut experts = registry.list();
        experts.sort_by(|a, b| a.id.cmp(&b.id));

        for expert in experts {
            let backend_kind = expert.backend.kind();
            let capabilities: Vec<String> = expert
                .modalities
                .capability_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect();

            // For in-process experts, pull live status / tool counts so the
            // single-model "default" entry behaves identically to before.
            let (status, tools_available, mcp_tools_count, mcp_servers_connected) =
                if matches!(
                    expert.backend,
                    crate::model_registry::ExpertBackend::InProcess
                ) {
                    // Look up by registry id first; the pipeline's own
                    // "default" routing accepts `None` to mean "the single
                    // configured model", which matches legacy behavior when
                    // the only entry is `default`.
                    let lookup_id = if expert.id == "default" {
                        None
                    } else {
                        Some(expert.id.as_str())
                    };

                    let core_status = models_with_status
                        .iter()
                        .find(|(id, _)| Some(id.as_str()) == lookup_id)
                        .map(|(_, s)| *s);

                    let is_loaded = matches!(core_status, Some(CoreModelStatus::Loaded) | None);
                    let (ta, mt, ms) = if is_loaded {
                        let tools_count = state.get_tools_count(lookup_id).unwrap_or(0);
                        let has_mcp = state.has_mcp_client(lookup_id).unwrap_or(false);
                        if has_mcp || tools_count > 0 {
                            (Some(tools_count > 0), Some(tools_count), Some(1))
                        } else {
                            (None, None, None)
                        }
                    } else {
                        (None, None, None)
                    };

                    (core_status.map(|s| s.to_string()), ta, mt, ms)
                } else {
                    // Remote/subprocess experts: M1 does not health-check
                    // them, so don't pretend to know status or tools.
                    (None, None, None, None)
                };

            model_objects.push(ModelObject {
                id: expert.id.clone(),
                object: "model",
                created,
                owned_by: "local",
                status,
                tools_available,
                mcp_tools_count,
                mcp_servers_connected,
                capabilities: Some(capabilities),
                backend: Some(backend_kind.to_string()),
            });
        }
    } else {
        // Legacy path — preserved verbatim for mistralrs-server callers that
        // never opt into the registry.
        model_objects.push(ModelObject {
            id: "default".to_string(),
            object: "model",
            created,
            owned_by: "local",
            status: None,
            tools_available: None,
            mcp_tools_count: None,
            mcp_servers_connected: None,
            capabilities: None,
            backend: None,
        });

        for (model_id, status) in &models_with_status {
            let (tools_available, mcp_tools_count, mcp_servers_connected) =
                if *status == CoreModelStatus::Loaded {
                    let tools_count = state.get_tools_count(Some(model_id)).unwrap_or(0);
                    let has_mcp = state.has_mcp_client(Some(model_id)).unwrap_or(false);

                    if has_mcp || tools_count > 0 {
                        (Some(tools_count > 0), Some(tools_count), Some(1))
                    } else {
                        (None, None, None)
                    }
                } else {
                    (None, None, None)
                };

            model_objects.push(ModelObject {
                id: model_id.clone(),
                object: "model",
                created,
                owned_by: "local",
                status: Some(status.to_string()),
                tools_available,
                mcp_tools_count,
                mcp_servers_connected,
                capabilities: None,
                backend: None,
            });
        }
    }

    Json(ModelObjects {
        object: "list",
        data: model_objects,
    })
}

#[utoipa::path(
  get,
  tag = "Mistral.rs",
  path = "/health",
  responses((status = 200, description = "Server is healthy"))
)]
pub async fn health() -> &'static str {
    "OK"
}

pub async fn system_info() -> Json<mistralrs_core::SystemInfo> {
    Json(collect_system_info())
}

pub async fn system_doctor() -> Json<mistralrs_core::DoctorReport> {
    Json(run_doctor())
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ReIsqRequest {
    #[schema(example = "Q4K")]
    ggml_type: String,
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/re_isq",
  request_body = ReIsqRequest,
  responses((status = 200, description = "Reapply ISQ to a non GGUF or GGML model."))
)]
pub async fn re_isq(
    State(state): ExtractedMistralRsState,
    Json(request): Json<ReIsqRequest>,
) -> Result<String, String> {
    let repr = format!("Re ISQ: {:?}", request.ggml_type);
    MistralRs::maybe_log_request(state.clone(), repr.clone());
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
  tag = "Mistral.rs",
  path = "/v1/models/unload",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model unloaded successfully", body = ModelStatusResponse),
    (status = 400, description = "Failed to unload model", body = ModelStatusResponse)
  )
)]
pub async fn unload_model(
    State(state): ExtractedMistralRsState,
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
                MistralRsError::ModelNotFound(_) => (ModelStatus::NotFound, None),
                MistralRsError::ModelAlreadyUnloaded(_) => (ModelStatus::Unloaded, None),
                MistralRsError::NoLoaderConfig(_) => (ModelStatus::NoLoaderConfig, None),
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
  tag = "Mistral.rs",
  path = "/v1/models/reload",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model reloaded successfully", body = ModelStatusResponse),
    (status = 400, description = "Failed to reload model", body = ModelStatusResponse)
  )
)]
pub async fn reload_model(
    State(state): ExtractedMistralRsState,
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
                MistralRsError::ModelNotFound(_) => (ModelStatus::NotFound, None),
                MistralRsError::ModelReloading(_) => (ModelStatus::Reloading, None),
                MistralRsError::ModelAlreadyLoaded(_) => (ModelStatus::Loaded, None),
                MistralRsError::ReloadFailed(msg) => {
                    (ModelStatus::InternalError, Some(msg.clone()))
                }
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
  tag = "Mistral.rs",
  path = "/v1/models/status",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model status", body = ModelStatusResponse),
    (status = 404, description = "Model not found", body = ModelStatusResponse)
  )
)]
pub async fn get_model_status(
    State(state): ExtractedMistralRsState,
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
    /// Optional max num images (vision)
    #[serde(default)]
    pub max_num_images: Option<usize>,
    /// Optional max image length (vision)
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
