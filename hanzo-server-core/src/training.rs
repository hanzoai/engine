//! ## Training API — the wire mirror of `hanzo-train`'s Tinker-shaped primitives.
//!
//! Lets a client drive `create → forward_backward → optim_step → sample →
//! save_weights` over HTTP against the same server that does inference:
//!
//! - `POST /v1/training/clients` — load a base model + inject LoRA (async load; poll status)
//! - `GET /v1/training/clients` / `GET /v1/training/clients/{id}` — list / inspect (+ loss history)
//! - `POST /v1/training/clients/{id}/forward_backward` — accumulate gradients
//! - `POST /v1/training/clients/{id}/optim_step` — apply AdamW to them
//! - `POST /v1/training/clients/{id}/sample` — decode from the current base+LoRA weights
//! - `POST /v1/training/clients/{id}/save_weights` — write the PEFT adapter for inference
//! - `DELETE /v1/training/clients/{id}` — drop the client and free its memory
//!
//! Model loads and train steps are compute-bound: every heavy call runs on the
//! blocking pool holding that client's own lock, so ops on one client serialize
//! while the async executor (and other clients) stay free. List/inspect read
//! separate metadata and never wait on a running step.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::extract::Path;
use axum::http::StatusCode;
use axum::{Extension, Json};
use hanzo_ml::{DType, Device};
use hanzo_train::data::{tokenize_example, Example};
use hanzo_train::{
    create_lora_training_client, AdamParams, Datum, ForwardBackwardOutput, LoraConfig, ModelInput,
    SamplingParams, TrainingClient,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `(status, message)` errors, rendered by axum as a plain-text response.
type TrainingError = (StatusCode, String);

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// Lifecycle of a training client.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum TrainingClientStatus {
    Loading,
    Ready,
    Failed,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct CreateTrainingClientRequest {
    /// Base model: a Hugging Face repo id or a local directory.
    #[schema(example = "HuggingFaceTB/SmolLM2-135M")]
    pub base_model: String,
    /// LoRA adapter shape. Defaults to rank 16 / alpha 32 over the seven
    /// llama-family projections.
    #[serde(default)]
    pub lora_config: LoraConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct TrainingClientInfo {
    pub id: String,
    pub base_model: String,
    pub status: TrainingClientStatus,
    /// Why the load failed, when `status == failed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub lora_config: LoraConfig,
    /// Number of trainable LoRA parameters; present once ready.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trainable_params: Option<usize>,
    pub forward_backward_calls: usize,
    pub optim_steps: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_loss: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct TrainingClientDetail {
    #[serde(flatten)]
    pub info: TrainingClientInfo,
    /// Loss of every `forward_backward` call, in order.
    pub loss_history: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct TrainingClientList {
    pub clients: Vec<TrainingClientInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct DeleteTrainingClientResponse {
    pub id: String,
    pub deleted: bool,
}

/// One supervised example: either raw text — tokenized server-side with the
/// client's tokenizer, prompt masked out, completion (+ EOS) supervised — or a
/// pre-tokenized [`Datum`] verbatim (Tinker style).
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum WireDatum {
    Text { prompt: String, completion: String },
    Tokens(Datum),
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct ForwardBackwardRequest {
    pub data: Vec<WireDatum>,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct OptimStepRequest {
    #[serde(default)]
    pub adam_params: AdamParams,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct OptimStepResponse {
    /// Total optimizer steps applied to this client so far.
    pub optim_steps: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct SampleRequest {
    /// Text prompt, tokenized with BOS. Exactly one of `prompt` / `tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Pre-tokenized prompt (BOS included by the caller if wanted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u32>>,
    #[serde(default)]
    pub sampling_params: SamplingParams,
    /// Sequences to decode; seeds derive from `sampling_params.seed + i`.
    #[serde(default = "default_num_samples")]
    pub num_samples: usize,
}

fn default_num_samples() -> usize {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct SampledSequence {
    pub tokens: Vec<u32>,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct SampleResponse {
    pub sequences: Vec<SampledSequence>,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct SaveWeightsRequest {
    /// Adapter name (`[A-Za-z0-9._-]`); becomes the directory name.
    pub name: String,
    /// Parent directory; defaults to `~/.cache/hanzo/adapters/{client_id}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct SaveWeightsResponse {
    /// Directory the PEFT adapter was written to — load it with the engine's
    /// inference LoRA loader to sample from the trained weights.
    pub path: String,
    pub format: String,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Registry of live training clients, injected as an axum `Extension`.
#[derive(Clone, Default)]
pub struct TrainingState {
    sessions: Arc<Mutex<HashMap<String, Arc<Session>>>>,
}

impl TrainingState {
    fn insert(&self, session: Arc<Session>) {
        self.sessions
            .lock()
            .expect("training sessions lock")
            .insert(session.id.clone(), session);
    }

    fn get(&self, id: &str) -> Result<Arc<Session>, TrainingError> {
        self.sessions
            .lock()
            .expect("training sessions lock")
            .get(id)
            .cloned()
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("no training client `{id}`")))
    }

    fn remove(&self, id: &str) -> Option<Arc<Session>> {
        self.sessions
            .lock()
            .expect("training sessions lock")
            .remove(id)
    }

    fn list(&self) -> Vec<Arc<Session>> {
        let mut sessions: Vec<_> = self
            .sessions
            .lock()
            .expect("training sessions lock")
            .values()
            .cloned()
            .collect();
        sessions.sort_by_key(|s| s.seq);
        sessions
    }
}

/// Counters and status — kept apart from the client so list/inspect never
/// block behind a running train step.
#[derive(Default)]
struct Meta {
    status: Option<TrainingClientStatus>,
    error: Option<String>,
    trainable_params: Option<usize>,
    forward_backward_calls: usize,
    optim_steps: usize,
    loss_history: Vec<f32>,
}

struct Session {
    id: String,
    /// Creation order, for stable listings.
    seq: u64,
    base_model: String,
    lora: LoraConfig,
    meta: Mutex<Meta>,
    /// `None` until the load completes. The lock serializes all heavy ops on
    /// this client; ops take it as an owned guard and move to the blocking pool.
    client: Arc<tokio::sync::Mutex<Option<TrainingClient>>>,
}

impl Session {
    fn new(base_model: String, lora: LoraConfig) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Self {
            id: format!("tc-{}", uuid::Uuid::new_v4().simple()),
            seq: SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            base_model,
            lora,
            meta: Mutex::new(Meta {
                status: Some(TrainingClientStatus::Loading),
                ..Meta::default()
            }),
            client: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    /// Blocking: load the base model + inject LoRA, then flip status.
    fn load(&self) {
        match create_lora_training_client(
            &self.base_model,
            self.lora.clone(),
            Device::Cpu,
            DType::F32,
        ) {
            Ok(client) => {
                let trainable_params = client.num_trainable_params();
                *self.client.blocking_lock() = Some(client);
                let mut meta = self.meta.lock().expect("training meta lock");
                meta.status = Some(TrainingClientStatus::Ready);
                meta.trainable_params = Some(trainable_params);
                tracing::info!(id = %self.id, model = %self.base_model, trainable_params, "training client ready");
            }
            Err(e) => {
                let mut meta = self.meta.lock().expect("training meta lock");
                meta.status = Some(TrainingClientStatus::Failed);
                meta.error = Some(format!("{e:#}"));
                tracing::warn!(id = %self.id, model = %self.base_model, error = %format!("{e:#}"), "training client failed to load");
            }
        }
    }

    fn info(&self) -> TrainingClientInfo {
        let meta = self.meta.lock().expect("training meta lock");
        TrainingClientInfo {
            id: self.id.clone(),
            base_model: self.base_model.clone(),
            status: meta.status.unwrap_or(TrainingClientStatus::Loading),
            error: meta.error.clone(),
            lora_config: self.lora.clone(),
            trainable_params: meta.trainable_params,
            forward_backward_calls: meta.forward_backward_calls,
            optim_steps: meta.optim_steps,
            last_loss: meta.loss_history.last().copied(),
        }
    }

    fn detail(&self) -> TrainingClientDetail {
        let loss_history = self
            .meta
            .lock()
            .expect("training meta lock")
            .loss_history
            .clone();
        TrainingClientDetail {
            info: self.info(),
            loss_history,
        }
    }

    /// Take this client's op lock, erroring while it is loading or failed.
    async fn ready_client(
        &self,
    ) -> Result<tokio::sync::OwnedMutexGuard<Option<TrainingClient>>, TrainingError> {
        let guard = self.client.clone().lock_owned().await;
        if guard.is_some() {
            return Ok(guard);
        }
        let meta = self.meta.lock().expect("training meta lock");
        Err(match meta.status {
            Some(TrainingClientStatus::Failed) => (
                StatusCode::CONFLICT,
                format!(
                    "training client `{}` failed to load: {}",
                    self.id,
                    meta.error.as_deref().unwrap_or("unknown error")
                ),
            ),
            _ => (
                StatusCode::CONFLICT,
                format!("training client `{}` is still loading", self.id),
            ),
        })
    }

    fn record_forward_backward(&self, loss: f32) {
        let mut meta = self.meta.lock().expect("training meta lock");
        meta.forward_backward_calls += 1;
        meta.loss_history.push(loss);
    }

    fn record_optim_step(&self) -> usize {
        let mut meta = self.meta.lock().expect("training meta lock");
        meta.optim_steps += 1;
        meta.optim_steps
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

fn internal(e: impl std::fmt::Display) -> TrainingError {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

fn bad_request(e: impl std::fmt::Display) -> TrainingError {
    (StatusCode::BAD_REQUEST, e.to_string())
}

/// Create a training client: registers it immediately (status `loading`) and
/// loads the base model in the background — poll `GET /v1/training/clients/{id}`.
#[utoipa::path(
  post,
  tag = "Hanzo",
  path = "/v1/training/clients",
  request_body = CreateTrainingClientRequest,
  responses((status = 200, description = "Training client registered; base model loading", body = TrainingClientInfo))
)]
pub async fn create_training_client(
    Extension(state): Extension<TrainingState>,
    Json(req): Json<CreateTrainingClientRequest>,
) -> Result<Json<TrainingClientInfo>, TrainingError> {
    if req.base_model.trim().is_empty() {
        return Err(bad_request("base_model is required"));
    }
    if req.lora_config.rank == 0 {
        return Err(bad_request("lora_config.rank must be >= 1"));
    }
    let session = Arc::new(Session::new(req.base_model, req.lora_config));
    state.insert(session.clone());
    let info = session.info();
    tokio::task::spawn_blocking(move || session.load());
    Ok(Json(info))
}

#[utoipa::path(
  get,
  tag = "Hanzo",
  path = "/v1/training/clients",
  responses((status = 200, description = "All live training clients", body = TrainingClientList))
)]
pub async fn list_training_clients(
    Extension(state): Extension<TrainingState>,
) -> Json<TrainingClientList> {
    Json(TrainingClientList {
        clients: state.list().iter().map(|s| s.info()).collect(),
    })
}

#[utoipa::path(
  get,
  tag = "Hanzo",
  path = "/v1/training/clients/{id}",
  responses(
    (status = 200, description = "Training client state + loss history", body = TrainingClientDetail),
    (status = 404, description = "No such training client")
  )
)]
pub async fn get_training_client(
    Extension(state): Extension<TrainingState>,
    Path(id): Path<String>,
) -> Result<Json<TrainingClientDetail>, TrainingError> {
    Ok(Json(state.get(&id)?.detail()))
}

/// Drop a training client. An in-flight op finishes first; memory frees after.
#[utoipa::path(
  delete,
  tag = "Hanzo",
  path = "/v1/training/clients/{id}",
  responses(
    (status = 200, description = "Training client removed", body = DeleteTrainingClientResponse),
    (status = 404, description = "No such training client")
  )
)]
pub async fn delete_training_client(
    Extension(state): Extension<TrainingState>,
    Path(id): Path<String>,
) -> Result<Json<DeleteTrainingClientResponse>, TrainingError> {
    match state.remove(&id) {
        Some(_) => Ok(Json(DeleteTrainingClientResponse { id, deleted: true })),
        None => Err((StatusCode::NOT_FOUND, format!("no training client `{id}`"))),
    }
}

/// Tokenize / validate the wire batch into `Datum`s with the client's tokenizer.
fn to_data(client: &TrainingClient, wire: &[WireDatum]) -> anyhow::Result<Vec<Datum>> {
    wire.iter()
        .enumerate()
        .map(|(i, w)| match w {
            WireDatum::Tokens(d) => {
                d.validate()?;
                Ok(d.clone())
            }
            WireDatum::Text { prompt, completion } => tokenize_example(
                client.tokenizer(),
                &Example {
                    prompt: prompt.clone(),
                    completion: completion.clone(),
                },
                client.bos_token_id(),
                client.eos_token_id(),
            )?
            .ok_or_else(|| anyhow::anyhow!("example {i} has no trainable tokens")),
        })
        .collect()
}

/// Forward the batch with gradient tracking and accumulate gradients
/// (Tinker's `forward_backward`). Gradients apply on the next `optim_step`.
#[utoipa::path(
  post,
  tag = "Hanzo",
  path = "/v1/training/clients/{id}/forward_backward",
  request_body = ForwardBackwardRequest,
  responses(
    (status = 200, description = "Loss over the supervised tokens of the batch", body = ForwardBackwardOutput),
    (status = 404, description = "No such training client"),
    (status = 409, description = "Client is still loading or failed to load")
  )
)]
pub async fn training_forward_backward(
    Extension(state): Extension<TrainingState>,
    Path(id): Path<String>,
    Json(req): Json<ForwardBackwardRequest>,
) -> Result<Json<ForwardBackwardOutput>, TrainingError> {
    if req.data.is_empty() {
        return Err(bad_request("data must be non-empty"));
    }
    let session = state.get(&id)?;
    let mut guard = session.ready_client().await?;
    let out = tokio::task::spawn_blocking(move || {
        let client = guard.as_mut().expect("ready_client guarantees Some");
        let data = to_data(client, &req.data)?;
        client.forward_backward(&data)
    })
    .await
    .map_err(internal)?
    .map_err(bad_request)?;
    session.record_forward_backward(out.loss);
    Ok(Json(out))
}

/// Apply AdamW to the accumulated gradients (Tinker's `optim_step`).
#[utoipa::path(
  post,
  tag = "Hanzo",
  path = "/v1/training/clients/{id}/optim_step",
  request_body = OptimStepRequest,
  responses(
    (status = 200, description = "Gradients applied", body = OptimStepResponse),
    (status = 404, description = "No such training client"),
    (status = 409, description = "Client is still loading or failed to load")
  )
)]
pub async fn training_optim_step(
    Extension(state): Extension<TrainingState>,
    Path(id): Path<String>,
    Json(req): Json<OptimStepRequest>,
) -> Result<Json<OptimStepResponse>, TrainingError> {
    let session = state.get(&id)?;
    let mut guard = session.ready_client().await?;
    tokio::task::spawn_blocking(move || {
        let client = guard.as_mut().expect("ready_client guarantees Some");
        client.optim_step(req.adam_params)
    })
    .await
    .map_err(internal)?
    .map_err(bad_request)?;
    Ok(Json(OptimStepResponse {
        optim_steps: session.record_optim_step(),
    }))
}

/// Decode from the current base+LoRA weights (Tinker's `sample`).
#[utoipa::path(
  post,
  tag = "Hanzo",
  path = "/v1/training/clients/{id}/sample",
  request_body = SampleRequest,
  responses(
    (status = 200, description = "Sampled sequences", body = SampleResponse),
    (status = 404, description = "No such training client"),
    (status = 409, description = "Client is still loading or failed to load")
  )
)]
pub async fn training_sample(
    Extension(state): Extension<TrainingState>,
    Path(id): Path<String>,
    Json(req): Json<SampleRequest>,
) -> Result<Json<SampleResponse>, TrainingError> {
    if req.num_samples == 0 {
        return Err(bad_request("num_samples must be >= 1"));
    }
    if req.prompt.is_some() == req.tokens.is_some() {
        return Err(bad_request("provide exactly one of `prompt` or `tokens`"));
    }
    let session = state.get(&id)?;
    let guard = session.ready_client().await?;
    let sequences = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<SampledSequence>> {
        let client = guard.as_ref().expect("ready_client guarantees Some");
        let ids = match (&req.prompt, &req.tokens) {
            (Some(prompt), None) => {
                let mut ids = Vec::new();
                if let Some(b) = client.bos_token_id() {
                    ids.push(b);
                }
                ids.extend(
                    client
                        .tokenizer()
                        .encode(prompt.as_str(), false)
                        .map_err(anyhow::Error::msg)?
                        .get_ids()
                        .iter()
                        .copied(),
                );
                ids
            }
            (None, Some(tokens)) => tokens.clone(),
            _ => unreachable!("validated above"),
        };
        (0..req.num_samples)
            .map(|i| {
                let params = SamplingParams {
                    seed: req.sampling_params.seed.wrapping_add(i as u64),
                    ..req.sampling_params.clone()
                };
                let tokens = client.sample(&ModelInput::from_ints(ids.clone()), &params)?;
                let text = client
                    .tokenizer()
                    .decode(&tokens, true)
                    .map_err(anyhow::Error::msg)?;
                Ok(SampledSequence { tokens, text })
            })
            .collect()
    })
    .await
    .map_err(internal)?
    .map_err(bad_request)?;
    Ok(Json(SampleResponse { sequences }))
}

fn valid_adapter_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Write the LoRA adapter in PEFT layout (Tinker's
/// `save_weights_and_get_sampling_client`): the returned path loads straight
/// into the engine's inference LoRA loader.
#[utoipa::path(
  post,
  tag = "Hanzo",
  path = "/v1/training/clients/{id}/save_weights",
  request_body = SaveWeightsRequest,
  responses(
    (status = 200, description = "PEFT adapter written", body = SaveWeightsResponse),
    (status = 404, description = "No such training client"),
    (status = 409, description = "Client is still loading or failed to load")
  )
)]
pub async fn training_save_weights(
    Extension(state): Extension<TrainingState>,
    Path(id): Path<String>,
    Json(req): Json<SaveWeightsRequest>,
) -> Result<Json<SaveWeightsResponse>, TrainingError> {
    if !valid_adapter_name(&req.name) {
        return Err(bad_request(
            "name must be non-empty [A-Za-z0-9._-] and not `.` / `..`",
        ));
    }
    let session = state.get(&id)?;
    let parent = match &req.dir {
        Some(dir) => PathBuf::from(dir),
        None => dirs::home_dir()
            .ok_or_else(|| internal("cannot resolve home directory for the default adapter dir"))?
            .join(".cache/hanzo/adapters")
            .join(&id),
    };
    let path = parent.join(&req.name);
    let guard = session.ready_client().await?;
    let path = tokio::task::spawn_blocking(move || {
        let client = guard.as_ref().expect("ready_client guarantees Some");
        client.save_weights_and_get_sampling_client(&path)
    })
    .await
    .map_err(internal)?
    .map_err(internal)?;
    Ok(Json(SaveWeightsResponse {
        path: path.display().to_string(),
        format: "peft".to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_datum_parses_text_and_tokens_forms() {
        let req: ForwardBackwardRequest = serde_json::from_str(
            r#"{"data": [
                {"prompt": "2+2=", "completion": "4"},
                {"model_input": {"tokens": [1, 2, 3]}, "target_tokens": [2, 3, 4], "weights": [0.0, 1.0, 1.0]}
            ]}"#,
        )
        .unwrap();
        assert_eq!(req.data.len(), 2);
        assert!(
            matches!(&req.data[0], WireDatum::Text { prompt, completion } if prompt == "2+2=" && completion == "4")
        );
        match &req.data[1] {
            WireDatum::Tokens(d) => {
                assert_eq!(d.model_input.tokens, vec![1, 2, 3]);
                assert_eq!(d.validate().unwrap(), 3);
            }
            other => panic!("expected tokens form, got {other:?}"),
        }
    }

    #[test]
    fn create_request_fills_lora_defaults() {
        let req: CreateTrainingClientRequest =
            serde_json::from_str(r#"{"base_model": "m"}"#).unwrap();
        assert_eq!(req.lora_config.rank, 16);
        assert_eq!(req.lora_config.alpha, 32.0);
        assert_eq!(req.lora_config.target_modules.len(), 7);

        let req: CreateTrainingClientRequest =
            serde_json::from_str(r#"{"base_model": "m", "lora_config": {"rank": 8}}"#).unwrap();
        assert_eq!(req.lora_config.rank, 8);
        assert_eq!(req.lora_config.alpha, 32.0);
    }

    #[test]
    fn adapter_names_reject_traversal() {
        assert!(valid_adapter_name("my-adapter_v1.2"));
        assert!(!valid_adapter_name(""));
        assert!(!valid_adapter_name("."));
        assert!(!valid_adapter_name(".."));
        assert!(!valid_adapter_name("a/b"));
        assert!(!valid_adapter_name("a\\b"));
    }

    #[tokio::test]
    async fn ops_on_missing_client_return_404() {
        let state = TrainingState::default();
        let err = get_training_client(Extension(state.clone()), Path("tc-none".into()))
            .await
            .unwrap_err();
        assert_eq!(err.0, StatusCode::NOT_FOUND);

        let err = training_optim_step(
            Extension(state),
            Path("tc-none".into()),
            Json(OptimStepRequest {
                adam_params: AdamParams::default(),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(err.0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn create_reports_failed_for_unloadable_local_dir() {
        // An existing directory with no config.json fails the local load fast
        // and offline — exercising create -> background load -> failed status.
        let dir = tempfile::tempdir().unwrap();
        let state = TrainingState::default();
        let info = create_training_client(
            Extension(state.clone()),
            Json(CreateTrainingClientRequest {
                base_model: dir.path().display().to_string(),
                lora_config: LoraConfig::default(),
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(info.status, TrainingClientStatus::Loading);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let detail = get_training_client(Extension(state.clone()), Path(info.id.clone()))
                .await
                .unwrap()
                .0;
            match detail.info.status {
                TrainingClientStatus::Failed => {
                    assert!(detail.info.error.is_some());
                    break;
                }
                _ if std::time::Instant::now() > deadline => {
                    panic!("load did not fail within 30s: {detail:?}")
                }
                _ => tokio::time::sleep(std::time::Duration::from_millis(25)).await,
            }
        }

        // Ops on a failed client are 409, and delete works.
        let err = training_forward_backward(
            Extension(state.clone()),
            Path(info.id.clone()),
            Json(ForwardBackwardRequest {
                data: vec![WireDatum::Text {
                    prompt: "p".into(),
                    completion: "c".into(),
                }],
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(err.0, StatusCode::CONFLICT);

        let deleted = delete_training_client(Extension(state.clone()), Path(info.id.clone()))
            .await
            .unwrap()
            .0;
        assert!(deleted.deleted);
        assert!(state.get(&info.id).is_err());
    }
}
