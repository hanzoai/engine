//! > **hanzo server core**
//!
//! ## About
//!
//! This crate powers hanzo server. It exposes the underlying functionality
//! allowing others to implement and extend the server implementation.
//!
//! ### Features
//! 1. Incorporate hanzo server into another axum.rs project.
//! 2. Hook into the hanzo server lifecycle.
//!
//! ### Example
//! ```no_run
//! use std::sync::Arc;
//!
//! use axum::{
//!     extract::State,
//!     routing::{get, post},
//!     Json, Router,
//! };
//! use utoipa::OpenApi;
//! use utoipa_swagger_ui::SwaggerUi;
//!
//! use hanzo_engine::{
//!     initialize_logging, AutoDeviceMapParams, ChatCompletionChunkResponse, ModelDType, ModelSelected,
//! };
//! use hanzo_server_core::{
//!     chat_completion::{
//!         create_streamer, handle_error, parse_request, process_non_streaming_response,
//!         ChatCompletionOnChunkCallback, ChatCompletionOnDoneCallback, ChatCompletionResponder,
//!     },
//!     handler_core::{create_response_channel, send_request},
//!     server::ServerBuilder,
//!     router::RouterBuilder,
//!     openai::ChatCompletionRequest,
//!     openapi_doc::get_openapi_doc,
//!     types::SharedState,
//! };
//!
//! #[derive(OpenApi)]
//! #[openapi(
//!     paths(root, custom_chat),
//!     tags(
//!         (name = "hello", description = "Hello world endpoints")
//!     ),
//!     info(
//!         title = "Hello World API",
//!         version = "1.0.0",
//!         description = "A simple API that responds with a greeting"
//!     )
//! )]
//! struct ApiDoc;
//!
//! #[derive(Clone)]
//! pub struct AppState {
//!     pub hanzo_state: SharedState,
//!     pub db_create: fn(),
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     initialize_logging();
//!
//!     let plain_model_id = String::from("meta-llama/Llama-3.2-1B-Instruct");
//!     let tokenizer_json = None;
//!     let arch = None;
//!     let organization = None;
//!     let write_uqff = None;
//!     let from_uqff = None;
//!     let imatrix = None;
//!     let calibration_file = None;
//!     let hf_cache_path = None;
//!
//!     let dtype = ModelDType::Auto;
//!     let topology = None;
//!     let max_seq_len = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN;
//!     let max_batch_size = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE;
//!     let matformer_config_path = None;
//!     let matformer_slice_name = None;
//!
//!     let model = ModelSelected::Plain {
//!         model_id: plain_model_id,
//!         tokenizer_json,
//!         arch,
//!         dtype,
//!         topology,
//!         organization,
//!         write_uqff,
//!         from_uqff,
//!         imatrix,
//!         calibration_file,
//!         max_seq_len,
//!         max_batch_size,
//!         hf_cache_path,
//!         matformer_config_path,
//!         matformer_slice_name,
//!     };
//!
//!     let shared_hanzo = ServerBuilder::new()
//!         .with_model(model)
//!         .with_in_situ_quant("8".to_string())
//!         .set_paged_attn(Some(true))
//!         .build()
//!         .await
//!         .unwrap();
//!
//!     let hanzo_base_path = "/api/mistral";
//!
//!     let hanzo_routes = RouterBuilder::new()
//!         .with_hanzo(shared_hanzo.clone())
//!         .with_include_swagger_routes(false)
//!         .with_base_path(hanzo_base_path)
//!         .build()
//!         .await
//!         .unwrap();
//!
//!     let hanzo_doc = get_openapi_doc(Some(hanzo_base_path));
//!     let mut api_docs = ApiDoc::openapi();
//!     api_docs.merge(hanzo_doc);
//!
//!     let app_state = Arc::new(AppState {
//!         hanzo_state: shared_hanzo,
//!         db_create: mock_db_call,
//!     });
//!
//!     let app = Router::new()
//!         .route("/", get(root))
//!         .route("/chat", post(custom_chat))
//!         .with_state(app_state.clone())
//!         .nest(hanzo_base_path, hanzo_routes)
//!         .merge(SwaggerUi::new("/api-docs").url("/api-docs/openapi.json", api_docs));
//!
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
//!     axum::serve(listener, app).await.unwrap();
//!
//!     println!("Listening on 0.0.0.0:3000");
//! }
//!
//! #[utoipa::path(
//!     get,
//!     path = "/",
//!     tag = "hello",
//!     responses(
//!         (status = 200, description = "Successful response with greeting message", body = String)
//!     )
//! )]
//! async fn root() -> &'static str {
//!     "Hello, World!"
//! }
//!
//! #[utoipa::path(
//!     post,
//!     tag = "Custom",
//!     path = "/chat",
//!     request_body = ChatCompletionRequest,
//!     responses((status = 200, description = "Chat completions"))
//! )]
//! pub async fn custom_chat(
//!     State(state): State<Arc<AppState>>,
//!     Json(oai_request): Json<ChatCompletionRequest>,
//! ) -> ChatCompletionResponder {
//!     let hanzo_state = state.hanzo_state.clone();
//!     let (tx, mut rx) = create_response_channel(None);
//!
//!     let (request, is_streaming) =
//!         match parse_request(oai_request, hanzo_state.clone(), tx, None, None, None).await {
//!             Ok(x) => x,
//!             Err(e) => return handle_error(hanzo_state, e.into()),
//!         };
//!
//!     dbg!(request.clone());
//!
//!     if let Err(e) = send_request(&hanzo_state, request).await {
//!         return handle_error(hanzo_state, e.into());
//!     }
//!
//!     if is_streaming {
//!         let db_fn = state.db_create;
//!
//!         let on_chunk: ChatCompletionOnChunkCallback =
//!             Box::new(move |mut chunk: ChatCompletionChunkResponse| {
//!                 dbg!(&chunk);
//!
//!                 if let Some(original_content) = &chunk.choices[0].delta.content {
//!                     chunk.choices[0].delta.content = Some(format!("CHANGED! {}", original_content));
//!                 }
//!
//!                 chunk.clone()
//!             });
//!
//!         let on_done: ChatCompletionOnDoneCallback =
//!             Box::new(move |chunks: &[ChatCompletionChunkResponse]| {
//!                 dbg!(chunks);
//!                 (db_fn)();
//!             });
//!
//!         let streamer = create_streamer(rx, hanzo_state.clone(), Some(on_chunk), Some(on_done));
//!
//!         ChatCompletionResponder::Sse(streamer)
//!     } else {
//!         let response = process_non_streaming_response(&mut rx, hanzo_state.clone()).await;
//!
//!         match &response {
//!             ChatCompletionResponder::Json(json_response) => {
//!                 dbg!(json_response);
//!                 (state.db_create)();
//!             }
//!             _ => {
//!                 //
//!             }
//!         }
//!
//!         response
//!     }
//! }
//!
//! pub fn mock_db_call() {
//!     println!("Saving to DB");
//! }
//! ```

pub mod animate;
pub mod anthropic;
pub mod approvals;
pub mod background_tasks;
pub mod cached_responses;
pub mod chat_completion;
mod completion_core;
pub mod completions;
pub mod embeddings;
pub mod files;
pub mod handler_core;
mod handlers;
pub mod image_generation;
pub mod model_registry;
pub mod openai;
pub mod openapi_doc;
pub mod responses;
pub mod responses_types;
pub mod route;
pub mod route_registry;
pub mod router;
pub mod server;
pub mod speech_generation;
pub mod streaming;
pub mod threed_generation;
pub mod types;
pub mod util;
pub mod video;
pub mod video_generation;
