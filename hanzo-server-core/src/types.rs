//! ## Custom types used in hanzo server core.

use std::sync::Arc;

use axum::extract::State;
use hanzo_engine::{Hanzo, Pipeline};

/// This is the underlying instance of hanzo.
pub type SharedState = Arc<Hanzo>;

/// This is the `SharedState` that has been extracted for an axum handler.
pub type ExtractedState = State<SharedState>;

pub(crate) type LoadedPipeline = Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>;

/// A callback function that processes streaming response chunks before they are sent to the client.
///
/// This hook allows modification of each chunk in the streaming response, enabling features like
/// content filtering, transformation, or logging. The callback receives a chunk and must return
/// a (potentially modified) chunk.
///
/// ### Examples
///
/// ```no_run
/// use hanzo_engine::ChatCompletionChunkResponse;
/// use hanzo_server_core::types::OnChunkCallback;
///
/// let on_chunk: OnChunkCallback<ChatCompletionChunkResponse> = Box::new(|mut chunk| {
///     // Log the chunk or modify its content
///     println!("Processing chunk: {:?}", chunk);
///     chunk
/// });
/// ```
pub type OnChunkCallback<R> = Box<dyn Fn(R) -> R + Send + Sync>;

/// A callback function that is executed when the streaming response completes.
///
/// This hook receives all chunks that were streamed during the response, allowing for
/// post-processing, analytics, or cleanup operations after the stream finishes.
///
/// ### Examples
///
/// ```no_run
/// use hanzo_engine::ChatCompletionChunkResponse;
/// use hanzo_server_core::types::OnDoneCallback;
///
/// let on_done: OnDoneCallback<ChatCompletionChunkResponse> = Box::new(|chunks| {
///     println!("Stream completed with {} chunks", chunks.len());
///     // Process all chunks for analytics
/// });
/// ```
pub type OnDoneCallback<R> = Box<dyn Fn(&[R]) + Send + Sync>;
