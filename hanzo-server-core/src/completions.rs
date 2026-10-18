//! ## Completions functionality and route handler.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use crate::{
    completion_core::{
        convert_stop_tokens, error_response, get_dry_sampling_params, handle_completion_error,
        handle_request_error, model_error, model_error_response, BaseCompletionResponder,
    },
    controls,
    defaults::Defaults,
    error::ApiError,
    handler_core::{
        base_process_non_streaming_response, create_response_channel, send_request,
        ModelErrorMessage,
    },
    openai::{CompletionRequest, Grammar},
    shape::{self, Framing},
    streaming::{base_create_streamer, error_event, keep_alive, BaseStreamer, DoneState},
    types::{ExtractedState, OnChunkCallback, OnDoneCallback, SharedState},
    util::validate_model_name,
};
use anyhow::Result;
use axum::{
    extract::{rejection::JsonRejection, Json, State},
    http::{self, StatusCode},
    response::{
        sse::{Event, KeepAliveStream},
        IntoResponse, Sse,
    },
    Extension,
};
use hanzo_engine::{
    CompletionChunkResponse, CompletionResponse, Constraint, Hanzo, NormalRequest, Request,
    RequestMessage, Response, SamplingParams,
};
use tokio::sync::mpsc::{Receiver, Sender};

/// A callback function that processes streaming response chunks before they are sent to the client.
///
/// This hook allows modification of each chunk in the streaming response, enabling features like
/// content filtering, transformation, or logging. The callback receives a chunk and must return
/// a (potentially modified) chunk.
///
/// ### Examples
///
/// ```no_run
/// use hanzo_server_core::completions::CompletionOnChunkCallback;
///
/// let on_chunk: CompletionOnChunkCallback = Box::new(|mut chunk| {
///     // Log the chunk or modify its content
///     println!("Processing chunk: {:?}", chunk);
///     chunk
/// });
/// ```
pub type CompletionOnChunkCallback = OnChunkCallback<CompletionChunkResponse>;

/// A callback function that is executed when the streaming response completes.
///
/// This hook receives all chunks that were streamed during the response, allowing for
/// post-processing, analytics, or cleanup operations after the stream finishes.
///
/// ### Examples
///
/// ```no_run
/// use hanzo_server_core::completions::CompletionOnDoneCallback;
///
/// let on_done: CompletionOnDoneCallback = Box::new(|chunks| {
///     println!("Stream completed with {} chunks", chunks.len());
///     // Process all chunks for analytics
/// });
/// ```
pub type CompletionOnDoneCallback = OnDoneCallback<CompletionChunkResponse>;

/// A streaming response handler.
///
/// It processes incoming response chunks from a model and converts them
/// into Server-Sent Events (SSE) format for real-time streaming to clients.
pub type CompletionStreamer =
    BaseStreamer<CompletionChunkResponse, CompletionOnChunkCallback, CompletionOnDoneCallback>;

impl futures::Stream for CompletionStreamer {
    type Item = Result<Event, axum::Error>;

    /// Polls the stream for the next Server-Sent Event.
    ///
    /// This method implements the core streaming logic:
    /// 1. Handles stream completion by sending `[DONE]` and executing callbacks
    /// 2. Processes incoming model responses and converts them to SSE events
    /// 3. Applies chunk modifications if a callback is provided
    /// 4. Stores chunks if completion callback is configured
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.done_state {
            DoneState::SendingDone => {
                // The usage chunk (`include_usage`) goes last, just before `[DONE]`.
                if let Some(tail) = self.tail.take() {
                    return Poll::Ready(Some(Event::default().json_data(tail)));
                }
                // https://platform.openai.com/docs/api-reference/completions/create
                // If true, returns a stream of events that happen during the Run as server-sent events, terminating when the Run enters a terminal state with a data: [DONE] message.
                self.done_state = DoneState::Done;
                return Poll::Ready(Some(Ok(Event::default().data("[DONE]"))));
            }
            DoneState::Done => {
                if let Some(on_done) = &self.on_done {
                    on_done(&self.chunks);
                }
                return Poll::Ready(None);
            }
            DoneState::Running => (),
        }

        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(resp)) => match resp {
                Response::CompletionModelError(msg, _) => {
                    Hanzo::maybe_log_error(self.state.clone(), &ModelErrorMessage(msg.to_string()));
                    // Done now, just need to send the [DONE]
                    self.done_state = DoneState::SendingDone;
                    Poll::Ready(Some(Ok(error_event(model_error(&msg)))))
                }
                Response::ValidationError(e) => {
                    self.done_state = DoneState::SendingDone;
                    Poll::Ready(Some(Ok(error_event(ApiError::from_error(
                        e.as_ref(),
                        StatusCode::BAD_REQUEST,
                    )))))
                }
                Response::InternalError(e) => {
                    Hanzo::maybe_log_error(self.state.clone(), &*e);
                    self.done_state = DoneState::SendingDone;
                    Poll::Ready(Some(Ok(error_event(ApiError::from_error(
                        e.as_ref(),
                        StatusCode::INTERNAL_SERVER_ERROR,
                    )))))
                }
                Response::CompletionChunk(mut response) => {
                    if response.choices.iter().all(|x| x.finish_reason.is_some()) {
                        self.done_state = DoneState::SendingDone;
                    }
                    // Done now, just need to send the [DONE]
                    Hanzo::maybe_log_response(self.state.clone(), &response);

                    let this = &mut *self;
                    if let Some(on_chunk) = &this.on_chunk {
                        response = on_chunk(response);
                    }

                    if this.store_chunks {
                        this.chunks.push(response.clone());
                    }

                    let (frame, tail) =
                        shape::chunk(&response, response.usage.as_ref(), &this.framing);
                    if tail.is_some() {
                        this.tail = tail;
                    }
                    Poll::Ready(Some(Event::default().json_data(frame)))
                }
                Response::AgenticToolCallProgress { .. }
                | Response::AgenticToolApprovalRequired { .. }
                | Response::File(_) => {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
                Response::Done(_) => unreachable!(),
                Response::CompletionDone(_) => unreachable!(),
                Response::Chunk(_) => unreachable!(),
                Response::ImageGeneration(_) => unreachable!(),
                Response::ModelError(_, _) => unreachable!(),
                Response::Speech { .. } => unreachable!(),
                Response::Frames { .. } => unreachable!(),
                Response::Transcription { .. } => unreachable!(),
                Response::Raw { .. } => unreachable!(),
                Response::Embeddings { .. } => unreachable!(),
            },
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Represents different types of completion responses.
pub type CompletionResponder =
    BaseCompletionResponder<CompletionResponse, KeepAliveStream<CompletionStreamer>>;

impl IntoResponse for CompletionResponder {
    /// Converts the completion responder into an HTTP response. Errors are OpenAI's envelope; a
    /// finished body carries usage details and `timings`.
    fn into_response(self) -> axum::response::Response {
        match self {
            CompletionResponder::Sse(s) => s.into_response(),
            CompletionResponder::Json(s) => Json(shape::body(&s, &s.usage)).into_response(),
            CompletionResponder::InternalError(e) => {
                error_response(e.as_ref(), http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            CompletionResponder::ValidationError(e) => {
                error_response(e.as_ref(), http::StatusCode::BAD_REQUEST)
            }
            CompletionResponder::ModelError(msg, response) => {
                model_error_response(&msg, shape::body(&response, &response.usage))
            }
        }
    }
}

impl CompletionResponder {
    /// Gives a finished body the served model id.
    pub fn framed(self, framing: &Framing) -> Self {
        let frame = |mut response: CompletionResponse| {
            if let Some(model) = &framing.model {
                response.model = model.clone();
            }
            response
        };
        match self {
            Self::Json(response) => Self::Json(frame(response)),
            Self::ModelError(msg, response) => Self::ModelError(msg, frame(response)),
            other => other,
        }
    }
}

/// Parses and validates a completion request.
///
/// This function transforms an OpenAI-compatible completion request into the
/// request format used by hanzo. The budget, sampling, drafter and `logit_bias` rules are the
/// chat route's; the default budget is `--completion-max-tokens`.
pub fn parse_request(
    oairequest: CompletionRequest,
    state: Arc<Hanzo>,
    defaults: &Defaults,
    tx: Sender<Response>,
) -> Result<(Request, bool)> {
    let repr = serde_json::to_string(&oairequest).expect("Serialization of request failed.");
    Hanzo::maybe_log_request(state.clone(), repr);

    // Validate that the requested model matches the loaded model
    validate_model_name(&oairequest.model, state.clone())
        .map_err(|e| ApiError::new(StatusCode::NOT_FOUND, e.to_string()))?;
    let model_id = (oairequest.model != "default").then(|| oairequest.model.clone());

    let max_tokens = controls::fold_budget(&[
        ("max_tokens", oairequest.max_tokens),
        ("max_completion_tokens", oairequest.max_completion_tokens),
    ])?;
    let sampling = controls::Sampling {
        temperature: oairequest.temperature,
        top_p: oairequest.top_p,
        top_k: oairequest.top_k,
        min_p: oairequest.min_p,
        presence_penalty: oairequest.presence_penalty,
        frequency_penalty: oairequest.frequency_penalty,
    }
    .resolve(defaults)?;
    let config = state
        .config(model_id.as_deref())
        .map_err(anyhow::Error::msg)?;
    let serial = controls::drafter(
        oairequest.drafter.as_deref(),
        config.drafter.as_ref().map(|drafter| drafter.name()),
    )?;
    let logits_bias = controls::logit_bias(oairequest.logit_bias.as_ref(), config.vocab_size)?;
    let max_len = controls::resolve_budget(
        max_tokens,
        defaults.completion_max_tokens,
        defaults.max_tokens_cap,
    )?;

    let stop_toks = convert_stop_tokens(oairequest.stop_seqs);

    let is_streaming = oairequest.stream.unwrap_or(false);

    let dry_params = get_dry_sampling_params(
        oairequest.dry_multiplier,
        oairequest.dry_sequence_breakers,
        oairequest.dry_base,
        oairequest.dry_allowed_length,
    )?;

    Ok((
        Request::Normal(Box::new(NormalRequest {
            id: state.next_request_id(),
            messages: RequestMessage::Completion {
                text: oairequest.prompt,
                echo_prompt: oairequest.echo_prompt,
                best_of: oairequest.best_of,
            },
            sampling_params: SamplingParams {
                temperature: sampling.temperature,
                top_k: sampling.top_k.map(|k| k as usize),
                top_p: sampling.top_p,
                min_p: sampling.min_p,
                top_n_logprobs: oairequest.logprobs.unwrap_or(1),
                frequency_penalty: sampling.frequency_penalty,
                presence_penalty: sampling.presence_penalty,
                repetition_penalty: oairequest.repetition_penalty,
                max_len: Some(max_len),
                stop_toks,
                logits_bias,
                n_choices: oairequest.n_choices,
                dry_params,
                seed: oairequest.seed.map(|seed| seed as u64),
                thinking_budget: None,
                serial,
            },
            response: tx,
            return_logprobs: oairequest.logprobs.is_some(),
            is_streaming,
            suffix: oairequest.suffix,
            constraint: match oairequest.grammar {
                Some(Grammar::Regex(regex)) => Constraint::Regex(regex),
                Some(Grammar::Lark(lark)) => Constraint::Lark(lark),
                Some(Grammar::JsonSchema(schema)) => Constraint::JsonSchema(schema),
                Some(Grammar::Llguidance(llguidance)) => Constraint::Llguidance(llguidance),
                None => Constraint::None,
            },
            tool_choice: oairequest.tool_choice,
            tools: oairequest.tools,
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
            model_id,
            truncate_sequence: oairequest.truncate_sequence.unwrap_or(false),
            session_id: None,
            files: None,
        })),
        is_streaming,
    ))
}

/// OpenAI-compatible completions endpoint handler.
#[utoipa::path(
    post,
    tag = "Hanzo",
    path = "/v1/completions",
    request_body = CompletionRequest,
    responses((status = 200, description = "Completions"))
)]
pub async fn completions(
    State(state): ExtractedState,
    Extension(defaults): Extension<Defaults>,
    body: Result<Json<CompletionRequest>, JsonRejection>,
) -> CompletionResponder {
    let oairequest = match body {
        Ok(Json(oairequest)) => oairequest,
        Err(rejection) => {
            return CompletionResponder::ValidationError(Box::new(ApiError::from(rejection)))
        }
    };
    let (tx, mut rx) = create_response_channel(None);
    let framing = Framing {
        model: defaults.served_name.clone(),
        include_usage: oairequest
            .stream_options
            .as_ref()
            .and_then(|o| o.include_usage)
            .unwrap_or(false),
        single_call: false,
    };

    let (request, is_streaming) = match parse_request(oairequest, state.clone(), &defaults, tx) {
        Ok(x) => x,
        Err(e) => return handle_request_error(state, e),
    };

    if let Err(e) = send_request(&state, request).await {
        return handle_error(state, e.into());
    }

    if is_streaming {
        CompletionResponder::Sse(create_streamer(rx, state, None, None, framing))
    } else {
        process_non_streaming_response(&mut rx, state)
            .await
            .framed(&framing)
    }
}

/// Handle route / generation errors and logging them.
pub fn handle_error(
    state: SharedState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> CompletionResponder {
    handle_completion_error(state, e)
}

/// Creates a SSE streamer for completions with optional callbacks. `framing` sets the served
/// model id and where usage goes (`include_usage`).
pub fn create_streamer(
    rx: Receiver<Response>,
    state: SharedState,
    on_chunk: Option<CompletionOnChunkCallback>,
    on_done: Option<CompletionOnDoneCallback>,
    framing: Framing,
) -> Sse<KeepAliveStream<CompletionStreamer>> {
    let streamer = base_create_streamer(rx, state, on_chunk, on_done, framing);
    Sse::new(streamer).keep_alive(keep_alive())
}

/// Process non-streaming completion responses.
pub async fn process_non_streaming_response(
    rx: &mut Receiver<Response>,
    state: SharedState,
) -> CompletionResponder {
    base_process_non_streaming_response(rx, state, match_responses, handle_error).await
}

/// Matches and processes different types of model responses into appropriate completion responses.
pub fn match_responses(state: SharedState, response: Response) -> CompletionResponder {
    match response {
        Response::InternalError(e) => {
            Hanzo::maybe_log_error(state, &*e);
            CompletionResponder::InternalError(e)
        }
        Response::CompletionModelError(msg, response) => {
            Hanzo::maybe_log_error(state.clone(), &ModelErrorMessage(msg.to_string()));
            Hanzo::maybe_log_response(state, &response);
            CompletionResponder::ModelError(msg, response)
        }
        Response::ValidationError(e) => CompletionResponder::ValidationError(e),
        Response::CompletionDone(response) => {
            Hanzo::maybe_log_response(state, &response);
            CompletionResponder::Json(response)
        }
        Response::CompletionChunk(_) => unreachable!(),
        Response::Chunk(_) => unreachable!(),
        Response::Done(_) => unreachable!(),
        Response::ModelError(_, _) => unreachable!(),
        Response::ImageGeneration(_) => unreachable!(),
        Response::Speech { .. } => unreachable!(),
        Response::Frames { .. } => unreachable!(),
        Response::Transcription { .. } => unreachable!(),
        Response::Raw { .. } => unreachable!(),
        Response::Embeddings { .. } => unreachable!(),
        Response::AgenticToolCallProgress { .. } => unreachable!(),
        Response::AgenticToolApprovalRequired { .. } => unreachable!(),
        Response::File(_) => unreachable!(),
    }
}
