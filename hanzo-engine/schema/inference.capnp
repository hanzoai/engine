@0xf1e2d3c4b5a69788;
# ZAP inference RPC for the internal engine<>node link.
# Binary wire replacement for the internal JSON the node sends to the engine's
# OpenAI HTTP API (`POST /ai|/v1/chat/completions`, `GET /v1/models`, `/health`,
# `/v1/system/info`). The external OpenAI/Anthropic HTTP+JSON API is unchanged;
# this is a parallel transport for first-party (node, db) callers.
#
# Field set mirrors `hanzo_engine::Request` / `hanzo_engine::Response` so the
# server side is a thin adapter over the existing `Sender<Request>` channel.

using Go = import "/go.capnp";
$Go.package("zapinfer");

# ---- shared scalars ----

enum Role {
  system @0;
  user @1;
  assistant @2;
  tool @3;
}

# One chat message. `content` is the OpenAI content: either plain text or a JSON
# array of content parts (text/image_url/...). We keep the JSON-array form as
# bytes so multimodal parts survive without re-modeling every OpenAI variant.
struct Message {
  role @0 :Role;
  union {
    text @1 :Text;
    partsJson @2 :Data;   # JSON array of content parts, UTF-8
  }
  name @3 :Text;          # optional tool/function name
  toolCallId @4 :Text;    # optional, for role=tool
}

enum ReasoningEffort {
  low @0;
  medium @1;
  high @2;
}

# Mirrors `SamplingParams` (only the wire-relevant knobs; server fills defaults).
struct SamplingParams {
  temperature @0 :Float64;
  topP @1 :Float64;
  topK @2 :Int64;          # <0 = unset
  minP @3 :Float64;
  maxTokens @4 :UInt32;    # 0 = unset -> model default
  frequencyPenalty @5 :Float32;
  presencePenalty @6 :Float32;
  seed @7 :UInt64;         # 0 = unset
  stopSeqs @8 :List(Text);
}

# Tool definition (OpenAI function tool). Schema kept as JSON bytes.
struct Tool {
  name @0 :Text;
  description @1 :Text;
  parametersJson @2 :Data; # JSON Schema, UTF-8
}

enum ToolChoiceKind {
  auto @0;
  none @1;
  required @2;
  named @3;       # use `toolChoiceName`
}

# Constraint / structured-output (mirrors `Constraint`).
struct Constraint {
  union {
    none @0 :Void;
    regex @1 :Text;
    jsonSchema @2 :Data;   # JSON bytes
    lark @3 :Text;
  }
}

# ---- requests ----

# Mirrors `NormalRequest` (the `Request::Normal` arm). The mpsc `response`
# Sender is replaced by the RPC return / `ResponseStream` callback.
struct ChatRequest {
  modelId @0 :Text;             # maps to NormalRequest.model_id (None if empty)
  messages @1 :List(Message);
  sampling @2 :SamplingParams;
  stream @3 :Bool;
  tools @4 :List(Tool);
  toolChoice @5 :ToolChoiceKind;
  toolChoiceName @6 :Text;
  constraint @7 :Constraint;
  enableThinking @8 :Bool;
  reasoningEffort @9 :ReasoningEffort;
  returnLogprobs @10 :Bool;
  webSearch @11 :Bool;
  enableCodeExecution @12 :Bool;
  maxToolRounds @13 :UInt32;    # 0 = unset
  sessionId @14 :Text;          # agentic session reuse; empty = new
  requestId @15 :UInt64;        # client correlation id (NormalRequest.id)
}

# Mirrors `TokenizationRequest`.
struct TokenizeRequest {
  modelId @0 :Text;
  union {
    messages @1 :List(Message);
    text @2 :Text;
  }
  tools @3 :List(Tool);
  addGenerationPrompt @4 :Bool;
  addSpecialTokens @5 :Bool;
  enableThinking @6 :Bool;
}

# Mirrors `DetokenizationRequest`.
struct DetokenizeRequest {
  modelId @0 :Text;
  tokens @1 :List(UInt32);
  skipSpecialTokens @2 :Bool;
}

# ---- responses ----

struct ToolCall {
  id @0 :Text;
  name @1 :Text;
  argumentsJson @2 :Data;  # JSON string of arguments, UTF-8
}

struct Usage {
  promptTokens @0 :UInt32;
  completionTokens @1 :UInt32;
  totalTokens @2 :UInt32;
  avgTokPerSec @3 :Float32;
  totalTimeSec @4 :Float32;
}

# Maps to `Response::Done` (`ChatCompletionResponse`).
struct ChatResponse {
  id @0 :Text;
  modelId @1 :Text;
  content @2 :Text;
  reasoningContent @3 :Text;
  toolCalls @4 :List(ToolCall);
  finishReason @5 :Text;
  usage @6 :Usage;
  sessionId @7 :Text;
}

# One streamed token chunk. Maps to `Response::Chunk`
# (`ChatCompletionChunkResponse`); `usage`/`finishReason` set on the final chunk.
struct ChatChunk {
  id @0 :Text;
  deltaContent @1 :Text;
  deltaReasoning @2 :Text;
  toolCalls @3 :List(ToolCall);
  finishReason @4 :Text;   # empty until final
  usage @5 :Usage;         # valid only when finishReason set
}

# Error arm. Maps to `Response::{InternalError,ValidationError,ModelError}`.
struct InferError {
  kind @0 :ErrorKind;
  message @1 :Text;

  enum ErrorKind {
    internal @0;
    validation @1;
    model @2;
  }
}

# ---- control / status ----

struct ModelInfo {
  id @0 :Text;
  kind @1 :Text;        # ModelCategory as string (text/vision/...)
  loaded @2 :Bool;
}

struct ModelList {
  models @0 :List(ModelInfo);
}

struct HealthStatus {
  ok @0 :Bool;
  version @1 :Text;
}

# `SystemInfo` is large and rarely on a hot path; ship it as JSON bytes to avoid
# re-modeling the whole diagnostics tree. Reuses the existing serde_json::Serialize.
struct SystemInfo {
  json @0 :Data;        # serde_json of `hanzo_engine::SystemInfo`, UTF-8
}

# ISQ requantization (maps to `Request::ReIsq`). `isqType` is the IsqType name.
struct ReIsqRequest {
  modelId @0 :Text;
  isqType @1 :Text;
}

# ---- streaming callback ----

# Server pushes chunks to the client during a streaming `chat`. One terminal
# event ends the stream: `done` (with the final aggregate) or `error`.
interface ResponseStream {
  chunk @0 (chunk :ChatChunk) -> ();
  done @1 (response :ChatResponse) -> ();
  error @2 (err :InferError) -> ();
}

# ---- main service ----

interface Inference {
  # Non-streaming chat. Mirrors POST /v1/chat/completions with stream=false.
  chat @0 (request :ChatRequest) -> (response :ChatResponse, err :InferError);

  # Streaming chat. Chunks are delivered to `sink`; the call returns when the
  # stream terminates. Mirrors stream=true (today SSE over HTTP).
  chatStream @1 (request :ChatRequest, sink :ResponseStream) -> ();

  tokenize @2 (request :TokenizeRequest) -> (tokens :List(UInt32));
  detokenize @3 (request :DetokenizeRequest) -> (text :Text);

  # Control / status. Mirror GET /v1/models, /health, /v1/system/info, /re_isq.
  listModels @4 () -> (models :ModelList);
  health @5 () -> (status :HealthStatus);
  systemInfo @6 () -> (info :SystemInfo);
  reIsq @7 (request :ReIsqRequest) -> ();
}
