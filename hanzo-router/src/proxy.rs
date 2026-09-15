//! The effectful front for [`crate::replica`]: an axum reverse proxy that picks a
//! replica per request (session pins + cache hints + capacity + health) and streams the
//! response back unchanged. SSE bodies pass through byte-for-byte; the in-flight
//! [`Lease`] rides the response stream and releases when it ends or the client
//! aborts. A background loop re-probes replica `/health` to evict and restore.
//!
//! Feature-gated (`proxy`) so the pure router core stays dependency-light.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use futures::Stream;
use serde::Deserialize;
use serde_json::Value;

use crate::replica::{Balancer, Lease, Replica, ReplicaSet, ReplicaStatus, RoutingHints};
use sha2::{Digest, Sha256};

const MAX_BODY_BYTES: usize = 32 << 20;
const PREFIX_CHARS: usize = 512;
/// Default background health re-probe cadence (see [`ServeConfig::probe_interval`]).
pub const DEFAULT_PROBE_INTERVAL: Duration = Duration::from_secs(5);
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// Liveness is "the server answers HTTP". Engines' own /health can run a
/// generation through the scheduler (SGLang does), which waits behind a long
/// prefill and would evict the one replica able to serve it.
const HEALTH_PATH: &str = "/v1/models";
/// Response header naming the replica that served the request (observability).
const REPLICA_HEADER: &str = "x-replica-id";
const CONV_HEADERS: [&str; 5] = [
    "x-hanzo-session",
    "x-smg-routing-key",
    "x-hanzo-conversation-id",
    "x-conversation-id",
    "x-session-id",
];
const SKIP_REQUEST_HEADERS: [&str; 5] = [
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
    "keep-alive",
];
const SKIP_RESPONSE_HEADERS: [&str; 6] = [
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "upgrade",
    "trailer",
];

struct ProxyState {
    balancer: Balancer,
    client: reqwest::Client,
    upstream_model: Option<String>,
    pool_file: Option<crate::pool_file::PoolFile>,
}

/// What to serve: the pool + where to bind + how often to re-probe.
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub balancer: Balancer,
    pub probe_interval: Duration,
    /// Rewrite each forwarded request's `model` to this before it reaches the
    /// replica. Engines that serve one model strict-match its id (or `default`)
    /// and 500 on anything else, so a fabric fronting `claude-*`-style clients
    /// pins the upstream id here. `None` forwards the client's `model` unchanged.
    pub upstream_model: Option<String>,
    /// Writable YAML used for durable operator retuning; None disables writes.
    pub config_path: Option<std::path::PathBuf>,
}

fn app(state: Arc<ProxyState>) -> Router {
    Router::new()
        .route(
            "/health",
            get(|| async { "ok" }).head(|| async { StatusCode::OK }),
        )
        .route("/api/hello", get(hello).head(hello))
        .route("/v1/models", get(list_models))
        .route(
            "/v1/replicas",
            get(list_replicas).post(add_replica).put(update_replica),
        )
        .fallback(proxy)
        .with_state(state)
}

/// Serve on an already-bound listener, running the health prober in the
/// background. The seam both [`serve`] and the tests build on.
pub async fn serve_listener(
    listener: tokio::net::TcpListener,
    balancer: Balancer,
    probe_interval: Duration,
    upstream_model: Option<String>,
) -> anyhow::Result<()> {
    serve_listener_config(listener, balancer, probe_interval, upstream_model, None).await
}

async fn serve_listener_config(
    listener: tokio::net::TcpListener,
    balancer: Balancer,
    probe_interval: Duration,
    upstream_model: Option<String>,
    config_path: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    anyhow::ensure!(!probe_interval.is_zero(), "probe interval must be positive");
    let state = Arc::new(ProxyState {
        balancer,
        client: reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(30))
            .tcp_keepalive(Some(Duration::from_secs(15)))
            .tcp_nodelay(true)
            .pool_idle_timeout(Some(Duration::from_secs(300)))
            .build()?,
        upstream_model,
        pool_file: config_path
            .map(crate::pool_file::PoolFile::new)
            .transpose()?,
    });
    tokio::spawn(probe_loop(state.clone(), probe_interval));
    axum::serve(listener, app(state)).await?;
    Ok(())
}

/// Bind `host:port` and serve until shutdown.
pub async fn serve(cfg: ServeConfig) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind((cfg.host.as_str(), cfg.port)).await?;
    tracing::info!("hanzo-router proxy listening on {}:{}", cfg.host, cfg.port);
    serve_listener_config(
        listener,
        cfg.balancer,
        cfg.probe_interval,
        cfg.upstream_model,
        cfg.config_path,
    )
    .await
}

/// Forward any non-admin path to a chosen replica, retrying past replicas that
/// refuse the connection (evicting them so the ring reroutes).
async fn proxy(State(state): State<Arc<ProxyState>>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let path_q = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| parts.uri.path().to_string());
    let body_bytes = match to_bytes(body, MAX_BODY_BYTES).await {
        Ok(b) => b,
        Err(_) => return err(StatusCode::PAYLOAD_TOO_LARGE, "request body too large"),
    };
    let json: Option<Value> = serde_json::from_slice(&body_bytes).ok();
    let model = json
        .as_ref()
        .and_then(|j| j.get("model"))
        .and_then(|v| v.as_str());
    let Some(set) = state.balancer.set_for(model) else {
        return err(StatusCode::NOT_FOUND, "no replica set for model");
    };
    let mut hints = routing_hints(&parts.headers, json.as_ref());
    // Count with the serving tokenizer before admission. Compaction includes
    // system prompts, tool schemas, tool results and the entire conversation;
    // a character estimate alone cannot enforce a near-limit context window.
    if parts.uri.path() == "/v1/messages" {
        if let Some(body) = json.as_ref() {
            if set.statuses().iter().any(|r| r.max_context != 0) {
                hints.token_counts = count_prompt_tokens(&state, &set, &parts.headers, body).await;
            }
        }
    }
    let workers = set.statuses();
    let eligible: Vec<_> = workers
        .iter()
        .filter(|r| hints.target.as_ref().is_none_or(|id| *id == r.id))
        .collect();
    if !eligible.is_empty()
        && eligible
            .iter()
            .all(|r| r.max_context != 0 && r.max_context < hints.required_tokens(&r.id))
    {
        let limit = eligible.iter().map(|r| r.max_context).max().unwrap_or(0);
        return api_error(StatusCode::BAD_REQUEST, "invalid_request_error", &format!(
            "prompt is too long: {} tokens including requested output exceeds the {} token context window of the configured replicas",
            eligible.iter().map(|r| hints.required_tokens(&r.id)).min().unwrap_or(0), limit
        ));
    }
    let wants_stream = json
        .as_ref()
        .and_then(|b| b.get("stream"))
        .and_then(Value::as_bool)
        == Some(true);
    let anthropic = parts.uri.path() == "/v1/messages";
    let dispatch = Box::pin(dispatch(state, parts, path_q, body_bytes, json, set, hints));
    if wants_stream {
        response_with_progress(dispatch, anthropic, Duration::from_secs(15)).await
    } else {
        dispatch.await
    }
}

async fn dispatch(
    state: Arc<ProxyState>,
    parts: axum::http::request::Parts,
    path_q: String,
    body_bytes: Bytes,
    json: Option<Value>,
    set: Arc<ReplicaSet>,
    hints: RoutingHints,
) -> Response {
    let fwd = forward_headers(&parts.headers);
    let mut excluded = HashSet::new();
    for _ in 0..set.len().max(1) {
        let Some(lease) = set.pick_agent(&hints, &excluded, Instant::now()) else {
            break;
        };
        let started = Instant::now();
        let wire_body = match json.as_ref() {
            Some(j) if j.is_object() => {
                let mut rewritten = j.clone();
                if let Some(model) = lease.upstream_model().or(state.upstream_model.as_deref()) {
                    rewritten["model"] = Value::String(model.to_owned());
                }
                normalize_reasoning_effort(&mut rewritten);
                normalize_messages(&mut rewritten);
                Bytes::from(serde_json::to_vec(&rewritten).expect("JSON value"))
            }
            _ => body_bytes.clone(),
        };
        let url = format!("{}{}", lease.url().trim_end_matches('/'), path_q);
        let send = state
            .client
            .request(parts.method.clone(), &url)
            .headers(fwd.clone())
            .body(wire_body)
            .send()
            .await;
        match send {
            Ok(resp) => return stream_response(resp, lease, set.clone(), hints, started),
            Err(e) => {
                let id = lease.id().to_string();
                excluded.insert(id.clone());
                drop(lease);
                tracing::warn!("replica {id} failed ({e}), evicting and trying next replica");
                set.mark_unhealthy(&id);
            }
        }
    }
    let mut response = err(
        StatusCode::SERVICE_UNAVAILABLE,
        "no available replica for routing hints",
    );
    response
        .headers_mut()
        .insert("retry-after", HeaderValue::from_static("1"));
    response
}

/// Some engines do not send HTTP headers until prefill completes. Keep the
/// public connection alive during that wait, not just after upstream headers.
/// Cancellation drops the pending request/lease; delayed errors remain errors.
async fn response_with_progress(
    mut pending: Pin<Box<dyn std::future::Future<Output = Response> + Send>>,
    anthropic: bool,
    interval: Duration,
) -> Response {
    tokio::select! {
        response = &mut pending => return response,
        _ = tokio::time::sleep(interval) => {},
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, axum::Error>>(4);
    tokio::spawn(async move {
        use futures::StreamExt;
        let mut ping = tokio::time::interval(interval);
        let response = loop {
            tokio::select! {
                _ = tx.closed() => return,
                response = &mut pending => break response,
                _ = ping.tick() => {
                    if tx.send(Ok(Bytes::from_static(b": keep-alive\n\n"))).await.is_err() { return; }
                }
            }
        };
        if !response.status().is_success() {
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 64 * 1024)
                .await
                .unwrap_or_default();
            let value = serde_json::from_slice::<Value>(&bytes).ok();
            let error = value.and_then(|v| v.get("error").cloned()).unwrap_or_else(|| serde_json::json!({
                "type": "api_error", "message": String::from_utf8_lossy(&bytes), "status": status.as_u16()
            }));
            let event = serde_json::json!({"type":"error","error":error});
            let wire = if anthropic {
                format!("event: error\ndata: {event}\n\n")
            } else {
                format!("data: {event}\n\n")
            };
            let _ = tx.send(Ok(Bytes::from(wire))).await;
            return;
        }
        let mut body = response.into_body().into_data_stream();
        loop {
            tokio::select! {
                _ = tx.closed() => return,
                chunk = body.next() => match chunk {
                    Some(chunk) => if tx.send(chunk).await.is_err() { return; },
                    None => return,
                }
            }
        }
    });
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    response
        .headers_mut()
        .insert("cache-control", HeaderValue::from_static("no-cache"));
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

/// Wrap the upstream byte stream so the [`Lease`] drops when the body ends.
fn stream_response(
    resp: reqwest::Response,
    lease: Lease,
    set: Arc<ReplicaSet>,
    hints: RoutingHints,
    started: Instant,
) -> Response {
    let status = resp.status();
    let src = resp.headers().clone();
    let replica = HeaderValue::from_str(lease.id()).ok();
    let is_sse = src
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.contains("text/event-stream"))
        .unwrap_or(false);
    let ping_deadline = Box::pin(tokio::time::sleep(Duration::from_secs(15)));
    let body = Body::from_stream(LeasedStream {
        inner: Box::pin(resp.bytes_stream()),
        lease: Some(lease),
        set,
        hints,
        started,
        first_byte: false,
        successful: status.is_success(),
        is_sse,
        ping_deadline,
    });
    let mut response = Response::new(body);
    *response.status_mut() = status;
    let dst = response.headers_mut();
    for (k, v) in src.iter() {
        if !SKIP_RESPONSE_HEADERS.contains(&k.as_str()) {
            dst.append(k.clone(), v.clone());
        }
    }
    if let Some(replica) = replica {
        dst.insert(REPLICA_HEADER, replica);
    }
    response
}

struct LeasedStream {
    inner: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>,
    lease: Option<Lease>,
    set: Arc<ReplicaSet>,
    hints: RoutingHints,
    started: Instant,
    first_byte: bool,
    successful: bool,
    is_sse: bool,
    ping_deadline: Pin<Box<tokio::time::Sleep>>,
}

impl Stream for LeasedStream {
    type Item = reqwest::Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        use std::future::Future;
        let this = self.get_mut();
        let next = this.inner.as_mut().poll_next(cx);
        match next {
            Poll::Ready(Some(Ok(bytes))) => {
                if !bytes.is_empty() && !this.first_byte {
                    this.first_byte = true;
                    if this.successful {
                        if let Some(lease) = &this.lease {
                            this.set.observe_ttft(lease.id(), this.started.elapsed());
                        }
                    }
                }
                this.ping_deadline
                    .as_mut()
                    .reset(tokio::time::Instant::now() + Duration::from_secs(15));
                Poll::Ready(Some(Ok(bytes)))
            }
            Poll::Ready(Some(Err(e))) => {
                this.successful = false;
                this.lease.take();
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                if let Some(lease) = this.lease.take() {
                    if this.successful && this.first_byte {
                        this.set
                            .observe_completion(lease.id(), &this.hints, Instant::now());
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => {
                if this.is_sse && this.ping_deadline.as_mut().poll(cx).is_ready() {
                    this.ping_deadline
                        .as_mut()
                        .reset(tokio::time::Instant::now() + Duration::from_secs(15));
                    Poll::Ready(Some(Ok(Bytes::from_static(b": keep-alive\n\n"))))
                } else {
                    Poll::Pending
                }
            }
        }
    }
}

/// The stable key a request routes on: an explicit conversation/session header if
/// present, else the shared prefix (system + first user turn) that agentic fan-out
/// requests hold in common, else the model id.
pub fn affinity_key(headers: &HeaderMap, body: Option<&Value>) -> String {
    for name in CONV_HEADERS {
        if let Some(v) = headers.get(name).and_then(|v| v.to_str().ok()) {
            if !v.is_empty() {
                return v.to_string();
            }
        }
    }
    let mut s = String::new();
    if let Some(body) = body {
        if let Some(sys) = body.get("system").and_then(|v| v.as_str()) {
            s.push_str(sys);
        }
        if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
            for m in msgs {
                match m.get("role").and_then(|v| v.as_str()) {
                    Some("system") => push_content(&mut s, m.get("content")),
                    Some("user") => {
                        push_content(&mut s, m.get("content"));
                        break;
                    }
                    _ => {}
                }
            }
        }
        if s.is_empty() {
            if let Some(t) = body.get("input").and_then(|v| v.as_str()) {
                s.push_str(t);
            }
        }
        if s.is_empty() {
            if let Some(m) = body.get("model").and_then(|v| v.as_str()) {
                s.push_str(m);
            }
        }
    }
    s.chars().take(PREFIX_CHARS).collect()
}

/// Normalize reasoning effort values (e.g. Claude Code's effort: "high" or "max")
/// to "medium" so backends whose chat templates support xhigh/medium/low (like Qwen)
/// or backends mapping effort onto discrete tiers do not fail with TemplateError.
fn normalize_reasoning_effort(val: &mut Value) {
    if let Some(obj) = val.as_object_mut() {
        if let Some(oc) = obj.get_mut("output_config").and_then(|v| v.as_object_mut()) {
            if let Some(effort) = oc.get("effort").and_then(|v| v.as_str()) {
                if effort == "high" || effort == "max" {
                    oc.insert("effort".to_string(), Value::String("medium".to_string()));
                }
            }
        }
        if let Some(effort) = obj.get("reasoning_effort").and_then(|v| v.as_str()) {
            if effort == "high" || effort == "max" {
                obj.insert(
                    "reasoning_effort".to_string(),
                    Value::String("medium".to_string()),
                );
            }
        }
    }
}

/// Normalize messages to ensure strict chat-template compatibility:
/// Any non-leading system message (e.g. injected developer prompts, tool results,
/// or system reminders mid-conversation) has its role changed to "user" with a "[System]: "
/// prefix so backends with strict Jinja templates (like Qwen) do not reject the turn.
fn normalize_messages(val: &mut Value) {
    if let Some(obj) = val.as_object_mut() {
        if let Some(messages) = obj.get_mut("messages").and_then(|v| v.as_array_mut()) {
            for (idx, msg) in messages.iter_mut().enumerate() {
                if let Some(blocks) = msg.get_mut("content").and_then(Value::as_array_mut) {
                    for block in blocks {
                        tool_reference_to_text(block);
                        if let Some(inner) = block.get_mut("content").and_then(Value::as_array_mut)
                        {
                            inner.iter_mut().for_each(tool_reference_to_text);
                        }
                    }
                }
                if idx > 0 {
                    if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
                        if role == "system" {
                            if let Some(mobj) = msg.as_object_mut() {
                                mobj.insert("role".to_string(), Value::String("user".to_string()));
                                if let Some(content) = mobj.get("content").and_then(|c| c.as_str())
                                {
                                    let new_content = format!("[System]: {}", content);
                                    mobj.insert("content".to_string(), Value::String(new_content));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// ToolSearch results carry `tool_reference` blocks, which only the first-party
/// API expands. Local engines reject them, poisoning every later turn of the
/// conversation, so they become the text the model needs: the tool's name.
fn tool_reference_to_text(block: &mut Value) {
    if block.get("type").and_then(Value::as_str) == Some("tool_reference") {
        let name = block.get("tool_name").and_then(Value::as_str).unwrap_or("");
        *block = serde_json::json!({"type": "text", "text": format!("Tool loaded: {name}")});
    }
}

/// Build fixed-size scoped keys without retaining prompts, credentials, or IDs.
/// Cloud may carry hints in metadata because its metered relay forwards JSON but
/// does not forward arbitrary request headers. Explicit headers take precedence.
fn routing_hints(headers: &HeaderMap, body: Option<&Value>) -> RoutingHints {
    fn digest(value: &Value) -> String {
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(value).expect("JSON value"))
        )
    }
    let header = |key: &str| {
        headers
            .get(key)
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty())
    };
    let meta = body.and_then(|b| b.get("metadata"));
    let hint = |name: &str, key: &str| {
        header(name).or_else(|| {
            meta.and_then(|m| m.get(key))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        })
    };
    // This listener is private to the authenticated cloud gateway. Its org/user
    // headers must come from that gateway; the bearer adds isolation for direct
    // private clients. No identity is ever taken from JSON metadata.
    let scope = serde_json::json!([
        header("x-org-id"),
        header("x-user-id"),
        header("authorization"),
        body.and_then(|b| b.get("model"))
    ]);
    let session = hint("x-session-id", "session_id")
        .or_else(|| CONV_HEADERS.iter().find_map(|h| header(h)))
        .map(|s| digest(&serde_json::json!([scope, "session", s])));
    let prefix = body.and_then(|b| {
        let mut initial = Vec::new();
        if let Some(messages) = b.get("messages").and_then(Value::as_array) {
            for m in messages {
                match m.get("role").and_then(Value::as_str) {
                    Some("system" | "developer") => initial.push(m.clone()),
                    Some("user") => {
                        initial.push(m.clone());
                        break;
                    }
                    _ => break,
                }
            }
        }
        let input = b.get("input");
        if initial.is_empty()
            && input.is_none()
            && b.get("system").is_none()
            && b.get("instructions").is_none()
        {
            return None;
        }
        Some(digest(&serde_json::json!([
            scope,
            "prefix",
            b.get("system"),
            b.get("instructions"),
            b.get("tools"),
            b.get("tool_choice"),
            initial,
            input
        ])))
    });
    let explicit_role = hint("x-agent-role", "agent_role").map(str::to_owned);
    let role = explicit_role.or_else(|| {
        if let Some(m) = body.and_then(|b| b.get("model")).and_then(Value::as_str) {
            let m_lower = m.to_lowercase();
            if m_lower.contains("haiku")
                || m_lower.contains("flash")
                || m_lower.contains("subagent")
                || m_lower.contains("review")
                || m_lower.contains("halo")
            {
                return Some("subagent".to_string());
            }
            if m_lower.contains("opus")
                || m_lower.contains("sonnet")
                || m_lower.contains("coder")
                || m_lower.contains("reason")
            {
                return Some("main".to_string());
            }
        }
        if let Some(qs) = meta
            .and_then(|m| m.get("query_source"))
            .and_then(Value::as_str)
        {
            if qs.contains("title")
                || qs.contains("summary")
                || qs.contains("background")
                || qs.contains("subagent")
            {
                return Some("subagent".to_string());
            }
        }
        None
    });
    RoutingHints {
        session,
        prefix,
        role,
        target: hint("x-target-replica", "target_replica").map(str::to_owned),
        approx_tokens: body.map(estimate_required_tokens).unwrap_or(0),
        token_counts: Default::default(),
    }
}

fn output_tokens(body: &Value) -> usize {
    ["max_completion_tokens", "max_output_tokens", "max_tokens"]
        .iter()
        .find_map(|key| body.get(*key).and_then(Value::as_u64))
        .unwrap_or(4096)
        .try_into()
        .unwrap_or(usize::MAX)
}

/// Scheduling estimate for APIs without an exact counting endpoint.
/// Serialize all prompt-bearing fields, including nested tool results/schemas.
fn estimate_required_tokens(body: &Value) -> usize {
    let bytes = [
        "system",
        "messages",
        "tools",
        "tool_choice",
        "input",
        "instructions",
        "prompt",
    ]
    .iter()
    .filter_map(|key| body.get(*key))
    .fold(0usize, |sum, value| {
        sum.saturating_add(serde_json::to_vec(value).map(|v| v.len()).unwrap_or(0))
    });
    bytes
        .div_ceil(3)
        .saturating_add(256)
        .saturating_add(output_tokens(body))
}

/// Count on each candidate's own tokenizer: pool members can serve different
/// models and chat templates, so counts are not interchangeable. A worker that
/// cannot answer is scheduled by the byte estimate instead.
async fn count_prompt_tokens(
    state: &ProxyState,
    set: &ReplicaSet,
    headers: &HeaderMap,
    body: &Value,
) -> HashMap<String, usize> {
    let workers = set
        .statuses()
        .into_iter()
        .filter(|r| r.healthy && r.max_context != 0);
    let counts = futures::future::join_all(workers.map(|worker| async move {
        let mut request = body.clone();
        if let Some(model) = worker
            .upstream_model
            .as_deref()
            .or(state.upstream_model.as_deref())
        {
            request["model"] = Value::String(model.to_owned());
        }
        normalize_reasoning_effort(&mut request);
        normalize_messages(&mut request);
        let response = state
            .client
            .post(format!(
                "{}/v1/messages/count_tokens",
                worker.url.trim_end_matches('/')
            ))
            .headers(forward_headers(headers))
            .timeout(Duration::from_secs(15))
            .json(&request)
            .send()
            .await
            .ok()
            .filter(|r| r.status().is_success())?;
        let value = response.json::<Value>().await.ok()?;
        let tokens = usize::try_from(value.get("input_tokens")?.as_u64()?).ok()?;
        Some((worker.id, tokens.saturating_add(output_tokens(body))))
    }))
    .await;
    counts.into_iter().flatten().collect()
}

fn api_error(status: StatusCode, kind: &str, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({"type":"error","error":{"type":kind,"message":message}})),
    )
        .into_response()
}

fn push_content(s: &mut String, content: Option<&Value>) {
    match content {
        Some(Value::String(t)) => s.push_str(t),
        Some(Value::Array(blocks)) => {
            for b in blocks {
                if let Some(t) = b.get("text").and_then(|v| v.as_str()) {
                    s.push_str(t);
                }
            }
        }
        _ => {}
    }
}

fn forward_headers(src: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (k, v) in src.iter() {
        if !SKIP_REQUEST_HEADERS.contains(&k.as_str()) {
            out.append(k.clone(), v.clone());
        }
    }
    out
}

fn err(status: StatusCode, msg: &str) -> Response {
    (status, msg.to_string()).into_response()
}

async fn hello() -> StatusCode {
    StatusCode::OK
}

async fn list_models(State(state): State<Arc<ProxyState>>) -> Response {
    let mut model_ids: Vec<String> = state.balancer.statuses().into_keys().collect();
    for primary in &["zen5.8", "zen5.8-coder", "qwen3.8"] {
        if !model_ids.iter().any(|m| m == primary) {
            model_ids.push(primary.to_string());
        }
    }
    model_ids.sort();
    let data: Vec<Value> = model_ids
        .into_iter()
        .map(|id| {
            let context = state
                .balancer
                .set_for(Some(&id))
                .map(|set| {
                    set.statuses()
                        .iter()
                        .map(|r| r.max_context)
                        .max()
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            serde_json::json!({
                "id": id,
                "object": "model",
                "created": 1726272000,
                "owned_by": "hanzo",
                "context_window": context,
                "max_context_length": context
            })
        })
        .collect();
    Json(serde_json::json!({
        "object": "list",
        "data": data
    }))
    .into_response()
}

async fn list_replicas(
    State(state): State<Arc<ProxyState>>,
) -> Json<BTreeMap<String, Vec<ReplicaStatus>>> {
    Json(state.balancer.statuses())
}

#[derive(Deserialize)]
struct RegisterBody {
    model: String,
    url: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    capacity: usize,
    #[serde(default)]
    max_context: usize,
    #[serde(default = "registration_weight")]
    weight: u32,
    #[serde(default)]
    roles: Vec<String>,
    #[serde(default)]
    upstream_model: Option<String>,
}
fn registration_weight() -> u32 {
    100
}

async fn add_replica(
    State(state): State<Arc<ProxyState>>,
    Json(body): Json<RegisterBody>,
) -> Json<BTreeMap<String, Vec<ReplicaStatus>>> {
    state.balancer.register(
        &body.model,
        Replica {
            id: body.id.unwrap_or_default(),
            url: body.url,
            capacity: body.capacity,
            max_context: body.max_context,
            weight: body.weight,
            roles: body.roles,
            upstream_model: body.upstream_model,
        },
    );
    Json(state.balancer.statuses())
}

async fn update_replica(
    State(state): State<Arc<ProxyState>>,
    Json(update): Json<crate::pool_file::WorkerUpdate>,
) -> Response {
    let Some(file) = &state.pool_file else {
        return err(
            StatusCode::CONFLICT,
            "durable worker settings require a config file",
        );
    };
    if let Err(message) = update.validate() {
        return err(StatusCode::BAD_REQUEST, message);
    }
    match file.update(&state.balancer, update).await {
        Ok(()) => Json(state.balancer.statuses()).into_response(),
        Err(message) => err(StatusCode::CONFLICT, &message),
    }
}

async fn probe_loop(state: Arc<ProxyState>, interval: Duration) {
    loop {
        tokio::time::sleep(interval).await;
        let sets: Vec<_> = state
            .balancer
            .models()
            .iter()
            .filter_map(|m| state.balancer.set_for(Some(m)))
            .collect();
        // One probe per upstream per cycle, concurrently: a URL shared by many
        // model pools is one server, and a slow one must not delay the others.
        let urls: HashSet<String> = sets
            .iter()
            .flat_map(|set| set.statuses().into_iter().map(|r| r.url))
            .collect();
        let health: HashMap<String, bool> =
            futures::future::join_all(urls.into_iter().map(|url| async {
                let probe = format!("{}{}", url.trim_end_matches('/'), HEALTH_PATH);
                let ok = matches!(
                    state.client.get(&probe).timeout(PROBE_TIMEOUT).send().await,
                    Ok(r) if r.status().is_success()
                );
                (url, ok)
            }))
            .await
            .into_iter()
            .collect();
        for set in sets {
            for st in set.statuses() {
                if let Some(ok) = health.get(&st.url) {
                    set.set_health(&st.id, *ok);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderName;

    #[tokio::test]
    async fn slow_headers_send_pings_preserve_errors_and_cancel_leases() {
        use futures::StreamExt;
        for mode in ["success", "error", "cancel"] {
            let set = Arc::new(ReplicaSet::new([Replica::new("worker")], 1));
            let lease = set
                .pick_agent(&RoutingHints::default(), &HashSet::new(), Instant::now())
                .unwrap();
            let (release, wait) = tokio::sync::oneshot::channel::<()>();
            let pending = Box::pin(async move {
                let _lease = lease;
                let _ = wait.await;
                if mode == "error" {
                    api_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "prompt is too long",
                    )
                } else {
                    ([("content-type", "text/event-stream")], "data: done\n\n").into_response()
                }
            });
            let response = response_with_progress(pending, true, Duration::from_millis(5)).await;
            assert_eq!(response.headers()["content-type"], "text/event-stream");
            let mut body = response.into_body().into_data_stream();
            assert_eq!(body.next().await.unwrap().unwrap(), ": keep-alive\n\n");
            assert_eq!(set.statuses()[0].inflight, 1);
            if mode == "cancel" {
                drop(body);
            } else {
                release.send(()).unwrap();
                let mut wire = Vec::new();
                while let Some(chunk) = body.next().await {
                    wire.extend_from_slice(&chunk.unwrap());
                }
                let wire = String::from_utf8(wire).unwrap();
                assert!(wire.contains(if mode == "error" {
                    "event: error\ndata:"
                } else {
                    "data: done"
                }));
                if mode == "error" {
                    assert!(wire.contains("prompt is too long"));
                }
            }
            tokio::time::timeout(Duration::from_secs(1), async {
                while set.statuses()[0].inflight != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
        }
    }

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                HeaderName::from_bytes(k.as_bytes()).unwrap(),
                v.parse().unwrap(),
            );
        }
        h
    }

    #[test]
    fn routing_metadata_is_scoped_and_prefix_covers_tools_and_long_context() {
        let mut body = serde_json::json!({
            "model": "zen-coder",
            "metadata": {"session_id": "cc-1", "agent_role": "reviewer"},
            "system": [{"type": "text", "text": "S".repeat(2000)}],
            "tools": [{"name":"read","input_schema":{"type":"object"}}],
            "messages": [{"role":"user","content":"repo A"}]
        });
        let org_a = headers(&[("x-org-id", "a")]);
        let first = routing_hints(&org_a, Some(&body));
        assert_eq!(first.role.as_deref(), Some("reviewer"));
        assert_eq!(first.session.as_ref().unwrap().len(), 64);
        assert_ne!(
            first.session,
            routing_hints(&headers(&[("x-org-id", "b")]), Some(&body)).session
        );
        assert_ne!(
            first.prefix,
            routing_hints(&headers(&[("x-org-id", "b")]), Some(&body)).prefix
        );
        body["messages"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"role":"assistant","content":"done"}));
        assert_eq!(first.prefix, routing_hints(&org_a, Some(&body)).prefix);
        body["tools"][0]["name"] = serde_json::json!("write");
        assert_ne!(first.prefix, routing_hints(&org_a, Some(&body)).prefix);
        body["messages"][0]["content"] = serde_json::json!("repo B");
        assert_ne!(first.prefix, routing_hints(&org_a, Some(&body)).prefix);
        body["model"] = serde_json::json!("other-model");
        assert_ne!(first.session, routing_hints(&org_a, Some(&body)).session);
        let explicit = routing_hints(&headers(&[("x-session-id", "override")]), Some(&body));
        assert_ne!(
            explicit.session,
            routing_hints(&HeaderMap::new(), Some(&body)).session
        );
        assert!(
            routing_hints(&HeaderMap::new(), Some(&serde_json::json!({"model":"m"})))
                .prefix
                .is_none()
        );
    }

    #[tokio::test]
    async fn leased_stream_releases_on_eof_error_and_abort() {
        use futures::StreamExt;
        for mode in ["complete", "abort", "error"] {
            let set = Arc::new(ReplicaSet::new([Replica::new("worker")], 1));
            let hints = RoutingHints {
                session: Some("s".into()),
                prefix: Some("p".into()),
                ..Default::default()
            };
            let lease = set
                .pick_agent(&hints, &HashSet::new(), Instant::now())
                .unwrap();
            let inner: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>> =
                if mode == "error" {
                    // An invalid URI makes a reqwest error without network access.
                    let error = reqwest::Client::new()
                        .get("not a url")
                        .send()
                        .await
                        .unwrap_err();
                    Box::pin(futures::stream::iter(vec![Err(error)]))
                } else {
                    Box::pin(futures::stream::iter(vec![Ok(Bytes::from_static(
                        b"data: [DONE]\n\n",
                    ))]))
                };
            let mut stream = LeasedStream {
                inner,
                lease: Some(lease),
                set: set.clone(),
                hints,
                started: Instant::now(),
                first_byte: false,
                successful: true,
                is_sse: false,
                ping_deadline: Box::pin(tokio::time::sleep(Duration::from_secs(15))),
            };
            assert_eq!(set.statuses()[0].inflight, 1);
            if mode != "abort" {
                while stream.next().await.is_some() {}
            } else {
                drop(stream);
            }
            assert_eq!(set.statuses()[0].inflight, 0, "{mode}");
            assert_eq!(set.statuses()[0].ttft_ewma_ms.is_some(), mode == "complete");
        }
    }

    #[test]
    fn conversation_header_wins() {
        let body = serde_json::json!({"model": "m", "messages": [{"role":"user","content":"hi"}]});
        let k = affinity_key(&headers(&[("x-conversation-id", "conv-7")]), Some(&body));
        assert_eq!(k, "conv-7");
    }

    #[test]
    fn prefix_is_system_plus_first_user_stable_across_turns() {
        let turn1 = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "You are a helpful agent with tools."},
                {"role": "user", "content": "List my files."}
            ]
        });
        let turn2 = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "You are a helpful agent with tools."},
                {"role": "user", "content": "List my files."},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "now delete them"}
            ]
        });
        let h = HeaderMap::new();
        let k1 = affinity_key(&h, Some(&turn1));
        let k2 = affinity_key(&h, Some(&turn2));
        assert_eq!(k1, k2, "same conversation prefix -> same key across turns");
        assert!(k1.contains("helpful agent") && k1.contains("List my files"));
    }

    #[test]
    fn anthropic_system_and_content_blocks() {
        let body = serde_json::json!({
            "model": "m",
            "system": "SYS",
            "messages": [{"role": "user", "content": [{"type":"text","text":"BLOCK"}]}]
        });
        let k = affinity_key(&HeaderMap::new(), Some(&body));
        assert_eq!(k, "SYSBLOCK");
    }

    #[test]
    fn falls_back_to_model_when_no_prefix() {
        let body = serde_json::json!({"model": "only-model"});
        assert_eq!(affinity_key(&HeaderMap::new(), Some(&body)), "only-model");
    }
}

// End-to-end: two mock engines behind the proxy, driven over real HTTP. Proves
// prefix-affinity stickiness, streaming passthrough with in-flight release, and
// probe-driven health eviction + auto-restore. Runs under `--features proxy`.
#[cfg(test)]
mod e2e {
    use super::*;
    use axum::extract::State;
    use axum::http::header::CONTENT_TYPE;
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use axum::Router;
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::net::TcpListener;

    struct Mock {
        id: String,
        alive: AtomicBool,
    }

    async fn mock_health(State(m): State<Arc<Mock>>) -> Response {
        if m.alive.load(Ordering::Acquire) {
            (StatusCode::OK, "ok").into_response()
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "down").into_response()
        }
    }

    async fn mock_chat(State(m): State<Arc<Mock>>) -> Response {
        if !m.alive.load(Ordering::Acquire) {
            return (StatusCode::SERVICE_UNAVAILABLE, "down").into_response();
        }
        let body = format!("data: {{\"replica\":\"{}\"}}\n\ndata: [DONE]\n\n", m.id);
        ([(CONTENT_TYPE, "text/event-stream")], body).into_response()
    }

    async fn spawn_mock(id: &str) -> (String, Arc<Mock>) {
        let mock = Arc::new(Mock {
            id: id.to_string(),
            alive: AtomicBool::new(true),
        });
        let app = Router::new()
            .route("/v1/models", get(mock_health))
            .route("/v1/chat/completions", post(mock_chat))
            .route("/v1/messages", post(mock_chat))
            .route("/v1/messages/count_tokens", post(|Json(body): Json<Value>| async move {
                if body["metadata"]["count_unavailable"] == true {
                    return StatusCode::NOT_FOUND.into_response();
                }
                Json(serde_json::json!({"input_tokens": body["metadata"]["test_tokens"].as_u64().unwrap_or(53)})).into_response()
            }))
            .with_state(mock.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), mock)
    }

    async fn free_url() -> String {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        drop(l);
        format!("http://{addr}")
    }

    async fn start_proxy(balancer: Balancer, interval: Duration) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            serve_listener(listener, balancer, interval, None)
                .await
                .unwrap()
        });
        format!("http://{addr}")
    }

    async fn chat(client: &reqwest::Client, base: &str, conv: &str) -> (StatusCode, String) {
        let resp = client
            .post(format!("{base}/v1/chat/completions"))
            .header("x-conversation-id", conv)
            .json(
                &serde_json::json!({"model": "qwen", "messages": [{"role":"user","content":conv}]}),
            )
            .send()
            .await
            .unwrap();
        let status = resp.status();
        (status, resp.text().await.unwrap())
    }

    fn replica_id(text: &str) -> String {
        let marker = "\"replica\":\"";
        let i = text.find(marker).expect("replica marker") + marker.len();
        let rest = &text[i..];
        rest[..rest.find('"').unwrap()].to_string()
    }

    async fn statuses(client: &reqwest::Client, base: &str) -> serde_json::Value {
        client
            .get(format!("{base}/v1/replicas"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn affinity_streaming_health_and_restore() {
        let (url_a, _a) = spawn_mock("A").await;
        let (url_b, mock_b) = spawn_mock("B").await;
        let balancer = Balancer::new(64);
        balancer.register("qwen", Replica::new(url_a));
        balancer.register("qwen", Replica::new(url_b));
        let base = start_proxy(balancer, Duration::from_millis(120)).await;
        let client = reqwest::Client::new();
        tokio::time::sleep(Duration::from_millis(200)).await;

        // (a) prefix-affinity: each conversation sticks to one replica across turns.
        let convs: Vec<String> = (0..12).map(|i| format!("conv-{i}")).collect();
        let mut home = HashMap::new();
        for c in &convs {
            let (s, t) = chat(&client, &base, c).await;
            assert_eq!(s, StatusCode::OK);
            let r = replica_id(&t);
            for _ in 0..3 {
                let (s, t) = chat(&client, &base, c).await;
                assert_eq!(s, StatusCode::OK);
                assert_eq!(replica_id(&t), r, "{c} must stick to one replica");
            }
            home.insert(c.clone(), r);
        }
        let used: HashSet<_> = home.values().cloned().collect();
        assert!(
            used.contains("A") && used.contains("B"),
            "both used: {home:?}"
        );

        // streaming passthrough released every in-flight slot.
        tokio::time::sleep(Duration::from_millis(50)).await;
        for row in statuses(&client, &base).await["qwen"].as_array().unwrap() {
            assert_eq!(row["inflight"].as_u64().unwrap(), 0, "lease drained: {row}");
        }

        // (b) kill B: the prober evicts it and its conversations reroute to A.
        let b_conv = home
            .iter()
            .find(|(_, v)| v.as_str() == "B")
            .unwrap()
            .0
            .clone();
        mock_b.alive.store(false, Ordering::Release);
        tokio::time::sleep(Duration::from_millis(320)).await;
        let (s, t) = chat(&client, &base, &b_conv).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(replica_id(&t), "A", "B down -> served by survivor A");

        // (c) restart B: existing sessions stay on the replacement cache owner.
        mock_b.alive.store(true, Ordering::Release);
        tokio::time::sleep(Duration::from_millis(320)).await;
        let (s, t) = chat(&client, &base, &b_conv).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(
            replica_id(&t),
            "A",
            "B restored -> existing session keeps its new cache owner"
        );
    }

    #[tokio::test]
    async fn cloud_metadata_routes_roles_and_rejects_unknown_targets() {
        let (url_a, _) = spawn_mock("A").await;
        let (url_b, _) = spawn_mock("B").await;
        let balancer = Balancer::new(2);
        let mut a = Replica::new(url_a);
        a.id = "spark".into();
        a.roles = vec!["main".into()];
        let mut b = Replica::new(url_b);
        b.id = "evo".into();
        b.roles = vec!["reviewer".into()];
        balancer.register("zen-coder", a);
        balancer.register("zen-coder", b);
        let base = start_proxy(balancer, Duration::from_secs(60)).await;
        let client = reqwest::Client::new();
        let mut body = serde_json::json!({"model":"zen-coder", "messages":[{"role":"user","content":"review"}],
            "metadata":{"session_id":"cc-1","agent_role":"reviewer"}});
        for role in ["reviewer", "main"] {
            body["metadata"]["agent_role"] = serde_json::json!(role);
            let resp = client
                .post(format!("{base}/v1/chat/completions"))
                .header("x-org-id", "acme")
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(
                resp.headers()[REPLICA_HEADER],
                "evo",
                "role changes cannot break pins"
            );
            assert_eq!(replica_id(&resp.text().await.unwrap()), "B");
        }
        // The same caller-chosen session ID in another org has no affinity to B.
        let resp = client
            .post(format!("{base}/v1/chat/completions"))
            .header("x-org-id", "other")
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.headers()[REPLICA_HEADER], "spark");
        assert_eq!(replica_id(&resp.text().await.unwrap()), "A");
        body["metadata"]["target_replica"] = serde_json::json!("unknown");
        let resp = client
            .post(format!("{base}/v1/chat/completions"))
            .header("x-org-id", "acme")
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers()["retry-after"], "1");
    }

    #[tokio::test]
    async fn exact_context_admission_routes_long_compaction_and_reserves_output() {
        let (spark_url, _) = spawn_mock("Spark").await;
        let (evo_url, _) = spawn_mock("Evo").await;
        let balancer = Balancer::new(8);
        let mut spark = Replica::new(spark_url);
        spark.id = "spark".into();
        spark.max_context = 1_000_000;
        let mut evo = Replica::new(evo_url);
        evo.id = "evo".into();
        evo.max_context = 65_536;
        evo.roles = vec!["reviewer".into()];
        balancer.register("zen-coder", spark);
        balancer.register("zen-coder", evo);
        // A distinct fallback makes this test catch suffixes losing their pool.
        balancer.register("default", Replica::new("http://127.0.0.1:1"));
        let base = start_proxy(balancer, Duration::from_secs(60)).await;
        let client = reqwest::Client::new();
        let models: Value = client
            .get(format!("{base}/v1/models"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(
            models["data"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"] == "zen-coder")
                .unwrap()["context_window"],
            1_000_000
        );
        let mut body = serde_json::json!({"model":"zen-coder[1m]", "max_tokens":1000,
            "messages":[{"role":"user","content":"summarize our conversation"}],
            "metadata":{"session_id":"compact", "agent_role":"reviewer", "test_tokens":64_536}});
        for (tokens, worker) in [(64_536, "evo"), (176_939, "spark"), (999_000, "spark")] {
            body["metadata"]["test_tokens"] = serde_json::json!(tokens);
            let response = client
                .post(format!("{base}/v1/messages"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()[REPLICA_HEADER], worker);
            let _ = response.bytes().await.unwrap();
        }
        body["metadata"]["test_tokens"] = serde_json::json!(999_001);
        let response = client
            .post(format!("{base}/v1/messages"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let error: Value = response.json().await.unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert!(error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("prompt is too long"));
        // Tokenizers unavailable: the byte estimate still routes, never refuses.
        body["metadata"]["count_unavailable"] = serde_json::json!(true);
        let response = client
            .post(format!("{base}/v1/messages"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let _ = response.bytes().await.unwrap();
        body["metadata"]["count_unavailable"] = serde_json::json!(false);
        body["metadata"]["test_tokens"] = serde_json::json!(176_939);
        body["metadata"]["target_replica"] = serde_json::json!("evo");
        assert_eq!(
            client
                .post(format!("{base}/v1/messages"))
                .json(&body)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[tokio::test]
    async fn dead_replica_evicted_never_breaks_a_request() {
        let (url_a, _a) = spawn_mock("A").await;
        let dead = free_url().await;
        let balancer = Balancer::new(64);
        balancer.register("qwen", Replica::new(url_a));
        let base = start_proxy(balancer, Duration::from_millis(120)).await;
        let client = reqwest::Client::new();
        // dynamic registration of a dead upstream.
        client
            .post(format!("{base}/v1/replicas"))
            .json(&serde_json::json!({"model": "qwen", "url": dead}))
            .send()
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(320)).await;
        for i in 0..20 {
            let (s, _) = chat(&client, &base, &format!("k{i}")).await;
            assert_eq!(s, StatusCode::OK, "dead replica must never break a request");
        }
        let dead_row = statuses(&client, &base).await["qwen"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["url"] == dead)
            .cloned()
            .unwrap();
        assert_eq!(dead_row["healthy"], false, "dead upstream probed unhealthy");
    }

    async fn echo_model(axum::Json(v): axum::Json<serde_json::Value>) -> Response {
        let m = v
            .get("model")
            .and_then(|x| x.as_str())
            .unwrap_or("<none>")
            .to_string();
        let body = format!("data: {{\"model\":\"{m}\"}}\n\ndata: [DONE]\n\n");
        ([(CONTENT_TYPE, "text/event-stream")], body).into_response()
    }

    #[tokio::test]
    async fn upstream_model_is_rewritten() {
        let app = Router::new()
            .route("/v1/models", get(|| async { "ok" }))
            .route("/v1/chat/completions", post(echo_model));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let balancer = Balancer::new(64);
        balancer.register("qwen", Replica::new(format!("http://{addr}")));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let paddr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            serve_listener(
                listener,
                balancer,
                Duration::from_secs(60),
                Some("default".into()),
            )
            .await
            .unwrap()
        });
        let base = format!("http://{paddr}");
        let client = reqwest::Client::new();
        tokio::time::sleep(Duration::from_millis(150)).await;

        let resp = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        let text = resp.text().await.unwrap();
        assert!(
            text.contains("\"model\":\"default\""),
            "client model must be rewritten to `default` upstream, got: {text}"
        );
    }

    #[tokio::test]
    async fn claude_code_hello_and_models_and_fallback() {
        let (url, _) = spawn_mock("A").await;
        let balancer = Balancer::new(2);
        balancer.register("zen5.8", Replica::new(url));
        let base = start_proxy(balancer, Duration::from_secs(60)).await;
        let client = reqwest::Client::new();

        // 1. /api/hello GET and HEAD
        let resp_get = client
            .get(format!("{base}/api/hello"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp_get.status(), StatusCode::OK);

        let resp_head = client
            .head(format!("{base}/api/hello"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp_head.status(), StatusCode::OK);

        // 2. /v1/models GET returns models list including zen5.8, zen5.8-coder, qwen3.8
        let resp_models = client
            .get(format!("{base}/v1/models"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp_models.status(), StatusCode::OK);
        let models_json: Value = resp_models.json().await.unwrap();
        let list = models_json.get("data").and_then(Value::as_array).unwrap();
        assert!(list
            .iter()
            .any(|m| m.get("id").and_then(Value::as_str) == Some("zen5.8")));
        assert!(list
            .iter()
            .any(|m| m.get("id").and_then(Value::as_str) == Some("qwen3.8")));

        // 3. Fallback routing: unrecognized claude model resolves to zen5.8
        let resp_msg = client
            .post(format!("{base}/v1/messages"))
            .json(&serde_json::json!({
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        let status = resp_msg.status();
        let body = resp_msg.text().await.unwrap();
        assert_eq!(status, StatusCode::OK, "fallback response error: {body}");
    }

    #[test]
    fn normalize_reasoning_effort_normalizes_high_and_max() {
        let mut v = serde_json::json!({
            "output_config": {"effort": "high"},
            "reasoning_effort": "max"
        });
        normalize_reasoning_effort(&mut v);
        assert_eq!(v["output_config"]["effort"], "medium");
        assert_eq!(v["reasoning_effort"], "medium");

        let mut v2 = serde_json::json!({
            "output_config": {"effort": "low"},
            "reasoning_effort": "medium"
        });
        normalize_reasoning_effort(&mut v2);
        assert_eq!(v2["output_config"]["effort"], "low");
        assert_eq!(v2["reasoning_effort"], "medium");
    }

    #[test]
    fn normalize_messages_turns_tool_references_into_text() {
        let mut v = serde_json::json!({"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "tool_reference", "tool_name": "Monitor"},
                {"type": "text", "text": "kept"}
            ]},
            {"type": "tool_reference", "tool_name": "WebFetch"}
        ]}]});
        normalize_messages(&mut v);
        let blocks = &v["messages"][0]["content"];
        assert_eq!(
            blocks[0]["content"][0],
            serde_json::json!({"type": "text", "text": "Tool loaded: Monitor"})
        );
        assert_eq!(blocks[0]["content"][1]["text"], "kept");
        assert_eq!(blocks[1]["text"], "Tool loaded: WebFetch");
    }

    #[test]
    fn normalize_messages_rewrites_non_leading_system() {
        let mut v = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "Remember to speak concisely."}
            ]
        });
        normalize_messages(&mut v);
        assert_eq!(v["messages"][0]["role"], "system");
        assert_eq!(v["messages"][0]["content"], "You are a helpful assistant.");
        assert_eq!(v["messages"][1]["role"], "user");
        assert_eq!(v["messages"][2]["role"], "user");
        assert_eq!(
            v["messages"][2]["content"],
            "[System]: Remember to speak concisely."
        );
    }

    #[test]
    fn routing_hints_infers_subagent_and_main_roles() {
        let headers = HeaderMap::new();

        // 1. Haiku model infers subagent
        let body_haiku = serde_json::json!({
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "fast summary"}]
        });
        let hints = routing_hints(&headers, Some(&body_haiku));
        assert_eq!(hints.role.as_deref(), Some("subagent"));

        // 2. Opus model infers main
        let body_opus = serde_json::json!({
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "deep coding"}]
        });
        let hints = routing_hints(&headers, Some(&body_opus));
        assert_eq!(hints.role.as_deref(), Some("main"));

        // 3. query_source generates title infers subagent
        let body_meta = serde_json::json!({
            "model": "zen5.8",
            "metadata": {"query_source": "generate_session_title"}
        });
        let hints = routing_hints(&headers, Some(&body_meta));
        assert_eq!(hints.role.as_deref(), Some("subagent"));

        // 4. Explicit header overrides inference
        let mut custom_headers = HeaderMap::new();
        custom_headers.insert("x-agent-role", HeaderValue::from_static("reviewer"));
        let hints = routing_hints(&custom_headers, Some(&body_opus));
        assert_eq!(hints.role.as_deref(), Some("reviewer"));
    }
}
