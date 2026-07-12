//! The effectful front for [`crate::replica`]: an axum reverse proxy that picks a
//! replica per request (prefix-affinity + least-loaded + health) and streams the
//! response back unchanged. SSE bodies pass through byte-for-byte; the in-flight
//! [`Lease`] rides the response stream and releases when it ends or the client
//! aborts. A background loop re-probes replica `/health` to evict and restore.
//!
//! Feature-gated (`proxy`) so the pure router core stays dependency-light.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};
use std::time::Duration;

use axum::body::{to_bytes, Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use futures::Stream;
use serde::Deserialize;
use serde_json::Value;

use crate::replica::{Balancer, Lease, Replica, ReplicaStatus};

const MAX_BODY_BYTES: usize = 32 << 20;
const PREFIX_CHARS: usize = 512;
/// Default background health re-probe cadence (see [`ServeConfig::probe_interval`]).
pub const DEFAULT_PROBE_INTERVAL: Duration = Duration::from_secs(5);
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const HEALTH_PATH: &str = "/health";
/// Response header naming the replica that served the request (observability).
const REPLICA_HEADER: &str = "x-hanzo-replica";
const CONV_HEADERS: [&str; 3] = [
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
}

fn app(state: Arc<ProxyState>) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/replicas", get(list_replicas).post(add_replica))
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
    let state = Arc::new(ProxyState {
        balancer,
        client: reqwest::Client::builder().build()?,
        upstream_model,
    });
    tokio::spawn(probe_loop(state.clone(), probe_interval));
    axum::serve(listener, app(state)).await?;
    Ok(())
}

/// Bind `host:port` and serve until shutdown.
pub async fn serve(cfg: ServeConfig) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind((cfg.host.as_str(), cfg.port)).await?;
    tracing::info!("hanzo-router proxy listening on {}:{}", cfg.host, cfg.port);
    serve_listener(listener, cfg.balancer, cfg.probe_interval, cfg.upstream_model).await
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
    let key = affinity_key(&parts.headers, json.as_ref());
    let body_bytes = match (state.upstream_model.as_deref(), json) {
        (Some(um), Some(mut j)) => {
            j["model"] = Value::String(um.to_string());
            serde_json::to_vec(&j).map(Bytes::from).unwrap_or(body_bytes)
        }
        _ => body_bytes,
    };
    let fwd = forward_headers(&parts.headers);
    for _ in 0..set.len().max(1) {
        let Some(lease) = set.pick(&key) else { break };
        let url = format!("{}{}", lease.url().trim_end_matches('/'), path_q);
        let send = state
            .client
            .request(parts.method.clone(), &url)
            .headers(fwd.clone())
            .body(body_bytes.clone())
            .send()
            .await;
        match send {
            Ok(resp) => return stream_response(resp, lease),
            Err(e) if e.is_connect() || e.is_timeout() => {
                let id = lease.id().to_string();
                drop(lease);
                tracing::warn!("replica {id} unreachable ({e}), evicting");
                set.mark_unhealthy(&id);
            }
            Err(e) => return err(StatusCode::BAD_GATEWAY, &format!("upstream error: {e}")),
        }
    }
    err(StatusCode::SERVICE_UNAVAILABLE, "no healthy replica")
}

/// Wrap the upstream byte stream so the [`Lease`] drops when the body ends.
fn stream_response(resp: reqwest::Response, lease: Lease) -> Response {
    let status = resp.status();
    let src = resp.headers().clone();
    let replica = HeaderValue::from_str(lease.id()).ok();
    let body = Body::from_stream(LeasedStream {
        inner: Box::pin(resp.bytes_stream()),
        _lease: lease,
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
    _lease: Lease,
}

impl Stream for LeasedStream {
    type Item = reqwest::Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next(cx)
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
        },
    );
    Json(state.balancer.statuses())
}

async fn probe_loop(state: Arc<ProxyState>, interval: Duration) {
    loop {
        tokio::time::sleep(interval).await;
        for model in state.balancer.models() {
            let Some(set) = state.balancer.set_for(Some(&model)) else {
                continue;
            };
            for st in set.statuses() {
                let url = format!("{}{}", st.url.trim_end_matches('/'), HEALTH_PATH);
                let ok = matches!(
                    state.client.get(&url).timeout(PROBE_TIMEOUT).send().await,
                    Ok(r) if r.status().is_success()
                );
                set.set_health(&st.id, ok);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderName;

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
            .route("/health", get(mock_health))
            .route("/v1/chat/completions", post(mock_chat))
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
                &serde_json::json!({"model": "qwen", "messages": [{"role":"user","content":"hi"}]}),
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

        // (c) restart B: the prober restores it and it reclaims its conversation.
        mock_b.alive.store(true, Ordering::Release);
        tokio::time::sleep(Duration::from_millis(320)).await;
        let (s, t) = chat(&client, &base, &b_conv).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(
            replica_id(&t),
            "B",
            "B restored -> reclaims its conversation"
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
            .route("/health", get(|| async { "ok" }))
            .route("/v1/chat/completions", post(echo_model));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let balancer = Balancer::new(64);
        balancer.register("qwen", Replica::new(format!("http://{addr}")));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let paddr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            serve_listener(listener, balancer, Duration::from_secs(60), Some("default".into()))
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
}
