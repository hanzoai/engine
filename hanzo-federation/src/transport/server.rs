//! axum-based HTTP server. Endpoints match transport.py:
//!
//! Public (no auth):
//!   GET  /                          — dashboard HTML
//!   GET  /v1/healthz                — {"ok": true}
//!   GET  /v1/topology               — topology view
//!   GET  /v1/metrics                — full metrics
//!
//! Authed (HMAC-SHA256 over `method|path|ts|sha256(body)`):
//!   PUT  /v1/round/{rid}/worker/{name}      — push delta
//!   GET  /v1/round/{rid}/aggregate          — pull consensus delta
//!   POST /v1/round/{rid}/end                — report loss/step

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::{header, HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Json, Router,
};
use std::net::SocketAddr;
use std::sync::Arc;

use crate::auth::{verify, DEFAULT_MAX_SKEW_SECS};
use crate::coordinator::{CoordinatorState, EndRoundPayload};

use super::{HDR_CODEC, HDR_SIG, HDR_TS, HDR_WORKER, PUBLIC_PATHS};

#[derive(Clone)]
struct AppState {
    inner: Arc<CoordinatorState>,
}

/// Run the coordinator HTTP server forever.
pub async fn serve(state: CoordinatorState, bind: SocketAddr) -> anyhow::Result<()> {
    let app_state = AppState {
        inner: Arc::new(state),
    };
    let app = Router::new()
        .route("/", get(dashboard))
        .route("/v1/healthz", get(healthz))
        .route("/v1/topology", get(topology))
        .route("/v1/metrics", get(metrics))
        .route("/v1/round/:rid/worker/:name", put(put_delta))
        .route("/v1/round/:rid/aggregate", get(get_aggregate))
        .route("/v1/round/:rid/end", post(post_end_round))
        .with_state(app_state);

    tracing::info!(%bind, "coordinator listening");
    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

// ── handlers ────────────────────────────────────────────────────────────────

async fn dashboard() -> Response {
    let html = dashboard_html();
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(html.into())
        .unwrap()
}

async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({"ok": true}))
}

async fn topology(State(s): State<AppState>) -> impl IntoResponse {
    Json(s.inner.topology_view())
}

async fn metrics(State(s): State<AppState>) -> impl IntoResponse {
    Json(s.inner.metrics())
}

async fn put_delta(
    State(s): State<AppState>,
    Path((rid, name)): Path<(u64, String)>,
    method_uri: ExtractMethodUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(r) = check_auth(&s, &method_uri.method, &method_uri.uri, &body, &headers, Some(&name))
    {
        return r;
    }
    if let Some(codec) = headers.get(HDR_CODEC).and_then(|v| v.to_str().ok()) {
        // Diagnostic only — body's per-tensor `codec` field is authoritative.
        tracing::debug!(round = rid, worker = %name, codec = %codec, bytes = body.len(), "delta in");
    }
    match s.inner.put_delta(rid, &name, body.to_vec()) {
        Ok(()) => Json(serde_json::json!({"ok": true})).into_response(),
        Err(e) => internal_error(&e),
    }
}

async fn get_aggregate(
    State(s): State<AppState>,
    Path(rid): Path<u64>,
    method_uri: ExtractMethodUri,
    headers: HeaderMap,
) -> Response {
    if let Err(r) = check_auth(&s, &method_uri.method, &method_uri.uri, &[], &headers, None) {
        return r;
    }
    match s.inner.get_aggregate(rid).await {
        Ok(blob) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header(header::CONTENT_LENGTH, blob.len())
            .body(blob.into())
            .unwrap(),
        Err(e) => internal_error(&e),
    }
}

async fn post_end_round(
    State(s): State<AppState>,
    Path(rid): Path<u64>,
    method_uri: ExtractMethodUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let worker = match check_auth(&s, &method_uri.method, &method_uri.uri, &body, &headers, None) {
        Ok(w) => w,
        Err(r) => return r,
    };
    let payload: EndRoundPayload = if body.is_empty() {
        EndRoundPayload {
            loss: None,
            step: None,
        }
    } else {
        match serde_json::from_slice(&body) {
            Ok(p) => p,
            Err(e) => return bad_request(&format!("bad end-round payload: {e}")),
        }
    };
    let who = worker.unwrap_or_else(|| "anon".to_string());
    s.inner.end_round(rid, &who, payload.loss, payload.step);
    Json(serde_json::json!({"ok": true})).into_response()
}

// ── auth helper ─────────────────────────────────────────────────────────────

/// Returns Ok(Some(worker)) when auth check succeeded, Ok(None) when running
/// in dev mode (no secrets), or Err(Response) which is the response to emit.
fn check_auth(
    s: &AppState,
    method: &Method,
    uri: &Uri,
    body: &[u8],
    headers: &HeaderMap,
    expected_worker: Option<&str>,
) -> Result<Option<String>, Response> {
    // Public paths bypass.
    let path = uri.path();
    if PUBLIC_PATHS.iter().any(|p| *p == path) {
        return Ok(None);
    }
    let secrets = s.inner.secrets();
    if secrets.is_empty() {
        // Dev mode: still echo X-Zen-Worker if present.
        let who = headers
            .get(HDR_WORKER)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        return Ok(Some(who.unwrap_or_else(|| "anon".to_string())));
    }
    let worker = headers
        .get(HDR_WORKER)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let sig = headers
        .get(HDR_SIG)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let ts: i64 = headers
        .get(HDR_TS)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let Some(secret) = secrets.get(worker) else {
        return Err(unauthorized());
    };
    if !verify(
        method.as_str(),
        path,
        body,
        secret,
        sig,
        ts,
        DEFAULT_MAX_SKEW_SECS,
    ) {
        return Err(unauthorized());
    }
    // For PUT /worker/{name} — the URL name must match the auth identity.
    if let Some(expected) = expected_worker {
        if expected != worker {
            return Err(forbidden());
        }
    }
    Ok(Some(worker.to_string()))
}

// ── tiny extractor that gives us both Method and Uri ────────────────────────

struct ExtractMethodUri {
    method: Method,
    uri: Uri,
}

// axum 0.8 dropped `#[axum::async_trait]`; native async-in-trait is used now.
impl<S> axum::extract::FromRequestParts<S> for ExtractMethodUri
where
    S: Send + Sync,
{
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        Ok(Self {
            method: parts.method.clone(),
            uri: parts.uri.clone(),
        })
    }
}

// ── responses ───────────────────────────────────────────────────────────────

fn unauthorized() -> Response {
    Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .body("unauthorized".into())
        .unwrap()
}
fn forbidden() -> Response {
    Response::builder()
        .status(StatusCode::FORBIDDEN)
        .body("forbidden".into())
        .unwrap()
}
fn internal_error(e: &anyhow::Error) -> Response {
    tracing::error!(error = %e, "internal error");
    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .body(format!("{e}").into())
        .unwrap()
}
fn bad_request(msg: &str) -> Response {
    Response::builder()
        .status(StatusCode::BAD_REQUEST)
        .body(msg.to_owned().into())
        .unwrap()
}

// ── dashboard HTML — ported verbatim from _dashboard_html() in transport.py ─

fn dashboard_html() -> &'static str {
    DASHBOARD_HTML
}

const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><title>zen lab</title>
<style>
  body { font: 13px/1.5 -apple-system, ui-monospace, Menlo, monospace;
         background: #0a0a0a; color: #e8e8e8; padding: 24px; max-width: 1000px; margin: auto; }
  h1 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
  h2 { font-size: 13px; margin: 24px 0 8px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0; }
  th, td { text-align: left; padding: 6px 12px; border-bottom: 1px solid #1a1a1a; }
  th { color: #888; font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  .bar { display: inline-block; height: 8px; background: #2da44e; border-radius: 2px; vertical-align: middle; }
  .ok { color: #2da44e; } .warn { color: #d29922; } .err { color: #f85149; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; padding: 1px 6px; border-radius: 8px; background: #1a1a1a; font-size: 11px; color: #888; }
  .pill.cuda { background: #1c5b1f; color: #7ed688; }
  .pill.rocm { background: #5b1c1c; color: #e88; }
  .pill.mlx  { background: #1c365b; color: #7eaad6; }
  .pill.mps  { background: #3a3a3a; color: #aaa; }
  small { color: #555; }
</style></head><body>
<h1>zen lab</h1>
<small id="updated">loading...</small>
<h2>workers</h2>
<table id="workers"><thead><tr>
  <th>name</th><th>host</th><th>backend</th><th class="num">mem</th>
  <th class="num">data %</th><th>experts pinned</th>
</tr></thead><tbody></tbody></table>
<h2>recent rounds</h2>
<table id="rounds"><thead><tr>
  <th class="num">round</th><th>workers received</th><th class="num">losses</th>
  <th class="num">duration</th><th>status</th>
</tr></thead><tbody></tbody></table>
<script>
async function tick() {
  try {
    const m = await (await fetch('/v1/metrics')).json();
    const t = m.topology;
    document.getElementById('updated').textContent = new Date().toLocaleTimeString() + ' — current round ' + m.current_round;
    const wb = document.querySelector('#workers tbody');
    wb.innerHTML = '';
    for (const w of t.workers) {
      const dw = t.data_weights[w.name] || 0;
      const pins = (t.expert_pins[w.name] || []).join(', ') || '<small>auto</small>';
      wb.insertAdjacentHTML('beforeend',
        `<tr><td>${w.name}</td><td><small>${w.host}</small></td>
         <td><span class="pill ${w.backend}">${w.backend}</span></td>
         <td class="num">${w.memory_gb} GB</td>
         <td class="num">${(dw*100).toFixed(1)}%</td>
         <td>${pins}</td></tr>`);
    }
    const rb = document.querySelector('#rounds tbody');
    rb.innerHTML = '';
    for (const r of m.rounds.slice(-15).reverse()) {
      const losses = Object.values(r.losses);
      const mean = losses.length ? (losses.reduce((a,b)=>a+b,0)/losses.length).toFixed(4) : '—';
      const status = r.aggregated ? '<span class="ok">aggregated</span>' :
                     `<span class="warn">${r.received.length}/${r.expected.length}</span>`;
      const dur = r.duration_s ? r.duration_s.toFixed(1) + 's' : '—';
      rb.insertAdjacentHTML('beforeend',
        `<tr><td class="num">${r.round_id}</td>
         <td><small>${r.received.join(', ')}</small></td>
         <td class="num">${mean}</td><td class="num">${dur}</td>
         <td>${status}</td></tr>`);
    }
  } catch (e) {
    document.getElementById('updated').innerHTML = '<span class="err">error: ' + e.message + '</span>';
  }
}
tick(); setInterval(tick, 2000);
</script></body></html>
"#;
