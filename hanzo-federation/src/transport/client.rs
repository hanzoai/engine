//! Worker-side HTTP client.
//!
//! Mirrors `TransportClient` in transport.py: `put_delta`, `get_aggregate`,
//! `end_round`, `healthz`. Auth headers identical.

use anyhow::{anyhow, Context, Result};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use serde_json::Value;
use std::time::Duration;

use super::{HDR_CODEC, HDR_SIG, HDR_TS, HDR_WORKER};
use crate::auth::sign;

#[derive(Debug, Clone)]
pub struct TransportClient {
    /// e.g. `http://spark.lan:8443`. Trailing slashes stripped on use.
    pub coordinator_url: String,
    pub worker_name: String,
    pub secret: Option<String>,
    pub timeout: Duration,
    /// Sent as `X-Zen-Codec` for diagnostics. `None` => header omitted (server
    /// dispatches on the body's per-tensor `codec` field regardless).
    pub codec_hint: Option<String>,
    inner: reqwest::Client,
}

impl TransportClient {
    pub fn new(coordinator_url: impl Into<String>, worker_name: impl Into<String>) -> Self {
        Self::with_secret(coordinator_url, worker_name, None)
    }

    pub fn with_secret(
        coordinator_url: impl Into<String>,
        worker_name: impl Into<String>,
        secret: Option<String>,
    ) -> Self {
        Self {
            coordinator_url: coordinator_url.into(),
            worker_name: worker_name.into(),
            secret,
            timeout: Duration::from_secs(600),
            codec_hint: None,
            inner: reqwest::Client::builder()
                .timeout(Duration::from_secs(600))
                .build()
                .expect("reqwest client builds with defaults"),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        // reqwest's per-call timeout — we set it inline for each request,
        // so the builder copy is unnecessary, but expose it for API symmetry.
        self
    }

    /// Set the `X-Zen-Codec` hint sent on every authenticated request.
    /// Typical values: `"bf16"`, `"bitdelta"`. Diagnostic only — the server
    /// dispatches on the body's per-tensor `codec` field.
    pub fn with_codec_hint(mut self, hint: impl Into<String>) -> Self {
        self.codec_hint = Some(hint.into());
        self
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.coordinator_url.trim_end_matches('/'), path)
    }

    fn auth_headers(&self, method: &str, path: &str, body: &[u8]) -> Result<HeaderMap> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/octet-stream"));
        if let Some(secret) = &self.secret {
            let (sig, ts) = sign(method, path, body, secret, None);
            h.insert(
                HeaderName::from_static(HDR_SIG),
                HeaderValue::from_str(&sig).context("sig is hex")?,
            );
            h.insert(
                HeaderName::from_static(HDR_TS),
                HeaderValue::from_str(&ts.to_string()).context("ts is ascii")?,
            );
            h.insert(
                HeaderName::from_static(HDR_WORKER),
                HeaderValue::from_str(&self.worker_name).context("name is ascii")?,
            );
        }
        if let Some(hint) = &self.codec_hint {
            if let Ok(v) = HeaderValue::from_str(hint) {
                h.insert(HeaderName::from_static(HDR_CODEC), v);
            }
        }
        Ok(h)
    }

    pub async fn healthz(&self) -> Result<Value> {
        let raw = self.request_raw("GET", "/v1/healthz", &[]).await?;
        if raw.is_empty() {
            return Ok(Value::Null);
        }
        Ok(serde_json::from_slice(&raw)?)
    }

    pub async fn topology(&self) -> Result<Value> {
        let raw = self.request_raw("GET", "/v1/topology", &[]).await?;
        Ok(serde_json::from_slice(&raw)?)
    }

    pub async fn metrics(&self) -> Result<Value> {
        let raw = self.request_raw("GET", "/v1/metrics", &[]).await?;
        Ok(serde_json::from_slice(&raw)?)
    }

    pub async fn put_delta(&self, round_id: u64, blob: Vec<u8>) -> Result<()> {
        let path = format!("/v1/round/{round_id}/worker/{}", self.worker_name);
        self.request_raw("PUT", &path, &blob).await?;
        Ok(())
    }

    pub async fn get_aggregate(&self, round_id: u64) -> Result<Vec<u8>> {
        let path = format!("/v1/round/{round_id}/aggregate");
        self.request_raw("GET", &path, &[]).await
    }

    pub async fn end_round(&self, round_id: u64, loss: f64, step: i64) -> Result<()> {
        let body = serde_json::to_vec(&serde_json::json!({"loss": loss, "step": step}))?;
        let path = format!("/v1/round/{round_id}/end");
        self.request_raw("POST", &path, &body).await?;
        Ok(())
    }

    async fn request_raw(&self, method: &str, path: &str, body: &[u8]) -> Result<Vec<u8>> {
        let url = self.url(path);
        let headers = self.auth_headers(method, path, body)?;
        let req = match method {
            "GET" => self.inner.get(&url),
            "POST" => self.inner.post(&url).body(body.to_vec()),
            "PUT" => self.inner.put(&url).body(body.to_vec()),
            "DELETE" => self.inner.delete(&url),
            other => return Err(anyhow!("unsupported method {other}")),
        };
        let resp = req
            .headers(headers)
            .timeout(self.timeout)
            .send()
            .await
            .with_context(|| format!("{method} {url}"))?;
        let status = resp.status();
        let bytes = resp.bytes().await.context("read response body")?;
        if !status.is_success() {
            return Err(anyhow!(
                "{method} {url} → HTTP {status}: {}",
                String::from_utf8_lossy(&bytes)
            ));
        }
        Ok(bytes.to_vec())
    }
}
