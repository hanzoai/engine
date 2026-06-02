//! Native ZAP listener for Hanzo Engine.
//!
//! Accepts ZAP binary protocol connections and forwards cloud service
//! requests to the local axum HTTP API.
//!
//! Flow: node --ZAP:{zap_port}--> engine --HTTP--> axum handler

use hanzo_zap::{cloud_handler, ZapServer};
use tracing::{error, info, warn};

/// Start the native ZAP listener for the engine.
pub async fn start_zap_server(listen_addr: &str, oai_port: u16) {
    info!("Starting engine ZAP server on {}", listen_addr);

    let server = ZapServer::new("hanzo-engine", listen_addr);

    // Shared HTTP client — reuse connection pool across requests
    let client = reqwest::Client::builder()
        .build()
        .expect("Failed to build HTTP client");

    let handler = cloud_handler(move |method, auth, body| {
        let method = method.clone();
        let auth = auth.clone();
        let body = body.clone();
        let client = client.clone();
        async move {
            let path = match method.as_str() {
                "chat.completions" => "/v1/chat/completions",
                "messages" => "/v1/messages",
                "models" => "/v1/models",
                other => {
                    warn!("ZAP: unknown method: {}", other);
                    return Ok((404, Vec::new(), format!("unknown method: {other}")));
                }
            };

            let url = format!("http://127.0.0.1:{}{}", oai_port, path);
            let is_get = method == "models";

            let mut req = if is_get {
                client.get(&url)
            } else {
                client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .body(body)
            };

            if !auth.is_empty() {
                req = req.header("Authorization", &auth);
            }

            let resp = req
                .send()
                .await
                .map_err(|e| format!("forward error: {e}"))?;
            let status = resp.status().as_u16() as u32;
            let resp_body = resp.bytes().await.map_err(|e| format!("body error: {e}"))?;

            Ok((status, resp_body.to_vec(), String::new()))
        }
    });

    if let Err(e) = server.serve(handler).await {
        error!("Engine ZAP server error: {}", e);
    }
}
