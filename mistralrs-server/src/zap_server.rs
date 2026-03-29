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

    let handler = cloud_handler(move |method, auth, body| {
        let method = method.clone();
        let auth = auth.clone();
        let body = body.clone();
        async move {
            match method.as_str() {
                "chat.completions" => {
                    let client = reqwest::Client::builder()
                        .use_rustls_tls()
                        .build()
                        .map_err(|e| format!("HTTP client error: {e}"))?;

                    let url = format!("http://127.0.0.1:{}/v1/chat/completions", oai_port);
                    let mut req = client
                        .post(&url)
                        .header("Content-Type", "application/json")
                        .body(body);

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
                _ => {
                    warn!("ZAP: unknown method: {}", method);
                    Ok((404, Vec::new(), format!("unknown method: {method}")))
                }
            }
        }
    });

    if let Err(e) = server.serve(handler).await {
        error!("Engine ZAP server error: {}", e);
    }
}
