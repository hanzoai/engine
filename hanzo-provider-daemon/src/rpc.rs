//! Pure Rust `std::net::TcpListener` HTTP/1.1 JSON RPC server & loopback client.
//!
//! Exposes a self-contained permissionless provider surface over real TCP sockets
//! with zero external HTTP crates and strict header/body DoS caps:
//! - `GET  /health`
//! - `GET  /v1/provider/manifest`
//! - `POST /v1/provider/quote`
//! - `POST /v1/provider/jobs`
//! - `POST /v1/provider/verify-receipt`
//! - `GET  /v1/provider/ledger/export`
//! - `POST /v1/provider/drain`

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::daemon::{
    now_unix_ms, DaemonError, ProviderDaemon, QuoteRequest, SignedJobLease, SignedQuote,
};
use crate::json::{parse_json, JsonValue};
use crate::ledger::CompletionReceipt;

/// Maximum permitted HTTP request header section (16 KiB).
pub const MAX_HTTP_HEADER_BYTES: usize = 16 * 1024;
/// Maximum permitted HTTP request body (256 KiB).
pub const MAX_HTTP_BODY_BYTES: usize = 256 * 1024;
/// Socket read/write timeout to prevent slowloris resource exhaustion.
pub const SOCKET_IO_TIMEOUT: Duration = Duration::from_secs(5);

/// Running multi-threaded HTTP/1.1 provider daemon server over a real TCP socket.
pub struct RpcServer {
    addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    accept_thread: Option<JoinHandle<()>>,
    daemon: Arc<ProviderDaemon>,
}

impl RpcServer {
    /// Binds to `bind_addr` (e.g. `"127.0.0.1:0"` for ephemeral loopback port)
    /// and starts serving concurrent HTTP/1.1 connections.
    pub fn start(bind_addr: &str, daemon: Arc<ProviderDaemon>) -> Result<Self, String> {
        let listener =
            TcpListener::bind(bind_addr).map_err(|e| format!("failed to bind {bind_addr}: {e}"))?;
        let addr = listener
            .local_addr()
            .map_err(|e| format!("failed to read local_addr: {e}"))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("failed to set nonblocking: {e}"))?;

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_bg = Arc::clone(&shutdown);
        let daemon_bg = Arc::clone(&daemon);

        let accept_thread = thread::Builder::new()
            .name(format!("provider-rpc-{}", addr.port()))
            .spawn(move || {
                let mut workers: Vec<JoinHandle<()>> = Vec::new();
                while !shutdown_bg.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((stream, _peer)) => {
                            let d = Arc::clone(&daemon_bg);
                            if let Ok(h) = thread::Builder::new()
                                .name("provider-rpc-conn".into())
                                .spawn(move || handle_connection(stream, d))
                            {
                                workers.push(h);
                            }
                            // Reap finished worker threads periodically so handles never leak.
                            if workers.len() >= 32 {
                                workers.retain(|h| !h.is_finished());
                            }
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(1));
                        }
                        Err(_) => break,
                    }
                }
                for h in workers {
                    let _ = h.join();
                }
            })
            .map_err(|e| format!("failed to spawn accept thread: {e}"))?;

        Ok(Self {
            addr,
            shutdown,
            accept_thread: Some(accept_thread),
            daemon,
        })
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn endpoint_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn daemon(&self) -> &Arc<ProviderDaemon> {
        &self.daemon
    }

    /// Gracefully stops accepting new connections and waits for active handlers to finish.
    pub fn stop(mut self) {
        self.shutdown_inner();
    }

    fn shutdown_inner(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        // Wake the accept loop promptly if needed.
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(50));
        if let Some(handle) = self.accept_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for RpcServer {
    fn drop(&mut self) {
        self.shutdown_inner();
    }
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn handle_connection(stream: TcpStream, daemon: Arc<ProviderDaemon>) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(SOCKET_IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(SOCKET_IO_TIMEOUT));

    let Ok(mut writer) = stream.try_clone() else {
        return;
    };
    let Ok(mut drain_reader) = stream.try_clone() else {
        return;
    };
    let req = match read_http_request(stream) {
        Ok(r) => r,
        Err((code, reason, msg)) => {
            let _ = write_json_error(&mut writer, code, reason, &msg);
            graceful_close(writer, &mut drain_reader);
            return;
        }
    };

    match dispatch(&req, &daemon) {
        Ok((code, reason, content_type, body)) => {
            let _ = write_http_response(&mut writer, code, reason, content_type, &body);
            let _ = writer.shutdown(std::net::Shutdown::Write);
        }
        Err(err) => {
            let (code, reason) = err.http_status();
            let _ = write_json_error(&mut writer, code, reason, &err.to_string());
            let _ = writer.shutdown(std::net::Shutdown::Write);
        }
    }
}

/// Half-closes the write side so the peer sees EOF after reading the error response,
/// then drains a strictly bounded number of already-queued bytes briefly to prevent
/// Linux from emitting an immediate TCP RST that discards the 413/431 response.
fn graceful_close(writer: TcpStream, reader: &mut TcpStream) {
    let _ = writer.shutdown(std::net::Shutdown::Write);
    let _ = reader.set_read_timeout(Some(Duration::from_millis(50)));
    let mut scratch = [0u8; 4096];
    let mut drained = 0usize;
    let max_drain = MAX_HTTP_BODY_BYTES.saturating_mul(2);
    while drained < max_drain {
        match reader.read(&mut scratch) {
            Ok(0) | Err(_) => break,
            Ok(n) => drained += n,
        }
    }
}

fn read_http_request(stream: TcpStream) -> Result<HttpRequest, (u16, &'static str, String)> {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    let n = reader
        .read_line(&mut request_line)
        .map_err(|e| (400, "Bad Request", format!("read request line failed: {e}")))?;
    if n == 0 {
        return Err((400, "Bad Request", "empty request".into()));
    }
    if request_line.len() > MAX_HTTP_HEADER_BYTES {
        return Err((
            431,
            "Request Header Fields Too Large",
            "request line too large".into(),
        ));
    }
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err((400, "Bad Request", "malformed HTTP request line".into()));
    }
    let method = parts[0].to_owned();
    let path = parts[1].to_owned();

    let mut header_bytes = request_line.len();
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .map_err(|e| (400, "Bad Request", format!("read header failed: {e}")))?;
        if read == 0 {
            break;
        }
        header_bytes = header_bytes.saturating_add(read);
        if header_bytes > MAX_HTTP_HEADER_BYTES {
            return Err((
                431,
                "Request Header Fields Too Large",
                "HTTP headers exceed 16 KiB limit".into(),
            ));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| (400, "Bad Request", "invalid Content-Length".into()))?;
                if content_length > MAX_HTTP_BODY_BYTES {
                    return Err((
                        413,
                        "Payload Too Large",
                        format!(
                            "Content-Length {content_length} exceeds limit {MAX_HTTP_BODY_BYTES}"
                        ),
                    ));
                }
            }
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| (400, "Bad Request", format!("incomplete request body: {e}")))?;
    }

    Ok(HttpRequest { method, path, body })
}

fn dispatch(
    req: &HttpRequest,
    daemon: &ProviderDaemon,
) -> Result<(u16, &'static str, &'static str, Vec<u8>), DaemonError> {
    let now = now_unix_ms();
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            let cap = daemon.capacity_snapshot();
            let ledger = daemon.ledger_summary();
            let mut m = BTreeMap::new();
            m.insert(
                "status".into(),
                JsonValue::Str(if cap.draining { "draining" } else { "ok" }.into()),
            );
            m.insert(
                "provider_id".into(),
                JsonValue::Str(daemon.provider_id().into()),
            );
            m.insert(
                "active_jobs".into(),
                JsonValue::Uint(cap.active_jobs as u64),
            );
            m.insert("used_vram_mib".into(), JsonValue::Uint(cap.used_vram_mib));
            m.insert("total_vram_mib".into(), JsonValue::Uint(cap.total_vram_mib));
            m.insert("ledger_entries".into(), JsonValue::Uint(ledger.entry_count));
            m.insert(
                "hanzo_control_plane_calls".into(),
                JsonValue::Uint(daemon.hanzo_control_plane_calls()),
            );
            Ok((
                200,
                "OK",
                "application/json",
                JsonValue::Object(m).to_compact_string().into_bytes(),
            ))
        }
        ("GET", "/v1/provider/manifest") => {
            let ann = daemon.signed_announcement(now);
            Ok((
                200,
                "OK",
                "application/json",
                ann.to_json().to_compact_string().into_bytes(),
            ))
        }
        ("POST", "/v1/provider/quote") => {
            let body_str = std::str::from_utf8(&req.body)
                .map_err(|_| DaemonError::BadRequest("request body is not valid UTF-8".into()))?;
            let json = parse_json(body_str).map_err(|e| DaemonError::BadRequest(e.to_string()))?;
            let qreq = QuoteRequest::from_json(&json)
                .map_err(|e| DaemonError::BadRequest(e.to_string()))?;
            let quote = daemon.issue_quote(&qreq, now)?;
            Ok((
                200,
                "OK",
                "application/json",
                quote.to_json().to_compact_string().into_bytes(),
            ))
        }
        ("POST", "/v1/provider/jobs") => {
            let body_str = std::str::from_utf8(&req.body)
                .map_err(|_| DaemonError::BadRequest("request body is not valid UTF-8".into()))?;
            let json = parse_json(body_str).map_err(|e| DaemonError::BadRequest(e.to_string()))?;
            let quote_v = json
                .get("quote")
                .ok_or_else(|| DaemonError::BadRequest("missing `quote` field".into()))?;
            let lease_v = json
                .get("lease")
                .ok_or_else(|| DaemonError::BadRequest("missing `lease` field".into()))?;
            let quote = SignedQuote::from_json(quote_v)
                .map_err(|e| DaemonError::BadRequest(e.to_string()))?;
            let lease = SignedJobLease::from_json(lease_v)
                .map_err(|e| DaemonError::BadRequest(e.to_string()))?;
            let res = daemon.execute_job(&quote, &lease, now)?;
            Ok((
                200,
                "OK",
                "application/json",
                res.to_json().to_compact_string().into_bytes(),
            ))
        }
        ("POST", "/v1/provider/verify-receipt") => {
            let body_str = std::str::from_utf8(&req.body)
                .map_err(|_| DaemonError::BadRequest("request body is not valid UTF-8".into()))?;
            let json = parse_json(body_str).map_err(|e| DaemonError::BadRequest(e.to_string()))?;
            let receipt = CompletionReceipt::from_json(&json)
                .map_err(|e| DaemonError::BadRequest(e.to_string()))?;
            let mut m = BTreeMap::new();
            match receipt.verify() {
                Ok(()) => {
                    m.insert("valid".into(), JsonValue::Bool(true));
                    Ok((
                        200,
                        "OK",
                        "application/json",
                        JsonValue::Object(m).to_compact_string().into_bytes(),
                    ))
                }
                Err(reason) => {
                    m.insert("valid".into(), JsonValue::Bool(false));
                    m.insert("error".into(), JsonValue::Str(reason));
                    Ok((
                        422,
                        "Unprocessable Entity",
                        "application/json",
                        JsonValue::Object(m).to_compact_string().into_bytes(),
                    ))
                }
            }
        }
        ("GET", "/v1/provider/ledger/export") => {
            let jsonl = daemon.export_ledger_jsonl();
            Ok((200, "OK", "application/x-ndjson", jsonl.into_bytes()))
        }
        ("POST", "/v1/provider/drain") => {
            daemon.initiate_drain();
            let mut m = BTreeMap::new();
            m.insert("draining".into(), JsonValue::Bool(true));
            Ok((
                200,
                "OK",
                "application/json",
                JsonValue::Object(m).to_compact_string().into_bytes(),
            ))
        }
        _ => Err(DaemonError::BadRequest(format!(
            "unknown route {} {}",
            req.method, req.path
        ))),
    }
}

fn write_json_error(
    stream: &mut TcpStream,
    code: u16,
    reason: &str,
    message: &str,
) -> std::io::Result<()> {
    let mut m = BTreeMap::new();
    m.insert("error".into(), JsonValue::Str(message.to_owned()));
    m.insert("status".into(), JsonValue::Uint(code as u64));
    let body = JsonValue::Object(m).to_compact_string();
    write_http_response(stream, code, reason, "application/json", body.as_bytes())
}

fn write_http_response(
    stream: &mut TcpStream,
    code: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

/// Minimal HTTP/1.1 response from [`http_request`].
#[derive(Clone, Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn body_text(&self) -> Result<&str, String> {
        std::str::from_utf8(&self.body).map_err(|e| e.to_string())
    }
}

/// Zero-dependency blocking HTTP/1.1 client over real TCP sockets.
pub fn http_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
) -> Result<HttpResponse, String> {
    let mut stream = TcpStream::connect_timeout(&addr, SOCKET_IO_TIMEOUT)
        .map_err(|e| format!("connect {addr} failed: {e}"))?;
    let _ = stream.set_read_timeout(Some(SOCKET_IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(SOCKET_IO_TIMEOUT));

    let payload = body.unwrap_or(&[]);
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("write headers failed: {e}"))?;
    if !payload.is_empty() {
        // If server immediately rejects an oversized request (e.g. HTTP 413),
        // the write may end early with BrokenPipe/ConnectionReset; still read the response.
        if let Err(e) = stream.write_all(payload) {
            if !matches!(
                e.kind(),
                std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
            ) {
                return Err(format!("write body failed: {e}"));
            }
        }
    }
    let _ = stream.flush();
    let _ = stream.shutdown(std::net::Shutdown::Write);

    let mut raw = Vec::new();
    if let Err(e) = stream.read_to_end(&mut raw) {
        if raw.is_empty() {
            return Err(format!("read response failed: {e}"));
        }
    }

    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response (missing CRLFCRLF)".to_owned())?;
    let header_str = std::str::from_utf8(&raw[..split])
        .map_err(|_| "non-UTF8 HTTP response headers".to_owned())?;
    let status_line = header_str
        .lines()
        .next()
        .ok_or_else(|| "empty HTTP response status line".to_owned())?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| format!("invalid HTTP status line `{status_line}`"))?;

    Ok(HttpResponse {
        status,
        body: raw[split + 4..].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advertise::{CapabilityAnnouncement, GpuDevice, HardwareBackend, ModelCapability};
    use crate::daemon::{DeterministicEngineBackend, JobExecutionResult, ProviderConfig};
    use crate::identity::{ProviderKeyPair, TenantKeyPair};
    use crate::isolation::SandboxExecutionPolicy;
    use crate::ledger::verify_jsonl_chain;

    #[test]
    fn real_loopback_http_lifecycle_quote_job_verify_export_drain() {
        let kp = ProviderKeyPair::from_label("rpc-test-provider");
        let cfg = ProviderConfig {
            endpoint: "http://127.0.0.1:0".into(),
            devices: vec![GpuDevice {
                backend: HardwareBackend::Cuda,
                name: "RTX 4090".into(),
                vram_total_mib: 24_576,
                vram_usable_mib: 22_528,
                compute_units: 128,
                device_id: "cuda:0".into(),
            }],
            models: vec![ModelCapability {
                model_id: "zen-8b-q4".into(),
                quantization: "Q4_K_M".into(),
                vram_required_mib: 6_144,
                max_context_tokens: 8_192,
                price_micro_usd_per_1m: 150_000,
            }],
            max_concurrency: 2,
            heartbeat_ttl_secs: 60,
            quote_ttl_secs: 30,
            sandbox_policy: SandboxExecutionPolicy::default(),
        };
        let daemon = Arc::new(ProviderDaemon::new(
            kp,
            cfg,
            Arc::new(DeterministicEngineBackend {
                per_token_delay_us: 0,
                prefill_delay_us: 0,
            }),
        ));
        let server = RpcServer::start("127.0.0.1:0", Arc::clone(&daemon)).unwrap();
        let addr = server.addr();

        // 1. Health check
        let health = http_request(addr, "GET", "/health", None).unwrap();
        assert_eq!(health.status, 200);

        // 2. Signed manifest
        let mf = http_request(addr, "GET", "/v1/provider/manifest", None).unwrap();
        assert_eq!(mf.status, 200);
        let ann = CapabilityAnnouncement::from_json(&parse_json(mf.body_text().unwrap()).unwrap())
            .unwrap();
        ann.verify(now_unix_ms()).unwrap();

        // 3. Request signed quote
        let tenant = TenantKeyPair::from_label("rpc-tenant-1");
        let qreq = QuoteRequest {
            tenant_id: tenant.tenant_id().into(),
            model_id: "zen-8b-q4".into(),
            prompt_tokens_estimate: 12,
            max_tokens: 16,
        };
        let q_resp = http_request(
            addr,
            "POST",
            "/v1/provider/quote",
            Some(qreq.to_json().to_compact_string().as_bytes()),
        )
        .unwrap();
        assert_eq!(q_resp.status, 200);
        let quote =
            SignedQuote::from_json(&parse_json(q_resp.body_text().unwrap()).unwrap()).unwrap();
        quote.verify(now_unix_ms()).unwrap();

        // 4. Execute signed job lease
        let lease = SignedJobLease::sign_for_quote(
            &tenant,
            &quote,
            "rpc-job-1",
            77_001,
            "Run permissionless inference over loopback TCP",
            now_unix_ms(),
        );
        let mut job_body = BTreeMap::new();
        job_body.insert("quote".into(), quote.to_json());
        job_body.insert("lease".into(), lease.to_json());
        let j_resp = http_request(
            addr,
            "POST",
            "/v1/provider/jobs",
            Some(JsonValue::Object(job_body).to_compact_string().as_bytes()),
        )
        .unwrap();
        assert_eq!(j_resp.status, 200);
        let res = JobExecutionResult::from_json(&parse_json(j_resp.body_text().unwrap()).unwrap())
            .unwrap();
        res.receipt.verify().unwrap();

        // 5. Verify receipt via RPC endpoint
        let v_resp = http_request(
            addr,
            "POST",
            "/v1/provider/verify-receipt",
            Some(res.receipt.to_json().to_compact_string().as_bytes()),
        )
        .unwrap();
        assert_eq!(v_resp.status, 200);

        // 6. Export sovereign hash-chained JSONL ledger
        let exp = http_request(addr, "GET", "/v1/provider/ledger/export", None).unwrap();
        assert_eq!(exp.status, 200);
        let (entries, summary) =
            verify_jsonl_chain(exp.body_text().unwrap(), Some(daemon.provider_id())).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(summary.total_completion_tokens, 16);

        // 7. Oversized body rejected with 413
        let big = vec![b'a'; MAX_HTTP_BODY_BYTES + 16];
        let big_resp = http_request(addr, "POST", "/v1/provider/jobs", Some(&big)).unwrap();
        assert_eq!(big_resp.status, 413);

        // 8. Drain rejects new quotes with 503
        let dr = http_request(addr, "POST", "/v1/provider/drain", None).unwrap();
        assert_eq!(dr.status, 200);
        let q_after_drain = http_request(
            addr,
            "POST",
            "/v1/provider/quote",
            Some(qreq.to_json().to_compact_string().as_bytes()),
        )
        .unwrap();
        assert_eq!(q_after_drain.status, 503);

        server.stop();
    }
}
