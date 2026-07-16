//! Loading media bytes from a source string: a local path, or a `file:`, `data:`,
//! or `http:`/`https:` URL.
//!
//! [`Origin`] selects which source kinds a caller reaches. Every load yields at
//! most [`MAX_BYTES`]; every fetch is bounded by the timeouts and the redirect
//! limit below.

use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    path::Path,
    time::Duration,
};

use anyhow::{Context, Result};
use reqwest::{
    header::{CONTENT_TYPE, LOCATION},
    redirect::Policy,
};
use tokio::{fs::File, io::AsyncReadExt};
use url::Url;

/// Bytes any one source yields.
pub const MAX_BYTES: usize = 64 * 1024 * 1024;

/// Budget for a fetch, from request to last byte.
const TIMEOUT: Duration = Duration::from_secs(30);
/// Budget for establishing a connection.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Budget between successive reads of a response body.
const READ_TIMEOUT: Duration = Duration::from_secs(15);
/// Redirects a fetch follows.
const MAX_REDIRECTS: usize = 3;
/// Room for the `data:<mime>;base64,` prefix when sizing an encoded payload.
const DATA_URL_HEADER: usize = 4096;

/// Who supplied the source string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// The operator of this process: a CLI argument, or an in-process API call.
    /// Reads local paths and `file:` URLs, and fetches any host the operator names.
    Operator,
    /// A peer, over the network: a field of an API request. Serves `data:` URLs,
    /// and fetches http(s) hosts that resolve to globally routable addresses.
    Network,
}

/// Bytes from a media source, with the MIME type the source declared.
pub struct Loaded {
    pub bytes: Vec<u8>,
    pub mime: Option<String>,
}

impl std::fmt::Debug for Loaded {
    /// Reports the size of the body rather than the body.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Loaded")
            .field("bytes", &self.bytes.len())
            .field("mime", &self.mime)
            .finish()
    }
}

/// Load `source` under `origin`, yielding at most [`MAX_BYTES`].
pub async fn load(source: &str, origin: Origin) -> Result<Loaded> {
    load_within(source, origin, is_global, MAX_BYTES).await
}

/// [`load`], with the reachable addresses and the byte bound named explicitly.
async fn load_within(
    source: &str,
    origin: Origin,
    allow: fn(IpAddr) -> bool,
    max: usize,
) -> Result<Loaded> {
    let Ok(url) = Url::parse(source) else {
        // A string that is not a URL names a path on this machine.
        return match origin {
            Origin::Operator => Ok(Loaded {
                bytes: read_file(Path::new(source), max).await?,
                mime: None,
            }),
            Origin::Network => {
                anyhow::bail!("Source must be an http, https, or data URL: {source}")
            }
        };
    };

    match (url.scheme(), origin) {
        ("http" | "https", _) => fetch(url, origin, allow, max).await,
        ("data", _) => decode_data_url(url.as_str(), max),
        ("file", Origin::Operator) => {
            let path = url
                .to_file_path()
                .map_err(|()| anyhow::anyhow!("Not a readable file URL: {url}"))?;
            Ok(Loaded {
                bytes: read_file(&path, max).await?,
                mime: None,
            })
        }
        ("file", Origin::Network) => {
            anyhow::bail!("Source must be an http, https, or data URL: {url}")
        }
        (scheme, _) => anyhow::bail!("Unsupported URL scheme: {scheme}"),
    }
}

/// Fetch `url`, following redirects one hop at a time so that each hop's target
/// is resolved and checked before it is requested.
async fn fetch(mut url: Url, origin: Origin, allow: fn(IpAddr) -> bool, max: usize) -> Result<Loaded> {
    let mut hops = 0usize;
    loop {
        let mut builder = reqwest::Client::builder()
            .timeout(TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
            .redirect(Policy::none())
            // Connect directly, so that the addresses checked below are the
            // addresses the request reaches.
            .no_proxy();

        if origin == Origin::Network {
            let addrs = resolve(&url, allow).await?;
            let host = url.host_str().expect("a resolved URL has a host");
            // Reuse the checked answer for the connection rather than resolving
            // `host` a second time.
            builder = builder.resolve_to_addrs(host, &addrs);
        }

        let response = builder
            .build()?
            .get(url.clone())
            .send()
            .await
            .with_context(|| format!("Fetching {url}"))?;

        if response.status().is_redirection() {
            hops += 1;
            if hops > MAX_REDIRECTS {
                anyhow::bail!("{url} redirected more than {MAX_REDIRECTS} times.");
            }
            url = redirect_target(&url, &response)?;
            continue;
        }

        response
            .error_for_status_ref()
            .with_context(|| format!("Fetching {url}"))?;

        if response.content_length().is_some_and(|n| n > max as u64) {
            anyhow::bail!("{url} declares more than the {max} byte limit.");
        }

        let mime = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.split(';').next().unwrap_or(value).trim().to_string())
            .filter(|value| !value.is_empty());

        return Ok(Loaded {
            bytes: read_capped(response, max, &url).await?,
            mime,
        });
    }
}

/// Resolve the `Location` of a redirect against the URL that produced it.
fn redirect_target(current: &Url, response: &reqwest::Response) -> Result<Url> {
    let location = response
        .headers()
        .get(LOCATION)
        .ok_or_else(|| anyhow::anyhow!("{current} redirected without a Location."))?
        .to_str()
        .with_context(|| format!("{current} redirected to a non-UTF-8 Location."))?;
    let target = current
        .join(location)
        .with_context(|| format!("{current} redirected to an unparseable Location."))?;
    match target.scheme() {
        "http" | "https" => Ok(target),
        scheme => anyhow::bail!("{current} redirected to an unsupported scheme: {scheme}"),
    }
}

/// Read a response body, counting the bytes that arrive rather than the length
/// the response declares.
async fn read_capped(mut response: reqwest::Response, max: usize, url: &Url) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if bytes.len().saturating_add(chunk.len()) > max {
            anyhow::bail!("{url} sent more than the {max} byte limit.");
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// Read a file, taking at most `max` bytes off the handle. The size the metadata
/// reports selects the error, never the size of the allocation.
async fn read_file(path: &Path, max: usize) -> Result<Vec<u8>> {
    let metadata = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("Reading {}", path.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("Not a file: {}", path.display());
    }
    if metadata.len() > max as u64 {
        anyhow::bail!("{} is larger than the {max} byte limit.", path.display());
    }

    let file = File::open(path)
        .await
        .with_context(|| format!("Opening {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(max as u64 + 1)
        .read_to_end(&mut bytes)
        .await
        .with_context(|| format!("Reading {}", path.display()))?;
    if bytes.len() > max {
        anyhow::bail!("{} is larger than the {max} byte limit.", path.display());
    }
    Ok(bytes)
}

/// Decode a `data:` URL, sizing the encoded payload before decoding it.
fn decode_data_url(source: &str, max: usize) -> Result<Loaded> {
    let encoded = max
        .saturating_mul(4)
        .saturating_div(3)
        .saturating_add(DATA_URL_HEADER);
    if source.len() > encoded {
        anyhow::bail!("Data URL is larger than the {max} byte limit.");
    }
    let url = data_url::DataUrl::process(source)?;
    let mime = url.mime_type();
    let mime = format!("{}/{}", mime.type_, mime.subtype);
    let bytes = url.decode_to_vec()?.0;
    if bytes.len() > max {
        anyhow::bail!("Data URL is larger than the {max} byte limit.");
    }
    Ok(Loaded {
        bytes,
        mime: Some(mime),
    })
}

/// Resolve `url`'s host to the addresses a fetch may connect to, keeping the
/// answer only when every address it holds is one `allow` accepts.
async fn resolve(url: &Url, allow: fn(IpAddr) -> bool) -> Result<Vec<SocketAddr>> {
    let host = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("URL has no host: {url}"))?;
    reject_local_name(host, allow)?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| anyhow::anyhow!("URL has no port: {url}"))?;

    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .with_context(|| format!("Resolving {host}"))?
        .collect();
    if addrs.is_empty() {
        anyhow::bail!("{host} resolved to no addresses.");
    }
    for addr in &addrs {
        reject_private_ip(addr.ip(), allow)?;
    }
    Ok(addrs)
}

/// Accept a host name that names neither this machine nor a link-local peer.
fn reject_local_name(host: &str, allow: fn(IpAddr) -> bool) -> Result<()> {
    let name = host.trim_end_matches('.').to_ascii_lowercase();
    if name == "localhost" || name.ends_with(".localhost") || name.ends_with(".local") {
        anyhow::bail!("{host} names a local host.");
    }
    if let Ok(ip) = name.parse::<IpAddr>() {
        reject_private_ip(ip, allow)?;
    }
    Ok(())
}

/// Accept an address `allow` accepts.
fn reject_private_ip(ip: IpAddr, allow: fn(IpAddr) -> bool) -> Result<()> {
    if allow(ip) {
        Ok(())
    } else {
        anyhow::bail!("{ip} is not a globally routable address.");
    }
}

/// Whether `ip` is globally routable: assigned to the public internet rather
/// than to a private network, this machine, or a reserved range.
fn is_global(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => is_global_v4(ip),
        IpAddr::V6(ip) => match ip.to_ipv4_mapped() {
            Some(ip) => is_global_v4(ip),
            None => is_global_v6(ip),
        },
    }
}

fn is_global_v4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_unspecified()
        || ip.is_multicast()
        // Carrier-grade NAT, RFC 6598.
        || matches!(octets, [100, 64..=127, _, _])
        // IETF protocol assignments, RFC 6890.
        || matches!(octets, [192, 0, 0, _])
        // TEST-NET-1, benchmarking, TEST-NET-2, TEST-NET-3.
        || matches!(octets, [192, 0, 2, _])
        || matches!(octets, [198, 18 | 19, _, _])
        || matches!(octets, [198, 51, 100, _])
        || matches!(octets, [203, 0, 113, _])
        // Reserved, RFC 1112.
        || octets[0] >= 240)
}

fn is_global_v6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    !(ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_unique_local()
        || ip.is_unicast_link_local()
        || ip.is_multicast()
        // Documentation, RFC 3849.
        || (segments[0] == 0x2001 && segments[1] == 0x0db8))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tokio::io::AsyncWriteExt;

    /// Accepts the loopback test servers below in addition to the public
    /// internet, so that a hop reaching one is decided by the hop itself.
    fn allow_loopback(ip: IpAddr) -> bool {
        ip.is_loopback() || is_global(ip)
    }

    /// Serve `response` verbatim to every connection, on a loopback port.
    async fn serve(response: &'static [u8]) -> SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 2048];
                    let _ = sock.read(&mut buf).await;
                    let _ = sock.write_all(response).await;
                    let _ = sock.flush().await;
                });
            }
        });
        addr
    }

    #[tokio::test]
    async fn resolve_rejects_private_and_metadata_addresses() {
        for source in [
            // Cloud instance metadata, v4 and v6.
            "http://169.254.169.254/latest/meta-data",
            "http://[fd00:ec2::254]/latest/meta-data",
            // This machine.
            "http://127.0.0.1/x.png",
            "http://127.99.4.3/x.png",
            "http://[::1]/x.png",
            "http://localhost/x.png",
            "http://LOCALHOST./x.png",
            "http://printer.local/x.png",
            "http://0.0.0.0/x.png",
            "http://[::]/x.png",
            // An IPv4 loopback address spelled as IPv6.
            "http://[::ffff:127.0.0.1]/x.png",
            // RFC 1918.
            "http://10.0.0.1/x.png",
            "http://172.16.0.1/x.png",
            "http://172.31.255.254/x.png",
            "http://192.168.1.1/x.png",
            // Unique local and link local.
            "http://[fc00::1]/x.png",
            "http://[fe80::1]/x.png",
            // Carrier-grade NAT and reserved space.
            "http://100.64.0.1/x.png",
            "http://240.0.0.1/x.png",
        ] {
            let url = Url::parse(source).unwrap();
            assert!(
                resolve(&url, is_global).await.is_err(),
                "reached {source}, which is not globally routable"
            );
        }
    }

    #[test]
    fn global_addresses_stay_reachable() {
        for ip in ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"] {
            assert!(is_global(ip.parse().unwrap()), "{ip} is globally routable");
        }
    }

    #[tokio::test]
    async fn network_origin_refuses_local_sources() {
        let path = std::path::absolute("resources/rust-logo-32x32.png").unwrap();
        for source in [
            format!("file://{}", path.display()),
            "resources/rust-logo-32x32.png".to_string(),
            "/etc/passwd".to_string(),
            "file:///etc/passwd".to_string(),
        ] {
            assert!(
                load(&source, Origin::Network).await.is_err(),
                "read {source} for a network request"
            );
        }
    }

    #[tokio::test]
    async fn operator_origin_reads_local_sources() {
        let path = std::path::absolute("resources/rust-logo-32x32.png").unwrap();
        for source in [
            format!("file://{}", path.display()),
            "resources/rust-logo-32x32.png".to_string(),
        ] {
            let media = load(&source, Origin::Operator).await.unwrap();
            assert!(media.bytes.starts_with(b"\x89PNG"), "{source}");
        }
    }

    #[tokio::test]
    async fn file_over_the_limit_is_refused() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![0u8; 4096]).unwrap();
        file.flush().unwrap();

        assert!(read_file(file.path(), 1024).await.is_err());
        assert_eq!(read_file(file.path(), 8192).await.unwrap().len(), 4096);
    }

    #[tokio::test]
    async fn directories_and_devices_are_not_files() {
        assert!(read_file(Path::new("resources"), MAX_BYTES).await.is_err());
        assert!(read_file(Path::new("/dev/zero"), MAX_BYTES).await.is_err());
    }

    #[test]
    fn oversize_data_url_is_refused_before_decoding() {
        let source = format!("data:text/plain;base64,{}", "a".repeat(8192));
        assert!(decode_data_url(&source, 4).is_err());
    }

    #[test]
    fn data_url_over_the_limit_is_refused_after_decoding() {
        // 200 base64 characters clear the encoded bound of 100*4/3+4096, and
        // decode to 150 bytes, over the 100 byte limit.
        let source = format!("data:text/plain;base64,{}", "a".repeat(200));
        let err = decode_data_url(&source, 100).unwrap_err().to_string();
        assert!(err.contains("100 byte limit"), "{err}");
    }

    #[tokio::test]
    async fn fetch_caps_a_body_sent_without_a_declared_length() {
        // No Content-Length and no Transfer-Encoding: the body runs until the
        // connection closes, so its size is known only as it arrives.
        static RESPONSE: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\n\r\n";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 2048];
                    let _ = sock.read(&mut buf).await;
                    let _ = sock.write_all(RESPONSE).await;
                    // Far more than the caller's limit, in chunks, as a server
                    // that means to exhaust this process would send it.
                    for _ in 0..64 {
                        if sock.write_all(&[b'x'; 4096]).await.is_err() {
                            return;
                        }
                    }
                });
            }
        });

        let url = Url::parse(&format!("http://{addr}/x.png")).unwrap();
        let err = fetch(url, Origin::Network, allow_loopback, 512)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("512 byte limit"), "{err}");
    }

    #[tokio::test]
    async fn fetch_reads_no_more_than_a_declared_length() {
        // A body longer than Content-Length is framed to the length declared,
        // so the bytes past it never reach the caller.
        static RESPONSE: &[u8] =
            b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 4\r\n\r\n";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 2048];
                    let _ = sock.read(&mut buf).await;
                    let _ = sock.write_all(RESPONSE).await;
                    let _ = sock.write_all(&[b'x'; 4096]).await;
                    let _ = sock.flush().await;
                });
            }
        });

        let url = Url::parse(&format!("http://{addr}/x.png")).unwrap();
        let media = fetch(url, Origin::Network, allow_loopback, 512).await.unwrap();
        assert_eq!(media.bytes, b"xxxx");
    }

    #[tokio::test]
    async fn fetch_refuses_a_declared_length_over_the_limit() {
        static RESPONSE: &[u8] =
            b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 4096\r\n\r\n";
        let addr = serve(RESPONSE).await;

        let url = Url::parse(&format!("http://{addr}/x.png")).unwrap();
        let err = fetch(url, Origin::Network, allow_loopback, 512)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("512 byte limit"), "{err}");
    }

    #[tokio::test]
    async fn fetch_rejects_a_redirect_to_the_metadata_address() {
        static RESPONSE: &[u8] = b"HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest/meta-data\r\nContent-Length: 0\r\n\r\n";
        let addr = serve(RESPONSE).await;

        let url = Url::parse(&format!("http://{addr}/x.png")).unwrap();
        let err = fetch(url, Origin::Network, allow_loopback, MAX_BYTES)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("169.254.169.254"), "{err}");
        assert!(err.contains("not a globally routable address"), "{err}");
    }

    #[tokio::test]
    async fn fetch_rejects_a_redirect_to_a_private_host() {
        static RESPONSE: &[u8] =
            b"HTTP/1.1 301 Moved Permanently\r\nLocation: http://10.1.2.3:8080/admin\r\nContent-Length: 0\r\n\r\n";
        let addr = serve(RESPONSE).await;

        let url = Url::parse(&format!("http://{addr}/x.png")).unwrap();
        let err = fetch(url, Origin::Network, allow_loopback, MAX_BYTES)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("10.1.2.3"), "{err}");
    }

    #[tokio::test]
    async fn fetch_rejects_a_redirect_off_http() {
        static RESPONSE: &[u8] =
            b"HTTP/1.1 302 Found\r\nLocation: file:///etc/passwd\r\nContent-Length: 0\r\n\r\n";
        let addr = serve(RESPONSE).await;

        let url = Url::parse(&format!("http://{addr}/x.png")).unwrap();
        let err = fetch(url, Origin::Network, allow_loopback, MAX_BYTES)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported scheme: file"), "{err}");
    }

    #[tokio::test]
    async fn fetch_stops_after_the_redirect_limit() {
        // Redirects to itself, forever.
        static RESPONSE: &[u8] =
            b"HTTP/1.1 302 Found\r\nLocation: /again\r\nContent-Length: 0\r\n\r\n";
        let addr = serve(RESPONSE).await;

        let url = Url::parse(&format!("http://{addr}/x.png")).unwrap();
        let err = fetch(url, Origin::Network, allow_loopback, MAX_BYTES)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains(&format!("more than {MAX_REDIRECTS} times")), "{err}");
    }

    #[tokio::test]
    async fn fetch_reads_a_body_within_the_limit() {
        static RESPONSE: &[u8] =
            b"HTTP/1.1 200 OK\r\nContent-Type: image/gif; charset=binary\r\nContent-Length: 6\r\n\r\nGIF89a";
        let addr = serve(RESPONSE).await;

        let url = Url::parse(&format!("http://{addr}/x.gif")).unwrap();
        let media = fetch(url, Origin::Network, allow_loopback, MAX_BYTES)
            .await
            .unwrap();
        assert_eq!(media.bytes, b"GIF89a");
        assert_eq!(media.mime.as_deref(), Some("image/gif"));
    }

    #[tokio::test]
    async fn network_origin_fetches_are_checked() {
        // The same server the tests above reach with `allow_loopback` is out of
        // reach under the address policy `load` applies.
        static RESPONSE: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi";
        let addr = serve(RESPONSE).await;

        let source = format!("http://{addr}/x.png");
        let err = load(&source, Origin::Network).await.unwrap_err().to_string();
        assert!(err.contains("not a globally routable address"), "{err}");

        // The operator names their own hosts.
        let media = load(&source, Origin::Operator).await.unwrap();
        assert_eq!(media.bytes, b"hi");
    }

    #[tokio::test]
    async fn unsupported_schemes_are_refused() {
        for source in ["ftp://example.com/x.png", "gopher://example.com/x.png"] {
            for origin in [Origin::Operator, Origin::Network] {
                assert!(load(source, origin).await.is_err(), "{source}");
            }
        }
    }
}
