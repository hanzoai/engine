//! ## General utilities.

use hanzo_engine::AudioInput;
use hanzo_engine::Hanzo;
use image::DynamicImage;
use std::error::Error;
use std::sync::Arc;
use tokio::{
    fs::{self, File},
    io::AsyncReadExt,
};

/// Largest media body accepted from a remote URL. A request body is attacker-chosen; without a cap a
/// single `image_url` can stream until the process is OOM-killed.
const MAX_REMOTE_MEDIA_BYTES: u64 = 64 * 1024 * 1024;

/// Fetch `url` over http(s) for media, refusing what a server must not fetch on a caller's behalf.
///
/// `parse_image_url`/`parse_audio_url` run on `/v1/chat/completions` and `/v1/embeddings`, so the URL
/// arrives from an untrusted request body. A bare GET there is an SSRF primitive: the server will
/// happily fetch `http://169.254.169.254/...` (cloud metadata, i.e. credentials), reach hosts inside
/// the deployment's network that the caller cannot, follow a redirect from a public host into either,
/// and stream an unbounded body. This refuses all four:
///   - only http/https (the caller's scheme choice is not ours to trust),
///   - no redirects, so a public URL cannot hop to a private one after the check,
///   - resolved address must be globally routable -- no loopback/private/link-local/unspecified,
///   - body capped at MAX_REMOTE_MEDIA_BYTES, enforced on the stream, not on a claimed header.
async fn fetch_remote_media(url: &url::Url) -> Result<Vec<u8>, anyhow::Error> {
    if url.scheme() != "http" && url.scheme() != "https" {
        anyhow::bail!("Unsupported URL scheme for remote media: {}", url.scheme());
    }
    // Match on the parsed host, not `host_str()`: that returns an IPv6 literal still wrapped in
    // brackets ("[::1]"), which no resolver accepts -- the address would fail DNS instead of reaching
    // the check below, refusing for the wrong reason and never testing the IP at all.
    let host = url
        .host()
        .ok_or_else(|| anyhow::anyhow!("Remote media URL has no host: {url}"))?;
    let ips: Vec<std::net::IpAddr> = match host {
        // A literal address needs no resolver; check what was actually named.
        url::Host::Ipv4(v4) => vec![std::net::IpAddr::V4(v4)],
        url::Host::Ipv6(v6) => vec![std::net::IpAddr::V6(v6)],
        // A name is only as safe as its worst address: check every one it maps to, since a name can
        // resolve to a private address (and to several).
        url::Host::Domain(name) => {
            let port = url.port_or_known_default().unwrap_or(80);
            tokio::net::lookup_host((name, port))
                .await
                .map_err(|e| anyhow::anyhow!("Could not resolve remote media host '{name}': {e}"))?
                .map(|a| a.ip())
                .collect()
        }
    };
    if ips.is_empty() {
        anyhow::bail!("Remote media host '{host}' resolved to no addresses");
    }
    for ip in ips {
        let blocked = match ip {
            std::net::IpAddr::V4(v4) => {
                v4.is_loopback()
                    || v4.is_private()
                    || v4.is_link_local()
                    || v4.is_broadcast()
                    || v4.is_unspecified()
                    || v4.is_documentation()
            }
            std::net::IpAddr::V6(v6) => {
                v6.is_loopback()
                    || v6.is_unspecified()
                    || v6.is_unique_local()
                    || v6.is_unicast_link_local()
            }
        };
        if blocked {
            anyhow::bail!(
                "Refusing to fetch remote media from non-public address {ip} (host '{host}')"
            );
        }
    }

    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let resp = client.get(url.clone()).send().await?.error_for_status()?;

    // Cap on the stream: Content-Length is the sender's claim, not a limit.
    use futures::StreamExt;
    let mut out: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if out.len() as u64 + chunk.len() as u64 > MAX_REMOTE_MEDIA_BYTES {
            anyhow::bail!("Remote media at {url} exceeds the {MAX_REMOTE_MEDIA_BYTES} byte limit");
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

/// Parses and loads an image from a URL, file path, or data URL.
///
/// This function accepts various input formats and attempts to parse them in order:
/// 1. First tries to parse as a complete URL (http/https/file/data schemes)
/// 2. If that fails, checks if it's a local file path and converts to file URL
/// 3. Finally falls back to treating it as a malformed URL and returns an error
///
/// ### Arguments
///
/// * `url_unparsed` - A string that can be:
///   - An HTTP/HTTPS URL (e.g., "<https://example.com/image.png>")
///   - A file path (e.g., "/path/to/image.jpg" or "image.png")
///   - A data URL with base64 encoded image (e.g., "data:image/png;base64,...")
///   - A file URL (e.g., "file:///path/to/image.jpg")
///
/// ### Examples
///
/// ```ignore
/// use hanzo_server_core::util::parse_image_url;
///
/// // Load from HTTP URL
/// let image = parse_image_url("https://example.com/photo.jpg").await?;
///
/// // Load from local file path
/// let image = parse_image_url("./assets/logo.png").await?;
///
/// // Load from data URL
/// let image = parse_image_url("data:image/png;base64,iVBORw0KGgoAAAANS...").await?;
///
/// // Load from file URL
/// let image = parse_image_url("file:///home/user/picture.jpg").await?;
/// ```
pub async fn parse_image_url(url_unparsed: &str) -> Result<DynamicImage, anyhow::Error> {
    let url = if let Ok(url) = url::Url::parse(url_unparsed) {
        url
    } else if File::open(url_unparsed).await.is_ok() {
        url::Url::from_file_path(std::path::absolute(url_unparsed)?)
            .map_err(|_| anyhow::anyhow!("Could not parse file path: {}", url_unparsed))?
    } else {
        anyhow::bail!(
            "Invalid source '{}': not a valid URL (http/https/data) and file not found on server. \
             Use a full URL, a data URL, or an absolute file path that exists on the server.",
            url_unparsed
        )
    };

    let bytes = if url.scheme() == "http" || url.scheme() == "https" {
        fetch_remote_media(&url).await?
    } else if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Could not parse file path: {}", url))?;

        if let Ok(mut f) = File::open(&path).await {
            // Read from local file
            let metadata = fs::metadata(&path).await?;
            let mut buffer = vec![0; metadata.len() as usize];
            f.read_exact(&mut buffer).await?;
            buffer
        } else {
            anyhow::bail!("Could not open file at path: {}", url);
        }
    } else if url.scheme() == "data" {
        // Decode with base64
        let data_url = data_url::DataUrl::process(url.as_str())?;
        data_url.decode_to_vec()?.0
    } else {
        anyhow::bail!("Unsupported URL scheme: {}", url.scheme());
    };

    Ok(image::load_from_memory(&bytes)?)
}

/// Parses and loads an audio file from a URL, file path, or data URL.
pub async fn parse_audio_url(url_unparsed: &str) -> Result<AudioInput, anyhow::Error> {
    let url = if let Ok(url) = url::Url::parse(url_unparsed) {
        url
    } else if File::open(url_unparsed).await.is_ok() {
        url::Url::from_file_path(std::path::absolute(url_unparsed)?)
            .map_err(|_| anyhow::anyhow!("Could not parse file path: {}", url_unparsed))?
    } else {
        anyhow::bail!(
            "Invalid source '{}': not a valid URL (http/https/data) and file not found on server. \
             Use a full URL, a data URL, or an absolute file path that exists on the server.",
            url_unparsed
        )
    };

    let bytes = if url.scheme() == "http" || url.scheme() == "https" {
        fetch_remote_media(&url).await?
    } else if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Could not parse file path: {}", url))?;

        if let Ok(mut f) = File::open(&path).await {
            let metadata = fs::metadata(&path).await?;
            let mut buffer = vec![0; metadata.len() as usize];
            f.read_exact(&mut buffer).await?;
            buffer
        } else {
            anyhow::bail!("Could not open file at path: {}", url);
        }
    } else if url.scheme() == "data" {
        let data_url = data_url::DataUrl::process(url.as_str())?;
        data_url.decode_to_vec()?.0
    } else {
        anyhow::bail!("Unsupported URL scheme: {}", url.scheme());
    };

    AudioInput::from_bytes(&bytes)
}

/// Validates that the requested model matches one of the loaded models.
///
/// This function checks if the model parameter from an OpenAI API request
/// matches one of the models that are currently loaded by the server.
///
/// The special model name "default" can be used to bypass this validation,
/// which is useful for clients that require a model parameter but want
/// to use the default model.
///
/// ### Arguments
///
/// * `requested_model` - The model name from the API request
/// * `state` - The Hanzo state containing the loaded models info
///
/// ### Returns
///
/// Returns `Ok(())` if the model is available or if "default" is specified, otherwise returns an error.
pub fn validate_model_name(requested_model: &str, state: Arc<Hanzo>) -> Result<(), anyhow::Error> {
    // Allow "default" as a special case to bypass validation
    if requested_model == "default" {
        return Ok(());
    }

    if state
        .model_exists(requested_model)
        .map_err(|e| anyhow::anyhow!("Failed to resolve model: {}", e))?
    {
        return Ok(());
    }

    let available_models = state
        .list_models()
        .map_err(|e| anyhow::anyhow!("Failed to get available models: {}", e))?;

    if available_models.is_empty() {
        anyhow::bail!("No models are currently loaded.");
    }

    anyhow::bail!(
        "Requested model '{}' is not available. Available models: {}. Use 'default' to use the default model.",
        requested_model,
        available_models.join(", ")
    )
}

/// Sanitize error messages to remove internal implementation details like stack traces.
/// This ensures that sensitive internal information is not exposed to API clients.
///
/// The function traverses the error chain to find the deepest (root) error and returns its message.
/// This is useful for API responses where we want to provide meaningful error information
/// without exposing internal stack traces or implementation details.
///
/// ### Arguments
///
/// * `error` - The error to sanitize
///
/// ### Returns
///
/// The message from the root cause error in the error chain
///
/// ### Examples
///
/// ```ignore
/// use hanzo_server_core::util::sanitize_error_message;
///
/// // For a simple error without chain
/// let error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
/// assert_eq!(sanitize_error_message(&error), "File not found");
///
/// // For chained errors, returns the root cause
/// let root = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Access denied");
/// let wrapped = anyhow::Error::new(root).context("Failed to read file");
/// // This would return "Access denied" instead of "Failed to read file"
/// ```
pub fn sanitize_error_message(error: &(dyn Error + 'static)) -> String {
    // Traverse the error chain to find the deepest (root) error and return its message.
    let mut current: &dyn Error = error;

    // Keep traversing until we find an error with no source
    while let Some(source) = current.source() {
        current = source;
    }

    // Return the message of the root cause error
    current.to_string()
}

#[cfg(test)]
mod tests {
    use image::GenericImageView;

    use super::*;

    #[tokio::test]
    async fn test_parse_image_url() {
        // from URL
        let url = "https://www.rust-lang.org/logos/rust-logo-32x32.png";
        let image = parse_image_url(url).await.unwrap();
        assert_eq!(image.dimensions(), (32, 32));

        let url = "http://www.rust-lang.org/logos/rust-logo-32x32.png";
        let image = parse_image_url(url).await.unwrap();
        assert_eq!(image.dimensions(), (32, 32));

        // from file path
        let url = "resources/rust-logo-32x32.png";
        let image = parse_image_url(url).await.unwrap();
        assert_eq!(image.dimensions(), (32, 32));

        // URL must be an absolute path
        let absolute_path = std::path::absolute(url).unwrap();
        let url = format!("file://{}", absolute_path.as_os_str().to_str().unwrap());
        let image = parse_image_url(&url).await.unwrap();
        assert_eq!(image.dimensions(), (32, 32));

        // from base64 encoded image (rust-logo-32x32.png)
        let url = "
        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
        ";

        let url = format!("data:image/png;base64,{url}");
        let image = parse_image_url(&url).await.unwrap();
        assert_eq!(image.dimensions(), (32, 32));

        // audio from base64
        let audio_b64 = "UklGRiYAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQIAAAAAAA==";
        let url = format!("data:audio/wav;base64,{audio_b64}");
        let audio = parse_audio_url(&url).await.unwrap();
        assert_eq!(audio.sample_rate, 8000);
        assert_eq!(audio.samples.len(), 1);
    }

    #[test]
    fn test_sanitize_error_message_with_backtrace() {
        // Test error with backtrace
        let error_with_backtrace = "Failed to parse Forge Provider response: A weight is negative, too large or not a valid number
  0: hanzo_ml::error::Error::bt
  1: hanzo_engine::sampler::Sampler::sample_multinomial
  2: hanzo_engine::sampler::Sampler::sample_top_kp_min_p
  3: hanzo_engine::sampler::Sampler::sample
  4: hanzo_engine::pipeline::sampling::sample_sequence::{{closure}}";

        struct TestError(String);
        impl std::fmt::Display for TestError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
        impl std::fmt::Debug for TestError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
        impl std::error::Error for TestError {}

        let error = TestError(error_with_backtrace.to_string());
        let sanitized = sanitize_error_message(&error);

        // Since TestError has no source(), it should return the full message including backtrace
        assert_eq!(sanitized, error_with_backtrace);
        // The improved solution returns the root error as-is when there's no error chain
    }

    #[test]
    fn test_sanitize_error_message_without_backtrace() {
        // Test error without backtrace
        let simple_error = "Simple error message without backtrace";

        struct TestError(String);
        impl std::fmt::Display for TestError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
        impl std::fmt::Debug for TestError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
        impl std::error::Error for TestError {}

        let error = TestError(simple_error.to_string());
        let sanitized = sanitize_error_message(&error);

        assert_eq!(sanitized, simple_error);
    }

    #[test]
    fn test_sanitize_error_message_with_chain() {
        // Test error chain - the root cause should be extracted
        use std::fmt;

        #[derive(Debug)]
        struct RootError;
        impl fmt::Display for RootError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "Root cause: Database connection failed")
            }
        }
        impl std::error::Error for RootError {}

        #[derive(Debug)]
        struct MiddleError(Box<dyn std::error::Error>);
        impl fmt::Display for MiddleError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "Middle error: Service unavailable")
            }
        }
        impl std::error::Error for MiddleError {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&*self.0)
            }
        }

        #[derive(Debug)]
        struct TopError(Box<dyn std::error::Error>);
        impl fmt::Display for TopError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    "Top error: Request failed with backtrace\n  0: some::module::function"
                )
            }
        }
        impl std::error::Error for TopError {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&*self.0)
            }
        }

        let root = RootError;
        let middle = MiddleError(Box::new(root));
        let top = TopError(Box::new(middle));

        let sanitized = sanitize_error_message(&top);

        // Should return the root cause, not the top-level error with backtrace
        assert_eq!(sanitized, "Root cause: Database connection failed");
        assert!(!sanitized.contains("backtrace"));
        assert!(!sanitized.contains("Request failed"));
    }

    /// A URL naming a non-public address must be refused before any request is made.
    ///
    /// `parse_image_url`/`parse_audio_url` are reachable from the request body on
    /// `/v1/chat/completions` and `/v1/embeddings`, so an unguarded GET here lets a caller aim the
    /// server at `169.254.169.254` (cloud metadata: credentials) or at hosts inside the deployment
    /// that the caller cannot reach. Each case below is refused at resolution, so nothing is sent.
    #[tokio::test]
    async fn remote_media_refuses_non_public_addresses() {
        for url in [
            "http://169.254.169.254/latest/meta-data/", // cloud metadata
            "http://127.0.0.1:8080/admin",              // loopback
            "http://10.0.0.5/internal",                 // private
            "http://[::1]/",                            // v6 loopback
        ] {
            let parsed = url::Url::parse(url).unwrap();
            let err = fetch_remote_media(&parsed)
                .await
                .expect_err("must refuse a non-public address");
            let msg = err.to_string();
            assert!(
                msg.contains("non-public address") || msg.contains("resolved to no addresses"),
                "{url} was refused, but for the wrong reason: {msg}"
            );
        }
    }

    /// Only http/https reach the network path. `file:` is served by the local branch in
    /// `parse_image_url`, never by a fetch, so a caller cannot smuggle a local read through it.
    #[tokio::test]
    async fn remote_media_refuses_non_http_schemes() {
        for url in [
            "file:///etc/passwd",
            "ftp://example.com/x",
            "gopher://example.com/",
        ] {
            let parsed = url::Url::parse(url).unwrap();
            let err = fetch_remote_media(&parsed)
                .await
                .expect_err("must refuse scheme");
            assert!(
                err.to_string().contains("Unsupported URL scheme"),
                "{url}: {err}"
            );
        }
    }
}
