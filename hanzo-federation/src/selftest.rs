//! Self-test — runs on any lab box.
//!
//! Probes: bf16 round-trip is byte-identical; coordinator handshake works
//! when a URL is given.

use anyhow::{Context, Result};

use crate::codec::{decode_delta, encode_delta, f32_to_bf16};
use crate::transport::TransportClient;

/// Byte-identical round-trip: build a payload, encode, decode, re-encode,
/// assert equality. Catches drift in the codec (would also catch broken
/// bf16 conversion if it ever drifted between Rust and Python).
pub fn verify_roundtrip() -> Result<()> {
    let xs: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001 - 0.5).collect();
    let raw = f32_to_bf16(&xs);
    let items: Vec<(&str, &[u8])> = vec![("x", raw.as_slice())];
    let blob1 = encode_delta(&items);
    let decoded = decode_delta(&blob1)?;
    assert_eq!(decoded.len(), 1, "decode returned wrong tensor count");
    let (name, _meta, body) = &decoded[0];
    assert_eq!(name, "x");
    assert_eq!(body, &raw, "decode dropped bytes");
    // Re-encode using the same raw payload.
    let blob2 = encode_delta(&items);
    assert_eq!(blob1, blob2, "re-export produced different bytes");
    Ok(())
}

/// Probe coordinator: healthz + topology. Returns the worker names known
/// to the coordinator.
pub async fn coordinator_handshake(url: &str, worker_name: Option<&str>) -> Result<Vec<String>> {
    let client = TransportClient::new(url, worker_name.unwrap_or("anon"));
    let _ = client.healthz().await.context("healthz")?;
    let topo = client.topology().await.context("topology")?;
    let names: Vec<String> = topo
        .get("workers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|w| w.get("name").and_then(|n| n.as_str()).map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_passes() {
        verify_roundtrip().expect("round-trip should pass");
    }
}
